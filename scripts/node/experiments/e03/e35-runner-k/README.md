# E35: verify length decided in the model runner (promoted default)

Promoted on 2026-09-30 by owner decision. `cluster.env.example` selects these sources and
mounts `policy.flag` (content `hybrid`) read-only at `/tmp/glm53-e35-policy`, so the default
boots with the `hybrid` policy. `delta.env` is kept unchanged as the record of the measured
overlay: over the E31-MB recipe it leaves the flag in the container's `/tmp`, where the
measurement window wrote each arm's policy at runtime. The one-step return to E31-MB is
`scripts/node/reference/operational-20260930-e31-mb.env`. Without a restart, overwriting the
host's `policy.flag` on rank 0 in place (for example `printf ema > policy.flag`) returns the
verify length to the acceptance EMA within 0.5 s, until the next deploy restores `hybrid`.
Rank 0 still broadcasts its unchanged choice on every participating step under `ema`.

The adaptive-k policy picks 3 or 7 verified drafts per request from an EMA of past acceptance
that the asynchronous scheduler sees two steps late. The DFlash2 selector's confidence in each
draft position predicts acceptance, but the scheduler fixes a step's length before the draft it
verifies exists. The model runner prepares a step after the previous one has finished, so it
can use the previous draft's confidence without waiting, or wait for the current draft.

- `speculator.py` is the image's DFlash2 speculator plus an additions-only wrapper of
  `propose()`: tensor-parallel rank 0 computes the per-position confidence on the device and
  copies it without blocking into a pinned ring slot with an event, keeping each request's two
  newest drafts with a sequence number. The sequence advances on every real proposal, also
  under `ema`, so a draft recorded before a policy change never passes as the current or
  previous one.
- `model_runner.py` is the image's V2 model runner plus an additions-only wrapper of
  `execute_model()`. A step participates only when its SchedulerOutput holds one request with
  seven scheduled drafts, which is identical on every rank. Rank 0 decides the length from the
  policy flag and always broadcasts it over the tensor-parallel CPU group; every rank then
  verifies a copy of the SchedulerOutput trimmed to that many drafts. Trimmed drafts count as
  rejected for the scheduler, whose cache blocks were reserved for all seven.
- `adaptive_k_scheduler.py` is the E31-MB scheduler plus an additions-only rule that gives a
  lone request seven drafts under the `lag1` and `hybrid` policies.

| `/tmp/glm53-e35-policy` content (read on rank 0) | Behaviour |
| --- | --- |
| missing, `ema` or any other content | acceptance EMA, no trim |
| `lag1` | previous draft's confidence; no wait |
| `hybrid` (default) | as `lag1`, but waits for the current draft when the rule's margin is below 0.005 |

The rule compares calibrated expected tokens per millisecond at 3 and 7 drafts (58 and 81 ms
per step). The calibration and the hybrid margin come from a diagnostic window with k forced
to 7; the margin was chosen on in-sample prompts only. Logs: `E35_RUNNER_K_READY`,
`E35_CONF_RECORDER_READY` and `E35_SCHEDULER_READY`; every 200 participating steps rank 0
logs `E35_RUNNER_K` with decisions, waits, broadcast latency, missing data and errors.
`scripts/tests/test-e35-runner-k.py` pins the image prefixes, the additions-only patches, the
trim, the rule and the flag handling.
