# E03 replay views with an effective draft budget

The accepted default scheduler caps adaptive verification placeholders at
`min(adaptive_k, engine_maximum, effective_draft_budget)`. It preserves E03's
6,912-row mHC path, the replay views connector, FP8, DFlash2, graph configuration,
the dynamic budget table, the 15 GiB KV pool and the context limit. The frozen
baseline scheduler is unchanged. Consult the
[dated measurement report](../../../../../docs/benchmarks/experiments/2026-09-19-e03-replay-draft-budget.md)
for the original series, and the [current accepted reference](../../../../../docs/benchmarks/baselines/2026-09-19-e03.md)
for the final three suites and owner decision. The source manifest retains its frozen
preparation identity; it is not a current deployment-status record. Default IaC now
selects the measured module, while [live reproduction](../../../../../docs/historical_benchmarks/baselines/2026-09-19-e03/promotion.json)
remains separately pending.

R10's worker resolves `num_spec_tokens_to_schedule` from the `SchedulerOutput`
that produced its drafts. The original adaptive scheduler can replace that
output's three placeholders with five. The candidate uses the same output and
resolver as the worker, after the original adaptive decision, without deriving
another budget from the following batch or just the eligible decode requests.
With the existing table, the budget is five for one scheduled request and three
for two through six. A transition back to one request may still verify three
drafts produced by its preceding batch before using five again.

This mismatch also exists in the frozen baseline. It is an optimization hypothesis,
not a demonstrated cause of the parent candidate's C4 throughput difference.
The [parent report](../../../../../docs/benchmarks/experiments/2026-09-19-e03-replay-views.md)
records 82.994 tok/s across three full runs and 84.054 in a separate C4 check,
against the frozen baseline's 87.527. None of those measurements is replaced.

## Prepare while the current service stays online

Run from the repository root. The CPU tests need only Python's standard library:

```sh
python3 scripts/tests/test-adaptive-draft-budget.py
python3 scripts/tests/test-draft-budget-config.py
./scripts/check.sh
```

The integration test imports the exact pinned R10 `AsyncScheduler` and
`SchedulerOutput` fixtures, substituting storage for the heavyweight parent
scheduler. It exercises the actual subclass hook, the original mismatch, the
disabled flag, unchanged C1 policy decisions, batches 2–6, mixed prefill/decode,
finished requests, zero/default budgets, asynchronous 1→4→1 handouts and delayed
observations. It also runs the original policy tests against the candidate.
This verifies CPU wiring, not GPU graph execution or performance.

For an independently captured image source tree, the same test accepts
`--engine-source <private-directory-containing-vllm>`. It verifies the four source
hashes in [manifest.json](manifest.json) and the worker's budget consumption.
Source capture is read-only and requires the target to be in scope.

The defaults already select the accepted combination. To reconstruct the historical
candidate/parent comparison, create a private standalone parent overlay by concatenating
`scripts/node/reference/baseline-20260919.env`,
`scripts/node/experiments/e03/candidate.env` and
`scripts/node/experiments/e03/replay-views/delta.env`. Preserve the already measured
parent overlay when available. Copy it to a new candidate overlay and append this
directory's [delta.env](delta.env). Use ignored paths relative to the repository
and `umask 077`; do not replace the serving overlay. The new delta requires the
parent's connector and exactly one original scheduler mount.

Preview both overlays through the native launcher on ranks 0–3:

```sh
TP4_ENV=<private-parent-overlay> TP4_DRY_RUN=1 bash scripts/launcher/launch-glm53-tp4.sh 0
TP4_ENV=<private-candidate-overlay> TP4_DRY_RUN=1 bash scripts/launcher/launch-glm53-tp4.sh 0
# Repeat for ranks 1, 2 and 3; retain output privately.
```

Each candidate command must differ only in the scheduler mount source and the
addition of `-e VLLM_ADAPTIVE_K_RESPECT_DRAFT_BUDGET=1`. The import target and class
stay `adaptive_k_scheduler.AdaptiveKScheduler`. The flag defaults to zero in the
experimental module; zero retains the original adaptive sizing. The original
baseline module remains at its original source and hash for rollback.

The existing additive deploy selector includes this directory's Python module,
manifest and `SHA256SUMS`. Within the newly authorized window, verify its
`sha256sum -c SHA256SUMS` under `~/tp4/experiments/e03/draft-budget` on every rank
before startup, as well as the unchanged E03 manifest and operator replay payload.
Deployment of the new defaults selects this module for the next boot; it does not
prove activation in an already running process.

## Coordinated transition and activation

Prepare the candidate and parent return first, then obtain a **new coordinated
deployment/start-stop window**. A previous window does not cover another restart.
Stop with the currently serving parent overlay; deploy and start with the new
standalone overlay. Keep that overlay on subsequent lifecycle commands. Follow
the existing [operations procedure](../../../../../docs/operations.md), including
the fabric checks and both functional gates within 120 seconds of `/health` 200.

Verify the complete four-rank identity against the preserved recipe and candidate
source manifest. Do not alter frozen baseline pins to make `check-f0.py` pass an
experimental identity. The rank-0 engine log must include:

```text
AdaptiveKScheduler active (... mode=batch-uniform ... respect_draft_budget=1) engine_k=5 async=True
adaptive-k: draft-budget active=1 source=SchedulerOutput.resolve_num_spec_tokens_to_schedule engine_k=5
```

On the first actual limit, it logs `draft-budget first-cap budget=3` with aggregate
`steps`, `limited_steps`, `limited_requests` and `trimmed_tokens`. The same totals
appear at the existing policy log interval, without request identifiers. Counts
refer to placeholder handouts, not accepted tokens or measured compute savings.
Any policy fallback, missing signature or mismatched source invalidates activation;
stop measurements and report using the operational failure rules.

## Measurement and decision

Use the operator's independent Rigmark checkout directly. Start with **one C4-only
execution: three rounds of four 256-token code requests, 12 requests total**. Use
the native `--skip-decode --skip-prefill --concurrency 4 --concurrency-runs 3`
selector already recorded in the parent's
[C4 check](../../../../../docs/historical_benchmarks/experiments/2026-09-19-e03-replay-views/c4-recheck.json).
Retain its distinct source hash; it is separate from the frozen full-suite source.

If activation and measurement integrity pass, proceed with **three complete
54-request suites**, each a separate native invocation and receipt, on the same
loaded processes. A low initial C4 result alone does not cancel those suites.
Use the exact [frozen source and settings](../../../../../docs/historical_benchmarks/baselines/2026-09-19/baseline.json):
revision `ca0b5c11c0bf0cef7271cc0fc21192c7cbe8f087`, source SHA-256
`e0e92cff99f74ffa43d0a82541cf9db6010dd0a4ef3e8e7b84c5e1abd13a867f`,
comparison ID `glm53-nq2-3x-20260911-v1`, model `glm-5.3-flash`, seed `20260905`,
temperature 0, top-p 1, thinking disabled, five decode requests per workload with
8,192 output tokens, three cold/replay pairs at 8,192/32,768/65,536 input tokens,
and three rounds at concurrency 1/2/4 with 256 output tokens and the code workload.
Preserve graph and memory instrumentation settings. Generate a fresh private
`cache_salt` per execution and forward it to both chat and prefill requests.

Set `TP4_REPO_ROOT` to this repository's absolute path before changing checkouts.
Every `./rigmark run` must have a new absolute `--output` under
`$TP4_REPO_ROOT/docs/rigmark_reports/<date-experiment>/`, using `c4-screen.json`
and `run-1.json` through `run-3.json`. Follow the
[native report convention](../../../../../docs/rigmark_reports/README.md).
Inspect each saved result before the next run. On a transport/runtime failure,
preserve partial evidence and stop the series; do not restart automatically.
Output-budget stops and generated-answer quality are accounted for separately.

Keep the initial 12 requests separate from the 162 full-suite requests. Compare
all 16 native per-run metrics with both the fixed two-run baseline medians and
the parent's accepted values, retaining its documented C1 replacement. Report
median, per-run range, percentage deltas, memory, errors and actual counts. Check
for repeatable C4 recovery toward at least 87.527 tok/s while retaining replay
and higher-priority performance. Explicitly record partial recovery or noise;
conflicting primary outcomes require `decision_required`.

Archive unchanged native receipts privately, publish portable numeric extracts,
and update the dated report/index after measurement under the existing
[benchmark procedure](../../../../../docs/benchmarks/README.md). The owner accepted the measured combination after the separate C1 investigation.
Keep new measurements separate and follow the current operating authorization.

## Return to E03 plus replay views

In an authorized coordinated return, stop with this candidate's standalone
overlay, then deploy/start using the preserved **E03 plus replay views** overlay.
Verify the original scheduler hash
`4cea8f55fa51b0371a52a4dd6eb5511ccb0bccbd81e435e6840f68f4ba641518`, the replay
connector pin and both functional gates. This removes the cap flag and selects the
original scheduler while retaining E03, replay views and its existing cache. An
overlay-free start on the new defaults selects the accepted budget scheduler again.
The complete previous September 19 base is available through its reference overlay.
