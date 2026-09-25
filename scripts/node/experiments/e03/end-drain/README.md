# E29: no speculative step past a possible length finish

With asynchronous scheduling the engine dispatches a request's next step before the
output of the step in flight has come back. For a request limited by `max_tokens`, the
vendor guard skips that next step only when the step in flight is certain to finish it,
that is when a single token is missing. With seven draft tokens a step can produce up to
eight tokens, so a length-capped request usually finishes inside a step that could have
produced several: the step already queued behind it then verifies drafts for a finished
request, and a request that arrives as soon as the previous one ends waits for that step.

On E28b a 256-token request that arrived at an idle engine reached its first token in
0.339 s. One submitted as soon as a length-capped predecessor ended took 0.333–0.338 s
when the predecessor's last step emitted a single token (the vendor guard fired) and
0.388–0.439 s otherwise, over 40 predecessors. The standard suites submit their
one-request rounds and every warm replay exactly this way, so the effect appears there as
a slower first token after E28b's seven drafts, while steady decode improved.

E29 changes only the boundaries of a request, never the steps in between:

- `VLLM_E29_END_DRAIN=1` (scheduler): a running request whose committed outputs plus
  pending placeholders reach `max_tokens` is not given another step until the pending
  output arrives. Other requests are scheduled as usual. If drafts were rejected and
  tokens are still missing, the request resumes on the next step. EOS and stop strings
  are unchanged.
- `VLLM_E29_IDLE_COALESCE_MS` (engine core, 0–5 ms, 0 by default): when the engine is idle
  and a request arrives, the loop keeps taking arrivals until one fixed deadline after the
  first before scheduling. Several requests released together, as by agents or a benchmark
  barrier, are then prefilled in one step instead of the first alone. With E29's hold the
  engine is idle more often when such a group arrives, so the window protects concurrent
  first-token time. It refuses a nonzero window with data parallelism.
- `VLLM_E29_TRACE=1` logs arrivals (`E29_TRACE enqueue`), dispatches, retirements, holds,
  resumes and coalescing windows with one monotonic clock, for the diagnosis. Leave it off
  for performance measurements.

With all three variables unset or `0` both files behave as their bases.

**Promoted on September 25, 2026.** The default `cluster.env.example` mounts both files with
`VLLM_E29_END_DRAIN=1`, `VLLM_E29_IDLE_COALESCE_MS=4` and `VLLM_E29_TRACE=0`, measured as the
[E29 reference](../../../../../docs/benchmarks/baselines/2026-09-25-e29.md). The immediate return is
`TP4_ENV=scripts/node/reference/baseline-20260925-e28b.env`. The overlays below refuse the current
default; apply them on that E28b recipe to reproduce the measured loads.

## Files

- `scheduler.py` is the E27c scheduler that E28b mounts
  ([`../queued-cadence/scheduler.py`](../queued-cadence/scheduler.py)) with
  [`scheduler.patch`](scheduler.patch) applied.
- `core.py` is the serving image's `vllm/v1/engine/core.py` with [`core.patch`](core.patch)
  applied.

Both patches only add lines. [`manifest.json`](manifest.json) records the vendor and parent
hashes, and `SHA256SUMS` pins the files deployed to the nodes.

At startup the engine logs `E29_END_DRAIN_READY trace=<0|1>` and, when the window or the
trace is on, `E29_IDLE_COALESCE_READY ms=<W> trace=<0|1>`. The first hold, and then every
thousandth, logs `E29_END_DRAIN_HELD count=<n> resumed=<n>`; `resumed` counts holds whose
step did not finish the request.

## Use

The overlay applies to the default E28b recipe in `cluster.env` and refuses any other base,
a second application or an existing engine-core override. Load A enables the hold, keeps
the window at 0 ms and traces:

```sh
TP4_ENV=scripts/node/experiments/e03/end-drain/delta.env
```

Load B sets the window measured in load A (4 ms) and turns the trace off; it is the
configuration measured by the Rigmark suites:

```sh
TP4_ENV=scripts/node/experiments/e03/end-drain/delta-b.env
```

In load A the arrivals of 21 two- and four-request cohorts spread over 1.45 ms at the median
and 3.17 ms at most, while the engine dispatched the first request alone after about 0.7 ms.
Use the same `TP4_ENV` for every command of a load, stop with the overlay that started the
stack, and return to E28b with `down` under the overlay followed by `deploy`/`up` without
`TP4_ENV`.

## Expected cost

A held request whose pending step falls short is scheduled after that output instead of
overlapping it, one scheduling bubble near `max_tokens` (possibly repeated after further
rejections). An isolated request at an idle engine waits up to the coalescing window. Work
already dispatched to the workers is never cancelled, so a request that ends on EOS can
still leave one step behind it.
