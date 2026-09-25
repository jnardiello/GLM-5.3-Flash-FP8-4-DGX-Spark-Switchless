# E27c: keep the prefill cadence while requests are queued

E27 and E27b defer long prefills to one engine step in eight while other requests decode.
The serving image switches that cadence off whenever requests are left waiting in the
queue: after a release step that did not drain the queue, a capacity latch disables the
deferral until the queue is empty. When several long contexts arrive together, the first
stream to start decoding therefore stalls while the others prefill back to back, as it did
before E27. A diagnostic with four simultaneous 32K prompts measured 6.1, 9.0, 14.8 and
53 tok/s per stream on E27b, the same as E22b.

E27c is E27b plus `VLLM_E27C_CADENCE_WHEN_QUEUED=1`: a throttled step defers long prefills
even when the latch is set, as long as a request is decoding. Queued long prompts are then
prefilled one chunk per cadence step, and the streams already decoding keep generating in
the steps between. Short prefills below 2,048 remaining tokens are still admitted on
deferred steps, as in E27b. The expected cost is a longer time to first token for queued
long prompts, one cadence wait per chunk.

**Promoted on September 25, 2026.** The default `cluster.env.example` mounts this scheduler with `VLLM_E27B_SHORT_PREFILL_TOKENS=2048` and `VLLM_E27C_CADENCE_WHEN_QUEUED=1`, measured as the [E27c reference](../../../../../docs/benchmarks/baselines/2026-09-25-e27c.md). The immediate return is `TP4_ENV=scripts/node/reference/baseline-20260924-e27.env`. The overlay below refuses the current default; apply it on that E27 recipe to reproduce the measured candidate.

## Files

`scheduler.py` is the serving image's `vllm/v1/core/sched/scheduler.py` with
[`scheduler.patch`](scheduler.patch) applied. The patch is the E27b patch plus the latch
override; `manifest.json` records the vendor hash. The latch is still computed on every
release step, and the cadence and decode-eligibility checks are the vendor ones. With both
variables unset or `0`, every gate reduces to the vendor condition.

The engine logs `E27C_CADENCE_WHEN_QUEUED_READY interval=8` and
`E27B_SHORT_PREFILL_READY tokens=2048 interval=8` at startup. The first step that keeps the
cadence despite the latch, and then every thousandth, logs `E27C_CADENCE_KEPT_WHILE_QUEUED`.

## Use

```sh
python3 scripts/tests/test-queued-cadence-config.py
./scripts/check.sh
```

A measurement window uses one overlay for every command, following
[docs/operations.md](../../../../../docs/operations.md):

```sh
TP4_ENV=<serving overlay or none> ./scripts/tp4ctl down
TP4_ENV=scripts/node/experiments/e03/queued-cadence/delta.env ./scripts/deploy.sh
TP4_ENV=scripts/node/experiments/e03/queued-cadence/delta.env ./scripts/tp4ctl up
```

The launcher verifies this directory's `SHA256SUMS` whenever the scheduler is mounted.
To return, stop with the same overlay, then deploy and start with no `TP4_ENV`.
