# E27b: prefill cadence for long prefills only

E27 admits prefill work on one engine step in eight while other requests decode. That keeps
running requests generating while another client loads a long context. It also delays
short prompts: in C4, whose four streams start together, a stream admitted one step late
waits for the next cadence step, and per-stream TTFT rose 35%.

E27b keeps `--prefill-schedule-interval 8` but applies it only to long prefills. On a step
where the cadence defers prefill work, the scheduler still admits a prefill with fewer than
`VLLM_E27B_SHORT_PREFILL_TOKENS` (2,048) remaining uncached tokens, up to 2,048 such tokens
per step in total. Longer prefills wait for the cadence as in E27.

**Status: measured, then promoted inside E27c.** The [E27b report](../../../../../docs/benchmarks/experiments/2026-09-24-e27b-long-prefill-cadence.md) records its three suites and interference phase. The [E27c default](../queued-cadence/README.md) mounts the same patch plus one more flag, so this overlay now refuses the default; apply it on `scripts/node/reference/baseline-20260924-e27.env` to reproduce the measured candidate.

## How it works

`scheduler.py` is the serving image's `vllm/v1/core/sched/scheduler.py` with
[`scheduler.patch`](scheduler.patch) applied. `manifest.json` records the vendor hash.
The patch changes only the two deferral gates of `schedule()`:

- **Running prefill chunks.** A chunk is deferred unless its remaining tokens fit the
  short-prefill limit.
- **Waiting requests.** The remaining tokens are counted after the prefix cache and
  SparkCache lookup, so a long prompt that is mostly cached counts as short. A deferred
  long request is set aside and returned to the front of the waiting queue in order. It no
  longer stops the pass, so short requests queued behind it are still admitted.

The cadence, the decode-eligibility check and the capacity latch are the vendor ones. With
the variable unset or `0`, every gate reduces to the vendor condition.

2,048 is below the 2,304-token KDA block, so an admitted short prefill always finishes in
one final chunk. The per-step total bounds the extra step time to about one short prefill.
The short tail of a long prefill is also admitted without waiting for the cadence, which
may reduce part of E27's gain for the running requests.

The engine logs `E27B_SHORT_PREFILL_READY tokens=2048 interval=8` at startup. The first
admission and deferral, and then every thousandth, log `E27B_SHORT_PREFILL_ADMITTED` and
`E27B_LONG_PREFILL_DEFERRED`.

## Use

Run the offline checks from the repository root:

```sh
python3 scripts/tests/test-long-prefill-cadence-config.py
./scripts/check.sh
```

The test rebuilds the vendor file from the override and the patch, and checks the vendor
hash and the exact removed lines. It also covers the predicate boundaries (2,047 / 2,048 /
2,049), the per-step limit and four-rank launcher parity with the E27 default. The overlay
refuses any other base.

A measurement window uses one overlay for every command, following
[docs/operations.md](../../../../../docs/operations.md):

```sh
TP4_ENV=<serving overlay or none> ./scripts/tp4ctl down
TP4_ENV=scripts/node/experiments/e03/long-prefill-cadence/delta.env ./scripts/deploy.sh
TP4_ENV=scripts/node/experiments/e03/long-prefill-cadence/delta.env ./scripts/tp4ctl up
```

The launcher verifies this directory's `SHA256SUMS` whenever the scheduler is mounted.
To return, stop with the same overlay, then deploy and start with no `TP4_ENV`.
