# E27d: a per-step prefill cap while other requests decode

**Recorded outcome: discard.** The owner judged the benefit not significant for the owner's use, and
the service returned to the [E29 reference](../baselines/2026-09-25-e29.md).

- Six complete native Rigmark suites (324 requests) on two candidate loads, compared with the
  fixed E29 medians and one same-night E29 control suite.
- The protocol-2 interference phase and a four-simultaneous-32K diagnostic, each against a
  fresh live-E29 reference.

## Why

On E29 a long prompt is prefilled in chunks of up to 8,192 tokens on one engine step in eight.
Each such mixed step takes about 2.7–3 s, and every request already decoding waits for it.

E27d was chosen with an independent reviewer. The first candidate, a retune of the adaptive
draft-length policy's up threshold (`VLLM_ADAPTIVE_K_UP` 0.58 → 0.80), was rejected before
any service window: it had been tuned on the Rigmark code prompt, and a replay of ten held-out
short coding tasks predicted −0.7% (2/10 tasks improving).

## What the candidate changed

The E29 scheduler plus an additions-only patch, enabled with `VLLM_E27D_PREFILL_CAP=2304` (one
KDA cache block). While any decode was eligible (after E29's end-drain holds are excluded),
one step scheduled at most 2,304 prefill tokens in total, shared by running prefill chunks and
new or resumed prefills.

- Chunks were clipped before Mamba block alignment. The cap was debited only for allocated
  chunks and refunded if a scheduled chunk was preempted.
- With no eligible decode, chunks kept the E29 sizes.
- Rank 0 logged `E27D_PREFILL_CAP_READY tokens=2304 block=2304 interval=8` on both loads.

## Where it helped: long prompts arriving while others decode

Protocol 2, medians of three runs per cell, fresh E29 reference versus E27d:

| Arriving prompt, running decoders | Background decode during the prefill | Longest progress gap | Arriving request TTFT |
| --- | ---: | ---: | ---: |
| 1K, 1 | 10.4 → 11.1 tok/s | 0.67 → 0.67 s | 0.8 → 0.8 s |
| 1K, 3 | 6.9 → 6.2 tok/s | 0.68 → 0.68 s | 0.9 → 0.9 s |
| 8K, 1 | 6.2 → 11.7 tok/s | 2.67 → 1.19 s | 4.0 → 5.3 s |
| 8K, 3 | 6.7 → 11.3 tok/s | 2.67 → 1.25 s | 4.4 → 7.0 s |
| 32K, 1 | 7.1 → 19.5 tok/s | 2.94 → 1.37 s | 15.8 → 22.0 s |
| 32K, 3 | 6.7 → 13.0 tok/s | 3.06 → 1.33 s | 18.0 → 27.6 s |

With four 32K prompts arriving together, per-stream decode rose from 15.4 to 39.3 tok/s and the
median TTFT from 35.6 to 40.2 s. The cohort finished in 59.9–65.7 s, against 60.5–60.8 s.

The cost is the arriving request's first token: 33–60% later at 8K–32K when other requests are
decoding.

## Standard suites

Medians against the fixed E29 values:

| Metric | Load 1 | Load 2 | All six | E29 control |
| --- | ---: | ---: | ---: | ---: |
| Code decode | −1.73% | −1.05% | −1.39% | −3.82% |
| Code C1 | −4.12% | −1.17% | −2.09% | +2.47% |
| Code C2 | −1.28% | +0.33% | −0.47% | +0.61% |
| Code C4 | +0.58% | −0.72% | −0.27% | +1.85% |
| Prose decode | −2.74% | −1.61% | −2.17% | +0.75% |
| C1 / C2 / C4 per-stream TTFT | within ±0.8% | within ±0.8% | within ±0.8% | within ±0.8% |
| Prefill 8K / 32K / 64K cold | −7.2 / −2.0 / −0.5% | +0.5 / +0.4 / +0.3% | −1.3 / −1.3 / 0.0% | +1.7 / +0.6 / +0.3% |

- The cap never engaged during the load-2 suites: no `E27D_PREFILL_CAP_CLIPPED` line appeared.
  In the standard suites, requests arrive one at a time or with prompts shorter than one block.
- The load-1 losses on C1, prose and 8K cold prefill did not repeat on load 2. The E29 control
  also differed from the fixed E29 medians by −3.8% to +2.5% on single-request metrics.
- Single-request metrics therefore vary by several percent between loads. A small residual on
  prose and C1 (about −2% over six suites) has no mechanism in the candidate and was not
  resolved further.
- Integrity: 324/324 streams completed, with zero transport, protocol or runtime errors. Two
  code-decode answers on load 2 reached the 8,192-token budget and are recorded as
  output-budget stops. Post-boot gates passed on every load and restoration. The container
  environment, mounts and command matched the launcher dry-run on 4/4 ranks.

## Measurement notes

- **Client path.** Direct LAN HTTP to the rank-0 API, with zero probe loss in every suite.
- **Excluded series.** The first E29 interference reference was excluded. One client
  background stream received no data for 30 minutes and hit the client timeout, although the
  server was idle and had finished every request without error. The re-run carried a
  watchdog that captures both TCP ends of any such stall; it did not trigger.
- **One load is not enough.** The first load alone suggested a C1 loss that a second load
  did not reproduce. When a candidate cannot plausibly affect a metric, compare loads before
  attributing a few-percent difference to it.

[Portable numeric extract](../../historical_benchmarks/experiments/2026-09-25-e27d-prefill-cap/results.json) ·
[Benchmark index](../README.md)
