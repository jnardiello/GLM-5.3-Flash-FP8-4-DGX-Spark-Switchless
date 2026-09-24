# E23: 8-bit drafter copy of the shared lm_head and a hybrid drafter fc

**Recorded outcome: discard.** The owner closed the experiment and the service returned to
the [E22b reference](../baselines/2026-09-23-e22b.md). Three complete native Rigmark
suites, 162 requests, on one retained load, compared with the fixed E22b medians.

## What the candidate changed

E23 extended E22b to the two remaining large BF16 weights that the DFlash2 drafter reads on
every decode step, using the same INT8 group-128 Marlin format as E20–E22b:

- the drafter got its own INT8 copy of the target's vocab-parallel `lm_head` (38,720 × 4,096
  per rank) for top-k candidate selection. The target kept its BF16 `lm_head`, so it still
  chose every generated token;
- the drafter's `fc` projection (4,096 × 20,480) ran INT8 below 2,048 rows and its retained
  BF16 weight at or above that size.

The two modules added 247 MB of packed weights per rank. A byte-traffic model put the removed
decode traffic at about 240 MB per step and rank. A later review corrected the `fc` share to
about 83 MB, for roughly 157 + 83 MB.

## Result against E22b

| Metric | E22b median (range) | E23 median (per suite) | Change |
| --- | ---: | ---: | ---: |
| Code C1 | 41.625 (40.704–41.900) | 43.081 (43.482 / 43.081 / 42.655) | +3.50% |
| Code decode | 56.611 (56.376–56.793) | 57.161 (57.161 / 57.609 / 56.960) | +0.97% |
| Code C2 | 65.780 (64.676–66.832) | 64.492 (64.492 / 62.533 / 66.334) | -1.96% |
| Code C4 | 95.901 (90.213–101.571) | 94.085 (94.085 / 95.505 / 93.875) | -1.89% |
| Prose decode | 33.403 | 33.594 | +0.57% |
| Code / prose TTFT | 0.400 / 0.375 s | 0.387 / 0.376 s | -3.25% / +0.27% |
| Prefill 8K, cold | 2,610.637 (2,573.0–2,641.0) | 2,532.204 | -3.00% |
| Prefill 32K, cold | 2,733.601 (2,718.4–2,803.7) | 2,666.158 | -2.47% |
| Prefill 64K, cold | 2,613.315 | 2,606.358 | -0.27% |
| Replay 8K / 32K / 64K | — | — | +0.38% / -1.10% / -0.80% |

Throughput values are tokens per second. All 162 requests completed with zero validation,
protocol or runtime errors. The runs matched every prefill token count (54/54), passed all
decode output gates and caused no out-of-memory events. Drafter acceptance was 0.60–0.62 per
suite, the same range as E22b.

Matched probes on the same load qualified two of the suite signals:

- 30 identical C1 rounds gave the same C1 per-stream first-token median as E22b (0.387 s).
  The suite's +15% on that metric came from three quantized rounds.
- 10 identical 8K cold pairs gave -0.3%, so the 8K cold loss is not established.

The 32K cold loss is repeatable: every E23 suite fell below the E22b range.

## Why it was closed

The gains were small and repeatable on code and C1. The C2, C4 and cold prefill losses were
also small and pointed the same way. An independent review found no realistic E23 variant
with a clear win.

- **Estimated ceiling:** about +3% on code and C1, even if head-only, `fc`-only,
  vendor-fallback and shape-limited dispatch ablations removed every loss.
- **Likeliest ceiling:** a single surviving component near +1.5%.

That did not justify the reload windows, so the owner closed E23.

Suspected costs recorded for future drafter work:

- the direct BF16 `F.linear` fallback for large `fc` inputs may bypass the vendor's BF16
  linear backend;
- candidate selection also runs on the INT8 head during chunked-prefill proposals.

## Measurement notes

- **Client path.** The accepted suites ran over direct LAN HTTP to the rank-0 API, with zero
  probe loss throughout. The E22b reference used an SSH tunnel over a direct Tailscale path,
  about 1 ms slower per request. That favours the candidate and does not explain its losses.
- **Excluded series.** Earlier series on the same load were excluded and kept privately:
  - one suite deleted and re-run at the owner's request;
  - one partial suite stopped by the owner;
  - three suites or partial suites measured while the Tailscale path fell back to relays with
    20–83% loss.

  In the relayed suite, C2 and C4 fell by 23–29% while engine-side generation matched E22b.
  Check the client path before attributing concurrency losses to a candidate.

[Portable numeric extract](../../historical_benchmarks/experiments/2026-09-23-e23-drafter-head/results.json) ·
[E22b matched probes](../../historical_benchmarks/experiments/2026-09-23-e22b-drafter-context-bf16/matched-probes.json) ·
[Benchmark index](../README.md)
