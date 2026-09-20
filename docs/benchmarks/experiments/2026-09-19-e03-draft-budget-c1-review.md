# C1 investigation on the retained draft-budget candidate

**Outcome: promote — accepted by the owner.** The owner accepts the observed performance tradeoffs even if the remaining decreases are systematic. The [decision record](../../historical_benchmarks/experiments/2026-09-19-e03-draft-budget-c1-review/owner-decision.json) preserves the earlier measurement assessments and source hashes. The measured combination is encoded in the default IaC. The [subsequent September 20 deployment](../reproductions/2026-09-20-e03-iac.md) passed its functional and identity checks; the owner excluded that window's benchmark from evaluation.

This follow-up retains the same serving processes, code and configuration as [E03 plus replay views plus the draft-budget cap](2026-09-19-e03-replay-draft-budget.md). It completed **three C1-only executions / nine requests**, followed by **three complete native suites / 162 requests**: **171 requests in total**. These observations are separate from the original three suites; no earlier values or frozen references are replaced. Recovery without a runtime change does not establish a causal fix or prove that every decrease is noise.

**C1 recovers to 40.233 tok/s (−0.17% versus baseline)** in the complete-suite follow-up, with per-run values **40.233 / 39.786 / 41.190**. Its observed range overlaps the baseline's 40.050–40.550 range. The original −3.82% decrease is not reproduced in this series, and its cause remains unisolated.

The broader result remains mixed: **C2 +8.00%, C4 −0.68%, prose −2.17%** versus baseline. Only one of the three new C4 values exceeds the 87.527 tok/s target, compared with all three in the original candidate series. Long replay gains remain. The measurement-only assessment was unresolved; the owner's acceptance resolves the disposition while retaining these limitations.

## Separate C1-only repetitions

| Execution | C1 aggregate (tok/s) | vs frozen baseline |
| --- | ---: | ---: |
| C1-only 1 | 41.708 | +3.49% |
| C1-only 2 | 41.217 | +2.28% |
| C1-only 3 | 41.200 | +2.23% |
| Median of three | **41.217** | **+2.28%** |

The baseline C1 median is 40.300 tok/s from exactly two accepted full suites. The original draft-budget series measured 38.762 tok/s (−3.82%). All nine new isolated requests completed visibly at the 256-token limit, with exact API counts and no request/protocol/runtime errors. The existing native selector differs only in its skip-decode CLI handling; 26 common functions/classes, including request construction and concurrency timing, are unchanged from the frozen source.

## Complete-suite reproduction

Values below are medians of the native per-execution values for the completed suites. Throughput is tok/s; TTFT is seconds. The JSON retains all per-run values, observed ranges and sample standard deviations. Observed ranges are not confidence intervals.

At the owner's request, small changes, including decreases of roughly 1–2%, are labelled approximately unchanged in the baseline comparison. Exact deltas remain visible. This is a presentation convention, not a statistical equivalence finding.

| Metric | Frozen baseline | Original budget series | Follow-up | vs baseline | vs original budget |
| --- | ---: | ---: | ---: | ---: | ---: |
| Code decode | 53.891 | 53.735 | 53.809 | ≈ unchanged (-0.15%) | +0.14% |
| Code C1 aggregate | 40.300 | 38.762 | 40.233 | ≈ unchanged (-0.17%) | +3.79% |
| Code C2 aggregate | 57.252 | 62.094 | 61.834 | +8.00% | -0.42% |
| Code C4 aggregate | 87.527 | 90.572 | 86.932 | ≈ unchanged (-0.68%) | -4.02% |
| Prose decode | 31.878 | 31.718 | 31.186 | ≈ unchanged (-2.17%) | -1.68% |
| Code TTFT | 0.4015 | 0.4140 | 0.3960 | ≈ unchanged (-1.37%) | -4.35% |
| Prose TTFT | 0.3800 | 0.3810 | 0.3810 | ≈ unchanged (+0.26%) | +0.00% |
| C1 TTFT | 0.3995 | 0.3900 | 0.3430 | -14.14% | -12.05% |
| C2 TTFT | 0.4950 | 0.4680 | 0.4490 | -9.29% | -4.06% |
| C4 TTFT | 0.6795 | 0.5240 | 0.5720 | -15.82% | +9.16% |
| 8K cold | 2,354.666 | 2,571.838 | 2,558.401 | +8.65% | -0.52% |
| 8K replay | 9,557.898 | 9,436.190 | 9,417.164 | ≈ unchanged (-1.47%) | -0.20% |
| 32K cold | 2,534.081 | 2,674.728 | 2,661.974 | +5.05% | -0.48% |
| 32K replay | 35,245.167 | 37,189.665 | 37,177.429 | +5.48% | -0.03% |
| 64K cold | 2,421.733 | 2,686.065 | 2,616.073 | +8.02% | -2.61% |
| 64K replay | 35,440.692 | 39,592.667 | 40,078.253 | +13.09% | +1.23% |

Each complete suite uses the frozen source, prompts, comparison ID, seed, token limits, prefill depths and concurrency settings, with a fresh cache salt shared within that execution. Host and GPU memory instrumentation is retained from the baseline recipe. Each execution is checked before the next; native receipts, cards, logs and samples are retained privately.

## Evidence and limits

[Portable diagnostic extracts and their measurement limits](../../historical_benchmarks/experiments/2026-09-19-e03-draft-budget-c1-review/diagnostics.json).

- The one-request engine budget remains five; the CPU integration checks preserve its adaptive decisions. Periodic cap counters remain flat in the sampled C1-only windows. No accidental three-token C1 cap is established.
- A read-only four-rank GPU snapshot during the second full suite shows P0, 2,470–2,515 MHz SM clocks, 68–69 °C, and no active power or thermal clock-limit reason. A single snapshot does not establish conditions throughout the earlier series.
- A local sizing-only CPU test adds about two microseconds per uncapped call. This is a different host and is neither a production cost estimate nor evidence that the branch explains the measured decline.
- Historical slow C1 windows show no host OOM, effectively zero measured memory-pressure stalls and no cache publish completions during those windows. Periodic logs do not exclude all background work or unsampled transients.
- Client output-event counts differ: median 88 for the original candidate requests versus 82 in the isolated repeats, with median per-request mean output-event intervals of about 71.7 and 72.2 ms respectively (decode duration divided by event count minus one). This is consistent with a change in tokens delivered per output event, not a demonstrated engine or network cause. Actual server draft counters for the isolated repeats are preserved separately.
- The first complete follow-up suite recovers C1 at 40.233 tok/s (−0.17% versus baseline), after the same preceding workload. That workload alone is therefore insufficient to reproduce the original slowdown.
- Non-benchmark traffic occurred between the original series and this follow-up. Retained processes do not establish identical workload history or an immediate A/B comparison.
- The initial three slow suites remain valid measurements. Neither selected fast repeats nor different preceding workloads justify replacing them. A stable deployment penalty and the source of the observed variability remain unisolated.

## Integrity, memory and retained state

Completed streams and visible responses: **171/171**. API completed-request delta: **171**, matching all six executions and the intervening gaps. Native receipt, protocol and runtime errors: **0**. Finish reasons: **45 stop, 126 length**. Prefill input counts match **54/54**; native code, prose and structured output gates each pass **15/15**. No runs, metric blocks or slow samples were excluded.

All **24 rank measurement windows** record zero GPU OOMs, allocation retries and host OOM kills. Full-suite minimum available memory in GiB is:

| Rank | Full 1 | Full 2 | Full 3 |
| --- | ---: | ---: | ---: |
| 0 | 2.360 | 2.300 | 2.257 |
| 1 | 6.696 | 6.939 | 6.528 |
| 2 | 6.655 | 6.805 | 6.609 |
| 3 | 6.667 | 6.558 | 6.548 |

These are sampled minima, not guaranteed headroom. The frozen baseline's rank-0 minima are 2.216 / 2.225 GiB. The portable per-execution extracts retain pressure, allocator counters and lifetime-peak limits. Client-minus-server request-duration sums are **0.638 / 0.618 / 0.627 s** across the respective 54-request suites, comparable with the original series; no large transport timing anomaly recurred.

No serving code, recipe, model load or lifecycle action was changed. Final four-rank identity passes, including 19 source hashes per rank, unchanged process starts and `/health` 200. The service was left loaded and idle; the task's four host observers and SSH tunnel are stopped. All six native JSON receipts and the three complete-suite cards are retained. The C1-only renderer's missing-code-block warnings are presentation limitations, separate from successful request integrity.

The owner has accepted the measured candidate. Recording this decision added no measurements, runtime changes or lifecycle actions. The subsequent IaC deployment verification and exclusion of the September 20 benchmark are recorded in the [promotion record](../../historical_benchmarks/baselines/2026-09-19-e03/promotion.json). The frozen reference and both original measurement records remain unchanged.

[Portable results, counts, source hashes, all 16 metric comparisons and individual extracts](../../historical_benchmarks/experiments/2026-09-19-e03-draft-budget-c1-review/results.json).
