# E03 replay views follow-up

**Outcome: decision_required.** Three complete native runs plus owner-authorized
C1-only and C4-only repeats: **177 benchmark requests**, zero request errors. Only the anomalous
three-request C1 block from run 3 is excluded and replaced by the repeat; **162
requests contribute to the main comparisons**, with no whole run excluded. The
additional **12-request C4 consistency check** is reported separately below.
Long replay latency improves in every run.
Concurrency results and a timing anomaly in the third run limit the overall
performance conclusion. After reviewing the C4 repeat, the owner considered the
roughly five-token/s variation across four streams acceptable for the intended
workload. The healthy candidate was retained after that series and became the
parent of the separately measured
[draft-budget follow-up](2026-09-19-e03-replay-draft-budget.md). Promotion has not
been applied; this record's measurements and outcome are preserved.

The candidate preserves E03 and removes one complete intermediate snapshot copy
during Python cache restore. Model computation, cache format, scheduler, graphs
and the 15 GiB KV pool remain unchanged.

The comparison uses the frozen [September 19 baseline](../baselines/2026-09-19.md) (two accepted runs / 108 requests) and the measured [E03 candidate](2026-09-19-e03.md) (three runs / 162 requests). Neither was remeasured.

## Performance

Medians use three native measurement values. C1 uses full runs 1 and 2 plus the
C1-only repeat; the other 14 metrics use all three complete runs. Lower TTFT is
better; higher throughput is better. The frozen baseline uses exactly two accepted
runs. Original unadjusted values remain in the dataset. The C4-only consistency
check is separate and does not replace a value in this table.

| Metric | Frozen baseline | E03 | Replay candidate | Change vs E03 | Change vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| Code decode (tok/s) | 53.89 | 53.13 | 53.61 | +0.92% | -0.51% |
| Code C1 aggregate (tok/s) | 40.30 | 42.28 | 39.80 | -5.88% | -1.25% |
| Code C2 aggregate (tok/s) | 57.25 | 60.19 | 60.04 | -0.25% | +4.87% |
| Code C4 aggregate (tok/s) | 87.53 | 86.34 | 82.99 | -3.88% | -5.18% |
| Prose decode (tok/s) | 31.88 | 31.55 | 31.32 | -0.73% | -1.76% |
| Code TTFT (s) | 0.4015 | 0.3930 | 0.4080 | +3.82% | +1.62% |
| Prose TTFT (s) | 0.3800 | 0.3810 | 0.3830 | +0.52% | +0.79% |
| C1 TTFT (s) | 0.3995 | 0.3950 | 0.3900 | -1.27% | -2.38% |
| C2 TTFT (s) | 0.4950 | 0.4470 | 0.4570 | +2.24% | -7.68% |
| C4 TTFT (s) | 0.6795 | 0.5240 | 0.6840 | +30.53% | +0.66% |
| 8K cold (tok/s) | 2,354.67 | 2,568.57 | 2,636.50 | +2.64% | +11.97% |
| 8K replay (tok/s) | 9,557.90 | 9,654.56 | 9,611.23 | -0.45% | +0.56% |
| 32K cold (tok/s) | 2,534.08 | 2,775.89 | 2,672.64 | -3.72% | +5.47% |
| 32K replay (tok/s) | 35,245.17 | 33,059.39 | 36,019.79 | +8.95% | +2.20% |
| 64K cold (tok/s) | 2,421.73 | 2,667.62 | 2,545.22 | -4.59% | +5.10% |
| 64K replay (tok/s) | 35,440.69 | 34,578.46 | 39,277.55 | +13.59% | +10.83% |

## Prefill first-visible latency

For these native prefill requests TTFT and first-visible latency coincide. Values below are medians of the native per-run latency medians.

| Depth / phase | Baseline | E03 | Replay candidate | Difference vs E03 |
| --- | ---: | ---: | ---: | ---: |
| 8192 / cold | 3.479 s | 3.189 s | 3.107 s | -82.0 ms |
| 8192 / warm_replay | 0.857 s | 0.849 s | 0.852 s | +3.0 ms |
| 32768 / cold | 12.931 s | 11.805 s | 12.261 s | +456.0 ms |
| 32768 / warm_replay | 0.930 s | 0.991 s | 0.910 s | -81.0 ms |
| 65536 / cold | 27.066 s | 24.567 s | 25.749 s | +1182.0 ms |
| 65536 / warm_replay | 1.849 s | 1.895 s | 1.669 s | -226.0 ms |

The replay improvement is supported by all three run medians: **0.900 / 0.910 /
0.959 s at 32K**, and **1.669 / 1.621 / 1.761 s at 64K**. The three-run median
reduces TTFT by **81 ms at 32K and 226 ms at 64K** versus E03, and by **20 ms and
180 ms** versus the frozen baseline. The eight-token prefill responses have the
same first-visible and first-token timings; these values do not describe the time
to generate a long answer.

## Cache restore evidence

All **27/27 replay requests** restored on all four ranks: **108/108 rank restores**.
Coverage remains 6,912 / 32,256 / 64,512 tokens, with the same encoded sizes.
For each replay, the slowest rank's restore duration was selected; the values below
then take the median within each native run and across the three runs.

| Restore duration | E03 | Replay candidate | Difference |
| --- | ---: | ---: | ---: |
| 32K | 510.7 ms | 445.7 ms | -65.0 ms |
| 64K | 1,096.0 ms | 877.8 ms | -218.2 ms |

This supports a reduction in actual restore work. It does not establish the cause
of the earlier E03-versus-baseline replay difference. Per-rank timings and coverage
are retained in the portable run restore records.

## Variability and timing anomaly

The API counters and access logs contain exactly **54 requests per run**, matching
Rigmark, with no unaccounted requests or aborts. Competing user traffic was not
observed. In run 3, code and prose contain TTFT samples of **6.622 s** and **6.586 s**;
one 32K replay took **7.114 s**. Those latency samples remain in the original
receipts and aggregates; the owner requested only the C1 measurement be repeated.

Existing client and server measurements show a larger discrepancy in that run:

| Sum over 54 requests | Run 1 | Run 2 | Run 3 |
| --- | ---: | ---: | ---: |
| Client minus server TTFT | 1.443 s | 1.418 s | 34.966 s |
| Client minus server total duration | 0.638 s | 0.633 s | 80.200 s |
| Server queue duration | 4.761 s | 4.674 s | 4.738 s |

The timing boundaries differ, and these aggregate counters cannot identify each
affected sample or prove a cause. At the owner's request, run 3's C1 block, whose
aggregate was **24.933 tok/s**, was excluded only from C1 throughput and TTFT and
repeated once on the same loaded candidate. All other measurements were retained.

The native C1-only repeat made exactly **three requests** and measured
**39.795 / 39.281 / 41.623 tok/s**, with a median of **39.795 tok/s** and **0.380 s
TTFT**. Client-minus-server total duration was **0.036 s** across those requests;
there were no additional requests, stream errors or restarts. The revised C1
comparison uses **40.468 / 39.727 / 39.795 tok/s**, a median of **39.795**, versus
baseline **40.300** (**-1.25%**). The targeted repeat had different preceding
workload and idle duration from a full suite; this limitation remains explicit.

Rigmark required a native `--skip-decode` selector to run only C1, combined with
`--skip-prefill --concurrency 1`. Its request and timing function bodies are
unchanged, verified by AST comparison; 59 local tests passed. The repeat records
the different source hash and selected scope, while the original three full runs
retain the reference source hash. The
[C1 repeat record](../../historical_benchmarks/experiments/2026-09-19-e03-replay-views/c1-recheck.json)
contains the selector patch hash, protocol, request counts and individual rounds.
The [timing review](../../historical_benchmarks/experiments/2026-09-19-e03-replay-views/timing-review.json)
preserves the counters, affected latency examples and receipt hashes.

C2 throughput retains the gain over baseline (**+4.87%**). Code and prose change
by **-0.51% / -1.76%**; the small series does not establish effects of that size.
C4 throughput is **-5.18% versus baseline**, and C4 TTFT returns to approximately
baseline latency, losing E03's earlier improvement. Cold 32K/64K remain faster
than baseline but are slower than E03. These differences remain unresolved;
replay gains alone do not settle the primary concurrency tradeoff. Observed
per-run ranges are included in the dataset and are not confidence intervals.

## C4-only consistency check

The owner requested a repeat of only C4. One native invocation used
`--skip-decode --skip-prefill --concurrency 4 --concurrency-runs 3`: **three rounds
of four 256-token code requests**, with the same prompt construction, comparison
ID, seed, sampling, source and instrumentation as the C1 repeat. The candidate
remained loaded; there was no restart, additional warmup or other inference.

| C4 measurement | Throughput | Per-stream TTFT |
| --- | ---: | ---: |
| Frozen baseline, two-run median | 87.527 tok/s | 0.6795 s |
| E03, three-run median | 86.342 tok/s | 0.524 s |
| Replay candidate, original three-run median | 82.994 tok/s | 0.684 s |
| C4-only repeat, one execution | **84.054 tok/s** | **0.536 s** |

The repeat's three aggregate rates are **80.638 / 90.128 / 84.054 tok/s**. Its
median is **1.28% above** the original candidate median, **3.97% below baseline**
and **2.65% below E03**. The lower throughput therefore remains observable in the
repeat. TTFT is **12 ms above E03** and **143.5 ms below baseline**, recovering
the earlier latency advantage in this check.

All **12/12 streams** completed visibly with `length` at the intended 256-token
cap. API counters and access logs show exactly 12 requests, with no extra traffic.
Client-minus-server summed duration is **0.160 s** and summed TTFT difference is
**0.169 s** across all 12 requests. The large timing discrepancy seen in full run 3
did not recur. The four workers were unchanged; no CUDA OOM or allocation retry
was observed, and temporary host observers were stopped afterward.

This is one targeted check after a different preceding workload and idle duration.
It preserves the observed throughput difference without proving its cause or a
statistically significant connector effect. The owner assessed this C4 variation
as acceptable measurement noise for the intended four-stream workload. This is
the recorded operating judgment, rather than a statistical significance claim.
Original C4 measurements and the main
16-metric aggregates remain unchanged. The
[C4 repeat record](../../historical_benchmarks/experiments/2026-09-19-e03-replay-views/c4-recheck.json)
contains all rounds, protocol/source hashes, counter checks and evidence limits.

## Validation and protocol

The prepared connector passed CPU checks and the same compact GPU check on all four ranks after the coordinated stop: restored bytes and physical slots match exactly, with zero absolute/relative error, untouched blocks preserved and damaged framing rejected. These checks do not substitute for native performance measurements.

Native receipt, protocol and runtime errors: **0**. Completed streams and visible
responses: **177/177**, including the C1 and C4 repeats. Prefill input counts match **54/54**. Native decode output
gates pass **15/15** for each of code, prose and structured generation. Finish
reasons across all measurements are **45 stop and 132 length**; the latter are the
bounded prefill and concurrency responses. The accepted metric subset has
**45 stop and 117 length**. There are no code output-budget stops or excluded
whole runs; the three original C1 requests are excluded only from that metric
block. Output gates remain separate from performance integrity.

Rigmark revision, prompts, comparison ID, seed, request settings, prefill depths
and C1/C2/C4 settings match the frozen reference. Each full execution has 54
requests and a fresh private cache salt. One coordinated transition and one
candidate load cover the series. The same containers and workers were retained
through that measurement window.
Both post-boot gates passed within **3.21 seconds**; the gates and matching E03
activation request add **three separate validation requests**, for **180 total
inference requests** including the C1 and C4 repeats. No baseline remeasurement or
additional restart was performed.

## Memory and retained state

The same host sampler source and intervals and the same GPU allocator probe cover
all three runs. All 12 rank/run windows report **zero CUDA OOMs, allocation retries
and host OOM kills**. Rank-0 minimum available RAM is **2.58 / 2.55 / 2.84 GiB**,
versus baseline **2.22 / 2.23 GiB** and E03 **2.12 / 2.99 / 3.01 GiB**. These sampled
minima do not establish a guaranteed memory margin. The
[memory comparison](../../historical_benchmarks/experiments/2026-09-19-e03-replay-views/memory-comparison.json)
retains every rank's measurements and their limits.

KV capacity remains **1,344,328 tokens**, with **15 GiB per rank** and a
**262,144-token** context limit. Four-rank source identity and health 200 were
verified after each run. Temporary host observers were stopped; E03 plus the
replay connector was healthy and idle at that handoff. The measured E03 connector
and rollback overlay are retained. No promotion, commit or push was applied.

[Portable results, counts and source hashes](../../historical_benchmarks/experiments/2026-09-19-e03-replay-views/results.json) include all per-run values. Unchanged native receipts, GPU checks and operational logs are retained privately under the native report convention.
