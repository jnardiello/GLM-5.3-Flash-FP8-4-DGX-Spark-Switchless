# E03 replay with the effective draft budget

**Original measurement assessment: decision_required.** The owner subsequently accepted the candidate after the [separate C1 investigation](2026-09-19-e03-draft-budget-c1-review.md), including its observed tradeoffs even if systematic. The current disposition is **promote**, with the measured combination encoded in the default IaC and subsequent live reproduction pending. This report retains the original measurements and their assessment at completion.

3/3 full native suites, **162 requests**, plus a separate **12-request initial C4 screen**. C4 exceeds the frozen target in all three suites and C2 improves, while C1 throughput is lower in all three; these conflicting primary outcomes required the owner decision.

**The original acceptance criterion required preserving C1 within measurement variability while retaining the C2/C4 and replay gains, without significant penalties elsewhere.** This series alone did not meet that condition; the cause of the C1 reduction remains unresolved. The later owner decision accepts the measured tradeoffs. The completed measurements below are retained unchanged.

A [separate C1 investigation](2026-09-19-e03-draft-budget-c1-review.md) records later isolated repetitions and full-suite reproduction on the same loaded candidate. Those measurements do not replace this series or establish a causal fix by themselves.

The only serving change is an experimental scheduler that caps adaptive verification placeholders to `min(adaptive k, engine maximum, effective draft budget)` using the same R10 `SchedulerOutput` as the draft worker. The original scheduler and its hash are preserved. E03 mHC at 6,912 rows, replay views, FP8, DFlash2, budget table, graphs, 15 GiB KV per rank and 262,144-token context are retained.

The comparison uses the frozen [September 19 baseline](../baselines/2026-09-19.md), exactly two accepted runs / 108 requests, and [E03 plus replay views](2026-09-19-e03-replay-views.md). Parent C1 uses its documented replacement; the parent C4-only repeat remains separate. Neither reference was remeasured.

## Full-suite performance

Values are medians of native per-run metrics; parentheses show observed minimum–maximum across those runs. These ranges are not confidence intervals. Higher throughput and lower TTFT are better. The dataset also retains individual values and sample standard deviations.

| Metric | Baseline median (range) | E03 + replay median (range) | Candidate median (range) | vs baseline | vs parent |
| --- | ---: | ---: | ---: | ---: | ---: |
| Code decode (tok/s) | 53.891 (53.766–54.015) | 53.614 (53.305–55.451) | 53.735 (52.844–54.415) | -0.29% | +0.23% |
| Code C1 aggregate (tok/s) | 40.300 (40.050–40.550) | 39.795 (39.727–40.468) | 38.762 (38.749–39.159) | -3.82% | -2.60% |
| Code C2 aggregate (tok/s) | 57.252 (56.775–57.729) | 60.040 (57.650–60.211) | 62.094 (60.951–63.523) | +8.46% | +3.42% |
| Code C4 aggregate (tok/s) | 87.527 (87.031–88.023) | 82.994 (79.767–86.248) | 90.572 (88.783–92.596) | +3.48% | +9.13% |
| Prose decode (tok/s) | 31.878 (31.506–32.250) | 31.316 (31.167–31.395) | 31.718 (31.327–31.915) | -0.50% | +1.28% |
| Code TTFT (s) | 0.4015 (0.3930–0.4100) | 0.4080 (0.3940–0.4460) | 0.4140 (0.3930–0.4150) | +3.11% | +1.47% |
| Prose TTFT (s) | 0.3800 (0.3790–0.3810) | 0.3830 (0.3830–0.3830) | 0.3810 (0.3800–0.3810) | +0.26% | -0.52% |
| C1 TTFT (s) | 0.3995 (0.3900–0.4090) | 0.3900 (0.3800–0.3970) | 0.3900 (0.3410–0.4000) | -2.38% | +0.00% |
| C2 TTFT (s) | 0.4950 (0.4430–0.5470) | 0.4570 (0.4480–0.4580) | 0.4680 (0.4500–0.5490) | -5.45% | +2.41% |
| C4 TTFT (s) | 0.6795 (0.6780–0.6810) | 0.6840 (0.5360–0.7300) | 0.5240 (0.5230–0.6950) | -22.88% | -23.39% |
| 8K cold (tok/s) | 2,354.666 (2,345.911–2,363.420) | 2,636.501 (2,556.186–2,645.972) | 2,571.838 (2,407.872–2,573.133) | +9.22% | -2.45% |
| 8K replay (tok/s) | 9,557.898 (9,535.205–9,580.592) | 9,611.233 (9,606.511–9,659.455) | 9,436.190 (7,757.561–9,637.624) | -1.27% | -1.82% |
| 32K cold (tok/s) | 2,534.081 (2,518.250–2,549.911) | 2,672.640 (2,633.435–2,793.872) | 2,674.728 (2,611.688–2,784.779) | +5.55% | +0.08% |
| 32K replay (tok/s) | 35,245.167 (34,869.071–35,621.263) | 36,019.795 (34,182.613–36,393.765) | 37,189.665 (36,959.462–38,015.773) | +5.52% | +3.25% |
| 64K cold (tok/s) | 2,421.733 (2,390.227–2,453.238) | 2,545.215 (2,356.561–2,732.473) | 2,686.065 (2,593.238–2,707.803) | +10.92% | +5.53% |
| 64K replay (tok/s) | 35,440.692 (35,407.276–35,474.107) | 39,277.546 (37,209.535–40,433.081) | 39,592.667 (38,964.019–40,216.423) | +11.72% | +0.80% |

**C4 recovery is repeatable in this series:** 88.783 / 92.596 / 90.572 tok/s, all above the 87.527 target and the baseline’s observed 87.031–88.023 range. The median is **90.572 tok/s (+3.48% versus baseline, +9.13% versus E03 plus replay)**. C2 also improves in every suite: its 62.094 tok/s median is **+8.46% / +3.42%** versus those references.

**C1 is the primary tradeoff:** 39.159 / 38.762 / 38.749 tok/s, all below both reference ranges. Its **38.762 tok/s** median is **−3.82% versus baseline and −2.60% versus the parent**. The one-request budget remains five and CPU checks preserve its original adaptive decisions; this does not establish the cause of the measured decrease. C2/C4 gains do not automatically outweigh the repeated C1 reduction, so the outcome is **decision_required**.

The earlier [E03 series](2026-09-19-e03.md) measured C1 at **42.281 tok/s (+4.92% versus baseline)**. Adding replay views measured **39.795 tok/s (−1.25%)**, followed by this candidate's **38.762 tok/s (−3.82%)**. These are separate, noncontemporaneous series; the sequence does not establish which component caused the differences. C1 TTFT remains 0.390 s versus the replay parent, so the unchanged median TTFT does not explain the throughput reduction. C1 uses short 256-token rounds timed over their full duration; the separate long code-decode metric excludes TTFT and cannot establish recovery of C1.

Code decode (−0.29% versus baseline) and prose (−0.50%) have overlapping observed ranges with the baseline. Code TTFT is 12.5 ms higher at the median. Long replay gains are retained: 32K TTFT is **0.881 s**, versus 0.910 s in the parent and 0.930 s in the baseline; 64K is **1.655 s**, versus 1.669 s and 1.849 s. The 64K difference from the parent is within its observed range.

8K replay is less consistent: native per-run throughput is **9,436.190 / 7,757.561 / 9,637.624 tok/s**, with a median **−1.27% versus baseline / −1.82% versus the parent**. The second run’s slower samples remain included. Cold 8K is −2.45% versus the parent but +9.22% versus baseline. No sample, metric block or complete run was excluded.

Client-minus-server summed request duration is **0.622 / 0.643 / 0.630 s** across the respective 54-request suites; TTFT differences are **1.422 / 1.416 / 1.457 s**. Exact API counts match all three suites, and the much larger parent timing anomaly did not recur. These aggregate counters do not identify the cause of individual timing variations.

## Separate initial C4 screen

The initial three rounds measured **83.455 / 94.717 / 91.563 tok/s**, with a native aggregate median of **91.563 tok/s** and **0.523 s per-stream TTFT**. All 12 streams completed visibly at the 256-token budget. It is **+4.61%** versus the frozen C4 median (87.527), **+10.32%** versus the parent full-suite median (82.994), and **+8.93%** versus the parent’s separate C4 repeat (84.054). It does not add a fourth value to the full-suite median.

This screen uses the parent repeat’s native selector source (`3380f0dc…`); the full suites use the clean frozen source (`e0e92cff…`). The exact source and receipt hashes, rounds and counts are retained in the [C4 screen](../../historical_benchmarks/experiments/2026-09-19-e03-replay-draft-budget/c4-screen.json). The native JSON is complete, but its terminal card was not rendered: that renderer expects a decode/code block and logged a presentation warning. All three complete-suite cards are retained; no receipt was rewritten.

## Replay latency and memory

Native prefill TTFT equals first-visible latency for every included eight-token response. The measurements do not describe latency to generate a long answer.

| Depth / phase | Baseline TTFT | Parent TTFT | Candidate TTFT | Candidate per-run values |
| --- | ---: | ---: | ---: | --- |
| 8192 / cold | 3.479 s | 3.107 s | 3.185 s | 3.184 / 3.185 / 3.402 |
| 8192 / warm_replay | 0.857 s | 0.852 s | 0.868 s | 0.868 / 1.056 / 0.850 |
| 32768 / cold | 12.931 s | 12.261 s | 12.251 s | 11.767 / 12.547 / 12.251 |
| 32768 / warm_replay | 0.930 s | 0.910 s | 0.881 s | 0.862 / 0.887 / 0.881 |
| 65536 / cold | 27.066 s | 25.749 s | 24.399 s | 24.203 / 24.399 / 25.272 |
| 65536 / warm_replay | 1.849 s | 1.669 s | 1.655 s | 1.655 / 1.630 / 1.682 |

Per-run restore records verify nine replay requests and 36 rank restores per complete suite, with unchanged 6,912 / 32,256 / 64,512-token coverage. Memory uses the baseline host sampler source and intervals plus the unchanged GPU allocator probe.

| Rank | Baseline minimum available GiB, per run | Parent, per run | Candidate, per run |
| --- | --- | --- | --- |
| 0 | 2.216 / 2.225 | 2.577 / 2.550 / 2.839 | 2.411 / 2.892 / 2.697 |
| 1 | 5.827 / 5.973 | 6.569 / 7.195 / 7.361 | 6.741 / 7.020 / 6.664 |
| 2 | 5.990 / 6.393 | 6.932 / 7.083 / 6.806 | 6.789 / 7.248 / 7.174 |
| 3 | 6.409 / 6.647 | 6.417 / 6.903 / 6.918 | 6.574 / 7.061 / 7.110 |

Across 3 suite windows on four ranks: **0 CUDA OOMs, 0 allocation retries and 0 host OOM kills**. Available-memory minima are sampled, not guaranteed headroom; allocator peaks span the worker lifetime and exclude driver/NCCL/SIRCL allocations. Detailed current/peak counters, sampling gaps, host pressure and OOM/retry deltas are in the [memory comparison](../../historical_benchmarks/experiments/2026-09-19-e03-replay-draft-budget/memory-comparison.json).

## Integrity and operating state

Full-suite completed streams: **162/162**; visible responses: **162/162**. Native receipt, protocol and runtime errors: **0**. Finish reasons: **45 stop, 117 length**. Prefill input counts match **54/54**. Native output gates are retained separately: **code 15/15, prose 15/15, structured 15/15**. No runs or metric blocks are excluded.

Four-rank identity, API counters and access logs are checked after every execution. Runtime records include client/server timing sums and periodic cap counters; the last logged counter omits the unlogged tail. Both documented functional gates passed within 3.295 seconds of observed `/health` 200, before any benchmark. These two gate requests are separate from benchmark counts. All measurements retain the same four container starts and one model load.

The initial C4 receipt and first full-suite appliance metadata retain preparation placeholders saying host observers were not started and GPU verification was pending. Actual four-rank host JSONL and GPU samples prove both were active during those windows; subsequent metadata was corrected. Native receipt bytes are unchanged and the erratum is included in the dataset.

Each complete suite uses the frozen source, prompts, comparison ID, seed, explicit token limits, prefill depths and concurrency settings, with one fresh private cache salt shared by its chat/prefill requests and cold/replay pairs. No additional kernel test, Nsight capture or generated-code quality audit was run.

The healthy E03 plus replay views plus draft-budget candidate remains loaded and idle on all four ranks. Temporary host observers and the task tunnel are stopped. No promotion, rollback, commit or push was performed. A later lifecycle transition needs a new coordinated authorization.

Final offline validation: `./scripts/check.sh` **PASS**, including the 50-test candidate scheduler suite and four-rank manifest/mount/launcher previews. `git diff --check` passed. The frozen scheduler hash and baseline records remain unchanged.

The budget mismatch also exists in the baseline. This experiment can measure the cap’s effect under this recipe; it cannot establish the cause of the parent’s earlier C4 loss. Small, noncontemporaneous samples limit conclusions about differences within observed variability.

[Portable results, all 16 metrics, protocol/source hashes and evidence limits](../../historical_benchmarks/experiments/2026-09-19-e03-replay-draft-budget/results.json). Original native JSON, cards, logs and memory samples are retained privately under the native report convention.
