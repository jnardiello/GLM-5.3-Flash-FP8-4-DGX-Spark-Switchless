# E37: two-round neighbor reductions on the four-rank ring

**Recorded outcome: discard; technical NO-GO from independent review.**
All measured butterfly candidates are slower than the production NCCL ring. E36 remains
the operational reference and its frozen records are unchanged. E36 is restored on all
four ranks: /health 200, both functional gates passed within 28.022 seconds of readiness,
and the operational identity check passed.

The screen measures collectives with the model stack stopped. It includes **zero native
Rigmark suites and zero model benchmark requests**. The comparison is the unchanged
production PyNccl/NCCL ring in profile P0 on the same four-node fabric, rather than a
historical isolated-latency table. The loaded recipe before the stop was
[E36](../baselines/2026-09-30-e36.md).

| Profile / butterfly payload | p50 µs, rows 4 / 8 / 32 | Ratio to P0/NCCL, rows 4 / 8 / 32 |
| --- | --- | --- |
| P0 FP32 | 67.552 / 85.824 / 212.960 | 1.414 / 1.595 / 2.238 |
| P0 BF16 | 55.360 / 67.392 / 168.000 | 1.159 / 1.252 / 1.765 |
| P1 FP32 | 63.488 / 79.840 / 213.024 | 1.329 / 1.483 / 2.238 |
| P1 BF16 | 55.200 / 65.536 / 167.776 | 1.155 / 1.218 / 1.763 |
| P2 FP32 | 81.984 / 112.800 / 245.792 | 1.716 / 2.096 / 2.583 |
| P2 BF16 | 63.488 / 89.856 / 194.528 | 1.329 / 1.669 / 2.044 |

P0/NCCL p50 is **47.776 / 53.824 / 95.168 µs**. The preregistered gate required a butterfly
ratio of at most **0.6 at all three sizes**, with p99 no worse. All **18 p50 ratio
bootstrap intervals at 95% have a lower bound above 1**, with a minimum of 1.113.
P0/FP32 intervals are [1.391, 1.419] / [1.558, 1.601] / [2.219, 2.242].
Its slowdown repeats in all three P0 blocks, so the conclusion does not depend on
differences between process launches.

P0/FP32 p99 is 307.364 / 390.209 / 466.589 µs, against NCCL's
285.851 / 273.822 / 365.055 µs. Some other p99 intervals overlap parity.
They do not offset the clear p50 failure.

The benchmark uses the installed Torch 2.13.0+cu130 and PyNccl wrapper with the existing
patched NCCL 2.30.7 binary, Ring and four channels. The production permitted protocol set
`LL,LL128,Simple` remains unchanged; no protocol is narrowed and no new NCCL binary is built.

- P0 keeps the production environment.
- P1 adds only `NCCL_ALLOC_P2P_NET_LL_BUFFERS=1`.
- P2 also sets `NCCL_P2P_LL_THRESHOLD=1048576`. This is the per-channel selection
  threshold; the configuration alone does not prove which protocol every transfer used.

Each profile starts four fresh processes, one per node. Butterfly round 1 uses
`rank ^ 1`; round 2 uses `rank ^ 3`. Both use physical ring edges. Canonical operand
order gives every rank the same result. The FP32 variant transfers a BF16 input and
an FP32 pair sum, or 3P bytes sent per rank for an original P-byte input. The BF16
variant sends 2P. The direct diagnostic exchanges both neighbors, forwards the
left input right, and sums the four inputs in global rank order; it sends 3P.
The production ring sends 1.5P. Two rounds therefore do not imply lower byte traffic.

Width is 4,096 BF16 values, with 1/4/8/16/32/64/128/256 rows. All four arms have
2,000 pilot operations per shape, 200 isolated calls and uninstrumented graph means.
NCCL and both butterfly arms then have three blocks of 10,000 operations at 4/8/32 rows.
Each graph contains 100 reductions with separate outputs and small compute operations
between them. Communication and arithmetic share one explicit CUDA stream.

The primary percentiles come from external CUDA event pairs around each complete
reduction. The maximum of the four rank durations is taken for each logical operation.
Graph averages include the simulated compute and are separately diagnostic.
Bootstrap uses 2,000 resamples of whole replay clusters; separate arm batches use
independent replay draws. Blocks pair within P0; comparisons across profiles draw
blocks independently. Three blocks and separate processes limit generalization and
do not eliminate between-process drift. Four ranks are not four independent repeats.

The numerical oracle sums bounded BF16 inputs exactly in CPU FP64, then rounds through
FP32 to BF16. Each shape has 32 qualifying mixed-amplitude/outlier fixtures plus three
diagnostics: extreme cancellation, half-ULP and signed zero. Rank bit identity and finite
outputs are required for every fixture; numerical non-inferiority uses the qualifying corpus.

| Rows | NCCL mean ULP | Butterfly BF16 mean ULP | Butterfly FP32 max / mean ULP |
| ---: | ---: | ---: | ---: |
| 4 | 8.000156 | 8.420874 | 0 / 0 |
| 8 | 8.413966 | 8.379641 | 0 / 0 |
| 32 | 8.041340 | 8.212017 | 0 / 0 |

FP32 satisfies numerical non-inferiority on this corpus. The cancellation diagnostic
demonstrates that this is not universal exactness. BF16 has a smaller maximum ULP than
NCCL at these sizes, but fails the mean error requirement at 4 and 32 rows in all profiles.

Accepted counts are **12 complete native rank receipts, 177 logical timing cells and
1,002,000 instrumented logical operations**: 192,000 pilot and 810,000 confirmation.
Numerical bit identity is **3,360/3,360 fixture/arm calls**; nonfinite outputs are
**0/875,642,880 checked logical elements**, including diagnostics. The qualifying corpus
has 66,715,648 elements per arm and profile. Warmups, isolated calls and plain graph
replays are separate from the instrumented operation count.

**The checked million-operation mixed soak was not run.** Independent review approved omitting it
because every candidate was already disqualified by confirmed latency. The timed
operations do not substitute for this soak and no soak PASS is claimed. Phase 2 and
native Rigmark are skipped after NO-GO.

For P1's FP32 configuration, the requested projection `(p50_A - p50_B) × 102`
yields **−1.603 / −2.654 / −12.021 ms**
saved at 4/8/32 rows: an added cost. Against the earlier E33 diagnostic step medians
of 60.7 ms for C1 and 123.6 ms for C4, these are **−2.640 / −4.372 / −9.726%**.
These projections assume 102 qualifying collectives per step. They are diagnostic
arithmetic estimates, not measurements of model throughput.

All rank processes complete with exit code zero. An archive transfer failure is recovered
with unchanged original bytes and independent native/local SHA-256 equality. A failed P2
preflight occurs before any GPU launch; the subsequent preflight passes. No accepted GPU
profile is repeated. Failure receipts and recovery details remain private. All three
completed profiles leave no benchmark containers or GPU processes.

The [portable numeric extract](../../historical_benchmarks/experiments/2026-10-01-e37-allreduce-butterfly/results.json)
contains the complete protocol, per-cell percentiles, numerical denominators, critical
ratios and confidence intervals, source identities and the 12 immutable receipt hashes.
Raw collective JSON and lifecycle logs are retained privately, separately identified
from native Rigmark receipts. No frozen baseline is extended or replaced.
