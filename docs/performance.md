# Measured performance

This page holds the current accepted performance reference, how it was measured and what
the deployed default changes. The [project README](../README.md) summarizes it; the
[benchmark archive](benchmarks/README.md) holds every earlier baseline and experiment.

## Current reference

Current accepted baseline: **28/09/2026 · E31, speculative-safe C4 tail ring**, measured
with upstream, unmodified [Rigmark](https://github.com/alexellis/rigmark) and the reference
flags of its GLM receipts: every default plus `reasoning_effort` low. **One complete suite
(54/54 requests, n = 1)**, zero measurement/runtime errors and 15/15 native output gates
passing. The comparison is an arm with the E29 arithmetic (legacy tail), measured on the same
load with the same flags.

| Workload | Current · 28/09/2026 E31 | vs same-load E29-equivalent arm ([📊 E31 report](benchmarks/baselines/2026-09-28-e31.md)) |
| --- | ---: | ---: |
| Code decode, one request | 65.19 tok/s | 64.56 tok/s · +0.97% |
| Code C1, end-to-end | 43.99 tok/s | 40.99 tok/s · +7.32% |
| Code C2, aggregate end-to-end | 68.37 tok/s | 73.10 tok/s · -6.48% |
| Code C4, aggregate end-to-end | 101.45 tok/s | 108.67 tok/s · -6.65% |
| Prose decode | 34.09 tok/s | 34.28 tok/s · -0.54% |
| Code TTFT | 0.475 s | 0.473 s · +0.42% |
| Prose TTFT | 0.374 s | 0.385 s · -2.86% |
| C1 per-stream TTFT | 0.421 s | 0.415 s · +1.45% |
| C2 per-stream TTFT | 0.469 s | 0.457 s · +2.63% |
| C4 per-stream TTFT | 0.548 s | 0.555 s · -1.26% |
| Prefill 8K, cold | 2,572.5 tok/s | 2,600.2 tok/s · -1.07% |
| Prefill 8K, replay | 8,848.1 tok/s | 8,708.4 tok/s · +1.60% |
| Prefill 32K, cold | 2,697.9 tok/s | 2,766.0 tok/s · -2.46% |
| Prefill 32K, replay | 33,944.4 tok/s | 32,797.3 tok/s · +3.50% |
| Prefill 64K, cold | 2,614.4 tok/s | 2,595.3 tok/s · +0.74% |
| Prefill 64K, replay | 39,576.6 tok/s | 40,326.5 tok/s · -1.86% |

Each value comes from a single suite, so differences of a few percent are within noise. The
C1/C2/C4 rows rest on three short rounds each and moved about ±7% in both directions.
Percentages use unrounded values. Higher throughput and lower TTFT are better.

- **Workloads.** C1/C2/C4 mean one, two or four concurrent requests. Decode excludes the
  initial wait; end-to-end speed includes it. TTFT is time to first token.
- **Output limits.** Concurrency outputs cap at 256 tokens; long decode allows 4,096.
- **Cache isolation.** Each suite has a fresh comparison ID, which isolates the prefix cache.
- **Scope.** These are inference measurements, not complete agent-task timings.

**The numbers above are not comparable with earlier ones.** Up to September 25 (E29), the
README showed three-suite medians measured with a local Rigmark fork, 8,192-token decode,
thinking off and a cache salt. Those records remain in the
[benchmark archive](benchmarks/README.md) under their own protocol.

## What E31 changes

The pooled indexer builds one key from every four tokens. It keeps each request's recent
rows in a four-slot tail so a pool can be completed across steps. A DFlash2 verify step
writes eight consecutive positions, including drafts that may be rejected. Their rows
overwrote committed members of the pool still being built, and after a rejection that pool
was completed from rejected-draft keys.

- **Fix:** the tail becomes a ring of 12 slots for seven drafts, so no row of one step can
  reach a committed member of the open pool. A GPU test with the real kernels found 200
  wrong pools with the old tail and none with the ring. vLLM merged the equivalent fix
  upstream on September 25.
- **Cost:** none measurable. Code and prose decode move less than 1% against the same-load
  E29-equivalent arm.
- **Head gate on tensor cores:** included but off. Its same-load comparison is unresolved.
- **Cache:** the SparkCache namespace was kept. Pools built before E31 could carry the old
  defect if they were persisted and restored.

The [current benchmark report](benchmarks/baselines/2026-09-28-e31.md) lists all three
arms, the leaf tests, the excluded series and limitations.

## Deployed default

The operational default keeps the E31 model, context and scheduler lineage, adds bounded
SparkCache disk transfers, and limits aggregate request admission. It uses a 14 GiB KV pool,
allocator trim before eligible eager prefills, a 6,912-token scheduling-step cap, six active
API admission slots and 128 queued requests. Admission slots do not imply resident engine
requests. This safety configuration has its own
[versioned operational identity](operational-identities/2026-09-29-memory-bounded.json);
it does not replace or alter the frozen E31 performance baseline. The complete
[protected 16 GiB rollback](../scripts/node/reference/operational-20260929-sparkcache-protected.env)
removes the allocator, step-cap and admission deltas while keeping cache protection. The
[E31 rollback](../scripts/node/reference/baseline-20260928-e31.env) also restores the measured
cache behavior. The [memory resilience page](resilience.md) explains these protections and
their test results.

The final native Rigmark suite on this default completed **54/54 requests with zero
recorded errors**. Against the initial protected recipe's single suite, code decode changed
−0.2%, C4 aggregate throughput −0.5%, 8K cold prefill −6.4%, and 8K replay −18.9%. These are
descriptive results on different loads, not a new performance baseline or proof of
equivalence. The
[resilience report](benchmarks/experiments/2026-09-29-sparkcache-resilience.md#final-native-measurement)
lists every metric of both suites.

The [current recipe](production-recipe.md), encoded in
[`cluster.env.example`](../cluster.env.example), combines the digest-pinned SparkRing image,
DFlash2 with adaptive verification capped by its effective draft budget, E03 mHC prefill
sharding, hybrid INT8/BF16 KDA input projections, E21 8-bit weights for the KDA output
and MLA attention projections, E22b 8-bit weights for the DFlash2 drafter, the E27 prefill
cadence with the E27c scheduler, seven draft tokens (E28b), the E29 length-finish hold and
idle coalescing, the E31 speculative-safe indexer tail, SparkCache replay views and
SIRCL/patched NCCL.
The operational **14 GiB KV pool per rank** is 12.5% smaller than E31's measured 16 GiB
pool; the **262,144-token context limit** is unchanged. Startup reports 1,194,033 KV tokens
(4.55 full contexts), so five complete 256K contexts cannot be resident at once; scheduling
must use waiting or preemption for that five-client workload.

## Earlier comparison figures

The graphs below still compare the earlier **E29 and E28b** medians under the old protocol.
The plotting script compares two frozen records, while E31's comparison arm lives inside
its own record, so no E31 figure was generated. Click an image for its SVG version.

[![Earlier E29 versus E28b baseline (old protocol): generation throughput, time to first token and percentage changes.](plots/comparisons/2026-09-25-e29-vs-2026-09-25-e28b/generation.png)](plots/comparisons/2026-09-25-e29-vs-2026-09-25-e28b/generation.svg)

[![Earlier E29 versus E28b baseline (old protocol): cold prefill, immediate replay and percentage changes at 8K, 32K and 64K.](plots/comparisons/2026-09-25-e29-vs-2026-09-25-e28b/prefill.png)](plots/comparisons/2026-09-25-e29-vs-2026-09-25-e28b/prefill.svg)

The [benchmark archive](benchmarks/README.md) retains earlier baselines, experiments
and separate reproduction results. See [local Rigmark reports](rigmark_reports/README.md)
for saving and viewing native receipts.
