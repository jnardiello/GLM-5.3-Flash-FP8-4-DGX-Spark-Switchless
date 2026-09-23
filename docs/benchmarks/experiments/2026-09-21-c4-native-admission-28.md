# C4 native admission: four active and 24 waiting

**Performance outcome:** `unresolved`. Functional verification passed and the single
owner-requested native Rigmark suite completed 54/54 requests, but one run cannot
establish repeatable performance.

**Integration disposition:** `owner_withdrawn`. The experiment is archived rather than
selected, the coordinated return to the complete original C4 recipe passed, and the
frozen E03 reference is unchanged. Because the candidate was withdrawn, its source-derived
overrides, overlay and local tests are not published in this repository; this record and
its portable extract are the published evidence, and the private native receipts named
below are retained by their digests.

This candidate keeps the pinned R10 image and adds 13 source-derived vLLM admission
overrides. It is a scoped backport rather than an image or vLLM release upgrade. FP8,
DFlash2, SparkCache, SIRCL, hybrid KDA, the E03 memory guards, 262,144-token context,
12 GiB KV per rank and 8,192 batched tokens remain selected. The final native cap counts
active plus waiting sequences: four may run and 24 may wait, and the 29th sequence
receives HTTP 503 at saturation. The native scheduler admits newly available slots
continuously.

## Functional verification

The work used two coordinated stages. The initial eight-slot stage qualified the
backport, stream-safe HTTP rejection, cleanup paths and bounded cold/replay capacity.
The second stage changed only the admission cap from 8 to 28; all 13 runtime override
files and the remaining recipe stayed unchanged.

| Stage | Gates after health | Admission result | Full-context result |
| --- | --- | --- | --- |
| Initial 8 total | 2/2 in 23.537 s | 4 active + 4 waiting; additional request 503; 8 completed, 2 cancelled; both cancellation slots recovered | Cold 4/4 and replay 4/4 at 261,632 prompt + 512 output tokens |
| Final 28 total | 2/2 in 11.931 s | 4 active + 24 waiting; 29th sequence 503; 28 completed, 2 cancelled; both cancellation slots recovered | Replay 4/4 at 261,632 prompt + 512 output tokens |

Across both stages, 58 functional generation attempts produced 52 completed responses,
four intentional cancellations and two intentional HTTP 503 overload responses. This
count includes the two coherent-response/tool gates in each stage and excludes the
separate Rigmark suite. Four-rank runtime identity passed after the 28-slot transition,
and health remained 200 after the admission probe. The default autostart still selects
the E03 base without an experiment overlay.

The initial cold stage recorded zero metric failures or new OOM/transport errors and
minimum sampled `MemAvailable` of 4.698 / 9.217 / 9.420 / 9.211 GiB by rank. The final
28-slot replay recorded 62 metric samples, zero failures or fatal/OOM candidates, and
minimum sampled `MemAvailable` of 5.389 / 8.925 / 10.109 / 10.035 GiB. Its maximum
sampling gap was 0.502 seconds. Four restore completions were observed on every rank.

## Recomputed native capacity

The final boot reported 640 raw GPU blocks per rank and native maximum concurrency
4.102564102564102, equivalent to 1,075,462 whole tokens and 156 blocks per maximum-size
request. Four physical requests plus the reserved null block require
`4 * 156 + 1 = 625 <= 640`. These values were recomputed from the candidate boot; the
historical C4 block targets were not imposed as acceptance thresholds.

The runtime did not expose individual KV-group page sizes, so the per-group
decomposition is unavailable. The aggregate metrics and bounded full-context requests
are the available cross-check. Aggregate scheduler samples do not independently prove
active residency on each rank, and restore logs do not expose a one-to-one
restore-to-request mapping. Sampled memory minima do not exclude unsampled transients.
Cold capacity was not repeated after the cap-only 8-to-28 change; its initial-stage
result remains separate evidence.

## Owner-withdrawn restoration

One coordinated transition stopped the native-admission overlay, deployed the complete
original C4 `c4-262k.env` recipe and started all four ranks. Down, deploy and up
returned zero. `/health` reached 200, then the coherent response and tool-call gates
both passed 28.486 seconds after first observed health.

The focused four-rank identity check passed every assertion: the original C4 limits
were four sequences, 262,144-token context, 12 GiB KV per rank and 8,192 batched tokens;
the queue-cap flag, all native-admission mounts and its compiler-cache namespace were
absent. The 13 affected in-image modules matched the original R10 source hashes. All 18
tracked standard/E03 Python mounts and the separately pinned C4 connector matched their
expected hashes. The image, base configuration, DFlash2, SparkCache, SIRCL, mHC and
draft-budget guards matched; one stack owned the GPU processes, health was 200 and idle,
the flusher was stopped, and rank-0 autostart remained enabled without an overlay or
drop-in. Default E03 autostart configuration therefore remains unchanged even though
the verified live recipe is the explicitly selected original C4 return. Restored boot
logs contained zero traceback, CUDA OOM, NCCL error or other error candidates across
the four ranks. They reported 1,075,462 KV tokens and 4.10 maximum concurrency at the
262,144-token request limit.

The standard live verifier independently exited zero with 166 PASS, 0 FAIL, 2 WARN and
7 SKIP results; the non-passing rows remain recorded as such. These two restoration
gates are separate from the archived experiment's 58 functional generation attempts
and the benchmark's 54 requests.

| Private receipt | SHA-256 |
| --- | --- |
| Coordinated transition summary | `7705c0c7b327fc901cd1d8d22f2cbedcf57fcbd9f7588ab107166b8cb4c863ec` |
| Focused four-rank identity summary | `1297cc567923e34ba9cccf3744719b41a60976edc2e9d9761844bd04a355bc76` |
| Standard live-verifier summary | `a98a8822937823c1cb0c1e2f1d2ea8db429849a1d2de2d5d9ebb99026ec0ef55` |
| Standard live-verifier log | `66fc5b81668af679dac9346eaaf273039526d2e9832841de49496e8b5e5f61bc` |

## Performance measurement

One complete native Rigmark suite / 54 requests ran under the owner-authorized
single-run scope. It used the fixed source fingerprint
`e0e92cff99f74ffa43d0a82541cf9db6010dd0a4ef3e8e7b84c5e1abd13a867f`
at revision `ca0b5c11c0bf0cef7271cc0fc21192c7cbe8f087`, Rigmark protocol 1.1.0,
prompt set 1.0.0 / `0c3ac4015813af6a1b967659bd2289990fde895a5e43ae065b894d1e1f909ecb`,
comparison ID `glm53-nq2-3x-20260911-v1` and served model `glm-5.3-flash`.
The settings matched the frozen E03 specification: seed 20260905, temperature 0,
top-p 1, thinking disabled; five code/prose/structured decode requests per workload
with an 8,192-token output budget; three cold/replay pairs at 8K, 32K and 64K with
eight output tokens observed per request; and three code rounds at concurrency 1/2/4
with a 256-token output budget. One fresh cache salt was shared within this suite's
cold/replay pairs and sent to chat and prefill requests; its raw value is omitted.

The suite used the same 16 metric source paths as the accepted E03 record. The native receipt,
protocol and runtime validators report no errors. All 54 streams completed visibly;
finish reasons were 15 stop and 39 length, all 15 decode gates passed, and all 18
prefill inputs matched their requested token counts.

| Metric | Fixed E03 median | Candidate run | Change |
| --- | ---: | ---: | ---: |
| Code decode (tok/s) | 53.809 | 54.057 | +0.46% |
| Code C1 aggregate (tok/s) | 40.233 | 40.520 | +0.71% |
| Code C2 aggregate (tok/s) | 61.834 | 61.367 | -0.76% |
| Code C4 aggregate (tok/s) | 86.932 | 91.960 | +5.78% |
| Prose decode (tok/s) | 31.186 | 31.391 | +0.66% |
| Code TTFT (s) | 0.396 | 0.393 | -0.76% |
| Prose TTFT (s) | 0.381 | 0.383 | +0.52% |
| Code C1 per-stream TTFT (s) | 0.343 | 0.397 | +15.74% |
| Code C2 per-stream TTFT (s) | 0.449 | 0.448 | -0.22% |
| Code C4 per-stream TTFT (s) | 0.572 | 0.612 | +6.99% |
| 8K cold prefill (tok/s) | 2,558.401 | 2,534.790 | -0.92% |
| 8K replay prefill (tok/s) | 9,417.164 | 9,552.829 | +1.44% |
| 32K cold prefill (tok/s) | 2,661.974 | 2,764.744 | +3.86% |
| 32K replay prefill (tok/s) | 37,177.429 | 37,014.236 | -0.44% |
| 64K cold prefill (tok/s) | 2,616.073 | 2,709.212 | +3.56% |
| 64K replay prefill (tok/s) | 40,078.253 | 39,651.692 | -1.06% |

The primary measurements are mixed: C4 throughput is +5.78%, while C1 and C4
per-stream TTFT are 54 ms and 40 ms slower respectively. C1, C2, decode, prose and
prefill measurements contain smaller changes in both directions. One suite cannot
establish that any gain or regression is repeatable, so the performance outcome remains
unresolved and this observation does not replace the three-run frozen reference. The
image and vLLM release remained the pinned R10 build, so the +5.78% C4 observation did
not follow a vLLM upgrade. The single run also cannot attribute it specifically to the
13 admission backports.

The 585.936-second window recorded minimum sampled `MemAvailable` of 7.049 / 10.554 /
10.682 / 10.878 GiB by rank. Maximum meminfo gaps were 0.641 / 0.399 / 0.387 / 0.391
seconds, with no meminfo errors, stable worker identity and zero host OOM, CUDA OOM or
allocator-retry deltas. Logs contain 54 rank-0 generation POSTs (36 chat-completion and
18 completion) and 18 tokenizer POSTs, 72 total, all HTTP 200; the headless ranks
received no API POSTs. The generation count matches the success-counter delta of 54.
Nine cache commits and
nine restores were observed per rank. The service finished healthy and idle, all host
observers and their SSH samplers stopped, and the four-rank runtime identity passed
again. At evidence capture the variant was active and idle, while default autostart
still selected E03 without the overlay. The owner later withdrew the native-admission
cap; its sources and receipts remain archived, outside active deployment and default
validation paths, and the coordinated return to the original C4 recipe is recorded
above.

The client OS differs from the frozen series, and its timing effect was not isolated.
The host sampler used the same source and target intervals as E03, but observer overhead
was not isolated and actual gaps are reported above. Its minima are sampled observations;
PyTorch allocator counters omit driver, NCCL and SIRCL allocations, and lifetime GPU
peaks extend outside this run. Host pressure counters cannot independently establish a
performance cause. Cache log totals do not identify a one-to-one request mapping.

[Portable results, counts, source hashes and receipt digests](../../historical_benchmarks/experiments/2026-09-21-c4-native-admission-28/results.json)
· [Benchmark index](../README.md)
