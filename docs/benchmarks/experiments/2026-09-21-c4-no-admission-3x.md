# Original C4 without native admission: three-suite comparison

**Status:** `complete`. **Performance outcome:** `unresolved`.

The owner requested three consecutive complete native Rigmark suites on the restored
original C4 recipe, retaining the existing weight load and making no further lifecycle
or recipe change. Four suites / 216 requests were measured: runs 1, 3 and 4 provide
the three accepted suites / 162 requests, while run 2 is excluded as documented below.
The result is complete but unresolved, with no performance-based promotion.

The measured recipe keeps the pinned R10 image, FP8, DFlash2, SparkCache, SIRCL, hybrid
KDA, the E03 mHC and draft-budget sources, four scheduled sequences, 262,144-token
context, 12 GiB KV per rank and 8,192 batched tokens. The native-admission queue cap,
its 13 source overrides and its separate compiler-cache namespace are absent. The
accepted E03 defaults, frozen reference bytes and autostart selection remain unchanged.

## Matched protocol

The execution specification is the frozen E03 record: Rigmark 1.1.0 at revision
`ca0b5c11c0bf0cef7271cc0fc21192c7cbe8f087`, source SHA-256
`e0e92cff99f74ffa43d0a82541cf9db6010dd0a4ef3e8e7b84c5e1abd13a867f`, prompt set
1.0.0 / `0c3ac4015813af6a1b967659bd2289990fde895a5e43ae065b894d1e1f909ecb`,
comparison ID `glm53-nq2-3x-20260911-v1` and served model `glm-5.3-flash`.

Each complete execution uses seed 20260905, temperature 0, top-p 1 and thinking
disabled. It includes five code, prose and structured decode requests with an
8,192-token output budget; three cold/replay pairs at 8K, 32K and 64K with eight output
tokens; and three code rounds at concurrency 1/2/4 with a 256-token output budget. Each
execution gets a fresh cache salt, shared only within that run's cold/replay pairs and
forwarded to both chat and prefill requests.

Native receipts are written directly by Rigmark, without a benchmark wrapper, to:

- `docs/rigmark_reports/2026-09-21-c4-no-admission-3x/run-1.json`
- `docs/rigmark_reports/2026-09-21-c4-no-admission-3x/run-2.json`
- `docs/rigmark_reports/2026-09-21-c4-no-admission-3x/run-3.json`
- `docs/rigmark_reports/2026-09-21-c4-no-admission-3x/run-4.json` (replacement for
  the excluded run 2; original receipt slots remain unchanged)

## Run accounting

| Scope | Accepted target | Completed/measured | Accepted | Excluded | Still needed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Complete native suites | 3 | 4 | 3 | 1 | 0 |
| Requests | 162 | 216 | 162 | 54 | 0 |

The accepted aggregate uses runs 1, 3 and 4 only. Run 2's 54 complete native rows stay
preserved as excluded diagnostic evidence and do not enter any aggregate.

Before run 1, one local Rigmark invocation exited 2 during metadata validation because
four required values were nested instead of present at the top level. It issued zero
model requests and created no native receipt. The corrected metadata passed Rigmark's
native `load_metadata` function before the accepted run. This preparation retry is not
an excluded suite, a transport failure or one of the four completed executions. The
rejected metadata and its private failure receipt hash to
`0bf1064f771167f748ef4a1b3bc010a7b5ddc734e742eea9758234e7e5a94546` and
`1c68bdf19962ba32f670dce80d6220caf48b8ccc7ec1316d4866c259810b2d42`.

### Accepted run 1

Run 1 completed all 54 requests with 54 DONE streams and 54 visible responses. Native
receipt validation, protocol validation and runtime errors were all zero. The three
decode output gates passed 15/15, and all 18 prefill token counts matched. Finish
reasons were 15 `stop` and 39 `length`; the latter are the expected bounded prefill and
concurrency outputs and remain recorded as native results.

| Metric | Run 1 | Fixed E03 median | Change |
| --- | ---: | ---: | ---: |
| Code decode throughput (tok/s) | 54.904 | 53.809 | +2.03% |
| Code C1 aggregate throughput (tok/s) | 39.315 | 40.233 | -2.28% |
| Code C2 aggregate throughput (tok/s) | 64.694 | 61.834 | +4.63% |
| Code C4 aggregate throughput (tok/s) | 85.523 | 86.932 | -1.62% |
| Prose decode throughput (tok/s) | 32.035 | 31.186 | +2.72% |
| Code TTFT (s) | 0.396 | 0.396 | 0.00% |
| Prose TTFT (s) | 0.384 | 0.381 | +0.79% |
| Code C1 per-stream TTFT (s) | 0.344 | 0.343 | +0.29% |
| Code C2 per-stream TTFT (s) | 0.447 | 0.449 | -0.45% |
| Code C4 per-stream TTFT (s) | 0.538 | 0.572 | -5.94% |
| 8K cold prefill (tok/s) | 2,591.766 | 2,558.401 | +1.30% |
| 8K replay prefill (tok/s) | 9,740.634 | 9,417.164 | +3.43% |
| 32K cold prefill (tok/s) | 2,796.617 | 2,661.974 | +5.06% |
| 32K replay prefill (tok/s) | 38,427.980 | 37,177.429 | +3.36% |
| 64K cold prefill (tok/s) | 2,718.159 | 2,616.073 | +3.90% |
| 64K replay prefill (tok/s) | 39,507.435 | 40,078.253 | -1.42% |

The before/after service evidence stayed healthy and idle, with 54 successful model
requests added. Logs recorded 36 chat, 18 completion and 18 tokenization responses, all
HTTP 200, with no OOM, allocator retry or fatal-log candidate. Sampled minimum
MemAvailable by rank was 6.105/10.611/10.638/10.692 GiB; maximum sampling gaps were
659/364/367/387 ms. These are observed diagnostic minima and gaps, not guarantees about
unsampled transients or evidence of a performance cause.

The native receipt SHA-256 is
`f42ac11d180285553febf29cdf95d3a40d6f9aa9e91b66e12546d30a64c378a7`.
The private portable extraction, acceptance and memory summaries hash to
`dc4567687118c3517d762940be2fd152d90e589377ef3fe9904ab800aa1b3691`,
`ec415bb2441f15e0c9e181e4218730948e7b4f332f4c81d466f92685c709f36a` and
`1692703ccfa303c35a614634e76a412d73965765f775de9b443f7597e03a6ed7`.
This per-run observation is one input to the final three-suite aggregate below.

### Excluded run 2

Run 2 also preserved 54/54 DONE streams and visible responses, zero native receipt or
protocol validation errors, no runtime error, 15/15 decode gates and 18/18 matched
prefill token counts. Exact-window logs nevertheless contained 41 successful chat
POST response headers and 18 successful completion POST response headers: five more
accepted generation requests than the native receipt's 36 chat plus 18 completion
requests. All 18 tokenization calls also returned HTTP 200, and the four rank logs
contained no fatal candidate. This confirms additional accepted request traffic. Run 2
is therefore excluded from the accepted aggregate
while its complete native rows and values remain diagnostic evidence. Private
connection-ownership evidence subsequently
verified a separate non-benchmark client as the source; site and client details remain
private.

| Diagnostic metric | Run 2 | Fixed E03 median | Change |
| --- | ---: | ---: | ---: |
| Code decode throughput (tok/s) | 53.905 | 53.809 | +0.18% |
| Code C1 aggregate throughput (tok/s) | 37.911 | 40.233 | -5.77% |
| Code C2 aggregate throughput (tok/s) | 31.272 | 61.834 | -49.43% |
| Code C4 aggregate throughput (tok/s) | 47.958 | 86.932 | -44.83% |
| Prose decode throughput (tok/s) | 31.088 | 31.186 | -0.31% |
| Code TTFT (s) | 0.396 | 0.396 | 0.00% |
| Prose TTFT (s) | 0.382 | 0.381 | +0.26% |
| Code C1 per-stream TTFT (s) | 0.481 | 0.343 | +40.23% |
| Code C2 per-stream TTFT (s) | 0.527 | 0.449 | +17.37% |
| Code C4 per-stream TTFT (s) | 0.557 | 0.572 | -2.62% |
| 8K cold prefill (tok/s) | 2,615.232 | 2,558.401 | +2.22% |
| 8K replay prefill (tok/s) | 9,538.214 | 9,417.164 | +1.29% |
| 32K cold prefill (tok/s) | 2,776.888 | 2,661.974 | +4.32% |
| 32K replay prefill (tok/s) | 37,747.151 | 37,177.429 | +1.53% |
| 64K cold prefill (tok/s) | 2,615.515 | 2,616.073 | -0.02% |
| 64K replay prefill (tok/s) | 40,145.670 | 40,078.253 | +0.17% |

The excluded native receipt and its portable extraction hash to
`f6d28b9f07d8bf8cfb2d35d9c5b2edd66f56cdc96bea747b4a1040ebff2e6566` and
`558b6385fe7ee6ca1d1d9de534530109e40516392fbeddd5edb9278e993cfd4c`.
The private exclusion receipt hashes to
`8815d031017e89bfd3b5f116c14ced1b9e48138b1582fc04686d0c80598142bd`;
it retains the exact log and observer evidence without publishing site identity.
The owner-requested quiet window was established after the competing client was paused.
Runs 3 and 4 then passed. Run 2 remains an excluded observation rather than a benchmark
failure or discard.

### Accepted run 3

Run 3 completed all 54 requests with 54 DONE streams and visible responses, zero native
receipt or protocol validation errors, no runtime error, 15/15 decode gates and 18/18
matched prefill token counts. Exact-window evidence contained precisely 36 chat, 18
completion and 18 tokenization HTTP 200 response headers, all generation clients were
loopback, and health was 200 and idle before and after with a success-counter delta of
54. All four workers remained stable, with no fatal log candidate, host/GPU OOM or
allocator retry.

| Metric | Run 3 | Fixed E03 median | Change |
| --- | ---: | ---: | ---: |
| Code decode throughput (tok/s) | 52.460 | 53.809 | -2.51% |
| Code C1 aggregate throughput (tok/s) | 39.943 | 40.233 | -0.72% |
| Code C2 aggregate throughput (tok/s) | 62.214 | 61.834 | +0.61% |
| Code C4 aggregate throughput (tok/s) | 89.197 | 86.932 | +2.61% |
| Prose decode throughput (tok/s) | 31.282 | 31.186 | +0.31% |
| Code TTFT (s) | 0.395 | 0.396 | -0.25% |
| Prose TTFT (s) | 0.382 | 0.381 | +0.26% |
| Code C1 per-stream TTFT (s) | 0.413 | 0.343 | +20.41% |
| Code C2 per-stream TTFT (s) | 0.467 | 0.449 | +4.01% |
| Code C4 per-stream TTFT (s) | 0.625 | 0.572 | +9.27% |
| 8K cold prefill (tok/s) | 2,606.937 | 2,558.401 | +1.90% |
| 8K replay prefill (tok/s) | 9,289.077 | 9,417.164 | -1.36% |
| 32K cold prefill (tok/s) | 2,669.210 | 2,661.974 | +0.27% |
| 32K replay prefill (tok/s) | 37,925.224 | 37,177.429 | +2.01% |
| 64K cold prefill (tok/s) | 2,606.453 | 2,616.073 | -0.37% |
| 64K replay prefill (tok/s) | 37,839.998 | 40,078.253 | -5.58% |

Sampled minimum MemAvailable by rank was 6.821/11.026/11.172/11.149 GiB;
maximum sampling gaps were 609/536/362/405 ms. The native receipt, portable extraction,
acceptance and memory summary hash to
`ddccaf7cbf412b1c5ecd6d75fe6f1636ab4815ef8f834608dfc012da34d9975b`,
`0bc4dd447b8adf4d6f8a3a99b7b61a4e147931f5c7790eb232b1107921593c04`,
`18e3780d5ea0c2dad4fbc25b07f3db04e73a129ffc7d11e999ca3f49b61673be`
and `eba4a35991ad151066cd746e9f87f370c07db2caee1468697245ad352cf8133c`.
These observer values have the same sampled-diagnostic limitations as run 1.

### Accepted run 4

Run 4 repeated run 3's complete integrity result: 54/54 DONE streams and visible
responses, zero native receipt or protocol validation errors, no runtime error, 15/15
decode gates and 18/18 matched prefill token counts. Its exact-window counts were 36
chat, 18 completion and 18 tokenization HTTP 200 response headers, all generation
clients were loopback, health was 200 and idle before and after, and the successful
request delta was 54. All four workers remained stable with no fatal log candidate,
host/GPU OOM or allocator retry.

Sampled minimum MemAvailable by rank was 6.509/10.882/10.987/11.084 GiB; maximum
sampling gaps were 708/581/338/425 ms. The native receipt, portable extraction,
acceptance and memory summary hash to
`ea1e6580e0fc7f1a241568fb68ef339e95cd061127d5c9a544edd6ca2494d4e2`,
`6d82b83f7fe93d1368d4c1baeeb47f6511859d9097485357ed6c46ab7b23f379`,
`baf1a956fed03a85a21e91e7f45e4aa3b853eef39293eef3b90d97671dc01905`
and `1d5bcbcee347c022fe60c5d511829566a3ab4ac15cf9edec3fa19526a0774009`.

## Accepted aggregate

The accepted integrity totals are 162/162 DONE streams, 162/162 visible responses,
45/45 decode gates and 54/54 matched prefill requests. Across three receipts there
were zero native validation errors, zero protocol errors and zero runtime errors.
Finish reasons were 45 `stop` and 117 `length`. Exact-window evidence totals 108 chat,
54 completion and 54 tokenization HTTP 200 response headers, with a successful-request
delta of 162 and no accepted-run OOM, allocator retry or fatal-log candidate.

The table retains every accepted native per-run value. `SD` is the sample standard
deviation across runs 1, 3 and 4; it is descriptive rather than a confidence interval.
Positive TTFT deltas are slower, while positive throughput deltas are faster.

| Metric | Run 1 | Run 3 | Run 4 | Median | Range | SD | E03 median | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `code_decode_throughput` | 54.904 | 52.460 | 52.912 | 52.912 | 52.460-54.904 | 1.300 | 53.809 | -1.67% |
| `code_c1_aggregate_end_to_end_throughput` | 39.315 | 39.943 | 39.090 | 39.315 | 39.090-39.943 | 0.442 | 40.233 | -2.28% |
| `code_c2_aggregate_end_to_end_throughput` | 64.694 | 62.214 | 61.053 | 62.214 | 61.053-64.694 | 1.860 | 61.834 | +0.61% |
| `code_c4_aggregate_end_to_end_throughput` | 85.523 | 89.197 | 88.543 | 88.543 | 85.523-89.197 | 1.960 | 86.932 | +1.85% |
| `prose_decode_throughput` | 32.035 | 31.282 | 30.753 | 31.282 | 30.753-32.035 | 0.644 | 31.186 | +0.31% |
| `code_ttft` | 0.396 | 0.395 | 0.413 | 0.396 | 0.395-0.413 | 0.010 | 0.396 | +0.00% |
| `prose_ttft` | 0.384 | 0.382 | 0.393 | 0.384 | 0.382-0.393 | 0.006 | 0.381 | +0.79% |
| `code_c1_per_stream_ttft` | 0.344 | 0.413 | 0.396 | 0.396 | 0.344-0.413 | 0.036 | 0.343 | +15.45% |
| `code_c2_per_stream_ttft` | 0.447 | 0.467 | 0.451 | 0.451 | 0.447-0.467 | 0.011 | 0.449 | +0.45% |
| `code_c4_per_stream_ttft` | 0.538 | 0.625 | 0.540 | 0.540 | 0.538-0.625 | 0.050 | 0.572 | -5.59% |
| `prefill_8k_cold_throughput` | 2,591.766 | 2,606.937 | 2,636.464 | 2,606.937 | 2,591.766-2,636.464 | 22.730 | 2,558.401 | +1.90% |
| `prefill_8k_replay_throughput` | 9,740.634 | 9,289.077 | 9,487.884 | 9,487.884 | 9,289.077-9,740.634 | 226.315 | 9,417.164 | +0.75% |
| `prefill_32k_cold_throughput` | 2,796.617 | 2,669.210 | 2,743.571 | 2,743.571 | 2,669.210-2,796.617 | 64.000 | 2,661.974 | +3.07% |
| `prefill_32k_replay_throughput` | 38,427.980 | 37,925.224 | 37,278.429 | 37,925.224 | 37,278.429-38,427.980 | 576.278 | 37,177.429 | +2.01% |
| `prefill_64k_cold_throughput` | 2,718.159 | 2,606.453 | 2,688.449 | 2,688.449 | 2,606.453-2,718.159 | 57.857 | 2,616.073 | +2.77% |
| `prefill_64k_replay_throughput` | 39,507.435 | 37,839.998 | 36,432.449 | 37,839.998 | 36,432.449-39,507.435 | 1,539.322 | 40,078.253 | -5.58% |

## Assessment

Primary results are mixed and do not establish a robust gain: code decode is -1.67%,
C1 throughput -2.28%, C2 +0.61% and C4 +1.85%. The C4 median lies within the frozen
E03 run range of 85.997-88.889 tok/s and has one accepted run below the reference
median, so it is not an established repeatable gain. C1 is below the fixed E03 median
in all three accepted runs; its 39.315 tok/s median is below the frozen 39.786-41.190
tok/s run range. This is a caution signal and is not dismissed as noise. Prose is
+0.31%. Code TTFT is unchanged, while C1 TTFT is 0.053 seconds / 15.45% slower with
wide per-run variation and C4 TTFT is 5.59% faster. The lower-priority 64K replay
result is 5.58% slower and also has wide variation.

Three historical comparisons do not establish formal statistical significance, a
robust qualification gain or a proven disqualifying regression. No fixed percentage
floor was applied. These mixed primary tradeoffs provide no automatic promotion or
discard decision, so the outcome is `unresolved`; an owner decision would be required
before accepting the tradeoffs and promoting this performance result.

The earlier native-admission run measured C4 at 91.960 tok/s, +5.78% versus E03. It was
one run on a different runtime, whereas this original-C4 median is 88.543 tok/s,
+1.85%. Neither comparison isolates admission behavior as the cause. The serving image,
content ID and pinned R10 release never changed, so the earlier +5.78% cannot be
attributed to a vLLM release upgrade.

Final focused identity verification passed on all four ranks after measurement,
including the original 13 R10 modules, standard/E03 mounts, connector, runtime flags,
one-stack ownership, health/idle state, autostart base and absence of native admission.
Its private summary SHA-256 is
`1297cc567923e34ba9cccf3744719b41a60976edc2e9d9761844bd04a355bc76`.
The private whole-series verification manifest hashes to
`268be8b505decfeb764fe5e95557d3e0c5215e6e2f53253aa4b5f8711c91b50c`.
The separate private final assessment hashes to
`d38f000f696ca3cf87c780afa9db0dd606049b7c1b926ee888ee091b76cbc4af`.
The original C4 service remains selected and healthy; the accepted E03 default and
frozen reference are unchanged.

[Portable final record and per-run extracts](../../historical_benchmarks/experiments/2026-09-21-c4-no-admission-3x/results.json)
· [Benchmark index](../README.md)
