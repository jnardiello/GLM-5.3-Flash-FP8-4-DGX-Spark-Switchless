# Production recipe

This page explains the current components and why they are present. Exact values and
one-step rollback comments live in [`cluster.env.example`](../cluster.env.example);
host/software pins live in `scripts/node/bootstrap/versions.env`, model file manifests
in `scripts/node/model-manifests/`, and NCCL pins in `scripts/node/nccl/`.

The **current operational recipe** uses the engine configuration from the
[accepted September 28 E31 performance reference](historical_benchmarks/baselines/2026-09-28-e31/baseline.json):
R10, SIRCL, hybrid KDA, E03 mHC prefill sharding, SparkCache replay views,
batch-uniform adaptive verification capped by the effective draft budget, E21
8-bit residual attention projections, E22b 8-bit DFlash2 drafter linears, the E27
prefill cadence (`--prefill-schedule-interval 8`), the E27c scheduler, E28b (seven draft
tokens for a single request), E29 (no speculative step
past a possible length finish, 4 ms idle coalescing) and E31 (a speculative-safe C4 tail
ring in the pooled indexer, head-gate switch off). It retains the 262,144-token
context limit. It also selects the protected SparkCache connector: disk transfers use
8 MiB pieces, stores and restores share a 1 GiB transient reservation budget per rank,
and admission preserves 1 GiB of `MemAvailable`. `cluster.env.example` encodes all of
these values directly, without an experiment overlay. The cache protection is an
operational default; it was not part of the E31 Rigmark measurement.
The operational memory layer uses a 14 GiB KV pool, allocator trim before eligible eager
prefills, a 6,912-token per-step scheduler cap, and finite rank-0 API admission (six active
slots, 128 queued). The measured E31 pool was 16 GiB. The operational pool reports 1,194,033
KV tokens, or 4.55 maximum-length contexts, so five complete 262,144-token contexts cannot
be resident together. In the five-client functional checks, all five clients overlapped,
while engine scheduling used waiting and preemption rather than five resident full contexts.
Those checks used an instrumented connector, an isolated cache namespace, and 3 GiB/2 GiB
eviction limits. Production uses the protected connector, its normal namespace, and the 0/0
disk-capacity policy, so the checks do not establish arbitrary production-cache growth or
new performance results.

The versioned [operational identity](operational-identities/2026-09-29-memory-bounded.json)
pins the frozen E31 record by hash and records the memory and protected-cache deltas. The
complete [protected 16 GiB rollback](../scripts/node/reference/operational-20260929-sparkcache-protected.env)
is the immediate operational predecessor: it preserves the bounded SparkCache connector but
restores the 16 GiB pool and removes admission, trim and step-cap selections. It does not
retain the complete memory-bounded protection. The historical
[E31 rollback](../scripts/node/reference/baseline-20260928-e31.env) restores the previous
connector, transfer config and cache namespace while retaining the measured E31 engine
recipe. It therefore restores cache management without changing the API or model settings.

The performance reference is one complete suite of upstream Rigmark with the reference
flags (54 requests, n = 1), measured on one load together with a same-load E29-equivalent
arm. By owner decision the promotion used four-rank launcher-command parity with that
measured load instead of a separate reproduction run; the
[promotion record](historical_benchmarks/baselines/2026-09-28-e31/promotion.json) records the
parity and live identity checks.

The [previous E29 reference](historical_benchmarks/baselines/2026-09-25-e29/baseline.json)
remains frozen at three suites / 162 requests, measured with the earlier Rigmark settings.
Its complete return is
[`baseline-20260925-e29.env`](../scripts/node/reference/baseline-20260925-e29.env). The
[E28b reference](historical_benchmarks/baselines/2026-09-25-e28b/baseline.json)
and its complete return [`baseline-20260925-e28b.env`](../scripts/node/reference/baseline-20260925-e28b.env),
the [E27c reference](historical_benchmarks/baselines/2026-09-25-e27c/baseline.json)
and its complete return [`baseline-20260925-e27c.env`](../scripts/node/reference/baseline-20260925-e27c.env),
the [E27 reference](historical_benchmarks/baselines/2026-09-24-e27/baseline.json)
and its complete return [`baseline-20260924-e27.env`](../scripts/node/reference/baseline-20260924-e27.env),
the [E22b reference](historical_benchmarks/baselines/2026-09-23-e22b/baseline.json)
and its complete return [`baseline-20260924-e22b.env`](../scripts/node/reference/baseline-20260924-e22b.env),
the [E21 reference](historical_benchmarks/baselines/2026-09-23-e21/baseline.json)
and its complete return [`baseline-20260923-e21.env`](../scripts/node/reference/baseline-20260923-e21.env),
the [E03 reference](historical_benchmarks/baselines/2026-09-19-e03/baseline.json)
and its complete return [`baseline-20260919-e03.env`](../scripts/node/reference/baseline-20260919-e03.env)
also remain available.
The [earlier September 19 base](historical_benchmarks/baselines/2026-09-19/baseline.json)
remains frozen at two accepted suites / 108 requests. Its complete return is
[`baseline-20260919.env`](../scripts/node/reference/baseline-20260919.env).
[September 18](historical_benchmarks/baselines/2026-09-18/baseline.json) and
[September 11](historical_benchmarks/baselines/2026-09-11/baseline.json) remain independent
historical references. Their rollback assets are retained under `scripts/node/reference/`;
September 12 filenames identify the later capture of the September 11 recipe.

## Current stack

| Layer | Current component | Purpose and source |
| --- | --- | --- |
| Hardware | four NVIDIA GB10 nodes, verified on ASUS Ascent GX10 | one GPU per TP rank; platform overrides belong in `cluster.env` |
| Network | two-port ConnectX-7 switchless RoCE ring | four direct edges, MTU 9000; see [`fabric.md`](fabric.md) |
| Serving engine | SparkRing/SparkCache R10 SM121 vLLM container pinned by registry digest (`IMAGE`) and content ID (`IMAGE_ID`) | rank 0 exposes the OpenAI-compatible API; ranks 1–3 are headless; the September 18 rollback uses the same image |
| Prefix cache | protected SparkCache connector selected by `scripts/node/experiments/e03/sparkcache-ram-budget/kv-transfer-config.json` (`SPARKCACHE_MODE=on`) | persistent cache in a dedicated protected namespace; 8 MiB disk transfers, a shared 1 GiB transient budget per rank and a 1 GiB admission floor; connector and corrected encoder are included in `third_party/sparkcache/` and pinned by SHA-256 |
| Transport | SIRCL bundle and runtime under `SIRCL_DIR`, started through its entrypoint; it carries the TP4 all-reduce and the single-rail sync prefill exchange | included in `third_party/sparkring-sircl/` and pinned by `scripts/node/sircl/SHA256SUMS`; the container runs with `--no-healthcheck` because that entrypoint never writes the image's readiness marker |
| Engine overrides | tracked vLLM modules under `scripts/node/overrides/` and `scripts/node/experiments/e03/`, including `prefill-cache-trim/`, `prefill-step-cap/` and `bounded-admission/` | cache allocation, allocator trim, GLM model/indexer, hybrid KDA scratch, per-call mHC sharding, E21/E22b conversions, the E29 scheduler with its 6,912-token cap, and rank-0 API admission |
| Target model | pinned `zai-org/GLM-5.3-Flash` FP8 snapshot | immutable file list and hashes under `scripts/node/model-manifests/` |
| Drafter | pinned `incoai/GLM-5.3-Flash-DFlash2` | fused speculative draft; non-commercial upstream terms apply |
| Expert kernels | vLLM Triton FP8 MoE with the GB10-specific JSON in `scripts/node/moe-configs/` | loads the selected platform configuration for the Triton backend |
| Speculation policy | `scripts/node/experiments/e03/draft-budget/adaptive_k_scheduler.py` in `batch-uniform` mode | adaptive length capped by the same `SchedulerOutput` draft budget used by the worker |
| Prefill cadence | native engine argument `--prefill-schedule-interval 8` (E27) with the E27c scheduler `scripts/node/experiments/e03/queued-cadence/scheduler.py` | while requests decode, long prefills run on one step in eight so running requests keep generating; short prefills pass at once, and the cadence stays on while requests are queued |
| Sparse attention | `scripts/node/sparse_attn_indexer_kpool_sm121.py` (September 11 recipe only) | SM121 K-pool compatibility patch bind-mounted over the image module when `SPARKCACHE_MODE=off`; the R10 image ships its own |
| Host tier | pinned kernel/packages and `iommu.passthrough=1` | verified host baseline; owned by `scripts/node/bootstrap/` and `scripts/node/host/` |
| Collectives | host-preloaded patched NCCL | prevents uncabled tree connections and uses the physical ring |

The annotated configuration owns model and container names, paths, revisions,
scheduler limits, and engine arguments. The measured memory and execution choices
are explained below.

## Runtime wiring and preflight

`scripts/deploy.sh` installs the launcher, controller, flusher, model helpers and
manifests, Python patches, and MoE JSONs. `scripts/deploy-host.sh` owns host scripts and
`/etc` material. [`scripts/node/README.md`](../scripts/node/README.md) maps every repository location
to its node destination.

Before a rank starts, the launcher requires:

- the target model `config.json`;
- the DFlash2 `model.safetensors`;
- the patched `libnccl.so.2`;
- an active IPv4-mapped RoCEv2 GID on every configured HCA, at the explicit index or
  selected automatically when `NCCL_IB_GID_INDEX=-1`;
- the sparse-attention indexer patch (September 11 recipe, `SPARKCACHE_MODE=off`);
- with `SPARKCACHE_MODE=on`: the local image content ID equal to `IMAGE_ID`, the
  SparkCache config, connector, and selected encoder matching their pinned SHA-256, the SIRCL manifest
  verified inside `SIRCL_DIR`, no `--kv-transfer-config` or connector reference in
  `EXTRA_VLLM_ARGS`, and each selected connector/encoder mounted exactly once by
  `EXTRA_DOCKER_ENV`;
- every bind-mount source named by `EXTRA_DOCKER_ENV`;
- the configured image locally and the rank's management address on its selected
  management interface.

It also validates four-rank topology, speculative-token and scheduling flags, and
rank-local hardware overrides. Missing mount sources are fatal because Docker would
otherwise create a directory at the source path and start with a broken target.
`scripts/verify-node.sh`, rather than the launcher, checks the pinned model revision
marker and file manifest.

The Current template selects the GID automatically (`NCCL_IB_GID_INDEX=-1`) on all ranks,
as both historical measured recipes did. Automatic selection serves hosts whose active HCA ports expose their
addressed IPv4 RoCEv2 mappings at different indexes; it does not relax the `AF_INET` or
RoCEv2 constraints.

Runtime scratch and compile caches live outside the model directory. The page-cache
flusher runs only while the weights load and is stopped after `/health` reaches 200.

## Hybrid KDA projections and memory

The accepted [E03 mHC path](../scripts/node/experiments/e03/README.md) runs for pure
eager prefills of exactly 6,912 BF16 target rows under TP4/DCP1. It leaves the first
mHC pre operation full-sized, then assigns 1,728 contiguous rows per rank, using
per-call reduce-scatter/all-gather. Attention/FFN consumers and all DFlash2 auxiliary
captures retain full ordered outputs. Decode, mixed batches, shorter tails and CUDA
graphs use the ordinary path. The feature is enabled by `SPARK_MHC_PREFILL_SHARD=1`.

Its source files retain their measured `experiments/e03/` paths and hashes. The separate
SparkCache namespace is required because sharding changes collective reduction order;
replay views preserve that same computation and encoded cache format. The previous
base model and cache remain available through the complete September 19 rollback.

The hybrid source modules keep the exact bytes measured for this baseline, including
historical `E20` names and preparation-era docstrings. Their activation is defined by
the mounts and model hook in the current recipe; the old scratch docstring's statement
that it is not installed no longer describes this configuration. `CREDITS.md` records
public provenance, and the dated baseline pins the source hashes.

After the original checkpoint loads, the model converts exactly 34 fused KDA input
projections from BF16 weights to group-128 INT8 storage with BF16 activations. Each
TP-local projection has shape `[6288, 4096]`; native Marlin packing pads it to
`[6400, 4096]`. The checkpoint files and other projections keep their existing
precision. Shapes, device, dtype, method, and absence of bias are checked before
any projection is converted.

Inputs with fewer than 2,048 flattened tokens use native Marlin W8A16. At 2,048
tokens or more, the same quantized weights are dequantized into one BF16 scratch
matrix shared serially by all 34 projections, then passed to native BF16 linear
execution. The scratch and inverse permutation together occupy about **49.13 MiB
per rank**. Both paths therefore use the same quantized weights; prefill does not
switch back to the original checkpoint weights. CPU integration tests are in
`scripts/tests/test-kda-hybrid.py`; CUDA numerics and graph execution require the
pinned engine and GPU environment.

E21 extends the same mechanism to 67 more modules per rank, loaded from
[`experiments/e03/bf16-residue/`](../scripts/node/experiments/e03/bf16-residue/README.md)
and enabled by `VLLM_E21_BF16_RESIDUE_W8A16=1`: the 34 KDA output projections
`[4096, 2048]`, and the 11 MLA output `[4096, 4096]`, fused QKV-A `[2048, 4096]` and
Q-B `[4096, 1536]` projections. The MLA projections are block FP8 in the checkpoint;
the vendor model code dequantizes them to BF16 on load, so the 8-bit step is a second
quantization, checked before the window at 0.65–0.71% relative weight error. The KDA
f/g gates, MLA `kv_b_proj`, the sparse-attention indexer, `lm_head` and the drafter
are unchanged. Large prefills reuse the shared BF16 scratch, which grows by 79,691,776
bytes per rank. Each rank logs `E21_BF16_RESIDUE_W8A16_READY` with the converted
module count. Converting about 1.18 GiB of BF16 weights per rank raised the sampled
minimum available memory on every rank.

E22b applies the same INT8 group-128 format to the DFlash2 speculative drafter, from
[`experiments/e03/drafter-w8a16/`](../scripts/node/experiments/e03/drafter-w8a16/README.md)
with `VLLM_E22_DRAFTER_W8A16=1` and `VLLM_E22_CONTEXT_KV_W8A16=0`. In each of the five
drafter layers it converts the QKV `[1536, 4096]`, output `[4096, 1024]`, gate/up
`[6144, 4096]` and down `[4096, 3072]` projections and the two grouped-convolution kernel
projections `[1024, 4096]`: 30 modules, always native Marlin because they see only the
query rows of each draft step, with no scratch. The drafter's fused context K/V
projection stays in BF16: converting it, in the first E22 candidate, cost about 3% of 8K
cold prefill in matched probes. `fc`, `lm_head`, `embed_tokens` and the candidate
selector are unchanged. The mounted `qwen3_dflash2.py` is the image's file byte for byte
plus one appended, flag-gated load hook. Each rank logs `E22_DRAFTER_W8A16_READY` with 30
modules; drafter weights drop from 540 MiB to 274 MiB per rank. The target model verifies
every drafted token, so the drafter's precision affects acceptance, which stayed close to
E21, rather than the generated tokens.

E27 adds the image's native `--prefill-schedule-interval 8` and changes scheduling only.

- **How it works.** While requests are in decode, the engine admits prefill work on one
  step in eight and runs decode-only steps in between. Those steps are eligible for the
  full decode CUDA graphs.
- **Why.** Without it, a step that carries a long prompt's prefill chunk (up to 8,192
  tokens, about 2.5 s) is the only step in which running requests advance. Another
  client's 32K cold prompt then holds them at about 1.5 tokens per second for 14 s.
- **Effect.** With the cadence, running requests keep 6–11 tokens per second during that
  prefill. The arriving request's first token comes 15–37% later.
- **Limits.**
  - A prefill with no running decode is not deferred.
  - Once requests are left waiting in the queue, the scheduler stops deferring until the
    queue drains. Several long prompts that arrive together therefore behave largely as
    before; the last one decodes faster but starts a few seconds later.
  - The longest pause of a running stream does not shrink, because a prefill step still
    carries a full chunk.
  - Weights and cache namespace are unchanged, but different batching can change answers
    slightly.
- **Cost in the standard suite.** C4, whose streams start together, paid +35% per-stream
  TTFT and -3% throughput against a same-day control. E27c removes that cost.

E27c mounts the image's `vllm/v1/core/sched/scheduler.py` with one reviewed patch
(`scripts/node/experiments/e03/queued-cadence/`) and sets two flags:

- **`VLLM_E27B_SHORT_PREFILL_TOKENS=2048`.** On a step where the cadence defers prefill
  work, a prefill with fewer than 2,048 remaining uncached tokens is still admitted, up to
  2,048 such tokens per step, and a deferred long request no longer blocks shorter ones
  queued behind it. C4 per-stream TTFT returns to 0.525 s, and a 1K prompt arriving next to
  three running agents reaches its first token in 0.91 s instead of 1.98 s. This part was
  measured alone as E27b.
- **`VLLM_E27C_CADENCE_WHEN_QUEUED=1`.** The cadence keeps deferring long prefills while
  requests are queued, instead of switching itself off. With four 32K prompts arriving
  together, the first stream decodes at about 12 tok/s instead of 6 while the others
  prefill; the last prompt starts about 7% later.
- **Limits.** Each cadence cycle still holds one prefill step of up to 8,192 tokens (about
  2.6 s), which caps a running stream near 12 tok/s during a long prefill. The cadence
  spreads the prefill cost so that streams keep moving; the net delay of a running agent
  is about the same as without it. With both flags unset the mounted file behaves as the
  image's scheduler.
- **Signature.** Rank 0 logs `E27C_CADENCE_WHEN_QUEUED_READY interval=8` and
  `E27B_SHORT_PREFILL_READY tokens=2048 interval=8` at startup.

E28b uses the full training block of the DFlash2 drafter for a single request:

- **Draft length.** `SPEC_TOKENS=7`, the per-batch table `[[1,1,7],[2,6,3]]` and
  `VLLM_ADAPTIVE_K_HI=7`: one request drafts up to seven tokens, batches of 2–6 keep three,
  and the adaptive low state keeps three, so prose still drops to three.
- **Why.** The drafter is trained with blocks of eight positions. With five tokens, code
  still accepted the fifth position in 39–58% of steps; with seven, the sixth and seventh
  are accepted in roughly 30% of code steps and over 90% of structured steps.
- **Effect.** Code decode +8.7% against E27c; structured output reaches 7.8 tokens per step.
  Alone, E28b delayed a request that arrived right after another one ended by about 50 ms,
  and cached replay at 8K–32K by up to 0.1 s. E29 removes most of that, as described below.
- **Graphs.** `--compilation-config={"max_cudagraph_capture_size":72}` keeps the E27c CUDA
  graph set; without it vLLM would capture sizes up to 96.

The E31 performance reference used **16 GiB per rank** (E28b), up from 15 GiB in E27c and earlier. Seven draft
tokens hold about 5% fewer KV tokens per GiB, and the larger pool restores the capacity for
five agents at the full context. The configured per-request context limit remains
262,144 tokens. The measured boot reported **1,365,066 tokens** of pooled KV capacity and a
theoretical 5.21-fold concurrency at that context length. These are engine allocation
reports, not a guarantee that six simultaneous requests can each occupy the full context
window. During the three E28b suites rank 0, which also runs the API server and engine
core, kept at least 2.1 GiB available; `earlyoom` on the nodes acts below 0.5 GiB. The
earlier 16 GiB failure on September 19 ran six parallel agents on a larger memory
footprint.

E29 changes only the boundaries of a request. The
[overlay](../scripts/node/experiments/e03/end-drain/README.md) mounts two files, each with an
additions-only patch:

- **Why.** With asynchronous scheduling the engine queues a request's next step before the
  output of the step in flight has returned. The vendor guard skips that step only when one
  token is missing. With seven drafts, a request limited by `max_tokens` usually finished
  inside a step that could produce several tokens, so the queued step verified drafts for a
  finished request and the next request waited behind it.
- **Hold (`VLLM_E29_END_DRAIN=1`, scheduler).** A running request whose committed outputs
  plus pending placeholders reach `max_tokens` is not given another step until that output
  arrives. It resumes if tokens are still missing. Other requests are scheduled normally,
  and structured-output requests keep the vendor pipeline.
- **Coalescing (`VLLM_E29_IDLE_COALESCE_MS=4`, engine core).** When requests arrive at an
  idle engine, the loop keeps taking arrivals for 4 ms after the first before scheduling. In
  a C2/C4 group the arrivals spread over at most 3.17 ms, and without the window the first
  request was prefilled alone in 22 of 24 groups. The window refuses data parallelism.
- **Trace (`VLLM_E29_TRACE=0`).** Arrival, dispatch, hold and resume logging exists for
  diagnosis and stays off.
- **Effect against E28b.** C1/C2/C4 per-stream TTFT −13.2%/−14.7%/−9.8%, code decode +3.2%,
  no metric worse beyond noise. A replay sent right after its cold request still waits for
  the SparkCache connector to publish that request; at 8K–32K it stays 56–82 ms slower than on
  E27c. A request that ends on EOS can still leave one step behind it.
- **Signature.** Rank 0 logs `E29_END_DRAIN_READY trace=0` and
  `E29_IDLE_COALESCE_READY ms=4 trace=0` at startup.

E31 fixes C4 pools that speculative decoding corrupted. The
[overlay](../scripts/node/experiments/e03/e31-indexer/README.md) mounts copies of the pooled
indexer and its C4 kernels:

- **Bug.** The indexer builds one pooled key from every four tokens and keeps each request's
  recent key/gate rows in a tail, so a pool can be completed across steps. The tail had four
  slots indexed by `position % 4`. A DFlash2 verify step writes 1 + 7 consecutive positions,
  including drafts that may be rejected. Their rows overwrote slots that still held committed
  members of the open pool, and after a rejection the next step completed that pool from
  rejected-draft keys. The target verify has no rollback; only the MTP speculator snapshots
  the tail.
- **Fix (`VLLM_GLM53_KPOOL_TAIL_RING`, on by default).** The tail becomes a ring of
  `4 * cdiv(4 + K, 4)` slots (12 for seven drafts) indexed by `position % ring`; the pool
  phase stays `position % 4`. No row of one step can reach a committed member of the open
  pool. An offline proof and a GPU leaf test show the legacy tail corrupting pools (200
  wrong pools in the leaf's random schedule) and the ring producing bit-exact pools.
- **Cost.** One suite per arm on one load, against the E29-equivalent arm: code decode
  +0.97%, prose −0.54%, code TTFT +0.42%, prefill within ±3.5%. C1/C2/C4 end-to-end moved
  +7.3%/−6.5%/−6.7%, which is within the noise of three short rounds.
- **Head gate (`VLLM_GLM53_INDEXER_GATE_TC`, off).** The same files can compute the indexer
  head gate as a BF16 tensor-core GEMM with FP32 output. It is exact in products and changed
  no top-512 selection in the leaf test, but its same-load comparison is unresolved: changing
  the gate changed the generated tokens and therefore draft acceptance. It stays off.
- **Runtime switches.** Both switches have in-container flag files
  (`/tmp/glm53-indexer-gate-tc`, `/tmp/glm53-kpool-tail-ring`) for same-load experiments.
  Switch only while running and waiting requests are 0.
- **Cache namespace.** The SparkCache namespace was kept. Pools built by speculative decode
  with the legacy tail could be restored for resumed sessions. It is not established that
  decode-built pools are persisted, so this remains a residual risk rather than a
  confirmed defect.
- **Upstream.** vLLM merged the equivalent fix for its own kpool compressor on September 25,
  2026 (PR #58454). The SparkRing source of the R10 image still has the four-slot tail;
  profiles that use MTP3 are covered by its snapshot/restore, so the defect affects DFlash2.
- **Signature.** Every rank logs `E31_INDEXER_GATE path=fp32` and
  `E31_KPOOL_TAIL_RING ring=12` on its first forward.

The GPU worker retains the allocator probe present during measurement. It reads
cached allocator counters once per second after warmup without synchronizing CUDA
or resetting peaks. External host-memory sampling was also active during the
accepted runs; its overhead was not measured separately. The frozen baseline
identifies the instrumented recipe.

For a bounded sequence-count functionality window on the **E03 base only**, the guarded
[`e03-c5/delta.env`](../scripts/node/experiments/e03-c5/delta.env) changes only
`MAX_NUM_SEQS` from six to five. Its sibling `fallback-max4.env` changes it to four.
Both require the accepted E03 image, replay connector, 262,144-token context and one
15 GiB KV argument before applying. They preserve all other engine and container
arguments. The fallback is the E03 recipe at four sequences; it does not use
the older 12 GiB C4 connector variant. Both pin the E03 cache configuration and refuse
the current default; combine them only with the E03 rollback.

## Adaptive draft length

DFlash2 produces a fused block of draft tokens. The adaptive scheduler verifies either
the low or high length for each request. It tracks whether the low-length prefix was
accepted, folds that Bernoulli signal into a per-request exponential moving average,
and switches state with hysteresis. New requests begin in the high state. Structured
workloads tend to remain high; prose tends to move low.

`SPEC_EXTRA_JSON` captures full CUDA-graph families for both verification sizes. The
MoE JSON mount, scheduler mount, `PYTHONPATH`, and policy variables share
`EXTRA_DOCKER_ENV`; removing one feature must preserve the others. The launcher does
not add the optional `--async-scheduling` CLI flag in the current recipe. In the pinned
vLLM/DFlash path, the custom class still derives from `AsyncScheduler`; it disables its
policy if the engine reports that required path unavailable.

`batch-uniform` remains the Current production mode: the scheduler chooses one k per step,
so every decode step replays a full CUDA graph. In `per-request` mode (September 11) a step that
mixes k=3 and k=5 requests runs the PIECEWISE decode graph; with code acceptance
measured at the 0.58 up-threshold, mixed steps appear as soon as two requests are
batched. The historical [September 18 three-run series](historical_benchmarks/baselines/2026-09-18/baseline.json) measured
C4 +18.2% against September 11 with C2 within noise. That result belongs to the
complete September 18 recipe; it does not isolate the scheduler's contribution.
The September 18 rollback retains `batch-uniform`.

The policy is CPU-testable without vLLM:

```sh
python3 scripts/node/patches/test_adaptive_k_policy.py
```

R10 supplies five draft tokens for one scheduled request and three for batches of two
through six. The current [draft-budget scheduler](../scripts/node/experiments/e03/draft-budget/README.md)
caps verification placeholders to `min(adaptive_k, engine_maximum, effective_draft_budget)`
using the producing `SchedulerOutput`, exactly as the worker resolves its drafts.
It does not infer the budget from the following batch. Policy thresholds, observations,
finished/prefill filters and async transitions are preserved. C1 retains a budget of five.

`VLLM_ADAPTIVE_K_RESPECT_DRAFT_BUDGET=1` and the measured scheduler mount are both
selected by the defaults. The previous scheduler bytes stay intact under `patches/`
for rollback. Startup and aggregate limiting counters are documented in
[operations](operations.md). The original mismatch was also present in the previous
baseline; it is not proof of the cause of any earlier throughput difference.

The implementation is derived from vLLM's Apache-2.0 scheduler interfaces and retains
its SPDX/provenance header. Exceptions in the optimization path log once and fall back
to base scheduling rather than taking down the endpoint.

## SparkCache prefix cache and SIRCL transport

The operational default
[streams synchronous snapshots and restores](../scripts/node/experiments/e03/sparkcache-ram-budget/README.md)
through disk in 8 MiB pieces, with a shared 1 GiB transient budget per rank and a 1 GiB
`MemAvailable` admission floor. It remains separate from the measured E31 performance
recipe and does not change the model's context window.

`SPARKCACHE_MODE=on` selects the Current configuration. The launcher builds `--kv-transfer-config`
from the tracked `scripts/node/experiments/e03/sparkcache-ram-budget/kv-transfer-config.json`
(deployed to `~/tp4/experiments/e03/sparkcache-ram-budget/`), which names the
`SparkContextCacheConnector`, the target and
drafter checkpoint hashes it accepts, a 4,096–262,144-token span, store and restore
enabled with `recompute` on a failed load, and the cache root under the runtime cache
volume. The connector and encoder come from the Apache-2.0
[SparkCache](https://github.com/FujitsuPolycom/sparkcache) project, with this project's
pending-publication change, memory corrections and replay views applied. The SIRCL
bundle and runtime come from the Apache-2.0 [SparkRing](https://github.com/FujitsuPolycom/sparkring)
project. Both are included under `third_party/`, and [third-party payload](third-party.md)
records the provenance of every file. `scripts/deploy.sh` places them at
`SPARKCACHE_CONNECTOR`, `SPARKCACHE_ENCODER`, and `SIRCL_DIR`
(see [`install-from-zero.md`](install-from-zero.md#8-prepare-the-sparkcache-and-sircl-payload)),
where they are verified against `scripts/node/sparkcache/SHA256SUMS` and
`scripts/node/sircl/SHA256SUMS` by the launcher and by `scripts/verify-node.sh`; the
SIRCL per-rank peer/GID files are site data, pinned by the gitignored
`scripts/node/sircl/SHA256SUMS.site` and generated by `scripts/sircl-site-files.sh`. The
lane disables the PCIe and FlashInfer all-reduce paths and vLLM plugins in the container
environment, starts through the SIRCL entrypoint, and turns the image healthcheck off;
`/health` 200 stays the only readiness definition.

The connector releases each completed saver item's references before waiting for
the next item, including after a failed commit. The encoder joins the page header
and payload parts in one allocation. These corrections reduce temporary memory
without changing the encoded bytes or cache format.
[`scripts/prepare-sparkcache.py`](../scripts/prepare-sparkcache.py) reproduces the
included files from the upstream encoder and the connector with the pending-publication
patch. It checks both input and output hashes and retains the original connector as
`spark_context_cache_connector-20260918.py` for rollback. Then run the pinned
[replay preparer](../scripts/node/experiments/e03/replay-views/prepare.py) on the corrected
connector. Its separate output, `spark_context_cache_connector-e03-replay-views.py`,
views already validated snapshot spans instead of copying the full body again. Mutable
per-layer copies, owner lifetime and final stream synchronization remain intact.
Finally, `sparkcache-ram-budget/prepare.py --check` verifies the protected connector and
its complete patch without changing the validated payload. Keep the protected connector
and all three prior versions for the E31, September 19 and September 18 rollbacks.

`SPARKCACHE_ENCODER` and `SPARKCACHE_ENCODER_SHA256` select and pin the encoder;
the connector has corresponding variables. `scripts/node/sparkcache/SHA256SUMS`
and the configuration pins must agree with the prepared files. The selected protected
JSON has its own cache namespace and is pinned by the operational identity. The E31
rollback JSON is the exact measured configuration and retains the E22b namespace recorded
in the frozen reference. Preserve both namespaces for their respective recipes; changing
either spelling changes both the configuration hash and the cache selected by the engine.

SparkCache persists across container restarts. For current native Rigmark measurements,
use a fresh `--comparison-id` for every suite and send no `cache_salt`; Rigmark's prompt
nonce isolates the cache while preserving its cold/replay protocol. Restarting the
container does not empty the persistent cache. Monitor free disk space on the runtime
cache volume.

A variant that changes weight precision or other calculations producing cached state
must use its own `spark_cache_root`. Unchanged checkpoint hashes do not establish cache
compatibility when weights are converted in memory. Keep the original cache for rollback;
update the variant's config hash and manifest together with its separate cache path.

The immediate operational rollback,
[`operational-20260929-sparkcache-protected.env`](../scripts/node/reference/operational-20260929-sparkcache-protected.env),
preserves the bounded SparkCache connector but restores the 16 GiB pool and removes the new
trim, step-cap and admission controls. The historical
[`baseline-20260928-e31.env`](../scripts/node/reference/baseline-20260928-e31.env) restores
the measured E31 recipe with unrestricted cache behavior.
[`baseline-20260925-e29.env`](../scripts/node/reference/baseline-20260925-e29.env)
also restores the E29 indexer.
[`baseline-20260925-e28b.env`](../scripts/node/reference/baseline-20260925-e28b.env)
restores the complete E28b recipe: seven draft tokens and a 16 GiB KV pool without the
E29 scheduler or engine core.
[`baseline-20260925-e27c.env`](../scripts/node/reference/baseline-20260925-e27c.env)
restores the complete E27c recipe: five draft tokens and a 15 GiB KV pool.
[`baseline-20260924-e27.env`](../scripts/node/reference/baseline-20260924-e27.env)
restores the complete E27 recipe: the image's own scheduler.
[`baseline-20260924-e22b.env`](../scripts/node/reference/baseline-20260924-e22b.env)
restores the complete E22b recipe, without the prefill cadence. E27 and E27c keep the E22b
cache namespace because they change no cached state; E28b and E29 keep it as well, and so
does E31 by owner decision.
[`baseline-20260923-e21.env`](../scripts/node/reference/baseline-20260923-e21.env)
restores the complete E21 recipe: the vendor drafter, no E22 module or flags and the E21
cache namespace. [`baseline-20260919-e03.env`](../scripts/node/reference/baseline-20260919-e03.env)
restores the complete E03 recipe: the E03 hook source and cache namespace, without the
E21 module or flag. The older [`baseline-20260919.env`](../scripts/node/reference/baseline-20260919.env)
restores the earlier September 19 model, scheduler, connector and cache namespace,
disables mHC, and retains hybrid KDA and 15 GiB KV. Stop with the serving recipe before
selecting any of them.

The older rollback to September 18 uses the complete
[`baseline-20260918.env`](../scripts/node/reference/baseline-20260918.env) overlay,
with the frozen `model-20260918.py`, `sparkcache-20260918.json`, and prepared original
connector. It restores the 16 GiB KV pool and BF16 KDA projections, removes the hybrid
and GPU-probe mounts, selects the original cache namespace, and leaves
`SPARKCACHE_ENCODER` empty to use the image's original encoder. First stop with the
currently serving delta, then select the September 18 overlay for deploy, up and all
subsequent lifecycle commands. Changing the KV argument alone does not restore the
older recipe. The September 11 recipe remains a separate
historical rollback. See the [recovery table](operations.md#recovery-and-rollback).

## Thinking-off compatibility

The pinned model snapshot changed its upstream chat template. The repository's
`scripts/render_chat_template.py` applies a narrow runtime adapter: when a request uses
`chat_template_kwargs: {"enable_thinking": false}`, the rendered assistant prefix
contains a closed empty `<think></think>` block before normal content, supplying a
closed prefix to request a direct answer.

This behavior is local compatibility code, not a documented native
GLM-5.3-Flash feature. The model's official reasoning controls are
`reasoning_effort: low`, `high`, and `max`. Keep tests for both the unchanged upstream
template and the local adapter in `scripts/tests/test-chat-template.py`; never describe
historical runs as thinking-off unless the generated request and response prove it.

An independent review closed a candidate defect that had been provisionally linked to
this mechanism: a code-benchmark prose/format failure reproduces at a comparable rate on
the same pinned model with zero local configuration, tracks how much the model reasons
before answering rather than which component served it, and is not attributable to this
adapter. The comparison used the same code prompt and an 8192-token budget; four
of five external-reference requests exhausted that budget on reasoning. This finding
is specific to that workload and does not establish answer quality for other tasks.
Generated-code quality audits remain separate from performance acceptance.

## Why these customizations remain

- FP8 is the selected lane because it fits the model and configured context on the
  four-node target.
- DFlash2 drafts a block in one pass, avoiding sequential MTP draft steps on this engine.
- Triton FP8 MoE uses the versioned GB10 configuration without changing model weights.
- The prefill cadence keeps running requests generating while another request
  prefills a long prompt, at the cost of that request's first token.
- Adaptive verification selects the low or high length per step from the batched
  requests' history; one length per step keeps decode on full CUDA graphs under
  concurrency.
- Hybrid 8-bit attention projections (KDA input/output and MLA) and the 8-bit drafter
  linears use compact weights for decode and the same weights through
  BF16 linear execution for large prefills, with one shared scratch allocation.
- SparkCache restores long shared prefixes from a persistent cache; its saver and
  encoder corrections reduce avoidable copies and retained payloads.
- SIRCL supplies the single-rail synchronous prefill transport that the R10 image and
  the connector expect on this ring.
- IOMMU passthrough is pinned and checked as part of the host baseline.
- Patched NCCL is structural: the uncabled diagonals make the stock tree connection
  plan unsuitable for this topology.

## Qualification and reproduction

The [current reference](historical_benchmarks/baselines/2026-09-28-e31/baseline.json) is one
complete suite of upstream Rigmark (`alexellis/rigmark` c5a0db0) with the reference flags:
every default plus `reasoning_effort` low, no cache salt, a fresh comparison ID per suite.
It was measured over direct LAN HTTP on one load, together with a same-load arm running the
E29 arithmetic (ring and gate off) and a head-gate arm. Every suite completed 54/54 streams
visibly, with zero measurement/protocol/runtime errors. Both functional gates passed after
the boot. Frozen records up to E29 used 8,192 decode tokens, thinking off and a cache salt,
so their values are not comparable with E31's.

Against the same-load E29-equivalent arm (one suite each):
- code decode +0.97%, prose −0.54%;
- code TTFT +0.42%, prose TTFT −2.86%;
- C1/C2/C4 end-to-end +7.3%/−6.5%/−6.7% (three short rounds each);
- cold prefill −2.5% to +0.7%, cached replay −1.9% to +3.5%.

See the [dated report](benchmarks/baselines/2026-09-28-e31.md) for all 16 metrics, the leaf
tests, the excluded series and the limits.

The [promotion record](historical_benchmarks/baselines/2026-09-28-e31/promotion.json)
records:
- the owner's decision to promote without a separate reproduction run;
- the four-rank launcher-command parity between the encoded default and the measured
  candidate;
- the deployment of the default and the live identity check. Follow the
[coordinated migration](operations.md#migrate-the-accepted-overlay-to-defaults) when a
default must be applied to a stack loaded from an overlay.

The performance figures describe inference, not complete agent tasks. Concurrency
requests have 256-token outputs; long decode allows 4,096 tokens and excludes TTFT. The
context limit is 262,144 tokens with 16 GiB KV per rank. Host memory was not sampled during
the E31 suites. No answer-quality audit is a performance gate.

## Security and licensing boundary

The API has no authentication, TLS, rate limit, or caller isolation. Host networking
also exposes unauthenticated NCCL traffic on private point-to-point links. Run on a
trusted LAN/VPN and add an authenticating proxy before broader exposure.

The deployment account has passwordless sudo, and rank 0 has a passphrase-less SSH
mesh to all ranks including itself. Compromise of those accounts is compromise of the
cluster. No credentials belong in the repository; site values belong only in ignored
local files.

Model, drafter, container, derived vLLM files, NCCL, the switchless overlay, the
SparkCache connector/encoder and the SIRCL payload keep their upstream terms.
[`CREDITS.md`](../CREDITS.md) records attribution, the known license uncertainty around
the overlay, and why the connector, encoder, and SIRCL files are pinned by hash rather than
redistributed.
