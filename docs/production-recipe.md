# Production recipe

This page explains the current components and why they are present. Exact values and
one-step rollback comments live in [`cluster.env.example`](../cluster.env.example);
host/software pins live in `scripts/node/bootstrap/versions.env`, model file manifests
in `scripts/node/model-manifests/`, and NCCL pins in `scripts/node/nccl/`.

The **Current** configuration is the [September 19, 2026 baseline](historical_benchmarks/baselines/2026-09-19/baseline.json).
It combines the digest-pinned SparkRing/SparkCache R10 image, SIRCL single-rail
transport, adaptive-k `batch-uniform`, hybrid KDA projections, cache-memory
corrections, and a 15 GiB KV pool per rank. The base recipe is encoded directly in
`cluster.env.example`, without an experiment overlay. The accepted performance
record contains two native Rigmark runs. A separate
[one-run IaC reproduction](historical_benchmarks/baselines/2026-09-19/reproduction.json) records deployment verification
and performance after redeploying the published defaults.

The [September 18](historical_benchmarks/baselines/2026-09-18/baseline.json) and [September 11](historical_benchmarks/baselines/2026-09-11/baseline.json) measurements
remain unchanged historical references. The complete September 18 rollback is
[`scripts/node/reference/baseline-20260918.env`](../scripts/node/reference/baseline-20260918.env).
The older September 11 recipe retains its own frozen launcher, controller, and
artifact manifest under `scripts/node/reference/`; filenames dated September 12
identify when those rollback artifacts were captured, not a new performance run.
Resolved site topology and host inventory remain private; see
[`operations.md`](operations.md#frozen-previous-rollback-reference).

## Current stack

| Layer | Current component | Purpose and source |
| --- | --- | --- |
| Hardware | four NVIDIA GB10 nodes, verified on ASUS Ascent GX10 | one GPU per TP rank; platform overrides belong in `cluster.env` |
| Network | two-port ConnectX-7 switchless RoCE ring | four direct edges, MTU 9000; see [`fabric.md`](fabric.md) |
| Serving engine | SparkRing/SparkCache R10 SM121 vLLM container pinned by registry digest (`IMAGE`) and content ID (`IMAGE_ID`) | rank 0 exposes the OpenAI-compatible API; ranks 1–3 are headless; the September 18 rollback uses the same image |
| Prefix cache | SparkCache connector selected by `scripts/node/sparkcache/kv-transfer-config.json` (`SPARKCACHE_MODE=on`) | persistent context cache in a namespace dedicated to the quantized projections; corrected connector and encoder are operator payload pinned by SHA-256 |
| Transport | SIRCL single-rail sync-prefill bundle and runtime under `SIRCL_DIR`, started through its entrypoint | operator payload pinned by `scripts/node/sircl/SHA256SUMS`; the container runs with `--no-healthcheck` because that entrypoint never writes the image's readiness marker |
| Engine overrides | nine vLLM modules under `scripts/node/overrides/` bind-mounted over the image | KV-cache interface/utilities, worker utilities, instrumented GPU worker, GLM model, pooled indexer/kpool ops, KDA quantization, and shared BF16 scratch |
| Target model | pinned `zai-org/GLM-5.3-Flash` FP8 snapshot | immutable file list and hashes under `scripts/node/model-manifests/` |
| Drafter | pinned `incoai/GLM-5.3-Flash-DFlash2` | fused speculative draft; non-commercial upstream terms apply |
| Expert kernels | vLLM Triton FP8 MoE with the GB10-specific JSON in `scripts/node/moe-configs/` | loads the selected platform configuration for the Triton backend |
| Speculation policy | `scripts/node/patches/adaptive_k_scheduler.py` in `batch-uniform` mode | one verification length per step, chosen from the batched requests' acceptance history |
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

The KV pool is **15 GiB per rank**, compared with 16 GiB in the September 18 and
September 11 references. The configured per-request context limit remains
262,144 tokens. The measured boot reported **1,344,328 tokens** of pooled KV
capacity and a theoretical 5.13-fold concurrency at that context length. These
are engine allocation reports, not a guarantee that six simultaneous requests
can each occupy the full context window.

The GPU worker retains the allocator probe present during measurement. It reads
cached allocator counters once per second after warmup without synchronizing CUDA
or resetting peaks. External host-memory sampling was also active during the
accepted runs; its overhead was not measured separately. The frozen baseline
identifies the instrumented recipe.

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

The implementation is derived from vLLM's Apache-2.0 scheduler interfaces and retains
its SPDX/provenance header. Exceptions in the optimization path log once and fall back
to base scheduling rather than taking down the endpoint.

## SparkCache prefix cache and SIRCL transport

`SPARKCACHE_MODE=on` selects the Current configuration. The launcher builds `--kv-transfer-config`
from the tracked `scripts/node/sparkcache/kv-transfer-config.json` (deployed to
`~/tp4/sparkcache/`), which names the `SparkContextCacheConnector`, the target and
drafter checkpoint hashes it accepts, a 4,096–262,144-token span, store and restore
enabled with `recompute` on a failed load, and the cache root under the runtime cache
volume. The connector, encoder, and SIRCL bundle/runtime are untracked operator
payload; the archived sources did not include a license notice. Their locations are
`SPARKCACHE_CONNECTOR`, `SPARKCACHE_ENCODER`, and `SIRCL_DIR`
(see [`install-from-zero.md`](install-from-zero.md#8-place-the-sparkcache-and-sircl-payload))
and verified against `scripts/node/sparkcache/SHA256SUMS` and
`scripts/node/sircl/SHA256SUMS` by the launcher and by `scripts/verify-node.sh`; the
SIRCL per-rank peer/GID files are site data, pinned by the gitignored
`scripts/node/sircl/SHA256SUMS.site` in the operator checkout. The
lane disables the PCIe and FlashInfer all-reduce paths and vLLM plugins in the container
environment, starts through the SIRCL entrypoint, and turns the image healthcheck off;
`/health` 200 stays the only readiness definition.

The connector releases each completed saver item's references before waiting for
the next item, including after a failed commit. The encoder joins the page header
and payload parts in one allocation. These corrections reduce temporary memory
without changing the encoded bytes or cache format. Use
[`scripts/prepare-sparkcache.py`](../scripts/prepare-sparkcache.py) with the original
operator-supplied connector and encoder to reproduce the selected files. It checks
both input and output hashes and retains the original connector as
`spark_context_cache_connector-20260918.py` for rollback. It does not download or
redistribute the payload.

`SPARKCACHE_ENCODER` and `SPARKCACHE_ENCODER_SHA256` select and pin the encoder;
the connector has corresponding variables. `scripts/node/sparkcache/SHA256SUMS`
and the configuration pins must agree with the prepared files. The tracked JSON
is the exact measured configuration, including its quantized-weight cache namespace;
its digest is recorded in the September 19 baseline. Preserve that namespace when
reproducing this recipe. Changing its spelling changes both the configuration hash
and the cache selected by the engine.

SparkCache persists across container restarts. For cold-prefill measurements, use a
fresh `cache_salt` per run and keep it unchanged for the cold/replay pairs within that
run. Confirm that the client forwards it to prefill requests; restarting the container
does not empty the persistent cache. Monitor free disk space on the runtime cache volume.

A variant that changes weight precision or other calculations producing cached state
must use its own `spark_cache_root`. Unchanged checkpoint hashes do not establish cache
compatibility when weights are converted in memory. Keep the original cache for rollback;
update the variant's config hash and manifest together with its separate cache path.

Rollback to September 18 uses the complete
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
- Adaptive verification selects the low or high length per step from the batched
  requests' history; one length per step keeps decode on full CUDA graphs under
  concurrency.
- Hybrid KDA execution uses compact weights for decode and the same weights through
  BF16 linear execution for large prefills, with one shared scratch allocation.
- SparkCache restores long shared prefixes from a persistent cache; its saver and
  encoder corrections reduce avoidable copies and retained payloads.
- SIRCL supplies the single-rail synchronous prefill transport that the R10 image and
  the connector expect on this ring.
- IOMMU passthrough is pinned and checked as part of the host baseline.
- Patched NCCL is structural: the uncabled diagonals make the stock tree connection
  plan unsuitable for this topology.

## Qualification and reproduction

The [September 19 baseline](historical_benchmarks/baselines/2026-09-19/baseline.json) contains **two complete native
Rigmark runs and 108 requests**, with the same loaded processes retained on all four
ranks and a fresh `cache_salt` per run. The operator accepted that two-run series;
a third run was excluded in full because competing traffic was reported. It is
not part of the measurements, and no replacement run is included. Each reported
median is the midpoint of the two native per-run values, not a three-run median
or a pooling of request-level observations.

The accepted runs recorded zero native validation errors across 108 requests,
zero runtime errors across two runs, 108/108 completed streams and visible
responses, and 30/30 native decode output gates passing. Finish reasons were
30 `stop` and 78 `length`; the latter came from bounded prefill and concurrency
requests. Code decode hit its output cap in **0/10** requests. All 36 prefill
requests matched their requested token counts. Both operational post-boot gates
passed within two minutes of `/health` 200.

Code decode was 53.8905 tok/s; aggregate end-to-end code throughput at concurrency
1/2/4 was 40.300/57.252/87.527 tok/s. Full prefill, TTFT, prose and historical comparisons are in the
[benchmark reports](benchmarks/README.md); the project README shows only the current baseline. Concurrency requests
use a 256-token output cap, so these measurements do not establish performance for
long parallel generations or complete agent tasks. The full instrumented recipe
was accepted together; individual kernel, cache-fix, and memory-budget contributions
are not isolated. Two runs provide less repeatability evidence than the historical
three-run references, and the KV pool is smaller.

The published defaults were subsequently deployed through repository IaC, with
`TP4_ENV` unset and a coordinated four-rank restart. The separate
[reproduction record](historical_benchmarks/baselines/2026-09-19/reproduction.json) reports **one native Rigmark run
and 54 requests**, compared with the unchanged two-run baseline medians. Node checks
recorded 166 PASS, 0 FAIL, 2 WARN and 7 SKIP. Operational identity passed before and
after the benchmark; all four containers remained running without restarts. Both
post-boot gates passed within 3.877 seconds of `/health` 200. The run had zero native
validation errors, 54/54 completed streams and visible responses, and 15/15 native
decode output gates passing.

Performance changes were mixed: C1 aggregate throughput was 39.379 versus 40.300 tok/s
(-2.29%), and C4 was 83.058 versus 87.527 tok/s (-5.11%). Code decode and C2 throughput
increased; all 16 comparisons are in the separate record. A single run cannot
establish repeatability, statistical parity, or the cause of these differences.
The same GPU allocator and external host memory probes were active; their overhead
remains unisolated. Images, weights and operator payloads were already available on
the prepared hosts, so this verifies live IaC redeployment rather than a fresh
bare-metal installation. The frozen baseline and its memory-capacity limits remain
unchanged. The September 18 reference's reproduction remains historical evidence
for that older recipe.

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
