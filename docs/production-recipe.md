# Production recipe

This page explains the current components and why they are present. Exact values and
one-step rollback comments live in [`cluster.env.example`](../cluster.env.example);
host/software pins live in `scripts/node/bootstrap/versions.env`, model file manifests
in `scripts/node/model-manifests/`, and NCCL pins in `scripts/node/nccl/`.

The **Current** configuration was qualified and promoted on September 18, 2026
([frozen baseline](baseline-f1.json)). It combines the SparkRing/SparkCache R10 image
pinned by registry digest, the SparkCache persistent prefix cache over the SIRCL single-rail
transport, and adaptive-k `batch-uniform`. It is encoded directly in
`cluster.env.example`, one rollback comment per block, so no window overlay is needed
to serve it.

The rollback-stable **Previous** runtime overlay and its public artifact manifest live in
`scripts/node/reference/`. They freeze the non-site recipe used by the Previous
baseline independently of later `cluster.env` defaults. Resolved node topology and the
observed host/runtime inventory remain only in a private mode-0600 archive produced by
`scripts/f0-reference.py`; see [`operations.md`](operations.md#frozen-previous-rollback-reference).

## Current stack

| Layer | Current component | Purpose and source |
| --- | --- | --- |
| Hardware | four NVIDIA GB10 nodes, verified on ASUS Ascent GX10 | one GPU per TP rank; platform overrides belong in `cluster.env` |
| Network | two-port ConnectX-7 switchless RoCE ring | four direct edges, MTU 9000; see [`fabric.md`](fabric.md) |
| Serving engine | SparkRing/SparkCache R10 SM121 vLLM container pinned by registry digest (`IMAGE`) and content ID (`IMAGE_ID`) | rank 0 exposes the OpenAI-compatible API; ranks 1–3 are headless; rollback uses the Previous tagged image |
| Prefix cache | SparkCache connector selected by `scripts/node/sparkcache/kv-transfer-config.json` (`SPARKCACHE_MODE=on`) | persistent context cache under the runtime cache volume; the connector module is operator payload pinned by SHA-256, not tracked |
| Transport | SIRCL single-rail sync-prefill bundle and runtime under `SIRCL_DIR`, started through its entrypoint | operator payload pinned by `scripts/node/sircl/SHA256SUMS`; the container runs with `--no-healthcheck` because that entrypoint never writes the image's readiness marker |
| Engine overrides | six vLLM modules under `scripts/node/overrides/` bind-mounted over the image | KV-cache interface and utilities, worker utilities, and the GLM-5.3 pooled indexer/kpool ops the R10 image needs with SparkCache |
| Target model | pinned `zai-org/GLM-5.3-Flash` FP8 snapshot | immutable file list and hashes under `scripts/node/model-manifests/` |
| Drafter | pinned `incoai/GLM-5.3-Flash-DFlash2` | fused speculative draft; non-commercial upstream terms apply |
| Expert kernels | vLLM Triton FP8 MoE with the GB10-specific JSON in `scripts/node/moe-configs/` | loads the selected platform configuration for the Triton backend |
| Speculation policy | `scripts/node/patches/adaptive_k_scheduler.py` in `batch-uniform` mode | one verification length per step, chosen from the batched requests' acceptance history |
| Sparse attention | `scripts/node/sparse_attn_indexer_kpool_sm121.py` (Previous configuration only) | SM121 K-pool compatibility patch bind-mounted over the image module when `SPARKCACHE_MODE=off`; the R10 image ships its own |
| Host tier | pinned kernel/packages and `iommu.passthrough=1` | verified host baseline; owned by `scripts/node/bootstrap/` and `scripts/node/host/` |
| Collectives | host-preloaded patched NCCL | prevents uncabled tree connections and uses the physical ring |

The configured context, sequence count, KV type and pool, model name, container name,
paths, image tag, revisions, and all engine arguments are intentionally read from the
annotated configuration rather than repeated here.

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
- the sparse-attention indexer patch (Previous configuration, `SPARKCACHE_MODE=off`);
- with `SPARKCACHE_MODE=on`: the local image content ID equal to `IMAGE_ID`, the
  SparkCache config and connector matching their pinned SHA-256, the SIRCL manifest
  verified inside `SIRCL_DIR`, no `--kv-transfer-config` or connector reference in
  `EXTRA_VLLM_ARGS`, and the connector mounted exactly once by `EXTRA_DOCKER_ENV`;
- every bind-mount source named by `EXTRA_DOCKER_ENV`;
- the configured image locally and the rank's management address on its selected
  management interface.

It also validates four-rank topology, speculative-token and scheduling flags, and
rank-local hardware overrides. Missing mount sources are fatal because Docker would
otherwise create a directory at the source path and start with a broken target.
`scripts/verify-node.sh`, rather than the launcher, checks the pinned model revision
marker and file manifest.

The Current template selects the GID automatically (`NCCL_IB_GID_INDEX=-1`) on all ranks,
as the frozen Previous reference overlay did; explicit index `3` is the Previous base rollback named
beside the value. Automatic selection serves hosts whose active HCA ports expose their
addressed IPv4 RoCEv2 mappings at different indexes; it does not relax the `AF_INET` or
RoCEv2 constraints.

Runtime scratch and compile caches live outside the model directory. The page-cache
flusher runs only while the weights load and is stopped after `/health` reaches 200.

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

`batch-uniform` is the Current production mode: the scheduler chooses one k per step,
so every decode step replays a full CUDA graph. In `per-request` mode (Previous) a step that
mixes k=3 and k=5 requests runs the PIECEWISE decode graph; with code acceptance
measured at the 0.58 up-threshold, mixed steps appear as soon as two requests are
batched, which cost C2/C4 on the SparkCache candidate. The [qualified three-run series](#qualification-and-reproduction) measured
C4 +18.2% with C2 within noise; `per-request` remains the rollback named beside
`VLLM_ADAPTIVE_K_MODE` and has not been re-measured on the Previous recipe alone.

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
volume. The connector module and the SIRCL bundle and runtime carry no license and are
not tracked: they are operator payload placed at `SPARKCACHE_CONNECTOR` and `SIRCL_DIR`
(see [`install-from-zero.md`](install-from-zero.md#8-place-the-sparkcache-and-sircl-payload))
and verified against `scripts/node/sparkcache/SHA256SUMS` and
`scripts/node/sircl/SHA256SUMS` by the launcher and by `scripts/verify-node.sh`; the
SIRCL per-rank peer/GID files are site data, pinned by the gitignored
`scripts/node/sircl/SHA256SUMS.site` in the operator checkout. The
lane disables the PCIe and FlashInfer all-reduce paths and vLLM plugins in the container
environment, starts through the SIRCL entrypoint, and turns the image healthcheck off;
`/health` 200 stays the only readiness definition.

Rollback to the Previous configuration is the set of values named in the rollback comments of
`cluster.env.example` (image tag and empty `IMAGE_ID`, `SPARKCACHE_MODE=off`, explicit
GID index, `SPEC_EXTRA_JSON`, `EXTRA_DOCKER_ENV`, `EXTRA_VLLM_ARGS`), then a deploy and a
full restart; see the
[recovery table](operations.md#recovery-and-rollback).

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
- SparkCache restores long shared prefixes from a persistent cache instead of
  recomputing them: replay prefill improved by about 109% at 8K and 49% at 32K
  against Previous; the 64K replay median was about 1.6% lower.
- SIRCL supplies the single-rail synchronous prefill transport that the R10 image and
  the connector expect on this ring.
- IOMMU passthrough is pinned and checked as part of the host baseline.
- Patched NCCL is structural: the uncabled diagonals make the stock tree connection
  plan unsuitable for this topology.

## Qualification and reproduction

**Current** was qualified on September 18, 2026 with three consecutive native Rigmark
runs from the same workstation used for **Previous**, on one loaded process with a
fresh `cache_salt` per run. All 162 requests completed with zero errors, zero output
length caps, and 45/45 native output gates passing. Performance comparisons use the
median of the three native per-run values. The [Current baseline](baseline-f1.json)
and [Previous baseline](baseline-f0.json) preserve the full measurements and method.

The decisive gain was code throughput with four concurrent requests: 84.05 versus
71.08 tok/s, with two-request throughput within measured noise. Concurrent-request
tests use at most 256 output tokens per request; these results do not establish
performance for long parallel generations or complete agent tasks. The complete
configuration was accepted together; the individual contribution of the cache and
transport to the prefill gains was not isolated.

Deployment through IaC then reproduced the configuration without an overlay: the
pinned image and payload were verified on four ranks, `/health` reached 200, and both
post-boot gates passed. One separate reproduction run completed 54/54 requests with
zero errors, zero length caps, and 15/15 native output gates passing. Code decode was
52.25 tok/s and end-to-end code throughput at concurrency 1/2/4 was
37.74/56.53/86.39 tok/s, all within the qualification series' respective ranges.
This single reproduction run does not replace the frozen three-run baseline.

## Security and licensing boundary

The API has no authentication, TLS, rate limit, or caller isolation. Host networking
also exposes unauthenticated NCCL traffic on private point-to-point links. Run on a
trusted LAN/VPN and add an authenticating proxy before broader exposure.

The deployment account has passwordless sudo, and rank 0 has a passphrase-less SSH
mesh to all ranks including itself. Compromise of those accounts is compromise of the
cluster. No credentials belong in the repository; site values belong only in ignored
local files.

Model, drafter, container, derived vLLM files, NCCL, the switchless overlay, the
SparkCache connector and the SIRCL payload keep their upstream terms.
[`CREDITS.md`](../CREDITS.md) records attribution, the known license uncertainty around
the overlay, and why the connector and SIRCL files are pinned by hash rather than
redistributed.
