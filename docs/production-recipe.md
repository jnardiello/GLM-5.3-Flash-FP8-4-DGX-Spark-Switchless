# Production recipe

This page explains the current components and why they are present. Exact values and
one-step rollback comments live in [`cluster.env.example`](../cluster.env.example);
host/software pins live in `scripts/node/bootstrap/versions.env`, model file manifests
in `scripts/node/model-manifests/`, and NCCL pins in `scripts/node/nccl/`.

The rollback-stable F0 runtime overlay and its public artifact manifest live in
`scripts/node/reference/`. They freeze the non-site recipe used by the standing F0
baseline independently of later `cluster.env` defaults. Resolved node topology and the
observed host/runtime inventory remain only in a private mode-0600 archive produced by
`scripts/f0-reference.py`; see [`operations.md`](operations.md#frozen-f0-rollback-reference).

## Current stack

| Layer | Current component | Purpose and source |
| --- | --- | --- |
| Hardware | four NVIDIA GB10 nodes, verified on ASUS Ascent GX10 | one GPU per TP rank; platform overrides belong in `cluster.env` |
| Network | two-port ConnectX-7 switchless RoCE ring | four direct edges, MTU 9000; see [`fabric.md`](fabric.md) |
| Serving engine | pinned SM121 vLLM container referenced by `IMAGE` | rank 0 exposes the OpenAI-compatible API; ranks 1–3 are headless |
| Target model | pinned `zai-org/GLM-5.3-Flash` FP8 snapshot | immutable file list and hashes under `scripts/node/model-manifests/` |
| Drafter | pinned `incoai/GLM-5.3-Flash-DFlash2` | fused speculative draft; non-commercial upstream terms apply |
| Expert kernels | vLLM Triton FP8 MoE with the GB10-specific JSON in `scripts/node/moe-configs/` | loads the selected platform configuration for the Triton backend |
| Speculation policy | `scripts/node/patches/adaptive_k_scheduler.py` | adjusts verification length per request from its own acceptance history |
| Sparse attention | `scripts/node/sparse_attn_indexer_kpool_sm121.py` | SM121 K-pool compatibility patch bind-mounted over the image module |
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
- the sparse-attention indexer patch;
- every bind-mount source named by `EXTRA_DOCKER_ENV`;
- the configured image locally and the rank's management address on its selected
  management interface.

It also validates four-rank topology, speculative-token and scheduling flags, and
rank-local hardware overrides. Missing mount sources are fatal because Docker would
otherwise create a directory at the source path and start with a broken target.
`scripts/verify-node.sh`, rather than the launcher, checks the pinned model revision
marker and file manifest.

The base production template keeps explicit GID index `3`. The frozen serving F0
reference deliberately overrides it with automatic `-1` selection on all ranks; this
is part of F0 identity, not a new template default. Automatic selection is otherwise an opt-in
for hosts whose active HCA ports expose their addressed IPv4 RoCEv2 mappings at
different indexes; it does not relax the `AF_INET` or RoCEv2 constraints.

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

The policy is CPU-testable without vLLM:

```sh
python3 scripts/node/patches/test_adaptive_k_policy.py
```

The implementation is derived from vLLM's Apache-2.0 scheduler interfaces and retains
its SPDX/provenance header. Exceptions in the optimization path log once and fall back
to base scheduling rather than taking down the endpoint.

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
adapter. See [`operations.md`](operations.md#e09-reasoning-diagnosis-attribution-closed-on-2026-09-17).

## Why these customizations remain

- FP8 is the selected lane because it fits the model and configured context on the
  four-node target.
- DFlash2 drafts a block in one pass, avoiding sequential MTP draft steps on this engine.
- Triton FP8 MoE uses the versioned GB10 configuration without changing model weights.
- Adaptive verification selects the low or high length from each request's own history.
- IOMMU passthrough is pinned and checked as part of the host baseline.
- Patched NCCL is structural: the uncabled diagonals make the stock tree connection
  plan unsuitable for this topology.

## Security and licensing boundary

The API has no authentication, TLS, rate limit, or caller isolation. Host networking
also exposes unauthenticated NCCL traffic on private point-to-point links. Run on a
trusted LAN/VPN and add an authenticating proxy before broader exposure.

The deployment account has passwordless sudo, and rank 0 has a passphrase-less SSH
mesh to all ranks including itself. Compromise of those accounts is compromise of the
cluster. No credentials belong in the repository; site values belong only in ignored
local files.

Model, drafter, container, derived vLLM files, NCCL, and the switchless overlay keep
their upstream terms. [`CREDITS.md`](../CREDITS.md) records attribution and the known
license uncertainty around the overlay.
