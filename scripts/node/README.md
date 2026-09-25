# Node assets

`scripts/node/` contains files that are installed on, mounted into, or used to build
artifacts for the four cluster hosts. Nothing in this directory runs merely because
it exists in the repository; deploy, bootstrap, launcher, and configuration choices
select the files explicitly.

The base configuration is the [accepted September 25 E29 recipe](../../docs/historical_benchmarks/baselines/2026-09-25-e29/baseline.json).
The previous E03 reference, the earlier September 19 base, September 18 and September 11 remain historical references with their own
rollback assets; the September 12 filenames below belong to the September 11
reference's later capture.

| Path | Purpose |
| --- | --- |
| `bootstrap/` | pinned host/package versions consumed by bootstrap and verification |
| `etc/` | templates and shared netplan, sysctl, sudoers, iptables, systemd, and GRUB material |
| `host/` | idempotent host control for IOMMU passthrough |
| `model-manifests/` | immutable filename, size, and SHA-256 manifests for supported model snapshots |
| `moe-configs/` | GB10 fused-MoE tuning JSON mounted into vLLM |
| `nccl/` | pinned NCCL build, switchless overlay, shape/checksum record, and atomic installer |
| `overrides/` | retained base vLLM modules, including hybrid KDA execution and the GPU allocator probe; the original model remains for rollback |
| `experiments/e03/` | accepted mHC modules, cache config and draft-budget scheduler, retaining measured paths/hashes; archived preparation overlays and checks |
| `patches/` | frozen previous adaptive scheduler and CPU-only policy tests, retained for rollback |
| `reference/` | complete previous September 19 and September 18 overlays, original GLM model and cache JSON; separate frozen September 11 overlay, launcher/controller, artifact pins and private-archive autostart template |
| `sparkcache/` | previous base cache JSON and manifest for current replay views plus rollback connector/encoder payloads |
| `sircl/` | SHA-256 manifest of the untracked SIRCL bundle/runtime payload |
| `flusher-unconditional.sh` | temporary page-cache flusher used while model weights load |
| `sparse_attn_indexer_kpool_sm121.py` | SM121 sparse-attention patch deployed as `sparse_attn_indexer_kpool.py`; mounted only in the September 11 configuration (`SPARKCACHE_MODE=off`) |
| `ssh-config.example` | optional workstation SSH alias example |
| `tp4-autostart.service.example` | rank-0 unit template that starts all four ranks |

## Where files go

| Repository source | Node destination | Owner |
| --- | --- | --- |
| launcher, controller, flusher, model and NCCL GID helpers | `~/tp4/` and `~/tp4/scripts/` | `scripts/deploy.sh` |
| `scripts/node/patches/*.py` except tests | `~/patches/` | `scripts/deploy.sh` |
| sparse-attention patch | `~/patches/sparse_attn_indexer_kpool.py` | `scripts/deploy.sh` |
| `scripts/node/moe-configs/*.json` | `~/tp4/moe-configs/` | `scripts/deploy.sh` |
| `scripts/node/model-manifests/*.json` | `~/tp4/node/model-manifests/` | `scripts/deploy.sh` |
| `scripts/node/host/*.sh` | `~/tp4/host/` | `scripts/deploy-host.sh` |
| generated `scripts/node/etc/<alias>/40-cx7.yaml` | `/etc/netplan/40-cx7.yaml` | bootstrap/deploy-host |
| generated fabric iptables environment | `/etc/default/tp4-fabric-iptables` | bootstrap/deploy-host |
| shared `scripts/node/etc/common/` files | `/etc/sysctl.d/`, `/etc/sudoers.d/`, `/usr/local/sbin/`, `/etc/systemd/system/` | bootstrap/deploy-host |
| GRUB drop-in | `/etc/default/grub.d/zz-tp4-perf.cfg` | bootstrap/deploy-host and `tp4-iommu.sh` |
| built NCCL library | `$NCCL_DIR/libnccl.so.2` | `scripts/node/nccl/install-nccl.sh` |
| E27c reference overlay selected through `TP4_ENV` (immediate rollback) | `~/tp4/scripts/node/reference/baseline-20260925-e27c.env` | `scripts/deploy.sh` |
| E27 reference overlay selected through `TP4_ENV` | `~/tp4/scripts/node/reference/baseline-20260924-e27.env` | `scripts/deploy.sh` |
| E22b reference overlay selected through `TP4_ENV` | `~/tp4/scripts/node/reference/baseline-20260924-e22b.env` | `scripts/deploy.sh` |
| E21 reference overlay selected through `TP4_ENV` | `~/tp4/scripts/node/reference/baseline-20260923-e21.env` | `scripts/deploy.sh` |
| E03 reference overlay selected through `TP4_ENV` | `~/tp4/scripts/node/reference/baseline-20260919-e03.env` | `scripts/deploy.sh` |
| September 19 reference overlay selected through `TP4_ENV` | `~/tp4/scripts/node/reference/baseline-20260919.env` | `scripts/deploy.sh` |
| September 18 reference overlay selected through `TP4_ENV` | `~/tp4/scripts/node/reference/baseline-20260918.env` | `scripts/deploy.sh` |
| `reference/model-20260918.py` and `reference/sparkcache-20260918.json` | `~/tp4/reference/` | `scripts/deploy.sh` |
| September 11 reference overlay selected through `TP4_ENV` | `~/tp4/scripts/node/reference/f0-20260912.env` | `scripts/deploy.sh` |
| Frozen `reference/tp4ctl-f0-20260912.sh` controller | `~/tp4/tp4ctl-f0-reference` | `scripts/deploy.sh` |
| `scripts/node/experiments/e03/` Python, JSON and source manifests | `~/tp4/experiments/e03/` (same relative layout) | `scripts/deploy.sh` |
| `scripts/node/overrides/**/*.py` | `~/tp4/overrides/…` (same relative layout) | `scripts/deploy.sh` |
| `scripts/node/sparkcache/kv-transfer-config.json` and `SHA256SUMS` | `~/tp4/sparkcache/` | `scripts/deploy.sh` |
| `scripts/node/sircl/SHA256SUMS` and the gitignored per-site `SHA256SUMS.site` | `~/tp4/sircl/` | `scripts/deploy.sh` |
| prepared connector and encoder | `SPARKCACHE_CONNECTOR` and `SPARKCACHE_ENCODER` under `~/tp4/sparkcache/` | operator, using `scripts/prepare-sparkcache.py`; pinned by configuration and manifest |
| preserved original connector for September 18 | `~/tp4/sparkcache/spark_context_cache_connector-20260918.py` | operator; emitted by the same preparer and selected by the September 18 overlay |
| untracked SIRCL bundle and runtime | `~/tp4/sircl/{bundle,runtime}/` | operator, per [`install-from-zero.md`](../../docs/install-from-zero.md#8-place-the-sparkcache-and-sircl-payload); verified by `verify-node.sh` and the launcher |

`scripts/deploy.sh` and `scripts/deploy-host.sh` are additive. They copy and verify
managed content but do not delete stray files or restart containers. The bootstrap
script activates `/etc` state only under `--apply`; it never reboots a node.

## Current engine and cache payload

The current recipe mounts 22 vLLM modules: retained cache allocation, worker/probe,
indexer and hybrid-KDA sources, the E03 model and mHC per-call sharding modules, the
E21 residual-projection module, the E22b drafter override and conversion module, and the
E29 scheduler and engine core from [`experiments/e03/end-drain/`](experiments/e03/end-drain/README.md).
The E29 scheduler is the E27c scheduler from
[`experiments/e03/queued-cadence/`](experiments/e03/queued-cadence/README.md) plus one
additions-only patch; the launcher verifies each directory's `SHA256SUMS` whenever it is mounted. The KDA hook is mounted from
[`experiments/e03/bf16-residue/`](experiments/e03/bf16-residue/README.md), which also
converts 67 KDA output and MLA attention projections per rank to the same INT8 format
when `VLLM_E21_BF16_RESIDUE_W8A16=1`; the launcher verifies that directory's
`SHA256SUMS` before starting.
The drafter files are mounted from [`experiments/e03/drafter-w8a16/`](experiments/e03/drafter-w8a16/README.md):
`qwen3_dflash2.py` is the image's file plus one appended load hook, and with
`VLLM_E22_DRAFTER_W8A16=1` and `VLLM_E22_CONTEXT_KV_W8A16=0` it converts 30 DFlash2
drafter linears to the same INT8 format, keeping the context K/V projection in BF16. The
launcher verifies that directory's `SHA256SUMS` too.
E03 applies only to 6,912-row pure eager prefills under TP4/DCP1, leaving decode and
CUDA graph paths ordinary. All measured source bytes and paths are retained. The hybrid path converts 34 KDA input projections to
group-128 INT8 storage, pads their TP-local `[6288, 4096]` shape to `[6400, 4096]`,
and uses Marlin below 2,048 flattened input tokens. At or above that threshold it
dequantizes the same weights into a shared scratch and runs BF16 linear execution.
The scratch and inverse map occupy about 49.13 MiB per rank. The GPU worker's
allocator probe is retained because it was active in the accepted measurement.

The configured KV pool is 16 GiB per rank (E28b); with seven draft tokens the measured
engine reported 1,365,066 tokens of pooled capacity while retaining the 262,144-token
per-request limit.
Those values do not guarantee six simultaneous full-context sessions. See
[`production-recipe.md`](../../docs/production-recipe.md#hybrid-kda-projections-and-memory)
for the measured configuration and its limits.

The SparkCache connector and encoder remain operator-supplied files. Prepare them
from the pinned original sources in a private staging directory:

```sh
python3 scripts/prepare-sparkcache.py \
  --connector /path/to/original/spark_context_cache_connector.py \
  --encoder /path/to/original/spark_context_cache_hybrid.py \
  --output-dir /path/to/private/prepared-payload
python3 scripts/node/experiments/e03/replay-views/prepare.py \
  --connector /path/to/private/prepared-payload/spark_context_cache_connector.py \
  --output /path/to/private/prepared-payload/spark_context_cache_connector-e03-replay-views.py
```

The preparer verifies both source hashes and resulting hashes. Its connector
correction releases completed saver items through a per-item function scope; its
encoder correction uses one join with identical output bytes. It emits those two files plus
`spark_context_cache_connector-20260918.py`, preserving the original for rollback.
The second command emits the separate current replay views connector; it verifies both
input/output pins and refuses to replace an existing file. Preserve the corrected base
connector for September 19 rollback as well. Stage all outputs on the configured paths; deployment does
not distribute unlicensed operator payload automatically.

`SPARKCACHE_CONNECTOR_SHA256` and `SPARKCACHE_ENCODER_SHA256` pin the selected
modules, and `SPARKCACHE_CONFIG_SHA256` pins the tracked JSON. Keep those values
consistent with `sparkcache/SHA256SUMS`. Preserve the JSON's measured cache namespace
when reproducing the Current recipe: its E22b JSON, which E27 and E27c keep unchanged, is
`experiments/e03/drafter-w8a16/kv-transfer-config-e22b.json`, separate from the E21, E03
and earlier cache computations, and its spelling is part of the hash.

For the immediate return to E27c, use
[`reference/baseline-20260925-e27c.env`](reference/baseline-20260925-e27c.env). It restores
five draft tokens and the 15 GiB KV pool.

For the return to E27, use
[`reference/baseline-20260924-e27.env`](reference/baseline-20260924-e27.env). It also removes
the E27c scheduler mount and its two flags.

For the return to E22b, use
[`reference/baseline-20260924-e22b.env`](reference/baseline-20260924-e22b.env). It also
removes the E27 `--prefill-schedule-interval 8` engine argument.

For the return to E21, use
[`reference/baseline-20260923-e21.env`](reference/baseline-20260923-e21.env). It restores
the vendor drafter and the E21 cache namespace and removes the E22 override, module and
flags.

For the return to E03, use
[`reference/baseline-20260919-e03.env`](reference/baseline-20260919-e03.env). It restores
the E03 hook source and cache namespace and removes the E21 module and flag.

For the earlier September 19 base, use
[`reference/baseline-20260919.env`](reference/baseline-20260919.env). It restores the
original model, scheduler, corrected connector and cache namespace, disables mHC,
and retains hybrid KDA and 15 GiB KV. See the coordinated procedure in operations.

For the older September 18 rollback, use
[`reference/baseline-20260918.env`](reference/baseline-20260918.env) with the same
`TP4_ENV` for deploy and the full coordinated transition. The overlay selects the
frozen model and JSON, the original connector, the image's encoder, the 16 GiB KV
pool, and the original set of six engine overrides. It removes hybrid and GPU-probe
mounts together; a KV-only change is not a complete rollback. The September 11
artifacts remain an independent older recovery path.

## Generated and local files

`cluster.env` is the source for node aliases, management addresses, fabric neighbors,
interface/HCA/GID selection, and renderer. Run:

```sh
./scripts/render-netplan.sh --write
./scripts/render-netplan.sh --check
```

This creates one gitignored netplan and fabric-iptables environment file per rank.
Never hand-edit them. `scripts/node/etc/common/99-tp4-nopasswd` and the rendered autostart unit
are also local and ignored; their `.example` files remain public templates.

## Runtime requirements

The launcher refuses to start a rank until the model, drafter, patched NCCL library,
sparse-attention patch (September 11 configuration), selected image, management address, usable configured
IPv4 RoCEv2 GIDs, and every bind-mount source exist. With `SPARKCACHE_MODE=on` it also
requires the image content ID named by `IMAGE_ID`, the SparkCache config, connector,
and selected encoder at their pinned SHA-256, and a verified SIRCL manifest. This prevents Docker from
silently creating a directory where a missing mount source should have been a file and
keeps an unverified payload from serving.

Passwordless sudo is a runtime dependency: the controller and launcher invoke Docker,
systemd, sysctl, and cache controls through `sudo -n`. The template grants
`NOPASSWD:ALL`; treat access to the deployment account as root-equivalent.

The public API and fabric are unauthenticated. Keep both on trusted private networks.
Installation and security prerequisites are in
[`docs/install-from-zero.md`](../../docs/install-from-zero.md); operational checks and
rollback are in [`docs/operations.md`](../../docs/operations.md).

## Verification

```sh
./scripts/deploy.sh --check
./scripts/deploy-host.sh --check
./scripts/verify-node.sh
./scripts/check.sh
```

The first three inspect deployed nodes and require site configuration. The final
command is fully offline and validates source syntax, manifests, templates, links,
fixtures, the adaptive-k policy, hybrid dispatch contracts, and payload preparation
without SSH, Docker, a GPU, or `cluster.env`. Offline checks do not constitute a new
live deployment of the accepted E29 defaults.
