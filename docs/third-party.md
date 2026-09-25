# Third-party payload: provenance and project changes

The current recipe needs two third-party components: the SparkCache prefix-cache
connector and encoder, and the SIRCL transport bundle and runtime. Both come from public
repositories under the Apache License 2.0. This checkout includes the files the recipe
mounts, with their licenses and notices, so an installation does not depend on those
repositories remaining available:

| Upstream | Included in | License |
| --- | --- | --- |
| [FujitsuPolycom/sparkcache](https://github.com/FujitsuPolycom/sparkcache) | [`third_party/sparkcache/`](../third_party/sparkcache/README.md): connector and hybrid page encoder | Apache-2.0, [`LICENSE`](../third_party/sparkcache/LICENSE) |
| [FujitsuPolycom/sparkring](https://github.com/FujitsuPolycom/sparkring) | [`third_party/sparkring-sircl/`](../third_party/sparkring-sircl/README.md): SIRCL vLLM integration, native library and serving runtime | Apache-2.0, [`LICENSE`](../third_party/sparkring-sircl/LICENSE), [`NOTICE`](../third_party/sparkring-sircl/NOTICE) and [`THIRD_PARTY_NOTICES.md`](../third_party/sparkring-sircl/THIRD_PARTY_NOTICES.md) |
| SparkRing R10 image `ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:0d4029b3…` | Not included: pulled by digest | Upstream component terms |

The tables below map every pinned file to its source. Each row says whether the file is
**upstream** (unchanged bytes), **modified** by this project, or **ours**.
[`scripts/tests/test-third-party-payload.py`](../scripts/tests/test-third-party-payload.py)
checks every included file against its pin, and the project's patches against the upstream
bytes.

## SparkCache connector and encoder

Pins: [`scripts/node/sparkcache/SHA256SUMS`](../scripts/node/sparkcache/SHA256SUMS),
[`scripts/prepare-sparkcache.py`](../scripts/prepare-sparkcache.py) and `cluster.env.example`.
Upstream base: SparkCache commit
[`66057174`](https://github.com/FujitsuPolycom/sparkcache/tree/66057174301a4759ca3a45207ea41016689449cb).
The R10 image installs that commit's `sparkcache/spark_context_cache_connector.py`
(`394775d48e35…`) and `sparkcache/spark_context_cache_hybrid.py` (`f02e67036f0a…`), byte
for byte.

| File in `third_party/sparkcache/` | SHA-256 | Source | Status |
| --- | --- | --- | --- |
| `spark_context_cache_connector-e03-replay-views.py` | `5893f8747aa0…` | Upstream connector plus patches 01, 02 and 03; current connector | **Modified** |
| `spark_context_cache_hybrid.py` | `11a2db855306…` | Upstream encoder plus patch 04; current encoder | **Modified** |
| `spark_context_cache_connector.py` | `23c1e05cc3bb…` | Upstream connector plus patches 01 and 02; September 19 rollback | **Modified** |
| `spark_context_cache_connector-20260918.py` | `a0bedc1c33a3…` | Upstream connector plus patch 01; September 18 rollback | **Modified** |

## SIRCL bundle and runtime

Pins: [`scripts/node/sircl/SHA256SUMS`](../scripts/node/sircl/SHA256SUMS). The native
library's source is SparkRing commit
[`b358a818`](https://github.com/FujitsuPolycom/sparkring/tree/b358a818786d8506086aaaabb9afe464fa2ccb49).

| File in `third_party/sparkring-sircl/` | Source | Status |
| --- | --- | --- |
| `bundle/sitecustomize.py`, `spark_collective_audit.py`, `spark_cudagraph_bucket_contract.py`, `spark_cudagraph_replay_timing.py`, `spark_dcp_collective_audit.py`, `spark_graph_status_reporter.py`, `spark_persistent_output_ring.py`, `spark_tp4_backend.py`, `spark_tp4_capability.py`, `spark_tp4_health_gate.py`, `spark_tp4_port_namespace.py`, `spark_tp4_query_contract.py`, `spark_tp4_query_row_provider.py`, `spark_tp4_vocab_allgather_backend.py`, `sparkring-overlay-manifest.json` (15 files) | Members of [`runtime/releases/glm53-dflash-sircl-overlay/overlay.tar.gz`](https://github.com/FujitsuPolycom/sparkring/tree/be2f646523e002829a05f9ca19e1fcbb565c319f/runtime/releases/glm53-dflash-sircl-overlay) at SparkRing `be2f6465`, byte-identical | **Upstream** |
| `bundle/libspark_transport_capi.so` | Native library built for CUDA SM121 from `spark_transport/` at SparkRing `b358a818` | **Upstream** binary |
| `bundle/sircl-bundle-manifest.json` | Build record of that library. Its 113 source hashes match SparkRing `b358a818` | **Upstream** |
| `runtime/common.env` | R10 serving environment, with the defaults of SparkRing's `launch-rank.sh` | **Upstream** |
| `runtime/entrypoint.sh` | R10 serving entrypoint plus one line that runs the GID check | **Modified** |
| `runtime/sircl_gid_check.py` | [`scripts/sircl_gid_check.py`](../scripts/sircl_gid_check.py); deployed into the runtime | **Ours** |
| `runtime/rank<N>.env`, `runtime/SHA256SUMS` | Generated for each site by [`scripts/sircl-site-files.sh`](../scripts/sircl-site-files.sh); pinned in the ignored `scripts/node/sircl/SHA256SUMS.site` | **Ours**, site data, never published |

No public SparkRing receipt records the library's exact build hash (`f53c88b4…`); its
sources and toolchain are recorded in the manifest. The
[SIRCL README](../third_party/sparkring-sircl/README.md) explains how to rebuild it.

## What this project changed

- **Pending-publication wait (connector, patch 01).** A replay that finds its prefix still
  being published waits up to `spark_cache_pending_wait_ms` for every rank to confirm it,
  instead of recomputing. Upstream SparkCache has no such option.
- **Memory corrections (connector patch 02, encoder patch 04).** The saver releases each
  completed item's references before waiting for the next item. The encoder joins the page
  header and payload in one allocation. Encoded bytes and cache format are unchanged.
  [`scripts/prepare-sparkcache.py`](../scripts/prepare-sparkcache.py) applies both.
- **Replay views (connector, patch 03).** A restore reads each layer as a view of the
  authenticated snapshot instead of copying it again.
  [`replay-views/prepare.py`](../scripts/node/experiments/e03/replay-views/prepare.py) applies it.
- **GID preflight (SIRCL runtime).** [`scripts/sircl_gid_check.py`](../scripts/sircl_gid_check.py)
  checks that each port's IPv4 RoCEv2 GID matches its peer before serving; the entrypoint
  runs it.

The complete diffs are in [`third_party/sparkcache/patches/`](../third_party/sparkcache/patches/)
and in the [SIRCL README](../third_party/sparkring-sircl/README.md#changes-made-by-this-project).
Everything else in these two components is upstream code.

## Installing the files

[`scripts/deploy.sh`](../scripts/deploy.sh) copies the included files to
`~/tp4/sparkcache/` and `~/tp4/sircl/` on every node, and the launcher verifies them before
Docker starts. The only step left to the operator is the site files: see
[installation step 8](install-from-zero.md#8-prepare-the-sparkcache-and-sircl-payload).
