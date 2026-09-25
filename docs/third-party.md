# Third-party payload: provenance and project changes

The current recipe needs two components that this checkout does not contain yet: the
SparkCache prefix-cache connector and encoder, and the SIRCL transport bundle and
runtime. Both come from public repositories under the Apache License 2.0:

| Upstream | Contents used here | License |
| --- | --- | --- |
| [FujitsuPolycom/sparkcache](https://github.com/FujitsuPolycom/sparkcache) | Prefix-cache connector (`sparkcache.spark_context_cache_connector`) and hybrid page encoder (`sparkcache.spark_context_cache_hybrid`) | [Apache-2.0](https://github.com/FujitsuPolycom/sparkcache/blob/main/LICENSE) |
| [FujitsuPolycom/sparkring](https://github.com/FujitsuPolycom/sparkring) | SIRCL transport: the vLLM integration files, the native `libspark_transport_capi.so` library and its serving entrypoint | [Apache-2.0](https://github.com/FujitsuPolycom/sparkring/blob/main/LICENSE) with a [NOTICE](https://github.com/FujitsuPolycom/sparkring/blob/main/NOTICE) |
| [SparkRing R10 image](https://github.com/FujitsuPolycom/sparkring) `ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:0d4029b3…` | vLLM build with both components installed; pulled by digest, not redistributed | Upstream component terms |

The table below maps every file the recipe pins by SHA-256 to its source. Each row says
whether the file is **upstream** (unchanged bytes), **modified** by this project, or
**ours**. Rows marked **provenance pending** come from the SparkRing R10 runtime, but
their exact bytes have not yet been matched to a public commit. Until they are, every file
here is obtained by the operator, and the checkout ships only hashes, links and the
tools that apply this project's changes.

## SparkCache connector and encoder

The hashes are pinned in [`scripts/prepare-sparkcache.py`](../scripts/prepare-sparkcache.py)
and [`scripts/node/sparkcache/SHA256SUMS`](../scripts/node/sparkcache/SHA256SUMS).

| File | SHA-256 | Source | Status |
| --- | --- | --- | --- |
| Encoder, original | `f02e67036f0a…` | [`sparkcache@66057174` `sparkcache/spark_context_cache_hybrid.py`](https://github.com/FujitsuPolycom/sparkcache/blob/66057174301a4759ca3a45207ea41016689449cb/sparkcache/spark_context_cache_hybrid.py), byte-identical; also installed in the R10 image | **Upstream** |
| Encoder, mounted | `11a2db855306…` | Original encoder plus one memory correction applied by `prepare-sparkcache.py` | **Modified** |
| Connector, original input | `a0bedc1c33a3…` | The R10 image's connector (SparkCache, [module at `66057174`](https://github.com/FujitsuPolycom/sparkcache/blob/66057174301a4759ca3a45207ea41016689449cb/sparkcache/spark_context_cache_connector.py)) plus this project's pending-publication change | **Modified**, base provenance pending |
| Connector, corrected | `23c1e05cc3bb…` | Original input plus one memory correction applied by `prepare-sparkcache.py` | **Modified** |
| Connector, mounted | `5893f8747aa0…` | Corrected connector plus replay views applied by [`replay-views/prepare.py`](../scripts/node/experiments/e03/replay-views/prepare.py) | **Modified** |

The connector's upstream base is the file installed in the digest-pinned R10 image. The
[SparkRing recipe for this image](https://github.com/FujitsuPolycom/sparkring/blob/main/recipes/sparkcache/README.md)
names SparkCache merge `66057174`. The image's encoder matches that commit byte for byte;
its connector does not. The exact base will be recorded here with a complete diff.

## SIRCL bundle and runtime

The hashes are pinned in [`scripts/node/sircl/SHA256SUMS`](../scripts/node/sircl/SHA256SUMS).

| File | Source | Status |
| --- | --- | --- |
| `bundle/sitecustomize.py`, `spark_collective_audit.py`, `spark_cudagraph_bucket_contract.py`, `spark_cudagraph_replay_timing.py`, `spark_dcp_collective_audit.py`, `spark_graph_status_reporter.py`, `spark_persistent_output_ring.py`, `spark_tp4_backend.py`, `spark_tp4_capability.py`, `spark_tp4_health_gate.py`, `spark_tp4_port_namespace.py`, `spark_tp4_query_contract.py`, `spark_tp4_query_row_provider.py`, `spark_tp4_vocab_allgather_backend.py`, `sparkring-overlay-manifest.json` (15 files) | [`sparkring@be2f6465` `runtime/releases/glm53-dflash-sircl-overlay/overlay.tar.gz`](https://github.com/FujitsuPolycom/sparkring/tree/be2f646523e002829a05f9ca19e1fcbb565c319f/runtime/releases/glm53-dflash-sircl-overlay), byte-identical to its members | **Upstream** |
| `bundle/libspark_transport_capi.so` | Native SIRCL library from the SparkRing R10 runtime; [build script](https://github.com/FujitsuPolycom/sparkring/blob/main/runtime/sparkring/jovian-r33/sircl/build-sircl-cu133-sm121.sh) and [SIRCL guide](https://github.com/FujitsuPolycom/sparkring/blob/main/docs/architecture/sircl.md) | Upstream, **provenance pending** |
| `bundle/sircl-bundle-manifest.json` | Bundle manifest from the SparkRing R10 runtime | Upstream, **provenance pending** |
| `runtime/entrypoint.sh`, `runtime/common.env` | Serving entrypoint and shared environment from the SparkRing R10 runtime | Upstream, **provenance pending** |
| `runtime/sircl_gid_check.py` | [`scripts/sircl_gid_check.py`](../scripts/sircl_gid_check.py) in this repository | **Ours** |
| Per-rank peer and GID files | Generated for each site's fabric; pinned only in the ignored `scripts/node/sircl/SHA256SUMS.site` | **Ours**, site data, never published |

## What this project changed

- **Pending-publication wait (connector).** A replay that finds its prefix still being
  published waits up to `spark_cache_pending_wait_ms` (2,000 ms) for every rank to confirm
  it, instead of recomputing. Upstream SparkCache has no such option.
- **Memory corrections (connector and encoder).** The saver releases each completed
  item's references before waiting for the next item. The encoder joins the page header
  and payload in one allocation. Encoded bytes and cache format are unchanged.
  [`scripts/prepare-sparkcache.py`](../scripts/prepare-sparkcache.py) applies both and
  verifies every input and output hash.
- **Replay views (connector).** [`replay-views/prepare.py`](../scripts/node/experiments/e03/replay-views/prepare.py)
  adds the replay path measured since E03.
- **GID preflight (SIRCL runtime).** [`scripts/sircl_gid_check.py`](../scripts/sircl_gid_check.py)
  checks that each port's IPv4 RoCEv2 GID matches its peer before serving.

Everything else in these two components is upstream code.

## Obtaining the files

Follow [installation step 8](install-from-zero.md#8-place-the-sparkcache-and-sircl-payload).

- **Encoder.** Extract it from the digest-pinned R10 image, or download it from the
  SparkCache commit linked above.
- **Connector.** The pinned input already contains this project's pending-publication
  change, which is not yet published. Until it is, the image's connector alone does not
  reproduce that input, and `prepare-sparkcache.py` refuses it. Contact the maintainer of
  this repository for the connector input.
- **Preparation.** Run the two preparation tools, which refuse any input whose hash
  differs.
- **SIRCL.** Take the fifteen upstream files from the SparkRing overlay linked above.
  Obtain the files marked provenance pending from the SparkRing R10 runtime. Verify all
  of them against `scripts/node/sircl/SHA256SUMS`.

Keep the SparkRing `NOTICE` with any copy.
