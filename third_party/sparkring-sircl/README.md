# SparkRing SIRCL transport

SIRCL is SparkRing's RDMA transport. In this recipe it carries the TP4 all-reduce and the
single-rail prefill exchange. The files come from
[SparkRing](https://github.com/FujitsuPolycom/sparkring), licensed under the Apache License
2.0. [`LICENSE`](LICENSE), [`NOTICE`](NOTICE) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)
are copied from SparkRing commit
[`b358a818786d8506086aaaabb9afe464fa2ccb49`](https://github.com/FujitsuPolycom/sparkring/tree/b358a818786d8506086aaaabb9afe464fa2ccb49),
the source of the native library. Keep them with any copy of these files.

`scripts/deploy.sh` copies `bundle/` to `~/tp4/sircl/bundle/` and `runtime/` to
`~/tp4/sircl/runtime/` on each node, together with this project's
[`scripts/sircl_gid_check.py`](../../scripts/sircl_gid_check.py). The launcher verifies them
against [`scripts/node/sircl/SHA256SUMS`](../../scripts/node/sircl/SHA256SUMS) before starting
Docker; the container mounts them at `/opt/spark-sircl` and `/opt/sircl-serving`.

## Provenance

| Files | Origin | Status |
| --- | --- | --- |
| `bundle/` Python modules and `sparkring-overlay-manifest.json` (15 files) | Members of [`runtime/releases/glm53-dflash-sircl-overlay/overlay.tar.gz`](https://github.com/FujitsuPolycom/sparkring/tree/be2f646523e002829a05f9ca19e1fcbb565c319f/runtime/releases/glm53-dflash-sircl-overlay) at SparkRing `be2f6465`, byte-identical | Upstream |
| `bundle/libspark_transport_capi.so` | SparkRing native library for CUDA SM121, built from `spark_transport/` at SparkRing `b358a818` | Upstream binary |
| `bundle/sircl-bundle-manifest.json` | Build record of that library: source commit, SHA-256 of all 113 native source files, toolchain and target image | Upstream |
| `runtime/common.env` | Shared SIRCL serving environment of the R10 runtime, with the defaults of SparkRing's `runtime/glm53-flash-jj-r8-gb10/launch-rank.sh` | Upstream |
| `runtime/entrypoint.sh` | R10 serving entrypoint plus one line added by this project | Modified |

All 113 source hashes in `sircl-bundle-manifest.json` match the files at SparkRing
`b358a818`. The library itself is a binary build: its SHA-256 (`f53c88b4…`) is pinned by
the manifest and by `common.env`, but no public SparkRing receipt records that exact build.
The [SparkRing build instructions](https://github.com/FujitsuPolycom/sparkring/blob/b358a818786d8506086aaaabb9afe464fa2ccb49/spark_transport/README.md#build)
rebuild it from that commit, with the toolchain listed in the manifest. A rebuilt binary
has a different hash, so it needs new pins in `common.env`, `entrypoint.sh`, the manifest
and `SHA256SUMS`, and its own validation.

## Changes made by this project

**Modified file notice:** `runtime/entrypoint.sh` differs from the R10 entrypoint by one
line, which runs this project's GID check before `exec vllm serve`:

```sh
python3 -S /opt/sircl-serving/sircl_gid_check.py --fabric-ifaces enp1s0f0np0 enp1s0f1np1
```

The interface names are those of the ASUS Ascent GX10. A system whose first two fabric
ports have other names must change this line, then the `runtime/entrypoint.sh` pin in
`scripts/node/sircl/SHA256SUMS`.

## Site files

The runtime also needs four per-rank files (`rank<N>.env`: two ring-peer addresses, two
RDMA devices, two GID indexes) and a `runtime/SHA256SUMS` that lists every mounted file.
They describe one site's fabric, so they are not in this directory.
`scripts/sircl-site-files.py` generates them from `cluster.env` into the ignored
`scripts/node/sircl/site/`.
