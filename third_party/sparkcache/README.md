# SparkCache connector and encoder

These are the prefix-cache files the recipe mounts into the R10 container. They derive
from [SparkCache](https://github.com/FujitsuPolycom/sparkcache) commit
[`66057174301a4759ca3a45207ea41016689449cb`](https://github.com/FujitsuPolycom/sparkcache/tree/66057174301a4759ca3a45207ea41016689449cb),
licensed under the Apache License 2.0. The license, copied from that commit, is in
[`LICENSE`](LICENSE). The SparkCache repository has no `NOTICE` file.

The SparkCache files installed in the digest-pinned R10 image
(`sparkcache/spark_context_cache_connector.py`, SHA-256 `394775d4…`, and
`sparkcache/spark_context_cache_hybrid.py`, `f02e6703…`) are byte-identical to the same
paths at that commit.

## Files

`scripts/deploy.sh` copies every `.py` file here to `~/tp4/sparkcache/` on each node.
The launcher verifies the selected connector and encoder against the SHA-256 pins in
`cluster.env` before starting Docker.

| File | SHA-256 | Origin | Use |
| --- | --- | --- | --- |
| `spark_context_cache_connector-e03-replay-views.py` | `5893f8747aa0…` | Upstream connector plus patches 01, 02 and 03 | Current connector (E03 onward) |
| `spark_context_cache_hybrid.py` | `11a2db855306…` | Upstream encoder plus patch 04 | Current encoder |
| `spark_context_cache_connector.py` | `23c1e05cc3bb…` | Upstream connector plus patches 01 and 02 | September 19 rollback |
| `spark_context_cache_connector-20260918.py` | `a0bedc1c33a3…` | Upstream connector plus patch 01 | September 18 rollback |

## Changes made by this project

**Modified files notice:** this project changed all four files above. Each patch in
[`patches/`](patches/) is the complete unified diff for one change, so the files can be
compared with upstream line by line. All other code is unchanged SparkCache code.

| Patch | Files | Change |
| --- | --- | --- |
| [`01-connector-pending-publication.patch`](patches/01-connector-pending-publication.patch) | Connector | A replay that finds its prefix still being published by another rank waits up to `spark_cache_pending_wait_ms` for every rank to confirm it, instead of recomputing. It adds the option, the wait and its trace logging. |
| [`02-connector-memory.patch`](patches/02-connector-memory.patch) | Connector | The saver commits each item in its own call frame, so the item's references are released before the next queue wait. [`scripts/prepare-sparkcache.py`](../../scripts/prepare-sparkcache.py) applies it. |
| [`03-connector-replay-views.patch`](patches/03-connector-replay-views.patch) | Connector | A restore reads each layer as a view of the authenticated snapshot instead of copying its whole body again. [`replay-views/prepare.py`](../../scripts/node/experiments/e03/replay-views/prepare.py) applies it. |
| [`04-encoder-memory.patch`](patches/04-encoder-memory.patch) | Encoder | Joins the page header and payload in one allocation. `prepare-sparkcache.py` applies it. |

The encoded bytes and the on-disk cache format are unchanged. `scripts/tests/test-third-party-payload.py`
reverses the patches and checks that the result is byte-identical to upstream.
