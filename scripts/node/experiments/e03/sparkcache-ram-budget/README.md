# SparkCache bounded disk transfers

The operational default limits admitted transient cache work to **1 GiB per
worker/rank**, shared by stores and restores. Snapshot capture, publication and restore
transfer payloads in **8 MiB pieces**, without building a complete snapshot in RAM.
Before opening a transfer it also requires **1 GiB of remaining Linux `MemAvailable`**, after
subtracting all outstanding peak reservations. Unknown availability rejects the work.
The model's context window remains unchanged. Under memory pressure stores are skipped; rejected
restores use the existing rank-synchronous `recompute` policy without invalidating data.

This protection is separate from the frozen E31 performance baseline: E31 measured the
previous replay connector and cache config. The
[versioned operational identity](../../../../../docs/operational-identities/2026-09-29-sparkcache-protected.json)
pins E31 by hash and records this cache delta. Preparing or testing the payload does not
deploy or restart the service, and it does not change the frozen measurements.

## Budget and ownership

`kv-transfer-config.json` selects `spark_cache_cpu_budget_bytes=1073741824` and
`spark_cache_min_available_bytes=1073741824`; both require positive integers. Reservations
use `8 * 8 MiB + 64 * span_tokens + 8 MiB`. This covers overlapping transfer buffers,
Python positions/block tables and metadata. Payload reservations are independent of the
complete snapshot size: at 179,712 tokens, one operation reserves about **83 MiB**, even
when its on-disk snapshot exceeds 1 GiB. These are conservative accounted bytes, not a
measured RSS peak. The exact encoded size still comes from the registered page layout.

Capture detaches selected GPU pages to an anonymous temporary file on the cache volume
before `wait_for_save` returns and vLLM may reuse those pages. Even a single page larger
than 8 MiB is split into slices. The background saver then publishes immutable objects
and the authenticated manifest last. Restore hashes object and snapshot bytes while
reading bounded pieces directly into the selected destination pages. No full-layer or
full-snapshot join is used. Completion requires all checksums and the final CUDA fence;
if any check fails, every destination block is rejected through the existing recompute
policy. Partial writes are never reported as a successful restore.

Temporary staging needs up to one snapshot's size on disk while the saver runs, in
addition to published cache objects. The file is unlinked/anonymous and closes on
capture failure, failed enqueue, completed commit or process exit. Bounded writeback and
cache-drop hints limit dirty-page accumulation where the OS supports them; kernel cache
reclamation remains an OS policy. Staging adds disk traffic, and capture blocks the
producer until every page is detached. Runtime latency and throughput remain unmeasured.
`CACHE_DIR` must reside on a disk-backed filesystem with enough free space for staging
and publication; a RAM-backed mount such as `tmpfs` defeats this design.

The budget is held while a snapshot is produced, queued and committed. A producer must
drop its references before the saver can release it. A cancelled restore retains its
reservation until its owner returns from placement and the CUDA stream drains. Failure
to drain leaves the reservation charged. Concurrent operations use the same locked
counter. Request cancellation itself never returns memory to the budget.

The cap covers **accounted transient cache work**, not the process's total RSS, model
weights, preallocated KV pool, filesystem cache, allocator retention, or unrelated
processes. The availability floor is checked at admission; another allocator can consume
memory afterward. This is not an OS-enforced RSS ceiling or a guarantee against every
host-memory failure. Disk-cache capacity is a separate policy.

## Supported cache paths

Only synchronous `block_pages_v1` snapshots are supported. Startup rejects native CUDA
restore and upstream streaming/asynchronous page capture (which use different ownership
contracts). The bounded disk transfer here is separate from those upstream features.
Every store publishes a complete
independent page snapshot; it cannot read an older base during extension commit. Restores
accept only exact `page_snapshot` roots whose authenticated `snapshot_encoded_bytes` and
`committed_tokens` match the requested geometry. Deltas, aliases and legacy chunk roots
miss without payload reads. Explicit payload integrity sweeps are disabled; admitted
restores still verify checksums and placement geometry. Metadata discovery is unchanged.

The default uses a separate disk namespace so legacy roots are not repeatedly offered. It
does not delete old cache data. Full snapshots may consume more disk and skip more cache
work than delta publication; performance remains unmeasured.

The underlying storage contract was inspected at SparkCache commit
[`66057174301a4759ca3a45207ea41016689449cb`](https://github.com/FujitsuPolycom/sparkcache/blob/66057174301a4759ca3a45207ea41016689449cb/sparkcache/persistent_context_cache/cache_manifest.py):
plain snapshot roots authenticate their full encoded size and object coverage. The
candidate adapts publication to stream objects to disk and reads each extent in bounded
pieces. It validates file lengths before reading and incrementally verifies checksums;
an enlarged or truncated object cannot cause an unbounded payload read. Metadata still
uses the pinned upstream parser. The on-disk page format remains compatible.

## Reproduce and verify locally

```sh
python3 scripts/node/experiments/e03/sparkcache-ram-budget/prepare.py --check
python3 scripts/tests/test-sparkcache-memory-budget.py
python3 scripts/tests/test-sparkcache-ram-connector.py
python3 scripts/tests/test-sparkcache-stream-io.py
python3 scripts/tests/test-sparkcache-stream-connector.py
python3 scripts/tests/test-sparkcache-ram-config.py
./scripts/check.sh
```

`prepare.py` embeds the budget helper, `stream_io.py` and `bounded_connector.py` into
a new copy of the pinned E03 connector. One queue-handoff substitution lets the producer
transfer ownership to the saver. It also generates patch 05 for provenance. The launcher
verifies the entire resulting connector and configuration using their recipe pins;
no new unpinned runtime import is needed.

`delta.env` is retained as a parity witness for the configuration that was loaded before
its promotion. Applied to the measured E31 recipe, it produces the protected 16 GiB
operational predecessor, not the current memory-bounded default. Its fail-closed guards
reject any recipe that already selects the protected connector, so operators must not
select it with `TP4_ENV` now.

## Coordinated default activation and rollback

Use the protected migration in the existing
[operations procedure](../../../../../docs/operations.md#migrate-the-protected-sparkcache-default-without-reloading-the-model)
when the same bounded-transfer variant is healthy on all four ranks and site-specific
launcher parity is exact. It deploys the default and retires only the verified SparkCache
autostart selection without restarting the containers or reloading model weights.

If any rank is absent, health is inconsistent, another recipe is serving or parity is not
exact, stop and report. Recover the whole group first. In a later authorized window, stop
with the recipe that is serving, remove only the verified candidate `TP4_ENV` autostart
selection and reload systemd, deploy/start the protected default with no `TP4_ENV`, and repeat
both functional gates. Preserve unrelated drop-ins. Never repair or restart only one serving
rank.

The expected boot signature on every worker is:

```text
SPARKCACHE_CPU_BUDGET_READY cap_bytes=1073741824 floor_bytes=1073741824
SPARKCACHE_DISK_STREAM_READY chunk_bytes=8388608
```

Require `/health` 200 and both documented functional gates within two minutes. Record
the default operational identity; it is separate from the frozen E31 identity. Then validate a
small cache store/replay and the large-context cancellation/reissue case while sampling
host memory on all four ranks. Verify a long snapshot completes both store and restore
without a full-size RAM allocation, and pressure produces `SPARKCACHE_CPU_BUDGET_SKIP`
without worker death. Confirm fallback completion after a rejected/corrupt restore.
Record GPU/runtime evidence separately from the frozen performance record.

The immediate operational rollback is a coordinated `down` with the serving recipe, then
deploy/start with
`TP4_ENV=scripts/node/reference/operational-20260929-sparkcache-protected.env` and repeat
the gates. Keep that overlay for subsequent lifecycle commands and autostart. It preserves
the cache bounds but restores 16 GiB KV and removes allocator trim, the step cap and API
admission, so it is not the complete memory-bounded protection. Use the historical
`baseline-20260928-e31.env` only to restore the prior namespace and unrestricted cache-memory
behavior deliberately. Never change one serving rank independently.
