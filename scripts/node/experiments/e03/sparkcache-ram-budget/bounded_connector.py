# SPDX-License-Identifier: Apache-2.0
"""Appended to the pinned connector by prepare.py; not imported separately."""

from dataclasses import replace as _memory_replace
from sparkcache.spark_context_cache_hybrid import (
    page_snapshot_encoded_size, encode_page_snapshot_header,
)


@dataclass
class _CpuStoreTicket:
    reservation: MemoryReservation
    snapshot_id: int | None = None
    queued: bool = False
    resource: object | None = None


_UnbudgetedSparkContextCacheConnector = SparkContextCacheConnector


class SparkContextCacheConnector(_UnbudgetedSparkContextCacheConnector):
    """Bound optional cache payload work before any CPU snapshot/read occurs."""

    def __init__(self, vllm_config, role, kv_cache_config):
        transfer = vllm_config.kv_transfer_config
        config = parse_connector_config(vllm_config, transfer, kv_cache_config)
        if (config.storage_mode != "block_pages_v1"
                or config.native_restore_enabled
                or config.streaming_snapshots_enabled
                or config.async_page_capture_enabled):
            raise RuntimeError("CPU cache budget requires synchronous block pages")
        if transfer.kv_load_failure_policy != "recompute":
            raise RuntimeError("CPU cache budget requires recompute on load failure")
        cap = transfer.get_from_extra_config("spark_cache_cpu_budget_bytes", 1 << 30)
        floor = transfer.get_from_extra_config("spark_cache_min_available_bytes", 1 << 30)
        self._cpu_budget = MemoryBudget(cap, floor)
        self._cpu_store_lock = threading.RLock()
        self._cpu_store_tickets = {}
        self._cpu_producer_tickets = None
        super().__init__(vllm_config, role, kv_cache_config)
        logger.info("SPARKCACHE_CPU_BUDGET_READY cap_bytes=%d floor_bytes=%d", cap, floor)
        logger.info("SPARKCACHE_DISK_STREAM_READY chunk_bytes=%d", STREAM_CHUNK_BYTES)

    def _cpu_geometry(self, plan):
        layout = self._page_layout
        if layout is None:
            raise RuntimeError("CPU cache budget requires a registered page layout")
        groups = self._select_group_blocks_for_span(
            plan.group_block_ids, plan.span_tokens,
            recurrent_boundary_blocks=plan.recurrent_boundary_blocks,
        )
        counts = tuple(map(len, groups))
        if len(counts) != len(layout.groups) or any(count <= 0 for count in counts):
            raise RuntimeError("CPU cache budget rejected invalid page geometry")
        encoded = page_snapshot_encoded_size(layout, counts)
        # Payload buffers are bounded independently of the snapshot length.
        # Allow eight chunks for gather, device/host copies and disk IO, plus
        # Python positions/block tables and a fixed metadata allowance.
        peak = 8 * STREAM_CHUNK_BYTES + 64 * plan.span_tokens + (8 << 20)
        return counts, encoded, peak

    def _reserve_cpu(self, peak, operation):
        reservation = self._cpu_budget.try_reserve(peak)
        if reservation is None:
            logger.info("SPARKCACHE_CPU_BUDGET_SKIP operation=%s peak_bytes=%d "
                        "reserved_bytes=%d", operation, peak,
                        self._cpu_budget.reserved_bytes)
        return reservation

    def wait_for_save(self):
        # The producer may still refer to the snapshot after queue.put(). The
        # saver cannot return its budget until this entire base frame is gone.
        with self._cpu_store_lock:
            self._cpu_producer_tickets = []
            failure = None
            try:
                super().wait_for_save()
            except Exception as error:
                # Drop exception traceback frames (and their payload locals)
                # before returning unqueued reservations to other threads.
                failure = f"{type(error).__name__}: {error}"
            tickets = self._cpu_producer_tickets
            self._cpu_producer_tickets = None
            for ticket in tickets:
                if not ticket.queued:
                    self._cpu_store_tickets.pop(ticket.snapshot_id, None)
                    if ticket.resource is not None:
                        ticket.resource.release()
                    ticket.reservation.release()
            if failure is not None:
                raise RuntimeError(failure)

    def _snapshot_store(self, plan):
        if self._cpu_producer_tickets is None:
            raise RuntimeError("snapshot must have a budgeted producer owner")
        counts, encoded, peak = self._cpu_geometry(plan)
        reservation = self._reserve_cpu(peak, "store")
        if reservation is None:
            raise RuntimeError("optional cache store exceeds CPU memory budget")
        ticket = _CpuStoreTicket(reservation)
        self._cpu_producer_tickets.append(ticket)
        # Detach every selected page to disk before vLLM can reuse its blocks.
        # Independent publication never reads an older extension base.
        full_plan = _memory_replace(plan, base_context_digest="", base_span_tokens=0)
        snapshot = self._capture_stream_snapshot(full_plan, counts, encoded, ticket)
        ticket.snapshot_id = id(snapshot)
        self._cpu_store_tickets[id(snapshot)] = ticket
        return snapshot

    def _enqueue_budgeted_snapshot(self, snapshot):
        ticket = self._cpu_store_tickets[id(snapshot)]
        self._store_queue.put(snapshot)
        ticket.queued = True

    def _store_worker_main(self):
        while True:
            snapshot = self._store_queue.get()
            if snapshot is None:
                return
            self._commit_store_snapshot(snapshot)
            with self._cpu_store_lock:
                ticket = self._cpu_store_tickets.pop(id(snapshot))
                del snapshot
                if ticket.resource is not None:
                    ticket.resource.release()
                ticket.reservation.release()

    def _raw_page_rows(self, layer):
        tensor = self._layer_tensors[layer.name]
        # view() must alias the actual KV storage. Never reshape/contiguous the
        # complete pool: either could silently copy every allocated KV page.
        rows = tensor.view(torch.uint8).view(tensor.shape[0], -1)
        if rows.shape[1] != layer.bytes_per_page:
            raise RuntimeError("streaming page geometry differs from registered KV")
        return rows

    def _stream_segments(self, plan):
        groups = self._select_group_blocks_for_span(
            plan.group_block_ids, plan.span_tokens,
            recurrent_boundary_blocks=plan.recurrent_boundary_blocks,
        )
        for group, block_ids in zip(self._page_layout.groups, groups, strict=True):
            for layer in group.layers:
                rows = self._raw_page_rows(layer)
                page_bytes = layer.bytes_per_page
                if page_bytes <= STREAM_CHUNK_BYTES:
                    batch_pages = max(1, STREAM_CHUNK_BYTES // page_bytes)
                    for start in range(0, len(block_ids), batch_pages):
                        selected = block_ids[start:start + batch_pages]
                        yield rows, selected, 0, page_bytes, True
                else:
                    # Even a single large recurrent page must not defeat the
                    # transfer cap. Slice its raw bytes without copying it.
                    for block in block_ids:
                        for offset in range(0, page_bytes, STREAM_CHUNK_BYTES):
                            size = min(STREAM_CHUNK_BYTES, page_bytes - offset)
                            yield rows, (block,), offset, size, False

    @staticmethod
    def _capture_segment(rows, blocks, offset, size, whole_pages):
        if whole_pages:
            index = torch.tensor(blocks, dtype=torch.long, device=rows.device)
            selected = rows.index_select(0, index)
        else:
            selected = rows[blocks[0], offset:offset + size]
        # Blocking D2H establishes ownership before source pages are released.
        return selected.to(device="cpu", non_blocking=False).numpy().tobytes()

    def _capture_stream_snapshot(self, plan, counts, encoded, ticket):
        spool = DiskSnapshot(self._root, encoded)
        ticket.resource = spool
        header = encode_page_snapshot_header(self._page_layout, counts)
        spool.append(header)
        for segment in self._stream_segments(plan):
            payload = self._capture_segment(*segment)
            spool.append(payload)
            del payload
        spool.seal()
        rank = self._worker_rank()
        return _HybridStoreSnapshot(
            plan=plan, rank=rank, identity=self._identity(rank),
            positions=tuple(range(plan.span_tokens)), encoded_pages=spool,
            block_counts=counts,
        )

    @staticmethod
    def _place_segment(rows, blocks, offset, size, whole_pages, payload):
        expected = len(blocks) * size
        if len(payload) != expected:
            raise RuntimeError("streaming restore segment length differs")
        source = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
        # CPU buffer reuse is safe only after this blocking H2D finishes.
        source = source.to(device=rows.device, non_blocking=False)
        if whole_pages:
            index = torch.tensor(blocks, dtype=torch.long, device=rows.device)
            rows.index_copy_(0, index, source.view(len(blocks), size))
        else:
            rows[blocks[0], offset:offset + size].copy_(source)

    def _restore_stream_snapshot(self, lookup, plan, counts, encoded, timing):
        header = encode_page_snapshot_header(self._page_layout, counts)
        started = time.perf_counter_ns()
        read_ns = submit_ns = 0
        with self._store.open_snapshot(
            lookup, layout=self._page_layout, result_block_counts=counts,
            result_boundary_tokens=plan.span_tokens,
        ) as reader:
            if reader.read_exact(len(header)) != header:
                raise RuntimeError("streaming snapshot header differs from page layout")
            read_ns += time.perf_counter_ns() - started
            with self._load_write_context():
                for rows, blocks, offset, size, whole_pages in self._stream_segments(plan):
                    started = time.perf_counter_ns()
                    payload = reader.read_exact(len(blocks) * size)
                    read_ns += time.perf_counter_ns() - started
                    started = time.perf_counter_ns()
                    self._place_segment(rows, blocks, offset, size, whole_pages, payload)
                    submit_ns += time.perf_counter_ns() - started
                    del payload
            # No successful completion can be reported before every checksum
            # passes. A failure discards all destination blocks via recompute.
            started = time.perf_counter_ns()
            reader.finish()
            read_ns += time.perf_counter_ns() - started
        if timing is not None:
            timing.page_bytes = encoded
            timing.chunk_count = chunk_count(plan.span_tokens, self._chunk_tokens)
            timing.observe("restore_read", read_ns)
            timing.observe("h2d_submit", submit_ns)
        return True

    def _load_hybrid_pages(self, lookup, plan, *, timing=None, native_lane=0):
        # Deltas, prefix aliases and legacy chunks may read a larger backing
        # root than the selected prefix. Miss without deleting healthy data.
        if lookup.root_kind != "page_snapshot":
            logger.info("SPARKCACHE_CPU_BUDGET_SKIP operation=restore reason=root_kind")
            return False
        counts, encoded, peak = self._cpu_geometry(plan)
        manifest = lookup._manifest or {}
        if (manifest.get("snapshot_encoded_bytes") != encoded
                or manifest.get("committed_tokens") != plan.span_tokens):
            logger.info("SPARKCACHE_CPU_BUDGET_SKIP operation=restore reason=geometry")
            return False
        reservation = self._reserve_cpu(peak, "restore")
        if reservation is None:
            return False
        failure = None
        try:
            result = self._restore_stream_snapshot(lookup, plan, counts, encoded, timing)
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
        # Cancellation never releases this reservation. The placement frame,
        # including any exception traceback payloads, is gone before release.
        # A failing fence leaves the budget charged (fail closed).
        if self._load_stream is not None:
            started = time.perf_counter_ns()
            self._load_stream.synchronize()
            if timing is not None:
                timing.observe("cuda_sync", time.perf_counter_ns() - started)
        reservation.release()
        if failure is not None:
            raise RuntimeError(failure)
        return result

    def _prepare_page_base_read_cohorts(self, load_plans):
        # Accepted roots are independent snapshots, so no shared base payload
        # may outlive a restore reservation or defer another load.
        return list(load_plans), [], {}

    def sweep_integrity(self):
        raise RuntimeError("CPU-budget connector does not support payload sweeps; "
                           "normal restores verify each admitted snapshot")
