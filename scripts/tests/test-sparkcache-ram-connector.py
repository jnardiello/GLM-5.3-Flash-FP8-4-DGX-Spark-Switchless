#!/usr/bin/env python3
"""Stdlib lifecycle tests for the SparkCache CPU-memory-budget connector."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import logging
from pathlib import Path
import queue
import sys
import threading
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[2]
CANDIDATE = REPO / "scripts/node/experiments/e03/sparkcache-ram-budget"
MEMORY_BUDGET = REPO / "third_party/sparkcache/spark_context_cache_memory_budget.py"
ENCODED_BYTES = 4096
STREAM_CHUNK_BYTES = 8 << 20


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


memory = load_module("sparkcache_memory_budget_connector_test", MEMORY_BUDGET)


@dataclass(frozen=True)
class Plan:
    span_tokens: int = 16
    group_block_ids: tuple[tuple[int, ...], ...] = ((4, 5),)
    recurrent_boundary_blocks: tuple[int, ...] = ()
    base_context_digest: str = "older-snapshot"
    base_span_tokens: int = 8


@dataclass
class Snapshot:
    plan: Plan
    payload_marker: object


class DiskResource:
    def __init__(self) -> None:
        self.released = False
        self.release_calls = 0

    def release(self) -> None:
        self.release_calls += 1
        self.released = True


class Transfer:
    def __init__(self, *, cap: int = 1 << 30, floor: int = 1,
                 failure_policy: str = "recompute") -> None:
        self.kv_load_failure_policy = failure_policy
        self.extra = {
            "spark_cache_cpu_budget_bytes": cap,
            "spark_cache_min_available_bytes": floor,
        }

    def get_from_extra_config(self, name, default):
        return self.extra.get(name, default)


class FailingQueue:
    def put(self, _value):
        raise RuntimeError("queue closed")


class BlockingStream:
    def __init__(self, *, fail: bool = False) -> None:
        self.entered = threading.Event()
        self.drain = threading.Event()
        self.fail = fail

    def synchronize(self) -> None:
        self.entered.set()
        self.drain.wait(2)
        if self.fail:
            raise RuntimeError("stream fence failed")


class BaseConnector:
    """Small execution double for the pinned connector's lifecycle callbacks."""

    def __init__(self, vllm_config, role, kv_cache_config) -> None:
        del role, kv_cache_config
        self._page_layout = SimpleNamespace(groups=(object(),), encoded_bytes=ENCODED_BYTES)
        self._load_stream = None
        self._store_queue = queue.Queue()
        self._store_thread = None
        self.plan = getattr(vllm_config, "plan", None)
        self.auto_store_worker = False
        self.hold_producer = None
        self.enqueued = threading.Event()
        self.capture_calls = 0
        self.capture_plans = []
        self.capture_encoded = []
        self.capture_error = None
        self.resources = []
        self.payload_marker = object()
        self.committed = []
        self.commit_completed = threading.Event()
        self.commit_error = None
        self.store_error = None
        self.restore_calls = 0
        self.restore_result = object()
        self.restore_error = None
        self.invalidations = 0

    def _select_group_blocks_for_span(
        self, groups, span_tokens, recurrent_boundary_blocks=None,
    ):
        del span_tokens, recurrent_boundary_blocks
        return groups

    def wait_for_save(self):
        if self.plan is None:
            return None
        try:
            snapshot = self._snapshot_store(self.plan)
            self._ensure_store_thread()
            self._enqueue_budgeted_snapshot(snapshot)
            self.enqueued.set()
            if self.hold_producer is not None:
                self.hold_producer.wait(2)
        except Exception as error:
            # The real producer reports an optional-cache failure and returns.
            self.store_error = f"{type(error).__name__}: {error}"
        return None

    def _ensure_store_thread(self):
        if self.auto_store_worker:
            self.start_store_worker()

    def start_store_worker(self):
        if self._store_thread is None:
            self._store_thread = threading.Thread(target=self._store_worker_main)
            self._store_thread.start()

    def stop_store_worker(self):
        if self._store_thread is not None and self._store_thread.is_alive():
            self._store_queue.put(None)
            self._store_thread.join(2)

    def _commit_store_snapshot(self, snapshot):
        if self.commit_error is not None:
            raise self.commit_error
        self.committed.append(snapshot)
        self.commit_completed.set()

    def _invalidate_after_failure(self, *args, **kwargs):
        del args, kwargs
        self.invalidations += 1


def parse_connector_config(vllm_config, transfer, kv_cache_config):
    del transfer, kv_cache_config
    return vllm_config.parsed_config


def load_candidate():
    sparkcache = ModuleType("sparkcache")
    sparkcache.__path__ = []
    hybrid = ModuleType("sparkcache.spark_context_cache_hybrid")
    hybrid.page_snapshot_encoded_size = lambda layout, counts: layout.encoded_bytes
    hybrid.encode_page_snapshot_header = lambda layout, counts: b"header"
    candidate = ModuleType("bounded_connector_test")
    candidate.__dict__.update({
        "__name__": "bounded_connector_test",
        "dataclass": dataclass,
        "MemoryBudget": memory.MemoryBudget,
        "MemoryReservation": memory.MemoryReservation,
        "SparkContextCacheConnector": BaseConnector,
        "parse_connector_config": parse_connector_config,
        "threading": threading,
        "time": time,
        "logger": logging.getLogger("sparkcache-ram-budget-test"),
        "STREAM_CHUNK_BYTES": STREAM_CHUNK_BYTES,
        "DiskSnapshot": DiskResource,
    })
    with patch.dict(sys.modules, {
        "sparkcache": sparkcache,
        "sparkcache.spark_context_cache_hybrid": hybrid,
        "bounded_connector_test": candidate,
    }):
        source = (CANDIDATE / "bounded_connector.py").read_text()
        exec(compile(source, str(CANDIDATE / "bounded_connector.py"), "exec"),
             candidate.__dict__)
    return candidate.SparkContextCacheConnector


BoundedConnector = load_candidate()


class Connector(BoundedConnector):
    """Use small streaming bodies while retaining the real budget wrappers."""

    def _capture_stream_snapshot(self, plan, counts, encoded, ticket):
        del counts
        self.capture_calls += 1
        self.capture_plans.append(plan)
        self.capture_encoded.append(encoded)
        resource = DiskResource()
        self.resources.append(resource)
        ticket.resource = resource
        if self.capture_error is not None:
            raise RuntimeError(self.capture_error)
        return Snapshot(plan=plan, payload_marker=self.payload_marker)

    def _restore_stream_snapshot(self, lookup, plan, counts, encoded, timing):
        del lookup, plan, counts, encoded, timing
        self.restore_calls += 1
        if self.restore_error is not None:
            raise ValueError(self.restore_error)
        return self.restore_result


def expected_peak(plan: Plan) -> int:
    return 8 * STREAM_CHUNK_BYTES + 64 * plan.span_tokens + (8 << 20)


def make_connector(*, reader=lambda: 1 << 40, cap=1 << 30, floor=1,
                   config=None, failure_policy="recompute", plan=Plan()):
    transfer = Transfer(cap=cap, floor=floor, failure_policy=failure_policy)
    parsed = config or SimpleNamespace(
        storage_mode="block_pages_v1",
        native_restore_enabled=False,
        streaming_snapshots_enabled=False,
        async_page_capture_enabled=False,
    )
    vllm_config = SimpleNamespace(
        kv_transfer_config=transfer, parsed_config=parsed, plan=plan,
    )
    connector = Connector(vllm_config, "worker", object())
    connector._cpu_budget = memory.MemoryBudget(cap, floor, reader)
    return connector


def lookup(plan: Plan, root_kind="page_snapshot", *, encoded=ENCODED_BYTES):
    return SimpleNamespace(root_kind=root_kind, _manifest={
        "snapshot_encoded_bytes": encoded,
        "committed_tokens": plan.span_tokens,
    })


def wait_for(predicate, message: str) -> None:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError(message)


class StoreBudgetTests(unittest.TestCase):
    def test_denied_stores_do_not_enter_snapshot_allocation(self):
        plan = Plan()
        peak = expected_peak(plan)
        cases = (
            ("cap", peak - 1, lambda: 1 << 40),
            ("floor", peak * 2, lambda: peak),
            ("unknown", peak * 2, lambda: None),
        )
        for name, cap, reader in cases:
            with self.subTest(name=name):
                connector = make_connector(cap=cap, floor=1, reader=reader, plan=plan)
                connector.wait_for_save()
                self.assertEqual(connector.capture_calls, 0)
                self.assertEqual(connector._cpu_budget.reserved_bytes, 0)
                self.assertIn("exceeds CPU memory budget", connector.store_error)

    def test_queued_store_holds_budget_until_saver_cleanup(self):
        plan = Plan()
        connector = make_connector(plan=plan)
        peak = expected_peak(plan)
        connector.wait_for_save()

        self.assertEqual(connector._cpu_budget.reserved_bytes, peak)
        self.assertEqual(connector.capture_calls, 1)
        self.assertEqual(connector.capture_plans[0].base_context_digest, "")
        self.assertEqual(connector.capture_plans[0].base_span_tokens, 0)
        self.assertEqual(plan.base_context_digest, "older-snapshot")

        connector.start_store_worker()
        wait_for(lambda: connector._cpu_budget.reserved_bytes == 0,
                 "saver did not return the reservation")
        self.assertEqual(len(connector.committed), 1)
        self.assertIs(connector.committed[0].payload_marker, connector.payload_marker)
        self.assertTrue(connector.resources[0].released)
        self.assertEqual(connector.resources[0].release_calls, 1)
        connector.stop_store_worker()

    def test_copy_and_enqueue_failures_return_budget_and_disk_resource(self):
        connector = make_connector()
        connector.capture_error = "copy failed"
        connector.wait_for_save()
        self.assertEqual(connector.capture_calls, 1)
        self.assertEqual(connector._cpu_budget.reserved_bytes, 0)
        self.assertEqual(connector._cpu_store_tickets, {})
        self.assertTrue(connector.resources[0].released)
        self.assertEqual(connector.resources[0].release_calls, 1)

        connector = make_connector()
        connector._store_queue = FailingQueue()
        connector.wait_for_save()
        self.assertEqual(connector.capture_calls, 1)
        self.assertEqual(connector._cpu_budget.reserved_bytes, 0)
        self.assertEqual(connector._cpu_store_tickets, {})
        self.assertTrue(connector.resources[0].released)
        self.assertEqual(connector.resources[0].release_calls, 1)

    def test_saver_cannot_release_before_producer_frame_returns(self):
        connector = make_connector()
        connector.auto_store_worker = True
        connector.hold_producer = threading.Event()
        producer = threading.Thread(target=connector.wait_for_save)
        producer.start()

        self.assertTrue(connector.commit_completed.wait(2))
        self.assertTrue(producer.is_alive())
        self.assertEqual(connector._cpu_budget.reserved_bytes, expected_peak(connector.plan))
        connector.hold_producer.set()
        producer.join(2)
        wait_for(lambda: connector._cpu_budget.reserved_bytes == 0,
                 "reservation survived producer and saver cleanup")
        connector.stop_store_worker()

    def test_unexpected_saver_failure_keeps_budget_charged(self):
        connector = make_connector()
        connector.wait_for_save()
        connector.commit_error = AssertionError("unexpected commit failure")
        with self.assertRaisesRegex(AssertionError, "unexpected commit failure"):
            connector._store_worker_main()
        self.assertEqual(connector._cpu_budget.reserved_bytes, expected_peak(connector.plan))
        self.assertFalse(connector.resources[0].released)

    def test_large_encoded_snapshot_is_admitted_by_bounded_peak(self):
        plan = Plan(span_tokens=180_000)
        connector = make_connector(plan=plan, cap=1 << 30)
        connector._page_layout.encoded_bytes = 1_100_000_000
        connector.wait_for_save()

        self.assertEqual(connector.capture_calls, 1)
        self.assertEqual(connector.capture_encoded, [1_100_000_000])
        self.assertLess(expected_peak(plan), 1 << 30)
        self.assertEqual(connector._cpu_budget.reserved_bytes, expected_peak(plan))
        connector.start_store_worker()
        wait_for(lambda: connector._cpu_budget.reserved_bytes == 0,
                 "large logical snapshot reservation was not returned")
        connector.stop_store_worker()


class RestoreBudgetTests(unittest.TestCase):
    def test_denied_restores_do_not_enter_payload_read(self):
        plan = Plan()
        peak = expected_peak(plan)
        cases = (
            ("cap", peak - 1, lambda: 1 << 40),
            ("floor", peak * 2, lambda: peak),
            ("unknown", peak * 2, lambda: None),
        )
        for name, cap, reader in cases:
            with self.subTest(name=name):
                connector = make_connector(cap=cap, floor=1, reader=reader)
                self.assertFalse(connector._load_hybrid_pages(lookup(plan), plan))
                self.assertEqual(connector.restore_calls, 0)
                self.assertEqual(connector._cpu_budget.reserved_bytes, 0)

    def test_cancelled_restore_holds_budget_until_stream_drains(self):
        plan = Plan()
        connector = make_connector()
        stream = BlockingStream()
        connector._load_stream = stream
        result = []
        restore = threading.Thread(
            target=lambda: result.append(connector._load_hybrid_pages(lookup(plan), plan))
        )
        restore.start()

        self.assertTrue(stream.entered.wait(2))
        self.assertEqual(connector.restore_calls, 1)
        self.assertEqual(connector._cpu_budget.reserved_bytes, expected_peak(plan))
        self.assertTrue(restore.is_alive())  # Simulated cancellation cannot free placement memory.
        stream.drain.set()
        restore.join(2)
        self.assertEqual(result, [connector.restore_result])
        self.assertEqual(connector._cpu_budget.reserved_bytes, 0)

    def test_small_restore_preserves_result_and_errors_release_budget(self):
        plan = Plan()
        connector = make_connector()
        self.assertIs(connector._load_hybrid_pages(lookup(plan), plan), connector.restore_result)
        self.assertEqual(connector._cpu_budget.reserved_bytes, 0)

        connector.restore_error = "bad payload"
        with self.assertRaisesRegex(RuntimeError, "ValueError: bad payload"):
            connector._load_hybrid_pages(lookup(plan), plan)
        self.assertEqual(connector._cpu_budget.reserved_bytes, 0)

    def test_failed_stream_fence_keeps_budget_charged(self):
        plan = Plan()
        connector = make_connector()
        stream = BlockingStream(fail=True)
        stream.drain.set()
        connector._load_stream = stream
        with self.assertRaisesRegex(RuntimeError, "stream fence failed"):
            connector._load_hybrid_pages(lookup(plan), plan)
        self.assertEqual(connector._cpu_budget.reserved_bytes, expected_peak(plan))

    def test_unsupported_roots_and_geometry_miss_without_invalidation(self):
        plan = Plan()
        connector = make_connector()
        for root_kind in ("page_delta", "legacy_chunks", "prefix_alias"):
            with self.subTest(root_kind=root_kind):
                self.assertFalse(connector._load_hybrid_pages(
                    lookup(plan, root_kind=root_kind), plan,
                ))
        self.assertFalse(connector._load_hybrid_pages(
            lookup(plan, encoded=ENCODED_BYTES + 1), plan,
        ))
        self.assertEqual(connector.restore_calls, 0)
        self.assertEqual(connector.invalidations, 0)
        self.assertEqual(connector._cpu_budget.reserved_bytes, 0)


class ConfigurationTests(unittest.TestCase):
    def test_valid_affordable_configuration_and_independent_cohorts(self):
        connector = make_connector()
        reservation = connector._cpu_budget.try_reserve(1)
        self.assertIsNotNone(reservation)
        reservation.release()
        plans = [Plan(span_tokens=8), Plan(span_tokens=24)]
        runnable, deferred, shared = connector._prepare_page_base_read_cohorts(plans)
        self.assertEqual(runnable, plans)
        self.assertEqual(deferred, [])
        self.assertEqual(shared, {})
        with self.assertRaisesRegex(RuntimeError, "does not support payload sweeps"):
            connector.sweep_integrity()

    def test_rejects_incompatible_or_nonpositive_configuration(self):
        base = dict(
            storage_mode="block_pages_v1",
            native_restore_enabled=False,
            streaming_snapshots_enabled=False,
            async_page_capture_enabled=False,
        )
        invalid = (
            {**base, "storage_mode": "chunks"},
            {**base, "native_restore_enabled": True},
            {**base, "streaming_snapshots_enabled": True},
            {**base, "async_page_capture_enabled": True},
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaisesRegex(
                RuntimeError, "requires synchronous block pages",
            ):
                make_connector(config=SimpleNamespace(**values))
        with self.assertRaisesRegex(RuntimeError, "requires recompute"):
            make_connector(failure_policy="fail")
        for cap, floor in ((0, 1), (1, 0), (-1, 1), (1, -1)):
            with self.subTest(cap=cap, floor=floor), self.assertRaises(ValueError):
                make_connector(cap=cap, floor=floor)


if __name__ == "__main__":
    unittest.main()
