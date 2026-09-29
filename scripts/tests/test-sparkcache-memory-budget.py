#!/usr/bin/env python3
"""Behavior tests for SparkCache host-memory admission; stdlib only."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import threading
import time
import unittest


REPO = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO / "third_party/sparkcache/spark_context_cache_memory_budget.py"
SPEC = importlib.util.spec_from_file_location("sparkcache_memory_budget", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
memory_budget = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(memory_budget)

MemoryBudget = memory_budget.MemoryBudget
read_mem_available = memory_budget.read_mem_available


class MemoryBudgetTests(unittest.TestCase):
    def test_concurrent_admission_cannot_oversubscribe(self):
        budget = MemoryBudget(300, 100, lambda: (time.sleep(0.01), 10_000)[1])
        start = threading.Barrier(9)
        results = []
        results_lock = threading.Lock()

        def reserve() -> None:
            start.wait()
            result = budget.try_reserve(100)
            with results_lock:
                results.append(result)

        threads = [threading.Thread(target=reserve) for _ in range(8)]
        for thread in threads:
            thread.start()
        start.wait()
        for thread in threads:
            thread.join()

        admitted = [result for result in results if result is not None]
        self.assertEqual(len(admitted), 3)
        self.assertEqual(budget.reserved_bytes, 300)
        for reservation in admitted:
            reservation.release()
        self.assertEqual(budget.reserved_bytes, 0)

    def test_store_and_cancelled_restore_hold_full_peaks_until_release(self):
        budget = MemoryBudget(100, 100, lambda: 1_000)
        store = budget.try_reserve(60)
        restore = budget.try_reserve(40)
        self.assertIsNotNone(store)
        self.assertIsNotNone(restore)
        self.assertEqual(budget.reserved_bytes, 100)
        self.assertIsNone(budget.try_reserve(1))

        restore.release()  # Simulate cancellation cleanup by the restore owner.
        self.assertEqual(budget.reserved_bytes, 60)
        replacement = budget.try_reserve(40)
        self.assertIsNotNone(replacement)
        self.assertEqual(budget.reserved_bytes, 100)

        store.release()
        replacement.release()
        self.assertEqual(budget.reserved_bytes, 0)

    def test_floor_unknown_reader_failure_and_oversize_fail_closed(self):
        budget = MemoryBudget(200, 100, lambda: 150)
        admitted = budget.try_reserve(50)
        self.assertIsNotNone(admitted)
        self.assertIsNone(budget.try_reserve(1))
        self.assertEqual(budget.reserved_bytes, 50)
        admitted.release()

        for reader in (lambda: None, lambda: "1000", lambda: -1, lambda: True):
            with self.subTest(reader=reader):
                closed = MemoryBudget(200, 100, reader)
                self.assertIsNone(closed.try_reserve(1))
                self.assertEqual(closed.reserved_bytes, 0)

        def failed_read():
            raise OSError("unreadable")

        closed = MemoryBudget(200, 100, failed_read)
        self.assertIsNone(closed.try_reserve(1))
        self.assertEqual(closed.reserved_bytes, 0)
        self.assertIsNone(MemoryBudget(10, 1, lambda: 1_000).try_reserve(11))

    def test_release_is_idempotent_and_context_manager_releases(self):
        budget = MemoryBudget(100, 1, lambda: 1_000)
        with self.assertRaises(AttributeError):
            budget.reserved_bytes = 1
        reservation = budget.try_reserve(40)
        self.assertIsNotNone(reservation)
        reservation.release()
        reservation.release()
        self.assertEqual(budget.reserved_bytes, 0)

        with budget.try_reserve(100) as held:
            self.assertIsNotNone(held)
            self.assertEqual(budget.reserved_bytes, 100)
        self.assertEqual(budget.reserved_bytes, 0)

    def test_constructor_and_request_validation(self):
        for name, args in (
            ("zero max", (0, 1)),
            ("negative max", (-1, 1)),
            ("zero floor", (1, 0)),
            ("negative floor", (1, -1)),
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                MemoryBudget(*args)
        for name, args in (
            ("bool max", (True, 1)),
            ("float max", (1.0, 1)),
            ("bool floor", (1, True)),
            ("float floor", (1, 1.0)),
        ):
            with self.subTest(name=name), self.assertRaises(TypeError):
                MemoryBudget(*args)
        with self.assertRaises(TypeError):
            MemoryBudget(1, 1, None)

        budget = MemoryBudget(100, 1, lambda: 1_000)
        for value in (0, -1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                budget.try_reserve(value)
        for value in (True, 1.0, "1"):
            with self.subTest(value=value), self.assertRaises(TypeError):
                budget.try_reserve(value)
        self.assertEqual(budget.reserved_bytes, 0)


class MemAvailableTests(unittest.TestCase):
    def read_text(self, contents: str):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "meminfo"
            path.write_text(contents, encoding="ascii")
            return read_mem_available(path)

    def test_parser_converts_proc_kib_to_bytes(self):
        self.assertEqual(
            self.read_text("MemTotal: 999 kB\nMemAvailable: 123456 kB\n"),
            123456 * 1024,
        )
        self.assertEqual(self.read_text("MemAvailable:\t0 kB\n"), 0)

    def test_parser_rejects_missing_malformed_or_duplicate_values(self):
        malformed = (
            "MemTotal: 123 kB\n",
            "MemAvailable: nope kB\n",
            "MemAvailable: -1 kB\n",
            "MemAvailable: 123 MB\n",
            "MemAvailable: 123 kB trailing\n",
            "MemAvailable: 1 kB\nMemAvailable: 2 kB\n",
        )
        for contents in malformed:
            with self.subTest(contents=contents):
                self.assertIsNone(self.read_text(contents))

    def test_reader_failure_returns_none(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(read_mem_available(Path(directory) / "missing"))
            self.assertIsNone(read_mem_available(directory))


if __name__ == "__main__":
    unittest.main()
