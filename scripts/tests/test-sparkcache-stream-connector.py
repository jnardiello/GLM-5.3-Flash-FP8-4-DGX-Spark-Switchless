#!/usr/bin/env python3
"""Stdlib byte-order and bounded-transfer tests for SparkCache disk streaming."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import importlib.util
import logging
from pathlib import Path
import sys
import threading
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[2]
CANDIDATE = REPO / "scripts/node/experiments/e03/sparkcache-ram-budget"
CODEC_PATH = REPO / "third_party/sparkcache/spark_context_cache_hybrid.py"
MEMORY_PATH = REPO / "third_party/sparkcache/spark_context_cache_memory_budget.py"
TEST_CHUNK_BYTES = 32


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


codec = load_module("sparkcache_stream_test_codec", CODEC_PATH)
memory = load_module("sparkcache_stream_test_memory", MEMORY_PATH)


@dataclass(frozen=True)
class Plan:
    span_tokens: int
    group_block_ids: tuple[tuple[int, ...], ...]
    recurrent_boundary_blocks: tuple[int, ...] = ()
    base_context_digest: str = ""
    base_span_tokens: int = 0


@dataclass
class HybridSnapshot:
    plan: Plan
    rank: int
    identity: object
    positions: tuple[int, ...]
    encoded_pages: object
    block_counts: tuple[int, ...]


class MockDiskSnapshot:
    instances = []

    def __init__(self, root, expected_bytes: int) -> None:
        del root
        self.expected_bytes = expected_bytes
        self.records = []
        self.body = bytearray()
        self.sealed = False
        self.released = False
        self.__class__.instances.append(self)

    def append(self, payload: bytes) -> None:
        if self.sealed:
            raise RuntimeError("append after seal")
        value = bytes(payload)
        self.records.append(value)
        self.body.extend(value)

    def seal(self) -> None:
        if len(self.body) != self.expected_bytes:
            raise RuntimeError("streamed snapshot length differs")
        self.sealed = True

    def release(self) -> None:
        self.released = True


class RawRows:
    def __init__(self, tensor: "FakeTensor") -> None:
        self.tensor = tensor
        self.name = tensor.name
        self.shape = (len(tensor.rows), len(tensor.rows[0]))
        self.device = "fake-device"

    def read(self, block: int, offset: int, size: int) -> bytes:
        return bytes(self.tensor.rows[block][offset:offset + size])

    def write(self, block: int, offset: int, payload: bytes) -> None:
        self.tensor.rows[block][offset:offset + len(payload)] = payload


class RawByteView:
    def __init__(self, tensor: "FakeTensor") -> None:
        self.tensor = tensor

    def view(self, rows: int, columns: int) -> RawRows:
        if rows != len(self.tensor.rows) or columns != -1:
            raise AssertionError("unexpected byte-view geometry")
        self.tensor.view_calls.append((rows, columns))
        return RawRows(self.tensor)


class FakeTensor:
    """A page pool whose only whole-pool operation is an aliasing byte view."""

    def __init__(self, name: str, rows: list[bytes]) -> None:
        self.name = name
        self.rows = [bytearray(row) for row in rows]
        self.shape = (len(rows), len(rows[0]))
        self.view_calls = []

    def view(self, dtype) -> RawByteView:
        if dtype is not FAKE_TORCH.uint8:
            raise AssertionError("only a uint8 alias is allowed")
        self.view_calls.append((dtype,))
        return RawByteView(self)

    def contiguous(self):
        raise AssertionError("streaming copied the complete page pool")

    def reshape(self, *args):
        raise AssertionError(f"streaming reshaped the complete page pool: {args}")

    def cpu(self):
        raise AssertionError("streaming moved the complete page pool to CPU")

    def numpy(self):
        raise AssertionError("streaming converted the complete page pool")


FAKE_TORCH = SimpleNamespace(uint8=object())


class MockReader:
    def __init__(self, body: bytes, corrupt_at_read: int | None = None) -> None:
        self.body = body
        self.corrupt_at_read = corrupt_at_read
        self.offset = 0
        self.read_calls = 0
        self.read_sizes = []
        self.finish_calls = 0
        self.verified = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read_exact(self, size: int) -> bytes:
        self.read_calls += 1
        self.read_sizes.append(size)
        if self.corrupt_at_read == self.read_calls:
            raise RuntimeError("stream checksum mismatch")
        end = self.offset + size
        if end > len(self.body):
            raise RuntimeError("stream snapshot is truncated")
        payload = self.body[self.offset:end]
        self.offset = end
        return payload

    def finish(self) -> None:
        self.finish_calls += 1
        if self.offset != len(self.body):
            raise RuntimeError("stream snapshot has trailing bytes")
        self.verified = True


class MockStore:
    def __init__(self, body: bytes, *, corrupt_at_read: int | None = None) -> None:
        self.body = body
        self.corrupt_at_read = corrupt_at_read
        self.reader = None

    def open_snapshot(self, lookup, **kwargs):
        del lookup, kwargs
        self.reader = MockReader(self.body, self.corrupt_at_read)
        return self.reader


class BaseConnector:
    def _select_group_blocks_for_span(
        self, groups, span_tokens, recurrent_boundary_blocks=None,
    ):
        del span_tokens, recurrent_boundary_blocks
        return groups

    def _worker_rank(self):
        return 0

    def _identity(self, rank):
        return ("identity", rank)

    @contextmanager
    def _load_write_context(self):
        yield


def load_candidate():
    sparkcache = ModuleType("sparkcache")
    sparkcache.__path__ = []
    candidate = ModuleType("sparkcache_stream_bounded_test")
    candidate.__dict__.update({
        "__name__": candidate.__name__,
        "dataclass": dataclass,
        "MemoryBudget": memory.MemoryBudget,
        "MemoryReservation": memory.MemoryReservation,
        "SparkContextCacheConnector": BaseConnector,
        "parse_connector_config": lambda *args: None,
        "threading": threading,
        "time": time,
        "logger": logging.getLogger(candidate.__name__),
        "STREAM_CHUNK_BYTES": TEST_CHUNK_BYTES,
        "DiskSnapshot": MockDiskSnapshot,
        "_HybridStoreSnapshot": HybridSnapshot,
        "torch": FAKE_TORCH,
        "chunk_count": lambda span, chunk: (span + chunk - 1) // chunk,
    })
    with patch.dict(sys.modules, {
        "sparkcache": sparkcache,
        "sparkcache.spark_context_cache_hybrid": codec,
        candidate.__name__: candidate,
    }):
        source = (CANDIDATE / "bounded_connector.py").read_text()
        exec(compile(source, str(CANDIDATE / "bounded_connector.py"), "exec"),
             candidate.__dict__)
    return candidate.SparkContextCacheConnector


Connector = load_candidate()


def row_bytes(layer: str, block: int, size: int) -> bytes:
    seed = sum(layer.encode()) + 29 * block
    return bytes((seed + offset) % 256 for offset in range(size))


def tensors_for(layout, *, fill: bool) -> dict[str, FakeTensor]:
    result = {}
    for group in layout.groups:
        for layer in group.layers:
            rows = [
                row_bytes(layer.name, block, layer.bytes_per_page)
                if fill else bytes(layer.bytes_per_page)
                for block in range(5)
            ]
            result[layer.name] = FakeTensor(layer.name, rows)
    return result


def connector(layout, tensors, *, store=None):
    instance = Connector.__new__(Connector)
    instance._page_layout = layout
    instance._layer_tensors = tensors
    instance._root = Path("/unused")
    instance._store = store
    instance._chunk_tokens = 4
    instance.capture_records = []
    instance.place_records = []

    def capture(rows, blocks, offset, size, whole_pages):
        if whole_pages:
            payload = b"".join(rows.read(block, offset, size) for block in blocks)
        else:
            payload = rows.read(blocks[0], offset, size)
        instance.capture_records.append(
            (rows.name, tuple(blocks), offset, size, whole_pages, len(payload))
        )
        return payload

    def place(rows, blocks, offset, size, whole_pages, payload):
        if len(payload) != len(blocks) * size:
            raise RuntimeError("streaming restore segment length differs")
        if whole_pages:
            for index, block in enumerate(blocks):
                rows.write(block, offset, payload[index * size:(index + 1) * size])
        else:
            rows.write(blocks[0], offset, payload)
        instance.place_records.append(
            (rows.name, tuple(blocks), offset, size, whole_pages, len(payload))
        )

    instance._capture_segment = capture
    instance._place_segment = place
    return instance


class StreamConnectorTests(unittest.TestCase):
    def setUp(self):
        small = codec.PageGroup(block_size=4, layers=(
            codec.PageLayer("a", "u8", (5,), 5),
            codec.PageLayer("b", "u8", (20,), 20),
        ))
        large = codec.PageGroup(block_size=8, layers=(
            codec.PageLayer("c", "u8", (75,), 75),
        ))
        self.layout = codec.PageLayout((small, large))
        self.plan = Plan(span_tokens=12, group_block_ids=((3, 0, 2), (4, 1)))
        self.counts = (3, 2)
        self.encoded = codec.page_snapshot_encoded_size(self.layout, self.counts)
        self.source_tensors = tensors_for(self.layout, fill=True)
        self.source = connector(self.layout, self.source_tensors)
        self.ticket = SimpleNamespace(resource=None)
        MockDiskSnapshot.instances.clear()
        self.snapshot = self.source._capture_stream_snapshot(
            self.plan, self.counts, self.encoded, self.ticket,
        )
        self.spool = self.ticket.resource

    def expected_payloads(self):
        result = {}
        for group, blocks in zip(self.layout.groups, self.plan.group_block_ids):
            for layer in group.layers:
                result[layer.name] = b"".join(
                    self.source_tensors[layer.name].rows[block] for block in blocks
                )
        return result

    def test_capture_matches_codec_for_heterogeneous_nonmonotonic_pages(self):
        expected = codec.encode_page_snapshot(
            self.layout, self.counts, self.expected_payloads(),
        )
        self.assertEqual(bytes(self.spool.body), expected)
        self.assertTrue(self.spool.sealed)
        self.assertIs(self.snapshot.encoded_pages, self.spool)
        self.assertEqual(self.snapshot.block_counts, self.counts)
        self.assertEqual(self.snapshot.positions, tuple(range(self.plan.span_tokens)))

        payload_sizes = [record[-1] for record in self.source.capture_records]
        self.assertTrue(payload_sizes)
        self.assertLessEqual(max(payload_sizes), TEST_CHUNK_BYTES)
        large_segments = [record for record in self.source.capture_records
                          if record[0] == "c"]
        self.assertEqual([record[3] for record in large_segments],
                         [32, 32, 11, 32, 32, 11])
        self.assertTrue(all(not record[4] for record in large_segments))

        for tensor in self.source_tensors.values():
            self.assertEqual(len(tensor.view_calls), 2)

    def test_restore_reconstructs_selected_pages_and_verifies_after_placement(self):
        destination_tensors = tensors_for(self.layout, fill=False)
        store = MockStore(bytes(self.spool.body))
        destination = connector(self.layout, destination_tensors, store=store)
        restored = destination._restore_stream_snapshot(
            object(), self.plan, self.counts, self.encoded, None,
        )

        self.assertTrue(restored)
        self.assertTrue(store.reader.verified)
        self.assertEqual(store.reader.finish_calls, 1)
        self.assertEqual(store.reader.offset, self.encoded)
        self.assertTrue(destination.place_records)
        self.assertLessEqual(max(record[-1] for record in destination.place_records),
                             TEST_CHUNK_BYTES)
        for group, blocks in zip(self.layout.groups, self.plan.group_block_ids):
            for layer in group.layers:
                source = self.source_tensors[layer.name]
                destination_tensor = destination_tensors[layer.name]
                for block in blocks:
                    self.assertEqual(destination_tensor.rows[block], source.rows[block])
                for block in set(range(5)) - set(blocks):
                    self.assertEqual(destination_tensor.rows[block],
                                     bytearray(layer.bytes_per_page))

    def test_corruption_after_partial_placement_raises_without_verification(self):
        destination_tensors = tensors_for(self.layout, fill=False)
        # Header and first payload read succeed; the next payload detects corruption.
        store = MockStore(bytes(self.spool.body), corrupt_at_read=3)
        destination = connector(self.layout, destination_tensors, store=store)
        with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
            destination._restore_stream_snapshot(
                object(), self.plan, self.counts, self.encoded, None,
            )

        self.assertEqual(len(destination.place_records), 1)
        self.assertEqual(store.reader.finish_calls, 0)
        self.assertFalse(store.reader.verified)


if __name__ == "__main__":
    unittest.main()
