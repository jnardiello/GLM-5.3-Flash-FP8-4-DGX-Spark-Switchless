#!/usr/bin/env python3
"""Offline tests for bounded SparkCache page-snapshot disk I/O."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import uuid


REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "scripts/node/experiments/e03/sparkcache-ram-budget/stream_io.py"
OBJECT_BYTES = 32
SCHEMA = "sparkcache-page-snapshot-manifest/v2"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


def sha256(value):
    return hashlib.sha256(value).hexdigest()


class CacheFormatError(ValueError):
    pass


class CommitConflict(RuntimeError):
    pass


class StateRecord(str, Enum):
    TARGET_CKV = "target_ckv"
    LOGICAL_POSITIONS = "logical_positions"


@dataclass(frozen=True)
class CacheIdentity:
    name: str = "test"
    chunk_tokens: int = 16
    publication_schema: str = ""
    record_schema: tuple[str, ...] = ("target_ckv", "logical_positions")

    @property
    def required_records(self):
        return frozenset(StateRecord(value) for value in self.record_schema)

    def to_wire(self):
        return {
            "name": self.name,
            "chunk_tokens": self.chunk_tokens,
            "publication_schema": self.publication_schema,
            "record_schema": list(self.record_schema),
        }

    @property
    def storage_key(self):
        return sha256(canonical(self.to_wire()))


@dataclass(frozen=True)
class Receipt:
    manifest_digest: str
    committed_tokens: int
    encoded_bytes: int
    allocated_bytes_upper_bound: int
    publication: object | None = None


@dataclass
class Attempt:
    kind: str
    logical: int = 0
    staged: int = 0
    unique: int = 0
    deduplicated: int = 0
    outcome: str = ""

    def describe_payload(self, size, reused=0):
        assert reused == 0
        self.logical = size

    def record_staged(self, size):
        self.staged += size

    def record_unique(self, size):
        self.unique += size

    def record_deduplicated(self, size):
        self.deduplicated += size

    @property
    def has_activity(self):
        return bool(self.logical or self.staged or self.unique or self.deduplicated)


def tracked(kind):
    def decorate(method):
        def wrapped(store, *args, **kwargs):
            attempt = Attempt(kind)
            store._attempt = attempt
            try:
                result = method(store, *args, **kwargs)
            except Exception:
                attempt.outcome = "failed"
                store.attempts.append(attempt)
                raise
            finally:
                store._attempt = None
            attempt.outcome = "committed"
            store.attempts.append(attempt)
            return replace(result, publication=attempt)
        return wrapped
    return decorate


class RootGuard(AbstractContextManager):
    def __init__(self, root, *, shared, blocking):
        self.root, self.shared, self.blocking = root, shared, blocking

    def __enter__(self):
        Path(self.root).mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, *args):
        return None


class ManifestStore:
    def __init__(self, root):
        self.root = Path(root)
        self.attempts = []
        self._attempt = None

    def _active_publication(self):
        if self._attempt is None:
            raise RuntimeError("publication telemetry scope is missing")
        return self._attempt

    def _manifest_path(self, identity, context_digest):
        return self.root / "manifests" / identity.storage_key / f"{context_digest}.json"


def ensure_directory(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def publish_immutable(path, payload):
    path = Path(path)
    ensure_directory(path.parent)
    temporary = path.with_name(f".{path.name}.writing-{uuid.uuid4().hex}")
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(fd, payload)
            os.fsync(fd)
        finally:
            os.close(fd)
        try:
            os.link(temporary, path)
        except FileExistsError:
            with path.open("rb") as stream:
                if stream.read() != payload:
                    raise CommitConflict(path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        fsync_directory(path.parent)


def validate_digest(value, field):
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError(f"{field} is not a digest")


def validate_root(manifest, *, identity, context_digest):
    if not isinstance(manifest, dict):
        raise CacheFormatError("manifest is not an object")
    authenticated = dict(manifest)
    metadata = authenticated.pop("metadata_sha256", None)
    if metadata != sha256(canonical(authenticated)):
        raise CacheFormatError("metadata checksum mismatch")
    if (
        manifest.get("schema") != SCHEMA
        or manifest.get("format_abi") != 1
        or manifest.get("storage_mode") != "block_pages_v1"
        or manifest.get("identity") != identity.to_wire()
        or manifest.get("context_digest") != context_digest
    ):
        raise CacheFormatError("manifest identity differs")
    validate_digest(manifest.get("snapshot_sha256"), "snapshot_sha256")
    descriptors = manifest.get("snapshot_objects")
    if not isinstance(descriptors, list) or not descriptors:
        raise CacheFormatError("snapshot objects are invalid")
    expected = 0
    for index, descriptor in enumerate(descriptors):
        size = descriptor.get("bytes")
        if (
            set(descriptor) != {"sha256", "bytes", "encoded_start", "encoded_end"}
            or descriptor.get("encoded_start") != expected
            or descriptor.get("encoded_end") != expected + size
            or size <= 0
            or size > OBJECT_BYTES
            or (index < len(descriptors) - 1 and size != OBJECT_BYTES)
        ):
            raise CacheFormatError("snapshot object geometry differs")
        validate_digest(descriptor["sha256"], "object sha256")
        expected += size
    if expected != manifest.get("snapshot_encoded_bytes"):
        raise CacheFormatError("snapshot coverage differs")
    return tuple(descriptors)


def install_fake_upstream():
    sparkcache = ModuleType("sparkcache")
    sparkcache.__path__ = []
    persistent = ModuleType("sparkcache.persistent_context_cache")
    persistent.__path__ = []
    manifest = ModuleType("sparkcache.persistent_context_cache.cache_manifest")
    manifest.__dict__.update({
        "FORMAT_ABI": 1,
        "CacheFormatError": CacheFormatError,
        "CacheIdentity": CacheIdentity,
        "CommitConflict": CommitConflict,
        "CommitReceipt": Receipt,
        "ManifestStore": ManifestStore,
        "StateRecord": StateRecord,
        "_PAGE_SNAPSHOT_MANIFEST_SCHEMA": SCHEMA,
        "_PAGE_SNAPSHOT_OBJECT_BYTES": OBJECT_BYTES,
        "_RootGuard": RootGuard,
        "_canonical_json": canonical,
        "_ensure_durable_directory": ensure_directory,
        "_fsync_directory": fsync_directory,
        "_is_page_snapshot_root": lambda value: (
            isinstance(value, dict) and value.get("schema") == SCHEMA
        ),
        "_publish_immutable": publish_immutable,
        "_sha256": sha256,
        "_tracked_publication": tracked,
        "_validate_digest": validate_digest,
        "_validate_page_snapshot_root": validate_root,
    })
    hybrid = ModuleType("sparkcache.spark_context_cache_hybrid")
    hybrid.page_snapshot_encoded_size = lambda layout, counts: layout.expected_bytes
    modules = {
        "sparkcache": sparkcache,
        "sparkcache.persistent_context_cache": persistent,
        "sparkcache.persistent_context_cache.cache_manifest": manifest,
        "sparkcache.spark_context_cache_hybrid": hybrid,
    }
    sys.modules.update(modules)
    return modules


FAKE_MODULES = install_fake_upstream()


def load_source(name):
    spec = importlib.util.spec_from_file_location(name, SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


stream_io = load_source("sparkcache_stream_io_test")


def append_snapshot(module, root, payload):
    snapshot = module.DiskSnapshot(root, len(payload))
    for start in range(0, len(payload), 11):
        snapshot.append(payload[start : start + 11])
    snapshot.seal()
    return snapshot


def committed(module, root, payload):
    identity = module._StreamCacheIdentity()
    context_digest = "a" * 64
    store = module.StreamingManifestStore(root)
    snapshot = append_snapshot(module, root, payload)
    receipt = store.commit_page_snapshot(
        identity=identity,
        context_digest=context_digest,
        span_tokens=32,
        snapshot=snapshot,
    )
    manifest_path = store._manifest_path(identity, context_digest)
    manifest = json.loads(manifest_path.read_bytes())
    lookup = SimpleNamespace(
        is_hit=True,
        reason="hit",
        root_kind="page_snapshot",
        _manifest=manifest,
    )
    layout = SimpleNamespace(expected_bytes=len(payload))
    return store, identity, context_digest, receipt, manifest_path, lookup, layout


class DiskSnapshotTests(unittest.TestCase):
    def test_private_exact_staging_and_idempotent_release(self):
        self.assertEqual(stream_io.STREAM_CHUNK_BYTES, 8 << 20)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = stream_io.DiskSnapshot(root, 12)
            self.assertEqual(stat.S_IMODE(os.fstat(snapshot._stream.fileno()).st_mode), 0o600)
            self.assertEqual(list(root.iterdir()), [])
            snapshot.append(memoryview(b"hello "))
            snapshot.append(bytearray(b"world!"))
            snapshot.seal()
            self.assertEqual(snapshot.total_bytes, 12)
            self.assertEqual(snapshot.read_range(3, 9), b"lo wor")
            with self.assertRaises(ValueError):
                snapshot.append(b"x")
            snapshot.release()
            snapshot.release()
            self.assertTrue(snapshot._stream.closed)

    def test_size_mismatch_aborts_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = stream_io.DiskSnapshot(Path(directory), 4)
            snapshot.append(b"abc")
            with self.assertRaisesRegex(ValueError, "size differs"):
                snapshot.seal()
            self.assertTrue(snapshot._stream.closed)


class StoreAndReaderTests(unittest.TestCase):
    PAYLOAD = b"SPHP1-header:" + bytes(range(73))

    def test_exact_manifest_boundary_crossing_checksums_and_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            result = committed(stream_io, Path(directory), self.PAYLOAD)
            store, _identity, _digest, receipt, _path, lookup, layout = result
            manifest = lookup._manifest
            self.assertEqual(
                [item["bytes"] for item in manifest["snapshot_objects"]],
                [32, 32, len(self.PAYLOAD) - 64],
            )
            self.assertEqual(manifest["snapshot_sha256"], sha256(self.PAYLOAD))
            authenticated = dict(manifest)
            metadata = authenticated.pop("metadata_sha256")
            self.assertEqual(metadata, sha256(canonical(authenticated)))
            self.assertEqual(receipt.publication.logical, len(self.PAYLOAD))
            self.assertEqual(receipt.publication.staged, len(self.PAYLOAD))
            for descriptor in manifest["snapshot_objects"]:
                path = store.root / "chunks" / f"{descriptor['sha256']}.spcc"
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

            with store.open_snapshot(
                lookup,
                layout=layout,
                result_block_counts=(2,),
                result_boundary_tokens=32,
            ) as reader:
                parts = [reader.read_exact(7), reader.read_exact(30)]
                remaining = len(self.PAYLOAD) - 37
                parts.extend(reader.read_exact(min(13, remaining)) for _ in range(remaining // 13))
                consumed = sum(map(len, parts))
                parts.append(reader.read_exact(len(self.PAYLOAD) - consumed))
            self.assertEqual(b"".join(parts), self.PAYLOAD)

    def test_corruption_and_full_checksum_failure_close_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            store, _, _, _, _, lookup, layout = committed(
                stream_io, Path(directory), self.PAYLOAD
            )
            descriptor = lookup._manifest["snapshot_objects"][0]
            path = store.root / "chunks" / f"{descriptor['sha256']}.spcc"
            with path.open("r+b") as output:
                output.write(b"X")
                output.flush()
                os.fsync(output.fileno())
            reader = store.open_snapshot(
                lookup, layout=layout, result_block_counts=(2,), result_boundary_tokens=32
            )
            with self.assertRaisesRegex(CacheFormatError, "object checksum"):
                reader.read_exact(32)
            self.assertIsNone(reader._fd)

        with tempfile.TemporaryDirectory() as directory:
            store, _, _, _, _, lookup, layout = committed(
                stream_io, Path(directory), self.PAYLOAD
            )
            altered = dict(lookup._manifest)
            altered["snapshot_sha256"] = "f" * 64
            altered["metadata_sha256"] = sha256(
                canonical({key: value for key, value in altered.items() if key != "metadata_sha256"})
            )
            lookup._manifest = altered
            reader = store.open_snapshot(
                lookup, layout=layout, result_block_counts=(2,), result_boundary_tokens=32
            )
            reader.read_exact(len(self.PAYLOAD))
            with self.assertRaisesRegex(CacheFormatError, "payload checksum"):
                reader.finish()

    def test_truncated_and_enlarged_files_fail_preflight(self):
        for change in ("truncate", "enlarge"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                store, _, _, _, _, lookup, layout = committed(
                    stream_io, Path(directory), self.PAYLOAD
                )
                descriptor = lookup._manifest["snapshot_objects"][0]
                path = store.root / "chunks" / f"{descriptor['sha256']}.spcc"
                with path.open("r+b") as output:
                    if change == "truncate":
                        output.truncate(descriptor["bytes"] - 1)
                    else:
                        output.seek(0, os.SEEK_END)
                        output.write(b"x")
                    output.flush()
                    os.fsync(output.fileno())
                with self.assertRaisesRegex(CacheFormatError, "geometry differs"):
                    store.open_snapshot(
                        lookup,
                        layout=layout,
                        result_block_counts=(2,),
                        result_boundary_tokens=32,
                    )

    def test_metadata_geometry_and_incomplete_read_fail_before_success(self):
        with tempfile.TemporaryDirectory() as directory:
            store, _, _, _, _, lookup, layout = committed(
                stream_io, Path(directory), self.PAYLOAD
            )
            altered = dict(lookup._manifest)
            altered["committed_tokens"] = 48
            lookup._manifest = altered
            with self.assertRaisesRegex(CacheFormatError, "metadata checksum"):
                store.open_snapshot(
                    lookup, layout=layout, result_block_counts=(2,), result_boundary_tokens=32
                )

        with tempfile.TemporaryDirectory() as directory:
            store, _, _, _, _, lookup, layout = committed(
                stream_io, Path(directory), self.PAYLOAD
            )
            with self.assertRaisesRegex(CacheFormatError, "incomplete"):
                with store.open_snapshot(
                    lookup, layout=layout, result_block_counts=(2,), result_boundary_tokens=32
                ) as reader:
                    reader.read_exact(1)

    def test_dedup_repair_and_manifest_publication_failure_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store, identity, digest, _, _, lookup, _ = committed(
                stream_io, root, self.PAYLOAD
            )
            second = append_snapshot(stream_io, root, self.PAYLOAD)
            receipt = store.commit_page_snapshot(
                identity=identity, context_digest=digest, span_tokens=32, snapshot=second
            )
            self.assertEqual(receipt.publication.deduplicated, len(self.PAYLOAD))
            descriptor = lookup._manifest["snapshot_objects"][0]
            object_path = root / "chunks" / f"{descriptor['sha256']}.spcc"
            object_path.write_bytes(b"!" * descriptor["bytes"])
            third = append_snapshot(stream_io, root, self.PAYLOAD)
            repaired = store.commit_page_snapshot(
                identity=identity, context_digest=digest, span_tokens=32, snapshot=third
            )
            self.assertGreater(repaired.publication.unique, 0)
            self.assertEqual(object_path.read_bytes(), self.PAYLOAD[:OBJECT_BYTES])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = stream_io.StreamingManifestStore(root)
            identity = CacheIdentity()
            snapshot = append_snapshot(stream_io, root, self.PAYLOAD)
            manifest_path = store._manifest_path(identity, "b" * 64)
            with patch.object(
                stream_io, "_stream_publish_immutable", side_effect=OSError("barrier failed")
            ), self.assertRaisesRegex(OSError, "barrier failed"):
                store.commit_page_snapshot(
                    identity=identity,
                    context_digest="b" * 64,
                    span_tokens=32,
                    snapshot=snapshot,
                )
            self.assertTrue(snapshot._stream.closed)
            self.assertFalse(manifest_path.exists())
            self.assertEqual(store.attempts[-1].outcome, "failed")

    def test_object_setup_failures_close_and_unlink_temporary_file(self):
        for failure in ("permissions", "telemetry"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                store = stream_io.StreamingManifestStore(root)
                snapshot = append_snapshot(stream_io, root, self.PAYLOAD)
                opened = []
                create = stream_io._stream_tempfile.NamedTemporaryFile

                def remember(*args, **kwargs):
                    temporary = create(*args, **kwargs)
                    opened.append((temporary, temporary.fileno(), Path(temporary.name)))
                    return temporary

                target, name = ((stream_io._stream_os, "fchmod") if failure == "permissions"
                                else (Attempt, "record_staged"))
                with patch.object(stream_io._stream_tempfile, "NamedTemporaryFile", side_effect=remember), \
                     patch.object(target, name, side_effect=OSError("setup failed")), \
                     self.assertRaisesRegex(OSError, "setup failed"):
                    store.commit_page_snapshot(
                        identity=CacheIdentity(), context_digest="c" * 64,
                        span_tokens=32, snapshot=snapshot,
                    )
                self.assertEqual(len(opened), 1)
                temporary, descriptor, path = opened[0]
                self.assertTrue(temporary.closed)
                self.assertFalse(path.exists())
                with self.assertRaises(OSError):
                    os.fstat(descriptor)
                self.assertTrue(snapshot._stream.closed)
                self.assertFalse(list(root.rglob("*.spcc")))


def load_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(
    os.environ.get("SPARKCACHE_PINNED_MANIFEST"),
    "set SPARKCACHE_PINNED_MANIFEST and its two dependency paths for pinned integration",
)
class PinnedUpstreamContractTests(unittest.TestCase):
    def test_real_guard_telemetry_validator_and_receipt(self):
        names = tuple(name for name in sys.modules if name == "sparkcache" or name.startswith("sparkcache."))
        saved = {name: sys.modules[name] for name in names}
        for name in names:
            del sys.modules[name]
        try:
            sparkcache = ModuleType("sparkcache")
            sparkcache.__path__ = []
            persistent = ModuleType("sparkcache.persistent_context_cache")
            persistent.__path__ = []
            sys.modules["sparkcache"] = sparkcache
            sys.modules["sparkcache.persistent_context_cache"] = persistent
            load_module_from_path(
                "sparkcache.page_base_read_flights",
                Path(os.environ["SPARKCACHE_PINNED_FLIGHTS"]),
            )
            telemetry = load_module_from_path(
                "sparkcache.publication_telemetry",
                Path(os.environ["SPARKCACHE_PINNED_TELEMETRY"]),
            )
            manifest = load_module_from_path(
                "sparkcache.persistent_context_cache.cache_manifest",
                Path(os.environ["SPARKCACHE_PINNED_MANIFEST"]),
            )
            payload = b"actual-upstream-contract"
            hybrid = ModuleType("sparkcache.spark_context_cache_hybrid")
            hybrid.page_snapshot_encoded_size = lambda layout, counts: len(payload)
            sys.modules["sparkcache.spark_context_cache_hybrid"] = hybrid
            actual = load_source("sparkcache_stream_io_actual_test")
            identity = manifest.CacheIdentity(
                target_checkpoint="1" * 64,
                draft_checkpoint="2" * 64,
                quantization_layout="q",
                rope_layout="r",
                tp_degree=1,
                dcp_degree=1,
                record_schema=("target_ckv", "logical_positions"),
            )
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                store = actual.StreamingManifestStore(root)
                snapshot = append_snapshot(actual, root, payload)
                receipt = store.commit_page_snapshot(
                    identity=identity,
                    context_digest="3" * 64,
                    span_tokens=256,
                    snapshot=snapshot,
                )
                self.assertIsInstance(receipt, manifest.CommitReceipt)
                self.assertIsInstance(receipt.publication, telemetry.PublicationByteReceipt)
                self.assertEqual(receipt.publication.outcome, "committed")
                lookup = store.lookup(identity, "3" * 64, verify_chunks=False)
                with store.open_snapshot(
                    lookup,
                    layout=object(),
                    result_block_counts=(1,),
                    result_boundary_tokens=256,
                ) as reader:
                    self.assertEqual(reader.read_exact(len(payload)), payload)
                counters = store.publication_telemetry_snapshot()
                self.assertEqual(counters.committed_publications, 1)
                self.assertEqual(counters.logical_payload_bytes, len(payload))
        finally:
            for name in tuple(
                name for name in sys.modules
                if name == "sparkcache" or name.startswith("sparkcache.")
            ):
                del sys.modules[name]
            sys.modules.update(saved)


if __name__ == "__main__":
    unittest.main()
