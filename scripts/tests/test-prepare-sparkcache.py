#!/usr/bin/env python3
"""Test cache preparation with small independent fixtures, without operator sources."""

from __future__ import annotations

import ast
import contextlib
import gc
import hashlib
import importlib.util
import io
import stat
import tempfile
import weakref
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("prepare_cache", REPO / "scripts/prepare-sparkcache.py")
assert spec and spec.loader
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)

CONNECTOR = '''from __future__ import annotations
import time

class Saver:
    def _store_worker_main(self) -> None:
        while True:
            snapshot = self._store_queue.get()
            if snapshot is None:
                return
            commit_started = time.perf_counter()
            if snapshot.skip:
                continue
            payload_alias = snapshot.payload
            self.committed.append(snapshot.label)
            self.observed.append(weakref.ref(payload_alias))

    def _publish_row_prefix_aliases(
        self,
    ):
        pass
'''
ENCODER = '''def encode(layout, counts, parts):
    return encode_page_snapshot_header(layout, counts) + b"".join(parts)
'''


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Payload:
    pass


class Snapshot:
    def __init__(self, label: str, skip: bool):
        self.label, self.skip, self.payload = label, skip, Payload()


class CheckingQueue:
    """Observe liveness at the next queue wait, keeping only weak references."""

    def __init__(self, items: list[tuple[str, bool]]):
        self.items = iter(items)
        self.references: list[weakref.ReferenceType] = []
        self.calls = 0

    def get(self):
        self.calls += 1
        gc.collect()
        assert all(ref() is None for ref in self.references), "previous snapshot retained at queue wait"
        item = next(self.items, None)
        if item is None:
            return None
        snapshot = Snapshot(*item)
        self.references.extend((weakref.ref(snapshot), weakref.ref(snapshot.payload)))
        return snapshot


def saver(source: str, items: list[tuple[str, bool]]):
    namespace = {"weakref": weakref}
    exec(compile(source, "synthetic-connector", "exec"), namespace)
    instance = namespace["Saver"]()
    instance._store_queue = CheckingQueue(items)
    instance.committed, instance.observed = [], []
    return instance


fixed_connector = prepare.transform_connector(CONNECTOR)
fixed_encoder = prepare.transform_encoder(ENCODER)
original = saver(CONNECTOR, [("one", False), ("two", False)])
try:
    original._store_worker_main()
except AssertionError as error:
    assert "retained at queue wait" in str(error)
else:
    raise AssertionError("fixture failed to reproduce the original snapshot retention")

for items, commits in (([], []), ([("skip", True), ("commit", False)], ["commit"]),
                       ([("one", False), ("two", False)], ["one", "two"])):
    fixed = saver(fixed_connector, items)
    calls = []
    commit = fixed._commit_store_snapshot

    def observe(snapshot):
        calls.append(snapshot.label)
        commit(snapshot)

    fixed._commit_store_snapshot = observe
    fixed._store_worker_main()
    assert calls == [label for label, _ in items]
    assert fixed.committed == commits
    assert fixed._store_queue.calls == len(items) + 1
    assert all(ref() is None for ref in fixed._store_queue.references + fixed.observed)

copies = []


class Header(bytes):
    def __add__(self, payload):
        copies.append(len(payload))
        return super().__add__(payload)


def header(layout, counts):
    return Header(f"{layout}:{counts}:".encode())


old_namespace = {"encode_page_snapshot_header": header}
new_namespace = {"encode_page_snapshot_header": header}
exec(ENCODER, old_namespace)
exec(fixed_encoder, new_namespace)
for parts in ([], [b"one"], [b"a", b"", b"bc"], [memoryview(b"bytes"), b"tail"]):
    copies.clear()
    expected = old_namespace["encode"]("layout", len(parts), parts)
    assert copies == [sum(len(part) for part in parts)]
    copies.clear()
    actual = new_namespace["encode"]("layout", len(parts), parts)
    assert actual == expected and copies == []
tree = ast.parse(fixed_encoder)
assert sum(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
           and node.func.attr == "join" for node in ast.walk(tree)) == 1
assert not any(isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add)
               for node in ast.walk(tree))

# Synthetic pins exercise the transactional checks without shipping private sources.
# The transformations above independently verify behavior before these hashes are used.
original_pins, prepared_pins = dict(prepare.ORIGINAL), dict(prepare.PREPARED)
try:
    prepare.ORIGINAL = {prepare.CONNECTOR: digest(CONNECTOR.encode()),
                        prepare.ENCODER: digest(ENCODER.encode())}
    prepare.PREPARED = {prepare.CONNECTOR: digest(fixed_connector.encode()),
                        prepare.ENCODER: digest(fixed_encoder.encode())}
    with tempfile.TemporaryDirectory(prefix="tp4-cache-preparation.") as temp:
        root = Path(temp)
        connector, encoder = root / "connector.py", root / "encoder.py"
        connector.write_text(CONNECTOR, encoding="utf-8")
        encoder.write_text(ENCODER, encoding="utf-8")

        def invoke(output: Path):
            with contextlib.redirect_stdout(io.StringIO()):
                prepare.prepare(connector, encoder, output)

        for path, source in ((connector, CONNECTOR), (encoder, ENCODER)):
            path.write_text(source + "# mismatch\n", encoding="utf-8")
            output = root / (path.stem + "-bad-input")
            try:
                invoke(output)
            except ValueError as error:
                assert "SHA-256 mismatch" in str(error) and not output.exists()
            else:
                raise AssertionError("mismatched original accepted")
            path.write_text(source, encoding="utf-8")

        for name in (prepare.CONNECTOR, prepare.ENCODER):
            saved = prepare.PREPARED[name]
            prepare.PREPARED[name] = "0" * 64
            output = root / (name + "-bad-result")
            try:
                invoke(output)
            except ValueError as error:
                assert "prepared" in str(error) and not output.exists()
            else:
                raise AssertionError("mismatched transformed result accepted")
            prepare.PREPARED[name] = saved

        output = root / "prepared"
        invoke(output)
        expected_outputs = {prepare.CONNECTOR: fixed_connector.encode(),
                            prepare.ENCODER: fixed_encoder.encode(),
                            prepare.ROLLBACK: CONNECTOR.encode()}
        before = {}
        for name, data in expected_outputs.items():
            path = output / name
            assert path.read_bytes() == data
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
            before[name] = (path.stat().st_ino, path.stat().st_mtime_ns)
        invoke(output)
        assert before == {name: ((output / name).stat().st_ino, (output / name).stat().st_mtime_ns)
                          for name in expected_outputs}

        conflicting = root / "conflicting"
        conflicting.mkdir()
        (conflicting / prepare.ROLLBACK).write_text("unrelated content", encoding="utf-8")
        try:
            invoke(conflicting)
        except ValueError as error:
            assert "different content" in str(error)
        else:
            raise AssertionError("unrelated rollback replaced")
        assert list(conflicting.iterdir()) == [conflicting / prepare.ROLLBACK]
        assert (conflicting / prepare.ROLLBACK).read_text() == "unrelated content"
finally:
    prepare.ORIGINAL, prepare.PREPARED = original_pins, prepared_pins

print("test-prepare-sparkcache: PASS")
