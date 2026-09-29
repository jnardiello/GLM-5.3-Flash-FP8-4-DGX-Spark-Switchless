#!/usr/bin/env python3
"""Offline safety tests for the resilience namespace and runtime adapter."""

from __future__ import annotations

import errno
import json
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[2]
RESILIENCE = REPO / "scripts/resilience"
sys.path.insert(0, str(RESILIENCE))
import common  # noqa: E402
import faultctl  # noqa: E402
import runtime  # noqa: E402


class FaultControlTests(unittest.TestCase):
    def make_root(self, parent: Path, campaign_id: str = "unit-test") -> Path:
        root = parent.resolve() / common.namespace_name(campaign_id)
        faultctl.initialize(root, campaign_id)
        return root

    def test_root_requires_exact_marker_and_refuses_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = self.make_root(parent)
            self.assertEqual(faultctl.checked_root(str(root), "unit-test"), root)
            with self.assertRaises(common.ContractError):
                faultctl.checked_root(str(parent / "ordinary-cache"), "unit-test", initialized=False)
            link = parent / "tp4-resilience-link-test"
            link.symlink_to(root, target_is_directory=True)
            with self.assertRaises(common.ContractError):
                faultctl.checked_root(str(link), "link-test", initialized=False)

    def test_mutations_are_recoverable_and_hash_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_root(Path(directory))
            chunk = root / "chunks" / "aa" / "object"
            chunk.parent.mkdir(parents=True)
            original = (b"0123456789abcdef" * 4096)
            chunk.write_bytes(original)
            for mutation in ("checksum", "truncated", "missing"):
                with self.subTest(mutation=mutation):
                    record = faultctl.mutate(root, "unit-test", mutation, "chunk", 0)
                    self.assertEqual(record["mutation"], mutation)
                    if mutation == "missing":
                        self.assertFalse(chunk.exists())
                    else:
                        self.assertNotEqual(common.sha256_file(chunk), record["before_sha256"])
                    restored = faultctl.restore_mutations(root)
                    self.assertIn(record["mutation_id"], restored["restored"])
                    self.assertEqual(chunk.read_bytes(), original)

    def test_digest_scopes_mutation_to_the_snapshot_object(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_root(Path(directory))
            digest = "a" * 64
            selected_sha = "b" * 64
            chunks = root / "chunks"
            chunks.mkdir()
            selected = chunks / f"{selected_sha}.spcc"
            unrelated = chunks / "unrelated"
            selected.write_bytes(b"selected payload")
            unrelated.write_bytes(b"unrelated payload")
            manifest = root / "manifests" / "identity" / f"{digest}.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps({"snapshot_objects": [{"sha256": selected_sha}]}))
            record = faultctl.mutate(root, "unit-test", "checksum", "chunk", 0, digest)
            self.assertEqual(record["context_digest"], digest)
            self.assertEqual(unrelated.read_bytes(), b"unrelated payload")
            self.assertNotEqual(selected.read_bytes(), b"selected payload")
            faultctl.restore_mutations(root)
            self.assertEqual(selected.read_bytes(), b"selected payload")

    def test_mutation_admission_preserves_the_64_mib_control_allowance(self):
        self.assertEqual(runtime._CONTROL_ALLOWANCE, 64 << 20)
        self.assertEqual(faultctl._CONTROL_ALLOWANCE, runtime._CONTROL_ALLOWANCE)
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_root(Path(directory))
            chunk = root / "chunks" / "object"
            chunk.parent.mkdir()
            chunk.write_bytes(b"payload")
            near_limit = (common.MAX_NAMESPACE_BYTES - faultctl._CONTROL_ALLOWANCE
                          - chunk.stat().st_size + 1)
            with patch.object(faultctl, "tree_bytes", return_value=near_limit), \
                    self.assertRaises(common.ContractError):
                faultctl.mutate(root, "unit-test", "checksum", "chunk", 0)

    def test_wait_stage_uses_fresh_event_and_readline_offsets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_root(Path(directory))
            events = root / ".tp4-resilience" / "events.jsonl"

            def publish():
                time.sleep(0.1)
                common.append_event(events, {"kind": "stage_waiting", "stage": "capture",
                                             "case_id": "case-1", "operation_id": "op"})

            thread = threading.Thread(target=publish)
            thread.start()
            event = faultctl.wait_stage(root, "case-1", "capture", 2)
            thread.join()
            self.assertEqual(event["operation_id"], "op")

    def test_event_cursor_hands_off_to_current_file_after_one_rotation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_root(Path(directory))
            events = root / ".tp4-resilience" / "events.jsonl"
            common.append_event(events, {"kind": "old", "case_id": "case-rotate"})
            cursor = faultctl.event_cursor(root)
            os.replace(events, events.with_suffix(".previous.jsonl"))
            common.append_event(events, {"kind": "stage_end", "stage": "capture",
                                         "case_id": "case-rotate", "outcome": "error"})
            event = faultctl.find_event(root, cursor, case_id="case-rotate",
                                        kind="stage_end", stage="capture", timeout=1)
            self.assertEqual(event["outcome"], "error")

    def test_prune_reservations_keeps_only_verified_live_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_root(Path(directory))
            reservations = root / ".tp4-resilience" / "reservations"
            reservations.mkdir()
            (reservations / "live.json").write_text(json.dumps(
                {"pid": 10, "process_start_ticks": 100}))
            (reservations / "stale.json").write_text(json.dumps(
                {"pid": 20, "process_start_ticks": 200}))
            def host_pid(pid, ticks):
                if (pid, ticks) == (10, 100):
                    return 1010
                raise common.ContractError("stale")
            with patch.object(faultctl, "_host_pid", side_effect=host_pid):
                result = faultctl.prune_reservations(root)
            self.assertEqual(result, {"removed": ["stale.json"], "retained": ["live.json"]})

    def test_barrier_returns_the_node_monotonic_clock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_root(Path(directory))
            output = io.StringIO()
            with patch.object(faultctl.time, "monotonic_ns", return_value=123456789), \
                    redirect_stdout(output):
                result = faultctl.main(["--root", str(root), "--campaign-id", "unit-test",
                                        "barrier"])
            self.assertEqual(result, 0)
            self.assertEqual(json.loads(output.getvalue()), {"monotonic_ns": 123456789})


class RuntimeSubprocessTests(unittest.TestCase):
    def run_runtime(self, source: str, *, cap: int = 65536) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "tp4-resilience-unit-runtime"
            control = root / ".tp4-resilience"
            control.mkdir(parents=True)
            common.atomic_json(control / "marker.json",
                               {"campaign_id": "unit-runtime", "schema": common.SCHEMA})
            common.atomic_json(control / "control.json", {
                "schema": common.SCHEMA, "campaign_id": "unit-runtime", "case_id": "unit",
                "mode": "off", "operation": "any", "remaining": 0, "after": 0,
                "delay_ms": 0, "generation": 1,
            })
            environment = os.environ.copy()
            environment.update({
                "PYTHONPATH": str(RESILIENCE),
                "TP4_RESILIENCE_CAMPAIGN_ID": "unit-runtime",
                "TP4_RESILIENCE_CACHE_ROOT": str(root),
                "TP4_RESILIENCE_TEST_ROOT": str(root),
                "TP4_RESILIENCE_MAX_BYTES": str(cap),
            })
            return subprocess.run([sys.executable, "-c", source], env=environment, text=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)

    def test_write_fault_is_scoped_and_finite(self):
        source = r'''
import json, os
from pathlib import Path
import runtime
runtime._process_start_ticks = lambda: 123
root = Path(os.environ["TP4_RESILIENCE_CACHE_ROOT"])
control = root / ".tp4-resilience/control.json"
runtime.install()
outside = root.parent / "outside"
outside.write_text("ok")
value = json.loads(control.read_text())
value.update(mode="eio", operation="write", after=0, remaining=1, generation=2)
control.write_text(json.dumps(value))
failed = False
try:
    (root / "first").write_bytes(b"payload")
except OSError as error:
    failed = error.errno == 5
(root / "second").write_bytes(b"payload")
print(json.dumps({"failed": failed, "outside": outside.read_text(),
                  "second": (root / "second").read_text()}))
'''
        result = self.run_runtime(source)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout),
                         {"failed": True, "outside": "ok", "second": "payload"})

    def test_install_rejects_a_symlinked_campaign_root(self):
        source = r'''
import json, os
from pathlib import Path
import runtime
runtime._process_start_ticks = lambda: 123
root = Path(os.environ["TP4_RESILIENCE_CACHE_ROOT"])
real = root.with_name("real-cache-root")
root.rename(real)
root.symlink_to(real, target_is_directory=True)
failed = False
try:
    runtime.install()
except RuntimeError as error:
    failed = "symlink" in str(error)
print(json.dumps({"failed": failed}))
'''
        result = self.run_runtime(source)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {"failed": True})

    def test_refresh_failure_still_unlocks_and_closes_quota_descriptor(self):
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory) / "quota.lock"
            descriptor = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
            runtime.fcntl.flock(descriptor, runtime.fcntl.LOCK_EX)
            actual_flock = runtime.fcntl.flock
            operations = []

            def tracked_flock(fd, operation):
                operations.append(operation)
                return actual_flock(fd, operation)

            with patch.object(runtime, "_refresh_quota_state",
                              side_effect=OSError(errno.EIO, "refresh failed")), \
                    patch.object(runtime.fcntl, "flock", side_effect=tracked_flock), \
                    self.assertRaises(OSError):
                runtime._unlock((descriptor, Path(directory)), refresh=True)
            self.assertIn(runtime.fcntl.LOCK_UN, operations)
            with self.assertRaises(OSError):
                os.fstat(descriptor)

    def test_enospc_is_consumed_by_temporary_file_write_not_directory_open(self):
        source = r'''
import errno, json, os, tempfile
from pathlib import Path
import runtime
runtime._process_start_ticks = lambda: 123
runtime.install()
root = Path(os.environ["TP4_RESILIENCE_CACHE_ROOT"])
control = root / ".tp4-resilience/control.json"
value = json.loads(control.read_text())
value.update(mode="enospc", operation="write", after=0, remaining=1, generation=2)
control.write_text(json.dumps(value))
opened = failed = False
try:
    with tempfile.TemporaryFile(dir=root) as stream:
        opened = True
        stream.write(b"snapshot bytes")
except OSError as error:
    failed = error.errno == errno.ENOSPC
events = [json.loads(line) for line in
          (root / ".tp4-resilience/events.jsonl").read_text().splitlines()]
hits = [event for event in events if event.get("kind") == "fault_hit"]
print(json.dumps({"opened": opened, "failed": failed, "hits": len(hits),
                  "mode": hits[0].get("mode") if hits else None}))
'''
        result = self.run_runtime(source)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout),
                         {"opened": True, "failed": True, "hits": 1, "mode": "enospc"})

    def test_read_fault_skips_manifest_probe_and_hits_snapshot_object(self):
        source = r'''
import errno, json, os
from pathlib import Path
import runtime
runtime._process_start_ticks = lambda: 123
runtime.install()
root = Path(os.environ["TP4_RESILIENCE_CACHE_ROOT"])
manifest = root / "manifests/identity/context.json"
chunk = root / "chunks/object.spcc"
manifest.parent.mkdir(parents=True); chunk.parent.mkdir(parents=True)
manifest.write_text("manifest"); chunk.write_text("payload")
control = root / ".tp4-resilience/control.json"
value = json.loads(control.read_text())
value.update(mode="eio", operation="read", after=0, remaining=1, generation=2)
control.write_text(json.dumps(value))
manifest_value = manifest.read_text()
failed = False
try:
    chunk.read_text()
except OSError as error:
    failed = error.errno == errno.EIO
events = [json.loads(line) for line in
          (root / ".tp4-resilience/events.jsonl").read_text().splitlines()]
hits = [event for event in events if event.get("kind") == "fault_hit"]
print(json.dumps({"manifest": manifest_value, "failed": failed, "hits": len(hits)}))
'''
        result = self.run_runtime(source)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout),
                         {"manifest": "manifest", "failed": True, "hits": 1})

    def test_cap_covers_buffered_and_anonymous_staging(self):
        source = r'''
import errno, json, os, tempfile
from pathlib import Path
import runtime
runtime._process_start_ticks = lambda: 123
runtime.install()
root = Path(os.environ["TP4_RESILIENCE_CACHE_ROOT"])
with (root / "linked").open("wb") as stream:
    stream.write(b"a" * 30000)
failed = False
try:
    with tempfile.TemporaryFile(dir=root) as stream:
        stream.write(b"b" * 30000)
except OSError as error:
    failed = error.errno == errno.ENOSPC
print(json.dumps({"failed": failed, "linked": (root / "linked").stat().st_size}))
'''
        result = self.run_runtime(source)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {"failed": True, "linked": 30000})

    def test_quota_serializes_concurrent_writers(self):
        source = r'''
import json, os, threading
from pathlib import Path
import runtime
runtime._process_start_ticks = lambda: 123
runtime.install()
root = Path(os.environ["TP4_RESILIENCE_CACHE_ROOT"])
barrier = threading.Barrier(9)
results = []
lock = threading.Lock()
def write(index):
    barrier.wait()
    try:
        (root / f"data-{index}").write_bytes(b"x" * 20000)
        result = "ok"
    except OSError:
        result = "full"
    with lock:
        results.append(result)
threads = [threading.Thread(target=write, args=(i,)) for i in range(8)]
for thread in threads: thread.start()
barrier.wait()
for thread in threads: thread.join()
used = sum(path.stat().st_size for path in root.glob("data-*"))
print(json.dumps({"ok": results.count("ok"), "full": results.count("full"), "used": used}))
'''
        result = self.run_runtime(source, cap=131072)
        self.assertEqual(result.returncode, 0, result.stderr)
        value = json.loads(result.stdout)
        self.assertGreater(value["ok"], 0)
        self.assertGreater(value["full"], 0)
        self.assertLessEqual(value["used"], 131072 - (32 << 10))

    def test_quota_lock_is_shared_across_processes(self):
        source = r'''
import json, multiprocessing, os
from pathlib import Path
import runtime
runtime._process_start_ticks = lambda: 123
runtime.install()
root = Path(os.environ["TP4_RESILIENCE_CACHE_ROOT"])
def write(index):
    try:
        (root / f"process-{index}").write_bytes(b"p" * 20000)
    except OSError:
        pass
context = multiprocessing.get_context("fork")
processes = [context.Process(target=write, args=(i,)) for i in range(8)]
for process in processes: process.start()
for process in processes: process.join(10)
sizes = [path.stat().st_size for path in root.glob("process-*")]
print(json.dumps({"files": sum(size > 0 for size in sizes), "used": sum(sizes),
                  "exitcodes": [process.exitcode for process in processes]}))
'''
        result = self.run_runtime(source, cap=131072)
        self.assertEqual(result.returncode, 0, result.stderr)
        value = json.loads(result.stdout)
        self.assertTrue(all(code == 0 for code in value["exitcodes"]))
        self.assertGreater(value["files"], 0)
        self.assertLess(value["files"], 8, value)
        self.assertLessEqual(value["used"], 131072 - (32 << 10))

    def test_reservation_records_are_atomic_under_threads(self):
        source = r'''
import json, os, threading
from pathlib import Path
import runtime
runtime._process_start_ticks = lambda: 123
runtime.install()
threads = [threading.Thread(target=runtime.write_reservation,
    args=("owner", value), kwargs={"action": "test", "peak_bytes": value})
    for value in range(20)]
for thread in threads: thread.start()
for thread in threads: thread.join()
root = Path(os.environ["TP4_RESILIENCE_CACHE_ROOT"])
records = list((root / ".tp4-resilience/reservations").glob("*.json"))
value = json.loads(records[0].read_text())
runtime.write_reservation("owner", 0, action="release", peak_bytes=0)
final = json.loads(records[0].read_text())
print(json.dumps({"records": len(records), "valid": isinstance(value["reserved_bytes"], int),
                  "final": final["reserved_bytes"]}))
'''
        result = self.run_runtime(source, cap=131072)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {"records": 1, "valid": True, "final": 0})

    def test_stage_pause_waits_for_controller_release(self):
        source = r'''
import json, os, threading, time
from pathlib import Path
import runtime
runtime._process_start_ticks = lambda: 123
runtime.install()
root = Path(os.environ["TP4_RESILIENCE_CACHE_ROOT"])
control = root / ".tp4-resilience/control.json"
value = json.loads(control.read_text())
value.update(case_id="phase-case", pause_stage="capture", generation=2)
control.write_text(json.dumps(value))
thread = threading.Thread(target=runtime.begin_stage, args=("capture", "operation"))
thread.start()
events = root / ".tp4-resilience/events.jsonl"
deadline = time.monotonic() + 2
waiting = False
while time.monotonic() < deadline:
    waiting = '"kind":"stage_waiting"' in events.read_text()
    if waiting: break
    time.sleep(.01)
value.update(pause_stage=None, generation=3)
value.update(case_id="next-case")
control.write_text(json.dumps(value))
thread.join(2)
records = [json.loads(line) for line in events.read_text().splitlines()
           if '"operation_id":"operation"' in line]
print(json.dumps({"waiting": waiting,
                  "released": any(row.get("kind") == "stage_released" for row in records),
                  "case_ids": sorted(set(row.get("case_id") for row in records)),
                  "alive": thread.is_alive()}))
'''
        result = self.run_runtime(source)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout),
                         {"waiting": True, "released": True,
                          "case_ids": ["phase-case"], "alive": False})


if __name__ == "__main__":
    unittest.main()
