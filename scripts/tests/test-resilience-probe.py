#!/usr/bin/env python3
"""Offline safety tests for the bounded TP4 resilience telemetry probe."""

from __future__ import annotations

import argparse
import errno
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch


REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "tp4_resilience_probe", REPO / "scripts/resilience/probe.py"
)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def _stat(pid: int, start_ticks: int) -> str:
    fields = ["0"] * 40
    fields[0] = "S"
    fields[19] = str(start_ticks)
    return f"{pid} (worker with ) name) " + " ".join(fields) + "\n"


def _proc_process(
    proc: Path, host_pid: int, parent: int, namespace_pid: int, start_ticks: int,
    *, rss_kib: int = 4, anon_kib: int = 3, threads: int = 2,
) -> None:
    root = proc / str(host_pid)
    (root / "fd").mkdir(parents=True)
    (root / "status").write_text(
        f"Name:\tworker\nPPid:\t{parent}\nNSpid:\t{host_pid}\t{namespace_pid}\n"
        f"VmRSS:\t{rss_kib} kB\nRssAnon:\t{anon_kib} kB\nThreads:\t{threads}\n",
        encoding="ascii",
    )
    (root / "stat").write_text(_stat(host_pid, start_ticks), encoding="ascii")


def _cgroup(proc: Path, cgroups: Path, root_pid: int, pids: list[int], oom_kill: int = 2) -> Path:
    relative = Path("docker/test")
    (proc / str(root_pid) / "cgroup").write_text(f"0::/{relative}\n", encoding="ascii")
    directory = cgroups / relative
    directory.mkdir(parents=True)
    (directory / "cgroup.procs").write_text("\n".join(map(str, pids)) + "\n", encoding="ascii")
    (directory / "memory.current").write_text("1000\n", encoding="ascii")
    (directory / "memory.stat").write_text("anon 600\nfile 300\nshmem 100\n", encoding="ascii")
    (directory / "memory.events").write_text(
        f"low 0\nhigh 0\nmax 0\noom 0\noom_kill {oom_kill}\n", encoding="ascii"
    )
    return directory


def _reservation(path: Path, pid: int, start_ticks: int, reserved: int, owner: str = "budget") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema": "tp4-resilience/v1", "campaign_id": "test-campaign",
        "reserved_bytes": reserved, "action": "release" if reserved == 0 else "admit",
        "peak_bytes": max(1, reserved), "pid": pid,
        "process_start_ticks": start_ticks, "owner_id": owner, "time_ns": 1,
    }), encoding="utf-8")


def _args(cache_root: str = "/does/not/exist", rank: int = 1) -> argparse.Namespace:
    return argparse.Namespace(
        container="test", cache_root=cache_root,
        container_cache_root="/cache/jit/tp4-resilience-test", rank=rank,
        api_port=8000, interval=0.2, once=False,
    )


class ProcessAndReservationTests(unittest.TestCase):
    def test_cgroup_nspid_start_and_memory_verify_live_owner(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            proc, cgroups = base / "proc", base / "cgroup"
            proc.mkdir()
            _proc_process(proc, 100, 0, 1, 1000, rss_kib=5, threads=3)
            _proc_process(proc, 101, 100, 42, 4242, rss_kib=7, threads=4)
            _cgroup(proc, cgroups, 100, [100, 101])
            cache_root = base / "cache"
            _reservation(cache_root / ".tp4-resilience/reservations/42-a.json", 42, 4242, 1234)
            with patch.object(probe, "PROC_ROOT", proc), patch.object(probe, "CGROUP_ROOT", cgroups):
                process = probe.process_tree(100)
                result = probe.cache(cache_root, process)

            self.assertEqual(process["rss_bytes"], 12 * 1024)
            self.assertEqual(process["rss_semantics"], "nonadditive_process_sum")
            self.assertEqual(process["cgroup_memory"]["current_bytes"], 1000)
            self.assertEqual(process["membership_source"], "cgroup_v2")
            self.assertTrue(process["identities_complete"])
            self.assertEqual(result["reservation_status"], "ok")
            self.assertEqual(result["reserved_bytes"], 1234)

    def test_proven_dead_owner_is_excluded_but_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            proc, cgroups = base / "proc", base / "cgroup"
            proc.mkdir()
            _proc_process(proc, 100, 0, 1, 1000)
            _proc_process(proc, 101, 100, 42, 4242)
            _cgroup(proc, cgroups, 100, [100, 101])
            root = base / "cache"
            _reservation(root / ".tp4-resilience/reservations/42-live.json", 42, 4242, 0, "live")
            _reservation(root / ".tp4-resilience/reservations/77-dead.json", 77, 7777, 99, "dead")
            with patch.object(probe, "PROC_ROOT", proc), patch.object(probe, "CGROUP_ROOT", cgroups):
                result = probe.cache(root, probe.process_tree(100))
            self.assertEqual(result["reservation_status"], "ok")
            self.assertEqual(result["reserved_bytes"], 0)
            self.assertEqual(result["dead_owner_records"], 1)

    def test_ancestry_fallback_cannot_prove_owner_dead(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            proc = base / "proc"
            proc.mkdir()
            _proc_process(proc, 100, 0, 1, 1000)
            root = base / "cache"
            _reservation(root / ".tp4-resilience/reservations/77-old.json", 77, 7777, 0)
            with patch.object(probe, "PROC_ROOT", proc):
                process = probe.process_tree(100)
                result = probe.cache(root, process)
            self.assertFalse(process["identities_complete"])
            self.assertEqual(result["reservation_status"], "no_verified_owners")
            self.assertEqual(result["unverifiable_reservation_records"], 1)
            self.assertIsNone(result["reserved_bytes"])

    def test_missing_instrumentation_differs_from_verified_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            process = {"identity_status": "ok", "identities_complete": True,
                       "identities": [{"host_pid": 7, "container_pid": 7,
                                       "start_ticks": 70}]}
            missing = probe.cache(root, process)
            _reservation(root / ".tp4-resilience/reservations/7-a.json", 7, 70, 0)
            present = probe.cache(root, process)
            self.assertEqual(missing["reservation_status"], "instrumentation_missing")
            self.assertIsNone(missing["reserved_bytes"])
            self.assertEqual(present["reservation_status"], "ok")
            self.assertEqual(present["reserved_bytes"], 0)

    def test_unreadable_reservation_directory_is_not_empty(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            process = {"identity_status": "ok", "identities_complete": True, "identities": []}
            original = probe.os.scandir

            def denied(path):
                if Path(path).name == "reservations":
                    raise PermissionError(errno.EACCES, "denied", path)
                return original(path)

            with patch.object(probe.os, "scandir", side_effect=denied):
                result = probe.cache(root, process)
            self.assertEqual(result["reservation_status"], "instrumentation_unreadable")
            self.assertIsNone(result["reserved_bytes"])

    def test_one_unreadable_record_prevents_verified_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            live = root / ".tp4-resilience/reservations/7-live.json"
            unreadable = root / ".tp4-resilience/reservations/8-denied.json"
            _reservation(live, 7, 70, 0, "live")
            _reservation(unreadable, 8, 80, 0, "denied")
            process = {"identity_status": "ok", "identities_complete": True,
                       "identities": [{"host_pid": 7, "container_pid": 7,
                                       "start_ticks": 70}]}
            original = Path.read_text

            def denied(path, *args, **kwargs):
                if path == unreadable:
                    raise PermissionError(errno.EACCES, "denied", path)
                return original(path, *args, **kwargs)

            with patch.object(Path, "read_text", denied):
                snapshot = probe._reservation_snapshot(root)
            result = probe._reservations(root, process, snapshot)
            self.assertEqual(result["reservation_status"], "partial")
            self.assertEqual(result["reserved_bytes"], 0)
            self.assertEqual(result["unverifiable_reservation_records"], 1)

    def test_unreadable_process_stat_cannot_prove_record_dead(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            proc, cgroups = base / "proc", base / "cgroup"
            proc.mkdir()
            _proc_process(proc, 100, 0, 1, 1000)
            _proc_process(proc, 101, 100, 42, 4242)
            _cgroup(proc, cgroups, 100, [100, 101])
            root = base / "cache"
            _reservation(root / ".tp4-resilience/reservations/42-live.json", 42, 4242, 0)
            denied_stat = proc / "101/stat"
            original = Path.read_text

            def denied(path, *args, **kwargs):
                if path == denied_stat:
                    raise PermissionError(errno.EACCES, "denied", path)
                return original(path, *args, **kwargs)

            with patch.object(probe, "PROC_ROOT", proc), patch.object(
                probe, "CGROUP_ROOT", cgroups
            ), patch.object(Path, "read_text", denied):
                process = probe.process_tree(100)
                result = probe.cache(root, process)
            self.assertFalse(process["identities_complete"])
            self.assertEqual(result["reservation_status"], "no_verified_owners")
            self.assertEqual(result["dead_owner_records"], 0)


class CacheAccountingTests(unittest.TestCase):
    def test_container_path_deleted_fd_is_counted_with_host_device(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            root = base / "cache"
            root.mkdir()
            backing = base / "open-file"
            backing.write_bytes(b"x" * 37)
            proc = base / "proc"
            descriptor = proc / "12/fd/7"
            descriptor.parent.mkdir(parents=True)
            descriptor.symlink_to(backing)
            process = {"identity_status": "ok", "identities_complete": True,
                       "identities": [{"host_pid": 12, "container_pid": 2,
                                       "start_ticks": 1}]}
            original = os.readlink

            def container_link(path):
                if Path(path) == descriptor:
                    return "/cache/jit/tp4-resilience-test/.snapshot-a (deleted)"
                return original(path)

            with patch.object(probe, "PROC_ROOT", proc), patch.object(
                probe.os, "readlink", side_effect=container_link
            ):
                result = probe.cache(
                    root, process, container_root=Path("/cache/jit/tp4-resilience-test")
                )
            self.assertEqual(result["anonymous_staging_bytes"], 37)
            self.assertEqual(result["staging_bytes"], 37)
            self.assertTrue(result["scan_complete"])

    def test_fd_eacces_is_unknown_and_incomplete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            process = {"identity_status": "ok", "identities_complete": True,
                       "identities": [{"host_pid": 12, "container_pid": 2,
                                       "start_ticks": 1}]}
            original = Path.iterdir

            def denied(path):
                if str(path).endswith("/12/fd"):
                    raise PermissionError(errno.EACCES, "denied", path)
                return original(path)

            with patch.object(Path, "iterdir", denied):
                result = probe.cache(root, process, container_root=Path("/cache/test"))
            self.assertIsNone(result["anonymous_staging_bytes"])
            self.assertIsNone(result["staging_bytes"])
            self.assertFalse(result["scan_complete"])

    def test_walk_eacces_and_missing_root_are_unknown(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "cache"
            root.mkdir()
            process = {"identity_status": "container_unknown", "identities": []}

            def failed_walk(path, **kwargs):
                kwargs["onerror"](PermissionError(errno.EACCES, "denied", path))
                return iter(())

            with patch.object(probe.os, "walk", side_effect=failed_walk):
                denied = probe.cache(root, process)
            missing = probe.cache(root / "missing", process)
            self.assertFalse(denied["scan_complete"])
            self.assertEqual(denied["accounting_status"], "partial")
            self.assertIsNone(missing["bytes"])
            self.assertEqual(missing["accounting_status"], "root_unavailable")

    def test_unreadable_events_do_not_report_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            events = root / ".tp4-resilience/events.jsonl"
            events.parent.mkdir()
            events.write_text("{}\n")
            original = Path.open

            def denied(path, *args, **kwargs):
                if path == events:
                    raise PermissionError(errno.EACCES, "denied", path)
                return original(path, *args, **kwargs)

            with patch.object(Path, "open", denied):
                result = probe.operation_events(root)
            self.assertEqual(result["status"], "unreadable")
            self.assertIsNone(result["events"])
            self.assertIsNone(result["io_errors"])

    def test_truncated_event_history_cannot_prove_zero_active(self):
        begin = json.dumps({"kind": "stage_begin", "stage": "restore", "operation_id": "x"})
        with patch.object(probe, "_tail_lines", return_value=([begin], False, None)):
            result = probe.operation_events(Path("/cache"))
        self.assertFalse(result["coverage_complete"])
        self.assertIsNone(result["stages"]["restore"]["active"])


class OomAndLogEvidenceTests(unittest.TestCase):
    def test_vmstat_and_cgroup_oom_deltas_are_monotonic(self):
        collector = probe.PersistentProbe(_args())
        collector._host_oom_baseline = 4
        collector._cgroup_oom_baseline = ("/cg", 2)
        process = {"cgroup_memory": {"path": "/cg", "events": {"oom_kill": 3}}}
        with patch.object(probe, "_counter", return_value=6):
            result = collector._oom(process, {"status": "ok", "earlyoom_kills": 0})
        self.assertEqual(result["host_oom_kills_since_start"], 2)
        self.assertEqual(result["cgroup_oom_kills_since_start"], 1)

    def test_unknown_oom_sources_are_none_not_zero(self):
        collector = probe.PersistentProbe(_args())
        with patch.object(probe, "_counter", return_value=None):
            result = collector._oom({}, {"status": "unknown", "earlyoom_kills": None})
        self.assertIsNone(result["host_oom_kills_since_start"])
        self.assertIsNone(result["cgroup_oom_kills_since_start"])
        self.assertIsNone(result["earlyoom_kills"])

    def test_earlyoom_counts_direct_sigterm_and_sigkill_without_escalation_double_count(self):
        output = "\n".join((
            json.dumps({"SYSLOG_IDENTIFIER": "earlyoom",
                        "MESSAGE": "sending SIGTERM to process 55 uid 1000"}),
            json.dumps({"SYSLOG_IDENTIFIER": "earlyoom",
                        "MESSAGE": "sending SIGKILL to process 56 uid 1000"}),
            json.dumps({"SYSLOG_IDENTIFIER": "earlyoom",
                        "MESSAGE": "process 55 did not exit after SIGTERM; sending SIGKILL"}),
            json.dumps({"SYSLOG_IDENTIFIER": "kernel",
                        "MESSAGE": "oom-kill: constraint=CONSTRAINT_NONE"}),
            json.dumps({"SYSLOG_IDENTIFIER": "kernel",
                        "MESSAGE": "Out of memory: Killed process 55"}),
        ))
        with patch.object(probe, "_command", return_value=(output, None)) as command:
            result = probe.earlyoom_evidence(1_000_000_000)
        self.assertEqual(result["earlyoom_kills"], 2)
        self.assertIn("earlyoom", command.call_args.args[0])
        self.assertIn("@1.000", command.call_args.args[0])

    def test_cuda_record_requires_fresh_matching_pid_and_exact_fallback(self):
        record = {"wall_time_ns": 2_000_000_000, "pid": 42, "sequence": 3,
                  "peaks_scope": "worker_process_no_reset",
                  "cuda_allocator": {"num_alloc_retries": 2}}
        trace = {"event": "fallback", "reason": "deadline"}
        output = ("E20_MEMORY_PROBE " + json.dumps(record) + "\n"
                  "unrelated recompute documentation\n"
                  "spark-context-cache: pending-trace " + json.dumps(trace) + "\n")
        with patch.object(probe, "_command", return_value=(output, None)), patch.object(
            probe, "TIME_NS", return_value=4_000_000_000
        ):
            matching = probe.container_logs("c", since_time_ns=1, valid_container_pids={42})
            mismatch = probe.container_logs("c", since_time_ns=1, valid_container_pids={9})
        self.assertEqual(matching["cuda"]["status"], "ok")
        self.assertEqual(matching["restore_fallbacks"], 1)
        self.assertEqual(mismatch["cuda"]["status"], "pid_mismatch")

    def test_cgroup_baseline_resets_on_container_path_change(self):
        collector = probe.PersistentProbe(_args())
        collector._host_oom_baseline = 4
        collector._cgroup_oom_baseline = ("/old", 9)
        first = {"cgroup_memory": {"path": "/new", "events": {"oom_kill": 2}}}
        second = {"cgroup_memory": {"path": "/new", "events": {"oom_kill": 3}}}
        with patch.object(probe, "_counter", return_value=4):
            reset = collector._oom(first, {"status": "ok", "earlyoom_kills": 0})
            matched = collector._oom(second, {"status": "ok", "earlyoom_kills": 0})
        self.assertEqual(reset["cgroup_oom_kills_since_start"], 0)
        self.assertEqual(reset["cgroup_oom_baseline_status"], "reset")
        self.assertEqual(matched["cgroup_oom_kills_since_start"], 1)
        self.assertTrue(matched["coverage"]["cgroup_oom_delta_known"])

    def test_cuda_oom_positive_evidence_is_sticky_across_sliding_log_window(self):
        collector = probe.PersistentProbe(_args())
        collector._publish("cache_bundle", {"process": {"identities": []}})
        positive = {"cuda": probe.gpu(), "cuda_ooms": 1,
                    "store_failures": 0, "restore_fallbacks": 0}
        clear_window = {**positive, "cuda_ooms": 0}
        earlyoom = {"status": "ok", "earlyoom_kills": 0}
        with patch.object(probe, "container_logs", side_effect=[positive, clear_window]), patch.object(
            probe, "earlyoom_evidence", return_value=earlyoom
        ):
            first = collector._evidence()
            second = collector._evidence()
        self.assertTrue(first["logs"]["cuda_oom_seen_since_start"])
        self.assertTrue(second["logs"]["cuda_oom_seen_since_start"])

    def test_truncated_zero_cuda_window_remains_unknown(self):
        collector = probe.PersistentProbe(_args())
        collector._publish("cache_bundle", {"process": {"identities": []}})
        window = {"cuda": probe.gpu(), "cuda_ooms": 0, "coverage_complete": False,
                  "store_failures": 0, "restore_fallbacks": 0}
        with patch.object(probe, "container_logs", return_value=window), patch.object(
            probe, "earlyoom_evidence", return_value={"status": "ok", "earlyoom_kills": 0}
        ):
            evidence = collector._evidence()
        self.assertIsNone(evidence["logs"]["cuda_oom_seen_since_start"])


class AdmissionLogTests(unittest.TestCase):
    @staticmethod
    def _record(**overrides):
        value = {
            "schema": "tp4_admission_v1", "event": "acquire",
            "monotonic_ns": 1_000_000_000, "active": 2, "queued": 3,
            "inflight": 5,
            "admitted_total": 11,
            "rejected_total": 7, "rejected_full": 4,
            "rejected_queue_timeout": 3, "request_timeout": 2,
            "body_too_large": 1, "body_idle": 6, "send_idle": 8,
        }
        value.update(overrides)
        return value

    @staticmethod
    def _logs(admission, *, coverage_complete=True, status="ok"):
        return {
            "status": status,
            "coverage_complete": coverage_complete,
            "admission": admission,
        }

    def test_docker_started_at_parser_preserves_nanoseconds(self):
        value = probe._rfc3339_time_ns("2026-09-29T06:28:52.163193153Z")
        self.assertIsNotNone(value)
        self.assertEqual(value % 1_000_000_000, 163_193_153)
        self.assertIsNone(probe._rfc3339_time_ns("2026-09-29 06:28:52"))

    def test_parser_selects_latest_complete_state_and_maps_gauges(self):
        ready = {"schema": "tp4_admission_v1", "max_active": 4}
        first = self._record(event="admit", active=0, queued=1, inflight=1)
        last = self._record(monotonic_ns=2_000_000_000, event="acquire")
        parsed = probe._admission_log_state([
            "TP4_ADMISSION_READY " + json.dumps(ready),
            "TP4_ADMISSION_STATE " + json.dumps(first),
            "TP4_ADMISSION_STATE " + json.dumps(last),
        ])
        self.assertEqual(parsed["status"], "ok")
        self.assertTrue(parsed["ready_seen"])
        self.assertEqual(probe._admission_api(parsed["state"])["admission_requests_waiting"], 3)
        self.assertEqual(probe._admission_api(parsed["state"])["admission_requests_inflight"], 5)
        self.assertEqual(probe._admission_api(parsed["state"])["admission_admitted_total"], 11)

    def test_ready_without_state_and_malformed_values_never_become_zero(self):
        ready = probe._admission_log_state([
            'TP4_ADMISSION_READY {"schema":"tp4_admission_v1","max_active":4}'
        ])
        ready_state = self._record(
            event="ready", active=0, queued=0, inflight=0, admitted_total=0,
            rejected_total=0, rejected_full=0, rejected_queue_timeout=0,
            request_timeout=0, body_too_large=0, body_idle=0, send_idle=0,
        )
        ready_state.pop("event")
        initialized = probe._admission_log_state([
            "TP4_ADMISSION_READY " + json.dumps(ready_state)
        ])
        malformed = self._record(active=False, queued=-1)
        invalid = probe._admission_log_state([
            "TP4_ADMISSION_STATE " + json.dumps(malformed)
        ])
        self.assertEqual(ready["status"], "ready_without_state")
        self.assertEqual(initialized["status"], "ok")
        self.assertEqual(initialized["state"]["event"], "ready")
        self.assertEqual(initialized["state"]["inflight"], 0)
        self.assertEqual(invalid["status"], "invalid")
        self.assertTrue(all(value is None for value in probe._admission_api(invalid["state"]).values()))

    def test_state_retains_only_for_same_verified_container_identity(self):
        collector = probe.PersistentProbe(_args(rank=0))
        first = {"status": "ok", "state": self._record(), "invalid_records": 0}
        absent = {"status": "not_observed", "state": None, "invalid_records": 0}
        container = {"pid": 10, "root_start_ticks": 20, "boot_id": "boot", "state": "running"}
        metadata = {"status": "current"}
        with (
            patch.object(collector, "_validated_container_state",
                         return_value=("running", "verified")),
            patch.object(probe, "MONOTONIC_NS", return_value=4_000_000_000),
            patch.object(probe, "_start_ticks_monotonic_ns", return_value=200_000_000),
        ):
            observed = collector._bind_admission_state(self._logs(first), container, metadata)
            values, source = collector._admission_sample(
                observed, {"status": "current", "age_ns": 1,
                           "sampled_monotonic_ns": 3_999_999_999, "error": None},
                container, metadata)
            retained = collector._bind_admission_state(self._logs(absent), container, metadata)
            changed_pid = collector._bind_admission_state(
                self._logs(absent), {**container, "pid": 11}, metadata)
            collector._bind_admission_state(
                self._logs(first), {**container, "pid": 11}, metadata)
            changed_boot = collector._bind_admission_state(
                self._logs(absent), {**container, "pid": 11, "boot_id": "new"}, metadata)
        self.assertEqual(observed["state"]["queued"], 3)
        self.assertEqual(values["admission_requests_inflight"], 5)
        self.assertEqual(source["status"], "current")
        self.assertEqual(retained["state"]["queued"], 3)
        self.assertFalse(retained["observed_in_scan"])
        self.assertIsNone(changed_pid["state"])
        self.assertIsNone(changed_boot["state"])

    def test_malformed_new_transition_invalidates_retained_state(self):
        collector = probe.PersistentProbe(_args(rank=0))
        container = {"pid": 10, "root_start_ticks": 20, "boot_id": "boot", "state": "running"}
        metadata = {"status": "current"}
        valid = {"status": "ok", "state": self._record(), "invalid_records": 0}
        invalid = {"status": "invalid", "state": None, "invalid_records": 1}
        with (
            patch.object(collector, "_validated_container_state",
                         return_value=("running", "verified")),
            patch.object(probe, "MONOTONIC_NS", return_value=2_000_000_000),
            patch.object(probe, "_start_ticks_monotonic_ns", return_value=200_000_000),
        ):
            collector._bind_admission_state(self._logs(valid), container, metadata)
            bound = collector._bind_admission_state(self._logs(invalid), container, metadata)
            values, source = collector._admission_sample(
                bound, {"status": "current", "age_ns": 1,
                        "sampled_monotonic_ns": 1_999_999_999, "error": None},
                container, metadata)
        self.assertEqual(source["status"], "invalid")
        self.assertTrue(all(value is None for value in values.values()))

    def test_same_container_cumulative_counter_reset_is_unknown(self):
        collector = probe.PersistentProbe(_args(rank=0))
        container = {"pid": 10, "root_start_ticks": 20, "boot_id": "boot", "state": "running"}
        metadata = {"status": "current"}
        first = {"status": "ok", "state": self._record(), "invalid_records": 0}
        reset = {"status": "ok", "state": self._record(
            monotonic_ns=2_000_000_000, admitted_total=0), "invalid_records": 0}
        with (
            patch.object(collector, "_validated_container_state",
                         return_value=("running", "verified")),
            patch.object(probe, "MONOTONIC_NS", return_value=3_000_000_000),
            patch.object(probe, "_start_ticks_monotonic_ns", return_value=200_000_000),
        ):
            collector._bind_admission_state(self._logs(first), container, metadata)
            bound = collector._bind_admission_state(self._logs(reset), container, metadata)
        self.assertEqual(bound["status"], "invalid")
        self.assertIsNone(bound["state"])

    def test_identity_change_rejects_old_valid_zero_record(self):
        collector = probe.PersistentProbe(_args(rank=0))
        old = {"pid": 10, "root_start_ticks": 20, "boot_id": "boot", "state": "running"}
        new = {"pid": 11, "root_start_ticks": 2_000, "boot_id": "boot", "state": "running"}
        metadata = {"status": "current"}
        state = self._record(
            event="ready", active=0, queued=0, inflight=0, admitted_total=0,
            rejected_total=0, rejected_full=0, rejected_queue_timeout=0,
            request_timeout=0, body_too_large=0, body_idle=0, send_idle=0,
        )
        parsed = {"status": "ok", "state": state, "invalid_records": 0}
        with (
            patch.object(collector, "_validated_container_state",
                         return_value=("running", "verified")),
            patch.object(probe, "MONOTONIC_NS", return_value=3_000_000_000),
            patch.object(probe, "_start_ticks_monotonic_ns",
                         side_effect=lambda ticks: ticks * 1_000_000),
        ):
            self.assertIsNotNone(
                collector._bind_admission_state(self._logs(parsed), old, metadata)["state"])
            bound = collector._bind_admission_state(self._logs(parsed), new, metadata)
        self.assertEqual(bound["status"], "invalid")
        self.assertIsNone(bound["state"])
        self.assertTrue(all(value is None for value in probe._admission_api(bound["state"]).values()))

    def test_collection_failure_invalidates_retained_state(self):
        collector = probe.PersistentProbe(_args(rank=0))
        container = {"pid": 10, "root_start_ticks": 20, "boot_id": "boot", "state": "running"}
        metadata = {"status": "current"}
        parsed = {"status": "ok", "state": self._record(), "invalid_records": 0}
        with (
            patch.object(collector, "_validated_container_state",
                         return_value=("running", "verified")),
            patch.object(probe, "MONOTONIC_NS", return_value=2_000_000_000),
            patch.object(probe, "_start_ticks_monotonic_ns", return_value=200_000_000),
        ):
            collector._bind_admission_state(self._logs(parsed), container, metadata)
            failed = collector._bind_admission_state(
                self._logs(parsed, status="unknown"), container, metadata)
        self.assertEqual(failed["status"], "collection_error")
        self.assertIsNone(failed["state"])

    def test_truncated_tail_latest_release_proves_fresh_zero_gauge(self):
        collector = probe.PersistentProbe(_args(rank=0))
        container = {"pid": 10, "root_start_ticks": 20, "boot_id": "boot", "state": "running"}
        metadata = {"status": "current"}
        admit = self._record(
            monotonic_ns=1_100_000_000, event="admit", active=1, queued=0, inflight=1)
        release = self._record(
            monotonic_ns=1_200_000_000, event="release", active=0, queued=0, inflight=0)
        output = "\n".join([
            *(["unrelated"] * 4_998),
            "TP4_ADMISSION_STATE " + json.dumps(admit),
            "TP4_ADMISSION_STATE " + json.dumps(release),
        ])
        with patch.object(probe, "_command", return_value=(output, None)):
            logs = probe.container_logs("container")
        with (
            patch.object(collector, "_validated_container_state",
                         return_value=("running", "verified")),
            patch.object(probe, "MONOTONIC_NS", return_value=2_000_000_000),
            patch.object(probe, "_start_ticks_monotonic_ns", return_value=200_000_000),
        ):
            bound = collector._bind_admission_state(logs, container, metadata)
            values, source = collector._admission_sample(
                bound, {"status": "current", "age_ns": 1,
                        "sampled_monotonic_ns": 1_999_999_999, "error": None},
                container, metadata)
        self.assertFalse(logs["coverage_complete"])
        self.assertEqual(bound["state"]["event"], "release")
        self.assertEqual(values["admission_requests_active"], 0)
        self.assertEqual(values["admission_requests_waiting"], 0)
        self.assertEqual(values["admission_requests_inflight"], 0)
        self.assertEqual(source["status"], "current")
        self.assertFalse(source["log_tail_coverage_complete"])
        with (
            patch.object(probe, "_read", return_value=_stat(10, 20)),
            patch.object(probe, "MONOTONIC_NS", return_value=2_000_000_000),
            patch.object(probe, "_start_ticks_monotonic_ns", return_value=200_000_000),
        ):
            one_shot, one_shot_source = probe._one_shot_admission(logs, container)
        self.assertEqual(one_shot["admission_requests_inflight"], 0)
        self.assertEqual(one_shot_source["status"], "current")
        self.assertFalse(one_shot_source["log_tail_coverage_complete"])

    def test_truncated_tail_without_admission_marker_is_unknown(self):
        collector = probe.PersistentProbe(_args(rank=0))
        container = {"pid": 10, "root_start_ticks": 20, "boot_id": "boot", "state": "running"}
        metadata = {"status": "current"}
        prior = {"status": "ok", "state": self._record(), "invalid_records": 0}
        with patch.object(probe, "_command", return_value=("\n".join(
            ["unrelated"] * 5_000), None)):
            logs = probe.container_logs("container")
        with (
            patch.object(collector, "_validated_container_state",
                         return_value=("running", "verified")),
            patch.object(probe, "MONOTONIC_NS", return_value=2_000_000_000),
            patch.object(probe, "_start_ticks_monotonic_ns", return_value=200_000_000),
        ):
            collector._bind_admission_state(self._logs(prior), container, metadata)
            bound = collector._bind_admission_state(logs, container, metadata)
        self.assertEqual(bound["status"], "coverage_incomplete")
        self.assertIsNone(bound["state"])
        self.assertTrue(all(value is None for value in probe._admission_api(bound["state"]).values()))

    def test_truncated_tail_malformed_newer_marker_invalidates_old_zero(self):
        collector = probe.PersistentProbe(_args(rank=0))
        container = {"pid": 10, "root_start_ticks": 20, "boot_id": "boot", "state": "running"}
        metadata = {"status": "current"}
        release = self._record(
            monotonic_ns=1_200_000_000, event="release", active=0, queued=0, inflight=0)
        output = "\n".join([
            *(["unrelated"] * 4_998),
            "TP4_ADMISSION_STATE " + json.dumps(release),
            'TP4_ADMISSION_STATE {"schema":"tp4_admission_v1","event":"acquire"}',
        ])
        with patch.object(probe, "_command", return_value=(output, None)):
            logs = probe.container_logs("container")
        with (
            patch.object(collector, "_validated_container_state",
                         return_value=("running", "verified")),
            patch.object(probe, "MONOTONIC_NS", return_value=2_000_000_000),
            patch.object(probe, "_start_ticks_monotonic_ns", return_value=200_000_000),
        ):
            bound = collector._bind_admission_state(logs, container, metadata)
        self.assertEqual(logs["admission"]["status"], "invalid")
        self.assertEqual(bound["status"], "invalid")
        self.assertIsNone(bound["state"])
        self.assertTrue(all(value is None for value in probe._admission_api(bound["state"]).values()))

    def test_complete_same_identity_scan_retains_long_idle_state(self):
        collector = probe.PersistentProbe(_args(rank=0))
        container = {"pid": 10, "root_start_ticks": 20, "boot_id": "boot", "state": "running"}
        metadata = {"status": "current"}
        parsed = {"status": "ok", "state": self._record(), "invalid_records": 0}
        absent = {"status": "not_observed", "state": None, "invalid_records": 0}
        with (
            patch.object(collector, "_validated_container_state",
                         return_value=("running", "verified")),
            patch.object(probe, "MONOTONIC_NS",
                         side_effect=[2_000_000_000, 21_000_000_000, 21_000_000_000]),
            patch.object(probe, "_start_ticks_monotonic_ns", return_value=200_000_000),
        ):
            collector._bind_admission_state(self._logs(parsed), container, metadata)
            retained = collector._bind_admission_state(self._logs(absent), container, metadata)
        self.assertEqual(retained["status"], "ok")
        self.assertEqual(retained["state_age_ns"], 20_000_000_000)
        self.assertFalse(retained["observed_in_scan"])

    def test_stale_collector_does_not_expose_last_known_counts(self):
        collector = probe.PersistentProbe(_args(rank=0))
        container = {"pid": 10, "root_start_ticks": 20, "boot_id": "boot", "state": "running"}
        admission = {"status": "ok", "state": self._record(),
                     "identity": {"boot_id": "boot", "container_pid": 10,
                                  "root_start_ticks": 20},
                     "observed_in_scan": True}
        with patch.object(collector, "_validated_container_state", return_value=("running", "verified")):
            values, source = collector._admission_sample(
                admission, {"status": "stale", "age_ns": 11_000_000_000,
                            "sampled_monotonic_ns": 1, "error": None},
                container, {"status": "current"})
        self.assertEqual(source["status"], "stale")
        self.assertIsNone(values["admission_requests_waiting"])

    def test_rank0_bootstrap_recovers_prior_ready_then_uses_probe_start_window(self):
        collector = probe.PersistentProbe(_args(rank=0))
        collector._publish("cache_bundle", {"process": {"identities": []}})
        collector._publish("container", {
            "pid": 10, "root_start_ticks": 20, "boot_id": "boot", "state": "running",
            "started_at_time_ns": 123,
        })
        ready = {"status": "ok", "state": self._record(event="ready"),
                 "invalid_records": 0}
        absent = {"status": "not_observed", "state": None, "invalid_records": 0}
        base = {"status": "ok", "coverage_complete": True, "cuda": probe.gpu(),
                "cuda_ooms": 0, "store_failures": 0, "restore_fallbacks": 0}
        with (
            patch.object(collector, "_validated_container_state",
                         return_value=("running", "verified")),
            patch.object(probe, "MONOTONIC_NS", return_value=2_000_000_000),
            patch.object(probe, "_start_ticks_monotonic_ns", return_value=200_000_000),
            patch.object(probe, "container_logs", side_effect=[
                {**base, "admission": ready}, {**base, "admission": absent},
            ]) as logs,
            patch.object(probe, "earlyoom_evidence",
                         return_value={"status": "ok", "earlyoom_kills": 0}),
        ):
            first = collector._evidence()
            second = collector._evidence()
        self.assertEqual(first["logs"]["admission"]["state"]["inflight"], 5)
        self.assertEqual(second["logs"]["admission"]["state"]["inflight"], 5)
        self.assertEqual(logs.call_args_list[0].kwargs["since_time_ns"], 123)
        self.assertEqual(logs.call_args_list[1].kwargs["since_time_ns"], collector.started_time_ns)


class CommandAndCadenceTests(unittest.TestCase):
    def test_cache_bundle_timestamps_and_reads_records_before_final_identity(self):
        collector = probe.PersistentProbe(_args("/cache"))
        collector._publish("container", {"pid": 11})
        order: list[str] = []
        snapshot = {"status": "ok", "records": [], "count": 0,
                    "invalid": 0, "unreadable": 0, "error": None}
        process = {"identity_status": "ok", "identities": [],
                   "identities_complete": True}

        def records(root):
            order.append("records")
            return snapshot

        def identities(pid):
            order.append("identity")
            return process

        def cache_value(root, process_arg, **kwargs):
            order.append("cache")
            self.assertIs(process_arg, process)
            self.assertIs(kwargs["reservation_snapshot"], snapshot)
            return {"scan_started_monotonic_ns": kwargs["scan_started_monotonic_ns"]}

        with patch.object(probe, "MONOTONIC_NS", return_value=555), patch.object(
            probe, "_reservation_snapshot", side_effect=records
        ), patch.object(probe, "process_tree", side_effect=identities), patch.object(
            probe, "cache", side_effect=cache_value
        ):
            bundle = collector._cache_bundle()
        self.assertEqual(order, ["records", "identity", "cache"])
        self.assertEqual(bundle["cache"]["scan_started_monotonic_ns"], 555)

    def test_timeout_terminates_complete_process_group(self):
        process = Mock(pid=123, returncode=None)
        process.communicate.side_effect = [subprocess.TimeoutExpired(["x"], 0.1), ("", None)]
        with patch.object(probe.subprocess, "Popen", return_value=process), patch.object(
            probe.os, "killpg"
        ) as killpg:
            output, error = probe._command(["x"], 0.1)
        self.assertIsNone(output)
        self.assertEqual(error, "timeout")
        killpg.assert_called_once_with(123, probe.signal.SIGTERM)

    def test_deadline_skips_overdue_slot_without_drift(self):
        self.assertEqual(probe.advance_deadline(200, 350, 200), (400, 1))
        self.assertEqual(probe.advance_deadline(400, 399, 200), (400, 0))

    def test_fake_clock_exposes_stale_cached_source(self):
        collector = probe.PersistentProbe(_args())
        with patch.object(probe, "MONOTONIC_NS", return_value=100):
            collector._publish("item", {"value": 1})
        with patch.object(probe, "MONOTONIC_NS", return_value=2_000_000_101):
            _, metadata = collector._get("item", {}, 2_000_000_000)
        self.assertEqual(metadata["status"], "stale")

    def test_start_primes_container_and_cache_before_threads_and_first_tick(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            proc = base / "proc"
            proc.mkdir()
            _proc_process(proc, 11, 0, 1, 111)
            args = _args(str(base))
            collector = probe.PersistentProbe(args)
            process = {"identity_status": "ok", "identities_complete": True,
                       "identities": [], "cgroup_memory": {}}
            cache_value = {"bytes": 0, "scan_complete": True, "sample_fresh": True,
                           "accounting_status": "ok", "reservation_status": "ok",
                           "reserved_bytes": 0, "operations": {}}
            container = {"pid": 11, "state": "running", "root_start_ticks": 111,
                         "status": "ok", "error": None}
            with (
                patch.object(probe, "PROC_ROOT", proc),
                patch.object(collector, "_container", return_value=container),
                patch.object(collector, "_cache_bundle",
                             return_value={"cache": cache_value, "process": process}),
                patch.object(collector, "_periodic", return_value=None),
                patch.object(probe, "meminfo", return_value={"mem_available_bytes": 2,
                                                             "swap_total_bytes": 3,
                                                             "swap_free_bytes": 4}),
                patch.object(probe, "pressure", return_value={}),
                patch.object(probe, "tcp_connections", return_value={}),
                patch.object(probe, "_counter", return_value=0),
            ):
                collector.start()
                result = collector.tick(probe.MONOTONIC_NS(), None, 0)
                collector.close()
            self.assertEqual(result["container_state"], "running")
            self.assertEqual(result["container_state_evidence"], "verified")
            self.assertTrue(result["cache"]["sample_fresh"])

    def test_stale_or_reused_inspect_pid_never_reports_running(self):
        collector = probe.PersistentProbe(_args())
        container = {"pid": 11, "state": "running", "root_start_ticks": 111}
        self.assertIsNone(collector._validated_container_state(container, {"status": "stale"})[0])
        with patch.object(probe, "_read", return_value=_stat(11, 222)):
            state, evidence = collector._validated_container_state(container, {"status": "current"})
        self.assertIsNone(state)
        self.assertEqual(evidence, "root_identity_mismatch")

    def test_close_has_one_global_bounded_join_deadline(self):
        collector = probe.PersistentProbe(_args())
        blocker = threading.Event()
        for _ in range(3):
            thread = threading.Thread(target=blocker.wait, args=(2,), daemon=True)
            thread.start()
            collector._threads.append(thread)
        started = time.monotonic()
        collector.close()
        elapsed = time.monotonic() - started
        blocker.set()
        self.assertLess(elapsed, 0.6)


class ApiCompatibilityTests(unittest.TestCase):
    def test_api_gauges_sum_requests_and_accept_kv_cache_alias(self):
        payload = (b'vllm:num_requests_running{engine="0"} 1\n'
                   b'vllm:num_requests_running{engine="1"} 2\n'
                   b'vllm:num_requests_waiting 4\n'
                   b'vllm:kv_cache_usage_perc 0.5\n')
        with patch.object(probe, "_fetch", return_value=(payload, None, 200)):
            result = probe.api_metrics(8000)
        self.assertEqual(result["requests_running"], 3.0)
        self.assertEqual(result["requests_waiting"], 4.0)
        self.assertEqual(result["gpu_cache_usage_perc"], 0.5)

    def test_sample_accepts_legacy_namespace_without_new_attribute(self):
        args = argparse.Namespace(container="x", cache_root="/missing", rank=1,
                                  api_port=8000, interval=1.0, once=True)
        with patch.object(probe, "_container_info", return_value={
            "pid": None, "state": None, "root_start_ticks": None,
            "status": "unknown", "error": "test",
        }), patch.object(probe, "container_logs", return_value={
            "cuda": probe.gpu(), "cuda_ooms": None, "store_failures": None,
            "restore_fallbacks": None,
        }), patch.object(probe, "earlyoom_evidence", return_value={
            "status": "unknown", "earlyoom_kills": None,
        }):
            result = probe.sample(args)
        self.assertEqual(result["schema"], probe.TELEMETRY_SCHEMA)
        self.assertIn("cache", result)


if __name__ == "__main__":
    unittest.main()
