#!/usr/bin/env python3
"""Emit bounded host/process/cache telemetry without double-counting GB10 memory."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import errno
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Callable
import urllib.error
import urllib.request


PROC_ROOT = Path("/proc")
CGROUP_ROOT = Path("/sys/fs/cgroup")
MONOTONIC_NS = time.monotonic_ns
TIME_NS = time.time_ns
TELEMETRY_SCHEMA = "tp4-resilience-telemetry/v1"
UNKNOWN_CUDA = {"cuda_free_bytes": None, "cuda_used_bytes": None}
ADMISSION_LOG_PREFIXES = ("TP4_ADMISSION_READY ", "TP4_ADMISSION_STATE ")
ADMISSION_RECORD_FIELDS = (
    "active", "queued", "inflight", "admitted_total", "rejected_total",
    "rejected_full", "rejected_queue_timeout", "request_timeout", "body_too_large",
    "body_idle", "send_idle",
)
ADMISSION_CUMULATIVE_FIELDS = (
    "admitted_total", "rejected_total", "rejected_full", "rejected_queue_timeout",
    "request_timeout", "body_too_large", "body_idle", "send_idle",
)
ADMISSION_API_FIELDS = {
    "active": "admission_requests_active",
    "queued": "admission_requests_waiting",
    "inflight": "admission_requests_inflight",
    "admitted_total": "admission_admitted_total",
    "rejected_total": "admission_rejected_total",
    "rejected_full": "admission_rejected_full",
    "rejected_queue_timeout": "admission_rejected_queue_timeout",
    "request_timeout": "admission_request_timeout",
    "body_too_large": "admission_body_too_large",
    "body_idle": "admission_body_idle",
    "send_idle": "admission_send_idle",
}
UNKNOWN_ADMISSION_API = {name: None for name in ADMISSION_API_FIELDS.values()}
UNKNOWN_PROCESS: dict[str, Any] = {
    "rss_bytes": None, "rss_anon_bytes": None, "processes": None,
    "threads": None, "fds": None, "identities": [],
    "identity_status": "container_unknown", "identities_complete": False,
    "fds_complete": False, "membership_source": None,
    "rss_semantics": "nonadditive_process_sum",
    "cgroup_memory": {"status": "unknown", "path": None, "current_bytes": None,
                      "anon_bytes": None, "file_bytes": None, "shmem_bytes": None,
                      "events": None},
}
UNKNOWN_CACHE: dict[str, Any] = {
    "bytes": None, "bytes_complete": False, "scan_complete": False,
    "sample_fresh": False, "bytes_lower_bound": None, "files_seen": None,
    "manifests": None, "staging_bytes": None, "named_staging_bytes": None,
    "anonymous_staging_bytes": None, "anonymous_staging_fds": None,
    "anonymous_staging_complete": False, "accounting_status": "unknown",
    "reservation_owners": [], "reserved_bytes": None,
    "reservation_status": "unknown", "operations": {
        "status": "unknown", "coverage_complete": False, "events": None,
        "io_errors": None, "stages": {}},
    "scan_started_monotonic_ns": None, "scan_completed_monotonic_ns": None,
}


def _read_result(path: Path) -> tuple[str | None, str | None]:
    try:
        return path.read_text(encoding="ascii", errors="replace"), None
    except FileNotFoundError:
        return None, "missing"
    except OSError as error:
        return None, f"{type(error).__name__}:{error.errno}"


def _read(path: Path) -> str | None:
    return _read_result(path)[0]


def meminfo() -> dict[str, int | None]:
    result: dict[str, int | None] = {
        "mem_available_bytes": None,
        "swap_total_bytes": None,
        "swap_free_bytes": None,
    }
    text = _read(PROC_ROOT / "meminfo")
    if text is None:
        return result
    names = {"MemAvailable": "mem_available_bytes", "SwapTotal": "swap_total_bytes",
             "SwapFree": "swap_free_bytes"}
    for line in text.splitlines():
        match = re.fullmatch(r"([A-Za-z]+):\s+(\d+) kB", line)
        if match and match.group(1) in names:
            result[names[match.group(1)]] = int(match.group(2)) * 1024
    return result


def pressure() -> dict[str, dict[str, float | int]] | None:
    text = _read(PROC_ROOT / "pressure" / "memory")
    if text is None:
        return None
    result: dict[str, dict[str, float | int]] = {}
    try:
        for line in text.splitlines():
            fields = line.split()
            row: dict[str, float | int] = {}
            for field in fields[1:]:
                key, value = field.split("=", 1)
                row[key] = int(value) if key == "total" else float(value)
            result[fields[0]] = row
    except (IndexError, ValueError):
        return None
    return result


def _counter(path: Path, name: str) -> int | None:
    text = _read(path)
    if text is None:
        return None
    for line in text.splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0] == name:
            try:
                return int(fields[1])
            except ValueError:
                return None
    return None


def _terminate_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.communicate(timeout=0.2)
        return
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except OSError:
        pass
    try:
        process.communicate(timeout=0.2)
    except subprocess.TimeoutExpired:
        pass


def _command(command: list[str], timeout: float) -> tuple[str | None, str | None]:
    """Run one root-owned command and kill its complete process group on timeout."""
    try:
        process = subprocess.Popen(
            command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except OSError as error:
        return None, type(error).__name__
    try:
        output, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _terminate_group(process)
        return None, "timeout"
    if process.returncode != 0:
        return None, f"exit_{process.returncode}"
    return output, None


def _start_ticks(stat: str | None) -> int | None:
    if stat is None:
        return None
    end = stat.rfind(")")
    try:
        return int(stat[end + 2:].split()[19])
    except (IndexError, ValueError):
        return None


def _cgroup_directory(pid: int) -> tuple[Path | None, str | None]:
    text, error = _read_result(PROC_ROOT / str(pid) / "cgroup")
    if text is None:
        return None, error
    for line in text.splitlines():
        fields = line.split(":", 2)
        if len(fields) == 3 and fields[0] == "0" and fields[1] == "":
            relative = fields[2].lstrip("/")
            candidate = (CGROUP_ROOT / relative).resolve(strict=False)
            base = CGROUP_ROOT.resolve(strict=False)
            try:
                if os.path.commonpath((str(base), str(candidate))) != str(base):
                    return None, "invalid_cgroup_path"
            except ValueError:
                return None, "invalid_cgroup_path"
            return candidate, None
    return None, "unified_cgroup_missing"


def _cgroup_pids(directory: Path, limit: int = 4096) -> tuple[set[int], bool, str | None]:
    pids: set[int] = set()
    complete = True
    first_error: str | None = None

    def failed(error: OSError) -> None:
        nonlocal complete, first_error
        complete = False
        first_error = first_error or f"{type(error).__name__}:{error.errno}"

    try:
        walker = os.walk(directory, followlinks=False, onerror=failed)
        for current, _, _ in walker:
            text, error = _read_result(Path(current) / "cgroup.procs")
            if text is None:
                if error != "missing":
                    complete = False
                    first_error = first_error or error
                continue
            for value in text.split():
                try:
                    pids.add(int(value))
                except ValueError:
                    complete = False
                    first_error = first_error or "invalid_cgroup_pid"
                if len(pids) > limit:
                    return set(sorted(pids)[:limit]), False, "cgroup_pid_limit"
    except OSError as error:
        failed(error)
    return pids, complete, first_error


def _process_row(pid: int) -> tuple[dict[str, Any] | None, str | None]:
    text, error = _read_result(PROC_ROOT / str(pid) / "status")
    if text is None:
        return None, error
    fields: dict[str, str] = {}
    for line in text.splitlines():
        if ":" in line:
            name, value = line.split(":", 1)
            fields[name] = value.strip()
    stat, stat_error = _read_result(PROC_ROOT / str(pid) / "stat")
    start_ticks = _start_ticks(stat)
    if start_ticks is None:
        return None, stat_error or "invalid_stat"
    try:
        namespace_pids = [int(value) for value in fields.get("NSpid", str(pid)).split()]
        return {
            "host_pid": pid,
            "parent_pid": int(fields["PPid"]),
            "container_pid": namespace_pids[-1],
            "namespace_pids": namespace_pids,
            "start_ticks": start_ticks,
            "name": fields.get("Name"),
            "rss_bytes": int(fields.get("VmRSS", "0 kB").split()[0]) * 1024,
            "rss_anon_bytes": int(fields.get("RssAnon", "0 kB").split()[0]) * 1024,
            "threads": int(fields.get("Threads", "0")),
        }, None
    except (KeyError, ValueError, IndexError):
        return None, "invalid_status"


def _ancestry_pids(root_pid: int) -> tuple[set[int], bool]:
    rows: dict[int, int] = {}
    try:
        directories = list(PROC_ROOT.iterdir())
    except OSError:
        return set(), False
    complete = True
    for directory in directories:
        if not directory.name.isdigit():
            continue
        text, error = _read_result(directory / "status")
        if text is None:
            complete = complete and error == "missing"
            continue
        match = re.search(r"^PPid:\s*(\d+)$", text, re.MULTILINE)
        if match is None:
            complete = False
            continue
        rows[int(directory.name)] = int(match.group(1))
    selected: set[int] = set()
    for pid in rows:
        cursor = pid
        seen: set[int] = set()
        while cursor in rows and cursor not in seen:
            if cursor == root_pid:
                selected.add(pid)
                break
            seen.add(cursor)
            cursor = rows[cursor]
    return selected, complete


def _cgroup_memory(directory: Path | None) -> dict[str, Any]:
    unknown = {
        "status": "unknown", "path": str(directory) if directory else None,
        "current_bytes": None, "anon_bytes": None, "file_bytes": None,
        "shmem_bytes": None, "events": None,
    }
    if directory is None:
        return unknown
    current = _read(directory / "memory.current")
    stat = _read(directory / "memory.stat")
    events = _read(directory / "memory.events")
    if current is None or stat is None or events is None:
        return unknown
    try:
        stats = {line.split()[0]: int(line.split()[1]) for line in stat.splitlines()}
        event_values = {line.split()[0]: int(line.split()[1]) for line in events.splitlines()}
        return {
            "status": "ok", "path": str(directory), "current_bytes": int(current),
            "anon_bytes": stats.get("anon"), "file_bytes": stats.get("file"),
            "shmem_bytes": stats.get("shmem"), "events": event_values,
        }
    except (IndexError, ValueError):
        return unknown


def _process_snapshot(root_pid: int | None, *, include_fds: bool = True) -> dict[str, Any]:
    empty = copy.deepcopy(UNKNOWN_PROCESS)
    if not root_pid:
        return empty
    cgroup, cgroup_error = _cgroup_directory(root_pid)
    if cgroup is not None:
        pids, membership_complete, membership_error = _cgroup_pids(cgroup)
        membership_source = "cgroup_v2"
    elif cgroup_error in {"missing", "unified_cgroup_missing"}:
        pids, _ = _ancestry_pids(root_pid)
        # Ancestry is diagnostic only: docker exec processes need not descend
        # from the inspected PID, so absence can never prove an owner dead.
        membership_complete = False
        membership_error = "cgroup_unavailable"
        membership_source = "parent_ancestry_fallback"
    else:
        pids, membership_complete, membership_error = {root_pid}, False, cgroup_error
        membership_source = "cgroup_unreadable"
    if root_pid not in pids:
        value = dict(empty)
        value.update({
            "identity_status": "container_pid_missing",
            "membership_source": membership_source,
            "membership_error": membership_error,
            "cgroup_memory": _cgroup_memory(cgroup),
        })
        return value
    identities: list[dict[str, Any]] = []
    rss = rss_anon = threads = 0
    rows_complete = True
    fd_count = 0
    fds_complete = include_fds
    for pid in sorted(pids):
        row, error = _process_row(pid)
        if row is None:
            if error != "missing":
                rows_complete = False
            continue
        rss += row["rss_bytes"]
        rss_anon += row["rss_anon_bytes"]
        threads += row["threads"]
        identities.append({key: row[key] for key in (
            "host_pid", "container_pid", "namespace_pids", "start_ticks", "name")})
        if include_fds:
            try:
                fd_count += len(list((PROC_ROOT / str(pid) / "fd").iterdir()))
            except OSError:
                fds_complete = False
    truncated = len(identities) > 4096
    identities = identities[:4096]
    identities_complete = membership_complete and rows_complete and not truncated
    return {
        "rss_bytes": rss,
        "rss_anon_bytes": rss_anon,
        "rss_semantics": "nonadditive_process_sum",
        "processes": len(identities),
        "threads": threads,
        "fds": fd_count if include_fds and fds_complete else None,
        "fds_complete": fds_complete,
        "identities": identities,
        "identities_truncated": truncated,
        "identities_complete": identities_complete,
        "identity_status": "ok" if rows_complete else "partial",
        "membership_source": membership_source,
        "membership_error": membership_error,
        "cgroup_memory": _cgroup_memory(cgroup),
    }


def process_tree(root_pid: int | None) -> dict[str, Any]:
    """Backward-compatible process collector with cgroup membership evidence."""
    return _process_snapshot(root_pid)


def _boot_id() -> str | None:
    value = _read(PROC_ROOT / "sys" / "kernel" / "random" / "boot_id")
    if value is None:
        return None
    value = value.strip()
    pattern = r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}"
    return value if re.fullmatch(pattern, value) else None


def _rfc3339_time_ns(value: str | None) -> int | None:
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,9}))?Z", value)
    if match is None:
        return None
    try:
        seconds = int(datetime.strptime(match.group(1), "%Y-%m-%dT%H:%M:%S")
                      .replace(tzinfo=timezone.utc).timestamp())
    except (OverflowError, ValueError):
        return None
    fraction = (match.group(2) or "").ljust(9, "0")
    return seconds * 1_000_000_000 + int(fraction or "0")


def _start_ticks_monotonic_ns(start_ticks: int) -> int | None:
    try:
        ticks_per_second = int(os.sysconf("SC_CLK_TCK"))
    except (OSError, ValueError):
        return None
    if ticks_per_second <= 0:
        return None
    return start_ticks * 1_000_000_000 // ticks_per_second


def _container_info(container: str) -> dict[str, Any]:
    output, error = _command(
        ["docker", "inspect", "--format",
         "{{.State.Pid}} {{.State.Status}} {{.State.StartedAt}}", container], 2.0,
    )
    try:
        assert output is not None
        fields = output.split()
        pid = int(fields[0])
        pid = pid if pid > 0 else None
        started_at = fields[2] if len(fields) > 2 else None
        return {
            "pid": pid, "state": fields[1], "root_start_ticks": (
                _start_ticks(_read(PROC_ROOT / str(pid) / "stat")) if pid else None),
            "boot_id": _boot_id(), "started_at": started_at,
            "started_at_time_ns": _rfc3339_time_ns(started_at),
            "status": "ok" if pid else "unknown", "error": None,
        }
    except (AssertionError, ValueError, IndexError):
        return {"pid": None, "state": None, "root_start_ticks": None,
                "boot_id": None, "started_at": None, "started_at_time_ns": None,
                "status": "unknown", "error": error or "invalid_inspect"}


def container_pid(container: str) -> tuple[int | None, str | None]:
    value = _container_info(container)
    return value["pid"], value["state"]


def tcp_connections(port: int) -> dict[str, int]:
    counts = {"established": 0, "listen": 0, "other": 0}
    for name in ("tcp", "tcp6"):
        text = _read(PROC_ROOT / "net" / name)
        if text is None:
            continue
        for line in text.splitlines()[1:]:
            fields = line.split()
            if len(fields) < 4:
                continue
            try:
                local_port = int(fields[1].split(":")[1], 16)
            except (IndexError, ValueError):
                continue
            if local_port == port:
                state = fields[3]
                key = "established" if state == "01" else "listen" if state == "0A" else "other"
                counts[key] += 1
    return counts


def _under(root: Path, value: str) -> bool:
    try:
        return os.path.commonpath((str(root), value)) == str(root)
    except ValueError:
        return False


def _anonymous_staging(
    host_root: Path, container_root: Path, process: dict[str, Any],
) -> tuple[int | None, int | None, bool, str | None]:
    if process.get("identity_status") not in {"ok", "partial"}:
        return None, None, False, "identity_unavailable"
    try:
        host_device = host_root.stat().st_dev
    except OSError as error:
        return None, None, False, f"host_root_{type(error).__name__}:{error.errno}"
    seen: set[tuple[int, int]] = set()
    total = descriptors = 0
    complete = bool(process.get("identities_complete"))
    first_error: str | None = None
    for identity in process.get("identities", []):
        fd_root = PROC_ROOT / str(identity["host_pid"]) / "fd"
        try:
            entries = list(fd_root.iterdir())
        except FileNotFoundError:
            continue
        except OSError as error:
            complete = False
            first_error = first_error or f"{type(error).__name__}:{error.errno}"
            continue
        for descriptor in entries:
            try:
                target = os.readlink(descriptor)
                metadata = descriptor.stat()
            except FileNotFoundError:
                continue
            except OSError as error:
                complete = False
                first_error = first_error or f"{type(error).__name__}:{error.errno}"
                continue
            if not target.endswith(" (deleted)"):
                continue
            container_path = target[:-10]
            if not _under(container_root, container_path) or metadata.st_dev != host_device:
                continue
            identity_key = (metadata.st_dev, metadata.st_ino)
            if identity_key not in seen:
                seen.add(identity_key)
                descriptors += 1
                total += metadata.st_size
    if not complete:
        return None, None, False, first_error or "identity_incomplete"
    return total, descriptors, True, None


def _reservation_snapshot(root: Path) -> dict[str, Any]:
    directory = root / ".tp4-resilience" / "reservations"
    try:
        with os.scandir(directory) as entries:
            paths = [Path(entry.path) for entry in entries
                     if entry.name.endswith(".json") and entry.is_file(follow_symlinks=False)]
    except FileNotFoundError:
        return {"status": "instrumentation_missing", "records": [], "count": None,
                "invalid": 0, "unreadable": 0, "error": None}
    except OSError as error:
        return {"status": "instrumentation_unreadable", "records": [], "count": None,
                "invalid": 0, "unreadable": 0,
                "error": f"{type(error).__name__}:{error.errno}"}
    records: list[dict[str, Any]] = []
    invalid = unreadable = 0
    for control in paths:
        try:
            value = json.loads(control.read_text(encoding="utf-8"))
            value["pid"] = int(value["pid"])
            value["process_start_ticks"] = int(value["process_start_ticks"])
            value["reserved_bytes"] = int(value["reserved_bytes"])
            if value["reserved_bytes"] < 0:
                raise ValueError("negative reservation")
            records.append(value)
        except PermissionError:
            unreadable += 1
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            invalid += 1
    return {"status": "ok" if not invalid and not unreadable else "partial",
            "records": records, "count": len(paths), "invalid": invalid,
            "unreadable": unreadable, "error": None}


def _reservations(
    root: Path, process: dict[str, Any], snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    snapshot = snapshot if snapshot is not None else _reservation_snapshot(root)
    base = {
        "reservation_owners": [], "reserved_bytes": None,
        "stale_reservation_records": 0, "dead_owner_records": 0,
        "unverifiable_reservation_records": 0, "reservation_records": None,
        "verified_owner_count": 0,
    }
    if snapshot["status"] == "instrumentation_missing":
        return {**base, "reservation_status": "instrumentation_missing"}
    if snapshot["status"] == "instrumentation_unreadable":
        return {**base, "reservation_status": "instrumentation_unreadable",
                "reservation_error": snapshot.get("error")}
    base["reservation_records"] = snapshot["count"]
    if snapshot["count"] == 0:
        return {**base, "reservation_status": "owner_records_missing"}
    if process.get("identity_status") not in {"ok", "partial"}:
        base["unverifiable_reservation_records"] = snapshot["count"]
        return {**base, "reservation_status": "owner_mapping_unknown"}
    by_inner_pid: dict[int, list[dict[str, Any]]] = {}
    for identity in process.get("identities", []):
        by_inner_pid.setdefault(int(identity["container_pid"]), []).append(identity)
    owners: list[dict[str, Any]] = []
    dead = 0
    invalid = int(snapshot["invalid"])
    unverifiable = int(snapshot["unreadable"])
    complete_identity = bool(process.get("identities_complete"))
    for value in snapshot["records"]:
        inner_pid = value["pid"]
        expected_start = value["process_start_ticks"]
        matches = by_inner_pid.get(inner_pid, [])
        if any(owner.get("start_ticks") == expected_start for owner in matches):
            owners.append(value)
        elif complete_identity:
            dead += 1
        else:
            unverifiable += 1
    base.update({
        "reservation_owners": owners,
        "stale_reservation_records": dead,
        "dead_owner_records": dead,
        "unverifiable_reservation_records": unverifiable + invalid,
        "invalid_reservation_records": invalid,
        "verified_owner_count": len(owners),
    })
    if not owners:
        status = "dead_owners_only" if dead and not invalid and not unverifiable else "no_verified_owners"
        return {**base, "reservation_status": status}
    base["reserved_bytes"] = sum(int(value["reserved_bytes"]) for value in owners)
    base["reservation_status"] = "ok" if not invalid and not unverifiable else "partial"
    return base


def _tail_lines(path: Path, limit: int) -> tuple[list[str] | None, bool, str | None]:
    try:
        with path.open("rb") as stream:
            size = stream.seek(0, os.SEEK_END)
            truncated = size > limit
            stream.seek(max(0, size - limit))
            payload = stream.read(limit)
    except FileNotFoundError:
        return None, False, "missing"
    except OSError as error:
        return None, False, f"{type(error).__name__}:{error.errno}"
    if truncated:
        payload = payload.split(b"\n", 1)[-1]
    return payload.decode("utf-8", "replace").splitlines(), not truncated, None


def operation_events(root: Path) -> dict[str, Any]:
    control = root / ".tp4-resilience"
    all_lines: list[str] = []
    coverage_complete = True
    found = False
    for name in ("events.previous.jsonl", "events.jsonl"):
        lines, complete, error = _tail_lines(control / name, 2 << 20)
        if lines is None:
            if error == "missing" and name == "events.previous.jsonl":
                continue
            if error == "missing":
                return {"status": "instrumentation_missing", "coverage_complete": False,
                        "events": None, "io_errors": None, "stages": {}}
            return {"status": "unreadable", "coverage_complete": False,
                    "events": None, "io_errors": None, "stages": {}, "error": error}
        found = True
        all_lines.extend(lines)
        coverage_complete = coverage_complete and complete
    if not found:
        return {"status": "instrumentation_missing", "coverage_complete": False,
                "events": None, "io_errors": None, "stages": {}}
    begins: dict[tuple[str, str], dict[str, Any]] = {}
    counts: dict[str, dict[str, int | None]] = {}
    io_errors = invalid = 0
    for line in all_lines:
        try:
            event = json.loads(line)
            kind, stage = event.get("kind"), event.get("stage")
            operation_id = event.get("operation_id")
            if kind in {"stage_begin", "stage_end"} and isinstance(stage, str):
                row = counts.setdefault(stage, {"begun": 0, "ended": 0, "active": 0})
                key_name = "begun" if kind == "stage_begin" else "ended"
                row[key_name] = int(row[key_name] or 0) + 1
                if isinstance(operation_id, str):
                    key = (stage, operation_id)
                    if kind == "stage_begin":
                        begins[key] = event
                    else:
                        begins.pop(key, None)
            if kind == "fault_hit" and event.get("mode") in {"eio", "enospc", "publication_eio"}:
                io_errors += 1
        except (json.JSONDecodeError, AttributeError):
            invalid += 1
    for stage, _ in begins:
        counts.setdefault(stage, {"begun": 0, "ended": 0, "active": 0})["active"] = (
            int(counts[stage]["active"] or 0) + 1)
    if not coverage_complete:
        for row in counts.values():
            row["active"] = None
    return {
        "status": "ok" if not invalid and coverage_complete else "partial",
        "coverage_complete": coverage_complete,
        "events": len(all_lines) - invalid,
        "invalid_events": invalid,
        "io_errors": io_errors,
        "stages": counts,
    }


def cache(
    root: Path,
    process: dict[str, Any] | None = None,
    *,
    container_root: Path | None = None,
    reservation_snapshot: dict[str, Any] | None = None,
    scan_started_monotonic_ns: int | None = None,
    deadline_ns: int | None = None,
    max_files: int = 200_000,
) -> dict[str, Any]:
    scan_started_monotonic_ns = (
        MONOTONIC_NS() if scan_started_monotonic_ns is None else scan_started_monotonic_ns)
    process = process or {"identity_status": "container_unknown", "identities": []}
    container_root = container_root or root
    try:
        root_metadata = root.stat()
        if not root.is_dir() or root.is_symlink():
            raise NotADirectoryError(errno.ENOTDIR, "cache root is not a plain directory", root)
    except OSError as error:
        reservations = _reservations(root, process, reservation_snapshot)
        return {
            "bytes": None, "bytes_complete": False, "scan_complete": False,
            "sample_fresh": True, "bytes_lower_bound": None, "files_seen": None,
            "manifests": None, "staging_bytes": None, "named_staging_bytes": None,
            "anonymous_staging_bytes": None, "anonymous_staging_fds": None,
            "anonymous_staging_complete": False, "accounting_status": "root_unavailable",
            "accounting_error": f"{type(error).__name__}:{error.errno}",
            "scan_started_monotonic_ns": scan_started_monotonic_ns,
            "scan_completed_monotonic_ns": MONOTONIC_NS(),
            **reservations, "operations": operation_events(root),
        }
    del root_metadata
    total = manifests = named_staging = files_seen = 0
    complete = True
    walk_error: str | None = None

    def walk_failed(error: OSError) -> None:
        nonlocal complete, walk_error
        if error.errno != errno.ENOENT:
            complete = False
            walk_error = walk_error or f"{type(error).__name__}:{error.errno}"

    for directory, _, files in os.walk(root, followlinks=False, onerror=walk_failed):
        for name in files:
            files_seen += 1
            if files_seen > max_files or (deadline_ns is not None and MONOTONIC_NS() > deadline_ns):
                complete = False
                walk_error = walk_error or "scan_limit"
                break
            path = Path(directory, name)
            try:
                metadata = path.lstat()
                if path.is_symlink():
                    continue
            except FileNotFoundError:
                continue
            except OSError as error:
                complete = False
                walk_error = walk_error or f"{type(error).__name__}:{error.errno}"
                continue
            total += metadata.st_size
            if "/manifests/" in str(path) and name.endswith(".json"):
                manifests += 1
            if ".writing-" in name or name.startswith(".snapshot"):
                named_staging += metadata.st_size
        if not complete:
            break
    anonymous, anonymous_fds, anonymous_complete, anonymous_error = _anonymous_staging(
        root, container_root, process)
    reservations = _reservations(root, process, reservation_snapshot)
    scan_complete = complete and anonymous_complete
    accounted = total + anonymous if anonymous is not None else total
    return {
        "bytes": accounted,
        "bytes_complete": scan_complete,
        "scan_complete": scan_complete,
        "sample_fresh": True,
        "bytes_lower_bound": accounted,
        "files_seen": files_seen,
        "manifests": manifests,
        "staging_bytes": named_staging + anonymous if anonymous is not None else None,
        "named_staging_bytes": named_staging,
        "anonymous_staging_bytes": anonymous,
        "anonymous_staging_fds": anonymous_fds,
        "anonymous_staging_complete": anonymous_complete,
        "accounting_status": "ok" if scan_complete else "partial",
        "accounting_error": walk_error or anonymous_error,
        "scan_started_monotonic_ns": scan_started_monotonic_ns,
        "scan_completed_monotonic_ns": MONOTONIC_NS(),
        **reservations,
        "operations": operation_events(root),
    }


def _fetch(url: str, timeout: float, limit: int) -> tuple[bytes | None, str | None, int | None]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read(limit), None, getattr(response, "status", 200)
    except urllib.error.HTTPError as error:
        return None, f"http_{error.code}", error.code
    except (OSError, ValueError) as error:
        return None, type(error).__name__, None


def api_metrics(port: int, timeout: float = 2.0) -> dict[str, float | None]:
    aliases = {
        "vllm:num_requests_running": "requests_running",
        "vllm:num_requests_waiting": "requests_waiting",
        "vllm:gpu_cache_usage_perc": "gpu_cache_usage_perc",
        "vllm:kv_cache_usage_perc": "gpu_cache_usage_perc",
    }
    result: dict[str, float | None] = {
        "requests_running": None, "requests_waiting": None, "gpu_cache_usage_perc": None,
    }
    payload, _, _ = _fetch(f"http://127.0.0.1:{port}/metrics", timeout, 4 << 20)
    if payload is None:
        return result
    found: dict[str, list[float]] = {name: [] for name in result}
    for line in payload.decode("utf-8", "replace").splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(None, 1)[0]
        if name in aliases:
            try:
                found[aliases[name]].append(float(line.rsplit(None, 1)[1]))
            except (IndexError, ValueError):
                pass
    for name, values in found.items():
        if values:
            result[name] = sum(values) if name.startswith("requests_") else max(values)
    return result


def health(port: int, timeout: float = 2.0) -> dict[str, Any]:
    _, error, status = _fetch(f"http://127.0.0.1:{port}/health", timeout, 64 << 10)
    return {"healthy": status == 200, "http_status": status,
            "status": "ok" if status == 200 else "unavailable", "error": error}


def _since_text(time_ns: int) -> str:
    return datetime.fromtimestamp(time_ns / 1_000_000_000, timezone.utc).isoformat()


def _admission_log_state(lines: list[str]) -> dict[str, Any]:
    """Return only the latest complete admission record from bounded log lines."""
    state: dict[str, Any] | None = None
    status = "not_observed"
    markers = invalid = 0
    ready_seen = False
    for line in lines:
        prefix = next((value for value in ADMISSION_LOG_PREFIXES if value in line), None)
        if prefix is None:
            continue
        markers += 1
        try:
            raw = json.loads(line.split(prefix, 1)[1])
            if not isinstance(raw, dict):
                raise TypeError("record")
            if prefix == "TP4_ADMISSION_READY ":
                if raw.get("schema") != "tp4_admission_v1":
                    raise ValueError("schema")
                ready_seen = True
                if not all(name in raw for name in (
                    "monotonic_ns", *ADMISSION_RECORD_FIELDS,
                )):
                    # A later READY marker can denote a restarted middleware instance.
                    # Without its gauges, an older state is no longer authoritative.
                    state = None
                    status = "ready_without_state"
                    continue
                raw = {**raw, "event": raw.get("event", "ready")}
            schema, event = raw.get("schema"), raw.get("event")
            monotonic_ns = raw.get("monotonic_ns")
            if schema != "tp4_admission_v1" or not isinstance(event, str) or not event:
                raise ValueError("identity")
            if (not isinstance(monotonic_ns, int) or isinstance(monotonic_ns, bool)
                    or monotonic_ns < 0):
                raise ValueError("monotonic_ns")
            counters: dict[str, int] = {}
            for name in ADMISSION_RECORD_FIELDS:
                value = raw.get(name)
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise ValueError(name)
                counters[name] = value
            if counters["inflight"] != counters["active"] + counters["queued"]:
                raise ValueError("inflight")
            if counters["rejected_total"] != (
                counters["rejected_full"] + counters["rejected_queue_timeout"]
            ):
                raise ValueError("rejected_total")
            state = {"schema": schema, "event": event,
                     "monotonic_ns": monotonic_ns, **counters}
            status = "ok"
        except (json.JSONDecodeError, TypeError, ValueError):
            invalid += 1
            # Any malformed newer admission marker makes an older state unsafe to expose.
            state = None
            status = "invalid"
    return {"status": status, "state": state, "markers": markers,
            "invalid_records": invalid, "ready_seen": ready_seen}


def _admission_api(state: dict[str, Any] | None) -> dict[str, int | None]:
    result = dict(UNKNOWN_ADMISSION_API)
    if state is not None:
        for source, target in ADMISSION_API_FIELDS.items():
            result[target] = state[source]
    return result


def _one_shot_admission(
    logs: dict[str, Any], container: dict[str, Any],
) -> tuple[dict[str, int | None], dict[str, Any]]:
    parsed = logs.get("admission") or {}
    state = parsed.get("state")
    pid, start_ticks = container.get("pid"), container.get("root_start_ticks")
    identity_ok = (
        container.get("state") == "running"
        and isinstance(container.get("boot_id"), str)
        and isinstance(pid, int)
        and isinstance(start_ticks, int)
        and _start_ticks(_read(PROC_ROOT / str(pid) / "stat")) == start_ticks
    )
    now = MONOTONIC_NS()
    state_age = (now - state["monotonic_ns"] if isinstance(state, dict) else None)
    root_started_ns = (_start_ticks_monotonic_ns(start_ticks)
                       if isinstance(start_ticks, int) else None)
    status = "current"
    if logs.get("status") != "ok":
        status = "unknown"
    elif (logs.get("coverage_complete") is not True
          and parsed.get("status") != "ok"):
        status = ("invalid" if parsed.get("status") == "invalid"
                  else "coverage_incomplete")
    elif not identity_ok:
        status = "identity_unknown"
    elif (parsed.get("status") != "ok" or state_age is None or state_age < 0
          or root_started_ns is None or state["monotonic_ns"] < root_started_ns):
        status = "invalid" if parsed.get("status") == "invalid" else "unknown"
    if status != "current":
        state = None
        state_age = None
    return _admission_api(state), {
        "status": status, "age_ns": 0, "sampled_monotonic_ns": now,
        "state_age_ns": state_age,
        "state_monotonic_ns": state.get("monotonic_ns") if state else None,
        "event": state.get("event") if state else None,
        "schema": state.get("schema") if state else None,
        "observed_in_latest_scan": parsed.get("status") == "ok",
        "evidence_status": parsed.get("status", "not_observed"),
        "log_tail_coverage_complete": logs.get("coverage_complete") is True,
        "container_identity_verified": identity_ok,
        "error": logs.get("error"),
    }


def container_logs(
    container: str,
    *,
    since_time_ns: int | None = None,
    valid_container_pids: set[int] | None = None,
) -> dict[str, Any]:
    command = ["docker", "logs"]
    if since_time_ns is not None:
        command.extend(("--since", _since_text(since_time_ns)))
    command.extend(("--tail", "5000", container))
    output, error = _command(command, 3.0)
    cuda = {**UNKNOWN_CUDA, "source": "E20_MEMORY_PROBE", "status": "unknown"}
    result = {
        "status": "unknown" if output is None else "ok", "error": error,
        "coverage_complete": None,
        "window_scope": "tail_5000" if since_time_ns is None else "since_time_tail_5000",
        "cuda": cuda, "cuda_ooms": None, "store_failures": None,
        "restore_fallbacks": None, "window_lines": None,
        "admission": {"status": "unknown", "state": None, "markers": None,
                      "invalid_records": None},
    }
    if output is None:
        return result
    lines = output.splitlines()
    result.update({"window_lines": len(lines), "coverage_complete": len(lines) < 5000,
                   "cuda_ooms": 0, "store_failures": 0, "restore_fallbacks": 0,
                   "admission": _admission_log_state(lines)})
    for line in lines:
        if "E20_MEMORY_PROBE " in line:
            try:
                record = json.loads(line.split("E20_MEMORY_PROBE ", 1)[1])
                wall_time = int(record["wall_time_ns"])
                record_pid = int(record["pid"])
                age_ns = max(0, TIME_NS() - wall_time)
                pid_ok = valid_container_pids is None or record_pid in valid_container_pids
                status = "ok" if age_ns <= 3_500_000_000 and pid_ok else (
                    "pid_mismatch" if not pid_ok else "stale")
                result["cuda"] = {
                    **UNKNOWN_CUDA, "source": "E20_MEMORY_PROBE", "status": status,
                    "allocator": record.get("cuda_allocator"),
                    "allocator_error": record.get("cuda_allocator_error"),
                    "host_pinned_allocator": record.get("host_pinned_allocator"),
                    "wall_time_ns": wall_time, "age_ns": age_ns, "pid": record_pid,
                    "sequence": record.get("sequence"), "peaks_scope": record.get("peaks_scope"),
                }
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                pass
        lowered = line.lower()
        if "cuda out of memory" in lowered or "torch.outofmemoryerror" in lowered:
            result["cuda_ooms"] += 1
        if "spark-context-cache: store failed (entry skipped):" in lowered:
            result["store_failures"] += 1
        if "spark-context-cache: pending-trace " in line:
            try:
                event = json.loads(line.split("spark-context-cache: pending-trace ", 1)[1])
                if event.get("event") == "fallback":
                    result["restore_fallbacks"] += 1
            except (json.JSONDecodeError, AttributeError):
                pass
    return result


def gpu(container: str | None = None) -> dict[str, Any]:
    return container_logs(container)["cuda"] if container else {
        **UNKNOWN_CUDA, "source": "E20_MEMORY_PROBE", "status": "unknown",
    }


def earlyoom_evidence(since_time_ns: int) -> dict[str, Any]:
    command = ["journalctl", "-u", "earlyoom", "--since", f"@{since_time_ns / 1e9:.3f}",
               "--no-pager", "-n", "2000", "-o", "json"]
    output, error = _command(command, 3.0)
    result = {"status": "unknown" if output is None else "ok", "error": error,
              "coverage_complete": None, "earlyoom_kills": None, "window_lines": None}
    if output is None:
        return result
    lines = output.splitlines()
    kills = 0
    invalid = 0
    for line in lines:
        try:
            event = json.loads(line)
            message = str(event.get("MESSAGE", ""))
            identifier = str(event.get("SYSLOG_IDENTIFIER", ""))
            if identifier == "earlyoom" and re.search(
                r"sending SIG(?:TERM|KILL) to process \d+", message, re.I
            ):
                kills += 1
        except (json.JSONDecodeError, AttributeError):
            invalid += 1
    result.update({"coverage_complete": len(lines) < 2000 and invalid == 0,
                   "earlyoom_kills": kills, "window_lines": len(lines),
                   "invalid_lines": invalid})
    return result


def _allocator_retries(logs: dict[str, Any]) -> int | None:
    allocator = logs.get("cuda", {}).get("allocator")
    value = allocator.get("num_alloc_retries") if isinstance(allocator, dict) else None
    return int(value) if isinstance(value, (int, float)) else None


def _sync_sample(args: argparse.Namespace) -> dict[str, Any]:
    started_mono, started_time = MONOTONIC_NS(), TIME_NS()
    host = meminfo()
    memory_pressure = pressure()
    host_oom = _counter(PROC_ROOT / "vmstat", "oom_kill")
    container = _container_info(args.container)
    cache_root = Path(args.cache_root)
    scan_started = MONOTONIC_NS()
    reservation_snapshot = _reservation_snapshot(cache_root)
    process = process_tree(container["pid"])
    container_root = Path(getattr(args, "container_cache_root", None) or args.cache_root)
    cache_value = cache(
        cache_root, process, container_root=container_root,
        reservation_snapshot=reservation_snapshot,
        scan_started_monotonic_ns=scan_started,
    )
    valid_pids = {int(row["container_pid"]) for row in process.get("identities", [])}
    logs = container_logs(args.container, since_time_ns=started_time,
                          valid_container_pids=valid_pids)
    earlyoom = earlyoom_evidence(started_time)
    value: dict[str, Any] = {
        "schema": TELEMETRY_SCHEMA, "time_ns": TIME_NS(), "monotonic_ns": started_mono,
        "rank": args.rank, "container_state": container["state"],
        "container_pid": container["pid"], **host, "memory_pressure": memory_pressure,
        "process": process, "connections": tcp_connections(args.api_port), "cache": cache_value,
        "cuda": logs["cuda"],
        "operations": {**cache_value["operations"],
                       "store_failures_log_window": logs["store_failures"],
                       "restore_fallbacks_log_window": logs["restore_fallbacks"]},
        "oom": {**earlyoom, "host_oom_kill_total": host_oom,
                "host_oom_kills_since_start": None,
                "cgroup_oom_kill_total": (process.get("cgroup_memory", {}).get("events") or {}).get("oom_kill"),
                "cgroup_oom_kills_since_start": None,
                "cuda_ooms_log_window": logs["cuda_ooms"],
                "cuda_allocator_retries": _allocator_retries(logs)},
    }
    if args.rank == 0:
        admission_api, admission_source = _one_shot_admission(logs, container)
        value["api"] = {**api_metrics(args.api_port), **admission_api}
        value["health"] = health(args.api_port)
        value["sources"] = {"admission": admission_source}
    value["collection"] = {"mode": "synchronous",
                           "sample_duration_ns": MONOTONIC_NS() - started_mono}
    return value


def sample(args: argparse.Namespace) -> dict[str, Any]:
    return _sync_sample(args)


class PersistentProbe:
    """Keep bounded command and filesystem work off the fixed-deadline path."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.interval_ns = int(args.interval * 1_000_000_000)
        self.started_time_ns = TIME_NS()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: dict[str, dict[str, Any]] = {}
        self._threads: list[threading.Thread] = []
        self._host_oom_baseline: int | None = None
        self._cgroup_oom_baseline: tuple[str, int] | None = None
        self._cuda_oom_seen: bool | None = None
        self._admission_identity: tuple[str, int, int] | None = None
        self._admission_state: dict[str, Any] | None = None
        self._admission_bootstrap_pending = args.rank == 0

    def _publish(self, name: str, value: Any, error: str | None = None) -> None:
        with self._lock:
            self._latest[name] = {"value": value, "sampled_monotonic_ns": MONOTONIC_NS(),
                                  "error": error}

    def _get(self, name: str, default: Any, stale_after_ns: int) -> tuple[Any, dict[str, Any]]:
        with self._lock:
            entry = copy.deepcopy(self._latest.get(name))
        if entry is None or entry["value"] is None:
            return copy.deepcopy(default), {
                "status": "error" if entry else "unknown", "age_ns": None,
                "sampled_monotonic_ns": None, "error": entry["error"] if entry else None,
            }
        age = max(0, MONOTONIC_NS() - entry["sampled_monotonic_ns"])
        status = "error" if entry["error"] else "stale" if age > stale_after_ns else "current"
        return entry["value"], {"status": status, "age_ns": age,
                                "sampled_monotonic_ns": entry["sampled_monotonic_ns"],
                                "error": entry["error"]}

    def _periodic(self, name: str, period: float, callback: Callable[[], Any]) -> None:
        deadline = MONOTONIC_NS()
        period_ns = int(period * 1_000_000_000)
        while not self._stop.is_set():
            try:
                self._publish(name, callback())
            except Exception as error:
                self._publish(name, None, type(error).__name__)
            deadline += period_ns
            deadline, _ = advance_deadline(deadline, MONOTONIC_NS(), period_ns)
            self._stop.wait(max(0, deadline - MONOTONIC_NS()) / 1_000_000_000)

    def _container(self) -> dict[str, Any]:
        return _container_info(self.args.container)

    def _cache_bundle(self) -> dict[str, Any]:
        container, _ = self._get("container", {"pid": None}, 4_000_000_000)
        root = Path(self.args.cache_root)
        scan_started = MONOTONIC_NS()
        # Reservation bytes are captured before the final membership snapshot;
        # an owner created before this scan cannot be misclassified as dead by
        # an identity list sampled earlier in the same bundle.
        reservation_snapshot = _reservation_snapshot(root)
        process = process_tree(container.get("pid"))
        value = cache(
            root, process,
            container_root=Path(
                getattr(self.args, "container_cache_root", None) or self.args.cache_root),
            reservation_snapshot=reservation_snapshot,
            scan_started_monotonic_ns=scan_started,
            deadline_ns=MONOTONIC_NS() + 750_000_000,
        )
        return {"cache": value, "process": process}

    def _endpoint(self) -> dict[str, Any]:
        return {"api": api_metrics(self.args.api_port, 0.5),
                "health": health(self.args.api_port, 0.5)}

    def _verified_container_identity(
        self, container: dict[str, Any], metadata: dict[str, Any],
    ) -> tuple[str, int, int] | None:
        state, _ = self._validated_container_state(container, metadata)
        boot_id = container.get("boot_id")
        pid, start_ticks = container.get("pid"), container.get("root_start_ticks")
        if (state != "running" or not isinstance(boot_id, str)
                or not isinstance(pid, int) or not isinstance(start_ticks, int)):
            return None
        return boot_id, pid, start_ticks

    def _bind_admission_state(
        self, logs: dict[str, Any], container: dict[str, Any], metadata: dict[str, Any],
    ) -> dict[str, Any]:
        identity = self._verified_container_identity(container, metadata)
        if identity is None:
            return {"status": "identity_unknown", "state": None, "identity": None,
                    "observed_in_scan": False, "state_age_ns": None}
        if identity != self._admission_identity:
            self._admission_identity = identity
            self._admission_state = None
        parsed = logs.get("admission") or {}
        coverage_complete = logs.get("coverage_complete") is True
        if logs.get("status") != "ok":
            self._admission_state = None
            return {
                "status": "collection_error", "state": None,
                "identity": {"boot_id": identity[0], "container_pid": identity[1],
                             "root_start_ticks": identity[2]},
                "observed_in_scan": False, "state_age_ns": None,
                "coverage_complete": coverage_complete,
            }
        if not coverage_complete and parsed.get("status") != "ok":
            self._admission_state = None
            return {
                "status": ("invalid" if parsed.get("status") == "invalid"
                           else "coverage_incomplete"),
                "state": None,
                "identity": {"boot_id": identity[0], "container_pid": identity[1],
                             "root_start_ticks": identity[2]},
                "observed_in_scan": False, "state_age_ns": None,
                "coverage_complete": False,
                "invalid_records": parsed.get("invalid_records"),
            }
        observed = parsed.get("status") == "ok"
        if observed:
            record = parsed.get("state")
            age = (MONOTONIC_NS() - record["monotonic_ns"]
                   if isinstance(record, dict) else -1)
            root_started_ns = _start_ticks_monotonic_ns(identity[2])
            if (age < 0 or root_started_ns is None
                    or record["monotonic_ns"] < root_started_ns):
                self._admission_state = None
                parsed = {"status": "invalid"}
            elif (self._admission_state is not None
                  and (record["monotonic_ns"] < self._admission_state["monotonic_ns"]
                       or any(
                    record[name] < self._admission_state[name]
                    for name in ADMISSION_CUMULATIVE_FIELDS
                  ))):
                self._admission_state = None
                parsed = {"status": "invalid"}
            else:
                self._admission_state = copy.deepcopy(record)
        elif parsed.get("status") == "invalid":
            self._admission_state = None
        state = copy.deepcopy(self._admission_state)
        age = (max(0, MONOTONIC_NS() - state["monotonic_ns"])
               if state is not None else None)
        return {
            "status": "ok" if state is not None else parsed.get("status", "not_observed"),
            "state": state,
            "identity": {"boot_id": identity[0], "container_pid": identity[1],
                         "root_start_ticks": identity[2]},
            "observed_in_scan": observed,
            "state_age_ns": age,
            "invalid_records": parsed.get("invalid_records"),
            "coverage_complete": coverage_complete,
        }

    def _evidence(self) -> dict[str, Any]:
        bundle, _ = self._get("cache_bundle", {"process": {}}, 2_000_000_000)
        container, container_meta = self._get(
            "container", {"pid": None, "state": None, "root_start_ticks": None,
                          "boot_id": None}, 4_000_000_000)
        valid_pids = {int(row["container_pid"])
                      for row in bundle.get("process", {}).get("identities", [])}
        bootstrap = self._admission_bootstrap_pending
        bootstrap_since = container.get("started_at_time_ns")
        logs = container_logs(self.args.container, since_time_ns=(
            bootstrap_since if bootstrap and isinstance(bootstrap_since, int)
            else self.started_time_ns),
                              valid_container_pids=valid_pids)
        self._admission_bootstrap_pending = logs.get("status") != "ok"
        logs["admission"] = self._bind_admission_state(logs, container, container_meta)
        if bootstrap:
            # The initial tail recovers the last same-container admission state.
            # Its older lines do not establish since-probe-start event counters.
            logs.update({"cuda_ooms": None, "store_failures": None,
                         "restore_fallbacks": None, "coverage_complete": None})
        if isinstance(logs.get("cuda_ooms"), int):
            if logs["cuda_ooms"] > 0:
                self._cuda_oom_seen = True
            elif logs.get("coverage_complete") is True and self._cuda_oom_seen is None:
                self._cuda_oom_seen = False
        logs["cuda_oom_seen_since_start"] = self._cuda_oom_seen
        return {
            "logs": logs,
            "earlyoom": earlyoom_evidence(self.started_time_ns),
        }

    def _prime(self) -> None:
        self._host_oom_baseline = _counter(PROC_ROOT / "vmstat", "oom_kill")
        container = self._container()
        self._publish("container", container)
        if container.get("pid"):
            cgroup_dir, _ = _cgroup_directory(int(container["pid"]))
            cgroup = _cgroup_memory(cgroup_dir)
            event_count = (cgroup.get("events") or {}).get("oom_kill")
            if cgroup.get("path") and isinstance(event_count, int):
                self._cgroup_oom_baseline = (cgroup["path"], event_count)
        bundle = self._cache_bundle()
        self._publish("cache_bundle", bundle)

    def start(self) -> None:
        self._prime()
        jobs = [
            ("container", 1.0, self._container),
            ("cache_bundle", 1.0, self._cache_bundle),
            ("evidence", 5.0, self._evidence),
        ]
        if self.args.rank == 0:
            jobs.append(("endpoint", max(1.0, self.args.interval), self._endpoint))
        for name, period, callback in jobs:
            thread = threading.Thread(target=self._periodic, args=(name, period, callback),
                                      name=f"tp4-probe-{name}", daemon=True)
            self._threads.append(thread)
            thread.start()

    def close(self) -> None:
        self._stop.set()
        deadline = MONOTONIC_NS() + 250_000_000
        for thread in self._threads:
            thread.join(max(0, deadline - MONOTONIC_NS()) / 1_000_000_000)

    def _validated_container_state(
        self, container: dict[str, Any], metadata: dict[str, Any],
    ) -> tuple[str | None, str]:
        if metadata.get("status") != "current" or container.get("state") != "running":
            return None, "stale_or_not_running"
        pid, expected = container.get("pid"), container.get("root_start_ticks")
        actual = _start_ticks(_read(PROC_ROOT / str(pid) / "stat")) if pid else None
        if actual is None or expected is None or actual != expected:
            return None, "root_identity_mismatch"
        return "running", "verified"

    def _admission_sample(
        self, admission: dict[str, Any], evidence_meta: dict[str, Any],
        container: dict[str, Any], container_meta: dict[str, Any],
    ) -> tuple[dict[str, int | None], dict[str, Any]]:
        current = self._verified_container_identity(container, container_meta)
        bound = admission.get("identity") or {}
        bound_identity = (bound.get("boot_id"), bound.get("container_pid"),
                          bound.get("root_start_ticks"))
        state = admission.get("state")
        status = evidence_meta.get("status", "unknown")
        evidence_status = admission.get("status", "not_observed")
        if status == "current":
            if current is None:
                status = "identity_unknown"
            elif bound_identity != current:
                status = "identity_mismatch"
            elif evidence_status != "ok" or not isinstance(state, dict):
                status = "invalid" if evidence_status == "invalid" else "unknown"
        if status != "current":
            state = None
        source = {
            "status": status,
            "age_ns": evidence_meta.get("age_ns"),
            "sampled_monotonic_ns": evidence_meta.get("sampled_monotonic_ns"),
            "error": evidence_meta.get("error"),
            "state_age_ns": (max(0, MONOTONIC_NS() - state["monotonic_ns"])
                             if state is not None else None),
            "state_monotonic_ns": state.get("monotonic_ns") if state else None,
            "event": state.get("event") if state else None,
            "schema": state.get("schema") if state else None,
            "observed_in_latest_scan": admission.get("observed_in_scan", False),
            "evidence_status": evidence_status,
            "log_tail_coverage_complete": admission.get("coverage_complete"),
            "container_identity_verified": current is not None and bound_identity == current,
        }
        return _admission_api(state), source

    def _oom(self, process: dict[str, Any], earlyoom: dict[str, Any]) -> dict[str, Any]:
        host_total = _counter(PROC_ROOT / "vmstat", "oom_kill")
        host_delta = (host_total - self._host_oom_baseline
                      if host_total is not None and self._host_oom_baseline is not None
                      and host_total >= self._host_oom_baseline else None)
        cgroup = process.get("cgroup_memory", {})
        cgroup_total = (cgroup.get("events") or {}).get("oom_kill")
        baseline = self._cgroup_oom_baseline
        cgroup_path = cgroup.get("path")
        cgroup_baseline_status = "unknown"
        cgroup_delta = None
        if isinstance(cgroup_total, int) and isinstance(cgroup_path, str):
            if baseline and cgroup_path == baseline[0] and cgroup_total >= baseline[1]:
                cgroup_delta = cgroup_total - baseline[1]
                cgroup_baseline_status = "matched"
            else:
                cgroup_baseline_status = "reset" if baseline else "initialized"
                self._cgroup_oom_baseline = (cgroup_path, cgroup_total)
                cgroup_delta = 0
        return {**earlyoom, "host_oom_kill_total": host_total,
                "host_oom_kills_since_start": host_delta,
                "cgroup_oom_kill_total": cgroup_total,
                "cgroup_oom_kills_since_start": cgroup_delta,
                "cgroup_oom_baseline_status": cgroup_baseline_status,
                "earlyoom_status": earlyoom.get("status"),
                "coverage": {
                    "host_oom_delta_known": host_delta is not None,
                    "cgroup_oom_delta_known": cgroup_delta is not None,
                    "earlyoom_known": earlyoom.get("status") == "ok"
                    and earlyoom.get("coverage_complete") is True
                    and earlyoom.get("earlyoom_kills") is not None,
                }}

    def tick(
        self, scheduled_ns: int, previous_ns: int | None, missed_deadlines: int,
    ) -> dict[str, Any]:
        started = MONOTONIC_NS()
        # Safety-critical host counters are first and require no commands or tree scans.
        host = meminfo()
        memory_pressure = pressure()
        container, container_meta = self._get(
            "container", {"pid": None, "state": None, "root_start_ticks": None},
            4_000_000_000,
        )
        bundle, cache_meta = self._get(
            "cache_bundle",
            {"cache": copy.deepcopy(UNKNOWN_CACHE),
             "process": copy.deepcopy(UNKNOWN_PROCESS)},
            2_000_000_000,
        )
        cache_value, process = bundle["cache"], bundle["process"]
        cache_value["sample_fresh"] = cache_meta["status"] == "current"
        state, state_evidence = self._validated_container_state(container, container_meta)
        evidence, evidence_meta = self._get(
            "evidence", {"logs": {"cuda": gpu(), "cuda_ooms": None,
                                    "store_failures": None, "restore_fallbacks": None},
                         "earlyoom": {"status": "unknown", "earlyoom_kills": None}},
            10_000_000_000,
        )
        oom = self._oom(process, evidence["earlyoom"])
        oom.update({"cuda_ooms_log_window": evidence["logs"].get("cuda_ooms"),
                    "cuda_oom_seen_since_start": evidence["logs"].get(
                        "cuda_oom_seen_since_start"),
                    "cuda_allocator_retries": _allocator_retries(evidence["logs"])})
        oom["coverage"]["cuda_oom_known"] = (
            evidence["logs"].get("cuda_oom_seen_since_start") is not None)
        now = MONOTONIC_NS()
        value: dict[str, Any] = {
            "schema": TELEMETRY_SCHEMA, "time_ns": TIME_NS(), "monotonic_ns": now,
            "rank": self.args.rank, "container_state": state,
            "container_state_evidence": state_evidence, "container_pid": container.get("pid"),
            **host, "memory_pressure": memory_pressure, "process": process,
            "connections": tcp_connections(self.args.api_port), "cache": cache_value,
            "cuda": evidence["logs"]["cuda"],
            "operations": {**cache_value.get("operations", {}),
                           "store_failures_log_window": evidence["logs"].get("store_failures"),
                           "restore_fallbacks_log_window": evidence["logs"].get("restore_fallbacks")},
            "oom": oom,
            "cadence": {"target_interval_ns": self.interval_ns,
                        "scheduled_monotonic_ns": scheduled_ns,
                        "actual_interval_ns": None if previous_ns is None else now - previous_ns,
                        "lateness_ns": max(0, now - scheduled_ns),
                        "deadlines_missed_since_previous": missed_deadlines},
            "sources": {
                "host": {"status": "current" if host["mem_available_bytes"] is not None
                          and memory_pressure is not None else "unknown", "age_ns": 0,
                         "sampled_monotonic_ns": now},
                "container": {**container_meta, "evidence_status": state_evidence},
                "cache": {**cache_meta, "evidence_status": cache_value.get("accounting_status")},
                "process": {**cache_meta, "evidence_status": process.get("identity_status")},
                "logs_and_oom": evidence_meta,
            },
        }
        if self.args.rank == 0:
            endpoint, endpoint_meta = self._get(
                "endpoint", {"api": {"requests_running": None, "requests_waiting": None,
                                      "gpu_cache_usage_perc": None},
                             "health": {"healthy": False, "http_status": None,
                                        "status": "unknown", "error": None}},
                3_000_000_000,
            )
            admission_api, admission_source = self._admission_sample(
                evidence["logs"].get("admission", {}), evidence_meta,
                container, container_meta)
            value["api"] = {**endpoint["api"], **admission_api}
            value["health"] = endpoint["health"]
            value["sources"]["endpoint"] = endpoint_meta
            value["sources"]["admission"] = admission_source
        value["cadence"]["sample_duration_ns"] = MONOTONIC_NS() - started
        return value


def advance_deadline(deadline_ns: int, now_ns: int, interval_ns: int) -> tuple[int, int]:
    if now_ns <= deadline_ns:
        return deadline_ns, 0
    missed = math.ceil((now_ns - deadline_ns) / interval_ns)
    return deadline_ns + missed * interval_ns, missed


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--container", required=True)
    result.add_argument("--cache-root", required=True)
    result.add_argument("--container-cache-root")
    result.add_argument("--rank", type=int, choices=range(4), required=True)
    result.add_argument("--api-port", type=int, default=8000)
    result.add_argument("--interval", type=float, default=1.0)
    result.add_argument("--once", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not 0.1 <= args.interval <= 60:
        print("probe: interval must be 0.1..60 seconds", file=sys.stderr)
        return 2
    if args.container_cache_root is None:
        args.container_cache_root = args.cache_root
    if args.once:
        print(json.dumps(sample(args), sort_keys=True), flush=True)
        return 0
    probe = PersistentProbe(args)
    interval_ns = int(args.interval * 1_000_000_000)
    scheduled, previous, missed = MONOTONIC_NS(), None, 0
    probe.start()
    try:
        while True:
            value = probe.tick(scheduled, previous, missed)
            previous = value["monotonic_ns"]
            print(json.dumps(value, sort_keys=True), flush=True)
            scheduled += interval_ns
            scheduled, missed = advance_deadline(scheduled, MONOTONIC_NS(), interval_ns)
            probe._stop.wait(max(0, scheduled - MONOTONIC_NS()) / 1_000_000_000)
    except KeyboardInterrupt:
        return 0
    finally:
        probe.close()


if __name__ == "__main__":
    raise SystemExit(main())
