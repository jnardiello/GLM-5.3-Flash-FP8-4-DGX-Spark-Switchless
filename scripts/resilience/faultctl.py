#!/usr/bin/env python3
"""Initialize and manipulate only an exclusive TP4 resilience cache namespace."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
from typing import Any

from common import (
    MAX_NAMESPACE_BYTES,
    SCHEMA,
    ContractError,
    atomic_json,
    canonical_json,
    namespace_name,
    tree_bytes,
    sha256_file,
    validate_campaign_id,
)


MODES = {"off", "enospc", "eio", "delay", "publication_eio"}
MUTATIONS = {"missing", "truncated", "checksum"}
_CONTROL_ALLOWANCE = 64 << 20


def checked_root(raw: str, campaign_id: str, *, initialized: bool = True) -> Path:
    supplied = Path(raw).expanduser().absolute()
    if supplied.name != namespace_name(campaign_id):
        raise ContractError(
            f"cache root basename must be {namespace_name(campaign_id)}"
        )
    if supplied.is_symlink():
        raise ContractError("cache root must not be a symlink")
    root = supplied.resolve(strict=False)
    if initialized:
        marker = root / ".tp4-resilience" / "marker.json"
        expected = {"campaign_id": campaign_id, "schema": SCHEMA}
        try:
            actual = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ContractError("cache root lacks a valid campaign marker") from error
        if actual != expected:
            raise ContractError("cache root marker differs")
    return root


def initialize(root: Path, campaign_id: str) -> dict[str, Any]:
    if root.exists():
        if root.is_symlink() or not root.is_dir():
            raise ContractError("existing cache root is not a plain directory")
        entries = list(root.iterdir())
        if entries:
            # Re-entry is accepted only for the same valid marker. No existing
            # pre-campaign cache can be adopted accidentally.
            return status(checked_root(str(root), campaign_id))
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(root, 0o700)
    control = root / ".tp4-resilience"
    control.mkdir(mode=0o700)
    atomic_json(control / "marker.json", {"campaign_id": campaign_id, "schema": SCHEMA})
    set_control(root, campaign_id, mode="off", operation="any", case_id="init",
                remaining=0, after=0, delay_ms=0)
    return status(root)


def set_control(
    root: Path,
    campaign_id: str,
    *,
    mode: str,
    operation: str,
    case_id: str,
    remaining: int,
    after: int,
    delay_ms: int,
    pause_stage: str | None = None,
) -> dict[str, Any]:
    if mode not in MODES:
        raise ContractError(f"unsupported fault mode: {mode}")
    if operation not in {"any", "read", "write"}:
        raise ContractError("operation must be any, read, or write")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
           for value in (remaining, after, delay_ms)):
        raise ContractError("remaining, after, and delay_ms must be non-negative integers")
    if delay_ms > 60_000:
        raise ContractError("delay_ms must not exceed 60000")
    value = {
        "schema": SCHEMA,
        "campaign_id": campaign_id,
        "case_id": case_id,
        "mode": mode,
        "operation": operation,
        "remaining": remaining,
        "after": after,
        "delay_ms": delay_ms,
        "pause_stage": pause_stage,
        "generation": time.time_ns(),
    }
    atomic_json(root / ".tp4-resilience" / "control.json", value)
    return value


def status(root: Path) -> dict[str, Any]:
    used = tree_bytes(root)
    manifests = sum(path.is_file() and not path.is_symlink()
                    for path in (root / "manifests").glob("**/*.json"))
    chunks = sum(path.is_file() and not path.is_symlink()
                 for path in (root / "chunks").glob("**/*"))
    staging = sum(path.is_file() and not path.is_symlink()
                  for path in root.glob("**/*") if ".writing" in path.name)
    control_path = root / ".tp4-resilience" / "control.json"
    try:
        control = json.loads(control_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        control = None
    return {
        "schema": SCHEMA,
        "root": str(root),
        "used_bytes": used,
        "cap_bytes": MAX_NAMESPACE_BYTES,
        "within_cap": used <= MAX_NAMESPACE_BYTES,
        "manifest_count": manifests,
        "chunk_count": chunks,
        "staging_count": staging,
        "control": control,
    }


def _relative(root: Path, path: Path) -> str:
    if path.is_symlink():
        raise ContractError("artifact must not be a symlink")
    resolved = path.resolve(strict=False)
    if os.path.commonpath((str(root), str(resolved))) != str(root):
        raise ContractError("artifact escaped the campaign namespace")
    return str(resolved.relative_to(root))


def _candidates(root: Path, kind: str, digest: str | None = None) -> list[Path]:
    if kind == "chunk":
        if digest:
            manifests = list((root / "manifests").glob(f"**/{digest}.json"))
            if len(manifests) != 1:
                raise ContractError(f"expected one manifest for digest {digest}, found {len(manifests)}")
            try:
                manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise ContractError("selected manifest is unreadable") from error
            names: set[str] = set()
            def collect(value: Any) -> None:
                if isinstance(value, dict):
                    for key, item in value.items():
                        if key in {"sha256", "digest", "object_digest", "chunk_digest"} and isinstance(item, str):
                            names.add(item)
                        collect(item)
                elif isinstance(value, list):
                    for item in value:
                        collect(item)
            collect(manifest)
            paths = [path for name in names for path in (root / "chunks").glob(f"**/{name}*")]
        else:
            paths = list((root / "chunks").glob("**/*"))
    elif kind == "manifest":
        paths = list((root / "manifests").glob(f"**/{digest}.json" if digest else "**/*.json"))
    else:
        raise ContractError("kind must be chunk or manifest")
    return sorted(path for path in paths if path.is_file() and not path.is_symlink())


def mutate(root: Path, campaign_id: str, mutation: str, kind: str, index: int,
           digest: str | None = None) -> dict[str, Any]:
    if mutation not in MUTATIONS:
        raise ContractError(f"unsupported mutation: {mutation}")
    if digest is not None and (len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest)):
        raise ContractError("digest must be 64 lowercase hexadecimal characters")
    candidates = _candidates(root, kind, digest)
    if not candidates:
        raise ContractError(f"no {kind} artifact is available")
    if index < 0 or index >= len(candidates):
        raise ContractError(f"artifact index {index} is outside 0..{len(candidates)-1}")
    target = candidates[index]
    relative = _relative(root, target)
    journal_dir = root / ".tp4-resilience" / "mutations"
    journal_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    mutation_id = hashlib.sha256(f"{campaign_id}:{time.time_ns()}:{relative}".encode()).hexdigest()[:20]
    backup = journal_dir / f"{mutation_id}.backup"
    before = sha256_file(target)
    quota_path = root / ".tp4-resilience" / "quota.lock"
    quota_fd = os.open(quota_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(quota_fd, fcntl.LOCK_EX)
        if mutation == "missing":
            os.replace(target, backup)
        else:
            size = target.stat().st_size
            state_path = root / ".tp4-resilience" / "quota-state.json"
            state_used = 0
            try:
                state_used = int(json.loads(state_path.read_text(encoding="utf-8"))["used_bytes"])
            except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                pass
            used = max(tree_bytes(root), state_used)
            allowance = min(_CONTROL_ALLOWANCE, MAX_NAMESPACE_BYTES // 4)
            if used + size > MAX_NAMESPACE_BYTES - allowance:
                raise ContractError("mutation backup would exceed the namespace cap")
            shutil.copy2(target, backup)
            if mutation == "truncated":
                if size < 2:
                    raise ContractError("selected artifact is too small to truncate")
                with target.open("r+b") as stream:
                    stream.truncate(max(1, size // 2))
            else:
                with target.open("r+b") as stream:
                    first = stream.read(1)
                    if not first:
                        raise ContractError("selected artifact is empty")
                    stream.seek(0)
                    stream.write(bytes((first[0] ^ 1,)))
                    stream.flush()
                    os.fsync(stream.fileno())
    finally:
        fcntl.flock(quota_fd, fcntl.LOCK_UN)
        os.close(quota_fd)
    record = {
        "schema": SCHEMA,
        "campaign_id": campaign_id,
        "mutation_id": mutation_id,
        "mutation": mutation,
        "kind": kind,
        "context_digest": digest,
        "relative_path": relative,
        "backup": backup.name,
        "before_sha256": before,
        "time_ns": time.time_ns(),
        "restored": False,
    }
    atomic_json(journal_dir / f"{mutation_id}.json", record)
    return record


def restore_mutations(root: Path) -> dict[str, Any]:
    journal_dir = root / ".tp4-resilience" / "mutations"
    restored: list[str] = []
    if not journal_dir.exists():
        return {"restored": restored}
    for journal in sorted(journal_dir.glob("*.json")):
        record = json.loads(journal.read_text(encoding="utf-8"))
        if record.get("restored"):
            continue
        target = root / record["relative_path"]
        backup = journal_dir / record["backup"]
        _relative(root, target)
        _relative(root, backup)
        if not backup.is_file() or backup.is_symlink():
            raise ContractError(f"mutation backup is missing: {record['mutation_id']}")
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.replace(backup, target)
        if sha256_file(target) != record["before_sha256"]:
            raise ContractError(f"restored artifact hash differs: {record['mutation_id']}")
        record["restored"] = True
        record["restored_time_ns"] = time.time_ns()
        atomic_json(journal, record)
        restored.append(record["mutation_id"])
    return {"restored": restored}


def _event_paths(root: Path) -> list[Path]:
    directory = root / ".tp4-resilience"
    return [directory / "events.previous.jsonl", directory / "events.jsonl"]


def event_cursor(root: Path) -> dict[str, int]:
    path = root / ".tp4-resilience" / "events.jsonl"
    try:
        metadata = path.stat()
        return {"device": metadata.st_dev, "inode": metadata.st_ino, "position": metadata.st_size}
    except FileNotFoundError:
        return {"device": 0, "inode": 0, "position": 0}


def find_event(root: Path, cursor: dict[str, int], *, case_id: str, kind: str,
               stage: str | None = None, mode: str | None = None,
               timeout: float = 0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while True:
        paths: list[tuple[Path, os.stat_result]] = []
        for path in _event_paths(root):
            try:
                paths.append((path, path.stat()))
            except FileNotFoundError:
                continue
        if cursor["inode"]:
            origin = [(path, metadata) for path, metadata in paths
                      if (metadata.st_dev, metadata.st_ino) ==
                      (cursor["device"], cursor["inode"])]
            scans = [(path, cursor["position"]) for path, _ in origin]
            # One rotation renames the cursor inode to events.previous.jsonl;
            # continue from its old offset, then hand off to the new current file.
            if any(path.name == "events.previous.jsonl" for path, _ in origin):
                scans.extend((path, 0) for path, metadata in paths
                             if path.name == "events.jsonl" and
                             (metadata.st_dev, metadata.st_ino) !=
                             (cursor["device"], cursor["inode"]))
        else:
            scans = [(path, 0) for path, _ in paths]
        for path, position in scans:
            try:
                with path.open("r", encoding="utf-8") as stream:
                    stream.seek(position)
                    for line in iter(stream.readline, ""):
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if (event.get("kind") == kind and event.get("case_id") == case_id
                                and (stage is None or event.get("stage") == stage)
                                and (mode is None or event.get("mode") == mode)):
                            return event
            except FileNotFoundError:
                continue
        if time.monotonic() >= deadline:
            break
        time.sleep(0.05)
    raise TimeoutError(f"event {kind} was not observed within {timeout}s")


def wait_stage(root: Path, case_id: str, stage: str, timeout: float) -> dict[str, Any]:
    return find_event(root, event_cursor(root), case_id=case_id, kind="stage_waiting",
                      stage=stage, timeout=timeout)


def _host_pid(inner_pid: int, start_ticks: int) -> int:
    matches: list[int] = []
    for status_path in Path("/proc").glob("[0-9]*/status"):
        try:
            fields = dict(line.split(":", 1) for line in status_path.read_text().splitlines() if ":" in line)
            nspid = [int(item) for item in fields.get("NSpid", "").split()]
            stat = (status_path.parent / "stat").read_text(encoding="ascii")
            actual_start = int(stat[stat.rfind(")") + 2:].split()[19])
            if nspid and nspid[-1] == inner_pid and actual_start == start_ticks:
                matches.append(int(status_path.parent.name))
        except (OSError, ValueError, IndexError):
            continue
    if len(matches) != 1:
        raise ContractError(f"expected one live process for stage identity, found {len(matches)}")
    return matches[0]


def signal_stage_process(root: Path, case_id: str, stage: str, inner_pid: int,
                         start_ticks: int, signal_name: str) -> dict[str, Any]:
    allowed = {"TERM": signal.SIGTERM, "KILL": signal.SIGKILL,
               "STOP": signal.SIGSTOP, "CONT": signal.SIGCONT}
    if signal_name not in allowed:
        raise ContractError("unsupported worker signal")
    matched = False
    for path in _event_paths(root):
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                event = json.loads(line)
                if (event.get("kind") == "stage_waiting" and event.get("case_id") == case_id
                        and event.get("stage") == stage and event.get("pid") == inner_pid
                        and event.get("process_start_ticks") == start_ticks):
                    matched = True
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
    if not matched:
        raise ContractError("signal target lacks matching fresh stage evidence")
    host_pid = _host_pid(inner_pid, start_ticks)
    os.kill(host_pid, allowed[signal_name])
    return {"host_pid": host_pid, "container_pid": inner_pid,
            "process_start_ticks": start_ticks, "signal": signal_name}


def prune_reservations(root: Path) -> dict[str, Any]:
    directory = root / ".tp4-resilience" / "reservations"
    removed: list[str] = []
    retained: list[str] = []
    if not directory.is_dir():
        return {"removed": removed, "retained": retained}
    for path in directory.glob("*.json"):
        live = False
        if not path.is_symlink():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                _host_pid(int(value["pid"]), int(value["process_start_ticks"]))
                live = True
            except (OSError, UnicodeError, json.JSONDecodeError, KeyError,
                    TypeError, ValueError, ContractError):
                pass
        if live:
            retained.append(path.name)
        else:
            path.unlink(missing_ok=True)
            removed.append(path.name)
    return {"removed": removed, "retained": retained}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--root", required=True)
    result.add_argument("--campaign-id", required=True)
    sub = result.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    sub.add_parser("status")
    sub.add_parser("barrier")
    control = sub.add_parser("set")
    control.add_argument("--mode", choices=sorted(MODES), required=True)
    control.add_argument("--operation", choices=("any", "read", "write"), default="any")
    control.add_argument("--case-id", required=True)
    control.add_argument("--remaining", type=int, default=1)
    control.add_argument("--after", type=int, default=0)
    control.add_argument("--delay-ms", type=int, default=0)
    control.add_argument("--pause-stage", choices=("capture", "publication", "restore"))
    mutation = sub.add_parser("mutate")
    mutation.add_argument("--mutation", choices=sorted(MUTATIONS), required=True)
    mutation.add_argument("--kind", choices=("chunk", "manifest"), required=True)
    mutation.add_argument("--index", type=int, default=0)
    mutation.add_argument("--digest")
    sub.add_parser("restore")
    sub.add_parser("cursor")
    sub.add_parser("prune-reservations")
    event = sub.add_parser("event")
    event.add_argument("--cursor", required=True, help="device:inode:position from cursor")
    event.add_argument("--case-id", required=True)
    event.add_argument("--kind", choices=("fault_hit", "stage_begin", "stage_end", "stage_waiting"), required=True)
    event.add_argument("--stage", choices=("capture", "publication", "restore"))
    event.add_argument("--mode", choices=sorted(MODES - {"off"}))
    event.add_argument("--timeout", type=float, default=0)
    signal_parser = sub.add_parser("signal-stage")
    signal_parser.add_argument("--case-id", required=True)
    signal_parser.add_argument("--stage", choices=("capture", "publication", "restore"), required=True)
    signal_parser.add_argument("--pid", type=int, required=True)
    signal_parser.add_argument("--start-ticks", type=int, required=True)
    signal_parser.add_argument("--signal", choices=("TERM", "KILL", "STOP", "CONT"), required=True)
    wait = sub.add_parser("wait-stage")
    wait.add_argument("--case-id", required=True)
    wait.add_argument("--stage", choices=("capture", "publication", "restore"), required=True)
    wait.add_argument("--timeout", type=float, default=60)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        campaign_id = validate_campaign_id(args.campaign_id)
        root = checked_root(args.root, campaign_id, initialized=args.command != "init")
        if args.command == "init":
            result = initialize(root, campaign_id)
        elif args.command == "status":
            result = status(root)
        elif args.command == "barrier":
            # Compared only with probe samples from this same host.  A host
            # monotonic timestamp avoids wall-clock skew between ranks and
            # proves that the cache scan began after the case completed.
            result = {"monotonic_ns": time.monotonic_ns()}
        elif args.command == "set":
            result = set_control(root, campaign_id, mode=args.mode,
                                 operation=args.operation, case_id=args.case_id,
                                 remaining=args.remaining, after=args.after,
                                 delay_ms=args.delay_ms, pause_stage=args.pause_stage)
        elif args.command == "mutate":
            result = mutate(root, campaign_id, args.mutation, args.kind, args.index, args.digest)
        elif args.command == "restore":
            result = restore_mutations(root)
        elif args.command == "cursor":
            result = event_cursor(root)
        elif args.command == "prune-reservations":
            result = prune_reservations(root)
        elif args.command == "event":
            try:
                device, inode, position = (int(item) for item in args.cursor.split(":"))
            except (TypeError, ValueError) as error:
                raise ContractError("cursor must be device:inode:position") from error
            result = find_event(root, {"device": device, "inode": inode, "position": position},
                                case_id=args.case_id, kind=args.kind, stage=args.stage,
                                mode=args.mode, timeout=args.timeout)
        elif args.command == "signal-stage":
            result = signal_stage_process(root, args.case_id, args.stage, args.pid,
                                          args.start_ticks, args.signal)
        else:
            result = wait_stage(root, args.case_id, args.stage, args.timeout)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (ContractError, OSError, TimeoutError, ValueError) as error:
        print(f"faultctl: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
