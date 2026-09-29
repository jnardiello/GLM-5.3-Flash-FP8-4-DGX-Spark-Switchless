#!/usr/bin/env python3
"""Test-overlay fault and telemetry adapter, scoped to one marked cache root."""

from __future__ import annotations

import errno
import builtins
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from typing import Any


SCHEMA = "tp4-resilience/v1"
MAX_BYTES = 8 << 30
_LOCK = threading.RLock()
_LOCAL = threading.local()
_INSTALLED = False
_ORIGINAL_WRITE = os.write
_ORIGINAL_PWRITE = getattr(os, "pwrite", None)
_ORIGINAL_WRITEV = getattr(os, "writev", None)
_ORIGINAL_FTRUNCATE = os.ftruncate
_ORIGINAL_TRUNCATE = os.truncate
_ORIGINAL_FDOPEN = os.fdopen
_ORIGINAL_IO_OPEN = io.open
_ORIGINAL_BUILTIN_OPEN = builtins.open
_CONTROL_ALLOWANCE = 64 << 20
_FAULT_GENERATION: int | None = None
_FAULT_AFTER = 0
_FAULT_REMAINING = 0


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _settings() -> tuple[Path, str, int]:
    campaign_id = os.environ.get("TP4_RESILIENCE_CAMPAIGN_ID", "")
    root = Path(os.environ.get("TP4_RESILIENCE_CACHE_ROOT", ""))
    cap_text = os.environ.get("TP4_RESILIENCE_MAX_BYTES", str(MAX_BYTES))
    test_root = os.environ.get("TP4_RESILIENCE_TEST_ROOT")
    expected = Path(f"/cache/jit/tp4-resilience-{campaign_id}")
    if test_root:
        expected = Path(test_root)
    if (
        not campaign_id
        or root != expected
        or root.name != f"tp4-resilience-{campaign_id}"
        or not cap_text.isdigit()
        or not 4096 <= int(cap_text) <= MAX_BYTES
        or (test_root is not None and int(cap_text) == MAX_BYTES)
    ):
        raise RuntimeError("invalid TP4 resilience namespace contract")
    return root, campaign_id, int(cap_text)


def _control_dir() -> Path:
    root, _, _ = _settings()
    return root / ".tp4-resilience"


def _within(path: str | os.PathLike[str]) -> bool:
    root, _, _ = _settings()
    try:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        return os.path.commonpath((str(root), str(candidate.resolve(strict=False)))) == str(root)
    except (OSError, TypeError, ValueError):
        return False


def _is_internal(path: str | os.PathLike[str]) -> bool:
    try:
        return os.path.commonpath((str(_control_dir()), str(Path(path).resolve(strict=False)))) == str(_control_dir())
    except (OSError, TypeError, ValueError):
        return False


def _raw_append(path: Path, value: dict[str, Any]) -> None:
    _LOCAL.busy = True
    try:
        encoded = _canonical(value) + b"\n"
        if path.exists() and path.stat().st_size >= 16 << 20:
            rotated = path.with_suffix(".previous.jsonl")
            os.replace(path, rotated)
        descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            _ORIGINAL_WRITE(descriptor, encoded)
        finally:
            os.close(descriptor)
    finally:
        _LOCAL.busy = False


def emit_event(kind: str, **fields: Any) -> None:
    root, campaign_id, _ = _settings()
    directory = root / ".tp4-resilience"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    _LOCAL.busy = True
    try:
        case_id = _load_control().get("case_id")
    finally:
        _LOCAL.busy = False
    event = {
        "schema": SCHEMA,
        "campaign_id": campaign_id,
        "kind": kind,
        "pid": os.getpid(),
        "process_start_ticks": _process_start_ticks(),
        "thread": threading.current_thread().name,
        "time_ns": time.time_ns(),
        "case_id": case_id,
        **fields,
    }
    _raw_append(directory / "events.jsonl", event)


def current_case_id() -> str | None:
    """Return the case that owns a newly starting instrumented operation."""
    _LOCAL.busy = True
    try:
        value = _load_control().get("case_id")
    finally:
        _LOCAL.busy = False
    return value if isinstance(value, str) else None


def begin_stage(stage: str, operation_id: str, **fields: Any) -> str | None:
    """Emit a phase edge and honor a bounded, test-only synchronization pause."""
    _LOCAL.busy = True
    try:
        control = _load_control()
    finally:
        _LOCAL.busy = False
    case_id = fields.get("case_id")
    if not isinstance(case_id, str):
        case_id = control.get("case_id")
    fields["case_id"] = case_id if isinstance(case_id, str) else None
    emit_event("stage_begin", stage=stage, operation_id=operation_id, **fields)
    generation = control.get("generation")
    if control.get("pause_stage") != stage:
        return fields["case_id"]
    emit_event("stage_waiting", stage=stage, operation_id=operation_id, **fields)
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        time.sleep(0.02)
        _LOCAL.busy = True
        try:
            current = _load_control()
        finally:
            _LOCAL.busy = False
        if (current.get("generation") != generation
                or current.get("pause_stage") != stage):
            emit_event("stage_released", stage=stage, operation_id=operation_id, **fields)
            return fields["case_id"]
    emit_event("stage_pause_timeout", stage=stage, operation_id=operation_id, **fields)
    raise TimeoutError(f"campaign stage pause timed out: {stage}")


def _process_start_ticks() -> int:
    stat = Path("/proc/self/stat").read_text(encoding="ascii")
    end = stat.rfind(")")
    if end < 0:
        raise RuntimeError("cannot parse /proc/self/stat")
    return int(stat[end + 2:].split()[19])


def write_reservation(
    owner_id: str,
    reserved_bytes: int,
    *,
    action: str,
    peak_bytes: int,
) -> None:
    root, campaign_id, _ = _settings()
    directory = root / ".tp4-resilience"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    value = {
        "schema": SCHEMA,
        "campaign_id": campaign_id,
        "reserved_bytes": reserved_bytes,
        "action": action,
        "peak_bytes": peak_bytes,
        "pid": os.getpid(),
        "process_start_ticks": _process_start_ticks(),
        "owner_id": owner_id,
        "time_ns": time.time_ns(),
    }
    reservations = directory / "reservations"
    reservations.mkdir(mode=0o700, exist_ok=True)
    target = reservations / f"{os.getpid()}-{owner_id}.json"
    with _LOCK:
        _LOCAL.busy = True
        try:
            fd, name = tempfile.mkstemp(prefix=".reservation.", dir=reservations)
            temporary = Path(name)
            try:
                os.fchmod(fd, 0o600)
                with _ORIGINAL_FDOPEN(fd, "wb") as stream:
                    stream.write(_canonical(value) + b"\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, target)
            finally:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
        finally:
            _LOCAL.busy = False
        emit_event("reservation", **value)


def _load_control() -> dict[str, Any]:
    path = _control_dir() / "control.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {"mode": "off"}
    return value if isinstance(value, dict) else {"mode": "off"}


def _matches(path: str | os.PathLike[str], write: bool, control: dict[str, Any]) -> bool:
    if not _within(path) or _is_internal(path):
        return False
    operation = control.get("operation", "any")
    if operation == "write" and not write:
        return False
    if operation == "read" and write:
        return False
    if control.get("mode") == "publication_eio" and "/manifests/" not in str(path):
        return False
    # Scheduler-side manifest probes must not consume the worker restore fault.
    # Streaming page payload reads are always below the campaign chunks root.
    if (control.get("mode") in {"eio", "delay"} and not write
            and "/chunks/" not in str(path)):
        return False
    return True


def _fault(path: str | os.PathLike[str], write: bool) -> None:
    global _FAULT_GENERATION, _FAULT_AFTER, _FAULT_REMAINING
    if getattr(_LOCAL, "busy", False):
        return
    with _LOCK:
        # Reject unrelated/internal paths before reading the control file. This
        # keeps the audit hook from recursively auditing its own read.
        if not _within(path) or _is_internal(path):
            return
        _LOCAL.busy = True
        try:
            control = _load_control()
            if not _matches(path, write, control):
                return
            generation = control.get("generation")
            if generation != _FAULT_GENERATION:
                _FAULT_GENERATION = generation if isinstance(generation, int) else None
                _FAULT_AFTER = control.get("after", 0)
                _FAULT_REMAINING = control.get("remaining", 0)
            if (not isinstance(_FAULT_AFTER, int) or not isinstance(_FAULT_REMAINING, int)
                    or _FAULT_REMAINING <= 0):
                return
            if _FAULT_AFTER > 0:
                _FAULT_AFTER -= 1
                return
            _FAULT_REMAINING -= 1
            mode = control.get("mode", "off")
            delay_ms = control.get("delay_ms", 0)
            emit_event("fault_hit", case_id=control.get("case_id"), mode=mode,
                       operation="write" if write else "read", remaining=_FAULT_REMAINING,
                       path_sha256=hashlib.sha256(os.fsencode(path)).hexdigest())
            if mode == "delay":
                if isinstance(delay_ms, int) and 0 < delay_ms <= 60_000:
                    time.sleep(delay_ms / 1000)
                return
            if mode in {"eio", "publication_eio"}:
                raise OSError(errno.EIO, "injected campaign I/O error")
            if mode == "enospc":
                raise OSError(errno.ENOSPC, "injected campaign capacity error")
        finally:
            _LOCAL.busy = False


def _audit(event: str, args: tuple[Any, ...]) -> None:
    if event != "open" or not args or isinstance(args[0], int):
        return
    mode = args[1] if len(args) > 1 else "r"
    flags = args[2] if len(args) > 2 else 0
    write = bool(isinstance(mode, str) and any(char in mode for char in "wax+"))
    write = write or bool(isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
    try:
        if Path(args[0]).is_dir():
            return
    except (OSError, TypeError, ValueError):
        pass
    # Write faults are consumed only when bytes reach _guard_growth. In
    # particular, TemporaryFile may probe O_TMPFILE on the directory and
    # swallow that open failure before falling back to a named temporary.
    if write:
        return
    _fault(args[0], write)


def _fd_path(descriptor: int) -> Path | None:
    try:
        target = Path(os.readlink(f"/proc/self/fd/{descriptor}"))
    except OSError:
        try:  # F_GETPATH, used only by the macOS offline tests.
            raw = fcntl.fcntl(descriptor, 50, b"\0" * 1024)
            target = Path(raw.split(b"\0", 1)[0].decode())
        except (OSError, UnicodeError, ValueError):
            return None
    return target if _within(target) and not _is_internal(target) else None


def _tree_bytes(root: Path) -> int:
    total = 0
    for directory, _, files in os.walk(root, followlinks=False):
        for name in files:
            path = Path(directory, name)
            try:
                metadata = path.lstat()
            except FileNotFoundError:
                continue
            if not path.is_symlink():
                total += metadata.st_size
    return total


def _anonymous_bytes(root: Path) -> int:
    seen: set[tuple[int, int]] = set()
    total = 0
    for process in Path("/proc").glob("[0-9]*/fd"):
        try:
            descriptors = list(process.iterdir())
        except OSError:
            continue
        for descriptor in descriptors:
            try:
                target = os.readlink(descriptor)
                metadata = descriptor.stat()
            except OSError:
                continue
            if not target.endswith(" (deleted)"):
                continue
            unlinked = target[:-10]
            if os.path.commonpath((str(root), unlinked)) != str(root):
                continue
            identity = (metadata.st_dev, metadata.st_ino)
            if identity not in seen:
                seen.add(identity)
                total += metadata.st_size
    return total


def _quota_lock(root: Path) -> int:
    path = root / ".tp4-resilience" / "quota.lock"
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX)
    return descriptor


def _write_quota_state(root: Path, used_bytes: int) -> None:
    value = {"schema": SCHEMA, "used_bytes": used_bytes, "time_ns": time.time_ns()}
    directory = root / ".tp4-resilience"
    descriptor, name = tempfile.mkstemp(prefix=".quota-state.", dir=directory)
    temporary = Path(name)
    _LOCAL.busy = True
    try:
        os.fchmod(descriptor, 0o600)
        with _ORIGINAL_FDOPEN(descriptor, "wb") as stream:
            stream.write(_canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, directory / "quota-state.json")
    finally:
        _LOCAL.busy = False
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _refresh_quota_state(root: Path) -> int:
    used = _tree_bytes(root) + _anonymous_bytes(root)
    _write_quota_state(root, used)
    return used


def _unlock(token: tuple[int, Path] | int | None, *, refresh: bool = False) -> None:
    if token is not None:
        descriptor, root = token if isinstance(token, tuple) else (token, None)
        try:
            if refresh and root is not None:
                _refresh_quota_state(root)
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


def _guard_growth(
    descriptor: int, length: int, offset: int | None = None
) -> tuple[int, Path] | None:
    path = _fd_path(descriptor)
    if path is None:
        return None
    _fault(path, True)
    root, _, cap = _settings()
    lock = _quota_lock(root)
    try:
        metadata = os.fstat(descriptor)
        position = os.lseek(descriptor, 0, os.SEEK_CUR) if offset is None else offset
        growth = max(0, position + length - metadata.st_size)
        used = _tree_bytes(root) + _anonymous_bytes(root)
        allowance = min(_CONTROL_ALLOWANCE, cap // 4)
        if growth and used + growth > cap - allowance:
            emit_event("namespace_cap", used_bytes=used, requested_growth=growth,
                       cap_bytes=cap)
            raise OSError(errno.ENOSPC, "campaign namespace 8 GiB cap")
        return lock, root
    except BaseException:
        _unlock(lock, refresh=True)
        raise


def _write(descriptor: int, payload: Any) -> int:
    lock = _guard_growth(descriptor, len(payload))
    try:
        return _ORIGINAL_WRITE(descriptor, payload)
    finally:
        _unlock(lock, refresh=True)


def _pwrite(descriptor: int, payload: Any, offset: int) -> int:
    lock = _guard_growth(descriptor, len(payload), offset)
    try:
        assert _ORIGINAL_PWRITE is not None
        return _ORIGINAL_PWRITE(descriptor, payload, offset)
    finally:
        _unlock(lock, refresh=True)


def _writev(descriptor: int, buffers: Any) -> int:
    lock = _guard_growth(descriptor, sum(len(buffer) for buffer in buffers))
    try:
        assert _ORIGINAL_WRITEV is not None
        return _ORIGINAL_WRITEV(descriptor, buffers)
    finally:
        _unlock(lock, refresh=True)


def _truncate_fd(descriptor: int, length: int) -> None:
    current = os.fstat(descriptor).st_size
    lock = _guard_growth(descriptor, max(0, length - current), current)
    try:
        _ORIGINAL_FTRUNCATE(descriptor, length)
    finally:
        _unlock(lock, refresh=True)


def _truncate_path(path: Any, length: int) -> None:
    descriptor = os.open(path, os.O_WRONLY)
    try:
        _truncate_fd(descriptor, length)
    finally:
        os.close(descriptor)


class _QuotaFile:
    def __init__(self, stream: Any):
        self._stream = stream

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def __enter__(self):
        self._stream.__enter__()
        return self

    def __exit__(self, *args: Any) -> Any:
        return self._stream.__exit__(*args)

    def __iter__(self):
        return iter(self._stream)

    def write(self, payload: Any) -> int:
        length = len(payload.encode(self._stream.encoding or "utf-8")
                     if isinstance(payload, str) else payload)
        lock = _guard_growth(self._stream.fileno(), length)
        try:
            written = self._stream.write(payload)
            # The quota lock remains held until a buffered write reaches the
            # kernel, so another process cannot pass admission on stale sizes.
            self._stream.flush()
            return written
        finally:
            _unlock(lock, refresh=True)

    def writelines(self, lines: Any) -> None:
        for line in lines:
            self.write(line)

    def truncate(self, size: int | None = None) -> int:
        if size is None:
            size = self._stream.tell()
        current = os.fstat(self._stream.fileno()).st_size
        lock = _guard_growth(self._stream.fileno(), max(0, size - current), current)
        try:
            return self._stream.truncate(size)
        finally:
            _unlock(lock, refresh=True)


def _wrap_stream(stream: Any) -> Any:
    try:
        path = _fd_path(stream.fileno())
    except (AttributeError, OSError):
        return stream
    return _QuotaFile(stream) if path is not None else stream


def _io_open(*args: Any, **kwargs: Any) -> Any:
    return _wrap_stream(_ORIGINAL_IO_OPEN(*args, **kwargs))


def _builtin_open(*args: Any, **kwargs: Any) -> Any:
    return _wrap_stream(_ORIGINAL_BUILTIN_OPEN(*args, **kwargs))


def _fdopen(*args: Any, **kwargs: Any) -> Any:
    return _wrap_stream(_ORIGINAL_FDOPEN(*args, **kwargs))


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    root, campaign_id, cap = _settings()
    if root.resolve(strict=False) != root:
        raise RuntimeError("campaign namespace root must not contain symlinks")
    marker = root / ".tp4-resilience" / "marker.json"
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("campaign namespace is not initialized") from error
    if value != {"campaign_id": campaign_id, "schema": SCHEMA}:
        raise RuntimeError("campaign namespace marker differs")
    sys.addaudithook(_audit)
    os.write = _write
    if _ORIGINAL_PWRITE is not None:
        os.pwrite = _pwrite
    if _ORIGINAL_WRITEV is not None:
        os.writev = _writev
    os.ftruncate = _truncate_fd
    os.truncate = _truncate_path
    os.fdopen = _fdopen
    io.open = _io_open
    builtins.open = _builtin_open
    _INSTALLED = True
    emit_event("adapter_ready", max_bytes=cap)
