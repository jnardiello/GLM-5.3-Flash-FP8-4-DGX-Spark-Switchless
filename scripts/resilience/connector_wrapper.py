#!/usr/bin/env python3
"""Test-only wrapper around a hash-pinned bounded SparkCache connector."""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import uuid


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load resilience component {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_runtime_path = Path(os.environ["TP4_RESILIENCE_RUNTIME"])
_base_path = Path(os.environ["TP4_RESILIENCE_BASE_CONNECTOR"])
_expected = os.environ["TP4_RESILIENCE_BASE_SHA256"]
if hashlib.sha256(_base_path.read_bytes()).hexdigest() != _expected:
    raise RuntimeError("resilience base connector hash differs")

_runtime = _load(_runtime_path, "_tp4_resilience_runtime")
_runtime.install()
_base = _load(_base_path, "_tp4_resilience_base_connector")

for _name, _value in vars(_base).items():
    if not _name.startswith("__"):
        globals()[_name] = _value


class _ObservedMemoryBudget(_base.MemoryBudget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._resilience_owner_id = f"{id(self):x}"
        _runtime.write_reservation(self._resilience_owner_id, 0, action="init", peak_bytes=0)

    def try_reserve(self, peak_bytes):
        _base._require_positive_int("peak_bytes", peak_bytes)
        with self._lock:
            previous_reserved = self._reserved_bytes
            try:
                available = self._available_reader()
            except Exception:
                available = None
            new_reserved = self._reserved_bytes + peak_bytes
            admitted = (not isinstance(available, bool) and isinstance(available, int)
                        and available >= 0 and new_reserved <= self._max_bytes
                        and available - new_reserved >= self._min_available_bytes)
            reservation = _base.MemoryReservation(self, peak_bytes) if admitted else None
            if admitted:
                self._reserved_bytes = new_reserved
            try:
                _runtime.write_reservation(
                    self._resilience_owner_id, self._reserved_bytes,
                    action="admit" if admitted else "reject", peak_bytes=peak_bytes)
            except BaseException:
                if reservation is not None:
                    self._reserved_bytes = previous_reserved
                    reservation._released = True
                raise
            return reservation

    def _release(self, reservation):
        with self._lock:
            if reservation._released:
                return
            if reservation._budget is not self:
                raise ValueError("reservation belongs to a different budget")
            peak = reservation._peak_bytes
            self._reserved_bytes -= peak
            reservation._released = True
            try:
                _runtime.write_reservation(self._resilience_owner_id, self._reserved_bytes,
                                           action="release", peak_bytes=peak)
            except BaseException:
                # Accounting is already released. Leave the last durable record
                # conservatively nonzero rather than killing the saver thread.
                pass


_base.MemoryBudget = _ObservedMemoryBudget


def _best_effort_emit(kind, **fields):
    """Keep test instrumentation failures out of the serving/cache paths."""
    try:
        _runtime.emit_event(kind, **fields)
        return None
    except BaseException as error:  # noqa: BLE001 - instrumentation is never authoritative
        return f"instrumentation:{type(error).__name__}"


def _best_effort_begin(stage, operation_id, **fields):
    case_id = None
    instrumentation_error = None
    try:
        case_id = _runtime.current_case_id()
    except BaseException as error:  # noqa: BLE001 - continue the real cache operation
        instrumentation_error = f"instrumentation:{type(error).__name__}"
    try:
        observed = _runtime.begin_stage(
            stage, operation_id, case_id=case_id, **fields)
        if isinstance(observed, str):
            case_id = observed
    except BaseException as error:  # noqa: BLE001 - continue the real cache operation
        instrumentation_error = f"instrumentation:{type(error).__name__}"
    return case_id, instrumentation_error


class SparkContextCacheConnector(_base.SparkContextCacheConnector):
    @staticmethod
    def _resilience_digest(value):
        for candidate in (value, getattr(value, "plan", None)):
            digest = getattr(candidate, "digest", None)
            if isinstance(digest, str):
                return digest
        return None

    def _capture_stream_snapshot(self, *args, **kwargs):
        operation_id = uuid.uuid4().hex
        digest = self._resilience_digest(args[0] if args else kwargs.get("plan"))
        case_id, instrumentation_error = _best_effort_begin(
            "capture", operation_id, digest=digest)
        try:
            result = super()._capture_stream_snapshot(*args, **kwargs)
        except BaseException as error:
            _best_effort_emit("stage_end", stage="capture", operation_id=operation_id,
                              digest=digest, case_id=case_id, outcome="error",
                              error=type(error).__name__,
                              **({"instrumentation_error": instrumentation_error}
                                 if instrumentation_error else {}))
            raise
        _best_effort_emit(
            "stage_end", stage="capture", operation_id=operation_id, digest=digest,
            case_id=case_id, outcome="ok",
            **({"instrumentation_error": instrumentation_error}
               if instrumentation_error else {}),
        )
        return result

    def _commit_store_snapshot(self, *args, **kwargs):
        operation_id = uuid.uuid4().hex
        snapshot = args[0] if args else kwargs.get("snapshot")
        digest = self._resilience_digest(snapshot)
        context = {
            "digest": digest,
            "operation_id": operation_id,
            "case_id": None,
            "instrumentation_error": None,
        }
        self._resilience_publication_context = context
        try:
            (context["case_id"], context["instrumentation_error"]) = (
                _best_effort_begin("publication", operation_id, digest=digest))
            return super()._commit_store_snapshot(*args, **kwargs)
        finally:
            if getattr(self, "_resilience_publication_context", None) is context:
                del self._resilience_publication_context

    def _finish_store(self, digest, *, committed, evicted=False,
                      additional_digests=(), error=None):
        # The base commit catches publication failures, so this callback is the
        # authoritative outcome. Forward its complete signature before telemetry.
        super()._finish_store(
            digest, committed=committed, evicted=evicted,
            additional_digests=additional_digests, error=error,
        )
        context = getattr(self, "_resilience_publication_context", None)
        if not isinstance(context, dict) or context.get("digest") != digest:
            return
        instrumentation_error = context.get("instrumentation_error")
        error_name = type(error).__name__ if error is not None else None
        if not committed and error_name is None:
            error_name = "evicted" if evicted else "not_committed"
        outcome = "ok" if committed and error is None else "error"
        _best_effort_emit(
            "stage_end", stage="publication", operation_id=context["operation_id"],
            digest=digest, case_id=context.get("case_id"), outcome=outcome,
            committed=bool(committed),
            evicted=bool(evicted), additional_digest_count=len(additional_digests),
            **({"error": error_name} if error_name else {}),
            **({"instrumentation_error": instrumentation_error}
               if instrumentation_error else {}),
        )

    def _restore_stream_snapshot(self, *args, **kwargs):
        operation_id = uuid.uuid4().hex
        plan = args[1] if len(args) > 1 else kwargs.get("plan")
        digest = self._resilience_digest(plan)
        case_id, instrumentation_error = _best_effort_begin(
            "restore", operation_id, digest=digest)
        try:
            result = super()._restore_stream_snapshot(*args, **kwargs)
        except BaseException as error:
            _best_effort_emit("stage_end", stage="restore", operation_id=operation_id,
                              digest=digest, case_id=case_id, outcome="error",
                              error=type(error).__name__,
                              **({"instrumentation_error": instrumentation_error}
                                 if instrumentation_error else {}))
            raise
        _best_effort_emit(
            "stage_end", stage="restore", operation_id=operation_id, digest=digest,
            case_id=case_id, outcome="ok",
            **({"instrumentation_error": instrumentation_error}
               if instrumentation_error else {}),
        )
        return result

    def shutdown(self):
        try:
            return super().shutdown()
        finally:
            reserved = getattr(getattr(self, "_cpu_budget", None), "reserved_bytes", None)
            _best_effort_emit("connector_shutdown", reserved_bytes=reserved)
