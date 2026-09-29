#!/usr/bin/env python3
"""Deadline-bounded production resilience campaign with private JSON receipts."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Callable
from urllib.parse import urlparse

from common import (
    MAX_NAMESPACE_BYTES,
    MIN_MEM_AVAILABLE_BYTES,
    SCHEMA,
    ContractError,
    append_event,
    atomic_json,
    canonical_json,
    load_json,
    namespace_name,
    require_private_dir,
    sha256_file,
    validate_campaign_id,
)
from combination_cases import run_cache_eio_cancel_restore


CONTEXT_TOKENS = (8_000, 32_000, 64_000, 128_000, 180_000, 262_128)
CONCURRENCIES = (1, 2, 4, 5, 6)
LONG_DECODE_CASE_ID = "context-245760-c5-long-decode-cold"
LONG_DECODE_PROMPT_TOKENS = 245_760
LONG_DECODE_COMPLETION_TOKENS = 16_384
LONG_DECODE_REQUEST_LIMIT_SECONDS = 3600
QUEUE_WAVES = (8, 16, 32, 64)
TARGET_REPEATS = 3
SSH = ("ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
       "-o", "ConnectTimeout=10")
TARGETED_STOP_MARGIN_SECONDS = 120
SOAK_PROTOCOL_VERSION = 2
TARGETED_CASE_PROTOCOL_VERSION = 2
SOAK_CONVERSATION_COUNT = 3
SOAK_CONTEXT_TARGETS = (512, 8_000, 32_000, 64_000)
SOAK_CANCEL_EVERY = 10
SOAK_IDLE_EVERY = 20
SOAK_IDLE_SECONDS = 30
SOAK_FINAL_PROOF_RESERVE_SECONDS = 60
SOAK_BENIGN_ABORT_MAX_SECONDS = 120
SOAK_MAX_HISTORY_MESSAGES = 32
GUARDED_SOCKET_GRACE_SECONDS = 5.0
WORKER_FAULT_MIN_REMAINING_SECONDS = 2700 + 120 + 10 + 300
FINAL_RESTORE_PROOF_RESERVE_SECONDS = 120 + 120 + 10 + 60
TEST_CACHE_MAX_BYTES = 3 << 30
TEST_CACHE_LOW_WATERMARK_BYTES = 2 << 30
RUNTIME_VARIANTS = {
    "default": "delta.env",
    "prefill-cache-trim": "delta-prefill-trim.env",
    "prefill-step-cap": "delta-prefill-step-cap.env",
}
PREFILL_TRIM_MANIFEST = "scripts/node/experiments/e03/prefill-cache-trim/manifest.json"
DEFAULT_WORKER_SOURCE = "scripts/node/overrides/vllm/v1/worker/gpu_worker.py"
TRIM_WORKER_SOURCE = "scripts/node/experiments/e03/prefill-cache-trim/gpu_worker.py"
WORKER_TARGET = "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py"
STEP_CAP_MANIFEST = "scripts/node/experiments/e03/prefill-step-cap/manifest.json"
DEFAULT_SCHEDULER_SOURCE = "scripts/node/experiments/e03/end-drain/scheduler.py"
STEP_CAP_SCHEDULER_SOURCE = "scripts/node/experiments/e03/prefill-step-cap/scheduler.py"
SCHEDULER_TARGET = "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py"
STEP_CAP_ENVIRONMENT = "VLLM_RESILIENCE_STEP_TOKEN_CAP=6912"
ADMISSION_MANIFEST = "scripts/node/experiments/e03/bounded-admission/manifest.json"
ADMISSION_SOURCE = "scripts/node/experiments/e03/bounded-admission/middleware.py"
ADMISSION_STAGED_NAME = "tp4_admission.py"
ADMISSION_TARGET = "/opt/tp4/tp4_admission.py"
ADMISSION_IMPORT = "tp4_admission.BoundedAdmissionMiddleware"
ADMISSION_ENVIRONMENT = {
    "TP4_ADMISSION_MAX_ACTIVE": "6",
    "TP4_ADMISSION_MAX_QUEUED": "128",
    "TP4_ADMISSION_MAX_BODY_BYTES": "8388608",
    "TP4_ADMISSION_QUEUE_TIMEOUT_SECONDS": "1800",
    "TP4_ADMISSION_REQUEST_TIMEOUT_SECONDS": "3600",
    "TP4_ADMISSION_BODY_IDLE_SECONDS": "30",
    "TP4_ADMISSION_SEND_IDLE_SECONDS": "30",
}
ADMISSION_GUARD_MODE = "all_post_except"
ADMISSION_BYPASS_PATHS = {
    "/is_scaling_elastic_ep", "/ping", "/scale_elastic_ep",
}
ADMISSION_BYPASS_PATTERN = "/v1/responses/{nonempty-single-segment}/cancel"
ADMISSION_MAX_QUEUED = 128
ADMISSION_MAX_ACTIVE = 6
ADMISSION_OVERFLOW_EXTRA = 16
ADMISSION_OVERLOAD_CODES = {"admission_queue_full", "admission_queue_timeout"}


def _append_private_event(path: Path, event: dict[str, Any]) -> None:
    """Append one complete local receipt, handling and rejecting short writes."""
    payload = dict(event)
    payload.setdefault("schema", SCHEMA)
    payload.setdefault("time_ns", time.time_ns())
    remaining = memoryview(canonical_json(payload) + b"\n")
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("receipt append made no forward progress")
            remaining = remaining[written:]
    finally:
        os.close(descriptor)


def parse_sse(payload: bytes) -> dict[str, Any]:
    """Return completion and error evidence from an OpenAI-compatible SSE body."""
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        return {"done": False, "error_event": False, "finish_reasons": [],
                "parse_errors": [f"UnicodeDecodeError: {error}"]}
    done = False
    error_event = False
    finish_reasons: list[str] = []
    parse_errors: list[str] = []
    for block in text.replace("\r\n", "\n").split("\n\n"):
        event_name = None
        data: list[str] = []
        for line in block.splitlines():
            if line.startswith("event:"):
                event_name = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].lstrip())
        if not data:
            continue
        encoded = "\n".join(data)
        if encoded.strip() == "[DONE]":
            done = True
            continue
        try:
            event = json.loads(encoded)
        except json.JSONDecodeError as error:
            parse_errors.append(f"JSONDecodeError: {error}")
            continue
        if event_name == "error" or isinstance(event, dict) and event.get("error") is not None:
            error_event = True
        if isinstance(event, dict):
            for choice in event.get("choices") or []:
                reason = choice.get("finish_reason") if isinstance(choice, dict) else None
                if isinstance(reason, str) and reason:
                    finish_reasons.append(reason)
    return {"done": done, "error_event": error_event,
            "finish_reasons": finish_reasons, "parse_errors": parse_errors}


def _targeted_case_protocol(case_id: str) -> int | None:
    if case_id == "cache-shared-prefix" or case_id.startswith("cancel-"):
        return TARGETED_CASE_PROTOCOL_VERSION
    return None


def _complete_sse_prefix(result: dict[str, Any]) -> dict[str, Any]:
    text = result.get("body_text")
    if not isinstance(text, str):
        return {"done": False, "error_event": False,
                "finish_reasons": [], "parse_errors": []}
    normalized = text.replace("\r\n", "\n")
    boundary = normalized.rfind("\n\n")
    if boundary < 0:
        return {"done": False, "error_event": False,
                "finish_reasons": [], "parse_errors": []}
    complete_prefix = normalized[:boundary + 2]
    if "\ufffd" in complete_prefix:
        return {"done": False, "error_event": False,
                "finish_reasons": [], "parse_errors": ["invalid UTF-8 in complete SSE event"]}
    return parse_sse(complete_prefix.encode("utf-8"))


def _observed_stream_completion(result: object) -> bool:
    if not isinstance(result, dict):
        return False
    framed = _complete_sse_prefix(result)
    reported_reasons = result.get("finish_reasons")
    framed_reasons = framed.get("finish_reasons")
    reasons = [*(reported_reasons if isinstance(reported_reasons, list) else []),
               *(framed_reasons if isinstance(framed_reasons, list) else [])]
    clean_finish = (any(isinstance(reason, str) and reason for reason in reasons)
                    and result.get("error_event") is False
                    and framed.get("error_event") is False
                    and not framed.get("parse_errors"))
    return result.get("done") is True or framed.get("done") is True or clean_finish


def _malformed_complete_sse_event(result: dict[str, Any]) -> bool:
    """Distinguish malformed framed events from an interrupted trailing fragment."""
    if not result.get("parse_errors"):
        return False
    if result.get("read_complete") is not False:
        return True
    text = result.get("body_text")
    if not isinstance(text, str):
        return True
    return bool(_complete_sse_prefix(result)["parse_errors"])


def _valid_fixed_completion(result: object, prompt_tokens: int,
                            completion_tokens: int = 16) -> bool:
    if not isinstance(result, dict) or result.get("status") != 200:
        return False
    body = result.get("body")
    if not isinstance(body, dict) or body.get("error") is not None:
        return False
    choices = body.get("choices")
    usage = body.get("usage")
    return (isinstance(choices, list) and bool(choices)
            and isinstance(choices[0], dict)
            and choices[0].get("finish_reason") == "length"
            and isinstance(usage, dict)
            and usage.get("prompt_tokens") == prompt_tokens
            and usage.get("completion_tokens") == completion_tokens)


def _intentional_overload_code(result: object) -> str | None:
    if not isinstance(result, dict) or result.get("status") != 503:
        return None
    headers = result.get("headers")
    retry_after = None
    if isinstance(headers, dict):
        retry_after = next((value for key, value in headers.items()
                            if isinstance(key, str) and key.lower() == "retry-after"), None)
    body = result.get("body")
    error = body.get("error") if isinstance(body, dict) else None
    code = error.get("code") if isinstance(error, dict) else None
    if (not isinstance(retry_after, str) or not retry_after.strip()
            or not isinstance(error, dict) or error.get("type") != "overloaded"
            or code not in ADMISSION_OVERLOAD_CODES):
        return None
    return code


def _request_cancellation_evidence(request: Any,
                                   cancellation: dict[str, Any]) -> dict[str, Any]:
    result = getattr(request, "result", None)
    finished_ns = getattr(request, "finished_ns", None)
    cancel_call_ns = cancellation.get("cancel_call_ns")
    finished_before_cancel = (
        finished_ns <= cancel_call_ns
        if isinstance(finished_ns, int) and isinstance(cancel_call_ns, int)
        else None)
    terminated = not request.thread.is_alive()
    observed_completion = _observed_stream_completion(result)
    interrupted = bool(
        cancellation.get("started_before_cancel")
        and cancellation.get("thread_alive_before_cancel")
        and not finished_before_cancel
        and terminated
        and not observed_completion
        and (getattr(request, "error", None) is not None
             or isinstance(result, dict) and result.get("read_complete") is False)
    )
    cancelled = getattr(request, "cancelled", None)
    return {
        "request_id": getattr(request, "request_id", None),
        "http_status": getattr(request, "http_status", None),
        "response_headers": getattr(request, "response_headers", {}),
        "result": result,
        "error": getattr(request, "error", None),
        "guard_abort_cause": getattr(request, "guard_abort_cause", None),
        "started_ns": getattr(request, "started_ns", None),
        "first_byte_ns": getattr(request, "first_byte_ns", None),
        "finished_ns": finished_ns,
        "first_byte_hex": getattr(request, "first_byte_hex", None),
        "cancelled_flag": cancelled.is_set() if hasattr(cancelled, "is_set") else None,
        "cancellation": dict(cancellation),
        "terminated": terminated,
        "finished_before_cancel": finished_before_cancel,
        "observed_completion": observed_completion,
        "interrupted": interrupted,
    }


def _cancellation_outcome(receipt: dict[str, Any]) -> tuple[str, str | None]:
    result = receipt.get("result")
    cancellation = receipt.get("cancellation") or {}
    pre_cancel = cancellation.get("pre_cancel") or {}
    guard_cause = receipt.get("guard_abort_cause")
    status = receipt.get("http_status")
    if isinstance(status, int) and status != 200:
        return "fail", f"request returned HTTP {status}"
    if isinstance(pre_cancel.get("http_status"), int) and pre_cancel["http_status"] != 200:
        return "fail", "request returned an HTTP error before cancellation"
    if (isinstance(result, dict)
            and (result.get("error_event") is True
                 or _complete_sse_prefix(result).get("error_event") is True)):
        return "fail", "request stream contained an API error"
    if isinstance(result, dict) and _malformed_complete_sse_event(result):
        return "fail", "request stream contained an API or parse error"
    if guard_cause:
        if guard_cause == "safety_event":
            return "fail", "safety guard aborted the request"
        return "pending", "HTTP guard aborted before intentional cancellation was proven"
    if pre_cancel.get("error") is not None:
        return "fail", "request failed before intentional cancellation"
    if not receipt.get("terminated"):
        return "fail", "client transport did not terminate"
    if receipt.get("finished_before_cancel") is None:
        return "pending", "request finish ordering relative to cancellation was not proven"
    if receipt.get("finished_before_cancel") is True:
        if receipt.get("observed_completion"):
            return "pending", "request completed before cancellation"
        return "fail", "request ended unsuccessfully before cancellation"
    if not cancellation.get("started_before_cancel"):
        return "pending", "request start was not proven before cancellation"
    if not cancellation.get("thread_alive_before_cancel"):
        return "pending", "request was not active when cancellation was requested"
    if receipt.get("observed_completion"):
        return "pending", "request completion was observed before cancellation took effect"
    if receipt.get("cancelled_flag") is not True:
        return "pending", "intentional cancellation was not observed by the request"
    if not receipt.get("interrupted"):
        return "pending", "transport interruption after cancellation was not proven"
    return "pass", None


def validate_config(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError("campaign config must be a JSON object")
    cfg = dict(value)
    campaign_id = validate_campaign_id(cfg.get("campaign_id"))
    duration = cfg.get("duration_hours")
    soak = cfg.get("soak_hours")
    restore = cfg.get("final_restore_minutes")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not 8 <= duration <= 12:
        raise ContractError("duration_hours must be 8..12")
    if isinstance(soak, bool) or not isinstance(soak, (int, float)) or soak < 2:
        raise ContractError("soak_hours must be at least 2")
    if restore != 45:
        raise ContractError("final_restore_minutes must be exactly 45")
    if soak * 60 + restore >= duration * 60:
        raise ContractError("duration must leave time for targeted cases before soak/restoration")
    if "deadline_extension_hours" in cfg:
        extension = cfg["deadline_extension_hours"]
        if (isinstance(extension, bool) or not isinstance(extension, (int, float))
                or not math.isfinite(extension) or extension <= 0 or extension > 12):
            raise ContractError(
                "deadline_extension_hours must be finite and greater than 0 through 12")
        if duration + extension > 24:
            raise ContractError("effective campaign duration must not exceed 24 hours")
    endpoint = urlparse(str(cfg.get("endpoint", "")))
    if endpoint.scheme != "http" or not endpoint.hostname or endpoint.path.rstrip("/") not in {"", "/v1"}:
        raise ContractError("endpoint must be an http://host:port[/v1] URL")
    if endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
        raise ContractError("endpoint must not contain credentials, query, or fragment")
    for key in ("nodes", "host_cache_roots"):
        items = cfg.get(key)
        if not isinstance(items, list) or len(items) != 4:
            raise ContractError(f"{key} must contain four rank-ordered values")
        if not all(isinstance(item, str) and item for item in items):
            raise ContractError(f"{key} entries must be non-empty strings")
        if any(any(char.isspace() for char in item) for item in items):
            raise ContractError(f"{key} entries must not contain whitespace")
    if any(node.startswith("-") for node in cfg["nodes"]):
        raise ContractError("node targets must not begin with '-'")
    if len(set(cfg["nodes"])) != 4:
        raise ContractError("nodes must contain four distinct targets")
    expected_name = namespace_name(campaign_id)
    if any(Path(root).name != expected_name or not Path(root).is_absolute()
           for root in cfg["host_cache_roots"]):
        raise ContractError(f"host_cache_roots must be absolute and end in {expected_name}")
    if cfg.get("container_cache_root") != f"/cache/jit/{expected_name}":
        raise ContractError("container_cache_root differs from the exclusive namespace")
    output = Path(str(cfg.get("output_dir", ""))).expanduser()
    if not output.is_absolute():
        raise ContractError("output_dir must be an absolute private path")
    if not isinstance(cfg.get("model"), str) or not cfg["model"]:
        raise ContractError("model is required")
    if not isinstance(cfg.get("container"), str) or not cfg["container"]:
        raise ContractError("container is required")
    if any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for char in cfg["container"]):
        raise ContractError("container contains unsupported characters")
    repo_root = Path(str(cfg.get("repo_root", ""))).expanduser()
    if not repo_root.is_absolute():
        raise ContractError("repo_root must be an absolute checkout path")
    variant = cfg.get("runtime_variant", "default")
    if variant not in RUNTIME_VARIANTS:
        raise ContractError(
            "runtime_variant must be default, prefill-cache-trim, or prefill-step-cap")
    cfg["runtime_variant"] = variant
    relative = cfg.get("tp4_env")
    expected = f"scripts/resilience/.campaign/{campaign_id}/{RUNTIME_VARIANTS[variant]}"
    if not isinstance(relative, str) or relative.startswith("/") or ".." in relative:
        raise ContractError(f"tp4_env must be {expected}")
    if relative != expected:
        raise ContractError(f"tp4_env must be {expected}")
    started = cfg.get("started_at_unix")
    if (started is not None and (isinstance(started, bool) or not isinstance(started, (int, float))
                                 or started <= 0 or started > time.time() + 300)):
        raise ContractError("started_at_unix must be a Unix timestamp no more than five minutes ahead")
    rigmark = cfg.get("rigmark")
    if not isinstance(rigmark, dict):
        raise ContractError("rigmark.initial and rigmark.final receipt paths are required")
    receipts = [Path(str(rigmark.get(name, ""))).expanduser() for name in ("initial", "final")]
    if any(not path.is_absolute() for path in receipts) or receipts[0] == receipts[1]:
        raise ContractError("rigmark receipts must be distinct absolute paths")
    priority = cfg.get("targeted_case_priority", [])
    if (not isinstance(priority, list)
            or not all(isinstance(case_id, str) and case_id for case_id in priority)):
        raise ContractError("targeted_case_priority must be a list of case IDs")
    if len(priority) != len(set(priority)):
        raise ContractError("targeted_case_priority must not contain duplicates")
    cfg["targeted_case_priority"] = list(priority)
    if "targeted_case_selection" in cfg:
        selection = cfg["targeted_case_selection"]
        if (not isinstance(selection, list) or not selection
                or not all(isinstance(case_id, str) and case_id
                           for case_id in selection)):
            raise ContractError(
                "targeted_case_selection must be a non-empty list of case IDs")
        if len(selection) != len(set(selection)):
            raise ContractError("targeted_case_selection must not contain duplicates")
        cfg["targeted_case_selection"] = list(selection)
    early_soak = cfg.get("soak_stop_after_required_seconds", False)
    if not isinstance(early_soak, bool):
        raise ContractError("soak_stop_after_required_seconds must be a boolean")
    cfg["soak_stop_after_required_seconds"] = early_soak
    if "kv_cache_memory_bytes" in cfg:
        kv_bytes = cfg["kv_cache_memory_bytes"]
        if isinstance(kv_bytes, bool) or not isinstance(kv_bytes, int) or kv_bytes <= 0:
            raise ContractError("kv_cache_memory_bytes must be a positive integer")
    if "bounded_admission" in cfg and cfg["bounded_admission"] is not True:
        raise ContractError("bounded_admission must be true when present")
    final_restore_keys = ("final_restore_env", "final_operational_identity")
    configured_final_restore = [key for key in final_restore_keys if key in cfg]
    if configured_final_restore and len(configured_final_restore) != len(final_restore_keys):
        raise ContractError(
            "final_restore_env and final_operational_identity must be configured together")
    for key in configured_final_restore:
        relative = cfg[key]
        if (not isinstance(relative, str) or not relative
                or Path(relative).is_absolute() or ".." in Path(relative).parts):
            raise ContractError(f"{key} must be a repository-relative path without '..'")
    return cfg


def _stable_sha256(path: Path, label: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"{label} is missing, non-regular, or symlinked: {path}")
    try:
        before = path.stat()
        payload = path.read_bytes()
        after = path.stat()
    except OSError as error:
        raise ContractError(f"cannot read {label}: {error}") from error
    if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)):
        raise ContractError(f"{label} changed while it was being hashed")
    return hashlib.sha256(payload).hexdigest()


def _final_restore_contract(cfg: dict[str, Any]) -> dict[str, Any] | None:
    if "final_restore_env" not in cfg:
        return None
    root = Path(cfg["repo_root"])
    env_relative = cfg["final_restore_env"]
    identity_relative = cfg["final_operational_identity"]
    env_path = root / env_relative
    identity_path = root / identity_relative
    env_sha = _stable_sha256(env_path, "final restoration environment")
    identity_sha = _stable_sha256(identity_path, "final operational identity")
    try:
        identity = load_json(identity_path)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot read final operational identity: {error}") from error
    if _stable_sha256(identity_path, "final operational identity") != identity_sha:
        raise ContractError("final operational identity changed while it was being loaded")
    identity_id = identity.get("identity_id") if isinstance(identity, dict) else None
    if not isinstance(identity_id, str) or not identity_id:
        raise ContractError("final operational identity must contain a non-empty identity_id")
    return {
        "restore_env": env_relative,
        "restore_env_sha256": env_sha,
        "operational_identity": identity_relative,
        "operational_identity_sha256": identity_sha,
        "identity_id": identity_id,
    }


def _worker_contract(cfg: dict[str, Any]) -> dict[str, Any]:
    root = Path(cfg["repo_root"])
    parent_sha = _stable_sha256(root / DEFAULT_WORKER_SOURCE,
                                "protected-default worker source")
    result: dict[str, Any] = {
        "manifest_sha256": None,
        "parent_path": DEFAULT_WORKER_SOURCE,
        "parent_sha256": parent_sha,
        "candidate_path": None,
        "candidate_sha256": None,
    }
    if cfg["runtime_variant"] == "default":
        return result
    manifest_path = root / PREFILL_TRIM_MANIFEST
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ContractError("prefill-cache-trim manifest is missing, non-regular, or symlinked")
    manifest = load_json(manifest_path)
    parent = manifest.get("parent") if isinstance(manifest, dict) else None
    candidate = manifest.get("candidate") if isinstance(manifest, dict) else None
    if (not isinstance(manifest, dict)
            or manifest.get("schema") != "tp4-prefill-cache-trim-candidate-v1"
            or not isinstance(parent, dict) or not isinstance(candidate, dict)
            or parent.get("path") != DEFAULT_WORKER_SOURCE
            or candidate.get("path") != TRIM_WORKER_SOURCE):
        raise ContractError("prefill-cache-trim manifest has an unexpected source contract")
    candidate_sha = candidate.get("sha256")
    if (parent.get("sha256") != parent_sha or not isinstance(candidate_sha, str)
            or len(candidate_sha) != 64
            or any(char not in "0123456789abcdef" for char in candidate_sha)):
        raise ContractError("prefill-cache-trim manifest contains an invalid worker SHA-256")
    if _stable_sha256(root / TRIM_WORKER_SOURCE,
                      "prefill-cache-trim worker source") != candidate_sha:
        raise ContractError("prefill-cache-trim worker differs from its manifest")
    result.update(manifest_sha256=_stable_sha256(
        manifest_path, "prefill-cache-trim manifest"),
        candidate_path=TRIM_WORKER_SOURCE, candidate_sha256=candidate_sha)
    return result


def _scheduler_contract(cfg: dict[str, Any]) -> dict[str, str] | None:
    if cfg["runtime_variant"] != "prefill-step-cap":
        return None
    root = Path(cfg["repo_root"])
    parent_sha = _stable_sha256(root / DEFAULT_SCHEDULER_SOURCE,
                                "protected-default scheduler source")
    manifest_path = root / STEP_CAP_MANIFEST
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ContractError("prefill-step-cap manifest is missing, non-regular, or symlinked")
    manifest = load_json(manifest_path)
    parent = manifest.get("parent") if isinstance(manifest, dict) else None
    candidate = manifest.get("candidate") if isinstance(manifest, dict) else None
    selection = manifest.get("selection") if isinstance(manifest, dict) else None
    if (not isinstance(manifest, dict)
            or manifest.get("schema") != "tp4-prefill-step-cap-candidate-v1"
            or not isinstance(parent, dict) or not isinstance(candidate, dict)
            or not isinstance(selection, dict)
            or parent.get("path") != DEFAULT_SCHEDULER_SOURCE
            or parent.get("sha256") != parent_sha
            or candidate.get("path") != STEP_CAP_SCHEDULER_SOURCE
            or candidate.get("mount_target") != SCHEDULER_TARGET
            or selection.get("environment") != STEP_CAP_ENVIRONMENT):
        raise ContractError("prefill-step-cap manifest has an unexpected source contract")
    candidate_sha = candidate.get("sha256")
    if (not isinstance(candidate_sha, str) or len(candidate_sha) != 64
            or any(char not in "0123456789abcdef" for char in candidate_sha)
            or _stable_sha256(root / STEP_CAP_SCHEDULER_SOURCE,
                              "prefill-step-cap scheduler source") != candidate_sha):
        raise ContractError("prefill-step-cap scheduler differs from its manifest")
    return {
        "manifest_sha256": _stable_sha256(manifest_path, "prefill-step-cap manifest"),
        "parent_path": DEFAULT_SCHEDULER_SOURCE,
        "parent_sha256": parent_sha,
        "candidate_path": STEP_CAP_SCHEDULER_SOURCE,
        "candidate_sha256": candidate_sha,
        "mount_target": SCHEDULER_TARGET,
        "environment": STEP_CAP_ENVIRONMENT,
    }


def _admission_contract(cfg: dict[str, Any]) -> dict[str, Any] | None:
    if cfg.get("bounded_admission") is not True:
        return None
    root = Path(cfg["repo_root"])
    manifest_path = root / ADMISSION_MANIFEST
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ContractError("bounded-admission manifest is missing, non-regular, or symlinked")
    manifest = load_json(manifest_path)
    candidate = manifest.get("candidate") if isinstance(manifest, dict) else None
    selection = manifest.get("selection") if isinstance(manifest, dict) else None
    guard = manifest.get("guard") if isinstance(manifest, dict) else None
    if (not isinstance(candidate, dict) or not isinstance(selection, dict)
            or not isinstance(guard, dict)
            or manifest.get("schema") != "tp4-bounded-admission-candidate-v1"
            or candidate.get("path") != ADMISSION_SOURCE
            or candidate.get("mount_target") != ADMISSION_TARGET
            or candidate.get("import_string") != ADMISSION_IMPORT
            or selection.get("environment") != ADMISSION_ENVIRONMENT
            or guard.get("mode") != ADMISSION_GUARD_MODE
            or not isinstance(guard.get("bypass_paths"), list)
            or not all(isinstance(path, str) for path in guard.get("bypass_paths", ()))
            or len(guard.get("bypass_paths", ())) != len(ADMISSION_BYPASS_PATHS)
            or set(guard.get("bypass_paths", ())) != ADMISSION_BYPASS_PATHS
            or guard.get("bypass_pattern") != ADMISSION_BYPASS_PATTERN):
        raise ContractError("bounded-admission manifest has an unexpected source contract")
    candidate_sha = candidate.get("sha256")
    if (not isinstance(candidate_sha, str) or len(candidate_sha) != 64
            or any(char not in "0123456789abcdef" for char in candidate_sha)
            or _stable_sha256(root / ADMISSION_SOURCE,
                              "bounded-admission middleware") != candidate_sha):
        raise ContractError("bounded-admission middleware differs from its manifest")
    return {
        "manifest_path": ADMISSION_MANIFEST,
        "manifest_sha256": _stable_sha256(
            manifest_path, "bounded-admission manifest"),
        "source_path": ADMISSION_SOURCE,
        "source_sha256": candidate_sha,
        "mount_target": ADMISSION_TARGET,
        "import_string": ADMISSION_IMPORT,
        "environment": dict(ADMISSION_ENVIRONMENT),
    }


def runtime_config_identity(cfg: dict[str, Any], worker: dict[str, Any],
                            scheduler: dict[str, str] | None,
                            admission: dict[str, Any] | None = None) -> dict[str, Any]:
    overlay = Path(cfg["repo_root"]) / cfg["tp4_env"]
    variant = cfg["runtime_variant"]
    uses_trim = variant in {"prefill-cache-trim", "prefill-step-cap"}
    identity = {
        "runtime_variant": variant,
        "tp4_env": cfg["tp4_env"],
        "tp4_env_sha256": _stable_sha256(overlay, "campaign runtime overlay"),
        "worker_manifest": PREFILL_TRIM_MANIFEST if uses_trim else None,
        "worker_manifest_sha256": worker["manifest_sha256"],
        "expected_worker_sha256": (worker["candidate_sha256"] if uses_trim
                                    else worker["parent_sha256"]),
        "candidate_worker_sha256": worker["candidate_sha256"] if uses_trim else None,
        "scheduler_manifest": STEP_CAP_MANIFEST if scheduler else None,
        "scheduler_manifest_sha256": scheduler["manifest_sha256"] if scheduler else None,
        "expected_scheduler_sha256": scheduler["candidate_sha256"] if scheduler else None,
        "candidate_scheduler_sha256": scheduler["candidate_sha256"] if scheduler else None,
    }
    if "kv_cache_memory_bytes" in cfg:
        identity["kv_cache_memory_bytes"] = cfg["kv_cache_memory_bytes"]
    if admission is not None:
        identity.update(
            bounded_admission=True,
            admission_manifest=admission["manifest_path"],
            admission_manifest_sha256=admission["manifest_sha256"],
            admission_source=admission["source_path"],
            admission_source_sha256=admission["source_sha256"],
            admission_mount_target=admission["mount_target"],
            admission_import_string=admission["import_string"],
            admission_environment=dict(admission["environment"]),
        )
    return identity


def _command_option_values(command: object, option: str) -> list[str] | None:
    """Extract one-argument or two-argument option values from Docker Config.Cmd."""
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        return None
    values: list[str] = []
    index = 0
    while index < len(command):
        item = command[index]
        if item == option:
            if index + 1 >= len(command):
                return None
            values.append(command[index + 1])
            index += 2
            continue
        if item.startswith(option + "="):
            values.append(item.split("=", 1)[1])
        index += 1
    return values


def build_plan(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    def add(case_id: str, area: str, **parameters: Any) -> None:
        cases.append({"id": case_id, "area": area, "parameters": parameters,
                      "target_repetitions": TARGET_REPEATS})

    cases.append({"id": "rigmark-initial", "area": "performance", "manual": True,
                  "native_only": True, "target_repetitions": 1})
    for tokens in CONTEXT_TOKENS:
        for concurrency in CONCURRENCIES:
            for cache in ("cold", "replay"):
                add(f"context-{tokens}-c{concurrency}-{cache}", "contexts",
                    tokens=tokens, concurrency=concurrency, cache=cache)
    add(LONG_DECODE_CASE_ID, "contexts",
        tokens=LONG_DECODE_PROMPT_TOKENS, concurrency=5, cache="cold",
        max_tokens=LONG_DECODE_COMPLETION_TOKENS,
        min_tokens=LONG_DECODE_COMPLETION_TOKENS,
        ignore_eos=True, long_decode_pressure=True)
    max_queue = cfg.get("max_queue_clients", 256)
    if (isinstance(max_queue, bool) or not isinstance(max_queue, int) or max_queue < 64
            or max_queue > 1024 or max_queue & (max_queue - 1)):
        raise ContractError("max_queue_clients must be a power of two from 64 through 1024")
    queue_clients = list(QUEUE_WAVES)
    while queue_clients[-1] < max_queue:
        queue_clients.append(queue_clients[-1] * 2)
    for clients in queue_clients:
        add(f"queue-wave-{clients}", "queue", clients=clients, mixed=True)
    if cfg.get("bounded_admission") is True:
        add("admission-overflow", "queue",
            clients=ADMISSION_MAX_ACTIVE + ADMISSION_MAX_QUEUED
                    + ADMISSION_OVERFLOW_EXTRA,
            mixed=False, admission_overflow=True)
    for variant in ("over-context", "invalid-parameters", "malformed-json", "partial-upload"):
        add(f"api-{variant}", "api", variant=variant)
    for stage in ("queue", "prefill", "capture", "publication", "restore", "decode", "multi"):
        add(f"cancel-{stage}", "cancellation", stage=stage)
    for behavior in ("slow-read", "suspended-read", "abandon"):
        add(f"slow-client-{behavior}", "slow_clients", behavior=behavior)
    for variant in ("distinct-snapshots", "shared-prefix", "manifest-growth", "checksum",
                    "missing", "truncated", "enospc", "eio", "delay", "publication-interrupted"):
        add(f"cache-{variant}", "cache", variant=variant)
    for rank in (0, 1):
        for signal in ("TERM", "KILL", "STOP"):
            add(f"worker-r{rank}-{signal.lower()}", "worker", rank=rank, signal=signal)
    add("combination-cache-eio-cancel-restore", "combinations",
        prerequisites=["cache-eio", "cancel-restore"])
    add("combination-queue-slow-cancel", "combinations",
        prerequisites=["queue-wave-64", "slow-client-slow-read", "cancel-queue"])
    cases.append({"id": "soak-mixed", "area": "duration",
                  "parameters": {"hours": cfg["soak_hours"],
                                 "protocol_version": SOAK_PROTOCOL_VERSION},
                  "target_repetitions": 1})
    cases.append({"id": "rigmark-final", "area": "performance", "manual": True,
                  "native_only": True, "target_repetitions": 1})
    priority = cfg.get("targeted_case_priority", [])
    by_id = {case["id"]: case for case in cases}
    invalid = [case_id for case_id in priority
               if (case_id not in by_id or by_id[case_id].get("manual")
                   or by_id[case_id]["area"] == "duration")]
    if invalid:
        raise ContractError(
            "targeted_case_priority contains unknown or non-targeted case IDs: "
            + ", ".join(invalid))
    selection = cfg.get("targeted_case_selection")
    invalid_selection = ([] if selection is None else [
        case_id for case_id in selection
        if (case_id not in by_id or by_id[case_id].get("manual")
            or by_id[case_id]["area"] == "duration")])
    if invalid_selection:
        raise ContractError(
            "targeted_case_selection contains unknown or non-targeted case IDs: "
            + ", ".join(invalid_selection))
    initial = by_id["rigmark-initial"]
    terminal = [by_id["soak-mixed"], by_id["rigmark-final"]]
    targeted = [case for case in cases
                if case["id"] not in {"rigmark-initial", "soak-mixed", "rigmark-final"}]
    prioritized = [by_id[case_id] for case_id in priority]
    priority_set = set(priority)
    ordered = [initial, *prioritized,
               *(case for case in targeted if case["id"] not in priority_set), *terminal]
    positions = {case["id"]: index for index, case in enumerate(ordered)}
    invalid_combinations = [case["id"] for case in ordered
                            if case["area"] == "combinations"
                            and any(positions[prerequisite] > positions[case["id"]]
                                    for prerequisite in case["parameters"]["prerequisites"])]
    if invalid_combinations:
        raise ContractError(
            "targeted_case_priority places a combination before its prerequisites: "
            + ", ".join(invalid_combinations))
    return ordered


class GuardAbortError(ConnectionAbortedError):
    def __init__(self, cause: str):
        self.guard_abort_cause = cause
        super().__init__(f"campaign HTTP guard aborted request: {cause}")


class ClientCapacityError(RuntimeError):
    """The local driver could not create a required request-side thread."""


class SoakReceiptError(OSError):
    """A duration receipt could not be durably appended."""


class PhaseDeadlineError(TimeoutError):
    """A bounded campaign control operation reached its phase deadline."""

    guard_abort_cause = "phase_deadline"


class HTTPClient:
    def __init__(self, endpoint: str, model: str, timeout: float = 900):
        parsed = urlparse(endpoint)
        self.host = parsed.hostname or ""
        self.port = parsed.port or 80
        self.base = parsed.path.rstrip("/") or "/v1"
        self.model = model
        self.timeout = timeout
        self.abort_event: threading.Event | None = None
        self.deadline: Callable[[], float] | None = None

    def set_guard(self, abort_event: threading.Event, deadline: Callable[[], float]) -> None:
        self.abort_event, self.deadline = abort_event, deadline

    def raw(self, path: str, payload: bytes, *, headers: dict[str, str] | None = None,
            read_limit: int = 32 << 20) -> dict[str, Any]:
        remaining = self.deadline() - time.monotonic() if self.deadline else self.timeout
        if self.abort_event is not None and self.abort_event.is_set():
            raise GuardAbortError("safety_event")
        if remaining <= 0:
            raise GuardAbortError("phase_deadline")
        socket_timeout = (max(0.1, remaining) + GUARDED_SOCKET_GRACE_SECONDS
                          if self.deadline else self.timeout)
        connection = http.client.HTTPConnection(self.host, self.port, timeout=socket_timeout)
        finished = threading.Event()
        aborted = threading.Event()
        abort_cause: list[str | None] = [None]
        transport: socket.socket | None = None
        def abort_watch() -> None:
            while not finished.wait(0.05):
                expired = self.deadline is not None and time.monotonic() >= self.deadline()
                stopped = self.abort_event is not None and self.abort_event.is_set()
                if expired or stopped:
                    abort_cause[0] = "safety_event" if stopped else "phase_deadline"
                    aborted.set()
                    try:
                        active_socket = transport or connection.sock
                        if active_socket is not None:
                            active_socket.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    connection.close()
        watcher = threading.Thread(target=abort_watch, daemon=True)
        try:
            watcher.start()
        except (OSError, RuntimeError) as error:
            connection.close()
            raise ClientCapacityError(
                f"HTTP guard watcher could not start: {error}") from error
        started = time.time_ns()
        try:
            actual_headers = {"Content-Type": "application/json", **(headers or {})}
            connection.request("POST", path, body=payload, headers=actual_headers)
            # Keep the transport itself: getresponse() may clear connection.sock
            # for Connection: close while HTTPResponse.fp is still reading it.
            transport = connection.sock
            response = connection.getresponse()
            body = response.read(read_limit + 1)
            if aborted.is_set():
                raise GuardAbortError(abort_cause[0] or "campaign_guard")
            if len(body) > read_limit:
                raise RuntimeError("response exceeds receipt limit")
            try:
                decoded = json.loads(body)
            except (UnicodeError, json.JSONDecodeError):
                decoded = None
            return {"status": response.status, "headers": dict(response.getheaders()),
                    "body": decoded, "body_text": None if decoded is not None else body.decode("utf-8", "replace"),
                    "started_ns": started, "finished_ns": time.time_ns()}
        except GuardAbortError:
            raise
        except Exception as error:
            if aborted.is_set():
                raise GuardAbortError(abort_cause[0] or "campaign_guard") from error
            raise
        finally:
            finished.set()
            connection.close()
            watcher.join(timeout=0.2)

    def json(self, path: str, value: Any, **kwargs: Any) -> dict[str, Any]:
        return self.raw(path, json.dumps(value, separators=(",", ":")).encode(), **kwargs)

    def tokenize_messages(self, messages: list[dict[str, str]]) -> int:
        result = self.json("/tokenize", {
            "model": self.model,
            "messages": messages,
            "add_generation_prompt": True,
            "chat_template_kwargs": {"reasoning_effort": "low"},
        })
        body = result.get("body")
        count = body.get("count") if isinstance(body, dict) else None
        if not isinstance(count, int):
            tokens = body.get("tokens") if isinstance(body, dict) else None
            count = len(tokens) if isinstance(tokens, list) else None
        if result["status"] != 200 or not isinstance(count, int):
            raise RuntimeError("/tokenize did not return an exact token count")
        return count

    def tokenize(self, content: str) -> int:
        return self.tokenize_messages([{"role": "user", "content": content}])

    def exact_messages(self, target: int, history: list[dict[str, str]],
                       newest_user_prefix: str) -> list[dict[str, str]]:
        if target < 1 or target > 64_000:
            raise ValueError("soak target token count must be between 1 and 64000")
        if len(history) > SOAK_MAX_HISTORY_MESSAGES:
            raise ValueError("soak conversation history exceeds its bounded message count")
        base = [dict(message) for message in history]

        def candidate(padding: int) -> list[dict[str, str]]:
            return [*base, {"role": "user",
                            "content": newest_user_prefix + (" x" * padding)}]

        low, high = 0, target + 1024
        while self.tokenize_messages(candidate(high)) < target:
            high *= 2
        found: list[dict[str, str]] | None = None
        while low <= high:
            middle = (low + high) // 2
            messages = candidate(middle)
            count = self.tokenize_messages(messages)
            if count == target:
                found = messages
                break
            if count < target:
                low = middle + 1
            else:
                high = middle - 1
        if found is None:
            for padding in range(max(0, low - 8), low + 9):
                messages = candidate(padding)
                if self.tokenize_messages(messages) == target:
                    found = messages
                    break
        if found is None:
            raise RuntimeError(f"cannot synthesize exactly {target} chat tokens")
        return found

    def exact_prompt(self, target: int, prefix: str) -> str:
        if target < 1:
            raise ValueError("target token count must be positive")
        low, high = 0, target + 1024
        while self.tokenize(prefix + (" x" * high)) < target:
            high *= 2
        found: str | None = None
        while low <= high:
            middle = (low + high) // 2
            candidate = prefix + (" x" * middle)
            count = self.tokenize(candidate)
            if count == target:
                found = candidate
                break
            if count < target:
                low = middle + 1
            else:
                high = middle - 1
        if found is None:
            for n in range(max(0, low - 8), low + 9):
                candidate = prefix + (" x" * n)
                if self.tokenize(candidate) == target:
                    found = candidate
                    break
        if found is None:
            raise RuntimeError(f"cannot synthesize exactly {target} chat tokens")
        return found

    def chat_payload_messages(self, messages: list[dict[str, str]], *,
                              stream: bool = False, max_tokens: int = 16,
                              min_tokens: int | None = None,
                              ignore_eos: bool = False) -> dict[str, Any]:
        payload = {"model": self.model, "temperature": 0, "max_tokens": max_tokens,
                   "stream": stream,
                   "chat_template_kwargs": {"reasoning_effort": "low"},
                   "messages": [dict(message) for message in messages]}
        if min_tokens is not None:
            payload["min_tokens"] = min_tokens
        if ignore_eos:
            payload["ignore_eos"] = True
        return payload

    def chat_payload(self, prompt: str, *, stream: bool = False, max_tokens: int = 16,
                     min_tokens: int | None = None,
                     ignore_eos: bool = False) -> dict[str, Any]:
        return self.chat_payload_messages([{"role": "user", "content": prompt}],
                                          stream=stream, max_tokens=max_tokens,
                                          min_tokens=min_tokens,
                                          ignore_eos=ignore_eos)

    def chat_messages(self, messages: list[dict[str, str]], *, request_id: str,
                      max_tokens: int = 16) -> dict[str, Any]:
        return self.json(f"{self.base}/chat/completions",
                         self.chat_payload_messages(messages, max_tokens=max_tokens),
                         headers={"X-Request-Id": request_id})

    def chat(self, prompt: str, *, request_id: str, max_tokens: int = 16) -> dict[str, Any]:
        return self.json(f"{self.base}/chat/completions",
                         self.chat_payload(prompt, max_tokens=max_tokens),
                         headers={"X-Request-Id": request_id})


class CancelableRequest:
    def __init__(self, client: HTTPClient, payload: dict[str, Any], request_id: str):
        self.client, self.payload, self.request_id = client, payload, request_id
        self.connection: http.client.HTTPConnection | None = None
        self.result: dict[str, Any] | None = None
        self.error: str | None = None
        self.started = threading.Event()
        self.first_byte = threading.Event()
        self.cancelled = threading.Event()
        self.transport: socket.socket | None = None
        self.http_status: int | None = None
        self.response_headers: dict[str, str] = {}
        self.started_ns: int | None = None
        self.first_byte_ns: int | None = None
        self.finished_ns: int | None = None
        self.first_byte_hex: str | None = None
        self.guard_abort_cause: str | None = None
        self.client_capacity_error = False
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _record_received(self, payload: bytes, *, read_complete: bool,
                         truncated: bool = False) -> None:
        stream = parse_sse(payload)
        self.result = {"status": self.http_status, "received_bytes": len(payload),
                       "body_text": payload.decode("utf-8", "replace"),
                       "read_complete": read_complete, "receipt_truncated": truncated,
                       "stream_complete": bool(stream["done"]), **stream}

    def _run(self) -> None:
        remaining = (self.client.deadline() - time.monotonic()
                     if self.client.deadline else self.client.timeout)
        if self.cancelled.is_set():
            self.error = "CancelledError: request cancelled before connect"
            self.finished_ns = time.time_ns()
            return
        socket_timeout = (max(0.1, remaining) + GUARDED_SOCKET_GRACE_SECONDS
                          if self.client.deadline else self.client.timeout)
        connection = http.client.HTTPConnection(self.client.host, self.client.port,
                                                timeout=socket_timeout)
        self.connection = connection
        finished = threading.Event()
        def abort_watch() -> None:
            while not finished.wait(0.05):
                expired = (self.client.deadline is not None
                           and time.monotonic() >= self.client.deadline())
                stopped = (self.client.abort_event is not None
                           and self.client.abort_event.is_set())
                if expired or stopped or self.cancelled.is_set():
                    if expired or stopped:
                        self.guard_abort_cause = (
                            "safety_event" if stopped else "phase_deadline")
                    self.cancel()
        watcher = threading.Thread(target=abort_watch, daemon=True)
        try:
            watcher.start()
        except (OSError, RuntimeError) as error:
            self.client_capacity_error = True
            self.error = f"ClientCapacityError: HTTP guard watcher could not start: {error}"
            self.finished_ns = time.time_ns()
            connection.close()
            return
        received = bytearray()
        receipt_limit = 8 << 20
        receipt_truncated = False
        try:
            self.started_ns = time.time_ns()
            body = json.dumps(self.payload, separators=(",", ":")).encode()
            if self.cancelled.is_set():
                raise ConnectionAbortedError("request cancelled before send")
            connection.request("POST", f"{self.client.base}/chat/completions", body=body,
                               headers={"Content-Type": "application/json",
                                        "X-Request-Id": self.request_id})
            self.transport = connection.sock
            if self.cancelled.is_set():
                self.cancel()
                raise ConnectionAbortedError("request cancelled while connecting")
            self.started.set()
            response = connection.getresponse()
            self.http_status = response.status
            self.response_headers = dict(response.getheaders())
            first = response.read(1)
            if first:
                received.extend(first)
                self.first_byte_ns = time.time_ns()
                self.first_byte_hex = first.hex()
                self.first_byte.set()
            read_chunk = getattr(response, "read1", response.read)
            while True:
                remaining_bytes = max(0, receipt_limit - len(received))
                chunk = read_chunk(min(64 << 10, remaining_bytes + 1))
                if not chunk:
                    break
                if len(chunk) > remaining_bytes:
                    received.extend(chunk[:remaining_bytes])
                    receipt_truncated = True
                    raise RuntimeError("cancellation response exceeds receipt limit")
                received.extend(chunk)
                if self.cancelled.is_set():
                    raise ConnectionAbortedError("request cancelled while reading stream")
            if self.cancelled.is_set():
                raise ConnectionAbortedError("request cancelled while reading stream")
            self._record_received(bytes(received), read_complete=True)
        except Exception as error:  # expected when the main thread cancels the socket
            partial = getattr(error, "partial", b"")
            if isinstance(partial, bytes) and partial:
                remaining_bytes = max(0, receipt_limit - len(received))
                received.extend(partial[:remaining_bytes])
                receipt_truncated = len(partial) > remaining_bytes
            if received:
                self._record_received(bytes(received), read_complete=False,
                                      truncated=receipt_truncated)
            self.error = f"{type(error).__name__}: {error}"
        finally:
            self.finished_ns = time.time_ns()
            finished.set()
            connection.close()
            watcher.join(timeout=0.2)

    def start(self) -> None:
        self.thread.start()

    def cancel(self) -> None:
        self.cancelled.set()
        if self.connection is not None:
            try:
                active_socket = self.transport or self.connection.sock
                active_socket.shutdown(socket.SHUT_RDWR) if active_socket else None
            except OSError:
                pass


class Telemetry:
    def __init__(self, cfg: dict[str, Any], output: Path):
        self.cfg, self.output = cfg, output
        self.stop_event = threading.Event()
        self.safety_event = threading.Event()
        self.expected_down = False
        self.expected_down_until = 0.0
        self.reasons: list[str] = []
        self.processes: list[subprocess.Popen[bytes]] = []
        self.stderr_streams: list[Any] = []
        self.threads: list[threading.Thread] = []
        self.reader_threads: list[threading.Thread] = []
        self.latest: dict[int, dict[str, Any]] = {}
        self.receiver_state: dict[int, dict[str, Any]] = {}
        self.lock = threading.Lock()
        self.last_seen: dict[int, float] = {}
        self.container_pids: dict[int, int] = {}
        self.interval = 1.0
        self.started = 0.0

    def _trip(self, reason: str) -> None:
        if reason not in self.reasons:
            self.reasons.append(reason)
        self.safety_event.set()

    def arm_expected_down(self, timeout: float) -> None:
        if timeout <= 0:
            raise ValueError("expected-down timeout must be positive")
        self.expected_down = True
        self.expected_down_until = time.monotonic() + min(timeout, 2700.0)

    def clear_expected_down(self) -> None:
        self.expected_down = False
        self.expected_down_until = 0.0

    def _expected_down_active(self, now: float | None = None) -> bool:
        if not self.expected_down:
            return False
        now = time.monotonic() if now is None else now
        if now < self.expected_down_until:
            return True
        self.clear_expected_down()
        self._trip("expected restart window expired")
        return False

    def start(self, interval: float) -> None:
        self.stop()
        self.stop_event = threading.Event()
        self.interval = interval
        self.started = time.monotonic()
        self.last_seen = {rank: self.started for rank in range(4)}
        self.receiver_state = {rank: {
            "receive_monotonic": None, "append_complete_monotonic": None,
            "source_time_ns": None, "source_monotonic_ns": None,
        } for rank in range(4)}
        remote = f"tp4/scripts/resilience/.campaign/{self.cfg['campaign_id']}/probe.py"
        for rank, (host, root) in enumerate(zip(self.cfg["nodes"], self.cfg["host_cache_roots"])):
            remote_command = shlex.join(["sudo", "-n", "python3", remote,
                "--container", self.cfg["container"], "--cache-root", root,
                "--rank", str(rank), "--api-port",
                str(urlparse(self.cfg["endpoint"]).port or 80),
                "--container-cache-root", self.cfg["container_cache_root"],
                "--interval", str(interval)])
            command = [*SSH, host, remote_command]
            stderr_path = self.output / f"telemetry-rank{rank}.stderr.log"
            stderr_fd = os.open(stderr_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            stderr_stream = os.fdopen(stderr_fd, "ab", buffering=0)
            try:
                process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                           stderr=stderr_stream, bufsize=-1)
            except BaseException:
                stderr_stream.close()
                raise
            self.stderr_streams.append(stderr_stream)
            self.processes.append(process)
            thread = threading.Thread(target=self._reader, args=(rank, process), daemon=True)
            thread.start()
            self.threads.append(thread)
            self.reader_threads.append(thread)
        watchdog = threading.Thread(target=self._watchdog, daemon=True)
        watchdog.start()
        self.threads.append(watchdog)

    def _watchdog(self) -> None:
        grace = 10.0
        while not self.stop_event.wait(min(0.2, self.interval)):
            now = time.monotonic()
            if now - self.started <= grace or self._expected_down_active(now):
                continue
            with self.lock:
                stale = [rank for rank in range(4)
                         if now - self.last_seen.get(rank, self.started) > max(2.0, 4 * self.interval)]
            if stale:
                self._trip(f"management telemetry stale for ranks {stale}")
                try:
                    self._persist_receiver_diagnostic(now, stale)
                except Exception as error:
                    self._trip(
                        "telemetry receiver diagnostic write failed: "
                        f"{type(error).__name__}: {error}")
                return

    def _persist_receiver_diagnostic(self, now: float, stale: list[int]) -> None:
        with self.lock:
            receiver = {rank: dict(self.receiver_state.get(rank, {}))
                        for rank in range(4)}
            last_seen = dict(self.last_seen)
            latest = {rank: self.latest.get(rank, {}) for rank in range(4)}
        ranks = []
        for rank in range(4):
            sample = latest.get(rank) or {}
            ranks.append({
                "rank": rank,
                "stale": rank in stale,
                "last_seen_age_seconds": max(0.0, now - last_seen.get(rank, self.started)),
                "receive_monotonic": receiver[rank].get("receive_monotonic"),
                "append_complete_monotonic": receiver[rank].get(
                    "append_complete_monotonic"),
                "source_time_ns": receiver[rank].get("source_time_ns",
                                                     sample.get("time_ns")),
                "source_monotonic_ns": receiver[rank].get(
                    "source_monotonic_ns", sample.get("monotonic_ns")),
                "process_returncode": (self.processes[rank].poll()
                                       if rank < len(self.processes) else None),
                "stderr_path": str(self.output / f"telemetry-rank{rank}.stderr.log"),
            })
        _append_private_event(
            self.output / "telemetry-receiver-diagnostics.jsonl",
            {"kind": "telemetry_receiver_stale", "time_ns": time.time_ns(),
             "watchdog_monotonic": now, "stale_ranks": list(stale), "ranks": ranks})

    def _sample_reasons(self, rank: int, sample: dict[str, Any], now: float) -> list[str]:
        reasons: list[str] = []
        grace_elapsed = now - self.started > 10
        expected_down = self._expected_down_active(now)
        memory = sample.get("mem_available_bytes")
        if memory is None and grace_elapsed:
            reasons.append(f"rank {rank} MemAvailable is unknown")
        elif isinstance(memory, int) and memory < MIN_MEM_AVAILABLE_BYTES:
            reasons.append(f"rank {rank} MemAvailable below 768 MiB")
        cache = sample.get("cache") or {}
        if (not expected_down and grace_elapsed
                and (cache.get("scan_complete") is not True
                     or cache.get("sample_fresh") is not True)):
            reasons.append(f"rank {rank} cache accounting is incomplete or stale")
        if isinstance(cache.get("bytes"), int) and cache["bytes"] > MAX_NAMESPACE_BYTES:
            reasons.append(f"rank {rank} namespace exceeds 8 GiB")
        if not expected_down and cache.get("bytes") is None and grace_elapsed:
            reasons.append(f"rank {rank} namespace size is unknown")
        if not expected_down and sample.get("container_state") != "running":
            reasons.append(f"rank {rank} worker is not running")
        pid = sample.get("container_pid")
        previous_pid = self.container_pids.get(rank)
        if expected_down and isinstance(pid, int):
            self.container_pids[rank] = pid
        elif isinstance(pid, int) and previous_pid is None:
            self.container_pids[rank] = pid
        elif (not expected_down and isinstance(pid, int)
              and previous_pid is not None and pid != previous_pid):
            reasons.append(f"rank {rank} container process changed unexpectedly")

        oom = sample.get("oom") or {}
        coverage = oom.get("coverage") or {}
        required = {
            "host_oom_kills_since_start": "host_oom_delta_known",
            "cgroup_oom_kills_since_start": "cgroup_oom_delta_known",
        }
        for key, coverage_key in required.items():
            value = oom.get(key)
            known = (coverage.get(coverage_key) is True
                     and isinstance(value, int) and not isinstance(value, bool))
            if known and value > 0:
                reasons.append(f"rank {rank} new earlyoom/OOM evidence ({key})")
            if (grace_elapsed and not known
                    and (key.startswith("host_") or not expected_down)):
                reasons.append(f"rank {rank} OOM coverage is unknown ({key})")
        for key in ("earlyoom_kills", "cuda_ooms_log_window", "cuda_oom_seen_since_start"):
            value = oom.get(key)
            if (isinstance(value, int) and not isinstance(value, bool) and value > 0
                    or value is True):
                reasons.append(f"rank {rank} new earlyoom/OOM evidence ({key})")
        return reasons

    def _reader(self, rank: int, process: subprocess.Popen[bytes]) -> None:
        path = self.output / f"telemetry-rank{rank}.jsonl"
        assert process.stdout is not None
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(descriptor, "ab") as stream:
            for line in process.stdout:
                received = time.monotonic()
                with self.lock:
                    self.receiver_state.setdefault(rank, {})[
                        "receive_monotonic"] = received
                stream.write(line)
                stream.flush()
                appended = time.monotonic()
                with self.lock:
                    self.receiver_state.setdefault(rank, {})[
                        "append_complete_monotonic"] = appended
                try:
                    sample = json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if not isinstance(sample, dict):
                    continue
                with self.lock:
                    self.latest[rank] = sample
                    now = appended
                    self.last_seen[rank] = now
                    self.receiver_state[rank].update(
                        source_time_ns=sample.get("time_ns"),
                        source_monotonic_ns=sample.get("monotonic_ns"))
                reasons = self._sample_reasons(rank, sample, now)
                if reasons:
                    for reason in reasons:
                        self._trip(reason)
        if not self.stop_event.is_set():
            self._trip(f"management telemetry stream ended for rank {rank}")

    def stop(self) -> None:
        if not self.processes:
            return
        self.stop_event.set()
        for process in self.processes:
            process.terminate()
        for process in self.processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        for thread in self.threads:
            thread.join(timeout=5)
        for process, reader in zip(self.processes, self.reader_threads):
            if process.stdout is not None and not reader.is_alive():
                process.stdout.close()
        for stream in self.stderr_streams:
            stream.close()
        self.processes.clear()
        self.stderr_streams.clear()
        self.threads.clear()
        self.reader_threads.clear()

    def settle_expected_restart(self, since: float, timeout: float = 10) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.lock:
                ready = all(self.last_seen.get(rank, 0) > since
                            and self.latest.get(rank, {}).get("container_state") == "running"
                            and isinstance(self.latest.get(rank, {}).get("container_pid"), int)
                            for rank in range(4))
                if ready:
                    self.container_pids = {rank: self.latest[rank]["container_pid"]
                                           for rank in range(4)}
                    return True
            time.sleep(0.1)
        self._trip("post-restart telemetry did not establish four running ranks")
        return False


class Campaign:
    def __init__(self, cfg: dict[str, Any], *, acknowledge_faults: bool):
        self.cfg = cfg
        if cfg.get("started_at_unix") is None:
            raise ContractError("execution requires started_at_unix from the initial Rigmark start")
        if not (Path(cfg["repo_root"]) / "scripts/tp4ctl").is_file():
            raise ContractError("repo_root does not contain scripts/tp4ctl")
        self.output = require_private_dir(Path(cfg["output_dir"]), create=True)
        self.events = self.output / "events.jsonl"
        self.state_path = self.output / "campaign.json"
        self.plan = build_plan(cfg)
        self.worker_contract = _worker_contract(cfg)
        self.scheduler_contract = _scheduler_contract(cfg)
        self.admission_contract = _admission_contract(cfg)
        self.final_restore_contract = _final_restore_contract(cfg)
        self.runtime_config_identity = runtime_config_identity(
            cfg, self.worker_contract, self.scheduler_contract,
            self.admission_contract)
        self.initial_native_rigmark_identity = {
            "identity_scope": "protected-default/native-rigmark",
            "runtime_variant": "default",
            "tp4_env": None,
            "tp4_env_sha256": None,
            "expected_worker_sha256": self.worker_contract["parent_sha256"],
            "candidate_worker_sha256": None,
        }
        self.final_native_rigmark_identity = {
            **self.initial_native_rigmark_identity,
            "restoration_verified": False,
        }
        # Kept as a compatibility alias for callers constructing focused test doubles.
        self.native_rigmark_identity = self.initial_native_rigmark_identity
        self.http = HTTPClient(cfg["endpoint"], cfg["model"])
        self.telemetry = Telemetry(cfg, self.output)
        self.acknowledge_faults = acknowledge_faults
        self.state = self._initial_state()
        initial_valid, initial_evidence = self._receipt_evidence(
            Path(cfg["rigmark"]["initial"]), self.state["started_ns"])
        if not initial_valid:
            raise ContractError(f"initial Rigmark receipt is not ready: {initial_evidence['reason']}")
        remaining = (self.state["deadline_ns"] - time.time_ns()) / 1_000_000_000
        self.end_monotonic = time.monotonic() + max(0, remaining)
        self.restore_at = self.end_monotonic - cfg["final_restore_minutes"] * 60
        self.soak_at = (self.restore_at - SOAK_FINAL_PROOF_RESERVE_SECONDS
                        - cfg["soak_hours"] * 3600)
        self.targeted_stop_at = self.soak_at - TARGETED_STOP_MARGIN_SECONDS
        self.active_deadline = self.targeted_stop_at
        self.http.set_guard(self.telemetry.safety_event, lambda: self.active_deadline)

    def _initial_state(self) -> dict[str, Any]:
        base_duration_ns = int(
            self.cfg["duration_hours"] * 3600 * 1_000_000_000)
        extension_hours = self.cfg.get("deadline_extension_hours", 0)
        effective_duration_ns = base_duration_ns + int(
            extension_hours * 3600 * 1_000_000_000)
        if self.state_path.exists():
            value = load_json(self.state_path)
            if value.get("campaign_id") != self.cfg["campaign_id"]:
                raise ContractError("existing receipt belongs to another campaign")
            recorded_base = (value.get("limits") or {}).get("deadline_hours")
            if recorded_base != self.cfg["duration_hours"]:
                raise ContractError("existing receipt base duration differs from configuration")
            if "deadline_ns" not in value:
                value["deadline_ns"] = int(value["started_ns"]) + base_duration_ns
            requested_deadline = int(value["started_ns"]) + effective_duration_ns
            current_deadline = int(value["deadline_ns"])
            if current_deadline < int(value["started_ns"]) + base_duration_ns:
                raise ContractError("existing receipt deadline is shorter than its base duration")
            if current_deadline > requested_deadline:
                raise ContractError("deadline_extension_hours cannot shrink an existing deadline")
            if current_deadline < requested_deadline:
                value.setdefault("deadline_extension_history", []).append({
                    "recorded_ns": time.time_ns(),
                    "old_deadline_ns": current_deadline,
                    "new_deadline_ns": requested_deadline,
                    "configured_extension_hours": extension_hours,
                    "added_hours": ((requested_deadline - current_deadline)
                                    / 3_600_000_000_000),
                    "reason": "explicit owner deadline extension",
                })
                value["deadline_ns"] = requested_deadline
            value["limits"]["deadline_extension_hours"] = extension_hours
            value["limits"]["effective_deadline_hours"] = (
                self.cfg["duration_hours"] + extension_hours)
            recorded = value.get("runtime_config_identity")
            restoration = value.get("restoration")
            gates = restoration.get("gates") if isinstance(restoration, dict) else None
            identity = restoration.get("identity") if isinstance(restoration, dict) else None
            restored = ("restoration_completed_ns" in value
                        and isinstance(gates, dict) and gates.get("pass") is True
                        and isinstance(identity, dict) and identity.get("returncode") == 0)
            has_prior_work = bool(value.get("cases"))
            if ((recorded is not None and recorded != self.runtime_config_identity)
                    or recorded is None and has_prior_work) and not restored:
                raise ContractError(
                    "runtime configuration differs from an unrestored or unrecorded active attempt")
            return value
        started_ns = int(float(self.cfg["started_at_unix"]) * 1_000_000_000)
        value = {"schema": "tp4-resilience-campaign/v1", "campaign_id": self.cfg["campaign_id"],
                "started_ns": started_ns,
                "deadline_ns": started_ns + effective_duration_ns,
                "status": "running", "cases": {},
                "runtime_config_identity": self.runtime_config_identity,
                "limits": {"namespace_bytes": MAX_NAMESPACE_BYTES,
                           "mem_available_stop_bytes": MIN_MEM_AVAILABLE_BYTES,
                           "deadline_hours": self.cfg["duration_hours"],
                           "deadline_extension_hours": extension_hours,
                           "effective_deadline_hours": (
                               self.cfg["duration_hours"] + extension_hours),
                           "final_restore_minutes": self.cfg["final_restore_minutes"]}}
        if extension_hours:
            value["deadline_extension_history"] = [{
                "recorded_ns": time.time_ns(),
                "old_deadline_ns": started_ns + base_duration_ns,
                "new_deadline_ns": value["deadline_ns"],
                "configured_extension_hours": extension_hours,
                "added_hours": extension_hours,
                "reason": "explicit owner deadline extension",
            }]
        return value

    def save(self) -> None:
        atomic_json(self.state_path, self.state)

    def append_driver_event(self, event: dict[str, Any]) -> None:
        """Append a complete private driver/helper receipt."""
        _append_private_event(self.events, event)

    @staticmethod
    def _receipt_evidence(path: Path, threshold_ns: int | None) -> tuple[bool, dict[str, Any]]:
        if path.is_symlink() or not path.is_file():
            return False, {"reason": "receipt is missing, non-regular, or symlinked"}
        try:
            before = path.stat()
            payload = path.read_bytes()
            after = path.stat()
            value = json.loads(payload)
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            return False, {"reason": f"receipt is not a complete JSON object: {error}"}
        if not isinstance(value, dict):
            return False, {"reason": "receipt JSON root is not an object"}
        if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)):
            return False, {"reason": "receipt changed while it was being validated"}
        if threshold_ns is None or before.st_mtime_ns < threshold_ns:
            return False, {"reason": "receipt predates its required phase"}
        return True, {"receipt": str(path), "bytes": before.st_size,
                      "sha256": hashlib.sha256(payload).hexdigest(),
                      "note": "native Rigmark artifact recorded without judging its result"}

    def record(self, case: dict[str, Any], repetition: int, status: str,
               evidence: dict[str, Any]) -> None:
        if case.get("manual"):
            identity = (self.final_native_rigmark_identity
                        if case.get("id") == "rigmark-final"
                        else self.initial_native_rigmark_identity)
        else:
            identity = self.runtime_config_identity
        row = self.state["cases"].setdefault(case["id"], {"area": case["area"],
                    "parameters": case.get("parameters", {}), "target_repetitions": case["target_repetitions"],
                    "runs": []})
        if case.get("area") == "duration":
            row["active_parameters"] = dict(case.get("parameters", {}))
        row["runs"].append({"repetition": repetition, "status": status,
                            "time_ns": time.time_ns(),
                            "runtime_config_identity": dict(identity),
                            "evidence": evidence})
        append_event(self.events, {"kind": "case", "case_id": case["id"],
                                   "repetition": repetition, "status": status,
                                   "runtime_variant": identity["runtime_variant"],
                                   "identity_scope": identity.get("identity_scope", "campaign-overlay")})
        self.save()

    @staticmethod
    def _successes_after_last_failure(
            runs: list[dict[str, Any]], qualifying: str,
            runtime_config_identity: dict[str, Any] | None = None,
            protocol_version: int | None = None) -> int:
        last_failure = max((index for index, run in enumerate(runs)
                            if run.get("status") == "fail"), default=-1)
        return sum(run.get("status") == qualifying
                   and (runtime_config_identity is None
                        or run.get("runtime_config_identity") == runtime_config_identity)
                   and (protocol_version is None
                        or (run.get("evidence") or {}).get("protocol_version") == protocol_version)
                   for run in runs[last_failure + 1:])

    def _captured_chat(self, prompt: str, request_id: str,
                       request_options: dict[str, Any] | None = None) -> dict[str, Any]:
        started_ns = time.time_ns()
        payload_receipt = None
        try:
            if request_options is None:
                result = dict(self.http.chat(prompt, request_id=request_id))
            else:
                payload = self.http.chat_payload(prompt, **request_options)
                payload_receipt = {
                    "sha256": hashlib.sha256(canonical_json(payload)).hexdigest(),
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "max_tokens": payload.get("max_tokens"),
                    "min_tokens": payload.get("min_tokens"),
                    "ignore_eos": payload.get("ignore_eos"),
                }
                result = dict(self.http.json(
                    f"{self.http.base}/chat/completions", payload,
                    headers={"X-Request-Id": request_id}))
            result.setdefault("started_ns", started_ns)
            result.setdefault("finished_ns", time.time_ns())
            result.update(request_id=request_id, error=None, error_origin=None,
                          guard_abort_cause=None)
            if payload_receipt is not None:
                result["request_payload_receipt"] = payload_receipt
            return result
        except Exception as error:
            result = {
                "request_id": request_id,
                "status": None,
                "started_ns": started_ns,
                "finished_ns": time.time_ns(),
                "error": f"{type(error).__name__}: {error}",
                "error_origin": ("client_capacity"
                                 if isinstance(error, ClientCapacityError)
                                 else "client_request"),
                "guard_abort_cause": getattr(error, "guard_abort_cause", None),
            }
            if payload_receipt is not None:
                result["request_payload_receipt"] = payload_receipt
            return result

    def _request_group(self, prompts: list[str], case_id: str,
                       request_options: dict[str, Any] | None = None) -> list[dict[str, Any]]:

        with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
            futures = {}
            for index, prompt in enumerate(prompts):
                request_id = f"{case_id}-{index}"
                future = pool.submit(
                    self._captured_chat, prompt, request_id, request_options)
                futures[future] = request_id
            results: list[dict[str, Any]] = []
            for future in as_completed(futures):
                request_id = futures[future]
                try:
                    result = dict(future.result())
                except Exception as error:
                    result = {"request_id": request_id, "status": None,
                              "started_ns": None, "finished_ns": time.time_ns(),
                              "error": f"{type(error).__name__}: {error}",
                              "guard_abort_cause": None}
                results.append(result)
            return results

    @staticmethod
    def _valid(results: list[dict[str, Any]]) -> bool:
        for result in results:
            body = result.get("body")
            if result.get("status") != 200 or not isinstance(body, dict) or not body.get("choices"):
                return False
        return True

    @staticmethod
    def _prompt_token_counts(results: list[dict[str, Any]]) -> list[int | None]:
        counts: list[int | None] = []
        for result in results:
            body = result.get("body")
            usage = body.get("usage") if isinstance(body, dict) else None
            count = usage.get("prompt_tokens") if isinstance(usage, dict) else None
            counts.append(count if isinstance(count, int) and not isinstance(count, bool) else None)
        return counts

    @staticmethod
    def _publication_digest(events: list[dict[str, Any]]) -> str | None:
        valid = [event for event in events
                 if event.get("outcome") == "ok" and event.get("committed") is True
                 and isinstance(event.get("digest"), str)]
        digests = {event["digest"] for event in valid}
        return next(iter(digests)) if len(events) == 4 and len(valid) == 4 and len(digests) == 1 else None

    def _fresh_waiting(self) -> int | None:
        with self.telemetry.lock:
            sample = self.telemetry.latest.get(0, {})
            endpoint = (sample.get("sources") or {}).get("endpoint") or {}
            admission = (sample.get("sources") or {}).get("admission") or {}
            api = sample.get("api") or {}
            waiting = api.get("requests_waiting")
        if (endpoint.get("status") != "current"
                or not isinstance(endpoint.get("age_ns"), int)
                or endpoint["age_ns"] > 2_000_000_000
                or not isinstance(waiting, (int, float))):
            return None
        if getattr(self, "cfg", {}).get("bounded_admission") is True:
            admission_waiting = api.get("admission_requests_waiting")
            if (admission.get("status") != "current"
                    or not isinstance(admission.get("age_ns"), int)
                    or admission["age_ns"] > 2_000_000_000
                    or not isinstance(admission_waiting, (int, float))):
                return None
            return int(waiting) + int(admission_waiting)
        return int(waiting)

    def _fresh_admission_waiting(self) -> int | None:
        if getattr(self, "cfg", {}).get("bounded_admission") is not True:
            return None
        with self.telemetry.lock:
            sample = self.telemetry.latest.get(0, {})
            admission = (sample.get("sources") or {}).get("admission") or {}
            value = (sample.get("api") or {}).get("admission_requests_waiting")
        if (admission.get("status") != "current"
                or not isinstance(admission.get("age_ns"), int)
                or admission["age_ns"] > 2_000_000_000
                or not isinstance(value, (int, float))):
            return None
        return int(value)

    def _fresh_admission_active(self) -> int | None:
        if getattr(self, "cfg", {}).get("bounded_admission") is not True:
            return None
        with self.telemetry.lock:
            sample = self.telemetry.latest.get(0, {})
            admission = (sample.get("sources") or {}).get("admission") or {}
            value = (sample.get("api") or {}).get("admission_requests_active")
        if (admission.get("status") != "current"
                or not isinstance(admission.get("age_ns"), int)
                or admission["age_ns"] > 2_000_000_000
                or not isinstance(value, (int, float))):
            return None
        return int(value)

    def wait_for_api_idle(self, timeout: float = 30) -> tuple[str, dict[str, Any]]:
        deadline = min(self.active_deadline, time.monotonic() + timeout)
        last: dict[str, Any] = {}
        while time.monotonic() < deadline:
            with self.telemetry.lock:
                sample = self.telemetry.latest.get(0, {})
                source = (sample.get("sources") or {}).get("endpoint") or {}
                admission_source = (sample.get("sources") or {}).get("admission") or {}
                api = sample.get("api") or {}
                connections = sample.get("connections") or {}
            last = {"source": source, "admission_source": admission_source,
                    "api": api, "connections": connections}
            fresh = (source.get("status") == "current"
                     and isinstance(source.get("age_ns"), int)
                     and source["age_ns"] <= 2_000_000_000)
            counters = [api.get("requests_running"), api.get("requests_waiting"),
                        connections.get("established")]
            if getattr(self, "cfg", {}).get("bounded_admission") is True:
                fresh = (fresh and admission_source.get("status") == "current"
                         and isinstance(admission_source.get("age_ns"), int)
                         and admission_source["age_ns"] <= 2_000_000_000)
                counters.extend((api.get("admission_requests_active"),
                                 api.get("admission_requests_waiting"),
                                 api.get("admission_requests_inflight")))
            if fresh and all(isinstance(value, (int, float)) for value in counters):
                if all(value == 0 for value in counters):
                    return "pass", last
            time.sleep(0.1)
        known_counters = [
            (last.get("api") or {}).get("requests_running"),
            (last.get("api") or {}).get("requests_waiting"),
            (last.get("connections") or {}).get("established")]
        if getattr(self, "cfg", {}).get("bounded_admission") is True:
            known_counters.extend((
                (last.get("api") or {}).get("admission_requests_active"),
                (last.get("api") or {}).get("admission_requests_waiting"),
                (last.get("api") or {}).get("admission_requests_inflight")))
        known_busy = all(isinstance(value, (int, float)) for value in known_counters)
        if time.monotonic() >= self.active_deadline and not self.telemetry.safety_event.is_set():
            return "pending", {**last, "reason": "phase deadline cut short the idle proof"}
        return ("fail" if known_busy else "pending",
                {**last, "reason": "API requests/connections did not prove idle"})

    def _context(self, case: dict[str, Any], repetition: int) -> tuple[str, dict[str, Any]]:
        p = case["parameters"]
        prefix = f"campaign {self.cfg['campaign_id']} case {case['id']} repetition {repetition}."
        expected_completion_tokens = p.get("max_tokens", 16)
        pressure_case = p.get("long_decode_pressure") is True
        evidence: dict[str, Any] = {
            "token_count": p["tokens"],
            "completion_token_count": expected_completion_tokens,
        }
        if p["cache"] == "cold":
            prompts = [self.http.exact_prompt(p["tokens"], f"{prefix} client {index}.")
                       for index in range(p["concurrency"])]
        else:
            self._set_case(case["id"])
            prompt = self.http.exact_prompt(p["tokens"], prefix)
            publication_cursors = [self._cursor(rank) for rank in range(4)]
            seed = self.http.chat(prompt, request_id=f"{case['id']}-seed")
            evidence["seed"] = seed
            seed_counts = self._prompt_token_counts([seed])
            evidence["seed_prompt_tokens"] = seed_counts[0]
            if not _valid_fixed_completion(seed, p["tokens"]):
                return "fail", evidence
            publications, errors = self._events_all(
                publication_cursors, case["id"], "stage_end", stage="publication")
            evidence["seed_publications"], evidence["seed_publication_errors"] = publications, errors
            digest = self._publication_digest(publications) if not errors else None
            if digest is None:
                return "pending", {**evidence,
                    "reason": "seeded four-rank SparkCache publication was not proven committed"}
            evidence["seed_digest"] = digest
            restore_cursors = [self._cursor(rank) for rank in range(4)]
            prompts = [prompt] * p["concurrency"]
        if pressure_case:
            request_options = {
                "max_tokens": expected_completion_tokens,
                "min_tokens": p["min_tokens"],
                "ignore_eos": p["ignore_eos"],
            }
            prompt_hashes = [hashlib.sha256(prompt.encode()).hexdigest()
                             for prompt in prompts]
            evidence.update(
                total_token_budget=p["tokens"] + expected_completion_tokens,
                distinct_prompt_sha256=prompt_hashes,
                request_parameters=dict(request_options),
            )
            if len(set(prompt_hashes)) != p["concurrency"]:
                return "fail", {**evidence,
                    "reason": "long-decode pressure prompts were not distinct"}
            previous_deadline = self.active_deadline
            request_started = time.monotonic()
            request_deadline = min(
                previous_deadline,
                request_started + LONG_DECODE_REQUEST_LIMIT_SECONDS)
            evidence["request_deadline"] = {
                "limit_seconds": LONG_DECODE_REQUEST_LIMIT_SECONDS,
                "started_monotonic_ns": int(request_started * 1_000_000_000),
                "deadline_monotonic_ns": int(request_deadline * 1_000_000_000),
                "limited_by": ("case_local_limit"
                               if request_deadline < previous_deadline
                               else "campaign_phase_deadline"),
            }
            self.active_deadline = request_deadline
            try:
                results = self._request_group(
                    prompts, case["id"], request_options=request_options)
            finally:
                self.active_deadline = previous_deadline
        else:
            results = self._request_group(prompts, case["id"])
        counts = self._prompt_token_counts(results)
        evidence.update(responses=results, prompt_tokens=counts)
        request_errors = [result for result in results if result.get("error")]
        completed = [result for result in results if not result.get("error")]
        planned_cutoff = (
            bool(request_errors) and len(results) == len(prompts)
            and all(result.get("guard_abort_cause") == "phase_deadline"
                    for result in request_errors)
            and all(_valid_fixed_completion(result, p["tokens"],
                                            expected_completion_tokens)
                    for result in completed)
            and time.monotonic() >= self.active_deadline
            and not self.telemetry.safety_event.is_set())
        if planned_cutoff:
            evidence["reason"] = "phase deadline interrupted the in-flight context group"
            evidence["planned_cutoff"] = True
            return "pending", evidence
        if (len(results) != len(prompts)
                or not all(_valid_fixed_completion(result, p["tokens"],
                                                   expected_completion_tokens)
                           for result in results)):
            return "fail", evidence
        if pressure_case:
            started = [result.get("started_ns") for result in results]
            finished = [result.get("finished_ns") for result in results]
            overlap = (all(isinstance(value, int) for value in [*started, *finished])
                       and max(started) <= min(finished))
            evidence["all_request_intervals_overlap"] = overlap
            if not overlap:
                return "fail", {**evidence,
                    "reason": "five concurrent request intervals did not overlap"}
        if p["cache"] == "replay":
            restores, errors = self._events_all(
                restore_cursors, case["id"], "stage_end", stage="restore")
            evidence["restore_events"], evidence["restore_errors"] = restores, errors
            restored = [event for event in restores
                        if event.get("outcome") == "ok" and event.get("digest") == evidence["seed_digest"]]
            if len(restores) != 4 or len(restored) != 4 or errors:
                return "pending", {**evidence,
                    "reason": "fresh four-rank SparkCache restore was not proven"}
        return "pass", evidence

    def _queue(self, case: dict[str, Any], repetition: int) -> tuple[str, dict[str, Any]]:
        clients = case["parameters"]["clients"]
        bounded_admission = getattr(self, "cfg", {}).get("bounded_admission") is True
        admission_overflow = case["parameters"].get("admission_overflow") is True
        long_tokens = (32_000 if admission_overflow
                       else min(32_000 * (2 ** repetition), 128_000))
        long_prompt = self.http.exact_prompt(long_tokens, f"{case['id']} long {repetition}.")
        short_prompt = self.http.exact_prompt(512, f"{case['id']} short {repetition}.")
        prompts = ([short_prompt] * clients if admission_overflow else
                   [long_prompt if index % 3 == 0 else short_prompt
                    for index in range(clients)])
        blockers = [CancelableRequest(
            self.http, self.http.chat_payload(long_prompt, stream=True, max_tokens=2048),
            f"{case['id']}-blocker-{index}") for index in range(6)]
        queue_observed = False
        admission_queue_observed = False
        started_blockers: list[CancelableRequest] = []
        blocker_start_errors: dict[str, str] = {}
        blocker_pre_cancel: dict[str, dict[str, Any]] = {}
        results_by_index: dict[int, dict[str, Any]] = {}
        results_lock = threading.Lock()
        futures: list[tuple[int, str, Any]] = []
        pool: ThreadPoolExecutor | None = None
        capacity_stop = threading.Event()
        capacity_reason: list[str | None] = [None]
        unknown_submission_indices: set[int] = set()
        blocker_activation: dict[str, Any] = {
            "required_active": len(blockers), "established": not bounded_admission,
            "last_active": None,
        }

        def store(index: int, result: dict[str, Any]) -> None:
            with results_lock:
                results_by_index[index] = result

        def store_if_absent(index: int, result: dict[str, Any]) -> None:
            with results_lock:
                results_by_index.setdefault(index, result)

        def unsubmitted(index: int, request_id: str, error: BaseException | str,
                        *, submitted: bool | None = False,
                        dispatch_state: str = "not_dispatched",
                        error_origin: str = "client_capacity") -> None:
            stamp = time.time_ns()
            detail = error if isinstance(error, str) else f"{type(error).__name__}: {error}"
            result = {
                "request_id": request_id, "submitted": submitted,
                "dispatch_state": dispatch_state, "status": None,
                "started_ns": None, "finished_ns": stamp,
                "error": f"{error_origin}_error: {detail}",
                "error_origin": error_origin, "guard_abort_cause": None,
            }
            if submitted is None:
                store_if_absent(index, result)
            else:
                store(index, result)

        def run_client(index: int, prompt: str, request_id: str) -> dict[str, Any]:
            if capacity_stop.is_set():
                result = {
                    "request_id": request_id, "status": None,
                    "started_ns": None, "finished_ns": time.time_ns(),
                    "error": ("client_capacity_error: request entered the executor "
                              "after submission capacity was exhausted"),
                    "error_origin": "client_capacity", "guard_abort_cause": None,
                    "dispatch_state": "entered_not_sent",
                }
                store(index, result)
                return result
            try:
                result = dict(self._captured_chat(prompt, request_id))
            except BaseException as error:
                result = {
                    "request_id": request_id, "status": None,
                    "started_ns": None, "finished_ns": time.time_ns(),
                    "error": f"client_callable_error: {type(error).__name__}: {error}",
                    "error_origin": "client_callable",
                    "guard_abort_cause": getattr(error, "guard_abort_cause", None),
                }
                result["dispatch_state"] = "entered"
                store(index, result)
                raise
            result["dispatch_state"] = "entered"
            store(index, result)
            return result

        try:
            for blocker_index, blocker in enumerate(blockers):
                try:
                    blocker.start()
                    started_blockers.append(blocker)
                except Exception as error:
                    blocker_start_errors[blocker.request_id] = (
                        f"blocker_start_error: {type(error).__name__}: {error}")
                    if blocker.thread.is_alive():
                        started_blockers.append(blocker)
                    for later in blockers[blocker_index + 1:]:
                        blocker_start_errors[later.request_id] = (
                            "blocker_start_error: not started after an earlier blocker failure")
                    for index in range(clients):
                        unsubmitted(index, f"{case['id']}-wave-{index}",
                                    "occupying blocker setup failed")
                    break
            if not blocker_start_errors and bounded_admission:
                activation_deadline = min(self.active_deadline, time.monotonic() + 30)
                blocker_activation["started_monotonic"] = time.monotonic()
                blocker_activation["deadline_monotonic"] = activation_deadline
                while time.monotonic() < activation_deadline:
                    if self.telemetry.safety_event.is_set():
                        break
                    active = self._fresh_admission_active()
                    blocker_activation["last_active"] = active
                    if active is not None and active >= len(blockers):
                        blocker_activation["established"] = True
                        blocker_activation["established_monotonic"] = time.monotonic()
                        break
                    time.sleep(0.05)
                if not blocker_activation["established"]:
                    blocker_activation["reason"] = (
                        "fresh bounded-admission active gauge did not prove all blockers")
                    for index in range(clients):
                        unsubmitted(
                            index, f"{case['id']}-wave-{index}",
                            blocker_activation["reason"], error_origin="precondition")
            if not blocker_start_errors and blocker_activation["established"]:
                try:
                    pool = ThreadPoolExecutor(max_workers=clients)
                except Exception as error:
                    for index in range(clients):
                        unsubmitted(index, f"{case['id']}-wave-{index}", error)
                if pool is not None:
                    for index, prompt in enumerate(prompts):
                        request_id = f"{case['id']}-wave-{index}"
                        try:
                            future = pool.submit(run_client, index, prompt, request_id)
                        except Exception as error:
                            capacity_reason[0] = f"{type(error).__name__}: {error}"
                            capacity_stop.set()
                            for _, _, pending in futures:
                                pending.cancel()
                            unknown_submission_indices.add(index)
                            unsubmitted(index, request_id, error, submitted=None,
                                        dispatch_state="submission_unknown")
                            for unsubmitted_index in range(index + 1, clients):
                                unsubmitted(unsubmitted_index,
                                            f"{case['id']}-wave-{unsubmitted_index}", error)
                            break
                        futures.append((index, request_id, future))
                while any(not future.done() for _, _, future in futures):
                    if self.telemetry.safety_event.is_set():
                        break
                    waiting = self._fresh_waiting()
                    queue_observed = queue_observed or waiting is not None and waiting > 0
                    admission_waiting = self._fresh_admission_waiting()
                    admission_queue_observed = (
                        admission_queue_observed
                        or admission_waiting is not None and admission_waiting > 0)
                    time.sleep(0.05)
        finally:
            for blocker in started_blockers:
                blocker_pre_cancel[blocker.request_id] = {
                    "status": blocker.http_status,
                    "error": blocker.error,
                    "guard_abort_cause": getattr(blocker, "guard_abort_cause", None),
                    "finished_ns": blocker.finished_ns,
                    "thread_alive": blocker.thread.is_alive(),
                }
                blocker.cancel()
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=capacity_stop.is_set())
            for blocker in started_blockers:
                blocker.thread.join(timeout=10)
        for index, request_id, future in futures:
            with results_lock:
                result = results_by_index.get(index)
            if result is None:
                if future.cancelled():
                    unsubmitted(index, request_id,
                                capacity_reason[0] or "executor cancelled pending work",
                                submitted=True, dispatch_state="not_dispatched")
                    continue
                try:
                    result = dict(future.result())
                except BaseException as error:
                    result = {
                        "request_id": request_id, "status": None,
                        "started_ns": None, "finished_ns": time.time_ns(),
                        "error": f"client_future_error: {type(error).__name__}: {error}",
                        "error_origin": "client_future", "guard_abort_cause": None,
                        "dispatch_state": "entered_unknown",
                    }
            result["submitted"] = True
            store(index, result)
        with results_lock:
            for index in unknown_submission_indices:
                results_by_index[index]["submitted"] = None
        results = [results_by_index[index] for index in range(clients)]
        blocker_receipts = [{
            "request_id": blocker.request_id,
            "started": blocker in started_blockers,
            "status": blocker.http_status,
            "headers": blocker.response_headers,
            "result": blocker.result,
            "error": blocker_start_errors.get(blocker.request_id) or blocker.error,
            "error_origin": ("blocker_start" if blocker.request_id in blocker_start_errors
                             else "client_capacity" if getattr(
                                 blocker, "client_capacity_error", False)
                             else "client_request" if (blocker_pre_cancel.get(
                                 blocker.request_id) or {}).get("error") else None),
            "pre_cancel": blocker_pre_cancel.get(blocker.request_id),
            "started_ns": blocker.started_ns,
            "first_byte_ns": blocker.first_byte_ns,
            "finished_ns": blocker.finished_ns,
            "first_byte_hex": blocker.first_byte_hex,
            "terminated": (not blocker.thread.is_alive()
                           if blocker in started_blockers else True),
        } for blocker in blockers]
        client_errors = [result for result in results if result.get("error")]
        phase_aborts = [result for result in client_errors
                        if result.get("guard_abort_cause") == "phase_deadline"]
        client_capacity_errors = [result for result in client_errors
                                  if result.get("error_origin") == "client_capacity"]
        service_client_errors = [result for result in client_errors
                                 if result not in client_capacity_errors]
        completed_client_responses = [result for result in results
                                      if not result.get("error")]
        intentional_overloads = ([result for result in completed_client_responses
                                  if _intentional_overload_code(result) is not None]
                                 if bounded_admission else [])
        accepted_responses = [result for result in completed_client_responses
                              if result.get("status") == 200]
        invalid_completed_responses = [
            result for result in completed_client_responses
            if result not in accepted_responses and result not in intentional_overloads]
        completed_responses_valid = (
            not invalid_completed_responses and self._valid(accepted_responses))
        if admission_overflow:
            completed_responses_valid = (
                completed_responses_valid
                and all(_valid_fixed_completion(result, 512)
                        for result in accepted_responses))
        blocker_capacity_errors = [row for row in blocker_receipts
                                   if row["error_origin"] in {
                                       "blocker_start", "client_capacity"}]
        blocker_failures = [row for row in blocker_receipts
                            if (row["error_origin"] == "client_request"
                                and (row.get("pre_cancel") or {}).get(
                                    "guard_abort_cause") != "phase_deadline")
                            or (row["error_origin"] is None
                                and isinstance((row.get("pre_cancel") or {}).get(
                                    "status"), int)
                                and row["pre_cancel"]["status"] != 200)]
        blocker_phase_aborts = [row for row in blocker_receipts
                                if (row.get("pre_cancel") or {}).get(
                                    "guard_abort_cause") == "phase_deadline"]
        response_counts = {
            "requested": clients,
            "submitted": sum(result.get("submitted") is True for result in results),
            "recorded": len(results),
            "http_200": sum(result.get("status") == 200 for result in results),
            "errors": len(client_errors),
            "guard_aborts": sum(result.get("guard_abort_cause") is not None
                                for result in results),
            "client_capacity_errors": len(client_capacity_errors),
            "submission_unknown": sum(result.get("submitted") is None
                                        for result in results),
            "blocker_failures_before_cancel": len(blocker_failures),
            "blocker_capacity_errors": len(blocker_capacity_errors),
            "blocker_phase_aborts": len(blocker_phase_aborts),
        }
        if bounded_admission:
            accounted = (len(accepted_responses) + len(intentional_overloads)
                         + len(invalid_completed_responses) + len(client_errors))
            response_counts["accepted"] = len(accepted_responses)
            response_counts["intentional_overload_rejections"] = len(
                intentional_overloads)
            response_counts["invalid_completed"] = len(invalid_completed_responses)
            response_counts["accounted"] = accounted
            response_counts["accounting_conserved"] = accounted == clients
            response_counts["admission_queue_full"] = sum(
                _intentional_overload_code(result) == "admission_queue_full"
                for result in intentional_overloads)
            response_counts["admission_queue_timeout"] = sum(
                _intentional_overload_code(result) == "admission_queue_timeout"
                for result in intentional_overloads)
        evidence = {"clients": clients, "long_tokens": long_tokens, "responses": results,
                    "queue_observed": queue_observed,
                    "admission_queue_observed": admission_queue_observed,
                    "blocker_activation": blocker_activation,
                    "response_counts": response_counts, "blockers": blocker_receipts,
                    "blockers_terminated": all(row["terminated"] for row in blocker_receipts)}
        idle_status, idle = self.wait_for_api_idle()
        evidence["idle"] = {"status": idle_status, "evidence": idle}
        capacity_observed = bool(client_capacity_errors or blocker_capacity_errors)
        evidence["health_200"] = None
        if capacity_observed or bounded_admission:
            try:
                evidence["health_200"] = self.health_code() == 200
            except Exception as error:
                evidence["health_error"] = f"{type(error).__name__}: {error}"
        if len(results) != clients or not evidence["blockers_terminated"]:
            return "fail", evidence
        if (bounded_admission and not blocker_activation["established"]
                and not blocker_start_errors):
            if (self.telemetry.safety_event.is_set() or blocker_failures
                    or idle_status == "fail" or evidence["health_200"] is not True):
                return "fail", evidence
            evidence["reason"] = blocker_activation["reason"]
            return "pending", evidence
        if bounded_admission:
            evidence["intentional_overloads"] = [
                {"request_id": result.get("request_id"),
                 "code": _intentional_overload_code(result),
                 "status": result.get("status"), "headers": result.get("headers"),
                 "body": result.get("body")}
                for result in intentional_overloads]
            if not completed_responses_valid:
                return "fail", evidence
            if not response_counts["accounting_conserved"]:
                return "fail", evidence
            if intentional_overloads and clients <= ADMISSION_MAX_QUEUED:
                evidence["reason"] = (
                    "bounded admission rejected a wave that fits its configured queue")
                return "fail", evidence
            if evidence["health_200"] is not True:
                return "fail", evidence
        planned_cutoff = (bool(phase_aborts or blocker_phase_aborts)
                          and all(result in phase_aborts
                                  or result in client_capacity_errors
                                  for result in client_errors)
                          and not blocker_failures
                          and completed_responses_valid
                          and time.monotonic() >= self.active_deadline
                          and not self.telemetry.safety_event.is_set())
        pure_client_capacity = (
            capacity_observed
            and not service_client_errors and not blocker_failures
            and not blocker_phase_aborts and idle_status == "pass"
            and evidence["health_200"] is True
            and completed_responses_valid
            and not self.telemetry.safety_event.is_set())
        if pure_client_capacity:
            evidence["client_capacity_limited"] = True
            evidence["reason"] = (
                "local client thread or descriptor capacity prevented the full wave; "
                "no service admission limit was inferred")
            return "pending", evidence
        if client_errors or blocker_failures or blocker_phase_aborts:
            return ("pending" if planned_cutoff else "fail"), evidence
        if bounded_admission:
            if admission_overflow:
                if not admission_queue_observed:
                    evidence["reason"] = (
                        "the bounded-admission frontend queue was not freshly observed")
                    return "pending", evidence
                if response_counts["admission_queue_full"] < 1:
                    evidence["reason"] = (
                        "the bounded-admission hard queue bound produced no explicit full rejection")
                    return "pending", evidence
        elif not self._valid(results):
            return "fail", evidence
        if not queue_observed:
            return "pending", evidence
        return idle_status, evidence

    def _api(self, case: dict[str, Any], repetition: int) -> tuple[str, dict[str, Any]]:
        variant = case["parameters"]["variant"]
        if variant == "malformed-json":
            result = self.http.raw(f"{self.http.base}/chat/completions", b'{"model":')
        elif variant == "partial-upload":
            sock = socket.create_connection((self.http.host, self.http.port), timeout=10)
            payload = b'{"model":"' + self.http.model.encode()
            headers = (f"POST {self.http.base}/chat/completions HTTP/1.1\r\nHost: {self.http.host}\r\n"
                       "Content-Type: application/json\r\nContent-Length: 100000\r\n\r\n").encode()
            sock.sendall(headers + payload)
            sock.close()
            result = {"status": "disconnected", "sent_bytes": len(payload)}
        elif variant == "invalid-parameters":
            result = self.http.json(f"{self.http.base}/chat/completions",
                                    {"model": self.http.model, "max_tokens": -1, "messages": []})
        else:
            prompt = self.http.exact_prompt(262_145, f"{case['id']} {repetition}.")
            result = self.http.chat(prompt, request_id=case["id"])
        healthy = self.health_code() == 200
        rejected = result.get("status") == "disconnected" or isinstance(result.get("status"), int) and 400 <= result["status"] < 500
        return ("pass" if rejected and healthy else "fail", {"response": result, "health_200": healthy})

    def health_code(self) -> int | None:
        connection = http.client.HTTPConnection(self.http.host, self.http.port, timeout=10)
        try:
            connection.request("GET", "/health")
            return connection.getresponse().status
        except OSError:
            return None
        finally:
            connection.close()

    def _slow(self, case: dict[str, Any], repetition: int) -> tuple[str, dict[str, Any]]:
        behavior = case["parameters"]["behavior"]
        prompt = self.http.exact_prompt(8_000, f"{case['id']} {repetition}.")
        connection = http.client.HTTPConnection(self.http.host, self.http.port, timeout=120)
        body = json.dumps(self.http.chat_payload(prompt, stream=True, max_tokens=256)).encode()
        connection.request("POST", f"{self.http.base}/chat/completions", body=body,
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        received = 0
        if behavior == "slow-read":
            for _ in range(8):
                received += len(response.read(1))
                time.sleep(0.25)
        elif behavior == "suspended-read":
            time.sleep(2)
        connection.close()
        time.sleep(1)
        healthy = self.health_code() == 200
        idle_status, idle = self.wait_for_api_idle()
        return (idle_status if healthy else "fail",
                {"received_bytes": received, "health_200": healthy, "idle": idle})

    def _ssh(self, rank: int, *command: str, timeout: float = 120,
             deadline: float | None = None) -> subprocess.CompletedProcess[str]:
        has_campaign_deadline = deadline is not None or hasattr(self, "active_deadline")
        if deadline is not None:
            command_deadline = deadline
        elif hasattr(self, "active_deadline"):
            command_deadline = self.active_deadline
        else:
            command_deadline = time.monotonic() + timeout
        remaining = command_deadline - time.monotonic()
        if remaining <= 0:
            return subprocess.CompletedProcess(command, 124, "", "campaign phase deadline reached")
        effective_timeout = min(timeout, remaining)
        deadline_limited = has_campaign_deadline and remaining <= timeout
        try:
            return subprocess.run([*SSH, self.cfg["nodes"][rank], shlex.join(command)], text=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  timeout=effective_timeout, check=False)
        except subprocess.TimeoutExpired as error:
            phase_deadline_reached = (
                deadline_limited and time.monotonic() >= command_deadline)
            return subprocess.CompletedProcess(command, 124,
                error.stdout if isinstance(error.stdout, str) else "",
                ("SSH command timed out at campaign phase deadline"
                 if phase_deadline_reached else
                 f"SSH command timed out after {effective_timeout:.3f} seconds"))

    def _faultctl(self, rank: int, arguments: list[str], timeout: float = 120,
                  deadline: float | None = None) -> dict[str, Any]:
        remote = f"tp4/scripts/resilience/.campaign/{self.cfg['campaign_id']}/faultctl.py"
        command = ["sudo", "-n", "python3", remote, "--root", self.cfg["host_cache_roots"][rank],
                   "--campaign-id", self.cfg["campaign_id"], *arguments]
        completed = self._ssh(rank, *command, timeout=timeout, deadline=deadline)
        if completed.returncode:
            if completed.returncode == 124 and "phase deadline" in completed.stderr:
                raise PhaseDeadlineError(
                    f"rank {rank} faultctl reached the campaign phase deadline")
            raise RuntimeError(f"rank {rank} faultctl failed: {completed.stderr.strip()}")
        return json.loads(completed.stdout)

    def _drain_barriers(self) -> dict[int, int]:
        barriers: dict[int, int] = {}
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(self._faultctl, rank, ["barrier"]): rank for rank in range(4)}
            for future in as_completed(futures):
                rank = futures[future]
                value = future.result()
                barrier = value.get("monotonic_ns")
                if not isinstance(barrier, int):
                    raise RuntimeError(f"rank {rank} returned an invalid drain barrier")
                barriers[rank] = barrier
        return barriers

    def _reset_faults(self, case_id: str) -> dict[str, Any]:
        """Best-effort all-rank cleanup with a deadline independent of the case."""
        started = time.monotonic()
        deadline = min(self.soak_at, started + 60.0)
        evidence: dict[str, Any] = {"case_id": case_id, "ranks": {}, "errors": []}

        def reset_rank(rank: int) -> tuple[int, dict[str, Any]]:
            row: dict[str, Any] = {}
            try:
                row["control"] = self._faultctl(rank, [
                    "set", "--mode", "off", "--case-id", case_id, "--remaining", "0"],
                    timeout=60, deadline=deadline)
            except Exception as error:
                row["control_error"] = f"{type(error).__name__}: {error}"
            try:
                row["restore"] = self._faultctl(
                    rank, ["restore"], timeout=60, deadline=deadline)
            except Exception as error:
                row["restore_error"] = f"{type(error).__name__}: {error}"
            return rank, row

        if deadline <= started:
            evidence["errors"].append("cleanup deadline is exhausted")
        else:
            with ThreadPoolExecutor(max_workers=4) as pool:
                for rank, row in pool.map(reset_rank, range(4)):
                    evidence["ranks"][rank] = row
                    for key in ("control_error", "restore_error"):
                        if key in row:
                            evidence["errors"].append(f"rank {rank} {key}: {row[key]}")
        evidence["elapsed_seconds"] = time.monotonic() - started
        return evidence

    def preserve_cluster_evidence(self, label: str) -> dict[str, Any]:
        """Stream current inspect/log evidence to private files before recovery."""
        safe_label = "".join(char if char.isalnum() or char in "-." else "-"
                             for char in label)[:80] or "incident"
        capture_id = f"{time.time_ns()}-{safe_label}"
        directory = require_private_dir(self.output / "evidence" / capture_id, create=True)
        receipt: dict[str, Any] = {"capture_id": capture_id, "label": label,
                                   "started_ns": time.time_ns(), "artifacts": []}
        self.state.setdefault("evidence_captures", []).append(receipt)
        append_event(self.events, {"kind": "evidence_capture_started", "capture_id": capture_id,
                                   "label": label})
        self.save()

        budget = min(10.0, max(0.0, self.end_monotonic - time.monotonic()))
        if budget <= 0:
            receipt.update(finished_ns=time.time_ns(), status="deadline")
            self.save()
            return receipt
        deadline = time.monotonic() + budget
        running: list[tuple[subprocess.Popen[bytes], Any, Any, dict[str, Any]]] = []
        for rank, host in enumerate(self.cfg["nodes"]):
            for kind, docker_args in (
                ("inspect", ["inspect", self.cfg["container"]]),
                ("logs", ["logs", "--timestamps", self.cfg["container"]]),
            ):
                stdout_path = directory / f"rank{rank}-{kind}.out"
                stderr_path = directory / f"rank{rank}-{kind}.err"
                artifact = {"rank": rank, "kind": kind,
                            "stdout": str(stdout_path), "stderr": str(stderr_path)}
                receipt["artifacts"].append(artifact)
                stdout = os.fdopen(os.open(stdout_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), "wb")
                stderr = os.fdopen(os.open(stderr_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), "wb")
                remote_command = shlex.join(["sudo", "-n", "docker", *docker_args])
                try:
                    process = subprocess.Popen([*SSH, host, remote_command], stdout=stdout,
                                               stderr=stderr, start_new_session=True)
                    running.append((process, stdout, stderr, artifact))
                except OSError as error:
                    stdout.close()
                    stderr.close()
                    artifact.update(returncode=None, timed_out=False,
                                    launch_error=f"{type(error).__name__}: {error}")

        while time.monotonic() < deadline and any(item[0].poll() is None for item in running):
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        for process, stdout, stderr, artifact in running:
            timed_out = process.poll() is None
            if timed_out:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=0.5)
            finally:
                stdout.close()
                stderr.close()
            artifact.update(returncode=process.returncode, timed_out=timed_out,
                            stdout_bytes=Path(artifact["stdout"]).stat().st_size,
                            stderr_bytes=Path(artifact["stderr"]).stat().st_size)
        complete = (len(receipt["artifacts"]) == 8
                    and all(item.get("returncode") == 0 and not item.get("timed_out")
                            for item in receipt["artifacts"]))
        receipt.update(finished_ns=time.time_ns(), status="captured" if complete else "incomplete")
        append_event(self.events, {"kind": "evidence_capture_finished", "capture_id": capture_id})
        self.save()
        return receipt

    def _cursor(self, rank: int) -> str:
        value = self._faultctl(rank, ["cursor"])
        return f"{value['device']}:{value['inode']}:{value['position']}"

    def _event(self, rank: int, cursor: str, case_id: str, kind: str, *,
               stage: str | None = None, mode: str | None = None,
               timeout: float = 30) -> dict[str, Any]:
        arguments = ["event", "--cursor", cursor, "--case-id", case_id,
                     "--kind", kind, "--timeout", str(timeout)]
        if stage:
            arguments.extend(("--stage", stage))
        if mode:
            arguments.extend(("--mode", mode))
        return self._faultctl(rank, arguments, timeout=timeout + 10)

    def _events_all(self, cursors: list[str], case_id: str, kind: str, *,
                    stage: str | None = None, mode: str | None = None,
                    timeout: float = 30) -> tuple[list[dict[str, Any]], list[str]]:
        found: list[dict[str, Any]] = []
        errors: list[str] = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(self._event, rank, cursors[rank], case_id, kind,
                                   stage=stage, mode=mode, timeout=timeout): rank
                       for rank in range(4)}
            for future, rank in ((future, futures[future]) for future in futures):
                try:
                    found.append(future.result())
                except Exception as error:
                    errors.append(f"rank {rank}: {type(error).__name__}: {error}")
        return found, errors

    def _set_case(self, case_id: str, *, pause_stage: str | None = None,
                  pause_ranks: set[int] | None = None) -> None:
        for rank in range(4):
            arguments = ["set", "--mode", "off", "--case-id", case_id,
                         "--remaining", "0"]
            if pause_stage and (pause_ranks is None or rank in pause_ranks):
                arguments.extend(("--pause-stage", pause_stage))
            self._faultctl(rank, arguments)

    def _cache(self, case: dict[str, Any], repetition: int) -> tuple[str, dict[str, Any]]:
        result: tuple[str, dict[str, Any]] | None = None
        evidence: dict[str, Any] = {}
        try:
            result = self._cache_action(case, repetition)
            evidence = result[1]
        finally:
            try:
                evidence["cleanup"] = self._reset_faults(case["id"])
                evidence["cleanup_errors"] = evidence["cleanup"]["errors"]
            except Exception as error:
                evidence["cleanup_errors"] = [f"{type(error).__name__}: {error}"]
        if evidence.get("cleanup_errors"):
            return "fail", evidence
        assert result is not None
        return result

    def _cache_action(self, case: dict[str, Any], repetition: int) -> tuple[str, dict[str, Any]]:
        variant = case["parameters"]["variant"]
        evidence: dict[str, Any] = {"actions": []}
        if variant == "shared-prefix":
            evidence["protocol_version"] = TARGETED_CASE_PROTOCOL_VERSION
        action_error: str | None = None
        try:
            self._set_case(case["id"])
            prompt = self.http.exact_prompt(32_000, f"{case['id']} {repetition}.")
            store_cursors = [self._cursor(rank) for rank in range(4)]
            baseline = self.http.chat(prompt, request_id=f"{case['id']}-store")
            evidence["store"] = baseline
            publications, publication_errors = self._events_all(
                store_cursors, case["id"], "stage_end", stage="publication")
            evidence["store_publications"] = publications
            evidence["store_publication_errors"] = publication_errors
            if (variant == "shared-prefix"
                    and not _valid_fixed_completion(baseline, 32_000)):
                evidence["reason"] = "shared-prefix seed response failed its exact response contract"
                return "fail", evidence
            if baseline.get("status") != 200:
                return "fail", evidence
            digest = self._publication_digest(publications) if not publication_errors else None
            if digest is None:
                evidence["reason"] = "exact four-rank committed snapshot publication was not proven"
                return "pending", evidence
            if variant == "manifest-growth":
                evidence["manifests_before_growth"] = [
                    self._faultctl(rank, ["status"]).get("manifest_count") for rank in range(4)]
            action_cursors = [self._cursor(rank) for rank in range(4)]
            if variant in {"checksum", "missing", "truncated"}:
                mutation = variant
                for rank in range(4):
                    evidence["actions"].append(self._faultctl(rank, [
                        "mutate", "--mutation", mutation, "--kind", "chunk",
                        "--index", "0", "--digest", digest]))
            elif variant in {"enospc", "eio", "delay", "publication-interrupted"}:
                mode = {"publication-interrupted": "publication_eio"}.get(variant, variant)
                operation = "write" if variant in {"enospc", "publication-interrupted"} else "read"
                for rank in range(4):
                    self._faultctl(rank, ["set", "--mode", mode, "--operation", operation,
                                          "--case-id", case["id"], "--remaining", "1",
                                          "--delay-ms", "500" if mode == "delay" else "0"])
            if variant in {"enospc", "publication-interrupted"}:
                request_prompt = self.http.exact_prompt(32_000,
                    f"{case['id']} injected-write {repetition} unique.")
            elif variant == "distinct-snapshots":
                request_prompt = self.http.exact_prompt(32_000,
                    f"{case['id']} distinct {repetition}.")
            elif variant == "shared-prefix":
                request_prompt = self.http.exact_prompt(
                    34_304, prompt + " shared extension")
                evidence["shared_prefix"] = {
                    "seed_prompt_tokens": 32_000,
                    "extended_prompt_tokens": 34_304,
                    "literal_prefix_preserved": request_prompt.startswith(prompt),
                    "seed_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                }
                if not evidence["shared_prefix"]["literal_prefix_preserved"]:
                    evidence["reason"] = "shared-prefix synthesis did not preserve the seed literal"
                    return "fail", evidence
            else:
                request_prompt = prompt
            replay = self.http.chat(request_prompt, request_id=f"{case['id']}-replay")
            evidence["replay"] = replay
            if variant == "shared-prefix" and not _valid_fixed_completion(replay, 34_304):
                evidence["reason"] = (
                    "shared-prefix extension response failed its exact response contract")
                return "fail", evidence
            if variant == "manifest-growth":
                results = [self.http.chat(self.http.exact_prompt(
                    8_000, f"{case['id']} {repetition} growth {i}."),
                    request_id=f"{case['id']}-growth-{i}") for i in range(8)]
                evidence["growth_requests"] = results
            if variant in {"enospc", "eio", "delay", "publication-interrupted"}:
                mode = {"publication-interrupted": "publication_eio"}.get(variant, variant)
                hits, errors = self._events_all(action_cursors, case["id"], "fault_hit", mode=mode)
                evidence["fault_hits"], evidence["fault_hit_errors"] = hits, errors
                stage = {"enospc": "capture", "eio": "restore", "delay": "restore",
                         "publication-interrupted": "publication"}[variant]
                stage_events, stage_errors = self._events_all(
                    action_cursors, case["id"], "stage_end", stage=stage)
                evidence["fault_stage"] = stage
                evidence["fault_stage_events"], evidence["fault_stage_errors"] = stage_events, stage_errors
            if variant in {"checksum", "missing", "truncated", "distinct-snapshots",
                           "shared-prefix", "manifest-growth", "eio"}:
                if variant == "shared-prefix":
                    restores, restore_errors = self._events_all(
                        action_cursors, case["id"], "stage_end", stage="restore")
                    evidence["shared_prefix_restores"] = restores
                    evidence["shared_prefix_restore_errors"] = restore_errors
                publications, errors = self._events_all(
                    action_cursors, case["id"], "stage_end", stage="publication")
                evidence["action_publications"], evidence["action_publication_errors"] = publications, errors
        except Exception as error:
            action_error = f"{type(error).__name__}: {error}"
            evidence["action_error"] = action_error
        if action_error is not None:
            if time.monotonic() >= self.targeted_stop_at and not self.telemetry.safety_event.is_set():
                return "pending", {**evidence, "reason": "targeted deadline interrupted cache case"}
            return "fail", evidence
        healthy = self.health_code() == 200
        evidence["health_200"] = healthy
        if variant in {"enospc", "eio", "delay", "publication-interrupted"}:
            if len(evidence.get("fault_hits", [])) != 4 or evidence.get("fault_hit_errors"):
                return "pending", {**evidence, "reason": "injected fault did not hit all four ranks"}
            stage_events = evidence.get("fault_stage_events", [])
            expected_outcome = "ok" if variant == "delay" else "error"
            matching = [event for event in stage_events if event.get("outcome") == expected_outcome]
            if (len(stage_events) != 4 or len(matching) != 4
                    or evidence.get("fault_stage_errors")):
                return "pending", {**evidence,
                    "reason": f"four-rank {evidence.get('fault_stage')} {expected_outcome} was not proven"}
        if variant in {"checksum", "missing", "truncated"}:
            if (evidence.get("action_publication_errors")
                    or self._publication_digest(evidence.get("action_publications", [])) is None):
                return "pending", {**evidence,
                    "reason": "fallback recompute and committed publication were not proven"}
        if variant == "eio":
            if (evidence.get("action_publication_errors")
                    or self._publication_digest(evidence.get("action_publications", [])) is None):
                return "pending", {**evidence,
                    "reason": "EIO fallback recompute and committed publication were not proven"}
        if variant == "shared-prefix":
            restores = evidence.get("shared_prefix_restores", [])
            if (evidence.get("shared_prefix_restore_errors") or len(restores) != 4
                    or any(event.get("outcome") != "ok"
                           or event.get("digest") != digest
                           or event.get("instrumentation_error")
                           for event in restores)):
                return "pending", {**evidence,
                    "reason": "fresh four-rank restore of the shared seed digest was not proven"}
        if variant in {"distinct-snapshots", "shared-prefix", "manifest-growth"}:
            action = evidence.get("action_publications", [])
            action_digest = self._publication_digest(action)
            if (evidence.get("action_publication_errors") or action_digest is None
                    or action_digest == digest):
                return "pending", {**evidence, "reason": "second four-rank snapshot publication was not proven"}
        if variant == "manifest-growth":
            evidence["manifests_after_growth"] = [
                self._faultctl(rank, ["status"]).get("manifest_count") for rank in range(4)]
            before = evidence["manifests_before_growth"]
            after = evidence["manifests_after_growth"]
            if not all(isinstance(old, int) and isinstance(new, int) and new > old
                       for old, new in zip(before, after)):
                return "pending", {**evidence, "reason": "manifest count growth was not proven on every rank"}
        valid_growth = all(result.get("status") == 200 for result in evidence.get("growth_requests", []))
        success = (self._valid([evidence.get("replay", {})]) and healthy and valid_growth
                   and not evidence.get("cleanup_errors"))
        return ("pass" if success else "fail", evidence)

    def _cancel(self, case: dict[str, Any], repetition: int) -> tuple[str, dict[str, Any]]:
        stage = case["parameters"]["stage"]
        evidence: dict[str, Any] = {"protocol_version": TARGETED_CASE_PROTOCOL_VERSION}
        if stage == "prefill":
            evidence["reason"] = "no exact prefill phase edge is exposed by the runtime"
            return "pending", evidence
        prompt_tokens = 32_000 if stage not in {"capture", "publication", "restore"} else 128_000
        prompt = self.http.exact_prompt(prompt_tokens, f"{case['id']} {repetition}.")
        if stage == "restore":
            store = self.http.chat(prompt, request_id=f"{case['id']}-seed")
            evidence["seed"] = store
            if not _valid_fixed_completion(store, prompt_tokens):
                evidence["reason"] = "restore seed failed its exact response contract"
                return "fail", evidence
        phase_stage = stage if stage in {"capture", "publication", "restore"} else None
        request = CancelableRequest(self.http, self.http.chat_payload(prompt, stream=True, max_tokens=256),
                                    f"{case['id']}-{repetition}")
        synchronized = False
        additional: list[CancelableRequest] = []
        blockers: list[CancelableRequest] = []
        cancellations: dict[int, dict[str, Any]] = {}

        def cancel_request(item: CancelableRequest) -> None:
            started_event = getattr(item, "started", None)
            result = getattr(item, "result", None)
            pre_cancel = {
                "http_status": getattr(item, "http_status", None),
                "error": getattr(item, "error", None),
                "finished_ns": getattr(item, "finished_ns", None),
                "result": dict(result) if isinstance(result, dict) else result,
            }
            cancellation = {
                "requested_ns": time.time_ns(),
                "started_before_cancel": (
                    started_event.is_set() if hasattr(started_event, "is_set")
                    else isinstance(getattr(item, "started_ns", None), int)),
                "thread_alive_before_cancel": item.thread.is_alive(),
                "finished_ns_before_cancel": getattr(item, "finished_ns", None),
                "pre_cancel": pre_cancel,
            }
            cancellation["cancel_call_ns"] = time.time_ns()
            try:
                item.cancel()
            except Exception as error:
                cancellation["error"] = f"{type(error).__name__}: {error}"
            cancellation["returned_ns"] = time.time_ns()
            cancellations[id(item)] = cancellation

        try:
            self._set_case(case["id"], pause_stage=phase_stage)
            if stage == "decode":
                request.start()
                deadline = min(self.active_deadline, time.monotonic() + 120)
                while time.monotonic() < deadline and request.thread.is_alive():
                    if request.first_byte.wait(
                            timeout=min(0.1, max(0, deadline - time.monotonic()))):
                        synchronized = True
                        break
            elif phase_stage:
                cursor = self._cursor(0)
                request.start()
                evidence["stage"] = self._event(
                    0, cursor, case["id"], "stage_waiting", stage=stage, timeout=120)
                synchronized = evidence["stage"].get("kind") == "stage_waiting"
            elif stage == "queue":
                blockers = [CancelableRequest(
                    self.http, self.http.chat_payload(prompt, stream=True, max_tokens=2048),
                    f"{case['id']}-block-{index}") for index in range(6)]
                for blocker in blockers:
                    blocker.start()
                request.start()
                deadline = min(time.monotonic() + 30, self.active_deadline)
                while time.monotonic() < deadline:
                    waiting = self._fresh_waiting()
                    if waiting is not None and waiting > 0:
                        evidence["global_waiting_requests"] = waiting
                        evidence["queue_waiting_tied_to_request"] = False
                        break
                    time.sleep(0.1)
            elif stage == "multi":
                additional = [CancelableRequest(
                    self.http, self.http.chat_payload(prompt, stream=True, max_tokens=256),
                    f"{case['id']}-{repetition}-{index}") for index in range(3)]
                requests = [request, *additional]
                for item in requests:
                    item.start()
                synchronized = all(item.started.wait(timeout=10) for item in requests)
            else:
                request.start()
        except Exception as error:
            evidence["stage_error"] = f"{type(error).__name__}: {error}"
        finally:
            requests = [request, *additional, *blockers]
            for item in requests:
                cancel_request(item)
            try:
                evidence["cleanup"] = self._reset_faults(case["id"])
            except Exception as error:
                evidence["cleanup"] = {"errors": [f"{type(error).__name__}: {error}"]}
            shutdown_deadline = min(self.active_deadline, time.monotonic() + 30)
            for item in requests:
                try:
                    item.thread.join(timeout=max(0, shutdown_deadline - time.monotonic()))
                except RuntimeError:  # thread was constructed but never started
                    pass
        cancelled = [request, *additional]
        cancelled_receipts = [
            _request_cancellation_evidence(item, cancellations[id(item)])
            for item in cancelled]
        blocker_receipts = [
            _request_cancellation_evidence(item, cancellations[id(item)])
            for item in blockers]
        evidence.update({
            "synchronized": synchronized,
            "cancelled": cancelled_receipts,
            "blockers": blocker_receipts,
            "blockers_terminated": all(receipt["terminated"] for receipt in blocker_receipts),
        })
        if evidence["cleanup"].get("errors"):
            return "fail", evidence

        cancellation_outcomes = [_cancellation_outcome(receipt)
                                 for receipt in cancelled_receipts]
        blocker_outcomes = [_cancellation_outcome(receipt)
                            for receipt in blocker_receipts]
        evidence["cancellation_outcomes"] = [
            {"request_id": receipt.get("request_id"), "status": status, "reason": reason}
            for receipt, (status, reason) in zip(cancelled_receipts, cancellation_outcomes)]
        evidence["blocker_outcomes"] = [
            {"request_id": receipt.get("request_id"), "status": status, "reason": reason}
            for receipt, (status, reason) in zip(blocker_receipts, blocker_outcomes)]
        failures = [reason for status, reason in [*cancellation_outcomes, *blocker_outcomes]
                    if status == "fail"]
        if any(cancellation.get("error") for cancellation in cancellations.values()):
            failures.append("one or more client cancellation calls raised an error")

        targeted_stop_at = getattr(self, "targeted_stop_at", self.active_deadline)
        targeted_expired = (time.monotonic() >= targeted_stop_at
                            and not self.telemetry.safety_event.is_set())
        substantive_failures = [reason for reason in failures
                                if reason != "client transport did not terminate"]
        if targeted_expired and not substantive_failures:
            evidence["reason"] = "targeted deadline interrupted cancellation case"
            return "pending", evidence
        if failures:
            evidence["reason"] = failures[0]
            return "fail", evidence
        try:
            reissue = self.http.chat(prompt, request_id=f"{case['id']}-reissue")
            evidence["reissue"] = reissue
            evidence["health_200"] = self.health_code() == 200
            idle_status, idle = self.wait_for_api_idle()
            evidence["idle"] = {"status": idle_status, "evidence": idle}
            barriers = self._drain_barriers()
            drain_status, drain = self.wait_for_drain(barriers)
            evidence["reservation_and_staging_drain"] = {
                "status": drain_status, "barriers": barriers, "evidence": drain}
        except Exception as error:
            evidence["recovery_error"] = f"{type(error).__name__}: {error}"
            recovery_response = evidence.get("reissue")
            known_recovery_failure = (
                (recovery_response is not None
                 and not _valid_fixed_completion(recovery_response, prompt_tokens))
                or evidence.get("health_200") is False
                or (evidence.get("idle") or {}).get("status") == "fail")
            if (getattr(error, "guard_abort_cause", None) == "phase_deadline"
                    and time.monotonic() >= getattr(
                        self, "targeted_stop_at", self.active_deadline)
                    and not self.telemetry.safety_event.is_set()
                    and not known_recovery_failure):
                evidence["reason"] = "targeted deadline interrupted cancellation case"
                return "pending", evidence
            return "fail", evidence
        if (drain_status == "fail"
                and time.monotonic() >= getattr(
                    self, "targeted_stop_at", self.active_deadline)
                and not self.telemetry.safety_event.is_set()
                and _valid_fixed_completion(reissue, prompt_tokens)
                and evidence["health_200"] is True
                and idle_status != "fail"):
            evidence["reason"] = (
                "targeted deadline interrupted the post-cancellation drain proof")
            return "pending", evidence
        if (not _valid_fixed_completion(reissue, prompt_tokens)
                or evidence["health_200"] is not True
                or idle_status == "fail" or drain_status == "fail"):
            evidence["reason"] = "post-cancellation response, health, idle, or drain failed"
            return "fail", evidence
        if idle_status != "pass" or drain_status != "pass":
            evidence["reason"] = "post-cancellation idle and fresh drain were not proven"
            return "pending", evidence
        pending = [reason for status, reason in [*cancellation_outcomes, *blocker_outcomes]
                   if status == "pending"]
        if pending:
            evidence["reason"] = pending[0]
            return "pending", evidence
        if not synchronized:
            evidence["reason"] = (
                "global queue depth was observed but could not be tied to the cancelled request"
                if stage == "queue" and evidence.get("global_waiting_requests")
                else "the requested cancellation phase was not synchronized")
            return "pending", evidence
        return "pass", evidence

    def _controller(self, overlay: bool, command: str, timeout: float = 2700,
                    *, tp4_env: str | None = None,
                    deadline_monotonic: float | None = None) -> dict[str, Any]:
        environment = os.environ.copy()
        environment["TP4_HOSTS"] = " ".join(self.cfg["nodes"])
        if overlay and tp4_env is not None:
            raise ValueError("controller cannot select both campaign and explicit TP4_ENV")
        selected_env = self.cfg["tp4_env"] if overlay else tp4_env
        if selected_env is not None:
            environment["TP4_ENV"] = selected_env
        else:
            environment.pop("TP4_ENV", None)
        deadline = (self.active_deadline if deadline_monotonic is None
                    else min(self.active_deadline, deadline_monotonic))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return {"command": command, "overlay": overlay, "returncode": None,
                    "tp4_env": selected_env,
                    "reason": "campaign phase deadline reached before controller invocation"}
        process = subprocess.Popen([str(Path(self.cfg["repo_root"]) / "scripts/tp4ctl"), command],
                                   cwd=self.cfg["repo_root"], env=environment, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        output_lines: list[str] = []
        health_seen: list[float] = []
        def read_output() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                output_lines.append(line)
                if command in {"up", "restart"} and "health 200 after" in line and not health_seen:
                    health_seen.append(time.monotonic())
        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        try:
            process.wait(timeout=min(timeout, remaining))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
            reader.join(timeout=1)
            output = "".join(output_lines)
            if process.stdout is not None:
                process.stdout.close()
            return {"command": command, "overlay": overlay, "returncode": None,
                    "tp4_env": selected_env,
                    "reason": "controller timed out at campaign phase deadline",
                    "output_tail": output[-16000:]}
        finished = time.monotonic()
        reader.join(timeout=1)
        output = "".join(output_lines)
        if process.stdout is not None:
            process.stdout.close()
        result = {"command": command, "overlay": overlay, "tp4_env": selected_env,
                  "returncode": process.returncode,
                  "output_tail": output[-16000:], "finished_monotonic": finished}
        if command in {"up", "restart"} and process.returncode == 0:
            if not health_seen:
                return {**result, "returncode": None,
                        "reason": "controller succeeded without a timestamped health 200 line"}
            result["health_200_monotonic"] = health_seen[0]
        return result

    def _gates(self, health_since: float) -> dict[str, Any]:
        deadline = min(self.active_deadline, health_since + 120)
        if time.monotonic() >= deadline:
            return {"pass": False, "reason": "two-minute gate deadline already expired"}
        client = HTTPClient(self.cfg["endpoint"], self.cfg["model"], timeout=120)
        client.set_guard(threading.Event(), lambda: deadline)
        try:
            coherent = client.json(f"{client.base}/chat/completions", {
                "model": client.model, "temperature": 0, "max_tokens": 64,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": "What is the capital of Italy? Reply with one sentence."}],
            })
            tool = client.json(f"{client.base}/chat/completions", {
                "model": client.model, "max_tokens": 256,
                "messages": [{"role": "user", "content": "What is the weather in Milan?"}],
                "tools": [{"type": "function", "function": {"name": "get_weather",
                    "description": "Get weather for a city", "parameters": {"type": "object",
                    "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
                "tool_choice": "auto",
            })
            message1 = ((coherent.get("body") or {}).get("choices") or [{}])[0].get("message") or {}
            content = message1.get("content") or ""
            message2 = ((tool.get("body") or {}).get("choices") or [{}])[0].get("message") or {}
            calls = message2.get("tool_calls") or []
            function = calls[0].get("function", {}) if calls and isinstance(calls[0], dict) else {}
            try:
                arguments = json.loads(function.get("arguments", ""))
            except (TypeError, json.JSONDecodeError):
                arguments = None
            gate1 = coherent.get("status") == 200 and isinstance(content, str) and any(
                city in content.lower() for city in ("rome", "roma"))
            gate2 = (tool.get("status") == 200 and function.get("name") == "get_weather"
                     and arguments is not None and "milan" in json.dumps(arguments).lower())
            return {"pass": gate1 and gate2 and time.monotonic() <= deadline,
                    "coherent": coherent, "tool": tool, "gate1": gate1, "gate2": gate2,
                    "tool_arguments": arguments}
        except Exception as error:
            return {"pass": False, "error": f"{type(error).__name__}: {error}"}

    def _worker(self, case: dict[str, Any], repetition: int) -> tuple[str, dict[str, Any]]:
        if not self.acknowledge_faults or not self.cfg.get("allow_worker_faults", False):
            return "pending", {"reason": "worker fault execution was not explicitly enabled"}
        remaining = self.active_deadline - time.monotonic()
        if remaining < WORKER_FAULT_MIN_REMAINING_SECONDS:
            return "pending", {"reason": "insufficient targeted-phase time for bounded recovery",
                               "remaining_seconds": max(0, remaining),
                               "required_seconds": WORKER_FAULT_MIN_REMAINING_SECONDS}
        p = case["parameters"]
        prompt = self.http.exact_prompt(128_000, f"{case['id']} {repetition} in-flight cache capture.")
        interrupted = CancelableRequest(self.http, self.http.chat_payload(prompt, stream=True, max_tokens=256),
                                        f"{case['id']}-interrupted")
        evidence: dict[str, Any] = {}
        try:
            self._set_case(case["id"], pause_stage="capture", pause_ranks={p["rank"]})
            cursor = self._cursor(p["rank"])
            interrupted.start()
            stage = self._event(p["rank"], cursor, case["id"], "stage_waiting",
                                stage="capture", timeout=120)
            evidence["stage"] = stage
            arguments = ["signal-stage", "--case-id", case["id"], "--stage", "capture",
                         "--pid", str(stage["pid"]), "--start-ticks", str(stage["process_start_ticks"]),
                         "--signal", p["signal"]]
            expected_window = max(0.1, self.active_deadline - time.monotonic())
            self.telemetry.arm_expected_down(expected_window)
            evidence["signal"] = self._faultctl(p["rank"], arguments)
            if p["signal"] == "STOP":
                time.sleep(2)
                arguments[-1] = "CONT"
                evidence["continue"] = self._faultctl(p["rank"], arguments)
        except Exception as error:
            evidence["signal_error"] = f"{type(error).__name__}: {error}"
        finally:
            try:
                evidence["release"] = self._reset_faults(case["id"])
            except Exception as error:
                evidence["release_error"] = f"{type(error).__name__}: {error}"
        if "signal" not in evidence:
            self.telemetry.clear_expected_down()
            interrupted.cancel()
            interrupted.thread.join(timeout=30)
            evidence["health_200"] = self.health_code() == 200
            released = ("release_error" not in evidence
                        and not evidence.get("release", {}).get("errors"))
            return ("pending" if released else "fail"), evidence
        try:
            evidence["pre_recovery_capture"] = self.preserve_cluster_evidence(
                f"{case['id']}-r{repetition}-before-restart")
        except Exception as error:
            evidence["pre_recovery_capture"] = {
                "status": "error", "error": f"{type(error).__name__}: {error}"}
        recovery_started = time.monotonic()
        evidence["recovery"] = self._controller(True, "restart")
        try:
            if evidence["recovery"]["returncode"] == 0:
                evidence["telemetry_settled"] = self.telemetry.settle_expected_restart(recovery_started)
                evidence["reservation_prune"] = [
                    self._faultctl(rank, ["prune-reservations"]) for rank in range(4)]
        except Exception as error:
            evidence["post_recovery_error"] = f"{type(error).__name__}: {error}"
        finally:
            self.telemetry.clear_expected_down()
        interrupted.thread.join(timeout=30)
        stream = interrupted.result or {}
        completed_successfully = (
            stream.get("status") == 200 and stream.get("stream_complete") is True
            and stream.get("error_event") is False and not stream.get("parse_errors")
            and bool(stream.get("finish_reasons")))
        request_failed = (interrupted.error is not None or interrupted.result is None
                          or not completed_successfully)
        evidence["interrupted_request"] = {"result": interrupted.result, "error": interrupted.error,
                                            "terminated": not interrupted.thread.is_alive(),
                                            "failed": request_failed,
                                            "completed_successfully": completed_successfully}
        evidence["gates"] = (self._gates(evidence["recovery"]["health_200_monotonic"])
                             if evidence["recovery"]["returncode"] == 0 else {"pass": False})
        if not evidence["gates"]["pass"]:
            try:
                evidence["gate_failure_capture"] = self.preserve_cluster_evidence(
                    f"{case['id']}-r{repetition}-gate-failure")
            except Exception as error:
                evidence["gate_failure_capture"] = {
                    "status": "error", "error": f"{type(error).__name__}: {error}"}
            evidence["shutdown_after_gate_failure"] = self._controller(True, "down", timeout=600)
        return ("pass" if evidence["interrupted_request"]["terminated"]
                and (p["signal"] == "STOP" or evidence["interrupted_request"]["failed"])
                and evidence["recovery"]["returncode"] == 0
                and evidence.get("telemetry_settled") is True
                and evidence["pre_recovery_capture"].get("status") == "captured"
                and "release_error" not in evidence
                and not evidence.get("release", {}).get("errors")
                and "post_recovery_error" not in evidence
                and evidence["gates"]["pass"] else "fail", evidence)

    @staticmethod
    def _assistant_content(response: dict[str, Any]) -> tuple[bool, str, bool]:
        body = response.get("body")
        choices = body.get("choices") if isinstance(body, dict) else None
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            return False, "", False
        message = choices[0].get("message")
        if not isinstance(message, dict):
            return False, "", False
        if "content" not in message:
            return False, "", False
        content = message.get("content")
        if isinstance(content, str):
            return True, content, False
        if content is None:
            return True, "", True
        return False, "", False

    def _soak_resource_snapshot(self) -> dict[str, Any]:
        with self.telemetry.lock:
            samples = dict(self.telemetry.latest)
        fields = ("time_ns", "monotonic_ns", "mem_available_bytes", "swap_total_bytes",
                  "swap_free_bytes", "memory_pressure", "process", "api", "health",
                  "connections", "cache", "cuda", "operations", "oom", "cadence",
                  "sources", "container_state", "container_pid")
        selected = {str(rank): {key: sample[key] for key in fields if key in sample}
                    for rank, sample in samples.items() if isinstance(sample, dict)}
        return json.loads(json.dumps(selected))

    def _soak_drain_proof(self, label: str, *,
                          deadline: float | None = None) -> tuple[str, dict[str, Any]]:
        previous_deadline = self.active_deadline
        proof_boundary = self.restore_at if deadline is None else deadline
        proof_deadline = min(proof_boundary, time.monotonic() + 60.0)
        self.active_deadline = proof_deadline
        evidence: dict[str, Any] = {"label": label, "deadline_monotonic": proof_deadline}
        try:
            evidence["health_200"] = self.health_code() == 200
            idle_status, idle = self.wait_for_api_idle()
            evidence["api_idle"] = {"status": idle_status, "evidence": idle}
            barriers = self._drain_barriers()
            drain_status, drain = self.wait_for_drain(barriers)
            evidence["barriers"] = barriers
            evidence["reservation_and_staging_drain"] = {
                "status": drain_status, "evidence": drain}
        except Exception as error:
            evidence["error"] = f"{type(error).__name__}: {error}"
            deadline_exhausted = (isinstance(error, PhaseDeadlineError)
                                  and time.monotonic() >= proof_deadline
                                  and not self.telemetry.safety_event.is_set())
            return ("pending" if deadline_exhausted else "fail"), evidence
        finally:
            self.active_deadline = previous_deadline
        if not evidence["health_200"] or idle_status == "fail" or drain_status == "fail":
            return "fail", evidence
        if idle_status != "pass" or drain_status != "pass":
            return "pending", evidence
        return "pass", evidence

    def _soak(self, case: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        invoked = time.monotonic()
        required_seconds = float(self.cfg["soak_hours"]) * 3600
        hard_load_deadline = self.restore_at - SOAK_FINAL_PROOF_RESERVE_SECONDS
        load_deadline = hard_load_deadline
        receipt_stamp = time.time_ns()
        receipt = self.output / f"soak-v{SOAK_PROTOCOL_VERSION}-{receipt_stamp}.jsonl"
        receipt_suffix = 0
        while receipt.exists():
            receipt_suffix += 1
            receipt = self.output / (
                f"soak-v{SOAK_PROTOCOL_VERSION}-{receipt_stamp}-{receipt_suffix}.jsonl")
        receipt_count = 0
        categories = {key: 0 for key in ("ordinary", "growth", "replay", "cancel", "idle")}
        errors: list[str] = []
        deadline_events: list[str] = []
        pending_reason: str | None = None
        failure = False
        deadline_aborted = False

        self.state["active_soak_receipt"] = {
            "protocol_version": SOAK_PROTOCOL_VERSION, "path": str(receipt),
            "count": 0, "status": "initializing", "started_ns": receipt_stamp}
        self.save()

        def write_receipt(value: dict[str, Any]) -> None:
            nonlocal receipt_count
            try:
                _append_private_event(receipt, {"kind": "soak_receipt",
                                                "protocol_version": SOAK_PROTOCOL_VERSION,
                                                **value})
                receipt_count += 1
                self.state["active_soak_receipt"].update(
                    count=receipt_count, status="active", last_write_ns=time.time_ns())
                self.save()
            except Exception as error:
                raise SoakReceiptError(
                    f"cannot append soak receipt: {type(error).__name__}: {error}") from error

        def streamed_http(request: CancelableRequest) -> dict[str, Any]:
            return {
                "status": request.http_status, "headers": request.response_headers,
                "result": request.result, "error": request.error,
                "accepted_output": (request.result.get("body_text")
                                    if isinstance(request.result, dict) else None),
                "started_ns": request.started_ns,
                "first_byte_ns": request.first_byte_ns,
                "finished_ns": request.finished_ns,
                "first_byte_hex": request.first_byte_hex,
            }

        def proof_was_deadline_truncated(status: str, proof: dict[str, Any],
                                         request_started: float) -> bool:
            if (time.monotonic() < load_deadline
                    or self.telemetry.safety_event.is_set()
                    or time.monotonic() - request_started > SOAK_BENIGN_ABORT_MAX_SECONDS
                    or proof.get("health_200") is not True):
                return False
            error = proof.get("error")
            if error is not None:
                return (isinstance(error, str)
                        and error.startswith("PhaseDeadlineError:"))
            idle = proof.get("api_idle") or {}
            idle_status = idle.get("status")
            idle_reason = (idle.get("evidence") or {}).get("reason")
            drain = proof.get("reservation_and_staging_drain") or {}
            drain_status = drain.get("status")
            drain_reason = (drain.get("evidence") or {}).get("reason")
            return (status in {"pending", "fail"}
                    and idle_status in {"pass", "pending"}
                    and drain_status in {"pass", "pending", "fail"}
                    and (idle_status != "pending" or idle_reason == (
                        "phase deadline cut short the idle proof"))
                    and (drain_status == "pass" or drain_reason == (
                        "reservation and staging drain were not proven before timeout"))
                    and (idle_status == "pending" or drain_status != "pass"))

        if invoked > self.soak_at:
            self.state["active_soak_receipt"]["status"] = "late_start"
            self.save()
            return "pending", {
                "protocol_version": SOAK_PROTOCOL_VERSION,
                "reason": "insufficient time for the required soak and final drain proof",
                "started_monotonic": invoked, "planned_load_end_monotonic": load_deadline,
                "soak_boundary_monotonic": self.soak_at,
                "restore_boundary_monotonic": self.restore_at,
                "required_seconds": required_seconds,
            }

        conversations = [{"conversation_id": f"conversation-{index}", "epoch": 0,
                          "turn": 0, "target_index": 0, "pending_replay": False,
                          "messages": [], "replay_messages": None,
                          "replay_payload_sha256": None}
                         for index in range(SOAK_CONVERSATION_COUNT)]
        ordinary_index = attempt = 0
        self.telemetry.start(1.0)
        started = time.monotonic()
        if started > self.soak_at:
            self.state["active_soak_receipt"].update(
                status="late_after_telemetry_start", load_started_monotonic=started)
            self.save()
            return "pending", {
                "protocol_version": SOAK_PROTOCOL_VERSION,
                "reason": "telemetry restart left insufficient time for the required soak",
                "started_monotonic": started, "invoked_monotonic": invoked,
                "planned_load_end_monotonic": load_deadline,
                "soak_boundary_monotonic": self.soak_at,
                "restore_boundary_monotonic": self.restore_at,
                "required_seconds": required_seconds,
            }
        if self.cfg.get("soak_stop_after_required_seconds", False):
            load_deadline = min(hard_load_deadline, started + required_seconds)
        self.state["active_soak_receipt"].update(
            status="active", load_started_monotonic=started,
            planned_load_end_monotonic=load_deadline,
            hard_load_end_monotonic=hard_load_deadline,
            stop_after_required_seconds=self.cfg.get(
                "soak_stop_after_required_seconds", False))
        self.save()
        self.active_deadline = load_deadline

        while time.monotonic() < load_deadline and not self.telemetry.safety_event.is_set():
            attempt += 1
            request_started_mono = time.monotonic()
            request_started_ns = time.time_ns()
            request_id = (f"soak-{self.cfg['campaign_id']}-v{SOAK_PROTOCOL_VERSION}-"
                          f"{receipt_stamp}-{attempt}")
            request_type = "cancel" if attempt % SOAK_CANCEL_EVERY == 0 else "ordinary"
            row: dict[str, Any] = {
                "attempt": attempt, "request_id": request_id, "type": request_type,
                "request_started_ns": request_started_ns, "request_payload": None,
                "conversation_id": None, "turn": None, "target_tokens": None,
            }
            request: CancelableRequest | None = None
            try:
                if request_type == "cancel":
                    messages = self.http.exact_messages(
                        8_000, [], f"campaign {self.cfg['campaign_id']} soak cancellation {attempt}.")
                    payload = self.http.chat_payload_messages(messages, stream=True, max_tokens=64)
                    row.update(request_payload=payload, target_tokens=8_000)
                    request = CancelableRequest(self.http, payload, request_id)
                    request.start()
                    first_byte = request.first_byte.wait(
                        timeout=min(120, max(0, load_deadline - time.monotonic())))
                    cancelled_ns = time.time_ns()
                    cancel_requested_before_finish = request.thread.is_alive()
                    request.cancel()
                    cleanup_deadline = min(self.restore_at, time.monotonic() + 10)
                    request.thread.join(timeout=max(0, cleanup_deadline - time.monotonic()))
                    terminated = not request.thread.is_alive()
                    guard_abort_cause = getattr(request, "guard_abort_cause", None)
                    if guard_abort_cause is None and cancel_requested_before_finish:
                        guard_abort_cause = (
                            "safety_event" if self.telemetry.safety_event.is_set()
                            else "phase_deadline"
                            if time.monotonic() >= load_deadline else None)
                    proof_status, proof = self._soak_drain_proof(
                        f"cancel-{attempt}", deadline=load_deadline)
                    request_deadline_truncated = (
                        not first_byte and time.monotonic() >= load_deadline
                        and guard_abort_cause == "phase_deadline"
                        and not self.telemetry.safety_event.is_set()
                        and time.monotonic() - request_started_mono
                        <= SOAK_BENIGN_ABORT_MAX_SECONDS)
                    proof_deadline_truncated = proof_was_deadline_truncated(
                        proof_status, proof, request_started_mono)
                    deadline_truncated = (
                        request_deadline_truncated or proof_deadline_truncated)
                    observed_completion = (isinstance(request.result, dict)
                                           and (request.result.get("done") is True
                                                or (bool(request.result.get("finish_reasons"))
                                                    and request.result.get(
                                                        "error_event") is False
                                                    and not request.result.get(
                                                        "parse_errors"))))
                    interrupted = bool(cancel_requested_before_finish and terminated
                                       and not observed_completion)
                    row.update(
                        request_finished_ns=request.finished_ns or time.time_ns(),
                        http=streamed_http(request),
                        guard_abort_cause=guard_abort_cause,
                        cancellation={"first_byte": first_byte, "cancelled_ns": cancelled_ns,
                                      "requested_before_finish": cancel_requested_before_finish,
                                      "interrupted": interrupted, "terminated": terminated,
                                      "deadline_truncated": deadline_truncated,
                                      "post_cancel_proof": proof,
                                      "post_cancel_proof_status": proof_status})
                    write_receipt(row)
                    if not terminated:
                        failure = True
                        errors.append("soak cancellation request did not terminate")
                        break
                    if (first_byte or request.http_status is not None) and request.http_status != 200:
                        failure = True
                        errors.append(f"soak cancellation request returned HTTP {request.http_status}")
                        break
                    if not first_byte:
                        if request_deadline_truncated:
                            if proof_status == "pass" or proof_deadline_truncated:
                                deadline_aborted = True
                                deadline_events.append(
                                    "cancellation request ended at the load deadline")
                            else:
                                if proof_status == "fail":
                                    failure = True
                                    errors.append(
                                        "post-cancellation idle or resource drain failed")
                                else:
                                    pending_reason = (
                                        "post-cancellation idle or resource drain was not proven")
                            break
                        failure = True
                        errors.append("soak cancellation request produced no first byte")
                        break
                    if proof_status == "fail":
                        if deadline_truncated:
                            deadline_aborted = True
                            deadline_events.append(
                                "post-cancellation proof truncated at load deadline")
                            break
                        failure = True
                        errors.append("post-cancellation idle or resource drain failed")
                        break
                    if proof_status != "pass":
                        if deadline_truncated:
                            deadline_aborted = True
                            deadline_events.append(
                                "post-cancellation proof truncated at load deadline")
                            break
                        pending_reason = "post-cancellation idle or resource drain was not proven"
                        break
                    if interrupted:
                        categories["cancel"] += 1
                else:
                    state = conversations[ordinary_index % len(conversations)]
                    ordinary_index += 1
                    is_replay = bool(state["pending_replay"])
                    if is_replay:
                        messages = [dict(message) for message in state["replay_messages"]]
                        subtype = "replay"
                    else:
                        target = SOAK_CONTEXT_TARGETS[state["target_index"]]
                        prefix = (f"campaign {self.cfg['campaign_id']} soak "
                                  f"{state['conversation_id']} epoch {state['epoch']} "
                                  f"turn {state['turn']}. Continue this conversation.")
                        messages = self.http.exact_messages(target, state["messages"], prefix)
                        subtype = "growth"
                    target = SOAK_CONTEXT_TARGETS[state["target_index"]]
                    payload = self.http.chat_payload_messages(messages, max_tokens=16)
                    payload_sha = hashlib.sha256(json.dumps(
                        payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                    if is_replay and payload_sha != state["replay_payload_sha256"]:
                        raise RuntimeError("soak replay payload differs from its growth request")
                    row.update(type=subtype, conversation_id=state["conversation_id"],
                               conversation_epoch=state["epoch"], turn=state["turn"],
                               target_tokens=target, request_payload=payload,
                               request_payload_sha256=payload_sha,
                               replay_of_payload_sha256=(state["replay_payload_sha256"]
                                                         if is_replay else None))
                    response = self.http.json(
                        f"{self.http.base}/chat/completions", payload,
                        headers={"X-Request-Id": request_id})
                    row.update(request_finished_ns=response.get("finished_ns", time.time_ns()),
                               http=response, guard_abort_cause=None)
                    assistant_valid, assistant, visible_content_null = self._assistant_content(response)
                    row["accepted_assistant_content"] = assistant
                    row["visible_content_null"] = visible_content_null
                    write_receipt(row)
                    if response.get("status") != 200 or not assistant_valid:
                        failure = True
                        errors.append("soak conversation response was not a valid HTTP 200 assistant message")
                        break
                    categories["ordinary"] += 1
                    categories[subtype] += 1
                    if is_replay:
                        state["pending_replay"] = False
                        state["target_index"] += 1
                        if state["target_index"] == len(SOAK_CONTEXT_TARGETS):
                            state.update(epoch=state["epoch"] + 1, turn=0, target_index=0,
                                         messages=[], replay_messages=None,
                                         replay_payload_sha256=None)
                    else:
                        state["messages"] = [*messages, {"role": "assistant", "content": assistant}]
                        if len(state["messages"]) > SOAK_MAX_HISTORY_MESSAGES:
                            raise RuntimeError("soak conversation exceeded its bounded history")
                        state["replay_messages"] = [dict(message) for message in messages]
                        state["replay_payload_sha256"] = payload_sha
                        state["pending_replay"] = True
                        state["turn"] += 1
            except BaseException as error:
                now = time.monotonic()
                request_seconds = now - request_started_mono
                error_text = f"{type(error).__name__}: {error}"
                if not isinstance(error, Exception):
                    cancellation_cleanup: dict[str, Any] | None = None
                    if request is not None:
                        requested_ns = time.time_ns()
                        alive_before = request.thread.is_alive()
                        cancellation_cleanup = {
                            "requested_ns": requested_ns,
                            "thread_alive_before_cancel": alive_before,
                        }
                        try:
                            request.cancel()
                        except Exception as cancel_error:
                            cancellation_cleanup["cancel_error"] = (
                                f"{type(cancel_error).__name__}: {cancel_error}")
                        try:
                            request.thread.join(timeout=min(
                                5.0, max(0, self.restore_at - time.monotonic())))
                        except Exception as join_error:
                            cancellation_cleanup["join_error"] = (
                                f"{type(join_error).__name__}: {join_error}")
                        cancellation_cleanup.update(
                            returned_ns=time.time_ns(),
                            terminated=not request.thread.is_alive())
                        row["http"] = streamed_http(request)
                        row["guard_abort_cause"] = getattr(
                            request, "guard_abort_cause", None)
                        row["cancellation"] = cancellation_cleanup
                    row.update(request_finished_ns=row.get("request_finished_ns", time.time_ns()),
                               terminal_error=error_text,
                               incomplete_due_to_base_exception=True)
                    try:
                        write_receipt(row)
                    except Exception as receipt_error:
                        self.state["active_soak_receipt"]["receipt_error"] = (
                            f"{type(receipt_error).__name__}: {receipt_error}")
                    self.state["active_soak_receipt"].update(
                        count=receipt_count, status="interrupted",
                        interrupted_error=error_text,
                        interrupted_ns=time.time_ns())
                    try:
                        self.save()
                    except Exception as save_error:
                        add_note = getattr(error, "add_note", None)
                        if callable(add_note):
                            add_note("soak interruption state save failed: "
                                     f"{type(save_error).__name__}: {save_error}")
                    raise
                receipt_failed = isinstance(error, SoakReceiptError)
                if row.get("request_finished_ns") is None:
                    row.update(request_finished_ns=time.time_ns(), http=None,
                               error=error_text,
                               guard_abort_cause=getattr(error, "guard_abort_cause", None))
                    try:
                        write_receipt(row)
                    except Exception as receipt_error:
                        receipt_failed = True
                        errors.append(f"{type(receipt_error).__name__}: {receipt_error}")
                if (isinstance(error, GuardAbortError)
                        and error.guard_abort_cause == "phase_deadline"
                        and now >= load_deadline and not receipt_failed
                        and not self.telemetry.safety_event.is_set()):
                    deadline_events.append(error_text)
                    if request_seconds <= SOAK_BENIGN_ABORT_MAX_SECONDS:
                        deadline_aborted = True
                    else:
                        pending_reason = (
                            "the final request exceeded the bounded deadline-abort allowance")
                        errors.append(error_text)
                    break
                if receipt_failed:
                    failure = True
                    errors.append(error_text)
                elif now >= load_deadline and request_seconds > 900:
                    failure = True
                    errors.append(error_text)
                    errors.append(f"request remained in flight for {request_seconds:.3f}s at soak deadline")
                elif isinstance(error, RuntimeError) and "cannot synthesize exactly" in str(error):
                    pending_reason = "prompt synthesis limitation"
                    errors.append(error_text)
                else:
                    failure = True
                    errors.append(error_text)
                break

            if attempt % SOAK_IDLE_EVERY == 0:
                remaining = load_deadline - time.monotonic()
                if remaining >= SOAK_IDLE_SECONDS:
                    idle_started = time.monotonic()
                    idle_row: dict[str, Any] = {
                        "type": "idle", "attempt": attempt,
                        "idle_started_ns": time.time_ns()}
                    try:
                        idle_row.update(health_before=self.health_code(),
                                        resources_before=self._soak_resource_snapshot())
                        time.sleep(SOAK_IDLE_SECONDS)
                        idle_row.update(idle_finished_ns=time.time_ns(),
                                        duration_seconds=time.monotonic() - idle_started,
                                        health_after=self.health_code(),
                                        resources_after=self._soak_resource_snapshot())
                        write_receipt(idle_row)
                    except BaseException as error:
                        error_text = f"{type(error).__name__}: {error}"
                        if not isinstance(error, SoakReceiptError):
                            idle_row.update(idle_finished_ns=time.time_ns(),
                                            duration_seconds=time.monotonic() - idle_started,
                                            error=error_text,
                                            incomplete_due_to_base_exception=(
                                                not isinstance(error, Exception)))
                            try:
                                write_receipt(idle_row)
                            except Exception as receipt_error:
                                errors.append(
                                    f"{type(receipt_error).__name__}: {receipt_error}")
                        if not isinstance(error, Exception):
                            raise
                        failure = True
                        errors.append(error_text)
                        break
                    if idle_row["health_before"] != 200 or idle_row["health_after"] != 200:
                        failure = True
                        errors.append("health failed around a soak idle window")
                        break
                    categories["idle"] += 1

        load_finished = time.monotonic()
        final_proof_status, final_proof = self._soak_drain_proof("soak-final")
        receipt_reference: dict[str, Any] = {"path": str(receipt), "count": receipt_count,
                                             "bytes": 0, "sha256": None}
        try:
            if receipt.exists():
                receipt_reference["bytes"] = receipt.stat().st_size
                receipt_reference["sha256"] = sha256_file(receipt)
        except Exception as error:
            failure = True
            error_text = f"receipt finalization failed: {type(error).__name__}: {error}"
            errors.append(error_text)
            receipt_reference["error"] = error_text
        try:
            self.state["active_soak_receipt"].update(
                receipt_reference,
                status=("finalization_error" if "error" in receipt_reference else "finalized"),
                finalized_ns=time.time_ns())
            self.save()
        except Exception as error:
            failure = True
            error_text = f"receipt state finalization failed: {type(error).__name__}: {error}"
            errors.append(error_text)
            receipt_reference.setdefault("error", error_text)
        missing = [name for name, count in categories.items() if count == 0]
        evidence = {
            "protocol_version": SOAK_PROTOCOL_VERSION,
            "load_started_monotonic": started, "load_finished_monotonic": load_finished,
            "planned_load_end_monotonic": load_deadline,
            "hard_load_end_monotonic": hard_load_deadline,
            "stop_after_required_seconds": self.cfg.get(
                "soak_stop_after_required_seconds", False),
            "elapsed_seconds": load_finished - started, "required_seconds": required_seconds,
            "attempts": attempt,
            "deadline_aborted": deadline_aborted, "category_counts": categories,
            "missing_categories": missing, "request_errors": errors,
            "deadline_events": deadline_events,
            "safety_reasons": list(self.telemetry.reasons),
            "final_proof_status": final_proof_status, "final_proof": final_proof,
            "receipt": receipt_reference,
        }
        if failure or self.telemetry.safety_event.is_set() or final_proof_status == "fail":
            return "fail", evidence
        if pending_reason or errors or load_finished - started < required_seconds or missing \
                or final_proof_status != "pass":
            if pending_reason:
                evidence["reason"] = pending_reason
            elif load_finished - started < required_seconds:
                evidence["reason"] = "required soak duration was not reached"
            elif missing:
                evidence["reason"] = "required mixed-load categories were not all completed"
            elif errors:
                evidence["reason"] = "request or receipt errors remain in the soak evidence"
            else:
                evidence["reason"] = "final idle and resource drain were not proven"
            return "pending", evidence
        return "pass", evidence

    def _combination(self, case: dict[str, Any], repetition: int) -> tuple[str, dict[str, Any]]:
        prerequisites: dict[str, Any] = {}
        for case_id in case["parameters"]["prerequisites"]:
            row = self.state["cases"].get(case_id, {})
            protocol = _targeted_case_protocol(case_id)
            successful = self._successes_after_last_failure(
                row.get("runs", []), "pass", self.runtime_config_identity, protocol)
            prerequisites[case_id] = {
                "successful_repetitions": successful,
                "target_repetitions": row.get("target_repetitions", TARGET_REPEATS),
                "status": row.get("status", "pending"),
                "protocol_version": protocol,
            }
        if any(value["successful_repetitions"] < value["target_repetitions"]
               for value in prerequisites.values()):
            return "pending", {
                "reason": "combination prerequisites have not all passed",
                "prerequisites": prerequisites, "repetition": repetition}
        if case["id"] == "combination-cache-eio-cancel-restore":
            status, evidence = run_cache_eio_cancel_restore(
                self, case, repetition,
                lambda payload, request_id: CancelableRequest(
                    self.http, payload, request_id))
            evidence["prerequisites"] = prerequisites
            return status, evidence
        return "pending", {
            "reason": "no reviewed synchronized driver exists for this combined scenario",
            "prerequisites": prerequisites, "repetition": repetition}

    def execute_case(self, case: dict[str, Any], repetition: int) -> tuple[str, dict[str, Any]]:
        if case.get("manual"):
            receipt = self.cfg.get("rigmark", {}).get("initial" if "initial" in case["id"] else "final")
            path = Path(receipt) if receipt else None
            threshold = (self.state.get("restoration_completed_ns") if "final" in case["id"]
                         else self.state["started_ns"])
            if path:
                valid, evidence = self._receipt_evidence(path, threshold)
                return ("recorded" if valid else "pending"), evidence
            return "pending", {"reason": "native Rigmark receipt path is not configured"}
        return {
            "contexts": self._context, "queue": self._queue, "api": self._api,
            "slow_clients": self._slow, "cache": self._cache,
            "cancellation": self._cancel, "worker": self._worker,
            "combinations": self._combination,
        }[case["area"]](case, repetition)

    def wait_for_drain(self, barriers: dict[int, int]) -> tuple[str, dict[str, Any]]:
        deadline = min(self.active_deadline,
                       time.monotonic() + float(self.cfg.get("drain_timeout_seconds", 300)))
        last: dict[int, Any] = {}
        while time.monotonic() < deadline:
            with self.telemetry.lock:
                samples = dict(self.telemetry.latest)
            ready = True
            for rank in range(4):
                sample = samples.get(rank)
                cache = (sample or {}).get("cache") or {}
                scan_started = cache.get("scan_started_monotonic_ns")
                if not isinstance(scan_started, int) or scan_started <= barriers.get(rank, 0):
                    ready = False
                    last[rank] = {"scan_started_monotonic_ns": scan_started,
                                  "barrier_monotonic_ns": barriers.get(rank)}
                    continue
                last[rank] = {key: cache.get(key) for key in
                              ("reserved_bytes", "reservation_status", "scan_complete", "sample_fresh",
                               "scan_started_monotonic_ns", "staging_bytes",
                               "anonymous_staging_complete", "accounting_status")}
                last[rank]["barrier_monotonic_ns"] = barriers.get(rank)
                if (cache.get("reservation_status") != "ok" or cache.get("scan_complete") is not True
                        or cache.get("sample_fresh") is not True
                        or cache.get("accounting_status") != "ok"
                        or cache.get("anonymous_staging_complete") is not True):
                    ready = False
                elif cache.get("reserved_bytes") != 0 or cache.get("staging_bytes") != 0:
                    ready = False
            if ready:
                return "pass", {"ranks": last}
            time.sleep(0.2)
        known_nonzero = any(value.get("reservation_status") == "ok"
                            and (isinstance(value.get("reserved_bytes"), int)
                                 and value["reserved_bytes"] != 0
                                 or isinstance(value.get("staging_bytes"), int)
                                 and value["staging_bytes"] != 0)
                            for value in last.values())
        return ("fail" if known_nonzero else "pending",
                {"ranks": last,
                 "reason": "reservation and staging drain were not proven before timeout"})

    def _final_restore_selection(self, fatal: bool) -> dict[str, Any]:
        contract = self.final_restore_contract
        selected = "protected-default"
        if contract is None:
            reason = "no conditional final restoration target is configured"
        elif fatal:
            reason = "campaign has a fatal result"
        elif self.telemetry.safety_event.is_set():
            reason = "campaign safety stop is active"
        else:
            row = self.state.get("cases", {}).get("soak-mixed", {})
            soak_passes = self._successes_after_last_failure(
                row.get("runs", []), "pass", self.runtime_config_identity,
                SOAK_PROTOCOL_VERSION)
            if soak_passes < 1:
                reason = "current-runtime two-hour soak has not passed"
            else:
                selected = "configured-final"
                reason = "current-runtime two-hour soak passed without fatal or safety stop"
        return {
            "selected": selected,
            "reason": reason,
            "configured": contract is not None,
            "contract": dict(contract) if contract is not None else None,
        }

    def _validate_selected_final_restore(self,
                                         selection: dict[str, Any]) -> dict[str, Any]:
        if selection.get("selected") != "configured-final":
            return selection
        contract = self.final_restore_contract
        assert contract is not None
        try:
            current = _final_restore_contract(self.cfg)
        except Exception as error:
            current = None
            mismatch = f"{type(error).__name__}: {error}"
        else:
            mismatch = None if current == contract else "configured final target pins changed"
        remote_verification: list[dict[str, Any]] = []
        if mismatch is None:
            remote_path = f"tp4/{contract['restore_env']}"
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = {pool.submit(
                    self._ssh, rank, "sha256sum", remote_path): rank
                    for rank in range(4)}
                for future in as_completed(futures):
                    rank = futures[future]
                    try:
                        completed = future.result()
                        fields = (completed.stdout or "").strip().split(maxsplit=1)
                        actual = fields[0] if fields else None
                        row = {"rank": rank, "returncode": completed.returncode,
                               "sha256": actual,
                               "error": (completed.stderr or "").strip() or None}
                    except Exception as error:
                        row = {"rank": rank, "returncode": None, "sha256": None,
                               "error": f"{type(error).__name__}: {error}"}
                    remote_verification.append(row)
            remote_verification.sort(key=lambda row: row["rank"])
            if any(row["returncode"] != 0
                   or row["sha256"] != contract["restore_env_sha256"]
                   for row in remote_verification):
                mismatch = "final restoration environment failed all-rank SHA-256 verification"
        if mismatch is None:
            return {**selection, "remote_env_verification": remote_verification}
        return {
            "selected": "protected-default",
            "reason": "configured final target failed pin validation; protected default selected",
            "configured": True,
            "contract": dict(contract),
            "candidate_validation_error": mismatch,
            "remote_env_verification": remote_verification,
        }

    def _safe_final_restore_selection(self, fatal: bool) -> dict[str, Any]:
        try:
            return self._final_restore_selection(fatal)
        except BaseException as error:
            return {
                "selected": "protected-default",
                "reason": "final restoration selection failed; protected default selected",
                "configured": self.final_restore_contract is not None,
                "contract": (dict(self.final_restore_contract)
                             if self.final_restore_contract is not None else None),
                "selection_error": f"{type(error).__name__}: {error}",
            }

    def restore_default(self) -> dict[str, Any]:
        self.active_deadline = self.end_monotonic
        selection = self._validate_selected_final_restore(
            dict(self.state.get("final_restore_selection") or {
                "selected": "protected-default",
                "reason": "no conditional final restoration selection was recorded",
                "configured": self.final_restore_contract is not None,
                "contract": (dict(self.final_restore_contract)
                             if self.final_restore_contract is not None else None),
            }))
        self.state["final_restore_selection"] = selection
        use_final = selection["selected"] == "configured-final"
        contract = self.final_restore_contract if use_final else None
        final_env = contract["restore_env"] if contract is not None else None
        self.final_native_rigmark_identity = {
            **self.initial_native_rigmark_identity,
            "restoration_verified": False,
        }
        if contract is not None:
            self.final_native_rigmark_identity = {
                "identity_scope": "restored-operational/native-rigmark",
                "runtime_variant": "configured-final",
                "tp4_env": contract["restore_env"],
                "tp4_env_sha256": contract["restore_env_sha256"],
                "operational_identity": contract["operational_identity"],
                "operational_identity_sha256": contract["operational_identity_sha256"],
                "identity_id": contract["identity_id"],
                "restoration_verified": False,
            }
        result = {"selection": dict(selection),
                  "down_overlay": self._controller(True, "down")}
        if result["down_overlay"]["returncode"] == 0:
            result["up_default"] = self._controller(
                False, "up", tp4_env=final_env,
                deadline_monotonic=(self.end_monotonic
                                    - FINAL_RESTORE_PROOF_RESERVE_SECONDS))
        else:
            result["up_default"] = {"returncode": None, "reason": "overlay teardown failed"}
        if result["up_default"].get("returncode") == 0:
            result["gates"] = self._gates(result["up_default"]["health_200_monotonic"])
            if not result["gates"]["pass"]:
                try:
                    result["gate_failure_capture"] = self.preserve_cluster_evidence(
                        "final-restoration-gate-failure")
                except Exception as error:
                    result["gate_failure_capture"] = {
                        "status": "error", "error": f"{type(error).__name__}: {error}"}
                result["shutdown_after_gate_failure"] = self._controller(
                    False, "down", timeout=600, tp4_env=final_env)
            else:
                environment = os.environ.copy()
                environment["TP4_HOSTS"] = " ".join(self.cfg["nodes"])
                if final_env is None:
                    environment.pop("TP4_ENV", None)
                else:
                    environment["TP4_ENV"] = final_env
                remaining = self.end_monotonic - time.monotonic()
                identity_command = [sys.executable, "scripts/check-f0.py"]
                if contract is not None:
                    identity_command.extend(
                        ["--identity", contract["operational_identity"]])
                try:
                    identity = subprocess.run(
                        identity_command, cwd=self.cfg["repo_root"],
                        env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        timeout=min(120, max(0.1, remaining)), check=False)
                    result["identity"] = {"returncode": identity.returncode,
                                          "command": identity_command,
                                          "output_tail": identity.stdout[-16000:]}
                    self.final_native_rigmark_identity["restoration_verified"] = (
                        identity.returncode == 0)
                except subprocess.TimeoutExpired as error:
                    result["identity"] = {"returncode": None, "reason": "identity deadline",
                                          "command": identity_command,
                                          "output_tail": error.stdout[-16000:] if isinstance(error.stdout, str) else ""}
                if result["identity"].get("returncode") != 0:
                    try:
                        result["identity_failure_capture"] = self.preserve_cluster_evidence(
                            "final-restoration-identity-failure")
                    except Exception as error:
                        result["identity_failure_capture"] = {
                            "status": "error",
                            "error": f"{type(error).__name__}: {error}",
                        }
                    result["shutdown_after_identity_failure"] = self._controller(
                        False, "down", timeout=60, tp4_env=final_env,
                        deadline_monotonic=self.end_monotonic)
        return result

    def verify_overlay(self) -> dict[str, Any]:
        bundle = Path(self.cfg["repo_root"]) / Path(self.cfg["tp4_env"]).parent
        manifest = load_json(bundle / "manifest.json")
        evidence: dict[str, Any] = {"ranks": []}
        variant = self.cfg["runtime_variant"]
        uses_trim = variant in {"prefill-cache-trim", "prefill-step-cap"}
        uses_step_cap = variant == "prefill-step-cap"
        worker_key = "candidate" if uses_trim else "parent"
        expected_worker_sha = self.worker_contract[f"{worker_key}_sha256"]
        worker_source_suffix = "/" + self.worker_contract[f"{worker_key}_path"].removeprefix(
            "scripts/node/")
        selected_name = Path(self.cfg["tp4_env"]).name
        manifest_files = manifest.get("files") if isinstance(manifest, dict) else None
        if (not isinstance(manifest_files, dict)
                or manifest.get("runtime_variant") != variant
                or manifest.get("selected_delta") != selected_name
                or manifest_files.get(selected_name) !=
                self.runtime_config_identity["tp4_env_sha256"]):
            raise RuntimeError("selected runtime overlay differs from its generated manifest")
        kv_configured = "kv_cache_memory_bytes" in self.cfg
        if (kv_configured != ("kv_cache_memory_bytes" in manifest)
                or (kv_configured and manifest.get("kv_cache_memory_bytes") !=
                    self.cfg["kv_cache_memory_bytes"])):
            raise RuntimeError("KV cache memory request differs from its generated manifest")
        admission_configured = self.admission_contract is not None
        if (admission_configured != ("bounded_admission" in manifest)
                or (admission_configured
                    and manifest.get("bounded_admission") != self.admission_contract)):
            raise RuntimeError(
                "bounded-admission request differs from its generated manifest")
        expected_env = {
            "TP4_RESILIENCE_CAMPAIGN_ID": self.cfg["campaign_id"],
            "TP4_RESILIENCE_CACHE_ROOT": self.cfg["container_cache_root"],
            "TP4_RESILIENCE_MAX_BYTES": str(MAX_NAMESPACE_BYTES),
            "TP4_RESILIENCE_BASE_SHA256": manifest["base_connector_sha256"],
            "SPARK_CONTEXT_CACHE_MAX_BYTES": str(TEST_CACHE_MAX_BYTES),
            "SPARK_CONTEXT_CACHE_LOW_WATERMARK_BYTES": str(TEST_CACHE_LOW_WATERMARK_BYTES),
        }
        destinations = {"/opt/tp4-resilience/base_connector.py",
                        "/opt/tp4-resilience/runtime.py"}
        for rank in range(4):
            inspected = self._ssh(rank, "sudo", "-n", "docker", "inspect", self.cfg["container"])
            if inspected.returncode:
                raise RuntimeError(f"rank {rank} docker inspect failed")
            value = json.loads(inspected.stdout)[0]
            environment_rows = value["Config"].get("Env", [])
            environment = dict(item.split("=", 1) for item in environment_rows if "=" in item)
            mounts = {item["Destination"]: item for item in value.get("Mounts", [])}
            kv_values: list[str] | None = None
            max_model_values: list[str] | None = None
            if kv_configured:
                command = value["Config"].get("Cmd")
                kv_values = _command_option_values(command, "--kv-cache-memory-bytes")
                max_model_values = _command_option_values(command, "--max-model-len")
                kv_alias_values = _command_option_values(command, "--kv-cache-memory")
                if kv_values != [str(self.cfg["kv_cache_memory_bytes"])]:
                    raise RuntimeError(
                        f"rank {rank} KV cache memory argument is missing, duplicated, or differs")
                if kv_alias_values != []:
                    raise RuntimeError(
                        f"rank {rank} unsupported KV cache memory alias is present")
                if max_model_values != ["262144"]:
                    raise RuntimeError(
                        f"rank {rank} maximum model length is missing, duplicated, or differs")
            admission_actual = None
            if admission_configured:
                command = value["Config"].get("Cmd")
                middleware_values = _command_option_values(command, "--middleware")
                if middleware_values != [ADMISSION_IMPORT]:
                    raise RuntimeError(
                        f"rank {rank} bounded-admission middleware argument differs")
                for key, expected in ADMISSION_ENVIRONMENT.items():
                    entries = [item for item in environment_rows
                               if item.startswith(key + "=")]
                    if entries != [f"{key}={expected}"]:
                        raise RuntimeError(
                            f"rank {rank} bounded-admission environment differs")
                admission_mounts = [item for item in value.get("Mounts", [])
                                    if item.get("Destination") == ADMISSION_TARGET]
                admission_suffix = (
                    f"/scripts/resilience/.campaign/{self.cfg['campaign_id']}/"
                    f"{ADMISSION_STAGED_NAME}")
                if (len(admission_mounts) != 1
                        or not str(admission_mounts[0].get("Source", "")).endswith(
                            admission_suffix)):
                    raise RuntimeError(f"rank {rank} bounded-admission mount differs")
                admission_hash = self._ssh(
                    rank, "sudo", "-n", "docker", "exec", self.cfg["container"],
                    "sha256sum", ADMISSION_TARGET)
                admission_actual = (
                    admission_hash.stdout.split()[0]
                    if admission_hash.returncode == 0 and admission_hash.stdout.split()
                    else None)
                if admission_actual != self.admission_contract["source_sha256"]:
                    raise RuntimeError(
                        f"rank {rank} in-container bounded-admission hash differs")
            if any(environment.get(key) != expected for key, expected in expected_env.items()):
                raise RuntimeError(f"rank {rank} resilience environment differs")
            if ((uses_trim and environment.get("VLLM_PREFILL_CACHE_TRIM") != "1")
                    or not uses_trim and "VLLM_PREFILL_CACHE_TRIM" in environment):
                raise RuntimeError(f"rank {rank} runtime variant environment differs")
            if ((uses_step_cap
                 and environment.get("VLLM_RESILIENCE_STEP_TOKEN_CAP") != "6912")
                    or not uses_step_cap
                    and "VLLM_RESILIENCE_STEP_TOKEN_CAP" in environment):
                raise RuntimeError(f"rank {rank} step-cap environment differs")
            if not destinations.issubset(mounts):
                raise RuntimeError(f"rank {rank} resilience mounts are incomplete")
            worker_mounts = [item for item in value.get("Mounts", [])
                             if item.get("Destination") == WORKER_TARGET]
            worker_mount = worker_mounts[0] if len(worker_mounts) == 1 else None
            if (not isinstance(worker_mount, dict)
                    or not str(worker_mount.get("Source", "")).endswith(worker_source_suffix)):
                raise RuntimeError(f"rank {rank} runtime variant worker mount differs")
            worker_hash = self._ssh(rank, "sudo", "-n", "docker", "exec",
                                    self.cfg["container"], "sha256sum", WORKER_TARGET)
            worker_actual = (worker_hash.stdout.split()[0] if worker_hash.returncode == 0
                             and worker_hash.stdout.split() else None)
            if worker_actual != expected_worker_sha:
                raise RuntimeError(f"rank {rank} in-container worker hash differs")
            scheduler_actual = None
            if uses_step_cap:
                scheduler_mounts = [item for item in value.get("Mounts", [])
                                    if item.get("Destination") == SCHEDULER_TARGET]
                scheduler_mount = scheduler_mounts[0] if len(scheduler_mounts) == 1 else None
                scheduler_suffix = "/" + self.scheduler_contract["candidate_path"].removeprefix(
                    "scripts/node/")
                if (not isinstance(scheduler_mount, dict)
                        or not str(scheduler_mount.get("Source", "")).endswith(
                            scheduler_suffix)):
                    raise RuntimeError(f"rank {rank} step-cap scheduler mount differs")
                scheduler_hash = self._ssh(rank, "sudo", "-n", "docker", "exec",
                                           self.cfg["container"], "sha256sum",
                                           SCHEDULER_TARGET)
                scheduler_actual = (scheduler_hash.stdout.split()[0]
                                    if scheduler_hash.returncode == 0
                                    and scheduler_hash.stdout.split() else None)
                if scheduler_actual != self.scheduler_contract["candidate_sha256"]:
                    raise RuntimeError(f"rank {rank} in-container scheduler hash differs")
            connector_mounts = [item for item in value.get("Mounts", [])
                                if item.get("Destination", "").endswith(
                                    "/spark_context_cache_connector.py")]
            if len(connector_mounts) != 1 or not connector_mounts[0]["Source"].endswith("/connector_wrapper.py"):
                raise RuntimeError(f"rank {rank} wrapper mount differs")
            remote_dir = f"tp4/scripts/resilience/.campaign/{self.cfg['campaign_id']}"
            selected_hash = self._ssh(rank, "sha256sum", f"{remote_dir}/{selected_name}")
            selected_actual = (selected_hash.stdout.split()[0] if selected_hash.returncode == 0
                               and selected_hash.stdout.split() else None)
            if selected_actual != self.runtime_config_identity["tp4_env_sha256"]:
                raise RuntimeError(f"rank {rank} selected runtime overlay hash differs")
            for name, expected in manifest["files"].items():
                hashed = self._ssh(rank, "sha256sum", f"{remote_dir}/{name}")
                actual = hashed.stdout.split()[0] if hashed.returncode == 0 and hashed.stdout.split() else None
                if actual != expected:
                    raise RuntimeError(f"rank {rank} staged hash differs for {name}")
            status = self._faultctl(rank, ["status"])
            if not status.get("within_cap"):
                raise RuntimeError(f"rank {rank} namespace already exceeds cap")
            rank_evidence = {"rank": rank, "environment": True,
                             "mounts": True, "staged_hashes": True,
                             "runtime_variant": variant,
                             "tp4_env_sha256": selected_actual,
                             "worker_sha256": worker_actual,
                             "scheduler_sha256": scheduler_actual,
                             "admission_sha256": admission_actual,
                             "namespace_bytes": status.get("used_bytes")}
            if kv_configured:
                rank_evidence.update(
                    kv_cache_memory_bytes=int(kv_values[0]), max_model_len=262144)
            evidence["ranks"].append(rank_evidence)
        evidence["pass"] = True
        return evidence

    def run(self) -> int:
        if ("restoration" in self.state or "restoration_completed_ns" in self.state
                or self.state.get("status") not in {None, "running"}):
            previous = {"archived_ns": time.time_ns()}
            for key in ("status", "restoration", "restoration_completed_ns", "finished_ns",
                        "final_fault_capture", "targeted_cutoff_cleanup", "targeted_cutoff_proof",
                        "targeted_phase_cutoff_ns", "stop_reasons", "unhandled_error",
                        "run_start_cleanup", "overlay_preflight", "pre_soak_cleanup",
                        "runtime_config_identity", "planned_case_order",
                        "final_restore_selection", "execution_policy"):
                if key in self.state:
                    previous[key] = self.state.pop(key)
            if "runtime_config_identity" not in previous:
                previous["runtime_config_identity"] = {
                    "runtime_variant": "default",
                    "tp4_env": (f"scripts/resilience/.campaign/"
                                f"{self.cfg['campaign_id']}/delta.env"),
                    "tp4_env_sha256": None,
                    "candidate_worker_sha256": None,
                    "identity_recorded": False,
                    "note": "legacy protected-default attempt predates runtime identity receipts",
                }
            self.state.setdefault("previous_attempts", []).append(previous)
        self.state["runtime_config_identity"] = dict(self.runtime_config_identity)
        self.state["planned_case_order"] = [case["id"] for case in self.plan]
        selected = self.cfg.get("targeted_case_selection")
        execution_policy = {
            "targeted_case_selection": (list(selected) if selected is not None else None),
            "soak_stop_after_required_seconds": self.cfg.get(
                "soak_stop_after_required_seconds", False),
        }
        prior_policy = self.state.get("execution_policy")
        if isinstance(prior_policy, dict) and prior_policy != execution_policy:
            self.state.setdefault("execution_policy_history", []).append({
                **prior_policy, "superseded_ns": time.time_ns()})
        self.state["execution_policy"] = execution_policy
        self.state["status"] = "running"
        self.state.pop("unhandled_error", None)
        self.save()
        self.telemetry.start(0.2)
        fatal = False
        unhandled = False
        targeted_cutoff = False
        cutoff_cleanup_done = False
        queue_client_capacity: dict[str, Any] | None = None
        try:
            self.state["run_start_cleanup"] = self._reset_faults("run-start-cleanup")
            if self.state["run_start_cleanup"]["errors"]:
                fatal = True
            try:
                self.state["overlay_preflight"] = self.verify_overlay()
            except Exception as error:
                self.state["overlay_preflight"] = {"pass": False,
                    "error": f"{type(error).__name__}: {error}"}
                fatal = True
            self.save()
            for case in self.plan:
                if case["id"] == "rigmark-final":
                    continue
                if case["area"] == "duration":
                    prior = self.state["cases"].get(case["id"], {}).get("runs", [])
                    if self._successes_after_last_failure(
                            prior, "pass", self.runtime_config_identity,
                            SOAK_PROTOCOL_VERSION):
                        continue
                    self.state["pre_soak_cleanup"] = self._reset_faults("pre-soak")
                    if self.state["pre_soak_cleanup"]["errors"]:
                        fatal = True
                    if fatal or self.telemetry.safety_event.is_set():
                        status, evidence = "pending", {
                            "reason": "campaign stopped before soak",
                            "cleanup": self.state["pre_soak_cleanup"],
                        }
                    else:
                        try:
                            status, evidence = self._soak(case)
                        except Exception as error:
                            status, evidence = "fail", {
                                "error": f"{type(error).__name__}: {error}",
                                "reason": "soak raised before producing its normal receipt"}
                    self.record(case, 1, status, evidence)
                    fatal = fatal or status == "fail" or self.telemetry.safety_event.is_set()
                    continue
                if (selected is not None and not case.get("manual")
                        and case["id"] not in selected):
                    row = self.state["cases"].setdefault(case["id"], {
                        "area": case["area"], "parameters": case.get("parameters", {}),
                        "target_repetitions": case["target_repetitions"], "runs": []})
                    row["current_attempt_selection"] = {
                        "selected": False, "status": "pending",
                        "reason": "not selected for this targeted execution attempt",
                    }
                    continue
                if targeted_cutoff:
                    continue
                if (case["area"] == "queue" and queue_client_capacity is not None
                        and case["parameters"]["clients"]
                        > queue_client_capacity["clients"]):
                    repetition = 1 + max((run.get("repetition", 0) for run in
                                          self.state["cases"].get(
                                              case["id"], {}).get("runs", [])), default=0)
                    self.record(case, repetition, "pending", {
                        "reason": (
                            "not attempted after a smaller wave proved a local client "
                            "capacity limit"),
                        "not_attempted": True,
                        "client_capacity_boundary": dict(queue_client_capacity),
                    })
                    continue
                row = self.state["cases"].get(case["id"], {})
                existing = row.get("runs", [])
                qualifying = "recorded" if case.get("manual") else "pass"
                identity = None if case.get("manual") else self.runtime_config_identity
                protocol = _targeted_case_protocol(case["id"])
                successes = self._successes_after_last_failure(
                    existing, qualifying, identity, protocol)
                for _ in range(max(0, case["target_repetitions"] - successes)):
                    repetition = 1 + max((run.get("repetition", 0) for run in
                                          self.state["cases"].get(case["id"], {}).get("runs", [])), default=0)
                    cutoff = time.monotonic() >= self.targeted_stop_at
                    if cutoff:
                        targeted_cutoff = True
                        self.state["targeted_phase_cutoff_ns"] = time.time_ns()
                        cutoff_proof: dict[str, Any] = {}
                        if not cutoff_cleanup_done:
                            cleanup = self._reset_faults("targeted-cutoff-cleanup")
                            self.state["targeted_cutoff_cleanup"] = cleanup
                            cutoff_proof["cleanup"] = cleanup
                            cutoff_cleanup_done = True
                            fatal = fatal or bool(cleanup["errors"])
                        self.active_deadline = self.soak_at
                        cutoff_proof["health_200"] = self.health_code() == 200
                        idle_status, idle = self.wait_for_api_idle()
                        cutoff_proof["idle"] = {"status": idle_status,
                                                "evidence": idle}
                        try:
                            barriers = self._drain_barriers()
                            drain_status, drain = self.wait_for_drain(barriers)
                            cutoff_proof.update(barriers=barriers,
                                                drain_status=drain_status, drain=drain)
                        except Exception as error:
                            cutoff_proof.update(drain_status="fail",
                                                error=f"{type(error).__name__}: {error}")
                        cutoff_proof["pass"] = (
                            cutoff_proof["health_200"] is True
                            and idle_status == "pass"
                            and cutoff_proof.get("drain_status") == "pass"
                            and not cutoff_proof.get("cleanup", {}).get("errors"))
                        self.state["targeted_cutoff_proof"] = cutoff_proof
                        fatal = fatal or not cutoff_proof["pass"]
                        self.save()
                        break
                    if self.telemetry.safety_event.is_set() or fatal:
                        self.record(case, repetition, "pending", {"reason": "campaign cutoff or stop condition"})
                        fatal = fatal or self.telemetry.safety_event.is_set()
                        break
                    try:
                        status, evidence = self.execute_case(case, repetition)
                    except Exception as error:
                        if (getattr(error, "guard_abort_cause", None) == "phase_deadline"
                                and time.monotonic() >= self.targeted_stop_at
                                and not self.telemetry.safety_event.is_set()):
                            status, evidence = "pending", {
                                "reason": "targeted phase deadline aborted the in-flight case",
                                "error": f"{type(error).__name__}: {error}"}
                            targeted_cutoff = True
                        else:
                            status, evidence = "fail", {"error": f"{type(error).__name__}: {error}"}
                    if not case.get("manual") and status in {"pass", "pending"}:
                        if targeted_cutoff or time.monotonic() >= self.targeted_stop_at:
                            targeted_cutoff = True
                            if not cutoff_cleanup_done:
                                evidence["cutoff_cleanup"] = self._reset_faults(
                                    "targeted-cutoff-cleanup")
                                cutoff_cleanup_done = True
                            self.active_deadline = self.soak_at
                            evidence["post_cutoff_health_200"] = self.health_code() == 200
                            idle_status, idle = self.wait_for_api_idle()
                            evidence["post_cutoff_idle"] = {
                                "status": idle_status, "evidence": idle}
                        try:
                            barriers = self._drain_barriers()
                            drain_status, drain = self.wait_for_drain(barriers)
                        except Exception as error:
                            drain_status = "fail"
                            drain = {"reason": "post-case drain barrier failed",
                                     "error": f"{type(error).__name__}: {error}"}
                        if not targeted_cutoff and time.monotonic() >= self.targeted_stop_at:
                            targeted_cutoff = True
                            if not cutoff_cleanup_done:
                                evidence["cutoff_cleanup"] = self._reset_faults(
                                    "targeted-cutoff-cleanup")
                                cutoff_cleanup_done = True
                            self.active_deadline = self.soak_at
                            evidence["post_cutoff_health_200"] = self.health_code() == 200
                            idle_status, idle = self.wait_for_api_idle()
                            evidence["post_cutoff_idle"] = {
                                "status": idle_status, "evidence": idle}
                            if drain_status != "pass":
                                try:
                                    barriers = self._drain_barriers()
                                    drain_status, drain = self.wait_for_drain(barriers)
                                except Exception as error:
                                    drain_status = "fail"
                                    drain = {"reason": "post-cutoff drain retry failed",
                                             "error": f"{type(error).__name__}: {error}"}
                        evidence["reservation_drain"] = drain
                        if (status == "pending" and evidence.get(
                                "client_capacity_limited") is True
                                and drain_status != "pass"):
                            status = "fail"
                            evidence["reason"] = (
                                "client capacity was reached but a clean post-wave drain "
                                "was not proven")
                        elif (drain_status == "fail"
                              or status == "pass" and drain_status != "pass"):
                            status = drain_status
                        if (case["area"] == "queue" and status == "pending"
                                and evidence.get("client_capacity_limited") is True
                                and drain_status == "pass"):
                            queue_client_capacity = {
                                "case_id": case["id"],
                                "clients": case["parameters"]["clients"],
                                "repetition": repetition,
                            }
                            evidence["higher_queue_waves_deferred"] = True
                        if targeted_cutoff and (drain_status != "pass"
                                or evidence.get("post_cutoff_health_200") is not True
                                or (evidence.get("post_cutoff_idle") or {}).get(
                                    "status") != "pass"
                                or evidence.get("cutoff_cleanup", {}).get("errors")):
                            status = "fail"
                            evidence["reason"] = (
                                "planned cutoff did not prove a healthy drained service before soak")
                    self.record(case, repetition, status, evidence)
                    if case["id"] == "rigmark-initial" and status != "recorded":
                        fatal = True
                    if status == "fail":
                        fatal = True
                        break
                    if (case["area"] == "queue"
                            and evidence.get("client_capacity_limited") is True):
                        break
                if fatal:
                    break
            # Every case is present even when the fixed deadline left no attempt.
            for case in self.plan:
                row = self.state["cases"].setdefault(case["id"], {"area": case["area"],
                    "parameters": case.get("parameters", {}), "target_repetitions": case["target_repetitions"],
                    "runs": []})
                if case["area"] == "duration":
                    row["active_parameters"] = dict(case.get("parameters", {}))
                qualifying = "recorded" if case.get("manual") else "pass"
                identity = None if case.get("manual") else self.runtime_config_identity
                protocol = (SOAK_PROTOCOL_VERSION if case["area"] == "duration"
                            else _targeted_case_protocol(case["id"]))
                successes = self._successes_after_last_failure(
                    row["runs"], qualifying, identity, protocol)
                failures = sum(run.get("status") == "fail" for run in row["runs"])
                row["successful_repetitions"] = successes
                row["remaining_successes"] = max(0, case["target_repetitions"] - successes)
                row["historical_failures"] = failures
                if row["remaining_successes"]:
                    row["status"] = "fail" if failures else "pending"
                    if (selected is not None and not case.get("manual")
                            and case["area"] != "duration" and case["id"] not in selected):
                        row["pending_reason"] = (
                            "not selected for this targeted execution attempt")
                else:
                    row["status"] = "pass_with_failures" if failures else qualifying
            self.state["status"] = "stopping"
            self.state["stop_reasons"] = list(self.telemetry.reasons)
            self.save()
        except BaseException as error:
            fatal = True
            unhandled = True
            self.state["status"] = "failed_unhandled"
            self.state["unhandled_error"] = f"{type(error).__name__}: {error}"
            self.state["stop_reasons"] = list(self.telemetry.reasons)
            self.save()
            raise
        finally:
            self.state["stop_reasons"] = list(self.telemetry.reasons)
            if fatal or self.telemetry.safety_event.is_set():
                try:
                    self.state["final_fault_capture"] = self.preserve_cluster_evidence(
                        "campaign-stop-before-restoration")
                except Exception as error:
                    self.state["final_fault_capture"] = {
                        "status": "error", "error": f"{type(error).__name__}: {error}"}
                    self.save()
            self.state["final_restore_selection"] = self._safe_final_restore_selection(fatal)
            self.save()
            self.telemetry.stop()
            try:
                self.state["restoration"] = self.restore_default()
            except Exception as error:
                self.state["restoration"] = {"error": f"{type(error).__name__}: {error}"}
            restoration = self.state["restoration"]
            self.state["restoration_completed_ns"] = time.time_ns()
            self.save()
            final_case = next(case for case in self.plan if case["id"] == "rigmark-final")
            final_receipt = Path(self.cfg["rigmark"]["final"])
            def final_is_fresh() -> bool:
                return self._receipt_evidence(
                    final_receipt, self.state["restoration_completed_ns"])[0]
            while ((not fatal or time.monotonic() >= self.soak_at)
                   and restoration.get("gates", {}).get("pass")
                   and restoration.get("identity", {}).get("returncode") == 0
                   and time.monotonic() < self.end_monotonic and not final_is_fresh()):
                time.sleep(min(5, max(0, self.end_monotonic - time.monotonic())))
            final_status, final_evidence = self.execute_case(final_case, 1)
            self.record(final_case, 1, final_status, final_evidence)
            final_row = self.state["cases"][final_case["id"]]
            final_successes = self._successes_after_last_failure(final_row["runs"], "recorded")
            final_failures = sum(run.get("status") == "fail" for run in final_row["runs"])
            final_row["successful_repetitions"] = final_successes
            final_row["remaining_successes"] = max(0, 1 - final_successes)
            final_row["historical_failures"] = final_failures
            if final_successes:
                final_row["status"] = "pass_with_failures" if final_failures else "recorded"
            else:
                final_row["status"] = "fail" if final_failures else "pending"
            restored = (restoration.get("gates", {}).get("pass")
                        and restoration.get("identity", {}).get("returncode") == 0)
            all_required = all(row.get("remaining_successes", 1) == 0
                               for row in self.state["cases"].values())
            any_failures = any(row.get("historical_failures", 0) > 0
                               for row in self.state["cases"].values())
            if not restored:
                self.state["status"] = "restoration_failed"
            elif final_status != "recorded":
                self.state["status"] = "incomplete_final_rigmark"
            elif unhandled:
                self.state["status"] = "failed_unhandled"
            elif not all_required:
                self.state["status"] = "incomplete"
            elif any_failures:
                self.state["status"] = "complete_with_failures"
            else:
                self.state["status"] = "complete"
            self.state["finished_ns"] = time.time_ns()
            self.save()
        return 0 if self.state["status"] == "complete" else 1


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", required=True)
    result.add_argument("--plan", action="store_true", help="validate and print the matrix without network access")
    result.add_argument("--execute", action="store_true")
    result.add_argument("--ack-exclusive-production-window", action="store_true")
    result.add_argument("--ack-service-faults", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        cfg = validate_config(load_json(Path(args.config)))
        plan = build_plan(cfg)
        if args.plan or not args.execute:
            print(json.dumps({"campaign_id": cfg["campaign_id"], "cases": plan,
                              "case_count": len(plan)}, indent=2, sort_keys=True))
            return 0
        if not args.ack_exclusive_production_window:
            raise ContractError("--execute requires --ack-exclusive-production-window")
        if not args.ack_service_faults:
            raise ContractError("--execute requires --ack-service-faults")
        return Campaign(cfg, acknowledge_faults=args.ack_service_faults).run()
    except (ContractError, OSError, ValueError) as error:
        print(f"campaign: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
