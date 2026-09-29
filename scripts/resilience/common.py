#!/usr/bin/env python3
"""Shared, dependency-free helpers for the private resilience campaign."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any


SCHEMA = "tp4-resilience/v1"
MAX_NAMESPACE_BYTES = 8 << 30
MIN_MEM_AVAILABLE_BYTES = 768 << 20
CAMPAIGN_RE = re.compile(r"[a-z0-9][a-z0-9-]{2,47}")


class ContractError(ValueError):
    """The campaign configuration violates a fail-closed safety contract."""


def validate_campaign_id(value: str) -> str:
    if not isinstance(value, str) or CAMPAIGN_RE.fullmatch(value) is None:
        raise ContractError("campaign_id must match [a-z0-9][a-z0-9-]{2,47}")
    return value


def namespace_name(campaign_id: str) -> str:
    return f"tp4-resilience-{validate_campaign_id(campaign_id)}"


def validate_container_root(value: str, campaign_id: str) -> str:
    expected = f"/cache/jit/{namespace_name(campaign_id)}"
    if value != expected:
        raise ContractError(f"container cache root must be exactly {expected}")
    return value


def require_private_dir(path: Path, *, create: bool = False) -> Path:
    path = path.expanduser().resolve()
    if create:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(path, 0o700)
    if not path.is_dir():
        raise ContractError(f"private output directory does not exist: {path}")
    if path.stat().st_mode & 0o077:
        raise ContractError(f"private output directory must have mode 0700: {path}")
    return path


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any, *, mode: int = 0o600) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_json(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def append_event(path: Path, event: dict[str, Any]) -> None:
    payload = dict(event)
    payload.setdefault("schema", SCHEMA)
    payload.setdefault("time_ns", time.time_ns())
    encoded = canonical_json(payload) + b"\n"
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, encoded)
    finally:
        os.close(descriptor)


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot read valid JSON from {path}: {error}") from error


def tree_bytes(root: Path) -> int:
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
