"""Shared I/O for the fidelity harness: manifests, HTTP retry, logprob parsing, storage.

Everything except the npz helpers is stdlib-only so the offline tests run with the
system interpreter. Nothing here prints or stores token text: only ids and numbers.
"""

from __future__ import annotations

import array
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request


REPO = Path(__file__).resolve().parents[2]
NAN = float("nan")
NEG_INF = float("-inf")


# ---------------------------------------------------------------- manifests


class ManifestError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def global_sha256(entries: list[dict]) -> str:
    text = "".join(f"{e['id']}:{e['sha256']}\n" for e in sorted(entries, key=lambda e: e["id"]))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_manifest(path: Path, key: str = "windows") -> tuple[dict, list[dict]]:
    """Load a corpus or decode manifest and check its global hash."""
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = manifest.get(key)
    if not isinstance(entries, list):
        raise ManifestError(f"{path}: missing list '{key}'")
    ids = [e["id"] for e in entries]
    if len(set(ids)) != len(ids):
        raise ManifestError(f"{path}: duplicate ids")
    expected = manifest.get("global_sha256")
    if expected and global_sha256(entries) != expected:
        raise ManifestError(f"{path}: global_sha256 mismatch")
    return manifest, entries


def read_tokens(path: Path) -> list[int]:
    """Read a raw little-endian uint32 token file."""
    data = Path(path).read_bytes()
    if len(data) % 4:
        raise ManifestError(f"{path}: size is not a multiple of 4")
    values = array.array("I")
    if values.itemsize != 4:
        raise ManifestError("platform array('I') is not 32-bit")
    values.frombytes(data)
    if sys.byteorder == "big":
        values.byteswap()
    return values.tolist()


def load_entry_tokens(manifest_path: Path, entry: dict) -> list[int]:
    """Read one window/prompt, verifying its sha256 and length before use."""
    path = Path(manifest_path).parent / entry["path"]
    if sha256_file(path) != entry["sha256"]:
        raise ManifestError(f"{entry['id']}: sha256 mismatch")
    tokens = read_tokens(path)
    if "n_tokens" in entry and len(tokens) != entry["n_tokens"]:
        raise ManifestError(f"{entry['id']}: n_tokens mismatch")
    return tokens


def select_entries(entries: list[dict], subset: Path | None = None,
                   categories: list[str] | None = None, limit: int | None = None) -> list[dict]:
    chosen = entries
    if subset:
        wanted = {line.strip() for line in Path(subset).read_text().splitlines()
                  if line.strip() and not line.startswith("#")}
        chosen = [e for e in chosen if e["id"] in wanted]
    if categories:
        chosen = [e for e in chosen if e.get("category") in set(categories)]
    chosen = sorted(chosen, key=lambda e: e["id"])
    if limit is not None:
        chosen = chosen[:limit]
    return chosen


# ---------------------------------------------------------------- HTTP


class ClientError(RuntimeError):
    """A 4xx answer: recorded and skipped, never retried."""

    def __init__(self, status: int, message: str, attempts: int = 1):
        super().__init__(f"HTTP {status}")
        self.status = status
        self.message = message
        self.attempts = attempts


class TransientError(RuntimeError):
    """Connection failure or 5xx after all retries."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class Aborted(RuntimeError):
    pass


def _error_message(body: bytes) -> str:
    """Server error message (numbers and fixed text), capped; never the request."""
    try:
        data = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return ""
    if isinstance(data, dict):
        err = data.get("error", data)
        if isinstance(err, dict):
            text = err.get("message") or err.get("detail") or ""
        else:
            text = str(err)
        return str(text)[:300]
    return ""


def http_json(url: str, body: dict | None = None, timeout: float = 600.0) -> dict:
    """One request; raises urllib/socket errors or HTTPError unchanged."""
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="GET" if body is None else "POST",
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def post_with_retry(url: str, body, *, retries: int = 5, backoff: float = 2.0,
                    timeout: float = 600.0, abort_file: Path | None = None,
                    sleep=time.sleep) -> tuple[dict, int]:
    """POST with exponential backoff on connection errors and 5xx. Returns (json, attempts).

    `body` may be a callable that builds the body per attempt (a fresh cache_salt each).
    """
    attempt = 0
    while True:
        attempt += 1
        status = None
        try:
            return http_json(url, body() if callable(body) else body, timeout), attempt
        except urllib.error.HTTPError as error:
            status = error.code
            try:
                payload = error.read()
            finally:
                error.close()
            if 400 <= status < 500:
                raise ClientError(status, _error_message(payload), attempt) from None
            reason = f"HTTP {status}"
        except (urllib.error.URLError, ConnectionError, socket.timeout, TimeoutError,
                OSError, ValueError) as error:
            reason = type(error).__name__
        if attempt > retries:
            raise TransientError(f"{reason} after {attempt} attempts", status)
        if abort_file and Path(abort_file).exists():
            raise Aborted(str(abort_file))
        sleep(backoff * (2 ** (attempt - 1)))


def server_model_id(base_url: str, timeout: float = 30.0) -> str:
    data = http_json(base_url.rstrip("/") + "/v1/models", None, timeout)
    return data["data"][0]["id"]


def redact_host(base_url: str) -> str:
    parts = urllib.parse.urlsplit(base_url)
    netloc = "<host>" + (f":{parts.port}" if parts.port else "")
    return urllib.parse.urlunsplit((parts.scheme, netloc, parts.path, "", ""))


def fresh_salt() -> str:
    return secrets.token_hex(32)


def salt_hash(salt: str) -> str:
    return hashlib.sha256(salt.encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def git_identity() -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                                  text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    return {"commit": run("rev-parse", "HEAD") or None,
            "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


# ---------------------------------------------------------------- parsing


class ParseError(RuntimeError):
    pass


def token_id(key, fallback=None) -> int:
    """Parse 123, "123" or "token_id:123" (optionally via a decoded_token fallback)."""
    for candidate in (key, fallback):
        if isinstance(candidate, bool):
            continue
        if isinstance(candidate, int):
            return candidate
        if isinstance(candidate, str):
            text = candidate.strip()
            if text.startswith("token_id:"):
                text = text[len("token_id:"):]
            if text.lstrip("-").isdigit():
                return int(text)
    raise ParseError("unparseable token key")


def _float(value) -> float:
    if value is None:
        return NAN
    return float(value)


def _topk_rows(entries: list[tuple[int, float, int | None]], k: int) -> tuple[list[int], list[float]]:
    """Top-k by rank when known, else by logprob; padded with -1 / -inf."""
    ordered = sorted(entries, key=lambda e: (e[2] if e[2] is not None and e[2] > 0 else math.inf,
                                             -e[1] if not math.isnan(e[1]) else math.inf))
    ids = [e[0] for e in ordered[:k]]
    lps = [e[1] for e in ordered[:k]]
    return ids + [-1] * (k - len(ids)), lps + [NEG_INF] * (k - len(lps))


def parse_prompt_logprobs(response: dict, ids: list[int], k: int) -> dict:
    """Convert vLLM `prompt_logprobs` into per-position lists matching the npz schema."""
    choice = (response.get("choices") or [{}])[0]
    rows = choice.get("prompt_logprobs", response.get("prompt_logprobs"))
    if not isinstance(rows, list):
        raise ParseError("missing prompt_logprobs")
    n = len(ids)
    if len(rows) != n:
        raise ParseError(f"prompt_logprobs length {len(rows)} != {n}")
    lp_actual, rank_actual = [NAN], [-1]
    topk_ids, topk_lp = [[-1] * k], [[NEG_INF] * k]
    missing = 0
    for position in range(1, n):
        row = rows[position]
        if not isinstance(row, dict):
            raise ParseError(f"position {position}: not a dict")
        entries = []
        for key, value in row.items():
            if isinstance(value, dict):
                tid = token_id(key, value.get("decoded_token"))
                rank = value.get("rank")
                entries.append((tid, _float(value.get("logprob")),
                                int(rank) if isinstance(rank, (int, float)) else None))
            else:
                entries.append((token_id(key), _float(value), None))
        actual = ids[position]
        hit = [e for e in entries if e[0] == actual]
        if hit:
            lp_actual.append(hit[0][1])
            rank_actual.append(hit[0][2] if hit[0][2] is not None else -1)
        else:
            missing += 1
            lp_actual.append(NAN)
            rank_actual.append(-1)
        row_ids, row_lp = _topk_rows(entries, k)
        topk_ids.append(row_ids)
        topk_lp.append(row_lp)
    return {"ids": list(ids), "lp_actual": lp_actual, "rank_actual": rank_actual,
            "topk_ids": topk_ids, "topk_lp": topk_lp, "missing_actual": missing}


def parse_generation_logprobs(response: dict, k: int) -> dict:
    """Convert completions `logprobs` (tokens as token_id:N) into per-position lists."""
    choice = (response.get("choices") or [None])[0]
    if not isinstance(choice, dict) or not isinstance(choice.get("logprobs"), dict):
        raise ParseError("missing logprobs")
    logprobs = choice["logprobs"]
    tokens = logprobs.get("tokens") or []
    token_lps = logprobs.get("token_logprobs") or []
    tops = logprobs.get("top_logprobs") or []
    if not (len(tokens) == len(token_lps) == len(tops)):
        raise ParseError("logprobs arrays differ in length")
    gen_ids = [token_id(t) for t in tokens]
    topk_ids, topk_lp = [], []
    for top in tops:
        entries = [(token_id(key), _float(value), None) for key, value in (top or {}).items()]
        row_ids, row_lp = _topk_rows(entries, k)
        topk_ids.append(row_ids)
        topk_lp.append(row_lp)
    return {"gen_ids": gen_ids, "lp_actual": [_float(v) for v in token_lps],
            "topk_ids": topk_ids, "topk_lp": topk_lp,
            "finish_reason": choice.get("finish_reason"), "usage": response.get("usage")}


# ---------------------------------------------------------------- storage


PROMPT_ARRAYS = {"ids": "int32", "lp_actual": "float32", "rank_actual": "int32",
                 "topk_ids": "int32", "topk_lp": "float32"}
GEN_ARRAYS = {"gen_ids": "int32", "lp_actual": "float32", "topk_ids": "int32",
              "topk_lp": "float32"}


def _umask_mode() -> int:
    current = os.umask(0)
    os.umask(current)
    return 0o666 & ~current


def _atomic(path: Path, writer, mode: int | None = None) -> None:
    """Write via a temporary file and rename; default mode follows the umask."""
    mode = _umask_mode() if mode is None else mode
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def write_json_atomic(path: Path, data, mode: int | None = None) -> None:
    payload = (json.dumps(data, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic(path, lambda handle: handle.write(payload), mode)


def read_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def save_npz(path: Path, data: dict, schema: dict) -> None:
    import numpy as np
    arrays = {name: np.asarray(data[name], dtype=dtype) for name, dtype in schema.items()}
    _atomic(path, lambda handle: np.savez(handle, **arrays))


def validate_npz(path: Path, schema: dict, length: int | None, k: int) -> bool:
    """True when the npz loads and matches the expected dtypes and shapes."""
    try:
        import numpy as np
        with np.load(path) as data:
            first = next(iter(schema))
            n = data[first].shape[0] if length is None else length
            for name, dtype in schema.items():
                arr = data[name]
                if arr.dtype != np.dtype(dtype) or arr.shape[0] != n:
                    return False
                if name.startswith("topk") and arr.shape != (n, k):
                    return False
        return True
    except Exception:
        return False


# ---------------------------------------------------------------- collection loop


def ensure_run_json(run_dir: Path, fields: dict, boot_json: Path | None) -> dict:
    """Create run.json or check that a resumed run keeps its arm, K and model."""
    path = Path(run_dir) / "run.json"
    existing = read_json(path)
    now = utc_now()
    if existing:
        for key in ("arm", "run", "K", "kind", "server_model_id"):
            if key in fields and existing.get(key) != fields[key]:
                raise SystemExit(f"run.json {key} mismatch: {existing.get(key)!r} != {fields[key]!r}")
        record = existing
    else:
        record = dict(fields, started_utc=now, sessions=[])
        if boot_json:
            record["boot"] = json.loads(Path(boot_json).read_text(encoding="utf-8"))
    record["sessions"].append({"started_utc": now, "ended_utc": None, "harness_git": git_identity()})
    record["harness_git"] = record["sessions"][-1]["harness_git"]
    write_json_atomic(path, record)
    return record


def close_run_json(run_dir: Path, record: dict, counts: dict) -> None:
    now = utc_now()
    record["ended_utc"] = now
    record["sessions"][-1].update(ended_utc=now, counts=counts)
    write_json_atomic(Path(run_dir) / "run.json", record)


def run_items(items: list, process, is_done, abort_file: Path | None, out=print) -> dict:
    """Run `process(index, item)` for each pending item; stop between items on abort.

    `process` returns a status string. Counts are returned; `aborted` is set when the
    abort file appeared (the caller exits non-zero).
    """
    counts = {"ok": 0, "skipped": 0, "failed": 0, "aborted": False}
    total = len(items)
    for index, item in enumerate(items, 1):
        if is_done(item):
            counts["skipped"] += 1
            continue
        if abort_file and Path(abort_file).exists():
            out(f"abort: {abort_file} present; stopping before {item['id']} ({index}/{total})")
            counts["aborted"] = True
            break
        try:
            status = process(index, item)
        except Aborted:
            out(f"abort: {abort_file} present during retry of {item['id']}")
            counts["aborted"] = True
            break
        counts["ok" if status == "ok" else "failed"] += 1
    return counts


def request_and_record(url: str, body: dict, sidecar_base: dict, *, retries: int, backoff: float,
                       timeout: float, abort_file: Path | None, fixed_salt: str | None = None,
                       sleep=time.sleep):
    """POST and return (response or None, sidecar dict). Aborted propagates.

    Each HTTP attempt carries a fresh cache_salt unless `fixed_salt` is given (replay
    pairs); the sidecar records only the sha256 of the salt of the last attempt.
    """
    started = time.monotonic()
    sidecar = dict(sidecar_base)
    last = {}

    def make_body():
        last["salt"] = fixed_salt or fresh_salt()
        return dict(body, cache_salt=last["salt"])

    try:
        response, attempts = post_with_retry(url, make_body, retries=retries, backoff=backoff,
                                             timeout=timeout, abort_file=abort_file, sleep=sleep)
        sidecar.update(status="ok", http_attempts=attempts)
    except ClientError as error:
        response = None
        sidecar.update(status="client_error", http_status=error.status, http_attempts=error.attempts,
                       error_message=error.message)
    except TransientError as error:
        response = None
        sidecar.update(status="transient_error", http_status=error.status,
                       http_attempts=retries + 1, error_message=str(error))
    sidecar["elapsed_s"] = round(time.monotonic() - started, 3)
    if "salt" in last:
        sidecar["cache_salt_sha256"] = salt_hash(last["salt"])
    return response, sidecar
