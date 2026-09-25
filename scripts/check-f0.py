#!/usr/bin/env python3
"""Read-only operational identity check against the selected frozen baseline."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import datetime as dt
import ipaddress
import json
import math
import os
import re
import shlex
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable


REPO = Path(__file__).resolve().parents[1]
BASELINE = REPO / "docs/historical_benchmarks/baselines/2026-09-25-e28b/baseline.json"
ADAPTIVE_DEFAULTS = {
    "VLLM_ADAPTIVE_K_ENABLE": "1", "VLLM_ADAPTIVE_K_LO": "3",
    "VLLM_ADAPTIVE_K_HI": "5", "VLLM_ADAPTIVE_K_MODE": "per-request",
    "VLLM_ADAPTIVE_K_SEED": "1.0", "VLLM_ADAPTIVE_K_DOWN": "0.42",
    "VLLM_ADAPTIVE_K_UP": "0.58", "VLLM_ADAPTIVE_K_ALPHA": "0.15",
    "VLLM_ADAPTIVE_K_SIGNAL": "pos", "VLLM_ADAPTIVE_K_LOG_EVERY": "200",
    "VLLM_ADAPTIVE_K_RESPECT_DRAFT_BUDGET": "0",
}

LOAD_RECIPE = r'''
set -euo pipefail
repo=$1
TP4_LOG_TAG='[check-f0]'
. "$repo/scripts/lib/common.sh"
. "$repo/cluster.env"
tp4_check_env "$repo"
base_nodes="$NODES"; base_tp4_hosts="${TP4_HOSTS-}"; base_mgmt_ips="$MGMT_IPS"
base_hosts="${TP4_HOSTS:-$NODES}"; base_master_ip="$MASTER_IP"; base_master_port="$MASTER_PORT"
base_api_port="$API_PORT"; base_fabric_prefix_re="${FABRIC_PREFIX_RE:-}"
base_container="$CONTAINER"; base_model_dir="$MODEL_DIR"; base_draft_dir="$DRAFT_DIR"
base_served_name="$SERVED_NAME"; base_extra_docker_env="${EXTRA_DOCKER_ENV:-}"
base_extra_vllm_args="${EXTRA_VLLM_ARGS:-}"
for i in 0 1 2 3; do
  printf -v "base_fabric_target_$i" '%s' "${FABRIC_TARGETS[$i]}"
  printf -v "base_mgmt_if_$i" '%s' "$(tp4_resolve_rank_value "$i" MGMT_IF MGMT_IF_BY_RANK "$TP4_DEFAULT_MGMT_IF")"
  printf -v "base_fabric_ifaces_$i" '%s' "$(tp4_resolve_rank_value "$i" FABRIC_IFACES FABRIC_IFACES_BY_RANK "$TP4_DEFAULT_FABRIC_IFACES")"
  printf -v "base_hca_$i" '%s' "$(tp4_resolve_rank_value "$i" NCCL_IB_HCA NCCL_IB_HCA_BY_RANK "$TP4_DEFAULT_NCCL_IB_HCA")"
  for prefix in base_fabric_target base_mgmt_if base_fabric_ifaces base_hca; do
    readonly "${prefix}_$i"
  done
done
readonly base_nodes base_tp4_hosts base_mgmt_ips base_hosts base_master_ip base_master_port
readonly base_api_port base_fabric_prefix_re base_container base_model_dir base_draft_dir
readonly base_served_name base_extra_docker_env base_extra_vllm_args
tp4_load_env "$repo" --require --overlay
. "$repo/scripts/node/bootstrap/versions.env"
# A digest-pinned IMAGE (F1 lane) is its own pin; versions.env pins the tagged F0 image.
case "$IMAGE" in *@sha256:*) IMAGE_DIGEST=$IMAGE ;; esac
emit() { printf '%s\0%s\0' "$1" "$2"; }
emit base_nodes "$base_nodes"; emit base_tp4_hosts "$base_tp4_hosts"; emit base_mgmt_ips "$base_mgmt_ips"
emit base_hosts "$base_hosts"; emit base_master_ip "$base_master_ip"; emit base_master_port "$base_master_port"
emit base_api_port "$base_api_port"; emit base_fabric_prefix_re "$base_fabric_prefix_re"
emit base_container "$base_container"; emit base_model_dir "$base_model_dir"
emit base_draft_dir "$base_draft_dir"; emit base_served_name "$base_served_name"
emit base_extra_docker_env "$base_extra_docker_env"
emit base_extra_vllm_args "$base_extra_vllm_args"
emit nodes "$NODES"; emit tp4_hosts "${TP4_HOSTS-}"; emit mgmt_ips "$MGMT_IPS"
emit hosts "${TP4_HOSTS:-$NODES}"
emit master_ip "$MASTER_IP"; emit master_port "$MASTER_PORT"; emit api_port "$API_PORT"
emit container "$CONTAINER"; emit image "$IMAGE"; emit image_digest "$IMAGE_DIGEST"
emit model_dir "$MODEL_DIR"; emit draft_dir "$DRAFT_DIR"
emit model_repo "$MODEL_REPO"; emit model_rev "$MODEL_REV"; emit draft_rev "$DRAFT_REV"
emit served_name "$SERVED_NAME"; emit max_model_len "$MAX_MODEL_LEN"
emit max_num_seqs "$MAX_NUM_SEQS"; emit kv_cache_dtype "$KV_CACHE_DTYPE"
emit batched_tokens "$BATCHED_TOKENS"; emit spec_tokens "$SPEC_TOKENS"
emit spec_extra_json "${SPEC_EXTRA_JSON:-}"; emit async_scheduling "$ASYNC_SCHEDULING"
emit extra_docker_env "${EXTRA_DOCKER_ENV:-}"; emit extra_vllm_args "${EXTRA_VLLM_ARGS:-}"
emit sparkcache_mode "${SPARKCACHE_MODE:-off}"
emit spark_mhc_prefill_shard "${SPARK_MHC_PREFILL_SHARD:-0}"
emit sparkcache_config_sha256 "${SPARKCACHE_CONFIG_SHA256:-}"
emit sparkcache_connector_sha256 "${SPARKCACHE_CONNECTOR_SHA256:-}"
emit sparkcache_encoder_sha256 "${SPARKCACHE_ENCODER_SHA256:-}"
emit fabric_prefix_re "${FABRIC_PREFIX_RE:-}"
for i in 0 1 2 3; do
  for prefix in base_fabric_target base_mgmt_if base_fabric_ifaces base_hca; do
    name="${prefix}_$i"; emit "$name" "${!name}"
  done
  emit "fabric_target_$i" "${FABRIC_TARGETS[$i]}"
  emit "mgmt_if_$i" "$(tp4_resolve_rank_value "$i" MGMT_IF MGMT_IF_BY_RANK "$TP4_DEFAULT_MGMT_IF")"
  emit "fabric_ifaces_$i" "$(tp4_resolve_rank_value "$i" FABRIC_IFACES FABRIC_IFACES_BY_RANK "$TP4_DEFAULT_FABRIC_IFACES")"
  emit "hca_$i" "$(tp4_resolve_rank_value "$i" NCCL_IB_HCA NCCL_IB_HCA_BY_RANK "$TP4_DEFAULT_NCCL_IB_HCA")"
  emit "gid_index_$i" "$(tp4_resolve_rank_value "$i" NCCL_IB_GID_INDEX NCCL_IB_GID_INDEX_BY_RANK "$TP4_DEFAULT_NCCL_IB_GID_INDEX")"
done
'''

# One extra read-only probe complements verify-node.sh --quick: it checks live process,
# container and fabric state that the static verifier deliberately does not own.
REMOTE_PROBE = r'''
import base64, datetime as dt, json, os, re, subprocess, sys
from pathlib import Path

p = json.loads(base64.urlsafe_b64decode(sys.argv[1]).decode())
errors = []

def run(argv, accepted=(0,), timeout=10, include_stderr=False):
    try:
        c = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        errors.append({"check": argv[0], "error": "timeout"})
        return "", -1
    if c.returncode not in accepted:
        errors.append({"check": argv[0], "returncode": c.returncode})
    return (c.stdout + (c.stderr if include_stderr else "")).strip(), c.returncode

ids_raw, _ = run(["sudo", "-n", "docker", "ps", "-q"])
ids = ids_raw.splitlines() if ids_raw else []
objects = []
if ids:
    raw, _ = run(["sudo", "-n", "docker", "inspect", *ids])
    try: objects = json.loads(raw)
    except json.JSONDecodeError as exc:
        errors.append({"check": "docker inspect", "error": type(exc).__name__})

def name(obj): return (obj.get("Name") or "").lstrip("/")
def uses_gpu(obj):
    host = obj.get("HostConfig") or {}
    for request in host.get("DeviceRequests") or []:
        caps = request.get("Capabilities") or []
        if request.get("Driver") == "nvidia" or any("gpu" in row for row in caps): return True
    env = (obj.get("Config") or {}).get("Env") or []
    return host.get("Runtime") == "nvidia" or any(
        item.startswith("NVIDIA_VISIBLE_DEVICES=") and item.split("=", 1)[1] not in ("", "void")
        for item in env)

matches = [obj for obj in objects if name(obj) == p["container"]]
container = matches[0] if len(matches) == 1 else {}
config = container.get("Config") or {}
environment = dict(item.split("=", 1) for item in config.get("Env") or [] if "=" in item)
safe_env_names = (
    "VLLM_ADAPTIVE_K_ENABLE", "VLLM_ADAPTIVE_K_LO", "VLLM_ADAPTIVE_K_HI",
    "VLLM_ADAPTIVE_K_MODE", "VLLM_ADAPTIVE_K_SEED", "VLLM_ADAPTIVE_K_DOWN",
    "VLLM_ADAPTIVE_K_UP", "VLLM_ADAPTIVE_K_ALPHA", "VLLM_ADAPTIVE_K_SIGNAL",
    "VLLM_ADAPTIVE_K_LOG_EVERY", "NCCL_ALGO", "NCCL_IB_HCA", "NCCL_IB_GID_INDEX",
    "NCCL_IB_ROCE_VERSION_NUM", "NCCL_IB_ADDR_FAMILY", "NCCL_IB_QPS_PER_CONNECTION",
    "VLLM_E27B_SHORT_PREFILL_TOKENS", "VLLM_E27C_CADENCE_WHEN_QUEUED",
)
identity = p.get("runtime_identity") or {}
safe_env_names = set(safe_env_names) | set(identity.get("environment", {}))
safe_environment = {key: environment[key] for key in safe_env_names if key in environment}
foreign_gpu_container_count = sum(
    name(obj) != p["container"] and uses_gpu(obj) for obj in objects)
mounts = {item.get("Destination"): item.get("Source") for item in container.get("Mounts") or []}
model_marker = ""
if mounts.get("/model"):
    try: model_marker = (Path(mounts["/model"]) / ".glm53-fp8-synced").read_text().strip()
    except OSError as exc: errors.append({"check": "model marker", "error": type(exc).__name__})
draft_commit = ""
draft_config_sha = ""
draft_metadata_present = False
if mounts.get("/draft"):
    import hashlib
    draft = Path(mounts["/draft"])
    try: draft_config_sha = hashlib.sha256((draft / "config.json").read_bytes()).hexdigest()
    except OSError as exc: errors.append({"check": "drafter config", "error": type(exc).__name__})
    metadata = draft / ".cache/huggingface/download/config.json.metadata"
    draft_metadata_present = metadata.exists()
    if draft_metadata_present:
        try: draft_commit = metadata.read_text().splitlines()[0]
        except (OSError, IndexError) as exc:
            errors.append({"check": "drafter revision metadata", "error": type(exc).__name__})
digests = []
if container.get("Image"):
    raw, _ = run(["sudo", "-n", "docker", "image", "inspect", container["Image"],
                  "--format", "{{json .RepoDigests}}"])
    try: digests = json.loads(raw) or []
    except json.JSONDecodeError as exc:
        errors.append({"check": "docker image inspect", "error": type(exc).__name__})

command = config.get("Cmd") or []
safe_options = (
    "--served-model-name", "--tensor-parallel-size", "--nnodes", "--node-rank",
    "--master-addr", "--master-port",
    "--max-model-len", "--max-num-seqs", "--max-num-batched-tokens", "--kv-cache-dtype",
    "--kv-cache-memory-bytes", "--kv-cache-memory",
    "--scheduler-cls", "--moe-backend", "--speculative-config",
    "--prefill-schedule-interval", "--compilation-config",
)
option_values = {key: [] for key in safe_options}
for index, item in enumerate(command):
    for key in safe_options:
        if item == key:
            option_values[key].append(
                command[index + 1] if index + 1 < len(command) and
                not command[index + 1].startswith("--") else None)
        elif item.startswith(key + "="): option_values[key].append(item.split("=", 1)[1])
speculative = {}
if len(option_values["--speculative-config"]) == 1:
    try:
        raw_speculative = json.loads(option_values["--speculative-config"][0])
        for key in ("method", "model", "num_speculative_tokens", "num_speculative_tokens_per_batch_size"):
            if key in raw_speculative: speculative[key] = raw_speculative[key]
    except (json.JSONDecodeError, TypeError):
        errors.append({"check": "speculative configuration", "error": "invalid JSON"})
option_values.pop("--speculative-config")

configured_pids = set()
if container:
    top, _ = run(["sudo", "-n", "docker", "top", p["container"], "-eo", "pid"])
    configured_pids = {int(line) for line in top.splitlines()[1:] if line.strip().isdigit()}

# Hash the files the running container actually sees, including bind mounts. This does
# not import Python modules, initialize CUDA, or alter the serving process.
runtime_files = {}
paths = list(identity.get("container_file_sha256", {}))
if container and paths:
    hashes, _ = run(["sudo", "-n", "docker", "exec", p["container"],
                     "sha256sum", "--", *paths], timeout=15)
    for line in hashes.splitlines():
        fields = line.split(None, 1)
        if len(fields) == 2 and re.fullmatch(r"[0-9a-f]{64}", fields[0]):
            runtime_files[fields[1].lstrip(" *")] = fields[0]

runtime_workers = []
if identity:
    nccl_path = identity.get("environment", {}).get("VLLM_NCCL_SO_PATH")
    for pid in configured_pids:
        try: comm = Path(f"/proc/{pid}/comm").read_text().strip()
        except OSError: continue
        if not comm.startswith("VLLM::Worker"): continue
        maps, _ = run(["sudo", "-n", "cat", f"/proc/{pid}/maps"])
        runtime_workers.append({"pid": pid, "patched_nccl_loaded":
                                bool(nccl_path and any(line.endswith(nccl_path)
                                                       for line in maps.splitlines()))})

runtime_receipts = {}
if container and identity.get("kda_boot_receipt"):
    # Bound the history to startup so a long-running memory probe cannot make this
    # identity check grow indefinitely with the container's age.
    started = (container.get("State") or {}).get("StartedAt", "")
    try:
        end = dt.datetime.fromisoformat(started.replace("Z", "+00:00")) + dt.timedelta(minutes=40)
        logs, _ = run(["sudo", "-n", "docker", "logs", "--since", started,
                       "--until", end.isoformat(), p["container"]], timeout=20, include_stderr=True)
        for line in logs.splitlines():
            if (p.get("rank") == 0 and identity.get("scheduler_boot_signature")
                    and identity["scheduler_boot_signature"] in line):
                runtime_receipts["scheduler_boot_signature"] = True
            if "E20_KDA_INPUT_W8A16_READY " in line:
                try: runtime_receipts["kda"] = json.loads(line.split("E20_KDA_INPUT_W8A16_READY ", 1)[1])
                except json.JSONDecodeError: pass
            if "E21_BF16_RESIDUE_W8A16_READY " in line:
                try: runtime_receipts["e21"] = json.loads(line.split("E21_BF16_RESIDUE_W8A16_READY ", 1)[1])
                except json.JSONDecodeError: pass
            for signature in identity.get("boot_lines", []):
                if p.get("rank") == 0 and signature in line:
                    runtime_receipts.setdefault("boot_lines", []).append(signature)
            if "E22_DRAFTER_W8A16_READY " in line:
                try: runtime_receipts["e22"] = json.loads(line.split("E22_DRAFTER_W8A16_READY ", 1)[1])
                except json.JSONDecodeError: pass
    except ValueError:
        errors.append({"check": "container start time", "error": "invalid timestamp"})
if container and identity.get("memory_probe"):
    logs, _ = run(["sudo", "-n", "docker", "logs", "--tail", "80", p["container"]],
                  timeout=10, include_stderr=True)
    for line in logs.splitlines():
        if "E20_MEMORY_PROBE " in line:
            try: sample = json.loads(line.split("E20_MEMORY_PROBE ", 1)[1])
            except json.JSONDecodeError: continue
            runtime_receipts["memory_probe"] = {
                key: sample.get(key) for key in ("schema", "phase", "rank", "pid", "wall_time_ns")}
gpu_raw, gpu_rc = run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"])
gpu_pids = {int(line) for line in gpu_raw.splitlines() if line.strip().isdigit()}

unit, unit_rc = run(["systemctl", "is-active", "tp4-flusher"], accepted=(0, 3, 4))
_, legacy_rc = run(["pgrep", "-f", "[f]lusher-unconditional"], accepted=(0, 1))

ip_raw, _ = run(["ip", "-o", "-4", "addr", "show"])
interfaces = {}
try: prefix = re.compile(p["fabric_prefix_re"])
except re.error as exc:
    errors.append({"check": "fabric prefix", "error": type(exc).__name__})
    prefix = re.compile(r"a^")
for line in ip_raw.splitlines():
    fields = line.split()
    if len(fields) >= 4 and fields[2] == "inet" and prefix.search(fields[3].split("/", 1)[0]):
        try: interfaces[fields[1]] = Path(f"/sys/class/net/{fields[1]}/mtu").read_text().strip()
        except OSError as exc:
            errors.append({"check": "fabric interface MTU", "error": type(exc).__name__})
jumbo = []
for target in p["fabric_targets"]:
    _, rc = run(["ping", "-M", "do", "-s", "8972", "-c", "2", "-W", "3", target], timeout=8)
    jumbo.append(rc == 0)

print(json.dumps({
    "errors": errors, "running_container_count": len(matches),
    "all_running_container_count": len(objects),
    "foreign_gpu_container_count": foreign_gpu_container_count,
    "foreign_gpu_pid_count": len(gpu_pids - configured_pids) if gpu_rc == 0 else None,
    "container": {"image_reference": config.get("Image", ""), "image_digests": digests,
                  "image_id": container.get("Image", ""),
                  "model_path": command[0] if command else "", "options": option_values,
                  "async_flag_count": sum(item == "--async-scheduling" or
                                          item.startswith("--async-scheduling=") for item in command),
                  "speculative": speculative, "environment": safe_environment,
                  "model_marker": model_marker,
                  "draft_commit": draft_commit,
                  "draft_metadata_present": draft_metadata_present,
                  "draft_config_sha": draft_config_sha,
                  "model_mount": mounts.get("/model") == os.path.expandvars(p["model_dir"]),
                  "draft_mount": mounts.get("/draft") == os.path.expandvars(p["draft_dir"]),
                  "runtime_files": runtime_files, "runtime_receipts": runtime_receipts,
                  "runtime_workers": runtime_workers},
    "flusher": {"unit_state": unit, "unit_rc": unit_rc, "legacy_process": legacy_rc == 0},
    "fabric_interfaces": interfaces, "jumbo_pings": jumbo,
}, sort_keys=True))
'''


class CheckFailure(RuntimeError):
    pass


def run_command(argv: list[str], *, input_text: str | None = None, timeout: float = 90) -> dict[str, Any]:
    try:
        result = subprocess.run(
            argv, input=input_text, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=timeout, check=False,
        )
        return {"returncode": result.returncode, "stdout": result.stdout,
                "stderr": result.stderr, "timed_out": False}
    except subprocess.TimeoutExpired as exc:
        return {"returncode": None, "stdout": str(exc.stdout or ""),
                "stderr": str(exc.stderr or ""), "timed_out": True}


def load_recipe(timeout: float) -> tuple[dict[str, str], dict[str, Any]]:
    result = subprocess.run(
        ["bash", "-c", LOAD_RECIPE, "check-f0", str(REPO)],
        capture_output=True, timeout=timeout, check=False,
    )
    diagnostic = {"returncode": result.returncode}
    if result.returncode:
        raise CheckFailure("effective configuration is invalid")
    fields = result.stdout.split(b"\0")
    if fields and not fields[-1]: fields.pop()
    if len(fields) % 2: raise CheckFailure("effective configuration returned malformed data")
    return ({fields[i].decode(): fields[i + 1].decode() for i in range(0, len(fields), 2)},
            diagnostic)


def expected_f0(baseline: Path | None = None) -> dict[str, Any]:
    record = json.loads((baseline or BASELINE).read_text(encoding="utf-8"))
    system = record["system"]
    seqs = re.search(r"max sequences (\d+)", system["scheduler"])
    batched = re.search(r"max batched tokens (\d+)", system["scheduler"])
    if not seqs or not batched: raise CheckFailure("frozen baseline scheduler identity is not parseable")
    # F1 records the pullable digest reference separately from the descriptive image field.
    image_digest = system.get("serving_image_digest") or system["serving_image"]
    adaptive = dict(ADAPTIVE_DEFAULTS)
    policy = system.get("adaptive_k") or {}
    adaptive["VLLM_ADAPTIVE_K_RESPECT_DRAFT_BUDGET"] = str(int(policy.get("respect_draft_budget", False)))
    for key, name in (("VLLM_ADAPTIVE_K_MODE", "mode"), ("VLLM_ADAPTIVE_K_LO", "k_lo"),
                      ("VLLM_ADAPTIVE_K_HI", "k_hi"), ("VLLM_ADAPTIVE_K_UP", "up"),
                      ("VLLM_ADAPTIVE_K_DOWN", "down"), ("VLLM_ADAPTIVE_K_ALPHA", "alpha"),
                      ("VLLM_ADAPTIVE_K_SEED", "seed"), ("VLLM_ADAPTIVE_K_SIGNAL", "signal")):
        if name in policy:
            value = policy[name]
            adaptive[key] = str(float(value)) if name in ("up", "down", "alpha", "seed") else str(value)
    kv_bytes = system.get("kv_cache_memory_bytes_per_rank")
    if kv_bytes is None:
        match = re.search(r"\b(\d+)-byte pool per rank\b", system["kv_cache"])
        if not match: raise CheckFailure("frozen baseline KV memory budget is not parseable")
        kv_bytes = int(match.group(1))
    return {
        "baseline_id": record["baseline_id"],
        "model_repo": system["model"], "model_rev": system["model_revision"],
        "draft_rev": system["drafter"]["revision"], "image_digest": image_digest,
        "max_model_len": str(system["context_limit_tokens"]), "max_num_seqs": seqs.group(1),
        "batched_tokens": batched.group(1), "kv_cache_dtype": system["kv_cache"].split()[0],
        "spec_tokens": str(system["drafter"]["draft_tokens"]),
        "adaptive_tokens": system["drafter"]["adaptive_verification_tokens"],
        "adaptive_env": adaptive,
        "kv_cache_memory_bytes": str(kv_bytes),
        # Absent in records before E27, which ran with the engine default of 1 (no cadence).
        "prefill_schedule_interval": str(
            (system.get("engine_arguments") or {}).get("prefill_schedule_interval", 1)),
        # Absent in records before E28, which let vLLM choose the CUDA graph capture limit.
        "max_cudagraph_capture_size": str(
            (system.get("engine_arguments") or {}).get("max_cudagraph_capture_size", "")),
        "image_id": system.get("serving_image_id"),
        "runtime_identity": system.get("operational_identity") or {},
        "sparkcache_mode": "on" if system.get("sparkcache") else "off",
        "spark_mhc_prefill_shard": str(int(system.get("mhc_prefill", {}).get("enabled", False))),
        "payload_pins": {key: value for key, value in {
            "sparkcache_config_sha256": system.get("sparkcache", {}).get("kv_transfer_config_sha256"),
            "sparkcache_connector_sha256": system.get("sparkcache", {}).get("connector_sha256"),
            "sparkcache_encoder_sha256": system.get("sparkcache", {}).get("hybrid_encoder_sha256"),
        }.items() if value},
    }


def flag_values(command: list[str], name: str) -> list[str | None]:
    values: list[str | None] = []
    for index, item in enumerate(command):
        if item == name:
            values.append(command[index + 1] if index + 1 < len(command)
                          and not command[index + 1].startswith("--") else None)
        elif item.startswith(name + "="): values.append(item.split("=", 1)[1])
    return values


def docker_env(value: str) -> dict[str, str]:
    tokens, result, index = shlex.split(value), {}, 0
    while index < len(tokens):
        item, assignment = tokens[index], ""
        if item in ("-e", "--env") and index + 1 < len(tokens):
            index += 1; assignment = tokens[index]
        elif item.startswith("-e") and item != "-e": assignment = item[2:]
        elif item.startswith("--env="): assignment = item.split("=", 1)[1]
        if "=" in assignment:
            key, value = assignment.split("=", 1); result[key] = value
        index += 1
    return result


def adaptive_env(value: str) -> dict[str, str]:
    parsed = docker_env(value)
    return {key: parsed.get(key, default) for key, default in ADAPTIVE_DEFAULTS.items()}


# Scheduler flags that only records from E27c on may carry; older records require absence.
SCHEDULER_FLAGS = ("VLLM_E27B_SHORT_PREFILL_TOKENS", "VLLM_E27C_CADENCE_WHEN_QUEUED")


def compilation_flag(expected: dict[str, Any]) -> list[str]:
    """Expected --compilation-config values, compared without JSON quotes."""
    size = expected.get("max_cudagraph_capture_size", "")
    return [] if not size else ["{max_cudagraph_capture_size:" + size + "}"]


def unquoted(values: list[str | None]) -> list[str]:
    return [(value or "").replace('"', "") for value in values]


def interval_flag(expected: dict[str, Any]) -> list[str]:
    """Expected --prefill-schedule-interval values: none for the engine default of 1."""
    value = expected.get("prefill_schedule_interval", "1")
    return [] if value == "1" else [value]


def recipe_problems(recipe: dict[str, str], expected: dict[str, Any]) -> list[str]:
    problems = []
    protected = (
        "nodes", "tp4_hosts", "mgmt_ips", "hosts", "master_ip", "master_port",
        "api_port", "fabric_prefix_re",
        "container", "model_dir", "draft_dir", "served_name",
    )
    for key in protected:
        if recipe.get(key) != recipe.get("base_" + key):
            problems.append(f"TP4_ENV changed protected field: {key}")
    for rank in range(4):
        if recipe.get(f"fabric_target_{rank}") != recipe.get(f"base_fabric_target_{rank}"):
            problems.append(f"TP4_ENV changed protected fabric topology: rank {rank}")
        for key in ("mgmt_if", "fabric_ifaces", "hca"):
            if recipe.get(f"{key}_{rank}") != recipe.get(f"base_{key}_{rank}"):
                problems.append(f"TP4_ENV changed protected {key}: rank {rank}")
    for key in ("model_repo", "model_rev", "draft_rev", "image_digest", "max_model_len",
                "max_num_seqs", "batched_tokens", "kv_cache_dtype", "spec_tokens"):
        if recipe.get(key) != str(expected[key]): problems.append(f"effective recipe mismatch: {key}")
    # A historical baseline may be selected by an overlay over the current base.
    # Compare runtime policy with that selected baseline, not the unselected base;
    # the site/container/topology protections above still apply to every overlay.
    env = adaptive_env(recipe.get("extra_docker_env", ""))
    for key, value in expected.get("adaptive_env", ADAPTIVE_DEFAULTS).items():
        if env[key] != value: problems.append("effective baseline adaptive policy: " + key)
    raw_env = docker_env(recipe.get("extra_docker_env", ""))
    if "NCCL_IB_QPS_PER_CONNECTION" in raw_env: problems.append("NQ2 QPS delta still present")
    args = shlex.split(recipe.get("extra_vllm_args", ""))
    if flag_values(args, "--scheduler-cls") != ["adaptive_k_scheduler.AdaptiveKScheduler"]:
        problems.append("baseline scheduler class")
    if flag_values(args, "--moe-backend") != ["triton"]: problems.append("baseline MoE backend")
    if (flag_values(args, "--kv-cache-memory-bytes") + flag_values(args, "--kv-cache-memory")
            != [expected["kv_cache_memory_bytes"]]):
        problems.append("baseline KV memory budget")
    if flag_values(args, "--prefill-schedule-interval") != interval_flag(expected):
        problems.append("baseline prefill schedule interval")
    if unquoted(flag_values(args, "--compilation-config")) != compilation_flag(expected):
        problems.append("baseline compilation config")
    if recipe.get("sparkcache_mode", "off") != expected["sparkcache_mode"]:
        problems.append("baseline SparkCache mode")
    if recipe.get("spark_mhc_prefill_shard", "0") != expected["spark_mhc_prefill_shard"]:
        problems.append("baseline mHC prefill flag")
    for key, value in expected.get("payload_pins", {}).items():
        if recipe.get(key) != value:
            problems.append("baseline payload pin: " + key)
    identity = expected.get("runtime_identity") or {}
    for key, value in identity.get("environment", {}).items():
        # Launcher-owned values, such as NCCL GID and LD_PRELOAD, are checked on
        # the actual container; only explicit EXTRA_DOCKER_ENV entries live here.
        if key in raw_env and raw_env[key] != str(value):
            problems.append("baseline runtime environment: " + key)
    for key in SCHEDULER_FLAGS:
        if key in raw_env and key not in identity.get("environment", {}):
            problems.append("baseline runtime environment: unexpected " + key)
        elif key in identity.get("environment", {}) and key not in raw_env:
            problems.append("baseline runtime environment: " + key)
    try: table = json.loads("{" + recipe.get("spec_extra_json", "") + "}").get(
        "num_speculative_tokens_per_batch_size")
    except json.JSONDecodeError: table = None
    if table != [[1, 1, expected["adaptive_tokens"][1]], [2, 6, expected["adaptive_tokens"][0]]]:
        problems.append("baseline adaptive graph table")
    if recipe.get("async_scheduling") != "0": problems.append("optional async CLI flag present")
    if len(recipe.get("hosts", "").split()) != 4: problems.append("expected four SSH ranks")
    for rank in range(4):
        if not recipe.get(f"hca_{rank}"): problems.append(f"rank {rank}: missing baseline HCA")
        if recipe.get(f"gid_index_{rank}") != "-1": problems.append(f"rank {rank}: baseline GID mode")
        targets = recipe.get(f"fabric_target_{rank}", "").split()
        if len(targets) != 2: problems.append(f"rank {rank}: fabric target count")
        for target in targets:
            try: ipaddress.IPv4Address(target)
            except ValueError: problems.append(f"rank {rank}: invalid fabric target")
    return problems


def fabric_prefix(recipe: dict[str, str]) -> str:
    if recipe.get("fabric_prefix_re"): return recipe["fabric_prefix_re"]
    octets = recipe["fabric_target_0"].split()[0].split(".")
    if len(octets) != 4: raise CheckFailure("cannot derive fabric prefix")
    return "^" + re.escape(".".join(octets[:2]) + ".")


def probe_rank(rank: int, host: str, recipe: dict[str, str], timeout: float) -> dict[str, Any]:
    verify = run_command([str(REPO / "scripts/verify-node.sh"), "--quick", "--host", host], timeout=timeout)
    payload = base64.urlsafe_b64encode(json.dumps({
        "rank": rank, "container": recipe["container"], "fabric_prefix_re": fabric_prefix(recipe),
        "fabric_targets": recipe[f"fabric_target_{rank}"].split(),
        "model_dir": recipe["model_dir"], "draft_dir": recipe["draft_dir"],
        "runtime_identity": recipe.get("runtime_identity", {}),
    }).encode()).decode()
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
           "-o", f"ConnectTimeout={max(1, min(10, int(timeout)))}",
           "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2",
           host, "python3", "-", payload]
    remote_result = run_command(ssh, input_text=REMOTE_PROBE, timeout=timeout)
    remote: dict[str, Any] = {"probe_failed": "timeout" if remote_result["timed_out"] else "ssh"}
    if remote_result["returncode"] == 0:
        try: remote = json.loads(remote_result["stdout"])
        except json.JSONDecodeError: remote = {"probe_failed": "invalid response"}
    return {
        "rank": rank,
        "verify_node": {key: verify[key] for key in ("returncode", "timed_out")},
        "remote": remote,
        "remote_command": {
            key: remote_result[key] for key in ("returncode", "timed_out")
        },
    }


def reportable_rank_probe(item: dict[str, Any]) -> dict[str, Any]:
    """Drop captured helper output while retaining bounded structured status."""
    return {
        "rank": item.get("rank"),
        "verify_node": {
            key: item.get("verify_node", {}).get(key)
            for key in ("returncode", "timed_out")
        },
        "remote_command": {
            key: item.get("remote_command", {}).get(key)
            for key in ("returncode", "timed_out")
        },
        "remote": item.get("remote", {}),
    }


def endpoint_probe(base_url: str, timeout: float) -> dict[str, Any]:
    result: dict[str, Any] = {"errors": []}; bodies = {}
    for path in ("/health", "/metrics"):
        try:
            with urllib.request.urlopen(base_url.rstrip("/") + path, timeout=min(timeout, 10)) as response:
                result[path + "_status"] = response.status
                bodies[path] = response.read(2 * 1024 * 1024).decode("utf-8", "replace")
        except (OSError, urllib.error.URLError) as exc:
            result[path + "_status"] = None
            result["errors"].append({"path": path, "error": type(exc).__name__})
    for metric in ("vllm:num_requests_running", "vllm:num_requests_waiting"):
        values = []
        pattern = re.compile(rf"^{re.escape(metric)}(?:\{{[^\n]*\}})?\s+(\S+)$")
        for line in bodies.get("/metrics", "").splitlines():
            suffix = line[len(metric):] if line.startswith(metric) else ""
            if not line.startswith(metric) or (suffix and suffix[0] != "{" and
                                                not suffix[0].isspace()):
                continue
            match = pattern.fullmatch(line)
            if not match:
                result["errors"].append({"path": "/metrics", "error": "invalid metric"})
                continue
            try:
                values.append(float(match.group(1)))
            except ValueError:
                result["errors"].append({"path": "/metrics", "error": "invalid metric"})
        result[metric] = values
    return result


def evaluate(recipe: dict[str, str], expected: dict[str, Any], ranks: list[dict[str, Any]], endpoint: dict[str, Any]) -> list[str]:
    problems = recipe_problems(recipe, expected); by_rank = {item["rank"]: item for item in ranks}
    for rank in range(4):
        if rank not in by_rank: problems.append(f"rank {rank}: missing probe"); continue
        item = by_rank[rank]; verify = item["verify_node"]; remote = item["remote"]
        if verify["timed_out"]: problems.append(f"rank {rank}: verify-node timeout")
        elif verify["returncode"] != 0: problems.append(f"rank {rank}: verify-node failed")
        if remote.get("probe_failed"): problems.append(f"rank {rank}: probe {remote['probe_failed']}"); continue
        if remote.get("errors"): problems.append(f"rank {rank}: remote command failed")
        if remote.get("running_container_count") != 1: problems.append(f"rank {rank}: running container count")
        container = remote.get("container") or {}; options = container.get("options") or {}
        if container.get("image_reference") != recipe.get("image"): problems.append(f"rank {rank}: image reference")
        image_id_matches = (expected.get("image_id") is not None and
                            container.get("image_id") == expected["image_id"])
        if expected["image_digest"] not in (container.get("image_digests") or []) and not image_id_matches:
            problems.append(f"rank {rank}: running image digest")
        if expected.get("image_id") and not image_id_matches:
            problems.append(f"rank {rank}: running image content ID")
        if container.get("model_marker") != expected["model_rev"]:
            problems.append(f"rank {rank}: model revision marker")
        if not container.get("model_mount") or not container.get("draft_mount"):
            problems.append(f"rank {rank}: model or drafter mount")
        if container.get("model_path") != "/model": problems.append(f"rank {rank}: model path")
        expected_flags = {"--served-model-name": recipe.get("served_name"),
                          "--tensor-parallel-size": "4", "--nnodes": "4",
                          "--node-rank": str(rank), "--master-addr": recipe.get("master_ip"),
                          "--master-port": recipe.get("master_port"),
                          "--max-model-len": expected["max_model_len"],
                          "--max-num-seqs": expected["max_num_seqs"],
                          "--max-num-batched-tokens": expected["batched_tokens"],
                          "--kv-cache-dtype": expected["kv_cache_dtype"],
                          "--scheduler-cls": "adaptive_k_scheduler.AdaptiveKScheduler",
                          "--moe-backend": "triton"}
        for option, value in expected_flags.items():
            if options.get(option) != [value]: problems.append(f"rank {rank}: command {option}")
        if (options.get("--kv-cache-memory-bytes", []) + options.get("--kv-cache-memory", [])
                != [expected["kv_cache_memory_bytes"]]):
            problems.append(f"rank {rank}: command --kv-cache-memory-bytes")
        if options.get("--prefill-schedule-interval", []) != interval_flag(expected):
            problems.append(f"rank {rank}: command --prefill-schedule-interval")
        if unquoted(options.get("--compilation-config", [])) != compilation_flag(expected):
            problems.append(f"rank {rank}: command --compilation-config")
        speculative = container.get("speculative") or {}
        if (speculative.get("method"), speculative.get("model"),
                speculative.get("num_speculative_tokens")) != ("dflash", "/draft", int(expected["spec_tokens"])):
            problems.append(f"rank {rank}: speculative configuration")
        if speculative.get("num_speculative_tokens_per_batch_size") != [
                [1, 1, expected["adaptive_tokens"][1]], [2, 6, expected["adaptive_tokens"][0]]]:
            problems.append(f"rank {rank}: adaptive graph table")
        if container.get("async_flag_count") != 0: problems.append(f"rank {rank}: optional async CLI flag")
        env = container.get("environment") or {}
        for key, value in expected.get("adaptive_env", ADAPTIVE_DEFAULTS).items():
            if env.get(key, value) != value: problems.append(f"rank {rank}: adaptive policy {key}")
        if "NCCL_IB_QPS_PER_CONNECTION" in env: problems.append(f"rank {rank}: NQ2 QPS delta")
        selectors = {"NCCL_ALGO": "Ring", "NCCL_IB_HCA": recipe.get(f"hca_{rank}"),
                     "NCCL_IB_GID_INDEX": "-1", "NCCL_IB_ROCE_VERSION_NUM": "2",
                     "NCCL_IB_ADDR_FAMILY": "AF_INET"}
        if any(env.get(key) != value for key, value in selectors.items()):
            problems.append(f"rank {rank}: baseline collective identity")
        identity = expected.get("runtime_identity") or {}
        if identity:
            workers = container.get("runtime_workers") or []
            if len(workers) != 1:
                problems.append(f"rank {rank}: runtime GPU worker count")
            elif (identity.get("environment", {}).get("VLLM_NCCL_SO_PATH") and
                  not workers[0].get("patched_nccl_loaded")):
                problems.append(f"rank {rank}: patched NCCL not loaded by GPU worker")
        for key, value in identity.get("environment", {}).items():
            if env.get(key) != str(value):
                problems.append(f"rank {rank}: runtime environment {key}")
        for key in SCHEDULER_FLAGS:
            if key in env and key not in identity.get("environment", {}):
                problems.append(f"rank {rank}: unexpected runtime environment {key}")
        for path, sha in identity.get("container_file_sha256", {}).items():
            if container.get("runtime_files", {}).get(path) != sha:
                problems.append(f"rank {rank}: runtime file {path}")
        receipts = container.get("runtime_receipts") or {}
        if (rank == 0 and identity.get("scheduler_boot_signature")
                and not receipts.get("scheduler_boot_signature")):
            problems.append("rank 0: draft-budget scheduler boot signature")
        if rank == 0:
            for signature in identity.get("boot_lines", []):
                if signature not in (receipts.get("boot_lines") or []):
                    problems.append("rank 0: boot signature " + signature)
        kda = receipts.get("kda") or {}
        for key, value in identity.get("kda_boot_receipt", {}).items():
            if key == "padded_n":
                modules = kda.get("receipts") or []
                if (len(modules) != identity["kda_boot_receipt"].get("modules") or
                        any(module.get("padded_n") != value for module in modules)):
                    problems.append(f"rank {rank}: KDA padding receipt")
            elif kda.get(key) != value:
                problems.append(f"rank {rank}: KDA boot receipt {key}")
        e21 = receipts.get("e21") or {}
        for key, value in identity.get("e21_boot_receipt", {}).items():
            if e21.get(key) != value:
                problems.append(f"rank {rank}: E21 boot receipt {key}")
        e22 = receipts.get("e22") or {}
        for key, value in identity.get("e22_boot_receipt", {}).items():
            if e22.get(key) != value:
                problems.append(f"rank {rank}: E22 boot receipt {key}")
        memory_probe = receipts.get("memory_probe") or {}
        if identity.get("memory_probe"):
            if (any(memory_probe.get(key) != value for key, value in identity["memory_probe"].items())
                    or memory_probe.get("rank") != rank):
                problems.append(f"rank {rank}: memory probe identity")
        if (remote.get("foreign_gpu_container_count") != 0
                or remote.get("foreign_gpu_pid_count") != 0):
            problems.append(f"rank {rank}: foreign GPU workload")
        flusher = remote.get("flusher") or {}
        if flusher.get("unit_state") != "inactive" or flusher.get("legacy_process"):
            problems.append(f"rank {rank}: flusher not proven inactive")
        interfaces = remote.get("fabric_interfaces") or {}
        if len(interfaces) < 2 or any(str(value) != "9000" for value in interfaces.values()):
            problems.append(f"rank {rank}: addressed MTU-9000 fabric")
        if remote.get("jumbo_pings") != [True, True]: problems.append(f"rank {rank}: jumbo directions")
    anchored_draft_hashes = set()
    for rank, item in by_rank.items():
        container = item.get("remote", {}).get("container", {})
        commit = container.get("draft_commit")
        if container.get("draft_metadata_present") or commit:
            if commit != expected["draft_rev"]:
                problems.append(f"rank {rank}: drafter revision marker")
            elif container.get("draft_config_sha"):
                anchored_draft_hashes.add(container["draft_config_sha"])
    if not anchored_draft_hashes:
        problems.append("drafter revision is not proven")
    elif len(anchored_draft_hashes) != 1:
        problems.append("drafter config revision anchors disagree")
    else:
        anchored_draft_hash = next(iter(anchored_draft_hashes))
        for rank, item in by_rank.items():
            sha = item.get("remote", {}).get("container", {}).get("draft_config_sha")
            if sha != anchored_draft_hash:
                problems.append(f"rank {rank}: drafter config identity")
    if endpoint.get("/health_status") != 200: problems.append("endpoint health")
    for metric in ("vllm:num_requests_running", "vllm:num_requests_waiting"):
        values = endpoint.get(metric)
        if (not isinstance(values, list) or not values
                or any(not math.isfinite(value) or value < 0 or value != 0 for value in values)):
            problems.append("endpoint idle: " + metric)
    if endpoint.get("errors"): problems.append("endpoint probe failed")
    return problems


def make_report_dir(repo: Path, root: Path | None = None) -> Path:
    target = (root or Path(tempfile.gettempdir())).resolve()
    if target == repo.resolve() or repo.resolve() in target.parents:
        raise CheckFailure("private report root must be outside the checkout")
    target.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix="tp4-f0-check-", dir=target)); path.chmod(0o700)
    return path


def write_report(directory: Path, report: dict[str, Any]) -> Path:
    path = directory / "report.json"; fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream: json.dump(report, stream, indent=2, sort_keys=True)
    return path


def print_summary(passed: bool, baseline_id: str = "BASELINE") -> None:
    print(f"{baseline_id} CHECK {'PASS' if passed else 'FAIL'}")


def positive(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0 or parsed > 300:
        raise argparse.ArgumentTypeError("timeout must be finite and between 0 and 300 seconds")
    return parsed


def checked_base_url(value: str) -> str:
    try: parsed = urllib.parse.urlsplit(value)
    except ValueError as exc: raise CheckFailure("base URL is invalid") from exc
    if (parsed.scheme != "http" or not parsed.hostname or parsed.username or parsed.password
            or parsed.query or parsed.fragment or parsed.path not in ("", "/")):
        raise CheckFailure("base URL must be an uncredentialed HTTP origin")
    try: parsed.port
    except ValueError as exc: raise CheckFailure("base URL port is invalid") from exc
    return value.rstrip("/")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", help="endpoint base URL, including a localhost SSH tunnel")
    parser.add_argument("--timeout", type=positive, default=90.0)
    parser.add_argument("--baseline", type=Path, default=BASELINE,
                        help="frozen baseline JSON to check against (default: %(default)s)")
    parser.add_argument("--report-root", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, *, rank_probe: Callable = probe_rank,
         http_probe: Callable = endpoint_probe) -> int:
    args = parse_args(argv)
    try: directory = make_report_dir(REPO, args.report_root)
    except Exception:
        print_summary(False)
        return 1
    report: dict[str, Any] = {
        "schema": 1, "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "baseline_file": str(args.baseline),
        "verified_scope": ["selected baseline recipe and verify-node static checks",
                           "running containers, GPU work, flusher and key command identity",
                           "addressed MTU-9000 fabric and eight jumbo directions",
                           "GET /health 200 and endpoint idle metrics"],
        "unverified_scope": ["full model/drafter hashes (use verify-node.sh --full-model)",
                             "functional inference gates (no inference request is sent)",
                             "engine async metadata (optional CLI absence and scheduler wiring are checked)",
                             "drafter content files are not hashed; ranks without revision metadata use a shared config.json hash anchored to a rank with the expected revision"],
    }
    passed = False
    try:
        recipe, diagnostic = load_recipe(args.timeout); expected = expected_f0(args.baseline)
        report["baseline_id"] = expected["baseline_id"]
        configuration_problems = recipe_problems(recipe, expected)
        if configuration_problems:
            report.update({"configuration_diagnostic": diagnostic,
                           "problems": configuration_problems})
            raise CheckFailure("configuration preflight failed")
        hosts = recipe["hosts"].split()
        if len(hosts) != 4: raise CheckFailure("effective recipe does not resolve four SSH ranks")
        base_url = checked_base_url(
            args.base_url or f"http://{recipe['master_ip']}:{recipe['api_port']}")
        # Pass the selected baseline's runtime pins to the remote read-only probe.
        recipe["runtime_identity"] = expected.get("runtime_identity", {})
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(rank_probe, rank, host, recipe, args.timeout)
                       for rank, host in enumerate(hosts)]
            endpoint = http_probe(base_url, args.timeout)
            ranks = [future.result() for future in futures]
        problems = evaluate(recipe, expected, ranks, endpoint); passed = not problems
        report.update({"configuration_diagnostic": diagnostic, "expected_baseline": expected,
                       "rank_probes": [reportable_rank_probe(item) for item in ranks],
                       "endpoint": endpoint, "problems": problems})
    except Exception as exc:
        if "problems" not in report:
            detail = str(exc)[:160] if isinstance(exc, CheckFailure) else "unexpected internal error"
            report["problems"] = [f"{type(exc).__name__}: {detail}"]
    report["status"] = "PASS" if passed else "FAIL"
    report["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_report(directory, report); print_summary(passed, report.get("baseline_id", "BASELINE"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
