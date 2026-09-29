#!/usr/bin/env python3
"""Build and optionally stage a private, test-only resilience overlay."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys

from common import (
    ContractError,
    MAX_NAMESPACE_BYTES,
    atomic_json,
    namespace_name,
    sha256_file,
    validate_campaign_id,
)


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUNTIME_FILES = ("runtime.py", "connector_wrapper.py", "faultctl.py", "probe.py", "common.py")
CONNECTOR_TARGET = "/usr/local/lib/python3.12/dist-packages/sparkcache/spark_context_cache_connector.py"
TEST_CACHE_MAX_BYTES = 3 << 30
TEST_CACHE_LOW_WATERMARK_BYTES = 2 << 30
DEFAULT_KV_CACHE_MEMORY_BYTES = 16 << 30
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
INTENDED_CONNECTOR_HOOKS = {
    "_capture_stream_snapshot", "_commit_store_snapshot", "_finish_store",
    "_restore_stream_snapshot", "shutdown",
}
RUNTIME_VARIANTS = {
    "default": ("delta.env", ()),
    "prefill-cache-trim": (
        "delta-prefill-trim.env",
        ("scripts/node/experiments/e03/prefill-cache-trim/delta.env",),
    ),
    "prefill-step-cap": (
        "delta-prefill-step-cap.env",
        ("scripts/node/experiments/e03/prefill-cache-trim/delta.env",
         "scripts/node/experiments/e03/prefill-step-cap/delta.env"),
    ),
}


def positive_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise argparse.ArgumentTypeError("value must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("value must be a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def kv_cache_override_text(requested_bytes: int) -> str:
    requested_bytes = positive_int(requested_bytes)
    return f"""# Test-only KV pool override; require exactly the protected 16 GiB selection.
_RES_KV_FROM=--kv-cache-memory-bytes={DEFAULT_KV_CACHE_MEMORY_BYTES}
_RES_KV_TO=--kv-cache-memory-bytes={requested_bytes}
if [[ "${{EXTRA_VLLM_ARGS-}}" == *$'\n'* ]]; then
  echo "resilience KV override refuses multiline vLLM arguments" >&2
  unset _RES_KV_FROM _RES_KV_TO
  return 1
fi
read -r -a _RES_KV_WORDS <<<"${{EXTRA_VLLM_ARGS-}}"
_RES_KV_OUT= _RES_KV_FOUND=0 _RES_KV_KEYS=0 _RES_KV_BAD=0
for _RES_KV_WORD in "${{_RES_KV_WORDS[@]}}"; do
  case "$_RES_KV_WORD" in
    --kv-cache-memory-bytes=*)
      _RES_KV_KEYS=$((_RES_KV_KEYS + 1)) ;;
    --kv-cache-memory-bytes|--kv-cache-memory|--kv-cache-memory=*)
      _RES_KV_BAD=$((_RES_KV_BAD + 1)) ;;
  esac
  if [ "$_RES_KV_WORD" = "$_RES_KV_FROM" ]; then
    _RES_KV_WORD=$_RES_KV_TO
    _RES_KV_FOUND=$((_RES_KV_FOUND + 1))
  fi
  _RES_KV_OUT+="${{_RES_KV_OUT:+ }}$_RES_KV_WORD"
done
if [ "$_RES_KV_FOUND" != 1 ] || [ "$_RES_KV_KEYS" != 1 ] || [ "$_RES_KV_BAD" != 0 ]; then
  echo "resilience KV override requires exactly one protected 16 GiB KV argument and no aliases" >&2
  unset _RES_KV_FROM _RES_KV_TO _RES_KV_WORDS _RES_KV_WORD _RES_KV_OUT \
    _RES_KV_FOUND _RES_KV_KEYS _RES_KV_BAD
  return 1
fi
EXTRA_VLLM_ARGS=$_RES_KV_OUT
unset _RES_KV_FROM _RES_KV_TO _RES_KV_WORDS _RES_KV_WORD _RES_KV_OUT \
  _RES_KV_FOUND _RES_KV_KEYS _RES_KV_BAD
"""


def admission_contract() -> dict[str, object]:
    manifest_path = REPO / ADMISSION_MANIFEST
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ContractError("bounded-admission manifest is missing, non-regular, or symlinked")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot read bounded-admission manifest: {error}") from error
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
        raise ContractError("bounded-admission manifest has an unexpected contract")
    source_path = candidate.get("path")
    source_sha = candidate.get("sha256")
    if (not isinstance(source_path, str) or source_path.startswith("/") or ".." in source_path
            or not isinstance(source_sha, str) or len(source_sha) != 64
            or any(char not in "0123456789abcdef" for char in source_sha)):
        raise ContractError("bounded-admission manifest has an invalid candidate pin")
    source = REPO / source_path
    if source.is_symlink() or not source.is_file() or sha256_file(source) != source_sha:
        raise ContractError("bounded-admission middleware differs from its manifest")
    return {
        "manifest_path": ADMISSION_MANIFEST,
        "manifest_sha256": sha256_file(manifest_path),
        "source_path": source_path,
        "source_sha256": source_sha,
        "mount_target": ADMISSION_TARGET,
        "import_string": ADMISSION_IMPORT,
        "environment": dict(ADMISSION_ENVIRONMENT),
    }


def admission_override_text(campaign_id: str) -> str:
    relative = f"scripts/resilience/.campaign/{campaign_id}"
    remote_source = f"$HOME/tp4/{relative}/{ADMISSION_STAGED_NAME}"
    environment = " ".join(
        f"-e {key}={value}" for key, value in ADMISSION_ENVIRONMENT.items())
    keys = "|".join(ADMISSION_ENVIRONMENT)
    return f"""# Test-only bounded HTTP admission middleware.
if [[ "${{EXTRA_DOCKER_ENV-}}" == *$'\n'* ]] || [[ "${{EXTRA_VLLM_ARGS-}}" == *$'\n'* ]]; then
  echo "bounded admission refuses multiline launcher arguments" >&2
  return 1
fi
read -r -a _RES_ADM_DOCKER <<<"${{EXTRA_DOCKER_ENV-}}"
_RES_ADM_PREV= _RES_ADM_ENVS=0 _RES_ADM_MOUNTS=0
for _RES_ADM_WORD in "${{_RES_ADM_DOCKER[@]}}"; do
  _RES_ADM_SPEC=
  case "$_RES_ADM_PREV" in -e|--env) _RES_ADM_SPEC=$_RES_ADM_WORD ;; esac
  case "$_RES_ADM_WORD" in -e?*) _RES_ADM_SPEC=${{_RES_ADM_WORD#-e}} ;; --env=*) _RES_ADM_SPEC=${{_RES_ADM_WORD#--env=}} ;; esac
  case "${{_RES_ADM_SPEC%%=*}}" in {keys}) _RES_ADM_ENVS=$((_RES_ADM_ENVS + 1)) ;; esac
  _RES_ADM_SPEC=
  case "$_RES_ADM_PREV" in -v|--volume) _RES_ADM_SPEC=$_RES_ADM_WORD ;; esac
  case "$_RES_ADM_WORD" in -v?*) _RES_ADM_SPEC=${{_RES_ADM_WORD#-v}} ;; --volume=*) _RES_ADM_SPEC=${{_RES_ADM_WORD#--volume=}} ;; esac
  case "$_RES_ADM_SPEC" in *:{ADMISSION_TARGET}|*:{ADMISSION_TARGET}:*) _RES_ADM_MOUNTS=$((_RES_ADM_MOUNTS + 1)) ;; esac
  _RES_ADM_PREV=$_RES_ADM_WORD
done
read -r -a _RES_ADM_VLLM <<<"${{EXTRA_VLLM_ARGS-}}"
_RES_ADM_ARGS=0
for _RES_ADM_WORD in "${{_RES_ADM_VLLM[@]}}"; do
  case "$_RES_ADM_WORD" in --middleware|--middleware=*) _RES_ADM_ARGS=$((_RES_ADM_ARGS + 1)) ;; esac
done
if [ "$_RES_ADM_ENVS" != 0 ] || [ "$_RES_ADM_MOUNTS" != 0 ] || [ "$_RES_ADM_ARGS" != 0 ]; then
  echo "bounded admission requires no pre-existing middleware mount, environment, or argument" >&2
  unset _RES_ADM_DOCKER _RES_ADM_VLLM _RES_ADM_PREV _RES_ADM_WORD _RES_ADM_SPEC \
    _RES_ADM_ENVS _RES_ADM_MOUNTS _RES_ADM_ARGS
  return 1
fi
EXTRA_DOCKER_ENV="${{EXTRA_DOCKER_ENV:+$EXTRA_DOCKER_ENV }}-v {remote_source}:{ADMISSION_TARGET}:ro {environment}"
EXTRA_VLLM_ARGS="${{EXTRA_VLLM_ARGS:+$EXTRA_VLLM_ARGS }}--middleware {ADMISSION_IMPORT}"
unset _RES_ADM_DOCKER _RES_ADM_VLLM _RES_ADM_PREV _RES_ADM_WORD _RES_ADM_SPEC \
  _RES_ADM_ENVS _RES_ADM_MOUNTS _RES_ADM_ARGS
"""


def validate_base_config(value: object) -> dict:
    if not isinstance(value, dict):
        raise ContractError("base SparkCache config must be a JSON object")
    extra = value.get("kv_connector_extra_config")
    if not isinstance(extra, dict):
        raise ContractError("base config lacks kv_connector_extra_config")
    required = {
        "spark_cache_cpu_budget_bytes": 1 << 30,
        "spark_cache_min_available_bytes": 1 << 30,
        "spark_cache_streaming_snapshots": False,
        "spark_cache_cuda_restore": False,
    }
    for key, expected in required.items():
        if extra.get(key) != expected:
            raise ContractError(f"base config requires {key}={expected!r}")
    capacity_keys = {
        "spark_cache_max_bytes", "spark_cache_low_watermark_bytes",
    }
    configured_capacity = sorted(capacity_keys & extra.keys())
    if configured_capacity:
        raise ContractError(
            "base config must leave resilience capacity to the test overlay: "
            + ", ".join(configured_capacity)
        )
    if value.get("kv_load_failure_policy") != "recompute":
        raise ContractError("base config must use recompute after restore failure")
    return value


def validate_wrapper_overrides(base: Path, wrapper: Path = HERE / "connector_wrapper.py") -> None:
    """Refuse accidental overrides of methods in the hash-pinned real connector."""
    try:
        base_tree = ast.parse(base.read_text(encoding="utf-8"))
        wrapper_tree = ast.parse(wrapper.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, SyntaxError) as error:
        raise ContractError(f"cannot inspect connector hooks: {error}") from error
    base_method_nodes = [
        item
        for node in base_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SparkContextCacheConnector"
        for item in node.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    base_methods = {item.name for item in base_method_nodes}
    wrapper_classes = [node for node in wrapper_tree.body
                       if isinstance(node, ast.ClassDef)
                       and node.name == "SparkContextCacheConnector"]
    if len(wrapper_classes) != 1:
        raise ContractError("resilience wrapper must define one connector subclass")
    wrapper_method_nodes = [item for item in wrapper_classes[0].body
                            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))]
    wrapper_methods = {item.name for item in wrapper_method_nodes}
    overrides = wrapper_methods & base_methods
    if overrides != INTENDED_CONNECTOR_HOOKS:
        raise ContractError(
            "resilience connector hook set differs: "
            f"expected {sorted(INTENDED_CONNECTOR_HOOKS)}, got {sorted(overrides)}"
        )

    def call_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple:
        arguments = node.args
        return (
            tuple(item.arg for item in arguments.posonlyargs),
            tuple(item.arg for item in arguments.args),
            arguments.vararg.arg if arguments.vararg else None,
            tuple(item.arg for item in arguments.kwonlyargs),
            arguments.kwarg.arg if arguments.kwarg else None,
            tuple(ast.dump(value, include_attributes=False)
                  for value in arguments.defaults),
            tuple(None if value is None else ast.dump(value, include_attributes=False)
                  for value in arguments.kw_defaults),
        )

    base_finish = [item for item in base_method_nodes if item.name == "_finish_store"]
    wrapper_finish = [item for item in wrapper_method_nodes if item.name == "_finish_store"]
    if (len(base_finish) != 1 or len(wrapper_finish) != 1
            or call_signature(base_finish[0]) != call_signature(wrapper_finish[0])):
        raise ContractError("resilience _finish_store signature differs from base connector")


def overlay_text(campaign_id: str, base_connector_sha: str, wrapper_sha: str,
                 config_sha: str) -> str:
    relative = f"scripts/resilience/.campaign/{campaign_id}"
    remote = f"$HOME/tp4/{relative}"
    container_root = f"/cache/jit/{namespace_name(campaign_id)}"
    return f"""# Generated private resilience overlay. Keep the same TP4_ENV through down.
_RES_TARGET={CONNECTOR_TARGET}
_RES_BASE=${{SPARKCACHE_CONNECTOR-}}
_RES_FROM="$_RES_BASE:$_RES_TARGET:ro"
_RES_WRAPPER='{remote}/connector_wrapper.py'
_RES_TO="$_RES_WRAPPER:$_RES_TARGET:ro"
read -r -a _RES_WORDS <<<"${{EXTRA_DOCKER_ENV-}}"
_RES_PREV= _RES_OUT= _RES_FOUND=0 _RES_TARGETS=0
for _RES_WORD in "${{_RES_WORDS[@]}}"; do
  if [ "$_RES_PREV" = -v ]; then
    case "$_RES_WORD" in *:"$_RES_TARGET"|*:"$_RES_TARGET":*) _RES_TARGETS=$((_RES_TARGETS + 1)) ;; esac
    if [ "$_RES_WORD" = "$_RES_FROM" ]; then
      _RES_WORD=$_RES_TO
      _RES_FOUND=$((_RES_FOUND + 1))
    fi
  fi
  _RES_OUT+="${{_RES_OUT:+ }}$_RES_WORD"
  _RES_PREV=$_RES_WORD
done
if [ "$_RES_FOUND" != 1 ] || [ "$_RES_TARGETS" != 1 ]; then
  echo "resilience overlay requires exactly one active bounded connector mount" >&2
  return 1
fi
EXTRA_DOCKER_ENV="$_RES_OUT -v $_RES_BASE:/opt/tp4-resilience/base_connector.py:ro -v \\{remote}/runtime.py:/opt/tp4-resilience/runtime.py:ro -e TP4_RESILIENCE_RUNTIME=/opt/tp4-resilience/runtime.py -e TP4_RESILIENCE_BASE_CONNECTOR=/opt/tp4-resilience/base_connector.py -e TP4_RESILIENCE_BASE_SHA256={base_connector_sha} -e TP4_RESILIENCE_CAMPAIGN_ID={campaign_id} -e TP4_RESILIENCE_CACHE_ROOT={container_root} -e TP4_RESILIENCE_MAX_BYTES={MAX_NAMESPACE_BYTES} -e SPARK_CONTEXT_CACHE_MAX_BYTES={TEST_CACHE_MAX_BYTES} -e SPARK_CONTEXT_CACHE_LOW_WATERMARK_BYTES={TEST_CACHE_LOW_WATERMARK_BYTES}"
SPARKCACHE_CONNECTOR='{remote}/connector_wrapper.py'
SPARKCACHE_CONNECTOR_SHA256={wrapper_sha}
SPARKCACHE_CONFIG='{remote}/kv-transfer-config.json'
SPARKCACHE_CONFIG_SHA256={config_sha}
unset _RES_TARGET _RES_BASE _RES_FROM _RES_WRAPPER _RES_TO _RES_WORDS _RES_WORD _RES_PREV _RES_OUT _RES_FOUND _RES_TARGETS
"""


def compose_variant(base: Path, components: tuple[str, ...]) -> tuple[bytes, list[dict[str, str]]]:
    parts: list[bytes] = []
    pins: list[dict[str, str]] = []
    for path in (base, *(REPO / relative for relative in components)):
        if path.is_symlink() or not path.is_file():
            raise ContractError(f"runtime overlay component is missing or symlinked: {path}")
        before = path.stat()
        payload = path.read_bytes()
        after = path.stat()
        if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)):
            raise ContractError(f"runtime overlay component changed while reading: {path}")
        try:
            payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ContractError(f"runtime overlay component is not UTF-8: {path}") from error
        parts.append(payload if payload.endswith(b"\n") else payload + b"\n")
        if path != base:
            pins.append({"path": str(path.relative_to(REPO)),
                         "sha256": hashlib.sha256(payload).hexdigest()})
    return b"".join(parts), pins


def prepare(args: argparse.Namespace) -> Path:
    campaign_id = validate_campaign_id(args.campaign_id)
    runtime_variant = getattr(args, "runtime_variant", "default")
    kv_cache_memory_bytes = getattr(args, "kv_cache_memory_bytes", None)
    bounded_admission = getattr(args, "bounded_admission", False)
    if not isinstance(bounded_admission, bool):
        raise ContractError("bounded_admission must be a boolean")
    admission = admission_contract() if bounded_admission else None
    if kv_cache_memory_bytes is not None:
        try:
            kv_cache_memory_bytes = positive_int(kv_cache_memory_bytes)
        except argparse.ArgumentTypeError as error:
            raise ContractError("kv_cache_memory_bytes must be a positive integer") from error
    if runtime_variant not in RUNTIME_VARIANTS:
        raise ContractError("runtime_variant must be default, prefill-cache-trim, or prefill-step-cap")
    base_config_path = Path(args.base_config).expanduser().resolve()
    base_connector = Path(args.base_connector).expanduser().resolve()
    if not base_connector.is_file():
        raise ContractError(f"base connector does not exist: {base_connector}")
    if b"STREAM_CHUNK_BYTES = 8 << 20" not in base_connector.read_bytes():
        raise ContractError("base connector does not expose the validated 8 MiB transfer size")
    validate_wrapper_overrides(base_connector)
    try:
        base_config = validate_base_config(json.loads(base_config_path.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot read base SparkCache config: {error}") from error
    output = REPO / "scripts" / "resilience" / ".campaign" / campaign_id
    if output.exists() and any(output.iterdir()) and not args.force:
        raise ContractError(f"output already exists; use --force after reviewing it: {output}")
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(output, 0o700)
    for derived_name, _ in RUNTIME_VARIANTS.values():
        if derived_name != "delta.env":
            (output / derived_name).unlink(missing_ok=True)
    (output / ADMISSION_STAGED_NAME).unlink(missing_ok=True)
    for name in RUNTIME_FILES:
        shutil.copy2(HERE / name, output / name)
    config = json.loads(json.dumps(base_config))
    config["kv_connector_extra_config"]["spark_cache_root"] = (
        f"/cache/jit/{namespace_name(campaign_id)}"
    )
    atomic_json(output / "kv-transfer-config.json", config)
    wrapper_sha = sha256_file(output / "connector_wrapper.py")
    config_sha = sha256_file(output / "kv-transfer-config.json")
    base_sha = sha256_file(base_connector)
    (output / "delta.env").write_text(
        overlay_text(campaign_id, base_sha, wrapper_sha, config_sha), encoding="utf-8"
    )
    os.chmod(output / "delta.env", 0o600)
    selected_name, components = RUNTIME_VARIANTS[runtime_variant]
    component_pins: list[dict[str, str]] = []
    if components:
        selected_payload, component_pins = compose_variant(output / "delta.env", components)
        (output / selected_name).write_bytes(selected_payload)
        os.chmod(output / selected_name, 0o600)
    if kv_cache_memory_bytes is not None:
        selected_path = output / selected_name
        selected_path.write_bytes(
            selected_path.read_bytes()
            + kv_cache_override_text(kv_cache_memory_bytes).encode("utf-8")
        )
        os.chmod(selected_path, 0o600)
    if admission is not None:
        source = REPO / str(admission["source_path"])
        shutil.copy2(source, output / ADMISSION_STAGED_NAME)
        if sha256_file(output / ADMISSION_STAGED_NAME) != admission["source_sha256"]:
            raise ContractError("staged bounded-admission middleware hash differs")
        os.chmod(output / ADMISSION_STAGED_NAME, 0o600)
        selected_path = output / selected_name
        selected_path.write_bytes(
            selected_path.read_bytes() + admission_override_text(campaign_id).encode("utf-8")
        )
        os.chmod(selected_path, 0o600)
    generated_names = [*RUNTIME_FILES, "kv-transfer-config.json", "delta.env"]
    if selected_name != "delta.env":
        generated_names.append(selected_name)
    if admission is not None:
        generated_names.append(ADMISSION_STAGED_NAME)
    manifest = {
        "schema": "tp4-resilience-overlay/v1",
        "campaign_id": campaign_id,
        "base_connector_sha256": base_sha,
        "base_config_sha256": sha256_file(base_config_path),
        "namespace": f"/cache/jit/{namespace_name(campaign_id)}",
        "max_namespace_bytes": MAX_NAMESPACE_BYTES,
        "runtime_variant": runtime_variant,
        "selected_delta": selected_name,
        "variant_components": component_pins,
        "files": {name: sha256_file(output / name) for name in generated_names},
    }
    if kv_cache_memory_bytes is not None:
        manifest["kv_cache_memory_bytes"] = kv_cache_memory_bytes
    if admission is not None:
        manifest["bounded_admission"] = admission
    atomic_json(output / "manifest.json", manifest)
    return output


def remote_init_command(campaign_id: str) -> str:
    relative = f"tp4/scripts/resilience/.campaign/{campaign_id}"
    remote_faultctl = f'$HOME/{relative}/faultctl.py'
    namespace = namespace_name(campaign_id)
    return (
        '. "$HOME/tp4/cluster.env"; '
        'case "$CACHE_DIR" in "$HOME"/*) cache_dir=$CACHE_DIR ;; '
        r''' '$HOME'/*) cache_dir="$HOME/${CACHE_DIR#\$HOME/}" ;; '''
        '/*) cache_dir=$CACHE_DIR ;; *) exit 2 ;; esac; '
        f'sudo -n install -d -o "$(id -u)" -g "$(id -g)" -m 700 '
        f'"$cache_dir/jit/{namespace}"; '
        f'sudo -n python3 "{remote_faultctl}" --root "$cache_dir/jit/{namespace}" '
        f'--campaign-id {campaign_id} init'
    )


def stage(output: Path, hosts: list[str], campaign_id: str) -> None:
    if len(hosts) != 4 or len(set(hosts)) != 4:
        raise ContractError("--stage-hosts requires four distinct rank-ordered SSH targets")
    relative = f"tp4/scripts/resilience/.campaign/{campaign_id}"
    for host in hosts:
        subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
                        "-o", "ConnectTimeout=10", host,
                        f'mkdir -p "$HOME/{relative}" && chmod 700 "$HOME/{relative}"'], check=True)
        for path in sorted(output.iterdir()):
            if path.is_file():
                subprocess.run(["scp", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
                                "-o", "ConnectTimeout=10", str(path), f"{host}:{relative}/{path.name}"], check=True)
        manifest_sha = sha256_file(output / "manifest.json")
        verifier = (
            "import hashlib,json,pathlib,sys;"
            "root=pathlib.Path.home()/sys.argv[1];"
            "manifest=root/'manifest.json';"
            "actual=hashlib.sha256(manifest.read_bytes()).hexdigest();"
            "expected=sys.argv[2];"
            "data=json.loads(manifest.read_text());"
            "bad=[n for n,h in data['files'].items() if "
            "hashlib.sha256((root/n).read_bytes()).hexdigest()!=h];"
            "sys.exit(0 if actual==expected and not bad else 1)"
        )
        verify_command = (f"python3 -c {shlex.quote(verifier)} "
                          f"{shlex.quote(relative)} {manifest_sha}")
        subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
                        "-o", "ConnectTimeout=10", host, verify_command], check=True)
        subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
                        "-o", "ConnectTimeout=10", host, remote_init_command(campaign_id)], check=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--campaign-id", required=True)
    result.add_argument("--base-config", required=True)
    result.add_argument("--base-connector", required=True)
    result.add_argument("--runtime-variant", choices=tuple(RUNTIME_VARIANTS), default="default")
    result.add_argument("--kv-cache-memory-bytes", type=positive_int,
                        help="test-only per-rank KV pool size in bytes")
    result.add_argument("--bounded-admission", action="store_true",
                        help="enable the pinned test-only bounded HTTP admission middleware")
    result.add_argument("--force", action="store_true")
    result.add_argument("--stage-hosts", nargs=4, metavar=("R0", "R1", "R2", "R3"))
    result.add_argument("--apply-stage", action="store_true",
                        help="copy the generated private bundle and initialize empty namespaces")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        output = prepare(args)
        if args.apply_stage:
            if not args.stage_hosts:
                raise ContractError("--apply-stage requires --stage-hosts")
            stage(output, args.stage_hosts, args.campaign_id)
        elif args.stage_hosts:
            raise ContractError("--stage-hosts has no effect without --apply-stage")
        print(output.relative_to(REPO))
        return 0
    except (ContractError, OSError, subprocess.SubprocessError) as error:
        print(f"prepare-overlay: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
