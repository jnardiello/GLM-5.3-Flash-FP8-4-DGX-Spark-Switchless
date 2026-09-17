#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import stat
import sys
import tempfile
from copy import deepcopy
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("check_f0", REPO / "scripts/check-f0.py")
assert SPEC and SPEC.loader
check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check)


def recipe() -> dict[str, str]:
    return {
        "nodes": "n0 n1 n2 n3",
        "base_nodes": "n0 n1 n2 n3",
        "tp4_hosts": "",
        "base_tp4_hosts": "",
        "mgmt_ips": "192.0.2.1 192.0.2.2 192.0.2.3 192.0.2.4",
        "base_mgmt_ips": "192.0.2.1 192.0.2.2 192.0.2.3 192.0.2.4",
        "hosts": "n0 n1 n2 n3",
        "base_hosts": "n0 n1 n2 n3",
        "master_ip": "192.0.2.1",
        "base_master_ip": "192.0.2.1",
        "master_port": "29520",
        "base_master_port": "29520",
        "api_port": "8000",
        "base_api_port": "8000",
        "container": "tp4",
        "base_container": "tp4",
        "image": "example/f0:tag",
        "image_digest": "example/f0@sha256:abc",
        "model_dir": "$HOME/model",
        "base_model_dir": "$HOME/model",
        "model_repo": "zai-org/GLM-5.3-Flash",
        "model_rev": "690b705278a3a58e538fcb37c2ca8b5f9511213c",
        "draft_rev": "bf582e4eacc1810f76656d1811693ff6c6737d2a",
        "draft_dir": "$HOME/draft",
        "base_draft_dir": "$HOME/draft",
        "served_name": "glm-5.3-flash",
        "base_served_name": "glm-5.3-flash",
        "max_model_len": "262144",
        "max_num_seqs": "6",
        "kv_cache_dtype": "fp8_e4m3",
        "batched_tokens": "8192",
        "spec_tokens": "5",
        "spec_extra_json": '"num_speculative_tokens_per_batch_size":[[1,1,5],[2,6,3]]',
        "async_scheduling": "0",
        "extra_docker_env": "-e VLLM_ADAPTIVE_K_MODE=per-request",
        "base_extra_docker_env": "-e VLLM_ADAPTIVE_K_MODE=per-request",
        "extra_vllm_args": (
            "--moe-backend triton "
            "--scheduler-cls adaptive_k_scheduler.AdaptiveKScheduler"
        ),
        "base_extra_vllm_args": (
            "--moe-backend triton "
            "--scheduler-cls adaptive_k_scheduler.AdaptiveKScheduler"
        ),
        "fabric_prefix_re": "",
        "base_fabric_prefix_re": "",
        **{f"gid_index_{rank}": "-1" for rank in range(4)},
        **{f"hca_{rank}": "rocep1s0f0,rocep1s0f1" for rank in range(4)},
        **{f"base_hca_{rank}": "rocep1s0f0,rocep1s0f1" for rank in range(4)},
        **{f"mgmt_if_{rank}": "mgmt0" for rank in range(4)},
        **{f"base_mgmt_if_{rank}": "mgmt0" for rank in range(4)},
        **{f"fabric_ifaces_{rank}": "fab0 fab1" for rank in range(4)},
        **{f"base_fabric_ifaces_{rank}": "fab0 fab1" for rank in range(4)},
        **{
            f"base_fabric_target_{rank}": f"10.42.{rank * 2}.2 10.42.{rank * 2 + 1}.2"
            for rank in range(4)
        },
        **{
            f"fabric_target_{rank}": f"10.42.{rank * 2}.2 10.42.{rank * 2 + 1}.2"
            for rank in range(4)
        },
    }


def expected() -> dict:
    value = check.expected_f0()
    value["image_digest"] = "example/f0@sha256:abc"
    return value


def probe(rank: int) -> dict:
    exp = expected()
    rec = recipe()
    command = [
        "/model",
        "--served-model-name", rec["served_name"],
        "--tensor-parallel-size", "4",
        "--nnodes", "4",
        "--node-rank", str(rank),
        "--master-addr", rec["master_ip"],
        "--master-port", rec["master_port"],
        "--max-model-len", exp["max_model_len"],
        "--max-num-seqs", exp["max_num_seqs"],
        "--max-num-batched-tokens", exp["batched_tokens"],
        "--kv-cache-dtype", exp["kv_cache_dtype"],
        "--speculative-config", json.dumps({
            "method": "dflash",
            "model": "/draft",
            "num_speculative_tokens": 5,
            "num_speculative_tokens_per_batch_size": [[1, 1, 5], [2, 6, 3]],
        }),
        "--scheduler-cls", "adaptive_k_scheduler.AdaptiveKScheduler",
        "--moe-backend", "triton",
    ]
    option_names = (
        "--served-model-name", "--tensor-parallel-size", "--nnodes", "--node-rank",
        "--master-addr", "--master-port",
        "--max-model-len", "--max-num-seqs", "--max-num-batched-tokens",
        "--kv-cache-dtype", "--scheduler-cls", "--moe-backend",
    )
    remote = {
        "errors": [],
        "running_container_count": 1,
        "foreign_gpu_container_count": 0,
        "foreign_gpu_pid_count": 0,
        "container": {
            "image_reference": rec["image"],
            "image_digests": [exp["image_digest"]],
            "model_path": "/model",
            "options": {name: check.flag_values(command, name) for name in option_names},
            "async_flag_count": 0,
            "speculative": json.loads(check.flag_values(command, "--speculative-config")[0]),
            "model_marker": exp["model_rev"],
            "draft_commit": exp["draft_rev"] if rank == 0 else "",
            "draft_metadata_present": rank == 0,
            "draft_config_sha": "a" * 64,
            "model_mount": True,
            "draft_mount": True,
            "environment": {
                "VLLM_ADAPTIVE_K_MODE": "per-request",
                "NCCL_ALGO": "Ring",
                "NCCL_IB_HCA": rec[f"hca_{rank}"],
                "NCCL_IB_GID_INDEX": "-1",
                "NCCL_IB_ROCE_VERSION_NUM": "2",
                "NCCL_IB_ADDR_FAMILY": "AF_INET",
            },
        },
        "flusher": {"unit_state": "inactive", "unit_rc": 3, "legacy_process": False},
        "fabric_interfaces": {"f0": "9000", "f1": "9000"},
        "jumbo_pings": [True, True],
    }
    return {
        "rank": rank,
        "verify_node": {"returncode": 0, "stdout": "PASS", "stderr": "", "timed_out": False},
        "remote": remote,
        "remote_command": {"returncode": 0, "stdout": "{}", "stderr": "", "timed_out": False},
    }


def endpoint(*, waiting: float | list[float] = 0,
             running: float | list[float] = 0) -> dict:
    return {
        "/health_status": 200,
        "vllm:num_requests_running": running if isinstance(running, list) else [running],
        "vllm:num_requests_waiting": waiting if isinstance(waiting, list) else [waiting],
        "errors": [],
    }


rec = recipe()
exp = expected()
healthy = [probe(rank) for rank in range(4)]
assert check.evaluate(rec, exp, healthy, endpoint()) == []

missing = healthy[:3]
assert any("missing probe" in item for item in check.evaluate(rec, exp, missing, endpoint()))

mismatch = deepcopy(rec)
mismatch["batched_tokens"] = "16384"
assert any("batched_tokens" in item for item in check.evaluate(mismatch, exp, healthy, endpoint()))

topology = deepcopy(rec)
topology["hosts"] = "wrong n1 n2 n3"
assert any("protected field: hosts" in item for item in check.recipe_problems(topology, exp))

for field in ("nodes", "tp4_hosts", "mgmt_ips", "master_port", "fabric_prefix_re"):
    changed = deepcopy(rec)
    changed[field] += " changed"
    assert any(f"protected field: {field}" in item
               for item in check.recipe_problems(changed, exp))

for field in ("mgmt_if", "fabric_ifaces", "hca"):
    changed = deepcopy(rec)
    changed[f"{field}_2"] += " changed"
    assert any(f"protected {field}: rank 2" in item
               for item in check.recipe_problems(changed, exp))

extra_env = deepcopy(rec)
extra_env["extra_docker_env"] += " -e UNREVIEWED_FLAG=1"
assert any("protected field: extra_docker_env" in item
           for item in check.recipe_problems(extra_env, exp))

extra_args = deepcopy(rec)
extra_args["extra_vllm_args"] += " --unreviewed-flag"
assert any("protected field: extra_vllm_args" in item
           for item in check.recipe_problems(extra_args, exp))

duplicate = deepcopy(healthy)
duplicate[0]["remote"]["container"]["options"]["--served-model-name"].append(rec["served_name"])
assert any("--served-model-name" in item for item in check.evaluate(rec, exp, duplicate, endpoint()))

wrong_master = deepcopy(healthy)
wrong_master[2]["remote"]["container"]["options"]["--master-port"] = ["29521"]
assert any("rank 2: command --master-port" in item
           for item in check.evaluate(rec, exp, wrong_master, endpoint()))

assert check.flag_values(["--option", "value", "--option"], "--option") == ["value", None]
assert check.flag_values(["--option", "--next", "value"], "--option") == [None]
trailing_option = deepcopy(healthy)
trailing_option[3]["remote"]["container"]["options"]["--master-addr"].append(None)
assert any("rank 3: command --master-addr" in item
           for item in check.evaluate(rec, exp, trailing_option, endpoint()))

adaptive = deepcopy(healthy)
adaptive[0]["remote"]["container"]["environment"]["VLLM_ADAPTIVE_K_ENABLE"] = "0"
assert any("adaptive policy" in item for item in check.evaluate(rec, exp, adaptive, endpoint()))

foreign = deepcopy(healthy)
foreign[2]["remote"]["foreign_gpu_pid_count"] = 1
assert any("foreign GPU" in item for item in check.evaluate(rec, exp, foreign, endpoint()))

bad_probe = deepcopy(healthy)
bad_probe[1]["remote"] = {"probe_failed": "timeout"}
assert any("probe timeout" in item for item in check.evaluate(rec, exp, bad_probe, endpoint()))

assert any("endpoint idle" in item for item in check.evaluate(rec, exp, healthy, endpoint(waiting=1)))
assert any("endpoint idle" in item for item in check.evaluate(rec, exp, healthy, endpoint(waiting=-1)))
assert any("endpoint idle" in item for item in check.evaluate(
    rec, exp, healthy, endpoint(running=[1, -1])))
assert any("endpoint idle" in item for item in check.evaluate(
    rec, exp, healthy, endpoint(waiting=[])))
assert any("endpoint idle" in item for item in check.evaluate(
    rec, exp, healthy, endpoint(waiting=[float("nan")])))


class FakeResponse:
    def __init__(self, body: bytes):
        self.status, self.body = 200, body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _limit: int) -> bytes:
        return self.body


original_urlopen = check.urllib.request.urlopen
check.urllib.request.urlopen = lambda url, timeout: FakeResponse(
    (b'vllm:num_requests_running{engine="0"} 1\n'
     b'vllm:num_requests_running{engine="1"} -1\n'
     b'vllm:num_requests_waiting 0\n'
     b'vllm:num_requests_waiting_by_reason{reason="capacity"} 0\n'
     b'vllm:num_requests_waiting_by_reason{reason="deferred"} 0\n'
     b'vllm:num_requests_waiting malformed metric\n') if url.endswith("/metrics") else b""
)
try:
    parsed_endpoint = check.endpoint_probe("http://127.0.0.1:8000", 1)
finally:
    check.urllib.request.urlopen = original_urlopen
assert parsed_endpoint["vllm:num_requests_running"] == [1.0, -1.0]
assert parsed_endpoint["vllm:num_requests_waiting"] == [0.0]
assert {"path": "/metrics", "error": "invalid metric"} in parsed_endpoint["errors"]
assert parsed_endpoint["errors"].count(
    {"path": "/metrics", "error": "invalid metric"}) == 1
assert any("endpoint idle" in item
           for item in check.evaluate(rec, exp, healthy, parsed_endpoint))

unproven_draft = deepcopy(healthy)
unproven_draft[0]["remote"]["container"]["draft_commit"] = ""
assert any("drafter revision" in item for item in check.evaluate(rec, exp, unproven_draft, endpoint()))

wrong_draft = deepcopy(healthy)
wrong_draft[1]["remote"]["container"]["draft_commit"] = "wrong-revision"
wrong_draft[1]["remote"]["container"]["draft_metadata_present"] = True
assert any("rank 1: drafter revision marker" in item
           for item in check.evaluate(rec, exp, wrong_draft, endpoint()))

shared_config_fallback = deepcopy(healthy)
shared_config_fallback[1]["remote"]["container"]["draft_commit"] = ""
assert check.evaluate(rec, exp, shared_config_fallback, endpoint()) == []

different_draft_config = deepcopy(shared_config_fallback)
different_draft_config[1]["remote"]["container"]["draft_config_sha"] = "b" * 64
assert any("rank 1: drafter config identity" in item
           for item in check.evaluate(rec, exp, different_draft_config, endpoint()))

sanitized = check.reportable_rank_probe({
    **healthy[0],
    "verify_node": {**healthy[0]["verify_node"], "stdout": "raw", "stderr": "raw"},
    "remote_command": {**healthy[0]["remote_command"], "stdout": "raw", "stderr": "raw"},
})
assert "stdout" not in sanitized["verify_node"] and "stderr" not in sanitized["verify_node"]
assert "stdout" not in sanitized["remote_command"] and "stderr" not in sanitized["remote_command"]

for value in ("nan", "inf", "301"):
    try: check.positive(value)
    except check.argparse.ArgumentTypeError: pass
    else: raise AssertionError(f"accepted unbounded timeout: {value}")

try: check.checked_base_url("http://user:secret@127.0.0.1:8000")
except check.CheckFailure: pass
else: raise AssertionError("accepted credentials in base URL")

timed = check.run_command(
    [sys.executable, "-c", "import time; time.sleep(0.2)"], timeout=0.01
)
assert timed["timed_out"] is True

with tempfile.TemporaryDirectory(prefix="tp4-check-f0-test.") as temp:
    report_dir = check.make_report_dir(REPO, Path(temp))
    secret = "private-host.example"
    report_path = check.write_report(report_dir, {"private": secret})
    assert report_path.parent == report_dir and not report_path.is_relative_to(REPO)
    assert stat.S_IMODE(report_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(report_path.stat().st_mode) == 0o600
    assert secret in report_path.read_text(encoding="utf-8")
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        check.print_summary(False)
    assert output.getvalue() == "F0 CHECK FAIL\n" and secret not in output.getvalue()

try:
    check.make_report_dir(REPO, REPO / "tmp-report")
except check.CheckFailure:
    pass
else:
    raise AssertionError("accepted a private report inside the checkout")

output = io.StringIO()
with contextlib.redirect_stdout(output):
    assert check.main(["--report-root", str(REPO)]) == 1
assert output.getvalue() == "F0 CHECK FAIL\n"

print("test-check-f0: PASS")
