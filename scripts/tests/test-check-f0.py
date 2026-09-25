#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import shutil
import stat
import sys
import tempfile
from copy import deepcopy
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
BASELINES = REPO / "docs/historical_benchmarks/baselines"
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
        "sparkcache_mode": "off",
        "extra_docker_env": "-e VLLM_ADAPTIVE_K_MODE=per-request",
        "base_extra_docker_env": "-e VLLM_ADAPTIVE_K_MODE=per-request",
        "extra_vllm_args": (
            "--kv-cache-memory-bytes=17179869184 "
            "--moe-backend triton "
            "--scheduler-cls adaptive_k_scheduler.AdaptiveKScheduler"
        ),
        "base_extra_vllm_args": (
            "--kv-cache-memory-bytes=17179869184 "
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
    value = check.expected_f0(BASELINES / "2026-09-11/baseline.json")
    value["image_digest"] = "example/f0@sha256:abc"
    # The fixtures describe the per-request policy regardless of which frozen baseline the
    # checkout carries; the baseline-driven expectation is covered separately below.
    value["adaptive_env"] = dict(check.ADAPTIVE_DEFAULTS)
    return value


def test_baseline_adaptive_policy_is_read_from_the_baseline() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        data = json.loads((BASELINES / "2026-09-11/baseline.json").read_text(encoding="utf-8"))
        data["system"]["adaptive_k"] = {"mode": "batch-uniform", "k_lo": 3, "k_hi": 5}
        data["system"]["serving_image_digest"] = "example/f1@sha256:def"
        path = Path(tmp) / "baseline.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        value = check.expected_f0(path)
    assert value["adaptive_env"]["VLLM_ADAPTIVE_K_MODE"] == "batch-uniform"
    assert value["adaptive_env"]["VLLM_ADAPTIVE_K_LO"] == "3"
    assert value["image_digest"] == "example/f1@sha256:def"


test_baseline_adaptive_policy_is_read_from_the_baseline()


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
        "--kv-cache-memory-bytes", exp["kv_cache_memory_bytes"],
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
        "--kv-cache-memory-bytes", "--prefill-schedule-interval",
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
extra_env["extra_docker_env"] += " -e VLLM_ADAPTIVE_K_UP=0.9"
assert any("effective baseline adaptive policy" in item
           for item in check.recipe_problems(extra_env, exp))

extra_args = deepcopy(rec)
extra_args["extra_vllm_args"] += " --kv-cache-memory-bytes=17179869184"
assert any("baseline KV memory budget" in item
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
    assert output.getvalue() == "BASELINE CHECK FAIL\n" and secret not in output.getvalue()

try:
    check.make_report_dir(REPO, REPO / "tmp-report")
except check.CheckFailure:
    pass
else:
    raise AssertionError("accepted a private report inside the checkout")

output = io.StringIO()
with contextlib.redirect_stdout(output):
    assert check.main(["--report-root", str(REPO)]) == 1
assert output.getvalue() == "BASELINE CHECK FAIL\n"


def baseline_fixture(path: Path) -> tuple[dict, dict, list]:
    exp = check.expected_f0(path)
    rec = recipe()
    rec.update(image=exp["image_digest"], image_digest=exp["image_digest"],
               sparkcache_mode=exp["sparkcache_mode"],
               spark_mhc_prefill_shard=exp["spark_mhc_prefill_shard"], **exp["payload_pins"])
    scheduler_env = {key: value for key, value in exp["runtime_identity"].get("environment", {}).items()
                     if key in check.SCHEDULER_FLAGS}
    for name in ("extra_docker_env", "base_extra_docker_env"):
        rec[name] = " ".join(f"-e {key}={value}" for key, value in
                             {**exp["adaptive_env"], **scheduler_env}.items())
    for name in ("extra_vllm_args", "base_extra_vllm_args"):
        rec[name] = rec[name].replace("17179869184", exp["kv_cache_memory_bytes"])
        if exp["prefill_schedule_interval"] != "1":
            rec[name] += " --prefill-schedule-interval " + exp["prefill_schedule_interval"]
    ranks = [probe(rank) for rank in range(4)]
    identity = exp["runtime_identity"]
    for rank, item in enumerate(ranks):
        c = item["remote"]["container"]
        c.update(image_reference=rec["image"], image_digests=[exp["image_digest"]],
                 image_id=exp["image_id"], runtime_files=deepcopy(identity.get("container_file_sha256", {})))
        c["options"]["--kv-cache-memory-bytes"] = [exp["kv_cache_memory_bytes"]]
        c["options"]["--prefill-schedule-interval"] = check.interval_flag(exp)
        c["environment"].update(exp["adaptive_env"])
        c["environment"].update(identity.get("environment", {}))
        c["runtime_workers"] = [{"pid": 100 + rank, "patched_nccl_loaded": True}]
        kda = deepcopy(identity.get("kda_boot_receipt", {}))
        padded_n = kda.pop("padded_n", None)
        if padded_n is not None:
            kda["receipts"] = [{"padded_n": padded_n} for _ in range(kda["modules"])]
        c["runtime_receipts"] = {"scheduler_boot_signature": True, "kda": kda, "memory_probe": {
            **identity.get("memory_probe", {}), "rank": rank}}
        if identity.get("e21_boot_receipt"):
            c["runtime_receipts"]["e21"] = deepcopy(identity["e21_boot_receipt"])
        if identity.get("e22_boot_receipt"):
            c["runtime_receipts"]["e22"] = deepcopy(identity["e22_boot_receipt"])
        if identity.get("boot_lines"):
            c["runtime_receipts"]["boot_lines"] = list(identity["boot_lines"])
    return rec, exp, ranks


assert check.BASELINE == BASELINES / "2026-09-25-e27c/baseline.json"
HISTORICAL = ("2026-09-11", "2026-09-18", "2026-09-19", "2026-09-19-e03", "2026-09-23-e21",
              "2026-09-23-e22b", "2026-09-24-e27", "2026-09-25-e27c")
for name in HISTORICAL:
    baseline_path = BASELINES / name / "baseline.json"
    baseline_recipe, baseline_expected, baseline_ranks = baseline_fixture(baseline_path)
    assert check.evaluate(baseline_recipe, baseline_expected, baseline_ranks, endpoint()) == [], name
    # Records before E27 ran without the cadence and must refuse it; E27 requires it.
    assert baseline_expected["prefill_schedule_interval"] == (
        "8" if name in ("2026-09-24-e27", "2026-09-25-e27c") else "1"), name
    # Records before E27c ran the image's scheduler and must refuse the E27c flags.
    flags = set(baseline_expected["runtime_identity"].get("environment", {})) & set(check.SCHEDULER_FLAGS)
    assert flags == (set(check.SCHEDULER_FLAGS) if name == "2026-09-25-e27c" else set()), name

current_recipe, current_expected, current_ranks = baseline_fixture(check.BASELINE)
assert current_expected["baseline_id"] == "2026-09-25-e27c"
scheduler_target = "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py"
assert scheduler_target in current_expected["runtime_identity"]["container_file_sha256"]
for key in check.SCHEDULER_FLAGS:
    missing = deepcopy(current_ranks)
    missing[1]["remote"]["container"]["environment"].pop(key)
    assert f"rank 1: runtime environment {key}" in check.evaluate(
        current_recipe, current_expected, missing, endpoint()), key
    missing_recipe = deepcopy(current_recipe)
    missing_recipe["extra_docker_env"] = missing_recipe["extra_docker_env"].replace(
        f" -e {key}=" + current_expected["runtime_identity"]["environment"][key], "")
    assert "baseline runtime environment: " + key in check.recipe_problems(
        missing_recipe, current_expected), key
wrong_scheduler = deepcopy(current_ranks)
wrong_scheduler[2]["remote"]["container"]["runtime_files"][scheduler_target] = "0" * 64
assert f"rank 2: runtime file {scheduler_target}" in check.evaluate(
    current_recipe, current_expected, wrong_scheduler, endpoint())
for signature in current_expected["runtime_identity"]["boot_lines"]:
    unsigned = deepcopy(current_ranks)
    unsigned[0]["remote"]["container"]["runtime_receipts"]["boot_lines"].remove(signature)
    assert "rank 0: boot signature " + signature in check.evaluate(
        current_recipe, current_expected, unsigned, endpoint()), signature
# The E27 record refuses a running or configured E27c scheduler.
e27_recipe, e27_expected, e27_ranks = baseline_fixture(BASELINES / "2026-09-24-e27/baseline.json")
for key in check.SCHEDULER_FLAGS:
    extra = deepcopy(e27_ranks)
    extra[1]["remote"]["container"]["environment"][key] = "1"
    assert f"rank 1: unexpected runtime environment {key}" in check.evaluate(
        e27_recipe, e27_expected, extra, endpoint()), key
    extra_recipe = deepcopy(e27_recipe)
    extra_recipe["extra_docker_env"] += f" -e {key}=1"
    assert "baseline runtime environment: unexpected " + key in check.recipe_problems(
        extra_recipe, e27_expected), key
assert current_expected["prefill_schedule_interval"] == "8"
for bad in ("", " --prefill-schedule-interval 4"):
    wrong_interval = deepcopy(current_recipe)
    wrong_interval["extra_vllm_args"] = wrong_interval["extra_vllm_args"].replace(
        " --prefill-schedule-interval 8", bad)
    assert "baseline prefill schedule interval" in check.recipe_problems(wrong_interval, current_expected), bad
for bad in ([], ["4"], ["8", "8"]):
    wrong_live_interval = deepcopy(current_ranks)
    wrong_live_interval[1]["remote"]["container"]["options"]["--prefill-schedule-interval"] = bad
    assert "rank 1: command --prefill-schedule-interval" in check.evaluate(
        current_recipe, current_expected, wrong_live_interval, endpoint()), bad
assert current_expected["runtime_identity"]["e21_boot_receipt"]["modules"] == 67
assert current_expected["runtime_identity"]["e22_boot_receipt"]["modules"] == 30
assert current_expected["runtime_identity"]["e22_boot_receipt"]["context_kv_w8a16"] is False
assert current_expected["kv_cache_memory_bytes"] == "16106127360"
assert current_expected["runtime_identity"]["kda_boot_receipt"]["prefill_bf16_min_tokens"] == 2048

wrong_kv = deepcopy(current_recipe)
for name in ("extra_vllm_args", "base_extra_vllm_args"):
    wrong_kv[name] = wrong_kv[name].replace("16106127360", "17179869184")
assert "baseline KV memory budget" in check.recipe_problems(wrong_kv, current_expected)
wrong_live_kv = deepcopy(current_ranks)
wrong_live_kv[2]["remote"]["container"]["options"]["--kv-cache-memory-bytes"] = ["17179869184"]
assert any("rank 2: command --kv-cache-memory-bytes" in problem for problem in
           check.evaluate(current_recipe, current_expected, wrong_live_kv, endpoint()))

for path in current_expected["runtime_identity"]["container_file_sha256"]:
    wrong_file = deepcopy(current_ranks)
    wrong_file[1]["remote"]["container"]["runtime_files"][path] = "0" * 64
    assert f"rank 1: runtime file {path}" in check.evaluate(
        current_recipe, current_expected, wrong_file, endpoint())

for field, value in (("modules", 33), ("prefill_bf16_min_tokens", 1024),
                     ("shared_scratch_bytes", 0), ("packed_bytes", 0)):
    wrong_kda = deepcopy(current_ranks)
    wrong_kda[0]["remote"]["container"]["runtime_receipts"]["kda"][field] = value
    assert f"rank 0: KDA boot receipt {field}" in check.evaluate(
        current_recipe, current_expected, wrong_kda, endpoint())

for field, value in (("modules", 34), ("added_scratch_bytes", 0), ("mla_layout", "no_q_lora"),
                     ("families", ["kda_o_proj"])):
    wrong_e21 = deepcopy(current_ranks)
    wrong_e21[0]["remote"]["container"]["runtime_receipts"]["e21"][field] = value
    assert f"rank 0: E21 boot receipt {field}" in check.evaluate(
        current_recipe, current_expected, wrong_e21, endpoint())
missing_e21 = deepcopy(current_ranks)
missing_e21[3]["remote"]["container"]["runtime_receipts"].pop("e21")
assert "rank 3: E21 boot receipt modules" in check.evaluate(
    current_recipe, current_expected, missing_e21, endpoint())
no_flag = deepcopy(current_ranks)
no_flag[1]["remote"]["container"]["environment"].pop("VLLM_E21_BF16_RESIDUE_W8A16")
assert "rank 1: runtime environment VLLM_E21_BF16_RESIDUE_W8A16" in check.evaluate(
    current_recipe, current_expected, no_flag, endpoint())

for field, value in (("modules", 31), ("context_kv_w8a16", True), ("added_scratch_bytes", 20971520),
                     ("families", ["qkv_proj"])):
    wrong_e22 = deepcopy(current_ranks)
    wrong_e22[0]["remote"]["container"]["runtime_receipts"]["e22"][field] = value
    assert f"rank 0: E22 boot receipt {field}" in check.evaluate(
        current_recipe, current_expected, wrong_e22, endpoint())
missing_e22 = deepcopy(current_ranks)
missing_e22[2]["remote"]["container"]["runtime_receipts"].pop("e22")
assert "rank 2: E22 boot receipt modules" in check.evaluate(
    current_recipe, current_expected, missing_e22, endpoint())
for flag in ("VLLM_E22_DRAFTER_W8A16", "VLLM_E22_CONTEXT_KV_W8A16"):
    no_e22_flag = deepcopy(current_ranks)
    no_e22_flag[3]["remote"]["container"]["environment"].pop(flag)
    assert f"rank 3: runtime environment {flag}" in check.evaluate(
        current_recipe, current_expected, no_e22_flag, endpoint())

wrong_padding = deepcopy(current_ranks)
wrong_padding[0]["remote"]["container"]["runtime_receipts"]["kda"]["receipts"][5]["padded_n"] = 6288
assert "rank 0: KDA padding receipt" in check.evaluate(
    current_recipe, current_expected, wrong_padding, endpoint())

wrong_probe = deepcopy(current_ranks)
wrong_probe[3]["remote"]["container"]["runtime_receipts"]["memory_probe"]["rank"] = 0
assert "rank 3: memory probe identity" in check.evaluate(
    current_recipe, current_expected, wrong_probe, endpoint())

wrong_nccl = deepcopy(current_ranks)
wrong_nccl[1]["remote"]["container"]["runtime_workers"][0]["patched_nccl_loaded"] = False
assert "rank 1: patched NCCL not loaded by GPU worker" in check.evaluate(
    current_recipe, current_expected, wrong_nccl, endpoint())

oci_import = deepcopy(current_ranks)
for item in oci_import:
    item["remote"]["container"]["image_digests"] = []
assert check.evaluate(current_recipe, current_expected, oci_import, endpoint()) == []
oci_import[0]["remote"]["container"]["image_id"] = "sha256:wrong"
assert "rank 0: running image content ID" in check.evaluate(
    current_recipe, current_expected, oci_import, endpoint())

# Exercise --baseline through main, not just the expectation loader: otherwise an
# ignored CLI argument silently compares historical deployments with the default.
original_load = check.load_recipe
try:
    with tempfile.TemporaryDirectory(prefix="tp4-baseline-selection.") as temp:
        for name in HISTORICAL:
            path = BASELINES / name / "baseline.json"
            rec, exp, ranks = baseline_fixture(path)
            check.load_recipe = lambda timeout: (deepcopy(rec), {"returncode": 0})
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                rc = check.main(["--baseline", str(path), "--report-root", temp],
                                rank_probe=lambda rank, host, recipe, timeout: deepcopy(ranks[rank]),
                                http_probe=lambda url, timeout: endpoint())
            assert rc == 0, name
            assert output.getvalue() == f"{exp['baseline_id']} CHECK PASS\n"
finally:
    check.load_recipe = original_load

compile(check.REMOTE_PROBE, "remote-identity-probe", "exec")

# Load the real current template and the shipped historical overlays. The base
# stays current while each overlay's effective runtime must match its own record.
original_repo = check.REPO
saved_overlay = os.environ.get("TP4_ENV")
try:
    with tempfile.TemporaryDirectory(prefix="tp4-baseline-overlay.") as temp:
        isolated = Path(temp)
        for relative in ("scripts/lib/common.sh", "scripts/node/bootstrap/versions.env",
                         "scripts/node/reference/baseline-20260924-e27.env",
                         "scripts/node/reference/baseline-20260924-e22b.env",
                         "scripts/node/reference/baseline-20260923-e21.env",
                         "scripts/node/reference/baseline-20260919-e03.env",
                         "scripts/node/reference/baseline-20260919.env",
                         "scripts/node/reference/baseline-20260918.env",
                         "scripts/node/reference/f0-20260912.env", "cluster.env.example"):
            target = isolated / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / relative, target)
        config = (REPO / "cluster.env.example").read_text(encoding="utf-8") + '''
NODES="n0 n1 n2 n3"
MGMT_IPS="192.0.2.21 192.0.2.22 192.0.2.23 192.0.2.24"
MASTER_IP=192.0.2.21
RELAY_DEST=operator@192.0.2.23
'''
        check.REPO = isolated
        for overlay, baseline in (
            (None, "2026-09-25-e27c"),
            ("scripts/node/reference/baseline-20260924-e27.env", "2026-09-24-e27"),
            ("scripts/node/reference/baseline-20260924-e22b.env", "2026-09-23-e22b"),
            ("scripts/node/reference/baseline-20260923-e21.env", "2026-09-23-e21"),
            ("scripts/node/reference/baseline-20260919-e03.env", "2026-09-19-e03"),
            ("scripts/node/reference/baseline-20260919.env", "2026-09-19"),
            ("scripts/node/reference/baseline-20260918.env", "2026-09-18"),
            ("scripts/node/reference/f0-20260912.env", "2026-09-11"),
        ):
            # The frozen F0 overlay predates SparkCache. An archived F0 source
            # restores an OFF base; it is not a complete lane switch over ON.
            base = config + ('\nSPARKCACHE_MODE=off\nSPARK_MHC_PREFILL_SHARD=0\n' if baseline == "2026-09-11" else "")
            (isolated / "cluster.env").write_text(base, encoding="utf-8")
            if overlay:
                os.environ["TP4_ENV"] = overlay
            else:
                os.environ.pop("TP4_ENV", None)
            effective, diagnostic = check.load_recipe(10)
            selected = check.expected_f0(BASELINES / baseline / "baseline.json")
            assert diagnostic["returncode"] == 0
            problems = check.recipe_problems(effective, selected)
            assert problems == [], (baseline, problems)
            if overlay:
                # Every historical return drops the E27c scheduler flags from the current base,
                # and every return before E27 also drops the cadence.
                against_current = check.recipe_problems(effective, current_expected)
                for key in check.SCHEDULER_FLAGS:
                    assert "baseline runtime environment: " + key in against_current, baseline
                assert ("baseline prefill schedule interval" in against_current) == (
                    baseline != "2026-09-24-e27"), baseline
                assert effective["extra_docker_env"] != effective["base_extra_docker_env"]
                if baseline == "2026-09-24-e27":
                    # E27c only adds the scheduler mount and flags: same engine arguments.
                    assert effective["extra_vllm_args"] == effective["base_extra_vllm_args"]
                elif baseline == "2026-09-23-e22b":
                    assert effective["extra_vllm_args"] != effective["base_extra_vllm_args"]
                elif baseline in ("2026-09-23-e21", "2026-09-19-e03"):
                    # Same KV and mHC as E22b; the cache namespace pin also differs.
                    assert effective["extra_vllm_args"] != effective["base_extra_vllm_args"]
                    assert "baseline payload pin: sparkcache_config_sha256" in check.recipe_problems(
                        effective, current_expected)
                elif baseline == "2026-09-19":
                    assert "baseline mHC prefill flag" in check.recipe_problems(effective, current_expected)
                else:
                    assert effective["extra_vllm_args"] != effective["base_extra_vllm_args"]
                    assert "baseline KV memory budget" in check.recipe_problems(effective, current_expected)
            changed_site = deepcopy(effective)
            changed_site["container"] += "-other"
            assert "TP4_ENV changed protected field: container" in check.recipe_problems(changed_site, selected)
finally:
    check.REPO = original_repo
    if saved_overlay is None:
        os.environ.pop("TP4_ENV", None)
    else:
        os.environ["TP4_ENV"] = saved_overlay

print("test-check-f0: PASS")

# The accepted features and payloads must fail closed when disabled or mismatched.
for key, value in current_expected["payload_pins"].items():
    wrong = deepcopy(current_recipe)
    wrong[key] = "0" * 64
    assert "baseline payload pin: " + key in check.recipe_problems(wrong, current_expected)
wrong = deepcopy(current_recipe)
wrong["spark_mhc_prefill_shard"] = "0"
assert "baseline mHC prefill flag" in check.recipe_problems(wrong, current_expected)
for key in ("SPARK_MHC_PREFILL_SHARD", "VLLM_ADAPTIVE_K_RESPECT_DRAFT_BUDGET"):
    wrong = deepcopy(current_ranks)
    wrong[2]["remote"]["container"]["environment"][key] = "0"
    assert any(key in error for error in check.evaluate(current_recipe, current_expected, wrong, endpoint()))
wrong = deepcopy(current_ranks)
wrong[0]["remote"]["container"]["runtime_receipts"].pop("scheduler_boot_signature")
assert "rank 0: draft-budget scheduler boot signature" in check.evaluate(current_recipe, current_expected, wrong, endpoint())
