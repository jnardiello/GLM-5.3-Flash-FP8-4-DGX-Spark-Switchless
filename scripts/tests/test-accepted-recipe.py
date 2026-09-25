#!/usr/bin/env python3
"""Verify accepted defaults against the measured overlay, rollback and saved receipts."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import tempfile

REPO = Path(__file__).resolve().parents[2]
REFERENCE = REPO / "docs/historical_benchmarks/baselines/2026-09-25-e29/baseline.json"
record = json.loads(REFERENCE.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
for item in record["provenance"].values():
    assert sha(REPO / item["path"]) == item["sha256"], item["path"]
for path, digest in record["system"]["source_files_sha256"].items():
    assert sha(REPO / path) == digest, path
for item in record["receipts"].values():
    extract = item["portable_extract"]
    assert sha(REPO / extract["path"]) == extract["sha256"]
    data = json.loads((REPO / extract["path"]).read_text())
    assert data["receipt_sha256"] == item["native_receipt_sha256"]
    assert data["request_count"] == data["completed_streams"] == data["visible_responses"] == 54
assert record["performance"]["included_run_count"] == 3
assert record["functional"]["measured_requests"]["count"] == 162
for row in record["performance"]["metrics"]:
    # A metric may exclude a run by recorded owner decision; the value and reason stay in the record.
    assert len(row["per_run"]) + len(row.get("excluded_per_run", [])) == 3, row["key"]
    assert all(item["reason"] for item in row.get("excluded_per_run", [])), row["key"]
    assert statistics.median(row["per_run"]) == row["median"], row["key"]

e03 = REPO / "scripts/node/experiments/e03"
previous = (REPO / "scripts/node/reference/baseline-20260919.env").read_text()
e03_rollback = (REPO / "scripts/node/reference/baseline-20260919-e03.env").read_text()
e21_rollback = (REPO / "scripts/node/reference/baseline-20260923-e21.env").read_text()
e22b_rollback = (REPO / "scripts/node/reference/baseline-20260924-e22b.env").read_text()
e27_rollback = (REPO / "scripts/node/reference/baseline-20260924-e27.env").read_text()
e27c_rollback = (REPO / "scripts/node/reference/baseline-20260925-e27c.env").read_text()
e28b_rollback = (REPO / "scripts/node/reference/baseline-20260925-e28b.env").read_text()
# The historical E03 measurement: previous base plus the three E03 deltas.
historical_e03 = "\n".join([previous, (e03 / "candidate.env").read_text(),
                            (e03 / "replay-views/delta.env").read_text(),
                            (e03 / "draft-budget/delta.env").read_text()])
# The historical E21 measurement: the complete E03 recipe plus the BF16-residue delta.
historical_e21 = "\n".join([e03_rollback, (e03 / "bf16-residue/delta.env").read_text()])
# The measured E22b load: the complete E21 recipe plus the drafter context-BF16 delta.
historical_e22b = "\n".join([e21_rollback, (e03 / "drafter-w8a16/delta-context-bf16.env").read_text()])
# The measured E27 load: the E22b default plus the native prefill cadence. Its overlay
# (sha256 recorded in the E27 promotion record) added exactly this one engine argument.
historical_e27 = "\n".join([e22b_rollback, 'EXTRA_VLLM_ARGS+=" --prefill-schedule-interval 8"'])
# The measured E27c load: the E27 default plus the queued-cadence overlay.
historical_e27c = "\n".join([e27_rollback, (e03 / "queued-cadence/delta.env").read_text()])
# The measured E28b load: the E28 recipe (E27c plus the draft-depth-7 overlay, promoted
# locally before the window) plus the 16 GiB KV overlay.
historical_e28b = "\n".join([e27c_rollback, (e03 / "draft-depth-7/delta.env").read_text(),
                             (e03 / "kv-16gib/delta.env").read_text()])
# The measured E29 candidate (load B): the complete E28b recipe plus the end-drain overlay.
historical_candidate = "\n".join([e28b_rollback, (e03 / "end-drain/delta-b.env").read_text()])
SCHED = "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py:ro"
E29_ADDED = [str(Path.home()) + "/tp4/experiments/e03/end-drain/scheduler.py:" + SCHED,
             "-v", str(Path.home()) + "/tp4/experiments/e03/end-drain/core.py:"
             "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py:ro",
             "-e", "VLLM_E29_END_DRAIN=1", "-e", "VLLM_E29_IDLE_COALESCE_MS=4", "-e", "VLLM_E29_TRACE=0"]
E29_REMOVED = [str(Path.home()) + "/tp4/experiments/e03/queued-cadence/scheduler.py:" + SCHED]
E28B_ADDED = ['{"method":"dflash","model":"/draft","num_speculative_tokens":7,'
              '"num_speculative_tokens_per_batch_size":[[1,1,7],[2,6,3]],"kv_cache_dtype":"fp8_e4m3"}',
              "-e", "VLLM_ADAPTIVE_K_HI=7", '--compilation-config={"max_cudagraph_capture_size":72}',
              "--kv-cache-memory-bytes=17179869184"]
E28B_REMOVED = ['{"method":"dflash","model":"/draft","num_speculative_tokens":5,'
                '"num_speculative_tokens_per_batch_size":[[1,1,5],[2,6,3]],"kv_cache_dtype":"fp8_e4m3"}',
                "--kv-cache-memory-bytes=16106127360"]
E27C_TOKENS = ["-v", str(Path.home()) + "/tp4/experiments/e03/queued-cadence/scheduler.py:"
               "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py:ro",
               "-e", "VLLM_E27B_SHORT_PREFILL_TOKENS=2048", "-e", "VLLM_E27C_CADENCE_WHEN_QUEUED=1"]
with tempfile.TemporaryDirectory(prefix="tp4-accepted-recipe-") as temp:
    root = Path(temp)
    shutil.copyfile(REPO / "scripts/launcher/launch-glm53-tp4.sh", root / "launch.sh")
    config = (REPO / "cluster.env.example").read_text() + '''
NODES="n0 n1 n2 n3"
MGMT_IPS="192.0.2.21 192.0.2.22 192.0.2.23 192.0.2.24"
MASTER_IP=192.0.2.21
RELAY_DEST=operator@192.0.2.23
'''
    (root / "cluster.env").write_text(config)
    (root / "measured.env").write_text(historical_candidate)
    (root / "rollback.env").write_text(previous)
    (root / "rollback-e03.env").write_text(e03_rollback)
    (root / "rollback-e21.env").write_text(e21_rollback)
    (root / "rollback-e22b.env").write_text(e22b_rollback)
    (root / "rollback-e27.env").write_text(e27_rollback)
    (root / "rollback-e27c.env").write_text(e27c_rollback)
    (root / "rollback-e28b.env").write_text(e28b_rollback)
    (root / "historical-e28b.env").write_text(historical_e28b)
    (root / "historical-e27c.env").write_text(historical_e27c)
    (root / "historical-e27.env").write_text(historical_e27)
    (root / "historical-e22b.env").write_text(historical_e22b)
    (root / "historical-e21.env").write_text(historical_e21)
    (root / "historical-e03.env").write_text(historical_e03)
    env = dict(os.environ, TP4_DRY_RUN="1")
    env.pop("TP4_ENV", None)
    forbidden = root / "forbidden.log"
    bindir = root / "bin"
    bindir.mkdir()
    for name in ("sudo", "docker", "ssh", "systemctl", "curl", "ip", "sysctl"):
        path = bindir / name
        path.write_text('#!/bin/sh\nprintf "%s\\n" "$0" >> "$TP4_FORBIDDEN"\nexit 97\n')
        path.chmod(0o700)
    env.update(PATH=str(bindir) + os.pathsep + env["PATH"], TP4_FORBIDDEN=str(forbidden))

    def launch(rank, overlay=None):
        selected = dict(env, TP4_ENV=overlay) if overlay else env
        result = subprocess.run(["bash", str(root / "launch.sh"), str(rank)],
                                env=selected, capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, result.stderr
        assert not forbidden.exists(), "Dry-run attempted an external action"
        return [line[2:] for line in result.stdout.splitlines() if line.startswith("  ")]

    for rank in range(4):
        current = launch(rank)
        assert current == launch(rank, "measured.env"), f"rank {rank}: changed measured command"
        mounts = dict(arg.split(":")[1::-1] for i, arg in enumerate(current) if i and current[i-1] == "-v")
        mount_count = Counter(arg.split(":")[1] for i, arg in enumerate(current) if i and current[i-1] == "-v")
        assert all(n == 1 for n in mount_count.values()), "Duplicate mount target"
        payloads = record["system"]["payload_packaging"]["operator_payloads"]
        private_targets = {item["container_path"] for item in payloads.values()}
        for target, digest in record["system"]["operational_identity"]["container_file_sha256"].items():
            assert target in mounts, target
            if target in private_targets:
                continue
            source = mounts[target]
            assert source.startswith(str(Path.home()) + "/tp4/")
            local = REPO / "scripts/node" / source.split("/tp4/", 1)[1]
            assert sha(local) == digest, local
        # Immediate rollback: exactly the measured E28b command, with the E27c scheduler.
        e28b_restored = launch(rank, "rollback-e28b.env")
        assert e28b_restored == launch(rank, "historical-e28b.env"), f"rank {rank}: E28b rollback drifted"
        assert Counter(current) - Counter(e28b_restored) == Counter(E29_ADDED), f"rank {rank}: E29 delta"
        assert Counter(e28b_restored) - Counter(current) == Counter(E29_REMOVED), f"rank {rank}: E29 removed"
        # Older E27c return: exactly the measured E27c command, five draft tokens and 15 GiB.
        e27c_restored = launch(rank, "rollback-e27c.env")
        assert e27c_restored == launch(rank, "historical-e27c.env"), f"rank {rank}: E27c rollback drifted"
        assert Counter(e28b_restored) - Counter(e27c_restored) == Counter(E28B_ADDED), f"rank {rank}: E28b delta"
        assert Counter(e27c_restored) - Counter(e28b_restored) == Counter(E28B_REMOVED), f"rank {rank}: E28b removed"
        # Older E27 return: exactly the measured E27 command, without the E27c scheduler.
        e27_restored = launch(rank, "rollback-e27.env")
        assert e27_restored == launch(rank, "historical-e27.env"), f"rank {rank}: E27 rollback drifted"
        assert not any("queued-cadence" in item or item.startswith("VLLM_E27") for item in e27_restored)
        assert Counter(e27c_restored) - Counter(e27_restored) == Counter(E27C_TOKENS), f"rank {rank}: E27c delta"
        assert Counter(e27_restored) - Counter(e27c_restored) == Counter(), f"rank {rank}: E27c removed items"
        # Older E22b return: exactly the measured E22b command, without the E27 cadence.
        e22b_restored = launch(rank, "rollback-e22b.env")
        assert e22b_restored == launch(rank, "historical-e22b.env"), f"rank {rank}: E22b rollback drifted"
        assert "--prefill-schedule-interval" not in e22b_restored
        assert e27_restored == e22b_restored + ["--prefill-schedule-interval", "8"], f"rank {rank}: E27 delta"
        # Older E21 return: exactly the measured E21 command, without any E22 element.
        e21_restored = launch(rank, "rollback-e21.env")
        assert e21_restored == launch(rank, "historical-e21.env"), f"rank {rank}: E21 rollback drifted"
        assert not any("drafter-w8a16" in item or "qwen3_dflash2" in item for item in e21_restored)
        assert not any(item.startswith("VLLM_E22_") for item in e21_restored)
        assert "VLLM_E21_BF16_RESIDUE_W8A16=1" in e21_restored
        assert "VLLM_E22_DRAFTER_W8A16=1" in current and "VLLM_E22_CONTEXT_KV_W8A16=0" in current
        # Older E03 return: exactly the measured E03 command, without any E21 element.
        e03_restored = launch(rank, "rollback-e03.env")
        assert e03_restored == launch(rank, "historical-e03.env"), f"rank {rank}: E03 rollback drifted"
        assert not any("e21_bf16_residue" in item for item in e03_restored)
        assert "VLLM_E21_BF16_RESIDUE_W8A16=1" not in e03_restored
        assert any(item.endswith("/tp4/overrides/vllm/models/glm5next/nvidia/e20_kda_w8a16.py:"
                                 "/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/"
                                 "e20_kda_w8a16.py:ro") for item in e03_restored)
        assert "VLLM_E21_BF16_RESIDUE_W8A16=1" in current
        # Older pre-E03 base remains a complete return as well.
        restored = launch(rank, "rollback.env")
        assert "SPARK_MHC_PREFILL_SHARD=0" in restored
        assert "VLLM_ADAPTIVE_K_RESPECT_DRAFT_BUDGET=1" not in restored
        assert not any("/experiments/e03/" in item for item in restored)
        assert not any("connector-e03-replay-views" in item for item in restored)
        assert "--kv-cache-memory-bytes=16106127360" in restored

print("test-accepted-recipe: PASS (four-rank command parity, mounted hashes, E27c, E27, E22b, E21, E03 and pre-E03 rollbacks, 3-run provenance)")
