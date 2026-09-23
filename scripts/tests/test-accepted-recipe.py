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
REFERENCE = REPO / "docs/historical_benchmarks/baselines/2026-09-23-e22b/baseline.json"
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
    assert len(row["per_run"]) == 3 and statistics.median(row["per_run"]) == row["median"]

e03 = REPO / "scripts/node/experiments/e03"
previous = (REPO / "scripts/node/reference/baseline-20260919.env").read_text()
e03_rollback = (REPO / "scripts/node/reference/baseline-20260919-e03.env").read_text()
e21_rollback = (REPO / "scripts/node/reference/baseline-20260923-e21.env").read_text()
# The historical E03 measurement: previous base plus the three E03 deltas.
historical_e03 = "\n".join([previous, (e03 / "candidate.env").read_text(),
                            (e03 / "replay-views/delta.env").read_text(),
                            (e03 / "draft-budget/delta.env").read_text()])
# The historical E21 measurement: the complete E03 recipe plus the BF16-residue delta.
historical_e21 = "\n".join([e03_rollback, (e03 / "bf16-residue/delta.env").read_text()])
# The measured E22b candidate: the complete E21 recipe plus the drafter context-BF16 delta.
historical_candidate = "\n".join([e21_rollback, (e03 / "drafter-w8a16/delta-context-bf16.env").read_text()])
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
        # Immediate rollback: exactly the measured E21 command, without any E22 element.
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

print("test-accepted-recipe: PASS (four-rank command parity, mounted hashes, E21, E03 and pre-E03 rollbacks, 3-run provenance)")
