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
REFERENCE = REPO / "docs/historical_benchmarks/baselines/2026-09-19-e03/baseline.json"
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
historical_candidate = "\n".join([previous, (e03 / "candidate.env").read_text(),
                                 (e03 / "replay-views/delta.env").read_text(),
                                 (e03 / "draft-budget/delta.env").read_text()])
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
        restored = launch(rank, "rollback.env")
        assert "SPARK_MHC_PREFILL_SHARD=0" in restored
        assert "VLLM_ADAPTIVE_K_RESPECT_DRAFT_BUDGET=1" not in restored
        assert not any("/experiments/e03/" in item for item in restored)
        assert not any("connector-e03-replay-views" in item for item in restored)
        assert "--kv-cache-memory-bytes=16106127360" in restored

print("test-accepted-recipe: PASS (four-rank command parity, mounted hashes, complete rollback, 3-run provenance)")
