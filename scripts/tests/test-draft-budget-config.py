#!/usr/bin/env python3
"""Manifest and four-rank command equivalence for E03 + replay + draft budget."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

REPO = Path(__file__).resolve().parents[2]
E03 = REPO / "scripts/node/experiments/e03"
CANDIDATE = E03 / "draft-budget"
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
manifest = json.loads((CANDIDATE / "manifest.json").read_text())
assert sha(REPO / manifest["base_scheduler"]) == manifest["base_scheduler_sha256"]
assert sha(CANDIDATE / "adaptive_k_scheduler.py") == manifest["candidate_scheduler_sha256"]
assert sha(E03 / "SHA256SUMS") == manifest["preserved"]["e03_source_manifest_sha256"]
for path, digest in manifest["overlay_parts_sha256"].items():
    assert sha(REPO / path) == digest, path
for line in (CANDIDATE / "SHA256SUMS").read_text().splitlines():
    digest, name = line.split("  ", 1)
    assert sha(CANDIDATE / name) == digest, name

parent = (E03 / "candidate.env").read_text() + "\n" + (E03 / "replay-views/delta.env").read_text()
delta = (CANDIDATE / "delta.env").read_text()
with tempfile.TemporaryDirectory(prefix="tp4-draft-budget-") as temp:
    root = Path(temp)
    shutil.copyfile(REPO / "scripts/launcher/launch-glm53-tp4.sh", root / "launch.sh")
    config = ((REPO / "cluster.env.example").read_text() + "\n"
              + (REPO / "scripts/node/reference/baseline-20260919.env").read_text()) + '''
NODES="n0 n1 n2 n3"
MGMT_IPS="192.0.2.21 192.0.2.22 192.0.2.23 192.0.2.24"
MASTER_IP=192.0.2.21
RELAY_DEST=operator@192.0.2.23
'''
    (root / "cluster.env").write_text(config)
    (root / "parent.env").write_text(parent)
    (root / "candidate.env").write_text(parent + "\n" + delta)
    env = dict(os.environ, TP4_DRY_RUN="1")
    env.pop("TP4_ENV", None)
    forbidden = root / "forbidden.log"
    bindir = root / "bin"
    bindir.mkdir()
    for name in ("sudo", "docker", "ssh", "systemctl", "curl", "ip", "sysctl"):
        p = bindir / name
        p.write_text('#!/bin/sh\nprintf "%s\\n" "$0" >> "$TP4_FORBIDDEN"\nexit 97\n')
        p.chmod(0o700)
    env.update(PATH=str(bindir) + os.pathsep + env["PATH"], TP4_FORBIDDEN=str(forbidden))

    def launch(overlay, rank=0):
        result = subprocess.run(["bash", str(root / "launch.sh"), str(rank)],
                                env=dict(env, TP4_ENV=overlay), capture_output=True,
                                text=True, timeout=10)
        assert not forbidden.exists(), "Dry-run attempted an external action"
        return result

    def argv(result):
        assert result.returncode == 0, result.stderr
        return [line[2:] for line in result.stdout.splitlines() if line.startswith("  ")]

    for rank in range(4):
        before, after = argv(launch("parent.env", rank)), argv(launch("candidate.env", rank))
        old_mount = str(Path.home()) + "/patches/adaptive_k_scheduler.py:/opt/tp4/adaptive_k_scheduler.py:ro"
        new_mount = str(Path.home()) + "/tp4/experiments/e03/draft-budget/adaptive_k_scheduler.py:/opt/tp4/adaptive_k_scheduler.py:ro"
        assert Counter(before) - Counter(after) == Counter([old_mount])
        assert Counter(after) - Counter(before) == Counter([new_mount, "-e", "VLLM_ADAPTIVE_K_RESPECT_DRAFT_BUDGET=1"])
        # The rest of the native command is identical, including ordering.
        restored = [old_mount if x == new_mount else x for x in after]
        flag_index = restored.index("VLLM_ADAPTIVE_K_RESPECT_DRAFT_BUDGET=1")
        assert restored[flag_index - 1] == "-e"
        del restored[flag_index - 1:flag_index + 1]
        assert restored == before
        mounts = [after[i + 1] for i, item in enumerate(after[:-1]) if item == "-v"]
        targets = [item.split(":")[1] for item in mounts]
        assert len(targets) == len(set(targets)), "Duplicate container mount"
        assert after[after.index("--node-rank") + 1] == str(rank)
        assert "--kv-cache-memory-bytes=16106127360" in after
        assert after[after.index("--max-model-len") + 1] == "262144"
        spec = json.loads(after[after.index("--speculative-config") + 1])
        assert spec["num_speculative_tokens"] == 5
        assert spec["num_speculative_tokens_per_batch_size"] == [[1, 1, 5], [2, 6, 3]]
        assert "SPARK_MHC_PREFILL_SHARD=1" in after
        assert "VLLM_ADAPTIVE_K_MODE=batch-uniform" in after
        assert after[after.index("--scheduler-cls") + 1] == "adaptive_k_scheduler.AdaptiveKScheduler"
        assert any("spark_context_cache_connector-e03-replay-views.py:" in v for v in mounts)

    for name, bad in {
        "no-parent": delta,
        "e03-only": (E03 / "candidate.env").read_text() + delta,
        "duplicate": parent + "\nEXTRA_DOCKER_ENV+=' -v $HOME/patches/adaptive_k_scheduler.py:/opt/tp4/adaptive_k_scheduler.py:ro'\n" + delta,
        "existing-flag": parent + "\nEXTRA_DOCKER_ENV+=' -e VLLM_ADAPTIVE_K_RESPECT_DRAFT_BUDGET=0'\n" + delta,
    }.items():
        (root / "bad.env").write_text(bad)
        assert launch("bad.env").returncode != 0, name

    # Confirm the existing additive deploy selector includes this nested payload.
    selected = subprocess.check_output([
        "find", "scripts/node/experiments/e03", "-type", "f", "(",
        "-name", "*.py", "-o", "-name", "*.json", "-o", "-name", "SHA256SUMS", ")"
    ], cwd=REPO, text=True).splitlines()
    for name in ("adaptive_k_scheduler.py", "manifest.json", "SHA256SUMS"):
        assert str((CANDIDATE / name).relative_to(REPO)) in selected

print("test-draft-budget-config: PASS (four ranks, parent rollback, pins, additive deploy selection)")
