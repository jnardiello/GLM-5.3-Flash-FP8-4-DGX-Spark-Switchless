#!/usr/bin/env python3
"""Offline E03 source/mount contract; no Torch, nodes or inference."""
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import runpy
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "scripts/node/experiments/e03"
manifest = json.loads((BASE / "manifest.json").read_text())
for name, pins in manifest["files"].items():
    assert hashlib.sha256((BASE / "overrides" / name).read_bytes()).hexdigest() == pins["candidate_sha256"], name
for row in (BASE / "SHA256SUMS").read_text().splitlines():
    digest, name = row.split("  ", 1)
    assert hashlib.sha256((BASE / name).read_bytes()).hexdigest() == digest, name

model_path = "vllm/models/glm5next/nvidia/model.py"
base_model = REPO / "scripts/node/overrides" / model_path
assert hashlib.sha256(base_model.read_bytes()).hexdigest() == manifest["files"][model_path]["base_sha256"]

def weight_code(path):
    tree = ast.parse(path.read_text())
    return [ast.dump(n) for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
            and ("load" in n.name or "finalize" in n.name or "remap" in n.name)]

assert weight_code(base_model) == weight_code(BASE / "overrides" / model_path)
cache = json.loads((BASE / "kv-transfer-config.json").read_text())
original = json.loads((REPO / "scripts/node/sparkcache/kv-transfer-config.json").read_text())
assert cache["kv_connector_extra_config"].pop("spark_cache_root") != original["kv_connector_extra_config"].pop("spark_cache_root")
assert cache == original

with tempfile.TemporaryDirectory(prefix="e03-config-") as temp:
    root = Path(temp)
    shutil.copyfile(REPO / "scripts/launcher/launch-glm53-tp4.sh", root / "launch.sh")
    shutil.copyfile(BASE / "candidate.env", root / "candidate.env")
    config = ((REPO / "cluster.env.example").read_text() + "\n"
              + (REPO / "scripts/node/reference/baseline-20260919.env").read_text())
    config += '\nNODES="n0 n1 n2 n3"\nMGMT_IPS="192.0.2.21 192.0.2.22 192.0.2.23 192.0.2.24"\nMASTER_IP=192.0.2.21\nRELAY_DEST=operator@192.0.2.23\n'
    (root / "cluster.env").write_text(config)
    env = dict(os.environ, TP4_DRY_RUN="1")
    env.pop("TP4_ENV", None)
    for rank in range(4):
        before = subprocess.check_output(["bash", str(root / "launch.sh"), str(rank)], env=env, text=True)
        after = subprocess.check_output(["bash", str(root / "launch.sh"), str(rank)],
                                       env=dict(env, TP4_ENV="candidate.env"), text=True)
        argv = [row[2:] for row in after.splitlines() if row.startswith("  ")]
        assert "SPARK_MHC_PREFILL_SHARD=0" in before and "SPARK_MHC_PREFILL_SHARD=1" in argv
        assert "--kv-cache-memory-bytes=16106127360" in argv
        assert "--max-model-len" in argv and "262144" in argv
        for rel in manifest["files"]:
            mounts = [a for a in argv if f":/usr/local/lib/python3.12/dist-packages/{rel}:" in a]
            assert len(mounts) == 1 and "/experiments/e03/" in mounts[0], (rel, mounts)
        for preserved in ("e20_kda_w8a16.py", "e20_hybrid_scratch.py", "gpu_worker.py", "adaptive_k_scheduler.py"):
            assert any(preserved in arg for arg in argv), preserved
        calls = []
        def read_command(command, **kwargs):
            if command[0] == "bash":
                return after
            assert command[0] in ("sudo", "nvidia-smi")
            return ""
        with (patch("sys.argv", ["gpu-launch.py", str(rank)]),
              patch("subprocess.check_output", side_effect=read_command),
              patch("subprocess.run", side_effect=lambda command, **kwargs: calls.append(command))):
            runpy.run_path(str(BASE / "gpu-launch.py"), run_name="__main__")
        command = calls[0]
        assert command[-1] == "/opt/e03/gpu-check.py"
        assert command[command.index("--entrypoint") + 1] == "python3"
        assert command[command.index("--name") + 1] == f"e03-gpu-check-rank{rank}"
        assert "--rm" in command and "-d" not in command
print("test-e03-config: PASS")
