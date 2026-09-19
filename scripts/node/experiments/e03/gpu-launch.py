#!/usr/bin/env python3
"""Run the model-free E03 GPU check using the native launcher's Docker recipe.

Start ranks 0..3 concurrently after a coordinated down. This never stops services.
"""
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[2]
rank = int(sys.argv[1])
assert 0 <= rank < 4
env = dict(os.environ, TP4_DRY_RUN="1",
           TP4_ENV="scripts/node/experiments/e03/candidate.env")
preview = subprocess.check_output(["bash", str(root / "launch-glm53-tp4.sh"), str(rank)],
                                  env=env, text=True)
argv = [row[2:] for row in preview.splitlines() if row.startswith("  ")]
assert argv[:4] == ["sudo", "docker", "run", "-d"]
image_index = next(i for i, value in enumerate(argv) if value.startswith("ghcr.io/"))
options, image = argv[:image_index], argv[image_index]
container = options[options.index("--name") + 1]
active = subprocess.check_output(["sudo", "-n", "docker", "ps", "--filter",
                                 f"name=^/{container}$", "--format", "{{.Names}}"], text=True)
assert not active.strip(), "Stop the four-rank service before this GPU check"
gpu = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
                               "--format=csv,noheader"], text=True)
assert not gpu.strip(), "A GPU compute process is still active; do not create a second context"
master = argv[argv.index("--master-addr") + 1] if "--master-addr" in argv else None
if master is None:
    # The serving CLI uses --nnodes/--node-rank and --master-addr in this image;
    # fail explicitly if the launcher interface ever changes.
    raise RuntimeError("Native launch preview has no --master-addr")
options.remove("-d")
options += ["--rm"]
options[options.index("--name") + 1] = f"e03-gpu-check-rank{rank}"
i = options.index("--restart")
del options[i:i + 2]
options[options.index("--entrypoint") + 1] = "python3"
options += ["-v", str(root / "experiments/e03") + ":/opt/e03:ro",
            "-e", f"RANK={rank}", "-e", f"MASTER_ADDR={master}",
            "-e", "MASTER_PORT=29603", "-e", "OMP_NUM_THREADS=1"]
subprocess.run([*options, image, "/opt/e03/gpu-check.py"], check=True)
