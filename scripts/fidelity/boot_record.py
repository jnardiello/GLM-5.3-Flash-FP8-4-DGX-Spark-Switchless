#!/usr/bin/env python3
"""Record the identity of one fidelity-campaign boot (read-only on the nodes).

For the currently serving stack it records the effective overlay and its sha256, the
container image ID, command hash and start time per rank, MODEL_REV/DRAFT_REV, the KV dtype,
the logprob cap, and the READY/signature lines from every rank's container log. Output goes to
a private JSON file; `--public` also writes a copy without hosts, addresses or paths.

    TP4_ENV=scripts/node/experiments/fidelity/cm.env \
      python3 scripts/fidelity/boot_record.py --label cm-1 --out data/fidelity/boots/cm-1.json
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SIGNATURES = re.compile(
    r"(_READY\b|READY |AdaptiveKScheduler active|draft-budget active|GPU KV cache size|"
    r"num_spec_tokens=|cudagraph_capture_sizes|Using TRITON Fp8 MoE backend|Using configuration from|"
    r"Using .*MoE backend|SPARK_MHC_PREFILL|E20_MEMORY_PROBE|max_logprobs|kv_cache_dtype|"
    r"Starting vLLM API server|Traceback|Error|ERROR)")


def env_value(key, overlay):
    cmd = f". ./cluster.env; {'. ' + shlex.quote(overlay) + ';' if overlay else ''} printf '%s' \"${{{key}}}\""
    return subprocess.run(["bash", "-c", cmd], cwd=REPO, capture_output=True, text=True, check=True).stdout


def ssh(host, cmd, timeout=60):
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, cmd],
                       capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--public", help="also write a sanitised copy here")
    ap.add_argument("--k", type=int, default=20, help="logprob K used by the collectors")
    a = ap.parse_args()
    overlay = os.environ.get("TP4_ENV", "")
    hosts = (os.environ.get("TP4_HOSTS") or env_value("NODES", overlay)).split()
    container = env_value("CONTAINER", overlay)
    rec = {
        "label": a.label,
        "recorded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "overlay": overlay or None,
        "overlay_sha256": hashlib.sha256((REPO / overlay).read_bytes()).hexdigest() if overlay else None,
        "cluster_env_sha256": hashlib.sha256((REPO / "cluster.env").read_bytes()).hexdigest(),
        "image": env_value("IMAGE", overlay),
        "image_id_pinned": env_value("IMAGE_ID", overlay) or None,
        "model_repo": env_value("MODEL_REPO", overlay),
        "model_rev": env_value("MODEL_REV", overlay),
        "draft_rev": env_value("DRAFT_REV", overlay),
        "kv_cache_dtype": env_value("KV_CACHE_DTYPE", overlay),
        "spec_tokens": env_value("SPEC_TOKENS", overlay),
        "extra_vllm_args": env_value("EXTRA_VLLM_ARGS", overlay),
        "k": a.k,
        "ranks": [],
    }
    for rank, host in enumerate(hosts):
        fmt = "{{.Image}}|{{.State.StartedAt}}|{{json .Config.Cmd}}"
        rc, out = ssh(host, f"sudo -n docker inspect --format '{fmt}' {shlex.quote(container)}")
        image_id, started, cmd_json = (out.strip().split("|", 2) + ["", "", ""])[:3] if rc == 0 else ("", "", "")
        rc2, logs = ssh(host, f"sudo -n docker logs {shlex.quote(container)} 2>&1 | tail -n 20000", timeout=120)
        lines = [l.strip()[:400] for l in logs.splitlines() if SIGNATURES.search(l)]
        rec["ranks"].append({
            "rank": rank, "host": host, "container_running": rc == 0, "image_id": image_id,
            "started": started, "cmd_sha256": hashlib.sha256(cmd_json.encode()).hexdigest() if cmd_json else None,
            "cmd": json.loads(cmd_json) if cmd_json.startswith("[") else None,
            "signature_lines": lines[-200:],
        })
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    os.chmod(a.out, 0o600)
    if a.public:
        pub = {k: v for k, v in rec.items() if k not in ("ranks",)}
        pub["ranks"] = [{"rank": r["rank"], "container_running": r["container_running"],
                         "image_id": r["image_id"], "cmd_sha256": r["cmd_sha256"],
                         "ready_lines": sorted({re.sub(r"^.*?(\b[A-Z0-9_]+_READY\b.*)$", r"\1", l)
                                                for l in r["signature_lines"] if "_READY" in l})}
                        for r in rec["ranks"]]
        Path(a.public).parent.mkdir(parents=True, exist_ok=True)
        Path(a.public).write_text(json.dumps(pub, indent=2) + "\n", encoding="utf-8")
    ok = all(r["container_running"] for r in rec["ranks"])
    print(f"boot {a.label}: {'4/4 running' if ok else 'NOT all ranks running'}; "
          f"image ids {sorted({r['image_id'][:19] for r in rec['ranks']})}; "
          f"cmd hashes {len({r['cmd_sha256'] for r in rec['ranks']})} distinct")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
