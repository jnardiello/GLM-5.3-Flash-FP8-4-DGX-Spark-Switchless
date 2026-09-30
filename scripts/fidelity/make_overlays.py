#!/usr/bin/env python3
"""Generate the fidelity campaign's complete TP4_ENV overlays from the frozen recipes.

Every overlay is a complete non-site recipe: the assignment lines of its base (a frozen
reference overlay, or the complete E29 recipe that was the cluster.env.example default) copied
verbatim, with only the declared deltas applied. Each overlay gets its own SparkCache
configuration whose only difference from the base configuration is `spark_cache_root`, so
no measurement ever reads or writes a production cache namespace.

    python3 scripts/fidelity/make_overlays.py            # (re)write overlays + configs
    python3 scripts/fidelity/make_overlays.py --check    # fail if the tree is stale
    python3 scripts/fidelity/make_overlays.py --diff ARM # base -> overlay diff of one arm
"""
import argparse
import difflib
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "scripts/node/experiments/fidelity"
REF = REPO / "scripts/node/reference"
NODE_ROOT = "$HOME/tp4/experiments/fidelity"

# Keys of a complete non-site overlay, in the order the frozen references use.
KEYS = [
    "NCCL_IB_GID_INDEX", "NCCL_IB_GID_INDEX_BY_RANK", "IMAGE", "IMAGE_ID", "LAUNCHER",
    "SPARKCACHE_MODE", "SPARKCACHE_CONFIG", "SPARKCACHE_CONFIG_SHA256", "SPARKCACHE_CONNECTOR",
    "SPARKCACHE_CONNECTOR_SHA256", "SPARKCACHE_ENCODER", "SPARKCACHE_ENCODER_SHA256",
    "SIRCL_DIR", "MODEL_DIR", "MODEL_REPO", "MODEL_REV", "DRAFT_DIR", "DRAFT_REV",
    "SERVED_NAME", "MAX_MODEL_LEN", "MAX_NUM_SEQS", "KV_CACHE_DTYPE", "BATCHED_TOKENS",
    "SPARK_MHC_PREFILL_SHARD", "BLOCK_SIZE", "GPU_MEM_UTIL", "SPEC_TOKENS", "SPEC_EXTRA_JSON",
    "ASYNC_SCHEDULING", "EXTRA_DOCKER_ENV", "EXTRA_VLLM_ARGS",
]
BASES = {
    # The campaign ran on the E29 default; since E31 its complete recipe is the E29 return.
    "e29": REF / "baseline-20260925-e29.env",
    "e28b": REF / "baseline-20260925-e28b.env",
    "e22b": REF / "baseline-20260924-e22b.env",
    "e21": REF / "baseline-20260923-e21.env",
    "e03": REF / "baseline-20260919-e03.env",
    "0919": REF / "baseline-20260919.env",
    "0918": REF / "baseline-20260918.env",
}
# Current recipes built with substitutions and appends cannot be parsed line by line: they
# are evaluated in bash (the template, then the listed deltas, then declared word swaps).
E36_DELTA = "scripts/node/experiments/e03/e36-lm-head-w8a16/delta.env"
E35_RETURN = "scripts/node/reference/operational-20260930-e35.env"
E36_FLAG = "/tmp/glm53-e36-lm-head"
# Since E36 is the default, the E35 recipe is the template through its one-step return.
EVALUATED = {
    "e35": ([E35_RETURN], []),
    # E36 in measurement mode keeps the BF16 head and follows a flag file (bf16 = Cnow).
    "e36fid": ([E35_RETURN, E36_DELTA],
               [(" -e VLLM_E36_KEEP_BF16=0", f" -e VLLM_E36_KEEP_BF16=1 -e VLLM_E36_FLAG={E36_FLAG}")]),
}
KV_MEASURE = 6442450944          # 6 GiB per rank (owner range 4-6 GiB)
MAX_LOGPROBS = 100               # K=20 default, K=100 sensitivity subset
MAX_LEN_MEASURE = 139264         # >= the longest window (131,000 tokens) + 1

# arm -> (base, measurement mode, extra key overrides, description)
ARMS = {
    "r0fp8-m": ("0918", True, {},
                "R0 reference and ladder rung 1: September 18 recipe (BF16 KDA projections, no hybrid KDA, "
                "no mHC, no E21/E22b) with its FP8 KV; B12X admits no BF16 KV (PLAN amendment 6)"),
    "l0919-m": ("0919", True, {}, "ladder rung 2: September 19 base (hybrid KDA)"),
    "cpre-m": ("e03", True, {}, "Cpre: E03 recipe (hybrid KDA + E03 mHC prefill sharding), my recipe before E21"),
    "le21-m": ("e21", True, {}, "ladder rung 4: E21 residual projections"),
    "le22b-m": ("e22b", True, {}, "ladder rung 5: E22b drafter conversion (negative control for target logits)"),
    "cm": ("e29", True, {}, "Cm: current E29 default in measurement mode (ladder rung 6)"),
    "r0-s": ("0918", False, {"_KV_BYTES": 12884901888},
             "R0 serving mode for tasks/voxel: September 18 recipe, production scheduling, 12 GiB KV "
             "for rank-0 memory (PLAN amendment 8)"),
    "cpre-s": ("e03", False, {}, "Cpre serving mode for tasks/voxel"),
    "cp-s": ("e29", False, {}, "Cp: E29 exactly, only the SparkCache namespace differs"),
    "le36-m": ("e36fid", True, {},
               "E36 in measurement mode: the E35 default plus the INT8 shared lm_head, BF16 kept and "
               f"selected per call by {E36_FLAG} (bf16 scores Cnow, int8 scores LE36, on one load)"),
}

# Site-template defaults (model and drafter paths) and the historical header label of the
# E29 base, kept so the measured overlays stay byte-identical to their boot records.
EXAMPLE = REPO / "cluster.env.example"
# The le36-m label names the base as measured on 2026-09-30, when the template was the E35
# default; it is kept so the measured overlay stays byte-identical to its boot record.
LABELS = {"e29": "cluster.env.example", "e35": f"cluster.env.example + {E35_RETURN} (evaluated)",
          "e36fid": f"cluster.env.example + {E36_DELTA} (evaluated)"}
ASSIGN = re.compile(r"^([A-Z][A-Z0-9_]*)=(.*)$")


def parse(path):
    """Ordered KEY -> raw right-hand side for single-line top-level assignments."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = ASSIGN.match(line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def unquote(raw):
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "'\"":
        return raw[1:-1]
    return raw


def shell_value(key, value):
    """Right-hand side that sources back to `value`; EXTRA_VLLM_ARGS keeps the escaped style."""
    if key == "EXTRA_VLLM_ARGS" or "'" in value:
        escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$").replace("`", "\\`")
        return f'"{escaped}"'
    return f"'{value}'"


def evaluated_lines(base):
    deltas, swaps = EVALUATED[base]
    script = ['set +u', 'source "$1" >/dev/null 2>&1']
    script += [f'source "{REPO / delta}" || exit 3' for delta in deltas]
    script += ['for k in "${@:2}"; do if declare -p "$k" 2>/dev/null | grep -q "^declare -a"; then '
               'eval "n=\\${#$k[@]}"; [ "$n" = 0 ] || exit 4; printf "%s\\0()\\0" "$k"; '
               'else printf "%s\\0%s\\0" "$k" "${!k-}"; fi; done']
    result = subprocess.run(["bash", "-c", "; ".join(script), "overlay", str(EXAMPLE), *KEYS],
                            capture_output=True, check=False)
    if result.returncode:
        raise SystemExit(f"evaluating base {base} failed: {result.stderr.decode()[-400:]}")
    fields = result.stdout.decode().split("\0")[:-1]
    raw = dict(zip(fields[::2], fields[1::2]))
    for old, new in swaps:
        if raw["EXTRA_DOCKER_ENV"].count(old) != 1:
            raise SystemExit(f"base {base}: expected one {old.strip()!r}")
        raw["EXTRA_DOCKER_ENV"] = raw["EXTRA_DOCKER_ENV"].replace(old, new)
    return {k: (v if v == "()" else shell_value(k, v)) for k, v in raw.items()}


def base_lines(base):
    if base in EVALUATED:
        vals = evaluated_lines(base)
        missing = [k for k in KEYS if k not in vals]
        if missing:
            raise SystemExit(f"base {base}: missing keys {missing}")
        return {k: vals[k] for k in KEYS}
    vals = parse(BASES[base])
    if base == "e29":
        vals = {k: v for k, v in vals.items() if k in KEYS}
    # Complete identity: the September 19+ references inherit model/drafter paths from cluster.env.
    example = parse(EXAMPLE)
    for key in ("MODEL_DIR", "DRAFT_DIR", "MODEL_REPO", "MODEL_REV", "DRAFT_REV", "SIRCL_DIR",
                "SPARKCACHE_ENCODER", "SPARKCACHE_ENCODER_SHA256"):
        vals.setdefault(key, example[key] if key in example else "''")
    missing = [k for k in KEYS if k not in vals]
    if missing:
        raise SystemExit(f"{BASES[base]}: missing keys {missing}")
    return {k: vals[k] for k in KEYS}


def sparkcache_config(base_vals, arm, measure=False):
    src = unquote(base_vals["SPARKCACHE_CONFIG"]).replace("$HOME/tp4/", "")
    local = {
        "reference/sparkcache-20260918.json": REF / "sparkcache-20260918.json",
    }.get(src, REPO / "scripts/node" / src)
    cfg = json.loads(local.read_text(encoding="utf-8"))
    cfg["kv_connector_extra_config"]["spark_cache_root"] = f"/cache/jit/sparkcache-fidelity-{arm}"
    if measure:
        # Amendment 11 (docs/fidelity/REPORT.md): a full scored run would store ~28 GB per rank. The connector stays
        # loaded (same scheduling path); with fresh salts no replay is possible either way.
        cfg["kv_connector_extra_config"]["spark_cache_store"] = False
        cfg["kv_connector_extra_config"]["spark_cache_restore"] = False
    data = (json.dumps(cfg, indent=2, sort_keys=True) + "\n").encode()
    return src, data


def vllm_args(raw, measure):
    args = unquote(raw)
    if not measure:
        return raw
    args, n = re.subn(r"--kv-cache-memory-bytes=\d+", f"--kv-cache-memory-bytes={KV_MEASURE}", args)
    if n != 1:
        raise SystemExit("expected exactly one --kv-cache-memory-bytes")
    return '"' + args + f" --max-logprobs {MAX_LOGPROBS}" + '"'


def build(arm, nospec=False):
    base, measure, extra, desc = ARMS[arm]
    extra = dict(extra)
    kv_bytes = extra.pop("_KV_BYTES", None)
    vals = base_lines(base)
    base_vals = dict(vals)
    name = arm + ("-nospec" if nospec else "")
    src_cfg, cfg_bytes = sparkcache_config(vals, name, measure)
    vals["SPARKCACHE_CONFIG"] = f"'{NODE_ROOT}/sparkcache/{name}.json'"
    vals["SPARKCACHE_CONFIG_SHA256"] = hashlib.sha256(cfg_bytes).hexdigest()
    if measure:
        vals["MAX_NUM_SEQS"] = "1"
        vals["MAX_MODEL_LEN"] = str(MAX_LEN_MEASURE)
        vals["EXTRA_VLLM_ARGS"] = vllm_args(vals["EXTRA_VLLM_ARGS"], True)
    if kv_bytes:
        args, n = re.subn(r"--kv-cache-memory-bytes=\d+", f"--kv-cache-memory-bytes={kv_bytes}",
                          unquote(vals["EXTRA_VLLM_ARGS"]))
        if n != 1:
            raise SystemExit("expected exactly one --kv-cache-memory-bytes")
        vals["EXTRA_VLLM_ARGS"] = '"' + args + '"'
    if nospec:
        vals["SPEC_TOKENS"] = "0"
        vals["SPEC_EXTRA_JSON"] = "''"
    vals.update(extra)
    deltas = [k for k in KEYS if vals[k] != base_vals[k]]
    header = [
        f"# Fidelity campaign overlay `{name}` (generated by scripts/fidelity/make_overlays.py; do not edit).",
        f"# {desc}{' — speculative decoding OFF' if nospec else ''}.",
        f"# Complete non-site recipe: base {LABELS[base] if base in LABELS else BASES[base].relative_to(REPO)} (non-site keys), deltas: {', '.join(deltas)}.",
        f"# SparkCache: {src_cfg} with spark_cache_root changed to a fidelity namespace"
        + (" and store/restore off (measurement)." if measure else "."),
        "# Use the same TP4_ENV for deploy, up, status and down in its window. See docs/fidelity/PLAN.md.",
    ]
    text = "\n".join(header + [f"{k}={vals[k]}" for k in KEYS]) + "\n"
    return name, text, cfg_bytes, base_vals, vals


def all_outputs():
    files = {}
    for arm, (base, measure, _, _) in ARMS.items():
        variants = [False, True] if measure else [False]
        for nospec in variants:
            name, text, cfg, _, _ = build(arm, nospec)
            files[OUT / f"{name}.env"] = text.encode()
            files[OUT / "sparkcache" / f"{name}.json"] = cfg
    sums = "".join(
        f"{hashlib.sha256(data).hexdigest()}  sparkcache/{path.name}\n"
        for path, data in sorted(files.items()) if path.parent.name == "sparkcache")
    files[OUT / "sparkcache" / "SHA256SUMS"] = sums.encode()
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--diff", metavar="ARM")
    a = ap.parse_args()
    if a.diff:
        arm, nospec = (a.diff[:-7], True) if a.diff.endswith("-nospec") else (a.diff, False)
        _, _, _, base_vals, vals = build(arm, nospec)
        before = [f"{k}={base_vals[k]}" for k in KEYS]
        after = [f"{k}={vals[k]}" for k in KEYS]
        sys.stdout.writelines(l + "\n" for l in difflib.unified_diff(
            before, after, f"base:{ARMS[arm][0]}", a.diff, n=0, lineterm=""))
        return 0
    files = all_outputs()
    stale = [p for p, d in files.items() if not p.exists() or p.read_bytes() != d]
    if a.check:
        for p in stale:
            print(f"stale: {p.relative_to(REPO)}", file=sys.stderr)
        return 1 if stale else 0
    for p, d in files.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(d)
    print(f"wrote {len(files)} files under {OUT.relative_to(REPO)} ({len(stale)} changed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
