#!/usr/bin/env python3
"""Write the pre-registered, seeded window/prompt subsets (docs/fidelity/REPORT.md).

Stratified by category, deterministic for a given manifest: every category contributes
round(fraction x its window count), at least one window. Outputs one id per line under
data/fidelity/corpus/subsets/ and prints counts only.

    python3 scripts/fidelity/make_subsets.py
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "data/fidelity/corpus"
SEED = "20260927"
WINDOW_SUBSETS = {"k100": 0.20, "crossboot": 0.20, "ladder": 0.30}
DECODE_SUBSETS = {"nospec": 50, "replay": 20, "floor": 30}


def stratified(windows, fraction, name):
    by_cat = defaultdict(list)
    for w in windows:
        by_cat[w["category"]].append(w["id"])
    rng = random.Random(f"{SEED}:{name}")
    out = []
    for cat in sorted(by_cat):
        ids = sorted(by_cat[cat])
        out += rng.sample(ids, max(1, round(fraction * len(ids))))
    return sorted(out)


def main():
    manifest = json.loads((CORPUS / "manifest.json").read_text())
    decode = json.loads((CORPUS / "decode_manifest.json").read_text())
    outdir = CORPUS / "subsets"
    outdir.mkdir(exist_ok=True)
    report = {"manifest_global_sha256": manifest["global_sha256"]}
    for name, frac in WINDOW_SUBSETS.items():
        ids = stratified(manifest["windows"], frac, name)
        (outdir / f"{name}.txt").write_text("\n".join(ids) + "\n")
        report[name] = len(ids)
    prompts = sorted(p["id"] for p in decode["prompts"])
    for name, n in DECODE_SUBSETS.items():
        ids = sorted(random.Random(f"{SEED}:decode:{name}").sample(prompts, n))
        (outdir / f"decode-{name}.txt").write_text("\n".join(ids) + "\n")
        report[f"decode-{name}"] = len(ids)
    (outdir / "subsets.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
