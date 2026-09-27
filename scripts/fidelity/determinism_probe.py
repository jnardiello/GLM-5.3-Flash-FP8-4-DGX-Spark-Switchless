#!/usr/bin/env python3
"""Repeatability probe for teacher-forced prompt logprobs (numbers only, no text).

Sends a fixed sequence of corpus windows (default X X X Y X Y Y) to /v1/completions with
prompt_logprobs and a fresh cache salt per request, then compares every pair of passes of
the same window: positions with bit-identical actual-token logprobs, first position whose
|delta| exceeds a threshold, max |delta| and top-1 agreement. It separates run-to-run
nondeterminism (repeats of X differ even back to back) from a dependence on the previous
request (back-to-back repeats agree but repeats after Y differ).

    data/fidelity/.venv/bin/python scripts/fidelity/determinism_probe.py \
        --base-url http://HOST:8000 --x w0001 --y w0002 --out probe.json
"""
import argparse
import itertools
import json
import secrets
import sys
import urllib.request
from array import array
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "data/fidelity/corpus"


def load(window_id):
    man = {w["id"]: w for w in json.loads((CORPUS / "manifest.json").read_text())["windows"]}
    a = array("I")
    a.frombytes((CORPUS / man[window_id]["path"]).read_bytes())
    return list(a)


def score(base, model, ids, k):
    body = {"model": model, "prompt": ids, "max_tokens": 1, "temperature": 0,
            "prompt_logprobs": k, "cache_salt": secrets.token_hex(32)}
    req = urllib.request.Request(base.rstrip("/") + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        pl = json.loads(r.read())["choices"][0]["prompt_logprobs"]
    lp = np.full(len(ids), np.nan)
    top1 = np.full(len(ids), -1)
    for i in range(1, len(ids)):
        d = pl[i]
        lp[i] = d[str(ids[i])]["logprob"]
        top1[i] = int(min(d.items(), key=lambda kv: kv[1].get("rank", 10 ** 9))[0])
    return lp, top1


def compare(a, b, thr):
    la, lb = a[0][1:], b[0][1:]
    d = np.abs(la - lb)
    over = np.where(d > thr)[0]
    same = d == 0
    first_diff = int(np.argmax(~same)) + 1 if (~same).any() else None
    return {"positions": int(len(d)), "identical": int(same.sum()), "first_nonidentical_pos": first_diff,
            "first_over_threshold_pos": int(over[0]) + 1 if len(over) else None,
            "over_threshold": int(len(over)), "max_abs_delta": float(d.max()),
            "top1_agreement": float((a[1][1:] == b[1][1:]).mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--x", default="w0001")
    ap.add_argument("--y", default="w0002")
    ap.add_argument("--sequence", default="XXXYXYY")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--threshold", type=float, default=1.0)
    ap.add_argument("--out")
    a = ap.parse_args()
    with urllib.request.urlopen(a.base_url.rstrip("/") + "/v1/models", timeout=30) as r:
        model = json.loads(r.read())["data"][0]["id"]
    windows = {"X": load(a.x), "Y": load(a.y)}
    passes = []
    for step, name in enumerate(a.sequence):
        passes.append((step, name, score(a.base_url, model, windows[name], a.k)))
        print(f"pass {step} {name} done", flush=True)
    report = {"x": a.x, "y": a.y, "sequence": a.sequence, "k": a.k, "threshold": a.threshold, "pairs": []}
    for (i, n1, r1), (j, n2, r2) in itertools.combinations(passes, 2):
        if n1 == n2:
            prev = lambda s: a.sequence[s - 1] if s > 0 else "-"
            report["pairs"].append({"window": n1, "passes": [i, j], "preceded_by": [prev(i), prev(j)],
                                    **compare(r1, r2, a.threshold)})
    for p in report["pairs"]:
        print(f"{p['window']} passes {p['passes']} after {p['preceded_by']}: identical {p['identical']}/{p['positions']}"
              f" first_nonidentical {p['first_nonidentical_pos']} max|d| {p['max_abs_delta']:.3f}"
              f" top1 {p['top1_agreement']:.3f}")
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
