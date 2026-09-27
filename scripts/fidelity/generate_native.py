#!/usr/bin/env python3
"""Generate the corpus's model-native windows on the reference arm (docs/fidelity/PLAN.md §4-5).

Samples one continuation per selected decode prompt (session prompts only, never public ones)
through `/v1/completions` with token-id input: temperature 1.0, top_p 0.95, a fixed seed per
prompt, `max_tokens` 4096 and a fresh cache salt, at bounded concurrency. Each result is saved
in the harness generation layout (`<id>.npz` with `gen_ids`, plus a `<id>.json` sidecar) so
`scripts/fidelity/corpus/build_corpus.py add-native --gen-dir DIR` can append the windows.
Prints ids and counts only.

    data/fidelity/.venv/bin/python scripts/fidelity/generate_native.py \
        --base-url http://HOST:8000 --out data/fidelity/raw/R0/native --count 60
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import secrets
import sys
import time
import urllib.error
import urllib.request
from array import array
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "data/fidelity/corpus"


def load_ids(path):
    a = array("I")
    a.frombytes(path.read_bytes())
    if sys.byteorder != "little":
        a.byteswap()
    return list(a)


def request(base, body, timeout, retries=4):
    data = json.dumps(body).encode()
    for attempt in range(retries + 1):
        req = urllib.request.Request(base.rstrip("/") + "/v1/completions", data=data,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return 200, json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code < 500 or attempt == retries:
                return e.code, None
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == retries:
                return 0, None
        time.sleep(2 ** attempt)
        body["cache_salt"] = secrets.token_hex(32)
        data = json.dumps(body).encode()
    return 0, None


def one(args, model, prompt):
    npz, side = args.out / f"{prompt['id']}.npz", args.out / f"{prompt['id']}.json"
    if args.abort_file and args.abort_file.exists():
        return prompt["id"], "aborted", 0
    if side.exists() and json.loads(side.read_text()).get("status") == "ok" and npz.exists():
        return prompt["id"], "skip", 0
    ids = load_ids(CORPUS / prompt["path"])
    seed = int(hashlib.sha256(f"native:{prompt['id']}".encode()).hexdigest()[:8], 16)
    body = {"model": model, "prompt": ids, "max_tokens": args.max_tokens, "temperature": 1.0,
            "top_p": 0.95, "seed": seed, "logprobs": 1, "return_tokens_as_token_ids": True,
            "cache_salt": secrets.token_hex(32)}
    t0 = time.time()
    status, resp = request(args.base_url, body, args.timeout)
    meta = {"status": "ok" if status == 200 else f"http_{status}", "seed": seed,
            "prompt_tokens": len(ids), "elapsed_s": round(time.time() - t0, 2),
            "temperature": 1.0, "top_p": 0.95, "max_tokens": args.max_tokens}
    if status == 200:
        ch = resp["choices"][0]
        toks = (ch.get("logprobs") or {}).get("tokens") or []
        gen = [int(str(t).split(":", 1)[1]) for t in toks]
        if not gen or any(not str(t).startswith("token_id:") for t in toks):
            meta["status"] = "parse_error"
        else:
            tmp = npz.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, gen_ids=np.asarray(gen, dtype=np.int32))
            os.replace(tmp, npz)
            meta.update(gen_tokens=len(gen), finish_reason=ch.get("finish_reason"), usage=resp.get("usage"))
    side.write_text(json.dumps(meta, indent=1) + "\n")
    return prompt["id"], meta["status"], meta.get("gen_tokens", 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--count", type=int, default=60)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--seed", default="20260927")
    ap.add_argument("--source", choices=("session", "public", "native"), default="session",
                    help="prompts to continue: session/public decode prompts, or the long-form native prompts")
    ap.add_argument("--abort-file", type=Path, help="stop issuing requests once this file exists")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.source == "native":
        pool = json.loads((CORPUS / "native_prompts_manifest.json").read_text())["prompts"]
    else:
        manifest = json.loads((CORPUS / "decode_manifest.json").read_text())
        pool = sorted((p for p in manifest["prompts"]
                       if (p["source"] == "public") == (a.source == "public")), key=lambda p: p["id"])
    chosen = sorted(random.Random(f"{a.seed}:native:{a.source}").sample(pool, min(a.count, len(pool))),
                    key=lambda p: p["id"])
    (a.out / f"selection-{a.source}.json").write_text(json.dumps([p["id"] for p in chosen]) + "\n")
    with urllib.request.urlopen(a.base_url.rstrip("/") + "/v1/models", timeout=30) as r:
        model = json.loads(r.read())["data"][0]["id"]
    done = 0
    with concurrent.futures.ThreadPoolExecutor(a.concurrency) as pool:
        for pid, status, n in pool.map(lambda p: one(a, model, p), chosen):
            done += 1
            print(f"[{done}/{len(chosen)}] {pid} {status} gen_tokens={n}", flush=True)
    if a.abort_file and a.abort_file.exists():
        print("native: aborted by the memory sampler; completed results are kept")
        return 3
    bad = [json.loads((a.out / f"{p['id']}.json").read_text())["status"] for p in chosen]
    bad = [s for s in bad if s != "ok"]
    print(f"native: {len(chosen) - len(bad)} ok, {len(bad)} not ok")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
