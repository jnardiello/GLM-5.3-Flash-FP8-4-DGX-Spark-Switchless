#!/usr/bin/env python3
"""Fidelity smoke test: does the served engine return usable prompt and generation logprobs?

Three checks against an OpenAI-compatible vLLM endpoint, printing numbers only:

1. prompt_logprobs=K on /v1/completions with token-id input (every position after the first
   must carry the actual token);
2. logprobs=K on a greedy generation (with speculative decoding active, if the recipe has it);
3. the same prompt twice with the same cache_salt: are the second request's prompt logprobs
   complete and equal to the first (i.e. not suppressed or altered by a prefix/SparkCache hit)?

Use --tokens to choose the prompt length. Keep it below 4,096 on a production recipe: shorter
spans are never stored by SparkCache, so no production cache namespace is written.

    python3 scripts/fidelity/smoke.py --base-url http://HOST:8000 --tokens 3000 --out smoke.json
"""
import argparse
import json
import math
import secrets
import sys
import time
import urllib.request


def post(base, path, body, timeout=900):
    req = urllib.request.Request(base.rstrip("/") + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read()), time.time() - t0
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode(errors="replace")[:400]}, time.time() - t0


def model_id(base):
    with urllib.request.urlopen(base.rstrip("/") + "/v1/models", timeout=30) as r:
        return json.loads(r.read())["data"][0]["id"]


def synthetic_prompt(n, seed=20260927):
    # Deterministic ordinary token ids (no special tokens: ids 1000..60000).
    x, out = seed, []
    for _ in range(n):
        x = (1103515245 * x + 12345) % (2 ** 31)
        out.append(1000 + x % 59000)
    return out


def prompt_lp_summary(resp, ids, k):
    pl = resp["choices"][0].get("prompt_logprobs")
    if pl is None:
        return {"present": False}
    missing = sum(1 for i in range(1, len(ids))
                  if pl[i] is None or str(ids[i]) not in {str(t) for t in pl[i]})
    widths = [len(pl[i]) for i in range(1, len(pl)) if pl[i]]
    actual = [pl[i][str(ids[i])]["logprob"] for i in range(1, len(ids))
              if pl[i] and str(ids[i]) in pl[i]]
    return {"present": True, "len": len(pl), "first_is_null": pl[0] is None,
            "missing_actual": missing, "min_entries": min(widths) if widths else 0,
            "max_entries": max(widths) if widths else 0, "k": k,
            "mean_actual_lp": sum(actual) / len(actual) if actual else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--tokens", type=int, default=3000)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--gen-tokens", type=int, default=64)
    ap.add_argument("--out")
    a = ap.parse_args()
    model = model_id(a.base_url)
    ids = synthetic_prompt(a.tokens)
    report = {"model": model, "prompt_tokens": len(ids), "k": a.k}

    salt = secrets.token_hex(32)
    body = {"model": model, "prompt": ids, "max_tokens": 1, "temperature": 0,
            "prompt_logprobs": a.k, "cache_salt": salt}
    st, r1, dt = post(a.base_url, "/v1/completions", body)
    report["prompt_logprobs"] = {"status": st, "seconds": round(dt, 3),
                                 **(prompt_lp_summary(r1, ids, a.k) if st == 200 else {"error": r1.get("error")})}
    if st == 200:
        report["prompt_logprobs"]["usage"] = r1.get("usage")

    st, r2, dt = post(a.base_url, "/v1/completions", body)          # same salt: potential hit
    same = None
    if st == 200 and report["prompt_logprobs"].get("present"):
        p1, p2 = r1["choices"][0]["prompt_logprobs"], r2["choices"][0].get("prompt_logprobs")
        if p2 is not None:
            diffs = [abs(p1[i][str(ids[i])]["logprob"] - p2[i][str(ids[i])]["logprob"])
                     for i in range(1, len(ids))
                     if p1[i] and p2[i] and str(ids[i]) in p1[i] and str(ids[i]) in p2[i]]
            same = {"max_abs_diff_actual_lp": max(diffs) if diffs else None, "compared": len(diffs)}
    report["repeat_same_salt"] = {"status": st, "seconds": round(dt, 3),
                                  **(prompt_lp_summary(r2, ids, a.k) if st == 200 else {"error": r2.get("error")}),
                                  "usage": r2.get("usage") if st == 200 else None, "vs_first": same}

    gbody = {"model": model, "prompt": ids[:512], "max_tokens": a.gen_tokens, "temperature": 0,
             "logprobs": a.k, "return_tokens_as_token_ids": True, "cache_salt": secrets.token_hex(32)}
    st, g, dt = post(a.base_url, "/v1/completions", gbody)
    gen = {"status": st, "seconds": round(dt, 3)}
    if st == 200:
        lp = g["choices"][0].get("logprobs") or {}
        toks, tlp, top = lp.get("tokens") or [], lp.get("token_logprobs") or [], lp.get("top_logprobs") or []
        gen.update({"tokens": len(toks), "token_logprobs": len(tlp),
                    "top_min": min((len(t) for t in top if t), default=0),
                    "ids_form": bool(toks) and all(str(t).startswith("token_id:") for t in toks),
                    "nan": sum(1 for v in tlp if v is None or (isinstance(v, float) and math.isnan(v))),
                    "finish_reason": g["choices"][0].get("finish_reason"), "usage": g.get("usage")})
    else:
        gen["error"] = g.get("error")
    report["generation_logprobs"] = gen
    text = json.dumps(report, indent=2)
    print(text)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    ok = (report["prompt_logprobs"].get("missing_actual") == 0
          and report["generation_logprobs"].get("token_logprobs", 0) > 0)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
