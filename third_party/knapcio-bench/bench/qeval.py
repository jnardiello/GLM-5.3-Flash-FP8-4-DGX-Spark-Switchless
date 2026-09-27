#!/usr/bin/env python3
"""Scored quality gate for DS-V4.1-Flash config changes.

    python3 qeval.py run before --url http://127.0.0.1:8888/v1/chat/completions
    ... change config, reboot ...
    python3 qeval.py run after  --url ...
    python3 qeval.py compare qeval-before.json qeval-after.json

55 auto-scored tasks (code executed against hidden asserts, JSON schema-checked, numeric answers
matched, format constraints enforced, prose checked only for objective degeneration). No LLM judge,
so the score is reproducible.

Run at concurrency 1. Greedy output is only reproducible run to run when the batch composition is
fixed (Mia's README, "Correctness notes"); at concurrency > 1 a pass->fail flip can be batching
noise rather than the config change. `--concurrency N` exists for speed but forfeits that.

The comparison is PAIRED: the same tasks before and after, so what matters is the flip matrix
(pass->fail vs fail->pass), not the two pass rates. McNemar's exact test gives the p-value.
"""
import argparse, concurrent.futures as cf, json, math, statistics as st, sys, time, urllib.request
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import qeval_tasks as qt


def ask(url, task, timeout):
    body = {"model": "GLM-5.3-Flash-FP8", "temperature": 0, "max_tokens": task["max_tokens"],
            "stream": False, "chat_template_kwargs": {"reasoning_effort": "high" if task["thinking"] else "low"},
            # Per-request DSpark acceptance telemetry. Costs no GPU time: the counters are
            # incremented unconditionally in batch_result_processor.py:773-777 and this flag
            # only decides whether they are serialised back onto the response.
            "return_spec_tokens_details": True,
            "messages": [{"role": "user", "content": task["prompt"]}]}
    t0 = time.time()
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=timeout))
    dt = time.time() - t0
    ch = r["choices"][0]
    msg = ch["message"]
    content = msg.get("content") or ""
    n = r["usage"]["completion_tokens"]
    return {"content": content,
            "reasoning_chars": len(msg.get("reasoning_content") or msg.get("reasoning") or ""),
            "finish_reason": ch.get("finish_reason"), "completion_tokens": n,
            "seconds": round(dt, 3), "tok_s": round(n / dt, 2) if dt > 0 else None,
            "spec": (r.get("sglext") or {}).get("spec_tokens_details")}


def score_one(url, task, timeout):
    out = {"id": task["id"], "category": task["category"]}
    try:
        out.update(ask(url, task, timeout))
    except Exception as exc:
        return {**out, "pass": False, "why": f"request failed: {exc!r}"}
    try:
        ok, why = task["checker"](out["content"])
    except Exception as exc:
        ok, why = False, f"checker raised {exc!r}"
    out["pass"] = bool(ok)
    out["why"] = "" if ok else str(why)[:160]
    return out


def run(label, url, timeout, concurrency, only, limit=0):
    tasks = [t for t in qt.TASKS if not only or t["category"] in only]
    if limit: tasks = tasks[:limit]
    print(f"{len(tasks)} tasks, concurrency {concurrency}, {url}")
    t0 = time.time()
    if concurrency > 1:
        with cf.ThreadPoolExecutor(concurrency) as ex:
            results = list(ex.map(lambda t: score_one(url, t, timeout), tasks))
    else:
        results = []
        for t in tasks:
            r = score_one(url, t, timeout)
            results.append(r)
            print(f"  {'PASS' if r['pass'] else 'FAIL'}  {r['id']:<22} {r.get('why','')[:70]}", flush=True)
    wall = time.time() - t0
    speeds = [r["tok_s"] for r in results if r.get("tok_s")]
    trunc = sum(1 for r in results if r.get("finish_reason") == "length")
    out = {"label": label, "url": url, "concurrency": concurrency, "wall_s": round(wall, 1),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "median_tok_s": round(st.median(speeds), 2) if speeds else None,
           "truncated": trunc, "results": results}
    npass = sum(1 for r in results if r["pass"])
    print(f"\n{npass}/{len(results)} passed ({100*npass/len(results):.1f}%), "
          f"{trunc} hit the token cap, wall {wall/60:.1f} min, "
          f"median {out['median_tok_s']} tok/s")
    by = {}
    for r in results:
        by.setdefault(r["category"], []).append(r["pass"])
    for c, v in sorted(by.items()):
        print(f"   {c:<8} {sum(v):>2}/{len(v)}")
    json.dump(out, open(f"qeval-{label}.json", "w"), indent=1)
    print(f"wrote qeval-{label}.json")


def mcnemar_p(b, c):
    """Two-sided exact McNemar: b = pass->fail, c = fail->pass, H0: p = 0.5."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


PRIMARY   = ("code", "reason", "math")      # what "quality" means here, per the owner
SECONDARY = ("json", "format")             # instruction following: reported, not decisive
GUARD     = ("prose",)                     # objective degeneration only, never a quality score


def _block(ids, ra, rb, cats):
    kept = broke = fixed = bf = 0
    flips = []
    for i in ids:
        if ra[i]["category"] not in cats:
            continue
        pa_, pb_ = ra[i]["pass"], rb[i]["pass"]
        if pa_ and pb_: kept += 1
        elif pa_ and not pb_: broke += 1; flips.append(("BROKE", i, rb[i].get("why", "")))
        elif pb_: fixed += 1; flips.append(("fixed", i, ra[i].get("why", "")))
        else: bf += 1
    return kept, broke, fixed, bf, flips


def _line(tag, kept, broke, fixed, bf):
    n = kept + broke + fixed + bf
    if n == 0:
        return None
    before, after = kept + broke, kept + fixed
    rel = (after - before) / before * 100 if before else 0.0
    p = mcnemar_p(broke, fixed)
    print(f"{tag:<28} {before:>3}/{n} -> {after:>3}/{n}   "
          f"kept {kept:>3}  BROKE {broke:>3}  fixed {fixed:>3}  both-fail {bf:>3}   "
          f"{rel:+6.1f}%   p={p:.3f}")
    return rel, p, broke, fixed


def compare(pa, pb):
    a, b = json.load(open(pa)), json.load(open(pb))
    ra = {r["id"]: r for r in a["results"]}
    rb = {r["id"]: r for r in b["results"]}
    ids = [t["id"] for t in qt.TASKS if t["id"] in ra and t["id"] in rb]
    print(f"paired on {len(ids)} tasks: {a['label']} -> {b['label']}\n")
    print(f"{'':<28} {'before':>7} {'after':>7}")
    pk, pbk, pf, pbf, pflips = _block(ids, ra, rb, PRIMARY)
    prim = _line("PRIMARY code+reason+math", pk, pbk, pf, pbf)
    sk, sbk, sf, sbf, sflips = _block(ids, ra, rb, SECONDARY)
    _line("secondary json+format", sk, sbk, sf, sbf)
    gk, gbk, gf, gbf, gflips = _block(ids, ra, rb, GUARD)
    _line("guard prose (degeneration)", gk, gbk, gf, gbf)

    percat = {}
    for i in ids:
        c = ra[i]["category"]
        d = percat.setdefault(c, [0, 0, 0, 0])
        pa_, pb_ = ra[i]["pass"], rb[i]["pass"]
        d[0 if (pa_ and pb_) else 1 if pa_ else 2 if pb_ else 3] += 1
    print(f"\n{'category':<9}{'kept':>6}{'broke':>7}{'fixed':>7}{'both-fail':>11}")
    for c, d in sorted(percat.items()):
        print(f"{c:<9}{d[0]:>6}{d[1]:>7}{d[2]:>7}{d[3]:>11}")

    allflips = pflips + sflips + gflips
    if allflips:
        print("\nflips:")
        for kind, i, why in allflips:
            print(f"  {kind:<6} {i:<22} {why[:80]}")

    sa, sb = a.get("median_tok_s"), b.get("median_tok_s")
    rel, p, broke, fixed = prim
    print()
    if gbk:
        print(f"GUARD TRIPPED: {gbk} prose task(s) degenerated -- inspect before accepting.")
    if sa and sb:
        spd = 100 * (sb / sa - 1)
        print(f"median single-stream {sa} -> {sb} tok/s ({spd:+.1f}%)")
        if rel < -0.5 and spd > 0:
            print(f"TRADE: {abs(rel):.1f}% of code+reasoning for {spd:+.1f}% speed"
                  f"{'' if p <= 0.05 else '  -- but the quality drop is within noise at this sample size'}")
        elif spd > 0 and rel >= -0.5:
            print(f"FREE: {spd:+.1f}% speed, primary quality unchanged (p={p:.3f})")
    n_prim = pk + pbk + pf + pbf
    print(f"\nresolution: {n_prim} primary tasks. A net loss of 5 with no gains is p={mcnemar_p(5,0):.3f}; "
          f"below ~4 net breaks this harness cannot separate the change from noise.")
    if a.get("concurrency", 1) > 1 or b.get("concurrency", 1) > 1:
        print("NOTE: a run used concurrency > 1; some flips may be batching noise, not the config.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    s = ap.add_subparsers(dest="cmd", required=True)
    r = s.add_parser("run"); r.add_argument("label")
    r.add_argument("--url", default="http://127.0.0.1:8093/v1/chat/completions")
    r.add_argument("--timeout", type=float, default=900)
    r.add_argument("--concurrency", type=int, default=1)
    r.add_argument("--only", default="", help="comma-separated categories")
    r.add_argument("--limit", type=int, default=0, help="first N tasks (smoke test)")
    c = s.add_parser("compare"); c.add_argument("a"); c.add_argument("b")
    n = ap.parse_args()
    if n.cmd == "run":
        run(n.label, n.url, n.timeout, n.concurrency, [x for x in n.only.split(",") if x], n.limit)
    else:
        sys.exit(compare(n.a, n.b))
