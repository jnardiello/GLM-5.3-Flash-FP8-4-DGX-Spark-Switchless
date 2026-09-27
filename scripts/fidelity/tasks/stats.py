#!/usr/bin/env python3
"""Analyse run_tasks.py output: paired significance tests for qeval, and
descriptive comparisons for hardset. Stdlib only.

Reads   data/fidelity/tasks/<arm>/<set>/<mode>/run<k>/<item>.json
Writes  docs/fidelity/metrics/tasks-<set>.json   (aggregate JSON, no response text)

qeval:
  - exact two-sided McNemar test on paired greedy outcomes between two arms
    (binomial tail on discordant pairs; run 1 of each arm's greedy results).
  - discordance table (both-pass, both-fail, arm-a-only, arm-b-only).
  - paired bootstrap 95% CI (resample items with replacement, B=2000,
    seed 20260927) for the difference of per-item pass rates, where an
    item's pass rate is averaged over its sampled-mode runs.
  - Wilson 95% CI per arm on the pooled greedy pass rate.
  - minimum detectable effect (MDE) on the paired pass-rate difference, from
    the observed discordance and for the all-discordant worst case (see mde_approx).
  - z.ai repeat agreement: of the 3 greedy repeats, the fraction of items
    where all 3 agree on pass/fail and on the extracted final answer text.

hardset:
  - descriptive only: finish-reason counts, answer/reasoning length stats,
    truncation counts, and exact final-answer-text agreement with a
    reference arm (only for items an answer can be extracted from).
"""
import argparse
import json
import math
import pathlib
import random
import statistics as st
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
DATA_ROOT = REPO_ROOT / "data" / "fidelity" / "tasks"
METRICS_DIR = REPO_ROOT / "docs" / "fidelity" / "metrics"

BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 20260927


# ------------------------------------------------------------------- loading

def load_run(arm, set_name, mode, run_idx):
    """item_id -> parsed result dict, for results that parsed and ran ok."""
    d = DATA_ROOT / arm / set_name / mode / f"run{run_idx}"
    out = {}
    if not d.is_dir():
        return out
    for p in sorted(d.glob("*.json")):
        try:
            rec = json.loads(p.read_text())
        except Exception:
            continue
        out[rec.get("item", p.stem)] = rec
    return out


def count_runs(arm, set_name, mode):
    d = DATA_ROOT / arm / set_name / mode
    if not d.is_dir():
        return 0
    n = 0
    while (d / f"run{n + 1}").is_dir():
        n += 1
    return n


# --------------------------------------------------------------- statistics

def mcnemar_exact_p(b, c):
    """Two-sided exact McNemar test: b = a-pass/b-fail, c = a-fail/b-pass,
    H0: P(b) = P(c) = 0.5 among discordant pairs (binomial tail, stdlib)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def wilson_ci(successes, n, z=1.959963984540054):
    """Wilson score 95% CI for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    lo = (centre - half) / denom
    hi = (centre + half) / denom
    return (p, max(0.0, lo), min(1.0, hi))


def mde_approx(n_items, discordant=None, alpha=0.05, power=0.80):
    """Approximate minimum detectable effect on the paired pass-rate difference.

    Normal approximation to McNemar's test: the paired difference (b - c) / n has
    variance about d / n under the null, where d is the discordant fraction, so
    MDE = (z_alpha/2 + z_beta) * sqrt(d / n). d is the observed discordant fraction
    when `discordant` is given, otherwise 1 (every item discordant, the worst case).
    A planning heuristic, not a substitute for the exact McNemar p-value.
    """
    if n_items <= 0:
        return None
    z_alpha = 1.959963984540054  # two-sided alpha=0.05
    z_beta = {0.80: 0.8416212335729143, 0.90: 1.2815515655446004}.get(round(power, 2), 0.8416212335729143)
    d = 1.0 if discordant is None else discordant / n_items
    if d <= 0:
        return None
    return round((z_alpha + z_beta) * math.sqrt(d / n_items), 4)


def bootstrap_diff_ci(item_pass_rates_a, item_pass_rates_b, item_ids, B=BOOTSTRAP_B, seed=BOOTSTRAP_SEED):
    """Paired bootstrap 95% CI for mean(pass_rate_b - pass_rate_a), resampling
    items with replacement. item_pass_rates_* map item_id -> mean pass rate
    over that item's sampled-mode runs."""
    rng = random.Random(seed)
    ids = [i for i in item_ids if i in item_pass_rates_a and i in item_pass_rates_b]
    if not ids:
        return {"n_items": 0, "diff": None, "ci95": [None, None]}
    diffs = [item_pass_rates_b[i] - item_pass_rates_a[i] for i in ids]
    point = st.fmean(diffs)
    boots = []
    n = len(ids)
    for _ in range(B):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        boots.append(st.fmean(sample))
    boots.sort()
    lo = boots[int(0.025 * B)]
    hi = boots[min(B - 1, int(0.975 * B))]
    return {"n_items": n, "diff": round(point, 4), "ci95": [round(lo, 4), round(hi, 4)], "B": B, "seed": seed}


# ------------------------------------------------------------------- qeval

def qeval_pass_map(arm, mode, run_idx):
    recs = load_run(arm, "qeval", mode, run_idx)
    return {i: r.get("grader", {}).get("pass") for i, r in recs.items() if r.get("ok")}


def qeval_item_pass_rate(arm, mode, n_runs):
    """item_id -> mean pass rate over 1..n_runs (only over runs that ran ok)."""
    per_item = {}
    for run_idx in range(1, n_runs + 1):
        pm = qeval_pass_map(arm, mode, run_idx)
        for item_id, passed in pm.items():
            per_item.setdefault(item_id, []).append(1.0 if passed else 0.0)
    return {i: st.fmean(v) for i, v in per_item.items() if v}


def qeval_analysis(arm_a, arm_b):
    out = {}
    pa = qeval_pass_map(arm_a, "greedy", 1)
    pb = qeval_pass_map(arm_b, "greedy", 1)
    ids = sorted(set(pa) & set(pb))
    both_pass = both_fail = a_only = b_only = 0
    for i in ids:
        if pa[i] and pb[i]:
            both_pass += 1
        elif not pa[i] and not pb[i]:
            both_fail += 1
        elif pa[i] and not pb[i]:
            a_only += 1
        else:
            b_only += 1
    out["greedy_discordance"] = {
        "arm_a": arm_a, "arm_b": arm_b, "n_paired": len(ids),
        "both_pass": both_pass, "both_fail": both_fail,
        f"{arm_a}_only_pass": a_only, f"{arm_b}_only_pass": b_only,
    }
    out["mcnemar_exact_p"] = mcnemar_exact_p(a_only, b_only)
    out["wilson_ci"] = {
        arm_a: dict(zip(("rate", "lo95", "hi95"), wilson_ci(sum(pa.values()), len(pa)))) if pa else None,
        arm_b: dict(zip(("rate", "lo95", "hi95"), wilson_ci(sum(pb.values()), len(pb)))) if pb else None,
    }
    out["mde_approx"] = {"n_items": len(ids), "discordant_pairs": a_only + b_only,
                         "abs_pass_rate_delta": mde_approx(len(ids), a_only + b_only),
                         "abs_pass_rate_delta_all_discordant": mde_approx(len(ids)),
                         "method": "normal approximation to McNemar, (z_a/2 + z_b) * sqrt(d / n), "
                                   "alpha 0.05 two-sided, power 0.80"}

    n_runs_a = count_runs(arm_a, "qeval", "sampled")
    n_runs_b = count_runs(arm_b, "qeval", "sampled")
    if n_runs_a and n_runs_b:
        ra = qeval_item_pass_rate(arm_a, "sampled", n_runs_a)
        rb = qeval_item_pass_rate(arm_b, "sampled", n_runs_b)
        out["sampled_bootstrap_diff"] = bootstrap_diff_ci(ra, rb, sorted(set(ra) & set(rb)))
    else:
        out["sampled_bootstrap_diff"] = {"note": "sampled-mode results missing for one or both arms"}

    for arm in (arm_a, arm_b):
        n_rep = count_runs(arm, "qeval", "greedy")
        if n_rep >= 3:
            out[f"{arm}_repeat_agreement"] = repeat_agreement(arm, n_rep)
    return out


def repeat_agreement(arm, n_runs):
    runs = [load_run(arm, "qeval", "greedy", k) for k in range(1, n_runs + 1)]
    all_ids = set()
    for r in runs:
        all_ids |= set(r)
    n_pass_agree = n_answer_agree = n_checked = 0
    for item_id in sorted(all_ids):
        recs = [r[item_id] for r in runs if item_id in r and r[item_id].get("ok")]
        if len(recs) < n_runs:
            continue
        n_checked += 1
        passes = [rec.get("grader", {}).get("pass") for rec in recs]
        answers = [rec["response"]["content"].strip() for rec in recs]
        if len(set(passes)) == 1:
            n_pass_agree += 1
        if len(set(answers)) == 1:
            n_answer_agree += 1
    if n_checked == 0:
        return {"n_items": 0, "pass_agreement": None, "answer_agreement": None}
    return {
        "n_items": n_checked,
        "n_repeats": n_runs,
        "pass_agreement": round(n_pass_agree / n_checked, 4),
        "answer_agreement": round(n_answer_agree / n_checked, 4),
    }


# ----------------------------------------------------------------- hardset

def hardset_descriptive(arms, reference_arm=None):
    out = {"arms": {}}
    per_arm_answers = {}
    for arm in arms:
        n_runs = count_runs(arm, "hardset", "greedy") or count_runs(arm, "hardset", "sampled")
        mode = "greedy" if count_runs(arm, "hardset", "greedy") else "sampled"
        recs_by_run = [load_run(arm, "hardset", mode, k) for k in range(1, (n_runs or 0) + 1)]
        finish = {}
        ans_lens, reason_lens, trunc = [], [], 0
        first_run_answers = {}
        total = 0
        for k, recs in enumerate(recs_by_run, start=1):
            for item_id, rec in recs.items():
                if not rec.get("ok"):
                    continue
                total += 1
                fr = rec["response"].get("finish_reason")
                finish[fr] = finish.get(fr, 0) + 1
                if fr == "length":
                    trunc += 1
                ans_lens.append(len(rec["response"]["content"]))
                reason_lens.append(len(rec["response"].get("reasoning_content") or ""))
                if k == 1:
                    first_run_answers[item_id] = rec["response"]["content"].strip()
        per_arm_answers[arm] = first_run_answers
        out["arms"][arm] = {
            "n_results": total,
            "finish_reasons": finish,
            "truncated": trunc,
            "answer_len_chars": {"mean": round(st.fmean(ans_lens), 1) if ans_lens else None,
                                  "median": st.median(ans_lens) if ans_lens else None},
            "reasoning_len_chars": {"mean": round(st.fmean(reason_lens), 1) if reason_lens else None,
                                     "median": st.median(reason_lens) if reason_lens else None},
        }
    if reference_arm and reference_arm in per_arm_answers:
        ref = per_arm_answers[reference_arm]
        for arm in arms:
            if arm == reference_arm:
                continue
            other = per_arm_answers.get(arm, {})
            common = sorted(set(ref) & set(other))
            agree = sum(1 for i in common if ref[i] == other[i])
            out["arms"].setdefault(arm, {})["exact_answer_agreement_with_" + reference_arm] = {
                "n_common": len(common), "n_exact_agree": agree,
                "rate": round(agree / len(common), 4) if common else None,
            }
    return out


# --------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["qeval", "hardset"], required=True)
    ap.add_argument("--arms", nargs="+", required=True, help="two or more arm names as used under data/fidelity/tasks/<arm>/")
    ap.add_argument("--reference-arm", default=None, help="hardset only: exact-answer agreement reference")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    if args.set == "qeval":
        if len(args.arms) != 2:
            ap.error("qeval analysis compares exactly two arms")
        result = {"set": "qeval", "arms": args.arms, **qeval_analysis(args.arms[0], args.arms[1])}
    else:
        result = {"set": "hardset", "arms": args.arms,
                  **hardset_descriptive(args.arms, args.reference_arm)}

    out_path = pathlib.Path(args.out) if args.out else METRICS_DIR / f"tasks-{args.set}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    print(f"\nwrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
