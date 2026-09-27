#!/usr/bin/env python3
"""Build the fidelity campaign report: docs/fidelity/REPORT.md and report.html.

Reads the public aggregates only: docs/fidelity/metrics-v2/*.json, corpus-summary.json,
boots/*.json, optional task results (metrics/tasks-*.json) and corruption-probe summaries
(metrics/corruption-*.json), voxel checks
(voxel/checks.json), the figure index plots/plots.json with its SVG files, and an
optional hand-written verdict (verdict.md; absent means the verdict is "pending"), and an optional
plain-language summary (eli5.md) shown above it.
The HTML file is self-contained: inline CSS, figures as inline SVG, no scripts and no
external requests, light and dark themes, readable at phone width.

After writing, a leak check scans the report and the figure SVGs for absolute user
paths, private corpus paths, IPv4 addresses, corpus window ids and, when the ignored
site files exist, the site's hostnames and addresses and the corpus project names. A
finding names the category and file only and fails the build (exit 2).

Stdlib only.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[2]
REGIMES = ("dense", "sparse", "all")
BUCKETS = ("0-2K", "2-8K", "8-32K", "32-64K", "64K+")
REGIME_NAME = {"dense": "Dense (≤ 2,048 conditioning tokens)", "sparse": "Sparse (> 2,048)", "all": "All positions"}
OUTCOME = {"within_margin": "within", "exceeds_margin": "exceeds", "unresolved": "unresolved"}
PAIRS = [  # (public name, heading, pre-registered)
    ("cm-vs-r0", "Cm (E29 in measurement mode) vs R0", True),
    ("cpre-vs-r0", "Cpre (recipe before E21) vs R0", False),
    ("n-vs-r0", "N (NVFP4 recipe) vs R0", False),
    ("cm-vs-cpre", "Cm vs Cpre", False),
    ("n-vs-cm", "N vs Cm", False),
]
ARMS = [
    ("R0", "`r0fp8-m` (serving: `r0-s`)", "September 18 recipe: vendor FP8 weights as served, FP8 KV, no hybrid KDA, "
     "no mHC, no E21/E22b", "reference"),
    ("Cm", "`cm`", "E29 default in measurement mode", "pre-registered candidate"),
    ("Cp", "`cp-s` / default", "E29 default in production (qeval tasks and corruption probe)", "candidate"),
    ("Cpre", "`cpre-m`, `cpre-s`", "September 19 E03 recipe (hybrid KDA + E03 mHC), before E21", "descriptive"),
    ("N", "`n-m`, `n-s`", "Published NVFP4 recipe (NVFP4 expert weights, its own engine build) on this fabric",
     "descriptive"),
    ("Z", "none", "Cloud `glm-5.3-flash` (behavioural context for tasks only)", "context"),
    ("Ladder", "`l0919-m`, `le21-m`, `le22b-m`", "Frozen intermediate references between R0 and Cm", "attribution"),
]
LIMITATIONS = [
    "**Coarse KL is a lower bound.** Tokens outside both top-K rows are merged into one cell, so by the "
    "data-processing inequality every KL here is a lower bound on the full-vocabulary KL. A small value does "
    "not certify a small full KL; the covered probability mass and top-K overlap are reported with each "
    "comparison, and any within-margin verdict is scoped to these coarse metrics.",
    "**R0 is not a BF16 reference.** R0 is the vendor FP8 checkpoint served by the September 18 recipe on "
    "this stack, with an FP8 KV cache (BF16 KV would need a different attention backend than the recipe as "
    "served, amendment 6). Every arm shares that "
    "FP8-KV error. The numbers are therefore not comparable with published full-vocabulary KL figures measured "
    "against a BF16 model (for example values around 0.021 nats).",
    "**Finite corpus.** {corpus} from coding-agent sessions, "
    "synthetic Italian conversations and model-generated continuations. Other workloads may differ. Owner "
    "exclusions, when listed, drop whole windows without re-measurement.",
    "**Engine nondeterminism beyond 2,048 tokens.** On this engine build, positions conditioned on more than "
    "2,048 tokens differ from run to run even for the same arm (amendment 12); single-execution sparse-regime "
    "differences are operational disagreement. Three executions per arm were collected (amendment 13), but no "
    "validated estimator exists for fidelity between population-average distributions under unequal execution "
    "variability, so the sparse verdict stays unresolved. The indexer explanation remains a hypothesis. The NVFP4 engine build is nondeterministic "
    "from the first positions.",
    "**No cloud reference.** The cloud arm (Z) was deferred for lack of a spending cap, so the tasks have no "
    "hosted-model comparison.",
    "**Tasks cannot show 2 pp equivalence.** {tasks}; a non-significant difference is not equivalence.",
    "**Measurement mode.** Prompt scoring runs cold and serially (one sequence, fresh cache salts). The "
    "production-bridge controls and the DFlash2 exact-match check were not run (amendment 15 limits the "
    "serving phase to tasks and the probe), so the "
    "distribution verdict is scoped to that mode; the tasks and the corruption probe ran on the production "
    "recipe (Cp), speculative decoding included.",
    "**Descoped work.** Decode-set generations, sampled and hardset task runs, tasktime, Cpre serving "
    "(amendment 15) and the voxel showcase (amendment 16) were not run.",
]
# Figures whose inputs were descoped (amendments 15-16): listed once instead of shown as placeholders.
DESCOPED_FIGURES = {"09-decode-path": "decode-set generations (amendment 15)",
                    "13-voxel-showcase": "voxel showcase (amendment 16)"}
HOW_TO_READ = [
    "**KL divergence (nats)** measures how much the arm's next-token probabilities differ from R0's at one "
    "position; larger means more different. It is computed on a partition: the tokens in both top-20 lists, "
    "the actual next token and one \"everything else\" bucket. 0 means agreement on that partition, and the "
    "value can only under-estimate the full difference.",
    "**Top-1 agreement** is how often both put the same token first: the token greedy decoding would pick.",
    "**ΔNLL** is the change in surprise (negative log-probability) for the token that actually came next. "
    "Positive means the arm found real text less likely than R0 did.",
    "**Perplexity change** translates ΔNLL into a percentage: exp(ΔNLL) − 1, so +0.01 nats per token is about "
    "+1% perplexity (worse) and −0.01 nats about −1% (better); an interval that includes 0 means no detectable "
    "quality difference from R0.",
    "**Dense and sparse.** Positions that see at most 2,048 earlier tokens (dense) are reproducible on this "
    "engine; beyond 2,048 (sparse) even R0 disagrees with itself between runs.",
    "**Floor and excess.** The floor is R0 compared with a second run of itself; excess is how far an arm goes "
    "beyond that. Verdict labels: *within* (95% upper bound below the pre-registered margin), *exceeds* (95% "
    "lower bound at or above it), *unresolved* (neither).",
]


# ---------------------------------------------------------------- inputs


def read_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_inputs(docs: Path) -> dict:
    metrics = {}
    for path in sorted((docs / "metrics-v2").glob("*.json")):
        data = read_json(path)
        if isinstance(data, dict):
            metrics[path.stem] = data
    boots = []
    for path in sorted((docs / "boots").glob("*.json")):
        data = read_json(path)
        if isinstance(data, dict):
            boots.append(data)
    tasks = [t for t in (read_json(p) for p in sorted((docs / "metrics").glob("tasks-*.json"))) if isinstance(t, dict)]
    tasks.sort(key=lambda t: t.get("set") != "qeval")
    verdict_path = docs / "verdict.md"
    verdict = verdict_path.read_text(encoding="utf-8").strip() if verdict_path.exists() else None
    eli5_path = docs / "eli5.md"
    eli5 = eli5_path.read_text(encoding="utf-8").strip() if eli5_path.exists() else None
    probes = [p for p in (read_json(q) for q in sorted((docs / "metrics").glob("corruption-*.json")))
              if isinstance(p, dict)]
    plots = (read_json(docs / "plots" / "plots.json") or {}).get("figures", [])
    return {"docs": docs, "metrics": metrics, "summary": metrics.pop("summary", None),
            "corpus": read_json(docs / "corpus-summary.json"), "boots": boots, "tasks": tasks,
            "voxel": read_json(docs / "voxel" / "checks.json"), "verdict": verdict, "eli5": eli5,
            "probes": probes, "plots": plots}


# ---------------------------------------------------------------- formatting


def num(x, digits=4):
    if x is None:
        return "–"
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "–"
    if x == 0:
        return "0"
    if abs(x) < 10 ** -(digits - 1) or abs(x) >= 10 ** 6:
        return f"{x:.{max(1, digits - 2)}e}"
    return f"{x:.{digits}g}" if abs(x) < 1 else f"{x:,.{max(0, digits - len(str(int(abs(x)))))}f}"


def ci(c, scale=1.0, digits=4):
    if not isinstance(c, dict) or c.get("estimate") is None:
        return "–"
    est = num(c["estimate"] * scale, digits)
    if c.get("ci_low") is None or c.get("ci_high") is None:
        return est
    return f"{est} [{num(c['ci_low'] * scale, digits)}, {num(c['ci_high'] * scale, digits)}]"


def bound(c, key, scale=1.0):
    if not isinstance(c, dict) or c.get(key) is None:
        return "–"
    return num(c[key] * scale)


def outcome(label):
    return OUTCOME.get(label, label or "pending")


def count(x):
    return f"{int(x):,}" if isinstance(x, (int, float)) else "–"


# ---------------------------------------------------------------- document model
# Blocks: ("h", level, text) ("p", text) ("ul", [items]) ("table", headers, rows)
# ("code", text) ("fig", entry) ("box", title, blocks) ("verdict_md", text) ("eli5_md", text)


def pair_status(inp, name):
    res = inp["metrics"].get(name)
    if not res:
        return None, "not analysed yet"
    if res.get("status") not in ("ok", "partial"):
        missing = ", ".join(res.get("missing_runs", [])) or res.get("status")
        return None, f"missing runs: {missing}"
    return res, None


def task_mde_text(m):
    """Approximate paired MDE of a qeval comparison (stats.py mde_approx fields)."""
    if m.get("abs_pass_rate_delta") is None:
        return "approximate MDE unavailable"
    text = (f"approximate paired MDE {num(m['abs_pass_rate_delta'] * 100, 3)} pp from "
            f"{m.get('discordant_pairs', '–')} discordant pairs")
    if m.get("abs_pass_rate_delta_all_discordant") is not None:
        text += f" ({num(m['abs_pass_rate_delta_all_discordant'] * 100, 3)} pp if every item were discordant)"
    return text


def verdict_lines(inp):
    """Five status lines against the pre-registered thresholds (automatic, not the verdict)."""
    lines = []
    cm, why = pair_status(inp, "cm-vs-r0")
    thr = (inp["summary"] or {}).get("thresholds") or {"kl_mean_excess_nats": 0.002, "kl_p99_excess_nats": 0.02,
                                                       "top1_drop_pp": 0.5}
    margins = (f"margins on the 95% upper bounds: mean excess < {thr['kl_mean_excess_nats']} nats, p99 excess < "
               f"{thr['kl_p99_excess_nats']} nats, top-1 drop < {thr['top1_drop_pp']} pp")
    if cm is None:
        lines += [f"Cm vs R0, dense regime: pending ({why}).", "Cm vs R0, sparse regime: pending.",
                  "Cm vs R0, overall: pending."]
    else:
        v = cm.get("verdict") or {}
        d = (cm["regimes"].get("dense") or {}).get("excess") or {}
        lines.append(f"Cm vs R0, dense regime: **{outcome((v.get('dense') or {}).get('outcome'))}**. Mean excess "
                     f"{ci(d.get('kl_mean_excess'))} nats, p99 excess {ci(d.get('kl_p99_excess'))} nats, top-1 drop "
                     f"{ci(d.get('top1_drop_pp'))} pp ({margins}).")
        sp = v.get("sparse") or {}
        lines.append(f"Cm vs R0, sparse regime: **{outcome(sp.get('outcome'))}**"
                     + (f" ({sp['reason']})." if sp.get("reason") else "."))
        lines.append(f"Cm vs R0, overall (dense and sparse): **{outcome((v.get('all') or {}).get('outcome'))}**.")
    qeval = [t for t in inp["tasks"] if t.get("set") == "qeval"]
    if qeval:
        parts = []
        for t in qeval:
            d = t.get("greedy_discordance") or {}
            parts.append(f"{d.get('arm_a')} vs {d.get('arm_b')}: McNemar p = {num(t.get('mcnemar_exact_p'), 3)}")
        lines.append("Tasks: " + "; ".join(parts) + ". Equivalence requires the paired 95% CI within ±2 pp.")
    else:
        lines.append("Tasks (R0 vs Cp, Cp vs Z): pending; no task results yet.")
    comps = (inp["summary"] or {}).get("comparisons") or {}
    # A bit-identity check that ran reports pass or fail; both mean the data exist.
    have = ("ok", "partial", "pass", "fail")
    done = [n for n, c in comps.items() if c.get("status") in have]
    missing = [n for n, c in comps.items() if c.get("status") not in have]
    lines.append(f"Data completeness: {len(done)} of {len(comps)} configured analyses have data"
                 + (f"; pending: {', '.join(sorted(missing))}." if missing else "."))
    return lines


def nvfp4_line(inp):
    n, why = pair_status(inp, "n-vs-r0")
    if n is None:
        return f"NVFP4 (N, no pre-registered threshold): pending ({why})."
    parts = []
    for regime in ("dense", "sparse"):
        c = (n["regimes"].get(regime) or {}).get("contrast") or {}
        if c:
            parts.append(f"{regime} mean KL vs R0 {ci(c.get('kl', {}).get('mean_ci'))} nats, top-1 agreement "
                         f"{ci(c.get('top1_agreement'), 100, 3)}%")
    cm, _ = pair_status(inp, "cm-vs-r0")
    if cm:
        c = (cm["regimes"].get("dense") or {}).get("contrast") or {}
        parts.append(f"for comparison Cm dense {ci(c.get('kl', {}).get('mean_ci'))} nats")
    w = n.get("windows") or {}
    return (f"NVFP4 (N, descriptive, no pre-registered threshold; {w.get('compared')} of {w.get('expected')} "
            f"windows): " + "; ".join(parts) + ". Its sparse regime is unresolved: repeated executions are descriptive, "
            "with no validated estimator for fidelity between population-average distributions.")


def ppl_ci(c):
    """Perplexity change in % (exp(ΔNLL) − 1) with its 95% CI, from a ΔNLL CI dict."""
    if not isinstance(c, dict) or c.get("estimate") is None:
        return "–"
    f = lambda x: (math.exp(x) - 1.0) * 100.0  # noqa: E731
    text = f"{f(c['estimate']):+.2f}%"
    if c.get("ci_low") is not None and c.get("ci_high") is not None:
        text += f" [{f(c['ci_low']):+.2f}, {f(c['ci_high']):+.2f}]"
    return text


def ppl_reading(c):
    if not isinstance(c, dict) or c.get("ci_low") is None or c.get("ci_high") is None:
        return "pending"
    if c["ci_low"] <= 0 <= c["ci_high"]:
        return "no detectable change"
    return "worse than R0" if c["ci_low"] > 0 else "better than R0"


def dnll_of(res, regime, cat=None):
    blk = ((res or {}).get("regimes") or {}).get(regime) or {}
    if cat is not None:
        blk = (blk.get("by_category") or {}).get(cat) or {}
    return (blk.get("contrast") or {}).get("delta_nll")


def quality_lines(inp):
    """Plain-language answers: E29 vs FP8 and E29 vs NVFP4, as perplexity change with CIs."""
    lines = []
    cm, why = pair_status(inp, "cm-vs-r0")
    if cm is None:
        lines.append(f"**E29 vs FP8:** pending ({why}).")
    else:
        parts = [f"{r} {ppl_ci(dnll_of(cm, r))} ({ppl_reading(dnll_of(cm, r))})" for r in REGIMES]
        floor, _ = pair_status(inp, "floor-r0")
        ref = f"; R0 against itself: all positions {ppl_ci(dnll_of(floor, 'all'))}" if floor else ""
        lines.append("**E29 vs FP8:** perplexity change of E29 (Cm) against R0: " + "; ".join(parts) + ref + ".")
    n, why = pair_status(inp, "n-vs-cm")
    if n is None:
        lines.append(f"**E29 vs NVFP4:** pending ({why}).")
    else:
        parts = [f"{r} {ppl_ci(dnll_of(n, r))}" for r in REGIMES]
        nr, _ = pair_status(inp, "n-vs-r0")
        vs_r0 = f"; NVFP4 against R0: all positions {ppl_ci(dnll_of(nr, 'all'))}" if nr else ""
        lines.append("**E29 vs NVFP4:** perplexity change of NVFP4 (N) against E29 (Cm): " + "; ".join(parts)
                     + vs_r0 + " (positive: NVFP4 predicts real text worse than the stated reference).")
    return lines


def quality_table(inp):
    rows = []
    for name, label in (("cm-vs-r0", "E29 (Cm) vs R0"), ("cpre-vs-r0", "Cpre vs R0"), ("n-vs-r0", "NVFP4 (N) vs R0"),
                        ("n-vs-cm", "NVFP4 (N) vs E29 (Cm)"), ("floor-r0", "R0 vs itself (run B vs A)")):
        res, why = pair_status(inp, name)
        rows.append([label] + ([ppl_ci(dnll_of(res, r)) for r in REGIMES] if res else ["pending"] * 3))
    lad = inp["metrics"].get("ladder") or {}
    for e in lad.get("cumulative_from_first") or []:
        if e.get("status") in ("ok", "partial") and not e["cand"].startswith(("Cm/", "Cpre/")):
            c = ((e.get(lad.get("regime", "dense")) or {}).get("contrast") or {}).get("delta_nll")
            rows.append([f"{e['cand'].split('/')[0]} vs R0 (ladder subset, {e.get('windows', {}).get('compared')} "
                         "windows)", ppl_ci(c), "–", "–"])
    return ("table", ["Comparison", "Dense (≤ 2,048)", "Sparse (> 2,048)", "All positions"], rows)


def final_outcome(inp):
    cm, _ = pair_status(inp, "cm-vs-r0")
    if cm is None or not cm.get("verdict"):
        return {"dense": "pending", "sparse": "pending", "all": "pending"}
    return {r: outcome((cm["verdict"].get(r) or {}).get("outcome")) for r in REGIMES}


def contrast_table(res, with_floor):
    headers = ["Regime", "Positions", "Mean KL [95% CI]", "p99 KL", "Top-1 agreement %", "ΔNLL [95% CI]",
               "Covered mass (ref)", "Top-K overlap (median)"]
    rows = []
    for regime in REGIMES:
        blk = res["regimes"].get(regime) or {}
        c = blk.get("contrast")
        if not c:
            rows.append([REGIME_NAME[regime], "–", blk.get("status", "–")] + ["–"] * 5)
            continue
        rows.append([REGIME_NAME[regime], count(blk.get("positions_joint_valid")), ci(c["kl"].get("mean_ci")),
                     num(c["kl"].get("p99")), ci(c.get("top1_agreement"), 100, 4), ci(c.get("delta_nll")),
                     num((c.get("covered_mass_ref") or {}).get("mean")),
                     num((c.get("topk_overlap") or {}).get("median"), 3)])
    tables = [("table", headers, rows)]
    if with_floor:
        headers = ["Regime", "Floor mean KL", "Mean excess [95% CI]", "UB95", "p99 excess [95% CI]", "UB95",
                   "Top-1 drop pp [95% CI]", "UB95", "Verdict"]
        rows = []
        verdict = res.get("verdict") or {}
        for regime in REGIMES:
            blk = res["regimes"].get(regime) or {}
            ex = blk.get("excess") or {}
            f = (blk.get("floor") or {}).get("kl", {}).get("mean")
            v = verdict.get(regime) or {}
            label = outcome(v.get("outcome"))
            if regime == "sparse" and v.get("reason"):
                label += " (needs ≥ 3 executions)" if "executions" in v["reason"] else ""
            rows.append([REGIME_NAME[regime], num(f), ci(ex.get("kl_mean_excess")),
                         bound(ex.get("kl_mean_excess"), "ub95"), ci(ex.get("kl_p99_excess")),
                         bound(ex.get("kl_p99_excess"), "ub95"), ci(ex.get("top1_drop_pp")),
                         bound(ex.get("top1_drop_pp"), "ub95"), label])
        tables.append(("table", headers, rows))
    return tables


def build_blocks(inp) -> list:
    B = []
    summary = inp["summary"] or {}
    B.append(("h", 1, "GLM-5.3-Flash fidelity campaign: report"))
    gen = summary.get("generated_utc", "not yet analysed")
    commit = ((summary.get("harness_git") or {}).get("commit") or "")[:12]
    dirty = " with uncommitted changes" if (summary.get("harness_git") or {}).get("dirty") else ""
    B.append(("p", f"How far the E29 recipe's next-token distributions deviate from the vendor FP8 model served "
                   f"without this repository's precision and runtime changes (R0), and how a generic NVFP4 recipe "
                   f"compares. Pre-registration and amendments: [PLAN.md](PLAN.md). Metrics generated {gen}"
                   + (f" from harness commit `{commit}`{dirty}." if commit else ".")))

    if inp.get("eli5"):
        B.append(("h", 2, "In plain words"))
        B.append(("eli5_md", inp["eli5"]))

    # 1. Verdict
    B.append(("h", 2, "1. Verdict"))
    if inp["verdict"]:
        B.append(("verdict_md", inp["verdict"]))
    else:
        B.append(("p", "**Verdict: pending.** The owner-reviewed verdict (`docs/fidelity/verdict.md`) has not been "
                       "written. The lines below are the automatic status of the pre-registered criteria on the "
                       "data measured so far; they are not the verdict."))
    fo = final_outcome(inp)
    B.append(("p", f"**Pre-registered outcome, Cm vs R0:** dense {fo['dense']} · sparse {fo['sparse']} · "
                   f"overall {fo['all']}."))
    B.append(("ul", verdict_lines(inp)))
    B.append(("p", "**NVFP4 answer (separate).** " + nvfp4_line(inp)))
    B.append(("box", "How to read this", [("ul", HOW_TO_READ)]))
    B.append(("h", 3, "Quality at a glance: E29 vs FP8, E29 vs NVFP4"))
    B.append(("p", "Perplexity change on the real next token, exp(ΔNLL) − 1 with 95% group-bootstrap CIs; lower is "
                   "better. This answers \"is it worse?\"; the KL results below answer \"is it different?\"."))
    B.append(("ul", quality_lines(inp)))
    B.append(quality_table(inp))
    for entry in inp["plots"]:
        if entry.get("group") == "quality":
            B.append(("fig", entry))

    # 2. Method, arms, overlays
    B.append(("h", 2, "2. Method, arms and overlays"))
    runs = summary.get("runs") or {}
    measured = {k.split("/")[0] for k, v in runs.items() if v.get("windows_ok")}
    task_arms = {arm for t in inp["tasks"] for arm in (t.get("arms") or [])}
    probe_arms = {p.get("label") for p in inp["probes"]}

    def scored(arm):
        if arm == "Ladder":
            return "prompt scoring (ladder subset)" if measured & {"L0919", "LE21", "LE22b"} else "no"
        parts = [label for label, arms in (("prompt scoring", measured), ("tasks", task_arms),
                                           ("corruption probe", probe_arms)) if arm in arms]
        return ", ".join(parts) or ("not run (deferred)" if arm == "Z" else "no")
    B.append(("table", ["Arm", "Overlay", "Recipe", "Role", "Scored in this report"],
              [[a, o, r, role, scored(a)] for a, o, r, role in ARMS]))
    B.append(("p", "Every measurement arm uses the same measurement deltas, intended not to change numerics: a 6 GiB "
                   "KV pool per rank, one sequence at a time, `--max-logprobs 100`, a fresh `cache_salt` per request "
                   "and the SparkCache store/restore disabled in its own namespace. Prompt scoring sends each "
                   "corpus window teacher-forced with `prompt_logprobs` K = 20 (K = 100 on a fixed 20% subset)."))
    if inp["boots"]:
        rows = []
        for b in inp["boots"]:
            markers = sorted({line.split()[0] for r in b.get("ranks", []) for line in r.get("ready_lines", [])
                              if line.split()})
            rows.append([b.get("label", "–"), Path(b.get("overlay") or "–").name,
                         (b.get("overlay_sha256") or "")[:12] or "–",
                         f"{b.get('model_repo', '–')} @ {(b.get('model_rev') or '')[:10]}",
                         (b.get("image") or "").split("@")[-1][:19] or "–", b.get("kv_cache_dtype", "–"),
                         str(b.get("spec_tokens", "–")), str(b.get("k", "–")),
                         ", ".join(m.replace("_READY", "") for m in markers) or "none"])
        B.append(("p", "Boot identities recorded for each measurement or serving boot:"))
        B.append(("table", ["Boot", "Overlay", "Overlay sha256", "Model @ revision", "Image digest", "KV", "Spec k",
                            "K", "READY markers"], rows))
    method = summary.get("method") or []
    if method:
        B.append(("h", 3, "Statistical method"))
        B.append(("ul", method))
    if summary.get("bootstrap"):
        bs = summary["bootstrap"]
        grouping = summary.get("grouping") or {}
        B.append(("p", f"Bootstrap: B = {bs.get('B')}, seed {bs.get('seed')}, unit {bs.get('unit')} "
                       f"({grouping.get('groups', '–')} groups over {grouping.get('windows', '–')} windows), "
                       f"{bs.get('estimator')}; {bs.get('ci')}."))

    # 3. Corpus
    B.append(("h", 2, "3. Corpus"))
    cs = inp["corpus"]
    if not cs:
        B.append(("p", "Corpus summary pending."))
    else:
        B.append(("p", f"{count(cs.get('windows'))} windows, {count(cs.get('total_tokens'))} tokens, "
                       f"{count(cs.get('scored_positions'))} scored positions; {count(cs.get('windows_at_least_3072'))} "
                       f"windows have at least 3,072 tokens. Manifest "
                       f"{'frozen' if cs.get('frozen') else 'provisional (owner exclusions not yet listed)'}, "
                       f"global sha256 `{(cs.get('manifest_global_sha256') or '')[:16]}…`. Rendered with the pinned "
                       f"tokenizer and chat template of `{cs.get('model_repo')}` at `{(cs.get('model_rev') or '')[:10]}`."))
        for key, title in (("by_category", "Category"), ("by_source", "Source")):
            rows = [[k.replace("_", " "), count(v.get("windows")), count(v.get("tokens")),
                     f"{v.get('share_of_tokens', 0) * 100:.1f}%"] for k, v in (cs.get(key) or {}).items()]
            B.append(("table", [title, "Windows", "Tokens", "Share of tokens"], rows))
        hist = cs.get("length_histogram") or []
        if hist:
            B.append(("table", ["Window length (tokens)", "Windows"],
                      [[f"{h['min']:,}–{h['max_exclusive'] - 1:,}", str(h["count"])] for h in hist if h.get("count")]))
        red = cs.get("redaction_counts") or {}
        if red:
            B.append(("p", "Redactions before rendering (counts only): "
                           + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in red.items() if v) + "."))
        dp = cs.get("decode_prompts") or {}
        if dp:
            B.append(("p", f"Decode set: {dp.get('count')} prompts ("
                           + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in (dp.get('by_source') or {}).items())
                           + ")."))

    # 4. Results
    B.append(("h", 2, "4. Results"))
    B.append(("p", "All KL values are coarse top-K lower bounds in nats. Confidence intervals are 95% percentile "
                   "bootstraps over source groups; UB95 is the one-sided 95% upper bound the verdict uses. Sparse-regime "
                   "excess from one execution is additional operational disagreement, not fidelity."))
    for name, heading, prereg in PAIRS:
        B.append(("h", 3, heading + (" (pre-registered)" if prereg else " (descriptive)")))
        res, why = pair_status(inp, name)
        if res is None:
            B.append(("p", f"Pending: {why}."))
            continue
        w = res.get("windows") or {}
        B.append(("p", f"Reference `{res.get('ref')}`, candidate `{res.get('cand')}`, floor `{res.get('floor') or 'none'}`; "
                       f"{w.get('compared')} of {w.get('expected')} windows ({res.get('status')}), {res.get('groups')} "
                       f"source groups."))
        B.extend(contrast_table(res, bool(res.get("floor"))))

    B.append(("h", 3, "Repeatability floor"))
    floor, why = pair_status(inp, "floor-r0")
    if floor is None:
        B.append(("p", f"Pending: {why}."))
    else:
        B.append(("p", "R0 run B against run A on the same boot. The dense regime is bit-identical; the sparse regime "
                       "is not (amendment 12)."))
        B.extend(contrast_table(floor, False))
        buckets = floor.get("by_position_bucket") or {}
        if buckets:
            B.append(("table", ["Position bucket", "Positions", "Mean KL [95% CI]", "Top-1 agreement %"],
                      [[b, count(blk.get("positions_joint_valid")), ci(blk["contrast"]["kl"].get("mean_ci")),
                        ci(blk["contrast"].get("top1_agreement"), 100, 4)]
                       for b, blk in sorted(buckets.items(), key=lambda kv: BUCKETS.index(kv[0])
                                            if kv[0] in BUCKETS else len(BUCKETS)) if blk.get("contrast")]))
    cb, why = pair_status(inp, "crossboot-r0")
    B.append(("p", "Cross-boot R0 floor: " + (f"pending ({why})." if cb is None else
                                               f"mean KL {ci(cb['regimes']['all']['contrast']['kl'].get('mean_ci'))} nats "
                                               f"over {cb['windows']['compared']} windows.")))

    rep = inp["metrics"].get("repeated-crossboot")
    B.append(("h", 3, "Sparse regime from repeated executions"))
    if not rep:
        B.append(("p", "Pending: not analysed yet."))
    else:
        rows = []
        for key, blk in (rep.get("pairs") or {}).items():
            sp = (blk.get("regimes") or {}).get("sparse") or {}
            rows.append([key.replace("-vs-", " vs "), blk.get("status", "–"),
                         ci(sp.get("kl_averaged")), ci(sp.get("cross_pairwise")), ci(sp.get("within_ref_pairwise")),
                         ci(sp.get("within_cand_pairwise")), ci(sp.get("top1_drop_pairwise_pp")),
                         outcome((blk.get("verdict") or {}).get("sparse")),
                         sp.get("reason") or blk.get("reason", "")])
        B.append(("p", f"At least {rep.get('min_executions', 3)} executions per arm on the stratified cross-boot subset "
                       "(amendment 13); descriptive quantities (mean coarse KL, nats)."))
        B.append(("table", ["Pair", "Status", "KL of averaged distributions", "Cross-arm pairwise KL",
                            "Within ref pairwise KL", "Within cand pairwise KL", "Top-1 drop pp (pairwise)",
                            "Sparse verdict", "Note"], rows))

    bits = inp["metrics"].get("bit-identity")
    B.append(("h", 3, "Dense-regime bit identity"))
    if not bits or not bits.get("checks"):
        B.append(("p", "Pending: not analysed yet."))
    else:
        B.append(("p", "Rows 1–2,048 of the actual-token logprob and top-K arrays compared bit for bit between two "
                       "executions of the same arm."))
        B.append(("table", ["Check", "Runs", "Status", "Windows identical", "Rows differing / compared"],
                  [[name, f"{c.get('ref')} vs {c.get('cand')}", c.get("status", "–"),
                    f"{c.get('windows_identical', '–')} / {c.get('windows_compared', '–')}",
                    f"{count(c.get('rows_differing'))} / {count(c.get('rows_compared'))}"]
                   for name, c in bits["checks"].items()]))
        controls = (inp["metrics"].get("ladder") or {}).get("negative_controls") or {}
        if controls:
            B.append(("p", "Ladder negative controls (steps that must not change target logits; a strict control "
                           "must be bit-identical):"))
            B.append(("table", ["Step", "Strict", "Status", "Windows identical", "Rows differing / compared"],
                      [[f"{c.get('ref')} → {c.get('cand')}", "yes" if c.get("strict") else "no", c.get("status", "–"),
                        f"{c.get('windows_identical', '–')} / {c.get('windows_compared', '–')}",
                        f"{count(c.get('rows_differing'))} / {count(c.get('rows_compared'))}"]
                       for c in controls.values()]))

    mde = inp["metrics"].get("mde")
    B.append(("h", 3, "Minimum detectable effect"))
    if not mde or mde.get("status") != "ok":
        B.append(("p", "Pending."))
    else:
        g = mde.get("governing_mde_kl_mean_nats") or {}
        B.append(("p", "MDE of the mean excess KL (2.8 × bootstrap SE of reference-only contrasts): "
                       + ", ".join(f"{r} {num(g.get(r))} nats" for r in REGIMES)
                       + ". A zero MDE means the contrast is bit-identical in that regime."))

    k = inp["metrics"].get("k-cm-vs-r0")
    B.append(("h", 3, "K sensitivity"))
    if not k or k.get("status") not in ("ok", "partial"):
        B.append(("p", "Pending" + (f" (missing runs: {', '.join(k.get('missing_runs', []))})." if k else ".")))
    else:
        rows = []
        for regime in REGIMES:
            blk = (k.get("regimes") or {}).get(regime) or {}
            if "high_K" not in blk:
                continue
            rows.append([REGIME_NAME[regime], ci(blk["high_K"]["kl"].get("mean_ci")),
                         ci(blk["truncated_low_K"]["kl"].get("mean_ci")), ci(blk.get("kl_mean_high_minus_truncated")),
                         ci(blk.get("top1_agreement_truncated_minus_high_pp"))])
        B.append(("table", ["Regime", "Mean KL, K = 100", f"Mean KL, truncated K = {k.get('K_low')}",
                            "Difference", "Top-1 difference pp"], rows))

    lad = inp["metrics"].get("ladder")
    B.append(("h", 3, "Ladder attribution (dense regime)"))
    if not lad:
        B.append(("p", "Pending: not analysed yet."))
    else:
        rows = []
        regime = lad.get("regime", "dense")
        for kind, entries in (("cumulative", lad.get("cumulative_from_first") or []), ("step", lad.get("steps") or [])):
            for e in entries:
                c = (e.get(regime) or {}).get("contrast") or {}
                rows.append([kind, f"{e['ref']} → {e['cand']}" + (" (negative control)" if e.get("negative_control") else ""),
                             e.get("status", "–"), ci(c.get("kl", {}).get("mean_ci")),
                             ci(c.get("top1_agreement"), 100, 4)])
        B.append(("table", ["Kind", "Step", "Status", "Mean KL [95% CI]", "Top-1 agreement %"], rows))

    B.append(("h", 3, "Tasks"))
    if not inp["tasks"]:
        B.append(("p", "Pending: task runs happen on the serving boots after the measurement boots."))
    for t in inp["tasks"]:
        if t.get("set") == "qeval":
            d = t.get("greedy_discordance") or {}
            a, b = d.get("arm_a"), d.get("arm_b")
            diff = t.get("sampled_bootstrap_diff") or {}
            B.append(("p", f"qeval, {a} vs {b}: {d.get('n_paired')} paired items; exact McNemar p = "
                           f"{num(t.get('mcnemar_exact_p'), 3)}; sampled pass-rate difference "
                           + (f"{num(diff['diff'] * 100, 3)} pp [{num(diff['ci95'][0] * 100, 3)}, "
                              f"{num(diff['ci95'][1] * 100, 3)}]" if diff.get("diff") is not None
                              else "not measured (sampled runs descoped, amendment 15)")
                           + "; " + task_mde_text(t.get("mde_approx") or {}) + "."))
            B.append(("table", ["", f"{b} pass", f"{b} fail"],
                      [[f"{a} pass", str(d.get("both_pass")), str(d.get(f"{a}_only_pass"))],
                       [f"{a} fail", str(d.get(f"{b}_only_pass")), str(d.get("both_fail"))]]))
            B.append(("table", ["Arm", "Greedy pass rate [Wilson 95% CI]"],
                      [[arm, f"{w['rate'] * 100:.1f}% [{w['lo95'] * 100:.1f}, {w['hi95'] * 100:.1f}]"]
                       for arm, w in (t.get("wilson_ci") or {}).items() if w]))
        elif t.get("set") == "hardset":
            arms = t.get("arms") if isinstance(t.get("arms"), dict) else {}
            B.append(("table", ["hardset arm", "Results", "Truncated", "Finish reasons"],
                      [[arm, str(v.get("n_results", "–")), str(v.get("truncated", "–")),
                        ", ".join(f"{k} {n}" for k, n in (v.get("finish_reasons") or {}).items())]
                       for arm, v in arms.items()]))

    if inp["probes"]:
        B.append(("h", 3, "Corruption probe"))
        B.append(("p", "40 Italian prompts and 10 tool-call prompts at temperature 0 on the serving recipes "
                       "(motivated by vLLM issue 54150 on ModelOpt NVFP4 checkpoints). UTF-8 validity is checked on "
                       "the concatenated token bytes; U+FFFD counts replacement characters in the returned text; "
                       "repetition flags are heuristic. The probe did not run on R0."))
        B.append(("table", ["Arm", "Italian replies", "Invalid UTF-8", "With U+FFFD (characters)", "Repetition flags",
                            "Truncated", "Tool calls made", "Tool-call parse failures"],
                  [[p.get("label", "–"), str(p.get("italian_prompts", "–")), str(p.get("italian_invalid_utf8", "–")),
                    f"{p.get('italian_with_fffd', '–')} ({p.get('italian_fffd_total', '–')})",
                    str(p.get("italian_repetition_flags", "–")), str(p.get("italian_truncated", "–")),
                    f"{p.get('tool_calls_made', '–')} / {p.get('tool_prompts', '–')}",
                    str(p.get("tool_parse_failures", "–"))] for p in inp["probes"]]))

    B.append(("h", 3, "Voxel showcase"))
    vox = (inp["voxel"] or {}).get("results") or []
    if not vox:
        B.append(("p", "Deferred to a possible next step (amendment 16); no rendered outputs in this campaign."))
    else:
        B.append(("p", f"{len(vox)} rendered outputs; "
                       f"{sum(1 for r in vox if not r.get('console_errors') and not r.get('load_error'))} without console "
                       "errors. The blind gallery is published separately. A visual sanity check only."))

    missing_runs = sorted(k for k, v in runs.items() if not v.get("windows_ok"))
    if runs:
        B.append(("h", 3, "Run inventory"))
        B.append(("table", ["Run", "K", "Windows scored", "Invalid rows", "Missing actual-token entries"],
                  [[k, str(v.get("K") or "–"), f"{v.get('windows_ok', 0)} / "
                    f"{(v.get('missingness_vs_manifest') or {}).get('windows_expected', '–')}",
                    count(((v.get("missingness_vs_manifest") or {}).get("invalid_rows") or {}).get("all")),
                    count(((v.get("missingness_vs_manifest") or {}).get("missing_actual") or {}).get("all"))]
                   for k, v in sorted(runs.items()) if v.get("windows_ok")]))
        if missing_runs:
            B.append(("p", "Configured but not yet measured: " + ", ".join(f"`{r}`" for r in missing_runs) + "."))

    B.append(("h", 3, "Figures"))
    if not inp["plots"]:
        B.append(("p", "Pending: run `scripts/fidelity/plots.py`."))
    skipped = []
    for entry in inp["plots"]:
        if entry.get("status") == "pending" and entry.get("name") in DESCOPED_FIGURES:
            skipped.append(f"{entry.get('title')}: {DESCOPED_FIGURES[entry['name']]}")
        elif entry.get("group") != "quality":
            B.append(("fig", entry))
    if skipped:
        B.append(("p", "Not produced, inputs descoped: " + "; ".join(skipped) + "."))

    # 5. Limitations
    B.append(("h", 2, "5. Limitations"))
    corpus = (f"{count(cs.get('windows'))} {'frozen' if cs.get('frozen') else 'provisional'} windows "
              f"({count(cs.get('scored_positions'))} scored positions)" if cs else "The corpus")
    mdes = [(t.get("mde_approx") or {}).get("abs_pass_rate_delta") for t in inp["tasks"] if t.get("set") == "qeval"]
    mdes = [m for m in mdes if m is not None]
    tasks_line = (f"With 75 qeval items and the observed discordance, the approximate paired MDE is "
                  f"{num(min(mdes) * 100, 2)}–{num(max(mdes) * 100, 2)} pp" if mdes
                  else "With 75 qeval items the paired MDE is far above 2 pp")
    B.append(("ul", [item.replace("{corpus}", corpus).replace("{tasks}", tasks_line) for item in LIMITATIONS]))

    # 6. Reproduction
    B.append(("h", 2, "6. Reproduction"))
    B.append(("p", "Collection needs the four-node cluster and the private corpus; the analysis, figures and both "
                   "reports regenerate from the raw runs with one command. Collector details: "
                   "[scripts/fidelity/README.md](../../scripts/fidelity/README.md)."))
    B.append(("code", "\n".join([
        "# Overlays and their exact deltas",
        "python3 scripts/fidelity/make_overlays.py --check",
        "python3 scripts/fidelity/make_overlays.py --diff cm",
        "",
        "# Per measurement boot (client on a workstation, never on rank 0)",
        "PY=data/fidelity/.venv/bin/python",
        "$PY scripts/fidelity/collect_prompt_logprobs.py --base-url http://<api-host>:<port> \\",
        "  --arm <ARM> --run prompt-a --K 20 --boot-json <boot-identity.json> --abort-file data/fidelity/ABORT",
        "$PY scripts/fidelity/determinism_probe.py --base-url http://<api-host>:<port> --out <probe.json>",
        "",
        "# Metrics, figures, REPORT.md and report.html from data/fidelity/raw/",
        "bash scripts/fidelity/make_all.sh",
        "",
        "# Offline checks",
        "$PY scripts/fidelity/analyze_campaign.py --selftest",
        "python3 scripts/tests/test-fidelity-report.py",
    ])))
    return B


# ---------------------------------------------------------------- markdown


def md_cell(text):
    return str(text).replace("|", "\\|").replace("\n", " ")


def render_md(blocks) -> str:
    out = []
    for b in blocks:
        kind = b[0]
        if kind == "h":
            out.append("#" * b[1] + " " + b[2])
        elif kind == "p":
            out.append(b[1])
        elif kind == "ul":
            out.append("\n".join(f"- {item}" for item in b[1]))
        elif kind == "table":
            headers, rows = b[1], b[2]
            if not rows:
                continue
            lines = ["| " + " | ".join(md_cell(h) for h in headers) + " |",
                     "|" + "|".join(" --- " for _ in headers) + "|"]
            lines += ["| " + " | ".join(md_cell(c) for c in row) + " |" for row in rows]
            out.append("\n".join(lines))
        elif kind == "code":
            out.append("```sh\n" + b[1] + "\n```")
        elif kind in ("verdict_md", "eli5_md"):
            out.append(b[1])
        elif kind == "box":
            out.append(f"**{b[1]}**\n\n" + render_md(b[2]).strip())
        elif kind == "fig":
            e = b[1]
            status = e.get("status", "")
            text = f"![{e.get('title')}](plots/{e.get('png')})\n\n*{e.get('title')}.* {e.get('caption', '')}"
            if status != "ok":
                text += f" Status: **{status}**."
            notes = [n for n in e.get("notes", []) if n]
            if notes and status != "ok":
                text += " " + " ".join(notes)
            out.append(text)
    return "\n\n".join(out).rstrip() + "\n"


# ---------------------------------------------------------------- html

CSS = """
:root{--bg:#f7f8fa;--fg:#172b46;--muted:#536478;--card:#ffffff;--line:#dfe5ed;--accent:#087f74;
--within:#087f74;--exceeds:#b45309;--unresolved:#536478;--code:#eef1f5}
@media (prefers-color-scheme:dark){:root{--bg:#0f1722;--fg:#e6edf5;--muted:#9fb0c3;--card:#172231;
--line:#2a3a4e;--accent:#4cc2b5;--within:#4cc2b5;--exceeds:#f0a35c;--unresolved:#9fb0c3;--code:#1e2b3c}}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",
Roboto,"Helvetica Neue",Arial,sans-serif;overflow-wrap:break-word}
main{max-width:980px;margin:0 auto;padding:20px 16px 48px}
h1{font-size:1.7rem;line-height:1.25;margin:.4em 0}
h2{font-size:1.35rem;margin:1.8em 0 .5em;padding-top:.4em;border-top:1px solid var(--line)}
h3{font-size:1.1rem;margin:1.4em 0 .4em}
p,li{max-width:72ch}
a{color:var(--accent)}
code{background:var(--code);padding:.05em .3em;border-radius:4px;font-size:.9em;overflow-wrap:anywhere}
pre{background:var(--code);padding:12px;border-radius:8px;overflow-x:auto;font-size:.85em;line-height:1.45}
pre code{background:none;padding:0;overflow-wrap:normal}
.table-wrap{overflow-x:auto;margin:.8em 0;border:1px solid var(--line);border-radius:8px;background:var(--card)}
table{border-collapse:collapse;width:100%;font-size:.88em}
th,td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top;white-space:nowrap}
th{color:var(--muted);font-weight:600}
tr:last-child td{border-bottom:none}
.eli5{background:var(--code);border-radius:10px;padding:6px 18px;margin:1em 0;font-size:1.03em}
.eli5 li{margin:.35em 0}
.verdict{background:var(--card);border:1px solid var(--line);border-left:5px solid var(--accent);border-radius:10px;
padding:14px 18px;margin:1em 0}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin:.6em 0}
.chip{border:1px solid currentColor;border-radius:999px;padding:2px 10px;font-size:.85em;font-weight:600}
.chip.within{color:var(--within)}.chip.exceeds{color:var(--exceeds)}.chip.unresolved,.chip.pending{color:var(--unresolved)}
.verdict li{margin:.35em 0}.verdict li>p{margin:.45em 0}.verdict li>ul{margin:.3em 0}
.box{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 18px;margin:1em 0}
.box h3{margin-top:.4em}
figure{margin:1.2em 0;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px}
.plot{background:#ffffff;border-radius:6px;overflow:hidden}
.plot svg{display:block;width:100%;height:auto}
figcaption{font-size:.9em;color:var(--muted);margin-top:8px}
.status{font-weight:600}
.muted{color:var(--muted);font-size:.9em}
"""


def inline_html(text: str) -> str:
    """Escape, then render `code`, **bold**, *em* and [text](url) links (relative or https only)."""
    parts = re.split(r"(`[^`]+`)", text)
    out = []
    for part in parts:
        if part.startswith("`") and part.endswith("`") and len(part) > 1:
            out.append(f"<code>{html.escape(part[1:-1])}</code>")
            continue
        s = html.escape(part, quote=False)
        s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", s)
        s = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", lambda m: f'<a href="{html.escape(m.group(2))}">{m.group(1)}</a>', s)
        out.append(s)
    return "".join(out)


def md_to_html(text: str) -> str:
    """Minimal Markdown for the hand-written verdict: headings, paragraphs, inline markup, and
    ordered or unordered lists whose items may hold indented bullets and paragraphs (one level)."""
    out, para = [], []
    lst = None  # {"tag": "ul" | "ol", "items": [[[kind, value], ...], ...]}, kind "p" or "ul"
    blank = False
    bullet = re.compile(r"^([-*]|\d+\.)\s+")

    def flush_para():
        if para:
            out.append(f"<p>{inline_html(' '.join(para))}</p>")
            para.clear()

    def flush_list():
        nonlocal lst
        if lst:
            items = []
            for parts in lst["items"]:
                inner = []
                for kind, value in parts:
                    if kind == "ul":
                        inner.append("<ul>" + "".join(f"<li>{inline_html(v)}</li>" for v in value) + "</ul>")
                    else:
                        inner.append(f"<p>{inline_html(value)}</p>" if inner else inline_html(value))
                items.append("<li>" + "".join(inner) + "</li>")
            out.append(f"<{lst['tag']}>" + "".join(items) + f"</{lst['tag']}>")
            lst = None

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            flush_para()
            blank = True
            continue
        m = bullet.match(stripped)
        body = bullet.sub("", stripped) if m else stripped
        if lst and line[:1] in (" ", "\t"):
            parts = lst["items"][-1]
            if m:
                if parts[-1][0] == "ul":
                    parts[-1][1].append(body)
                else:
                    parts.append(["ul", [body]])
            elif blank:
                parts.append(["p", body])
            elif parts[-1][0] == "ul":
                parts[-1][1][-1] += " " + body
            else:
                parts[-1][1] += " " + body
        elif stripped.startswith("#"):
            flush_para()
            flush_list()
            level = min(4, len(stripped) - len(stripped.lstrip("#")) + 2)
            out.append(f"<h{level}>{inline_html(stripped.lstrip('#').strip())}</h{level}>")
        elif m:
            flush_para()
            tag = "ol" if stripped[0].isdigit() else "ul"
            if lst and lst["tag"] != tag:
                flush_list()
            if not lst:
                lst = {"tag": tag, "items": []}
            lst["items"].append([["p", body]])
        else:
            flush_list()
            para.append(stripped)
        blank = False
    flush_para()
    flush_list()
    return "\n".join(out)


def svg_inline(path: Path, label: str) -> str | None:
    try:
        svg = path.read_text(encoding="utf-8")
    except OSError:
        return None
    svg = re.sub(r"<\?xml[^>]*\?>", "", svg)
    svg = re.sub(r"<!DOCTYPE[^>]*>", "", svg, flags=re.S)
    start = svg.find("<svg")
    if start < 0:
        return None
    end = svg.find(">", start)
    tag = svg[start:end]
    tag = re.sub(r'\s(width|height)="[^"]*"', "", tag)
    tag += f' role="img" aria-label="{html.escape(label)}"'
    return (svg[:start] + tag + svg[end:]).strip()


def render_html(blocks, inp) -> str:
    fo = final_outcome(inp)
    body = []
    title = "GLM-5.3-Flash fidelity campaign"
    in_verdict = False
    for b in blocks:
        kind = b[0]
        if kind == "h":
            if in_verdict and b[1] <= 2:
                body.append("</section>")
                in_verdict = False
            if b[1] == 1:
                title = b[2]
            anchor = re.sub(r"[^a-z0-9]+", "-", b[2].lower()).strip("-")
            body.append(f'<h{b[1]} id="{anchor}">{inline_html(b[2])}</h{b[1]}>')
            if b[1] == 2 and b[2].startswith("1."):
                body.append('<section class="verdict">')
                body.append('<div class="chips">' + "".join(
                    f'<span class="chip {html.escape(v)}">Cm vs R0 · {r}: {html.escape(v)}</span>'
                    for r, v in (("dense", fo["dense"]), ("sparse", fo["sparse"]), ("overall", fo["all"]))) + "</div>")
                in_verdict = True
        elif kind == "p":
            body.append(f"<p>{inline_html(b[1])}</p>")
        elif kind == "ul":
            body.append("<ul>" + "".join(f"<li>{inline_html(i)}</li>" for i in b[1]) + "</ul>")
        elif kind == "table":
            if not b[2]:
                continue
            head = "".join(f"<th>{inline_html(str(h))}</th>" for h in b[1])
            rows = "".join("<tr>" + "".join(f"<td>{inline_html(str(c))}</td>" for c in row) + "</tr>" for row in b[2])
            body.append(f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>')
        elif kind == "code":
            body.append(f"<pre><code>{html.escape(b[1])}</code></pre>")
        elif kind == "verdict_md":
            body.append(md_to_html(b[1]))
        elif kind == "eli5_md":
            body.append('<section class="eli5">' + md_to_html(b[1]) + "</section>")
        elif kind == "box":
            inner = render_html_fragment(b[2])
            body.append(f'<aside class="box"><h3>{html.escape(b[1])}</h3>{inner}</aside>')
        elif kind == "fig":
            e = b[1]
            svg = svg_inline(inp["docs"] / "plots" / e.get("svg", ""), e.get("title", "figure"))
            status = e.get("status", "")
            notes = " ".join(n for n in e.get("notes", []) if n) if status != "ok" else ""
            cap = (f"<strong>{html.escape(e.get('title', ''))}.</strong> {inline_html(e.get('caption', ''))}"
                   + (f' <span class="status">Status: {html.escape(status)}.</span> {inline_html(notes)}'
                      if status != "ok" else ""))
            plot = f'<div class="plot">{svg}</div>' if svg else '<p class="muted">Figure file missing.</p>'
            body.append(f"<figure>{plot}<figcaption>{cap}</figcaption></figure>")
    if in_verdict:
        body.append("</section>")
    return ("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
            "<meta name=\"color-scheme\" content=\"light dark\">\n"
            f"<title>{html.escape(title)}</title>\n<style>{CSS}</style>\n</head>\n<body>\n<main>\n"
            + "\n".join(body) + "\n</main>\n</body>\n</html>\n")


def render_html_fragment(blocks) -> str:
    out = []
    for b in blocks:
        if b[0] == "ul":
            out.append("<ul>" + "".join(f"<li>{inline_html(i)}</li>" for i in b[1]) + "</ul>")
        elif b[0] == "p":
            out.append(f"<p>{inline_html(b[1])}</p>")
    return "".join(out)


# ---------------------------------------------------------------- leak check

GENERIC_PATTERNS = [
    ("absolute user path", re.compile(r"/Users/|/home/[a-z]")),
    ("private corpus path", re.compile(r"data/fidelity/corpus")),
    ("private address", re.compile(r"192\.168\.|\b10\.10\.")),
    ("IPv4 address", re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")),
    ("corpus window id", re.compile(r"(?<![\w/+])w\d{4}(?![\w/+])")),
]


def private_terms(cluster_env: Path | None, manifest: Path | None, public_text: str) -> list:
    """Site hostnames/addresses and corpus project names, minus terms already published."""
    terms = set()
    if cluster_env and cluster_env.exists():
        for line in cluster_env.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.match(r"\s*(NODES|NODE_HOSTNAMES|MGMT_IPS|MASTER_IP)(?:_BY_RANK)?=(.*)", line)
            if m:
                for token in re.split(r"[\s\"'()@,]+", m.group(2)):
                    if len(token) >= 3 and not token.startswith("$"):
                        terms.add(("site value", token))
    if manifest and manifest.exists():
        data = read_json(manifest) or {}
        for w in data.get("windows", []):
            project = str(w.get("project") or "")
            if len(project) >= 4:
                terms.add(("corpus project name", project))
    return sorted((label, t) for label, t in terms if t not in public_text)


def leak_check(files, terms) -> list:
    findings = []
    for path in files:
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        # Embedded base64 payloads (voxel screenshots) are opaque bytes, not text.
        text = re.sub(r"base64,[A-Za-z0-9+/=\s]+", "base64,", text)
        for label, pattern in GENERIC_PATTERNS:
            if pattern.search(text):
                findings.append((Path(path).name, label))
        for label, term in terms:
            if re.search(r"(?<![\w-])" + re.escape(term) + r"(?![\w-])", text):
                findings.append((Path(path).name, label))
    return sorted(set(findings))


# ---------------------------------------------------------------- main


def write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--docs", type=Path, default=REPO / "docs/fidelity")
    p.add_argument("--cluster-env", type=Path, default=REPO / "cluster.env",
                   help="ignored site configuration, read only for the leak check")
    p.add_argument("--manifest", type=Path, default=REPO / "data/fidelity/corpus/manifest.json",
                   help="private corpus manifest, read only for the leak check")
    p.add_argument("--no-leak-check", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    inp = load_inputs(args.docs)
    blocks = build_blocks(inp)
    md_path, html_path = args.docs / "REPORT.md", args.docs / "report.html"
    write(md_path, render_md(blocks))
    write(html_path, render_html(blocks, inp))
    print(f"wrote {md_path.name} and {html_path.name} ({len(inp['plots'])} figures, "
          f"verdict {'present' if inp['verdict'] else 'pending'})")
    if args.no_leak_check:
        return 0
    public = (args.docs / "corpus-summary.json").read_text(encoding="utf-8") \
        if (args.docs / "corpus-summary.json").exists() else ""
    terms = private_terms(args.cluster_env, args.manifest, public)
    files = [md_path, html_path] + sorted((args.docs / "plots").glob("*.svg"))
    findings = leak_check(files, terms)
    if findings:
        for name, label in findings:
            print(f"LEAK {name}: {label}", file=sys.stderr)
        return 2
    print(f"leak check: clean ({len(files)} files, {len(GENERIC_PATTERNS)} patterns, {len(terms)} private terms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
