#!/usr/bin/env python3
"""Fidelity campaign figures (docs/fidelity/PLAN.md, owner brief section 10).

Reads the public aggregates in docs/fidelity/metrics-v2/, optional task and voxel
results, the private determinism probes in data/fidelity/prelim/ and, for the
per-position figures, the raw prompt-logprob runs in data/fidelity/raw/ (the private
metrics-v2 arrays hold per-window sums only, so per-position coarse KL is recomputed
with metrics.np_coarse_kl). Writes PNG and SVG pairs to docs/fidelity/plots/ and a
plots.json index (titles, captions, status and notes) for build_report.py.

Every figure renders with partial data: a missing arm is skipped with a note and a
missing input produces a labelled placeholder. No token ids, text, window ids,
hostnames or paths appear in any output.

Requires numpy and matplotlib==3.11.2 (data/fidelity/.venv/bin/python).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fidelity_io as fio  # noqa: E402
import metrics as m  # noqa: E402

import numpy as np  # noqa: E402

REPO = fio.REPO
DENSE_MAX = 2048

# Repository plot style (scripts/plot-baseline-comparison.py) plus two arm hues,
# validated as a categorical set (all pairs pass CVD and normal-vision separation).
TEAL, INK, MUTED, GRID = "#087f74", "#172b46", "#536478", "#dfe5ed"
FLOOR_COLOR = "#92a5ba"
ORANGE, VIOLET = "#c2410c", "#6b4fc9"
EXTRA = ("#2a6fbd", "#b8327a")

REF_RUN = "R0/prompt-a"
FLOOR_RUN = "R0/prompt-b"
# (label, candidate run, public comparison name, colour); the reference is R0 run A.
ARMS = [("Cpre", "Cpre/prompt-a", "cpre-vs-r0", VIOLET),
        ("Cm", "Cm/prompt-a", "cm-vs-r0", TEAL),
        ("N", "N/prompt-a", "n-vs-r0", ORANGE)]
FLOOR = ("R0 B vs A (floor)", FLOOR_RUN, "floor-r0", FLOOR_COLOR)
GEN_PAIRS = [("R0 repeat (floor)", "R0/gen-floor", FLOOR_COLOR), ("Cpre", "Cpre/gen-a", VIOLET),
             ("Cm", "Cm/gen-a", TEAL), ("N", "N/gen-a", ORANGE)]
GEN_REF = "R0/gen-a"
REGIME_LABEL = {"dense": "Dense regime (≤ 2,048 conditioning tokens)",
                "sparse": "Sparse regime (> 2,048 conditioning tokens)", "all": "All positions"}
SUBTITLE = "GLM-5.3-Flash · four GB10 nodes · coarse top-K KL is a lower bound on the full-vocabulary KL"
KL_FLOOR = 1e-6


# ---------------------------------------------------------------- style helpers


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "path"})
    import matplotlib.pyplot as plt
    return plt


def style(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, loc="left", fontsize=13, color=INK, fontweight="bold", pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10.5, color=MUTED, labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10.5, color=MUTED, labelpad=8)
    ax.tick_params(axis="both", length=0, labelcolor=MUTED, pad=6, labelsize=10)
    ax.grid(color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)


def header(fig, title, subtitle=SUBTITLE):
    h = fig.get_figheight()
    fig.text(0.03, 1 - 0.22 / h, title, fontsize=18, fontweight="bold", color=INK, va="top")
    fig.text(0.03, 1 - 0.62 / h, subtitle, fontsize=10.5, color=MUTED, va="top")


def footer(fig, notes, limit=5):
    """Notes stacked upwards from the bottom edge, 0.22 inch apart."""
    step = 0.22 / fig.get_figheight()
    shown = notes[:limit]
    for i, note in enumerate(shown):
        fig.text(0.03, 0.015 + step * (len(shown) - 1 - i), note, fontsize=9.5, color=MUTED)


def fig_legend(fig, items, y=0.875):
    """Figure-level legend under the subtitle; items are (label, colour)."""
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=colour, label=label) for label, colour in items], loc="upper left",
               bbox_to_anchor=(0.025, y), ncols=len(items), frameon=False, fontsize=10.5, labelcolor=INK,
               columnspacing=2.5)


def empty_panel(ax, text):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=11, color=MUTED, wrap=True,
            transform=ax.transAxes)


class Out:
    def __init__(self, out_dir: Path, plt):
        self.dir = out_dir
        self.plt = plt
        self.index = []

    def save(self, fig, name):
        self.dir.mkdir(parents=True, exist_ok=True)
        import matplotlib
        # A per-figure salt keeps SVG element ids distinct when figures are inlined together.
        matplotlib.rcParams["svg.hashsalt"] = f"fidelity-{name}"
        fig.savefig(self.dir / f"{name}.png", dpi=150, facecolor="white", metadata={"Software": "Matplotlib"})
        svg = self.dir / f"{name}.svg"
        fig.savefig(svg, facecolor="white", metadata={"Date": None, "Creator": "Matplotlib"})
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
        self.plt.close(fig)

    def placeholder(self, name, title, lines):
        height = 2.4 + 0.3 * len(lines[:5])
        fig = self.plt.figure(figsize=(14, height), facecolor="white")
        header(fig, title)
        fig.text(0.03, 1 - 1.3 / height, "Pending", fontsize=16, fontweight="bold", color=MUTED, va="top")
        for i, line in enumerate(lines[:5]):
            fig.text(0.03, 1 - (1.85 + 0.3 * i) / height, line, fontsize=11, color=MUTED, va="top")
        self.save(fig, name)


def pending(ctx, name, title, lines):
    ctx["out"].placeholder(name, title, lines)
    return {"status": "pending", "notes": list(lines)}


# ---------------------------------------------------------------- inputs


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def metric(metrics_dir: Path, name: str):
    data = read_json(metrics_dir / f"{name}.json")
    if not data or data.get("status") not in ("ok", "partial"):
        return None
    return data


def load_prompt(run_dir: Path, wid: str):
    sidecar = read_json(run_dir / "prompt" / f"{wid}.json")
    path = run_dir / "prompt" / f"{wid}.npz"
    if not sidecar or sidecar.get("status") != "ok" or not path.exists():
        return None
    try:
        with np.load(path) as data:
            arrays = {k: data[k] for k in ("ids", "lp_actual", "topk_ids", "topk_lp")}
    except Exception:
        return None
    if arrays["topk_ids"].shape != arrays["topk_lp"].shape or arrays["topk_ids"].shape[0] != arrays["ids"].shape[0]:
        return None
    return arrays


def per_position(raw_root: Path, ref: str, cands: dict, stride: int = 1) -> dict:
    """Per-position coarse KL, top-1 and delta NLL of each candidate run against `ref`.

    Returns {label: {"kl", "top1", "dnll", "pos", "ntok", "windows"}} for candidates with
    at least one scored window. Rows are subsampled deterministically (every `stride`-th
    position) when stride > 1. Window ids stay in memory only.
    """
    ref_dir = raw_root / ref
    out = {label: {"kl": [], "top1": [], "dnll": [], "pos": [], "ntok": [], "windows": 0} for label in cands}
    wids = sorted(p.stem for p in (ref_dir / "prompt").glob("*.npz")) if (ref_dir / "prompt").is_dir() else []
    started = time.time()
    for wid in wids:
        r = load_prompt(ref_dir, wid)
        if r is None:
            continue
        n = r["ids"].shape[0]
        rows = np.arange(1, n)
        if stride > 1:
            rows = rows[rows % stride == 0]
        if rows.size == 0:
            continue
        rid, rlp, ra = r["topk_ids"][rows], r["topk_lp"][rows], r["lp_actual"][rows]
        eid = r["ids"][rows].astype(np.int64)
        for label, run in cands.items():
            c = load_prompt(raw_root / run, wid)
            if c is None or c["ids"].shape[0] != n or not np.array_equal(c["ids"], r["ids"]):
                continue
            cid, clp, ca = c["topk_ids"][rows], c["topk_lp"][rows], c["lp_actual"][rows]
            kl, valid, _, _ = m.np_coarse_kl(rid, rlp, cid, clp, eid, ra, ca)
            top1 = m.np_top1(rid, rlp, cid, clp)
            ok = valid & (top1 >= 0)
            ra64, ca64 = ra.astype(np.float64), ca.astype(np.float64)
            dst = out[label]
            dst["kl"].append(np.where(ok, kl, np.nan).astype(np.float32))
            dst["top1"].append(np.where(ok, top1, -1).astype(np.int8))
            dst["dnll"].append(np.where(np.isfinite(ra64) & np.isfinite(ca64), ra64 - ca64, np.nan).astype(np.float32))
            dst["pos"].append(rows.astype(np.int32))
            dst["ntok"].append(np.full(rows.size, n, np.int32))
            dst["windows"] += 1
    result = {}
    for label, dst in out.items():
        if not dst["windows"]:
            continue
        result[label] = {k: (np.concatenate(v) if isinstance(v, list) else v) for k, v in dst.items()}
        print(f"per-position {label}: windows={dst['windows']} positions={result[label]['kl'].size}", flush=True)
    print(f"per-position data in {time.time() - started:.1f}s", flush=True)
    return result


def regime_sel(d, regime):
    finite = np.isfinite(d["kl"])
    if regime == "dense":
        return finite & (d["pos"] <= DENSE_MAX)
    if regime == "sparse":
        return finite & (d["pos"] > DENSE_MAX)
    return finite


def prefill_path(d):
    """1 = BF16 (chunk of >= 2,048 rows), 0 = Marlin, with 8,192-token chunks from position 0."""
    start = (d["pos"] // m.BATCHED_TOKENS) * m.BATCHED_TOKENS
    return (np.minimum(m.BATCHED_TOKENS, d["ntok"] - start) >= m.MARLIN_BELOW_ROWS).astype(np.int8)


def series(pp):
    """Ordered (label, data, colour, is_floor) for the per-position comparisons present."""
    out = []
    if FLOOR[0] in pp:
        out.append((FLOOR[0], pp[FLOOR[0]], FLOOR_COLOR, True))
    for label, _, _, colour in ARMS:
        if label in pp:
            out.append((f"{label} vs R0", pp[label], colour, False))
    return out


def missing_note(pp, arms=ARMS):
    missing = [label for label, *_ in arms if label not in pp]
    return [f"Not yet measured (skipped): {', '.join(missing)}."] if missing else []


def ci_err(ci, scale=1.0):
    """(estimate, [[lower err], [upper err]]) from a metrics-v2 CI dict, or (None, None)."""
    if not ci or ci.get("estimate") is None:
        return None, None
    est = ci["estimate"] * scale
    lo = ci.get("ci_low")
    hi = ci.get("ci_high")
    if lo is None or hi is None:
        return est, None
    return est, [[max(0.0, est - lo * scale)], [max(0.0, hi * scale - est)]]


# ---------------------------------------------------------------- figures


def fig_cdf(ctx):
    name, title = ctx["fig"], "Per-position coarse KL: cumulative distribution"
    pp = ctx["pp"]
    rows = series(pp)
    if not any(not f for *_, f in rows):
        return pending(ctx, name, title, ["No candidate arm (Cpre, Cm, N) has been scored against R0 yet."])
    plt = ctx["plt"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4), facecolor="white")
    fig.subplots_adjust(left=0.06, right=0.98, top=0.78, bottom=0.26, wspace=0.18)
    header(fig, title)
    x = np.logspace(-6, 2, 400)
    for ax, regime in zip(axes, ("dense", "sparse")):
        for label, d, colour, is_floor in rows:
            v = np.sort(d["kl"][regime_sel(d, regime)].astype(np.float64))
            if v.size == 0:
                continue
            y = np.searchsorted(v, x, side="right") / v.size
            ax.plot(x, y, color=colour, lw=2, ls="--" if is_floor else "-",
                    label=f"{label} · {d['windows']} windows")
        ax.set_xscale("log")
        ax.set_ylim(0, 1.02)
        style(ax, REGIME_LABEL[regime], "Coarse KL(R0 ‖ arm), nats (log scale)", "Fraction of positions ≤ x")
        ax.legend(frameon=False, fontsize=9.5, loc="lower right", labelcolor=INK)
    notes = ["Values below 10⁻⁶ nats, including exact zeros, are counted at the left edge. "
             "The dense R0 floor is bit-identical (KL = 0 everywhere)."] + missing_note(pp)
    footer(fig, notes)
    ctx["out"].save(fig, name)
    return {"status": "ok" if not missing_note(pp) else "partial", "notes": notes}


def fig_position(ctx):
    name, title = ctx["fig"], "Coarse KL along the context: median and p99 per position bin"
    pp = ctx["pp"]
    arms = [(label, colour) for label, _, _, colour in ARMS]
    if not any(label in pp for label, _ in arms):
        return pending(ctx, name, title, ["No candidate arm (Cpre, Cm, N) has been scored against R0 yet."])
    plt = ctx["plt"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.2), facecolor="white")
    fig.subplots_adjust(left=0.06, right=0.98, top=0.78, bottom=0.26, wspace=0.12)
    header(fig, title)
    edges = np.unique(np.round(2.0 ** np.arange(0, 17.01, 0.25)).astype(np.int64))
    floor = pp.get(FLOOR[0])

    def binned(d):
        sel = np.isfinite(d["kl"])
        pos, kl = d["pos"][sel], d["kl"][sel].astype(np.float64)
        idx = np.digitize(pos, edges) - 1
        centres, med, p99 = [], [], []
        for b in range(len(edges) - 1):
            v = kl[idx == b]
            if v.size < 50:
                continue
            centres.append(math.sqrt(edges[b] * edges[b + 1]))
            med.append(max(np.median(v), KL_FLOOR))
            p99.append(max(np.percentile(v, 99), KL_FLOOR))
        return np.asarray(centres), np.asarray(med), np.asarray(p99)

    fb = binned(floor) if floor is not None else None
    for ax, (label, colour) in zip(axes, arms):
        if label not in pp:
            empty_panel(ax, f"{label}: not yet measured")
            ax.set_title(f"{label} vs R0", loc="left", fontsize=13, color=INK, fontweight="bold")
            continue
        c, med, p99 = binned(pp[label])
        ax.fill_between(c, med, p99, color=colour, alpha=0.12, lw=0)
        ax.plot(c, med, color=colour, lw=2, label="median")
        ax.plot(c, p99, color=colour, lw=2, ls=(0, (4, 2)), label="p99")
        if fb is not None:
            ax.plot(fb[0], fb[1], color=FLOOR_COLOR, lw=1.6, label="R0 floor median")
            ax.plot(fb[0], fb[2], color=FLOOR_COLOR, lw=1.6, ls=(0, (4, 2)), label="R0 floor p99")
        ax.axvline(DENSE_MAX, color=MUTED, lw=1, ls=":")
        ax.text(DENSE_MAX * 1.08, 0.97, "2,048", transform=ax.get_xaxis_transform(), fontsize=9, color=MUTED,
                va="top")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(KL_FLOOR / 2, 100)
        style(ax, f"{label} vs R0 · {pp[label]['windows']} windows", "Context position (tokens, log scale)",
              "Coarse KL, nats (log scale)" if ax is axes[0] else None)
        ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK)
    notes = [f"Bins of a quarter octave; bins with fewer than 50 positions are omitted; values below "
             f"{KL_FLOOR:g} nats (including zeros) are drawn at {KL_FLOOR:g}.",
             "Positions beyond 32K come from five long-context windows only."] + missing_note(pp)
    footer(fig, notes)
    ctx["out"].save(fig, name)
    return {"status": "ok" if not missing_note(pp) else "partial", "notes": notes}


def fig_tail(ctx):
    name, title = ctx["fig"], "Tail of the per-position KL: P(KL > x)"
    pp = ctx["pp"]
    rows = series(pp)
    if not any(not f for *_, f in rows):
        return pending(ctx, name, title, ["No candidate arm (Cpre, Cm, N) has been scored against R0 yet."])
    plt = ctx["plt"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4), facecolor="white")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.26, wspace=0.16)
    header(fig, title)
    x = np.logspace(-4, 2, 300)
    for ax, regime in zip(axes, ("dense", "sparse")):
        for label, d, colour, is_floor in rows:
            v = np.sort(d["kl"][regime_sel(d, regime)].astype(np.float64))
            if v.size == 0:
                continue
            surv = 1.0 - np.searchsorted(v, x, side="right") / v.size
            keep = surv > 0
            if not keep.any():
                ax.plot([], [], color=colour, lw=2, ls="--" if is_floor else "-", label=f"{label} · none above 10⁻⁴")
                continue
            ax.plot(x[keep], surv[keep], color=colour, lw=2, ls="--" if is_floor else "-", label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-7, 1.5)
        style(ax, REGIME_LABEL[regime], "Coarse KL threshold x, nats (log scale)",
              "Fraction of positions with KL > x" if regime == "dense" else None)
        ax.legend(frameon=False, fontsize=9.5, loc="lower left", labelcolor=INK)
    notes = ["Curves end where no position exceeds x. The floor is R0 run B against run A on the same boot."] \
        + missing_note(pp)
    footer(fig, notes)
    ctx["out"].save(fig, name)
    return {"status": "ok" if not missing_note(pp) else "partial", "notes": notes}


def json_comps(ctx, with_floor=True):
    """Ordered (label, result, block-key, colour) for public pair comparisons present."""
    out = []
    if with_floor and ctx["m"].get("floor-r0"):
        out.append(("R0 B vs A (floor)", ctx["m"]["floor-r0"], FLOOR_COLOR))
    for label, _, comp, colour in ARMS:
        if ctx["m"].get(comp):
            out.append((f"{label} vs R0", ctx["m"][comp], colour))
    return out


def json_missing(ctx):
    missing = [label for label, _, comp, _ in ARMS if not ctx["m"].get(comp)]
    return [f"Not yet measured (skipped): {', '.join(missing)}."] if missing else []


def partial_notes(comps):
    notes = []
    for label, res, _ in comps:
        w = res.get("windows") or {}
        if res.get("status") == "partial":
            notes.append(f"{label}: partial, {w.get('compared')} of {w.get('expected')} windows.")
    return notes


def hbar_groups(ax, cats, comps, value_of, scale=1.0):
    """Grouped horizontal bars with CI whiskers; value_of(result, cat) -> CI dict or None."""
    n = max(1, len(comps))
    height = 0.8 / n
    for j, (label, res, colour) in enumerate(comps):
        for i, cat in enumerate(cats):
            est, err = ci_err(value_of(res, cat), scale)
            if est is None:
                continue
            y = i - 0.4 + height * (j + 0.5)
            ax.barh(y, est, height=height * 0.85, color=colour, zorder=2)
            if err is not None:
                ax.errorbar(est, y, xerr=err, fmt="none", ecolor=INK, elinewidth=1, capsize=2, zorder=3)
    ax.set_yticks(range(len(cats)), [c.replace("_", " ") for c in cats], fontsize=10, color=INK)
    ax.set_ylim(len(cats) - 0.5, -0.5)


def fig_category(ctx):
    name, title = ctx["fig"], "Mean coarse KL and top-1 agreement by corpus category (95% CIs)"
    comps = json_comps(ctx)
    if not any(c[1].get("name") != "floor-r0" for c in comps):
        return pending(ctx, name, title, ["No candidate comparison in docs/fidelity/metrics-v2/ yet."])
    plt = ctx["plt"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10.5), facecolor="white")
    fig.subplots_adjust(left=0.12, right=0.98, top=0.83, bottom=0.17, wspace=0.12, hspace=0.35)
    header(fig, title)
    fig_legend(fig, [(label, colour) for label, _, colour in comps], y=0.89)
    cats = sorted({c for _, res, _ in comps for r in ("dense", "sparse")
                   for c in (res.get("regimes", {}).get(r, {}).get("by_category") or {})})
    for row, regime in enumerate(("dense", "sparse")):
        def kl_of(res, cat, regime=regime):
            blk = (res.get("regimes", {}).get(regime, {}).get("by_category") or {}).get(cat) or {}
            return blk.get("contrast", {}).get("kl", {}).get("mean_ci")

        def t1_of(res, cat, regime=regime):
            blk = (res.get("regimes", {}).get(regime, {}).get("by_category") or {}).get(cat) or {}
            return blk.get("contrast", {}).get("top1_agreement")

        hbar_groups(axes[row][0], cats, comps, kl_of)
        short = regime.capitalize()
        style(axes[row][0], f"{short} regime · mean KL", "Mean coarse KL, nats")
        hbar_groups(axes[row][1], cats, comps, t1_of, 100.0)
        style(axes[row][1], f"{short} regime · top-1 agreement", "Top-1 agreement with R0, %")
        axes[row][1].set_xlim(0, 100)
        axes[row][1].set_yticklabels([])
    notes = ["Dense ≤ 2,048 conditioning tokens, sparse > 2,048. The dense R0 floor is exactly zero (no bar). "
             "Whiskers: 95% percentile bootstrap over source groups."] \
        + partial_notes(comps) + json_missing(ctx)
    footer(fig, notes)
    ctx["out"].save(fig, name)
    return {"status": "partial" if json_missing(ctx) or partial_notes(comps) else "ok", "notes": notes}


def fig_path(ctx):
    name, title = ctx["fig"], "Coarse KL by prefill path: Marlin W8A16 chunks vs dequantised BF16 chunks"
    comps = json_comps(ctx)
    pp = ctx["pp"]
    if not any(c[1].get("name") != "floor-r0" for c in comps):
        return pending(ctx, name, title, ["No candidate comparison in docs/fidelity/metrics-v2/ yet."])
    plt = ctx["plt"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.4), facecolor="white")
    fig.subplots_adjust(left=0.1, right=0.98, top=0.76, bottom=0.26, wspace=0.3)
    header(fig, title)
    fig_legend(fig, [(label, colour) for label, _, colour in comps])
    counts = ""
    paths = ["marlin", "bf16"]

    def path_of(res, path):
        return ((res.get("by_prefill_path") or {}).get(path) or {}).get("contrast", {}).get("kl", {}).get("mean_ci")

    hbar_groups(axes[0], paths, comps, path_of)
    axes[0].set_yticklabels(["Marlin chunks\n(< 2,048 rows)", "BF16 chunks\n(≥ 2,048 rows)"])
    style(axes[0], "All positions · 95% CIs", "Mean coarse KL, nats")
    cells = [("dense", 0), ("dense", 1), ("sparse", 0), ("sparse", 1)]
    labels = ["dense · Marlin", "dense · BF16", "sparse · Marlin", "sparse · BF16"]
    rows = series(pp)
    if rows:
        height = 0.8 / len(rows)
        for j, (label, d, colour, _) in enumerate(rows):
            path = prefill_path(d)
            for i, (regime, p) in enumerate(cells):
                sel = regime_sel(d, regime) & (path == p)
                if not sel.any():
                    continue
                y = i - 0.4 + height * (j + 0.5)
                axes[1].barh(y, float(np.mean(d["kl"][sel])), height=height * 0.85, color=colour, zorder=2)
        d0 = rows[-1][1]
        path0 = prefill_path(d0)
        counts = ", ".join(f"{lab} {int((regime_sel(d0, r) & (path0 == p)).sum()):,}"
                           for lab, (r, p) in zip(labels, cells))
        axes[1].set_yticks(range(len(cells)), labels, fontsize=10, color=INK)
        axes[1].set_ylim(len(cells) - 0.5, -0.5)
        style(axes[1], "Split by regime · point estimates", "Mean coarse KL, nats")
    else:
        empty_panel(axes[1], "Per-position data not available")
    notes = ["Path tag assumes 8,192-token prefill chunks from position 0 at concurrency 1; a chunk shorter than "
             "2,048 rows took the Marlin W8A16 path.",
             "Marlin chunks are the last chunk of a window, so the path split is confounded with position and window "
             "length; compare within a regime."] + ([f"Positions per cell: {counts}."] if rows else []) \
        + partial_notes(comps) + json_missing(ctx)
    footer(fig, notes)
    ctx["out"].save(fig, name)
    return {"status": "partial" if json_missing(ctx) or partial_notes(comps) else "ok", "notes": notes}


def load_gen(run_dir: Path):
    out = {}
    gdir = run_dir / "gen"
    if not gdir.is_dir():
        return out
    for path in sorted(gdir.glob("*.npz")):
        sidecar = read_json(path.with_suffix(".json"))
        if not sidecar or sidecar.get("status") != "ok":
            continue
        try:
            with np.load(path) as data:
                out[path.stem] = {k: data[k] for k in ("gen_ids", "topk_ids", "topk_lp")}
        except Exception:
            continue
    return out


def fig_decode(ctx):
    name, title = ctx["fig"], "Decode path: KL along the shared greedy prefix and survival of identical prefixes"
    raw = ctx["raw"]
    ref = load_gen(raw / GEN_REF)
    present = []
    if ref:
        for label, run, colour in GEN_PAIRS:
            cand = load_gen(raw / run)
            if cand:
                present.append((label, cand, colour))
    if not present:
        return pending(ctx, name, title, [
            "Greedy decode-set generations (R0 reference and at least one arm) have not been collected yet.",
            "Expected runs: R0/gen-a (reference), R0/gen-floor, Cm/gen-a, Cpre/gen-a, N/gen-a."])
    plt = ctx["plt"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.4), facecolor="white")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.26, wspace=0.22)
    header(fig, title)
    edges = np.unique(np.round(2.0 ** np.arange(0, 11.01, 0.5)).astype(np.int64)) - 1
    notes = []
    for label, cand, colour in present:
        common = sorted(set(ref) & set(cand))
        times, events, kl_pos, kl_val = [], [], [], []
        for pid in common:
            r, c = ref[pid], cand[pid]
            length, div = m.generation_prefix(r["gen_ids"].tolist(), c["gen_ids"].tolist())
            if length:
                kl, valid, _, _ = m.np_coarse_kl(r["topk_ids"][:length], r["topk_lp"][:length],
                                                 c["topk_ids"][:length], c["topk_lp"][:length])
                kl_pos.append(np.arange(length)[valid])
                kl_val.append(kl[valid])
            times.append(div if div is not None else length)
            events.append(div is not None)
        if kl_pos:
            pos, val = np.concatenate(kl_pos), np.concatenate(kl_val)
            idx = np.digitize(pos, edges) - 1
            cx, cy = [], []
            for b in range(len(edges) - 1):
                sel = idx == b
                if sel.sum() >= 5:
                    cx.append(edges[b] + 1)
                    cy.append(max(float(val[sel].mean()), KL_FLOOR))
            axes[0].plot(cx, cy, color=colour, lw=2, marker="o", ms=4, label=label)
        # Kaplan-Meier survival of the identical prefix; sequences that end identically are censored.
        t = np.asarray(times)
        e = np.asarray(events)
        grid = np.unique(np.concatenate([[0], t[e]]))
        surv, s = [], 1.0
        for g in grid:
            at_risk = int((t >= g).sum())
            d = int(((t == g) & e).sum())
            if at_risk:
                s *= 1.0 - d / at_risk
            surv.append(s)
        axes[1].step(np.maximum(grid, 1), surv, where="post", color=colour, lw=2,
                     label=f"{label} · {len(common)} prompts, {int(e.sum())} diverged")
        notes.append(f"{label}: {len(common)} paired prompts, {int((~e).sum())} censored (identical to the end).")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    style(axes[0], "Mean coarse KL at generated position (identical prefix)", "Generated position (log scale)",
          "Mean coarse KL, nats")
    axes[0].legend(frameon=False, fontsize=9.5, labelcolor=INK)
    axes[1].set_xscale("log")
    axes[1].set_ylim(0, 1.02)
    style(axes[1], "Survival of the identical greedy prefix", "Generated tokens (log scale)",
          "Fraction of prompts still identical")
    axes[1].legend(frameon=False, fontsize=9.5, labelcolor=INK)
    missing = [label for label, *_ in GEN_PAIRS if label not in {p[0] for p in present}]
    if missing:
        notes.append(f"Not yet collected (skipped): {', '.join(missing)}.")
    footer(fig, ["Positions up to and including the first divergence are compared; beyond it the contexts differ."]
           + notes[:3])
    ctx["out"].save(fig, name)
    return {"status": "partial" if missing else "ok", "notes": notes}


def task_files(docs: Path):
    out = []
    for path in sorted((docs / "metrics").glob("tasks-*.json")):
        data = read_json(path)
        if isinstance(data, dict):
            out.append(data)
    return out


def fig_tasks(ctx):
    name, title = ctx["fig"], "Task pass rate per arm (Wilson 95% CIs) and paired discordance"
    qeval = [t for t in task_files(ctx["docs"]) if t.get("set") == "qeval" and t.get("wilson_ci")]
    if not qeval:
        return pending(ctx, name, title, ["No task results in docs/fidelity/metrics/tasks-*.json yet.",
                                             "Tasks run on the serving boots after the measurement boots."])
    plt = ctx["plt"]
    rates = {}
    for t in qeval:
        for arm, w in (t.get("wilson_ci") or {}).items():
            if w and arm not in rates:
                rates[arm] = w
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.4), facecolor="white", gridspec_kw={"width_ratios": [1, 1.2]})
    fig.subplots_adjust(left=0.08, right=0.98, top=0.78, bottom=0.26, wspace=0.15)
    header(fig, title)
    arms = list(rates)
    colours = {"R0": FLOOR_COLOR, "Cp": TEAL, "Cm": TEAL, "Cpre": VIOLET, "N": ORANGE}
    for i, arm in enumerate(arms):
        w = rates[arm]
        est = w["rate"] * 100
        axes[0].barh(i, est, height=0.6, color=colours.get(arm, EXTRA[i % 2]), zorder=2)
        axes[0].errorbar(est, i, xerr=[[est - w["lo95"] * 100], [w["hi95"] * 100 - est]], fmt="none",
                         ecolor=INK, capsize=3, zorder=3)
        axes[0].annotate(f"{est:.1f}%", (est, i), xytext=(6, 8), textcoords="offset points", fontsize=9.5, color=INK)
    axes[0].set_yticks(range(len(arms)), arms, fontsize=11, color=INK)
    axes[0].set_ylim(len(arms) - 0.5, -0.5)
    axes[0].set_xlim(0, 100)
    style(axes[0], "qeval greedy pass rate", "Pass rate, %")
    ax = axes[1]
    ax.axis("off")
    ax.set_title("Paired greedy discordance (exact McNemar)", loc="left", fontsize=13, color=INK, fontweight="bold")
    cells = []
    for t in qeval:
        d = t.get("greedy_discordance") or {}
        a, b = d.get("arm_a"), d.get("arm_b")
        diff = (t.get("sampled_bootstrap_diff") or {})
        ci = diff.get("ci95") if diff.get("diff") is not None else None
        cells.append([f"{a} vs {b}", str(d.get("n_paired")), str(d.get("both_pass")), str(d.get("both_fail")),
                      str(d.get(f"{a}_only_pass")), str(d.get(f"{b}_only_pass")),
                      f"{t.get('mcnemar_exact_p'):.3g}" if t.get("mcnemar_exact_p") is not None else "–",
                      f"{diff['diff'] * 100:+.1f} [{ci[0] * 100:+.1f}, {ci[1] * 100:+.1f}]" if ci else "pending"])
    table = ax.table(cellText=cells, colLabels=["pair", "n", "both\npass", "both\nfail", "A only\npass",
                                                "B only\npass", "McNemar\np", "sampled Δ pp\n[95% CI]"],
                     loc="upper left", cellLoc="center", colWidths=[0.17, 0.07, 0.09, 0.09, 0.1, 0.1, 0.12, 0.26])
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 2.2)
    for cell in table.get_celld().values():
        cell.set_edgecolor(GRID)
        cell.get_text().set_color(INK)
    notes = ["Pre-registered: no significant R0 vs Cp difference and a point estimate within 2 pp; equivalence needs "
             "the paired 95% CI inside ±2 pp (amendment 13)."]
    footer(fig, notes)
    ctx["out"].save(fig, name)
    return {"status": "ok", "notes": notes}


def fig_ladder(ctx):
    name, title = ctx["fig"], "Ladder attribution in the dense regime: R0 → L0919 → Cpre → LE21 → LE22b → Cm"
    lad = read_json(ctx["metrics"] / "ladder.json")
    if not lad or not lad.get("chain"):
        return pending(ctx, name, title, ["No ladder analysis in docs/fidelity/metrics-v2/ yet."])
    regime = lad.get("regime", "dense")
    plt = ctx["plt"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.4), facecolor="white")
    fig.subplots_adjust(left=0.1, right=0.98, top=0.78, bottom=0.26, wspace=0.35)
    header(fig, title)
    any_ok = False
    for ax, key, heading in ((axes[0], "cumulative_from_first", "Cumulative: KL(R0 ‖ rung)"),
                             (axes[1], "steps", "Per step: KL(previous rung ‖ rung)")):
        entries = lad.get(key) or []
        labels = []
        for i, e in enumerate(entries):
            rung = e["cand"].split("/")[0]
            labels.append(rung if key == "cumulative_from_first" else f"{e['ref'].split('/')[0]} → {rung}")
            if e.get("status") not in ("ok", "partial"):
                ax.text(0, i, "  pending", va="center", fontsize=9.5, color=MUTED)
                continue
            any_ok = True
            ci = (e.get(regime) or {}).get("contrast", {}).get("kl", {}).get("mean_ci")
            est, err = ci_err(ci)
            if est is None:
                continue
            colour = FLOOR_COLOR if e.get("negative_control") else TEAL
            ax.barh(i, est, height=0.6, color=colour, zorder=2)
            if err is not None:
                ax.errorbar(est, i, xerr=err, fmt="none", ecolor=INK, capsize=3, zorder=3)
            ax.annotate(f"{est:.4f}", (est, i), xytext=(6, 8), textcoords="offset points", fontsize=9, color=INK)
        ax.set_yticks(range(len(labels)), labels, fontsize=10, color=INK)
        ax.set_ylim(len(labels) - 0.5, -0.5)
        style(ax, heading, "Mean coarse KL, nats (95% CI)")
    notes = [f"Subset: {lad.get('subset', 'ladder subset')}; regime: {regime}. Grey bar = negative control "
             "(drafter, scheduler or KV pool only; must sit on the floor)."]
    if not any_ok:
        notes.append("No rung measured yet.")
    elif lad.get("status") != "ok":
        notes.append("Some rungs not yet measured (shown as pending).")
    footer(fig, notes)
    ctx["out"].save(fig, name)
    return {"status": "ok" if lad.get("status") == "ok" else ("partial" if any_ok else "pending"), "notes": notes}


def fig_k(ctx):
    name, title = ctx["fig"], "K sensitivity: Cm vs R0 at K = 100 and truncated to K = 20"
    k = read_json(ctx["metrics"] / "k-cm-vs-r0.json")
    if not k or k.get("status") not in ("ok", "partial"):
        return pending(ctx, name, title, ["The K = 100 subset has not been scored on both R0 and Cm yet"
                                             + (f" (missing: {', '.join(k.get('missing_runs', []))})." if k else ".")])
    plt = ctx["plt"]
    fig, ax = plt.subplots(1, 1, figsize=(12, 6), facecolor="white")
    fig.subplots_adjust(left=0.14, right=0.96, top=0.74, bottom=0.26)
    header(fig, title)
    fig_legend(fig, [(f"K = {k.get('K_low')} (same rows, truncated)", FLOOR_COLOR), ("K = 100", TEAL)])
    regimes = [r for r in ("dense", "sparse", "all") if "high_K" in (k.get("regimes", {}).get(r) or {})]
    for i, regime in enumerate(regimes):
        blk = k["regimes"][regime]
        for j, (key, colour, label) in enumerate((("truncated_low_K", FLOOR_COLOR, f"K = {k.get('K_low')} (truncated)"),
                                                  ("high_K", TEAL, "K = 100"))):
            est, err = ci_err(blk[key]["kl"].get("mean_ci"))
            if est is None:
                continue
            y = i - 0.2 + 0.4 * j
            ax.barh(y, est, height=0.36, color=colour, zorder=2)
            if err is not None:
                ax.errorbar(est, y, xerr=err, fmt="none", ecolor=INK, capsize=3, zorder=3)
            ax.annotate(f"{est:.4f}", (est + (err[1][0] if err else 0), y), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(regimes)), [REGIME_LABEL[r].split(" (")[0] for r in regimes], fontsize=10, color=INK)
    ax.set_ylim(len(regimes) - 0.5, -0.5)
    ax.set_xlim(right=ax.get_xlim()[1] * 1.12)
    style(ax, "Mean coarse KL on the same rows", "Mean coarse KL, nats (95% CI)")
    w = k.get("windows") or {}
    notes = [f"Same K = 100 rows truncated to K = {k.get('K_low')}; a larger K refines the partition, so the gap "
             "shows how much the lower bound tightens.", f"Windows compared: {w.get('compared')} of {w.get('expected')}."]
    footer(fig, notes)
    ctx["out"].save(fig, name)
    return {"status": k.get("status"), "notes": notes}


def fig_voxel(ctx):
    name, title = ctx["fig"], "Voxel showcase: greedy renders per arm (visual sanity check only)"
    checks = read_json(ctx["docs"] / "voxel" / "checks.json")
    entries = []
    for r in (checks or {}).get("results", []):
        parts = Path(r.get("html_path", "")).parts
        if "voxel" not in parts:
            continue
        tail = parts[parts.index("voxel") + 1:]
        if len(tail) != 3 or not r.get("screenshot"):
            continue
        shot = REPO / r["screenshot"]
        if not shot.exists() or not tail[2].startswith("greedy-run1"):
            continue
        c = r.get("constraints") or {}
        ok = sum(bool(c.get(k)) for k in ("voxel_count_ok", "pagoda_floors_ok", "reported_fps_ok")) if c else None
        entries.append((tail[0], tail[1], shot, len(r.get("console_errors") or []), ok))
    if not entries:
        return pending(ctx, name, title, ["No rendered voxel outputs in docs/fidelity/voxel/checks.json yet."])
    import matplotlib.image as mpimg
    plt = ctx["plt"]
    arms = sorted({e[0] for e in entries})
    prompts = sorted({e[1] for e in entries})
    fig, axes = plt.subplots(len(arms), len(prompts), figsize=(6 * len(prompts), 4 * len(arms) + 1.6),
                             facecolor="white", squeeze=False)
    fig.subplots_adjust(left=0.03, right=0.98, top=1 - 1.5 / (4 * len(arms) + 1.6), bottom=0.04, hspace=0.3)
    header(fig, title)
    lookup = {(e[0], e[1]): e for e in entries}
    for i, arm in enumerate(arms):
        for j, prompt in enumerate(prompts):
            ax = axes[i][j]
            ax.axis("off")
            e = lookup.get((arm, prompt))
            if e is None:
                ax.text(0.5, 0.5, "not rendered", ha="center", va="center", color=MUTED, transform=ax.transAxes)
                continue
            ax.imshow(mpimg.imread(e[2]))
            checks_txt = f" · constraints {e[4]}/3" if e[4] is not None else ""
            ax.set_title(f"{arm} · {prompt} · console errors {e[3]}{checks_txt}", loc="left", fontsize=10, color=INK)
    ctx["out"].save(fig, name)
    return {"status": "ok", "notes": ["Frame rates are relative (software WebGL); no precision conclusion is drawn."]}


def fig_repeatability(ctx):
    name, title = ctx["fig"], "Repeatability: each arm against itself, and the 2,048-token onset"
    floor = ctx["m"].get("floor-r0")
    probes = {}
    for arm, fname in (("R0", "determinism-r0.json"), ("Cm", "determinism-cm.json"), ("N", "determinism-n.json")):
        data = read_json(ctx["prelim"] / fname)
        if data and data.get("pairs"):
            probes[arm] = data["pairs"]
    if not floor and not probes:
        return pending(ctx, name, title, ["Neither the R0 floor nor any determinism probe is available yet."])
    plt = ctx["plt"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.8), facecolor="white")
    fig.subplots_adjust(left=0.05, right=0.98, top=0.8, bottom=0.27, wspace=0.3)
    header(fig, title)
    notes = []
    if floor:
        buckets = [b for b in m.BUCKET_LABELS if b in (floor.get("by_position_bucket") or {})]
        for i, b in enumerate(buckets):
            c = floor["by_position_bucket"][b]["contrast"]
            est, err = ci_err(c["kl"].get("mean_ci"))
            axes[0].bar(i, est, width=0.6, color=FLOOR_COLOR, zorder=2)
            if err is not None:
                axes[0].errorbar(i, est, yerr=err, fmt="none", ecolor=INK, capsize=3, zorder=3)
            axes[0].annotate(f"{est:.3f}", (i, est), xytext=(0, 5), textcoords="offset points", ha="center",
                             fontsize=9, color=INK)
            est, err = ci_err(c["top1_agreement"], 100.0)
            axes[1].bar(i, est, width=0.6, color=FLOOR_COLOR, zorder=2)
            if err is not None:
                axes[1].errorbar(i, est, yerr=err, fmt="none", ecolor=INK, capsize=3, zorder=3)
            axes[1].annotate(f"{est:.1f}%", (i, est), xytext=(0, 5), textcoords="offset points", ha="center",
                             fontsize=9, color=INK)
        for ax in axes[:2]:
            ax.set_xticks(range(len(buckets)), buckets, fontsize=10, color=INK)
        style(axes[0], "R0 run B vs run A · mean KL", "Context position bucket", "Mean coarse KL, nats (95% CI)")
        style(axes[1], "R0 run B vs run A · top-1 agreement", "Context position bucket", "Top-1 agreement, %")
        axes[1].set_ylim(0, 105)
    else:
        for ax in axes[:2]:
            empty_panel(ax, "R0 floor comparison not available")
    ax = axes[2]
    colours = {"R0": FLOOR_COLOR, "Cm": TEAL, "N": ORANGE}
    labels = []
    for i, arm in enumerate(("R0", "Cm", "N")):
        pairs = probes.get(arm)
        labels.append(arm)
        if not pairs:
            ax.text(1.5, i, "  probe pending", va="center", fontsize=9.5, color=MUTED)
            continue
        first = [p["first_nonidentical_pos"] for p in pairs if p.get("first_nonidentical_pos")]
        jitter = np.linspace(-0.2, 0.2, len(first)) if len(first) > 1 else [0.0]
        ax.scatter(first, i + np.asarray(jitter), s=40, color=colours[arm], edgecolor="white", linewidth=1, zorder=3)
        t1 = np.mean([p["top1_agreement"] for p in pairs]) * 100
        ax.text(1.2, i - 0.3, f"median {int(np.median(first)):,} · whole-window top-1 {t1:.1f}%", va="center",
                fontsize=9, color=INK)
        notes.append(f"{arm} probe: {len(pairs)} same-window pass pairs, first non-identical position "
                     f"{min(first):,}–{max(first):,}.")
    ax.axvline(DENSE_MAX, color=MUTED, lw=1, ls=":")
    ax.text(DENSE_MAX * 1.08, 0.02, "2,048", transform=ax.get_xaxis_transform(), fontsize=9, color=MUTED)
    ax.set_xscale("log")
    ax.set_xlim(1, 20000)
    ax.set_yticks(range(len(labels)), labels, fontsize=11, color=INK)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    style(ax, "Determinism probe · first non-identical position", "Position (log scale)")
    footer(fig, ["Probe: two windows scored X X X Y X Y Y with fresh cache salts; each dot is one pair of passes of "
                 "the same window. N uses a different engine build."] + notes[:3])
    ctx["out"].save(fig, name)
    return {"status": "ok" if floor and len(probes) == 3 else "partial", "notes": notes}


def fig_precision(ctx):
    name, title = ctx["fig"], "Precision vs R0: mean KL and ΔNLL per regime (95% CIs)"
    comps = json_comps(ctx)
    if not any(c[1].get("name") != "floor-r0" for c in comps):
        return pending(ctx, name, title, ["No candidate comparison in docs/fidelity/metrics-v2/ yet."])
    plt = ctx["plt"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.4), facecolor="white")
    fig.subplots_adjust(left=0.14, right=0.98, top=0.76, bottom=0.22, wspace=0.12)
    header(fig, title)
    fig_legend(fig, [(label, colour) for label, _, colour in comps])
    regimes = ["dense", "sparse", "all"]
    n = len(comps)
    for ax, key, xlabel in ((axes[0], "kl", "Mean coarse KL(R0 ‖ arm), nats"),
                            (axes[1], "delta_nll", "Mean ΔNLL of the actual token (arm − R0), nats")):
        for j, (label, res, colour) in enumerate(comps):
            for i, regime in enumerate(regimes):
                c = (res.get("regimes", {}).get(regime) or {}).get("contrast") or {}
                ci = c.get("kl", {}).get("mean_ci") if key == "kl" else c.get("delta_nll")
                est, err = ci_err(ci)
                if est is None:
                    continue
                y = i - 0.3 + 0.6 * (j + 0.5) / n
                if err is not None:
                    lo, hi = est - err[0][0], est + err[1][0]
                    ax.plot([lo, hi], [y, y], color=colour, lw=2, zorder=2)
                ax.scatter([est], [y], s=60, color=colour, edgecolor="white", linewidth=1.5, zorder=3)
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_yticks(range(len(regimes)), [REGIME_LABEL[r].split(" (")[0] for r in regimes], fontsize=11, color=INK)
        ax.set_ylim(len(regimes) - 0.5, -0.5)
        style(ax, None, xlabel)
    axes[1].set_yticklabels([])
    notes = ["Sparse-regime values from a single execution measure operational disagreement, not fidelity "
             "(amendment 13); the sparse verdict needs three executions per arm."] + partial_notes(comps) \
        + json_missing(ctx)
    footer(fig, notes)
    ctx["out"].save(fig, name)
    return {"status": "partial" if json_missing(ctx) or partial_notes(comps) else "ok", "notes": notes}


# ---------------------------------------------------------------- quality figures (perplexity vs R0)

LADDER_ARMS = [("L0919", "#2a6fbd"), ("LE21", "#b8327a"), ("LE22b", "#8a6a12")]
RUNG_LABEL = {"L0919": "+ hybrid KDA (L0919)", "Cpre": "+ E03 mHC (Cpre)", "LE21": "+ E21 residual W8A16 (LE21)",
              "LE22b": "+ E22b drafter (LE22b)", "Cm": "+ E27–E29 (Cm = E29)"}
ARM_NAME = {"Cm": "Cm (E29)", "N": "N (NVFP4)", "Cpre": "Cpre (before E21)"}
QUALITY_NOTE = ("Perplexity change = exp(ΔNLL) − 1, where ΔNLL = NLL(arm) − NLL(R0) per real next token; "
                "lower is better, and a CI that straddles 0 means no detectable difference from R0.")


def ppl(ci):
    """(estimate %, [[lower err], [upper err]]) of exp(ΔNLL) − 1 from a ΔNLL CI dict, or (None, None)."""
    if not ci or ci.get("estimate") is None:
        return None, None
    f = lambda x: (math.exp(x) - 1.0) * 100.0  # noqa: E731
    est = f(ci["estimate"])
    if ci.get("ci_low") is None or ci.get("ci_high") is None:
        return est, None
    return est, [[max(0.0, est - f(ci["ci_low"]))], [max(0.0, f(ci["ci_high"]) - est)]]


def quality_comps(ctx):
    """(label, result, colour) for the arms against R0 with public pair results."""
    return [(label, ctx["m"][comp], colour) for label, _, comp, colour in ARMS if ctx["m"].get(comp)]


def ladder_json(ctx):
    lad = read_json(ctx["metrics"] / "ladder.json")
    return lad if isinstance(lad, dict) and lad.get("chain") else None


def dot_whisker(ax, y, est, err, colour, hollow=False, size=60):
    if err is not None:
        ax.plot([est - err[0][0], est + err[1][0]], [y, y], color=colour, lw=2, zorder=2)
    ax.scatter([est], [y], s=size, color="white" if hollow else colour, edgecolor=colour,
               linewidth=2 if hollow else 1.5, zorder=3)


def fig_quality(ctx):
    name, title = ctx["fig"], "Quality vs FP8: perplexity change against R0 (95% CIs)"
    comps = quality_comps(ctx)
    if not comps:
        return pending(ctx, name, title, ["No arm has been compared with R0 in docs/fidelity/metrics-v2/ yet."])
    plt = ctx["plt"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.4), facecolor="white", gridspec_kw={"width_ratios": [1, 1.25]})
    fig.subplots_adjust(left=0.1, right=0.98, top=0.76, bottom=0.25, wspace=0.28)
    header(fig, title)
    lad = ladder_json(ctx)
    ladder_pts = []
    if lad:
        for e in lad.get("cumulative_from_first") or []:
            arm = e["cand"].split("/")[0]
            ci = ((e.get(lad.get("regime", "dense")) or {}).get("contrast") or {}).get("delta_nll")
            if e.get("status") in ("ok", "partial") and arm in dict(LADDER_ARMS) and ppl(ci)[0] is not None:
                ladder_pts.append((arm, dict(LADDER_ARMS)[arm], ci, (e.get("windows") or {}).get("compared")))
    legend = [(ARM_NAME.get(label, label), colour) for label, _, colour in comps]
    legend += [(f"{arm} (ladder subset, dense)", colour) for arm, colour, _, _ in ladder_pts]
    fig_legend(fig, legend)
    regimes = ["dense", "sparse", "all"]
    ax = axes[0]
    n = len(comps) + len(ladder_pts)
    for i, regime in enumerate(regimes):
        rows = [(colour, ((res.get("regimes", {}).get(regime) or {}).get("contrast") or {}).get("delta_nll"))
                for _, res, colour in comps]
        if regime == "dense":
            rows += [(colour, ci) for _, colour, ci, _ in ladder_pts]
        for j, (colour, ci) in enumerate(rows):
            est, err = ppl(ci)
            if est is not None:
                dot_whisker(ax, i - 0.3 + 0.6 * (j + 0.5) / max(n, 1), est, err, colour)
    ax.axvline(0, color=INK, lw=1.2)
    ax.set_yticks(range(3), ["Dense\n≤ 2,048 tokens", "Sparse\n> 2,048 tokens", "All positions"], fontsize=11,
                  color=INK)
    ax.set_ylim(2.5, -0.5)
    style(ax, "By regime", "Perplexity change vs R0, % · lower is better")
    ax = axes[1]
    cats = sorted({c for _, res, _ in comps for c in ((res.get("regimes", {}).get("all") or {}).get("by_category") or {})})
    for i, cat in enumerate(cats):
        for j, (_, res, colour) in enumerate(comps):
            blk = ((res.get("regimes", {}).get("all") or {}).get("by_category") or {}).get(cat) or {}
            est, err = ppl((blk.get("contrast") or {}).get("delta_nll"))
            if est is not None:
                dot_whisker(ax, i - 0.3 + 0.6 * (j + 0.5) / len(comps), est, err, colour)
    ax.axvline(0, color=INK, lw=1.2)
    ax.set_yticks(range(len(cats)), [c.replace("_", " ") for c in cats], fontsize=11, color=INK)
    ax.set_ylim(len(cats) - 0.5, -0.5)
    style(ax, "By corpus category · all positions", "Perplexity change vs R0, % · lower is better")
    notes = [QUALITY_NOTE,
             "E29 (Cm) predicts real text as well as FP8 (R0) wherever its interval straddles the zero line.",
             "Whiskers: 95% percentile bootstrap over source groups. Sparse positions carry run-to-run noise even "
             "for R0 against itself."] + partial_notes([(f"{label} vs R0", res, c) for label, res, c in comps]) \
        + json_missing(ctx)
    footer(fig, notes)
    ctx["out"].save(fig, name)
    return {"status": "partial" if json_missing(ctx) or partial_notes(comps) else "ok", "notes": notes[3:]}


def group_map(path: Path) -> dict:
    data = read_json(path)
    return data if isinstance(data, dict) else {}


def self_pair(raw: Path, ref: str, cand: str, groups: dict, B=m.BOOT_B, seed=m.BOOT_SEED):
    """Run-to-run noise of one arm: KL, top-1 and ΔNLL of `cand` against `ref` on their shared windows,
    per regime, with a percentile bootstrap over the analyzer's source groups (windows when unavailable)."""
    ref_dir, cand_dir = raw / ref, raw / cand
    if not (cand_dir / "prompt").is_dir():
        return None
    per = {r: {"kl": [], "top1": [], "dnll": [], "n": [], "nd": [], "unit": []} for r in ("dense", "all")}
    units = []
    for path in sorted((cand_dir / "prompt").glob("*.npz")):
        wid = path.stem
        r, c = load_prompt(ref_dir, wid), load_prompt(cand_dir, wid)
        if r is None or c is None or not np.array_equal(r["ids"], c["ids"]):
            continue
        rows = np.arange(1, r["ids"].shape[0])
        eid = r["ids"][rows].astype(np.int64)
        kl, valid, _, _ = m.np_coarse_kl(r["topk_ids"][rows], r["topk_lp"][rows], c["topk_ids"][rows],
                                         c["topk_lp"][rows], eid, r["lp_actual"][rows], c["lp_actual"][rows])
        top1 = m.np_top1(r["topk_ids"][rows], r["topk_lp"][rows], c["topk_ids"][rows], c["topk_lp"][rows])
        ok = valid & (top1 >= 0)
        ra, ca = r["lp_actual"][rows].astype(np.float64), c["lp_actual"][rows].astype(np.float64)
        dn = ok & np.isfinite(ra) & np.isfinite(ca)
        unit = groups.get(wid, wid)
        units.append(unit)
        for regime in ("dense", "all"):
            sel = rows <= DENSE_MAX if regime == "dense" else np.ones(rows.size, bool)
            d = per[regime]
            d["kl"].append(float(kl[ok & sel].sum()))
            d["top1"].append(float((top1[ok & sel] == 1).sum()))
            d["n"].append(int((ok & sel).sum()))
            d["dnll"].append(float((ra - ca)[dn & sel].sum()))
            d["nd"].append(int((dn & sel).sum()))
            d["unit"].append(unit)
    if not units:
        return None
    keys = sorted(set(units))
    out = {"windows": len(units), "units": len(keys), "unit": "source group" if groups else "window"}
    idx = m.bootstrap_indices(len(keys), B, seed)
    for regime, d in per.items():
        lab = np.asarray([keys.index(u) for u in d["unit"]])
        sums = {k: np.bincount(lab, weights=np.asarray(d[k], float), minlength=len(keys))
                for k in ("kl", "top1", "dnll", "n", "nd")}
        res = {}
        for key, den, scale in (("kl", "n", 1.0), ("top1", "n", 1.0), ("dnll", "nd", 1.0)):
            num_, den_ = sums[key], sums[den]
            est = num_.sum() / den_.sum() * scale if den_.sum() else None
            reps = np.sort([num_[i].sum() / den_[i].sum() for i in (np.asarray(x) for x in idx) if den_[i].sum()])
            res[key] = {"estimate": est, "ci_low": float(np.percentile(reps, 2.5)) if reps.size else None,
                        "ci_high": float(np.percentile(reps, 97.5)) if reps.size else None}
        out[regime] = res
    return out


LABEL_OFFSET = {"R0 self": (8, 8, "left"), "Cpre": (8, 8, "left"), "Cm": (10, -16, "left"),
                "N": (8, 8, "left"), "N self": (10, -34, "left")}


def fig_scatter(ctx):
    name, title = ctx["fig"], "Different vs worse: deviation from R0 against perplexity change"
    comps = quality_comps(ctx)
    if not comps:
        return pending(ctx, name, title, ["No arm has been compared with R0 in docs/fidelity/metrics-v2/ yet."])
    plt = ctx["plt"]
    noise = self_pair(ctx["raw"], "N/prompt-a", "N/prompt-rep2", ctx["groups"])
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.2), facecolor="white")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.76, bottom=0.25, wspace=0.18)
    header(fig, title)
    legend = [("R0 vs itself (run B vs A)", FLOOR_COLOR)] + [(ARM_NAME.get(l, l), c) for l, _, c in comps]
    fig_legend(fig, legend + ([("N vs itself (run A vs rep2), hollow", ORANGE)] if noise else []))
    floor = ctx["m"].get("floor-r0")
    for ax, regime in zip(axes, ("dense", "all")):
        pts = []
        if floor:
            pts.append(("R0 self", (floor["regimes"].get(regime) or {}).get("contrast") or {}, FLOOR_COLOR, False))
        pts += [(label, (res["regimes"].get(regime) or {}).get("contrast") or {}, colour, False)
                for label, res, colour in comps]
        if noise:
            pts.append(("N self", {"kl": {"mean_ci": noise[regime]["kl"]}, "delta_nll": noise[regime]["dnll"],
                                   "top1_agreement": noise[regime]["top1"]}, ORANGE, True))
        for label, c, colour, hollow in pts:
            x, xerr = ci_err((c.get("kl") or {}).get("mean_ci"))
            y, yerr = ppl(c.get("delta_nll"))
            if x is None or y is None:
                continue
            if xerr is not None:
                ax.plot([x - xerr[0][0], x + xerr[1][0]], [y, y], color=colour, lw=1.6, zorder=2)
            if yerr is not None:
                ax.plot([x, x], [y - yerr[0][0], y + yerr[1][0]], color=colour, lw=1.6, zorder=2)
            ax.scatter([x], [y], s=90, color="white" if hollow else colour, edgecolor=colour, linewidth=2, zorder=3)
            t1 = (c.get("top1_agreement") or {}).get("estimate")
            if t1 is not None:
                dx, dy, ha = LABEL_OFFSET.get(label, (8, 8, "left"))
                ax.annotate(f"{label} · top-1 {t1 * 100:.1f}%", (x, y), xytext=(dx, dy), textcoords="offset points",
                            fontsize=9.5, color=INK, ha=ha)
        ax.axhline(0, color=INK, lw=1.2)
        ax.set_xlim(-0.02, ax.get_xlim()[1] * 1.2)
        style(ax, "Dense regime (≤ 2,048 conditioning tokens)" if regime == "dense" else "All positions",
              "Mean coarse KL(R0 ‖ arm), nats · a lower bound on the full KL",
              "Perplexity change vs R0, % · lower is better" if regime == "dense" else None)
    notes = ["Right = different from R0; up = worse at predicting real text. Different is not necessarily worse: a "
             "point far right on the zero line changes the distribution without a perplexity cost.",
             QUALITY_NOTE]
    if noise:
        notes.append(f"N vs itself: {noise['windows']} cross-boot windows, bootstrap over {noise['units']} "
                     f"{noise['unit']}s; N's engine is nondeterministic from the first positions.")
    else:
        notes.append("N run-to-run noise (N/prompt-rep2) not available yet.")
    footer(fig, notes + json_missing(ctx))
    ctx["out"].save(fig, name)
    return {"status": "partial" if json_missing(ctx) else "ok", "notes": notes[2:] + json_missing(ctx)}


def bit_identical_dense(raw: Path, ref: str, cand: str, windows) -> bool | None:
    """True when rows 1..2,048 of lp_actual, topk_ids and topk_lp are bitwise equal on every shared window."""
    compared = 0
    for wid in windows:
        r, c = load_prompt(raw / ref, wid), load_prompt(raw / cand, wid)
        if r is None or c is None:
            continue
        rows = slice(1, DENSE_MAX + 1)
        k = min(r["topk_ids"].shape[1], c["topk_ids"].shape[1])
        for a, b in ((r["lp_actual"][rows], c["lp_actual"][rows]),
                     (r["topk_ids"][rows, :k], c["topk_ids"][rows, :k]),
                     (r["topk_lp"][rows, :k], c["topk_lp"][rows, :k])):
            if a.shape != b.shape or a.tobytes() != b.tobytes():
                return False
        compared += 1
    return True if compared else None


def fig_waterfall(ctx):
    name, title = ctx["fig"], "Ladder: where the deviation and the perplexity change come from (dense regime)"
    lad = ladder_json(ctx)
    ok_steps = [s for s in (lad or {}).get("steps", []) if s.get("status") in ("ok", "partial")]
    if not lad or not ok_steps:
        return pending(ctx, name, title, [
            "No ladder rung has been measured yet (runs L0919/LE21/LE22b prompt-ladder).",
            "Chain: R0 → L0919 (hybrid KDA) → Cpre (+E03 mHC) → LE21 (+E21) → LE22b (+E22b drafter) → Cm (E29)."])
    regime = lad.get("regime", "dense")
    subset_path = ctx["corpus"] / (lad.get("subset") or "subsets/ladder.txt")
    windows = [w.strip() for w in subset_path.read_text().splitlines() if w.strip() and not w.startswith("#")] \
        if subset_path.exists() else []
    plt = ctx["plt"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.2), facecolor="white")
    fig.subplots_adjust(left=0.22, right=0.98, top=0.8, bottom=0.25, wspace=0.5)
    header(fig, title)
    identical = []
    steps = lad.get("steps") or []
    ax = axes[0]
    for i, s in enumerate(steps):
        rung = s["cand"].split("/")[0]
        control = s.get("negative_control")
        colour = FLOOR_COLOR if control else TEAL
        if s.get("status") not in ("ok", "partial"):
            ax.text(0, i, "  pending", va="center", fontsize=9.5, color=MUTED)
            continue
        est, err = ci_err(((s.get(regime) or {}).get("contrast") or {}).get("kl", {}).get("mean_ci"))
        same = bit_identical_dense(ctx["raw"], s["ref"], s["cand"], windows) if windows else None
        if same:
            identical.append(rung)
        if est is not None:
            ax.barh(i, est, height=0.6, color=colour, zorder=2)
            if err is not None:
                ax.errorbar(est, i, xerr=err, fmt="none", ecolor=INK, capsize=3, zorder=3)
        w = s.get("windows") or {}
        tag = "≡ bit-identical" if same else f"{num_fmt(est)}"
        ax.annotate(f"{tag} · {w.get('compared')} win.", (est or 0, i), xytext=(6, 9), textcoords="offset points",
                    fontsize=9, color=INK)
    ax.set_yticks(range(len(steps)), [RUNG_LABEL.get(s["cand"].split("/")[0], s["cand"].split("/")[0])
                                      + (" · control" if s.get("negative_control") else "") for s in steps],
                  fontsize=10, color=INK)
    ax.set_ylim(len(steps) - 0.5, -0.5)
    style(ax, "Step: KL(previous rung ‖ rung)", "Mean coarse KL, nats (95% CI)")
    ax = axes[1]
    cum = lad.get("cumulative_from_first") or []
    labels = ["R0 (FP8 reference)"]
    ax.scatter([0], [0], s=60, color=FLOOR_COLOR, zorder=3)
    for i, e in enumerate(cum, start=1):
        rung = e["cand"].split("/")[0]
        labels.append(rung + (" (E29)" if rung == "Cm" else ""))
        if e.get("status") not in ("ok", "partial"):
            ax.text(0, i, "  pending", va="center", fontsize=9.5, color=MUTED)
            continue
        est, err = ppl(((e.get(regime) or {}).get("contrast") or {}).get("delta_nll"))
        if est is None:
            continue
        dot_whisker(ax, i, est, err, TEAL)
        ax.annotate(f"{est:+.2f}% · {(e.get('windows') or {}).get('compared')} win.", (est, i), xytext=(8, 9),
                    textcoords="offset points", fontsize=9, color=INK)
    ax.axvline(0, color=INK, lw=1.2)
    ax.set_yticks(range(len(labels)), labels, fontsize=10, color=INK)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    style(ax, "Cumulative perplexity change vs R0", "Perplexity change vs R0, % · lower is better")
    notes = [f"Ladder subset ({len(windows) or '?'} windows), dense regime; 95% CIs bootstrap over source groups "
             "(analyzer grouping). Grey steps are negative controls (drafter, scheduler or KV pool only).",
             "≡ bit-identical: rows 1–2,048 of the actual-token logprob and top-K arrays equal bit for bit on every "
             "shared window.", f"Negative-control attribution: {lad.get('attribution', 'n/a')}."]
    if lad.get("status") != "ok":
        notes.append("Some rungs are not yet measured or partial; cumulative values use the windows each rung has.")
    footer(fig, notes)
    ctx["out"].save(fig, name)
    return {"status": "ok" if lad.get("status") == "ok" else "partial",
            "notes": notes[2:] + ([f"Bit-identical steps: {', '.join(identical)}."] if identical else [])}


def num_fmt(x):
    return "–" if x is None else f"{x:.4f}"


FIGURES = [  # (name, function, needs per-position data, caption, report group)
    ("01-quality-vs-fp8", fig_quality, False,
     "Perplexity change of each arm against R0 (the vendor FP8 recipe), exp(ΔNLL) − 1, per regime and per corpus "
     "category, with 95% CIs. Lower is better; E29 (Cm) predicts real text as well as FP8 when its bar straddles "
     "zero.", "quality"),
    ("02-different-vs-worse", fig_scatter, False,
     "Deviation from R0 (mean coarse KL, a lower bound) against perplexity change, one point per arm with 95% CI "
     "crosses, R0 against itself at the origin and N against itself as its run-to-run noise.", "quality"),
    ("03-ladder-waterfall", fig_waterfall, False,
     "Recipe ladder R0 → L0919 → Cpre → LE21 → LE22b → Cm in the dense regime: KL added by each step and the "
     "cumulative perplexity change against R0, with bit-identical steps marked.", "quality"),
    ("04-kl-cdf", fig_cdf, True,
     "Cumulative distribution of per-position coarse KL against R0 for each measured arm, with the R0 run-B-vs-A "
     "floor, separately for the dense and sparse regimes.", "results"),
    ("05-kl-by-position", fig_position, True,
     "Median and p99 of per-position coarse KL in quarter-octave position bins, per arm, with the R0 floor. The "
     "dotted line marks 2,048 conditioning tokens.", "results"),
    ("06-kl-tail-survival", fig_tail, True,
     "Tail survival P(KL > x) on log-log axes: how often large deviations occur, per regime.", "results"),
    ("07-by-category", fig_category, False,
     "Mean coarse KL and top-1 agreement against R0 by corpus category, per regime, with 95% group-bootstrap CIs.",
     "results"),
    ("08-by-prefill-path", fig_path, True,
     "Mean coarse KL by prefill path (Marlin W8A16 chunks under 2,048 rows vs dequantised BF16 chunks).", "results"),
    ("09-decode-path", fig_decode, False,
     "Greedy decode: KL along the identical prefix and Kaplan-Meier survival of identical prefixes.", "results"),
    ("10-task-pass-rate", fig_tasks, False,
     "Task pass rate per arm with Wilson 95% CIs, and paired greedy discordance with exact McNemar tests.", "results"),
    ("11-ladder-attribution", fig_ladder, False,
     "Dense-regime attribution along the recipe ladder, cumulative from R0 and per step, on the ladder subset.",
     "results"),
    ("12-k-sensitivity", fig_k, False,
     "Mean coarse KL of Cm vs R0 at K = 100 and truncated to K = 20 on the same rows.", "results"),
    ("13-voxel-showcase", fig_voxel, False,
     "Greedy voxel renders per arm and prompt (a visual sanity check; no precision conclusion).", "results"),
    ("14-repeatability", fig_repeatability, False,
     "Run-to-run repeatability: R0 against itself by position bucket, and the first non-identical position in "
     "each arm's determinism probe.", "results"),
    ("15-precision-vs-r0", fig_precision, False,
     "Summary of Cm, Cpre and N against R0: mean coarse KL and mean ΔNLL per regime with 95% CIs, with the floor.",
     "results"),
]
TITLES = {
    "01-quality-vs-fp8": "Quality vs FP8",
    "02-different-vs-worse": "Different vs worse",
    "03-ladder-waterfall": "Ladder waterfall",
    "04-kl-cdf": "Per-position KL: cumulative distribution",
    "05-kl-by-position": "KL along the context",
    "06-kl-tail-survival": "KL tail survival",
    "07-by-category": "KL and top-1 agreement by category",
    "08-by-prefill-path": "KL by prefill path",
    "09-decode-path": "Decode path",
    "10-task-pass-rate": "Task pass rates",
    "11-ladder-attribution": "Ladder attribution",
    "12-k-sensitivity": "K sensitivity",
    "13-voxel-showcase": "Voxel showcase",
    "14-repeatability": "Repeatability and the 2,048 onset",
    "15-precision-vs-r0": "Precision vs R0",
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--docs", type=Path, default=REPO / "docs/fidelity")
    p.add_argument("--raw-root", type=Path, default=REPO / "data/fidelity/raw")
    p.add_argument("--prelim", type=Path, default=REPO / "data/fidelity/prelim")
    p.add_argument("--corpus", type=Path, default=REPO / "data/fidelity/corpus",
                   help="private corpus directory (ladder subset list only)")
    p.add_argument("--groups", type=Path, default=REPO / "data/fidelity/metrics-v2/groups.json",
                   help="private window -> source-group map written by analyze_campaign.py")
    p.add_argument("--stride", type=int, default=1,
                   help="keep every n-th position for the per-position figures (deterministic subsample)")
    p.add_argument("--only", nargs="*", help="figure names to render (default: all)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    plt = setup_matplotlib()
    metrics_dir = args.docs / "metrics-v2"
    out = Out(args.docs / "plots", plt)
    names = [f[0] for f in FIGURES if not args.only or f[0] in args.only]
    ctx = {"plt": plt, "out": out, "docs": args.docs, "metrics": metrics_dir, "raw": args.raw_root,
           "prelim": args.prelim, "corpus": args.corpus, "groups": group_map(args.groups), "m": {}}
    for _, _, comp, _ in ARMS + [FLOOR]:
        ctx["m"][comp] = metric(metrics_dir, comp)
    needs_pp = any(f[2] for f in FIGURES if f[0] in names)
    ctx["pp"] = {}
    if needs_pp:
        cands = {FLOOR[0]: FLOOR_RUN, **{label: run for label, run, _, _ in ARMS}}
        ctx["pp"] = per_position(args.raw_root, REF_RUN, cands, max(1, args.stride))
    index_path = args.docs / "plots" / "plots.json"
    index = {f["name"]: f for f in (read_json(index_path) or {}).get("figures", [])}
    failures = 0
    for fname, func, _, caption, group in FIGURES:
        if fname not in names:
            continue
        ctx["fig"] = fname
        try:
            info = func(ctx)
        except Exception as exc:  # render a labelled placeholder rather than abort the whole report
            failures += 1
            traceback.print_exc()
            plt.close("all")
            out.placeholder(fname, TITLES[fname], [f"Rendering failed ({type(exc).__name__}); see the build log."])
            info = {"status": "error", "notes": [f"rendering failed: {type(exc).__name__}"]}
        index[fname] = {"name": fname, "title": TITLES[fname], "caption": caption, "group": group,
                        "status": info["status"],
                        "notes": info.get("notes", []), "png": f"{fname}.png", "svg": f"{fname}.svg"}
        print(f"figure {fname}: {info['status']}", flush=True)
    ordered = [index[f[0]] for f in FIGURES if f[0] in index]
    fio.write_json_atomic(index_path, {"schema": "fidelity-plots/1",
                                       "stride": max(1, args.stride), "figures": ordered})
    if failures:
        print(f"{failures} figure(s) failed and were replaced by placeholders", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
