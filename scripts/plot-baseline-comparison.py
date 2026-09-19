#!/usr/bin/env python3
"""Render the current frozen baseline; requires matplotlib==3.11.2.

Reads the 16 saved medians only. No benchmark requests are sent.
The historical CLI filename is retained for existing callers.
"""

from __future__ import annotations

import argparse
from datetime import date
from decimal import Decimal
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "docs/historical_benchmarks/baselines/2026-09-19-e03/baseline.json"
OUTPUT_DIR = ROOT / "docs/plots/baselines/2026-09-19-e03"
TEAL, INK, MUTED = "#087f74", "#172b46", "#536478"

GENERATION = [
    ("code_decode_throughput", "Code decode"),
    ("prose_decode_throughput", "Prose decode"),
    ("code_c1_aggregate_end_to_end_throughput", "Code · 1 request\nend-to-end"),
    ("code_c2_aggregate_end_to_end_throughput", "Code · 2 requests\naggregate, end-to-end"),
    ("code_c4_aggregate_end_to_end_throughput", "Code · 4 requests\naggregate, end-to-end"),
]
LATENCY = [
    ("code_ttft", "Code decode"),
    ("prose_ttft", "Prose decode"),
    ("code_c1_per_stream_ttft", "Code · 1 request"),
    ("code_c2_per_stream_ttft", "Code · 2 requests\nper stream"),
    ("code_c4_per_stream_ttft", "Code · 4 requests\nper stream"),
]
COLD = [(f"prefill_{depth}k_cold_throughput", f"{depth}K tokens") for depth in (8, 32, 64)]
REPLAY = [(f"prefill_{depth}k_replay_throughput", f"{depth}K tokens") for depth in (8, 32, 64)]


def read_baseline():
    baseline = json.loads(BASELINE.read_text())
    metrics = baseline["performance"]["metrics"]
    expected = {key for key, _ in GENERATION + LATENCY + COLD + REPLAY}
    if len(metrics) != 16 or {row["key"] for row in metrics} != expected:
        raise ValueError("The current frozen baseline must provide all 16 metrics")
    medians = {row["key"]: Decimal(str(row["median"])) for row in metrics}
    if any(not value.is_finite() or value < 0 for value in medians.values()):
        raise ValueError("Baseline medians must be finite and nonnegative")
    measured_on = date.fromisoformat(baseline["measured_on"]).strftime("%d %b %Y")
    runs = baseline["performance"]["included_run_count"]
    requests = baseline["functional"]["measured_requests"]["count"]
    caption = f"{measured_on} · Median of {runs} accepted runs · {requests} requests"
    return medians, caption


def panel(ax, rows, medians, title, xlabel, *, seconds=False):
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    values = [float(medians[key]) for key, _ in rows]
    ax.barh(range(len(rows)), values, height=0.55, color=TEAL, zorder=2)
    for index, (key, _) in enumerate(rows):
        precision = 3 if seconds else 1 if key.startswith("prefill_") else 2
        label = f"{medians[key]:,.{precision}f}"
        ax.annotate(label, (values[index], index), xytext=(7, 0),
                    textcoords="offset points", va="center", fontsize=11, color=TEAL)
    ax.set_title(title, loc="left", fontsize=15, color=INK, fontweight="bold", pad=20)
    ax.set_yticks(range(len(rows)), [label for _, label in rows], fontsize=11, color=INK)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xscale("linear")
    ax.set_xlim(0, max(values) * 1.25 or 1)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}" if seconds else f"{x:,.0f}"))
    ax.set_xlabel(xlabel, fontsize=11, color=MUTED, labelpad=12)
    ax.tick_params(axis="both", length=0, labelcolor=MUTED, pad=9)
    ax.grid(axis="x", color="#dfe5ed", linewidth=0.8, zorder=1)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)


def figure(output_dir, name, title, panels, medians, caption):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14.5, 7.4), facecolor="white")
    axes = [fig.add_axes([0.15, 0.22, 0.30, 0.48]), fig.add_axes([0.64, 0.22, 0.31, 0.48])]
    fig.text(0.035, 0.935, title, fontsize=23, fontweight="bold", color=INK)
    fig.text(0.035, 0.88, "GLM-5.3-Flash · TP4 / four GB10 nodes · Native Rigmark",
             fontsize=12, color=MUTED)
    fig.text(0.035, 0.825, caption, fontsize=12, color=TEAL)
    for ax, (rows, heading, xlabel, seconds) in zip(axes, panels):
        panel(ax, rows, medians, heading, xlabel, seconds=seconds)
    fig.text(0.035, 0.11, "Bars and labels show the frozen baseline medians, matching the README table.",
             fontsize=10.5, color=MUTED)
    fig.text(0.035, 0.065, "Decode excludes the initial wait; end-to-end throughput includes it. K = 1,024 tokens.",
             fontsize=10.5, color=MUTED)
    fig.savefig(output_dir / f"{name}.png", dpi=160, facecolor="white", metadata={"Software": "Matplotlib"})
    svg = output_dir / f"{name}.svg"
    fig.savefig(svg, facecolor="white", metadata={"Date": None, "Creator": "Matplotlib"})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    medians, caption = read_baseline()
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                                "svg.hashsalt": "tp4-rigmark-baseline", "svg.fonttype": "path"})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure(args.output_dir, "generation", "Generation throughput and first-token latency", [
        (GENERATION, "Throughput · higher is better", "Tokens per second", False),
        (LATENCY, "Time to first token · lower is better", "Seconds", True),
    ], medians, caption)
    figure(args.output_dir, "prefill", "Long-context prefill and prefix-cache reuse", [
        (COLD, "Cold prefill · higher is better", "Effective prefill tokens per second", False),
        (REPLAY, "Immediate replay · higher is better", "Effective prefill tokens per second", False),
    ], medians, caption)
    print(f"Rendered two figures from 16 current frozen medians to {args.output_dir}")


if __name__ == "__main__":
    main()
