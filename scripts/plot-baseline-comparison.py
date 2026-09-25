#!/usr/bin/env python3
"""Compare the current and previous frozen baselines; requires matplotlib==3.11.2.

Reads all 16 saved medians from each record. No benchmark requests are sent.
"""

from __future__ import annotations

import argparse
from datetime import date
from decimal import Decimal
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "docs/historical_benchmarks/baselines/2026-09-25-e28b/baseline.json"
PREVIOUS_BASELINE = ROOT / "docs/historical_benchmarks/baselines/2026-09-25-e27c/baseline.json"
OUTPUT_DIR = ROOT / "docs/plots/comparisons/2026-09-25-e28b-vs-2026-09-25-e27c"
TEAL, INK, MUTED = "#087f74", "#172b46", "#536478"
PREVIOUS_COLOR = "#92a5ba"
# Match the owner's descriptive convention in the accepted reports: within about the
# measured run-to-run noise. A display choice, not a statistical test or acceptance rule.
APPROX_UNCHANGED = Decimal("2.2")

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


def read_baseline(path, label):
    baseline = json.loads(path.read_text(), parse_float=Decimal)
    metrics = baseline["performance"]["metrics"]
    expected = {key for key, _ in GENERATION + LATENCY + COLD + REPLAY}
    if len(metrics) != 16 or {row["key"] for row in metrics} != expected:
        raise ValueError(f"{path}: the frozen baseline must provide all 16 metrics")
    medians = {row["key"]: Decimal(row["median"]) for row in metrics}
    if any(not value.is_finite() or value <= 0 for value in medians.values()):
        raise ValueError(f"{path}: baseline medians must be finite and positive")
    measured_on = date.fromisoformat(baseline["measured_on"]).strftime("%d/%m/%Y")
    runs = baseline["performance"]["included_run_count"]
    requests = baseline["functional"]["measured_requests"]["count"]
    caption = f"{label} · {measured_on}\n{runs} accepted runs · {requests} requests"
    return medians, caption


def panel(ax, rows, current, previous, title, xlabel, *, seconds=False):
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    maximum = 0
    for medians, offset, color in ((previous, -0.17, PREVIOUS_COLOR),
                                    (current, 0.17, TEAL)):
        values = [float(medians[key]) for key, _ in rows]
        positions = [index + offset for index in range(len(rows))]
        maximum = max(maximum, *values)
        ax.barh(positions, values, height=0.29, color=color, zorder=2)
        for index, (key, _) in enumerate(rows):
            precision = 4 if seconds else 1 if key.startswith("prefill_") else 2
            label = f"{medians[key]:,.{precision}f}"
            ax.annotate(label, (values[index], positions[index]), xytext=(7, 0),
                        textcoords="offset points", va="center", fontsize=11.5,
                        color=MUTED if offset < 0 else TEAL)

    ax.text(1.21, 1.04, "Δ vs previous", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=11.5, color=INK, fontweight="bold")
    for index, (key, _) in enumerate(rows):
        delta = (current[key] / previous[key] - 1) * 100
        unchanged = abs(delta) <= APPROX_UNCHANGED
        improved = delta < 0 if seconds else delta > 0
        color = MUTED if unchanged else TEAL if improved else "#b45309"
        label = f"{delta:+.2f}%".replace("-", "−")
        if unchanged:
            label += "\n≈ unchanged"
        ax.text(1.21, index, label, transform=ax.get_yaxis_transform(),
                ha="center", va="center", fontsize=11.5, color=color,
                fontweight="bold", linespacing=1.5)

    ax.set_title(title, loc="left", fontsize=15, color=INK, fontweight="bold", pad=20)
    ax.set_yticks(range(len(rows)), [label for _, label in rows], fontsize=12, color=INK)
    ax.set_ylim(len(rows) - 0.45, -0.55)
    ax.set_xscale("linear")
    ax.set_xlim(0, maximum * 1.38)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}" if seconds else f"{x:,.0f}"))
    ax.set_xlabel(xlabel, fontsize=11, color=MUTED, labelpad=12)
    ax.tick_params(axis="both", length=0, labelcolor=MUTED, pad=9)
    ax.grid(axis="x", color="#dfe5ed", linewidth=0.8, zorder=1)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)


def figure(output_dir, name, title, panels, current, previous, captions, note):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    height = 9.4 if name == "generation" else 7.8
    fig = plt.figure(figsize=(17.2, height), facecolor="white")
    axes = [fig.add_axes([0.145, 0.23, 0.26, 0.47]), fig.add_axes([0.650, 0.23, 0.255, 0.47])]
    fig.text(0.035, 0.935, title, fontsize=23, fontweight="bold", color=INK)
    fig.text(0.035, 0.88, "GLM-5.3-Flash · Four GB10 nodes · E28b seven draft tokens and 16 GiB KV over E27c · Native Rigmark",
             fontsize=12, color=MUTED)
    fig.legend(handles=[Patch(color=PREVIOUS_COLOR, label=captions[0]),
                        Patch(color=TEAL, label=captions[1])],
               loc="upper left", bbox_to_anchor=(0.03, 0.845), ncols=2,
               frameon=False, fontsize=12, labelcolor=INK, columnspacing=4)
    for ax, (rows, heading, xlabel, seconds) in zip(axes, panels):
        panel(ax, rows, current, previous, heading, xlabel, seconds=seconds)
    fig.text(0.035, 0.11, "Δ = (current / previous − 1) × 100, calculated from unrounded frozen medians.",
             fontsize=10.5, color=MUTED)
    fig.text(0.035, 0.073, "≈ unchanged: owner-accepted changes of roughly 1–2%; this is not a statistical test.",
             fontsize=10.5, color=MUTED)
    fig.text(0.035, 0.036, note, fontsize=10.5, color=MUTED)
    fig.savefig(output_dir / f"{name}.png", dpi=160, facecolor="white", metadata={"Software": "Matplotlib"})
    svg = output_dir / f"{name}.svg"
    fig.savefig(svg, facecolor="white", metadata={"Date": None, "Creator": "Matplotlib"})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    current, current_caption = read_baseline(BASELINE, "Current E28b")
    previous, previous_caption = read_baseline(PREVIOUS_BASELINE, "Previous E27c")
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                                "svg.hashsalt": "tp4-rigmark-current-vs-previous", "svg.fonttype": "path"})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    captions = previous_caption, current_caption
    figure(args.output_dir, "generation", "Generation: current vs previous baseline", [
        (GENERATION, "Generation throughput", "Tokens per second · higher is better", False),
        (LATENCY, "First-token latency", "Seconds · lower is better", True),
    ], current, previous, captions,
        "Decode excludes initial wait; end-to-end throughput includes it. Concurrency TTFT is per stream.")
    figure(args.output_dir, "prefill", "Prefill: current vs previous baseline", [
        (COLD, "Cold prefill", "Effective tokens per second · higher is better", False),
        (REPLAY, "Immediate replay", "Effective tokens per second · higher is better", False),
    ], current, previous, captions,
        "Cold prefill uses a fresh cache salt; replay reuses the prefix within each run. K = 1,024 tokens.")
    print(f"Rendered two comparison figures from 16 frozen metrics per baseline to {args.output_dir}")


if __name__ == "__main__":
    main()
