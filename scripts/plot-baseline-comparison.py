#!/usr/bin/env python3
"""Plot the public Rigmark records; requires matplotlib==3.11.2.

Reads saved public JSON records only. Medians use the frozen baseline
runs, matching the README table. The separate IaC reproduction is a point only.
"""

from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "docs/baseline-f0.json",
    "docs/baseline-2026-09-19.json",
    "docs/reproduction-2026-09-19.json",
)
BLUE, TEAL, GOLD = "#2563eb", "#087f74", "#c47708"
INK, MUTED = "#172b46", "#536478"

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
CONTEXT_SOURCE = "docs/context-decode-probes.json"


def read_series():
    initial, latest, reproduction = [
        {row["key"]: row for row in json.loads((ROOT / name).read_text())["performance"]["metrics"]}
        for name in SOURCES
    ]
    expected = {key for key, _ in GENERATION + LATENCY + COLD + REPLAY}
    if not set(initial) == set(latest) == set(reproduction) == expected:
        raise ValueError("The three records must provide the same 16 metrics")
    series = {}
    for key in expected:
        old, new, replay = initial[key], latest[key], reproduction[key]
        if not (len(old["per_run"]) == 3 and len(new["per_run"]) == 2):
            raise ValueError(f"Unexpected run count: {key}")
        for field in ("unit", "source_path"):
            if not old[field] == new[field] == replay[field]:
                raise ValueError(f"Metric definition differs: {key}, {field}")
        for record in (old, new):
            if abs(median(record["per_run"]) - record["median"]) > 1e-9:
                raise ValueError(f"Frozen median differs from its run values: {key}")
        series[key] = (old["median"], new["median"], replay["reproduction_single_run_value"])
    return series


def panel(ax, rows, series, title, xlabel, *, seconds=False):
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    values = []
    for row_index, (key, _) in enumerate(rows):
        for group, (center, color) in enumerate(zip(series[key][:2], (BLUE, TEAL))):
            y = row_index + (-0.18 if group == 0 else 0.18)
            values.append(center)
            ax.barh(y, center, height=0.27, color=color, zorder=2)
            if group == 1:
                values.append(series[key][2])
                ax.scatter(series[key][2], y, s=52, marker="D", color=GOLD,
                           edgecolors="white", linewidths=0.9, zorder=3)
            precision = 3 if seconds else 1 if key.startswith("prefill_") else 2
            label = f"{Decimal(str(center)):,.{precision}f}"
            ax.text(1.025, y, label, transform=ax.get_yaxis_transform(), va="center",
                    color=color, fontsize=10.5)
    ax.set_title(title, loc="left", fontsize=15, color=INK, fontweight="bold", pad=20)
    ax.text(1.025, 1.04, "Median", transform=ax.transAxes, color=MUTED, fontsize=10)
    ax.set_yticks(range(len(rows)), [label for _, label in rows], fontsize=11, color=INK)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(0, max(values) * 1.08)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}" if seconds else f"{x:,.0f}"))
    ax.set_xlabel(xlabel, fontsize=11, color=MUTED, labelpad=12)
    ax.tick_params(axis="both", length=0, labelcolor=MUTED, pad=9)
    ax.grid(axis="x", color="#dfe5ed", linewidth=0.8, zorder=1)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)


def figure(output_dir, name, title, panels, series):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig = plt.figure(figsize=(14.5, 7.4), facecolor="white")
    axes = [fig.add_axes([0.15, 0.22, 0.29, 0.48]), fig.add_axes([0.64, 0.22, 0.27, 0.48])]
    fig.text(0.035, 0.935, title, fontsize=23, fontweight="bold", color=INK)
    fig.text(0.035, 0.885, "GLM-5.3-Flash · TP4 / four GB10 nodes · Native Rigmark run summaries",
             fontsize=12, color=MUTED)
    handles = [
        Patch(color=BLUE, label="11 Sep 2026 · baseline"),
        Patch(color=TEAL, label="19 Sep 2026 · baseline"),
        Line2D([], [], color=GOLD, marker="D", linestyle="none", label="19 Sep 2026 · IaC reproduction"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.028, 0.847),
               ncols=3, frameon=False, fontsize=11, columnspacing=2.3, labelcolor=INK)
    for ax, (rows, heading, xlabel, seconds) in zip(axes, panels):
        panel(ax, rows, series, heading, xlabel, seconds=seconds)
    fig.text(0.035, 0.115, "Bars and labels = baseline medians, matching the README table · 11 Sep: 3 runs · 19 Sep: 2 accepted runs",
             fontsize=10.5, color=MUTED)
    fig.text(0.035, 0.07, "Gold diamonds = the separate IaC reproduction, excluded from baseline medians.",
             fontsize=10.5, color=MUTED)
    save_figure(fig, output_dir, name)


def save_figure(fig, output_dir, name):
    import matplotlib.pyplot as plt

    fig.savefig(output_dir / f"{name}.png", dpi=160, facecolor="white", metadata={"Software": "Matplotlib"})
    svg = output_dir / f"{name}.svg"
    fig.savefig(svg, facecolor="white", metadata={"Date": None, "Creator": "Matplotlib"})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def context_figure(output_dir, series):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    probes = json.loads((ROOT / CONTEXT_SOURCE).read_text())
    contexts = probes["context_tokens"]
    if contexts != [8192, 32768, 65536] or probes["completion_tokens_per_request"] != 8:
        raise ValueError("Expected 8-token decode probes at 8K, 32K and 64K")
    runs = probes["runs"]
    if [run["kind"] for run in runs] != ["accepted", "accepted", "iac_reproduction"]:
        raise ValueError("Expected two accepted decode-probe runs followed by the IaC reproduction")
    if any(run["date"] != "2026-09-19" for run in runs):
        raise ValueError("Expected September 19 decode probes")
    if any(len(run["decode_tokens_per_second"][str(context)]) != 3 for run in runs for context in contexts):
        raise ValueError("Expected three cold requests per context per run")
    decode = []
    for context in contexts:
        per_run = [median(run["decode_tokens_per_second"][str(context)]) for run in runs]
        decode.append((None, median(per_run[:2]), per_run[2]))
    prefill = [series[key] for key, _ in COLD]

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 8), facecolor="white")
    fig.subplots_adjust(left=0.085, right=0.965, bottom=0.28, top=0.70, wspace=0.25)
    fig.text(0.035, 0.94, "Throughput versus context length", fontsize=23, fontweight="bold", color=INK)
    fig.text(0.035, 0.89, "GLM-5.3-Flash · TP4 / four GB10 nodes · Cold prefill requests from native Rigmark runs",
             fontsize=12, color=MUTED)
    handles = [
        Line2D([], [], color=BLUE, linewidth=2.4, label="11 Sep 2026 · median of three runs"),
        Line2D([], [], color=TEAL, linewidth=2.4, label="19 Sep 2026 · median of two accepted runs"),
        Line2D([], [], color=GOLD, marker="D", linestyle="none", label="Separate IaC reproduction"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.028, 0.85),
               ncols=3, frameon=False, fontsize=10.5, columnspacing=1.6, labelcolor=INK)
    for ax, measurements, title in zip(axes, (prefill, decode),
                                 ("Prefill · effective input tok/s", "Decode · 19 Sep, 8-token probe")):
        for group, color in enumerate((BLUE, TEAL)):
            medians = [row[group] for row in measurements]
            if medians[0] is None:
                continue
            ax.plot(contexts, medians, color=color, linewidth=2.4, marker="o", markersize=6, zorder=3)
            for context, row, center in zip(contexts, measurements, medians):
                if group == 1:
                    ax.scatter(context, row[2], s=52, marker="D", color=GOLD,
                               edgecolors="white", linewidths=0.9, zorder=4)
                label = f"{Decimal(str(center)):,.1f}"
                label_y = max(center, row[2]) if group == 1 else center
                ax.annotate(label, (context, label_y), xytext=(0, 12 if group == 1 else -21),
                            textcoords="offset points", ha="center", fontsize=10.5, color=color)
        ax.set_title(title, loc="left", fontsize=15, fontweight="bold", color=INK, pad=22)
        ax.set_xticks(contexts, ["8K", "32K", "64K"])
        ax.set_xlim(0, 73728)
        ax.set_xlabel("Input context length (tokens; K = 1,024)", fontsize=11, color=MUTED, labelpad=12)
        ax.set_ylabel("Tokens per second", fontsize=11, color=MUTED, labelpad=10)
        ax.set_ylim(0, max(value for row in measurements for value in row if value is not None) * 1.22)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
        ax.tick_params(axis="both", length=0, labelcolor=MUTED, pad=9)
        ax.grid(color="#dfe5ed", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.text(0.035, 0.17, "Lines and labels = baseline medians · 11 Sep: 3 runs · 19 Sep: 2 accepted runs",
             fontsize=10.5, color=MUTED)
    fig.text(0.035, 0.12, "Decode: 8 output tokens, 7 / first-to-last output time. Diagnostic only; no sustained decode or initial-baseline curve available.",
             fontsize=10.5, color=INK)
    fig.text(0.035, 0.075, "Prefill medians match the README table. Gold diamonds show the separate IaC run, excluded from baseline medians.",
             fontsize=10.5, color=MUTED)
    save_figure(fig, output_dir, "context-throughput")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/plots")
    args = parser.parse_args()
    series = read_series()
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                                "svg.hashsalt": "tp4-rigmark-comparison", "svg.fonttype": "path"})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure(args.output_dir, "generation-comparison", "Generation throughput and first-token latency", [
        (GENERATION, "Throughput · higher is better", "Tokens per second", False),
        (LATENCY, "Time to first token · lower is better", "Seconds", True),
    ], series)
    figure(args.output_dir, "prefill-comparison", "Long-context prefill and prefix-cache reuse", [
        (COLD, "Cold prefill · higher is better", "Effective prefill tokens per second", False),
        (REPLAY, "Immediate replay · higher is better", "Effective prefill tokens per second", False),
    ], series)
    context_figure(args.output_dir, series)
    print(f"Rendered three figures from 16 summary metrics and 27 short decode probes to {args.output_dir}")


if __name__ == "__main__":
    main()
