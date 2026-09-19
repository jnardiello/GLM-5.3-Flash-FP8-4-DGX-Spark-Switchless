#!/usr/bin/env python3
"""Plot the public Rigmark records; requires matplotlib==3.11.2.

Reads public JSON records only. The September 19 chart series contains its two
accepted runs plus the separate IaC reproduction; it does not replace a baseline.
"""

from __future__ import annotations

import argparse
import json
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
        series[key] = (old["per_run"], new["per_run"] + [replay["reproduction_single_run_value"]])
    return series


def panel(ax, rows, series, title, xlabel, *, logarithmic=False):
    from matplotlib.ticker import FuncFormatter, MaxNLocator, NullLocator

    values = []
    for row_index, (key, _) in enumerate(rows):
        if row_index % 2 == 0:
            ax.axhspan(row_index - 0.47, row_index + 0.47, color="#f2f5f9", zorder=0)
        for group, (runs, color) in enumerate(zip(series[key], (BLUE, TEAL))):
            y = row_index + (-0.18 if group == 0 else 0.18)
            values.extend(runs)
            ax.hlines(y, min(runs), max(runs), color=color, linewidth=2, alpha=0.7, zorder=2)
            ax.vlines(median(runs), y - 0.12, y + 0.12, color=color, linewidth=2.6, zorder=3)
            for run_index, (value, offset) in enumerate(zip(runs, (-0.07, 0, 0.07))):
                iac = group == 1 and run_index == 2
                ax.scatter(value, y + offset, s=46, marker="D" if iac else "o",
                           color=GOLD if iac else color, edgecolors="white", linewidths=0.8, zorder=4)
                if logarithmic and value > 2:
                    ax.annotate(f"{value:.3f} s", (value, y + offset), xytext=(-5, 9),
                                textcoords="offset points", ha="right", color=color, fontsize=10)
            label = f"{median(runs):.3f}" if logarithmic else f"{median(runs):,.2f}"
            ax.text(1.025, y, label, transform=ax.get_yaxis_transform(), va="center",
                    color=color, fontsize=10.5)
    ax.set_title(title, loc="left", fontsize=15, color=INK, fontweight="bold", pad=20)
    ax.text(1.025, 1.04, "Median", transform=ax.transAxes, color=MUTED, fontsize=10)
    ax.set_yticks(range(len(rows)), [label for _, label in rows], fontsize=11, color=INK)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(min(values) * 0.88, max(values) * 1.15)
    if logarithmic:
        ax.set_xscale("log", base=2)
        ax.set_xlim(0.25, 8)
        ax.set_xticks([0.25, 0.5, 1, 2, 4, 8])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    else:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.set_xlabel(xlabel, fontsize=11, color=MUTED, labelpad=12)
    ax.tick_params(axis="both", length=0, labelcolor=MUTED, pad=9)
    ax.grid(axis="x", color="#dfe5ed", linewidth=0.8, zorder=1)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)


def figure(output_dir, name, title, panels, series):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(14.5, 7.4), facecolor="white")
    axes = [fig.add_axes([0.15, 0.22, 0.29, 0.48]), fig.add_axes([0.64, 0.22, 0.27, 0.48])]
    fig.text(0.035, 0.935, title, fontsize=23, fontweight="bold", color=INK)
    fig.text(0.035, 0.885, "GLM-5.3-Flash · TP4 / four GB10 nodes · Native Rigmark run summaries",
             fontsize=12, color=MUTED)
    handles = [
        Line2D([], [], color=BLUE, marker="o", linewidth=2, label="11 Sep 2026 · three runs"),
        Line2D([], [], color=TEAL, marker="o", linewidth=2, label="19 Sep 2026 · two accepted runs"),
        Line2D([], [], color=GOLD, marker="D", linestyle="none", label="19 Sep 2026 · IaC reproduction"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.028, 0.847),
               ncols=3, frameon=False, fontsize=11, columnspacing=2.3, labelcolor=INK)
    for ax, (rows, heading, xlabel, logarithmic) in zip(axes, panels):
        panel(ax, rows, series, heading, xlabel, logarithmic=logarithmic)
    fig.text(0.035, 0.115, "Points = individual runs · Lines = observed min–max · Vertical ticks = median of the three plotted values",
             fontsize=10.5, color=MUTED)
    fig.text(0.035, 0.07, "19 Sep combines two accepted runs and one separate deployment reproduction. The frozen two-run baseline is unchanged.",
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
    decode = [None, [
        [median(run["decode_tokens_per_second"][str(context)]) for run in runs]
        for context in contexts
    ]]
    prefill = [[series[key][group] for key, _ in COLD] for group in range(2)]

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 8), facecolor="white")
    fig.subplots_adjust(left=0.085, right=0.965, bottom=0.28, top=0.70, wspace=0.25)
    fig.text(0.035, 0.94, "Throughput versus context length", fontsize=23, fontweight="bold", color=INK)
    fig.text(0.035, 0.89, "GLM-5.3-Flash · TP4 / four GB10 nodes · Cold prefill requests from native Rigmark runs",
             fontsize=12, color=MUTED)
    handles = [
        Line2D([], [], color=BLUE, linewidth=2.4, label="11 Sep 2026 · median of three runs"),
        Line2D([], [], color=TEAL, linewidth=2.4, label="19 Sep 2026 · median of three available runs"),
        Line2D([], [], color=GOLD, marker="D", linestyle="none", label="Separate IaC reproduction"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.028, 0.85),
               ncols=3, frameon=False, fontsize=10.5, columnspacing=1.6, labelcolor=INK)
    for ax, groups, title in zip(axes, (prefill, decode),
                                 ("Prefill · effective input tok/s", "Decode · 19 Sep, 8-token probe")):
        for group, (values, color) in enumerate(zip(groups, (BLUE, TEAL))):
            if values is None:
                continue
            medians = [median(runs) for runs in values]
            ax.fill_between(contexts, [min(runs) for runs in values], [max(runs) for runs in values],
                            color=color, alpha=0.12, linewidth=0)
            ax.plot(contexts, medians, color=color, linewidth=2.4, zorder=3)
            for context, runs, center in zip(contexts, values, medians):
                for run_index, value in enumerate(runs):
                    iac = group == 1 and run_index == 2
                    ax.scatter(context, value, s=48, marker="D" if iac else "o",
                               color=GOLD if iac else color, edgecolors="white", linewidths=0.8, zorder=4)
                label = f"{center:,.0f}" if ax is axes[0] else f"{center:.1f}"
                label_y = max(runs) if group == 1 else min(runs)
                ax.annotate(label, (context, label_y), xytext=(0, 12 if group == 1 else -21),
                            textcoords="offset points", ha="center", fontsize=10.5, color=color)
        ax.set_title(title, loc="left", fontsize=15, fontweight="bold", color=INK, pad=22)
        ax.set_xticks(contexts, ["8K", "32K", "64K"])
        ax.set_xlim(0, 73728)
        ax.set_xlabel("Input context length (tokens; K = 1,024)", fontsize=11, color=MUTED, labelpad=12)
        ax.set_ylabel("Tokens per second", fontsize=11, color=MUTED, labelpad=10)
        ax.set_ylim(0, max(value for values in groups if values is not None for runs in values for value in runs) * 1.22)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
        ax.tick_params(axis="both", length=0, labelcolor=MUTED, pad=9)
        ax.grid(color="#dfe5ed", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.text(0.035, 0.17, "Points = per-run medians of 3 requests · Lines = medians of 3 runs · Shading = observed run range",
             fontsize=10.5, color=MUTED)
    fig.text(0.035, 0.12, "Decode: 8 output tokens, 7 / first-to-last output time. Diagnostic only; no sustained decode or initial-baseline curve available.",
             fontsize=10.5, color=INK)
    fig.text(0.035, 0.075, "19 Sep combines two accepted runs and one separate IaC reproduction. Frozen baseline records remain unchanged.",
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
        (LATENCY, "Time to first token · lower is better", "Seconds · logarithmic scale", True),
    ], series)
    figure(args.output_dir, "prefill-comparison", "Long-context prefill and prefix-cache reuse", [
        (COLD, "Cold prefill · higher is better", "Effective prefill tokens per second", False),
        (REPLAY, "Immediate replay · higher is better", "Effective prefill tokens per second", False),
    ], series)
    context_figure(args.output_dir, series)
    print(f"Rendered three figures from 16 summary metrics and 27 short decode probes to {args.output_dir}")


if __name__ == "__main__":
    main()
