#!/usr/bin/env python3
"""Plot the lowest available memory in each resilience test; requires matplotlib==3.11.2.

Reads the per-checkpoint MemAvailable minima of the September 29 campaign from its
portable results. Each row is one KV14 checkpoint: rank 0 as a point, the three other
ranks as the range of their individual minima. No cluster requests are sent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "docs/historical_benchmarks/experiments/2026-09-29-sparkcache-resilience/results.json"
OUTPUT_DIR = ROOT / "docs/plots/experiments/2026-09-29-sparkcache-resilience"
NAME = "memory-by-test"
TEAL, INK, MUTED, GRID = "#087f74", "#172b46", "#536478", "#dfe5ed"
GIB = 2**30
# Fixed row order, top to bottom. † marks checkpoints before bounded API admission.
CHECKPOINTS = (
    ("kv14_max_context_checkpoint_1", "Full context, one client: cold + replay †"),
    ("kv14_queue_cancellation_checkpoint_1", "Queues + cancellations: one wave failed †"),
    ("kv14_bounded_admission_checkpoint_1", "150-client bursts + full-context check"),
    ("kv14_c5_max_context_cold_checkpoint_1", "Five clients at full context, 16-token output"),
    ("kv14_c5_long_decode_checkpoint_1", "Five clients at full context, 16,384-token output"),
    ("kv14_bounded_targeted_checkpoint_1", "Queues, slow clients, API and cache faults"),
    ("kv14_first_mixed_soak_failure_1", "Mixed load, 40 min: idle check failed"),
    ("kv14_owner_stopped_mixed_soak_2", "Mixed load, 99 min: stopped early"),
)


def rank_minima(key, checkpoint):
    """Return the four per-rank minima in GiB from any of the three checkpoint schemas."""
    observation = checkpoint.get("resource_observation") or checkpoint.get("telemetry") or {}
    if "ranks" in observation:
        by_rank = {row["rank"]: row["mem_available_min_bytes"] for row in observation["ranks"]}
        values = [by_rank.get(rank) for rank in range(4)]
    else:
        values = observation.get("mem_available_min_bytes_by_rank") or []
    if len(values) != 4 or any(not isinstance(value, int) or value <= 0 for value in values):
        raise SystemExit(f"{RESULTS_PATH}: {key} lacks four positive MemAvailable minima")
    return [value / GIB for value in values]


def load():
    results = json.loads(RESULTS_PATH.read_text())
    rows = []
    for key, label in CHECKPOINTS:
        if key not in results:
            raise SystemExit(f"{RESULTS_PATH}: missing checkpoint {key}")
        rows.append((label, rank_minima(key, results[key])))
    guard = results["limits"]["stop_mem_available_bytes"] / GIB
    return rows, guard


def plot(output_dir, rows, guard):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(14, 8.2), facecolor="white")
    ax = fig.add_axes([0.345, 0.25, 0.62, 0.55])
    fig.text(0.035, 0.935, "Minimum available memory during resilience tests",
             fontsize=20, fontweight="bold", color=INK)
    fig.text(0.035, 0.89, "Four nodes · 14 GiB KV cache per node · lowest sampled MemAvailable in each "
             "test checkpoint · higher is safer", fontsize=11.5, color=MUTED)

    for index, (_, minima) in enumerate(rows):
        low, high = min(minima[1:]), max(minima[1:])
        ax.plot([low, high], [index, index], color=INK, linewidth=2, solid_capstyle="butt", zorder=3)
        for value in (low, high):
            ax.plot([value, value], [index - 0.16, index + 0.16], color=INK, linewidth=2, zorder=3)
        ax.annotate(f"{low:.2f}–{high:.2f}", (high, index), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=10.5, color=INK)
        ax.scatter([minima[0]], [index], s=90, color=TEAL, edgecolors="white", linewidths=1.5, zorder=4)
        ax.annotate(f"{minima[0]:.2f}", (minima[0], index), xytext=(0, 11), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10.5, color=TEAL, fontweight="bold")

    ax.axvline(guard, color=MUTED, linewidth=1.3, linestyle=(0, (4, 3)), zorder=2)
    ax.text(guard + 0.12, len(rows) - 0.3, f"Test stop guard: {guard * 1024:.0f} MiB",
            fontsize=10, color=MUTED, va="center")
    ax.set_yticks(range(len(rows)), [label for label, _ in rows], fontsize=11, color=INK)
    ax.set_ylim(len(rows) + 0.05, -0.7)
    ax.set_xlim(0, 11)
    ax.set_xticks(range(0, 12))
    ax.set_xlabel("Lowest sampled available memory (GiB)", fontsize=11, color=MUTED, labelpad=10)
    ax.tick_params(axis="both", length=0, labelcolor=MUTED, pad=8)
    ax.tick_params(axis="y", labelcolor=INK)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=1)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)

    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=9, markerfacecolor=TEAL,
               markeredgecolor="white", label="Rank 0 (API server + engine)"),
        Line2D([], [], color=INK, linewidth=2, label="Ranks 1–3: range of their individual minima"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.03, 0.87), ncols=2,
               frameon=False, fontsize=11, labelcolor=INK, columnspacing=3)
    notes = (
        "Each value is the lowest sample over a whole test checkpoint. Workloads, durations, sampling cadence "
        "and configurations differ, so rows are not controlled comparisons.",
        "Five clients overlapped, which does not mean five fully resident contexts. Bursts included the expected "
        "HTTP 503 rejections. † Before bounded API admission was added.",
        "Mixed loads sent one request at a time, up to 64K context; neither completed the planned 120 minutes. "
        "Sampled minima do not prove the absence of leaks.",
    )
    for index, note in enumerate(notes):
        fig.text(0.035, 0.115 - 0.035 * index, note, fontsize=9.5, color=MUTED)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{NAME}.png", dpi=160, facecolor="white", metadata={"Software": "Matplotlib"})
    svg = output_dir / f"{NAME}.svg"
    fig.savefig(svg, facecolor="white", metadata={"Date": None, "Creator": "Matplotlib"})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    rows, guard = load()
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                                "svg.hashsalt": "tp4-resilience-memory", "svg.fonttype": "path"})
    plot(args.output_dir, rows, guard)
    print(f"Rendered {NAME}.png and {NAME}.svg from {len(rows)} checkpoints to {args.output_dir}")


if __name__ == "__main__":
    main()
