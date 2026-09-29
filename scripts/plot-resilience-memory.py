#!/usr/bin/env python3
"""Plot the lowest free memory on the busiest node in each resilience test; requires matplotlib==3.11.2.

Reads the per-checkpoint MemAvailable minima of the September 29 campaign from its
portable results. Each bar is one KV14 checkpoint on rank 0, which also runs the API;
the footnote states the lowest value on the other three ranks. No cluster requests
are sent.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "docs/historical_benchmarks/experiments/2026-09-29-sparkcache-resilience/results.json"
OUTPUT_DIR = ROOT / "docs/plots/experiments/2026-09-29-sparkcache-resilience"
NAME = "memory-by-test"
TEAL, INK, MUTED, GRID = "#087f74", "#172b46", "#536478", "#dfe5ed"
STOP_FILL, STOP_EDGE = "#fbe4e2", "#b42318"
GIB = 2**30
# Fixed row order, top to bottom: (checkpoint, plain label, muted note). † marks the
# checkpoints that ran before bounded API admission.
CHECKPOINTS = (
    ("kv14_max_context_checkpoint_1", "One user, longest conversation (256K tokens) †", None),
    ("kv14_c5_max_context_cold_checkpoint_1", "Five users, longest conversations, short answers", None),
    ("kv14_c5_long_decode_checkpoint_1", "Five users, longest conversations, long answers", None),
    ("kv14_bounded_admission_checkpoint_1", "150 requests at once (extra ones turned away)", None),
    ("kv14_queue_cancellation_checkpoint_1", "Queued and cancelled requests †", "one run stopped by a monitoring check"),
    ("kv14_bounded_targeted_checkpoint_1", "Slow clients, bad requests, cache failures", None),
    ("kv14_first_mixed_soak_failure_1", "Mixed use, 40 minutes", "an end-of-run check failed"),
    ("kv14_owner_stopped_mixed_soak_2", "Mixed use, 99 minutes", "ended early by us"),
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
    for key, label, note in CHECKPOINTS:
        if key not in results:
            raise SystemExit(f"{RESULTS_PATH}: missing checkpoint {key}")
        rows.append((label, note, rank_minima(key, results[key])))
    guard = results["limits"]["stop_mem_available_bytes"] / GIB
    return rows, guard


def plot(output_dir, rows, guard):
    import matplotlib.pyplot as plt
    from matplotlib.transforms import blended_transform_factory

    fig = plt.figure(figsize=(14, 8), facecolor="white")
    ax = fig.add_axes([0.36, 0.22, 0.58, 0.6])
    fig.text(0.035, 0.935, "Free memory left on the busiest node during each stress test",
             fontsize=20, fontweight="bold", color=INK)
    fig.text(0.035, 0.89, "Node 1 of 4 also runs the API · lowest value seen in each test · more is safer",
             fontsize=12, color=MUTED)

    ax.axvspan(0, guard, color=STOP_FILL, zorder=1, linewidth=0)
    ax.axvline(guard, color=STOP_EDGE, linewidth=1.4, linestyle=(0, (4, 3)), zorder=2)
    ax.text(guard + 0.08, -0.75, f"Safety stop ({guard:.2f} GiB): tests end automatically below this line",
            fontsize=10.5, color=STOP_EDGE, va="center")

    labels = blended_transform_factory(ax.transAxes, ax.transData)
    for index, (label, note, minima) in enumerate(rows):
        value = minima[0]
        ax.barh(index, value, height=0.62, color=TEAL, zorder=3)
        ax.annotate(f"{value:.1f} GiB", (value, index), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=13, fontweight="bold", color=TEAL)
        ax.text(-0.02, index - (0.13 if note else 0), label, transform=labels, ha="right",
                va="center", fontsize=12, color=INK)
        if note:
            ax.text(-0.02, index + 0.26, note, transform=labels, ha="right", va="center",
                    fontsize=10, color=MUTED, style="italic")

    ax.set_yticks([])
    ax.set_ylim(len(rows) - 0.4, -1.05)
    ax.set_xlim(0, 6)
    ax.set_xticks(range(0, 7), [f"{tick} GiB" for tick in range(0, 7)])
    ax.tick_params(axis="x", length=0, labelcolor=MUTED, labelsize=11, pad=8)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)

    others = math.floor(min(value for _, _, minima in rows for value in minima[1:]) * 10) / 10
    notes = (
        f"The other three nodes always kept at least {others:.1f} GiB free. None of the noted stops was caused "
        "by low memory.",
        "Each bar is the lowest value in one test. Tests differ in load and length, so the bars are not a ranking.",
        "† Run before the limit on simultaneous requests was added.",
    )
    for index, note in enumerate(notes):
        fig.text(0.035, 0.11 - 0.035 * index, note, fontsize=10.5, color=MUTED)

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
