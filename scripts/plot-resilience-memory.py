#!/usr/bin/env python3
"""Plot free memory over the September 29 mixed load; requires matplotlib==3.11.2.

Reads the portable resilience CSV and checks it against its summary hash. The time
axis uses each node's own monotonic clock; request counts on the tick labels join
controller and node wall clocks that were not calibrated and are approximate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "docs/historical_benchmarks/experiments/2026-09-29-sparkcache-resilience/portable-data"
CSV_PATH = DATA_DIR / "requests-memory.csv"
SUMMARY_PATH = DATA_DIR / "summary.json"
OUTPUT_DIR = ROOT / "docs/plots/experiments/2026-09-29-sparkcache-resilience"
NAME = "memory-over-time"
TEAL, INK, MUTED, GRID = "#087f74", "#172b46", "#536478", "#dfe5ed"
VIOLET = "#6b4fc9"
GIB = 2**30
TICK_MINUTES = (0, 20, 40, 60, 80)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load():
    summary = json.loads(SUMMARY_PATH.read_text())
    if sha256_file(CSV_PATH) != summary["derived_csv"]["sha256"]:
        raise SystemExit(f"{CSV_PATH}: SHA-256 differs from {SUMMARY_PATH.name}")
    first = {}
    bins = {group: defaultdict(lambda: [float("inf"), float("-inf")]) for group in (0, 1)}
    rank0_requests = []
    final_drain = {}
    with CSV_PATH.open(newline="") as stream:
        for row in csv.DictReader(stream):
            rank = int(row["rank"])
            first.setdefault(rank, int(row["monotonic_ns"]))
            if row["mem_available_known"] != "True":
                continue
            minutes = (int(row["monotonic_ns"]) - first[rank]) / 60e9
            gib = int(row["mem_available_bytes"]) / GIB
            low_high = bins[0 if rank == 0 else 1][int(minutes)]
            low_high[0] = min(low_high[0], gib)
            low_high[1] = max(low_high[1], gib)
            if rank == 0:
                rank0_requests.append((minutes, int(row["cumulative_generation_requests_submitted"])))
            if row["sample_class"] == "final_drain":
                final_drain[rank] = gib
    if sorted(first) != [0, 1, 2, 3] or sorted(final_drain) != [0, 1, 2, 3]:
        raise SystemExit(f"{CSV_PATH}: expected samples and a final drain for ranks 0-3")
    total = summary["request_counts"]["total_generation_requests_submitted"]
    if max(count for _, count in rank0_requests) != total:
        raise SystemExit(f"{CSV_PATH}: request count differs from {SUMMARY_PATH.name}")
    return bins, rank0_requests, final_drain, summary


def requests_at(rank0_requests, minute):
    count = 0
    for minutes, submitted in rank0_requests:
        if minutes > minute:
            break
        count = submitted
    return count


def plot(output_dir, bins, rank0_requests, final_drain, summary):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    duration = summary["source_elapsed_seconds"] / 60
    total = summary["request_counts"]["total_generation_requests_submitted"]
    cancels = summary["request_counts"]["cancellation_requests_submitted"]
    fig = plt.figure(figsize=(14, 7.8), facecolor="white")
    ax = fig.add_axes([0.075, 0.25, 0.755, 0.52])
    fig.text(0.035, 0.935, f"Free memory did not trend down over {duration:.0f} minutes and {total} requests",
             fontsize=20, fontweight="bold", color=INK)
    fig.text(0.035, 0.885, "Free memory (MemAvailable) on each node · lowest and highest sample per minute · "
             "higher is safer · 0 GiB = no free memory left", fontsize=11.5, color=MUTED)

    series = ((0, TEAL, "Rank 0 (API server + engine)"), (1, VIOLET, "Ranks 1–3 (engine)"))
    for group, color, _ in series:
        minutes = sorted(bins[group])
        x = [min(minute + 0.5, duration) for minute in minutes]
        low = [bins[group][minute][0] for minute in minutes]
        high = [bins[group][minute][1] for minute in minutes]
        ax.fill_between(x, low, high, color=color, alpha=0.16, linewidth=0, zorder=2)
        ax.plot(x, low, color=color, linewidth=2, zorder=3)

    rank0 = bins[0]
    lowest_minute = min(rank0, key=lambda minute: rank0[minute][0])
    lowest = rank0[lowest_minute][0]
    ax.annotate(f"Lowest: {lowest:.2f} GiB free", (lowest_minute + 0.5, lowest), xytext=(0, -26),
                textcoords="offset points", ha="left", fontsize=11, color=INK,
                arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 1})
    others = [final_drain[rank] for rank in (1, 2, 3)]
    end_x = duration
    ax.text(end_x + 1.2, final_drain[0], f"Rank 0\n{final_drain[0]:.1f} GiB at the end",
            fontsize=10.5, color=INK, va="center")
    ax.text(end_x + 1.2, sum(others) / 3, f"Ranks 1–3\n{min(others):.1f}–{max(others):.1f} GiB at the end",
            fontsize=10.5, color=INK, va="center")
    ax.text(0.5, 0.25, "0 GiB = no free memory left", fontsize=10, color=MUTED, va="bottom")

    ticks = list(TICK_MINUTES) + [duration]
    labels = [f"{minute:.0f} min\n≈{requests_at(rank0_requests, minute)} requests" for minute in TICK_MINUTES]
    labels[0] = "0 min\n0 requests"
    labels.append(f"{duration:.0f} min\n{total} requests")
    ax.set_xticks(ticks, labels, fontsize=10.5, color=MUTED)
    ax.set_xlim(0, duration)
    ax.set_ylim(0, 12)
    ax.set_yticks(range(0, 13, 2))
    ax.set_ylabel("Free memory, GiB", fontsize=11, color=MUTED, labelpad=10)
    ax.tick_params(axis="both", length=0, labelcolor=MUTED, pad=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=1)
    ax.axhline(0, color=INK, linewidth=1.2, zorder=4)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)

    handles = [Line2D([], [], color=color, linewidth=2, label=f"{label} · lowest per minute")
               for _, color, label in series]
    handles.append(Patch(color=MUTED, alpha=0.2, label="Range within each minute"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.03, 0.865), ncols=3,
               frameon=False, fontsize=11, labelcolor=INK, columnspacing=3)
    notes = (
        f"Load: one request at a time, with conversations growing to 64K tokens, cached replays and {cancels} "
        f"cancellations. The operator ended it after {duration:.0f} of 120 planned minutes.",
        "Sampled every second (every 200 ms near the end). Request counts join controller and node clocks "
        "that were not calibrated, so they are approximate.",
        "Five clients at the full 256K context were tested separately. Both rises in free memory are "
        "observed, not explained; this run does not prove the absence of leaks.",
    )
    for index, note in enumerate(notes):
        fig.text(0.035, 0.105 - 0.035 * index, note, fontsize=10, color=MUTED)

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
    bins, rank0_requests, final_drain, summary = load()
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                                "svg.hashsalt": "tp4-resilience-memory", "svg.fonttype": "path"})
    plot(args.output_dir, bins, rank0_requests, final_drain, summary)
    print(f"Rendered {NAME}.png and {NAME}.svg to {args.output_dir}")


if __name__ == "__main__":
    main()
