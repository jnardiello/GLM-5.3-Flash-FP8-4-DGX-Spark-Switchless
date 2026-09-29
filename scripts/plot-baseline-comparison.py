#!/usr/bin/env python3
"""Compare two frozen Rigmark measurements; requires matplotlib==3.11.2.

By default, compares E31 with the E29-equivalent arm measured on the same load with
the same upstream Rigmark flags; both come from the E31 record. `--comparison
e29-vs-e28b` reproduces the archived E29 versus E28b figures (old protocol).
Reads all 16 saved metrics. No benchmark requests are sent.
"""

from __future__ import annotations

import argparse
from datetime import date
from decimal import Decimal
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E31_BASELINE = ROOT / "docs/historical_benchmarks/baselines/2026-09-28-e31/baseline.json"
E29_BASELINE = ROOT / "docs/historical_benchmarks/baselines/2026-09-25-e29/baseline.json"
E28B_BASELINE = ROOT / "docs/historical_benchmarks/baselines/2026-09-25-e28b/baseline.json"
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

COMPARISONS = {
    "e31-same-load": {
        "output_dir": ROOT / "docs/plots/comparisons/2026-09-28-e31-vs-e29-same-load",
        "titles": ("Generation: E31 vs the same-load E29-equivalent arm",
                   "Prefill: E31 vs the same-load E29-equivalent arm"),
        "subtitle": ("GLM-5.3-Flash · Four GB10 nodes · E31 speculative-safe tail ring vs E29 arithmetic · "
                     "Upstream Rigmark, Alex Ellis's reference flags"),
        "delta_heading": "Δ vs E29 arm",
        "approx_unchanged": None,
        # Format labels like the README table, which rounds the binary float value.
        "float_labels": True,
        "footer": ("Δ = (E31 / same-load arm − 1) × 100, calculated from unrounded values.",
                   "Each arm is one complete suite (n = 1) on the same load, so differences of a few percent "
                   "are within noise."),
        "prefill_note": ("Cold prefill is isolated by a fresh comparison ID per suite; replay reuses the prefix "
                         "within each suite. K = 1,024 tokens."),
    },
    "e29-vs-e28b": {
        "output_dir": ROOT / "docs/plots/comparisons/2026-09-25-e29-vs-2026-09-25-e28b",
        "titles": ("Generation: current vs previous baseline", "Prefill: current vs previous baseline"),
        "subtitle": "GLM-5.3-Flash · Four GB10 nodes · E29 end-drain and idle coalescing over E28b · Native Rigmark",
        "delta_heading": "Δ vs previous",
        "approx_unchanged": APPROX_UNCHANGED,
        "float_labels": False,
        "footer": ("Δ = (current / previous − 1) × 100, calculated from unrounded frozen medians.",
                   "≈ unchanged: owner-accepted changes of roughly 1–2%; this is not a statistical test."),
        "prefill_note": "Cold prefill uses a fresh cache salt; replay reuses the prefix within each run. K = 1,024 tokens.",
    },
}


def checked_values(path, values):
    expected = {key for key, _ in GENERATION + LATENCY + COLD + REPLAY}
    if len(values) != 16 or set(values) != expected:
        raise ValueError(f"{path}: the frozen record must provide all 16 metrics")
    if any(not value.is_finite() or value <= 0 for value in values.values()):
        raise ValueError(f"{path}: frozen values must be finite and positive")
    return values


def read_baseline(path, label):
    baseline = json.loads(path.read_text(), parse_float=Decimal)
    metrics = baseline["performance"]["metrics"]
    if len(metrics) != 16:
        raise ValueError(f"{path}: the frozen baseline must provide all 16 metrics")
    medians = checked_values(path, {row["key"]: Decimal(row["median"]) for row in metrics})
    measured_on = date.fromisoformat(baseline["measured_on"]).strftime("%d/%m/%Y")
    runs = baseline["performance"]["included_run_count"]
    requests = baseline["functional"]["measured_requests"]["count"]
    caption = f"{label} · {measured_on}\n{runs} accepted runs · {requests} requests"
    return medians, caption


def read_same_load(path):
    """Return E31's values and its same-load reference arm, with legend captions."""
    record = json.loads(path.read_text(), parse_float=Decimal)
    metrics = record["performance"]["metrics"]
    if len(metrics) != 16:
        raise ValueError(f"{path}: the frozen record must provide all 16 metrics")
    current = checked_values(path, {row["key"]: Decimal(row["median"]) for row in metrics})
    runs = {row["same_load_reference"]["run"] for row in metrics}
    if len(runs) != 1:
        raise ValueError(f"{path}: every metric must use one same-load reference run")
    reference = checked_values(path, {row["key"]: Decimal(row["same_load_reference"]["value"])
                                      for row in metrics})
    extract = record["receipts"][runs.pop()]["portable_extract"]
    extract_path = ROOT / extract["path"]
    if hashlib.sha256(extract_path.read_bytes()).hexdigest() != extract["sha256"]:
        raise ValueError(f"{extract_path}: SHA-256 differs from {path}")
    reference_requests = json.loads(extract_path.read_text())["request_count"]
    measured_on = date.fromisoformat(record["measured_on"]).strftime("%d/%m/%Y")
    suites = record["performance"]["included_run_count"]
    requests = record["functional"]["measured_requests"]["count"]
    captions = (f"E29-equivalent arm, same load · {measured_on}\n1 suite · {reference_requests} requests",
                f"E31 · {measured_on}\n{suites} suite · {requests} requests")
    return current, reference, captions


def panel(ax, rows, current, previous, title, xlabel, profile, *, seconds=False):
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
            value = float(medians[key]) if profile["float_labels"] else medians[key]
            label = f"{value:,.{precision}f}"
            ax.annotate(label, (values[index], positions[index]), xytext=(7, 0),
                        textcoords="offset points", va="center", fontsize=11.5,
                        color=MUTED if offset < 0 else TEAL)

    ax.text(1.21, 1.04, profile["delta_heading"], transform=ax.transAxes,
            ha="center", va="bottom", fontsize=11.5, color=INK, fontweight="bold")
    for index, (key, _) in enumerate(rows):
        delta = (current[key] / previous[key] - 1) * 100
        threshold = profile["approx_unchanged"]
        unchanged = threshold is not None and abs(delta) <= threshold
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


def figure(output_dir, name, title, panels, current, previous, captions, note, profile):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    height = 9.4 if name == "generation" else 7.8
    fig = plt.figure(figsize=(17.2, height), facecolor="white")
    axes = [fig.add_axes([0.145, 0.23, 0.26, 0.47]), fig.add_axes([0.650, 0.23, 0.255, 0.47])]
    fig.text(0.035, 0.935, title, fontsize=23, fontweight="bold", color=INK)
    fig.text(0.035, 0.88, profile["subtitle"], fontsize=12, color=MUTED)
    fig.legend(handles=[Patch(color=PREVIOUS_COLOR, label=captions[0]),
                        Patch(color=TEAL, label=captions[1])],
               loc="upper left", bbox_to_anchor=(0.03, 0.845), ncols=2,
               frameon=False, fontsize=12, labelcolor=INK, columnspacing=4)
    for ax, (rows, heading, xlabel, seconds) in zip(axes, panels):
        panel(ax, rows, current, previous, heading, xlabel, profile, seconds=seconds)
    fig.text(0.035, 0.11, profile["footer"][0], fontsize=10.5, color=MUTED)
    fig.text(0.035, 0.073, profile["footer"][1], fontsize=10.5, color=MUTED)
    fig.text(0.035, 0.036, note, fontsize=10.5, color=MUTED)
    fig.savefig(output_dir / f"{name}.png", dpi=160, facecolor="white", metadata={"Software": "Matplotlib"})
    svg = output_dir / f"{name}.svg"
    fig.savefig(svg, facecolor="white", metadata={"Date": None, "Creator": "Matplotlib"})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--comparison", choices=sorted(COMPARISONS), default="e31-same-load")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    profile = COMPARISONS[args.comparison]
    output_dir = args.output_dir or profile["output_dir"]
    if args.comparison == "e31-same-load":
        current, previous, captions = read_same_load(E31_BASELINE)
    else:
        current, current_caption = read_baseline(E29_BASELINE, "Current E29")
        previous, previous_caption = read_baseline(E28B_BASELINE, "Previous E28b")
        captions = previous_caption, current_caption
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                                "svg.hashsalt": "tp4-rigmark-current-vs-previous", "svg.fonttype": "path"})
    output_dir.mkdir(parents=True, exist_ok=True)
    generation_title, prefill_title = profile["titles"]
    figure(output_dir, "generation", generation_title, [
        (GENERATION, "Generation throughput", "Tokens per second · higher is better", False),
        (LATENCY, "First-token latency", "Seconds · lower is better", True),
    ], current, previous, captions,
        "Decode excludes initial wait; end-to-end throughput includes it. Concurrency TTFT is per stream.", profile)
    figure(output_dir, "prefill", prefill_title, [
        (COLD, "Cold prefill", "Effective tokens per second · higher is better", False),
        (REPLAY, "Immediate replay", "Effective tokens per second · higher is better", False),
    ], current, previous, captions, profile["prefill_note"], profile)
    print(f"Rendered two {args.comparison} comparison figures from 16 frozen metrics to {output_dir}")


if __name__ == "__main__":
    main()
