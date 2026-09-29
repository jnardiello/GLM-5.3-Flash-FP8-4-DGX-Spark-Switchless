#!/usr/bin/env python3
"""Recreate the requests-to-MemAvailable figure from its portable bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
CSV_NAME = "requests-memory.csv"
FIGURE_NAME = "requests-vs-memavailable"
CSV_FIELDS = (
    "rank", "time_ns", "monotonic_ns",
    "cumulative_generation_requests_submitted",
    "cumulative_ordinary_requests_submitted",
    "cumulative_cancellation_requests_submitted",
    "graph_request_ordinal", "request_ordinal_join",
    "active_request_index", "active_request_type", "active_request_attempt",
    "sample_class", "drain_proof_label", "drain_proof_scan_started_monotonic_ns",
    "raw_sample_join_status", "drain_proof_match_status",
    "mem_available_bytes", "mem_available_known", "host_source_status",
    "host_source_age_ns", "cadence_target_interval_ns",
    "cadence_actual_interval_ns", "gap_before", "gap_before_ns",
    "clock_join_alignment",
)
INT_FIELDS = {
    "rank", "time_ns", "monotonic_ns",
    "cumulative_generation_requests_submitted",
    "cumulative_ordinary_requests_submitted",
    "cumulative_cancellation_requests_submitted", "graph_request_ordinal",
    "active_request_index", "active_request_attempt",
    "drain_proof_scan_started_monotonic_ns", "mem_available_bytes",
    "host_source_age_ns", "cadence_target_interval_ns",
    "cadence_actual_interval_ns", "gap_before_ns",
}
BOOL_FIELDS = {"mem_available_known", "gap_before"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.chmod(path, 0o600)


def load_inputs(bundle: Path) -> tuple[
        list[dict[str, Any]], dict[str, Any], dict[str, str]]:
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    files = manifest.get("files") or {}
    if manifest.get("schema") != "tp4-requests-memory-readme-manifest/v1":
        raise RuntimeError("portable graph bundle manifest schema is invalid")
    for name in (CSV_NAME, "summary.json"):
        expected = files.get(name)
        if not isinstance(expected, str) or sha256_file(bundle / name) != expected:
            raise RuntimeError(f"portable graph input hash mismatch: {name}")
    summary = json.loads((bundle / "summary.json").read_text())
    csv_hash = sha256_file(bundle / CSV_NAME)
    if (summary.get("derived_csv") or {}).get("sha256") != csv_hash:
        raise RuntimeError("portable CSV does not match summary provenance")
    with (bundle / CSV_NAME).open(newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise RuntimeError("portable CSV columns do not match the graph schema")
        rows: list[dict[str, Any]] = []
        for source in reader:
            row: dict[str, Any] = {}
            for name in CSV_FIELDS:
                value = source[name]
                if name in INT_FIELDS:
                    row[name] = int(value) if value != "" else None
                elif name in BOOL_FIELDS:
                    if value not in {"True", "False"}:
                        raise RuntimeError(f"portable CSV boolean is invalid: {name}")
                    row[name] = value == "True"
                else:
                    row[name] = value or None
            rows.append(row)
    if len(rows) != (summary.get("derived_csv") or {}).get("rows"):
        raise RuntimeError("portable CSV row count does not match summary")
    return rows, summary, {
        "source_bundle_manifest_sha256": sha256_file(manifest_path),
        "source_summary_sha256": files["summary.json"],
        "source_derived_csv_sha256": files[CSV_NAME],
    }


def under_load_series(selected: list[dict[str, Any]], total: int) -> tuple[
        list[int], list[float], list[int], list[float]]:
    bins: dict[int, list[dict[str, Any]]] = defaultdict(list)
    uncertain_bins: set[int] = set()
    previous = None
    for row in selected:
        if (row["sample_class"] == "under_load"
                and row["active_request_index"] is not None):
            bins[int(row["active_request_index"])].append(row)
        if row["gap_before"]:
            if (row["sample_class"] == "under_load"
                    and row["active_request_index"] is not None):
                uncertain_bins.add(int(row["active_request_index"]))
            if (previous is not None and previous["sample_class"] == "under_load"
                    and previous["active_request_index"] is not None):
                uncertain_bins.add(int(previous["active_request_index"]))
        previous = row
    x = list(range(1, total + 1))
    y, observed_x, observed_y = [], [], []
    for index in x:
        values = [row["mem_available_bytes"] for row in bins.get(index, [])
                  if isinstance(row["mem_available_bytes"], int)]
        uncertain = (index in uncertain_bins or any(
            not row["mem_available_known"] for row in bins.get(index, [])))
        minimum = min(values) / 2 ** 30 if values else math.nan
        y.append(math.nan if uncertain else minimum)
        if values:
            observed_x.append(index)
            observed_y.append(minimum)
    return x, y, observed_x, observed_y


def plot_figure(rows: list[dict[str, Any]], summary: dict[str, Any],
                svg: Path, png: Path) -> str:
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.5,
        "svg.fonttype": "path", "svg.hashsalt": "tp4-requests-memory-v1",
    })
    import matplotlib.pyplot as plt

    total = summary["request_counts"]["total_generation_requests_submitted"]
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for rank, axis in enumerate(axes.flat):
        selected = [row for row in rows if row["rank"] == rank]
        x, y, observed_x, observed_y = under_load_series(selected, total)
        axis.plot(x, y, color="#0072B2", linewidth=1.0,
                  label="under-load minimum (complete sampled bin)")
        axis.scatter(observed_x, observed_y, color="#0072B2", s=7, alpha=0.45,
                     label="observed request minimum")
        pause = {}
        for row in selected:
            if (row["sample_class"] in {"idle", "between_requests"}
                    and isinstance(row["mem_available_bytes"], int)):
                pause[row["cumulative_generation_requests_submitted"]] = row
        if pause:
            axis.scatter(list(pause), [row["mem_available_bytes"] / 2 ** 30
                                      for row in pause.values()],
                         facecolors="none", edgecolors="#666666", s=18,
                         label="last pause / idle sample")
        drains = [row for row in selected
                  if row["request_ordinal_join"] in {
                      "exact-cancel-proof-identity", "exact-final-proof-identity"}
                  and isinstance(row["mem_available_bytes"], int)]
        if drains:
            axis.scatter([row["graph_request_ordinal"] for row in drains],
                         [row["mem_available_bytes"] / 2 ** 30 for row in drains],
                         marker="*", s=48, color="#D55E00", edgecolor="black",
                         linewidth=0.35, label="exact fresh-drain sample")
        axis.set_title(f"Rank {rank}")
        axis.set_ylabel("MemAvailable (GiB)")
        axis.grid(axis="y", color="#D7DEE8", linewidth=0.55)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(loc="best", fontsize=7.2)
    for axis in axes[-1]:
        axis.set_xlabel("Cumulative submitted generation requests")
    if summary["source_checkpoint_status"] == "owner_stopped":
        seconds = int(summary["source_elapsed_seconds"])
        minutes, seconds = divmod(seconds, 60)
        subtitle = (
            f"{minutes}m{seconds:02d}s mixed load; ended early at operator request")
    else:
        subtitle = f"sealed soak status: {summary['source_checkpoint_status']}"
    figure.suptitle(
        f"Memory headroom vs submitted generation requests\n{subtitle}",
        y=0.975, fontsize=14, fontweight="bold")
    figure.subplots_adjust(left=0.07, right=0.985, top=0.86, bottom=0.115,
                           hspace=0.30, wspace=0.17)
    figure.text(
        0.5, 0.015,
        "Growth/replay/cancel submissions count on x. Hollow points are pause/idle; stars "
        "are exact drains. Node/controller wall clocks are uncalibrated; missing data stays gaps.",
        ha="center", va="bottom", fontsize=8, color="#4C596A")
    figure.savefig(svg, format="svg", dpi=160, metadata={"Date": None})
    figure.savefig(png, format="png", dpi=160,
                   metadata={"Software": f"matplotlib {matplotlib.__version__}"})
    plt.close(figure)
    os.chmod(svg, 0o600); os.chmod(png, 0o600)
    return matplotlib.__version__


def replot(bundle: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    rows, summary, provenance = load_inputs(bundle)
    temporary = output.with_name(f".{output.name}.part-{os.getpid()}")
    temporary.mkdir(mode=0o700, parents=False)
    try:
        version = plot_figure(rows, summary,
                              temporary / f"{FIGURE_NAME}.svg",
                              temporary / f"{FIGURE_NAME}.png")
        files = {name: sha256_file(temporary / name) for name in (
            f"{FIGURE_NAME}.svg", f"{FIGURE_NAME}.png")}
        write_json(temporary / "manifest.json", {
            "schema": "tp4-requests-memory-replot-manifest/v1",
            "files": files, "matplotlib_version": version,
            "portable_inputs": provenance,
        })
        os.replace(temporary, output)
        return {"status": "PASS", "output": str(output),
                "rows": len(rows), "matplotlib_version": version}
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replot", nargs="?", const=HERE, type=Path,
                        metavar="PORTABLE_BUNDLE", required=True)
    parser.add_argument("--output", type=Path, default=HERE / "replot")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(replot(args.replot, args.output), indent=2, sort_keys=True))
        return 0
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"requests-memory-replot: ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
