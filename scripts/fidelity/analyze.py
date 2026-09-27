#!/usr/bin/env python3
"""Compare raw fidelity runs: per-position metrics (private) and aggregates (public).

Each comparison names a candidate and a reference run (`<arm>/<run>`), a kind
(`prompt` or `gen`) and optionally a floor comparison over the same windows (for
example a second reference run against the first). Per-position arrays go to
data/fidelity/metrics/<name>.npz; aggregate JSON without token ids or text goes to
docs/fidelity/metrics/<name>.json and summary.json. Requires numpy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fidelity_io as fio  # noqa: E402
import metrics as m  # noqa: E402

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


REPO = fio.REPO
BUCKET_EDGES = [2048, 8192, 32768, 65536]
P99_GROUPS = ("overall", "category")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--comparisons", type=Path, help="JSON list of comparisons")
    p.add_argument("--k-pairs", type=Path,
                   help='JSON list of {"name","low","high"}: the same arms scored at a low and a high K')
    p.add_argument("--raw-root", type=Path, default=REPO / "data/fidelity/raw")
    p.add_argument("--manifest", type=Path, default=REPO / "data/fidelity/corpus/manifest.json")
    p.add_argument("--decode-manifest", type=Path, default=REPO / "data/fidelity/corpus/decode_manifest.json")
    p.add_argument("--private-out", type=Path, default=REPO / "data/fidelity/metrics")
    p.add_argument("--public-out", type=Path, default=REPO / "docs/fidelity/metrics")
    p.add_argument("--bootstrap", type=int, default=m.BOOT_B, help="bootstrap replicates")
    p.add_argument("--selftest", action="store_true", help="run the pipeline on synthetic data")
    args = p.parse_args(argv)
    if not args.selftest and not args.comparisons:
        p.error("--comparisons is required unless --selftest")
    return args


# ---------------------------------------------------------------- loading


def load_raw(run_dir: Path, kind: str, item_id: str, schema: dict):
    sub = "prompt" if kind == "prompt" else "gen"
    sidecar = fio.read_json(run_dir / sub / f"{item_id}.json")
    path = run_dir / sub / f"{item_id}.npz"
    if not sidecar or sidecar.get("status") != "ok" or not path.exists():
        return None
    with np.load(path) as data:
        return {name: data[name] for name in schema}


def run_k(run_dir: Path):
    record = fio.read_json(run_dir / "run.json") or {}
    return record.get("K")


def _bucket(positions):
    return np.digitize(positions, BUCKET_EDGES).astype(np.int8)


def _dnll(ref_lp, cand_lp):
    ref_lp, cand_lp = ref_lp.astype(np.float64), cand_lp.astype(np.float64)
    ok = np.isfinite(ref_lp) & np.isfinite(cand_lp)
    return np.where(ok, ref_lp - cand_lp, np.nan)


def prompt_positions(args, comp, entries, truncate_k=None, cand=None, ref=None):
    """Per-position metrics for a prompt comparison over the manifest windows."""
    cand_dir = args.raw_root / (cand or comp["cand"])
    ref_dir = args.raw_root / (ref or comp["ref"])
    parts, windows, missing = [], [], {"cand": 0, "ref": 0}
    for entry in entries:
        c = load_raw(cand_dir, "prompt", entry["id"], fio.PROMPT_ARRAYS)
        r = load_raw(ref_dir, "prompt", entry["id"], fio.PROMPT_ARRAYS)
        missing["cand"] += c is None
        missing["ref"] += r is None
        if c is None or r is None:
            continue
        ids = r["ids"]
        if not np.array_equal(ids, c["ids"]):
            raise SystemExit(f"{comp['name']}: {entry['id']} token ids differ between runs")
        tokens = fio.load_entry_tokens(args.manifest, entry)
        if not np.array_equal(ids, np.asarray(tokens, np.int64)):
            raise SystemExit(f"{comp['name']}: {entry['id']} raw ids differ from the corpus")
        n = ids.shape[0]
        if n < 2:
            continue
        k = truncate_k or r["topk_ids"].shape[1]
        kc = truncate_k or c["topk_ids"].shape[1]
        kl, valid, clamped, floored = m.np_coarse_kl(
            r["topk_ids"][1:, :k], r["topk_lp"][1:, :k], c["topk_ids"][1:, :kc], c["topk_lp"][1:, :kc],
            ids[1:].astype(np.int64), r["lp_actual"][1:], c["lp_actual"][1:])
        pos = np.arange(1, n, dtype=np.int64)
        start = (pos // m.BATCHED_TOKENS) * m.BATCHED_TOKENS
        size = np.minimum(m.BATCHED_TOKENS, n - start)
        parts.append({
            "window": np.full(n - 1, len(windows), np.int32), "position": pos.astype(np.int32),
            "kl": kl, "valid": valid, "clamped": clamped, "floored": floored,
            "top1": m.np_top1(r["topk_ids"][1:, :k], r["topk_lp"][1:, :k],
                              c["topk_ids"][1:, :kc], c["topk_lp"][1:, :kc]),
            "dnll": _dnll(r["lp_actual"][1:], c["lp_actual"][1:]),
            "missing_actual": np.isnan(r["lp_actual"][1:]) | np.isnan(c["lp_actual"][1:]),
            "bucket": _bucket(pos), "path": (size >= m.MARLIN_BELOW_ROWS).astype(np.int8),
        })
        windows.append(entry)
    return finish(parts, windows, missing)


def gen_positions(args, comp, entries):
    """Per-position metrics along the identical greedy prefix of paired generations."""
    cand_dir, ref_dir = args.raw_root / comp["cand"], args.raw_root / comp["ref"]
    parts, windows, missing, per_prompt = [], [], {"cand": 0, "ref": 0}, []
    for entry in entries:
        c = load_raw(cand_dir, "gen", entry["id"], fio.GEN_ARRAYS)
        r = load_raw(ref_dir, "gen", entry["id"], fio.GEN_ARRAYS)
        missing["cand"] += c is None
        missing["ref"] += r is None
        if c is None or r is None:
            continue
        length, divergence = m.generation_prefix(r["gen_ids"].tolist(), c["gen_ids"].tolist())
        per_prompt.append((length, -1 if divergence is None else divergence,
                           min(len(r["gen_ids"]), len(c["gen_ids"]))))
        if length == 0:
            windows.append(entry)
            continue
        kl, valid, clamped, floored = m.np_coarse_kl(
            r["topk_ids"][:length], r["topk_lp"][:length], c["topk_ids"][:length], c["topk_lp"][:length])
        same = r["gen_ids"][:length] == c["gen_ids"][:length]
        pos = np.arange(length, dtype=np.int64)
        parts.append({
            "window": np.full(length, len(windows), np.int32), "position": pos.astype(np.int32),
            "kl": kl, "valid": valid, "clamped": clamped, "floored": floored,
            "top1": m.np_top1(r["topk_ids"][:length], r["topk_lp"][:length],
                              c["topk_ids"][:length], c["topk_lp"][:length]),
            "dnll": np.where(same, _dnll(r["lp_actual"][:length], c["lp_actual"][:length]), np.nan),
            "missing_actual": np.isnan(r["lp_actual"][:length]) | np.isnan(c["lp_actual"][:length]),
            "bucket": _bucket(pos + int(entry.get("n_tokens", 0))),
            "path": np.full(length, -1, np.int8),
        })
        windows.append(entry)
    pp = finish(parts, windows, missing)
    pp["per_prompt"] = np.asarray(per_prompt, np.int64).reshape(-1, 3)
    return pp


def finish(parts, windows, missing):
    keys = ("window", "position", "kl", "valid", "clamped", "floored", "top1", "dnll",
            "missing_actual", "bucket", "path")
    if parts:
        arrays = {key: np.concatenate([part[key] for part in parts]) for key in keys}
    else:
        arrays = {key: np.zeros(0) for key in keys}
        arrays["window"] = np.zeros(0, np.int32)
    return {**arrays, "windows": windows, "missing": missing}


# ---------------------------------------------------------------- aggregation


def group_masks(pp) -> dict:
    """name -> (group family, boolean mask over positions)."""
    n = pp["kl"].shape[0]
    out = {"overall": ("overall", np.ones(n, bool))}
    win = pp["window"]
    for field in ("category", "source"):
        labels = np.asarray([w.get(field, "unknown") for w in pp["windows"]] or ["unknown"])
        per_pos = labels[win] if n else np.zeros(0, labels.dtype)
        for label in sorted(set(labels.tolist())):
            if pp["windows"]:
                out[f"{field}:{label}"] = (field, per_pos == label)
    for index, label in enumerate(m.BUCKET_LABELS):
        mask = pp["bucket"] == index
        if mask.any():
            out[f"bucket:{label}"] = ("bucket", mask)
    for index, label in enumerate(m.PATHS):
        mask = pp["path"] == index
        if mask.any():
            out[f"path:{label}"] = ("path", mask)
    return out


def per_window(pp, select, values):
    """Window index -> (sum, count) for the selected positions."""
    win = pp["window"][select]
    size = len(pp["windows"])
    sums = np.bincount(win, weights=values[select], minlength=size)
    counts = np.bincount(win, minlength=size)
    return sums, counts


def window_values(pp, select):
    """List of per-window value arrays (only windows with values), and their indices."""
    win, vals = pp["window"][select], pp["kl"][select]
    order = np.argsort(win, kind="stable")
    win, vals = win[order], vals[order]
    ids, starts = np.unique(win, return_index=True)
    return list(ids), np.split(vals, starts[1:]) if len(ids) else []


def ratio_ci(sums, counts, B):
    keep = counts > 0
    if keep.sum() == 0:
        return {"estimate": None}
    return m.np_ratio_bootstrap(sums[keep], counts[keep], B)


def group_block(pp, mask, B, with_p99):
    valid = mask & pp["valid"]
    kl_sums, kl_counts = per_window(pp, valid, np.nan_to_num(pp["kl"]))
    t_valid = mask & (pp["top1"] >= 0)
    t_sums, t_counts = per_window(pp, t_valid, (pp["top1"] == 1).astype(np.float64))
    d_valid = mask & np.isfinite(pp["dnll"])
    d_sums, d_counts = per_window(pp, d_valid, np.nan_to_num(pp["dnll"]))
    kl = m.np_summary(pp["kl"][valid])
    kl["mean_ci"] = ratio_ci(kl_sums, kl_counts, B)
    if with_p99 and valid.any():
        kl["p99_ci"] = m.np_percentile_bootstrap(window_values(pp, valid)[1], 99.0, B)
    return {
        "windows": int((kl_counts > 0).sum()),
        "positions_scored": int(mask.sum()),
        "positions_valid": int(valid.sum()),
        "invalid": int((mask & ~pp["valid"]).sum()),
        "clamped": int((mask & pp["clamped"]).sum()),
        "floored": int((mask & pp["floored"]).sum()),
        "missing_actual": int((mask & pp["missing_actual"]).sum()),
        "kl": kl,
        "top1_agreement": dict(ratio_ci(t_sums, t_counts, B), n=int(t_valid.sum())),
        "delta_nll": dict(ratio_ci(d_sums, d_counts, B), n=int(d_valid.sum())),
    }


def aggregate(pp, B):
    blocks = {}
    for name, (family, mask) in group_masks(pp).items():
        blocks[name] = group_block(pp, mask, B, family in P99_GROUPS)
    out = {"overall": blocks.pop("overall")}
    for family, key in (("category", "by_category"), ("source", "by_source"),
                        ("bucket", "by_position_bucket"), ("path", "by_prefill_path")):
        prefix = family + ":"
        out[key] = {name[len(prefix):]: block for name, block in blocks.items() if name.startswith(prefix)}
    return out


def _aligned(cp, fp, cmask, fmask, select_c, select_f, values_c, values_f):
    """Per-window (sum, count) for cand and floor on common window ids with data on both."""
    cs, cc = per_window(cp, cmask & select_c, values_c)
    fs, fc = per_window(fp, fmask & select_f, values_f)
    cidx = {w["id"]: i for i, w in enumerate(cp["windows"])}
    fidx = {w["id"]: i for i, w in enumerate(fp["windows"])}
    common = [(cidx[i], fidx[i]) for i in sorted(set(cidx) & set(fidx))
              if cc[cidx[i]] > 0 and fc[fidx[i]] > 0]
    ci = np.asarray([a for a, _ in common], np.int64)
    fi = np.asarray([b for _, b in common], np.int64)
    return cs[ci], cc[ci], fs[fi], fc[fi], common


def excess(cp, fp, floor_overall, B):
    """Candidate minus floor over the same windows, per group, with paired bootstrap."""
    cgroups, fgroups = group_masks(cp), group_masks(fp)
    out = {}
    for name in cgroups:
        if name not in fgroups:
            continue
        family, cmask = cgroups[name]
        fmask = fgroups[name][1]
        cs, cc, fs, fc, common = _aligned(cp, fp, cmask, fmask, cp["valid"], fp["valid"],
                                          np.nan_to_num(cp["kl"]), np.nan_to_num(fp["kl"]))
        if not common:
            continue
        block = {"windows": len(common),
                 "kl_mean_excess": m.np_paired_ratio_diff(cs, cc, fs, fc, B)}
        ts, tc, us, uc, tcommon = _aligned(cp, fp, cmask, fmask, cp["top1"] >= 0, fp["top1"] >= 0,
                                           (cp["top1"] == 1).astype(float), (fp["top1"] == 1).astype(float))
        if tcommon:
            block["top1_drop_pp"] = m.np_paired_ratio_diff(us, uc, ts, tc, B, scale=100.0)
        if family in P99_GROUPS:
            cw_ids, cw = window_values(cp, cmask & cp["valid"])
            fw_ids, fw = window_values(fp, fmask & fp["valid"])
            cmap = {cp["windows"][i]["id"]: v for i, v in zip(cw_ids, cw)}
            fmap = {fp["windows"][i]["id"]: v for i, v in zip(fw_ids, fw)}
            ids = sorted(set(cmap) & set(fmap))
            if ids:
                block["kl_p99_excess"] = m.np_paired_percentile_diff(
                    [cmap[i] for i in ids], [fmap[i] for i in ids], 99.0, B)
        out[name] = block
    result = {"overall": out.pop("overall", None), "groups": out}
    kl_se = floor_overall["kl"]["mean_ci"].get("se")
    top1_se = floor_overall["top1_agreement"].get("se")
    result["mde"] = {"kl_mean_nats": m.mde(kl_se),
                     "top1_pp": None if top1_se is None else m.mde(top1_se) * 100.0,
                     "basis": "2.8 x bootstrap SE of the floor's mean (80% power, alpha 0.05 two-sided)"}
    return result


def generation_block(pp):
    rows = pp["per_prompt"]
    if rows.size == 0:
        return {"prompts": 0}
    diverged = rows[:, 1] >= 0
    first = rows[diverged, 1]
    return {
        "prompts": int(rows.shape[0]),
        "identical_over_common_length": int((~diverged).sum()),
        "diverged": int(diverged.sum()),
        "first_divergence": m.np_summary(first.astype(np.float64)) if first.size else None,
        "diverged_within": {str(t): int((first < t).sum()) for t in (16, 64, 256)},
        "compared_positions": int(rows[:, 0].sum()),
        "common_length_total": int(rows[:, 2].sum()),
    }


# ---------------------------------------------------------------- output


def clean(value):
    """JSON-safe copy: NaN/inf -> None, numpy scalars -> Python."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if np is not None and isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_public(path: Path, data) -> None:
    payload = json.dumps(clean(data), indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fio._atomic(path, lambda handle: handle.write(payload.encode("utf-8")))


def write_private(path: Path, pp) -> None:
    arrays = {key: pp[key] for key in ("window", "position", "kl", "valid", "clamped", "floored",
                                       "top1", "dnll", "missing_actual", "bucket", "path")}
    arrays["window_ids"] = np.asarray([w["id"] for w in pp["windows"]] or [""], dtype="U32")
    if "per_prompt" in pp:
        arrays["per_prompt"] = pp["per_prompt"]
    fio._atomic(path, lambda handle: np.savez(handle, **arrays), mode=0o600)


def headline(result):
    overall = result["overall"]
    out = {"kind": result["kind"], "cand": result["cand"], "ref": result["ref"], "floor": result["floor"],
           "windows": overall["windows"], "positions_valid": overall["positions_valid"],
           "kl_mean": overall["kl"]["mean_ci"], "kl_p99": overall["kl"].get("p99"),
           "kl_p99_ci": overall["kl"].get("p99_ci"), "top1_agreement": overall["top1_agreement"],
           "delta_nll_mean": overall["delta_nll"]}
    if result.get("excess_vs_floor"):
        ex = result["excess_vs_floor"]
        out["excess"] = {"overall": ex["overall"], "mde": ex["mde"]}
    if result.get("generation"):
        out["generation"] = result["generation"]
    return out


def k_sensitivity(args, pair, comps, results, entries, B):
    low, high = comps[pair["low"]], comps[pair["high"]]
    if low["kind"] != "prompt" or high["kind"] != "prompt":
        raise SystemExit(f"k-pair {pair['name']}: both comparisons must be kind prompt")
    k_low = min(filter(None, [run_k(args.raw_root / low["cand"]), run_k(args.raw_root / low["ref"])]))
    k_high = min(filter(None, [run_k(args.raw_root / high["cand"]), run_k(args.raw_root / high["ref"])]))
    lp, hp = results[pair["low"]]["_pp"], results[pair["high"]]["_pp"]
    truncated = prompt_positions(args, high, entries, truncate_k=k_low)
    hs, hc, ls, lc, common = _aligned(hp, lp, np.ones_like(hp["valid"]), np.ones_like(lp["valid"]),
                                      hp["valid"], lp["valid"], np.nan_to_num(hp["kl"]), np.nan_to_num(lp["kl"]))
    ts, tc, hs2, hc2, _ = _aligned(truncated, hp, np.ones_like(truncated["valid"]), np.ones_like(hp["valid"]),
                                   truncated["valid"], hp["valid"],
                                   np.nan_to_num(truncated["kl"]), np.nan_to_num(hp["kl"]))
    return {
        "schema": "fidelity-k-sensitivity/1", "name": pair["name"], "low": pair["low"], "high": pair["high"],
        "K_low": k_low, "K_high": k_high, "windows": len(common),
        "kl_mean_high_minus_low": m.np_paired_ratio_diff(hs, hc, ls, lc, B),
        "kl_mean_high": results[pair["high"]]["overall"]["kl"]["mean_ci"],
        "kl_mean_low": results[pair["low"]]["overall"]["kl"]["mean_ci"],
        "same_data_high_minus_truncated_to_K_low": m.np_paired_ratio_diff(hs2, hc2, ts, tc, B),
        "note": "coarse KL is a lower bound; a larger K refines the partition, so the high-K value "
                "is expected to be at least the truncated one on the same data",
    }


def analyze(args) -> dict:
    if np is None:
        raise SystemExit("analyze.py requires numpy (use data/fidelity/.venv/bin/python)")
    comparisons = json.loads(args.comparisons.read_text(encoding="utf-8"))
    names = [c["name"] for c in comparisons]
    if len(set(names)) != len(names):
        raise SystemExit("duplicate comparison names")
    comps = {c["name"]: c for c in comparisons}
    for comp in comparisons:
        if comp.get("floor") and comp["floor"] not in comps:
            raise SystemExit(f"{comp['name']}: unknown floor {comp['floor']}")
    manifests = {}

    def entries_for(kind):
        if kind not in manifests:
            path, key = (args.manifest, "windows") if kind == "prompt" else (args.decode_manifest, "prompts")
            manifests[kind] = fio.load_manifest(path, key)[1]
        return manifests[kind]

    results = {}
    for comp in comparisons:
        kind = comp.get("kind", "prompt")
        if kind == "prompt":
            pp = prompt_positions(args, comp, entries_for(kind))
        elif kind == "gen":
            pp = gen_positions(args, comp, entries_for(kind))
        else:
            raise SystemExit(f"{comp['name']}: unknown kind {kind}")
        result = {"schema": "fidelity-metrics/1", "name": comp["name"], "kind": kind,
                  "cand": comp["cand"], "ref": comp["ref"], "floor": comp.get("floor"),
                  "K": {"cand": run_k(args.raw_root / comp["cand"]), "ref": run_k(args.raw_root / comp["ref"])},
                  "partition": ("top-K intersection + actual token + rest" if kind == "prompt"
                                else "top-K intersection + rest"),
                  "bound": "coarse KL is a lower bound on the true KL (data-processing inequality)",
                  "windows_in_manifest": len(entries_for(kind)),
                  "windows_compared": len(pp["windows"]), "windows_missing": pp["missing"],
                  "bootstrap": {"B": args.bootstrap, "seed": m.BOOT_SEED, "unit": "window",
                                "estimator": "token-weighted ratio", "ci": "percentile 95%"}}
        result.update(aggregate(pp, args.bootstrap))
        if kind == "gen":
            result["generation"] = generation_block(pp)
        write_private(args.private_out / f"{comp['name']}.npz", pp)
        result["_pp"] = pp
        results[comp["name"]] = result
        print(f"{comp['name']}: windows={len(pp['windows'])} positions={result['overall']['positions_valid']} "
              f"kl_mean={result['overall']['kl']['mean']}")
    for comp in comparisons:
        if comp.get("floor"):
            floor = results[comp["floor"]]
            if floor["kind"] != results[comp["name"]]["kind"]:
                raise SystemExit(f"{comp['name']}: floor kind differs")
            results[comp["name"]]["excess_vs_floor"] = excess(
                results[comp["name"]]["_pp"], floor["_pp"], floor["overall"], args.bootstrap)
    k_results = {}
    if args.k_pairs:
        for pair in json.loads(args.k_pairs.read_text(encoding="utf-8")):
            k_results[pair["name"]] = k_sensitivity(args, pair, comps, results, entries_for("prompt"),
                                                    args.bootstrap)
            write_public(args.public_out / f"{pair['name']}.json", k_results[pair["name"]])
    for name, result in results.items():
        public = {k: v for k, v in result.items() if k != "_pp"}
        write_public(args.public_out / f"{name}.json", public)
    summary = {"schema": "fidelity-summary/1", "generated_utc": fio.utc_now(), "harness_git": fio.git_identity(),
               "comparisons": {name: headline(r) for name, r in results.items()},
               "k_sensitivity": k_results}
    write_public(args.public_out / "summary.json", summary)
    return {"results": results, "k": k_results, "summary": summary}


# ---------------------------------------------------------------- selftest


def _log_softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=1, keepdims=True))


def _topk(logp, k):
    idx = np.argsort(-logp, axis=1, kind="stable")[:, :k]
    return idx.astype(np.int32), np.take_along_axis(logp, idx, 1).astype(np.float32)


def _write_prompt_run(root, arm, run, k, windows, tokens, noise, seed):
    run_dir = root / arm / run
    fio.write_json_atomic(run_dir / "run.json", {"arm": arm, "run": run, "K": k, "kind": "prompt"})
    for index, entry in enumerate(windows):
        ids = np.asarray(tokens[entry["id"]], np.int64)
        base = np.random.default_rng(1000 + index).normal(0, 3.0, (len(ids), 500))
        logp = _log_softmax(base + np.random.default_rng(seed * 7919 + index).normal(0, noise, base.shape))
        top_ids, top_lp = _topk(logp, k)
        lp_actual = logp[np.arange(len(ids)), ids].astype(np.float32)
        rank = (logp > logp[np.arange(len(ids)), ids][:, None]).sum(1) + 1
        top_ids[0], top_lp[0], lp_actual[0], rank[0] = -1, -np.inf, np.nan, -1
        fio.save_npz(run_dir / "prompt" / f"{entry['id']}.npz",
                     {"ids": ids, "lp_actual": lp_actual, "rank_actual": rank, "topk_ids": top_ids,
                      "topk_lp": top_lp}, fio.PROMPT_ARRAYS)
        fio.write_json_atomic(run_dir / "prompt" / f"{entry['id']}.json", {"status": "ok", "K": k})


def _write_gen_run(root, arm, run, k, prompts, noise, seed):
    run_dir = root / arm / run
    fio.write_json_atomic(run_dir / "run.json", {"arm": arm, "run": run, "K": k, "kind": "gen"})
    for index, entry in enumerate(prompts):
        base = np.random.default_rng(5000 + index).normal(0, 3.0, (64, 500))
        base[:, 7] += 4.0 + np.linspace(0, 1, 64)  # a stable greedy choice that noise can flip
        logp = _log_softmax(base + np.random.default_rng(seed * 104729 + index).normal(0, noise, base.shape))
        top_ids, top_lp = _topk(logp, k)
        gen = top_ids[:, 0].copy()
        fio.save_npz(run_dir / "gen" / f"{entry['id']}.npz",
                     {"gen_ids": gen, "lp_actual": top_lp[:, 0], "topk_ids": top_ids, "topk_lp": top_lp},
                     fio.GEN_ARRAYS)
        fio.write_json_atomic(run_dir / "gen" / f"{entry['id']}.json", {"status": "ok", "K": k})


def _write_manifest(path, key, entries, tokens):
    rows = []
    for entry in entries:
        rel = f"tokens/{entry['id']}.u32"
        data = np.asarray(tokens[entry["id"]], "<u4").tobytes()
        (path.parent / "tokens").mkdir(parents=True, exist_ok=True)
        (path.parent / rel).write_bytes(data)
        rows.append(dict(entry, n_tokens=len(tokens[entry["id"]]), path=rel,
                         sha256=hashlib.sha256(data).hexdigest()))
    fio.write_json_atomic(path, {"schema": "fidelity-corpus/1", "frozen": True, key: rows,
                                 "global_sha256": fio.global_sha256(rows)})


def selftest() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="fidelity-selftest-"))
    try:
        rng = np.random.default_rng(20260927)
        lengths = [600, 2500, 3100, 10000, 1200, 4000]
        cats = ["agentic_code", "long_context", "italian_chat", "structured_json"]
        windows = [{"id": f"w{i + 1:04d}", "category": cats[i % 4], "source": "synthetic",
                    "project": "selftest"} for i in range(len(lengths))]
        tokens = {w["id"]: rng.integers(0, 500, n).tolist() for w, n in zip(windows, lengths)}
        corpus = tmp / "corpus"
        _write_manifest(corpus / "manifest.json", "windows", windows, tokens)
        prompts = [{"id": f"d{i + 1:03d}", "category": cats[i % 4], "source": "synthetic"} for i in range(6)]
        ptokens = {p["id"]: rng.integers(0, 500, 40).tolist() for p in prompts}
        _write_manifest(corpus / "decode_manifest.json", "prompts", prompts, ptokens)
        raw = tmp / "raw"
        _write_prompt_run(raw, "R0", "a", 20, windows, tokens, 0.02, 1)
        _write_prompt_run(raw, "R0", "b", 20, windows, tokens, 0.02, 2)
        _write_prompt_run(raw, "Cm", "a", 20, windows, tokens, 0.15, 3)
        _write_prompt_run(raw, "R0", "a-k50", 50, windows, tokens, 0.02, 1)
        _write_prompt_run(raw, "Cm", "a-k50", 50, windows, tokens, 0.15, 3)
        _write_gen_run(raw, "R0", "ga", 20, prompts, 0.05, 11)
        _write_gen_run(raw, "R0", "gb", 20, prompts, 0.05, 12)
        _write_gen_run(raw, "Cm", "g", 20, prompts, 0.8, 13)
        comparisons = [
            {"name": "floor-r0", "cand": "R0/b", "ref": "R0/a", "floor": None, "kind": "prompt"},
            {"name": "cm-vs-r0", "cand": "Cm/a", "ref": "R0/a", "floor": "floor-r0", "kind": "prompt"},
            {"name": "cm-vs-r0-k50", "cand": "Cm/a-k50", "ref": "R0/a-k50", "floor": None, "kind": "prompt"},
            {"name": "gen-floor-r0", "cand": "R0/gb", "ref": "R0/ga", "floor": None, "kind": "gen"},
            {"name": "gen-cm-vs-r0", "cand": "Cm/g", "ref": "R0/ga", "floor": "gen-floor-r0", "kind": "gen"},
        ]
        (tmp / "comparisons.json").write_text(json.dumps(comparisons))
        (tmp / "k-pairs.json").write_text(json.dumps([{"name": "k-cm-vs-r0", "low": "cm-vs-r0",
                                                        "high": "cm-vs-r0-k50"}]))
        args = parse_args(["--comparisons", str(tmp / "comparisons.json"), "--k-pairs", str(tmp / "k-pairs.json"),
                           "--raw-root", str(raw), "--manifest", str(corpus / "manifest.json"),
                           "--decode-manifest", str(corpus / "decode_manifest.json"),
                           "--private-out", str(tmp / "private"), "--public-out", str(tmp / "public"),
                           "--bootstrap", "200"])
        out = analyze(args)
        res = out["results"]
        floor, cand = res["floor-r0"]["overall"], res["cm-vs-r0"]["overall"]
        checks = []

        def check(name, ok):
            checks.append((name, bool(ok)))

        check("all positions scored", cand["positions_scored"] == sum(n - 1 for n in lengths))
        check("no invalid positions", cand["invalid"] == 0 and floor["invalid"] == 0)
        check("floor KL positive and below candidate", 0 < floor["kl"]["mean"] < cand["kl"]["mean"])
        ex = res["cm-vs-r0"]["excess_vs_floor"]["overall"]
        check("mean excess CI above zero", ex["kl_mean_excess"]["ci_low"] > 0)
        check("p99 excess positive", ex["kl_p99_excess"]["estimate"] > 0)
        check("top-1 drop positive", ex["top1_drop_pp"]["estimate"] > 0)
        check("MDE reported", res["cm-vs-r0"]["excess_vs_floor"]["mde"]["kl_mean_nats"] > 0)
        check("both prefill paths", set(res["cm-vs-r0"]["by_prefill_path"]) == {"marlin", "bf16"})
        check("position buckets", {"0-2K", "2-8K", "8-32K"} <= set(res["cm-vs-r0"]["by_position_bucket"]))
        check("categories", set(res["cm-vs-r0"]["by_category"]) == set(cats))
        k = out["k"]["k-cm-vs-r0"]
        same = k["same_data_high_minus_truncated_to_K_low"]["estimate"]
        check("K truncation reproduces the low-K run", abs(k["kl_mean_high_minus_low"]["estimate"] - same) < 1e-9)
        check("higher K refines the bound", same >= -1e-12)
        gen = res["gen-cm-vs-r0"]["generation"]
        check("generation divergence found", gen["diverged"] > 0 and gen["compared_positions"] > 0)
        check("gen excess computed", res["gen-cm-vs-r0"]["excess_vs_floor"]["overall"] is not None)
        # numpy per-position values against the stdlib reference on one long window.
        with np.load(tmp / "private" / "cm-vs-r0.npz") as data:
            index = int(np.where(data["window_ids"] == "w0004")[0][0])
            sel = data["window"] == index
            kl_np, path_np = data["kl"][sel], data["path"][sel]
        with np.load(raw / "R0/a/prompt/w0004.npz") as r, np.load(raw / "Cm/a/prompt/w0004.npz") as c:
            worst = 0.0
            for p in range(1, 10000, 97):
                value = m.coarse_kl(r["topk_ids"][p].tolist(), r["topk_lp"][p].tolist(),
                                    c["topk_ids"][p].tolist(), c["topk_lp"][p].tolist(),
                                    extra=(int(r["ids"][p]), float(r["lp_actual"][p]), float(c["lp_actual"][p])))
                worst = max(worst, abs(value.kl - kl_np[p - 1]))
        check("numpy matches stdlib KL", worst < 1e-9)
        expected_paths = [m.PATHS.index(x) for x in m.prefill_paths(10000)[1:]]
        check("prefill path tags", path_np.tolist() == expected_paths)
        public = "".join(p.read_text() for p in (tmp / "public").glob("*.json"))
        check("public JSON has no ids or salts", not any(s in public for s in ('"ids"', "topk", "salt\"")))
        check("summary lists every comparison",
              set(json.loads((tmp / "public/summary.json").read_text())["comparisons"]) == set(res))
        for name, ok in checks:
            print(f"selftest: {'PASS' if ok else 'FAIL'} {name}")
        failed = [name for name, ok in checks if not ok]
        print(f"selftest: {'PASS' if not failed else 'FAIL'} ({len(checks) - len(failed)}/{len(checks)})")
        return 1 if failed else 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.selftest:
        return selftest()
    analyze(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
