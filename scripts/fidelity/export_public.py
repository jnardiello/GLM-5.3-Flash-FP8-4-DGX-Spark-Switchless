#!/usr/bin/env python3
"""Export a portable, privacy-safe extract of the fidelity campaign's experiment data.

Reads the ignored campaign data (data/fidelity/) and the public aggregates
(docs/fidelity/) and writes docs/historical_benchmarks/experiments/2026-09-27-fidelity/:
results.json, runs.json, corpus-hashes.json, per-window CSVs for the model-native
windows generated from public prompts only, task and probe results, gate results,
memory samples without host names, a determinism summary and a README.

Privacy rules enforced here: no corpus text, token ids, logprob arrays, private window
ids, project names, host names, addresses, usernames, absolute paths or raw salts.
Private-session and synthetic windows appear only in aggregates and as SHA-256 digests
keyed by "h" + the first 12 hex digits of their content digest. Rank records (host,
command line, log lines) are dropped from run metadata.

The output is built in a temporary directory, leak-checked (the report's patterns and
private terms plus site keys, the local username and secret patterns), and only then
replaces the output directory. Any finding names the file and category, never the value,
and exits 2 without touching the output. The result is deterministic: re-running on
unchanged inputs produces identical bytes.

numpy is required only when data/fidelity/metrics-v2/*.npz exists (per-window metrics).
"""

from __future__ import annotations

import argparse
import csv
import getpass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_report as br  # noqa: E402

EXPERIMENT_ID = "2026-09-27-fidelity"
REGIMES = ("dense", "sparse", "all")
SIDES = ("contrast", "floor")
FIELDS = ("count", "kl_sum", "top1_sum", "cov_ref_sum", "cov_cand_sum")
CI_KEYS = ("estimate", "ci_low", "ci_high", "lb95", "ub95", "se", "B")
HEADLINE_PAIRS = ("cm-vs-r0", "n-vs-r0")
BOOT_KEYS = ("label", "recorded_utc", "overlay", "overlay_sha256", "cluster_env_sha256", "image", "image_id_pinned",
             "model_repo", "model_rev", "draft_rev", "kv_cache_dtype", "spec_tokens", "extra_vllm_args", "k")
RUN_KEYS = ("schema", "arm", "run", "kind", "K", "request_model", "server_model_id", "request_defaults",
            "started_utc", "ended_utc", "harness_git", "manifest_global_sha256")
# Boot label prefix -> arm (first match wins; "cpre-" must precede "cm"/"cp-s").
BOOT_ARM = (("r0fp8-m", "R0"), ("r0-s", "R0"), ("cpre-", "Cpre"), ("cm", "Cm"), ("cp-s", "Cp"), ("e29", "Cp"),
            ("n-", "N"), ("l0919", "L0919"), ("le21", "LE21"), ("le22b", "LE22b"))
# The unspecified bind address is not a site value; model answers use it in config examples.
EXTRA_IPV4_ALLOW = ("127.0.0.1", "0.0.0.0")
SECRET_PATTERNS = [
    ("Hugging Face token", re.compile(r"hf_[A-Za-z0-9]{30,}")),
    ("API key", re.compile(r"sk-[A-Za-z0-9_-]{20,}")),
    ("GitHub token", re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}")),
    ("AWS access key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("private key", re.compile(r"BEGIN [A-Z ]*PRIVATE KEY")),
]
EXTRA_PATTERNS = [
    ("absolute home path", re.compile(r"/Users/|/home/")),
    ("window id", re.compile(r"w\d{4}")),
]
IPV4 = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
LIMITATIONS = [
    "Coarse top-20 KL is a lower bound on the full-vocabulary KL: tokens outside both top-K rows share one cell.",
    "R0 is the vendor FP8 checkpoint served with an FP8 KV cache, not a BF16 reference; every arm shares that error.",
    "The sparse regime (more than 2,048 conditioning tokens) is unresolved: the engine is nondeterministic there "
    "even for the same arm, and single executions measure operational disagreement.",
    "The corpus is provisional (not frozen) and finite: coding-agent sessions, synthetic Italian conversations and "
    "model-native continuations. Other workloads may differ.",
    "The voxel checks and the cloud reference arm (Z) are deferred and not part of this extract.",
    "Per-window rows are published only for model-native windows generated from public prompts; every other window "
    "contributes to aggregates only.",
]


# ---------------------------------------------------------------- helpers


def read_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def combined_digest(paths) -> str:
    """SHA-256 over sorted '<sha256>  <file name>\\n' lines; names never leave this function."""
    lines = sorted(f"{sha256_file(p)}  {Path(p).name}\n" for p in paths)
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def dumps(obj) -> str:
    return json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def pick(d, keys):
    return {k: d[k] for k in keys if isinstance(d, dict) and k in d}


def ci(c):
    return pick(c, CI_KEYS) if isinstance(c, dict) else None


def ppl_change_pct(c):
    """exp(ΔNLL) − 1 in %, derived from the published ΔNLL estimate and CI."""
    if not isinstance(c, dict) or c.get("estimate") is None:
        return None
    f = lambda x: None if x is None else (math.exp(x) - 1.0) * 100.0  # noqa: E731
    return {"estimate": f(c.get("estimate")), "ci_low": f(c.get("ci_low")), "ci_high": f(c.get("ci_high"))}


def boot_arm(label: str | None):
    for prefix, arm in BOOT_ARM:
        if label and label.startswith(prefix):
            return arm
    return None


def num_out(x):
    """Stable CSV cell for numbers; integral floats keep their integer value."""
    if x is None:
        return ""
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if math.isfinite(x) and x == int(x) and abs(x) < 2 ** 53:
            return str(int(x))
        return repr(x)
    return str(x)


def csv_text(header, rows) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    for row in rows:
        w.writerow([num_out(row.get(h)) for h in header])
    return buf.getvalue()


# ---------------------------------------------------------------- corpus classes


def classify_corpus(data: Path, repo: Path) -> dict:
    """Window id -> {key, class, category, n_tokens, sha256, prompt_id, prompt_set}. Private ids stay in memory."""
    manifest = read_json(data / "corpus/manifest.json")
    if not isinstance(manifest, dict):
        raise SystemExit("export: corpus manifest missing or unreadable")
    decode = read_json(data / "corpus/decode_manifest.json") or {}
    native_manifest = read_json(data / "corpus/native_prompts_manifest.json") or {}
    prov = (read_json(data / "corpus/native-provenance.json") or {}).get("windows", {})
    prompts = {p["id"]: p for p in decode.get("prompts", [])}
    prompts.update({p["id"]: dict(p, source="public") for p in native_manifest.get("prompts", [])})
    out, used = {}, {}
    for w in sorted(manifest["windows"], key=lambda w: w["id"]):
        entry = {"category": w.get("category"), "n_tokens": w.get("n_tokens"), "sha256": w.get("sha256")}
        src = w.get("source")
        if src == "synthetic_it":
            entry["class"] = "synthetic"
        elif src == "r0_native":
            pid = (prov.get(w["id"]) or {}).get("decode_id")
            prompt = prompts.get(pid)
            entry["class"] = "public" if prompt and prompt.get("source") == "public" else "private"
            if entry["class"] == "public":
                entry["prompt_id"] = pid
                entry["prompt_set"] = prompt.get("project")
                entry["prompt_sha256"] = prompt.get("sha256")
                entry["prompt_n_tokens"] = prompt.get("n_tokens")
        else:
            entry["class"] = "private"
        if entry["class"] == "public":
            n = used.get(entry["prompt_id"], 0) + 1
            used[entry["prompt_id"]] = n
            entry["key"] = entry["prompt_id"] + (f"-{n}" if n > 1 else "")
        else:
            entry["key"] = "h" + str(w.get("sha256") or "")[:12]
        out[w["id"]] = entry
    keys = [e["key"] for e in out.values()]
    if len(set(keys)) != len(keys):
        raise SystemExit("export: public/pseudonymous window keys are not unique")
    public_sets = {e["prompt_set"] for e in out.values() if e["class"] == "public" and e.get("prompt_set")}
    return {"windows": out, "manifest": manifest, "public_sets": sorted(public_sets),
            "native_source_ok": native_manifest.get("source_file_sha256") == (
                sha256_file(repo / "scripts/fidelity/native_prompts.json")
                if (repo / "scripts/fidelity/native_prompts.json").exists() else None),
            "provenance_skipped": (read_json(data / "corpus/native-provenance.json") or {}).get("skipped")}


def corpus_hashes(corpus) -> dict:
    rows = []
    for e in corpus["windows"].values():
        row = {"key": e["key"], "class": e["class"], "category": e["category"], "n_tokens": e["n_tokens"],
               "sha256": e["sha256"]}
        if e["class"] == "public":
            row["prompt_set"] = e.get("prompt_set")
        rows.append(row)
    rows.sort(key=lambda r: r["key"])
    m = corpus["manifest"]
    return {"schema": 1, "record_type": "fidelity_corpus_hashes", "experiment_id": EXPERIMENT_ID,
            "key_rule": "public windows: public prompt id (suffix -N if one prompt produced several windows); "
                        "other windows: 'h' + first 12 hex digits of the window sha256",
            "manifest_global_sha256": m.get("global_sha256"), "model_repo": m.get("model_repo"),
            "model_rev": m.get("model_rev"), "tokenizer_sha256": m.get("tokenizer_sha256"),
            "chat_template_sha256": m.get("chat_template_sha256"), "windows": rows}


# ---------------------------------------------------------------- runs


def run_dirs(data: Path):
    return sorted(p for p in (data / "raw").glob("*/*") if p.is_dir())


def redact_boot(boot: dict) -> dict:
    out = pick(boot, BOOT_KEYS)
    ranks = boot.get("ranks") or []
    out["rank_count"] = len(ranks)
    out["rank_image_ids"] = sorted({r.get("image_id") for r in ranks if r.get("image_id")})
    return out


def base_url_redacted(url):
    if not isinstance(url, str):
        return None
    m = re.match(r"^(\w+)://[^/:]+(:\d+)?(.*)$", url)
    return f"{m.group(1)}://<host>{m.group(2) or ''}{m.group(3)}" if m else "<redacted>"


def export_runs(data: Path, corpus) -> tuple[dict, list, list]:
    """runs.json content, per-window run rows for public windows, and private-original digests."""
    windows = corpus["windows"]
    runs, public_rows, originals = [], [], []
    for d in run_dirs(data):
        arm, name = d.parent.name, d.name
        rel = f"raw/{arm}/{name}"
        meta = read_json(d / "run.json")
        rec = {"arm": arm, "run": name}
        if isinstance(meta, dict):
            originals.append({"path": f"{rel}/run.json", "sha256": sha256_file(d / "run.json")})
            rec.update(pick(meta, RUN_KEYS))
            rec["base_url"] = base_url_redacted(meta.get("base_url"))
            rec["sessions"] = [pick(s, ("started_utc", "ended_utc")) for s in meta.get("sessions") or []]
            if isinstance(meta.get("boot"), dict):
                rec["boot"] = redact_boot(meta["boot"])
        wfiles = sorted((d / "prompt").glob("*.json"))
        if wfiles:
            originals.append({"path": f"{rel}/prompt/*.json", "files": len(wfiles),
                              "combined_sha256": combined_digest(wfiles)})
            agg = {"records": 0, "status": {}, "by_class": {}, "prompt_tokens": 0, "missing_actual": 0,
                   "http_attempts": 0, "retried_windows": 0, "elapsed_s": 0.0}
            for f in wfiles:
                w = read_json(f) or {}
                info = windows.get(w.get("id") or f.stem)
                cls = info["class"] if info else "unknown"
                agg["records"] += 1
                agg["status"][str(w.get("status"))] = agg["status"].get(str(w.get("status")), 0) + 1
                c = agg["by_class"].setdefault(cls, {"windows": 0, "n_tokens": 0})
                c["windows"] += 1
                c["n_tokens"] += int(w.get("n_tokens") or 0)
                agg["prompt_tokens"] += int(w.get("prompt_tokens") or 0)
                agg["missing_actual"] += int(w.get("missing_actual") or 0)
                agg["http_attempts"] += int(w.get("http_attempts") or 0)
                agg["retried_windows"] += int((w.get("http_attempts") or 0) > 1)
                agg["elapsed_s"] += float(w.get("elapsed_s") or 0.0)
                if info and cls == "public":
                    public_rows.append({"key": info["key"], "arm": arm, "run": name, "status": w.get("status"),
                                        "K": w.get("K"), "n_tokens": w.get("n_tokens"),
                                        "prompt_tokens": w.get("prompt_tokens"),
                                        "missing_actual": w.get("missing_actual"),
                                        "http_attempts": w.get("http_attempts"), "elapsed_s": w.get("elapsed_s"),
                                        "cache_salt_sha256": w.get("cache_salt_sha256")})
            agg["elapsed_s"] = round(agg["elapsed_s"], 3)
            rec["windows"] = agg
        gfiles = sorted(p for p in d.glob("[dn][0-9]*.json"))
        if gfiles:
            rec["kind"] = rec.get("kind") or "native_generation"
            rec["generation"] = native_generation(d, gfiles, corpus, originals, rel)
        if not wfiles and not gfiles:
            rec["windows"] = {"records": 0, "note": "run metadata only; no per-window records in this run"}
        runs.append(rec)
    public_rows.sort(key=lambda r: (r["key"], r["arm"], r["run"]))
    return {"schema": 1, "record_type": "fidelity_runs", "experiment_id": EXPERIMENT_ID,
            "notes": "Rank records (host names, command lines, log lines, start times) are dropped; rank count "
                     "and image ids remain. Salts appear only as SHA-256 digests in window-runs-public.csv. "
                     "Window aggregates are grouped by source class (public, private, synthetic).",
            "runs": runs}, public_rows, originals


def native_generation(d: Path, gfiles, corpus, originals, rel) -> dict:
    """Aggregate native-generation records by source class; public prompts also yield rows."""
    by_prompt = {e["prompt_id"]: e for e in corpus["windows"].values() if e.get("prompt_id")}
    originals.append({"path": f"{rel}/[dn]*.json", "files": len(gfiles), "combined_sha256": combined_digest(gfiles)})
    agg = {}
    for f in gfiles:
        g = read_json(f) or {}
        cls = "public" if f.stem in by_prompt else "private"
        a = agg.setdefault(cls, {"prompts": 0, "prompt_tokens": 0, "gen_tokens": 0, "finish_reason": {},
                                 "status": {}, "elapsed_s": 0.0})
        a["prompts"] += 1
        a["prompt_tokens"] += int(g.get("prompt_tokens") or 0)
        a["gen_tokens"] += int(g.get("gen_tokens") or 0)
        fr = str(g.get("finish_reason"))
        a["finish_reason"][fr] = a["finish_reason"].get(fr, 0) + 1
        st = str(g.get("status"))
        a["status"][st] = a["status"].get(st, 0) + 1
        a["elapsed_s"] += float(g.get("elapsed_s") or 0.0)
    for a in agg.values():
        a["elapsed_s"] = round(a["elapsed_s"], 3)
    return {"by_class": agg, "decode_prompts_without_window": corpus.get("provenance_skipped")}


def windows_public(data: Path, corpus) -> list:
    rows = []
    gen_dir = data / "raw/R0/native"
    for e in corpus["windows"].values():
        if e["class"] != "public":
            continue
        g = read_json(gen_dir / f"{e['prompt_id']}.json") or {}
        rows.append({"key": e["key"], "prompt_id": e["prompt_id"], "prompt_set": e.get("prompt_set"),
                     "prompt_sha256": e.get("prompt_sha256"), "prompt_n_tokens": e.get("prompt_n_tokens"),
                     "category": e["category"], "window_n_tokens": e["n_tokens"], "window_sha256": e["sha256"],
                     "gen_prompt_tokens": g.get("prompt_tokens"), "gen_tokens": g.get("gen_tokens"),
                     "finish_reason": g.get("finish_reason"), "seed": g.get("seed"),
                     "temperature": g.get("temperature"), "top_p": g.get("top_p"), "max_tokens": g.get("max_tokens"),
                     "gen_elapsed_s": g.get("elapsed_s")})
    rows.sort(key=lambda r: r["key"])
    return rows


WINDOWS_PUBLIC_HEADER = ["key", "prompt_id", "prompt_set", "prompt_sha256", "prompt_n_tokens", "category",
                         "window_n_tokens", "window_sha256", "gen_prompt_tokens", "gen_tokens", "finish_reason",
                         "seed", "temperature", "top_p", "max_tokens", "gen_elapsed_s"]
WINDOW_RUNS_HEADER = ["key", "arm", "run", "status", "K", "n_tokens", "prompt_tokens", "missing_actual",
                      "http_attempts", "elapsed_s", "cache_salt_sha256"]
WINDOW_METRICS_HEADER = ["key", "comparison", "regime", "side"] + list(FIELDS)


def window_metrics(data: Path, corpus, originals) -> list | None:
    files = sorted((data / "metrics-v2").glob("*.npz"))
    if not files:
        return None
    try:
        import numpy as np
    except ImportError:
        raise SystemExit("export: numpy is required to read metrics-v2/*.npz "
                         "(use data/fidelity/.venv/bin/python)")
    windows = corpus["windows"]
    rows = []
    for path in files:
        originals.append({"path": f"metrics-v2/{path.name}", "sha256": sha256_file(path)})
        with np.load(path, allow_pickle=False) as npz:
            z = {k: npz[k] for k in npz.files}
            ids = [str(x) for x in z["window_ids"]]
            for i, wid in enumerate(ids):
                info = windows.get(wid)
                if not info or info["class"] != "public":
                    continue
                for regime in REGIMES:
                    for side in SIDES:
                        cols = {"count": f"{regime}_count"}
                        cols.update({f: f"{regime}_{side}_{f}" for f in FIELDS if f != "count"})
                        if cols["kl_sum"] not in z:
                            continue
                        row = {"key": info["key"], "comparison": path.stem, "regime": regime, "side": side}
                        for f, col in cols.items():
                            v = z[col][i].item() if col in z else None
                            row[f] = v
                        rows.append(row)
    groups = data / "metrics-v2/groups.json"
    if groups.exists():
        originals.append({"path": "metrics-v2/groups.json", "sha256": sha256_file(groups)})
    rows.sort(key=lambda r: (r["key"], r["comparison"], REGIMES.index(r["regime"]), SIDES.index(r["side"])))
    return rows


# ---------------------------------------------------------------- memory, gates, determinism


def export_memory(data: Path, originals) -> tuple[str, dict, list]:
    lines, summary, aborts = [], {}, []
    for path in sorted((data / "mem").glob("*.jsonl")):
        originals.append({"path": f"mem/{path.name}", "sha256": sha256_file(path)})
        boot = path.stem
        per_rank = {}
        for raw in path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            rec = json.loads(raw)
            out = {"boot": boot, "rank": rec.get("rank"), "ts": rec.get("ts"),
                   "mem_available_kib": rec.get("mem_available_kib")}
            if "error" in rec:
                out["error"] = rec["error"]
            lines.append(json.dumps(out, sort_keys=True))
            r = per_rank.setdefault(str(rec.get("rank")), {"values": [], "errors": 0})
            if isinstance(rec.get("mem_available_kib"), (int, float)):
                r["values"].append(rec["mem_available_kib"])
            if "error" in rec:
                r["errors"] += 1
        summary[boot] = {}
        for rank, r in sorted(per_rank.items()):
            v = sorted(r["values"])
            summary[boot][rank] = {"samples": len(v) + r["errors"], "errors": r["errors"],
                                   "min_kib": v[0] if v else None,
                                   "p05_kib": v[max(0, math.ceil(0.05 * len(v)) - 1)] if v else None,
                                   "median_kib": statistics.median(v) if v else None}
    for path in sorted((data / "mem").glob("ABORT-*")):
        originals.append({"path": f"mem/{path.name}", "sha256": sha256_file(path)})
        text = path.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"^(\S+)\s+rank(\d+)\s+MemAvailable\s+(\d+)\s+KiB", text)
        aborts.append({"marker": path.name, "ts": m.group(1) if m else None,
                       "rank": int(m.group(2)) if m else None, "mem_available_kib": int(m.group(3)) if m else None})
    return "".join(line + "\n" for line in lines), summary, aborts


def export_determinism(data: Path, corpus, originals) -> dict:
    windows = corpus["windows"]
    letters, probes = {}, []
    for path in sorted((data / "prelim").glob("determinism-*.json")):
        d = read_json(path)
        if not isinstance(d, dict):
            continue
        originals.append({"path": f"prelim/{path.name}", "sha256": sha256_file(path)})
        role = {}
        for r in ("x", "y"):
            wid = d.get(r)
            if wid not in letters:
                letters[wid] = chr(ord("A") + len(letters))
            info = windows.get(wid) or {}
            role[r.upper()] = f"{info.get('class', 'unknown')} window {letters[wid]}"
        short = {k: v.rsplit(" ", 1)[-1] for k, v in role.items()}
        pairs = []
        for p in d.get("pairs", []):
            q = pick(p, ("passes", "positions", "identical", "first_nonidentical_pos", "first_over_threshold_pos",
                         "over_threshold", "max_abs_delta", "top1_agreement"))
            q["window"] = role.get(p.get("window"), "unknown")
            q["preceded_by"] = [short.get(x, x) for x in p.get("preceded_by", [])]
            pairs.append(q)
        firsts = [p["first_nonidentical_pos"] for p in pairs if p.get("first_nonidentical_pos") is not None]
        probes.append({"boot": path.stem[len("determinism-"):], "k": d.get("k"), "threshold": d.get("threshold"),
                       "sequence": "".join(short.get(c, c) for c in str(d.get("sequence", ""))),
                       "windows": sorted(set(role.values())), "pairs": pairs,
                       "summary": {"pairs": len(pairs), "min_first_nonidentical_pos": min(firsts) if firsts else None,
                                   "max_abs_delta": max((p.get("max_abs_delta") or 0) for p in pairs) if pairs else None,
                                   "positions_compared": sum(p.get("positions") or 0 for p in pairs)}})
    return {"schema": 1, "record_type": "fidelity_determinism_probe", "experiment_id": EXPERIMENT_ID,
            "notes": "Same two windows re-scored in the listed pass sequence on one boot; windows are labelled by "
                     "source class and letter. Positions are 0-based prompt positions; max_abs_delta is the largest "
                     "absolute top-K logprob difference.",
            "probes": probes}


# ---------------------------------------------------------------- results


def load_public(docs: Path) -> dict:
    metrics = {p.stem: read_json(p) for p in sorted((docs / "metrics-v2").glob("*.json"))}
    boots = {}
    for p in sorted((docs / "boots").glob("*.json")):
        b = read_json(p)
        if isinstance(b, dict):
            boots[b.get("label") or p.stem] = b
    tasks = {p.stem: read_json(p) for p in sorted((docs / "metrics").glob("tasks-*.json"))}
    return {"metrics": metrics, "summary": metrics.get("summary") or {}, "boots": boots, "tasks": tasks,
            "corpus": read_json(docs / "corpus-summary.json") or {}}


def headline(pub) -> dict:
    out = {}
    for name in HEADLINE_PAIRS:
        res = pub["metrics"].get(name)
        if not isinstance(res, dict):
            out[name] = {"status": "missing"}
            continue
        entry = {"status": res.get("status"), "ref": res.get("ref"), "cand": res.get("cand"),
                 "role": res.get("role"), "windows": res.get("windows"), "regimes": {}}
        for regime in REGIMES:
            r = (res.get("regimes") or {}).get(regime) or {}
            c, fl, ex = r.get("contrast") or {}, r.get("floor") or {}, r.get("excess") or {}
            kl = c.get("kl") or {}
            entry["regimes"][regime] = {
                "positions": r.get("positions"), "windows": r.get("windows"), "groups": r.get("groups"),
                "kl_mean": ci(kl.get("mean_ci")) or {"estimate": kl.get("mean")},
                "kl_median": kl.get("median"), "kl_p99": kl.get("p99"),
                "floor_kl_mean": (fl.get("kl") or {}).get("mean"),
                "top1_agreement": ci(c.get("top1_agreement")),
                "delta_nll_nats": ci(c.get("delta_nll")),
                "perplexity_change_pct": ppl_change_pct(c.get("delta_nll")),
                "excess": {k: ci(v) for k, v in sorted(ex.items())},
                "verdict": ((res.get("verdict") or {}).get(regime) or {}).get("outcome"),
            }
        out[name] = entry
    return out


def arms_block(pub, runs) -> dict:
    desc = {a[0]: {"boots_hint": a[1], "description": a[2], "role": a[3]} for a in br.ARMS}
    arms = {}
    for label, b in sorted(pub["boots"].items()):
        arm = boot_arm(label)
        if not arm:
            continue
        a = arms.setdefault(arm, {"boots": []})
        a["boots"].append(pick(b, BOOT_KEYS))
    for rec in runs["runs"]:
        a = arms.setdefault(rec["arm"], {"boots": []})
        a.setdefault("runs", []).append(rec["run"])
    for arm, a in arms.items():
        a.update(desc.get("Ladder" if arm in ("L0919", "LE21", "LE22b") else arm, {}))
        a["boots"].sort(key=lambda b: b.get("label") or "")
    return dict(sorted(arms.items()))


def tasks_block(data: Path, pub) -> dict:
    per_arm = {}
    for path in sorted((data / "tasks").rglob("*.json")):
        t = read_json(path) or {}
        arm = path.relative_to(data / "tasks").parts[0]
        key = f"{arm}/{t.get('set')}/{t.get('mode')}/run{t.get('run')}"
        a = per_arm.setdefault(key, {"items": 0, "pass": 0, "fail": 0, "request_ok": 0, "finish_reason": {},
                                     "completion_tokens": 0, "wall_s": 0.0})
        a["items"] += 1
        ok = bool((t.get("grader") or {}).get("pass"))
        a["pass" if ok else "fail"] += 1
        a["request_ok"] += int(bool(t.get("ok")))
        fr = str((t.get("response") or {}).get("finish_reason"))
        a["finish_reason"][fr] = a["finish_reason"].get(fr, 0) + 1
        a["completion_tokens"] += int((t.get("usage") or {}).get("completion_tokens") or 0)
        a["wall_s"] += float(t.get("wall_s") or 0.0)
    for a in per_arm.values():
        a["wall_s"] = round(a["wall_s"], 3)
    comps = {name: pick(v, ("set", "arms", "greedy_discordance", "mcnemar_exact_p", "wilson_ci", "mde_approx"))
             for name, v in pub["tasks"].items() if isinstance(v, dict)}
    return {"per_run": per_arm, "comparisons": comps,
            "note": "Public knapcio qeval tasks, greedy decoding; full per-item results under tasks/."}


def results(data, docs, corpus, pub, runs, originals, mem_summary, aborts, public_count, extract_files) -> dict:
    summary = pub["summary"]
    cm = pub["metrics"].get("cm-vs-r0") or {}
    verdict = cm.get("verdict") or {}
    r0_labels = sorted({r["boot"]["label"] for r in runs["runs"] if r["arm"] == "R0" and r.get("boot")})
    classes = {}
    for e in corpus["windows"].values():
        c = classes.setdefault(e["class"], {"windows": 0, "tokens": 0, "by_category": {}})
        c["windows"] += 1
        c["tokens"] += int(e["n_tokens"] or 0)
        c["by_category"][e["category"]] = c["by_category"].get(e["category"], 0) + 1
    prompt_runs = [r for r in runs["runs"] if isinstance(r.get("windows"), dict) and r["windows"].get("records")]
    status = {}
    for r in prompt_runs:
        for k, v in r["windows"]["status"].items():
            status[k] = status.get(k, 0) + v
    cachehit = read_json(data / "smoke/r0fp8-m-cachehit.json") or {}
    corruption = {}
    for p in sorted((data / "probes").glob("corruption-*.json")):
        corruption[p.stem.split("-", 1)[1]] = (read_json(p) or {}).get("summary")
    exclude = data / "corpus/exclude.txt"
    active_excl = [l for l in exclude.read_text(encoding="utf-8").splitlines()
                   if l.strip() and not l.lstrip().startswith("#")] if exclude.exists() else []
    for rel in ("corpus/manifest.json", "corpus/decode_manifest.json", "corpus/native-provenance.json",
                "corpus/native_prompts_manifest.json", "corpus/build-info.json", "corpus/exclude.txt"):
        if (data / rel).exists():
            originals.append({"path": rel, "sha256": sha256_file(data / rel)})
    published = [{"path": p.relative_to(docs).as_posix(), "sha256": sha256_file(p)}
                 for p in sorted(docs.rglob("*")) if p.is_file() and not p.name.startswith(".")]
    return {
        "schema": 1,
        "record_type": "fidelity_campaign_extract",
        "experiment_id": EXPERIMENT_ID,
        "title": "GLM-5.3-Flash fidelity campaign: E29 serving recipe and NVFP4 versus the vendor FP8 model (R0)",
        "outcome": {regime: (verdict.get(regime) or {}).get("outcome") for regime in REGIMES},
        "outcome_detail": {"comparison": "cm-vs-r0", "verdict": verdict,
                           "source": "docs/fidelity/metrics-v2/cm-vs-r0.json"},
        "reference": {"arm": "R0", "measurement_boots": [pick(pub["boots"].get(l) or {"label": l}, BOOT_KEYS)
                                                         for l in r0_labels],
                      "serving_boots": sorted(l for l in pub["boots"] if l.startswith("r0-s"))},
        "arms": arms_block(pub, runs),
        "coverage": {
            "windows": len(corpus["windows"]),
            "by_source_class": dict(sorted(classes.items())),
            "public_window_rows": public_count,
            "scored_positions": pub["corpus"].get("scored_positions"),
            "positions_by_regime": {regime: ((cm.get("regimes") or {}).get(regime) or {}).get("positions")
                                    for regime in REGIMES},
            "grouping": summary.get("grouping"),
            "bootstrap": summary.get("bootstrap"),
            "thresholds": summary.get("thresholds"),
            "runs_per_arm": {arm: sorted(r["run"] for r in runs["runs"] if r["arm"] == arm)
                             for arm in sorted({r["arm"] for r in runs["runs"]})},
            "native_prompt_source_matches_public_file": corpus["native_source_ok"],
        },
        "headline": headline(pub),
        "tasks": tasks_block(data, pub),
        "corruption": corruption,
        "integrity": {
            "window_status": dict(sorted(status.items())),
            "missing_actual_token_entries": sum(r["windows"]["missing_actual"] for r in prompt_runs),
            "retried_windows": sum(r["windows"]["retried_windows"] for r in prompt_runs),
            "metrics_missingness": {k: v.get("missingness_vs_manifest")
                                    for k, v in sorted((summary.get("runs") or {}).items())},
            "cache_hit_suppression": {
                "probe": "smoke/r0fp8-m-cachehit.json",
                "cold_prompt_logprobs": (cachehit.get("cold") or {}).get("prompt_logprobs_len"),
                "same_salt_repeat_prompt_logprobs": (cachehit.get("same_salt_repeat") or {}).get("prompt_logprobs_len"),
                "fresh_salt_prompt_logprobs": (cachehit.get("fresh_salt") or {}).get("prompt_logprobs_len"),
                "campaign_salt_policy": next((r.get("request_defaults", {}).get("cache_salt") for r in runs["runs"]
                                              if r.get("request_defaults", {}).get("cache_salt")), None),
            } if cachehit else "unknown (no cache-hit probe found)",
            "memory_aborts": aborts,
            "memory_sampler_timeouts": sum(r["errors"] for b in mem_summary.values() for r in b.values()),
            "memory_by_boot": mem_summary,
        },
        "exclusions": [
            {"what": "per-window rows for private-session windows, model-native windows from private prompts and "
                     "synthetic Italian windows",
             "count": sum(1 for e in corpus["windows"].values() if e["class"] != "public"),
             "reason": "private or derived from private sessions; published only as aggregates and digests"},
            {"what": "decode prompts without a native window", "count": corpus.get("provenance_skipped"),
             "reason": "recorded as skipped by the native generation step"},
            {"what": "owner corpus exclusions", "count": len(active_excl)},
            {"what": "runs with metadata only and no per-window records",
             "runs": sorted(f"{r['arm']}/{r['run']}" for r in runs["runs"]
                            if (r.get("windows") or {}).get("records") == 0)},
            {"what": "corpus text, token ids, logprob arrays, transcripts, logs, audits, inventories, parity "
                     "captures, tokenizer files, private notes, preliminary aggregates and operator runbooks",
             "reason": "private content or site-specific session records; the public report and this extract "
                       "carry the reusable results"},
        ],
        "limitations": LIMITATIONS,
        "published_files": published,
        "extract_files": extract_files,
        "private_originals": sorted(originals, key=lambda o: o["path"]),
    }


# ---------------------------------------------------------------- README


README = """# Fidelity campaign extract (2026-09-27)

Portable, privacy-safe extract of the GLM-5.3-Flash fidelity campaign: how far the E29
serving recipe (Cm in measurement mode, Cp in production) and the NVFP4 recipe (N)
deviate from the vendor FP8 model served by the September 18 recipe (R0), with the
recipe before E21 (Cpre) and the ladder arms (L0919, LE21, LE22b). The readable report
is [the fidelity report](../../../fidelity/REPORT.md); this folder holds the numbers
behind it. Every file here is generated; do not edit by hand.

## Files

| File | Contents |
| --- | --- |
| `results.json` | Outcome per regime, reference and arm identities, coverage, headline metrics (copied from the public aggregates; perplexity change is exp(ΔNLL) − 1), task and corruption summaries, integrity, exclusions, limitations, digests of the public report files, of every file in this folder and of the private originals |
| `runs.json` | Every measurement run: request settings, redacted boot identity (rank count and image ids, no rank records) and per-run window aggregates by source class |
| `corpus-hashes.json` | One row per corpus window: key, source class, category, token count and content SHA-256 |
| `windows-public.csv` | Model-native windows generated from public prompts: prompt provenance and generation settings |
| `window-metrics-public.csv` | Per-window sums for those windows from every comparison: positions, top-20 KL, top-1 agreement and covered mass, per regime and side (contrast or floor) |
| `window-runs-public.csv` | Per-run records for those windows: status, token counts, attempts, time and cache-salt digest |
| `determinism.json` | Same-arm re-scoring probes per boot: first differing position and largest logprob difference per pass pair |
| `memory.jsonl` | Host memory samples per boot and rank (host names removed) |
| `gates/` | Functional gate results after each boot |
| `probes/` | Corruption probes (UTF-8 and tool-call flags, no text) for Cp and N |
| `tasks/` | Per-item qeval results for R0, Cp and N: public tasks, model answers and grader verdicts |
| `smoke/` | Logprob and cache-hit smoke checks |
| `nvfp4-manifest-hf.json` | File digests of the NVFP4 checkpoint snapshot |

## What stays private

Corpus text, token ids, logprob arrays, transcripts, session logs, reviewer audits,
host inventories, parity captures, tokenizer files and operator runbooks stay in the
ignored campaign data. Windows from private coding sessions, model-native windows from
private prompts and the synthetic Italian conversations appear only in aggregates and
as digests. Their key is `h` plus the first 12 hex digits of the window SHA-256; the
campaign's internal window ids are not published. Public windows use the public prompt
id (knapcio hardset/qeval `d` prompts, `scripts/fidelity/native_prompts.json` `n`
prompts). Host names, addresses, usernames, absolute paths, project names and raw cache
salts are removed; salts appear only as SHA-256 digests.

`private_originals` in `results.json` identifies each private source by its path
relative to the campaign data directory and its SHA-256. A directory of per-window or
per-prompt records is identified by a combined digest: SHA-256 of the sorted lines
`<sha256>  <file name>\\n`.

## Regenerate

```sh
data/fidelity/.venv/bin/python scripts/fidelity/export_public.py
```

The script needs the ignored campaign data, numpy for the per-window metrics, and the
public aggregates in `docs/fidelity/`. It builds the folder in a temporary directory,
runs a leak check (absolute paths, addresses, window ids, site host names and
addresses, local username, corpus project names and credential patterns), and replaces
this folder only when the check is clean. Output is deterministic. Allowlisted: IPv4
literals that occur in the public qeval task source, loopback and the unspecified
address `0.0.0.0` in model answers, and the names of the public prompt sets.
"""


# ---------------------------------------------------------------- leak check


def site_terms(cluster_env: Path | None, manifest: Path | None, public_sets) -> list:
    terms = set(br.private_terms(cluster_env, manifest, ""))
    terms = {(label, t) for label, t in terms if not (label == "corpus project name" and t in public_sets)}
    if cluster_env and cluster_env.exists():
        text = cluster_env.read_text(encoding="utf-8", errors="replace")
        for ip in IPV4.findall(text):
            if ip not in EXTRA_IPV4_ALLOW:
                terms.add(("site address", ip))
        for m in re.finditer(r"^\s*RELAY_DEST=[\"']?([A-Za-z_][\w.-]*)@", text, re.M):
            if len(m.group(1)) >= 3:
                terms.add(("site user", m.group(1)))
    try:
        user = getpass.getuser()
    except Exception:  # pragma: no cover - no user database
        user = ""
    if len(user) >= 3:
        terms.add(("local username", user))
    home = os.path.expanduser("~")
    if home and home != "~" and len(home) > 1:
        terms.add(("home path", home))
    return sorted(terms)


def ipv4_allowlist(qeval_tasks: Path | None) -> set:
    allow = set(EXTRA_IPV4_ALLOW)
    if qeval_tasks and qeval_tasks.exists():
        allow |= set(IPV4.findall(qeval_tasks.read_text(encoding="utf-8", errors="replace")))
    return allow


def scan(files, terms, allow_ips) -> list:
    findings = []
    for path in files:
        path = Path(path)
        if path.suffix in (".npz", ".u32"):
            findings.append((path, "forbidden file type"))
            continue
        hits = {label for _, label in br.leak_check([path], terms)}
        text = path.read_text(encoding="utf-8", errors="replace")
        if hits & {"IPv4 address", "private address"}:
            masked = IPV4.sub(lambda m: "" if m.group(0) in allow_ips else m.group(0), text)
            for label, pattern in br.GENERIC_PATTERNS:
                if label in ("IPv4 address", "private address") and not pattern.search(masked):
                    hits.discard(label)
        for label, pattern in SECRET_PATTERNS + EXTRA_PATTERNS:
            if pattern.search(text):
                hits.add(label)
        findings += [(path, label) for label in hits]
    return sorted(findings, key=lambda f: (str(f[0]), f[1]))


# ---------------------------------------------------------------- main


def copy_tree(src: Path, dst: Path, pattern: str):
    for p in sorted(src.glob(pattern)):
        if p.is_file():
            target = dst / p.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, target)


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build(args, tmp: Path) -> Path:
    data, docs = args.data, args.docs
    corpus = classify_corpus(data, args.repo)
    pub = load_public(docs)
    runs, run_rows, originals = export_runs(data, corpus)
    metric_rows = window_metrics(data, corpus, originals)
    mem_text, mem_summary, aborts = export_memory(data, originals)
    determinism = export_determinism(data, corpus, originals)
    wp = windows_public(data, corpus)

    write_text(tmp / "runs.json", dumps(runs))
    write_text(tmp / "corpus-hashes.json", dumps(corpus_hashes(corpus)))
    write_text(tmp / "windows-public.csv", csv_text(WINDOWS_PUBLIC_HEADER, wp))
    write_text(tmp / "window-runs-public.csv", csv_text(WINDOW_RUNS_HEADER, run_rows))
    if metric_rows is not None:
        write_text(tmp / "window-metrics-public.csv", csv_text(WINDOW_METRICS_HEADER, metric_rows))
    write_text(tmp / "determinism.json", dumps(determinism))
    write_text(tmp / "memory.jsonl", mem_text)
    copy_tree(data / "boots", tmp / "gates", "*-gates.json")
    copy_tree(data / "probes", tmp / "probes", "corruption-*.json")
    copy_tree(data / "tasks", tmp / "tasks", "**/*.json")
    copy_tree(data / "smoke", tmp / "smoke", "*.json")
    if (data / "nvfp4/manifest-hf.json").exists():
        shutil.copyfile(data / "nvfp4/manifest-hf.json", tmp / "nvfp4-manifest-hf.json")
    write_text(tmp / "README.md", README)
    extract_files = [{"path": p.relative_to(tmp).as_posix(), "sha256": sha256_file(p)}
                     for p in sorted(tmp.rglob("*")) if p.is_file()]
    res = results(data, docs, corpus, pub, runs, originals, mem_summary, aborts, len(wp), extract_files)
    write_text(tmp / "results.json", dumps(res))
    return tmp


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, default=REPO / "data/fidelity", help="ignored campaign data")
    p.add_argument("--docs", type=Path, default=REPO / "docs/fidelity", help="public campaign aggregates")
    p.add_argument("--out", type=Path, default=REPO / "docs/historical_benchmarks/experiments" / EXPERIMENT_ID)
    p.add_argument("--repo", type=Path, default=REPO, help="checkout holding scripts/fidelity/native_prompts.json")
    p.add_argument("--cluster-env", type=Path, default=REPO / "cluster.env",
                   help="ignored site configuration, read only for the leak check")
    p.add_argument("--qeval-tasks", type=Path, default=REPO / "third_party/knapcio-bench/bench/qeval_tasks.py",
                   help="public task source whose IPv4 literals are allowlisted")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out = args.out
    tmp = out.parent / (out.name + ".export-tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        build(args, tmp)
        corpus_sets = classify_corpus(args.data, args.repo)["public_sets"]
        terms = site_terms(args.cluster_env, args.data / "corpus/manifest.json", corpus_sets)
        files = sorted(p for p in tmp.rglob("*") if p.is_file())
        findings = scan(files, terms, ipv4_allowlist(args.qeval_tasks))
        if findings:
            for path, label in findings:
                print(f"LEAK {path.relative_to(tmp).as_posix()}: {label}", file=sys.stderr)
            shutil.rmtree(tmp)
            return 2
        if out.exists():
            shutil.rmtree(out)
        tmp.rename(out)
    except BaseException:
        if tmp.exists():
            shutil.rmtree(tmp)
        raise
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"leak check: clean ({len(files)} files, {len(br.GENERIC_PATTERNS) + len(SECRET_PATTERNS) + len(EXTRA_PATTERNS)}"
          f" patterns, {len(terms)} private terms)")
    print(f"wrote {out.name}: {len(files)} files, {total:,} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
