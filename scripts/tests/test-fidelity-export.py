#!/usr/bin/env python3
"""Offline tests for scripts/fidelity/export_public.py; stdlib only, no cluster or network.

Builds a tiny synthetic campaign tree (one private-session window, one synthetic Italian
window, native windows from a public, a private and a native prompt, one run with rank
records, memory samples with host names, a determinism probe, a task answer with public
IPv4 literals) and checks that rank and host fields are dropped, that only public-native
windows get per-window rows keyed by their public prompt id, that no internal window id
reaches the output, that an injected site value fails the leak check without writing the
output, and that two runs produce identical bytes. With numpy installed, the per-window
metric rows from metrics-v2/*.npz are checked too.
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "fidelity"))

import export_public as ep  # noqa: E402

try:
    import numpy as np
except ImportError:  # pragma: no cover - system interpreter without numpy
    np = None

SITE_HOST = "nodea-sitehost"
SITE_IP = "10.20.30.40"
PRIVATE_PROJECT = "secret-proj-alpha"


def write(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data if isinstance(data, str) else json.dumps(data, indent=1), encoding="utf-8")


def sha(c: str) -> str:
    return c * 64


def window(wid, category, source, project, n, digest):
    return {"id": wid, "category": category, "source": source, "project": project, "n_tokens": n,
            "sha256": digest, "path": f"tokens/{wid}.u32"}


def ci(x):
    return {"estimate": x, "ci_low": x * 0.9, "ci_high": x * 1.1, "lb95": x * 0.92, "ub95": x * 1.08, "B": 200}


def regime(kl):
    c = {"kl": {"mean": kl, "mean_ci": ci(kl), "p99": kl * 10, "median": kl / 3}, "top1_agreement": ci(0.9),
         "delta_nll": ci(0.001)}
    return {"positions": 100, "windows": 5, "groups": 4, "contrast": c, "floor": {"kl": {"mean": 0.0}},
            "excess": {"kl_mean_excess": ci(kl), "top1_drop_pp": ci(1.0)}}


def build_tree(root: Path, numpy_metrics: bool) -> dict:
    data, docs, repo = root / "data", root / "docs", root / "repo"
    ids = ["w0001", "w0002", "w0003", "w0004", "w0005"]
    write(data / "corpus/manifest.json", {"global_sha256": sha("0"), "model_repo": "zai-org/GLM-5.3-Flash",
                                          "model_rev": "r", "windows": [
        window("w0001", "agentic_code", "claude_code", PRIVATE_PROJECT, 3000, sha("a")),
        window("w0002", "italian_chat", "synthetic_it", "synthetic-it", 900, sha("b")),
        window("w0003", "model_native", "r0_native", "knapcio-bench/hardset", 300, sha("c")),
        window("w0004", "model_native", "r0_native", "private-proj-beta", 400, sha("d")),
        window("w0005", "model_native", "r0_native", "native_prompts", 200, sha("e"))]})
    write(data / "corpus/decode_manifest.json", {"prompts": [
        {"id": "d001", "category": "session", "source": "omp", "project": "private-proj-beta", "n_tokens": 50,
         "sha256": sha("1")},
        {"id": "d101", "category": "public", "source": "public", "project": "knapcio-bench/hardset",
         "n_tokens": 40, "sha256": sha("2")}]})
    write(data / "corpus/native_prompts_manifest.json", {"source_file_sha256": sha("9"), "prompts": [
        {"id": "n001", "category": "native_prompt", "source": "public", "project": "native_prompts",
         "n_tokens": 30, "sha256": sha("3")}]})
    write(data / "corpus/native-provenance.json", {"skipped": {"missing": 1}, "windows": {
        "w0003": {"decode_id": "d101"}, "w0004": {"decode_id": "d001"}, "w0005": {"decode_id": "n001"}}})
    write(data / "corpus/exclude.txt", "# comment only\n")
    boot = {"label": "cm-1", "overlay_sha256": sha("f"), "image": "img@sha256:" + sha("7"), "kv_cache_dtype": "fp8",
            "ranks": [{"rank": r, "host": SITE_HOST, "cmd": ["vllm", "--host", SITE_IP], "image_id": "sha256:x",
                       "signature_lines": [f"ready on {SITE_HOST}"]} for r in range(4)]}
    for arm in ("R0", "Cm"):
        run = data / "raw" / arm / "prompt-a"
        write(run / "run.json", {"schema": "fidelity-raw/1", "arm": arm, "run": "prompt-a", "kind": "prompt",
                                 "base_url": f"http://{SITE_IP}:8000", "boot": dict(boot, label=f"{arm.lower()}-1"),
                                 "request_defaults": {"cache_salt": "fresh 32-byte hex per HTTP attempt"},
                                 "sessions": [{"started_utc": "t0", "ended_utc": None, "harness_git": {}}]})
        for i, wid in enumerate(ids):
            write(run / "prompt" / f"{wid}.json", {"id": wid, "status": "ok", "K": 20, "n_tokens": 100 + i,
                                                   "prompt_tokens": 100 + i, "missing_actual": 0,
                                                   "http_attempts": 1, "elapsed_s": 0.5,
                                                   "cache_salt_sha256": sha(str(i))})
    write(data / "raw/Cm/gen-nospec/run.json", {"arm": "Cm", "run": "gen-nospec", "kind": "gen", "boot": boot})
    for pid in ("d001", "d101", "n001"):
        write(data / f"raw/R0/native/{pid}.json", {"status": "ok", "seed": 7, "prompt_tokens": 40, "gen_tokens": 9,
                                                   "finish_reason": "stop", "elapsed_s": 1.0, "temperature": 1.0,
                                                   "top_p": 0.95, "max_tokens": 64})
    write(data / "mem/cm-1-measure.jsonl", "".join(
        json.dumps({"ts": f"t{i}", "rank": i % 4, "host": SITE_HOST, "mem_available_kib": 1000 + i}) + "\n"
        for i in range(8)) + json.dumps({"ts": "t9", "rank": 0, "host": SITE_HOST, "mem_available_kib": None,
                                         "error": "timeout"}) + "\n")
    write(data / "mem/ABORT-r0-s-1-incident", "2026-09-27T01:00:00+00:00 rank0 MemAvailable 700000 KiB\n")
    write(data / "boots/cm-1-gates.json", {"health": 200, "gate1": {"pass": True, "content": "Paris"}, "pass": True})
    write(data / "probes/corruption-Cp.json", {"summary": {"label": "Cp", "italian_prompts": 1},
                                               "rows": [{"kind": "italian", "i": 0, "fffd": 0}]})
    write(data / "tasks/R0/qeval/greedy/run1/json_nested.json", {
        "ok": True, "arm": "local", "set": "qeval", "mode": "greedy", "run": 1, "item": "json_nested",
        "response": {"content": '{"host": "0.0.0.0", "check": "192.168.0.1"}', "finish_reason": "stop"},
        "usage": {"completion_tokens": 5}, "wall_s": 1.0, "grader": {"pass": True}})
    write(data / "prelim/determinism-r0.json", {"x": "w0001", "y": "w0002", "sequence": "XXY", "k": 20,
                                                "threshold": 1.0, "pairs": [
        {"window": "X", "passes": [0, 1], "preceded_by": ["-", "X"], "positions": 2999, "identical": 2050,
         "first_nonidentical_pos": 2052, "first_over_threshold_pos": 2060, "over_threshold": 3,
         "max_abs_delta": 20.5, "top1_agreement": 0.9}]})
    write(data / "smoke/r0fp8-m-cachehit.json", {"cold": {"prompt_logprobs_len": 8000},
                                                 "same_salt_repeat": {"prompt_logprobs_len": 1088},
                                                 "fresh_salt": {"prompt_logprobs_len": 8000}})
    if numpy_metrics:
        n = len(ids)
        arrays = {"window_ids": np.array(ids)}
        for reg in ("all", "dense", "sparse"):
            arrays[f"{reg}_count"] = np.arange(n, dtype=np.int64) + 10
            for side in ("contrast", "floor"):
                for f in ("kl_sum", "top1_sum", "cov_ref_sum", "cov_cand_sum"):
                    arrays[f"{reg}_{side}_{f}"] = np.linspace(0.5, 2.5, n)
        (data / "metrics-v2").mkdir(parents=True, exist_ok=True)
        np.savez(data / "metrics-v2/cm-vs-r0.npz", **arrays)
        write(data / "metrics-v2/groups.json", {w: f"session:{i}" for i, w in enumerate(ids)})
    pair = {"name": "cm-vs-r0", "status": "ok", "ref": "R0/prompt-a", "cand": "Cm/prompt-a",
            "windows": {"compared": 5, "expected": 5},
            "regimes": {"dense": regime(0.1), "sparse": regime(0.3), "all": regime(0.2)},
            "verdict": {"dense": {"outcome": "exceeds_margin"}, "sparse": {"outcome": "unresolved", "reason": "r"},
                        "all": {"outcome": "exceeds_margin"}}}
    write(docs / "metrics-v2/cm-vs-r0.json", pair)
    write(docs / "metrics-v2/summary.json", {"bootstrap": {"B": 200, "seed": 1}, "grouping": {"groups": 4},
                                             "thresholds": {"kl_mean_excess_nats": 0.002}, "runs": {}})
    write(docs / "corpus-summary.json", {"scored_positions": 4795})
    write(docs / "boots/r0-1.json", {"label": "r0-1", "overlay_sha256": sha("f"), "image": "img"})
    write(docs / "metrics/tasks-qeval-cp-vs-r0.json", {"set": "qeval", "arms": ["R0", "Cp"], "mcnemar_exact_p": 1.0})
    write(repo / "scripts/fidelity/native_prompts.json", {"prompts": []})
    write(root / "qeval_tasks.py", "assert valid_ipv4('192.168.0.1') is True\n")
    write(root / "cluster.env", f"NODES=({SITE_HOST} nodeb-sitehost)\nMASTER_IP={SITE_IP}\n"
                                f"MGMT_IPS=(10.20.30.41 10.20.30.42)\n")
    return {"data": data, "docs": docs, "repo": repo}


def run_export(root: Path, out: Path) -> tuple[int, str, str]:
    args = ["--data", str(root / "data"), "--docs", str(root / "docs"), "--out", str(out), "--repo",
            str(root / "repo"), "--cluster-env", str(root / "cluster.env"), "--qeval-tasks",
            str(root / "qeval_tasks.py")]
    so, se = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(so), contextlib.redirect_stderr(se):
        rc = ep.main(args)
    return rc, so.getvalue(), se.getvalue()


def all_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from all_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from all_keys(v)


def files_of(out: Path) -> dict:
    return {p.relative_to(out).as_posix(): p.read_bytes() for p in sorted(out.rglob("*")) if p.is_file()}


class ExportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        build_tree(self.root, numpy_metrics=False)
        self.out = self.root / "out"

    def tearDown(self):
        self.tmp.cleanup()

    def test_clean_export_drops_private_fields(self):
        rc, stdout, stderr = run_export(self.root, self.out)
        self.assertEqual(rc, 0, stderr)
        self.assertIn("leak check: clean", stdout)
        runs = json.loads((self.out / "runs.json").read_text())
        keys = set(all_keys(runs))
        for forbidden in ("host", "cmd", "ranks", "signature_lines"):
            self.assertNotIn(forbidden, keys)
        prompt_a = [r for r in runs["runs"] if r["run"] == "prompt-a"]
        self.assertTrue(prompt_a)
        for r in prompt_a:
            self.assertEqual(r["base_url"], "http://<host>:8000")
            self.assertEqual(r["boot"]["rank_count"], 4)
            self.assertEqual(r["windows"]["by_class"]["private"]["windows"], 2)
        for line in (self.out / "memory.jsonl").read_text().splitlines():
            self.assertNotIn("host", json.loads(line))

    def test_public_rows_only(self):
        self.assertEqual(run_export(self.root, self.out)[0], 0)
        with open(self.out / "windows-public.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual([r["key"] for r in rows], ["d101", "n001"])
        with open(self.out / "window-runs-public.csv", newline="") as f:
            self.assertEqual({r["key"] for r in csv.DictReader(f)}, {"d101", "n001"})
        hashes = json.loads((self.out / "corpus-hashes.json").read_text())["windows"]
        by_key = {w["key"]: w for w in hashes}
        self.assertEqual(len(hashes), 5)
        self.assertEqual(by_key["h" + "a" * 12]["class"], "private")
        self.assertEqual(by_key["h" + "b" * 12]["class"], "synthetic")
        self.assertEqual(by_key["h" + "d" * 12]["class"], "private")
        self.assertEqual(by_key["d101"]["class"], "public")
        det = json.loads((self.out / "determinism.json").read_text())["probes"][0]
        self.assertEqual(det["sequence"], "AAB")
        self.assertEqual(det["pairs"][0]["window"], "private window A")
        res = json.loads((self.out / "results.json").read_text())
        self.assertEqual(res["outcome"], {"dense": "exceeds_margin", "sparse": "unresolved", "all": "exceeds_margin"})
        self.assertEqual(res["integrity"]["memory_aborts"][0]["mem_available_kib"], 700000)
        for name, blob in files_of(self.out).items():
            text = blob.decode("utf-8", errors="replace")
            self.assertIsNone(re.search(r"w\d{4}", text), name)
            for term in (SITE_HOST, SITE_IP, PRIVATE_PROJECT, "private-proj-beta"):
                self.assertNotIn(term, text, name)
        self.assertFalse([p for p in self.out.rglob("*") if p.suffix in (".npz", ".u32")])

    def test_injected_site_value_fails(self):
        gates = self.root / "data/boots/cm-1-gates.json"
        write(gates, {"health": 200, "note": f"served by {SITE_HOST}"})
        rc, _, stderr = run_export(self.root, self.out)
        self.assertEqual(rc, 2)
        self.assertIn("gates/cm-1-gates.json: site value", stderr)
        self.assertNotIn(SITE_HOST, stderr)
        self.assertFalse(self.out.exists())
        self.assertFalse(list(self.root.glob("*.export-tmp")))

    def test_injected_address_in_task_fails(self):
        path = self.root / "data/tasks/R0/qeval/greedy/run1/json_nested.json"
        write(path, {"response": {"content": "connect to 172.16.5.9"}, "grader": {"pass": True}})
        rc, _, stderr = run_export(self.root, self.out)
        self.assertEqual(rc, 2)
        self.assertIn("IPv4 address", stderr)

    def test_deterministic(self):
        a, b = self.root / "a", self.root / "b"
        self.assertEqual(run_export(self.root, a)[0], 0)
        self.assertEqual(run_export(self.root, b)[0], 0)
        self.assertEqual(files_of(a), files_of(b))
        self.assertEqual(run_export(self.root, a)[0], 0)
        self.assertEqual(files_of(a), files_of(b))

    def test_npz_metrics_public_only(self):
        if np is None:
            self.skipTest("numpy not available")
        build_tree(self.root, numpy_metrics=True)
        rc, _, stderr = run_export(self.root, self.out)
        self.assertEqual(rc, 0, stderr)
        with open(self.out / "window-metrics-public.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual({r["key"] for r in rows}, {"d101", "n001"})
        self.assertEqual(len(rows), 2 * 3 * 2)
        d101 = [r for r in rows if r["key"] == "d101" and r["regime"] == "dense" and r["side"] == "contrast"][0]
        self.assertEqual(d101["count"], "12")
        originals = json.loads((self.out / "results.json").read_text())["private_originals"]
        self.assertIn("metrics-v2/cm-vs-r0.npz", {o["path"] for o in originals})


if __name__ == "__main__":
    unittest.main()
