#!/usr/bin/env python3
"""Offline smoke test for scripts/fidelity/build_report.py; stdlib only.

Builds REPORT.md and report.html from a tiny synthetic docs/fidelity tree (one pre-registered
comparison, a floor, a missing arm, one figure and one missing figure file) and checks the
section order, pending placeholders, the verdict file, HTML self-containment and the leak check.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "fidelity"))

import build_report as br  # noqa: E402


def ci(estimate):
    return {"estimate": estimate, "ci_low": estimate * 0.9, "ci_high": estimate * 1.1,
            "lb95": estimate * 0.92, "ub95": estimate * 1.08, "se": estimate * 0.05, "B": 2000}


def contrast(kl, top1):
    return {"kl": {"mean": kl, "mean_ci": ci(kl), "p99": kl * 10, "median": kl / 3},
            "top1_agreement": ci(top1), "delta_nll": ci(0.001),
            "covered_mass_ref": {"mean": 0.9}, "covered_mass_cand": {"mean": 0.9},
            "topk_overlap": {"median": 17.0}}


def regime(kl, floor_kl, top1=0.9):
    return {"positions": 1000, "positions_joint_valid": 1000, "groups": 5,
            "contrast": contrast(kl, top1), "floor": contrast(floor_kl, 1.0),
            "excess": {"kl_mean_excess": ci(kl - floor_kl), "kl_p99_excess": ci(0.5), "top1_drop_pp": ci(2.0)},
            "by_category": {}}


SVG = ('<?xml version="1.0" encoding="utf-8" standalone="no"?>\n'
       '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">\n'
       '<svg xmlns="http://www.w3.org/2000/svg" width="720pt" height="360pt" viewBox="0 0 720 360">'
       '<rect width="720" height="360" fill="#ffffff"/></svg>\n')


def write(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")


def synthetic_docs(root: Path) -> Path:
    docs = root / "docs"
    verdict = {"dense": {"outcome": "exceeds_margin", "criteria": {}},
               "sparse": {"outcome": "unresolved", "reason": "sparse-regime verdict needs >= 3 executions per arm"},
               "all": {"outcome": "exceeds_margin"}}
    write(docs / "metrics-v2" / "cm-vs-r0.json", {
        "name": "cm-vs-r0", "type": "pair", "status": "ok", "role": "pre-registered", "ref": "R0/prompt-a",
        "cand": "Cm/prompt-a", "floor": "floor-r0", "windows": {"expected": 3, "compared": 3}, "groups": 2,
        "regimes": {"dense": regime(0.18, 0.0), "sparse": regime(0.29, 0.27), "all": regime(0.26, 0.19)},
        "verdict": verdict})
    write(docs / "metrics-v2" / "floor-r0.json", {
        "name": "floor-r0", "type": "pair", "status": "ok", "ref": "R0/prompt-a", "cand": "R0/prompt-b",
        "floor": None, "windows": {"expected": 3, "compared": 3}, "groups": 2,
        "regimes": {r: {"positions_joint_valid": 1000, "contrast": contrast(0.1, 0.9)} for r in br.REGIMES},
        "by_position_bucket": {"2-8K": {"positions_joint_valid": 10, "contrast": contrast(0.2, 0.8)},
                               "0-2K": {"positions_joint_valid": 10, "contrast": contrast(0.0, 1.0)}}})
    write(docs / "metrics-v2" / "n-vs-r0.json", {"name": "n-vs-r0", "type": "pair", "status": "missing_runs",
                                                 "missing_runs": ["N/prompt-a"]})
    write(docs / "metrics-v2" / "summary.json", {
        "schema": "fidelity-campaign-summary/1", "generated_utc": "2026-09-27T00:00:00+00:00",
        "harness_git": {"commit": "0123456789abcdef", "dirty": False},
        "thresholds": {"kl_mean_excess_nats": 0.002, "kl_p99_excess_nats": 0.02, "top1_drop_pp": 0.5},
        "method": ["Coarse KL is a lower bound."],
        "runs": {"R0/prompt-a": {"K": 20, "windows_ok": 3}, "Cm/prompt-a": {"K": 20, "windows_ok": 3},
                 "N/prompt-a": {"K": None, "windows_ok": 0}},
        "comparisons": {"cm-vs-r0": {"status": "ok"}, "floor-r0": {"status": "ok"},
                        "n-vs-r0": {"status": "missing_runs"}}})
    write(docs / "corpus-summary.json", {
        "windows": 3, "total_tokens": 9000, "scored_positions": 8997, "windows_at_least_3072": 2, "frozen": False,
        "model_repo": "org/model", "model_rev": "abcdef0123",
        "by_category": {"agentic_code": {"windows": 2, "tokens": 6000, "share_of_tokens": 0.667}},
        "by_source": {"synthetic_it": {"windows": 1, "tokens": 3000, "share_of_tokens": 0.333}},
        "length_histogram": [{"min": 2048, "max_exclusive": 3072, "count": 1}]})
    write(docs / "boots" / "cm-1.json", {"label": "cm-1", "overlay": "scripts/node/experiments/fidelity/cm.env",
                                         "overlay_sha256": "ab" * 32, "model_repo": "org/model", "model_rev": "c" * 40,
                                         "image": "registry.example/img@sha256:" + "d" * 64,
                                         "kv_cache_dtype": "fp8_e4m3", "spec_tokens": "7", "k": 20,
                                         "ranks": [{"ready_lines": ["E29_END_DRAIN_READY trace=0"]}]})
    write(docs / "plots" / "01-kl-cdf.svg", SVG)
    write(docs / "plots" / "01-quality-vs-fp8.svg", SVG)
    write(docs / "plots" / "plots.json", {"figures": [
        {"name": "01-quality-vs-fp8", "title": "Quality vs FP8", "caption": "Perplexity.", "status": "ok",
         "group": "quality", "notes": [], "png": "01-quality-vs-fp8.png", "svg": "01-quality-vs-fp8.svg"},
        {"name": "01-kl-cdf", "group": "results", "title": "Per-position KL", "caption": "CDF.", "status": "partial",
         "notes": ["Not yet measured (skipped): N."], "png": "01-kl-cdf.png", "svg": "01-kl-cdf.svg"},
        {"name": "06-decode-path", "title": "Decode path", "caption": "Decode.", "status": "pending",
         "notes": ["Generations not collected yet."], "png": "06-decode-path.png", "svg": "06-decode-path.svg"}]})
    return docs


class ReportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="fidelity-report-test-"))
        self.docs = synthetic_docs(self.tmp)
        self.args = ["--docs", str(self.docs), "--cluster-env", str(self.tmp / "none.env"),
                     "--manifest", str(self.tmp / "none.json")]

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def build(self, extra=()):
        rc = br.main(self.args + list(extra))
        md = (self.docs / "REPORT.md").read_text(encoding="utf-8")
        page = (self.docs / "report.html").read_text(encoding="utf-8")
        return rc, md, page

    def test_sections_in_owner_order_and_pending(self):
        rc, md, _ = self.build()
        self.assertEqual(rc, 0)
        heads = [line for line in md.splitlines() if line.startswith("## ")]
        self.assertEqual(heads, ["## 1. Verdict", "## 2. Method, arms and overlays", "## 3. Corpus",
                                 "## 4. Results", "## 5. Limitations", "## 6. Reproduction"])
        self.assertIn("**Verdict: pending.**", md)
        self.assertIn("NVFP4 answer (separate)", md)
        self.assertIn("pending (missing runs: N/prompt-a)", md)
        self.assertIn("dense exceeds · sparse unresolved · overall exceeds", md)
        self.assertIn("Status: **pending**", md)
        self.assertIn("lower bound", md)
        self.assertIn("bash scripts/fidelity/make_all.sh", md)
        self.assertIn("[Amendments](#amendments)", md)
        self.assertIn("### Amendments", md)
        self.assertIn("6. **R0 uses FP8 KV.**", md)
        self.assertNotIn("PLAN.md", md)
        # bucket rows follow position order, not file order
        self.assertLess(md.index("| 0-2K |"), md.index("| 2-8K |"))

    def test_quality_first(self):
        rc, md, page = self.build()
        self.assertEqual(rc, 0)
        verdict, method = md.index("## 1. Verdict"), md.index("## 2. Method")
        quality = md.index("### Quality at a glance")
        self.assertTrue(verdict < quality < md.index("![Quality vs FP8]") < method)
        self.assertGreater(md.index("![Per-position KL]"), method)
        self.assertIn("**E29 vs FP8:** perplexity change of E29 (Cm) against R0", md)
        self.assertIn("**E29 vs NVFP4:** pending", md)
        self.assertIn("Perplexity change** translates ΔNLL", md)
        self.assertLess(page.index("Quality at a glance"), page.index("2. Method"))
        # exp(0.001) - 1 = +0.10%; CI 0.0009..0.0011 -> +0.09..+0.11
        self.assertIn("| E29 (Cm) vs R0 | +0.10% [+0.09, +0.11] |", md)
        self.assertEqual(br.ppl_reading({"estimate": 0.0, "ci_low": -0.1, "ci_high": 0.1}), "no detectable change")
        self.assertEqual(br.ppl_reading({"estimate": 0.2, "ci_low": 0.1, "ci_high": 0.3}), "worse than R0")

    def test_verdict_file_used(self):
        write(self.docs / "verdict.md", "Cm stays **within** margin.\n\n- line two")
        rc, md, page = self.build()
        self.assertEqual(rc, 0)
        self.assertNotIn("Verdict: pending", md)
        self.assertIn("Cm stays **within** margin.", md)
        self.assertIn("<strong>within</strong>", page)
        self.assertIn("<li>line two</li>", page)

    def test_eli5_file_above_verdict(self):
        write(self.docs / "eli5.md", "E29 is **as good as** FP8.\n\n- NVFP4 is worse.")
        rc, md, page = self.build()
        self.assertEqual(rc, 0)
        heads = [line for line in md.splitlines() if line.startswith("## ")]
        self.assertEqual(heads[:2], ["## In plain words", "## 1. Verdict"])
        self.assertLess(md.index("E29 is **as good as** FP8."), md.index("## 1. Verdict"))
        self.assertIn('<section class="eli5"><p>E29 is <strong>as good as</strong> FP8.</p>', page)
        self.assertLess(page.index('class="eli5"'), page.index('class="verdict"'))

    def test_summary_figures_follow_plain_words(self):
        write(self.docs / "eli5.md", "Plain summary.")
        plots = json.loads((self.docs / "plots" / "plots.json").read_text(encoding="utf-8"))
        plots["figures"].append({"name": "16-quality-by-context-length", "title": "Quality by conversation length",
                                 "group": "summary", "status": "ok", "caption": "c", "png": "16.png",
                                 "svg": "16.svg"})
        write(self.docs / "plots" / "plots.json", json.dumps(plots))
        rc, md, _ = self.build()
        self.assertEqual(rc, 0)
        self.assertEqual(md.count("![Quality by conversation length]"), 1)
        self.assertLess(md.index("Plain summary."), md.index("![Quality by conversation length]"))
        self.assertLess(md.index("![Quality by conversation length]"), md.index("## 1. Verdict"))

    def test_corruption_probe_and_descoped_figures(self):
        write(self.docs / "metrics" / "corruption-N.json", json.dumps(
            {"label": "N", "italian_prompts": 40, "italian_invalid_utf8": 0, "italian_with_fffd": 3,
             "italian_fffd_total": 8, "italian_repetition_flags": 0, "italian_truncated": 0,
             "tool_prompts": 10, "tool_calls_made": 10, "tool_parse_failures": 0}))
        plots = json.loads((self.docs / "plots" / "plots.json").read_text(encoding="utf-8"))
        plots["figures"].append({"name": "13-voxel-showcase", "title": "Voxel showcase", "group": "results",
                                 "status": "pending", "caption": "c", "png": "13.png", "svg": "13.svg"})
        write(self.docs / "plots" / "plots.json", json.dumps(plots))
        rc, md, _ = self.build()
        self.assertEqual(rc, 0)
        self.assertIn("### Corruption probe", md)
        self.assertIn("| N | 40 | 0 | 3 (8) | 0 | 0 | 10 / 10 | 0 |", md)
        self.assertIn("Not produced, inputs descoped: Voxel showcase: voxel showcase (amendment 16).", md)
        self.assertNotIn("![Voxel showcase]", md)
        self.assertIn("| descriptive | corruption probe |", md)

    def test_verdict_nested_lists(self):
        html = br.md_to_html("Intro.\n\n1. **First** point.\n   - sub a;\n   - sub b\n     continued.\n\n"
                             "   Closing paragraph.\n2. Second.\n\n- plain\n\nAfter.")
        self.assertIn("<p>Intro.</p>", html)
        self.assertIn("<ol><li><strong>First</strong> point.<ul><li>sub a;</li><li>sub b continued.</li></ul>"
                      "<p>Closing paragraph.</p></li><li>Second.</li></ol>", html)
        self.assertIn("<ul><li>plain</li></ul>", html)
        self.assertTrue(html.endswith("<p>After.</p>"))

    def test_html_self_contained(self):
        _, _, page = self.build()
        self.assertTrue(page.startswith("<!DOCTYPE html>"))
        self.assertIn("prefers-color-scheme:dark", page)
        self.assertIn('name="viewport"', page)
        self.assertIn(".table-wrap{overflow-x:auto", page)
        self.assertIn("How to read this", page)
        self.assertIn('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 360" role="img"', page)
        self.assertNotIn("<?xml", page)
        self.assertNotIn("<!DOCTYPE svg", page)
        self.assertIsNone(re.search(r"<(script|link|img|iframe)\b", page))
        self.assertIsNone(re.search(r'(src|href)="https?://', page))
        self.assertIn("Figure file missing", page)
        self.assertLess(page.index("Verdict"), page.index("<figure>"))

    def test_leak_check_flags_window_ids_and_site_values(self):
        write(self.docs / "verdict.md", "Window w0042 looked odd.")
        rc, _, _ = self.build()
        self.assertEqual(rc, 2)
        write(self.docs / "verdict.md", "Served from rank0-host.")
        write(self.tmp / "site.env", 'NODE_HOSTNAMES="rank0-host rank1-host"\n')
        rc = br.main(["--docs", str(self.docs), "--cluster-env", str(self.tmp / "site.env"),
                      "--manifest", str(self.tmp / "none.json")])
        self.assertEqual(rc, 2)
        self.assertEqual(br.main(["--docs", str(self.docs), "--cluster-env", str(self.tmp / "site.env"),
                                  "--manifest", str(self.tmp / "none.json"), "--no-leak-check"]), 0)

    def test_leak_patterns(self):
        files = []
        for i, text in enumerate(["/Users/someone/x", "data/fidelity/corpus/tokens", "host 192.168.1.2",
                                  "10.10.0.1", "w0001", "ok 3.11.2 and sha256:0d4029b3"]):
            path = self.tmp / f"f{i}.txt"
            path.write_text(text)
            files.append(path)
        found = {name for name, _ in br.leak_check(files, [])}
        self.assertEqual(found, {"f0.txt", "f1.txt", "f2.txt", "f3.txt", "f4.txt"})

    def test_empty_docs_still_builds(self):
        empty = self.tmp / "empty"
        empty.mkdir()
        rc = br.main(["--docs", str(empty), "--cluster-env", str(self.tmp / "none"), "--manifest",
                      str(self.tmp / "none")])
        self.assertEqual(rc, 0)
        md = (empty / "REPORT.md").read_text(encoding="utf-8")
        self.assertIn("Corpus summary pending.", md)
        self.assertIn("Cm vs R0, dense regime: pending", md)


if __name__ == "__main__":
    unittest.main(verbosity=1)
