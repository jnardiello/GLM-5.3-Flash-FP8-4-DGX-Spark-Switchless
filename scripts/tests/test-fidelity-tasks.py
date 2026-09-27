#!/usr/bin/env python3
"""Offline tests for scripts/fidelity/tasks/{run_tasks,stats}.py; stdlib only,
no cluster or network beyond a local fake HTTP server on 127.0.0.1.

Covers exact McNemar against hand-computed values, bootstrap determinism,
z.ai cost-cap logic, result-file resume logic, and an end-to-end run of
run_tasks.py against a fake /v1/chat/completions + /v1/models server.
"""
from __future__ import annotations

import http.server
import json
import math
import shutil
import sys
import tempfile
import threading
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "fidelity" / "tasks"))
sys.path.insert(0, str(REPO / "third_party" / "knapcio-bench" / "bench"))

import run_tasks as rt  # noqa: E402
import stats as st_  # noqa: E402
import qeval_tasks as qt  # noqa: E402


# --------------------------------------------------------------- McNemar / CI

class McNemarTest(unittest.TestCase):
    def test_hand_computed(self):
        # n=10 discordant, k=min(1,9)=1: tail = (C(10,0)+C(10,1))/2^10 = 11/1024
        # p = 2 * 11/1024 = 22/1024 = 0.021484375
        self.assertAlmostEqual(st_.mcnemar_exact_p(1, 9), 22 / 1024, places=12)
        self.assertAlmostEqual(st_.mcnemar_exact_p(9, 1), 22 / 1024, places=12)

    def test_symmetric_no_discordance(self):
        self.assertEqual(st_.mcnemar_exact_p(0, 0), 1.0)

    def test_perfectly_balanced(self):
        # b=c=5: k=5, n=10, tail = sum(C(10,i) for i in 0..5)/1024
        tail = sum(math.comb(10, i) for i in range(6)) / 1024
        self.assertAlmostEqual(st_.mcnemar_exact_p(5, 5), min(1.0, 2 * tail), places=12)

    def test_matches_vendored_formula(self):
        # cross-check against the (unmodified) vendored qeval.py implementation
        sys.path.insert(0, str(REPO / "third_party" / "knapcio-bench" / "bench"))
        import qeval as qe  # noqa: E402
        for b, c in [(0, 0), (1, 9), (3, 3), (7, 2), (0, 5)]:
            self.assertAlmostEqual(st_.mcnemar_exact_p(b, c), qe.mcnemar_p(b, c), places=12)


class WilsonTest(unittest.TestCase):
    def test_known_bounds(self):
        # Wilson 95% CI for 5/10: well-known approx bounds ~ (0.237, 0.763)
        p, lo, hi = st_.wilson_ci(5, 10)
        self.assertAlmostEqual(p, 0.5, places=9)
        self.assertAlmostEqual(lo, 0.2366, places=3)
        self.assertAlmostEqual(hi, 0.7634, places=3)

    def test_zero_n(self):
        self.assertEqual(st_.wilson_ci(0, 0), (0.0, 0.0, 0.0))


class BootstrapTest(unittest.TestCase):
    def test_determinism(self):
        a = {f"i{i}": (i % 3) / 2 for i in range(20)}
        b = {f"i{i}": ((i + 1) % 3) / 2 for i in range(20)}
        ids = sorted(a)
        r1 = st_.bootstrap_diff_ci(a, b, ids, B=200, seed=20260927)
        r2 = st_.bootstrap_diff_ci(a, b, ids, B=200, seed=20260927)
        self.assertEqual(r1, r2)

    def test_different_seed_can_differ(self):
        a = {f"i{i}": (i % 5) / 4 for i in range(30)}
        b = {f"i{i}": ((i + 2) % 5) / 4 for i in range(30)}
        ids = sorted(a)
        r1 = st_.bootstrap_diff_ci(a, b, ids, B=200, seed=1)
        r2 = st_.bootstrap_diff_ci(a, b, ids, B=200, seed=2)
        self.assertEqual(r1["diff"], r2["diff"])  # point estimate is seed-independent
        # CIs need not be identical across seeds (resampling noise); just sane bounds.
        self.assertLessEqual(r1["ci95"][0], r1["ci95"][1])
        self.assertLessEqual(r2["ci95"][0], r2["ci95"][1])

    def test_empty_intersection(self):
        r = st_.bootstrap_diff_ci({"a": 1.0}, {"b": 1.0}, ["a", "b"])
        self.assertEqual(r["n_items"], 0)


# ------------------------------------------------------------------ cost cap

class SpendLedgerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_worst_case_cost_formula(self):
        cost = rt.SpendLedger.worst_case_cost(prompt_tokens=1000, max_tokens=2000)
        expected = 1000 * rt.ZAI_PRICE["input"] + 2000 * rt.ZAI_PRICE["output"]
        self.assertAlmostEqual(cost, expected, places=12)

    def test_would_exceed_blocks_before_request(self):
        ledger = rt.SpendLedger(self.tmp / "ledger.json", max_usd=0.001)
        # 1000 prompt tokens + 16384 max_tokens output at $0.50/M output alone
        # is ~$0.0082, comfortably over a $0.001 cap.
        self.assertTrue(ledger.would_exceed(1000, 16384))

    def test_would_not_exceed_under_generous_cap(self):
        ledger = rt.SpendLedger(self.tmp / "ledger.json", max_usd=1000.0)
        self.assertFalse(ledger.would_exceed(1000, 16384))

    def test_actual_cost_accounts_for_cached_input(self):
        usage = {"prompt_tokens": 1000, "completion_tokens": 500,
                  "prompt_tokens_details": {"cached_tokens": 400}}
        cost = rt.SpendLedger.actual_cost(usage)
        expected = (600 * rt.ZAI_PRICE["input"] + 400 * rt.ZAI_PRICE["cached_input"]
                    + 500 * rt.ZAI_PRICE["output"])
        self.assertAlmostEqual(cost, expected, places=12)

    def test_record_updates_cumulative_spend_and_persists(self):
        path = self.tmp / "ledger.json"
        ledger = rt.SpendLedger(path, max_usd=10.0)
        ledger.record("item1", 1, {"prompt_tokens": 100, "completion_tokens": 100})
        self.assertGreater(ledger.spent, 0)
        self.assertTrue(path.exists())
        reloaded = rt.SpendLedger(path, max_usd=10.0)
        self.assertAlmostEqual(reloaded.spent, ledger.spent, places=9)
        self.assertEqual(len(reloaded.entries), 1)

    def test_would_exceed_uses_running_total(self):
        ledger = rt.SpendLedger(self.tmp / "ledger.json", max_usd=0.01)
        ledger.record("item1", 1, {"prompt_tokens": 1000, "completion_tokens": 16384})
        # first record already spent ~$0.00834 of the $0.01 cap; even a tiny
        # next worst-case request (~$0.0082) should trip the remaining budget.
        self.assertTrue(ledger.would_exceed(10, 16384))


# --------------------------------------------------------------- resume logic

class ResumeLogicTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_missing_file_is_invalid(self):
        self.assertFalse(rt.result_is_valid(self.tmp / "missing.json"))

    def test_corrupt_json_is_invalid(self):
        p = self.tmp / "bad.json"
        p.write_text("{not json")
        self.assertFalse(rt.result_is_valid(p))

    def test_error_record_is_invalid(self):
        p = self.tmp / "err.json"
        p.write_text(json.dumps({"ok": False, "error": "boom"}))
        self.assertFalse(rt.result_is_valid(p))

    def test_missing_response_is_invalid(self):
        p = self.tmp / "norresp.json"
        p.write_text(json.dumps({"ok": True}))
        self.assertFalse(rt.result_is_valid(p))

    def test_valid_record_is_valid(self):
        p = self.tmp / "ok.json"
        p.write_text(json.dumps({"ok": True, "response": {"content": "hi"}}))
        self.assertTrue(rt.result_is_valid(p))


# --------------------------------------------------------------------- fake server

class FakeHandler(http.server.BaseHTTPRequestHandler):
    hit_count = 0
    hit_lock = threading.Lock()

    def log_message(self, *a):  # silence
        pass

    def do_GET(self):
        if self.path == "/v1/models":
            self._json(200, {"data": [{"id": "fake-model"}]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._json(404, {"error": "not found"})
            return
        with FakeHandler.hit_lock:
            FakeHandler.hit_count += 1
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        prompt = body["messages"][0]["content"]
        content = FakeHandler.canned_answer(prompt)
        resp = {
            "choices": [{"message": {"content": content, "reasoning_content": "because."},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 42, "completion_tokens": 7, "total_tokens": 49},
        }
        self._json(200, resp)

    @staticmethod
    def canned_answer(prompt):
        # A deterministic canned answer that exactly satisfies fmt_json_only's
        # checker when that prompt is seen; a harmless generic reply otherwise.
        if "Output ONLY the JSON array" in prompt:
            return "[1, 2, 3]"
        return "a plain canned answer for testing"

    def _json(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class EndToEndTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        FakeHandler.hit_count = 0
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeHandler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.base_url = f"http://127.0.0.1:{self.port}"

    def test_hardset_run_writes_expected_files_and_resumes(self):
        FakeHandler.hit_count = 0
        rc = rt.main([
            "--set", "hardset", "--mode", "greedy",
            "--base-url", self.base_url, "--model", "fake-model",
            "--concurrency", "4", "--out-root", str(self.tmp),
        ])
        self.assertEqual(rc, 0)
        n_prompts = len(rt.load_items("hardset"))
        out_dir = self.tmp / "local" / "hardset" / "greedy" / "run1"
        files = sorted(out_dir.glob("*.json"))
        self.assertEqual(len(files), n_prompts)
        first_hits = FakeHandler.hit_count
        self.assertEqual(first_hits, n_prompts)

        rec = json.loads(files[0].read_text())
        self.assertTrue(rec["ok"])
        self.assertEqual(rec["response"]["content"], "a plain canned answer for testing")
        self.assertEqual(rec["reasoning_effort_used"], "high")
        self.assertEqual(rec["max_tokens_used"], rt.DEFAULT_MAX_TOKENS)
        self.assertIn("task_default_effort", rec)
        self.assertIn("task_default_max_tokens", rec)
        self.assertNotIn("api_key", json.dumps(rec).lower())

        # second invocation must resume: no new HTTP hits.
        rc2 = rt.main([
            "--set", "hardset", "--mode", "greedy",
            "--base-url", self.base_url, "--model", "fake-model",
            "--concurrency", "4", "--out-root", str(self.tmp),
        ])
        self.assertEqual(rc2, 0)
        self.assertEqual(FakeHandler.hit_count, first_hits, "resume must not re-request completed items")

    def test_qeval_single_item_grader_runs_via_vendored_checker(self):
        # Isolate to one cheap, deterministic format task so the fake server's
        # canned answer is graded pass=True by the vendored (unmodified) checker.
        target = next(t for t in qt.TASKS if t["id"] == "fmt_json_only")
        original_tasks = qt.TASKS
        qt.TASKS = [target]
        try:
            rc = rt.main([
                "--set", "qeval", "--mode", "greedy",
                "--base-url", self.base_url, "--model", "fake-model",
                "--concurrency", "1", "--out-root", str(self.tmp),
            ])
        finally:
            qt.TASKS = original_tasks
        self.assertEqual(rc, 0)
        out_path = self.tmp / "local" / "qeval" / "greedy" / "run1" / "fmt_json_only.json"
        self.assertTrue(out_path.exists())
        rec = json.loads(out_path.read_text())
        self.assertTrue(rec["ok"])
        self.assertIn("grader", rec)
        self.assertTrue(rec["grader"]["pass"], rec["grader"])

    def test_zai_arm_refuses_without_max_usd(self):
        with self.assertRaises(SystemExit):
            rt.main(["--set", "hardset", "--mode", "greedy", "--zai"])


if __name__ == "__main__":
    unittest.main()
