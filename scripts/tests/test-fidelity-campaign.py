#!/usr/bin/env python3
"""Offline tests for scripts/fidelity/analyze_campaign.py; no cluster or network.

The configuration, grouping, regime and verdict logic are stdlib-only. With numpy
installed, the tests also check the group bootstrap against metrics.py, the pooled
percentile, coverage, the shared-partition estimator and a small end-to-end run.
"""

from __future__ import annotations

import copy
import itertools
import math
from pathlib import Path
import random
import sys
import unittest


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/fidelity"))
import analyze_campaign as ac  # noqa: E402
import metrics as m  # noqa: E402

try:
    import numpy as np
except ImportError:  # pragma: no cover - system interpreter without numpy
    np = None

CONFIG = REPO / "scripts/fidelity/campaign.config.json"


class Config(unittest.TestCase):
    def setUp(self):
        self.cfg = ac.load_config(CONFIG)
        self.comps = {c["name"]: c for c in self.cfg["comparisons"]}

    def pairs(self):
        return {(c["ref"], c["cand"]): c for c in self.cfg["comparisons"] if c["type"] == "pair"}

    def test_shipped_config_covers_the_plan(self):
        pairs = self.pairs()
        self.assertIsNone(pairs[("R0/prompt-a", "R0/prompt-b")].get("floor"))
        for ref, cand in (("R0/prompt-a", "Cm/prompt-a"), ("R0/prompt-a", "Cpre/prompt-a"),
                          ("Cpre/prompt-a", "Cm/prompt-a"), ("R0/prompt-a", "N/prompt-a"),
                          ("Cm/prompt-a", "N/prompt-a")):
            self.assertEqual(pairs[(ref, cand)]["floor"], "floor-r0", (ref, cand))
        pre = [c["name"] for c in self.cfg["comparisons"] if c.get("role") == "pre-registered"]
        self.assertEqual(pre, ["cm-vs-r0"])
        k = [c for c in self.cfg["comparisons"] if c["type"] == "k_sensitivity"]
        self.assertEqual((k[0]["ref"], k[0]["cand"], k[0]["k_low"]), ("R0/prompt-k100", "Cm/prompt-k100", 20))
        ladder = [c for c in self.cfg["comparisons"] if c["type"] == "ladder"][0]
        self.assertEqual([ac.arm_of(r) for r in ladder["chain"]], ["R0", "L0919", "Cpre", "LE21", "LE22b", "Cm"])
        self.assertEqual(ladder["regime"], "dense")
        rep = [c for c in self.cfg["comparisons"] if c["type"] == "repeated"][0]
        self.assertEqual(set(rep["arms"]), {"R0", "Cm", "Cpre", "N"})
        self.assertGreaterEqual(rep["executions"], ac.MIN_EXECUTIONS)

    def test_execution_sets_include_cross_boot(self):
        # Finding 6: R0 uses exactly A, B and the cross-boot execution; the others A, rep2, rep3.
        rep = [c for c in self.cfg["comparisons"] if c["type"] == "repeated"][0]
        self.assertEqual(rep["executions"], 3)
        self.assertEqual(rep["arms"]["R0"], ["R0/prompt-a", "R0/prompt-b", "R0/prompt-crossboot"])
        for arm in ("Cm", "Cpre", "N"):
            self.assertEqual(rep["arms"][arm], [f"{arm}/prompt-a", f"{arm}/prompt-rep2", f"{arm}/prompt-rep3"])
        self.assertIsNone(ac.select_executions(rep["arms"]["R0"], {"R0/prompt-a", "R0/prompt-b"}, 3))

    def test_negative_controls_and_identity_checks(self):
        # Finding 3: the strict LE21 -> LE22b control and the same-recipe identity checks are configured.
        ladder = [c for c in self.cfg["comparisons"] if c["type"] == "ladder"][0]
        self.assertEqual(ladder["strict_negative_controls"], ["LE21/prompt-ladder->LE22b/prompt-ladder"])
        checks = {(c["ref"], c["cand"]) for comp in self.cfg["comparisons"] if comp["type"] == "bit_identity"
                  for c in comp["checks"]}
        for pair in (("R0/prompt-a", "R0/prompt-b"), ("Cm/prompt-a", "Cm/prompt-rep2"),
                     ("Cpre/prompt-a", "Cpre/prompt-rep2")):
            self.assertIn(pair, checks)

    def test_mde_uses_paired_null_contrasts(self):
        mde = [c for c in self.cfg["comparisons"] if c["type"] == "mde"][0]
        self.assertEqual(mde["null_candidates"], ["R0/prompt-crossboot"])
        self.assertEqual(self.cfg["bootstrap"], {"B": 2000, "seed": 20260927})
        self.assertEqual(self.cfg["thresholds"], ac.THRESHOLDS)

    def test_config_has_no_site_values(self):
        text = CONFIG.read_text()
        for needle in ("http", "/Users/", "/home/", "192.168.", "10.0."):
            self.assertNotIn(needle, text)

    def test_config_errors(self):
        bad = copy.deepcopy(self.cfg)
        bad["comparisons"].append(dict(self.cfg["comparisons"][0]))
        with self.assertRaises(ac.ConfigError):
            ac.validate_config(bad)
        bad = copy.deepcopy(self.cfg)
        bad["comparisons"][2]["floor"] = "cm-vs-r0"  # a floor must be a floor-less pair
        with self.assertRaises(ac.ConfigError):
            ac.validate_config(bad)
        bad = copy.deepcopy(self.cfg)
        bad["comparisons"][0]["type"] = "gen"
        with self.assertRaises(ac.ConfigError):
            ac.validate_config(bad)
        bad = copy.deepcopy(self.cfg)
        next(c for c in bad["comparisons"] if c["type"] == "repeated")["pairs"].append(["R0", "Z"])
        with self.assertRaises(ac.ConfigError):
            ac.validate_config(bad)
        bad = copy.deepcopy(self.cfg)
        next(c for c in bad["comparisons"] if c["type"] == "ladder")["strict_negative_controls"] = ["R0/x->Cm/y"]
        with self.assertRaises(ac.ConfigError):
            ac.validate_config(bad)


class Grouping(unittest.TestCase):
    META = {"w1": {"segment": "abc:0"}, "w2": {"segment": "abc:3"}, "w3": {"segment": "def:0"},
            "w4": {"file": "conv07.json"}, "w5": {"file": "conv07.json"}}
    NATIVE = {"windows": {"w6": {"decode_id": "d004", "prompt_tokens": 10},
                          "w7": {"decode_id": "d120", "prompt_tokens": 10},
                          "w8": {"decode_id": "n003", "prompt_tokens": 10}}}
    DECODE = {"d004": {"segment": "abc:5"}, "d120": {"origin": "qeval", "task_id": "t"}}

    def test_rules(self):
        entries = [{"id": f"w{i}", "project": "p", "category": "c"} for i in range(1, 10)]
        groups, rules = ac.build_groups(entries, self.META, self.NATIVE, self.DECODE)
        self.assertEqual(groups["w1"], groups["w2"])      # one session, two segments
        self.assertEqual(groups["w1"], groups["w6"])      # native continuation of that session's prompt
        self.assertNotEqual(groups["w1"], groups["w3"])
        self.assertEqual(groups["w4"], groups["w5"])      # one Italian conversation
        self.assertNotEqual(groups["w7"], groups["w8"])   # distinct prompts
        self.assertTrue(groups["w9"].startswith("project:"))
        self.assertEqual(rules, {"session": 3, "conversation": 2, "native_session_prompt": 1,
                                 "native_prompt": 2, "project_category_fallback": 1})

    def test_session_of_keeps_colons_in_key(self):
        self.assertEqual(ac.session_of("a:b:7"), "a:b")


class Regimes(unittest.TestCase):
    def test_rows(self):
        self.assertEqual(ac.regime_rows(1), {"all": 0, "dense": 0, "sparse": 0})
        self.assertEqual(ac.regime_rows(2049), {"all": 2048, "dense": 2048, "sparse": 0})
        self.assertEqual(ac.regime_rows(2050), {"all": 2049, "dense": 2048, "sparse": 1})


class Verdicts(unittest.TestCase):
    def ci(self, lb, ub):
        return {"estimate": (lb + ub) / 2, "lb95": lb, "ub95": ub}

    def test_criterion(self):
        self.assertEqual(ac.criterion(self.ci(-0.001, 0.0019), 0.002), ac.WITHIN)
        self.assertEqual(ac.criterion(self.ci(0.001, 0.003), 0.002), ac.UNRESOLVED)
        self.assertEqual(ac.criterion(self.ci(0.002, 0.004), 0.002), ac.EXCEEDS)
        self.assertEqual(ac.criterion(None, 0.002), ac.UNRESOLVED)
        self.assertEqual(ac.criterion({"estimate": 0.0}, 0.002), ac.UNRESOLVED)  # no bounds: not estimable

    def test_combine(self):
        self.assertEqual(ac.combine([ac.WITHIN] * 3), ac.WITHIN)
        self.assertEqual(ac.combine([ac.WITHIN, ac.UNRESOLVED]), ac.UNRESOLVED)
        self.assertEqual(ac.combine([ac.UNRESOLVED, ac.EXCEEDS]), ac.EXCEEDS)
        self.assertEqual(ac.combine([]), ac.UNRESOLVED)

    def test_ladder_attribution_requires_complete_passing_controls(self):
        full = {"status": "pass", "complete": True, "rows_compared": 10}
        self.assertEqual(ac.ladder_attribution([full]), "available")
        for gate in (dict(full, complete=False), dict(full, rows_compared=0), {"status": "missing_runs"},
                     {"status": "no_shared_windows"}):
            self.assertTrue(ac.ladder_attribution([full, gate]).startswith("pending"), gate)
        self.assertTrue(ac.ladder_attribution([]).startswith("pending"))
        self.assertEqual(ac.ladder_attribution([full, {"status": "fail"}]), "withheld: negative control failed")

    def test_select_executions(self):
        order = ["R0/prompt-a", "R0/prompt-rep2", "R0/prompt-rep3", "R0/prompt-b"]
        self.assertIsNone(ac.select_executions(order, {"R0/prompt-a", "R0/prompt-b"}, 3))
        self.assertEqual(ac.select_executions(order, {"R0/prompt-a", "R0/prompt-b", "R0/prompt-rep3"}, 3),
                         ["R0/prompt-a", "R0/prompt-rep3", "R0/prompt-b"])


def binary_rows(ps_per_row):
    """Rows over a two-token vocabulary (K = 2, actual token 0) with P(token 0) per position."""
    out = []
    for ps in ps_per_row:
        n = len(ps)
        lp = np.stack([np.log(ps), np.log1p(-ps)], 1)
        ids = np.tile(np.asarray([0, 1], np.int32), (n + 1, 1))
        ids[0] = -1
        out.append({"ids": np.zeros(n + 1, np.int32), "topk_ids": ids,
                    "topk_lp": np.vstack([[-np.inf, -np.inf], lp]).astype(np.float32),
                    "lp_actual": np.concatenate([[np.nan], lp[:, 0]]).astype(np.float32)})
    return out


def sparse_view(n):
    return {"grp": np.arange(n) % 8, "win": np.zeros(n, np.int64), "pos": np.full(n, 3000)}


class FakeData:
    """Minimal stand-in for analyze_campaign.Data: runs of in-memory windows."""

    def __init__(self, runs):
        self.runs = runs

    def run(self, key):
        windows = self.runs.get(key, {})
        k = next(iter(windows.values()))["topk_ids"].shape[1] if windows else None
        return {"windows": set(windows), "K": k}

    def load(self, key, wid):
        return self.runs[key][wid]


@unittest.skipIf(np is None, "numpy not installed")
class Numpy(unittest.TestCase):
    def test_group_ratio_matches_metrics_window_bootstrap(self):
        rng = random.Random(3)
        sums = [rng.uniform(0, 5) for _ in range(17)]
        counts = [rng.randint(1, 40) for _ in range(17)]
        grp = np.repeat(np.arange(17), counts)
        values = np.concatenate([np.full(c, s / c) for s, c in zip(sums, counts)])
        g = ac.Groups(grp, np.ones(grp.size, bool), 300, m.BOOT_SEED)
        est, reps = g.ratio(*g.sums(values))
        ref = m.ratio_bootstrap(sums, counts, B=300)
        self.assertAlmostEqual(est, ref["estimate"], places=12)
        mine = ac.ci_dict(est, reps)
        self.assertAlmostEqual(mine["ci_low"], ref["ci_low"], places=10)
        self.assertAlmostEqual(mine["ci_high"], ref["ci_high"], places=10)

    def test_pooled_matches_metrics(self):
        rng = np.random.default_rng(5)
        windows = [rng.exponential(1.0, rng.integers(1, 3000)) for _ in range(40)]
        labels = np.concatenate([np.full(len(w), i) for i, w in enumerate(windows)])
        mine = ac.Pooled(np.concatenate(windows), labels, 40)
        ref = m._Pooled(windows)
        self.assertEqual(mine.percentile(99.0), ref.percentile(99.0))
        idx = np.asarray(m.bootstrap_indices(40, 50, 7))
        for row in idx:
            mult = np.bincount(row, minlength=40)
            for q in (99.0, 50.0, 1.0):  # 50 and 1 exercise the full-scan fallback
                self.assertEqual(mine.percentile(q, mult), ref.percentile(q, mult))

    def test_coverage_and_overlap(self):
        rng = np.random.default_rng(11)
        n, V, K = 300, 60, 8
        lp_r = np.log(rng.dirichlet(np.ones(V), n))
        lp_c = np.log(rng.dirichlet(np.ones(V), n))
        ri = np.argsort(-lp_r, 1)[:, :K]
        ci = np.argsort(-lp_c, 1)[:, :K]
        rl, cl = np.take_along_axis(lp_r, ri, 1), np.take_along_axis(lp_c, ci, 1)
        eid = rng.integers(0, V, n)
        ea, eb = lp_r[np.arange(n), eid], lp_c[np.arange(n), eid]
        cov_r, cov_c, overlap = ac.np_coverage(ri, rl, ci, cl, eid, ea, eb)
        for p in range(n):
            shared = set(ri[p].tolist()) & set(ci[p].tolist())
            cells = shared | {int(eid[p])}
            self.assertEqual(overlap[p], len(shared))
            self.assertAlmostEqual(cov_r[p], sum(math.exp(lp_r[p, t]) for t in cells), places=10)
            self.assertAlmostEqual(cov_c[p], sum(math.exp(lp_c[p, t]) for t in cells), places=10)

    def test_single_group_has_no_bounds(self):
        # Finding 7: fewer than two groups keeps the estimate but no bounds, so criteria are unresolved.
        out = ac.ci_dict(0.5, [0.5] * 100, groups=1)
        self.assertEqual((out["estimate"], out["ub95"], out["lb95"], out["se"]), (0.5, None, None, None))
        self.assertEqual(ac.criterion(out, 0.002), ac.UNRESOLVED)
        g = ac.Groups(np.zeros(10, np.int64), np.ones(10, bool), 200, m.BOOT_SEED)
        self.assertIsNone(g.ci(*g.ratio(*g.sums(np.ones(10))))["ub95"])

    def test_prefill_path_uses_predictor_position(self):
        # Finding 8: row p is produced at position p - 1. In a 10,000-token window rows 1..8192 come
        # from the first 8,192-token chunk (bf16) and rows 8193..9999 from the 1,808-row chunk (marlin).
        pos = np.arange(1, 10000)
        paths = ac.row_paths(pos, np.full(pos.size, 10000))
        self.assertEqual(int(paths[pos == 8192][0]), m.PATHS.index("bf16"))
        self.assertEqual(int(paths[pos == 8193][0]), m.PATHS.index("marlin"))
        expected = [m.PATHS.index(x) for x in m.prefill_paths(10000)[:-1]]  # tag of position p - 1
        self.assertEqual(paths.tolist(), expected)

    def test_nonfinite_actual_scores_are_accounted(self):
        # Finding 9: nonfinite actual-token scores are excluded from delta NLL and counted per side.
        rng = np.random.default_rng(19)
        n, V, K = 40, 30, 5

        def row(seed):
            lp = np.log(np.random.default_rng(seed).dirichlet(np.ones(V), n))
            top = np.argsort(-lp, 1)[:, :K].astype(np.int32)
            ids = rng.integers(0, V, n) if seed == 1 else row.ids
            row.ids = ids
            return {"ids": ids.astype(np.int32), "topk_ids": top,
                    "topk_lp": np.take_along_axis(lp, top, 1).astype(np.float32),
                    "lp_actual": lp[np.arange(n), ids].astype(np.float32)}
        ref, cand = row(1), row(2)
        cand["lp_actual"][[3, 4]] = -np.inf
        cand["lp_actual"][5] = np.nan
        ref["lp_actual"][6] = -np.inf
        # +inf outside the shared cells invalidates the KL row: still counted (before masking).
        outside = next(i for i in range(8, n) if ref["ids"][i] not in ref["topk_ids"][i]
                       and ref["ids"][i] not in cand["topk_ids"][i])
        cand["lp_actual"][outside] = np.inf
        C = ac.pair_window(ref, cand)
        self.assertFalse(C["ok"][outside - 1])
        Vl = {"grp": np.arange(n - 1) % 3, "win": np.zeros(n - 1, np.int64)}
        blk = ac.block(Vl, C, None, np.ones(n - 1, bool), 100, m.BOOT_SEED, with_p99=False)
        side = blk["contrast"]
        counts = blk["actual_token_nonfinite"]["contrast"]
        self.assertEqual(counts["cand"], {"missing": 1, "zero_probability": 2, "other_nonfinite": 1,
                                          "infinite_nll": 2})
        self.assertEqual(counts["ref"]["infinite_nll"], 1)
        self.assertEqual(blk["excluded_by_joint_mask"], 1)
        self.assertTrue(side["delta_nll"]["finite_subset"])
        self.assertEqual(side["delta_nll"]["excluded_nonfinite"], 4)
        self.assertEqual(side["delta_nll"]["n"], blk["positions_joint_valid"] - 4)

    def test_delta_nll_bounds_need_two_contributing_groups(self):
        # KL is valid in three groups, but finite delta NLL occurs in group 0 only.
        n = 30
        C = {"kl": np.full(n, 0.1), "ok": np.ones(n, bool), "top1": np.ones(n, bool),
             "dnll": np.where(np.arange(n) % 3 == 0, 0.2, np.nan), "cov_ref": np.ones(n, np.float32),
             "cov_cand": np.ones(n, np.float32), "overlap": np.full(n, 5, np.int16),
             "clamped": np.zeros(n, bool), "floored": np.zeros(n, bool),
             "ra_state": np.zeros(n, np.int8), "ca_state": np.where(np.arange(n) % 3 == 0, 0, 1).astype(np.int8)}
        Vl = {"grp": np.arange(n) % 3, "win": np.zeros(n, np.int64)}
        blk = ac.block(Vl, C, None, np.ones(n, bool), 100, m.BOOT_SEED, with_p99=False)
        delta = blk["contrast"]["delta_nll"]
        self.assertEqual((blk["groups"], delta["groups"], delta["n"]), (3, 1, 10))
        self.assertAlmostEqual(delta["estimate"], 0.2)
        self.assertEqual((delta["ub95"], delta["se"]), (None, None))
        self.assertIsNotNone(blk["contrast"]["kl"]["mean_ci"]["ub95"])

    def test_empty_pair_data_is_boolean_and_blocks_are_clean(self):
        pair = ac.PairData([], [])
        for key in ("ok", "top1", "clamped", "floored"):
            self.assertEqual(pair.arr[key].dtype, bool, key)
        Vl = {"grp": np.zeros(0, np.int64), "win": np.zeros(0, np.int64)}
        blk = ac.block(Vl, pair.arr, pair.arr, np.zeros(0, bool), 50, m.BOOT_SEED)
        self.assertEqual((blk["status"], blk["positions"]), ("no_positions", 0))

    def test_rest_near_one_in_repeated_partition(self):
        near = math.log1p(-1e-17)
        L, ok, _ = ac.partition_logprobs([np.asarray([[1]])] * 2, [np.asarray([[math.log(0.5)]]),
                                                                   np.asarray([[near]])],
                                         np.asarray([1]), [np.asarray([math.log(0.5)]), np.asarray([near])])
        self.assertTrue(ok.all())
        self.assertAlmostEqual(float(L[1, 0, -1]), math.log(1e-17), places=9)
        expected = 0.5 * (math.log(0.5) - near) + 0.5 * (math.log(0.5) - math.log(1e-17))
        self.assertAlmostEqual(float(ac.kl_rows(L[0], L[1])[0]), expected, places=9)

    def test_multi_cell_rest_in_repeated_partition(self):
        # Reviewer example through the shared partition: float32 logprobs of P = (0.25, 0.25)
        # against log Q = (-1e-17, log 5e-18); expected 28.838100 (not 28.491526).
        ids = np.asarray([[1, 2]])
        ref = np.asarray([[math.log(0.25)] * 2], np.float32)
        cand = np.asarray([[-1e-17, math.log(5e-18)]], np.float32)
        L, ok, _ = ac.partition_logprobs([ids, ids], [ref, cand], np.asarray([1]),
                                         [ref[:, 0], cand[:, 0]])
        self.assertTrue(ok.all())
        self.assertAlmostEqual(float(ac.kl_rows(L[0], L[1])[0]), 28.838100, places=5)

    def test_variance_subtraction_counterexample_is_not_classified(self):
        # Finding 1 (reviewer counterexample): E = 3, reference Bernoulli(0.9) in every execution,
        # candidate executions equally likely Bernoulli(0.55) or Bernoulli(0.9999). The KL between the
        # population-average distributions is 0.053521, but both variance-subtraction estimates are
        # negative; they must not produce a verdict.
        combos = list(itertools.product([0.55, 0.9999], repeat=3))
        cand = [np.array([c[j] for c in combos]) for j in range(3)]
        A = ac.repeated_window(binary_rows([np.full(len(combos), 0.9)] * 3 + cand), 3)
        averaged = float(np.mean(A["D"] - (A["WR"] + A["WC"]) / 6))
        pairwise = float(np.mean(A["X"] - (A["WR"] + A["WC"]) / 2))
        mix = (0.55 + 0.9999) / 2
        truth = 0.9 * math.log(0.9 / mix) + 0.1 * math.log(0.1 / (1 - mix))
        self.assertAlmostEqual(truth, 0.053521, places=6)
        self.assertAlmostEqual(averaged, -0.010056, places=6)
        self.assertAlmostEqual(pairwise, -0.062245, places=6)
        blk = ac.repeated_block(sparse_view(len(combos)), A, np.ones(len(combos), bool), 3, 200,
                                m.BOOT_SEED, ac.THRESHOLDS)
        self.assertEqual(blk["outcome"], ac.UNRESOLVED)
        self.assertEqual(set(blk["criteria"].values()), {ac.UNRESOLVED})
        self.assertEqual(blk["reason"], ac.SPARSE_REASON)
        self.assertLess(blk["diagnostics"]["averaged_minus_noise"]["estimate"], 0)
        self.assertLess(blk["diagnostics"]["cross_minus_within_pairwise"]["estimate"], 0)

    def test_repeated_p99_is_descriptive(self):
        # Finding 2: identical arms whose executions choose Bernoulli(0.6) or Bernoulli(0.9) still give
        # a positive p99(cross) - p99(within); it is reported descriptively and never classified.
        combos = list(itertools.product([0.6, 0.9], repeat=6))
        rows = [np.array([c[j] for c in combos]) for j in range(6)]
        A = ac.repeated_window(binary_rows(rows), 3)
        blk = ac.repeated_block(sparse_view(len(combos)), A, np.ones(len(combos), bool), 3, 200,
                                m.BOOT_SEED, ac.THRESHOLDS)
        self.assertGreater(blk["kl_p99_cross_minus_within"]["estimate"], ac.THRESHOLDS["kl_p99_excess_nats"])
        self.assertEqual(blk["criteria"]["kl_p99_excess"], ac.UNRESOLVED)
        self.assertEqual(blk["outcome"], ac.UNRESOLVED)

    def test_bit_identity(self):
        # Finding 3: bitwise comparison of rows 1..2048; later rows are ignored; NaN compares bitwise.
        base = binary_rows([np.full(3000, 0.7)])[0]
        other = {k: v.copy() for k, v in base.items()}
        other["topk_lp"][2500, 0] += 1.0     # sparse row: ignored
        fake = FakeData({"A/r": {"w1": base, "w2": base}, "B/r": {"w1": base, "w2": other},
                         "C/r": {"w1": base}})
        self.assertEqual(ac.bit_identity(fake, "A/r", "B/r", ["w1", "w2"])["status"], "pass")
        other["lp_actual"][700] = np.nextafter(other["lp_actual"][700], np.float32(0))
        other["topk_ids"][900, 1] = 5
        res = ac.bit_identity(fake, "A/r", "B/r", ["w1", "w2"])
        self.assertEqual((res["status"], res["rows_differing"], res["first_differing_row"],
                          res["windows_identical"], res["rows_compared"]), ("fail", 2, 700, 1, 4096))
        partial = ac.bit_identity(fake, "A/r", "C/r", ["w1", "w2"])
        self.assertEqual((partial["status"], partial["complete"]), ("pass", False))
        self.assertEqual(ac.bit_identity(fake, "A/r", "Z/r", ["w1"])["status"], "missing_runs")

    def test_underflow_in_repeated_partition(self):
        # Finding 4 (repeated path): a finite -1000 logprob is not a zero probability.
        ids = np.asarray([[1, 2]])
        ref = [ids, np.asarray([[math.log(5e-5)] * 2])]
        cand = [ids, np.asarray([[-1000.0, -1000.0]])]
        L, ok, _ = ac.partition_logprobs([ref[0], cand[0]], [ref[1], cand[1]], np.asarray([1]),
                                         [np.asarray([math.log(5e-5)]), np.asarray([-1000.0])])
        kl = ac.kl_rows(L[0], L[1])
        expected = 2 * 5e-5 * (math.log(5e-5) + 1000.0) + 0.9999 * math.log(0.9999)
        self.assertTrue(ok.all())
        self.assertAlmostEqual(float(kl[0]), expected, places=12)
        mixed = ac.log_mean_exp(np.stack([L[1], L[1]]))
        self.assertTrue(np.allclose(mixed, L[1]))

    def test_selftest_pipeline(self):
        # End to end: ladder gating, MDE from paired null contrasts only, identity checks, accounting.
        self.assertEqual(ac.selftest(), 0)

    def test_shared_partition_with_two_rows_equals_coarse_kl(self):
        rng = np.random.default_rng(13)
        n, V, K = 400, 80, 10
        lps = [np.log(rng.dirichlet(np.ones(V) * 0.3, n)) for _ in range(2)]
        ids = [np.argsort(-lp, 1)[:, :K].astype(np.int32) for lp in lps]
        tops = [np.take_along_axis(lp, i, 1).astype(np.float32) for lp, i in zip(lps, ids)]
        eid = rng.integers(0, V, n)
        acts = [lp[np.arange(n), eid].astype(np.float32) for lp in lps]
        P, ok, _ = ac.partition_logprobs(ids, tops, eid, acts)
        kl = ac.kl_rows(P[0], P[1])
        ref, valid, _, _ = m.np_coarse_kl(ids[0], tops[0], ids[1], tops[1], eid, acts[0], acts[1])
        self.assertTrue(np.array_equal(ok, valid))
        self.assertTrue(np.allclose(kl[ok], ref[ok], rtol=0, atol=1e-12))

    def test_repeated_window_identical_rows_is_zero(self):
        rng = np.random.default_rng(17)
        n, V, K = 50, 40, 6
        lp = np.log(rng.dirichlet(np.ones(V), n))
        ids = rng.integers(0, V, n)
        top = np.argsort(-lp, 1)[:, :K].astype(np.int32)
        row = {"ids": ids.astype(np.int32), "topk_ids": top,
               "topk_lp": np.take_along_axis(lp, top, 1).astype(np.float32),
               "lp_actual": lp[np.arange(n), ids].astype(np.float32)}
        out = ac.repeated_window([row] * 6, 3)
        self.assertTrue(out["ok"].all())
        for key in ("D", "WR", "WC", "X"):
            self.assertEqual(float(np.abs(out[key]).max()), 0.0, key)
        for key in ("aRR", "aCC", "aRC"):
            self.assertEqual(float(out[key].min()), 1.0, key)


if __name__ == "__main__":
    unittest.main(verbosity=1)
