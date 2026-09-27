#!/usr/bin/env python3
"""Offline tests for scripts/fidelity/metrics.py; stdlib only, no cluster or network.

Covers coarse top-K KL (exact cases, the lower-bound property against the full KL,
the actual-token cell, NaN/-inf/clamp/floor handling), top-1 agreement, delta NLL,
generation-prefix comparison, prefill-path tagging and bootstrap determinism. When
numpy is importable, the fast path is also checked against the stdlib path.
"""

from __future__ import annotations

import decimal
import math
from pathlib import Path
import random
import struct
import sys
import unittest


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/fidelity"))
import metrics as m  # noqa: E402

LN = math.log


def log_softmax(logits):
    top = max(logits)
    total = math.fsum(math.exp(x - top) for x in logits)
    return [x - top - math.log(total) for x in logits]


def full_kl(p_lp, q_lp):
    return math.fsum(math.exp(a) * (a - b) for a, b in zip(p_lp, q_lp))


def struct_f32(x):
    """Round a Python float to float32 and back (the stored logprob precision)."""
    return struct.unpack("f", struct.pack("f", x))[0]


def decimal_coarse_kl(ref_lp, cand_lp, digits=60):
    """High-precision coarse KL over shared cells {all given} + rest, from the exact float inputs."""
    with decimal.localcontext() as ctx:
        ctx.prec = digits
        p = [decimal.Decimal(x).exp() for x in ref_lp]
        q = [decimal.Decimal(x).exp() for x in cand_lp]
        total = sum((pi * (decimal.Decimal(a) - decimal.Decimal(b)) for pi, a, b in zip(p, ref_lp, cand_lp)),
                    decimal.Decimal(0))
        rest_p, rest_q = 1 - sum(p), 1 - sum(q)
        if rest_p > 0:
            total += rest_p * (rest_p / rest_q).ln()
        return float(total)


def topk(lps, k):
    order = sorted(range(len(lps)), key=lambda i: -lps[i])[:k]
    return order, [lps[i] for i in order]


def random_pair(rng, vocab=1000, scale=3.0, noise=0.5):
    logits = [rng.gauss(0, scale) for _ in range(vocab)]
    other = [x + rng.gauss(0, noise) for x in logits]
    return log_softmax(logits), log_softmax(other)


class CoarseKL(unittest.TestCase):
    def test_identical_is_zero(self):
        ids, lps = [5, 7, 9], [LN(0.5), LN(0.2), LN(0.1)]
        result = m.coarse_kl(ids, lps, ids, lps)
        self.assertEqual(result, m.KL(0.0, True, False, False))

    def test_hand_computed(self):
        result = m.coarse_kl([1, 2], [LN(0.5), LN(0.3)], [2, 1], [LN(0.4), LN(0.4)])
        expected = 0.5 * LN(0.5 / 0.4) + 0.3 * LN(0.3 / 0.4) + 0.2 * LN(0.2 / 0.2)
        self.assertAlmostEqual(result.kl, expected, places=12)
        # Partial intersection {1}: rest is 0.5 vs 0.6.
        result = m.coarse_kl([1, 2], [LN(0.5), LN(0.3)], [1, 3], [LN(0.4), LN(0.3)])
        self.assertAlmostEqual(result.kl, 0.5 * LN(0.5 / 0.4) + 0.5 * LN(0.5 / 0.6), places=12)
        # Empty intersection: KL of a single rest cell is zero.
        self.assertEqual(m.coarse_kl([1], [LN(0.5)], [2], [LN(0.5)]).kl, 0.0)

    def test_lower_bound_on_full_kl(self):
        for seed in range(60):
            rng = random.Random(seed)
            p, q = random_pair(rng, noise=rng.choice([0.05, 0.5, 2.0]))
            exact = full_kl(p, q)
            for k in (1, 5, 20):
                ri, rl = topk(p, k)
                ci, cl = topk(q, k)
                coarse = m.coarse_kl(ri, rl, ci, cl)
                self.assertTrue(coarse.valid)
                self.assertGreaterEqual(coarse.kl, -1e-12)
                self.assertLessEqual(coarse.kl, exact + 1e-12, (seed, k))
                actual = rng.randrange(len(p))
                refined = m.coarse_kl(ri, rl, ci, cl, extra=(actual, p[actual], q[actual]))
                self.assertLessEqual(refined.kl, exact + 1e-12, (seed, k, actual))
                self.assertGreaterEqual(refined.kl, coarse.kl - 1e-12)

    def test_actual_token_outside_topk(self):
        ri, rl = [1, 2], [LN(0.5), LN(0.3)]
        ci, cl = [1, 2], [LN(0.5), LN(0.3)]
        base = m.coarse_kl(ri, rl, ci, cl)
        # Token 9 has 0.1 vs 0.01 inside an otherwise identical 0.2 rest.
        extra = m.coarse_kl(ri, rl, ci, cl, extra=(9, LN(0.1), LN(0.01)))
        self.assertEqual(base.kl, 0.0)
        self.assertAlmostEqual(extra.kl, 0.1 * LN(0.1 / 0.01) + 0.1 * LN(0.1 / 0.19), places=12)
        # Inside the intersection it adds nothing; NaN on either side is ignored.
        self.assertEqual(m.coarse_kl(ri, rl, ci, cl, extra=(1, LN(0.5), LN(0.5))).kl, 0.0)
        self.assertEqual(m.coarse_kl(ri, rl, ci, cl, extra=(9, math.nan, LN(0.01))).kl, 0.0)
        # Actual token only in ref top-K: its cand logprob comes from `extra`.
        known = m.coarse_kl([1, 9], [LN(0.5), LN(0.3)], [1, 2], [LN(0.5), LN(0.3)],
                            extra=(9, LN(0.3), LN(0.05)))
        self.assertAlmostEqual(known.kl, 0.3 * LN(0.3 / 0.05) + 0.2 * LN(0.2 / 0.45), places=12)

    def test_nan_and_neg_inf(self):
        self.assertFalse(m.coarse_kl([1, 2], [LN(0.5), math.nan], [1, 2], [LN(0.5), LN(0.2)]).valid)
        self.assertTrue(math.isnan(m.coarse_kl([-1], [-math.inf], [1], [LN(0.5)]).kl))
        # -inf on the reference side is zero probability: 0 * log 0 = 0.
        zero = m.coarse_kl([1, 2], [LN(0.5), -math.inf], [1, 2], [LN(0.5), LN(0.1)])
        self.assertTrue(zero.valid)
        self.assertAlmostEqual(zero.kl, 0.5 * LN(0.5 / 0.4), places=12)
        # -inf on the candidate where the reference has mass: floored, finite, counted.
        floor = m.coarse_kl([1, 2], [LN(0.5), LN(0.1)], [1, 2], [LN(0.5), -math.inf])
        self.assertTrue(floor.floored and math.isfinite(floor.kl))
        # Padding (-1 / -inf) is ignored.
        pad = m.coarse_kl([1, -1], [LN(0.5), -math.inf], [1, -1], [LN(0.5), -math.inf])
        self.assertEqual(pad, m.KL(0.0, True, False, False))

    def test_underflowing_candidate_logprob_is_not_zero(self):
        # Candidate logprobs of -1000 underflow exp() in float64 but are finite: the
        # known-cell terms must use the logprobs, not a Q_FLOOR substitute.
        ref_lp, cand_lp = [LN(0.00005), LN(0.00005)], [-1000.0, -1000.0]
        expected = 2 * 0.00005 * (LN(0.00005) + 1000.0) + 0.9999 * LN(0.9999 / 1.0)
        got = m.coarse_kl([1, 2], ref_lp, [1, 2], cand_lp)
        self.assertTrue(got.valid)
        self.assertFalse(got.floored)  # underflow is not a genuine zero
        self.assertAlmostEqual(got.kl, expected, places=12)
        self.assertGreater(got.kl, 0.09)
        # The same through the actual-token cell.
        extra = m.coarse_kl([1], [LN(0.5)], [1], [LN(0.5)], extra=(7, LN(0.00005), -1000.0))
        self.assertAlmostEqual(extra.kl, 0.00005 * (LN(0.00005) + 1000.0)
                               + 0.49995 * LN(0.49995 / 0.5), places=12)
        # A genuine zero (-inf) still takes the floor and is counted.
        genuine = m.coarse_kl([1, 2], ref_lp, [1, 2], [-math.inf, -math.inf])
        self.assertTrue(genuine.floored)
        self.assertAlmostEqual(genuine.kl, 2 * 0.00005 * (LN(0.00005) - LN(m.Q_FLOOR))
                               + 0.9999 * LN(0.9999), places=12)
        if m.np is not None:
            np = m.np
            kl, valid, _, floored = m.np_coarse_kl(
                np.asarray([[1, 2], [1, 2]]), np.asarray([ref_lp, ref_lp]),
                np.asarray([[1, 2], [1, 2]]), np.asarray([cand_lp, [-math.inf, -math.inf]]),
                np.asarray([1, 1]), np.asarray([ref_lp[0]] * 2), np.asarray([-1000.0, -math.inf]))
            self.assertTrue(valid.all())
            self.assertAlmostEqual(float(kl[0]), expected, places=12)
            self.assertAlmostEqual(float(kl[1]), genuine.kl, places=12)
            self.assertEqual(floored.tolist(), [False, True])

    def test_rest_mass_near_one_is_not_rounded_to_zero(self):
        # Known mass 1 - 1e-17 rounds to 1.0 in probability space; the rest must stay 1e-17
        # (from the logprobs), not become a genuine zero that takes the 1e-12 floor.
        near = math.log1p(-1e-17)
        expected = 0.5 * (LN(0.5) - near) + 0.5 * (LN(0.5) - LN(1e-17))
        got = m.coarse_kl([1], [LN(0.5)], [1], [near])
        self.assertTrue(got.valid)
        self.assertFalse(got.floored)
        self.assertAlmostEqual(got.kl, expected, places=9)
        self.assertGreater(got.kl, 18.0)   # the old rounding gave about 13.1
        # Compensated complement: tiny rest kept, exact exhaustion is a genuine zero.
        self.assertAlmostEqual(m._log_rest([-1e-17], m.CLAMP_TOL)[0], LN(1e-17), places=12)
        self.assertAlmostEqual(m._log_rest([-50.0], m.CLAMP_TOL)[0], -math.exp(-50.0), places=30)
        self.assertEqual(m._log_rest([LN(0.5), LN(0.5)], m.CLAMP_TOL), (-math.inf, False))
        if m.np is not None:
            np = m.np
            kl, valid, clamped, floored = m.np_coarse_kl(
                np.asarray([[1]]), np.asarray([[LN(0.5)]]), np.asarray([[1]]), np.asarray([[near]]))
            self.assertTrue(valid[0] and not floored[0] and not clamped[0])
            self.assertAlmostEqual(float(kl[0]), expected, places=9)
            rest, clamp, over = m.np_log_rest(np.asarray([[-1e-17], [-50.0], [0.0], [4e-7], [1e-3], [-np.inf]]))
            self.assertAlmostEqual(float(rest[0]), LN(1e-17), places=9)
            self.assertAlmostEqual(float(rest[1]), -math.exp(-50.0), places=30)
            self.assertEqual(rest[2:5].tolist(), [-math.inf] * 3)
            self.assertEqual(float(rest[5]), 0.0)
            self.assertEqual(clamp.tolist(), [False, False, False, True, False, False])
            self.assertEqual(over.tolist(), [False, False, False, False, True, False])

    def test_rest_multi_cell_cancellation(self):
        # Reviewer example: float32 logprobs of P = (0.25, 0.25) against log Q = (-1e-17, log 5e-18).
        # A log-sum-exp complement loses the 5e-18 cell (28.491526); the compensated rest keeps it.
        f32 = struct_f32
        ref_lp = [f32(LN(0.25)), f32(LN(0.25))]
        cand_lp = [f32(-1e-17), f32(LN(5e-18))]
        expected = decimal_coarse_kl(ref_lp, cand_lp)
        self.assertAlmostEqual(expected, 28.838100, places=5)
        got = m.coarse_kl([1, 2], ref_lp, [1, 2], cand_lp)
        self.assertTrue(got.valid and not got.floored)
        self.assertAlmostEqual(got.kl, expected, delta=1e-9 * expected)
        if m.np is not None:
            np = m.np
            kl, valid, _, _ = m.np_coarse_kl(np.asarray([[1, 2]]), np.asarray([ref_lp], np.float32),
                                             np.asarray([[1, 2]]), np.asarray([cand_lp], np.float32))
            self.assertTrue(valid[0])
            self.assertAlmostEqual(float(kl[0]), expected, delta=1e-9 * expected)

    def test_rest_near_one_random_against_high_precision(self):
        rng = random.Random(20260927)
        rows = []
        for _ in range(200):
            top = -10.0 ** rng.uniform(-17, -6)          # largest cell, probability close to 1
            gap = -math.expm1(top)
            smalls = [LN(gap * rng.uniform(0.01, 0.3) / 3) for _ in range(3)]  # rest >= 10% of the gap
            cand = [top] + smalls
            ref = [LN(rng.uniform(0.05, 0.2)) for _ in range(4)]
            rows.append((ref, cand))
            expected = decimal_coarse_kl(ref, cand)
            got = m.coarse_kl([1, 2, 3, 4], ref, [1, 2, 3, 4], cand)
            self.assertTrue(got.valid)
            self.assertAlmostEqual(got.kl, expected, delta=1e-9 * abs(expected))
        if m.np is not None:
            np = m.np
            ids = np.tile(np.asarray([1, 2, 3, 4]), (len(rows), 1))
            kl, valid, _, _ = m.np_coarse_kl(ids, np.asarray([r for r, _ in rows]), ids,
                                             np.asarray([c for _, c in rows]))
            self.assertTrue(valid.all())
            for (ref, cand), value in zip(rows, kl):
                expected = decimal_coarse_kl(ref, cand)
                self.assertAlmostEqual(float(value), expected, delta=1e-9 * abs(expected))

    def test_rest_clamp_and_floor(self):
        over = 1.0 + 5e-7
        clamped = m.coarse_kl([1, 2], [LN(0.5), LN(over - 0.5)], [1, 2], [LN(0.5), LN(0.4)])
        self.assertTrue(clamped.valid and clamped.clamped)
        broken = m.coarse_kl([1, 2], [LN(0.5), LN(0.6)], [1, 2], [LN(0.5), LN(0.4)])
        self.assertFalse(broken.valid)
        # Reference rest 0.1 against candidate rest 0: floor 1e-12.
        floored = m.coarse_kl([1, 2], [LN(0.5), LN(0.4)], [1, 2], [LN(0.5), LN(0.5)])
        self.assertTrue(floored.floored)
        self.assertAlmostEqual(floored.kl, 0.4 * LN(0.4 / 0.5) + 0.1 * LN(0.1 / m.Q_FLOOR), places=9)


class Scalars(unittest.TestCase):
    def test_top1_and_nll(self):
        self.assertTrue(m.top1_agree([3, 4], [-0.1, -2.0], [4, 3], [-3.0, -0.2]))
        self.assertFalse(m.top1_agree([3, 4], [-0.1, -2.0], [4, 3], [-0.1, -2.0]))
        self.assertIsNone(m.top1_agree([-1], [-math.inf], [3], [-0.1]))
        self.assertIsNone(m.top1_agree([3], [math.nan], [3], [-0.1]))
        self.assertAlmostEqual(m.delta_nll(-1.0, -1.5), 0.5)
        self.assertIsNone(m.delta_nll(math.nan, -1.0))
        self.assertIsNone(m.delta_nll(-1.0, -math.inf))

    def test_generation_prefix(self):
        self.assertEqual(m.generation_prefix([1, 2, 3], [1, 2, 3]), (3, None))
        self.assertEqual(m.generation_prefix([1, 2, 3, 4], [1, 2, 9, 4]), (3, 2))
        self.assertEqual(m.generation_prefix([1, 2], [1, 2, 3]), (2, None))
        self.assertEqual(m.generation_prefix([], [1]), (0, None))
        row = ([1, 2], [LN(0.6), LN(0.3)])
        other = ([2, 1], [LN(0.6), LN(0.3)])
        ref = {"gen_ids": [1, 1, 1], "topk_ids": [row[0]] * 3, "topk_lp": [row[1]] * 3}
        cand = {"gen_ids": [1, 2, 1], "topk_ids": [row[0], other[0], row[0]],
                "topk_lp": [row[1], other[1], row[1]]}
        result = m.compare_generations(ref, cand)
        self.assertEqual((result["length"], result["first_divergence"]), (2, 1))
        self.assertEqual(result["kl"][0].kl, 0.0)
        self.assertGreater(result["kl"][1].kl, 0.0)
        self.assertEqual(result["top1"], [True, False])


class Tagging(unittest.TestCase):
    def test_prefill_paths(self):
        self.assertEqual(m.prefill_chunk_sizes(10000), [8192, 1808])
        self.assertEqual(m.prefill_chunk_sizes(16384), [8192, 8192])
        self.assertEqual(m.prefill_path(0, 10000), "bf16")
        self.assertEqual(m.prefill_path(8191, 10000), "bf16")
        self.assertEqual(m.prefill_path(8192, 10000), "marlin")
        self.assertEqual(m.prefill_path(100, 1000), "marlin")
        self.assertEqual(m.prefill_path(100, 2048), "bf16")
        self.assertEqual(m.prefill_path(8192, 8192 + 2047), "marlin")
        self.assertEqual(m.prefill_path(8192, 8192 + 2048), "bf16")
        paths = m.prefill_paths(10000)
        self.assertEqual(len(paths), 10000)
        self.assertEqual(paths, [m.prefill_path(p, 10000) for p in range(10000)])

    def test_buckets(self):
        cases = {0: "0-2K", 2047: "0-2K", 2048: "2-8K", 8191: "2-8K", 8192: "8-32K",
                 32767: "8-32K", 32768: "32-64K", 65535: "32-64K", 65536: "64K+", 131072: "64K+"}
        for position, label in cases.items():
            self.assertEqual(m.position_bucket(position), label)


class Aggregation(unittest.TestCase):
    def setUp(self):
        rng = random.Random(7)
        self.windows = [[rng.expovariate(50.0) for _ in range(rng.randint(5, 40))] for _ in range(25)]
        self.floor = [[v * 0.5 for v in w] for w in self.windows]

    def test_summary(self):
        s = m.summary([1.0, 2.0, 3.0, 4.0, math.nan, None])
        self.assertEqual((s["n"], s["mean"], s["median"], s["max"]), (4, 2.5, 2.5, 4.0))
        self.assertAlmostEqual(s["p90"], 3.7)
        self.assertEqual(m.summary([])["n"], 0)

    def test_bootstrap_determinism(self):
        self.assertEqual(m.bootstrap_indices(10, 50), m.bootstrap_indices(10, 50))
        self.assertNotEqual(m.bootstrap_indices(10, 50), m.bootstrap_indices(10, 50, seed=1))
        sums = [math.fsum(w) for w in self.windows]
        counts = [len(w) for w in self.windows]
        first = m.ratio_bootstrap(sums, counts, B=300)
        self.assertEqual(first, m.ratio_bootstrap(sums, counts, B=300))
        self.assertAlmostEqual(first["estimate"], math.fsum(sums) / sum(counts))
        self.assertLess(first["ci_low"], first["estimate"])
        self.assertGreater(first["ci_high"], first["estimate"])
        self.assertAlmostEqual(m.mde(first["se"]), 2.8 * first["se"])

    def test_paired_excess(self):
        sa = [math.fsum(w) for w in self.windows]
        sb = [math.fsum(w) for w in self.floor]
        counts = [len(w) for w in self.windows]
        diff = m.paired_ratio_diff(sa, counts, sb, counts, B=300)
        self.assertAlmostEqual(diff["estimate"], math.fsum(sa) / sum(counts) / 2, places=12)
        self.assertGreater(diff["ci_low"], 0.0)
        same = m.paired_ratio_diff(sa, counts, sa, counts, B=100)
        self.assertEqual((same["estimate"], same["ci_low"], same["ci_high"]), (0.0, 0.0, 0.0))
        p99 = m.paired_percentile_diff(self.windows, self.floor, 99.0, B=200)
        self.assertGreater(p99["estimate"], 0.0)
        self.assertEqual(p99, m.paired_percentile_diff(self.windows, self.floor, 99.0, B=200))


@unittest.skipIf(m.np is None, "numpy not installed; fast path not exercised")
class NumpyAgreement(unittest.TestCase):
    def test_kl_and_top1(self):
        np = m.np
        rng = random.Random(11)
        rows = []
        for i in range(300):
            p, q = random_pair(rng, vocab=200, noise=rng.choice([0.01, 0.3, 1.5]))
            ri, rl = topk(p, 8)
            ci, cl = topk(q, 8)
            actual = rng.randrange(200)
            ep, eq = p[actual], q[actual]
            if i % 17 == 0:
                rl[3] = math.nan
            if i % 19 == 0:
                ri, rl = ri[:5] + [-1] * 3, rl[:5] + [-math.inf] * 3
            if i % 23 == 0:
                cl[1] = -math.inf
            if i % 29 == 0:
                eq = math.nan
            rows.append((ri, rl, ci, cl, actual, ep, eq))
        cols = list(zip(*rows))
        arr = [np.asarray(c) for c in cols]
        kl, valid, clamped, floored = m.np_coarse_kl(
            arr[0].astype(np.int32), arr[1].astype(np.float64), arr[2].astype(np.int32),
            arr[3].astype(np.float64), arr[4], arr[5], arr[6], chunk=37)
        top1 = m.np_top1(arr[0], arr[1], arr[2], arr[3])
        for i, (ri, rl, ci, cl, actual, ep, eq) in enumerate(rows):
            ref = m.coarse_kl(ri, rl, ci, cl, extra=(actual, ep, eq))
            self.assertEqual(ref.valid, bool(valid[i]), i)
            self.assertEqual(ref.clamped, bool(clamped[i]), i)
            self.assertEqual(ref.floored, bool(floored[i]), i)
            if ref.valid:
                self.assertAlmostEqual(ref.kl, float(kl[i]), delta=1e-12 + 1e-9 * ref.kl)
            agree = m.top1_agree(ri, rl, ci, cl)
            self.assertEqual(-1 if agree is None else int(agree), int(top1[i]), i)

    def test_aggregates(self):
        rng = random.Random(3)
        windows = [[rng.expovariate(20.0) for _ in range(rng.randint(3, 30))] for _ in range(20)]
        floor = [[v * rng.uniform(0.3, 0.9) for v in w] for w in windows]
        sums = [math.fsum(w) for w in windows]
        fsums = [math.fsum(w) for w in floor]
        counts = [len(w) for w in windows]
        values = [v for w in windows for v in w]
        for key, value in m.summary(values).items():
            self.assertAlmostEqual(value, m.np_summary(m.np.asarray(values))[key], places=12)
        pairs = [
            (m.ratio_bootstrap(sums, counts, B=200), m.np_ratio_bootstrap(sums, counts, B=200)),
            (m.paired_ratio_diff(sums, counts, fsums, counts, B=200, scale=100.0),
             m.np_paired_ratio_diff(sums, counts, fsums, counts, B=200, scale=100.0)),
            (m.percentile_bootstrap(windows, 99.0, B=200), m.np_percentile_bootstrap(windows, 99.0, B=200)),
            (m.paired_percentile_diff(windows, floor, 99.9, B=200),
             m.np_paired_percentile_diff(windows, floor, 99.9, B=200)),
        ]
        for slow, fast in pairs:
            for key in slow:
                self.assertAlmostEqual(slow[key], fast[key], places=10, msg=key)


if __name__ == "__main__":
    unittest.main(verbosity=1)
