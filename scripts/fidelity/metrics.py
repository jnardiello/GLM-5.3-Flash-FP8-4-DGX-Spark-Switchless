"""Fidelity metrics: coarse top-K KL, top-1 agreement, delta NLL, tagging and bootstrap.

The core functions are stdlib-only and operate on per-position lists. The `np_*`
functions are an optional numpy fast path used by analyze.py; tests check that both
agree. Coarse KL is computed over the partition {each token in topK_ref and topK_cand}
plus {rest}; by the data-processing inequality it is a lower bound on the true KL.
"""

from __future__ import annotations

import functools
import math
import random
from typing import NamedTuple

try:  # optional fast path
    import numpy as np
except ImportError:  # pragma: no cover - exercised by the stdlib tests
    np = None


BATCHED_TOKENS = 8192
MARLIN_BELOW_ROWS = 2048
PATHS = ("marlin", "bf16")
BUCKETS = ((0, 2048, "0-2K"), (2048, 8192, "2-8K"), (8192, 32768, "8-32K"),
           (32768, 65536, "32-64K"), (65536, math.inf, "64K+"))
BUCKET_LABELS = tuple(label for _, _, label in BUCKETS)
BOOT_B = 2000
BOOT_SEED = 20260927
CLAMP_TOL = 1e-6
Q_FLOOR = 1e-12
LOG_Q_FLOOR = math.log(Q_FLOOR)
MDE_FACTOR = 2.8  # 80% power, two-sided alpha 0.05: (1.96 + 0.84) * SE
QUANTILES = (("median", 50.0), ("p90", 90.0), ("p99", 99.0), ("p99_9", 99.9))


# ---------------------------------------------------------------- per position


class KL(NamedTuple):
    kl: float
    valid: bool
    clamped: bool
    floored: bool


INVALID = KL(math.nan, False, False, False)


def _row(ids, lps):
    """id -> logprob for real entries; None if a real entry is NaN or the row is empty."""
    row = {}
    for tid, lp in zip(ids, lps):
        if tid < 0:
            continue
        if math.isnan(lp):
            return None
        row[tid] = lp
    return row or None


def _term_lp(lp_p: float, lp_q: float) -> tuple[float, bool]:
    """Known-cell term exp(lp_p) * (lp_p - lp_q) from logprobs.

    Working in log space keeps a finite but very negative logprob finite: exp() underflow
    must not turn it into zero probability. Only -inf is a genuine zero; a genuine zero
    candidate against positive reference mass uses the Q_FLOOR and is counted.
    """
    if lp_p == -math.inf:
        return 0.0, False
    p = math.exp(lp_p)
    if lp_q == -math.inf:
        return p * (lp_p - LOG_Q_FLOOR), p > 0.0
    return p * (lp_p - lp_q), False


def _rest_from_cells(lps, tol: float) -> tuple[float | None, bool]:
    """Compensated complement: rest = -expm1(lp_max) - fsum(exp(other cells)), in float64.

    The largest cell enters through expm1, so a cell with probability close to 1 keeps its
    tiny complement, and fsum keeps the small cells that a log-sum-exp would absorb.
    Returns (rest, clamped); a negative rest within `tol` is clamped to 0, beyond it None.
    """
    finite = [float(lp) for lp in lps if lp != -math.inf]
    if not finite:
        return 1.0, False
    top = max(finite)
    finite.remove(top)
    rest = -math.expm1(top) - math.fsum(math.exp(lp) for lp in finite)
    if rest >= 0.0:
        return rest, False
    if rest >= -tol:
        return 0.0, True
    return None, False


def _log_rest(lps, tol: float) -> tuple[float | None, bool]:
    """(log of the rest mass, clamped) from the known cells' logprobs; -inf is a genuine zero
    (the cells exhaust the distribution, exactly or within `tol`), None is inconsistent.
    A known mass of at most 0.5 uses log1p(-mass); above it the compensated complement."""
    mass = math.fsum(math.exp(lp) for lp in lps if lp != -math.inf)
    if mass <= 0.5:
        return math.log1p(-mass), False  # rest >= 0.5: log1p keeps the small known mass
    rest, clamped = _rest_from_cells(lps, tol)
    if rest is None:
        return None, False
    return (math.log(rest) if rest > 0.0 else -math.inf), clamped


def coarse_kl(ref_ids, ref_lp, cand_ids, cand_lp, extra=None, tol: float = CLAMP_TOL) -> KL:
    """KL(P_ref || P_cand) on {topK_ref intersect topK_cand} + {rest}.

    `extra` = (token_id, ref_logprob, cand_logprob) adds the actual token as its own
    cell when it is outside the intersection and known (not NaN) on both sides.
    """
    ref = _row(ref_ids, ref_lp)
    cand = _row(cand_ids, cand_lp)
    if ref is None or cand is None:
        return INVALID
    cells = [(lp, cand[tid]) for tid, lp in ref.items() if tid in cand]  # logprob pairs
    if extra is not None:
        tid, rlp, clp = extra
        if tid >= 0 and not (tid in ref and tid in cand) and not math.isnan(rlp) and not math.isnan(clp):
            cells.append((rlp, clp))
    rest_p, clamp_p = _log_rest([a for a, _ in cells], tol)
    rest_q, clamp_q = _log_rest([b for _, b in cells], tol)
    if rest_p is None or rest_q is None:
        return INVALID
    total, floored = 0.0, False
    for a, b in cells:
        value, hit = _term_lp(a, b)
        total += value
        floored |= hit
    value, hit = _term_lp(rest_p, rest_q)
    return KL(total + value, True, clamp_p or clamp_q, floored or hit)


def top1_id(ids, lps):
    best, best_lp = None, None
    for tid, lp in zip(ids, lps):
        if tid < 0:
            continue
        if math.isnan(lp):
            return None
        if best is None or lp > best_lp:
            best, best_lp = tid, lp
    return best


def top1_agree(ref_ids, ref_lp, cand_ids, cand_lp):
    """True/False, or None when either side has no usable row."""
    a, b = top1_id(ref_ids, ref_lp), top1_id(cand_ids, cand_lp)
    if a is None or b is None:
        return None
    return a == b


def delta_nll(ref_lp_actual: float, cand_lp_actual: float):
    """NLL_cand - NLL_ref for the actual token; None when either is not finite."""
    if not (math.isfinite(ref_lp_actual) and math.isfinite(cand_lp_actual)):
        return None
    return ref_lp_actual - cand_lp_actual


def generation_prefix(ref_ids, cand_ids) -> tuple[int, int | None]:
    """(positions to compare, first divergent position or None).

    Positions 0..j inclusive share an identical prefix, so their distributions are
    comparable; j is the first divergence, or the end of the shorter sequence.
    """
    common = min(len(ref_ids), len(cand_ids))
    for position in range(common):
        if ref_ids[position] != cand_ids[position]:
            return position + 1, position
    return common, None


def compare_generations(ref: dict, cand: dict) -> dict:
    """Per-position KL and top-1 along the identical greedy prefix of two generations."""
    length, divergence = generation_prefix(ref["gen_ids"], cand["gen_ids"])
    kls, top1 = [], []
    for p in range(length):
        kls.append(coarse_kl(ref["topk_ids"][p], ref["topk_lp"][p],
                             cand["topk_ids"][p], cand["topk_lp"][p]))
        top1.append(top1_agree(ref["topk_ids"][p], ref["topk_lp"][p],
                               cand["topk_ids"][p], cand["topk_lp"][p]))
    return {"length": length, "first_divergence": divergence, "kl": kls, "top1": top1,
            "common_length": min(len(ref["gen_ids"]), len(cand["gen_ids"]))}


# ---------------------------------------------------------------- tagging


def prefill_chunk_sizes(n: int, batched: int = BATCHED_TOKENS) -> list[int]:
    return [min(batched, n - start) for start in range(0, n, batched)]


def prefill_path(position: int, n: int, batched: int = BATCHED_TOKENS,
                 marlin_below: int = MARLIN_BELOW_ROWS) -> str:
    """Hybrid-KDA path of the prefill chunk that contains `position`."""
    start = (position // batched) * batched
    return "marlin" if min(batched, n - start) < marlin_below else "bf16"


def prefill_paths(n: int, batched: int = BATCHED_TOKENS, marlin_below: int = MARLIN_BELOW_ROWS) -> list[str]:
    out = []
    for size in prefill_chunk_sizes(n, batched):
        out.extend(["marlin" if size < marlin_below else "bf16"] * size)
    return out


def position_bucket(position: int) -> str:
    for low, high, label in BUCKETS:
        if low <= position < high:
            return label
    raise ValueError(position)


# ---------------------------------------------------------------- aggregation (stdlib)


def percentile_sorted(values, q: float) -> float:
    """Linear-interpolation percentile of an ascending sequence (numpy 'linear')."""
    n = len(values)
    if n == 0:
        return math.nan
    h = (n - 1) * q / 100.0
    lo = int(math.floor(h))
    hi = min(lo + 1, n - 1)
    return values[lo] + (h - lo) * (values[hi] - values[lo])


def summary(values) -> dict:
    data = sorted(v for v in values if v is not None and not math.isnan(v))
    if not data:
        return {"n": 0, "mean": None, "median": None, "p90": None, "p99": None, "p99_9": None, "max": None}
    out = {"n": len(data), "mean": math.fsum(data) / len(data)}
    for name, q in QUANTILES:
        out[name] = percentile_sorted(data, q)
    out["max"] = data[-1]
    return out


def bootstrap_indices(n_windows: int, B: int = BOOT_B, seed: int = BOOT_SEED) -> list[list[int]]:
    """Window resampling indices, shared by the stdlib and numpy paths."""
    rng = random.Random(seed)
    population = range(n_windows)
    return [rng.choices(population, k=n_windows) for _ in range(B)]


def _ci(replicates: list[float], estimate: float) -> dict:
    reps = sorted(r for r in replicates if not math.isnan(r))
    if len(reps) < 2:
        return {"estimate": estimate, "ci_low": None, "ci_high": None, "se": None, "B": len(reps)}
    mean = math.fsum(reps) / len(reps)
    se = math.sqrt(math.fsum((r - mean) ** 2 for r in reps) / (len(reps) - 1))
    return {"estimate": estimate, "ci_low": percentile_sorted(reps, 2.5),
            "ci_high": percentile_sorted(reps, 97.5), "se": se, "B": len(reps)}


def _ratio(sums, counts, idx=None) -> float:
    if idx is None:
        num, den = math.fsum(sums), sum(counts)
    else:
        num, den = math.fsum(sums[i] for i in idx), sum(counts[i] for i in idx)
    return num / den if den else math.nan


def ratio_bootstrap(sums, counts, B: int = BOOT_B, seed: int = BOOT_SEED) -> dict:
    """Token-weighted mean sum/count with a window-level percentile bootstrap CI."""
    indices = bootstrap_indices(len(sums), B, seed)
    return _ci([_ratio(sums, counts, idx) for idx in indices], _ratio(sums, counts))


def paired_ratio_diff(sums_a, counts_a, sums_b, counts_b, B: int = BOOT_B,
                      seed: int = BOOT_SEED, scale: float = 1.0) -> dict:
    """(ratio_a - ratio_b) * scale over the same windows, resampled in pairs."""
    indices = bootstrap_indices(len(sums_a), B, seed)
    estimate = (_ratio(sums_a, counts_a) - _ratio(sums_b, counts_b)) * scale
    reps = [(_ratio(sums_a, counts_a, idx) - _ratio(sums_b, counts_b, idx)) * scale for idx in indices]
    return _ci(reps, estimate)


def _pooled_percentile(windows, q: float, idx=None) -> float:
    chosen = windows if idx is None else [windows[i] for i in idx]
    return percentile_sorted(sorted(v for w in chosen for v in w), q)


def percentile_bootstrap(windows, q: float, B: int = BOOT_B, seed: int = BOOT_SEED) -> dict:
    """Pooled token percentile with a window-level bootstrap CI."""
    indices = bootstrap_indices(len(windows), B, seed)
    return _ci([_pooled_percentile(windows, q, idx) for idx in indices], _pooled_percentile(windows, q))


def paired_percentile_diff(windows_a, windows_b, q: float, B: int = BOOT_B, seed: int = BOOT_SEED) -> dict:
    indices = bootstrap_indices(len(windows_a), B, seed)
    estimate = _pooled_percentile(windows_a, q) - _pooled_percentile(windows_b, q)
    reps = [_pooled_percentile(windows_a, q, idx) - _pooled_percentile(windows_b, q, idx) for idx in indices]
    return _ci(reps, estimate)


def mde(se) -> float | None:
    return None if se is None else MDE_FACTOR * se


# ---------------------------------------------------------------- numpy fast path


def _require_numpy():
    if np is None:
        raise RuntimeError("numpy is required for the fast path")


def np_coarse_kl(ref_ids, ref_lp, cand_ids, cand_lp, extra_ids=None, ref_extra=None,
                 cand_extra=None, tol: float = CLAMP_TOL, chunk: int | None = None):
    """Vectorised coarse_kl over rows. Returns (kl, valid, clamped, floored) arrays."""
    _require_numpy()
    n = ref_ids.shape[0]
    kl = np.full(n, np.nan)
    valid = np.zeros(n, bool)
    clamped = np.zeros(n, bool)
    floored = np.zeros(n, bool)
    if chunk is None:  # bound the (rows, K_ref, K_cand) match matrix to ~4M cells
        chunk = max(64, (1 << 22) // max(1, ref_ids.shape[1] * cand_ids.shape[1]))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for s in range(0, n, chunk):
            e = min(n, s + chunk)
            ri, ci = ref_ids[s:e], cand_ids[s:e]
            rl, cl = ref_lp[s:e].astype(np.float64), cand_lp[s:e].astype(np.float64)
            rreal, creal = ri >= 0, ci >= 0
            bad = ((np.isnan(rl) & rreal).any(1) | (np.isnan(cl) & creal).any(1)
                   | ~rreal.any(1) | ~creal.any(1))
            match = (ri[:, :, None] == ci[:, None, :]) & rreal[:, :, None]
            inter = match.any(2)
            col = np.argmax(match, axis=2)
            lp = np.where(inter, rl, -np.inf)  # known cells stay in log space (no underflow)
            lq = np.where(inter, np.take_along_axis(cl, col, axis=1), -np.inf)
            pp, qq = [lp], [lq]
            if extra_ids is not None:
                eid = extra_ids[s:e]
                elr = ref_extra[s:e].astype(np.float64)
                elc = cand_extra[s:e].astype(np.float64)
                both = ((ri == eid[:, None]).any(1)) & ((ci == eid[:, None]).any(1))
                use = (eid >= 0) & ~both & ~np.isnan(elr) & ~np.isnan(elc)
                lpe = np.where(use, elr, -np.inf)
                lqe = np.where(use, elc, -np.inf)
                pp.append(lpe[:, None])
                qq.append(lqe[:, None])
            cells_p, cells_q = np.concatenate(pp, 1), np.concatenate(qq, 1)
            rest_p, clamp_p, over_p = np_log_rest(cells_p, tol)
            rest_q, clamp_q, over_q = np_log_rest(cells_q, tol)
            over = over_p | over_q
            clamp = (clamp_p | clamp_q) & ~over
            terms, hit = np_kl_terms(np.concatenate([cells_p, rest_p[:, None]], 1),
                                     np.concatenate([cells_q, rest_q[:, None]], 1))
            ok = ~bad & ~over
            kl[s:e] = np.where(ok, terms.sum(1), np.nan)
            valid[s:e] = ok
            clamped[s:e] = clamp & ok
            floored[s:e] = hit.any(1) & ok
    return kl, valid, clamped, floored


def np_log_rest(L, tol: float = CLAMP_TOL):
    """Vectorised _log_rest over rows of known-cell logprobs L (n, cells).

    When the known mass is at most 0.5 the log rest is log1p(-mass); otherwise
    rest = -expm1(max) - (Neumaier-compensated sum of the other cells' probabilities, added
    in ascending order), in float64. Returns (log rest, clamped, over): log rest is -inf when
    the cells exhaust the distribution (exactly or within `tol`, clamped); `over` marks rows
    whose cells exceed 1 by more than `tol` (invalid).
    """
    _require_numpy()
    L = np.asarray(L, np.float64)
    n = L.shape[0]
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        arg = np.argmax(np.where(np.isnan(L), -np.inf, L), axis=1) if L.shape[1] else np.zeros(n, np.int64)
        top = L[np.arange(n), arg] if L.shape[1] else np.full(n, -np.inf)
        others = np.exp(L)
        if L.shape[1]:
            others[np.arange(n), arg] = 0.0
        others = np.sort(others, axis=1)
        total = np.zeros(n)
        comp = np.zeros(n)
        for j in range(others.shape[1]):  # Neumaier summation, ascending
            x = others[:, j]
            t = total + x
            comp += np.where(np.abs(total) >= np.abs(x), (total - t) + x, (x - t) + total)
            total = t
        rest = -np.expm1(top) - (total + comp)
        mass = np.exp(top) + (total + comp)
        small = mass <= 0.5  # rest >= 0.5: log1p keeps the small known mass
        over = ~small & (rest < -tol)
        clamped = ~small & (rest < 0.0) & ~over
        log_rest = np.where(small, np.log1p(-np.where(small, mass, 0.0)),
                            np.where(rest > 0.0, np.log(np.where(rest > 0.0, rest, 1.0)), -np.inf))
    return log_rest, clamped, over


def np_kl_terms(LP, LQ):
    """Elementwise exp(LP) * (LP - LQ) from logprobs; -inf is a genuine zero.

    A genuine zero candidate (-inf) against positive reference mass uses LOG_Q_FLOOR
    and is flagged. Returns (terms, floored) arrays of LP's shape.
    """
    _require_numpy()
    with np.errstate(invalid="ignore", over="ignore"):
        pos = LP > -np.inf
        weight = np.exp(np.where(pos, LP, 0.0))
        zero_q = LQ == -np.inf
        terms = np.where(pos, weight * (LP - np.where(zero_q, LOG_Q_FLOOR, LQ)), 0.0)
        return terms, pos & zero_q & (weight > 0)


def np_top1(ref_ids, ref_lp, cand_ids, cand_lp):
    """Returns int8 array: 1 agree, 0 disagree, -1 invalid."""
    _require_numpy()

    def best(ids, lp):
        real = ids >= 0
        bad = (np.isnan(lp) & real).any(1) | ~real.any(1)
        arg = np.argmax(np.where(real, np.nan_to_num(lp, nan=-np.inf), -np.inf), axis=1)
        return ids[np.arange(ids.shape[0]), arg], bad

    a, bad_a = best(ref_ids, ref_lp)
    b, bad_b = best(cand_ids, cand_lp)
    out = (a == b).astype(np.int8)
    out[bad_a | bad_b] = -1
    return out


def np_percentile_sorted(sorted_values, q: float) -> float:
    n = sorted_values.shape[0]
    if n == 0:
        return math.nan
    h = (n - 1) * q / 100.0
    lo = int(math.floor(h))
    hi = min(lo + 1, n - 1)
    return float(sorted_values[lo] + (h - lo) * (sorted_values[hi] - sorted_values[lo]))


def np_summary(values) -> dict:
    _require_numpy()
    data = np.sort(np.asarray(values, np.float64)[~np.isnan(values)])
    if data.size == 0:
        return summary([])
    out = {"n": int(data.size), "mean": math.fsum(data.tolist()) / data.size}
    for name, q in QUANTILES:
        out[name] = np_percentile_sorted(data, q)
    out["max"] = float(data[-1])
    return out


@functools.lru_cache(maxsize=32)
def _np_indices(n_windows, B, seed):
    idx = np.asarray(bootstrap_indices(n_windows, B, seed), dtype=np.int64).reshape(B, n_windows)
    idx.flags.writeable = False
    return idx


def np_ratio_bootstrap(sums, counts, B: int = BOOT_B, seed: int = BOOT_SEED) -> dict:
    _require_numpy()
    sums, counts = np.asarray(sums, np.float64), np.asarray(counts, np.float64)
    idx = _np_indices(len(sums), B, seed)
    with np.errstate(invalid="ignore", divide="ignore"):
        reps = sums[idx].sum(1) / counts[idx].sum(1)
    return _ci(reps.tolist(), _ratio(sums.tolist(), counts.tolist()))


def np_paired_ratio_diff(sums_a, counts_a, sums_b, counts_b, B: int = BOOT_B,
                         seed: int = BOOT_SEED, scale: float = 1.0) -> dict:
    _require_numpy()
    arrays = [np.asarray(x, np.float64) for x in (sums_a, counts_a, sums_b, counts_b)]
    idx = _np_indices(len(arrays[0]), B, seed)
    with np.errstate(invalid="ignore", divide="ignore"):
        reps = (arrays[0][idx].sum(1) / arrays[1][idx].sum(1)
                - arrays[2][idx].sum(1) / arrays[3][idx].sum(1)) * scale
    estimate = (_ratio(arrays[0].tolist(), arrays[1].tolist())
                - _ratio(arrays[2].tolist(), arrays[3].tolist())) * scale
    return _ci(reps.tolist(), estimate)


class _Pooled:
    """Sorted pooled values with window labels, for weighted-percentile bootstrap."""

    def __init__(self, windows):
        values = [np.asarray(w, np.float64) for w in windows]
        labels = np.concatenate([np.full(len(v), i, np.int64) for i, v in enumerate(values)]) \
            if values else np.zeros(0, np.int64)
        flat = np.concatenate(values) if values else np.zeros(0)
        order = np.argsort(flat, kind="stable")
        self.values, self.labels, self.n = flat[order], labels[order], len(values)

    def percentile(self, q: float, multiplicity=None) -> float:
        if multiplicity is None:
            return np_percentile_sorted(self.values, q)
        cum = np.cumsum(multiplicity[self.labels])
        total = int(cum[-1]) if cum.size else 0
        if total == 0:
            return math.nan
        h = (total - 1) * q / 100.0
        lo = int(math.floor(h))
        hi = min(lo + 1, total - 1)
        v_lo = self.values[np.searchsorted(cum, lo, side="right")]
        v_hi = self.values[np.searchsorted(cum, hi, side="right")]
        return float(v_lo + (h - lo) * (v_hi - v_lo))


def np_percentile_bootstrap(windows, q: float, B: int = BOOT_B, seed: int = BOOT_SEED) -> dict:
    _require_numpy()
    pooled = _Pooled(windows)
    idx = _np_indices(pooled.n, B, seed)
    reps = [pooled.percentile(q, np.bincount(row, minlength=pooled.n)) for row in idx]
    return _ci(reps, pooled.percentile(q))


def np_paired_percentile_diff(windows_a, windows_b, q: float, B: int = BOOT_B,
                              seed: int = BOOT_SEED) -> dict:
    _require_numpy()
    a, b = _Pooled(windows_a), _Pooled(windows_b)
    idx = _np_indices(a.n, B, seed)
    reps = []
    for row in idx:
        mult = np.bincount(row, minlength=a.n)
        reps.append(a.percentile(q, mult) - b.percentile(q, mult))
    return _ci(reps, a.percentile(q) - b.percentile(q))
