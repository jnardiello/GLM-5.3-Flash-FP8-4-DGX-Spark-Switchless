#!/usr/bin/env python3
"""Campaign-level fidelity analysis: regimes, coverage, source-group bootstrap and verdicts.

Implements the campaign analysis in docs/fidelity/REPORT.md (method, amendments 12-13). Every prompt
comparison is reported for all positions, the dense regime (conditioning on at most
2,048 tokens) and the sparse regime (more than 2,048), with identical valid-position
masks for a contrast and its floor. Uncertainty comes from a percentile bootstrap over
source groups (session, conversation or prompt family), B = 2000, seed 20260927.
Sparse-regime fidelity comes from distributions averaged over at least three
executions per arm. Private per-window sums go to data/fidelity/metrics-v2/; public
aggregates without token ids, text, salts or project names go to docs/fidelity/metrics-v2/.
The configuration parser, grouping and verdict logic are stdlib-only; the analysis
needs numpy (data/fidelity/.venv/bin/python).
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fidelity_io as fio  # noqa: E402
import metrics as m  # noqa: E402

try:
    import numpy as np
except ImportError:  # pragma: no cover - the stdlib parts are tested without numpy
    np = None


REPO = fio.REPO
CONFIG_SCHEMA = "fidelity-campaign-config/1"
DENSE_MAX_CONDITIONING = 2048  # row p of a window predicts token p from p conditioning tokens
REGIMES = ("all", "dense", "sparse")
THRESHOLDS = {"kl_mean_excess_nats": 0.002, "kl_p99_excess_nats": 0.02, "top1_drop_pp": 0.5}
MIN_EXECUTIONS = 3
WITHIN, EXCEEDS, UNRESOLVED = "within_margin", "exceeds_margin", "unresolved"
BUCKET_EDGES = (2048, 8192, 32768, 65536)
TYPES = ("pair", "k_sensitivity", "ladder", "repeated", "mde", "bit_identity")
SPARSE_REASON = ("no validated estimator for the KL between population-average distributions under unequal "
                 "execution variability; repeated-execution quantities are descriptive (independent review)")
DIAGNOSTIC_NOTE = ("variance-subtraction diagnostics: with unequal execution variability they do not estimate the KL "
                   "between population-average distributions and can be negative while that KL is positive; "
                   "never used for verdicts")
P99_NOTE = ("p99 of the per-position mean cross-arm pairwise KL minus p99 of the per-position mean within-arm "
            "pairwise KL; nonzero even for identical arms with variable executions; descriptive only")
TOP_FRACTION = 0.05  # pooled-percentile bootstrap scans only the top slice, with exact fallback

METHOD = [
    "Coarse KL(P_ref || P_cand) per position over the partition {tokens in both top-K rows} + "
    "{actual token, when outside that intersection and known on both sides} + {rest}. Known-cell terms "
    "are computed from logprobs, exp(lp_ref) * (lp_ref - lp_cand), so a very negative finite logprob never "
    "underflows to zero probability; only -inf is a genuine zero (floored at 1e-12 and counted). By the "
    "data-processing inequality the value is a LOWER BOUND on the full-vocabulary KL; a small value does "
    "not certify a small full KL. Covered mass (probability of the shared cells including the actual "
    "token, per side) and the top-K overlap size are reported with every comparison, and any "
    "within-margin verdict is scoped to these coarse metrics.",
    "Regimes: row p of a window predicts token p after p conditioning tokens. Dense = p <= 2048; "
    "sparse = p > 2048; all = every scored row (p >= 1). The 2,048 boundary is defined by conditioning "
    "count; the sparse-attention indexer explanation remains a hypothesis. The prefill-path tag of row p "
    "is that of the chunk containing its predictor position p - 1 (8,192-token chunks; chunks shorter than "
    "2,048 rows are tagged marlin).",
    "Masks: a contrast and its floor are evaluated on the same windows and on the same positions: a "
    "position counts only when the coarse KL and the top-1 rows are valid in both the contrast and the "
    "floor. Delta NLL is a finite-subset mean: it additionally needs a finite actual-token logprob on every "
    "side; exclusions are counted, and nonfinite actual-token scores are published per side as missing "
    "(NaN), zero_probability (-inf, i.e. infinite NLL) or other. Missing windows, invalid rows and "
    "nonfinite actual-token scores are also published per run and regime.",
    "Bootstrap: percentile bootstrap over source groups, not windows or tokens (B = 2000, seed "
    "20260927). Means are token-weighted ratio estimators (sum over resampled groups / count). Contrast "
    "and floor share the resampled groups (paired). Each estimate carries a two-sided 95% percentile "
    "interval (2.5-97.5) and one-sided 95% bounds (5th and 95th percentiles of the replicates). With fewer "
    "than two contributing groups the point estimate is kept and bounds and SE are unavailable, so any "
    "criterion on it is unresolved.",
    "Definitions (amendment 13): mean excess = KL(ref, cand) - KL(ref, ref') on the same positions; "
    "p99 excess = p99(ref vs cand) - p99(ref vs ref'), a difference of pooled p99s, not a p99 of "
    "differences; agreement drop = agreement(ref, ref') - agreement(ref, cand) in percentage points.",
    "Verdicts per criterion: within_margin when the one-sided 95% upper bound is below the margin; "
    "exceeds_margin when the one-sided 95% lower bound is at or above it; otherwise unresolved. A regime "
    "is within_margin when all three criteria are, exceeds_margin when any criterion is, and unresolved "
    "otherwise. Margins (pre-registered): mean excess 0.002 nats, p99 excess 0.02 nats, agreement drop "
    "0.5 pp. They are pre-registered for Cm vs R0 only; other comparisons carry the same evaluation "
    "as descriptive context. A regime is unresolved when its floor is not estimable.",
    "Sparse regime: single-execution excess over the R0 floor is reported only as additional "
    "operational disagreement, because a less variable arm can show negative excess despite systematic "
    "drift. The sparse verdict is unresolved: there is no validated estimator for the KL between "
    "population-average distributions when execution variability differs between arms. The overall "
    "verdict is the conjunction of the dense and sparse verdicts (so it can be exceeds_margin, never "
    "within_margin, while the sparse verdict is unresolved); the single-execution overall criteria are "
    "published alongside for transparency.",
    "Repeated executions (descriptive): for each position the partition is the set of tokens present in "
    "the top-K rows of every execution of both arms, plus the actual token (when outside that set and "
    "known in every execution), plus rest, so each arm's equal-weight mixture over its executions is exact "
    "on the partition (mixture logprobs by log-sum-exp) and every KL is a lower bound on its full "
    "counterpart. Reported: D = KL(mixture_ref || mixture_cand), within-arm pairwise KL W_arm (ordered "
    "pairs of distinct executions), cross-arm pairwise KL X, pairwise top-1 agreements, covered mass and "
    "shared-cell count, with group-bootstrap intervals. The variance-subtraction quantities "
    "D - (W_ref + W_cand) / (2E) and X - (W_ref + W_cand) / 2 are labelled diagnostics: with unequal "
    "execution variability they do not estimate the KL between population-average distributions and can "
    "be negative while it is positive. The p99 of cross minus within pairwise KL is nonzero even for "
    "identical arms with variable executions. None of these enters a verdict. The reference arm uses R0 "
    "A, R0 B and the cross-boot R0 execution; the other arms use A, rep2 and rep3; with fewer the pair is "
    "reported as insufficient_executions.",
    "Bit identity: same-recipe repeats and ladder negative controls are checked for bitwise equality of "
    "lp_actual, topk_ids and topk_lp over rows 1..2048 (dense regime) on every shared window; the report "
    "gives windows and rows compared, rows differing and the first differing row. Ladder attribution is "
    "withheld when a strict negative control (a step that must not change target logits) fails, and is "
    "pending while the control cannot be evaluated.",
    "MDE: 2.8 x the bootstrap SE (80% power, two-sided alpha 0.05) of paired reference-only null "
    "contrasts KL(R0 A, R0 X) - KL(R0 A, R0 B) on the same positions, for each further R0 execution X; the "
    "governing MDE is the largest. The SE of the floor mean is reported separately as context. Without a "
    "paired null contrast the MDE is unavailable. A zero SE (bit-identical dense contrast) gives a "
    "degenerate MDE of zero.",
    "Grouping: windows from one Claude Code or omp session share a group (overlapping or sequential "
    "windows of one session are not independent); synthetic Italian windows are grouped by "
    "conversation file; model-native windows continuing a session decode prompt join that session's "
    "group, the others are grouped by their decode or native prompt id; any window without provenance "
    "falls back to its project and category. Group identifiers stay private; only counts are published.",
    "Model-native windows additionally report continuation-only positions (rows at or after the prompt "
    "length, i.e. the R0-generated tokens).",
]


class ConfigError(ValueError):
    pass


# ---------------------------------------------------------------- stdlib: config, groups, verdicts


def arm_of(run_key: str) -> str:
    return run_key.split("/", 1)[0]


def load_config(path: Path) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_config(cfg)


def validate_config(cfg: dict) -> dict:
    if cfg.get("schema") != CONFIG_SCHEMA:
        raise ConfigError(f"config schema must be {CONFIG_SCHEMA}")
    comps = cfg.get("comparisons")
    if not isinstance(comps, list) or not comps:
        raise ConfigError("config needs a non-empty 'comparisons' list")
    names = [c.get("name") for c in comps]
    if None in names or len(set(names)) != len(names):
        raise ConfigError("every comparison needs a unique name")
    by_name = {c["name"]: c for c in comps}
    required = {"pair": ("ref", "cand"), "k_sensitivity": ("ref", "cand", "k_low"),
                "ladder": ("chain",), "repeated": ("subset", "arms", "pairs"), "mde": ("floor",),
                "bit_identity": ("checks",)}
    for comp in comps:
        kind = comp.get("type")
        if kind not in TYPES:
            raise ConfigError(f"{comp['name']}: unknown type {kind!r}")
        for key in required[kind]:
            if key not in comp:
                raise ConfigError(f"{comp['name']}: missing '{key}'")
        floor = comp.get("floor")
        if floor is not None:
            target = by_name.get(floor)
            if target is None or target.get("type") != "pair" or target.get("floor"):
                raise ConfigError(f"{comp['name']}: floor {floor!r} must name a pair comparison without floor")
        if kind == "ladder":
            if len(comp["chain"]) < 2:
                raise ConfigError(f"{comp['name']}: a ladder needs at least two rungs")
            steps = {f"{a}->{b}" for a, b in zip(comp["chain"], comp["chain"][1:])}
            for step in comp.get("strict_negative_controls", []) + comp.get("negative_control_steps", []):
                if step not in steps:
                    raise ConfigError(f"{comp['name']}: negative control {step!r} is not a ladder step")
        if kind == "bit_identity":
            for check in comp["checks"]:
                if not all(key in check for key in ("name", "ref", "cand")):
                    raise ConfigError(f"{comp['name']}: every check needs name, ref and cand")
        if kind == "repeated":
            for pair in comp["pairs"]:
                if len(pair) != 2 or any(arm not in comp["arms"] for arm in pair):
                    raise ConfigError(f"{comp['name']}: pair {pair} names an unknown arm")
            if int(comp.get("executions", MIN_EXECUTIONS)) < 2:
                raise ConfigError(f"{comp['name']}: executions must be at least 2")
    return cfg


def session_of(segment: str) -> str:
    """windows-meta 'segment' is '<session key>:<segment index>'; windows of one session share a group."""
    return segment.rsplit(":", 1)[0]


def group_key(entry: dict, windows_meta: dict, native: dict, decode_meta: dict) -> tuple[str, str]:
    """(private group key, rule name) for one corpus window."""
    wid = entry["id"]
    meta = windows_meta.get(wid) or {}
    if "segment" in meta:
        return "session:" + session_of(meta["segment"]), "session"
    if "file" in meta:
        return "conversation:" + str(meta["file"]), "conversation"
    prov = (native.get("windows") or {}).get(wid)
    if prov is not None:
        did = prov.get("decode_id", "")
        dmeta = decode_meta.get(did) or {}
        if "segment" in dmeta:
            return "session:" + session_of(dmeta["segment"]), "native_session_prompt"
        return "prompt:" + did, "native_prompt"
    return f"project:{entry.get('project', '')}/{entry.get('category', '')}", "project_category_fallback"


def build_groups(entries, windows_meta, native, decode_meta):
    groups, rules = {}, {}
    for entry in entries:
        key, rule = group_key(entry, windows_meta, native, decode_meta)
        groups[entry["id"]] = key
        rules[rule] = rules.get(rule, 0) + 1
    return groups, rules


def regime_rows(n_tokens: int) -> dict:
    """Scored rows (p >= 1) of an n-token window per regime."""
    scored = max(0, n_tokens - 1)
    dense = min(scored, DENSE_MAX_CONDITIONING)
    return {"all": scored, "dense": dense, "sparse": scored - dense}


def criterion(ci: dict | None, margin: float) -> str:
    """within_margin if the one-sided 95% UB < margin, exceeds if the one-sided LB >= margin."""
    if not ci or ci.get("ub95") is None or ci.get("lb95") is None:
        return UNRESOLVED
    if ci["ub95"] < margin:
        return WITHIN
    if ci["lb95"] >= margin:
        return EXCEEDS
    return UNRESOLVED


def combine(outcomes) -> str:
    outcomes = list(outcomes)
    if outcomes and all(o == WITHIN for o in outcomes):
        return WITHIN
    if any(o == EXCEEDS for o in outcomes):
        return EXCEEDS
    return UNRESOLVED


def select_executions(candidates: list, available: set, count: int):
    """The first `count` configured runs that exist, or None when fewer exist."""
    chosen = [run for run in candidates if run in available]
    return chosen[:count] if len(chosen) >= count else None


# ---------------------------------------------------------------- numpy: statistics


@functools.lru_cache(maxsize=64)
def multiplicity(n_groups: int, B: int, seed: int):
    """(B, n_groups) resampling counts, from the same index stream as metrics.bootstrap_indices."""
    idx = np.asarray(m.bootstrap_indices(n_groups, B, seed), np.int64).reshape(B, n_groups)
    out = np.zeros((B, n_groups), np.int64)
    rows = np.repeat(np.arange(B), n_groups)
    np.add.at(out, (rows, idx.ravel()), 1)
    out.flags.writeable = False
    return out


def ci_dict(estimate, reps, groups=None) -> dict:
    """Point estimate with percentile bounds; bounds and SE are unavailable when fewer than
    two independent groups contribute (the bootstrap would report a spurious zero width)."""
    est = None if estimate is None or not math.isfinite(estimate) else float(estimate)
    reps = np.sort(np.asarray(reps, np.float64)[np.isfinite(reps)])
    out = {"estimate": est, "ci_low": None, "ci_high": None, "lb95": None, "ub95": None,
           "se": None, "B": int(reps.size)}
    if groups is not None and groups < 2:
        out.update(B=0, note="fewer than two independent groups: bounds unavailable")
        return out
    if est is None or reps.size < 2:
        return out
    out.update(ci_low=m.np_percentile_sorted(reps, 2.5), ci_high=m.np_percentile_sorted(reps, 97.5),
               lb95=m.np_percentile_sorted(reps, 5.0), ub95=m.np_percentile_sorted(reps, 95.0),
               se=float(np.std(reps, ddof=1)))
    return out


def _ratio(num, den):
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


class Groups:
    """Per-group sums over a selection, with a shared group resampling."""

    def __init__(self, grp, sel, B, seed):
        counts = np.bincount(grp[sel], minlength=0)
        self.keep = np.flatnonzero(counts)
        remap = np.full(max(len(counts), 1), -1, np.int64)
        remap[self.keep] = np.arange(self.keep.size)
        self.labels = remap[grp[sel]] if sel.any() else np.zeros(0, np.int64)
        self.n = int(self.keep.size)
        self.sel = sel
        self.count = np.bincount(self.labels, minlength=self.n).astype(np.float64)
        self.M = multiplicity(self.n, B, seed) if self.n else None

    def sums(self, values, mask=None):
        v = values[self.sel]
        lab = self.labels
        if mask is not None:
            mk = mask[self.sel]
            v, lab = v[mk], lab[mk]
        s = np.bincount(lab, weights=v.astype(np.float64), minlength=self.n)
        c = np.bincount(lab, minlength=self.n).astype(np.float64)
        return s, c

    def ratio(self, s, c, scale=1.0):
        est = _ratio(s.sum(), c.sum()) * scale
        reps = _ratio(self.M @ s, self.M @ c) * scale
        return est, reps

    def ci(self, estimate, reps) -> dict:
        return ci_dict(estimate, reps, self.n)


class Pooled:
    """Pooled values with group labels: weighted linear-interpolation percentiles.

    Same definition as metrics._Pooled; the bootstrap scans only the top slice of the
    sorted values and falls back to the full scan when the percentile lies below it.
    """

    def __init__(self, values, labels, n_groups):
        order = np.argsort(values, kind="stable")
        self.v = np.asarray(values, np.float64)[order]
        self.l = np.asarray(labels, np.int64)[order]
        self.n = n_groups
        size = self.v.size
        top = min(size, max(4096, int(math.ceil(size * TOP_FRACTION))))
        self.start = size - top
        self.counts = np.bincount(self.l, minlength=n_groups)
        self.below = np.bincount(self.l[:self.start], minlength=n_groups)
        self.top_l, self.top_v = self.l[self.start:], self.v[self.start:]

    def percentile(self, q, mult=None) -> float:
        if mult is None:
            return m.np_percentile_sorted(self.v, q)
        total = int(mult @ self.counts)
        if total == 0:
            return math.nan
        h = (total - 1) * q / 100.0
        lo = int(math.floor(h))
        hi = min(lo + 1, total - 1)
        below = int(mult @ self.below)
        if lo >= below:
            cum = np.cumsum(mult[self.top_l])
            v_lo = self.top_v[np.searchsorted(cum, lo - below, side="right")]
            v_hi = self.top_v[np.searchsorted(cum, hi - below, side="right")]
        else:
            cum = np.cumsum(mult[self.l])
            v_lo = self.v[np.searchsorted(cum, lo, side="right")]
            v_hi = self.v[np.searchsorted(cum, hi, side="right")]
        return float(v_lo + (h - lo) * (v_hi - v_lo))

    def reps(self, q, M):
        return np.asarray([self.percentile(q, row) for row in M])


def dist(values) -> dict:
    values = np.asarray(values, np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0, "mean": None, "p10": None, "median": None}
    s = np.sort(values)
    return {"n": int(s.size), "mean": float(s.mean()), "p10": m.np_percentile_sorted(s, 10.0),
            "median": m.np_percentile_sorted(s, 50.0)}


# ---------------------------------------------------------------- numpy: per position


def np_coverage(ref_ids, ref_lp, cand_ids, cand_lp, extra_ids, ref_extra, cand_extra, chunk=None):
    """Covered mass per side (shared top-K cells plus the actual-token cell of np_coarse_kl)
    and the top-K overlap size |topK_ref & topK_cand|."""
    n = ref_ids.shape[0]
    cov_r = np.full(n, np.nan)
    cov_c = np.full(n, np.nan)
    overlap = np.zeros(n, np.int16)
    if chunk is None:
        chunk = max(64, (1 << 22) // max(1, ref_ids.shape[1] * cand_ids.shape[1]))
    with np.errstate(invalid="ignore", over="ignore"):
        for s in range(0, n, chunk):
            e = min(n, s + chunk)
            ri, ci = ref_ids[s:e], cand_ids[s:e]
            rl, cl = ref_lp[s:e].astype(np.float64), cand_lp[s:e].astype(np.float64)
            match = (ri[:, :, None] == ci[:, None, :]) & (ri >= 0)[:, :, None]
            inter = match.any(2)
            col = np.argmax(match, axis=2)
            p = np.where(inter, np.exp(rl), 0.0).sum(1)
            q = np.where(inter, np.take_along_axis(np.exp(cl), col, axis=1), 0.0).sum(1)
            eid = extra_ids[s:e]
            elr, elc = ref_extra[s:e].astype(np.float64), cand_extra[s:e].astype(np.float64)
            both = (ri == eid[:, None]).any(1) & (ci == eid[:, None]).any(1)
            use = (eid >= 0) & ~both & ~np.isnan(elr) & ~np.isnan(elc)
            cov_r[s:e] = p + np.where(use, np.exp(elr), 0.0)
            cov_c[s:e] = q + np.where(use, np.exp(elc), 0.0)
            overlap[s:e] = inter.sum(1)
    return cov_r, cov_c, overlap


def pair_window(r, c, k=None) -> dict:
    ids = r["ids"]
    kr = k or r["topk_ids"].shape[1]
    kc = k or c["topk_ids"].shape[1]
    rid, rlp = r["topk_ids"][1:, :kr], r["topk_lp"][1:, :kr]
    cid, clp = c["topk_ids"][1:, :kc], c["topk_lp"][1:, :kc]
    eid = ids[1:].astype(np.int64)
    ra, ca = r["lp_actual"][1:], c["lp_actual"][1:]
    kl, valid, clamped, floored = m.np_coarse_kl(rid, rlp, cid, clp, eid, ra, ca)
    top1 = m.np_top1(rid, rlp, cid, clp)
    cov_r, cov_c, overlap = np_coverage(rid, rlp, cid, clp, eid, ra, ca)
    ra64, ca64 = ra.astype(np.float64), ca.astype(np.float64)
    finite = np.isfinite(ra64) & np.isfinite(ca64)
    return {"kl": kl, "ok": valid & (top1 >= 0), "top1": top1 == 1,
            "dnll": np.where(finite, ra64 - ca64, np.nan), "cov_ref": cov_r.astype(np.float32),
            "cov_cand": cov_c.astype(np.float32), "overlap": overlap, "clamped": clamped, "floored": floored,
            "ra_state": actual_state(ra64), "ca_state": actual_state(ca64)}


ACTUAL_STATES = {1: "missing", 2: "zero_probability", 3: "other_nonfinite"}


def actual_state(lp):
    """0 finite; 1 NaN (missing); 2 -inf (genuine zero probability: infinite NLL); 3 other."""
    state = np.full(lp.shape, 3, np.int8)
    state[np.isfinite(lp)] = 0
    state[np.isnan(lp)] = 1
    state[lp == -np.inf] = 2
    return state


def row_top1(ids, lp):
    real = ids >= 0
    bad = (np.isnan(lp) & real).any(1) | ~real.any(1)
    arg = np.argmax(np.where(real, np.nan_to_num(lp, nan=-np.inf), -np.inf), axis=1)
    return ids[np.arange(ids.shape[0]), arg], bad


def partition_logprobs(tids, tlps, eid, alps, tol=m.CLAMP_TOL):
    """Logprobs of R rows on a shared partition: tokens in every row's top-K, the actual token
    (outside that set, known in every row) and rest. Known cells keep their logprobs, so a very
    negative finite value never underflows to a zero probability; -inf is a genuine zero.
    Returns (R, n, K+2) logprobs, valid (n), cells (n)."""
    R = len(tids)
    base = tids[0]
    n, K = base.shape
    in_all = base >= 0
    cols = [np.broadcast_to(np.arange(K), (n, K))]
    for j in range(1, R):
        match = base[:, :, None] == tids[j][:, None, :]
        in_all = in_all & match.any(2)
        cols.append(np.argmax(match, axis=2))
    bad = np.zeros(n, bool)
    for j in range(R):
        real = tids[j] >= 0
        bad |= (np.isnan(tlps[j]) & real).any(1) | ~real.any(1)
    in_set = ((base == eid[:, None]) & in_all).any(1)
    known = np.ones(n, bool)
    for j in range(R):
        known &= ~np.isnan(alps[j])
    use = (eid >= 0) & ~in_set & known
    logs = np.empty((R, n, K + 2))
    over = np.zeros(n, bool)
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        for j in range(R):
            lj = np.where(in_all, np.take_along_axis(tlps[j].astype(np.float64), cols[j], 1), -np.inf)
            ej = np.where(use, alps[j].astype(np.float64), -np.inf)
            rest, _, over_j = m.np_log_rest(np.concatenate([lj, ej[:, None]], 1), tol)
            over |= over_j
            logs[j, :, :K] = lj
            logs[j, :, K] = ej
            logs[j, :, K + 1] = rest
    return logs, ~bad & ~over, in_all.sum(1) + use


def kl_rows(LP, LQ):
    """KL over the last axis from logprobs (metrics.np_kl_terms: no underflow, -inf is zero)."""
    return m.np_kl_terms(LP, LQ)[0].sum(-1)


def log_mean_exp(L):
    """Log of the mean probability over axis 0 (an equal-weight mixture), computed stably."""
    top = L.max(0)
    safe = np.where(np.isfinite(top), top, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return safe + np.log(np.exp(L - safe).mean(0))


def repeated_window(rows, E, chunk=4096) -> dict:
    """Per-position averaged/pairwise KLs for E reference rows followed by E candidate rows."""
    ids = rows[0]["ids"]
    K = min(r["topk_ids"].shape[1] for r in rows)
    n = ids.shape[0] - 1
    keys = ("D", "WR", "WC", "X", "aRR", "aCC", "aRC", "covR", "covC", "cells")
    out = {key: np.zeros(n) for key in keys}
    out["ok"] = np.zeros(n, bool)
    ref_pairs = [(i, j) for i in range(E) for j in range(E) if i != j]
    cross = [(i, E + j) for i in range(E) for j in range(E)]
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        tids = [r["topk_ids"][1 + s:1 + e, :K] for r in rows]
        tlps = [r["topk_lp"][1 + s:1 + e, :K] for r in rows]
        alps = [r["lp_actual"][1 + s:1 + e] for r in rows]
        eid = ids[1 + s:1 + e].astype(np.int64)
        P, ok, cells = partition_logprobs(tids, tlps, eid, alps)
        tops = [row_top1(t, lp.astype(np.float64)) for t, lp in zip(tids, tlps)]
        for _, bad in tops:
            ok &= ~bad
        ref_mean, cand_mean = log_mean_exp(P[:E]), log_mean_exp(P[E:])
        out["D"][s:e] = kl_rows(ref_mean, cand_mean)
        out["WR"][s:e] = np.mean([kl_rows(P[i], P[j]) for i, j in ref_pairs], axis=0)
        out["WC"][s:e] = np.mean([kl_rows(P[E + i], P[E + j]) for i, j in ref_pairs], axis=0)
        out["X"][s:e] = np.mean([kl_rows(P[i], P[j]) for i, j in cross], axis=0)
        t = [top for top, _ in tops]
        out["aRR"][s:e] = np.mean([t[i] == t[j] for i in range(E) for j in range(i + 1, E)], axis=0)
        out["aCC"][s:e] = np.mean([t[E + i] == t[E + j] for i in range(E) for j in range(i + 1, E)], axis=0)
        out["aRC"][s:e] = np.mean([t[i] == t[j] for i, j in cross], axis=0)
        out["covR"][s:e] = np.exp(ref_mean[:, :K + 1]).sum(1)
        out["covC"][s:e] = np.exp(cand_mean[:, :K + 1]).sum(1)
        out["cells"][s:e] = cells
        out["ok"][s:e] = ok
    return out


# ---------------------------------------------------------------- data access


class PairData:
    def __init__(self, parts: list, win_ids: list):
        self.win_ids = win_ids
        self.index, start = {}, 0
        for wid, part in zip(win_ids, parts):
            size = part["kl"].shape[0]
            self.index[wid] = (start, start + size)
            start += size
        keys = ("kl", "ok", "top1", "dnll", "cov_ref", "cov_cand", "overlap", "clamped", "floored",
                "ra_state", "ca_state")
        empty = {"ok": bool, "top1": bool, "clamped": bool, "floored": bool, "ra_state": np.int8,
                 "ca_state": np.int8, "overlap": np.int16, "cov_ref": np.float32, "cov_cand": np.float32}
        self.arr = {k: (np.concatenate([p[k] for p in parts]) if parts else np.zeros(0, empty.get(k, np.float64)))
                    for k in keys}


class Data:
    def __init__(self, args):
        self.args = args
        self.manifest = args.manifest
        _, entries = fio.load_manifest(args.manifest, "windows")
        self.entries = sorted(entries, key=lambda e: e["id"])
        self.by_id = {e["id"]: e for e in self.entries}
        corpus = args.manifest.parent
        native = fio.read_json(corpus / "native-provenance.json") or {}
        self.groups, self.group_rules = build_groups(
            self.entries, fio.read_json(corpus / "windows-meta.json") or {}, native,
            fio.read_json(corpus / "decode-meta.json") or {})
        self.prompt_tokens = {wid: int(row["prompt_tokens"]) for wid, row in (native.get("windows") or {}).items()}
        self.categories = sorted({e.get("category", "unknown") for e in self.entries})
        self.runs, self.pairs, self._tokens = {}, {}, {}

    def subset(self, name) -> list:
        if not name:
            return [e["id"] for e in self.entries]
        path = self.manifest.parent / name
        wanted = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
                  if line.strip() and not line.startswith("#")]
        unknown = [w for w in wanted if w not in self.by_id]
        if unknown:
            raise SystemExit(f"subset {name}: {len(unknown)} ids not in the manifest")
        return sorted(set(wanted))

    def tokens(self, wid):
        if wid not in self._tokens:
            self._tokens[wid] = np.asarray(fio.load_entry_tokens(self.manifest, self.by_id[wid]), np.int64)
        return self._tokens[wid]

    def _read(self, run_dir: Path, wid: str):
        sidecar = fio.read_json(run_dir / "prompt" / f"{wid}.json")
        path = run_dir / "prompt" / f"{wid}.npz"
        if not sidecar or sidecar.get("status") != "ok" or not path.exists():
            return None
        try:
            with np.load(path) as data:
                arrays = {name: data[name] for name in fio.PROMPT_ARRAYS}
        except Exception:
            return None
        n = arrays["ids"].shape[0]
        if any(arrays[name].shape[0] != n for name in arrays) or arrays["topk_ids"].shape != arrays["topk_lp"].shape:
            return None
        return arrays

    def run(self, key: str) -> dict:
        """Inventory of one run (snapshot): ok windows, K and per-window validity counts."""
        if key in self.runs:
            return self.runs[key]
        run_dir = self.args.raw_root / key
        record = fio.read_json(run_dir / "run.json") or {}
        info = {"key": key, "K": record.get("K"), "windows": set(), "stats": {}}
        if (run_dir / "prompt").is_dir():
            for entry in self.entries:
                arrays = self._read(run_dir, entry["id"])
                if arrays is None:
                    continue
                if not np.array_equal(arrays["ids"].astype(np.int64), self.tokens(entry["id"])):
                    raise SystemExit(f"{key}: {entry['id']} raw ids differ from the corpus")
                pos = np.arange(1, arrays["ids"].shape[0])
                tid, tlp = arrays["topk_ids"][1:], arrays["topk_lp"][1:]
                real = tid >= 0
                invalid = (np.isnan(tlp) & real).any(1) | ~real.any(1)
                missing = np.isnan(arrays["lp_actual"][1:])
                zero = arrays["lp_actual"][1:] == -np.inf  # genuine zero probability: infinite NLL
                other = ~np.isfinite(arrays["lp_actual"][1:]) & ~missing & ~zero  # e.g. +inf
                dense = pos <= DENSE_MAX_CONDITIONING
                info["stats"][entry["id"]] = {
                    "invalid_rows": {"dense": int((invalid & dense).sum()), "sparse": int((invalid & ~dense).sum())},
                    "missing_actual": {"dense": int((missing & dense).sum()), "sparse": int((missing & ~dense).sum())},
                    "zero_probability_actual": {"dense": int((zero & dense).sum()), "sparse": int((zero & ~dense).sum())},
                    "other_nonfinite_actual": {"dense": int((other & dense).sum()),
                                               "sparse": int((other & ~dense).sum())},
                }
                info["windows"].add(entry["id"])
                if info["K"] is None:
                    info["K"] = int(tid.shape[1])
        self.runs[key] = info
        print(f"inventory {key}: windows_ok={len(info['windows'])} K={info['K']}", flush=True)
        return info

    def load(self, key, wid):
        return self._read(self.args.raw_root / key, wid)

    def pair(self, ref: str, cand: str, k=None) -> PairData:
        cache_key = (ref, cand, k)
        if cache_key not in self.pairs:
            started = time.time()
            wids = sorted(self.run(ref)["windows"] & self.run(cand)["windows"])
            parts, kept = [], []
            for wid in wids:
                r, c = self.load(ref, wid), self.load(cand, wid)
                if r is None or c is None:
                    continue
                parts.append(pair_window(r, c, k))
                kept.append(wid)
            self.pairs[cache_key] = PairData(parts, kept)
            print(f"pair {cand} vs {ref} (K={k or 'run'}): windows={len(kept)} "
                  f"positions={sum(p['kl'].shape[0] for p in parts)} in {time.time() - started:.1f}s", flush=True)
        return self.pairs[cache_key]

    def missingness(self, key: str, expected: list) -> dict:
        info = self.run(key)
        out = {"windows_expected": len(expected), "windows_ok": 0, "windows_missing": 0,
               "rows_in_missing_windows": {r: 0 for r in REGIMES},
               "invalid_rows": {r: 0 for r in REGIMES}, "missing_actual": {r: 0 for r in REGIMES},
               "zero_probability_actual": {r: 0 for r in REGIMES}, "other_nonfinite_actual": {r: 0 for r in REGIMES}}
        for wid in expected:
            if wid in info["windows"]:
                out["windows_ok"] += 1
                stats = info["stats"][wid]
                for field in ("invalid_rows", "missing_actual", "zero_probability_actual", "other_nonfinite_actual"):
                    for regime in ("dense", "sparse"):
                        out[field][regime] += stats[field][regime]
                        out[field]["all"] += stats[field][regime]
            else:
                out["windows_missing"] += 1
                for regime, count in regime_rows(self.by_id[wid]["n_tokens"]).items():
                    out["rows_in_missing_windows"][regime] += count
        return out

    def view(self, expected: list, pairs: list):
        """Aligned per-position labels and arrays over the windows present in every pair."""
        wids = [w for w in expected if all(w in p.index for p in pairs)]
        sizes = np.asarray([pairs[0].index[w][1] - pairs[0].index[w][0] for w in wids], np.int64)
        for p in pairs[1:]:
            if any(p.index[w][1] - p.index[w][0] != size for w, size in zip(wids, sizes)):
                raise SystemExit("pair views disagree on window lengths")
        arrays = []
        for p in pairs:
            idx = (np.concatenate([np.arange(*p.index[w]) for w in wids]) if wids else np.zeros(0, np.int64))
            arrays.append({k: v[idx] for k, v in p.arr.items()})
        V = self.labels(wids, sizes)
        return V, arrays

    def labels(self, wids, sizes) -> dict:
        keys = sorted({self.groups[w] for w in wids})
        gidx = {k: i for i, k in enumerate(keys)}
        win = np.repeat(np.arange(len(wids)), sizes) if wids else np.zeros(0, np.int64)
        pos = (np.concatenate([np.arange(1, s + 1) for s in sizes]) if wids else np.zeros(0, np.int64))
        n_tok = np.repeat(sizes + 1, sizes) if wids else np.zeros(0, np.int64)
        prompt_len = np.asarray([self.prompt_tokens.get(w, -1) for w in wids], np.int64)
        per_pos_prompt = prompt_len[win] if wids else np.zeros(0, np.int64)
        cats = np.asarray([self.categories.index(self.by_id[w].get("category", "unknown")) for w in wids], np.int64)
        return {
            "wids": wids, "win": win, "pos": pos, "G": len(keys),
            "grp": np.asarray([gidx[self.groups[w]] for w in wids], np.int64)[win] if wids else np.zeros(0, np.int64),
            "cat": cats[win] if wids else np.zeros(0, np.int64),
            "cont": (per_pos_prompt >= 0) & (pos >= per_pos_prompt),
            "bucket": np.digitize(pos, BUCKET_EDGES), "path": row_paths(pos, n_tok),
        }


def row_paths(pos, n_tok):
    """Prefill path per logprob row (index into metrics.PATHS): row p is produced by the hidden state
    of predictor position p - 1, so it takes the path of the chunk containing p - 1."""
    start = ((pos - 1) // m.BATCHED_TOKENS) * m.BATCHED_TOKENS
    return (np.minimum(m.BATCHED_TOKENS, n_tok - start) >= m.MARLIN_BELOW_ROWS).astype(np.int8)


def regime_mask(V, regime):
    if regime == "dense":
        return V["pos"] <= DENSE_MAX_CONDITIONING
    if regime == "sparse":
        return V["pos"] > DENSE_MAX_CONDITIONING
    return np.ones(V["pos"].shape[0], bool)


# ---------------------------------------------------------------- blocks


def side_block(g: Groups, S, dn) -> dict:
    """Metrics of one side on the joint mask held by `g`."""
    ks, kc = g.sums(np.nan_to_num(S["kl"]))
    summary = m.np_summary(S["kl"][g.sel])
    summary["mean_ci"] = g.ci(*g.ratio(ks, kc))
    ts, tc = g.sums(S["top1"].astype(np.float64))
    ds, dc = g.sums(np.nan_to_num(S["dnll"]), dn)
    dnll_groups = int(np.count_nonzero(dc))  # bounds need two groups with finite delta NLL
    delta = dict(ci_dict(*g.ratio(ds, dc), groups=dnll_groups) if dc.sum() else ci_dict(None, []),
                 n=int(dc.sum()), groups=dnll_groups, finite_subset=True,
                 excluded_nonfinite=int(g.sel.sum() - dc.sum()),
                 note="mean over positions whose actual-token logprob is finite on every side of the "
                      "contrast and its floor; nonfinite scores are counted in the block's actual_token_nonfinite")
    out = {"kl": summary,
           "top1_agreement": dict(g.ci(*g.ratio(ts, tc)), n=int(tc.sum())),
           "delta_nll": delta,
           "clamped": int(S["clamped"][g.sel].sum()), "floored": int(S["floored"][g.sel].sum())}
    for field in ("cov_ref", "cov_cand"):
        cs, cc = g.sums(S[field].astype(np.float64))
        out[{"cov_ref": "covered_mass_ref", "cov_cand": "covered_mass_cand"}[field]] = dict(
            dist(S[field][g.sel]), mean_ci=g.ci(*g.ratio(cs, cc)))
    out["topk_overlap"] = dist(S["overlap"][g.sel])
    return out


def nonfinite_counts(S, sel) -> dict:
    """Nonfinite actual-token scores per side over the selected positions, before any masking
    (a +inf or other nonfinite score can invalidate the KL row itself)."""
    out = {}
    for side, key in (("ref", "ra_state"), ("cand", "ca_state")):
        states = S[key][sel]
        out[side] = {name: int((states == code).sum()) for code, name in ACTUAL_STATES.items()}
        out[side]["infinite_nll"] = out[side]["zero_probability"]
    return out


def block(V, C, F, sel, B, seed, with_p99=True) -> dict:
    """Contrast (and floor) metrics on the joint valid mask within `sel`, with paired excess."""
    joint = sel & C["ok"] & (F["ok"] if F is not None else True)
    dn = np.isfinite(C["dnll"]) & (np.isfinite(F["dnll"]) if F is not None else True)
    out = {"positions": int(sel.sum()), "positions_joint_valid": int(joint.sum()),
           "excluded_by_joint_mask": int(sel.sum() - joint.sum()),
           "windows": int(np.unique(V["win"][joint]).size), "groups": 0,
           "actual_token_nonfinite": {"contrast": nonfinite_counts(C, sel)}}
    if F is not None:
        out["actual_token_nonfinite"]["floor"] = nonfinite_counts(F, sel)
    if not joint.any():
        out["status"] = "no_positions"
        return out
    g = Groups(V["grp"], joint, B, seed)
    out["groups"] = g.n
    out["contrast"] = side_block(g, C, dn)
    cp = Pooled(C["kl"][joint], g.labels, g.n) if with_p99 else None
    c_reps = cp.reps(99.0, g.M) if with_p99 else None
    if with_p99:
        out["contrast"]["kl"]["p99_ci"] = g.ci(cp.percentile(99.0), c_reps)
    if F is not None:
        out["floor"] = side_block(g, F, dn)
        cs, cc = g.sums(np.nan_to_num(C["kl"]))
        fs, fc = g.sums(np.nan_to_num(F["kl"]))
        ce, cr = g.ratio(cs, cc)
        fe, fr = g.ratio(fs, fc)
        ex = {"kl_mean_excess": g.ci(ce - fe, cr - fr)}
        ts, tc = g.sums(C["top1"].astype(np.float64))
        us, uc = g.sums(F["top1"].astype(np.float64))
        te, tr = g.ratio(ts, tc, 100.0)
        ue, ur = g.ratio(us, uc, 100.0)
        ex["top1_drop_pp"] = g.ci(ue - te, ur - tr)
        estimate = out["contrast"]["kl"]["p99"] - out["floor"]["kl"]["p99"]
        if with_p99:
            fp = Pooled(F["kl"][joint], g.labels, g.n)
            ex["kl_p99_excess"] = g.ci(estimate, c_reps - fp.reps(99.0, g.M))
        else:
            ex["kl_p99_excess"] = {"estimate": estimate, "note": "point estimate only"}
        out["excess"] = ex
    return out


def criteria_of(blk, thresholds) -> dict:
    ex = blk.get("excess") or {}
    return {"kl_mean_excess": criterion(ex.get("kl_mean_excess"), thresholds["kl_mean_excess_nats"]),
            "kl_p99_excess": criterion(ex.get("kl_p99_excess"), thresholds["kl_p99_excess_nats"]),
            "top1_drop_pp": criterion(ex.get("top1_drop_pp"), thresholds["top1_drop_pp"])}


# ---------------------------------------------------------------- analyses


def pair_analysis(data, comp, comps, B, seed, thresholds, sparse_links) -> tuple[dict, dict | None]:
    floor = comps.get(comp.get("floor")) if comp.get("floor") else None
    runs = [comp["ref"], comp["cand"]] + ([floor["ref"], floor["cand"]] if floor else [])
    runs = list(dict.fromkeys(runs))
    result = {"schema": "fidelity-campaign-pair/1", "name": comp["name"], "type": "pair",
              "role": comp.get("role", "descriptive"), "ref": comp["ref"], "cand": comp["cand"],
              "floor": comp.get("floor"), "subset": comp.get("subset"),
              "K": {"ref": data.run(comp["ref"])["K"], "cand": data.run(comp["cand"])["K"]}}
    missing = [r for r in runs if not data.run(r)["windows"]]
    if missing:
        result.update(status="missing_runs", missing_runs=missing)
        return result, None
    expected = data.subset(comp.get("subset"))
    pairs = [data.pair(comp["ref"], comp["cand"])] + ([data.pair(floor["ref"], floor["cand"])] if floor else [])
    V, arrays = data.view(expected, pairs)
    C, F = arrays[0], (arrays[1] if floor else None)
    if not V["wids"]:
        result.update(status="empty", note="no window is present in every run of the comparison",
                      windows={"expected": len(expected), "compared": 0},
                      missingness={r: data.missingness(r, expected) for r in runs})
        return result, None
    result.update(status="ok" if len(V["wids"]) == len(expected) else "partial",
                  windows={"expected": len(expected), "compared": len(V["wids"])},
                  groups=V["G"], missingness={r: data.missingness(r, expected) for r in runs})
    regimes = {}
    for regime in REGIMES:
        sel = regime_mask(V, regime)
        blk = block(V, C, F, sel, B, seed, with_p99=True)
        blk["by_category"] = {cat: block(V, C, F, sel & (V["cat"] == i), B, seed, with_p99=False)
                              for i, cat in enumerate(data.categories) if (sel & (V["cat"] == i)).any()}
        if (sel & V["cont"]).any():
            blk["model_native_continuation"] = block(V, C, F, sel & V["cont"], B, seed, with_p99=False)
        regimes[regime] = blk
    result["regimes"] = regimes
    all_mask = np.ones(V["pos"].shape[0], bool)
    result["by_position_bucket"] = {label: block(V, C, F, V["bucket"] == i, B, seed, with_p99=False)
                                    for i, label in enumerate(m.BUCKET_LABELS) if (V["bucket"] == i).any()}
    result["by_prefill_path"] = {label: block(V, C, F, all_mask & (V["path"] == i), B, seed, with_p99=False)
                                 for i, label in enumerate(m.PATHS) if (V["path"] == i).any()}
    if floor:
        result["verdict"] = pair_verdict(regimes, thresholds, sparse_links.get((arm_of(comp["ref"]),
                                                                                 arm_of(comp["cand"]))))
    private = private_sums(V, C, F)
    return result, private


def pair_verdict(regimes, thresholds, sparse_link) -> dict:
    out = {"thresholds": thresholds}
    dense = regimes["dense"]
    if dense.get("status") == "no_positions" or "excess" not in dense:
        out["dense"] = {"outcome": UNRESOLVED, "reason": "floor not estimable in the dense regime"}
    else:
        crit = criteria_of(dense, thresholds)
        out["dense"] = {"outcome": combine(crit.values()), "criteria": crit}
    sparse = regimes["sparse"]
    single = criteria_of(sparse, thresholds) if "excess" in sparse else None
    if sparse.get("status") == "no_positions":
        out["sparse"] = {"outcome": UNRESOLVED, "reason": "no sparse positions"}
    else:
        out["sparse"] = {"outcome": UNRESOLVED, "reason": SPARSE_REASON,
                         "single_execution_criteria_operational_disagreement": single}
        if sparse_link and sparse_link.get("status") == "ok":
            out["sparse"]["repeated_execution_diagnostics"] = sparse_link["source"]
        else:
            out["sparse"]["repeated_executions"] = (
                f"fewer than {MIN_EXECUTIONS} executions per arm (amendment 13)"
                + (f": {sparse_link['reason']}" if sparse_link else ""))
    single_all = criteria_of(regimes["all"], thresholds) if "excess" in regimes["all"] else None
    out["all"] = {"outcome": combine([out["dense"]["outcome"], out["sparse"]["outcome"]]),
                  "rule": "conjunction of the dense and sparse verdicts",
                  "single_execution_criteria": single_all}
    return out


def private_sums(V, C, F) -> dict:
    """Per-window, per-regime joint counts and sums (private; carries window ids)."""
    out = {"window_ids": np.asarray(V["wids"] or [""], dtype="U32")}
    n_w = len(V["wids"])
    for regime in REGIMES:
        joint = regime_mask(V, regime) & C["ok"] & (F["ok"] if F is not None else True)
        win = V["win"][joint]
        out[f"{regime}_count"] = np.bincount(win, minlength=n_w)
        for side, S in (("contrast", C), ("floor", F)):
            if S is None:
                continue
            out[f"{regime}_{side}_kl_sum"] = np.bincount(win, weights=S["kl"][joint], minlength=n_w)
            out[f"{regime}_{side}_top1_sum"] = np.bincount(win, weights=S["top1"][joint].astype(float), minlength=n_w)
            out[f"{regime}_{side}_cov_ref_sum"] = np.bincount(win, weights=S["cov_ref"][joint].astype(float),
                                                              minlength=n_w)
            out[f"{regime}_{side}_cov_cand_sum"] = np.bincount(win, weights=S["cov_cand"][joint].astype(float),
                                                               minlength=n_w)
    return out


def repeated_analysis(data, comp, B, seed, thresholds) -> tuple[dict, dict]:
    E = int(comp.get("executions", MIN_EXECUTIONS))
    subset = data.subset(comp["subset"])
    arms = {}
    for arm, candidates in comp["arms"].items():
        available = [r for r in candidates if data.run(r)["windows"] & set(subset)]
        arms[arm] = {"configured": candidates, "available": available,
                     "used": select_executions(candidates, set(available), E)}
    result = {"schema": "fidelity-campaign-repeated/1", "name": comp["name"], "type": "repeated",
              "subset": comp["subset"], "executions_per_arm": E, "min_executions": MIN_EXECUTIONS,
              "arms": arms, "pairs": {}}
    links = {}
    for ref_arm, cand_arm in comp["pairs"]:
        key = f"{cand_arm}-vs-{ref_arm}"
        used = (arms[ref_arm]["used"], arms[cand_arm]["used"])
        if used[0] is None or used[1] is None or E < MIN_EXECUTIONS:
            short = [a for a, u in ((ref_arm, used[0]), (cand_arm, used[1])) if u is None]
            reason = (f"fewer than {E} executions available for {', '.join(short)}" if short
                      else f"configured executions {E} < {MIN_EXECUTIONS}")
            result["pairs"][key] = {"status": "insufficient_executions", "reason": reason,
                                    "verdict": {r: UNRESOLVED for r in REGIMES}}
            links[(ref_arm, cand_arm)] = {"status": "insufficient", "reason": reason}
            continue
        run_keys = used[0] + used[1]
        common = set(subset)
        for run in run_keys:
            common &= data.run(run)["windows"]
        wids, parts = [], []
        started = time.time()
        for wid in sorted(common):
            rows = [data.load(run, wid) for run in run_keys]
            if any(r is None for r in rows):
                continue
            parts.append(repeated_window(rows, E))
            wids.append(wid)
        print(f"repeated {key}: windows={len(wids)} in {time.time() - started:.1f}s", flush=True)
        sizes = np.asarray([p["D"].shape[0] for p in parts], np.int64)
        V = data.labels(wids, sizes)
        A = {k: (np.concatenate([p[k] for p in parts]) if parts else np.zeros(0)) for k in parts[0]} if parts else None
        block_out = {"status": "ok", "runs": {ref_arm: used[0], cand_arm: used[1]},
                     "windows": {"expected": len(subset), "compared": len(wids)}, "groups": V["G"],
                     "regimes": {}}
        for regime in REGIMES:
            sel = regime_mask(V, regime) if A is not None else None
            blk = repeated_block(V, A, sel, E, B, seed, thresholds) if A is not None else {"status": "no_positions"}
            block_out["regimes"][regime] = blk
        block_out["verdict"] = {r: block_out["regimes"][r].get("outcome", UNRESOLVED) for r in REGIMES}
        result["pairs"][key] = block_out
        sparse = block_out["regimes"]["sparse"]
        if sparse.get("status") == "no_positions":
            links[(ref_arm, cand_arm)] = {"status": "insufficient", "reason": "no sparse positions"}
        else:
            links[(ref_arm, cand_arm)] = {"status": "ok", "source": f"{comp['name']}:{key}"}
    statuses = [v["status"] for v in result["pairs"].values()]
    result["status"] = "ok" if "ok" in statuses else "insufficient_executions"
    return result, links


def repeated_block(V, A, sel, E, B, seed, thresholds) -> dict:
    joint = sel & A["ok"]
    out = {"positions": int(sel.sum()), "positions_joint_valid": int(joint.sum()),
           "excluded_by_joint_mask": int(sel.sum() - joint.sum())}
    if not joint.any():
        out.update(status="no_positions", outcome=UNRESOLVED)
        return out
    g = Groups(V["grp"], joint, B, seed)
    out["groups"] = g.n
    noise = (A["WR"] + A["WC"]) / (2.0 * E)
    within_pair = (A["WR"] + A["WC"]) / 2.0
    series = {
        "kl_averaged": A["D"], "within_ref_pairwise": A["WR"], "within_cand_pairwise": A["WC"],
        "cross_pairwise": A["X"], "covered_mass_ref_averaged": A["covR"],
        "covered_mass_cand_averaged": A["covC"], "shared_cells": A["cells"],
    }
    for name, values in series.items():
        out[name] = g.ci(*g.ratio(*g.sums(values)))
    diagnostics = {"noise_expectation_averaged": noise, "averaged_minus_noise": A["D"] - noise,
                   "cross_minus_within_pairwise": A["X"] - within_pair}
    out["diagnostics"] = {name: g.ci(*g.ratio(*g.sums(v))) for name, v in diagnostics.items()}
    out["diagnostics"]["note"] = DIAGNOSTIC_NOTE
    agree = {"ref_ref": A["aRR"], "cand_cand": A["aCC"], "ref_cand": A["aRC"]}
    out["top1_agreement_pairwise"] = {k: g.ci(*g.ratio(*g.sums(v), 100.0)) for k, v in agree.items()}
    drop = (A["aRR"] + A["aCC"]) / 2.0 - A["aRC"]
    out["top1_drop_pairwise_pp"] = dict(g.ci(*g.ratio(*g.sums(drop), 100.0)), note="descriptive only")
    cp = Pooled(A["X"][joint], g.labels, g.n)
    wp = Pooled(within_pair[joint], g.labels, g.n)
    out["kl_p99_cross_minus_within"] = dict(
        g.ci(cp.percentile(99.0) - wp.percentile(99.0), cp.reps(99.0, g.M) - wp.reps(99.0, g.M)), note=P99_NOTE)
    out["criteria"] = {"kl_mean_excess": UNRESOLVED, "kl_p99_excess": UNRESOLVED, "top1_drop_pp": UNRESOLVED}
    out["outcome"] = UNRESOLVED
    out["reason"] = SPARSE_REASON
    return out


def k_analysis(data, comp, B, seed) -> dict:
    result = {"schema": "fidelity-campaign-k/1", "name": comp["name"], "type": "k_sensitivity",
              "ref": comp["ref"], "cand": comp["cand"], "K_low": comp["k_low"],
              "K_high": {"ref": data.run(comp["ref"])["K"], "cand": data.run(comp["cand"])["K"]}}
    missing = [r for r in (comp["ref"], comp["cand"]) if not data.run(r)["windows"]]
    if missing:
        result.update(status="missing_runs", missing_runs=missing)
        return result
    expected = data.subset(comp.get("subset"))
    high = data.pair(comp["ref"], comp["cand"])
    low = data.pair(comp["ref"], comp["cand"], int(comp["k_low"]))
    V, (H, L) = data.view(expected, [high, low])
    if not V["wids"]:
        result.update(status="empty", windows={"expected": len(expected), "compared": 0})
        return result
    result.update(status="ok" if len(V["wids"]) == len(expected) else "partial",
                  windows={"expected": len(expected), "compared": len(V["wids"])})
    result["regimes"] = {}
    for regime in REGIMES:
        blk = block(V, H, L, regime_mask(V, regime), B, seed, with_p99=False)
        if "contrast" in blk:
            blk["high_K"], blk["truncated_low_K"] = blk.pop("contrast"), blk.pop("floor")
            ex = blk.pop("excess")
            blk["kl_mean_high_minus_truncated"] = ex["kl_mean_excess"]
            blk["top1_agreement_truncated_minus_high_pp"] = ex["top1_drop_pp"]
        result["regimes"][regime] = blk
    result["note"] = ("same rows truncated to K_low: a larger K refines the partition, so the high-K coarse KL "
                      "is at least the truncated one; the gap shows how much the lower bound tightens")
    return result


def ladder_analysis(data, comp, comps, B, seed) -> dict:
    chain = comp["chain"]
    regime = comp.get("regime", "dense")
    floor = comps.get(comp.get("floor")) if comp.get("floor") else None
    expected = data.subset(comp.get("subset"))
    strict = list(comp.get("strict_negative_controls", []))
    controls = set(strict) | set(comp.get("negative_control_steps", []))
    result = {"schema": "fidelity-campaign-ladder/1", "name": comp["name"], "type": "ladder",
              "subset": comp.get("subset"), "regime": regime, "chain": chain, "floor": comp.get("floor"),
              "steps": [], "cumulative_from_first": [], "negative_controls": {}}
    floor_pair = None
    if floor and data.run(floor["ref"])["windows"] and data.run(floor["cand"])["windows"]:
        floor_pair = data.pair(floor["ref"], floor["cand"])

    def one(ref, cand, control=False):
        entry = {"ref": ref, "cand": cand}
        if control:
            entry["negative_control"] = True
        missing = [r for r in (ref, cand) if not data.run(r)["windows"]]
        if missing:
            entry.update(status="missing_runs", missing_runs=missing)
            return entry
        pairs = [data.pair(ref, cand)] + ([floor_pair] if floor_pair else [])
        V, arrays = data.view(expected, pairs)
        if not V["wids"]:
            entry.update(status="empty", windows={"expected": len(expected), "compared": 0})
            return entry
        entry.update(status="ok" if len(V["wids"]) == len(expected) else "partial",
                     windows={"expected": len(expected), "compared": len(V["wids"])})
        entry[regime] = block(V, arrays[0], arrays[1] if floor_pair else None, regime_mask(V, regime), B, seed,
                              with_p99=True)
        return entry

    for i in range(len(chain) - 1):
        result["steps"].append(one(chain[i], chain[i + 1], f"{chain[i]}->{chain[i + 1]}" in controls))
    for rung in chain[1:]:
        result["cumulative_from_first"].append(one(chain[0], rung))
    statuses = [s["status"] for s in result["steps"]]
    result["status"] = ("ok" if all(s == "ok" for s in statuses)
                        else "missing_runs" if all(s == "missing_runs" for s in statuses) else "partial")
    for step in sorted(controls):
        ref, cand = step.split("->")
        check = bit_identity(data, ref, cand, expected)
        result["negative_controls"][step] = dict(check, strict=step in strict)
    result["attribution"] = ladder_attribution([result["negative_controls"][step] for step in strict])
    return result


def ladder_attribution(gates: list) -> str:
    """Attribution is available only when every strict control passed bit identity on all configured
    windows (complete) with rows compared; withheld when any failed; pending otherwise."""
    if any(g.get("status") == "fail" for g in gates):
        return "withheld: negative control failed"
    if gates and all(g.get("status") == "pass" and g.get("complete") is True and g.get("rows_compared", 0) > 0
                     for g in gates):
        return "available"
    return "pending: every strict negative control must pass on all configured windows with rows compared"


def _bits(a):
    return a.view(np.dtype(f"u{a.dtype.itemsize}")) if a.dtype.kind == "f" else a


def bit_identity(data, ref, cand, expected, max_row=DENSE_MAX_CONDITIONING) -> dict:
    """Bitwise equality of lp_actual, topk_ids and topk_lp for rows 1..max_row on shared windows."""
    out = {"ref": ref, "cand": cand, "rows": f"1..{max_row}", "arrays": ["lp_actual", "topk_ids", "topk_lp"]}
    missing = [r for r in (ref, cand) if not data.run(r)["windows"]]
    if missing:
        out.update(status="missing_runs", missing_runs=missing)
        return out
    k_ref, k_cand = data.run(ref)["K"], data.run(cand)["K"]
    k = min(k_ref, k_cand)
    if k_ref != k_cand:
        out["K_note"] = f"K differs ({k_ref} vs {k_cand}); the first {k} columns are compared"
    shared = data.run(ref)["windows"] & data.run(cand)["windows"]
    counts = {"windows_expected": len(expected), "windows_compared": 0, "windows_identical": 0,
              "rows_compared": 0, "rows_differing": 0}
    first = None
    for wid in expected:
        if wid not in shared:
            continue
        r, c = data.load(ref, wid), data.load(cand, wid)
        if r is None or c is None:
            continue
        end = min(r["ids"].shape[0], max_row + 1)
        rows = slice(1, end)
        diff = (_bits(r["lp_actual"][rows]) != _bits(c["lp_actual"][rows]))
        diff |= (r["topk_ids"][rows, :k] != c["topk_ids"][rows, :k]).any(1)
        diff |= (_bits(r["topk_lp"][rows, :k]) != _bits(c["topk_lp"][rows, :k])).any(1)
        counts["windows_compared"] += 1
        counts["rows_compared"] += int(diff.size)
        counts["rows_differing"] += int(diff.sum())
        if diff.any():
            row = int(np.argmax(diff)) + 1
            first = row if first is None else min(first, row)
        else:
            counts["windows_identical"] += 1
    out.update(counts, first_differing_row=first)
    if counts["windows_compared"] == 0:
        out["status"] = "no_shared_windows"
    else:
        out["status"] = "fail" if counts["rows_differing"] else "pass"
        out["complete"] = counts["windows_compared"] == len(expected)
    return out


def identity_analysis(data, comp) -> dict:
    result = {"schema": "fidelity-campaign-identity/1", "name": comp["name"], "type": "bit_identity",
              "regime": f"dense (rows 1..{DENSE_MAX_CONDITIONING})", "checks": {}}
    for check in comp["checks"]:
        result["checks"][check["name"]] = bit_identity(data, check["ref"], check["cand"],
                                                       data.subset(check.get("subset")))
    statuses = {c["status"] for c in result["checks"].values()}
    result["status"] = "fail" if "fail" in statuses else "pass" if statuses == {"pass"} else "partial"
    return result


def mde_analysis(data, comp, comps, B, seed) -> dict:
    floor = comps[comp["floor"]]
    result = {"schema": "fidelity-campaign-mde/1", "name": comp["name"], "type": "mde",
              "basis": "2.8 x bootstrap SE (80% power, two-sided alpha 0.05) of paired reference-only contrasts",
              "floor": comp["floor"], "contrasts": []}
    if not (data.run(floor["ref"])["windows"] and data.run(floor["cand"])["windows"]):
        result.update(status="missing_runs", missing_runs=[floor["ref"], floor["cand"]])
        return result
    fpair = data.pair(floor["ref"], floor["cand"])
    expected = data.subset(floor.get("subset"))
    V, (F,) = data.view(expected, [fpair])
    entry = {"contrast": f"{floor['cand']} vs {floor['ref']}", "kind": "floor_mean", "regimes": {},
             "note": "SE of the floor mean itself; context only, not used for the governing MDE"}
    for regime in REGIMES:
        blk = block(V, F, None, regime_mask(V, regime), B, seed, with_p99=False)
        kl_se = blk.get("contrast", {}).get("kl", {}).get("mean_ci", {}).get("se")
        t_se = blk.get("contrast", {}).get("top1_agreement", {}).get("se")
        entry["regimes"][regime] = {"positions": blk["positions_joint_valid"], "groups": blk["groups"],
                                    "kl_mean_se": kl_se, "mde_kl_mean_nats": m.mde(kl_se),
                                    "mde_top1_pp": None if t_se is None else m.mde(t_se) * 100.0}
    result["floor_mean"] = entry
    for other in comp.get("null_candidates", []):
        entry = {"contrast": f"KL({floor['ref']}, {other}) - KL({floor['ref']}, {floor['cand']})",
                 "kind": "null_excess", "subset": comp.get("subset")}
        if not data.run(other)["windows"]:
            entry.update(status="missing_runs", missing_runs=[other])
            result["contrasts"].append(entry)
            continue
        V, (C, F) = data.view(data.subset(comp.get("subset")), [data.pair(floor["ref"], other), fpair])
        entry["regimes"] = {}
        for regime in REGIMES:
            blk = block(V, C, F, regime_mask(V, regime), B, seed, with_p99=False)
            ex = blk.get("excess", {})
            kl_se = (ex.get("kl_mean_excess") or {}).get("se")
            t_se = (ex.get("top1_drop_pp") or {}).get("se")
            entry["regimes"][regime] = {"positions": blk["positions_joint_valid"], "groups": blk["groups"],
                                        "null_excess": ex.get("kl_mean_excess"),
                                        "kl_mean_excess_se": kl_se, "mde_kl_mean_nats": m.mde(kl_se),
                                        "mde_top1_pp": m.mde(t_se)}
        result["contrasts"].append(entry)
    governing = {}
    for regime in REGIMES:
        values = [c["regimes"][regime]["mde_kl_mean_nats"] for c in result["contrasts"]
                  if c["kind"] == "null_excess" and "regimes" in c
                  and c["regimes"][regime]["mde_kl_mean_nats"] is not None]
        governing[regime] = max(values) if values else None
    result["governing_mde_kl_mean_nats"] = governing
    available = any(v is not None for v in governing.values())
    result["status"] = "ok" if available else "unavailable"
    if not available:
        result["governing_note"] = ("MDE unavailable: no paired reference-only null contrast "
                                    "(needs a further reference execution)")
    result["note"] = "a zero SE means the contrast is bit-identical in that regime; the MDE is then degenerate"
    return result


# ---------------------------------------------------------------- driver


def headline(result) -> dict:
    out = {"type": result["type"], "status": result.get("status")}
    if result["type"] == "pair":
        out.update(role=result["role"], ref=result["ref"], cand=result["cand"], floor=result["floor"])
    if result.get("missing_runs"):
        out["missing_runs"] = result["missing_runs"]
    if result["type"] == "pair" and result.get("regimes"):
        out.update(windows=result["windows"], groups=result["groups"])
        out["regimes"] = {}
        for regime, blk in result["regimes"].items():
            if "contrast" not in blk:
                out["regimes"][regime] = {"status": blk.get("status")}
                continue
            c = blk["contrast"]
            row = {"positions": blk["positions_joint_valid"], "kl_mean": c["kl"]["mean_ci"],
                   "kl_p99": c["kl"].get("p99"), "top1_agreement": c["top1_agreement"],
                   "covered_mass_ref_mean": c["covered_mass_ref"]["mean"],
                   "covered_mass_cand_mean": c["covered_mass_cand"]["mean"],
                   "topk_overlap": c["topk_overlap"]}
            if "excess" in blk:
                row["excess"] = blk["excess"]
            out["regimes"][regime] = row
        if "verdict" in result:
            out["verdict"] = {r: result["verdict"][r]["outcome"] for r in REGIMES}
    elif result["type"] == "repeated":
        out["pairs"] = {k: {"status": v["status"], "verdict": v.get("verdict")} for k, v in result["pairs"].items()}
    elif result["type"] == "mde":
        out["governing_mde_kl_mean_nats"] = result.get("governing_mde_kl_mean_nats")
    elif result["type"] == "bit_identity":
        out["checks"] = {k: {"status": v["status"], "rows_differing": v.get("rows_differing"),
                             "rows_compared": v.get("rows_compared")} for k, v in result["checks"].items()}
    elif result["type"] == "ladder":
        out["attribution"] = result.get("attribution")
        out["steps"] = [{"ref": s["ref"], "cand": s["cand"], "status": s["status"],
                         "kl_mean": (s.get(result["regime"], {}).get("contrast", {}).get("kl", {}).get("mean_ci"))}
                        for s in result["steps"]]
    elif result["type"] == "k_sensitivity" and result.get("regimes"):
        out["kl_mean_high_minus_truncated"] = {r: b.get("kl_mean_high_minus_truncated")
                                               for r, b in result["regimes"].items()}
    return out


def analyze(args) -> dict:
    if np is None:
        raise SystemExit("analyze_campaign.py requires numpy (use data/fidelity/.venv/bin/python)")
    import analyze as legacy  # JSON cleaning and atomic public writes
    cfg = load_config(args.config)
    boot = cfg.get("bootstrap", {})
    B = args.bootstrap or int(boot.get("B", m.BOOT_B))
    seed = int(boot.get("seed", m.BOOT_SEED))
    thresholds = dict(THRESHOLDS, **cfg.get("thresholds", {}))
    comps = {c["name"]: c for c in cfg["comparisons"]}
    data = Data(args)
    started = time.time()
    results, links, private = {}, {}, {}
    order = sorted(cfg["comparisons"], key=lambda c: ["repeated", "pair", "k_sensitivity", "ladder", "mde",
                                                      "bit_identity"].index(c["type"]))
    for comp in order:
        kind = comp["type"]
        if kind == "repeated":
            results[comp["name"]], found = repeated_analysis(data, comp, B, seed, thresholds)
            links.update(found)
        elif kind == "pair":
            results[comp["name"]], sums = pair_analysis(data, comp, comps, B, seed, thresholds, links)
            if sums is not None:
                private[comp["name"]] = sums
        elif kind == "k_sensitivity":
            results[comp["name"]] = k_analysis(data, comp, B, seed)
        elif kind == "ladder":
            results[comp["name"]] = ladder_analysis(data, comp, comps, B, seed)
        elif kind == "bit_identity":
            results[comp["name"]] = identity_analysis(data, comp)
        else:
            results[comp["name"]] = mde_analysis(data, comp, comps, B, seed)
        print(f"{comp['name']}: {results[comp['name']].get('status')}", flush=True)
    for name, sums in private.items():
        path = args.private_out / f"{name}.npz"
        fio._atomic(path, lambda handle, s=sums: np.savez(handle, **s), mode=0o600)
    fio.write_json_atomic(args.private_out / "groups.json", data.groups, mode=0o600)
    inventory = {}
    for key, info in sorted(data.runs.items()):
        inventory[key] = {"K": info["K"], "windows_ok": len(info["windows"]),
                          "missingness_vs_manifest": data.missingness(key, [e["id"] for e in data.entries])}
    for name, result in results.items():
        legacy.write_public(args.public_out / f"{name}.json", result)
    summary = {
        "schema": "fidelity-campaign-summary/1", "generated_utc": fio.utc_now(), "harness_git": fio.git_identity(),
        "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
        "bootstrap": {"B": B, "seed": seed, "unit": "source group", "estimator": "token-weighted ratio",
                      "ci": "percentile 95% two-sided; one-sided 95% bounds lb95/ub95"},
        "regimes": {"dense": f"conditioning tokens <= {DENSE_MAX_CONDITIONING}",
                    "sparse": f"conditioning tokens > {DENSE_MAX_CONDITIONING}", "all": "every scored row"},
        "thresholds": thresholds, "bound": "coarse KL is a lower bound on the full-vocabulary KL",
        "grouping": {"rules": data.group_rules, "groups": len(set(data.groups.values())),
                     "windows": len(data.entries)},
        "runs": inventory, "method": METHOD,
        "comparisons": {name: headline(r) for name, r in results.items()},
        "elapsed_s": round(time.time() - started, 1),
    }
    legacy.write_public(args.public_out / "summary.json", summary)
    return {"results": results, "summary": summary}


# ---------------------------------------------------------------- selftest


def _log_softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=1, keepdims=True))


def _write_run(root, arm, run, k, windows, tokens, arm_sd, arm_seed, exec_seed, sparse_sd, only=None,
               actual_overrides=None):
    """Synthetic run: shared base logits, a fixed per-arm shift and per-execution noise on sparse rows."""
    run_dir = root / arm / run
    fio.write_json_atomic(run_dir / "run.json", {"arm": arm, "run": run, "K": k, "kind": "prompt"})
    for index, entry in enumerate(windows):
        if only is not None and entry["id"] not in only:
            continue
        ids = np.asarray(tokens[entry["id"]], np.int64)
        n = len(ids)
        logits = np.random.default_rng(1000 + index).normal(0, 3.0, (n, 300))
        if arm_sd:
            logits = logits + np.random.default_rng(arm_seed * 7919 + index).normal(0, arm_sd, logits.shape)
        if sparse_sd:
            noise = np.random.default_rng(exec_seed * 104729 + index).normal(0, sparse_sd, logits.shape)
            noise[:DENSE_MAX_CONDITIONING + 1] = 0.0
            logits = logits + noise
        logp = _log_softmax(logits)
        top_ids = np.argsort(-logp, axis=1, kind="stable")[:, :k].astype(np.int32)
        top_lp = np.take_along_axis(logp, top_ids, 1).astype(np.float32)
        lp_actual = logp[np.arange(n), ids].astype(np.float32)
        rank = (logp > logp[np.arange(n), ids][:, None]).sum(1) + 1
        top_ids[0], top_lp[0], lp_actual[0], rank[0] = -1, -np.inf, np.nan, -1
        for row, value in (actual_overrides or {}).get(entry["id"], []):
            lp_actual[row] = value
        fio.save_npz(run_dir / "prompt" / f"{entry['id']}.npz",
                     {"ids": ids, "lp_actual": lp_actual, "rank_actual": rank, "topk_ids": top_ids,
                      "topk_lp": top_lp}, fio.PROMPT_ARRAYS)
        fio.write_json_atomic(run_dir / "prompt" / f"{entry['id']}.json", {"status": "ok", "K": k})


def _selftest_corpus(tmp: Path):
    rng = np.random.default_rng(20260927)
    spec = [("agentic_code", "claude_code", 3000), ("agentic_code", "claude_code", 2600),
            ("agentic_code", "omp", 1500), ("structured_json", "omp", 2300), ("italian_chat", "synthetic_it", 2700),
            ("italian_chat", "synthetic_it", 2200), ("model_native", "r0_native", 2500),
            ("model_native", "r0_native", 900), ("long_context", "claude_code", 4200),
            ("agentic_code", "omp", 3300), ("structured_json", "claude_code", 2100), ("agentic_code", "omp", 2900)]
    windows = [{"id": f"w{i + 1:04d}", "category": c, "source": s, "project": f"proj-secret-{i % 3}"}
               for i, (c, s, _) in enumerate(spec)]
    tokens = {w["id"]: rng.integers(0, 300, n).tolist() for w, (_, _, n) in zip(windows, spec)}
    corpus = tmp / "corpus"
    rows = []
    for w in windows:
        data = np.asarray(tokens[w["id"]], "<u4").tobytes()
        (corpus / "tokens").mkdir(parents=True, exist_ok=True)
        (corpus / "tokens" / f"{w['id']}.u32").write_bytes(data)
        rows.append(dict(w, n_tokens=len(tokens[w["id"]]), path=f"tokens/{w['id']}.u32",
                         sha256=hashlib.sha256(data).hexdigest()))
    fio.write_json_atomic(corpus / "manifest.json", {"schema": "fidelity-corpus/1", "frozen": True, "windows": rows,
                                                     "global_sha256": fio.global_sha256(rows)})
    meta = {"w0001": {"segment": "sessA:0"}, "w0002": {"segment": "sessA:1"}, "w0003": {"segment": "sessB:0"},
            "w0004": {"segment": "sessB:0"}, "w0005": {"file": "conv01.json"}, "w0006": {"file": "conv01.json"},
            "w0009": {"segment": "sessC:0"}, "w0010": {"segment": "sessD:0"}, "w0011": {"segment": "sessE:2"},
            "w0012": {"segment": "sessF:0"}}
    fio.write_json_atomic(corpus / "windows-meta.json", meta)
    fio.write_json_atomic(corpus / "native-provenance.json", {"windows": {
        "w0007": {"decode_id": "d001", "prompt_tokens": 1800, "gen_tokens": 700},
        "w0008": {"decode_id": "n001", "prompt_tokens": 300, "gen_tokens": 600}}})
    fio.write_json_atomic(corpus / "decode-meta.json", {"d001": {"segment": "sessA:3"}})
    (corpus / "subsets").mkdir(exist_ok=True)
    (corpus / "subsets" / "crossboot.txt").write_text("".join(f"{w['id']}\n" for w in windows[:10]))
    (corpus / "subsets" / "ladder.txt").write_text("".join(f"{w['id']}\n" for w in windows[::2]))
    (corpus / "subsets" / "k100.txt").write_text("".join(f"{w['id']}\n" for w in windows[1::2]))
    return corpus, windows, tokens


def selftest_config() -> dict:
    return {"schema": CONFIG_SCHEMA, "bootstrap": {"B": 2000, "seed": m.BOOT_SEED}, "comparisons": [
        {"name": "floor-r0", "type": "pair", "ref": "R0/a", "cand": "R0/b"},
        {"name": "cm-vs-r0", "type": "pair", "ref": "R0/a", "cand": "Cm/a", "floor": "floor-r0",
         "role": "pre-registered"},
        {"name": "q-vs-r0", "type": "pair", "ref": "R0/a", "cand": "Q/a", "floor": "floor-r0"},
        {"name": "n-vs-r0", "type": "pair", "ref": "R0/a", "cand": "N/a", "floor": "floor-r0"},
        {"name": "cpre-vs-r0", "type": "pair", "ref": "R0/a", "cand": "Cpre/a", "floor": "floor-r0"},
        {"name": "k-cm-vs-r0", "type": "k_sensitivity", "ref": "R0/k50", "cand": "Cm/k50", "k_low": 20,
         "subset": "subsets/k100.txt"},
        {"name": "ladder", "type": "ladder", "subset": "subsets/ladder.txt", "regime": "dense",
         "floor": "floor-r0", "chain": ["R0/a", "L1/ladder", "Cpre/a", "L2/ladder", "Cm/a"],
         "strict_negative_controls": ["Cpre/a->L2/ladder"], "negative_control_steps": ["L2/ladder->Cm/a"]},
        {"name": "repeated", "type": "repeated", "subset": "subsets/crossboot.txt", "executions": 3,
         "arms": {"R0": ["R0/a", "R0/rep2", "R0/rep3"], "Cm": ["Cm/a", "Cm/rep2", "Cm/rep3"],
                  "Q": ["Q/a", "Q/rep2", "Q/rep3"], "Cpre": ["Cpre/a", "Cpre/rep2", "Cpre/rep3"]},
         "pairs": [["R0", "Cm"], ["R0", "Q"], ["R0", "Cpre"]]},
        {"name": "mde", "type": "mde", "floor": "floor-r0", "subset": "subsets/crossboot.txt",
         "null_candidates": ["R0/rep2", "R0/rep3", "R0/missing"]},
        {"name": "identity", "type": "bit_identity", "checks": [
            {"name": "r0-a-vs-b", "ref": "R0/a", "cand": "R0/b"},
            {"name": "cm-a-vs-rep2", "ref": "Cm/a", "cand": "Cm/rep2", "subset": "subsets/crossboot.txt"},
            {"name": "cm-vs-r0", "ref": "R0/a", "cand": "Cm/a"},
            {"name": "absent", "ref": "R0/a", "cand": "X/a"}]},
    ]}


def selftest() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="fidelity-campaign-selftest-"))
    try:
        corpus, windows, tokens = _selftest_corpus(tmp)
        raw = tmp / "raw"
        cross = {w["id"] for w in windows[:10]}
        k100 = {w["id"] for w in windows[1::2]}
        sd = 0.6
        _write_run(raw, "R0", "a", 20, windows, tokens, 0, 0, 1, sd)
        _write_run(raw, "R0", "b", 20, windows, tokens, 0, 0, 2, sd)
        _write_run(raw, "R0", "rep2", 20, windows, tokens, 0, 0, 3, sd, only=cross)
        _write_run(raw, "R0", "rep3", 20, windows, tokens, 0, 0, 4, sd, only=cross)
        # Nonfinite actual-token scores in Cm/a: three genuine zeros (-inf) and one missing (NaN).
        # A +inf actual-token score (row 30) invalidates its KL row and must still be counted.
        overrides = {"w0003": [(10, -np.inf), (11, -np.inf), (12, -np.inf), (20, np.nan), (30, np.inf)]}
        _write_run(raw, "Cm", "a", 20, windows, tokens, 0.3, 7, 11, sd, actual_overrides=overrides)
        _write_run(raw, "Cm", "rep2", 20, windows, tokens, 0.3, 7, 12, sd, only=cross)
        _write_run(raw, "Cm", "rep3", 20, windows, tokens, 0.3, 7, 13, sd, only=cross)
        # Q: the same model as R0 with independent executions (no systematic difference).
        for run, s in (("a", 21), ("rep2", 22), ("rep3", 23)):
            _write_run(raw, "Q", run, 20, windows, tokens, 0, 0, s, sd, only=None if run == "a" else cross)
        _write_run(raw, "Cpre", "a", 20, windows, tokens, 0.2, 9, 31, sd)
        _write_run(raw, "N", "a", 20, windows, tokens, 0.5, 5, 41, sd, only={w["id"] for w in windows[:5]})
        _write_run(raw, "R0", "k50", 50, windows, tokens, 0, 0, 1, sd, only=k100)
        _write_run(raw, "Cm", "k50", 50, windows, tokens, 0.3, 7, 11, sd, only=k100)
        _write_run(raw, "L1", "ladder", 20, windows, tokens, 0.1, 3, 51, sd, only={w["id"] for w in windows[::2]})
        # L2: the Cpre recipe on another execution (dense rows bit-identical to Cpre/a).
        _write_run(raw, "L2", "ladder", 20, windows, tokens, 0.2, 9, 61, sd, only={w["id"] for w in windows[::2]})
        (tmp / "config.json").write_text(json.dumps(selftest_config()))
        args = parse_args(["--config", str(tmp / "config.json"), "--raw-root", str(raw),
                           "--manifest", str(corpus / "manifest.json"), "--private-out", str(tmp / "private"),
                           "--public-out", str(tmp / "public"), "--bootstrap", "300"])
        out = analyze(args)
        res = out["results"]
        checks = []

        def check(name, ok):
            checks.append((name, bool(ok)))

        floor = res["floor-r0"]["regimes"]
        check("floor dense KL is exactly zero", floor["dense"]["contrast"]["kl"]["mean"] == 0.0
              and floor["dense"]["contrast"]["kl"]["max"] == 0.0)
        check("floor sparse KL positive", floor["sparse"]["contrast"]["kl"]["mean"] > 0)
        cm = res["cm-vs-r0"]
        check("regimes partition positions", cm["regimes"]["all"]["positions"]
              == cm["regimes"]["dense"]["positions"] + cm["regimes"]["sparse"]["positions"])
        check("all positions scored", cm["regimes"]["all"]["positions"] == sum(len(t) - 1 for t in tokens.values()))
        dense_ex = cm["regimes"]["dense"]["excess"]
        check("Cm dense excess equals its KL (zero floor)",
              abs(dense_ex["kl_mean_excess"]["estimate"] - cm["regimes"]["dense"]["contrast"]["kl"]["mean"]) < 1e-12)
        check("Cm dense excess CI above zero", dense_ex["kl_mean_excess"]["ci_low"] > 0)
        check("Cm dense verdict exceeds margin", cm["verdict"]["dense"]["outcome"] == EXCEEDS)
        check("coverage and overlap reported", 0 < cm["regimes"]["dense"]["contrast"]["covered_mass_ref"]["mean"] <= 1
              and cm["regimes"]["dense"]["contrast"]["topk_overlap"]["p10"] is not None)
        check("one-sided bounds inside the two-sided interval",
              dense_ex["kl_mean_excess"]["ci_low"] <= dense_ex["kl_mean_excess"]["lb95"]
              <= dense_ex["kl_mean_excess"]["ub95"] <= dense_ex["kl_mean_excess"]["ci_high"])
        check("p99 excess with bounds", cm["regimes"]["sparse"]["excess"]["kl_p99_excess"]["ub95"] is not None)
        check("sparse verdict unresolved even with repeated executions (no validated estimator)",
              cm["verdict"]["sparse"]["outcome"] == UNRESOLVED and cm["verdict"]["sparse"]["reason"] == SPARSE_REASON
              and cm["verdict"]["sparse"]["repeated_execution_diagnostics"] == "repeated:Cm-vs-R0")
        check("Cpre sparse unresolved, fewer than three executions reported",
              res["cpre-vs-r0"]["verdict"]["sparse"]["outcome"] == UNRESOLVED
              and "executions" in res["cpre-vs-r0"]["verdict"]["sparse"]["repeated_executions"])
        all_blk = cm["regimes"]["all"]["contrast"]
        check("nonfinite actual-token scores counted per side",
              cm["regimes"]["all"]["actual_token_nonfinite"]["contrast"]["cand"]
              == {"missing": 1, "zero_probability": 3, "other_nonfinite": 1, "infinite_nll": 3}
              and cm["regimes"]["all"]["actual_token_nonfinite"]["contrast"]["ref"]["zero_probability"] == 0
              and cm["regimes"]["all"]["excluded_by_joint_mask"] >= 1
              and cm["missingness"]["Cm/a"]["other_nonfinite_actual"]["dense"] == 1
              and all_blk["delta_nll"]["excluded_nonfinite"] == 4 and all_blk["delta_nll"]["finite_subset"]
              and cm["missingness"]["Cm/a"]["zero_probability_actual"]["dense"] == 3
              and cm["missingness"]["Cm/a"]["missing_actual"]["dense"] == 1)
        single = cm["regimes"]["all"]["by_category"]["long_context"]
        check("one group: point estimate kept, bounds unavailable",
              single["groups"] == 1 and single["contrast"]["kl"]["mean_ci"]["estimate"] > 0
              and single["contrast"]["kl"]["mean_ci"]["ub95"] is None
              and single["excess"]["kl_mean_excess"]["ub95"] is None)
        check("overall is the conjunction of regimes",
              res["cpre-vs-r0"]["verdict"]["all"]["outcome"] == EXCEEDS
              and res["q-vs-r0"]["verdict"]["dense"]["outcome"] == WITHIN
              and res["q-vs-r0"]["verdict"]["all"]["outcome"]
              == combine([WITHIN, res["q-vs-r0"]["verdict"]["sparse"]["outcome"]]))
        check("groups fewer than windows", cm["groups"] < cm["windows"]["compared"])
        check("model-native continuation block", "model_native_continuation" in cm["regimes"]["all"]
              and cm["regimes"]["all"]["model_native_continuation"]["positions"] == (2500 - 1800) + (900 - 300))
        check("categories per regime", set(cm["regimes"]["dense"]["by_category"]) == {w["category"] for w in windows})
        check("partial run flagged", res["n-vs-r0"]["status"] == "partial"
              and res["n-vs-r0"]["windows"]["compared"] == 5)
        check("missingness per run and regime",
              res["n-vs-r0"]["missingness"]["N/a"]["rows_in_missing_windows"]["sparse"] > 0)
        rep = res["repeated"]["pairs"]
        cm_rep, q_rep = rep["Cm-vs-R0"]["regimes"]["sparse"], rep["Q-vs-R0"]["regimes"]["sparse"]
        check("repeated: averaged KL and labelled diagnostics reported",
              cm_rep["kl_averaged"]["estimate"] > q_rep["kl_averaged"]["estimate"] > 0
              and cm_rep["diagnostics"]["cross_minus_within_pairwise"]["ci_low"] > 0
              and "never used for verdicts" in cm_rep["diagnostics"]["note"])
        check("repeated: every regime and criterion unresolved",
              all(rep[k]["regimes"][r]["outcome"] == UNRESOLVED and
                  set(rep[k]["regimes"][r]["criteria"].values()) == {UNRESOLVED}
                  for k in ("Cm-vs-R0", "Q-vs-R0") for r in REGIMES))
        check("repeated: insufficient executions reported", rep["Cpre-vs-R0"]["status"] == "insufficient_executions")
        check("K sensitivity: higher K tightens the bound",
              res["k-cm-vs-r0"]["regimes"]["all"]["kl_mean_high_minus_truncated"]["estimate"] >= -1e-12)
        ladder = res["ladder"]
        check("ladder steps", [s["status"] for s in ladder["steps"]] == ["ok"] * 4 and "dense" in ladder["steps"][0])
        controls = ladder["negative_controls"]
        check("ladder: strict negative control bit-identical, attribution available",
              controls["Cpre/a->L2/ladder"]["status"] == "pass" and controls["Cpre/a->L2/ladder"]["strict"]
              and controls["Cpre/a->L2/ladder"]["rows_compared"] > 0 and ladder["attribution"] == "available")
        check("ladder: failing non-strict control reported",
              controls["L2/ladder->Cm/a"]["status"] == "fail" and not controls["L2/ladder->Cm/a"]["strict"])
        ident = res["identity"]["checks"]
        check("bit identity: same-recipe dense repeat passes", ident["r0-a-vs-b"]["status"] == "pass"
              and ident["r0-a-vs-b"]["windows_identical"] == len(windows))
        check("bit identity: altered rows found exactly",
              ident["cm-a-vs-rep2"]["status"] == "fail" and ident["cm-a-vs-rep2"]["rows_differing"] == 5
              and ident["cm-a-vs-rep2"]["first_differing_row"] == 10
              and ident["cm-vs-r0"]["first_differing_row"] == 1 and ident["absent"]["status"] == "missing_runs")
        mde = res["mde"]
        check("MDE governed by paired null contrasts only, dense degenerate",
              mde["governing_mde_kl_mean_nats"]["dense"] == 0.0 and mde["governing_mde_kl_mean_nats"]["sparse"] > 0
              and all(c["kind"] == "null_excess" for c in mde["contrasts"]) and "floor_mean" in mde)
        check("MDE missing candidate flagged", mde["contrasts"][-1]["status"] == "missing_runs")
        cfg = selftest_config()
        cfg["comparisons"] += [
            {"name": "x-vs-r0", "type": "pair", "ref": "R0/a", "cand": "X/a", "floor": "floor-r0"},
            {"name": "ladder-withheld", "type": "ladder", "subset": "subsets/ladder.txt", "floor": "floor-r0",
             "chain": ["Cpre/a", "L2/ladder", "Cm/a"], "strict_negative_controls": ["L2/ladder->Cm/a"]},
            {"name": "mde-unavailable", "type": "mde", "floor": "floor-r0", "null_candidates": ["R0/missing"]},
            {"name": "empty-pair", "type": "pair", "ref": "L1/ladder", "cand": "Cm/k50", "floor": "floor-r0"}]
        (tmp / "config2.json").write_text(json.dumps(cfg))
        res2 = analyze(parse_args(["--config", str(tmp / "config2.json"), "--raw-root", str(raw),
                                   "--manifest", str(corpus / "manifest.json"),
                                   "--private-out", str(tmp / "private2"), "--public-out", str(tmp / "public2"),
                                   "--bootstrap", "50"]))["results"]
        missing = res2["x-vs-r0"]
        check("missing run -> missing_runs", missing["status"] == "missing_runs" and missing["missing_runs"] == ["X/a"])
        check("comparison without shared windows reports empty cleanly",
              res2["empty-pair"]["status"] == "empty" and res2["empty-pair"]["windows"]["compared"] == 0)
        check("ladder attribution withheld when a strict control fails",
              res2["ladder-withheld"]["attribution"] == "withheld: negative control failed")
        check("MDE unavailable without a paired null contrast",
              res2["mde-unavailable"]["status"] == "unavailable"
              and set(res2["mde-unavailable"]["governing_mde_kl_mean_nats"].values()) == {None}
              and res2["mde-unavailable"]["floor_mean"]["regimes"]["sparse"]["kl_mean_se"] > 0)
        public = "".join(p.read_text() for p in (tmp / "public").glob("*.json"))
        check("public JSON has no ids, projects, groups or salts",
              not any(s in public for s in ('"ids"', '"topk_ids":', '"topk_lp":', "salt\"", "proj-secret", "sessA",
                                            "conv01", "w0001")))
        check("private sums written", (tmp / "private" / "cm-vs-r0.npz").exists()
              and oct((tmp / "private" / "cm-vs-r0.npz").stat().st_mode & 0o777) == "0o600")
        for name, ok in checks:
            print(f"selftest: {'PASS' if ok else 'FAIL'} {name}")
        failed = [name for name, ok in checks if not ok]
        print(f"selftest: {'PASS' if not failed else 'FAIL'} ({len(checks) - len(failed)}/{len(checks)})")
        return 1 if failed else 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, help="campaign configuration JSON")
    p.add_argument("--raw-root", type=Path, default=REPO / "data/fidelity/raw")
    p.add_argument("--manifest", type=Path, default=REPO / "data/fidelity/corpus/manifest.json")
    p.add_argument("--private-out", type=Path, default=REPO / "data/fidelity/metrics-v2")
    p.add_argument("--public-out", type=Path, default=REPO / "docs/fidelity/metrics-v2")
    p.add_argument("--bootstrap", type=int, default=None, help="override the configured replicates")
    p.add_argument("--selftest", action="store_true", help="run the pipeline on synthetic data")
    args = p.parse_args(argv)
    if not args.selftest and not args.config:
        p.error("--config is required unless --selftest")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.selftest:
        return selftest()
    analyze(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
