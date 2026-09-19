# SPDX-License-Identifier: Apache-2.0
# Derived from the vLLM project (Apache-2.0): AdaptiveKScheduler subclasses
# vllm.v1.core.sched.async_scheduler.AsyncScheduler and re-implements its methods.
# Modifications (c) 2026 Jacopo Nardiello, MIT-licensed where separable.
# See CREDITS.md.
"""Experimental adaptive scheduler that respects R10's effective draft budget.

Derived from scripts/node/patches/adaptive_k_scheduler.py, whose bytes remain frozen.
Policy, hysteresis, candidate filters and the three-slot async observation ring are
unchanged. Only the final placeholder length is capped when the opt-in flag is on:
min(adaptive k, engine maximum, SchedulerOutput's resolved draft budget).

R10's AsyncScheduler and V2 worker both call
SchedulerOutput.resolve_num_spec_tokens_to_schedule(engine_maximum). Use the SAME
output here: its drafts are consumed by the next verification step. Do not infer
that budget from a future batch, the number of eligible decode requests, or the
current output's scheduled_spec_decode_tokens (which verify older drafts).

With the existing [[1,1,5],[2,6,3]] table the effective budget is 5 for one
scheduled request and 3 for two through six, including mixed prefill/decode
batches. The cap never increases k or modifies policy state. A None budget is
resolved to the engine maximum by R10; zero stays zero.

Installation: replace only the /opt/tp4/adaptive_k_scheduler.py mount and set
VLLM_ADAPTIVE_K_RESPECT_DRAFT_BUDGET=1 via this experiment's delta.env. The flag
defaults to 0, retaining the original scheduler's sizing. All other VLLM_ADAPTIVE_K_*
knobs retain their defaults and meanings; see the frozen source and the experiment
README. Aggregate cap counters log on the first limit and at the existing policy
log interval. Exceptions retain the original one-time fallback to AsyncScheduler;
any fallback invalidates experiment activation and must stop measurements.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

__all__ = ["AdaptiveKConfig", "AdaptiveKPolicy", "AdaptiveKScheduler", "should_observe",
           "placeholder_len", "DraftedRing", "filter_candidates", "assign_lengths"]

_MODES = ("per-request", "batch-uniform")
_SIGNALS = ("pos", "mean")


def _env_int(env, name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_float(env, name: str, default: float) -> float:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


@dataclass(frozen=True)
class AdaptiveKConfig:
    enabled: bool = True
    k_lo: int = 3
    k_hi: int = 5
    up: float = 0.58
    down: float = 0.42
    alpha: float = 0.15
    seed: float = 1.0
    mode: str = "per-request"
    signal: str = "pos"
    log_every: int = 200
    respect_draft_budget: bool = False

    def __post_init__(self) -> None:
        if self.k_lo < 1 or self.k_hi < self.k_lo:
            raise ValueError(f"adaptive-k: need 1 <= k_lo <= k_hi, got {self.k_lo}/{self.k_hi}")
        if not (0.0 <= self.down <= self.up <= 1.0):
            raise ValueError(f"adaptive-k: need 0 <= down <= up <= 1, got {self.down}/{self.up}")
        if not (0.0 < self.alpha <= 1.0):
            raise ValueError(f"adaptive-k: need 0 < alpha <= 1, got {self.alpha}")
        if not (0.0 <= self.seed <= 1.0):
            raise ValueError(f"adaptive-k: need 0 <= seed <= 1, got {self.seed}")
        if self.mode not in _MODES:
            raise ValueError(f"adaptive-k: mode must be one of {_MODES}, got {self.mode!r}")
        if self.signal not in _SIGNALS:
            raise ValueError(f"adaptive-k: signal must be one of {_SIGNALS}, got {self.signal!r}")
        if self.log_every < 0:
            raise ValueError("adaptive-k: log_every must be >= 0")

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> "AdaptiveKConfig":
        env = os.environ if environ is None else environ
        return cls(
            enabled=_env_int(env, "VLLM_ADAPTIVE_K_ENABLE", 1) != 0,
            k_lo=_env_int(env, "VLLM_ADAPTIVE_K_LO", 3),
            k_hi=_env_int(env, "VLLM_ADAPTIVE_K_HI", 5),
            up=_env_float(env, "VLLM_ADAPTIVE_K_UP", 0.58),
            down=_env_float(env, "VLLM_ADAPTIVE_K_DOWN", 0.42),
            alpha=_env_float(env, "VLLM_ADAPTIVE_K_ALPHA", 0.15),
            seed=_env_float(env, "VLLM_ADAPTIVE_K_SEED", 1.0),
            mode=(env.get("VLLM_ADAPTIVE_K_MODE") or "per-request").strip() or "per-request",
            signal=(env.get("VLLM_ADAPTIVE_K_SIGNAL") or "pos").strip() or "pos",
            log_every=_env_int(env, "VLLM_ADAPTIVE_K_LOG_EVERY", 200),
            respect_draft_budget=_env_int(env, "VLLM_ADAPTIVE_K_RESPECT_DRAFT_BUDGET", 0) != 0,
        )

    def describe(self) -> str:
        return (
            f"enabled={int(self.enabled)} k_lo={self.k_lo} k_hi={self.k_hi} up={self.up} "
            f"down={self.down} alpha={self.alpha} seed={self.seed} mode={self.mode} "
            f"signal={self.signal} log_every={self.log_every} "
            f"respect_draft_budget={int(self.respect_draft_budget)}"
        )


@dataclass
class _ReqState:
    ema: float
    k: int


@dataclass
class AdaptiveKPolicy:
    """Per-request EMA of low-position acceptance -> draft length with hysteresis."""

    cfg: AdaptiveKConfig
    _state: dict[str, _ReqState] = field(default_factory=dict)
    counters: dict[str, int] = field(
        default_factory=lambda: {
            "observations": 0,
            "decisions_lo": 0,
            "decisions_hi": 0,
            "switches": 0,
            "steps_uniform_lo": 0,
            "steps_uniform_hi": 0,
        }
    )

    # -- state -------------------------------------------------------------
    def _get(self, req_id: str) -> _ReqState:
        st = self._state.get(req_id)
        if st is None:
            st = _ReqState(ema=self.cfg.seed, k=self._k_for(self.cfg.seed, current=None))
            self._state[req_id] = st
        return st

    def _k_for(self, ema: float, current: int | None) -> int:
        if ema >= self.cfg.up:
            return self.cfg.k_hi
        if ema <= self.cfg.down:
            return self.cfg.k_lo
        return self.cfg.k_hi if current is None else current

    def ema(self, req_id: str) -> float | None:
        st = self._state.get(req_id)
        return None if st is None else st.ema

    def tracked(self) -> int:
        return len(self._state)

    # -- signal ------------------------------------------------------------
    def signal(self, num_accepted: int, num_draft: int) -> float:
        """Per-step signal in [0, 1] from the accepted-prefix length.

        Accepted tokens are a prefix (rejection sampling stops at the first
        rejection), so both signals look only at positions 1..k_lo, which are
        drafted whatever k was actually verified. ``pos`` (default): 1.0 when
        the whole prefix of length k_lo was accepted, else 0.0 — its mean is
        the marginal P(position k_lo accepted). ``mean``: the average of the
        first k_lo marginals, ``min(num_accepted, k_lo) / k_lo``. Drafts
        shorter than k_lo (e.g. a chunk boundary) use their own length.
        """
        denom = min(self.cfg.k_lo, max(num_draft, 1))
        accepted = min(max(num_accepted, 0), denom)
        if self.cfg.signal == "pos":
            return 1.0 if accepted >= denom else 0.0
        return accepted / denom

    def observe(self, req_id: str, num_accepted: int, num_draft: int) -> float:
        if num_draft <= 0:
            return self._get(req_id).ema
        st = self._get(req_id)
        a = self.cfg.alpha
        st.ema = a * self.signal(num_accepted, num_draft) + (1.0 - a) * st.ema
        self.counters["observations"] += 1
        return st.ema

    # -- decisions ---------------------------------------------------------
    def decide(self, req_id: str) -> int:
        st = self._get(req_id)
        k = self._k_for(st.ema, current=st.k)
        if k != st.k:
            self.counters["switches"] += 1
            st.k = k
        self.counters["decisions_hi" if k == self.cfg.k_hi else "decisions_lo"] += 1
        return k

    def decide_batch(self, req_ids) -> int:
        """One k for the whole step: k_hi only if every request is in the high state."""
        ks = [self.decide(r) for r in req_ids]
        if ks and all(k == self.cfg.k_hi for k in ks):
            self.counters["steps_uniform_hi"] += 1
            return self.cfg.k_hi
        self.counters["steps_uniform_lo"] += 1
        return self.cfg.k_lo

    def evict(self, req_id: str) -> None:
        self._state.pop(req_id, None)

    def counters_line(self) -> str:
        c = self.counters
        return (
            f"tracked={len(self._state)} obs={c['observations']} "
            f"decisions lo/hi={c['decisions_lo']}/{c['decisions_hi']} switches={c['switches']} "
            f"uniform-steps lo/hi={c['steps_uniform_lo']}/{c['steps_uniform_hi']}"
        )


def should_observe(
    req_id: str,
    drafted: "set[str] | frozenset[str]",
    num_draft: int,
    has_generated: bool,
    num_sampled_per_step: int,
    is_stale: bool = False,
    drop_stale: bool = False,
    kv_load_failed: bool = False,
) -> bool:
    """Whether one request's step result may feed the EMA (pure, testable).

    Conservative guards (they skip at least what the base skips): the request
    must be in the drafted set (placeholders handed out by this scheduler, not
    the base's ``[-1]`` resume padding), have a countable output, no stale
    output from a preemption (whether or not it would be delivered), and the
    step must carry no KV-load failure (step-global, the base is per request).
    """
    if kv_load_failed or num_draft <= 0 or req_id not in drafted:
        return False
    if not has_generated and num_sampled_per_step != 0:
        return False
    if is_stale and drop_stale:
        return False
    if is_stale:
        return False
    return True


def placeholder_len(k_req: int, engine_k: int) -> int:
    """Draft length to hand a request: never more than the engine drafts."""
    return max(0, min(int(k_req), int(engine_k)))


class DraftedRing:
    """The req_ids this scheduler handed placeholders to, over the last N hand-outs.

    Three slots on the async path: the step reported by ``update_from_output`` at
    the iteration of N received its placeholders at the iteration of N-2.
    """

    def __init__(self, slots: int = 3) -> None:
        if slots < 1:
            raise ValueError("DraftedRing needs >= 1 slot")
        self.slots = slots
        self._ring: list[frozenset[str]] = []

    def push(self, req_ids) -> None:
        self._ring.append(frozenset(req_ids))
        if len(self._ring) > self.slots:
            del self._ring[0]

    def union(self) -> frozenset[str]:
        out: frozenset[str] = frozenset()
        for s in self._ring:
            out |= s
        return out

    def __contains__(self, req_id: str) -> bool:
        return any(req_id in s for s in self._ring)

    def __len__(self) -> int:
        return len(self._ring)


def filter_candidates(items) -> list[str]:
    """req_ids that get their placeholders re-sized this step.

    ``items``: iterable of ``(req_id, is_finished, is_prefill_chunk, has_placeholders)``
    for the requests in ``scheduler_output.num_scheduled_tokens``. A request that
    the base skipped (prefill chunk) or gave no placeholders keeps whatever it
    has; finished requests are ignored.
    """
    out = []
    for req_id, is_finished, is_prefill_chunk, has_placeholders in items:
        if is_finished or is_prefill_chunk or not has_placeholders:
            continue
        out.append(req_id)
    return out


def assign_lengths(policy: AdaptiveKPolicy, req_ids, mode: str, engine_k: int) -> dict[str, int]:
    """Placeholder length per candidate request for the next step (pure)."""
    req_ids = list(req_ids)
    if not req_ids:
        return {}
    if mode == "batch-uniform":
        k_step = placeholder_len(policy.decide_batch(req_ids), engine_k)
        return {r: k_step for r in req_ids}
    return {r: placeholder_len(policy.decide(r), engine_k) for r in req_ids}


# ---------------------------------------------------------------------------
# vLLM scheduler subclass (only when vLLM is importable)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised only inside the vLLM image
    from vllm.logger import init_logger as _init_logger
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler as _BaseScheduler

    logger = _init_logger("vllm.adaptive_k")  # under the vllm.* hierarchy so INFO lines reach the log
    _HAVE_VLLM = True
except Exception:  # noqa: BLE001 - any import problem means "policy-only module"
    logger = logging.getLogger(__name__)
    _BaseScheduler = object  # type: ignore[assignment,misc]
    _HAVE_VLLM = False


if _HAVE_VLLM:

    class AdaptiveKScheduler(_BaseScheduler):  # type: ignore[misc,valid-type]
        """AsyncScheduler that sizes each request's draft placeholders adaptively.

        Matches vLLM 487ecf187:
          async_scheduler.py:13-17  __init__ (self._spec_token_placeholders = [-1]*num_spec_tokens)
          async_scheduler.py:19-49  _update_after_schedule(self, scheduler_output)
                                    L23-25 placeholders from num_spec_tokens_to_schedule,
                                    L26-28 skip prefill chunks, L44 request.spec_token_ids = placeholders
          scheduler.py:1681         update_from_output(self, scheduler_output, model_runner_output)
          scheduler.py:1777-1787    num_accepted = max(len(generated) - num_sampled_tokens_per_step, 0)
          scheduler.py:2327         _free_request(self, request, delay_free_blocks=False)
        The sync path (scheduler.py:2173 update_draft_token_ids) is intentionally not
        supported: the class disables itself when async scheduling is off.
        """

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self._ak_failed = False
            self._ak_steps = 0
            self._ak_ring = DraftedRing(3)
            self._ak_budget_counters = dict(steps=0, limited_steps=0,
                                           limited_requests=0, trimmed_tokens=0)
            cfg = AdaptiveKConfig(enabled=False)
            engine_k = 0
            try:
                cfg = AdaptiveKConfig.from_env()
                async_on = bool(getattr(self.scheduler_config, "async_scheduling", False))
                engine_k = int(getattr(self, "num_spec_tokens", 0) or 0)
                if cfg.enabled and not async_on:
                    logger.error(
                        "adaptive-k: async scheduling is OFF; this class sizes AsyncScheduler "
                        "placeholders and is not meaningful on the sync path. Disabling the policy."
                    )
                    cfg = AdaptiveKConfig(**{**cfg.__dict__, "enabled": False})
                if cfg.enabled and engine_k < 1:
                    logger.error("adaptive-k: engine drafts no tokens (num_spec_tokens=%s); disabling", engine_k)
                    cfg = AdaptiveKConfig(**{**cfg.__dict__, "enabled": False})
                if cfg.enabled and engine_k < cfg.k_hi:
                    logger.warning(
                        "adaptive-k: engine drafts %s tokens but k_hi=%s; clamping k_hi to %s",
                        engine_k, cfg.k_hi, engine_k,
                    )
                    cfg = AdaptiveKConfig(
                        **{**cfg.__dict__, "k_hi": engine_k, "k_lo": min(cfg.k_lo, engine_k)}
                    )
            except Exception:  # noqa: BLE001 - never let the policy kill the engine core
                logger.exception("adaptive-k: configuration failed, running as the base AsyncScheduler")
                cfg = AdaptiveKConfig(enabled=False)
            self._ak_cfg = cfg
            self._ak = AdaptiveKPolicy(cfg)
            self._ak_engine_k = max(engine_k, 0)
            # read-only placeholder lists per draft length, shared like the base's
            self._ak_placeholders: dict[int, list[int]] = {
                k: [-1] * k for k in range(0, self._ak_engine_k + 1)
            }
            table = getattr(getattr(self.vllm_config, "speculative_config", None),
                            "num_speculative_tokens_per_batch_size", None)
            logger.info(
                "adaptive-k: AdaptiveKScheduler active (%s) engine_k=%s async=%s dynamic_sd_table=%s "
                "(adaptive verification with optional effective draft budget cap)",
                cfg.describe(), self._ak_engine_k,
                getattr(self.scheduler_config, "async_scheduling", None), table,
            )

            logger.info(
                "adaptive-k: draft-budget active=%s source=SchedulerOutput.resolve_num_spec_tokens_to_schedule "
                "engine_k=%s",
                int(cfg.enabled and cfg.respect_draft_budget), self._ak_engine_k,
            )

        def _ak_budget_counters_line(self) -> str:
            c = self._ak_budget_counters
            return (
                f"draft-budget steps={c['steps']} limited_steps={c['limited_steps']} "
                f"limited_requests={c['limited_requests']} trimmed_tokens={c['trimmed_tokens']}"
            )

        # -- size next step's placeholders per request ---------------------
        def _update_after_schedule(self, scheduler_output) -> None:
            super()._update_after_schedule(scheduler_output)
            if not self._ak_cfg.enabled or self._ak_failed:
                return
            try:
                self._ak_size_placeholders(scheduler_output)
            except Exception:  # noqa: BLE001
                self._ak_failed = True
                logger.exception("adaptive-k: placeholder sizing failed, falling back to AsyncScheduler")

        def _ak_size_placeholders(self, scheduler_output) -> None:
            reqs = self.requests
            items = []
            for req_id in scheduler_output.num_scheduled_tokens:
                request = reqs.get(req_id)
                if request is None:
                    continue
                items.append((
                    req_id,
                    request.is_finished(),
                    bool(getattr(request, "is_prefill_chunk", False)),
                    bool(request.spec_token_ids),
                ))
            candidates = filter_candidates(items)
            self._ak_ring.push(candidates)
            lengths = assign_lengths(self._ak, candidates, self._ak_cfg.mode, self._ak_engine_k)
            if self._ak_cfg.respect_draft_budget and lengths:
                # R10's worker resolves this very output when producing next-step
                # drafts. Do not recompute from candidates or a later batch size.
                budget = max(0, min(self._ak_engine_k, int(
                    scheduler_output.resolve_num_spec_tokens_to_schedule(self._ak_engine_k)
                )))
                limited = sum(k > budget for k in lengths.values())
                trimmed = sum(max(0, k - budget) for k in lengths.values())
                c = self._ak_budget_counters
                c["steps"] += 1
                c["limited_steps"] += int(limited > 0)
                c["limited_requests"] += limited
                c["trimmed_tokens"] += trimmed
                lengths = {r: min(k, budget) for r, k in lengths.items()}
                if limited and c["limited_steps"] == 1:
                    logger.info("adaptive-k: draft-budget first-cap budget=%s %s",
                                budget, self._ak_budget_counters_line())
            for req_id, k in lengths.items():
                reqs[req_id].spec_token_ids = self._ak_placeholders[k]

        # -- observe acceptance (before the base mutates/frees) ------------
        def update_from_output(self, scheduler_output, model_runner_output):
            if self._ak_cfg.enabled and not self._ak_failed:
                try:
                    self._ak_observe(scheduler_output, model_runner_output)
                except Exception:  # noqa: BLE001
                    self._ak_failed = True
                    logger.exception("adaptive-k: observe failed, falling back to AsyncScheduler")
            return super().update_from_output(scheduler_output, model_runner_output)

        def _ak_observe(self, scheduler_output, model_runner_output) -> None:
            sched_spec = scheduler_output.scheduled_spec_decode_tokens
            if not sched_spec:
                return
            kv_out = getattr(model_runner_output, "kv_connector_output", None)
            kv_load_failed = bool(kv_out is not None and getattr(kv_out, "invalid_block_ids", None))
            sampled_token_ids = model_runner_output.sampled_token_ids
            req_id_to_index = model_runner_output.req_id_to_index
            num_sampled = self.num_sampled_tokens_per_step
            drafted = self._ak_ring.union()  # last three hand-outs (async runs one step ahead)
            for req_id, spec_token_ids in sched_spec.items():
                request = self.requests.get(req_id)
                if request is None or request.is_finished():
                    continue
                req_index = req_id_to_index.get(req_id)
                if req_index is None:
                    continue
                generated = sampled_token_ids[req_index] if sampled_token_ids else []
                if not should_observe(
                    req_id,
                    drafted,
                    len(spec_token_ids),
                    bool(generated),
                    num_sampled,
                    is_stale=getattr(request, "num_stale_output_tokens", 0) > 0,
                    drop_stale=bool(getattr(request, "drop_stale_output", False)),
                    kv_load_failed=kv_load_failed,
                ):
                    continue
                num_accepted = max(len(generated) - num_sampled, 0)
                self._ak.observe(req_id, num_accepted, len(spec_token_ids))
            self._ak_steps += 1
            if self._ak_cfg.log_every and self._ak_steps % self._ak_cfg.log_every == 0:
                logger.info("adaptive-k: %s; %s", self._ak.counters_line(),
                            self._ak_budget_counters_line())

        # -- eviction --------------------------------------------------------
        def _free_request(self, request, delay_free_blocks: bool = False):
            try:
                self._ak.evict(request.request_id)
            except Exception:  # noqa: BLE001
                pass
            return super()._free_request(request, delay_free_blocks)

else:  # policy-only import (tests, tooling)

    class AdaptiveKScheduler:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("AdaptiveKScheduler needs vLLM; only AdaptiveKPolicy is available here")
