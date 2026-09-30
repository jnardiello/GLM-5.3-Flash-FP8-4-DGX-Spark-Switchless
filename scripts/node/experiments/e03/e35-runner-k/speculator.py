# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import gumbel_noised_argmax
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator


@triton.jit
def _selector_walk_kernel(
    scores_ptr,
    candidate_ptr,
    sample_pos_ptr,
    req_state_ptr,
    temperature_ptr,
    seeds_ptr,
    tokens_ptr,
    realized_scores_ptr,
    num_steps: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SAMPLE_PROBABILISTIC: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < top_k
    req_state = tl.load(req_state_ptr + row * num_steps)
    valid = req_state >= 0
    temperature = tl.load(temperature_ptr + req_state, mask=valid, other=0.0)
    seed = tl.load(seeds_ptr + req_state, mask=valid, other=0)
    previous = 0
    for step in range(num_steps):
        flat = row * num_steps + step
        score_base = (flat * top_k + previous) * top_k
        scores = tl.load(
            scores_ptr + score_base + offsets,
            mask=mask & valid,
            other=float("-inf"),
        ).to(tl.float64 if USE_FP64 else tl.float32)
        candidate_base = flat * top_k
        candidates = tl.load(
            candidate_ptr + candidate_base + offsets,
            mask=mask & valid,
            other=0,
        )

        # Candidate ids key the noise, matching the target's own sampling.
        position = tl.load(sample_pos_ptr + flat) - 1
        _, index = gumbel_noised_argmax(
            scores,
            candidates,
            mask & valid,
            seed,
            position,
            temperature if SAMPLE_PROBABILISTIC else 0.0,
            USE_FP64=USE_FP64,
        )

        tl.store(
            realized_scores_ptr + candidate_base + offsets,
            scores,
            mask=mask & valid,
        )
        token = tl.load(candidate_ptr + candidate_base + index, mask=valid, other=0)
        tl.store(tokens_ptr + flat, token, mask=valid)
        previous = index


@triton.jit
def _cache_draft_logits_kernel(
    draft_logits_ptr,
    cached_candidate_ptr,
    candidate_ptr,
    scores_ptr,
    req_state_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    num_steps: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    flat = tl.program_id(0)
    req_state = tl.load(req_state_ptr + flat)
    step = flat % num_steps
    offsets = tl.arange(0, BLOCK_K)
    mask = (req_state >= 0) & (offsets < top_k)
    candidate_base = flat * top_k
    cache_base = (req_state * num_steps + step) * top_k
    old_token_ids = tl.load(cached_candidate_ptr + cache_base + offsets, mask=mask)
    logits_base = (
        draft_logits_ptr
        + req_state * draft_logits_stride_0
        + step * draft_logits_stride_1
    )
    tl.store(logits_base + old_token_ids, -float("inf"), mask=mask)
    token_ids = tl.load(candidate_ptr + candidate_base + offsets, mask=mask)
    scores = tl.load(scores_ptr + candidate_base + offsets, mask=mask)
    tl.store(logits_base + token_ids, scores, mask=mask)
    tl.store(cached_candidate_ptr + cache_base + offsets, token_ids, mask=mask)


class DFlash2Speculator(DFlashSpeculator):
    _speculator_name = "DFlash2"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        draft_config = self.draft_model_config.hf_config.dflash_config
        self.selector_top_k = int(draft_config["selector_top_k"])
        self._anchor_indices = (
            torch.arange(self.max_num_reqs, dtype=torch.int64, device=device)
            * self.num_query_per_req
        )
        self._selector_scores = torch.empty(
            self.max_num_reqs,
            self.num_speculative_steps,
            self.selector_top_k,
            dtype=torch.float32,
            device=device,
        )
        self._cached_candidate_ids = torch.zeros(
            self._selector_scores.shape, dtype=torch.int64, device=device
        )

    def draft_logits_spec(self, vllm_config: VllmConfig) -> tuple[torch.dtype, float]:
        # fp32 so the walk and the rejection that checks it read the same
        # distribution; -inf because the cache kernel writes only the K
        # candidates.
        return torch.float32, -float("inf")

    def _sample_path(
        self,
        candidate_ids: torch.Tensor,
        scores: torch.Tensor,
        num_reqs: int,
    ) -> None:
        block_k = triton.next_power_of_2(self.selector_top_k)
        _selector_walk_kernel[(num_reqs,)](
            scores.contiguous(),
            candidate_ids.contiguous(),
            self.sample_pos,
            self.sample_idx_mapping,
            self.temperature,
            self.seeds,
            self.draft_tokens,
            self._selector_scores,
            num_steps=self.num_speculative_steps,
            top_k=self.selector_top_k,
            BLOCK_K=block_k,
            SAMPLE_PROBABILISTIC=self.draft_logits is not None,
            USE_FP64=self.use_fp64_gumbel,
            num_warps=1,
        )

    def _cache_draft_logits(self, candidate_ids: torch.Tensor, num_sample: int) -> None:
        draft_logits = self.draft_logits
        assert draft_logits is not None
        block_k = triton.next_power_of_2(self.selector_top_k)
        _cache_draft_logits_kernel[(num_sample,)](
            draft_logits,
            self._cached_candidate_ids,
            candidate_ids,
            self._selector_scores,
            self.sample_idx_mapping,
            draft_logits.stride(0),
            draft_logits.stride(1),
            num_steps=self.num_speculative_steps,
            top_k=self.selector_top_k,
            BLOCK_K=block_k,
            num_warps=1,
        )

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        last_hidden_states = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        num_sample = num_reqs * self.num_speculative_steps
        hidden_states = last_hidden_states[self.sample_indices[:num_sample]].view(
            num_reqs, self.num_speculative_steps, -1
        )
        candidate_ids, unary_logits = self.model.compute_candidates(
            hidden_states.flatten(0, 1)
        )
        candidate_ids = candidate_ids.view(
            num_reqs, self.num_speculative_steps, self.selector_top_k
        )
        unary_logits = unary_logits.view_as(candidate_ids)
        anchor_token_ids = self.input_buffers.input_ids[self._anchor_indices[:num_reqs]]
        scores = self.model.model.candidate_selector(
            candidate_ids,
            unary_logits,
            hidden_states,
            anchor_token_ids,
        )
        self._sample_path(candidate_ids, scores, num_reqs)
        if self.draft_logits is not None:
            self._cache_draft_logits(candidate_ids, num_sample)


# --- E35 candidate addition. Every byte above this marker is the serving image's
# vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py (Apache-2.0, vLLM project;
# sha256 1f6ff5ca9c8f38ff417aafd43bfa3116b5387bf0f7b58721acb2185781879836).
# With VLLM_E35_ENABLE=1, tensor-parallel rank 0 records after every real propose() the new
# draft's per-position confidence for the E35 runner: the softmax probability of the greedy
# choice among the selector's realized top-k scores, computed on the device and copied
# without blocking into a pinned ring slot with an event. `_e35_store` keeps, per request,
# the two newest (seq, event, host row) entries, seq counting real propose() calls, and
# `_e35_last_seq` the newest seq. The runner uses an entry only when its seq proves it is the
# current draft (last seq) or the one right before it (last seq - 1). The sequence advances on
# every real propose() on rank 0, recorded or not, so neither a failed attempt nor an "ema"
# interval leaves an older entry marked current or lag 1, and the ring slot advances only after
# a successful copy. Under the "ema" policy (or with the flag missing) nothing is computed.
import collections as _e35_collections
import os as _e35_os
import time as _e35_time

from vllm.logger import init_logger as _e35_init_logger

_e35_logger = _e35_init_logger(__name__)
_E35_ENABLED = _e35_os.environ.get("VLLM_E35_ENABLE", "") == "1"
_E35_POLICY_FLAG = _e35_os.environ.get("VLLM_E35_POLICY_FLAG", "")
_E35_SLOTS = 8
_E35_MAX_REQUESTS = 64


def _e35_policy_active(path, opener=open) -> bool:
    """True for a readable ASCII flag "lag1" or "hybrid"."""
    if not path:
        return False
    try:
        with opener(path, "rb") as stream:
            raw = stream.read(65)
        return len(raw) <= 64 and raw.decode("ascii").strip() in ("lag1", "hybrid")
    except (OSError, UnicodeDecodeError):
        return False


def _e35_is_tp_rank0():
    try:
        from vllm.distributed.parallel_state import get_tp_group
        return get_tp_group().rank_in_group == 0
    except Exception:  # noqa: BLE001 - never break serving
        return False


def _e35_remember(store, seq, req_ids, slot, event, max_requests=_E35_MAX_REQUESTS):
    """Pure bookkeeping: append (seq, event, row) per request, keep two, bound the requests."""
    for row, req_id in enumerate(req_ids):
        entries = store.pop(req_id, None) or _e35_collections.deque(maxlen=2)
        entries.append((seq, event, slot[row]))
        store[req_id] = entries
    while len(store) > max_requests:
        store.pop(next(iter(store)))


_e35_original_propose = DFlash2Speculator.propose


def _e35_propose(self, input_batch, attn_metadata, slot_mappings, last_hidden_states,
                 aux_hidden_states, num_sampled, num_rejected, *args, **kwargs):
    draft = _e35_original_propose(
        self, input_batch, attn_metadata, slot_mappings, last_hidden_states,
        aux_hidden_states, num_sampled, num_rejected, *args, **kwargs)
    if not _E35_ENABLED or kwargs.get("dummy_run") or kwargs.get("is_profile"):
        return draft
    state = self.__dict__.get("_e35_state")
    if state is None:
        state = self._e35_state = {"rank0": _e35_is_tp_rank0(), "slots": None, "next": 0,
                                   "checked_at": None, "active": False}
        self._e35_store = {}
        self._e35_last_seq = 0
    if not state["rank0"]:
        return draft
    now = _e35_time.monotonic()
    if state["checked_at"] is None or now - state["checked_at"] >= 0.5:
        state["checked_at"] = now
        state["active"] = _e35_policy_active(_E35_POLICY_FLAG)
    # Advance on every real proposal, before the policy check and the recording attempt: after
    # a failed attempt or an unrecorded "ema" interval no stored entry passes as current or lag 1.
    self._e35_last_seq += 1
    if not state["active"]:
        return draft
    try:
        if torch.cuda.is_current_stream_capturing():
            return draft
        num_reqs = input_batch.num_reqs
        scores = self._selector_scores
        if state["slots"] is None:
            state["slots"] = torch.empty((_E35_SLOTS,) + tuple(scores.shape[:2]),
                                         dtype=torch.float32, pin_memory=True)
        conf = torch.nan_to_num(torch.softmax(scores[:num_reqs], dim=-1).amax(dim=-1), nan=0.0)
        slot = state["slots"][state["next"]]
        slot[:num_reqs].copy_(conf, non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        _e35_remember(self._e35_store, self._e35_last_seq, list(input_batch.req_ids[:num_reqs]),
                      slot, event)
        state["next"] = (state["next"] + 1) % _E35_SLOTS
    except Exception as error:  # noqa: BLE001 - never break serving
        _e35_logger.warning("E35_CONF_RECORD_ERROR %s", type(error).__name__)
    return draft


DFlash2Speculator.propose = _e35_propose
_e35_logger.info("E35_CONF_RECORDER_READY enabled=%d", int(_E35_ENABLED))
