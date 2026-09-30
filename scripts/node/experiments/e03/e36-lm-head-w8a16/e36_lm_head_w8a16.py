# SPDX-License-Identifier: Apache-2.0
"""E36 candidate: INT8 W8A16 for the shared vocab-parallel ``lm_head``.

The DFlash2 drafter holds the target's ``ParallelLMHead`` as the same module object
(``vllm/v1/worker/gpu/spec_decode/dflash/utils.py``: ``dflash_model.lm_head =
target_lm_head``). Both of its users project through ``lm_head.quant_method.apply``
(``LogitsProcessor._apply_head``): the target's eager ``compute_logits`` and the drafter's
``get_top_k_tokens`` inside its CUDA graph. Converting the one module after both models are
loaded and before any graph is captured therefore changes both users with a single INT8
copy and frees the BF16 weight.

The packing is the accepted E21 one (``e21_bf16_residue._pack_projection``: INT8 symmetric
group-128, Marlin repack, N padded to a multiple of 256) without a BF16 scratch, so every
row count runs the Marlin kernel, in serving and in the fidelity measurement alike.

Environment, parsed strictly:

* ``VLLM_E36_LM_HEAD_W8A16``: a non-zero integer converts at load; ``0`` or unset does not.
* ``VLLM_E36_KEEP_BF16``: fidelity measurement only. A non-zero integer keeps the BF16
  weight as well, and each eager call follows the flag file below.
* ``VLLM_E36_FLAG``: path of a flag file holding ``bf16`` or ``int8`` (default ``int8``),
  re-read at most every 0.5 s. Graph-captured calls keep the path active at capture.
"""

from __future__ import annotations

import json
import math
import os
import time

import torch
from torch.nn import functional as F

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import gptq_quantize_weights
from vllm.v1.worker.gpu.spec_decode.eagle.utils import get_target_lm_head

from .e21_bf16_residue import GROUP_SIZE, WTYPE, ResidueW8A16Method, _pack_projection


logger = init_logger(__name__)

ENABLE_ENV = "VLLM_E36_LM_HEAD_W8A16"
KEEP_ENV = "VLLM_E36_KEEP_BF16"
FLAG_ENV = "VLLM_E36_FLAG"
PATHS = ("bf16", "int8")
FLAG_MAX_BYTES = 64
POLL_SECONDS = 0.5
CHECK_ROWS = 8
CHECK_SEED = 20261001
#: Rows quantized at a time for the weight-error receipt: groups run along K, so row chunks
#: give the exact error without a full-size transient.
CHECK_CHUNK_ROWS = 4096
#: A packing whose logits drift this far from BF16 on the check rows is refused as broken.
MAX_LOGITS_REL_L2 = 0.05


def strict_flag(name: str, environ: dict | None = None) -> bool:
    """A non-zero integer is on; unset, empty or ``0`` is off; anything else raises."""
    env = os.environ if environ is None else environ
    raw = str(env.get(name, "0")).strip()
    if raw == "":
        return False
    try:
        return int(raw) != 0
    except ValueError:
        raise RuntimeError(f"E36 {name} must be an integer, got {raw!r}") from None


def enabled(environ: dict | None = None) -> bool:
    return strict_flag(ENABLE_ENV, environ)


def parse_path(raw: bytes | None, default: str = "int8") -> str:
    """``bf16`` or ``int8`` from a flag file's bytes; missing or anything else is the default."""
    if raw is None or len(raw) > FLAG_MAX_BYTES:
        return default
    try:
        value = raw.decode("ascii").strip()
    except UnicodeDecodeError:
        return default
    return value if value in PATHS else default


def check_head(shape: tuple, dtype: str, is_cuda: bool, contiguous: bool, has_bias: bool,
               method: str, group_size: int = GROUP_SIZE) -> tuple[int, int]:
    """Refuse anything but an unbiased, unquantized, contiguous BF16 CUDA [N, K] head."""
    if len(shape) != 2:
        raise RuntimeError(f"E36 expects a 2-D lm_head weight, got shape {shape}")
    size_n, size_k = int(shape[0]), int(shape[1])
    if dtype != "torch.bfloat16" or not is_cuda or not contiguous:
        raise RuntimeError(f"E36 expects a contiguous BF16 CUDA lm_head, got {dtype} cuda={is_cuda}")
    if has_bias:
        raise RuntimeError("E36 refuses an lm_head with a bias")
    if method != "UnquantizedEmbeddingMethod":
        raise RuntimeError(f"E36 expects the unquantized embedding method, got {method}")
    if size_n <= 0 or size_k <= 0 or size_k % group_size or size_n % 16:
        raise RuntimeError(f"E36 cannot pack an lm_head of shape {shape} with group {group_size}")
    return size_n, size_k


class E36SwitchMethod(LinearMethodBase):
    """Fidelity measurement: the INT8 or the retained BF16 path, chosen by the flag file."""

    def __init__(self, int8_method: ResidueW8A16Method, bf16_method: UnquantizedEmbeddingMethod,
                 flag: str):
        self.int8_method = int8_method
        self.bf16_method = bf16_method
        self.flag = flag
        self.path = "int8"
        self.checked_at = None

    def create_weights(self, layer, *args, **kwargs) -> None:
        raise RuntimeError("E36 attaches after the original loader finishes")

    def poll(self) -> str:
        now = time.monotonic()
        if self.checked_at is None or now - self.checked_at >= POLL_SECONDS:
            self.checked_at = now
            try:
                with open(self.flag, "rb") as stream:
                    raw = stream.read(FLAG_MAX_BYTES + 1)
            except OSError:
                raw = None
            path = parse_path(raw)
            if path != self.path:
                logger.info("E36_LM_HEAD path=%s", path)
            self.path = path
        return self.path

    def apply(self, layer, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        if self.poll() == "bf16":
            return self.bf16_method.apply(layer, x, bias)
        return self.int8_method.apply(layer, x, bias)


@torch.no_grad()
def _logit_check(layer, weight: torch.Tensor, int8_method: ResidueW8A16Method) -> dict:
    """Seeded Gaussian rows through BF16 and INT8: relative L2 and top-1 agreement."""
    generator = torch.Generator(device=weight.device).manual_seed(CHECK_SEED)
    rows = torch.randn(CHECK_ROWS, weight.shape[1], generator=generator,
                       device=weight.device, dtype=torch.float32).to(weight.dtype)
    reference = F.linear(rows, weight).float()
    quantized = int8_method.apply(layer, rows).float()
    if quantized.shape != reference.shape:
        raise RuntimeError(f"E36 INT8 logits {tuple(quantized.shape)} != {tuple(reference.shape)}")
    agree = int((quantized.argmax(-1) == reference.argmax(-1)).sum().item())
    rel = float((quantized - reference).norm() / reference.norm().clamp_min(1e-12))
    if not math.isfinite(rel) or rel > MAX_LOGITS_REL_L2:
        raise RuntimeError(f"E36 INT8 logits deviate from BF16 by {rel!r} (limit {MAX_LOGITS_REL_L2})")
    return {"logits_rel_l2": round(rel, 6), "top1_agreement": f"{agree}/{CHECK_ROWS}"}


@torch.no_grad()
def _weight_rel_l2(weight: torch.Tensor) -> float:
    """Group-128 quantization error of the whole weight, one chunk of output rows at a time."""
    error = total = 0.0
    for start in range(0, weight.shape[0], CHECK_CHUNK_ROWS):
        chunk_kn = weight[start:start + CHECK_CHUNK_ROWS].T.contiguous()
        reference, quantized, scales, _, _ = gptq_quantize_weights(
            chunk_kn, WTYPE, GROUP_SIZE, act_order=False)
        exact = chunk_kn.float()
        error += float((reference.float() - exact).pow(2).sum())
        total += float(exact.pow(2).sum())
        del reference, quantized, scales, chunk_kn, exact
    rel = math.sqrt(error / max(total, 1e-24))
    if not math.isfinite(rel):
        raise RuntimeError("E36 weight quantization error is not finite")
    return round(rel, 6)


def convert_shared_lm_head(target_model, speculator, environ: dict | None = None) -> dict:
    """Convert the target's lm_head (and so the drafter's) in place; log and return a receipt."""
    # Resolve the head exactly as the DFlash sharing does: a *ForConditionalGeneration target
    # holds it on get_language_model(), a plain causal LM on itself.
    language_model = (target_model.get_language_model()
                      if hasattr(target_model, "get_language_model") else target_model)
    lm_head = get_target_lm_head(target_model, language_model)
    if not isinstance(lm_head, ParallelLMHead):
        raise RuntimeError(f"E36 expects a ParallelLMHead, got {type(lm_head).__name__}")
    drafter = getattr(getattr(speculator, "model", None), "lm_head", None)
    if speculator is not None and drafter is not lm_head:
        raise RuntimeError("E36 requires the drafter to hold the target's lm_head module")
    weight = lm_head.weight
    size_n, size_k = check_head(
        tuple(weight.shape), str(weight.dtype), weight.is_cuda, weight.is_contiguous(),
        getattr(lm_head, "bias", None) is not None, type(lm_head.quant_method).__name__)
    if size_n != lm_head.num_embeddings_per_partition or size_k != lm_head.embedding_dim:
        raise RuntimeError("E36 lm_head weight does not match its partition attributes")
    del weight
    keep = strict_flag(KEEP_ENV, environ)
    flag = (os.environ if environ is None else environ).get(FLAG_ENV, "")
    if keep and not flag:
        raise RuntimeError(f"E36 {KEEP_ENV}=1 requires {FLAG_ENV}")

    bf16_method = lm_head.quant_method
    bf16_param = lm_head._parameters["weight"]
    weight_rel = _weight_rel_l2(bf16_param)
    receipt = _pack_projection(lm_head, size_n, size_k, scratch=None)
    int8_method = lm_head.quant_method
    receipt.update(_logit_check(lm_head, bf16_param, int8_method))
    receipt.update({
        "shape": [size_n, size_k], "group_size": GROUP_SIZE, "weight_rel_l2": weight_rel,
        "shared_with_drafter": drafter is lm_head, "keep_bf16": keep,
        "freed_bytes": 0 if keep else receipt["original_bytes"],
        "tp_rank_vocab_start": int(lm_head.shard_indices.org_vocab_start_index),
    })
    if keep:
        lm_head.register_parameter("weight", bf16_param)
        lm_head.quant_method = E36SwitchMethod(int8_method, bf16_method, flag)
        receipt["flag"] = flag
    del bf16_param
    logger.info("E36_LM_HEAD_W8A16_READY %s", json.dumps(receipt, sort_keys=True))
    return receipt
