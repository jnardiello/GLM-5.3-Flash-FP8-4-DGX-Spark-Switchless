# SPDX-License-Identifier: Apache-2.0
"""E22 candidate: the accepted E20/E21 W8A16 mechanism applied to the DFlash2 drafter.

At load time each selected BF16 drafter linear becomes INT8 symmetric group-128,
packed for native Marlin, through E21's ``_pack_projection`` and ``ResidueW8A16Method``.
Nothing about the mechanism is new; three things differ from E21:

* the families are the drafter's own: per decoder layer ``self_attn.qkv_proj``,
  ``self_attn.o_proj``, ``mlp.gate_up_proj``, ``mlp.down_proj`` and the two grouped
  convolutions' ``kernel_projection``, plus the fused context K/V projection;
* the query-path families see only ``streams * (1 + draft length)`` rows, at most a few
  dozen, so they get no BF16 scratch and always run Marlin;
* the fused context K/V projection is a DFlash-specific ``torch.cat`` copy of the K/V
  rows of every ``qkv_proj``, built by the vendor loader. It sees every context row,
  thousands during prefill, so it keeps the E20 hybrid: Marlin below
  ``PREFILL_BF16_MIN_TOKENS`` rows, dequantized BF16 scratch at or above it. It is
  installed through the vendor's own ``_fused_kv_linear`` / ``_fused_kv_quant_method``
  hook, so the vendor forward is unchanged.

Excluded: ``fc`` (its ``[4096, 20480]`` scratch alone would be 160 MiB), ``lm_head`` and
``embed_tokens`` (shared with the target; not in the drafter checkpoint), and the tiny
``candidate_selector.hidden_projection``. Every expected shape is derived from the
drafter config and the tensor-parallel size; any mismatch refuses before a weight changes.
"""

from __future__ import annotations

import json
import os

import torch
from torch import nn

from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
    get_weight_perm,
)

from .e21_bf16_residue import (
    GROUP_SIZE,
    MARLIN_TILE,
    PREFILL_BF16_MIN_TOKENS,
    ShapedBF16Scratch,
    _pack_projection,
)


logger = init_logger(__name__)

ENABLE_ENV = "VLLM_E22_DRAFTER_W8A16"
#: Exact DFlash2 drafter census; a different one refuses the conversion.
EXPECTED_LAYERS = 5
#: Ceiling on the BF16 scratch this module may allocate per rank.
SCRATCH_BUDGET_BYTES = 32 * 1024 * 1024
CONTEXT_KV = "context_kv"
#: E22b: ``0`` leaves the fused context K/V projection in BF16, exactly as the vendor
#: loader builds it; the query-path families are converted either way.
CONTEXT_ENABLE_ENV = "VLLM_E22_CONTEXT_KV_W8A16"


def _flag(environ: dict | None, name: str, default: str) -> bool:
    env = os.environ if environ is None else environ
    raw = env.get(name, default).strip()
    if raw == "":
        return False
    try:
        return int(raw) != 0
    except ValueError:
        raise RuntimeError(f"E22 {name} must be an integer, got {raw!r}") from None


def enabled(environ: dict | None = None) -> bool:
    """Off unless :data:`ENABLE_ENV` is a non-zero integer."""
    return _flag(environ, ENABLE_ENV, "0")


def context_enabled(environ: dict | None = None) -> bool:
    """On unless :data:`CONTEXT_ENABLE_ENV` is set to 0 (E22b)."""
    return _flag(environ, CONTEXT_ENABLE_ENV, "1")


def _check_shape(name: str, index: int, shape: tuple[int, int]) -> tuple[int, int]:
    size_n, size_k = shape
    if size_n <= 0 or size_k <= 0 or size_n % MARLIN_TILE or size_k % GROUP_SIZE:
        raise RuntimeError(
            f"E22 family {name} shape {shape} at layer {index} does not satisfy the "
            f"Marlin group-{GROUP_SIZE} contract"
        )
    return shape


def expected_shapes(config, tp_size: int) -> dict[str, tuple[int, int]]:
    """Per-rank ``(N, K)`` of every family, from the drafter config alone."""
    hidden = int(config.hidden_size)
    heads = int(config.num_attention_heads)
    kv_heads = int(config.num_key_value_heads)
    head_dim = int(getattr(config, "head_dim", None) or hidden // heads)
    if heads % tp_size or int(config.intermediate_size) % tp_size:
        raise RuntimeError(f"E22 drafter config does not divide by TP {tp_size}")
    local_q = heads // tp_size * head_dim
    local_kv = max(1, kv_heads // tp_size) * head_dim
    local_inter = int(config.intermediate_size) // tp_size
    draft = config.dflash_config
    taps = int(draft["conv_kernel_size"])
    groups = hidden // int(draft["conv_group_size"])
    return {
        "qkv_proj": (local_q + 2 * local_kv, hidden),
        "o_proj": (hidden, local_q),
        "gate_up_proj": (2 * local_inter, hidden),
        "down_proj": (hidden, local_inter),
        "kernel_projection": (2 * taps * groups, hidden),
        CONTEXT_KV: (int(config.num_hidden_layers) * 2 * local_kv, hidden),
    }


def _validate(name: str, index: int, module: nn.Module, expected: tuple[int, int]) -> None:
    weight = getattr(module, "weight", None)
    if (
        weight is None
        or tuple(weight.shape) != expected
        or weight.dtype != torch.bfloat16
        or weight.device.type != "cuda"
        or not isinstance(getattr(module, "quant_method", None), UnquantizedLinearMethod)
        or getattr(module, "bias", None) is not None
    ):
        raise RuntimeError(
            f"E22 family {name} projection differs from the expected {expected} "
            f"BF16 contract at layer {index}"
        )


def _layer_modules(layer: nn.Module) -> list[tuple[str, nn.Module]]:
    try:
        return [
            ("qkv_proj", layer.self_attn.qkv_proj),
            ("o_proj", layer.self_attn.o_proj),
            ("gate_up_proj", layer.mlp.gate_up_proj),
            ("down_proj", layer.mlp.down_proj),
            ("kernel_projection", layer.attention_conv.kernel_projection),
            ("kernel_projection", layer.mlp_conv.kernel_projection),
        ]
    except AttributeError as error:
        raise RuntimeError(f"E22 drafter layer lacks an expected module: {error}") from None


def _context_kv(model: nn.Module, expected: tuple[int, int]) -> torch.Tensor:
    weight = getattr(model, "_fused_kv_weight", None)
    if (
        not hasattr(model, "_num_attn_layers")
        or not hasattr(model, "_fused_kv_linear")
        or getattr(model, "_fused_kv_quant_method", "absent") is not None
        or getattr(model, "_fused_kv_bias", "absent") is not None
        or weight is None
        or tuple(weight.shape) != expected
        or weight.dtype != torch.bfloat16
        or weight.device.type != "cuda"
    ):
        raise RuntimeError(
            "E22 requires the vendor's built BF16 fused context K/V projection "
            f"{expected} without bias or quantization"
        )
    return weight


def plan_conversion(
    model: nn.Module, tp_size: int, include_context: bool = True
) -> tuple[list[tuple], torch.Tensor | None]:
    """Resolve and validate every module before any weight is changed.

    With ``include_context`` false the fused context K/V projection is neither checked
    nor returned, and stays exactly as the vendor loader built it.
    """
    layers = list(getattr(model, "layers", []))
    if len(layers) != EXPECTED_LAYERS:
        raise RuntimeError(f"E22 expected {EXPECTED_LAYERS} drafter layers, got {len(layers)}")
    shapes = expected_shapes(model.config, tp_size)
    for name, shape in shapes.items():
        _check_shape(name, -1, shape)
    work: list[tuple] = []
    for index, layer in enumerate(layers):
        for name, module in _layer_modules(layer):
            _validate(name, index, module, shapes[name])
            work.append((name, index, module, shapes[name]))
    if not include_context:
        return work, None
    context_weight = _context_kv(model, shapes[CONTEXT_KV])
    return work, context_weight


@torch.no_grad()
def finalize_drafter_w8a16(model: nn.Module, environ: dict | None = None) -> None:
    """Convert the drafter families after the vendor loader, if enabled."""
    if not enabled(environ):
        return
    if getattr(model, "_e22_drafter_ready", False):
        return
    include_context = context_enabled(environ)
    work, context_weight = plan_conversion(
        model, get_tensor_model_parallel_world_size(), include_context
    )
    context_shape = tuple(context_weight.shape) if context_weight is not None else None
    budget = context_shape[0] * context_shape[1] * 2 if context_shape else 0
    if budget > SCRATCH_BUDGET_BYTES:
        raise RuntimeError(
            f"E22 needs {budget} scratch bytes, above the {SCRATCH_BUDGET_BYTES}-byte ceiling"
        )

    receipts = []
    for name, index, module, (size_n, size_k) in work:
        receipts.append({
            "family": name, "layer": index, "shape_nk": [size_n, size_k],
            **_pack_projection(module, size_n, size_k, None),
        })
    if context_weight is not None:
        device = context_weight.device
        perm = get_weight_perm(8, is_a_8bit=False)
        if perm.numel() != 1024:
            raise RuntimeError("Native Marlin weight permutation changed")
        inverse_weight_perm = torch.argsort(perm).to(device=device, dtype=torch.int32)
        scratch = ShapedBF16Scratch(context_shape[0], context_shape[1], device, inverse_weight_perm)
        model.register_buffer("e22_context_kv_scratch", scratch.scratch, persistent=False)
        # The vendor forward calls _fused_kv_quant_method.apply(_fused_kv_linear, ...)
        # when the method is set, so installing it needs no forward change.
        holder = model._fused_kv_linear
        holder.register_parameter("weight", nn.Parameter(context_weight, requires_grad=False))
        receipts.append({
            "family": CONTEXT_KV, "layer": -1, "shape_nk": list(context_shape),
            **_pack_projection(holder, context_shape[0], context_shape[1], scratch),
        })
        model._fused_kv_quant_method = holder.quant_method
        model._fused_kv_weight = None
        del context_weight
    model._e22_drafter_ready = True
    torch.cuda.empty_cache()
    logger.info(
        "E22_DRAFTER_W8A16_READY %s",
        json.dumps(
            {
                "modules": len(receipts),
                "families": sorted({r["family"] for r in receipts}),
                "prefill_bf16_min_tokens": PREFILL_BF16_MIN_TOKENS,
                "group_size": GROUP_SIZE,
                "context_kv_w8a16": include_context,
                "scratch_shape_nk": list(context_shape) if context_shape else None,
                "added_scratch_bytes": budget,
                "scratch_budget_bytes": SCRATCH_BUDGET_BYTES,
                "original_bytes": sum(r["original_bytes"] for r in receipts),
                "packed_bytes": sum(r["packed_bytes"] for r in receipts),
                "receipts": receipts,
            },
            sort_keys=True,
        ),
    )
