# SPDX-License-Identifier: Apache-2.0
"""E21 candidate: extend the accepted E20 W8A16 mechanism to further BF16 families.

The mechanism is unchanged from
``scripts/node/overrides/vllm/models/glm5next/nvidia/e20_kda_w8a16.py``: at load
time a BF16 linear weight becomes INT8 symmetric group-128, packed for native
Marlin; below ``PREFILL_BF16_MIN_TOKENS`` flattened input rows the apply path is
Marlin, at or above it the weight is dequantized into a BF16 scratch matrix and a
dense GEMM runs. Only three things are new here:

* families are described by :class:`FamilyPlan` records instead of module-level
  ``(OUTPUT_SIZE, INPUT_SIZE)`` constants, and every module is validated against
  its own family's expected shape;
* one scratch buffer per *distinct* shape, sized from the resolved shapes and
  capped by :data:`SCRATCH_BUDGET_BYTES`, instead of one fixed ``[N, K]`` buffer;
* the per-family expected module count is part of the plan, so a census mismatch
  refuses the conversion instead of converting fewer modules.

The accepted ``in_proj_qkvgfab`` family is *not* handled here. It keeps its own
module, its own ``[6288, 4096]`` scratch and its own Triton specialization; this
module only reuses that module's dequantization kernel and inverse permutation.

Every expected shape is derived at load time from the attributes the serving
classes set on themselves (``hidden_size``, ``local_projection_size``,
``num_local_heads``, ``qk_head_dim``, ``v_head_dim``, ``q_lora_rank``,
``kv_lora_rank``, ``qk_rope_head_dim``). The concrete numbers are *not* in the
repository for the MLA families: see this directory's README. A derived shape
that does not match the loaded weight exactly, or that does not satisfy the
Marlin group-128 storage contract, raises instead of converting, so an incorrect
assumption cannot silently quantize a weight.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    apply_gptq_marlin_linear,
    marlin_make_workspace_new,
    marlin_pad_qweight,
    marlin_pad_scales,
    marlin_padded_nk,
    marlin_permute_scales,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
    get_weight_perm,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    gptq_quantize_weights,
    pack_quantized_values_into_int32,
)
from vllm.scalar_type import scalar_types
from vllm.triton_utils import triton

# Reuse the accepted kernel rather than copying it: its PADDED_N/SIZE_N/SIZE_K
# are already Triton compile-time parameters, so a second shape only adds a
# specialization. The accepted module and its 49.125 MiB scratch are untouched.
from .e20_hybrid_scratch import _dequant_nk_kernel


logger = init_logger(__name__)

ENABLE_ENV = "VLLM_E21_BF16_RESIDUE_W8A16"
GROUP_SIZE = 128
WTYPE = scalar_types.uint8b128
PREFILL_BF16_MIN_TOKENS = 2048
MARLIN_TILE = 16
#: Ceiling on the *additional* BF16 scratch this module may allocate per rank.
#: Above it the conversion refuses rather than silently taking KV headroom.
SCRATCH_BUDGET_BYTES = 128 * 1024 * 1024
#: Exact F1 TP4 layer census. A different census refuses the conversion.
EXPECTED_LAYERS = {"kda": 34, "mla": 11}


def _kda_o_proj_shape(attn: nn.Module) -> tuple[int, int]:
    # RowParallelLinear(projection_size, hidden_size); the row-parallel shard is
    # the input side, so the TP-local weight is [hidden_size, local_projection].
    return (int(attn.hidden_size), int(attn.local_projection_size))


def _mla_o_proj_shape(attn: nn.Module) -> tuple[int, int]:
    # RowParallelLinear(num_heads * v_head_dim, hidden_size).
    return (int(attn.hidden_size), int(attn.num_local_heads) * int(attn.v_head_dim))


def _mla_q_b_proj_shape(attn: nn.Module) -> tuple[int, int]:
    # ColumnParallelLinear(q_lora_rank, num_heads * qk_head_dim).
    return (
        int(attn.num_local_heads) * int(attn.qk_head_dim),
        int(attn.q_lora_rank),
    )


def _mla_q_proj_shape(attn: nn.Module) -> tuple[int, int]:
    # ColumnParallelLinear(proj_input_size, num_heads * qk_head_dim). The class
    # keeps only hidden_size; a distinct proj_input_size makes this mismatch and
    # refuse, which is the intended outcome.
    return (
        int(attn.num_local_heads) * int(attn.qk_head_dim),
        int(attn.hidden_size),
    )


def _mla_kv_a_proj_with_mqa_shape(attn: nn.Module) -> tuple[int, int]:
    # ReplicatedLinear(proj_input_size, kv_lora_rank + qk_rope_head_dim).
    return (
        int(attn.kv_lora_rank) + int(attn.qk_rope_head_dim),
        int(attn.hidden_size),
    )


def _mla_fused_qkv_a_proj_shape(attn: nn.Module) -> tuple[int, int]:
    # DeepSeekV2FusedQkvAProjLinear(proj_input_size,
    #     [q_lora_rank, kv_lora_rank + qk_rope_head_dim]). The installed class
    # decides whether the q_a half is sharded; the replicated layout is declared
    # here and a sharded one refuses instead of converting.
    return (
        int(attn.q_lora_rank)
        + int(attn.kv_lora_rank)
        + int(attn.qk_rope_head_dim),
        int(attn.hidden_size),
    )


@dataclass(frozen=True)
class FamilyPlan:
    """One weight family: where it lives, how many there are, what shape it is."""

    name: str
    layer_kind: str
    attribute: str
    expected_modules: int
    expected_shape: Callable[[nn.Module], tuple[int, int]]
    #: ``None`` applies to every MLA layout; otherwise the required layout.
    mla_layout: str | None = None


#: The candidate bundle. Deliberately excluded, with reasons in the README:
#: the KDA f/g gates, the MLA kv_b_proj, lm_head and the DFlash2 drafter.
FAMILIES: tuple[FamilyPlan, ...] = (
    FamilyPlan("kda_o_proj", "kda", "o_proj", 34, _kda_o_proj_shape),
    FamilyPlan("mla_o_proj", "mla", "o_proj", 11, _mla_o_proj_shape),
    FamilyPlan(
        "mla_fused_qkv_a_proj", "mla", "fused_qkv_a_proj", 11,
        _mla_fused_qkv_a_proj_shape, "q_lora",
    ),
    FamilyPlan("mla_q_b_proj", "mla", "q_b_proj", 11, _mla_q_b_proj_shape, "q_lora"),
    FamilyPlan(
        "mla_kv_a_proj_with_mqa", "mla", "kv_a_proj_with_mqa", 11,
        _mla_kv_a_proj_with_mqa_shape, "no_q_lora",
    ),
    FamilyPlan("mla_q_proj", "mla", "q_proj", 11, _mla_q_proj_shape, "no_q_lora"),
)


def enabled(environ: dict | None = None) -> bool:
    """Off unless :data:`ENABLE_ENV` is a non-zero integer."""
    env = os.environ if environ is None else environ
    raw = env.get(ENABLE_ENV, "0").strip()
    if raw == "":
        return False
    try:
        return int(raw) != 0
    except ValueError:
        raise RuntimeError(f"E21 {ENABLE_ENV} must be an integer, got {raw!r}") from None


@torch.library.custom_op(
    "e21_tuning::dequant_marlin_bf16_linear_shaped", mutates_args=("scratch",)
)
def dequant_marlin_bf16_linear_shaped(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    inverse_weight_perm: torch.Tensor,
    scratch: torch.Tensor,
    size_n: int,
    size_k: int,
) -> torch.Tensor:
    # One opaque op includes both the scratch write and its GEMM read. The
    # declared mutation preserves the ordering between families sharing a shape.
    _dequant_into(qweight, scales, inverse_weight_perm, scratch, size_n, size_k)
    return F.linear(x, scratch)


@dequant_marlin_bf16_linear_shaped.register_fake
def _dequant_marlin_bf16_linear_shaped_fake(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    inverse_weight_perm: torch.Tensor,
    scratch: torch.Tensor,
    size_n: int,
    size_k: int,
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], size_n))


def _dequant_into(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    inverse_weight_perm: torch.Tensor,
    scratch: torch.Tensor,
    size_n: int,
    size_k: int,
) -> None:
    padded_n = qweight.shape[1] // 4
    _dequant_nk_kernel[(triton.cdiv(size_n * size_k, 1024),)](
        qweight,
        scales,
        inverse_weight_perm,
        scratch,
        PADDED_N=padded_n,
        SIZE_N=size_n,
        SIZE_K=size_k,
        BLOCK_SIZE=1024,
        num_warps=4,
    )


class ShapedBF16Scratch:
    """One BF16 ``[size_n, size_k]`` matrix shared serially by one shape's modules.

    Contents are overwritten on every call, so never retain the returned tensor
    as if it belonged to one module. Share only on the same CUDA stream.
    """

    def __init__(
        self,
        size_n: int,
        size_k: int,
        device: torch.device | str,
        inverse_weight_perm: torch.Tensor,
    ):
        self.size_n = size_n
        self.size_k = size_k
        self.inverse_weight_perm = inverse_weight_perm
        self.scratch = torch.empty((size_n, size_k), device=device, dtype=torch.bfloat16)

    @property
    def resident_bytes(self) -> int:
        return self.scratch.numel() * 2

    def check_storage(self, qweight: torch.Tensor, scales: torch.Tensor) -> None:
        padded_n = qweight.shape[1] // 4
        if (
            qweight.shape[0] != self.size_k // MARLIN_TILE
            or qweight.shape[1] != padded_n * 4
            or padded_n < self.size_n
            or padded_n % 64
            or tuple(scales.shape) != (self.size_k // GROUP_SIZE, padded_n)
            or qweight.dtype != torch.int32
            or scales.dtype != torch.bfloat16
            or not qweight.is_contiguous()
            or not scales.is_contiguous()
            or qweight.device != self.scratch.device
            or scales.device != self.scratch.device
        ):
            raise RuntimeError(
                "E21 scratch dequantization requires the E20 Marlin storage contract"
            )

    def dequantize(self, qweight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """Probe-only entry point; returns borrowed scratch for exact comparison."""
        self.check_storage(qweight, scales)
        _dequant_into(
            qweight, scales, self.inverse_weight_perm, self.scratch,
            self.size_n, self.size_k,
        )
        return self.scratch

    def linear(
        self, x: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor
    ) -> torch.Tensor:
        return dequant_marlin_bf16_linear_shaped(
            x,
            qweight,
            scales,
            self.inverse_weight_perm,
            self.scratch,
            self.size_n,
            self.size_k,
        )


class ResidueW8A16Method(LinearMethodBase):
    """Per-family W8A16 apply path. ``size_k`` is load-bearing at runtime."""

    def __init__(self, size_n: int, size_k: int, scratch: ShapedBF16Scratch | None = None):
        self.size_n = size_n
        self.size_k = size_k
        self.scratch = scratch

    def create_weights(self, layer: nn.Module, *args, **kwargs) -> None:
        raise RuntimeError("E21 attaches after the original BF16 loader finishes")

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens = x.numel() // self.size_k
        if self.scratch is not None and num_tokens >= PREFILL_BF16_MIN_TOKENS:
            return self.scratch.linear(x, layer.e21_qweight, layer.e21_scales)
        return apply_gptq_marlin_linear(
            input=x,
            weight=layer.e21_qweight,
            weight_scale=layer.e21_scales,
            weight_zp=layer.e21_empty,
            g_idx=layer.e21_empty,
            g_idx_sort_indices=layer.e21_empty,
            workspace=layer.e21_workspace,
            wtype=WTYPE,
            output_size_per_partition=self.size_n,
            input_size_per_partition=self.size_k,
            is_k_full=True,
            bias=bias,
            use_fp32_reduce=True,
            input_dtype=None,
        )


@torch.no_grad()
def _pack_projection(
    module: nn.Module,
    size_n: int,
    size_k: int,
    scratch: ShapedBF16Scratch | None = None,
) -> dict:
    weight_kn = module.weight.T.contiguous()
    reference, quantized, scales, _, _ = gptq_quantize_weights(
        weight_kn, WTYPE, GROUP_SIZE, act_order=False
    )
    del reference, weight_kn
    packed = pack_quantized_values_into_int32(
        quantized, WTYPE, packed_dim=0
    ).contiguous()
    del quantized
    padded_n, padded_k = marlin_padded_nk(size_n, size_k, GROUP_SIZE)
    padded_n = ((padded_n + 255) // 256) * 256
    if padded_k != size_k:
        raise RuntimeError(
            f"E21 requires an unpadded K; Marlin padded {size_k} to {padded_k}"
        )
    packed = marlin_pad_qweight(packed, size_n, size_k, padded_n, padded_k)
    empty = torch.empty(0, dtype=torch.int32, device=module.weight.device)
    repacked = ops.gptq_marlin_repack(
        packed, empty, padded_k, padded_n, WTYPE.size_bits
    )
    del packed
    scales = marlin_permute_scales(
        marlin_pad_scales(
            scales, size_n, size_k, padded_n, padded_k, GROUP_SIZE
        ),
        size_k=padded_k,
        size_n=padded_n,
        group_size=GROUP_SIZE,
    )
    workspace = marlin_make_workspace_new(module.weight.device)
    for name, value in (
        ("e21_qweight", repacked),
        ("e21_scales", scales),
        ("e21_empty", empty),
        ("e21_workspace", workspace),
    ):
        module.register_buffer(name, value, persistent=False)
    before = module.weight.numel() * module.weight.element_size()
    after = sum(
        t.numel() * t.element_size()
        for t in (repacked, scales, empty, workspace)
    )
    module.quant_method = ResidueW8A16Method(size_n, size_k, scratch)
    del module._parameters["weight"]
    return {
        "original_bytes": before,
        "packed_bytes": after,
        "padded_n": padded_n,
        "padded_k": padded_k,
    }


def _layers_by_kind(model: nn.Module) -> dict[str, list[tuple[int, nn.Module]]]:
    by_kind: dict[str, list[tuple[int, nn.Module]]] = {
        kind: [] for kind in EXPECTED_LAYERS
    }
    for index, layer in enumerate(model.layers):
        bucket = by_kind.get(getattr(layer, "layer_kind", None))
        if bucket is not None:
            bucket.append((index, layer.self_attn))
    for kind, expected in EXPECTED_LAYERS.items():
        if len(by_kind[kind]) != expected:
            raise RuntimeError(
                f"E21 expected {expected} {kind} layers, got {len(by_kind[kind])}"
            )
    return by_kind


def _mla_layout(mla_layers: list[tuple[int, nn.Module]]) -> str:
    fused = [
        index
        for index, attn in mla_layers
        if getattr(attn, "q_lora_rank", None) is not None
    ]
    if len(fused) == len(mla_layers):
        return "q_lora"
    if not fused:
        return "no_q_lora"
    raise RuntimeError(
        "E21 requires one uniform MLA projection layout; "
        f"{len(fused)} of {len(mla_layers)} MLA layers use q_lora_rank"
    )


def _expected_shape(plan: FamilyPlan, index: int, attn: nn.Module) -> tuple[int, int]:
    try:
        size_n, size_k = plan.expected_shape(attn)
    except AttributeError as error:
        raise RuntimeError(
            f"E21 family {plan.name} cannot derive its shape at layer {index}: {error}"
        ) from None
    if size_n <= 0 or size_k <= 0 or size_n % MARLIN_TILE or size_k % GROUP_SIZE:
        raise RuntimeError(
            f"E21 family {plan.name} derived shape ({size_n}, {size_k}) at layer "
            f"{index} does not satisfy the Marlin group-{GROUP_SIZE} contract"
        )
    return (size_n, size_k)


def _validate(
    plan: FamilyPlan, index: int, module: nn.Module, expected: tuple[int, int]
) -> None:
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
            f"E21 family {plan.name} projection differs from the expected "
            f"{expected} BF16 contract at layer {index}"
        )


def plan_conversion(model: nn.Module) -> tuple[list[tuple], str, list[tuple[int, int]]]:
    """Resolve and validate every family before any weight is changed.

    Returns the per-module work list, the resolved MLA layout and the distinct
    shapes needing a scratch buffer.
    """
    by_kind = _layers_by_kind(model)
    layout = _mla_layout(by_kind["mla"])
    work: list[tuple] = []
    for plan in FAMILIES:
        if plan.mla_layout is not None and plan.mla_layout != layout:
            continue
        found = 0
        for index, attn in by_kind[plan.layer_kind]:
            module = getattr(attn, plan.attribute, None)
            if module is None:
                raise RuntimeError(
                    f"E21 family {plan.name} is absent on {plan.layer_kind} layer {index}"
                )
            expected = _expected_shape(plan, index, attn)
            _validate(plan, index, module, expected)
            work.append((plan, index, module, expected))
            found += 1
        if found != plan.expected_modules:
            raise RuntimeError(
                f"E21 family {plan.name} expected {plan.expected_modules} modules, got {found}"
            )
    shapes = sorted({expected for _, _, _, expected in work})
    return work, layout, shapes


@torch.no_grad()
def finalize_bf16_residue_w8a16(
    model: nn.Module,
    inverse_weight_perm: torch.Tensor | None = None,
    environ: dict | None = None,
) -> None:
    """Convert the residue families after the accepted E20 pass, if enabled."""
    if not enabled(environ):
        return
    if getattr(model, "_e21_bf16_residue_ready", False):
        return
    work, layout, shapes = plan_conversion(model)
    budget = sum(size_n * size_k * 2 for size_n, size_k in shapes)
    if budget > SCRATCH_BUDGET_BYTES:
        raise RuntimeError(
            f"E21 needs {budget} additional scratch bytes for shapes {shapes}, "
            f"above the {SCRATCH_BUDGET_BYTES}-byte ceiling"
        )
    device = work[0][2].weight.device
    if inverse_weight_perm is None:
        perm = get_weight_perm(8, is_a_8bit=False)
        if perm.numel() != 1024:
            raise RuntimeError("Native Marlin weight permutation changed")
        inverse_weight_perm = torch.argsort(perm).to(device=device, dtype=torch.int32)
    scratches = {
        shape: ShapedBF16Scratch(shape[0], shape[1], device, inverse_weight_perm)
        for shape in shapes
    }
    for (size_n, size_k), owner in scratches.items():
        model.register_buffer(
            f"e21_bf16_scratch_{size_n}x{size_k}", owner.scratch, persistent=False
        )
    receipts = []
    for plan, index, module, expected in work:
        receipts.append(
            {
                "family": plan.name,
                "layer": index,
                "shape_nk": list(expected),
                **_pack_projection(module, expected[0], expected[1], scratches[expected]),
            }
        )
    model._e21_bf16_residue_ready = True
    # Release only unused conversion allocations before KV and graph allocation.
    torch.cuda.empty_cache()
    logger.info(
        "E21_BF16_RESIDUE_W8A16_READY %s",
        json.dumps(
            {
                "modules": len(receipts),
                "families": sorted({r["family"] for r in receipts}),
                "mla_layout": layout,
                "prefill_bf16_min_tokens": PREFILL_BF16_MIN_TOKENS,
                "group_size": GROUP_SIZE,
                "distinct_shapes_nk": [list(shape) for shape in shapes],
                "added_scratch_bytes": budget,
                "scratch_budget_bytes": SCRATCH_BUDGET_BYTES,
                "original_bytes": sum(r["original_bytes"] for r in receipts),
                "packed_bytes": sum(r["packed_bytes"] for r in receipts),
                "receipts": receipts,
            },
            sort_keys=True,
        ),
    )
