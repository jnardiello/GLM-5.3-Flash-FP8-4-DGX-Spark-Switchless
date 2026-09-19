# SPDX-License-Identifier: Apache-2.0
"""E20 candidate: post-load W8A16 for the 34 GLM KDA input projections.

Uses the installed vLLM quantization, packing and Marlin APIs. The post-load
integration follows JSpark3's Apache-2.0 trunk overlay (revision recorded in
the preparation archive); selection and TP4 dimensions are specific to F1.
"""

import json

import torch
from torch import nn

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
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    gptq_quantize_weights,
    pack_quantized_values_into_int32,
)
from vllm.scalar_type import scalar_types


logger = init_logger(__name__)
GROUP_SIZE = 128
INPUT_SIZE = 4096
OUTPUT_SIZE = 6288
WTYPE = scalar_types.uint8b128


PREFILL_BF16_MIN_TOKENS = 2048


class KDAInputW8A16Method(LinearMethodBase):
    def __init__(self, scratch=None):
        self.scratch = scratch

    def create_weights(self, layer: nn.Module, *args, **kwargs) -> None:
        raise RuntimeError("E20 attaches after the original BF16 loader finishes")

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens = x.numel() // INPUT_SIZE
        if self.scratch is not None and num_tokens >= PREFILL_BF16_MIN_TOKENS:
            return self.scratch.linear(x, layer.e20_qweight, layer.e20_scales)
        return apply_gptq_marlin_linear(
            input=x,
            weight=layer.e20_qweight,
            weight_scale=layer.e20_scales,
            weight_zp=layer.e20_empty,
            g_idx=layer.e20_empty,
            g_idx_sort_indices=layer.e20_empty,
            workspace=layer.e20_workspace,
            wtype=WTYPE,
            output_size_per_partition=OUTPUT_SIZE,
            input_size_per_partition=INPUT_SIZE,
            is_k_full=True,
            bias=bias,
            use_fp32_reduce=True,
            input_dtype=None,
        )


@torch.no_grad()
def _pack_projection(layer: nn.Module, scratch=None) -> dict:
    weight_kn = layer.weight.T.contiguous()
    reference, quantized, scales, _, _ = gptq_quantize_weights(
        weight_kn, WTYPE, GROUP_SIZE, act_order=False
    )
    del reference, weight_kn
    packed = pack_quantized_values_into_int32(
        quantized, WTYPE, packed_dim=0
    ).contiguous()
    del quantized
    padded_n, padded_k = marlin_padded_nk(
        OUTPUT_SIZE, INPUT_SIZE, GROUP_SIZE
    )
    padded_n = ((padded_n + 255) // 256) * 256
    packed = marlin_pad_qweight(
        packed, OUTPUT_SIZE, INPUT_SIZE, padded_n, padded_k
    )
    empty = torch.empty(0, dtype=torch.int32, device=layer.weight.device)
    repacked = ops.gptq_marlin_repack(
        packed, empty, padded_k, padded_n, WTYPE.size_bits
    )
    del packed
    scales = marlin_permute_scales(
        marlin_pad_scales(
            scales, OUTPUT_SIZE, INPUT_SIZE, padded_n, padded_k, GROUP_SIZE
        ),
        size_k=padded_k,
        size_n=padded_n,
        group_size=GROUP_SIZE,
    )
    workspace = marlin_make_workspace_new(layer.weight.device)
    for name, value in (
        ("e20_qweight", repacked),
        ("e20_scales", scales),
        ("e20_empty", empty),
        ("e20_workspace", workspace),
    ):
        layer.register_buffer(name, value, persistent=False)
    before = layer.weight.numel() * layer.weight.element_size()
    after = sum(
        t.numel() * t.element_size()
        for t in (repacked, scales, empty, workspace)
    )
    layer.quant_method = KDAInputW8A16Method(scratch)
    del layer._parameters["weight"]
    return {
        "original_bytes": before,
        "packed_bytes": after,
        "padded_n": padded_n,
        "padded_k": padded_k,
    }


@torch.no_grad()
def finalize_kda_input_w8a16(model: nn.Module) -> None:
    """Convert loaded TP-local weights before the first compiled forward."""
    if getattr(model, "_e20_kda_input_w8a16_ready", False):
        return
    selected = [
        (index, layer.self_attn.in_proj_qkvgfab)
        for index, layer in enumerate(model.layers)
        if layer.layer_kind == "kda"
    ]
    if len(selected) != 34:
        raise RuntimeError(f"E20 expected 34 KDA input projections, got {len(selected)}")
    # Validate the intended tensors before changing any projection.
    for index, layer in selected:
        weight = layer.weight
        if (
            tuple(weight.shape) != (OUTPUT_SIZE, INPUT_SIZE)
            or weight.dtype != torch.bfloat16
            or weight.device.type != "cuda"
            or not isinstance(layer.quant_method, UnquantizedLinearMethod)
            or layer.bias is not None
        ):
            raise RuntimeError(f"E20 KDA input projection differs from F1 at layer {index}")
    from .e20_hybrid_scratch import MarlinBF16Scratch

    scratch = MarlinBF16Scratch(selected[0][1].weight.device)
    model.register_buffer("e20_shared_bf16_scratch", scratch.scratch, persistent=False)
    model.register_buffer("e20_inverse_weight_perm", scratch.inverse_weight_perm, persistent=False)
    receipts = []
    for index, layer in selected:
        receipts.append({"layer": index, **_pack_projection(layer, scratch)})
    model._e20_kda_input_w8a16_ready = True
    # Release only unused conversion allocations before KV and graph allocation.
    torch.cuda.empty_cache()
    logger.info(
        "E20_KDA_INPUT_W8A16_READY %s",
        json.dumps(
            {
                "modules": len(receipts),
                "variant": "pad6400-bf16-scratch",
                "prefill_bf16_min_tokens": PREFILL_BF16_MIN_TOKENS,
                "shared_scratch_bytes": scratch.resident_bytes,
                "shape_nk": [OUTPUT_SIZE, INPUT_SIZE],
                "group_size": GROUP_SIZE,
                "original_bytes": sum(r["original_bytes"] for r in receipts),
                "packed_bytes": sum(r["packed_bytes"] for r in receipts),
                "receipts": receipts,
            },
            sort_keys=True,
        ),
    )
