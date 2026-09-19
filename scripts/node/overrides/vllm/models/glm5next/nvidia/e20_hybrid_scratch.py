# SPDX-License-Identifier: Apache-2.0
"""E20 operator-probe candidate: existing INT8 Marlin storage -> BF16 GEMM.

Not installed in serving. Supports only uint8b128, group size 128, BF16
activations/scales, no activation ordering, and the E20 KDA matrix dimensions.
The caller owns one scratch instance shared serially by all 34 projections.
Small-M decode should continue through the original E20 Marlin method unchanged.
"""

import torch
from torch.nn import functional as F

from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
    get_weight_perm,
)
from vllm.triton_utils import tl, triton


N = 6288
K = 4096
GROUP_SIZE = 128


@triton.jit
def _dequant_nk_kernel(
    qweight,
    scales,
    inverse_weight_perm,
    output,
    PADDED_N: tl.constexpr,
    SIZE_N: tl.constexpr,
    SIZE_K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Invert the native 16x64 Marlin tile and uint8b128 representation."""
    linear = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = linear < SIZE_N * SIZE_K
    n = linear // SIZE_K
    k = linear % SIZE_K

    # Original [K,N] -> reshape(K/16,16,N/16,16) -> permute(0,2,1,3).
    tiled = (
        (k // 16) * (PADDED_N * 16)
        + (n // 16) * 256
        + (k % 16) * 16
        + n % 16
    )
    # Native repack applies the 1024-element forward permutation before packing
    # consecutive uint8 values into int32. Invert to read a logical (k,n).
    inverse = tl.load(inverse_weight_perm + tiled % 1024)
    packed_linear = (tiled // 1024) * 1024 + inverse
    word = tl.load(qweight + packed_linear // 4, mask=valid, other=0)
    quantized = ((word >> ((packed_linear % 4) * 8)) & 255) - 128

    # Grouped scales use an 8x8 transpose, which is its own inverse.
    group_scale_linear = (k // 128) * PADDED_N + n
    scale_inner = group_scale_linear % 64
    permuted_scale_linear = (
        (group_scale_linear // 64) * 64
        + (scale_inner % 8) * 8
        + scale_inner // 8
    )
    scale = tl.load(scales + permuted_scale_linear, mask=valid, other=0)
    dequantized = quantized.to(tl.float32) * scale.to(tl.float32)
    tl.store(output + linear, dequantized.to(tl.bfloat16), mask=valid)


def _dequant_into(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    inverse_weight_perm: torch.Tensor,
    scratch: torch.Tensor,
) -> None:
    padded_n = qweight.shape[1] // 4
    _dequant_nk_kernel[(triton.cdiv(N * K, 1024),)](
        qweight,
        scales,
        inverse_weight_perm,
        scratch,
        PADDED_N=padded_n,
        SIZE_N=N,
        SIZE_K=K,
        BLOCK_SIZE=1024,
        num_warps=4,
    )


@torch.library.custom_op("e20_tuning::dequant_marlin_bf16_linear", mutates_args=("scratch",))
def dequant_marlin_bf16_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    inverse_weight_perm: torch.Tensor,
    scratch: torch.Tensor,
) -> torch.Tensor:
    # One opaque op includes both the scratch write and its GEMM read. The
    # declared mutation preserves the ordering between layers sharing scratch.
    _dequant_into(qweight, scales, inverse_weight_perm, scratch)
    return F.linear(x, scratch)


@dequant_marlin_bf16_linear.register_fake
def _dequant_marlin_bf16_linear_fake(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    inverse_weight_perm: torch.Tensor,
    scratch: torch.Tensor,
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], N))


class MarlinBF16Scratch:
    """Probe owner for a 49.125 MiB BF16 tensor plus a 4 KiB inverse map.

    Share serially on the same CUDA stream. The matrix contents are overwritten
    on every call, so never retain/reuse scratch as if it belonged to one layer.
    Prepare and warm the op before graph capture; this class allocates once.
    """

    def __init__(self, device: torch.device | str):
        perm = get_weight_perm(8, is_a_8bit=False)
        if perm.numel() != 1024:
            raise RuntimeError("Native Marlin weight permutation changed")
        self.inverse_weight_perm = torch.argsort(perm).to(device=device, dtype=torch.int32)
        self.scratch = torch.empty((N, K), device=device, dtype=torch.bfloat16)

    @property
    def resident_bytes(self) -> int:
        return self.scratch.numel() * 2 + self.inverse_weight_perm.numel() * 4

    def dequantize(self, qweight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """Probe-only entry point; returns borrowed scratch for exact comparison."""
        self.check_storage(qweight, scales)
        _dequant_into(qweight, scales, self.inverse_weight_perm, self.scratch)
        return self.scratch

    def check_storage(self, qweight: torch.Tensor, scales: torch.Tensor) -> None:
        padded_n = qweight.shape[1] // 4
        if (
            qweight.shape[0] != K // 16
            or qweight.shape[1] != padded_n * 4
            or padded_n < N
            or padded_n % 64
            or tuple(scales.shape) != (K // GROUP_SIZE, padded_n)
            or qweight.dtype != torch.int32
            or scales.dtype != torch.bfloat16
            or not qweight.is_contiguous()
            or not scales.is_contiguous()
            or qweight.device != self.scratch.device
            or scales.device != self.scratch.device
        ):
            raise RuntimeError("Scratch dequantization requires the E20 Marlin storage contract")

    def linear(self, x: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        return dequant_marlin_bf16_linear(
            x, qweight, scales, self.inverse_weight_perm, self.scratch
        )
