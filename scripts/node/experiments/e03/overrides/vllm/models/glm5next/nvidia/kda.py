# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3 KDA modeling adapter."""

import torch

from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
    KimiGatedDeltaNetAttention,
)


class Glm5NextLinearAttention(KimiGatedDeltaNetAttention):
    """Adapt the shared out-buffer KDA layer to GLM's tensor-returning block."""

    enable_b12x_kda_decode = True
    b12x_kda_null_state_index = 0

    def forward(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        *,
        defer_tp_reduction: bool = False,
    ) -> torch.Tensor:
        output = torch.empty_like(hidden_states)
        if defer_tp_reduction:
            super().forward(hidden_states, positions, output, defer_tp_reduction=True)
        else:
            super().forward(hidden_states, positions, output)
        return output


__all__ = ["Glm5NextLinearAttention"]
