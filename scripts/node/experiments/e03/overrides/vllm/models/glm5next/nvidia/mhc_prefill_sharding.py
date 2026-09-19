# SPDX-License-Identifier: Apache-2.0
# Adapted from FujitsuPolycom/SparkRing at 61f277bd0c97fbff892668e12ea04a330a45fa01.
# Local change: qualify TP4/DCP1 on R10; preserve the native TP communicator.
"""Experimental GLM eager-prefill ownership; no persistent decode state."""
from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

_LOG = logging.getLogger(__name__)
_REPORTS = 0
# R10 reserves five DFlash2 slots in the 8192-token budget; its unchanged
# 2304-token checkpoint alignment makes ordinary full prefill chunks 6912 rows.
PREFILL_ROWS = 6912


def configure(model: Any, config: Any, enabled: bool) -> None:
    """Agree on the opt-in during construction, before any graph is captured."""
    import torch
    from vllm.distributed import get_tp_group
    group = get_tp_group()
    votes = [None] * group.world_size
    torch.distributed.all_gather_object(votes, enabled, group=group.cpu_group)
    if any(vote != enabled for vote in votes):
        raise RuntimeError("All TP ranks must agree on SPARK_MHC_PREFILL_SHARD")
    if enabled:
        captures = config.compilation_config.cudagraph_capture_sizes or []
        if (group.world_size != 4 or config.scheduler_config.max_num_batched_tokens != 8192
                or any(size >= PREFILL_ROWS for size in captures)):
            raise RuntimeError("mHC prototype requires TP4, an 8K ceiling, and decode-only graph sizes")
    model._mhc_prefill_enabled = enabled
    model._mhc_prefill_parallel_config = config.parallel_config


def pure_prefill_metadata(metadata: Any, names: tuple[str, ...], rows: int) -> bool:
    """Use host counts only; missing or ambiguous metadata cannot opt in."""
    if not isinstance(metadata, dict) or not names or rows != PREFILL_ROWS:
        return False
    fields = ("num_decodes", "num_decode_tokens", "num_spec_decodes", "num_spec_decode_tokens")
    for name in names:
        item = metadata.get(name)
        if item is None:
            return False
        if type(getattr(item, "num_prefills", None)) is not int or item.num_prefills <= 0:
            return False
        if type(getattr(item, "num_prefill_tokens", None)) is not int or item.num_prefill_tokens != rows:
            return False
        if any(type(getattr(item, field, None)) is not int or getattr(item, field) != 0 for field in fields):
            return False
    return True


def validate_moe_deferral(runner: Any) -> None:
    config = runner.moe_config
    parallel = config.moe_parallel_config
    if (config.tp_size != 4 or config.dp_size != 1 or config.ep_size != 1
            or config.pcp_size != 1 or config.is_sequence_parallel
            or config.skip_final_all_reduce or parallel.use_all2all_kernels
            or runner._fused_output_is_reduced
            or runner.routed_output_transform is not None
            or runner.routed_input_transform is not None
            or type(runner.router).__name__ == "ZeroExpertRouter"):
        raise RuntimeError("mHC prefill requires one unreduced conventional TP4 MoE output")


def validate_model(model: Any) -> tuple[str, ...]:
    from vllm.model_executor.layers.linear import RowParallelLinear
    from vllm.model_executor.layers.mla import MultiHeadLatentAttentionWrapper
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
    from .attention import Glm5NextMLAAttention
    from .kda import Glm5NextLinearAttention

    config = model._mhc_prefill_parallel_config
    if (config.tensor_parallel_size != 4 or config.decode_context_parallel_size != 1
            or config.pipeline_parallel_size != 1 or config.data_parallel_size != 1
            or config.prefill_context_parallel_size != 1 or config.enable_expert_parallel
            or config.enable_eplb or model.is_sequence_parallel):
        raise RuntimeError("mHC prefill requires the qualified TP4/DCP1/PP1/DP1 layout")
    names = []
    for layer in model._active_layers:
        if not layer.mhc or layer.is_mtp_layer or layer._b12x_mhc is None:
            raise RuntimeError("mHC prefill requires B12X mHC on every base layer")
        attn = layer.self_attn
        if type(attn) is Glm5NextLinearAttention:
            names.append(attn.prefix)
        elif type(attn) is Glm5NextMLAAttention:
            if type(attn.mla_attn) is not MultiHeadLatentAttentionWrapper:
                raise RuntimeError("Unqualified outer MLA wrapper")
        else:
            raise RuntimeError("Unqualified GLM attention implementation")
        projection = attn.o_proj
        if (type(projection) is not RowParallelLinear or projection.tp_size != 4
                or not projection.reduce_results or projection.bias is not None):
            raise RuntimeError("Unqualified attention projection reduction")
        if layer._mlp_is_moe:
            if type(layer.mlp.experts) is not MoERunner:
                raise RuntimeError("Unqualified MoE runner")
            validate_moe_deferral(layer.mlp.experts)
        elif (type(layer.mlp.down_proj) is not RowParallelLinear
                or layer.mlp.down_proj.tp_size != 4
                or not layer.mlp.down_proj.reduce_results
                or layer.mlp.down_proj.bias is not None):
            raise RuntimeError("Unqualified dense FFN reduction")
    if not names:
        raise RuntimeError("mHC prefill requires explicit GDN metadata owners")
    return tuple(names)


@dataclass
class PrefillOwnership:
    comm: Any
    rank: int
    rows: int = PREFILL_ROWS
    rs_count: int = 0
    ag_count: int = 0

    def local_view(self, tensor: Any) -> Any:
        if tensor.shape[0] != self.rows:
            raise RuntimeError("mHC full-to-owner row count mismatch")
        q = self.rows // 4
        return tensor.narrow(0, self.rank * q, q)

    def _check(self, tensor: Any, expected_rows: int) -> None:
        import torch
        if not self.comm.available or self.comm.disabled:
            raise RuntimeError("mHC PyNccl communicator became unavailable")
        if (tensor.shape[0] != expected_rows or not tensor.is_cuda
                or tensor.device != self.comm.device
                or tensor.dtype != torch.bfloat16 or not tensor.is_contiguous()):
            raise RuntimeError("mHC collective requires contiguous BF16 owner/full rows")

    def reduce_scatter(self, partial: Any) -> Any:
        import torch
        self._check(partial, self.rows)
        if tuple(partial.shape) != (self.rows, 4096):
            raise RuntimeError("mHC reduce-scatter expects a full hidden TP partial")
        output = partial.new_empty((self.rows // 4, 4096))
        stream = torch.cuda.current_stream(partial.device)
        self.comm.reduce_scatter(output, partial, stream=stream)
        partial.record_stream(stream)
        output.record_stream(stream)
        self.rs_count += 1
        return output

    def all_gather(self, owned: Any) -> Any:
        import torch
        self._check(owned, self.rows // 4)
        output = owned.new_empty((self.rows, *owned.shape[1:]))
        stream = torch.cuda.current_stream(owned.device)
        self.comm.all_gather(output, owned, stream=stream)
        owned.record_stream(stream)
        output.record_stream(stream)
        self.ag_count += 1
        return output

    def finish(self, layers: int, auxiliary_gathers: int) -> None:
        global _REPORTS
        if self.rs_count != 2 * layers or self.ag_count != 2 * layers + auxiliary_gathers:
            raise RuntimeError(f"mHC collective accounting mismatch: RS={self.rs_count} AG={self.ag_count}")
        if _REPORTS < 8:
            _LOG.warning("SPARK_MHC_PREFILL rank=%d rows=%d owner_rows=%d rs=%d ag=%d aux=%d",
                         self.rank, self.rows, self.rows // 4, self.rs_count, self.ag_count, auxiliary_gathers)
            _REPORTS += 1


def maybe_create(model: Any, hidden: Any, positions: Any) -> PrefillOwnership | None:
    if not model._mhc_prefill_enabled or tuple(hidden.shape) != (PREFILL_ROWS, 4096):
        return None
    import torch
    if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
        return None
    from vllm.distributed import get_tp_group
    from vllm.forward_context import get_forward_context, is_forward_context_available
    if not is_forward_context_available():
        return None
    context = get_forward_context()
    if context.cudagraph_runtime_mode.name != "NONE" or context.ubatch_slices is not None:
        return None
    group = get_tp_group()
    if group.world_size != 4:
        raise RuntimeError("mHC prefill TP group is not four ranks")
    comm = getattr(group.device_communicator, "pynccl_comm", None)
    error = None
    try:
        if comm is None or not comm.available or comm.disabled or comm.world_size != 4:
            raise RuntimeError("mHC prefill requires the enabled TP PyNccl communicator")
        if comm.rank != group.rank_in_group or comm.device != hidden.device:
            raise RuntimeError("mHC communicator rank/device ownership mismatch")
        names = validate_model(model)
    except (AttributeError, RuntimeError) as exc:
        names = ()
        error = str(exc)
    eligible = (error is None and positions.shape[0] == PREFILL_ROWS
                and hidden.is_cuda and hidden.dtype == torch.bfloat16
                and pure_prefill_metadata(context.attn_metadata, names, PREFILL_ROWS))
    # Small host vote before changing ownership; no GPU metadata synchronization.
    # Rank-local capability or metadata differences cannot silently choose
    # incompatible reduction paths after one rank has produced partial output.
    votes = [None] * 4
    torch.distributed.all_gather_object(votes, (eligible, error), group=group.cpu_group)
    errors = [item[1] for item in votes if item[1] is not None]
    if errors:
        raise RuntimeError("mHC prefill capability vote failed: " + repr(errors))
    if not all(item[0] for item in votes):
        return None
    return PrefillOwnership(comm=comm, rank=group.rank_in_group)
