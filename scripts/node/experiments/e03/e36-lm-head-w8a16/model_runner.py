# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
NOTE: Coding style guide for this file:
This model runner is shared by all models: text and multimodal, generative
and embedding, public and private. As a result, this file must only contain
code that is common to every model. Model-specific behavior belongs in the
appropriate model-specific files.

In other words:
* Be paranoid about changing this file. It should remain stable.
* Be even more paranoid about adding new lines. It should remain minimal.

Even for shared features (for example, different parallelism modes), keep the
complexity out of this path. The less common the feature, the more it should be
hidden. Prefer utility functions defined elsewhere and call them from here,
instead of embedding feature-specific logic directly.
"""

import functools
import gc
import time
from copy import deepcopy
from typing import Any, NamedTuple

import numpy as np
import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.compilation.counter import compilation_counter
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import (
    get_dcp_group,
    get_pp_group,
)
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.all2all_utils import get_ep_all2all_manager
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
    bind_routed_experts_capturer,
)
from vllm.model_executor.layers.mamba.ops.ssu_dispatch import (
    initialize_mamba_ssu_backend,
)
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.offloader import (
    create_offloader,
    get_offloader,
    set_offloader,
)
from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.encoder_budget import (
    MultiModalBudget,
    get_dummy_encoder_profile_inputs,
)
from vllm.sequence import IntermediateTensors
from vllm.tasks import SupportedTask
from vllm.utils.mem_utils import DeviceMemoryProfiler, format_gib
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec
from vllm.v1.outputs import (
    DraftTokenIds,
    ECConnectorOutput,
    ModelRunnerOutput,
    RoutedExpertsTensors,
)
from vllm.v1.worker.block_table import get_block_table_width
from vllm.v1.worker.cp_utils import check_attention_cp_compatibility
from vllm.v1.worker.gpu import pcp_manager as pcp
from vllm.v1.worker.gpu.async_utils import (
    AsyncOutput,
    AsyncPoolingOutput,
    StepTimingCollector,
)
from vllm.v1.worker.gpu.attn_utils import (
    build_slot_mappings_by_layer,
    get_kv_cache_spec,
    init_attn_backend,
    init_kv_cache,
)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.buffer_utils import (
    async_copy_to_gpu,
    set_default_max_concurrency,
)
from vllm.v1.worker.gpu.cp_utils import prepare_dcp_local_seq_lens
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    ModelCudaGraphManager,
    normalize_model_token_inputs,
)
from vllm.v1.worker.gpu.cudagraph_utils import (
    profile_cudagraph_memory as _profile_cudagraph_memory,
)
from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.ec_connector import get_ec_connector
from vllm.v1.worker.gpu.eplb_utils import EPLBController, step_eplb_after
from vllm.v1.worker.gpu.input_batch import (
    InputBatch,
    InputBuffers,
    combine_sampled_and_draft_tokens,
    expand_idx_mapping,
    post_update,
    post_update_num_computed_tokens,
    prepare_pos_seq_lens,
    prepare_prefill_inputs,
    set_dummy_context,
)
from vllm.v1.worker.gpu.kv_connector import (
    NO_OP_KV_CONNECTOR,
    KVConnector,
    get_kv_connector,
)
from vllm.v1.worker.gpu.lora_utils import (
    LoraState,
    create_lora_capture_hook,
    get_lora_capture_cases,
    get_num_active_loras_for_dispatch,
)
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.mm.lora import set_active_mm_loras
from vllm.v1.worker.gpu.model_states import init_model_state
from vllm.v1.worker.gpu.pool.pooling_runner import PoolingRunner
from vllm.v1.worker.gpu.pp_utils import PPHandler
from vllm.v1.worker.gpu.sample.batch_shard import (
    BatchSharder,
    all_to_all_logits,
    gather_sampler_output,
)
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.prompt_logprob import PromptLogprobsWorker
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.shutdown import free_before_shutdown
from vllm.v1.worker.gpu.spec_decode import init_speculator
from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
    AdaptiveVerificationManager,
    maybe_create_adaptive_verification_manager,
)
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
    set_eagle3_aux_hidden_state_layers,
)
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import (
    RejectionSampler,
    get_max_chunk_logits,
)
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
from vllm.v1.worker.gpu.spec_decode.utils import (
    DraftTokensHandler,
    limit_draft_tokens,
)
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.gpu.structured_outputs import StructuredOutputsWorker
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin
from vllm.v1.worker.utils import (
    KVBlockZeroer,
    copy_kv_cache_blocks_inplace,
    get_uniform_decode_token_count,
)
from vllm.v1.worker.workspace import use_workspace_lane

logger = init_logger(__name__)


class GPUModelRunner(LoRAModelRunnerMixin):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.compilation_config = vllm_config.compilation_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self._draft_workspace_lane = int(
            self.speculative_config is not None and self.speculative_config.use_dspark()
        )
        self.observability_config = vllm_config.observability_config
        self.jit_warmup_registry = JitWarmupRegistry(vllm_config)

        self.device = device
        self.dtype = self.model_config.dtype
        self.kv_cache_dtype = self.dtype
        if self.cache_config.cache_dtype != "auto":
            # Quantized KV cache.
            self.kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[
                self.cache_config.cache_dtype
            ]

        # Lazily initialized in _init_kv_zero_meta() when the KV cache needs
        # zeroing (e.g. hybrid models with fp8 KV cache).
        self.kv_block_zeroer: KVBlockZeroer | None = None

        self.vocab_size = self.model_config.get_vocab_size()
        self.max_model_len = self.model_config.max_model_len
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        self.is_encoder_decoder = self.model_config.is_encoder_decoder

        self.output_copy_stream = torch.cuda.Stream(self.device)

        # Pipeline parallelism.
        self.use_pp = self.parallel_config.pipeline_parallel_size > 1
        self.is_first_pp_rank = get_pp_group().is_first_rank
        self.is_last_pp_rank = get_pp_group().is_last_rank

        # Size the UVA buffer pools to the max number of concurrent in-flight
        # steps. Must run before any pooled buffer is constructed
        set_default_max_concurrency(vllm_config.max_concurrent_batches)

        # PP broadcast/recv helper. Runs the collective on a side stream.
        self.pp_handler: PPHandler | None = None

        # Persistent buffer for intermediate tensors (non-first PP ranks).
        self.intermediate_tensors: IntermediateTensors | None = None

        # Data parallelism.
        self.dp_size = self.parallel_config.data_parallel_size
        self.dp_rank = self.parallel_config.data_parallel_rank

        # Detect EP all2all peer faults to prevent emitting corrupted output.
        # Only meaningful for MoE + DP with an FT-capable all2all backend.
        self.check_ep_fault = False
        if self.dp_size > 1 and self.model_config.is_moe:
            self.check_ep_fault = get_ep_all2all_manager().support_fault_tolerance

        # Decode context parallelism.
        self.dcp_size = self.parallel_config.decode_context_parallel_size
        self.use_dcp = self.dcp_size > 1
        self.dcp_rank = get_dcp_group().rank_in_group if self.use_dcp else 0
        self.cp_interleave = self.parallel_config.cp_kv_cache_interleave_size

        # Multimodal
        self.mm_registry = MULTIMODAL_REGISTRY
        self.supports_mm_inputs = self.mm_registry.supports_multimodal_inputs(
            self.model_config
        )
        self.uses_inputs_embeds = (
            self.supports_mm_inputs or self.model_config.enable_prompt_embeds
        )
        self.encoder_cache = None
        if self.supports_mm_inputs and self.is_first_pp_rank:
            self.encoder_cache = EncoderCache()
        self.ec_connector = get_ec_connector(vllm_config, self.encoder_cache)

        # Speculative decoding.
        self.speculator = None
        self.use_aux_hidden_state_outputs = False
        self.num_speculative_steps = vllm_config.num_speculative_tokens
        if self.speculative_config is not None:
            if self.is_last_pp_rank:
                self.speculator = init_speculator(self.vllm_config, self.device)

            if self.speculative_config.method in (
                "eagle3",
                "dflash",
                "dspark",
                "extract_hidden_states",
            ):
                # Drafting may require auxiliary hidden states from target model outputs
                self.use_aux_hidden_state_outputs = True
                if self.use_pp:
                    raise ValueError(
                        f"{self.speculative_config.method} with pipeline parallel "
                        "is not supported."
                    )

        # Draft tokens propagation - for spec-dec + struct outputs.
        self.draft_tokens_handler = DraftTokensHandler(self.device)

        self.pcp_manager: pcp.PCPManager | None = None

        # Pooling models.
        self.is_pooling_model = self.model_config.runner_type == "pooling"
        self.pooling_runner: PoolingRunner | None = None

        # Multi-module MTP feeds its modules the next num_speculative_steps prefill
        # tokens during chunked prefill. Other speculators only read the immediate
        # next one.
        num_prefill_lookahead = (
            self.num_speculative_steps
            if self.speculative_config is not None
            and self.speculative_config.use_multi_module_mtp()
            else 1
        )

        self.step_timing = StepTimingCollector()

        # General request states.
        self.req_states = RequestState(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            num_speculative_steps=self.num_speculative_steps,
            vocab_size=self.vocab_size,
            device=self.device,
            num_prefill_lookahead=num_prefill_lookahead,
        )
        self.adaptive_verification: AdaptiveVerificationManager | None = None
        self.input_buffers = InputBuffers(
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            device=self.device,
        )
        if self.use_pp:
            self.pp_handler = PPHandler(
                max_num_reqs=self.max_num_reqs,
                num_speculative_steps=self.num_speculative_steps,
                device=self.device,
            )

        # Samplers and decode_query_len created in load_model() after
        # model_state exists (num_new_sampled_tokens_per_step from ModelState).
        self.sampler: Sampler | None = None
        self.rejection_sampler: RejectionSampler | None = None
        self.batch_sharder: BatchSharder | None = None
        self.prompt_logprobs_worker: PromptLogprobsWorker | None = None
        self.structured_outputs_worker: StructuredOutputsWorker | None = None
        self.cudagraph_manager: ModelCudaGraphManager | None = None

        # LoRA-related workers.
        self.lora_state = LoraState(max_num_reqs=self.max_num_reqs)
        self.lora_capture_cases = [0]
        if self.lora_config:
            self.lora_capture_cases = get_lora_capture_cases(
                self.lora_config, self.compilation_config
            )

        # KV Connector if configured.
        self.kv_connector: KVConnector = NO_OP_KV_CONNECTOR

        # For transferring state from execute_model to subsequent sample_tokens call.
        self.execute_model_state: ExecuteModelState | None = None

        # Expert parallelism load balancer.
        self.eplb = EPLBController(self.parallel_config, self.device)
        self.routed_experts_capturer: RoutedExpertsCapturer | None = None

        set_offloader(create_offloader(self.vllm_config.offload_config))

    def update_max_model_len(self, max_model_len: int) -> None:
        self.max_model_len = max_model_len
        self.req_states.max_model_len = max_model_len
        update_model_len = getattr(
            getattr(self, "model", None), "update_max_model_len", None
        )
        if update_model_len is not None:
            update_model_len(max_model_len)
        if isinstance(self.speculator, DraftModelSpeculator):
            self.speculator.update_max_model_len(max_model_len)

    def init_routed_experts_capturer(self) -> None:
        """Initialize target-model capture on every participating worker."""
        self.routed_experts_capturer = RoutedExpertsCapturer(
            max_num_batched_tokens=self.max_num_tokens,
            vllm_config=self.vllm_config,
            kv_cache_config=self.kv_cache_config,
        )
        bind_routed_experts_capturer(self.model, self.routed_experts_capturer)

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        tasks: list[SupportedTask] = []
        if self.model_config.runner_type == "generate":
            tasks.extend(self.model_state.get_supported_generation_tasks())
        if self.is_pooling_model:
            # Do not rely on pooling_runner here, since this information is needed
            # on the first PP rank, while pooling_runner is only initialized
            # on the last PP rank.
            tasks.extend(PoolingRunner.get_supported_tasks(self.model))
        return tuple(tasks)

    def load_model(self, load_dummy_weights: bool = False, *args, **kwargs) -> None:
        time_before_load = time.perf_counter()
        if load_dummy_weights:
            self.load_config.load_format = "dummy"
        self.eplb.prepare_load()
        eplb_models_added = False
        with DeviceMemoryProfiler() as m:
            model_loader = get_model_loader(self.vllm_config.load_config)
            logger.info_once("Loading model from scratch...")

            self.model = model_loader.load_model(
                vllm_config=self.vllm_config, model_config=self.vllm_config.model_config
            )
            if self.lora_config:
                self.model = self.load_lora_model(
                    self.model, self.vllm_config, self.device
                )

            if self.use_aux_hidden_state_outputs:
                assert self.speculative_config is not None
                set_eagle3_aux_hidden_state_layers(self.model, self.speculative_config)
            if isinstance(self.speculator, DraftModelSpeculator):
                with use_workspace_lane(self._draft_workspace_lane):
                    self.speculator.load_model(self.model)
                    eplb_models_added = self.eplb.maybe_register_speculator(
                        self.speculator, self.speculative_config, load_dummy_weights
                    )
        time_after_load = time.perf_counter()

        self.model_memory_usage = m.consumed_memory
        logger.info(
            "Model loading took %s GiB memory and %.6f seconds",
            format_gib(m.consumed_memory),
            time_after_load - time_before_load,
        )

        # Initialize the components that require the model.
        self.model_state = init_model_state(
            self.vllm_config, self.model, self.encoder_cache, self.device
        )

        self.decode_query_len = (
            self.num_speculative_steps
            + self.model_state.num_new_sampled_tokens_per_step
        )

        if self.parallel_config.enable_batch_sharded_sampling:
            if hasattr(self.model, "compute_logits_local"):
                self.batch_sharder = BatchSharder(
                    max_num_reqs=self.max_num_reqs,
                    max_num_logits_per_req=self.decode_query_len,
                    device=self.device,
                )
                logger.info("Batch-sharded sampling enabled.")
            else:
                logger.warning_once(
                    "Disabling batch-sharded sampling: %s does not implement "
                    "compute_logits_local",
                    type(self.model).__name__,
                )

        # Initialize samplers. Model states may override via custom_sampler().
        if self.is_last_pp_rank and not self.is_pooling_model:
            self.sampler = Sampler(
                max_num_reqs=self.max_num_reqs,
                vocab_size=self.vocab_size,
                device=self.device,
                req_states=self.req_states,
                logprobs_mode=self.model_config.logprobs_mode,
                num_speculative_tokens=self.decode_query_len,
                use_fp64_gumbel=self.model_config.use_fp64_gumbel,
                enable_trace_replay=self.model_config.enable_trace_replay,
                reasoning_config=self.vllm_config.reasoning_config,
                return_sampling_mask=self.model_config.return_sampling_mask,
            )
            custom = self.model_state.custom_sampler(self.sampler)

            if custom:
                self.sampler, self.rejection_sampler = custom
            elif self.speculative_config is not None:
                self.rejection_sampler = RejectionSampler(
                    self.sampler,
                    self.speculative_config,
                    self.device,
                )
            self.prompt_logprobs_worker = PromptLogprobsWorker(
                self.max_num_reqs,
                logprobs_mode=self.model_config.logprobs_mode,
            )
            self.structured_outputs_worker = StructuredOutputsWorker(
                max_num_logits=self.max_num_reqs * self.decode_query_len,
                vocab_size=self.vocab_size,
                device=self.device,
                mask_stride=self.decode_query_len,
                num_bonus_tokens=self.model_state.num_new_sampled_tokens_per_step,
            )

        if self.is_pooling_model and self.is_last_pp_rank:
            self.pooling_runner = PoolingRunner(self.model, self.vllm_config)
        eplb_models_added |= self.eplb.maybe_register_model(
            self.model,
            self.model_config,
            load_dummy_weights,
        )
        self.eplb.maybe_start_async_loop(eplb_models_added)

        if not self.is_first_pp_rank:
            # For non-first PP ranks, create intermediate tensors sized
            # for the max capture size so they can be sliced per batch.
            # Save as persistent member so runtime can copy received data
            # into the same addresses that the CUDA graphs captured.
            self.intermediate_tensors = self.model.make_empty_intermediate_tensors(
                batch_size=self.max_num_tokens,
                dtype=self.model_config.dtype,
                device=self.device,
            )

        get_offloader().post_init()

    def get_model(self) -> nn.Module:
        return self.model

    def get_draft_model(self) -> nn.Module | None:
        speculator = self.speculator
        if not isinstance(speculator, DraftModelSpeculator):
            return None
        return speculator.model

    def reload_weights(self, *args, **kwargs) -> None:
        # TODO(Wentao): Use full version instead of import when fully migrated to v2
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner as GPUModelRunnerV1

        GPUModelRunnerV1.reload_weights(self, *args, **kwargs)  # type: ignore[arg-type]

    def update_config(self, *args, **kwargs) -> None:
        # TODO(Wentao): Use full version instead of import when fully migrated to v2
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner as GPUModelRunnerV1

        GPUModelRunnerV1.update_config(self, *args, **kwargs)  # type: ignore[arg-type]

        # v2 reads config via self.vllm_config (e.g. in load_model), so keep it
        # in sync with the attributes the v1 helper just replaced.
        self.vllm_config.model_config = self.model_config
        self.vllm_config.load_config = self.load_config

    @functools.cached_property
    def main_stream(self) -> torch.cuda.Stream:
        # Cache the default CUDA stream to avoid lookup overhead.
        return torch.cuda.current_stream(self.device)

    def get_encoder_timing_stats(self) -> dict[str, dict[str, float | int]]:
        encoder_runner = getattr(self.model_state, "encoder_runner", None)
        if encoder_runner is None:
            return {}
        return encoder_runner.get_encoder_timing_stats()

    def get_kv_cache_spec(self):
        return get_kv_cache_spec(self.vllm_config)

    def initialize_kv_cache(
        self, kv_cache_config: KVCacheConfig, is_profiling: bool = False
    ) -> None:
        kv_cache_config = deepcopy(kv_cache_config)
        self.kv_cache_config = kv_cache_config

        block_table_max_model_len = self.max_model_len
        if self.is_encoder_decoder:
            # Cross-attention block tables need to index encoder tokens, which
            # can exceed the decoder's max_model_len.
            block_table_max_model_len = max(
                block_table_max_model_len,
                self.scheduler_config.max_num_encoder_input_tokens,
                getattr(self.model_config.hf_config, "max_source_positions", 0),
            )

        block_sizes = []
        max_num_blocks_per_group = []
        group_cp_sizes = []
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            spec = kv_cache_group.kv_cache_spec
            block_sizes.append(spec.block_size)
            group_cp_sizes.append(
                1 if getattr(spec, "dcp_replicated", False) else self.dcp_size
            )
            # Let each cache type account for CP. Attention KV is DCP-sharded,
            # while Mamba/GDN recurrent state is replicated across DCP ranks.
            max_num_blocks = spec.max_num_blocks_per_req(
                self.vllm_config, block_table_max_model_len
            )
            # Preserve each cache type's alignment requirements after applying
            # its topology-aware block-table width.
            if isinstance(spec, MambaSpec):
                max_num_blocks = get_block_table_width(
                    max_num_blocks, spec.block_size, token_alignment=None
                )
            else:
                max_num_blocks = get_block_table_width(max_num_blocks, spec.block_size)
            max_num_blocks_per_group.append(max_num_blocks)

        target_attn_layer_names = None
        if isinstance(self.speculator, DraftModelSpeculator):
            # Adaptive verification validates target attention separately.
            target_attn_layer_names = {
                layer_name
                for group in self.kv_cache_config.kv_cache_groups
                for layer_name in group.layer_names
            } - self.speculator.draft_attn_layer_names
        self.attn_groups, attn_cg_support, self.kernel_block_sizes = init_attn_backend(
            self.kv_cache_config,
            self.vllm_config,
            self.device,
        )
        additional_attn_cg_support = self.model_state.get_additional_cg_support()
        attn_cg_support = attn_cg_support.narrow(*additional_attn_cg_support)
        # The speculator clears the flag at load time when the checkpoint has
        # no confidence head, so it holds the effective value.
        self.adaptive_verification = maybe_create_adaptive_verification_manager(
            enable_adaptive_verification=getattr(
                self.speculator, "enable_adaptive_verification", False
            ),
            attn_groups=self.attn_groups,
            attn_cg_support=attn_cg_support,
            req_states=self.req_states,
            query_start_loc=self.input_buffers.query_start_loc,
            num_bonus_tokens=self.model_state.num_new_sampled_tokens_per_step,
            max_total_logits=get_max_chunk_logits(self.vocab_size),
            vllm_config=self.vllm_config,
            target_layer_names=target_attn_layer_names,
            additional_attn_cg_support=additional_attn_cg_support,
        )

        self.block_tables = BlockTables(
            block_sizes=block_sizes,
            max_num_reqs=self.max_num_reqs,
            max_num_batched_tokens=self.max_num_tokens,
            max_num_blocks_per_group=max_num_blocks_per_group,
            device=self.device,
            kernel_block_sizes=self.kernel_block_sizes,
            cp_size=self.dcp_size,
            cp_rank=self.dcp_rank,
            cp_interleave=self.cp_interleave,
            group_cp_sizes=group_cp_sizes,
        )
        self.pcp_manager = pcp.maybe_build_pcp_manager(
            self.vllm_config,
            self.device,
            self.supports_mm_inputs,
            self.req_states,
            self.block_tables,
            cls=self.pcp_manager_cls,
        )
        initialize_mamba_ssu_backend(
            self.vllm_config.mamba_config, self.kv_cache_config
        )
        if self.adaptive_verification is not None:
            self.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
        cudagraph_mode = self.compilation_config.resolve_cudagraph_mode_and_sizes(
            attn_cg_support.min_cg_support,
            attn_cg_support.min_cg_attn_backend,
            self.decode_query_len,
            use_v2_model_runner=True,
            tensor_parallel_size=self.parallel_config.tensor_parallel_size,
            kv_cache_config=self.kv_cache_config,
            max_num_reqs=self.max_num_reqs,
            is_profiling=is_profiling,
        )
        self.cudagraph_manager = ModelCudaGraphManager(
            self.vllm_config,
            self.device,
            cudagraph_mode,
            decode_query_len=self.decode_query_len,
            lora_capture_cases=self.lora_capture_cases,
            varlen_decode=self.adaptive_verification is not None,
        )
        check_attention_cp_compatibility(self.vllm_config)
        if isinstance(self.speculator, DraftModelSpeculator):
            # HACK(woosuk)
            self.speculator.set_attn(
                self.model_state,
                self.kv_cache_config,
                self.block_tables,
                self.input_buffers,
                self.attn_groups,
            )
        if self.speculator is not None:
            # After set_attn, so the speculator can size its cudagraph mode
            # to its own attention support.
            self.speculator.init_cudagraph_manager(cudagraph_mode)

        self.kv_caches: list[torch.Tensor] = []
        kv_caches_dict = init_kv_cache(
            self.kv_caches,
            self.compilation_config.static_forward_context,
            self.kv_cache_config,
            self.device,
            self.kernel_block_sizes,
            self.vllm_config,
        )
        if is_profiling:
            self.kv_connector = NO_OP_KV_CONNECTOR
        else:
            self.kv_connector = get_kv_connector(self.vllm_config, kv_caches_dict)

    def _init_kv_zero_meta(self) -> None:
        """Build KV-block zeroing metadata; invoked from gpu_worker."""
        self.kv_block_zeroer = KVBlockZeroer(
            self.device,
            attn_groups_iter=(g for groups in self.attn_groups for g in groups),
            kernel_block_sizes=self.kernel_block_sizes,
            static_forward_context=self.compilation_config.static_forward_context,
        )

    @torch.inference_mode()
    @step_eplb_after(is_dummy=True)
    def _dummy_run(
        self,
        num_tokens: int,
        *args,
        skip_attn: bool = False,
        uniform_decode: bool = False,
        context_len: int = 0,
        skip_eplb: bool = False,
        is_profile: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if skip_attn and not is_profile:
            raise ValueError(
                "skip_attn must only be True for initial memory profiling."
            )

        # Create a dummy scheduler output.
        num_reqs = min(num_tokens, self.max_num_reqs)
        if uniform_decode:
            # HACK(lucas): for now since the worker is shared between MRV1 and MRV2,
            # and for spec-decode with MTP we want to make sure the dummy runs use
            # 1+num_speculative_tokens we use max here, this will likely be eventually
            # changed in the worker: https://github.com/vllm-project/vllm/pull/35243
            num_tokens = max(num_tokens, self.decode_query_len)
            num_reqs = num_tokens // self.decode_query_len
            assert num_tokens % self.decode_query_len == 0
        # Distribute the remainder evenly so no dummy request exceeds
        # ceil(num_tokens / num_reqs) <= max_model_len tokens.
        num_tokens_per_request = [
            num_tokens // num_reqs + (i >= num_reqs - num_tokens % num_reqs)
            for i in range(num_reqs)
        ]

        assert sum(num_tokens_per_request) == num_tokens
        num_scheduled_tokens = {
            f"_dummy_req_{i}": n for i, n in enumerate(num_tokens_per_request)
        }
        dummy_scheduler_output = SchedulerOutput.make_empty()
        dummy_scheduler_output.total_num_scheduled_tokens = num_tokens
        dummy_scheduler_output.num_scheduled_tokens = num_scheduled_tokens

        # Disable any use of KVConnector for dummy runs.
        self.kv_connector.set_disabled(True)

        # Get the intermediate tensors for the dummy run.
        intermediate_tensors = None
        if not self.is_first_pp_rank:
            assert self.intermediate_tensors is not None
            intermediate_tensors = self.intermediate_tensors[:num_tokens]

        max_loras = self.lora_config.max_loras if self.lora_config is not None else 0
        with self.maybe_dummy_run_with_lora(
            self.lora_config,
            num_scheduled_tokens=np.array(num_tokens_per_request, dtype=np.int32),
            num_sampled_tokens=None,
            remove_lora=True,
            num_active_loras=max_loras,
        ):
            # Execute the model.
            self.execute_model(
                dummy_scheduler_output,
                intermediate_tensors=intermediate_tensors,
                dummy_run=True,
                skip_attn_for_dummy_run=skip_attn,
                is_profile=is_profile,
                context_len=context_len,
            )
        self.kv_connector.set_disabled(False)

        # Non-last PP ranks don't produce output for sampling.
        if not self.is_last_pp_rank:
            return None, None

        assert self.execute_model_state is not None
        input_batch = self.execute_model_state.input_batch
        attn_metadata = self.execute_model_state.attn_metadata
        slot_mappings_by_layer = self.execute_model_state.slot_mappings_by_layer
        hidden_states = self.execute_model_state.hidden_states
        aux_hidden_states = self.execute_model_state.aux_hidden_states
        self.execute_model_state = None

        self.step_timing.forward_end()

        # dummy run the eagle speculator's propose to ensure DP/EP sync.
        if self.speculator is not None:
            assert self.sampler is not None
            self.step_timing.drafter_start()
            mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None
            if self.speculator.supports_mm_inputs:
                mm_inputs = (
                    [],
                    torch.zeros(
                        input_batch.num_tokens,
                        dtype=torch.bool,
                        device="cpu",
                    ),
                )

            # Let the target override the hidden state fed to the drafter
            # (e.g. DeepSeek V4 MTP needs the pre-hc_head residual). The
            # target returns a persistent buffer sized at max_num_batched_tokens;
            # slice to the active token count that propose() expects.
            spec_hidden_states = hidden_states
            if hasattr(self.model, "get_mtp_target_hidden_states"):
                pre_hc_hidden_states = self.model.get_mtp_target_hidden_states()
                spec_hidden_states = pre_hc_hidden_states[: hidden_states.shape[0]]  # type: ignore[union-attr]
            with use_workspace_lane(self._draft_workspace_lane):
                self.speculator.propose(
                    input_batch=input_batch,
                    attn_metadata=attn_metadata,
                    slot_mappings=slot_mappings_by_layer,
                    last_hidden_states=spec_hidden_states,
                    aux_hidden_states=aux_hidden_states,
                    num_sampled=torch.ones(
                        input_batch.num_reqs, dtype=torch.int32, device=self.device
                    ),
                    num_rejected=torch.zeros(
                        input_batch.num_reqs, dtype=torch.int32, device=self.device
                    ),
                    last_sampled=self.req_states.last_sampled_tokens,
                    next_prefill_tokens=self.req_states.next_prefill_tokens,
                    temperature=self.sampler.sampling_states.temperature.gpu,
                    seeds=self.sampler.sampling_states.seeds.gpu,
                    dummy_run=True,
                    skip_attn_for_dummy_run=skip_attn,
                    mm_inputs=mm_inputs,
                    is_profile=is_profile,
                )
            self.step_timing.drafter_end()

        assert hidden_states is not None  # Last PP rank always has hidden_states
        sample_hidden_states = hidden_states[input_batch.logits_indices]
        return hidden_states, sample_hidden_states

    @torch.inference_mode()
    def _dummy_sampler_run(self, hidden_states: torch.Tensor) -> None:
        num_reqs = hidden_states.shape[0]
        logits = self.model.compute_logits(hidden_states)
        dummy_input_batch = InputBatch.make_dummy(
            num_reqs, num_reqs, self.input_buffers
        )

        # NOTE(woosuk): During the initial memory profiling, the sampler may skip
        # top_k, top_p, and logprobs, using less GPU memory than what is possible
        # during actual execution.
        assert self.sampler is not None
        self.sampler(logits, dummy_input_batch)

    @torch.inference_mode()
    def _dummy_pooler_run(self, hidden_states: torch.Tensor) -> None:
        assert self.pooling_runner is not None
        self.pooling_runner.dummy_pooler_run(hidden_states)

    @torch.inference_mode()
    def profile_run(self) -> None:
        if self.supports_mm_inputs and self.is_first_pp_rank:
            mm_config = self.model_config.multimodal_config
            if mm_config is not None and not mm_config.skip_mm_profiling:
                mm_budget = MultiModalBudget(
                    self.vllm_config,
                    self.mm_registry,
                    enable_cache=False,
                )
                dummy_mm_inputs = get_dummy_encoder_profile_inputs(
                    self.mm_registry,
                    mm_budget,
                )
                self.model_state.encoder_runner.profile_encoder_cache(
                    dummy_mm_inputs, mm_budget
                )

        hidden_states, sample_hidden_states = self._dummy_run(
            self.max_num_tokens, skip_attn=True, is_profile=True
        )

        # Only run sampler/pooler on last PP rank (non-last ranks return None).
        if self.is_last_pp_rank:
            assert sample_hidden_states is not None
            if self.pooling_runner is None:
                self._dummy_sampler_run(sample_hidden_states)
            else:
                self._dummy_pooler_run(hidden_states)

        torch.accelerator.synchronize()
        del hidden_states, sample_hidden_states
        self.reset_encoder_cache()
        gc.collect()

    def post_kv_cache_wake_up(self) -> None:
        self.block_tables.init_block_table_layout_tensors()

    def reset_mm_cache(self) -> None:
        if self.encoder_cache is not None:
            self.encoder_cache.reset_mm_cache()

    def reset_encoder_cache(self) -> None:
        if self.encoder_cache is not None:
            self.encoder_cache.reset_encoder_cache()
        if self.pooling_runner is not None:
            self.pooling_runner.clear()

    @torch.inference_mode()
    def profile_cudagraph_memory(self) -> int:
        """Estimate the GPU memory required to capture CUDA graphs."""
        return _profile_cudagraph_memory(self)

    @torch.inference_mode()
    def capture_model(self) -> int:
        assert self.cudagraph_manager is not None
        capture_encoder = (
            self.model_state.supports_mm_inputs
            and self.model_state.encoder_runner.has_cudagraph()
        )
        capture_decoder = self.cudagraph_manager.needs_capture()
        if not capture_encoder and not capture_decoder:
            logger.warning(
                "Skipping encoder and decoder CUDA graph capture. To enable "
                "encoder capture, ensure `cudagraph_mm_encoder` is enabled; "
                "to enable decoder capture, ensure `cudagraph_mode` is not `NONE`."
            )
            return 0

        compilation_counter.num_gpu_runner_capture_triggers += 1

        start_time = time.perf_counter()
        gc.collect()
        torch.accelerator.empty_cache()
        start_free_gpu_memory = torch.accelerator.get_memory_info()[0]

        with self.maybe_setup_dummy_loras(self.lora_config):
            if capture_encoder:
                self.model_state.encoder_runner.capture()

            if capture_decoder:
                self.cudagraph_manager.capture(
                    self.model,
                    self.model_state,
                    self.input_buffers,
                    self.intermediate_tensors,
                    self.block_tables,
                    self.attn_groups,
                    self.kv_cache_config,
                    has_lora=self.lora_config is not None,
                    use_aux_hidden_state_outputs=self.use_aux_hidden_state_outputs,
                    lora_capture_hook=create_lora_capture_hook(self.lora_config, self),
                )
                if self.speculator is not None:
                    with use_workspace_lane(self._draft_workspace_lane):
                        self.speculator.capture()
                if self.adaptive_verification is not None:
                    with self.step_timing.collect() as timings:
                        for batch in self.adaptive_verification.batches_to_profile(
                            self.cudagraph_manager.captured_token_counts()
                        ):
                            self._dummy_run(**batch)
                    self.adaptive_verification.set_initial_cost_curves(timings)

        end_time = time.perf_counter()
        end_free_gpu_memory = torch.accelerator.get_memory_info()[0]
        elapsed_time = end_time - start_time
        cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory
        # This usually takes 5~20 seconds.
        logger.info(
            "Graph capturing finished in %.0f secs, took %.2f GiB",
            elapsed_time,
            cuda_graph_size / (1 << 30),
        )
        return cuda_graph_size

    def _remove_request(self, req_id: str) -> bool:
        # Call model_state.remove_request *before* req_states.remove_request
        # so the model_state can still look up the slot index.
        self.model_state.remove_request(req_id)
        req_idx = self.req_states.remove_request(req_id)
        if req_idx is None:
            return False
        if self.pooling_runner is not None:
            self.pooling_runner.remove_request(req_idx)
        if self.pp_handler is not None:
            self.pp_handler.on_req_idx_freed(req_idx)
        if self.encoder_cache is not None:
            self.encoder_cache.remove_request(req_id)
        if self.prompt_logprobs_worker is not None:
            self.prompt_logprobs_worker.remove_request(req_id)
        self.lora_state.remove_request(req_id)
        return True

    def finish_requests(self, scheduler_output: SchedulerOutput) -> None:
        finished_req_ids = scheduler_output.finished_req_ids
        if self.pooling_runner is not None:
            # Preempted docs keep their query-use reservation until rescheduled.
            self.pooling_runner.on_requests_finished(finished_req_ids)
        preempted_req_ids = scheduler_output.preempted_req_ids
        if preempted_req_ids:
            finished_req_ids = finished_req_ids.union(preempted_req_ids)
        # Sorted so every TP rank frees request slots in the same order.
        # Features like batch-sharded sampling derive rank request ownership
        # from the slot index.
        for req_id in sorted(finished_req_ids):
            self._remove_request(req_id)

    def free_states(self, scheduler_output: SchedulerOutput) -> None:
        if self.encoder_cache is not None:
            for mm_hash in scheduler_output.free_encoder_mm_hashes:
                self.encoder_cache.free_encoder_cache(mm_hash)

    def update_pp_decode_requests(self):
        # For non-last PP ranks, update decode requests with sampler output from
        # the prior step in which they were scheduled (pp_size steps ago).
        if self.pp_handler is not None:
            outputs = self.pp_handler.get_prev_sampled_outputs()
            if outputs is not None:
                self.postprocess_sampled(**outputs)

    def add_requests(self, scheduler_output: SchedulerOutput) -> None:
        for new_req_data in scheduler_output.scheduled_new_reqs:
            assert new_req_data.prefill_token_ids is not None
            req_id = new_req_data.req_id

            # Streaming input update: request already exists from a prior
            # chunk. Remove old state so it can be cleanly re-added below
            # with the updated prompt_token_ids and mm_features.
            self._remove_request(req_id)

            prompt_len = new_req_data.prompt_len
            sampling_params = new_req_data.sampling_params
            self.req_states.add_request(
                req_id=req_id,
                prompt_len=prompt_len,
                all_token_ids=new_req_data.prefill_token_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                max_tokens=sampling_params.max_tokens if sampling_params else 1,  # type: ignore[arg-type]
            )
            req_index = self.req_states.req_id_to_index[req_id]
            if self.adaptive_verification is not None:
                self.adaptive_verification.add_request(req_index)

            if self.pooling_runner is not None:
                assert new_req_data.pooling_params is not None
                assert new_req_data.prompt_token_ids is not None
                self.pooling_runner.add_request(
                    req_id,
                    req_index,
                    new_req_data.pooling_params,
                    new_req_data.prompt_token_ids,
                )

            if self.encoder_cache is not None:
                self.encoder_cache.add_request(req_id, new_req_data.mm_features)

            self.model_state.add_request(req_index, new_req_data)
            self.block_tables.append_block_ids(
                req_index, new_req_data.block_ids, overwrite=True
            )
            self.lora_state.add_request(req_id, req_index, new_req_data.lora_request)

            if self.is_last_pp_rank and new_req_data.sampling_params is not None:
                assert self.sampler is not None
                self.sampler.add_request(
                    req_index, prompt_len, new_req_data.sampling_params
                )
                assert self.prompt_logprobs_worker is not None
                self.prompt_logprobs_worker.add_request(
                    req_id, req_index, new_req_data.sampling_params
                )

        if scheduler_output.scheduled_new_reqs:
            self.req_states.apply_staged_writes()
            self.model_state.apply_staged_writes()
        if self.sampler is not None:
            self.sampler.apply_staged_writes()

    def update_requests(self, scheduler_output: SchedulerOutput) -> None:
        # Add new blocks and update num_computed_tokens for the existing requests.
        reqs = scheduler_output.scheduled_cached_reqs
        num_computed_tokens_np = self.req_states.num_computed_tokens_np
        for req_id, num_computed_tokens, req_new_block_ids in zip(
            reqs.req_ids, reqs.num_computed_tokens, reqs.new_block_ids
        ):
            req_index = self.req_states.req_id_to_index[req_id]
            num_computed_tokens_np[req_index] = num_computed_tokens
            if req_new_block_ids is not None:
                self.block_tables.append_block_ids(
                    req_index, req_new_block_ids, overwrite=False
                )

        # Update CPU num_computed_prefill_tokens.
        np.minimum(
            self.req_states.num_computed_tokens_np,
            self.req_states.prefill_len.np,
            out=self.req_states.num_computed_prefill_tokens,
        )

        # Zero GPU memory for freshly allocated cache blocks to prevent
        # stale NaN/data from corrupting attention or SSM computation.
        if scheduler_output.new_block_ids_to_zero:
            assert self.kv_block_zeroer is not None
            self.kv_block_zeroer.zero_block_ids(scheduler_output.new_block_ids_to_zero)

        # Apply copy-on-write block copies for partial prefix-cache hits, after
        # zeroing new blocks and before the forward pass reads them.
        if scheduler_output.kv_cache_block_copies:
            copy_kv_cache_blocks_inplace(
                self.kv_caches,
                self.kv_cache_config.num_blocks,
                scheduler_output.kv_cache_block_copies,
            )

    def gather_batch_req_state(
        self, scheduler_output: SchedulerOutput, dummy_run: bool
    ) -> tuple["BatchReqState | None", int | None]:
        """Gather CPU request state for the scheduled batch, in batch order.
        Returns (batch_state, uniform_decode_token_count)
        """
        num_tokens_per_req = scheduler_output.num_scheduled_tokens
        num_reqs = len(num_tokens_per_req)
        num_toks = scheduler_output.total_num_scheduled_tokens
        max_query_len = max(scheduler_output.num_scheduled_tokens.values())

        if dummy_run:
            # Dummy batches are uniform by construction.
            return None, get_uniform_decode_token_count(
                num_reqs, num_toks, max_query_len, has_prefill=False
            )

        draft_tokens = scheduler_output.scheduled_spec_decode_tokens
        # batch_idx -> req_id
        req_ids = sort_batch_req_ids(
            num_tokens_per_req, draft_tokens, self.decode_query_len
        )

        numtoks_iter = map(num_tokens_per_req.__getitem__, req_ids)
        num_scheduled_tokens = np.fromiter(numtoks_iter, dtype=np.int32, count=num_reqs)

        idx_mapping_iter = map(self.req_states.req_id_to_index.__getitem__, req_ids)
        idx_mapping_np = np.fromiter(idx_mapping_iter, dtype=np.intp, count=num_reqs)
        prefill_len_np = self.req_states.prefill_len.np[idx_mapping_np]
        num_computed_prefill_tokens_np = self.req_states.num_computed_prefill_tokens[
            idx_mapping_np
        ]
        is_prefilling_np = num_computed_prefill_tokens_np < prefill_len_np

        if self.adaptive_verification is not None and draft_tokens:
            num_toks = self.adaptive_verification.get_num_tokens(
                num_tokens_per_req, draft_tokens
            )

        batch_state = BatchReqState(
            req_ids=req_ids,
            num_scheduled_tokens=num_scheduled_tokens,
            num_tokens=num_toks,
            idx_mapping_np=idx_mapping_np,
            prefill_len_np=prefill_len_np,
            num_computed_prefill_tokens_np=num_computed_prefill_tokens_np,
            is_prefilling_np=is_prefilling_np,
            has_prefill=bool(is_prefilling_np.any()),
        )
        return batch_state, get_uniform_decode_token_count(
            num_reqs, num_toks, max_query_len, batch_state.has_prefill
        )

    def prepare_inputs(
        self,
        scheduler_output: SchedulerOutput,
        batch_req_state: "BatchReqState",
        batch_desc: BatchExecutionDescriptor,
    ) -> InputBatch:
        num_tokens = batch_req_state.num_tokens
        num_tokens_after_padding = batch_desc.num_tokens
        assert num_tokens > 0
        if envs.VLLM_MOE_SKIP_PADDING:
            # Mark trailing cudagraph-padding rows so kernels can skip work for
            # them when supported.
            is_padding = self.input_buffers.is_padding
            is_padding[:num_tokens].fill_(False)
            is_padding[num_tokens:num_tokens_after_padding].fill_(True)

        req_ids = batch_req_state.req_ids
        num_scheduled_tokens_np = batch_req_state.num_scheduled_tokens
        idx_mapping_np = batch_req_state.idx_mapping_np
        idx_mapping = async_copy_to_gpu(idx_mapping_np, device=self.device)
        num_reqs = len(req_ids)

        # Get the number of draft tokens for each request.
        draft_tokens = scheduler_output.scheduled_spec_decode_tokens
        num_draft_tokens_per_req = None
        if not draft_tokens:
            # No draft token scheduled (common case).
            total_num_draft_tokens = 0
            total_num_logits = num_reqs
            cu_num_logits_np = np.arange(num_reqs + 1, dtype=np.int32)
            cu_num_logits = torch.arange(
                num_reqs + 1, device=self.device, dtype=torch.int32
            )
            expanded_idx_mapping = idx_mapping
            expanded_local_pos = torch.zeros(
                num_reqs, dtype=torch.int32, device=self.device
            )
        else:
            num_draft_tokens_per_req = np.fromiter(
                (len(draft_tokens.get(req_id, ())) for req_id in req_ids),
                dtype=np.int32,
                count=num_reqs,
            )
            num_bonus_tokens = self.model_state.num_new_sampled_tokens_per_step
            total_num_draft_tokens = int(num_draft_tokens_per_req.sum())
            total_num_logits = num_reqs * num_bonus_tokens + total_num_draft_tokens
            num_logits = num_draft_tokens_per_req + num_bonus_tokens
            cu_num_logits_np = np.empty(num_reqs + 1, dtype=np.int32)
            cu_num_logits_np[0] = 0
            np.cumsum(num_logits, out=cu_num_logits_np[1:])
            cu_num_logits = async_copy_to_gpu(cu_num_logits_np, device=self.device)

        adaptive_verification = (
            self.adaptive_verification if num_draft_tokens_per_req is not None else None
        )
        num_scheduled_tokens_upper_bound = num_scheduled_tokens_np
        if adaptive_verification is not None:
            # num_scheduled_tokens represents the draft budget evenly distributed across
            # all verification requests, `reallocate_drafts` will unevenly assign the
            # draft budget to requests on the GPU side only.
            num_scheduled_tokens_np, cu_num_logits_np = (
                adaptive_verification.compact_batch(
                    num_draft_tokens_per_req,
                    num_scheduled_tokens_np,
                    cu_num_logits_np,
                )
            )

        # Get query_start_loc.
        # num_reqs_padded is None for PIECEWISE graphs (no request padding needed)
        num_reqs_padded = batch_desc.num_reqs or num_reqs
        query_start_loc_np = np.empty(self.max_num_reqs + 1, dtype=np.int32)
        query_start_loc_np[0] = 0
        np.cumsum(num_scheduled_tokens_np, out=query_start_loc_np[1 : num_reqs + 1])
        # Pad for full CUDA graph mode.
        # Some attention backends like FA3 require query_start_loc to be non-decreasing.
        query_start_loc_np[num_reqs + 1 :] = num_tokens
        query_start_loc = self.input_buffers.query_start_loc
        async_copy_to_gpu(query_start_loc_np, out=query_start_loc)
        if adaptive_verification is not None:
            cu_num_logits, query_start_loc, total_num_draft_tokens = (
                adaptive_verification.reallocate_drafts(req_ids, idx_mapping)
            )
            total_num_logits = num_reqs * num_bonus_tokens + total_num_draft_tokens
        if draft_tokens:
            expanded_idx_mapping, expanded_local_pos = expand_idx_mapping(
                idx_mapping, total_num_logits, cu_num_logits, self.decode_query_len
            )
        query_start_loc_np = query_start_loc_np[: num_reqs_padded + 1]
        query_start_loc = query_start_loc[: num_reqs_padded + 1]

        # Get prefill tokens if any.
        if batch_req_state.has_prefill:
            prepare_prefill_inputs(
                self.input_buffers.input_ids,
                self.req_states.next_prefill_tokens,
                idx_mapping,
                query_start_loc,
                self.req_states.all_token_ids.gpu,
                self.req_states.prefill_len.gpu,
                self.req_states.num_computed_tokens.gpu,
            )

        # Prepare positions and seq_lens.
        prepare_pos_seq_lens(
            idx_mapping,
            query_start_loc,
            self.req_states.num_computed_tokens.gpu,
            self.input_buffers.positions,
            self.input_buffers.seq_lens,
        )
        seq_lens = self.input_buffers.seq_lens[:num_reqs_padded]

        dcp_local_seq_lens = None
        if self.use_dcp:
            # Prepare dcp local seq_lens.
            prepare_dcp_local_seq_lens(
                self.input_buffers.dcp_local_seq_lens,
                self.input_buffers.seq_lens,
                num_reqs,
                self.dcp_size,
                self.dcp_rank,
                self.cp_interleave,
            )
            dcp_local_seq_lens = self.input_buffers.dcp_local_seq_lens[:num_reqs_padded]

        # Some input token ids are directly read from the last sampled tokens
        # and draft tokens. Also, get the logits indices to sample tokens from.
        logits_indices = combine_sampled_and_draft_tokens(
            self.input_buffers.input_ids,
            idx_mapping,
            self.req_states.last_sampled_tokens,
            query_start_loc,
            seq_lens,
            self.req_states.prefill_len.gpu,
            self.req_states.draft_tokens,
            cu_num_logits,
            total_num_logits,
            self.model_state.num_new_sampled_tokens_per_step,
        )

        # CPU upper bound on seq_lens; padded entries left at zero.
        num_computed_tokens_np = self.req_states.num_computed_tokens_np[idx_mapping_np]
        seq_lens_cpu_upper_bound_np = np.zeros(num_reqs_padded, dtype=np.int32)
        np.add(
            num_computed_tokens_np,
            num_scheduled_tokens_upper_bound,
            out=seq_lens_cpu_upper_bound_np[:num_reqs],
        )
        seq_lens_cpu_upper_bound = torch.from_numpy(seq_lens_cpu_upper_bound_np)

        max_seq_len_np = None
        if self.use_pp:
            # max_seq_len is only consumed by the PP `compute_need_sampled_mask`
            max_seq_len_np = self.req_states.max_seq_len[idx_mapping_np]

        prompt_lens = None
        if self.model_config.rswa_window is not None:
            # prompt_lens is only used in R-SWA case.
            prompt_lens = self.req_states.prompt_len.gpu[idx_mapping]

        input_batch = InputBatch(
            req_ids=req_ids,
            num_reqs=num_reqs,
            num_reqs_after_padding=num_reqs_padded,
            idx_mapping=idx_mapping,
            idx_mapping_np=idx_mapping_np,
            expanded_idx_mapping=expanded_idx_mapping,
            expanded_local_pos=expanded_local_pos,
            num_scheduled_tokens=num_scheduled_tokens_upper_bound,
            num_tokens=num_tokens,
            num_tokens_after_padding=num_tokens_after_padding,
            num_draft_tokens=total_num_draft_tokens,
            num_draft_tokens_per_req=num_draft_tokens_per_req,
            query_start_loc=query_start_loc,
            query_start_loc_np=query_start_loc_np,
            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            dcp_local_seq_lens=dcp_local_seq_lens,
            num_computed_tokens_np=num_computed_tokens_np,
            prefill_len_np=batch_req_state.prefill_len_np,
            num_computed_prefill_tokens_np=batch_req_state.num_computed_prefill_tokens_np,
            is_prefilling_np=batch_req_state.is_prefilling_np,
            has_prefill=batch_req_state.has_prefill,
            max_seq_len_np=max_seq_len_np,
            input_ids=self.input_buffers.input_ids[:num_tokens_after_padding],
            positions=self.input_buffers.positions[:num_tokens_after_padding],
            is_padding=self.input_buffers.is_padding[:num_tokens_after_padding],
            logits_indices=logits_indices,
            cu_num_logits=cu_num_logits,
            cu_num_logits_np=cu_num_logits_np,
            has_structured_output_reqs=scheduler_output.has_structured_output_requests,
            prompt_lens=prompt_lens,
            max_query_len=(
                int(num_scheduled_tokens_upper_bound.max())
                if adaptive_verification is not None
                else None
            ),
        )
        return pcp.maybe_partition_pcp_batch(self.pcp_manager, input_batch)

    def prepare_attn(
        self, input_batch: InputBatch
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        if self.pcp_manager is not None:
            return self.pcp_manager.prepare_attn(input_batch)

        # Block tables: num_kv_cache_groups x [num_reqs_padded, max_num_blocks].
        block_tables = self.block_tables.gather_block_tables(
            input_batch.idx_mapping,
            num_reqs_padded=input_batch.num_reqs_after_padding,
        )
        # Slot mappings: [num_kv_cache_groups, num_tokens_padded].
        # Kernel pads beyond num_tokens with PAD_SLOT_ID.
        slot_mappings = self.block_tables.compute_slot_mappings(
            input_batch.idx_mapping,
            input_batch.query_start_loc,
            input_batch.positions,
            num_tokens_padded=input_batch.num_tokens_after_padding,
        )
        return block_tables, slot_mappings

    def prepare_dummy_attn(
        self, input_batch: InputBatch
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        block_tables = self.block_tables.get_dummy_block_tables(input_batch.num_reqs)
        slot_mappings = pcp.maybe_get_pcp_dummy_slot_mappings(
            self.pcp_manager, self.block_tables, input_batch.num_tokens
        )
        return block_tables, slot_mappings

    def sample(
        self,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        grammar_output: GrammarOutput | None,
    ) -> tuple[SamplerOutput, torch.Tensor, torch.Tensor]:
        shard_metadata = None
        global_input_batch = input_batch
        if self.batch_sharder is not None:
            # Shard the inputs along the batch dimension to sample in parallel
            # across TP ranks.
            input_batch, sorted_logits_indices, grammar_output, shard_metadata = (
                self.batch_sharder.shard_sampler_inputs(input_batch, grammar_output)
            )
            # The hidden states must be gathered in rank-owner-sorted order
            # before computing the partial-vocab logits, so that the all-to-all
            # produces full-vocab logits for just the locally-owned requests.
            sample_hidden_states = hidden_states[sorted_logits_indices]
            local_logits = self.model.compute_logits_local(sample_hidden_states)
            logits = all_to_all_logits(local_logits, shard_metadata)
            logits = logits[:, : self.vocab_size]
        else:
            sample_hidden_states = hidden_states[input_batch.logits_indices]
            logits = self.model.compute_logits(sample_hidden_states)

        if grammar_output is not None:
            # Apply grammar bitmask to the logits in-place.
            assert self.structured_outputs_worker is not None
            self.structured_outputs_worker.apply_grammar_bitmask(
                logits,
                input_batch,
                grammar_output.structured_output_request_ids,
                grammar_output.grammar_bitmask,
            )

        sampler_output: SamplerOutput | None
        if input_batch.num_reqs == 0:
            # This rank owns no requests this step. It contributes an
            # all-padding block to the gather below.
            sampler_output = None
        elif input_batch.num_draft_tokens == 0 or self.rejection_sampler is None:
            assert self.sampler is not None
            sampler_output = self.sampler(logits, input_batch)
        else:
            # Rejection sampling for spec decoding.
            assert self.rejection_sampler is not None
            assert self.speculator is not None
            sampler_output = self.rejection_sampler(
                logits,
                input_batch,
                # Draft logits are needed for probabilistic rejection sampling.
                self.speculator.draft_logits,
            )

        if shard_metadata is not None:
            # Gather the sharded sampler outputs from the TP ranks into a single
            # sampler output.
            assert self.sampler is not None
            sampler_output = gather_sampler_output(
                sampler_output,
                shard_metadata,
                device=self.device,
                global_batch=global_input_batch,
                local_batch=input_batch,
                gather_num_nans=self.sampler.compute_nans,
                logprobs_dims=self.sampler.get_logprobs_dims(
                    global_input_batch.idx_mapping_np,
                    # Rejection sampler does not return logprob token ids.
                    include_token_ids=(
                        global_input_batch.num_draft_tokens == 0
                        or self.rejection_sampler is None
                    ),
                ),
            )

        assert sampler_output is not None
        return sampler_output, sampler_output.num_sampled, sampler_output.num_rejected

    def postprocess_sampled(
        self,
        idx_mapping: torch.Tensor,  # May include -1 for masked entries
        sampled_tokens: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        query_start_loc: torch.Tensor | None = None,
    ) -> None:
        # Update the number of computed tokens.
        if self.is_last_pp_rank:
            assert self.sampler is not None
            output_bin_counts = self.sampler.penalties_state.output_bin_counts
        else:
            output_bin_counts = None
        post_update(
            idx_mapping,
            self.req_states.num_computed_tokens.gpu,
            self.req_states.last_sampled_tokens,
            output_bin_counts,
            sampled_tokens,
            num_sampled,
            num_rejected,
            query_start_loc,
            self.req_states.all_token_ids.gpu,
            self.req_states.total_len.gpu,
        )

        self.model_state.postprocess_state(
            idx_mapping, num_sampled, self.req_states.num_computed_tokens.gpu
        )

    def _merge_ec_connector_no_forward(
        self, scheduler_output: SchedulerOutput, output: ModelRunnerOutput
    ) -> ModelRunnerOutput:
        """Let the EC connector send/recv on a step with no work to run."""
        return ModelRunnerOutput.with_ec_conn_output(
            output,
            self.ec_connector.no_forward(scheduler_output).ec_connector_output,
        )

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        is_profile: bool = False,
        context_len: int = 0,
    ) -> ModelRunnerOutput | IntermediateTensors | None:
        if not dummy_run:
            # Update the request states.
            self.update_pp_decode_requests()
            self.finish_requests(scheduler_output)
            self.free_states(scheduler_output)
            self.add_requests(scheduler_output)
            self.update_requests(scheduler_output)
            self.block_tables.apply_staged_writes()
            if scheduler_output.total_num_scheduled_tokens == 0:
                # No need to run the model.
                empty_output = self.kv_connector.no_forward(scheduler_output)
                return self._merge_ec_connector_no_forward(
                    scheduler_output, empty_output
                )

        # Get batch descriptor and sync across DP ranks.
        num_reqs = len(scheduler_output.num_scheduled_tokens)
        num_toks = scheduler_output.total_num_scheduled_tokens
        max_query_len = max(scheduler_output.num_scheduled_tokens.values())
        batch_req_state, uniform_tok_count = self.gather_batch_req_state(
            scheduler_output, dummy_run
        )
        if batch_req_state is not None:
            num_toks = batch_req_state.num_tokens

        num_active_loras = 0
        if self.lora_config:
            req_ids = list(scheduler_output.num_scheduled_tokens.keys())
            num_active_loras = get_num_active_loras_for_dispatch(
                self.lora_config, self.lora_state, req_ids, dummy_run
            )

        skip_compiled = False
        if self.is_encoder_decoder and scheduler_output.scheduled_encoder_inputs:
            # Encoder-decoder models such as Whisper should run eager/non-compiled
            # when encoder inputs are scheduled, because this step updates
            # cross-attention cache with dynamic encoder outputs.
            skip_compiled = True

        batch_desc, num_tokens_across_dp = dispatch_cg_and_sync_dp(
            self.cudagraph_manager,
            num_reqs,
            num_toks,
            uniform_tok_count,
            self.dp_size,
            self.dp_rank,
            max_query_len=max_query_len,
            need_eager=is_profile or skip_compiled,
            num_active_loras=num_active_loras,
        )

        if batch_desc.num_tokens == 0:
            # All DP ranks have zero tokens to run.
            empty_output = self.kv_connector.no_forward(scheduler_output)
            return self._merge_ec_connector_no_forward(scheduler_output, empty_output)

        if not dummy_run:
            # Common case.
            # Prepare all the inputs and copy to the input buffers.
            assert batch_req_state is not None
            input_batch = self.prepare_inputs(
                scheduler_output, batch_req_state, batch_desc
            )
            block_tables, slot_mappings = self.prepare_attn(input_batch)
            # Mamba "align" pre-copy: migrate recurrent state across block
            # boundaries before the forward. Runs only on real batches, and
            # before model_state.prepare_attn gathers num_accepted_tokens so the
            # boundary reset is visible to the attention metadata.
            self.model_state.preprocess_state(
                input_batch,
                block_tables,
                self.kv_cache_config,
                self.req_states.num_computed_tokens.gpu,
            )

            if self.lora_config:
                # Activate LoRA adapters.
                lora_inputs = self.lora_state.make_lora_inputs(
                    input_batch.req_ids,
                    input_batch.idx_mapping_np,
                    input_batch.num_scheduled_tokens,
                )
                self._set_active_loras(*lora_inputs)
        else:
            # No actual tokens to run. A dummy run for DP or memory profiling.
            dummy_num_reqs = batch_desc.num_reqs or num_reqs
            input_batch = InputBatch.make_dummy(
                dummy_num_reqs,
                batch_desc.num_tokens,
                self.input_buffers,
                max_query_len=batch_desc.max_query_len,
            )
            if not skip_attn_for_dummy_run:
                block_tables, slot_mappings = self.prepare_dummy_attn(input_batch)
                if context_len:
                    set_dummy_context(
                        input_batch,
                        self.block_tables,
                        context_len,
                        self.kv_cache_config.num_blocks,
                        self.max_model_len,
                    )
            else:
                assert batch_desc.cg_mode != CUDAGraphMode.FULL, (
                    "Attention metadata must be prepared for dummy runs when using "
                    "FULL cudagraph mode."
                )
                block_tables = None
                slot_mappings = None

        attn_metadata = None
        slot_mappings_by_layer = None
        if not (dummy_run and skip_attn_for_dummy_run):
            assert slot_mappings is not None
            slot_mappings_by_layer = build_slot_mappings_by_layer(
                slot_mappings, self.kv_cache_config
            )
            assert block_tables is not None
            attn_groups = self.attn_groups
            if dummy_run and is_profile:
                # Mamba layers take a cheap warmup path with no metadata;
                # attention metadata is still built so those kernels tune.
                attn_groups = [
                    [g for g in groups if not isinstance(g.kv_cache_spec, MambaSpec)]
                    for groups in attn_groups
                ]
            attn_metadata = self.model_state.prepare_attn(
                input_batch,
                batch_desc.cg_mode,
                block_tables,
                slot_mappings,
                attn_groups,
                self.kv_cache_config,
                # FULL replay reads capture-time metadata buffers. Re-stage them
                # from the zeroed dummy block tables instead of retaining state
                # indices from the previous real batch.
                for_capture=dummy_run and batch_desc.cg_mode == CUDAGraphMode.FULL,
            )

        input_ids = input_batch.input_ids
        inputs_embeds = None
        ec_connector_output = None
        if self.uses_inputs_embeds and self.is_first_pp_rank:
            # Prepare inputs_embeds (MM encoder outputs and/or prompt_embeds
            # overlay). Only first PP rank prepares them.
            if dummy_run:
                # Obtain embeddings of correct shape for compiled model.
                inputs_embeds = self.model_state.dummy_inputs_embeds(
                    input_batch.num_tokens_after_padding
                )
            else:
                scheduled_encoder_inputs = scheduler_output.scheduled_encoder_inputs
                if self.supports_mm_inputs and self.lora_config is not None:
                    set_active_mm_loras(
                        model=self.model,
                        lora_manager=self.lora_manager,
                        encoder_cache=self.encoder_cache,
                        req_id_to_index=self.req_states.req_id_to_index,
                        lora_state=self.lora_state,
                        scheduled_encoder_inputs=scheduled_encoder_inputs,
                    )
                with self.ec_connector.maybe_get_output(
                    scheduler_output
                ) as ec_connector_output:
                    inputs_embeds = self.model_state.prepare_inputs_embeds(
                        scheduled_encoder_inputs, input_batch, self.req_states
                    )
        model_inputs = {
            "input_ids": input_ids,
            "positions": input_batch.positions,
            "inputs_embeds": inputs_embeds,
            "intermediate_tensors": None,
            # NOTE: Values returned by `prepare_inputs` will override the default
            # values above.
            **self.model_state.prepare_inputs(input_batch, self.req_states),
        }
        normalize_model_token_inputs(self.model, model_inputs)
        if not self.is_first_pp_rank:
            # Update for non-first PP ranks.
            model_inputs["input_ids"] = None
            model_inputs["inputs_embeds"] = None

            # Prepare the intermediate tensors.
            assert intermediate_tensors is not None
            assert self.intermediate_tensors is not None
            n = input_batch.num_tokens_after_padding
            new_tensors = {
                k: v[:n]
                if dummy_run
                else v[:n].copy_(intermediate_tensors.tensors[k][:n])
                for k, v in self.intermediate_tensors.tensors.items()
            }
            model_inputs["intermediate_tensors"] = IntermediateTensors(new_tensors)
            del intermediate_tensors

        # Update the EPLB meta.
        self.eplb.prepare_forward(self.model_config, input_batch.num_tokens)

        self.step_timing.record_batch(
            input_batch, batch_desc.cg_mode == CUDAGraphMode.FULL
        )
        self.step_timing.forward_start()

        # Run model.
        if batch_desc.cg_mode == CUDAGraphMode.FULL:
            # Use explicit cudagraph replay for FULL mode.
            # NOTE(woosuk): Here, we don't need to pass the input tensors,
            # because they are already copied to the CUDA graph input buffers.
            assert self.cudagraph_manager is not None
            self.kv_connector.pre_forward(scheduler_output)
            model_output = self.cudagraph_manager.run_fullgraph(batch_desc)
        else:
            # For piecewise and eager mode, just call model().
            batch_descriptor = BatchDescriptor(
                num_tokens=input_batch.num_tokens_after_padding,
                has_lora=self.lora_config is not None,
                num_active_loras=batch_desc.num_active_loras,
            )

            with set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=input_batch.num_tokens_after_padding,
                cudagraph_runtime_mode=batch_desc.cg_mode,
                num_tokens_across_dp=num_tokens_across_dp,
                batch_descriptor=batch_descriptor,
                slot_mapping=slot_mappings_by_layer,
                skip_compiled=skip_compiled,
                is_padding=input_batch.is_padding,
            ):
                self.kv_connector.pre_forward(scheduler_output)
                if batch_desc.cg_mode == CUDAGraphMode.PIECEWISE:
                    # Run the PIECEWISE graph (compiled PW cudagraph or breakable
                    # cudagraph, chosen inside run_pw_graph). cg_mode is only
                    # PIECEWISE after the cudagraph manager exists.
                    assert self.cudagraph_manager is not None
                    model_output = self.cudagraph_manager.run_pw_graph(
                        self.model, model_inputs
                    )
                else:
                    # Eager (NONE): call the raw model directly.
                    model_output = self.model(**model_inputs)

        if self.is_last_pp_rank:
            if self.use_aux_hidden_state_outputs:
                assert isinstance(model_output, tuple)
                hidden_states, aux_hidden_states = model_output
            else:
                assert isinstance(model_output, torch.Tensor)
                hidden_states = model_output
                aux_hidden_states = None
            output_intermediate_tensors = None
        else:
            assert isinstance(model_output, IntermediateTensors)
            hidden_states = None
            aux_hidden_states = None
            output_intermediate_tensors = model_output

        routed_experts = None
        if not dummy_run and (capturer := self.routed_experts_capturer) is not None:
            assert slot_mappings is not None
            routed_experts = capturer.get_routed_experts(slot_mappings, num_toks)

        finished_req_ids = scheduler_output.finished_req_ids
        self.execute_model_state = ExecuteModelState(
            input_batch=input_batch,
            attn_metadata=attn_metadata,
            slot_mappings_by_layer=slot_mappings_by_layer,
            hidden_states=hidden_states,
            aux_hidden_states=aux_hidden_states,
            finished_req_ids=finished_req_ids,
            ec_connector_output=ec_connector_output,
            routed_experts=routed_experts,
            num_spec_tokens_to_schedule=(
                scheduler_output.resolve_num_spec_tokens_to_schedule(
                    self.num_speculative_steps
                )
            ),
        )

        if not self.is_last_pp_rank:
            # Non-last PP rank: return IntermediateTensors for sending.
            return output_intermediate_tensors
        return None

    @torch.inference_mode()
    @step_eplb_after()
    def sample_tokens(
        self, grammar_output: GrammarOutput | None
    ) -> AsyncOutput | ModelRunnerOutput | None:
        if self.execute_model_state is None:
            # The prior execute_model call must have failed.
            return None

        input_batch = self.execute_model_state.input_batch
        attn_metadata = self.execute_model_state.attn_metadata
        slot_mappings_by_layer = self.execute_model_state.slot_mappings_by_layer
        hidden_states = self.execute_model_state.hidden_states
        aux_hidden_states = self.execute_model_state.aux_hidden_states
        finished_req_ids = self.execute_model_state.finished_req_ids
        ec_connector_output = self.execute_model_state.ec_connector_output
        routed_experts = self.execute_model_state.routed_experts
        num_spec_tokens_to_schedule = (
            self.execute_model_state.num_spec_tokens_to_schedule
        )
        self.execute_model_state = None

        if not self.is_last_pp_rank:
            # Non-last PP rank: hidden_states is None because this rank produced
            # IntermediateTensors instead of final hidden states. Receive the
            # sampled tokens broadcast from the last rank and update local state.
            assert self.pp_handler is not None
            all_decode_next = self.pp_handler.receive(input_batch)
            # Optimistically update num_computed_tokens for entire batch here.
            # Will be adjusted for rejections if necessary in update_requests.
            self.postprocess_num_computed_tokens(input_batch)
            if not all_decode_next:
                # Might contain non-final prefill chunks, which will be scheduled
                # in the immediate next step (rather than in pp_size steps).
                self.model_state.postprocess_state(input_batch.idx_mapping, 0)

            # Post-step KV connector related operations.
            kv_connector_output = self.kv_connector.post_forward(finished_req_ids)
            # The first PP rank holds the encoder cache, so pass its EC output on.
            output = ModelRunnerOutput.with_kv_conn_output_only(kv_connector_output)
            return ModelRunnerOutput.with_ec_conn_output(output, ec_connector_output)

        # Last rank: sample tokens
        hidden_states, input_batch = pcp.maybe_restore_pcp_for_sampling(
            self.pcp_manager, hidden_states, input_batch
        )

        sampler_output, num_sampled, num_rejected = self.sample(
            hidden_states, input_batch, grammar_output
        )

        if self.pp_handler is not None:
            # Broadcast to non-last PP ranks (handles spec decode multi-token).
            self.pp_handler.broadcast(
                sampler_output.sampled_token_ids,
                num_sampled,
                num_rejected,
                input_batch,
            )

        assert self.prompt_logprobs_worker is not None
        prompt_logprobs_dict = self.prompt_logprobs_worker.compute_prompt_logprobs(
            self.model.compute_logits,
            hidden_states,
            input_batch,
            self.req_states.all_token_ids.gpu,
            self.req_states.num_computed_tokens.gpu,
            self.req_states.prompt_len.np,
        )

        # Prepare the model runner output.
        model_runner_output = ModelRunnerOutput(
            req_ids=input_batch.req_ids,
            # NOTE(woosuk): req_id_to_index is unused in this model runner.
            # Only for compatibility with the existing model runner and scheduler.
            req_id_to_index={req_id: i for i, req_id in enumerate(input_batch.req_ids)},
            sampled_token_ids=None,  # type: ignore
            prompt_logprobs_dict=prompt_logprobs_dict,  # type: ignore[arg-type]
        )
        # Start async output copy here so that it can overlap with speculator proposal.
        async_output = AsyncOutput(
            model_runner_output=model_runner_output,
            sampler_output=sampler_output,
            num_sampled_tokens=num_sampled,
            main_stream=self.main_stream,
            copy_stream=self.output_copy_stream,
            check_ep_fault=self.check_ep_fault,
            routed_experts=routed_experts,
        )

        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None
        if self.speculator is not None and self.speculator.supports_mm_inputs:
            # Get cached multimodal embeddings for draft forward.
            # NOTE: This is done here because postprocess updates
            # num_computed_prefill_tokens.
            # The EAGLE/MTP drafter reads one position ahead of the target.
            # TODO(TheEpicDolphin): Gather MM embeddings for all speculative
            # steps during multi-module MTP.
            mm_inputs = self.model_state.gather_mm_embeddings(
                input_batch, draft_lookahead=1
            )

        # Postprocess results and update request states.
        # NOTE: This is intentionally done after creating the AsyncOutput,
        # ensuring that `copy_event` is recorded before calling postprocess.
        # This sequencing may slightly reduce latency as async D2H copy does not
        # need to wait for the postprocess to finish.
        self.postprocess_sampled(
            input_batch.idx_mapping,
            sampler_output.sampled_token_ids,
            num_sampled,
            num_rejected,
            input_batch.query_start_loc,
        )

        draft_tokens_for_next_step: torch.Tensor | None = None
        if self.speculator is not None and num_spec_tokens_to_schedule > 0:
            assert self.sampler is not None
            # Let the target override the hidden state fed to the drafter
            # (e.g. DeepSeek V4 MTP needs the pre-hc_head residual). The
            # target returns a persistent buffer sized at max_num_batched_tokens;
            # slice to the active token count that propose() expects.
            spec_hidden_states = hidden_states
            if hasattr(self.model, "get_mtp_target_hidden_states"):
                pre_hc_hidden_states = self.model.get_mtp_target_hidden_states()
                spec_hidden_states = pre_hc_hidden_states[: hidden_states.shape[0]]  # type: ignore[union-attr]
            with use_workspace_lane(self._draft_workspace_lane):
                draft_tokens = self.speculator.propose(
                    input_batch,
                    attn_metadata,
                    slot_mappings_by_layer,
                    spec_hidden_states,
                    aux_hidden_states,
                    num_sampled,
                    num_rejected,
                    self.req_states.last_sampled_tokens,
                    self.req_states.next_prefill_tokens,
                    self.sampler.sampling_states.temperature.gpu,
                    self.sampler.sampling_states.seeds.gpu,
                    num_speculative_tokens=num_spec_tokens_to_schedule,
                    mm_inputs=mm_inputs,
                )
                draft_tokens = limit_draft_tokens(
                    draft_tokens,
                    num_spec_tokens_to_schedule,
                    self.num_speculative_steps,
                )
            num_draft_tokens = draft_tokens.shape[1]
            if num_draft_tokens > 0:
                self.req_states.draft_tokens[
                    input_batch.idx_mapping, :num_draft_tokens
                ] = draft_tokens
                draft_tokens_for_next_step = self.req_states.draft_tokens[
                    input_batch.idx_mapping, :num_draft_tokens
                ]
            else:
                draft_tokens_for_next_step = draft_tokens
            if self.adaptive_verification is not None:
                self.adaptive_verification.record_confidences(
                    self.speculator.draft_token_confidence_probs, input_batch
                )
        elif self.speculator is not None:
            draft_tokens_for_next_step = self.req_states.draft_tokens[
                input_batch.idx_mapping, :0
            ]

        if self.num_speculative_steps > 0:
            # Spec-decode and diffusion LLMs both use draft tokens but the latter does
            # not have a speculator (i.e. self.speculator is None)
            self.draft_tokens_handler.set_draft_tokens(
                input_batch,
                (
                    draft_tokens_for_next_step
                    if draft_tokens_for_next_step is not None
                    else self.req_states.draft_tokens[input_batch.idx_mapping]
                ),
            )

        # Post-step KV connector related operations.
        kv_connector_output = self.kv_connector.post_forward(finished_req_ids)
        model_runner_output.kv_connector_output = kv_connector_output
        model_runner_output.ec_connector_output = ec_connector_output

        return async_output

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        return self.draft_tokens_handler.get_draft_tokens()

    @torch.inference_mode()
    @step_eplb_after()
    def pool(self) -> AsyncPoolingOutput | ModelRunnerOutput | None:
        if self.execute_model_state is None:
            # The prior execute_model call must have failed.
            return None

        input_batch = self.execute_model_state.input_batch
        hidden_states = self.execute_model_state.hidden_states
        finished_req_ids = self.execute_model_state.finished_req_ids
        ec_connector_output = self.execute_model_state.ec_connector_output
        self.execute_model_state = None

        # Post-step KV connector related operations.
        kv_connector_output = self.kv_connector.post_forward(finished_req_ids)

        if not self.is_last_pp_rank:
            self.postprocess_num_computed_tokens(input_batch)
            output = ModelRunnerOutput.with_kv_conn_output_only(kv_connector_output)
            return ModelRunnerOutput.with_ec_conn_output(output, ec_connector_output)

        assert self.pooling_runner is not None
        pooler_output, finished_mask = self.pooling_runner.pool(
            hidden_states, input_batch, self.req_states
        )

        # Build the model runner output.
        model_runner_output = ModelRunnerOutput(
            req_ids=input_batch.req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(input_batch.req_ids)},
            kv_connector_output=kv_connector_output,
            ec_connector_output=ec_connector_output,
        )
        async_output = AsyncPoolingOutput(
            model_runner_output=model_runner_output,
            pooler_output=pooler_output,
            finished_mask=finished_mask,
            main_stream=self.main_stream,
            copy_stream=self.output_copy_stream,
        )

        self.postprocess_num_computed_tokens(input_batch)
        return async_output

    def postprocess_num_computed_tokens(self, input_batch: InputBatch) -> None:
        # Update the number of computed tokens.
        post_update_num_computed_tokens(
            input_batch.idx_mapping,
            self.req_states.num_computed_tokens.gpu,
            input_batch.query_start_loc,
        )

    def shutdown(self) -> None:
        """Release GPU tensors (model weights, KV caches, workspace) so that
        memory is reclaimable when running in the same process."""
        torch.accelerator.synchronize()
        self.cudagraph_manager = None
        if hasattr(self, "kv_caches"):
            self.kv_caches.clear()
        if hasattr(self, "attn_groups"):
            self.attn_groups.clear()
        if hasattr(self, "kv_cache_config"):
            del self.kv_cache_config
        if hasattr(self, "model_state") and self.model_state.supports_mm_inputs:
            self.model_state.encoder_runner.clear()
        free_before_shutdown(self.vllm_config)
        if hasattr(self, "model_state"):
            del self.model_state
        if getattr(self, "speculator", None) is not None:
            self.speculator = None
        if hasattr(self, "model"):
            del self.model

        gc.collect()
        torch.accelerator.empty_cache()
        logger.debug("Cleaned up model weights, KV caches, and workspace")

    ########### EPLB methods start ###########
    @property
    def eplb_state(self):
        return self.eplb.state

    @eplb_state.setter
    def eplb_state(self, state) -> None:
        self.eplb.state = state

    @property
    def eep_eplb_suppressed(self) -> bool:
        return self.eplb.suppressed

    @eep_eplb_suppressed.setter
    def eep_eplb_suppressed(self, suppressed: bool) -> None:
        self.eplb.suppressed = suppressed

    def setup_eplb_from_mapping(
        self,
        expanded_physical_to_logical: torch.Tensor,
    ) -> None:
        self.eplb.setup_from_mapping(
            self.model,
            self.model_config,
            expanded_physical_to_logical,
        )

    ########### EPLB methods end ###########

    # Out-of-tree hardware runners can select a PCP manager class.
    @property
    def pcp_manager_cls(self) -> type[pcp.PCPManager]:
        return pcp.PCPManager


class ExecuteModelState(NamedTuple):
    input_batch: InputBatch
    attn_metadata: dict[str, Any] | None
    slot_mappings_by_layer: dict[str, torch.Tensor] | None
    hidden_states: torch.Tensor | None
    aux_hidden_states: list[torch.Tensor] | None
    finished_req_ids: set[str]
    ec_connector_output: ECConnectorOutput | None
    routed_experts: RoutedExpertsTensors | None
    num_spec_tokens_to_schedule: int


class BatchReqState(NamedTuple):
    """CPU request state for a scheduled batch, in batch (sorted) order."""

    req_ids: list[str]
    num_scheduled_tokens: np.ndarray  # [num_reqs]
    # May be less than scheduler_output.total_num_scheduled_tokens:
    # adaptive verification trims the draft budget before running.
    num_tokens: int
    idx_mapping_np: np.ndarray  # [num_reqs]
    prefill_len_np: np.ndarray  # [num_reqs]
    num_computed_prefill_tokens_np: np.ndarray  # [num_reqs]
    is_prefilling_np: np.ndarray  # [num_reqs]
    has_prefill: bool


def sort_batch_req_ids(
    num_tokens_per_req: dict[str, int],
    draft_tokens: dict[str, list[int]],
    decode_query_len: int,
) -> list[str]:
    # Order verification/decode -> short_extend -> prefill;
    # split_decodes_and_prefills relies on decode-like requests leading.
    key = lambda r: (
        not draft_tokens.get(r),
        (num := num_tokens_per_req[r]) != decode_query_len,
        num,
    )
    return sorted(num_tokens_per_req, key=key)


# --- E35 candidate addition. Every byte above this marker is the serving image's
# vllm/v1/worker/gpu/model_runner.py (Apache-2.0, vLLM project;
# sha256 f232398c93cf3193166136e823343840a41ebde8669253f56ce63a0023930cb4).
# Confidence-based verify length decided in the runner. Enabled at boot with
# VLLM_E35_ENABLE=1. A step participates only when its SchedulerOutput holds exactly one
# request with k_hi scheduled drafts (identical on every rank). Tensor-parallel rank 0 reads
# the policy flag (VLLM_E35_POLICY_FLAG: "ema", "lag1" or "hybrid") and decides k in
# {k_lo, k_hi} from the DFlash2 speculator's per-position draft confidence:
# - ema: k_hi (no trim);
# - lag1: the previous draft's confidence, already on the host, no wait;
# - hybrid: as lag1 unless its margin is small, then wait for the current draft's copy.
# Rank 0 always broadcasts its k over the tensor-parallel CPU group, so every rank trims
# identically; errors on rank 0 broadcast k_hi. With k < k_hi the runner verifies a copy of
# the SchedulerOutput holding k drafts: the trimmed drafts count as rejected for the
# scheduler, whose blocks were reserved for k_hi.
import dataclasses as _e35_dataclasses
import math as _e35_math
import os as _e35_os
import time as _e35_time

_e35_logger = init_logger(__name__ + ".e35")
_E35_ENABLED = _e35_os.environ.get("VLLM_E35_ENABLE", "") == "1"
_E35_POLICY_FLAG = _e35_os.environ.get("VLLM_E35_POLICY_FLAG", "")
_E35_FLAG_MAX_BYTES = 64
_E35_K_LO, _E35_K_HI = 3, 7
_E35_STEP_MS = {3: 58.0, 7: 81.0}
_E35_HYBRID_MARGIN = 0.005
_E35_HYBRID_WAIT_MS = 2.8
_E35_LOG_EVERY = 200
_E35_CALIBRATION = (
    ((0.0, 0.0), (0.17521, 0.0), (0.35857, 0.227273), (0.43052, 0.252381), (0.48871, 0.252381), (0.53882, 0.272727), (0.59207, 0.406504), (0.64739, 0.406504), (0.6896, 0.465686), (0.73191, 0.465686), (0.77488, 0.567568), (0.81432, 0.567568), (0.85625, 0.567568), (0.88639, 0.615385), (0.92149, 0.615385), (0.94208, 0.653509), (0.95798, 0.653509), (0.96899, 0.653509), (0.97837, 0.731544), (0.98582, 0.731544), (0.99106, 0.804348), (0.99408, 0.886076), (0.99661, 0.9), (0.99777, 0.90566), (0.99861, 0.90566), (0.99914, 0.955556), (0.99949, 0.965517), (0.9997, 0.977444), (0.99982, 0.981928), (0.99989, 0.981928), (0.99993, 0.986928), (0.99996, 0.986928), (0.99998, 0.98939), (0.99999, 0.98939), (1.0, 0.99543)),
    ((0.0, 0.0), (0.065938, 0.0), (0.193289, 0.03125), (0.251084, 0.05042), (0.291154, 0.05042), (0.326954, 0.066667), (0.364286, 0.15873), (0.400331, 0.160494), (0.434068, 0.168675), (0.467635, 0.177273), (0.499877, 0.177273), (0.531832, 0.266667), (0.562884, 0.266667), (0.59542, 0.334197), (0.627072, 0.334197), (0.657826, 0.334197), (0.689615, 0.334197), (0.716948, 0.334197), (0.753988, 0.361111), (0.788158, 0.438931), (0.82514, 0.438931), (0.857447, 0.438931), (0.881748, 0.454545), (0.908198, 0.454545), (0.92927, 0.606061), (0.947542, 0.644628), (0.964039, 0.644628), (0.97524, 0.644628), (0.985624, 0.72381), (0.99026, 0.72381), (0.994038, 0.838384), (0.99639, 0.867647), (0.99778, 0.867647), (0.99887, 0.8875), (0.99939, 0.960396), (0.99967, 0.964286), (0.99983, 0.978723), (0.99991, 0.985714), (0.99995, 0.987474), (0.99997, 0.987474), (0.99998, 0.987474), (0.99999, 0.987474), (1.0, 0.987474)),
    ((0.0, 0.0), (0.04346, 0.0), (0.115546, 0.007968), (0.151517, 0.007968), (0.190699, 0.007968), (0.214428, 0.036145), (0.24055, 0.051613), (0.267861, 0.051613), (0.29728, 0.052632), (0.324057, 0.052632), (0.353089, 0.052632), (0.37688, 0.10596), (0.404392, 0.10596), (0.433308, 0.10596), (0.459036, 0.10596), (0.49144, 0.115385), (0.522493, 0.153846), (0.548851, 0.198582), (0.575547, 0.198582), (0.606968, 0.198582), (0.636152, 0.198582), (0.667039, 0.27381), (0.700375, 0.27381), (0.732384, 0.27381), (0.763629, 0.32243), (0.798432, 0.32243), (0.833183, 0.32243), (0.866428, 0.380952), (0.895945, 0.515306), (0.925381, 0.515306), (0.948411, 0.6), (0.966748, 0.704615), (0.97976, 0.704615), (0.987281, 0.704615), (0.992268, 0.704615), (0.995723, 0.834061), (0.99768, 0.834061), (0.99885, 0.862745), (0.99941, 0.9375), (0.99968, 0.983871), (0.99985, 0.983871), (0.99992, 0.985075), (0.99996, 0.988372), (0.99998, 0.996132), (0.99999, 0.996132), (1.0, 0.996132)),
    ((0.0, 0.0), (0.019982, 0.0), (0.070206, 0.0), (0.097152, 0.0), (0.122281, 0.0), (0.141746, 0.008152), (0.160551, 0.008152), (0.179799, 0.008152), (0.199097, 0.008152), (0.218644, 0.018182), (0.242208, 0.025), (0.265001, 0.026247), (0.285578, 0.026247), (0.310368, 0.026247), (0.333759, 0.026247), (0.358001, 0.046392), (0.379178, 0.046392), (0.406054, 0.05), (0.436827, 0.050725), (0.462997, 0.10559), (0.490511, 0.10559), (0.518722, 0.165563), (0.550173, 0.165563), (0.578798, 0.169355), (0.60869, 0.207692), (0.63827, 0.207692), (0.674542, 0.227848), (0.709856, 0.252874), (0.751071, 0.277108), (0.789226, 0.297619), (0.83148, 0.311927), (0.872736, 0.49697), (0.905043, 0.49697), (0.93465, 0.556962), (0.960912, 0.756906), (0.975573, 0.756906), (0.986361, 0.757576), (0.992522, 0.815789), (0.996233, 0.828571), (0.99837, 0.830645), (0.9992, 0.964912), (0.9996, 0.972892), (0.99982, 0.972892), (0.99991, 0.972892), (0.99995, 0.972892), (0.99998, 0.993958), (0.99999, 0.993958), (1.0, 0.993958)),
    ((0.0, 0.0), (0.009793, 0.0), (0.042006, 0.0), (0.061761, 0.0), (0.076223, 0.0), (0.094067, 0.0), (0.108669, 0.002404), (0.123614, 0.002404), (0.139552, 0.002404), (0.156473, 0.002404), (0.1709, 0.002404), (0.185338, 0.007092), (0.201802, 0.013514), (0.216843, 0.013514), (0.236811, 0.013514), (0.257347, 0.013514), (0.277544, 0.013514), (0.300176, 0.020833), (0.326297, 0.041237), (0.347108, 0.041237), (0.371944, 0.041237), (0.395843, 0.041237), (0.422527, 0.047945), (0.446771, 0.047945), (0.479351, 0.107143), (0.50811, 0.137566), (0.53809, 0.137566), (0.568102, 0.16129), (0.606811, 0.178571), (0.645818, 0.178571), (0.686835, 0.291139), (0.735139, 0.3125), (0.782318, 0.375), (0.825127, 0.431034), (0.877547, 0.463415), (0.913497, 0.525424), (0.945833, 0.680628), (0.970723, 0.680628), (0.98452, 0.680628), (0.99213, 0.844828), (0.996112, 0.892857), (0.998441, 0.910995), (0.99927, 0.910995), (0.99966, 0.943548), (0.99985, 0.952381), (0.99992, 0.96875), (0.99998, 0.992366), (1.0, 0.992366)),
    ((0.0, 0.0), (0.003357, 0.0), (0.028453, 0.0), (0.039549, 0.0), (0.049633, 0.0), (0.062897, 0.0), (0.073938, 0.0), (0.083856, 0.0), (0.094174, 0.0), (0.107587, 0.0), (0.121995, 0.0), (0.135209, 0.0), (0.146081, 0.0), (0.162476, 0.007463), (0.175011, 0.007463), (0.187085, 0.007463), (0.20087, 0.007463), (0.216308, 0.007463), (0.233198, 0.011561), (0.253601, 0.011561), (0.274991, 0.011561), (0.297576, 0.011561), (0.320417, 0.032641), (0.344854, 0.032641), (0.376506, 0.032641), (0.40388, 0.032641), (0.432232, 0.034884), (0.469338, 0.122449), (0.505452, 0.133333), (0.543638, 0.161491), (0.58009, 0.161491), (0.617939, 0.207792), (0.662997, 0.207792), (0.706096, 0.266667), (0.751958, 0.266667), (0.792247, 0.369048), (0.842994, 0.392857), (0.88674, 0.42735), (0.935263, 0.576923), (0.967306, 0.605505), (0.98378, 0.686275), (0.99312, 0.840909), (0.997292, 0.88), (0.99879, 0.88), (0.99951, 0.959184), (0.99983, 0.959184), (0.99994, 0.990566), (0.99999, 0.990566), (1.0, 0.990566)),
    ((0.0, 0.0), (0.001174, 0.0), (0.017155, 0.0), (0.025609, 0.0), (0.033107, 0.0), (0.039954, 0.0), (0.049361, 0.0), (0.057141, 0.0), (0.065679, 0.0), (0.073486, 0.0), (0.081981, 0.0), (0.09181, 0.0), (0.100386, 0.0), (0.112069, 0.0), (0.123412, 0.0), (0.135168, 0.0), (0.146728, 0.0125), (0.159432, 0.0125), (0.170875, 0.015337), (0.184001, 0.015337), (0.196745, 0.015337), (0.214472, 0.015337), (0.234748, 0.018987), (0.259237, 0.018987), (0.280991, 0.018987), (0.309465, 0.018987), (0.338552, 0.018987), (0.373951, 0.052023), (0.401256, 0.052023), (0.433582, 0.087302), (0.472835, 0.087302), (0.511104, 0.166667), (0.552829, 0.166667), (0.590132, 0.173913), (0.637153, 0.196429), (0.685771, 0.311475), (0.735238, 0.311475), (0.786529, 0.364407), (0.83486, 0.364407), (0.888138, 0.421053), (0.934936, 0.47561), (0.96918, 0.616667), (0.984952, 0.756098), (0.994414, 0.823529), (0.998141, 0.823529), (0.99943, 0.935484), (0.99984, 0.962264), (0.99998, 0.990385), (1.0, 1.0)),
)


def e35_parse_policy(raw) -> str:
    """"lag1" or "hybrid" for exact ASCII flag content, else "ema"."""
    if raw is None or len(raw) > _E35_FLAG_MAX_BYTES:
        return "ema"
    try:
        text = raw.decode("ascii").strip()
    except UnicodeDecodeError:
        return "ema"
    return text if text in ("lag1", "hybrid") else "ema"


def _e35_read_flag(path, opener=open):
    if not path:
        return None
    try:
        with opener(path, "rb") as stream:
            return stream.read(_E35_FLAG_MAX_BYTES + 1)
    except OSError:
        return None


class _E35PolicySwitch:
    def __init__(self, path, clock=_e35_time.monotonic, reader=_e35_read_flag):
        self.path, self.clock, self.reader = path, clock, reader
        self.checked_at = None
        self.policy = "ema"

    def poll(self) -> str:
        now = self.clock()
        if self.checked_at is None or now - self.checked_at >= 0.5:
            self.checked_at = now
            self.policy = e35_parse_policy(self.reader(self.path))
        return self.policy


def _e35_interp(knots, s):
    if s <= knots[0][0]:
        return knots[0][1]
    for (x0, y0), (x1, y1) in zip(knots, knots[1:]):
        if s <= x1:
            return y0 if x1 == x0 else y0 + (y1 - y0) * (s - x0) / (x1 - x0)
    return knots[-1][1]


def e35_predict(conf) -> list:
    """Calibrated survival P(a >= j), non-increasing in j."""
    probs, survival, previous = [], 1.0, 1.0
    for j, knots in enumerate(_E35_CALIBRATION):
        c = float(conf[j]) if j < len(conf) else 0.0
        if not _e35_math.isfinite(c):
            c = 0.0
        survival *= min(max(c, 0.0), 1.0)
        previous = min(previous, _e35_interp(knots, survival))
        probs.append(previous)
    return probs


def e35_margin(conf, extra_ms: float = 0.0) -> float:
    """Expected tokens per millisecond at k_hi minus at k_lo."""
    p = e35_predict(conf)
    hi = (1 + sum(p[:_E35_K_HI])) / (_E35_STEP_MS[_E35_K_HI] + extra_ms)
    lo = (1 + sum(p[:_E35_K_LO])) / (_E35_STEP_MS[_E35_K_LO] + extra_ms)
    return hi - lo


def e35_decide(policy, previous, wait_current):
    """(k, source) from the policy, the previous draft's confidence (or None) and a
    callable that waits for the current draft's confidence (returns None if unusable)."""
    if policy not in ("lag1", "hybrid"):
        return _E35_K_HI, "ema"
    if previous is not None:
        margin = e35_margin(previous)
        if policy == "lag1" or abs(margin) > _E35_HYBRID_MARGIN:
            return (_E35_K_HI if margin > 0 else _E35_K_LO), "lag1"
    elif policy == "lag1":
        return _E35_K_HI, "no_data"
    current = wait_current()
    if current is None:
        return _E35_K_HI, "no_data"
    return (_E35_K_HI if e35_margin(current, _E35_HYBRID_WAIT_MS) > 0 else _E35_K_LO), "lag0"


def e35_participating_request(scheduler_output):
    """The lone request id when the step verifies exactly k_hi drafts for one request."""
    counts = scheduler_output.num_scheduled_tokens
    drafts = scheduler_output.scheduled_spec_decode_tokens or {}
    if len(counts) != 1 or len(drafts) != 1:
        return None
    (req_id, num_tokens), = counts.items()
    spec = drafts.get(req_id)
    if spec is None or len(spec) != _E35_K_HI or num_tokens != _E35_K_HI + 1:
        return None
    return req_id


def e35_trim(scheduler_output, req_id, k):
    """A copy verifying only the first k drafts of req_id; the original is untouched."""
    removed = len(scheduler_output.scheduled_spec_decode_tokens[req_id]) - k
    counts = dict(scheduler_output.num_scheduled_tokens)
    counts[req_id] -= removed
    drafts = dict(scheduler_output.scheduled_spec_decode_tokens)
    drafts[req_id] = list(drafts[req_id][:k])
    return _e35_dataclasses.replace(
        scheduler_output, num_scheduled_tokens=counts, scheduled_spec_decode_tokens=drafts,
        total_num_scheduled_tokens=scheduler_output.total_num_scheduled_tokens - removed)


def _e35_confidence(speculator, req_id, index, wait):
    """Host confidence of the request's draft from `index` recorded steps ago (0 = the
    current draft, recorded by the latest propose(); 1 = the one right before it), or None
    when the request has no such entry. Older entries are never used: their ring slots may
    already hold newer drafts."""
    store = getattr(speculator, "_e35_store", None)
    entries = store.get(req_id) if store else None
    last_seq = getattr(speculator, "_e35_last_seq", None)
    if not entries or last_seq is None:
        return None
    # The request must hold the current draft (recorded by the latest propose) and, for the
    # previous draft, the entry right before it: never bridge a gap in its own history.
    if entries[-1][0] != last_seq:
        return None
    if index == 0:
        _, event, host = entries[-1]
    elif len(entries) >= 2 and entries[-2][0] == last_seq - 1:
        _, event, host = entries[-2]
    else:
        return None
    if wait:
        event.synchronize()
    elif not event.query():
        return None
    return host.tolist() if hasattr(host, "tolist") else list(host)


class _E35State:
    def __init__(self):
        from vllm.distributed.parallel_state import get_tp_group
        self.tp = get_tp_group()
        self.rank0 = self.tp.rank_in_group == 0
        self.switch = _E35PolicySwitch(_E35_POLICY_FLAG) if self.rank0 else None
        self.buffer = torch.zeros(1, dtype=torch.int32)
        self.counters = {"steps": 0, "k_lo": 0, "k_hi": 0, "lag1": 0, "lag0": 0,
                         "no_data": 0, "ema": 0, "errors": 0}
        self.waits, self.broadcasts = [], []

    def broadcast(self, k: int) -> int:
        self.buffer[0] = k
        torch.distributed.broadcast(self.buffer, src=self.tp.ranks[0], group=self.tp.cpu_group)
        return int(self.buffer[0])

    def log(self, policy):
        c = self.counters
        if not self.rank0 or c["steps"] % _E35_LOG_EVERY:
            return
        pct = lambda v, q: round(sorted(v)[min(len(v) - 1, int(q * len(v)))], 3) if v else None
        _e35_logger.info(
            "E35_RUNNER_K policy=%s %s wait_ms_p50=%s wait_ms_p90=%s bcast_ms_p50=%s "
            "bcast_ms_p90=%s", policy, " ".join(f"{k}={v}" for k, v in c.items()),
            pct(self.waits, 0.5), pct(self.waits, 0.9), pct(self.broadcasts, 0.5),
            pct(self.broadcasts, 0.9))
        self.waits.clear()
        self.broadcasts.clear()


_e35_original_execute_model = GPUModelRunner.execute_model


def _e35_execute_model(self, scheduler_output, *args, **kwargs):
    dummy = kwargs.get("dummy_run", args[1] if len(args) > 1 else False)
    if not _E35_ENABLED or dummy:
        return _e35_original_execute_model(self, scheduler_output, *args, **kwargs)
    req_id = e35_participating_request(scheduler_output)
    if req_id is None:
        return _e35_original_execute_model(self, scheduler_output, *args, **kwargs)
    state = self.__dict__.get("_e35_state")
    if state is None:
        try:
            state = self._e35_state = _E35State()
        except Exception:
            # Peers are about to wait in the broadcast; fail loudly rather than strand them.
            _e35_logger.exception("E35_RUNNER_K state initialisation failed")
            raise
    k, policy = _E35_K_HI, "ema"
    if state.rank0:
        try:
            policy = state.switch.poll()
            speculator = getattr(self, "speculator", None)
            previous = _e35_confidence(speculator, req_id, 1, wait=False)
            started = [None]

            def wait_current():
                started[0] = _e35_time.perf_counter()
                return _e35_confidence(speculator, req_id, 0, wait=True)

            k, source = e35_decide(policy, previous, wait_current)
            if started[0] is not None:
                state.waits.append((_e35_time.perf_counter() - started[0]) * 1000.0)
            state.counters[source] += 1
        except Exception as error:  # noqa: BLE001 - rank 0 still broadcasts k_hi
            k = _E35_K_HI
            state.counters["errors"] += 1
            if state.counters["errors"] <= 3:
                _e35_logger.warning("E35_RUNNER_K_ERROR %s", type(error).__name__)
    started = _e35_time.perf_counter()
    k = state.broadcast(k)
    if state.rank0:
        state.broadcasts.append((_e35_time.perf_counter() - started) * 1000.0)
    state.counters["steps"] += 1
    state.counters["k_lo" if k < _E35_K_HI else "k_hi"] += 1
    state.log(policy)
    if k < _E35_K_HI:
        scheduler_output = e35_trim(scheduler_output, req_id, k)
    return _e35_original_execute_model(self, scheduler_output, *args, **kwargs)


GPUModelRunner.execute_model = _e35_execute_model
_e35_logger.info("E35_RUNNER_K_READY enabled=%d flag=%s margin=%s wait_ms=%s", int(_E35_ENABLED),
                 _E35_POLICY_FLAG or "unset", _E35_HYBRID_MARGIN, _E35_HYBRID_WAIT_MS)


# --- E36 candidate addition. Every byte above this marker is the promoted E35 runner
# (scripts/node/experiments/e03/e35-runner-k/model_runner.py,
# sha256 4aa67bc0379d1e39649b7eb5297f247eaa0ad7597ed087e876dc81a4075e2e36).
# With VLLM_E36_LM_HEAD_W8A16 set to a non-zero integer, the vocab-parallel lm_head that the
# target shares with the DFlash2 drafter becomes INT8 W8A16 once both models are loaded and
# before any CUDA graph is captured (vllm/models/glm5next/nvidia/e36_lm_head_w8a16.py). A
# failed conversion raises: a rank that kept BF16 would gather mismatched logit shards.
import os as _e36_os

_e36_original_load_model = GPUModelRunner.load_model


def _e36_load_model(self, *args, **kwargs):
    result = _e36_original_load_model(self, *args, **kwargs)
    if _e36_os.environ.get("VLLM_E36_LM_HEAD_W8A16", "0").strip() not in ("", "0"):
        from vllm.models.glm5next.nvidia import e36_lm_head_w8a16 as _e36

        if _e36.enabled():
            _e36.convert_shared_lm_head(self.model, getattr(self, "speculator", None))
    return result


GPUModelRunner.load_model = _e36_load_model
