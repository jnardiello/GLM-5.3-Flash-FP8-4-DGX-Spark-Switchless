#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One process per rank, no model weights. Run only after coordinated TP4 down.

Requires RANK, MASTER_ADDR and MASTER_PORT, the R10 image, patched NCCL and
the same rank-local fabric environment as the serving launcher.
"""
import datetime
import json
import os

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.models.deepseek_v4.nvidia.b12x import B12xMHCResidual
from vllm.models.glm5next.nvidia.mhc_prefill_sharding import PREFILL_ROWS, PrefillOwnership
from vllm.v1.worker.workspace import init_workspace_manager


def main():
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    dist.init_process_group("gloo", rank=rank, world_size=4,
                            timeout=datetime.timedelta(seconds=180))
    comm = PyNcclCommunicator(dist.group.WORLD, device=device)
    owner = PrefillOwnership(comm, rank)
    init_workspace_manager(device)
    results = []

    def check(label, actual, expected, *, exact=False):
        a, b = actual.float(), expected.float()
        delta = (a - b).abs()
        results.append({"check": label, "shape": list(a.shape),
                        "max_absolute_error": delta.max().item(),
                        "max_relative_error": (delta / b.abs().clamp_min(1e-6)).max().item(),
                        "relative_l2_error": (delta.norm() / b.norm().clamp_min(1e-6)).item(),
                        "exact": exact})
        if exact:
            assert torch.equal(actual, expected), label
        else:
            torch.testing.assert_close(a, b, atol=0.03125, rtol=0.02, msg=label)

    # Distinct quarter tags plus within-quarter row tags, all exactly BF16 sums.
    rows = torch.arange(PREFILL_ROWS, device=device)
    tags = ((rows // (PREFILL_ROWS // 4)) * 8 + (rows % 8)).to(torch.bfloat16)
    partial = (tags[:, None].expand(PREFILL_ROWS, 4096) + rank).contiguous()
    reduced = comm.all_reduce(partial)
    local = owner.reduce_scatter(partial)
    check("constructed_reduce_scatter", local, owner.local_view(reduced), exact=True)
    full = owner.all_gather(local)
    expected = (tags[:, None].expand(PREFILL_ROWS, 4096) * 4 + 6).contiguous()
    check("constructed_sum_and_order", full, expected, exact=True)

    torch.manual_seed(3100 + rank)
    partial = torch.randn((PREFILL_ROWS, 4096), device=device, dtype=torch.bfloat16)
    reduced = comm.all_reduce(partial)
    local = owner.reduce_scatter(partial)
    check("bf16_reduce_scatter_vs_all_reduce", local, owner.local_view(reduced))
    check("bf16_all_gather_vs_all_reduce", owner.all_gather(local), reduced)
    del partial, reduced, local, full, expected

    # Real B12X kernels on full and owner rows, including deferred post/pre.
    torch.manual_seed(4401)
    mhc = B12xMHCResidual(hidden_size=4096, hc_mult=4, rms_eps=1e-6,
                         hc_eps=1e-6, sinkhorn_iters=20)
    x = torch.randn((PREFILL_ROWS, 4096), device=device, dtype=torch.bfloat16)
    fn = torch.randn((24, 16384), device=device, dtype=torch.float32) * 0.01
    broadcast = fn.view(24, 4, 4096).sum(1).contiguous()
    scale = torch.full((3,), 0.1, device=device, dtype=torch.float32)
    base = torch.zeros(24, device=device, dtype=torch.float32)
    norm = torch.ones(4096, device=device, dtype=torch.bfloat16)
    kwargs = {"norm_weight": norm, "norm_eps": 1e-6}
    full_state = mhc.run_pre(x, broadcast, scale, base, **kwargs)
    local_state = mhc.run_pre(owner.local_view(x), broadcast, scale, base, **kwargs)
    for i, (a, b) in enumerate(zip(local_state, full_state)):
        check(f"b12x_pre_{i}", a, owner.local_view(b))
    # The candidate keeps the first pre full, then owns views of its state.
    local_state = tuple(owner.local_view(value) for value in full_state)
    for step in range(2):
        # A projection-like partial varies by rank, requiring exactly one reduction.
        partial = (x * ((rank + 1) / 16)).contiguous()
        full_x = comm.all_reduce(partial)
        local_x = owner.reduce_scatter(partial)
        full_state = mhc.run_post_pre(full_x, *full_state[:3], fn, scale, base, **kwargs)
        local_state = mhc.run_post_pre(local_x, *local_state[:3], fn, scale, base, **kwargs)
        for i, (a, b) in enumerate(zip(local_state, full_state)):
            check(f"b12x_deferred_post_pre_{step}_{i}", a, owner.local_view(b))
        check(f"normalized_consumer_order_{step}", owner.all_gather(local_state[3]), full_state[3])
        # Same contraction as DFlash2 auxiliaries and the last base layer.
        full_post = mhc.run_post(full_x, *full_state[:3])
        local_post = mhc.run_post(local_x, *local_state[:3])
        check(f"b12x_post_{step}", local_post, owner.local_view(full_post))
        from vllm.model_executor.layers.mhc import hc_contract
        check(f"dflash_final_full_order_{step}", owner.all_gather(hc_contract(local_post, 4)),
              hc_contract(full_post, 4))
    torch.cuda.synchronize()
    dist.barrier()
    print("E03_GPU_CHECK " + json.dumps({"rank": rank, "status": "PASS", "checks": results,
          "peak_allocated_bytes": torch.cuda.max_memory_allocated()}, sort_keys=True), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
