# E31: pooled-indexer candidates

This overlay holds two changes to the GLM pooled indexer. Both can be switched at runtime on
one load.

**Head gate on tensor cores.** Every MLA layer's pooled indexer weights its 32 index heads
with a gate projected from the layer input (`[rows, 4096] x [4096, 32]`). The accepted
recipe computes that gate as an FP32 GEMM. It runs on the SIMT units and first copies
`hidden_states` to FP32, on every decode step in every MLA layer. Both operands are exact
BF16 values. A BF16 tensor-core GEMM with FP32 accumulation and FP32 output therefore forms
the same products and never rounds the result to BF16. Only the accumulation order differs,
which can reorder pools whose indexer scores are nearly tied. The idea follows the
`GLM53_INDEXER_GATE_TC` switch in the Apache-2.0
[RiNGSiDE patch](https://github.com/othexmr/GLM-5.3-Flash-NVFP4-2x-4x-DGX-Sparks-RiNGSiDE/blob/776054810b1603b0250b55edaafecad4881a9bf5/patches/vllm/tp4/models-glm5next-nvidia-attention.py.patch).

**Speculative-safe C4 tail ring (fix candidate).** The indexer builds one pooled key from
every four tokens. Each request keeps a small tail of recent key/gate rows so that a pool
can be completed across steps. The accepted recipe's tail has four slots, indexed by
`position % 4`. A DFlash verify step writes 1 + K consecutive positions, and later drafts
in the step may be rejected. Their rows overwrite slots that still hold committed members
of the open pool. After the rejection, the next step completes that pool from
rejected-draft keys. Take positions 0 and 1 as committed, and a verify step over 2..9 that
accepts only 2. The next step then builds pool 0..3 from the keys of 8, 9, 6 and the new
3. The target verify has no rollback. The E31 ring has `4 * cdiv(4 + K, 4)` slots (12 for
K = 7), indexed by `position % ring`, and the ape/pool phase stays `position % 4`. No row
of one step can then reach a committed member of the open pool. The design follows the
compressor ring of vLLM PR #58454, which RiNGSiDE also ports.

**Status:**

- **Tail ring:** promoted on September 28, 2026 as the
  [E31 reference](../../../../../docs/benchmarks/baselines/2026-09-28-e31.md).
- **Head gate:** unresolved, off.

The default `cluster.env.example` mounts both files from this directory and names both flag
files, with the switches unset (gate `0`, ring `1`). The production `pooled_indexer.py` and
`ops/glm_kpool.py` under `scripts/node/overrides/` remain the E29 sources, used by the
rollback `scripts/node/reference/baseline-20260925-e29.env`. The overlay below refuses the
E31 default: apply it on that E29 recipe to reproduce the measured load.

## Files

- `pooled_indexer.py` is the production
  [`pooled_indexer.py`](../../../overrides/vllm/models/glm5next/nvidia/pooled_indexer.py) with
  [`pooled_indexer.patch`](pooled_indexer.patch) applied. The patch adds lines and replaces
  only the tail shape.
- `glm_kpool.py` is the production
  [`ops/glm_kpool.py`](../../../overrides/vllm/models/glm5next/nvidia/ops/glm_kpool.py) with
  [`glm_kpool.patch`](glm_kpool.patch) applied. The patch adds the `tail_ring` argument and
  replaces the six tail-offset lines. With `tail_ring = 4`, every index equals the legacy one.
- `delta.env` is the `TP4_ENV` overlay. It swaps only the `pooled_indexer.py` and
  `glm_kpool.py` mounts and adds `-e VLLM_GLM53_INDEXER_GATE_TC_FLAG=/tmp/glm53-indexer-gate-tc`
  and `-e VLLM_GLM53_KPOOL_TAIL_RING_FLAG=/tmp/glm53-kpool-tail-ring`.
- `leaf_gate_tc.py` and `leaf_kpool_tail_ring.py` are single-GPU leaf tests.
- [`manifest.json`](manifest.json) records the base, patch and provenance hashes, and
  `SHA256SUMS` pins the files deployed to the nodes. The launcher verifies it whenever an
  `e31-indexer` file is mounted.

`scripts/tests/test-e31-kpool-tail-ring.py` proves the ring offline. It re-implements the
kernels' index arithmetic, checked verbatim against `glm_kpool.py`. It then simulates chunked
prefill and verify steps with random acceptance. The legacy tail corrupts pools; the E31
ring never does.

## Switches

Each switch has an import-time default and an optional flag file:

| Switch | Default when unset | Flag file variable | `1` | `0` |
| --- | --- | --- | --- | --- |
| `VLLM_GLM53_INDEXER_GATE_TC` | `0` | `VLLM_GLM53_INDEXER_GATE_TC_FLAG` | tensor-core gate | FP32 gate |
| `VLLM_GLM53_KPOOL_TAIL_RING` | `1` | `VLLM_GLM53_KPOOL_TAIL_RING_FLAG` | E31 ring | legacy 4 slots |

- **Defaults.** A switch value other than `0` or `1` stops the engine at import.
- **Flag files.** A flag file containing `1` or `0` overrides the default. Each worker
  re-reads it at most every 0.5 s. A missing or unreadable file, or any other content, means
  the default. When the flag variable is unset, the indexer never touches the filesystem.
- **Head-gate log.** Each worker logs `E31_INDEXER_GATE path=fp32` or
  `E31_INDEXER_GATE path=bf16-tc` on its first forward and on every change.
- **Head-gate probe.** The first BF16 call probes `torch.mm(..., out_dtype=torch.float32)`.
  If that probe fails, the worker logs `E31_INDEXER_GATE bf16-tc unavailable` and stays on
  FP32 until it restarts.
- **Ring log.** Each worker logs `E31_KPOOL_TAIL_RING ring=<n>` on its first forward and on
  every change: 12 for the E31 ring with seven drafts, 4 for the legacy tail.
- **Switch the ring only while the server is idle.** Use the procedure in
  [Same-load A/B](#same-load-ab). The tail contents belong to one ring size, so a running
  request would complete pools from the wrong slots.

The indexer forward is `@eager_break_during_capture`, so it runs eagerly on every CUDA-graph
replay. Both switches therefore take effect without recapture or restart.

## Leaf tests (maintenance window)

Run the leaf tests on one node whose serving stack is **down**, never next to a serving
engine. Deploy copies this directory to `~/tp4/experiments/e03/e31-indexer/`. Use the
serving image so Torch, Triton and vLLM match:

```sh
sudo docker run --rm --gpus all --entrypoint python3 \
  -v "$HOME/tp4/experiments/e03/e31-indexer:/e31:ro" "$IMAGE" \
  /e31/leaf_gate_tc.py --ab-blocks 5
sudo docker run --rm --gpus all --entrypoint python3 \
  -v "$HOME/tp4/experiments/e03/e31-indexer:/e31:ro" "$IMAGE" \
  /e31/leaf_kpool_tail_ring.py --spec-tokens 7
```

`leaf_gate_tc.py` reports, per row count (M = 4 to 48):

- maximum absolute and relative error;
- top-512 pool-selection mismatches, mean Jaccard and near-tie counts, on synthetic scores;
- median eager and CUDA-graph latency;
- with `--ab-blocks`, medians from alternating A/B blocks in one process.

To use a real checkpoint gate, mount `MODEL_DIR` read-only as well. Then pass
`--weights /model/<shard>.safetensors`, adding `--key <tensor name>` when the first
`*weights_proj.weight` in that shard is not the one wanted. The last output line is a JSON
summary. `candidate_path` shows whether `out_dtype` was available.

`leaf_kpool_tail_ring.py` runs the E31 Triton kernels with the production page geometry:
2304-token parent pages, each followed by its FP8 index tail. Several requests share mixed
batches (decode requests first, then prefill requests). They use non-contiguous state slots
and shuffled parent pages. Each request is prefilled in random chunks, then runs verify
steps of 1 + k rows with random acceptance; rejected drafts carry random keys.

Every committed pool is compared with a reference built by one prefill over the committed
tokens only. Sampled pools are also compared with an independent Torch computation of the
pool formula. The script reports, for the reported counterexample and the random schedule:

- the reference checks;
- per ring (legacy 4 and E31): wrong pools, including non-finite ones, pools that differ at
  all, and the maximum absolute difference.

It exits zero only if all of these hold:

- the reference is finite, non-zero, distinct from pool to pool, and matches the
  independent computation;
- the legacy ring corrupts pools in both schedules;
- the E31 ring corrupts none.

`scripts/tests/test-e31-kpool-tail-ring.py` runs the same schedules and geometry through
the offline simulation.

## Same-load A/B

One load serves every variant. Switching needs no restart and no weight reload.

1. **Start the load.** The E31 default already mounts both files and names both flag files,
   so a same-load A/B of either switch needs no overlay. To reproduce the measured load from
   the E29 recipe instead, stop the running stack with its own overlay. Then deploy and start
   all four ranks with one `TP4_ENV` file that concatenates
   `scripts/node/reference/baseline-20260925-e29.env` and
   `scripts/node/experiments/e03/e31-indexer/delta.env`, using the same `TP4_ENV` for every
   command of the window. The load starts on the FP32 gate and
   the E31 ring. Complete the usual health and functional gates.
2. **Select a variant.** Quiesce every client. On rank 0's `/metrics`, require both
   `vllm:num_requests_running` and `vllm:num_requests_waiting` to be 0 before writing. Then
   write the same values on **all four ranks**:

   ```sh
   sudo docker exec "$CONTAINER" sh -c 'echo 1 > /tmp/glm53-indexer-gate-tc'
   sudo docker exec "$CONTAINER" sh -c 'echo 1 > /tmp/glm53-kpool-tail-ring'
   ```

   For the head-gate A/B, keep the ring fixed and alternate the gate: `0` = A, FP32; `1` =
   B, tensor cores. The E29 recipe itself corresponds to gate `0` and ring `0`.
3. **Verify before each suite.** Re-check `/metrics`: running and waiting must still be 0,
   otherwise stop the series. `sudo docker exec "$CONTAINER" cat` of each flag file must print
   the same value on all four ranks. After at least one second, send one short warm-up
   request. The latest `E31_INDEXER_GATE` and `E31_KPOOL_TAIL_RING` lines in each rank's
   `sudo docker logs "$CONTAINER"` must then name the intended variant. Stop the series on
   any of these:
   - a missing rank;
   - a different value on one rank;
   - an `unavailable` warning.
4. **Run the suites.** Alternate complete native Rigmark suites A, B, A, B, ... on this one
   load, with the current Rigmark policy in `AGENTS.md`: upstream Rigmark, the reference
   flags, no `cache_salt`, a fresh comparison ID and a new `--output` file per suite.
   Never switch during a suite. Ranks on different variants could build or select
   different pools for the same request.
5. **Return.** Run `down` under this overlay, then `deploy`/`up` without `TP4_ENV`.

Report the variants from the same load against each other, with the actual suite counts.

## Promotion note

The legacy tail may already have stored wrong decode-built pools in the cache of the E29
recipe's SparkCache namespace. At the E31 promotion the owner kept that namespace, with no
restart and the warm cache kept. Pools built with the legacy mapping could therefore be
restored for resumed sessions. It is not established that decode-built pools are persisted,
and the external review did not treat this as a confirmed blocker. A fresh namespace would
remove the risk at the cost of a cold cache.
