# E22b: 8-bit weights for the DFlash2 drafter

This component applies the accepted E20/E21 mechanism, INT8 symmetric group-128 weights
with native Marlin, to 30 BF16 linears of the DFlash2 speculative drafter. The drafter's
fused context K/V projection stays in BF16. The drafter checkpoint and every target-model
computation are unchanged.

**Promoted on September 23, 2026.** The default `cluster.env.example` encodes it, measured
as the [E22b reference](../../../../../docs/benchmarks/baselines/2026-09-23-e22b.md); the
immediate return is `TP4_ENV=scripts/node/reference/baseline-20260923-e21.env`. The default
recipe sets `VLLM_E22_DRAFTER_W8A16=1` and `VLLM_E22_CONTEXT_KV_W8A16=0`. With the drafter
flag unset or `0`, the drafter loads exactly as the serving image's own code does.

## What it converts

| Family | Modules | Shape per rank (N × K) |
| --- | ---: | --- |
| `self_attn.qkv_proj` | 5 | 1536 × 4096 |
| `self_attn.o_proj` | 5 | 4096 × 1024 |
| `mlp.gate_up_proj` | 5 | 6144 × 4096 |
| `mlp.down_proj` | 5 | 4096 × 3072 |
| `kernel_projection` of both grouped convolutions | 10 | 1024 × 4096 |

These linears see only the query rows of each draft step, streams × (1 + draft length),
so they always run Marlin and need no scratch. Not converted:
- the fused context K/V projection, which projects every context token; converting it,
  in the superseded [E22 candidate](../../../../../docs/benchmarks/experiments/2026-09-23-e22-drafter-w8a16.md),
  cost about 3% of 8K cold prefill in matched probes;
- `fc`, whose large-prefill scratch would need 160 MiB;
- `lm_head` and `embed_tokens`, absent from the drafter checkpoint and shared with the
  target;
- `candidate_selector.hidden_projection`, with negligible traffic.

## Files

- `qwen3_dflash2.py` is the serving image's `vllm/model_executor/models/qwen3_dflash2.py`
  byte for byte, with one block appended after a marker line. The block gives
  `DFlash2Qwen3ForCausalLM` a `load_weights` that first runs the inherited vendor loader,
  then, when the flag is set, calls `finalize_drafter_w8a16`. The test checks that
  everything before the marker has the vendor hash recorded in `manifest.json`.
- `e22_drafter_w8a16.py` validates every module against shapes derived from the drafter
  config and the tensor-parallel size, then converts. It reuses E21's `_pack_projection`
  and `ResidueW8A16Method` unchanged. Any mismatch refuses before a weight changes. Each
  rank logs `E22_DRAFTER_W8A16_READY` with 30 modules, five families,
  `context_kv_w8a16` false and 0 added scratch bytes.
- `kv-transfer-config-e22b.json` differs from the E21 configuration only in its
  `spark_cache_root`: SparkCache stores drafter state, and the converted drafter changes it.
- `delta-context-bf16.env` is the measured overlay, applied on the complete E21 recipe. It
  refuses the current default.

## Checks

A CPU emulation on the real drafter weights covered TP4 rank-0 partitions of layers 0
and 4, with inputs of 16, 24 and 36 rows. It measured relative weight error of
0.65–0.73% and output error of 0.51–0.98%, the same order as E21. The target model
verifies every drafted token, so the drafter's precision changes acceptance, which
stayed close to E21, rather than the generated tokens. The launcher verifies this
directory's `SHA256SUMS` whenever its files are mounted; `scripts/tests/test-drafter-w8a16-config.py`
covers pins, the load hook, family selection, refusals and four-rank launcher parity.
