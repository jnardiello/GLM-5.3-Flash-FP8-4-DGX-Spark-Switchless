# E21: 8-bit weights for the residual BF16 attention projections

This candidate extends the accepted E20 mechanism, INT8 symmetric group-128 weights with
native Marlin below 2,048 flattened input rows and a dequantized BF16 GEMM at or above it,
from the KDA input projection to four further families. It was measured in three
complete native suites and promoted to the default recipe on September 23, 2026.

The default recipe sets `VLLM_E21_BF16_RESIDUE_W8A16=1`. With the flag unset or `0`,
the hook module performs exactly the accepted E20 conversion and nothing else.

## What it converts

| Family | Layers | Shape per rank (N × K) | Stored in the checkpoint as |
| --- | ---: | --- | --- |
| KDA `o_proj` | 34 | 4096 × 2048 | BF16 |
| MLA `o_proj` | 11 | 4096 × 4096 | block FP8, dequantized to BF16 on load |
| MLA `fused_qkv_a_proj` | 11 | 2048 × 4096 | block FP8, dequantized to BF16 on load |
| MLA `q_b_proj` | 11 | 4096 × 1536 | block FP8, dequantized to BF16 on load |

The MLA rows deserve a note. The checkpoint stores those projections as block FP8, but its
`quant_method` is `fp8`, and the model code passes a quantization config to MLA layers only
for `modelopt_mixed` checkpoints. The model loader therefore dequantizes them to BF16 and
serves them as BF16 GEMMs. For these families the INT8 step is a second quantization of an
already dequantized weight, so numeric error is measured against the FP8-dequantized
reference, not against an original BF16 weight that does not exist.

Excluded on purpose: the KDA f/g gate projections, which drive a recurrent decay state
where small errors accumulate; MLA `kv_b_proj`, which the attention backend absorbs and
which sits on the FP8 KV cache and SparkCache path; the sparse-attention indexer
projections, which select the tokens attention reads; `lm_head`; and the DFlash2 drafter.

## Expected effect and its limits

An offline attribution estimates these families at about 6.6% of decode kernel time. The
accepted E20 conversion turned 9.1% of kernel time into roughly a 4-5% end-to-end gain, so
the working estimate here is about 3.0-3.6% end-to-end. That ratio comes from only two
accepted E20 runs whose comparison also changed the KV pool, so treat the estimate as a
reason to measure, not as a prediction. The frozen reference's run-to-run spread is about
1.7-1.8% on the primary concurrency metrics, so an effect below about 2% is not
distinguishable with three suites.

Converting these weights frees roughly 0.5 GiB per rank net of the 76 MiB of added
scratch, because 8-bit weights replace BF16 ones. The KV pool stays fixed.

## Prepare while the current service stays online

The offline test needs only Python's standard library:

```sh
python3 scripts/tests/test-bf16-residue-config.py
./scripts/check.sh
```

It verifies the pins, that the hook module is the accepted source plus one gated call,
family selection and shape validation against the checkpoint dimensions recorded in
[manifest.json](manifest.json), the exact 2,048-row dispatch threshold, the flag-off and
idempotent paths, four-rank launcher parity, and the overlay refusals. It does not execute
Marlin, Triton or CUDA.

Two checks must pass before a window is requested, because either failure would otherwise
cost a coordinated restart:

1. **The installed fused QKV-A projection.** The forward of `DeepSeekV2FusedQkvAProjLinear`
   comes from the engine image and is not in this repository. Upstream vLLM has a
   low-latency path in that forward that reads `self.weight` directly. Read the class
   source from the pinned image without starting a serving container, and confirm that
   path is absent or cannot activate for this shape and GPU. If it can, drop the
   `fused_qkv_a_proj` family before any window.
2. **Numeric error on real weights.** For one layer of each family, quantize the rank-0
   partition exactly as the candidate does and record the relative L2 error of weights and
   of outputs on random inputs at M = 1, 6 and 16. E20 recorded about 0.70%. A family whose
   error is materially larger stops the candidate before the window.

## Coordinated transition

**Promoted on September 23, 2026.** The default `cluster.env.example` now encodes this
candidate, measured as the [E21 reference](../../../../../docs/benchmarks/baselines/2026-09-23-e21.md);
the immediate return is `TP4_ENV=scripts/node/reference/baseline-20260919-e03.env`. The
transition below is the record of the measurement window and applies only to a site
configuration that still holds the E03 recipe.

This overlay applies on top of the default E03 recipe at six sequences and refuses any other
base, including an operational overlay that changes the sequence count, because the
comparison reference was measured at six. Use the operations procedure in
[docs/operations.md](../../../../../docs/operations.md), including the fabric checks and both
functional gates within 120 seconds of `/health` 200.

1. Stop the stack with the overlay that started it.
2. Deploy and start with `TP4_ENV=scripts/node/experiments/e03/bf16-residue/delta.env`, and
   keep that value for every command in the window.
3. Verify the rank-0 engine log contains both `E20_KDA_INPUT_W8A16_READY` and
   `E21_BF16_RESIDUE_W8A16_READY` with 67 modules, the four families listed above,
   `mla_layout` `q_lora` and 79,691,776 added scratch bytes. A missing signature or any
   refusal message invalidates the activation.
4. Run three complete native Rigmark suites on the same loaded processes, following the
   frozen reference's source and settings and a fresh cache salt per suite.

## Return

Stop with this overlay, then deploy and start with the overlay the owner selected before
the window. The accepted E20 hook and scratch files are never modified, and the candidate
used its own SparkCache namespace, so the accepted cache is untouched.

Compare all 16 metrics with the frozen E03 medians and record the outcome under the
[benchmark procedure](../../../../../docs/benchmarks/README.md). Conflicting primary
results are `decision_required`.
