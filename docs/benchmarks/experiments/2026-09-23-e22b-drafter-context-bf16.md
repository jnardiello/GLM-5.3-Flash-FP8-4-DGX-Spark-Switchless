# E22b: 8-bit drafter weights with the context K/V projection kept in BF16

**Recorded outcome: promote, accepted by the owner.** Three complete native Rigmark suites,
162 requests, on one retained load after one coordinated transition. Code decode +3.03%,
C1 +2.23% and prose +2.97% against E21, with no primary throughput regression. The result
became the [September 23 E22b reference](../baselines/2026-09-23-e22b.md).

## What changed

E22b is [E22](2026-09-23-e22-drafter-w8a16.md) with one family removed: the drafter's fused
context K/V projection stays in BF16, exactly as the vendor loader builds it. The other 30
drafter linears use the accepted E20/E21 INT8 group-128 format with native Marlin, since
they only see a few dozen rows per step. The drafter checkpoint and the target model are
unchanged, and the cache uses its own namespace.

| Family | Modules | Shape per rank (N × K) |
| --- | ---: | --- |
| `qkv_proj` | 5 | 1536 × 4096 |
| `o_proj` | 5 | 4096 × 1024 |
| `gate_up_proj` | 5 | 6144 × 4096 |
| `down_proj` | 5 | 4096 × 3072 |
| `kernel_projection` of both grouped convolutions | 10 | 1024 × 4096 |

Not converted: the context K/V projection; `fc`, whose large-prefill scratch would need
160 MiB; `lm_head` and `embed_tokens`, which the drafter shares with the target; and the
small candidate-selector projection. Sources, manifest, overlays and tests are in
[`scripts/node/experiments/e03/drafter-w8a16/`](../../../scripts/node/experiments/e03/drafter-w8a16/README.md).

## Pre-window checks

A CPU emulation on real drafter weights, TP4 rank-0 partitions of layers 0 and 4 at 16, 24
and 36 input rows, measured relative weight error of 0.65–0.73% and output error of
0.51–0.98%, the same order as E21. The drafter override is the serving image's
`qwen3_dflash2.py` byte for byte plus one appended, flag-gated load hook; the offline test
checks the vendor hash of everything before the marker.

## Measurement

The window first ran matched probes on the loaded E22 candidate, then one transition to
E21 for the same probes, then one transition to E22b for the probes and three suites.
E22b reached `/health` 200 in 1,027 seconds; both functional gates passed 3.0 seconds
later; all four ranks logged the E21 signature and `E22_DRAFTER_W8A16_READY` with 30
modules. No container restarted during the series and no competing traffic was observed.

| Metric | E21 | E22b | Change |
| --- | ---: | ---: | ---: |
| Code decode | 54.945 tok/s | 56.611 tok/s | +3.03% |
| Code C1 | 40.718 tok/s | 41.625 tok/s | +2.23% |
| Code C2 | 64.726 tok/s | 65.780 tok/s | +1.63% |
| Code C4 | 95.524 tok/s | 95.901 tok/s | +0.39% |
| Prose decode | 32.441 tok/s | 33.403 tok/s | +2.97% |
| C1 per-stream TTFT | 0.3500 s | 0.3520 s | +0.57% |
| Prefill 8K, cold | 2,644.825 tok/s | 2,610.637 tok/s | -1.29% |

All 16 metrics, per-run values, the matched probes and memory are in the
[baseline report](../baselines/2026-09-23-e22b.md).

## Evidence and limits

- 162/162 requests completed visibly with zero native, protocol or runtime errors; all
  native gates pass 45/45.
- C4 run 1 was 90.213 tok/s, below the E21 range, with high concurrent per-stream TTFT;
  runs 2 and 3 were 101.571 and 95.901 tok/s. Concurrent first-token times move in
  engine-step units in both baselines.
- C2 per-stream TTFT is +4.78% and 64K cold prefill -1.38%; both are within run-to-run
  spread and recorded as observed.
- Drafter acceptance was 0.621 / 0.611 / 0.611 per suite, close to E21 (0.616–0.638) and
  above E22 (0.583–0.608).
- An external review of the E22 results and plan found the method sound and asked for the
  matched probes that led to this variant.
- By owner decision, promotion used launcher-command parity with the measured load instead
  of a separate reproduction run.

[Portable numeric extract and receipt hashes](../../historical_benchmarks/experiments/2026-09-23-e22b-drafter-context-bf16/results.json) · [Benchmark index](../README.md)
