# E27c: keep the prefill cadence while requests are queued

**Recorded outcome: promote.** The owner accepted it; it is the
[September 25 E27c reference](../baselines/2026-09-25-e27c.md).

## Why

The serving image switches the prefill cadence off after a release step that leaves
requests waiting, until the queue is empty. When several long contexts arrive together,
the first stream to start decoding therefore stalls while the others prefill back to
back. That was the situation E27 was meant to fix, and the four-simultaneous-32K
diagnostic showed it unchanged on E27 and [E27b](2026-09-24-e27b-long-prefill-cadence.md).

E27c is E27b plus `VLLM_E27C_CADENCE_WHEN_QUEUED=1`: a throttled step defers long prefills
even when that capacity latch is set, as long as a request is decoding. The latch is still
computed; the cadence and decode-eligibility checks are the vendor ones. The
[candidate README](../../../scripts/node/experiments/e03/queued-cadence/README.md)
describes the patch.

## Measurement

One coordinated transition from E27b, both post-boot gates 3.2 s after `/health` 200,
`check-f0`, both boot lines and the mounted scheduler hash on all four ranks. Then, on one
retained load over direct LAN HTTP: the four-simultaneous-32K diagnostic, the protocol-2
interference phase and **three complete native suites**.

Integrity: 162/162 requests completed with zero native, protocol or runtime errors; prose
and structured gates 15/15; code gate 14/15, because one code answer reached the
8,192-token output budget; prefill token counts 54/54; no lost client probe.

## Results

Four 32K prompts arriving together, in order of first token:

| | 1st stream | 2nd | 3rd | 4th |
| --- | ---: | ---: | ---: | ---: |
| Decode, E27b | 6.1–6.2 tok/s | 9.0–9.1 | 14.7–15.0 | 52–54 |
| Decode, E27c | **11.3–12.5** | **10.9–11.0** | 14.8–15.0 | 68–72 |
| First token, E27b | 15–16 s | 29–30 s | 40–41 s | 55 s |
| First token, E27c | 16–18 s | 29–31 s | 43–44 s | 58–60 s |

The log line `E27C_CADENCE_KEPT_WHILE_QUEUED` confirms that the cadence stayed on with
requests queued. The first stream doubles its speed; it stays near 12 tok/s because each
cadence cycle still holds one prefill step of up to 8,192 tokens, about 2.6 s.

With a single arriving prompt, E27c behaves as E27b in the interference phase. The only
difference beyond noise is the agents' longest pause at 32K, 2.94–2.97 s instead of
2.65 s, in one series.

| Metric | Median | vs E27 | vs E22b control (24/09) | vs E27b |
| --- | ---: | ---: | ---: | ---: |
| Code decode | 56.1 | +1.84% | -0.04% | -2.57% |
| C1 end-to-end | 43.6 | -1.79% | +1.49% | +2.27% |
| C2 aggregate | 66.5 | +2.41% | +4.02% | -0.36% |
| C4 aggregate | 96.7 | +1.24% | -2.07% | +1.21% |
| Prose decode | 33.1 | +1.52% | -0.76% | -0.60% |
| Code TTFT | 0.390 s | -3.23% | +0.13% | +0.78% |
| Prose TTFT | 0.376 s | +0.00% | +0.67% | +0.00% |
| C1 per-stream TTFT | 0.340 s | -9.57% | -6.85% | -1.45% |
| C2 per-stream TTFT | 0.446 s | -2.19% | -3.57% | -3.04% |
| C4 per-stream TTFT | 0.525 s | -27.08% | -1.50% | +0.19% |
| Prefill 8K, cold | 2,558.9 | -1.93% | -3.37% | -2.86% |
| Prefill 8K, replay | 9,432.0 | -1.64% | -1.07% | -1.34% |
| Prefill 32K, cold | 2,689.3 | -3.26% | -1.44% | -1.14% |
| Prefill 32K, replay | 37,056.3 | -1.58% | +0.61% | -1.48% |
| Prefill 64K, cold | 2,646.9 | -1.62% | +0.05% | -0.48% |
| Prefill 64K, replay | 39,931.7 | -0.29% | +0.12% | -0.22% |

In the prefill phase nobody decodes, so E27b and E27c run identical code there. Their
differences of up to 2.9% on prefill and 2.6% on code decode therefore estimate
load-to-load variation, and put E27b's C1 −4.0% against E27 in the same range.

## Decision

Against E27, C4 per-stream TTFT improves by 27% and several long contexts arriving
together no longer freeze the first stream, while prefill and replay are 0.3–3.3% lower,
within the observed variation between loads. Under the ranking rules the prefill changes
made this `decision_required`; the owner decided to promote.

The next lever is a smaller prefill chunk, or a longer interval, while others decode, to
raise the first stream above about 12 tok/s.

[Portable numeric extract](../../historical_benchmarks/experiments/2026-09-25-e27c-queued-cadence/results.json) ·
[Interference extract](../../historical_benchmarks/experiments/2026-09-25-e27c-queued-cadence/interference.json) ·
[Four-context diagnostic](../../historical_benchmarks/experiments/2026-09-25-e27c-queued-cadence/long-context-diagnostic.json) ·
[Benchmark index](../README.md)
