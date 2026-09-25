# E27b: the prefill cadence for long prefills only

**Recorded outcome: superseded.** The owner accepted E27b as a candidate; it was promoted
inside [E27c](2026-09-25-e27c-queued-cadence.md), which adds one more flag to the same
scheduler patch.

## Why

E27 admits prefill work on one engine step in eight while other requests decode. That kept
running agents generating while another client loaded a long context, but short prompts
waited for the cadence too. In C4, whose streams start together, per-stream TTFT rose 35%
against a same-day E22b control.

E27b keeps `--prefill-schedule-interval 8` and mounts the serving image's scheduler with
one reviewed patch. On a step where the cadence defers prefill work, a prefill with fewer
than 2,048 remaining uncached tokens is still admitted, up to 2,048 such tokens per step.
Longer prefills wait for the cadence; a deferred long request no longer blocks shorter
ones queued behind it. 2,048 is below the 2,304-token KDA block, so an admitted short
prefill always finishes in one chunk. The [candidate README](../../../scripts/node/experiments/e03/long-prefill-cadence/README.md)
describes the patch.

## Measurement

The window ran on September 24 over direct LAN HTTP to the rank-0 API:

1. **Protocol-2 interference phase on E27**, the same-day reference, with the corrected
   Rigmark extension and a new 1K-token cell.
2. One coordinated transition to E27b, both post-boot gates 3.0 s after `/health` 200,
   `check-f0`, the boot line `E27B_SHORT_PREFILL_READY` and the mounted scheduler hash on all
   four ranks.
3. On one retained E27b load: the interference phase, **three complete native suites**
   with the frozen E22b/E27 Rigmark source and settings, and the four-simultaneous-32K
   diagnostic.

Integrity: 162/162 requests completed with zero native, protocol or runtime errors;
code, prose and structured gates 15/15 each; prefill token counts 54/54; no lost client
probe.

## Results

| Metric | Median | vs E27 | vs E22b control (24/09) |
| --- | ---: | ---: | ---: |
| Code decode | 57.6 | +4.53% | +2.59% |
| C1 end-to-end | 42.6 | -3.98% | -0.77% |
| C2 aggregate | 66.8 | +2.78% | +4.40% |
| C4 aggregate | 95.5 | +0.04% | -3.23% |
| Prose decode | 33.3 | +2.14% | -0.16% |
| Code TTFT | 0.387 s | -3.97% | -0.64% |
| Prose TTFT | 0.376 s | +0.00% | +0.67% |
| C1 per-stream TTFT | 0.345 s | -8.24% | -5.48% |
| C2 per-stream TTFT | 0.460 s | +0.88% | -0.54% |
| C4 per-stream TTFT | 0.524 s | -27.22% | -1.69% |
| Prefill 8K, cold | 2,634.2 | +0.95% | -0.53% |
| Prefill 8K, replay | 9,559.9 | -0.31% | +0.28% |
| Prefill 32K, cold | 2,720.4 | -2.14% | -0.30% |
| Prefill 32K, replay | 37,614.7 | -0.10% | +2.13% |
| Prefill 64K, cold | 2,659.6 | -1.14% | +0.53% |
| Prefill 64K, replay | 40,018.6 | -0.07% | +0.34% |

- **Short prompts no longer wait.** C4 per-stream TTFT returns to the level before E27.
  With one or three agents generating, a 1K prompt reaches its first token in 0.80–0.91 s
  instead of 1.28–1.98 s, and an 8K prompt in 3.8–3.9 s instead of 4.2–5.8 s.
- **Long prompts behave as in E27.** During a 32K prefill the agents keep generating at
  about 7 tok/s, and their net delay is unchanged.
- **Several long contexts arriving together still stall the first stream** (6.1–6.2 tok/s
  in the four-simultaneous-32K diagnostic, as with E27): the image's capacity latch
  switches the cadence off while requests are queued.
- **C1 end-to-end is −4.0% against E27** in all three suites. With a single request E27b
  schedules exactly as E27, and C1 is −0.8% against the same-day E22b control. The owner
  treated it as measurement noise.

## Decision

The owner accepted E27b as a candidate and asked for E27c on top of it, to address the
queued case. E27c was promoted and contains E27b; this record keeps E27b's own evidence.

[Portable numeric extract](../../historical_benchmarks/experiments/2026-09-24-e27b-long-prefill-cadence/results.json) ·
[Interference extract](../../historical_benchmarks/experiments/2026-09-24-e27b-long-prefill-cadence/interference.json) ·
[Four-context diagnostic](../../historical_benchmarks/experiments/2026-09-24-e27b-long-prefill-cadence/long-context-diagnostic.json) ·
[Benchmark index](../README.md)
