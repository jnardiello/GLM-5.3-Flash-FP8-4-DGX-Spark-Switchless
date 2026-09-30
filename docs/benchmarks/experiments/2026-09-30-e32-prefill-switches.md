# E32: runtime switches for the prefill allocator trim and step cap

**Recorded outcome: discard.** The owner closed the experiment after the native suites. The
memory-bounded default is unchanged: trim before every eager prefill step, 6,912-token step cap.

- One load, switched at runtime through flag files: a factorial screen of six arms in six
  blocks (144 measurements), an indexer-gate check, a 22-case resilience re-check, then four
  complete native Rigmark suites A–P–P–A (216 requests).
- Comparison reference: arm A on the same load, which reproduces the
  [memory-bounded operational identity](../../operational-identities/2026-09-29-memory-bounded.json).
  The frozen [E31 record](../baselines/2026-09-28-e31.md) remains the performance baseline; it
  used a 16 GiB pool without trim, cap or admission, so it is not a same-load comparison here.

## Why

The memory-bounded default returns the PyTorch allocator's cached blocks to the system before
every eager prefill step over 72 tokens, and caps a scheduling step at 6,912 tokens. Both
protect host memory, the trim at a cost: a long cold prompt is prefilled in several chunks,
and every chunk after the first pays a reclaim of about 950 MiB (18 ms on rank 0).

## What the candidate changed

An overlay of the default's worker and scheduler added flag files, re-read at most every
0.5 s, so each variant could be selected without a restart. An invalid or missing flag fell
back to the startup behaviour with a warning.

| Switch | Values |
| --- | --- |
| Trim | `1` before every eager prefill step (default); `0` off; `new` only on steps that schedule a new request; `pressure:<MiB>` only below that MemAvailable |
| Step cap | `6912` (default); `0` off; `solo` off while at most one request is active |

## Screen

Paired time-to-first-token differences against arm A in the same block, median of six
blocks (negative is faster):

| Arm (trim, cap) | 32K cold | 64K cold | 17K agent-turn cold | 8K / 32K / 64K replay |
| --- | ---: | ---: | ---: | ---: |
| B (`0`, `6912`) | −1.7% | −1.6% | −2.0% | +0.2 / +0.3 / −1.6% |
| C (`1`, `0`) | −0.2% | 0.0% | −0.5% | +1.4 / +2.5 / +0.5% |
| D (`0`, `0`) | −1.7% | −1.6% | −1.9% | −1.1 / +1.7 / −2.2% |
| E (`new`, `solo`) | −1.1% | −1.7% | −1.5% | +1.3 / +2.8 / +1.8% |
| F (`pressure:3072`, `solo`) | −1.7% | −1.5% | −1.8% | −0.1 / +2.1 / −0.8% |

- The cost is the trim between chunks: every arm without it gains 1–2% on long cold prompts.
- The cap switches had no effect. Intermediate chunks stay at or below 6,912 tokens through
  the KDA block alignment in every mode.
- `0` removes the host-memory protection, and `pressure:3072` never triggered in the
  resilience re-check, so its mechanism stayed unvalidated. `new` was therefore the finalist.
- The tensor-core indexer head gate measured 78.37 against 78.44 ms per verify step (FP32),
  within a 77.95–79.65 ms spread: no effect; it stays off.

## Resilience re-check

Both finalists passed 22/22 cases with the 6,912-token cap: two concurrent cold 8K prompts,
staggered arrivals, two waves of five concurrent 262,128-token contexts, and the 8K case again
after the stress. The lowest rank-0 MemAvailable was 4.84 GiB (`pressure:3072`, no trim) and
4.63 GiB (`new`, 15 trims, all on new-request steps). The other ranks stayed at or above 9.1 GiB.

## Native suites

Arm A is `1`/`6912`; arm P is `new`/`6912`. Each arm value is the mean of its two suites; the
A4-versus-A1 drift is the noise reference.

| Metric | A | P | P vs A | A drift |
| --- | ---: | ---: | ---: | ---: |
| Code decode | 63.3 tok/s | 63.8 tok/s | +0.8% | −2.7% |
| Prose decode | 33.7 tok/s | 34.7 tok/s | +3.0% | −0.3% |
| C1 aggregate | 45.9 tok/s | 45.7 tok/s | −0.4% | +2.7% |
| C2 aggregate | 75.0 tok/s | 72.8 tok/s | −3.0% | −0.4% |
| C4 aggregate | 110.3 tok/s | 104.5 tok/s | −5.2% | −2.0% |
| Cold TTFT 8K / 32K / 64K | 3.12 / 11.82 / 23.52 s | 3.09 / 11.63 / 23.11 s | −0.8 / −1.6 / −1.7% | −0.1 / +0.5 / −0.3% |
| Replay TTFT 8K / 32K / 64K | 0.917 / 0.926 / 1.619 s | 0.946 / 0.933 / 1.679 s | +3.2 / +0.8 / +3.8% | 0.0 / −2.0 / +0.7% |

- **The cost moves to the next request.** Without trims between chunks, the cached ~950 MiB
  is reclaimed on the next step that schedules a new request, which is usually the replay.
  Rank-0 trim receipts show nine such reclaims per P suite (median 14.5 ms) and none in A.
  P saves about 0.4 s on a 64K cold prompt and adds about 0.06 s to every 64K replay.
  Coding-agent sessions replay a growing context on every turn, so the trade is negative after
  a few turns.
- **C2 and C4 are noise.** In the concurrency phase both arms logged the same nine trims per
  suite (one per round, 2–6 MiB each), so the switch cannot explain the difference. P also
  stayed above the frozen E31 values, 68.4 (C2) and 101.4 (C4) tok/s.
- **Integrity:** 216/216 requests completed without errors; 60/60 decode output gates passed
  with a normal stop. The memory guard (stop below 0.75 GiB) never triggered; the lowest
  rank-0 MemAvailable was 4.94 GiB.

## Measurement notes

- **Client and protocol.** Upstream Rigmark `c5a0db0` with the reference flags (defaults plus
  `reasoning_effort` low), no `cache_salt`, a fresh comparison ID per suite, direct LAN HTTP.
- **Excluded series.** The first A suite ran after every node's root filesystem had filled
  up with SparkCache data. The store could not write, and the 64K replay fell back to
  recomputing (2,643 tok/s). The series was stopped and repeated after a cleanup; the
  receipt hash is kept in the extract. The operational default has since gained a SparkCache
  disk-capacity policy.
- **One load.** All measurements share one weight load; the screen and the suites agree on
  the direction of every effect reported above.

[Portable numeric extract](../../historical_benchmarks/experiments/2026-09-30-e32-prefill-switches/results.json) ·
[Benchmark index](../README.md)
