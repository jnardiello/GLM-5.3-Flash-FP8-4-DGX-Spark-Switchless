# E27: native prefill cadence while other requests decode

**Recorded outcome: promote.** The owner accepted it; it is the
[September 24 E27 reference](../baselines/2026-09-24-e27.md).

## How the candidate was found

The search started with long-context decode. A diagnostic reused the frozen Rigmark client
and prefill prompt construction to send one request at a time with 128 to 64K tokens of
cold context. Decode stayed flat, between 75 and 80 tok/s, so context length itself was not
a lever. The filler text is repetitive and drafter acceptance was about 97%, so these
absolute values are higher than real workloads.

The same diagnostic then sent four 32K contexts together. Per-stream decode was 6.2, 9.4,
16.0 and 44 tok/s, in the order in which their prefills finished. Engine statistics showed
the cause: while a long prompt prefilled, generation for every running request fell to
0.1–2.9 tokens per second in total. Each step that carries a prefill chunk of up to 8,192
tokens lasts about 2.5 s, and running requests advance once per step.

The serving image already has a native knob for this, `--prefill-schedule-interval`. While
requests decode, it admits prefill work on one step in N and runs decode-only steps in
between. It stops deferring while requests wait in the queue, and a prefill with nobody
decoding is unaffected. E27 sets N = 8 and changes nothing else.

## Measurement

The work was done in one window on one retained load, and each measurement was compared
with E22b:

1. **Rigmark interference phase**, run first. This is a new optional phase: background
   decode before and during another request's cold prefill, at 8K and 32K, with 1 and 3
   running decoders and 3 runs per cell. The E22b reference was measured earlier the same
   day with the same client.
2. **Three complete native suites** with the frozen E22b Rigmark source and settings.
3. **Four 32K contexts arriving together**, repeating the diagnostic.
4. **Two same-day E22b control suites** after one transition back, over the same direct LAN
   client path. The frozen E22b reference used an SSH tunnel.

Results, with details in the [baseline report](../baselines/2026-09-24-e27.md):

- **Decode of running requests during another request's long prefill:** 2.1–2.2 → 6.1–11.5
  tok/s at 8K, and 1.4–1.6 → 7.0–8.7 tok/s at 32K.
- **The arriving request's TTFT:** +15% to +37%, only while others decode.
- **Against the same-day control:**
  - C4 per-stream TTFT +35% and C4 aggregate −3.3%;
  - code and prose decode −1.9% and −2.2%, not separable from run variation;
  - prefill, replay, C1 and C2 unchanged.
- **Four simultaneous 32K contexts:** unchanged against E22b, because the cadence is not
  applied while requests wait.

Measured integrity:
- 162/162 candidate requests and 108/108 control requests, with zero errors;
- 0 lost client probes;
- both post-boot gates PASS.

The live identity check first failed on a pending systemd reload on one rank, unrelated to
E27. After `systemctl daemon-reload`, which restarts no service, it passed.

## Decision

Under the ranking rules, the C4 TTFT cost is a primary-metric regression traded for a gain
that the standard suites do not measure. The outcome was therefore `decision_required`,
and the owner decided to promote.

The obvious follow-up is E27b: apply the cadence only to prefills with at least 2,048
remaining tokens, so that short concurrent prompts are never deferred.

[Portable numeric extract](../../historical_benchmarks/experiments/2026-09-24-e27-prefill-interval/results.json) ·
[Interference extract](../../historical_benchmarks/experiments/2026-09-24-e27-prefill-interval/interference.json) ·
[Long-context diagnostic](../../historical_benchmarks/experiments/2026-09-24-e27-prefill-interval/long-context-diagnostic.json) ·
[Benchmark index](../README.md)
