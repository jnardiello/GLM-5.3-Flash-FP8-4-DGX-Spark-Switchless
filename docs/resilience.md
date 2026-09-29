# Memory resilience

This page summarizes the memory protections in the deployed default and what the
September 29, 2026 resilience campaign observed. The
[detailed campaign report](benchmarks/experiments/2026-09-29-sparkcache-resilience.md)
records every case, count, hash and telemetry gap. The
[campaign runbook](resilience-campaign.md) describes how to run such a window on your own
cluster.

## In plain words

- **The problem.** On a GB10 node the operating system and the GPU share 128 GB of memory.
  With the model loaded only a few GiB remain free, and rank 0, which also runs the API
  server, has the least. On the earlier protected recipe, before allocator trim and the step
  cap, two simultaneous 8K-token prompts pushed rank 0 below the campaign's 768 MiB safety
  guard (about 564 MiB free at the last sample).
- **The fix.** The deployed default is designed to make heavy use wait instead of consuming
  memory. The cache connector moves data within a fixed budget; on top of that, unused GPU
  memory is trimmed before eligible prompt processing, prompts are processed in bounded
  steps, the KV pool shrinks from 16 to 14 GiB per rank, and the API accepts a bounded
  number of requests before answering HTTP 503.
- **What we tested.** Five clients at the full 262,144-token context, bursts of 150 clients,
  cancellations, slow clients, invalid requests, injected cache faults and a 99-minute
  mixed load of 840 requests.
- **What we saw.** No out-of-memory kill was observed in the sampled telemetry. On the
  14 GiB pool the lowest sampled free memory on rank 0 was 2.55 GiB, during the 150-client
  bursts. Over the 99-minute load
  free memory rose in two steps during the first 35 minutes and then held steady, as the
  figure below shows.
- **What remains open.** The first long mixed load failed an idle check, and the second was
  ended by the operator before its planned two hours. A 32-client queue wave failed once on
  a controller telemetry check; it and the 64-client wave remain deferred, as do worker
  crashes and combined faults.

**Bottom line:** under the tested loads the deployed default queued or rejected work
instead of running out of memory. That is sampled evidence, not a guarantee for every load.

## Protections

The maximum context remains **262,144 tokens**. Five clients can submit full-context
requests, with waiting and preemption when their KV state cannot fit together.

| Protection | Change and observed result |
| --- | --- |
| SparkCache memory budget | 8 MiB transfers, shared 1 GiB transient budget per rank and a 1 GiB admission floor. Requests completed under the tested cache faults; recompute fallback is inferred from those receipts. Fresh drain proofs verify released reservations and staging. |
| Prefill working memory | Allocator trim plus a 6,912-token step cap; remaining work queues. The corrected 4k/C2 and 8k/C2 cases each passed three repetitions after the earlier memory-guard stops. |
| KV headroom | 14 GiB per rank. Three waves of five requests completed: every request used 245,760 input + 16,384 output = 262,144 tokens. Minimum sampled rank-0 headroom was 3.71 GiB; waiting/preemption was exercised. |
| Bounded API queue | Six active admission slots and 128 waiting. With six blocker streams active, each of three 150-client waves admitted 128 wave requests and explicitly rejected 22 with HTTP 503; accepted requests were accounted for. |
| Cancellation and cache faults | A separately sealed set covers 16 cases × three repetitions and 192 fresh rank drain proofs; invalid API inputs, slow clients and cache failures retain explicit outcomes. |

The [performance page](performance.md#deployed-default) reports the final native Rigmark
suite on this default and the complete rollbacks.

## Memory over a long mixed load

The final mixed load ran for **98m59s**, with 756 complete ordinary responses,
84 intentionally cancelled requests and 41 idle periods. All cancellations and the
terminal drain have fresh four-rank release proofs. The operator ended the run early
after the pause measurements stabilized: its status is **owner-stopped**, and the original
two-hour gate remains incomplete. This workload was sequential, with growing conversations
through 64k and replay; the five-client full-context tests above were separate.

[![Free memory per node over the 99-minute mixed load: rank 0 stays between about 3 and 6 GiB free, ranks 1–3 between about 8 and 10.7 GiB, with two early rises and no downward trend.](plots/experiments/2026-09-29-sparkcache-resilience/memory-over-time.png)](plots/experiments/2026-09-29-sparkcache-resilience/memory-over-time.svg)

| Rank | Minimum sampled available RAM | Available RAM after final drain |
| --- | ---: | ---: |
| 0 | 3.086 GiB | 5.789 GiB |
| 1 | 7.965 GiB | 10.549 GiB |
| 2 | 7.983 GiB | 10.581 GiB |
| 3 | 7.921 GiB | 10.534 GiB |

The figure counts all 840 submitted generation requests, including cancellations. It shows
`MemAvailable`, so higher means more headroom; host and CUDA memory are
not added together. Rank-0 idle headroom rose from 4.006 GiB to about 5.8 GiB and stayed
near that level later in this run. Sampling targets 1 s, with an overlapping 200 ms
handoff monitor during the last ten requests; request counts on the time axis join
controller and node clocks that were not calibrated. The
[figure generator](../scripts/plot-resilience-memory.py) reads the
[portable CSV, summary and reproduction script](historical_benchmarks/experiments/2026-09-29-sparkcache-resilience/portable-data/manifest.json),
which retain the measurements and hashes. The report also keeps a
[per-rank figure against cumulative requests](plots/experiments/2026-09-29-sparkcache-resilience/requests-vs-memavailable.svg)
that separates load, idle and drain samples.

## Limits

The earlier mixed-load attempt remains failed because its API-idle proof timed out;
a later drain does not erase that result. Worker-crash tests, larger queues and further
fault combinations remain deferred. Test cache eviction was 3 GiB toward 2 GiB, whereas
production retains its normal disk policy: these results do not establish bounded disk
growth, unsampled memory headroom or immunity to every failure. The
[report](benchmarks/experiments/2026-09-29-sparkcache-resilience.md#operational-decision-and-remaining-coverage)
records the complete case matrix and telemetry gaps.
