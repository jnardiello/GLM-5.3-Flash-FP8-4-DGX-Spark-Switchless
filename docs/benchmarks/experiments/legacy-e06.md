# E06: increase prefill scheduling budget to 16384

**Recorded outcome:** discard under the recorded memory constraints.

Partial native receipt and original failure report are available. The JSON retains 15 decode rows; 21 calls completed out of 22 attempted, with 6 completed prefill calls documented without saved native rows.

The first cold 32K prefill exhausted rank-0 unified-memory headroom and earlyoom terminated the worker. Concurrency and 64K phases were not reached. This is a failed partial run, not a valid series median.

The imported native JSON is incomplete: prefill and concurrency sections are absent. The validator reports 7 findings for the incomplete run and missing sections. No missing native rows or full-run median are reconstructed. The numeric extract includes the 15 saved decode measurements and their supported per-workload summaries.

[Numeric dataset and source digests](../../historical_benchmarks/experiments/legacy-e06/results.json) · [Benchmark index](../README.md)
