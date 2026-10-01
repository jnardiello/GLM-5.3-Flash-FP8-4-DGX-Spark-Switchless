# E37 collective microbenchmark extract

Recorded outcome: **discard**, independent review gives technical NO-GO. This is a
four-node CUDA collective microbenchmark, with zero native Rigmark suites or requests.
The [report](../../../benchmarks/experiments/2026-10-01-e37-allreduce-butterfly.md)
describes the measured algorithms, decision and retained error history.

[results.json](results.json) retains the frozen protocol, source and native receipt
hashes, 177 timing cells, 96 qualifying numerical aggregates and the critical
three-block comparisons against P0/NCCL. Arrays use the adjacent column names.
Timing percentiles take the maximum rank duration for each logical operation;
numerical counts count the common logical output once, rather than four times.
Private site configuration, absolute paths and payloads are omitted.

All 12 native rank receipts are complete and retained privately without byte changes.
P0's recovered archive transfer failure and P2's failed preflight before GPU launch are
preserved in private evidence. The checked million-operation mixed soak
is not run because confirmed latency disqualifies every candidate. The 1,002,000
timed operations do not establish a checked soak PASS. E36 and frozen baselines
remain unchanged.
