# Eager-prefill allocator-cache trim diagnostic

This experimental overlay tests one narrow explanation for the memory guard reached by the
first cold 8K concurrency-2 case: free PyTorch allocator pages retained from the preceding
cold/replay requests. It is not a promoted recipe, a performance improvement, or evidence
that concurrency is safe.

`gpu_worker.py` is the validated production worker plus one strict opt-in hook. After worker
warmup, after prior pipeline sends finish, and before a known eager prefill forward, the hook
calls `torch.accelerator.empty_cache()`. A step qualifies only when the real
`SchedulerOutput` contains a new request or a cached request whose `is_context_phase` is
true, and its scheduled-token count exceeds every configured CUDA Graph capture size. The
E31 configuration has capture sizes through 72. Missing or malformed graph or scheduler
metadata skips the trim and emits `PREFILL_CACHE_TRIM_SKIP`; decode-only and captured steps
are never trimmed. Dummy and warmup calls run while the hook is disarmed.

Every worker emits `PREFILL_CACHE_TRIM_READY` after warmup. Every selected step emits one
JSON `PREFILL_CACHE_TRIM` receipt with rank, PID, wall and monotonic timestamps, scheduled
tokens, new-request count, the current `before`/`empty_cache`/`after` stage, before/after
CUDA allocated and reserved bytes, host `MemAvailable`, duration, and the change in reserved
bytes. The reclaimed value is a diagnostic observation, not a memory-safety guarantee. A
Torch, allocator, or telemetry error emits an error receipt at its true stage and is
re-raised; it cannot be counted as a successful trim.

`delta.env` changes only the existing `gpu_worker.py` mount and adds
`VLLM_PREFILL_CACHE_TRIM=1`. It refuses an unmatched or already modified mount list. Use the
same `TP4_ENV` for every command in the window. Stop the whole four-rank stack under the
recipe that started it; start this overlay only through the coordinated procedure in
`docs/operations.md`. Return by stopping with this overlay and starting the default without
`TP4_ENV`.

The preserved campaign history remains a failure: after three cold 8K C1 requests and three
8K replays, the first cold 8K C2 case crossed the 768 MiB `MemAvailable` guard on rank 0.
The diagnostic does not relabel that stopped case.

For the finite candidate check, start from a healthy four-rank load, pass both functional
gates, and require one ready identity per rank. First run one cold 4K C1 case, drain requests
and SparkCache reservations, then run one cold 4K C2 case and drain again. If both complete
above the guard, repeat the previously failing cold 8K C2 case three times, draining between
repetitions. Stop on the guard, an unexpected worker exit, or any trim error. Preserve all
four-rank receipts and existing telemetry. A zero-reclaim receipt is valid evidence that the
allocator had no free pages to release; it is not a pass.

`manifest.json` and `SHA256SUMS` pin the production parent and candidate. The offline test
verifies the exact source delta, overlay substitution, fail-closed classification, warmup
gate, call ordering, receipts, and error propagation without importing Torch or vLLM.
