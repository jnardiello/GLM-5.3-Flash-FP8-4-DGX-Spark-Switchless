# Prefill step cap diagnostic

This test-only candidate keeps the protected E31 configuration at an 8,192-token
batch ceiling while limiting each scheduler output to 6,912 target tokens. It is a
bounded diagnostic for the observed 4K concurrency-two memory guard, not a promoted
recipe or a claim of general memory admission.

`scheduler.py` is the immutable E29 scheduler plus `scheduler.patch`. With
`VLLM_RESILIENCE_STEP_TOKEN_CAP=6912`, startup refuses unless the configured scheduler
budget is 8,192, the scheduler block is 2,304 and the graph ceiling is 72. The scheduler
starts `token_budget` at 6,912 while retaining the 8,192-token `input_budget`, model
buffers, draft-slot accounting, Mamba alignment, cache policy and preemption refunds.
Unset or zero preserves the E29 behavior.

Source `delta.env` after the prefill-cache-trim delta. It replaces exactly the E29
scheduler mount, appends the strict opt-in and leaves the selected worker and every
other Docker argument unchanged. Keep the composed `TP4_ENV` for every lifecycle
command in the test window.

The startup signature is `RESILIENCE_STEP_TOKEN_CAP_READY configured=8192
effective=6912 block=2304 eager_above=72`. Each output above the graph ceiling emits one
`RESILIENCE_STEP_TOKEN_CAP` JSON receipt with wall and monotonic timestamps, PID,
scheduler sequence, scheduled target-token count, new-request count, configured budget
and effective cap. Decode-sized outputs at or below 72 do not emit a receipt.

Run one cold 4K C1 request, drain, then the previously failing cold 4K C2 case three
times with a full drain between repetitions. Require every eager receipt to report at
most 6,912 target tokens, every request to complete, SparkCache reservations and staging
to return to zero, and all four ranks to remain above the campaign memory guard. Stop
and preserve evidence on the first guard, integrity error or worker loss. A passing
diagnostic establishes only this bounded case; longer contexts and mixed loads remain
pending until measured.
