# Bounded API admission candidate

This experimental ASGI middleware bounds host-side request retention before vLLM reads the
body, parses JSON or tokenizes prompts. It is a resilience candidate, not a promoted default,
and its limits have not been established as suitable for mixed production traffic.

Mount `middleware.py` at `/opt/tp4/tp4_admission.py`, retain the existing
`PYTHONPATH=/opt/tp4`, and add exactly one
`--middleware tp4_admission.BoundedAdmissionMiddleware`. The module is instantiated only by
the rank-0 API process; mounting it and setting the same environment on every rank keeps the
four-rank recipe symmetric.

The candidate guards every POST request, including inference aliases, tokenization helpers and
unknown paths. It explicitly bypasses `/ping`, `/scale_elastic_ep`,
`/is_scaling_elastic_ep`, and the segment-anchored `/v1/responses/{response_id}/cancel` route.
Cancellation must remain callable while active slots are occupied. Health, metrics and every
non-POST request also bypass it. Bypassed control requests retain their native body handling;
the middleware does not apply its body limit to them. The initial test values are:

| Environment | Initial value | Meaning |
| --- | ---: | --- |
| `TP4_ADMISSION_MAX_ACTIVE` | `6` | Requests allowed into parsing, tokenization and serving |
| `TP4_ADMISSION_MAX_QUEUED` | `128` | Additional requests retained in the FIFO admission wait |
| `TP4_ADMISSION_MAX_BODY_BYTES` | `8388608` | Per-request body limit |
| `TP4_ADMISSION_QUEUE_TIMEOUT_SECONDS` | `1800` | Maximum admission wait |
| `TP4_ADMISSION_REQUEST_TIMEOUT_SECONDS` | `3600` | Wall time after an active slot is acquired |
| `TP4_ADMISSION_BODY_IDLE_SECONDS` | `30` | Maximum wait for each active request-body message |
| `TP4_ADMISSION_SEND_IDLE_SECONDS` | `30` | Maximum wait for each response send |

All seven settings are required to parse as positive values. The queue has finite membership,
and the active semaphore is FIFO under the pinned Python runtime. A queued request is never
passed to the downstream ASGI application, so this module does not read or buffer its body.
Uvicorn and the kernel can still retain bounded transport data outside the application. ASGI
cannot observe a queued client's disconnect without reading the body; that entry is released
when it reaches the active slot, reaches its queue timeout, its task is cancelled, or the
server shuts down. This is the principal delay in queued-cancellation cleanup.

A full or expired queue returns HTTP 503, `Retry-After`, and an OpenAI-shaped error with type
`overloaded` and code `admission_queue_full` or `admission_queue_timeout`. An active request
that reaches its wall timeout before response headers gets 503 and
`admission_request_timeout`; body limit and body-idle failures get 413 and 408. Once response
headers have been sent, timeout and stream failures abort the response instead of appending a
false JSON success. An active slot remains owned until the downstream application returns,
fails or is cancelled, including streamed response delivery.

After acquiring an active slot, the middleware reads and retains at most one bounded request
body before calling FastAPI. This keeps its 413 and 408 responses outside FastAPI's body-parser
exception handling and caps middleware body storage at `MAX_ACTIVE * MAX_BODY_BYTES`. Once the
complete body is coalesced into one replay message, disconnect-listener receives are not subject
to the body-idle timer; the request wall timer still applies. Coalescing can transiently copy an
active body; transport buffers and FastAPI's later JSON/model allocations remain additional.
The pinned image uses FastAPI 0.136.3 and Starlette 1.6.0; its FastAPI routing layer converts
generic body-read exceptions to HTTP 400, which is why validation happens before downstream.

Rank 0 emits one `TP4_ADMISSION_READY` JSON record with all selected limits and
`TP4_ADMISSION_STATE` JSON after each counter transition. State includes a monotonic timestamp,
`active`, `queued`, `inflight`, cumulative `admitted_total` and rejection/timeout counters,
without request IDs, paths, headers or
payloads. Use those records to measure queue occupancy, rejection rate and release behavior.

The 128-entry queue and timeout values are starting points for a bounded test. They do not
prove a memory bound for the engine, Uvicorn sockets, client transport buffers or outputs held
by vLLM. Measure host memory with the campaign guard, exercise slow and disconnected clients,
and retain the existing scheduler cap, allocator trim and KV-pool controls while evaluating
this candidate. A result is promotable only after the mixed-load, cancellation and soak cases
establish a sustainable limit and its latency/performance cost is documented.

`manifest.json` pins the runtime image identity and deployed module. `SHA256SUMS` contains only
the files required on the nodes. The offline test exercises admission ordering, overload,
cancellation, body and stream bounds, exception cleanup, telemetry and manifest integrity.
