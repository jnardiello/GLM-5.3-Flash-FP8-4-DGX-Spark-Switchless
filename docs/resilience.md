# Production resilience campaign

This runbook exercises the protected SparkCache operational default during one exclusive
initial 8–12 hour production window. Every real client must remain stopped for the
whole window. The campaign may disrupt the inference service and cache, but it never reboots
a host, changes the fabric or interrupts power. All worker recovery uses the existing
four-rank `scripts/tp4ctl` procedure.

The resilience driver is not a benchmark runner. Run upstream, unmodified Rigmark directly
on the protected default before loading the test overlay and again after the driver restores
the selected operational target, using the current baseline flags and a fresh comparison ID. Keep those native
receipts separate and name their private paths in the campaign configuration. A recorded Rigmark
file means only that the external measurement exists; the resilience driver does not reinterpret
or promote it. Without the optional conditional restoration fields described below, the selected
target remains that same protected default.

## Preconditions and authorization

Before the window, require all of the following:

- The four rank-ordered SSH targets, deployment account and management access are confirmed.
- The protected default is deployed, `/health` is 200, both functional gates pass and
  `scripts/check-f0.py` reports the operational identity.
- The service window explicitly covers full-cluster lifecycle operations, deliberate worker
  signals, cache fault injection and synthetic inference load.
- The physical and management path remains available if the coordinated controller cannot
  recover the group.
- A private absolute receipt directory exists with mode `0700`. Do not place site addresses,
  raw telemetry or incident receipts in tracked documentation.
- Linux `/proc`, passwordless `sudo`, `docker`, `ssh`, `scp` and `sha256sum` are available on
  all four targets. The probe and fault controller run through `sudo -n` because worker-created
  instrumentation records are root-owned and mode 0600.

Stop before activating the overlay if any rank is missing, a second stack or foreign GPU job
exists, health is inconsistent, or the current connector/config hashes differ from the
protected default. The adapter is coupled to the reviewed bounded connector and refuses a
different base hash.

## Prepare the test-only overlay

Choose a unique lowercase campaign identifier. Generation writes an ignored bundle below
`scripts/resilience/.campaign/<id>/`. The first command is local and makes no remote change.
Use a restrictive umask for every private configuration, plan and receipt created during the
window:

```sh
umask 077
python3 scripts/resilience/prepare_overlay.py \
  --campaign-id <unique-id> \
  --base-config scripts/node/experiments/e03/sparkcache-ram-budget/kv-transfer-config.json \
  --base-connector third_party/sparkcache/spark_context_cache_connector-ram-budget.py \
  --runtime-variant default \
  --kv-cache-memory-bytes 15032385536 \
  --bounded-admission
```

Review `manifest.json`, `kv-transfer-config.json` and `delta.env`. The generated SparkCache
configuration must differ from the protected default only at `spark_cache_root`, which must be
`/cache/jit/tp4-resilience-<id>`. The overlay mounts the unmodified protected connector at a
private path, verifies its SHA-256 before loading it, and mounts the instrumented wrapper at the
normal connector path. Production payload files are neither rewritten nor replaced on disk.
The KV option is an explicit campaign candidate, not a new protected default. Put the same
positive integer in the campaign JSON as `kv_cache_memory_bytes`; omit both settings to retain
the selected recipe's existing KV pool. The 15,032,385,536-byte example reduces a 16 GiB pool by
12.5%, reserving about 2 GiB more unified-memory headroom while leaving the 262,144-token model
limit unchanged. The tradeoff is less aggregate KV capacity for concurrent requests. The driver
records the value in runtime identity and requires the generated manifest plus every live Docker
command to contain exactly one matching `--kv-cache-memory-bytes` option and exactly one
`--max-model-len 262144` option; it does not adapt either value after startup.

`--bounded-admission` is another test-only candidate switch. It stages the source pinned by the
bounded-admission manifest, adds one read-only middleware mount and its exact environment, and
adds one `--middleware tp4_admission.BoundedAdmissionMiddleware` argument. Put
`"bounded_admission": true` in the private campaign JSON when using it. Omit both the generator
flag and JSON key to preserve the previous overlay bytes and runtime identity. Preflight checks
the manifest and overlay hashes, the sole middleware argument, mount, source hash and every
configured environment value on all four ranks before load begins. The pinned contract admits
every POST except its named cheap control/ping paths and the single-response cancellation route;
non-POST health and metrics reads bypass it. The generator rejects a manifest describing another
route policy.

With the four targets still in rank order, stage the reviewed bundle and initialize an empty,
mode-0700 namespace on each node:

```sh
python3 scripts/resilience/prepare_overlay.py \
  --campaign-id <unique-id> \
  --base-config scripts/node/experiments/e03/sparkcache-ram-budget/kv-transfer-config.json \
  --base-connector third_party/sparkcache/spark_context_cache_connector-ram-budget.py \
  --runtime-variant <same-runtime-variant> \
  --kv-cache-memory-bytes 15032385536 \
  --bounded-admission \
  --force --stage-hosts <rank0> <rank1> <rank2> <rank3> --apply-stage
```

Staging regenerates the bundle: repeat the same optional KV value and bounded-admission flag, or
omit each option from both generation and staging when retaining the selected recipe.

`faultctl.py init` refuses to adopt a nonempty directory without the exact campaign marker.
Staging verifies the copied `manifest.json` hash and every file hash on each target before it
initializes the namespace.
Every cache mutation is limited to the marked `tp4-resilience-<id>` directory. The adapter
holds one cross-process file lock around admission and writes, counts linked files plus the
connector's anonymous staging files, and reserves control headroom below the hard 8 GiB
namespace limit. Buffered writes, `os.write`, `pwrite`, vectored writes and truncation used by
the reviewed connector go through the same check. A different connector hash or namespace
contract fails startup.

The overlay also sets the connector's native capacity to 3 GiB and its low-water mark to
2 GiB, so native eviction remains confined to the exclusive campaign namespace. It replaces
the operational default's disk-capacity pair instead of adding a second one, and refuses any
other capacity form. The generated
SparkCache JSON still changes only the cache root; these capacity values are temporary overlay
environment policy. A largest-context spool plus publication can still need more headroom than
the native capacity allows. If a case cannot create its case-bound target, record it as pending;
do not relax the 8 GiB adapter cap or use a pre-existing cache.

Copy `scripts/resilience/campaign.example.json` to a mode-0600 file outside the checkout and
fill it with the four SSH targets, expanded host-side cache roots, rank-0 endpoint, checkout
path and private receipt directory. The same absolute cache path may appear four times when
each node uses the same deployment home; rank order still matters. Each host root must have the
exact basename `tp4-resilience-<id>`. Set `allow_worker_faults` to true only for a window that
includes the listed TERM, KILL and STOP cases. Validate the complete matrix without network access:

An operator-approved candidate for the final operational boot can be configured with the paired
repository-relative fields `final_restore_env` and `final_operational_identity`. The driver pins
both regular, nonsymlinked files by SHA-256 when it starts. It selects that target only after a
current-runtime protocol-2 soak has run for at least two hours and passed with no fatal result or
safety stop. Omit both fields to retain the protected default; providing only one is a
configuration error. An early abort, failed or incomplete soak, safety stop, or changed pin
selects the protected default without `TP4_ENV`. A selected candidate is started with its final
environment and verified with `scripts/check-f0.py --identity` against the configured identity;
the final native Rigmark row records that verified restoration identity. The initial native row
always retains the protected-default identity.

The optional `runtime_variant` is `default` when omitted and requires the generated
`scripts/resilience/.campaign/<id>/delta.env`. `prefill-cache-trim` deterministically composes
that file with the reviewed allocator diagnostic and generates `delta-prefill-trim.env`.
`prefill-step-cap` composes the same base, allocator diagnostic, and reviewed scheduler cap in
that order and generates `delta-prefill-step-cap.env`. Select the same value with
`prepare_overlay.py --runtime-variant`; do not hand-copy or append a derived overlay. The
generator hashes the selected file and every component in `manifest.json`, and the driver
rejects every other variant/path pairing.
For execution it records the selected variant, overlay path and SHA-256, the local candidate
manifest SHA-256 for the diagnostic variant, and the expected worker SHA-256. The default worker
pin comes directly from the protected operational source; default execution does not depend on
the experimental manifest. Step-cap identity additionally records the pinned scheduler manifest
and source hash. Preflight requires the matching trim environment, exact worker mount and
in-container worker hash; step-cap also requires `VLLM_RESILIENCE_STEP_TOKEN_CAP=6912`, its exact
scheduler mount, and its in-container scheduler hash on every rank. This diagnostic selection
does not change the campaign namespace, deadlines, fault budget, or Rigmark contract.

An optional `targeted_case_priority` array may list distinct automatic targeted-case IDs that
must run first in a constrained window. The driver rejects unknown IDs and the manual Rigmark or
duration cases. It moves only the named targeted cases to the front in the given order, retains
the original relative order of every other targeted case, and always keeps initial Rigmark first
and soak/final Rigmark last. The full ordered case-ID list is stored with each run attempt; this
setting does not change case parameters, required repetitions, deadlines or existing receipts.

For an explicitly reduced campaign scope, `targeted_case_selection` selects the distinct
automatic targeted-case IDs to execute. It does not remove the full matrix, overwrite
previous receipts or turn deferred work into a pass. Initial native measurement, the soak,
restoration and final native measurement remain mandatory. Omit the field to retain the
full targeted matrix. Reusing completed cases still requires the matching runtime identity
and evidence protocol; selected cases that already qualify are not repeated.

Set `soak_stop_after_required_seconds` to the boolean `true` to finish mixed load after its
configured duration, at least two hours, and proceed after the existing final proof. Its
clock starts with actual soak telemetry; the original campaign start and absolute deadline
remain unchanged. A cutoff that prevents the full duration cannot qualify as a pass.
Neither option relaxes fault cleanup, health, API idle, cancellation accounting or fresh
four-rank reservation/staging checks. Deferred cases keep the overall matrix incomplete.
A combination may be prioritized only after every one of its prerequisite case IDs; the driver
rejects an order that could run combined faults before their individual evidence exists.

Only after the operator explicitly extends the same exclusive window, an optional
`deadline_extension_hours` adds a finite positive duration of at most 12 hours, with
at most 24 hours total in one driver configuration. Keep `duration_hours` and
`started_at_unix` unchanged. Resuming records the previous and extended absolute
deadlines in `deadline_extension_history`; repeating the same extension does not
add it again, and shrinking an existing deadline is rejected. The two-hour minimum
soak and final 45-minute restoration reserve remain in place. This setting does not
authorize an extension by itself or reset the experiment's elapsed time.

```sh
umask 077
python3 scripts/resilience/campaign.py --config /private/path/campaign.json --plan \
  > /private/path/campaign-plan.json
```

Before loading the overlay, start the campaign clock and run the initial native Rigmark suite on
the protected default. Save its required receipt and write the start time, including that native
run, as the numeric `started_at_unix` value in the private configuration. For example:

```sh
date +%s
# Run native ./rigmark run ... --output <configured rigmark.initial path>
```

Use a fresh comparison ID and all current reference flags. Use the generated overlay for deploy,
start, status and eventual down. Never set it in the rank-0 autostart unit.

```sh
export TP4_ENV=scripts/resilience/.campaign/<unique-id>/delta.env
./scripts/deploy.sh
./scripts/tp4ctl restart
```

After `/health` reaches 200, repeat both functional gates from `docs/operations.md`. On every
rank require an `adapter_ready` event and the protected connector boot signatures. Confirm a
telemetry sample reports `reservation_status=ok`, a complete/fresh cache scan and at least one
live reservation owner. The overlay is temporary campaign instrumentation and is not the
default identity checked by `scripts/check-f0.py`.

## Run and timing

After the overlay passes `/health` and both gates, start the driver. It refuses execution without
the fixed `started_at_unix` and an initial native receipt that is a fresh, nonsymlinked regular
file containing one complete JSON object:

```sh
umask 077
python3 scripts/resilience/campaign.py \
  --config /private/path/campaign.json --execute \
  --ack-exclusive-production-window --ack-service-faults
```

The two acknowledgements are deliberate execution guards. Plan mode, overlay generation and
offline tests do not need them. During execution the driver:

1. samples every rank at 200 ms for targeted cases and at one second for the soak;
2. executes three repetitions of each targeted case until the deadline or a stop condition;
3. stops starting targeted work two minutes before the latest soak start and uses that margin to
   abort, clean up, drain and health-check any in-flight case. The latest soak start is the
   restoration boundary minus the configured duration and a separate 60-second final-proof
   reserve. Mixed load continues until that reserve begins; if targeted work ends earlier, the
   soak starts earlier and runs longer by default. With `soak_stop_after_required_seconds`,
   it finishes after the configured load duration and final proof instead;
4. reserves the final 45 minutes for overlay teardown, one selected operational start, both
   functional gates, the applicable `scripts/check-f0.py` identity check and the final native
   Rigmark suite;
5. records `restoration` in `campaign.json`, then waits until the absolute campaign deadline for
   the operator to create the configured `rigmark.final` receipt on the restored default. That
   receipt is recorded only after it is a complete JSON object; its native result is preserved
   without being reclassified by the resilience driver.

Watch the private state file during the final window. Run native Rigmark only after
`restoration.gates.pass` is true and `restoration.identity.returncode` is zero. The driver marks
the campaign incomplete when the final receipt is absent; it never manufactures or wraps that
measurement.

The driver never loops a restart. An unexpected worker death, management loss, integrity
signal, cache growth beyond 8 GiB or `MemAvailable` below 768 MiB stops further load growth.
It preserves receipts and moves to the single coordinated restoration attempt. The 768 MiB
threshold protects the window; reaching it is not evidence that the service resisted pressure.
Before a fault-driven restart or restoration, the driver streams current `docker inspect` and
`docker logs --timestamps` output for all four ranks into mode-0600 files under the private
receipt directory. Collection is parallel and has one ten-second overall bound; timeouts and
command failures remain explicit in the capture receipt instead of blocking recovery.
At run start, after every cache or cancellation action, and at the targeted cutoff, the driver
uses a separate cleanup deadline to try `mode=off` and artifact restoration on every rank even
if another rank fails. Cleanup never extends the request deadline. A planned cutoff is pending
only when cleanup, health, API idle and a fresh reservation drain all succeed; otherwise it is a
failure. A cancellation drain that first reaches the targeted cutoff stays pending only while
its reissue and health are valid and idle has not failed; the common cutoff proof must then
repeat and pass both idle and the fresh four-rank drain before soak begins.

## Matrix and phase evidence

The generated plan includes exact chat-token contexts of 8k, 32k, 64k, 128k, 180k and
262,128 tokens, each at concurrency 1, 2, 4, 5 and 6, with cold unique prefixes and seeded replay.
Each completed response must report the exact prompt count, 16 completion tokens and a
length finish. Concurrent clients may queue or be preempted by the engine; passing a
case does not establish that all their complete contexts were resident simultaneously.

The separate `context-245760-c5-long-decode-cold` case sends five distinct cold prompts
of 245,760 tokens with `max_tokens=16384`, `min_tokens=16384` and `ignore_eos=true`.
Each total remains within 262,144 tokens. Requiring the full output budget keeps
completed prefills live longer than the 16-token context checks. The case records
prompt hashes and canonical logical-payload hashes, forcing parameters and request start/end times; all five
client intervals must overlap, and exact-length responses plus fresh four-rank drain
remain required. These are synthetic pressure requests. Client overlap does not prove
five full contexts were simultaneously resident: use engine running/waiting counts,
KV occupancy and native preemption counters to describe the actual behavior.
Its client request group is limited to 3,600 seconds or the earlier campaign phase
deadline. Expiry of the case limit fails the case and retains all five outcomes;
an earlier planned phase cutoff remains pending. The prior campaign deadline is
restored before classifying the result or collecting the drain proof.
The driver verifies every response's `usage.prompt_tokens` against the requested size. A replay
passes only after a committed four-rank seed publication and a fresh same-digest restore event
on every rank; a GPU-prefix hit or recompute without those events stays pending. Standard context
cases retain their 16-token output limit. Queue waves start at 8, 16, 32 and 64 clients and double to
the configured bound; each mixes short requests with long contexts. The next wave is not
started after a safety stop or the soak cutoff. Its evidence retains a result or explicit
client-side error, request ID and start/finish timestamps for every requested wave client,
including attempts that could not be submitted, plus the six occupying streams and their
termination state. Requested, submitted and recorded counts stay separate. An occupying stream
that failed before intentional cancellation also fails the case. A local file-descriptor or
thread limit therefore remains a client error in the receipt rather than being mislabeled as a
service bound. Work that entered the executor even when submission raised keeps its actual
response; acknowledged, unknown and never-dispatched attempts remain distinct. A proven local
client-capacity limit is pending and nonfatal only when health, API idle and the common fresh
post-wave drain all pass. The driver then leaves larger queue waves pending without sending them;
it records the smaller client-side boundary and does not call it a service admission limit. A
queue wave may combine explicit local-capacity errors with explicit phase-deadline aborts, but it
remains pending only when every completed response is valid. HTTP and protocol errors still fail.
A bounded-admission campaign adds a separate three-repetition `admission-overflow` case. It keeps
six 32k occupying streams active, submits 150 reproducible 512-token requests with 16-token
outputs, and requires a freshly observed frontend queue plus at least one explicit queue-full
response. Only HTTP 503 responses with `Retry-After`, OpenAI error type `overloaded`, and code
`admission_queue_full` or `admission_queue_timeout` count as intentional overload. All accepted
responses must remain valid; request timeouts, body errors, unknown 503s and any rejection in a
normal wave of at most 128 clients fail. Requested, submitted, accepted, rejected and recorded
counts are retained. Health, API idle and the common four-rank drain still gate the result.
A context group cut only by the planned phase deadline follows the same completed-response rule.
Both paths require the common cutoff cleanup, health and fresh drain before mixed load begins; the
wall clock alone never converts an unrelated exception into a deadline abort.

API cases cover an over-limit context, invalid parameters, malformed JSON and an interrupted
upload. Slow-client cases read one byte at a time, suspend reads, or abandon the connection.
Cancellation cases cover queued work, capture, publication, restore, decode and four simultaneous
requests. Capture, publication and restore count as synchronized only when the wrapper emits a
fresh `stage_waiting` event for that case while the test-only controller pause is still held;
the socket closes before the controller releases that phase. Decode requires a first stream
byte. A cancellation passes only when the request was started and still active at the recorded
cancellation time, its client thread terminates, and neither `[DONE]` nor a clean finish reason
was observed. A request that completed before cancellation stays pending; an HTTP error, API
error event, malformed stream or other pre-cancellation failure fails the case. The receipt keeps
the complete transport result, headers, error and start/first-byte/cancellation/finish timestamps
for every cancelled request and every occupying request. Seed and immediate reissue responses
must report the exact requested prompt-token count, 16 completion tokens and `finish_reason=length`.
Every case also requires health, API idle and a fresh four-rank zero-reservation/staging drain.
Queue cancellation records the native waiting-request counter, but stays pending because that
global counter cannot identify the particular cancelled request. The current runtime has no exact
prefill edge, so that case remains `pending` rather than converting a timed disconnect into a
pass.

Cache cases create distinct snapshots and shared prefixes, grow manifests, and test missing,
truncated and checksum-damaged objects. The shared-prefix case first publishes an exact 32,000-token
seed, then synthesizes an exact 34,304-token request whose literal prefix is that seed. It requires
a fresh successful four-rank restore of the seed digest followed by a different committed
four-rank publication for the longer request; a same-boundary extension or a restore with a
different digest stays pending. Both responses must meet the same exact 16-token completion
contract. `faultctl` moves or backs up the selected artifact,
hashes it in bounded chunks and restores it after the request. ENOSPC, EIO, delay and
publication interruption are injected only on matching opens or writes inside the campaign
namespace. A successful injected-I/O case also requires a fresh, case-bound `fault_hit` on all
four ranks and the expected stage result: an error for ENOSPC, EIO or interrupted publication,
and a completed restore for delay. Publication evidence comes from the connector's authoritative
finish callback and counts only `committed=true`. Corruption selects an object referenced by the exact snapshot digest produced for
that case, never the first object in the namespace. A successful cache-fault case requires a
valid response through miss/recompute, fresh publication evidence, health 200 and successful
artifact restoration; a partial restore is never accepted as a hit.

Worker cases signal rank 0 and one secondary rank while that rank is stopped at a proven capture
phase. The phase event supplies the connector process PID and `/proc` start time; `faultctl`
maps that namespace identity to exactly one host process before sending the signal. It never
signals container PID 1 by assumption.
TERM and KILL require the interrupted stream to fail; an SSE error, malformed stream or missing
finish reason is failure evidence even when HTTP returned 200 and `[DONE]`. A temporary STOP/CONT
request may finish normally after CONT. All three are followed by one
`tp4ctl restart` with the campaign overlay, then `/health` and both functional gates. The driver
does not repair a rank in isolation. It leaves a worker case pending without sending a signal
when the remaining targeted-phase budget cannot contain the bounded restart, evidence, gates
and drain steps.

The matrix also names cache-EIO-plus-restore-cancellation and queue-plus-slow-client-cancellation
combinations, with their individual cases as prerequisites. The first runs only after three
same-runtime successes for both cache EIO and restore cancellation. Cancellation and shared-prefix
receipts use targeted-case protocol 2; historical rows without that version remain preserved but
do not satisfy current repetition totals or combination prerequisites. It seeds a committed
four-rank snapshot, pauses all four matching restore operations, requests cancellation, releases
one read EIO per rank, and requires fresh rank-local fault hits and matching restore-error
operation IDs. Cleanup must then succeed, the request must be proven interrupted before a
completed SSE event, and reissue, health, API-idle and fresh four-rank drain must pass. Missing
phase evidence remains pending; a concrete request, cleanup or recovery failure fails. The
queue-plus-slow-client composition remains explicitly pending because it has no reviewed
synchronized driver. The final mixed soak combines
ordinary load across three deterministic round-robin conversations, exact request replay,
bounded growth through 512, 8k, 32k and 64k chat tokens, cancellations and measured idle
windows. Each growth request retains its prior role/message prefix and appends the actual
assistant content accepted from the service; the driver never invents an assistant turn.
The immediately following replay for that conversation uses the exact saved request payload,
then the next growth adds one new user turn. Histories are capped at 32 messages and reset only
after the 64k replay. This duration case does not turn the unimplemented queue/slow-client
composition into a pass. If exact token synthesis becomes impossible, the soak records a driver
limitation as pending rather than a service failure. Starting too late to complete the configured
two hours plus the reserved final proof is also pending unless an actual request, safety or
resource failure occurred. A duration pass requires observed ordinary, growth, replay,
cancellation and idle counts; elapsed time alone cannot pass it. The last minute before the
restoration boundary is reserved for the final health, idle and fresh cache-accounting proof and
does not count toward the configured minimum duration. If that boundary truncates only a bounded
post-cancellation proof, the row records `deadline_truncated`, gives no cancellation credit and
defers the resource decision to the mandatory final fresh proof. An unrelated transport, HTTP,
receipt or cleanup error remains a failure or pending defect rather than a deadline abort.

## Telemetry and drain proof

Each JSONL sample records `MemAvailable`, swap, PSI, the container process tree's RSS, threads
and descriptors, TCP connections, vLLM running/waiting requests and existing GPU/cache
counters, cache bytes, staging bytes, manifest count, process identity and earlyoom/OOM
evidence where available. On GB10, RSS, `MemAvailable` and CUDA counters describe overlapping
unified memory and are never added together.

Probe stdout remains buffered binary data and is appended byte-for-byte to each rank's private
JSONL file. The receiver records the monotonic time immediately before each append and after it
completes. If the watchdog declares a stream stale, it appends a private diagnostic containing
those boundaries, last valid source timestamps, receive age, probe process status and the path of
that rank's mode-0600 SSH stderr log. These receipts help distinguish a remote sampling gap from a
local reader or append stall without weakening freshness checks.

The wrapper writes one reservation record per live budget owner with its container PID and
process start ticks. The host probe maps namespace PIDs through `NSpid`; stale, missing or
unverifiable owners produce an unknown value instead of zero. After every automatic case, the
driver obtains a node-local monotonic barrier through `faultctl`, then requires all four ranks
to report a cache scan that began strictly after that rank's barrier. This prevents a recently
emitted sample from reusing an older cached zero. The fresh scan must report:

- `reservation_status` equal to `ok`;
- `scan_complete` and `sample_fresh` true;
- total live `reserved_bytes` equal to zero;
- complete cache accounting with `staging_bytes` equal to zero, including anonymous staging.

Persistent nonzero reservations fail the case. Missing or unverifiable instrumentation leaves
it pending. This proof belongs only to the test overlay; it does not claim that an
uninstrumented production process exposed the same counter.

When bounded admission is selected, queue and idle proof also require a current admission source.
Queue depth is the native vLLM waiting gauge plus the middleware waiting gauge. Idle requires
native running/waiting, middleware active/waiting/inflight and established connections all to be
zero. The admitted-total and rejection counters are cumulative diagnostics and are never treated
as drain gauges. Missing, stale or identity-mismatched admission telemetry never masquerades as
zero.

During the bounded coordinated worker-restart window, a stopped container can make its cache
and cgroup accounting temporarily unavailable. Those transient unknown values do not trip the
monitor until the window ends; known namespace usage above 8 GiB and `MemAvailable` below
768 MiB remain immediate stops. Host and cgroup OOM decisions use probe-start deltas, so an old
host lifetime total is diagnostic only. Unknown host OOM-delta coverage fails closed after the
startup grace period.

## Receipts, failures and correction

The private `campaign.json` contains every planned case, target repetition count, actual run,
status and evidence. It also persists safety stop reasons. `events.jsonl` records transitions;
per-rank telemetry stays alongside it. Every grouped synthetic request has its request ID and
start/finish timestamps even when the transport fails. Guard-driven aborts also identify whether
the safety event or phase deadline caused the interruption.
The duration protocol writes a separate mode-0600 JSONL row for every request attempt and idle
window. Request rows contain the complete submitted JSON body, request/conversation/turn/type
identity, target, HTTP response or transport error, timestamps and guard cause. Cancellation rows
also retain the bounded response bytes received before interruption, mark whether the stream was
complete, and record first-byte, cancellation and bounded thread-termination facts plus a fresh
post-cancel API-idle and four-rank reservation/staging drain. A request that completed before the
cancel call does not satisfy the cancellation category. Idle rows preserve actual start/end times
and health/resource snapshots. The final duration evidence stores the JSONL byte count, row count
and SHA-256, and requires a final health, API-idle and fresh four-rank
zero-reservation/zero-staging proof before restoration. A receipt write, stat or hash error remains
explicit and cannot pass the duration case. The protocol version is attached to each duration
run; an older soak receipt cannot satisfy the current duration case after resume. The active
receipt path and durable row count are saved before load begins and after each complete append.
The local writer retries short writes. An operator interruption during a cancellation requests
socket shutdown, waits a bounded interval for that request and records its available HTTP bytes
and cleanup state before the interruption propagates to coordinated restoration. Guarded HTTP
connections use a small transport-timeout grace so the 50 ms campaign watcher records the actual
safety or phase-deadline cause; unguarded requests retain their configured timeout.
Statuses are deliberately narrow:

- `pass`: the case-specific result, health check and reservation drain all passed;
- `recorded`: an external native Rigmark receipt exists, without a resilience verdict;
- `fail`: a concrete request, recovery, integrity, resource or gate requirement failed;
- `pending`: the case did not run, missed its exact phase edge, exceeded the deadline, or lacked
  enough telemetry to prove its result.
- `pass_with_failures`: three required successes occurred after the last preserved failure; the
  failure remains visible and the global result is `complete_with_failures`.

The global result is `complete` only when every required automatic case passed and both native
Rigmark receipts were recorded. Any pending case makes it `incomplete`; skipped prefill remains
pending because the runtime exposes no exact prefill phase edge. A planned cutoff abort is
pending only after the service is healthy and the post-case reservation drain is proven.

At the first defect, retain the receipt and stop escalation. Diagnose and correct it outside the
driver. The driver makes its one protected-default restoration attempt and exits with the final
Rigmark case pending. After the correction, reload the same reviewed overlay through the existing
four-rank controller and resume with the same configuration and receipt directory, only within
the original absolute deadline or its explicitly authorized extension. The failing case needs three new successful repetitions before
later cases continue; failures do not consume that success count and older evidence remains in
the receipt. On resume, the previous terminal status and restoration evidence move to
`previous_attempts`; current status returns to `running`, so an operator cannot mistake stale
restoration evidence for the active attempt. Each new automatic case run carries the active
runtime configuration identity, and successes from another runtime identity do not satisfy its
repetition count. The native initial Rigmark row carries the explicit
`protected-default/native-rigmark` identity; the final row carries the identity of the successfully
verified restoration target. A
legacy case receipt remains unchanged; its archived attempt is marked as a pre-identity
protected-default run with an unknown overlay hash. A different runtime identity may begin only
after completed default restoration has both passing functional gates and a successful identity
check; a completion timestamp alone is insufficient. The original start remains unchanged
across every resume; a deadline changes only through the explicit extension above.
The driver does not patch code or call a permanent
supervisor.

At handoff, report the executed matrix, failures and reproductions, corrections, measured
limits, pending cases, final default identity and the exact recipe. A stopped or skipped case is
never counted as passed, and a campaign with no observed failure does not prove immunity to
unbounded or untested faults.
