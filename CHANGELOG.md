# Changelog

Changes to the deployment, tools, and public documentation are recorded here.
Versions and releases are created only at the owner's explicit request.

## Unreleased

### Added

- A reproducible requests-versus-memory figure, portable CSV and summary of the
  98m59s mixed-load run: 756 complete responses, 84 intentional cancellations and
  fresh four-rank drains. The operator ended the stable run early; the original
  two-hour gate remains incomplete and the earlier failed attempt is preserved.
  Coverage notes distinguish the overlapping campaign and handoff samplers.
- Final native Rigmark evidence on the bounded production recipe, kept separate
  from the frozen E31 baseline: one suite, 54 complete requests, no recorded errors.
  Initial/final performance differences are descriptive across different loads;
  the slower 8K replay is documented alongside code and concurrency results.
- A sealed partial mixed-load checkpoint preserves per-rank memory samples and a
  load-correlated CSV, including source age, idle periods and final-drain memory. Its
  failed idle check coincided with five non-driver inference requests; the later
  successful drain does not turn the incomplete two-hour test into a pass.
  Subsequent operational recovery is recorded separately from the failed run.
- A separate bounded targeted checkpoint records three repetitions of
  16 queue, maximum-context replay, cancellation, slow-client, API and cache-fault cases on the 14 GiB
  recipe: 144 request attempts, 192 fresh rank drain proofs and complete sampled
  OOM/retry coverage. The selected 23-case driver summary has 69 qualifying repetitions;
  duration, final production checks and deferred fault coverage remain distinct.
- Resilience campaigns can select a reduced set of targeted cases and finish the
  soak after its required duration. Completed evidence remains reusable only for
  the matching runtime and protocol; deferred cases stay pending. The original
  campaign clock, minimum two-hour load, final proofs and restoration checks remain
  intact, allowing an earlier return to the operational service.
- Resilience context cases now include five concurrent clients through the full
  262,144-token context limit, with exact prompt and output accounting. An explicit
  operator-authorized deadline extension preserves the original campaign start,
  records the old and new deadlines, and retains bounded restoration and soak phases.
  A distinct five-client long-decode case uses 245,760 prompt tokens and forces
  16,384 output tokens within the same context limit, recording logical payload identity
  and overlapping client intervals to exercise longer KV retention. A 3,600-second
  client deadline bounds this case independently of the campaign deadline;
  expiry fails the case while preserving all request outcomes.
  A sealed short-output C5 cold checkpoint records three repetitions, 15 complete
  maximum-context responses, 12 fresh rank drain proofs and 4.25 GiB minimum
  available memory on rank 0. Five admitted clients reached at most two running
  engine requests; this result does not establish five resident full contexts.
  A separate long-output checkpoint completes all 15 responses across three waves,
  retains 3.71 GiB minimum sampled rank-0 headroom and records 22 preemptions with
  fresh four-rank drains. Retrospective output rates explicitly include queue,
  prefill and scheduler pauses; they are not native performance measurements.
  Coverage notes distinguish the captured telemetry tail, cache drain proofs,
  independently sampled API gauges and the test namespace's cache policy.
- An optional bounded-admission experiment uses vLLM's native ASGI middleware hook
  to queue requests before prompt parsing and tokenization. It limits active and
  waiting requests, bounds body size and stalled reads/writes, and reports explicit
  overload responses. Health and metrics remain available independently. Queue
  counters and release checks distinguish frontend waiting from the engine queue;
  the measured admission settings are now part of the operational default described below.
  A separate production candidate excludes fault injection and retains the normal
  cache namespace. Explicit `check-f0 --identity` verifies its four-rank payload,
  admission and KV settings; a complete rollback restores the protected 16 GiB recipe.
  A sealed live checkpoint records three overload repetitions with 128 accepted
  wave requests and 22 explicit full-queue rejections each, complete response
  accounting and fresh four-rank drain. This establishes the tested admission
  behavior, not many distinct resident contexts or a promotion verdict.
- An isolated scheduler experiment limits each output to 6,912 target tokens while
  retaining the configured 8,192-token capacity and existing draft-slot accounting.
  Remaining prefill work stays queued. The cap is now part of the composed operational
  recipe; it is not an independent guarantee of memory safety or performance.
- An opt-in prefill allocator-trim experiment, isolated from the validated worker
  payload. It measures unused CUDA memory released before eligible eager prefills;
  its deployed source hashes are checked before launch. The measured worker is now
  selected by the operational default together with the other memory controls.
- A separate protected-SparkCache resilience report with the initial native Rigmark
  receipt hash, actual request counts and numeric extracts. It records successful
  8k/C1 cold/replay repeats and the first 8k/C2 memory-guard stop with its evidence
  limitations, plus a default-only 4k/C2 reproduction that isolates CUDA allocator
  reservation and concurrent working-memory pressure. Stress cases and final measurements remain separate from
  frozen E31 results.
  A separate allocator-trim diagnostic records actual reclaimed memory and its
  repeated guard failure, without counting it as a successful correction.
  The subsequent scheduler-cap candidate records three successful two-client
  repetitions at both 4k and 8k, with exact token counts, drained reservations and
  sampled memory minima; 8k prefill is partly serialized through the queue.
  These bounded results alone do not establish a general memory bound.
  A subsequent context checkpoint records 34 complete three-repeat cases through
  128k/C6 and 180k/C1, plus a partial 180k/C2 case, with separate replay-seed
  counts and four-rank restore and resource-drain evidence. Unrecorded interrupted
  work is excluded. A separate checkpoint records three repetitions of four API and
  nine cache cases, with 84 cache responses, explicit fault evidence and measured
  sampling cadence. Shared-prefix and legacy cancellation observations retain their
  evidence limits; sustained load and remaining fault coverage stay pending.
  A separate 14 GiB KV candidate completes three cold and three replay requests at
  262,128 prompt tokens plus 16 output tokens, with nine responses including replay
  seeds, four-rank release proofs and a measured 2.72 GiB minimum available memory
  on rank 0. This is maximum-context evidence, not a queue, soak or promotion verdict.
  A disjoint checkpoint records three successful 8- and 16-client queue waves and
  three multiple-cancellation repetitions. The 32-client wave remains failed after
  a controller telemetry timeout, with uninterrupted node samples and no observed
  OOM; the delivery-delay cause remains unresolved.
- A bounded production resilience campaign under `scripts/resilience/`, documented in
  `docs/resilience.md`: reproducible context and queue cases, cancellation and slow-client
  probes, isolated cache fault injection, worker faults with coordinated recovery, private
  receipts and a fixed restoration deadline. Test overlays keep cache mutations in a
  marked namespace capped at 8 GiB per node, with native eviction of synthetic cache
  entries at 3 GiB toward a 2 GiB low watermark. Reservation telemetry distinguishes verified
  zero from missing evidence. Native Rigmark remains the separate performance interface;
  incomplete or unsupported cases remain explicitly pending. Namespace initialization uses
  `sudo` to traverse Docker-owned cache directories without changing their permissions.
  Failed fault cleanup stops the campaign even when a case is otherwise pending;
  soak accounting distinguishes stalled requests from bounded deadline cancellations
  and requires cancelled client threads to terminate.
  Generated runtime mounts preserve the target node's home directory when an overlay
  is inspected from a workstation with a different home path.
  Explicit runtime variants verify the loaded worker and retain per-attempt recipe
  hashes; resuming after a correction preserves the original deadline and requires
  new passing repetitions under the active recipe.
  The duration phase retains actual conversation history and exact replays, records
  private request and cancellation receipts, and requires observed load, idle periods
  and fresh four-rank resource drain. Empty visible reasoning-only responses remain
  valid. A separate final-proof reserve preserves the full two-hour load requirement.
  Queue receipts retain client failures, and an optional targeted-case order allows
  fault coverage before expensive remaining contexts without moving phase deadlines.
  Planned deadline cancellations retain their cause; local client-capacity limits
  remain pending only after service and resource checks, with actual dispatched
  responses preserved even when executor submission fails.
  A combined cache-read EIO and restore cancellation uses existing four-rank phase
  events and requires successful prerequisites, cleanup and recovery evidence;
  interrupted cleanup and recovery preserve partial receipts before propagating.
  Cancellation and shared-prefix cases now use a separate evidence protocol:
  completed streams cannot count as cancellations, individual client outcomes and
  validated reissued requests are retained, and historical results do not satisfy new
  repetitions. The shared-prefix test crosses a complete cache block and requires
  both a seed restore and a distinct committed snapshot on all four ranks.
  An optional KV-pool experiment records the requested byte budget in its overlay
  identity and verifies the loaded value on every rank while retaining the
  262,144-token context limit. A smaller pool trades concurrent resident context
  capacity for memory headroom; every variant retains its separate runtime identity.
  Planned targeted-phase cutoffs keep interrupted cancellation cases pending so
  the soak can proceed; guard aborts cannot qualify as intentional cancellations.
  KV candidate validation rejects conflicting alias spellings as well as duplicate
  byte-budget options.
  Private receiver timestamps and SSH diagnostics help locate telemetry interruptions
  without relaxing safety thresholds. An optional pinned production return is selected
  only after a successful two-hour soak; early stops retain the protected return.
  Final native measurement receipts identify the verified restoration recipe separately
  from the initial measurement.
- SparkCache disk-transfer protection with 8 MiB payload pieces, a shared
  1 GiB transient-work budget per rank and a 1 GiB admission headroom floor. Capture
  stages to an anonymous disk file; publication and restore stream bounded pieces
  without a complete snapshot in RAM. Under pressure stores are skipped
  and restores recompute; queued and cancelled work stays charged until its owner
  drains. It verifies checksums before reporting successful restore, publishes independent
  snapshots atomically, cleans up staging files and uses a separate cache namespace.
  The operational default selects these protections separately from the unchanged
  E31 performance reference. A versioned operational identity and complete E31
  rollback distinguish the protected recipe from the historical unrestricted cache
  path. Identity checks compare every running container's cache configuration with
  the pinned JSON and require the protection boot signatures on all four ranks.
  Live migration requires four-rank launcher parity and identity verification;
  GPU resilience validation requires a coordinated window. See
  `scripts/node/experiments/e03/sparkcache-ram-budget/`.
- E31 indexer overlay (`scripts/node/experiments/e03/e31-indexer/`), now promoted (see Changed).
  A `TP4_ENV` overlay on the E29 default swaps only the `pooled_indexer.py` and
  `ops/glm_kpool.py` mounts. The two changes are selected at runtime by in-container flag
  files, so they can be compared on one load without a restart:
  - The indexer head gate can run as a BF16 tensor-core GEMM with FP32 output instead of an
    FP32 GEMM, after the Apache-2.0 RiNGSiDE `GLM53_INDEXER_GATE_TC` patch. It is off by
    default in the overlay.
  - A fix candidate for C4 pools corrupted by speculative decoding, on by default in the
    overlay. The per-request tail that completes four-token pools had four slots, so the
    rows of a DFlash verify step, including rejected drafts, overwrote committed members of
    the open pool. The tail becomes a ring of `4 * cdiv(4 + K, 4)` slots, 12 for seven
    drafts.
  - Leaf tests for both changes run on one GPU in a maintenance window.
  - `scripts/tests/test-e31-kpool-tail-ring.py` proves the ring offline against the
    kernels' index arithmetic. The legacy tail corrupts pools, the ring never does.
  - `scripts/tests/test-e31-indexer-config.py` checks the patches, switches, launcher parity
    and refusals.
  - The launcher verifies the candidate's `SHA256SUMS` when its files are mounted.
  - The README gives the same-load A/B procedure and the SparkCache residual risk. The
    production override files under `scripts/node/overrides/` and the E29 record are
    unchanged.
- Fidelity campaign tooling (`docs/fidelity/REPORT.md`): it measures how far the E29 recipe's
  next-token distributions and task outcomes deviate from the vendor FP8 model served
  without the local precision and runtime changes.
  - `scripts/fidelity/make_overlays.py` generates complete `TP4_ENV` overlays under
    `scripts/node/experiments/fidelity/` from the frozen references and the E29 default.
    Measurement overlays change only the declared measurement keys: 6 GiB KV, one
    sequence, a 139,264-token window, `--max-logprobs 100` and SparkCache store/restore
    off. The reference keeps the FP8 KV of its frozen recipe, because B12X cannot use BF16
    KV. Serving overlays keep their base recipe, with at most a smaller KV pool. Each
    overlay has its own SparkCache namespace. `--check` detects stale files and `--diff`
    prints any overlay's delta against its base.
  - `scripts/deploy.sh` also deploys `scripts/node/experiments/fidelity/` (SparkCache
    configurations and launchers) to `~/tp4/experiments/fidelity/`. These files stay inert
    unless a fidelity overlay selects them.
  - `data/fidelity/` is ignored: corpus text, token IDs, raw logprobs, audits and site
    inventories stay local. `scripts/check.sh` skips it.
  - Measurement harness in `scripts/fidelity/`:
    - Collectors for teacher-forced prompt logprobs and greedy generations through the vLLM
      completions API. They use a fresh cache salt per attempt, retry with backoff, resume,
      and stop on an abort file.
    - `mem_sampler.py` samples MemAvailable per rank and raises that abort file below a
      rank-0 floor.
    - `analyze.py` reports coarse top-K KL, which is a lower bound on the true KL, together
      with top-1 agreement and ΔNLL. Breakdowns cover category, source, position bucket
      and prefill path, with window-bootstrap CIs, noise-floor excess, MDE and
      K-sensitivity.
    - `smoke.py`, `gates.py` and `boot_record.py` check logprob support, run the two
      functional gates and record each boot's identity.
  - `scripts/fidelity/corpus/` builds the private corpus: redacted agent-session windows,
    Italian and model-native windows, and a 150-prompt decode set. It renders with the
    runtime chat template and the pinned tokenizer, verifies token files against their
    manifests, and publishes only the numeric `docs/fidelity/corpus-summary.json`.
  - `scripts/fidelity/tasks/` is the endpoint layer, for local vLLM arms and a cost-capped
    z.ai arm, over the knapcio qeval, hardset and tasktime sets vendored unmodified in
    `third_party/knapcio-bench/` (MIT). Its statistics are exact McNemar, Wilson intervals
    and a paired bootstrap.
  - `scripts/fidelity/voxel/` fetches the two public voxel-pagoda prompts verbatim and runs
    them against each arm. It renders the outputs headlessly, checks the Pagoda Bench
    constraints and builds an anonymised gallery.
  - `scripts/check.sh` runs the new stdlib tests (metrics, collectors, redaction, tasks,
    campaign analysis and report) and `make_overlays.py --check`.
  - `scripts/node/experiments/fidelity/launch-nvfp4-tp4.sh` runs Alex Ellis's published
    NVFP4 engine layer on this repository's fabric for the comparison arm.
  - `scripts/fidelity/analyze_campaign.py` and `campaign.config.json` report every
    comparison separately for dense (≤ 2,048 conditioning tokens), sparse and overall
    regimes. They add covered probability mass and top-K overlap, a bootstrap over source
    sessions, excess over the repeatability floor with one-sided bounds and
    `within_margin`/`exceeds_margin`/`unresolved` verdicts. Repeated executions give the
    averaged-distribution estimate for the nondeterministic sparse regime; the analysis also
    covers MDE, K sensitivity and ladder attribution. KL is computed in log space, with a
    compensated complement for the rest bucket, so near-identical distributions do not
    round to spurious values.
  - `scripts/fidelity/plots.py`, `build_report.py` and `make_all.sh` regenerate all metrics,
    the task statistics and probe summaries, fifteen PNG/SVG figures,
    `docs/fidelity/REPORT.md` and a self-contained `report.html` in one command. The report
    opens with the verdict and the perplexity-change quality figures, and a built-in leak
    check keeps site values out of the public outputs.
  - `corruption_probe.py` counts invalid UTF-8, U+FFFD replacement characters, repetition
    locks and tool-call parse failures on Italian and tool-call prompts (motivated by vLLM
    issue 54150 for ModelOpt NVFP4 checkpoints). `run_serving_chain.sh` runs the
    serving-mode boots for tasks, voxel runs and probes, and ends by restoring the plain
    default with both gates and `check-f0.py`.
  - `determinism_probe.py`, `generate_native.py`, `make_subsets.py`, `run_measure_boot.sh`
    and `run_arm_chain.sh` drive the measurement boots: a repeatability probe, model-native
    corpus windows, seeded subsets and resumable measurement steps under the rank-memory
    abort.
  - First campaign results in `docs/fidelity/`: the report, its verdict, aggregated metrics,
    boot identities and figures. E29 differs from the vendor FP8 model in the dense regime
    well beyond the pre-registered KL margin, but shows no detectable perplexity change
    (all positions +0.14% [−0.32, +0.56]). The sparse regime remains unresolved because the
    engine is not deterministic beyond 2,048 tokens. The NVFP4 comparison arm deviates
    further and is 6.9% [4.5, 9.2] worse in perplexity than E29 overall. Voxel runs and the
    cloud arm are deferred.
  - The report opens with a plain-language summary (`docs/fidelity/eli5.md`, "In plain
    words"). It shows the corruption-probe results and the ladder negative controls, and
    says which arm was scored by prompt scoring, tasks or the probe. Descoped work
    (decode-set generations, sampled task runs, the voxel showcase, the cloud arm) is
    named as descoped instead of pending. The task limitation gives both MDE
    approximations: 6.5–7.5 pp from the observed discordance, about 32 pp if every item
    were discordant. `tasks/stats.py` now computes the paired pass-rate MDE; it
    previously reported the sign-test shift (16.2 pp) under that name.
  - After an independent review, the verdict, the plain-language summary and the README
    state "no detectable quality loss on this corpus" rather than equivalence. They label
    ladder steps against the previous rung, and describe qeval as 75 mixed tasks. They
    also report NVFP4's short-context perplexity advantage and name the Italian corpus
    as synthetic.
  - `scripts/fidelity/export_public.py` writes the campaign's portable data to
    `docs/historical_benchmarks/experiments/2026-09-27-fidelity/`: identity, coverage,
    headline results and hashes of the private originals, run metadata, qeval per-item
    results, corruption-probe rows, functional gates and memory samples. Per-window
    metrics are included only for windows generated from public prompts. Site values
    are removed, and a built-in leak check guards the output. Raw logprobs, token IDs,
    corpus text, and per-window metrics and records from private sessions stay in the
    ignored `data/fidelity/`. Private-session windows appear publicly only in aggregates
    and in `corpus-hashes.json`, as one row each: category, token count and SHA-256.
  - `run_serving_chain.sh` validates every overlay and step before the first boot. It
    exits non-zero after a failed step (both voxel processes are checked), a memory
    abort at any point or a failed final `check-f0.py`, restoring the plain default
    first. `tasks/run_tasks.py` exits non-zero on collection errors or cost-capped
    skips. `run_arm_chain.sh`
    accepts an empty `CURRENT_TP4_ENV` for the default recipe. The fidelity README warns
    that qeval grading runs model-generated Python, so it belongs in a disposable
    environment.
  - The campaign plan is not published. The report lists the protocol amendments that the
    text, figure notes and overlay headers cite, and the plan stays in the private campaign
    records.
  - The README has a "Measured quality" section, next to "Measured performance", with
    "vs vendor FP8" and "vs NVFP4" subsections. It explains in plain language how to read
    each number (same first choice, distance, perplexity change, tasks, repeat runs,
    broken characters), gives both tables a "What it means" column and refers to the
    current recipe without internal names. It shows
    two plain-language charts (`16-quality-by-context-length`, `17-quality-by-kind-of-text`,
    which the report also shows under "In plain words") and links the report. The
    technical quality figures use plain labels (Vendor FP8,
    Current recipe, Earlier recipe, NVFP4); their notes no longer claim equal prediction,
    and figure 1 leaves the recipe ladder to its own figure. The benchmark index lists
    the campaign.

- Archived the discarded E27d experiment: a per-step cap of 2,304 prefill tokens while other
  requests decode, on the E29 default. The archive holds its report and portable numeric
  extract, and the benchmark index has a new row for it.
  - Protocol 2: background decode during another request's 8K–32K cold prefill rose from
    6–7 to 11–20 tok/s and the longest gap fell from about 3.0 to 1.3 s. The arriving
    request's first token came 33–60% later.
  - Six suites on two loads stayed within the variation seen between loads; the cap never
    engaged in the standard suites.
  - The owner judged the benefit not significant. Its overlay and code are not part of the
    tree, and the default recipe is unchanged.
- The checkout now includes the SparkCache and SIRCL payload the recipe mounts, so an
  installation no longer depends on external repositories or maintainer-supplied files.
  - `third_party/sparkcache/` holds the current and rollback connectors and the encoder,
    the upstream Apache-2.0 license, and one complete patch per project change against
    SparkCache `66057174`: pending-publication wait, two memory corrections and replay
    views.
  - `third_party/sparkring-sircl/` holds the 17 bundle files, the serving entrypoint and
    environment, and SparkRing's LICENSE, NOTICE and third-party notices. The native library
    was built from SparkRing `b358a818`; its manifest pins all 113 source files, which match
    that commit. The entrypoint's only change is the added GID check.
  - `scripts/deploy.sh` copies both to `~/tp4/sparkcache/` and `~/tp4/sircl/` with
    `scripts/sircl_gid_check.py`. The files are byte-identical to the pins, so a deploy
    leaves a running installation unchanged.
- Added `scripts/sircl-site-files.sh`. It generates the private SIRCL per-rank peer, device
  and GID files, their runtime manifest and `SHA256SUMS.site` from `cluster.env`, following
  the ring plan of `render-netplan.sh`. It refuses to overwrite existing site files without
  `--force`. The deploy copies them when the checkout holds them.
- Added `scripts/tests/test-third-party-payload.py` to the offline check. It verifies every
  included file against its pin, reverses each patch back to the upstream bytes, checks the
  internal hashes of the SIRCL bundle and the licenses, and tests the site-file generator.
- Added `docs/third-party.md`. It maps every pinned SparkCache and SIRCL file to its source
  and marks it upstream, modified by this project or this project's own, and lists the
  project's changes.
- Published the September 25 E29 frozen baseline and its promotion record. E29 is the E28b
  recipe plus the end-drain overlay in its load B configuration.
  - The default `EXTRA_DOCKER_ENV` mounts `experiments/e03/end-drain/scheduler.py` instead of
    the E27c scheduler and `experiments/e03/end-drain/core.py` over the image's engine core,
    and sets `VLLM_E29_END_DRAIN=1`, `VLLM_E29_IDLE_COALESCE_MS=4` and `VLLM_E29_TRACE=0`.
    The dry-run default is argument-for-argument identical to the measured load on all four
    ranks.
  - Against E28b: C1/C2/C4 per-stream TTFT −13.2% / −14.7% / −9.8%, code decode +3.2%, prose
    +2.5%, no metric worse beyond noise. Three native suites, 162 requests, zero errors,
    45/45 output gates.
  - `scripts/check-f0.py` selects the E29 identity by default and checks both mounted files,
    the three flags and both rank-0 boot lines. Older records now also refuse the E29 flags.
  - The launcher verifies `experiments/e03/end-drain/SHA256SUMS` whenever either file is
    mounted.
- Added `scripts/node/reference/baseline-20260925-e28b.env`, the complete E28b recipe, as the
  one-step rollback from E29.
- Added the E29 end-drain candidate overlay under `scripts/node/experiments/e03/end-drain/`,
  applied on the E28b default. With asynchronous scheduling and seven drafts, a request
  limited by `max_tokens` usually finishes inside a step that could produce several tokens,
  so one more speculative step for it is already queued and delays the next request's first
  token by about 50–100 ms. The overlay changes only request boundaries:
  - `VLLM_E29_END_DRAIN=1` (scheduler) does not dispatch another step for a request whose
    step in flight may reach `max_tokens`; the request resumes if tokens are still missing.
  - `VLLM_E29_IDLE_COALESCE_MS` (engine core, 0–5 ms) collects requests that arrive together
    at an idle engine before the first schedule, so they are prefilled in one step.
  - `VLLM_E29_TRACE=1` logs arrival, dispatch, hold and resume times for diagnosis.
  Both overrides are their base plus an additions-only patch; with the flags unset they
  behave as E28b. Offline tests cover provenance, flag parsing, four-rank launcher parity,
  overlay refusals, and a model of the async accounting (invariant, placeholders, liveness,
  no step dispatched past a length finish). `delta.env` is the traced diagnosis load;
  `delta-b.env` sets the 4 ms window measured there.
- Published the E29 experiment record and report: three native suites (162 requests, zero
  errors) against the E28b medians show C1 per-stream TTFT −13.2%, C2 −14.7%, C4 −9.8%,
  code decode +3.2%, cached replay +3.6% / +2.7% / −1.0% at 8K / 32K / 64K. The rest of the
  replay cost is a wait for the cache connector to publish the preceding request. Three
  owner-requested prefill re-runs are recorded separately; over six executions replay is
  +2.2% / +2.0% / +1.1% against E28b. The owner promoted it.
- Published the September 25 E28b frozen baseline and its promotion record. E28b is the E27c
  recipe with seven draft tokens for a single request and a 16 GiB KV pool per rank.
  - The DFlash2 drafter is trained with blocks of 8, so one request now drafts seven tokens
    (`SPEC_TOKENS=7`, table `[[1,1,7],[2,6,3]]`, `VLLM_ADAPTIVE_K_HI=7`); batches of 2–6
    keep three. `--compilation-config={"max_cudagraph_capture_size":72}` keeps the E27c CUDA
    graph set.
  - Seven tokens hold about 5% fewer KV tokens per GiB, so the pool grows from 15 to 16 GiB:
    1,365,066 tokens, 5.21 full 262,144-token contexts. Rank 0 kept at least 2.1 GiB
    available during the three suites.
  - Against E27c: code decode +8.7%, C2 +1.4%, C4 −1.2%, C1 −0.9%, C1 per-stream TTFT
    +15.9%, cached replay −8.1% at 8K and −10.2% at 32K. The two C4 medians use suites 2–3:
    the owner excluded suite 1's C4 block, whose rounds all started staggered; the values
    stay in the record. Three native suites, 162 requests,
    zero errors, 45/45 output gates; the default was deployed onto the measured load
    without a restart and the live identity check passed.
- Published the E28 and E28b experiment records: suite extracts, the drafter acceptance
  probes (E27c and E28, with and without thinking), per-second host-memory samples, owner
  decisions, reports and index rows. E28, seven draft tokens with the 15 GiB pool, is
  recorded as superseded.
- Added the E28 and E28b candidate overlays under `scripts/node/experiments/e03/draft-depth-7/`
  and `scripts/node/experiments/e03/kv-16gib/`, and an offline test covering four-rank
  launcher parity of both with the E27c recipe and the E28b default, and overlay refusals.
- Added `scripts/node/reference/baseline-20260925-e27c.env`, the complete E27c recipe, as the
  one-step rollback from E28b.
- Published the September 25 E27c frozen baseline and its promotion record. E27c keeps
  E27's `--prefill-schedule-interval 8` and mounts the serving image's scheduler with one
  reviewed patch from `scripts/node/experiments/e03/queued-cadence/`.
  - Short prefills (fewer than 2,048 remaining tokens, up to 2,048 per step) are admitted
    at once, and a deferred long request no longer blocks shorter ones behind it.
  - The cadence stays on while requests are queued, instead of switching itself off.
  - Against E27: C4 per-stream TTFT -27.1%, C2 +2.4%, C4 +1.2%, C1 -1.8%, cold prefill
    -1.6% to -3.3%. With four 32K prompts arriving together, the first stream decodes at
    about 12 tok/s instead of 6; the last starts about 7% later.
  - Three native suites, 162 requests, zero errors, 44/45 output gates (one code answer
    reached the output budget); the default was deployed onto the measured load without a
    restart and the live identity check passed.
- Published the E27b and E27c experiment records: per-suite extracts, protocol-2
  interference extracts (E27, E27b, E27c on the same client), four-simultaneous-32K
  diagnostics, owner decisions, reports and index rows. E27b, the short-prefill part alone
  (`scripts/node/experiments/e03/long-prefill-cadence/`), is recorded as superseded: the
  owner accepted it as a candidate and it was promoted inside E27c.
- Added the E27b and E27c candidate directories with their overlays, manifests and
  `SHA256SUMS`, which the launcher verifies whenever a scheduler is mounted, and offline
  tests covering vendor provenance, the exact patch scope, the 2,047/2,048/2,049-token
  boundaries, the per-step limit, flag parsing, four-rank launcher parity and overlay
  refusals.
- Added `scripts/node/reference/baseline-20260924-e27.env`, the complete E27 recipe, as the
  one-step rollback from E27c.
- Published the September 24 E27 frozen baseline, its promotion record and the E27
  experiment records.
  - Frozen baseline: three native suites, 162 requests.
  - Experiment records: per-suite extracts; two same-day E22b control suites over the same
    LAN client path; the Rigmark interference extract; the long-context diagnostic; and
    the owner decision.
  - Reports, the benchmark index, README figures comparing E27 with E22b, and the updated
    recipe, operations, installation and node guides.
- Added `scripts/node/reference/baseline-20260924-e22b.env`, the complete E22b recipe, as the
  one-step rollback from E27.
- Archived the discarded E23 experiment. It gave the DFlash2 drafter an INT8 copy of the
  shared `lm_head` and a hybrid INT8/BF16 `fc` projection. The archive holds its report and
  portable numeric extract, and the benchmark index has a new row for it.
  - Three suites against E22b showed small gains on C1 (+3.50%) and code decode (+0.97%).
  - C2 and C4 lost about 2%, and 32K cold prefill lost 2.47%.
  - The owner closed the experiment. Its overlay and code are not part of the tree, and the
    default recipe is unchanged.
- Added the E22b drafter component under `scripts/node/experiments/e03/drafter-w8a16/`: it
  converts 30 BF16 linears of the DFlash2 drafter to the accepted E20/E21 INT8 group-128
  Marlin format and keeps the drafter's context K/V projection in BF16. Its drafter
  override is the serving image's file byte for byte plus one appended, flag-gated load
  hook. The launcher verifies the directory's `SHA256SUMS` whenever its files are
  mounted, and an offline test covers pins and vendor provenance, the hook, family
  selection and refusals, and four-rank launcher parity.
- Added the September 23 E22b frozen reference: three complete native Rigmark suites /
  162 requests on one retained load, with code decode +3.03%, C1 +2.23% and prose +2.97%
  against E21, C4 unchanged, zero errors and 45/45 native gates. Added its report, the E22b
  experiment report and extracts, the matched first-token and prefill probes against E21,
  the owner decision, the promotion record and E22b-versus-E21 figures. The first candidate,
  E22, which also converted the context K/V projection, is archived as a superseded
  benchmark record only; its overlay and cache configuration are not kept.
- Added `scripts/node/reference/baseline-20260923-e21.env`, the complete E21 recipe, as the
  one-step rollback from the E22b default.
- Added the prepared E21 candidate, which extends the accepted E20 INT8 group-128 Marlin
  mechanism to the KDA output projection and the MLA output, fused QKV-A and Q-B
  projections behind a flag that is off by default. Its guarded overlay applies only on
  the default E03 recipe at six sequences, selects a separate SparkCache namespace, and
  refuses any other base. An offline test covers pins, the gated hook, family selection
  and shape validation, the prefill dispatch threshold, four-rank launcher parity and
  overlay refusals. The candidate was later measured and promoted; see Changed.
- Added the September 23 E21 frozen reference: three complete native Rigmark suites /
  162 requests on one retained candidate load, with C4 +9.88%, C2 +4.68% and prose
  +4.02% against E03, zero errors and a 43/45 native gate count from two code requests
  that reached the output budget. Added its report, experiment extract, owner decision,
  promotion record and E21-versus-E03 comparison figures.
- Added `scripts/node/reference/baseline-20260919-e03.env`, the complete E03 recipe, as
  the one-step rollback from the E21 default.
- Added guarded current-E03 max-five-sequence functionality and max-four fallback
  overlays, with full-cluster procedures that retain the 15 GiB KV pool, context,
  replay connector and all unrelated accepted runtime settings. Recorded the live C5
  identity, post-boot gates and three-request decode smoke as functionality-only
  evidence, leaving the frozen E03 performance reference unchanged.
- Archived the three-suite native Rigmark record for the restored original C4 recipe
  without the withdrawn admission cap. Four suites / 216 measured requests are preserved:
  runs 1, 3 and 4 form the accepted three-suite / 162-request aggregate, while run 2 is
  retained diagnostically but excluded after verified competing non-benchmark traffic.
  Primary medians are mixed, so the outcome is unresolved with no promotion, and the
  frozen E03 reference, its bytes and medians remain unchanged.
- Archived the owner-withdrawn native vLLM admission experiment for the pinned R10 C4
  recipe, covering its four-active and 24-waiting cap, HTTP 503 at saturation,
  cancellation recovery and bounded full-context replay. One 54-request native suite
  observed mixed throughput and latency changes that a single run cannot establish as
  repeatable, so the performance outcome is unresolved and no release upgrade is implied.
  The source-derived overrides stay out of the public configuration, default payload and
  offline test set, and the verified coordinated return to the complete original C4
  recipe is recorded.
- Archived the September 20 C4 262,144-token capacity activation as a lifecycle and
  capacity record: a coordinated four-rank transition, two timely post-boot gates, four
  full-context request bodies submitted in a cold and a replay phase with verified per-rank
  commits and loads, and a one-request smoke. No native Rigmark suite ran, so the record
  supports no throughput, latency or prefill claim, its separately pinned connector is not
  tracked here, and the accepted E03 recipe and frozen performance records remain unchanged.
- Recorded the completed E03 default-IaC deployment, two timely functional gates
  and four-rank identity checks. The separate 54-request benchmark is excluded by
  owner instruction and retained only for provenance; the accepted September 19
  performance reference, frozen bytes and medians remain unchanged.
- Added a dated previous-baseline delta column to the README and PNG/SVG comparisons
  of all 16 current versus previous September 19 metrics, with paired values, exact
  percentage changes and the accepted approximately-unchanged labels. The historical
  September 11 delta column, frozen records and original figures are retained.
- Added a frozen accepted E03/replay/draft-budget reference from the final three native
  Rigmark suites (162 requests), with all 16 metrics, receipt hashes, variability and
  memory/error accounting. Earlier candidate series and isolated checks remain archived.
- Added current-only generation/latency and cold/replay PNG/SVG figures, linked from the
  README and the dated report; historical figures and frozen baseline bytes are retained.
- Added the complete previous September 19 rollback and offline four-rank equivalence
  checks between the new defaults and the measured candidate, including mounted hashes,
  feature activation, payload pins and historical identity selection.
- Added the measured Apache-2.0 E03 mHC TP4/DCP1 port and draft-budget scheduler, their
  source manifests and CPU tests, plus hash-checked replay connector preparation from
  operator-supplied payload. Licenses and upstream notices remain attached.
- Added dated benchmark reports and result archives for baselines, reproductions and
  experiments, preserving incomplete and excluded evidence with explicit outcomes.
- Added a local native-report directory and a mandatory absolute `--output` convention
  for every Rigmark run, with consolidated agent instructions for execution and archiving.
- Added a two-panel context-length chart for cold prefill and the available
  eight-token decode probes, with reproducible extracted measurements and explicit
  limits on sustained decode and initial-baseline comparison.
- Added reproducible comparison charts showing all 16 metrics
  from September 11 versus the two accepted September 19 runs plus its separate IaC
  reproduction, with that run identified and the frozen baseline unchanged.
- Added the September 19 performance reference and a dated historical comparison with
  September 11 and September 18. The new reference reports two valid native Rigmark
  runs (108 requests), performance tradeoffs and reduced KV capacity; historical records
  remain unchanged.
- Recorded a separate live IaC redeployment and 54-request Rigmark comparison, with
  deployment checks, single-run performance tradeoffs and the frozen baseline unchanged.
- Added hybrid INT8 KDA input projections with a shared BF16 prefill scratch buffer and
  the measured GPU allocator probe, preserving the tested runtime source identities.
- Added the digest-pinned SparkRing/SparkCache image, persistent prefix caching and
  SIRCL single-rail prefill transport, with engine overrides for cache allocation,
  pooled indexing and checkpoint mapping.
- Added hash-verified preparation of operator-supplied SparkCache memory corrections.
  Connector, encoder and transport payloads are pinned without redistribution.
- Added a complete September 18 rollback overlay with the original model, cache config
  and connector pin. Private archive capture and restore tooling preserves the earlier
  September 11 runtime independently of the current defaults.
- Added bounded read-only baseline identity checks, validated automatic RoCEv2 GID
  selection and SIRCL preflight for the configured fabric.
- Added offline coverage for syntax, public links, manifests, templates, lifecycle,
  preflight, cache preparation, hybrid dispatch and the adaptive scheduler.

### Changed

- Made the memory-bounded operational recipe the default while retaining the frozen
  E31 performance reference: 14 GiB KV per rank with the 262,144-token context limit,
  eager-prefill allocator trim, a 6,912-token scheduler-step cap, six active API
  admission slots, 128 queued requests, and protected SparkCache. Complete rollbacks
  retain the protected 16 GiB predecessor and measured E31 recipe. Five full contexts
  require waiting or preemption; test-only cache eviction does not bound production
  disk growth. Default identity checks verify the composed protections on all ranks.
- `scripts/check-f0.py` now selects the protected SparkCache operational identity by
  default and reports `<identity-id> CHECK PASS|FAIL`. Historical checks require an
  explicit `--baseline` and its matching rollback recipe. Each rank's live transfer
  configuration is compared in full with the pinned JSON. The complete E31 rollback
  restores the prior cache behavior, while the frozen performance records stay unchanged.
- The default engine recipe retains the September 28 E31 reference: the E29 recipe with the pooled
  indexer and its C4 kernels mounted from `scripts/node/experiments/e03/e31-indexer/` and
  the two flag-file variables set. The speculative-safe tail ring is on and the head-gate
  switch is off.
  - Rigmark: one suite of the promoted arm (n = 1, 54 requests) measured with upstream
    Rigmark and the reference flags, against a same-load E29-equivalent arm. Code decode
    +0.97%, prose −0.54%, prefill within ±3.5%, concurrency within the noise of three
    short rounds.
  - GPU leaf test: the old tail left 200 wrong pools, the ring none.
  - The applied default is launcher-identical on all four ranks to the measured overlay, so
    the running service needs no restart. The SparkCache namespace is kept; restored pools
    built by the old tail remain a recorded residual risk.
  - New frozen record `docs/historical_benchmarks/baselines/2026-09-28-e31/` with its
    promotion record. Portable extracts, leaf-test summaries, the owner decisions and the
    two excluded series are in
    `docs/historical_benchmarks/experiments/2026-09-28-e31-indexer/`.
  - Reports: `docs/benchmarks/baselines/2026-09-28-e31.md` and
    `docs/benchmarks/experiments/2026-09-28-e31-indexer.md`.
  - `scripts/check-f0.py` retains the E31 engine identity checks: the two indexer hashes, both
    flag-file variables and both `E31_` log lines, and refuses the switch variables
    themselves.
  - `scripts/node/reference/baseline-20260925-e29.env` retains the earlier E29 return;
    the immediate operational rollback is now `baseline-20260928-e31.env`. The
    operations guide documents both and the runtime-only ring switch.
  - The earlier records remain byte-identical. The fidelity overlay generator reads the E29
    recipe from that rollback, so its overlays stay byte-identical to their boot records.
- Benchmark policy (`AGENTS.md`): measure with upstream, unmodified Rigmark and exactly
  the reference flags: every default plus `reasoning_effort` low, no cache salt, a fresh
  comparison ID per suite.
  - Candidates are compared with a same-load reference arm; quick screens may run one
    suite per arm.
  - Frozen records up to E29 used other settings and are not comparable. The README says
    so beside the E31 values and keeps the E29 versus E28b figures as a labelled historical
    comparison.
- The launcher accepts `SPEC_TOKENS=0` (with an empty `SPEC_EXTRA_JSON`) and then omits
  `--speculative-config`; measurement overlays use it when logprobs require speculation
  off. Every value of 1 or more builds the same command as before: all four ranks of the
  E29 default and of the eight reference overlays produce identical dry runs.

- CREDITS, the README prerequisites, installation step 8 and the production recipe now
  link SparkCache and SparkRing SIRCL directly and state their Apache-2.0 licenses. They
  replace the earlier "no license notice" wording, which described only the archived copies.
  Installation step 8 is now "Prepare the SparkCache and SIRCL payload". It generates the
  site files instead of asking the operator to obtain and place the payload. AGENTS, the
  node README, operations, fabric, the template comments and the verification messages
  follow.
- The E29 promotion record now records the live apply: one owner-authorized coordinated
  restart onto the default, `/health` 200, both functional gates, `check-f0` PASS and the
  E29 scheduler, engine core, flags and boot lines on the running ranks.
- The default recipe in `cluster.env.example` is now E28b. `scripts/check-f0.py` validates
  seven draft tokens, the adaptive table and high state, the CUDA graph limit
  (`--compilation-config`) and the 16 GiB KV budget by default, and requires the absence of
  the graph limit for older records. The README, recipe, operations, installation and node
  guides, `AGENTS.md` and the comparison figures now describe E28b against E27c; the E27c
  report is marked superseded.
- The default recipe in `cluster.env.example` is now E27c. `scripts/check-f0.py`
  validates the E27c scheduler hash, both flags and the rank-0 boot lines by default, and
  requires the flags' absence for older records. The README, recipe, operations,
  installation and node guides, `AGENTS.md` and the comparison figures now describe E27c
  against E27.
- The E27 baseline report is marked superseded and explains what later measurements
  showed: the running agents' net delay is about the same as without the cadence (the
  3-5x figures are rates inside the prefill window), short prompts waited for the cadence,
  and several long contexts arriving together were not fixed.
- Corrected the September 24 E27 records after an independent review; no measured value
  changed.
  - The frozen baseline no longer claims that the host memory observers ran. Its new
    `corrections` field lists the edit, and the promotion record carries the new hash.
  - The interference extract records hashed cache salts and the phase's limitations.
  - The results extract names every metric that moved by more than 2%, and marks code
    decode, prose and C4 against the same-day control as unresolved.
  - The reports, README, recipe guide and template no longer claim that several long
    prompts behave exactly as before, that answers or memory are untouched, or that
    decode-only steps always use full CUDA graphs.
- The README performance table now shows two comparisons for every metric: against the
  frozen E22b medians, as before, and against the same-day E22b control over the same
  LAN client path. The frozen comparison alone hid the C4 cost (−3.3%) and overstated
  C1 (+6.6% frozen against +3.3% same-day).
- Made E27 the default IaC recipe: `cluster.env.example` adds the image's native
  `--prefill-schedule-interval 8`.
  - While requests are in decode, prefill runs on one engine step in eight, with
    decode-only steps in between.
  - In Rigmark's interference phase, running requests decode 3–5× faster while another
    request cold-prefills 8–32K tokens: 1.4–2.2 → 6.1–11.5 tok/s. That request's first
    token comes 15–37% later.
  - Against a same-day E22b control, C4 per-stream TTFT is +35% and C4 throughput -3.3%;
    the owner accepted that trade-off.
  - `scripts/check-f0.py` now selects the E27 reference and checks the argument in the
    effective recipe and in every rank's command. Records before E27 require its absence.
  - The candidate overlay is retired from the tree; its hash is in the promotion record.
- Made E22b the default IaC recipe: the drafter override `qwen3_dflash2.py` and
  `e22_drafter_w8a16.py` are mounted from `experiments/e03/drafter-w8a16/`, the default sets
  `VLLM_E22_DRAFTER_W8A16=1` and `VLLM_E22_CONTEXT_KV_W8A16=0`, and SparkCache uses the E22b
  namespace. 30 DFlash2 drafter linears use 8-bit weights; the context K/V projection stays
  BF16. `check-f0.py` validates the new September 23 E22b reference by default, including the
  `E22_DRAFTER_W8A16_READY` boot receipt. The encoded default produces a launcher command
  identical on all four ranks to the measured candidate. The README, figures, recipe,
  operations, installation and node guides describe E22b, with E21 as the previous
  reference and the immediate rollback.
- Made E21 the default IaC recipe: the KDA hook mounts from `experiments/e03/bf16-residue/`,
  the default adds the E21 module and `VLLM_E21_BF16_RESIDUE_W8A16=1`, and SparkCache uses
  the E21 namespace. The launcher verifies that directory's `SHA256SUMS` before starting,
  and `check-f0.py` validates the new reference by default, including the
  `E21_BF16_RESIDUE_W8A16_READY` boot receipt. The encoded default produces a launcher
  command identical on all four ranks to the measured candidate. The README, recipe,
  operations, installation and node guides describe E21, with E03 as the previous
  reference; the C5 overlays are documented as E03-only history.
- Changed the promotion rule in `AGENTS.md`: the accepted suites are the measurement, so a
  promotion proves four-rank launcher parity and live identity instead of reloading and
  rerunning the benchmark. A reproduction run happens only on explicit owner request, and
  `docs/operations.md` describes applying a default without restart while the measured
  candidate is still serving.
- Made the accepted E03 mHC 6,912-row path, replay views and effective draft-budget
  scheduler the default IaC recipe, preserving measured runtime hashes, FP8/DFlash2,
  hybrid KDA, graph budgets, 15 GiB KV and context. Updated installation, boot signatures,
  rollback, node/patch guides and agent instructions. Live deployment/reproduction of
  the newly encoded defaults remains explicitly pending; the serving stack is retained.
- Selected the new three-suite reference in the identity checker, README and plotting
  tool. The README retains historical percentages relative to September 11; the dated
  report includes the previous September 19 base and exact deltas. Small changes of roughly 1–2% are
  labelled approximately unchanged where requested, without claiming statistical parity.
- Moved frozen benchmark JSON and historical PNG/SVG figures into dated archives,
  preserving their bytes and hashes and updating operational references.
- Simplified the README to current-baseline values and figures, retaining percentage
  changes against the linked September 11 reference and preserving history in reports.
- Allowed benchmark Markdown reports alongside the four required operating guides;
  native local receipts remain outside public documentation checks and Git tracking.
- Added percentage changes from the initial to the latest baseline in the README
  performance table, calculated from unrounded medians with TTFT direction clarified.
- Made the September 19 configuration the base IaC recipe: hybrid KDA execution,
  corrected cache allocations, a separate persistent-cache namespace and 15 GiB KV per
  rank. The per-request context limit remains 262,144 tokens. Live IaC reproduction
  results are kept separate from the accepted measurements.
- Selected batch-uniform adaptive verification to keep concurrent decode on full CUDA
  graphs; retained explicit B12X attention, Triton compute backends and the drafter's
  separate FP8 cache type.
- Made the identity checker select September 19 by default while retaining explicit
  historical baseline selection. Checks include KV budget, loaded source hashes,
  hybrid/probe receipts and mapped patched NCCL.
- Updated README, credits, installation, operations, component documentation and agent
  guidance around dated baselines and reproducible target configuration. Preserved
  upstream JSpark3 license and attribution notices.
- Organized public instructions around installation, operations, fabric and production
  components. Agent entry points route fresh checkouts to the required guide and exact
  artifact/configuration sources; local site values and working history remain ignored.
- Documented native Rigmark comparisons, profiling, memory-budget changes and explicit
  request/trace accounting. Preserved measured run counts, performance tradeoffs and
  separation of benchmark speed from generated-answer quality.
- Clarified incremental debugging, reference-model comparisons, evidence required before
  interrupting initialization, and operator-authorized restoration without remeasuring
  frozen baselines.

### Fixed

- `scripts/fetch-fp8-weights.sh` finds the release manifests when run from `~/tp4` on
  rank 0, as the installation guide documents. It used to look only under the checkout
  layout, `scripts/node/model-manifests/`, which `scripts/deploy.sh` does not create
  on a node.
- Fixed a dead `IMAGE` assignment in the configuration template: the historical F0 rollback
  line had lost its comment marker, so it read as an active setting that the real
  digest-pinned `IMAGE` below silently overrode. Editing it had no effect. It is now marked
  as reference only, leaving one active image pin and its content ID.
- Resolve the baseline path for rollback from the sealed archive's own manifest,
  supporting both historical and reorganized source layouts without rewriting archives.
- Aligned chart medians and numeric labels with the frozen README baseline values.
  The README uses simple current-baseline charts; archived comparisons retain the
  separate IaC run outside the frozen summaries.
- Counted deployment drift from file-status rows, excluding diagnostic summaries that
  previously inflated the reported number of mismatched files.
- Released completed SparkCache saver payload references before the next queue wait and
  removed a full-size copy during page encoding, using pinned prepared files.
- Added encoder hash/mount verification alongside the connector and corrected explicit
  `--baseline` selection, including historical restore commands.
- Corrected dense drafter cache allocation, pooled-indexer tail handling and physical
  selection, and legacy checkpoint quantization exclusions.
- Hardened image-content and payload verification, home-relative mounts, topology and
  lifecycle validation. Disabled the unused image healthcheck marker for native SIRCL;
  `/health` remains the readiness signal.
- Added bounded flusher-shutdown retry and phase/exit diagnostics while preserving the
  full-cluster lifecycle and coordinated recovery contract.
- Rejected malformed, out-of-range and ambiguous fabric addresses before netplan
  generation; retained a single local source for generated site configuration.
- Preserved adaptive-policy observation timing and fallback behavior, and verified the
  local chat-template compatibility adapter.
- Excluded private working notes from public checks and rejected public links to them.

### Removed

- Removed the initial-baseline delta column from the README performance table, keeping
  only the dated previous-baseline comparison and aligning the explanatory text and
  agent guidance.
- Removed the unused GPU-clock diagnostic tool and its dedicated documentation. The
  tested GB10 driver did not change effective load clocks; production and rollback
  recipes do not invoke the tool. Host lifecycle tests retain their generic coverage.
- Removed duplicate and superseded descriptions of the current baseline from the
  unreleased notes, keeping dated evidence and rollback instructions in their owners.
