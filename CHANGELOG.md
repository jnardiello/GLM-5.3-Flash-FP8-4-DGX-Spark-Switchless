# Changelog

All repository changes are recorded here incrementally. Entries describe concrete
effects; release sections are created only when the owner explicitly authorizes a
release.

## Unreleased

### Added

- Froze the qualified E09 batch-uniform state as measurement baseline **F1** in
  [`docs/baseline-f1.json`](docs/baseline-f1.json) by explicit owner request: the three
  receipts' hashes, per-run values and medians for all sixteen F0 metrics, the true
  recipe identity (offline-imported image ID without registry digest, overlay, launcher,
  connector, adaptive-k `batch-uniform` parameters, SIRCL transport), functional counts
  (162/162 requests, 0 errors, 0 caps, gates 45/45, post-boot gates 2/2) and the
  spec-decode counters as diagnostics. `AGENTS.md` now compares future variants with the
  fixed F1 medians from the same workstation; F0 remains the historic record and the
  only IaC-backed operational reference (`scripts/check-f0.py` and the archive restore
  still target F0) until the F1 recipe is promoted into IaC.
- Qualified an E09 fix for the concurrency regression with a same-client series,
  recorded in
  [`operations.md`](docs/operations.md#e09-same-client-performance-series-and-batch-uniform-qualification-on-2026-09-18).
  Re-measured from the F0 workstation, the SparkCache candidate showed C2 −7.9% and
  C4 −5.6% against frozen F0 (2 runs, owner stop). One-factor controls attributed the
  loss to mixed k=3/k=5 verify steps under concurrency (PIECEWISE decode graphs), not to
  the connector: SparkCache without DFlash2 scaled C2/C1 1.68 and C4/C1 2.75. With the
  documented `VLLM_ADAPTIVE_K_MODE=batch-uniform` as the only change, 3/3 full native
  runs (162/162 requests complete, 0 errors, 0 length caps, gates 45/45) give C1 +3.2%,
  C2 +0.7% (within noise), C4 +18.2% (the decisive gain),
  code +2.7%, prose +3.6%, prefill 8K/32K replay +109%/+49%, and a single −1.6% on 64K
  replay; spec-decode acceptance 53.5–54.9% with zero mixed steps. The candidate is
  promote-eligible; promotion is not started and needs the full R10 recipe in IaC.
  Rigmark was reset to upstream `c5a0db0`; its dropped `cache_salt` prefill passthrough
  invalidated one receipt (kept, excluded) and was reapplied as an uncommitted 12-line
  measurement fix. Window overlays stay gitignored; no `deploy.sh` ran because `--check`
  reports node drift on launcher, controller and scheduler.
- Made incremental debugging mandatory in `AGENTS.md`: start with the simplest
  plausible explanation and a small distinguishing test, propose a matched cloud or
  reference-model control before cluster-internal diagnosis, and require evidence
  before escalating complexity. Clarified that Rigmark qualifies performance and
  measurement integrity; generated-code quality, formatting and optional Go audits
  must not block the benchmark. Preserve native receipts and disclose output limits
  and unavailable visible-response latency without relabeling their gates. This
  supersedes the earlier use of E09 quality failures as a performance stop condition.
- Closed the attribution of the E09 code-workload reasoning/format defect with an
  independent, owner-authorized review, documented in
  [`operations.md`](docs/operations.md#e09-reasoning-diagnosis-attribution-closed-on-2026-09-17)
  and cross-referenced from
  [`production-recipe.md`](docs/production-recipe.md#thinking-off-compatibility).
  An external reference request to the same pinned model, with zero local
  configuration and the unmodified Rigmark code prompt, reproduces the defect at a
  comparable rate: 1/5 passes format and an independent `go test -race` oracle,
  the other four exhaust the shared 8192-token budget on reasoning alone. Re-read
  of the already-collected true/High native receipts shows the same pattern
  locally (89-656 reasoning characters, well under budget, no truncation), and the
  interrupted true/Max attempt reached the same 8192-token cap as the external
  reference. The defect tracks how much the model reasons on this prompt against
  a fixed benchmark budget, not this deployment's thinking-off compatibility code,
  FP8, KDA, or connector. The E09 SparkCache candidate's own qualification remains
  unresolved on its own merits; no recipe, lifecycle or promotion change occurred.
- Added the [E09 reasoning qualification record](E09-QUALIFICATION-2026-09-17.md)
  with paired Low correctness results and restored, pinned Rigmark tooling.
  Both thinking values pass code format 5/5, but Go failures prevent Low from
  qualifying. High completes all 15 native requests but passes format only 3/5,
  compilation 4/5, model tests 3/4 and independent/race suites 1/4; the latter
  successful sample adds forbidden prose. One implementation double-counts
  refill after a backward clock. Outcome unresolved: stop before confirmation
  and DFlash, retaining the healthy idle four-rank target-only runtime.
  Owner conversation is permitted during correctness pilots, so their timing
  does not qualify performance. The interrupted Max attempt and missing raw
  outputs are disclosed separately: 45 complete native requests plus three Max
  requests issued, two completed and one interrupted. All 98 Rigmark tests and
  the infrastructure offline check pass. No serving recipe or lifecycle changed.
- Added an evidence-based E09 code-output root cause analysis and remediation
  plan. Forty CPU render/parser combinations reproduce the forced closed
  thinking prefix with default Max effort, a protocol mismatch already present
  in F0; its causal role remains a hypothesis pending a request-only comparison.
  Offline replay of the five saved target-only Go answers passes their own
  tests in 3/5 cases; both versions of the repeated fifth test fail. Reproduced
  native audit extraction errors distinguish measurement defects from model
  errors. The plan keeps fixes in Rigmark, preserves frozen F0 and requires
  separate correctness and performance qualification. No new inference,
  lifecycle action or serving-configuration change occurred during the RCA.
- Completed the owner-authorized E09 target-only diagnostic to isolate repeated
  code output: four-rank dry-runs remove only `--speculative-config`, retaining
  all engine, connector, scheduler, model and fabric pins. The adaptive policy
  disables itself when the engine drafts zero tokens. One coordinated load
  passed both boot gates within 3.832 seconds. The unchanged native Rigmark
  decode-only invocation completed 15/15 requests with all basic gates passing,
  no foreign traffic and counter 17. Only 2/5 code answers meet the two-block,
  no-prose constraints; sample 5 self-corrects and repeats its test file even
  without DFlash. No code answer reached the cap. Disabling speculation is
  insufficient to fix the defect; its cause remains unresolved. All four ranks
  remain healthy and idle in the target-only configuration, with no promotion,
  performance qualification or automatic reference reload.
- Recorded one complete E09 replacement run with native integrity PASS, but a
  later manual review found seven code blocks and self-revision prose where the
  prompt required two blocks and no prose; the native basic gate does not check
  those requirements. Run 2 was already active and was interrupted after six
  completed requests and one interrupted stream. The model remains healthy and
  loaded, with exclusive traffic and counter 116. The owner requested another
  complete native attempt before a whole-experiment verdict. That attempt
  completed 54 requests on the same processes and reproduced the anomaly:
  code basic gate 4/5, requested two-block/no-prose format 2/5, with one repeated
  implementation capped at 8192 tokens. All nine replay continuations match and
  restores pass on 4/4 ranks, with no foreign traffic and final counter 170.
  Preserve native integrity and content results separately. The candidate is
  unqualified, overall optimization remains unresolved and no three-run median
  or component-level cause is claimed; the healthy model remains loaded.
- Added an explicit single-rail SIRCL GID preflight that runs before vLLM imports:
  reject stale indexes, inactive ports, wrong netdevs, non-IPv4/RoCEv2 GIDs and
  mismatched neighbor addresses without selecting a replacement automatically.
  September 17 inspection found the old runtime still selected unavailable index
  4 on rank 1/3; all eight selected ports currently expose the matching GID at 3.
  The new coordinated boot passed with one weight load and both functional gates
  within 3.013 seconds of health. Live identity, all sources, effective settings,
  mapped libraries and idle counter 2 passed on 4/4 ranks. The first native run
  passed 54/54 streams, basic output gates 15/15 and equal replay continuations
  9/9, with nine verified restores on every rank. A foreign chat POST more than
  three hours after its finish broke exclusivity before run 2; preserve the clean
  measured interval separately from the failed full-series gate. Qualification
  stops at 1/3 runs and 54/162 requests, unresolved with no three-run median or
  promotion. The healthy candidate remains loaded; the earlier code failure's
  cause remains unresolved. Failed prerequisite/supervisor preparation checks
  are retained with zero weight loads and zero inference requests.
- Added a portable E09 handover after the authorized September 17 boot failed
  before health: SIRCL rejected unavailable GID index 4 on rank 1/3, despite
  passing jumbo checks and matching E09 pins. One weight-load attempt sent zero
  gate or native benchmark requests. Coordinated down passed and all four nodes
  were verified stopped. The handover preserves private evidence, the uncommitted
  checkout and exact recovery steps; RDMA diagnosis remains open, with no new
  performance conclusion, automatic reference reload or promotion.
- Recorded the 2026-09-17 E09 resume check after the owner-reported Internet outage.
  SSH reaches all four nodes, but the measured process is gone: ranks 0/1 have no
  containers, ranks 2/3 retain different exited containers from September 16, all
  GPUs are idle and the API refuses connections. No inference or lifecycle action
  ran. The E09 image, ten source pins per rank, all four dry-runs, controller and
  native client still match; a private archive contains a concrete coordinated
  restart proposal. A new boot and three-run series require a new authorized
  window; the original contaminated result and frozen F0 remain unchanged.
- Stopped the authorized E09 pending-publication native series after 1/3 Rigmark
  runs and 54/162 requests: code completion passed 4/5, with one repetitive output
  capped at 8192 tokens before the observed foreign traffic. Eight additional
  non-loopback chat admissions contaminated the run; six additional completions
  explain counter 83 instead of 77, with two admissions unreconciled. The remaining
  108 requests were skipped. All nine native cache replays hit on 4/4 ranks with
  equal continuations, but qualification failed and performance is unresolved.
  A new private archive preserves native results, all-rank logs and the fixed-F0
  table with no claimed three-run median. The unchanged healthy candidate remains
  loaded under the no-automatic-reference-reload instruction; no promotion occurred.
- Passed the owner-authorized E09 pending-publication diagnostic qualification
  with one coordinated weight load: two boot gates, 21/21 requests, 9/9 equal
  cold/replay continuations and external hits 3/3 in each A/B/C condition.
  Every rank verified nine restores. Bounded state traces expose both pending
  publication and delayed scheduler visibility; admission preserved 4/4
  confirmation with observed waits of 4.830–15.829 ms. The trace revision also
  passes 60 CPU cases. The healthy candidate remains loaded with counter 23;
  timeout/error injection, native performance and promotion remain unqualified.
  Explicit private boot/stop/rollback signatures preserve the ongoing
  no-automatic-reference-reload window and unchanged production defaults.
- Recorded three separately authorized E09 cold controls: all returned identical,
  complete 32K/eight-token continuations matching the original cold, with no
  external restores. The earlier replay divergence remains unresolved. A new
  private archive contains a local pending-publication connector candidate with
  bounded waiting, mandatory four-rank confirmation and recomputation on failure
  or timeout; 59 CPU tests pass, including real background disk commit and R10
  normal/asynchronous no-forward progress. The candidate is not deployed or GPU
  qualified; the same healthy ON process remains loaded, with success counter
  169 and frozen F0 unchanged.
- Recorded the E09 replay-timing diagnosis and its mandatory stop after 2/21 requests:
  the first immediate 32K cold/replay pair returned complete eight-token streams but
  differed by one space. All nine planned pairs and actual denominators are reported;
  the remaining 19 requests were skipped. Four-rank logs confirm an external miss and
  a rank-0 commit log 642.283 ms after replay invocation, but the unmeasured delayed
  conditions leave the cause inconclusive. A separate private archive preserves the
  evidence and bounded-wait, full-quorum correction proposal; the verified ON process
  remains loaded, with no connector change, OFF control, promotion or F0 reload.
- Added an offline E09 diagnosis with all 27 replay pairs and 81 concurrent rounds,
  preserving the original benchmark archive and the loaded experimental process.
  CPU evidence reproduces missing-quorum admission after worker commit completion
  and the connector's one-to-four-rank reply-routing change. The report separates
  the strongly supported publication-timing explanation from unmeasured decode
  coordination costs, records lower observed C2/C4 speculative acceptance, and
  specifies follow-up tests without executing inference or another model load.
- Added a frozen F0 rollback reference with a portable non-site overlay, exact artifact
  pins, a separately rendered autostart drop-in, and `scripts/f0-reference.py` for
  fail-closed private capture, offline/live verification, and non-executing restore plans.
  The capture preserves the pre-change dirty source, completed IaC, resolved private site
  recipe, per-rank runtime/host/fabric receipts, generated network files, and the exact
  loaded NCCL library bytes without copying model weights or credential-bearing material.
  Live checks retain raw receipts while excluding only documented lifetime, timer, and
  packet-counter fields from stable signatures, and preserve detailed private differences.
- Added `scripts/check-f0.py`, a concise read-only F0 operational check with parallel
  bounded rank probes, private diagnostics, localhost-tunnel support, and fixture tests.
- Added an Italian optimization plan with the accepted nine-experiment catalog,
  source and hypothesis boundaries, and a fully specified three-run E01 mode-only
  Rigmark median comparison against the frozen F0 baseline.
- Added opt-in `NCCL_IB_GID_INDEX=-1` support with fail-closed validation that every
  selected active HCA has an IPv4-mapped RoCEv2 GID matching its fabric netdev address,
  both addressed edges are covered, and extra Docker arguments cannot override the
  validated selectors; the verified explicit index `3` remains the production default.
- Documented the workload priorities and the quality, prefill-throughput, and prose
  non-regression requirements.
- Added a README badge linking to the maintainer's X profile.
- Added this changelog and made a matching `Unreleased` entry mandatory for future
  code, configuration, and documentation changes.
- Added one offline validation command for syntax, links, command help, model
  manifests, chat-template rendering, host lifecycle, preflight, and the adaptive-k
  policy.

### Changed

- Completed and verified the 2026-09-12 F0 rollback point as private archive
  `f0-20260912-v4` (manifest SHA-256
  `670a1877278108112375cdc029ae005aaf49a84b0f43f7c2c616333bc3369b9a`).
  Offline integrity, live four-rank identity and source staging passed. At capture time,
  operational readiness was separately failed because rank 0 had disk/loaded-state
  `NeedDaemonReload=yes` for fabric, autostart and flusher; capture made no repair. A later
  authorized and verified rank 0 daemon reload resolved that difference and the F0
  operational check passed, without changing the frozen archive. The owner chose to skip
  the full restore rehearsal. Runtime restore remains separate from host, network, kernel,
  firmware and boot recovery, which require separate authorization and independent
  recovery access or owner physical availability.
- Tested the owner-authorized E07a direct-ring SIRCL single-rail sync-prefill candidate,
  distinct from the proposed ASIC virtual mesh. It pins SparkRing
  `b358a818786d8506086aaaabb9afe464fa2ccb49`, vLLM
  `487ecf187d3dfe74d2cf6119a92881dba403c219` and the native library SHA-256
  `f53c88b4bf885533c4d6de60dae1a9e46cf26db0695b37265955b7cc74934119`.
  Only BF16 `[Q,4096]` at `Q=1024/2048/4096/8192` uses SIRCL; `Q=128`, graphs, decode and
  other signatures retain F0 NCCL. Model, image, DFlash2, scheduler and host fabric remain
  unchanged apart from the staged payload and serving overlay. The four-rank numeric gate
  passed 20/20 exact cases, and serving passed both post-health gates plus native SIRCL
  session proof on all ranks. Three native Rigmark runs completed 162/162 requests with one
  weight load, the same processes, zero receipt-validation errors and no foreign traffic.
  R3 reached the 8192-token limit in one long-code sample with `finish_reason=length`, so
  code passed 14/15 gates while prose and structured passed 15/15 each. This mandatory gate
  failure leaves the series unresolved and not promotable; no causal attribution to SIRCL
  is claimed. Diagnostic three-value medians were code 50.513, C1 38.911, C2 53.891, C4
  73.481 and prose 28.769 token/s, versus frozen F0 50.401, 37.223, 57.009, 71.084 and
  29.220. The owner retained the healthy four-rank E07a runtime for experimental use during
  local SparkCache integration, without promotion or an F0 restore.
- Advanced E09 from the pinned-source porting study to an owner-authorized trial of the JJ
  R10 image digest
  `sha256:0d4029b3b7023cf32c37ac20279469c9a2ee16a057f25aae3bcfee9ee5fb660f`
  with expected image ID
  `sha256:5e32aaa1bbe3559e81db7706ed4286248f18d27cfdb186f6b851bf786eb43075`;
  the offline-import overlay uses the content ID directly and rejects its synthetic tag.
  The trial preserves the current FP8 target and DFlash2 revisions, TP4/DCP1, 16-GiB
  `fp8_e4m3` KV cache with block 2304, adaptive k3/k5, and E07a single-rail sync-prefill
  SIRCL. It first qualifies the new engine with SparkCache off, then compares SparkCache on
  with SIRCL, followed by the same cache configuration on NCCL. NVFP4, the packaged
  dual-rail/fused SIRCL recipe, weight changes and a pure-F0 rerun remain outside scope.
  A new private workspace records the fail-closed image inspection, exact E07a rollback
  contract, and an offline-validated OFF/SIRCL overlay and launcher that omit the old vLLM
  indexer mount, select the required R10 B12X sparse-MLA backend with block 2304, keep its
  separate PCIe all-reduce path plus R10's new default FlashInfer all-reduce disabled, and
  reject a cache connector. The operations
  guide now defines the R10 boot
  signatures and rollback, while the public template keeps the production image unchanged;
  no R10 candidate had been started when this entry was written. The local OCI source
  capture records 2,502 Python files, 17/17 critical files byte-identical to the pinned e02
  composition, and the stale package version `0.1.dev1+gd377796e8` as a separate metadata
  value rather than an engine-composition pin. The owner selected an additive image-archive
  fan-out over the existing DAC ring: rank 0 copies to ranks 1 and 3 directly and reaches
  rank 2 through a rank-0-owned SSH `ProxyCommand` over rank 1. All three transfers run in
  parallel with pinned source and destination hashes, existing trusted host identities and
  no management-data fallback, route change, host-key addition or service action. Its first
  completed snapshot verified 58 parts on every destination and copied 4,026,531,840 new
  bytes per rank in 36.973 seconds; a second 23.584-second pass completed all 73 parts and
  verified the 9,673,946,624-byte archive at its pinned SHA-256 on every rank. All-four-rank canonical checkpoint manifests also
  matched byte for byte, fixing the target and draft identities used by the prepared cache
  config. The local, not-started ON preparation now adds a tested SparkCache connector
  compatibility override that binds non-empty vLLM `cache_salt` values to all external keys
  through a domain-separated digest while preserving the legacy key for `None` or empty;
  it neither retains nor logs plaintext salts. The R10/SIRCL numeric gate passed 20/20,
  but the first OFF boot then failed before health when rank 3 found no attention backend
  for the DFlash2 noncausal sliding-window signature (head 128, BF16, `fp8_ds_mla`, block
  2304, `use_mla=false`). The observed rank-0 load was only 19/62; no inference gate or
  Rigmark request ran. The attempt is not qualified and OFF qualification still gates any
  cache-ON transition. A separate attempt-2 overlay prepared for the retry added only the existing
  R10 draft-scoped `SpeculativeConfig.kv_cache_dtype=fp8_e4m3`, retaining the adaptive
  k3/k5 table, global FP8 target cache, B12X, block 2304, transport and SparkCache-OFF
  state. Four-rank dry-run comparison proves that single JSON-field difference; BF16 was
  not selected and the failed attempt's frozen files remain unchanged. The local
  SparkCache-ON preparation now has a separate attempt-2 overlay and manifest that inherit
  the same draft-only FP8 field; the earlier ON overlay without it is not eligible to run.
  Attempt 2 passed the earlier DFlash2 initialization point on ranks 1-3, then failed
  before health during `determine_available_memory` profiling with a propagated
  `cutlass_gemm_caller ... Invalid status` error. The retained receipt does not identify
  the original operator stack, exact worker rank, layer, or shape. Rome, tool-call and
  Rigmark remained at 0 requests/runs. The subsequent E07a restore completed `up` with
  exit 0, reached health 200, and passed Rome plus tool-call 2/2. The final check confirmed
  one correct E07a stack on every rank, idle 0/0 and 47 °C with all thermal and power-brake
  flags inactive. Those values describe the restored runtime and do not assign the cause
  of the R10 failure. With the complete stack stopped again, an isolated B12X diagnostic
  executed all seven cases without a runtime error: its block path met the frozen numeric
  threshold for 2/4 shapes and its online Cutlass path for 3/3. A two-class block exclusion
  selected Marlin before arithmetic and was not adopted. The exact explicit-Triton probe
  then selected `TritonFp8BlockScaledMMKernel` and passed 7/7 cases, including block 4/4
  and online Cutlass 3/3, with no weights mounted; its receipt SHA-256 is
  `02a50dd112fd83a5e92d27ed226fce74bd3ba39cda8fa551c25ee3048c798d84`.
  Serving attempt 3 then used the sole attempt-2 delta `--linear-backend triton`; it had
  no `VLLM_DISABLED_KERNELS`, image, weight, KV, transport or cache change. All four ranks
  loaded target and draft, and the earlier profiling stage passed, but KV-cache view
  creation failed before health because the BLHNC block stride 14217984 differed from page
  size 1179648 and manager block 2304 could not split into 36 kernel blocks of 64. The
  runtime suggested block 64 or `VLLM_KV_CACHE_LAYOUT=LBNHC`; neither is selected by this
  record. Rome, tool-call and Rigmark remained at 0 POSTs, 0/3 runs and 0/162 requests.
  The owned `up` was stopped, full-cluster `down` passed, and all four ranks were verified
  with zero containers, GPU processes and active flushers. E07a restoration then completed
  `up` with exit 0 and health 200. The operator reported a sandbox network denial for the
  initial local gate watcher; its receipt retains the TERM request for the exact PID, but
  not raw stderr or an observed exit status. A later un-escalated local reproduction,
  explicitly separate from the original watcher, retained `URLError` with errno 1. The
  repeated Rome and tool-call gates passed 2/2, but about
  six minutes and 27 seconds after controller completion, so the 120-second timing rule was
  not met. No functional failure or concealing restart was recorded. The final check found
  one exact E07a stack on all four ranks, restart counts 0, idle 0/0, 47 °C, and all thermal
  and power-brake flags inactive. A later CPU-only probe of the exact R10 image reproduced
  the BLHNC geometry error and passed both allocator controls in 3.886 seconds without GPU
  visibility, weights or E07a changes. It verifies the guard rather than a supported remedy
  and is not a fourth serving boot. The failure is scoped to this exact recipe and does not establish
  general R10 or SparkCache incompatibility. A new local SparkCache-ON overlay inherits the
  exact explicit-Triton OFF recipe; after removing its read-only connector mount and
  `--kv-transfer-config`, four-rank dry runs are identical to OFF. SparkCache ON remains
  unstarted and blocked.
  The owner then authorized an attempt-4 compatibility patch for the exact BLHNC failure.
  Three read-only module overrides preserve the target B12X/KDA shared allocation, assign
  each DFlash draft layer a dense backing allocation, and budget the one global block pool
  as padded shared-target bytes plus all draft pages. An exact-image Beast0 CPU test passed
  with 3 scheduler blocks and 5 draft layers: 6 allocations, 60,171,264 bytes within a
  60,761,088-byte test budget, distinct storages, multi-block copy, replicated DCP metadata,
  unchanged target-only views, and the original stride/page guard reproduced. E07a retained
  the same container, process and health 200 through the isolated test. The preparation is
  frozen under manifest SHA-256
  `e6825e093f5f8ae52b0e5b1de3fea27d990f51d5402da1efea4d1b1069a41571`. Its serving boot
  loaded target and draft on all four ranks, completed graphs and KV view creation, and
  reached health 200 with 1,435,070 cache tokens (5.47x at context 262144). The first Rome
  request nevertheless returned HTTP 200 with exactly 64 `!` characters and
  `finish_reason=length`; the tool gate was skipped and Rigmark remained at 0/3 runs and
  0/162 requests. This exact attempt-4 recipe is discarded and coordinated `down` passed.
  E07a restoration completed `up` with exit 0 in 971.241 seconds, reached health 200 and
  passed Rome plus tool-call 2/2 within 8.826 seconds of first health. The final check found
  one exact E07a stack per rank, restart counts 0, idle 0/0, 48 °C and inactive thermal and
  power-brake flags. The failure does not establish general R10 or SparkCache incompatibility;
  SparkCache was off and its ON candidate remains unstarted and blocked.
  The owner then authorized attempt 5 with the upstream pooled-tail correction from commit
  `618562444d73742e4e872defcad4d37f477f5c59`. The reviewed revision also applies the same
  `live_history` calculation to R10's separate physical helper used by DCP1 decode-only.
  An initial runner was rejected for an unavailable Docker runtime before creating a
  container; the corrected exact-image test made the original source fail the expected
  eight short-tail route cases and made the revision pass all 16 logical and physical
  cases, including prompt length 25, history boundaries and long physical remapping. A
  separate weight-free CUDA replay of the opt-in numeric diagnostics passed. Four-rank
  staging, dry-run, hashes and stopped-state checks passed, but the first instrumented OFF
  boot failed during model loading because its wrapper did not present the instrumented
  GLM5 model required by `model_runner`. The controller was stopped with exit 143 after
  457.45 seconds; the post-stop check found zero containers, idle GPUs and inactive flushers
  on all four ranks. No health, functional request, inference or Rigmark run occurred. This
  records a diagnostic-wrapper failure, not a failure of the pooled-tail fix. Diagnostic
  boots remain invalid for Rigmark. A second boot with the corrected wrapper loaded target
  and draft on all four ranks, completed `up` in 1002.05 seconds and reached health, then
  failed the first content gate with HTTP 200, 64 exclamation marks and
  `finish_reason=length`; the tool gate was skipped. Its 16 diagnostic records all came from
  four startup warmup steps per rank before health, so the failed request itself was not
  observed and the cause remains unresolved. Coordinated `down` passed in 12.973 seconds.
  No clean OFF, native Rigmark or SparkCache ON run started; the exact instrumented recipe
  is discarded. E07a restoration completed `up` in 971.111 seconds and passed both
  functional gates within 8.801 seconds of health. Final receipts confirm the exact
  four-rank recipe, restart counts 0, idle 0/0, exactly two gate requests, stopped
  flushers and 48–49 °C without active thermal or power-brake flags. A local flusher
  assertion was too narrow for collected transient units (exit 4 with `inactive`);
  no node repair was needed. SparkCache ON was staged on all four nodes but never
  launched. The owner requested that further root cause analysis be performed
  directly by the root agent.
- Added a direct E09 root cause analysis distinguishing resolved startup blockers from the
  unresolved repeated-content failure, with corrected warmup attribution, bounded checkpoint
  evidence and the unproven SharedExperts lifetime lead. Documented the private request-bound
  diagnostic's passing ID/CPU checks and its outstanding GPU/serving validation; production
  settings remain unchanged.
- Executed the request-bound E09 diagnostic: the real request still returned 64 exclamation
  marks, with huge first-block residuals and zero final logits on all four ranks. Reproduced
  a missing legacy quantization-exclusion mapping that makes R10 treat BF16 KDA projections
  as serialized FP8 with unloaded scale parameters. The private clean mapper correction
  passes 170 precision selections, unchanged mapping of all 76,108 checkpoint keys and
  eight GPU projection comparisons. Its clean OFF boot corrected the Rome response but
  failed the tool gate; full serving correctness was not established. Recorded
  the owner's continuous diagnosis window and cancellation of the E07a reload already in
  progress, with coordinated cleanup and zero restore-gate requests. No promotion or
  production-default change was made.
- Reproduced a second E09 generation defect: pure-decode cache slots were already physical
  but the missing B12X provider method made the backend translate them twice. The private
  provider correction passes six GPU dispatch/address cases with unchanged mixed-prefill
  and DCP2 paths. Recorded all nine OFF diagnostic requests, including five incorrect tool
  calls, and the owner's direct transition to one corrected SparkCache ON boot followed by
  three native Rigmark runs after the functional gates. All four live configurations
  match the staged ON recipe; startup completed and both gates passed within 3.073 seconds
  of health. Updated the
  active overlay and rollback guidance without promoting or changing production defaults.
- Completed three native R10/SIRCL/SparkCache runs on one corrected ON process: 162
  requests, zero native validation or stream errors, 45/45 basic output gates and 0/15
  long-code caps. Recorded real external restores for 21/27 replay requests on all four
  ranks and equal eight-token cold/replay continuations in 27/27 pairs. Published the fixed
  F0 versus three-run-median comparison, including the consistent C2 shortfall, variable
  32K replay latency and the owner-requested move of the benchmark client from macOS to
  Beast0. The result remains unresolved without promotion. Final verification confirms
  four unchanged processes, exactly 164 requests including the two boot gates, stopped
  flushers and no active thermal flags in sampled load/final observations; the experimental
  process stays running under the continuous-window instruction. Archived exact commands,
  configuration, source payloads, failures and native receipts for review and rollback.
- `scripts/deploy.sh` now installs a second SHA-identical controller at
  `~/tp4/tp4ctl-f0-reference`; future F0 restoration and its prepared autostart drop-in
  use that verified regular file, avoiding controller source embedded in a remote shell.
- Recorded the first E04 exact mixed-batch CUDA graph attempt as unresolved and not benchmarked after
  an operational controller and gate execution failure: the static dispatcher and four-rank
  runtime identity passed, but functional gates and all three planned native runs were skipped
  (0/162 requests). This execution does not assess the CUDA graph variant; after verified
  all-rank cleanup, an owner-authorized direct E04 retry passed its stopped-state preflight and
  functional gates. Its first native run was clean; its second completed 54 valid requests but
  was excluded from performance comparison after six extra chat POST attempts and, separately,
  four successful completions beyond the native count were observed in its bounded window;
  sanitized logs do not map one set to the other. With only one
  clean run there is no E04 median and the performance result remains unresolved; the owner
  closed E04 as showing no demonstrated benefit, skipped the third run and a replacement
  series, declined promotion, and restored the F0 lifecycle and functional gates with one
  separate weight load; the final automatic F0 operational check also passed.
- Recorded the owner's decision to discard E05 native MTP-k3/local-argmax for current
  general use and retain F0 after three valid native runs and 162 requests with zero
  native errors: long-code decode and C2 regressed despite C1 and C4 gains. This does not
  establish that MTP is generally unhelpful, and no further E05 runs are planned. The
  C1/C2/C4 workloads cap each agent at 256 tokens and do not establish long parallel-
  generation performance. F0 serving and functional gates were restored, while the
  final operational check still reports only rank 3's fabric systemd unit as needing a
  daemon reload; no repair or unsupported cause attribution was made.
- Recorded E01 as owner-discarded after one valid 54-request run, with the remaining runs
  stopped and no three-run median, and allowed an authorized prepared candidate to follow
  a variant directly from a fresh F0-derived recipe without an intermediate F0 reload.
- Renamed the repository from `GLM-5.3-Flash-FP8-4-DGX-Spark` to
  `GLM-5.3-Flash-FP8-4-DGX-Spark-Switchless` and clarified that its verified
  ConnectX-7 fabric connects the nodes directly without a dedicated network switch.
- Renamed the repository from `tp4-glm53-fp8-gx10` to
  `GLM-5.3-Flash-FP8-4-DGX-Spark`.
- Centered the README tables for clearer presentation on GitHub.
- Documented the configured 256K (262,144-token) context window in the README and
  corrected the GitHub About description.
- Consolidated operator documentation into five task-oriented guides and shortened
  the repository and agent entry points.
- Moved the controller, launcher, and public node assets under `scripts/` while
  preserving their installed paths on cluster hosts.
- Moved the remaining ignored node configuration and operator tools under
  `scripts/node/`, leaving no root-level `node/` directory.
- Documented `scripts/node/` as the source of files installed on cluster hosts, including
  runtime patches, host configuration, model manifests, and the patched NCCL build.
- Published the 2026-09-05 loopback benchmark aggregate and its method limits without
  private paths, raw logs, or node addresses.
- Reworked the README benchmark summary into a six-metric comparison against a fixed
  initial baseline, with protocol limits and secondary metrics kept in `docs/bench.md`.
- Consolidated attribution and third-party terms in `CREDITS.md`.
- Recorded the permanent project rule to run checks locally and never use GitHub
  Actions.
- Frozen the three native F0 runs measured on 2026-09-11 as the permanent comparison
  baseline, with exact per-run metrics, fixed medians, functional counts, public recipe
  identity, Rigmark settings, source fingerprints, and receipt hashes recorded in
  `docs/baseline-f0.json`. Future experiments reuse it unless the owner explicitly asks
  to rerun or replace it; each variant runs three consecutive times by default, preserves
  one loaded process across all three runs, and reports their median and request counts.
- Defined direct use of the independent Rigmark suite at `~/workspace/jacopo/rigmark`
  as the benchmark interface: one all-node variant at a time is compared with the fixed
  F0 baseline under matching inputs and conditions across performance, error, and
  functional results, with gains judged beyond measured noise rather than a fixed
  floor. Outcomes now use promote, discard, or unresolved, with reproducible evidence
  and restoration or IaC-backed promotion before the next owner-agreed experiment;
  measurement fixes remain in Rigmark instead of private wrappers or a parallel
  admission framework. Durable ranked success criteria are led by agentic code
  generation C1 and two-to-four-agent
  concurrency, followed by prose and prefill without higher-priority regressions;
  structured-generation speed remains diagnostic, conflicting primary outcomes require
  a decision, and correctness, error integrity, and healthy four-rank operation remain
  mandatory gates.

### Fixed

- Hardened the fast F0 checker to reject overlay changes across the complete protected
  topology and launch argument sets, verify the live distributed master identity, retain
  only bounded structured probe status, reject malformed options and exact idle-metric
  samples without rejecting suffixed sibling series, and require any available drafter
  revision metadata to match the frozen reference.
- Corrected installation and fabric-recovery guidance to require all five runtime
  signatures before declaring the cluster ready.
- Made lifecycle commands fail closed on unverifiable rank, container, or flusher state;
  failed starts now clean all four ranks, and effective overlays cannot change TP4
  cardinality or the configured container name.
- Rejected out-of-range, malformed, and ambiguous leading-zero fabric IPv4 addresses
  before netplan generation.
- Made help, deploy discovery, and offline tests tolerate optional internal profiling,
  collective microbenchmark, MoE tuning, and publication tooling being absent.
- Updated public links and source comments after the documentation consolidation.
- Clarified parallel-stack discovery, fabric versus manual RDMA checks, boot ordering,
  feature rollback boundaries, and fail-closed backup and validation of a rebuilt NCCL
  candidate.

### Removed

- Removed the repository benchmark harness, published performance results, and
  benchmark-only deployment and validation hooks; post-boot functional gates now live
  in the operations guide.
- Removed historical studies, raw experiment indexes, internal tuning tools, mirror
  tooling, and the outdated performance graphic from the public repository surface;
  local copies remain available outside the published file set.
- Removed the GitHub Actions workflow; offline validation remains available through
  `scripts/check.sh`.
