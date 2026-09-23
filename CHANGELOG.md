# Changelog

Changes to the deployment, tools, and public documentation are recorded here.
Versions and releases are created only at the owner's explicit request.

## Unreleased

### Added

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
