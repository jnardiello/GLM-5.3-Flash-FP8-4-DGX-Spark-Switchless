# Changelog

Changes to the deployment, tools, and public documentation are recorded here.
Versions and releases are created only at the owner's explicit request.

## Unreleased

### Added

- Added reproducible comparison charts showing all 16 metrics and individual runs
  from September 11 versus the two accepted September 19 runs plus its separate IaC
  reproduction, with that run identified and the frozen baseline unchanged.
- Added the September 19 performance reference and a dated README comparison with
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

- Removed the unused GPU-clock diagnostic tool and its dedicated documentation. The
  tested GB10 driver did not change effective load clocks; production and rollback
  recipes do not invoke the tool. Host lifecycle tests retain their generic coverage.
- Removed duplicate and superseded descriptions of the current baseline from the
  unreleased notes, keeping dated evidence and rollback instructions in their owners.
