# Changelog

Changes to the deployment, tools, and public documentation are recorded here.
Versions and releases are created only at the owner's explicit request.

## Unreleased

### Added

- Added the Current production configuration: the digest-pinned SparkRing/SparkCache
  image, persistent prefix caching, and SIRCL single-rail prefill transport. Tracked
  SHA-256 manifests verify operator-supplied connector and transport files; the payload
  itself is not redistributed.
- Added six engine override modules for cache allocation, the pooled indexer, and
  checkpoint mapping, plus the connector configuration. Deployment installs these
  assets, and node verification reports SparkCache and SIRCL payload integrity.
- Added frozen [Current](docs/baseline-f1.json) and [Previous](docs/baseline-f0.json)
  performance references with per-run values, medians, recipe identities, benchmark
  settings, request counts, and evidence hashes.
- Added a read-only operational identity checker with bounded four-rank probes,
  localhost-tunnel support, and private diagnostic reports.
- Added private capture, verification, and restore-plan tooling for the Previous
  configuration, backed by a portable reference overlay, immutable launcher, artifact
  manifests, and a dedicated restore controller. Restore plans distinguish runtime
  recovery from host and network recovery.
- Added validated automatic RoCEv2 GID selection and a SIRCL-specific preflight that
  rejects stale device, GID, address, and peer selections before loading weights.
- Added offline validation for syntax, public links, command help, manifests,
  chat-template rendering, lifecycle, preflight, and the adaptive scheduler.

### Changed

- Clarified the repository's public installer purpose and documentation rules: write
  reusable guides for external DGX Spark TP4 users and keep local operational history private.
- Made Current the base IaC configuration, with no experiment overlay required:
  SparkCache enabled, automatic NCCL GID selection, explicit B12X attention and Triton
  compute backends, a separate FP8 drafter cache type, and documented Previous rollback.
- Selected `batch-uniform` adaptive verification so concurrent requests use one draft
  verification length per step and retain full decode graphs. The qualified three-run
  series completed 162 requests without errors; a separate 54-request run reproduced
  the configuration through IaC. See the
  [qualification summary](docs/production-recipe.md#qualification-and-reproduction).
- Made the identity checker validate Current by default while retaining explicit
  Previous-baseline selection and Previous archive capture.
- Organized public documentation around installation, operations, fabric, and the
  production recipe. The README presents Current / Previous measurements and routes
  readers to complete procedures; internal identifiers remain in technical filenames
  and machine-readable records.
- Separated experiment plans, handover, diagnosis reports, and session chronology from
  public documentation. Local copies remain available and ignored by Git; the public
  changelog summarizes lasting changes and the guides retain applicable conclusions.
- Established native Rigmark as the performance interface, comparing three consecutive
  variant runs with the frozen Current medians. Code and concurrent-request performance
  lead evaluation; generated-answer quality audits are separate from performance gates.
- Documented incremental debugging and reference-model comparisons before attributing
  unexpected output to cluster components.
- Organized controllers, launchers, and node assets under `scripts/`, preserving their
  installed host paths. Documented the verified switchless ring and 256K context window.

### Fixed

- Hardened the SparkCache launcher with image-ID and payload-hash verification,
  duplicate connector rejection, and exactly-once connector mounting. Disabled competing
  all-reduce paths and plugin autoload; `/health` is the readiness signal when the
  SIRCL entrypoint cannot supply the image's Docker healthcheck marker.
- Corrected engine overrides for dense drafter cache allocation, pooled-indexer tail
  handling and physical selection, and legacy checkpoint quantization exclusions.
- Made digest-pinned images verify against their own digest and expanded home-relative
  bind-mount sources consistently between the launcher and node verification.
- Hardened topology, argument, idle-state, and distributed identity validation. Lifecycle
  failures stop the full cluster; overlays cannot change rank count or container identity.
- Rejected malformed, out-of-range, and ambiguous leading-zero fabric addresses before
  netplan generation; generated site files remain derived from one local configuration.
- Preserved observation timing, stale-output guards, and fallback behavior in the
  adaptive scheduler; kept coverage for the local chat-template compatibility adapter.
- Clarified boot ordering, post-boot gates, all six runtime signatures, GID selection,
  parallel-stack discovery, and feature-specific rollback boundaries.
- Made the Markdown checker exclude internal notes and reject public links to them,
  including when those files are present in the operator's checkout.
- Kept help, discovery, and offline checks usable without optional internal tools.

### Removed

- Removed internal handover, optimization plans, and diagnostic reports from the
  tracked file set, and moved detailed operating history out of public guides.
- Removed the repository benchmark harness and benchmark-only deployment hooks;
  performance measurement uses Rigmark and post-boot gates remain in Operations.
- Removed internal tuning and mirror tooling, historical studies, and the obsolete
  performance graphic from the public repository surface.
- Removed GitHub Actions; validation runs locally through `scripts/check.sh`.
