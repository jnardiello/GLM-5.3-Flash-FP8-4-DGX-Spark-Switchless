# Benchmark reports

Readable reports live here; machine-readable frozen records and portable experiment extracts live in [historical benchmark datasets](../historical_benchmarks/README.md). The [project README](../../README.md) shows the current baseline. Historical records remain available regardless of whether a candidate was accepted, discarded, unresolved or stopped.

## Frozen references and reproduction

| Record | Included native runs / requests | Report |
| --- | ---: | --- |
| 2026-09-11 baseline | 3 / 162 | [Report](baselines/2026-09-11.md) |
| 2026-09-18 baseline | 3 / 162 | [Report](baselines/2026-09-18.md) |
| 2026-09-19 baseline | 2 / 108 | [Report](baselines/2026-09-19.md) |
| 2026-09-19 IaC reproduction | 1 / 54, separate | [Report](reproductions/2026-09-19-iac.md) |

[Historical comparison table and archived charts](comparisons/2026-09-19-vs-2026-09-11.md).

The README's current-only figures are generated from the 16 saved baseline medians.
From the repository root, run `python3 scripts/plot-baseline-comparison.py` in an
environment with `matplotlib==3.11.2`. It writes PNG/SVG files under
`docs/plots/baselines/2026-09-19/` without running inference. Historical comparison
images are retained unchanged in `docs/plots/comparisons/`.

## Experiment outcomes

| Experiment | Evidence / recorded outcome |
| --- | --- |
| [2026-09-18-e20-w8a16](experiments/2026-09-18-e20-w8a16.md) | Original W8A16: 3 runs/162 requests; separate pilot 36; decision deferred |
| [2026-09-19-e20-screen](experiments/2026-09-19-e20-screen.md) | Two 13-request diagnostic screens; no disposition |
| [2026-09-19-e20-hybrid-kv16](experiments/2026-09-19-e20-hybrid-kv16.md) | One full run and one partial runtime failure |
| [2026-09-19-e20-cachefix-stop](experiments/2026-09-19-e20-cachefix-stop.md) | Owner stop after two completed calls; no native receipt |
| [2026-09-19-e20-kv15](experiments/2026-09-19-e20-kv15.md) | Two accepted runs/108 requests; third excluded |
| [legacy-e01](experiments/legacy-e01.md) | Summary only: discarded by owner |
| [legacy-e04](experiments/legacy-e04.md) | Summary only: unresolved; one clean performance run |
| [legacy-e05](experiments/legacy-e05.md) | Summary only: discarded by owner, three-run summaries |
| [legacy-e06](experiments/legacy-e06.md) | Partial native receipt: discarded after memory failure |
| [legacy-e07a](experiments/legacy-e07a.md) | Summary only: unresolved, three-run summaries |

## Reading and preserving results

- Frozen medians retain their original denominator. A later deployment run does not silently extend an accepted baseline.
- Complete-run counts, partial-call accounting, normal token caps and native gate failures are reported separately. Missing timings are unavailable, not zero.
- A replaced candidate is not automatically classified as discarded. Preserve the recorded decision and any later acceptance as separate facts.
- Public extracts omit request bodies, generated responses, raw cache salts and site identifiers. Receipt SHA-256 digests establish which immutable private originals were used. Extracts are not byte-for-byte native receipts.
- Summary-only reports cannot supply missing request rows or per-run distributions. The September 11 original receipts were not available; its frozen per-run metric triples remain intact.
- No benchmark, cluster request or baseline remeasurement was performed to create this archive. Nsight profiling and kernel microbenchmarks are separate from native full-suite performance results. Planned experiments without results are not listed as completed outcomes.

For future work, preserve native output privately first, then add a portable dataset under `historical_benchmarks/experiments/`, a report here and an index row. Record the measured recipe, protocol, included/excluded runs, counts, source hashes, outcome and evidence limits. Keep raw native receipt bytes unchanged.
