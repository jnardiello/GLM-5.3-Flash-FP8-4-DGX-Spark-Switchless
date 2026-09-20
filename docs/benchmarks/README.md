# Benchmark reports

Readable reports live here; machine-readable frozen records and portable experiment extracts live in [historical benchmark datasets](../historical_benchmarks/README.md). The [project README](../../README.md) shows the current baseline. Historical records remain available regardless of whether a candidate was accepted, discarded, unresolved or stopped.

## Frozen references and reproduction

| Record | Included native runs / requests | Report |
| --- | ---: | --- |
| 2026-09-11 baseline | 3 / 162 | [Report](baselines/2026-09-11.md) |
| 2026-09-18 baseline | 3 / 162 | [Report](baselines/2026-09-18.md) |
| 2026-09-19 previous base | 2 / 108 | [Report](baselines/2026-09-19.md) |
| 2026-09-19 IaC reproduction of previous base | 1 / 54, separate | [Report](reproductions/2026-09-19-iac.md) |
| 2026-09-19 E03 accepted current reference | 3 / 162 | [Report](baselines/2026-09-19-e03.md); [IaC promotion status](../historical_benchmarks/baselines/2026-09-19-e03/promotion.json) |
| 2026-09-20 E03 IaC benchmark | 0 included; 1 / 54 excluded | [Excluded report](reproductions/2026-09-20-e03-iac.md); owner rejects this run as reliable evidence; deployment and functional checks recorded separately |

[Historical comparison table and archived charts](comparisons/2026-09-19-vs-2026-09-11.md).

The README's comparison figures show all 16 saved medians for the current E03 reference
and the previous September 19 base, with percentage changes from unrounded values.
The legends distinguish the same-date records: three accepted runs / 162 requests for
current E03 and two / 108 for the previous base. Small changes retain exact deltas
alongside the owner's “≈ unchanged” display convention.
From the repository root, run `python3 scripts/plot-baseline-comparison.py` in an
environment with `matplotlib==3.11.2`. It writes PNG/SVG files under
`docs/plots/comparisons/2026-09-19-e03-vs-2026-09-19/` without running inference.
The [dated current report](baselines/2026-09-19-e03.md) links the figures and both
frozen sources. Original current-only figures remain unchanged in
`docs/plots/baselines/2026-09-19-e03/`; older comparisons remain archived separately.

## Experiment outcomes

| Experiment | Evidence / recorded outcome |
| --- | --- |
| [2026-09-19-e03-draft-budget-c1-review](experiments/2026-09-19-e03-draft-budget-c1-review.md) | 3 C1-only executions / 9 requests + 3 full suites / 162 requests, same loaded candidate; C1 −0.17%, C4 −0.68%, prose −2.17% versus baseline; owner accepts the tradeoffs, outcome promote; subsequent IaC deployment verified, September 20 benchmark excluded; original values unchanged |
| [2026-09-19-e03-replay-draft-budget](experiments/2026-09-19-e03-replay-draft-budget.md) | 3/3 full suites / 162 requests + separate 12-request C4 screen; effective draft-budget cap; original decision_required assessment preserved, later owner acceptance recorded in the C1 follow-up |
| [2026-09-19-e03-replay-views](experiments/2026-09-19-e03-replay-views.md) | 3 full runs + C1/C4 repeats / 177 requests; 162 in main comparison, 12 in separate C4 check; only anomalous C1 replaced; C4 repeat -3.97% throughput vs baseline; parent of the draft-budget follow-up, decision_required |
| [2026-09-19-e03](experiments/2026-09-19-e03.md) | 3 runs / 162 requests; cold prefill +9–10%, C2 +5.1%, replay tradeoff; decision_required; preserved as parent of the separately measured replay follow-up |
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
- Native benchmarks, diagnostic probes and imported summaries are identified separately in each record. Frozen references are not extended by later experiments. Nsight profiling and kernel microbenchmarks are separate from native full-suite performance results. Planned experiments without results are not listed as completed outcomes.

For future work, preserve native output privately first, then add a portable dataset under `historical_benchmarks/experiments/`, a report here and an index row. Record the measured recipe, protocol, included/excluded runs, counts, source hashes, outcome and evidence limits. Keep raw native receipt bytes unchanged.
