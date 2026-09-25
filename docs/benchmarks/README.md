# Benchmark reports

Readable reports live here; machine-readable frozen records and portable experiment extracts live in [historical benchmark datasets](../historical_benchmarks/README.md). The [project README](../../README.md) shows the current baseline. Historical records remain available regardless of whether a candidate was accepted, discarded, unresolved or stopped.

## Frozen references and reproduction

| Record | Included native runs / requests | Report |
| --- | ---: | --- |
| 2026-09-11 baseline | 3 / 162 | [Report](baselines/2026-09-11.md) |
| 2026-09-18 baseline | 3 / 162 | [Report](baselines/2026-09-18.md) |
| 2026-09-19 previous base | 2 / 108 | [Report](baselines/2026-09-19.md) |
| 2026-09-19 IaC reproduction of previous base | 1 / 54, separate | [Report](reproductions/2026-09-19-iac.md) |
| 2026-09-19 E03 previous reference | 3 / 162 | [Report](baselines/2026-09-19-e03.md); [IaC promotion status](../historical_benchmarks/baselines/2026-09-19-e03/promotion.json) |
| 2026-09-20 E03 IaC benchmark | 0 included; 1 / 54 excluded | [Excluded report](reproductions/2026-09-20-e03-iac.md); owner rejects this run as reliable evidence; deployment and functional checks recorded separately |
| 2026-09-23 E21 previous reference | 3 / 162 | [Report](baselines/2026-09-23-e21.md); [promotion record](../historical_benchmarks/baselines/2026-09-23-e21/promotion.json) |
| 2026-09-23 E22b previous reference | 3 / 162 | [Report](baselines/2026-09-23-e22b.md); [promotion record](../historical_benchmarks/baselines/2026-09-23-e22b/promotion.json) |
| 2026-09-24 E27 previous reference | 3 / 162 | [Report](baselines/2026-09-24-e27.md); [promotion record](../historical_benchmarks/baselines/2026-09-24-e27/promotion.json) |
| 2026-09-25 E27c accepted current reference | 3 / 162 | [Report](baselines/2026-09-25-e27c.md); [promotion record](../historical_benchmarks/baselines/2026-09-25-e27c/promotion.json) |

[Historical comparison table and archived charts](comparisons/2026-09-19-vs-2026-09-11.md).

The README's comparison figures show all 16 saved medians for the current E27c reference
and the previous E27 reference, with percentage changes from unrounded values. Both
records have three accepted runs / 162 requests, measured over the same direct LAN client
path. Small changes retain exact deltas alongside the owner's “≈ unchanged” display
convention.
From the repository root, run `python3 scripts/plot-baseline-comparison.py` in an
environment with `matplotlib==3.11.2`. It writes PNG/SVG files under
`docs/plots/comparisons/2026-09-25-e27c-vs-2026-09-24-e27/` without running inference.
The [dated current report](baselines/2026-09-25-e27c.md) links the figures and both
frozen sources. Earlier comparisons, including E27 versus E22b, remain archived unchanged
under `docs/plots/`.

## Experiment outcomes

| Experiment | Evidence / recorded outcome |
| --- | --- |
| [2026-09-25-e27c-queued-cadence](experiments/2026-09-25-e27c-queued-cadence.md) | 3 full suites / 162 requests on one retained load, the protocol-2 interference phase and a four-simultaneous-32K diagnostic; against E27: C4 per-stream TTFT -27.1%, C2 +2.4%, C4 +1.2%, C1 -1.8%, cold prefill -1.6% to -3.3%; with four 32K prompts arriving together the first stream decodes at 11.3-12.5 tok/s instead of 6.2. **Promote**, owner accepted; current reference. |
| [2026-09-24-e27b-long-prefill-cadence](experiments/2026-09-24-e27b-long-prefill-cadence.md) | 3 full suites / 162 requests on one retained load, the protocol-2 interference phase against a same-day E27 reference and a four-simultaneous-32K diagnostic; against E27: C4 per-stream TTFT -27.2%, code decode +4.5%, C1 -4.0% (treated as noise by the owner); several simultaneous long contexts still stalled the first stream. **Superseded**: accepted as a candidate and promoted inside E27c. |
| [2026-09-24-e27-prefill-interval](experiments/2026-09-24-e27-prefill-interval.md) | 3 full suites / 162 requests on one retained load, the new Rigmark interference phase, a four-simultaneous-32K diagnostic and 2 same-day E22b LAN control suites / 108 requests; running agents decode 3–5× faster during another request's 8–32K cold prefill; C4 per-stream TTFT +35% and C4 -3.3% against the control; zero errors; owner accepts, outcome promote; became the September 24 E27 reference |
| [2026-09-23-e23-drafter-head](experiments/2026-09-23-e23-drafter-head.md) | 3 full suites / 162 requests on one retained load plus matched probes; against E22b: C1 +3.50%, code decode +0.97%; C2 -1.96%, C4 -1.89%; 32K cold prefill -2.47% below the E22b range in every suite. Zero errors; relayed-client series excluded; outcome discard, owner closed it |
| [2026-09-23-e22b-drafter-context-bf16](experiments/2026-09-23-e22b-drafter-context-bf16.md) | 3 full suites / 162 requests on one retained load plus matched diagnostic probes; code decode +3.03%, C1 +2.23%, prose +2.97% against E21, C4 unchanged, zero errors, gates 45/45; owner accepts, outcome promote; became the September 23 E22b reference |
| [2026-09-23-e22-drafter-w8a16](experiments/2026-09-23-e22-drafter-w8a16.md) | 3 full suites / 162 requests plus one supplementary suite and matched probes; code, C1 and prose gains, C4 unchanged, a real 8K cold prefill loss (-3.0% in matched probes) from the converted context K/V projection; outcome superseded by E22b |
| [2026-09-23-e21-bf16-residue](experiments/2026-09-23-e21-bf16-residue.md) | 3 full suites / 162 requests on one retained load; C4 +9.88% and C2 +4.68% with every suite outside the E03 range, prose +4.02%, C1 +1.21%; code gate 13/15 from two output-budget stops; zero errors; owner accepts, outcome promote; became the September 23 E21 reference without a separate reproduction run |
| [2026-09-21-e03-c5-15gib](experiments/2026-09-21-e03-c5-15gib.md) | One decode-only native execution / 3 requests; 3/3 native gates and both post-boot gates passed; owner retains the max-five operational recipe; performance unresolved, with no promotion or new baseline |
| [2026-09-21-c4-no-admission-3x](experiments/2026-09-21-c4-no-admission-3x.md) | 4 suites / 216 measured requests: runs 1, 3 and 4 form the accepted three-suite / 162-request aggregate; run 2 is excluded for a verified competing non-benchmark client. Mixed primary results include code -1.67%, C1 -2.28%, C2 +0.61% and C4 +1.85%; outcome `unresolved`, no promotion |
| [2026-09-21-c4-native-admission-28](experiments/2026-09-21-c4-native-admission-28.md) | Performance `unresolved`; integration `owner_withdrawn`, with verified coordinated return to the complete original C4 recipe. Archived R10 backport evidence covers the final four-active/24-waiting cap, HTTP 503, cancellation recovery and bounded replay; one full native suite / 54 requests observed C4 throughput +5.78% and C1/C4 TTFT +15.74%/+6.99% versus fixed E03, with no vLLM release upgrade or permanent promotion |
| [2026-09-20-c4-262k-activation](experiments/2026-09-20-c4-262k-activation.md) | Lifecycle and capacity record with no native receipt: one coordinated four-rank transition, both post-boot gates 3.078 seconds after health, and four full-context request bodies at 261,632 prompt tokens completed in a cold and a replay phase with four verified per-rank commits and loads. Zero native Rigmark suites and zero requests, so no throughput, latency or prefill value is supported; no promotion and no new baseline |
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
