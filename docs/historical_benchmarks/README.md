# Historical benchmark datasets

This archive preserves frozen baseline records and portable extracts of experiment
results. Start with the [benchmark reports](../benchmarks/README.md) for readable
tables, measured recipes, decisions and evidence limits.

| Location | Contents |
| --- | --- |
| [`baselines/2026-09-11/`](baselines/2026-09-11/baseline.json) | Initial Rigmark reference, three runs |
| [`baselines/2026-09-18/`](baselines/2026-09-18/baseline.json) | Historical reference, three runs |
| [`baselines/2026-09-19/`](baselines/2026-09-19/baseline.json) | Previous base, two accepted runs; separate reproduction and diagnostic probe records |
| [`baselines/2026-09-19-e03/`](baselines/2026-09-19-e03/baseline.json) | Current accepted reference, three final full suites; separate [IaC promotion status](baselines/2026-09-19-e03/promotion.json) |
| `experiments/<date-experiment>/` | Portable results for accepted, discarded, unresolved, incomplete or excluded measurements |

Baseline JSON files are immutable. Their original embedded paths describe provenance;
the links in the dated reports identify their current locations. The September 19
[IaC reproduction](baselines/2026-09-19/reproduction.json) and
[short decode probes](baselines/2026-09-19/context-decode-probes.json) remain separate
from the accepted baseline medians.

Experiment extracts contain supported measurements, counts and source digests.
Incomplete runs retain their recorded limits; summary-only records cannot replace
missing native request rows. Each experiment has a matching report in
[`docs/benchmarks`](../benchmarks/README.md), including its outcome and exclusions.

Native JSON, cards and logs stay in the ignored
[`docs/rigmark_reports`](../rigmark_reports/README.md) directory. Preserve those
originals unchanged before publishing a portable extract. Figures have their own
archive under `docs/plots/baselines/`, `docs/plots/experiments/` or
`docs/plots/comparisons/`; the corresponding reports link to them.

Follow the [agent benchmark procedure](../../AGENTS.md#optimization-workflow) when
adding a run. Every invocation must specify an absolute `--output` path inside
`docs/rigmark_reports`; update the report index as new evidence becomes available.
