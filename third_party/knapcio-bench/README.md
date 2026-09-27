# knapcio-bench (vendored)

Origin: https://github.com/knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4
Commit: `dddb034717cb84c362d8a370b81723fa4ee0c4d2` (2026-09-26)
Vendored: 2026-09-27

## Files

- `bench/qeval.py` — the auto-scored quality-gate runner and exact McNemar
  comparison CLI (paired greedy before/after comparison).
- `bench/qeval_tasks.py` — the 75-task set (25 code, 8 json, 14 math,
  16 reason, 7 format, 5 prose) with deterministic checkers.
- `bench/hardset.py` — 30 harder prompts with no grader, for blind
  qualitative comparison.
- `bench/tasktime/` — the six seeded agentic repositories and the
  `run_tasktime.sh` OpenCode-launcher timing harness, unmodified.

All files above are copied byte-for-byte from the upstream commit above
("unmodified" — no vendoring patches applied). This repository's own endpoint
layer (`scripts/fidelity/tasks/`) imports these modules and calls their public
functions/`TASKS`/`PROMPTS` without editing them.

## Licence

Verified against the upstream `LICENSE` and `NOTICE`: the upstream top-level
`LICENSE` is an MIT grant (Copyright (c) 2026 knapcio) that "applies to
repository-owned material except where a file specifies another licence."
`NOTICE` enumerates every non-MIT exception (vLLM/SGLang-derived overlay
files, the Flash Linear Attention kernel, the tonyd2wild patch with no
declared licence, the DeepSeek/AGPL-derived CUDA transfer, RoCEnante/b12x,
and model weights) — none of `bench/qeval.py`, `bench/qeval_tasks.py`,
`bench/hardset.py` or `bench/tasktime/**` appear in that list, and none of
those files carry a per-file SPDX header, copyright notice, or licence
statement of their own. They are therefore repository-owned MIT material
under the top-level grant, and are vendored here under the MIT `LICENSE`
text in this directory (same text as upstream's top-level `LICENSE`,
scoped to the files listed above).

`bench/bench_matrix.py`, `bench/compare_kld_strict.py`, `bench/compare_time.py`,
`bench/conc_bench.py`, `bench/final_sparkdash.py`, `bench/kld_probe.py`,
`bench/prefill_checked.py`, `bench/run_prefill.py` and the `bench/test_*.py`
files were left un-vendored: they are out of scope for this task-set/tasktime
work (fidelity/KLD/prefill tooling, not the qeval/hardset/tasktime task sets).
