# E03: FP8 mHC prefill sharding

This accepted component ports the Apache-2.0 SparkRing mHC package at
[`61f277bd`](https://github.com/FujitsuPolycom/sparkring/tree/61f277bd0c97fbff892668e12ea04a330a45fa01/runtime/glm53-spark-mtp3-mesh/performance/mhc-prefill)
onto the September 19 R10 FP8 recipe. The default IaC now combines it with replay views and the draft-budget scheduler;
[deployment/reproduction status](../../../../docs/historical_benchmarks/baselines/2026-09-19-e03/promotion.json)
is recorded separately from its accepted measurements.
Admission uses 6,912 target rows: R10 reserves five DFlash2 slots from its unchanged
8,192-token budget, then aligns full chunks to 2,304-token checkpoints. The initial
8,192-row guard could not activate; its evidence remains in the
[experiment report](../../../../docs/benchmarks/experiments/2026-09-19-e03.md).
Serving activation must be verified separately from model-free kernel checks.
`manifest.json` records source hashes and `source.patch` shows the delta against
the actual R10 modules. The [license and notice](../../../../third_party/sparkring-mhc/NOTICE)
are retained. FP8 checkpoint mapping, E20 projections, cache fixes, DFlash2,
scheduler, CUDA graphs, 15 GiB KV and maximum context remain unchanged.

The E21 defaults keep `SPARK_MHC_PREFILL_SHARD=1` and the nine E03 module mounts. The
promoted [E21 layer](bf16-residue/README.md) mounts the KDA hook from `bf16-residue/`, adds
its residual-projection module and selects its own cache namespace; the complete E03
recipe, with the measured E03 cache namespace, is
`scripts/node/reference/baseline-20260919-e03.env`. Do not reuse entries from the previous base.
The source manifest and historical `candidate.env` remain unchanged. To reconstruct
the original E03-only experiment on the new defaults, prepend the complete
`scripts/node/reference/baseline-20260919.env` to `candidate.env` in one private
standalone overlay. Frozen overlay comments refer to the earlier base. Current default
installation uses no experiment overlay and follows the installation/operations guides.

Only pure eager prefills with exactly 6,912 BF16 rows use the path. The first mHC
pre remains full-sized; subsequent mHC operations own 1,728 contiguous rows per
rank. Attention and FFN still consume all rows. Per-call `defer_tp_reduction`
leaves one TP partial for reduce-scatter; all-gather restores each consumer's
input. Final outputs and every DFlash2 auxiliary capture return in full order.
TP4/DCP1 is required; global MoE sequence/expert parallelism stays disabled.
Decode, mixed batches, shorter tails and CUDA graphs use the ordinary path.

## Checks and coordinated execution

Run `./scripts/check.sh` from the repository root. Run the additional CPU Torch
tests with `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python3
scripts/tests/test-e03-mhc-runtime.py` in an environment containing Torch, such as
the pinned image. These tests execute the candidate's actual methods with CPU
collectives and cover DFlash2 outputs, reduction counts and the next ordinary call.
They do not load weights or create a CUDA context.

Within one authorized window, use
`TP4_ENV=<private-standalone-E03-overlay>` consistently with the existing
launcher, deploy and controller. Preview all four launchers with `TP4_DRY_RUN=1`,
stage using `scripts/deploy.sh`, and verify the deployed hashes. Prepare the base
return before stopping. Complete the coordinated `down`, then start the following
on all four nodes concurrently, replacing `<rank>` with 0 through 3:

```sh
python3 "$HOME/tp4/experiments/e03/gpu-launch.py" <rank>
```

This derives networking, NCCL and mounts from the native launcher, refuses an active
serving container or GPU workload, and launches a temporary model-free check container.
The check uses TCP port 29603 for rendezvous and the existing TP PyNccl transport.
Require four `E03_GPU_CHECK` PASS records before `up`. It reports absolute and
relative errors for BF16 collectives and B12X full/sharded pre, post/pre and post,
including DFlash2 contraction. Constructed sums and row order must match exactly;
random BF16 comparisons allow reduction-order rounding (`atol=0.03125`, `rtol=0.02`).

After `up`, complete both [functional gates](../../../../docs/operations.md#post-boot-functional-gates)
within two minutes of health 200. Verify source hashes and the E20 boot signatures.
Ordinary long prefills must log `SPARK_MHC_PREFILL` on all four ranks with
`rows=6912 owner_rows=1728 rs=90 ag=95 aux=5`; `aux` counts DFlash2 auxiliary
all-gathers. Logging is limited to the first eight eligible forwards per process.
For the historical E03-only experiment, the current identity checker rejects the
missing replay/budget components. Keep the old records unchanged; the E03 rollback is
checked against the E03/replay/draft-budget reference and the current defaults against E21.

Use the [native benchmark procedure](../../../../AGENTS.md#run-the-benchmark) with
the September 19 settings and fresh salts. Keep the three complete 54-request
executions and their actual counts separate from the frozen two-run baseline.
No extra answer-quality audit is part of this experiment.

The separately prepared [replay views follow-up](replay-views/README.md) retains
E03 and reduces temporary copies during Python cache restore. Its checks and
native results are recorded separately from this measured E03 candidate.

## Return to the measured base

In an authorized return window, `down` with the candidate overlay, then deploy and
`up` with `TP4_ENV=scripts/node/reference/baseline-20260919.env`. This selects the original modules and cache; no weights,
cache directories or image are removed. Complete both gates and `check-f0.py --baseline docs/historical_benchmarks/baselines/2026-09-19/baseline.json`.
Do not reload the base automatically after a failed check or benchmark.
