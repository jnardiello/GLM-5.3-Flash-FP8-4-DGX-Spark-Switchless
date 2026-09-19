# E03 replay views connector

This component is selected by the accepted defaults. Its measured payload remains
operator-supplied; [IaC reproduction status](../../../../../docs/historical_benchmarks/baselines/2026-09-19-e03/promotion.json)
is separate from the archived experiments.

This follow-up preserves the measured E03 computation and cache format. It replaces
one whole-snapshot materialization in the Python restore path with memory views of
the already authenticated, validated page spans. The existing writable copy for
each layer, physical block placement and final stream synchronization remain intact.
The encoded owner and all views stay alive until the method returns.

The candidate passed compact CPU and four-rank GPU placement checks. It does not
establish the cause of E03's replay latency difference. Read the
[E03 report](../../../../../docs/benchmarks/experiments/2026-09-19-e03.md#replay-follow-up)
for the saved-log analysis and CPU evidence. The original codec API, store path,
integrity checks, KV pool, scheduler and E03 mHC configuration are preserved.
Native performance measurements are kept in the separate
[replay candidate report](../../../../../docs/benchmarks/experiments/2026-09-19-e03-replay-views.md).

## Prepare and check

The connector and encoder are operator-supplied payload under the terms in
[CREDITS](../../../../../CREDITS.md). Do not redistribute the generated connector.
`prepare.py` requires the current connector hash and verifies its exact output hash;
it refuses to overwrite an existing file. From the repository root:

```sh
umask 077
python3 scripts/node/experiments/e03/replay-views/prepare.py \
  --connector <private-current-connector.py> \
  --output <new-private-connector.py>
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
  python3 scripts/node/experiments/e03/replay-views/check.py \
  --connector <private-current-connector.py> \
  --encoder <private-current-encoder.py>
```

The CPU check requires Torch. It runs the actual old/new restore methods with a
small synthetic snapshot and heterogeneous page tables, checking exact bytes,
non-monotonic physical slots, untouched blocks, buffer lifetime through completion
and rejection of damaged framing. It asserts that no CUDA context was created.
The store is a fixture; this check does not measure disk I/O or model latency.

The default recipe already selects this connector. To reconstruct the historical
E03 plus replay experiment, concatenate the complete previous-base
`scripts/node/reference/baseline-20260919.env`, E03's `candidate.env` and this directory's
`delta.env` into one ignored private overlay. Frozen overlay comments describe the
previous base; do not append them directly to the accepted defaults.
Use that same `TP4_ENV` for launcher previews, deploy and all lifecycle commands.
The only launcher argument change from E03 is the connector mount. Keep E03's cache
namespace because model computation and encoded bytes are identical.

Within a newly authorized coordinated window, place the prepared connector at the
path selected by `delta.env` on all four nodes, without replacing the current file.
Verify its hash before stopping. After the full stop, the same check supports
`--device cuda` inside the pinned image without model weights. Retain its JSON receipt.
Then use the existing controller to load the candidate once, complete both
[functional gates](../../../../../docs/operations.md#post-boot-functional-gates),
verify all four connector hashes, and run the native Rigmark comparison separately
from E03 and the frozen baseline. No reload or inference is performed by these tools.

To return to measured E03 in an authorized window, stop with the follow-up overlay
and deploy/start with a private standalone overlay containing the September 19 rollback
followed by `scripts/node/experiments/e03/candidate.env`. Both connector
files and the existing E03 cache are retained. Do not restore or remeasure the
frozen baseline automatically.
