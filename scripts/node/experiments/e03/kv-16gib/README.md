# E28b: 16 GiB KV pool

**Promoted on September 25, 2026.** The default `cluster.env.example` encodes it; this overlay
applied on the E28 recipe (E27c plus the draft-depth-7 overlay) is the record of the measured
candidate. See the [E28b reference](../../../../../docs/benchmarks/baselines/2026-09-25-e28b.md).

E28 drafts seven tokens for a single request, which leaves 4.9% fewer KV tokens in the 15 GiB
pool (1,278,751, or 4.88 full 262,144-token contexts). E28b raises the pool to 16 GiB per rank
so that five full contexts fit together. Nothing else changes.

```sh
TP4_ENV=scripts/node/experiments/e03/kv-16gib/delta.env ./scripts/deploy.sh
TP4_ENV=scripts/node/experiments/e03/kv-16gib/delta.env ./scripts/tp4ctl up
```

The earlier 16 GiB failure (September 19) ran six parallel agents with a larger memory
footprint; check free memory on rank 0 after boot and under load.
