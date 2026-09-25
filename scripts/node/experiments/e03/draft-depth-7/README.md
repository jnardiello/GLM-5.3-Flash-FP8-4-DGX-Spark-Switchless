# E28: seven draft tokens for a single request

The DFlash2 drafter is trained with blocks of 8 positions, so it can propose up to seven
tokens per verification step. The E27c recipe uses five for one request and three for
batches of 2–6. An audit at one request, with Rigmark's prompts, measured the fraction of
steps that accepted each draft position:

| Workload | Tokens per step | Position 0 / 1 / 2 / 3 / 4 |
| --- | ---: | --- |
| Code | 4.1–4.8 | 0.87–0.94 / 0.74–0.87 / 0.64–0.79 / 0.46–0.66 / 0.39–0.58 |
| Structured | 5.9 | 0.99 / 0.99 / 0.97 / 0.97 / 0.96 |
| Prose (adaptive low state, three tokens) | 1.95 | 0.57–0.60 / 0.26 / 0.10–0.11 |

Code and structured output still accept the fifth position often, so two more positions can
add tokens per step. Verifying two more tokens costs a little more per step.

**Status: measured, then promoted inside E28b** with a 16 GiB KV pool. The default
`cluster.env.example` encodes it, so this overlay now refuses the default; apply it on
`scripts/node/reference/baseline-20260925-e27c.env` to reproduce the measured candidate.
See the [E28 report](../../../../../docs/benchmarks/experiments/2026-09-25-e28-draft-depth-7.md).

## Change

The overlay applies on the default E27c recipe and changes four things:

- `SPEC_TOKENS=7`;
- the per-batch table becomes `[[1,1,7],[2,6,3]]`, so only a single request may draft seven
  tokens and batches of 2–6 keep three;
- `VLLM_ADAPTIVE_K_HI=7`: the adaptive high state becomes seven, while the low state stays
  three, so prose keeps dropping to three;
- `--compilation-config={"max_cudagraph_capture_size":72}` keeps the captured CUDA graph set
  of E27c. Without it vLLM would raise the limit to 96 and capture more graphs, and rank 0
  has little free memory.

Weights, the drafter and the SparkCache namespace are unchanged. Verification stays exact,
so answers are the same as with five draft tokens, apart from the usual effects of different
batching.

## Use

```sh
python3 scripts/tests/test-draft-depth-7-config.py
./scripts/check.sh
```

A measurement window uses one overlay for every command, following
[docs/operations.md](../../../../../docs/operations.md):

```sh
TP4_ENV=<serving overlay or none> ./scripts/tp4ctl down
TP4_ENV=scripts/node/experiments/e03/draft-depth-7/delta.env ./scripts/deploy.sh
TP4_ENV=scripts/node/experiments/e03/draft-depth-7/delta.env ./scripts/tp4ctl up
```

At startup the adaptive scheduler logs `k_hi=7`, the engine reports `num_spec_tokens=7`, and
the capture sizes end at 72. To return, stop with the same overlay, then deploy and start with
no `TP4_ENV`.
