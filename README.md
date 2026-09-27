# GLM-5.3-Flash FP8 on four NVIDIA GB10 nodes (switchless)

[![Follow me on X](https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/jnardiello)

Run GLM-5.3-Flash FP8 with vLLM across four NVIDIA GB10 systems, connected directly
through a switchless ConnectX-7 RoCE ring. This is my daily driver for coding and
parallel agents, with a **256K context window (262,144 tokens)**. The repository
contains the infrastructure as code, runtime patches, and guides to
[install it with an agent](#install-with-an-agent) on compatible hardware.

## Measured performance

Current accepted baseline: **25/09/2026 · E29, no speculative step past a length finish**,
measured with native [Rigmark](https://github.com/alexellis/rigmark). Frozen medians of
**three complete runs: 162/162 requests**, zero measurement/runtime errors and 45/45 native
output gates passing.

| Workload | Current · 25/09/2026 E29 | vs previous [📊 25/09/2026 E28b](docs/benchmarks/baselines/2026-09-25-e28b.md) |
| --- | ---: | ---: |
| Code decode, one request | 62.95 tok/s | 61.01 tok/s · +3.18% |
| Code C1, end-to-end | 43.92 tok/s | 43.18 tok/s · ≈ unchanged (+1.71%) |
| Code C2, aggregate end-to-end | 67.52 tok/s | 67.47 tok/s · ≈ unchanged (+0.07%) |
| Code C4, aggregate end-to-end | 97.35 tok/s | 95.54 tok/s · ≈ unchanged (+1.89%) |
| Prose decode | 34.13 tok/s | 33.28 tok/s · +2.54% |
| Code TTFT | 0.386 s | 0.390 s · ≈ unchanged (-1.03%) |
| Prose TTFT | 0.376 s | 0.376 s · ≈ unchanged (+0.00%) |
| C1 per-stream TTFT | 0.342 s | 0.394 s · -13.20% |
| C2 per-stream TTFT | 0.394 s | 0.462 s · -14.72% |
| C4 per-stream TTFT | 0.472 s | 0.523 s · -9.75% |
| Prefill 8K, cold | 2,615.9 tok/s | 2,558.4 tok/s · +2.25% |
| Prefill 8K, replay | 8,982.1 tok/s | 8,670.6 tok/s · +3.59% |
| Prefill 32K, cold | 2,764.8 tok/s | 2,739.3 tok/s · ≈ unchanged (+0.93%) |
| Prefill 32K, replay | 34,155.5 tok/s | 33,262.0 tok/s · +2.69% |
| Prefill 64K, cold | 2,682.0 tok/s | 2,692.7 tok/s · ≈ unchanged (-0.40%) |
| Prefill 64K, replay | 38,880.9 tok/s | 39,291.0 tok/s · ≈ unchanged (-1.04%) |

Both baselines were measured over the same direct LAN client path. Percentages use
unrounded values. “≈ unchanged” marks owner-accepted changes of roughly 1–2%, with the
exact delta retained; it does not establish statistical equivalence.
Higher throughput and lower TTFT are better. C1/C2/C4 mean one, two or four concurrent
requests. Decode excludes the initial wait; end-to-end speed includes it. TTFT is time
to first token. Concurrency outputs cap at 256 tokens; long decode allows 8,192.
Cold prefill uses a fresh cache salt per run. These are inference measurements,
not complete agent-task timings.

**What E29 changes.** With asynchronous scheduling the engine queues a request's next step
before the output of the step in flight has come back. With E28b's seven draft tokens, a
request limited by `max_tokens` usually finished inside a step that could produce several
tokens, so the step queued behind it verified drafts for a finished request. A request
arriving right after it waited for that step, which cost E28b about 50 ms of first-token
time.

- **Hold near the length limit:** the scheduler does not queue another step while the step
  in flight may finish the request. First-token time returns to 0.342 s with one request
  (C1) and improves to 0.394 s with two (C2) and 0.472 s with four (C4).
- **Idle coalescing:** at an idle engine, requests arriving within 4 ms are prefilled in one
  step, so agents released together start together.
- **Kept from E28b:** decode is unchanged or slightly faster. Neither change alters the
  steps in between, the seven draft tokens or the 16 GiB KV pool.
- **Remaining cost:** a cached replay sent right after its cold request waits for the cache
  connector to publish that request. At 8K and 32K it stays 56–82 ms slower than on E27c.

The [current benchmark report](docs/benchmarks/baselines/2026-09-25-e29.md) lists per-run
values, the diagnosis, the prefill re-runs, the memory samples and limitations.

The graphs compare current and previous medians side by side, with each delta calculated
against the previous September 25 E28b baseline. Click an image for its SVG version.

[![Current E29 versus previous E28b baseline: generation throughput, time to first token and percentage changes.](docs/plots/comparisons/2026-09-25-e29-vs-2026-09-25-e28b/generation.png)](docs/plots/comparisons/2026-09-25-e29-vs-2026-09-25-e28b/generation.svg)

[![Current E29 versus previous E28b baseline: cold prefill, immediate replay and percentage changes at 8K, 32K and 64K.](docs/plots/comparisons/2026-09-25-e29-vs-2026-09-25-e28b/prefill.png)](docs/plots/comparisons/2026-09-25-e29-vs-2026-09-25-e28b/prefill.svg)

The [benchmark archive](docs/benchmarks/README.md) retains earlier baselines, experiments
and separate reproduction results. See [local Rigmark reports](docs/rigmark_reports/README.md)
for saving and viewing native receipts.

The [current recipe](docs/production-recipe.md), encoded in
[`cluster.env.example`](cluster.env.example), combines the digest-pinned SparkRing image,
DFlash2 with adaptive verification capped by its effective draft budget, E03 mHC prefill
sharding, hybrid INT8/BF16 KDA input projections, E21 8-bit weights for the KDA output
and MLA attention projections, E22b 8-bit weights for the DFlash2 drafter, the E27 prefill
cadence with the E27c scheduler, seven draft tokens (E28b), the E29 length-finish hold and
idle coalescing, SparkCache replay views and SIRCL/patched NCCL.
The **16 GiB KV pool per rank** (E28b) and the **262,144-token context limit** apply.

## Quality vs vendor FP8

**Measured 27/09/2026 on 2.5M teacher-forced tokens: 424 windows of coding-agent sessions,
synthetic Italian chats and model-written text.** The reference is the vendor FP8 weights
served on the same four nodes with an FP8 KV cache. E29 does not match it bit for bit,
but shows no detectable quality loss: its dense, sparse and all-position perplexity
intervals all include zero. A public NVFP4 recipe is about twice as far from FP8 on
short contexts (single-execution coarse KL). It is slightly better there, but clearly
worse overall and beyond 2K tokens.

| vs vendor FP8 | E29 | NVFP4 |
| --- | ---: | ---: |
| Same next token, ≤ 2K context | 83.4% | 75.1% |
| KL divergence, ≤ 2K context (nats) | 0.18 | 0.35 |
| Perplexity change, ≤ 2K context | +0.03% [−0.56, +0.60] | −3.33% [−5.43, −1.22] |
| Perplexity change, all tokens | +0.14% [−0.32, +0.56] | +7.07% [+4.61, +9.37] |
| Perplexity change, > 2K context | +0.19% [−0.34, +0.73] | +12.05% [+8.30, +15.62] |
| Perplexity change, agentic code | +0.35% [−0.39, +1.11] | +9.30% [+6.76, +12.08] |
| Perplexity change, synthetic Italian chat | −0.04% [−0.19, +0.12] | +10.05% [+9.32, +10.82] |
| qeval tasks, mixed (FP8: 73/75) | 73/75 | 74/75 |
| Italian replies with garbled characters | 0/40 | 3/40 |

- **Perplexity change** is exp(ΔNLL) − 1 on the token that actually came next, with 95%
  bootstrap intervals over source sessions. Lower is better; an interval that contains 0
  means no detectable change.
- **KL divergence** covers the shared top-20 tokens, the actual token and a rest bucket,
  so it is a lower bound. Most recipe changes measured here produce a dense KL of this
  size; the FP8 recipe before E21 also sits at 0.18.
- **Beyond 2,048 tokens** the engine is not deterministic, even for the FP8 reference, so
  the fine-grained distance there remains unresolved. No quality change is detected there.
- **The garbled-text probe** ran on E29 and NVFP4 only. The 75 qeval tasks mix code,
  reasoning, maths, JSON, formatting and prose. At this size they show no difference, but
  they cannot prove equivalence.

[![Different vs worse: KL distance from vendor FP8 against perplexity change for E29, the earlier recipe and NVFP4, with 95% intervals.](docs/fidelity/plots/02-different-vs-worse.png)](docs/fidelity/plots/02-different-vs-worse.svg)

[![Quality vs FP8: perplexity change of each recipe against vendor FP8 by context regime and corpus category, with 95% intervals.](docs/fidelity/plots/01-quality-vs-fp8.png)](docs/fidelity/plots/01-quality-vs-fp8.svg)

### E29 vs NVFP4

The NVFP4 arm reproduces the engine layer of
[Alex Ellis's four-node recipe](https://github.com/alexellis/glm-5.3-flash-4x-dgx-spark-switchless/tree/e2d0839a92f118a0a874abcfc1fe8012ee69978e)
on this fabric: the `LibertAIDAI/GLM-5.3-Flash-NVFP4` checkpoint with 4-bit expert weights,
running on its own engine build. It measures that checkpoint-plus-engine combination as
published, not NVFP4 as a format. Measured head to head on the same corpus, **E29 is the
more precise recipe**.

| NVFP4 measured against E29 | Result |
| --- | ---: |
| Same next token, ≤ 2K context | 75.2% [73.7, 76.7] |
| KL divergence, ≤ 2K context (nats) | 0.35 [0.32, 0.38] |
| Perplexity change, ≤ 2K context | −3.36% [−5.39, −1.31] |
| Perplexity change, > 2K context | +11.84% [+7.94, +15.45] |
| Perplexity change, all tokens | +6.93% [+4.49, +9.22] |
| Perplexity change, agentic code | +8.91% [+6.31, +11.73] |
| Perplexity change, synthetic Italian chat | +10.09% [+9.35, +10.88] |
| Repeat runs, ≤ 2K context (rows that differ) | E29 0 of 155,390 · NVFP4 155,206 of 155,390 |
| qeval tasks, mixed | E29 73/75 · NVFP4 74/75 (McNemar p = 1.00) |
| Italian replies with garbled characters | E29 0/40 · NVFP4 3/40 |
| Tool calls parsed | E29 10/10 · NVFP4 10/10 |

- **Short contexts:** NVFP4 predicts real text about 3% better than E29, a result that is
  present in the data but unexplained.
- **Beyond 2K tokens:** the difference reverses and grows to about 12%, and NVFP4 is
  clearly worse on agentic code and Italian.
- **Determinism:** E29 repeats itself bit for bit on short contexts, while NVFP4 differs
  from its own previous run almost everywhere.
- **Tasks and tool calls:** no difference at this sample size.
- **Garbled characters:** these match a known issue with ModelOpt NVFP4 checkpoints on vLLM
  (issue 54150). With so few prompts, 3/40 against 0/40 is a signal to watch rather than
  a statistically significant difference.

The [fidelity report](docs/fidelity/REPORT.md) opens with a
[plain-language summary](docs/fidelity/REPORT.md#in-plain-words). It also covers the method,
the recipe ladder that attributes the difference to individual steps, and the limitations.
[`report.html`](docs/fidelity/report.html) is a self-contained version with every figure.
The [experiment archive](docs/historical_benchmarks/experiments/2026-09-27-fidelity/README.md)
holds the portable data.

## Install with an agent

Start with a local checkout and an agent that can read its files and use SSH. The
verified hardware is ASUS Ascent GX10; the agent must inventory your actual four
nodes before creating the site configuration.

| Prerequisite | What you need |
| --- | --- |
| Nodes | Four DGX Spark-class systems, one NVIDIA GB10 and 128 GB unified memory each |
| Fabric | Two usable ConnectX-7/RoCE ports per node; four DACs in the ring 0 ↔ 1 ↔ 2 ↔ 3 ↔ 0; verified at 200 Gb/s and MTU 9000 |
| Hosts and access | Ubuntu, NVIDIA driver, Docker with GPU support, `rdma-core`, and SSH access over a trusted management LAN/VPN |
| Storage | At least 330 GiB free per node for a fresh model fetch, plus image and runtime-cache space |
| Pinned artifacts | Image, target weights, drafter, and patched NCCL from the [installation procedure](docs/install-from-zero.md) |
| Third-party payload | Included: the [SparkCache](https://github.com/FujitsuPolycom/sparkcache) connector and encoder and the [SparkRing SIRCL](https://github.com/FujitsuPolycom/sparkring) bundle and runtime, both Apache-2.0, under [`third_party/`](third_party/); generate the SIRCL site files as in [payload preparation](docs/install-from-zero.md#8-prepare-the-sparkcache-and-sircl-payload) |

[Third-party payload](docs/third-party.md) lists every included file: where it comes
from, which bytes are upstream and what this project changed. DFlash2 carries
non-commercial terms; review [credits and licenses](CREDITS.md). The API has no
authentication or TLS, so keep it on a trusted network or behind an authenticating
proxy.

Replace the placeholders below, then give this prompt to the agent in the checkout:

```text
Install this repository's accepted September 25, 2026 E29 recipe on my four nodes.
Read AGENTS.md, docs/install-from-zero.md, and docs/operations.md first.
Use docs/historical_benchmarks/baselines/2026-09-25-e29/baseline.json and cluster.env.example as the reference.

SSH targets in rank order:
0: <user@rank0-host>
1: <user@rank1-host>
2: <user@rank2-host>
3: <user@rank3-host>
Authorized maintenance window and actions: <describe the agreed scope>

Run the read-only preflight on all four targets and use their actual hardware
and network mappings. Confirm the cable map and keep site values in ignored
cluster.env. Follow the documented artifact preparation, deployment, startup,
functional gates, and identity checks. Preserve the pinned recipe and hashes.
If a required artifact is missing, report exactly what I must supply.
Continue actions already authorized without asking again at each tool call;
ask only about missing inputs or actions outside that scope.
Do not run Rigmark or replace frozen measurements unless I request it.
```

The [agent contract](AGENTS.md#fresh-checkout-reproduction-contract) supplies the
short execution checklist. The guides below own the complete procedures.

## Documentation

| Goal | Guide |
| --- | --- |
| Give an agent the repository contract | [Agent entry point](AGENTS.md) |
| Install on prepared hosts | [Install from zero](docs/install-from-zero.md) |
| Inspect, deploy, start, stop, recover, or roll back | [Operations](docs/operations.md) |
| Cable, configure, or diagnose the RoCE ring | [Fabric](docs/fabric.md) |
| Understand the current components and customizations | [Production recipe](docs/production-recipe.md) |
| Understand files installed on the nodes | [Node assets](scripts/node/README.md) |
| Check host/software pins and bootstrap | [Bootstrap pins](scripts/node/bootstrap/README.md) |
| Manage host IOMMU configuration | [Host controls](scripts/node/host/README.md) |
| Understand container patches | [Patch guide](scripts/node/patches/README.md) |
| Generate site network files | [Netplan renderer](scripts/render-netplan.md) |
| Build and install patched NCCL | [NCCL guide](scripts/node/nccl/README.md) |
| Maintain workstation scripts | [Shared shell helpers](scripts/lib/README.md) |
| Review benchmark results and history | [Benchmark reports](docs/benchmarks/README.md), [native report storage](docs/rigmark_reports/README.md) |
| Check output quality against vendor FP8 | [Fidelity report](docs/fidelity/REPORT.md), [fidelity tooling](scripts/fidelity/README.md) |
| Review changes and third-party terms | [Changelog](CHANGELOG.md), [credits](CREDITS.md), [license](LICENSE) |

## Use the endpoint

Once the cluster has passed readiness and its post-boot gates, the OpenAI-compatible
API is available on rank 0. From a client inside the trusted network:

```sh
curl http://<MGMT_IP_RANK0>:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "glm-5.3-flash",
    "temperature": 0,
    "max_tokens": 64,
    "chat_template_kwargs": {"enable_thinking": false},
    "messages": [{"role": "user", "content": "Reply with READY."}]
  }'
```

`enable_thinking=false` selects the local chat-template adapter. Its behavior and the
model's reasoning controls are explained in
[Thinking-off compatibility](docs/production-recipe.md#thinking-off-compatibility).

## Development checks

Install Python 3.9+ and `Jinja2==3.1.6`, then run the offline check:

```sh
python3 -m pip install 'Jinja2==3.1.6'
./scripts/check.sh
```

The check requires no GPU, Docker daemon, SSH connection, site configuration, or model
weights. It validates syntax, public documentation links, command help, manifests,
chat-template rendering, host and controller lifecycle fixtures, preflight, and the
adaptive-k policy.
