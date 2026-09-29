# GLM-5.3-Flash FP8 on four NVIDIA GB10 nodes (switchless)

[![Follow me on X](https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/jnardiello)

Run GLM-5.3-Flash FP8 with vLLM across four NVIDIA GB10 systems, connected directly
through a switchless ConnectX-7 RoCE ring. This is my daily driver for coding and
parallel agents, with a **256K context window (262,144 tokens)**. The repository
contains the infrastructure as code, runtime patches, and guides to
[install it with an agent](#install-with-an-agent) on compatible hardware.

## Measured performance

Current accepted baseline: **28/09/2026 · E31, speculative-safe C4 tail ring**, measured
with upstream, unmodified [Rigmark](https://github.com/alexellis/rigmark) and the reference
flags of its GLM receipts: every default plus `reasoning_effort` low. **One complete suite
(54/54 requests, n = 1)**, zero measurement/runtime errors and 15/15 native output gates
passing. The comparison is an arm with the E29 arithmetic (legacy tail), measured on the same
load with the same flags.

The operational default keeps the E31 model, context and scheduler lineage, adds bounded
SparkCache disk transfers, and limits aggregate request admission. It uses a 14 GiB KV pool,
allocator trim before eligible eager prefills, a 6,912-token scheduling-step cap, six active
API admission slots and 128 queued requests. Admission slots do not imply resident engine
requests. This safety configuration has its own
[versioned operational identity](docs/operational-identities/2026-09-29-memory-bounded.json);
it does not replace or alter the frozen E31 performance baseline. The complete
[protected 16 GiB rollback](scripts/node/reference/operational-20260929-sparkcache-protected.env)
removes the allocator, step-cap and admission deltas while keeping cache protection. The
[E31 rollback](scripts/node/reference/baseline-20260928-e31.env) also restores the measured
cache behavior.

| Workload | Current · 28/09/2026 E31 | vs same-load E29-equivalent arm ([📊 E31 report](docs/benchmarks/baselines/2026-09-28-e31.md)) |
| --- | ---: | ---: |
| Code decode, one request | 65.19 tok/s | 64.56 tok/s · +0.97% |
| Code C1, end-to-end | 43.99 tok/s | 40.99 tok/s · +7.32% |
| Code C2, aggregate end-to-end | 68.37 tok/s | 73.10 tok/s · -6.48% |
| Code C4, aggregate end-to-end | 101.45 tok/s | 108.67 tok/s · -6.65% |
| Prose decode | 34.09 tok/s | 34.28 tok/s · -0.54% |
| Code TTFT | 0.475 s | 0.473 s · +0.42% |
| Prose TTFT | 0.374 s | 0.385 s · -2.86% |
| C1 per-stream TTFT | 0.421 s | 0.415 s · +1.45% |
| C2 per-stream TTFT | 0.469 s | 0.457 s · +2.63% |
| C4 per-stream TTFT | 0.548 s | 0.555 s · -1.26% |
| Prefill 8K, cold | 2,572.5 tok/s | 2,600.2 tok/s · -1.07% |
| Prefill 8K, replay | 8,848.1 tok/s | 8,708.4 tok/s · +1.60% |
| Prefill 32K, cold | 2,697.9 tok/s | 2,766.0 tok/s · -2.46% |
| Prefill 32K, replay | 33,944.4 tok/s | 32,797.3 tok/s · +3.50% |
| Prefill 64K, cold | 2,614.4 tok/s | 2,595.3 tok/s · +0.74% |
| Prefill 64K, replay | 39,576.6 tok/s | 40,326.5 tok/s · -1.86% |

Each value comes from a single suite, so differences of a few percent are within noise. The
C1/C2/C4 rows rest on three short rounds each and moved about ±7% in both directions.
Percentages use unrounded values. Higher throughput and lower TTFT are better.

- **Workloads.** C1/C2/C4 mean one, two or four concurrent requests. Decode excludes the
  initial wait; end-to-end speed includes it. TTFT is time to first token.
- **Output limits.** Concurrency outputs cap at 256 tokens; long decode allows 4,096.
- **Cache isolation.** Each suite has a fresh comparison ID, which isolates the prefix cache.
- **Scope.** These are inference measurements, not complete agent-task timings.

**The numbers above are not comparable with earlier ones.** Up to September 25 (E29), this
README showed three-suite medians measured with a local Rigmark fork, 8,192-token decode,
thinking off and a cache salt. Those records remain in the
[benchmark archive](docs/benchmarks/README.md) under their own protocol.

**What E31 changes.** The pooled indexer builds one key from every four tokens. It keeps each
request's recent rows in a four-slot tail so a pool can be completed across steps. A DFlash2
verify step writes eight consecutive positions, including drafts that may be rejected.
Their rows overwrote committed members of the pool still being built, and after a rejection
that pool was completed from rejected-draft keys.

- **Fix:** the tail becomes a ring of 12 slots for seven drafts, so no row of one step can
  reach a committed member of the open pool. A GPU test with the real kernels found 200
  wrong pools with the old tail and none with the ring. vLLM merged the equivalent fix
  upstream on September 25.
- **Cost:** none measurable. Code and prose decode move less than 1% against the same-load
  E29-equivalent arm.
- **Head gate on tensor cores:** included but off. Its same-load comparison is unresolved.
- **Cache:** the SparkCache namespace was kept. Pools built before E31 could carry the old
  defect if they were persisted and restored.

The [current benchmark report](docs/benchmarks/baselines/2026-09-28-e31.md) lists all three
arms, the leaf tests, the excluded series and limitations.

The graphs compare E31 with the E29-equivalent arm measured on the same load with the same
flags, the values of the table above. Click an image for its SVG version.

[![E31 versus the same-load E29-equivalent arm, upstream Rigmark with Alex Ellis's reference flags: generation throughput, time to first token and percentage changes.](docs/plots/comparisons/2026-09-28-e31-vs-e29-same-load/generation.png)](docs/plots/comparisons/2026-09-28-e31-vs-e29-same-load/generation.svg)

[![E31 versus the same-load E29-equivalent arm, upstream Rigmark with Alex Ellis's reference flags: cold prefill, immediate replay and percentage changes at 8K, 32K and 64K.](docs/plots/comparisons/2026-09-28-e31-vs-e29-same-load/prefill.png)](docs/plots/comparisons/2026-09-28-e31-vs-e29-same-load/prefill.svg)

The [benchmark archive](docs/benchmarks/README.md) retains earlier baselines, experiments
and separate reproduction results. See [local Rigmark reports](docs/rigmark_reports/README.md)
for saving and viewing native receipts.

The [current recipe](docs/production-recipe.md), encoded in
[`cluster.env.example`](cluster.env.example), combines the digest-pinned SparkRing image,
DFlash2 with adaptive verification capped by its effective draft budget, E03 mHC prefill
sharding, hybrid INT8/BF16 KDA input projections, E21 8-bit weights for the KDA output
and MLA attention projections, E22b 8-bit weights for the DFlash2 drafter, the E27 prefill
cadence with the E27c scheduler, seven draft tokens (E28b), the E29 length-finish hold and
idle coalescing, the E31 speculative-safe indexer tail, SparkCache replay views and
SIRCL/patched NCCL.
The operational **14 GiB KV pool per rank** is 12.5% smaller than E31's measured 16 GiB
pool; the **262,144-token context limit** is unchanged. Startup reports 1,194,033 KV tokens
(4.55 full contexts), so five complete 256K contexts cannot be resident at once; scheduling
must use waiting or preemption for that five-client workload.

## Measured quality

**Measured 27/09/2026.** Does the current recipe answer as well as the official model it
is built from? We gave the same 2.5 million tokens of text to three setups on these four
nodes: the current recipe, the vendor's FP8 model served without this repository's
changes, and a public NVFP4 recipe. The text is coding-agent sessions, synthetic Italian
chats and model-written answers. At every token we recorded two things: which token each
model would pick next, and how surprised it was by the token that actually followed.

**How to read the numbers**

- **Token and context.** A token is a piece of a word, and 2K tokens are roughly 1,500
  words of conversation. "First 2K tokens" covers short conversations; "beyond 2K" covers
  the rest of longer ones.
- **Same first choice.** How often two models would write the same next token. Below
  100% is not a problem in itself: often several tokens are equally good, like two good
  writers choosing different synonyms.
- **Distance (KL).** How different the whole list of likely next tokens is. 0 means
  identical, and larger means more different. It measures difference, not quality, and
  it can only underestimate the true difference.
- **Perplexity change.** The main quality number: how much more (positive) or less
  (negative) surprised a model is by real text than the vendor model is. 0% means
  equally good, and lower is better. The brackets give the range where the true value
  very likely lies (95% confidence). If that range includes 0, no difference could be
  measured.
- **Tasks passed.** 75 small, automatically graded exercises: code, reasoning, maths,
  JSON, formatting and prose. With this many, only differences larger than about 7
  points would show up.
- **Same output when run twice.** Whether the same input produces exactly the same
  predictions again.
- **Broken characters.** The replacement symbol � appearing inside an answer, a sign of
  corrupted text.

### vs vendor FP8

**Same quality, not a bit-for-bit copy.** The current recipe changes how the model is
computed, so it does not always pick the same token as the vendor model. It predicts
real text just as well.

| What we measured | Current recipe | What it means |
| --- | ---: | --- |
| Same first choice as vendor FP8, first 2K tokens | 83% | Small numeric changes move the top pick |
| Perplexity change, all text | +0.1% [−0.3, +0.6] | No measurable loss |
| Perplexity change, first 2K tokens | 0.0% [−0.6, +0.6] | No measurable loss |
| Perplexity change, beyond 2K tokens | +0.2% [−0.3, +0.7] | No measurable loss |
| Perplexity change, agentic code | +0.4% [−0.4, +1.1] | No measurable loss |
| Perplexity change, synthetic Italian chats | 0.0% [−0.2, +0.1] | No measurable loss |
| Tasks passed (vendor FP8: 73/75) | 73/75 | Same result |

[![Quality loss against vendor FP8 by kind of text: the current recipe stays at about 0% everywhere, NVFP4 loses 3% to 10%.](docs/fidelity/plots/17-quality-by-kind-of-text.png)](docs/fidelity/plots/17-quality-by-kind-of-text.svg)

### vs NVFP4

NVFP4 stores the expert weights, which make up most of the model, in 4 bits instead of 8
to save memory. We ran
[Alex Ellis's public NVFP4 recipe](https://github.com/alexellis/glm-5.3-flash-4x-dgx-spark-switchless/tree/e2d0839a92f118a0a874abcfc1fe8012ee69978e),
with its own checkpoint and engine, on the same four nodes. The results describe that
recipe as published, not the NVFP4 format in general.

**The current recipe is more precise, especially on longer text.** Both columns are
measured against the vendor FP8 model.

| Compared with vendor FP8 | Current recipe | NVFP4 | What it means |
| --- | ---: | ---: | --- |
| Same first choice, first 2K tokens | 83% | 75% | NVFP4 departs more often from vendor FP8's top pick |
| Distance (KL), first 2K tokens | 0.18 | 0.35 | NVFP4's predictions are about twice as far from vendor FP8 |
| Perplexity change, first 2K tokens | 0.0% | −3.3% | NVFP4 is slightly better on short text (not yet explained) |
| Perplexity change, all text | +0.1% | +7.1% | NVFP4 is clearly worse overall; the current recipe shows no measurable loss |
| Perplexity change, beyond 2K tokens | +0.2% | +12.1% | NVFP4's loss grows on longer text |
| Perplexity change, agentic code | +0.4% | +9.3% | NVFP4 is clearly worse on code |
| Perplexity change, synthetic Italian chats | 0.0% | +10.1% | NVFP4 is clearly worse in Italian |
| Same output when run twice, first 2K tokens | always | almost never | NVFP4's engine does not repeat itself |
| Tasks passed | 73/75 | 74/75 | No difference at this sample size |
| Italian answers with broken characters | 0 of 40 | 3 of 40 | A warning sign for NVFP4, not yet significant |

Every NVFP4 perplexity difference is larger than its measuring error; none of the current
recipe's is. The report gives the ranges.

- **Tasks and tool calls:** no difference at this sample size. Both called tools
  correctly in 10 of 10 tests.
- **Broken characters:** they match a known vLLM issue with NVFP4 checkpoints (issue
  54150). 3 of 40 is a warning sign, not yet a statistically significant difference.

[![Quality loss against vendor FP8 by conversation length: the current recipe stays at about 0%, NVFP4 goes from −3% on short text to +12% and +16% on longer conversations.](docs/fidelity/plots/16-quality-by-context-length.png)](docs/fidelity/plots/16-quality-by-context-length.svg)

**Good to know**

- **Measurement mode.** The distance and perplexity numbers come from a measurement mode
  that sends one request at a time. The tasks and the broken-character test ran on the
  normal serving setup.
- **Beyond 2K tokens** the serving engine gives slightly different predictions from run
  to run, even for the vendor model, so there only overall quality is compared.
- **Tasks.** 75 tasks are too few to prove that two setups are equivalent; they can only
  reveal large differences.

The [full report](docs/fidelity/REPORT.md) starts with a
[plain-language summary](docs/fidelity/REPORT.md#in-plain-words). It covers the method,
where the differences come from and the limitations.
[`report.html`](docs/fidelity/report.html) is a self-contained version with every figure,
and the [experiment archive](docs/historical_benchmarks/experiments/2026-09-27-fidelity/README.md)
holds the data.

## Measured resilience

**Tested 29/09/2026.** The deployed default adds memory protections so that heavy use
waits instead of exhausting memory: a fixed SparkCache transfer budget, allocator trim
and a 6,912-token step cap for prompt processing, a 14 GiB KV pool per rank, and an API
queue that answers HTTP 503 when full. The campaign tested five clients at the full
262,144-token context, bursts of 150 clients, cancellations, slow clients, injected
cache faults and a 99-minute mixed load of 840 requests. No out-of-memory kill was
observed in the sampled telemetry, and free memory did not trend down during the long
load. The first long load failed an idle check and the second was ended early by the
operator, so no two-hour run has passed; worker crashes, larger queues and combined
faults remain deferred.

[![Lowest sampled available memory in each of eight resilience test checkpoints: rank 0 between 2.55 and 4.48 GiB, ranks 1–3 between 7.54 and 9.60 GiB, all above the 768 MiB test stop guard.](docs/plots/experiments/2026-09-29-sparkcache-resilience/memory-by-test.png)](docs/plots/experiments/2026-09-29-sparkcache-resilience/memory-by-test.svg)

The [campaign report](docs/benchmarks/experiments/2026-09-29-sparkcache-resilience.md)
records every case, the per-rank memory, the final Rigmark suite and the limits.

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
Install this repository's accepted September 28, 2026 E31 recipe on my four nodes.
Read AGENTS.md, docs/install-from-zero.md, and docs/operations.md first.
Use docs/historical_benchmarks/baselines/2026-09-28-e31/baseline.json and cluster.env.example as the reference.

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
| Run an exclusive service and cache resilience window | [Resilience campaign](docs/resilience.md) |
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
