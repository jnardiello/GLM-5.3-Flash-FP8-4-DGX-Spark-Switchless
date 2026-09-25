# GLM-5.3-Flash FP8 on four NVIDIA GB10 nodes (switchless)

[![Follow me on X](https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/jnardiello)

Run GLM-5.3-Flash FP8 with vLLM across four NVIDIA GB10 systems, connected directly
through a switchless ConnectX-7 RoCE ring. This is my daily driver for coding and
parallel agents, with a **256K context window (262,144 tokens)**. The repository
contains the infrastructure as code, runtime patches, and guides to
[install it with an agent](#install-with-an-agent) on compatible hardware.

## Measured performance

Current accepted baseline: **25/09/2026 · E28b, seven draft tokens and a 16 GiB KV pool**,
measured with native [Rigmark](https://github.com/alexellis/rigmark). Frozen medians of
**three complete runs: 162/162 requests**, zero measurement/runtime errors and 45/45 native
output gates passing.

| Workload | Current · 25/09/2026 E28b | vs previous [📊 25/09/2026 E27c](docs/benchmarks/baselines/2026-09-25-e27c.md) |
| --- | ---: | ---: |
| Code decode, one request | 61.01 tok/s | 56.12 tok/s · +8.70% |
| Code C1, end-to-end | 43.18 tok/s | 43.58 tok/s · ≈ unchanged (-0.92%) |
| Code C2, aggregate end-to-end | 67.47 tok/s | 66.52 tok/s · ≈ unchanged (+1.43%) |
| Code C4, aggregate end-to-end | 95.54 tok/s | 96.70 tok/s · ≈ unchanged (-1.19%) |
| Prose decode | 33.28 tok/s | 33.12 tok/s · ≈ unchanged (+0.48%) |
| Code TTFT | 0.390 s | 0.390 s · ≈ unchanged (+0.00%) |
| Prose TTFT | 0.376 s | 0.376 s · ≈ unchanged (+0.00%) |
| C1 per-stream TTFT | 0.394 s | 0.340 s · +15.88% |
| C2 per-stream TTFT | 0.462 s | 0.446 s · +3.59% |
| C4 per-stream TTFT | 0.523 s | 0.525 s · ≈ unchanged (-0.38%) |
| Prefill 8K, cold | 2,558.4 tok/s | 2,558.9 tok/s · ≈ unchanged (-0.02%) |
| Prefill 8K, replay | 8,670.6 tok/s | 9,432.0 tok/s · -8.07% |
| Prefill 32K, cold | 2,739.3 tok/s | 2,689.3 tok/s · ≈ unchanged (+1.86%) |
| Prefill 32K, replay | 33,262.0 tok/s | 37,056.3 tok/s · -10.24% |
| Prefill 64K, cold | 2,692.7 tok/s | 2,646.9 tok/s · ≈ unchanged (+1.73%) |
| Prefill 64K, replay | 39,291.0 tok/s | 39,931.7 tok/s · ≈ unchanged (-1.60%) |

Both baselines were measured over the same direct LAN client path. Percentages use
unrounded values. The two C4 rows use suites 2–3: the owner excluded suite 1's C4 block,
whose three rounds all started staggered; the excluded values stay in the frozen record. “≈ unchanged” marks owner-accepted changes of roughly 1–2%, with the
exact delta retained; it does not establish statistical equivalence.
Higher throughput and lower TTFT are better. C1/C2/C4 mean one, two or four concurrent
requests. Decode excludes the initial wait; end-to-end speed includes it. TTFT is time
to first token. Concurrency outputs cap at 256 tokens; long decode allows 8,192.
Cold prefill uses a fresh cache salt per run. These are inference measurements,
not complete agent-task timings.

**What E28b changes.** The DFlash2 drafter is trained with blocks of eight positions, but
earlier recipes let it propose only five tokens per step. E28b lets a single request draft
seven; batches of 2–6 keep three.

- **Long answers decode faster:** code decode +8.7%. Code and structured output often
  accept the sixth and seventh token; structured output reaches 7.8 tokens per step.
- **Costs:** short answers gain nothing and a single request's first token comes about
  50 ms later; cached replay at 8K–32K takes about 0.1 s longer.
- **KV pool 16 GiB per rank:** seven draft tokens hold about 5% fewer KV tokens per GiB,
  so the pool grows from 15 to 16 GiB. Capacity is 1,365,066 tokens, 5.21 full 262,144-token
  contexts: five agents at the full context fit together. Rank 0 kept at least 2.1 GB of
  memory available during the three suites.

Everything from E27c is retained: long prefills paced while agents generate, short prompts
admitted at once, and the cadence kept while requests are queued. The
[current benchmark report](docs/benchmarks/baselines/2026-09-25-e28b.md) lists per-run
values, the acceptance probes, the memory samples and limitations.

The graphs compare current and previous medians side by side, with each delta calculated
against the previous September 25 E27c baseline. Click an image for its SVG version.

[![Current E28b versus previous E27c baseline: generation throughput, time to first token and percentage changes.](docs/plots/comparisons/2026-09-25-e28b-vs-2026-09-25-e27c/generation.png)](docs/plots/comparisons/2026-09-25-e28b-vs-2026-09-25-e27c/generation.svg)

[![Current E28b versus previous E27c baseline: cold prefill, immediate replay and percentage changes at 8K, 32K and 64K.](docs/plots/comparisons/2026-09-25-e28b-vs-2026-09-25-e27c/prefill.png)](docs/plots/comparisons/2026-09-25-e28b-vs-2026-09-25-e27c/prefill.svg)

The [benchmark archive](docs/benchmarks/README.md) retains earlier baselines, experiments
and separate reproduction results. See [local Rigmark reports](docs/rigmark_reports/README.md)
for saving and viewing native receipts.

The [current recipe](docs/production-recipe.md), encoded in
[`cluster.env.example`](cluster.env.example), combines the digest-pinned SparkRing image,
DFlash2 with adaptive verification capped by its effective draft budget, E03 mHC prefill
sharding, hybrid INT8/BF16 KDA input projections, E21 8-bit weights for the KDA output
and MLA attention projections, E22b 8-bit weights for the DFlash2 drafter, the E27 prefill
cadence with the E27c scheduler, seven draft tokens (E28b), SparkCache replay views and
SIRCL/patched NCCL.
The **16 GiB KV pool per rank** (E28b) and the **262,144-token context limit** apply.

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
| Operator payload | Original SparkCache connector, encoder from the pinned image, and SIRCL bundle/runtime; follow [payload preparation](docs/install-from-zero.md#8-place-the-sparkcache-and-sircl-payload) |

The operator payload is pinned by hash and is not redistributed here. DFlash2 carries
non-commercial terms; review [credits and licenses](CREDITS.md). The API has no
authentication or TLS, so keep it on a trusted network or behind an authenticating
proxy.

Replace the placeholders below, then give this prompt to the agent in the checkout:

```text
Install this repository's accepted September 25, 2026 E28b recipe on my four nodes.
Read AGENTS.md, docs/install-from-zero.md, and docs/operations.md first.
Use docs/historical_benchmarks/baselines/2026-09-25-e28b/baseline.json and cluster.env.example as the reference.

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
If an operator payload is missing, report exactly what I must supply.
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
