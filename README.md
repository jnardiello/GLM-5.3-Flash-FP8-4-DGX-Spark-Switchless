# GLM-5.3-Flash FP8 on four NVIDIA GB10 nodes (switchless)

[![Follow me on X](https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/jnardiello)

Run GLM-5.3-Flash FP8 with vLLM across four NVIDIA GB10 systems, connected directly
through a switchless ConnectX-7 RoCE ring. This is my daily driver for coding and
parallel agents, with a **256K context window (262,144 tokens)**. The repository
contains the infrastructure as code, runtime patches, and guides to
[install it with an agent](#install-with-an-agent) on compatible hardware.

## Measured performance

Current accepted baseline: **25/09/2026 · E27c, long prefills paced while agents
generate**, measured with native [Rigmark](https://github.com/alexellis/rigmark).
Frozen medians of **three complete runs: 162/162 requests**, zero measurement/runtime
errors and 44/45 native output gates passing (one code answer reached the 8,192-token
output budget).

| Workload | Current · 25/09/2026 E27c | vs previous [📊 24/09/2026 E27](docs/benchmarks/baselines/2026-09-24-e27.md) |
| --- | ---: | ---: |
| Code decode, one request | 56.12 tok/s | 55.11 tok/s · ≈ unchanged (+1.84%) |
| Code C1, end-to-end | 43.58 tok/s | 44.38 tok/s · ≈ unchanged (-1.79%) |
| Code C2, aggregate end-to-end | 66.52 tok/s | 64.95 tok/s · +2.41% |
| Code C4, aggregate end-to-end | 96.70 tok/s | 95.51 tok/s · ≈ unchanged (+1.24%) |
| Prose decode | 33.12 tok/s | 32.62 tok/s · ≈ unchanged (+1.52%) |
| Code TTFT | 0.390 s | 0.403 s · -3.23% |
| Prose TTFT | 0.376 s | 0.376 s · ≈ unchanged (+0.00%) |
| C1 per-stream TTFT | 0.340 s | 0.376 s · -9.57% |
| C2 per-stream TTFT | 0.446 s | 0.456 s · ≈ unchanged (-2.19%) |
| C4 per-stream TTFT | 0.525 s | 0.720 s · -27.08% |
| Prefill 8K, cold | 2,558.9 tok/s | 2,609.3 tok/s · ≈ unchanged (-1.93%) |
| Prefill 8K, replay | 9,432.0 tok/s | 9,589.5 tok/s · ≈ unchanged (-1.64%) |
| Prefill 32K, cold | 2,689.3 tok/s | 2,779.9 tok/s · -3.26% |
| Prefill 32K, replay | 37,056.3 tok/s | 37,652.9 tok/s · ≈ unchanged (-1.58%) |
| Prefill 64K, cold | 2,646.9 tok/s | 2,690.4 tok/s · ≈ unchanged (-1.62%) |
| Prefill 64K, replay | 39,931.7 tok/s | 40,046.3 tok/s · ≈ unchanged (-0.29%) |

Both baselines were measured over the same direct LAN client path. Percentages use
unrounded values. “≈ unchanged” marks owner-accepted changes of roughly 1–2%, with the
exact delta retained; it does not establish statistical equivalence.
Higher throughput and lower TTFT are better. C1/C2/C4 mean one, two or four concurrent
requests. Decode excludes the initial wait; end-to-end speed includes it. TTFT is time
to first token. Concurrency outputs cap at 256 tokens; long decode allows 8,192.
Cold prefill uses a fresh cache salt per run. These are inference measurements,
not complete agent-task timings.

**What E27c changes.** E27 added the engine's native `--prefill-schedule-interval 8`:
while requests generate, prefill work runs on one engine step in eight, so a long prompt
arriving from another client no longer freezes the agents already generating. It had two
costs, which E27c removes with a small patch of the image's scheduler:

- **Short prompts waited for the cadence too.** E27c admits a prompt with fewer than
  2,048 tokens left to compute at once. C4 per-stream TTFT returns from 0.720 s to
  0.525 s, the level before E27.
- **The cadence switched itself off when requests queued up.** When several long
  contexts arrived together, the first stream to start generating still stalled while the
  others prefilled. E27c keeps the cadence on while requests are queued.

| Four 32K prompts arriving together | 1st stream | 2nd | 3rd | 4th |
| --- | ---: | ---: | ---: | ---: |
| Decode, E27 | 6.2 tok/s | 9.1 | 15.0–15.3 | 56–57 |
| Decode, **E27c** | **11.3–12.5** | **10.9–11.0** | 14.8–15.0 | 68–72 |
| First token, E27 → E27c | 15–16 → 16–18 s | 29–30 → 29–31 s | 41 → 43–44 s | 55 → 58–60 s |

With a single long prompt arriving next to running agents, E27c keeps E27's behaviour:
they go on generating at about 7 tok/s instead of about 1.5 during a 32K prefill. Short
and medium prompts now reach their first token much sooner:

| Prompt arriving while agents generate | Agents | Its first token, E27 → E27c | Longest pause of the agents |
| --- | ---: | ---: | ---: |
| 1K tokens | 3 | 1.98 → **0.91** s | 0.68 → 0.67 s |
| 8K tokens | 3 | 5.79 → **3.74** s | 2.66 → 2.66 s |
| 32K tokens | 3 | 18.6 → 18.4 s | 2.66 → 2.97 s |

The cadence spreads the arriving prefill instead of removing it: the agents' total delay
is about the same, but their streams keep moving. A cadence step still carries one prefill
chunk of up to 8,192 tokens (about 2.6 s), which caps the first stream near 12 tok/s when
several long contexts arrive together. The [current benchmark report](docs/benchmarks/baselines/2026-09-25-e27c.md)
lists per-run values, the interference phase, counts and limitations.

The graphs compare current and previous medians side by side, with each delta calculated
against the previous September 24 E27 baseline. Click an image for its SVG version.

[![Current E27c versus previous September 24 E27 baseline: generation throughput, time to first token and percentage changes.](docs/plots/comparisons/2026-09-25-e27c-vs-2026-09-24-e27/generation.png)](docs/plots/comparisons/2026-09-25-e27c-vs-2026-09-24-e27/generation.svg)

[![Current E27c versus previous September 24 E27 baseline: cold prefill, immediate replay and percentage changes at 8K, 32K and 64K.](docs/plots/comparisons/2026-09-25-e27c-vs-2026-09-24-e27/prefill.png)](docs/plots/comparisons/2026-09-25-e27c-vs-2026-09-24-e27/prefill.svg)

The [benchmark archive](docs/benchmarks/README.md) retains earlier baselines, experiments
and separate reproduction results. See [local Rigmark reports](docs/rigmark_reports/README.md)
for saving and viewing native receipts.

The [current recipe](docs/production-recipe.md), encoded in
[`cluster.env.example`](cluster.env.example), combines the digest-pinned SparkRing image,
DFlash2 with adaptive verification capped by its effective draft budget, E03 mHC prefill
sharding, hybrid INT8/BF16 KDA input projections, E21 8-bit weights for the KDA output
and MLA attention projections, E22b 8-bit weights for the DFlash2 drafter, the E27 prefill
cadence with the E27c scheduler, SparkCache replay views and SIRCL/patched NCCL.
The **15 GiB KV pool per rank** and **262,144-token context limit** are retained.

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
Install this repository's accepted September 25, 2026 E27c recipe on my four nodes.
Read AGENTS.md, docs/install-from-zero.md, and docs/operations.md first.
Use docs/historical_benchmarks/baselines/2026-09-25-e27c/baseline.json and cluster.env.example as the reference.

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
