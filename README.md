# GLM-5.3-Flash FP8 on four NVIDIA GB10 nodes (switchless)

[![Follow me on X](https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/jnardiello)

Run GLM-5.3-Flash FP8 with vLLM across four NVIDIA GB10 systems, connected directly
through a switchless ConnectX-7 RoCE ring. This is my daily driver for coding and
parallel agents, with a **256K context window (262,144 tokens)**. The repository
contains the infrastructure as code, runtime patches, and guides to
[install it with an agent](#install-with-an-agent) on compatible hardware.

## Measured performance

Current accepted baseline: **19/09/2026 · E03 + replay views + draft budget**, measured
with native [Rigmark](https://github.com/alexellis/rigmark). Frozen medians of
**three complete runs: 162/162 requests**, zero measurement/runtime errors and
45/45 native output gates passing.

| Workload | Current · 19/09/2026 E03 | Change vs [📊 11/09/2026](docs/benchmarks/baselines/2026-09-11.md) |
| --- | ---: | ---: |
| Code decode, one request | 53.81 tok/s | +6.8% |
| Code C1, end-to-end | 40.23 tok/s | +8.1% |
| Code C2, aggregate end-to-end | 61.83 tok/s | +8.5% |
| Code C4, aggregate end-to-end | 86.93 tok/s | +22.3% |
| Prose decode | 31.19 tok/s | +6.7% |
| Code TTFT | 0.396 s | -3.4% |
| Prose TTFT | 0.381 s | -4.0% |
| C1 per-stream TTFT | 0.343 s | -15.1% |
| C2 per-stream TTFT | 0.449 s | -27.1% |
| C4 per-stream TTFT | 0.572 s | -38.8% |
| Prefill 8K, cold | 2,558.4 tok/s | +21.1% |
| Prefill 8K, replay | 9,417.2 tok/s | +102.5% |
| Prefill 32K, cold | 2,662.0 tok/s | +20.9% |
| Prefill 32K, replay | 37,177.4 tok/s | +72.7% |
| Prefill 64K, cold | 2,616.1 tok/s | +18.8% |
| Prefill 64K, replay | 40,078.3 tok/s | +11.4% |

Percentages use unrounded medians relative to the initial September 11 baseline.
Higher throughput and lower TTFT are better. C1/C2/C4 mean one, two or four concurrent
requests. Decode excludes the initial wait; end-to-end speed includes it. TTFT is time
to first token. Concurrency outputs cap at 256 tokens; long decode allows 8,192.
Cold prefill uses a fresh cache salt per run. These are inference measurements,
not complete agent-task timings.

The owner accepts the remaining small performance tradeoffs. The [current benchmark
report](docs/benchmarks/baselines/2026-09-19-e03.md) compares all 16 metrics with the
previous base and records variability, memory, counts and limitations. Earlier suites
and isolated checks remain separately archived. The default recipe matches the measured
candidate; [deployment and reproduction of these new defaults](docs/historical_benchmarks/baselines/2026-09-19-e03/promotion.json)
remain pending.

The graphs show only the current accepted values. Click an image for its SVG version.

[![Current baseline generation throughput and time to first token.](docs/plots/baselines/2026-09-19-e03/generation.png)](docs/plots/baselines/2026-09-19-e03/generation.svg)

[![Current baseline cold prefill and immediate cache replay at 8K, 32K and 64K.](docs/plots/baselines/2026-09-19-e03/prefill.png)](docs/plots/baselines/2026-09-19-e03/prefill.svg)

The [benchmark archive](docs/benchmarks/README.md) retains earlier baselines, experiments
and separate reproduction results. See [local Rigmark reports](docs/rigmark_reports/README.md)
for saving and viewing native receipts.

The [current recipe](docs/production-recipe.md), encoded in
[`cluster.env.example`](cluster.env.example), combines the digest-pinned SparkRing image,
DFlash2 with adaptive verification capped by its effective draft budget, E03 mHC prefill
sharding, hybrid INT8/BF16 KDA projections, SparkCache replay views and SIRCL/patched NCCL.
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
Install this repository's accepted September 19, 2026 E03 recipe on my four nodes.
Read AGENTS.md, docs/install-from-zero.md, and docs/operations.md first.
Use docs/historical_benchmarks/baselines/2026-09-19-e03/baseline.json and cluster.env.example as the reference.

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
