# GLM-5.3-Flash FP8 on four NVIDIA GB10 nodes (switchless)

[![Follow me on X](https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/jnardiello)

Run GLM-5.3-Flash FP8 with vLLM across four NVIDIA GB10 systems, connected directly
through a switchless ConnectX-7 RoCE ring. This is my daily driver for coding and
parallel agents, with a **256K context window (262,144 tokens)**. The repository
contains the infrastructure as code, runtime patches, and guides to
[install it with an agent](#install-with-an-agent) on compatible hardware.

## Measured performance

Native [Rigmark](https://github.com/alexellis/rigmark) measurements from the same
workstation, shown as medians of per-run values. Dates use day/month/year.

<div align="center">

| Workload | 11/09/2026 | 18/09/2026 | 19/09/2026 | Change, 19/09 vs 11/09 |
| --- | ---: | ---: | ---: | ---: |
| Code decode, one request | 50.40 tok/s | 51.78 tok/s | 53.89 tok/s | +6.9% |
| Prose decode | 29.22 tok/s | 30.27 tok/s | 31.88 tok/s | +9.1% |
| Code, one request (end-to-end) | 37.22 tok/s | 38.43 tok/s | 40.30 tok/s | +8.3% |
| Code, two concurrent requests (aggregate, end-to-end) | 57.01 tok/s | 57.39 tok/s | 57.25 tok/s | +0.4% |
| Code, four concurrent requests (aggregate, end-to-end) | 71.08 tok/s | 84.05 tok/s | 87.53 tok/s | +23.1% |
| Prefill 8K, cold | 2,112.2 tok/s | 2,393.4 tok/s | 2,354.7 tok/s | +11.5% |
| Prefill 32K, cold | 2,202.0 tok/s | 2,502.1 tok/s | 2,534.1 tok/s | +15.1% |
| Prefill 64K, cold | 2,201.3 tok/s | 2,254.0 tok/s | 2,421.7 tok/s | +10.0% |
| Code, two concurrent requests, per-stream TTFT | 0.616 s | 0.447 s | 0.495 s | -19.6% |
| Code, four concurrent requests, per-stream TTFT | 0.935 s | 0.597 s | 0.680 s | -27.3% |

</div>

Percentage changes use the unrounded medians and are relative to September 11.
Higher throughput and lower TTFT are better.

September 11 and 18 each use **three runs**. September 19 uses **exactly two completed
runs: 108/108 requests**, zero measurement/runtime errors, and 30/30 native output
gates passing. Its third run was excluded entirely because of competing traffic.
The records for [September 11](docs/baseline-f0.json), [September 18](docs/baseline-f1.json),
and [September 19](docs/baseline-2026-09-19.json) include all 16 metrics and evidence hashes.

Decode speed excludes the initial wait; end-to-end speed includes it. TTFT is time to
first token. Concurrency tests cap output at 256 tokens per request; long decode tests
allow 8192. These measure inference, not complete agent tasks. Cold prefill processes
a new prefix, using a fresh `cache_salt` per run.

Compared with September 18, September 19 has higher code/prose throughput and longer
concurrent-request TTFT, with **15 GiB KV per rank versus 16 GiB** previously. Allocator and host memory probes
were active; their overhead was not isolated. Two runs give less repeatability evidence
than three. A separate [one-run IaC reproduction](docs/reproduction-2026-09-19.json)
records live deployment checks and 54 requests; the frozen medians above remain unchanged.

<details>
<summary>Run-by-run comparison charts: initial versus current configuration</summary>

The charts show all three September 11 runs and three available September 19 runs:
the two accepted runs plus the separate IaC reproduction, marked with a gold diamond.
In the first two images, each point is one native per-run metric; lines show the
observed range. Median ticks and labels summarize the three plotted values per
configuration. This descriptive view combines two separate September 19 boots; the table above keeps the frozen
two-run baseline. The run excluded for competing traffic is not included.

[![Generation throughput and TTFT for three runs of each configuration; the IaC run is marked separately.](docs/plots/generation-comparison.png)](docs/plots/generation-comparison.svg)

[![Cold prefill and immediate cache replay at 8K, 32K and 64K, showing all individual runs and their medians.](docs/plots/prefill-comparison.png)](docs/plots/prefill-comparison.svg)

[![Throughput versus input context length at 8K, 32K and 64K: cold prefill compares September 11 and 19; diagnostic eight-token decode tails show September 19 only.](docs/plots/context-throughput.png)](docs/plots/context-throughput.svg)

The third image plots context length on X and tok/s on Y. Lines connect medians of
the three per-run values; shading spans their observed range. The prefill panel uses
the same cold-request metrics as above. The decode panel uses the **8-token output
tails of those prefill requests**, with each run represented by its median of three
requests. These short probes are sensitive to streaming bursts and do not measure
sustained decode. Only September 19 is shown for decode: the initial public record
does not contain these rates, and its raw receipts were unavailable for extraction.
The [27 extracted measurements and source hashes](docs/context-decode-probes.json)
make the additional panel reproducible.

Click any image for the SVG version. The TTFT axis is logarithmic so every
observation, including the initial 5.712-second result, remains visible.
Regenerate all three figures from the public JSON records with
`python3 scripts/plot-baseline-comparison.py` in an environment containing
`matplotlib==3.11.2`. The script also writes PNG copies for the README.

</details>

The [current recipe](docs/production-recipe.md), encoded in
[`cluster.env.example`](cluster.env.example), combines the digest-pinned SparkRing
image, DFlash2 with adaptive verification, hybrid INT8/BF16 KDA projections,
SparkCache prefix reuse, and SIRCL/patched NCCL transport.

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
Install this repository's September 19, 2026 recipe on my four nodes.
Read AGENTS.md, docs/install-from-zero.md, and docs/operations.md first.
Use docs/baseline-2026-09-19.json and cluster.env.example as the reference.

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
