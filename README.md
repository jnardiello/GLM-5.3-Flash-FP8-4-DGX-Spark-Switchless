# GLM-5.3-Flash FP8 on four NVIDIA GB10 nodes (switchless)

[![Follow me on X](https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/jnardiello)

Run GLM-5.3-Flash FP8 with vLLM across four NVIDIA GB10 systems, connected directly
through a switchless ConnectX-7 RoCE ring. This repository provides the configuration,
installation scripts, runtime patches, and operating procedures for the cluster.

GLM-5.3-Flash is my daily driver for heavy coding workloads and parallel agent use.
The deployment is tuned for code generation and concurrent requests, with a
**256K context window (262,144 tokens)**. The verified hardware is four ASUS Ascent
GX10 nodes; other GB10 systems may need different hardware and host settings.

## Current configuration

- **Model:** GLM-5.3-Flash FP8, tensor-parallel across four GPUs.
- **Engine:** SparkRing/SparkCache vLLM image pinned by registry digest.
- **Prefix reuse:** persistent SparkCache storage for long shared contexts.
- **Speculative decoding:** DFlash2 with an adaptive verification length shared by
  every request in a batch (`batch-uniform`).
- **Communication:** SIRCL single-rail prefill transport and patched NCCL for the
  switchless ring.

The current configuration is the default in
[`cluster.env.example`](cluster.env.example). The
[production recipe](docs/production-recipe.md) explains its components and rollback.

The SparkCache connector and SIRCL bundle/runtime are operator-supplied files, pinned
by SHA-256 but not redistributed here. Read the
[payload installation step](docs/install-from-zero.md#8-place-the-sparkcache-and-sircl-payload)
and [third-party terms](CREDITS.md) before setting up this configuration. The DFlash2
checkpoint used by this recipe carries non-commercial terms.

## Measured performance

**Current** is the configuration qualified on September 18, 2026. **Previous** is the
configuration measured on September 11, before SparkCache and batch-uniform adaptive
verification became the defaults. Each column reports medians across three native
[Rigmark](https://github.com/alexellis/rigmark) runs from the same workstation.

<div align="center">

| Workload | Previous | Current |
| --- | ---: | ---: |
| Code decode, one request | 50.4 tok/s | 51.8 tok/s |
| Code, one request (end-to-end) | 37.2 tok/s | 38.4 tok/s |
| Code, two concurrent requests (aggregate, end-to-end) | 57.0 tok/s | 57.4 tok/s |
| Code, four concurrent requests (aggregate, end-to-end) | 71.1 tok/s | 84.1 tok/s |
| Code, four concurrent requests, per-stream TTFT | 0.94 s | 0.60 s |
| Prose decode | 29.2 tok/s | 30.3 tok/s |
| Prefill 8K / 32K, cached prefix replay | 4.6k / 21.5k tok/s | 9.7k / 32.0k tok/s |
| Prefill 64K, cold | 2.2k tok/s | 2.3k tok/s |

</div>

Decode throughput measures generation after the first token; end-to-end throughput
includes the initial wait. TTFT is time to first token. Concurrent-request tests use
up to 256 output tokens per request, while the long decode tests allow up to 8192.
These are inference benchmarks; they do not measure complete agent tasks. Cold prefill
processes a new prefix, while replay reuses a cached prefix; each run uses a fresh
`cache_salt`.

The [current baseline](docs/baseline-f1.json) and
[previous baseline](docs/baseline-f0.json) contain all metrics, per-run values, settings,
and evidence hashes. Current qualification completed 162/162 requests with zero errors.
The [qualification summary](docs/production-recipe.md#qualification-and-reproduction)
also records reproduction through the repository's deployment scripts.

## Hardware

<div align="center">

| Item | Verified configuration |
| --- | --- |
| Nodes | 4 × ASUS Ascent GX10, one NVIDIA GB10 and 128 GB unified memory each |
| Storage | 1 TB NVMe per node; allow at least 330 GiB free for a fresh model fetch |
| Fabric | one ConnectX-7 per node, two addressed ports, 4 × QSFP28 DAC, 200 Gb/s, MTU 9000 |
| Topology | closed ring: rank 0 ↔ 1 ↔ 2 ↔ 3 ↔ 0, one private /24 per link |
| Management | separate Ethernet or trusted VPN path for SSH and the rank-0 API |

</div>

Site addresses and hardware selections belong in the gitignored `cluster.env`, created
from the public template. The API and fabric require a trusted private network. The
API has no authentication, TLS, rate limit, or caller isolation; use a trusted VPN or
an authenticating reverse proxy for access beyond that network.

## Start here

For installation, begin with four hosts running Ubuntu, NVIDIA drivers, Docker with
GPU support, and `rdma-core`, then follow the installation guide in order. It covers
preflight, site configuration, bootstrap, images, weights, payload, deployment, and
post-boot verification. For an existing cluster, begin with the operations guide.
Agents working in this repository must read [AGENTS.md](AGENTS.md) first.

| Goal | Guide |
| --- | --- |
| Install on prepared hosts | [Install from zero](docs/install-from-zero.md) |
| Inspect, deploy, start, stop, recover, or roll back | [Operations](docs/operations.md) |
| Cable, configure, or diagnose the RoCE ring | [Fabric](docs/fabric.md) |
| Understand the current components and customizations | [Production recipe](docs/production-recipe.md) |
| Understand files installed on the nodes | [Node assets](scripts/node/README.md) |
| Build and install patched NCCL | [NCCL guide](scripts/node/nccl/README.md) |

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

## Changes and credits

[CHANGELOG.md](CHANGELOG.md) records changes to the deployment and tooling.
[CREDITS.md](CREDITS.md) lists sources and third-party terms. Original project material
is under [LICENSE](LICENSE); derived files and fetched artifacts retain their own terms.
