# Handover to Astra — F1 promoted, next prefill/decode experiments

Audience: **Astra** (`gpt-6-astra`, Codex CLI harness), resuming optimization work on the
four-rank TP4 GLM-5.3-Flash cluster. Written 2026-09-18 by the previous agent at the
owner's request. This file is public: it contains no addresses, host aliases, paths
outside the repository, credentials, or payload. Private evidence lives in the owner's
archives and is referenced only by name.

Read [`AGENTS.md`](AGENTS.md) first; it overrides anything here. Then read the one
document for your task: [`docs/operations.md`](docs/operations.md) (status, lifecycle,
gates, rollback) or [`docs/production-recipe.md`](docs/production-recipe.md) (what the
recipe is and why).

## 1. Current state

**Standard reference: F1**, frozen in [`docs/baseline-f1.json`](docs/baseline-f1.json)
(three native Rigmark runs, 162/162 requests complete, 0 errors, 0 length caps, gates
45/45). Since 2026-09-18 F1 is also the **IaC base recipe**: `cluster.env.example`
describes it completely and it is served without any `TP4_ENV` overlay.

| Layer | F1 value |
| --- | --- |
| Image | `ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:0d4029b3…` (R10 SparkRing/SparkCache), `IMAGE_ID=sha256:5e32aaa1…`, pulled by digest on all ranks |
| Prefix cache | SparkCache connector, `SPARKCACHE_MODE=on`, config `scripts/node/sparkcache/kv-transfer-config.json` |
| Transport | SIRCL single-rail sync prefill (bundle + runtime + entrypoint), `--no-healthcheck` |
| Engine | vLLM with six override modules (`scripts/node/overrides/`), B12X attention, Triton MoE and linear, `--kv-cache-memory-bytes=17179869184`, max 6 sequences, 8192 batched tokens, block 2304, FP8 KV |
| Speculation | DFlash2 drafter, 5 draft tokens, adaptive-k **`batch-uniform`** (k 3/5, up 0.58, down 0.42), drafter KV `fp8_e4m3` |
| Fabric | patched NCCL on the switchless ring, automatic RoCEv2 GID selection (`NCCL_IB_GID_INDEX=-1`) |

F1 medians against the historic F0 (same workstation client, fresh `cache_salt` per run):

| Metric | F0 median | F1 median |
| --- | ---: | ---: |
| Code decode (tok/s) | 50.40 | 51.78 |
| Code C1 aggregate (tok/s) | 37.22 | 38.43 |
| Code C2 aggregate (tok/s) | 57.01 | 57.39 |
| Code C4 aggregate (tok/s) | 71.08 | 84.05 |
| Prose decode (tok/s) | 29.22 | 30.27 |
| Code TTFT (s) | 0.410 | 0.398 |
| Code C4 per-stream TTFT (s) | 0.935 | 0.597 |
| Prefill 8K cold / replay (tok/s) | 2,112 / 4,650 | 2,393 / 9,720 |
| Prefill 32K cold / replay (tok/s) | 2,202 / 21,524 | 2,502 / 31,977 |
| Prefill 64K cold / replay (tok/s) | 2,201 / 35,971 | 2,254 / 35,378 |

The full sixteen-metric table with per-run values is in the baseline JSON. Compare every
future variant with these **fixed F1 medians**; do not rerun or replace F1 without an
explicit owner request.

Cluster state at hand-off: F1 serving from IaC, healthy on four ranks; see
[section 3](#3-promotion-window-and-reproduction).

## 2. Last change: F1 promotion into IaC

The owner decided on 2026-09-18 to promote the whole qualified E09 state ("promote
everything and stop") **without an attribution control**: it is deliberately unknown how
much of the prefill gain survives without the connector. Do not propose that control
again unless the owner asks.

What changed in the repository (details in `CHANGELOG.md` → Unreleased):

- `scripts/launcher/launch-glm53-tp4.sh`: one `SPARKCACHE_MODE=on` lane with image-ID
  gate, SHA-256 verification of the kv-transfer config, connector and SIRCL payload,
  refusal of a duplicate `--kv-transfer-config`, exactly-once connector mount,
  `--no-healthcheck`, PCIe/FlashInfer all-reduce and vLLM plugins disabled. The F0
  `PATCH_FILE` mount belongs to the `off` lane only.
- New tracked assets: `scripts/node/overrides/` (Apache-2.0-derived vLLM modules),
  `scripts/node/sparkcache/kv-transfer-config.json`, and SHA-256 manifests
  (`scripts/node/sparkcache/SHA256SUMS`, `scripts/node/sircl/SHA256SUMS`).
- `scripts/deploy.sh` ships them; `scripts/verify-node.sh` reports `sparkcache payload`
  and `sircl payload` rows, expands `$HOME` mount sources, and treats a digest-pinned
  `IMAGE` as its own pin.
- `scripts/check-f0.py` checks the F1 identity by default (`--baseline
  docs/baseline-f0.json` for F0). The frozen F0 reference keeps its exact launcher bytes in
  `scripts/node/reference/launch-glm53-tp4-f0-20260912.sh`.
- Documentation: stack table and SparkCache/SIRCL section, six boot signatures, F1→F0
  rollback row, promotion record, payload-placement install step, credits.

## 3. Promotion window and reproduction

Recorded in full in
[`docs/operations.md`](docs/operations.md#f1-promotion-into-iac-on-2026-09-18). Summary:

- The cluster **serves F1 from the IaC base recipe** since 2026-09-18 07:29 UTC, with no
  `TP4_ENV`: four healthy ranks, `/health` 200 after 713 s, both post-boot gates PASS,
  pinned content ID and registry digest on 4/4, `batch-uniform` confirmed in the rank-0
  log. `~/tp4/cluster.env` on the nodes is the F1 recipe, so rank-0 autostart brings back
  the same stack after a reboot.
- One reproduction run (owner direction: one, not three), `mini-f1-iac-r1`: 54/54
  requests, 0 errors, gates 15/15; code 52.25 tok/s, C1/C2/C4 37.74/56.53/86.39 tok/s
  (all inside the F1 range), prose 29.91 tok/s, prefill in line or better. Verdict:
  IaC reproduces F1. Keep comparing against the frozen F1 medians, not this run.
- `scripts/check-f0.py` reports one pre-existing host item: a rank with
  `NeedDaemonReload` on the fabric iptables unit. Ask the owner before any
  `systemctl daemon-reload` or host deploy.

## 4. Things that will bite you

- **Payload is not in git and must never be.** The SparkCache connector and the SIRCL
  bundle/runtime have no license (see [`CREDITS.md`](CREDITS.md)). They live on each
  rank under `~/tp4/sparkcache/` and `~/tp4/sircl/{bundle,runtime}/`; `.gitignore`
  blocks them. The SIRCL `rank<N>.env` files carry the site's fabric peers and GIDs, so
  their hashes live only in the gitignored `scripts/node/sircl/SHA256SUMS.site` of the
  operator checkout. Never edit a manifest to match a file.
- **The repository is public.** Before any commit, review the full diff and every new
  file for addresses, aliases, home paths, tokens and payload.
- **Old window overlays are incompatible with the F1 base.** The E09 overlays under the
  ignored `scripts/node/etc/local/` add SIRCL on top of F0 and now fail closed. Write
  every new candidate as a small delta on top of F1, and use the same `TP4_ENV` for
  every command in its window, including `down`.
- **Readiness is `GET /health` 200 only.** The lane runs with `--no-healthcheck`, so
  `docker ps` shows no health state; never use `/v1/models` for readiness.
- **SparkCache is persistent across restarts.** Cold-prefill measurements need a fresh
  `cache_salt` per run. Rigmark at the pinned commit forwards `cache_salt` to prefill
  requests (upstream plus the one local fix `ca0b5c1`); without it, a "cold" run reads
  the previous run's cache. The cache lives under the runtime cache volume; watch disk.
- **Rigmark measures performance, not answer quality.** The E09 code-prose/format issue
  was closed as model behavior against a fixed token budget, reproduced on the official
  model with no local configuration. It is not a stop condition.
- **Known, pre-existing host items (not caused by F1, need their own host window):**
  `scripts/deploy-host.sh --check` reports host-script and sudoers drift on all ranks,
  and one rank reports `NeedDaemonReload` for the fabric iptables unit. Both touch `/etc`
  or systemd and need explicit owner authorization.
- **No F1 archive.** `scripts/f0-reference.py` captures only the historic F0 reference.
  The F1 rollback reference is git plus the payload manifests; generalizing the archive
  tool is an open item, not a prerequisite.

## 5. Plan for future experiments

Owner direction: build on F1 and look for further **prefill and decode** gains. Rank
results by the `AGENTS.md` criteria: agentic code C1 and two-to-four-agent concurrency
first, prose next, prefill welcome without regressions. One experiment at a time, three
consecutive native Rigmark runs in one loaded process, same client, fresh `cache_salt`
per run, fixed-F1-median comparison, verdict promote / discard / unresolved, restore F1
if not promoted. Agree each experiment with the owner before its window.

1. **E06 — larger prefill chunks (`--max-num-batched-tokens` 8192 → 16384). Design
   check first.** F1 cold prefill is flat at about 2.3–2.5k tok/s from 8K to 64K, so it
   is compute-bound per rank and larger chunks could help. But the SIRCL bundle admits
   synchronous-prefill collectives only for 1024/2048/4096/8192 query rows (fused Q8192
   capacity), and the pooled-indexer/kpool buffers in `scripts/node/overrides/` are sized
   from `max_num_batched_tokens`. A 16384 budget may push prefill chunks off the SIRCL path
   onto the NCCL fallback, and it raises indexer memory. Before booting, confirm from the
   bundle source what happens to a 16384-row chunk (fallback or error), and budget the
   extra buffer memory against `--kv-cache-memory-bytes`. If SIRCL cannot take it, a
   useful variant may be 8192 with prefill-chunk tuning rather than 16384.
2. **All-reduce paths in the SparkCache lane.** The launcher forces
   `VLLM_ENABLE_PCIE_ALLREDUCE=0` and `VLLM_ALLREDUCE_USE_FLASHINFER=0` so they cannot
   interfere with SIRCL. Re-enabling the FlashInfer (or B12X PCIe) all-reduce for decode
   only could cut per-step latency on TP4 decode, which drives C1, C2 and C4. This needs
   a launcher knob; today the values are hard-coded in the ON lane. High interference
   risk: gate hard and roll back immediately on any transport error.
3. **E02 — JIT warmup of the mHC path.** Cheap; targets first-request TTFT and stability
   after boot, small steady-state gain.
4. **Speculation tuning for prose.** Prose decode stays near 30 tok/s. Candidates:
   adaptive-k signal `mean`, or a lower `k_lo`, measured for prose gains that leave code
   and concurrency unchanged. `batch-uniform` must stay; mixed-k steps cost C2/C4.
5. **KV layout knobs not yet tried** (`VLLM_KV_CACHE_LAYOUT`, alternative block size),
   decode-side, cheap to try, but check SparkCache compatibility of the cache geometry
   first (the connector persists blocks).
6. **E03 — token-sharded FP8 prefill.** The only lever on cold-prefill arithmetic per
   rank; it needs a verified FP8 port. Treat it as a project, not a single window.

Already excluded: E04 (exact-shape CUDA graphs showed worse C2 in its one valid run) and
E05 (MTP, rejected by the owner). The SparkCache attribution control is declined by the
owner.

## 6. Authorization reminders

`AGENTS.md` governs. In short: node discovery, deploy, start/stop, host changes, pulls
and downloads need an authorized window; commits, pushes, tags and releases each need an
explicit request; never print or commit secrets; never automate weight deletion; never
restart one rank alone.
