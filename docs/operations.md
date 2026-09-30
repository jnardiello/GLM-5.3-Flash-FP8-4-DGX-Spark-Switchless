# Operations

Use this guide for an existing cluster: handoff, read-only status, deployment,
lifecycle, recovery, rollback, and promotion. Installation and artifact downloads are
in [`install-from-zero.md`](install-from-zero.md).

## Authorization boundary

Authorization is scoped to named targets, actions, and a maintenance window. Continue
through all actions already authorized for the same window. Ask before expanding that
scope to privileged bootstrap/downloads, host network or reboot changes, deploy or
service lifecycle, recipe promotion, deletion, or publication. Commits, pushes,
tags, pull requests, releases, and weight purges require an explicit request.

Read-only inspection may proceed when the owner has already placed the four targets
in scope. Never request or expose credentials. Site values may live in ignored local
configuration and mode-0600 reports; do not commit or publish them.

Stop and report instead of repairing when a rank is missing, two stacks exist, health
is inconsistent, a foreign GPU workload is present, or discovered state differs from
the requested recipe.

## First handoff

Before touching a node, establish:

1. whether the task is installation, operation, recovery, or local-only work;
2. the four SSH targets in rank order and the deployment account;
3. the human-confirmed ring cable map and allowed private subnets;
4. that the API remains on a trusted LAN/VPN and use is compatible with the DFlash2
   license described in [`CREDITS.md`](../CREDITS.md);
5. the concrete success condition and the actions already authorized.

For unknown hardware or topology, run the read-only preflight from
[`install-from-zero.md`](install-from-zero.md) and present its proposed map before
generating files. A failed strict host-key check requires out-of-band fingerprint
verification.

## Read-only status

Prerequisite: a filled local `cluster.env`, SSH access to all four ranks, and the
targets in scope.

```sh
./scripts/tp4ctl status
./scripts/tp4ctl health
./scripts/tp4ctl fabric-check
./scripts/deploy-host.sh --no-push --run tp4-iommu.sh --status
```

`status` must show the configured container with the same name and image on each rank.
It filters by that name, so also inspect the unfiltered container list and GPU compute
processes on all four nodes; this catches a second stack under another name:

```sh
. ./cluster.env
for n in ${TP4_HOSTS:-$NODES}; do
  printf '\n=== %s ===\n' "$n"
  ssh "$n" 'sudo -n docker ps --no-trunc --format "{{.Names}}\t{{.Image}}\t{{.Status}}"; nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader'
done
```

Expected: exactly one TP4 serving stack is present, its four configured containers are
the only inference containers, and every GPU compute process belongs to that stack.
Stop on a differently named serving container, a second inference stack, or any foreign
GPU workload. Do not stop it or repair state during discovery.

`health` requires `/health` 200 and performs a smoke completion. `fabric-check` reports
the addressed fabric ports and their MTU and requires all eight jumbo pings; perform the
separate speed and RDMA/HCA/GID probes in [`fabric.md`](fabric.md). Off the management
LAN, run the health probe from rank 0 because the local `MASTER_IP` may be unreachable:

```sh
ssh <ALIAS_RANK0> 'curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8000/health'
```

Then verify these signatures that `docker ps` does not prove:

| Signature | Read-only check | Expected result |
| --- | --- | --- |
| Four-rank identity | `./scripts/tp4ctl status`, `docker inspect` | the configured `CONTAINER` is `Up` once on every rank; its `RepoDigests` entry equals `IMAGE` and `.Image` equals `IMAGE_ID` |
| Triton MoE configuration | `./scripts/tp4ctl logs` on rank 0 | `Using TRITON Fp8 MoE backend` and `Using configuration from …NVIDIA_GB10…json` |
| Host IOMMU tier | `deploy-host.sh ... tp4-iommu.sh --status` | passthrough on all four ranks, drop-in installed, GRUB synchronized |
| NCCL HCA/GID selection | `./scripts/verify-node.sh` | every configured HCA has an active IPv4-mapped RoCEv2 GID on its addressed fabric netdev; explicit index or automatic `-1` mode is identified |
| Adaptive scheduler | rank-0 log | `AdaptiveKScheduler active (enabled=1 … mode=batch-uniform engine_k=7 async=True)` and a `num_speculative_tokens_per_batch_size` table in engine initialization |
| Draft length and operational KV pool | rank-0 log, `docker inspect`, `./scripts/check-f0.py` | `num_spec_tokens=7`, `engine_k=7`, CUDA graph ceiling 72, `--kv-cache-memory-bytes=15032385536`, and `GPU KV cache size: 1,194,033 tokens` (4.55x at 262,144); five full maximum contexts cannot be resident together |
| SparkCache lane | `docker inspect`, `./scripts/verify-node.sh` | `--kv-transfer-config` naming `SparkContextCacheConnector` in the command, entrypoint `/opt/sircl-serving/entrypoint.sh`, the connector, encoder, engine override and SIRCL mounts present, `sparkcache payload` and `sircl payload` rows PASS; `docker ps` shows no health state because the lane runs with `--no-healthcheck` |
| Hybrid KDA | rank logs, `docker inspect`, `./scripts/check-f0.py` | `E20_KDA_INPUT_W8A16_READY` on every rank: 34 modules, group 128, padded N=6400, threshold 2048, shared scratch 51,515,392 bytes; matching source hashes and `E20_MEMORY_PROBE` records |
| E21 residual projections | rank logs, `docker inspect`, `./scripts/check-f0.py` | `E21_BF16_RESIDUE_W8A16_READY` on every rank: 67 modules in the `kda_o_proj`, `mla_fused_qkv_a_proj`, `mla_o_proj` and `mla_q_b_proj` families, `mla_layout=q_lora`, group 128, threshold 2048, added scratch 79,691,776 bytes; `VLLM_E21_BF16_RESIDUE_W8A16=1` and matching `experiments/e03/bf16-residue/` source hashes |
| E22b drafter conversion | rank logs, `docker inspect`, `./scripts/check-f0.py` | `E22_DRAFTER_W8A16_READY` on every rank: 30 modules in the `down_proj`, `gate_up_proj`, `kernel_projection`, `o_proj` and `qkv_proj` families, `context_kv_w8a16` false, 0 added scratch bytes; `VLLM_E22_DRAFTER_W8A16=1`, `VLLM_E22_CONTEXT_KV_W8A16=0` and matching `experiments/e03/drafter-w8a16/` source hashes; the drafter CUDA graphs are captured fresh at every boot |
| E27 prefill cadence | `docker inspect`, `./scripts/check-f0.py` | `--prefill-schedule-interval 8` exactly once in every rank's command; no boot log line, because the argument is native to the image |
| E27c scheduler | rank-0 logs, `docker inspect`, `./scripts/check-f0.py` | `E27C_CADENCE_WHEN_QUEUED_READY interval=8` and `E27B_SHORT_PREFILL_READY tokens=2048 interval=8` on rank 0; `VLLM_E27B_SHORT_PREFILL_TOKENS=2048`, `VLLM_E27C_CADENCE_WHEN_QUEUED=1` and the E29 scheduler below (which contains the E27c patch) mounted over `vllm/v1/core/sched/scheduler.py` on every rank; `E27C_CADENCE_KEPT_WHILE_QUEUED` appears the first time long prompts are queued while others decode |
| E29 end-drain and idle coalescing | rank-0 logs, `docker inspect`, `./scripts/check-f0.py` | `E29_END_DRAIN_READY trace=0` and `E29_IDLE_COALESCE_READY ms=4 trace=0` on rank 0; `VLLM_E29_END_DRAIN=1`, `VLLM_E29_IDLE_COALESCE_MS=4` and `VLLM_E29_TRACE=0` on every rank, with the `experiments/e03/end-drain/scheduler.py` hash mounted over `vllm/v1/core/sched/scheduler.py` and the `experiments/e03/end-drain/core.py` hash over `vllm/v1/engine/core.py`; the first request held at a length finish logs `E29_END_DRAIN_HELD` |
| E31 indexer (tail ring on, head gate off) | rank logs, `docker inspect`, `./scripts/check-f0.py` | `E31_INDEXER_GATE path=fp32` and `E31_KPOOL_TAIL_RING ring=12` on every rank, logged by the first forward; `VLLM_GLM53_INDEXER_GATE_TC_FLAG=/tmp/glm53-indexer-gate-tc` and `VLLM_GLM53_KPOOL_TAIL_RING_FLAG=/tmp/glm53-kpool-tail-ring` on every rank and no `VLLM_GLM53_INDEXER_GATE_TC` or `VLLM_GLM53_KPOOL_TAIL_RING` variable, with the `experiments/e03/e31-indexer/pooled_indexer.py` and `glm_kpool.py` hashes mounted over `vllm/models/glm5next/nvidia/pooled_indexer.py` and `ops/glm_kpool.py`; `ring=4` or `path=bf16-tc` means a flag file was changed and is not the accepted state |
| Protected SparkCache default | all rank logs, deployed recipe, `docker inspect`, `./scripts/check-f0.py` | connector SHA-256 `aa046965637b…`, config SHA-256 `d99bd6720572…`, `SPARKCACHE_CPU_BUDGET_READY cap_bytes=1073741824 floor_bytes=1073741824` and `SPARKCACHE_DISK_STREAM_READY chunk_bytes=8388608` on every rank; the versioned operational identity pins the full values |
| SparkCache disk capacity | all rank logs, `docker inspect`, `./scripts/check-f0.py` | `SPARK_CONTEXT_CACHE_MAX_BYTES=214748364800` and `SPARK_CONTEXT_CACHE_LOW_WATERMARK_BYTES=171798691840` on every rank, and each rank's `sparkcache: config role=WORKER` line reports `max_bytes=214748364800 low_bytes=171798691840 ttl_seconds=0`; `max_bytes=0` means the store has no disk limit |
| Memory bounds and API admission | rank logs, `docker inspect`, `./scripts/check-f0.py` | `PREFILL_CACHE_TRIM_READY rank=<rank> enabled=1 trigger=pre_eager_prefill` on every rank; `RESILIENCE_STEP_TOKEN_CAP_READY configured=8192 effective=6912 block=2304 eager_above=72` and `TP4_ADMISSION_READY` on rank 0; limits are six active admission slots, 128 queued, 8 MiB body, 1,800 s queue, 3,600 s request, and 30 s body/send idle; active slots do not imply resident engine requests |
| E35 verify length (policy `hybrid`) | rank logs, `docker inspect`, `./scripts/check-f0.py` | `E35_RUNNER_K_READY enabled=1 flag=/tmp/glm53-e35-policy margin=0.005 wait_ms=2.8` and `E35_CONF_RECORDER_READY enabled=1` on every rank; `E35_SCHEDULER_READY flag=/tmp/glm53-e35-policy k_hi=7` on rank 0; `VLLM_E35_ENABLE=1` and `VLLM_E35_POLICY_FLAG=/tmp/glm53-e35-policy` on every rank, with the `experiments/e03/e35-runner-k/` speculator and scheduler hashes mounted over `vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py` and `/opt/tp4/adaptive_k_scheduler.py` (the runner is the E36 copy, see the next row), and `policy.flag` (content `hybrid`) read-only at `/tmp/glm53-e35-policy`; every 200 participating steps rank 0 logs `E35_RUNNER_K` with its decisions, waits, broadcast latency and `errors=0` |
| E36 INT8 shared lm_head | all rank logs, `docker inspect`, `./scripts/check-f0.py` | `E36_LM_HEAD_W8A16_READY` on every rank with `"freed_bytes": 317194240`, `"keep_bf16": false`, `"shape": [38720, 4096]`, `"shared_with_drafter": true`, a `tp_rank_vocab_start` of 0, 38720, 77440 or 116160, a group-128 weight error near 0.72% and a logit error below 1%; `VLLM_E36_LM_HEAD_W8A16=1` and `VLLM_E36_KEEP_BF16=0` on every rank and no `VLLM_E36_FLAG`; the `experiments/e03/e36-lm-head-w8a16/` runner (the E35 runner plus the load-time conversion) mounted over `vllm/v1/worker/gpu/model_runner.py` and `e36_lm_head_w8a16.py` over `vllm/models/glm5next/nvidia/e36_lm_head_w8a16.py`, both read-only |

The operational default retains the [accepted E31 engine recipe](benchmarks/baselines/2026-09-28-e31.md) and those
signatures, including the E03 `SPARK_MHC_PREFILL_SHARD=1` and its measured source
mounts, the E21 residual projections, the E22b drafter conversion with its own cache
namespace, the E27 prefill cadence, the E27c scheduler patch, and seven draft tokens for a
single request, and the E29 length-finish hold and 4 ms
idle-coalescing window. E31 replaces the pooled indexer and its C4 kernels with the
speculative-safe tail ring, and keeps the head-gate switch off. Eligible eager long prefills log
`SPARK_MHC_PREFILL rows=6912 owner_rows=1728 rs=90 ag=95 aux=5` on all four ranks;
this is a workload activation receipt, not a requirement to run extra inference during
an identity-only check. Runtime sources keep their `experiments/e03/` paths to preserve
measured hashes; the E21 hook and module live under `experiments/e03/bf16-residue/`, the
E22b drafter override and module under `experiments/e03/drafter-w8a16/`, the E27c
scheduler under `experiments/e03/queued-cadence/` and the E31 indexer under
`experiments/e03/e31-indexer/`. Leave the two E31 flag files absent or at their defaults
(gate `0`, ring `1`); change them only for a same-load experiment, following that
directory's README.
Do not append the historical experiment overlays to the new defaults. `CACHE_DIR` must be a
disk-backed filesystem with free space for the persistent cache and at least one in-flight
snapshot staging file. Do not place it on tmpfs: the bounded connector uses disk staging to
avoid holding a full serialized snapshot in RAM.

The selected protected connector SHA-256 is
`aa046965637b685ec1f6a00e427eb46279e28bf0d49ee546236bc774be2de9bd`.
The scheduler SHA-256 is
`89dbca0f780911aae3f5f5b3954237a9271a30cc9905d56f70705e77696b8ef8`, the E35 copy of the
draft-budget scheduler (`697f99bf1951535dcc3381776fa744ad8204d83d2606f341b617bac7a31949b4`)
with one additions-only rule. Its rank-0 startup signatures are
`draft-budget active=1 source=SchedulerOutput.resolve_num_spec_tokens_to_schedule engine_k=7`
and `E35_SCHEDULER_READY flag=/tmp/glm53-e35-policy k_hi=7`,
alongside the enabled batch-uniform async policy. When concurrent requests trigger a
limit, expect `draft-budget first-cap budget=3` and aggregate `limited_requests` and
`trimmed_tokens`. These count placeholder handouts, not accepted tokens or saved compute.
An adaptive-policy fallback invalidates activation.

The optional [prefill allocator diagnostic](../scripts/node/experiments/e03/prefill-cache-trim/README.md)
uses a separate worker overlay. Its launcher verifies the deployed source manifest;
each rank must emit `PREFILL_CACHE_TRIM_READY ... enabled=1` after warmup and emit
before/after allocator receipts for eligible eager prefills. These signatures apply
only to that experiment. They are not part of the protected default or evidence that
concurrent requests fit in memory.

The separate [prefill step-cap candidate](../scripts/node/experiments/e03/prefill-step-cap/README.md)
retains the configured 8,192-token capacity and uses an effective 6,912-token target
budget per scheduler output. Its launcher verifies the scheduler manifest. Rank 0
must emit `RESILIENCE_STEP_TOKEN_CAP_READY configured=8192 effective=6912 block=2304 eager_above=72`;
each eager output has a `RESILIENCE_STEP_TOKEN_CAP` receipt. The remaining request
tokens are scheduled later through the existing alignment and draft-slot rules.
These experiment signatures do not establish an aggregate memory bound or change
the protected default. Stop the candidate with its full overlay and return to the
default without `TP4_ENV`, using the coordinated procedure and both functional gates.
The resilience overlay can additionally select a smaller KV pool. Its optional
`kv_cache_memory_bytes` identity must match exactly one loaded
`--kv-cache-memory-bytes` value on each rank, with `--max-model-len 262144`
unchanged. Record the engine's actual KV-token capacity and test the largest context
and concurrent admission before drawing a resilience conclusion; see
[the campaign procedure](resilience.md).

The one-step operational return is
`TP4_ENV=scripts/node/reference/operational-20260930-e35.env`: it removes only E36.
`TP4_ENV=scripts/node/reference/operational-20260930-e31-mb.env` removes E36 and E35, and
`TP4_ENV=scripts/node/reference/operational-20260929-memory-bounded.env` removes E36, E35 and the
SparkCache disk limit. The complete immediate operational return is
`TP4_ENV=scripts/node/reference/operational-20260929-sparkcache-protected.env`.
It keeps the bounded SparkCache connector, 8 MiB transfers and 1 GiB transient/admission
budgets, but restores the 16 GiB KV pool and removes allocator trim, the 6,912-token step cap,
API admission and the disk limit. It therefore does not retain the complete memory-bounded
protection.
The historical `TP4_ENV=scripts/node/reference/baseline-20260928-e31.env` return restores
the measured E31 recipe with the prior replay connector, transfer config and cache namespace;
it also removes the SparkCache memory limits.
`baseline-20260925-e29.env` additionally restores the E29 indexer.
`baseline-20260925-e28b.env` restores the E28b recipe with the E27c scheduler and
the image's engine core. `baseline-20260925-e27c.env` restores the E27c recipe with five draft
tokens and a 15 GiB KV pool.
`baseline-20260924-e27.env` restores the E27 recipe with the image's own scheduler.
`baseline-20260924-e22b.env` restores the E22b recipe without the prefill cadence.
`baseline-20260923-e21.env` restores the E21 recipe with the vendor drafter, no E22 module
or flags and the E21 cache namespace. `baseline-20260919-e03.env` restores the E03
recipe, and the older
`baseline-20260919.env` restores the earlier September 19 base.
Use the same effective recipe for all commands in a coordinated service window.

For a boot caused by rank-0 autostart, inspect the units too:

```sh
ssh <ALIAS_RANK0> 'systemctl status tp4-autostart tp4-fabric-iptables --no-pager'
```

Expected: one coherent four-rank stack, `/health` 200, green fabric, all signatures,
and no unexpected active `tp4-flusher` after readiness. Stop on any missing signature,
unreachable rank, or health mismatch. `/v1/models` is never a readiness check.
`tp4ctl status` exits nonzero unless it can verify the configured container running on
all four ranks and receive `/health` 200.

### Fast operational identity check

Run the reusable read-only check when a single concise operational identity and idle
verdict is needed:

```sh
./scripts/check-f0.py
./scripts/check-f0.py --base-url http://127.0.0.1:8000
TP4_ENV=scripts/node/reference/baseline-20260928-e31.env ./scripts/check-f0.py --baseline docs/historical_benchmarks/baselines/2026-09-28-e31/baseline.json
TP4_ENV=scripts/node/reference/baseline-20260925-e29.env ./scripts/check-f0.py --baseline docs/historical_benchmarks/baselines/2026-09-25-e29/baseline.json
TP4_ENV=scripts/node/reference/baseline-20260925-e28b.env ./scripts/check-f0.py --baseline docs/historical_benchmarks/baselines/2026-09-25-e28b/baseline.json
TP4_ENV=scripts/node/reference/baseline-20260925-e27c.env ./scripts/check-f0.py --baseline docs/historical_benchmarks/baselines/2026-09-25-e27c/baseline.json
TP4_ENV=scripts/node/reference/baseline-20260924-e27.env ./scripts/check-f0.py --baseline docs/historical_benchmarks/baselines/2026-09-24-e27/baseline.json
TP4_ENV=scripts/node/reference/baseline-20260924-e22b.env ./scripts/check-f0.py --baseline docs/historical_benchmarks/baselines/2026-09-23-e22b/baseline.json
TP4_ENV=scripts/node/reference/baseline-20260923-e21.env ./scripts/check-f0.py --baseline docs/historical_benchmarks/baselines/2026-09-23-e21/baseline.json
TP4_ENV=scripts/node/reference/baseline-20260919-e03.env ./scripts/check-f0.py --baseline docs/historical_benchmarks/baselines/2026-09-19-e03/baseline.json
TP4_ENV=scripts/node/reference/baseline-20260919.env ./scripts/check-f0.py --baseline docs/historical_benchmarks/baselines/2026-09-19/baseline.json
TP4_ENV=scripts/node/reference/baseline-20260918.env ./scripts/check-f0.py --baseline docs/historical_benchmarks/baselines/2026-09-18/baseline.json
./scripts/check-f0.py --baseline docs/historical_benchmarks/baselines/2026-09-11/baseline.json
```

The second form uses an already-established localhost SSH tunnel when the management LAN
is not directly reachable. The checker reads the local `cluster.env` and honors `TP4_ENV`
as the effective delta. By default it validates the
[E36 operational identity](operational-identities/2026-09-30-e36-lm-head.json),
which pins the frozen **September 28 E31** engine identity by hash and adds the bounded
connector, KV14, trim, scheduler-cap, API-admission and SparkCache disk-capacity identities,
the E35 speculator, scheduler copy, policy file, variables and boot lines, and the E36 runner,
conversion module, variables and per-rank receipts. It still checks
seven draft tokens, the adaptive table and high state, the CUDA graph limit, the 14 GiB KV
budget, the E29 scheduler and engine-core hashes, the E31 indexer hashes and flag files, the E27c, E29 and E31 flags and rank-0 boot lines, the E27 prefill cadence in the recipe
and every rank's command, the E22b drafter sources/flags/boot signature, the E21 sources/flag/boot signature, the mHC sources/flag, protected connector, draft-budget scheduler/activation, hybrid KDA,
KV byte budget and image content ID. Historical `--baseline` selection requires an
explicit record and its matching complete runtime recipe. An identity-changing delta
fails the check.

The checker runs bounded SSH probes for all four ranks in parallel with strict host-key checking.
It verifies the effective recipe against the selected identity's pins, the configured running
container and key command/environment identity, image digest and model marker, the
selected identity's runtime source hashes and boot receipts, unfiltered
GPU containers and processes, inactive flusher, addressed MTU-9000 interfaces, all eight
jumbo directions, `/health` 200, and zero running/waiting requests. Stdout is exactly one
concise `<selected-identity> CHECK PASS` or `<selected-identity> CHECK FAIL` line. If
configuration or identity loading fails before selection, the label is the neutral fallback
`BASELINE CHECK FAIL`. Details and command errors go to a mode-0600 JSON report in a new
mode-0700 system temporary directory outside the checkout.

PASS covers only those operational checks. It does not hash every model/drafter file,
replay engine async metadata, or send the coherent-response and tool-call requests. Use
`scripts/verify-node.sh --full-model` for full artifact verification and the
[post-boot functional gates](#post-boot-functional-gates) after a changed boot.

### Frozen previous rollback reference

This archive tool captures the **September 11** configuration. Use it only when that
configuration is serving; it cannot capture either newer baseline. The newer baselines use their IaC recipe and payload manifests; the
[qualification record](production-recipe.md#qualification-and-reproduction) distinguishes
measured configurations from any later reproduction run. For a September 11 capture, use a new
private archive outside the checkout. The tracked portable part is
`scripts/node/reference/f0-20260912.env`; it freezes every non-site September 11 runtime knob,
automatic GID selection, image and artifact pins without changing `CONTAINER`. The
private archive adds the resolved addresses, aliases, interfaces, HCA choices, renderer,
ports, exact installed configuration and the observed unit state. It also keeps the
61,581,280-byte NCCL library, rather than relying on a mutable `.rollback` file.

```sh
TP4_ENV=<currently-serving-overlay> python3 scripts/f0-reference.py capture \
  --archive <new-private-directory> \
  --prechange-source <private-pre-change-source-snapshot> \
  --evidence-dir <private-read-only-check-receipts>
python3 scripts/f0-reference.py verify --offline --archive <private-directory>
python3 scripts/f0-reference.py verify --live --archive <private-directory>
python3 scripts/f0-reference.py plan-restore --archive <private-directory>
```

`capture` refuses an existing directory. Every command receipt records rank, timestamps,
argv, return status, timeout, output hashes and truncation. Docker environment capture is
an explicit allowlist; raw inspect data, credentials, SSH material, browser state and
NetworkManager secret profiles are excluded. Unsupported queries and query errors remain
explicit, and required errors, truncation or a runtime identity change make the capture
incomplete. The before/after identity covers the full filtered container command,
environment, mounts, image, start identity and restart count. Loaded NCCL is proven from
the process mapping and hashed through that process root; a static mount hash alone is not
accepted as proof of loaded bytes.

Live comparison keeps each raw command receipt unchanged, but excludes three documented
clock/counter fields from the stable host signature: address preferred/valid lifetime
countdowns, link `info_data.gc_timer`, and `iptables-save` generation comments plus
built-in-chain packet/byte counters. Interface addresses and MTU, link configuration,
chain policy, firewall rule order and inline rule comments remain exact comparison inputs.
Each live verification report privately retains the four current receipts and identifies
the sections and keys that differ instead of reporting only an opaque signature mismatch.

The archive separates integrity from operational readiness. Its files and directories
must be mode 0600 and 0700, its SHA-256 manifest must cover the exact file set, and the
saved site configuration must reproduce the generated netplan and firewall environment.
Those checks can pass while `scripts/check-f0.py` reports a current operational issue,
such as disk state awaiting `systemctl daemon-reload`; the receipt records that state and
capture never repairs it. Images and model weights are pinned and inventoried but not
duplicated. Full target-model integrity still comes from the immutable model manifest;
drafter availability and revision metadata do not claim a full weight hash.

`plan-restore` performs another read-only comparison and writes its result and exact
commands to a new private report. It never executes those commands. If the live runtime
differs, give `--current-overlay <relative-path>` only after confirming that it is the
overlay of the currently running process; otherwise the coordinated `down` command is
left blocked. The generated order first stages a SHA-verified controller as a regular
file, stops all four ranks with the currently serving overlay, then selects September 11, deploys
the archived IaC, atomically installs the archived NCCL bytes, and prepares autostart.
The reference overlay is deployed at the same relative path by setting
`TP4_ENV=scripts/node/reference/f0-20260912.env` on `scripts/deploy.sh`.

The live launcher supports the Current configuration, so the manifest pins the exact
September 11 launcher bytes as the frozen copy `scripts/node/reference/launch-glm53-tp4-f0-20260912.sh`
(same SHA-256 as before). Its controller is independently frozen at
`scripts/node/reference/tp4ctl-f0-20260912.sh`, also with its original SHA-256.
Deployment installs that copy as `tp4ctl-f0-reference`; fixes to the operational
`scripts/tp4ctl` do not change September 11. Restore plans use the controller inside the
verified archive, including the original `scripts/tp4ctl` path in older sealed archives.
Existing archive files and manifests stay unchanged. The September 11 overlay is valid
only together with the archived September 11 IaC that `plan-restore` deploys; on top of
the Current base `cluster.env` it inherits
`SPARKCACHE_MODE=on` and `IMAGE_ID`, and the launcher refuses the September 11 image (fail-closed).
Use that verified archive workflow for a September 11 return; do not reconstruct
its recipe by mixing individual historical values with the current base.
`f0-reference.py capture` selects the September 11 baseline named in the current
reference manifest. Generated restore checks use the path recorded in the sealed
archive's own manifest, so older archives retain their original source layout.

## Post-boot functional gates

Run both gates within two minutes of `/health` reaching 200 after any changed boot.
They verify response and tool-call behavior.

### Coherent response and thinking-off behavior

```sh
curl -s http://<MGMT_IP_RANK0>:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"glm-5.3-flash",
    "temperature":0,
    "max_tokens":64,
    "chat_template_kwargs":{"enable_thinking":false},
    "messages":[{"role":"user","content":"What is the capital of Italy? Reply with one sentence."}]
  }' | python3 -m json.tool
```

Pass: normal response content is present and coherently names Rome. The local template
adapter closes an empty `<think></think>` block for this request flag; see
[`production-recipe.md`](production-recipe.md). This is local compatibility behavior,
not an official reasoning mode.

### Structured tool call

```sh
curl -s http://<MGMT_IP_RANK0>:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"glm-5.3-flash",
    "max_tokens":256,
    "messages":[{"role":"user","content":"What is the weather in Milan?"}],
    "tools":[{"type":"function","function":{"name":"get_weather","description":"Get weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],
    "tool_choice":"auto"
  }' | python3 -m json.tool
```

Pass: `choices[0].message.tool_calls[0].function.name` is `get_weather` and its
arguments are valid JSON containing Milan. If either gate fails, take the full stack
down and report the failure; never repair or restart one serving rank in isolation.

## Deploy a repository or recipe change

Prerequisites: inspect the current status; identify one rollback; update the real
`cluster.env` and annotated `cluster.env.example` together when a production knob
changes; update `CHANGELOG.md`; and have deploy/restart actions within the authorized
window.

```sh
$EDITOR cluster.env
./scripts/deploy.sh --check
./scripts/deploy.sh
./scripts/tp4ctl restart
```

`deploy.sh` is additive: it copies and hashes managed files without deleting node
content or touching a running container. It does replace `~/tp4/cluster.env`, which is
what rank-0 autostart uses next. `restart` is disruptive and always cycles all ranks.

`EXTRA_DOCKER_ENV` is one word-split string carrying the tuned MoE JSON, the adaptive
scheduler mount, `PYTHONPATH`, policy variables, the retained base, E03, E21 and E22b vLLM override mounts, the
SIRCL bundle/runtime mounts with their entrypoint, and the connector and encoder mounts. An overlay
replaces the complete value. Preserve every unrelated entry, avoid spaces/globs in
values, and never clear the string while `--scheduler-cls
adaptive_k_scheduler.AdaptiveKScheduler` remains in `EXTRA_VLLM_ARGS`.

The accepted configuration includes `scripts/node/overrides/`, the measured sources and
config under `scripts/node/experiments/e03/` (including `bf16-residue/` and `drafter-w8a16/`), `scripts/node/sparkcache/kv-transfer-config.json`
and payload `SHA256SUMS` manifests in the deploy set. It also stages the frozen
September 18 model and cache config for rollback. The same deploy places the included
SparkCache connector/encoder and SIRCL bundle/runtime from `third_party/`, and the
generated SIRCL site files when the checkout holds them (see
[`install-from-zero.md`](install-from-zero.md#8-prepare-the-sparkcache-and-sircl-payload)).
Let `./scripts/verify-node.sh` confirm both payload rows before `restart`.

Expected: every copied file matches its source, all ranks launch in order 3→2→1→0,
`/health` reaches 200, and all runtime signatures return. Run the
[post-boot functional gates](#post-boot-functional-gates) within two minutes, followed
by any task-specific verification. Stop the stack immediately if a gate fails.

## Migrate the accepted overlay to defaults

Accepting measurements, encoding IaC and applying the default are separate steps.
First encode the measured recipe in `cluster.env.example`, prepare a new ignored site
configuration from it, preserving the current node addresses, accounts, paths,
interfaces and topology, and confirm with `TP4_DRY_RUN=1` that its launcher command
on each of the four ranks is identical to the measured overlay's. The accepted suites
are the measurement; promotion does not rerun them.

**Measured candidate still serving.** When the four serving processes are the ones
that produced the accepted suites and the dry runs match, apply the default without a
restart:

1. Install the prepared site configuration locally, unset `TP4_ENV`, run
   `./scripts/deploy.sh --check`, then `./scripts/deploy.sh`.
2. In the same step, remove only the candidate's verified autostart overlay selection on
   rank 0 and run `systemctl daemon-reload`, so the next boot selects the same default
   recipe. Preserve every unrelated drop-in. A frozen experiment overlay may reject the
   new base, so never leave that candidate selection active.
3. Run `./scripts/check-f0.py` and retain its four-rank identity report.

**A different recipe is serving.** Retain the serving `cluster.env` and its exact
overlay until the full-cluster stop, because frozen experiment overlays expect the
previous base. Prepare and verify the payload and the complete rollback first. Then,
in an authorized window:

1. Stop all ranks using the still-active site configuration and serving `TP4_ENV`.
2. Install the prepared site configuration locally, unset `TP4_ENV`, and deploy.
   Remove only the verified obsolete overlay selection and reload systemd; preserve every
   unrelated drop-in.
3. Verify the pinned payloads and all eight jumbo pings; start the four-rank service
   once. Complete both functional gates within two minutes of `/health` 200.
4. Run `./scripts/check-f0.py` and retain its four-rank identity report.

For a performance promotion, record parity and live identity in its promotion record.
For an operational-only default such as cache protection, keep the frozen performance
record unchanged and retain a separate operational identity and private migration receipt.
A reproduction benchmark runs only on explicit owner request, with the frozen
source/settings, a fresh comparison ID and unique absolute outputs, recorded separately
from the accepted measurements; the current protocol sends no `cache_salt`.

### Migrate the protected SparkCache default without reloading the model

Use this path only when all four configured ranks are serving the bounded-transfer
variant coherently, `/health` is 200, both functional gates have passed, and there is no
second stack. Capture the four process IDs/start times, commands, protected config and
payload hashes, plus rank 0's effective autostart unit and complete drop-in list. In a
private directory, render the active E31-plus-budget recipe and the prepared default with
the same site configuration. Require byte-for-byte launcher-command equality on ranks
0–3. A local template parity test does not replace this site-specific proof.

Inspect the one SparkCache overlay drop-in recorded as created for that candidate. Remove
it only when its path and bytes match that record and it contains only the exact
`TP4_ENV` selection being retired. Preserve every other directive and file. If provenance,
content or ownership is uncertain, leave it in place and report the mismatch.

With those prerequisites satisfied:

1. Install the prepared protected `cluster.env` locally with no `TP4_ENV`; run
   `./scripts/deploy.sh --check`, then `./scripts/deploy.sh`. Deployment updates the next
   autostart recipe but does not restart the running containers.
2. Remove only the verified SparkCache selection described above and run
   `systemctl daemon-reload` on rank 0. Confirm the effective unit now selects no overlay.
3. Re-read all four process IDs/start times and require them to be unchanged. Require
   `/health` 200, then run the default `./scripts/check-f0.py` and retain its private
   report with the four-rank parity evidence.

Do not edit the frozen E31 baseline or its promotion record for this operational migration.
If a rank is missing, health is inconsistent, commands differ, another stack exists or the
drop-in cannot be identified exactly, stop this procedure and report. Recover the whole
group through the coordinated recovery procedure before considering a later migration;
never reload or repair one rank to make the prerequisites appear true.

## Start, stop, restart, logs, and power

```sh
./scripts/tp4ctl up
./scripts/tp4ctl down
./scripts/tp4ctl restart
./scripts/tp4ctl logs [<node>]
./scripts/tp4ctl poweroff
```

`up`, `down`, `restart`, and `poweroff` are disruptive. Never restart a single rank:
it cannot rejoin the existing communicator. `up` refuses a degraded fabric, requires
the page-cache flusher active on all ranks, verifies stale containers absent, launches
workers before rank 0, waits for `/health`, and verifies the flusher stopped. A failed
prerequisite or partial flusher start occurs before teardown and leaves an existing
serving stack alone. Once prelaunch teardown begins, a teardown or launch error,
readiness timeout, interruption, or a persistent final flusher failure triggers a best-effort
full four-rank container teardown and flusher shutdown, then returns failure.

After readiness, a failed flusher stop/verification logs the host, last attempted phase
and exit status. The controller waits one second and repeats the complete stop and
absence verification on **all four ranks once**. A second failure triggers the full
cleanup above. Missing collected transient units are accepted only after process absence
is verified; SSH, sudo and probe errors remain failures.

The readiness timeout is **35 minutes**, with `/health` polled every 30 seconds.
Keep full timestamped logs from all ranks during distributed initialization and weight
loading. A pause in log output alone does not justify interrupting startup. Inspect
worker stacks with available diagnostic tools if progress pauses, and allow the
readiness timeout to elapse unless a worker terminates, memory is exhausted, a fatal
error occurs or a rank is lost.

`down` attempts both stop operations on every rank and succeeds only after it verifies
the configured container and flusher absent everywhere; an already absent container or
flusher is successful. `restart` and `poweroff` stop before starting or powering off
anything when that verification is incomplete. `poweroff` asks interactively and leaves
rank 0 until last.

Expected after `up`: readiness and both functional gates. Expected after `down`: no matching
container or flusher on any rank. Stop and report partial teardown or launch; do not
repair only the failed node.

## Configuration overlays

The [SparkCache bounded-transfer implementation](../scripts/node/experiments/e03/sparkcache-ram-budget/README.md)
is part of the operational default: a shared 1 GiB transient-work budget per rank, a
1 GiB host admission floor and 8 MiB disk-transfer pieces. It avoids a complete snapshot
in RAM. Its worker boot signatures are
`SPARKCACHE_CPU_BUDGET_READY cap_bytes=1073741824 floor_bytes=1073741824` and
`SPARKCACHE_DISK_STREAM_READY chunk_bytes=8388608`; pressure skips
log `SPARKCACHE_CPU_BUDGET_SKIP` and restores recompute. The guide gives the supported
cache paths, separate namespace, runtime validation and E31 rollback. The historical
`sparkcache-ram-budget/delta.env` is retained only to prove command parity with the
previously loaded E31-plus-delta configuration; do not apply it on top of the default.

A local overlay named by `TP4_ENV` is sourced after production `cluster.env`. It is a
delta and remains inert when not named:

```sh
TP4_ENV=path/to/window.env ./scripts/deploy.sh
TP4_ENV=path/to/window.env ./scripts/tp4ctl restart
TP4_ENV=path/to/window.env ./scripts/tp4ctl status
TP4_ENV=path/to/window.env ./scripts/verify-node.sh
TP4_ENV=path/to/window.env ./scripts/tp4ctl down
```

Derive each new overlay from the Current base recipe. An overlay prepared for an older
base may duplicate connector or transport configuration already supplied by Current.
Reapply only the intended delta and review the merged configuration before deployment.

Use the same `TP4_ENV` on every command in that window. Keep `CONTAINER` unchanged so
plain production commands still find exactly one stack. To leave the window, restart
with no `TP4_ENV`; the base recipe is sourced again.

An automatic-GID window may set only `NCCL_IB_GID_INDEX=-1` (and clear a previously
non-empty `NCCL_IB_GID_INDEX_BY_RANK` in the overlay). The launcher preserves unrelated
`EXTRA_DOCKER_ENV` entries, rejects selector overrides there, and still forces
AF_INET/RoCEv2. Its local HCA/GID preflight must pass on all four ranks before serving.

The controller and launcher revalidate the merged recipe before remote work: `NODES`,
`MGMT_IPS`, `FABRIC_TARGETS`, and any `TP4_HOSTS` resolution remain exactly four ranks,
`MASTER_IP` remains the first management address, and an overlay cannot change
`CONTAINER`.

Expected: the overlay changes only listed keys and the boot signatures identify the
intended recipe. Stop on a missing overlay, changed container name, or failed gate.

### Historical E03 C5 functionality window

This procedure was defined while E03 was the default recipe. The C5 overlays check the
E03 cache configuration and refuse the current default, and `TP4_ENV` names a single file,
so they cannot be stacked on the E03 rollback. They remain as the record of that window;
a sequence-count window on the current base needs a new delta derived from E27c.

As recorded, it first stopped any currently serving experiment with the exact overlay
that launched it, returned to the E03 defaults and verified both functional gates plus
`./scripts/check-f0.py` before beginning this window. Do not carry forward an older
C4 overlay, its 12 GiB KV pool, or its connector.

```sh
TP4_ENV=<currently-serving-overlay> ./scripts/tp4ctl down
unset TP4_ENV
./scripts/deploy.sh
./scripts/tp4ctl fabric-check
./scripts/tp4ctl up
# Complete both functional gates, then run ./scripts/check-f0.py.
```

Start C5 as one coordinated four-rank transition:

```sh
./scripts/tp4ctl down
export TP4_ENV=scripts/node/experiments/e03-c5/delta.env
./scripts/deploy.sh
./scripts/tp4ctl fabric-check
./scripts/tp4ctl up
# Complete both functional gates within two minutes of /health 200.
```

Inspect all four container commands. Require `--max-num-seqs 5`,
`--max-model-len 262144`, and exactly one
`--kv-cache-memory-bytes=16106127360`, together with the normal E03 image, replay
connector, mHC and draft-budget signatures. If this window calls for a bounded Rigmark
functionality probe, run one native execution with `--skip-prefill --skip-concurrency
--runs 1` and a unique absolute `--output`. This is one request for each of the three
decode workloads. Inspect the saved receipt and stop; do not continue into a complete suite,
repeat series or promotion decision. The default `check-f0.py` identity intentionally
pins the frozen max-six performance reference and rejects this delta. Do not weaken it
or publish a replacement baseline; keep any max-sequence-only operational identity
receipt private and explicitly separate from performance evidence.

If C5 fails, take the whole stack down with the C5 overlay still selected, then start
the four-sequence fallback as another coordinated transition:

```sh
./scripts/tp4ctl down
export TP4_ENV=scripts/node/experiments/e03-c5/fallback-max4.env
./scripts/deploy.sh
./scripts/tp4ctl fabric-check
./scripts/tp4ctl up
# Repeat both functional gates; require the same signatures with --max-num-seqs 4.
```

Keep the active overlay selected for status, verification and down. Before leaving C5
unattended, configure rank-0 autostart with
`Environment=TP4_ENV=scripts/node/experiments/e03-c5/delta.env` and run
`systemctl daemon-reload`. When selecting the fallback, replace that drop-in value with
`fallback-max4.env` during the same coordinated transition; do not leave autostart on
C5 or the unmodified max-six base. To return to accepted E03 defaults, stop with the
active overlay, unset `TP4_ENV`, deploy, start once, remove the autostart overlay
selection, run `systemctl daemon-reload`, and repeat both gates plus
`./scripts/check-f0.py`. Never repair or restart only one rank.

## Recovery and rollback

Begin with read-only status and choose the narrowest matching rollback. Every restart
below is full-cluster and must fall within an authorized service window.

| Condition | Recovery | Verification |
| --- | --- | --- |
| Overlay result is bad | `./scripts/tp4ctl restart` with no `TP4_ENV` | base `cluster.env` signatures and gates return |
| Production engine knob is bad | restore the rollback documented beside the value in `cluster.env.example`, update local `cluster.env`, deploy, restart | all runtime signatures plus task gate |
| Return the lm_head to BF16 (E35) | the [E35 return](#return-to-e35) | no `e36-lm-head-w8a16` mount, `VLLM_E36_` variable or `E36_` log line on any rank; the E35 runner mounted; `./scripts/check-f0.py --identity docs/operational-identities/2026-09-30-e35-return.json` with the same overlay, both gates |
| Return the verify length to E31-MB | at once: overwrite `~/tp4/experiments/e03/e35-runner-k/policy.flag` on rank 0 in place with `ema`; persistent: the [E31-MB return](#return-to-e31-mb) | at once: the next `E35_RUNNER_K` line reports `policy=ema` and `check-f0` reports the changed policy file; persistent: no `e35-runner-k` mount, `VLLM_E35_` variable or `E35_` log line on any rank; `./scripts/check-f0.py --identity docs/operational-identities/2026-09-30-e31-mb-return.json` with the same overlay, both gates |
| Remove the SparkCache disk limit | use the [memory-bounded return](#remove-the-sparkcache-disk-limit) | no `SPARK_CONTEXT_CACHE_` variable on any rank, `max_bytes=0` in each rank's `sparkcache: config` line, every other memory-bounded signature retained; `./scripts/check-f0.py --identity docs/operational-identities/2026-09-30-memory-bounded-return.json` with the same overlay, both gates |
| Restore measured E31 cache behavior | use the [complete E31 rollback](#restore-the-measured-e31-cache-behavior) | E31 engine identity retained; previous replay connector/config/cache namespace restored; `./scripts/check-f0.py --baseline docs/historical_benchmarks/baselines/2026-09-28-e31/baseline.json` with the same overlay, both gates |
| Restore the previous E29 reference | use the [complete E29 rollback](#restore-the-previous-e29-reference) | production `pooled_indexer.py` and `ops/glm_kpool.py` mounted, no `VLLM_GLM53_` variable, no `E31_` log line, E29 scheduler and engine core retained, same cache namespace; `./scripts/check-f0.py --baseline docs/historical_benchmarks/baselines/2026-09-25-e29/baseline.json` with the same overlay, both gates |
| Restore the older E28b reference | use the [complete E28b rollback](#restore-the-older-e28b-reference) | E27c scheduler mounted over `vllm/v1/core/sched/scheduler.py`, no engine-core mount, no `VLLM_E29_` flag or boot line, seven draft tokens and 16 GiB KV retained, same cache namespace; `./scripts/check-f0.py --baseline docs/historical_benchmarks/baselines/2026-09-25-e28b/baseline.json` with the same overlay, both gates |
| Restore the older E27c reference | use the [complete E27c rollback](#restore-the-older-e27c-reference) | `num_spec_tokens=5`, no `VLLM_ADAPTIVE_K_HI` or `--compilation-config`, `--kv-cache-memory-bytes=16106127360`, same scheduler mount and cache namespace; `./scripts/check-f0.py --baseline docs/historical_benchmarks/baselines/2026-09-25-e27c/baseline.json` with the same overlay, both gates |
| Restore the older E27 reference | use the [complete E27 rollback](#restore-the-older-e27-reference) | no scheduler mount over `vllm/v1/core/sched/scheduler.py`, no `VLLM_E27B_`/`VLLM_E27C_` flag or boot line, `--prefill-schedule-interval 8` retained, same cache namespace; `./scripts/check-f0.py --baseline docs/historical_benchmarks/baselines/2026-09-24-e27/baseline.json` with the same overlay, both gates |
| Restore the older E22b reference | use the [complete E22b rollback](#restore-the-older-e22b-reference) | no `--prefill-schedule-interval` in any rank's command, same mounts, flags and cache namespace; `./scripts/check-f0.py --baseline docs/historical_benchmarks/baselines/2026-09-23-e22b/baseline.json` with the same overlay, both gates |
| Restore the older E21 reference | use the [complete E21 rollback](#restore-the-older-e21-reference) | vendor drafter, no E22 mount, flags or boot signature, E21 cache namespace; `./scripts/check-f0.py --baseline docs/historical_benchmarks/baselines/2026-09-23-e21/baseline.json` with the same overlay, both gates |
| Restore the E03 reference | use the [complete E03 rollback](#restore-the-previous-e03-reference) | E03 hook source and cache namespace, no E21 mount, flag or boot signature; `./scripts/check-f0.py --baseline docs/historical_benchmarks/baselines/2026-09-19-e03/baseline.json` with the same overlay, both gates |
| Restore the earlier September 19 base | use the [complete September 19 rollback](#restore-the-previous-september-19-base) | original scheduler/model/connector/cache, mHC disabled, 15 GiB KV; historical identity check and both gates |
| Restore the September 18 baseline | use the complete [September 18 overlay](#restore-the-september-18-baseline) for deploy and the coordinated transition | 16 GiB KV, original KDA model and connector, original cache namespace, no hybrid helper/encoder/probe mounts; `./scripts/check-f0.py --baseline docs/historical_benchmarks/baselines/2026-09-18/baseline.json` with the same overlay, both gates |
| Restore the September 11 baseline | use the [frozen archive restore](#restore-from-the-frozen-previous-archive): verify the archive and run its prepared restore plan, including the original base and overlay | `PATCH_FILE` mount present, no `--kv-transfer-config`, `mode=per-request`, automatic GID selection, generated identity-check command using the archived baseline path PASS, both gates |
| Model revision is bad | restore the previous pinned revision and manifest named beside `MODEL_REV`, deploy fetch tooling, rerun the manifest fetch and `verify-node.sh --full-model`, then restart | identical revision markers and complete hashes on all ranks |
| Adaptive scheduler must be removed | apply the coupled rollback beside its settings: scheduler flag, mount, policy env, speculative length/table; preserve the MoE mount | no adaptive line, intended fixed-k init, MoE config still loaded |
| Tuned MoE config must be removed | remove only its mount; preserve scheduler entries | expected default-MoE line, Triton backend and adaptive scheduler remain |
| Triton MoE backend must be removed | remove only `--moe-backend triton` and the tuned MoE mount; preserve the adaptive scheduler flag, mount, and policy variables | engine selects its default MoE backend and the adaptive signature remains |
| IOMMU passthrough must be reverted | run `./scripts/tp4ctl down` before `./scripts/deploy-host.sh --run tp4-iommu.sh --revert`, then reboot ranks 3→2→1→0; after rank 0, wait for any autostart already in progress and do not issue a duplicate `up` (see the [boot sequence](install-from-zero.md#3-audit-and-bootstrap-the-hosts)) | status reports translated mode; fabric remains green |
| Kernel or boot tier must be reverted | run `./scripts/tp4ctl down`, select the previously installed GRUB entry without purging the current kernel, then reboot ranks 3→2→1→0 with rank 0 last | `uname -r` reports the intended kernel on all ranks; static verification, jumbo pings, Ethernet speed, RDMA port state, and HCA/GID selection all pass before serving |
| Patched NCCL file drifted | reinstall atomically with `scripts/node/nccl/install-nccl.sh` | SHA matches on every rank, then fabric-check and full restart |

An IOMMU revert exit code 4 means GRUB was not safely regenerated: do not reboot.
Never use `EXTRA_DOCKER_ENV=""` as a generic rollback. Never purge a model to recover
space without a fresh disk census and explicit owner decision.

### Return to E35

Use [`operational-20260930-e35.env`](../scripts/node/reference/operational-20260930-e35.env) when
the INT8 `lm_head` itself must be withdrawn. It swaps the E36 runner back to the E35 runner
and removes exactly the E36 module mount and its two variables; it refuses any other E36
form. In an authorized window, stop with the recipe that is serving, then select the return
for deploy, every lifecycle command and the identity check:

```sh
export TP4_ENV=scripts/node/reference/operational-20260930-e35.env
./scripts/deploy.sh
./scripts/tp4ctl fabric-check
./scripts/tp4ctl up
# Complete both functional gates within two minutes of /health 200.
./scripts/check-f0.py --identity docs/operational-identities/2026-09-30-e35-return.json
```

Persist the selection for autostart with the drop-in procedure below, using
`60-e35-return.conf` and this overlay path instead of the protected16 values.

### Return to E31-MB

The E35 policy file offers an immediate return without a restart. On rank 0, overwrite the
deployed file in place, keeping the same file so the running container sees the change:

```sh
printf 'ema\n' > ~/tp4/experiments/e03/e35-runner-k/policy.flag
```

Within 0.5 s the verify length follows the acceptance average again. Rank 0 still broadcasts
the unchanged choice, and `check-f0` reports the changed policy file until the next deploy
restores `hybrid`. For a persistent return, use
[`operational-20260930-e31-mb.env`](../scripts/node/reference/operational-20260930-e31-mb.env).
It swaps the E35 scheduler copy back to the draft-budget scheduler and removes exactly the E36
runner and module mounts, the E35 speculator and policy mounts and the four `VLLM_E35_` and
`VLLM_E36_` variables; it refuses any other E35 or E36 form. In an authorized window, stop with the recipe that is serving, then select the
return for deploy, every lifecycle command and the identity check:

```sh
export TP4_ENV=scripts/node/reference/operational-20260930-e31-mb.env
./scripts/deploy.sh
./scripts/tp4ctl fabric-check
./scripts/tp4ctl up
# Complete both functional gates within two minutes of /health 200.
./scripts/check-f0.py --identity docs/operational-identities/2026-09-30-e31-mb-return.json
```

Persist the selection for autostart with the drop-in procedure below, using
`60-e31-mb-return.conf` and this overlay path instead of the protected16 values.

### Remove the SparkCache disk limit

Use [`operational-20260929-memory-bounded.env`](../scripts/node/reference/operational-20260929-memory-bounded.env)
only when the disk-capacity policy itself must be withdrawn. It removes E36 and E35 (as the
E31-MB return does) and exactly the two `SPARK_CONTEXT_CACHE_` capacity assignments of the default,
and keeps every other memory-bounded setting; without a limit the store grows until the disk is full, so monitor
free space on every rank. The overlay refuses a base without exactly the E36 and E35
selections and that pair. In an
authorized window, stop with the recipe that is serving, then select the return for deploy,
every lifecycle command and the identity check:

```sh
export TP4_ENV=scripts/node/reference/operational-20260929-memory-bounded.env
./scripts/deploy.sh
./scripts/tp4ctl fabric-check
./scripts/tp4ctl up
# Complete both functional gates within two minutes of /health 200.
./scripts/check-f0.py --identity docs/operational-identities/2026-09-30-memory-bounded-return.json
```

Before leaving it unattended, persist the selection for autostart with the drop-in
procedure below, using `60-memory-bounded-return.conf` and this overlay path instead of the
protected16 values. Never keep two drop-ins that set `TP4_ENV`.

### Restore the protected 16 GiB operational predecessor

Use [`operational-20260929-sparkcache-protected.env`](../scripts/node/reference/operational-20260929-sparkcache-protected.env)
to remove bounded admission, allocator trim and the 6,912-token step cap while retaining
the protected SparkCache namespace, 262,144-token context, six-sequence scheduler limit and
the complete E31 engine lineage. It restores the 16 GiB KV pool. In an authorized window,
first stop the cluster with the exact recipe that is serving. Then select the rollback for
deploy, every lifecycle command, and the explicit historical operational identity check:

```sh
export TP4_ENV=scripts/node/reference/operational-20260929-sparkcache-protected.env
./scripts/deploy.sh
./scripts/tp4ctl fabric-check
./scripts/tp4ctl up
# Complete both functional gates within two minutes of /health 200.
./scripts/check-f0.py --identity docs/operational-identities/2026-09-29-sparkcache-protected.json
```

Keep the overlay selected for every lifecycle command. Before leaving this rollback
unattended, inspect `systemctl cat tp4-autostart.service` and the effective environment on
rank 0. Use a distinct
`/etc/systemd/system/tp4-autostart.service.d/60-protected16-rollback.conf` containing exactly:

```ini
[Service]
Environment=TP4_ENV=scripts/node/reference/operational-20260929-sparkcache-protected.env
```

Do not install it alongside an active `60-e31-rollback.conf` or another drop-in that sets
`TP4_ENV`. Before changing any file, inspect `systemctl cat tp4-autostart.service` and stop
if an unrecognized drop-in sets `TP4_ENV`. Validate every present known rollback file before
mutating either one, then remove the E31 selection and create the protected16 selection with
one reload at most:

```sh
protected_dropin=/etc/systemd/system/tp4-autostart.service.d/60-protected16-rollback.conf
e31_dropin=/etc/systemd/system/tp4-autostart.service.d/60-e31-rollback.conf
sudo install -d -m 0755 /etc/systemd/system/tp4-autostart.service.d
if sudo test -e "$protected_dropin"; then
  printf '%s\n' '[Service]' \
    'Environment=TP4_ENV=scripts/node/reference/operational-20260929-sparkcache-protected.env' \
    | sudo cmp -s - "$protected_dropin" || exit 1
fi
if sudo test -e "$e31_dropin"; then
  printf '%s\n' '[Service]' \
    'Environment=TP4_ENV=scripts/node/reference/baseline-20260928-e31.env' \
    | sudo cmp -s - "$e31_dropin" || exit 1
fi
unit_changed=0
if sudo test -e "$e31_dropin"; then
  sudo rm -- "$e31_dropin"
  unit_changed=1
fi
if ! sudo test -e "$protected_dropin"; then
  printf '%s\n' '[Service]' \
    'Environment=TP4_ENV=scripts/node/reference/operational-20260929-sparkcache-protected.env' \
    | sudo tee "$protected_dropin" >/dev/null
  sudo chmod 0644 "$protected_dropin"
  unit_changed=1
fi
if [ "$unit_changed" = 1 ]; then
  sudo systemctl daemon-reload
fi
```

Preserve every unrelated directive and drop-in. Require
`systemctl show tp4-autostart.service -p Environment --value` to contain exactly one
`TP4_ENV` assignment, selecting the protected16 overlay above.

Returning to the memory-bounded default requires a coordinated stop with the protected16
overlay. Before any mutation, inspect `systemctl cat tp4-autostart.service` and
`systemctl show tp4-autostart.service -p Environment --value`; abort if the E31 file, an
unrecognized `TP4_ENV` source or any conflicting effective assignment is present. Use this
separate non-creating removal block; never reuse the creation block:

```sh
protected_dropin=/etc/systemd/system/tp4-autostart.service.d/60-protected16-rollback.conf
e31_dropin=/etc/systemd/system/tp4-autostart.service.d/60-e31-rollback.conf
if sudo test -e "$e31_dropin"; then
  echo "refusing protected16 removal while the E31 rollback drop-in exists" >&2
  exit 1
fi
if sudo test -e "$protected_dropin"; then
  printf '%s\n' '[Service]' \
    'Environment=TP4_ENV=scripts/node/reference/operational-20260929-sparkcache-protected.env' \
    | sudo cmp -s - "$protected_dropin" || exit 1
  sudo rm -- "$protected_dropin"
  sudo systemctl daemon-reload
fi
```

If the file was absent, change no unit file and do not run `daemon-reload`. Require
`systemctl show tp4-autostart.service -p Environment --value` to have no `TP4_ENV`; an E31
or unrecognized assignment is a conflict, not something to remove by guess. Then deploy/start
without `TP4_ENV`, complete both functional gates and run the default `./scripts/check-f0.py`.

### Restore the measured E31 cache behavior

Use the complete
[`baseline-20260928-e31.env`](../scripts/node/reference/baseline-20260928-e31.env) when the
protected connector must be rolled back. It retains the E31 indexer, E29 scheduler,
16 GiB KV pool, 262,144-token context and every other engine setting, while restoring the
previous replay connector, E22b transfer config and cache namespace.

In an authorized window, stop with the recipe that is serving, then use the rollback for
deploy and every lifecycle command:

```sh
export TP4_ENV=scripts/node/reference/baseline-20260928-e31.env
./scripts/deploy.sh
./scripts/tp4ctl fabric-check
./scripts/tp4ctl up
# Complete both functional gates within two minutes of /health 200.
./scripts/check-f0.py --baseline docs/historical_benchmarks/baselines/2026-09-28-e31/baseline.json
```

Keep the overlay selected for every lifecycle command. Before leaving the rollback
unattended, create
`/etc/systemd/system/tp4-autostart.service.d/60-e31-rollback.conf` on rank 0 with exactly:

```ini
[Service]
Environment=TP4_ENV=scripts/node/reference/baseline-20260928-e31.env
```

Do not install this file while `60-protected16-rollback.conf` or another drop-in sets
`TP4_ENV`; transition between rollback overlays only after checking and removing the old
file's exact documented content. Preserve unrelated directives and drop-ins.
Run `sudo systemctl daemon-reload` only when a rollback drop-in was created, replaced or
removed; an already exact E31 file requires no reload. Then require
`systemctl show tp4-autostart.service -p Environment --value` to contain exactly the
`TP4_ENV=scripts/node/reference/baseline-20260928-e31.env` assignment. Returning to the
memory-bounded default requires a coordinated stop with this rollback overlay. Verify that the
named file, when present, still has exactly the two lines above, remove only that file and run
`sudo systemctl daemon-reload` only after removal. If it was absent, change no unit file and
do not reload. Require the assignment to be absent from `systemctl show`.
Then deploy/start without `TP4_ENV`, complete both functional gates and run the default
`./scripts/check-f0.py` check.

### Restore the previous E29 reference

The immediate complete return is
[`baseline-20260925-e29.env`](../scripts/node/reference/baseline-20260925-e29.env). It
mounts the production `pooled_indexer.py` and `ops/glm_kpool.py` again and removes the two
E31 flag-file variables; every other mount, flag and the cache namespace are unchanged, so
the same persistent cache is reused.

Without a restart, writing `0` to `/tmp/glm53-kpool-tail-ring` in the container on all four
ranks returns the pool arithmetic to the legacy four-slot tail (E29 behaviour, including its
speculative-decoding defect). Do it only with running and waiting requests at 0 on
`/metrics`, as described in
[`experiments/e03/e31-indexer/README.md`](../scripts/node/experiments/e03/e31-indexer/README.md).
The running identity is still E31 until the complete return below is deployed and started.

In the authorized window, first stop using the currently serving recipe. Then:

```sh
export TP4_ENV=scripts/node/reference/baseline-20260925-e29.env
./scripts/deploy.sh
./scripts/tp4ctl fabric-check
./scripts/tp4ctl up
# Complete both functional gates within two minutes of /health 200.
./scripts/check-f0.py --baseline docs/historical_benchmarks/baselines/2026-09-25-e29/baseline.json
```

Keep this overlay for subsequent lifecycle commands and configure autostart to use it
before an unattended reboot. Returning to current defaults requires a coordinated stop
with the rollback overlay, then deploy/start without it and removal of that autostart
selection.

### Restore the older E28b reference

The complete return is
[`baseline-20260925-e28b.env`](../scripts/node/reference/baseline-20260925-e28b.env). It
restores the E29 return with the E27c scheduler instead of the E29 one, without the
engine-core mount and the three `VLLM_E29_` flags. It keeps seven draft tokens, the 16 GiB KV
pool, the other mounts and the cache namespace, so the same persistent cache is reused.

In the authorized window, first stop using the currently serving recipe. Then:

```sh
export TP4_ENV=scripts/node/reference/baseline-20260925-e28b.env
./scripts/deploy.sh
./scripts/tp4ctl fabric-check
./scripts/tp4ctl up
# Complete both functional gates within two minutes of /health 200.
./scripts/check-f0.py --baseline docs/historical_benchmarks/baselines/2026-09-25-e28b/baseline.json
```

Keep this overlay for subsequent lifecycle commands and configure autostart to use it
before an unattended reboot. Returning to current defaults requires a coordinated stop
with the rollback overlay, then deploy/start without it and removal of that autostart
selection.

### Restore the older E27c reference

The complete return is
[`baseline-20260925-e27c.env`](../scripts/node/reference/baseline-20260925-e27c.env). It
restores the E28b return with five draft tokens and the 15 GiB KV pool, and removes
`VLLM_ADAPTIVE_K_HI=7` and the CUDA graph limit; the scheduler, other mounts and the cache
namespace are unchanged, so the same persistent cache is reused.

In the authorized window, first stop using the currently serving recipe. Then:

```sh
export TP4_ENV=scripts/node/reference/baseline-20260925-e27c.env
./scripts/deploy.sh
./scripts/tp4ctl fabric-check
./scripts/tp4ctl up
# Complete both functional gates within two minutes of /health 200.
./scripts/check-f0.py --baseline docs/historical_benchmarks/baselines/2026-09-25-e27c/baseline.json
```

Keep this overlay for subsequent lifecycle commands and configure autostart to use it
before an unattended reboot. Returning to current defaults requires a coordinated stop
with the rollback overlay, then deploy/start without it and removal of that autostart
selection.

### Restore the older E27 reference

The complete return is
[`baseline-20260924-e27.env`](../scripts/node/reference/baseline-20260924-e27.env). It
restores the E27c return and also removes the E27c scheduler mount and its two flags; the
prefill cadence, other mounts and the cache namespace are unchanged, so the same persistent
cache is reused.

In the authorized window, first stop using the currently serving recipe. Then:

```sh
export TP4_ENV=scripts/node/reference/baseline-20260924-e27.env
./scripts/deploy.sh
./scripts/tp4ctl fabric-check
./scripts/tp4ctl up
# Complete both functional gates within two minutes of /health 200.
./scripts/check-f0.py --baseline docs/historical_benchmarks/baselines/2026-09-24-e27/baseline.json
```

Keep this overlay for subsequent lifecycle commands and configure autostart to use it
before an unattended reboot. Returning to current defaults requires a coordinated stop
with the rollback overlay, then deploy/start without it and removal of that autostart
selection.

### Restore the older E22b reference

The complete return is
[`baseline-20260924-e22b.env`](../scripts/node/reference/baseline-20260924-e22b.env). It
removes the E27 prefill cadence and the E27c scheduler; the other mounts, flags and the
cache namespace are unchanged, so the same persistent cache is reused.

In the authorized window, first stop using the currently serving recipe. Then:

```sh
export TP4_ENV=scripts/node/reference/baseline-20260924-e22b.env
./scripts/deploy.sh
./scripts/tp4ctl fabric-check
./scripts/tp4ctl up
# Complete both functional gates within two minutes of /health 200.
./scripts/check-f0.py --baseline docs/historical_benchmarks/baselines/2026-09-23-e22b/baseline.json
```

Keep this overlay for subsequent lifecycle commands and configure autostart to use it
before an unattended reboot. Returning to current defaults requires a coordinated stop
with the rollback overlay, then deploy/start without it and removal of that autostart
selection.

### Restore the older E21 reference

The complete return is
[`baseline-20260923-e21.env`](../scripts/node/reference/baseline-20260923-e21.env). It
restores the vendor DFlash2 drafter and the E21 cache namespace and removes the E22
override, module and flags; every other engine and container argument is unchanged. Keep
both cache directories available; neither recipe reuses the other's cache.

In the authorized window, first stop using the currently serving recipe. Then:

```sh
export TP4_ENV=scripts/node/reference/baseline-20260923-e21.env
./scripts/deploy.sh
./scripts/tp4ctl fabric-check
./scripts/tp4ctl up
# Complete both functional gates within two minutes of /health 200.
./scripts/check-f0.py --baseline docs/historical_benchmarks/baselines/2026-09-23-e21/baseline.json
```

Keep this overlay for subsequent lifecycle commands and configure autostart to use
it before an unattended reboot. Returning to current defaults requires a coordinated
stop with the rollback overlay, then deploy/start without it and removal of that
autostart selection.

### Restore the previous E03 reference

The immediate complete return is
[`baseline-20260919-e03.env`](../scripts/node/reference/baseline-20260919-e03.env). It
restores the E03 hook source and E03 cache namespace and removes the E21 module and
flag; every other engine and container argument is unchanged. Keep both cache
directories available; neither recipe reuses the other's cache.

In the authorized window, first stop using the currently serving recipe. Then:

```sh
export TP4_ENV=scripts/node/reference/baseline-20260919-e03.env
./scripts/deploy.sh
./scripts/tp4ctl fabric-check
./scripts/tp4ctl up
# Complete both functional gates within two minutes of /health 200.
./scripts/check-f0.py --baseline docs/historical_benchmarks/baselines/2026-09-19-e03/baseline.json
```

Keep this overlay for subsequent lifecycle commands and configure autostart to use
it before an unattended reboot. Returning to current defaults requires a coordinated
stop with the rollback overlay, then deploy/start without it and removal of that
autostart selection.

### Restore the previous September 19 base

This older complete return is
[`baseline-20260919.env`](../scripts/node/reference/baseline-20260919.env). It disables
mHC sharding, restores the frozen adaptive scheduler and previous connector, and selects
the previous cache namespace while retaining hybrid KDA, 15 GiB KV and the context limit.
Keep both connectors, every source and both cache directories available.

In the authorized window, first stop using the currently serving recipe. Then:

```sh
export TP4_ENV=scripts/node/reference/baseline-20260919.env
./scripts/deploy.sh
./scripts/tp4ctl fabric-check
./scripts/tp4ctl up
# Complete both functional gates within two minutes of /health 200.
./scripts/check-f0.py --baseline docs/historical_benchmarks/baselines/2026-09-19/baseline.json
```

Keep this overlay for subsequent lifecycle commands and configure autostart to use
it before an unattended reboot. Returning to current defaults requires a coordinated
stop with the rollback overlay, then deploy/start without it and removal of any
rollback autostart selection. Neither direction deletes or reuses the other cache.

### Restore the September 18 baseline

This older rollback is the complete non-site overlay
[`baseline-20260918.env`](../scripts/node/reference/baseline-20260918.env). It restores
the original model module and cache namespace, the 16 GiB pool, and the original
connector. The image, scheduler, SIRCL transport and checkpoint revisions are unchanged.
Prepare the original connector as `spark_context_cache_connector-20260918.py` with
[`prepare-sparkcache.py`](../scripts/prepare-sparkcache.py) and stage it beside the
current payload on all four ranks before the maintenance window. Deployment installs
the frozen model and cache JSON under `~/tp4/reference/`.

Within an authorized window, stop using the currently serving delta (if any), then
select the rollback for all subsequent commands:

```sh
# First down uses the currently serving TP4_ENV, or none for the base recipe.
./scripts/tp4ctl down
export TP4_ENV=scripts/node/reference/baseline-20260918.env
./scripts/deploy.sh
./scripts/tp4ctl up
# Complete both post-boot functional gates within two minutes of /health 200.
./scripts/check-f0.py --baseline docs/historical_benchmarks/baselines/2026-09-18/baseline.json
```

Keep this overlay for later lifecycle commands. For unattended autostart, install a
systemd drop-in that sets this same `TP4_ENV` and run `systemctl daemon-reload`; otherwise
a later reboot would select the protected operational default. Returning to the current defaults
requires a coordinated transition with no overlay and removal of that drop-in. Preserve
both persistent-cache directories; do not reuse their contents across quantization
recipes. The frozen baseline medians are not remeasured as part of rollback.

### Restore from the frozen previous archive

Treat runtime restoration and host restoration as separate decisions. For runtime September 11:

1. select one archive, verify its SHA manifest offline, compare it live, and confirm the
   four intended targets plus image, model, drafter and exact NCCL availability;
2. review observed paths, permissions, owners and symlinks against intended destinations;
3. in a future authorized window, stage the archived controller as a regular file and
   verify its hash before one coordinated four-rank `down` using the overlay that launched
   the current process;
4. restore only the archived repository-managed runtime assets through `scripts/deploy.sh`
   and the NCCL atomic installer, checking every destination hash;
5. render the prepared autostart drop-in with the deployment user. It explicitly selects
   the September 11 overlay and its verified reference controller for both start and stop. Install it
   only after review and run `systemctl daemon-reload`; do not treat the previously loaded
   unit as if it already contained the disk changes;
6. require two addressed MTU-9000 ports per rank, all eight jumbo pings, Ethernet speed,
   RDMA/HCA/GID checks and absence of a second stack or foreign GPU work; then run one
   coordinated four-rank `up`;
7. from the first `/health` 200, run both functional gates above within 120 seconds and
   finish with the September 11 operational check. A failed gate requires full-cluster stop and a
   report, never single-rank repair.

The captured host state is comparison evidence, not an unattended host restore program.
Do not apply netplan, flush the global firewall, activate NetworkManager profiles, change
devlink/eSwitch/TC/offloads, packages, drivers, NIC firmware, kernel, IOMMU, GRUB or boot
parameters as part of runtime rollback. Preserve management, Tailscale, SSH and unrelated
configuration. Any host or fabric change needs its own authorization, an exact owned-file
diff, and verified independent recovery access or the owner's physical availability. If a
reboot is later approved, stop all four ranks first and reboot rank 0 last.

The archive records what was observed and what the prepared September 11 restore would install.
Creating and verifying it does not rehearse a full restore and does not prove unattended
recovery from kernel, driver, firmware, boot-loader or network failure.

## Keep an accepted change

For an explicitly authorized exclusive production load and fault window, use the
[resilience campaign runbook](resilience.md). Its temporary overlay confines cache
faults to a separate namespace, retains private receipts and restores the protected
default through the same coordinated controller. Performance checks still use native
Rigmark from its independent checkout.

After the owner accepts a recipe change, persist the value and rollback in its source
file, update any affected runtime signature and `CHANGELOG.md`, and run
`./scripts/check.sh`. Run native Rigmark from its independent checkout. Raw logs,
payloads, site configuration and working records stay private; a sanitized frozen
baseline record may be published under `docs/` with dates, counts and evidence hashes.

Do not expose node addresses, private paths, or logs in public documents. A commit,
tag, release, or public announcement remains a separate explicit action.
