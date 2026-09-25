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
| E28b draft length and KV pool | rank-0 log, `docker inspect`, `./scripts/check-f0.py` | `num_spec_tokens=7` and `cudagraph_capture_sizes` ending at 72 in engine initialization, `engine_k=7` in the draft-budget line, `VLLM_ADAPTIVE_K_HI=7` and `--compilation-config={"max_cudagraph_capture_size":72}` on every rank, `--kv-cache-memory-bytes=17179869184`, and `GPU KV cache size: 1,365,066 tokens` (5.21x at 262,144) |
| SparkCache lane | `docker inspect`, `./scripts/verify-node.sh` | `--kv-transfer-config` naming `SparkContextCacheConnector` in the command, entrypoint `/opt/sircl-serving/entrypoint.sh`, the connector, encoder, engine override and SIRCL mounts present, `sparkcache payload` and `sircl payload` rows PASS; `docker ps` shows no health state because the lane runs with `--no-healthcheck` |
| Hybrid KDA and KV pool | rank logs, `docker inspect`, `./scripts/check-f0.py` | `E20_KDA_INPUT_W8A16_READY` on every rank: 34 modules, group 128, padded N=6400, threshold 2048, shared scratch 51,515,392 bytes; `--kv-cache-memory-bytes=16106127360`; matching source hashes and `E20_MEMORY_PROBE` records |
| E21 residual projections | rank logs, `docker inspect`, `./scripts/check-f0.py` | `E21_BF16_RESIDUE_W8A16_READY` on every rank: 67 modules in the `kda_o_proj`, `mla_fused_qkv_a_proj`, `mla_o_proj` and `mla_q_b_proj` families, `mla_layout=q_lora`, group 128, threshold 2048, added scratch 79,691,776 bytes; `VLLM_E21_BF16_RESIDUE_W8A16=1` and matching `experiments/e03/bf16-residue/` source hashes |
| E22b drafter conversion | rank logs, `docker inspect`, `./scripts/check-f0.py` | `E22_DRAFTER_W8A16_READY` on every rank: 30 modules in the `down_proj`, `gate_up_proj`, `kernel_projection`, `o_proj` and `qkv_proj` families, `context_kv_w8a16` false, 0 added scratch bytes; `VLLM_E22_DRAFTER_W8A16=1`, `VLLM_E22_CONTEXT_KV_W8A16=0` and matching `experiments/e03/drafter-w8a16/` source hashes; the drafter CUDA graphs are captured fresh at every boot |
| E27 prefill cadence | `docker inspect`, `./scripts/check-f0.py` | `--prefill-schedule-interval 8` exactly once in every rank's command; no boot log line, because the argument is native to the image |
| E27c scheduler | rank-0 logs, `docker inspect`, `./scripts/check-f0.py` | `E27C_CADENCE_WHEN_QUEUED_READY interval=8` and `E27B_SHORT_PREFILL_READY tokens=2048 interval=8` on rank 0; `VLLM_E27B_SHORT_PREFILL_TOKENS=2048`, `VLLM_E27C_CADENCE_WHEN_QUEUED=1` and the E29 scheduler below (which contains the E27c patch) mounted over `vllm/v1/core/sched/scheduler.py` on every rank; `E27C_CADENCE_KEPT_WHILE_QUEUED` appears the first time long prompts are queued while others decode |
| E29 end-drain and idle coalescing | rank-0 logs, `docker inspect`, `./scripts/check-f0.py` | `E29_END_DRAIN_READY trace=0` and `E29_IDLE_COALESCE_READY ms=4 trace=0` on rank 0; `VLLM_E29_END_DRAIN=1`, `VLLM_E29_IDLE_COALESCE_MS=4` and `VLLM_E29_TRACE=0` on every rank, with the `experiments/e03/end-drain/scheduler.py` hash mounted over `vllm/v1/core/sched/scheduler.py` and the `experiments/e03/end-drain/core.py` hash over `vllm/v1/engine/core.py`; the first request held at a length finish logs `E29_END_DRAIN_HELD` |

The default [accepted E29 recipe](benchmarks/baselines/2026-09-25-e29.md) retains those
signatures, including the E03 `SPARK_MHC_PREFILL_SHARD=1` and its measured source
mounts, the E21 residual projections, the E22b drafter conversion with its own cache
namespace, the E27 prefill cadence, the E27c scheduler patch, and seven draft tokens for a
single request with a 16 GiB KV pool (E28b). E29 adds the length-finish hold and the 4 ms
idle-coalescing window. Eligible eager long prefills log
`SPARK_MHC_PREFILL rows=6912 owner_rows=1728 rs=90 ag=95 aux=5` on all four ranks;
this is a workload activation receipt, not a requirement to run extra inference during
an identity-only check. Runtime sources keep their `experiments/e03/` paths to preserve
measured hashes; the E21 hook and module live under `experiments/e03/bf16-residue/`, the
E22b drafter override and module under `experiments/e03/drafter-w8a16/` and the E27c
scheduler under `experiments/e03/queued-cadence/`.
Do not append the historical experiment overlays to the new defaults.

The selected replay connector SHA-256 is
`5893f8747aa093874c46a0185f93c786265d99b5a4cd8da849a7471132422d66`.
The scheduler SHA-256 is
`697f99bf1951535dcc3381776fa744ad8204d83d2606f341b617bac7a31949b4`.
Its rank-0 startup signature is
`draft-budget active=1 source=SchedulerOutput.resolve_num_spec_tokens_to_schedule engine_k=7`,
alongside the enabled batch-uniform async policy. When concurrent requests trigger a
limit, expect `draft-budget first-cap budget=3` and aggregate `limited_requests` and
`trimmed_tokens`. These count placeholder handouts, not accepted tokens or saved compute.
An adaptive-policy fallback invalidates activation.

The complete immediate return is `TP4_ENV=scripts/node/reference/baseline-20260925-e28b.env`:
the E28b recipe with the E27c scheduler and the image's engine core, with the same cache
namespace. `baseline-20260925-e27c.env` restores the E27c recipe with five draft tokens and a
15 GiB KV pool.
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

### Fast baseline operational check

Run the reusable read-only check when a single concise baseline identity and idle
verdict is needed:

```sh
./scripts/check-f0.py
./scripts/check-f0.py --base-url http://127.0.0.1:8000
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
as the effective delta. By default it validates the **September 25 E29** identity from
[the current reference](historical_benchmarks/baselines/2026-09-25-e29/baseline.json), including
seven draft tokens, the adaptive table and high state, the CUDA graph limit, the 16 GiB KV
budget, the E29 scheduler and engine-core hashes, the E27c and E29 flags and rank-0 boot lines, the E27 prefill cadence in the recipe
and every rank's command, the E22b drafter sources/flags/boot signature, the E21 sources/flag/boot signature, the mHC sources/flag, replay connector, draft-budget scheduler/activation, hybrid KDA,
KV byte budget and image content ID. Historical `--baseline` selection also requires
the matching complete runtime recipe. A baseline-changing delta fails the check.

The checker runs bounded SSH probes for all four ranks in parallel with strict host-key checking.
It verifies the effective recipe against the selected baseline's pins, the configured running
container and key command/environment identity, image digest and model marker, the
selected baseline’s runtime source hashes and boot receipts, unfiltered
GPU containers and processes, inactive flusher, addressed MTU-9000 interfaces, all eight
jumbo directions, `/health` 200, and zero running/waiting requests. Stdout is exactly one
concise `F0 CHECK PASS` or `F0 CHECK FAIL` line; details and command errors go to a
mode-0600 JSON report in a new mode-0700 system temporary directory outside the checkout.

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
2. In the same step, remove any autostart overlay selection on rank 0 and run
   `systemctl daemon-reload`, so the next boot selects the same default recipe. A
   frozen experiment overlay may reject the new base, so never leave it selected.
3. Run `./scripts/check-f0.py` and retain its four-rank identity report.

**A different recipe is serving.** Retain the serving `cluster.env` and its exact
overlay until the full-cluster stop, because frozen experiment overlays expect the
previous base. Prepare and verify the payload and the complete rollback first. Then,
in an authorized window:

1. Stop all ranks using the still-active site configuration and serving `TP4_ENV`.
2. Install the prepared site configuration locally, unset `TP4_ENV`, and deploy.
   Remove any obsolete autostart overlay selection and reload systemd.
3. Verify the pinned payloads and all eight jumbo pings; start the four-rank service
   once. Complete both functional gates within two minutes of `/health` 200.
4. Run `./scripts/check-f0.py` and retain its four-rank identity report.

Record the parity and live identity in the promotion record. A reproduction benchmark
runs only on explicit owner request, with the frozen source/settings, fresh salts and
unique absolute outputs, recorded separately from the accepted measurements.

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
| Restore the previous E28b reference | use the [complete E28b rollback](#restore-the-previous-e28b-reference) | E27c scheduler mounted over `vllm/v1/core/sched/scheduler.py`, no engine-core mount, no `VLLM_E29_` flag or boot line, seven draft tokens and 16 GiB KV retained, same cache namespace; `./scripts/check-f0.py --baseline docs/historical_benchmarks/baselines/2026-09-25-e28b/baseline.json` with the same overlay, both gates |
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

### Restore the previous E28b reference

The immediate complete return is
[`baseline-20260925-e28b.env`](../scripts/node/reference/baseline-20260925-e28b.env). It
mounts the E27c scheduler again, removes the engine-core mount and the three `VLLM_E29_`
flags, and keeps seven draft tokens, the 16 GiB KV pool, the other mounts and the cache
namespace, so the same persistent cache is reused.

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
a later reboot would select the accepted E29 default recipe. Returning to the current defaults
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

After the owner accepts a recipe change, persist the value and rollback in its source
file, update any affected runtime signature and `CHANGELOG.md`, and run
`./scripts/check.sh`. Run native Rigmark from its independent checkout. Raw logs,
payloads, site configuration and working records stay private; a sanitized frozen
baseline record may be published under `docs/` with dates, counts and evidence hashes.

Do not expose node addresses, private paths, or logs in public documents. A commit,
tag, release, or public announcement remains a separate explicit action.
