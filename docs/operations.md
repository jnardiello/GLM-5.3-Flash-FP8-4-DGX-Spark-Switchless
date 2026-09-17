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

Then verify five signatures that `docker ps` does not prove:

| Signature | Read-only check | Expected result |
| --- | --- | --- |
| Four-rank identity | `./scripts/tp4ctl status` | the configured `CONTAINER` is `Up` once on every rank |
| Triton MoE configuration | `./scripts/tp4ctl logs` on rank 0 | `Using TRITON Fp8 MoE backend` and `Using configuration from …NVIDIA_GB10…json` |
| Host IOMMU tier | `deploy-host.sh ... tp4-iommu.sh --status` | passthrough on all four ranks, drop-in installed, GRUB synchronized |
| NCCL HCA/GID selection | `./scripts/verify-node.sh` | every configured HCA has an active IPv4-mapped RoCEv2 GID on its addressed fabric netdev; explicit index or automatic `-1` mode is identified |
| Adaptive scheduler | rank-0 log | `AdaptiveKScheduler active (enabled=1 …)` and a `num_speculative_tokens_per_batch_size` table in engine initialization |

For a boot caused by rank-0 autostart, inspect the units too:

```sh
ssh <ALIAS_RANK0> 'systemctl status tp4-autostart tp4-fabric-iptables --no-pager'
```

Expected: one coherent four-rank stack, `/health` 200, green fabric, all signatures,
and no unexpected active `tp4-flusher` after readiness. Stop on any missing signature,
unreachable rank, or health mismatch. `/v1/models` is never a readiness check.
`tp4ctl status` exits nonzero unless it can verify the configured container running on
all four ranks and receive `/health` 200.

### Fast F0 operational check

Run the reusable read-only check when a single concise F0 identity and idle verdict is
needed:

```sh
./scripts/check-f0.py
./scripts/check-f0.py --base-url http://127.0.0.1:8000
```

The second form uses an already-established localhost SSH tunnel when the management LAN
is not directly reachable. The checker reads the local `cluster.env` and honors `TP4_ENV`
as the effective delta. An F0-changing delta fails the check.

The checker runs bounded SSH probes for all four ranks in parallel with strict host-key checking.
It verifies the effective recipe against the frozen F0 pins, the configured running
container and key command/environment identity, image digest and model marker, unfiltered
GPU containers and processes, inactive flusher, addressed MTU-9000 interfaces, all eight
jumbo directions, `/health` 200, and zero running/waiting requests. Stdout is exactly one
concise `F0 CHECK PASS` or `F0 CHECK FAIL` line; details and command errors go to a
mode-0600 JSON report in a new mode-0700 system temporary directory outside the checkout.

PASS covers only those operational checks. It does not hash every model/drafter file,
replay engine async metadata, or send the coherent-response and tool-call requests. Use
`scripts/verify-node.sh --full-model` for full artifact verification and the
[post-boot functional gates](#post-boot-functional-gates) after a changed boot.

### Frozen F0 rollback reference

Before a transport or host experiment, capture the serving four-rank state into a new
private archive outside the checkout. The tracked portable part is
`scripts/node/reference/f0-20260912.env`; it freezes every non-site F0 runtime knob,
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
file, stops all four ranks with the currently serving overlay, then selects F0, deploys
the archived IaC, atomically installs the archived NCCL bytes, and prepares autostart.
The reference overlay is deployed at the same relative path by setting
`TP4_ENV=scripts/node/reference/f0-20260912.env` on `scripts/deploy.sh`.

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
scheduler mount, `PYTHONPATH`, and policy variables. An overlay replaces the complete
value. Preserve every unrelated entry, avoid spaces/globs in values, and never clear
the string while `--scheduler-cls adaptive_k_scheduler.AdaptiveKScheduler` remains in
`EXTRA_VLLM_ARGS`.

Expected: every copied file matches its source, all ranks launch in order 3→2→1→0,
`/health` reaches 200, and the five signatures return. Run the
[post-boot functional gates](#post-boot-functional-gates) within two minutes, followed
by any task-specific verification. Stop the stack immediately if a gate fails.

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
readiness timeout, interruption, or failed final flusher stop triggers a best-effort
full four-rank container teardown and flusher shutdown, then returns failure.

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

### Experimental E09 JJ R10 window

E09 uses a private overlay; it does not change the production image in
`cluster.env.example`. The first candidate must change the engine image while keeping the
current FP8 target and DFlash2 revisions, TP4/DCP1, the 16-GiB `fp8_e4m3` KV pool, block
2304, Triton MoE, adaptive k3/k5 and E07a single-rail sync-prefill SIRCL. Its launcher must
omit the old vLLM-487 sparse-indexer mount, use R10's `--kv-cache-memory-bytes` spelling,
select B12X for sparse MLA while leaving KDA on native selection, disable the distinct
B12X PCIe and FlashInfer all-reduce paths, and reject cache-connector arguments while
SparkCache is OFF. Do not use the packaged NVFP4 or dual-rail/fused SIRCL launchers.

The published registry reference is
`ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:0d4029b3b7023cf32c37ac20279469c9a2ee16a057f25aae3bcfee9ee5fb660f`.
An offline OCI import may have no `RepoDigests`, so the private overlay may use the equally
immutable content reference
`sha256:5e32aaa1bbe3559e81db7706ed4286248f18d27cfdb186f6b851bf786eb43075`.
Before `up`, require `docker image inspect` to return that exact image ID on all four ranks;
never use the synthetic import tag.
For the private OCI archive staging step, use the reviewed one-shot DAC fan-out from rank 0:
copies to ranks 1 and 3 use their direct ring edges, while rank 2 uses rank 0's existing SSH
identity through a `ProxyCommand` over rank 1. Bind each direct hop to its expected fabric
source address and check the destination against its existing management-IP host-key alias.
The three copies may run in parallel, but every selected complete part must pass its pinned
source hash before transfer and its destination hash afterward. Stop and report on a DAC or
identity failure; do not fall back to management data transfer or add host keys, routes or
network configuration.
Use only the regular controller `~/tp4/tp4ctl-e04-window-6cb6f07c`, SHA-256
`6cb6f07cc60a13c86dfed8c01156c13d3f1fb88ca611bd499989841f96688a82`, owned by the
deployment account with mode `0700`; verify it before each `down` and `up`.
Before either lifecycle action, also require every rank's launcher and overlay SHA-256 to
match the reviewed `ARTIFACTS.sha256` in the private E09 workspace.
After readiness, inspect the effective argv and mounts on every rank: target/drafter paths,
TP4, block 2304 and 17179869184-byte cache must match; no old indexer or KV connector may
appear. The rank-0 log must identify FP8 target loading, DFlash2, Triton MoE and
`AdaptiveKScheduler active (enabled=1 … k_lo=3 k_hi=5 … async=True)`. Run the two
functional gates within 120 seconds of `/health` 200.

If the OFF boot fails because R10 propagates the target's `fp8_ds_mla` dtype into the
DFlash2 cache, use a new reviewed overlay rather than modifying the failed attempt. The
retry may add only `"kv_cache_dtype":"fp8_e4m3"` to the existing speculative JSON; R10
applies this field to the drafter cache. Keep the global target `--kv-cache-dtype
fp8_e4m3`, B12X, block 2304, adaptive k3/k5 table, transport and SparkCache-OFF state
unchanged. Require a four-rank dry-run showing that single JSON-field delta and verify its
separate artifact manifest before coordinated lifecycle. Use this retry overlay for its
eventual `down`; a failed retry returns directly to the exact E07a rollback.

Only after this engine-only candidate qualifies may a second overlay enable SparkCache on
the same R10 image and SIRCL path. Qualify SIRCL first, then compare the same cache and
non-transport parameters on NCCL. Before the cache-ON `up`, require the separately reviewed
ON launcher, overlay, connector override and KV-transfer config to match their private
manifest on every rank. The rendered Docker argv must differ from OFF only by one read-only
mount over the image's verified imported connector path and one `--kv-transfer-config`.
When OFF qualification uses the draft-only `fp8_e4m3` retry, the ON overlay and its
separate manifest must inherit that exact speculative field; reject the earlier ON overlay
that lacks it.
Require the startup log to identify `SparkContextCacheConnector`, profile
`glm53-flash-hybrid`, root `/cache/jit/sparkcache-context`, store/restore enabled and both
streaming snapshots and CUDA restore disabled. Its target and draft checkpoint identities
must match the canonical all-four-rank SHA-256 values recorded in the private E09 contract.

The local compatibility override binds every non-empty vLLM `cache_salt` to the external
digest chain without retaining or logging the plaintext; `None` and the empty string retain
legacy keys. Prove one external restore with the same salt after natural GPU-cache pressure.
Use `POST /reset_prefix_cache` with both destructive options omitted only if that router is
already exposed; do not enable a development router for the canary, reset external storage,
delete cache files, or accept a different-salt miss as restore evidence. The three Rigmark
runs use their already frozen distinct salts to prevent cross-run reuse.

A failed boot or gate requires coordinated full-cluster down using the active E09 overlay
and restoration of the exact E07a payload and overlay
from `glm53-e07a-20260912`; verify manifest SHA-256
`626eca7eeb2ec2f39464eb49e8f7d4e409c5b4d4e697e1a9660dcbdd509664fc`, then repeat
`/health` and both gates. This window does not authorize host, network, kernel, firmware or
reboot changes, and does not promote R10.

The actual draft-FP8 retry passed the earlier DFlash2 initialization point on ranks 1-3,
then failed before health during `determine_available_memory` profiling with a propagated
`cutlass_gemm_caller ... Invalid status` error. The retained receipt does not contain the
original operator stack, exact worker rank, layer, or shape, so do not infer them. A later
isolated probe with `KernelConfig(linear_backend="triton")` selected the R10 Triton block
kernel and passed all seven block and online numeric cases with no model weights mounted.

Serving attempt 3 retained the complete attempt-2 recipe and added only
`--linear-backend triton`; it did not use `VLLM_DISABLED_KERNELS` or change image, weights,
KV sizing, transport, attention backend, or cache state. All four ranks loaded target and
draft weights, and the attempt passed the earlier profiling stage. It then failed before
health while `create_kv_cache_views` checked a BLHNC block stride of 14217984 against a
page size of 1179648: manager block 2304 could not split into 36 kernel blocks of 64. The
runtime reported block 64 or `VLLM_KV_CACHE_LAYOUT=LBNHC` as suggestions; this record does
not select either one. No functional POST or Rigmark request ran. This failure is scoped to
the exact attempt-3 recipe and does not establish general R10 or SparkCache incompatibility.
SparkCache ON remains unstarted and blocked. Any later candidate requires a new reviewed
overlay and the normal image, health, functional and startup-signature gates; do not modify
the frozen failed-attempt artifacts. The owned attempt-3 `up` exited 143 after cancellation,
the coordinated four-rank `down` passed, and the post-down receipt verified zero containers,
GPU processes and active flushers on all ranks. E07a restoration completed `up` with exit 0
and health 200. The operator reported that the sandbox denied the first local gate watcher's
network call; its receipt retains the TERM request for the exact PID, but not raw stderr or
an observed exit status. A later un-escalated local reproduction retained `URLError` with
underlying errno 1; it is diagnostic evidence rather than the original watcher stderr. A
later Rome and tool-call check passed 2/2, but it started
about six minutes and 27 seconds after controller completion and therefore did not meet the
120-second timing requirement. Record this as a monitoring/timing miss with a late functional
PASS, without an extra restart. The final check verified one exact E07a stack on all four
ranks, restart counts 0, idle 0/0, 47 °C and inactive thermal and power-brake flags. A
CPU-only exact-image probe then reproduced the same BLHNC stride/page guard and passed its
no-split and dense-layout allocator controls without GPU visibility, weights or a fourth
serving boot. This confirms the guard only; it does not verify a supported recipe change.

The owner-authorized attempt-4 preparation keeps the complete attempt-3 recipe and adds
only three read-only R10 module overrides from
`~/tp4/candidate-windows/e09-r10-cache-fix-20260913/overrides`: `kv_cache_interface.py`,
`core/kv_cache_utils.py`, and `worker/utils.py`. They retain allocation id 0 for target
B12X/KDA pages and give each DFlash draft layer its own dense allocation while preserving
one global block pool. The memory divisor is the padded shared-target bytes plus every
draft-layer page, so the configured 17179869184-byte KV budget is not exceeded. Generic KV
offload remains disabled in this recipe and is outside this override's qualification.

Before an attempt-4 `up`, require all three module hashes and the OFF overlay hash to match
private manifest `cache-fix-attempt4/CACHE-FIX-ARTIFACTS.sha256` (manifest SHA-256
`e6825e093f5f8ae52b0e5b1de3fea27d990f51d5402da1efea4d1b1069a41571`) on all four ranks.
The rendered argv must normalize exactly to attempt 3 after removing those three `:ro`
mounts. The exact-image CPU test passed with five independent draft storages, multi-block
copy, unchanged target-only views, replicated DCP metadata, one shared `num_blocks`, and
allocation below its test budget; E07a's container, process and health remained unchanged.
This local result did not replace the usual image, `/health`, functional gates, live source
hashes and native Rigmark checks. The executed attempt-4 boot loaded target and draft on all
four ranks, completed graph capture and KV view creation, then reached `/health` 200. Its
actual KV capacity was 1,435,070 tokens, or 5.47x at max context 262144. The first Rome gate
returned HTTP 200 but exactly 64 `!` characters with `finish_reason=length`; the tool gate
was skipped, and no native Rigmark run or request started. The exact attempt-4 recipe is
discarded and must not be relaunched automatically. Coordinated four-rank `down` passed.
E07a restoration completed `up` with exit 0 in 971.241 seconds, reached health 200, and
passed Rome plus tool-call 2/2 within 8.826 seconds of first health. The final check found
one exact E07a stack on each rank, restart counts 0, idle 0/0, 48 °C and inactive thermal
and power-brake flags. The content failure does not establish general R10 or SparkCache
incompatibility, because SparkCache remained off. SparkCache ON remains unstarted and
blocked.

Attempt 5 keeps the exact attempt-4 recipe and adds the pooled-tail correction from
upstream commit `618562444d73742e4e872defcad4d37f477f5c59`. Because the DCP1 decode-only
path calls the separate physical expansion helper, the reviewed revision applies the same
`live_history=min(2048, complete_pools*4)` calculation to both logical and physical R10
kernels. Require private patch/test manifest SHA-256
`206d2c462c38cca87ce9ee3e8576a53500ee925cc5803d7ce9536c26a860dade` and instrumented
candidate manifest SHA-256
`21ddf661ced69099f9bcd5a0b66e964d12a80312fd0a88e43047523584ac324d` before using the
revision-3 overlay. The first test runner named an unavailable Docker runtime and was
rejected before any container or GPU case; it is not test evidence. With the corrected,
previously validated `--gpus all` invocation, the original image source failed the exact
eight short-tail route cases and the revision-3 override passed all 16 logical/physical
cases, including a 25-token prompt, history boundaries and long physical remapping. The
separate weight-free CUDA replay for numeric diagnostics also passed.

The first attempt-5 OFF boot was diagnostic-only: four additional read-only source mounts
and `E09_NUMERIC_DIAGNOSTICS=1` record bounded target and DFlash summaries. Even if health
and functional gates pass, do not use that process for Rigmark. A clean measurement or
SparkCache transition requires coordinated `down` with the instrumented overlay and a new
weight load with every diagnostic mount and environment entry removed. At this recorded
stage that first boot failed during model loading because the diagnostic wrapper did not
present the instrumented GLM5 model expected by `model_runner`; its logs reported
`E09 numeric diagnostics require the instrumented GLM5 model`. The owned controller was
stopped with exit 143 after 457.45 seconds, then all four ranks were verified with zero
containers, idle GPUs and inactive flushers. It reached no health, functional request,
inference or Rigmark run. This is a diagnostic-wrapper failure and does not establish a
pooled-tail fix failure. A second boot with the corrected wrapper loaded target and draft
on all four ranks, completed `up` with exit 0 in 1002.05 seconds and reached health, then
failed its first content gate with HTTP 200, 64 exclamation marks and
`finish_reason=length`; the tool gate was skipped. Its 16 diagnostic records were the four
startup warmup steps on each rank, all before health. Warmup exhausted the bounded recording
budget, so the failed request has no diagnostic record; DFlash output does not establish the
state of target auxiliary layers, and the cause remains unresolved. Coordinated `down`
passed in 12.973 seconds. No clean OFF, native Rigmark or SparkCache ON run started. Discard
this exact instrumented recipe. E07a restoration completed `up` with exit 0 in 971.111
seconds, reached health 200 at 12:10:13 UTC, and passed Rome plus tool-call 2/2 within
8.801 seconds. Final receipts verify one exact E07a stack per rank, restart counts 0,
matching SIRCL process environment, idle 0/0, exactly two completed gate requests and
49/49/48/49 °C with thermal and power-brake flags inactive. The controller verified
all four flushers stopped and absent. A later local assertion incorrectly accepted only
`systemctl is-active` exit 3; exit 4 with `inactive` identifies the already-collected
transient unit and required no node repair. The four owned local SSH log followers were
closed; serving processes were preserved. SparkCache ON was staged on all four nodes but
never launched. Further root cause analysis is owner-requested and stays with the root
agent; these restoration receipts do not authorize an automatic retry of R10.

The [direct root cause analysis](../OPTIMIZATION-PLAN.md#e09-root-cause-analysis) records the request-selection
defect in the previous trace and the executed replacement. The new recorder passed 10/10
CUDA checks, and its four-rank boot reached health. The actual named request again returned
64 exclamation marks; 16 post-health records locate huge residual values in the first
decoder block followed by zero final hidden states and logits. The tool gate was skipped;
one coherent request failed and no Rigmark request ran. Coordinated down passed.

The owner then authorized continuous direct diagnosis, accepted extended downtime and
explicitly suspended repeated reference reloads. The E07a restoration already in progress
was cancelled through the exact owned controller and its full-cluster TERM cleanup;
its watcher sent zero gate requests. This owner direction supersedes the automatic E07a
restoration step for this continuous window. Do not label the cancelled boot as restored.

The isolated GPU investigation reproduced a precision-selection fault: legacy FP8
quantization exclusions use `model.layers.*`, while the nested runtime uses
`language_model.model.layers.*`. The inherited mapper handled the checkpoint's tensor
namespace but missed these exclusion names. BF16 KDA projections acquired FP8 scale
parameters that the checkpoint never loaded. A clean model-module override adds the
missing alias; all 170 KDA selections and eight GPU projection comparisons pass, with
native checkpoint weight mappings and dense-MLP FP8 selection preserved.

The clean OFF candidate was `scripts/node/etc/local/e09-r10-off-quant-map-fix.env`
(SHA-256 `44f73ea265467e75a353e3a096a482c15c9ac780ed4da17945831d06b21ed657`). It mounts
only the corrected model module from `~/tp4/candidate-windows/e09-r10-quant-map-fix-20260913`
in place of all prior numeric-diagnostic overrides. The existing allocator and pooled-tail
corrections remain. Require its module SHA-256
`7ae82a961e505fd5ad1c2c289ea1338a15be94317082c0d02e02f1017e7fe47f` on every rank,
absence of numeric diagnostic environment entries and mounts, and the existing R10 OFF
startup signature. It completed `up` in 726.890 seconds and reached health at
14:15:58.506493 UTC. Rome passed with normal content; the tool gate failed, so the initial
gate result was 1/2. Direct diagnosis retained the loaded process under the owner's
continuous-window override: nine inference requests, all HTTP 200, comprising Rome 1/1
correct, tools 1/6 correct and 5/6 incorrect, and two intentionally capped diagnostic
completions. No Rigmark request ran on that OFF process.

The pure-decode pooled indexer produced physical cache slots but omitted the B12X provider
method, causing the backend to translate them again as logical indexes. The direct GPU
reproducer confirms six wrong-address cases in the original dispatch and zero wrong slots
with the provider fix; mixed-prefill and DCP2 behavior are preserved. Full attention
computation was stubbed in that reproducer, so serving gates remain necessary.

The next active window uses `scripts/node/etc/local/e09-r10-on-physical-selection-fix.env`
(SHA-256 `5a63029a1ed99ff3fd5ee161ac184393f0dc1e6b756c652aa1961f1575ecb306`) with the
existing ON launcher `launch-glm53-tp4-e09-r10-sparkcache-pool-fix-attempt5.sh`
(SHA-256 `d85801569b20f4f7def294bf6781665a5064804a1b4de0141d6d984fb58447d7`). Its new
read-only pooled-indexer module lives under
`~/tp4/candidate-windows/e09-r10-physical-selection-fix-20260913/overrides/` and must hash to
`a8b55ac2a80b94b263160a5a17bd88fbf0663cdc536a3a0c4447b390ee28dfe9`. Retain the clean
model mapper and previous allocation/tail fixes. Require the R10 image, absence of numeric
diagnostics, and `--kv-transfer-config` selecting `SparkContextCacheConnector`; the
connector and configuration hashes are recorded in the direct-analysis section of the
optimization plan. The ON boot completed in 758.013 seconds after successful coordinated
OFF down and fabric checks. Health reached 200 at 15:07:47.100277 UTC; Rome and tool-call
gates passed 2/2 within 3.073 seconds. Actual mounts, environment, image and all ten source
hashes match on all four ranks. All three native Rigmark runs subsequently completed on
this process: 162 requests, zero native validation findings or stream errors, 45/45 basic
decode output gates and no long-code caps in 15 requests. External restore occurred for
21/27 native replay requests on all four ranks; all 27 cold/replay pairs returned equal
eight-token continuations. The full fixed-F0/three-run-median comparison is in the
optimization plan.

The owner authorized this combined corrected ON boot followed directly by three native
Rigmark runs on the same process after the passing gates. Use the ON overlay for its eventual
coordinated `down`; do not use the earlier OFF overlay or automatically reload E07a/F0
inside the explicitly extended diagnosis window. No production default has been promoted.

Final verification at 15:45:08 UTC found the same four container IDs, restart counts zero,
matching source hashes, health 200, idle 0/0 and stopped flushers. Both the API log and
success counter account for exactly 162 native requests plus the two successful boot
gates. Thermal samples under prefill/C4 load reached at most 78 °C; the final idle sample
was 62/61/61/62 °C. No thermal or hardware power-brake flag was active in those samples;
this was not continuous monitoring. The combined performance result is unresolved and
requires an owner decision: C2 is consistently below F0, and immediate cache replays have
large latency variation. The experimental process remains running for that review under
the current no-automatic-reference-reload instruction. The private archive is
`glm53-e09-r10-sparkcache-20260913`; promotion and another candidate remain separate steps.

### E09 pending-publication diagnostic qualification

The owner authorized one coordinated transition from the corrected ON process to
the local pending-publication candidate. The initial check matched all four saved
processes and source pins, health 200, idle 0/0 and success counter 169. Four-rank
staging and dry-runs passed; effective Docker arguments differ only by the connector
mount source and the extra-config entry spark_cache_pending_wait_ms=2000.

Use scripts/node/etc/local/e09-r10-on-physical-selection-fix.env for the old
process's coordinated down, then scripts/node/etc/local/e09-r10-on-pending-publication.env
for the candidate's up and eventual down. Continue to use the pinned mode-0700
tp4ctl-e04-window-6cb6f07c controller. Keep the same image, checkpoint identities,
physical-selection/model-mapper/allocation/tail fixes, SIRCL, KV geometry and scheduler.
The private payload is candidate-windows/e09-r10-pending-publication-20260913.

| Artifact | SHA-256 |
| --- | --- |
| Candidate connector | a0bedc1c33a316d3c56ae857652c4a17acac9a82b053a4a71689684f33b75745 |
| KV-transfer config | a38abd2e03f2f19857af7105179ea2bf155fb55474504023734a777faf012142 |
| Pending-publication overlay | c67cef891e32c090b5d92c14d7d49982c109999843d249c060edb21651e7dfd2 |
| launch-glm53-tp4-e09-r10-pending-publication.sh | 66271a15916375339a669e862fbf094f3d6d4c9e67ca83cf1a2aee8ede89bd92 |

The connector adds bounded transition traces to the previously CPU-tested candidate.
They associate issued publication, worker outcome, report emission/receipt and
admission/fallback by salted digest, attempt ticket and worker generation; plaintext
salts are excluded. Both 60 connector CPU cases and seven diagnostic protocol cases
passed before transition. This is not a live qualification result.

After all-four source/image checks and the existing jumbo fabric gate, arm the two
documented functional gates for completion within 120 seconds of health 200. Verify
the new four-rank runtime against its dry-run and pins. Then run the separately
prepared 21-request A/B/C diagnostic through the unchanged Beast0 Rigmark client,
retaining exact output equality and stopping subsequent probes on HTTP, token-count,
stream or output failure. The scope is diagnostic correctness/cache qualification,
not native performance benchmarks or promotion. The original stopped diagnosis is
not resumed or overwritten.

A failed boot or boot gate requires coordinated down with the candidate overlay;
preserve receipts and report. Do not repair one rank or automatically reload a
reference. The previous corrected ON recipe remains the concrete rollback artifact,
but restarting it is a separate owner-directed action within the extended window.

The authorized up completed in 728.456 seconds. Health reached 200 at
20:22:45.919366 UTC on 2026-09-13, and both gates passed within 3.223 seconds.
All 21 diagnostic requests subsequently passed: nine exact cold/replay pairs,
external hits 3/3 per condition and nine verified restores on every rank.
Targeted traces showed a still-pending publication in the first immediate pair
and delayed scheduler visibility in a one-second pair; both reached the required
four-rank confirmation before restore admission. Observed publication waits were
4.830–15.829 ms. No error/timeout was injected and no native performance run was
performed. This is a diagnostic pass, not promotion.

Final receipts show the same four candidate process identities and source pins,
zero restarts, health 200, idle 0/0, inactive flushers and exactly 23 completed
requests including the two gates. This candidate remains the loaded experimental
process; use the pending-publication overlay above for its eventual down.
The complete evidence is in private archive glm53-e09-pending-live-20260913.

### E09 pending-publication native-series stop

The later owner-authorized three-run native Rigmark series used the same candidate
process and recipe above. No weights were reloaded and no new gate requests were
sent. Its initial four-rank identity/idle check passed with counter 23. Only run 1
completed, at 21:34:58–21:48:57 UTC on 2026-09-13: **54 native requests, 1/3 runs**.
Code completion passed 4/5, with one repetitive 8192-token capped output. The run
also received eight additional non-loopback chat admissions; six extra counted
completions raised the total to 83 instead of 77. Two admissions remain unreconciled
with finish counters. The failed code output preceded the observed foreign traffic.

Stop disposition: **discard for promotion in this series; performance unresolved**.
Runs 2 and 3 were not invoked (108 requests skipped), and no three-run median exists.
Nine native replay hits with equal outputs and current 4/4 confirmations remain
valid cache observations but do not pass the failed code gate or qualify performance.
The plan and private glm53-e09-pending-native-20260913 archive retain all denominators,
native values, the capped output, traffic timestamps and complete four-rank logs.

Post-run receipts on 2026-09-14 verify unchanged container, host/GPU process and
source identities, zero restarts, health 200, idle 0/0, inactive flushers and counter
83. Rank 0 fails the expected benchmark count after passing its identity/recipe
checks; do not rewrite that receipt as a pass. The candidate remains loaded under
the owner's explicit suspension of automatic reference reloads. Keep using
scripts/node/etc/local/e09-r10-on-pending-publication.env for any eventual coordinated
down; no stop, repair, recipe change or promotion occurred in this benchmark batch.
Investigate the saved code degeneration and arrange an exclusive client window
before separately authorizing additional inference or another candidate transition.

### E09 resume check on 2026-09-17

The owner requested resumption after an Internet outage. SSH reaches all four ranks,
but the previously measured process is gone. Ranks 0/1 have no containers; ranks 2/3
retain different v11 DFlash2 containers that started and exited with code 1 on
September 16. No GPU compute process is present, every flusher is inactive, and
rank-0 /health refuses the connection. No inference or lifecycle operation was sent.
The previous statements about a loaded candidate are historical observations.

The E09 R10 image and all ten source pins per rank still match. The four launcher
dry-runs exactly match the qualified candidate, the pinned controller passes its
hash/type/mode check, and the native client still matches. The prior result folder
contains no run-2/run-3 artifacts. Private archive glm53-e09-pending-resume-20260917
preserves these receipts, the two exited-container logs and the proposed restart.

At that check there was no active serving overlay. A new, explicitly authorized window
was proposed for coordinated E09 startup using the pending-publication overlay and
pinned controller above, with a fresh fabric prerequisite and both post-boot gates
within 120 seconds of health 200. The proposal then runs three new native Rigmark
repetitions on that one new process; it does not resume or replace the contaminated
September 13 series. No lifecycle, reference reload or production promotion has
been executed by this resume check.

### E09 restart failure and handover on 2026-09-17

The owner subsequently approved that startup and three new native runs. One
coordinated boot began at 05:19:42 UTC. The engine failed its SIRCL capability vote
at 05:30:12 UTC: rank 1 could not use `rocep1s0f0:4`, and rank 3 could not use
`rocep1s0f1:4`. NCCL was configured for automatic GID selection (`-1`); passing
jumbo pings and E09 source pins did not validate the independent SIRCL selectors.
No health 200, functional-gate request or native benchmark run was reached.

After four-rank log capture, terminating the identified owned gate watcher
activated the existing supervisor's coordinated failure cleanup. Its generic
`gate watcher failed` result describes that intervention, not the engine root
error. Down completed successfully at 05:35:04 UTC using the same pending overlay.
At 05:36:45 UTC all four ranks had no running containers or GPU processes and
inactive flushers. No automatic F0/E07a reload or second boot was performed.

The owner requested investigation until resolution, then a portable handover.
Before another boot, inspect the actual all-rank RDMA/GID tables and SIRCL
configuration and verify the corrected selectors before loading weights. The
private `HANDOVER-E09-2026-09-17.md` delivered to the owner's mini and archive
`glm53-e09-pending-restart-20260917` contain the receipts, exact pins and next steps.
The last recorded state is fully stopped; do not relaunch the completed supervisor.

### E09 SIRCL GID recovery on 2026-09-17

The resumed read-only inspection confirmed four stopped nodes, unchanged E09
launcher/overlay pins, identical complete SIRCL payloads and eight selected links
at 200 Gb/s, MTU 9000, ACTIVE/LinkUp. All eight selected ports expose their matching
IPv4 RoCEv2 entry at index 3. The inherited `runtime/rank1.env` still sets GID0=4,
and `runtime/rank3.env` still sets GID1=4; those entries are unavailable. The
entrypoint sources these files after Docker environment setup. This establishes
the stale SIRCL selection, independently of NCCL's automatic selection. It does
not establish when or why the host GID tables changed.

The new private window is
`candidate-windows/e09-r10-pending-publication-20260913/gid-recovery-20260917`.
Its runtime changes only those two GID values, adds the explicit preflight before
`exec vllm`, and regenerates `SHA256SUMS`. The old bundle and all cache, scheduler,
geometry and engine code remain pinned. The inherited gate attestation remains
historical September 12 evidence, not qualification of this recovery.

Use `scripts/node/etc/local/e09-r10-on-pending-publication-gid-recovery.env`
for every operation, including coordinated down, with the existing pinned
`tp4ctl-e04-window-6cb6f07c`. Four dry-runs differ from the preceding recipe only
in the runtime mount source. Roll back the candidate artifact by selecting the
previous pending-publication overlay; its stale selectors must be resolved before
any serving restart. Do not automatically reload F0 or E07a.

| New artifact | SHA-256 |
| --- | --- |
| Recovery overlay | e4ac8df05391afcdd8541d0ac4d2287358432cb63cf9213ce702fdf676932e8e |
| Runtime SHA256SUMS | 27734025085eb889a3be05dbab204ded14277c91e9afdf093e31c964a18e3c0f |
| Runtime entrypoint | 81b22e7939f9a2ed1e5ea7aca9bacfbf81e46e86f73e00bf87cf3ddc171c02b4 |
| SIRCL GID checker | 7e7cd40ff0144360b64014296512b04a012a09867de896492cdd4504ab1f448e |

Ten CPU regression tests pass. An initial weight-free prerequisite probe found
that the image lacks `ip`; the checker now uses standard-library Linux ioctls.
The corrected preflight passes in the actual E09 image on 4/4 nodes. Native SIRCL
capability records reproduce exactly the old rank-1/rank-3 errors with the old
selectors, and contain no errors with the corrected selectors. These checks load
the native library and query CUDA availability but do not load weights or test
collective data transfer. Preserve the original failed boot and all September 13
benchmark results.

The first supervisor launch stopped before invoking the controller because its
Docker absence check expected uppercase `No`; the installed Docker returns
lowercase `no such object`. Its log and source remain in `receipts/`, with zero
weight loads or inference requests. The corrected monitor passes that real-response
regression and watches all four container states, capturing failure logs before
coordinated cleanup.

The actual coordinated up began at **06:11:14 UTC** and completed with **one weight
load** in 691.385 seconds. Health first returned 200 at **06:22:33.048 UTC**; Rome
content and the valid `get_weather` call for Milan passed **2/2 in 3.013 seconds**.
All four live argv/mounts, 41 source files per rank, effective SIRCL process settings,
mapped SIRCL/NCCL libraries and GPU worker identities matched. Flushers were inactive,
the API idle at 0/0, and the success counter exactly 2. The boot failure is resolved.
The first newly labelled native Rigmark run completed at **06:34:36 UTC** after
starting at **06:24:19 UTC**. Its native receipt validates: **54/54 complete
streams**, basic code/prose/structured gates **5/5 each**, no code length cap,
and **9/9 equal cold/replay continuations**. The bounded run log contains exactly
36 chat, 18 completion and 18 tokenization POSTs, all loopback and HTTP 200.
All nine replays have current-generation committed/held reports from 4/4 ranks,
cache hits, and nine verified restores per rank. Scheduler waits range from
4.911 to 52.833 ms; no live timeout/error fault injection was performed.

The detached run finished while the local session was interrupted. At
**09:49:57 UTC**, over three hours after its finish, a non-loopback chat POST
returned HTTP 200 and produced model activity plus a 52,992-token cache restore.
This does not contaminate the earlier native interval, but breaks the exclusive
series before run 2. The full-window validation remains **FAIL**; its separate
bounded analysis does not replace that receipt. The later success counter is
still 56, so that counter alone cannot establish exclusivity or the foreign
request's completion outcome. Do not launch the old run-2/run-3 commands or
silently relax the gate.

Actual qualification count: **1/3 runs, 54/162 native requests**. Performance is
**unresolved**, with no three-run median or promotion. The previous September 13
code degeneration was not reproduced in these five outputs; its cause remains
unresolved. The final four-rank check preserves the same containers and GPU
workers, zero restarts, all 41 source pins per rank, mapped libraries, health 200,
inactive flushers and idle 0/0. Keep the healthy candidate loaded under the owner's
no-automatic-reference-reload instruction. No native job remains running.
The private window retains `run1-validation.json`, `run1-bounded-evidence.json`,
`run1-cache-traces.json`, `qualification-summary.json`, native outputs and all-rank
logs. Any replacement series must record the intervening workload, new labels,
salts and initial counters; it must not merge this single run into a claimed
uninterrupted three-run result.

### E09 new exclusive native series on 2026-09-17

The owner authorized a replacement series with `procedi`. The private evidence
window is `native-exclusive-20260917-1130`, alongside `gid-recovery-20260917`.
A fresh check confirms the original four loaded processes, all 41 source pins
per rank, health 200, idle 0/0 and success counter 56. No lifecycle operation,
new boot gates or extra warmup is required for the unchanged serving recipe.
The new series boundary precedes its first request; every run uses a new label
and salt, unchanged native Rigmark protocol, and the original GID-recovery
overlay. Run 1 started at 11:31:50 UTC. Expected counters are
56 → 110 → 164 → 218. Verify all traffic from this new boundary, including gaps
between runs, and validate each result before the next. The earlier single run
and failed exclusivity receipt remain excluded from this new aggregation.
Run 1 completed at 11:42:59 UTC with 54/54 native streams, basic code gates 5/5,
equal replay continuations 9/9, all-rank verified restores and no foreign traffic.
Manual content review subsequently found seven code blocks and self-revision
prose in code sample 5, despite its explicit one-implementation/one-test-block,
no-prose prompt. The native basic gate checks nonempty normal completion and
does not detect this format violation. The other four outputs satisfy those
two format constraints. This review is not a semantic Go execution audit.

The second run had already started when the manual review completed. Its exact
owned client was interrupted with SIGINT at 11:47:53 UTC: six requests completed
and a seventh stream was interrupted; no complete native JSON was emitted.
The run's stdout and server logs are preserved, but its partial output bodies
were not persisted by the native client. No third run started. Final check at
11:49:10 UTC: unchanged healthy 4/4 processes, idle 0/0, counter 116 and no
foreign POSTs since this series boundary. Retain the native PASS receipt and
manual format FAIL separately; no three-run median is available.

The owner then explicitly requested another attempt before declaring the whole
experiment failed. A single complete native confirmation run began at
11:53:25 UTC in `native-confirmation-20260917-1153`, on the same processes, with
the same protocol and fresh label/salt/boundary. Expected counter: 116 → 170.
Preserve the previous anomalous output and inspect all five new code outputs.
The confirmation completed at **12:06:03 UTC**, with **54/54 complete streams**,
zero native receipt-validation errors, all expected loopback HTTP 200 POSTs and
counter **170**. The native code gate passes **4/5**: sample 4 repeats complete
implementations/test blocks until the 8192-token cap, ending with an incomplete
test block. Samples 1/2 also emit self-revision prose and additional code blocks.
Only **2/5** meet the two-block/no-prose instructions. All output bodies and
the review are preserved in `code-output-review.json` and `code-output*.txt`.
The native validation remains FAIL for the basic output gate; source/transport
integrity is recorded separately from output correctness.

All nine confirmation cache replays hit with equal continuations and verified
restores on 4/4 ranks; scheduler waits are 5.012–67.424 ms. The live check at
12:06:49 UTC confirms the original four containers/workers, zero restarts, all
source pins, mapped libraries, health 200, idle 0/0 and inactive flushers.
Across the replacement and confirmation windows: two complete native runs
(108 receipt-validated requests), six more completed requests in the interrupted
run, and one deliberately interrupted stream. Basic code gates pass 9/10 and
the two-block/no-prose review passes 6/10 across the complete results; partial
run bodies are unavailable and excluded from those denominators.

The owner-requested additional attempt reproduces the output anomaly. Keep the
overall optimization verdict **unresolved**: this candidate is not qualified
for promotion, and the root cause or attribution to R10, the connector, SIRCL,
or model numerics has not been established. No three-run median is available.
F0 remains frozen; retain the healthy candidate loaded under the owner's
no-automatic-reference-reload instruction. No benchmark job remains active.

### E09 target-only diagnostic on 2026-09-17

The owner authorized a coordinated target-only comparison after the additional
native run reproduced repeated code output. Read-only analysis found no external
cache hits/restores during the five short code prompts (132–137 tokens, below
the connector's 4096-token minimum). Offline template rendering reproduces all
five prompt token counts and the closed thinking prefix; the native client keeps
reasoning separate, and all five recorded reasoning fields are empty. These
findings narrow the diagnosis without establishing a component-level cause.

The private window is
`candidate-windows/e09-r10-pending-publication-20260913/target-only-20260917`.
Use `scripts/node/etc/local/e09-r10-on-target-only-20260917.env` for its up and
eventual coordinated down. The preceding process is stopped with
`scripts/node/etc/local/e09-r10-on-pending-publication-gid-recovery.env`.
Keep the pinned `tp4ctl-e04-window-6cb6f07c` controller. Four-rank dry-runs prove
the only effective Docker argument change is removal of `--speculative-config`
and its JSON value. All 43 checked source files per rank match; eight GID
selectors and the eight directed jumbo pings pass. Initial counter is 170,
with the original stack healthy and idle before the transition.

| Artifact | SHA-256 |
| --- | --- |
| Target-only launcher | ae3df54b6eacf6ed366e6304acbcb829ad1fc70a1a10bb42210e8d97a7462218 |
| Target-only overlay | 037361b34a39f83fbb7fb912dffe4e7e8aa1fead1e8cca7a024192161683cc19 |

Expected boot signature: `speculative_config=None`, no drafter load, and the
unchanged AdaptiveKScheduler reporting its policy disabled with `engine_k=0`
and no dynamic draft table. Preserve the target FP8, B12X/Triton paths, SIRCL
GID-recovery runtime, connector and cache configuration. The unused draft mount
remains present; no weights or upstream files are changed.

One coordinated load and both standard gates within 120 seconds of health 200
must pass before the native client runs. A failed boot/gate requires coordinated
down with the target-only overlay and retained logs; do not reload a reference
automatically. The unchanged native client uses `--skip-prefill` and
`--skip-concurrency`: five code, five prose and five structured cases with the
same comparison ID, prompt text, seed and sampling settings, plus a fresh salt.
Expected counter after two boot gates and fifteen diagnostic requests is 17.
Inspect every code output. This single diagnostic invocation is not a three-run
performance qualification and cannot establish promotion or a fixed-F0 median.

The transition completed with one weight load. Health first returned 200 at
**13:36:31.495 UTC** and both gates passed within **3.832 seconds**. Actual
configuration shows `speculative_config=None`, `enabled=0 engine_k=0 async=True
dynamic_sd_table=None`; the scheduler's ERROR-level notice announcing zero draft
tokens is the expected policy-disable branch. Without draft allocations, the
observed KV capacity is 2,513,253 tokens; the configured 16-GiB pool is unchanged.

The native decode-only invocation ran **13:38:32–13:50:41 UTC** and completed
15/15 streams, all basic gates 5/5, zero receipt-validation errors and zero
foreign POSTs. The complete boot/window log has exactly 17 loopback HTTP-200
chat POSTs, including the two boot gates. Code format still passes only **2/5**:
samples 2/4 append prose, and sample 5 self-corrects and publishes a second test
file (three code blocks and prose). All five stop normally; none reaches the
8192-token cap. Their completion counts are 2294, 1869, 1886, 2049 and 3326.
All five reasoning fields are empty. No external cache hits, restores, snapshots
or pending-publication events appear in any rank's new log. The post-run identity
and source check passes on the same four processes, restart 0, health 200 and
idle counter 17; all owned jobs have exited and no next run is queued.

The output anomaly therefore occurs without DFlash. Disabling speculation is
insufficient to fix it; this does not isolate its root cause or establish how
DFlash affects the earlier repeated-to-cap behavior. The single diagnostic
has no three-run performance median, no semantic Go execution audit, and no
promotion. Preserve `diagnostic-integrity.json`, `code-output-review.json`,
`code-output*.txt`, `cache-events.json`, `qualification-summary.json` and the
native receipt (SHA-256
`c0db977776995db29b41c947b884da30a6852d0729172941f71cf116430eb468`).

The corrected DFlash recipe remains the rollback artifact, but no automatic
second load is part of this diagnostic. Preserve earlier sealed evidence.

### E09 code-output RCA on 2026-09-17

The subsequent [root cause analysis](../E09-ROOT-CAUSE-ANALYSIS-2026-09-17.md) identifies
a concrete hypothesis: the locally adapted template closes thinking immediately
while retaining the default Max effort instruction. The installed GLM parser
also disables reasoning extraction when the same flag is false. Forty offline
CPU render/parser combinations verify this path and show that enabling thinking
reproduces the upstream template for the five saved prompts. Official model
documentation specifies always-on thinking. The causal effect on code quality
still needs a controlled request-only comparison; the adapter and flag already
exist in F0, so this is not evidence of a new E09 regression.

Offline execution of saved target-only answers with local Go 1.26 passes the
model-supplied tests for samples 1–3 and fails samples 4–5. Sample 4 expects 100
successes from only 50 calls; sample 5 fails NaN/Inf validation with both its
original and revised test. These are forensic replays, not a native Docker audit
or proof of general semantic correctness. The separate native audit extractor
also mishandles filename fences and does not reject extra prose; its fixes
belong in Rigmark. Earlier sealed receipts remain unchanged.

The RCA plan specifies native A/B request parameters and unchanged prompts,
sampling, process identity and default three invocations per arm. This is a
proposal, not an active job or a new lifecycle authorization. Last read-only
check at 16:10:41–16:10:44 UTC confirms the same healthy target-only containers
on 4/4 ranks, zero restarts, idle 0/0 and counter 17. The RCA sent zero inference
requests and changed no serving settings. Retain the target-only overlay for
any eventual coordinated down and the no-automatic-reference-reload instruction.

### E09 reasoning diagnosis concluded on 2026-09-17

The owner-authorized [pilot qualification](../E09-QUALIFICATION-2026-09-17.md)
ends unresolved. At equal Low effort, thinking false and true both pass code
format 5/5, but both fail Go correctness. True/High passes format 3/5, compilation
4/5, model tests 3/4 and independent/race suites 1/4; its successful Go sample
adds forbidden prose. The two observed Max completions reach 8192 tokens; the
interrupted native invocation has no raw receipt. These results do not establish
thinking-off as the cause. No profile qualifies for confirmation or DFlash.

The owner clarified that their concurrent conversation is permitted during
correctness diagnosis. Do not use its presence to reject these pilots or their
timings to qualify performance. Three complete native invocations contain 45
requests; Max adds three issued requests, two completed and one interrupted
client stream. False/Max and false/High were not completed or substituted.

Final checks at 20:11:27 UTC pass all four process/source identities, health 200,
zero restarts and idle 0/0, with counter 87 including owner conversations.
All diagnostic clients and isolated audit containers have ended. The target-only
runtime stays loaded, with no reload, API/template change, promotion or automatic
reference restore. Retain its target-only overlay and pinned controller for any
eventual coordinated down. Rigmark fixes pass 98 tests including real Docker;
the repository offline check passes. The new report records exact pins, audit
hashes, omitted phases and the bounded numerical diagnosis proposed as follow-up.

### E09 reasoning diagnosis: attribution closed on 2026-09-17

An independent, owner-authorized review closes the attribution question left
open above. It used no cluster access: an external reference request to the
same pinned model (`z-ai/glm-5.3-flash`, confirmed on OpenRouter as the
identical target model) with the unmodified Rigmark code-workload system and
user prompt, temperature 0, top_p 1, and the same 8192-token total budget, and
no `chat_template_kwargs` or other local-deployment parameter. Five requests,
zero local configuration. Result: **1/5 passes the two-block/no-prose format
and an independent `go test -race` oracle; the other four all end with
`finish_reason: "length"`, spending 6,635-8,192 of the 8,192-token budget on
reasoning alone before finishing or without any visible answer.** Raw
responses, reasoning text and the independent Go oracle output are archived
outside the checkout, in the owner's private E09 investigation archive under
`E09-EXTERNAL-REFERENCE-20260917`
(`summary.json` SHA-256 `11415d4aada72fc8c7420757e840d9b8f5f68082d31e70ae82e6030afe80b60e`).

The same review re-read the already-collected true/High native receipts
(`pilot-true-high-20260917/run.json`, referenced above) rather than issuing
new inference. All five completions there total only 2,185-2,625 tokens,
finish with `stop`, and carry 89-656 reasoning characters: well under budget,
not truncated by a parser or template limit. The interrupted true/Max attempt
reached the 8192-token cap on two of its three issued requests before being
stopped, the same failure mode as four of five samples on the clean external
reference. Across the full locally tested effort spectrum, the failure
pattern tracks how much the model reasons before answering, not which local
component served it: Low (zero reasoning) and High (minimal reasoning) both
answer within budget but fail correctness or format; Max (extended reasoning)
exhausts the budget, matching the external reference exactly.

**Conclusion: the code-workload prose/format/correctness defect diagnosed
above is not attributable to this deployment.** It reproduces, at a
comparable rate, on the same model with zero local configuration, and the
local effort spectrum brackets the same budget-exhaustion failure mode
observed externally. Do not open a numeric (FP8/KDA), connector, or
thinking-off-compatibility investigation for this specific defect; none of
those components differ between the failing external reference and this
deployment. The token budget Rigmark uses for this workload is not always
sufficient for how much this model reasons on this prompt; that is a
benchmark/prompt parameter, external to this repository, not a cluster
defect. This does not reopen or change the "Thinking-off compatibility"
mechanism described in
[`production-recipe.md`](production-recipe.md#thinking-off-compatibility),
which remains unrelated to this finding. The E09 SparkCache candidate's own
qualification remains **unresolved** on its own merits (see above); this
entry only closes the attribution of the reasoning/format defect, it does not
qualify or promote any candidate.

## Recovery and rollback

Begin with read-only status and choose the narrowest matching rollback. Every restart
below is full-cluster and must fall within an authorized service window.

| Condition | Recovery | Verification |
| --- | --- | --- |
| Overlay result is bad | `./scripts/tp4ctl restart` with no `TP4_ENV` | base `cluster.env` signatures and gates return |
| Production engine knob is bad | restore the rollback documented beside the value in `cluster.env.example`, update local `cluster.env`, deploy, restart | five signatures plus task gate |
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

### Restore from the frozen F0 archive

Treat runtime restoration and host restoration as separate decisions. For runtime F0:

1. select one archive, verify its SHA manifest offline, compare it live, and confirm the
   four intended targets plus image, model, drafter and exact NCCL availability;
2. review observed paths, permissions, owners and symlinks against intended destinations;
3. in a future authorized window, stage the archived controller as a regular file and
   verify its hash before one coordinated four-rank `down` using the overlay that launched
   the current process;
4. restore only the archived repository-managed runtime assets through `scripts/deploy.sh`
   and the NCCL atomic installer, checking every destination hash;
5. render the prepared autostart drop-in with the deployment user. It explicitly selects
   the F0 overlay and its verified reference controller for both start and stop. Install it
   only after review and run `systemctl daemon-reload`; do not treat the previously loaded
   unit as if it already contained the disk changes;
6. require two addressed MTU-9000 ports per rank, all eight jumbo pings, Ethernet speed,
   RDMA/HCA/GID checks and absence of a second stack or foreign GPU work; then run one
   coordinated four-rank `up`;
7. from the first `/health` 200, run both functional gates above within 120 seconds and
   finish with the F0 operational check. A failed gate requires full-cluster stop and a
   report, never single-rank repair.

The captured host state is comparison evidence, not an unattended host restore program.
Do not apply netplan, flush the global firewall, activate NetworkManager profiles, change
devlink/eSwitch/TC/offloads, packages, drivers, NIC firmware, kernel, IOMMU, GRUB or boot
parameters as part of runtime rollback. Preserve management, Tailscale, SSH and unrelated
configuration. Any host or fabric change needs its own authorization, an exact owned-file
diff, and verified independent recovery access or the owner's physical availability. If a
reboot is later approved, stop all four ranks first and reboot rank 0 last.

The archive records what was observed and what the prepared F0 restore would install.
Creating and verifying it does not rehearse a full restore and does not prove unattended
recovery from kernel, driver, firmware, boot-loader or network failure.

## Keep an accepted change

After the owner accepts a recipe change, persist the value and rollback in its source
file, update any affected runtime signature and `CHANGELOG.md`, and run
`./scripts/check.sh`. Evaluation tooling and result records remain outside this
repository.

Do not expose node addresses, private paths, or logs in public documents. A commit,
tag, release, or public announcement remains a separate explicit action.
