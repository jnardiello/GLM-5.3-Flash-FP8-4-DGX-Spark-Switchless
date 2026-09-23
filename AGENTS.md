# Agent entry point

This repository is infrastructure as code for an agent starting from a fresh checkout
to reproduce the documented GLM-5.3-Flash service on four compatible NVIDIA GB10 nodes
using TP4. The verified hardware is ASUS Ascent GX10; discover each target's interfaces
and host configuration before adapting the site template. Read the required document
for each part of the requested task before substantive work:

| Task | Required document |
| --- | --- |
| New hardware, first handoff, image or weight installation | [`docs/install-from-zero.md`](docs/install-from-zero.md) |
| Status, deploy, start/stop, recovery, rollback, functional gates or promotion | [`docs/operations.md`](docs/operations.md) |
| Cabling, addressing, MTU, RoCE, HCA/GID or NCCL failure | [`docs/fabric.md`](docs/fabric.md) |
| Current model, image, scheduler, patches or host recipe | [`docs/production-recipe.md`](docs/production-recipe.md) |
| Local code or documentation only | relevant row above, then `CHANGELOG.md` and `./scripts/check.sh`; do not probe the cluster automatically |

## Fresh-checkout reproduction contract

For installation, read both [`docs/install-from-zero.md`](docs/install-from-zero.md)
and [`docs/operations.md`](docs/operations.md). The default recipe is
[`cluster.env.example`](cluster.env.example); its measured identity and performance
record is [`docs/historical_benchmarks/baselines/2026-09-23-e22b/baseline.json`](docs/historical_benchmarks/baselines/2026-09-23-e22b/baseline.json). Preserve its
non-site settings unless the owner requests a variant. Use this checklist to navigate
the existing procedures:

1. **Establish scope and check the checkout.** Identify the four targets in rank order,
   deployment account, intended service, and actions already authorized. Run
   `./scripts/check.sh` locally. An installation window that already covers downloads,
   bootstrap, deploy, and service startup covers those steps throughout the runbook;
   ask only for missing scope, not again at each step.
2. **Inventory all four nodes.** Once the targets are in scope, run
   `TP4_HOSTS='<rank0> <rank1> <rank2> <rank3>' ./scripts/agent-preflight.sh --report <absolute-private-path>`.
   Use its per-node inventory and `proposed_config` to establish GPU/RAM/disk capacity,
   OS/driver/tooling, management interfaces, RDMA ports, HCA/GID mappings, and renderer.
   Confirm the physical ring and private subnets with the owner; discovery cannot infer
   an unverified cable map. Preserve existing workloads during inspection.
3. **Resolve the site configuration.** Copy the annotated template to ignored
   `cluster.env`. Fill `NODES`, `NODE_HOSTNAMES`, `MGMT_IPS`, `MASTER_IP`, fabric peers,
   paths, and the deployment account's transfer destination. Set the documented
   `MGMT_IF`, `FABRIC_IFACES`, `NCCL_IB_HCA`, `NCCL_IB_GID_INDEX`, and
   `NETPLAN_RENDERER` scalars or four-element `*_BY_RANK` arrays from that inventory.
   Retain the default automatic GID selection where its validated prerequisites hold.
   Generate files with `scripts/render-netplan.sh --write`, then `--check`; never copy
   the maintainer's site values or hand-edit generated files.
4. **Account for every artifact before startup.** Follow the installation guide for
   host pins, the image digest/content ID, target model manifest, drafter revision,
   patched NCCL, and deployed runtime overrides. Prepare the operator-supplied
   SparkCache connector and encoder with `scripts/prepare-sparkcache.py`; its inputs
   and outputs must match the recorded hashes. Supply the SIRCL bundle/runtime and
   verify its private per-rank peer/GID files as described in
   [payload installation](docs/install-from-zero.md#8-place-the-sparkcache-and-sircl-payload).
   Check [`CREDITS.md`](CREDITS.md) for acquisition and use terms. Required operator
   payload is not included in the checkout. If it is missing, name the exact artifact,
   expected pin, and acquisition/preparation step; continue independent preparation.
   Never fabricate payload or change a checksum to accept a substitute. A rebuilt NCCL
   binary follows its documented candidate procedure, not automatic pin replacement.
5. **Preview, install, and verify.** With `cluster.env` complete and no `TP4_ENV` for
   the default recipe, preview each native launcher command locally:

   ```sh
   for rank in 0 1 2 3; do
     TP4_DRY_RUN=1 bash scripts/launcher/launch-glm53-tp4.sh "$rank"
   done
   ```

   Keep this output private because it contains site values. Dry-run checks command
   construction; it does not establish artifact or hardware readiness. Continue the
   authorized bootstrap, artifact installation, deployment, and coordinated startup
   in the installation guide's order. Use its existing static/fabric checks, wait for
   `/health` 200, complete both documented functional gates within two minutes, and
   verify the current identity with `scripts/check-f0.py`. If autostart is already
   loading or serving, follow the guide without launching a second stack.
6. **Report the actual result.** Record the effective recipe, four-rank identity,
   gate results, and any remaining limitation in private evidence. Keep a healthy
   service running unless another lifecycle action is authorized and required.
   Installation does not automatically authorize performance measurement: run native
   Rigmark only when requested, keep new results separate, and never remeasure or
   replace the frozen reference automatically. Successful installation establishes
   the verified service state; throughput reproduction requires its own measurements.

## Public purpose and documentation

Write repository documentation for external users installing and operating their own
cluster. Public guides must stand on their own, with reusable instructions, explicit
prerequisites and configurable examples that do not depend on the maintainer's site
or knowledge of previous agent sessions.

Before saving documentation, distinguish reusable product guidance from local working
notes. Keep site addresses, host aliases, private paths, session chronology, incident
receipts and experiment diaries in ignored local files or private archives outside the
checkout. Bring generally applicable lessons into the public guides as concise operating
instructions; keep the detailed local history in those private records. Apply this
principle to new documentation and updates to existing files, including the changelog.

## Sources of truth

- `cluster.env` is the active site and production configuration. It is gitignored.
- `cluster.env.example` is its annotated public template. Keep recipe values and
  one-step rollback comments there instead of copying them into guides.
- `scripts/render-netplan.sh --write` derives every per-node netplan and fabric
  iptables environment file from `cluster.env`. Never hand-edit generated files.
- `scripts/node/bootstrap/versions.env`, `scripts/node/model-manifests/`, and
  `scripts/node/nccl/` own their respective pins.
  [`scripts/node/README.md`](scripts/node/README.md) maps node-side assets.
- A `TP4_ENV` file is a delta sourced after `cluster.env`; use the same value for
  every command in its window, including `down`. Never override `CONTAINER`.

## Mandatory debugging discipline

**Start with the simplest plausible explanation and the cheapest test that can
distinguish it. Escalate incrementally to more complex hypotheses only when the
evidence requires it. This is a mandatory working rule, not an optional preference.**

1. State the observed symptom separately from the suspected cause. Before each test,
   identify the hypothesis, the result that would support or reject it, and the next
   action. Prefer a small reproduction over a broad diagnostic campaign.
2. For unexpected model output, first suggest a matched request to the official cloud
   model or another reference deployment, before investigating cluster internals.
   Match prompt, model, sampling, reasoning settings and token budget where possible;
   record differences. Reuse existing reference evidence. Run external requests only
   within the authorized scope; lack of a reference does not establish a local defect.
3. If the same failure occurs on the reference, investigate the shared model, prompt,
   request parameters or benchmark assumptions first. Do not keep treating that symptom
   as evidence against this cluster without a new observation that distinguishes it.
4. Change one factor at a time and inspect the result before the next experiment.
   Prefer request/configuration checks and existing logs before instrumentation, kernel
   changes, model reloads or engine variants. Every escalation needs evidence explaining
   why the simpler explanations are insufficient; complexity is not evidence of rigor.
5. Preserve the user's objective. Do not turn performance measurement into an answer-
   quality project, add unrelated acceptance gates, or let an optional diagnostic block
   the requested work. Stop a diagnostic branch when its hypothesis loses support and
   return to the original task.

The E09 cloud control is the concrete lesson: model-generated code/format failures
also observed on the reference do not justify an FP8/KDA/connector investigation or
blocking the performance benchmark on Go compilation, model tests or a custom oracle.
Earlier E09 quality-based stop decisions are historical records, not prerequisites
for resuming performance measurements.

## Optimization workflow

Use the independent Rigmark suite from the operator’s Rigmark checkout directly as the
benchmark interface. Do not recreate it with private wrapper scripts, blanket campaign
qualification prerequisites, or a parallel benchmark/admission framework; necessary
measurement fixes belong in Rigmark.

Run one experiment at a time. Use the owner-accepted **September 23, 2026 E22b**
reference in [`docs/historical_benchmarks/baselines/2026-09-23-e22b/baseline.json`](docs/historical_benchmarks/baselines/2026-09-23-e22b/baseline.json)
for future comparisons. Its fixed medians use exactly three complete native Rigmark
suites (162 requests) measured on one retained load of the E22b candidate, which adds
8-bit weights for 30 DFlash2 drafter linears to the E21 recipe while keeping the
drafter's context K/V projection in BF16.

The IaC defaults in `cluster.env.example` select those measured sources: the E03 mHC
6,912-row prefill, replay views and effective draft-budget cap, the E21 residual
projections, the E22b drafter conversion and its cache namespace, retaining hybrid KDA
and 15 GiB KV per rank. `scripts/check-f0.py` selects this identity by default without
inference. The [promotion record](docs/historical_benchmarks/baselines/2026-09-23-e22b/promotion.json)
records how the default was applied. The immediate rollback is
`scripts/node/reference/baseline-20260923-e21.env`, the complete E21 recipe.

The [previous E21 reference](docs/historical_benchmarks/baselines/2026-09-23-e21/baseline.json)
and the [E03 reference](docs/historical_benchmarks/baselines/2026-09-19-e03/baseline.json)
remain immutable at three suites / 162 requests each; their complete returns are
`baseline-20260923-e21.env` and `baseline-20260919-e03.env`, and the older pre-E03 return
is `scripts/node/reference/baseline-20260919.env`.
The [earlier September 19 base](docs/historical_benchmarks/baselines/2026-09-19/baseline.json)
remains immutable at exactly two accepted native runs / 108 requests; its third run
was excluded for competing traffic. The [September 18](docs/historical_benchmarks/baselines/2026-09-18/baseline.json)
and [September 11](docs/historical_benchmarks/baselines/2026-09-11/baseline.json) records also
remain immutable. Select a historical identity explicitly with `--baseline` and its
matching complete rollback recipe. Never remeasure or replace a frozen record without
an explicit owner request.

### Run the benchmark

Use the selected frozen record's `rigmark.source`, prompts and settings as the
execution specification. Keep version/source hash, comparison ID, model, seed,
request body, reasoning controls, token limits, prefill depths and concurrency
settings matched. Do not substitute Rigmark's defaults for explicit recorded values.
For the current reference, the complete suite has 54 requests: five requests for each
of three decode workloads, three cold/replay pairs at each of three prefill depths,
and three rounds at concurrency 1/2/4. `--runs 5` is an internal suite setting;
**three variant runs means three complete suite executions**, normally 162 requests.

Run one experiment at a time. By default run each variant three consecutive times,
with one authorized coordinated four-rank transition and one weight load retained
across the series. Use comparable idle, warmup and cache conditions. Generate a fresh
`cache_salt` for every complete execution, forwarded to both chat and prefill requests;
preserve it within that run's cold/replay pairs. Keep the comparison ID matched to
the reference so the benchmark prompt construction remains comparable.

**Every native `./rigmark run` command must specify `--output`:**

```sh
--output "$TP4_REPO_ROOT/docs/rigmark_reports/<date-experiment>/run-<n>.json"
```

Set `TP4_REPO_ROOT` to the absolute path of this repository before changing into the
independent Rigmark checkout. Use a unique dated experiment directory and a new run
filename; never overwrite receipts. Use `umask 077` for local originals. Native cards
are saved beside the JSON when the run completes. See
[report storage and viewing](docs/rigmark_reports/README.md) for the full convention
and the native `report`/`compare` commands. Do not create a benchmark wrapper.

### Evaluate and preserve results

After each execution inspect the saved result and errors before starting the next.
On failure, preserve partial receipts/logs, diagnose from the available evidence,
report the cause or remaining uncertainty, and stop the series. Do not automatically
restart, unload the model or restore a baseline outside the owner's explicit lifecycle
authorization. Do not turn a normal output-budget stop or an answer-quality issue into
an infrastructure failure. The functional and measurement rules below still apply.

For every performance metric compare the fixed **current operational baseline**
median with the median of the variant's native per-run values. Record actual run and
request counts. Integrity and error results use explicit totals and denominators,
not medians. Judge code, concurrency, prose, prefill and latency beyond measured noise,
with no fixed percentage floor. Keep reproduction runs separate from frozen baselines.
An owner stop or excluded run can shorten a series; report its actual size and never
claim a three-run median without three accepted runs. Preserve excluded measurements
with their reason, but keep them out of accepted aggregates.

Record the outcome as promote, discard, unresolved or decision_required, supported
by the evidence and owner decision. Retain incomplete, owner-stopped and diagnostic
results with their actual status. A superseded experiment is not automatically a
discarded experiment. Before an authorized next candidate, derive its complete recipe
from the current baseline and remove the previous delta; no intermediate baseline
reload is required. If work stops, report the loaded state and coordinate any required
restoration with the owner rather than unloading automatically.

A promotion encodes the measured recipe in existing IaC and updates recipe and rollback
documentation. The accepted suites are already the measurement, so do not reload or
rerun them. Prove instead that the encoded default produces a launcher command
identical, on all four ranks, to the measured candidate, and that the running identity
matches it with `scripts/check-f0.py`. When the measured candidate is still serving,
deploy the default and retire its overlay without a restart. Record the parity and
live identity in the promotion record. A separate reproduction run is made only on
explicit owner request and stays separate from the frozen baseline. Commit and push
only within the owner's authorization.

### Archive and publish

Use these locations for every experiment, including discarded or incomplete ones:

| Location | Contents |
| --- | --- |
| `docs/rigmark_reports/<date-experiment>/` | Ignored native JSON, cards and logs, unchanged and privately retained |
| `docs/historical_benchmarks/baselines/<date>/` | Frozen public baseline records and separate reproduction evidence |
| `docs/historical_benchmarks/experiments/<date-experiment>/` | Portable result extracts, counts, protocol/configuration identity and source hashes |
| `docs/benchmarks/` | Index and Markdown reports for baselines, experiments and comparisons |
| `docs/plots/` | PNG/SVG figures under dated baseline, experiment or comparison directories |

Archive the available evidence at the end of every run, including partial results;
update the experiment report and index when its status changes. Public reports must
include the comparison reference, settings, actual counts, exclusions, metric
aggregation, limitations and documented outcome. Publish numeric extracts with native
receipt hashes, omitting site values, private paths, raw salts and generated payloads.
If only a historic summary survives, label it as summary-only; do not invent receipts.
Keep frozen JSON bytes and hashes unchanged when relocating records. Their original
embedded path strings are provenance; use the index for current paths.

The README shows current accepted values and a single delta column relative to the
**immediately previous baseline**, with its date and a link to its report. Its figures compare
current and previous values side by side, with percentage deltas. These presentation
references do not replace the current operational baseline used to assess new experiments.
Calculate percentages from unrounded frozen medians and format decimal labels consistently.
Keep older comparisons, original figures and diagnostic probes in the archive. Read
[the benchmark index](docs/benchmarks/README.md) before publishing results.

Rank success by these criteria:

1. Agentic code generation C1 and parallel work with two to four agents are primary.
   Repeatable gains beyond measured noise in code or concurrency may justify unchanged
   or slightly slower prose. Do not silently trade a measured regression in one primary
   code or concurrency metric for another; record conflicting primary outcomes as
   `decision_required` unless existing owner direction resolves them.
2. Prose gains qualify only when code and parallel performance remain unchanged within
   measurement uncertainty.
3. Structured-generation speed is diagnostic and never qualifies a variant by itself.
4. Prefill gains are welcome only without regressions beyond noise in higher-priority
   code, concurrency, or prose metrics, subject to the explicit code-led prose exception
   above.

Measurement integrity, transport/error accounting and healthy four-rank operation are
required and cannot be traded for speed. Rigmark's role here is performance measurement:
generation throughput, time to first token/first visible response, prefill and concurrency.
Generated-answer quality, Markdown format, Go compilation, model-written tests and
independent or race-detector audits are not performance acceptance gates. Do not run
those audits as part of the benchmark workflow unless the owner separately requests them.
A normal token-budget stop is a measured output limit, not by itself a cluster failure.
Record native output gates and finish reasons honestly without rewriting receipts or
turning FAIL into PASS; assess which timing/throughput metrics are supported separately.
If no visible response exists, report its latency as unavailable rather than zero or a
successful delivery. Do not discard otherwise valid speed measurements solely because
the generated answer fails a quality check. Record performance tradeoffs explicitly;
insufficient or conflicting performance evidence remains unresolved or decision_required.

## Work rules

Inspect before editing and preserve unrelated work. Local code and documentation
changes requested by the owner may proceed. Node and external changes have narrower
authorization boundaries:

- Obtain target-specific approval before remote discovery when the targets were not
  already placed in scope.
- Confirm that the current authorization already covers privileged bootstrap or
  downloads; host network changes or reboots; deploy or start/stop/poweroff; and
  promotion. Ask only when the action or maintenance window is new.
  One maintenance window does not authorize the next.
- Purges, commits, pushes, tags, pull requests, releases, and public announcements
  each require an explicit request. Never automate weight deletion.
- Never request, print, or commit passwords, keys, tokens, or cookies. Private site
  values may be stored only in ignored local configuration; do not expose or commit
  them. Preflight reports stay outside the checkout with mode `0600`.

The API binds rank 0 on the host network with no authentication. Keep it on a trusted
LAN/VPN. The deploy account has `NOPASSWD:ALL`, and rank 0 has a passphrase-less SSH
mesh to all four nodes, including itself; protect those accounts as root-equivalent.

## Operational invariants

- This is exactly four ranks, one GB10 each. Never restart or repair one serving rank
  in isolation; stop and use a full-cluster procedure.
- Health means `GET /health` returns 200. Do not use `/v1/models` for readiness.
- Before `up`, require two addressed MTU-9000 fabric interfaces per node and all eight
  direct-neighbor jumbo pings. Rank 0 is rebooted last because autostart launches TP4.
- `scripts/deploy.sh` is additive but replaces `~/tp4/cluster.env`; that file becomes
  the next autostart recipe even before a restart.
- `EXTRA_DOCKER_ENV` carries both the MoE config and scheduler mount. Edit only the
  intended entries; clearing it leaves the selected scheduler unimportable.
- After any changed boot, run the coherent-response and tool-call gates in
  `docs/operations.md` within
  two minutes of `/health` 200. On failure, stop the stack and report.
- Never spawn cluster-served subagents while the stack is down.
- If a rank is missing, two stacks exist, health is inconsistent, or a prerequisite
  differs from the requested recipe, stop and report instead of repairing by guess.

## Change discipline

Every repository change to code, configuration, or documentation must update the
`Unreleased` section of [`CHANGELOG.md`](CHANGELOG.md) in the same change. Describe
the concrete user or operator effect under `Added`, `Changed`, `Fixed`, or `Removed`.
When the owner explicitly authorizes a release, rename `Unreleased` to that version
and actual date, then open a new empty `Unreleased` section. Do not version, commit,
or publish automatically.

Run `./scripts/check.sh` before handoff. For any recipe change, also update the relevant
`cluster.env.example` rollback and the appropriate boot signature in
`docs/operations.md`. Purely editorial changes still require a changelog entry and the
offline check.

Do not introduce or use GitHub Actions or workflow files in this repository. Run the
required validation locally with `./scripts/check.sh`.
