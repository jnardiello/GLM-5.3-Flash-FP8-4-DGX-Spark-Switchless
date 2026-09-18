# Agent entry point

This public repository distributes a general purpose installer for GLM-5.3-Flash
on four DGX Spark nodes using TP4. Read the one document for the requested task
before substantive work:

| Task | Required document |
| --- | --- |
| New hardware, first handoff, image or weight installation | [`docs/install-from-zero.md`](docs/install-from-zero.md) |
| Status, deploy, start/stop, recovery, rollback, functional gates or promotion | [`docs/operations.md`](docs/operations.md) |
| Cabling, addressing, MTU, RoCE, HCA/GID or NCCL failure | [`docs/fabric.md`](docs/fabric.md) |
| Current model, image, scheduler, patches or host recipe | [`docs/production-recipe.md`](docs/production-recipe.md) |
| Local code or documentation only | relevant row above, then `CHANGELOG.md` and `./scripts/check.sh`; do not probe the cluster automatically |

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

Use the independent Rigmark suite at `~/workspace/jacopo/rigmark` directly as the
benchmark interface. Do not recreate it with private wrapper scripts, blanket campaign
qualification prerequisites, or a parallel benchmark/admission framework; necessary
measurement fixes belong in Rigmark.

Run one experiment at a time. Use the frozen baseline **F1** recorded in
[`docs/baseline-f1.json`](docs/baseline-f1.json) for every future comparison; its three
native Rigmark runs and fixed medians, measured from the same workstation with a fresh
`cache_salt` per run, are the standing reference. Do not rerun or replace that baseline
without an explicit owner request. F1 is the IaC base recipe in `cluster.env.example`
(image pinned by registry digest, `SPARKCACHE_MODE=on`): restore it with a plain
`scripts/deploy.sh` and a coordinated restart without `TP4_ENV`. Use
`scripts/check-f0.py` for a fast, read-only operational identity check against F1 (it
does not rerun Rigmark or send inference requests). [`docs/baseline-f0.json`](docs/baseline-f0.json)
is the historic F0 record; its lane stays reachable through the rollback comments in
`cluster.env.example` and `scripts/check-f0.py --baseline docs/baseline-f0.json`.
Agree the variant repetition count with
the owner only when explicit later direction changes the standing protocol. By default,
run every variant three consecutive times. Manually apply it to all four Beast nodes in
one coordinated transition, load its weights once, and preserve that process across all
three runs. Use the same Rigmark version, suites, prompts, and parameters under comparable
idle, warmup, and cache conditions.

For every performance metric, compare the fixed F1 median with the median of the three
native per-run variant values. Label the actual variant run and request counts. Keep
measurement-integrity and error results as explicit counts or totals with denominators
rather than medians. The results report requires fixed-baseline and variant-median columns; delta
columns and a separate statistical framework are not required. Judge prefill and
generation throughput, TTFT, concurrency, and functional results beyond measured noise,
with no fixed percentage floor. Record commands, configuration, results, and failures,
then classify the outcome as promote, discard, or unresolved. Restore the reference when
discarded or unresolved unless the next candidate is already prepared and authorized.
An owner stop may end the three-run series early; report the actual run and request counts
and do not claim a three-run median. A direct transition to the next candidate needs no
intermediate reference reload: derive its complete recipe from F1, remove the previous
delta, and use the usual coordinated four-rank transition and functional gates. Continue
to compare against the fixed F1 medians. Restore the F1 state whenever work stops without
a prepared next candidate.

A promote decision requires encoding the tested change in existing repository IaC;
update the relevant documentation, rollback guidance, and changelog; reapply it through
IaC from the reference; and verify actual configuration, cluster gates, and Rigmark
reproduce the manual result before committing within the session's explicit
authorization. Finish the experiment, then agree the next one with the owner.

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
