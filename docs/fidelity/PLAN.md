# Fidelity campaign: plan and pre-registration

**Question.** How far do the E29 recipe's answers deviate from the vendor FP8 model served
without this repository's precision and runtime changes? A second question: how does a
generic NVFP4 recipe compare on the same measurements?

**Status.** Pre-registered on 2026-09-27, before any measurement. The owner told us to proceed
without waiting for plan approval, so the thresholds below are the owner's suggested values,
adopted as the pre-registered defaults. Any owner change is recorded in
[the amendment log](#amendments) before the affected analysis runs. The results go in
`docs/fidelity/REPORT.md` and the self-contained `docs/fidelity/report.html`, which open with
the verdict and includes all plots.

Site values (hosts, addresses, paths) never appear here. Private evidence lives under the
ignored `data/fidelity/`.

## 1. Return to GLM E29 (phase 3)

If the four ranks run another workload, record its state privately before stopping it
(containers, images, mounts, weight paths, configuration hashes, host changes and per-node
disk). Stop it with its own procedure. Any cleanup of its weights, images, volumes or host
changes runs only through audited scripts.

| Step | Procedure |
| --- | --- |
| Inventory (read-only) | containers, images, GPU processes, disk, NCCL hash, netplan hash, iptables, routes, kernel command line |
| Stop the other workload | its own `stop` procedure; confirm no container, GPU process or foreign mount remains |
| Restore missing artifacts | copy a verified checkpoint from another rank over the direct fabric link, then `scripts/fetch-fp8-weights.sh` at the pinned revision to verify every file against the release manifest and write the markers |
| Verify | `deploy.sh --check`, `verify-node.sh` (manifests, image ID, NCCL, SIRCL/SparkCache payloads), `tp4ctl fabric-check`, IOMMU status |
| Serve E29 | `deploy.sh`, `tp4ctl up`, both functional gates within two minutes of `/health` 200, `check-f0.py` PASS |
| Smoke | three requests: `prompt_logprobs` and `logprobs` accepted, behaviour with DFlash2, effect of prefix/SparkCache hits |

Copying and verifying a 306 GiB checkpoint over the fabric takes about 25 min, and a load
takes 15–35 min. A LAN download to one rank followed by a fan-out can take hours; the fetch
script certifies the bytes either way.

## 2. Disk budget per node

| Item | Size per rank |
| --- | ---: |
| GLM-5.3-Flash FP8 (`zai-org`, rev `690b7052…`) | 306 GiB |
| DFlash2 drafter (rev `bf582e4e…`) | 2.3 GiB |
| R10 SparkRing image (ID `5e32aaa1…`) | 21 GB |
| NVFP4 checkpoint (`LibertAIDAI/GLM-5.3-Flash-NVFP4`, rev `9e0d74e3…`) | ~182 GiB |
| NVFP4 recipe image (`tonyd2wild/vllm-glm53-flash@sha256:4def0ef6…`) | ~31 GB (unverified) |

Check free space on every rank before any download or pull, and run them only while the
stack is down (amendment 7).

The fidelity SparkCache namespaces are sized in the smoke test from the bytes stored for one
8K window. If a measurement arm's namespace would not fit a worker, the arm runs with its own
namespace and store/restore disabled. With fresh salts no replay is possible either way, and
the connector does not touch the logits. The deviation would be recorded here.
Superseded experiment namespaces under the compile cache are the next candidates for an
audited cleanup.

## 3. Arms and overlays

Every measurement configuration is a complete `TP4_ENV` overlay under
`scripts/node/experiments/fidelity/`. `scripts/fidelity/make_overlays.py` generates it from
a frozen reference, or from the E29 default's non-site keys; `--diff <arm>` prints the exact
delta. Each overlay has its own SparkCache configuration, identical to its base except for
`spark_cache_root` (`/cache/jit/sparkcache-fidelity-<arm>`). No production namespace is read
or written.

Measurement deltas, applied identically to every measurement arm (`-m`), with no effect on
numerics:

- `--kv-cache-memory-bytes=6442450944` (6 GiB per rank) replaces the production pool. This
  leaves room for prompt-logprob logits: an 8,192-row chunk over the 154,880-token vocabulary
  is 2.5–5 GB.
- `MAX_NUM_SEQS=1`, `MAX_MODEL_LEN=139264` (the longest window is ≤ 131,000 tokens).
- `--max-logprobs 100`.
- Unchanged: `BATCHED_TOKENS=8192`, the scheduler, mounts and kernel settings. The hybrid KDA
  path therefore switches from Marlin W8A16 to dequantised BF16 at 2,048 rows exactly as in
  production, and E03 mHC sharding applies to its eager prefill shape.
- Speculative decoding stays on if the smoke test shows prompt and generation logprobs with
  DFlash2. Otherwise every measurement arm uses its `-nospec` twin (`SPEC_TOKENS=0`, which
  omits `--speculative-config`; launcher dry runs of all existing recipes are unchanged).
- Every request carries a fresh `cache_salt`.

| Arm | Overlay | Base | Delta keys beyond the measurement deltas | Role |
| --- | --- | --- | --- | --- |
| R0 | `r0fp8-m` | `baseline-20260918.env` | — (FP8 KV; BF16 KV is impossible on B12X, amendment 6) | vendor FP8 weights as served: BF16 KDA projections, no hybrid KDA, no mHC, no E21/E22b |
| Cm | `cm` | E29 default | — | candidate in measurement mode |
| Cp | `cp-s` / default | E29 default | SparkCache namespace only | candidate in production (tasks, voxel, speculation and SparkCache checks); the final boot uses the plain default |
| Cpre | `cpre-m`, `cpre-s` | `baseline-20260919-e03.env` | — | my recipe before E21 (already hybrid KDA + E03 mHC) |
| N | `n-m`, `n-s` | Alex Ellis's NVFP4 recipe | see below | NVFP4 comparison |
| Z | — | z.ai `glm-5.3-flash` | — | cloud reference for tasks and voxel (no logprobs) |
| R0 serving | `r0-s` | `baseline-20260918.env` | 12 GiB KV (amendment 8), own namespace | R0 tasks and voxel with production scheduling |
| Ladder | `l0919-m`, `le21-m`, `le22b-m` (from R0 = `r0fp8-m`, via `cpre-m`, to `cm`) | the frozen references | — | attribution; E22b onward are negative controls |

If the image or backend refuses BF16 KV for R0, R0 falls back to `fp8_e4m3` (`r0fp8-m`, the
first ladder rung). The report then states that R0 shares the FP8-KV error.

**N (NVFP4).** Recipe `alexellis/glm-5.3-flash-4x-dgx-spark-switchless` at `e2d0839a`
(MIT): checkpoint `LibertAIDAI/GLM-5.3-Flash-NVFP4` rev `9e0d74e3…` (ModelOpt 0.45, NVFP4
expert weights), the same DFlash2 drafter, image `ghcr.io/tonyd2wild/vllm-glm53-flash`
(`sm121-v11-dflash2`, digest `4def0ef6…`, vLLM `0.1.dev20051+g487ecf187`), serve arguments
verbatim: Marlin MoE, FP8 KV, 12 GiB pool, static k=7, upstream chat template, default
`reasoning_effort: max`, and no batched-token override. It is run by
`experiments/fidelity/launch-nvfp4-tp4.sh`: his engine layer on our fabric. Recorded
deviations from his published recipe:
- our patched NCCL build replaces his; `NCCL_SWITCHLESS_RING_ONLY=1` is kept but ignored by ours;
- our HCA/GID selection and preflight, with `NCCL_IB_ROCE_VERSION_NUM=2`, `NCCL_IB_ADDR_FAMILY=AF_INET` and `NCCL_NVLS_ENABLE=0`;
- our readiness timeouts;
- our container name, so the controller manages the stack;
- no Ray, `mp` executor as published;
- our page-cache drop before loading.

The drafter revision is ours; he pins none. Known defect: vLLM issue #54150 (open; fix PR
#55073 unmerged). ModelOpt NVFP4 MoE checkpoints of this model, including this one, emit
invalid UTF-8 byte tokens on this engine build because of the fused w1/w3 global scale.
The corruption probe (§6) measures it.

## 4. Boot order

Every boot records:
- the overlay and its sha256;
- the image ID, `MODEL_REV` and `DRAFT_REV`;
- the READY lines (hybrid KDA, E21, E22, E27B/C, E29, as applicable);
- the KV dtype and K;
- the per-rank MemAvailable minimum from `scripts/fidelity/mem_sampler.py`, which runs during every measurement and aborts the client if rank 0 drops below 1 GiB.

Functional gates run after every boot. Clients run from the workstation over direct LAN HTTP,
never on rank 0.

1. **Cp** (E29 default, then `cp-s`): smoke; Cp tasks and voxel; decode set greedy with speculation on; SparkCache cold/replay check (same salt).
2. **R0 serving** (`r0-s`): R0 tasks and voxel; model-native generation (48 corpus decode prompts, temperature 1.0, top_p 0.95, `reasoning_effort: high`, 4,096 tokens) that becomes the `model_native` windows.
3. **R0** (`r0-m`): prompt logprobs run A, then run B on the same boot (repeatability floor); K=100 on a fixed 20% subset; decode set generation.
4. **Cm** (`cm`): prompt logprobs (K=20, plus the K=100 subset), decode set generation.
5. **Cpre** (`cpre-m`, then `cpre-s`): prompt logprobs, generation; then tasks and voxel.
6. **N** (`n-m`, then `n-s`): prompt logprobs, generation; then tasks, voxel and the corruption probe.
7. **R0 again** (`r0-m`): 20% subset for the cross-boot floor (skipped if the window is tight, and reported).
8. **Ladder**, only if Cm's excess KL exceeds the negligible threshold: `r0fp8-m`, `l0919-m`, `le21-m`, `le22b-m` on a 30% subset.
9. **Final: E29 default**, no overlay. Gates, `check-f0.py` PASS, `/health` 200. This is the state the cluster is left in, including after any abort once GLM is back.

## 5. Corpus (phase 1)

Built locally by `scripts/fidelity/corpus/`. Private text stays in `data/fidelity/corpus/`;
only [`corpus-summary.json`](corpus-summary.json) (counts, length histogram, hashes) is
published. The pipeline runs in this order:

1. Parse the sources.
2. Redact API keys, tokens, private keys, passwords, `.env` values and e-mail addresses with regexes plus an entropy heuristic, logging counts only.
3. Convert to GLM chat messages with synthesised tool schemas.
4. Render with the runtime chat template and the tokenizer of the pinned snapshot. Both are hash-identical to the nodes' files. Template kwargs: `{"reasoning_effort": "high"}`.

Sources:
- my Claude Code sessions;
- the owner's omp sessions;
- no Hermes logs; instead 30 synthetic Italian family-assistant conversations (school, calendar, recipes, photos described in text);
- public prompts (hardset, qeval) in the decode set only.

| Category | Target share of positions | Windows | Length rule |
| --- | ---: | ---: | --- |
| agentic code / tool traces | ~40% | ~160 | half 3,072–8,191; half 8,192 + 512–2,047 (last chunk < 2,048 rows → Marlin path) |
| long context | ~20% | 4 + 1 | 32K–64K, one ~128K |
| Italian chat | ~15% | ~45 | ≥ 2,560 |
| model-native (R0-generated, boot 2) | ~15% | ~48 | prompt + 4,096 generated |
| structured JSON / tool calls | ~10% | ~40 | as agentic |

Targets: ≥ 300 windows, ≥ 500,000 scored positions (expected ~1.2M), ≥ 90% of windows at ≥
3,072 tokens. The manifest records the source, length and sha256 of every window, plus a
global sha256. **Exclusions:** the owner's excluded projects were not yet listed at freeze
time. Measurement therefore runs on the provisional manifest. Excluded projects are later
dropped as whole windows, which requires no re-measurement because every metric is
per-window. The frozen manifest is the provisional one minus those windows, and the report
states this.

Decode set: 150 prompts ending on a user turn, rendered through the generation prefix: 100
from the sessions, 50 public (30 hardset, 20 qeval).

## 6. Measurements (phases 2, 4)

- **Prompt logprobs** (`scripts/fidelity/collect_prompt_logprobs.py`): `POST /v1/completions` with the window's token IDs, `max_tokens: 1`, `temperature: 0`, `prompt_logprobs: K`, fresh `cache_salt`; concurrency 1, resumable. Per position it stores the top-K ids/logprobs and the actual token's logprob and rank.
- **K = 20.** A fixed 20% subset (seeded, stratified by category) is also collected with K = 100 for R0 and Cm to check stability in K.
- **Generation** (`collect_generation.py`): greedy, `max_tokens: 1024`, top-K logprobs per generated token.
- **Speculation check:** Cp (speculation on) against Cm-nospec, exact match of the greedy decode set. DFlash2 is lossless if all tokens match up to numerical ties; divergences are listed with their logprob margin.
- **SparkCache check:** the same prompts cold and on replay with the same salt; generated-token logprobs compared.
- **Corruption probe (N, and Cp as control):** 40 Italian prompts plus the tool-call gate set at temperature 0. It reports invalid UTF-8 byte-token runs (U+FFFD in detokenised text, and token IDs that do not decode to valid UTF-8), repetition locks (a repeated n-gram of 4+ tokens over the last 25% of the output), and tool-call parse failures.

## 7. Metrics and statistics

- **Coarse-grained KL**, per position: KL(P_ref ‖ P_cand) over the partition {each token in top-K_ref ∩ top-K_cand} ∪ {rest}. The actual token is added as a known cell when both sides report it. By the data-processing inequality this is a **lower bound** on the full-vocabulary KL, and it is labelled as such wherever it is reported.
- **Top-1 agreement; exact ΔNLL** of the actual token.
- **Generation:** KL and top-1 along the identical greedy prefix, including the first divergent position; position of first divergence; survival of identical prefixes.
- **Aggregates:** mean, median, p90, p99, p99.9, max. Breakdowns by source/category, context-position bucket (0–2K, 2–8K, 8–32K, 32–64K, 64K+) and prefill path. The path tag assumes 8,192-token chunking at concurrency 1; chunks < 2,048 rows took Marlin.
- **CIs:** 95% bootstrap over windows (not tokens), B = 2,000, seed 20260927, token-weighted ratio estimator. **Floor:** R0 run B vs run A on the same boot, plus the cross-boot subset. **Excess** = comparison − floor on the same windows, with paired window resampling.
- **MDE** for the mean excess KL: 2.8 × the bootstrap SE of the floor mean (80% power, α = 0.05), reported with the results.
- **Tasks:** exact McNemar on paired greedy outcomes; paired bootstrap CI (items resampled) on the difference in per-item pass rate over sampled runs; Wilson CIs per arm; z.ai repeat agreement. With 75 qeval items the paired MDE is far above 2 pp: a discordance rate near 5% already puts the approximate MDE at 7–8 pp. The task test can therefore establish only the absence of a *significant* difference, not equivalence at 2 pp. The report says so.

## 8. Decision thresholds (pre-registered)

**Negligible deviation (Cm vs R0)** when all three hold:
- the 95% CI upper bound of the mean excess KL is < 0.002 nats;
- the p99 excess is < 0.02 nats;
- the top-1 agreement drop is < 0.5 pp.

**Tasks** pass when both hold:
- no significant R0-vs-Cp difference (exact McNemar, α = 0.05) and a point estimate within 2 pp;
- Cp within 3 pp of Z.

**Cpre and N** have no pass/fail thresholds. The same metrics are reported against R0 and
against Cm, on the same scale.

**Ladder negative controls.** The rungs from E22b onward (`le22b-m` → `cm`) change only the
drafter, scheduler or KV pool. In measurement mode their target-logit deviation must sit on
the noise floor. A control outside the floor's CI flags the harness, not the recipe.

**Integrity gates, never traded for speed.** Health, gates and identity on every boot:
- every window has 0 missing actual-token entries, or they are counted and excluded;
- no prompt logprobs are suppressed by cache hits;
- rank-0 MemAvailable stays ≥ 1 GiB.

## 9. Tasks and showcase (phases 5, 6)

- **Tasks.** `knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4` at `dddb0347` (MIT, repo-owned files only), vendored unmodified in `third_party/knapcio-bench/`. Our own endpoint layer is in `scripts/fidelity/tasks/`. Sets: qeval (75 deterministic tasks), hardset (30 prompts, no committed grader, reported descriptively), tasktime (6 agentic repos via OpenCode; R0, Cp, N, Z only).
- **Settings:** `reasoning_effort: high` and `max_tokens: 16384` for every item, overriding the set's own per-task low/high and budgets (both deviations recorded). Greedy: local arms once at temperature 0; Z three times with `do_sample: false`. Sampled: temperature 1.0, top_p 0.95, 5 runs per item per arm, seeded locally.
- **Order within the window:** greedy on every arm first; then sampled qeval for R0 and Cp (the thresholded pair); then Cpre and N; hardset sampled last. Tasktime runs if the window allows.
- **Voxel.** Two public prompts, fetched at run time and sent byte-identical to every arm. The first is Artificial Analysis Pagoda Bench (three.js 0.160.0, `window.__VOXEL__`). The second is the classic community voxel-pagoda prompt; x.com returns 402, so the source actually used is recorded. Settings: one greedy and three sampled runs per prompt and arm, `max_tokens` 65,536, `reasoning_effort: high`. Each output is rendered headless at 1,280×800: console errors, a screenshot at 5 s, a frame count (relative only; software WebGL) and the Pagoda Bench constraint checks. A blind gallery uses shuffled labels; the key is kept separately. This section is a visual sanity check; no precision conclusion is drawn from it.

## 10. Time and cost estimate

| Block | Estimate |
| --- | ---: |
| Return to GLM + smoke | 1–1.5 h |
| Boots (~9 × 15–25 min) | 2.5–3.5 h |
| Prompt logprobs (R0 A+B, Cm, Cpre, N, subsets; ~1.2M positions per full run) | 2–3 h (throughput measured in the smoke test) |
| Decode-set generation (5 arms × 150 × ≤ 1,024 tokens, concurrency 1) | ~4–5 h |
| Model-native generation | ~0.5 h |
| NVFP4 download (182 GiB) and image pull | 1–3 h, overlapped with other boots |
| Tasks, greedy, all arms | ~4 h |
| Tasks, sampled qeval ×5 (R0, Cp; then Cpre, N) | ~5–6 h per arm at concurrency 4–6 |
| Tasktime (4 arms × 3 runs × 6 tasks) | ~5 h |
| Voxel (4 local arms × 2 prompts × 4 runs) | ~3 h |

The full protocol needs about 40–50 cluster hours. It cannot finish in one night. The first
window does phases 3–4 and greedy tasks. Sampled task runs and tasktime continue in later
windows, and the report states what ran. **z.ai cost** at $0.15 / $0.50 per M input/output
tokens (list price; reasoning bills as output): tasks about $3, tasktime about $1–2, voxel
under $0.50. Suggested cap: **$10**. The Z arm needs an owner-set cap before it runs.

## 11. Risks

- The rank-0 memory ceiling (about 2 GiB MemAvailable under E29, earlyoom below 0.5 GiB). This is why the KV pool is reduced, `MAX_NUM_SEQS=1` is set, the sampler runs with a 1 GiB abort, and no client runs on rank 0.
- The engine may reject `prompt_logprobs` with DFlash2 (→ `-nospec` overlays) or BF16 KV with the SparkCache connector (→ R0 falls back to FP8 KV, reported).
- New overlays compile fresh CUDA graphs. Longer boots, within the 35-minute readiness limit.
- The NVFP4 recipe's image may need its own NCCL. Any host change goes through astra and is reverted before the final boot.
- A worker's disk could fill with SparkCache namespaces (§2).
- Z's serving stack is unknown and may change during the campaign.

## 12. Owner inputs

| Input | State |
| --- | --- |
| SSH targets, maintenance window, authority | ranks 0–3 given, with a campaign window; stopping the other workload, audited cleanup, production deploys and node actions explicitly authorised |
| astra | `codex exec -m gpt-6-astra`, reasoning effort max |
| Other workload on the ranks | found; recorded privately; stopped |
| Hermes logs | no (synthetic Italian corpus) |
| `reasoning_effort` for tasks | `high` (default) |
| Corpus exclusions | **missing**: measurement runs on the provisional manifest (§5) |
| z.ai spending cap | **missing**: Z arm deferred |

## Amendments

All amendments were recorded before the affected measurements ran.

1. **Phase 3 result (2026-09-27).** The E29 default is serving again:
   - The rank-0 checkpoint was restored and passes the full manifest (72 files, SHA-256 every file).
   - Deploy passed, fabric 8/8 jumbo, both gates passed 4 s after `/health` 200, and `check-f0.py` PASS.
   - The previous workload had disabled rank-0 `tp4-autostart`, so `check-f0.py` failed until an audited change re-enabled it, without starting it.
   - `scripts/fetch-fp8-weights.sh` could not find its manifests when run from `~/tp4` as documented. This is fixed.
2. **Smoke result.** With DFlash2 active, `prompt_logprobs` (K=20) and generation `logprobs` both return complete top-K data for every position. Speculative decoding therefore **stays on in every measurement arm**. The `-nospec` overlays are used only for the losslessness check: greedy decode set, spec off, on a 50-prompt subset of the E29 recipe (`cm-nospec`), compared by exact match against Cp.
3. **Smoke on the production recipe used 256-token prompts.** Rank 0 has about 2.5 GiB MemAvailable under E29, and a long prompt with prompt logprobs materialises its full logits. The long-prompt cache-hit test (≥ 4,096 tokens, same salt twice) and the SparkCache namespace sizing move to the first measurement-mode boot (6 GiB KV).
4. **Corpus.** The 30 Italian conversations each render to about 5.2K tokens, too short for two disjoint ≥ 3,072-token windows. Each therefore also gets one overlapping second-half window (≥ 2,560 tokens), giving 49 Italian windows. The model-native count rises from 48 to **60** prompts, so the total reaches ≥ 300 windows. Before model-native: 254 windows, 1.99M tokens, split agentic 59%, long 16%, structured 15%, Italian 11%.
5. **Boot order.** E29 must restart anyway to use its own cache namespace (`cp-s`), and the model-native windows must exist before any scoring. The campaign therefore starts with **R0 serving** (`r0-s`: native generation, then R0 tasks and voxel), then R0/Cm/Cpre measurement, then `cp-s`, then N. The final boot is unchanged.
6. **R0 cannot use BF16 KV.** The image's `mla_attention.py` (`_canonicalize_sparse_mla_kv_cache_dtype`) maps `auto`, `fp8` and `fp8_e4m3` to `fp8_ds_mla` for the B12X backend. The first R0 boot, with `KV_CACHE_DTYPE=auto`, logged `Using fp8_ds_mla KV cache format for B12X backend` and 1,435,070 KV tokens at 16 GiB, which is FP8 capacity.
   - BF16 KV would need a different attention backend and therefore different kernels than the vendor recipe as served.
   - The pre-registered fallback applies: **R0 uses the FP8 KV** of the frozen September 18 recipe (`r0fp8-m` for measurement; the running `r0-s` boot is already equivalent) and shares the FP8-KV error, and the report states this.
   - The ladder's first rung is R0 itself.
7. **Incident on the first R0 serving boot.** The NVFP4 image pull and checkpoint download ran while R0 served the concurrency-6 model-native generation.
   - The pulls cut MemAvailable by 2–3 GiB per node, and earlyoom killed `VLLM::Worker_TP` on two ranks.
   - No measurement was lost: the 60 model-native requests failed and are rerun.
   - Recovery was a coordinated four-rank stop. The downloads were completed and verified with the stack down; the rule is that no download, pull or copy ever runs on a serving node.
   - Model-native generation reruns at concurrency 4, the concurrency validated for production memory, and stops on the sampler's abort file.
8. **R0 serving needs a smaller KV pool.**
   - Without the hybrid-KDA INT8 projections, the September 18 recipe leaves rank 0 with only 1.98 GiB idle at a 16 GiB pool.
   - During the concurrency-4 model-native generation, rank 0 fell to 0.91 GiB. The sampler aborted and rank 0 stayed at 0.93 GiB when idle.
   - The stack was stopped in a coordinated way. `r0-s` now uses a 12 GiB KV pool, the value of the published NVFP4 recipe. Pool size changes capacity only, not numerics.
   - The 4 model-native generations completed before the abort are kept.
9. **Model-native windows.** R0 (`r0-s`) continuations of the 60 session decode prompts and the 50 public ones were short. Most agentic contexts end in a brief tool call, and the public tasks need little reasoning: 50,425 generated tokens, median about 250.
   - 60 public long-form prompts written for the campaign (`scripts/fidelity/native_prompts.json`, rendered by `build_corpus.py native-prompts`) were therefore added: temperature 1.0, top_p 0.95, fixed seeds, up to 8,192 tokens, concurrency 4.
   - Final provisional corpus: 424 windows, 2.51M positions.
   - Shares: agentic 47%, model-native 21% (170 windows, 284,386 R0-generated tokens), long context 13%, structured 12%, Italian 8%.
   - The Italian share stays below target: 30 conversations of about 5K tokens.
   - Subsets, seeded and stratified: K=100 and cross-boot 85 windows each, ladder 128, decode no-spec 50, replay 20, floor 30.
10. **Order within phase 4.** R0 tasks and voxel runs are deferred to a later `r0-s` boot. Alone they are not a comparison, and the distribution measurements (R0 A/B, Cm) decide the main verdict. The measurement boots therefore come first.
11. **Cache hits and the SparkCache store (first measurement boot).** An 8,000-token prompt was sent three times with `prompt_logprobs=20`:

   | Request | Prompt logprobs returned |
   | --- | --- |
   | cold | 7,999/7,999 positions (6.2 s) |
   | repeat with the same salt | 1,087 positions (1.3 s): 6,912 cached tokens were restored and their logprobs were not returned |
   | fresh salt | 7,999/7,999 |

   - A cache hit therefore suppresses prompt logprobs, which confirms the fresh-salt rule.
   - The store wrote about 89 MB per rank for that prompt, about 28 GB per rank for one full corpus run, more than the workers can hold across runs.
   - Per §2, measurement overlays keep the connector loaded with their own namespace but set `spark_cache_store` and `spark_cache_restore` to false. Serving overlays are unchanged.
   - Scoring throughput with prompt logprobs is about 1,300 tokens/s, so one full run takes about 32 min.
12. **R0 repeatability floor (recorded before any candidate data).** R0 run B against run A, same boot, all 424 windows, 2,514,441 positions:
   - **Positions 0–2K:** bit-identical (KL 0, top-1 agreement 100%).
   - **Positions beyond 2K:** strongly nondeterministic.
     - Mean coarse KL is 0.26–0.36 nats per bucket (p99 4.4–5.7).
     - Top-1 agreement is 77–81%.
     - Overall: mean 0.192 nats [0.170, 0.215], top-1 86.2%.
   - `scripts/fidelity/determinism_probe.py` (sequence X X X Y X Y Y, fresh salts) shows that every pass differs from every other, whatever request preceded it. Back-to-back repeats diverge too: the first non-identical position is 2,052, and max |Δ| reaches 17–28 nats. It is therefore run-to-run nondeterminism, not a dependence on earlier requests.
   - The onset coincides with the sparse-attention indexer's 2,048-token top-k.
   - Pre-registered response:
     - (a) every prompt comparison is also reported **separately for positions ≤ 2,048 (dense regime, zero floor) and > 2,048 (sparse regime)**;
     - (b) the negligible-deviation thresholds are evaluated on excess over the floor in each regime and overall, as specified;
     - (c) the determinism probe runs on every measurement arm (Cm, Cpre, N), since the NVFP4 arm uses a different engine build and indexer;
     - (d) this nondeterminism is itself a finding of the report. Its root cause is not investigated in this campaign unless the owner asks.
13. **Independent review of the plan (astra, verdict "sound with changes"; adopted by the owner before any candidate analysis).**
    - **Sparse regime.** Beyond 2,048 conditioning tokens, excess over the R0 floor is reported only as *additional operational disagreement*: a less variable arm can show negative excess despite systematic drift. Sparse-regime fidelity is estimated by averaging each arm's per-position distributions over **at least three executions** on the stratified `crossboot` subset (85 windows) and comparing the averaged distributions, with uncertainty. Without that, the sparse-regime verdict is **unresolved**.
    - **Coverage.** Coarse KL is a lower bound, and a small value does not certify a small full-vocabulary KL. Every comparison also reports the covered probability mass (sum over shared cells on each side) and the top-K overlap size. Any "within margin" verdict is scoped to the measured coarse metrics.
    - **Statistics.**
      - The bootstrap unit becomes the **source session, conversation or prompt family**, since overlapping windows from one source are not independent.
      - Uncertainty and MDE come from paired contrasts over repeated reference-only executions.
      - Cross-boot repeats are mandatory for **both R0 and Cm**.
    - **Definitions.**
      - p99 excess = p99(R0 vs arm) − p99(R0 vs R0).
      - Agreement drop = agreement(R0, R0) − agreement(R0, arm).
      - Mean excess, p99 excess and agreement drop each need a 95% upper bound, per regime.
      - Outcomes are *within margin*, *exceeds margin* or *unresolved*.
    - **Speculation check.** Cm versus Cm-nospec, otherwise identically configured, with on/on and off/off repeats, replaces Cp versus Cm-nospec. Logprob delivery with DFlash2 only establishes API compatibility. Every mismatch is kept, and ties are defined as equal top-1 logprobs within 1e-6.
    - **Negative controls** run unconditionally on a small subset. They are judged by a paired contrast against a tolerance derived from the repeated R0 contrasts. A failure is attributed among configuration, numerical execution and collection; it does not automatically indict the harness.
    - **Production bridge.** A bounded Cm↔Cp comparison (same prompts, production scheduling) and a cold↔replay control are added, one factor at a time. Without them, the distribution verdict is scoped to cold, serial measurement mode.
    - **Tasks.**
      - Equivalence requires the paired 95% CI of the pass-rate difference to lie **entirely inside ±2 pp** (±3 pp for Cp versus Z); otherwise the result is *unresolved*. With 75 qeval items the paired MDE is about 7 pp.
      - Because greedy decoding is now stochastic locally too, local greedy runs are repeated three times.
      - Sampled runs are analysed clustered by item.
      - Truncation, runtime failure and grading failure stay distinct outcomes.
    - **Wording.**
      - R0 is the September 18 stack with shared FP8 KV, not a BF16 reference.
      - N measures that checkpoint-plus-engine recipe, including its reported defect. A corrected engine would be a separate arm.
      - Z is behavioural context only.
    - **Details.**
      - Model-native windows report continuation-only positions separately.
      - Generation metrics report survival denominators and censoring; comparisons beyond the first divergence use teacher-forced fixed continuations only.
      - The 2,048 boundary is defined by conditioning-token count, and the indexer explanation remains a hypothesis.
      - UTF-8 validity is checked on concatenated token bytes, and repetition heuristics are flags to adjudicate.
      - Contrasts use identical valid-position masks, with missingness published per arm and regime.
14. **N launcher fabric fix and revised remaining order.**
    - The first `n-m` boot failed at NCCL QP setup (`ibv_modify_qp` timeout) because the launcher still carried the published recipe's fabric-specific `NCCL_IB_MERGE_NICS=0` and `NCCL_SWITCHLESS_RING_ONLY=1`. The stack was stopped in a coordinated way. Both variables were removed: fabric layer ours, exactly as this repository's September 11 lane ran the same image.
    - The second boot passed both gates.
    - Remaining measurement boots, each with the determinism probe and, per amendment 13, two further executions (`prompt-rep2`, `prompt-rep3`) on the `crossboot` subset:
      1. N (`prompt-a`, reps);
      2. Cpre (`prompt-a`, reps);
      3. ladder rungs `l0919-m`, `le21-m`, `le22b-m` (`prompt-ladder`; the dense regime is deterministic on the R10 stack, so one execution attributes the ≤ 2K deviation);
      4. Cm cross-boot (reps, K=100, decode set) with `cm-nospec` for the speculation check;
      5. R0 cross-boot (`prompt-crossboot`, decode set and its floor).
    - Then the serving boots (Cp, R0, Cpre, N) for tasks and voxel runs, and the final E29 default.
15. **Cross-boot result and owner-approved descoping (2026-09-27).**
    - **Cross-boot.** Cm scored on a second boot against its first boot (34 windows so far):
      - dense regime **bit-identical** (69,632 positions: KL 0, top-1 100%, every logprob equal);
      - sparse regime KL 0.321 and top-1 77.3%, the known nondeterminism.
    - **Consequence.** The dense-regime distances between recipes are recipe effects, not boot effects: Cm/R0 0.184, Cpre/R0 0.182, Cm/Cpre 0.182, top-1 about 83.5%. The ladder is therefore kept, and `le22b-m` versus `le21-m` is a strict negative control that must be exactly zero in the dense regime.
    - The near-equal pairwise distances indicate that any numerical change is amplified to a similar deviation. Quality differences are therefore read from ΔNLL.
    - **Descoped to shorten the campaign from about 16 to about 6 cluster hours**, with the owner's approval:
      - Decode-set generations (`gen-a`, `gen-floor`) and the DFlash2 exact-match check (`cm-nospec`, `gen-nospec`) are dropped. With every numerical change amplified, greedy continuations diverge within a few tokens, so prefix and exact-match statistics add little beyond the prompt-logprob results.
      - R0 cross-boot is reduced to `prompt-crossboot`, its third execution for the sparse estimator.
      - The serving phase covers only Cp, R0 and N. It runs qeval greedy once (the equivalence verdict stays *unresolved* at this power) and voxel greedy only, plus the corruption probe on N and on Cp as control. Cpre serving, hardset, sampled runs and tasktime are not run.
      - The Z arm stays deferred: there is no spending cap.
16. **Voxel showcase deferred (owner, 2026-09-27).** The voxel pagoda runs are removed from this campaign and recorded as a possible next step. Prompts and pipeline are ready: `scripts/fidelity/voxel/`, two greedy runs per arm, about 15 min per serving boot. Priority goes to the quantitative evidence and the reliability of the report. The serving phase keeps qeval greedy on Cp, R0 and N, plus the corruption probe on N and Cp.
