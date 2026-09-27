# Fidelity measurement harness

These tools measure how far a serving recipe (an "arm") moves the model's next-token
distributions away from a reference arm. They score a frozen token corpus
teacher-forced through the vLLM OpenAI-compatible `/v1/completions` endpoint, collect
greedy generations on a decode-path prompt set, and compare arms with a coarse top-K
KL divergence, top-1 agreement and delta NLL, bootstrapped over windows.

The corpus, raw logprobs and per-position metrics are private: they live under the
ignored `data/fidelity/` directory. Only aggregate JSON without token ids or text is
written to `docs/fidelity/metrics/`. No tool prints or stores token text; progress
lines carry window ids and numbers only. Salts are stored as SHA-256 hashes, except
in the private replay-salt files described below.

## Requirements

- The collectors and `analyze.py` need Python 3.10+ with `numpy` (they write and read
  `.npz` files). `mem_sampler.py` and the offline tests are stdlib-only.
- The server must allow `--max-logprobs` of at least K, and the corpus windows must fit
  `MAX_MODEL_LEN`. Prompt logprobs materialise vocabulary-sized logits per prefill
  chunk on the head rank; leave room in the KV-cache budget.
- Run clients from a workstation or a non-head node, never on rank 0.
- `tasks/run_tasks.py --set qeval` grades answers with the vendored knapcio checkers,
  which execute model-generated Python with the caller's permissions (isolated
  interpreter flags, a temporary directory and a timeout are not a sandbox). Run it in a
  disposable container, VM or unprivileged account without credentials or private files.

## Inputs

`data/fidelity/corpus/manifest.json` (schema `fidelity-corpus/1`) lists the scoring
windows:

```json
{"schema": "fidelity-corpus/1", "model_repo": "...", "model_rev": "...",
 "tokenizer_sha256": "...", "chat_template_sha256": "...", "template_kwargs": {},
 "frozen": true, "global_sha256": "...",
 "windows": [{"id": "w0001", "category": "agentic_code", "source": "claude_code",
              "project": "...", "n_tokens": 12345, "sha256": "...", "path": "tokens/w0001.u32"}]}
```

Token files are raw little-endian `uint32`. `sha256` covers the raw file bytes;
`global_sha256` is the SHA-256 of the UTF-8 text formed by one `"<id>:<sha256>\n"` line
per window, sorted by id. The collectors verify both before sending anything.
`decode_manifest.json` has the same shape with the key `prompts` (ids `d001`, ...);
each prompt is already rendered through the assistant generation prefix.

## Raw outputs

Each collection writes `data/fidelity/raw/<arm>/<run>/run.json` with the arm, run,
K, the base URL with its host replaced by `<host>`, the served model id from
`/v1/models`, start/end times (UTC) per session, the harness Git commit, the request
defaults and, when `--boot-json` is given, that file's content verbatim.

Prompt scoring writes `prompt/<window_id>.npz`:

| Array | Type | Content |
| --- | --- | --- |
| `ids` | int32[n] | the scored window tokens |
| `lp_actual` | float32[n] | logprob of the actual token; NaN at index 0 or when missing |
| `rank_actual` | int32[n] | its vLLM rank; -1 when unknown |
| `topk_ids` | int32[n,K] | top-K token ids by rank; -1 padding; row 0 is padding |
| `topk_lp` | float32[n,K] | their logprobs; -inf padding |

The sidecar `<window_id>.json` records the status (`ok`, `client_error`,
`transient_error`, `bad_response`, `manifest_error`), HTTP attempts, elapsed seconds,
`prompt_tokens` from usage, positions whose actual token was missing from the answer,
the SHA-256 of the last attempt's `cache_salt` and K.

Generation writes `gen/<prompt_id>.npz` with `gen_ids` int32[m], `lp_actual`
float32[m], `topk_ids` int32[m,K] and `topk_lp` float32[m,K], plus a sidecar with the
finish reason, usage, elapsed seconds and K.

## Commands

Start the memory sampler first. Hosts are given in rank order; when rank 0 drops
below `--abort-gib`, the sampler creates the abort file and the collectors stop
cleanly before their next request:

```sh
umask 077
python3 scripts/fidelity/mem_sampler.py --hosts <rank0> <rank1> <rank2> <rank3> \
  --out data/fidelity/raw/<arm>/mem-<run>.jsonl --abort-gib 1.0 \
  --abort-file data/fidelity/ABORT
python3 scripts/fidelity/mem_sampler.py --summary data/fidelity/raw/<arm>/mem-<run>.jsonl
```

Score the corpus (defaults: K 20, all windows; `--subset FILE`, `--categories`,
`--limit` select windows):

```sh
PY=data/fidelity/.venv/bin/python
$PY scripts/fidelity/collect_prompt_logprobs.py --base-url http://<api-host>:<port> \
  --arm Cm --run prompt-a --K 20 --boot-json <boot-identity.json> \
  --abort-file data/fidelity/ABORT
```

Each request is `{"model", "prompt": [ids], "max_tokens": 1, "temperature": 0,
"prompt_logprobs": K, "return_tokens_as_token_ids": true, "cache_salt"}`. Collect
greedy generations (default `--max-tokens 1024`):

```sh
$PY scripts/fidelity/collect_generation.py --base-url http://<api-host>:<port> \
  --arm Cm --run gen-a --K 20 --abort-file data/fidelity/ABORT
```

Both collectors run at concurrency 1, retry connection errors and HTTP 5xx with
exponential backoff (`--retries`, `--backoff`), record a 4xx answer and move on, and
exit 1 if any item failed or 3 after an abort. They resume: an item whose sidecar
says `ok` and whose array file loads with the expected shapes is skipped; other items
are requested again. Files are written through a temporary file and a rename. A
resumed run refuses a different arm, K, kind or served model.

Every HTTP attempt carries a fresh random 32-byte `cache_salt`, so neither the vLLM
prefix cache nor an external KV cache can replay an earlier computation. For a
cache-replay check, pass the same private salt file to a cold and a replay run:

```sh
$PY scripts/fidelity/collect_generation.py ... --arm Cp --run cold \
  --replay-salt data/fidelity/raw/Cp/replay-salts.json
$PY scripts/fidelity/collect_generation.py ... --arm Cp --run replay \
  --replay-salt data/fidelity/raw/Cp/replay-salts.json
```

The salt file is created with mode `0600` when missing and copied into each run
directory as `replay-salts.json`; keep it private.

## Analysis

`analyze.py --comparisons FILE` reads a JSON list such as
[`comparisons.example.json`](comparisons.example.json):

```json
[{"name": "prompt-floor-r0", "cand": "R0/prompt-b", "ref": "R0/prompt-a", "floor": null, "kind": "prompt"},
 {"name": "prompt-cm-vs-r0", "cand": "Cm/prompt-a", "ref": "R0/prompt-a", "floor": "prompt-floor-r0", "kind": "prompt"}]
```

```sh
$PY scripts/fidelity/analyze.py --comparisons comparisons.json [--k-pairs k-pairs.json]
$PY scripts/fidelity/analyze.py --selftest
```

It writes per-position arrays to `data/fidelity/metrics/<name>.npz` (mode `0600`),
aggregate JSON to `docs/fidelity/metrics/<name>.json`, and a combined
`docs/fidelity/metrics/summary.json`. `--selftest` builds a synthetic corpus and
raw runs in a temporary directory and runs the whole pipeline.

Metrics per position:

- **Coarse KL** `KL(P_ref || P_cand)` over the partition formed by every token in both
  top-K sets, the actual token as its own cell when it lies outside that intersection
  but is known on both sides (prompt scoring only), and one "rest" cell holding the
  remaining mass on each side. `-inf` logprobs are probability zero; a NaN or missing
  row makes the position invalid (excluded and counted); a negative rest within 1e-6
  is clamped to zero and counted; a positive reference mass against zero candidate
  mass uses a 1e-12 floor and is counted.
- **Top-1 agreement** of the two top-K rows and **delta NLL** (`NLL_cand - NLL_ref` for
  the actual token).
- Generations are compared along their identical greedy prefix: positions up to and
  including the first divergent token, or the end of the shorter sequence. The
  first-divergence position is reported per prompt.

Breakdowns cover category, source, context-position bucket (0-2K, 2-8K, 8-32K,
32-64K, 64K+ tokens) and prefill path. The path tag assumes 8,192-token prefill
chunks from position zero: a chunk shorter than 2,048 rows is tagged `marlin`
(W8A16 path), otherwise `bf16` (dequantised path).

Aggregates report the mean, median, p90, p99, p99.9 and maximum. Confidence intervals
are 95% percentile bootstraps over windows, not tokens (B = 2000, seed 20260927),
using the token-weighted ratio estimator. With a `floor` comparison over the same
windows, the report adds the mean excess KL (paired window resampling), the p99
excess, the top-1 agreement drop in percentage points, and the minimum detectable
effect at 80% power and two-sided alpha 0.05: 2.8 times the bootstrap SE of the
floor's mean.

`--k-pairs` takes `[{"name", "low", "high"}]`, naming two prompt comparisons of the
same arms scored at a low and a high K. The report compares their mean KL and, on
the high-K data alone, the effect of truncating the rows to the low K.

## Caveat: a lower bound

Merging tokens into cells can only lose information, so by the data-processing
inequality the coarse KL is a lower bound on the true KL between the full
distributions. Mass moving among tokens outside both top-K sets is invisible. Use the
K-sensitivity comparison to show how much the bound tightens with a larger K, and
report the values as lower bounds.

## Tests

```sh
python3 scripts/tests/test-fidelity-metrics.py
python3 scripts/tests/test-fidelity-collect.py
```

Both run in `./scripts/check.sh` with the system interpreter. They use a localhost
fake server and need no cluster, network or numpy; with numpy installed they also
check the fast path and the npz files.
