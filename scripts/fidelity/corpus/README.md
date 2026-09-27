# Fidelity corpus builder

Builds the teacher-forced scoring windows and the decode-path prompts for the fidelity
campaign from local agent sessions, synthetic Italian conversations and public prompts.
Corpus text and token IDs are private: they stay under the ignored `data/fidelity/corpus/`
directory. The only tracked output is `docs/fidelity/corpus-summary.json`, which holds counts,
length histograms, redaction counts and hashes, with no text, IDs or project names.

The builder prints counts and hashes only. On an unexpected error it prints the exception
type and code locations, never the message, because a message can quote data.

## Commands

Run with the campaign virtual environment (transformers, tokenizers and numpy; no torch):

```sh
PY=data/fidelity/.venv/bin/python
$PY scripts/fidelity/corpus/build_corpus.py build        # parse, redact, render, cut, write
$PY scripts/fidelity/corpus/build_corpus.py verify       # hashes, lengths, ids < vocabulary
$PY scripts/fidelity/corpus/build_corpus.py add-native --gen-dir data/fidelity/raw/<arm>/<run>/gen
$PY scripts/fidelity/corpus/build_corpus.py freeze       # only after the owner confirms exclusions
python3 scripts/tests/test-fidelity-redact.py            # redaction unit tests (stdlib only)
```

`build` refuses to overwrite a frozen manifest unless `--force` is given. `verify` also
runs under the system `python3`; it then checks ids against the configured vocabulary
size (154,880) instead of the tokenizer's.

## Inputs

| Input | Location | Notes |
| --- | --- | --- |
| Claude Code sessions | `~/.claude/projects/<project>/*.jsonl` and `<session>/subagents/*.jsonl` | one file = one session |
| omp sessions | `~/.omp/agent/sessions/<project>/**/*.jsonl` | one file = one session |
| Italian conversations | `data/fidelity/corpus/synthetic-it/it*.json` | `{"id","topic","messages":[{"role","content"}]}` |
| Public prompts | `third_party/knapcio-bench/bench/`: `hardset.py` (30) and `qeval_tasks.py` (20 of 75, seeded, stratified by category) | MIT; `--bench-dir` |
| Tokenizer and runtime template | `data/fidelity/tokenizer/` (`tokenizer.json`, `tp4-chat-template.jinja`) | pinned model revision |

`data/fidelity/corpus/exclude.txt` lists excluded project directories, one per line: a bare
name excludes it from both sources, `claude_code/<name>` or `omp/<name>` from one.

## Pipeline

1. **Parse** ([`sources.py`](sources.py)). Events are read in file order and split into
   segments at context compactions; the compaction summary opens the next segment.
   Hidden `thinking` blocks are dropped (not GLM-native, possibly encrypted), as are harness
   attachments, notifications and metadata events. Images in tool results are dropped.
   Events after the build cutoff are ignored, so a rebuild reads the same session prefix
   even while sessions keep growing; the cutoff is stored in `build-info.json` and reused
   unless `--cutoff` is given.
2. **Redact** ([`redact.py`](redact.py)) every string before rendering: message text, tool
   results and every string inside tool-call arguments. Rules: PEM private keys, JWTs,
   Anthropic/OpenAI/GitHub/Slack/AWS/Google/Hugging Face/GitLab tokens, `Bearer` tokens,
   env-style assignments whose upper-case key contains
   KEY/TOKEN/SECRET/PASS/PASSWORD/AUTH/CREDENTIAL/COOKIE/SESSION (numeric, boolean and
   `$`/`<`/`{` reference values are kept), `password`/`passwd`/`pwd`/`pass` assignments,
   e-mail addresses, and an entropy heuristic. Matches become `[REDACTED:<rule>]`; only
   per-rule counts are written (`redaction-counts.json`).
3. **Convert** ([`convert.py`](convert.py)) to the template's message form: tool calls carry
   `arguments` as a dict (the template iterates `arguments.items()`), tool results are
   `{"role": "tool", "tool_call_id", "content"}`. Tool schemas are synthesised per source
   from the observed argument keys and their most frequent JSON type, with an empty
   description; each segment lists only the tools it calls.
4. **Render and tokenise** ([`render.py`](render.py)) with the runtime template and
   `{"reasoning_effort": "high"}`, exactly as vLLM's chat path: render to a string, then
   `encode(add_special_tokens=False)`. At start-up the builder checks this against
   `apply_chat_template(tokenize=True)` on a synthetic conversation, with and without the
   generation prompt, and aborts on any difference. Windows are assembled from per-unit
   token arrays (a unit is a user, system or assistant message, the assistant unit with its
   tool results); every selected window and prompt is then re-rendered in full and must
   match token for token.
5. **Cut windows** (`build_corpus.py`), deterministic for a given seed (default 20260927)
   and cutoff:
   * `long_context`: one window of 126,976-131,000 tokens and four of 32,768-65,535 tokens
     from the longest segments of different sessions, each starting at the segment start,
     with `L mod 8192` spread over the four quarters of a chunk.
   * The rest of every segment is tiled into disjoint windows starting at unit
     boundaries: half in [3,072, 8,191] and half in [8,704, 10,239] tokens, so the last
     8,192-token prefill chunk has fewer than 2,048 rows.
   * `structured_json`: 40 tiles (20 per band) with the highest share of structured text
     (tool-call markup, short argument values and tool results that parse as JSON), at
     most three per session.
   * `agentic_code`: 160 tiles (80 per band) with at least one tool call, taken round-robin
     across sessions.
   * `italian_chat`: each conversation becomes one window, or two when both halves (the
     second re-rendered with the same system message) reach 3,072 tokens.
   * `model_native`: added later by `add-native` from a harness generation directory:
     decode prompt ids followed by the saved `gen_ids`, for every prompt whose sidecar
     status is `ok`. Re-running replaces the previous native windows. On a frozen
     manifest it needs `--allow-frozen` and changes `global_sha256`.
6. **Decode prompts**: 100 session prompts ending on a human user turn, rendered with the
   generation prompt, between 1,024 and 16,384 tokens (log-uniform target), from message
   ranges disjoint from every scored window, preferring sessions without windows; then the
   50 public prompts as single user turns.

## Entropy heuristic

A candidate is a run of at least 20 characters of `[A-Za-z0-9+/=_-]`. It is redacted when
its Shannon entropy is at least 4.0 bits per character, it uses at least two character
classes, and its non-benign parts total at least 20 characters. Parts are split at
`+/=_-`; decimal numbers, pure hexadecimal up to 40 characters (git SHAs and the groups
of a UUID) and word-like parts are benign. A part is word-like when its alphabetic pieces
(camelCase words or acronyms) average at least three characters, few are one or two
characters long, numbers do not alternate with words, and longer parts contain vowels.
Random tokens change character class every one or two characters and fail these tests,
while paths, identifiers such as `launch_glm53_tp4` or `Glm5NextForCausalLM`, SHAs, UUIDs
and compact timestamps pass. SHA-256 digests are not excluded explicitly: hexadecimal
cannot exceed 4.0 bits per character, so they practically never reach the threshold.
Random identifiers (request or task IDs) are redacted too; that is intended.

## Outputs (`data/fidelity/corpus/`, private, mode 0600)

| File | Contents |
| --- | --- |
| `manifest.json` | `fidelity-corpus/1` window manifest read by the harness |
| `tokens/<id>.u32` | raw little-endian uint32 token ids per window |
| `decode_manifest.json`, `decode/<id>.u32` | decode prompts (key `prompts`, ids `d001`...) |
| `sources-inventory.json` | per `source/project`: sessions, segments, events, message and tool counts, approximate characters and tokens |
| `redaction-counts.json` | redactions per rule, in total and per source |
| `windows-meta.json`, `decode-meta.json` | segment key hash, unit range, band and structured share per id |
| `build-info.json` | cutoff, seed, exclusions, template check, public prompt hashes, shortfalls |
| `native-provenance.json` | decode id, prompt and generated lengths and npz hash per native window |

`global_sha256` is the SHA-256 of the UTF-8 concatenation of `"{id}:{sha256}\n"` over the
entries sorted by id, as computed by [`fidelity_io.py`](../fidelity_io.py).

## Limitations

- Assistant text in the sessions comes from several models (mostly not GLM); only the
  `model_native` windows contain GLM output.
- Literal special-token strings inside message text (for example a quoted `<|user|>`)
  become special tokens, as they do on the server's chat path.
- Branches abandoned after a rewind stay in file order.
- The redaction rules are heuristics; review `redaction-counts.json` before freezing.
