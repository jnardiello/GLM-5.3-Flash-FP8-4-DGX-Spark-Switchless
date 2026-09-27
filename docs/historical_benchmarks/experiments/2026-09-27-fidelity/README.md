# Fidelity campaign extract (2026-09-27)

Portable, privacy-safe extract of the GLM-5.3-Flash fidelity campaign: how far the E29
serving recipe (Cm in measurement mode, Cp in production) and the NVFP4 recipe (N)
deviate from the vendor FP8 model served by the September 18 recipe (R0), with the
recipe before E21 (Cpre) and the ladder arms (L0919, LE21, LE22b). The readable report
is [the fidelity report](../../../fidelity/REPORT.md); this folder holds the numbers
behind it. Every file here is generated; do not edit by hand.

## Files

| File | Contents |
| --- | --- |
| `results.json` | Outcome per regime, reference and arm identities, coverage, headline metrics (copied from the public aggregates; perplexity change is exp(ΔNLL) − 1), task and corruption summaries, integrity, exclusions, limitations, digests of the public report files, of every file in this folder and of the private originals |
| `runs.json` | Every measurement run: request settings, redacted boot identity (rank count and image ids, no rank records) and per-run window aggregates by source class |
| `corpus-hashes.json` | One row per corpus window: key, source class, category, token count and content SHA-256 |
| `windows-public.csv` | Model-native windows generated from public prompts: prompt provenance and generation settings |
| `window-metrics-public.csv` | Per-window sums for those windows from every comparison: positions, top-20 KL, top-1 agreement and covered mass, per regime and side (contrast or floor) |
| `window-runs-public.csv` | Per-run records for those windows: status, token counts, attempts, time and cache-salt digest |
| `determinism.json` | Same-arm re-scoring probes per boot: first differing position and largest logprob difference per pass pair |
| `memory.jsonl` | Host memory samples per boot and rank (host names removed) |
| `gates/` | Functional gate results after each boot |
| `probes/` | Corruption probes (UTF-8 and tool-call flags, no text) for Cp and N |
| `tasks/` | Per-item qeval results for R0, Cp and N: public tasks, model answers and grader verdicts |
| `smoke/` | Logprob and cache-hit smoke checks |
| `nvfp4-manifest-hf.json` | File digests of the NVFP4 checkpoint snapshot |

## What stays private

Corpus text, token ids, logprob arrays, transcripts, session logs, reviewer audits,
host inventories, parity captures, tokenizer files and operator runbooks stay in the
ignored campaign data. Windows from private coding sessions, model-native windows from
private prompts and the synthetic Italian conversations appear only in aggregates and
as digests. Their key is `h` plus the first 12 hex digits of the window SHA-256; the
campaign's internal window ids are not published. Public windows use the public prompt
id (knapcio hardset/qeval `d` prompts, `scripts/fidelity/native_prompts.json` `n`
prompts). Host names, addresses, usernames, absolute paths, project names and raw cache
salts are removed; salts appear only as SHA-256 digests.

`private_originals` in `results.json` identifies each private source by its path
relative to the campaign data directory and its SHA-256. A directory of per-window or
per-prompt records is identified by a combined digest: SHA-256 of the sorted lines
`<sha256>  <file name>\n`.

## Regenerate

```sh
data/fidelity/.venv/bin/python scripts/fidelity/export_public.py
```

The script needs the ignored campaign data, numpy for the per-window metrics, and the
public aggregates in `docs/fidelity/`. It builds the folder in a temporary directory,
runs a leak check (absolute paths, addresses, window ids, site host names and
addresses, local username, corpus project names and credential patterns), and replaces
this folder only when the check is clean. Output is deterministic. Allowlisted: IPv4
literals that occur in the public qeval task source, loopback and the unspecified
address `0.0.0.0` in model answers, and the names of the public prompt sets.
