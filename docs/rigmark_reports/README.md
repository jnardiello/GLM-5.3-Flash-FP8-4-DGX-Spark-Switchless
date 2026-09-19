# Native Rigmark reports

Every benchmark run for this repository must supply `--output` with an absolute
filename inside this directory. Run Rigmark from its independent checkout and set
`TP4_REPO_ROOT` to the absolute path of **this** repository, not the Rigmark checkout:

```sh
TP4_REPO_ROOT=/absolute/path/to/this/repository

# Add this argument to the complete native ./rigmark run command:
--output "$TP4_REPO_ROOT/docs/rigmark_reports/<date-experiment>/run-<n>.json"
```

Use one dated, descriptive directory per experiment and a distinct filename for
each complete suite execution, such as `run-1.json`, `run-2.json`, and `run-3.json`.
An IaC reproduction uses its own directory. Never overwrite a previous run or reuse
an excluded run's filename. Rigmark creates the parent directories when saving a
result. Keep these local directories mode `0700` and their receipts/logs mode `0600`
(use `umask 077` before running the command).

Rigmark normally defaults to `results/<label>-<timestamp>.json` relative to its
working directory. The explicit output rule avoids writing to the wrong checkout
and keeps this repository's results together. A successful run also saves
`run-<n>.card.txt` beside its JSON. A failed or interrupted run may have only a
partial JSON or log; preserve what exists and record the missing evidence.

## View an existing result

From the Rigmark checkout, print the result card without making inference requests:

```sh
./rigmark report "$TP4_REPO_ROOT/docs/rigmark_reports/<date-experiment>/run-1.json"
```

Add `--save` to regenerate the `.card.txt` file. Open the JSON for the native
measurements, per-request rows and metadata; open `.card.txt` for the short readable
summary. `./rigmark compare <first.json> <second.json>` prints a native comparison
of compatible receipts. Preserve any reported protocol mismatches.

## Local originals and public history

Only this README is tracked here. Native JSON, cards and logs are ignored by Git
because they retain complete outputs, request settings and operator metadata.
Keep originals unchanged; identify them by SHA-256 when producing public extracts.

Publish portable numeric results under `docs/historical_benchmarks`, readable
reports under `docs/benchmarks`, and figures under `docs/plots`. Include every
documented outcome, including discarded, incomplete and excluded runs, without
putting them into accepted baseline medians. See the
[benchmark index](../benchmarks/README.md) and
[agent benchmark procedure](../../AGENTS.md#optimization-workflow).
