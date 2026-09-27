#!/usr/bin/env python3
"""Teacher-forced prompt scoring: store top-K prompt logprobs per corpus window.

For every window, POST /v1/completions with the token ids as the prompt, max_tokens 1,
temperature 0, prompt_logprobs K and a fresh cache_salt, then store the per-position
distribution under data/fidelity/raw/<arm>/<run>/prompt/. Concurrency 1, resumable,
and stoppable between requests through --abort-file. Output never contains token text.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fidelity_io as fio  # noqa: E402


DEFAULT_MANIFEST = fio.REPO / "data/fidelity/corpus/manifest.json"
DEFAULT_RAW = fio.REPO / "data/fidelity/raw"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base-url", required=True, help="server root, e.g. http://HOST:8000")
    p.add_argument("--arm", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--model", help="model name to request (default: the server's first model)")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    p.add_argument("--subset", type=Path, help="file with one window id per line")
    p.add_argument("--categories", nargs="+")
    p.add_argument("--limit", type=int)
    p.add_argument("--boot-json", type=Path, help="boot identity JSON copied into run.json")
    p.add_argument("--abort-file", type=Path)
    p.add_argument("--retries", type=int, default=5)
    p.add_argument("--backoff", type=float, default=2.0, help="first retry delay in seconds")
    p.add_argument("--timeout", type=float, default=1800.0)
    return p.parse_args(argv)


def is_complete(window_dir: Path, entry: dict, k: int, validate=fio.validate_npz) -> bool:
    sidecar = fio.read_json(window_dir / f"{entry['id']}.json")
    if not sidecar or sidecar.get("status") != "ok" or sidecar.get("K") != k:
        return False
    npz = window_dir / f"{entry['id']}.npz"
    return npz.exists() and validate(npz, fio.PROMPT_ARRAYS, entry.get("n_tokens"), k)


def collect(args, save=fio.save_npz, validate=fio.validate_npz, sleep=None, out=print) -> int:
    manifest, entries = fio.load_manifest(args.manifest, "windows")
    chosen = fio.select_entries(entries, args.subset, args.categories, args.limit)
    base = args.base_url.rstrip("/")
    served = fio.server_model_id(base)
    model = args.model or served
    run_dir = args.raw_root / args.arm / args.run
    window_dir = run_dir / "prompt"
    defaults = {"max_tokens": 1, "temperature": 0, "prompt_logprobs": args.K,
                "return_tokens_as_token_ids": True, "cache_salt": "fresh 32-byte hex per HTTP attempt"}
    record = fio.ensure_run_json(run_dir, {
        "schema": "fidelity-raw/1", "kind": "prompt", "arm": args.arm, "run": args.run,
        "K": args.K, "base_url": fio.redact_host(base), "server_model_id": served,
        "request_model": model, "request_defaults": defaults,
        "manifest_global_sha256": manifest.get("global_sha256"),
    }, args.boot_json)

    def process(index, entry):
        wid = entry["id"]
        try:
            ids = fio.load_entry_tokens(args.manifest, entry)
        except fio.ManifestError as error:
            fio.write_json_atomic(window_dir / f"{wid}.json",
                                  {"status": "manifest_error", "error_message": str(error), "K": args.K})
            out(f"[{index}/{len(chosen)}] {wid} status=manifest_error")
            return "manifest_error"
        body = {"model": model, "prompt": ids, "max_tokens": 1, "temperature": 0,
                "prompt_logprobs": args.K, "return_tokens_as_token_ids": True}
        kwargs = {} if sleep is None else {"sleep": sleep}
        response, sidecar = fio.request_and_record(
            base + "/v1/completions", body,
            {"id": wid, "K": args.K, "n_tokens": len(ids)},
            retries=args.retries, backoff=args.backoff, timeout=args.timeout,
            abort_file=args.abort_file, **kwargs)
        if response is not None:
            try:
                parsed = fio.parse_prompt_logprobs(response, ids, args.K)
                sidecar["prompt_tokens"] = (response.get("usage") or {}).get("prompt_tokens")
                sidecar["missing_actual"] = parsed["missing_actual"]
                save(window_dir / f"{wid}.npz", parsed, fio.PROMPT_ARRAYS)
            except fio.ParseError as error:
                sidecar.update(status="bad_response", error_message=str(error))
        fio.write_json_atomic(window_dir / f"{wid}.json", sidecar)
        out(f"[{index}/{len(chosen)}] {wid} n={len(ids)} status={sidecar['status']} "
            f"attempts={sidecar.get('http_attempts')} elapsed={sidecar['elapsed_s']}s "
            f"missing_actual={sidecar.get('missing_actual', '-')}")
        return sidecar["status"]

    counts = fio.run_items(chosen, process,
                           lambda e: is_complete(window_dir, e, args.K, validate),
                           args.abort_file, out)
    fio.close_run_json(run_dir, record, counts)
    out(f"done: ok={counts['ok']} skipped={counts['skipped']} failed={counts['failed']} "
        f"aborted={counts['aborted']}")
    if counts["aborted"]:
        return 3
    return 1 if counts["failed"] else 0


def main(argv=None) -> int:
    return collect(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
