#!/usr/bin/env python3
"""Greedy decode-path collection: store generated ids and top-K logprobs per prompt.

For every decode prompt, POST /v1/completions with temperature 0, max_tokens, logprobs K,
return_tokens_as_token_ids and a cache_salt, then store the result under
data/fidelity/raw/<arm>/<run>/gen/. With --replay-salt the salt per prompt is read from
(or created in) a private salt file so a `cold` and a `replay` run share it.
Output never contains token text.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fidelity_io as fio  # noqa: E402


DEFAULT_MANIFEST = fio.REPO / "data/fidelity/corpus/decode_manifest.json"
DEFAULT_RAW = fio.REPO / "data/fidelity/raw"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base-url", required=True, help="server root, e.g. http://HOST:8000")
    p.add_argument("--arm", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--model", help="model name to request (default: the server's first model)")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    p.add_argument("--subset", type=Path, help="file with one prompt id per line")
    p.add_argument("--categories", nargs="+")
    p.add_argument("--limit", type=int)
    p.add_argument("--boot-json", type=Path, help="boot identity JSON copied into run.json")
    p.add_argument("--replay-salt", type=Path,
                   help="private JSON map prompt id -> salt, created if missing, shared by cold/replay runs")
    p.add_argument("--abort-file", type=Path)
    p.add_argument("--retries", type=int, default=5)
    p.add_argument("--backoff", type=float, default=2.0, help="first retry delay in seconds")
    p.add_argument("--timeout", type=float, default=1800.0)
    return p.parse_args(argv)


def load_salts(path: Path, ids: list[str]) -> dict:
    """Read the salt map, add salts for new ids, and keep it private (0600)."""
    salts = fio.read_json(path) or {}
    missing = [pid for pid in ids if pid not in salts]
    for pid in missing:
        salts[pid] = fio.fresh_salt()
    if missing or not Path(path).exists():
        fio.write_json_atomic(path, salts, mode=0o600)
    return salts


def is_complete(gen_dir: Path, entry: dict, k: int, validate=fio.validate_npz) -> bool:
    sidecar = fio.read_json(gen_dir / f"{entry['id']}.json")
    if not sidecar or sidecar.get("status") != "ok" or sidecar.get("K") != k:
        return False
    npz = gen_dir / f"{entry['id']}.npz"
    return npz.exists() and validate(npz, fio.GEN_ARRAYS, None, k)


def collect(args, save=fio.save_npz, validate=fio.validate_npz, sleep=None, out=print) -> int:
    manifest, entries = fio.load_manifest(args.manifest, "prompts")
    chosen = fio.select_entries(entries, args.subset, args.categories, args.limit)
    base = args.base_url.rstrip("/")
    served = fio.server_model_id(base)
    model = args.model or served
    run_dir = args.raw_root / args.arm / args.run
    gen_dir = run_dir / "gen"
    salts = None
    if args.replay_salt:
        salts = load_salts(args.replay_salt, [e["id"] for e in chosen])
        fio.write_json_atomic(run_dir / "replay-salts.json", salts, mode=0o600)
    defaults = {"max_tokens": args.max_tokens, "temperature": 0, "logprobs": args.K,
                "return_tokens_as_token_ids": True,
                "cache_salt": "replay salt file" if salts else "fresh 32-byte hex per HTTP attempt"}
    record = fio.ensure_run_json(run_dir, {
        "schema": "fidelity-raw/1", "kind": "gen", "arm": args.arm, "run": args.run,
        "K": args.K, "base_url": fio.redact_host(base), "server_model_id": served,
        "request_model": model, "request_defaults": defaults,
        "manifest_global_sha256": manifest.get("global_sha256"),
    }, args.boot_json)

    def process(index, entry):
        pid = entry["id"]
        try:
            ids = fio.load_entry_tokens(args.manifest, entry)
        except fio.ManifestError as error:
            fio.write_json_atomic(gen_dir / f"{pid}.json",
                                  {"status": "manifest_error", "error_message": str(error), "K": args.K})
            out(f"[{index}/{len(chosen)}] {pid} status=manifest_error")
            return "manifest_error"
        body = {"model": model, "prompt": ids, "max_tokens": args.max_tokens, "temperature": 0,
                "logprobs": args.K, "return_tokens_as_token_ids": True}
        kwargs = {} if sleep is None else {"sleep": sleep}
        response, sidecar = fio.request_and_record(
            base + "/v1/completions", body,
            {"id": pid, "K": args.K, "prompt_tokens_sent": len(ids), "max_tokens": args.max_tokens,
             "replay_salt": bool(salts)},
            retries=args.retries, backoff=args.backoff, timeout=args.timeout,
            abort_file=args.abort_file, fixed_salt=salts[pid] if salts else None, **kwargs)
        if response is not None:
            try:
                parsed = fio.parse_generation_logprobs(response, args.K)
                sidecar.update(finish_reason=parsed["finish_reason"], usage=parsed["usage"],
                               gen_tokens=len(parsed["gen_ids"]))
                save(gen_dir / f"{pid}.npz", parsed, fio.GEN_ARRAYS)
            except fio.ParseError as error:
                sidecar.update(status="bad_response", error_message=str(error))
        fio.write_json_atomic(gen_dir / f"{pid}.json", sidecar)
        out(f"[{index}/{len(chosen)}] {pid} status={sidecar['status']} "
            f"gen={sidecar.get('gen_tokens', '-')} finish={sidecar.get('finish_reason', '-')} "
            f"attempts={sidecar.get('http_attempts')} elapsed={sidecar['elapsed_s']}s")
        return sidecar["status"]

    counts = fio.run_items(chosen, process, lambda e: is_complete(gen_dir, e, args.K, validate),
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
