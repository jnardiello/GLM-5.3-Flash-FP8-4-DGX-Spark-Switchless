#!/usr/bin/env python3
"""Reproduce the pinned cache memory fixes from the original connector and encoder.

The results are included in third_party/sparkcache/; this tool derives them from the
upstream encoder and the connector with the pending-publication patch. Both original
and resulting hashes are checked before any output is written. The original connector
is kept for rollback.
"""

import argparse
import ast
import hashlib
from pathlib import Path
import sys


CONNECTOR = "spark_context_cache_connector.py"
ENCODER = "spark_context_cache_hybrid.py"
ORIGINAL = {
    CONNECTOR: "a0bedc1c33a316d3c56ae857652c4a17acac9a82b053a4a71689684f33b75745",
    ENCODER: "f02e67036f0af6f6df57284c7c78694121a15ff5b5797e5287d34bb21a23a9f5",
}
PREPARED = {
    CONNECTOR: "23c1e05cc3bbabecd6b377b493bddfdd3125341d4a6cf63534d07419fdd73b9c",
    ENCODER: "11a2db855306b816a4f3d377d04207cc2d62112a381bcc54162099248bbf75f6",
}
ROLLBACK = "spark_context_cache_connector-20260918.py"


def checked(data: bytes, expected: str, label: str) -> bytes:
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"{label}: SHA-256 mismatch; no files written")
    return data


def transform_connector(source: str) -> str:
    """Move one commit into a call frame whose payload aliases die before waiting."""
    start = source.index("    def _store_worker_main(self) -> None:\n")
    body_start = source.index("            commit_started = time.perf_counter()\n", start)
    end = source.index("    def _publish_row_prefix_aliases(\n", body_start)
    body = source[body_start:end]
    # This continue belongs to the outer queue loop, which becomes a single call.
    if body.count("                continue\n") != 1:
        raise ValueError("unexpected saver control flow")
    body = body.replace("                continue\n", "                return\n")
    body = "".join(line[4:] if line.startswith("    ") else line
                   for line in body.splitlines(keepends=True))
    replacement = '''    def _store_worker_main(self) -> None:
        while True:
            snapshot = self._store_queue.get()
            if snapshot is None:
                return
            try:
                self._commit_store_snapshot(snapshot)
            finally:
                # Do not retain the previous payload while queue.get() blocks.
                del snapshot

    def _commit_store_snapshot(
        self, snapshot: _StoreSnapshot | _HybridStoreSnapshot
    ) -> None:
        """Commit one item so every payload alias dies before the next wait."""
'''
    return source[:start] + replacement + body + source[end:]


def transform_encoder(source: str) -> str:
    old = 'return encode_page_snapshot_header(layout, counts) + b"".join(parts)'
    if source.count(old) != 1:
        raise ValueError("unexpected encoder layout")
    return source.replace(old, 'return b"".join([encode_page_snapshot_header(layout, counts), *parts])')


def prepare(connector: Path, encoder: Path, output: Path) -> None:
    inputs = {CONNECTOR: connector, ENCODER: encoder}
    originals = {name: checked(path.read_bytes(), ORIGINAL[name], name)
                 for name, path in inputs.items()}
    transforms = {CONNECTOR: transform_connector, ENCODER: transform_encoder}
    outputs = {}
    for name, transform in transforms.items():
        data = transform(originals[name].decode("utf-8")).encode("utf-8")
        checked(data, PREPARED[name], f"prepared {name}")
        ast.parse(data, name)
        outputs[name] = data
    outputs[ROLLBACK] = originals[CONNECTOR]
    # Refuse to replace unrelated files; repeating the same preparation is safe.
    for name, data in outputs.items():
        path = output / name
        if path.is_symlink() or (path.exists() and path.read_bytes() != data):
            raise ValueError(f"output already exists with different content: {path}")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    for name, data in outputs.items():
        path = output / name
        if not path.exists():
            with path.open("xb") as stream:
                stream.write(data)
            path.chmod(0o600)
        print(f"{hashlib.sha256(data).hexdigest()}  {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connector", required=True, type=Path,
                        help="original connector with the pending-publication patch (September 18 pin)")
    parser.add_argument("--encoder", required=True, type=Path,
                        help="original hybrid encoder from the pinned R10 image")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="private staging directory; existing matching outputs are accepted")
    args = parser.parse_args()
    try:
        prepare(args.connector, args.encoder, args.output_dir)
    except (OSError, ValueError, SyntaxError) as error:
        print(f"prepare-sparkcache: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
