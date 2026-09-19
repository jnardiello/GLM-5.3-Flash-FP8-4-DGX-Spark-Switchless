#!/usr/bin/env python3
"""Prepare a replay-only copy reduction from the pinned operator connector.

The connector payload is not distributed. This prepares a separate file; it does
not modify the measured E03 connector, install anything, or restart the service.
"""
import argparse
import ast
import hashlib
from pathlib import Path

BASE_SHA256 = "23c1e05cc3bbabecd6b377b493bddfdd3125341d4a6cf63534d07419fdd73b9c"
CANDIDATE_SHA256 = "5893f8747aa093874c46a0185f93c786265d99b5a4cd8da849a7471132422d66"
ENCODER_SHA256 = "11a2db855306b816a4f3d377d04207cc2d62112a381bcc54162099248bbf75f6"


def transform(data: bytes) -> bytes:
    if hashlib.sha256(data).hexdigest() != BASE_SHA256:
        raise ValueError("expected the pinned September 19 connector")
    old = '''        payloads = decode_page_snapshot(
            layout,
            encoded_pages,
            tuple(len(group) for group in groups),
        )
'''
    new = '''        # page_plan has already validated every extent. Keep views of the
        # authenticated snapshot instead of copying its complete body again.
        # The writable per-layer copy and final stream synchronization below
        # retain their original ownership and completion semantics.
        encoded_view = memoryview(encoded_pages).cast("B")
        payloads = {
            span.layer_name: encoded_view[span.source_start : span.source_end]
            for span in page_plan.spans
        }
'''
    source = data.decode("utf-8")
    if source.count(old) != 1:
        raise ValueError("unexpected restore implementation")
    candidate = source.replace(old, new).encode("utf-8")
    if hashlib.sha256(candidate).hexdigest() != CANDIDATE_SHA256:
        raise ValueError("prepared connector hash differs")
    ast.parse(candidate)
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connector", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="new private file outside tracked payloads")
    args = parser.parse_args()
    candidate = transform(args.connector.read_bytes())
    # Exclusive creation prevents overwriting either the baseline or live E03.
    with args.output.open("xb") as stream:
        args.output.chmod(0o600)
        stream.write(candidate)
    print(f"{CANDIDATE_SHA256}  {args.output.name}")


if __name__ == "__main__":
    main()
