#!/usr/bin/env python3
"""Reproduce the CPU-budget connector and its provenance diff locally."""
from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
PAYLOAD = REPO / "third_party/sparkcache"
BASE = "spark_context_cache_connector-e03-replay-views.py"
BASE_SHA = "5893f8747aa093874c46a0185f93c786265d99b5a4cd8da849a7471132422d66"
OUTPUT = "spark_context_cache_connector-ram-budget.py"


def generate():
    base = (PAYLOAD / BASE).read_text()
    if hashlib.sha256(base.encode()).hexdigest() != BASE_SHA:
        raise ValueError("expected the pinned E03 replay connector")
    old = "                self._store_queue.put(snapshot)\n                logger.info("
    if base.count(old) != 1:
        raise ValueError("unexpected producer queue handoff")
    result = base.replace(old, "                self._enqueue_budgeted_snapshot(snapshot)\n                logger.info(")
    budget = (PAYLOAD / "spark_context_cache_memory_budget.py").read_text()
    budget = budget.replace("from __future__ import annotations\n", "")
    stream = Path(__file__).with_name("stream_io.py").read_text()
    stream = stream.replace("from __future__ import annotations\n", "")
    bounded = Path(__file__).with_name("bounded_connector.py").read_text()
    result += "\n\n# Project-authored CPU budget (embedded; no unpinned runtime import).\n" + budget
    result += "\n\n" + stream + "\n\nManifestStore = StreamingManifestStore\n"
    result += "\n\n" + bounded
    ast.parse(result)
    patch = "".join(difflib.unified_diff(base.splitlines(True), result.splitlines(True),
                                      fromfile=BASE, tofile=OUTPUT))
    return {PAYLOAD / OUTPUT: result,
            PAYLOAD / "patches/05-connector-ram-budget.patch": patch}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for path, content in generate().items():
        if args.check:
            if not path.is_file() or path.read_text() != content:
                raise SystemExit(f"stale generated payload: {path.name}")
        else:
            path.write_text(content)
        print(f"{hashlib.sha256(content.encode()).hexdigest()}  {path.name}")


if __name__ == "__main__":
    main()
