#!/usr/bin/env python3
"""Offline checks of the node verifier's selected operator-payload hashes."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
source = (REPO / "scripts/verify-node.sh").read_text(encoding="utf-8")
match = re.search(r"^check_selected_payload\(\) \{.*?^\}", source, re.M | re.S)
assert match, "selected-payload verifier missing"
function = match.group(0)


def check(path: str, expected: str) -> str:
    result = subprocess.run(
        ["bash", "-c", function + '\ncheck_selected_payload "$1" "$2"',
         "verify-selected-payload", path, expected],
        capture_output=True, text=True, check=True,
    )
    assert not result.stderr, result.stderr
    return result.stdout


with tempfile.TemporaryDirectory(prefix="tp4-verify-payload.") as temp:
    root = Path(temp)
    current = root / "spark_context_cache_connector.py"
    previous = root / "spark_context_cache_connector-20260918.py"
    encoder = root / "spark_context_cache_hybrid.py"
    current.write_text("current connector\n", encoding="utf-8")
    previous.write_text("historical connector\n", encoding="utf-8")
    encoder.write_text("current encoder\n", encoding="utf-8")
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()

    assert check(str(current), sha(current)) == "ok"
    assert check(str(previous), sha(previous)) == "ok"
    assert check(str(current), sha(previous)) == "mismatch"
    assert check(str(encoder), sha(encoder)) == "ok"
    assert check(str(root / "missing.py"), sha(current)) == "missing"
    assert check(str(current), "") == "invalid-pin"
    assert check(str(current), "not-a-sha256") == "invalid-pin"
    assert check("", "") == "disabled"

    # Exercise the launcher's supported home-relative forms without changing HOME.
    relative = os.path.relpath(current, Path.home())
    assert check("$HOME/" + relative, sha(current)) == "ok"
    assert check("~/" + relative, sha(current)) == "ok"
    original_sha = sha(current)
    current.write_text("changed after deployment\n", encoding="utf-8")
    assert check(str(current), original_sha) == "mismatch"

assert "P_SPARKCACHE_CONNECTOR_SHA256='${SPARKCACHE_CONNECTOR_SHA256:-}'" in source
assert "P_SPARKCACHE_ENCODER_SHA256='${SPARKCACHE_ENCODER_SHA256:-}'" in source
print("test-verify-node: PASS")
