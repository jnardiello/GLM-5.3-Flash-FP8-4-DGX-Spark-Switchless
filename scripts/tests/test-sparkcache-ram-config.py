#!/usr/bin/env python3
"""Offline payload provenance and four-rank launcher checks for cache RAM limits."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]
VARIANT = REPO / "scripts/node/experiments/e03/sparkcache-ram-budget"
PAYLOAD = REPO / "third_party/sparkcache"
OVERLAY = (VARIANT / "delta.env").read_text()
TEMPLATE = (REPO / "cluster.env.example").read_text()


class CacheRamConfigTests(unittest.TestCase):
    def test_generated_payload_and_pins(self):
        result = subprocess.run(["python3", str(VARIANT / "prepare.py"), "--check"],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        pins = dict(line.split(None, 1)[::-1] for line in
                    (REPO / "scripts/node/sparkcache/SHA256SUMS").read_text().splitlines())
        pins = {key.strip(): value for key, value in pins.items()}
        for name in ("spark_context_cache_connector-ram-budget.py",
                     "spark_context_cache_memory_budget.py"):
            self.assertEqual(hashlib.sha256((PAYLOAD / name).read_bytes()).hexdigest(), pins[name])
        for key, path in (("CONNECTOR", PAYLOAD / "spark_context_cache_connector-ram-budget.py"),
                          ("CONFIG", VARIANT / "kv-transfer-config.json")):
            pin = re.search(rf"^SPARKCACHE_{key}_SHA256=([a-f0-9]+)$", OVERLAY, re.M).group(1)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), pin)
        spec = importlib.util.spec_from_file_location("payload_test", Path(__file__).with_name("test-third-party-payload.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        original = module.reverse((PAYLOAD / "spark_context_cache_connector-ram-budget.py").read_text(),
                                  (PAYLOAD / "patches/05-connector-ram-budget.patch").read_text())
        self.assertEqual(original, (PAYLOAD / "spark_context_cache_connector-e03-replay-views.py").read_text())

    def test_configuration_delta(self):
        reference = json.loads((REPO / "scripts/node/experiments/e03/drafter-w8a16/kv-transfer-config-e22b.json").read_text())
        candidate = json.loads((VARIANT / "kv-transfer-config.json").read_text())
        extra = candidate["kv_connector_extra_config"]
        self.assertEqual(extra.pop("spark_cache_cpu_budget_bytes"), 1 << 30)
        self.assertEqual(extra.pop("spark_cache_min_available_bytes"), 1 << 30)
        self.assertNotEqual(extra["spark_cache_root"], reference["kv_connector_extra_config"]["spark_cache_root"])
        extra["spark_cache_root"] = reference["kv_connector_extra_config"]["spark_cache_root"]
        self.assertEqual(candidate, reference)
        identity = json.loads((REPO / "docs/operational-identities/2026-09-29-sparkcache-protected.json").read_text())
        self.assertEqual(identity["sparkcache"]["cache_namespace"],
                         json.loads((VARIANT / "kv-transfer-config.json").read_text())[
                             "kv_connector_extra_config"]["spark_cache_root"])

    def test_four_rank_launcher_changes_only_connector_and_config(self):
        with tempfile.TemporaryDirectory(prefix="tp4-cpu-budget.") as temporary:
            root = Path(temporary)
            launcher = root / "launch-glm53-tp4.sh"
            shutil.copyfile(REPO / "scripts/launcher/launch-glm53-tp4.sh", launcher)
            shutil.copyfile(REPO / "scripts/node/reference/baseline-20260928-e31.env",
                            root / "rollback-e31.env")
            shutil.copyfile(
                REPO / "scripts/node/reference/operational-20260929-sparkcache-protected.env",
                root / "protected.env")
            env = dict(os.environ, TP4_DRY_RUN="1")
            env.pop("TP4_ENV", None)
            fixture = '\nNODES="n0 n1 n2 n3"\nMGMT_IPS="192.0.2.21 192.0.2.22 192.0.2.23 192.0.2.24"\nMASTER_IP=192.0.2.21\nRELAY_DEST=operator@192.0.2.23\n'

            def launch(rank, overlay=None):
                (root / "cluster.env").write_text(TEMPLATE + fixture)
                selected = dict(env, TP4_ENV=str(overlay)) if overlay else env
                result = subprocess.run(["bash", str(launcher), str(rank)], env=selected,
                                        text=True, capture_output=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stderr)
                return [line[2:] for line in result.stdout.splitlines() if line.startswith("  ")]

            for rank in range(4):
                candidate = launch(rank, "protected.env")
                reference = launch(rank, "rollback-e31.env")
                self.assertEqual(len(reference), len(candidate))
                changes = [(a, b) for a, b in zip(reference, candidate) if a != b]
                self.assertEqual(len(changes), 2, changes)
                self.assertTrue(any("spark_context_cache_connector-ram-budget.py" in b for a, b in changes))
                self.assertTrue(any("sparkcache-ram-budget/kv-transfer-config.json" in b for a, b in changes))

    def test_overlay_refuses_unmatched_base(self):
        for key in ("SPARKCACHE_MODE", "SPARKCACHE_CONNECTOR_SHA256",
                    "SPARKCACHE_CONFIG_SHA256", "SPARKCACHE_ENCODER_SHA256"):
            result = subprocess.run(["bash", "-c", 'set -e\nsource "$1"\nsource "$2"\n' + key + '=wrong\nsource "$3"',
                                     "budget-test", str(REPO / "cluster.env.example"),
                                     str(REPO / "scripts/node/reference/baseline-20260928-e31.env"),
                                     str(VARIANT / "delta.env")],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0, key)
            self.assertIn("requires the current", result.stderr)


if __name__ == "__main__":
    unittest.main()
