#!/usr/bin/env python3
"""Check the vendored SparkCache and SIRCL payload against its pins and provenance."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SPARKCACHE = REPO / "third_party/sparkcache"
SIRCL = REPO / "third_party/sparkring-sircl"
UPSTREAM_CONNECTOR = "394775d48e35a7fb3988ad8e831d349c58f7b98c50720dbe00fad361f9fe533c"
UPSTREAM_ENCODER = "f02e67036f0af6f6df57284c7c78694121a15ff5b5797e5287d34bb21a23a9f5"
SIRCL_SOURCE_COMMIT = "b358a818786d8506086aaaabb9afe464fa2ccb49"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest(path: Path) -> dict[str, str]:
    entries = {}
    for line in path.read_text().splitlines():
        digest, name = line.split(None, 1)
        entries[name.strip()] = digest
    return entries


def reverse(text: str, patch: str) -> str:
    """Undo one unified diff exactly; any context mismatch fails."""
    lines = text.splitlines(keepends=True)
    hunks = re.split(r"^(?=@@ )", patch, flags=re.M)[1:]
    offset = 0
    for hunk in hunks:
        header, *body = hunk.splitlines(keepends=True)
        start = int(re.match(r"@@ -\d+(?:,\d+)? \+(\d+)", header).group(1))
        new = [line[1:] for line in body if line[:1] in " +"]
        old = [line[1:] for line in body if line[:1] in " -"]
        if not new:
            start += 1
        at = start - 1 + offset
        if lines[at:at + len(new)] != new:
            raise AssertionError(f"hunk {header.strip()} does not match")
        lines[at:at + len(new)] = old
        offset += len(old) - len(new)
    return "".join(lines)


class SparkCacheTest(unittest.TestCase):
    def test_files_match_pins(self):
        pins = manifest(REPO / "scripts/node/sparkcache/SHA256SUMS")
        for name in ("spark_context_cache_connector.py", "spark_context_cache_hybrid.py",
                     "spark_context_cache_connector-e03-replay-views.py"):
            self.assertEqual(sha(SPARKCACHE / name), pins[name], name)
        rollback = (REPO / "scripts/prepare-sparkcache.py").read_text()
        self.assertIn(sha(SPARKCACHE / "spark_context_cache_connector-20260918.py"), rollback)

    def test_default_recipe_selects_vendored_files(self):
        recipe = (REPO / "cluster.env.example").read_text()
        for key in ("CONNECTOR", "ENCODER"):
            path = re.search(rf"^SPARKCACHE_{key}='\$HOME/tp4/sparkcache/([^']+)'$", recipe, re.M).group(1)
            pin = re.search(rf"^SPARKCACHE_{key}_SHA256=([0-9a-f]{{64}})$", recipe, re.M).group(1)
            self.assertEqual(sha(SPARKCACHE / path), pin, key)

    def test_patches_reverse_to_upstream(self):
        patches = SPARKCACHE / "patches"
        connector = (SPARKCACHE / "spark_context_cache_connector-e03-replay-views.py").read_text()
        connector = reverse(connector, (patches / "03-connector-replay-views.patch").read_text())
        self.assertEqual(connector, (SPARKCACHE / "spark_context_cache_connector.py").read_text())
        connector = reverse(connector, (patches / "02-connector-memory.patch").read_text())
        self.assertEqual(connector, (SPARKCACHE / "spark_context_cache_connector-20260918.py").read_text())
        connector = reverse(connector, (patches / "01-connector-pending-publication.patch").read_text())
        self.assertEqual(hashlib.sha256(connector.encode()).hexdigest(), UPSTREAM_CONNECTOR)
        encoder = reverse((SPARKCACHE / "spark_context_cache_hybrid.py").read_text(),
                          (patches / "04-encoder-memory.patch").read_text())
        self.assertEqual(hashlib.sha256(encoder.encode()).hexdigest(), UPSTREAM_ENCODER)

    def test_license_and_readme(self):
        self.assertIn("Apache License", (SPARKCACHE / "LICENSE").read_text())
        readme = (SPARKCACHE / "README.md").read_text()
        for path in list(SPARKCACHE.glob("*.py")) + list((SPARKCACHE / "patches").iterdir()):
            self.assertTrue(path.name in readme, f"README does not name {path.name}")


class SirclTest(unittest.TestCase):
    def test_files_match_pins(self):
        pins = manifest(REPO / "scripts/node/sircl/SHA256SUMS")
        vendored = {str(p.relative_to(SIRCL)) for p in SIRCL.glob("bundle/*")}
        vendored |= {str(p.relative_to(SIRCL)) for p in SIRCL.glob("runtime/*")}
        self.assertEqual(vendored | {"runtime/sircl_gid_check.py"}, set(pins))
        for name in vendored:
            self.assertEqual(sha(SIRCL / name), pins[name], name)
        self.assertEqual(sha(REPO / "scripts/sircl_gid_check.py"), pins["runtime/sircl_gid_check.py"])

    def test_internal_pins_agree(self):
        native = sha(SIRCL / "bundle/libspark_transport_capi.so")
        overlay = sha(SIRCL / "bundle/sparkring-overlay-manifest.json")
        bundle = json.loads((SIRCL / "bundle/sircl-bundle-manifest.json").read_text())
        self.assertEqual(bundle["artifacts"]["native"]["sha256"], native)
        self.assertEqual(bundle["artifacts"]["python_overlay_manifest"]["sha256"], overlay)
        self.assertEqual(bundle["source"]["sparkring_commit"], SIRCL_SOURCE_COMMIT)
        common = (SIRCL / "runtime/common.env").read_text()
        entrypoint = (SIRCL / "runtime/entrypoint.sh").read_text()
        for digest in (native, overlay):
            self.assertIn(digest, common)
            self.assertIn(digest, entrypoint)

    def test_notices_and_readme(self):
        self.assertIn("Apache License", (SIRCL / "LICENSE").read_text())
        self.assertIn("SparkRing contributors", (SIRCL / "NOTICE").read_text())
        self.assertTrue((SIRCL / "THIRD_PARTY_NOTICES.md").is_file())
        readme = (SIRCL / "README.md").read_text()
        self.assertIn(SIRCL_SOURCE_COMMIT, readme)
        self.assertIn("sircl_gid_check.py --fabric-ifaces", readme)

    def test_registry_names_every_file(self):
        registry = (REPO / "docs/third-party.md").read_text()
        for path in list(SIRCL.glob("bundle/*")) + list(SIRCL.glob("runtime/*")) + list(SPARKCACHE.glob("*.py")):
            self.assertTrue(path.name in registry, f"docs/third-party.md does not name {path.name}")


ENV = """\
FABRIC_TARGETS=(
  "10.10.1.2 10.10.4.4"
  "10.10.1.1 10.10.2.3"
  "10.10.2.2 10.10.3.4"
  "10.10.3.3 10.10.4.1"
)
NCCL_IB_HCA="rocep1s0f0,rocep1s0f1"
NCCL_IB_GID_INDEX=-1
"""


class SiteFilesTest(unittest.TestCase):
    def run_tool(self, root: Path, env: str, *args: str) -> subprocess.CompletedProcess:
        (root / "cluster.env").write_text(env)
        return subprocess.run(
            [str(REPO / "scripts/sircl-site-files.sh"), "--env", str(root / "cluster.env"),
             "--out", str(root / "out"), *args],
            stdin=subprocess.DEVNULL, capture_output=True, text=True)

    def test_ring_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self.run_tool(root, ENV)
            self.assertEqual(result.returncode, 0, result.stderr)
            site = root / "out/site"
            expected = {0: ("10.10.1.2", "10.10.4.4"), 1: ("10.10.1.1", "10.10.2.3"),
                        2: ("10.10.3.4", "10.10.2.2"), 3: ("10.10.3.3", "10.10.4.1")}
            for rank, (peer0, peer1) in expected.items():
                self.assertEqual((site / f"rank{rank}.env").read_text(),
                                 f"SPARK_TP4_PEER0={peer0}\nSPARK_TP4_PEER1={peer1}\n"
                                 "SPARK_TP4_DEVICE0=rocep1s0f0\nSPARK_TP4_DEVICE1=rocep1s0f1\n"
                                 "SPARK_TP4_GID0=3\nSPARK_TP4_GID1=3\n")
            sums = manifest(site / "SHA256SUMS")
            self.assertEqual(len(sums), 24)
            portable = manifest(REPO / "scripts/node/sircl/SHA256SUMS")
            self.assertEqual(sums["/opt/spark-sircl/libspark_transport_capi.so"],
                             portable["bundle/libspark_transport_capi.so"])
            self.assertEqual(sums["/opt/sircl-serving/rank2.env"], sha(site / "rank2.env"))
            site_pins = manifest(root / "out/SHA256SUMS.site")
            self.assertEqual(site_pins["runtime/SHA256SUMS"], sha(site / "SHA256SUMS"))
            self.assertEqual(len(site_pins), 5)

            refused = self.run_tool(root, ENV)
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("--force", refused.stderr)
            self.assertEqual(self.run_tool(root, ENV, "--force", "--gid-index", "5").returncode, 0)
            self.assertIn("SPARK_TP4_GID0=5\n", (site / "rank0.env").read_text())

    def test_rejections(self):
        bad = {
            "wrong-node-number": ENV.replace("10.10.1.2 10.10.4.4", "10.10.1.3 10.10.4.4"),
            "one-peer": ENV.replace("10.10.1.2 10.10.4.4", "10.10.1.2"),
            "one-device": ENV.replace('"rocep1s0f0,rocep1s0f1"', '"rocep1s0f0"'),
            "bad-gid": ENV.replace("NCCL_IB_GID_INDEX=-1", "NCCL_IB_GID_INDEX=300"),
            "other-ifaces": ENV + 'FABRIC_IFACES="eth1 eth2 eth3 eth4"\n',
        }
        for name, env in bad.items():
            with tempfile.TemporaryDirectory() as tmp:
                result = self.run_tool(Path(tmp), env)
                self.assertNotEqual(result.returncode, 0, name)
                self.assertFalse((Path(tmp) / "out/SHA256SUMS.site").exists(), name)


if __name__ == "__main__":
    unittest.main(verbosity=1)
