#!/usr/bin/env python3
"""Offline contract for the E27c queued-cadence candidate; no Torch, GPU, node or network.

E27c is E27b plus one flag that keeps the prefill cadence while requests are queued. Covers
pins and vendor provenance, that the patch is exactly E27b's plus the latch override, the
flag parsing, four-rank launcher parity with the E27 default, and overlay refusals.
Scheduling behaviour under load needs the qualified engine and an authorized window.
"""

from __future__ import annotations

import ast
from collections import Counter
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[2]
E27B = REPO / "scripts/node/experiments/e03/long-prefill-cadence"
CANDIDATE = REPO / "scripts/node/experiments/e03/queued-cadence"
OVERRIDE = CANDIDATE / "scheduler.py"
PATCH = CANDIDATE / "scheduler.patch"
MANIFEST = json.loads((CANDIDATE / "manifest.json").read_text())
FILE = MANIFEST["files"]["scheduler.py"]
MOUNT = ("/tp4/experiments/e03/queued-cadence/scheduler.py:"
         "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py:ro")
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
_spec = importlib.util.spec_from_file_location(
    "e27b_test", REPO / "scripts/tests/test-long-prefill-cadence-config.py")
e27b_test = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e27b_test)


def added(patch: Path) -> list[str]:
    return [line[1:] for line in patch.read_text().splitlines()
            if line.startswith("+") and not line.startswith("+++")]


class Provenance(unittest.TestCase):
    def test_pins(self):
        sums = dict(reversed(line.split("  ", 1)) for line in
                    (CANDIDATE / "SHA256SUMS").read_text().splitlines())
        self.assertEqual(set(sums), {"manifest.json", "scheduler.py"})
        for name, digest in sums.items():
            self.assertEqual(sha(CANDIDATE / name), digest, name)
        self.assertEqual(FILE["candidate_sha256"], sha(OVERRIDE))
        self.assertEqual(FILE["patch_sha256"], sha(PATCH))
        self.assertEqual(MANIFEST["includes"]["manifest_sha256"], sha(E27B / "manifest.json"))
        self.assertEqual(MANIFEST["status"], "promoted")

    def test_override_is_vendor_plus_patch(self):
        text = OVERRIDE.read_text()
        for before, after in e27b_test.hunks(PATCH.read_text()):
            self.assertEqual(text.count(after), 1, after[:80])
            text = text.replace(after, before)
        self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), FILE["vendor_sha256"])

    def test_patch_is_e27b_plus_latch_override(self):
        removed = [line[1:].strip() for line in PATCH.read_text().splitlines()
                   if line.startswith("-") and not line.startswith("---")]
        self.assertEqual(removed, [
            "throttle_prefills and not self.prefill_capacity_bound",
            "if defer_prefills and request.is_prefill_chunk:",
            "elif defer_prefills and num_computed_tokens < request.num_tokens - 1:",
        ])
        extra = Counter(added(PATCH)) - Counter(added(E27B / "scheduler.patch"))
        self.assertEqual(Counter(added(E27B / "scheduler.patch")) - Counter(added(PATCH)),
                         Counter())
        text = "\n".join(extra.elements())
        self.assertIn("not self.prefill_capacity_bound or self.e27c_cadence_when_queued", text)
        # The latch is still computed by the vendor line; E27c never assigns it.
        self.assertNotIn("prefill_capacity_bound =", text)


class Flag(unittest.TestCase):
    def test_environment(self):
        tree = ast.parse(OVERRIDE.read_text(), str(OVERRIDE))
        node = next(copy.deepcopy(n) for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name == "_e27c_cadence_when_queued")
        ns: dict = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(OVERRIDE), "exec"), ns)
        read = ns["_e27c_cadence_when_queued"]
        saved = os.environ.pop("VLLM_E27C_CADENCE_WHEN_QUEUED", None)
        try:
            self.assertFalse(read())
            os.environ["VLLM_E27C_CADENCE_WHEN_QUEUED"] = "1"
            self.assertTrue(read())
            os.environ["VLLM_E27C_CADENCE_WHEN_QUEUED"] = "yes"
            with self.assertRaises(ValueError):
                read()
        finally:
            os.environ.pop("VLLM_E27C_CADENCE_WHEN_QUEUED", None)
            if saved is not None:
                os.environ["VLLM_E27C_CADENCE_WHEN_QUEUED"] = saved


class Launcher(unittest.TestCase):
    def test_four_rank_parity_and_refusals(self):
        delta = (CANDIDATE / "delta.env").read_text()
        with tempfile.TemporaryDirectory(prefix="tp4-e27c-") as temp:
            root = Path(temp)
            shutil.copyfile(REPO / "scripts/launcher/launch-glm53-tp4.sh", root / "launch.sh")
            (root / "cluster.env").write_text((REPO / "cluster.env.example").read_text() + '''
NODES="n0 n1 n2 n3"
MGMT_IPS="192.0.2.21 192.0.2.22 192.0.2.23 192.0.2.24"
MASTER_IP=192.0.2.21
RELAY_DEST=operator@192.0.2.23
''')
            e22b = (REPO / "scripts/node/reference/baseline-20260924-e22b.env").read_text() + "\n"
            # The promoted E27c default already carries the scheduler; the candidate was
            # measured on the E27 recipe, restored here by its complete rollback.
            e27 = (REPO / "scripts/node/reference/baseline-20260924-e27.env").read_text() + "\n"
            (root / "empty.env").write_text(e27)
            (root / "candidate.env").write_text(e27 + delta)
            env = dict(os.environ, TP4_DRY_RUN="1")
            env.pop("TP4_ENV", None)
            forbidden = root / "forbidden.log"
            bindir = root / "bin"
            bindir.mkdir()
            for name in ("sudo", "docker", "ssh", "systemctl", "curl", "ip", "sysctl"):
                p = bindir / name
                p.write_text('#!/bin/sh\nprintf "%s\\n" "$0" >> "$TP4_FORBIDDEN"\nexit 97\n')
                p.chmod(0o700)
            env.update(PATH=str(bindir) + os.pathsep + env["PATH"], TP4_FORBIDDEN=str(forbidden))

            def launch(overlay, rank=0):
                result = subprocess.run(["bash", str(root / "launch.sh"), str(rank)],
                                        env=dict(env, TP4_ENV=overlay), capture_output=True,
                                        text=True, timeout=20)
                self.assertFalse(forbidden.exists(), "Dry-run attempted an external action")
                return result

            def argv(result):
                self.assertEqual(result.returncode, 0, result.stderr)
                return [line[2:] for line in result.stdout.splitlines() if line.startswith("  ")]

            for rank in range(4):
                before, after = argv(launch("empty.env", rank)), argv(launch("candidate.env", rank))
                self.assertEqual(Counter(before) - Counter(after), Counter())
                self.assertEqual(Counter(after) - Counter(before), Counter(
                    ["-v", str(Path.home()) + MOUNT, "-e", "VLLM_E27B_SHORT_PREFILL_TOKENS=2048",
                     "-e", "VLLM_E27C_CADENCE_WHEN_QUEUED=1"]))
                self.assertEqual(after[after.index("--prefill-schedule-interval") + 1], "8")

            e27b_delta = (E27B / "delta.env").read_text()
            bad = {
                "on-e22b-rollback": e22b + delta,
                "on-e27c-default": delta,
                "applied-twice": e27 + delta + "\n" + delta,
                "on-e27b": e27 + e27b_delta + "\n" + delta,
                "cadence-twice": e27 + 'EXTRA_VLLM_ARGS+=" --prefill-schedule-interval 8"\n' + delta,
                "threshold-present": e27 + 'EXTRA_VLLM_ARGS+=" --long-prefill-token-threshold=2304"\n' + delta,
                "max-seqs-5": e27 + "MAX_NUM_SEQS=5\n" + delta,
            }
            for name, text in bad.items():
                (root / "bad.env").write_text(text)
                self.assertNotEqual(launch("bad.env").returncode, 0, name)

    def test_readme_names_the_overlay(self):
        text = (CANDIDATE / "README.md").read_text()
        self.assertIn("TP4_ENV=scripts/node/experiments/e03/queued-cadence/delta.env", text)
        self.assertIn("VLLM_E27C_CADENCE_WHEN_QUEUED", text)


if __name__ == "__main__":
    unittest.main(verbosity=1)
