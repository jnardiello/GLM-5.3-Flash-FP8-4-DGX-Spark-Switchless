#!/usr/bin/env python3
"""Offline contract for the E27b long-prefill cadence candidate; no Torch, GPU, node or network.

Covers pins and vendor provenance of the scheduler override, the exact scope of its patch,
the short-prefill predicate and per-step limit, four-rank launcher parity with the E27
default, and overlay refusals. Scheduling behaviour under load needs the qualified engine
and an authorized window.
"""

from __future__ import annotations

import ast
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[2]
CANDIDATE = REPO / "scripts/node/experiments/e03/long-prefill-cadence"
OVERRIDE = CANDIDATE / "scheduler.py"
PATCH = CANDIDATE / "scheduler.patch"
MANIFEST = json.loads((CANDIDATE / "manifest.json").read_text())
FILE = MANIFEST["files"]["scheduler.py"]
MOUNT = ("/tp4/experiments/e03/long-prefill-cadence/scheduler.py:"
         "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py:ro")
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def hunks(patch: str) -> list[tuple[str, str]]:
    """(before, after) text of every hunk in a unified diff."""
    result, before, after = [], None, None
    for line in patch.splitlines(keepends=True):
        if line.startswith(("---", "+++")):
            continue
        if line.startswith("@@"):
            if before is not None:
                result.append(("".join(before), "".join(after)))
            before, after = [], []
        elif line.startswith(" "):
            before.append(line[1:])
            after.append(line[1:])
        elif line.startswith("-"):
            before.append(line[1:])
        elif line.startswith("+"):
            after.append(line[1:])
    result.append(("".join(before), "".join(after)))
    return result


def load(names: set[str], namespace: dict) -> dict:
    """Execute selected real top-level functions of the override."""
    tree = ast.parse(OVERRIDE.read_text(), str(OVERRIDE))
    selected = [copy.deepcopy(node) for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in selected} == names
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(module, str(OVERRIDE), "exec"), namespace)
    return namespace


def method(name: str):
    """One real Scheduler method, compiled against the given globals later."""
    tree = ast.parse(OVERRIDE.read_text(), str(OVERRIDE))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
    return next(copy.deepcopy(n) for n in cls.body
                if isinstance(n, ast.FunctionDef) and n.name == name)


class Provenance(unittest.TestCase):
    def test_pins(self):
        sums = dict(reversed(line.split("  ", 1)) for line in
                    (CANDIDATE / "SHA256SUMS").read_text().splitlines())
        self.assertEqual(set(sums), {"manifest.json", "scheduler.py"})
        for name, digest in sums.items():
            self.assertEqual(sha(CANDIDATE / name), digest, name)
        self.assertEqual(FILE["candidate_sha256"], sha(OVERRIDE))
        self.assertEqual(FILE["patch_sha256"], sha(PATCH))
        self.assertEqual(MANIFEST["status"], "measured_superseded")

    def test_override_is_vendor_plus_patch(self):
        text = OVERRIDE.read_text()
        for before, after in hunks(PATCH.read_text()):
            self.assertEqual(text.count(after), 1, after[:80])
            text = text.replace(after, before)
        self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), FILE["vendor_sha256"])

    def test_patch_only_changes_the_two_deferral_gates(self):
        removed = [line[1:].strip() for line in PATCH.read_text().splitlines()
                   if line.startswith("-") and not line.startswith("---")]
        self.assertEqual(removed, [
            "if defer_prefills and request.is_prefill_chunk:",
            "elif defer_prefills and num_computed_tokens < request.num_tokens - 1:",
        ])
        added = "".join(line[1:] for line in PATCH.read_text().splitlines(keepends=True)
                        if line.startswith("+") and not line.startswith("+++"))
        # The capacity latch and cadence inputs stay the vendor ones.
        for untouched in ("prefill_capacity_bound =", "throttle_prefills and",
                          "has_eligible_decode =", "current_step"):
            self.assertNotIn(untouched, added)


class Predicate(unittest.TestCase):
    def setUp(self):
        self.ns = load({"_e27b_admits", "_e27b_short_prefill_tokens"}, {})

    def test_boundaries(self):
        admits = self.ns["_e27b_admits"]
        self.assertTrue(admits(2047, 0, 2048))
        self.assertFalse(admits(2048, 0, 2048))
        self.assertFalse(admits(2049, 0, 2048))
        self.assertFalse(admits(0, 0, 2048))
        self.assertTrue(admits(1024, 1024, 2048))
        self.assertFalse(admits(1025, 1024, 2048))
        for remaining in (1, 2, 100, 2047, 10**6):
            self.assertFalse(admits(remaining, 0, 0), "disabled must be the vendor gate")

    def test_environment(self):
        read = self.ns["_e27b_short_prefill_tokens"]
        saved = os.environ.pop("VLLM_E27B_SHORT_PREFILL_TOKENS", None)
        try:
            self.assertEqual(read(), 0)
            os.environ["VLLM_E27B_SHORT_PREFILL_TOKENS"] = "2048"
            self.assertEqual(read(), 2048)
            os.environ["VLLM_E27B_SHORT_PREFILL_TOKENS"] = "-1"
            with self.assertRaises(ValueError):
                read()
        finally:
            os.environ.pop("VLLM_E27B_SHORT_PREFILL_TOKENS", None)
            if saved is not None:
                os.environ["VLLM_E27B_SHORT_PREFILL_TOKENS"] = saved

    def test_step_limit_accumulates(self):
        class Log:
            def info(self, *args):
                pass
        ns = dict(self.ns, logger=Log())
        module = ast.Module(body=[method("_e27b_admit")], type_ignores=[])
        exec(compile(module, str(OVERRIDE), "exec"), ns)

        class Scheduler:
            _e27b_admit = ns["_e27b_admit"]
            e27b_admitted = 0
            e27b_deferred = 0
            e27b_step_short_prefill = 0
        scheduler = Scheduler()
        scheduler.e27b_short_prefill_tokens = 2048
        self.assertTrue(scheduler._e27b_admit(1300))
        self.assertFalse(scheduler._e27b_admit(1300))
        self.assertTrue(scheduler._e27b_admit(700))
        self.assertEqual(scheduler.e27b_step_short_prefill, 2000)
        scheduler.e27b_short_prefill_tokens = 0
        scheduler.e27b_step_short_prefill = 0
        self.assertFalse(scheduler._e27b_admit(5))


class Launcher(unittest.TestCase):
    def test_four_rank_parity_and_refusals(self):
        delta = (CANDIDATE / "delta.env").read_text()
        with tempfile.TemporaryDirectory(prefix="tp4-e27b-") as temp:
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
                    ["-v", str(Path.home()) + MOUNT, "-e", "VLLM_E27B_SHORT_PREFILL_TOKENS=2048"]))
                self.assertEqual(after[after.index("--prefill-schedule-interval") + 1], "8")
                self.assertEqual(after[after.index("--max-num-seqs") + 1], "6")
                self.assertEqual(after[after.index("--max-num-batched-tokens") + 1], "8192")

            bad = {
                "on-e22b-rollback": e22b + delta,
                "on-e27c-default": delta,
                "applied-twice": e27 + delta + "\n" + delta,
                "other-cadence": e27 + 'EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS/--prefill-schedule-interval 8/--prefill-schedule-interval 4}"\n' + delta,
                "cadence-twice": e27 + 'EXTRA_VLLM_ARGS+=" --prefill-schedule-interval 8"\n' + delta,
                "threshold-present": e27 + 'EXTRA_VLLM_ARGS+=" --long-prefill-token-threshold=2304"\n' + delta,
                "batch-changed": e27 + "BATCHED_TOKENS=16384\n" + delta,
                "max-seqs-5": e27 + "MAX_NUM_SEQS=5\n" + delta,
                "no-adaptive-k": e27 + 'EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS/ --scheduler-cls adaptive_k_scheduler.AdaptiveKScheduler/}"\n' + delta,
            }
            for name, text in bad.items():
                (root / "bad.env").write_text(text)
                self.assertNotEqual(launch("bad.env").returncode, 0, name)

    def test_readme_names_the_overlay(self):
        text = (CANDIDATE / "README.md").read_text()
        self.assertIn("TP4_ENV=scripts/node/experiments/e03/long-prefill-cadence/delta.env", text)
        self.assertIn("VLLM_E27B_SHORT_PREFILL_TOKENS", text)


if __name__ == "__main__":
    unittest.main(verbosity=1)
