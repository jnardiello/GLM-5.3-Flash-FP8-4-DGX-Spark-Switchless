#!/usr/bin/env python3
"""Offline contract for the E29 end-drain candidate; no Torch, GPU, node or network.

E29 is E28b plus a flagged length-finish hold in the scheduler and a flagged idle
coalescing window and trace in the engine core. Covers pins and provenance, that both
overrides are exactly their base plus an additions-only patch, flag parsing, four-rank
launcher parity with the E28b default, and overlay refusals. Behaviour under load needs
the qualified engine and an authorized window.
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
CANDIDATE = REPO / "scripts/node/experiments/e03/end-drain"
PARENT = REPO / "scripts/node/experiments/e03/queued-cadence/scheduler.py"
MANIFEST = json.loads((CANDIDATE / "manifest.json").read_text())
SITE = "/usr/local/lib/python3.12/dist-packages/vllm/"
SCHED_FROM = f"/tp4/experiments/e03/queued-cadence/scheduler.py:{SITE}v1/core/sched/scheduler.py:ro"
SCHED_TO = f"/tp4/experiments/e03/end-drain/scheduler.py:{SITE}v1/core/sched/scheduler.py:ro"
CORE = f"/tp4/experiments/e03/end-drain/core.py:{SITE}v1/engine/core.py:ro"
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
_spec = importlib.util.spec_from_file_location(
    "e27b_test", REPO / "scripts/tests/test-long-prefill-cadence-config.py")
e27b_test = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e27b_test)


def functions(path: Path, names: set[str]) -> dict:
    """Execute selected top-level functions of an override, with os imported."""
    tree = ast.parse(path.read_text(), str(path))
    body = [copy.deepcopy(n) for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace: dict = {"os": os}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class Provenance(unittest.TestCase):
    def test_pins(self):
        sums = dict(reversed(line.split("  ", 1)) for line in
                    (CANDIDATE / "SHA256SUMS").read_text().splitlines())
        self.assertEqual(set(sums), {"core.py", "manifest.json", "scheduler.py"})
        for name, digest in sums.items():
            self.assertEqual(sha(CANDIDATE / name), digest, name)
        for name, entry in MANIFEST["files"].items():
            self.assertEqual(entry["candidate_sha256"], sha(CANDIDATE / name), name)
            self.assertEqual(entry["patch_sha256"], sha(CANDIDATE / entry["patch"]), name)
        self.assertEqual(MANIFEST["files"]["scheduler.py"]["parent_sha256"], sha(PARENT))
        self.assertEqual(MANIFEST["parent"]["baseline_sha256"], sha(
            REPO / "docs/historical_benchmarks/baselines/2026-09-25-e28b/baseline.json"))
        self.assertEqual(MANIFEST["status"], "promoted")

    def test_overrides_are_base_plus_additions(self):
        for name, base_sha in (("scheduler.py", sha(PARENT)),
                               ("core.py", MANIFEST["files"]["core.py"]["vendor_sha256"])):
            patch = (CANDIDATE / MANIFEST["files"][name]["patch"]).read_text()
            removed = [line for line in patch.splitlines()
                       if line.startswith("-") and not line.startswith("---")]
            self.assertEqual(removed, [], name)
            text = (CANDIDATE / name).read_text()
            for before, after in e27b_test.hunks(patch):
                self.assertEqual(text.count(after), 1, f"{name}: {after[:80]}")
                text = text.replace(after, before)
            self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), base_sha, name)

    def test_vendor_guard_is_kept(self):
        text = (CANDIDATE / "scheduler.py").read_text()
        self.assertEqual(text.count(
            "and request.num_computed_tokens + 2 - request.num_output_placeholders"), 1)
        guard = text.index("request.num_computed_tokens + 2 - request.num_output_placeholders")
        self.assertLess(guard, text.index("if self._e29_would_hold(request):"))


class Flags(unittest.TestCase):
    def setUp(self):
        self.saved = {k: os.environ.pop(k, None) for k in
                      ("VLLM_E29_END_DRAIN", "VLLM_E29_TRACE", "VLLM_E29_IDLE_COALESCE_MS")}

    def tearDown(self):
        for key, value in self.saved.items():
            os.environ.pop(key, None)
            if value is not None:
                os.environ[key] = value

    def test_scheduler_flag(self):
        flag = functions(CANDIDATE / "scheduler.py", {"_e29_flag"})["_e29_flag"]
        self.assertFalse(flag("VLLM_E29_END_DRAIN"))
        os.environ["VLLM_E29_END_DRAIN"] = "1"
        self.assertTrue(flag("VLLM_E29_END_DRAIN"))
        os.environ["VLLM_E29_END_DRAIN"] = "yes"
        with self.assertRaises(ValueError):
            flag("VLLM_E29_END_DRAIN")

    def test_core_flags(self):
        ns = functions(CANDIDATE / "core.py", {"_e29_idle_coalesce_seconds", "_e29_trace"})
        window, trace = ns["_e29_idle_coalesce_seconds"], ns["_e29_trace"]
        self.assertEqual(window(), 0)
        self.assertFalse(trace())
        os.environ["VLLM_E29_IDLE_COALESCE_MS"] = "3"
        self.assertAlmostEqual(window(), 0.003)
        os.environ["VLLM_E29_IDLE_COALESCE_MS"] = "5"
        self.assertAlmostEqual(window(), 0.005)
        for bad in ("-1", "5.5", "50"):
            os.environ["VLLM_E29_IDLE_COALESCE_MS"] = bad
            with self.assertRaises(ValueError):
                window()
        os.environ["VLLM_E29_TRACE"] = "1"
        self.assertTrue(trace())
        os.environ["VLLM_E29_TRACE"] = "true"
        with self.assertRaises(ValueError):
            trace()


class Launcher(unittest.TestCase):
    def test_four_rank_parity_and_refusals(self):
        delta = (CANDIDATE / "delta.env").read_text()
        with tempfile.TemporaryDirectory(prefix="tp4-e29-") as temp:
            root = Path(temp)
            shutil.copyfile(REPO / "scripts/launcher/launch-glm53-tp4.sh", root / "launch.sh")
            (root / "cluster.env").write_text((REPO / "cluster.env.example").read_text() + '''
NODES="n0 n1 n2 n3"
MGMT_IPS="192.0.2.21 192.0.2.22 192.0.2.23 192.0.2.24"
MASTER_IP=192.0.2.21
RELAY_DEST=operator@192.0.2.23
''')
            e27c = (REPO / "scripts/node/reference/baseline-20260925-e27c.env").read_text() + "\n"
            # The promoted E29 default already carries the overlay; the candidate was measured
            # on the E28b recipe, restored here by its complete rollback.
            e28b = (REPO / "scripts/node/reference/baseline-20260925-e28b.env").read_text() + "\n"
            (root / "empty.env").write_text(e28b)
            (root / "candidate.env").write_text(e28b + delta)
            (root / "candidate-b.env").write_text(e28b + (CANDIDATE / "delta-b.env").read_text())
            (root / "default.env").write_text("")
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

            home = str(Path.home())
            for rank in range(4):
                before, after = argv(launch("empty.env", rank)), argv(launch("candidate.env", rank))
                self.assertEqual(Counter(before) - Counter(after), Counter([home + SCHED_FROM]))
                self.assertEqual(Counter(after) - Counter(before), Counter(
                    [home + SCHED_TO, "-v", home + CORE,
                     "-e", "VLLM_E29_END_DRAIN=1", "-e", "VLLM_E29_IDLE_COALESCE_MS=0",
                     "-e", "VLLM_E29_TRACE=1"]))
                self.assertFalse(any('"' in arg for arg in after if "end-drain" in arg))
                load_b = argv(launch("candidate-b.env", rank))
                self.assertEqual(Counter(load_b) - Counter(after), Counter(
                    ["VLLM_E29_IDLE_COALESCE_MS=4", "VLLM_E29_TRACE=0"]))
                self.assertEqual(Counter(after) - Counter(load_b), Counter(
                    ["VLLM_E29_IDLE_COALESCE_MS=0", "VLLM_E29_TRACE=1"]))

            decoy = f"-e DUMMY=\\$HOME{SCHED_FROM}"   # literal $HOME, as in the mount
            (root / "decoy.env").write_text(e28b + f'EXTRA_DOCKER_ENV="{decoy} $EXTRA_DOCKER_ENV"\n' + delta)
            words = argv(launch("decoy.env"))
            self.assertIn(f"DUMMY=$HOME{SCHED_FROM}", words)   # -e values are not expanded
            self.assertIn(home + SCHED_TO, words)
            self.assertNotIn(home + SCHED_FROM, words)

            bad = {
                "on-e27c-rollback": e27c + delta,
                "on-e29-default": delta,
                "applied-twice": e28b + delta + "\n" + delta,
                "kv-15-gib": e28b + 'EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS/--kv-cache-memory-bytes=17179869184/--kv-cache-memory-bytes=16106127360}"\n' + delta,
                "five-drafts": e28b + "SPEC_TOKENS=5\n" + delta,
                "async-forced": e28b + "ASYNC_SCHEDULING=1\n" + delta,
                "core-override-present": e28b + f'EXTRA_DOCKER_ENV+=" -v /x/core.py:{SITE}v1/engine/core.py:ro"\n' + delta,
                "flag-present": e28b + 'EXTRA_DOCKER_ENV+=" -e VLLM_E29_END_DRAIN=0"\n' + delta,
                "no-e27c-flag": e28b + 'EXTRA_DOCKER_ENV="${EXTRA_DOCKER_ENV/ -e VLLM_E27C_CADENCE_WHEN_QUEUED=1/}"\n' + delta,
                "short-prefill-20480": e28b + 'EXTRA_DOCKER_ENV="${EXTRA_DOCKER_ENV/VLLM_E27B_SHORT_PREFILL_TOKENS=2048/VLLM_E27B_SHORT_PREFILL_TOKENS=20480}"\n' + delta,
                "e27c-flag-twice": e28b + 'EXTRA_DOCKER_ENV+=" -e VLLM_E27C_CADENCE_WHEN_QUEUED=0"\n' + delta,
                "k-hi-twice": e28b + 'EXTRA_DOCKER_ENV+=" -e VLLM_ADAPTIVE_K_HI=5"\n' + delta,
                "second-scheduler-mount": e28b + f'EXTRA_DOCKER_ENV+=" -v /x/scheduler.py:{SITE}v1/core/sched/scheduler.py:ro"\n' + delta,
                "kv-twice": e28b + 'EXTRA_VLLM_ARGS+=" --kv-cache-memory-bytes=17179869184"\n' + delta,
                "kv-on-a-second-line": e28b + "EXTRA_VLLM_ARGS=\"$EXTRA_VLLM_ARGS\"$'\\n--kv-cache-memory-bytes=16106127360'\n" + delta,
                "docker-env-on-a-second-line": e28b + "EXTRA_DOCKER_ENV=\"$EXTRA_DOCKER_ENV\"$'\\n-e VLLM_ADAPTIVE_K_HI=5'\n" + delta,
                "env-long-form": e28b + 'EXTRA_DOCKER_ENV+=" --env=VLLM_ADAPTIVE_K_HI=5"\n' + delta,
                "env-attached": e28b + 'EXTRA_DOCKER_ENV+=" -eVLLM_E27C_CADENCE_WHEN_QUEUED=0"\n' + delta,
                "second-scheduler-mount-rw": e28b + f'EXTRA_DOCKER_ENV+=" -v /x/scheduler.py:{SITE}v1/core/sched/scheduler.py:rw"\n' + delta,
                "kv-inside-another-argument": e28b + 'EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS/--kv-cache-memory-bytes=17179869184/--served-model-name=--kv-cache-memory-bytes=17179869184}"\n' + delta,
            }
            for rank in range(4):
                self.assertEqual(argv(launch("default.env", rank)), argv(launch("candidate-b.env", rank)))

            for name, text in bad.items():
                (root / "bad.env").write_text(text)
                self.assertNotEqual(launch("bad.env").returncode, 0, name)

    def test_readme_names_the_overlay(self):
        text = (CANDIDATE / "README.md").read_text()
        self.assertIn("TP4_ENV=scripts/node/experiments/e03/end-drain/delta.env", text)
        self.assertIn("TP4_ENV=scripts/node/experiments/e03/end-drain/delta-b.env", text)
        for flag in ("VLLM_E29_END_DRAIN", "VLLM_E29_IDLE_COALESCE_MS", "VLLM_E29_TRACE"):
            self.assertIn(flag, text)


if __name__ == "__main__":
    unittest.main(verbosity=1)
