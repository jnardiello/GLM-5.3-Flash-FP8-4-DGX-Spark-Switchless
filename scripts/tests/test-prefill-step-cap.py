#!/usr/bin/env python3
"""Offline contract for the strict opt-in 6,912-token scheduler-output cap."""
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
import time
from types import SimpleNamespace
import unittest


REPO = Path(__file__).resolve().parents[2]
CANDIDATE = REPO / "scripts/node/experiments/e03/prefill-step-cap"
PARENT = REPO / "scripts/node/experiments/e03/end-drain/scheduler.py"
SCHEDULER = CANDIDATE / "scheduler.py"
TRIM_DELTA = REPO / "scripts/node/experiments/e03/prefill-cache-trim/delta.env"
PROTECTED16 = REPO / "scripts/node/reference/operational-20260929-sparkcache-protected.env"
TARGET = "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py"
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()  # noqa: E731


def helper_namespace() -> dict:
    tree = ast.parse(SCHEDULER.read_text(), str(SCHEDULER))
    wanted = {
        "_resilience_step_token_cap",
        "_resilience_effective_token_budget",
        "_resilience_step_receipt",
    }
    body = [copy.deepcopy(node) for node in tree.body
            if (isinstance(node, ast.Assign)
                and any(isinstance(name, ast.Name)
                        and name.id.startswith("_RESILIENCE_")
                        for name in node.targets))
            or (isinstance(node, ast.FunctionDef) and node.name in wanted)]
    namespace = {"os": os, "time": time}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SCHEDULER), "exec"),
         namespace)
    return namespace


def scheduler_method(name: str):
    tree = ast.parse(SCHEDULER.read_text(), str(SCHEDULER))
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == "Scheduler")
    method = next(copy.deepcopy(node) for node in cls.body
                  if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {"Request": object}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(SCHEDULER), "exec"),
         namespace)
    return namespace[name]


class Provenance(unittest.TestCase):
    def test_manifest_patch_and_deployed_hashes(self):
        manifest = json.loads((CANDIDATE / "manifest.json").read_text())
        self.assertEqual(manifest["schema"], "tp4-prefill-step-cap-candidate-v1")
        self.assertEqual(manifest["parent"], {
            "path": "scripts/node/experiments/e03/end-drain/scheduler.py",
            "sha256": sha(PARENT),
        })
        self.assertEqual(manifest["candidate"]["path"],
                         "scripts/node/experiments/e03/prefill-step-cap/scheduler.py")
        self.assertEqual(manifest["candidate"]["sha256"], sha(SCHEDULER))
        self.assertEqual(manifest["candidate"]["patch_sha256"],
                         sha(CANDIDATE / "scheduler.patch"))
        self.assertEqual(manifest["candidate"]["mount_target"], TARGET)
        self.assertEqual(manifest["selection"]["environment"],
                         "VLLM_RESILIENCE_STEP_TOKEN_CAP=6912")
        with tempfile.TemporaryDirectory() as directory:
            rebuilt = Path(directory) / "scheduler.py"
            rebuilt.write_bytes(PARENT.read_bytes())
            result = subprocess.run(
                ["patch", "-s", str(rebuilt), str(CANDIDATE / "scheduler.patch")],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(rebuilt.read_bytes(), SCHEDULER.read_bytes())
        sums = dict(line.split(None, 1)[::-1] for line in
                    (CANDIDATE / "SHA256SUMS").read_text().splitlines())
        self.assertEqual({name.strip() for name in sums},
                         {"scheduler.py", "manifest.json"})
        for name, digest in sums.items():
            self.assertEqual(sha(CANDIDATE / name.strip()), digest)

    def test_launcher_manifest_case_accepts_bundle_and_rejects_corruption(self):
        launcher = (REPO / "scripts/launcher/launch-glm53-tp4.sh").read_text()
        marker = "# The opt-in step cap keeps its scheduler payload separate from the E29 parent.\n"
        start = launcher.index("case ", launcher.index(marker))
        end = launcher.index("\nesac", start) + len("\nesac")
        manifest_case = launcher[start:end]
        with tempfile.TemporaryDirectory() as directory:
            env_dir = Path(directory)
            deployed = env_dir / "experiments/e03/prefill-step-cap"
            deployed.mkdir(parents=True)
            for name in ("scheduler.py", "manifest.json", "SHA256SUMS"):
                (deployed / name).write_bytes((CANDIDATE / name).read_bytes())
            env = dict(os.environ, ENV_DIR=str(env_dir), DRY_RUN="0",
                       EXTRA_DOCKER_ENV="-v /x/prefill-step-cap/scheduler.py:/x:ro")
            good = subprocess.run(["bash", "-c", manifest_case], env=env,
                                  capture_output=True, text=True)
            self.assertEqual(good.returncode, 0, good.stderr)
            (deployed / "scheduler.py").write_bytes(b"corrupt\n")
            bad = subprocess.run(["bash", "-c", manifest_case], env=env,
                                 capture_output=True, text=True)
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn("prefill-step-cap source manifest failed", bad.stderr)


class HelpersAndAccounting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = helper_namespace()
        cls.split = staticmethod(scheduler_method("_mamba_block_aligned_split"))

    def setUp(self):
        self.saved = os.environ.pop("VLLM_RESILIENCE_STEP_TOKEN_CAP", None)

    def tearDown(self):
        os.environ.pop("VLLM_RESILIENCE_STEP_TOKEN_CAP", None)
        if self.saved is not None:
            os.environ["VLLM_RESILIENCE_STEP_TOKEN_CAP"] = self.saved

    def test_strict_flag_budget_and_receipt(self):
        flag = self.ns["_resilience_step_token_cap"]
        budget = self.ns["_resilience_effective_token_budget"]
        self.assertEqual(flag(), 0)
        os.environ["VLLM_RESILIENCE_STEP_TOKEN_CAP"] = "6912"
        self.assertEqual(flag(), 6912)
        for bad in ("1", "6911", "8192", "x"):
            os.environ["VLLM_RESILIENCE_STEP_TOKEN_CAP"] = bad
            with self.assertRaises(ValueError):
                flag()
        self.assertEqual(budget(8192, 0), 8192)
        self.assertEqual(budget(8192, 6912), 6912)
        receipt = self.ns["_resilience_step_receipt"](
            schedule_sequence=9, scheduled_target_tokens=6304,
            num_scheduled_new_requests=2, configured_token_budget=8192,
            effective_token_cap=6912, wall_time_ns=lambda: 11,
            monotonic_ns=lambda: 12, pid_fn=lambda: 13)
        self.assertEqual(receipt, {
            "schema": "prefill-step-cap-v1", "wall_time_ns": 11,
            "monotonic_ns": 12, "pid": 13, "schedule_sequence": 9,
            "scheduled_target_tokens": 6304,
            "num_scheduled_new_requests": 2,
            "configured_token_budget": 8192, "effective_token_cap": 6912,
        })

    def fake_scheduler(self):
        return SimpleNamespace(
            cache_config=SimpleNamespace(block_size=2304), use_eagle=True,
            mamba_has_prefill_checkpoint_blocks=True,
            max_num_scheduled_tokens=8192,
            scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
            hash_block_size=2304, mamba_partial_cache_hit=False)

    @staticmethod
    def request(prompt: int):
        return SimpleNamespace(num_computed_tokens=0, num_prompt_tokens=prompt,
                               num_tokens=prompt, shared_prefix_boundary=0)

    def test_real_alignment_spec_slots_and_preemption_refund(self):
        scheduler = self.fake_scheduler()
        budget = self.ns["_resilience_effective_token_budget"](8192, 6912)
        input_budget, draft_slots = 8192, 8
        long_chunk = self.split(scheduler, self.request(32768),
                                min(32768, budget, input_budget - draft_slots))
        self.assertEqual(long_chunk, 6912)

        first = self.split(scheduler, self.request(4000),
                           min(4000, budget, input_budget - draft_slots))
        budget -= first
        input_budget -= first + draft_slots
        second = self.split(scheduler, self.request(4000),
                            min(4000, budget, input_budget - draft_slots))
        self.assertEqual((first, second), (4000, 2304))
        budget -= second
        input_budget -= second + draft_slots
        self.assertEqual((budget, input_budget), (608, 1872))
        # These are the unchanged scheduler refund expressions for a preempted step.
        budget += second
        input_budget += second + draft_slots
        self.assertEqual((budget, input_budget), (2912, 4184))

    def test_cap_wiring_keeps_input_spec_alignment_and_refund_paths(self):
        text = SCHEDULER.read_text()
        self.assertEqual(text.count("token_budget = effective_token_cap"), 1)
        self.assertEqual(text.count(
            "input_budget = self.scheduler_config.max_num_batched_tokens"), 1)
        self.assertEqual(text.count("input_budget -= num_new_tokens + draft_slots"), 2)
        self.assertEqual(text.count("input_budget += restored + draft_slots"), 1)
        self.assertEqual(text.count("token_budget += restored"), 1)
        self.assertEqual(text.count(
            "num_new_tokens = self._mamba_block_aligned_split("), 2)
        self.assertIn("assert total_num_scheduled_tokens <= effective_token_cap", text)
        self.assertIn(
            "total_num_scheduled_tokens > self.resilience_eager_threshold", text)


class Overlay(unittest.TestCase):
    @staticmethod
    def docker_argv(stdout: str) -> list[str]:
        values = [line[2:] for line in stdout.splitlines() if line.startswith("  ")]
        if values[:2] != ["sudo", "docker"]:
            raise AssertionError(values[:4])
        return values

    def test_four_rank_composed_parity_and_refusals(self):
        cap_delta = (CANDIDATE / "delta.env").read_text()
        trim_delta = TRIM_DELTA.read_text()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copyfile(REPO / "scripts/launcher/launch-glm53-tp4.sh",
                            root / "launch.sh")
            (root / "cluster.env").write_text(
                (REPO / "cluster.env.example").read_text()
                + "\n" + PROTECTED16.read_text()
                + '\nNODES="n0 n1 n2 n3"\n'
                + 'MGMT_IPS="192.0.2.21 192.0.2.22 192.0.2.23 192.0.2.24"\n'
                + 'MASTER_IP=192.0.2.21\nRELAY_DEST=operator@192.0.2.23\n')
            (root / "trim.env").write_text(trim_delta)
            (root / "candidate.env").write_text(trim_delta + "\n" + cap_delta)
            env = dict(os.environ, TP4_DRY_RUN="1")
            env.pop("TP4_ENV", None)

            def launch(name: str, rank: int = 0):
                return subprocess.run(
                    ["bash", str(root / "launch.sh"), str(rank)],
                    env={**env, "TP4_ENV": name}, capture_output=True,
                    text=True, timeout=20)

            home = str(Path.home())
            old = home + "/tp4/experiments/e03/end-drain/scheduler.py:" + TARGET + ":ro"
            new = home + "/tp4/experiments/e03/prefill-step-cap/scheduler.py:" + TARGET + ":ro"
            worker = home + "/tp4/experiments/e03/prefill-cache-trim/gpu_worker.py:"
            for rank in range(4):
                before_result, after_result = launch("trim.env", rank), launch("candidate.env", rank)
                self.assertEqual(before_result.returncode, 0, before_result.stderr)
                self.assertEqual(after_result.returncode, 0, after_result.stderr)
                before, after = self.docker_argv(before_result.stdout), self.docker_argv(after_result.stdout)
                self.assertEqual(Counter(before) - Counter(after), Counter([old]))
                self.assertEqual(Counter(after) - Counter(before), Counter(
                    [new, "-e", "VLLM_RESILIENCE_STEP_TOKEN_CAP=6912"]))
                self.assertTrue(any(value.startswith(worker) for value in after))
                self.assertIn("VLLM_PREFILL_CACHE_TRIM=1", after)

            bad = {
                "twice": trim_delta + "\n" + cap_delta + "\n" + cap_delta,
                "wrong-budget": "BATCHED_TOKENS=4096\n" + trim_delta + "\n" + cap_delta,
                "wrong-block": "BLOCK_SIZE=256\n" + trim_delta + "\n" + cap_delta,
                "prior-flag": trim_delta + '\nEXTRA_DOCKER_ENV+=" -e VLLM_RESILIENCE_STEP_TOKEN_CAP=0"\n' + cap_delta,
            }
            for label, value in bad.items():
                (root / "bad.env").write_text(value)
                result = launch("bad.env")
                self.assertNotEqual(result.returncode, 0, label)


if __name__ == "__main__":
    unittest.main()
