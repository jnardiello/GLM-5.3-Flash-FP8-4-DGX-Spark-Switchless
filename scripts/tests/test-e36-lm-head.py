#!/usr/bin/env python3
"""Offline contract for the E36 shared lm_head INT8 candidate."""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True

REPO = Path(__file__).resolve().parents[2]
CANDIDATE = REPO / "scripts/node/experiments/e03/e36-lm-head-w8a16"
RUNNER = CANDIDATE / "model_runner.py"
MODULE = CANDIDATE / "e36_lm_head_w8a16.py"
PARENT = REPO / "scripts/node/experiments/e03/e35-runner-k/model_runner.py"
PARENT_SHA = "4aa67bc0379d1e39649b7eb5297f247eaa0ad7597ed087e876dc81a4075e2e36"
MARKER = b"\n\n# --- E36 candidate addition."
VLLM = "/usr/local/lib/python3.12/dist-packages/vllm"
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()  # noqa: E731


def pure_functions():
    """The torch-free helpers of the module, executed without importing torch or vLLM."""
    tree = ast.parse(MODULE.read_text())
    wanted = {"strict_flag", "enabled", "parse_path", "check_head"}
    body = [node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in wanted
            or isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in {"ENABLE_ENV", "PATHS", "FLAG_MAX_BYTES"}
                for t in node.targets)]
    namespace = {"os": os, "GROUP_SIZE": 128}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(MODULE), "exec"), namespace)
    return namespace


class Provenance(unittest.TestCase):
    def test_prefix_patch_manifest_and_sums(self):
        self.assertEqual(sha(PARENT), PARENT_SHA)
        raw = RUNNER.read_bytes()
        self.assertEqual(raw[:raw.index(MARKER)], PARENT.read_bytes())
        manifest = json.loads((CANDIDATE / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "promoted_default")
        entries = {entry["path"]: entry for entry in manifest["candidates"]}
        runner = entries[str(RUNNER.relative_to(REPO))]
        self.assertEqual(runner["sha256"], sha(RUNNER))
        self.assertEqual(runner["parent"]["sha256"], PARENT_SHA)
        patch = CANDIDATE / "model_runner.patch"
        self.assertEqual(runner["patch_sha256"], sha(patch))
        removed = [line for line in patch.read_text().splitlines()
                   if line.startswith("-") and not line.startswith("---")]
        self.assertEqual(removed, [])
        with tempfile.TemporaryDirectory(prefix="e36-patch-") as temporary:
            rebuilt = Path(temporary) / "model_runner.py"
            rebuilt.write_bytes(PARENT.read_bytes())
            result = subprocess.run(["patch", "-s", str(rebuilt), str(patch)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(rebuilt.read_bytes(), RUNNER.read_bytes())
        self.assertEqual(entries[str(MODULE.relative_to(REPO))]["sha256"], sha(MODULE))
        sums = dict(line.split(None, 1)[::-1] for line in
                    (CANDIDATE / "SHA256SUMS").read_text().splitlines())
        self.assertEqual({name.strip() for name in sums},
                         {"e36_lm_head_w8a16.py", "manifest.json", "model_runner.py"})
        for name, digest in sums.items():
            self.assertEqual(sha(CANDIDATE / name.strip()), digest)

    def test_head_is_resolved_like_the_drafter_sharing(self):
        source = MODULE.read_text()
        self.assertIn("from vllm.v1.worker.gpu.spec_decode.eagle.utils import get_target_lm_head", source)
        self.assertIn("lm_head = get_target_lm_head(target_model, language_model)", source)
        self.assertIn("target_model.get_language_model()", source)

    def test_runner_addition_only_wraps_load_model(self):
        addition = RUNNER.read_bytes()[len(PARENT.read_bytes()):].decode()
        self.assertIn("GPUModelRunner.load_model = _e36_load_model", addition)
        self.assertNotIn("execute_model", addition)
        self.assertIn("convert_shared_lm_head(self.model", addition)

    def test_launcher_verifies_the_bundle(self):
        launcher = (REPO / "scripts/launcher/launch-glm53-tp4.sh").read_text()
        marker = "# The E36 lm_head candidate carries its own source manifest.\n"
        start = launcher.index("case ", launcher.index(marker))
        case = launcher[start:launcher.index("\nesac", start) + len("\nesac")]
        with tempfile.TemporaryDirectory(prefix="e36-launcher-") as temporary:
            deployed = Path(temporary) / "experiments/e03/e36-lm-head-w8a16"
            deployed.mkdir(parents=True)
            for name in ("model_runner.py", "e36_lm_head_w8a16.py", "manifest.json", "SHA256SUMS"):
                (deployed / name).write_bytes((CANDIDATE / name).read_bytes())
            env = dict(os.environ, ENV_DIR=temporary, DRY_RUN="0",
                       EXTRA_DOCKER_ENV="-v /x/e36-lm-head-w8a16/model_runner.py:/x:ro")
            good = subprocess.run(["bash", "-c", case], env=env, capture_output=True, text=True)
            self.assertEqual(good.returncode, 0, good.stderr)
            (deployed / "e36_lm_head_w8a16.py").write_bytes(b"corrupt\n")
            bad = subprocess.run(["bash", "-c", case], env=env, capture_output=True, text=True)
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn("E36 lm-head source manifest failed", bad.stderr)


E35_RETURN = REPO / "scripts/node/reference/operational-20260930-e35.env"


class Overlay(unittest.TestCase):
    """The measured overlay applies to E35, which the template reaches through its return."""
    def source(self, prelude=""):
        command = (f'source "$1"; source "$3" || exit 4; {prelude} before=$EXTRA_DOCKER_ENV; '
                   'source "$2" || exit 3; '
                   'printf "%s\\n--AFTER--\\n%s\\n" "$before" "$EXTRA_DOCKER_ENV"; '
                   'set | grep -c "^_E36_\\|^_E6R_" || true; declare -F | grep -c "_E36\\|_E6R" || true')
        return subprocess.run(["bash", "-c", command, "e36", str(REPO / "cluster.env.example"),
                               str(CANDIDATE / "delta.env"), str(E35_RETURN)], capture_output=True, text=True)

    def test_default_is_the_measured_overlay(self):
        command = ('source "$1"; default=$EXTRA_DOCKER_ENV; source "$3" || exit 4; '
                   'source "$2" || exit 3; [ "$default" = "$EXTRA_DOCKER_ENV" ]')
        result = subprocess.run(["bash", "-c", command, "e36", str(REPO / "cluster.env.example"),
                                 str(CANDIDATE / "delta.env"), str(E35_RETURN)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        on_default = subprocess.run(["bash", "-c", 'source "$1"; source "$2"', "e36",
                                     str(REPO / "cluster.env.example"), str(CANDIDATE / "delta.env")],
                                    capture_output=True, text=True)
        self.assertNotEqual(on_default.returncode, 0)

    def test_swaps_the_runner_and_adds_the_module(self):
        result = self.source()
        self.assertEqual(result.returncode, 0, result.stderr)
        before, rest = result.stdout.split("\n--AFTER--\n")
        lines = rest.strip().split("\n")
        after, leaks = lines[0], lines[1:]
        self.assertEqual(leaks, ["0", "0"])
        home = "$HOME/tp4/experiments/e03"
        expected = (before.replace(f"{home}/e35-runner-k/model_runner.py:",
                                   f"{home}/e36-lm-head-w8a16/model_runner.py:")
                    + f" -v {home}/e36-lm-head-w8a16/e36_lm_head_w8a16.py:"
                      f"{VLLM}/models/glm5next/nvidia/e36_lm_head_w8a16.py:ro"
                    + " -e VLLM_E36_LM_HEAD_W8A16=1 -e VLLM_E36_KEEP_BF16=0")
        self.assertEqual(after, expected)
        self.assertEqual(before.count("e35-runner-k/model_runner.py"), 1)
        # The E35 speculator, scheduler copy and policy stay selected.
        self.assertIn("e35-runner-k/speculator.py", after)
        self.assertIn("e35-runner-k/policy.flag", after)

    def test_refusals(self):
        self.assertNotEqual(self.source('source "$2";').returncode, 0)
        for prelude in (
                'EXTRA_DOCKER_ENV+=" --env-file /x";',
                'EXTRA_DOCKER_ENV+=" --volume=/a:/b";',
                'EXTRA_DOCKER_ENV+=" -e VLLM_E36_FLAG=/x";',
                f'EXTRA_DOCKER_ENV+=" -v /x:{VLLM}/v1/worker/gpu/model_runner.py:ro";',
                'EXTRA_DOCKER_ENV=${EXTRA_DOCKER_ENV/e35-runner-k\\/model_runner/other\\/model_runner};',
                'EXTRA_DOCKER_ENV=${EXTRA_DOCKER_ENV/e21_bf16_residue.py:ro/e21_other.py:ro};',
                'EXTRA_DOCKER_ENV="$EXTRA_DOCKER_ENV"$\'\\n\';'):
            self.assertNotEqual(self.source(prelude).returncode, 0, prelude)


class PureHelpers(unittest.TestCase):
    def setUp(self):
        self.ns = pure_functions()

    def test_strict_flags(self):
        enabled = self.ns["enabled"]
        self.assertFalse(enabled({}))
        self.assertFalse(enabled({"VLLM_E36_LM_HEAD_W8A16": "0"}))
        self.assertFalse(enabled({"VLLM_E36_LM_HEAD_W8A16": " "}))
        self.assertTrue(enabled({"VLLM_E36_LM_HEAD_W8A16": "1"}))
        with self.assertRaises(RuntimeError):
            enabled({"VLLM_E36_LM_HEAD_W8A16": "yes"})

    def test_parse_path(self):
        parse = self.ns["parse_path"]
        for raw, want in ((b"bf16", "bf16"), (b"bf16\n", "bf16"), (b"int8", "int8"), (None, "int8"),
                          (b"BF16", "int8"), (b"x" * 65, "int8"), (b"\xff", "int8"), (b"", "int8")):
            self.assertEqual(parse(raw), want, raw)

    def test_check_head(self):
        check = self.ns["check_head"]
        good = dict(shape=(38720, 4096), dtype="torch.bfloat16", is_cuda=True, contiguous=True,
                    has_bias=False, method="UnquantizedEmbeddingMethod")
        self.assertEqual(check(**good), (38720, 4096))
        for change in ({"shape": (38720,)}, {"shape": (38720, 4000)}, {"shape": (38721, 4096)},
                       {"dtype": "torch.float16"}, {"is_cuda": False}, {"contiguous": False},
                       {"has_bias": True}, {"method": "ResidueW8A16Method"}):
            with self.assertRaises(RuntimeError, msg=str(change)):
                check(**{**good, **change})


if __name__ == "__main__":
    unittest.main()
