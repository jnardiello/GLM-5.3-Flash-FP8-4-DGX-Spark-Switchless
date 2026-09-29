#!/usr/bin/env python3
"""Offline contract for the E31 indexer candidate; no Torch, GPU, node or network.

E31 is the E29 default with the production pooled_indexer.py and ops/glm_kpool.py swapped
for copies with two runtime switches: the head gate as a BF16 tensor-core GEMM and the
speculative-safe C4 tail ring (its index arithmetic is proven in test-e31-kpool-tail-ring.py).
Covers pins and provenance, that each copy is exactly production plus its patch with the
expected replaced lines, the switch and flag-file semantics, the head-gate path choice with
its FP32 fallback, the tail-ring selection, four-rank launcher parity with the E29
default, overlay refusals and the leaf tests' --help.
"""

from __future__ import annotations

import ast
from collections import Counter
import copy
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest


REPO = Path(__file__).resolve().parents[2]
CANDIDATE = REPO / "scripts/node/experiments/e03/e31-indexer"
PROTECTED16 = REPO / "scripts/node/reference/operational-20260929-sparkcache-protected.env"
OVERRIDES = REPO / "scripts/node/overrides/vllm/models/glm5next/nvidia"
PRODUCTION = OVERRIDES / "pooled_indexer.py"
KPOOL = OVERRIDES / "ops/glm_kpool.py"
E29 = REPO / "docs/historical_benchmarks/baselines/2026-09-25-e29/baseline.json"
MANIFEST = json.loads((CANDIDATE / "manifest.json").read_text())
DEST = "/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/pooled_indexer.py"
FROM = f"/tp4/overrides/vllm/models/glm5next/nvidia/pooled_indexer.py:{DEST}:ro"
TO = f"/tp4/experiments/e03/e31-indexer/pooled_indexer.py:{DEST}:ro"
FLAG = "VLLM_GLM53_INDEXER_GATE_TC_FLAG=/tmp/glm53-indexer-gate-tc"
KDEST = "/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/ops/glm_kpool.py"
KFROM = f"/tp4/overrides/vllm/models/glm5next/nvidia/ops/glm_kpool.py:{KDEST}:ro"
KTO = f"/tp4/experiments/e03/e31-indexer/glm_kpool.py:{KDEST}:ro"
RING_FLAG = "VLLM_GLM53_KPOOL_TAIL_RING_FLAG=/tmp/glm53-kpool-tail-ring"
# Lines each patch replaces; everything else is added.
REPLACED = {
    "pooled_indexer.py": ["                (self.max_seqs, 2, _POOL_SIZE, _INDEX_HEAD_DIM),"],
    "glm_kpool.py": [
        "                    + slot * tail_stride_2",
        "                base = state_slot * tail_stride_0 + slot * tail_stride_2 + dim",
        "        base = state_slot * tail_stride_0 + physical_slot * tail_stride_2",
        "            state_slot * tail_stride_0 + tail_stride_1 + slot * tail_stride_2 + dim",
        "        tail_offset = state_slot * tail_stride_0 + slot * tail_stride_2 + dim",
        "        tail_offset = state_slot * tail_stride_0 + slot * tail_stride_2 + dim",
    ],
}
BASES = {"pooled_indexer.py": PRODUCTION, "glm_kpool.py": KPOOL}
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731


def apply_patch(text: str, patch: str, reverse: bool = False) -> str:
    """Apply a unified diff at its recorded line numbers, checking every context line."""
    lines, out, cursor = text.splitlines(keepends=True), [], 0
    old_sign, new_sign = ("+", "-") if reverse else ("-", "+")
    body = patch.splitlines(keepends=True)
    for index, line in enumerate(body):
        if not line.startswith("@@"):
            continue
        old_start = int(line.split()[2 if reverse else 1][1:].split(",")[0])
        out += lines[cursor:old_start - 1]
        cursor = old_start - 1
        for row in body[index + 1:]:
            if row.startswith("@@"):
                break
            if row[0] in (" ", old_sign):
                assert lines[cursor] == row[1:], (cursor, row)
                cursor += 1
            if row[0] in (" ", new_sign):
                out.append(row[1:])
    return "".join(out + lines[cursor:])


def e31_namespace(env: dict[str, str], clock: list[float], opened: list[str]) -> dict:
    """Execute the candidate's E31 module statements and head-gate method in isolation."""
    tree = ast.parse((CANDIDATE / "pooled_indexer.py").read_text())
    body = []
    for node in tree.body:
        if ((isinstance(node, ast.FunctionDef) and node.name in ("_e31_switch", "_e31_on"))
                or (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id in ("_E31_GATE_TC", "_E31_TAIL_RING", "_e31"))):
            body.append(copy.deepcopy(node))
        if isinstance(node, ast.ClassDef):
            body += [copy.deepcopy(n) for n in node.body
                     if isinstance(n, ast.FunctionDef) and n.name == "_project_head_weights"]
    logs: list[str] = []

    def fake_open(path, *args, **kwargs):
        opened.append(path)
        return open(path, *args, **kwargs)

    namespace = {
        "os": SimpleNamespace(environ=env), "open": fake_open, "logs": logs,
        "time": SimpleNamespace(monotonic=lambda: clock[0]),
        "logger": SimpleNamespace(info=lambda f, *a: logs.append(f % a),
                                  warning=lambda f, *a: logs.append(f % a)),
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), "pooled_indexer.py", "exec"), namespace)
    return namespace


class Provenance(unittest.TestCase):
    def test_pins(self):
        sums = dict(reversed(line.split("  ", 1)) for line in
                    (CANDIDATE / "SHA256SUMS").read_text().splitlines())
        self.assertEqual(set(sums), {"glm_kpool.py", "leaf_gate_tc.py", "leaf_kpool_tail_ring.py",
                                     "manifest.json", "pooled_indexer.py"})
        for name, digest in sums.items():
            self.assertEqual(sha(CANDIDATE / name), digest, name)
        record = json.loads(E29.read_text())
        self.assertEqual(set(MANIFEST["files"]), set(BASES))
        for name, base in BASES.items():
            entry = MANIFEST["files"][name]
            self.assertEqual(entry["candidate_sha256"], sha(CANDIDATE / name), name)
            self.assertEqual(entry["patch_sha256"], sha(CANDIDATE / entry["patch"]), name)
            self.assertEqual(entry["base_path"], str(base.relative_to(REPO)), name)
            frozen = record["system"]["source_files_sha256"][entry["base_path"]]
            self.assertEqual(sha(base), frozen, name)   # production stays the E29 file
            self.assertEqual(entry["base_sha256"], frozen, name)
        self.assertEqual(MANIFEST["files"]["pooled_indexer.py"]["mount_target"], DEST)
        self.assertEqual(MANIFEST["files"]["glm_kpool.py"]["mount_target"], KDEST)
        self.assertEqual(MANIFEST["parent"]["baseline_sha256"], sha(E29))
        self.assertEqual(MANIFEST["status"], "promoted")

    def test_candidates_are_production_plus_patch(self):
        for name, base in BASES.items():
            patch = (CANDIDATE / MANIFEST["files"][name]["patch"]).read_text()
            removed = [line[1:] for line in patch.splitlines()
                       if line.startswith("-") and not line.startswith("---")]
            self.assertEqual(removed, REPLACED[name], name)
            candidate = (CANDIDATE / name).read_text()
            self.assertEqual(apply_patch(base.read_text(), patch), candidate, name)
            self.assertEqual(apply_patch(candidate, patch, reverse=True), base.read_text(), name)

    def test_fp32_path_is_kept(self):
        text = (CANDIDATE / "pooled_indexer.py").read_text()
        self.assertIn('_e31_switch("VLLM_GLM53_INDEXER_GATE_TC", "0")', text)
        self.assertIn('_e31_switch("VLLM_GLM53_KPOOL_TAIL_RING", "1")', text)
        self.assertIn("            self._weights_proj_fp32 = self.weights_proj.weight.detach().float()\n"
                      "        return F.linear(hidden_states.float(), self._weights_proj_fp32)\n", text)
        self.assertIn("@eager_break_during_capture\n    def forward(", text)


class Switch(unittest.TestCase):
    def test_default_and_boot_value(self):
        clock, opened = [100.0], []
        ns = e31_namespace({}, clock, opened)
        self.assertFalse(ns["_e31_on"](ns["_E31_GATE_TC"]))
        self.assertTrue(ns["_e31_on"](ns["_E31_TAIL_RING"]))
        ns = e31_namespace({"VLLM_GLM53_INDEXER_GATE_TC": "1",
                            "VLLM_GLM53_KPOOL_TAIL_RING": "0"}, clock, opened)
        self.assertTrue(ns["_e31_on"](ns["_E31_GATE_TC"]))
        self.assertFalse(ns["_e31_on"](ns["_E31_TAIL_RING"]))
        self.assertEqual(opened, [])   # no flag variable, no filesystem access
        for name in ("VLLM_GLM53_INDEXER_GATE_TC", "VLLM_GLM53_KPOOL_TAIL_RING"):
            for bad in ("true", "2", ""):
                with self.assertRaises(ValueError):
                    e31_namespace({name: bad}, clock, opened)

    def test_flag_file(self):
        with tempfile.TemporaryDirectory(prefix="e31-flag-") as temp:
            flag = Path(temp) / "flag"
            clock, opened = [100.0], []
            ns = e31_namespace({"VLLM_GLM53_INDEXER_GATE_TC_FLAG": str(flag)}, clock, opened)
            on = lambda: ns["_e31_on"](ns["_E31_GATE_TC"])  # noqa: E731
            self.assertFalse(on())                  # missing file: env default 0
            flag.write_text("1\n")
            self.assertFalse(on())                  # cached for 0.5 s
            clock[0] += 0.5
            self.assertTrue(on())
            flag.write_text("0")
            clock[0] += 0.25
            self.assertTrue(on())
            clock[0] += 0.25
            self.assertFalse(on())
            flag.write_text("yes")
            clock[0] += 1
            self.assertFalse(on())                  # other content: env default
            self.assertEqual(len(opened), 4)
            ns = e31_namespace({"VLLM_GLM53_KPOOL_TAIL_RING_FLAG": str(flag)}, clock, opened)
            ring = lambda: ns["_e31_on"](ns["_E31_TAIL_RING"])  # noqa: E731
            self.assertTrue(ring())                 # other content: ring default 1
            self.assertFalse(ns["_e31_on"](ns["_E31_GATE_TC"]))
            flag.write_text("0")
            clock[0] += 0.5
            self.assertFalse(ring())
            flag.unlink()
            clock[0] += 1
            self.assertTrue(ring())                 # missing file: ring default 1
            self.assertEqual(opened, [str(flag)] * 7)   # gate switch without a flag: none

    def test_ring_selection_feeds_every_update(self):
        text = (CANDIDATE / "pooled_indexer.py").read_text()
        self.assertEqual(text.count("update_decode_pools("), 1)
        call = text[text.index("        update_decode_pools("):]
        self.assertIn("            tail_ring=tail_ring,\n        )\n", call[:call.index("        )\n") + 10])
        self.assertIn('logger.info("E31_KPOOL_TAIL_RING ring=%d", tail_ring)', text)

    def test_path_choice_and_fallback(self):
        with tempfile.TemporaryDirectory(prefix="e31-flag-") as temp:
            flag = Path(temp) / "flag"
            clock = [100.0]
            ns = e31_namespace({"VLLM_GLM53_INDEXER_GATE_TC_FLAG": str(flag)}, clock, [])
            calls: list[str] = []
            state = {"raise": None}

            def mm(a, b, out_dtype=None):
                calls.append(f"mm:{a.dtype}:{out_dtype}")
                if state["raise"]:
                    raise state["raise"]
                return "tc"

            tensor = lambda dtype: SimpleNamespace(  # noqa: E731
                dtype=dtype, t=lambda: "Wt", float=lambda: SimpleNamespace(dtype="float32"),
                detach=lambda: SimpleNamespace(float=lambda: "W32"))
            ns["torch"] = SimpleNamespace(bfloat16="bfloat16", float32="float32", mm=mm)
            ns["F"] = SimpleNamespace(linear=lambda h, w: calls.append(f"linear:{h.dtype}:{w}") or "fp32")
            module = SimpleNamespace(_weights_proj_fp32=None,
                                     weights_proj=SimpleNamespace(weight=tensor("bfloat16")))
            project = lambda h: ns["_project_head_weights"](module, h)  # noqa: E731
            hidden = tensor("bfloat16")
            self.assertEqual(project(hidden), "fp32")
            flag.write_text("1")
            clock[0] += 1
            self.assertEqual(project(hidden), "tc")
            self.assertEqual(project(tensor("float16")), "fp32")   # non-BF16 input
            state["raise"] = TypeError("unexpected keyword argument 'out_dtype'")
            self.assertEqual(project(hidden), "fp32")               # probe failure falls back
            state["raise"] = None
            self.assertEqual(project(hidden), "fp32")               # and stays on FP32
            self.assertEqual(calls, ["linear:float32:W32", "mm:bfloat16:float32", "linear:float32:W32",
                                     "mm:bfloat16:float32", "linear:float32:W32", "linear:float32:W32"])
            self.assertEqual(ns["logs"], [
                "E31_INDEXER_GATE path=fp32", "E31_INDEXER_GATE path=bf16-tc",
                "E31_INDEXER_GATE path=fp32", "E31_INDEXER_GATE path=bf16-tc",
                "E31_INDEXER_GATE bf16-tc unavailable: unexpected keyword argument 'out_dtype'",
                "E31_INDEXER_GATE path=fp32"])


class Launcher(unittest.TestCase):
    def test_four_rank_parity_and_refusals(self):
        delta = (CANDIDATE / "delta.env").read_text()
        with tempfile.TemporaryDirectory(prefix="tp4-e31-") as temp:
            root = Path(temp)
            shutil.copyfile(REPO / "scripts/launcher/launch-glm53-tp4.sh", root / "launch.sh")
            (root / "cluster.env").write_text((REPO / "cluster.env.example").read_text() + '''
NODES="n0 n1 n2 n3"
MGMT_IPS="192.0.2.21 192.0.2.22 192.0.2.23 192.0.2.24"
MASTER_IP=192.0.2.21
RELAY_DEST=operator@192.0.2.23
''')
            # The measured E31 overlay applies on the complete E29 recipe. Compare it with
            # the protected 16 GiB operational predecessor, which layers only SparkCache
            # protection over E31.
            e29 = (REPO / "scripts/node/reference/baseline-20260925-e29.env").read_text() + "\n"
            e31 = (REPO / "scripts/node/reference/baseline-20260928-e31.env").read_text() + "\n"
            (root / "default.env").write_text(e29)
            (root / "candidate.env").write_text(e29 + delta)
            (root / "promoted.env").write_text(e31)
            (root / "protected.env").write_text(PROTECTED16.read_text())
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
                before, after = argv(launch("default.env", rank)), argv(launch("candidate.env", rank))
                # Promotion parity: the default dry-run output is byte-identical to the overlay's.
                self.assertEqual(launch("promoted.env", rank).stdout, launch("candidate.env", rank).stdout)
                self.assertEqual(Counter(before) - Counter(after),
                                 Counter([home + FROM, home + KFROM]))
                self.assertEqual(Counter(after) - Counter(before),
                                 Counter([home + TO, home + KTO, "-e", FLAG, "-e", RING_FLAG]))
                protected = argv(launch("protected.env", rank))
                e31_cfg = (f"<canonical JSON of {home}/tp4/experiments/e03/drafter-w8a16/"
                           "kv-transfer-config-e22b.json>")
                protected_cfg = (f"<canonical JSON of {home}/tp4/experiments/e03/"
                                 "sparkcache-ram-budget/kv-transfer-config.json>")
                connector_target = ("/usr/local/lib/python3.12/dist-packages/sparkcache/"
                                    "spark_context_cache_connector.py:ro")
                e31_connector = (f"{home}/tp4/sparkcache/"
                                 f"spark_context_cache_connector-e03-replay-views.py:{connector_target}")
                protected_connector = (f"{home}/tp4/sparkcache/"
                                       f"spark_context_cache_connector-ram-budget.py:{connector_target}")
                self.assertEqual(Counter(after) - Counter(protected),
                                 Counter([e31_cfg, e31_connector]))
                self.assertEqual(Counter(protected) - Counter(after),
                                 Counter([protected_cfg, protected_connector]))

            docker_env = re.search(r"^EXTRA_DOCKER_ENV='([^']*)'$", e29, re.M).group(1)
            self.assertEqual(docker_env.count("$HOME" + FROM), 1)
            self.assertEqual(docker_env.count("$HOME" + KFROM), 1)
            e28b = (REPO / "scripts/node/reference/baseline-20260925-e28b.env").read_text() + "\n"
            site = "/usr/local/lib/python3.12/dist-packages/vllm/"
            bad = {
                "on-e28b-rollback": e28b + delta,
                "on-e31-default": delta,
                "applied-twice": delta + "\n" + delta,
                "flag-present": 'EXTRA_DOCKER_ENV+=" -e VLLM_GLM53_INDEXER_GATE_TC=1"\n' + delta,
                "flag-file-present": f'EXTRA_DOCKER_ENV+=" -e {FLAG}"\n' + delta,
                "second-indexer-mount": f'EXTRA_DOCKER_ENV+=" -v /x/pooled_indexer.py:{DEST}:ro"\n' + delta,
                "no-indexer-mount": f"EXTRA_DOCKER_ENV='{docker_env.replace(' -v $HOME' + FROM, '')}'\n" + delta,
                "other-indexer-mount": f"EXTRA_DOCKER_ENV='{docker_env.replace('$HOME' + FROM, '/x' + FROM)}'\n" + delta,
                "ring-present": 'EXTRA_DOCKER_ENV+=" -e VLLM_GLM53_KPOOL_TAIL_RING=0"\n' + delta,
                "ring-flag-present": f'EXTRA_DOCKER_ENV+=" -e {RING_FLAG}"\n' + delta,
                "second-kpool-mount": f'EXTRA_DOCKER_ENV+=" -v /x/glm_kpool.py:{KDEST}:ro"\n' + delta,
                "no-kpool-mount": f"EXTRA_DOCKER_ENV='{docker_env.replace(' -v $HOME' + KFROM, '')}'\n" + delta,
                "other-kpool-mount": f"EXTRA_DOCKER_ENV='{docker_env.replace('$HOME' + KFROM, '/x' + KFROM)}'\n" + delta,
                "second-e29-scheduler": f'EXTRA_DOCKER_ENV+=" -v \\$HOME/tp4/experiments/e03/end-drain/scheduler.py:{site}v1/core/sched/scheduler.py:ro"\n' + delta,
                "env-long-form": 'EXTRA_DOCKER_ENV+=" --env=X=1"\n' + delta,
                "env-attached": 'EXTRA_DOCKER_ENV+=" -eX=1"\n' + delta,
                "multi-line": "EXTRA_DOCKER_ENV=\"$EXTRA_DOCKER_ENV\"$'\\n-e X=1'\n" + delta,
            }
            bad = {name: text if name in ("on-e28b-rollback", "on-e31-default") else e29 + text
                   for name, text in bad.items()}
            for name, text in bad.items():
                (root / "bad.env").write_text(text)
                result = launch("bad.env")
                self.assertNotEqual(result.returncode, 0, name)
                self.assertIn("E31 re", result.stderr, name)   # refused by the overlay itself

    def test_launcher_verifies_the_manifest(self):
        text = (REPO / "scripts/launcher/launch-glm53-tp4.sh").read_text()
        self.assertIn('(cd "$ENV_DIR/experiments/e03/e31-indexer" && sha256sum -c SHA256SUMS)', text)


class Documentation(unittest.TestCase):
    def test_leaf_help_needs_no_torch(self):
        for leaf, options in (
                ("leaf_gate_tc.py", ("--weights", "--rows", "--pools", "--topk", "--iters",
                                     "--ab-blocks")),
                ("leaf_kpool_tail_ring.py", ("--spec-tokens", "--prompt", "--steps", "--tol"))):
            result = subprocess.run([sys.executable, str(CANDIDATE / leaf), "--help"],
                                    capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            for option in options:
                self.assertIn(option, result.stdout, leaf)

    def test_readme_names_the_overlay(self):
        text = (CANDIDATE / "README.md").read_text()
        self.assertIn("scripts/node/experiments/e03/e31-indexer/delta.env", text)
        self.assertIn("scripts/node/reference/baseline-20260925-e29.env", text)
        for word in ("VLLM_GLM53_INDEXER_GATE_TC", "/tmp/glm53-indexer-gate-tc", "all four ranks",
                     "VLLM_GLM53_KPOOL_TAIL_RING", "/tmp/glm53-kpool-tail-ring",
                     "vllm:num_requests_running", "vllm:num_requests_waiting",
                     "Never switch during a suite", "SparkCache namespace"):
            self.assertIn(word, text)


if __name__ == "__main__":
    unittest.main(verbosity=1)
