#!/usr/bin/env python3
"""Offline contract for the E21 BF16-residue candidate; no Torch, GPU, node or network.

Covers pins, the flag-gated hook, family selection and validation, the prefill dispatch
threshold, four-rank launcher parity and overlay refusals. Native Marlin/Triton numerics,
CUDA graph capture and throughput need the qualified engine and an authorized window.
"""

from __future__ import annotations

import ast
from collections import Counter
import copy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
from typing import Callable
import unittest


REPO = Path(__file__).resolve().parents[2]
E03 = REPO / "scripts/node/experiments/e03"
CANDIDATE = E03 / "bf16-residue"
ACCEPTED_HOOK = REPO / "scripts/node/overrides/vllm/models/glm5next/nvidia/e20_kda_w8a16.py"
CANDIDATE_HOOK = CANDIDATE / "e20_kda_w8a16.py"
RESIDUE = CANDIDATE / "e21_bf16_residue.py"
MODEL = E03 / "overrides/vllm/models/glm5next/nvidia/model.py"
MANIFEST = json.loads((CANDIDATE / "manifest.json").read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def load(path: Path, names: set[str], namespace: dict, keep_decorators: set[str] = frozenset()):
    """Execute selected real top-level definitions with stubbed dependencies."""
    tree = ast.parse(path.read_text(), str(path))
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            node = copy.deepcopy(node)
            if node.name not in keep_decorators:
                node.decorator_list = []
            selected.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id in names for t in targets):
                selected.append(copy.deepcopy(node))
    found = {getattr(n, "name", None) for n in selected} | {
        t.id for n in selected if isinstance(n, (ast.Assign, ast.AnnAssign))
        for t in (n.targets if isinstance(n, ast.Assign) else [n.target])
        if isinstance(t, ast.Name)
    }
    assert names <= found, (path, names - found)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", level=0,
        names=[ast.alias(name="annotations")]), *selected], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


class Tensor:
    def __init__(self, shape, dtype="bf16", device="cuda"):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = types.SimpleNamespace(type=device)

    def numel(self):
        result = 1
        for size in self.shape:
            result *= size
        return result


class Unquantized:
    pass


class Quantized:
    pass


def linear(n, k, **overrides):
    values = dict(weight=Tensor((n, k)), quant_method=Unquantized(), bias=None)
    values.update(overrides)
    return types.SimpleNamespace(**values)


CONFIG = MANIFEST["source_identity"]["config_values"]
TP = 4


def kda_attn():
    return types.SimpleNamespace(
        hidden_size=CONFIG["hidden_size"], local_projection_size=2048,
        o_proj=linear(CONFIG["hidden_size"], 2048),
    )


def mla_attn():
    heads = CONFIG["num_attention_heads"] // TP
    qk = CONFIG["qk_nope_head_dim"] + CONFIG["qk_rope_head_dim"]
    return types.SimpleNamespace(
        hidden_size=CONFIG["hidden_size"], num_local_heads=heads, qk_head_dim=qk,
        v_head_dim=CONFIG["v_head_dim"], q_lora_rank=CONFIG["q_lora_rank"],
        kv_lora_rank=CONFIG["kv_lora_rank"], qk_rope_head_dim=CONFIG["qk_rope_head_dim"],
        o_proj=linear(CONFIG["hidden_size"], heads * CONFIG["v_head_dim"]),
        q_b_proj=linear(heads * qk, CONFIG["q_lora_rank"]),
        fused_qkv_a_proj=linear(
            CONFIG["q_lora_rank"] + CONFIG["kv_lora_rank"] + CONFIG["qk_rope_head_dim"],
            CONFIG["hidden_size"],
        ),
    )


def fake_model():
    # The checkpoint's layer_types place deepseek_sparse_attention every fourth layer
    # from index 3; every other layer is linear attention (KDA).
    layers = []
    for index in range(CONFIG["num_hidden_layers"]):
        is_mla = index % 4 == 3
        layers.append(types.SimpleNamespace(
            layer_kind="mla" if is_mla else "kda",
            self_attn=mla_attn() if is_mla else kda_attn(),
        ))
    return types.SimpleNamespace(layers=layers)


def residue_namespace():
    calls = {"marlin": 0, "scratch": 0}

    def marlin(**kwargs):
        calls["marlin"] += 1
        return "marlin"

    # dataclass resolves string annotations through sys.modules[cls.__module__].
    module = types.ModuleType("e21_test_residue")
    sys.modules[module.__name__] = module
    namespace = module.__dict__
    namespace.update({
        "dataclass": dataclass, "Callable": Callable,
        "os": os, "json": json,
        "nn": types.SimpleNamespace(Module=object),
        "torch": types.SimpleNamespace(bfloat16="bf16", cuda=types.SimpleNamespace(empty_cache=lambda: None)),
        "UnquantizedLinearMethod": Unquantized, "LinearMethodBase": object,
        "apply_gptq_marlin_linear": marlin,
        "logger": types.SimpleNamespace(info=lambda *a, **k: None),
    })
    names = {
        "ENABLE_ENV", "GROUP_SIZE", "WTYPE", "PREFILL_BF16_MIN_TOKENS", "MARLIN_TILE",
        "SCRATCH_BUDGET_BYTES", "EXPECTED_LAYERS",
        "_kda_o_proj_shape", "_mla_o_proj_shape", "_mla_q_b_proj_shape", "_mla_q_proj_shape",
        "_mla_kv_a_proj_with_mqa_shape", "_mla_fused_qkv_a_proj_shape",
        "FamilyPlan", "FAMILIES", "enabled", "_layers_by_kind", "_mla_layout",
        "_expected_shape", "_validate", "plan_conversion", "ResidueW8A16Method",
        "finalize_bf16_residue_w8a16",
    }
    namespace["scalar_types"] = types.SimpleNamespace(uint8b128="uint8b128")
    load(RESIDUE, names, namespace, keep_decorators={"FamilyPlan"})
    return namespace, calls


class Pins(unittest.TestCase):
    def test_manifest_and_sums(self):
        for name, entry in MANIFEST["files"].items():
            self.assertEqual(sha(CANDIDATE / name), entry["candidate_sha256"], name)
            if entry["base"]:
                self.assertEqual(sha(REPO / entry["base"]), entry["base_sha256"], name)
        for path, digest in MANIFEST["source_identity"]["reviewed_runtime_sources_sha256"].items():
            self.assertEqual(sha(REPO / path), digest, path)
        self.assertEqual(
            sha(REPO / "docs/historical_benchmarks/baselines/2026-09-19-e03/baseline.json"),
            MANIFEST["parent"]["baseline_sha256"],
        )
        rows = (CANDIDATE / "SHA256SUMS").read_text().splitlines()
        self.assertTrue(rows)
        for row in rows:
            digest, name = row.split("  ", 1)
            self.assertEqual(sha(CANDIDATE / name), digest, name)

    def test_cache_config_differs_only_in_root(self):
        accepted = json.loads((E03 / "kv-transfer-config.json").read_text())
        candidate = json.loads((CANDIDATE / "kv-transfer-config.json").read_text())
        a_extra = accepted.pop("kv_connector_extra_config")
        c_extra = candidate.pop("kv_connector_extra_config")
        self.assertEqual(accepted, candidate)
        self.assertNotEqual(a_extra.pop("spark_cache_root"), c_extra.pop("spark_cache_root"))
        self.assertEqual(a_extra, c_extra)


class Hook(unittest.TestCase):
    ADDED = {"e21_residue_enabled", "E21_ENABLE_ENV"}
    HOOKED = "finalize_kda_input_w8a16"

    @staticmethod
    def named(tree):
        out = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                out[node.name] = node
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        out[target.id] = node
        return out

    def test_candidate_hook_is_accepted_source_plus_one_gated_call(self):
        accepted = self.named(ast.parse(ACCEPTED_HOOK.read_text()))
        candidate = self.named(ast.parse(CANDIDATE_HOOK.read_text()))
        self.assertEqual(set(candidate) - set(accepted), self.ADDED)
        self.assertEqual(set(accepted) - set(candidate), set())
        for name, node in accepted.items():
            if name == self.HOOKED:
                continue
            self.assertEqual(ast.dump(node), ast.dump(candidate[name]), name)
        a_body = accepted[self.HOOKED].body
        c_body = candidate[self.HOOKED].body
        self.assertEqual(len(c_body), len(a_body) + 1)
        for a_node, c_node in zip(a_body, c_body):
            self.assertEqual(ast.dump(a_node), ast.dump(c_node))
        gate = c_body[-1]
        self.assertIsInstance(gate, ast.If)
        self.assertEqual(ast.unparse(gate.test), "e21_residue_enabled()")
        self.assertIn("finalize_bf16_residue_w8a16", ast.unparse(gate))

    def test_hook_flag_matches_residue_flag(self):
        ns = load(CANDIDATE_HOOK, {"E21_ENABLE_ENV", "e21_residue_enabled"}, {"os": os})
        residue, _ = residue_namespace()
        self.assertEqual(ns["E21_ENABLE_ENV"], residue["ENABLE_ENV"])
        self.assertEqual(ns["E21_ENABLE_ENV"], MANIFEST["activation"]["flag"])
        for value, expected in (("", False), ("0", False), (" 0 ", False), ("1", True), ("2", True)):
            env = {ns["E21_ENABLE_ENV"]: value}
            self.assertIs(ns["e21_residue_enabled"](env), expected, value)
            self.assertIs(residue["enabled"](env), expected, value)
        self.assertFalse(ns["e21_residue_enabled"]({}))
        with self.assertRaises(RuntimeError):
            ns["e21_residue_enabled"]({ns["E21_ENABLE_ENV"]: "yes"})

    def test_conversion_runs_after_fp8_dequant_load(self):
        # load_weights dequantizes the legacy block-FP8 MLA projections; the E20 entry,
        # and therefore the E21 pass, is only reached from process_weights_after_loading.
        tree = ast.parse(MODEL.read_text())
        load_calls, finalize_calls = [], []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "load_weights":
                load_calls += [n for n in ast.walk(node) if isinstance(n, ast.Call)
                               and getattr(n.func, "id", "") == "_try_load_fp8_attn_proj"]
            if isinstance(node, ast.FunctionDef) and node.name == "process_weights_after_loading":
                finalize_calls += [n for n in ast.walk(node) if isinstance(n, ast.Call)
                                   and getattr(n.func, "id", "") == "finalize_kda_input_w8a16"]
        self.assertTrue(load_calls)
        self.assertTrue(finalize_calls)


class Selection(unittest.TestCase):
    def setUp(self):
        self.ns, self.calls = residue_namespace()

    def test_bundle_resolves_to_manifest(self):
        work, layout, shapes = self.ns["plan_conversion"](fake_model())
        self.assertEqual(layout, MANIFEST["activation"]["expected_signature_fields"]["mla_layout"])
        self.assertEqual(len(work), MANIFEST["activation"]["expected_signature_fields"]["modules"])
        by_family = Counter(plan.name for plan, *_ in work)
        expected = {f["name"]: f["layers"] for f in MANIFEST["families"]}
        self.assertEqual(dict(by_family), expected)
        for plan, _, _, shape in work:
            declared = next(f for f in MANIFEST["families"] if f["name"] == plan.name)
            self.assertEqual(list(shape), declared["shape_nk_per_rank"], plan.name)
        scratch = sum(n * k * 2 for n, k in shapes)
        self.assertEqual(scratch, MANIFEST["activation"]["expected_signature_fields"]["added_scratch_bytes"])
        self.assertLessEqual(scratch, self.ns["SCRATCH_BUDGET_BYTES"])

    def test_no_q_lora_families_never_fire(self):
        work, layout, _ = self.ns["plan_conversion"](fake_model())
        self.assertEqual(layout, "q_lora")
        names = {plan.name for plan, *_ in work}
        self.assertNotIn("mla_q_proj", names)
        self.assertNotIn("mla_kv_a_proj_with_mqa", names)

    def test_excluded_families_are_not_planned(self):
        planned = {(p.layer_kind, p.attribute) for p in self.ns["FAMILIES"]}
        for attribute in ("kv_b_proj", "f_a_proj", "f_b_proj", "g_a_proj", "g_b_proj",
                          "in_proj_qkvgfab", "lm_head", "wq_b", "wk", "weights_proj"):
            self.assertFalse(any(attr == attribute for _, attr in planned), attribute)

    def refuse(self, mutate):
        model = fake_model()
        mutate(model)
        with self.assertRaises(RuntimeError):
            self.ns["plan_conversion"](model)

    def test_refusals(self):
        mla = lambda m: next(l.self_attn for l in m.layers if l.layer_kind == "mla")
        kda = lambda m: next(l.self_attn for l in m.layers if l.layer_kind == "kda")
        self.refuse(lambda m: setattr(kda(m).o_proj, "weight", Tensor((4096, 2049))))
        self.refuse(lambda m: setattr(mla(m).q_b_proj, "weight", Tensor((4096, 1536), dtype="fp16")))
        self.refuse(lambda m: setattr(mla(m).o_proj, "weight", Tensor((4096, 4096), device="cpu")))
        self.refuse(lambda m: setattr(mla(m).fused_qkv_a_proj, "bias", Tensor((2048,))))
        self.refuse(lambda m: setattr(kda(m).o_proj, "quant_method", Quantized()))
        self.refuse(lambda m: delattr(mla(m), "q_b_proj"))
        self.refuse(lambda m: delattr(mla(m), "q_lora_rank"))
        self.refuse(lambda m: m.layers.pop(0))
        self.refuse(lambda m: setattr(mla(m), "q_lora_rank", None))

    def test_dispatch_threshold_is_exact(self):
        scratch_calls = []
        scratch = types.SimpleNamespace(linear=lambda x, q, s: scratch_calls.append(x) or "scratch")
        layer = types.SimpleNamespace(e21_qweight=None, e21_scales=None, e21_empty=None, e21_workspace=None)
        method = self.ns["ResidueW8A16Method"](4096, 2048, scratch)
        threshold = self.ns["PREFILL_BF16_MIN_TOKENS"]
        self.assertEqual(threshold, 2048)
        self.assertEqual(method.apply(layer, Tensor((threshold - 1, 2048))), "marlin")
        self.assertEqual(method.apply(layer, Tensor((threshold, 2048))), "scratch")
        self.assertEqual(method.apply(layer, Tensor((1, 2048))), "marlin")
        no_scratch = self.ns["ResidueW8A16Method"](4096, 2048, None)
        self.assertEqual(no_scratch.apply(layer, Tensor((threshold * 4, 2048))), "marlin")
        self.assertEqual(len(scratch_calls), 1)

    def test_flag_off_and_idempotence(self):
        def boom(model):
            raise AssertionError("plan_conversion must not run")
        self.ns["plan_conversion"] = boom
        finalize = self.ns["finalize_bf16_residue_w8a16"]
        finalize(fake_model(), environ={})
        finalize(fake_model(), environ={self.ns["ENABLE_ENV"]: "0"})
        ready = fake_model()
        ready._e21_bf16_residue_ready = True
        finalize(ready, environ={self.ns["ENABLE_ENV"]: "1"})


class Launcher(unittest.TestCase):
    def test_four_rank_parity_and_refusals(self):
        delta = (CANDIDATE / "delta.env").read_text()
        c5 = (E03.parent / "e03-c5/delta.env").read_text()
        with tempfile.TemporaryDirectory(prefix="tp4-e21-") as temp:
            root = Path(temp)
            shutil.copyfile(REPO / "scripts/launcher/launch-glm53-tp4.sh", root / "launch.sh")
            (root / "cluster.env").write_text((REPO / "cluster.env.example").read_text() + '''
NODES="n0 n1 n2 n3"
MGMT_IPS="192.0.2.21 192.0.2.22 192.0.2.23 192.0.2.24"
MASTER_IP=192.0.2.21
RELAY_DEST=operator@192.0.2.23
''')
            # E21 is now a complete rollback overlay; the candidate overlay applies only on
            # the complete E03 recipe.
            e03 = (REPO / "scripts/node/reference/baseline-20260919-e03.env").read_text() + "\n"
            e21 = (REPO / "scripts/node/reference/baseline-20260923-e21.env").read_text() + "\n"
            (root / "empty.env").write_text("")
            (root / "e21.env").write_text(e21)
            (root / "e03.env").write_text(e03)
            (root / "candidate.env").write_text(e03 + delta)
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
            target = "/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/"
            old_hook = f"{home}/tp4/overrides/vllm/models/glm5next/nvidia/e20_kda_w8a16.py:{target}e20_kda_w8a16.py:ro"
            new_hook = f"{home}/tp4/experiments/e03/bf16-residue/e20_kda_w8a16.py:{target}e20_kda_w8a16.py:ro"
            residue = f"{home}/tp4/experiments/e03/bf16-residue/e21_bf16_residue.py:{target}e21_bf16_residue.py:ro"
            old_cfg = f"<canonical JSON of {home}/tp4/experiments/e03/kv-transfer-config.json>"
            new_cfg = f"<canonical JSON of {home}/tp4/experiments/e03/bf16-residue/kv-transfer-config.json>"
            flag = f"{MANIFEST['activation']['flag']}=1"
            for rank in range(4):
                before, after = argv(launch("e03.env", rank)), argv(launch("e21.env", rank))
                # The E21 rollback is exactly the measured candidate command.
                self.assertEqual(after, argv(launch("candidate.env", rank)))
                self.assertEqual(Counter(before) - Counter(after), Counter([old_hook, old_cfg]))
                self.assertEqual(Counter(after) - Counter(before),
                                 Counter([new_hook, new_cfg, "-v", residue, "-e", flag]))
                mounts = [after[i + 1] for i, item in enumerate(after[:-1]) if item == "-v"]
                targets = [item.split(":")[1] for item in mounts]
                self.assertEqual(len(targets), len(set(targets)), "Duplicate container mount")
                self.assertEqual(after[after.index("--node-rank") + 1], str(rank))
                self.assertIn("--kv-cache-memory-bytes=16106127360", after)
                self.assertEqual(after[after.index("--max-model-len") + 1], "262144")
                self.assertEqual(after[after.index("--max-num-seqs") + 1], "6")
                self.assertIn("SPARK_MHC_PREFILL_SHARD=1", after)
                self.assertTrue(any("e20_hybrid_scratch.py:" in m for m in mounts))

            bad = {
                "on-c5": e03 + c5 + "\n" + delta,
                "duplicate-hook": e03 + "EXTRA_DOCKER_ENV+=' -v $HOME/tp4/overrides/vllm/models/glm5next/nvidia/e20_kda_w8a16.py:/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/e20_kda_w8a16.py:ro'\n" + delta,
                "flag-present": e03 + "EXTRA_DOCKER_ENV+=' -e VLLM_E21_BF16_RESIDUE_W8A16=0'\n" + delta,
                "residue-present": e03 + "EXTRA_DOCKER_ENV+=' -v $HOME/x/e21_bf16_residue.py:/tmp/e21_bf16_residue.py:ro'\n" + delta,
                "kv-changed": e03 + 'EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS/--kv-cache-memory-bytes=16106127360/--kv-cache-memory-bytes=12884901888}"\n' + delta,
                "applied-twice": e03 + delta + "\n" + delta,
                # Neither the candidate delta nor the historical E03-only C5 overlay may apply
                # on the E21 recipe or on the current default.
                "delta-on-e21": e21 + delta,
                "delta-on-default": delta,
                "c5-on-default": c5,
            }
            for name, text in bad.items():
                (root / "bad.env").write_text(text)
                self.assertNotEqual(launch("bad.env").returncode, 0, name)

    def test_deploy_selector_stages_payload(self):
        selected = subprocess.check_output([
            "find", "scripts/node/experiments/e03", "-type", "f", "(",
            "-name", "*.py", "-o", "-name", "*.json", "-o", "-name", "SHA256SUMS", ")"
        ], cwd=REPO, text=True).splitlines()
        for name in ("e20_kda_w8a16.py", "e21_bf16_residue.py", "kv-transfer-config.json",
                     "manifest.json", "SHA256SUMS"):
            self.assertIn(str((CANDIDATE / name).relative_to(REPO)), selected, name)


if __name__ == "__main__":
    unittest.main(verbosity=1)
