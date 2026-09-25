#!/usr/bin/env python3
"""Offline contract for the E22b drafter W8A16 component; no Torch, GPU, node or network.

Covers pins and vendor provenance of the drafter override, the flag-gated load hook,
family selection and validation against the drafter config, four-rank launcher parity
with the measured overlay and the E21 rollback, and overlay refusals. Native Marlin numerics, CUDA graph capture,
acceptance and throughput need the qualified engine and an authorized window.
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
import types
import unittest


REPO = Path(__file__).resolve().parents[2]
E03 = REPO / "scripts/node/experiments/e03"
CANDIDATE = E03 / "drafter-w8a16"
PARENT = E03 / "bf16-residue"
OVERRIDE = CANDIDATE / "qwen3_dflash2.py"
MODULE = CANDIDATE / "e22_drafter_w8a16.py"
MANIFEST = json.loads((CANDIDATE / "manifest.json").read_text())
CONFIG = MANIFEST["source_identity"]["config_values"]
TP = 4
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def load(path: Path, names: set[str], namespace: dict):
    """Execute selected real top-level definitions with stubbed dependencies."""
    tree = ast.parse(path.read_text(), str(path))
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            node = copy.deepcopy(node)
            node.decorator_list = []
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in names for t in node.targets
        ):
            selected.append(copy.deepcopy(node))
    found = {getattr(n, "name", None) for n in selected} | {
        t.id for n in selected if isinstance(n, ast.Assign) for t in n.targets
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


class Unquantized:
    pass


class Quantized:
    pass


def linear(n, k, **overrides):
    values = dict(weight=Tensor((n, k)), quant_method=Unquantized(), bias=None)
    values.update(overrides)
    return types.SimpleNamespace(**values)


def shape_of(name):
    return next(f["shape_nk_per_rank"] for f in MANIFEST["families"] if f["name"] == name)


def fake_drafter():
    config = types.SimpleNamespace(
        hidden_size=CONFIG["hidden_size"], intermediate_size=CONFIG["intermediate_size"],
        num_hidden_layers=CONFIG["num_hidden_layers"],
        num_attention_heads=CONFIG["num_attention_heads"],
        num_key_value_heads=CONFIG["num_key_value_heads"], head_dim=CONFIG["head_dim"],
        dflash_config={"conv_kernel_size": CONFIG["conv_kernel_size"],
                       "conv_group_size": CONFIG["conv_group_size"]},
    )
    layers = [
        types.SimpleNamespace(
            self_attn=types.SimpleNamespace(qkv_proj=linear(*shape_of("qkv_proj")),
                                            o_proj=linear(*shape_of("o_proj"))),
            mlp=types.SimpleNamespace(gate_up_proj=linear(*shape_of("gate_up_proj")),
                                      down_proj=linear(*shape_of("down_proj"))),
            attention_conv=types.SimpleNamespace(kernel_projection=linear(*shape_of("kernel_projection"))),
            mlp_conv=types.SimpleNamespace(kernel_projection=linear(*shape_of("kernel_projection"))),
        )
        for _ in range(CONFIG["num_hidden_layers"])
    ]
    return types.SimpleNamespace(
        config=config, layers=layers, _num_attn_layers=len(layers),
        _fused_kv_linear=object(), _fused_kv_quant_method=None, _fused_kv_bias=None,
        _fused_kv_weight=Tensor((2560, 4096)),
    )


def module_namespace():
    parent = load(PARENT / "e21_bf16_residue.py",
                  {"GROUP_SIZE", "MARLIN_TILE", "PREFILL_BF16_MIN_TOKENS"}, {"scalar_types": None})
    namespace = {
        "os": os, "json": json,
        "nn": types.SimpleNamespace(Module=object),
        "torch": types.SimpleNamespace(bfloat16="bf16"),
        "UnquantizedLinearMethod": Unquantized,
        "logger": types.SimpleNamespace(info=lambda *a, **k: None),
        "GROUP_SIZE": parent["GROUP_SIZE"], "MARLIN_TILE": parent["MARLIN_TILE"],
        "PREFILL_BF16_MIN_TOKENS": parent["PREFILL_BF16_MIN_TOKENS"],
    }
    names = {
        "ENABLE_ENV", "EXPECTED_LAYERS", "SCRATCH_BUDGET_BYTES", "CONTEXT_KV", "enabled",
        "CONTEXT_ENABLE_ENV", "_flag", "context_enabled",
        "_check_shape", "expected_shapes", "_validate", "_layer_modules", "_context_kv",
        "plan_conversion", "finalize_drafter_w8a16",
    }
    return load(MODULE, names, namespace)


class Pins(unittest.TestCase):
    def test_manifest_and_sums(self):
        for name, entry in MANIFEST["files"].items():
            self.assertEqual(sha(CANDIDATE / name), entry["candidate_sha256"], name)
            if entry.get("base"):
                self.assertEqual(sha(REPO / entry["base"]), entry["base_sha256"], name)
        for path, digest in MANIFEST["source_identity"]["reviewed_runtime_sources_sha256"].items():
            self.assertEqual(sha(REPO / path), digest, path)
        self.assertEqual(
            sha(REPO / "docs/historical_benchmarks/baselines/2026-09-23-e21/baseline.json"),
            MANIFEST["parent"]["baseline_sha256"],
        )
        rows = (CANDIDATE / "SHA256SUMS").read_text().splitlines()
        self.assertEqual(len(rows), 4)
        for row in rows:
            digest, name = row.split("  ", 1)
            self.assertEqual(sha(CANDIDATE / name), digest, name)

    def test_override_is_vendor_bytes_plus_appended_block(self):
        entry = MANIFEST["files"]["qwen3_dflash2.py"]
        data = OVERRIDE.read_bytes()
        marker = entry["vendor_prefix_marker"].encode()
        self.assertEqual(data.count(marker), 1)
        prefix, _, _ = data.partition(marker)
        self.assertEqual(hashlib.sha256(prefix).hexdigest(), entry["vendor_sha256"])
        self.assertEqual(entry["vendor_sha256"],
                         MANIFEST["source_identity"]["reviewed_vendor_sources_sha256"][
                             "vllm/model_executor/models/qwen3_dflash2.py"])
        vendor = ast.parse(prefix.decode())
        self.assertFalse(any(isinstance(n, ast.FunctionDef) and n.name == "load_weights"
                             for n in ast.walk(vendor)),
                         "the vendor DFlash2 class must inherit load_weights")

    def test_cache_config_differs_only_in_root(self):
        self.check_root_only("kv-transfer-config-e22b.json")

    def check_root_only(self, name):
        accepted = json.loads((PARENT / "kv-transfer-config.json").read_text())
        candidate = json.loads((CANDIDATE / name).read_text())
        a_extra = accepted.pop("kv_connector_extra_config")
        c_extra = candidate.pop("kv_connector_extra_config")
        self.assertEqual(accepted, candidate)
        self.assertNotEqual(a_extra.pop("spark_cache_root"), c_extra.pop("spark_cache_root"))
        self.assertEqual(a_extra, c_extra)
        self.assertEqual(c_extra["spark_cache_draft_policy"], "separate")


class Hook(unittest.TestCase):
    def appended(self):
        marker = MANIFEST["files"]["qwen3_dflash2.py"]["vendor_prefix_marker"]
        return ast.parse((marker + OVERRIDE.read_text().partition(marker)[2]).lstrip("\n"))

    def test_appended_block_is_one_gated_load_hook(self):
        body = self.appended().body
        self.assertEqual([type(n).__name__ for n in body], ["FunctionDef", "Assign"])
        function, assign = body
        self.assertEqual(ast.unparse(assign), "DFlash2Qwen3ForCausalLM.load_weights = _e22_load_weights")
        steps = function.body
        # The vendor loader runs first, so the fused context K/V copy already exists.
        self.assertEqual(ast.unparse(steps[0]),
                         "loaded = super(DFlash2Qwen3ForCausalLM, self).load_weights(weights)")
        gate = next(n for n in steps if isinstance(n, ast.If))
        self.assertIn("VLLM_E22_DRAFTER_W8A16", ast.unparse(gate.test))
        self.assertIn("finalize_drafter_w8a16(self.model)", ast.unparse(gate))
        self.assertEqual(ast.unparse(steps[-1]), "return loaded")

    def test_flag_is_shared_and_strict(self):
        ns = module_namespace()
        self.assertEqual(MANIFEST["activation"]["flags"][ns["ENABLE_ENV"]], "1")
        self.assertEqual(MANIFEST["activation"]["flags"][ns["CONTEXT_ENABLE_ENV"]], "0")
        for value, expected in (("", False), ("0", False), (" 0 ", False), ("1", True)):
            self.assertIs(ns["enabled"]({ns["ENABLE_ENV"]: value}), expected, value)
        self.assertFalse(ns["enabled"]({}))
        with self.assertRaises(RuntimeError):
            ns["enabled"]({ns["ENABLE_ENV"]: "yes"})

    def test_reused_primitives_exist_in_e21(self):
        tree = ast.parse(MODULE.read_text())
        imported = next(n for n in tree.body if isinstance(n, ast.ImportFrom)
                        and n.module == "e21_bf16_residue" and n.level == 1)
        e21 = {getattr(n, "name", None) for n in ast.parse(
            (PARENT / "e21_bf16_residue.py").read_text()).body} | {
            t.id for n in ast.parse((PARENT / "e21_bf16_residue.py").read_text()).body
            if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
        for alias in imported.names:
            self.assertIn(alias.name, e21, alias.name)


class Selection(unittest.TestCase):
    def setUp(self):
        self.ns = module_namespace()

    def test_census_resolves_to_manifest(self):
        # E22b: the context K/V projection is neither checked nor converted, even when the
        # vendor copy would not satisfy the conversion contract.
        model = fake_drafter()
        model._fused_kv_weight = None
        work, context = self.ns["plan_conversion"](model, TP, include_context=False)
        self.assertIsNone(context)
        by_family = Counter(name for name, *_ in work)
        expected = {f["name"]: f["modules"] for f in MANIFEST["families"]}
        self.assertEqual(dict(by_family), expected)
        fields = MANIFEST["activation"]["expected_signature_fields"]
        self.assertEqual(sum(by_family.values()), fields["modules"])
        self.assertEqual(sorted(by_family), fields["families"])
        self.assertIs(fields["context_kv_w8a16"], False)
        self.assertEqual(fields["added_scratch_bytes"], 0)
        for name, _, _, shape in work:
            self.assertEqual(list(shape), shape_of(name), name)
        self.assertTrue(self.ns["context_enabled"]({}))
        self.assertFalse(self.ns["context_enabled"]({self.ns["CONTEXT_ENABLE_ENV"]: "0"}))
        with self.assertRaises(RuntimeError):
            self.ns["context_enabled"]({self.ns["CONTEXT_ENABLE_ENV"]: "no"})

    def test_shapes_follow_config(self):
        shapes = self.ns["expected_shapes"](fake_drafter().config, TP)
        for family in MANIFEST["families"]:
            self.assertEqual(list(shapes[family["name"]]), family["shape_nk_per_rank"])

    def test_excluded_modules_are_not_selected(self):
        source = MODULE.read_text()
        tree = ast.parse(source)
        census = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                      and n.name == "_layer_modules")
        text = ast.unparse(census)
        for name in ("fc", "lm_head", "embed_tokens", "hidden_projection"):
            self.assertNotIn(f".{name}", text, name)

    def refuse(self, mutate):
        model = fake_drafter()
        mutate(model)
        with self.assertRaises(RuntimeError):
            self.ns["plan_conversion"](model, TP)

    def test_refusals(self):
        first = lambda m: m.layers[0]
        self.refuse(lambda m: setattr(first(m).self_attn.qkv_proj, "weight", Tensor((1537, 4096))))
        self.refuse(lambda m: setattr(first(m).mlp.down_proj, "weight", Tensor((4096, 3072), dtype="fp16")))
        self.refuse(lambda m: setattr(first(m).mlp.gate_up_proj, "weight", Tensor((6144, 4096), device="cpu")))
        self.refuse(lambda m: setattr(first(m).self_attn.o_proj, "bias", Tensor((4096,))))
        self.refuse(lambda m: setattr(first(m).mlp_conv.kernel_projection, "quant_method", Quantized()))
        self.refuse(lambda m: delattr(first(m), "attention_conv"))
        self.refuse(lambda m: m.layers.pop())
        self.refuse(lambda m: delattr(m, "_num_attn_layers"))
        self.refuse(lambda m: setattr(m, "_fused_kv_weight", None))
        self.refuse(lambda m: setattr(m, "_fused_kv_weight", Tensor((2048, 4096))))
        self.refuse(lambda m: setattr(m, "_fused_kv_quant_method", Quantized()))
        self.refuse(lambda m: setattr(m, "_fused_kv_bias", Tensor((2560,))))
        with self.assertRaises(RuntimeError):
            self.ns["plan_conversion"](fake_drafter(), 3)

    def test_flag_off_and_idempotence(self):
        def boom(*args):
            raise AssertionError("plan_conversion must not run")
        self.ns["plan_conversion"] = boom
        finalize = self.ns["finalize_drafter_w8a16"]
        finalize(fake_drafter(), environ={})
        finalize(fake_drafter(), environ={self.ns["ENABLE_ENV"]: "0"})
        ready = fake_drafter()
        ready._e22_drafter_ready = True
        finalize(ready, environ={self.ns["ENABLE_ENV"]: "1"})


class Launcher(unittest.TestCase):
    def test_four_rank_parity_and_refusals(self):
        delta = (CANDIDATE / "delta-context-bf16.env").read_text()
        with tempfile.TemporaryDirectory(prefix="tp4-e22b-") as temp:
            root = Path(temp)
            shutil.copyfile(REPO / "scripts/launcher/launch-glm53-tp4.sh", root / "launch.sh")
            (root / "cluster.env").write_text((REPO / "cluster.env.example").read_text() + '''
NODES="n0 n1 n2 n3"
MGMT_IPS="192.0.2.21 192.0.2.22 192.0.2.23 192.0.2.24"
MASTER_IP=192.0.2.21
RELAY_DEST=operator@192.0.2.23
''')
            e03 = (REPO / "scripts/node/reference/baseline-20260919-e03.env").read_text() + "\n"
            # E22b is the default; its measured overlay applies on the complete E21 recipe,
            # which the immediate rollback restores.
            e21 = (REPO / "scripts/node/reference/baseline-20260923-e21.env").read_text() + "\n"
            # E27 is now the default; its complete E22b return must equal the measured E22b load.
            e22b = (REPO / "scripts/node/reference/baseline-20260924-e22b.env").read_text() + "\n"
            (root / "empty.env").write_text("")
            (root / "e21.env").write_text(e21)
            (root / "e22b.env").write_text(e22b)
            (root / "candidate.env").write_text(e21 + delta)
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
            src = f"{home}/tp4/experiments/e03/drafter-w8a16/"
            override = f"{src}qwen3_dflash2.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_dflash2.py:ro"
            module = f"{src}e22_drafter_w8a16.py:/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/e22_drafter_w8a16.py:ro"
            old_cfg = f"<canonical JSON of {home}/tp4/experiments/e03/bf16-residue/kv-transfer-config.json>"
            new_cfg = f"<canonical JSON of {src}kv-transfer-config-e22b.json>"
            flags = ["VLLM_E22_DRAFTER_W8A16=1", "VLLM_E22_CONTEXT_KV_W8A16=0"]
            for rank in range(4):
                before, after = argv(launch("e21.env", rank)), argv(launch("candidate.env", rank))
                self.assertEqual(Counter(before) - Counter(after), Counter([old_cfg]))
                self.assertEqual(Counter(after) - Counter(before),
                                 Counter([new_cfg, "-v", override, "-v", module, "-e", flags[0], "-e", flags[1]]))
                # The E22b return is exactly the measured E22b command; the E29 default adds the
                # prefill cadence, the E29 scheduler (E27c patch included) and engine core with
                # their flags, and the E28b draft length and KV pool.
                self.assertEqual(argv(launch("e22b.env", rank)), after)
                current = argv(launch("empty.env", rank))
                spec = lambda k: ('{"method":"dflash","model":"/draft","num_speculative_tokens":%d,'
                                  '"num_speculative_tokens_per_batch_size":[[1,1,%d],[2,6,3]],'
                                  '"kv_cache_dtype":"fp8_e4m3"}' % (k, k))
                self.assertEqual(Counter(after) - Counter(current),
                                 Counter([spec(5), "--kv-cache-memory-bytes=16106127360"]))
                self.assertEqual(Counter(current) - Counter(after), Counter([
                    "--prefill-schedule-interval", "8", "-v",
                    str(Path.home()) + "/tp4/experiments/e03/end-drain/scheduler.py:"
                    "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py:ro",
                    "-e", "VLLM_E27B_SHORT_PREFILL_TOKENS=2048", "-e", "VLLM_E27C_CADENCE_WHEN_QUEUED=1",
                    spec(7), "-e", "VLLM_ADAPTIVE_K_HI=7", '--compilation-config={"max_cudagraph_capture_size":72}',
                    "--kv-cache-memory-bytes=17179869184", "-v",
                    str(Path.home()) + "/tp4/experiments/e03/end-drain/core.py:"
                    "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py:ro",
                    "-e", "VLLM_E29_END_DRAIN=1", "-e", "VLLM_E29_IDLE_COALESCE_MS=4", "-e", "VLLM_E29_TRACE=0"]))
                mounts = [after[i + 1] for i, item in enumerate(after[:-1]) if item == "-v"]
                targets = [item.split(":")[1] for item in mounts]
                self.assertEqual(len(targets), len(set(targets)), "Duplicate container mount")
                self.assertEqual(after[after.index("--node-rank") + 1], str(rank))
                self.assertEqual(after[after.index("--max-num-seqs") + 1], "6")
                self.assertIn("VLLM_E21_BF16_RESIDUE_W8A16=1", after)
                self.assertTrue(any("e21_bf16_residue.py:" in m for m in mounts))

            bad = {
                "on-e03-rollback": e03 + delta,
                "applied-twice": e21 + delta + "\n" + delta,
                "flag-present": e21 + "EXTRA_DOCKER_ENV+=' -e VLLM_E22_CONTEXT_KV_W8A16=1'\n" + delta,
                "override-present": e21 + "EXTRA_DOCKER_ENV+=' -v $HOME/x/qwen3_dflash2.py:/tmp/qwen3_dflash2.py:ro'\n" + delta,
                "kv-changed": e21 + 'EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS/--kv-cache-memory-bytes=16106127360/--kv-cache-memory-bytes=12884901888}"\n' + delta,
                "max-seqs-5": e21 + "MAX_NUM_SEQS=5\n" + delta,
                # The promoted default already carries the drafter conversion.
                "on-default": delta,
            }
            for name, text in bad.items():
                (root / "bad.env").write_text(text)
                self.assertNotEqual(launch("bad.env").returncode, 0, name)

    def test_deploy_selector_stages_payload(self):
        selected = subprocess.check_output([
            "find", "scripts/node/experiments/e03", "-type", "f", "(",
            "-name", "*.py", "-o", "-name", "*.json", "-o", "-name", "SHA256SUMS", ")"
        ], cwd=REPO, text=True).splitlines()
        for name in ("qwen3_dflash2.py", "e22_drafter_w8a16.py", "kv-transfer-config-e22b.json",
                     "manifest.json", "SHA256SUMS"):
            self.assertIn(str((CANDIDATE / name).relative_to(REPO)), selected, name)


if __name__ == "__main__":
    unittest.main(verbosity=1)
