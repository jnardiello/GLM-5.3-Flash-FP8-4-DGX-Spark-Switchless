#!/usr/bin/env python3
"""CPU tests of hybrid dispatch/lifetime contracts, without Torch or a GPU.

Execute the production Python functions with narrow dependency stubs. These tests
cover configuration and integration; native Marlin/Triton numerics and CUDA graph
execution require the qualified engine/GPU environment.
"""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
import sys
import threading
import time
import types
import unittest
from unittest.mock import patch


OVERRIDES = Path(__file__).resolve().parents[1] / "node/overrides/vllm"
KDA = OVERRIDES / "models/glm5next/nvidia/e20_kda_w8a16.py"
SCRATCH = OVERRIDES / "models/glm5next/nvidia/e20_hybrid_scratch.py"
WORKER = OVERRIDES / "v1/worker/gpu_worker.py"


def definitions(path, names, namespace):
    """Load selected real definitions, removing only dependency decorators."""
    tree = ast.parse(path.read_text(), str(path))
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            node = copy.deepcopy(node)
            node.decorator_list = []
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in names for target in node.targets
        ):
            selected.append(copy.deepcopy(node))
    assert len(selected) == len(names), (path, names)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", level=0,
        names=[ast.alias(name="annotations")]), *selected], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


class Tensor:
    def __init__(self, shape, dtype="bf16", device="cuda", contiguous=True):
        self.shape = shape
        self.dtype = dtype
        self.device = types.SimpleNamespace(type=device)
        self._contiguous = contiguous

    def numel(self):
        result = 1
        for size in self.shape:
            result *= size
        return result

    def is_contiguous(self):
        return self._contiguous

    def contiguous(self):
        return self

    @property
    def T(self):
        return Tensor(tuple(reversed(self.shape)), self.dtype, self.device.type)

    def element_size(self):
        return 4 if self.dtype == "int32" else 2

    def new_empty(self, shape):
        return Tensor(shape, self.dtype)


class Unquantized:
    pass


class HybridTests(unittest.TestCase):
    def kda_namespace(self):
        self.marlin_calls = []
        self.packed = []
        namespace = {
            "__name__": "hybrid_test.kda", "__package__": "hybrid_test",
            "LinearMethodBase": object, "UnquantizedLinearMethod": Unquantized,
            "torch": types.SimpleNamespace(bfloat16="bf16",
                cuda=types.SimpleNamespace(empty_cache=lambda: None)),
            "apply_gptq_marlin_linear": lambda **kwargs: self.marlin_calls.append(kwargs) or "marlin",
            "WTYPE": "uint8b128", "json": json,
            "logger": types.SimpleNamespace(info=lambda *args: None),
            "_pack_projection": lambda layer, scratch: self.packed.append((layer, scratch)) or
                {"original_bytes": 1, "packed_bytes": 1, "padded_n": 6400, "padded_k": 4096},
        }
        return definitions(KDA, {
            "GROUP_SIZE", "INPUT_SIZE", "OUTPUT_SIZE", "PREFILL_BF16_MIN_TOKENS",
            "KDAInputW8A16Method", "finalize_kda_input_w8a16",
        }, namespace)

    def model(self, count=34):
        layers = [types.SimpleNamespace(layer_kind="kda", self_attn=types.SimpleNamespace(
            in_proj_qkvgfab=types.SimpleNamespace(weight=Tensor((6288, 4096)),
                quant_method=Unquantized(), bias=None))) for _ in range(count)]
        layers.insert(3, types.SimpleNamespace(layer_kind="attention"))
        model = types.SimpleNamespace(layers=layers, buffers={})
        model.register_buffer = lambda name, value, persistent: model.buffers.__setitem__(name, value)
        return model

    def test_dispatch_uses_flattened_token_count_and_exact_threshold(self):
        namespace = self.kda_namespace()
        scratch_calls = []
        scratch = types.SimpleNamespace(linear=lambda *args: scratch_calls.append(args) or "scratch")
        method = namespace["KDAInputW8A16Method"](scratch)
        layer = types.SimpleNamespace(e20_qweight=object(), e20_scales=object(),
            e20_empty=object(), e20_workspace=object())
        for shape, expected in [((6, 4096), "marlin"), ((2047, 4096), "marlin"),
                                ((2048, 4096), "scratch"), ((2, 1024, 4096), "scratch"),
                                ((8192, 4096), "scratch")]:
            with self.subTest(shape=shape):
                self.assertEqual(method.apply(layer, Tensor(shape)), expected)
        self.assertEqual(len(self.marlin_calls), 2)
        self.assertEqual(len(scratch_calls), 3)
        self.assertIs(self.marlin_calls[0]["weight"], scratch_calls[0][1])
        self.assertIs(self.marlin_calls[0]["weight_scale"], scratch_calls[0][2])
        self.assertTrue(self.marlin_calls[0]["use_fp32_reduce"])
        self.assertEqual(self.marlin_calls[0]["output_size_per_partition"], 6288)
        self.assertEqual(namespace["KDAInputW8A16Method"]().apply(layer, Tensor((8192, 4096))), "marlin")

    def test_all_projections_are_validated_before_any_conversion(self):
        namespace = self.kda_namespace()
        cases = [("shape", (6287, 4096)), ("dtype", "fp16"),
                 ("device", types.SimpleNamespace(type="cpu"))]
        for attr, value in cases:
            with self.subTest(attribute=attr):
                model = self.model()
                setattr(model.layers[-1].self_attn.in_proj_qkvgfab.weight, attr, value)
                with self.assertRaisesRegex(RuntimeError, "projection differs"):
                    namespace["finalize_kda_input_w8a16"](model)
                self.assertEqual(self.packed, [])
        for attribute, value in [("bias", object()), ("quant_method", object())]:
            model = self.model()
            setattr(model.layers[-1].self_attn.in_proj_qkvgfab, attribute, value)
            with self.assertRaisesRegex(RuntimeError, "projection differs"):
                namespace["finalize_kda_input_w8a16"](model)
            self.assertEqual(self.packed, [])
        with self.assertRaisesRegex(RuntimeError, "expected 34"):
            namespace["finalize_kda_input_w8a16"](self.model(33))
        self.assertEqual(self.packed, [])

    def test_one_shared_scratch_and_idempotent_finalization(self):
        namespace = self.kda_namespace()
        instances = []

        def make_scratch(device):
            instance = types.SimpleNamespace(scratch=object(), inverse_weight_perm=object(), resident_bytes=51515392)
            instances.append(instance)
            return instance

        module = types.ModuleType("hybrid_test.e20_hybrid_scratch")
        module.MarlinBF16Scratch = make_scratch
        model = self.model()
        with patch.dict(sys.modules, {module.__name__: module}):
            namespace["finalize_kda_input_w8a16"](model)
            namespace["finalize_kda_input_w8a16"](model)
        self.assertEqual(len(instances), 1)
        self.assertEqual(len(self.packed), 34)
        self.assertTrue(all(scratch is instances[0] for _, scratch in self.packed))
        self.assertIs(model.buffers["e20_shared_bf16_scratch"], instances[0].scratch)
        self.assertIs(model.buffers["e20_inverse_weight_perm"], instances[0].inverse_weight_perm)

    def test_packing_pads_to_6400_and_releases_original_parameter(self):
        namespace = self.kda_namespace()
        quantization_calls = []

        def quantize(weight, weight_type, group_size, act_order):
            quantization_calls.append((weight.shape, group_size, act_order))
            return weight, Tensor(weight.shape), Tensor((32, 6288)), None, None

        def pad_weight(weight, original_n, original_k, padded_n, padded_k):
            self.assertEqual((original_n, original_k, padded_n, padded_k), (6288, 4096, 6400, 4096))
            return Tensor((padded_k // 4, padded_n), "int32")

        def repack(weight, empty, padded_k, padded_n, bits):
            self.assertEqual(bits, 8)
            self.assertEqual(weight.shape, (1024, 6400))
            return Tensor((padded_k // 16, padded_n * 4), "int32")

        namespace.update({
            "WTYPE": types.SimpleNamespace(size_bits=8),
            "gptq_quantize_weights": quantize,
            "pack_quantized_values_into_int32": lambda weight, weight_type, packed_dim: Tensor((1024, 6288), "int32"),
            "marlin_padded_nk": lambda *args: (6336, 4096),
            "marlin_pad_qweight": pad_weight,
            "marlin_pad_scales": lambda scales, n, k, pn, pk, group: Tensor((pk // group, pn)),
            "marlin_permute_scales": lambda scales, **kwargs: scales,
            "marlin_make_workspace_new": lambda device: Tensor((16,), "int32"),
            "ops": types.SimpleNamespace(gptq_marlin_repack=repack),
        })
        namespace["torch"].int32 = "int32"
        namespace["torch"].empty = lambda size, dtype, device: Tensor((size,), dtype)
        definitions(KDA, {"_pack_projection"}, namespace)
        weight = Tensor((6288, 4096))
        layer = types.SimpleNamespace(weight=weight, _parameters={"weight": weight}, buffers={})
        layer.register_buffer = lambda name, tensor, persistent: layer.buffers.__setitem__(name, tensor)
        scratch = object()
        receipt = namespace["_pack_projection"](layer, scratch)
        self.assertEqual(quantization_calls, [((4096, 6288), 128, False)])
        self.assertEqual((receipt["padded_n"], receipt["padded_k"]), (6400, 4096))
        self.assertNotIn("weight", layer._parameters)
        self.assertEqual(set(layer.buffers), {"e20_qweight", "e20_scales", "e20_empty", "e20_workspace"})
        self.assertIs(layer.quant_method.scratch, scratch)
        self.assertLess(receipt["packed_bytes"], receipt["original_bytes"])

    def test_scratch_storage_rejects_incompatible_layouts(self):
        namespace = definitions(SCRATCH, {"N", "K", "GROUP_SIZE", "MarlinBF16Scratch"},
            {"torch": types.SimpleNamespace(int32="int32", bfloat16="bf16")})
        cls = namespace["MarlinBF16Scratch"]
        owner = cls.__new__(cls)
        owner.scratch = Tensor((6288, 4096))
        weight, scales = Tensor((256, 25600), "int32"), Tensor((32, 6400))
        owner.check_storage(weight, scales)
        for attr, value in [("shape", (255, 25600)), ("dtype", "int8"),
                            ("_contiguous", False), ("device", types.SimpleNamespace(type="cpu"))]:
            with self.subTest(attribute=attr):
                bad = copy.copy(weight)
                setattr(bad, attr, value)
                with self.assertRaisesRegex(RuntimeError, "storage contract"):
                    owner.check_storage(bad, scales)
        with self.assertRaisesRegex(RuntimeError, "storage contract"):
            owner.check_storage(weight, Tensor((31, 6400)))

    def test_custom_op_declares_mutation_and_finishes_gemm_before_return(self):
        tree = ast.parse(SCRATCH.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "dequant_marlin_bf16_linear")
        decorator = function.decorator_list[0]
        mutation = next(item.value for item in decorator.keywords if item.arg == "mutates_args")
        self.assertEqual(ast.literal_eval(mutation), ("scratch",))
        events = []
        namespace = definitions(SCRATCH, {"dequant_marlin_bf16_linear", "_dequant_marlin_bf16_linear_fake", "N"}, {
            "_dequant_into": lambda *args: events.append(("write", args[-1])),
            "F": types.SimpleNamespace(linear=lambda x, scratch: events.append(("read", scratch)) or "output"),
        })
        scratch = object()
        output = namespace["dequant_marlin_bf16_linear"](object(), object(), object(), object(), scratch)
        self.assertEqual(output, "output")
        self.assertEqual(events, [("write", scratch), ("read", scratch)])
        fake = namespace["_dequant_marlin_bf16_linear_fake"](Tensor((2, 2048, 4096)), None, None, None, None)
        self.assertEqual(fake.shape, (2, 2048, 6288))


class MemoryProbeTests(unittest.TestCase):
    def test_probe_reads_counters_without_reset_or_cuda_synchronization(self):
        records = []
        sampled = threading.Event()
        forbidden_calls = []

        def forbidden(*args, **kwargs):
            forbidden_calls.append(True)
            raise AssertionError("sampling must not synchronize or reset counters")

        def log(format_string, encoded):
            records.append(json.loads(encoded))
            sampled.set()

        stats = {"allocated_bytes.all.current": 20, "reserved_bytes.all.current": 32,
                 "allocated_bytes.all.peak": 40, "num_alloc_retries": 2,
                 "allocated_bytes.small_pool.current": 5}
        cuda = types.SimpleNamespace(memory=types.SimpleNamespace(), memory_stats=lambda device: stats,
            synchronize=forbidden, reset_peak_memory_stats=forbidden, empty_cache=forbidden)
        namespace = definitions(WORKER, {"_start_e20_memory_probe"}, {"time": time, "os": __import__("os")})
        stop = namespace["_start_e20_memory_probe"](types.SimpleNamespace(cuda=cuda), "cuda", 2,
            types.SimpleNamespace(info=log))
        try:
            self.assertTrue(sampled.wait(2), "memory probe did not produce a record")
        finally:
            stop.set()
        self.assertEqual(forbidden_calls, [])
        self.assertEqual(records[0]["rank"], 2)
        self.assertEqual(records[0]["cuda_reserved_minus_allocated_bytes"], 12)
        self.assertEqual(records[0]["cuda_allocator"]["allocated_bytes.all.peak"], 40)
        self.assertNotIn("allocated_bytes.small_pool.current", records[0]["cuda_allocator"])
        self.assertEqual(records[0]["host_pinned_allocator_status"], "api_unavailable")


if __name__ == "__main__":
    unittest.main(verbosity=2)
