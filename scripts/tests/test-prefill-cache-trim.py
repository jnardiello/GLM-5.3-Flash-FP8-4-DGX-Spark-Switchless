#!/usr/bin/env python3
"""Offline contract for the opt-in eager-prefill allocator-cache trim."""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest


REPO = Path(__file__).resolve().parents[2]
CANDIDATE = REPO / "scripts/node/experiments/e03/prefill-cache-trim"
PROTECTED16 = REPO / "scripts/node/reference/operational-20260929-sparkcache-protected.env"
BASE = REPO / "scripts/node/overrides/vllm/v1/worker/gpu_worker.py"
WORKER = CANDIDATE / "gpu_worker.py"
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()  # noqa: E731


def helper_namespace() -> dict:
    tree = ast.parse(WORKER.read_text())
    wanted = {
        "_prefill_cache_trim_decision",
        "_trim_allocator_cache",
        "_maybe_trim_prefill_cache",
    }
    body = [node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in wanted]
    namespace = {"json": json, "time": SimpleNamespace(monotonic_ns=lambda: 0)}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(WORKER), "exec"), namespace)
    return namespace


class FakeCached:
    def __init__(self, phases: dict[str, bool], error: Exception | None = None):
        self.req_ids = list(phases)
        self.phases = phases
        self.error = error

    def is_context_phase(self, req_id):
        if self.error:
            raise self.error
        return self.phases[req_id]


def output(*, total=8192, new=False, phases=None):
    return SimpleNamespace(
        total_num_scheduled_tokens=total,
        scheduled_new_reqs=[object()] if new else [],
        scheduled_cached_reqs=FakeCached(phases or {}),
    )


class FakeLogger:
    def __init__(self):
        self.records = []

    def info(self, fmt, *args):
        self.records.append(("info", fmt % args))

    def warning(self, fmt, *args):
        self.records.append(("warning", fmt % args))

    def error(self, fmt, *args):
        self.records.append(("error", fmt % args))


class ProvenanceAndOverlay(unittest.TestCase):
    def test_manifest_hashes_and_exact_patch(self):
        manifest = json.loads((CANDIDATE / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "experimental_diagnostic")
        self.assertEqual(manifest["parent"]["sha256"], sha(BASE))
        self.assertEqual(manifest["candidate"]["sha256"], sha(WORKER))
        patch = CANDIDATE / "gpu_worker.patch"
        self.assertEqual(manifest["candidate"]["patch_sha256"], sha(patch))
        with tempfile.TemporaryDirectory(prefix="prefill-trim-patch-") as temporary:
            reconstructed = Path(temporary) / "gpu_worker.py"
            reconstructed.write_bytes(BASE.read_bytes())
            result = subprocess.run(
                ["patch", "-s", str(reconstructed), str(patch)],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(reconstructed.read_bytes(), WORKER.read_bytes())
        sums = dict(line.split(None, 1)[::-1] for line in
                    (CANDIDATE / "SHA256SUMS").read_text().splitlines())
        self.assertEqual({name.strip() for name in sums},
                         {"gpu_worker.py", "manifest.json"})
        for name, digest in sums.items():
            self.assertEqual(sha(CANDIDATE / name.strip()), digest)

    def test_launcher_manifest_case_accepts_bundle_and_rejects_corruption(self):
        launcher = (REPO / "scripts/launcher/launch-glm53-tp4.sh").read_text()
        marker = '# The opt-in allocator diagnostic has a separate, deployable source manifest.\n'
        start = launcher.index("case ", launcher.index(marker))
        end = launcher.index("\nesac", start) + len("\nesac")
        manifest_case = launcher[start:end]
        with tempfile.TemporaryDirectory(prefix="prefill-trim-launcher-") as temporary:
            env_dir = Path(temporary)
            deployed = env_dir / "experiments/e03/prefill-cache-trim"
            deployed.mkdir(parents=True)
            for name in ("gpu_worker.py", "manifest.json", "SHA256SUMS"):
                (deployed / name).write_bytes((CANDIDATE / name).read_bytes())
            env = dict(os.environ, ENV_DIR=str(env_dir), DRY_RUN="0",
                       EXTRA_DOCKER_ENV="-v /x/prefill-cache-trim/gpu_worker.py:/x:ro")
            good = subprocess.run(["bash", "-c", manifest_case], env=env,
                                  capture_output=True, text=True)
            self.assertEqual(good.returncode, 0, good.stderr)
            (deployed / "gpu_worker.py").write_bytes(b"corrupt\n")
            bad = subprocess.run(["bash", "-c", manifest_case], env=env,
                                 capture_output=True, text=True)
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn("source manifest failed", bad.stderr)

    def test_overlay_changes_only_worker_mount_and_flag(self):
        command = ('source "$1"; source "$2"; before=$EXTRA_DOCKER_ENV; '
                   'source "$3"; printf "%s\\n--AFTER--\\n%s\\n" '
                   '"$before" "$EXTRA_DOCKER_ENV"')
        result = subprocess.run(
            ["bash", "-c", command, "trim-test", str(REPO / "cluster.env.example"),
             str(PROTECTED16), str(CANDIDATE / "delta.env")],
            capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        before, after = result.stdout.split("\n--AFTER--\n")
        old = "$HOME/tp4/overrides/vllm/v1/worker/gpu_worker.py"
        new = "$HOME/tp4/experiments/e03/prefill-cache-trim/gpu_worker.py"
        self.assertEqual(before.split().count(old + ":/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py:ro"), 1)
        self.assertEqual(after.split().count(new + ":/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py:ro"), 1)
        expected = before.replace(old, new) + " -e VLLM_PREFILL_CACHE_TRIM=1"
        self.assertEqual(after.strip(), expected)
        self.assertNotIn("per_process_memory_fraction", after)
        self.assertNotIn("garbage_collection_threshold", after)

    def test_overlay_refuses_prior_selection(self):
        command = ('source "$1"; EXTRA_DOCKER_ENV+=" -e VLLM_PREFILL_CACHE_TRIM=1"; '
                   'source "$2"')
        result = subprocess.run(
            ["bash", "-c", command, "trim-test", str(REPO / "cluster.env.example"),
             str(CANDIDATE / "delta.env")], capture_output=True, text=True
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires exactly one validated", result.stderr)


class Classification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = helper_namespace()

    def decision(self, value, captures=(1, 2, 4, 8, 16, 32, 64, 72)):
        return self.ns["_prefill_cache_trim_decision"](value, captures)

    def test_real_scheduler_fields_select_only_eager_prefill(self):
        self.assertEqual(self.decision(output(new=True)), (True, "eager_prefill"))
        self.assertEqual(self.decision(output(phases={"r": True})),
                         (True, "eager_prefill"))
        self.assertEqual(self.decision(output(phases={"r": False})),
                         (False, "decode_only"))
        self.assertEqual(self.decision(output(total=72, new=True)),
                         (False, "captured_or_small_prefill"))

    def test_unknown_inputs_fail_closed(self):
        self.assertEqual(self.decision(output(new=True), None),
                         (False, "missing_capture_sizes"))
        self.assertEqual(self.decision(output(new=True), (72, "bad")),
                         (False, "unknown_capture_sizes"))
        value = output(phases={"r": True})
        value.scheduled_cached_reqs.error = RuntimeError("unknown")
        self.assertEqual(self.decision(value), (False, "unknown_cached_phase"))
        value.scheduled_new_reqs = [object()]
        self.assertEqual(self.decision(value), (False, "unknown_cached_phase"))

    def test_flag_and_warmup_gates(self):
        maybe = self.ns["_maybe_trim_prefill_cache"]
        calls, logger = [], FakeLogger()
        args = dict(scheduler_output=output(new=True), capture_sizes=[72],
                    trim_fn=lambda metadata: calls.append(metadata), rank=0,
                    trim_logger=logger)
        self.assertFalse(maybe(enabled=False, runtime_ready=True, **args))
        self.assertFalse(maybe(enabled=True, runtime_ready=False, **args))
        self.assertEqual(calls, [])
        self.assertTrue(maybe(enabled=True, runtime_ready=True, **args))
        self.assertEqual(calls, [{"total_num_scheduled_tokens": 8192,
                                  "num_scheduled_new_requests": 1}])


class TrimAndIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = helper_namespace()

    def fake_torch(self, calls, snapshots, error=None):
        def memory_stats(device):
            calls.append(("stats", device))
            return snapshots.pop(0)

        def empty_cache():
            calls.append(("empty_cache",))
            if error:
                raise error

        return SimpleNamespace(
            cuda=SimpleNamespace(memory_stats=memory_stats),
            accelerator=SimpleNamespace(empty_cache=empty_cache),
        )

    def test_receipt_and_torch_invocation_order(self):
        calls, logger, ticks = [], FakeLogger(), iter((100, 160))
        snapshots = [
            {"allocated_bytes.all.current": 80, "reserved_bytes.all.current": 120},
            {"allocated_bytes.all.current": 80, "reserved_bytes.all.current": 90},
        ]
        mem = iter((200, 230))
        self.ns["_trim_allocator_cache"](
            self.fake_torch(calls, snapshots), "cuda:0", 2, logger,
            {"total_num_scheduled_tokens": 8192,
             "num_scheduled_new_requests": 2},
            monotonic_ns=lambda: next(ticks),
            wall_time_ns=lambda: 1234,
            pid_fn=lambda: 4321,
            mem_available_fn=lambda: calls.append(("mem",)) or next(mem),
        )
        self.assertEqual(calls, [("stats", "cuda:0"), ("mem",), ("empty_cache",),
                                 ("stats", "cuda:0"), ("mem",)])
        level, line = logger.records[-1]
        receipt = json.loads(line.split(" ", 1)[1])
        self.assertEqual((level, receipt["status"], receipt["rank"]),
                         ("info", "ok", 2))
        self.assertEqual(receipt["stage"], "after")
        self.assertEqual(receipt["wall_time_ns"], 1234)
        self.assertEqual(receipt["monotonic_ns"], 100)
        self.assertEqual(receipt["pid"], 4321)
        self.assertEqual(receipt["total_num_scheduled_tokens"], 8192)
        self.assertEqual(receipt["num_scheduled_new_requests"], 2)
        self.assertEqual(receipt["reclaimed_reserved_bytes"], 30)
        self.assertEqual(receipt["duration_ns"], 60)

    def test_torch_error_emits_failed_receipt_and_reraises(self):
        calls, logger, ticks = [], FakeLogger(), iter((100, 130))
        snapshots = [{"allocated_bytes.all.current": 80,
                      "reserved_bytes.all.current": 120}]
        with self.assertRaisesRegex(RuntimeError, "trim failed"):
            self.ns["_trim_allocator_cache"](
                self.fake_torch(calls, snapshots, RuntimeError("trim failed")),
                "cuda:0", 1, logger,
                {"total_num_scheduled_tokens": 4096,
                 "num_scheduled_new_requests": 1},
                monotonic_ns=lambda: next(ticks),
                wall_time_ns=lambda: 1234,
                pid_fn=lambda: 4321,
                mem_available_fn=lambda: 200,
            )
        self.assertEqual(calls, [("stats", "cuda:0"), ("empty_cache",)])
        level, line = logger.records[-1]
        receipt = json.loads(line.split(" ", 1)[1])
        self.assertEqual((level, receipt["status"]), ("error", "error"))
        self.assertEqual(receipt["stage"], "empty_cache")
        self.assertEqual(receipt["total_num_scheduled_tokens"], 4096)
        self.assertNotIn("after", receipt)

    def test_hook_is_after_pp_wait_before_forward_and_armed_after_warmup(self):
        tree = ast.parse(WORKER.read_text())
        worker = next(node for node in tree.body
                      if isinstance(node, ast.ClassDef) and node.name == "Worker")
        execute = next(node for node in worker.body
                       if isinstance(node, ast.FunctionDef) and node.name == "execute_model")
        source = ast.get_source_segment(WORKER.read_text(), execute)
        self.assertLess(source.index("handle.wait()"),
                        source.index("_maybe_trim_prefill_cache("))
        self.assertLess(source.index("_maybe_trim_prefill_cache("),
                        source.index("self.model_runner.execute_model("))
        compile_method = next(node for node in worker.body if isinstance(node, ast.FunctionDef)
                              and node.name == "compile_or_warm_up_model")
        compile_source = ast.get_source_segment(WORKER.read_text(), compile_method)
        self.assertLess(compile_source.index("_prefill_cache_trim_runtime_ready = False"),
                        compile_source.index("warmup_sizes"))
        armed = compile_source.index("_prefill_cache_trim_runtime_ready = True")
        for runtime_call in ("self.model_runner.capture_model()", "warmup_kernels(",
                             "self.model_runner._dummy_run("):
            self.assertLess(compile_source.index(runtime_call), armed, runtime_call)
        self.assertEqual(compile_source.count("_prefill_cache_trim_runtime_ready = True"), 1)
        self.assertLess(armed,
                        compile_source.index("PREFILL_CACHE_TRIM_READY"))
        self.assertLess(compile_source.index("PREFILL_CACHE_TRIM_READY"),
                        compile_source.rindex("return CompilationTimes"))


if __name__ == "__main__":
    unittest.main()
