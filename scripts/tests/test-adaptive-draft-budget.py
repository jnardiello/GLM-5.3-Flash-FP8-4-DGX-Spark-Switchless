#!/usr/bin/env python3
"""CPU integration of the candidate with pinned R10 AsyncScheduler/SchedulerOutput.

The fixtures are exact image sources. Only the heavyweight parent Scheduler,
request storage and unused imports are doubles; the async hook and budget resolver
are real. --engine-source additionally verifies a private capture from the image,
including the scheduler producer and V2 worker consumer. No nodes or GPU are used.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, field
import hashlib
import importlib.machinery
import importlib.util
import json
import logging
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
CANDIDATE = REPO / "scripts/node/experiments/e03/draft-budget"
FIXTURES = REPO / "scripts/tests/fixtures/r10-draft-budget"
BASE = REPO / "scripts/node/patches/adaptive_k_scheduler.py"
PINS = json.loads((CANDIDATE / "manifest.json").read_text())


def load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module


class ParentScheduler:
    """Storage/output double; no alternative implementation of async sizing."""

    def __init__(self, engine_k=5, async_on=True):
        self.num_spec_tokens = engine_k
        self.num_sampled_tokens_per_step = 1
        self.scheduler_config = SimpleNamespace(async_scheduling=async_on)
        self.parallel_config = SimpleNamespace(pipeline_parallel_size=1)
        self.vllm_config = SimpleNamespace(speculative_config=SimpleNamespace(
            num_speculative_tokens_per_batch_size=[[1, 1, 5], [2, 6, 3]]))
        self.requests = {}
        self.use_v2_model_runner = True
        self.current_step = 0
        self.base_updates = 0

    def _update_after_schedule(self, output):
        self.base_updates += 1

    def update_from_output(self, output, runner):
        self.observations_before_base = self._ak.counters["observations"]
        # The real parent can truncate generated lists and free stopped requests.
        for tokens in runner.sampled_token_ids:
            tokens.clear()
        for req_id in getattr(runner, "finish_ids", ()):
            self._free_request(self.requests[req_id])
        return "base-output"

    def _free_request(self, request, delay_free_blocks=False):
        self.requests.pop(request.request_id, None)
        return "base-free"


def import_runtime():
    modules = {}
    for name in ("vllm", "vllm.logger", "vllm.config", "vllm.config.ec_manager_config",
                 "vllm.multimodal", "vllm.multimodal.utils", "vllm.v1", "vllm.v1.core",
                 "vllm.v1.core.sched", "vllm.v1.core.sched.scheduler", "vllm.v1.request"):
        modules[name] = ModuleType(name)
        modules[name].__path__ = []
    modules["vllm.logger"].init_logger = logging.getLogger
    modules["vllm.config.ec_manager_config"].EncoderCacheManagerMetadata = object
    modules["vllm.multimodal.utils"].strip_covered_mm_data = lambda value: value
    modules["vllm.v1.core.sched.scheduler"].Scheduler = ParentScheduler
    modules["vllm.v1.request"].Request = object
    modules["vllm.v1.request"].RequestStatus = SimpleNamespace(RUNNING="running")
    with patch.dict(sys.modules, modules):
        output = load("vllm.v1.core.sched.output", FIXTURES / "output.py.fixture")
        load("vllm.v1.core.sched.async_scheduler", FIXTURES / "async_scheduler.py.fixture")
        baseline = load("frozen_adaptive_k", BASE)
        candidate = load("budget_adaptive_k", CANDIDATE / "adaptive_k_scheduler.py")
    return output.SchedulerOutput, baseline, candidate


Output, baseline, candidate = import_runtime()


@dataclass
class Request:
    request_id: str
    spec_token_ids: list[int] = field(default_factory=list)
    is_prefill_chunk: bool = False
    finished: bool = False
    use_structured_output: bool = False
    num_output_placeholders: int = 0
    num_stale_output_tokens: int = 0
    drop_stale_output: bool = False

    def is_finished(self):
        return self.finished


def scheduler(module=candidate, cap=True, **kwargs):
    env = {"VLLM_ADAPTIVE_K_MODE": "batch-uniform",
           "VLLM_ADAPTIVE_K_RESPECT_DRAFT_BUDGET": str(int(cap))}
    with patch.dict(os.environ, env, clear=True):
        return module.AdaptiveKScheduler(**kwargs)


def output(req_ids, budget, drafts=None):
    result = Output.make_empty()
    result.num_scheduled_tokens = {r: 1 for r in req_ids}
    result.total_num_scheduled_tokens = len(req_ids)
    result.num_spec_tokens_to_schedule = budget
    result.scheduled_spec_decode_tokens = drafts or {}
    return result


def handout(sched, ids, budget, drafts=None):
    for r in ids:
        sched.requests.setdefault(r, Request(r))
    result = output(ids, budget, drafts)
    sched._update_after_schedule(result)
    assert not sched._ak_failed
    return result, {r: len(sched.requests[r].spec_token_ids) for r in ids}


def runner(ids, accepted=3, **extra):
    return SimpleNamespace(req_id_to_index={r: i for i, r in enumerate(ids)},
                           sampled_token_ids=[list(range(accepted + 1)) for _ in ids],
                           kv_connector_output=None, **extra)


class DraftBudgetTests(unittest.TestCase):
    def test_pins_and_unchanged_policy_observation_methods(self):
        self.assertEqual(hashlib.sha256(BASE.read_bytes()).hexdigest(), PINS["base_scheduler_sha256"])
        for name in ("async_scheduler.py", "output.py"):
            self.assertEqual(hashlib.sha256((FIXTURES / (name + ".fixture")).read_bytes()).hexdigest(),
                             PINS["engine_sources_sha256"]["vllm/v1/core/sched/" + name])
        def functions(path):
            tree = ast.parse(path.read_text())
            return {n.name: ast.dump(n) for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        old, new = functions(BASE), functions(CANDIDATE / "adaptive_k_scheduler.py")
        for name in ("should_observe", "filter_candidates", "assign_lengths", "placeholder_len",
                     "_get", "_k_for", "signal", "observe", "decide", "decide_batch", "evict",
                     "update_from_output", "_free_request", "_update_after_schedule"):
            self.assertEqual(old[name], new[name], name)

    def test_actual_hook_reproduces_mismatch_and_caps_all_affected_batches(self):
        for size in range(2, 7):
            ids = [str(i) for i in range(size)]
            old, new = scheduler(baseline), scheduler()
            _, old_lengths = handout(old, ids, 3)
            _, new_lengths = handout(new, ids, 3)
            self.assertEqual(set(old_lengths.values()), {5})
            self.assertEqual(set(new_lengths.values()), {3})
            self.assertEqual(new.base_updates, 1)
            self.assertEqual(new._ak_budget_counters, dict(steps=1, limited_steps=1,
                             limited_requests=size, trimmed_tokens=2 * size))
            self.assertEqual(new._ak.counters, old._ak.counters)

    def test_flag_off_is_original_sizing_and_default_is_off(self):
        self.assertFalse(candidate.AdaptiveKConfig.from_env({}).respect_draft_budget)
        sched = scheduler(cap=False)
        _, lengths = handout(sched, ["a", "b", "c", "d"], 3)
        self.assertEqual(set(lengths.values()), {5})
        self.assertEqual(sum(sched._ak_budget_counters.values()), 0)

    def test_c1_and_adaptive_low_state_are_unchanged(self):
        old, new = scheduler(baseline), scheduler()
        for accepted in ([5] * 3 + [0] * 10 + [5] * 12):
            for sched in (old, new):
                _, lengths = handout(sched, ["a"], 5)
                sched._ak.observe("a", accepted, lengths["a"])
            self.assertEqual(old.requests["a"].spec_token_ids, new.requests["a"].spec_token_ids)
            self.assertEqual(old._ak.ema("a"), new._ak.ema("a"))
            self.assertEqual(old._ak.counters, new._ak.counters)
        self.assertEqual(new._ak_budget_counters["limited_requests"], 0)

    def test_uniform_low_and_engine_and_resolved_budget_limits(self):
        sched = scheduler()
        for _ in range(8):
            sched._ak.observe("low", 0, 5)
        _, lengths = handout(sched, ["low", "high"], 5)
        self.assertEqual(lengths, {"low": 3, "high": 3})
        self.assertEqual(sched._ak_budget_counters["limited_requests"], 0)
        for budget, expected in ((None, 5), (99, 5), (2, 2), (0, 0)):
            _, lengths = handout(scheduler(), ["a"], budget)
            self.assertEqual(lengths, {"a": expected})
        _, lengths = handout(scheduler(engine_k=2), ["a"], 5)
        self.assertEqual(lengths, {"a": 2})

    def test_same_output_budget_not_candidate_or_future_batch_size(self):
        sched = scheduler()
        # One decode candidate, but the producer's batch includes prefill requests.
        sched.requests["decode"] = Request("decode")
        for r in ("p1", "p2", "p3"):
            sched.requests[r] = Request(r, [99], is_prefill_chunk=True)
        _, lengths = handout(sched, list(sched.requests), 3)
        self.assertEqual(lengths["decode"], 3)
        for r in ("p1", "p2", "p3"):
            self.assertEqual(sched.requests[r].spec_token_ids, [99])
        # Explicit output budget wins even when it differs from a inferred lookup.
        _, lengths = handout(sched, ["decode"], 3)
        self.assertEqual(lengths, {"decode": 3})
        _, lengths = handout(sched, ["a", "b", "c", "d"], 5)
        self.assertEqual(set(lengths.values()), {5})

    def test_async_1_to_4_to_1_uses_producer_budget_and_prior_drafts(self):
        sched = scheduler()
        first, lens1 = handout(sched, ["a"], 5)
        # The next schedule verifies five drafts produced under C1, but hands
        # out three placeholders for the drafts THIS C4 output will produce.
        second, lens4 = handout(sched, ["a", "b", "c", "d"], 3, {"a": [-1] * 5})
        self.assertEqual(lens1, {"a": 5})
        self.assertEqual(set(lens4.values()), {3})
        self.assertEqual(len(second.scheduled_spec_decode_tokens["a"]), 5)
        sched.update_from_output(first, runner(["a"]))
        # schedule(N) precedes delivery(N-1). Finished requests still present
        # during handout must retain the base's placeholders and be unobserved.
        sched.requests["b"].finished = True
        third, lens1again = handout(sched, ["a"], 5, {"a": [-1] * 3})
        self.assertEqual(lens1again, {"a": 5})
        self.assertEqual(len(third.scheduled_spec_decode_tokens["a"]), 3)
        delivered = runner(["a"], finish_ids=["b", "c", "d"])
        self.assertEqual(sched.update_from_output(second, delivered), "base-output")
        self.assertEqual(sched.observations_before_base, 1)
        self.assertEqual(sched._ak.ema("a"), 1)
        self.assertEqual(list(sched.requests), ["a"])
        self.assertEqual(sched._ak.tracked(), 1)
        self.assertTrue(all(not tokens for tokens in delivered.sampled_token_ids))
        self.assertEqual(sched._ak_budget_counters, dict(steps=3, limited_steps=1,
                         limited_requests=4, trimmed_tokens=8))
        self.assertEqual(sched.requests["a"].num_output_placeholders, 11)
        self.assertEqual(len(sched._ak_ring), 3)

    def test_finished_prefill_unscheduled_and_no_placeholder_filters(self):
        sched = scheduler()
        sched.requests = {
            "live": Request("live"), "done": Request("done", finished=True),
            "prefill": Request("prefill", [44], is_prefill_chunk=True),
            "waiting": Request("waiting", [55]),
        }
        for r in sched.requests:
            for _ in range(8):
                sched._ak.observe(r, 0, 5)
        handout(sched, ["live", "done", "prefill"], 5)
        self.assertEqual(sched.requests["live"].spec_token_ids, [-1] * 3)
        self.assertEqual(sched.requests["done"].spec_token_ids, [-1] * 5)
        self.assertEqual(sched.requests["prefill"].spec_token_ids, [44])
        self.assertEqual(sched.requests["waiting"].spec_token_ids, [55])
        self.assertEqual(sched._ak_ring.union(), {"live"})
        handout(sched, ["live"], 0)
        self.assertEqual(sched.requests["live"].spec_token_ids, [])
        self.assertEqual(sched._ak_budget_counters["steps"], 1)

    def test_observation_guards_and_three_handout_lookahead(self):
        for guard in ("stale", "kv", "finished", "empty", "not-drafted"):
            sched = scheduler()
            handout(sched, ["a"], 3)
            result = output(["a"], 5, {"a": [-1] * 3})
            model = runner(["a"])
            if guard == "stale": sched.requests["a"].num_stale_output_tokens = 1
            if guard == "kv": model.kv_connector_output = SimpleNamespace(invalid_block_ids={1})
            if guard == "finished": sched.requests["a"].finished = True
            if guard == "empty": model.sampled_token_ids = [[]]
            if guard == "not-drafted":
                for _ in range(3): handout(sched, [], 5)
            sched.update_from_output(result, model)
            self.assertEqual(sched._ak.counters["observations"], 0, guard)
        sched = scheduler()
        handout(sched, ["old"], 3)
        handout(sched, ["middle"], 5)
        handout(sched, ["new"], 3)
        sched.update_from_output(output(["old"], 5, {"old": [-1] * 3}), runner(["old"]))
        self.assertEqual(sched._ak.counters["observations"], 1)

    def test_boot_first_limit_and_periodic_aggregate_signatures(self):
        with self.assertLogs("vllm.adaptive_k", level="INFO") as logs:
            sched = scheduler()
            for _ in range(2): handout(sched, ["a", "b"], 3)
            sched._ak_steps = 199
            sched.update_from_output(output(["a"], 3, {"a": [-1] * 3}), runner(["a"]))
        text = "\n".join(logs.output)
        self.assertIn("draft-budget active=1 source=SchedulerOutput.resolve_num_spec_tokens_to_schedule engine_k=5", text)
        self.assertEqual(text.count("draft-budget first-cap"), 1)
        self.assertIn("limited_steps=2 limited_requests=4 trimmed_tokens=8", text)

    def test_sync_disabled_and_original_error_fallback(self):
        with self.assertLogs("vllm.adaptive_k", level="ERROR"):
            self.assertFalse(scheduler(async_on=False)._ak_cfg.enabled)
        sched = scheduler()
        sched.requests["a"] = Request("a")
        with patch.object(sched._ak, "decide_batch", side_effect=RuntimeError("test")):
            with self.assertLogs("vllm.adaptive_k", level="ERROR"):
                sched._update_after_schedule(output(["a"], 3))
        self.assertTrue(sched._ak_failed)
        self.assertEqual(sched.requests["a"].spec_token_ids, [-1] * 3)


def verify_engine_capture(path):
    for name, digest in PINS["engine_sources_sha256"].items():
        assert hashlib.sha256((path / name).read_bytes()).hexdigest() == digest, name
    worker = ast.parse((path / "vllm/v1/worker/gpu/model_runner.py").read_text())
    calls = [n for n in ast.walk(worker) if isinstance(n, ast.Call)]
    assert any(isinstance(n.func, ast.Attribute) and n.func.attr == "resolve_num_spec_tokens_to_schedule"
               and ast.unparse(n.func.value) == "scheduler_output"
               and ast.unparse(n.args[0]) == "self.num_speculative_steps" for n in calls)
    assert any(isinstance(n.func, ast.Attribute) and n.func.attr == "propose"
               and any(k.arg == "num_speculative_tokens" and ast.unparse(k.value) == "num_spec_tokens_to_schedule"
                       for k in n.keywords) for n in calls)
    assert any(isinstance(n.func, ast.Name) and n.func.id == "limit_draft_tokens"
               and ast.unparse(n.args[1]) == "num_spec_tokens_to_schedule" for n in calls)
    print("Pinned R10 producer/consumer source capture: PASS")


def load_tests(loader, tests, pattern):
    # Reuse all frozen policy tests against the candidate, including calibration,
    # without editing the baseline tests or changing their module on disk.
    with patch.dict(sys.modules, {"adaptive_k_scheduler": candidate}):
        regression = load("budget_policy_regression", BASE.parent / "test_adaptive_k_policy.py")
    tests.addTests(loader.loadTestsFromModule(regression))
    return tests


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-source", type=Path)
    args, rest = parser.parse_known_args()
    if args.engine_source:
        verify_engine_capture(args.engine_source)
    unittest.main(argv=[sys.argv[0], *rest])
