#!/usr/bin/env python3
"""Offline policy tests for the E29 end-drain candidate; no Torch, GPU, node or network.

Executes the real `_e29_end_drain_holds` from the scheduler override and the real
`_e29_remaining` and `_e29_coalesce` from the engine-core override. A small model of the
async engine (batch queue of two, output placeholders, draft rejection, length trimming)
checks the accounting invariant C = N + O + P - 1, that placeholders never go negative,
that every request finishes, and that the hold removes steps dispatched past a length
finish while the vendor guard alone leaves them.
"""

from __future__ import annotations

import ast
from collections import deque
import copy
import enum
import os
from pathlib import Path
import queue
import random
import unittest


REPO = Path(__file__).resolve().parents[2]
CANDIDATE = REPO / "scripts/node/experiments/e03/end-drain"


def load(path: Path, names: set[str], namespace: dict) -> dict:
    tree = ast.parse(path.read_text(), str(path))
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            found[node.name] = copy.deepcopy(node)
    assert set(found) == names, names - set(found)
    module = ast.Module(body=list(found.values()), type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


_holds = load(CANDIDATE / "scheduler.py", {"_e29_end_drain_holds"}, {})["_e29_end_drain_holds"]


def holds(output_tokens, placeholders, max_tokens, is_prefill_chunk, structured=False):
    return _holds(output_tokens, placeholders, max_tokens, is_prefill_chunk, structured)


class Rule(unittest.TestCase):
    def test_boundaries(self):
        self.assertFalse(holds(10, 0, 10, False))       # nothing in flight
        self.assertFalse(holds(0, 8, 8, True))          # prefill chunk
        self.assertFalse(holds(0, 8, None, False))      # no length limit
        self.assertTrue(holds(1, 8, 9, False))          # equality
        self.assertFalse(holds(1, 7, 9, False))         # one short
        self.assertTrue(holds(1, 9, 9, False))          # beyond
        self.assertTrue(holds(0, 1, 1, False))          # M=1 after prefill
        self.assertFalse(holds(0, 1, 2, False))         # M=2 needs one more step
        self.assertTrue(holds(1, 8, 8, False))          # eight-token replay, step 2 in flight
        self.assertFalse(holds(1, 8, 8, False, True))   # structured output keeps the vendor path
        for m in range(200, 257):
            self.assertTrue(holds(m - 8, 8, m, False))
            self.assertFalse(holds(m - 9, 8, m, False))

    def test_vendor_guard_implies_hold(self):
        # Vendor guard: C + 2 - P >= N + M with C = N + O + P - 1, i.e. O + 1 >= M.
        for m in (1, 2, 8, 256):
            for o in range(m):
                for p in range(1, 10):
                    if o + 1 >= m:
                        self.assertTrue(holds(o, p, m, False))


class Request:
    def __init__(self, rid, prompt, max_tokens, accept, arrive=0):
        self.rid, self.n, self.m, self.accept, self.arrive = rid, prompt, max_tokens, accept, arrive
        self.o = self.p = self.c = 0
        self.prefilled = self.finished = False
        self.k_next = 0
        self.wasted = 0
        self.resumes = 0
        self.held = False


def simulate(requests, end_drain, budget=lambda batch: 7 if batch == 1 else 3, limit=20000):
    """Model the async engine; return the requests after all finished."""
    pending = sorted(requests, key=lambda r: r.arrive)
    running: list[Request] = []
    batch_queue: deque = deque()
    tick = 0
    while pending or running or batch_queue:
        tick += 1
        assert tick < limit, "no progress"
        while pending and pending[0].arrive <= tick:
            running.append(pending.pop(0))
        if len(batch_queue) < 2 and running:
            step = []
            for r in running:
                if r.finished:
                    continue
                if r.p > 0 and r.c + 2 - r.p >= r.n + r.m:      # vendor guard
                    continue
                if end_drain and holds(r.o, r.p, r.m, not r.prefilled and r.c < r.n):
                    r.held = True
                    continue
                if r.held:
                    r.resumes += 1
                    r.held = False
                if not r.prefilled and r.c < r.n:
                    r.c += r.n
                    r.p += 1
                    step.append((r, 0, True))
                else:
                    k = r.k_next
                    r.c += 1 + k
                    r.p += 1 + k
                    step.append((r, k, False))
            for r, _, _ in step:
                r.k_next = budget(len(step))
            batch_queue.appendleft(step)
            if len(batch_queue) < 2:
                continue
        if not batch_queue:
            continue
        step = batch_queue.pop()
        for r, k, prefill in step:
            if r.finished:
                r.wasted += 1 + k
                continue
            if prefill:
                r.prefilled = True
                generated = 1
            else:
                accepted = min(k, next(r.accept))
                rejected = k - accepted
                r.c -= rejected
                r.p -= rejected
                generated = accepted + 1
            delivered = min(generated, r.m - r.o)
            r.p -= delivered
            r.o += delivered
            assert r.p >= 0, (r.rid, r.p)
            if r.o >= r.m:
                r.finished = True
            else:
                assert r.c == r.n + r.o + r.p - 1, (r.rid, r.c, r.n, r.o, r.p)
        running = [r for r in running if not r.finished]
    return requests


def script(kind, seed=0):
    rng = random.Random(seed)
    while True:
        if kind == "all":
            yield 99
        elif kind == "none":
            yield 0
        else:
            yield rng.choice((0, 0, 1, 2, 3, 4, 7))


class Accounting(unittest.TestCase):
    def test_single_requests(self):
        wasted_vendor = wasted_drain = 0
        for m in [1, 2, 8] + list(range(200, 257)):
            for kind, seed in (("all", 0), ("none", 0), ("mixed", m), ("mixed", m + 1000)):
                for drain in (False, True):
                    (r,) = simulate([Request("a", 133, m, script(kind, seed))], drain)
                    self.assertTrue(r.finished)
                    self.assertEqual(r.o, m)
                    if drain:
                        wasted_drain += r.wasted
                        self.assertEqual(r.wasted, 0, (m, kind, seed))
                    else:
                        wasted_vendor += r.wasted
        self.assertGreater(wasted_vendor, 0)
        self.assertEqual(wasted_drain, 0)

    def test_rejections_resume_the_held_request(self):
        (r,) = simulate([Request("a", 133, 256, script("none"))], True)
        self.assertTrue(r.finished)
        (r,) = simulate([Request("a", 133, 20, script("mixed", 7))], True)
        self.assertTrue(r.finished)
        self.assertEqual(r.wasted, 0)

    def test_mixed_batches_and_budget_changes(self):
        for seed in range(40):
            rng = random.Random(seed)
            reqs = [Request(f"r{i}", rng.randint(50, 3000), rng.choice([8, 64, 200, 256]),
                            script("mixed", seed * 10 + i), arrive=rng.randint(0, 60))
                    for i in range(rng.randint(2, 6))]
            done = simulate(reqs, True)
            for r in done:
                self.assertTrue(r.finished, r.rid)
                self.assertEqual(r.o, r.m)
                self.assertEqual(r.wasted, 0, (seed, r.rid))

    def test_all_rejected_matches_vendor_outcome(self):
        # With every draft rejected each step yields one token: same output, no waste either way.
        (a,) = simulate([Request("a", 133, 1000, script("none"))], False)
        (b,) = simulate([Request("a", 133, 1000, script("none"))], True)
        self.assertEqual((a.o, a.wasted), (b.o, b.wasted))


class State(enum.Enum):
    UNPAUSED = 0
    PAUSED_NEW = 1


class Clock:
    def __init__(self):
        self.now = 100.0

    def monotonic(self):
        return self.now


class FakeQueue:
    """Arrivals at absolute fake times; get(timeout) advances the clock."""

    def __init__(self, clock, arrivals):
        self.clock, self.arrivals = clock, sorted(arrivals)

    def get(self, timeout):
        if self.arrivals and self.arrivals[0][0] <= self.clock.now + timeout:
            at, item = self.arrivals.pop(0)
            self.clock.now = max(self.clock.now, at)
            return item
        self.clock.now += timeout
        raise queue.Empty


class Engine:
    def __init__(self, clock, arrivals, window, running=True, state=State.UNPAUSED):
        self.input_queue = FakeQueue(clock, arrivals)
        self.e29_coalesce_s, self.e29_trace = window, False
        self.handled, self.running = [], running
        self.unfinished = True
        engine = self
        self.scheduler = type("S", (), {"pause_state": state,
                                        "has_unfinished_requests": lambda s: engine.unfinished})()

    def is_running(self):
        return self.running

    def _handle_client_request(self, kind, request):
        self.handled.append(request)
        if kind == "ABORT":
            self.unfinished = False


class Coalesce(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        ns = load(CANDIDATE / "core.py", {"_e29_remaining", "_e29_coalesce"},
                  {"time": self.clock, "queue": queue, "PauseState": State,
                   "logger": type("L", (), {"info": lambda *a: None})()})
        self.coalesce = ns["_e29_coalesce"]
        self.remaining = ns["_e29_remaining"]

    def test_remaining(self):
        self.assertEqual(self.remaining(1.0, 2.0), 0.0)
        self.assertAlmostEqual(self.remaining(1.003, 1.001), 0.002)

    def test_fixed_deadline(self):
        first = self.clock.now
        arrivals = [(first + 0.001, ("ADD", "b")), (first + 0.0025, ("ADD", "c")),
                    (first + 0.0031, ("ADD", "d"))]
        engine = Engine(self.clock, arrivals, 0.003)
        self.coalesce(engine, first)
        self.assertEqual(engine.handled, ["b", "c"])
        self.assertLessEqual(self.clock.now - first, 0.003 + 1e-9)

    def test_deadline_counts_from_first_arrival(self):
        first = self.clock.now
        self.clock.now += 0.002   # time already spent draining
        engine = Engine(self.clock, [(first + 0.0035, ("ADD", "late"))], 0.003)
        self.coalesce(engine, first)
        self.assertEqual(engine.handled, [])

    def test_stops_when_the_last_request_is_aborted(self):
        first = self.clock.now
        arrivals = [(first + 0.001, ("ABORT", "a")), (first + 0.002, ("ADD", "b"))]
        engine = Engine(self.clock, arrivals, 0.005)
        self.coalesce(engine, first)
        self.assertEqual(engine.handled, ["a"])
        self.assertAlmostEqual(self.clock.now - first, 0.001)

    def test_stops_on_shutdown_or_pause(self):
        first = self.clock.now
        arrivals = [(first + 0.001, ("ADD", "b"))]
        self.coalesce(Engine(self.clock, list(arrivals), 0.003, running=False), first)
        engine = Engine(self.clock, list(arrivals), 0.003, state=State.PAUSED_NEW)
        self.coalesce(engine, first)
        self.assertEqual(engine.handled, [])


class Wiring(unittest.TestCase):
    def test_core_hook(self):
        text = (CANDIDATE / "core.py").read_text()
        self.assertIn("and self.e29_coalesce_s\n                    and req[0] == EngineCoreRequestType.ADD", text)
        added = "\n".join(line for line in (CANDIDATE / "core.patch").read_text().splitlines()
                          if line.startswith("+") and not line.startswith("+++"))
        self.assertEqual(added.count("self.scheduler.has_unfinished_requests()"), 2)
        self.assertNotIn("self.scheduler.has_requests()", added)
        self.assertIn("self._e29_coalesce(e29_first)", text)
        self.assertIn("and self.process_input_queue_block", text)
        self.assertIn('raise ValueError("VLLM_E29_IDLE_COALESCE_MS requires data_parallel_size 1")',
                      text)

    def test_scheduler_hook(self):
        text = (CANDIDATE / "scheduler.py").read_text()
        self.assertEqual(text.count("_e29_end_drain_holds("), 2)   # definition and use
        self.assertIn("request.num_output_placeholders,\n            request.max_tokens,\n"
                      "            request.is_prefill_chunk,\n            request.use_structured_output,", text)
        self.assertEqual(text.count("self._e29_would_hold(request)"), 2)   # loop and eligibility
        resume = text.index("# E29: count resumes from the final scheduled set.")
        self.assertLess(text.index("def _update_after_schedule"), resume)
        preempt = text.index("def _preempt_request")
        self.assertLess(preempt, text.index("# E29: preemption resets placeholders"))
        self.assertIn("self.e29_held_ids.discard(request.request_id)", text)


if __name__ == "__main__":
    os.environ.pop("VLLM_E29_END_DRAIN", None)
    unittest.main(verbosity=1)
