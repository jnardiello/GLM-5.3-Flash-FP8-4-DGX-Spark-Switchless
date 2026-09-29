#!/usr/bin/env python3
"""Offline proof of the E31 speculative-safe C4 tail ring; stdlib only, no Torch or GPU.

The GLM pooled indexer keeps each request's last key/gate rows in a tail and builds a
four-token pool whenever position % 4 == 3. A DFlash verify step writes 1 + K rows,
including drafts that may be rejected. With the legacy four-slot tail, rejected rows
overwrite committed members of the still-open pool. This test re-implements the index
arithmetic of the three kernels in the E31 glm_kpool.py with token IDs as keys. It checks
that arithmetic against the kernel source, then simulates chunked prefill followed by
verify steps with random acceptance. Every committed pool must hold its four committed
tokens. The legacy ring (4) must fail, and the E31 ring 4 * cdiv(4 + K, 4) must never fail.
"""

from __future__ import annotations

import ast
import importlib.util
import math
from pathlib import Path
import random
from types import SimpleNamespace
import unittest


REPO = Path(__file__).resolve().parents[2]
CANDIDATE = REPO / "scripts/node/experiments/e03/e31-indexer"
POOL_SIZE = 4


def ring_for(spec_tokens: int) -> int:
    """The E31 pooled_indexer allocation: 4 * cdiv(4 + K, 4)."""
    ring_pools = math.ceil((POOL_SIZE + spec_tokens) / POOL_SIZE)
    return POOL_SIZE * ring_pools


def decode_update(tail: list, cache: dict, rows: list, tail_ring: int) -> None:
    """_decode_update_kernel for one request, packed main slots: rows in order."""
    for position, key in rows:
        physical_slot = position % POOL_SIZE
        pool_start = position - physical_slot
        if position % POOL_SIZE == POOL_SIZE - 1:
            members = [tail[(pool_start + slot) % tail_ring] for slot in range(POOL_SIZE - 1)]
            cache[position // POOL_SIZE] = (*members, key)
        tail[position % tail_ring] = key


def prefill_pool(tail: list, cache: dict, rows: list, tail_ring: int) -> None:
    """_prefill_pool_kernel for one request: one program per pool completed in the chunk."""
    start, end = 0, len(rows)
    first_physical_slot = rows[start][0] % POOL_SIZE
    first_completion = start + (POOL_SIZE - 1 - first_physical_slot)
    for pool_ordinal in range(math.ceil(end / POOL_SIZE)):
        completion_row = first_completion + pool_ordinal * POOL_SIZE
        if completion_row >= end:
            continue
        completion_position = rows[completion_row][0]
        if completion_position % POOL_SIZE != POOL_SIZE - 1:
            continue
        members = []
        for slot in range(POOL_SIZE):
            source_row = completion_row - (POOL_SIZE - 1 - slot)
            if source_row >= start:
                members.append(rows[source_row][1])
            else:
                ring_slot = (completion_position - (POOL_SIZE - 1) + slot) % tail_ring
                members.append(tail[ring_slot])
        cache[completion_position // POOL_SIZE] = tuple(members)


def prefill_tail(tail: list, rows: list, tail_ring: int) -> None:
    """_prefill_tail_kernel for one request: the chunk's last rows, one per phase."""
    start, last_row = 0, len(rows) - 1
    last_physical_slot = rows[last_row][0] % POOL_SIZE
    for slot in range(POOL_SIZE):
        distance = (last_physical_slot - slot + POOL_SIZE) % POOL_SIZE
        source_row = last_row - distance
        if source_row >= start and rows[source_row][0] % POOL_SIZE == slot:
            source_position = rows[source_row][0]
            tail[source_position % tail_ring] = rows[source_row][1]


def simulate(seed: int, spec_tokens: int, tail_ring: int, steps: int = 200,
             prompt: int | None = None, schedule: list | None = None) -> int:
    """Return the number of committed pools whose members differ from the committed tokens."""
    rng = random.Random(seed)
    tail = [("stale", i) for i in range(ring_for(spec_tokens))]
    cache: dict = {}
    true: dict = {}
    token = lambda p: true.setdefault(p, ("token", p, rng.randrange(1 << 30)))  # noqa: E731
    prompt = rng.randrange(1, 160) if prompt is None else prompt
    position = 0
    while position < prompt:
        size = min(prompt - position, rng.randrange(1, 40))
        rows = [(p, token(p)) for p in range(position, position + size)]
        prefill_pool(tail, cache, rows, tail_ring)
        prefill_tail(tail, rows, tail_ring)
        position += size
    wrong: set = set()

    def check(pools: range) -> None:
        for pool in pools:
            if cache.get(pool) != tuple(true[POOL_SIZE * pool + j] for j in range(POOL_SIZE)):
                wrong.add(pool)

    committed = prompt   # the first verify row is the sampled token at this position
    checked = 0
    plan = schedule or [None] * steps
    for step in plan:
        if step is None:
            drafts = spec_tokens if rng.random() < 0.7 else rng.randrange(0, spec_tokens + 1)
            accepted = rng.randrange(0, drafts + 1)
        else:
            drafts, accepted = step
        rows = [(committed, token(committed))]
        for i in range(1, drafts + 1):
            p = committed + i
            rows.append((p, token(p) if i <= accepted else ("draft", p, rng.randrange(1 << 30))))
        decode_update(tail, cache, rows, tail_ring)
        committed += accepted + 1
        visible = committed // POOL_SIZE
        check(range(checked, visible))   # pools as they become visible
        checked = visible
    check(range(checked))                # and none changed afterwards
    return len(wrong)


def simulate_batches(batches: list, committed: list, spec_tokens: int, tail_ring: int,
                     seed: int = 0) -> int:
    """Run the GPU leaf's batch schedule through the simulated kernels: one tail per state
    slot, decode requests first, rejected drafts as fresh keys. Return wrong committed pools."""
    rng = random.Random(seed)
    slots = rng.sample(range(1, 4 * len(committed) + 3), len(committed))
    tails = {s: [("stale", s, i) for i in range(ring_for(spec_tokens))] for s in slots}
    cache: dict = {}
    true = lambda r, p: ("token", r, p)  # noqa: E731
    for batch in batches:
        drafts_seen = False
        for r, first, size, drafts, accepted in batch:
            if drafts is None:
                drafts_seen = True
                rows = [(p, true(r, p)) for p in range(first, first + size)]
                pools: dict = {}
                prefill_pool(tails[slots[r]], pools, rows, tail_ring)
                prefill_tail(tails[slots[r]], rows, tail_ring)
            else:
                assert not drafts_seen, "decode requests must precede prefill requests"
                rows = [(p, true(r, p) if p - first <= accepted else ("draft", r, p, rng.random()))
                        for p in range(first, first + size)]
                pools = {}
                decode_update(tails[slots[r]], pools, rows, tail_ring)
            cache.update({(r, pool): members for pool, members in pools.items()})
    return sum(cache.get((r, pool)) != tuple(true(r, POOL_SIZE * pool + j) for j in range(POOL_SIZE))
               for r, length in enumerate(committed) for pool in range(length // POOL_SIZE))


def load_leaf():
    spec = importlib.util.spec_from_file_location(
        "e31_leaf", CANDIDATE / "leaf_kpool_tail_ring.py")
    leaf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(leaf)   # stdlib imports only; Torch is imported inside main()
    return leaf


def function_source(path: Path, name: str) -> str:
    text = path.read_text()
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(text, node)


class SourceCoupling(unittest.TestCase):
    """The simulation's index expressions must be the kernel's, verbatim."""

    def test_kernels(self):
        kpool = CANDIDATE / "glm_kpool.py"
        expected = {
            "_decode_update_kernel": [
                "physical_slot = position % POOL_SIZE",
                "pool_start = position - physical_slot",
                "+ ((pool_start + slot) % tail_ring) * tail_stride_2",
                "ring_slot = (pool_start + slot) % tail_ring",
                "base = state_slot * tail_stride_0 + ring_slot * tail_stride_2 + dim",
                "base = state_slot * tail_stride_0 + (position % tail_ring) * tail_stride_2",
                "(position % POOL_SIZE == POOL_SIZE - 1)",
                "for row in tl.range(start, end):",
            ],
            "_prefill_pool_kernel": [
                "first_completion = start + (POOL_SIZE - 1 - first_physical_slot)",
                "completion_row = first_completion + pool_ordinal * POOL_SIZE",
                "write_pool &= completion_position % POOL_SIZE == POOL_SIZE - 1",
                "source_row = completion_row - (POOL_SIZE - 1 - slot)",
                "from_current_chunk = source_row >= start",
                "ring_slot = (completion_position - (POOL_SIZE - 1) + slot) % tail_ring",
                "ring_slot * tail_stride_2",
            ],
            "_prefill_tail_kernel": [
                "distance = (last_physical_slot - slot + POOL_SIZE) % POOL_SIZE",
                "source_row = last_row - distance",
                "(source_position % POOL_SIZE == slot)",
                "ring_slot = source_position % tail_ring",
                "tail_offset = state_slot * tail_stride_0 + ring_slot * tail_stride_2 + dim",
            ],
        }
        for name, needles in expected.items():
            source = function_source(kpool, name)
            self.assertIn("    tail_ring,\n", source, name)
            for needle in needles:
                self.assertIn(needle, source, f"{name}: {needle}")
        # Every tail access goes through a ring index; no legacy phase-slot offset remains.
        text = kpool.read_text()
        self.assertNotIn("+ slot * tail_stride_2", text)
        self.assertNotIn("physical_slot * tail_stride_2", text)
        self.assertEqual(text.count("ring_slot = (completion_position - (POOL_SIZE - 1) + slot)"
                                    " % tail_ring"), 2)
        wrapper = function_source(kpool, "update_decode_pools")
        self.assertIn("tail_ring: int = _POOL_SIZE,", wrapper)
        self.assertEqual(wrapper.count("            tail_ring,\n"), 3)   # all three launches

    def test_indexer_allocation(self):
        text = (CANDIDATE / "pooled_indexer.py").read_text()
        for needle in ("ring_pools = math.ceil((_POOL_SIZE + spec_tokens) / _POOL_SIZE)",
                       "self._tail_ring = _POOL_SIZE * ring_pools",
                       "(self.max_seqs, 2, self._tail_ring, _INDEX_HEAD_DIM)",
                       "tail_ring = self._tail_ring if _e31_on(_E31_TAIL_RING) else _POOL_SIZE",
                       "tail_ring=tail_ring,"):
            self.assertIn(needle, text)


class TailRing(unittest.TestCase):
    def test_ring_size(self):
        self.assertEqual([ring_for(k) for k in (0, 3, 7)], [4, 8, 12])
        for k in range(0, 33):
            self.assertGreaterEqual(ring_for(k), k + 4, k)

    def test_reported_counterexample(self):
        # Committed 0 and 1; one verify step over positions 2..9 accepts only row 2. The
        # next step completes pool 0 at position 3: the legacy tail then holds 8, 9, 6.
        plan = [(7, 0), (0, 0)]
        self.assertEqual(simulate(0, 7, 4, prompt=2, schedule=plan), 1)
        self.assertEqual(simulate(0, 7, 12, prompt=2, schedule=plan), 0)

    def test_legacy_ring_corrupts_pools(self):
        self.assertGreater(simulate(1, 7, 4), 0)
        self.assertGreater(simulate(1, 3, 4), 0)

    def test_e31_ring_is_exact(self):
        for spec_tokens in (7, 3, 1):
            ring = ring_for(spec_tokens)
            for seed in range(40):
                self.assertEqual(simulate(seed, spec_tokens, ring), 0, (spec_tokens, seed))

    def test_simulation_is_sharp(self):
        # A ring of K + 2 slots already corrupts pools; E31 keeps at least K + 4.
        self.assertGreater(sum(simulate(seed, 7, 9) for seed in range(10)), 0)
        self.assertGreater(sum(simulate(seed, 3, 5) for seed in range(10)), 0)

    def test_without_speculation_legacy_is_exact(self):
        for seed in range(20):
            self.assertEqual(simulate(seed, 0, 4), 0, seed)


class LeafSchedule(unittest.TestCase):
    """The GPU leaf's schedules and geometry, run through the proven simulation."""

    def setUp(self):
        self.leaf = load_leaf()

    def args(self, seed):
        return SimpleNamespace(spec_tokens=7, requests=4, prompt_max=600, max_chunk=128,
                               steps=60, seed=seed)

    def test_random_batches_are_mixed_and_contiguous(self):
        batches, committed = self.leaf.random_schedule(self.args(0))
        position = [0] * 4
        mixed = 0
        for batch in batches:
            kinds = [b[3] is not None for b in batch]
            self.assertEqual(kinds, sorted(kinds, reverse=True))   # decodes first
            mixed += len(set(kinds)) == 2
            for r, first, size, drafts, accepted in batch:
                self.assertEqual(first, position[r])
                self.assertTrue(drafts is None or size == drafts + 1)
                position[r] += size if drafts is None else accepted + 1
        self.assertEqual(position, committed)
        self.assertGreater(mixed, 0)
        self.assertGreater(max(len(b) for b in batches), 1)

    def test_leaf_schedules_separate_the_rings(self):
        batches, committed = self.leaf.counterexample(7)
        self.assertEqual(simulate_batches(batches, committed, 7, 4), 1)
        self.assertEqual(simulate_batches(batches, committed, 7, 12), 0)
        for seed in range(5):
            batches, committed = self.leaf.random_schedule(self.args(seed))
            self.assertGreater(simulate_batches(batches, committed, 7, 4, seed), 0, seed)
            self.assertEqual(simulate_batches(batches, committed, 7, 12, seed), 0, seed)

    def test_geometry_mirrors_production(self):
        indexer = (REPO / "scripts/node/overrides/vllm/models/glm5next/nvidia/pooled_indexer.py"
                   ).read_text()
        for needle in ("_INDEX_CACHE_WIDTH = 132", "_INDEX_PAGE_SIZE = 64", "_MLA_RECORD_BYTES = 528",
                       "subpages_per_parent = block_size // (_POOL_SIZE * _INDEX_PAGE_SIZE)",
                       "parent_stride_pages = parent_stride_bytes // _INDEX_PAGE_BYTES",
                       "tail_offset = int(main_cache.storage_offset()) + semantic_page_bytes",
                       "stride=(_INDEX_PAGE_BYTES, _INDEX_CACHE_WIDTH, 1),"):
            self.assertIn(needle, indexer)
        self.assertEqual((self.leaf.WIDTH, self.leaf.PAGE, self.leaf.RECORD), (132, 64, 528))
        self.assertEqual(self.leaf.geometry(2304), (9, 2304 * 528, 153 * 8448, 153))
        kpool = (CANDIDATE / "glm_kpool.py").read_text()
        for name in ("_decode_update_kernel", "_prefill_pool_kernel"):
            source = function_source(CANDIDATE / "glm_kpool.py", name)
            for needle in ("pool_offset = model_page_offset // POOL_SIZE",
                           "child_page = pool_offset // PAGE_SIZE",
                           "child_offset = pool_offset - child_page * PAGE_SIZE",
                           "parent_page * parent_stride_pages + child_page\n",
                           ") * PAGE_SIZE + child_offset"):
                self.assertIn(needle, source, name)
        self.assertIn("scale_byte_offset = page_bytes + PAGE_SIZE * HEAD_DIM + offset * 4", kpool)
        rng = random.Random(3)
        for _ in range(2000):
            parent, position = rng.randrange(1, 40), rng.randrange(0, 10 ** 6)
            slot = parent * 2304 + position % 2304               # the leaf's slot mapping
            parent_page = slot // 2304                           # kernel transliteration
            pool_offset = (slot - parent_page * 2304) // POOL_SIZE
            location = (parent_page * 153 + pool_offset // 64) * 64 + pool_offset % 64
            page, entry = self.leaf.pool_location(parent, position, 2304, 153)
            self.assertEqual((page, entry), (location // 64, location % 64))


if __name__ == "__main__":
    unittest.main(verbosity=1)
