#!/usr/bin/env python3
"""E31 leaf test: C4 pools built through speculative verify steps, legacy tail vs E31 ring.

Runs the E31 glm_kpool.update_decode_pools Triton kernels on one idle GPU inside the
serving image (vLLM is needed only for its Triton import), with production page geometry
(2304-token parent pages carrying their FP8 index tail).

Several requests share mixed batches: decode requests first, then prefill requests. They use
non-contiguous state slots and shuffled parent pages. Each request is prefilled in chunks,
then takes DFlash-style verify steps of 1 + k rows; rejected drafts carry random keys.

Every committed pool is compared with a reference built by one prefill over the committed
tokens. Sampled pools are also compared with an independent Torch computation of the pool
formula (softmax(gate + ape)-weighted keys, FWHT, FP8 scale).

The run passes only if all of these hold:
- the reference is finite, non-zero and distinct, and matches the independent pools;
- the legacy ring (4) corrupts pools, on the reported counterexample and on the random
  schedule;
- the E31 ring 4 * cdiv(4 + K, 4) corrupts none.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import random
import sys

POOL, PAGE, DIM, WIDTH, RECORD = 4, 64, 128, 132, 528
PAGE_BYTES = PAGE * WIDTH


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--spec-tokens", type=int, default=7, help="K drafts per verify step (>= 2)")
    p.add_argument("--requests", type=int, default=4)
    p.add_argument("--prompt-max", type=int, default=3000, help="crosses a 2304-token page")
    p.add_argument("--max-chunk", type=int, default=512)
    p.add_argument("--steps", type=int, default=150, help="verify steps per request")
    p.add_argument("--block", type=int, default=2304, help="model block size (multiple of 256)")
    p.add_argument("--spot", type=int, default=24, help="pools per request checked independently")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tol", type=float, default=0.05,
                   help="wrong pool: max |diff| above this fraction of the pool's max |ref|")
    args = p.parse_args()
    if args.spec_tokens < 2 or args.block % (POOL * PAGE) or args.requests < 1:
        p.error("need --spec-tokens >= 2, --block divisible by 256 and --requests >= 1")
    return args


def random_schedule(args) -> tuple[list, list]:
    """Batches of (request, first, size, drafts, accepted); drafts None marks prefill."""
    rng = random.Random(args.seed)
    prompts = [rng.randrange(1, args.prompt_max + 1) for _ in range(args.requests)]
    state = [{"pos": 0, "steps": 0, "start": 3 * r} for r in range(args.requests)]
    batches, step = [], 0
    while any(s["steps"] < args.steps for s in state):
        decode, prefill = [], []
        for r, s in enumerate(state):
            if step < s["start"] or s["steps"] >= args.steps:
                continue
            if s["pos"] < prompts[r]:
                size = min(prompts[r] - s["pos"], rng.randrange(1, args.max_chunk + 1))
                prefill.append((r, s["pos"], size, None, None))
                s["pos"] += size
            else:
                k = args.spec_tokens
                drafts = k if rng.random() < 0.7 else rng.randrange(0, k + 1)
                accepted = rng.randrange(0, drafts + 1)
                decode.append((r, s["pos"], drafts + 1, drafts, accepted))
                s["pos"] += accepted + 1
                s["steps"] += 1
        batches.append(decode + prefill)   # update_decode_pools: decode requests first
        step += 1
    return batches, [s["pos"] for s in state]


def counterexample(spec_tokens: int) -> tuple[list, list]:
    """Committed 0 and 1; a verify step over 2..2+K accepts only 2; then position 3."""
    return [[(0, 0, 2, None, None)], [(0, 2, spec_tokens + 1, spec_tokens, 0)],
            [(0, 3, 1, 0, 0)]], [4]


def geometry(block: int) -> tuple[int, int, int, int]:
    """pooled_indexer._index_cache_view: block MLA records, then block / 256 C4 pages."""
    subpages, semantic = block // (POOL * PAGE), block * RECORD
    stride_bytes = math.ceil((semantic + subpages * PAGE_BYTES) / PAGE_BYTES) * PAGE_BYTES
    return subpages, semantic, stride_bytes, stride_bytes // PAGE_BYTES


def pool_location(parent, position, block: int, parent_stride_pages: int):
    """The kernels' packed mapping of a completion row to (virtual C4 page, entry)."""
    pool_offset = position % block // POOL
    return parent * parent_stride_pages + pool_offset // PAGE, pool_offset % PAGE


def main() -> int:
    args = parse_args()
    import torch

    if not torch.cuda.is_available():
        sys.exit("CUDA is required; --help works without it")
    spec = importlib.util.spec_from_file_location(
        "e31_glm_kpool", Path(__file__).with_name("glm_kpool.py"))
    kpool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kpool)
    dev = torch.device("cuda")
    rng = random.Random(args.seed + 1)
    ring_max = POOL * math.ceil((POOL + args.spec_tokens) / POOL)
    block = args.block
    subpages, semantic, stride_bytes, parent_stride_pages = geometry(block)
    max_seqs = 4 * args.requests + 3
    slots = rng.sample(range(1, max_seqs), args.requests)   # non-contiguous, never 0
    g = torch.Generator(device=dev).manual_seed(args.seed)

    def randn(*shape, scale=1.0):
        return (scale * torch.randn(*shape, generator=g, device=dev)).to(torch.bfloat16)

    ape = randn(POOL, DIM, scale=0.1)
    hadamard = torch.ones(1, 1, device=dev)
    while hadamard.shape[0] < DIM:
        hadamard = torch.cat([torch.cat([hadamard, hadamard], 1),
                              torch.cat([hadamard, -hadamard], 1)], 0)

    def experiment(batches, committed, ring, keys, gates, drafts, pages):
        total = int(pages.max()) + 1
        buf = torch.zeros(total * stride_bytes, dtype=torch.uint8, device=dev)
        main = torch.as_strided(buf, (total, block, RECORD), (stride_bytes, RECORD, 1))
        vpages = (total - 1) * parent_stride_pages + subpages
        index = torch.as_strided(main, (vpages, PAGE, WIDTH), (PAGE_BYTES, WIDTH, 1),
                                 storage_offset=semantic)
        tail = torch.zeros(max_seqs, 2, ring_max, DIM, dtype=torch.bfloat16, device=dev)
        for batch in batches:
            rows, qsl, state = [], [0], []
            for r, first, size, n_drafts, accepted in batch:
                for p in range(first, first + size):
                    true = n_drafts is None or p - first <= accepted
                    rows.append((r, p, true))
                qsl.append(len(rows))
                state.append(slots[r])
            req = torch.tensor([r for r, _, _ in rows], device=dev)
            pos = torch.tensor([p for _, p, _ in rows], device=dev)
            true = torch.tensor([t for _, _, t in rows], device=dev).unsqueeze(1)
            parent = pages[req, pos // block]
            slot_mapping = parent * block + pos % block
            key = torch.where(true, keys[req, pos], drafts[0][req, pos])
            gate = torch.where(true, gates[req, pos], drafts[1][req, pos])
            kpool.update_decode_pools(
                index, tail, torch.tensor(state, dtype=torch.int32, device=dev),
                torch.tensor(qsl, dtype=torch.int32, device=dev), key.contiguous(),
                gate.contiguous(), ape, slot_mapping, pos, len(batch),
                num_decode_requests=sum(b[3] is not None for b in batch),
                max_query_len=max(b[2] for b in batch), model_block_size=block,
                parent_stride_pages=parent_stride_pages, tail_ring=ring)
        torch.cuda.synchronize()
        raw = buf.cpu()
        out = []
        for r, length in enumerate(committed):
            p = torch.arange(length // POOL) * POOL + POOL - 1
            vpage, entry = pool_location(pages[r].cpu()[p // block], p, block,
                                         parent_stride_pages)
            base = semantic + vpage * PAGE_BYTES
            values = raw[(base + entry * DIM).unsqueeze(1) + torch.arange(DIM)]
            scales = raw[(base + PAGE * DIM + entry * 4).unsqueeze(1) + torch.arange(4)]
            out.append(values.view(torch.float8_e4m3fn).float()
                       * scales.contiguous().view(torch.float32))
        return out

    def independent(keys, gates, r, pool):
        k = keys[r, pool * POOL:pool * POOL + POOL].float()
        score = gates[r, pool * POOL:pool * POOL + POOL].float() + ape.float()
        weight = torch.exp(score - score.max(0).values)
        x = ((weight * k).sum(0) / weight.sum(0).clamp_min(1e-20)).to(torch.bfloat16).float()
        x = (x @ hadamard * 0.08838834764831845).to(torch.bfloat16).float()
        scale = torch.exp2(torch.ceil(torch.log2(x.abs().max().clamp_min(1e-4) / 448.0)))
        return x.cpu(), scale.cpu()   # pools are read back on the host

    def compare(got, ref):
        wrong = inexact = 0
        worst = 0.0
        for a, b in zip(got, ref):
            diff = (a - b).abs().amax(1)
            bad = ~torch.isfinite(a).all(1) | ~torch.isfinite(diff)
            wrong += int((bad | (diff > args.tol * b.abs().amax(1))).sum())
            inexact += int((bad | (diff > 0)).sum())
            if diff.numel() and torch.isfinite(diff).all():
                worst = max(worst, float(diff.max()))
        return {"wrong_pools": wrong, "inexact_pools": inexact, "max_abs_diff": worst}

    def arms(batches, committed):
        n, cap = len(committed), max(committed) + args.spec_tokens + 2
        keys, gates = randn(n, cap, DIM), randn(n, cap, DIM)
        drafts = (randn(n, cap, DIM), randn(n, cap, DIM))
        assert all(torch.isfinite(t.float()).all() for t in (keys, gates, *drafts, ape))
        count = math.ceil(cap / block)
        order = list(range(1, n * count + 1))   # page 0 unused; shuffled parent pages
        rng.shuffle(order)
        pages = torch.tensor(order, device=dev).view(n, count)
        whole = [[(r, 0, length, None, None) for r, length in enumerate(committed)]]
        ref = experiment(whole, committed, POOL, keys, gates, drafts, pages)
        checks = {"finite": all(bool(torch.isfinite(x).all()) for x in ref),
                  "nonzero": all(bool((x.abs().amax(1) > 0).all()) for x in ref),
                  "distinct": all(bool(((x[1:] - x[:-1]).abs().amax(1) > 0).all()) for x in ref)}
        misses = 0
        for r, x in enumerate(ref):
            picks = {0, len(x) - 1} | set(rng.sample(range(len(x)), min(args.spot, len(x))))
            for pool in sorted(q for q in picks if 0 <= q < len(x)):
                want, scale = independent(keys, gates, r, pool)
                bound = want.abs() / 8 + scale / 256 + 1e-6   # one FP8 E4M3 step
                misses += int(((x[pool] - want).abs() > bound).any())
        checks["independent_misses"] = misses
        result = {"reference": checks}
        for name, ring in (("legacy", POOL), ("e31", ring_max)):
            got = experiment(batches, committed, ring, keys, gates, drafts, pages)
            result[name] = compare(got, ref)
        return result

    counter = arms(*counterexample(args.spec_tokens))
    batches, committed = random_schedule(args)
    randomized = arms(batches, committed)
    ok = all(r["reference"]["finite"] and r["reference"]["nonzero"]
             and r["reference"]["distinct"] and r["reference"]["independent_misses"] == 0
             and r["legacy"]["wrong_pools"] > 0 and r["e31"]["wrong_pools"] == 0
             for r in (counter, randomized))
    for name, r in (("counterexample", counter), ("random", randomized)):
        print(f"{name}: {r}", file=sys.stderr)
    print(json.dumps({"experiment": "e31-kpool-tail-ring", "pass": ok,
                      "spec_tokens": args.spec_tokens, "ring_max": ring_max, "block": block,
                      "parent_stride_pages": parent_stride_pages, "state_slots": slots,
                      "seed": args.seed, "committed_tokens": committed, "batches": len(batches),
                      "counterexample": counter, "random": randomized,
                      "torch": torch.__version__}, separators=(",", ":")))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
