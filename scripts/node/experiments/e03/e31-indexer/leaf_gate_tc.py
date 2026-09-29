#!/usr/bin/env python3
"""E31 leaf test: GLM indexer head gate as a BF16 tensor-core GEMM with FP32 output.

Compares the serving FP32 path, F.linear(h.float(), W.float()), with the candidate
torch.mm(h, W.t(), out_dtype=torch.float32) on one idle GPU. Reports numeric error,
a top-512 pool-selection proxy, eager and CUDA-graph timings, and an optional
alternating A/B timing that mirrors a same-load serving comparison. Torch only;
no vLLM import. The idea follows the Apache-2.0 RiNGSiDE GLM53_INDEXER_GATE_TC patch.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--weights", help="BF16 [32, 4096] gate tensor (.pt or a .safetensors shard)")
    p.add_argument("--key", help="tensor name in --weights (default: first *weights_proj.weight)")
    p.add_argument("--rows", default="4,8,16,32,48", help="comma-separated M values")
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--pools", type=int, default=16384)
    p.add_argument("--topk", type=int, default=512)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--ab-blocks", type=int, default=0,
                   help="alternate A/B blocks per M in one process (0 = skip)")
    p.add_argument("--ab-iters", type=int, default=50, help="iterations per A/B block")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_weight(args, torch):
    if not args.weights:
        g = torch.Generator(device="cpu").manual_seed(args.seed)
        w = torch.randn(args.heads, args.hidden, generator=g) * args.hidden ** -0.5
        return w.to(torch.bfloat16)
    pick = lambda keys: args.key or next(k for k in keys if k.endswith("weights_proj.weight"))  # noqa: E731
    if args.weights.endswith(".safetensors"):
        from safetensors import safe_open  # optional; reads only the selected tensor
        with safe_open(args.weights, framework="pt") as f:
            obj = f.get_tensor(pick(list(f.keys())))
    else:
        obj = torch.load(args.weights, map_location="cpu")
        if isinstance(obj, dict):
            obj = obj[pick(list(obj))]
    if tuple(obj.shape) != (args.heads, args.hidden):
        sys.exit(f"gate weight shape {tuple(obj.shape)} != ({args.heads}, {args.hidden})")
    return obj.to(torch.bfloat16)


def main() -> int:
    args = parse_args()
    import torch
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        sys.exit("CUDA is required for the measurement; --help works without it")
    dev = torch.device("cuda")
    torch.manual_seed(args.seed)
    w = load_weight(args, torch).to(dev).contiguous()
    w32 = w.float()

    def ref(h):
        return F.linear(h.float(), w32)

    def cand(h):
        return torch.mm(h, w.t(), out_dtype=torch.float32)

    try:
        cand(torch.zeros(1, args.hidden, dtype=torch.bfloat16, device=dev))
        path = "bf16_tc_out_dtype_fp32"
    except (TypeError, RuntimeError) as exc:
        print(f"out_dtype unsupported ({exc}); candidate falls back to FP32", file=sys.stderr)
        cand, path = ref, "fp32_fallback"

    def median_us(fn, h, graph):
        if graph:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    fn(h)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn(h)
            call = g.replay
        else:
            call = lambda: fn(h)  # noqa: E731
        for _ in range(args.warmup):
            call()
        pairs = []
        for _ in range(args.iters):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(); call(); b.record()
            pairs.append((a, b))
        torch.cuda.synchronize()
        return statistics.median(a.elapsed_time(b) * 1e3 for a, b in pairs)

    def ab_medians(h):
        times = {"A": [], "B": []}
        fns = {"A": ref, "B": cand}
        for block in range(2 * args.ab_blocks):
            name = "AB"[block % 2]
            for _ in range(args.warmup):
                fns[name](h)
            pairs = []
            for _ in range(args.ab_iters):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record(); fns[name](h); b.record()
                pairs.append((a, b))
            torch.cuda.synchronize()
            times[name] += [a.elapsed_time(b) * 1e3 for a, b in pairs]
        return {k: round(statistics.median(v), 2) for k, v in times.items()}

    results = []
    for m in (int(x) for x in args.rows.split(",")):
        h = torch.randn(m, args.hidden, device=dev).to(torch.bfloat16)
        r, c = ref(h), cand(h)
        diff = (r - c).abs()
        rel = (diff / r.abs().clamp_min(1e-6)).max().item()
        # Pool-selection proxy: ReLU(q.k) per head and pool, weighted by each gate.
        scores = torch.relu(torch.randn(m, args.heads, args.pools, device=dev))
        tot_r = torch.einsum("mh,mhp->mp", r, scores)
        tot_c = torch.einsum("mh,mhp->mp", c, scores)
        sel_r = tot_r.topk(args.topk, dim=1).indices
        sel_c = tot_c.topk(args.topk, dim=1).indices
        hit = torch.zeros(m, args.pools, dtype=torch.int8, device=dev)
        hit.scatter_(1, sel_r, 1)
        common = hit.gather(1, sel_c).sum(dim=1).float()
        mismatched = int((args.topk - common).sum().item())
        jaccard = (common / (2 * args.topk - common)).mean().item()
        # Near ties: pools whose reference total lies within the largest total
        # difference of the reference k-th score, i.e. those that could flip.
        tol = (tot_r - tot_c).abs().max()
        kth = tot_r.topk(args.topk, dim=1).values[:, -1:]
        near_ties = int(((tot_r - kth).abs() <= tol).sum().item()) - m
        row = {
            "m": m, "max_abs": diff.max().item(), "max_rel": rel,
            "topk_mismatched": mismatched, "topk_total": m * args.topk,
            "jaccard_mean": round(jaccard, 6), "near_ties": near_ties,
            "eager_us": {"ref": round(median_us(ref, h, False), 2),
                         "cand": round(median_us(cand, h, False), 2)},
            "graph_us": {"ref": round(median_us(ref, h, True), 2),
                         "cand": round(median_us(cand, h, True), 2)},
        }
        if args.ab_blocks:
            row["ab_eager_us"] = ab_medians(h)
        results.append(row)
        print(f"M={m:3d} max_abs={row['max_abs']:.3e} max_rel={rel:.3e} "
              f"topk_mismatch={mismatched}/{m * args.topk} jaccard={jaccard:.6f} "
              f"near_ties={near_ties} eager_us={row['eager_us']} graph_us={row['graph_us']}"
              + (f" ab_us={row['ab_eager_us']}" if args.ab_blocks else ""), file=sys.stderr)

    print(json.dumps({
        "experiment": "e31-indexer-gate-tc", "candidate_path": path,
        "torch": torch.__version__, "device": torch.cuda.get_device_name(dev),
        "weights": "file" if args.weights else f"random(seed={args.seed})",
        "iters": args.iters, "pools": args.pools, "topk": args.topk, "rows": results,
    }, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
