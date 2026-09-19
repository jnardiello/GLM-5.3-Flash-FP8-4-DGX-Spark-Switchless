#!/usr/bin/env python3
"""Compare the actual old/new restore methods on small heterogeneous page tables.

Requires Torch and the pinned operator payload. CPU is the default and must not
create a CUDA context. The optional CUDA check belongs in a coordinated window
with the serving stack stopped. Neither mode loads model weights.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import sys
import time
from types import SimpleNamespace


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def method(source, codec, torch):
    tree = ast.parse(source)
    node = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "_load_hybrid_pages")
    namespace = dict(vars(codec), torch=torch, time=time,
                     logger=logging.getLogger("replay-check"),
                     chunk_count=lambda tokens, size: (tokens + size - 1) // size)
    exec(compile("from __future__ import annotations\n" + ast.unparse(node),
                 "<operator-restore-method>", "exec"), namespace)
    return namespace[node.name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connector", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    prepare = module("replay_prepare", Path(__file__).with_name("prepare.py"))
    source = args.connector.read_bytes()
    candidate = prepare.transform(source)
    assert hashlib.sha256(args.encoder.read_bytes()).hexdigest() == prepare.ENCODER_SHA256
    codec = module("replay_codec", args.encoder)
    import torch
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    original = method(source, codec, torch)
    changed = method(candidate, codec, torch)
    layout = codec.PageLayout((
        codec.PageGroup(16, (
            codec.PageLayer("attention.fp8", "uint8", (4, 8), 32),
            codec.PageLayer("attention.state", "bfloat16", (3, 4), 24),
        )),
        codec.PageGroup(32, (
            codec.PageLayer("recurrent.state", "float32", (2, 3), 24),
        ), reuse_policy="recurrent_align"),
    ))
    # Non-monotonic physical slots expose accidental logical/physical reordering.
    groups = ((6, 1, 8), (4, 0))
    counts = tuple(map(len, groups))
    payloads = {}
    tensors = {}
    for group, count in zip(layout.groups, counts):
        for layer in group.layers:
            dtype = getattr(torch, layer.dtype)
            data = (torch.arange(count * layer.bytes_per_page // dtype.itemsize)
                    .remainder(97).to(dtype).reshape(count, *layer.page_shape))
            payloads[layer.name] = bytes(data.view(torch.uint8).flatten().tolist())
            tensors[layer.name] = data
    snapshot = codec.encode_page_snapshot(layout, counts, payloads)
    cases = []

    def run(fn, encoded):
        destinations = {
            layer.name: torch.full((10, *layer.page_shape), 113,
                                   dtype=getattr(torch, layer.dtype), device=args.device)
            for group in layout.groups for layer in group.layers
        }
        checkpoints = []
        expected_hash = hashlib.sha256(encoded).hexdigest()

        def synchronize():
            if args.device == "cuda":
                torch.cuda.synchronize()
            # Both owner and source bytes survive until placement is complete.
            assert hashlib.sha256(encoded).hexdigest() == expected_hash
            checkpoints.append("synchronized")

        owner = SimpleNamespace(
            _page_layout=layout, _native_restore_enabled=False, _chunk_tokens=16,
            _layer_tensors=destinations,
            _select_group_blocks_for_span=lambda ids, span: ids,
            _store=SimpleNamespace(restore_page_snapshot=lambda *a, **kw: encoded),
            _load_write_context=contextlib.nullcontext,
            _load_stream=SimpleNamespace(synchronize=synchronize),
        )
        plan = SimpleNamespace(group_block_ids=groups, span_tokens=48, request_id="cpu-check")
        assert fn(owner, SimpleNamespace(root_kind="page_snapshot"), plan)
        assert checkpoints == ["synchronized"]
        for group, slots in zip(layout.groups, groups):
            for layer in group.layers:
                result = destinations[layer.name].cpu()
                assert torch.equal(result[list(slots)], tensors[layer.name])
                untouched = [i for i in range(10) if i not in slots]
                assert torch.all(result[untouched] == 113)
        return destinations

    for kind in (bytes, bytearray):
        before = run(original, kind(snapshot))
        after = run(changed, kind(snapshot))
        assert all(torch.equal(before[name], after[name]) for name in before)
        cases.append(f"{kind.__name__}: exact pages, slots, untouched blocks, completion")
    # The same validator must reject damaged framing before opening placement.
    for label, bad in (("truncated", snapshot[:-1]), ("wrong header", b"BAD" + snapshot[3:])):
        for fn in (original, changed):
            try:
                run(fn, bad)
            except codec.HybridCodecError:
                pass
            else:
                raise AssertionError(f"accepted {label}")
        cases.append(f"{label}: rejected by both methods")
    if args.device == "cpu":
        assert not torch.cuda.is_initialized(), "CPU check created a CUDA context"
    print(json.dumps({"status": "PASS", "device": args.device,
                      "torch": torch.__version__, "cases": cases,
                      "absolute_error": 0, "relative_error": 0,
                      "base_sha256": prepare.BASE_SHA256,
                      "candidate_sha256": prepare.CANDIDATE_SHA256,
                      "encoder_sha256": prepare.ENCODER_SHA256}, indent=2))


if __name__ == "__main__":
    main()
