#!/usr/bin/env python3
"""Build the fidelity corpus: scored windows, decode prompts, manifests and summaries.

Subcommands (run with data/fidelity/.venv/bin/python):

  build        parse sessions, redact, render, cut windows and decode prompts, write
               manifests, private inventories and the public summary
  add-native   append model_native windows (decode prompt + saved greedy generation)
  freeze       mark manifest.json frozen (only after the owner confirms exclusions)
  verify       check every token file against its manifest hash and the vocabulary

Output never contains corpus text: only counts, lengths and hashes. See README.md.
"""

from __future__ import annotations

import argparse
import array
import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path(__file__).resolve().parents[3]
CORPUS = REPO / "data/fidelity/corpus"
SUMMARY = REPO / "docs/fidelity/corpus-summary.json"
DEFAULT_BENCH = REPO / "third_party/knapcio-bench/bench"
MODEL_REPO = "zai-org/GLM-5.3-Flash"
MODEL_REV = "690b705278a3a58e538fcb37c2ca8b5f9511213c"
SCHEMA = "fidelity-corpus/1"
CONFIG_VOCAB = 154880
SEED = 20260927

SHORT_BAND = (3072, 8191)
MEDIUM_BAND = (8192 + 512, 8192 + 2047)
LONG_BAND = (32768, 65535)
HUGE_BAND = (126976, 131000)
N_LONG = 4
N_STRUCTURED = 40
N_AGENTIC = 160
N_DECODE_SESSION = 100
DECODE_LEN = (1024, 16384)
ITALIAN_MIN = 3072
ITALIAN_TAIL_MIN = 2560   # overlapping second-half Italian windows (docs/fidelity/REPORT.md, amendment 4)
HIST_BINS = [0, 1024, 2048, 3072, 4096, 8192, 8704, 10240, 16384, 32768, 65536, 131072, 262144]


def log(message: str) -> None:
    print(message, flush=True)


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def write_json(path: Path, data, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=False)
        handle.write("\n")
    os.chmod(tmp, mode)
    os.replace(tmp, path)


def u32_bytes(ids) -> bytes:
    values = array.array("I", ids)
    if sys.byteorder == "big":
        values.byteswap()
    return values.tobytes()


def read_u32(path: Path) -> array.array:
    values = array.array("I")
    values.frombytes(path.read_bytes())
    if sys.byteorder == "big":
        values.byteswap()
    return values


def global_sha256(entries: list) -> str:
    text = "".join(f"{e['id']}:{e['sha256']}\n" for e in sorted(entries, key=lambda e: e["id"]))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def save_entry(root: Path, subdir: str, entry: dict, ids) -> dict:
    data = u32_bytes(ids)
    rel = f"{subdir}/{entry['id']}.u32"
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    os.chmod(path, 0o600)
    return {"id": entry["id"], "category": entry["category"], "source": entry["source"],
            "project": entry["project"], "n_tokens": len(ids),
            "sha256": hashlib.sha256(data).hexdigest(), "path": rel}


def histogram(lengths: list) -> list:
    rows = []
    for lo, hi in zip(HIST_BINS, HIST_BINS[1:]):
        rows.append({"min": lo, "max_exclusive": hi, "count": sum(1 for n in lengths if lo <= n < hi)})
    return rows


def read_excludes(path: Path) -> set:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# One project directory name per line (bare name = both sources, or\n"
                        "# claude_code/<name> / omp/<name>). Lines starting with # are ignored.\n",
                        encoding="utf-8")
        os.chmod(path, 0o600)
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")}


# ---------------------------------------------------------------- segments


class Segment:
    """One conversation segment, rendered as a tools header plus per-unit token arrays."""

    def __init__(self, sid: int, session, index: int, units: list, tools, header: list, unit_ids: list):
        self.sid = sid
        self.session = session
        self.index = index
        self.units = units
        self.tools = tools
        self.header = header
        self.unit_ids = unit_ids
        self.unit_len = [len(ids) for ids in unit_ids]
        self.cum = [0]
        for n in self.unit_len:
            self.cum.append(self.cum[-1] + n)
        self.used: list = []           # (first_unit, last_unit) ranges already taken
        self.calls = [sum(len(m.get("tool_calls") or []) for m in unit) for unit in units]
        from convert import structured_chars
        self.struct = [structured_chars(unit) for unit in units]

    @property
    def key(self) -> str:
        return f"{self.session.key}:{self.index}"

    def available(self, start: int) -> int:
        return len(self.header) + self.cum[-1] - self.cum[start]

    def end_unit(self, start: int, length: int) -> int | None:
        """Last unit touched by a window of `length` tokens starting at unit `start`."""
        need = length - len(self.header)
        if need <= 0 or self.cum[-1] - self.cum[start] < need:
            return None
        target = self.cum[start] + need
        lo, hi = start, len(self.unit_len) - 1
        while lo < hi:                  # first e with cum[e + 1] >= target
            mid = (lo + hi) // 2
            if self.cum[mid + 1] >= target:
                hi = mid
            else:
                lo = mid + 1
        return lo

    def free(self, first: int, last: int) -> bool:
        return all(last < a or first > b for a, b in self.used)

    def window_ids(self, start: int, length: int) -> list:
        ids = list(self.header)
        unit = start
        while len(ids) < length:
            ids.extend(self.unit_ids[unit])
            unit += 1
        return ids[:length]

    def messages(self, first: int, last: int) -> list:
        return [m for unit in self.units[first:last + 1] for m in unit]

    def score(self, first: int, last: int) -> float:
        structured = sum(self.struct[i][0] for i in range(first, last + 1))
        total = sum(self.struct[i][1] for i in range(first, last + 1))
        return structured / total if total else 0.0


def load_segments(renderer, excluded: set, cutoff: str):
    from convert import ToolSchemas, redact_segment
    from redact import Redactor
    from render import split_units
    import sources

    redactors = {"claude_code": Redactor(), "omp": Redactor()}
    inventory: dict = {}
    schemas = ToolSchemas()
    sessions = []
    for session in sources.load_sessions(excluded, cutoff):
        redactor = redactors[session.source]
        session.segments = [redact_segment(seg, redactor) for seg in session.segments]
        for seg in session.segments:
            schemas.observe(session.source, seg)
        sessions.append(session)
        row = inventory.setdefault(f"{session.source}/{session.project}", {
            "source": session.source, "sessions": 0, "segments": 0, "events": 0, "user_human": 0,
            "user_other": 0, "assistant": 0, "tool_results": 0, "tool_calls": 0, "system": 0,
            "approx_chars": 0})
        row["sessions"] += 1
        row["segments"] += len(session.segments)
        for src, dst in (("events", "events"), ("user_human", "user_human"), ("user_other", "user_other"),
                         ("assistant", "assistant"), ("tool", "tool_results"), ("tool_calls", "tool_calls"),
                         ("system", "system"), ("chars", "approx_chars")):
            row[dst] += session.stats.get(src, 0)
    log(f"parsed sessions={len(sessions)} projects={len(inventory)}")

    segments = []
    for session in sessions:
        for index, seg in enumerate(session.segments):
            units = split_units(seg)
            if not units:
                continue
            tools = schemas.for_segment(session.source, [m for u in units for m in u])
            unit_ids = renderer.encode_many(renderer.unit_texts(units))
            segments.append(Segment(len(segments), session, index, units, tools,
                                    renderer.header_ids(tools), unit_ids))
    for session in sessions:
        session.segments = None        # free the unsplit copies
    for seg in segments:
        row = inventory[f"{seg.session.source}/{seg.session.project}"]
        row["approx_tokens"] = row.get("approx_tokens", 0) + len(seg.header) + seg.cum[-1]
    log(f"segments={len(segments)} tokens={sum(s.cum[-1] + len(s.header) for s in segments)}")
    counts = {name: {k: v for k, v in r.snapshot().items()} for name, r in redactors.items()}
    return segments, inventory, counts


# ---------------------------------------------------------------- selection


def pick_length(rng: random.Random, band: tuple) -> int:
    return rng.randint(band[0], band[1])


def select_long(segments: list, rng: random.Random, shortfalls: list) -> list:
    chosen = []
    order = sorted(segments, key=lambda s: (-s.available(0), s.key))
    used_sessions = set()
    plan = [("huge", pick_length(rng, HUGE_BAND))]
    for j in range(N_LONG):
        residue = int((j + rng.random()) * 8192 / N_LONG)
        plan.append(("long", rng.randint(4, 7) * 8192 + residue))
    for kind, length in plan:
        candidates = [s for s in order if s.available(0) >= length and s.free(0, 0)]
        fresh = [s for s in candidates if s.session.key not in used_sessions]
        pool = fresh or candidates
        if not pool:
            shortfalls.append(f"long_context: no segment has {length} tokens for a {kind} window")
            continue
        seg = pool[0]
        last = seg.end_unit(0, length)
        seg.used.append((0, last))
        used_sessions.add(seg.session.key)
        chosen.append({"category": "long_context", "seg": seg, "first": 0, "last": last,
                       "length": length, "band": kind})
    return chosen


def tile(segments: list, rng: random.Random) -> list:
    tiles = []
    for seg in segments:
        start = max([b + 1 for a, b in seg.used] or [0])
        while start < len(seg.units):
            band = "short" if rng.random() < 0.5 else "medium"
            length = pick_length(rng, SHORT_BAND if band == "short" else MEDIUM_BAND)
            if seg.available(start) < length:
                if seg.available(start) < SHORT_BAND[0]:
                    break
                band = "short"
                length = rng.randint(SHORT_BAND[0], min(SHORT_BAND[1], seg.available(start)))
            last = seg.end_unit(start, length)
            if last is None:
                break
            tiles.append({"seg": seg, "first": start, "last": last, "length": length, "band": band,
                          "calls": sum(seg.calls[start:last + 1]), "score": seg.score(start, last)})
            start = last + 1
    return tiles


def take(tile_: dict, category: str) -> dict:
    tile_["seg"].used.append((tile_["first"], tile_["last"]))
    return dict(tile_, category=category)


def select_structured(tiles: list, shortfalls: list, cap: int = 3) -> list:
    chosen, per_session = [], {}
    for band in ("short", "medium"):
        need = N_STRUCTURED // 2
        pool = sorted((t for t in tiles if t["band"] == band and t["calls"] > 0),
                      key=lambda t: (-t["score"], t["seg"].key, t["first"]))
        for t in pool:
            if need == 0:
                break
            key = t["seg"].session.key
            if per_session.get(key, 0) >= cap or not t["seg"].free(t["first"], t["last"]):
                continue
            per_session[key] = per_session.get(key, 0) + 1
            chosen.append(take(t, "structured_json"))
            need -= 1
        if need:
            shortfalls.append(f"structured_json: {need} {band} windows missing")
    return chosen


def select_agentic(tiles: list, rng: random.Random, shortfalls: list) -> list:
    by_session: dict = {}
    for t in tiles:
        if t["calls"] > 0:
            by_session.setdefault(t["seg"].session.key, []).append(t)
    keys = sorted(by_session)
    rng.shuffle(keys)
    for key in keys:
        rng.shuffle(by_session[key])
    need = {"short": N_AGENTIC // 2, "medium": N_AGENTIC - N_AGENTIC // 2}
    chosen = []
    progress = True
    while progress and sum(need.values()) > 0:
        progress = False
        for key in keys:
            for band in sorted(need, key=lambda b: -need[b]):
                if need[band] == 0:
                    continue
                pool = [t for t in by_session[key] if t["band"] == band and t["seg"].free(t["first"], t["last"])]
                if pool:
                    t = pool[0]
                    by_session[key].remove(t)
                    chosen.append(take(t, "agentic_code"))
                    need[band] -= 1
                    progress = True
                    break
            if sum(need.values()) == 0:
                break
    for band, n in need.items():
        if n:
            shortfalls.append(f"agentic_code: {n} {band} windows missing")
    return chosen


def select_decode(segments: list, renderer, rng: random.Random, shortfalls: list) -> list:
    gen = renderer.generation_prompt_ids()
    windowed = {s.session.key for s in segments if s.used}
    candidates: dict = {}
    for seg in segments:
        for u, unit in enumerate(seg.units):
            if unit[0]["role"] == "user" and unit[0].get("_human"):
                candidates.setdefault(seg.session.key, []).append((seg, u))
    keys = sorted(candidates)
    rng.shuffle(keys)
    keys.sort(key=lambda k: k in windowed)          # sessions without windows first (stable)
    for key in keys:
        rng.shuffle(candidates[key])
    chosen = []
    progress = True
    while progress and len(chosen) < N_DECODE_SESSION:
        progress = False
        for key in keys:
            while candidates[key]:
                seg, u = candidates[key].pop()
                if not seg.free(u, u):
                    continue
                target = math.exp(rng.uniform(math.log(DECODE_LEN[0]), math.log(DECODE_LEN[1])))
                best = None
                fixed = len(seg.header) + len(gen)
                for start in range(u, -1, -1):
                    if not seg.free(start, u):
                        break
                    length = fixed + seg.cum[u + 1] - seg.cum[start]
                    if length > DECODE_LEN[1]:
                        break
                    if length >= DECODE_LEN[0] and (best is None or abs(length - target) < abs(best[1] - target)):
                        best = (start, length)
                if best is None:
                    continue
                start, length = best
                seg.used.append((start, u))
                chosen.append({"seg": seg, "first": start, "last": u, "length": length})
                progress = True
                break
            if len(chosen) >= N_DECODE_SESSION:
                break
    if len(chosen) < N_DECODE_SESSION:
        shortfalls.append(f"decode: {N_DECODE_SESSION - len(chosen)} session prompts missing")
    return chosen


# ---------------------------------------------------------------- italian and public


def load_italian(renderer, directory: Path, rng: random.Random):
    from convert import redact_segment
    from redact import Redactor
    from render import split_units

    redactor = Redactor()
    windows = []
    files = sorted(directory.glob("it*.json")) if directory.is_dir() else []
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            messages = [{"role": m["role"], "content": m["content"]} for m in data["messages"]
                        if m.get("role") in ("system", "user", "assistant") and isinstance(m.get("content"), str)]
        except (ValueError, KeyError, TypeError):
            log(f"italian: skipped malformed file #{files.index(path) + 1}")
            continue
        units = split_units(redact_segment(messages, redactor))
        if not units:
            continue
        ids = renderer.encode_many(renderer.unit_texts(units))
        header = renderer.header_ids(None)
        system = units[0] if units[0][0]["role"] == "system" else None
        sys_len = len(ids[0]) if system else 0
        total = len(header) + sum(len(x) for x in ids)
        best = None
        for k in range(1 if system else 0, len(units)):
            if units[k][0]["role"] != "user" or k == (1 if system else 0):
                continue
            first = len(header) + sum(len(x) for x in ids[:k])
            second = len(header) + sys_len + sum(len(x) for x in ids[k:])
            if first >= ITALIAN_MIN and second >= ITALIAN_MIN:
                balance = abs(first - second)
                if best is None or balance < best[0]:
                    best = (balance, k)
        conv = {"file": path.name}
        if best is None:
            windows.append(dict(conv, messages=[m for u in units for m in u],
                                ids=header + [t for x in ids for t in x]))
            # Too short for two disjoint windows: add one overlapping (sliding) window of the
            # second half, re-rendered after the system turn, when it is long enough.
            starts = [k for k in range(2 if system else 1, len(units)) if units[k][0]["role"] == "user"]
            if starts:
                k = min(starts, key=lambda j: abs(sum(len(x) for x in ids[:j]) - total / 2))
                tail_ids = (ids[0] if system else []) + [t for x in ids[k:] for t in x]
                if len(header) + len(tail_ids) >= ITALIAN_TAIL_MIN:
                    head = (list(system) if system else [])
                    windows.append(dict(conv, messages=head + [m for u in units[k:] for m in u],
                                        ids=header + tail_ids))
        else:
            k = best[1]
            windows.append(dict(conv, messages=[m for u in units[:k] for m in u],
                                ids=header + [t for x in ids[:k] for t in x]))
            head = (list(system) if system else [])
            tail_ids = (ids[0] if system else []) + [t for x in ids[k:] for t in x]
            windows.append(dict(conv, messages=head + [m for u in units[k:] for m in u],
                                ids=header + tail_ids))
        assert total > 0
    return windows, len(files), redactor.snapshot()


def load_public(renderer, bench: Path, rng: random.Random):
    def module(name):
        spec = importlib.util.spec_from_file_location(f"_bench_{name}", bench / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    hardset = module("hardset")
    qeval = module("qeval_tasks")
    prompts = [("hardset", pid, text) for pid, _cat, text in hardset.PROMPTS]
    by_cat: dict = {}
    for task in sorted(qeval.TASKS, key=lambda t: t["id"]):
        by_cat.setdefault(task["category"], []).append(task)
    total = sum(len(v) for v in by_cat.values())
    quota = {cat: max(1, round(20 * len(v) / total)) for cat, v in by_cat.items()}
    while sum(quota.values()) > 20:
        quota[max(quota, key=lambda c: (quota[c], c))] -= 1
    while sum(quota.values()) < 20:
        quota[max(by_cat, key=lambda c: (len(by_cat[c]) - quota[c], c))] += 1
    picked = []
    for cat in sorted(by_cat):
        picked += rng.sample(by_cat[cat], quota[cat])
    prompts += [("qeval", t["id"], t["prompt"]) for t in sorted(picked, key=lambda t: t["id"])]
    out = []
    for origin, pid, text in prompts:
        messages = [{"role": "user", "content": text}]
        out.append({"origin": origin, "task_id": pid, "ids": renderer.tokens(messages, None, True)})
    provenance = {"hardset_py_sha256": sha256_path(bench / "hardset.py"),
                  "qeval_tasks_py_sha256": sha256_path(bench / "qeval_tasks.py"),
                  "qeval_ids": [t["id"] for t in sorted(picked, key=lambda t: t["id"])],
                  "license": "MIT (knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4)"}
    return out, provenance


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- build


def verify_window(renderer, messages: list, tools, ids: list, generation: bool = False) -> None:
    full = renderer.tokens(messages, tools, generation)
    if full[:len(ids)] != ids or (generation and full != ids):
        raise SystemExit("verify: assembled tokens differ from the full template render")


def cmd_build(args) -> int:
    from render import Renderer

    manifest_path = CORPUS / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()).get("frozen") and not args.force:
        log("build: manifest is frozen; refusing to rebuild (use --force only on owner request)")
        return 2
    info_path = CORPUS / "build-info.json"
    previous = json.loads(info_path.read_text()) if info_path.exists() else {}
    cutoff = args.cutoff or previous.get("cutoff") or utc_now()
    excluded = read_excludes(CORPUS / "exclude.txt")
    renderer = Renderer()
    equivalence = renderer.check_equivalence()
    log(f"template equivalence: {equivalence}")
    if not all(equivalence.values()):
        raise SystemExit("render/tokenize path differs from apply_chat_template(tokenize=True)")

    segments, inventory, redaction = load_segments(renderer, excluded, cutoff)
    shortfalls: list = []
    rng = random.Random(args.seed)
    long_windows = select_long(segments, random.Random(f"{args.seed}:long"), shortfalls)
    tiles = tile(segments, random.Random(f"{args.seed}:tile"))
    structured = select_structured(tiles, shortfalls)
    agentic = select_agentic(tiles, random.Random(f"{args.seed}:agentic"), shortfalls)
    decode = select_decode(segments, renderer, random.Random(f"{args.seed}:decode"), shortfalls)
    log(f"tiles={len(tiles)} long={len(long_windows)} structured={len(structured)} "
        f"agentic={len(agentic)} decode_session={len(decode)}")

    # Session windows: deterministic shuffle, then Italian windows (ids stay stable when
    # the Italian set changes), native windows are appended later by add-native.
    session_windows = long_windows + structured + agentic
    session_windows.sort(key=lambda w: (w["seg"].key, w["first"]))
    rng.shuffle(session_windows)
    italian, italian_files, italian_counts = load_italian(renderer, args.italian_dir,
                                                          random.Random(f"{args.seed}:italian"))
    redaction["synthetic_it"] = italian_counts

    for sub in ("tokens", "decode"):
        target = CORPUS / sub
        if target.is_dir():
            for old in target.glob("*.u32"):
                old.unlink()
    entries, meta = [], {}
    number = 0
    for w in session_windows:
        number += 1
        seg = w["seg"]
        ids = seg.window_ids(w["first"], w["length"])
        verify_window(renderer, seg.messages(w["first"], w["last"]), seg.tools, ids)
        entry = {"id": f"w{number:04d}", "category": w["category"], "source": seg.session.source,
                 "project": seg.session.project}
        entries.append(save_entry(CORPUS, "tokens", entry, ids))
        meta[entry["id"]] = {"segment": seg.key, "units": [w["first"], w["last"]], "band": w["band"],
                             "tool_calls": sum(seg.calls[w["first"]:w["last"] + 1]),
                             "structured_score": round(seg.score(w["first"], w["last"]), 4)}
    for w in italian:
        number += 1
        verify_window(renderer, w["messages"], None, w["ids"])
        entry = {"id": f"w{number:04d}", "category": "italian_chat", "source": "synthetic_it",
                 "project": "synthetic-it"}
        entries.append(save_entry(CORPUS, "tokens", entry, w["ids"]))
        meta[entry["id"]] = {"file": w["file"]}

    prompts, pmeta = [], {}
    gen = renderer.generation_prompt_ids()
    for i, d in enumerate(decode, 1):
        seg = d["seg"]
        ids = list(seg.header) + [t for x in seg.unit_ids[d["first"]:d["last"] + 1] for t in x] + list(gen)
        verify_window(renderer, seg.messages(d["first"], d["last"]), seg.tools, ids, generation=True)
        entry = {"id": f"d{i:03d}", "category": "session", "source": seg.session.source,
                 "project": seg.session.project}
        prompts.append(save_entry(CORPUS, "decode", entry, ids))
        pmeta[entry["id"]] = {"segment": seg.key, "units": [d["first"], d["last"]]}
    public, public_prov = load_public(renderer, args.bench_dir, random.Random(f"{args.seed}:public"))
    for p in public:
        i = len(prompts) + 1
        entry = {"id": f"d{i:03d}", "category": "public", "source": "public",
                 "project": f"knapcio-bench/{p['origin']}"}
        prompts.append(save_entry(CORPUS, "decode", entry, p["ids"]))
        pmeta[entry["id"]] = {"origin": p["origin"], "task_id": p["task_id"]}

    header = {"schema": SCHEMA, "model_repo": MODEL_REPO, "model_rev": MODEL_REV,
              "tokenizer_sha256": renderer.tokenizer_sha256, "chat_template_sha256": renderer.template_sha256,
              "template_kwargs": dict(renderer.kwargs), "frozen": False}
    write_json(manifest_path, dict(header, global_sha256=global_sha256(entries), windows=entries))
    write_json(CORPUS / "decode_manifest.json", dict(header, global_sha256=global_sha256(prompts), prompts=prompts))
    write_json(CORPUS / "windows-meta.json", meta)
    write_json(CORPUS / "decode-meta.json", pmeta)
    write_json(CORPUS / "sources-inventory.json", dict(sorted(inventory.items())))
    totals: dict = {}
    for per_source in redaction.values():
        for rule, n in per_source.items():
            totals[rule] = totals.get(rule, 0) + n
    write_json(CORPUS / "redaction-counts.json", {"total": dict(sorted(totals.items())), "by_source": redaction})
    write_json(info_path, {"cutoff": cutoff, "seed": args.seed, "excluded": sorted(excluded),
                           "built_utc": utc_now(), "template_equivalence": equivalence,
                           "tokenization": "render(tokenize=False) then encode(add_special_tokens=False)",
                           "italian_files": italian_files, "public": public_prov,
                           "shortfalls": shortfalls})
    write_summary()
    for line in shortfalls:
        log(f"shortfall: {line}")
    log(f"windows={len(entries)} prompts={len(prompts)} italian_files={italian_files}")
    return 0


# ---------------------------------------------------------------- native, freeze, summary, verify


def cmd_add_native(args) -> int:
    import numpy as np

    manifest_path = CORPUS / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("frozen") and not args.allow_frozen:
        log("add-native: manifest is frozen; adding native windows changes global_sha256 "
            "(pass --allow-frozen and record the new hash)")
        return 2
    decode = json.loads((CORPUS / "decode_manifest.json").read_text(encoding="utf-8"))
    native_manifest = CORPUS / "native_prompts_manifest.json"
    native_prompts = (json.loads(native_manifest.read_text(encoding="utf-8"))["prompts"]
                      if native_manifest.exists() else [])
    decode = dict(decode, prompts=decode["prompts"] + native_prompts)
    kept = [w for w in manifest["windows"] if w["category"] != "model_native"]
    for w in manifest["windows"]:
        if w["category"] == "model_native":
            (CORPUS / w["path"]).unlink(missing_ok=True)
    number = max(int(w["id"][1:]) for w in kept) if kept else 0
    added, skipped = [], {}
    provenance = {}
    for prompt in sorted(decode["prompts"], key=lambda p: p["id"]):
        sidecar = args.gen_dir / f"{prompt['id']}.json"
        npz = args.gen_dir / f"{prompt['id']}.npz"
        status = json.loads(sidecar.read_text()).get("status") if sidecar.exists() else "missing"
        if status != "ok" or not npz.exists():
            skipped[status or "unknown"] = skipped.get(status or "unknown", 0) + 1
            continue
        prompt_path = CORPUS / prompt["path"]
        if hashlib.sha256(prompt_path.read_bytes()).hexdigest() != prompt["sha256"]:
            raise SystemExit(f"add-native: decode prompt {prompt['id']} hash mismatch")
        with np.load(npz) as data:
            gen = [int(x) for x in data["gen_ids"]]
        if len(gen) < args.min_gen_tokens:
            skipped["short_generation"] = skipped.get("short_generation", 0) + 1
            continue
        if any(x < 0 or x >= CONFIG_VOCAB for x in gen):
            raise SystemExit(f"add-native: generated id out of range for {prompt['id']}")
        number += 1
        entry = {"id": f"w{number:04d}", "category": "model_native", "source": args.source,
                 "project": prompt["project"]}
        added.append(save_entry(CORPUS, "tokens", entry, list(read_u32(prompt_path)) + gen))
        provenance[entry["id"]] = {"decode_id": prompt["id"], "prompt_tokens": prompt["n_tokens"],
                                   "gen_tokens": len(gen), "npz_sha256": sha256_path(npz)}
    windows = kept + added
    manifest["windows"] = windows
    manifest["global_sha256"] = global_sha256(windows)
    write_json(manifest_path, manifest)
    write_json(CORPUS / "native-provenance.json", {"gen_dir_name": args.gen_dir.name, "source": args.source,
                                                   "decode_global_sha256": decode["global_sha256"],
                                                   "windows": provenance, "skipped": skipped})
    write_summary()
    log(f"add-native: added={len(added)} skipped={skipped} windows={len(windows)}")
    return 0


def cmd_native_prompts(args) -> int:
    """Render public long-form prompts (scripts/fidelity/native_prompts.json) for model-native generation."""
    from render import Renderer

    renderer = Renderer()
    texts = json.loads(args.file.read_text(encoding="utf-8"))["prompts"]
    prompts = []
    for i, text in enumerate(texts, 1):
        entry = {"id": f"n{i:03d}", "category": "native_prompt", "source": "public", "project": "native_prompts"}
        prompts.append(save_entry(CORPUS, "native_prompts", entry,
                                  renderer.tokens([{"role": "user", "content": text}], None, True)))
    write_json(CORPUS / "native_prompts_manifest.json",
               {"schema": "fidelity-corpus/1", "model_repo": MODEL_REPO, "model_rev": MODEL_REV,
                "tokenizer_sha256": renderer.tokenizer_sha256, "chat_template_sha256": renderer.template_sha256,
                "template_kwargs": renderer.kwargs, "global_sha256": global_sha256(prompts),
                "source_file_sha256": sha256_path(args.file), "prompts": prompts})
    log(f"native-prompts: {len(prompts)} rendered, tokens {sum(x['n_tokens'] for x in prompts)}")
    return 0


def cmd_freeze(args) -> int:
    path = CORPUS / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("frozen"):
        log("freeze: already frozen")
        return 0
    if verify(quiet=True):
        return 1
    manifest["frozen"] = True
    write_json(path, manifest)
    write_summary()
    log(f"freeze: frozen global_sha256={manifest['global_sha256']}")
    return 0


def write_summary() -> None:
    manifest = json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))
    decode = json.loads((CORPUS / "decode_manifest.json").read_text(encoding="utf-8"))
    redaction = json.loads((CORPUS / "redaction-counts.json").read_text(encoding="utf-8"))
    windows = manifest["windows"]
    total = sum(w["n_tokens"] for w in windows)

    def table(rows, field):
        out = {}
        for w in rows:
            cell = out.setdefault(w[field], {"windows": 0, "tokens": 0})
            cell["windows"] += 1
            cell["tokens"] += w["n_tokens"]
        for cell in out.values():
            cell["share_of_tokens"] = round(cell["tokens"] / total, 4) if total else 0.0
        return dict(sorted(out.items()))

    lengths = [w["n_tokens"] for w in windows]
    summary = {
        "schema": "fidelity-corpus-summary/1",
        "model_repo": manifest["model_repo"], "model_rev": manifest["model_rev"],
        "tokenizer_sha256": manifest["tokenizer_sha256"], "chat_template_sha256": manifest["chat_template_sha256"],
        "template_kwargs": manifest["template_kwargs"], "frozen": manifest["frozen"],
        "manifest_global_sha256": manifest["global_sha256"],
        "decode_manifest_global_sha256": decode["global_sha256"],
        "windows": len(windows), "total_tokens": total,
        "scored_positions": sum(max(0, n - 1) for n in lengths),
        "windows_at_least_3072": sum(1 for n in lengths if n >= 3072),
        "by_category": table(windows, "category"), "by_source": table(windows, "source"),
        "length_histogram": histogram(lengths),
        "long_context_lengths_mod_8192": sorted(w["n_tokens"] % 8192 for w in windows
                                                if w["category"] == "long_context"),
        "decode_prompts": {"count": len(decode["prompts"]),
                           "by_source": {s: sum(1 for p in decode["prompts"] if p["source"] == s)
                                         for s in sorted({p["source"] for p in decode["prompts"]})},
                           "length_histogram": histogram([p["n_tokens"] for p in decode["prompts"]])},
        "redaction_counts": redaction["total"],
    }
    write_json(SUMMARY, summary, mode=0o644)


def verify(quiet: bool = False) -> int:
    from render import TOKENIZER_DIR
    try:
        from tokenizers import Tokenizer
        vocab = Tokenizer.from_file(str(TOKENIZER_DIR / "tokenizer.json")).get_vocab_size(with_added_tokens=True)
    except ImportError:
        vocab = CONFIG_VOCAB
    limit = min(vocab, CONFIG_VOCAB)
    problems = 0
    for name, key in (("manifest.json", "windows"), ("decode_manifest.json", "prompts")):
        manifest = json.loads((CORPUS / name).read_text(encoding="utf-8"))
        entries = manifest[key]
        if manifest.get("global_sha256") != global_sha256(entries):
            problems += 1
            log(f"verify: {name} global_sha256 mismatch")
        if len({e["id"] for e in entries}) != len(entries):
            problems += 1
            log(f"verify: {name} duplicate ids")
        bad_hash = bad_len = bad_id = 0
        max_id = 0
        for e in entries:
            path = CORPUS / e["path"]
            data = path.read_bytes() if path.exists() else b""
            if hashlib.sha256(data).hexdigest() != e["sha256"]:
                bad_hash += 1
                continue
            ids = read_u32(path)
            if len(ids) != e["n_tokens"]:
                bad_len += 1
            top = max(ids) if len(ids) else 0
            max_id = max(max_id, top)
            if top >= limit:
                bad_id += 1
        problems += bad_hash + bad_len + bad_id
        if not quiet or bad_hash or bad_len or bad_id:
            log(f"verify: {name} entries={len(entries)} hash_mismatch={bad_hash} length_mismatch={bad_len} "
                f"id_out_of_range={bad_id} max_id={max_id} vocab_limit={limit}")
    if not quiet:
        log("verify: PASS" if problems == 0 else f"verify: FAIL ({problems} problems)")
    return 1 if problems else 0


def main(argv=None) -> int:
    global CORPUS, SUMMARY
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, default=CORPUS, help="corpus directory (default: %(default)s)")
    parser.add_argument("--summary", type=Path, default=SUMMARY, help="public summary path (default: %(default)s)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--cutoff", help="ignore session events after this UTC time "
                   "(default: the previous build's cutoff, else now)")
    b.add_argument("--seed", type=int, default=SEED)
    b.add_argument("--bench-dir", type=Path, default=DEFAULT_BENCH,
                   help="knapcio bench/ directory with hardset.py and qeval_tasks.py")
    b.add_argument("--italian-dir", type=Path, help="Italian conversations (default: <corpus>/synthetic-it)")
    b.add_argument("--force", action="store_true", help="rebuild even when the manifest is frozen")
    n = sub.add_parser("add-native")
    n.add_argument("--gen-dir", type=Path, required=True, help="harness gen/ directory (<id>.npz + <id>.json)")
    n.add_argument("--source", default="r0_native")
    n.add_argument("--min-gen-tokens", type=int, default=1)
    n.add_argument("--allow-frozen", action="store_true",
                   help="add to a frozen manifest (the plan adds native windows after freezing)")
    np_ = sub.add_parser("native-prompts")
    np_.add_argument("--file", type=Path, default=REPO / "scripts/fidelity/native_prompts.json")
    sub.add_parser("freeze")
    sub.add_parser("verify")
    args = parser.parse_args(argv)
    CORPUS, SUMMARY = args.corpus.resolve(), args.summary.resolve()
    if args.cmd == "build" and args.italian_dir is None:
        args.italian_dir = CORPUS / "synthetic-it"
    commands = {"build": cmd_build, "add-native": cmd_add_native, "freeze": cmd_freeze,
                "native-prompts": cmd_native_prompts,
                "verify": lambda _args: verify()}
    return commands[args.cmd](args)


def guarded_main() -> int:
    """Run main; on an unexpected error print only its type and code locations.

    Exception messages can quote data (a key, a snippet), so they are never printed.
    """
    import traceback
    try:
        return main()
    except SystemExit:
        raise
    except BaseException as error:  # noqa: BLE001
        frames = traceback.extract_tb(error.__traceback__)
        where = " <- ".join(f"{Path(f.filename).name}:{f.lineno}" for f in reversed(frames[-6:]))
        print(f"error: {type(error).__name__} at {where}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(guarded_main())
