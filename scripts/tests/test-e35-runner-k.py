#!/usr/bin/env python3
"""Offline contract for the E35 runner verify-length candidate."""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest


REPO = Path(__file__).resolve().parents[2]
CANDIDATE = REPO / "scripts/node/experiments/e03/e35-runner-k"
RUNNER = CANDIDATE / "model_runner.py"
SPECULATOR = CANDIDATE / "speculator.py"
SCHEDULER = CANDIDATE / "adaptive_k_scheduler.py"
SCHEDULER_PARENT = REPO / "scripts/node/experiments/e03/draft-budget/adaptive_k_scheduler.py"
IMAGE = {RUNNER: "f232398c93cf3193166136e823343840a41ebde8669253f56ce63a0023930cb4",
         SPECULATOR: "1f6ff5ca9c8f38ff417aafd43bfa3116b5387bf0f7b58721acb2185781879836"}
MARKER = b"\n\n# --- E35 candidate addition."
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()  # noqa: E731


class FakeLogger:
    def __init__(self):
        self.records = []

    def info(self, fmt, *args):
        self.records.append(("info", fmt % args))

    def warning(self, fmt, *args):
        self.records.append(("warning", fmt % args))


def prefix(path):
    raw = path.read_bytes()
    return raw[:raw.index(MARKER)]


def addition(path):
    raw = path.read_bytes()
    return raw[raw.index(MARKER):].decode()


class Provenance(unittest.TestCase):
    def test_prefixes_patches_manifest_and_sums(self):
        manifest = json.loads((CANDIDATE / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "promoted_default")
        self.assertEqual((CANDIDATE / "policy.flag").read_bytes(), b"hybrid\n")
        entries = {entry["path"]: entry for entry in manifest["candidates"]}
        parents = {RUNNER: prefix(RUNNER), SPECULATOR: prefix(SPECULATOR),
                   SCHEDULER: SCHEDULER_PARENT.read_bytes()}
        for path, digest in IMAGE.items():
            self.assertEqual(hashlib.sha256(parents[path]).hexdigest(), digest, path.name)
        for candidate, parent_bytes in parents.items():
            entry = entries[str(candidate.relative_to(REPO))]
            patch = CANDIDATE / (candidate.stem + ".patch")
            self.assertEqual(entry["sha256"], sha(candidate))
            self.assertEqual(entry["patch_sha256"], sha(patch))
            self.assertEqual(entry["parent"]["sha256"], hashlib.sha256(parent_bytes).hexdigest())
            removed = [line for line in patch.read_text().splitlines()
                       if line.startswith("-") and not line.startswith("---")]
            self.assertEqual(removed, [], patch.name)
            with tempfile.TemporaryDirectory(prefix="e35-patch-") as temporary:
                rebuilt = Path(temporary) / candidate.name
                rebuilt.write_bytes(parent_bytes)
                result = subprocess.run(["patch", "-s", str(rebuilt), str(patch)],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(rebuilt.read_bytes(), candidate.read_bytes())
        sums = dict(line.split(None, 1)[::-1] for line in
                    (CANDIDATE / "SHA256SUMS").read_text().splitlines())
        self.assertEqual({name.strip() for name in sums},
                         {"adaptive_k_scheduler.py", "manifest.json", "model_runner.py", "speculator.py"})
        for name, digest in sums.items():
            self.assertEqual(sha(CANDIDATE / name.strip()), digest)

    def test_launcher_verifies_the_bundle(self):
        launcher = (REPO / "scripts/launcher/launch-glm53-tp4.sh").read_text()
        marker = "# The E35 runner verify-length candidate carries its own source manifest.\n"
        start = launcher.index("case ", launcher.index(marker))
        case = launcher[start:launcher.index("\nesac", start) + len("\nesac")]
        with tempfile.TemporaryDirectory(prefix="e35-launcher-") as temporary:
            deployed = Path(temporary) / "experiments/e03/e35-runner-k"
            deployed.mkdir(parents=True)
            for name in ("model_runner.py", "speculator.py", "adaptive_k_scheduler.py",
                         "manifest.json", "SHA256SUMS"):
                (deployed / name).write_bytes((CANDIDATE / name).read_bytes())
            env = dict(os.environ, ENV_DIR=temporary, DRY_RUN="0",
                       EXTRA_DOCKER_ENV="-v /x/e35-runner-k/model_runner.py:/x:ro")
            good = subprocess.run(["bash", "-c", case], env=env, capture_output=True, text=True)
            self.assertEqual(good.returncode, 0, good.stderr)
            (deployed / "model_runner.py").write_bytes(b"corrupt\n")
            bad = subprocess.run(["bash", "-c", case], env=env, capture_output=True, text=True)
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn("E35 runner-k source manifest failed", bad.stderr)


RETURN = REPO / "scripts/node/reference/operational-20260930-e31-mb.env"
MB_RETURN = REPO / "scripts/node/reference/operational-20260929-memory-bounded.env"
POLICY_MOUNT = " -v $HOME/tp4/experiments/e03/e35-runner-k/policy.flag:/tmp/glm53-e35-policy:ro"


class Overlay(unittest.TestCase):
    """The measured overlay applies to E31-MB, which the template reaches through its return."""
    def source(self, prelude=""):
        command = (f'source "$1"; source "$3" || exit 4; {prelude} before=$EXTRA_DOCKER_ENV; '
                   'source "$2" || exit 3; '
                   'printf "%s\\n--AFTER--\\n%s\\n" "$before" "$EXTRA_DOCKER_ENV"; '
                   'set | grep -c "^_E35_" || true')
        return subprocess.run(["bash", "-c", command, "e35", str(REPO / "cluster.env.example"),
                               str(CANDIDATE / "delta.env"), str(RETURN)],
                              capture_output=True, text=True)

    def test_swaps_the_scheduler_and_adds_two_mounts(self):
        result = self.source()
        self.assertEqual(result.returncode, 0, result.stderr)
        before, rest = result.stdout.split("\n--AFTER--\n")
        after, count = rest.strip().rsplit("\n", 1)
        self.assertEqual(count, "0")
        vllm = "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu"
        expected = (before.replace("/draft-budget/adaptive_k_scheduler.py", "/e35-runner-k/adaptive_k_scheduler.py")
                    + f" -v $HOME/tp4/experiments/e03/e35-runner-k/model_runner.py:{vllm}/model_runner.py:ro"
                    + f" -v $HOME/tp4/experiments/e03/e35-runner-k/speculator.py:{vllm}/spec_decode/dflash2/speculator.py:ro"
                    + " -e VLLM_E35_ENABLE=1 -e VLLM_E35_POLICY_FLAG=/tmp/glm53-e35-policy")
        self.assertEqual(after, expected)

    def test_default_is_the_measured_overlay_plus_the_policy_mount(self):
        # Since E36 the template reaches the E35 default through the E35 return.
        command = ('source "$1"; source "$4" || exit 5; default=$EXTRA_DOCKER_ENV; source "$1"; '
                   'source "$3" || exit 4; source "$2" || exit 3; '
                   'printf "%s\\n--MEASURED--\\n%s\\n" "$default" "$EXTRA_DOCKER_ENV"')
        result = subprocess.run(["bash", "-c", command, "e35", str(REPO / "cluster.env.example"),
                                 str(CANDIDATE / "delta.env"), str(RETURN),
                                 str(REPO / "scripts/node/reference/operational-20260930-e35.env")],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        default, measured = result.stdout.rstrip("\n").split("\n--MEASURED--\n")
        self.assertEqual(default, measured + POLICY_MOUNT)

    def test_returns_refuse_other_forms(self):
        for env in (RETURN, MB_RETURN):
            for prelude in ('source "$2";', 'EXTRA_DOCKER_ENV+=" -e VLLM_E35_ENABLE=1";',
                            'EXTRA_DOCKER_ENV=${EXTRA_DOCKER_ENV/policy.flag/other.flag};',
                            'EXTRA_DOCKER_ENV=${EXTRA_DOCKER_ENV/e35-runner-k\\/adaptive/draft-budget\\/adaptive};'):
                result = subprocess.run(
                    ["bash", "-c", f'source "$1"; {prelude} source "$2"', "e35",
                     str(REPO / "cluster.env.example"), str(env)], capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0, (env.name, prelude))
            clean = subprocess.run(
                ["bash", "-c", 'source "$1"; source "$2" || exit 3; set | grep -c "^_E3R_\\|^_MBR_" || true; '
                 'declare -F | grep -c _MBR_ || true', "e35", str(REPO / "cluster.env.example"), str(env)],
                capture_output=True, text=True)
            self.assertEqual(clean.returncode, 0, clean.stderr)
            self.assertEqual(clean.stdout.split(), ["0", "0"], env.name)

    def test_refusals(self):
        self.assertNotEqual(self.source('source "$2";').returncode, 0)
        on_default = subprocess.run(["bash", "-c", 'source "$1"; source "$2"', "e35",
                                     str(REPO / "cluster.env.example"), str(CANDIDATE / "delta.env")],
                                    capture_output=True, text=True)
        self.assertNotEqual(on_default.returncode, 0)
        for extra in (" --volume /x:/opt/tp4/adaptive_k_scheduler.py", " -e VLLM_E35_ENABLE=1",
                      " -e VLLM_E34_POLICY_FLAG=/x", " --env-file /x",
                      " -v /x:/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/model_runner.py:ro",
                      " -v /other:/opt/tp4/adaptive_k_scheduler.py:ro"):
            result = subprocess.run(
                ["bash", "-c", 'source "$1"; source "$4" || exit 4; EXTRA_DOCKER_ENV+="$3"; source "$2"',
                 "e35", str(REPO / "cluster.env.example"), str(CANDIDATE / "delta.env"), extra,
                 str(RETURN)],
                capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0, extra)
        self.assertNotEqual(self.source('EXTRA_DOCKER_ENV=${EXTRA_DOCKER_ENV/draft-budget/x};').returncode, 0)
        self.assertNotEqual(self.source('SPEC_TOKENS=5;').returncode, 0)
        self.assertNotEqual(self.source('EXTRA_DOCKER_ENV=${EXTRA_DOCKER_ENV/VLLM_ADAPTIVE_K_HI=7/VLLM_ADAPTIVE_K_HI=5};').returncode, 0)


# --- runner addition against stubs -------------------------------------------------------------

@dataclasses.dataclass
class FakeSchedulerOutput:
    num_scheduled_tokens: dict
    total_num_scheduled_tokens: int
    scheduled_spec_decode_tokens: dict
    other: str = "kept"


def decode_output(req="r", drafts=7, extra=None):
    counts = {req: drafts + 1}
    spec = {req: [-1] * drafts}
    if extra:
        counts[extra] = 1
    return FakeSchedulerOutput(counts, sum(counts.values()), spec)


class FakeEvent:
    def __init__(self, done=True):
        self.done, self.synced = done, 0

    def query(self):
        return self.done

    def synchronize(self):
        self.synced += 1
        self.done = True


def load_runner(enabled="1", flag_path="", rank=0, broadcast_value=None):
    log = FakeLogger()
    calls, broadcasts = [], []

    class GPUModelRunner:
        def execute_model(self, scheduler_output, *args, **kwargs):
            calls.append((scheduler_output, args, kwargs))
            return "output"

    class Buffer(list):
        pass

    def zeros(n, dtype=None):
        return Buffer([0] * n)

    def broadcast(buffer, src, group):
        broadcasts.append((list(buffer), src, group))
        if broadcast_value is not None and rank != 0:
            buffer[0] = broadcast_value

    torch = SimpleNamespace(zeros=zeros, int32="int32",
                            distributed=SimpleNamespace(broadcast=broadcast))
    parallel = ModuleType("vllm.distributed.parallel_state")
    parallel.get_tp_group = lambda: SimpleNamespace(rank_in_group=rank, ranks=[10, 11, 12, 13],
                                                    cpu_group="cpu")
    stubs = {"vllm": ModuleType("vllm"), "vllm.distributed": ModuleType("vllm.distributed"),
             "vllm.distributed.parallel_state": parallel}
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    env = {k: os.environ.get(k) for k in ("VLLM_E35_ENABLE", "VLLM_E35_POLICY_FLAG")}
    os.environ["VLLM_E35_ENABLE"], os.environ["VLLM_E35_POLICY_FLAG"] = enabled, flag_path
    ns = {"GPUModelRunner": GPUModelRunner, "torch": torch, "init_logger": lambda name: log,
          "__name__": "e35runner"}
    try:
        exec(compile(addition(RUNNER), str(RUNNER), "exec"), ns)
    finally:
        for key, value in env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    ns["_restore"] = saved
    return ns, GPUModelRunner, log, calls, broadcasts


def restore(saved):
    for name, module in saved.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


class RunnerLogic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns, *_ = load_runner()
        restore(cls.ns["_restore"])

    def test_policy_participation_and_trim(self):
        parse = self.ns["e35_parse_policy"]
        for raw, want in ((b"lag1", "lag1"), (b"hybrid\n", "hybrid"), (b"ema", "ema"), (None, "ema"),
                          (b"LAG1", "ema"), (b"x" * 65, "ema"), (b"\xff", "ema")):
            self.assertEqual(parse(raw), want, raw)
        part = self.ns["e35_participating_request"]
        self.assertEqual(part(decode_output()), "r")
        self.assertIsNone(part(decode_output(drafts=3)))
        self.assertIsNone(part(decode_output(extra="s")))
        self.assertIsNone(part(FakeSchedulerOutput({"r": 900}, 900, {})))
        original = decode_output()
        trimmed = self.ns["e35_trim"](original, "r", 3)
        self.assertEqual((trimmed.num_scheduled_tokens, trimmed.total_num_scheduled_tokens,
                          trimmed.scheduled_spec_decode_tokens, trimmed.other),
                         ({"r": 4}, 4, {"r": [-1] * 3}, "kept"))
        self.assertEqual((original.num_scheduled_tokens, len(original.scheduled_spec_decode_tokens["r"])),
                         ({"r": 8}, 7))

    def test_rule_and_decisions(self):
        decide, margin = self.ns["e35_decide"], self.ns["e35_margin"]
        high, low = [0.999] * 7, [0.9, 0.5, 0.3, 0.1, 0.1, 0.1, 0.1]
        self.assertGreater(margin(high), 0)
        self.assertLess(margin(low), 0)
        never = lambda: self.fail("must not wait")  # noqa: E731
        self.assertEqual(decide("ema", high, never), (7, "ema"))
        self.assertEqual(decide("lag1", high, never), (7, "lag1"))
        self.assertEqual(decide("lag1", low, never), (3, "lag1"))
        self.assertEqual(decide("lag1", None, never), (7, "no_data"))
        self.assertEqual(decide("hybrid", high, never), (7, "lag1"))
        waited = []
        uncertain = next(c for c in ([x / 100] * 7 for x in range(50, 100))
                         if abs(margin(c)) <= self.ns["_E35_HYBRID_MARGIN"])
        self.assertEqual(decide("hybrid", uncertain, lambda: waited.append(1) or low), (3, "lag0"))
        self.assertEqual(waited, [1])
        self.assertEqual(decide("hybrid", None, lambda: None), (7, "no_data"))
        probs = self.ns["e35_predict"]([0.99] * 7)
        self.assertEqual(probs, sorted(probs, reverse=True))

    def test_confidence_uses_only_the_current_and_previous_draft(self):
        conf = self.ns["_e35_confidence"]
        current, previous = FakeEvent(done=False), FakeEvent(done=True)
        spec = SimpleNamespace(_e35_last_seq=10, _e35_store={
            "r": [(9, previous, [0.5] * 7), (10, current, [0.9] * 7)],
            "old": [(7, FakeEvent(), [0.1] * 7), (8, FakeEvent(), [0.2] * 7)]})
        self.assertEqual(conf(spec, "r", 1, wait=False), [0.5] * 7)
        self.assertIsNone(conf(spec, "r", 0, wait=False))           # not yet complete
        self.assertEqual(conf(spec, "r", 0, wait=True), [0.9] * 7)
        self.assertEqual(current.synced, 1)
        self.assertIsNone(conf(spec, "old", 1, wait=False))        # stale: never used
        gap = SimpleNamespace(_e35_last_seq=11, _e35_store={"a": [(9, FakeEvent(), [0.5] * 7),
                                                                (10, FakeEvent(), [0.9] * 7)]})
        self.assertIsNone(conf(gap, "a", 0, wait=False))           # another request's propose came last
        self.assertIsNone(conf(gap, "a", 1, wait=False))
        hole = SimpleNamespace(_e35_last_seq=10, _e35_store={"a": [(8, FakeEvent(), [0.5] * 7),
                                                                 (10, FakeEvent(), [0.9] * 7)]})
        self.assertIsNone(conf(hole, "a", 1, wait=False))          # its previous draft is older
        self.assertIsNone(conf(spec, "missing", 1, wait=False))
        self.assertIsNone(conf(SimpleNamespace(), "r", 1, wait=False))

    def run_wrapper(self, rank=0, flag="lag1", conf_prev=None, enabled="1", output=None,
                    broadcast_value=None, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            flag_path = Path(tmp) / "flag"
            if flag is not None:
                flag_path.write_text(flag)
            ns, cls, log, calls, broadcasts = load_runner(enabled, str(flag_path), rank, broadcast_value)
            try:
                runner = cls()
                store = {"r": [(1, FakeEvent(), conf_prev), (2, FakeEvent(False), [0.9] * 7)]} if conf_prev else {}
                runner.speculator = SimpleNamespace(_e35_store=store, _e35_last_seq=2)
                out = cls.execute_model(runner, output or decode_output(), **kwargs)
            finally:
                restore(ns["_restore"])
        return out, calls, broadcasts, log, runner

    def test_wrapper(self):
        low = [0.9, 0.5, 0.3, 0.1, 0.1, 0.1, 0.1]
        out, calls, bc, _, runner = self.run_wrapper(conf_prev=low)
        self.assertEqual(out, "output")
        self.assertEqual(calls[0][0].num_scheduled_tokens, {"r": 4})
        self.assertEqual(bc, [([3], 10, "cpu")])
        self.assertEqual(runner._e35_state.counters["lag1"], 1)
        out, calls, bc, _, _ = self.run_wrapper(conf_prev=low, flag="ema")
        self.assertEqual((calls[0][0].num_scheduled_tokens, bc), ({"r": 8}, [([7], 10, "cpu")]))
        out, calls, bc, _, runner = self.run_wrapper(rank=2, conf_prev=low, broadcast_value=3)
        self.assertEqual((calls[0][0].num_scheduled_tokens, len(bc)), ({"r": 4}, 1))
        self.assertEqual(runner._e35_state.broadcasts, [])            # telemetry on rank 0 only
        for kwargs in ({"enabled": "0"}, {"dummy_run": True}, {"output": decode_output(drafts=3)},
                       {"output": decode_output(extra="s")}):
            out, calls, bc, _, _ = self.run_wrapper(conf_prev=low, **kwargs)
            self.assertEqual(bc, [], kwargs)
        # A rank-0 failure still broadcasts k_hi, so no rank diverges.
        out, calls, bc, log, runner = self.run_wrapper(conf_prev=["bad"] * 7)
        self.assertEqual(bc, [([7], 10, "cpu")])
        self.assertEqual(calls[0][0].num_scheduled_tokens, {"r": 8})
        self.assertEqual(runner._e35_state.counters["errors"], 1)


# --- speculator addition and scheduler copy -----------------------------------------------------

class FakeTensor:
    """Nested lists; slicing returns a view whose copy_ writes into the parent, like torch."""

    def __init__(self, data, parent=None, start=0):
        self.data, self.parent, self.start = data, parent, start

    def __getitem__(self, key):
        if isinstance(key, slice):
            return FakeTensor(self.data[key], self, key.start or 0)
        return FakeTensor(self.data[key])

    def tolist(self):
        return self.data

    @property
    def shape(self):
        shape, data = [], self.data
        while isinstance(data, list):
            shape.append(len(data))
            data = data[0] if data else None
        return tuple(shape)

    def copy_(self, other, non_blocking=False):
        target = self.parent.data if self.parent is not None else self.data
        offset = self.start if self.parent is not None else 0
        for i, row in enumerate(other.data):
            target[offset + i] = list(row)
            self.data[i] = list(row)
        return self


def recorder_torch(fail_event=False):
    import math

    def softmax(x, dim=-1):
        out = []
        for request in x.data:
            rows = []
            for scores in request:
                top = max(scores)
                weights = [math.exp(s - top) for s in scores]
                rows.append([w / sum(weights) for w in weights])
            out.append(rows)
        return SimpleNamespace(amax=lambda dim=-1: FakeTensor([[max(r) for r in req] for req in out]))

    def event():
        if fail_event:
            raise RuntimeError("no event")
        return SimpleNamespace(record=lambda: None)

    return SimpleNamespace(
        float32="float32", softmax=softmax, nan_to_num=lambda t, nan=0.0: t,
        empty=lambda shape, dtype=None, pin_memory=False: FakeTensor(
            [[[0.0] * shape[2] for _ in range(shape[1])] for _ in range(shape[0])]),
        cuda=SimpleNamespace(is_current_stream_capturing=lambda: False, Event=event))


def load_recorder(flag_text, rank=0, fail_event=False):
    log = FakeLogger()

    class DFlash2Speculator:
        def propose(self, *args, **kwargs):
            return "draft"

    stubs = {"vllm": ModuleType("vllm"), "vllm.logger": ModuleType("vllm.logger"),
             "vllm.distributed": ModuleType("vllm.distributed"),
             "vllm.distributed.parallel_state": ModuleType("vllm.distributed.parallel_state")}
    stubs["vllm.logger"].init_logger = lambda name: log
    stubs["vllm.distributed.parallel_state"].get_tp_group = lambda: SimpleNamespace(rank_in_group=rank)
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    tmp = tempfile.TemporaryDirectory()
    flag = Path(tmp.name) / "flag"
    if flag_text is not None:
        flag.write_text(flag_text)
    env = {k: os.environ.get(k) for k in ("VLLM_E35_ENABLE", "VLLM_E35_POLICY_FLAG")}
    os.environ["VLLM_E35_ENABLE"], os.environ["VLLM_E35_POLICY_FLAG"] = "1", str(flag)
    ns = {"DFlash2Speculator": DFlash2Speculator, "torch": recorder_torch(fail_event),
          "__name__": "e35rec"}
    try:
        exec(compile(addition(SPECULATOR), str(SPECULATOR), "exec"), ns)
    finally:
        for key, value in env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return ns, DFlash2Speculator, log, saved, tmp


class Recorder(unittest.TestCase):
    def propose(self, cls, spec, **kwargs):
        batch = SimpleNamespace(num_reqs=1, req_ids=["r"])
        return cls.propose(spec, batch, "attn", "slots", "hidden", None, "s", "r", **kwargs)

    def make(self, cls):
        spec = cls()
        spec._selector_scores = FakeTensor([[[2.0, 0.0]] * 7])
        return spec

    def test_records_with_sequence_under_lag1_only(self):
        ns, cls, log, saved, tmp = load_recorder("lag1")
        try:
            spec = self.make(cls)
            for _ in range(3):
                self.assertEqual(self.propose(cls, spec), "draft")
            self.assertEqual(spec._e35_last_seq, 3)
            self.assertEqual([entry[0] for entry in spec._e35_store["r"]], [2, 3])
            self.assertEqual(len(spec._e35_store["r"][-1][2].tolist()), 7)
            self.propose(cls, spec, dummy_run=True)
            self.assertEqual(spec._e35_last_seq, 3)
        finally:
            restore(saved); tmp.cleanup()
        for flag, rank in (("ema", 0), (None, 0), ("lag1", 1)):
            ns, cls, log, saved, tmp = load_recorder(flag, rank=rank)
            try:
                spec = self.make(cls)
                self.propose(cls, spec)
                self.assertEqual(getattr(spec, "_e35_store", {}), {}, (flag, rank))
            finally:
                restore(saved); tmp.cleanup()

    def test_ema_interval_advances_the_sequence(self):
        ns, cls, log, saved, tmp = load_recorder("lag1")
        try:
            spec = self.make(cls)
            flag = Path(ns["_E35_POLICY_FLAG"])
            for policy in ("lag1", "lag1", "ema", "lag1"):
                flag.write_text(policy)
                spec.__dict__.get("_e35_state", {})["checked_at"] = None
                self.propose(cls, spec)
            self.assertEqual(spec._e35_last_seq, 4)
            # The pre-ema draft 2 is not adjacent to draft 4, so the runner finds no lag 1 entry.
            self.assertEqual([entry[0] for entry in spec._e35_store["r"]], [2, 4])
        finally:
            restore(saved); tmp.cleanup()

    def test_failed_recording_leaves_nothing_current(self):
        ns, cls, log, saved, tmp = load_recorder("hybrid", fail_event=True)
        try:
            spec = self.make(cls)
            self.propose(cls, spec)
            self.assertEqual(spec._e35_last_seq, 1)
            self.assertEqual(spec._e35_store, {})
            self.assertEqual(spec._e35_state["next"], 0)             # ring did not advance
            self.assertTrue(any("E35_CONF_RECORD_ERROR" in r for _, r in log.records))
        finally:
            restore(saved); tmp.cleanup()


class SpeculatorAndScheduler(unittest.TestCase):
    def test_remember_keeps_two_and_bounds(self):
        log = FakeLogger()

        class DFlash2Speculator:
            def propose(self, *args, **kwargs):
                return "draft"

        stubs = {"vllm": ModuleType("vllm"), "vllm.logger": ModuleType("vllm.logger")}
        stubs["vllm.logger"].init_logger = lambda name: log
        saved = {name: sys.modules.get(name) for name in stubs}
        sys.modules.update(stubs)
        ns = {"DFlash2Speculator": DFlash2Speculator, "torch": SimpleNamespace(), "__name__": "e35spec"}
        try:
            exec(compile(addition(SPECULATOR), str(SPECULATOR), "exec"), ns)
        finally:
            restore(saved)
        store = {}
        remember = ns["_e35_remember"]
        remember(store, 1, ["a", "b"], ["rowA1", "rowB1"], "e1", max_requests=2)
        remember(store, 2, ["a"], ["rowA2"], "e2", max_requests=2)
        remember(store, 3, ["a", "c"], ["rowA3", "rowC3"], "e3", max_requests=2)
        self.assertEqual(list(store), ["a", "c"])
        self.assertEqual([entry[0] for entry in store["a"]], [2, 3])
        self.assertEqual(store["a"][-1][2], "rowA3")

    def test_scheduler_force_and_additions(self):
        sys.dont_write_bytecode = True
        spec = importlib.util.spec_from_file_location("e35_sched", SCHEDULER)
        module = importlib.util.module_from_spec(spec)
        sys.modules["e35_sched"] = module
        spec.loader.exec_module(module)
        force = module.e35_force_single
        self.assertEqual(force({"r": 3}, "lag1", 7, 7, 1), {"r": module.placeholder_len(7, 7)})
        self.assertEqual(force({"r": 3}, "hybrid", 7, 7, 1), {"r": module.placeholder_len(7, 7)})
        self.assertEqual(force({"r": 3}, "ema", 7, 7, 1), {"r": 3})
        self.assertEqual(force({"r": 3, "s": 3}, "lag1", 7, 7, 2), {"r": 3, "s": 3})
        self.assertEqual(force({"r": 3}, "lag1", 7, 7, 2), {"r": 3})   # decode beside a prefill
        self.assertEqual(module.e35_parse_policy(b"hybrid"), "hybrid")
        self.assertEqual(module.e35_parse_policy(b"conf:10"), "ema")
        text = SCHEDULER.read_text()
        self.assertIn("lengths = e35_force_single(lengths, self._e35_switch.poll()", text)
        self.assertIn("len(scheduler_output.num_scheduled_tokens))", text)
        self.assertIn("E35_SCHEDULER_READY", text)


if __name__ == "__main__":
    unittest.main()
