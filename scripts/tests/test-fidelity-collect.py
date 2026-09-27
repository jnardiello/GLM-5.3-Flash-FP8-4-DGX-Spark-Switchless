#!/usr/bin/env python3
"""Offline tests for the fidelity collectors and memory sampler; stdlib only.

A localhost `http.server` fake vLLM returns prompt_logprobs / logprobs in the vLLM
formats, including a transient 503 and a 400. Tests cover parsing, retry/backoff,
resume, the abort file, replay salts and that no token text or raw salt is written.
Without numpy, array files are stored as JSON through the injectable save/validate
hooks; with numpy, the real npz path is exercised as well.
"""

from __future__ import annotations

import array
import contextlib
import hashlib
import http.server
import io
import json
import math
import os
from pathlib import Path
import socket
import stat
import sys
import tempfile
import threading
import unittest


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/fidelity"))
import fidelity_io as fio  # noqa: E402
import collect_generation as cg  # noqa: E402
import collect_prompt_logprobs as cp  # noqa: E402
import mem_sampler as ms  # noqa: E402

try:
    import numpy  # noqa: F401
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

K = 4
SECRET = "SECRETWORD"  # decoded_token text that must never reach disk or stdout


def rank_of(position: int) -> int:
    return 1 if position % 2 == 0 else K + 3  # odd positions: actual outside the top-K


def prompt_rows(prompt: list[int], drop_actual_at: int | None) -> list:
    rows = [None]
    for p in range(1, len(prompt)):
        actual = prompt[p]
        row = {}
        others = [(actual + 13 * (j + 1)) % 1000 for j in range(K + 2)]
        rank = rank_of(p)
        ranks = [r for r in range(1, K + 1) if r != rank]
        for tid, r in zip(others, ranks):
            row[str(tid)] = {"logprob": -0.5 * r, "rank": r, "decoded_token": f"token_id:{tid}"}
        if p != drop_actual_at:
            row[str(actual)] = {"logprob": -0.5 * rank, "rank": rank, "decoded_token": SECRET}
        rows.append(row)
    return rows


def gen_logprobs(prompt: list[int], max_tokens: int) -> dict:
    ids = [(prompt[-1] + 5 * i) % 1000 for i in range(min(6, max_tokens))]
    tops = []
    for tid in ids:
        tops.append({f"token_id:{tid}": -0.1, **{f"token_id:{(tid + j) % 1000}": -1.0 - j for j in range(1, K)}})
    return {"tokens": [f"token_id:{t}" for t in ids], "token_logprobs": [-0.1] * len(ids),
            "top_logprobs": tops, "text_offset": list(range(len(ids)))}


class FakeVLLM(http.server.BaseHTTPRequestHandler):
    state: dict = {}

    def log_message(self, *args):
        pass

    def _send(self, status: int, data: dict):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(200, {"object": "list", "data": [{"id": "glm-5.3-flash"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        state = self.state
        state["requests"].append(body)
        first = body["prompt"][0]
        plan = state["plan"].get(first, [])
        if plan:
            code = plan.pop(0)
            if code != 200:
                self._send(code, {"error": {"message": f"injected {code}", "code": code}})
                return
        prompt = body["prompt"]
        if "prompt_logprobs" in body:
            choice = {"index": 0, "text": "x", "finish_reason": "length",
                      "prompt_logprobs": prompt_rows(prompt, state["drop"].get(first))}
            usage = {"prompt_tokens": len(prompt), "completion_tokens": 1}
        else:
            lp = gen_logprobs(prompt, body["max_tokens"])
            choice = {"index": 0, "text": SECRET, "finish_reason": "stop", "logprobs": lp}
            usage = {"prompt_tokens": len(prompt), "completion_tokens": len(lp["tokens"])}
        self._send(200, {"id": "cmpl-1", "object": "text_completion", "choices": [choice], "usage": usage})


def write_tokens(path: Path, tokens: list[int]) -> str:
    values = array.array("I", tokens)
    if sys.byteorder == "big":
        values.byteswap()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = values.tobytes()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def write_manifest(root: Path, name: str, key: str, items: dict) -> Path:
    rows = []
    for index, (item_id, tokens) in enumerate(sorted(items.items())):
        rel = f"tokens/{item_id}.u32"
        rows.append({"id": item_id, "category": ["agentic_code", "italian_chat"][index % 2],
                     "source": "synthetic_it", "project": "test", "n_tokens": len(tokens),
                     "sha256": write_tokens(root / rel, tokens), "path": rel})
    path = root / name
    path.write_text(json.dumps({"schema": "fidelity-corpus/1", key: rows,
                                "global_sha256": fio.global_sha256(rows)}))
    return path


def json_save(path, data, schema):
    fio.write_json_atomic(path, {name: data[name] for name in schema})


def json_validate(path, schema, length, k):
    data = fio.read_json(path)
    if not data:
        return False
    n = len(data[next(iter(schema))]) if length is None else length
    return all(len(data[name]) == n for name in schema) and all(
        len(row) == k for name in schema if name.startswith("topk") for row in data[name])


class ServerCase(unittest.TestCase):
    def setUp(self):
        FakeVLLM.state = {"requests": [], "plan": {}, "drop": {}}
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeVLLM)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    def run_quiet(self, func, *args, **kwargs):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = func(*args, **kwargs)
        return code, buffer.getvalue()


class Parsing(unittest.TestCase):
    def test_token_id(self):
        self.assertEqual(fio.token_id(12), 12)
        self.assertEqual(fio.token_id("12"), 12)
        self.assertEqual(fio.token_id("token_id:154879"), 154879)
        self.assertEqual(fio.token_id("text", "token_id:7"), 7)
        with self.assertRaises(fio.ParseError):
            fio.token_id("text", "more text")

    def test_prompt_logprobs(self):
        ids = [5, 40, 41, 42, 43]
        rows = prompt_rows(ids, drop_actual_at=3)
        rows[4] = {int(k): v for k, v in rows[4].items()}  # int keys are accepted too
        parsed = fio.parse_prompt_logprobs({"choices": [{"prompt_logprobs": rows}]}, ids, K)
        self.assertEqual(parsed["missing_actual"], 1)
        self.assertTrue(math.isnan(parsed["lp_actual"][0]))
        self.assertEqual(parsed["topk_ids"][0], [-1] * K)
        self.assertEqual(parsed["topk_lp"][0], [-math.inf] * K)
        self.assertEqual(parsed["lp_actual"][2], -0.5)
        self.assertEqual(parsed["rank_actual"][2], 1)
        self.assertEqual(parsed["topk_ids"][2][0], 41)
        self.assertEqual(parsed["lp_actual"][1], -0.5 * (K + 3))  # outside the top-K, still known
        self.assertEqual(parsed["rank_actual"][1], K + 3)
        self.assertNotIn(40, parsed["topk_ids"][1])
        self.assertEqual(parsed["topk_lp"][1], [-0.5 * r for r in range(1, K + 1)])
        self.assertTrue(math.isnan(parsed["lp_actual"][3]))
        self.assertEqual(parsed["rank_actual"][3], -1)
        self.assertEqual(parsed["rank_actual"][4], 1)
        with self.assertRaises(fio.ParseError):
            fio.parse_prompt_logprobs({"choices": [{"prompt_logprobs": rows[:-1]}]}, ids, K)

    def test_generation_logprobs(self):
        parsed = fio.parse_generation_logprobs(
            {"choices": [{"logprobs": gen_logprobs([9], 3), "finish_reason": "length"}]}, K)
        self.assertEqual(parsed["gen_ids"], [9, 14, 19])
        self.assertEqual([row[0] for row in parsed["topk_ids"]], [9, 14, 19])
        self.assertEqual(parsed["topk_lp"][0], [-0.1, -2.0, -3.0, -4.0][:1] + [-2.0, -3.0, -4.0])
        self.assertEqual(parsed["finish_reason"], "length")
        with self.assertRaises(fio.ParseError):
            fio.parse_generation_logprobs({"choices": [{"text": "x"}]}, K)

    def test_redact_host(self):
        self.assertEqual(fio.redact_host("http://10.1.2.3:8000/"), "http://<host>:8000/")
        self.assertEqual(fio.redact_host("https://example.org"), "https://<host>")


class Retry(ServerCase):
    def test_503_then_success_and_400(self):
        FakeVLLM.state["plan"] = {1: [503, 503], 2: [400]}
        delays = []
        body = {"prompt": [1, 2], "prompt_logprobs": K, "max_tokens": 1}
        response, attempts = fio.post_with_retry(self.base + "/v1/completions", body,
                                                 retries=3, backoff=0.5, sleep=delays.append)
        self.assertEqual(attempts, 3)
        self.assertEqual(delays, [0.5, 1.0])
        self.assertIn("prompt_logprobs", response["choices"][0])
        with self.assertRaises(fio.ClientError) as caught:
            fio.post_with_retry(self.base + "/v1/completions", dict(body, prompt=[2, 3]),
                                retries=3, sleep=delays.append)
        self.assertEqual(caught.exception.status, 400)
        self.assertEqual(len(delays), 2)  # no retry on 4xx

    def test_connection_refused(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        delays = []
        with self.assertRaises(fio.TransientError):
            fio.post_with_retry(f"http://127.0.0.1:{port}/v1/completions", {}, retries=2,
                                backoff=1.0, timeout=2, sleep=delays.append)
        self.assertEqual(delays, [1.0, 2.0])


class PromptCollector(ServerCase):
    def setUp(self):
        super().setUp()
        self.windows = {"w0001": [1] + list(range(100, 111)), "w0002": [2] + list(range(200, 208)),
                        "w0003": [3] + list(range(300, 306))}
        self.manifest = write_manifest(self.root / "corpus", "manifest.json", "windows", self.windows)
        FakeVLLM.state["plan"] = {2: [503], 3: [400]}
        FakeVLLM.state["drop"] = {1: 3}

    def args(self, run="r1", *extra):
        return cp.parse_args(["--base-url", self.base, "--arm", "R0", "--run", run, "--K", str(K),
                              "--manifest", str(self.manifest), "--raw-root", str(self.root / "raw"),
                              "--backoff", "0", *extra])

    def collect(self, args, **kwargs):
        kwargs.setdefault("save", json_save)
        kwargs.setdefault("validate", json_validate)
        return self.run_quiet(cp.collect, args, sleep=lambda s: None, **kwargs)

    def test_collect_retry_resume_and_privacy(self):
        code, out = self.collect(self.args())
        self.assertEqual(code, 1)  # w0003 answered 400
        requests = FakeVLLM.state["requests"]
        self.assertEqual(len(requests), 4)  # w0001, w0002 (503 + ok), w0003 (400, no retry)
        for body in requests:
            self.assertEqual((body["max_tokens"], body["temperature"], body["prompt_logprobs"]), (1, 0, K))
            self.assertTrue(body["return_tokens_as_token_ids"])
            self.assertEqual(len(body["cache_salt"]), 64)
        self.assertEqual(len({b["cache_salt"] for b in requests}), 4)
        run_dir = self.root / "raw/R0/r1"
        w1 = json.loads((run_dir / "prompt/w0001.json").read_text())
        self.assertEqual((w1["status"], w1["http_attempts"], w1["missing_actual"], w1["prompt_tokens"], w1["K"]),
                         ("ok", 1, 1, 12, K))
        self.assertEqual(w1["cache_salt_sha256"], fio.salt_hash(requests[0]["cache_salt"]))
        w2 = json.loads((run_dir / "prompt/w0002.json").read_text())
        self.assertEqual((w2["status"], w2["http_attempts"]), ("ok", 2))
        # The retry after the 503 used a fresh salt; the sidecar hashes the last one.
        self.assertEqual(w2["cache_salt_sha256"], fio.salt_hash(requests[2]["cache_salt"]))
        w3 = json.loads((run_dir / "prompt/w0003.json").read_text())
        self.assertEqual((w3["status"], w3["http_status"]), ("client_error", 400))
        self.assertFalse((run_dir / "prompt/w0003.npz").exists())
        arrays = json.loads((run_dir / "prompt/w0001.npz").read_text())
        self.assertEqual(arrays["ids"], self.windows["w0001"])
        self.assertEqual(len(arrays["topk_ids"]), 12)
        run = json.loads((run_dir / "run.json").read_text())
        self.assertEqual((run["arm"], run["K"], run["server_model_id"]), ("R0", K, "glm-5.3-flash"))
        self.assertTrue(run["base_url"].startswith("http://<host>:"))
        self.assertIn("commit", run["harness_git"])
        written = "".join(p.read_text() for p in run_dir.rglob("*") if p.is_file()) + out
        self.assertNotIn(SECRET, written)
        for body in requests:
            self.assertNotIn(body["cache_salt"], written)
        self.assertIn("[1/3] w0001 n=12 status=ok", out)

        # Resume: complete windows are skipped, the failed one is retried.
        FakeVLLM.state["plan"] = {3: [400]}
        code, out = self.collect(self.args())
        self.assertEqual(code, 1)
        self.assertEqual(len(FakeVLLM.state["requests"]), 5)
        self.assertEqual(FakeVLLM.state["requests"][-1]["prompt"][0], 3)
        self.assertEqual(len(json.loads((run_dir / "run.json").read_text())["sessions"]), 2)
        # A corrupt array file is re-collected.
        (run_dir / "prompt/w0002.npz").write_text("{}")
        code, _ = self.collect(self.args())
        self.assertEqual(FakeVLLM.state["requests"][-2]["prompt"][0], 2)
        # A resumed run refuses a different K.
        with self.assertRaises(SystemExit):
            self.collect(cp.parse_args(["--base-url", self.base, "--arm", "R0", "--run", "r1", "--K", "8",
                                        "--manifest", str(self.manifest), "--raw-root", str(self.root / "raw")]))

    def test_abort_file(self):
        abort = self.root / "abort"
        abort.write_text("stop")
        code, out = self.collect(self.args("r2", "--abort-file", str(abort)))
        self.assertEqual(code, 3)
        self.assertEqual(FakeVLLM.state["requests"], [])
        self.assertIn("abort:", out)
        abort.unlink()
        calls = []

        def save(path, data, schema):
            json_save(path, data, schema)
            calls.append(path.name)
            if len(calls) == 1:
                abort.write_text("stop")  # appears while the first window is in flight

        code, out = self.collect(self.args("r2", "--abort-file", str(abort)), save=save)
        self.assertEqual((code, calls), (3, ["w0001.npz"]))

    def test_subset_limit_categories_and_manifest_check(self):
        subset = self.root / "subset.txt"
        subset.write_text("w0002\nw0003\n")
        FakeVLLM.state["plan"] = {}
        code, _ = self.collect(self.args("r3", "--subset", str(subset), "--limit", "1"))
        self.assertEqual((code, [b["prompt"][0] for b in FakeVLLM.state["requests"]]), (0, [2]))
        code, _ = self.collect(self.args("r4", "--categories", "italian_chat"))
        self.assertEqual(FakeVLLM.state["requests"][-1]["prompt"][0], 2)
        tokens = self.root / "corpus/tokens/w0001.u32"
        tokens.write_bytes(tokens.read_bytes()[:-4] + b"\x00\x00\x00\x00")
        before = len(FakeVLLM.state["requests"])
        code, _ = self.collect(self.args("r5", "--limit", "1"))
        self.assertEqual((code, len(FakeVLLM.state["requests"])), (1, before))
        side = json.loads((self.root / "raw/R0/r5/prompt/w0001.json").read_text())
        self.assertEqual(side["status"], "manifest_error")

    @unittest.skipUnless(HAVE_NUMPY, "numpy not installed; npz write not exercised")
    def test_npz_roundtrip(self):
        import numpy as np
        FakeVLLM.state["plan"] = {}
        code, _ = self.run_quiet(cp.collect, self.args("np"), sleep=lambda s: None)
        self.assertEqual(code, 0)
        path = self.root / "raw/R0/np/prompt/w0001.npz"
        with np.load(path) as data:
            self.assertEqual(data["ids"].dtype, np.int32)
            self.assertEqual(data["topk_ids"].shape, (12, K))
            self.assertEqual(data["topk_lp"].dtype, np.float32)
            self.assertTrue(np.isnan(data["lp_actual"][0]))
            self.assertTrue(np.isneginf(data["topk_lp"][0]).all())
            self.assertEqual(int(data["rank_actual"][1]), K + 3)
        self.assertTrue(fio.validate_npz(path, fio.PROMPT_ARRAYS, 12, K))
        self.assertFalse(fio.validate_npz(path, fio.PROMPT_ARRAYS, 13, K))
        count = len(FakeVLLM.state["requests"])
        self.run_quiet(cp.collect, self.args("np"), sleep=lambda s: None)
        self.assertEqual(len(FakeVLLM.state["requests"]), count)
        path.write_bytes(b"corrupt")
        self.run_quiet(cp.collect, self.args("np"), sleep=lambda s: None)
        self.assertEqual(len(FakeVLLM.state["requests"]), count + 1)


class GenerationCollector(ServerCase):
    def setUp(self):
        super().setUp()
        self.prompts = {"d001": [7, 8, 9], "d002": [4, 5, 6, 7]}
        self.manifest = write_manifest(self.root / "corpus", "decode_manifest.json", "prompts", self.prompts)

    def args(self, run, *extra):
        return cg.parse_args(["--base-url", self.base, "--arm", "Cp", "--run", run, "--K", str(K),
                              "--max-tokens", "5", "--manifest", str(self.manifest),
                              "--raw-root", str(self.root / "raw"), "--backoff", "0", *extra])

    def test_generation_and_replay_salt(self):
        FakeVLLM.state["plan"] = {7: [503]}
        salts = self.root / "private/salts.json"
        code, out = self.run_quiet(cg.collect, self.args("cold", "--replay-salt", str(salts)),
                                   save=json_save, validate=json_validate, sleep=lambda s: None)
        self.assertEqual(code, 0)
        code, _ = self.run_quiet(cg.collect, self.args("replay", "--replay-salt", str(salts)),
                                 save=json_save, validate=json_validate, sleep=lambda s: None)
        self.assertEqual(code, 0)
        by_prompt = {}
        for body in FakeVLLM.state["requests"]:
            self.assertEqual((body["logprobs"], body["max_tokens"], body["temperature"]), (K, 5, 0))
            by_prompt.setdefault(body["prompt"][0], set()).add(body["cache_salt"])
        self.assertEqual({k: len(v) for k, v in by_prompt.items()}, {7: 1, 4: 1})
        self.assertNotEqual(by_prompt[7], by_prompt[4])
        self.assertEqual(stat.S_IMODE(os.stat(salts).st_mode), 0o600)
        copy = self.root / "raw/Cp/cold/replay-salts.json"
        self.assertEqual(json.loads(copy.read_text()), json.loads(salts.read_text()))
        self.assertEqual(stat.S_IMODE(os.stat(copy).st_mode), 0o600)
        cold = json.loads((self.root / "raw/Cp/cold/gen/d001.json").read_text())
        replay = json.loads((self.root / "raw/Cp/replay/gen/d001.json").read_text())
        self.assertEqual(cold["cache_salt_sha256"], replay["cache_salt_sha256"])
        self.assertEqual((cold["finish_reason"], cold["gen_tokens"], cold["http_attempts"]), ("stop", 5, 2))
        arrays = json.loads((self.root / "raw/Cp/cold/gen/d001.npz").read_text())
        self.assertEqual(arrays["gen_ids"], [9, 14, 19, 24, 29])
        self.assertEqual(len(arrays["topk_lp"][0]), K)
        public = [p for p in (self.root / "raw").rglob("*") if p.is_file() and "salts" not in p.name]
        written = "".join(p.read_text() for p in public) + out
        self.assertNotIn(SECRET, written)
        for salt in json.loads(salts.read_text()).values():
            self.assertNotIn(salt, written)
        # Without --replay-salt, each request gets a fresh salt.
        self.run_quiet(cg.collect, self.args("fresh"), save=json_save, validate=json_validate,
                       sleep=lambda s: None)
        fresh = [b["cache_salt"] for b in FakeVLLM.state["requests"][-2:]]
        self.assertTrue(set(fresh).isdisjoint(json.loads(salts.read_text()).values()))


class MemSampler(unittest.TestCase):
    def test_sample_abort_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = root / "fake-ssh"
            fake.write_text("#!/bin/sh\n# args: -o BatchMode=yes HOST COMMAND\n"
                            "case \"$3\" in\n"
                            "  h0) echo 'MemAvailable:     900000 kB' ;;\n"
                            "  h1) echo 'MemAvailable:   20000000 kB' ;;\n"
                            "  *) exit 255 ;;\n"
                            "esac\n")
            fake.chmod(0o755)
            out, abort = root / "mem.jsonl", root / "abort"
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = ms.main(["--hosts", "h0", "h1", "h2", "--out", str(out), "--abort-file", str(abort),
                                "--ssh", str(fake), "--count", "2", "--interval", "0"])
            self.assertEqual(code, 2)
            self.assertTrue(abort.exists())
            records = [json.loads(line) for line in out.read_text().splitlines()]
            self.assertEqual(len(records), 6)
            self.assertEqual([r["rank"] for r in records[:3]], [0, 1, 2])
            self.assertEqual(records[0]["mem_available_kib"], 900000)
            self.assertIsNone(records[2]["mem_available_kib"])
            self.assertIn("ABORT", stderr.getvalue())
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                self.assertEqual(ms.main(["--summary", str(out)]), 0)
            self.assertIn("rank 0 h0: min MemAvailable 0.86 GiB", buffer.getvalue())
            self.assertIn("rank 1 h1", buffer.getvalue())
        self.assertEqual(ms.parse_meminfo("MemAvailable:   123 kB\n"), 123)
        self.assertIsNone(ms.parse_meminfo("garbage"))


if __name__ == "__main__":
    unittest.main(verbosity=1)
