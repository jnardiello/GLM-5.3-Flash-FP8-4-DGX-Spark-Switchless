#!/usr/bin/env python3
"""Offline contract tests for the resilience matrix and prompt synthesis."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/resilience"))
import campaign  # noqa: E402
import common  # noqa: E402


def fixed_response(prompt_tokens: int, *, finish_reason: str = "length",
                   completion_tokens: int = 16, status: int = 200) -> dict:
    return {
        "status": status,
        "body": {
            "choices": [{"finish_reason": finish_reason, "message": {"content": None}}],
            "usage": {"prompt_tokens": prompt_tokens,
                      "completion_tokens": completion_tokens},
        },
    }


def config(output: str) -> dict:
    campaign_id = "unit-campaign"
    name = common.namespace_name(campaign_id)
    repo = Path(output) / "checkout"
    bundle = repo / "scripts/resilience/.campaign" / campaign_id
    bundle.mkdir(parents=True, exist_ok=True)
    (repo / "scripts/tp4ctl").write_text("#!/bin/sh\n")
    (bundle / "delta.env").write_text("# canonical test overlay\n")
    (bundle / "delta-prefill-trim.env").write_text("# trim test overlay\n")
    (bundle / "delta-prefill-step-cap.env").write_text("# step-cap test overlay\n")
    parent = repo / campaign.DEFAULT_WORKER_SOURCE
    parent.parent.mkdir(parents=True, exist_ok=True)
    parent.write_text("# protected-default worker\n")
    parent_sha = hashlib.sha256(parent.read_bytes()).hexdigest()
    candidate = repo / "scripts/node/experiments/e03/prefill-cache-trim"
    candidate.mkdir(parents=True, exist_ok=True)
    (candidate / "gpu_worker.py").write_text("# trim worker\n")
    candidate_sha = hashlib.sha256((candidate / "gpu_worker.py").read_bytes()).hexdigest()
    (candidate / "manifest.json").write_text(json.dumps({
        "schema": "tp4-prefill-cache-trim-candidate-v1",
        "parent": {
            "path": "scripts/node/overrides/vllm/v1/worker/gpu_worker.py",
            "sha256": parent_sha,
        },
        "candidate": {
            "path": "scripts/node/experiments/e03/prefill-cache-trim/gpu_worker.py",
            "sha256": candidate_sha,
        },
    }))
    scheduler_parent = repo / campaign.DEFAULT_SCHEDULER_SOURCE
    scheduler_parent.parent.mkdir(parents=True, exist_ok=True)
    scheduler_parent.write_text("# end-drain scheduler\n")
    scheduler_parent_sha = hashlib.sha256(scheduler_parent.read_bytes()).hexdigest()
    step = repo / "scripts/node/experiments/e03/prefill-step-cap"
    step.mkdir(parents=True, exist_ok=True)
    (step / "scheduler.py").write_text("# capped scheduler\n")
    scheduler_sha = hashlib.sha256((step / "scheduler.py").read_bytes()).hexdigest()
    (step / "manifest.json").write_text(json.dumps({
        "schema": "tp4-prefill-step-cap-candidate-v1",
        "parent": {"path": campaign.DEFAULT_SCHEDULER_SOURCE,
                   "sha256": scheduler_parent_sha},
        "candidate": {"path": campaign.STEP_CAP_SCHEDULER_SOURCE,
                      "sha256": scheduler_sha,
                      "mount_target": campaign.SCHEDULER_TARGET},
        "selection": {"environment": campaign.STEP_CAP_ENVIRONMENT},
    }))
    admission = repo / "scripts/node/experiments/e03/bounded-admission"
    admission.mkdir(parents=True, exist_ok=True)
    (admission / "middleware.py").write_text("# bounded admission middleware\n")
    admission_sha = hashlib.sha256((admission / "middleware.py").read_bytes()).hexdigest()
    (admission / "manifest.json").write_text(json.dumps({
        "schema": "tp4-bounded-admission-candidate-v1",
        "candidate": {"path": campaign.ADMISSION_SOURCE,
                      "sha256": admission_sha,
                      "mount_target": campaign.ADMISSION_TARGET,
                      "import_string": campaign.ADMISSION_IMPORT},
        "selection": {"environment": campaign.ADMISSION_ENVIRONMENT},
        "guard": {
            "mode": campaign.ADMISSION_GUARD_MODE,
            "bypass_paths": sorted(campaign.ADMISSION_BYPASS_PATHS),
            "bypass_pattern": campaign.ADMISSION_BYPASS_PATTERN,
        },
    }))
    overlay_files = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (bundle / "delta.env", bundle / "delta-prefill-trim.env",
                     bundle / "delta-prefill-step-cap.env")
    }
    (bundle / "manifest.json").write_text(json.dumps({
        "base_connector_sha256": "a" * 64,
        "runtime_variant": "default",
        "selected_delta": "delta.env",
        "files": overlay_files,
    }))
    return {
        "campaign_id": campaign_id, "duration_hours": 8, "soak_hours": 2,
        "final_restore_minutes": 45, "endpoint": "http://rank0:8000/v1",
        "model": "glm-5.3-flash", "container": "glm53-tp4",
        "nodes": ["rank0", "rank1", "rank2", "rank3"],
        "host_cache_roots": [f"/srv/rank{i}/cache/jit/{name}" for i in range(4)],
        "container_cache_root": f"/cache/jit/{name}", "output_dir": output,
        "repo_root": str(repo),
        "tp4_env": f"scripts/resilience/.campaign/{campaign_id}/delta.env",
        "max_queue_clients": 256,
        "started_at_unix": time.time(),
        "rigmark": {"initial": str(Path(output) / "rigmark-initial.json"),
                    "final": str(Path(output) / "rigmark-final.json")},
    }


class ConfigTests(unittest.TestCase):
    def test_final_restore_target_requires_a_complete_relative_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = config(directory)
            raw["final_restore_env"] = "scripts/resilience/.campaign/unit/production.env"
            with self.assertRaisesRegex(common.ContractError, "configured together"):
                campaign.validate_config(raw)
            raw["final_operational_identity"] = "../identity.json"
            with self.assertRaisesRegex(common.ContractError, "repository-relative"):
                campaign.validate_config(raw)

    def test_deadline_extension_validation_rejects_invalid_values(self):
        for extension in (True, 0, -1, 13, float("nan"), float("inf")):
            with self.subTest(extension=extension), tempfile.TemporaryDirectory() as directory:
                raw = config(directory)
                raw["deadline_extension_hours"] = extension
                with self.assertRaisesRegex(common.ContractError,
                                            "deadline_extension_hours"):
                    campaign.validate_config(raw)

    def test_matrix_covers_required_areas_and_repetitions(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = campaign.validate_config(config(directory))
        plan = campaign.build_plan(cfg)
        contexts = [case for case in plan if case["area"] == "contexts"]
        ordinary = [case for case in contexts
                    if not case["parameters"].get("long_decode_pressure")]
        self.assertEqual(len(ordinary), 6 * 5 * 2)
        self.assertEqual({case["parameters"]["tokens"] for case in ordinary},
                         set(campaign.CONTEXT_TOKENS))
        pressure = [case for case in contexts
                    if case["id"] == campaign.LONG_DECODE_CASE_ID]
        self.assertEqual(len(pressure), 1)
        self.assertEqual(pressure[0]["parameters"], {
            "tokens": 245_760, "concurrency": 5, "cache": "cold",
            "max_tokens": 16_384, "min_tokens": 16_384,
            "ignore_eos": True, "long_decode_pressure": True,
        })
        self.assertEqual(pressure[0]["parameters"]["tokens"]
                         + pressure[0]["parameters"]["max_tokens"], 262_144)
        self.assertEqual(pressure[0]["target_repetitions"], 3)
        c5 = [case for case in contexts
              if case["parameters"]["tokens"] == 262_128
              and case["parameters"]["concurrency"] == 5]
        self.assertEqual({case["id"] for case in c5},
                         {"context-262128-c5-cold", "context-262128-c5-replay"})
        self.assertTrue(all(case["target_repetitions"] == 3 for case in c5))
        queues = [case["parameters"]["clients"] for case in plan if case["area"] == "queue"]
        self.assertEqual(queues, [8, 16, 32, 64, 128, 256])
        self.assertEqual({case["area"] for case in plan},
                         {"performance", "contexts", "queue", "api", "cancellation",
                          "slow_clients", "cache", "worker", "combinations", "duration"})
        for case in plan:
            self.assertEqual(case["target_repetitions"], 1 if case["area"] in {"performance", "duration"} else 3)
        soak = next(case for case in plan if case["area"] == "duration")
        self.assertEqual(soak["parameters"]["protocol_version"], campaign.SOAK_PROTOCOL_VERSION)

    def test_targeted_priority_reorders_only_targeted_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            base = config(directory)
            original = campaign.build_plan(campaign.validate_config(base))
            prioritized = copy.deepcopy(base)
            prioritized["targeted_case_priority"] = [
                "api-over-context", "cancel-restore", "queue-wave-64"]
            reordered = campaign.build_plan(campaign.validate_config(prioritized))
            self.assertEqual(reordered[0]["id"], "rigmark-initial")
            self.assertEqual([case["id"] for case in reordered[1:4]],
                             prioritized["targeted_case_priority"])
            self.assertEqual([case["id"] for case in reordered[-2:]],
                             ["soak-mixed", "rigmark-final"])
            self.assertCountEqual([case["id"] for case in reordered],
                                  [case["id"] for case in original])
            expected_remaining = [case["id"] for case in original[1:-2]
                                  if case["id"] not in
                                  prioritized["targeted_case_priority"]]
            self.assertEqual([case["id"] for case in reordered[4:-2]],
                             expected_remaining)

            duplicate = copy.deepcopy(base)
            duplicate["targeted_case_priority"] = ["api-over-context"] * 2
            with self.assertRaises(common.ContractError):
                campaign.validate_config(duplicate)
            for case_id in ("unknown-case", "rigmark-initial", "soak-mixed",
                            "rigmark-final"):
                invalid = copy.deepcopy(base)
                invalid["targeted_case_priority"] = [case_id]
                with self.subTest(case_id=case_id), self.assertRaises(common.ContractError):
                    campaign.build_plan(campaign.validate_config(invalid))

            combination = copy.deepcopy(base)
            combination["targeted_case_priority"] = [
                "cache-eio", "cancel-restore",
                "combination-cache-eio-cancel-restore"]
            combination_plan = campaign.build_plan(campaign.validate_config(combination))
            self.assertEqual([case["id"] for case in combination_plan[1:4]],
                             combination["targeted_case_priority"])
            premature = copy.deepcopy(base)
            premature["targeted_case_priority"] = [
                "combination-cache-eio-cancel-restore"]
            with self.assertRaisesRegex(common.ContractError, "before its prerequisites"):
                campaign.build_plan(campaign.validate_config(premature))

            c5_first = copy.deepcopy(base)
            c5_first["targeted_case_priority"] = [
                "context-262128-c5-cold", "context-262128-c5-replay"]
            c5_plan = campaign.build_plan(campaign.validate_config(c5_first))
            self.assertEqual([case["id"] for case in c5_plan[1:3]],
                             c5_first["targeted_case_priority"])

            pressure_first = copy.deepcopy(base)
            pressure_first["targeted_case_priority"] = [campaign.LONG_DECODE_CASE_ID]
            pressure_plan = campaign.build_plan(
                campaign.validate_config(pressure_first))
            self.assertEqual(pressure_plan[1]["id"], campaign.LONG_DECODE_CASE_ID)

    def test_targeted_selection_is_nonempty_distinct_and_automatic(self):
        with tempfile.TemporaryDirectory() as directory:
            base = config(directory)
            selected = copy.deepcopy(base)
            selected["targeted_case_selection"] = [
                "api-over-context", "cache-eio"]
            selected_cfg = campaign.validate_config(selected)
            selected_plan = campaign.build_plan(selected_cfg)
            base_plan = campaign.build_plan(campaign.validate_config(base))
            self.assertEqual([case["id"] for case in selected_plan],
                             [case["id"] for case in base_plan])
            self.assertEqual(selected_cfg["targeted_case_selection"],
                             selected["targeted_case_selection"])

            for invalid_selection in ([], "api-over-context", [""],
                                      ["api-over-context", "api-over-context"]):
                invalid = copy.deepcopy(base)
                invalid["targeted_case_selection"] = invalid_selection
                with self.subTest(selection=invalid_selection), \
                        self.assertRaisesRegex(common.ContractError,
                                               "targeted_case_selection"):
                    campaign.validate_config(invalid)
            for case_id in ("unknown-case", "rigmark-initial", "soak-mixed",
                            "rigmark-final"):
                invalid = copy.deepcopy(base)
                invalid["targeted_case_selection"] = [case_id]
                with self.subTest(case_id=case_id), \
                        self.assertRaisesRegex(common.ContractError,
                                               "targeted_case_selection"):
                    campaign.build_plan(campaign.validate_config(invalid))

    def test_early_soak_policy_is_strict_boolean(self):
        with tempfile.TemporaryDirectory() as directory:
            base = config(directory)
            self.assertFalse(campaign.validate_config(base)[
                "soak_stop_after_required_seconds"])
            enabled = copy.deepcopy(base)
            enabled["soak_stop_after_required_seconds"] = True
            self.assertTrue(campaign.validate_config(enabled)[
                "soak_stop_after_required_seconds"])
            for invalid in (0, 1, None, "true", []):
                bad = copy.deepcopy(base)
                bad["soak_stop_after_required_seconds"] = invalid
                with self.subTest(value=invalid), \
                        self.assertRaisesRegex(common.ContractError,
                                               "soak_stop_after_required_seconds"):
                    campaign.validate_config(bad)

    def test_bounded_admission_adds_one_repeated_short_overflow_case(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = config(directory)
            raw["bounded_admission"] = True
            plan = campaign.build_plan(campaign.validate_config(raw))
            prioritized = copy.deepcopy(raw)
            prioritized["targeted_case_priority"] = [
                "queue-wave-8", "admission-overflow"]
            ordered = campaign.build_plan(campaign.validate_config(prioritized))
        overflow = [case for case in plan if case["id"] == "admission-overflow"]
        self.assertEqual(len(overflow), 1)
        self.assertEqual(overflow[0]["area"], "queue")
        self.assertEqual(overflow[0]["target_repetitions"], 3)
        self.assertEqual(overflow[0]["parameters"], {
            "clients": (campaign.ADMISSION_MAX_ACTIVE
                        + campaign.ADMISSION_MAX_QUEUED
                        + campaign.ADMISSION_OVERFLOW_EXTRA),
            "mixed": False, "admission_overflow": True})
        self.assertEqual([case["id"] for case in ordered[1:3]],
                         ["queue-wave-8", "admission-overflow"])

    def test_invalid_deadlines_namespace_and_queue_cap_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            base = config(directory)
            for key, value in (("duration_hours", 7), ("soak_hours", 1),
                               ("final_restore_minutes", 30)):
                bad = copy.deepcopy(base); bad[key] = value
                with self.subTest(key=key), self.assertRaises(common.ContractError):
                    campaign.validate_config(bad)
            bad = copy.deepcopy(base); bad["host_cache_roots"][0] += "-wrong"
            with self.assertRaises(common.ContractError):
                campaign.validate_config(bad)
            bad = copy.deepcopy(base); bad["max_queue_clients"] = 96
            with self.assertRaises(common.ContractError):
                campaign.build_plan(campaign.validate_config(bad))
            for invalid_kv in (None, True, False, 0, -1, 1.5, "15032385536"):
                bad = copy.deepcopy(base)
                bad["kv_cache_memory_bytes"] = invalid_kv
                with self.subTest(kv_cache_memory_bytes=invalid_kv), \
                        self.assertRaisesRegex(common.ContractError, "positive integer"):
                    campaign.validate_config(bad)
            selected = copy.deepcopy(base)
            selected["kv_cache_memory_bytes"] = 15_032_385_536
            self.assertEqual(campaign.validate_config(selected)["kv_cache_memory_bytes"],
                             15_032_385_536)
            admission = copy.deepcopy(base)
            admission["bounded_admission"] = True
            self.assertTrue(campaign.validate_config(admission)["bounded_admission"])
            for invalid in (False, None, 1, "true", {}):
                bad = copy.deepcopy(base)
                bad["bounded_admission"] = invalid
                with self.subTest(bounded_admission=invalid), self.assertRaisesRegex(
                        common.ContractError, "bounded_admission"):
                    campaign.validate_config(bad)

    def test_runtime_variant_accepts_only_its_exact_namespace_overlay(self):
        with tempfile.TemporaryDirectory() as directory:
            base = config(directory)
            canonical = campaign.validate_config(base)
            self.assertEqual(canonical["runtime_variant"], "default")

            trim = copy.deepcopy(base)
            trim["runtime_variant"] = "prefill-cache-trim"
            trim["tp4_env"] = (
                "scripts/resilience/.campaign/unit-campaign/delta-prefill-trim.env")
            self.assertEqual(campaign.validate_config(trim)["runtime_variant"],
                             "prefill-cache-trim")
            step = copy.deepcopy(base)
            step["runtime_variant"] = "prefill-step-cap"
            step["tp4_env"] = (
                "scripts/resilience/.campaign/unit-campaign/delta-prefill-step-cap.env")
            self.assertEqual(campaign.validate_config(step)["runtime_variant"],
                             "prefill-step-cap")

            for variant, path in (
                    ("unknown", base["tp4_env"]),
                    ("default", trim["tp4_env"]),
                    ("prefill-cache-trim", base["tp4_env"]),
                    ("prefill-cache-trim", "scripts/resilience/delta-prefill-trim.env"),
                    ("prefill-step-cap", trim["tp4_env"]),
                    ("prefill-step-cap", base["tp4_env"])):
                bad = copy.deepcopy(base)
                bad.update(runtime_variant=variant, tp4_env=path)
                with self.subTest(variant=variant, path=path), self.assertRaises(
                        common.ContractError):
                    campaign.validate_config(bad)

    def test_example_has_no_real_site_values_and_plans_offline(self):
        example = REPO / "scripts/resilience/campaign.example.json"
        text = example.read_text()
        for needle in ("192.168.", "10.0.", "/Users/", "beast"):
            self.assertNotIn(needle, text)
        completed = __import__("subprocess").run(
            [sys.executable, str(REPO / "scripts/resilience/campaign.py"),
             "--config", str(example), "--plan"], text=True,
            stdout=__import__("subprocess").PIPE, stderr=__import__("subprocess").PIPE)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertGreater(json.loads(completed.stdout)["case_count"], 80)


class PromptTests(unittest.TestCase):
    def test_exact_prompt_is_verified_by_tokenizer(self):
        client = campaign.HTTPClient("http://rank0:8000/v1", "model")
        client.tokenize = lambda content: content.count(" x") + 7
        prompt = client.exact_prompt(123, "prefix")
        self.assertEqual(client.tokenize(prompt), 123)

    def test_tokenizer_uses_root_route_and_matched_reasoning_kwargs(self):
        client = campaign.HTTPClient("http://rank0:8000/v1", "model")
        seen = {}
        def fake(path, value, **kwargs):
            seen.update(path=path, value=value)
            return {"status": 200, "body": {"count": 17}}
        client.json = fake
        self.assertEqual(client.tokenize("hello"), 17)
        self.assertEqual(seen["path"], "/tokenize")
        self.assertEqual(seen["value"]["chat_template_kwargs"], {"reasoning_effort": "low"})

    def test_exact_messages_retains_history_and_fills_only_newest_user_turn(self):
        client = campaign.HTTPClient("http://rank0:8000/v1", "model")
        history = [{"role": "user", "content": "first"},
                   {"role": "assistant", "content": "actual answer"}]
        original = copy.deepcopy(history)
        client.tokenize_messages = lambda messages: 10 + messages[-1]["content"].count(" x")
        messages = client.exact_messages(25, history, "next")
        self.assertEqual(history, original)
        self.assertEqual(messages[:-1], history)
        self.assertEqual(messages[-1]["role"], "user")
        self.assertEqual(client.tokenize_messages(messages), 25)

    def test_sse_requires_finish_reason_and_reports_error_events(self):
        completed = campaign.parse_sse(
            b'data: {"choices":[{"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')
        self.assertTrue(completed["done"])
        self.assertEqual(completed["finish_reasons"], ["stop"])
        failed = campaign.parse_sse(
            b'event: error\ndata: {"error":{"message":"engine died"}}\n\ndata: [DONE]\n\n')
        self.assertTrue(failed["done"])
        self.assertTrue(failed["error_event"])
        self.assertEqual(failed["finish_reasons"], [])

    def test_abort_guard_shuts_down_a_blocked_transport_socket(self):
        client_socket, peer_socket = socket.socketpair()
        class Connection(campaign.http.client.HTTPConnection):
            def connect(self): self.sock = client_socket
        headers_sent = threading.Event()
        def server():
            request = b""
            while b"\r\n\r\n" not in request:
                request += peer_socket.recv(4096)
            peer_socket.sendall(
                b"HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: 100\r\n\r\n")
            headers_sent.set()
            peer_socket.recv(1)
        client = campaign.HTTPClient("http://rank0:8000/v1", "model", timeout=30)
        stopped = threading.Event()
        client.set_guard(stopped, lambda: time.monotonic() + 30)
        errors = []
        def invoke():
            try:
                client.raw("/blocked", b"{}")
            except Exception as error:
                errors.append(error)
        server_thread = threading.Thread(target=server)
        server_thread.start()
        with patch.object(campaign.http.client, "HTTPConnection", Connection):
            thread = threading.Thread(target=invoke)
            thread.start()
            self.assertTrue(headers_sent.wait(timeout=1))
            stopped.set()
            thread.join(timeout=1)
        client_socket.close()
        peer_socket.close()
        server_thread.join(timeout=1)
        self.assertFalse(thread.is_alive())
        self.assertTrue(errors)
        self.assertIsInstance(errors[0], campaign.GuardAbortError)
        self.assertEqual(errors[0].guard_abort_cause, "safety_event")

    def test_cancelable_request_retains_connection_close_transport(self):
        client_socket, peer_socket = socket.socketpair()
        class Connection(campaign.http.client.HTTPConnection):
            def connect(self): self.sock = client_socket
        headers_sent = threading.Event()
        def server():
            request = b""
            while b"\r\n\r\n" not in request:
                request += peer_socket.recv(4096)
            peer_socket.sendall(
                b"HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: 100\r\n\r\n")
            headers_sent.set()
            peer_socket.recv(1)
        client = campaign.HTTPClient("http://rank0:8000/v1", "model", timeout=30)
        client.set_guard(threading.Event(), lambda: time.monotonic() + 30)
        request = campaign.CancelableRequest(client, {"stream": True}, "cancel-test")
        server_thread = threading.Thread(target=server)
        server_thread.start()
        with patch.object(campaign.http.client, "HTTPConnection", Connection):
            request.start()
            self.assertTrue(headers_sent.wait(timeout=1))
            request.cancel()
            request.thread.join(timeout=1)
        client_socket.close()
        peer_socket.close()
        server_thread.join(timeout=1)
        self.assertFalse(request.thread.is_alive())
        self.assertIsNotNone(request.error)

    def test_cancel_before_connect_is_not_lost(self):
        client = campaign.HTTPClient("http://rank0:8000/v1", "model")
        request = campaign.CancelableRequest(client, {"stream": True}, "cancel-early")
        request.cancel()
        with patch.object(campaign.http.client, "HTTPConnection") as connection:
            request.start()
            request.thread.join(timeout=1)
        connection.assert_not_called()
        self.assertIn("cancelled before connect", request.error)

    def test_cancel_during_connect_is_sticky_until_transport_assignment(self):
        client_socket, peer_socket = socket.socketpair()
        entered = threading.Event()
        release = threading.Event()
        response_called = threading.Event()
        class Connection:
            def __init__(self, *_args, **_kwargs): self.sock = None
            def request(self, *_args, **_kwargs):
                entered.set()
                release.wait(timeout=1)
                self.sock = client_socket
            def getresponse(self):
                response_called.set()
                raise AssertionError("cancelled request reached getresponse")
            def close(self): pass
        client = campaign.HTTPClient("http://rank0:8000/v1", "model", timeout=30)
        client.set_guard(threading.Event(), lambda: time.monotonic() + 30)
        request = campaign.CancelableRequest(client, {"stream": True}, "cancel-connect")
        with patch.object(campaign.http.client, "HTTPConnection", Connection):
            request.start()
            self.assertTrue(entered.wait(timeout=1))
            request.cancel()
            release.set()
            request.thread.join(timeout=1)
        client_socket.close()
        peer_socket.close()
        self.assertFalse(request.thread.is_alive())
        self.assertFalse(response_called.is_set())
        self.assertIn("cancelled while connecting", request.error)

    def test_cancelable_exposes_http_status_and_headers_before_result_validation(self):
        class Response:
            status = 429
            def getheaders(self): return [("Retry-After", "3")]
            def read(self, size): return b"x" if size == 1 else b""
        class Connection:
            sock = None
            def __init__(self, *_args, **_kwargs): pass
            def request(self, *_args, **_kwargs): pass
            def getresponse(self): return Response()
            def close(self): pass
        client = campaign.HTTPClient("http://rank0:8000/v1", "model")
        request = campaign.CancelableRequest(client, {"stream": True}, "status-test")
        with patch.object(campaign.http.client, "HTTPConnection", Connection):
            request.start()
            request.thread.join(timeout=1)
        self.assertEqual(request.http_status, 429)
        self.assertEqual(request.response_headers["Retry-After"], "3")

    def test_cancelable_preserves_incomplete_read_bytes(self):
        class Response:
            status = 200
            calls = 0
            def getheaders(self): return [("Content-Type", "text/event-stream")]
            def read(self, size):
                self.calls += 1
                if self.calls == 1:
                    return b"d"
                raise campaign.http.client.IncompleteRead(b'ata: {"choices":[]}', 100)
        class Connection:
            sock = None
            def __init__(self, *_args, **_kwargs): pass
            def request(self, *_args, **_kwargs): pass
            def getresponse(self): return Response()
            def close(self): pass
        client = campaign.HTTPClient("http://rank0:8000/v1", "model")
        request = campaign.CancelableRequest(client, {"stream": True}, "partial-test")
        with patch.object(campaign.http.client, "HTTPConnection", Connection):
            request.start()
            request.thread.join(timeout=1)
        self.assertFalse(request.thread.is_alive())
        self.assertIn("IncompleteRead", request.error)
        self.assertEqual(request.result["body_text"], 'data: {"choices":[]}')
        self.assertEqual(request.result["received_bytes"], len(b'data: {"choices":[]}'))
        self.assertFalse(request.result["read_complete"])
        self.assertFalse(request.result["stream_complete"])

    def test_cancelable_preserves_prior_chunks_and_done_before_reset(self):
        tail = (b'ata: {"choices":[{"finish_reason":"stop"}]}\n\n'
                b'data: [DONE]\n\n')
        class Response:
            status = 200
            chunks = 0
            def getheaders(self): return [("Content-Type", "text/event-stream")]
            def read(self, size): return b"d"
            def read1(self, size):
                self.chunks += 1
                if self.chunks == 1:
                    return tail
                raise OSError("reset after completed event")
        class Connection:
            sock = None
            def __init__(self, *_args, **_kwargs): pass
            def request(self, *_args, **_kwargs): pass
            def getresponse(self): return Response()
            def close(self): pass
        client = campaign.HTTPClient("http://rank0:8000/v1", "model")
        request = campaign.CancelableRequest(client, {"stream": True}, "chunk-reset")
        with patch.object(campaign.http.client, "HTTPConnection", Connection):
            request.start()
            request.thread.join(timeout=1)
        self.assertFalse(request.thread.is_alive())
        self.assertIn("reset after completed event", request.error)
        self.assertEqual(request.result["body_text"].encode(), b"d" + tail)
        self.assertFalse(request.result["read_complete"])
        self.assertTrue(request.result["done"])
        self.assertTrue(request.result["stream_complete"])
        self.assertEqual(request.result["finish_reasons"], ["stop"])

    def test_raw_stop_during_connect_retries_after_transport_assignment(self):
        client_socket, peer_socket = socket.socketpair()
        entered = threading.Event()
        release = threading.Event()
        class Connection:
            def __init__(self, *_args, **_kwargs): self.sock = None
            def request(self, *_args, **_kwargs):
                entered.set()
                release.wait(timeout=1)
                self.sock = client_socket
            def getresponse(self):
                client_socket.recv(1)
                raise AssertionError("shutdown did not interrupt the transport")
            def close(self): pass
        stopped = threading.Event()
        client = campaign.HTTPClient("http://rank0:8000/v1", "model", timeout=30)
        client.set_guard(stopped, lambda: time.monotonic() + 30)
        errors = []
        def invoke():
            try:
                client.raw("/blocked-connect", b"{}")
            except Exception as error:
                errors.append(error)
        with patch.object(campaign.http.client, "HTTPConnection", Connection):
            thread = threading.Thread(target=invoke)
            thread.start()
            self.assertTrue(entered.wait(timeout=1))
            stopped.set()
            threading.Event().wait(0.06)
            release.set()
            thread.join(timeout=1)
        client_socket.close()
        peer_socket.close()
        self.assertFalse(thread.is_alive())
        self.assertTrue(errors)
        self.assertIsInstance(errors[0], campaign.GuardAbortError)
        self.assertEqual(errors[0].guard_abort_cause, "safety_event")

    def test_guarded_request_uses_the_remaining_phase_deadline(self):
        timeouts = []
        class Response:
            status = 200
            def read(self, _limit): return b"{}"
            def getheaders(self): return []
        class Connection:
            sock = None
            def __init__(self, _host, _port, timeout): timeouts.append(timeout)
            def request(self, *_args, **_kwargs): pass
            def getresponse(self): return Response()
            def close(self): pass
        client = campaign.HTTPClient("http://rank0:8000/v1", "model")
        client.set_guard(threading.Event(), lambda: time.monotonic() + 1800)
        with patch.object(campaign.http.client, "HTTPConnection", Connection):
            self.assertEqual(client.raw("/test", b"{}")["status"], 200)
        self.assertGreater(timeouts[0], 1700)

    def test_http_guard_watcher_start_failure_is_client_capacity(self):
        class Connection:
            sock = None
            def __init__(self, *_args, **_kwargs): self.closed = False
            def close(self): self.closed = True
        class Watcher:
            def start(self): raise RuntimeError("can't start new thread")
        client = campaign.HTTPClient("http://rank0:8000/v1", "model")
        client.set_guard(threading.Event(), lambda: time.monotonic() + 10)
        value = campaign.Campaign.__new__(campaign.Campaign)
        value.http = client
        with patch.object(campaign.http.client, "HTTPConnection", Connection), \
                patch.object(campaign.threading, "Thread", return_value=Watcher()):
            result = value._captured_chat("prompt", "watcher-capacity")
        self.assertEqual(result["error_origin"], "client_capacity")
        self.assertIn("ClientCapacityError", result["error"])

    def test_real_loopback_phase_deadline_is_owned_by_guard_watcher(self):
        for repetition in range(3):
            listener = socket.socket()
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            port = listener.getsockname()[1]
            server_errors = []
            def serve():
                try:
                    peer, _ = listener.accept()
                    with peer:
                        request = b""
                        while b"\r\n\r\n" not in request:
                            chunk = peer.recv(4096)
                            if not chunk:
                                return
                            request += chunk
                        while peer.recv(4096):
                            pass
                except Exception as error:
                    server_errors.append(error)
                finally:
                    listener.close()
            server = threading.Thread(target=serve)
            server.start()
            client = campaign.HTTPClient(
                f"http://127.0.0.1:{port}/v1", "model", timeout=30)
            deadline = time.monotonic() + 0.125
            client.set_guard(threading.Event(), lambda: deadline)
            started = time.monotonic()
            with self.assertRaises(campaign.GuardAbortError) as caught:
                client.raw("/stalled", b"{}")
            elapsed = time.monotonic() - started
            server.join(timeout=1)
            self.assertFalse(server.is_alive(), repetition)
            self.assertFalse(server_errors, server_errors)
            self.assertEqual(caught.exception.guard_abort_cause, "phase_deadline")
            self.assertGreaterEqual(elapsed, 0.10)
            self.assertLess(elapsed, 1.0)

    def test_private_event_writer_retries_short_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.jsonl"
            real_write = campaign.os.write
            calls = []
            def short_write(descriptor, payload):
                calls.append(len(payload))
                return real_write(descriptor, payload[:7])
            with patch.object(campaign.os, "write", side_effect=short_write):
                campaign._append_private_event(path, {
                    "kind": "unit", "payload": "x" * 100_000})
            lines = path.read_text().splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])["payload"], "x" * 100_000)
            self.assertGreater(len(calls), 100)


class FakeTelemetry:
    def __init__(self):
        self.safety_event = threading.Event()
        self.reasons = []
        self.latest = {}
        self.lock = threading.Lock()
        self.expected_down = False
        self.stopped = False

    def start(self, interval):
        self.interval = interval

    def stop(self):
        self.stopped = True

    def arm_expected_down(self, timeout):
        self.expected_down = True

    def clear_expected_down(self):
        self.expected_down = False


class TelemetryTests(unittest.TestCase):
    def test_start_uses_buffered_binary_stdout_and_preserves_pipe_bytes(self):
        raw = (b'{"time_ns":123,"monotonic_ns":456,"container_state":"running",'
               b'"container_pid":7}\n[]\nraw-\xff\n')
        with tempfile.TemporaryDirectory() as directory:
            helper = Path(directory) / "probe-writer.py"
            helper.write_text(
                "import os\n"
                f"os.write(1, {raw!r})\n")
            cfg = {"campaign_id": "unit", "container": "container",
                   "nodes": [f"rank{rank}" for rank in range(4)],
                   "host_cache_roots": [f"/cache/{rank}" for rank in range(4)],
                   "endpoint": "http://rank0:8000/v1",
                   "container_cache_root": "/cache/unit"}
            telemetry = campaign.Telemetry(cfg, Path(directory))
            with patch.object(campaign, "SSH", (sys.executable, str(helper))):
                telemetry.start(0.2)
                self.assertTrue(all(isinstance(process.stdout, io.BufferedReader)
                                    for process in telemetry.processes))
                deadline = time.monotonic() + 2
                paths = [Path(directory) / f"telemetry-rank{rank}.jsonl"
                         for rank in range(4)]
                while (time.monotonic() < deadline
                       and not all(path.is_file() and path.read_bytes() == raw
                                   for path in paths)):
                    time.sleep(0.01)
                self.assertTrue(all(path.read_bytes() == raw for path in paths))
                telemetry.stop()

    def test_reader_preserves_raw_bytes_and_persists_stale_receiver_boundaries(self):
        raw = (b'{"time_ns":123,"monotonic_ns":456,"container_state":"running",'
               b'"container_pid":7}\nnot-json-\xff\n')
        class Process:
            def __init__(self, payload, returncode):
                self.stdout = io.BytesIO(payload)
                self.returncode = returncode
            def poll(self): return self.returncode

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            telemetry = campaign.Telemetry({}, output)
            telemetry.started = time.monotonic() - 5
            telemetry.last_seen = {rank: telemetry.started for rank in range(4)}
            telemetry.receiver_state = {rank: {} for rank in range(4)}
            telemetry.processes = [Process(b"", rank) for rank in range(4)]
            telemetry.processes[0] = Process(raw, 0)
            telemetry._sample_reasons = lambda rank, sample, now: []
            telemetry.stop_event.set()
            telemetry._reader(0, telemetry.processes[0])
            self.assertEqual((output / "telemetry-rank0.jsonl").read_bytes(), raw)
            state = telemetry.receiver_state[0]
            self.assertIsInstance(state["receive_monotonic"], float)
            self.assertIsInstance(state["append_complete_monotonic"], float)
            self.assertLessEqual(state["receive_monotonic"],
                                 state["append_complete_monotonic"])
            self.assertEqual(state["source_time_ns"], 123)
            self.assertEqual(state["source_monotonic_ns"], 456)

            now = time.monotonic()
            telemetry._persist_receiver_diagnostic(now, [0, 2])
            diagnostic_path = output / "telemetry-receiver-diagnostics.jsonl"
            diagnostic = json.loads(diagnostic_path.read_text().splitlines()[-1])
            self.assertEqual(diagnostic["stale_ranks"], [0, 2])
            self.assertEqual([row["process_returncode"] for row in diagnostic["ranks"]],
                             [0, 1, 2, 3])
            self.assertEqual(diagnostic["ranks"][0]["source_time_ns"], 123)
            self.assertGreaterEqual(diagnostic["ranks"][2]["last_seen_age_seconds"], 5)
            self.assertEqual(diagnostic_path.stat().st_mode & 0o077, 0)

    def test_start_writes_each_ssh_stderr_to_a_private_file(self):
        created = []
        class Process:
            def __init__(self, command, **kwargs):
                self.command, self.kwargs = command, kwargs
                self.stdout = io.BytesIO()
                created.append(self)
            def terminate(self): pass
            def wait(self, timeout=None): return 0
            def poll(self): return None
        class Thread:
            def __init__(self, *args, **kwargs): pass
            def start(self): pass
            def join(self, timeout=None): pass
            def is_alive(self): return False

        with tempfile.TemporaryDirectory() as directory:
            cfg = {"campaign_id": "unit", "container": "container",
                   "nodes": [f"rank{rank}" for rank in range(4)],
                   "host_cache_roots": [f"/cache/{rank}" for rank in range(4)],
                   "endpoint": "http://rank0:8000/v1",
                   "container_cache_root": "/cache/unit"}
            telemetry = campaign.Telemetry(cfg, Path(directory))
            with patch.object(campaign.subprocess, "Popen", Process), \
                    patch.object(campaign.threading, "Thread", Thread):
                telemetry.start(0.2)
                streams = list(telemetry.stderr_streams)
                self.assertEqual(len(created), 4)
                self.assertTrue(all(item.kwargs["bufsize"] == -1 for item in created))
                self.assertTrue(all(item.kwargs["stderr"] is not campaign.subprocess.PIPE
                                    for item in created))
                for rank in range(4):
                    path = Path(directory) / f"telemetry-rank{rank}.stderr.log"
                    self.assertTrue(path.is_file())
                    self.assertEqual(path.stat().st_mode & 0o077, 0)
                telemetry.stop()
            self.assertTrue(all(stream.closed for stream in streams))

    def test_stop_does_not_close_stdout_while_its_reader_is_alive(self):
        class Process:
            def __init__(self): self.stdout = io.BytesIO(b"pending")
            def terminate(self): pass
            def wait(self, timeout=None): return 0
        class Reader:
            def join(self, timeout=None): pass
            def is_alive(self): return True

        telemetry = campaign.Telemetry({}, Path("/unused"))
        process, reader = Process(), Reader()
        stream = process.stdout
        telemetry.processes = [process]
        telemetry.reader_threads = [reader]
        telemetry.threads = [reader]
        telemetry.stop()
        self.assertFalse(stream.closed)
        stream.close()


class CampaignLifecycleTests(unittest.TestCase):
    def make_campaign(self, directory, runtime_variant="default",
                      kv_cache_memory_bytes=None, bounded_admission=False,
                      final_restore=False, deadline_extension_hours=None):
        raw = config(directory)
        raw["runtime_variant"] = runtime_variant
        if kv_cache_memory_bytes is not None:
            raw["kv_cache_memory_bytes"] = kv_cache_memory_bytes
        if bounded_admission:
            raw["bounded_admission"] = True
        if deadline_extension_hours is not None:
            raw["deadline_extension_hours"] = deadline_extension_hours
        if final_restore:
            final_env = Path(raw["repo_root"]) / "final-production.env"
            final_env.write_text("# final candidate\n")
            final_identity = Path(raw["repo_root"]) / "final-identity.json"
            final_identity.write_text(json.dumps({"identity_id": "final-unit"}))
            raw.update(final_restore_env="final-production.env",
                       final_operational_identity="final-identity.json")
        selected = campaign.RUNTIME_VARIANTS[runtime_variant]
        raw["tp4_env"] = f"scripts/resilience/.campaign/unit-campaign/{selected}"
        manifest_path = (Path(raw["repo_root"]) /
                         "scripts/resilience/.campaign/unit-campaign/manifest.json")
        manifest = json.loads(manifest_path.read_text())
        manifest.update(runtime_variant=runtime_variant, selected_delta=selected)
        if kv_cache_memory_bytes is not None:
            manifest["kv_cache_memory_bytes"] = kv_cache_memory_bytes
        if bounded_admission:
            source = Path(raw["repo_root"]) / campaign.ADMISSION_SOURCE
            staged = manifest_path.parent / campaign.ADMISSION_STAGED_NAME
            staged.write_bytes(source.read_bytes())
            manifest["files"][campaign.ADMISSION_STAGED_NAME] = hashlib.sha256(
                staged.read_bytes()).hexdigest()
            manifest["bounded_admission"] = campaign._admission_contract(raw)
        manifest_path.write_text(json.dumps(manifest))
        cfg = campaign.validate_config(raw)
        Path(cfg["rigmark"]["initial"]).write_text("{}")
        value = campaign.Campaign(cfg, acknowledge_faults=True)
        value.telemetry = FakeTelemetry()
        if final_restore:
            expected = value.final_restore_contract["restore_env_sha256"]
            value._ssh = lambda rank, *args, **kwargs: campaign.subprocess.CompletedProcess(
                args, 0, f"{expected}  tp4/final-production.env\n", "")
        value._reset_faults = lambda case_id: {"case_id": case_id, "ranks": {}, "errors": []}
        value.http.set_guard(value.telemetry.safety_event, lambda: value.active_deadline)
        return value

    def test_initial_native_receipt_must_be_fresh_regular_complete_json(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = campaign.validate_config(config(directory))
            receipt = Path(cfg["rigmark"]["initial"])
            with self.assertRaisesRegex(common.ContractError, "initial Rigmark receipt"):
                campaign.Campaign(cfg, acknowledge_faults=True)

            receipt.write_text('{"incomplete":')
            with self.assertRaisesRegex(common.ContractError, "complete JSON object"):
                campaign.Campaign(cfg, acknowledge_faults=True)

            receipt.write_text("{}")
            old = float(cfg["started_at_unix"]) - 10
            os.utime(receipt, (old, old))
            with self.assertRaisesRegex(common.ContractError, "predates"):
                campaign.Campaign(cfg, acknowledge_faults=True)

            receipt.unlink()
            target = Path(directory) / "real-initial.json"
            target.write_text("{}")
            receipt.symlink_to(target)
            with self.assertRaisesRegex(common.ContractError, "symlinked"):
                campaign.Campaign(cfg, acknowledge_faults=True)

    def test_explicit_deadline_extension_preserves_start_and_is_monotonic(self):
        with tempfile.TemporaryDirectory() as directory:
            first = self.make_campaign(directory)
            started_ns = first.state["started_ns"]
            original_deadline = first.state["deadline_ns"]
            first.save()

            extended = self.make_campaign(directory, deadline_extension_hours=6)
            self.assertEqual(extended.state["started_ns"], started_ns)
            self.assertEqual(extended.state["deadline_ns"] - original_deadline,
                             6 * 3600 * 1_000_000_000)
            history = extended.state["deadline_extension_history"]
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0]["old_deadline_ns"], original_deadline)
            self.assertEqual(history[0]["new_deadline_ns"],
                             extended.state["deadline_ns"])
            self.assertEqual(history[0]["reason"],
                             "explicit owner deadline extension")
            self.assertEqual(extended.state["limits"]["effective_deadline_hours"], 14)
            extended.save()

            repeated = self.make_campaign(directory, deadline_extension_hours=6)
            self.assertEqual(len(repeated.state["deadline_extension_history"]), 1)
            with self.assertRaisesRegex(common.ContractError, "cannot shrink"):
                self.make_campaign(directory, deadline_extension_hours=5)
            changed_base = config(directory)
            changed_base.update(duration_hours=9, deadline_extension_hours=6)
            changed_cfg = campaign.validate_config(changed_base)
            Path(changed_cfg["rigmark"]["initial"]).write_text("{}")
            with self.assertRaisesRegex(common.ContractError, "base duration differs"):
                campaign.Campaign(changed_cfg, acknowledge_faults=True)

    def test_native_receipt_records_valid_json_without_judging_result(self):
        with tempfile.TemporaryDirectory() as directory:
            value = self.make_campaign(directory)
            value.state["restoration_completed_ns"] = time.time_ns()
            final = Path(value.cfg["rigmark"]["final"])
            final.write_text('{"status":"FAIL"}')
            case = {"id": "rigmark-final", "area": "performance", "manual": True,
                    "target_repetitions": 1}
            status, evidence = value.execute_case(case, 1)
            self.assertEqual(status, "recorded")
            self.assertIn("without judging", evidence["note"])
            value.record(case, 1, status, evidence)
            identity = value.state["cases"]["rigmark-final"]["runs"][0][
                "runtime_config_identity"]
            self.assertEqual(identity["identity_scope"],
                             "protected-default/native-rigmark")
            self.assertEqual(identity["runtime_variant"], "default")
            self.assertIsNone(identity["tp4_env"])

    def test_default_runtime_does_not_require_the_experimental_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = campaign.validate_config(config(directory))
            (Path(cfg["repo_root"]) / campaign.PREFILL_TRIM_MANIFEST).unlink()
            (Path(cfg["repo_root"]) / campaign.ADMISSION_MANIFEST).unlink()
            Path(cfg["rigmark"]["initial"]).write_text("{}")
            value = campaign.Campaign(cfg, acknowledge_faults=True)
            self.assertIsNone(value.worker_contract["manifest_sha256"])
            self.assertIsNone(value.runtime_config_identity["candidate_worker_sha256"])
            self.assertNotIn("kv_cache_memory_bytes", value.runtime_config_identity)
            self.assertNotIn("bounded_admission", value.runtime_config_identity)
            bundle_manifest = (Path(cfg["repo_root"]) /
                               "scripts/resilience/.campaign/unit-campaign/manifest.json")
            manifest = json.loads(bundle_manifest.read_text())
            manifest["kv_cache_memory_bytes"] = 15_032_385_536
            bundle_manifest.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(RuntimeError, "generated manifest"):
                value.verify_overlay()

    def test_failure_restores_and_resume_requires_three_new_successes(self):
        with tempfile.TemporaryDirectory() as directory:
            final = Path(directory) / "rigmark-final.json"
            initial = Path(directory) / "rigmark-initial.json"
            initial.write_text("{}")
            final.write_text("{}")
            target = {"id": "target", "area": "contexts", "parameters": {},
                      "target_repetitions": 3}
            final_case = {"id": "rigmark-final", "area": "performance", "manual": True,
                          "target_repetitions": 1}
            initial_case = {"id": "rigmark-initial", "area": "performance", "manual": True,
                            "target_repetitions": 1}
            first = self.make_campaign(directory)
            first.plan = [initial_case, target, final_case]
            first.verify_overlay = lambda: {"pass": True}
            first.restore_default = lambda: {"gates": {"pass": True}, "identity": {"returncode": 0}}
            first.preserve_cluster_evidence = lambda label: {"status": "captured", "label": label}
            first.wait_for_drain = lambda after: ("pass", {})
            first.execute_case = lambda case, rep: (
                ("recorded", {}) if case.get("manual") else ("fail", {"defect": True}))
            self.assertEqual(first.run(), 1)
            self.assertTrue(first.telemetry.stopped)
            self.assertIn("restoration", first.state)
            first.state["targeted_cutoff_cleanup"] = {"errors": []}
            first.state["targeted_phase_cutoff_ns"] = 123
            first.state["unhandled_error"] = "old error"
            first.state["run_start_cleanup"] = {"attempt": "old"}
            first.state["overlay_preflight"] = {"attempt": "old"}
            first.state["pre_soak_cleanup"] = {"attempt": "old"}
            # Model the receipt written by the pre-identity driver. Preserve its
            # case bytes; the resumed attempt must not relabel those old runs.
            first.state.pop("runtime_config_identity", None)
            for row in first.state["cases"].values():
                for run in row["runs"]:
                    run.pop("runtime_config_identity", None)
            original_started = first.state["started_ns"]
            original_deadline = first.state["deadline_ns"]
            first.save()

            second = self.make_campaign(directory, "prefill-cache-trim")
            second.plan = [initial_case, target, final_case]
            second.verify_overlay = lambda: {"pass": True}
            second.restore_default = lambda: {"gates": {"pass": True}, "identity": {"returncode": 0}}
            second.preserve_cluster_evidence = lambda label: {"status": "captured", "label": label}
            second._drain_barriers = lambda: {rank: 1 for rank in range(4)}
            second.wait_for_drain = lambda after: ("pass", {})
            second.execute_case = lambda case, rep: (
                ("recorded", {}) if case.get("manual") else ("pass", {"corrected": True}))
            second.end_monotonic = time.monotonic()
            self.assertEqual(second.run(), 1)
            runs = second.state["cases"]["target"]["runs"]
            self.assertEqual([run["status"] for run in runs], ["fail", "pass", "pass", "pass"])
            self.assertEqual(second.state["cases"]["target"]["successful_repetitions"], 3)
            self.assertEqual(second.state["cases"]["target"]["status"], "pass_with_failures")
            self.assertEqual(second.state["status"], "complete_with_failures")
            self.assertEqual(second.state["started_ns"], original_started)
            self.assertEqual(second.state["deadline_ns"], original_deadline)
            self.assertNotIn("runtime_config_identity", runs[0])
            self.assertTrue(all(run["runtime_config_identity"]["runtime_variant"] ==
                                "prefill-cache-trim" for run in runs[1:]))
            self.assertEqual(second.state["runtime_config_identity"]["runtime_variant"],
                             "prefill-cache-trim")
            previous = second.state["previous_attempts"][-1]
            self.assertIn("restoration", previous)
            self.assertIn("restoration_completed_ns", previous)
            self.assertEqual(previous["status"], "incomplete")
            self.assertIn("final_fault_capture", previous)
            self.assertIn("targeted_cutoff_cleanup", previous)
            self.assertEqual(previous["targeted_phase_cutoff_ns"], 123)
            self.assertIn("stop_reasons", previous)
            self.assertEqual(previous["unhandled_error"], "old error")
            self.assertEqual(previous["run_start_cleanup"], {"attempt": "old"})
            self.assertEqual(previous["overlay_preflight"], {"attempt": "old"})
            self.assertEqual(previous["pre_soak_cleanup"], {"attempt": "old"})
            self.assertEqual(previous["planned_case_order"],
                             ["rigmark-initial", "target", "rigmark-final"])
            self.assertEqual(second.state["planned_case_order"],
                             ["rigmark-initial", "target", "rigmark-final"])
            self.assertFalse(previous["runtime_config_identity"]["identity_recorded"])
            self.assertEqual(previous["runtime_config_identity"]["runtime_variant"],
                             "default")

    def test_pass_pass_fail_requires_three_new_successes(self):
        runs = [{"status": status} for status in ("pass", "pass", "fail")]
        self.assertEqual(campaign.Campaign._successes_after_last_failure(runs, "pass"), 0)
        runs.extend({"status": "pass"} for _ in range(3))
        self.assertEqual(campaign.Campaign._successes_after_last_failure(runs, "pass"), 3)

        default = {"runtime_variant": "default"}
        trim = {"runtime_variant": "prefill-cache-trim"}
        variant_runs = [{"status": "pass", "runtime_config_identity": default} for _ in range(3)]
        self.assertEqual(campaign.Campaign._successes_after_last_failure(
            variant_runs, "pass", trim), 0)
        variant_runs.extend(
            {"status": "pass", "runtime_config_identity": trim} for _ in range(3))
        self.assertEqual(campaign.Campaign._successes_after_last_failure(
            variant_runs, "pass", trim), 3)
        duration_runs = [{"status": "pass", "runtime_config_identity": trim,
                          "evidence": {"protocol_version": 1}},
                         {"status": "pass", "runtime_config_identity": trim,
                          "evidence": {"protocol_version": campaign.SOAK_PROTOCOL_VERSION}}]
        self.assertEqual(campaign.Campaign._successes_after_last_failure(
            duration_runs, "pass", trim, campaign.SOAK_PROTOCOL_VERSION), 1)
        historical_cancel = [
            {"status": "pass", "runtime_config_identity": trim, "evidence": {}}
            for _ in range(3)]
        self.assertEqual(campaign.Campaign._successes_after_last_failure(
            historical_cancel, "pass", trim,
            campaign.TARGETED_CASE_PROTOCOL_VERSION), 0)
        historical_cancel.extend(
            {"status": "pass", "runtime_config_identity": trim,
             "evidence": {"protocol_version":
                          campaign.TARGETED_CASE_PROTOCOL_VERSION}}
            for _ in range(3))
        self.assertEqual(campaign.Campaign._successes_after_last_failure(
            historical_cancel, "pass", trim,
            campaign.TARGETED_CASE_PROTOCOL_VERSION), 3)

    def test_fixed_completion_requires_exact_usage_and_length_finish(self):
        self.assertTrue(campaign._valid_fixed_completion(fixed_response(32_000), 32_000))
        self.assertTrue(campaign._valid_fixed_completion(
            fixed_response(245_760, completion_tokens=16_384),
            245_760, 16_384))
        invalid = [
            fixed_response(31_999),
            fixed_response(32_000, finish_reason="stop"),
            fixed_response(32_000, completion_tokens=15),
            fixed_response(32_000, status=503),
            {"status": 200, "body": {"choices": [], "usage": {
                "prompt_tokens": 32_000, "completion_tokens": 16}}},
            {"status": 200, "body": {"error": {"message": "bad"},
                                      "choices": [{"finish_reason": "length"}],
                                      "usage": {"prompt_tokens": 32_000,
                                                "completion_tokens": 16}}},
        ]
        for response in invalid:
            with self.subTest(response=response):
                self.assertFalse(campaign._valid_fixed_completion(response, 32_000))

        self.assertFalse(campaign._valid_fixed_completion(
            fixed_response(245_760, finish_reason="stop", completion_tokens=8_000),
            245_760, 16_384))

    def test_long_decode_payload_is_forced_without_changing_default_payload(self):
        client = campaign.HTTPClient("http://rank0:8000/v1", "model")
        ordinary = client.chat_payload("ordinary")
        self.assertNotIn("min_tokens", ordinary)
        self.assertNotIn("ignore_eos", ordinary)
        self.assertEqual(ordinary["max_tokens"], 16)

        forced = client.chat_payload(
            "pressure", max_tokens=16_384, min_tokens=16_384,
            ignore_eos=True)
        self.assertEqual(forced["max_tokens"], 16_384)
        self.assertEqual(forced["min_tokens"], 16_384)
        self.assertIs(forced["ignore_eos"], True)

    def test_resume_requires_three_current_protocol_cancellation_successes(self):
        with tempfile.TemporaryDirectory() as directory:
            value = self.make_campaign(directory)
            target = {"id": "cancel-decode", "area": "cancellations",
                      "parameters": {"stage": "decode"}, "target_repetitions": 3}
            final = {"id": "rigmark-final", "area": "performance", "manual": True,
                     "target_repetitions": 1}
            old_runs = [{"repetition": index + 1, "status": "pass",
                         "runtime_config_identity": dict(value.runtime_config_identity),
                         "evidence": {"legacy": True}}
                        for index in range(3)]
            value.state["cases"][target["id"]] = {
                "area": target["area"], "parameters": target["parameters"],
                "target_repetitions": 3, "runs": old_runs}
            value.plan = [target, final]
            value.verify_overlay = lambda: {"pass": True}
            value._drain_barriers = lambda: {rank: rank for rank in range(4)}
            value.wait_for_drain = lambda _barriers: ("pass", {})
            value.restore_default = lambda: {
                "gates": {"pass": True}, "identity": {"returncode": 0}}
            value.execute_case = lambda case, repetition: (
                ("recorded", {}) if case.get("manual") else
                ("pass", {"protocol_version":
                          campaign.TARGETED_CASE_PROTOCOL_VERSION}))
            value.end_monotonic = time.monotonic()

            self.assertEqual(value.run(), 0)
            row = value.state["cases"][target["id"]]
            self.assertEqual(len(row["runs"]), 6)
            self.assertEqual(row["successful_repetitions"], 3)
            self.assertEqual([run["evidence"].get("protocol_version")
                              for run in row["runs"][-3:]],
                             [campaign.TARGETED_CASE_PROTOCOL_VERSION] * 3)

    def test_duration_record_preserves_historical_parameters_and_marks_active_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            value = campaign.Campaign.__new__(campaign.Campaign)
            value.state = {"cases": {"soak-mixed": {
                "area": "duration", "parameters": {"hours": 2},
                "target_repetitions": 1,
                "runs": [{"status": "pass", "evidence": {"protocol_version": 1}}],
            }}}
            value.runtime_config_identity = {"runtime_variant": "default"}
            value.native_rigmark_identity = {"runtime_variant": "default"}
            value.events = Path(directory) / "events.jsonl"
            value.save = lambda: None
            case = {"id": "soak-mixed", "area": "duration",
                    "parameters": {"hours": 2,
                                   "protocol_version": campaign.SOAK_PROTOCOL_VERSION},
                    "target_repetitions": 1}
            value.record(case, 1, "pending", {"protocol_version": campaign.SOAK_PROTOCOL_VERSION})
            row = value.state["cases"]["soak-mixed"]
            self.assertEqual(row["parameters"], {"hours": 2})
            self.assertEqual(row["active_parameters"]["protocol_version"],
                             campaign.SOAK_PROTOCOL_VERSION)
            self.assertEqual(row["runs"][0]["evidence"]["protocol_version"], 1)

    def test_pending_required_case_keeps_global_status_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            value = self.make_campaign(directory)
            target = {"id": "target", "area": "contexts", "parameters": {},
                      "target_repetitions": 1}
            initial = {"id": "rigmark-initial", "area": "performance", "manual": True,
                       "target_repetitions": 1}
            final = {"id": "rigmark-final", "area": "performance", "manual": True,
                     "target_repetitions": 1}
            value.plan = [initial, target, final]
            value.verify_overlay = lambda: {"pass": True}
            value.restore_default = lambda: {"gates": {"pass": True}, "identity": {"returncode": 0}}
            value.execute_case = lambda case, rep: (
                ("recorded", {}) if case.get("manual") else ("pending", {"reason": "unproven"}))
            value._drain_barriers = lambda: {rank: 1 for rank in range(4)}
            value.wait_for_drain = lambda barriers: ("pass", {})
            value.end_monotonic = time.monotonic()
            self.assertEqual(value.run(), 1)
            self.assertEqual(value.state["status"], "incomplete")
            self.assertEqual(value.state["cases"]["target"]["status"], "pending")

    def test_targeted_selection_skips_only_current_attempt_and_preserves_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            value = self.make_campaign(directory)
            value.cfg["targeted_case_selection"] = ["api-over-context"]
            initial = {"id": "rigmark-initial", "area": "performance",
                       "manual": True, "target_repetitions": 1}
            selected = {"id": "api-over-context", "area": "api",
                        "parameters": {"variant": "over-context"},
                        "target_repetitions": 3}
            completed = {"id": "api-invalid-parameters", "area": "api",
                         "parameters": {"variant": "invalid-parameters"},
                         "target_repetitions": 3}
            excluded = {"id": "api-malformed-json", "area": "api",
                        "parameters": {"variant": "malformed-json"},
                        "target_repetitions": 3}
            soak = {"id": "soak-mixed", "area": "duration",
                    "parameters": {"hours": 2,
                                   "protocol_version": campaign.SOAK_PROTOCOL_VERSION},
                    "target_repetitions": 1}
            final = {"id": "rigmark-final", "area": "performance",
                     "manual": True, "target_repetitions": 1}
            value.plan = [initial, selected, completed, excluded, soak, final]
            current_passes = lambda count, protocol=None: [{
                "repetition": index + 1, "status": "pass",
                "runtime_config_identity": dict(value.runtime_config_identity),
                "evidence": ({} if protocol is None else {
                    "protocol_version": protocol}),
            } for index in range(count)]
            for case in (selected, completed):
                value.state["cases"][case["id"]] = {
                    "area": case["area"], "parameters": case["parameters"],
                    "target_repetitions": 3, "runs": current_passes(3)}
            value.state["cases"][soak["id"]] = {
                "area": soak["area"], "parameters": soak["parameters"],
                "target_repetitions": 1,
                "runs": current_passes(1, campaign.SOAK_PROTOCOL_VERSION)}
            attempted = []
            value.verify_overlay = lambda: {"pass": True}
            value.execute_case = lambda case, repetition: (
                attempted.append(case["id"]) or ("recorded", {}))
            value.restore_default = lambda: {
                "gates": {"pass": True}, "identity": {"returncode": 0}}
            value.end_monotonic = time.monotonic()

            self.assertEqual(value.run(), 1)
            self.assertEqual(attempted, ["rigmark-initial", "rigmark-final"])
            self.assertEqual(value.state["execution_policy"], {
                "targeted_case_selection": ["api-over-context"],
                "soak_stop_after_required_seconds": False,
            })
            self.assertEqual(value.state["cases"][completed["id"]]["status"], "pass")
            excluded_row = value.state["cases"][excluded["id"]]
            self.assertEqual(excluded_row["status"], "pending")
            self.assertEqual(excluded_row["runs"], [])
            self.assertIn("not selected", excluded_row["pending_reason"])
            self.assertFalse(excluded_row["current_attempt_selection"]["selected"])

    def test_absent_targeted_selection_executes_the_complete_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            value = self.make_campaign(directory)
            initial = {"id": "rigmark-initial", "area": "performance",
                       "manual": True, "target_repetitions": 1}
            target = {"id": "api-over-context", "area": "api", "parameters": {},
                      "target_repetitions": 1}
            final = {"id": "rigmark-final", "area": "performance",
                     "manual": True, "target_repetitions": 1}
            value.plan = [initial, target, final]
            attempted = []
            value.verify_overlay = lambda: {"pass": True}
            value._drain_barriers = lambda: {rank: rank for rank in range(4)}
            value.wait_for_drain = lambda barriers: ("pass", {})
            def execute(case, repetition):
                attempted.append(case["id"])
                return (("recorded", {}) if case.get("manual") else ("pass", {}))
            value.execute_case = execute
            value.restore_default = lambda: {
                "gates": {"pass": True}, "identity": {"returncode": 0}}
            value.end_monotonic = time.monotonic()

            self.assertEqual(value.run(), 0)
            self.assertEqual(attempted,
                             ["rigmark-initial", "api-over-context", "rigmark-final"])
            self.assertEqual(value.state["execution_policy"], {
                "targeted_case_selection": None,
                "soak_stop_after_required_seconds": False,
            })

    def test_late_cutoff_cleanup_cannot_clear_an_existing_fatal_error(self):
        with tempfile.TemporaryDirectory() as directory:
            value = self.make_campaign(directory)
            target = {"id": "target", "area": "contexts", "parameters": {},
                      "target_repetitions": 1}
            final = {"id": "rigmark-final", "area": "performance", "manual": True,
                     "target_repetitions": 1}
            value.plan = [target, final]
            value.verify_overlay = lambda: (_ for _ in ()).throw(
                RuntimeError("preflight failed"))
            value.targeted_stop_at = time.monotonic() - 1
            value._reset_faults = lambda case_id: {"case_id": case_id, "errors": []}
            captures = []
            value.preserve_cluster_evidence = lambda label: captures.append(label) or {
                "status": "captured"}
            value.restore_default = lambda: {
                "gates": {"pass": True}, "identity": {"returncode": 0}}
            value.execute_case = lambda case, repetition: ("recorded", {})
            value.end_monotonic = time.monotonic()
            self.assertEqual(value.run(), 1)
            self.assertIn("campaign-stop-before-restoration", captures)
            self.assertFalse(value.state["overlay_preflight"]["pass"])

    def test_cutoff_cleanup_and_drain_complete_before_soak(self):
        with tempfile.TemporaryDirectory() as directory:
            value = self.make_campaign(directory)
            target = {"id": "target", "area": "contexts", "parameters": {},
                      "target_repetitions": 1}
            soak = {"id": "soak-mixed", "area": "duration",
                    "parameters": {"hours": 2,
                                   "protocol_version": campaign.SOAK_PROTOCOL_VERSION},
                    "target_repetitions": 1}
            final = {"id": "rigmark-final", "area": "performance", "manual": True,
                     "target_repetitions": 1}
            value.plan = [target, soak, final]
            now = time.monotonic()
            value.targeted_stop_at = now - 1
            value.soak_at = now + 60
            value.end_monotonic = now
            actions = []
            value.verify_overlay = lambda: {"pass": True}
            value._reset_faults = lambda case_id: actions.append(f"cleanup:{case_id}") or {
                "case_id": case_id, "errors": []}
            value.health_code = lambda: actions.append("health") or 200
            value.wait_for_api_idle = lambda: actions.append("idle") or ("pass", {})
            value._drain_barriers = lambda: actions.append("barriers") or {
                rank: 100 + rank for rank in range(4)}
            value.wait_for_drain = lambda barriers: actions.append("drain") or ("pass", {})
            def soak_run(case):
                actions.append("soak")
                self.assertTrue(value.state["targeted_cutoff_proof"]["pass"])
                self.assertEqual(value.active_deadline, value.soak_at)
                return "pending", {"protocol_version": campaign.SOAK_PROTOCOL_VERSION,
                                   "reason": "unit stop"}
            value._soak = soak_run
            value.execute_case = lambda case, repetition: (
                ("recorded", {}) if case.get("manual") else
                self.fail("targeted case started after cutoff"))
            value.restore_default = lambda: {
                "gates": {"pass": True}, "identity": {"returncode": 0}}
            self.assertEqual(value.run(), 1)
            self.assertLess(actions.index("cleanup:targeted-cutoff-cleanup"),
                            actions.index("idle"))
            self.assertLess(actions.index("idle"),
                            actions.index("barriers"))
            self.assertLess(actions.index("drain"), actions.index("cleanup:pre-soak"))
            self.assertLess(actions.index("cleanup:pre-soak"), actions.index("soak"))
            self.assertEqual(
                value.state["cases"]["soak-mixed"]["active_parameters"]["protocol_version"],
                campaign.SOAK_PROTOCOL_VERSION)

    def test_soak_exception_is_recorded_before_restoration(self):
        with tempfile.TemporaryDirectory() as directory:
            value = self.make_campaign(directory)
            initial = {"id": "rigmark-initial", "area": "performance", "manual": True,
                       "target_repetitions": 1}
            soak = {"id": "soak-mixed", "area": "duration", "parameters": {},
                    "target_repetitions": 1}
            final = {"id": "rigmark-final", "area": "performance", "manual": True,
                     "target_repetitions": 1}
            value.plan = [initial, soak, final]
            value.verify_overlay = lambda: {"pass": True}
            value.execute_case = lambda case, rep: ("recorded", {})
            value._soak = lambda case: (_ for _ in ()).throw(TimeoutError("deadline"))
            value.restore_default = lambda: {"gates": {"pass": True}, "identity": {"returncode": 0}}
            value.preserve_cluster_evidence = lambda label: {"status": "captured"}
            value.end_monotonic = time.monotonic()
            self.assertEqual(value.run(), 1)
            runs = value.state["cases"]["soak-mixed"]["runs"]
            self.assertEqual(runs[0]["status"], "fail")
            self.assertIn("TimeoutError", runs[0]["evidence"]["error"])

    def test_pre_soak_cleanup_failure_is_fatal_and_leaves_soak_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            value = self.make_campaign(directory)
            soak = {"id": "soak-mixed", "area": "duration", "parameters": {},
                    "target_repetitions": 1}
            final = {"id": "rigmark-final", "area": "performance", "manual": True,
                     "target_repetitions": 1}
            value.plan = [soak, final]
            value.verify_overlay = lambda: {"pass": True}
            value._reset_faults = lambda case_id: {
                "case_id": case_id,
                "errors": ["rank 2 restore failed"] if case_id == "pre-soak" else [],
            }
            value._soak = lambda case: self.fail("soak started after cleanup failure")
            value.execute_case = lambda case, repetition: ("recorded", {})
            value.restore_default = lambda: {
                "gates": {"pass": True}, "identity": {"returncode": 0}}
            value.preserve_cluster_evidence = lambda label: {"status": "captured"}
            value.end_monotonic = time.monotonic()
            self.assertEqual(value.run(), 1)
            run = value.state["cases"]["soak-mixed"]["runs"][0]
            self.assertEqual(run["status"], "pending")
            self.assertEqual(run["evidence"]["cleanup"]["errors"],
                             ["rank 2 restore failed"])

    def test_deadline_is_persisted_across_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            first = self.make_campaign(directory)
            self.assertAlmostEqual(first.restore_at - first.soak_at,
                                   2 * 3600 + campaign.SOAK_FINAL_PROOF_RESERVE_SECONDS,
                                   delta=0.01)
            self.assertAlmostEqual(first.soak_at - first.targeted_stop_at,
                                   campaign.TARGETED_STOP_MARGIN_SECONDS, delta=0.01)
            first.save()
            deadline = first.state["deadline_ns"]
            second = self.make_campaign(directory)
            self.assertEqual(second.state["deadline_ns"], deadline)
            self.assertLessEqual(second.end_monotonic - __import__("time").monotonic(), 8 * 3600)

    def test_active_attempt_refuses_a_different_runtime_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            first = self.make_campaign(directory)
            first.record({"id": "started", "area": "contexts", "parameters": {},
                          "target_repetitions": 1}, 1, "pending", {"reason": "active"})
            with self.assertRaisesRegex(common.ContractError, "runtime configuration differs"):
                self.make_campaign(directory, "prefill-cache-trim")

    def test_failed_restoration_timestamp_does_not_authorize_variant_change(self):
        with tempfile.TemporaryDirectory() as directory:
            first = self.make_campaign(directory)
            first.state.update(
                status="restoration_failed",
                restoration_completed_ns=time.time_ns(),
                restoration={"gates": {"pass": False},
                             "identity": {"returncode": 0}},
            )
            first.save()
            with self.assertRaisesRegex(common.ContractError, "runtime configuration differs"):
                self.make_campaign(directory, "prefill-cache-trim")

    def test_gates_parse_only_documented_response_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            value = self.make_campaign(directory)
            responses = iter([
                {"status": 200, "body": {"choices": [{"message": {"content": "Paris"}}], "note": "Rome"}},
                {"status": 200, "body": {"choices": [{"message": {"content": "get_weather Milan"}}]}},
            ])
            with patch.object(campaign.HTTPClient, "json", side_effect=lambda *a, **k: next(responses)):
                result = value._gates(time.monotonic())
            self.assertFalse(result["pass"])

    def test_controller_sets_exact_hosts_and_caps_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            value = self.make_campaign(directory)
            value.active_deadline = __import__("time").monotonic() + 2
            class Process:
                pid = 123
                returncode = 0
                stdout = io.StringIO("ok\n")
                def wait(self, timeout): return 0
            with patch("campaign.subprocess.Popen", return_value=Process()) as invoked:
                result = value._controller(True, "status", timeout=2700)
            self.assertEqual(result["returncode"], 0)
            self.assertEqual(invoked.call_args.kwargs["env"]["TP4_HOSTS"], "rank0 rank1 rank2 rank3")
            self.assertTrue(invoked.call_args.kwargs["start_new_session"])

    def test_controller_deadline_terminates_its_process_group(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            marker = root / "late-child"
            controller = root / "scripts/tp4ctl"
            controller.write_text(
                "#!/bin/sh\ntrap 'kill $child 2>/dev/null; wait $child 2>/dev/null; exit 124' TERM\n"
                f"(sleep 1; touch {marker}) &\nchild=$!\nwait $child\n")
            controller.chmod(0o755)
            value = self.make_campaign(directory)
            value.cfg["repo_root"] = str(root)
            value.active_deadline = time.monotonic() + 0.1
            result = value._controller(False, "status", timeout=10)
            time.sleep(1.1)
            self.assertIsNone(result["returncode"])
            self.assertFalse(marker.exists())

    def test_ssh_distinguishes_command_timeout_from_campaign_deadline(self):
        value = campaign.Campaign.__new__(campaign.Campaign)
        value.cfg = {"nodes": ["rank0", "rank1", "rank2", "rank3"]}
        timeout_error = campaign.subprocess.TimeoutExpired("ssh", 120)

        value.active_deadline = 1_000.0
        with patch.object(campaign.time, "monotonic", return_value=100.0), \
                patch.object(campaign.subprocess, "run", side_effect=timeout_error) as run:
            ordinary = value._ssh(0, "true", timeout=120)
        self.assertEqual(ordinary.returncode, 124)
        self.assertNotIn("phase deadline", ordinary.stderr)
        self.assertEqual(run.call_args.kwargs["timeout"], 120)

        value.active_deadline = 110.0
        clock = iter((100.0, 110.0))
        with patch.object(campaign.time, "monotonic", side_effect=lambda: next(clock)), \
                patch.object(campaign.subprocess, "run", side_effect=timeout_error) as run:
            phased = value._ssh(0, "true", timeout=120)
        self.assertEqual(phased.returncode, 124)
        self.assertIn("phase deadline", phased.stderr)
        self.assertEqual(run.call_args.kwargs["timeout"], 10)

    def test_overlay_preflight_proves_selected_worker_environment_mount_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            value = self.make_campaign(directory, "prefill-cache-trim")
            env = {
                "TP4_RESILIENCE_CAMPAIGN_ID": value.cfg["campaign_id"],
                "TP4_RESILIENCE_CACHE_ROOT": value.cfg["container_cache_root"],
                "TP4_RESILIENCE_MAX_BYTES": str(common.MAX_NAMESPACE_BYTES),
                "TP4_RESILIENCE_BASE_SHA256": "a" * 64,
                "SPARK_CONTEXT_CACHE_MAX_BYTES": str(campaign.TEST_CACHE_MAX_BYTES),
                "SPARK_CONTEXT_CACHE_LOW_WATERMARK_BYTES": str(
                    campaign.TEST_CACHE_LOW_WATERMARK_BYTES),
                "VLLM_PREFILL_CACHE_TRIM": "1",
            }
            inspected = json.dumps([{
                "Config": {"Env": [f"{key}={item}" for key, item in env.items()]},
                "Mounts": [
                    {"Destination": "/opt/tp4-resilience/base_connector.py",
                     "Source": "/private/base.py"},
                    {"Destination": "/opt/tp4-resilience/runtime.py",
                     "Source": "/private/runtime.py"},
                    {"Destination":
                     "/usr/local/lib/python3.12/dist-packages/sparkcache/"
                     "spark_context_cache_connector.py",
                     "Source": "/private/connector_wrapper.py"},
                    {"Destination": campaign.WORKER_TARGET,
                     "Source":
                     "/home/test/tp4/experiments/e03/prefill-cache-trim/gpu_worker.py"},
                ],
            }])
            worker_hash = value.worker_contract["candidate_sha256"]
            manifest_files = json.loads((Path(value.cfg["repo_root"]) /
                "scripts/resilience/.campaign/unit-campaign/manifest.json").read_text())["files"]
            def ssh(rank, *arguments):
                if "exec" in arguments:
                    output = worker_hash + "  " + campaign.WORKER_TARGET + "\n"
                elif arguments[0] == "sha256sum":
                    output = (manifest_files[Path(arguments[1]).name] + "  " +
                              arguments[1] + "\n")
                else:
                    output = inspected
                return type("Result", (), {"returncode": 0, "stdout": output})()
            value._ssh = ssh
            value._faultctl = lambda rank, args: {"within_cap": True, "used_bytes": 0}
            result = value.verify_overlay()
            self.assertTrue(result["pass"])
            self.assertEqual({row["runtime_variant"] for row in result["ranks"]},
                             {"prefill-cache-trim"})

            worker_hash = "d" * 64
            with self.assertRaisesRegex(RuntimeError, "in-container worker hash differs"):
                value.verify_overlay()

            default = self.make_campaign(str(Path(directory) / "default"))
            default_inspect = json.loads(inspected)
            default_inspect[0]["Config"]["Env"] = [
                item for item in default_inspect[0]["Config"]["Env"]
                if not item.startswith("VLLM_PREFILL_CACHE_TRIM=")]
            default_inspect[0]["Config"]["Env"].append("VLLM_PREFILL_CACHE_TRIM=1")
            default._ssh = lambda rank, *arguments: type(
                "Result", (), {"returncode": 0, "stdout": json.dumps(default_inspect)})()
            default._faultctl = lambda rank, args: {"within_cap": True, "used_bytes": 0}
            with self.assertRaisesRegex(RuntimeError,
                                        "runtime variant environment differs"):
                default.verify_overlay()

    def test_step_cap_preflight_proves_worker_scheduler_flag_mounts_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            kv_bytes = 15_032_385_536
            value = self.make_campaign(
                directory, "prefill-step-cap", kv_cache_memory_bytes=kv_bytes)
            env = {
                "TP4_RESILIENCE_CAMPAIGN_ID": value.cfg["campaign_id"],
                "TP4_RESILIENCE_CACHE_ROOT": value.cfg["container_cache_root"],
                "TP4_RESILIENCE_MAX_BYTES": str(common.MAX_NAMESPACE_BYTES),
                "TP4_RESILIENCE_BASE_SHA256": "a" * 64,
                "SPARK_CONTEXT_CACHE_MAX_BYTES": str(campaign.TEST_CACHE_MAX_BYTES),
                "SPARK_CONTEXT_CACHE_LOW_WATERMARK_BYTES": str(
                    campaign.TEST_CACHE_LOW_WATERMARK_BYTES),
                "VLLM_PREFILL_CACHE_TRIM": "1",
                "VLLM_RESILIENCE_STEP_TOKEN_CAP": "6912",
            }
            inspected = json.dumps([{
                "Config": {
                    "Env": [f"{key}={item}" for key, item in env.items()],
                    "Cmd": ["serve", "--max-model-len", "262144",
                            f"--kv-cache-memory-bytes={kv_bytes}"],
                },
                "Mounts": [
                    {"Destination": "/opt/tp4-resilience/base_connector.py",
                     "Source": "/private/base.py"},
                    {"Destination": "/opt/tp4-resilience/runtime.py",
                     "Source": "/private/runtime.py"},
                    {"Destination":
                     "/usr/local/lib/python3.12/dist-packages/sparkcache/"
                     "spark_context_cache_connector.py",
                     "Source": "/private/connector_wrapper.py"},
                    {"Destination": campaign.WORKER_TARGET,
                     "Source":
                     "/home/test/tp4/experiments/e03/prefill-cache-trim/gpu_worker.py"},
                    {"Destination": campaign.SCHEDULER_TARGET,
                     "Source":
                     "/home/test/tp4/experiments/e03/prefill-step-cap/scheduler.py"},
                ],
            }])
            manifest_files = json.loads((Path(value.cfg["repo_root"]) /
                "scripts/resilience/.campaign/unit-campaign/manifest.json").read_text())["files"]
            def ssh_for(inspect_payload, scheduler_digest=None):
                def ssh(rank, *arguments):
                    if "exec" in arguments:
                        target = arguments[-1]
                        digest = (value.worker_contract["candidate_sha256"]
                                  if target == campaign.WORKER_TARGET else
                                  scheduler_digest or
                                  value.scheduler_contract["candidate_sha256"])
                        output = f"{digest}  {target}\n"
                    elif arguments[0] == "sha256sum":
                        output = (f"{manifest_files[Path(arguments[1]).name]}  "
                                  f"{arguments[1]}\n")
                    else:
                        output = inspect_payload
                    return type("Result", (), {"returncode": 0, "stdout": output})()
                return ssh
            value._ssh = ssh_for(inspected)
            value._faultctl = lambda rank, args: {"within_cap": True, "used_bytes": 0}
            result = value.verify_overlay()
            self.assertTrue(result["pass"])
            self.assertEqual({row["runtime_variant"] for row in result["ranks"]},
                             {"prefill-step-cap"})
            self.assertEqual({row["scheduler_sha256"] for row in result["ranks"]},
                             {value.scheduler_contract["candidate_sha256"]})
            self.assertEqual(value.runtime_config_identity["kv_cache_memory_bytes"],
                             kv_bytes)
            self.assertEqual({row["kv_cache_memory_bytes"] for row in result["ranks"]},
                             {kv_bytes})
            self.assertEqual({row["max_model_len"] for row in result["ranks"]},
                             {262144})

            manifest_path = (Path(value.cfg["repo_root"]) /
                             "scripts/resilience/.campaign/unit-campaign/manifest.json")
            manifest = json.loads(manifest_path.read_text())
            manifest["kv_cache_memory_bytes"] = kv_bytes + 1
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(RuntimeError, "generated manifest"):
                value.verify_overlay()
            manifest.pop("kv_cache_memory_bytes")
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(RuntimeError, "generated manifest"):
                value.verify_overlay()
            manifest["kv_cache_memory_bytes"] = kv_bytes
            manifest_path.write_text(json.dumps(manifest))

            for mode in ("missing", "duplicate", "mismatch", "alias-equals",
                         "alias-pair", "max-model"):
                bad_command = json.loads(inspected)
                command = bad_command[0]["Config"]["Cmd"]
                if mode == "missing":
                    command.pop()
                elif mode == "duplicate":
                    command.append(f"--kv-cache-memory-bytes={kv_bytes}")
                elif mode == "mismatch":
                    command[-1] = f"--kv-cache-memory-bytes={kv_bytes + 1}"
                elif mode == "alias-equals":
                    command.append(f"--kv-cache-memory={kv_bytes}")
                elif mode == "alias-pair":
                    command.extend(("--kv-cache-memory", str(kv_bytes)))
                else:
                    command[2] = "262143"
                value._ssh = ssh_for(json.dumps(bad_command))
                expected = "maximum model length" if mode == "max-model" else "KV cache memory"
                with self.subTest(mode=mode), self.assertRaisesRegex(RuntimeError, expected):
                    value.verify_overlay()

            missing_env = json.loads(inspected)
            missing_env[0]["Config"]["Env"] = [
                item for item in missing_env[0]["Config"]["Env"]
                if not item.startswith("VLLM_RESILIENCE_STEP_TOKEN_CAP=")]
            value._ssh = ssh_for(json.dumps(missing_env))
            with self.assertRaisesRegex(RuntimeError, "step-cap environment differs"):
                value.verify_overlay()

            e29_mount = json.loads(inspected)
            for mount in e29_mount[0]["Mounts"]:
                if mount["Destination"] == campaign.SCHEDULER_TARGET:
                    mount["Source"] = (
                        "/home/test/tp4/experiments/e03/end-drain/scheduler.py")
            value._ssh = ssh_for(json.dumps(e29_mount))
            with self.assertRaisesRegex(RuntimeError, "step-cap scheduler mount differs"):
                value.verify_overlay()

            value._ssh = ssh_for(inspected, "d" * 64)
            with self.assertRaisesRegex(RuntimeError,
                                        "in-container scheduler hash differs"):
                value.verify_overlay()

    def test_bounded_admission_preflight_proves_pin_mount_environment_and_argument(self):
        with tempfile.TemporaryDirectory() as directory:
            value = self.make_campaign(directory, bounded_admission=True)
            env = {
                "TP4_RESILIENCE_CAMPAIGN_ID": value.cfg["campaign_id"],
                "TP4_RESILIENCE_CACHE_ROOT": value.cfg["container_cache_root"],
                "TP4_RESILIENCE_MAX_BYTES": str(common.MAX_NAMESPACE_BYTES),
                "TP4_RESILIENCE_BASE_SHA256": "a" * 64,
                "SPARK_CONTEXT_CACHE_MAX_BYTES": str(campaign.TEST_CACHE_MAX_BYTES),
                "SPARK_CONTEXT_CACHE_LOW_WATERMARK_BYTES": str(
                    campaign.TEST_CACHE_LOW_WATERMARK_BYTES),
                **campaign.ADMISSION_ENVIRONMENT,
            }
            inspected_value = [{
                "Config": {
                    "Env": [f"{key}={item}" for key, item in env.items()],
                    "Cmd": ["serve", "--middleware", campaign.ADMISSION_IMPORT],
                },
                "Mounts": [
                    {"Destination": "/opt/tp4-resilience/base_connector.py",
                     "Source": "/private/base.py"},
                    {"Destination": "/opt/tp4-resilience/runtime.py",
                     "Source": "/private/runtime.py"},
                    {"Destination":
                     "/usr/local/lib/python3.12/dist-packages/sparkcache/"
                     "spark_context_cache_connector.py",
                     "Source": "/private/connector_wrapper.py"},
                    {"Destination": campaign.WORKER_TARGET,
                     "Source": "/home/test/tp4/scripts/node/overrides/vllm/v1/worker/"
                               "gpu_worker.py"},
                    {"Destination": campaign.ADMISSION_TARGET,
                     "Source": ("/home/test/tp4/scripts/resilience/.campaign/"
                                "unit-campaign/tp4_admission.py")},
                ],
            }]
            bundle_manifest = Path(value.cfg["repo_root"]) / (
                "scripts/resilience/.campaign/unit-campaign/manifest.json")
            manifest_files = json.loads(bundle_manifest.read_text())["files"]

            def ssh_for(payload, admission_sha=None):
                def ssh(rank, *arguments):
                    if "exec" in arguments:
                        target = arguments[-1]
                        digest = (value.admission_contract["source_sha256"]
                                  if target == campaign.ADMISSION_TARGET
                                  else value.worker_contract["parent_sha256"])
                        if target == campaign.ADMISSION_TARGET and admission_sha is not None:
                            digest = admission_sha
                        output = f"{digest}  {target}\n"
                    elif arguments[0] == "sha256sum":
                        output = (f"{manifest_files[Path(arguments[1]).name]}  "
                                  f"{arguments[1]}\n")
                    else:
                        output = json.dumps(payload)
                    return type("Result", (), {"returncode": 0, "stdout": output})()
                return ssh

            value._ssh = ssh_for(inspected_value)
            value._faultctl = lambda rank, args: {"within_cap": True, "used_bytes": 0}
            result = value.verify_overlay()
            self.assertTrue(result["pass"])
            self.assertTrue(value.runtime_config_identity["bounded_admission"])
            self.assertEqual({row["admission_sha256"] for row in result["ranks"]},
                             {value.admission_contract["source_sha256"]})

            for mode in ("missing-env", "duplicate-env", "missing-arg",
                         "duplicate-arg", "missing-mount", "bad-hash"):
                bad = copy.deepcopy(inspected_value)
                bad_sha = None
                if mode == "missing-env":
                    bad[0]["Config"]["Env"] = bad[0]["Config"]["Env"][1:]
                elif mode == "duplicate-env":
                    key = next(iter(campaign.ADMISSION_ENVIRONMENT))
                    bad[0]["Config"]["Env"].append(
                        f"{key}={campaign.ADMISSION_ENVIRONMENT[key]}")
                elif mode == "missing-arg":
                    bad[0]["Config"]["Cmd"] = ["serve"]
                elif mode == "duplicate-arg":
                    bad[0]["Config"]["Cmd"].extend(
                        ["--middleware", campaign.ADMISSION_IMPORT])
                elif mode == "missing-mount":
                    bad[0]["Mounts"] = [mount for mount in bad[0]["Mounts"]
                                         if mount["Destination"] != campaign.ADMISSION_TARGET]
                else:
                    bad_sha = "d" * 64
                value._ssh = ssh_for(bad, bad_sha)
                expected = ("hash" if mode == "bad-hash" else
                            "mount" if mode == "missing-mount" else
                            "argument" if mode.endswith("arg") else "environment")
                with self.subTest(mode=mode), self.assertRaisesRegex(RuntimeError, expected):
                    value.verify_overlay()

    def test_worker_fault_arms_expected_down_before_signal_and_rejects_truncated_200(self):
        value = campaign.Campaign.__new__(campaign.Campaign)
        value.acknowledge_faults = True
        value.cfg = {"allow_worker_faults": True}
        value.http = type("HTTP", (), {
            "exact_prompt": lambda self, tokens, prefix: "prompt",
            "chat_payload": lambda self, prompt, **kwargs: {},
        })()
        telemetry = FakeTelemetry()
        telemetry.settle_expected_restart = lambda since: True
        value.telemetry = telemetry
        value.active_deadline = time.monotonic() + 4000
        value._set_case = lambda *args, **kwargs: None
        value._reset_faults = lambda case_id: {"case_id": case_id, "errors": []}
        value._cursor = lambda rank: "1:2:3"
        value._event = lambda *args, **kwargs: {
            "pid": 22, "process_start_ticks": 33, "kind": "stage_waiting"}
        signal_guard = []
        def faultctl(rank, arguments):
            if arguments[0] == "signal-stage":
                signal_guard.append(telemetry.expected_down)
                if not telemetry.expected_down:
                    telemetry.safety_event.set()
                return {"signal": "KILL"}
            return {"removed": [], "retained": []}
        value._faultctl = faultctl
        value._controller = lambda *args, **kwargs: {
            "returncode": 0, "health_200_monotonic": time.monotonic()}
        value._gates = lambda since: {"pass": True}
        value.preserve_cluster_evidence = lambda label: {"status": "captured", "label": label}
        class Thread:
            def join(self, timeout=None): pass
            def is_alive(self): return False
        class Request:
            def __init__(self, *args, **kwargs):
                self.thread, self.error = Thread(), None
                self.result = {"status": 200, "stream_complete": False}
            def start(self): pass
            def cancel(self): pass
        case = {"id": "worker-r0-kill", "parameters": {"rank": 0, "signal": "KILL"}}
        with patch.object(campaign, "CancelableRequest", Request):
            status, evidence = value._worker(case, 1)
        self.assertEqual(signal_guard, [True])
        self.assertFalse(telemetry.safety_event.is_set())
        self.assertTrue(evidence["interrupted_request"]["failed"])
        self.assertEqual(status, "pass")

    def test_worker_fault_is_pending_when_bounded_recovery_cannot_fit(self):
        value = campaign.Campaign.__new__(campaign.Campaign)
        value.acknowledge_faults = True
        value.cfg = {"allow_worker_faults": True}
        value.active_deadline = time.monotonic() + 10
        status, evidence = value._worker(
            {"id": "worker-r0-kill", "parameters": {"rank": 0, "signal": "KILL"}}, 1)
        self.assertEqual(status, "pending")
        self.assertIn("insufficient", evidence["reason"])

    def test_worker_stop_allows_a_valid_completed_stream(self):
        value = campaign.Campaign.__new__(campaign.Campaign)
        value.acknowledge_faults = True
        value.cfg = {"allow_worker_faults": True}
        value.active_deadline = time.monotonic() + 4000
        value.http = type("HTTP", (), {
            "exact_prompt": lambda self, tokens, prefix: "prompt",
            "chat_payload": lambda self, prompt, **kwargs: {},
        })()
        telemetry = FakeTelemetry()
        telemetry.settle_expected_restart = lambda since: True
        value.telemetry = telemetry
        value._set_case = lambda *args, **kwargs: None
        value._reset_faults = lambda case_id: {"case_id": case_id, "errors": []}
        value._cursor = lambda rank: "1:2:3"
        value._event = lambda *args, **kwargs: {
            "pid": 22, "process_start_ticks": 33, "kind": "stage_waiting"}
        value._faultctl = lambda rank, arguments: ({"signal": arguments[-1]}
                                                   if arguments[0] == "signal-stage"
                                                   else {"removed": [], "retained": []})
        value._controller = lambda *args, **kwargs: {
            "returncode": 0, "health_200_monotonic": time.monotonic()}
        value._gates = lambda since: {"pass": True}
        value.preserve_cluster_evidence = lambda label: {"status": "captured"}
        class Thread:
            def join(self, timeout=None): pass
            def is_alive(self): return False
        class Request:
            def __init__(self, *args, **kwargs):
                self.thread, self.error = Thread(), None
                self.result = {"status": 200, "stream_complete": True, "done": True,
                               "error_event": False, "parse_errors": [],
                               "finish_reasons": ["stop"]}
            def start(self): pass
            def cancel(self): pass
        case = {"id": "worker-r0-stop", "parameters": {"rank": 0, "signal": "STOP"}}
        with patch.object(campaign, "CancelableRequest", Request), \
                patch.object(campaign.time, "sleep", return_value=None):
            status, evidence = value._worker(case, 1)
        self.assertEqual(status, "pass")
        self.assertTrue(evidence["interrupted_request"]["completed_successfully"])
        self.assertFalse(evidence["interrupted_request"]["failed"])

    def test_worker_without_signal_fails_when_release_cleanup_fails(self):
        value = campaign.Campaign.__new__(campaign.Campaign)
        value.acknowledge_faults = True
        value.cfg = {"allow_worker_faults": True}
        value.active_deadline = time.monotonic() + 4000
        value.http = type("HTTP", (), {
            "exact_prompt": lambda self, tokens, prefix: "prompt",
            "chat_payload": lambda self, prompt, **kwargs: {},
        })()
        value.telemetry = FakeTelemetry()
        value._set_case = lambda *args, **kwargs: None
        value._reset_faults = lambda case_id: {
            "case_id": case_id, "errors": ["rank 1 mode off failed"]}
        value._cursor = lambda rank: "cursor"
        value._event = lambda *args, **kwargs: (_ for _ in ()).throw(
            TimeoutError("no capture edge"))
        value.health_code = lambda: 200
        class Thread:
            def join(self, timeout=None): pass
            def is_alive(self): return False
        class Request:
            def __init__(self, *args, **kwargs):
                self.thread, self.result, self.error = Thread(), None, None
            def start(self): pass
            def cancel(self): pass
        case = {"id": "worker-r0-kill", "parameters": {"rank": 0, "signal": "KILL"}}
        with patch.object(campaign, "CancelableRequest", Request):
            status, evidence = value._worker(case, 1)
        self.assertEqual(status, "fail")
        self.assertNotIn("signal", evidence)
        self.assertTrue(evidence["release"]["errors"])

    def test_context_replay_requires_exact_usage_and_four_rank_restore(self):
        value = campaign.Campaign.__new__(campaign.Campaign)
        value.cfg = {"campaign_id": "unit"}
        response = fixed_response(8_000)
        value.http = type("HTTP", (), {
            "exact_prompt": lambda self, target, prefix: "prompt",
            "chat": lambda self, prompt, request_id: response,
        })()
        value._set_case = lambda case_id: None
        value._cursor = lambda rank: f"cursor-{rank}"
        value._request_group = lambda prompts, case_id: [response]
        publications = [{"outcome": "ok", "committed": True, "digest": "a" * 64}
                        for _ in range(4)]
        restores = [{"outcome": "ok", "digest": "a" * 64} for _ in range(4)]
        events = iter([(publications, []), (restores, [])])
        value._events_all = lambda *args, **kwargs: next(events)
        case = {"id": "context-replay", "parameters": {
            "tokens": 8_000, "concurrency": 1, "cache": "replay"}}
        status, evidence = value._context(case, 1)
        self.assertEqual(status, "pass")
        self.assertEqual(evidence["prompt_tokens"], [8_000])

        wrong = fixed_response(7_999)
        value.http = type("HTTP", (), {
            "exact_prompt": lambda self, target, prefix: "prompt",
            "chat": lambda self, prompt, request_id: wrong,
        })()
        value._request_group = lambda prompts, case_id: [wrong]
        status, _ = value._context(case, 2)
        self.assertEqual(status, "fail")

    def test_c5_full_context_requires_five_exact_length_completions(self):
        value = campaign.Campaign.__new__(campaign.Campaign)
        value.cfg = {"campaign_id": "unit"}
        value.http = type("HTTP", (), {
            "exact_prompt": lambda self, target, prefix: f"prompt-{target}-{prefix}",
        })()
        case = {"id": "context-262128-c5-cold", "parameters": {
            "tokens": 262_128, "concurrency": 5, "cache": "cold"}}
        responses = [fixed_response(262_128) for _ in range(5)]
        value._request_group = lambda prompts, case_id: (
            self.assertEqual(len(prompts), 5) or list(responses))
        status, evidence = value._context(case, 1)
        self.assertEqual(status, "pass")
        self.assertEqual(len(evidence["responses"]), 5)
        self.assertEqual(evidence["prompt_tokens"], [262_128] * 5)

        responses[-1] = fixed_response(262_128, completion_tokens=15)
        self.assertEqual(value._context(case, 2)[0], "fail")

    def test_c5_long_decode_uses_distinct_prompts_and_exact_forced_budget(self):
        value = campaign.Campaign.__new__(campaign.Campaign)
        value.cfg = {"campaign_id": "unit"}
        original_deadline = time.monotonic() + 7200
        value.active_deadline = original_deadline
        client = campaign.HTTPClient("http://rank0:8000/v1", "model")
        client.exact_prompt = lambda target, prefix: f"{prefix}|tokens={target}"
        sent: list[tuple[dict, dict]] = []
        submitted = threading.Barrier(5)

        def json_request(_path, payload, **kwargs):
            sent.append((copy.deepcopy(payload), copy.deepcopy(kwargs)))
            submitted.wait(timeout=2)
            return fixed_response(245_760, completion_tokens=16_384)

        client.json = json_request
        value.http = client
        case = {"id": campaign.LONG_DECODE_CASE_ID, "parameters": {
            "tokens": 245_760, "concurrency": 5, "cache": "cold",
            "max_tokens": 16_384, "min_tokens": 16_384,
            "ignore_eos": True, "long_decode_pressure": True,
        }}
        status, evidence = value._context(case, 1)
        self.assertEqual(status, "pass")
        self.assertEqual(len(sent), 5)
        prompts = [payload["messages"][0]["content"] for payload, _ in sent]
        self.assertEqual(len(set(prompts)), 5)
        for payload, kwargs in sent:
            self.assertEqual(payload["max_tokens"], 16_384)
            self.assertEqual(payload["min_tokens"], 16_384)
            self.assertIs(payload["ignore_eos"], True)
            self.assertEqual(payload["chat_template_kwargs"], {
                "reasoning_effort": "low"})
            self.assertIn("X-Request-Id", kwargs["headers"])
        self.assertEqual(evidence["total_token_budget"], 262_144)
        self.assertEqual(len(set(evidence["distinct_prompt_sha256"])), 5)
        self.assertIs(evidence["all_request_intervals_overlap"], True)
        self.assertEqual(evidence["request_deadline"]["limit_seconds"], 3600)
        self.assertEqual(evidence["request_deadline"]["limited_by"],
                         "case_local_limit")
        self.assertEqual(value.active_deadline, original_deadline)
        for response in evidence["responses"]:
            self.assertIsInstance(response["started_ns"], int)
            self.assertGreaterEqual(response["finished_ns"], response["started_ns"])
            receipt = response["request_payload_receipt"]
            self.assertEqual(receipt["max_tokens"], 16_384)
            self.assertEqual(receipt["min_tokens"], 16_384)
            self.assertIs(receipt["ignore_eos"], True)
            self.assertRegex(receipt["sha256"], r"^[0-9a-f]{64}$")

        def early_eos(_path, payload, **kwargs):
            return fixed_response(245_760, finish_reason="stop",
                                  completion_tokens=8_000)

        client.json = early_eos
        self.assertEqual(value._context(case, 2)[0], "fail")
        self.assertEqual(value.active_deadline, original_deadline)

    def test_long_decode_deadline_restoration_and_classification(self):
        case = {"id": campaign.LONG_DECODE_CASE_ID, "parameters": {
            "tokens": 245_760, "concurrency": 5, "cache": "cold",
            "max_tokens": 16_384, "min_tokens": 16_384,
            "ignore_eos": True, "long_decode_pressure": True,
        }}

        def make_value(deadline):
            value = campaign.Campaign.__new__(campaign.Campaign)
            value.cfg = {"campaign_id": "unit"}
            value.active_deadline = deadline
            value.telemetry = FakeTelemetry()
            value.http = type("HTTP", (), {
                "exact_prompt": lambda self, target, prefix: f"{prefix}|{target}",
            })()
            return value

        def phase_errors():
            return [{"request_id": f"long-{index}", "status": None,
                     "started_ns": index, "finished_ns": index + 1,
                     "error": "GuardAbortError: phase deadline",
                     "guard_abort_cause": "phase_deadline"}
                    for index in range(5)]

        clock = [100.0]
        local = make_value(10_000.0)
        def local_group(prompts, case_id, request_options):
            self.assertEqual(len(prompts), 5)
            self.assertEqual(local.active_deadline, 3_700.0)
            clock[0] = 3_701.0
            return phase_errors()
        local._request_group = local_group
        with patch.object(campaign.time, "monotonic", side_effect=lambda: clock[0]):
            status, evidence = local._context(case, 1)
        self.assertEqual(status, "fail")
        self.assertEqual(len(evidence["responses"]), 5)
        self.assertEqual(evidence["request_deadline"]["limited_by"],
                         "case_local_limit")
        self.assertEqual(local.active_deadline, 10_000.0)

        clock[0] = 100.0
        global_cutoff = make_value(150.0)
        def global_group(prompts, case_id, request_options):
            self.assertEqual(global_cutoff.active_deadline, 150.0)
            clock[0] = 151.0
            return phase_errors()
        global_cutoff._request_group = global_group
        with patch.object(campaign.time, "monotonic", side_effect=lambda: clock[0]):
            status, evidence = global_cutoff._context(case, 1)
        self.assertEqual(status, "pending")
        self.assertTrue(evidence["planned_cutoff"])
        self.assertEqual(evidence["request_deadline"]["limited_by"],
                         "campaign_phase_deadline")
        self.assertEqual(global_cutoff.active_deadline, 150.0)

        clock[0] = 100.0
        exceptional = make_value(10_000.0)
        exceptional._request_group = lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("request group failed"))
        with patch.object(campaign.time, "monotonic", side_effect=lambda: clock[0]), \
                self.assertRaisesRegex(RuntimeError, "request group failed"):
            exceptional._context(case, 1)
        self.assertEqual(exceptional.active_deadline, 10_000.0)

    def test_context_classifies_only_clean_phase_cutoff_as_pending(self):
        value = campaign.Campaign.__new__(campaign.Campaign)
        value.cfg = {"campaign_id": "unit"}
        value.active_deadline = time.monotonic() - 1
        value.telemetry = FakeTelemetry()
        value.http = type("HTTP", (), {
            "exact_prompt": lambda self, target, prefix: "prompt",
        })()
        case = {"id": "context-cutoff", "parameters": {
            "tokens": 8_000, "concurrency": 2, "cache": "cold"}}
        valid = fixed_response(8_000)
        phase = {"status": None, "error": "GuardAbortError: phase deadline",
                 "guard_abort_cause": "phase_deadline"}
        value._request_group = lambda prompts, case_id: [valid, phase]
        status, evidence = value._context(case, 1)
        self.assertEqual(status, "pending")
        self.assertTrue(evidence["planned_cutoff"])

        transport = {"status": None, "error": "OSError: reset",
                     "guard_abort_cause": None}
        value._request_group = lambda prompts, case_id: [valid, transport]
        self.assertEqual(value._context(case, 2)[0], "fail")

    def test_inflight_context_cutoff_cleans_drains_and_starts_soak(self):
        with tempfile.TemporaryDirectory() as directory:
            value = self.make_campaign(directory)
            target = {"id": "context-8000-c2-cold", "area": "contexts",
                      "parameters": {"tokens": 8_000, "concurrency": 2,
                                     "cache": "cold"}, "target_repetitions": 1}
            soak = {"id": "soak-mixed", "area": "duration",
                    "parameters": {"hours": 2,
                                   "protocol_version": campaign.SOAK_PROTOCOL_VERSION},
                    "target_repetitions": 1}
            final = {"id": "rigmark-final", "area": "performance", "manual": True,
                     "target_repetitions": 1}
            value.plan = [target, soak, final]
            value.verify_overlay = lambda: {"pass": True}
            value.http = type("HTTP", (), {
                "exact_prompt": lambda self, target, prefix: "prompt",
            })()
            actions = []
            def group(prompts, case_id):
                value.targeted_stop_at = time.monotonic() - 1
                value.active_deadline = value.targeted_stop_at
                return [{"status": None, "error": "GuardAbortError: phase deadline",
                         "guard_abort_cause": "phase_deadline"} for _ in prompts]
            value._request_group = group
            value.soak_at = time.monotonic() + 60
            value._reset_faults = lambda case_id: actions.append(
                f"cleanup:{case_id}") or {"case_id": case_id, "errors": []}
            value.health_code = lambda: actions.append("health") or 200
            value.wait_for_api_idle = lambda: actions.append("idle") or ("pass", {})
            value._drain_barriers = lambda: actions.append("barriers") or {
                rank: rank for rank in range(4)}
            value.wait_for_drain = lambda barriers: actions.append("drain") or ("pass", {})
            value._soak = lambda case: actions.append("soak") or (
                "pending", {"protocol_version": campaign.SOAK_PROTOCOL_VERSION})
            value.restore_default = lambda: {
                "gates": {"pass": True}, "identity": {"returncode": 0}}
            value.end_monotonic = time.monotonic()
            self.assertEqual(value.run(), 1)
            target_run = value.state["cases"][target["id"]]["runs"][0]
            self.assertEqual(target_run["status"], "pending")
            self.assertTrue(target_run["evidence"]["planned_cutoff"])
            self.assertIn("reservation_drain", target_run["evidence"])
            self.assertTrue(target_run["evidence"]["post_cutoff_health_200"])
            self.assertLess(actions.index("cleanup:targeted-cutoff-cleanup"),
                            actions.index("drain"))
            self.assertLess(actions.index("cleanup:pre-soak"), actions.index("soak"))

    def test_run_cutoff_softens_only_explicit_phase_deadline_exceptions(self):
        for error, expected, soak_expected in (
                (campaign.PhaseDeadlineError("deadline"), "pending", True),
                (OSError("late transport failure"), "fail", False)):
            with self.subTest(error=type(error).__name__), \
                    tempfile.TemporaryDirectory() as directory:
                value = self.make_campaign(directory)
                target = {"id": "target", "area": "contexts", "parameters": {},
                          "target_repetitions": 1}
                soak = {"id": "soak-mixed", "area": "duration",
                        "parameters": {"hours": 2,
                                       "protocol_version": campaign.SOAK_PROTOCOL_VERSION},
                        "target_repetitions": 1}
                final = {"id": "rigmark-final", "area": "performance",
                         "manual": True, "target_repetitions": 1}
                value.plan = [target, soak, final]
                value.verify_overlay = lambda: {"pass": True}
                value.soak_at = time.monotonic() + 60
                soaked = []
                def execute(case, repetition):
                    if case.get("manual"):
                        return "recorded", {}
                    value.targeted_stop_at = time.monotonic() - 1
                    raise error
                value.execute_case = execute
                value._reset_faults = lambda case_id: {
                    "case_id": case_id, "errors": []}
                value.health_code = lambda: 200
                value.wait_for_api_idle = lambda: ("pass", {})
                value._drain_barriers = lambda: {rank: rank for rank in range(4)}
                value.wait_for_drain = lambda barriers: ("pass", {})
                value._soak = lambda case: soaked.append(case["id"]) or (
                    "pending", {"protocol_version": campaign.SOAK_PROTOCOL_VERSION})
                value.restore_default = lambda: {
                    "gates": {"pass": True}, "identity": {"returncode": 0}}
                value.preserve_cluster_evidence = lambda label: {"status": "captured"}
                value.end_monotonic = time.monotonic()
                self.assertEqual(value.run(), 1)
                run = value.state["cases"]["target"]["runs"][0]
                self.assertEqual(run["status"], expected)
                self.assertEqual(bool(soaked), soak_expected)
                if type(error) is OSError:
                    self.assertNotIn("targeted phase deadline", run["evidence"].get(
                        "reason", ""))

    def test_client_capacity_is_nonfatal_only_after_a_clean_common_drain(self):
        for drain_status, expected in (("pass", "pending"), ("pending", "fail")):
            with self.subTest(drain_status=drain_status), \
                    tempfile.TemporaryDirectory() as directory:
                value = self.make_campaign(directory)
                target = {"id": "queue-wave-unit", "area": "queue",
                          "parameters": {"clients": 8}, "target_repetitions": 1}
                final = {"id": "rigmark-final", "area": "performance",
                         "manual": True, "target_repetitions": 1}
                value.plan = [target, final]
                value.verify_overlay = lambda: {"pass": True}
                value.execute_case = lambda case, repetition: (
                    ("recorded", {}) if case.get("manual") else
                    ("pending", {"client_capacity_limited": True,
                                 "reason": "local client capacity"}))
                value._drain_barriers = lambda: {rank: rank for rank in range(4)}
                value.wait_for_drain = lambda barriers: (
                    drain_status, {"status": drain_status})
                value.restore_default = lambda: {
                    "gates": {"pass": True}, "identity": {"returncode": 0}}
                value.preserve_cluster_evidence = lambda label: {"status": "captured"}
                value.end_monotonic = time.monotonic()
                self.assertEqual(value.run(), 1)
                run = value.state["cases"][target["id"]]["runs"][0]
                self.assertEqual(run["status"], expected)
                self.assertEqual(run["evidence"]["reservation_drain"],
                                 {"status": drain_status})

    def test_clean_client_capacity_defers_only_larger_queue_waves(self):
        with tempfile.TemporaryDirectory() as directory:
            value = self.make_campaign(directory)
            queue64 = {"id": "queue-wave-64", "area": "queue",
                       "parameters": {"clients": 64}, "target_repetitions": 3}
            queue128 = {"id": "queue-wave-128", "area": "queue",
                        "parameters": {"clients": 128}, "target_repetitions": 3}
            final = {"id": "rigmark-final", "area": "performance",
                     "manual": True, "target_repetitions": 1}
            value.plan = [queue64, queue128, final]
            value.verify_overlay = lambda: {"pass": True}
            attempted = []
            def execute(case, repetition):
                if case.get("manual"):
                    return "recorded", {}
                attempted.append((case["id"], repetition))
                return "pending", {"client_capacity_limited": True,
                                   "reason": "local client capacity"}
            value.execute_case = execute
            value._drain_barriers = lambda: {rank: rank for rank in range(4)}
            value.wait_for_drain = lambda barriers: ("pass", {"clean": True})
            value.restore_default = lambda: {
                "gates": {"pass": True}, "identity": {"returncode": 0}}
            value.end_monotonic = time.monotonic()
            self.assertEqual(value.run(), 1)
            self.assertEqual(attempted, [("queue-wave-64", 1)])
            smaller = value.state["cases"]["queue-wave-64"]["runs"]
            larger = value.state["cases"]["queue-wave-128"]["runs"]
            self.assertEqual(len(smaller), 1)
            self.assertTrue(smaller[0]["evidence"]["higher_queue_waves_deferred"])
            self.assertEqual(len(larger), 1)
            self.assertEqual(larger[0]["status"], "pending")
            self.assertTrue(larger[0]["evidence"]["not_attempted"])
            self.assertEqual(larger[0]["evidence"][
                "client_capacity_boundary"]["clients"], 64)

    def test_request_group_preserves_success_and_failure_receipts(self):
        value = campaign.Campaign.__new__(campaign.Campaign)
        def chat(_prompt, request_id):
            if request_id.endswith("-1"):
                raise ConnectionError("peer disconnected")
            if request_id.endswith("-2"):
                raise campaign.GuardAbortError("safety_event")
            return {"status": 200, "body": {"choices": [{}]}}
        value.http = type("HTTP", (), {"chat": staticmethod(chat)})()
        results = value._request_group(["one", "two", "three"], "context-unit")
        by_id = {result["request_id"]: result for result in results}
        self.assertEqual(set(by_id), {"context-unit-0", "context-unit-1", "context-unit-2"})
        self.assertEqual(by_id["context-unit-0"]["status"], 200)
        self.assertIsNone(by_id["context-unit-0"]["error"])
        self.assertIsNone(by_id["context-unit-1"]["status"])
        self.assertIn("ConnectionError", by_id["context-unit-1"]["error"])
        self.assertIsNone(by_id["context-unit-1"]["guard_abort_cause"])
        self.assertEqual(by_id["context-unit-2"]["guard_abort_cause"], "safety_event")
        for result in results:
            self.assertIsInstance(result["started_ns"], int)
            self.assertGreaterEqual(result["finished_ns"], result["started_ns"])
        self.assertFalse(value._valid(results))

    def test_cache_mutation_failure_restores_earlier_ranks(self):
        value = campaign.Campaign.__new__(campaign.Campaign)
        value.cfg = {}
        value.targeted_stop_at = time.monotonic() + 60
        value.soak_at = time.monotonic() + 120
        value.active_deadline = value.targeted_stop_at
        value.telemetry = FakeTelemetry()
        response = {"status": 200, "body": {"choices": [{}]}}
        value.http = type("HTTP", (), {
            "exact_prompt": lambda self, target, prefix: "prompt",
            "chat": lambda self, prompt, request_id: response,
        })()
        value._set_case = lambda case_id: None
        value._cursor = lambda rank: f"cursor-{rank}"
        publications = [{"outcome": "ok", "committed": True, "digest": "b" * 64}
                        for _ in range(4)]
        value._events_all = lambda *args, **kwargs: (publications, [])
        calls = []
        def faultctl(rank, arguments, **kwargs):
            calls.append((rank, arguments[0]))
            if arguments[0] == "mutate" and rank == 1:
                raise RuntimeError("mutation failed")
            return {"restored": []}
        value._faultctl = faultctl
        status, evidence = value._cache(
            {"id": "cache-checksum", "parameters": {"variant": "checksum"}}, 1)
        self.assertEqual(status, "fail")
        self.assertIn("mutation failed", evidence["action_error"])
        self.assertCountEqual([rank for rank, command in calls if command == "restore"], [0, 1, 2, 3])

    def test_shared_prefix_crosses_boundary_and_requires_seed_restore_then_new_publication(self):
        seed_digest, extension_digest = "a" * 64, "b" * 64
        value = campaign.Campaign.__new__(campaign.Campaign)
        value.targeted_stop_at = time.monotonic() + 60
        value.telemetry = FakeTelemetry()
        calls = []
        seed_prompt = "seed-prompt"
        extension_prompt = seed_prompt + " shared extension padded"

        class HTTP:
            def exact_prompt(self, target, prefix):
                calls.append(("exact_prompt", target, prefix))
                if target == 32_000:
                    return seed_prompt
                self_test.assertEqual(target, 34_304)
                self_test.assertTrue(prefix.startswith(seed_prompt))
                return extension_prompt
            def chat(self, prompt, request_id):
                calls.append(("chat", request_id, prompt))
                return fixed_response(32_000 if request_id.endswith("store") else 34_304)

        self_test = self
        value.http = HTTP()
        value._set_case = lambda case_id: None
        value._cursor = lambda rank: f"cursor-{rank}"
        value.health_code = lambda: 200
        publications = [{"outcome": "ok", "committed": True,
                         "digest": seed_digest} for _ in range(4)]
        restores = [{"outcome": "ok", "digest": seed_digest} for _ in range(4)]
        extensions = [{"outcome": "ok", "committed": True,
                       "digest": extension_digest} for _ in range(4)]
        event_calls = []
        def events(_cursors, _case_id, _kind, *, stage=None, **_kwargs):
            event_calls.append(stage)
            if event_calls == ["publication"]:
                return publications, []
            if stage == "restore":
                return restores, []
            return extensions, []
        value._events_all = events
        case = {"id": "cache-shared-prefix", "parameters": {"variant": "shared-prefix"}}
        status, evidence = value._cache_action(case, 1)
        self.assertEqual(status, "pass")
        self.assertEqual(event_calls, ["publication", "restore", "publication"])
        self.assertEqual(evidence["protocol_version"],
                         campaign.TARGETED_CASE_PROTOCOL_VERSION)
        self.assertTrue(evidence["shared_prefix"]["literal_prefix_preserved"])
        self.assertEqual(calls[2], ("exact_prompt", 34_304,
                                    seed_prompt + " shared extension"))

        for mode in ("missing_rank", "wrong_digest"):
            with self.subTest(mode=mode):
                event_calls.clear()
                broken = list(restores)
                if mode == "missing_rank":
                    broken = broken[:3]
                else:
                    broken[-1] = {"outcome": "ok", "digest": "c" * 64}
                def broken_events(_cursors, _case_id, _kind, *, stage=None, **_kwargs):
                    event_calls.append(stage)
                    if event_calls == ["publication"]:
                        return publications, []
                    if stage == "restore":
                        return broken, []
                    return extensions, []
                value._events_all = broken_events
                broken_status, broken_evidence = value._cache_action(case, 2)
                self.assertEqual(broken_status, "pending")
                self.assertIn("shared seed digest", broken_evidence["reason"])

    def test_fault_reset_tries_off_and_restore_on_all_ranks_with_own_deadline(self):
        value = campaign.Campaign.__new__(campaign.Campaign)
        value.soak_at = time.monotonic() + 120
        value.active_deadline = time.monotonic() - 1
        original_deadline = value.active_deadline
        calls = []
        def faultctl(rank, arguments, **kwargs):
            calls.append((rank, arguments[0], kwargs.get("deadline")))
            if rank == 1 and arguments[0] == "set":
                raise RuntimeError("set failed")
            return {"ok": True}
        value._faultctl = faultctl
        result = value._reset_faults("unit-cleanup")
        self.assertEqual(value.active_deadline, original_deadline)
        self.assertCountEqual([(rank, command) for rank, command, _ in calls],
                              [(rank, command) for rank in range(4)
                               for command in ("set", "restore")])
        self.assertTrue(all(deadline > time.monotonic() for _, _, deadline in calls))
        self.assertEqual(len(result["errors"]), 1)

    def test_partial_case_arm_still_resets_all_cache_ranks(self):
        value = campaign.Campaign.__new__(campaign.Campaign)
        value.cfg = {}
        value.targeted_stop_at = time.monotonic() + 60
        value.soak_at = time.monotonic() + 120
        value.active_deadline = value.targeted_stop_at
        value.telemetry = FakeTelemetry()
        calls = []
        def faultctl(rank, arguments, **kwargs):
            calls.append((rank, arguments[0]))
            if arguments[0] == "set" and rank == 1 and arguments[4] == "cache-checksum":
                raise RuntimeError("partial arm")
            return {"ok": True}
        value._faultctl = faultctl
        status, evidence = value._cache(
            {"id": "cache-checksum", "parameters": {"variant": "checksum"}}, 1)
        self.assertEqual(status, "fail")
        self.assertIn("partial arm", evidence["action_error"])
        self.assertCountEqual([rank for rank, command in calls if command == "restore"],
                              [0, 1, 2, 3])

    def test_cleanup_failure_overrides_cache_and_cancel_pending_without_reissue(self):
        value = campaign.Campaign.__new__(campaign.Campaign)
        value.cfg = {}
        value.targeted_stop_at = time.monotonic() + 60
        value.active_deadline = value.targeted_stop_at
        value.telemetry = FakeTelemetry()
        chat_calls = []
        value.http = type("HTTP", (), {
            "exact_prompt": lambda self, target, prefix: "prompt",
            "chat_payload": lambda self, prompt, **kwargs: {},
            "chat": lambda self, prompt, request_id: chat_calls.append(request_id) or {
                "status": 200, "body": {"choices": [{}]}},
        })()
        value._set_case = lambda *args, **kwargs: None
        value._cursor = lambda rank: f"cursor-{rank}"
        value._events_all = lambda *args, **kwargs: ([], [])
        value._reset_faults = lambda case_id: {
            "case_id": case_id, "errors": ["cleanup failed"]}
        status, evidence = value._cache(
            {"id": "cache-checksum", "parameters": {"variant": "checksum"}}, 1)
        self.assertEqual(status, "fail")
        self.assertTrue(evidence["cleanup_errors"])

        class Signal:
            def wait(self, timeout=None): return False
        class Thread:
            def join(self, timeout=None): pass
            def is_alive(self): return False
        class Request:
            def __init__(self, *args, **kwargs):
                self.first_byte, self.thread = Signal(), Thread()
                self.error, self.result = None, None
            def start(self): pass
            def cancel(self): pass
        chat_calls.clear()
        with patch.object(campaign, "CancelableRequest", Request):
            status, evidence = value._cancel(
                {"id": "cancel-decode", "parameters": {"stage": "decode"}}, 1)
        self.assertEqual(status, "fail")
        self.assertTrue(evidence["cleanup"]["errors"])
        self.assertEqual(chat_calls, [])

    def test_stage_wait_error_cancels_and_drains_request_before_pending(self):
        value = campaign.Campaign.__new__(campaign.Campaign)
        value.cfg = {}
        value.active_deadline = time.monotonic() + 60
        value.telemetry = FakeTelemetry()
        value.http = type("HTTP", (), {
            "exact_prompt": lambda self, target, prefix: "prompt",
            "chat_payload": lambda self, prompt, **kwargs: {},
            "chat": lambda self, prompt, request_id: fixed_response(128_000),
        })()
        value._set_case = lambda *args, **kwargs: None
        value._cursor = lambda rank: "cursor"
        value._event = lambda *args, **kwargs: (_ for _ in ()).throw(
            TimeoutError("stage edge missing"))
        actions = []
        value._reset_faults = lambda case_id: actions.append("reset") or {
            "case_id": case_id, "errors": []}
        value.health_code = lambda: 200
        value.wait_for_api_idle = lambda: ("pass", {"requests_running": 0})
        value._drain_barriers = lambda: {rank: rank for rank in range(4)}
        value.wait_for_drain = lambda barriers: ("pass", {"ranks": barriers})
        requests = []
        class Thread:
            def __init__(self): self.alive = False
            def join(self, timeout=None): actions.append("join")
            def is_alive(self): return self.alive
        class Request:
            def __init__(self, *args, **kwargs):
                self.thread, self.error, self.result = Thread(), None, None
                requests.append(self)
            def start(self): self.thread.alive = True
            def cancel(self):
                actions.append("cancel")
                self.thread.alive = False
        with patch.object(campaign, "CancelableRequest", Request):
            status, evidence = value._cancel(
                {"id": "cancel-capture", "parameters": {"stage": "capture"}}, 1)
        self.assertEqual(status, "pending")
        self.assertFalse(requests[0].thread.is_alive())
        self.assertLess(actions.index("cancel"), actions.index("reset"))
        self.assertEqual(evidence["idle"], {
            "status": "pass", "evidence": {"requests_running": 0}})

    def test_decode_cancellation_wait_stops_at_active_deadline(self):
        value = campaign.Campaign.__new__(campaign.Campaign)
        value.cfg = {}
        value.active_deadline = time.monotonic() + 0.08
        value.telemetry = FakeTelemetry()
        value.http = type("HTTP", (), {
            "exact_prompt": lambda self, target, prefix: "prompt",
            "chat_payload": lambda self, prompt, **kwargs: {},
            "chat": lambda self, prompt, request_id: fixed_response(32_000),
        })()
        value._set_case = lambda *args, **kwargs: None
        value._reset_faults = lambda case_id: {"case_id": case_id, "errors": []}
        value.health_code = lambda: 200
        value.wait_for_api_idle = lambda: ("pass", {})
        value._drain_barriers = lambda: {rank: rank for rank in range(4)}
        value.wait_for_drain = lambda barriers: ("pass", {"ranks": barriers})
        class Thread:
            alive = False
            def join(self, timeout=None): pass
            def is_alive(self): return self.alive
        class Request:
            def __init__(self, *args, **kwargs):
                self.first_byte, self.thread = threading.Event(), Thread()
                self.error = self.result = None
            def start(self): self.thread.alive = True
            def cancel(self): self.thread.alive = False
        started = time.monotonic()
        with patch.object(campaign, "CancelableRequest", Request):
            status, evidence = value._cancel(
                {"id": "cancel-decode", "parameters": {"stage": "decode"}}, 1)
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertEqual(status, "pending")
        self.assertFalse(evidence["synchronized"])

    def test_cancellation_predicate_rejects_completion_and_pre_cancel_errors(self):
        base = {
            "terminated": True, "http_status": 200, "finished_before_cancel": False,
            "observed_completion": False, "interrupted": True, "cancelled_flag": True,
            "result": {"done": False, "finish_reasons": [], "error_event": False,
                       "parse_errors": [], "read_complete": False},
            "cancellation": {"started_before_cancel": True,
                             "thread_alive_before_cancel": True,
                             "pre_cancel": {"error": None, "http_status": 200}},
        }
        self.assertEqual(campaign._cancellation_outcome(base), ("pass", None))
        for result in (
                {**base, "observed_completion": True,
                 "result": {"done": True, "finish_reasons": [],
                            "error_event": False, "parse_errors": []}},
                {**base, "observed_completion": True,
                 "result": {"done": False, "finish_reasons": ["length"],
                            "error_event": False, "parse_errors": []}}):
            with self.subTest(result=result["result"]):
                self.assertEqual(campaign._cancellation_outcome(result)[0], "pending")
        pre_error = copy.deepcopy(base)
        pre_error["cancellation"]["pre_cancel"]["error"] = "ConnectionError: early"
        self.assertEqual(campaign._cancellation_outcome(pre_error)[0], "fail")
        malformed = copy.deepcopy(base)
        malformed["result"]["parse_errors"] = ["JSONDecodeError"]
        self.assertEqual(campaign._cancellation_outcome(malformed)[0], "fail")
        http_error = copy.deepcopy(base)
        http_error["http_status"] = 503
        self.assertEqual(campaign._cancellation_outcome(http_error)[0], "fail")
        phase_guard = copy.deepcopy(base)
        phase_guard["guard_abort_cause"] = "phase_deadline"
        phase_guard["cancellation"]["pre_cancel"]["error"] = (
            "ConnectionAbortedError: campaign HTTP guard aborted request: phase_deadline")
        self.assertEqual(campaign._cancellation_outcome(phase_guard)[0], "pending")
        safety_guard = copy.deepcopy(base)
        safety_guard["guard_abort_cause"] = "safety_event"
        safety_guard["cancellation"]["pre_cancel"]["error"] = (
            "ConnectionAbortedError: campaign HTTP guard aborted request: safety_event")
        self.assertEqual(campaign._cancellation_outcome(safety_guard)[0], "fail")

    def test_cancellation_allows_only_an_incomplete_trailing_sse_fragment(self):
        def streamed(payload, *, read_complete=False):
            return {
                "status": 200, "body_text": payload.decode("utf-8", "replace"),
                "read_complete": read_complete, **campaign.parse_sse(payload),
            }

        base = {
            "terminated": True, "http_status": 200, "finished_before_cancel": False,
            "observed_completion": False, "interrupted": True, "cancelled_flag": True,
            "cancellation": {"started_before_cancel": True,
                             "thread_alive_before_cancel": True,
                             "pre_cancel": {"error": None, "http_status": 200}},
        }
        incomplete_json = {
            **base, "result": streamed(b'data: {"choices":')}
        self.assertTrue(incomplete_json["result"]["parse_errors"])
        self.assertEqual(campaign._cancellation_outcome(incomplete_json), ("pass", None))

        incomplete_utf8 = {
            **base, "result": streamed(
                b'data: {"choices":[]}\n\ndata: {"choices":\xff')}
        self.assertTrue(incomplete_utf8["result"]["parse_errors"])
        self.assertEqual(campaign._cancellation_outcome(incomplete_utf8), ("pass", None))

        done_before_tail = {
            **base, "result": streamed(b'data: [DONE]\n\ndata: \xff')}
        done_before_tail["observed_completion"] = campaign._observed_stream_completion(
            done_before_tail["result"])
        self.assertEqual(campaign._cancellation_outcome(done_before_tail)[0], "pending")

        api_error_before_tail = {
            **base, "result": streamed(
                b'event: error\ndata: {"error":{"message":"bad"}}\n\ndata: \xff')}
        self.assertEqual(campaign._cancellation_outcome(api_error_before_tail)[0], "fail")

        malformed_complete = {
            **base, "result": streamed(
                b'data: {bad}\n\ndata: {"choices":')}
        self.assertEqual(campaign._cancellation_outcome(malformed_complete)[0], "fail")

        malformed_eof = {
            **base, "result": streamed(b'data: {bad}\n\n', read_complete=True)}
        self.assertEqual(campaign._cancellation_outcome(malformed_eof)[0], "fail")

        not_cancelled = copy.deepcopy(incomplete_json)
        not_cancelled["cancelled_flag"] = False
        self.assertEqual(campaign._cancellation_outcome(not_cancelled)[0], "pending")

        class StoppedThread:
            def is_alive(self): return False
        request = type("Request", (), {
            "thread": StoppedThread(), "result": incomplete_json["result"],
            "finished_ns": 150, "cancelled": threading.Event(),
        })()
        request.cancelled.set()
        raced = campaign._request_cancellation_evidence(request, {
            "requested_ns": 100, "cancel_call_ns": 200,
            "started_before_cancel": True, "thread_alive_before_cancel": True,
            "pre_cancel": {"error": None, "http_status": 200},
        })
        self.assertTrue(raced["finished_before_cancel"])
        self.assertNotEqual(campaign._cancellation_outcome(raced)[0], "pass")
        missing_order = campaign._request_cancellation_evidence(request, {
            "requested_ns": 100, "started_before_cancel": True,
            "thread_alive_before_cancel": True,
            "pre_cancel": {"error": None, "http_status": 200},
        })
        self.assertIsNone(missing_order["finished_before_cancel"])
        self.assertEqual(campaign._cancellation_outcome(missing_order)[0], "pending")

    def test_decode_cancellation_records_timing_transport_and_fresh_drain(self):
        value = campaign.Campaign.__new__(campaign.Campaign)
        value.cfg = {}
        value.active_deadline = time.monotonic() + 60
        value.telemetry = FakeTelemetry()
        value.http = type("HTTP", (), {
            "exact_prompt": lambda self, target, prefix: "prompt",
            "chat_payload": lambda self, prompt, **kwargs: {},
            "chat": lambda self, prompt, request_id: fixed_response(32_000),
        })()
        value._set_case = lambda *args, **kwargs: None
        value._reset_faults = lambda case_id: {"case_id": case_id, "errors": []}
        value.health_code = lambda: 200
        value.wait_for_api_idle = lambda: ("pass", {"requests_running": 0})
        value._drain_barriers = lambda: {rank: 100 + rank for rank in range(4)}
        value.wait_for_drain = lambda barriers: ("pass", {"fresh": True})

        class Thread:
            def __init__(self): self.alive = False
            def join(self, timeout=None): pass
            def is_alive(self): return self.alive
        class Request:
            def __init__(self, _client, _payload, request_id):
                self.request_id = request_id
                self.thread, self.started, self.first_byte = Thread(), threading.Event(), threading.Event()
                self.cancelled = threading.Event()
                self.result = self.error = None
                self.http_status = None
                self.response_headers = {}
                self.started_ns = self.first_byte_ns = self.finished_ns = None
                self.first_byte_hex = None
                self.guard_abort_cause = None
            def start(self):
                self.thread.alive = True
                self.started.set()
                self.first_byte.set()
                self.http_status = 200
                self.response_headers = {"content-type": "text/event-stream"}
                self.started_ns = self.first_byte_ns = time.time_ns()
                self.first_byte_hex = "64"
            def cancel(self):
                self.cancelled.set()
                self.thread.alive = False
                self.result = {"status": 200, "read_complete": False, "done": False,
                               "stream_complete": False, "finish_reasons": [],
                               "error_event": False, "parse_errors": []}
                self.error = "ConnectionAbortedError: intentional cancellation"
                self.finished_ns = time.time_ns()

        with patch.object(campaign, "CancelableRequest", Request):
            status, evidence = value._cancel(
                {"id": "cancel-decode", "parameters": {"stage": "decode"}}, 1)
        self.assertEqual(status, "pass")
        self.assertEqual(evidence["protocol_version"],
                         campaign.TARGETED_CASE_PROTOCOL_VERSION)
        receipt = evidence["cancelled"][0]
        self.assertTrue(receipt["interrupted"])
        self.assertTrue(receipt["cancellation"]["started_before_cancel"])
        self.assertLessEqual(receipt["cancellation"]["requested_ns"],
                             receipt["cancellation"]["cancel_call_ns"])
        self.assertLessEqual(receipt["cancellation"]["cancel_call_ns"],
                             receipt["cancellation"]["returned_ns"])
        self.assertEqual(receipt["http_status"], 200)
        self.assertEqual(receipt["response_headers"]["content-type"], "text/event-stream")
        self.assertEqual(evidence["reservation_and_staging_drain"]["status"], "pass")

    def test_cancellation_drain_failure_is_pending_only_after_targeted_cutoff(self):
        class Thread:
            def __init__(self): self.alive = False
            def join(self, timeout=None): pass
            def is_alive(self): return self.alive
        class Request:
            def __init__(self, _client, _payload, request_id):
                self.request_id = request_id
                self.thread, self.started = Thread(), threading.Event()
                self.first_byte, self.cancelled = threading.Event(), threading.Event()
                self.result = self.error = None
                self.http_status = None
                self.response_headers = {}
                self.started_ns = self.first_byte_ns = self.finished_ns = None
                self.first_byte_hex = self.guard_abort_cause = None
            def start(self):
                self.thread.alive = True
                self.started.set(); self.first_byte.set()
                self.http_status = 200
                self.started_ns = self.first_byte_ns = time.time_ns()
            def cancel(self):
                self.cancelled.set(); self.thread.alive = False
                self.result = {"status": 200, "body_text": "data: ",
                               "read_complete": False, "done": False,
                               "stream_complete": False, "finish_reasons": [],
                               "error_event": False, "parse_errors": []}
                self.error = "ConnectionAbortedError: intentional cancellation"
                self.finished_ns = time.time_ns()

        def run(expired):
            value = campaign.Campaign.__new__(campaign.Campaign)
            value.cfg = {}
            value.telemetry = FakeTelemetry()
            now = time.monotonic()
            value.active_deadline = now + 60
            value.targeted_stop_at = now - 1 if expired else now + 60
            value.http = type("HTTP", (), {
                "exact_prompt": lambda self, target, prefix: "prompt",
                "chat_payload": lambda self, prompt, **kwargs: {},
                "chat": lambda self, prompt, request_id: fixed_response(32_000),
            })()
            value._set_case = lambda *args, **kwargs: None
            value._reset_faults = lambda case_id: {"case_id": case_id, "errors": []}
            value.health_code = lambda: 200
            value.wait_for_api_idle = lambda: ("pass", {})
            value._drain_barriers = lambda: {rank: rank for rank in range(4)}
            value.wait_for_drain = lambda barriers: ("fail", {"reason": "late proof"})
            with patch.object(campaign, "CancelableRequest", Request):
                return value._cancel(
                    {"id": "cancel-decode", "parameters": {"stage": "decode"}}, 1)

        status, evidence = run(False)
        self.assertEqual(status, "fail")
        self.assertIn("drain failed", evidence["reason"])
        status, evidence = run(True)
        self.assertEqual(status, "pending")
        self.assertIn("targeted deadline", evidence["reason"])

    def test_cancellation_cutoff_is_pending_only_for_explicit_phase_abort(self):
        class Thread:
            def __init__(self): self.alive = False
            def join(self, timeout=None): pass
            def is_alive(self): return self.alive
        class Request:
            terminate_on_cancel = True
            def __init__(self, _client, _payload, request_id):
                self.request_id = request_id
                self.thread, self.started = Thread(), threading.Event()
                self.first_byte, self.cancelled = threading.Event(), threading.Event()
                self.result = self.error = None
                self.http_status = None
                self.response_headers = {}
                self.started_ns = self.first_byte_ns = self.finished_ns = None
                self.first_byte_hex = self.guard_abort_cause = None
            def start(self):
                self.thread.alive = True
                self.started.set()
                self.first_byte.set()
                self.started_ns = self.first_byte_ns = time.time_ns()
                self.http_status = 200
            def cancel(self):
                self.cancelled.set()
                if self.terminate_on_cancel:
                    self.thread.alive = False
                    self.result = {"read_complete": False, "done": False,
                                   "finish_reasons": [], "error_event": False,
                                   "parse_errors": [], "body_text": "data:"}
                    self.error = "ConnectionAbortedError: intentional"
                    self.finished_ns = time.time_ns()

        def run(error, *, safety=False, already_expired=False, terminate=True):
            value = campaign.Campaign.__new__(campaign.Campaign)
            value.cfg = {}
            value.telemetry = FakeTelemetry()
            now = time.monotonic()
            value.targeted_stop_at = now - 1 if already_expired else now + 60
            value.active_deadline = value.targeted_stop_at
            calls = []
            class HTTP:
                def exact_prompt(self, target, prefix): return "prompt"
                def chat_payload(self, prompt, **kwargs): return {}
                def chat(self, prompt, request_id):
                    calls.append(request_id)
                    value.targeted_stop_at = time.monotonic() - 1
                    if safety:
                        value.telemetry.safety_event.set()
                    raise error
            value.http = HTTP()
            value._set_case = lambda *args, **kwargs: None
            value._reset_faults = lambda case_id: {"case_id": case_id, "errors": []}
            Request.terminate_on_cancel = terminate
            with patch.object(campaign, "CancelableRequest", Request):
                status, evidence = value._cancel(
                    {"id": "cancel-decode", "parameters": {"stage": "decode"}}, 1)
            return status, evidence, calls

        status, evidence, calls = run(campaign.GuardAbortError("phase_deadline"))
        self.assertEqual(status, "pending")
        self.assertTrue(calls)
        self.assertIn("targeted deadline", evidence["reason"])
        self.assertIn("GuardAbortError", evidence["recovery_error"])

        status, _evidence, _calls = run(OSError("transport failed"))
        self.assertEqual(status, "fail")
        status, _evidence, _calls = run(
            campaign.GuardAbortError("phase_deadline"), safety=True)
        self.assertEqual(status, "fail")

        status, evidence, calls = run(
            AssertionError("reissue must not run"), already_expired=True, terminate=False)
        self.assertEqual(status, "pending")
        self.assertEqual(calls, [])
        self.assertFalse(evidence["cancelled"][0]["terminated"])

    def test_queue_cancellation_is_not_tied_by_global_depth_and_records_blocker_failure(self):
        value = campaign.Campaign.__new__(campaign.Campaign)
        value.cfg = {}
        value.active_deadline = time.monotonic() + 60
        value.telemetry = FakeTelemetry()
        value.http = type("HTTP", (), {
            "exact_prompt": lambda self, target, prefix: "prompt",
            "chat_payload": lambda self, prompt, **kwargs: {},
            "chat": lambda self, prompt, request_id: fixed_response(32_000),
        })()
        value._set_case = lambda *args, **kwargs: None
        value._fresh_waiting = lambda: 1
        value._reset_faults = lambda case_id: {"case_id": case_id, "errors": []}
        value.health_code = lambda: 200
        value.wait_for_api_idle = lambda: ("pass", {})
        value._drain_barriers = lambda: {rank: rank for rank in range(4)}
        value.wait_for_drain = lambda barriers: ("pass", {})

        class Thread:
            def __init__(self): self.alive = False
            def join(self, timeout=None): pass
            def is_alive(self): return self.alive
        class Request:
            def __init__(self, _client, _payload, request_id):
                self.request_id = request_id
                self.thread, self.started = Thread(), threading.Event()
                self.cancelled = threading.Event()
                self.first_byte = threading.Event()
                self.result = self.error = None
                self.http_status = None
                self.response_headers = {}
                self.started_ns = self.first_byte_ns = self.finished_ns = None
                self.first_byte_hex = self.guard_abort_cause = None
            def start(self):
                self.started.set()
                self.started_ns = time.time_ns()
                if self.request_id.endswith("block-0"):
                    self.http_status = 500
                    self.error = "HTTPError: blocker failed"
                    self.finished_ns = time.time_ns()
                else:
                    self.thread.alive = True
            def cancel(self):
                self.cancelled.set()
                if self.thread.alive:
                    self.thread.alive = False
                    self.result = {"read_complete": False, "done": False,
                                   "finish_reasons": [], "error_event": False,
                                   "parse_errors": []}
                    self.error = "ConnectionAbortedError: intentional"
                    self.finished_ns = time.time_ns()

        with patch.object(campaign, "CancelableRequest", Request):
            status, evidence = value._cancel(
                {"id": "cancel-queue", "parameters": {"stage": "queue"}}, 1)
        self.assertEqual(status, "fail")
        self.assertFalse(evidence["queue_waiting_tied_to_request"])
        self.assertEqual(len(evidence["blockers"]), 6)
        blocker = evidence["blockers"][0]
        self.assertEqual(blocker["http_status"], 500)
        self.assertEqual(blocker["cancellation"]["pre_cancel"]["error"],
                         "HTTPError: blocker failed")
        self.assertEqual(evidence["blocker_outcomes"][0]["status"], "fail")

    def _soak_fixture(self, directory, clock, hours):
        value = campaign.Campaign.__new__(campaign.Campaign)
        value.cfg = {"campaign_id": "unit-campaign", "soak_hours": hours}
        value.output = Path(directory)
        value.state = {}
        value.save = lambda: None
        value.telemetry = FakeTelemetry()
        value.telemetry.latest = {rank: {
            "time_ns": rank, "monotonic_ns": rank + 10,
            "mem_available_bytes": 1 << 30, "swap_total_bytes": 0,
            "swap_free_bytes": 0, "memory_pressure": {"some": {"avg10": 0.0}},
            "process": {"rss_bytes": 1000 + rank}, "oom": {"coverage": {}},
            "sources": {"host": {"status": "current"}},
        } for rank in range(4)}
        value.soak_at = 0
        value.restore_at = hours * 3600 + campaign.SOAK_FINAL_PROOF_RESERVE_SECONDS
        value.active_deadline = 0
        value.health_code = lambda: 200
        return value

    def test_soak_uses_actual_history_exact_replay_complete_receipts_and_two_hours(self):
        class Clock:
            now = 0.0
            def monotonic(self): return self.now
            def time_ns(self): return int((1_000_000 + self.now) * 1_000_000_000)
            def advance(self, seconds): self.now += seconds
            def sleep(self, seconds): self.advance(seconds)
        clock = Clock()

        class HTTP:
            base = "/v1"
            def __init__(self): self.responses, self.sent = 0, []
            def exact_messages(self, target, history, prefix):
                return [*copy.deepcopy(history),
                        {"role": "user", "content": f"{prefix} target={target}"}]
            def chat_payload_messages(self, messages, *, stream=False, max_tokens=16):
                return {"model": "model", "temperature": 0, "max_tokens": max_tokens,
                        "stream": stream,
                        "chat_template_kwargs": {"reasoning_effort": "low"},
                        "messages": copy.deepcopy(messages)}
            def json(self, path, payload, *, headers):
                self.sent.append(copy.deepcopy(payload))
                self.responses += 1
                started = clock.time_ns()
                clock.advance(300)
                content = f"actual-{self.responses}"
                return {"status": 200, "headers": {},
                        "body": {"choices": [{"message": {"role": "assistant",
                                                             "content": content}}],
                                 "usage": {"prompt_tokens": 512}},
                        "body_text": None, "started_ns": started,
                        "finished_ns": clock.time_ns()}

        class Signal:
            def wait(self, timeout=None):
                clock.advance(min(300, timeout or 300))
                return True
        class Thread:
            alive = True
            def join(self, timeout=None): pass
            def is_alive(self): return self.alive
        class Request:
            def __init__(self, client, payload, request_id):
                self.first_byte, self.thread = Signal(), Thread()
                self.http_status, self.response_headers = 200, {"content-type": "text/event-stream"}
                self.result = {"status": 200, "received_bytes": 1, "stream_complete": False}
                self.error = None
                self.started_ns = self.first_byte_ns = self.finished_ns = clock.time_ns()
                self.first_byte_hex = "64"
            def start(self): pass
            def cancel(self):
                self.thread.alive = False
                self.finished_ns = clock.time_ns()

        with tempfile.TemporaryDirectory() as directory:
            value = self._soak_fixture(directory, clock, 2)
            value.cfg["soak_stop_after_required_seconds"] = True
            value.restore_at = 3 * 3600 + campaign.SOAK_FINAL_PROOF_RESERVE_SECONDS
            value.soak_at = 3600
            value.http = HTTP()
            proofs = []
            value._soak_drain_proof = lambda label, **kwargs: proofs.append(
                (label, kwargs.get("deadline"))) or ("pass", {"label": label})
            with patch.object(campaign.time, "monotonic", side_effect=clock.monotonic), \
                    patch.object(campaign.time, "time_ns", side_effect=clock.time_ns), \
                    patch.object(campaign.time, "sleep", side_effect=clock.sleep), \
                    patch.object(campaign, "CancelableRequest", Request):
                status, evidence = value._soak({"id": "soak-mixed"})
            self.assertEqual(status, "pass")
            self.assertGreaterEqual(evidence["elapsed_seconds"], 2 * 3600)
            self.assertEqual(evidence["planned_load_end_monotonic"], 2 * 3600)
            self.assertEqual(evidence["hard_load_end_monotonic"], 3 * 3600)
            self.assertTrue(evidence["stop_after_required_seconds"])
            self.assertTrue(all(evidence["category_counts"][key] > 0 for key in
                                ("ordinary", "growth", "replay", "cancel", "idle")))
            self.assertEqual(proofs, [("cancel-10", 7200), ("cancel-20", 7200),
                                      ("soak-final", None)])
            receipt = Path(evidence["receipt"]["path"])
            rows = [json.loads(line) for line in receipt.read_text().splitlines()]
            self.assertEqual(len(rows), evidence["receipt"]["count"])
            self.assertEqual(hashlib.sha256(receipt.read_bytes()).hexdigest(),
                             evidence["receipt"]["sha256"])
            growth = next(row for row in rows if row["type"] == "growth"
                          and row["conversation_id"] == "conversation-0" and row["turn"] == 0)
            replay = next(row for row in rows if row["type"] == "replay"
                          and row["conversation_id"] == "conversation-0" and row["turn"] == 1)
            next_growth = next(row for row in rows if row["type"] == "growth"
                               and row["conversation_id"] == "conversation-0" and row["turn"] == 1)
            self.assertEqual(replay["request_payload"], growth["request_payload"])
            self.assertEqual(replay["request_payload_sha256"], growth["request_payload_sha256"])
            self.assertEqual(next_growth["request_payload"]["messages"][:-1], [
                *growth["request_payload"]["messages"],
                {"role": "assistant", "content": growth["accepted_assistant_content"]},
            ])
            ordinary_payloads = [row["request_payload"] for row in rows
                                 if row["type"] in {"growth", "replay"}]
            self.assertEqual(ordinary_payloads, value.http.sent)
            cancel = next(row for row in rows if row["type"] == "cancel")
            self.assertTrue(cancel["cancellation"]["first_byte"])
            self.assertTrue(cancel["cancellation"]["terminated"])
            self.assertEqual(cancel["http"]["status"], 200)
            idle = next(row for row in rows if row["type"] == "idle")
            self.assertGreaterEqual(idle["duration_seconds"], campaign.SOAK_IDLE_SECONDS)
            self.assertEqual(set(idle["resources_before"]), {"0", "1", "2", "3"})
            self.assertEqual(idle["resources_before"]["0"]["mem_available_bytes"], 1 << 30)
            self.assertIn("process", idle["resources_before"]["0"])
            self.assertIn("oom", idle["resources_before"]["0"])
            self.assertNotIn("memory", idle["resources_before"]["0"])

    def test_early_soak_runs_to_fixed_pre_restoration_boundary(self):
        class Clock:
            now = 0.0
            def monotonic(self): return self.now
            def time_ns(self): return int((50_000 + self.now) * 1_000_000_000)
        clock = Clock()
        planned_load_end = 8 * 3600 - 45 * 60 - campaign.SOAK_FINAL_PROOF_RESERVE_SECONDS
        class HTTP:
            base = "/v1"
            def exact_messages(self, target, history, prefix):
                return [*copy.deepcopy(history), {"role": "user", "content": prefix}]
            def chat_payload_messages(self, messages, *, stream=False, max_tokens=16):
                return {"messages": copy.deepcopy(messages), "max_tokens": max_tokens}
            def json(self, path, payload, *, headers):
                clock.now = min(planned_load_end, clock.now + 3600)
                return {"status": 200, "body": {"choices": [
                    {"message": {"content": "actual"}}]},
                    "finished_ns": clock.time_ns()}
        with tempfile.TemporaryDirectory() as directory:
            value = self._soak_fixture(directory, clock, 2)
            value.restore_at = 8 * 3600 - 45 * 60
            value.soak_at = (value.restore_at - campaign.SOAK_FINAL_PROOF_RESERVE_SECONDS
                             - 2 * 3600)
            value.http = HTTP()
            value._soak_drain_proof = lambda label, **kwargs: ("pass", {})
            with patch.object(campaign.time, "monotonic", side_effect=clock.monotonic), \
                    patch.object(campaign.time, "time_ns", side_effect=clock.time_ns), \
                    patch.object(campaign, "SOAK_CANCEL_EVERY", 1000), \
                    patch.object(campaign, "SOAK_IDLE_EVERY", 1000):
                status, evidence = value._soak({"id": "soak-mixed"})
            self.assertEqual(status, "pending")
            self.assertEqual(evidence["planned_load_end_monotonic"], planned_load_end)
            self.assertEqual(evidence["load_finished_monotonic"], planned_load_end)
            self.assertGreater(evidence["elapsed_seconds"], 2 * 3600)
            self.assertIn("cancel", evidence["missing_categories"])

    def test_opt_in_soak_stops_after_two_actual_hours_without_moving_hard_boundary(self):
        class Clock:
            now = 0.0
            def monotonic(self): return self.now
            def time_ns(self): return int((60_000 + self.now) * 1_000_000_000)
        clock = Clock()
        hard_load_end = 8 * 3600 - 45 * 60 - campaign.SOAK_FINAL_PROOF_RESERVE_SECONDS
        required = 2 * 3600
        class HTTP:
            base = "/v1"
            def exact_messages(self, target, history, prefix):
                return [*copy.deepcopy(history), {"role": "user", "content": prefix}]
            def chat_payload_messages(self, messages, *, stream=False, max_tokens=16):
                return {"messages": copy.deepcopy(messages), "max_tokens": max_tokens}
            def json(self, path, payload, *, headers):
                clock.now = min(required, clock.now + 1800)
                return {"status": 200, "body": {"choices": [
                    {"message": {"content": "actual"}}]},
                    "finished_ns": clock.time_ns()}
        with tempfile.TemporaryDirectory() as directory:
            value = self._soak_fixture(directory, clock, 2)
            value.cfg["soak_stop_after_required_seconds"] = True
            value.restore_at = 8 * 3600 - 45 * 60
            value.soak_at = hard_load_end - required
            value.http = HTTP()
            value._soak_drain_proof = lambda label, **kwargs: ("pass", {})
            with patch.object(campaign.time, "monotonic", side_effect=clock.monotonic), \
                    patch.object(campaign.time, "time_ns", side_effect=clock.time_ns), \
                    patch.object(campaign, "SOAK_CANCEL_EVERY", 1000), \
                    patch.object(campaign, "SOAK_IDLE_EVERY", 1000):
                status, evidence = value._soak({"id": "soak-mixed"})
            self.assertEqual(status, "pending")
            self.assertEqual(evidence["planned_load_end_monotonic"], required)
            self.assertEqual(evidence["load_finished_monotonic"], required)
            self.assertEqual(evidence["elapsed_seconds"], required)
            self.assertEqual(evidence["hard_load_end_monotonic"], hard_load_end)
            self.assertTrue(evidence["stop_after_required_seconds"])
            self.assertEqual(value.state["active_soak_receipt"][
                "planned_load_end_monotonic"], required)

            clock.now = value.soak_at + 1
            late = self._soak_fixture(directory, clock, 2)
            late.cfg["soak_stop_after_required_seconds"] = True
            late.restore_at = value.restore_at
            late.soak_at = value.soak_at
            with patch.object(campaign.time, "monotonic", side_effect=clock.monotonic), \
                    patch.object(campaign.time, "time_ns", side_effect=clock.time_ns):
                late_status, late_evidence = late._soak({"id": "soak-mixed"})
            self.assertEqual(late_status, "pending")
            self.assertIn("insufficient time", late_evidence["reason"])
            self.assertEqual(late_evidence["planned_load_end_monotonic"], hard_load_end)

    def test_assistant_null_content_with_reasoning_is_valid_but_malformed_is_not(self):
        valid, content, visible_null = campaign.Campaign._assistant_content({
            "body": {"choices": [{"message": {
                "role": "assistant", "content": None, "reasoning": "thinking"}}]}})
        self.assertTrue(valid)
        self.assertEqual(content, "")
        self.assertTrue(visible_null)
        malformed, _, _ = campaign.Campaign._assistant_content({
            "body": {"choices": [{"message": {"content": 17}}]}})
        self.assertFalse(malformed)
        missing, _, _ = campaign.Campaign._assistant_content({
            "body": {"choices": [{"message": {"reasoning": "thinking"}}]}})
        self.assertFalse(missing)

        class Clock:
            now = 0.0
            def monotonic(self): return self.now
            def time_ns(self): return int((40_000 + self.now) * 1_000_000_000)
        clock = Clock()
        class HTTP:
            base = "/v1"
            def exact_messages(self, target, history, prefix):
                return [{"role": "user", "content": prefix}]
            def chat_payload_messages(self, messages, *, stream=False, max_tokens=16):
                return {"messages": copy.deepcopy(messages), "max_tokens": max_tokens}
            def json(self, path, payload, *, headers):
                clock.now = 2
                return {"status": 200, "body": {"choices": [{"message": {
                    "role": "assistant", "content": None, "reasoning": "actual thought"}}]},
                    "started_ns": 1, "finished_ns": 2}
        with tempfile.TemporaryDirectory() as directory:
            value = self._soak_fixture(directory, clock, 1 / 3600)
            value.http = HTTP()
            value._soak_drain_proof = lambda label, **kwargs: ("pass", {})
            with patch.object(campaign.time, "monotonic", side_effect=clock.monotonic), \
                    patch.object(campaign.time, "time_ns", side_effect=clock.time_ns):
                status, evidence = value._soak({"id": "soak-mixed"})
            self.assertEqual(status, "pending")
            row = json.loads(Path(evidence["receipt"]["path"]).read_text())
            self.assertTrue(row["visible_content_null"])
            self.assertEqual(row["accepted_assistant_content"], "")
            self.assertEqual(row["http"]["body"]["choices"][0]["message"]["reasoning"],
                             "actual thought")
            self.assertFalse(evidence["request_errors"])

    def test_soak_missing_mix_is_pending_and_failed_response_is_preserved(self):
        class Clock:
            now = 0.0
            def monotonic(self): return self.now
            def time_ns(self): return int((1_000 + self.now) * 1_000_000_000)
        class HTTP:
            base = "/v1"
            status = 200
            def exact_messages(self, target, history, prefix):
                return [{"role": "user", "content": prefix}]
            def chat_payload_messages(self, messages, *, stream=False, max_tokens=16):
                return {"messages": copy.deepcopy(messages), "stream": stream,
                        "max_tokens": max_tokens}
            def json(self, path, payload, *, headers):
                clock.now = 2
                return {"status": self.status, "headers": {},
                        "body": {"choices": [{"message": {"content": "actual"}}]},
                        "started_ns": 1, "finished_ns": 2}

        with tempfile.TemporaryDirectory() as directory:
            clock = Clock()
            value = self._soak_fixture(directory, clock, 1 / 3600)
            value.http = HTTP()
            value._soak_drain_proof = lambda label, **kwargs: ("pass", {})
            with patch.object(campaign.time, "monotonic", side_effect=clock.monotonic), \
                    patch.object(campaign.time, "time_ns", side_effect=clock.time_ns):
                status, evidence = value._soak({"id": "soak-mixed"})
            self.assertEqual(status, "pending")
            self.assertIn("replay", evidence["missing_categories"])

            clock.now = 0
            value = self._soak_fixture(directory, clock, 1 / 3600)
            value.http = HTTP()
            value.http.status = 503
            value._soak_drain_proof = lambda label, **kwargs: ("pass", {})
            with patch.object(campaign.time, "monotonic", side_effect=clock.monotonic), \
                    patch.object(campaign.time, "time_ns", side_effect=clock.time_ns):
                status, evidence = value._soak({"id": "soak-mixed"})
            self.assertEqual(status, "fail")
            row = json.loads(Path(evidence["receipt"]["path"]).read_text().splitlines()[0])
            self.assertEqual(row["http"]["status"], 503)
            self.assertIsNotNone(row["request_payload"])

    def test_only_explicit_short_phase_abort_is_benign_near_soak_deadline(self):
        class Clock:
            now = 0.0
            def monotonic(self): return self.now
            def time_ns(self): return int((20_000 + self.now) * 1_000_000_000)
        clock = Clock()
        class HTTP:
            base = "/v1"
            error = ValueError("unrelated failure")
            def exact_messages(self, target, history, prefix):
                return [{"role": "user", "content": prefix}]
            def chat_payload_messages(self, messages, *, stream=False, max_tokens=16):
                return {"messages": copy.deepcopy(messages), "max_tokens": max_tokens}
            def json(self, path, payload, *, headers):
                clock.now = 1
                raise self.error

        with tempfile.TemporaryDirectory() as directory:
            for error, expected, aborted in (
                    (ValueError("unrelated failure"), "fail", False),
                    (campaign.GuardAbortError("phase_deadline"), "pending", True)):
                with self.subTest(error=type(error).__name__):
                    clock.now = 0
                    value = self._soak_fixture(directory, clock, 1 / 3600)
                    value.http = HTTP()
                    value.http.error = error
                    value._soak_drain_proof = lambda label, **kwargs: ("pass", {})
                    with patch.object(campaign.time, "monotonic", side_effect=clock.monotonic), \
                            patch.object(campaign.time, "time_ns", side_effect=clock.time_ns):
                        status, evidence = value._soak({"id": "soak-mixed"})
                    self.assertEqual(status, expected)
                    self.assertEqual(evidence["deadline_aborted"], aborted)
                    if aborted:
                        self.assertEqual(evidence["request_errors"], [])
                        self.assertTrue(evidence["deadline_events"])
                    else:
                        self.assertIn("ValueError", evidence["request_errors"][0])

    def test_receipt_write_failure_at_deadline_cannot_pass(self):
        class Clock:
            now = 0.0
            def monotonic(self): return self.now
            def time_ns(self): return int((30_000 + self.now) * 1_000_000_000)
        clock = Clock()
        class HTTP:
            base = "/v1"
            def exact_messages(self, target, history, prefix):
                return [{"role": "user", "content": prefix}]
            def chat_payload_messages(self, messages, *, stream=False, max_tokens=16):
                return {"messages": copy.deepcopy(messages), "max_tokens": max_tokens}
            def json(self, path, payload, *, headers):
                clock.now = 1
                return {"status": 200, "body": {"choices": [
                    {"message": {"content": "actual"}}]}, "finished_ns": clock.time_ns()}
        with tempfile.TemporaryDirectory() as directory:
            value = self._soak_fixture(directory, clock, 1 / 3600)
            value.http = HTTP()
            value._soak_drain_proof = lambda label, **kwargs: ("pass", {})
            with patch.object(campaign.time, "monotonic", side_effect=clock.monotonic), \
                    patch.object(campaign.time, "time_ns", side_effect=clock.time_ns), \
                    patch.object(campaign, "_append_private_event",
                                 side_effect=OSError("disk error")):
                status, evidence = value._soak({"id": "soak-mixed"})
            self.assertEqual(status, "fail")
            self.assertEqual(evidence["receipt"]["count"], 0)
            self.assertIn("SoakReceiptError", evidence["request_errors"][0])
            self.assertEqual(evidence["final_proof_status"], "pass")

    def test_receipt_hash_failure_keeps_reference_and_final_proof(self):
        class Clock:
            now = 0.0
            def monotonic(self): return self.now
            def time_ns(self): return int((35_000 + self.now) * 1_000_000_000)
        clock = Clock()
        class HTTP:
            base = "/v1"
            def exact_messages(self, target, history, prefix):
                return [{"role": "user", "content": prefix}]
            def chat_payload_messages(self, messages, *, stream=False, max_tokens=16):
                return {"messages": copy.deepcopy(messages), "max_tokens": max_tokens}
            def json(self, path, payload, *, headers):
                clock.now = 1
                return {"status": 200, "body": {"choices": [
                    {"message": {"content": "actual"}}]},
                    "finished_ns": clock.time_ns()}
        with tempfile.TemporaryDirectory() as directory:
            value = self._soak_fixture(directory, clock, 1 / 3600)
            value.http = HTTP()
            proofs = []
            value._soak_drain_proof = lambda label, **kwargs: proofs.append(label) or (
                "pass", {"label": label})
            with patch.object(campaign.time, "monotonic", side_effect=clock.monotonic), \
                    patch.object(campaign.time, "time_ns", side_effect=clock.time_ns), \
                    patch.object(campaign, "SOAK_CANCEL_EVERY", 1000), \
                    patch.object(campaign, "SOAK_IDLE_EVERY", 1000), \
                    patch.object(campaign, "sha256_file", side_effect=OSError("hash I/O")):
                status, evidence = value._soak({"id": "soak-mixed"})
            self.assertEqual(status, "fail")
            self.assertEqual(proofs, ["soak-final"])
            self.assertEqual(evidence["receipt"]["count"], 1)
            self.assertGreater(evidence["receipt"]["bytes"], 0)
            self.assertIsNone(evidence["receipt"]["sha256"])
            self.assertIn("hash I/O", evidence["receipt"]["error"])

    def test_idle_baseexception_writes_incomplete_row_then_propagates(self):
        class Clock:
            now = 0.0
            def monotonic(self): return self.now
            def time_ns(self): return int((36_000 + self.now) * 1_000_000_000)
        clock = Clock()
        class HTTP:
            base = "/v1"
            def exact_messages(self, target, history, prefix):
                return [{"role": "user", "content": prefix}]
            def chat_payload_messages(self, messages, *, stream=False, max_tokens=16):
                return {"messages": copy.deepcopy(messages), "max_tokens": max_tokens}
            def json(self, path, payload, *, headers):
                clock.now = 0.1
                return {"status": 200, "body": {"choices": [
                    {"message": {"content": "actual"}}]},
                    "finished_ns": clock.time_ns()}
        with tempfile.TemporaryDirectory() as directory:
            value = self._soak_fixture(directory, clock, 1)
            value.http = HTTP()
            value.health_code = lambda: (_ for _ in ()).throw(KeyboardInterrupt("operator"))
            with patch.object(campaign.time, "monotonic", side_effect=clock.monotonic), \
                    patch.object(campaign.time, "time_ns", side_effect=clock.time_ns), \
                    patch.object(campaign, "SOAK_CANCEL_EVERY", 1000), \
                    patch.object(campaign, "SOAK_IDLE_EVERY", 1):
                with self.assertRaises(KeyboardInterrupt):
                    value._soak({"id": "soak-mixed"})
            receipt = next(Path(directory).glob("soak-v*.jsonl"))
            rows = [json.loads(line) for line in receipt.read_text().splitlines()]
            self.assertEqual([row["type"] for row in rows], ["growth", "idle"])
            self.assertTrue(rows[1]["incomplete_due_to_base_exception"])
            self.assertIn("KeyboardInterrupt", rows[1]["error"])

    def test_soak_cancellation_failure_keeps_complete_attempt_receipt(self):
        class Clock:
            now = 0.0
            def monotonic(self): return self.now
            def time_ns(self): return int((10_000 + self.now) * 1_000_000_000)
        clock = Clock()
        class HTTP:
            def exact_messages(self, target, history, prefix):
                return [{"role": "user", "content": prefix}]
            def chat_payload_messages(self, messages, *, stream=False, max_tokens=16):
                return {"messages": copy.deepcopy(messages), "stream": stream,
                        "max_tokens": max_tokens}
        class Signal:
            def wait(self, timeout=None): return False
        class Thread:
            def join(self, timeout=None): pass
            def is_alive(self): return False
        class Request:
            def __init__(self, client, payload, request_id):
                self.first_byte, self.thread = Signal(), Thread()
                self.http_status, self.response_headers = None, {}
                self.result, self.error = None, "TimeoutError: no response"
                self.started_ns = self.first_byte_ns = None
                self.finished_ns = clock.time_ns()
                self.first_byte_hex = None
            def start(self): pass
            def cancel(self): pass
        with tempfile.TemporaryDirectory() as directory:
            value = self._soak_fixture(directory, clock, 1 / 3600)
            value.http = HTTP()
            value._soak_drain_proof = lambda label, **kwargs: ("pass", {})
            with patch.object(campaign.time, "monotonic", side_effect=clock.monotonic), \
                    patch.object(campaign.time, "time_ns", side_effect=clock.time_ns), \
                    patch.object(campaign, "SOAK_CANCEL_EVERY", 1), \
                    patch.object(campaign, "CancelableRequest", Request):
                status, evidence = value._soak({"id": "soak-mixed"})
            self.assertEqual(status, "fail")
            row = json.loads(Path(evidence["receipt"]["path"]).read_text())
            self.assertEqual(row["type"], "cancel")
            self.assertFalse(row["cancellation"]["first_byte"])
            self.assertTrue(row["cancellation"]["terminated"])
            self.assertEqual(row["http"]["error"], "TimeoutError: no response")
            self.assertIsNotNone(row["request_payload"])

    def test_early_transport_failure_waited_to_deadline_is_not_a_phase_abort(self):
        class Clock:
            now = 0.0
            def monotonic(self): return self.now
            def time_ns(self): return int((55_000 + self.now) * 1_000_000_000)
        clock = Clock()
        class HTTP:
            base = "/v1"
            def exact_messages(self, target, history, prefix):
                return [{"role": "user", "content": prefix}]
            def chat_payload_messages(self, messages, *, stream=False, max_tokens=16):
                return {"messages": copy.deepcopy(messages), "stream": stream,
                        "max_tokens": max_tokens}
        class Signal:
            def wait(self, timeout=None):
                clock.now = 1
                return False
        class Thread:
            def join(self, timeout=None): pass
            def is_alive(self): return False
        class Request:
            def __init__(self, client, payload, request_id):
                self.first_byte, self.thread = Signal(), Thread()
                self.http_status, self.response_headers = None, {}
                self.result = None
                self.error = "RemoteDisconnected: peer closed early"
                self.started_ns = 1
                self.first_byte_ns = self.first_byte_hex = None
                self.finished_ns = 2
                self.guard_abort_cause = None
            def start(self): pass
            def cancel(self): pass
        with tempfile.TemporaryDirectory() as directory:
            value = self._soak_fixture(directory, clock, 1 / 3600)
            value.http = HTTP()
            value._soak_drain_proof = lambda label, **kwargs: ("pass", {})
            with patch.object(campaign.time, "monotonic", side_effect=clock.monotonic), \
                    patch.object(campaign.time, "time_ns", side_effect=clock.time_ns), \
                    patch.object(campaign, "SOAK_CANCEL_EVERY", 1), \
                    patch.object(campaign, "CancelableRequest", Request):
                status, evidence = value._soak({"id": "soak-mixed"})
            self.assertEqual(status, "fail")
            self.assertFalse(evidence["deadline_aborted"])
            row = json.loads(Path(evidence["receipt"]["path"]).read_text())
            self.assertFalse(row["cancellation"]["requested_before_finish"])
            self.assertIsNone(row["guard_abort_cause"])
            self.assertTrue(any("produced no first byte" in error
                                for error in evidence["request_errors"]))

    def test_no_first_byte_phase_abort_needs_a_clean_post_cancel_proof(self):
        class Clock:
            now = 0.0
            def monotonic(self): return self.now
            def time_ns(self): return int((58_000 + self.now) * 1_000_000_000)
        clock = Clock()
        class HTTP:
            base = "/v1"
            def exact_messages(self, target, history, prefix):
                return [{"role": "user", "content": prefix}]
            def chat_payload_messages(self, messages, *, stream=False, max_tokens=16):
                return {"messages": copy.deepcopy(messages), "stream": stream,
                        "max_tokens": max_tokens}
        class Signal:
            def wait(self, timeout=None):
                clock.now = 1
                return False
        class Thread:
            alive = True
            join_timeouts = []
            def join(self, timeout=None):
                type(self).join_timeouts.append(timeout)
                if timeout is None or timeout > 0:
                    self.alive = False
            def is_alive(self): return self.alive
        class Request:
            def __init__(self, client, payload, request_id):
                self.first_byte, self.thread = Signal(), Thread()
                self.http_status, self.response_headers = None, {}
                self.result = None
                self.error = "GuardAbortError: phase deadline"
                self.guard_abort_cause = "phase_deadline"
                self.started_ns = 1
                self.first_byte_ns = self.first_byte_hex = None
                self.finished_ns = 2
            def start(self): pass
            def cancel(self): pass
        with tempfile.TemporaryDirectory() as directory:
            for proof, expected in (
                    (("pass", {"health_200": True}), "pending"),
                    (("fail", {"health_200": True,
                               "error": "RuntimeError: unrelated"}), "fail")):
                with self.subTest(proof=proof[0], expected=expected):
                    clock.now = 0
                    Thread.join_timeouts = []
                    value = self._soak_fixture(directory, clock, 1 / 3600)
                    value.http = HTTP()
                    value._soak_drain_proof = lambda label, **kwargs: proof
                    with patch.object(campaign.time, "monotonic", side_effect=clock.monotonic), \
                            patch.object(campaign.time, "time_ns", side_effect=clock.time_ns), \
                            patch.object(campaign, "SOAK_CANCEL_EVERY", 1), \
                            patch.object(campaign, "CancelableRequest", Request):
                        status, evidence = value._soak({"id": "soak-mixed"})
                    self.assertEqual(status, expected)
                    self.assertGreater(Thread.join_timeouts[0], 0)
                    row = json.loads(Path(evidence["receipt"]["path"]).read_text())
                    self.assertTrue(row["cancellation"]["deadline_truncated"])
                    self.assertEqual(evidence["deadline_aborted"], expected == "pending")
                    if expected == "fail":
                        self.assertIn("post-cancellation", evidence["request_errors"][0])

    def test_done_before_eof_cancel_does_not_count_as_interrupted(self):
        class Clock:
            now = 0.0
            def monotonic(self): return self.now
            def time_ns(self): return int((60_000 + self.now) * 1_000_000_000)
        clock = Clock()
        class HTTP:
            base = "/v1"
            def exact_messages(self, target, history, prefix):
                return [{"role": "user", "content": prefix}]
            def chat_payload_messages(self, messages, *, stream=False, max_tokens=16):
                return {"messages": copy.deepcopy(messages), "stream": stream,
                        "max_tokens": max_tokens}
        class Signal:
            def wait(self, timeout=None):
                clock.now += 0.5
                return True
        class Thread:
            def __init__(self, alive): self.alive = alive
            def join(self, timeout=None): pass
            def is_alive(self): return self.alive
        class Request:
            created = 0
            def __init__(self, client, payload, request_id):
                type(self).created += 1
                alive = True
                self.first_byte, self.thread = Signal(), Thread(alive)
                self.http_status, self.response_headers = 200, {}
                self.result = ({"read_complete": False, "stream_complete": True,
                                "done": True, "finish_reasons": ["stop"],
                                "body_text": "data: [DONE]\n\n"}
                               if type(self).created == 1 else
                               {"read_complete": False, "stream_complete": False,
                                "body_text": "data:"})
                self.error = "ConnectionAbortedError: cancelled"
                self.started_ns = self.first_byte_ns = self.finished_ns = clock.time_ns()
                self.first_byte_hex = "64"
            def start(self): pass
            def cancel(self):
                self.thread.alive = False
                self.finished_ns = clock.time_ns()
        with tempfile.TemporaryDirectory() as directory:
            value = self._soak_fixture(directory, clock, 1 / 3600)
            value.http = HTTP()
            value._soak_drain_proof = lambda label, **kwargs: ("pass", {})
            with patch.object(campaign.time, "monotonic", side_effect=clock.monotonic), \
                    patch.object(campaign.time, "time_ns", side_effect=clock.time_ns), \
                    patch.object(campaign, "SOAK_CANCEL_EVERY", 1), \
                    patch.object(campaign, "CancelableRequest", Request):
                status, evidence = value._soak({"id": "soak-mixed"})
            self.assertEqual(status, "pending")
            self.assertEqual(evidence["category_counts"]["cancel"], 1)
            rows = [json.loads(line) for line in
                    Path(evidence["receipt"]["path"]).read_text().splitlines()]
            self.assertTrue(rows[0]["cancellation"]["requested_before_finish"])
            self.assertFalse(rows[0]["cancellation"]["interrupted"])
            self.assertTrue(rows[1]["cancellation"]["interrupted"])

    def test_soak_refuses_late_start_and_final_proof_uses_fresh_barriers(self):
        with tempfile.TemporaryDirectory() as directory:
            value = campaign.Campaign.__new__(campaign.Campaign)
            value.cfg = {"campaign_id": "unit-campaign", "soak_hours": 2}
            value.output = Path(directory)
            value.state = {}
            value.save = lambda: None
            value.telemetry = FakeTelemetry()
            value.soak_at = time.monotonic() - 1
            value.restore_at = time.monotonic() + 8_000
            status, evidence = value._soak({"id": "soak-mixed"})
            self.assertEqual(status, "pending")
            self.assertIn("insufficient time", evidence["reason"])

            value.active_deadline = time.monotonic() + 5
            value.restore_at = time.monotonic() + 60
            value.health_code = lambda: 200
            value.wait_for_api_idle = lambda: ("pass", {"requests_running": 0})
            value._drain_barriers = lambda: {rank: 100 + rank for rank in range(4)}
            seen = []
            value.wait_for_drain = lambda barriers: seen.append(barriers) or ("pass", {"ranks": {}})
            status, proof = value._soak_drain_proof("final")
            self.assertEqual(status, "pass")
            self.assertEqual(seen, [{rank: 100 + rank for rank in range(4)}])
            self.assertEqual(proof["reservation_and_staging_drain"]["status"], "pass")

    def test_soak_proof_classifies_only_its_own_deadline_exhaustion_pending(self):
        self.assertEqual(campaign.PhaseDeadlineError.guard_abort_cause,
                         "phase_deadline")
        clock = {"now": 0.0}
        value = campaign.Campaign.__new__(campaign.Campaign)
        value.telemetry = FakeTelemetry()
        value.active_deadline = 5
        value.restore_at = 60
        value.health_code = lambda: 200
        value.wait_for_api_idle = lambda: ("pass", {})
        def deadline_barrier():
            clock["now"] = 10
            raise campaign.PhaseDeadlineError("proof deadline")
        value._drain_barriers = deadline_barrier
        with patch.object(campaign.time, "monotonic", side_effect=lambda: clock["now"]):
            status, evidence = value._soak_drain_proof("cancel", deadline=10)
        self.assertEqual(status, "pending")
        self.assertIn("PhaseDeadlineError", evidence["error"])

        clock["now"] = 0
        value._drain_barriers = lambda: (_ for _ in ()).throw(RuntimeError("unrelated"))
        with patch.object(campaign.time, "monotonic", side_effect=lambda: clock["now"]):
            status, _ = value._soak_drain_proof("cancel", deadline=10)
        self.assertEqual(status, "fail")

    def test_soak_deadline_truncated_cancel_defers_to_final_fresh_proof(self):
        class Clock:
            now = 0.0
            def monotonic(self): return self.now
            def time_ns(self): return int((70_000 + self.now) * 1_000_000_000)
        clock = Clock()
        class HTTP:
            base = "/v1"
            def exact_messages(self, target, history, prefix):
                return [{"role": "user", "content": prefix}]
            def chat_payload_messages(self, messages, *, stream=False, max_tokens=16):
                return {"messages": copy.deepcopy(messages), "stream": stream,
                        "max_tokens": max_tokens}
        class Signal:
            def wait(self, timeout=None):
                clock.now = 1
                return True
        class Thread:
            alive = True
            def join(self, timeout=None): self.alive = False
            def is_alive(self): return self.alive
        class Request:
            def __init__(self, client, payload, request_id):
                self.first_byte, self.thread = Signal(), Thread()
                self.http_status, self.response_headers = 200, {}
                self.result = {"read_complete": False, "done": False,
                               "finish_reasons": [], "error_event": False,
                               "parse_errors": [], "body_text": "d"}
                self.error = "ConnectionAbortedError: phase deadline"
                self.guard_abort_cause = "phase_deadline"
                self.started_ns = self.first_byte_ns = self.finished_ns = clock.time_ns()
                self.first_byte_hex = "64"
            def start(self): pass
            def cancel(self): self.thread.alive = False
        with tempfile.TemporaryDirectory() as directory:
            value = self._soak_fixture(directory, clock, 1 / 3600)
            value.http = HTTP()
            proofs = []
            def proof(label, **kwargs):
                proofs.append(label)
                if label.startswith("cancel-"):
                    return "pending", {
                        "health_200": True,
                        "api_idle": {"status": "pending", "evidence": {
                            "reason": "phase deadline cut short the idle proof"}},
                        "reservation_and_staging_drain": {"status": "pass"},
                    }
                return "pass", {"health_200": True, "fresh": True}
            value._soak_drain_proof = proof
            with patch.object(campaign.time, "monotonic", side_effect=clock.monotonic), \
                    patch.object(campaign.time, "time_ns", side_effect=clock.time_ns), \
                    patch.object(campaign, "SOAK_CANCEL_EVERY", 1), \
                    patch.object(campaign, "CancelableRequest", Request):
                status, evidence = value._soak({"id": "soak-mixed"})
            self.assertEqual(status, "pending")
            self.assertTrue(evidence["deadline_aborted"])
            self.assertEqual(evidence["category_counts"]["cancel"], 0)
            self.assertEqual(evidence["final_proof_status"], "pass")
            self.assertEqual(proofs, ["cancel-1", "soak-final"])
            row = json.loads(Path(evidence["receipt"]["path"]).read_text())
            self.assertTrue(row["cancellation"]["deadline_truncated"])

    def test_soak_baseexception_during_cancel_persists_partial_receipt_and_cleanup(self):
        class Clock:
            now = 0.0
            def monotonic(self): return self.now
            def time_ns(self): return int((80_000 + self.now) * 1_000_000_000)
        clock = Clock()
        class HTTP:
            base = "/v1"
            def exact_messages(self, target, history, prefix):
                return [{"role": "user", "content": prefix}]
            def chat_payload_messages(self, messages, *, stream=False, max_tokens=16):
                return {"messages": copy.deepcopy(messages), "stream": stream,
                        "max_tokens": max_tokens}
        class Signal:
            def wait(self, timeout=None): raise KeyboardInterrupt("operator")
        class Thread:
            alive = True
            def join(self, timeout=None): self.alive = False
            def is_alive(self): return self.alive
        class Request:
            instance = None
            def __init__(self, client, payload, request_id):
                type(self).instance = self
                self.first_byte, self.thread = Signal(), Thread()
                self.http_status, self.response_headers = 200, {"x-test": "partial"}
                self.result = {"received_bytes": 7, "body_text": "partial",
                               "read_complete": False, "stream_complete": False}
                self.error = "ConnectionAbortedError: partial"
                self.guard_abort_cause = None
                self.started_ns = self.first_byte_ns = 1
                self.finished_ns = None
                self.first_byte_hex = "70"
                self.cancelled = False
            def start(self): pass
            def cancel(self): self.cancelled = True
        with tempfile.TemporaryDirectory() as directory:
            value = self._soak_fixture(directory, clock, 1)
            value.http = HTTP()
            with patch.object(campaign.time, "monotonic", side_effect=clock.monotonic), \
                    patch.object(campaign.time, "time_ns", side_effect=clock.time_ns), \
                    patch.object(campaign, "SOAK_CANCEL_EVERY", 1), \
                    patch.object(campaign, "CancelableRequest", Request):
                with self.assertRaises(KeyboardInterrupt):
                    value._soak({"id": "soak-mixed"})
            self.assertTrue(Request.instance.cancelled)
            active = value.state["active_soak_receipt"]
            self.assertEqual(active["status"], "interrupted")
            self.assertEqual(active["count"], 1)
            row = json.loads(Path(active["path"]).read_text())
            self.assertEqual(row["http"]["result"]["body_text"], "partial")
            self.assertTrue(row["cancellation"]["terminated"])
            self.assertTrue(row["incomplete_due_to_base_exception"])
            self.assertIn("KeyboardInterrupt", row["terminal_error"])

    def test_soak_rechecks_latest_start_after_telemetry_restart(self):
        class Clock:
            now = 0.0
            def monotonic(self): return self.now
            def time_ns(self): return int((90_000 + self.now) * 1_000_000_000)
        clock = Clock()
        class Telemetry(FakeTelemetry):
            def start(self, interval): clock.now = 2
        with tempfile.TemporaryDirectory() as directory:
            value = self._soak_fixture(directory, clock, 1 / 3600)
            value.telemetry = Telemetry()
            value.soak_at = 1
            with patch.object(campaign.time, "monotonic", side_effect=clock.monotonic), \
                    patch.object(campaign.time, "time_ns", side_effect=clock.time_ns):
                status, evidence = value._soak({"id": "soak-mixed"})
            self.assertEqual(status, "pending")
            self.assertEqual(value.state["active_soak_receipt"]["status"],
                             "late_after_telemetry_start")
            self.assertEqual(evidence["started_monotonic"], 2)

    def test_over_context_prompt_itself_exceeds_the_maximum(self):
        value = campaign.Campaign.__new__(campaign.Campaign)
        seen = []
        value.http = type("HTTP", (), {
            "base": "/v1", "model": "model",
            "exact_prompt": lambda self, target, prefix: seen.append(target) or "prompt",
            "chat": lambda self, prompt, request_id: {"status": 400},
        })()
        value.health_code = lambda: 200
        status, _ = value._api(
            {"id": "api-over-context", "parameters": {"variant": "over-context"}}, 1)
        self.assertEqual(status, "pass")
        self.assertEqual(seen, [262_145])

    def test_queue_preserves_success_error_and_guard_abort_receipts(self):
        value = campaign.Campaign.__new__(campaign.Campaign)
        value.active_deadline = time.monotonic() + 10
        value.telemetry = FakeTelemetry()
        value._fresh_waiting = lambda: 1
        value.wait_for_api_idle = lambda: ("pass", {"requests_running": 0})
        value.health_code = lambda: 200
        class HTTP:
            def exact_prompt(self, target, prefix): return f"{target}:{prefix}"
            def chat_payload(self, prompt, **kwargs): return {"prompt": prompt, **kwargs}
            def chat(self, prompt, *, request_id):
                time.sleep(0.02)
                if request_id.endswith("-1"):
                    raise OSError("connection reset")
                if request_id.endswith("-2"):
                    raise campaign.GuardAbortError("safety_event")
                return {"status": 200, "body": {"choices": [{"message": {"content": "ok"}}]}}
        value.http = HTTP()
        class Thread:
            alive = True
            def join(self, timeout=None): pass
            def is_alive(self): return self.alive
        class Blocker:
            def __init__(self, client, payload, request_id):
                self.request_id, self.thread = request_id, Thread()
                self.http_status, self.response_headers = 200, {}
                self.result = {"read_complete": False, "stream_complete": False}
                self.error = None
                self.started_ns = self.first_byte_ns = self.finished_ns = 1
                self.first_byte_hex = "64"
            def start(self): pass
            def cancel(self):
                self.error = "ConnectionAbortedError: cancelled"
                self.thread.alive = False
        with patch.object(campaign, "CancelableRequest", Blocker):
            status, evidence = value._queue(
                {"id": "queue-wave-test", "parameters": {"clients": 3}}, 1)
        self.assertEqual(status, "fail")
        self.assertEqual(evidence["response_counts"], {
            "requested": 3, "submitted": 3, "recorded": 3, "http_200": 1,
            "errors": 2, "guard_aborts": 1,
            "client_capacity_errors": 0, "submission_unknown": 0,
            "blocker_failures_before_cancel": 0, "blocker_capacity_errors": 0,
            "blocker_phase_aborts": 0})
        self.assertEqual({row["request_id"] for row in evidence["responses"]}, {
            "queue-wave-test-wave-0", "queue-wave-test-wave-1",
            "queue-wave-test-wave-2"})
        self.assertEqual(len(evidence["blockers"]), 6)
        self.assertTrue(evidence["blockers_terminated"])

    def test_bounded_admission_queue_requires_exact_overload_and_fresh_idle_gauges(self):
        value = campaign.Campaign.__new__(campaign.Campaign)
        value.cfg = {"bounded_admission": True}
        value.active_deadline = time.monotonic() + 10
        value.telemetry = FakeTelemetry()
        value.telemetry.latest[0] = {
            "sources": {
                "endpoint": {"status": "current", "age_ns": 1},
                "admission": {"status": "current", "age_ns": 1},
            },
            "api": {"requests_running": 0, "requests_waiting": 0,
                    "admission_requests_active": 0,
                    "admission_requests_waiting": 3,
                    "admission_requests_inflight": 3},
            "connections": {"established": 0},
        }
        self.assertEqual(value._fresh_waiting(), 3)
        self.assertEqual(value._fresh_admission_waiting(), 3)
        release = threading.Event()
        value._fresh_admission_active = lambda: 6
        value._fresh_admission_waiting = lambda: release.set() or 128
        value.wait_for_api_idle = lambda: ("pass", {"admission_requests_inflight": 0})
        value.health_code = lambda: 200

        class HTTP:
            invalid_overload = False
            def exact_prompt(self, target, prefix): return f"{target}:{prefix}"
            def chat_payload(self, prompt, **kwargs): return {"prompt": prompt, **kwargs}
            def chat(self, prompt, request_id):
                self.assert_short = prompt.startswith("512:")
                release.wait(timeout=1)
                index = int(request_id.rsplit("-", 1)[1])
                if index < 4:
                    if self.invalid_overload:
                        return {"status": 503, "headers": {"Retry-After": "1"},
                                "body": {"error": {"type": "server_error",
                                                   "code": "admission_request_timeout"}}}
                    return {"status": 503, "headers": {"Retry-After": "1"},
                            "body": {"error": {"type": "overloaded",
                                               "code": "admission_queue_full"}}}
                return fixed_response(512)
        value.http = HTTP()

        class Thread:
            alive = True
            def join(self, timeout=None): pass
            def is_alive(self): return self.alive
        class Blocker:
            def __init__(self, client, payload, request_id):
                self.request_id, self.thread = request_id, Thread()
                self.http_status, self.response_headers = 200, {}
                self.result = {"read_complete": False, "stream_complete": False}
                self.error = None
                self.started_ns = self.first_byte_ns = self.finished_ns = 1
                self.first_byte_hex = "64"
            def start(self): pass
            def cancel(self): self.thread.alive = False

        case = {"id": "admission-overflow", "parameters": {
            "clients": campaign.ADMISSION_MAX_ACTIVE + campaign.ADMISSION_MAX_QUEUED
                       + campaign.ADMISSION_OVERFLOW_EXTRA,
            "admission_overflow": True}}
        with patch.object(campaign, "CancelableRequest", Blocker):
            status, evidence = value._queue(case, 1)
        self.assertEqual(status, "pass", evidence)
        self.assertTrue(evidence["admission_queue_observed"])
        self.assertEqual(evidence["response_counts"]["intentional_overload_rejections"], 4)
        self.assertEqual(evidence["response_counts"]["recorded"], case["parameters"]["clients"])
        self.assertEqual(evidence["response_counts"]["errors"], 0)
        self.assertTrue(all(row.get("status") in {200, 503}
                            for row in evidence["responses"]))

        value.http.invalid_overload = True
        with patch.object(campaign, "CancelableRequest", Blocker):
            invalid_status, invalid_evidence = value._queue(case, 2)
        self.assertEqual(invalid_status, "fail")
        self.assertEqual(
            invalid_evidence["response_counts"]["intentional_overload_rejections"], 0)
        value.http.invalid_overload = False
        with patch.object(campaign, "CancelableRequest", Blocker):
            below_cap_status, below_cap = value._queue(
                {"id": "queue-wave-8", "parameters": {"clients": 8}}, 1)
        self.assertEqual(below_cap_status, "fail")
        self.assertIn("fits its configured queue", below_cap["reason"])

        valid = {"status": 503, "headers": {"retry-after": "2"},
                 "body": {"error": {"type": "overloaded",
                                    "code": "admission_queue_timeout"}}}
        self.assertEqual(campaign._intentional_overload_code(valid),
                         "admission_queue_timeout")
        for invalid in (
            {**valid, "headers": {}},
            {**valid, "body": {"error": {"type": "server_error",
                                          "code": "admission_request_timeout"}}},
            {**valid, "body": {"error": {"type": "overloaded", "code": "other"}}},
        ):
            self.assertIsNone(campaign._intentional_overload_code(invalid))

        value.wait_for_api_idle = campaign.Campaign.wait_for_api_idle.__get__(
            value, campaign.Campaign)
        value.telemetry.latest[0]["api"].update(
            admission_requests_waiting=0, admission_requests_inflight=1)
        value.active_deadline = time.monotonic() + 0.02
        idle_status, _ = value.wait_for_api_idle(timeout=0.01)
        self.assertNotEqual(idle_status, "pass")
        value.telemetry.latest[0]["api"].update(
            admission_requests_inflight=0, admission_requests_active=0)
        value.active_deadline = time.monotonic() + 1
        self.assertEqual(value.wait_for_api_idle(timeout=0.1)[0], "pass")

    def test_bounded_queue_waits_for_six_fresh_active_blockers_before_wave(self):
        value = campaign.Campaign.__new__(campaign.Campaign)
        value.cfg = {"bounded_admission": True}
        value.active_deadline = time.monotonic() + 0.02
        value.telemetry = FakeTelemetry()
        value._fresh_admission_active = lambda: 5
        value.wait_for_api_idle = lambda: ("pending", {"reason": "deadline"})
        value.health_code = lambda: 200
        chats = []
        value.http = type("HTTP", (), {
            "exact_prompt": lambda self, target, prefix: "prompt",
            "chat_payload": lambda self, prompt, **kwargs: {},
            "chat": lambda self, prompt, request_id: chats.append(request_id),
        })()

        class Thread:
            def __init__(self): self.alive = True
            def join(self, timeout=None): pass
            def is_alive(self): return self.alive
        class Blocker:
            created = []
            def __init__(self, client, payload, request_id):
                self.request_id, self.thread = request_id, Thread()
                self.http_status, self.response_headers = None, {}
                self.result = self.error = None
                self.started_ns = self.first_byte_ns = self.finished_ns = None
                self.first_byte_hex = None
                type(self).created.append(self)
            def start(self): pass
            def cancel(self): self.thread.alive = False

        with patch.object(campaign, "CancelableRequest", Blocker):
            status, evidence = value._queue(
                {"id": "admission-overflow", "parameters": {
                    "clients": 4, "admission_overflow": True}}, 1)
        self.assertEqual(status, "pending")
        self.assertFalse(evidence["blocker_activation"]["established"])
        self.assertEqual(evidence["blocker_activation"]["last_active"], 5)
        self.assertFalse(chats)
        self.assertTrue(evidence["blockers_terminated"])
        self.assertTrue(all(row["submitted"] is False
                            for row in evidence["responses"]))
        self.assertTrue(all(not blocker.thread.is_alive() for blocker in Blocker.created))

    def test_queue_collects_all_inflight_receipts_after_safety_stop(self):
        value = campaign.Campaign.__new__(campaign.Campaign)
        value.active_deadline = time.monotonic() + 10
        value.telemetry = FakeTelemetry()
        value._fresh_waiting = lambda: 1
        value.wait_for_api_idle = lambda: ("pass", {})
        value.health_code = lambda: 200
        class HTTP:
            def exact_prompt(self, target, prefix): return "prompt"
            def chat_payload(self, prompt, **kwargs): return {"prompt": prompt}
            def chat(self, prompt, *, request_id):
                value.telemetry.safety_event.wait(timeout=1)
                raise campaign.GuardAbortError("safety_event")
        value.http = HTTP()
        class Thread:
            alive = True
            def join(self, timeout=None): pass
            def is_alive(self): return self.alive
        class Blocker:
            def __init__(self, client, payload, request_id):
                self.request_id, self.thread = request_id, Thread()
                self.http_status, self.response_headers = None, {}
                self.result = self.error = None
                self.started_ns = self.first_byte_ns = self.finished_ns = None
                self.first_byte_hex = None
            def start(self): pass
            def cancel(self): self.thread.alive = False
        timer = threading.Timer(0.05, value.telemetry.safety_event.set)
        timer.start()
        try:
            with patch.object(campaign, "CancelableRequest", Blocker):
                status, evidence = value._queue(
                    {"id": "queue-safety", "parameters": {"clients": 3}}, 1)
        finally:
            timer.join(timeout=1)
        self.assertEqual(status, "fail")
        self.assertEqual(evidence["response_counts"]["recorded"], 3)
        self.assertEqual(evidence["response_counts"]["guard_aborts"], 3)
        self.assertTrue(all(row["guard_abort_cause"] == "safety_event"
                            for row in evidence["responses"]))

    def test_queue_rejects_pre_cancel_blocker_failures_but_preserves_phase_cutoff(self):
        value = campaign.Campaign.__new__(campaign.Campaign)
        value.active_deadline = time.monotonic() + 10
        value.telemetry = FakeTelemetry()
        value._fresh_waiting = lambda: 1
        value.wait_for_api_idle = lambda: ("pass", {})
        value.health_code = lambda: 200
        value.http = type("HTTP", (), {
            "exact_prompt": lambda self, target, prefix: "prompt",
            "chat_payload": lambda self, prompt, **kwargs: {},
            "chat": lambda self, prompt, request_id: {
                "status": 200, "body": {"choices": [{}]}},
        })()
        class Thread:
            def __init__(self, alive=True): self.alive = alive
            def join(self, timeout=None): pass
            def is_alive(self): return self.alive
        class Blocker:
            mode = "failure"
            created = 0
            def __init__(self, client, payload, request_id):
                index = type(self).created
                type(self).created += 1
                self.request_id, self.response_headers = request_id, {}
                self.http_status = 503 if index == 0 and self.mode == "failure" else None
                self.error = ("OSError: early reset"
                              if index == 1 and self.mode == "failure"
                              else "ConnectionAbortedError: phase deadline"
                              if index == 0 and self.mode == "phase" else None)
                self.guard_abort_cause = (
                    "phase_deadline" if index == 0 and self.mode == "phase" else None)
                self.thread = Thread(alive=self.error is None)
                self.result = None
                self.started_ns = self.first_byte_ns = self.finished_ns = None
                self.first_byte_hex = None
            def start(self): pass
            def cancel(self): self.thread.alive = False
        with patch.object(campaign, "CancelableRequest", Blocker):
            status, evidence = value._queue(
                {"id": "queue-blocker-fail", "parameters": {"clients": 2}}, 1)
        self.assertEqual(status, "fail")
        self.assertEqual(evidence["response_counts"][
            "blocker_failures_before_cancel"], 2)

        Blocker.mode = "phase"
        Blocker.created = 0
        value.active_deadline = time.monotonic() - 1
        with patch.object(campaign, "CancelableRequest", Blocker):
            status, evidence = value._queue(
                {"id": "queue-blocker-cutoff", "parameters": {"clients": 2}}, 1)
        self.assertEqual(status, "pending")
        self.assertEqual(evidence["response_counts"]["blocker_phase_aborts"], 1)

    def test_queue_cutoff_rejects_invalid_completions_and_allows_capacity_phase_mix(self):
        value = campaign.Campaign.__new__(campaign.Campaign)
        value.active_deadline = time.monotonic() - 1
        value.telemetry = FakeTelemetry()
        value._fresh_waiting = lambda: 1
        value.wait_for_api_idle = lambda: ("pending", {
            "reason": "phase deadline cut short the idle proof"})
        value.health_code = lambda: 200
        class HTTP:
            mode = "http500"
            def exact_prompt(self, target, prefix): return "prompt"
            def chat_payload(self, prompt, **kwargs): return {"prompt": prompt}
            def chat(self, prompt, request_id):
                index = int(request_id.rsplit("-", 1)[1])
                if index == 1:
                    raise campaign.GuardAbortError("phase_deadline")
                if self.mode == "capacity":
                    raise campaign.ClientCapacityError("local thread limit")
                if self.mode == "malformed200":
                    return {"status": 200, "body": {"choices": []}}
                return {"status": 500, "body": {"error": "server"}}
        value.http = HTTP()
        class Thread:
            alive = True
            def join(self, timeout=None): pass
            def is_alive(self): return self.alive
        class Blocker:
            def __init__(self, client, payload, request_id):
                self.request_id, self.thread = request_id, Thread()
                self.http_status, self.response_headers = None, {}
                self.result = self.error = None
                self.started_ns = self.first_byte_ns = self.finished_ns = None
                self.first_byte_hex = None
            def start(self): pass
            def cancel(self): self.thread.alive = False
        case = {"id": "queue-cutoff-mixed", "parameters": {"clients": 2}}
        with patch.object(campaign, "CancelableRequest", Blocker):
            for mode in ("http500", "malformed200"):
                with self.subTest(mode=mode):
                    value.http.mode = mode
                    status, evidence = value._queue(case, 1)
                    self.assertEqual(status, "fail")
                    self.assertEqual(evidence["response_counts"]["guard_aborts"], 1)
            value.http.mode = "capacity"
            status, evidence = value._queue(case, 1)
        self.assertEqual(status, "pending")
        self.assertEqual(evidence["response_counts"]["guard_aborts"], 1)
        self.assertEqual(evidence["response_counts"]["client_capacity_errors"], 1)

    def test_queue_submission_and_blocker_start_failures_keep_every_attempt(self):
        value = campaign.Campaign.__new__(campaign.Campaign)
        value.active_deadline = time.monotonic() + 10
        value.telemetry = FakeTelemetry()
        value._fresh_waiting = lambda: 1
        value.wait_for_api_idle = lambda: ("pass", {})
        value.health_code = lambda: 200
        class HTTP:
            mode = "valid"
            def exact_prompt(self, target, prefix): return "prompt"
            def chat_payload(self, prompt, **kwargs): return {"prompt": prompt}
            def chat(self, prompt, request_id):
                if self.mode == "http500":
                    return {"status": 500, "body": {"error": "server"}}
                if self.mode == "malformed200":
                    return {"status": 200, "body": {"choices": []}}
                return {"status": 200, "body": {"choices": [{}]}}
        value.http = HTTP()
        class Thread:
            def __init__(self): self.started = self.alive = False
            def join(self, timeout=None):
                if not self.started:
                    raise AssertionError("joined an unstarted blocker")
            def is_alive(self): return self.alive
        class Blocker:
            created = []
            fail_index = None
            def __init__(self, client, payload, request_id):
                self.index = len(type(self).created)
                type(self).created.append(self)
                self.request_id, self.thread = request_id, Thread()
                self.http_status, self.response_headers = None, {}
                self.result = self.error = None
                self.started_ns = self.first_byte_ns = self.finished_ns = None
                self.first_byte_hex = None
            def start(self):
                if self.index == type(self).fail_index:
                    raise OSError("cannot start blocker thread")
                self.thread.started = self.thread.alive = True
            def cancel(self):
                if not self.thread.started:
                    raise AssertionError("cancelled an unstarted blocker")
                self.thread.alive = False
        class Future:
            def __init__(self, result=None, *, done=True):
                self.value, self.finished, self.was_cancelled = result, done, False
            def done(self): return self.finished or self.was_cancelled
            def cancel(self):
                if self.finished:
                    return False
                self.was_cancelled = True
                return True
            def cancelled(self): return self.was_cancelled
            def result(self): return self.value
        class AfterEnqueuePool:
            def __init__(self, max_workers): self.calls = 0
            def submit(self, function, *args):
                self.calls += 1
                if self.calls == 3:
                    # CPython queues the work item before _adjust_thread_count;
                    # model a worker completing it even though submit raises.
                    function(*args)
                    raise OSError("client thread limit")
                return Future(function(*args))
            def shutdown(self, wait=True, cancel_futures=False):
                self.wait, self.cancel_futures = wait, cancel_futures

        with patch.object(campaign, "CancelableRequest", Blocker), \
                patch.object(campaign, "ThreadPoolExecutor", AfterEnqueuePool):
            status, evidence = value._queue(
                {"id": "queue-submit", "parameters": {"clients": 4}}, 1)
        self.assertEqual(status, "pending")
        self.assertEqual(len(evidence["responses"]), 4)
        self.assertEqual([row["submitted"] for row in evidence["responses"]],
                         [True, True, None, False])
        self.assertEqual(evidence["response_counts"]["requested"], 4)
        self.assertEqual(evidence["response_counts"]["submitted"], 2)
        self.assertEqual(evidence["response_counts"]["submission_unknown"], 1)
        self.assertEqual(evidence["response_counts"]["http_200"], 3)
        self.assertEqual(evidence["response_counts"]["errors"], 1)
        self.assertEqual(evidence["responses"][2]["dispatch_state"], "entered")
        self.assertEqual(evidence["responses"][3]["error_origin"], "client_capacity")
        self.assertTrue(evidence["blockers_terminated"])

        for mode in ("http500", "malformed200"):
            with self.subTest(mode=mode):
                value.http.mode = mode
                with patch.object(campaign, "CancelableRequest", Blocker), \
                        patch.object(campaign, "ThreadPoolExecutor", AfterEnqueuePool):
                    mixed_status, mixed = value._queue(
                        {"id": f"queue-submit-{mode}",
                         "parameters": {"clients": 4}}, 1)
                self.assertEqual(mixed_status, "fail")
                self.assertEqual(mixed["response_counts"]["submission_unknown"], 1)
                self.assertTrue(any(row.get("status") in {200, 500}
                                    for row in mixed["responses"]
                                    if row.get("error") is None))
        value.http.mode = "valid"

        class CancelPendingPool:
            def __init__(self, max_workers): self.calls = 0; self.pending = []
            def submit(self, function, *args):
                self.calls += 1
                if self.calls == 2:
                    raise OSError("client thread limit after enqueue")
                future = Future(done=False)
                self.pending.append(future)
                return future
            def shutdown(self, wait=True, cancel_futures=False):
                if cancel_futures:
                    for future in self.pending:
                        future.cancel()
        with patch.object(campaign, "CancelableRequest", Blocker), \
                patch.object(campaign, "ThreadPoolExecutor", CancelPendingPool):
            status, evidence = value._queue(
                {"id": "queue-submit-pending", "parameters": {"clients": 4}}, 1)
        self.assertEqual(status, "pending")
        self.assertEqual([row["submitted"] for row in evidence["responses"]],
                         [True, None, False, False])
        self.assertEqual([row["dispatch_state"] for row in evidence["responses"]],
                         ["not_dispatched", "submission_unknown",
                          "not_dispatched", "not_dispatched"])
        self.assertEqual(evidence["response_counts"]["client_capacity_errors"], 4)

        Blocker.created = []
        Blocker.fail_index = 2
        with patch.object(campaign, "CancelableRequest", Blocker):
            status, evidence = value._queue(
                {"id": "queue-blocker", "parameters": {"clients": 4}}, 1)
        self.assertEqual(status, "pending")
        self.assertEqual(len(evidence["responses"]), 4)
        self.assertTrue(all(not row["submitted"] for row in evidence["responses"]))
        self.assertEqual([row["started"] for row in evidence["blockers"]],
                         [True, True, False, False, False, False])
        self.assertEqual(evidence["response_counts"]["blocker_capacity_errors"], 4)
        self.assertTrue(evidence["blockers_terminated"])

    def test_combinations_require_prerequisites_and_dispatch_only_reviewed_case(self):
        value = campaign.Campaign.__new__(campaign.Campaign)
        value.runtime_config_identity = {"runtime_variant": "default"}
        passed = [{"status": "pass", "runtime_config_identity":
                   value.runtime_config_identity} for _ in range(3)]
        value.state = {"cases": {
            "cache-eio": {"runs": passed, "target_repetitions": 3,
                          "status": "pass"},
            "cancel-restore": {"runs": list(passed), "target_repetitions": 3,
                               "status": "pass"},
        }}
        case = {"id": "combination-cache-eio-cancel-restore",
                "parameters": {"prerequisites": ["cache-eio", "cancel-restore"]}}
        with patch.object(campaign, "run_cache_eio_cancel_restore") as run:
            status, evidence = value._combination(case, 1)
        self.assertEqual(status, "pending")
        self.assertEqual(evidence["prerequisites"]["cancel-restore"]
                         ["successful_repetitions"], 0)
        run.assert_not_called()

        current_cancel = [{"status": "pass", "runtime_config_identity":
                           value.runtime_config_identity,
                           "evidence": {"protocol_version":
                                        campaign.TARGETED_CASE_PROTOCOL_VERSION}}
                          for _ in range(3)]
        value.state["cases"]["cancel-restore"]["runs"] = current_cancel
        with patch.object(campaign, "run_cache_eio_cancel_restore",
                          return_value=("pass", {"combined": True})) as run:
            status, evidence = value._combination(case, 1)
        self.assertEqual(status, "pass")
        self.assertTrue(evidence["combined"])
        self.assertEqual(set(evidence["prerequisites"]), {"cache-eio", "cancel-restore"})
        run.assert_called_once()

        value.state["cases"]["cancel-restore"]["runs"] = current_cancel[:2]
        with patch.object(campaign, "run_cache_eio_cancel_restore") as run:
            status, evidence = value._combination(case, 2)
        self.assertEqual(status, "pending")
        self.assertIn("prerequisites", evidence["reason"])
        run.assert_not_called()

        value.state["cases"].update({
            "queue-wave-64": {"runs": list(passed), "target_repetitions": 3},
            "slow-client-slow-read": {"runs": list(passed), "target_repetitions": 3},
            "cancel-queue": {"runs": list(current_cancel), "target_repetitions": 3},
        })
        queue_case = {"id": "combination-queue-slow-cancel", "parameters": {
            "prerequisites": ["queue-wave-64", "slow-client-slow-read", "cancel-queue"]}}
        status, evidence = value._combination(queue_case, 1)
        self.assertEqual(status, "pending")
        self.assertIn("no reviewed synchronized driver", evidence["reason"])

    def test_default_identity_does_not_inherit_overlay_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            value = self.make_campaign(directory)
            value.end_monotonic = time.monotonic() + 5
            value._controller = lambda overlay, command, **kwargs: {
                "returncode": 0, "health_200_monotonic": time.monotonic()}
            value._gates = lambda since: {"pass": True}
            completed = type("Completed", (), {"returncode": 0, "stdout": "ok"})()
            with patch.dict(os.environ, {"TP4_ENV": "overlay.env"}), \
                    patch.object(campaign.subprocess, "run", return_value=completed) as invoked:
                result = value.restore_default()
            self.assertEqual(result["identity"]["returncode"], 0)
            self.assertNotIn("TP4_ENV", invoked.call_args.kwargs["env"])

    def test_final_restore_candidate_requires_current_soak_and_binds_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            value = self.make_campaign(directory, final_restore=True)
            self.assertEqual(value._final_restore_selection(False)["selected"],
                             "protected-default")
            value.state["cases"]["soak-mixed"] = {"runs": [{
                "status": "pass",
                "runtime_config_identity": dict(value.runtime_config_identity),
                "evidence": {"protocol_version": campaign.SOAK_PROTOCOL_VERSION},
            }]}
            self.assertEqual(value._final_restore_selection(True)["selected"],
                             "protected-default")
            value.telemetry.safety_event.set()
            self.assertEqual(value._final_restore_selection(False)["selected"],
                             "protected-default")
            value.telemetry.safety_event.clear()
            selection = value._final_restore_selection(False)
            self.assertEqual(selection["selected"], "configured-final")
            value.state["final_restore_selection"] = selection
            value.end_monotonic = time.monotonic() + 5
            calls = []
            def controller(overlay, command, **kwargs):
                calls.append((overlay, command, kwargs))
                return {"returncode": 0, "health_200_monotonic": time.monotonic()}
            value._controller = controller
            value._gates = lambda since: {"pass": True}
            completed = type("Completed", (), {"returncode": 0, "stdout": "ok"})()
            with patch.object(campaign.subprocess, "run", return_value=completed) as invoked:
                result = value.restore_default()
            self.assertEqual(calls[0][:2], (True, "down"))
            self.assertEqual(calls[1][0:2], (False, "up"))
            self.assertEqual(calls[1][2]["tp4_env"], "final-production.env")
            self.assertEqual(
                calls[1][2]["deadline_monotonic"],
                value.end_monotonic - campaign.FINAL_RESTORE_PROOF_RESERVE_SECONDS)
            args = invoked.call_args.args[0]
            self.assertEqual(args[-2:], ["--identity", "final-identity.json"])
            self.assertEqual(invoked.call_args.kwargs["env"]["TP4_ENV"],
                             "final-production.env")
            self.assertEqual(result["selection"]["selected"], "configured-final")
            self.assertTrue(value.final_native_rigmark_identity["restoration_verified"])
            self.assertEqual(value.final_native_rigmark_identity["identity_id"],
                             "final-unit")
            final_case = {"id": "rigmark-final", "area": "performance",
                          "manual": True, "target_repetitions": 1}
            value.record(final_case, 1, "recorded", {})
            recorded = value.state["cases"]["rigmark-final"]["runs"][-1]
            self.assertEqual(recorded["runtime_config_identity"]["identity_id"],
                             "final-unit")

    def test_final_restore_falls_back_on_local_or_remote_overlay_pin_change(self):
        for failure in ("local", "remote"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                value = self.make_campaign(directory, final_restore=True)
                value.state["cases"]["soak-mixed"] = {"runs": [{
                    "status": "pass",
                    "runtime_config_identity": dict(value.runtime_config_identity),
                    "evidence": {"protocol_version": campaign.SOAK_PROTOCOL_VERSION},
                }]}
                selection = value._final_restore_selection(False)
                calls = []
                if failure == "local":
                    (Path(value.cfg["repo_root"]) / "final-production.env").write_text(
                        "# changed after pin\n")
                    value._ssh = lambda *args, **kwargs: self.fail(
                        "remote verification ran after a local pin mismatch")
                else:
                    expected = value.final_restore_contract["restore_env_sha256"]
                    def remote(rank, *args, **kwargs):
                        calls.append((rank, args))
                        actual = "0" * 64 if rank == 2 else expected
                        return campaign.subprocess.CompletedProcess(
                            args, 0, f"{actual}  tp4/final-production.env\n", "")
                    value._ssh = remote
                checked = value._validate_selected_final_restore(selection)
                self.assertEqual(checked["selected"], "protected-default")
                self.assertIn("candidate_validation_error", checked)
                if failure == "remote":
                    self.assertEqual({rank for rank, _ in calls}, {0, 1, 2, 3})
                    self.assertEqual(
                        len(checked["remote_env_verification"]), 4)

    def test_final_restore_selection_exception_falls_back_to_default(self):
        with tempfile.TemporaryDirectory() as directory:
            value = self.make_campaign(directory, final_restore=True)
            with patch.object(value, "_final_restore_selection",
                              side_effect=RuntimeError("selection bug")):
                selection = value._safe_final_restore_selection(False)
            self.assertEqual(selection["selected"], "protected-default")
            self.assertIn("RuntimeError: selection bug", selection["selection_error"])

    def test_failed_final_identity_captures_evidence_and_stops_same_recipe(self):
        outcomes = (
            type("Completed", (), {"returncode": 1, "stdout": "identity mismatch"})(),
            campaign.subprocess.TimeoutExpired(
                [sys.executable, "scripts/check-f0.py"], 1, output="identity timeout"),
        )
        for outcome in outcomes:
            with self.subTest(outcome=type(outcome).__name__), \
                    tempfile.TemporaryDirectory() as directory:
                value = self.make_campaign(directory, final_restore=True)
                value.state["cases"]["soak-mixed"] = {"runs": [{
                    "status": "pass",
                    "runtime_config_identity": dict(value.runtime_config_identity),
                    "evidence": {"protocol_version": campaign.SOAK_PROTOCOL_VERSION},
                }]}
                value.state["final_restore_selection"] = (
                    value._final_restore_selection(False))
                value.end_monotonic = time.monotonic() + 5
                calls = []
                def controller(overlay, command, **kwargs):
                    calls.append((overlay, command, kwargs))
                    return {"returncode": 0,
                            "health_200_monotonic": time.monotonic()}
                value._controller = controller
                value._gates = lambda since: {"pass": True}
                captures = []
                value.preserve_cluster_evidence = lambda label: (
                    captures.append(label) or {"status": "captured"})
                if isinstance(outcome, BaseException):
                    invocation = patch.object(
                        campaign.subprocess, "run", side_effect=outcome)
                else:
                    invocation = patch.object(
                        campaign.subprocess, "run", return_value=outcome)
                with invocation:
                    result = value.restore_default()
                self.assertNotEqual(result["identity"]["returncode"], 0)
                self.assertEqual(captures,
                                 ["final-restoration-identity-failure"])
                self.assertEqual(calls[-1][0:2], (False, "down"))
                self.assertEqual(calls[-1][2]["tp4_env"],
                                 "final-production.env")
                self.assertEqual(calls[-1][2]["timeout"], 60)
                self.assertEqual(calls[-1][2]["deadline_monotonic"],
                                 value.end_monotonic)
                self.assertEqual(
                    result["shutdown_after_identity_failure"]["returncode"], 0)
                self.assertFalse(
                    value.final_native_rigmark_identity["restoration_verified"])

    def test_controller_deadline_override_never_extends_absolute_deadline(self):
        value = campaign.Campaign.__new__(campaign.Campaign)
        value.cfg = {"nodes": ["rank0", "rank1", "rank2", "rank3"],
                     "tp4_env": "overlay.env", "repo_root": "/unused"}
        value.active_deadline = time.monotonic() - 1
        with patch.object(campaign.subprocess, "Popen") as invoked:
            result = value._controller(
                False, "down", deadline_monotonic=time.monotonic() + 60)
        self.assertIsNone(result["returncode"])
        self.assertIn("deadline reached", result["reason"])
        invoked.assert_not_called()

    def test_drain_requires_a_fresh_node_local_scan_after_each_barrier(self):
        value = campaign.Campaign.__new__(campaign.Campaign)
        value.cfg = {"drain_timeout_seconds": 0.01}
        value.active_deadline = time.monotonic() + 2
        value.telemetry = FakeTelemetry()
        barriers = {rank: 100 + rank for rank in range(4)}

        def samples(scan_offset, reserved):
            return {rank: {"cache": {
                "scan_started_monotonic_ns": barriers[rank] + scan_offset,
                "reservation_status": "ok", "reserved_bytes": reserved if rank == 0 else 0,
                "scan_complete": True, "sample_fresh": True, "staging_bytes": 0,
                "anonymous_staging_complete": True, "accounting_status": "ok",
            }} for rank in range(4)}

        value.telemetry.latest = samples(0, 0)
        self.assertEqual(value.wait_for_drain(barriers)[0], "pending")
        value.telemetry.latest = samples(1, 4096)
        self.assertEqual(value.wait_for_drain(barriers)[0], "fail")
        value.telemetry.latest = samples(1, 0)
        self.assertEqual(value.wait_for_drain(barriers)[0], "pass")
        value.telemetry.latest = samples(1, 0)
        value.telemetry.latest[2]["cache"]["staging_bytes"] = 4096
        self.assertEqual(value.wait_for_drain(barriers)[0], "fail")

    def test_evidence_capture_streams_files_and_records_timeouts(self):
        with tempfile.TemporaryDirectory() as directory:
            helper = Path(directory) / "fake-ssh.py"
            helper.write_text(
                "import sys, time\n"
                "host, command = sys.argv[1:3]\n"
                "if host == 'rank1': time.sleep(5)\n"
                "print(host, command, 'evidence')\n")
            value = self.make_campaign(directory)
            value.end_monotonic = time.monotonic() + 0.25
            with patch.object(campaign, "SSH", (sys.executable, str(helper))):
                receipt = value.preserve_cluster_evidence("unit/fault")
            self.assertEqual(len(receipt["artifacts"]), 8)
            self.assertEqual(receipt["status"], "incomplete")
            self.assertTrue(any(item["timed_out"] for item in receipt["artifacts"]))
            for item in receipt["artifacts"]:
                path = Path(item["stdout"])
                self.assertTrue(path.is_file())
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)


class TelemetrySafetyTests(unittest.TestCase):
    def setUp(self):
        self.telemetry = campaign.Telemetry({}, Path("/unused"))
        self.now = time.monotonic()
        self.telemetry.started = self.now - 20

    def sample(self, *, memory=1 << 30, cache_bytes=None, scan_complete=False,
               host_known=True, cgroup_known=False, host_delta=0, cgroup_delta=None):
        return {
            "mem_available_bytes": memory, "container_state": None, "container_pid": None,
            "cache": {"bytes": cache_bytes, "scan_complete": scan_complete,
                      "sample_fresh": scan_complete},
            "oom": {"host_oom_kill_total": 15,
                    "host_oom_kills_since_start": host_delta,
                    "cgroup_oom_kills_since_start": cgroup_delta,
                    "earlyoom_kills": 0, "cuda_ooms_log_window": 0,
                    "cuda_oom_seen_since_start": False,
                    "coverage": {"host_oom_delta_known": host_known,
                                 "cgroup_oom_delta_known": cgroup_known}},
        }

    def test_expected_restart_skips_transient_cache_unknown_but_not_hard_limits(self):
        self.telemetry.arm_expected_down(60)
        reasons = self.telemetry._sample_reasons(0, self.sample(), self.now)
        self.assertEqual(reasons, [])
        low = self.telemetry._sample_reasons(
            0, self.sample(memory=(768 << 20) - 1), self.now)
        self.assertTrue(any("MemAvailable below" in reason for reason in low))
        large = self.telemetry._sample_reasons(
            0, self.sample(cache_bytes=(8 << 30) + 1), self.now)
        self.assertTrue(any("exceeds 8 GiB" in reason for reason in large))

    def test_oom_uses_since_start_counts_and_fails_closed_on_host_coverage(self):
        self.telemetry.arm_expected_down(60)
        old_total = self.telemetry._sample_reasons(0, self.sample(), self.now)
        self.assertFalse(any("OOM evidence" in reason for reason in old_total))
        positive = self.telemetry._sample_reasons(0, self.sample(host_delta=1), self.now)
        self.assertTrue(any("host_oom_kills_since_start" in reason for reason in positive))
        unknown = self.telemetry._sample_reasons(
            0, self.sample(host_known=False, host_delta=None), self.now)
        self.assertTrue(any("OOM coverage is unknown" in reason for reason in unknown))

    def test_expected_restart_window_expires(self):
        self.telemetry.arm_expected_down(0.01)
        reasons = self.telemetry._sample_reasons(
            0, self.sample(cache_bytes=0, scan_complete=True,
                           cgroup_known=True, cgroup_delta=0),
            self.telemetry.expected_down_until + 0.01)
        self.assertTrue(self.telemetry.safety_event.is_set())
        self.assertTrue(any("worker is not running" in reason for reason in reasons))


if __name__ == "__main__":
    unittest.main()
