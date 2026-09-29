#!/usr/bin/env python3
"""Offline tests for the bounded ASGI admission candidate."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
import unittest
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "scripts/node/experiments/e03/bounded-admission"
MODULE_PATH = CANDIDATE / "middleware.py"
SPEC = importlib.util.spec_from_file_location("tp4_admission_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

TEST_ENV = {
    "TP4_ADMISSION_MAX_ACTIVE": "1",
    "TP4_ADMISSION_MAX_QUEUED": "2",
    "TP4_ADMISSION_MAX_BODY_BYTES": "8",
    "TP4_ADMISSION_QUEUE_TIMEOUT_SECONDS": "0.2",
    "TP4_ADMISSION_REQUEST_TIMEOUT_SECONDS": "0.2",
    "TP4_ADMISSION_BODY_IDLE_SECONDS": "0.05",
    "TP4_ADMISSION_SEND_IDLE_SECONDS": "0.05",
}


def scope(
    path: str = "/v1/chat/completions",
    *,
    method: str = "POST",
    headers: list[tuple[bytes, bytes]] | None = None,
    request_id: str = "request",
) -> dict[str, Any]:
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "method": method,
        "path": path,
        "headers": headers or [],
        "test_request_id": request_id,
    }


class Receive:
    def __init__(self, messages: list[dict[str, Any]] | None = None) -> None:
        self.messages = list(messages or [
            {"type": "http.request", "body": b"", "more_body": False}
        ])
        self.calls = 0

    async def __call__(self) -> dict[str, Any]:
        self.calls += 1
        if self.messages:
            return self.messages.pop(0)
        return {"type": "http.disconnect"}


class Send:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    async def __call__(self, message: dict[str, Any]) -> None:
        self.messages.append(message)

    @property
    def status(self) -> int | None:
        starts = [m for m in self.messages if m["type"] == "http.response.start"]
        return starts[0]["status"] if starts else None

    @property
    def headers(self) -> dict[bytes, bytes]:
        starts = [m for m in self.messages if m["type"] == "http.response.start"]
        return dict(starts[0]["headers"]) if starts else {}

    @property
    def json(self) -> dict[str, Any]:
        body = b"".join(
            m.get("body", b"")
            for m in self.messages
            if m["type"] == "http.response.body"
        )
        return json.loads(body)


async def respond(send: Callable[[dict[str, Any]], Awaitable[None]]) -> None:
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"ok"})


async def wait_until(predicate: Callable[[], bool], timeout: float = 0.5) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0)


class BoundedAdmissionTests(unittest.IsolatedAsyncioTestCase):
    def make(self, app: Any, **overrides: str) -> Any:
        env = {**TEST_ENV, **overrides}
        with patch.dict(os.environ, env, clear=False):
            return MODULE.BoundedAdmissionMiddleware(app)

    async def test_fifo_active_and_queue_full_without_reading_body(self) -> None:
        started: list[str] = []
        gates = {name: asyncio.Event() for name in ("a", "b")}

        async def app(request_scope: dict[str, Any], receive: Any, send: Any) -> None:
            name = request_scope["test_request_id"]
            started.append(name)
            await gates[name].wait()
            await respond(send)

        middleware = self.make(
            app,
            TP4_ADMISSION_MAX_ACTIVE="1",
            TP4_ADMISSION_MAX_QUEUED="1",
        )
        first_send, second_send, rejected_send = Send(), Send(), Send()
        first = asyncio.create_task(middleware(scope(request_id="a"), Receive(), first_send))
        await wait_until(lambda: started == ["a"])
        second = asyncio.create_task(
            middleware(scope(request_id="b"), Receive(), second_send)
        )
        await wait_until(lambda: middleware._queued == 1)
        rejected_receive = Receive()
        await middleware(scope(request_id="c"), rejected_receive, rejected_send)

        self.assertEqual(rejected_send.status, 503)
        self.assertEqual(rejected_send.json["error"]["type"], "overloaded")
        self.assertEqual(
            rejected_send.json["error"]["code"], "admission_queue_full"
        )
        self.assertIn(b"retry-after", rejected_send.headers)
        self.assertEqual(rejected_receive.calls, 0)

        gates["a"].set()
        await wait_until(lambda: started == ["a", "b"])
        gates["b"].set()
        await asyncio.gather(first, second)
        self.assertEqual(middleware._active, 0)
        self.assertEqual(middleware._queued, 0)
        self.assertEqual(middleware._inflight, 0)

    async def test_three_waiters_start_in_fifo_order(self) -> None:
        started: list[str] = []
        gates = {name: asyncio.Event() for name in ("a", "b", "c")}

        async def app(request_scope: dict[str, Any], receive: Any, send: Any) -> None:
            name = request_scope["test_request_id"]
            started.append(name)
            await gates[name].wait()
            await respond(send)

        middleware = self.make(app)
        tasks = []
        for name in ("a", "b", "c"):
            tasks.append(
                asyncio.create_task(middleware(scope(request_id=name), Receive(), Send()))
            )
            await wait_until(lambda: middleware._inflight == len(tasks))
        self.assertEqual(started, ["a"])
        gates["a"].set()
        await wait_until(lambda: started == ["a", "b"])
        gates["b"].set()
        await wait_until(lambda: started == ["a", "b", "c"])
        gates["c"].set()
        await asyncio.gather(*tasks)

    async def test_inference_alias_is_guarded_while_control_posts_bypass(self) -> None:
        gate = asyncio.Event()
        started = asyncio.Event()

        async def app(request_scope: dict[str, Any], receive: Any, send: Any) -> None:
            if request_scope.get("test_request_id") == "active":
                started.set()
                await gate.wait()
            await respond(send)

        middleware = self.make(
            app,
            TP4_ADMISSION_MAX_ACTIVE="1",
            TP4_ADMISSION_MAX_QUEUED="1",
        )
        active = asyncio.create_task(
            middleware(scope(request_id="active"), Receive(), Send())
        )
        await started.wait()
        queued = asyncio.create_task(
            middleware(scope("/tokenize", request_id="queued"), Receive(), Send())
        )
        await wait_until(lambda: middleware._queued == 1)

        for path in ("/v1/responses", "/invocations", "/unknown-post"):
            result = Send()
            await middleware(scope(path), Receive(), result)
            self.assertEqual(result.status, 503, path)
            self.assertEqual(result.json["error"]["code"], "admission_queue_full")

        for path in (
            "/ping",
            "/scale_elastic_ep",
            "/is_scaling_elastic_ep",
            "/v1/responses/response-1/cancel",
        ):
            result = Send()
            await middleware(scope(path), Receive(), result)
            self.assertEqual(result.status, 200, path)

        near_match = Send()
        await middleware(
            scope("/v1/responses/response-1/cancel/extra"), Receive(), near_match
        )
        self.assertEqual(near_match.status, 503)

        gate.set()
        await asyncio.gather(active, queued)

    async def test_queued_and_active_cancellation_release_slots(self) -> None:
        gates = {"active": asyncio.Event(), "replacement": asyncio.Event()}
        started: list[str] = []

        async def app(request_scope: dict[str, Any], receive: Any, send: Any) -> None:
            name = request_scope["test_request_id"]
            started.append(name)
            await gates[name].wait()
            await respond(send)

        middleware = self.make(app)
        active = asyncio.create_task(
            middleware(scope(request_id="active"), Receive(), Send())
        )
        await wait_until(lambda: started == ["active"])
        queued = asyncio.create_task(
            middleware(scope(request_id="queued"), Receive(), Send())
        )
        await wait_until(lambda: middleware._queued == 1)
        queued.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await queued
        self.assertEqual((middleware._active, middleware._queued), (1, 0))

        replacement = asyncio.create_task(
            middleware(scope(request_id="replacement"), Receive(), Send())
        )
        await wait_until(lambda: middleware._queued == 1)
        active.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await active
        await wait_until(lambda: started == ["active", "replacement"])
        self.assertEqual(middleware._counts["cancelled_queued"], 1)
        self.assertEqual(middleware._counts["cancelled_active"], 1)
        gates["replacement"].set()
        await replacement
        self.assertEqual((middleware._active, middleware._queued, middleware._inflight), (0, 0, 0))

    async def test_queue_timeout_is_explicit_overload_and_releases(self) -> None:
        gate = asyncio.Event()

        async def app(request_scope: dict[str, Any], receive: Any, send: Any) -> None:
            await gate.wait()

        middleware = self.make(
            app,
            TP4_ADMISSION_QUEUE_TIMEOUT_SECONDS="0.01",
        )
        active = asyncio.create_task(middleware(scope(), Receive(), Send()))
        await wait_until(lambda: middleware._active == 1)
        queued_send = Send()
        await middleware(scope(request_id="queued"), Receive(), queued_send)
        self.assertEqual(queued_send.status, 503)
        self.assertEqual(
            queued_send.json["error"]["code"], "admission_queue_timeout"
        )
        self.assertEqual((middleware._active, middleware._queued, middleware._inflight), (1, 0, 1))
        active.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await active

    async def test_content_length_and_streamed_body_limits(self) -> None:
        async def consume(request_scope: dict[str, Any], receive: Any, send: Any) -> None:
            while True:
                message = await receive()
                if message["type"] != "http.request" or not message.get("more_body"):
                    break
            await respond(send)

        middleware = self.make(
            consume,
            TP4_ADMISSION_MAX_BODY_BYTES="4",
        )
        header_receive, header_send = Receive(), Send()
        await middleware(
            scope(headers=[(b"content-length", b"5")]),
            header_receive,
            header_send,
        )
        self.assertEqual(header_send.status, 413)
        self.assertEqual(header_receive.calls, 0)

        exact_send = Send()
        await middleware(
            scope(headers=[(b"content-length", b"4")]),
            Receive([
                {"type": "http.request", "body": b"ab", "more_body": True},
                {"type": "http.request", "body": b"cd", "more_body": False},
            ]),
            exact_send,
        )
        self.assertEqual(exact_send.status, 200)

        streamed_send = Send()
        await middleware(
            scope(),
            Receive([
                {"type": "http.request", "body": b"abc", "more_body": True},
                {"type": "http.request", "body": b"de", "more_body": False},
            ]),
            streamed_send,
        )
        self.assertEqual(streamed_send.status, 413)
        self.assertEqual(
            streamed_send.json["error"]["code"], "request_body_too_large"
        )
        self.assertEqual(middleware._active, 0)

    async def test_body_failures_precede_downstream_broad_exception_handler(self) -> None:
        calls = 0

        async def broad_catching_app(
            request_scope: dict[str, Any], receive: Any, send: Any
        ) -> None:
            nonlocal calls
            calls += 1
            try:
                while (await receive()).get("more_body", False):
                    pass
            except Exception:
                await send({"type": "http.response.start", "status": 400, "headers": []})
                await send({"type": "http.response.body", "body": b"caught"})

        middleware = self.make(
            broad_catching_app,
            TP4_ADMISSION_MAX_BODY_BYTES="4",
            TP4_ADMISSION_BODY_IDLE_SECONDS="0.01",
        )
        too_large = Send()
        await middleware(
            scope(),
            Receive([{"type": "http.request", "body": b"12345", "more_body": False}]),
            too_large,
        )
        self.assertEqual(too_large.status, 413)
        self.assertEqual(calls, 0)

        partial_sent = False
        never = asyncio.Event()

        async def idle_receive() -> dict[str, Any]:
            nonlocal partial_sent
            if not partial_sent:
                partial_sent = True
                return {"type": "http.request", "body": b"1", "more_body": True}
            await never.wait()
            raise AssertionError("unreachable")

        idle = Send()
        await middleware(scope(), idle_receive, idle)
        self.assertEqual(idle.status, 408)
        self.assertEqual(calls, 0)

    async def test_invalid_content_length_is_rejected_before_admission(self) -> None:
        called = False

        async def app(request_scope: dict[str, Any], receive: Any, send: Any) -> None:
            nonlocal called
            called = True

        middleware = self.make(app)
        for headers in (
            [(b"content-length", b"-1")],
            [(b"content-length", b"1"), (b"Content-Length", b"1")],
        ):
            result = Send()
            await middleware(scope(headers=headers), Receive(), result)
            self.assertEqual(result.status, 400)
            self.assertEqual(result.json["error"]["code"], "invalid_content_length")
        self.assertFalse(called)
        self.assertEqual(middleware._inflight, 0)

    async def test_body_idle_and_preheader_request_timeout_are_explicit(self) -> None:
        never = asyncio.Event()

        partial_sent = False

        async def partial_receive() -> dict[str, Any]:
            nonlocal partial_sent
            if not partial_sent:
                partial_sent = True
                return {"type": "http.request", "body": b"a", "more_body": True}
            await never.wait()
            raise AssertionError("unreachable")

        async def read_app(request_scope: dict[str, Any], receive: Any, send: Any) -> None:
            while (await receive()).get("more_body", False):
                pass

        body_middleware = self.make(
            read_app,
            TP4_ADMISSION_BODY_IDLE_SECONDS="0.01",
        )
        body_send = Send()
        await body_middleware(scope(), partial_receive, body_send)
        self.assertEqual(body_send.status, 408)
        self.assertEqual(body_send.json["error"]["code"], "request_body_timeout")

        async def blocked_app(request_scope: dict[str, Any], receive: Any, send: Any) -> None:
            await never.wait()

        wall_middleware = self.make(
            blocked_app,
            TP4_ADMISSION_REQUEST_TIMEOUT_SECONDS="0.01",
        )
        wall_send = Send()
        await wall_middleware(scope(), Receive(), wall_send)
        self.assertEqual(wall_send.status, 503)
        self.assertEqual(
            wall_send.json["error"]["code"], "admission_request_timeout"
        )
        self.assertEqual(wall_middleware._active, 0)

    async def test_body_idle_stops_after_complete_body_and_between_sends(self) -> None:
        never = asyncio.Event()

        class CompleteThenBlock:
            def __init__(self) -> None:
                self.complete = False

            async def __call__(self) -> dict[str, Any]:
                if not self.complete:
                    self.complete = True
                    return {"type": "http.request", "body": b"{}", "more_body": False}
                await never.wait()
                raise AssertionError("unreachable")

        async def streaming_app(
            request_scope: dict[str, Any], receive: Any, send: Any
        ) -> None:
            await receive()
            disconnect_listener = asyncio.create_task(receive())
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await asyncio.sleep(0.03)
            await send({"type": "http.response.body", "body": b"ok"})
            disconnect_listener.cancel()
            try:
                await disconnect_listener
            except asyncio.CancelledError:
                pass

        middleware = self.make(
            streaming_app,
            TP4_ADMISSION_BODY_IDLE_SECONDS="0.01",
            TP4_ADMISSION_SEND_IDLE_SECONDS="0.01",
            TP4_ADMISSION_REQUEST_TIMEOUT_SECONDS="0.2",
        )
        result = Send()
        await middleware(scope(), CompleteThenBlock(), result)
        self.assertEqual(result.status, 200)
        self.assertEqual(middleware._counts["body_idle"], 0)
        self.assertEqual(middleware._counts["send_idle"], 0)

    async def test_request_wall_timeout_covers_trickling_body(self) -> None:
        app_calls = 0

        async def app(request_scope: dict[str, Any], receive: Any, send: Any) -> None:
            nonlocal app_calls
            app_calls += 1

        async def trickle() -> dict[str, Any]:
            await asyncio.sleep(0.005)
            return {"type": "http.request", "body": b"x", "more_body": True}

        middleware = self.make(
            app,
            TP4_ADMISSION_MAX_BODY_BYTES="1024",
            TP4_ADMISSION_BODY_IDLE_SECONDS="0.02",
            TP4_ADMISSION_REQUEST_TIMEOUT_SECONDS="0.025",
        )
        result = Send()
        await middleware(scope(), trickle, result)
        self.assertEqual(result.status, 503)
        self.assertEqual(
            result.json["error"]["code"], "admission_request_timeout"
        )
        self.assertEqual(app_calls, 0)
        self.assertEqual(middleware._counts["body_idle"], 0)
        self.assertEqual(middleware._counts["request_timeout"], 1)
        self.assertEqual((middleware._active, middleware._inflight), (0, 0))

    async def test_slow_writer_after_headers_aborts_and_releases(self) -> None:
        never = asyncio.Event()

        async def app(request_scope: dict[str, Any], receive: Any, send: Any) -> None:
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"stream"})

        messages: list[dict[str, Any]] = []

        async def slow_send(message: dict[str, Any]) -> None:
            messages.append(message)
            if message["type"] == "http.response.body":
                await never.wait()

        middleware = self.make(app, TP4_ADMISSION_SEND_IDLE_SECONDS="0.01")
        with self.assertRaises(MODULE._SendIdle):
            await middleware(scope(), Receive(), slow_send)
        self.assertEqual([message["type"] for message in messages], [
            "http.response.start",
            "http.response.body",
        ])
        self.assertEqual((middleware._active, middleware._inflight), (0, 0))

    async def test_stalled_response_start_never_gets_second_start(self) -> None:
        never = asyncio.Event()

        async def app(request_scope: dict[str, Any], receive: Any, send: Any) -> None:
            await send({"type": "http.response.start", "status": 200, "headers": []})

        messages: list[dict[str, Any]] = []

        async def stalled_start(message: dict[str, Any]) -> None:
            messages.append(message)
            await never.wait()

        middleware = self.make(
            app,
            TP4_ADMISSION_REQUEST_TIMEOUT_SECONDS="0.01",
            TP4_ADMISSION_SEND_IDLE_SECONDS="0.2",
        )
        with self.assertRaises(MODULE._RequestTimeout):
            await middleware(scope(), Receive(), stalled_start)
        self.assertEqual(
            [message["type"] for message in messages], ["http.response.start"]
        )
        self.assertEqual((middleware._active, middleware._inflight), (0, 0))

    async def test_downstream_exception_and_bypass_do_not_leak_slots(self) -> None:
        gate = asyncio.Event()
        guarded_started = asyncio.Event()

        async def app(request_scope: dict[str, Any], receive: Any, send: Any) -> None:
            if request_scope["path"] == "/health":
                await respond(send)
                return
            if request_scope.get("test_request_id") == "raise":
                raise RuntimeError("downstream")
            guarded_started.set()
            await gate.wait()

        middleware = self.make(app)
        active = asyncio.create_task(middleware(scope(), Receive(), Send()))
        await guarded_started.wait()
        health_send = Send()
        await middleware(scope("/health", method="GET"), Receive(), health_send)
        self.assertEqual(health_send.status, 200)
        self.assertEqual((middleware._active, middleware._inflight), (1, 1))
        active.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await active

        with self.assertRaisesRegex(RuntimeError, "downstream"):
            await middleware(scope(request_id="raise"), Receive(), Send())
        self.assertEqual((middleware._active, middleware._inflight), (0, 0))

        async def timeout_app(request_scope: dict[str, Any], receive: Any, send: Any) -> None:
            raise TimeoutError("downstream timeout")

        timeout_middleware = self.make(timeout_app)
        with self.assertRaisesRegex(TimeoutError, "downstream timeout"):
            await timeout_middleware(scope(), Receive(), Send())
        self.assertEqual((timeout_middleware._active, timeout_middleware._inflight), (0, 0))

    async def test_ready_and_state_logs_are_stable_json_without_request_data(self) -> None:
        async def app(request_scope: dict[str, Any], receive: Any, send: Any) -> None:
            await respond(send)

        with self.assertLogs("vllm.tp4_admission", level="INFO") as logs:
            middleware = self.make(app)
            await middleware(scope(request_id="secret-id"), Receive(), Send())
        self.assertEqual(MODULE._LOG.name, "vllm.tp4_admission")
        ready = [line for line in logs.output if "TP4_ADMISSION_READY " in line]
        states = [line for line in logs.output if "TP4_ADMISSION_STATE " in line]
        self.assertEqual(len(ready), 1)
        ready_record = json.loads(ready[0].split("TP4_ADMISSION_READY ", 1)[1])
        self.assertEqual(ready_record["event"], "ready")
        self.assertEqual(ready_record["max_active"], 1)
        self.assertEqual(
            (ready_record["active"], ready_record["queued"], ready_record["inflight"]),
            (0, 0, 0),
        )
        self.assertEqual(ready_record["admitted_total"], 0)
        self.assertEqual(ready_record["rejected_total"], 0)
        self.assertEqual(
            [json.loads(line.split("TP4_ADMISSION_STATE ", 1)[1])["event"] for line in states],
            ["admit", "acquire", "release"],
        )
        for line in states:
            record = json.loads(line.split("TP4_ADMISSION_STATE ", 1)[1])
            self.assertEqual(record["inflight"], record["active"] + record["queued"])
            self.assertEqual(
                record["rejected_total"],
                record["rejected_full"] + record["rejected_queue_timeout"],
            )
        self.assertTrue(all("secret-id" not in line for line in logs.output))

    async def test_invalid_environment_fails_closed(self) -> None:
        async def app(request_scope: dict[str, Any], receive: Any, send: Any) -> None:
            return None

        invalid = {
            "TP4_ADMISSION_MAX_ACTIVE": "0",
            "TP4_ADMISSION_MAX_QUEUED": "True",
            "TP4_ADMISSION_MAX_BODY_BYTES": "-1",
            "TP4_ADMISSION_QUEUE_TIMEOUT_SECONDS": "nan",
            "TP4_ADMISSION_REQUEST_TIMEOUT_SECONDS": "inf",
            "TP4_ADMISSION_BODY_IDLE_SECONDS": "0",
            "TP4_ADMISSION_SEND_IDLE_SECONDS": "-0.1",
        }
        for name, value in invalid.items():
            with self.subTest(name=name), patch.dict(
                os.environ, {**TEST_ENV, name: value}, clear=False
            ):
                with self.assertRaisesRegex(ValueError, name):
                    MODULE.BoundedAdmissionMiddleware(app)


class BoundedAdmissionArtifactTests(unittest.TestCase):
    def test_manifest_and_deployed_checksums(self) -> None:
        manifest = json.loads((CANDIDATE / "manifest.json").read_text())
        self.assertEqual(manifest["schema"], "tp4-bounded-admission-candidate-v1")
        self.assertEqual(
            (manifest["runtime"]["fastapi"], manifest["runtime"]["starlette"]),
            ("0.136.3", "1.6.0"),
        )
        self.assertEqual(
            manifest["candidate"]["mount_target"], "/opt/tp4/tp4_admission.py"
        )
        self.assertEqual(
            manifest["candidate"]["import_string"],
            "tp4_admission.BoundedAdmissionMiddleware",
        )
        self.assertEqual(manifest["guard"]["mode"], "all_post_except")
        self.assertEqual(
            manifest["guard"]["bypass_paths"], sorted(MODULE._BYPASSED_POST_PATHS)
        )
        self.assertEqual(
            manifest["guard"]["bypass_pattern"],
            "/v1/responses/{nonempty-single-segment}/cancel",
        )
        self.assertEqual(manifest["selection"]["environment"], {
            name: value for name, value in MODULE._ENV_DEFAULTS.items()
        })
        cluster_template = (ROOT / "cluster.env.example").read_text()
        self.assertIn(f'IMAGE={manifest["runtime"]["image"]}', cluster_template)
        self.assertIn(f'IMAGE_ID={manifest["runtime"]["image_id"]}', cluster_template)
        candidate_sha = hashlib.sha256(MODULE_PATH.read_bytes()).hexdigest()
        self.assertEqual(manifest["candidate"]["sha256"], candidate_sha)

        checksum_lines = (CANDIDATE / "SHA256SUMS").read_text().splitlines()
        self.assertEqual(
            {line.split("  ", 1)[1] for line in checksum_lines},
            {"manifest.json", "middleware.py"},
        )
        for line in checksum_lines:
            expected, filename = line.split("  ", 1)
            self.assertEqual(
                hashlib.sha256((CANDIDATE / filename).read_bytes()).hexdigest(),
                expected,
            )


if __name__ == "__main__":
    unittest.main()
