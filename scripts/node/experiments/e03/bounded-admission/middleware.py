"""Bounded ASGI admission for the experimental TP4 resilience overlay."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from collections import deque
from collections.abc import Awaitable, Callable, Mapping, MutableMapping
from typing import Any


try:  # pragma: no cover - vLLM is present only in the serving image
    from vllm.logger import init_logger as _init_logger

    _LOG = _init_logger("vllm.tp4_admission")
except ImportError:
    _LOG = logging.getLogger("vllm.tp4_admission")
_BYPASSED_POST_PATHS = frozenset({
    "/ping",
    "/scale_elastic_ep",
    "/is_scaling_elastic_ep",
})

_ENV_DEFAULTS: Mapping[str, str] = {
    "TP4_ADMISSION_MAX_ACTIVE": "6",
    "TP4_ADMISSION_MAX_QUEUED": "128",
    "TP4_ADMISSION_MAX_BODY_BYTES": "8388608",
    "TP4_ADMISSION_QUEUE_TIMEOUT_SECONDS": "1800",
    "TP4_ADMISSION_REQUEST_TIMEOUT_SECONDS": "3600",
    "TP4_ADMISSION_BODY_IDLE_SECONDS": "30",
    "TP4_ADMISSION_SEND_IDLE_SECONDS": "30",
}

Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]
Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]
ASGIApp = Callable[[MutableMapping[str, Any], Receive, Send], Awaitable[None]]


class _BodyIdle(Exception):
    pass


class _BodyTooLarge(Exception):
    pass


class _SendIdle(Exception):
    pass


class _QueueTimeout(Exception):
    pass


class _RequestTimeout(Exception):
    pass


async def _bounded(
    awaitable: Awaitable[Any], seconds: float, timeout_error: type[Exception]
) -> Any:
    deadline = asyncio.timeout(seconds)
    try:
        async with deadline:
            return await awaitable
    except TimeoutError as exc:
        if not deadline.expired():
            raise
        raise timeout_error from exc


def _positive_int(name: str) -> int:
    raw = os.environ.get(name, _ENV_DEFAULTS[name])
    if not raw.isdecimal() or int(raw) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(raw)


def _positive_float(name: str) -> float:
    raw = os.environ.get(name, _ENV_DEFAULTS[name])
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive finite number") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return value


def _content_length(scope: Mapping[str, Any]) -> int | None:
    values = [
        value
        for name, value in scope.get("headers", [])
        if bytes(name).lower() == b"content-length"
    ]
    if not values:
        return None
    if len(values) != 1:
        raise ValueError("duplicate Content-Length")
    try:
        raw = bytes(values[0]).decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("invalid Content-Length") from exc
    if not raw.isdecimal():
        raise ValueError("invalid Content-Length")
    return int(raw)


def _is_response_cancel(path: str) -> bool:
    parts = path.split("/")
    return (
        len(parts) == 5
        and parts[:3] == ["", "v1", "responses"]
        and bool(parts[3])
        and parts[4] == "cancel"
    )


def _is_guarded(scope: Mapping[str, Any]) -> bool:
    if scope.get("type") != "http" or scope.get("method") != "POST":
        return False
    path = scope.get("path")
    return (
        isinstance(path, str)
        and path not in _BYPASSED_POST_PATHS
        and not _is_response_cancel(path)
    )


class BoundedAdmissionMiddleware:
    """Bound expensive API work before request-body parsing and tokenization."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self.max_active = _positive_int("TP4_ADMISSION_MAX_ACTIVE")
        self.max_queued = _positive_int("TP4_ADMISSION_MAX_QUEUED")
        self.max_body_bytes = _positive_int("TP4_ADMISSION_MAX_BODY_BYTES")
        self.queue_timeout = _positive_float(
            "TP4_ADMISSION_QUEUE_TIMEOUT_SECONDS"
        )
        self.request_timeout = _positive_float(
            "TP4_ADMISSION_REQUEST_TIMEOUT_SECONDS"
        )
        self.body_idle = _positive_float("TP4_ADMISSION_BODY_IDLE_SECONDS")
        self.send_idle = _positive_float("TP4_ADMISSION_SEND_IDLE_SECONDS")
        self._slots = asyncio.Semaphore(self.max_active)
        self._active = 0
        self._queued = 0
        self._inflight = 0
        self._admitted_total = 0
        self._counts = {
            "rejected_full": 0,
            "rejected_queue_timeout": 0,
            "request_timeout": 0,
            "body_too_large": 0,
            "body_idle": 0,
            "send_idle": 0,
            "cancelled_queued": 0,
            "cancelled_active": 0,
        }
        ready = {
            "schema": "tp4_admission_v1",
            "event": "ready",
            "max_active": self.max_active,
            "max_queued": self.max_queued,
            "max_body_bytes": self.max_body_bytes,
            "queue_timeout_seconds": self.queue_timeout,
            "request_timeout_seconds": self.request_timeout,
            "body_idle_seconds": self.body_idle,
            "send_idle_seconds": self.send_idle,
            "monotonic_ns": time.monotonic_ns(),
            "active": 0,
            "queued": 0,
            "inflight": 0,
            "admitted_total": 0,
            "rejected_total": 0,
            **self._counts,
        }
        _LOG.info("TP4_ADMISSION_READY %s", json.dumps(ready, sort_keys=True))

    def _state(self, event: str) -> None:
        record = {
            "schema": "tp4_admission_v1",
            "event": event,
            "monotonic_ns": time.monotonic_ns(),
            "active": self._active,
            "queued": self._queued,
            "inflight": self._inflight,
            "admitted_total": self._admitted_total,
            "rejected_total": (
                self._counts["rejected_full"]
                + self._counts["rejected_queue_timeout"]
            ),
            **self._counts,
        }
        _LOG.info("TP4_ADMISSION_STATE %s", json.dumps(record, sort_keys=True))

    async def _send_error(
        self,
        send: Send,
        status: int,
        message: str,
        error_type: str,
        code: str,
        *,
        retry_after: bool = False,
    ) -> None:
        body = json.dumps(
            {
                "error": {
                    "message": message,
                    "type": error_type,
                    "param": None,
                    "code": code,
                }
            },
            separators=(",", ":"),
        ).encode("utf-8")
        headers = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode("ascii")),
        ]
        if retry_after:
            headers.append(
                (b"retry-after", str(max(1, math.ceil(self.queue_timeout))).encode())
            )
        try:
            await _bounded(
                send({"type": "http.response.start", "status": status, "headers": headers}),
                self.send_idle,
                _SendIdle,
            )
            await _bounded(
                send({"type": "http.response.body", "body": body}),
                self.send_idle,
                _SendIdle,
            )
        except _SendIdle:
            self._counts["send_idle"] += 1
            self._state("send_idle")
            raise

    async def __call__(self, scope: MutableMapping[str, Any], receive: Receive, send: Send) -> None:
        if not _is_guarded(scope):
            await self.app(scope, receive, send)
            return

        try:
            content_length = _content_length(scope)
        except ValueError as exc:
            await self._send_error(
                send, 400, str(exc), "invalid_request_error", "invalid_content_length"
            )
            return
        if content_length is not None and content_length > self.max_body_bytes:
            self._counts["body_too_large"] += 1
            self._state("reject_body_header")
            await self._send_error(
                send,
                413,
                "Request body exceeds the configured limit.",
                "invalid_request_error",
                "request_body_too_large",
            )
            return

        if self._inflight >= self.max_active + self.max_queued:
            self._counts["rejected_full"] += 1
            self._state("reject_full")
            await self._send_error(
                send,
                503,
                "The admission queue is full.",
                "overloaded",
                "admission_queue_full",
                retry_after=True,
            )
            return

        self._inflight += 1
        self._admitted_total += 1
        self._queued += 1
        self._state("admit")
        acquired = False
        try:
            try:
                await _bounded(
                    self._slots.acquire(), self.queue_timeout, _QueueTimeout
                )
            except _QueueTimeout:
                self._queued -= 1
                self._inflight -= 1
                self._counts["rejected_queue_timeout"] += 1
                self._state("reject_queue_timeout")
                await self._send_error(
                    send,
                    503,
                    "Admission queue wait timed out.",
                    "overloaded",
                    "admission_queue_timeout",
                    retry_after=True,
                )
                return
            except asyncio.CancelledError:
                self._queued -= 1
                self._inflight -= 1
                self._counts["cancelled_queued"] += 1
                self._state("cancel_queued")
                raise

            acquired = True
            self._queued -= 1
            self._active += 1
            self._state("acquire")
            received = 0
            buffered: deque[MutableMapping[str, Any]] = deque()
            body = bytearray()
            response_started = False

            async def read_body() -> None:
                nonlocal received
                while True:
                    message = await _bounded(receive(), self.body_idle, _BodyIdle)
                    if message.get("type") != "http.request":
                        if body:
                            payload = bytes(body)
                            body.clear()
                            buffered.append({
                                "type": "http.request",
                                "body": payload,
                                "more_body": True,
                            })
                        buffered.append(message)
                        return
                    chunk = message.get("body", b"")
                    received += len(chunk)
                    if received > self.max_body_bytes:
                        raise _BodyTooLarge
                    body.extend(chunk)
                    if not message.get("more_body", False):
                        replay = dict(message)
                        replay["body"] = bytes(body)
                        body.clear()
                        replay["more_body"] = False
                        buffered.append(replay)
                        return

            async def replay_receive() -> MutableMapping[str, Any]:
                if buffered:
                    return buffered.popleft()
                return await receive()

            async def limited_send(message: MutableMapping[str, Any]) -> None:
                nonlocal response_started
                if message.get("type") == "http.response.start":
                    response_started = True
                try:
                    await _bounded(send(message), self.send_idle, _SendIdle)
                except _SendIdle:
                    raise

            try:
                async def run_active() -> None:
                    await read_body()
                    await self.app(scope, replay_receive, limited_send)

                await _bounded(
                    run_active(),
                    self.request_timeout,
                    _RequestTimeout,
                )
            except asyncio.CancelledError:
                self._counts["cancelled_active"] += 1
                self._state("cancel_active")
                raise
            except _BodyTooLarge:
                self._counts["body_too_large"] += 1
                self._state("reject_body_stream")
                if response_started:
                    raise
                await self._send_error(
                    send,
                    413,
                    "Request body exceeds the configured limit.",
                    "invalid_request_error",
                    "request_body_too_large",
                )
            except _BodyIdle:
                self._counts["body_idle"] += 1
                self._state("body_idle")
                if response_started:
                    raise
                await self._send_error(
                    send,
                    408,
                    "Request body read timed out.",
                    "invalid_request_error",
                    "request_body_timeout",
                )
            except _SendIdle:
                self._counts["send_idle"] += 1
                self._state("send_idle")
                raise
            except _RequestTimeout:
                self._counts["request_timeout"] += 1
                self._state("request_timeout")
                if response_started:
                    raise
                await self._send_error(
                    send,
                    503,
                    "Request processing timed out.",
                    "server_error",
                    "admission_request_timeout",
                    retry_after=True,
                )
        finally:
            if acquired:
                self._active -= 1
                self._inflight -= 1
                self._slots.release()
                self._state("release")
