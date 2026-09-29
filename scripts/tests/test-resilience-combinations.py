#!/usr/bin/env python3
"""Offline tests for reviewed compound resilience cases."""

from __future__ import annotations

import copy
from pathlib import Path
import sys
import threading
import time
import unittest


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/resilience"))
import campaign as campaign_driver  # noqa: E402
import combination_cases  # noqa: E402


DIGEST = "d" * 64


class FakeThread:
    def __init__(self, alive: bool = True, terminate_on_join: bool = False,
                 join_error=None):
        self.alive = alive
        self.terminate_on_join = terminate_on_join
        self.join_error = join_error

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        if self.join_error is not None:
            raise self.join_error
        if self.terminate_on_join:
            self.alive = False


class FakeRequest:
    def __init__(self, payload, request_id, *, completed=False, done_before_eof=False,
                 stuck=False, missing=False, partial=False, join_error=None):
        self.payload = copy.deepcopy(payload)
        self.request_id = request_id
        self.thread = FakeThread(alive=not completed, join_error=join_error)
        if completed or done_before_eof:
            self.result = {
                "read_complete": not done_before_eof, "stream_complete": True,
                "done": True, "finish_reasons": ["stop"],
                "error_event": False, "parse_errors": [],
                "body_text": "data: [DONE]\n\n",
            }
        elif partial:
            self.result = {
                "read_complete": False, "stream_complete": False,
                "done": False, "finish_reasons": [],
                "error_event": False, "parse_errors": [],
                "body_text": "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}",
            }
        else:
            self.result = None
        self.error = None
        self.http_status = 200 if completed or done_before_eof else None
        self.response_headers = {"content-type": "text/event-stream"}
        self.started_ns = time.time_ns()
        self.first_byte_ns = self.started_ns if completed or done_before_eof else None
        self.finished_ns = self.started_ns if completed else None
        self.first_byte_hex = "64" if completed or done_before_eof else None
        self.cancelled = threading.Event()
        self.stuck = stuck
        self.missing = missing

    def start(self):
        pass

    def cancel(self):
        self.cancelled.set()
        if not self.thread.alive:
            return
        if self.stuck:
            return
        self.thread.alive = False
        self.finished_ns = time.time_ns()
        if not self.missing:
            self.error = "ConnectionAbortedError: cancelled"


class FakeHTTP:
    def __init__(self):
        self.chat_calls = []
        self.interrupt_on_chat_call = None

    def exact_prompt(self, target, prefix):
        assert target == 32_000
        return "exact-32k-prompt"

    def chat_payload(self, prompt, **kwargs):
        return {"messages": [{"role": "user", "content": prompt}], **kwargs}

    def chat(self, prompt, request_id):
        self.chat_calls.append((prompt, request_id))
        if len(self.chat_calls) == self.interrupt_on_chat_call:
            raise KeyboardInterrupt("reissue stop")
        return {"status": 200, "body": {"choices": [{}]}}


class FakeTelemetry:
    def __init__(self):
        self.safety_event = threading.Event()


class FakeCampaign:
    def __init__(self):
        self.active_deadline = time.monotonic() + 300
        self.telemetry = FakeTelemetry()
        self.http = FakeHTTP()
        self.faultctl_calls = []
        self.cursor_calls = 0
        self.event_calls = []
        self.reset_calls = []
        self.cleanup_errors = []
        self.event_variant = "success"
        self.health = 200
        self.idle_status = "pass"
        self.drain_status = "pass"
        self.driver_events = []
        self.driver_event_error = None
        self.cleanup_base_error = None
        self.set_errors = {"arm": {}, "release": {}}

    @staticmethod
    def _valid(results):
        return all(row.get("status") == 200 and row.get("body", {}).get("choices")
                   for row in results)

    @staticmethod
    def _publication_digest(events):
        valid = [event for event in events
                 if event.get("outcome") == "ok" and event.get("committed") is True]
        digests = {event.get("digest") for event in valid}
        return DIGEST if len(valid) == 4 and digests == {DIGEST} else None

    def _set_case(self, case_id):
        self.case_id = case_id

    def _cursor(self, rank):
        generation = self.cursor_calls // 4
        self.cursor_calls += 1
        return f"cursor-{rank}-{generation}"

    def _events_all(self, cursors, case_id, kind, *, stage=None, mode=None, timeout=30):
        self.event_calls.append((kind, stage, list(cursors)))
        if kind == "stage_end" and stage == "publication":
            return ([{"outcome": "ok", "committed": True, "digest": DIGEST}
                     for _ in range(4)], [])
        if kind == "stage_waiting":
            events = [{"digest": DIGEST, "operation_id": f"operation-{rank}",
                       "time_ns": 100 + rank}
                      for rank in range(4)]
            if self.event_variant == "missing-wait":
                return events[:3], ["rank 3: timeout"]
            if self.event_variant == "partial-middle-wait":
                return [events[0], events[2], events[3]], ["rank 1: timeout"]
            return events, []
        if kind == "fault_hit":
            events = [{"mode": "eio", "operation": "read", "time_ns": 200 + rank}
                      for rank in range(4)]
            if self.event_variant == "pre-cancel-only":
                return [], [f"rank {rank}: timeout" for rank in range(4)]
            if self.event_variant == "missing-hit":
                return events[:3], ["rank 3: timeout"]
            if self.event_variant == "mismatched-boundary":
                events[1]["time_ns"] = 1
            return events, []
        if kind == "stage_end" and stage == "restore":
            events = [{"outcome": "error", "digest": DIGEST,
                       "operation_id": f"operation-{rank}", "time_ns": 300 + rank}
                      for rank in range(4)]
            if self.event_variant == "pre-cancel-only":
                return [], [f"rank {rank}: timeout" for rank in range(4)]
            if self.event_variant == "mismatched-operation":
                events[2]["operation_id"] = "wrong-operation"
            if self.event_variant == "instrumentation-error":
                events[2]["instrumentation_error"] = "TimeoutError: pause timed out"
            return events, []
        raise AssertionError((kind, stage, mode))

    def _faultctl(self, rank, arguments):
        self.faultctl_calls.append((rank, list(arguments)))
        phase = "arm" if "--pause-stage" in arguments else "release"
        error = self.set_errors[phase].get(rank)
        if error is not None:
            raise error
        return {"rank": rank, "generation": len(self.faultctl_calls)}

    def _reset_faults(self, case_id):
        self.reset_calls.append(case_id)
        if self.cleanup_base_error is not None:
            raise self.cleanup_base_error
        return {"case_id": case_id, "errors": list(self.cleanup_errors)}

    def health_code(self):
        return self.health

    def wait_for_api_idle(self):
        return self.idle_status, {"requests_running": 0, "requests_waiting": 0}

    def _drain_barriers(self):
        return {rank: 100 + rank for rank in range(4)}

    def wait_for_drain(self, barriers):
        return self.drain_status, {"ranks": {rank: {"reserved_bytes": 0,
            "staging_bytes": 0} for rank in range(4)}}

    def append_driver_event(self, event):
        if self.driver_event_error is not None:
            raise self.driver_event_error
        self.driver_events.append(copy.deepcopy(event))


CASE = {"id": "combination-cache-eio-cancel-restore"}


class CombinationTests(unittest.TestCase):
    def run_case(self, campaign=None, request_options=None):
        value = campaign or FakeCampaign()
        created = []
        def factory(payload, request_id):
            request = FakeRequest(payload, request_id, **(request_options or {}))
            created.append(request)
            return request
        status, evidence = combination_cases.run_cache_eio_cancel_restore(
            value, CASE, 2, factory)
        return value, created, status, evidence

    def test_success_proves_seed_rank_operations_fault_cancel_and_recovery(self):
        value, requests, status, evidence = self.run_case()
        self.assertEqual(status, "pass")
        self.assertEqual(evidence["seed_digest"], DIGEST)
        self.assertTrue(evidence["cancelled_request"]["interrupted"])
        self.assertEqual(len(evidence["fault_hits"]), 4)
        self.assertEqual(len(evidence["restore_ends"]), 4)
        self.assertEqual(evidence["request"]["payload"], requests[0].payload)
        self.assertEqual(value.reset_calls, [CASE["id"]])
        set_calls = [arguments for _, arguments in value.faultctl_calls]
        self.assertEqual(len(set_calls), 8)
        self.assertTrue(all("--pause-stage" in row for row in set_calls[:4]))
        self.assertTrue(all("--pause-stage" not in row for row in set_calls[4:]))
        self.assertTrue(all(row[row.index("--remaining") + 1] == "1" for row in set_calls))
        self.assertEqual(evidence["reservation_and_staging_drain"]["status"], "pass")
        self.assertEqual(len(value.http.chat_calls), 2)
        waiting_call = next(row for row in value.event_calls
                            if row[:2] == ("stage_waiting", "restore"))
        hit_call = next(row for row in value.event_calls if row[0] == "fault_hit")
        end_call = next(row for row in value.event_calls
                        if row[:2] == ("stage_end", "restore"))
        self.assertEqual(waiting_call[2], [f"cursor-{rank}-1" for rank in range(4)])
        self.assertEqual(hit_call[2], [f"cursor-{rank}-2" for rank in range(4)])
        self.assertEqual(end_call[2], hit_call[2])
        self.assertEqual(evidence["post_cancel_cursors"], hit_call[2])

    def test_missing_hit_and_mismatched_rank_operation_stay_pending(self):
        for variant in ("missing-hit", "mismatched-operation", "mismatched-boundary",
                        "missing-wait", "partial-middle-wait", "pre-cancel-only",
                        "instrumentation-error"):
            with self.subTest(variant=variant):
                value = FakeCampaign()
                value.event_variant = variant
                _, _, status, evidence = self.run_case(value)
                self.assertEqual(status, "pending")
                self.assertIn("reason", evidence)
                self.assertEqual(value.reset_calls, [CASE["id"]])
                if variant == "partial-middle-wait":
                    self.assertEqual(
                        [row["rank"] for row in evidence["restore_waiting"]],
                        [None, None, None],
                    )
                if variant == "pre-cancel-only":
                    self.assertEqual(len(evidence["fault_hit_errors"]), 4)
                    self.assertEqual(len(evidence["restore_end_errors"]), 4)

    def test_completed_before_cancel_is_pending_and_full_response_is_retained(self):
        _, _, status, evidence = self.run_case(request_options={"completed": True})
        self.assertEqual(status, "pending")
        receipt = evidence["cancelled_request"]
        self.assertTrue(receipt["finished_before_cancel"])
        self.assertTrue(receipt["completed_stream"])
        self.assertIn("[DONE]", receipt["result"]["body_text"])

    def test_done_before_eof_is_completion_not_cancelled_interruption(self):
        _, _, status, evidence = self.run_case(
            request_options={"done_before_eof": True}
        )
        self.assertEqual(status, "pending")
        receipt = evidence["cancelled_request"]
        self.assertFalse(receipt["finished_before_cancel"])
        self.assertTrue(receipt["observed_completion"])
        self.assertTrue(receipt["completed_stream"])
        self.assertFalse(receipt["interrupted"])

    def test_missing_interruption_proof_is_pending(self):
        _, _, status, evidence = self.run_case(request_options={"missing": True})
        self.assertEqual(status, "pending")
        self.assertFalse(evidence["cancelled_request"]["interrupted"])

    def test_stuck_transport_and_cleanup_or_resource_failures_are_failures(self):
        _, _, status, evidence = self.run_case(request_options={"stuck": True})
        self.assertEqual(status, "fail")
        self.assertFalse(evidence["cancelled_request"]["terminated"])

        cleanup = FakeCampaign()
        cleanup.cleanup_errors = ["rank 2 reset failed"]
        _, _, status, evidence = self.run_case(cleanup)
        self.assertEqual(status, "fail")
        self.assertEqual(evidence["cleanup"]["errors"], cleanup.cleanup_errors)

        resource = FakeCampaign()
        resource.drain_status = "fail"
        _, _, status, evidence = self.run_case(resource)
        self.assertEqual(status, "fail")
        self.assertIn("resource validation failed", evidence["reason"])

    def test_deadline_is_pending_and_still_runs_cleanup(self):
        value = FakeCampaign()
        value.active_deadline = time.monotonic() - 1
        _, requests, status, evidence = self.run_case(value)
        self.assertEqual(status, "pending")
        self.assertEqual(requests, [])
        self.assertEqual(value.reset_calls, [CASE["id"]])
        self.assertEqual(evidence["cleanup"]["errors"], [])

    def test_deadline_during_recovery_stays_pending_after_cleanup(self):
        value = FakeCampaign()
        def idle():
            value.active_deadline = time.monotonic() - 1
            return "pending", {"reason": "deadline"}
        value.wait_for_api_idle = idle
        _, _, status, evidence = self.run_case(value)
        self.assertEqual(status, "pending")
        self.assertEqual(value.reset_calls, [CASE["id"]])
        self.assertNotIn("reservation_and_staging_drain", evidence)
        self.assertIn("deadline", evidence["reason"])

    def test_only_explicit_phase_guard_makes_late_recovery_error_pending(self):
        class PhaseGuard(OSError):
            guard_abort_cause = "phase_deadline"

        for error, expected in ((PhaseGuard("guarded"), "pending"),
                                (OSError("transport failed"), "fail")):
            with self.subTest(error=type(error).__name__):
                value = FakeCampaign()
                original_chat = value.http.chat
                calls = 0
                def chat(prompt, request_id):
                    nonlocal calls
                    calls += 1
                    if calls == 2:
                        value.active_deadline = time.monotonic() - 1
                        raise error
                    return original_chat(prompt, request_id)
                value.http.chat = chat
                _, _, status, evidence = self.run_case(value)
                self.assertEqual(status, expected)
                self.assertIn(type(error).__name__, evidence["recovery_error"])
                self.assertEqual(value.reset_calls, [CASE["id"]])

    def test_only_explicit_phase_guard_makes_late_action_error_pending(self):
        class PhaseGuard(OSError):
            guard_abort_cause = "phase_deadline"

        for error, expected in ((PhaseGuard("guarded"), "pending"),
                                (OSError("transport failed"), "fail")):
            with self.subTest(error=type(error).__name__):
                value = FakeCampaign()
                def chat(prompt, request_id):
                    value.active_deadline = time.monotonic() - 1
                    raise error
                value.http.chat = chat
                _, _, status, evidence = self.run_case(value)
                self.assertEqual(status, expected)
                self.assertIn(type(error).__name__, evidence["action_error"])
                self.assertEqual(value.reset_calls, [CASE["id"]])

    def test_actual_campaign_phase_deadline_during_drain_is_pending(self):
        self.assertEqual(
            campaign_driver.PhaseDeadlineError.guard_abort_cause, "phase_deadline"
        )
        value = FakeCampaign()
        def deadline():
            raise campaign_driver.PhaseDeadlineError("drain deadline")
        value._drain_barriers = deadline
        _, _, status, evidence = self.run_case(value)
        self.assertEqual(status, "pending")
        self.assertIn("PhaseDeadlineError", evidence["recovery_error"])
        self.assertEqual(value.reset_calls, [CASE["id"]])

    def test_phase_deadline_only_arm_and_release_failures_are_pending(self):
        for phase in ("arm", "release"):
            with self.subTest(phase=phase):
                value = FakeCampaign()
                value.set_errors[phase] = {
                    rank: campaign_driver.PhaseDeadlineError(f"rank {rank} deadline")
                    for rank in (1, 3)
                }
                _, _, status, evidence = self.run_case(value)
                self.assertEqual(status, "pending")
                control = evidence[f"fault_{phase}"]
                self.assertEqual(
                    [failure["rank"] for failure in control["failures"]], [1, 3]
                )
                self.assertEqual(
                    {failure["guard_abort_cause"] for failure in control["failures"]},
                    {"phase_deadline"},
                )
                calls = [row for row in value.faultctl_calls
                         if ("--pause-stage" in row[1]) == (phase == "arm")]
                self.assertEqual([rank for rank, _ in calls], [0, 1, 2, 3])
                self.assertEqual(value.reset_calls, [CASE["id"]])

    def test_mixed_phase_deadline_and_generic_set_failures_remain_fail(self):
        for phase in ("arm", "release"):
            with self.subTest(phase=phase):
                value = FakeCampaign()
                value.set_errors[phase] = {
                    0: campaign_driver.PhaseDeadlineError("rank 0 deadline"),
                    2: OSError("rank 2 transport error"),
                }
                _, _, status, evidence = self.run_case(value)
                self.assertEqual(status, "fail")
                control = evidence[f"fault_{phase}"]
                self.assertEqual(len(control["errors"]), 2)
                self.assertEqual(
                    [failure["guard_abort_cause"] for failure in control["failures"]],
                    ["phase_deadline", None],
                )
                calls = [row for row in value.faultctl_calls
                         if ("--pause-stage" in row[1]) == (phase == "arm")]
                self.assertEqual([rank for rank, _ in calls], [0, 1, 2, 3])
                self.assertEqual(value.reset_calls, [CASE["id"]])

    def test_cancellation_error_precedes_phase_expiry_and_still_cleans_up(self):
        value = FakeCampaign()
        requests = []
        class Request(FakeRequest):
            def cancel(self):
                self.cancelled.set()
                self.thread.alive = False
                self.finished_ns = time.time_ns()
                value.active_deadline = time.monotonic() - 1
                raise OSError("cancel transport failed")
        def factory(payload, request_id):
            request = Request(payload, request_id)
            requests.append(request)
            return request
        status, evidence = combination_cases.run_cache_eio_cancel_restore(
            value, CASE, 1, factory
        )
        self.assertEqual(status, "fail")
        self.assertIn("cancel transport failed", evidence["cancellation"]["error"])
        self.assertEqual(evidence["reason"], "client cancellation raised an error")
        self.assertNotIn("post_cancel_cursors", evidence)
        self.assertNotIn("fault_release", evidence)
        self.assertEqual(value.reset_calls, [CASE["id"]])
        self.assertTrue(evidence["cancelled_request"]["terminated"])

    def test_base_exception_propagates_after_client_and_fault_cleanup(self):
        value = FakeCampaign()
        original = value._events_all
        def events(*args, **kwargs):
            if args[2] == "stage_waiting":
                raise KeyboardInterrupt("stop")
            return original(*args, **kwargs)
        value._events_all = events
        requests = []
        def factory(payload, request_id):
            request = FakeRequest(payload, request_id)
            requests.append(request)
            return request
        with self.assertRaises(KeyboardInterrupt):
            combination_cases.run_cache_eio_cancel_restore(value, CASE, 1, factory)
        receipt = value.driver_events[0]
        self.assertEqual(value.reset_calls, [CASE["id"]])
        self.assertTrue(requests[0].cancelled.is_set())
        self.assertFalse(requests[0].thread.is_alive())
        self.assertEqual(receipt["kind"], "combination_interrupted")
        self.assertEqual(receipt["status"], "interrupted")
        self.assertEqual(receipt["evidence"]["request"]["payload"], requests[0].payload)
        self.assertTrue(receipt["evidence"]["cancellation"]["finalizer_only"])
        self.assertEqual(receipt["evidence"]["cleanup"]["errors"], [])

    def test_base_exception_receipt_failure_is_attached_and_not_swallowed(self):
        value = FakeCampaign()
        original = value._events_all
        def events(*args, **kwargs):
            if args[2] == "stage_waiting":
                raise KeyboardInterrupt("stop")
            return original(*args, **kwargs)
        value._events_all = events
        value.driver_event_error = OSError("receipt disk error")
        with self.assertRaises(KeyboardInterrupt) as raised:
            combination_cases.run_cache_eio_cancel_restore(
                value, CASE, 1,
                lambda payload, request_id: FakeRequest(payload, request_id),
            )
        self.assertEqual(value.reset_calls, [CASE["id"]])
        self.assertEqual(str(raised.exception), "stop")
        self.assertEqual(value.driver_events, [])
        self.assertTrue(any("receipt persistence failed" in note
                            for note in getattr(raised.exception, "__notes__", [])))

    def test_base_exception_during_recovery_persists_complete_cleanup_evidence(self):
        value = FakeCampaign()
        def interrupted_health():
            raise KeyboardInterrupt("recovery stop")
        value.health_code = interrupted_health
        with self.assertRaises(KeyboardInterrupt):
            self.run_case(value)
        receipt = value.driver_events[0]
        evidence = receipt["evidence"]
        self.assertEqual(receipt["exception"], "KeyboardInterrupt: recovery stop")
        self.assertEqual(evidence["cleanup"]["errors"], [])
        self.assertTrue(evidence["cancelled_request"]["terminated"])
        self.assertIn("payload", evidence["request"])

    def test_base_exception_in_reissue_cleanup_and_join_persists_exactly_once(self):
        for phase in ("reissue", "cleanup", "join"):
            with self.subTest(phase=phase):
                value = FakeCampaign()
                expected = f"{phase} stop"
                request_options = {"partial": True}
                if phase == "reissue":
                    value.http.interrupt_on_chat_call = 2
                elif phase == "cleanup":
                    value.cleanup_base_error = KeyboardInterrupt(expected)
                else:
                    request_options["join_error"] = KeyboardInterrupt(expected)
                requests = []
                def factory(payload, request_id):
                    request = FakeRequest(payload, request_id, **request_options)
                    requests.append(request)
                    return request
                with self.assertRaises(KeyboardInterrupt) as raised:
                    combination_cases.run_cache_eio_cancel_restore(
                        value, CASE, 3, factory
                    )
                self.assertEqual(str(raised.exception), expected)
                self.assertEqual(len(value.driver_events), 1)
                receipt = value.driver_events[0]
                evidence = receipt["evidence"]
                self.assertEqual(receipt["status"], "interrupted")
                self.assertEqual(receipt["exception"], f"KeyboardInterrupt: {expected}")
                self.assertEqual(evidence["request"]["payload"], requests[0].payload)
                self.assertEqual(
                    evidence["cancelled_request"]["result"]["body_text"],
                    requests[0].result["body_text"],
                )
                self.assertTrue(evidence["cancelled_request"]["terminated"])
                self.assertIn("cleanup", evidence)
                self.assertIn("cancellation", evidence)
                self.assertEqual(value.reset_calls, [CASE["id"]])
                if phase == "cleanup":
                    self.assertIn("interrupted", evidence["cleanup"]["errors"][0])
                else:
                    self.assertEqual(evidence["cleanup"]["errors"], [])


if __name__ == "__main__":
    unittest.main()
