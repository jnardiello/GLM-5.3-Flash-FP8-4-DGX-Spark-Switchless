#!/usr/bin/env python3
"""Reviewed compound resilience cases built from the campaign's existing controls."""

from __future__ import annotations

import time
from typing import Any, Callable


class _Outcome(Exception):
    def __init__(self, status: str, reason: str):
        self.status = status
        self.reason = reason
        super().__init__(reason)


def _safety_stopped(campaign: Any) -> bool:
    return bool(campaign.telemetry.safety_event.is_set())


def _event_timeout(campaign: Any) -> float:
    if _safety_stopped(campaign):
        raise _Outcome("fail", "campaign safety guard activated")
    remaining = campaign.active_deadline - time.monotonic()
    if remaining <= 0:
        raise _Outcome("pending", "active case deadline reached before event proof")
    return min(120.0, remaining)


def _ranked(
    events: list[dict[str, Any]], errors: list[str]
) -> list[dict[str, Any]]:
    complete = len(events) == 4 and not errors
    result: list[dict[str, Any]] = []
    for rank, event in enumerate(events):
        source_rank = event.get("rank")
        result.append({
            "rank": (rank if complete else source_rank
                     if (isinstance(source_rank, int) and not isinstance(source_rank, bool)
                         and 0 <= source_rank < 4) else None),
            "event": event,
        })
    return result


def _valid_response(response: dict[str, Any]) -> bool:
    body = response.get("body")
    return (response.get("status") == 200 and isinstance(body, dict)
            and bool(body.get("choices")))


def _set_all(campaign: Any, arguments: list[str]) -> dict[str, Any]:
    evidence: dict[str, Any] = {"ranks": {}, "errors": [], "failures": []}
    for rank in range(4):
        try:
            evidence["ranks"][rank] = campaign._faultctl(rank, arguments)
        except Exception as error:
            evidence["errors"].append(f"rank {rank}: {type(error).__name__}: {error}")
            evidence["failures"].append({
                "rank": rank,
                "error_type": type(error).__name__,
                "error": str(error),
                "guard_abort_cause": getattr(error, "guard_abort_cause", None),
            })
    return evidence


def _only_phase_deadline_failures(campaign: Any, evidence: dict[str, Any]) -> bool:
    failures = evidence.get("failures", [])
    return bool(failures) and not _safety_stopped(campaign) and all(
        failure.get("guard_abort_cause") == "phase_deadline"
        for failure in failures
    )


def _request_evidence(request: Any, cancellation: dict[str, Any]) -> dict[str, Any]:
    result = getattr(request, "result", None)
    finished_ns = getattr(request, "finished_ns", None)
    requested_ns = cancellation.get("requested_ns")
    finished_before_cancel = (
        isinstance(finished_ns, int)
        and isinstance(requested_ns, int)
        and finished_ns <= requested_ns
    )
    finish_reasons = result.get("finish_reasons") if isinstance(result, dict) else None
    observed_completion = bool(
        isinstance(result, dict)
        and (
            result.get("done") is True
            or (
                isinstance(finish_reasons, list)
                and any(isinstance(reason, str) and reason for reason in finish_reasons)
                and result.get("error_event") is False
                and result.get("parse_errors") == []
            )
        )
    )
    completed_stream = observed_completion
    terminated = not request.thread.is_alive()
    interrupted = bool(
        cancellation.get("thread_alive_before_cancel")
        and not finished_before_cancel
        and terminated
        and not completed_stream
        and (getattr(request, "error", None) is not None
             or isinstance(result, dict) and result.get("read_complete") is False)
    )
    cancelled = getattr(request, "cancelled", None)
    return {
        "request_id": getattr(request, "request_id", None),
        "http_status": getattr(request, "http_status", None),
        "response_headers": getattr(request, "response_headers", {}),
        "result": result,
        "error": getattr(request, "error", None),
        "started_ns": getattr(request, "started_ns", None),
        "first_byte_ns": getattr(request, "first_byte_ns", None),
        "finished_ns": finished_ns,
        "first_byte_hex": getattr(request, "first_byte_hex", None),
        "cancelled_flag": cancelled.is_set() if hasattr(cancelled, "is_set") else None,
        "terminated": terminated,
        "finished_before_cancel": finished_before_cancel,
        "observed_completion": observed_completion,
        "completed_stream": completed_stream,
        "interrupted": interrupted,
    }


def run_cache_eio_cancel_restore(
    campaign: Any,
    case: dict[str, Any],
    repetition: int,
    request_factory: Callable[[dict[str, Any], str], Any],
) -> tuple[str, dict[str, Any]]:
    """Inject read EIO while cancelling a replay paused at four-rank restore."""
    evidence: dict[str, Any] = {
        "case_id": case["id"],
        "repetition": repetition,
        "combination": "cache-eio-cancel-restore",
    }
    try:
        return _run_cache_eio_cancel_restore(
            campaign, case, repetition, request_factory, evidence
        )
    except BaseException as error:
        event = {
            "kind": "combination_interrupted",
            "case_id": evidence["case_id"],
            "repetition": evidence["repetition"],
            "status": "interrupted",
            "exception": f"{type(error).__name__}: {error}",
            "evidence": evidence,
        }
        try:
            campaign.append_driver_event(event)
            evidence["interrupted_receipt"] = {"persisted": True}
        except Exception as persist_error:
            message = (
                "interrupted combination receipt persistence failed: "
                f"{type(persist_error).__name__}: {persist_error}"
            )
            evidence["interrupted_receipt"] = {"persisted": False, "error": message}
            add_note = getattr(error, "add_note", None)
            if callable(add_note):
                add_note(message)
        raise


def _run_cache_eio_cancel_restore(
    campaign: Any,
    case: dict[str, Any],
    repetition: int,
    request_factory: Callable[[dict[str, Any], str], Any],
    evidence: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    case_id = case["id"]
    status = "fail"
    prompt: str | None = None
    request: Any | None = None
    request_started = False
    cancellation: dict[str, Any] = {}

    try:
        if _safety_stopped(campaign):
            raise _Outcome("fail", "campaign safety guard was already active")
        _event_timeout(campaign)
        campaign._set_case(case_id)
        prompt = campaign.http.exact_prompt(
            32_000, f"{case_id} repetition {repetition} shared snapshot."
        )
        seed_cursors = [campaign._cursor(rank) for rank in range(4)]
        seed = campaign.http.chat(prompt, request_id=f"{case_id}-seed-{repetition}")
        evidence["seed"] = seed
        if not _valid_response(seed):
            raise _Outcome("fail", "seed request did not return a valid HTTP response")
        publications, publication_errors = campaign._events_all(
            seed_cursors,
            case_id,
            "stage_end",
            stage="publication",
            timeout=_event_timeout(campaign),
        )
        evidence["seed_publications"] = _ranked(publications, publication_errors)
        evidence["seed_publication_errors"] = publication_errors
        digest = None if publication_errors else campaign._publication_digest(publications)
        evidence["seed_digest"] = digest
        if digest is None:
            raise _Outcome(
                "pending", "exact four-rank committed seed publication was not proven"
            )

        _event_timeout(campaign)
        arm_arguments = [
            "set", "--mode", "eio", "--operation", "read",
            "--case-id", case_id, "--remaining", "1",
            "--pause-stage", "restore",
        ]
        evidence["fault_arm"] = _set_all(campaign, arm_arguments)
        if evidence["fault_arm"]["errors"]:
            if _only_phase_deadline_failures(campaign, evidence["fault_arm"]):
                raise _Outcome("pending", "phase deadline interrupted all-rank fault arm")
            raise _Outcome("fail", "failed to arm the restore fault on every rank")

        _event_timeout(campaign)
        action_cursors = [campaign._cursor(rank) for rank in range(4)]
        payload = campaign.http.chat_payload(prompt, stream=True, max_tokens=256)
        request_id = f"{case_id}-replay-{repetition}"
        evidence["request"] = {"request_id": request_id, "payload": payload}
        request = request_factory(payload, request_id)
        request.start()
        request_started = True

        waiting, waiting_errors = campaign._events_all(
            action_cursors,
            case_id,
            "stage_waiting",
            stage="restore",
            timeout=_event_timeout(campaign),
        )
        evidence["restore_waiting"] = _ranked(waiting, waiting_errors)
        evidence["restore_waiting_errors"] = waiting_errors
        waiting_valid = (
            not waiting_errors
            and len(waiting) == 4
            and all(
                event.get("digest") == digest
                and isinstance(event.get("operation_id"), str)
                and bool(event["operation_id"])
                for event in waiting
            )
        )
        if not waiting_valid:
            raise _Outcome(
                "pending", "four-rank restore synchronization for the seed digest was not proven"
            )

        _event_timeout(campaign)
        cancellation.update(
            requested_ns=time.time_ns(),
            thread_alive_before_cancel=request.thread.is_alive(),
            finished_ns_before_cancel=getattr(request, "finished_ns", None),
        )
        try:
            request.cancel()
        except Exception as error:
            cancellation["error"] = f"{type(error).__name__}: {error}"
        cancellation["returned_ns"] = time.time_ns()
        evidence["cancellation"] = cancellation
        if cancellation.get("error"):
            raise _Outcome("fail", "client cancellation raised an error")

        _event_timeout(campaign)
        post_cancel_cursors = [campaign._cursor(rank) for rank in range(4)]
        evidence["post_cancel_cursors"] = post_cancel_cursors
        release_arguments = [
            "set", "--mode", "eio", "--operation", "read",
            "--case-id", case_id, "--remaining", "1",
        ]
        evidence["fault_release"] = _set_all(campaign, release_arguments)
        if evidence["fault_release"]["errors"]:
            if _only_phase_deadline_failures(campaign, evidence["fault_release"]):
                raise _Outcome("pending", "phase deadline interrupted all-rank fault release")
            raise _Outcome("fail", "failed to release the restore pause on every rank")

        fault_hits, hit_errors = campaign._events_all(
            post_cancel_cursors,
            case_id,
            "fault_hit",
            mode="eio",
            timeout=_event_timeout(campaign),
        )
        restore_ends, restore_errors = campaign._events_all(
            post_cancel_cursors,
            case_id,
            "stage_end",
            stage="restore",
            timeout=_event_timeout(campaign),
        )
        evidence["fault_hits"] = _ranked(fault_hits, hit_errors)
        evidence["fault_hit_errors"] = hit_errors
        evidence["restore_ends"] = _ranked(restore_ends, restore_errors)
        evidence["restore_end_errors"] = restore_errors
        hits_valid = (
            not hit_errors
            and len(fault_hits) == 4
            and all(
                event.get("mode") == "eio"
                and event.get("operation") == "read"
                and isinstance(event.get("time_ns"), int)
                and isinstance(waiting[rank].get("time_ns"), int)
                and waiting[rank]["time_ns"] <= event["time_ns"]
                for rank, event in enumerate(fault_hits)
            )
        )
        restore_valid = (
            not restore_errors
            and len(fault_hits) == 4
            and len(restore_ends) == 4
            and all(
                event.get("outcome") == "error"
                and event.get("digest") == digest
                and event.get("operation_id") == waiting[rank].get("operation_id")
                and not event.get("instrumentation_error")
                and isinstance(event.get("time_ns"), int)
                and isinstance(fault_hits[rank].get("time_ns"), int)
                and fault_hits[rank]["time_ns"] <= event["time_ns"]
                for rank, event in enumerate(restore_ends)
            )
        )
        if not hits_valid:
            raise _Outcome("pending", "four fresh rank-local read EIO hits were not proven")
        if not restore_valid:
            raise _Outcome(
                "pending", "four matching rank-local restore error boundaries were not proven"
            )
        status = "pass"
    except _Outcome as outcome:
        status = outcome.status
        evidence["reason"] = outcome.reason
    except Exception as error:
        evidence["action_error"] = f"{type(error).__name__}: {error}"
        if (getattr(error, "guard_abort_cause", None) == "phase_deadline"
                and not _safety_stopped(campaign)):
            status = "pending"
            evidence["reason"] = "phase deadline guard interrupted the combination"
        else:
            status = "fail"
    finally:
        try:
            try:
                if request is not None and request_started:
                    if "requested_ns" not in cancellation:
                        cancellation.update(
                            requested_ns=time.time_ns(),
                            thread_alive_before_cancel=request.thread.is_alive(),
                            finished_ns_before_cancel=getattr(request, "finished_ns", None),
                            finalizer_only=True,
                        )
                    evidence["cancellation"] = cancellation
                    try:
                        request.cancel()
                    except Exception as error:
                        cancellation.setdefault(
                            "error", f"{type(error).__name__}: {error}"
                        )
                    cancellation.setdefault("returned_ns", time.time_ns())
                    evidence["cancellation"] = cancellation
            finally:
                evidence["cleanup"] = {
                    "errors": ["cleanup was interrupted before returning evidence"]
                }
                try:
                    evidence["cleanup"] = campaign._reset_faults(case_id)
                except Exception as error:
                    evidence["cleanup"] = {
                        "errors": [f"{type(error).__name__}: {error}"]
                    }
        finally:
            if request is not None and request_started:
                evidence["cancelled_request"] = _request_evidence(request, cancellation)
                try:
                    request.thread.join(timeout=30)
                except Exception as error:
                    evidence.setdefault("client_cleanup_errors", []).append(
                        f"{type(error).__name__}: {error}"
                    )
                evidence["cancelled_request"] = _request_evidence(request, cancellation)

    cleanup_errors = evidence.get("cleanup", {}).get("errors", [])
    request_receipt = evidence.get("cancelled_request")
    if cleanup_errors or evidence.get("client_cleanup_errors"):
        status = "fail"
        evidence["reason"] = "combination cleanup failed"
    elif isinstance(request_receipt, dict):
        if not request_receipt["terminated"]:
            status = "fail"
            evidence["reason"] = "cancelled client transport did not terminate"
        elif (status == "pass" and
              (request_receipt["finished_before_cancel"]
               or request_receipt["completed_stream"])):
            status = "pending"
            evidence["reason"] = "request completed before cancellation was proven"
        elif status == "pass" and not request_receipt["interrupted"]:
            status = "pending"
            evidence["reason"] = "client interruption after cancellation was not proven"
        elif (isinstance(request_receipt.get("http_status"), int)
              and request_receipt["http_status"] != 200):
            status = "fail"
            evidence["reason"] = "cancelled request returned an unexpected HTTP status"

    if _safety_stopped(campaign):
        status = "fail"
        evidence["reason"] = "campaign safety guard activated"
    elif cleanup_errors or evidence.get("client_cleanup_errors"):
        pass
    elif time.monotonic() >= campaign.active_deadline:
        if status != "fail":
            status = "pending"
            evidence["reason"] = "active case deadline prevented recovery proof"
    elif prompt is not None:
        try:
            reissue = campaign.http.chat(prompt, request_id=f"{case_id}-reissue-{repetition}")
            evidence["reissue"] = reissue
            evidence["health_200"] = campaign.health_code() == 200
            idle_status, idle = campaign.wait_for_api_idle()
            evidence["api_idle"] = {"status": idle_status, "evidence": idle}
            if (time.monotonic() >= campaign.active_deadline
                    and not _safety_stopped(campaign)):
                raise _Outcome("pending", "active case deadline interrupted recovery proof")
            barriers = campaign._drain_barriers()
            drain_status, drain = campaign.wait_for_drain(barriers)
            evidence["reservation_and_staging_drain"] = {
                "status": drain_status, "barriers": barriers, "evidence": drain
            }
            if (not _valid_response(reissue) or not evidence["health_200"]
                    or idle_status == "fail" or drain_status == "fail"):
                status = "fail"
                evidence["reason"] = "post-cleanup service or resource validation failed"
            elif idle_status != "pass" or drain_status != "pass":
                if status != "fail":
                    status = "pending"
                    evidence["reason"] = "post-cleanup idle or resource proof is incomplete"
        except _Outcome as outcome:
            if status != "fail":
                status = outcome.status
                evidence["reason"] = outcome.reason
        except Exception as error:
            evidence["recovery_error"] = f"{type(error).__name__}: {error}"
            if (getattr(error, "guard_abort_cause", None) == "phase_deadline"
                    and not _safety_stopped(campaign) and status != "fail"):
                status = "pending"
                evidence["reason"] = "phase deadline guard interrupted recovery proof"
            else:
                status = "fail"
                evidence["reason"] = "post-cleanup recovery validation raised an error"

    if _safety_stopped(campaign):
        status = "fail"
        evidence["reason"] = "campaign safety guard activated"

    return status, evidence
