# SPDX-License-Identifier: Apache-2.0
"""Project-authored host-memory admission for the SparkCache integration.

Reservations account for a caller's full peak without allocating that memory.
They remain charged until the caller explicitly releases them.
"""

from __future__ import annotations

import os
from pathlib import Path
import threading
from typing import Callable


def read_mem_available(
    path: str | os.PathLike[str] = "/proc/meminfo",
) -> int | None:
    """Return Linux MemAvailable in bytes, or ``None`` when it is unusable."""
    try:
        contents = Path(path).read_text(encoding="ascii")
    except (OSError, UnicodeError):
        return None

    available_kib: int | None = None
    for line in contents.splitlines():
        fields = line.split()
        if not fields or fields[0] != "MemAvailable:":
            continue
        if available_kib is not None or len(fields) != 3 or fields[2] != "kB":
            return None
        if not fields[1].isdigit():
            return None
        available_kib = int(fields[1])

    if available_kib is None:
        return None
    return available_kib * 1024


def _require_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


class MemoryReservation:
    """A held portion of a :class:`MemoryBudget`."""

    __slots__ = ("_budget", "_peak_bytes", "_released")

    def __init__(self, budget: "MemoryBudget", peak_bytes: int) -> None:
        self._budget = budget
        self._peak_bytes = peak_bytes
        self._released = False

    def release(self) -> None:
        """Release this reservation; repeated calls have no effect."""
        self._budget._release(self)

    def __enter__(self) -> "MemoryReservation":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.release()


class MemoryBudget:
    """Serialize conservative host-memory admission across concurrent workers."""

    def __init__(
        self,
        max_bytes: int,
        min_available_bytes: int,
        available_reader: Callable[[], int | None] = read_mem_available,
    ) -> None:
        _require_positive_int("max_bytes", max_bytes)
        _require_positive_int("min_available_bytes", min_available_bytes)
        if not callable(available_reader):
            raise TypeError("available_reader must be callable")
        self._max_bytes = max_bytes
        self._min_available_bytes = min_available_bytes
        self._available_reader = available_reader
        self._reserved_bytes = 0
        self._lock = threading.Lock()

    @property
    def reserved_bytes(self) -> int:
        with self._lock:
            return self._reserved_bytes

    def try_reserve(self, peak_bytes: int) -> MemoryReservation | None:
        """Reserve a full peak when both the cap and availability floor allow it."""
        _require_positive_int("peak_bytes", peak_bytes)
        with self._lock:
            try:
                available = self._available_reader()
            except Exception:
                return None
            if (
                isinstance(available, bool)
                or not isinstance(available, int)
                or available < 0
            ):
                return None

            new_reserved = self._reserved_bytes + peak_bytes
            if new_reserved > self._max_bytes:
                return None
            if available - new_reserved < self._min_available_bytes:
                return None

            reservation = MemoryReservation(self, peak_bytes)
            self._reserved_bytes = new_reserved
            return reservation

    def _release(self, reservation: MemoryReservation) -> None:
        with self._lock:
            if reservation._released:
                return
            if reservation._budget is not self:
                raise ValueError("reservation belongs to a different budget")
            self._reserved_bytes -= reservation._peak_bytes
            reservation._released = True
