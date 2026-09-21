"""Fail-closed lifecycle for one local device worker incarnation.

Worker state is a function of the *number* of active flights, never a single
``active_step``: with ``max_inflight > 1`` a RUNNING worker owns several
independent tickets at once. A failure that cannot prove device quiescence
moves the worker to FAILED and blocks admission until an explicit rebuild
mints a new incarnation.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Hashable
from enum import StrEnum
from threading import RLock

from ayaka.utils.validation import require_int

__all__ = ["FlightKey", "WorkerLifecycle", "WorkerState"]

FlightKey = Hashable


class WorkerState(StrEnum):
    CREATED = "created"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    CLOSING = "closing"
    CLOSED = "closed"
    FAILED = "failed"


class WorkerLifecycle:
    """Thread-safe state, flight accounting and identity for one worker.

    ``RUNNING`` is derived: a READY worker with one or more active flights.
    ``FAILED`` is absorbing until :meth:`recover` runs a caller-supplied
    reinitializer, because an ambiguous device failure must never be clearable
    while the old streams/slabs/graphs are still reachable.
    """

    def __init__(self, *, rank: int = 0, pid: int | None = None, device_index: int = 0) -> None:
        require_int(rank, "rank", minimum=0)
        require_int(device_index, "device_index", minimum=0)
        self.rank = rank
        self._pid = os.getpid() if pid is None else pid
        self._device_index = device_index
        self._state = WorkerState.CREATED
        self._flights: dict[FlightKey, str] = {}
        self._failure: str | None = None
        self._generation = 0
        self._lock = RLock()

    @property
    def state(self) -> WorkerState:
        with self._lock:
            if self._state is WorkerState.READY and self._flights:
                return WorkerState.RUNNING
            return self._state

    @property
    def accepting(self) -> bool:
        with self._lock:
            return self._state is WorkerState.READY

    @property
    def active_flights(self) -> tuple[FlightKey, ...]:
        with self._lock:
            return tuple(self._flights)

    @property
    def num_flights(self) -> int:
        with self._lock:
            return len(self._flights)

    @property
    def failure(self) -> str | None:
        with self._lock:
            return self._failure

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    @property
    def pid(self) -> int:
        with self._lock:
            return self._pid

    @property
    def device_index(self) -> int:
        with self._lock:
            return self._device_index

    def begin_initializing(self) -> None:
        with self._lock:
            if self._state is not WorkerState.CREATED:
                raise RuntimeError("only a created worker can initialize")
            self._state = WorkerState.INITIALIZING

    def mark_ready(self) -> None:
        with self._lock:
            if self._state is not WorkerState.INITIALIZING:
                raise RuntimeError("worker must be initializing before it is ready")
            self._state = WorkerState.READY

    def begin_flight(self, key: FlightKey, *, tag: str = "") -> None:
        """Register work before any enqueue; reject unless the worker admits."""
        with self._lock:
            if self._state is not WorkerState.READY:
                raise RuntimeError(f"worker is {self._state.value} and cannot accept work")
            if key in self._flights:
                raise ValueError("flight is already active on this worker")
            self._flights[key] = str(tag)

    def end_flight(self, key: FlightKey) -> bool:
        """Drop a flight once its quiescence proof completed; safe to repeat."""
        with self._lock:
            return self._flights.pop(key, None) is not None

    def fail(self, error: object) -> None:
        """Enter FAILED; active flights stay registered (unknown quiescence)."""
        with self._lock:
            if self._state is WorkerState.CLOSED:
                return
            self._failure = str(error)
            self._state = WorkerState.FAILED

    def recover(self, reinitialize: Callable[[], None]) -> int:
        """Rebuild resources with new streams/buffers, then mint an incarnation."""
        with self._lock:
            if self._state is not WorkerState.FAILED:
                raise RuntimeError("only a failed worker can recover")
            if self._flights:
                raise RuntimeError("cannot recover a worker with active flights")
            self.validate_runtime()
            # Serialize recovery against close and another recovery. A callback
            # failure leaves this incarnation FAILED, with its identity intact.
            reinitialize()
            self._generation += 1
            self._failure = None
            self._state = WorkerState.READY
            return self._generation

    def request_closing(self) -> WorkerState:
        """Stop admission; FAILED/CLOSED are not silently cleared."""
        with self._lock:
            if self._state in (
                WorkerState.CREATED,
                WorkerState.INITIALIZING,
                WorkerState.READY,
            ):
                self._state = WorkerState.CLOSING
            return self.state

    def mark_closed(self) -> None:
        with self._lock:
            if self._flights:
                raise RuntimeError("cannot close a worker with active flights")
            self._state = WorkerState.CLOSED

    def validate_runtime(self, *, pid: int | None = None, device_index: int | None = None) -> None:
        """Refuse a fork or wrong-device use instead of reusing stale resources."""
        with self._lock:
            expected_pid = self._pid
            expected_device = self._device_index
        current_pid = os.getpid() if pid is None else pid
        if current_pid != expected_pid:
            raise RuntimeError(
                f"worker on device {expected_device} was created in pid {expected_pid} and "
                f"is being used in pid {current_pid}: device resources do not survive fork"
            )
        if device_index is not None and device_index != expected_device:
            raise RuntimeError(
                f"worker owns device {expected_device} but was asked for device {device_index}"
            )

    def __repr__(self) -> str:
        return (
            f"<WorkerLifecycle rank={self.rank} device={self._device_index} "
            f"state={self.state.value} flights={self.num_flights} gen={self._generation}>"
        )
