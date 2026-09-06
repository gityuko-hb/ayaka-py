from __future__ import annotations

from collections.abc import Callable, Mapping
from enum import Enum, auto
from threading import RLock
from typing import Any

from ayaka.distributed.metadata import DistributedKVMetadata
from ayaka.exceptions import InvariantViolationError


class RankLocalKVWorkerState(Enum):
    CREATED = auto()
    READY = auto()
    RUNNING = auto()
    FAILED = auto()
    CLOSED = auto()


class RankLocalKVWorker:
    """Fail-closed lifecycle for one process/rank's KV execution state."""

    def __init__(self, rank: int) -> None:
        if rank < 0:
            raise ValueError("rank must be non-negative")
        self.rank = rank
        self._state = RankLocalKVWorkerState.CREATED
        self._active_step: int | None = None
        self._last_step = -1
        self._failure: str | None = None
        self._generation = 0
        self._lock = RLock()

    @property
    def state(self) -> RankLocalKVWorkerState:
        with self._lock:
            return self._state

    @property
    def failure(self) -> str | None:
        with self._lock:
            return self._failure

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    def start(self) -> None:
        with self._lock:
            if self._state is not RankLocalKVWorkerState.CREATED:
                raise RuntimeError("only a created KV worker can start")
            self._state = RankLocalKVWorkerState.READY

    def begin(self, metadata: DistributedKVMetadata) -> Mapping[str, Any]:
        metadata.verify()
        with self._lock:
            if self._state is not RankLocalKVWorkerState.READY:
                raise RuntimeError(f"rank {self.rank} KV worker is not ready")
            if metadata.step_id <= self._last_step:
                raise InvariantViolationError(
                    f"rank {self.rank} received stale KV step {metadata.step_id}"
                )
            self._active_step = metadata.step_id
            self._state = RankLocalKVWorkerState.RUNNING
            return metadata.payload

    def complete(self, step_id: int) -> None:
        with self._lock:
            if self._state is not RankLocalKVWorkerState.RUNNING:
                raise RuntimeError("KV worker has no running step")
            if self._active_step != step_id:
                raise InvariantViolationError("KV worker completion step mismatch")
            self._last_step = step_id
            self._active_step = None
            self._state = RankLocalKVWorkerState.READY

    def fail(self, error: BaseException | str) -> None:
        with self._lock:
            self._failure = str(error)
            self._active_step = None
            self._state = RankLocalKVWorkerState.FAILED

    def recover(self, reinitialize: Callable[[], None]) -> None:
        """Rebuild rank-local storage/streams before rejoining collectives."""
        with self._lock:
            if self._state is not RankLocalKVWorkerState.FAILED:
                raise RuntimeError("only a failed KV worker can recover")
        reinitialize()
        with self._lock:
            self._generation += 1
            self._failure = None
            self._state = RankLocalKVWorkerState.READY

    def close(self) -> None:
        with self._lock:
            if self._state is RankLocalKVWorkerState.RUNNING:
                raise RuntimeError("cannot close a KV worker with an active step")
            self._state = RankLocalKVWorkerState.CLOSED
