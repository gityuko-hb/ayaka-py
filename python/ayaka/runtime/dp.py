"""Data-parallel composition of in-process replica engines.

Each replica keeps its own scheduler, executor and KV manager. The
orchestrator owns only routing and fan-out: new requests are assigned to a
replica by the (optionally prefix-aware) router, every ``step()`` drives all
replicas, and aborts are forwarded to the owning replica only. There is no
cross-replica state — assignments are sticky and released explicitly when the
caller retires a request.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ayaka.distributed.coordinator import DataParallelRouter
from ayaka.request.lifecycle import RequestLifecycle
from ayaka.request.schema import Request

__all__ = ["ReplicaEngine", "DataParallelOrchestrator"]


@runtime_checkable
class ReplicaEngine(Protocol):
    """The per-replica surface the orchestrator drives."""

    def submit(self, request: Request) -> RequestLifecycle: ...

    def abort(self, request_id: str) -> bool: ...

    def step(self) -> bool: ...

    @property
    def has_unfinished(self) -> bool: ...


class DataParallelOrchestrator:
    """Fan a request stream out to replica engines (#8 · DP scheduler loop)."""

    def __init__(
        self,
        *,
        router: DataParallelRouter,
        engines: dict[int, ReplicaEngine],
    ) -> None:
        expected = {replica.rank for replica in router.replicas}
        if set(engines) != expected:
            raise ValueError(
                f"engines must cover exactly the router replica ranks {sorted(expected)}"
            )
        self._router = router
        self._engines = dict(engines)

    @property
    def router(self) -> DataParallelRouter:
        return self._router

    @property
    def world_size(self) -> int:
        return len(self._engines)

    def engine_for(self, rank: int) -> ReplicaEngine:
        return self._engines[rank]

    def assignment(self, request_id: str) -> int | None:
        replica = self._router.assignment(request_id)
        return None if replica is None else replica.rank

    def submit(self, request: Request) -> RequestLifecycle:
        replica = self._router.assign(str(request.request_id))
        return self._engines[replica.rank].submit(request)

    def abort(self, request_id: str) -> bool:
        replica = self._router.assignment(request_id)
        if replica is None:
            return False
        return self._engines[replica.rank].abort(request_id)

    def forget(self, request_id: str) -> None:
        """Drop the sticky assignment once the request is retired everywhere."""
        self._router.release(request_id)

    def step(self) -> bool:
        """One scheduling iteration per replica; True when any replica moved."""
        did = False
        for engine in self._engines.values():
            if engine.step():
                did = True
        return did

    @property
    def has_unfinished(self) -> bool:
        return any(engine.has_unfinished for engine in self._engines.values())
