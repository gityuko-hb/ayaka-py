from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from threading import RLock
from typing import Any

from ayaka.distributed.completion import CompletionFence
from ayaka.distributed.metadata import DistributedKVMetadata
from ayaka.distributed.process_group import DistributedStepError, TorchDistributedKVProcessGroup
from ayaka.distributed.transfer import TensorParallelKVStorage, build_replicated_transfer_engine
from ayaka.distributed.worker import RankLocalKVWorker
from ayaka.exceptions import InvariantViolationError
from ayaka.memory.manager import RuntimeMemoryManager
from ayaka.memory.tiering import HostKVStorage, TieringConfig, TransferEngine


class DistributedKVCoordinator:
    """Process-group consensus and rank-local lifecycle around KV metadata."""

    def __init__(
        self,
        process_group: TorchDistributedKVProcessGroup,
        *,
        source_rank: int = 0,
    ) -> None:
        if not 0 <= source_rank < process_group.world_size:
            raise ValueError("source_rank outside process group")
        self.process_group = process_group
        self.source_rank = source_rank
        self.worker = RankLocalKVWorker(process_group.rank)
        self.worker.start()

    def agree(self, local_grant: bool, *, timeout_s: float | None = None) -> bool:
        try:
            return self.process_group.all_grant(local_grant, timeout_s=timeout_s)
        except BaseException as exc:
            self.worker.fail(exc)
            raise

    def distribute_execution_view(
        self,
        view: Any | None,
        *,
        timeout_s: float | None = None,
    ) -> Mapping[str, Any]:
        metadata = (
            DistributedKVMetadata.from_execution_view(view)
            if self.process_group.rank == self.source_rank
            else None
        )
        try:
            received = self.process_group.broadcast_metadata(
                metadata,
                source_rank=self.source_rank,
                timeout_s=timeout_s,
            )
            return self.worker.begin(received)
        except BaseException as exc:
            self.worker.fail(exc)
            raise

    def complete_step(
        self,
        step_id: int,
        *,
        completion: CompletionFence | None = None,
        timeout_s: float | None = None,
    ) -> None:
        try:
            if completion is not None and not completion.wait(timeout_s):
                raise DistributedStepError(
                    "rank-local device completion timed out before KV barrier"
                )
            self.process_group.barrier(timeout_s=timeout_s)
            self.worker.complete(step_id)
        except BaseException as exc:
            self.worker.fail(exc)
            raise

    def recover_rank(
        self,
        reinitialize: Callable[[], TorchDistributedKVProcessGroup | None],
    ) -> None:
        """Rebuild rank resources and optionally replace the failed process group."""
        replacement: TorchDistributedKVProcessGroup | None = None

        def rebuild() -> None:
            nonlocal replacement
            replacement = reinitialize()

        self.worker.recover(rebuild)
        if replacement is not None:
            if replacement.rank != self.worker.rank:
                self.worker.fail("replacement process group changed local rank")
                raise InvariantViolationError("replacement process group changed local rank")
            if not 0 <= self.source_rank < replacement.world_size:
                self.worker.fail("replacement process group changed world geometry")
                raise InvariantViolationError(
                    "replacement process group changed world geometry"
                )
            self.process_group = replacement


class TensorParallelMemoryGroup:
    """One logical memory manager driving N rank-local KV storages (A12-01)."""

    def __init__(
        self,
        *,
        storages: TensorParallelKVStorage,
        total_pages: int,
        page_size: int,
        max_sequences: int,
        max_sequence_tokens: int | None = None,
        tiering: TieringConfig | None = None,
        transfer_engine: TransferEngine | None = None,
    ) -> None:
        self._storages = storages
        engine = transfer_engine
        mirrors: tuple[HostKVStorage, ...] = ()
        if tiering is not None and engine is None:
            engine, mirrors = build_replicated_transfer_engine(
                storages,
                host_capacity_pages=tiering.host_capacity_pages,
            )
        self._host_mirrors = mirrors
        self._manager = RuntimeMemoryManager(
            total_pages=total_pages,
            page_size=page_size,
            max_sequences=max_sequences,
            max_sequence_tokens=max_sequence_tokens,
            storage=storages,
            tiering=tiering,
            transfer_engine=engine,
        )

    @property
    def manager(self) -> RuntimeMemoryManager:
        return self._manager

    @property
    def storages(self) -> TensorParallelKVStorage:
        return self._storages

    @property
    def world_size(self) -> int:
        return self._storages.world_size

    @property
    def host_mirrors(self) -> tuple[HostKVStorage, ...]:
        return self._host_mirrors

    def rank_storage(self, rank: int) -> Any:
        return self._storages.rank_storage(rank)

    def verify_logical_consistency(self) -> None:
        storages = self._storages
        if self._manager.storage is not storages:
            raise InvariantViolationError(
                "the group manager is not backed by this group's tensor-parallel storage"
            )
        capacity = storages.capacity_pages
        page_size = storages.page_size
        for rank in range(storages.world_size):
            storage = storages.rank_storage(rank)
            if int(storage.capacity_pages) != capacity or int(storage.page_size) != page_size:
                raise InvariantViolationError(
                    f"rank {rank} page geometry diverged from the group's logical ordering"
                )

    def rank_block_tables(self, block_table: Sequence[int]) -> tuple[tuple[int, ...], ...]:
        shared = tuple(int(page) for page in block_table)
        return tuple(shared for _ in range(self.world_size))


@dataclass(frozen=True, slots=True)
class DataParallelReplica:
    """One independent replica: its own manager, its own everything."""

    rank: int
    manager: RuntimeMemoryManager


class DataParallelRouter:
    """Routes requests to replica-local memory managers (#28.2, #61.2)."""

    def __init__(self, replicas: Sequence[DataParallelReplica]) -> None:
        if not replicas:
            raise ValueError("a data-parallel router needs at least one replica")
        ranks = [replica.rank for replica in replicas]
        if len(set(ranks)) != len(ranks):
            raise ValueError("replica ranks must be unique")
        self._replicas = tuple(replicas)
        self._by_rank = {replica.rank: replica for replica in self._replicas}
        self._assignments: dict[str, int] = {}
        self._lock = RLock()

    @property
    def world_size(self) -> int:
        return len(self._replicas)

    @property
    def replicas(self) -> tuple[DataParallelReplica, ...]:
        return self._replicas

    def replica(self, rank: int) -> DataParallelReplica:
        return self._by_rank[rank]

    def assignment(self, request_id: str) -> DataParallelReplica | None:
        with self._lock:
            rank = self._assignments.get(request_id)
            return None if rank is None else self._by_rank[rank]

    def assign(self, request_id: str) -> DataParallelReplica:
        with self._lock:
            existing = self._assignments.get(request_id)
            if existing is not None:
                return self._by_rank[existing]
            chosen = max(
                self._replicas,
                key=lambda replica: (replica.manager.snapshot().free_pages, -replica.rank),
            )
            self._assignments[request_id] = chosen.rank
            return chosen

    def release(self, request_id: str) -> None:
        with self._lock:
            self._assignments.pop(request_id, None)
