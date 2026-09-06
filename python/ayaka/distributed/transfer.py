from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from itertools import count
from threading import RLock
from typing import Any

from ayaka.distributed.topology import PeerTopology, _is_cuda_device, _optional_torch
from ayaka.exceptions import StorageUnavailableError
from ayaka.handles import KVPageHandle, SequenceHandle
from ayaka.memory.tiering import (
    HostKVStorage,
    TransferDirection,
    TransferEngine,
    TransferMetrics,
    TransferOutcome,
    TransferState,
    TransferTicket,
    build_transfer_engine,
)


def _device_context(torch: Any | None, device: str | None) -> Any:
    """Enter ``device``'s CUDA context, or do nothing when there is none."""
    if torch is None or device is None or not _is_cuda_device(device):
        return nullcontext()
    return torch.cuda.device(device)

class TensorParallelKVStorage:
    """One KV storage per TP rank behind a single logical page geometry."""

    def __init__(self, storages: Sequence[Any]) -> None:
        if not storages:
            raise ValueError("a tensor-parallel storage needs at least one rank storage")
        self._storages = tuple(storages)
        first = self._storages[0]
        self._capacity_pages = int(first.capacity_pages)
        self._page_size = int(first.page_size)
        for rank, storage in enumerate(self._storages):
            if int(storage.capacity_pages) != self._capacity_pages:
                raise ValueError(
                    f"rank {rank} capacity_pages {storage.capacity_pages} disagrees with rank 0 "
                    f"{self._capacity_pages}; TP ranks must share one page geometry"
                )
            if int(storage.page_size) != self._page_size:
                raise ValueError(
                    f"rank {rank} page_size {storage.page_size} disagrees with rank 0 "
                    f"{self._page_size}; TP ranks must share one page geometry"
                )
        self._devices = tuple(str(getattr(storage, "device", "cpu")) for storage in self._storages)

    @property
    def world_size(self) -> int:
        return len(self._storages)

    @property
    def capacity_pages(self) -> int:
        return self._capacity_pages

    @property
    def page_size(self) -> int:
        return self._page_size

    @property
    def devices(self) -> tuple[str, ...]:
        return self._devices

    def rank_storage(self, rank: int) -> Any:
        """The storage owned by ``rank``."""
        if not 0 <= rank < self.world_size:
            raise IndexError(f"rank {rank} outside world size {self.world_size}")
        return self._storages[rank]

    def buffers(self) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        """Refuse: a TP group has one mirror per rank, not one mirror."""
        raise StorageUnavailableError(
            "a tensor-parallel storage has one host mirror per rank; build the tier with "
            "build_replicated_transfer_engine(...) and inject it as transfer_engine="
        )


@dataclass(slots=True)
class _ReplicatedTransfer:
    """Bookkeeping for one logical transfer fanned out across ranks."""

    ticket: TransferTicket
    outstanding: int
    failures: list[str] = field(default_factory=list)


class ReplicatedTransferEngine:
    """Fans one logical page move out to every TP rank."""

    def __init__(self, engines: Sequence[TransferEngine]) -> None:
        if not engines:
            raise ValueError("a replicated transfer engine needs at least one rank engine")
        self._engines = tuple(engines)
        self._bytes_per_page = sum(int(engine.bytes_per_page) for engine in self._engines)
        self._ids = count()
        self._pending: dict[int, _ReplicatedTransfer] = {}
        self._routes: dict[tuple[int, int], int] = {}
        self._submitted_total = 0
        self._completed_total = 0
        self._failed_total = 0
        self._lock = RLock()

    @property
    def world_size(self) -> int:
        return len(self._engines)

    @property
    def bytes_per_page(self) -> int:
        """Bytes one *logical* page costs, summed over every rank's shard."""
        return self._bytes_per_page

    def rank_engine(self, rank: int) -> TransferEngine:
        return self._engines[rank]

    def submit(
        self,
        direction: TransferDirection,
        *,
        device_page: int,
        host_slot: int,
        issue_epoch: int,
    ) -> TransferTicket:
        """Submit the same page/slot move on every rank under one ticket."""
        with self._lock:
            composite_id = next(self._ids)
            ticket = TransferTicket(
                ticket_id=composite_id,
                direction=direction,
                device_page=device_page,
                host_slot=host_slot,
                issue_epoch=issue_epoch,
            )
            record = _ReplicatedTransfer(ticket=ticket, outstanding=len(self._engines))
            self._pending[composite_id] = record
            self._submitted_total += 1
            for rank, engine in enumerate(self._engines):
                try:
                    rank_ticket = engine.submit(
                        direction,
                        device_page=device_page,
                        host_slot=host_slot,
                        issue_epoch=issue_epoch,
                    )
                except Exception as exc:
                    record.outstanding -= 1
                    record.failures.append(f"rank {rank} submit: {exc}")
                    continue
                self._routes[(rank, rank_ticket.ticket_id)] = composite_id
            return ticket

    def poll(self) -> tuple[TransferOutcome, ...]:
        """Resolve only the logical transfers every rank has finished."""
        with self._lock:
            for rank, engine in enumerate(self._engines):
                for outcome in engine.poll():
                    self._absorb(rank, outcome)
            return self._harvest()

    def drain(self) -> tuple[TransferOutcome, ...]:
        """Block until every rank is idle, then resolve everything."""
        with self._lock:
            for rank, engine in enumerate(self._engines):
                for outcome in engine.drain():
                    self._absorb(rank, outcome)
            return self._harvest()

    def _absorb(self, rank: int, outcome: TransferOutcome) -> None:
        composite_id = self._routes.pop((rank, outcome.ticket.ticket_id), None)
        if composite_id is None:
            return
        record = self._pending.get(composite_id)
        if record is None:
            return
        record.outstanding -= 1
        if outcome.state is TransferState.FAILED:
            record.failures.append(f"rank {rank}: {outcome.error}")

    def _harvest(self) -> tuple[TransferOutcome, ...]:
        resolved: list[TransferOutcome] = []
        for composite_id in [key for key, rec in self._pending.items() if rec.outstanding <= 0]:
            record = self._pending.pop(composite_id)
            if record.failures:
                self._failed_total += 1
                resolved.append(
                    TransferOutcome(
                        ticket=record.ticket,
                        state=TransferState.FAILED,
                        error="; ".join(record.failures),
                    )
                )
            else:
                self._completed_total += 1
                resolved.append(
                    TransferOutcome(ticket=record.ticket, state=TransferState.COMPLETED)
                )
        return tuple(resolved)

    @property
    def metrics(self) -> TransferMetrics:
        """Logical transfer counts, with byte totals summed over ranks."""
        with self._lock:
            rank_metrics = [engine.metrics for engine in self._engines]
            return TransferMetrics(
                submitted_total=self._submitted_total,
                completed_total=self._completed_total,
                failed_total=self._failed_total,
                bytes_to_host_total=sum(m.bytes_to_host_total for m in rank_metrics),
                bytes_to_device_total=sum(m.bytes_to_device_total for m in rank_metrics),
                pending=len(self._pending),
            )


def build_replicated_transfer_engine(
    storages: TensorParallelKVStorage,
    *,
    host_capacity_pages: int,
    prefer_async: bool = True,
) -> tuple[ReplicatedTransferEngine, tuple[HostKVStorage, ...]]:
    """Build one host mirror and one copy engine per rank."""
    torch = _optional_torch()
    mirrors: list[HostKVStorage] = []
    engines: list[TransferEngine] = []
    for rank in range(storages.world_size):
        storage = storages.rank_storage(rank)
        with _device_context(torch, storages.devices[rank]):
            mirror = HostKVStorage(storage, capacity_pages=host_capacity_pages)
            engines.append(build_transfer_engine(mirror, prefer_async=prefer_async))
        mirrors.append(mirror)
    return ReplicatedTransferEngine(engines), tuple(mirrors)


@dataclass(frozen=True, slots=True)
class KVTransferDescriptor:
    """One rank-to-rank KV move."""

    sequence: SequenceHandle
    page_handles: tuple[KVPageHandle, ...]
    source_rank: int
    destination_rank: int
    token_range: tuple[int, int]

    def __post_init__(self) -> None:
        if self.source_rank == self.destination_rank:
            raise ValueError("a transfer descriptor must name two distinct ranks")
        start, end = self.token_range
        if start < 0 or end < start:
            raise ValueError("token_range must be a non-negative, non-decreasing pair")


@dataclass(frozen=True, slots=True)
class PeerTransferReport:
    """What one :meth:`PeerPageMover.move` actually did."""

    pages: int
    bytes_moved: int
    peer_direct: bool
    host_staged: bool


class PeerPageMover:
    """Copies whole KV pages between two ranks' storages."""

    def __init__(
        self,
        storages: TensorParallelKVStorage,
        *,
        topology: PeerTopology | None = None,
        force_host_staging: bool = False,
    ) -> None:
        self._storages = storages
        self._topology = topology if topology is not None else PeerTopology.probe(storages.devices)
        self._force_host_staging = bool(force_host_staging)

    @property
    def topology(self) -> PeerTopology:
        return self._topology

    def move(self, descriptor: KVTransferDescriptor) -> PeerTransferReport:
        """Copy every page named by ``descriptor`` from source to destination."""
        source = self._storages.rank_storage(descriptor.source_rank)
        destination = self._storages.rank_storage(descriptor.destination_rank)
        peer_direct = not self._force_host_staging and self._topology.can_access(
            descriptor.source_rank, descriptor.destination_rank
        )
        moved_bytes = 0
        for handle in descriptor.page_handles:
            moved_bytes += self._copy_page(
                source, destination, handle.index, staged=not peer_direct
            )
        return PeerTransferReport(
            pages=len(descriptor.page_handles),
            bytes_moved=moved_bytes,
            peer_direct=peer_direct,
            host_staged=not peer_direct,
        )

    @staticmethod
    def _copy_page(source: Any, destination: Any, page: int, *, staged: bool) -> int:
        source_first, source_second = source.buffers()
        destination_first, destination_second = destination.buffers()
        moved = 0
        pairs = zip(
            (*source_first, *source_second),
            (*destination_first, *destination_second),
            strict=True,
        )
        for source_buffer, destination_buffer in pairs:
            payload = source_buffer[page]
            if staged:
                payload = payload.to("cpu")
            destination_buffer[page].copy_(payload)
            moved += int(source_buffer[page].numel()) * int(source_buffer.element_size())
        return moved
