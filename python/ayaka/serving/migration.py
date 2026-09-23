"""Chờ-KV migration: drive parked requests out of ``WAITING_REMOTE_KV``.

The router parks a request when it believes a peer holds reusable prefix KV.
This controller owns the wait: it stages one node-transfer ticket per parked
request, resolves completions into ``engine.release_remote_kv`` and failures
(transfer errors or an expired lease) into ``engine.abandon_remote_kv`` so the
request falls back to a full local prefill.

Lease semantics: every staged fetch carries a deadline. A stalled or slow
peer can never park a request forever — ``poll`` abandons expired stages and
cancels their transport tickets. Real KV bytes arrive through ``chunk_source``
(the layer-wise transfer workstream); without one, reference-complete tickets
settle immediately, which is the correct contract for the chờ-KV milestone.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ayaka.distributed.node_transport import NodeTransferEngine, NodeTransferTicket
from ayaka.memory.tiering import TransferState
from ayaka.serving.router import RemoteKVPending

__all__ = ["MigrationController"]

#: ``chunk_source(request_id, destination_node, token_range) -> bytes chunks``
ChunkSource = Callable[[str, str, tuple[int, int]], Sequence[bytes]]


@dataclass(frozen=True, slots=True)
class _StagedFetch:
    ticket: NodeTransferTicket
    destination: str
    deadline: float
    prefix_tokens: int


@dataclass(frozen=True, slots=True)
class MigrationSnapshot:
    """Cumulative migration accounting; bounded, diagnostics-only."""

    staged_total: int
    completed_total: int
    abandoned_total: int
    lease_expired_total: int
    pending: int


class MigrationController:
    """Stage, poll and settle remote-KV fetches for parked requests."""

    def __init__(
        self,
        engine,
        *,
        transport: NodeTransferEngine,
        pending: RemoteKVPending,
        chunk_source: ChunkSource | None = None,
        lease_timeout: float = 30.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if not isinstance(transport, NodeTransferEngine):
            raise TypeError("transport must be NodeTransferEngine")
        if not isinstance(pending, RemoteKVPending):
            raise TypeError("pending must be RemoteKVPending")
        if lease_timeout <= 0:
            raise ValueError("lease_timeout must be positive")
        self._engine = engine
        self._transport = transport
        self._pending = pending
        self._chunk_source = chunk_source
        self._lease_timeout = lease_timeout
        self._clock = clock or time.monotonic
        self._lock = threading.RLock()
        self._staged: dict[str, _StagedFetch] = {}
        self._staged_total = 0
        self._completed_total = 0
        self._abandoned_total = 0
        self._lease_expired_total = 0

    # ------------------------------------------------------------------
    # Admission-time staging (owner thread)
    # ------------------------------------------------------------------

    def stage(self, request_id: str, destination_node: str, *, prefix_tokens: int = 0) -> None:
        """Stage one fetch for a request parked in WAITING_REMOTE_KV."""
        with self._lock:
            if request_id in self._staged:
                raise ValueError(f"request {request_id!r} is already staged")
        token_range = (0, prefix_tokens)
        chunks = (
            self._chunk_source(request_id, destination_node, token_range)
            if self._chunk_source is not None
            else ()
        )
        ticket = self._transport.submit(
            chunks,
            source_node=destination_node,
            destination_node=self._transport.node_id,
            token_range=token_range,
        )
        with self._lock:
            self._staged[request_id] = _StagedFetch(
                ticket=ticket,
                destination=destination_node,
                deadline=self._clock() + self._lease_timeout,
                prefix_tokens=prefix_tokens,
            )
            self._staged_total += 1

    def drop(self, request_id: str) -> None:
        """Forget tracking when the request went terminal while staged."""
        with self._lock:
            fetch = self._staged.pop(request_id, None)
        if fetch is not None:
            self._transport.cancel(fetch.ticket)

    # ------------------------------------------------------------------
    # Polling (owner thread, one call per loop tick)
    # ------------------------------------------------------------------

    def poll(self) -> None:
        """Resolve finished tickets, expire leases, clean dead requests."""
        now = self._clock()
        with self._lock:
            expired = [
                request_id for request_id, fetch in self._staged.items() if now >= fetch.deadline
            ]
        for request_id in expired:
            with self._lock:
                fetch = self._staged.pop(request_id, None)
            if fetch is None:
                continue
            self._transport.cancel(fetch.ticket)
            with self._lock:
                self._lease_expired_total += 1
                self._abandoned_total += 1
            self._engine.abandon_remote_kv(request_id)
        for outcome in self._transport.poll():
            with self._lock:
                match = next(
                    (
                        request_id
                        for request_id, fetch in self._staged.items()
                        if fetch.ticket.ticket_id == outcome.ticket.ticket_id
                    ),
                    None,
                )
            if match is None:
                continue  # already expired/abandoned; the ticket resolved late
            if outcome.state is TransferState.COMPLETED:
                with self._lock:
                    fetch = self._staged.pop(match)
                    self._completed_total += 1
                self._engine.release_remote_kv(match, num_cached_tokens=fetch.prefix_tokens)
            else:
                with self._lock:
                    self._staged.pop(match)
                    self._abandoned_total += 1
                self._engine.abandon_remote_kv(match)
        self._discard_terminal()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def pending_requests(self) -> frozenset[str]:
        return self._pending.pending_ids

    def snapshot(self) -> MigrationSnapshot:
        with self._lock:
            return MigrationSnapshot(
                staged_total=self._staged_total,
                completed_total=self._completed_total,
                abandoned_total=self._abandoned_total,
                lease_expired_total=self._lease_expired_total,
                pending=len(self._staged),
            )

    def _discard_terminal(self) -> None:
        """Parked requests aborted elsewhere must stop holding fetch state."""
        with self._lock:
            request_ids = tuple(self._staged)
        for request_id in request_ids:
            lifecycle = self._engine.requests.find(request_id)
            if lifecycle is not None and lifecycle.is_terminal:
                self.drop(request_id)
