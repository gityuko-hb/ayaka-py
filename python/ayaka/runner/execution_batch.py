"""Immutable host snapshot and borrowed flight lease at the worker boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ayaka.lora.variant import ExecutionVariant
from ayaka.memory.capacity import ResourceGeneration
from ayaka.sched.plan import PreparedStep
from ayaka.types import ForwardMode

if TYPE_CHECKING:
    from ayaka.executor.ticket import TicketId
    from ayaka.lora.binding import AdapterBinding
    from ayaka.worker.base import WorkerStep


@dataclass(frozen=True, slots=True)
class ExecutionBatch:
    """Borrow ``PreparedStep`` without acquiring, launching or retiring anything.

    Token/position tuples are immutable host snapshots and may be inspected
    while another flight runs. Device views stay under the existing lease.
    Successor tokens must already have been published by completion; this
    factory never reads a pending device output to manufacture a next batch.
    Empty scheduling is handled before PreparedStep (which forbids idle work).
    """

    prepared: PreparedStep
    generation: ResourceGeneration | None
    ticket_id: TicketId | None = None
    variant: ExecutionVariant | None = None
    adapters: tuple[AdapterBinding | None, ...] = ()

    def __post_init__(self) -> None:
        self.validate()

    @classmethod
    def from_prepared(
        cls, prepared: PreparedStep, *, generation: ResourceGeneration | None = None
    ) -> ExecutionBatch:
        lease = prepared.buffers
        return cls(prepared, generation, None if lease is None else lease.ticket_id)

    @classmethod
    def from_worker(cls, step: WorkerStep) -> ExecutionBatch:
        return cls(step.prepared, step.generation, step.ticket_id)

    def validate(self) -> None:
        """Validate identity/capacity again before staging or launching."""
        self.prepared.validate()
        if self.variant is None:
            if self.adapters:
                raise ValueError("adapter rows require a structural variant")
        elif len(self.adapters) != self.request_count:
            raise ValueError("adapter bindings must follow packed request order")
        if self.generation is not None and not isinstance(self.generation, ResourceGeneration):
            raise TypeError("batch generation must be a ResourceGeneration")
        lease = self.prepared.buffers
        if lease is not None:
            buffers = lease.buffers  # also rejects an already retired lease
            if lease.ticket_id != self.ticket_id:
                raise ValueError("batch and buffer lease belong to different tickets")
            if self.generation is not None and lease.generation != self.generation.buffers:
                raise ValueError("batch buffers belong to a stale generation")
            if self.request_count > buffers.spec.max_num_seqs:
                raise ValueError("batch exceeds request capacity")
            if self.prepared.step.padded_num_tokens > buffers.spec.max_num_batched_tokens:
                raise ValueError("batch exceeds token capacity")

    @property
    def request_count(self) -> int:
        return len(self.prepared.step.slices)

    @property
    def token_count(self) -> int:
        return self.prepared.step.num_tokens

    @property
    def phase(self) -> ForwardMode:
        return self.prepared.step.forward_mode

    @property
    def token_ids(self) -> tuple[int, ...]:
        return self.prepared.step.token_ids

    @property
    def positions(self) -> tuple[int, ...]:
        return self.prepared.step.positions
