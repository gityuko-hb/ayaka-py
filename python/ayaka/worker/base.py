"""Step and worker contracts between the executor and a device owner.

``WorkerStep`` is the immutable execution view handed to a worker: step/plan
identity, the prepared KV view, the captured resource generation and the
workspace-growth signal. It carries no mutable ``RequestLifecycle`` and no
stop-policy state, so the worker never reaches back into the scheduler. Lease
ownership stays with the ticket; passing a step does not duplicate it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

from ayaka.executor.ticket import (
    CompletionFence,
    ExecutionTicket,
    SampleOutputs,
    TicketId,
)
from ayaka.memory.capacity import ResourceGeneration
from ayaka.sched.plan import PreparedStep
from ayaka.utils.validation import require_frozen, require_int
from ayaka.worker.lifecycle import WorkerState

__all__ = ["StepWorker", "WorkerOutcome", "WorkerStep"]


@dataclass(frozen=True, slots=True)
class WorkerStep:
    """Immutable execution view for one worker submission."""

    ticket_id: TicketId
    step_id: int
    generation: ResourceGeneration | None
    prepared: PreparedStep
    workspace_grew: bool = False

    def __post_init__(self) -> None:
        require_frozen(self.ticket_id, "worker step.ticket_id")
        require_int(self.step_id, "worker step.step_id", minimum=0)
        if self.generation is not None and not isinstance(self.generation, ResourceGeneration):
            raise TypeError("worker step generation must be a ResourceGeneration or None")
        if not isinstance(self.prepared, PreparedStep):
            raise TypeError("worker step must carry a PreparedStep")
        if self.prepared.step.step_id != self.step_id:
            raise ValueError("worker step identity disagrees with the prepared plan")
        if type(self.workspace_grew) is not bool:
            raise TypeError("workspace_grew must be a boolean")

    @classmethod
    def from_ticket(cls, ticket: ExecutionTicket) -> WorkerStep:
        """Build the immutable view; the ticket keeps every lease and mutation right."""
        resources = ticket._resources
        return cls(
            ticket_id=ticket.id,
            step_id=ticket.prepared.step.step_id,
            generation=getattr(resources, "generation", None),
            prepared=ticket.prepared,
            workspace_grew=bool(getattr(resources, "workspace_grew", False)),
        )


@dataclass(frozen=True, slots=True)
class WorkerOutcome:
    """Samples plus the fence that covers every possibly submitted consumer."""

    samples: SampleOutputs
    fence: CompletionFence

    def __post_init__(self) -> None:
        if not isinstance(self.samples, SampleOutputs):
            raise TypeError("worker outcome samples must be SampleOutputs")
        if not callable(getattr(self.fence, "query", None)):
            raise TypeError("worker outcome fence must provide a nonblocking query")


@runtime_checkable
class StepWorker(Protocol):
    """Device owner: context, streams, runner, fences, drain and shutdown.

    Callers (the executor adapter) may only use this surface. Implementations
    reject work before enqueue whenever they are not accepting, so a FAILED or
    closing worker can never receive a partially prepared ticket.
    """

    @property
    def device(self) -> torch.device: ...

    @property
    def state(self) -> WorkerState: ...

    @property
    def accepting(self) -> bool: ...

    @property
    def incarnation(self) -> int: ...

    def initialize(self) -> None: ...

    def execute(self, step: WorkerStep) -> WorkerOutcome: ...

    def drain(self, step: WorkerStep) -> CompletionFence: ...

    def request_closing(self) -> WorkerState: ...

    def shutdown(self) -> bool: ...

    def close(self) -> None: ...
