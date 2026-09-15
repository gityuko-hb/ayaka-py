"""Ownership and completion contracts for asynchronous execution.

Ownership/lease contracts are torch-free; ``SampleOutputs`` is the one runtime
payload and stays device-resident until the completion boundary materializes
it (M0: no ``.tolist()`` inside the sampler or coordinator).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

import torch

from ayaka.handles import SequenceHandle
from ayaka.sampling.logprobs import (
    PromptLogprobSliceReport,
    PromptLogprobTensors,
    SampleLogprobTensors,
)
from ayaka.sched.plan import PreparedStep
from ayaka.utils.validation import require_frozen, require_int


@dataclass(frozen=True, slots=True)
class TicketId:
    """Process-local executor incarnation plus monotonically increasing ordinal."""

    executor: int
    ordinal: int

    def __post_init__(self) -> None:
        require_int(self.executor, "executor", minimum=1)
        require_int(self.ordinal, "ordinal", minimum=1)


class TicketState(StrEnum):
    ADOPTED = "adopted"
    SUBMITTED = "submitted"
    DRAINING = "draining"
    QUARANTINED = "quarantined"
    COMPLETED = "completed"
    RETIRED = "retired"


class WorkState(StrEnum):
    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class FenceResult:
    """Failure and quiescence are independent; a query error proves neither."""

    state: WorkState
    quiescent: bool = False
    error: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.state, WorkState) or type(self.quiescent) is not bool:
            raise TypeError("invalid fence state/quiescence")
        if self.state is WorkState.PENDING and self.quiescent:
            raise ValueError("pending work cannot prove quiescence")
        if self.state is WorkState.SUCCEEDED and not self.quiescent:
            raise ValueError("success must cover every use represented by the fence")
        if self.state is WorkState.FAILED and not self.error:
            raise ValueError("a failed fence requires an error")


class CompletionFence(Protocol):
    """Nonblocking query. Exceptions imply unknown completion, never safe release."""

    def query(self) -> FenceResult: ...


class ExecutionResources(Protocol):
    """One exclusive bundle retained by a ticket until all consumers stop.

    adopt is atomic: failure leaves ownership with the caller. After adoption
    the caller must never roll back/free this bundle. mark_submitted runs before
    any backend enqueue. commit publishes KV only, returning sequence versions
    in packed order; it must not retire execution leases. retire releases last-use
    leases after logical publication/discard and is idempotent by ticket identity.
    discarded_sequences contains original generation-safe handles from this step;
    real resources release those requests behind their active KV lease.

    P1 supplies only a fake implementation. Physical KV/workspace/staging
    accounting and real allocator validation are implemented in P2. Exceptions
    after adoption quarantine the bundle; callbacks are not blindly retried.
    """

    def adopt(self, ticket_id: TicketId, prepared: PreparedStep) -> None: ...
    def mark_submitted(self, ticket_id: TicketId) -> None: ...
    def commit(self, ticket_id: TicketId) -> tuple[int, ...]: ...
    def retire(
        self,
        ticket_id: TicketId,
        *,
        succeeded: bool,
        discarded_sequences: tuple[SequenceHandle, ...] = (),
    ) -> None: ...


class TerminalStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class SampleOutputs:
    """Packed sampling results for one ticket, device-resident until publication.

    Rows follow packed sampling order (``BatchStepPlan.sampling_rows``). The
    completion boundary is the only place that materializes host/Python values;
    keeping the tensor alive is the caller's job until publication.

    ``logprobs`` (when present) reports raw logprobs for a subset of rows:
    ``logprob_rows[j]`` is the sampling-row index of report row ``j`` and must
    be unique and in range. Row count must agree with the tensor shapes.
    """

    token_ids: torch.Tensor
    logprobs: SampleLogprobTensors | None = None
    logprob_rows: tuple[int, ...] = ()
    prompt_logprobs: PromptLogprobTensors | None = None
    prompt_logprob_slices: tuple[PromptLogprobSliceReport, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.token_ids, torch.Tensor):
            raise TypeError("token_ids must be a torch.Tensor")
        if self.token_ids.dim() != 1:
            raise ValueError(f"token_ids must be 1-D, got {tuple(self.token_ids.shape)}")
        if self.token_ids.is_floating_point() or self.token_ids.is_complex():
            raise ValueError("token_ids must be integer dtype")
        if self.logprobs is None:
            if self.logprob_rows:
                raise ValueError("logprob_rows without logprobs payload")
        else:
            self._validate_generation_logprobs()
        if self.prompt_logprobs is None:
            if self.prompt_logprob_slices:
                raise ValueError("prompt_logprob_slices without a prompt_logprobs payload")
        else:
            self._validate_prompt_logprobs()

    def _validate_generation_logprobs(self) -> None:
        lp = self.logprobs
        assert lp is not None
        rows = self.logprob_rows
        if type(rows) is not tuple:
            raise TypeError("logprob_rows must be a tuple")
        if lp.token_logprob.dim() != 1 or lp.token_logprob.size(0) != len(rows):
            raise ValueError(
                f"token_logprob must be [{len(rows)}], got {tuple(lp.token_logprob.shape)}"
            )
        if lp.top_token_ids.dim() != 2 or lp.top_token_ids.size(0) != len(rows):
            raise ValueError(
                f"top_token_ids must be [{len(rows)}, K], got {tuple(lp.top_token_ids.shape)}"
            )
        if lp.top_logprobs.shape != lp.top_token_ids.shape:
            raise ValueError("top_logprobs shape must match top_token_ids")
        if not lp.token_logprob.is_floating_point() or not lp.top_logprobs.is_floating_point():
            raise ValueError("logprob tensors must be floating-point")
        if lp.top_token_ids.is_floating_point() or lp.top_token_ids.is_complex():
            raise ValueError("top_token_ids must be integer dtype")
        if len(set(rows)) != len(rows):
            raise ValueError("logprob_rows must be unique")
        for row in rows:
            require_int(row, "logprob sampling row")
            if not 0 <= row < self.token_ids.size(0):
                raise IndexError(
                    f"logprob row {row} outside the {self.token_ids.size(0)} sampling rows"
                )

    def _validate_prompt_logprobs(self) -> None:
        lp = self.prompt_logprobs
        assert lp is not None
        slices = self.prompt_logprob_slices
        if type(slices) is not tuple:
            raise TypeError("prompt_logprob_slices must be a tuple")
        scored = sum(len(report.scored_positions) for report in slices)
        if lp.token_logprob.dim() != 1 or lp.token_logprob.size(0) != scored:
            raise ValueError(
                f"prompt token_logprob must be [{scored}], got {tuple(lp.token_logprob.shape)}"
            )
        if lp.top_token_ids.dim() != 2 or lp.top_token_ids.size(0) != scored:
            raise ValueError(
                f"prompt top_token_ids must be [{scored}, K], got {tuple(lp.top_token_ids.shape)}"
            )
        if lp.top_logprobs.shape != lp.top_token_ids.shape:
            raise ValueError("prompt top_logprobs shape must match top_token_ids")
        if not lp.token_logprob.is_floating_point() or not lp.top_logprobs.is_floating_point():
            raise ValueError("prompt logprob tensors must be floating-point")
        if lp.top_token_ids.is_floating_point() or lp.top_token_ids.is_complex():
            raise ValueError("prompt top_token_ids must be integer dtype")
        previous = -1
        for report in slices:
            if not isinstance(report, PromptLogprobSliceReport):
                raise TypeError("prompt_logprob_slices must contain PromptLogprobSliceReport")
            if report.slice_index <= previous:
                raise ValueError("prompt_logprob_slices must be ordered by slice_index")
            previous = report.slice_index


@dataclass(frozen=True, slots=True)
class TerminalOutcome:
    """Executor-authored, quiescent result; not itself permission to free pages."""

    ticket_id: TicketId
    status: TerminalStatus
    samples: SampleOutputs | None = None
    error: str = ""

    def __post_init__(self) -> None:
        require_frozen(self.ticket_id, "terminal outcome.ticket_id")
        if not isinstance(self.status, TerminalStatus):
            raise TypeError("status must be TerminalStatus")
        if not isinstance(self.error, str):
            raise TypeError("error must be a string")
        if self.samples is not None and not isinstance(self.samples, SampleOutputs):
            raise TypeError("samples must be SampleOutputs")
        if self.status is not TerminalStatus.SUCCEEDED and self.samples is not None:
            raise ValueError("failed/cancelled work cannot publish samples")


@dataclass(slots=True)
class ExecutionTicket:
    """Live runtime owner, unlike the immutable PreparedStep it retains.

    Underscored fields are mutated only by the executor/completion coordinator.
    Retired tickets may remain in diagnostics; the active executor registry
    retains every non-retired ticket, including drain and quarantine failures.
    """

    id: TicketId
    prepared: PreparedStep
    _resources: ExecutionResources = field(repr=False)
    _state: TicketState = TicketState.ADOPTED
    _fences: list[CompletionFence] = field(default_factory=list, repr=False)
    _drain_fence: CompletionFence | None = field(default=None, repr=False)
    _needs_drain: bool = False
    _host_failure: bool = False
    _cancelled: bool = False
    _samples: SampleOutputs | None = None
    _error: str = ""
    _terminal: TerminalOutcome | None = None
    _settling: bool = False
    _retirement_complete: bool = False
    _execution_failed: bool = False
    _adopted_ns: int = 0
    _submitted_ns: int | None = None
    _drain_started_ns: int | None = None
    _terminal_ns: int | None = None

    @property
    def state(self) -> TicketState:
        return self._state

    @property
    def error(self) -> str:
        return self._error

    @property
    def terminal(self) -> TerminalOutcome | None:
        return self._terminal
