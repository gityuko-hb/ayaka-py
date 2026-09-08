from __future__ import annotations

import abc
from itertools import count

from ayaka.executor.ticket import (
    CompletionFence,
    ExecutionResources,
    ExecutionTicket,
    FenceResult,
    TerminalOutcome,
    TerminalStatus,
    TicketId,
    TicketState,
    WorkState,
)
from ayaka.sched.plan import PreparedStep
from ayaka.utils.validation import require_int

_EXECUTOR_IDS = count(1)


class Executor(abc.ABC):
    """Own tickets before enqueue and preserve them through partial launch failures.

    A single engine thread serializes these methods and CompletionCoordinator.
    CancellationToken may be signalled from another thread. Backend hooks must
    not re-enter the coordinator or mutate request lifecycle. No method waits
    or globally synchronizes a device. The former execute(plan) API is removed:
    a return from enqueue is not a completed step.
    """

    def __init__(self, *, max_inflight: int = 8) -> None:
        require_int(max_inflight, "max_inflight", minimum=1)
        self._max_inflight = max_inflight
        self._executor_id = next(_EXECUTOR_IDS)
        self._ordinal = count(1)
        self._last_step = -1
        self._tickets: dict[TicketId, ExecutionTicket] = {}
        self._initialized = False
        self._stopping = False
        self._closed = False

    def initialize(self) -> None:
        """Initialize host control state. Closed/stopping executors cannot restart."""
        if self._stopping or self._closed:
            raise RuntimeError("executor is stopping or closed")
        self._initialized = True

    @property
    def tickets(self) -> tuple[ExecutionTicket, ...]:
        return tuple(self._tickets.values())

    @property
    def closed(self) -> bool:
        return self._closed

    def get_ticket(self, ticket_id: TicketId) -> ExecutionTicket | None:
        return self._tickets.get(ticket_id)

    def has_submission_capacity(self) -> bool:
        return (
            self._initialized
            and not self._stopping
            and len(self._tickets) < self._max_inflight
            and not any(t.state is TicketState.QUARANTINED for t in self._tickets.values())
        )

    def adopt(self, prepared: PreparedStep, resources: ExecutionResources) -> ExecutionTicket:
        """Transfer a prepared resource bundle before any enqueue can occur.

        Admission failure leaves ownership with the caller. Step IDs increase
        at adoption; completion ordering is independent. The registry is
        populated before resources receive the ticket identity.
        """
        if not self.has_submission_capacity():
            raise RuntimeError("executor has no submission capacity")
        if prepared.step.step_id <= self._last_step:
            raise ValueError("adoption requires a fresh increasing step_id")
        ticket = ExecutionTicket(
            TicketId(self._executor_id, next(self._ordinal)), prepared, resources
        )
        self._tickets[ticket.id] = ticket
        try:
            resources.adopt(ticket.id, prepared)
        except BaseException:
            del self._tickets[ticket.id]
            raise
        self._last_step = prepared.step.step_id
        return ticket

    def _require_owned(self, ticket: ExecutionTicket) -> None:
        if self._tickets.get(ticket.id) is not ticket:
            raise ValueError("ticket is not active in this executor")

    def track_fence(self, ticket: ExecutionTicket, fence: CompletionFence) -> None:
        """Backend hook: register every producer/consumer fence before returning."""
        self._require_owned(ticket)
        if ticket.state is not TicketState.SUBMITTED:
            raise ValueError("fences can be registered only during submission")
        if not callable(getattr(fence, "query", None)):
            raise TypeError("fence must provide a nonblocking query")
        ticket._fences.append(fence)

    def set_samples(self, ticket: ExecutionTicket, samples: tuple[int, ...]) -> None:
        """Backend hook: set packed sampling results, validated before publication."""
        self._require_owned(ticket)
        if ticket.state not in (TicketState.SUBMITTED, TicketState.DRAINING):
            raise ValueError("ticket is not awaiting execution output")
        if type(samples) is not tuple:
            raise TypeError("samples must be a tuple")
        for sample in samples:
            require_int(sample, "sample token")
        ticket._samples = samples

    def launch(self, ticket: ExecutionTicket) -> None:
        """Enqueue after adoption. Ordinary launch errors remain observable on ticket.

        Any exception during backend enqueue may hide partially submitted work,
        including work for which fence recording failed. A separate whole-ticket
        drain proof is mandatory in that case, even if all recorded fences finish.
        """
        self._require_owned(ticket)
        if ticket.state is not TicketState.ADOPTED:
            raise ValueError("ticket can be launched only once")
        if self._stopping:
            self.cancel(ticket, "executor is stopping")
            return
        try:
            ticket._resources.mark_submitted(ticket.id)
        except BaseException as exc:
            self.quarantine(ticket, f"resource submission failed: {exc}", host_failure=True)
            if not isinstance(exc, Exception):
                raise
            return
        ticket._state = TicketState.SUBMITTED
        try:
            self._enqueue(ticket)
            if not ticket._fences:
                raise RuntimeError("enqueue returned without any completion fence")
        except BaseException as exc:
            ticket._error = f"launch failed: {exc}"
            ticket._needs_drain = True
            ticket._state = TicketState.DRAINING
            if not isinstance(exc, Exception):
                raise

    @abc.abstractmethod
    def _enqueue(self, ticket: ExecutionTicket) -> None:
        """Enqueue and register all compute/transfer fences; never release a bundle."""

    @abc.abstractmethod
    def _begin_drain(self, ticket: ExecutionTicket) -> CompletionFence:
        """Return a proof covering ALL possibly submitted work, including untracked work."""

    def cancel(self, ticket: ExecutionTicket, reason: str = "cancelled") -> None:
        """Cancel publication; submitted work must still finish or drain."""
        self._require_owned(ticket)
        if ticket._settling or ticket.state is TicketState.RETIRED:
            return
        if ticket.state is TicketState.COMPLETED:
            if ticket.terminal is not None and ticket.terminal.status is TerminalStatus.SUCCEEDED:
                ticket._cancelled = True
                ticket._error = reason
                self._finish(ticket, TerminalStatus.CANCELLED)
            return
        ticket._cancelled = True
        if not ticket._error:
            ticket._error = reason
        if ticket.state is TicketState.ADOPTED:
            self._finish(ticket, TerminalStatus.CANCELLED)
        elif ticket.state is not TicketState.QUARANTINED:
            ticket._state = TicketState.DRAINING

    def quarantine(
        self, ticket: ExecutionTicket, error: str, *, host_failure: bool = False
    ) -> None:
        """Retain ownership and stop admission; errors never imply quiescence."""
        self._require_owned(ticket)
        ticket._error = error
        ticket._state = TicketState.QUARANTINED
        ticket._host_failure |= host_failure
        ticket._needs_drain = not ticket._host_failure

    def drain(self, ticket: ExecutionTicket) -> None:
        """Start a nonblocking recovery proof if completion became uncertain.

        Host commit/retire errors require explicit owner recovery outside P1;
        a device fence cannot repair ambiguous host accounting.
        """
        self._require_owned(ticket)
        if ticket._host_failure or not ticket._needs_drain or ticket._drain_fence is not None:
            return
        try:
            ticket._drain_fence = self._begin_drain(ticket)
        except BaseException as exc:
            self.quarantine(ticket, f"cannot establish drain proof: {exc}")
            if not isinstance(exc, Exception):
                raise

    def _finish(self, ticket: ExecutionTicket, status: TerminalStatus) -> None:
        samples = ticket._samples if status is TerminalStatus.SUCCEEDED else ()
        ticket._terminal = TerminalOutcome(ticket.id, status, samples, ticket._error)
        ticket._state = TicketState.COMPLETED

    def _poll(self, ticket: ExecutionTicket) -> None:
        if ticket._host_failure or ticket.state in (TicketState.ADOPTED, TicketState.COMPLETED):
            return
        if ticket._needs_drain:
            self.drain(ticket)
            fences = () if ticket._drain_fence is None else (ticket._drain_fence,)
        else:
            fences = tuple(ticket._fences)
        if not fences:
            return
        all_quiescent = True
        failed = False
        for fence in fences:
            try:
                result = fence.query()
                if not isinstance(result, FenceResult):
                    raise TypeError("fence query must return FenceResult")
            except BaseException as exc:
                self.quarantine(ticket, f"completion query failed: {exc}")
                if not isinstance(exc, Exception):
                    raise
                return
            if result.state is WorkState.FAILED:
                failed = True
                ticket._execution_failed = True
                ticket._error = ticket._error or result.error
                if not result.quiescent:
                    self.quarantine(ticket, result.error)
                    return
            all_quiescent &= result.quiescent
        if not all_quiescent:
            return
        if ticket._needs_drain or failed or ticket._execution_failed:
            self._finish(ticket, TerminalStatus.FAILED)
        elif ticket._cancelled:
            self._finish(ticket, TerminalStatus.CANCELLED)
        elif len(ticket._samples) != len(ticket.prepared.step.sampling_rows):
            ticket._error = "sampling output count disagrees with prepared rows"
            self._finish(ticket, TerminalStatus.FAILED)
        else:
            self._finish(ticket, TerminalStatus.SUCCEEDED)

    def poll_terminal(self) -> tuple[TerminalOutcome, ...]:
        """Return unretired terminal outcomes; redelivery is intentional until ack."""
        result = []
        for ticket in self.tickets:
            self._poll(ticket)
            if ticket.state is TicketState.COMPLETED and ticket.terminal is not None:
                result.append(ticket.terminal)
        return tuple(result)

    def acknowledge_retired(self, ticket: ExecutionTicket) -> None:
        """Coordinator-only acknowledgement, after successful resource retirement."""
        self._require_owned(ticket)
        if ticket.state is not TicketState.COMPLETED or not ticket._retirement_complete:
            raise ValueError("only settled terminal work can retire")
        ticket._state = TicketState.RETIRED
        del self._tickets[ticket.id]

    def shutdown(self) -> bool:
        """Stop admission, drain, and close only after coordinator retires all tickets.

        Returns False while pending/quarantined/unprocessed work remains. Call
        CompletionCoordinator.shutdown to both settle outcomes and retry closure.
        No pending resource is freed by this method.
        """
        self._stopping = True
        for ticket in self.tickets:
            if ticket.state is TicketState.ADOPTED:
                self.cancel(ticket, "shutdown before launch")
            else:
                self.drain(ticket)
        if self._tickets:
            return False
        if not self._closed:
            self._close()
            self._closed = True
        return True

    @abc.abstractmethod
    def _close(self) -> None:
        """Backend teardown hook, reached only when no ticket owns resources."""
