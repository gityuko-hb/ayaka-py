"""Serialized, exactly-once logical completion and resource retirement for P1."""

from __future__ import annotations

from dataclasses import dataclass

from ayaka.executor.base import Executor
from ayaka.executor.ticket import (
    ExecutionResources,
    ExecutionTicket,
    TerminalOutcome,
    TerminalStatus,
    TicketId,
    TicketState,
)
from ayaka.request.lifecycle import LifecycleManager, RequestLifecycle
from ayaka.sched.plan import PreparedStep, ScheduledSlice
from ayaka.utils.validation import require_int


@dataclass(frozen=True, slots=True)
class PublishedSample:
    request_id: str
    sequence_epoch: int
    token_id: int


@dataclass(frozen=True, slots=True)
class CompletionResult:
    ticket_id: TicketId
    status: TerminalStatus
    published: tuple[PublishedSample, ...] = ()
    ignored_requests: tuple[str, ...] = ()
    error: str = ""


@dataclass(frozen=True, slots=True)
class ShutdownResult:
    closed: bool
    pending: tuple[TicketId, ...]
    completed: tuple[CompletionResult, ...]


class CompletionCoordinator:
    """Own request bindings through completion, independent of global report order.

    All calls and request state transitions run on one engine thread; only
    cancellation signalling may happen concurrently. Original lifecycle objects
    are retained until retirement, so reaping/reusing a request ID cannot redirect
    an old completion to its replacement. Resource callback failures quarantine
    the ticket; no finally block frees potentially live or ambiguously owned data.
    """

    def __init__(self, executor: Executor, requests: LifecycleManager) -> None:
        self.executor = executor
        self.requests = requests
        self._bindings: dict[TicketId, tuple[tuple[RequestLifecycle, ScheduledSlice], ...]] = {}

    def adopt(
        self,
        prepared: PreparedStep,
        resources: ExecutionResources,
        *,
        now_ns: int | None = None,
    ) -> ExecutionTicket:
        """Validate all requests, transfer ownership, then claim their P0 slices.

        Preflight failures leave resources with the caller. If request claiming
        races cancellation after transfer, skip that request and preserve valid
        peers. When none remain, return an owned cancelled ticket for retirement.
        """
        bindings = tuple(
            (self.requests.get(scheduled.request_id), scheduled)
            for scheduled in prepared.step.slices
        )
        for (request, scheduled), snapshot in zip(bindings, prepared.step.inputs, strict=True):
            actual = request.validate_begin_slice(prepared.step.step_id, scheduled)
            if actual != snapshot:
                raise ValueError("prepared input snapshot is stale")
        ticket = self.executor.adopt(prepared, resources)
        self._bindings[ticket.id] = bindings
        try:
            for request, scheduled in bindings:
                if request.token.is_cancelled:
                    continue
                try:
                    request.begin_slice(prepared.step.step_id, scheduled, now_ns=now_ns)
                except ValueError:
                    if not request.token.is_cancelled:
                        raise
            if not any(
                request.accepts_completion(prepared.step.step_id, scheduled)
                for request, scheduled in bindings
            ):
                self.executor.cancel(ticket, "all requests cancelled during adoption")
        except BaseException as exc:
            self.executor.cancel(ticket, f"request adoption failed: {exc}")
            if not isinstance(exc, Exception):
                raise
        return ticket

    def launch(self, ticket: ExecutionTicket) -> None:
        """Avoid enqueue when every captured request was cancelled or invalidated."""
        bindings = self._bindings.get(ticket.id)
        if bindings is None:
            raise ValueError("ticket was not adopted by this coordinator")
        if ticket.state is not TicketState.ADOPTED:
            if ticket.terminal is not None and ticket.terminal.status is TerminalStatus.CANCELLED:
                return
            raise ValueError("ticket can be launched only once")
        if not any(
            request.accepts_completion(ticket.prepared.step.step_id, scheduled)
            for request, scheduled in bindings
        ):
            self.executor.cancel(ticket, "all requests invalidated before launch")
            return
        self.executor.launch(ticket)

    def _discard(
        self,
        request: RequestLifecycle,
        scheduled: ScheduledSlice,
        ticket: ExecutionTicket,
        *,
        cancel: bool,
        fail: bool,
        now_ns: int | None,
    ) -> None:
        current = request.sequence_epoch == scheduled.sequence_epoch and not request.is_terminal
        if request.owns_slice(ticket.prepared.step.step_id, scheduled):
            request.discard_slice(ticket.prepared.step.step_id, scheduled)
        if current:
            if request.token.is_cancelled or cancel:
                request.token.cancel(ticket.error or "cancelled")
                request.machine.on_cancelled(now_ns=now_ns)
            elif fail:
                request.machine.on_failed(ticket.error or "execution failed", now_ns=now_ns)
        self.requests.retire_terminal(request)

    def handle(
        self,
        outcome: TerminalOutcome,
        *,
        now_ns: int | None = None,
    ) -> CompletionResult | None:
        """Settle an executor-authored result once; duplicate/stale delivery is a no-op."""
        ticket = self.executor.get_ticket(outcome.ticket_id)
        if ticket is None:
            return None
        if ticket.state is not TicketState.COMPLETED or ticket.terminal != outcome:
            return None
        if ticket._settling:
            return None
        bindings = self._bindings.get(ticket.id)
        if bindings is None:
            self.executor.quarantine(ticket, "missing request bindings", host_failure=True)
            return None
        ticket._settling = True
        published: list[PublishedSample] = []
        ignored: list[str] = []
        step_id = ticket.prepared.step.step_id
        try:
            if outcome.status is TerminalStatus.SUCCEEDED:
                # Validate before committing physical state. A changed incarnation
                # is simply ignored; an unexpected mismatch in a live one is a fault.
                for request, scheduled in bindings:
                    if request.accepts_completion(step_id, scheduled):
                        try:
                            request.validate_forward_commit(
                                step_id,
                                scheduled,
                                committed_state_version=scheduled.expected_state_version + 1,
                            )
                        except ValueError:
                            if not (
                                request.token.is_cancelled
                                or request.is_terminal
                                or request.sequence_epoch != scheduled.sequence_epoch
                            ):
                                raise
                versions = ticket._resources.commit(ticket.id)
                expected = tuple(s.expected_state_version + 1 for _, s in bindings)
                if type(versions) is not tuple:
                    raise TypeError("committed versions must be an immutable tuple")
                for version in versions:
                    require_int(version, "committed KV version")
                if versions != expected:
                    raise ValueError("resource commit returned unexpected KV versions")
                sample_index = 0
                for (request, scheduled), version in zip(bindings, versions, strict=True):
                    sample = outcome.samples[sample_index] if scheduled.sample_last_query else None
                    sample_index += int(scheduled.sample_last_query)
                    if not request.accepts_completion(step_id, scheduled):
                        ignored.append(request.request_id)
                        self._discard(
                            request, scheduled, ticket, cancel=False, fail=False, now_ns=now_ns
                        )
                        continue
                    try:
                        request.commit_computed_range(
                            step_id,
                            scheduled,
                            committed_state_version=version,
                            now_ns=now_ns,
                        )
                        # Cancellation may arrive between the predicate and the
                        # P0 method's own guard; treat it as cancellation, not a
                        # resource-accounting fault.
                        if sample is not None and request.accepts_completion(step_id, scheduled):
                            request.publish_sample(step_id, scheduled, sample)
                            published.append(
                                PublishedSample(
                                    request.request_id,
                                    scheduled.sequence_epoch,
                                    sample,
                                )
                            )
                    except ValueError:
                        if not (
                            request.token.is_cancelled
                            or request.is_terminal
                            or request.sequence_epoch != scheduled.sequence_epoch
                        ):
                            raise
                        self._discard(
                            request, scheduled, ticket, cancel=False, fail=False, now_ns=now_ns
                        )
                        ignored.append(request.request_id)
                        continue
                    if request.token.is_cancelled:
                        self._discard(
                            request, scheduled, ticket, cancel=True, fail=False, now_ns=now_ns
                        )
                        ignored.append(request.request_id)
            else:
                for request, scheduled in bindings:
                    self._discard(
                        request,
                        scheduled,
                        ticket,
                        cancel=outcome.status is TerminalStatus.CANCELLED,
                        fail=outcome.status is TerminalStatus.FAILED,
                        now_ns=now_ns,
                    )
                    ignored.append(request.request_id)
            ticket._resources.retire(
                ticket.id,
                succeeded=outcome.status is TerminalStatus.SUCCEEDED,
                discarded_sequences=tuple(
                    value.sequence
                    for value in ticket.prepared.step.inputs
                    if value.request_id in ignored
                ),
            )
            ticket._retirement_complete = True
            self.executor.acknowledge_retired(ticket)
            del self._bindings[ticket.id]
        except BaseException as exc:
            self.executor.quarantine(
                ticket, f"completion settlement failed: {exc}", host_failure=True
            )
            if not isinstance(exc, Exception):
                raise
            return None
        return CompletionResult(
            ticket.id, outcome.status, tuple(published), tuple(ignored), outcome.error
        )

    def poll(self, *, now_ns: int | None = None) -> tuple[CompletionResult, ...]:
        """Poll without blocking; independent requests may finish out of step order."""
        results = []
        for outcome in self.executor.poll_terminal():
            result = self.handle(outcome, now_ns=now_ns)
            if result is not None:
                results.append(result)
        return tuple(results)

    def shutdown(self, *, now_ns: int | None = None) -> ShutdownResult:
        """One nonblocking drain/settle/close pass; retry when completion changes."""
        self.executor.shutdown()
        results = self.poll(now_ns=now_ns)
        closed = self.executor.shutdown()
        return ShutdownResult(closed, tuple(t.id for t in self.executor.tickets), results)
