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
from ayaka.obs import runtime_event
from ayaka.request.lifecycle import LifecycleManager, RequestLifecycle
from ayaka.sampling.logprobs import LogprobResult, TokenLogprob
from ayaka.sampling.ops.sampling import SamplingSupportStatus
from ayaka.sched.plan import PreparedStep, ScheduledSlice
from ayaka.utils.validation import require_int


@dataclass(frozen=True, slots=True)
class PublishedSample:
    request_id: str
    sequence_epoch: int
    token_id: int
    logprobs: LogprobResult | None = None
    sampling_support: SamplingSupportReport | None = None


@dataclass(frozen=True, slots=True)
class SamplingSupportReport:
    """Host materialization của một captured support row.

    ``token_ids`` là support đã pack (top-k theo weight, capped theo
    ``sampling_support_max_tokens``); ``length`` là số token support thực tế
    SAU clamp. ``selected_logprob`` là log(w/mass) của token đã sample —
    semantics ``LogprobMode.SAMPLING``.
    """

    status: SamplingSupportStatus
    length: int
    selected_logprob: float
    token_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PromptLogprobChunk:
    """One slice's prompt-logprob report, keyed by request incarnation.

    Covers absolute prompt positions ``[start, end)``: every position appears
    once; ``None`` marks no-value positions (first prompt token, positions
    whose predecessor lies in the prefix-cached region). Positions are never
    re-forwarded just for reporting, so a cached prefix legitimately produces
    None — a chunk boundary never does.
    """

    request_id: str
    sequence_epoch: int
    start: int
    end: int
    token_ids: tuple[int, ...]
    logprobs: tuple[float | None, ...]
    top_logprobs: tuple[tuple[TokenLogprob, ...] | None, ...]


def _materialize_reports(samples) -> dict[int, LogprobResult]:
    """Materialize every logprob report row in ONE host sync; {} when disabled.

    Keyed by sampling-row index. Padding entries (token id -1) end the row's
    top list; the selected token's logprob is identical wherever it appears.
    """
    lp = samples.logprobs
    if lp is None:
        return {}
    token_values = [float(value) for value in lp.token_logprob.tolist()]
    top_ids = lp.top_token_ids.tolist()
    top_values = lp.top_logprobs.tolist()
    by_row: dict[int, LogprobResult] = {}
    for index, row in enumerate(samples.logprob_rows):
        top = tuple(
            TokenLogprob(int(token), float(value))
            for token, value in zip(top_ids[index], top_values[index], strict=True)
            if token >= 0
        )
        by_row[int(row)] = LogprobResult(token_values[index], top)
    return by_row


def _materialize_support(samples) -> dict[int, SamplingSupportReport]:
    """Materialize every support-capture row in ONE host sync; {} when off.

    Keyed by sampling-row index (packed sampling order). Range validation of
    ``row_indices`` lives HERE — SampleOutputs validation is shape-only to
    stay sync-free. ``token_ids`` beyond ``length`` là padding zero-weight.
    """
    support = samples.sampling_support
    if support is None:
        return {}
    rows = support.row_indices.tolist()
    statuses = support.statuses.tolist()
    lengths = support.lengths.tolist()
    selected = support.selected_logprobs.tolist()
    packed_ids = support.token_ids.tolist()
    limit = samples.token_ids.size(0)
    by_row: dict[int, SamplingSupportReport] = {}
    for index, row in enumerate(rows):
        row = int(row)
        if not 0 <= row < limit:
            raise IndexError(f"sampling support row {row} outside the {limit} sampling rows")
        length = int(lengths[index])
        by_row[row] = SamplingSupportReport(
            status=SamplingSupportStatus(int(statuses[index])),
            length=length,
            selected_logprob=float(selected[index]),
            token_ids=tuple(int(token) for token in packed_ids[index][:length]),
        )
    return by_row
    lp = samples.logprobs
    if lp is None:
        return {}
    token_values = [float(value) for value in lp.token_logprob.tolist()]
    top_ids = lp.top_token_ids.tolist()
    top_values = lp.top_logprobs.tolist()
    by_row: dict[int, LogprobResult] = {}
    for index, row in enumerate(samples.logprob_rows):
        top = tuple(
            TokenLogprob(int(token), float(value))
            for token, value in zip(top_ids[index], top_values[index], strict=True)
            if token >= 0
        )
        by_row[int(row)] = LogprobResult(token_values[index], top)
    return by_row


def _materialize_ids_logprobs(samples) -> dict[int, tuple[tuple[float, ...], float]]:
    """Materialize token_ids_logprobs values in ONE host sync; {} when off.

    Keyed by sampling-row index → (values theo count, token_logprob của
    token sample). Token id của từng value nằm ở ``request.sampling.
    token_ids_logprobs`` (host) — merge tại publish.
    """
    payload = samples.token_ids_logprobs
    if payload is None:
        return {}
    values = payload.logprobs.tolist()
    selected = payload.token_logprob.tolist()
    limit = samples.token_ids.size(0)
    by_row: dict[int, tuple[tuple[float, ...], float]] = {}
    for index, row in enumerate(samples.ids_logprob_rows):
        row = int(row)
        if not 0 <= row < limit:
            raise IndexError(f"ids logprob row {row} outside the {limit} sampling rows")
        count = int(samples.ids_logprob_counts[index])
        by_row[row] = (
            tuple(float(value) for value in values[index][:count]),
            float(selected[index]),
        )
    return by_row


def _merge_ids_logprobs(
    lp: LogprobResult | None,
    ids_values: tuple[tuple[float, ...], float] | None,
    token_ids: tuple[int, ...] | None,
) -> LogprobResult | None:
    """Ghép ids-logprob vào LogprobResult của row (tạo mới khi row chỉ ids)."""
    if ids_values is None:
        return lp
    values, token_logprob = ids_values
    ids_result = tuple(
        TokenLogprob(int(token), float(value))
        for token, value in zip(token_ids or (), values, strict=True)
    )
    if lp is None:
        return LogprobResult(token_logprob, (), ids_result)
    return LogprobResult(lp.token_logprob, lp.top_logprobs, ids_result)


def _materialize_prompt_chunks(
    samples, bindings: tuple[tuple[RequestLifecycle, ScheduledSlice], ...]
) -> tuple[PromptLogprobChunk, ...]:
    """Materialize every prompt-logprob chunk in ONE host sync; () when disabled.

    Chunk coverage ``[start, end)`` comes from the runner's descriptors; the
    prompt token ids come from the step's input snapshots. Scored positions
    are looked up in the packed device tensors; every other position in the
    range is a no-value marker.
    """
    lp = samples.prompt_logprobs
    if lp is None:
        return ()
    token_values = [float(value) for value in lp.token_logprob.tolist()]
    top_ids = lp.top_token_ids.tolist()
    top_values = lp.top_logprobs.tolist()
    chunks: list[PromptLogprobChunk] = []
    row = 0
    for report in samples.prompt_logprob_slices:
        request, _scheduled = bindings[report.slice_index]
        by_position = {position: index for index, position in enumerate(report.scored_positions)}
        logprobs: list[float | None] = []
        tops: list[tuple[TokenLogprob, ...] | None] = []
        token_ids: list[int] = []
        for position in range(report.start, report.end):
            token_ids.append(int(request.request.prompt_token_ids[position]))
            index = by_position.get(position)
            if index is None:
                logprobs.append(None)
                tops.append(None)
                continue
            global_row = row + index
            top = tuple(
                TokenLogprob(int(token), float(value))
                for token, value in zip(top_ids[global_row], top_values[global_row], strict=True)
                if token >= 0
            )
            logprobs.append(token_values[global_row])
            tops.append(top)
        row += len(report.scored_positions)
        chunks.append(
            PromptLogprobChunk(
                request_id=request.request_id,
                sequence_epoch=_scheduled.sequence_epoch,
                start=report.start,
                end=report.end,
                token_ids=tuple(token_ids),
                logprobs=tuple(logprobs),
                top_logprobs=tuple(tops),
            )
        )
    return tuple(chunks)


@dataclass(frozen=True, slots=True)
class CompletionResult:
    ticket_id: TicketId
    status: TerminalStatus
    published: tuple[PublishedSample, ...] = ()
    ignored_requests: tuple[str, ...] = ()
    error: str = ""
    prompt_logprobs: tuple[PromptLogprobChunk, ...] = ()


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

    def _trace(
        self,
        event: str,
        ticket: ExecutionTicket,
        *,
        status: str | None = None,
        detail: str | None = None,
    ) -> None:
        worker = getattr(self.executor, "worker", None)
        for value in ticket.prepared.step.inputs:
            runtime_event(
                event,
                request_id=value.request_id,
                sequence_epoch=value.sequence_epoch,
                sequence=value.sequence,
                step_id=ticket.prepared.step.step_id,
                ticket_id=ticket.id,
                worker_incarnation=getattr(worker, "incarnation", None),
                resource_generation=getattr(ticket._resources, "generation", None),
                status=status,
                detail=detail,
            )

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
        self._trace("adopt", ticket)
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
        self._trace("submit", ticket, status=ticket.state.value)

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
        self._trace("complete", ticket, status=outcome.status.value, detail=outcome.error)
        published: list[PublishedSample] = []
        ignored: list[str] = []
        prompt_logprobs: list[PromptLogprobChunk] = []
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
                self._trace("commit", ticket, status=outcome.status.value)
                # Publication boundary: the single host materialization of the
                # step's device-resident sampling results. Everything upstream
                # (sampler, coordinator, ticket) keeps tensors; everything
                # downstream (publish, reporting) reads host values only.
                samples = outcome.samples
                if samples is None:
                    raise RuntimeError(
                        "succeeded outcome without sampling results; the executor "
                        "must validate samples before completing a ticket"
                    )
                tokens = tuple(int(t) for t in samples.token_ids.tolist())
                report_by_row = _materialize_reports(samples)
                support_by_row = _materialize_support(samples)
                ids_by_row = _materialize_ids_logprobs(samples)
                chunks_by_slice = {
                    report.slice_index: chunk
                    for report, chunk in zip(
                        samples.prompt_logprob_slices,
                        _materialize_prompt_chunks(samples, bindings),
                        strict=True,
                    )
                }
                prompt_logprobs.clear()
                sample_index = 0
                for slice_index, ((request, scheduled), version) in enumerate(
                    zip(bindings, versions, strict=True)
                ):
                    sample = tokens[sample_index] if scheduled.sample_last_query else None
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
                            runtime_event(
                                "publish",
                                request_id=request.request_id,
                                sequence_epoch=scheduled.sequence_epoch,
                                sequence=ticket.prepared.step.inputs[slice_index].sequence,
                                step_id=step_id,
                                ticket_id=ticket.id,
                                status=outcome.status.value,
                            )
                            merged = _merge_ids_logprobs(
                                report_by_row.get(sample_index - 1),
                                ids_by_row.get(sample_index - 1),
                                request.request.sampling.token_ids_logprobs,
                            )
                            published.append(
                                PublishedSample(
                                    request.request_id,
                                    scheduled.sequence_epoch,
                                    sample,
                                    merged,
                                    support_by_row.get(sample_index - 1),
                                )
                            )
                        chunk = chunks_by_slice.get(slice_index)
                        if chunk is not None:
                            prompt_logprobs.append(chunk)
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
            self._trace("retire", ticket, status=outcome.status.value)
            del self._bindings[ticket.id]
        except BaseException as exc:
            self.executor.quarantine(
                ticket, f"completion settlement failed: {exc}", host_failure=True
            )
            self._trace("settlement_failed", ticket, detail=str(exc))
            if not isinstance(exc, Exception):
                raise
            return None
        return CompletionResult(
            ticket.id,
            outcome.status,
            tuple(published),
            tuple(ignored),
            outcome.error,
            tuple(prompt_logprobs),
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
