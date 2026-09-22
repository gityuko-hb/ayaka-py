"""Resident KV ownership adapters for the serialized scheduler/executor loop."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, replace

from ayaka.executor.base import Executor
from ayaka.executor.completion import CompletionCoordinator
from ayaka.executor.ticket import ExecutionTicket, TerminalStatus, TicketId, TicketState
from ayaka.handles import SequenceHandle
from ayaka.kvcache.manager import KVCapacityError, LogicalKVManager
from ayaka.memory.capacity import ResourceGeneration
from ayaka.memory.pressure import PreemptionStatus
from ayaka.memory.state import ReservationFailure
from ayaka.memory.workspace import WorkspaceLease, WorkspaceManager
from ayaka.plan import (
    EMPTY_COMMUNICATION_PLAN,
    EMPTY_GRAPH_PLAN,
    EMPTY_MEMORY_PLAN,
    EMPTY_WEIGHT_RESIDENCY_PLAN,
    ExecutionPlan,
)
from ayaka.prefix.identity import PrefixCacheContext
from ayaka.request.lifecycle import RequestLifecycle
from ayaka.request.schema import Request
from ayaka.request.states import RequestState
from ayaka.runner.buffers import RunnerBufferExhausted, RunnerBuffers
from ayaka.sched.interfaces import StepPrepareError
from ayaka.sched.plan import BatchStepPlan, PreparedStep

PrefixContextProvider = Callable[[Request], PrefixCacheContext]


@dataclass(slots=True)
class PrefixReuseRequestMetrics:
    """One request's prefix accounting: ranking hint vs acquisition vs forwards.

    ``hint_tokens`` is the latest borrowed lookup estimate; ``acquired_tokens``
    is what attach actually validated and bound (never guessed from the hint);
    ``forwarded_tokens`` counts query rows whose forward completed and KV
    committed. Suffix-only compute means ``forwarded_tokens == (P - H) + G - 1``
    for ``P`` prompt tokens, ``G`` outputs and ``H`` acquired tokens.
    """

    hint_tokens: int = 0
    acquired_tokens: int = 0
    forwarded_tokens: int = 0


@dataclass(slots=True)
class PrefixReuseStats:
    """Aggregate prefix-reuse counters; evidence for R05, not ownership data."""

    lookups: int = 0
    hits: int = 0
    clean_misses: int = 0
    stale_hints: int = 0
    deferred: int = 0
    publish_calls: int = 0
    published: int = 0
    forwarded_tokens: int = 0


class PrefixReuseTelemetry:
    """Accounting facade for prefix hints, acquisitions, forwards and publishes.

    Records are keyed by request id and survive the request's sequence release
    so a consumer can read final counters after completion. A reused request id
    starts a fresh record; the aggregate counters never reset implicitly.
    """

    def __init__(self) -> None:
        self.records: dict[str, PrefixReuseRequestMetrics] = {}
        self.stats = PrefixReuseStats()

    def observe_hint(self, request_id: str, tokens: int) -> None:
        """Record the peak hint seen while the request waits; latest may be stale."""
        record = self.records.get(request_id)
        if record is None:
            record = self.records[request_id] = PrefixReuseRequestMetrics()
        record.hint_tokens = max(record.hint_tokens, max(0, int(tokens)))

    def observe_lookup(self, request_id: str) -> None:
        self.stats.lookups += 1
        if request_id not in self.records:
            self.records[request_id] = PrefixReuseRequestMetrics()

    def observe_acquire(self, request_id: str, *, matched: int, acquired: int) -> None:
        record = self.records.get(request_id) or self.records.setdefault(
            request_id, PrefixReuseRequestMetrics()
        )
        acquired = max(0, int(acquired))
        record.acquired_tokens = acquired
        if acquired:
            self.stats.hits += 1
        elif matched:
            # A capability existed at lookup but failed revalidation at attach.
            self.stats.stale_hints += 1
        else:
            self.stats.clean_misses += 1

    def observe_deferred(self, request_id: str) -> None:
        self.stats.deferred += 1
        if request_id not in self.records:
            self.records[request_id] = PrefixReuseRequestMetrics()

    def record_forward(self, request_id: str, tokens: int) -> None:
        record = self.records.get(request_id)
        if record is None:
            record = self.records[request_id] = PrefixReuseRequestMetrics()
        record.forwarded_tokens += max(0, int(tokens))
        self.stats.forwarded_tokens += max(0, int(tokens))

    def observe_publish(self, request_id: str, *, published: bool) -> None:
        self.stats.publish_calls += 1
        if published:
            self.stats.published += 1
        if request_id not in self.records:
            self.records[request_id] = PrefixReuseRequestMetrics()


def prefix_prompt_candidate(request: Request) -> tuple[int, ...]:
    """Prompt tokens a prefix may cover: everything except the last query.

    This is the single definition of the request-aware boundary rule
    ``attached_tokens <= prompt_len - 1``. The final prompt query always
    executes so its logits feed sampling; generic token lookup must never
    guess this boundary itself.
    """
    return tuple(request.prompt_token_ids[:-1])


def prompt_logprobs_requires_full_prompt(request: Request) -> bool:
    """Whether prompt scoring forces every prompt query to execute.

    A request with ``prompt_logprobs`` never uses prefix reuse: the sampler
    needs predecessor logits for every scored position, and no cached
    capability may silently drop them.
    """
    return request.sampling.prompt_logprobs is not None


class KVSequenceAllocator:
    """Own admitted handles; repeated terminal release is safe by generation."""

    def __init__(self, kv: LogicalKVManager) -> None:
        self.kv = kv
        self._sequences: dict[str, SequenceHandle] = {}
        self._owners: dict[SequenceHandle, str] = {}
        self.on_release: Callable[[SequenceHandle], None] | None = None

    def create(self, request_id: str) -> SequenceHandle:
        if request_id in self._sequences:
            raise ValueError(f"sequence already exists for {request_id}")
        sequence = self.kv.create_sequence(request_id)
        self._sequences[request_id] = sequence
        self._owners[sequence] = request_id
        return sequence

    def get(self, request_id: str) -> SequenceHandle:
        return self._sequences[request_id]

    def release(self, sequence: SequenceHandle) -> None:
        request_id = self._owners.get(sequence)
        if request_id is None:
            return
        self.kv.release_sequence(sequence)
        del self._sequences[request_id]
        del self._owners[sequence]
        if self.on_release is not None:
            self.on_release(sequence)

    def advance_epoch(self, completed_steps: int) -> int:
        # A recreated engine may start its local settled counter at zero.
        return self.kv.advance_epoch(max(completed_steps, self.kv.current_epoch))


class KVRequestPreparer:
    """Acquire resident prefixes and rebind recompute victims before snapshots.

    Contexts must encode model/weights, layout, adapters and isolation salt.
    Prefix lookup hints own no pages. Only attach_resume advances logical
    progress. The prompt boundary lives solely in
    :func:`prefix_prompt_candidate`. Both the homogeneous store and the
    grouped all-group boundary cache serve this path; a grouped capability
    that is no longer executable reports a clean miss.
    """

    def __init__(
        self, allocator: KVSequenceAllocator, *, prefix_context: PrefixContextProvider | None = None
    ) -> None:
        self.allocator = allocator
        self.kv = allocator.kv
        self.prefix_context = prefix_context
        self._contexts: dict[SequenceHandle, PrefixCacheContext] = {}
        allocator.on_release = self.forget
        self.prefix = allocator.kv.prefix_service if prefix_context is not None else None
        self.telemetry = PrefixReuseTelemetry()

    def _context(self, request: Request) -> PrefixCacheContext:
        if self.prefix_context is None:
            raise ValueError("prefix caching is disabled")
        context = self.prefix_context(request)
        # Encode both namespaces as an ordered pair; neither can erase the other.
        return replace(
            context, cache_salt=json.dumps((context.cache_salt, request.cache.cache_salt))
        )

    def estimate_cached_tokens(self, lifecycle: RequestLifecycle) -> int:
        if not lifecycle.request.cache.prefix_cache or prompt_logprobs_requires_full_prompt(
            lifecycle.request
        ):
            return 0
        if self.prefix is None:
            return 0
        # Keep at least the last prompt query for logits, including exact hits.
        match = self.prefix.lookup(
            prefix_prompt_candidate(lifecycle.request), context=self._context(lifecycle.request)
        )
        value = 0 if match is None else match.logical_position
        self.telemetry.observe_hint(lifecycle.request_id, value)
        return value

    def prepare_request(self, lifecycle: RequestLifecycle) -> bool:
        if lifecycle.is_terminal or lifecycle.token.is_cancelled or lifecycle.inflight_slice:
            return False
        sequence = self.allocator.get(lifecycle.request_id)
        state = self.kv.get_sequence(sequence)
        if state.busy or state.release_requested:
            return False
        resumed = lifecycle.state is RequestState.PREEMPTED
        if resumed:
            if state.committed_tokens:
                raise ValueError("recompute resume requires an empty physical sequence")
        attached = 0
        if self.prefix is not None:
            context = self._context(lifecycle.request)
            if lifecycle.request.cache.store_kv:
                self._contexts[sequence] = context
            # Prompt scoring requires executing every scored query.
            if (
                not state.committed_tokens
                and lifecycle.request.cache.prefix_cache
                and not prompt_logprobs_requires_full_prompt(lifecycle.request)
            ):
                self.telemetry.observe_lookup(lifecycle.request_id)
                match = self.prefix.lookup(
                    prefix_prompt_candidate(lifecycle.request), context=context
                )
                if match is None:
                    self.telemetry.observe_acquire(lifecycle.request_id, matched=0, acquired=0)
                else:
                    if not self.prefix.ready(match):
                        # Admission hint, not a reservation: wait for COW headroom.
                        self.telemetry.observe_deferred(lifecycle.request_id)
                        return False
                    attached = self.prefix.acquire(
                        sequence,
                        match,
                        token_ids=prefix_prompt_candidate(lifecycle.request),
                        context=context,
                    )
                    self.telemetry.observe_acquire(
                        lifecycle.request_id, matched=match.logical_position, acquired=attached
                    )
        if resumed or attached:
            if resumed:
                lifecycle.machine.transition(RequestState.WAITING)
                lifecycle.machine.transition(RequestState.ADMITTED)
            state = self.kv.get_sequence(sequence)
            lifecycle.bind_sequence(
                sequence, state_version=state.version, computed_tokens=state.committed_tokens
            )
        elif (
            state.version != lifecycle.state_version
            or state.committed_tokens != lifecycle.computed_tokens
        ):
            # The physical sequence advanced or was reset outside this request's
            # snapshot (for example a recompute preemption that released its KV
            # before the lifecycle state was updated). Rebinding here prevents a
            # plan from being built on the pre-reset view and failing validation.
            lifecycle.bind_sequence(
                sequence, state_version=state.version, computed_tokens=state.committed_tokens
            )
        return True

    def publish(self, prepared: PreparedStep, discarded: tuple[SequenceHandle, ...]) -> None:
        if self.prefix is None:
            return
        for scheduled, value in zip(prepared.step.slices, prepared.step.inputs, strict=True):
            if value.sequence in discarded or scheduled.query_end != value.prompt_tokens:
                continue
            context = self._contexts.get(value.sequence)
            if context is None:
                continue
            tokens = value.known_tokens[: value.prompt_tokens]
            result = self.prefix.publish(value.sequence, tokens, context=context)
            self.telemetry.observe_publish(scheduled.request_id, published=result is not None)

    def forget(self, sequence: SequenceHandle) -> None:
        self._contexts.pop(sequence, None)

    def preempt(self, lifecycle: RequestLifecycle, sequence: SequenceHandle, *, mode: str) -> bool:
        if mode != "recompute" or lifecycle.inflight_slice or lifecycle.token.is_cancelled:
            return False
        if self.allocator.get(lifecycle.request_id) != sequence:
            raise ValueError("preemption sequence does not belong to request")
        result = self.kv.preempt_sequence(sequence)
        # Scheduler publishes PREEMPTED after this callback. Rebind on next selection.
        return result.status is PreemptionStatus.RELEASED


class KVExecutionResources:
    """Exclusive lease; only a quiescent completion may commit or retire it."""

    def __init__(
        self,
        prepared: PreparedStep,
        allocator: KVSequenceAllocator,
        requests: KVRequestPreparer | None = None,
        *,
        executor: Executor,
        workspace: WorkspaceManager | None = None,
    ) -> None:
        self.executor = executor
        self.prepared = prepared
        self.allocator = allocator
        self.kv = allocator.kv
        self.requests = requests
        self.workspace = workspace
        self.workspace_lease: WorkspaceLease | None = None
        self.workspace_grew = False
        self.buffer_lease = prepared.buffers
        self.generation: ResourceGeneration | None = self.kv.generation
        self.owner: TicketId | None = None
        self.submitted = False
        self.committed = False
        self.retired = False

    def adopt(self, ticket_id: TicketId, prepared: PreparedStep) -> None:
        if self.owner is not None or self.retired or prepared is not self.prepared:
            raise ValueError("prepared bundle has another owner or identity")
        ticket = self.executor.get_ticket(ticket_id)
        if ticket is None or ticket._resources is not self or ticket.prepared is not prepared:
            raise ValueError("ticket is not registered with the owning executor")
        prepared.validate()
        self.kv.validate_view(prepared.memory_view)
        if self.buffer_lease is not None:
            # The slot is now ticket-owned: no other step may stage into it
            # before this ticket's consumer-completion boundary.
            self.buffer_lease.bind(ticket_id)
        self.owner = ticket_id

    def _check(self, ticket_id: TicketId) -> None:
        if self.owner is None or ticket_id != self.owner:
            raise ValueError("ticket does not own KV resources")

    def _require_completion(self, ticket_id: TicketId, *, succeeded: bool) -> None:
        ticket = self.executor.get_ticket(ticket_id)
        if (
            ticket is None
            or ticket._resources is not self
            or ticket.state is not TicketState.COMPLETED
            or ticket.terminal is None
            or not ticket._settling
        ):
            raise ValueError("KV settlement requires executor-confirmed quiescent completion")
        if (ticket.terminal.status is TerminalStatus.SUCCEEDED) != succeeded:
            raise ValueError("KV settlement disagrees with the executor outcome")

    def _require_generation(self) -> None:
        """Settlement is refused once the runtime generation was replaced."""
        captured = self.generation
        if captured is None:
            return
        current = self.kv.generation
        if current is None or current != captured:
            raise ValueError("KV resources belong to a replaced runtime generation")

    def mark_submitted(self, ticket_id: TicketId) -> None:
        self._check(ticket_id)
        if self.submitted or self.retired:
            raise ValueError("KV bundle was already submitted or retired")
        self.kv.validate_view(self.prepared.memory_view)
        if self.workspace is not None and self.prepared.step.memory.workspaces:
            # The lease binds the step's scratch to this ticket until settlement.
            lease, grew = self.workspace.prepare(self.prepared.step.memory)
            self.workspace_lease = lease
            self.workspace_grew = grew
        self.kv.mark_step_in_flight(self.prepared.memory_view.lease)
        self.submitted = True

    def commit(self, ticket_id: TicketId) -> tuple[int, ...]:
        self._check(ticket_id)
        if not self.submitted or self.committed or self.retired:
            raise ValueError("KV bundle is not awaiting commit")
        self._require_completion(ticket_id, succeeded=True)
        self._require_generation()
        self.kv.commit_step(self.prepared.memory_view.lease)
        self.committed = True
        return tuple(
            self.kv.get_sequence(value.sequence).version for value in self.prepared.step.inputs
        )

    def retire(
        self,
        ticket_id: TicketId,
        *,
        succeeded: bool,
        discarded_sequences: tuple[SequenceHandle, ...] = (),
    ) -> None:
        self._check(ticket_id)
        if self.retired:
            return
        self._require_completion(ticket_id, succeeded=succeeded)
        if succeeded != self.committed:
            raise ValueError("retirement success disagrees with KV commit")
        self._require_generation()
        for sequence in discarded_sequences:
            self.allocator.release(sequence)
        lease = self.prepared.memory_view.lease
        if self.committed:
            self.kv.retire_step(lease)
        elif self.submitted:
            self.kv.fail_in_flight_step(lease, safe_epoch=self.kv.current_epoch)
        else:
            self.kv.abort_prepared_step(lease)
        if succeeded and self.requests is not None:
            self.requests.publish(self.prepared, discarded_sequences)
            for scheduled in self.prepared.step.slices:
                self.requests.telemetry.record_forward(scheduled.request_id, scheduled.query_count)
        if self.requests is not None:
            for sequence in discarded_sequences:
                self.requests.forget(sequence)
        self.kv.reclaim_deferred()
        self._release_workspace()
        self._release_buffers()
        self.retired = True

    def _release_workspace(self) -> None:
        if self.workspace_lease is not None:
            self.workspace_lease.close()
            self.workspace_lease = None

    def _release_buffers(self) -> None:
        if self.buffer_lease is not None:
            self.buffer_lease.release()
            self.buffer_lease = None

    def abort(self) -> None:
        """Roll back caller-owned preparation; never abort an adopted bundle."""
        if self.owner is not None:
            raise ValueError("only executor completion can release an adopted bundle")
        if not self.retired:
            self.kv.abort_prepared_step(self.prepared.memory_view.lease)
            self._release_workspace()
            self._release_buffers()
            self.retired = True


class KVStepRuntime:
    """Resolve physical reservations and transfer ownership to the coordinator."""

    def __init__(
        self,
        execution: ExecutionPlan,
        allocator: KVSequenceAllocator,
        coordinator: CompletionCoordinator,
        *,
        requests: KVRequestPreparer | None = None,
        graph_planner=None,
        workspace: WorkspaceManager | None = None,
        buffers: RunnerBuffers | None = None,
    ) -> None:
        self.execution = execution
        self.allocator = allocator
        self.coordinator = coordinator
        self.requests = requests
        self.graph_planner = graph_planner
        self.workspace = workspace
        self.buffers = buffers
        self._pending: dict[int, KVExecutionResources] = {}

    def prepare(self, step: BatchStepPlan) -> PreparedStep:
        step.validate()
        if step.step_id in self._pending:
            raise ValueError("step_id already owns a prepared bundle")
        if self.graph_planner is not None:
            if step.graph == EMPTY_GRAPH_PLAN:
                step = replace(step, graph=self.graph_planner.plan(step))
            self.graph_planner.validate(step)
        if (
            self.coordinator.executor.closed
            or not self.coordinator.executor.has_submission_capacity()
        ):
            raise StepPrepareError("executor has no submission capacity", transient=True)
        if step.execution_plan_id != self.execution.plan_id or not step.slices:
            raise ValueError("empty step or mismatched execution plan")
        if (
            step.dependencies
            or step.distributed is not None
            or step.communication != EMPTY_COMMUNICATION_PLAN
            or step.weight_residency != EMPTY_WEIGHT_RESIDENCY_PLAN
            or (step.memory != EMPTY_MEMORY_PLAN and self.workspace is None)
            or (step.graph != EMPTY_GRAPH_PLAN and self.graph_planner is None)
        ):
            raise ValueError("resident KV runtime does not support these auxiliary plans")
        kv = self.allocator.kv
        kv.reclaim_deferred()
        while True:
            try:
                view = kv.reserve(step)
                break
            except KVCapacityError as exc:
                if exc.reason is ReservationFailure.NO_CAPACITY:
                    # Release only as many cache-only pages as the reservation needs.
                    # Every retry is a fresh atomic reservation across all groups.
                    pressure = kv.evict_prefixes_for_pressure(1)
                    if pressure.pages_reclaimed:
                        continue
                    if any(kv.privatize_prefix_tail(value.sequence) for value in step.inputs):
                        # Removing cache sharing can eliminate a COW allocation
                        # even when eviction cannot free a request-owned page.
                        continue
                raise self._capacity_error(exc) from exc
        buffer_lease = None
        if self.buffers is not None:
            try:
                buffer_lease = self.buffers.acquire(step_id=step.step_id)
            except RunnerBufferExhausted as exc:
                # Backpressure, not a failed step: roll the KV reservation back
                # and let the scheduler retry once a flight retires.
                kv.abort_prepared_step(view.lease)
                raise StepPrepareError(str(exc), transient=True) from exc
            except BaseException:
                kv.abort_prepared_step(view.lease)
                raise
        try:
            prepared = PreparedStep(self.execution, step, view, buffer_lease)
            prepared.validate()
            resources = KVExecutionResources(
                prepared,
                self.allocator,
                self.requests,
                executor=self.coordinator.executor,
                workspace=self.workspace,
            )
            self._pending[step.step_id] = resources
            return prepared
        except BaseException:
            if buffer_lease is not None:
                buffer_lease.release()
            kv.abort_prepared_step(view.lease)
            raise

    @staticmethod
    def _capacity_error(error: KVCapacityError) -> StepPrepareError:
        return StepPrepareError(
            str(error),
            request_id=error.request_id,
            transient=error.reason
            in (ReservationFailure.NO_CAPACITY, ReservationFailure.SEQUENCE_BUSY),
        )

    def adopt(self, prepared: PreparedStep) -> ExecutionTicket:
        resources = self._pending.get(prepared.step.step_id)
        if resources is None or resources.prepared is not prepared:
            raise ValueError("prepared step is not owned by this runtime")
        try:
            return self.coordinator.adopt(prepared, resources)
        except BaseException:
            if resources.owner is None:
                resources.abort()
            raise
        finally:
            del self._pending[prepared.step.step_id]

    def discard(self, prepared: PreparedStep) -> None:
        resources = self._pending.get(prepared.step.step_id)
        if resources is None or resources.prepared is not prepared:
            raise ValueError("prepared step is not owned by this runtime")
        resources.abort()
        del self._pending[prepared.step.step_id]

    def cancel(self, ticket: ExecutionTicket, reason: str) -> None:
        self.coordinator.executor.cancel(ticket, reason)

    def close(self) -> None:
        """Abort only preparation still owned by this runtime."""
        for resources in tuple(self._pending.values()):
            resources.abort()
        self._pending.clear()
