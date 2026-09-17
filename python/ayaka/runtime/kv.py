"""Resident KV ownership adapters for the serialized scheduler/executor loop."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace

from ayaka.executor.completion import CompletionCoordinator
from ayaka.executor.ticket import ExecutionTicket, TicketId
from ayaka.handles import SequenceHandle
from ayaka.kvcache.manager import KVCapacityError, LogicalKVManager
from ayaka.memory.manager import RuntimeMemoryManager
from ayaka.memory.pressure import PreemptionStatus
from ayaka.memory.state import ReservationFailure
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
from ayaka.sched.interfaces import StepPrepareError
from ayaka.sched.plan import BatchStepPlan, PreparedStep

PrefixContextProvider = Callable[[Request], PrefixCacheContext]


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
        sequence = self.kv.backend.create_sequence(request_id)
        self._sequences[request_id] = sequence
        self._owners[sequence] = request_id
        return sequence

    def get(self, request_id: str) -> SequenceHandle:
        return self._sequences[request_id]

    def release(self, sequence: SequenceHandle) -> None:
        request_id = self._owners.get(sequence)
        if request_id is None:
            return
        self.kv.backend.release_sequence(sequence)
        del self._sequences[request_id]
        del self._owners[sequence]
        if self.on_release is not None:
            self.on_release(sequence)

    def advance_epoch(self, completed_steps: int) -> int:
        # A recreated engine may start its local settled counter at zero.
        return self.kv.backend.advance_epoch(max(completed_steps, self.kv.backend.current_epoch))


class KVRequestPreparer:
    """Acquire resident prefixes and rebind recompute victims before snapshots.

    Contexts must encode model/weights, layout, adapters and isolation salt.
    Prefix lookup hints own no pages. Only attach_prefix advances logical progress.
    Grouped canonical prefix caching is deliberately unsupported by its backend.
    """

    def __init__(
        self, allocator: KVSequenceAllocator, *, prefix_context: PrefixContextProvider | None = None
    ) -> None:
        self.allocator = allocator
        self.backend = allocator.kv.backend
        self.prefix_context = prefix_context
        self._contexts: dict[SequenceHandle, PrefixCacheContext] = {}
        allocator.on_release = self.forget
        if prefix_context is not None and not isinstance(self.backend, RuntimeMemoryManager):
            raise ValueError("canonical prefix caching is not implemented for grouped KV")

    def _context(self, request: Request) -> PrefixCacheContext:
        if self.prefix_context is None:
            raise ValueError("prefix caching is disabled")
        context = self.prefix_context(request)
        # Encode both namespaces as an ordered pair; neither can erase the other.
        return replace(
            context, cache_salt=json.dumps((context.cache_salt, request.cache.cache_salt))
        )

    def estimate_cached_tokens(self, lifecycle: RequestLifecycle) -> int:
        if not lifecycle.request.cache.prefix_cache:
            return 0
        if self.prefix_context is None or not isinstance(self.backend, RuntimeMemoryManager):
            return 0
        # Keep at least the last prompt query for logits, including exact hits.
        return self.backend.match_prefix(
            lifecycle.request.prompt_token_ids[:-1], context=self._context(lifecycle.request)
        ).matched_tokens

    def prepare_request(self, lifecycle: RequestLifecycle) -> bool:
        if lifecycle.is_terminal or lifecycle.token.is_cancelled or lifecycle.inflight_slice:
            return False
        sequence = self.allocator.get(lifecycle.request_id)
        state = self.backend.get_sequence(sequence)
        if state.busy or state.release_requested:
            return False
        resumed = lifecycle.state is RequestState.PREEMPTED
        if resumed:
            if state.committed_tokens:
                raise ValueError("recompute resume requires an empty physical sequence")
            lifecycle.machine.transition(RequestState.WAITING)
            lifecycle.machine.transition(RequestState.ADMITTED)
        attached = 0
        if self.prefix_context is not None and isinstance(self.backend, RuntimeMemoryManager):
            context = self._context(lifecycle.request)
            if lifecycle.request.cache.store_kv:
                self._contexts[sequence] = context
            # Prompt scoring requires executing every scored query.
            if (
                not state.committed_tokens
                and lifecycle.request.cache.prefix_cache
                and lifecycle.request.sampling.prompt_logprobs is None
            ):
                match = self.backend.match_prefix(
                    lifecycle.request.prompt_token_ids[:-1], context=context
                )
                if match.matched_tokens:
                    attached = self.backend.attach_prefix(sequence, match)
        if resumed or attached:
            state = self.backend.get_sequence(sequence)
            lifecycle.bind_sequence(
                sequence, state_version=state.version, computed_tokens=state.committed_tokens
            )
        return True

    def publish(self, prepared: PreparedStep, discarded: tuple[SequenceHandle, ...]) -> None:
        if self.prefix_context is None or not isinstance(self.backend, RuntimeMemoryManager):
            return
        for scheduled, value in zip(prepared.step.slices, prepared.step.inputs, strict=True):
            if value.sequence in discarded or scheduled.query_end != value.prompt_tokens:
                continue
            context = self._contexts.get(value.sequence)
            if context is None:
                continue
            tokens = value.known_tokens[: value.prompt_tokens]
            shareable = len(tokens) // self.backend.page_size * self.backend.page_size
            # Recompute or an identical request must not accumulate terminal refs.
            if self.backend.match_prefix(tokens, context=context).matched_tokens < shareable:
                self.backend.cache_prefix(value.sequence, tokens, context=context)

    def forget(self, sequence: SequenceHandle) -> None:
        self._contexts.pop(sequence, None)

    def preempt(self, lifecycle: RequestLifecycle, sequence: SequenceHandle, *, mode: str) -> bool:
        if mode != "recompute" or lifecycle.inflight_slice or lifecycle.token.is_cancelled:
            return False
        if self.allocator.get(lifecycle.request_id) != sequence:
            raise ValueError("preemption sequence does not belong to request")
        result = self.backend.preempt_sequence(sequence)
        # Scheduler publishes PREEMPTED after this callback. Rebind on next selection.
        return result.status is PreemptionStatus.RELEASED


class KVExecutionResources:
    """Exclusive lease; only a quiescent completion may commit or retire it."""

    def __init__(
        self,
        prepared: PreparedStep,
        allocator: KVSequenceAllocator,
        requests: KVRequestPreparer | None = None,
    ) -> None:
        self.prepared = prepared
        self.allocator = allocator
        self.kv = allocator.kv
        self.requests = requests
        self.owner: TicketId | None = None
        self.submitted = False
        self.committed = False
        self.retired = False

    def adopt(self, ticket_id: TicketId, prepared: PreparedStep) -> None:
        if self.owner is not None or self.retired or prepared is not self.prepared:
            raise ValueError("prepared bundle has another owner or identity")
        prepared.validate()
        self.kv.validate_view(prepared.memory_view)
        self.owner = ticket_id

    def _check(self, ticket_id: TicketId) -> None:
        if ticket_id != self.owner:
            raise ValueError("ticket does not own KV resources")

    def mark_submitted(self, ticket_id: TicketId) -> None:
        self._check(ticket_id)
        if self.submitted or self.retired:
            raise ValueError("KV bundle was already submitted or retired")
        self.kv.validate_view(self.prepared.memory_view)
        self.kv.backend.mark_step_in_flight(self.prepared.memory_view.lease)
        self.submitted = True

    def commit(self, ticket_id: TicketId) -> tuple[int, ...]:
        self._check(ticket_id)
        if not self.submitted or self.committed or self.retired:
            raise ValueError("KV bundle is not awaiting commit")
        backend = self.kv.backend
        backend.commit_step(self.prepared.memory_view.lease)
        self.committed = True
        return tuple(
            backend.get_sequence(value.sequence).version for value in self.prepared.step.inputs
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
        if succeeded != self.committed:
            raise ValueError("retirement success disagrees with KV commit")
        backend = self.kv.backend
        for sequence in discarded_sequences:
            self.allocator.release(sequence)
        lease = self.prepared.memory_view.lease
        if self.committed:
            backend.retire_step(lease)
        elif self.submitted:
            backend.fail_in_flight_step(lease, safe_epoch=backend.current_epoch)
        else:
            backend.abort_prepared_step(lease)
        if succeeded and self.requests is not None:
            self.requests.publish(self.prepared, discarded_sequences)
        if self.requests is not None:
            for sequence in discarded_sequences:
                self.requests.forget(sequence)
        backend.reclaim_deferred()
        self.retired = True

    def abort(self) -> None:
        """Roll back caller-owned preparation; never abort an adopted bundle."""
        if self.owner is not None:
            raise ValueError("only executor completion can release an adopted bundle")
        if not self.retired:
            self.kv.backend.abort_prepared_step(self.prepared.memory_view.lease)
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
    ) -> None:
        self.execution = execution
        self.allocator = allocator
        self.coordinator = coordinator
        self.requests = requests
        self.graph_planner = graph_planner
        self._pending: dict[int, KVExecutionResources] = {}

    def prepare(self, step: BatchStepPlan) -> PreparedStep:
        step.validate()
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
            or step.memory != EMPTY_MEMORY_PLAN
            or (step.graph != EMPTY_GRAPH_PLAN and self.graph_planner is None)
        ):
            raise ValueError("resident KV runtime does not support these auxiliary plans")
        kv = self.allocator.kv
        kv.backend.reclaim_deferred()
        while True:
            try:
                view = kv.reserve(step)
                break
            except KVCapacityError as exc:
                if exc.reason is ReservationFailure.NO_CAPACITY:
                    # Release only as many cache-only pages as the reservation needs.
                    # Every retry is a fresh atomic reservation across all groups.
                    pressure = kv.backend.evict_prefixes_for_pressure(1)
                    if pressure.pages_reclaimed:
                        continue
                raise self._capacity_error(exc) from exc
        try:
            prepared = PreparedStep(self.execution, step, view)
            prepared.validate()
            resources = KVExecutionResources(prepared, self.allocator, self.requests)
            self._pending[step.step_id] = resources
            return prepared
        except BaseException:
            kv.backend.abort_prepared_step(view.lease)
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
