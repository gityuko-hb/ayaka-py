"""Shared scheduler lifecycle machinery.

``SchedulerCore`` deliberately does *not* decide which request runs next.  It
centralizes request admission, sequence ownership, abort settlement, reports,
and common inspection so EagerScheduler and ContinuousScheduler do not drift in
lifecycle semantics.
"""

from __future__ import annotations

import time
from abc import abstractmethod
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

from ayaka.configs.base import ConfigError
from ayaka.configs.scheduler import ResolvedSchedulerPlan
from ayaka.executor.ticket import ExecutionTicket, TicketId
from ayaka.handles import SequenceHandle
from ayaka.plan import SamplingPlan
from ayaka.request.lifecycle import LifecycleManager, RequestLifecycle
from ayaka.request.schema import Request, RequestId
from ayaka.sampling.logprobs import LogprobMode
from ayaka.sched.base import BaseScheduler
from ayaka.sched.interfaces import OverloadedError, RequestPreparer, SequenceAllocator, StepRuntime
from ayaka.sched.outcome import FinishReason, RequestOutcome, RequestReport, SchedulerReport
from ayaka.sched.plan import PromptLogprobSlicePlan, RequestStepInput
from ayaka.sched.policy import QueueEntry

if TYPE_CHECKING:
    from ayaka.sampling.engine import SamplingCoordinator
    from ayaka.sched.plan import ScheduledSlice

__all__ = ["SchedulerCore", "HOT_WINDOW_SIZE"]

#: Number of eligible candidates kept in the active ranking window. New
#: requests land in the O(1) staging deque and are drained into this window in
#: arrival order, so starvation ranking (`order`) and policy sorting only ever
#: touch a bounded slice of the waiting queue.
HOT_WINDOW_SIZE = 128


class SchedulerCore(BaseScheduler):
    """Policy-free scheduler infrastructure shared by all implementations."""

    def __init__(
        self,
        plan: ResolvedSchedulerPlan,
        requests: LifecycleManager,
        runtime: StepRuntime,
        allocator: SequenceAllocator,
        *,
        text_stops: bool = False,
        clock: Callable[[], int] | None = None,
        sampling: SamplingCoordinator | None = None,
        request_preparer: RequestPreparer | None = None,
    ) -> None:
        if not isinstance(plan, ResolvedSchedulerPlan):
            raise TypeError("plan must be ResolvedSchedulerPlan")
        if not isinstance(requests, LifecycleManager):
            raise TypeError("requests must be LifecycleManager")
        if sampling is not None and plan.max_num_requests is not None:
            if sampling.max_batch_size < plan.max_num_requests:
                raise ValueError(
                    f"sampling capacity {sampling.max_batch_size} < max_num_requests "
                    f"{plan.max_num_requests}; slots are held for the whole admitted "
                    "lifetime, not just the running set"
                )

        self._request_preparer = request_preparer
        self._plan = plan
        self._requests = requests
        self._runtime = runtime
        self._allocator = allocator
        self._text_stops = text_stops
        self._clock = clock or time.monotonic_ns
        self._sampling = sampling

        # Two-level waiting queue: staging is an O(1)-append arrival deque;
        # `_hot` is the bounded ranking window drained in arrival order.
        # `_waiting_ids` is the O(1) membership/duplicate oracle for both tiers.
        self._staging: deque[QueueEntry] = deque()
        self._hot: list[QueueEntry] = []
        self._waiting_ids: set[str] = set()
        self._prefilling: dict[str, RequestLifecycle] = {}
        self._running: dict[str, RequestLifecycle] = {}
        # Dense decode ring parallel to `_running`: swap-with-last removal keeps
        # the array packed without rebuilding per-step lists.
        self._running_ids: list[str] = []
        self._running_index: dict[str, int] = {}
        self._sequences: dict[str, SequenceHandle] = {}
        # One observation per request, keyed for O(1) dedup on report.
        self._pending: dict[str, RequestReport] = {}

        # Multiple adopted-but-unsettled tickets may exist at once (pipeline
        # microbatches / overlap). Entries are keyed by ticket id; the union
        # of their request ids is the scheduling exclusion oracle.
        self._inflight: dict[TicketId, ExecutionTicket] = {}
        self._inflight_request_ids: dict[TicketId, frozenset[str]] = {}
        self._inflight_ids_all: set[str] = set()
        self._abort_pending: set[str] = set()
        self._resource_blocked = False

        self._next_ordinal = 0
        self._ordinals: dict[str, int] = {}
        self._round = 0
        self._step_id = -1
        self._report_step_id = -1

    # ------------------------------------------------------------------
    # Base lifecycle operations
    # ------------------------------------------------------------------

    def add_request(
        self,
        request: Request,
        *,
        defer_to_remote_kv: bool = False,
    ) -> RequestLifecycle:
        """Validate, bind a sequence, and enqueue the request.

        n > 1 mở rộng tại admission thành n request con độc lập
        ``f"{id}#c{i}"`` — seed riêng, stream RNG riêng, lifecycle riêng
        (nguyên tắc "expand at admission" thay vì fork động giữa chừng).
        Prompt KV bị NHÂN BẢN cho mỗi child — chưa có prefix sharing (paged
        runtime là backlog). Trả lifecycle của child đầu; caller thấy token
        theo child id. ``defer_to_remote_kv`` dừng ở WAITING_REMOTE_KV; một
        RemoteKVPending registry phải chủ động cho request đi tiếp.
        """
        if request.sampling.n > 1:
            first: RequestLifecycle | None = None
            for index in range(request.sampling.n):
                child_seed = (
                    None if request.sampling.seed is None else request.sampling.seed + index
                )
                child = Request(
                    RequestId(f"{request.request_id}#c{index}"),
                    request.prompt_token_ids,
                    sampling=replace(request.sampling, n=1, seed=child_seed),
                    stop=request.stop,
                )
                lifecycle = self.add_request(child, defer_to_remote_kv=defer_to_remote_kv)
                first = first or lifecycle
            assert first is not None
            return first
        self._plan.validate_request(request)
        self._validate_request_features(request)
        self._check_capacity()

        # Claim the sampling slot before the lifecycle so a capacity failure
        # leaves no half-registered request behind. ``create`` repeats the
        # duplicate check; doing it first keeps the slot accounting clean.
        if request.request_id in self._requests:
            raise ValueError(f"duplicate request_id {request.request_id!r}")
        if self._sampling is not None:
            self._sampling.add(
                request.request_id,
                request.sampling,
                request.prompt_token_ids,
                request_index=self._next_ordinal,
            )

        try:
            sequence = self._allocator.create(str(request.request_id))
        except BaseException:
            if self._sampling is not None:
                self._sampling.release(str(request.request_id))
            raise
        try:
            lifecycle = self._requests.create(request)
            self._requests.advance_to_queue(
                lifecycle.request_id,
                defer_to_remote_kv=defer_to_remote_kv,
            )
        except BaseException:
            self._allocator.release(sequence)
            if self._sampling is not None:
                self._sampling.release(str(request.request_id))
            raise
        lifecycle.bind_sequence(sequence, state_version=0, computed_tokens=0)
        self._sequences[lifecycle.request_id] = sequence
        self._waiting_add(self._new_queue_entry(lifecycle, sequence))
        self._queue_report(
            RequestReport(
                lifecycle.request_id,
                RequestOutcome.ADMITTED,
                lifecycle.sequence_epoch,
            )
        )
        return lifecycle

    def _validate_request_features(self, request: Request) -> None:
        params = request.sampling
        if not params.is_greedy and self._sampling is None:
            raise ConfigError(
                "request.sampling",
                "UNSUPPORTED_SAMPLING",
                "non-greedy sampling requires a wired SamplingCoordinator",
            )
        if params.prompt_logprobs is not None and self._sampling is None:
            raise ConfigError(
                "request.sampling.prompt_logprobs",
                "UNSUPPORTED_PROMPT_LOGPROBS",
                "prompt logprobs require a wired SamplingCoordinator",
            )
        if (
            params.logprob_mode is LogprobMode.SAMPLING
            and params.logprobs is None
            and params.token_ids_logprobs is None
        ):
            raise ConfigError(
                "request.sampling.logprob_mode",
                "INVALID_LOGPROB_MODE",
                "logprob_mode=sampling needs logprobs or token_ids_logprobs enabled",
            )
        if params.logprobs is not None and self._sampling is None:
            raise ConfigError(
                "request.sampling.logprobs",
                "UNSUPPORTED_LOGPROBS",
                "generation logprobs require a wired SamplingCoordinator",
            )
        if params.return_sampling_support and self._sampling is None:
            raise ConfigError(
                "request.sampling.return_sampling_support",
                "UNSUPPORTED_SAMPLING_SUPPORT",
                "sampling support capture requires a wired SamplingCoordinator",
            )
        if params.logit_bias and self._sampling is None:
            raise ConfigError(
                "request.sampling.logit_bias",
                "UNSUPPORTED_LOGIT_BIAS",
                "logit bias requires a wired SamplingCoordinator",
            )
        if params.token_ids_logprobs is not None and self._sampling is None:
            raise ConfigError(
                "request.sampling.token_ids_logprobs",
                "UNSUPPORTED_TOKEN_IDS_LOGPROBS",
                "token_ids_logprobs require a wired SamplingCoordinator",
            )
        if request.stop.stop_strings and not self._text_stops:
            raise ConfigError(
                "request.stop.stop_strings",
                "UNSUPPORTED_STOP",
                "text stop strings require a tokenizer-backed output owner",
            )

    def abort(self, request_id: str) -> bool:
        """Cancel one request without poisoning unrelated co-batched requests.

        If the request is already in an adopted mixed batch, only its cancellation
        token is marked.  The whole execution ticket is cancelled only when every
        request in that ticket is pending abort.  Sequence release/reporting is
        deferred until the ticket settles, preserving lease and state-version
        safety.
        """
        lifecycle = self._requests.find(request_id)
        if lifecycle is None or lifecycle.is_terminal:
            return False
        if request_id in self._abort_pending:
            return False

        lifecycle.token.cancel("aborted by scheduler")
        self._abort_pending.add(request_id)

        if self._owns_inflight(request_id):
            # Cancel every inflight ticket whose request set is fully aborted;
            # mixed tickets keep running for their still-live requests.
            for ticket_id, ids in list(self._inflight_request_ids.items()):
                if ids and ids.issubset(self._abort_pending):
                    self._runtime.cancel(
                        self._inflight[ticket_id], "all requests in ticket aborted"
                    )
            return True

        self._finalize_abort(request_id)
        return True

    @property
    def has_unfinished(self) -> bool:
        return self._requests.num_active > 0

    # schedule() and update_from_output() remain algorithm-specific.
    @abstractmethod
    def schedule(self) -> ExecutionTicket | None: ...

    @abstractmethod
    def update_from_output(self, output) -> None: ...

    # ------------------------------------------------------------------
    # Terminal/reporting operations shared by schedulers
    # ------------------------------------------------------------------

    def finish_request(
        self,
        request_id: str,
        reason: FinishReason,
        detail: str = "",
    ) -> SequenceHandle | None:
        lifecycle = self._requests.find(request_id)
        if (
            lifecycle is None
            or lifecycle.is_terminal
            or lifecycle.inflight_slice is not None
            or self._owns_inflight(request_id)
        ):
            return None
        self._drop(request_id)
        self._abort_pending.discard(request_id)
        if self._sampling is not None:
            self._sampling.release(request_id)
        self._queue_report(
            RequestReport(
                request_id,
                RequestOutcome.FINISHED,
                lifecycle.sequence_epoch,
                finish_reason=reason,
                detail=detail,
            )
        )
        return self._sequences.pop(request_id, None)

    def fail_request(self, request_id: str, detail: str) -> SequenceHandle | None:
        lifecycle = self._requests.find(request_id)
        if lifecycle is None or lifecycle.is_terminal:
            return None
        self._drop(request_id)
        self._abort_pending.discard(request_id)
        if self._sampling is not None:
            self._sampling.release(request_id)
        self._queue_report(
            RequestReport(
                request_id,
                RequestOutcome.FAILED,
                lifecycle.sequence_epoch,
                detail=detail,
            )
        )
        return self._sequences.pop(request_id, None)

    def flush_reports(self) -> bool:
        """Apply accumulated scheduler observations to LifecycleManager."""
        if not self._pending:
            return False
        self._report_step_id += 1
        pending = tuple(self._pending.values())
        report = SchedulerReport(
            step_id=self._report_step_id,
            reports=pending,
            num_running=len(self._running) + len(self._prefilling),
            num_waiting=len(self._waiting_ids),
            num_preempted_this_step=sum(
                item.outcome in (RequestOutcome.PREEMPTED_RECOMPUTE, RequestOutcome.PREEMPTED_SWAP)
                for item in pending
            ),
        )
        self._pending.clear()
        self._requests.apply(report)
        return True

    # ------------------------------------------------------------------
    # Inflight bookkeeping
    # ------------------------------------------------------------------

    def _set_inflight(self, ticket: ExecutionTicket, request_ids: Iterable[str]) -> None:
        ids = frozenset(request_ids)
        if not ids:
            raise ValueError("an adopted ticket must own at least one request")
        if ticket.id in self._inflight:
            raise ValueError("ticket is already registered as inflight")
        self._inflight[ticket.id] = ticket
        self._inflight_request_ids[ticket.id] = ids
        self._inflight_ids_all.update(ids)

    def _clear_inflight(self, ticket_id: TicketId) -> frozenset[str] | None:
        ids = self._inflight_request_ids.pop(ticket_id, None)
        if ids is None:
            return None
        del self._inflight[ticket_id]
        self._inflight_ids_all.difference_update(ids)
        return ids

    def _owns_inflight(self, request_id: str) -> bool:
        return request_id in self._inflight_ids_all

    @property
    def inflight_count(self) -> int:
        return len(self._inflight)

    def _finalize_settled_aborts(self, request_ids: Iterable[str]) -> None:
        for request_id in tuple(request_ids):
            if request_id in self._abort_pending:
                self._finalize_abort(request_id)

    def _finalize_abort(self, request_id: str) -> None:
        lifecycle = self._requests.find(request_id)
        self._drop(request_id)
        if self._sampling is not None:
            self._sampling.release(request_id)
        if lifecycle is not None and not lifecycle.is_terminal:
            self._queue_report(
                RequestReport(
                    request_id,
                    RequestOutcome.ABORTED,
                    lifecycle.sequence_epoch,
                    detail="aborted by scheduler",
                )
            )
        sequence = self._sequences.pop(request_id, None)
        if sequence is not None:
            self._allocator.release(sequence)
        self._abort_pending.discard(request_id)

    # ------------------------------------------------------------------
    # Shared queue/state helpers
    # ------------------------------------------------------------------

    def _new_queue_entry(
        self,
        lifecycle: RequestLifecycle,
        sequence: SequenceHandle | None,
    ) -> QueueEntry:
        ordinal = self._ordinals.get(lifecycle.request_id)
        if ordinal is None:
            ordinal = self._next_ordinal
            self._ordinals[lifecycle.request_id] = ordinal
            self._next_ordinal += 1
        return QueueEntry(
            lifecycle=lifecycle,
            ordinal=ordinal,
            ready_round=self._round,
            ready_ns=self._clock(),
            sequence=sequence,
        )

    def _waiting_add(self, entry: QueueEntry) -> None:
        """Append a request to the staging tier in O(1)."""
        self._staging.append(entry)
        self._waiting_ids.add(entry.lifecycle.request_id)

    def _waiting_remove(self, request_id: str) -> None:
        """Drop a request from both tiers in O(window) without rebuilding."""
        if request_id not in self._waiting_ids:
            return
        self._waiting_ids.discard(request_id)
        hot = self._hot
        for index, entry in enumerate(hot):
            if entry.lifecycle.request_id == request_id:
                del hot[index]
                return
        # Still staged: the drain skips it via `_waiting_ids`.

    def _drain_window(self) -> None:
        """Refill the ranking window from staging in arrival order."""
        if not self._staging:
            return
        capacity = HOT_WINDOW_SIZE - len(self._hot)
        if capacity <= 0:
            return
        admitted: list[QueueEntry] = []
        ids = self._waiting_ids
        staging = self._staging
        while staging and len(admitted) < capacity:
            entry = staging.popleft()
            if entry.lifecycle.request_id not in ids:
                continue  # dropped while staged
            admitted.append(entry)
        if admitted:
            self._hot.extend(admitted)
            self._refresh_window_hints(admitted)

    def _refresh_window_hints(self, admitted: list[QueueEntry]) -> None:
        """Ranking-hint refresh hook for entries entering the hot window."""

    def _iter_waiting(self) -> Iterator[QueueEntry]:
        """All waiting entries (hot ranking window first, then staging)."""
        yield from self._hot
        yield from self._staging

    def _requeue(self, lifecycle: RequestLifecycle) -> None:
        if lifecycle.is_terminal or lifecycle.token.is_cancelled:
            return
        if lifecycle.request_id in self._waiting_ids:
            return
        entry = self._new_queue_entry(lifecycle, self._sequences.get(lifecycle.request_id))
        if lifecycle.request_id in self._prefilling:
            # A continuation owns a partial-prefill slot and resident KV. Parking
            # it behind the staging queue can deadlock a full hot window: every
            # new partial prompt waits for the slot its hidden owner must release.
            # Keep that owner visible without enlarging the bounded window.
            if len(self._hot) >= HOT_WINDOW_SIZE:
                self._staging.appendleft(self._hot.pop())
            self._hot.insert(0, entry)
            self._waiting_ids.add(lifecycle.request_id)
            self._refresh_window_hints([entry])
        else:
            self._waiting_add(entry)

    def _queue_report(self, report: RequestReport) -> None:
        """Keep at most one pending observation per request/report step."""
        self._pending[report.request_id] = report

    def _check_capacity(self) -> None:
        cap = self._plan.max_num_requests
        if cap is not None and self._requests.num_active >= cap:
            raise OverloadedError(f"max_num_requests={cap} reached")
        queued = self._plan.max_queued_requests
        if queued is not None and len(self._waiting_ids) >= queued:
            raise OverloadedError(f"max_queued_requests={queued} reached")

    def _drop_waiting(self, request_id: str) -> None:
        self._waiting_remove(request_id)

    # ------------------------------------------------------------------
    # Dense running-set maintenance (swap-with-last)
    # ------------------------------------------------------------------

    def _running_add(self, request_id: str, lifecycle: RequestLifecycle) -> None:
        """Insert or refresh in the running set, preserving ring order."""
        if request_id in self._running:
            self._running[request_id] = lifecycle
            return
        self._running[request_id] = lifecycle
        self._running_index[request_id] = len(self._running_ids)
        self._running_ids.append(request_id)

    def _running_remove(self, request_id: str) -> None:
        """Remove in O(1) via swap-with-last so the decode ring stays dense."""
        if self._running.pop(request_id, None) is None:
            return
        index = self._running_index.pop(request_id)
        last = len(self._running_ids) - 1
        if index != last:
            moved = self._running_ids[last]
            self._running_ids[index] = moved
            self._running_index[moved] = index
        self._running_ids.pop()

    def _drop(self, request_id: str) -> None:
        self._prefilling.pop(request_id, None)
        self._running_remove(request_id)
        self._drop_waiting(request_id)

    def _seq_cap(self) -> int:
        return min(self._plan.max_num_seqs, self._plan.capabilities.max_num_seqs)

    def _sampling_plan(self, slices: Sequence[ScheduledSlice]) -> SamplingPlan:
        """Freeze this step's sampling shape before any forward work exists.

        Without a coordinator the engine keeps the greedy contract: the
        executor may take the argmax fast path. With one, ``plan_for`` pins the
        packed active-row map and reads ``all_greedy``/``any_penalty`` from host
        staging, so no device sync is needed to pick the kernel variant.
        """
        if self._sampling is None:
            num_rows = sum(1 for scheduled in slices if scheduled.sample_last_query)
            return SamplingPlan(num_rows=num_rows, all_greedy=True)
        return self._sampling.plan_for(slices)

    def _prompt_logprob_plan(
        self, slices: Sequence[ScheduledSlice], inputs: Sequence[RequestStepInput]
    ) -> tuple[PromptLogprobSlicePlan, ...]:
        """Resolve which prompt positions each slice's forward can score.

        Scheduler-owned decision (the invariant boundary): position ``p`` is
        scored through hidden row ``p - 1``, so only positions whose
        predecessor is produced by this step are planned. Positions inside the
        prefix-cached region ``[0, num_cached_tokens)`` and position 0 are
        excluded — they surface downstream as no-value markers and are never
        re-forwarded just for reporting.
        """
        if self._sampling is None:
            return ()
        entries: list[PromptLogprobSlicePlan] = []
        for index, (scheduled, value) in enumerate(zip(slices, inputs, strict=True)):
            lifecycle = self._requests.find(scheduled.request_id)
            if lifecycle is None:
                continue
            params = lifecycle.request.sampling
            if params.prompt_logprobs is None:
                continue
            cached = lifecycle.machine.num_cached_tokens
            lo = max(scheduled.query_start, cached + 1, 1)
            hi = min(scheduled.query_end, value.prompt_tokens)
            positions = tuple(range(lo, hi))
            if positions:
                entries.append(
                    PromptLogprobSlicePlan(
                        slice_index=index, k=params.prompt_logprobs, positions=positions
                    )
                )
        return tuple(entries)

    def _next_step_id(self) -> int:
        self._step_id += 1
        return self._step_id

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def plan(self) -> ResolvedSchedulerPlan:
        return self._plan

    @property
    def inflight_ticket(self) -> ExecutionTicket | None:
        """Single-flight compat view: the sole inflight ticket, or None."""
        if len(self._inflight) == 1:
            return next(iter(self._inflight.values()))
        return None

    @property
    def inflight_tickets(self) -> tuple[ExecutionTicket, ...]:
        return tuple(self._inflight.values())

    @property
    def resource_blocked(self) -> bool:
        return self._resource_blocked

    @property
    def num_waiting(self) -> int:
        return len(self._waiting_ids)

    @property
    def num_running(self) -> int:
        return len(self._running) + len(self._prefilling)

    def sequence_for(self, request_id: str) -> SequenceHandle | None:
        return self._sequences.get(request_id)
