from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from threading import RLock

from ayaka.exceptions import (
    InvalidHandleError,
    InvalidStateTransitionError,
    InvariantViolationError,
    SequenceBusyError,
    SequenceCapacityError,
    TransactionClosedError,
)
from ayaka.handles import (
    KVPageHandle,
    KVReservationHandle,
    MemoryTransactionHandle,
    PrefixHandle,
    PrefixMatchHandle,
    SequenceHandle,
    StepMemoryLeaseHandle,
)
from ayaka.kvcache.groups import KVCacheGroup
from ayaka.kvcache.layout import (
    LayoutFeature,
    LayoutFeatureCapability,
    LayoutFeatureCompatibilityError,
    LayoutFeatureIssue,
    LayoutFeatureIssueCode,
    prefix_cache_capability,
)
from ayaka.kvcache.retention.range import retained_page_range
from ayaka.kvcache.storage.ports import KVStorage
from ayaka.memory.allocator import PageAllocator
from ayaka.memory.pressure import (
    MemoryPressureMetrics,
    MemoryPressureResult,
    PreemptionStatus,
    PressureAction,
    PressureStatus,
    SequencePreemptionResult,
)
from ayaka.memory.sequence import (
    GroupedSequenceMemoryState as _SequenceState,
)
from ayaka.memory.sequence import (
    GroupedSequenceSnapshot,
    GroupPageTableEntry,
)
from ayaka.memory.state import (
    LeaseState,
    PageAllocationState,
    ReleaseStatus,
    ReservationFailure,
    TransactionState,
)
from ayaka.memory.transaction import (
    GroupedLeaseRecord as _LeaseRecord,
)
from ayaka.memory.transaction import (
    GroupedReservationRecord as _ReservationRecord,
)
from ayaka.memory.transaction import (
    GroupedTransactionRecord as _TransactionRecord,
)
from ayaka.memory.transaction import (
    GroupReservationPlan as _GroupReservationPlan,
)
from ayaka.memory.transaction import (
    ReservationResult,
)
from ayaka.memory.views import (
    GroupedExecutionMemoryView,
    GroupedLeakReport,
    GroupedMemorySnapshot,
    GroupedSequenceExecutionView,
    GroupKVWriteSlot,
    KVCacheGroupExecutionView,
    KVCacheGroupSnapshot,
)
from ayaka.prefix.identity import PrefixCacheContext
from ayaka.prefix.interface import PrefixLookupResult


@dataclass(slots=True)
class _GroupRuntime:
    """One group's live runtime: descriptor, allocator, padding, storage."""

    descriptor: KVCacheGroup
    allocator: PageAllocator
    padding_page: KVPageHandle
    storage: KVStorage | None


class KVCacheGroupManager:
    """Coordinate ownership across independent storage-compatible cache groups.

    One opaque transaction either reserves every required group page or leaves
    every allocator and committed sequence table unchanged. Epochs stay
    synchronized across groups (divergence is an invariant violation); prefix
    cache and recurrent-state features are structurally rejected.
    """

    def __init__(
        self,
        *,
        cache_groups: tuple[KVCacheGroup, ...],
        max_sequences: int,
        max_sequence_tokens: int,
        storages: Mapping[str, KVStorage] | None = None,
    ) -> None:
        groups = tuple(cache_groups)
        if not groups:
            raise ValueError("at least one KV cache group is required")
        if any(not isinstance(group, KVCacheGroup) for group in groups):
            raise TypeError("cache_groups must contain KVCacheGroup values")
        if len({group.name for group in groups}) != len(groups):
            raise ValueError("cache-group names must be unique")
        # Invariant 12: every layer belongs to exactly one group.
        layers = tuple(layer for group in groups for layer in group.layer_ids)
        if len(set(layers)) != len(layers):
            raise ValueError("a layer cannot belong to more than one cache group")
        if not isinstance(max_sequences, int) or isinstance(max_sequences, bool):
            raise TypeError("max_sequences must be an integer")
        if max_sequences <= 0:
            raise ValueError("max_sequences must be positive")
        if not isinstance(max_sequence_tokens, int) or isinstance(
            max_sequence_tokens,
            bool,
        ):
            raise TypeError("max_sequence_tokens must be an integer")
        if max_sequence_tokens <= 0:
            raise ValueError("max_sequence_tokens must be positive")

        supplied_storages = dict(storages or {})
        unknown_storages = set(supplied_storages) - {group.name for group in groups}
        if unknown_storages:
            raise ValueError(f"storages contain unknown groups: {sorted(unknown_storages)}")
        runtimes = []
        for group in groups:
            if group.storage_spec.capacity_pages < 2:
                raise ValueError(f"group {group.name} needs padding plus at least one usable page")
            storage = supplied_storages.get(group.name)
            if storage is not None:
                if not isinstance(storage, KVStorage):
                    raise TypeError(f"storage for {group.name} must implement KVStorage")
                if storage.spec != group.storage_spec:
                    raise ValueError(
                        f"storage for {group.name} does not match its group specification"
                    )
            # Each group owns an independent allocator and page namespace.
            allocator = PageAllocator(
                total_pages=group.storage_spec.capacity_pages,
                page_size=group.storage_spec.page_size,
            )
            padding_page = allocator.reserve_permanent_page()
            runtimes.append(
                _GroupRuntime(
                    descriptor=group,
                    allocator=allocator,
                    padding_page=padding_page,
                    storage=storage,
                )
            )

        self.cache_groups = groups
        self.max_sequence_tokens = max_sequence_tokens
        self.storages = supplied_storages
        self._group_runtimes = tuple(runtimes)
        self._runtime_by_name = {
            runtime.descriptor.name: runtime for runtime in self._group_runtimes
        }
        # Inline sequence arena (generation-safe, slot-recycled).
        self._sequence_generations = [0] * max_sequences
        self._sequence_states: list[_SequenceState | None] = [None] * max_sequences
        # Reversed so pop() yields slot 0 first: deterministic slot reuse.
        self._free_sequence_indices = list(reversed(range(max_sequences)))
        self._transactions: dict[int, _TransactionRecord] = {}
        self._reservations: dict[int, _ReservationRecord] = {}
        self._leases: dict[int, _LeaseRecord] = {}
        self._next_transaction_index = 0
        self._next_reservation_index = 0
        self._next_lease_index = 0
        self._pressure_reclaim_attempts_total = 0
        self._pressure_reclaim_progress_total = 0
        self._pressure_prefix_eviction_attempts_total = 0
        self._pressure_preemptions_total = 0
        self._lock = RLock()

    @property
    def current_epoch(self) -> int:
        """Common epoch across all group allocators (divergence is fatal).

        Raises:
            InvariantViolationError: if group allocator epochs diverged.
        """
        epochs = {runtime.allocator.current_epoch for runtime in self._group_runtimes}
        if len(epochs) != 1:
            raise InvariantViolationError("cache-group allocator epochs diverged")
        return next(iter(epochs))

    @property
    def page_size(self) -> int:
        """Conservative scheduler-facing page size estimate across all groups.

        Real per-group page sizes remain inside execution views; the scheduler
        uses this single deterministic estimate only for capacity heuristics.
        """

        return min(runtime.descriptor.storage_spec.page_size for runtime in self._group_runtimes)

    @property
    def prefix_capability(self) -> LayoutFeatureCapability:
        """Layout-level prefix capability of the configured groups."""
        return prefix_cache_capability(self.cache_groups) # type: ignore

    def _prefix_reuse_capability(self) -> LayoutFeatureCapability:
        """Report prefix reuse as unimplemented even for compatible layouts.

        A single full-retention MHA group is layout-compatible, but the
        grouped manager has no canonical page chain or prefix cache, so the
        capability degrades to ``PREFIX_CACHE_NOT_IMPLEMENTED`` instead of
        silently promising behavior.
        """
        layout = self.prefix_capability
        if layout.supported:
            return LayoutFeatureCapability(
                feature=LayoutFeature.PREFIX_CACHE,
                issues=(
                    LayoutFeatureIssue(
                        LayoutFeatureIssueCode.PREFIX_CACHE_NOT_IMPLEMENTED,
                        "advanced-layout managers do not implement A6 prefix lookup",
                    ),
                ),
            )
        return layout

    def prefix_cache_reuse_available(self) -> bool:
        return False

    def lookup_prefix(
        self,
        token_ids: tuple[int, ...],
        *,
        context: PrefixCacheContext,
        max_matched_tokens: int | None = None,
    ) -> PrefixLookupResult:
        raise LayoutFeatureCompatibilityError(self._prefix_reuse_capability())

    def discard_prefix_match(self, handle: PrefixMatchHandle) -> None:
        raise LayoutFeatureCompatibilityError(self._prefix_reuse_capability())

    def cache_prefix(
        self,
        sequence: SequenceHandle,
        token_ids: tuple[int, ...],
        *,
        context: PrefixCacheContext,
    ) -> PrefixHandle | None:
        raise LayoutFeatureCompatibilityError(self._prefix_reuse_capability())

    @property
    def pressure_metrics(self) -> MemoryPressureMetrics:
        """Cumulative pressure counters; reclamation totals sum over groups."""
        with self._lock:
            return MemoryPressureMetrics(
                kv_reclaims_total=sum(
                    runtime.allocator.snapshot().reclaimed_pages_total
                    for runtime in self._group_runtimes
                ),
                kv_prefix_evictions_total=0,
                pressure_reclaim_attempts_total=self._pressure_reclaim_attempts_total,
                pressure_reclaim_progress_total=self._pressure_reclaim_progress_total,
                pressure_prefix_eviction_attempts_total=(
                    self._pressure_prefix_eviction_attempts_total
                ),
                pressure_prefix_eviction_progress_total=0,
                pressure_preemptions_total=self._pressure_preemptions_total,
            )

    def create_sequence(self, request_id: str) -> SequenceHandle:
        """Allocate a fresh generation-safe grouped sequence identity.

        Raises:
            SequenceCapacityError: if the arena is full.
        """
        if not request_id:
            raise ValueError("request_id must not be empty")
        with self._lock:
            if not self._free_sequence_indices:
                raise SequenceCapacityError("sequence arena is full")
            index = self._free_sequence_indices.pop()
            # Bumping the generation invalidates every stale handle for the slot.
            generation = self._sequence_generations[index] + 1
            self._sequence_generations[index] = generation
            handle = SequenceHandle(index=index, generation=generation)
            self._sequence_states[index] = _SequenceState(
                handle=handle,
                request_id=request_id,
                group_tables={group.name: [] for group in self.cache_groups},
            )
            return handle

    def get_sequence(self, sequence: SequenceHandle) -> GroupedSequenceSnapshot:
        """Return an immutable snapshot of a grouped sequence."""
        with self._lock:
            state = self._get_sequence_state(sequence)
            return GroupedSequenceSnapshot(
                handle=state.handle,
                request_id=state.request_id,
                committed_tokens=state.committed_tokens,
                version=state.version,
                group_page_tables=tuple(
                    (group.name, tuple(state.group_tables[group.name]))
                    for group in self.cache_groups
                ),
                busy=(
                    state.pending_transaction_index is not None
                    or state.active_lease_index is not None
                ),
                release_requested=state.release_requested,
                blocked_until_epoch=state.blocked_until_epoch,
            )

    def begin_transaction(self, step_id: int) -> MemoryTransactionHandle:
        """Open a tentative-planning transaction for one scheduler step."""
        if not isinstance(step_id, int) or isinstance(step_id, bool):
            raise TypeError("step_id must be an integer")
        if step_id < 0:
            raise ValueError("step_id must be non-negative")
        with self._lock:
            index = self._next_transaction_index
            self._next_transaction_index += 1
            handle = MemoryTransactionHandle(index=index, generation=1, step_id=step_id)
            self._transactions[index] = _TransactionRecord(handle=handle)
            return handle

    def try_reserve(
        self,
        transaction: MemoryTransactionHandle,
        sequence: SequenceHandle,
        num_new_tokens: int,
        *,
        prefix_match: object | None = None,
    ) -> ReservationResult:
        """Atomically reserve append pages in every cache group.

        Every group needs one page per new logical block the append crosses.
        All groups are checked for capacity before anything is allocated; a
        later failure rolls back every group's allocation, so the reservation
        is all-or-nothing across groups.

        Args:
            transaction: Open transaction handle.
            sequence: Sequence to extend.
            num_new_tokens: Positive token count to plan for.
            prefix_match: Rejected for grouped managers (no A6 prefix path).

        Returns:
            Success (with an opaque reservation handle) or a structured
            failure.
        """

        with self._lock:
            tx = self._get_transaction(transaction)
            if tx.state is not TransactionState.OPEN:
                raise TransactionClosedError(f"transaction {transaction} is not open")
            if prefix_match is not None:
                return ReservationResult.failure(ReservationFailure.PREFIX_NOT_RESIDENT)
            if not isinstance(num_new_tokens, int) or isinstance(num_new_tokens, bool):
                raise TypeError("num_new_tokens must be an integer")
            if num_new_tokens <= 0:
                return ReservationResult.failure(ReservationFailure.INVALID_TOKEN_COUNT)
            try:
                state = self._get_sequence_state(sequence)
            except InvalidHandleError:
                return ReservationResult.failure(ReservationFailure.SEQUENCE_INVALID)
            if (
                state.pending_transaction_index is not None
                or state.active_lease_index is not None
                or state.release_requested
                or state.blocked_until_epoch > self.current_epoch
            ):
                return ReservationResult.failure(ReservationFailure.SEQUENCE_BUSY)

            final_tokens = state.committed_tokens + num_new_tokens
            if final_tokens > self.max_sequence_tokens:
                return ReservationResult.failure(ReservationFailure.REQUEST_TOO_LARGE)

            # Determine each group's missing write blocks before allocating so
            # capacity can be validated across every group up front.
            missing_by_group: dict[str, tuple[int, ...]] = {}
            for runtime in self._group_runtimes:
                runtime.allocator.reclaim_completed()
                existing_blocks = {
                    entry.logical_block for entry in state.group_tables[runtime.descriptor.name]
                }
                page_size = runtime.descriptor.storage_spec.page_size
                # dict.fromkeys dedups the block list while preserving order.
                write_blocks = tuple(
                    dict.fromkeys(
                        position // page_size
                        for position in range(state.committed_tokens, final_tokens)
                    )
                )
                missing = tuple(block for block in write_blocks if block not in existing_blocks)
                missing_by_group[runtime.descriptor.name] = missing
            required_pages = sum(len(pages) for pages in missing_by_group.values())
            # Pre-flight capacity check: fail before allocating anywhere.
            for runtime in self._group_runtimes:
                missing = missing_by_group[runtime.descriptor.name]
                if len(missing) > runtime.allocator.available_pages():
                    return ReservationResult.failure(
                        ReservationFailure.NO_CAPACITY,
                        required_pages=required_pages,
                    )

            allocated_by_group: dict[str, tuple[KVPageHandle, ...]] = {}
            try:
                for runtime in self._group_runtimes:
                    missing = missing_by_group[runtime.descriptor.name]
                    allocated = runtime.allocator.allocate(len(missing))
                    if allocated is None:
                        # Unreachable after the pre-flight check under the lock.
                        raise InvariantViolationError(
                            "cache-group capacity changed while manager lock was held"
                        )
                    allocated_by_group[runtime.descriptor.name] = allocated
                group_plans = tuple(
                    self._plan_group_append(
                        runtime,
                        tuple(state.group_tables[runtime.descriptor.name]),
                        missing_by_group[runtime.descriptor.name],
                        allocated_by_group[runtime.descriptor.name],
                        base_committed_tokens=state.committed_tokens,
                        final_tokens=final_tokens,
                    )
                    for runtime in self._group_runtimes
                )
            except Exception:
                # Roll back every already-allocated group in reverse order.
                for runtime in reversed(self._group_runtimes):
                    allocated = allocated_by_group.get(runtime.descriptor.name, ())
                    if allocated:
                        runtime.allocator.rollback_reserved(allocated)
                raise

            index = self._next_reservation_index
            self._next_reservation_index += 1
            reservation_handle = KVReservationHandle(
                index=index,
                generation=1,
                step_id=transaction.step_id,
            )
            self._reservations[index] = _ReservationRecord(
                handle=reservation_handle,
                sequence=sequence,
                base_sequence_version=state.version,
                base_committed_tokens=state.committed_tokens,
                num_new_tokens=num_new_tokens,
                group_plans=group_plans,
            )
            tx.reservation_handles.append(reservation_handle)
            state.pending_transaction_index = transaction.index
            return ReservationResult.success(
                reservation_handle,
                required_pages=required_pages,
                allocated_pages=required_pages,
            )

    def rollback_transaction(self, transaction: MemoryTransactionHandle) -> None:
        """Undo every grouped reservation in the transaction."""
        with self._lock:
            tx = self._get_transaction(transaction)
            self._rollback_transaction_locked(tx)

    def prepare_step(
        self,
        transaction: MemoryTransactionHandle,
    ) -> StepMemoryLeaseHandle:
        """Freeze a valid grouped plan into an execution lease.

        Validates every sequence's state against its reservation baseline and
        rolls the transaction back when anything changed or release was
        requested.
        """
        with self._lock:
            tx = self._get_transaction(transaction)
            if not tx.reservation_handles:
                raise InvalidStateTransitionError("cannot prepare an empty transaction")
            records = [self._get_reservation(handle) for handle in tx.reservation_handles]
            for record in records:
                state = self._get_sequence_state(record.sequence)
                if state.release_requested:
                    self._rollback_transaction_locked(tx)
                    raise InvalidStateTransitionError(
                        "a release was requested while the transaction was open"
                    )
                if (
                    state.pending_transaction_index != transaction.index
                    or state.active_lease_index is not None
                    or state.version != record.base_sequence_version
                    or state.committed_tokens != record.base_committed_tokens
                ):
                    self._rollback_transaction_locked(tx)
                    raise InvalidStateTransitionError(
                        "sequence changed while the group plan was tentative"
                    )

            index = self._next_lease_index
            self._next_lease_index += 1
            lease_handle = StepMemoryLeaseHandle(
                index=index,
                generation=1,
                step_id=transaction.step_id,
            )
            self._leases[index] = _LeaseRecord(
                handle=lease_handle,
                transaction=transaction,
                reservation_handles=tuple(tx.reservation_handles),
            )
            for record in records:
                state = self._get_sequence_state(record.sequence)
                state.pending_transaction_index = None
                state.active_lease_index = index
            tx.state = TransactionState.PREPARED
            self._transactions.pop(transaction.index)
            return lease_handle

    def build_execution_view(
        self,
        lease_handle: StepMemoryLeaseHandle,
    ) -> GroupedExecutionMemoryView:
        """Materialize a prepared grouped lease into execution metadata.

        Computes each group's retention window and attention window for the
        step, resolving the padding page for evicted blocks.

        Args:
            lease_handle: Prepared lease handle.

        Returns:
            The immutable grouped execution view.

        Raises:
            InvalidHandleError: if the lease handle is stale.
        """
        with self._lock:
            lease = self._get_lease(lease_handle)
            sequence_views = []
            for reservation_handle in lease.reservation_handles:
                record = self._get_reservation(reservation_handle)
                group_views = []
                for plan in record.group_plans:
                    runtime = self._runtime_by_name[plan.group_name]
                    descriptor = runtime.descriptor
                    page_size = descriptor.storage_spec.page_size
                    final_tokens = record.base_committed_tokens + record.num_new_tokens
                    # The full live interval after the step (post-retention).
                    retained_tokens = descriptor.retention.required_token_range(
                        descriptor.layer_ids[0],
                        final_tokens,
                    )
                    # The interval already readable for the first query of the
                    # step (pre-append length), used as attention start.
                    first_query_length = min(
                        final_tokens,
                        record.base_committed_tokens + 1,
                    )
                    attention_tokens = descriptor.retention.required_token_range(
                        descriptor.layer_ids[0],
                        first_query_length,
                    )
                    padding = runtime.allocator.physical_id(runtime.padding_page).value
                    group_views.append(
                        KVCacheGroupExecutionView(
                            group_name=descriptor.name,
                            layer_ids=descriptor.layer_ids,
                            storage_kind=descriptor.storage_spec.kind.value,
                            dtype=descriptor.storage_spec.dtype,
                            page_size=page_size,
                            logical_blocks=tuple(
                                entry.logical_block for entry in plan.planned_page_table
                            ),
                            block_table=tuple(
                                runtime.allocator.physical_id(entry.page).value
                                for entry in plan.planned_page_table
                            ),
                            write_slots=plan.write_slots,
                            attention_token_start=attention_tokens.start,
                            attention_token_stop=final_tokens,
                            retained_token_start=retained_tokens.start,
                            retained_token_stop=retained_tokens.stop,
                            padding_page=padding,
                            padding_slot=padding * page_size,
                        )
                    )
                sequence_views.append(
                    GroupedSequenceExecutionView(
                        sequence=record.sequence,
                        reservation=record.handle,
                        base_committed_tokens=record.base_committed_tokens,
                        num_reserved_tokens=record.num_new_tokens,
                        groups=tuple(group_views),
                    )
                )
            return GroupedExecutionMemoryView(
                step_id=lease_handle.step_id,
                lease=lease_handle,
                sequences=tuple(sequence_views),
            )

    def mark_step_in_flight(self, lease_handle: StepMemoryLeaseHandle) -> None:
        """Acquire transient ownership in every touched group before launch.

        Args:
            lease_handle: A PREPARED lease.

        Raises:
            InvalidStateTransitionError: if the lease is not PREPARED.
        """
        with self._lock:
            lease = self._get_lease(lease_handle)
            if lease.state is not LeaseState.PREPARED:
                raise InvalidStateTransitionError("only a prepared step can be launched")
            for runtime in self._group_runtimes:
                runtime.allocator.mark_inflight(
                    self._lease_touched_pages(lease, runtime.descriptor.name)
                )
            lease.state = LeaseState.IN_FLIGHT

    def complete_step(
        self,
        lease_handle: StepMemoryLeaseHandle,
        *,
        written_tokens: Mapping[KVReservationHandle, int] | None = None,
    ) -> None:
        """Publish KV metadata after a successful grouped execution step.

        Commits each group's reserved pages to LIVE, publishes valid-token
        counts, advances sequence committed length/version, releases in-flight
        ownership, and finally prunes pages the group's retention policy no
        longer needs (staging them for deferred free).

        Args:
            lease_handle: An IN_FLIGHT lease.
            written_tokens: Optional per-reservation written-token report.

        Raises:
            InvalidStateTransitionError: if the lease is not IN_FLIGHT or the
                written-token report disagrees with the plan.
        """
        with self._lock:
            lease = self._get_lease(lease_handle)
            if lease.state is not LeaseState.IN_FLIGHT:
                raise InvalidStateTransitionError("step must be in flight before completion")
            records = [self._get_reservation(handle) for handle in lease.reservation_handles]
            self._validate_written_counts(records, written_tokens)
            self._validate_lease_sequences(lease, records)

            # Commit new pages per group (RESERVED -> LIVE).
            for runtime in self._group_runtimes:
                new_pages = tuple(
                    page
                    for record in records
                    for plan in record.group_plans
                    if plan.group_name == runtime.descriptor.name
                    for page in plan.allocated_pages
                )
                runtime.allocator.commit_reserved(new_pages)
            for record in records:
                final_tokens = record.base_committed_tokens + record.num_new_tokens
                state = self._get_sequence_state(record.sequence)
                for plan in record.group_plans:
                    runtime = self._runtime_by_name[plan.group_name]
                    for entry in plan.planned_page_table:
                        runtime.allocator.set_valid_tokens(entry.page, entry.valid_tokens)
                state.committed_tokens = final_tokens
                state.version += 1

            for runtime in self._group_runtimes:
                runtime.allocator.unmark_inflight(
                    self._lease_touched_pages(lease, runtime.descriptor.name)
                )

            # Retention pruning per group: blocks outside the live window drop
            # their request refs and enter deferred free at the current epoch.
            for record in records:
                final_tokens = record.base_committed_tokens + record.num_new_tokens
                state = self._get_sequence_state(record.sequence)
                for plan in record.group_plans:
                    runtime = self._runtime_by_name[plan.group_name]
                    retained_blocks = set(
                        retained_page_range(
                            runtime.descriptor.retention,
                            layer_id=runtime.descriptor.layer_ids[0],
                            sequence_length=final_tokens,
                            page_size=runtime.descriptor.storage_spec.page_size,
                        )
                    )
                    retained_entries = []
                    for entry in plan.planned_page_table:
                        if entry.logical_block in retained_blocks:
                            retained_entries.append(entry)
                        else:
                            runtime.allocator.release_request_ref(
                                entry.page,
                                safe_epoch=self.current_epoch,
                            )
                    state.group_tables[plan.group_name] = retained_entries
                state.active_lease_index = None
            lease.state = LeaseState.COMPLETED
            self._finish_lease_locked(lease)

    def abort_prepared_step(self, lease_handle: StepMemoryLeaseHandle) -> None:
        """Cancel a grouped step before launch; reservations roll back.

        Args:
            lease_handle: A PREPARED lease.

        Raises:
            InvalidStateTransitionError: if the lease is not PREPARED.
        """
        with self._lock:
            lease = self._get_lease(lease_handle)
            if lease.state is not LeaseState.PREPARED:
                raise InvalidStateTransitionError(
                    "only a not-yet-launched step can be aborted immediately"
                )
            records = [self._get_reservation(handle) for handle in lease.reservation_handles]
            for record in records:
                for plan in record.group_plans:
                    self._runtime_by_name[plan.group_name].allocator.rollback_reserved(
                        plan.allocated_pages
                    )
                self._get_sequence_state(record.sequence).active_lease_index = None
            lease.state = LeaseState.ABORTED
            self._finish_lease_locked(lease)

    def fail_in_flight_step(
        self,
        lease_handle: StepMemoryLeaseHandle,
        *,
        safe_epoch: int,
    ) -> None:
        """Fail a launched grouped step; new pages abandon to deferred free.

        Partially written bytes are never trusted: every group's new pages are
        abandoned with ``safe_epoch`` and each sequence is blocked until that
        epoch.

        Args:
            lease_handle: An IN_FLIGHT lease.
            safe_epoch: Epoch whose completion makes abandoned pages reusable.

        Raises:
            ValueError: if ``safe_epoch`` precedes the completed epoch.
            InvalidStateTransitionError: if the lease is not IN_FLIGHT.
        """
        if safe_epoch < self.current_epoch:
            raise ValueError("safe_epoch cannot precede the completed epoch")
        with self._lock:
            lease = self._get_lease(lease_handle)
            if lease.state is not LeaseState.IN_FLIGHT:
                raise InvalidStateTransitionError("only an in-flight step can fail in flight")
            records = [self._get_reservation(handle) for handle in lease.reservation_handles]
            for runtime in self._group_runtimes:
                new_pages = tuple(
                    page
                    for record in records
                    for plan in record.group_plans
                    if plan.group_name == runtime.descriptor.name
                    for page in plan.allocated_pages
                )
                runtime.allocator.abandon_reserved(new_pages, safe_epoch=safe_epoch)
                runtime.allocator.unmark_inflight(
                    self._lease_touched_pages(lease, runtime.descriptor.name)
                )
            for record in records:
                state = self._get_sequence_state(record.sequence)
                state.active_lease_index = None
                state.blocked_until_epoch = max(state.blocked_until_epoch, safe_epoch)
                state.version += 1
            lease.state = LeaseState.FAILED
            self._finish_lease_locked(lease, failure_safe_epoch=safe_epoch)

    def release_sequence(
        self,
        sequence: SequenceHandle,
        *,
        safe_epoch: int | None = None,
    ) -> ReleaseStatus:
        """Release request ownership now or after the sequence's active step.

        Args:
            sequence: The sequence to release.
            safe_epoch: Epoch whose completion makes released pages reusable.

        Returns:
            ``RELEASED`` (dropped now) or ``DEFERRED`` (queued behind the step).
        """
        epoch = self.current_epoch if safe_epoch is None else safe_epoch
        if epoch < self.current_epoch:
            raise ValueError("safe_epoch cannot precede the completed epoch")
        with self._lock:
            state = self._get_sequence_state(sequence)
            if state.pending_transaction_index is not None or state.active_lease_index is not None:
                state.release_requested = True
                state.release_safe_epoch = max(state.release_safe_epoch, epoch)
                return ReleaseStatus.DEFERRED
            self._release_sequence_locked(sequence, safe_epoch=epoch)
            return ReleaseStatus.RELEASED

    def preempt_sequence(
        self,
        sequence: SequenceHandle,
        *,
        safe_epoch: int | None = None,
    ) -> SequencePreemptionResult:
        """Release every group's request-owned pages while preserving identity.

        Busy sequences yield ``BUSY``; sequences without resident KV yield a
        harmless ``NO_RESIDENT_STATE``.

        Args:
            sequence: The sequence to preempt.
            safe_epoch: Epoch whose completion makes released pages reusable.

        Returns:
            A logical preemption outcome.
        """
        epoch = self.current_epoch if safe_epoch is None else safe_epoch
        if epoch < self.current_epoch:
            raise ValueError("safe_epoch cannot precede the completed epoch")
        with self._lock:
            state = self._get_sequence_state(sequence)
            before = self.snapshot()
            if (
                state.pending_transaction_index is not None
                or state.active_lease_index is not None
                or state.release_requested
                or state.blocked_until_epoch > self.current_epoch
            ):
                return SequencePreemptionResult(
                    sequence=sequence,
                    status=PreemptionStatus.BUSY,
                    released_tokens=0,
                    released_request_pages=0,
                    pages_reclaimed=0,
                    pages_deferred=0,
                    free_pages_before=before.free_pages,
                    free_pages_after=before.free_pages,
                )
            released_pages = sum(len(entries) for entries in state.group_tables.values())
            if released_pages == 0:
                if state.committed_tokens:
                    raise InvariantViolationError(
                        "group tables are empty but committed tokens remain"
                    )
                return SequencePreemptionResult(
                    sequence=sequence,
                    status=PreemptionStatus.NO_RESIDENT_STATE,
                    released_tokens=0,
                    released_request_pages=0,
                    pages_reclaimed=0,
                    pages_deferred=0,
                    free_pages_before=before.free_pages,
                    free_pages_after=before.free_pages,
                )
            released_tokens = state.committed_tokens
            for runtime in self._group_runtimes:
                entries = state.group_tables[runtime.descriptor.name]
                for entry in entries:
                    runtime.allocator.release_request_ref(entry.page, safe_epoch=epoch)
                entries.clear()
            state.committed_tokens = 0
            state.version += 1
            pending = self.snapshot()
            reclaimed = sum(
                runtime.allocator.reclaim_completed() for runtime in self._group_runtimes
            )
            after = self.snapshot()
            self._pressure_preemptions_total += 1
            return SequencePreemptionResult(
                sequence=sequence,
                status=PreemptionStatus.RELEASED,
                released_tokens=released_tokens,
                released_request_pages=released_pages,
                pages_reclaimed=reclaimed,
                pages_deferred=max(
                    pending.deferred_free_pages - before.deferred_free_pages,
                    0,
                ),
                free_pages_before=before.free_pages,
                free_pages_after=after.free_pages,
            )

    def advance_epoch(self, completed_epoch: int) -> int:
        """Advance every group allocator's epoch together (invariant 11).

        Args:
            completed_epoch: Newest completed GPU step epoch; must not regress.

        Returns:
            Total pages reclaimed across all groups.
        """
        with self._lock:
            return sum(
                runtime.allocator.advance_epoch(completed_epoch) for runtime in self._group_runtimes
            )

    def reclaim_deferred(self) -> MemoryPressureResult:
        """Reclaim completed deferred pages across every group.

        Repeated calls without an epoch advance return ``NO_PROGRESS``.
        """
        with self._lock:
            before = self.snapshot()
            self._pressure_reclaim_attempts_total += 1
            reclaimed = sum(
                runtime.allocator.reclaim_completed() for runtime in self._group_runtimes
            )
            after = self.snapshot()
            if reclaimed:
                self._pressure_reclaim_progress_total += 1
            return MemoryPressureResult(
                action=PressureAction.RECLAIM_DEFERRED,
                status=PressureStatus.PROGRESSED if reclaimed else PressureStatus.NO_PROGRESS,
                requested_pages=0,
                cache_pages_evicted=0,
                pages_reclaimed=reclaimed,
                free_pages_before=before.free_pages,
                free_pages_after=after.free_pages,
                deferred_pages_before=before.deferred_free_pages,
                deferred_pages_after=after.deferred_free_pages,
                cached_pages_before=0,
                cached_pages_after=0,
            )

    def evict_prefixes_for_pressure(self, required_pages: int) -> MemoryPressureResult:
        """Grouped managers have no prefix cache to evict.

        Returns a structured ``NO_PROGRESS`` result so pressure handling stays
        uniform across manager implementations.
        """
        if not isinstance(required_pages, int) or isinstance(required_pages, bool):
            raise TypeError("required_pages must be an integer")
        if required_pages < 0:
            raise ValueError("required_pages must be non-negative")
        with self._lock:
            before = self.snapshot()
            self._pressure_prefix_eviction_attempts_total += 1
            return MemoryPressureResult(
                action=PressureAction.EVICT_PREFIX,
                status=PressureStatus.NO_PROGRESS,
                requested_pages=required_pages,
                cache_pages_evicted=0,
                pages_reclaimed=0,
                free_pages_before=before.free_pages,
                free_pages_after=before.free_pages,
                deferred_pages_before=before.deferred_free_pages,
                deferred_pages_after=before.deferred_free_pages,
                cached_pages_before=0,
                cached_pages_after=0,
            )

    def require_prefix_cache_supported(self) -> None:
        """Raise when the configured groups cannot use A6 prefix sharing.

        Raises:
            LayoutFeatureCompatibilityError: when prefix reuse is unsupported.
        """
        self.prefix_capability.require_supported()

    def snapshot(self) -> GroupedMemorySnapshot:
        """Return per-group accounting plus cross-group totals."""
        with self._lock:
            group_snapshots = []
            for runtime in self._group_runtimes:
                allocator = runtime.allocator.snapshot()
                group_snapshots.append(
                    KVCacheGroupSnapshot(
                        group_name=runtime.descriptor.name,
                        total_pages=allocator.total_pages,
                        usable_pages=allocator.usable_pages,
                        free_pages=allocator.free_pages,
                        reserved_pages=allocator.reserved_pages,
                        live_pages=allocator.live_pages,
                        deferred_free_pages=allocator.reclaim_pending_pages,
                        permanent_pages=allocator.permanent_pages,
                        request_owned_pages=allocator.request_owned_pages,
                        inflight_pages=allocator.inflight_pages,
                        reclaimed_pages_total=allocator.reclaimed_pages_total,
                        page_size=runtime.descriptor.storage_spec.page_size,
                        bytes_per_page=runtime.descriptor.storage_spec.bytes_per_page,
                    )
                )
            live_states = tuple(state for state in self._sequence_states if state is not None)
            return GroupedMemorySnapshot(
                groups=tuple(group_snapshots),
                committed_tokens=sum(state.committed_tokens for state in live_states),
                live_sequences=len(live_states),
                open_transactions=len(self._transactions),
                active_leases=len(self._leases),
            )

    def leak_report(self) -> GroupedLeakReport:
        """Return the full grouped accounting report for leak detection."""
        with self._lock:
            return GroupedLeakReport(
                snapshot=self.snapshot(),
                live_sequence_handles=tuple(
                    state.handle for state in self._sequence_states if state is not None
                ),
                open_transaction_ids=tuple(sorted(self._transactions)),
                active_lease_ids=tuple(sorted(self._leases)),
            )

    def assert_invariants(self) -> None:
        """Debug-only cross-component grouped invariant check.

        Verifies per-group allocator/padding invariants, free-slot
        consistency, that every group page table exactly matches its retention
        window, that allocator request refs match sequence ownership, and that
        reservation accounting matches every group's reserved pages.
        """
        with self._lock:
            for runtime in self._group_runtimes:
                runtime.allocator.assert_invariants()
                padding = runtime.allocator.get_meta(runtime.padding_page)
                if padding.allocation_state is not PageAllocationState.PERMANENT:
                    raise InvariantViolationError(
                        f"group {runtime.descriptor.name} lost its padding page"
                    )
            free_set = set(self._free_sequence_indices)
            if len(free_set) != len(self._free_sequence_indices):
                raise InvariantViolationError("duplicate free grouped-sequence slot")

            expected_refs: dict[tuple[str, KVPageHandle], int] = {}
            for index, state in enumerate(self._sequence_states):
                if state is None:
                    if index not in free_set:
                        raise InvariantViolationError(
                            "empty grouped-sequence slot is missing from free list"
                        )
                    continue
                if index in free_set:
                    raise InvariantViolationError("live grouped sequence appears in the free list")
                if state.handle.generation != self._sequence_generations[index]:
                    raise InvariantViolationError("grouped sequence generation mismatch")
                for runtime in self._group_runtimes:
                    name = runtime.descriptor.name
                    entries = state.group_tables[name]
                    logical_blocks = tuple(entry.logical_block for entry in entries)
                    # A committed group table must equal the retention window:
                    # evicted blocks were pruned at complete_step time.
                    expected_blocks = tuple(
                        retained_page_range(
                            runtime.descriptor.retention,
                            layer_id=runtime.descriptor.layer_ids[0],
                            sequence_length=state.committed_tokens,
                            page_size=runtime.descriptor.storage_spec.page_size,
                        )
                    )
                    if logical_blocks != expected_blocks:
                        raise InvariantViolationError(
                            f"group {name} page table disagrees with retention policy"
                        )
                    for entry in entries:
                        meta = runtime.allocator.get_meta(entry.page)
                        if meta.allocation_state is not PageAllocationState.LIVE:
                            raise InvariantViolationError(
                                "group page table refers to a non-live page"
                            )
                        if meta.valid_tokens != entry.valid_tokens:
                            raise InvariantViolationError(
                                "group page-table token count disagrees with allocator"
                            )
                        key = (name, entry.page)
                        expected_refs[key] = expected_refs.get(key, 0) + 1
                if state.pending_transaction_index is not None:
                    tx = self._transactions.get(state.pending_transaction_index)
                    if tx is None or not any(
                        self._get_reservation(handle).sequence == state.handle
                        for handle in tx.reservation_handles
                    ):
                        raise InvariantViolationError(
                            "sequence points to a missing grouped transaction"
                        )
                if state.active_lease_index is not None:
                    lease = self._leases.get(state.active_lease_index)
                    if lease is None or not any(
                        self._get_reservation(handle).sequence == state.handle
                        for handle in lease.reservation_handles
                    ):
                        raise InvariantViolationError("sequence points to a missing grouped lease")

            # Cross-check request refs against the sequence tables we counted.
            for (group_name, page), expected in expected_refs.items():
                meta = self._runtime_by_name[group_name].allocator.get_meta(page)
                if meta.request_refs != expected:
                    raise InvariantViolationError(
                        "group page request refs disagree with sequence ownership"
                    )

            # Reservation accounting must match reserved pages per group.
            for runtime in self._group_runtimes:
                reserved = {
                    page
                    for record in self._reservations.values()
                    for plan in record.group_plans
                    if plan.group_name == runtime.descriptor.name
                    for page in plan.allocated_pages
                }
                if len(reserved) != runtime.allocator.snapshot().reserved_pages:
                    raise InvariantViolationError(
                        f"group {runtime.descriptor.name} reservation accounting differs"
                    )

    def _plan_group_append(
        self,
        runtime: _GroupRuntime,
        existing_entries: tuple[GroupPageTableEntry, ...],
        missing_blocks: tuple[int, ...],
        allocated_pages: tuple[KVPageHandle, ...],
        *,
        base_committed_tokens: int,
        final_tokens: int,
    ) -> _GroupReservationPlan:
        """Build one group's post-append table and write slots.

        Maps each missing logical block to its newly reserved page, walks the
        append token range to emit write slots, and derives each planned
        entry's valid-token count as ``min(page_size, final - block*page_size)``
        (the tail block is partial).

        Returns:
            The per-group reservation plan.
        """
        if len(missing_blocks) != len(allocated_pages):
            raise InvariantViolationError("group allocation does not match missing blocks")
        # Existing blocks keep their committed pages; missing blocks get the
        # newly reserved ones.
        page_by_block = {entry.logical_block: entry.page for entry in existing_entries}
        page_by_block.update(dict(zip(missing_blocks, allocated_pages, strict=True)))
        page_size = runtime.descriptor.storage_spec.page_size
        write_slots = []
        for logical_position in range(base_committed_tokens, final_tokens):
            logical_block = logical_position // page_size
            page_offset = logical_position % page_size
            try:
                page = page_by_block[logical_block]
            except KeyError as error:
                raise InvariantViolationError(
                    "group append plan has no page for a write block"
                ) from error
            physical_page = runtime.allocator.physical_id(page).value
            write_slots.append(
                GroupKVWriteSlot(
                    logical_position=logical_position,
                    logical_block=logical_block,
                    physical_page=physical_page,
                    page_offset=page_offset,
                    flat_slot=physical_page * page_size + page_offset,
                )
            )
        planned_entries = tuple(
            GroupPageTableEntry(
                logical_block=logical_block,
                page=page,
                valid_tokens=min(page_size, final_tokens - logical_block * page_size),
            )
            for logical_block, page in sorted(page_by_block.items())
        )
        return _GroupReservationPlan(
            group_name=runtime.descriptor.name,
            planned_page_table=planned_entries,
            allocated_pages=allocated_pages,
            write_slots=tuple(write_slots),
        )

    def _rollback_transaction_locked(self, tx: _TransactionRecord) -> None:
        """Undo an OPEN grouped transaction under lock.

        Rolls back every group's allocations, clears sequence
        pending-transaction markers, drops reservation records, and honors any
        deferred release request that arrived meanwhile.
        """
        if tx.state is not TransactionState.OPEN:
            raise TransactionClosedError(f"transaction {tx.handle} is not open")
        affected = []
        for reservation_handle in tx.reservation_handles:
            record = self._get_reservation(reservation_handle)
            for plan in record.group_plans:
                self._runtime_by_name[plan.group_name].allocator.rollback_reserved(
                    plan.allocated_pages
                )
            state = self._get_sequence_state(record.sequence)
            if state.pending_transaction_index == tx.handle.index:
                state.pending_transaction_index = None
            affected.append(record.sequence)
            self._reservations.pop(reservation_handle.index)
        tx.state = TransactionState.ROLLED_BACK
        self._transactions.pop(tx.handle.index)
        for sequence in affected:
            state = self._get_sequence_state(sequence)
            if state.release_requested:
                self._release_sequence_locked(
                    sequence,
                    safe_epoch=state.release_safe_epoch,
                )

    def _finish_lease_locked(
        self,
        lease: _LeaseRecord,
        *,
        failure_safe_epoch: int | None = None,
    ) -> None:
        """Tear down a terminal grouped lease and honor deferred releases."""
        affected = []
        for reservation_handle in lease.reservation_handles:
            record = self._get_reservation(reservation_handle)
            affected.append(record.sequence)
            self._reservations.pop(reservation_handle.index)
        self._leases.pop(lease.handle.index)
        for sequence in affected:
            state = self._get_sequence_state(sequence)
            if state.release_requested:
                self._release_sequence_locked(
                    sequence,
                    safe_epoch=max(
                        state.release_safe_epoch,
                        failure_safe_epoch or self.current_epoch,
                    ),
                )

    def _release_sequence_locked(
        self,
        sequence: SequenceHandle,
        *,
        safe_epoch: int,
    ) -> None:
        """Drop every group's request refs and recycle the sequence slot.

        Requires the sequence to be idle; each group's entries are released
        with the given safe epoch before the slot returns to the arena.
        """
        state = self._get_sequence_state(sequence)
        if state.pending_transaction_index is not None or state.active_lease_index is not None:
            raise SequenceBusyError("cannot immediately release a busy sequence")
        for runtime in self._group_runtimes:
            entries = state.group_tables[runtime.descriptor.name]
            for entry in entries:
                runtime.allocator.release_request_ref(entry.page, safe_epoch=safe_epoch)
            entries.clear()
        state.committed_tokens = 0
        state.version += 1
        state.release_requested = False
        self._sequence_states[sequence.index] = None
        self._free_sequence_indices.append(sequence.index)

    def _validate_lease_sequences(
        self,
        lease: _LeaseRecord,
        records: Sequence[_ReservationRecord],
    ) -> None:
        """Reject completion when any sequence changed during the grouped lease."""
        for record in records:
            state = self._get_sequence_state(record.sequence)
            if (
                state.active_lease_index != lease.handle.index
                or state.version != record.base_sequence_version
                or state.committed_tokens != record.base_committed_tokens
            ):
                raise InvalidStateTransitionError(
                    "sequence changed while its grouped lease was active"
                )

    @staticmethod
    def _validate_written_counts(
        records: Sequence[_ReservationRecord],
        written_tokens: Mapping[KVReservationHandle, int] | None,
    ) -> None:
        """Validate an optional written-token report (A1: all tokens written)."""
        if written_tokens is None:
            return
        expected = {record.handle for record in records}
        if set(written_tokens) != expected:
            raise InvalidStateTransitionError(
                "written-token report does not match grouped reservations"
            )
        if any(written_tokens[record.handle] != record.num_new_tokens for record in records):
            raise InvalidStateTransitionError("every grouped reservation token must be written")

    def _lease_touched_pages(
        self,
        lease: _LeaseRecord,
        group_name: str,
    ) -> tuple[KVPageHandle, ...]:
        """All unique pages one group's reservations touch, deduplicated."""
        pages: list[KVPageHandle] = []
        for reservation_handle in lease.reservation_handles:
            record = self._get_reservation(reservation_handle)
            for plan in record.group_plans:
                if plan.group_name == group_name:
                    pages.extend(plan.touched_pages)
        return tuple(dict.fromkeys(pages))

    def _get_sequence_state(self, handle: SequenceHandle) -> _SequenceState:
        """Resolve a sequence handle, rejecting stale generations."""
        if handle.index >= len(self._sequence_states):
            raise InvalidHandleError(f"sequence index {handle.index} is out of range")
        state = self._sequence_states[handle.index]
        if state is None or state.handle.generation != handle.generation:
            raise InvalidHandleError(f"stale sequence handle: {handle}")
        return state

    def _get_transaction(
        self,
        handle: MemoryTransactionHandle,
    ) -> _TransactionRecord:
        """Resolve a transaction handle, rejecting stale generations."""
        record = self._transactions.get(handle.index)
        if record is None or record.handle != handle:
            raise InvalidHandleError(f"stale transaction handle: {handle}")
        return record

    def _get_reservation(
        self,
        handle: KVReservationHandle,
    ) -> _ReservationRecord:
        """Resolve a reservation handle, rejecting stale generations."""
        record = self._reservations.get(handle.index)
        if record is None or record.handle != handle:
            raise InvalidHandleError(f"stale reservation handle: {handle}")
        return record

    def _get_lease(self, handle: StepMemoryLeaseHandle) -> _LeaseRecord:
        """Resolve a lease handle, rejecting stale generations."""
        record = self._leases.get(handle.index)
        if record is None or record.handle != handle:
            raise InvalidHandleError(f"stale step-memory lease: {handle}")
        return record
