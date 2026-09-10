"""Scheduler-independent paged runtime-memory manager

Implements the full reservation/execution/reclamation lifecycle for the
homogeneous MHA/GQA path: sequences, page allocation, prefix reuse, deferred
free, and leak accounting, all behind opaque handles and immutable results.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from math import ceil
from threading import RLock

from ayaka.exceptions import (
    InvalidHandleError,
    InvalidStateTransitionError,
    InvariantViolationError,
    SequenceBusyError,
    StorageUnavailableError,
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
    PageTableEntry,
    SequenceArena,
    SequenceMemorySnapshot,
    SequencePageTable,
)
from ayaka.memory.state import LeaseState, PageAllocationState, ReleaseStatus, ReservationFailure
from ayaka.memory.tiering import (
    HostKVStorage,
    HostTierSnapshot,
    PromotedBlock,
    TieringConfig,
    TierManager,
    TransferEngine,
    build_transfer_engine,
)
from ayaka.memory.transaction import (
    LeaseRecord,
    LifecycleTransitions,
    ReservationRecord,
    ReservationResult,
    TransactionOrchestrator,
    TransactionRecord,
)
from ayaka.memory.views import (
    CacheView,
    ExecutionMemoryView,
    KVPageCopy,
    KVWriteSlot,
    LeakReport,
    MemorySnapshot,
    SequenceCacheView,
    SequenceExecutionView,
)
from ayaka.prefix.identity import (
    PrefixBlockIdentity,
    PrefixCacheContext,
    build_prefix_block_identities,
)
from ayaka.prefix.interface import CachedBlockInfo, PrefixLookupResult, PrefixMatch
from ayaka.prefix.store import RadixPagePrefixCache
from ayaka.utils.validation import require_int


class RuntimeMemoryManager:
    """Owns sequence KV blocks from reservation through safe reclamation.

    The manager is deliberately unaware of request priority, queue ordering,
    token budgets, and attention-kernel selection. A future scheduler sees only
    opaque handles, capacity snapshots, and structured reservation failures.

    Lifecycle protocol:
    ``begin_transaction`` -> ``try_reserve`` (tentative, reusable across the
    plan) -> ``prepare_step`` (freezes a lease) -> ``mark_step_in_flight`` ->
    ``complete_step`` / ``abort_prepared_step`` / ``fail_in_flight_step``.
    """

    def __init__(
        self,
        *,
        total_pages: int,
        page_size: int,
        max_sequences: int,
        max_sequence_tokens: int | None = None,
        storage: object | None = None,
        tiering: TieringConfig | None = None,
        host_storage: HostKVStorage | None = None,
        transfer_engine: TransferEngine | None = None,
    ) -> None:
        if total_pages < 2:
            raise ValueError("at least two pages are required, including padding")
        if max_sequences <= 0:
            raise ValueError("max_sequences must be positive")
        if max_sequence_tokens is not None and max_sequence_tokens <= 0:
            raise ValueError("max_sequence_tokens must be positive when provided")

        self.page_size = page_size
        self.max_sequence_tokens = max_sequence_tokens
        self.allocator = PageAllocator(total_pages=total_pages, page_size=page_size)
        self.sequences = SequenceArena(max_sequences)
        # One permanent padding page backs graph-padding writes; it is never
        # returned to the free pool
        self.padding_page = self.allocator.reserve_permanent_page()
        self.prefix_cache = RadixPagePrefixCache(
            allocator=self.allocator,
            page_size=page_size,
        )
        self.storage = storage
        self._validate_storage(storage, total_pages=total_pages, page_size=page_size)
        self.host_storage = host_storage

        # Tiering is strictly opt-in: with no config the manager keeps the
        # GPU-only code path, allocations included.
        self._tier = self._build_tier(
            tiering,
            host_storage=host_storage,
            transfer_engine=transfer_engine,
        )

        self._transactions: dict[int, TransactionRecord] = {}
        self._reservations: dict[int, ReservationRecord] = {}
        self._leases: dict[int, LeaseRecord] = {}
        self._prefix_matches: dict[int, tuple[PrefixMatchHandle, PrefixMatch]] = {}
        # Index counters are monotonic; generation stays 1 because handles are
        # never recycled while live (the dict entries are the source of truth).
        self._next_transaction_index = 0
        self._next_reservation_index = 0
        self._next_lease_index = 0
        self._next_prefix_match_index = 0
        self._kv_prefix_evictions_total = 0
        self._pressure_reclaim_attempts_total = 0
        self._pressure_reclaim_progress_total = 0
        self._pressure_prefix_eviction_attempts_total = 0
        self._pressure_prefix_eviction_progress_total = 0
        self._pressure_preemptions_total = 0
        self._lock = RLock()

    def _build_tier(
        self,
        config: TieringConfig | None,
        *,
        host_storage: HostKVStorage | None,
        transfer_engine: TransferEngine | None,
    ) -> TierManager | None:
        """Construct the optional host tier, or return None when it is off.

        A caller may inject an engine (tests and diagnostics do) or let the
        manager build one over a host mirror of the device storage. Enabling
        tiering without either is a configuration error rather than a silent
        no-op, because the caller clearly asked for host capacity.
        """

        if config is None:
            return None
        engine = transfer_engine
        if engine is None:
            mirror = host_storage
            if mirror is None:
                if self.storage is None:
                    raise StorageUnavailableError(
                        "tiering requires a KV storage or an explicit transfer engine"
                    )
                mirror = HostKVStorage(
                    self.storage,
                    capacity_pages=config.host_capacity_pages,
                )
                self.host_storage = mirror
            engine = build_transfer_engine(mirror)
        return TierManager(allocator=self.allocator, engine=engine, config=config)

    @property
    def tiering_enabled(self) -> bool:
        """Whether the opt-in host KV tier is active on this manager."""
        return self._tier is not None

    @property
    def tier_snapshot(self) -> HostTierSnapshot | None:
        """Placement and transfer accounting, or None when tiering is off."""
        with self._lock:
            return None if self._tier is None else self._tier.snapshot()

    def poll_transfers(self, *, drain: bool = False) -> int:
        """Publish every tier transfer that finished; returns how many landed.

        Called automatically from :meth:`advance_epoch` and
        :meth:`reclaim_deferred`, so the scheduler drives tier progress through
        the contract it already uses without ever naming a tier. ``drain``
        blocks until the copy stream is idle and is meant for tests and
        shutdown, not the hot path.
        """

        with self._lock:
            if self._tier is None:
                return 0
            landed = 0
            # Each pass publishes one more level of the promotion chain, so a
            # multi-block promotion converges in as many passes as it has
            # blocks; a pass that publishes nothing ends the loop.
            while True:
                evicted, promoted = self._tier.poll(
                    cached=self.prefix_cache.cached_block_index(),
                    drain=drain,
                )
                drain = False
                if not evicted and not promoted:
                    break
                for block in evicted:
                    if block.prefix_handle is not None:
                        # The block lives on the host now: drop the GPU cache
                        # reference exactly as an ordinary eviction would.
                        released = self.prefix_cache.release(
                            block.prefix_handle,
                            safe_epoch=block.safe_epoch,
                        )
                        self._kv_prefix_evictions_total += released
                    landed += 1
                published = sum(1 for block in promoted if self._publish_promotion_locked(block))
                landed += published
                if not evicted and not published:
                    break
            self._tier.reconcile(self.prefix_cache.cached_block_index())
            return landed

    def _publish_promotion_locked(self, block: PromotedBlock) -> bool:
        """Insert one promoted page back into the prefix cache.

        The promoted page is committed out of RESERVED into a request-owned
        LIVE page, handed to the cache, and then dropped back to cache-only
        ownership. Ancestors are borrowed with temporary request refs because
        ``insert`` revalidates the whole chain and requires one request-owned
        page per complete block.

        The cache, not the tier, decides which physical page survives: an
        insert reuses an existing canonical block if the same identity
        reappeared while the copy ran. Publication therefore reads the page
        back out of the cache, and a block that failed to land there aborts to
        the host tier instead of claiming a device placement it does not have.
        """

        if self._tier is None:  # pragma: no cover - guarded by callers
            return False
        borrowed: list[KVPageHandle] = []
        try:
            self.allocator.set_valid_tokens(block.page, self.page_size)
            self.allocator.commit_reserved([block.page])
            borrowed.append(block.page)
            for page in block.ancestor_pages:
                self.allocator.acquire_request_ref(page)
                borrowed.append(page)
            self.prefix_cache.insert(
                block.token_ids,
                (*block.ancestor_pages, block.page),
                context=block.context,
            )
        except Exception:
            self._release_tentative_prefix_pages(tuple(borrowed))
            self._tier.abort_promotion(block.identity)
            return False

        cached_page = next(
            (
                info.page
                for info in self.prefix_cache.cached_block_index()
                if info.identity == block.identity
            ),
            None,
        )
        self._release_tentative_prefix_pages(tuple(borrowed))
        if cached_page is None:
            self._tier.abort_promotion(block.identity)
            return False
        self._tier.publish_promotion(block.identity, device_page=cached_page)
        return True

    @property
    def current_epoch(self) -> int:
        """Epoch of the newest completed GPU step."""
        return self.allocator.current_epoch

    def prefix_cache_reuse_available(self) -> bool:
        """Return whether A6 prefix lookups are implemented by this manager."""

        return True

    @property
    def pressure_metrics(self) -> MemoryPressureMetrics:
        """Return exact cumulative counters owned by runtime memory."""

        with self._lock:
            allocator = self.allocator.snapshot()
            return MemoryPressureMetrics(
                kv_reclaims_total=allocator.reclaimed_pages_total,
                kv_prefix_evictions_total=self._kv_prefix_evictions_total,
                pressure_reclaim_attempts_total=self._pressure_reclaim_attempts_total,
                pressure_reclaim_progress_total=self._pressure_reclaim_progress_total,
                pressure_prefix_eviction_attempts_total=(
                    self._pressure_prefix_eviction_attempts_total
                ),
                pressure_prefix_eviction_progress_total=(
                    self._pressure_prefix_eviction_progress_total
                ),
                pressure_preemptions_total=self._pressure_preemptions_total,
            )

    @property
    def padding_physical_page(self) -> int:
        """Physical page index of the permanent padding page."""
        return self.allocator.physical_id(self.padding_page).value

    @property
    def padding_slot(self) -> int:
        """Flat slot address of the padding page's first position."""
        return self.padding_physical_page * self.page_size

    def create_sequence(self, request_id: str) -> SequenceHandle:
        """Allocate a fresh generation-safe sequence identity.

        Returns:
            The new sequence handle.

        Raises:
            SequenceCapacityError: if the arena is full.
        """
        with self._lock:
            return self.sequences.allocate(request_id)

    def get_sequence(self, sequence: SequenceHandle) -> SequenceMemorySnapshot:
        """Return an immutable snapshot of a sequence's KV state."""
        return self.sequences.get(sequence)

    def pin_prefix(self, sequence: SequenceHandle, num_tokens: int) -> tuple[PageTableEntry, ...]:
        """Pin a completed prefix, including partial tails, without copying bytes.

        The caller must unpin every returned entry exactly once. Pins freeze
        source bytes: subsequent appends reserve COW destinations. A truncated
        final entry deliberately exposes fewer positions than the physical page.
        Busy producers cannot publish even when some layers have finished.
        """
        require_int(num_tokens, "num_tokens", minimum=1)
        with self._lock:
            state = self.get_sequence(sequence)
            if (
                state.busy
                or state.release_requested
                or state.blocked_until_epoch > self.current_epoch
            ):
                raise SequenceBusyError("prefix producer has not retired successfully")
            if num_tokens > state.committed_tokens:
                raise ValueError("prefix exceeds completed KV")
            entries = tuple(
                PageTableEntry(entry.page, min(self.page_size, num_tokens - i * self.page_size))
                for i, entry in enumerate(state.page_table[: ceil(num_tokens / self.page_size)])
            )
            acquired = []
            try:
                for entry in entries:
                    self.allocator.pin_page(entry.page)
                    acquired.append(entry)
            except BaseException:
                self.unpin_prefix(tuple(acquired))
                raise
            return entries

    def unpin_prefix(self, entries: tuple[PageTableEntry, ...]) -> None:
        """Release the publisher's pins; request/lease owners remain independent."""
        with self._lock:
            for entry in entries:
                self.allocator.unpin_page(entry.page, safe_epoch=self.current_epoch)
            self.allocator.advance_epoch(self.current_epoch)

    def attach_pinned_prefix(
        self, sequence: SequenceHandle, entries: tuple[PageTableEntry, ...]
    ) -> int:
        """Revalidate generations and acquire all refs before publishing a resume.

        This operation accepts only pinned immutable resident pages from this
        manager. It does not turn a raw token match into valid state. The caller
        must validate model/context/token identity using its prefix registry.
        """
        with self._lock, self.sequences.mutate(sequence) as state:
            if state.committed_tokens or state.page_table.entries:
                raise InvalidStateTransitionError("resume requires an empty sequence")
            if (
                state.pending_transaction_index is not None
                or state.active_lease_index is not None
                or state.release_requested
                or state.blocked_until_epoch > self.current_epoch
            ):
                raise SequenceBusyError("resume requires an idle sequence")
            tokens = sum(e.valid_tokens for e in entries)
            SequencePageTable(list(entries)).validate(
                committed_tokens=tokens, page_size=self.page_size
            )
            if self.max_sequence_tokens is not None and tokens > self.max_sequence_tokens:
                raise ValueError("resume exceeds sequence limit")
            for entry in entries:
                meta = self.allocator.get_meta(entry.page)
                if not meta.is_pinned or meta.valid_tokens < entry.valid_tokens:
                    raise InvalidStateTransitionError("resume source is not immutable and valid")
            acquired = []
            try:
                for entry in entries:
                    self.allocator.acquire_request_ref(entry.page)
                    acquired.append(entry.page)
            except BaseException:
                for page in reversed(acquired):
                    self.allocator.release_request_ref(page, safe_epoch=self.current_epoch)
                raise
            state.page_table.entries = list(entries)
            state.committed_tokens = tokens
            state.version += 1
            return tokens

    def fork_sequence(
        self, source: SequenceHandle, destination: SequenceHandle, *, num_tokens: int | None = None
    ) -> int:
        """Share completed pages, including a read-only tail; append uses COW."""
        with self._lock:
            tokens = (
                self.get_sequence(source).committed_tokens if num_tokens is None else num_tokens
            )
            entries = self.pin_prefix(source, tokens)
            try:
                return self.attach_pinned_prefix(destination, entries)
            finally:
                self.unpin_prefix(entries)

    def cache_prefix(
        self,
        sequence: SequenceHandle,
        token_ids: Sequence[int],
        *,
        context: PrefixCacheContext,
    ) -> PrefixHandle | None:
        """Retain every complete committed page as a compatible prefix chain.

        Only full committed pages (``committed_tokens // page_size * page_size``
        leading tokens) can be shared; a partial tail page stays request-private
        (full-page sharing rule).

        Args:
            sequence: The sequence whose pages enter the cache.
            token_ids: Token IDs covering at least the shareable prefix.
            context: Compatibility context for the cached chain.

        Returns:
            The deepest cached block handle, or None when no full page exists.

        Raises:
            SequenceBusyError: if the sequence is busy or blocked.
            InvariantViolationError: if the shareable pages are not complete
                committed pages.
        """

        with self._lock, self.sequences.mutate(sequence) as state:
            if (
                state.pending_transaction_index is not None
                or state.active_lease_index is not None
                or state.release_requested
                or state.blocked_until_epoch > self.current_epoch
            ):
                raise SequenceBusyError("cannot cache a busy or blocked sequence")

            # floor(committed_tokens / page_size) * page_size: only complete
            # pages are cacheable.
            shareable_tokens = state.committed_tokens // self.page_size * self.page_size
            if shareable_tokens == 0:
                return None
            if len(token_ids) < shareable_tokens:
                raise ValueError("token_ids must cover every complete committed prefix page")

            block_count = shareable_tokens // self.page_size
            entries = state.page_table.entries[:block_count]
            if len(entries) != block_count or any(
                entry.valid_tokens != self.page_size for entry in entries
            ):
                raise InvariantViolationError(
                    "shareable prefix pages are not complete committed pages"
                )
            handle = self.prefix_cache.insert(
                tuple(token_ids[:shareable_tokens]),
                tuple(entry.page for entry in entries),
                context=context,
            )
            self._reconcile_tier_locked()
            return handle

    def match_prefix(
        self,
        token_ids: Sequence[int],
        *,
        context: PrefixCacheContext,
    ) -> PrefixMatch:
        """Return the longest compatible complete-page prefix.

        Internal (non-scheduler) matching API that exposes the raw page chain
        for manager-internal use; the scheduler-facing path is
        :meth:`lookup_prefix`.
        """
        with self._lock:
            return self.prefix_cache.match(token_ids, context=context)

    def lookup_prefix(
        self,
        token_ids: Sequence[int],
        *,
        context: PrefixCacheContext,
        max_matched_tokens: int | None = None,
    ) -> PrefixLookupResult:
        """Return logical match information plus a one-shot opaque handle.

        The scheduler never receives the physical page chain. The handle owns
        no page references: :meth:`try_reserve` must still revalidate the cache
        entry and acquire tentative request ownership transactionally.

        Args:
            token_ids: Candidate prompt tokens.
            context: Compatibility context for the lookup.
            max_matched_tokens: Optional cap on the searched prefix length.

        Returns:
            The matched token count plus an opaque, one-shot handle (empty when
            nothing matched).
        """

        if max_matched_tokens is not None:
            if not isinstance(max_matched_tokens, int) or isinstance(
                max_matched_tokens,
                bool,
            ):
                raise TypeError("max_matched_tokens must be an integer or None")
            if max_matched_tokens < 0:
                raise ValueError("max_matched_tokens must be non-negative")
        normalized = tuple(token_ids)
        # Cap the search window before hashing to bound work.
        searchable = normalized if max_matched_tokens is None else normalized[:max_matched_tokens]
        with self._lock:
            match = self.prefix_cache.match(searchable, context=context)
            self._request_promotions_locked(searchable, match, context=context)
            if match.matched_tokens == 0:
                return PrefixLookupResult.empty()
            handle = PrefixMatchHandle(
                index=self._next_prefix_match_index,
                generation=1,
            )
            self._next_prefix_match_index += 1
            self._prefix_matches[handle.index] = (handle, match)
            return PrefixLookupResult(
                matched_tokens=match.matched_tokens,
                handle=handle,
            )

    def discard_prefix_match(self, handle: PrefixMatchHandle) -> None:
        """Discard an unused lookup capability without changing page refs.

        The scheduler calls this when a prefix match is not used by any
        reservation, keeping the pending-match registry tidy.

        Raises:
            InvalidHandleError: if the handle is stale.
        """

        if not isinstance(handle, PrefixMatchHandle):
            raise TypeError("handle must be a PrefixMatchHandle")
        with self._lock:
            stored = self._prefix_matches.get(handle.index)
            if stored is None or stored[0] != handle:
                raise InvalidHandleError(f"stale prefix-match handle: {handle}")
            self._prefix_matches.pop(handle.index)

    def attach_prefix(
        self,
        sequence: SequenceHandle,
        match: PrefixMatch,
    ) -> int:
        """Attach a revalidated cache match to an empty sequence.

        Converts cache-page request refs into committed page-table entries so
        an already-cached prompt does not need recomputation.

        Args:
            sequence: A live, empty, idle sequence.
            match: A revalidated cache match.

        Returns:
            The number of tokens attached.

        Raises:
            InvalidStateTransitionError: if the sequence is not empty.
            SequenceBusyError: if the sequence is busy or blocked.
        """

        with self._lock, self.sequences.mutate(sequence) as state:
            if state.committed_tokens or state.page_table.entries:
                raise InvalidStateTransitionError(
                    "a cached prefix can only attach to an empty sequence"
                )
            if (
                state.pending_transaction_index is not None
                or state.active_lease_index is not None
                or state.release_requested
                or state.blocked_until_epoch > self.current_epoch
            ):
                raise SequenceBusyError("cannot attach a prefix to a busy sequence")
            if (
                self.max_sequence_tokens is not None
                and match.matched_tokens > self.max_sequence_tokens
            ):
                raise InvalidStateTransitionError(
                    "cached prefix exceeds the configured sequence limit"
                )

            pages = self.prefix_cache.acquire_match(match)
            if not pages:
                return 0
            state.page_table.entries = [
                PageTableEntry(page=page, valid_tokens=self.page_size) for page in pages
            ]
            state.committed_tokens = match.matched_tokens
            state.version += 1
            return match.matched_tokens

    def release_prefix(
        self,
        prefix: PrefixHandle,
        *,
        safe_epoch: int | None = None,
    ) -> int:
        """Drop a retained prefix entry and any now-unshared cache ancestors.

        Args:
            prefix: A retained terminal prefix handle.
            safe_epoch: Epoch whose completion makes released pages reusable;
                defaults to the current epoch.

        Returns:
            Number of cache references released.

        Raises:
            ValueError: if ``safe_epoch`` precedes the completed epoch.
        """

        epoch = self.current_epoch if safe_epoch is None else safe_epoch
        if epoch < self.current_epoch:
            raise ValueError("safe_epoch cannot precede the completed epoch")
        with self._lock:
            released = self.prefix_cache.release(prefix, safe_epoch=epoch)
            self._reconcile_tier_locked()
            return released

    def evict_prefixes(
        self,
        target_pages: int,
        *,
        safe_epoch: int | None = None,
    ) -> int:
        """Evict LRU prefix leaves and return the number of cache refs removed.

        Args:
            target_pages: Number of cache references to release.
            safe_epoch: Epoch whose completion makes released pages reusable;
                defaults to the current epoch.
        """

        epoch = self.current_epoch if safe_epoch is None else safe_epoch
        if epoch < self.current_epoch:
            raise ValueError("safe_epoch cannot precede the completed epoch")
        with self._lock:
            evicted = self.prefix_cache.evict(target_pages, safe_epoch=epoch)
            self._kv_prefix_evictions_total += evicted
            self._reconcile_tier_locked()
            return evicted

    def clear_prefix_cache(self, *, safe_epoch: int | None = None) -> int:
        """Evict every cached block; returns the number of refs released."""
        epoch = self.current_epoch if safe_epoch is None else safe_epoch
        if epoch < self.current_epoch:
            raise ValueError("safe_epoch cannot precede the completed epoch")
        with self._lock:
            evicted = self.prefix_cache.clear(safe_epoch=epoch)
            self._kv_prefix_evictions_total += evicted
            self._reconcile_tier_locked()
            return evicted

    def reclaim_deferred(self) -> MemoryPressureResult:
        """Reclaim only pages whose GPU-safe epoch has already completed.

        Repeated calls without an epoch advance are explicitly idempotent and
        return ``NO_PROGRESS``. The method never advances completion itself.

        Returns:
            A capacity-only outcome report.
        """

        with self._lock:
            # Land finished transfers first so the reported deltas describe
            # only the reclaim action and not tier bookkeeping.
            self.poll_transfers()
            before_allocator = self.allocator.snapshot()
            before_cache = self.prefix_cache.snapshot()
            self._pressure_reclaim_attempts_total += 1
            reclaimed = self.allocator.reclaim_completed()
            after_allocator = self.allocator.snapshot()
            after_cache = self.prefix_cache.snapshot()
            if reclaimed:
                self._pressure_reclaim_progress_total += 1
            return MemoryPressureResult(
                action=PressureAction.RECLAIM_DEFERRED,
                status=(PressureStatus.PROGRESSED if reclaimed else PressureStatus.NO_PROGRESS),
                requested_pages=0,
                cache_pages_evicted=0,
                pages_reclaimed=reclaimed,
                free_pages_before=before_allocator.free_pages,
                free_pages_after=after_allocator.free_pages,
                deferred_pages_before=before_allocator.reclaim_pending_pages,
                deferred_pages_after=after_allocator.reclaim_pending_pages,
                cached_pages_before=before_cache.cached_blocks,
                cached_pages_after=after_cache.cached_blocks,
            )

    def evict_prefixes_for_pressure(self, required_pages: int) -> MemoryPressureResult:
        """Evict deterministic LRU cache ownership until capacity is gained.

        ``required_pages`` is additional free capacity desired by the caller,
        not a physical-page identity. Active request refs remain untouched;
        cache-only pages are reclaimed immediately only when the current safe
        epoch permits it.

        Args:
            required_pages: Additional free pages the caller wants.

        Returns:
            A capacity-only outcome report.
        """

        if not isinstance(required_pages, int) or isinstance(required_pages, bool):
            raise TypeError("required_pages must be an integer")
        if required_pages < 0:
            raise ValueError("required_pages must be non-negative")
        with self._lock:
            # Landing pending transfers first turns already-spilled blocks
            # into free capacity before anything is dropped outright.
            self.poll_transfers()
            before_allocator = self.allocator.snapshot()
            before_cache = self.prefix_cache.snapshot()
            # Cap the target so a saturated pool never loops forever.
            desired_free_pages = min(
                before_allocator.usable_pages,
                before_allocator.free_pages + required_pages,
            )
            self._pressure_prefix_eviction_attempts_total += 1
            evicted = 0
            reclaimed = 0
            if self._tier is not None:
                # Preserve what we can on the host, then fall through to the
                # ordinary drop loop for whatever capacity is still missing.
                # Spilling is best effort by design: a slow or full host tier
                # never makes pressure handling weaker than GPU-only mode.
                missing = desired_free_pages - self.allocator.available_pages()
                if missing > 0:
                    self._spill_prefixes_locked(missing)
                    evicted += self._land_spills_locked()
                    reclaimed += self.allocator.reclaim_completed()
            # Evict one cache page at a time and reclaim immediately. Stops
            # as soon as the target is met, the cache is empty, or -- the case
            # that matters -- nothing the cache still owns could become free.
            # Without that last guard a pool whose cached pages are all pinned
            # by live requests loses its entire prefix cache and gains not one
            # page, which is strictly worse than doing nothing.
            while self.allocator.available_pages() < desired_free_pages:
                if self.allocator.snapshot().evictable_pages == 0:
                    break
                if self.prefix_cache.snapshot().cached_blocks == 0:
                    break
                evicted += self.prefix_cache.evict(
                    1,
                    safe_epoch=self.current_epoch,
                )
                reclaimed += self.allocator.reclaim_completed()

            after_allocator = self.allocator.snapshot()
            after_cache = self.prefix_cache.snapshot()
            progressed = evicted > 0 or reclaimed > 0
            self._kv_prefix_evictions_total += evicted
            if progressed:
                self._pressure_prefix_eviction_progress_total += 1
            return MemoryPressureResult(
                action=PressureAction.EVICT_PREFIX,
                status=(PressureStatus.PROGRESSED if progressed else PressureStatus.NO_PROGRESS),
                requested_pages=required_pages,
                cache_pages_evicted=evicted,
                pages_reclaimed=reclaimed,
                free_pages_before=before_allocator.free_pages,
                free_pages_after=after_allocator.free_pages,
                deferred_pages_before=before_allocator.reclaim_pending_pages,
                deferred_pages_after=after_allocator.reclaim_pending_pages,
                cached_pages_before=before_cache.cached_blocks,
                cached_pages_after=after_cache.cached_blocks,
            )

    def spill_prefixes_to_host(self, target_pages: int) -> int:
        """Start moving up to ``target_pages`` LRU cache blocks to the host.

        Returns the number of transfers submitted, not the number of pages
        freed: a spill only frees its GPU page once the copy has landed, which
        :meth:`poll_transfers` reports. Returns zero when tiering is off, so
        callers need no capability check.
        """

        if not isinstance(target_pages, int) or isinstance(target_pages, bool):
            raise TypeError("target_pages must be an integer")
        if target_pages < 0:
            raise ValueError("target_pages must be non-negative")
        with self._lock:
            return self._spill_prefixes_locked(target_pages)

    def _spill_prefixes_locked(self, target_pages: int) -> int:
        """Submit device-to-host copies for the coldest evictable leaves.

        Candidates come from the prefix cache in exactly the order
        :meth:`evict_prefixes` would drop them, so tiering changes where a
        cold block goes, never which block is chosen.

        Each candidate is spilled together with the ancestors its release will
        prune. The cache drops an unshared chain bottom-up, so spilling only
        the leaf would leave the host tier holding a block whose parents no
        longer exist -- unpromotable, and pinning a mirror slot for nothing.
        """

        if self._tier is None or target_pages <= 0:
            return 0
        self._reconcile_tier_locked()
        cached = self.prefix_cache.cached_block_index()
        by_identity = {info.identity: info for info in cached}
        submitted = 0
        for leaf in self.prefix_cache.evictable_leaves():
            if submitted >= target_pages:
                break
            chain = self._pruning_chain(leaf, by_identity)
            for position, info in enumerate(chain):
                ticket = self._tier.begin_eviction(info, release_anchor=position == 0)
                if ticket is None:
                    # A partial cascade is safe: whatever the release prunes
                    # without a host copy is simply lost, exactly as it would
                    # be in GPU-only mode.
                    break
                submitted += 1
        return submitted

    @staticmethod
    def _pruning_chain(
        leaf: CachedBlockInfo,
        by_identity: dict[PrefixBlockIdentity, CachedBlockInfo],
    ) -> tuple[CachedBlockInfo, ...]:
        """Return the blocks a leaf release drops, leaf first.

        Mirrors the cache's bottom-up prune rule: an ancestor goes with the
        leaf exactly when it is unretained and has no other child.
        """

        chain = [leaf]
        parent_identity = leaf.parent_identity
        while parent_identity is not None:
            parent = by_identity.get(parent_identity)
            if parent is None or parent.terminal or parent.children != 1:
                break
            chain.append(parent)
            parent_identity = parent.parent_identity
        return tuple(chain)

    def _land_spills_locked(self) -> int:
        """Wait for submitted spills to land; returns cache refs released.

        Pressure handling drains rather than polls on purpose. A spill holds a
        request reference on its source page for the whole copy, so an
        unfinished spill pins exactly the page the caller is trying to free --
        polling would report no progress and send the drop loop through the
        rest of the cache for capacity that is already on its way.
        """

        if self._tier is None:
            return 0
        before = self._kv_prefix_evictions_total
        self.poll_transfers(drain=True)
        return self._kv_prefix_evictions_total - before

    def _reconcile_tier_locked(self) -> None:
        """Refresh tier placement records from the cache's current contents."""
        if self._tier is not None:
            self._tier.reconcile(self.prefix_cache.cached_block_index())

    def _request_promotions_locked(
        self,
        token_ids: tuple[int, ...],
        match: PrefixMatch,
        *,
        context: PrefixCacheContext,
    ) -> int:
        """Start pulling back the host blocks that would extend a match.

        This is the whole of A11-07 on the read path: a lookup reports only
        what is device resident right now, and promotion happens in the
        background. The scheduler sees a prefix that grows over subsequent
        lookups, expressed purely through ``matched_tokens`` -- no tier state,
        no placement, and no new failure mode.
        """

        if self._tier is None or not self._tier.has_host_blocks:
            # Rebuilding the identity chain re-hashes the whole prefix, which
            # the match already did; skip it entirely while the host tier holds
            # nothing to promote.
            return 0
        identities = build_prefix_block_identities(
            token_ids,
            page_size=self.page_size,
            context=context,
        )
        matched_blocks = match.matched_tokens // self.page_size
        if matched_blocks >= len(identities):
            return 0
        started = 0
        for identity in self._tier.host_chain_after(identities, start=matched_blocks):
            if self._tier.begin_promotion(identity) is None:
                break
            started += 1
        return started

    def preempt_sequence(
        self,
        sequence: SequenceHandle,
        *,
        safe_epoch: int | None = None,
    ) -> SequencePreemptionResult:
        """Release request-owned KV while preserving sequence identity.

        Busy or writable sequences return a structured ``BUSY`` outcome. A
        second call after resident KV has already been released is a harmless
        ``NO_RESIDENT_STATE`` result.

        Args:
            sequence: The sequence to preempt.
            safe_epoch: Epoch whose completion makes released pages reusable.

        Returns:
            A logical preemption outcome.
        """

        epoch = self.current_epoch if safe_epoch is None else safe_epoch
        if epoch < self.current_epoch:
            raise ValueError("safe_epoch cannot precede the completed epoch")
        with self._lock, self.sequences.mutate(sequence) as state:
            before = self.allocator.snapshot()
            if (
                state.pending_transaction_index is not None
                or state.active_lease_index is not None
                or state.release_requested
                or state.blocked_until_epoch > self.current_epoch
                or any(
                    self.allocator.get_meta(entry.page).inflight_refs > 0
                    for entry in state.page_table.entries
                )
            ):
                # A step may still write these pages; releasing now would let
                # them be recycled underneath the in-flight kernel.
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
            if not state.page_table.entries:
                if state.committed_tokens:
                    raise InvariantViolationError(
                        "empty sequence page table retained committed tokens"
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
            released_pages = len(state.page_table.entries)
            for entry in state.page_table.entries:
                self.allocator.release_request_ref(entry.page, safe_epoch=epoch)
            state.page_table.entries.clear()
            state.committed_tokens = 0
            state.version += 1

            pending = self.allocator.snapshot()
            reclaimed = self.allocator.reclaim_completed()
            after = self.allocator.snapshot()
            self._pressure_preemptions_total += 1
            return SequencePreemptionResult(
                sequence=sequence,
                status=PreemptionStatus.RELEASED,
                released_tokens=released_tokens,
                released_request_pages=released_pages,
                pages_reclaimed=reclaimed,
                pages_deferred=max(
                    pending.reclaim_pending_pages - before.reclaim_pending_pages,
                    0,
                ),
                free_pages_before=before.free_pages,
                free_pages_after=after.free_pages,
            )

    def begin_transaction(self, step_id: int) -> MemoryTransactionHandle:
        """Open a tentative-planning transaction for one scheduler step.

        Args:
            step_id: Scheduler step identity; must be non-negative.

        Returns:
            An opaque transaction handle for reservation calls.
        """
        if step_id < 0:
            raise ValueError("step_id must be non-negative")
        with self._lock:
            index = self._next_transaction_index
            self._next_transaction_index += 1
            handle = MemoryTransactionHandle(index=index, generation=1, step_id=step_id)
            self._transactions[index] = TransactionRecord(handle=handle)
            return handle

    def try_reserve(
        self,
        transaction: MemoryTransactionHandle,
        sequence: SequenceHandle,
        num_new_tokens: int,
        *,
        prefix_match: PrefixMatchHandle | None = None,
    ) -> ReservationResult:
        """Tentatively reserve append slots without changing committed KV length.

        On success the reservation record holds the planned page table, newly
        RESERVED pages, and any tentatively acquired prefix refs; nothing is
        committed until ``prepare_step``/``complete_step``. Failure paths
        release any tentatively acquired refs so the plan is fully reversible.

        Args:
            transaction: Open transaction handle.
            sequence: Sequence to extend.
            num_new_tokens: Positive token count to plan for.
            prefix_match: Optional one-shot handle from
                :meth:`lookup_prefix`; consumed on use.

        Returns:
            Success (with opaque reservation handle) or a structured failure.
        """

        with self._lock:
            tx = self._get_transaction(transaction)
            LifecycleTransitions.require_open(tx)
            # Cheap argument validation comes first: consuming the one-shot
            # prefix handle on a rejection that has nothing to do with the
            # cache would destroy a match the caller could still have used.
            if num_new_tokens <= 0:
                return ReservationResult.failure(ReservationFailure.INVALID_TOKEN_COUNT)
            match = self._consume_prefix_match_locked(prefix_match)
            if prefix_match is not None and match is None:
                return ReservationResult.failure(ReservationFailure.PREFIX_NOT_RESIDENT)

            # Hold the arena lock across the whole plan: resolving the handle
            # and reserving against it must observe one consistent sequence.
            # ExitStack keeps the resolution failure inside its own try, so a
            # stale page handle raised later in the plan is not misreported as
            # an invalid sequence.
            with ExitStack() as arena_lock:
                try:
                    state = arena_lock.enter_context(self.sequences.mutate(sequence))
                except InvalidHandleError:
                    return ReservationResult.failure(ReservationFailure.SEQUENCE_INVALID)

                if (
                    state.pending_transaction_index is not None
                    or state.active_lease_index is not None
                    or state.release_requested
                    or state.blocked_until_epoch > self.current_epoch
                ):
                    return ReservationResult.failure(ReservationFailure.SEQUENCE_BUSY)

                attached_prefix_tokens = 0 if match is None else match.matched_tokens
                # A cache prefix may only attach to an empty sequence: committed
                # pages and cache pages would otherwise share the same positions.
                if match is not None and (
                    state.committed_tokens
                    or state.page_table.entries
                    or match.page_size != self.page_size
                ):
                    return ReservationResult.failure(ReservationFailure.PREFIX_NOT_RESIDENT)

                execution_base_tokens = state.committed_tokens + attached_prefix_tokens
                final_tokens = execution_base_tokens + num_new_tokens
                usable_pages = self.allocator.snapshot().usable_pages
                # ceil(final / page_size) - existing blocks = pages to allocate.
                required_total_pages = ceil(final_tokens / self.page_size)
                existing_page_count = len(state.page_table.entries) + (
                    attached_prefix_tokens // self.page_size
                )
                required_new_pages = required_total_pages - existing_page_count
                old_entries = state.page_table.snapshot()
                cow_tail = bool(
                    old_entries
                    and old_entries[-1].valid_tokens < self.page_size
                    and (
                        not self.allocator.get_meta(old_entries[-1].page).can_mutate
                        or self.allocator.get_meta(old_entries[-1].page).valid_tokens
                        != old_entries[-1].valid_tokens
                    )
                )
                required_new_pages += int(cow_tail)

                if (
                    self.max_sequence_tokens is not None and final_tokens > self.max_sequence_tokens
                ) or required_total_pages > usable_pages:
                    return ReservationResult.failure(
                        ReservationFailure.REQUEST_TOO_LARGE,
                        required_pages=max(0, required_new_pages),
                    )

                acquired_prefix_pages: tuple[KVPageHandle, ...] = ()
                if match is not None:
                    try:
                        acquired_prefix_pages = self.prefix_cache.acquire_match(match)
                    except (
                        InvalidHandleError,
                        InvalidStateTransitionError,
                        InvariantViolationError,
                    ):
                        # The chain vanished or changed between lookup and use.
                        return ReservationResult.failure(
                            ReservationFailure.PREFIX_NOT_RESIDENT,
                        )
                    if len(acquired_prefix_pages) * self.page_size != attached_prefix_tokens:
                        self._release_tentative_prefix_pages(acquired_prefix_pages)
                        raise InvariantViolationError(
                            "acquired prefix pages do not match the logical prefix length"
                        )

                allocated = self.allocator.allocate(max(0, required_new_pages))
                if allocated is None:
                    self._release_tentative_prefix_pages(acquired_prefix_pages)
                    return ReservationResult.failure(
                        ReservationFailure.NO_CAPACITY,
                        required_pages=max(0, required_new_pages),
                    )

                try:
                    prefix_entries = tuple(
                        PageTableEntry(page=page, valid_tokens=self.page_size)
                        for page in acquired_prefix_pages
                    )
                    # Plan the full post-append table; the committed table is
                    # unchanged until execution succeeds.
                    copies = ()
                    append_pages = allocated
                    if cow_tail:
                        tail = old_entries[-1]
                        copies = (KVPageCopy(tail.page, allocated[0], tail.valid_tokens),)
                        old_entries = old_entries[:-1] + (
                            PageTableEntry(allocated[0], tail.valid_tokens),
                        )
                        append_pages = allocated[1:]
                    planned_table, write_slots = self._plan_append(
                        old_entries + prefix_entries,
                        append_pages,
                        base_committed_tokens=execution_base_tokens,
                        num_new_tokens=num_new_tokens,
                    )
                except Exception:
                    self.allocator.rollback_reserved(allocated)
                    self._release_tentative_prefix_pages(acquired_prefix_pages)
                    raise

                index = self._next_reservation_index
                self._next_reservation_index += 1
                reservation_handle = KVReservationHandle(
                    index=index,
                    generation=1,
                    step_id=transaction.step_id,
                )
                record = ReservationRecord(
                    handle=reservation_handle,
                    sequence=sequence,
                    base_sequence_version=state.version,
                    base_committed_tokens=state.committed_tokens,
                    attached_prefix_tokens=attached_prefix_tokens,
                    num_new_tokens=num_new_tokens,
                    planned_page_table=planned_table,
                    allocated_pages=allocated,
                    acquired_prefix_pages=acquired_prefix_pages,
                    write_slots=write_slots,
                    copies=copies,
                )
                self._reservations[index] = record
                tx.reservation_handles.append(reservation_handle)
                # The sequence is now owned by exactly one transaction (invariant 7).
                state.pending_transaction_index = transaction.index
                return ReservationResult.success(
                    reservation_handle,
                    required_pages=max(0, required_new_pages),
                    allocated_pages=len(allocated),
                    attached_prefix_tokens=attached_prefix_tokens,
                )

    def rollback_transaction(self, transaction: MemoryTransactionHandle) -> None:
        """Undo every reservation in the transaction (tentative-plan discard).

        Rolls back reserved pages, releases tentatively acquired prefix refs,
        clears sequence pending-transaction markers, and honors any deferred
        release request made while the transaction was open.
        """
        with self._lock:
            tx = self._get_transaction(transaction)
            self._rollback_transaction_locked(tx)

    def prepare_step(self, transaction: MemoryTransactionHandle) -> StepMemoryLeaseHandle:
        """Freeze ownership without publishing KV; invalid plans roll back."""
        with self._lock:
            tx = self._get_transaction(transaction)
            records = [self._get_reservation(h) for h in tx.reservation_handles]

            def validate() -> None:
                for record in records:
                    with self.sequences.mutate(record.sequence) as state:
                        if (
                            state.release_requested
                            or state.pending_transaction_index != transaction.index
                            or state.active_lease_index is not None
                            or state.version != record.base_sequence_version
                            or state.committed_tokens != record.base_committed_tokens
                        ):
                            raise InvalidStateTransitionError(
                                "sequence changed while the plan was tentative"
                            )

            def activate(lease: LeaseRecord) -> None:
                for record in records:
                    with self.sequences.mutate(record.sequence) as state:
                        state.pending_transaction_index = None
                        state.active_lease_index = lease.handle.index

            def deactivate(lease: LeaseRecord) -> None:
                for record in records:
                    with self.sequences.mutate(record.sequence) as state:
                        if state.active_lease_index == lease.handle.index:
                            state.active_lease_index = None
                            state.pending_transaction_index = transaction.index

            index = self._next_lease_index
            self._next_lease_index += 1
            return TransactionOrchestrator.prepare(
                tx,
                StepMemoryLeaseHandle(index=index, generation=1, step_id=transaction.step_id),
                validate=validate,
                activate=activate,
                deactivate=deactivate,
                rollback=lambda: self._rollback_transaction_locked(tx),
                transactions=self._transactions,
                leases=self._leases,
            )

    def build_execution_view(
        self,
        lease_handle: StepMemoryLeaseHandle,
    ) -> ExecutionMemoryView:
        """Materialize the frozen lease into kernel-facing execution metadata.

        Resolves every planned page to its physical ID and exposes the padding
        page/slot for graph-padding writes.

        Args:
            lease_handle: Prepared lease handle.

        Returns:
            The immutable execution view for batch building.

        Raises:
            InvalidHandleError: if the lease handle is stale.
        """
        with self._lock:
            lease = self._get_lease(lease_handle)
            sequence_views = []
            for reservation_handle in lease.reservation_handles:
                record = self._get_reservation(reservation_handle)
                block_table = tuple(
                    self.allocator.physical_id(entry.page).value
                    for entry in record.planned_page_table
                )
                sequence_views.append(
                    SequenceExecutionView(
                        sequence=record.sequence,
                        reservation=record.handle,
                        base_committed_tokens=record.execution_base_tokens,
                        num_reserved_tokens=record.num_new_tokens,
                        block_table=block_table,
                        write_slots=record.write_slots,
                    )
                )
            return ExecutionMemoryView(
                step_id=lease_handle.step_id,
                lease=lease_handle,
                sequences=tuple(sequence_views),
                page_size=self.page_size,
                padding_page=self.padding_physical_page,
                padding_slot=self.padding_slot,
                copies=tuple(
                    copy
                    for h in lease.reservation_handles
                    for copy in self._get_reservation(h).copies
                ),
            )

    def validate_execution_view(self, view: ExecutionMemoryView) -> None:
        """Reject stale lease, sequence versions, and changed physical metadata."""
        with self._lock:
            lease = self._get_lease(view.lease)
            LifecycleTransitions.require_lease(lease, LeaseState.PREPARED)
            records = [self._get_reservation(h) for h in lease.reservation_handles]
            self._validate_lease_sequences(lease, records)
            if self.build_execution_view(view.lease) != view:
                raise InvalidStateTransitionError("execution memory view changed")

    def mark_step_in_flight(self, lease_handle: StepMemoryLeaseHandle) -> None:
        """Acquire transient ownership over every page the step may touch.

        Must be called before the GPU step launches so released pages cannot
        be recycled while the kernel may still write them.

        Args:
            lease_handle: A PREPARED lease.

        Raises:
            InvalidStateTransitionError: if the lease is not PREPARED.
        """
        with self._lock:
            lease = self._get_lease(lease_handle)
            LifecycleTransitions.require_lease(lease, LeaseState.PREPARED)
            pages = self._lease_touched_pages(lease)
            self.allocator.mark_inflight(pages)
            LifecycleTransitions.transition_lease(
                lease,
                expected=LeaseState.PREPARED,
                target=LeaseState.IN_FLIGHT,
            )

    def commit_step(
        self,
        lease_handle: StepMemoryLeaseHandle,
        *,
        written_tokens: Mapping[KVReservationHandle, int] | None = None,
    ) -> None:
        """Publish completed KV while retaining all execution refs and busy markers.

        The caller must have proof that all writes succeeded. Retirement is
        separate; a committed lease continues to prevent page/sequence reuse.
        """

        with self._lock:
            lease = self._get_lease(lease_handle)
            LifecycleTransitions.require_lease(lease, LeaseState.IN_FLIGHT)
            records = [self._get_reservation(handle) for handle in lease.reservation_handles]
            self._validate_written_counts(records, written_tokens)
            self._validate_lease_sequences(lease, records)

            new_pages = tuple(page for record in records for page in record.allocated_pages)
            touched_pages = self._lease_touched_pages(lease)
            for page in touched_pages:
                meta = self.allocator.get_meta(page)
                if meta.inflight_refs <= 0:
                    raise InvariantViolationError("lease page has no in-flight ownership")

            # RESERVED -> LIVE with one request ref per new page.
            self.allocator.commit_reserved(new_pages)
            for record in records:
                for entry in record.planned_page_table:
                    self.allocator.set_valid_tokens(entry.page, entry.valid_tokens)
                with self.sequences.mutate(record.sequence) as state:
                    state.page_table.entries = list(record.planned_page_table)
                    state.committed_tokens = record.execution_base_tokens + record.num_new_tokens
                    state.version += 1

                for copy in record.copies:
                    self.allocator.release_request_ref(copy.source, safe_epoch=self.current_epoch)

            for record in records:
                record.committed = True

            LifecycleTransitions.transition_lease(
                lease,
                expected=LeaseState.IN_FLIGHT,
                target=LeaseState.COMMITTED,
            )

    def retire_step(self, lease_handle: StepMemoryLeaseHandle) -> None:
        """Retire a COMMITTED lease after every consumer's last use.

        This is a host ownership operation, not a device synchronization.
        The caller is responsible for completion proof.
        """
        with self._lock:
            lease = self._get_lease(lease_handle)
            LifecycleTransitions.require_lease(lease, LeaseState.COMMITTED)
            self.allocator.unmark_inflight(self._lease_touched_pages(lease))
            for handle in lease.reservation_handles:
                record = self._get_reservation(handle)
                with self.sequences.mutate(record.sequence) as state:
                    state.active_lease_index = None
            LifecycleTransitions.transition_lease(
                lease,
                expected=LeaseState.COMMITTED,
                target=LeaseState.COMPLETED,
            )
            self._finish_lease_locked(lease)

    def complete_step(
        self,
        lease_handle: StepMemoryLeaseHandle,
        *,
        written_tokens: Mapping[KVReservationHandle, int] | None = None,
    ) -> None:
        """Compatibility API: commit and retire after proven whole-step completion."""
        with self._lock:
            self.commit_step(lease_handle, written_tokens=written_tokens)
            self.retire_step(lease_handle)

    def abort_prepared_step(self, lease_handle: StepMemoryLeaseHandle) -> None:
        """Cancel a prepared step before any GPU work is launched.

        Rolls new reservations back immediately (safe: no kernel may have
        touched their bytes), releases tentatively acquired prefix refs, and
        honors any deferred release request.

        Args:
            lease_handle: A PREPARED lease.

        Raises:
            InvalidStateTransitionError: if the lease is not PREPARED.
        """

        with self._lock:
            lease = self._get_lease(lease_handle)
            LifecycleTransitions.require_lease(lease, LeaseState.PREPARED)
            records = [self._get_reservation(handle) for handle in lease.reservation_handles]
            for record in records:
                self.allocator.rollback_reserved(record.allocated_pages)
                self._release_tentative_prefix_pages(record.acquired_prefix_pages)
                with self.sequences.mutate(record.sequence) as state:
                    state.active_lease_index = None
            LifecycleTransitions.transition_lease(
                lease,
                expected=LeaseState.PREPARED,
                target=LeaseState.ABORTED,
            )
            self._finish_lease_locked(lease)

    def fail_in_flight_step(
        self,
        lease_handle: StepMemoryLeaseHandle,
        *,
        safe_epoch: int,
    ) -> None:
        """Fail closed after a launch that may have partially written KV bytes.

        New pages are abandoned into deferred reclaim (never rolled back into
        immediate reuse), committed metadata stays unchanged, and each sequence
        is blocked until the supplied safe epoch.

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
            LifecycleTransitions.require_lease(lease, LeaseState.IN_FLIGHT)
            records = [self._get_reservation(handle) for handle in lease.reservation_handles]
            touched_pages = self._lease_touched_pages(lease)
            new_pages = tuple(page for record in records for page in record.allocated_pages)

            # Partially written bytes cannot be trusted: defer, do not reuse.
            self.allocator.abandon_reserved(new_pages, safe_epoch=safe_epoch)
            for record in records:
                self._release_tentative_prefix_pages(
                    record.acquired_prefix_pages,
                    safe_epoch=safe_epoch,
                )
            self.allocator.unmark_inflight(touched_pages)
            for record in records:
                with self.sequences.mutate(record.sequence) as state:
                    state.active_lease_index = None
                    state.blocked_until_epoch = max(state.blocked_until_epoch, safe_epoch)
                    state.version += 1
            LifecycleTransitions.transition_lease(
                lease,
                expected=LeaseState.IN_FLIGHT,
                target=LeaseState.FAILED,
            )
            self._finish_lease_locked(lease, failure_safe_epoch=safe_epoch)

    def release_sequence(
        self,
        sequence: SequenceHandle,
        *,
        safe_epoch: int | None = None,
    ) -> ReleaseStatus:
        """Release request ownership now or immediately after its active step.

        A busy sequence (open transaction or active lease) records the release
        request and returns ``DEFERRED``; the release runs when the step
        finishes. Otherwise request refs are dropped immediately.

        Args:
            sequence: The sequence to release.
            safe_epoch: Epoch whose completion makes released pages reusable.

        Returns:
            ``RELEASED`` (dropped now) or ``DEFERRED`` (queued behind the step).
        """

        epoch = self.current_epoch if safe_epoch is None else safe_epoch
        if epoch < self.current_epoch:
            raise ValueError("safe_epoch cannot precede the completed epoch")
        with self._lock, self.sequences.mutate(sequence) as state:
            if state.pending_transaction_index is not None or state.active_lease_index is not None:
                state.release_requested = True
                state.release_safe_epoch = max(state.release_safe_epoch, epoch)
                return ReleaseStatus.DEFERRED
            self._release_sequence_locked(sequence, safe_epoch=epoch)
            return ReleaseStatus.RELEASED

    def build_cache_view(self, sequences: Sequence[SequenceHandle]) -> CacheView:
        """Build the read-only attention view for a set of committed sequences.

        Args:
            sequences: Live sequence handles to expose.

        Returns:
            The immutable cache view for attention backends.
        """
        with self._lock:
            views = []
            for sequence in sequences:
                with self.sequences.mutate(sequence) as state:
                    pages = tuple(entry.page for entry in state.page_table.entries)
                    block_table = tuple(self.allocator.physical_id(page).value for page in pages)
                    views.append(
                        SequenceCacheView(
                            sequence=sequence,
                            committed_tokens=state.committed_tokens,
                            pages=pages,
                            block_table=block_table,
                        )
                    )
            return CacheView(
                page_size=self.page_size,
                sequences=tuple(views),
                padding_page=self.padding_physical_page,
                padding_slot=self.padding_slot,
            )

    def advance_epoch(self, completed_epoch: int) -> int:
        """Advance the completion watermark and reclaim now-safe pages.

        Args:
            completed_epoch: Newest completed GPU step epoch; must not regress.

        Returns:
            Number of pages reclaimed.
        """
        with self._lock:
            reclaimed = self.allocator.advance_epoch(completed_epoch)
            # The scheduler already calls this every step, so it is the
            # natural place to let tier transfers land without giving the
            # scheduler any tier-shaped API of its own.
            self.poll_transfers()
            return reclaimed

    def snapshot(self) -> MemorySnapshot:
        """Return approximate scheduler-facing capacity feedback.

        Fragmentation is the tail-page waste sum ``(-committed) % page_size``
        per non-empty sequence: tokens lost at each sequence's page boundary.
        """
        with self._lock:
            allocator = self.allocator.snapshot()
            prefix_cache = self.prefix_cache.snapshot()
            sequence_snapshots = [
                self.sequences.get(handle) for handle in self.sequences.live_handles()
            ]
            committed_tokens = sum(item.committed_tokens for item in sequence_snapshots)
            fragmentation = sum(
                (-item.committed_tokens) % self.page_size
                for item in sequence_snapshots
                if item.committed_tokens
            )
            return MemorySnapshot(
                total_pages=allocator.total_pages,
                usable_pages=allocator.usable_pages,
                free_pages=allocator.free_pages,
                reserved_pages=allocator.reserved_pages,
                live_pages=allocator.live_pages,
                deferred_free_pages=allocator.reclaim_pending_pages,
                permanent_pages=allocator.permanent_pages,
                request_owned_pages=allocator.request_owned_pages,
                cache_owned_pages=allocator.cache_owned_pages,
                shared_pages=allocator.shared_pages,
                inflight_pages=allocator.inflight_pages,
                page_size=self.page_size,
                committed_tokens=committed_tokens,
                internal_fragmentation_tokens=fragmentation,
                live_sequences=len(sequence_snapshots),
                open_transactions=len(self._transactions),
                active_leases=len(self._leases),
                cached_prefix_blocks=prefix_cache.cached_blocks,
                cached_prefix_entries=prefix_cache.terminal_entries,
                cached_prefix_tokens=prefix_cache.cached_tokens,
                pending_prefix_matches=len(self._prefix_matches),
                cache_evictable_pages=allocator.evictable_pages,
            )

    def leak_report(self) -> LeakReport:
        """Return the full accounting report for deterministic leak detection."""
        with self._lock:
            return LeakReport(
                snapshot=self.snapshot(),
                live_sequence_handles=self.sequences.live_handles(),
                open_transaction_ids=tuple(sorted(self._transactions)),
                active_lease_ids=tuple(sorted(self._leases)),
                cached_prefix_handles=self.prefix_cache.snapshot().handles,
            )

    def assert_invariants(self) -> None:
        """Debug-only exhaustive cross-component invariant check.

        Verifies allocator/prefix/sequence invariants plus the cross-cutting
        ownership equality: every page referenced by a committed table or a
        tentative prefix acquisition must carry exactly that many request refs.
        """
        with self._lock:
            self.allocator.assert_invariants()
            self.prefix_cache.assert_invariants()
            self.sequences.assert_invariants(page_size=self.page_size)

            sequence_page_refs: dict[KVPageHandle, int] = {}
            for handle in self.sequences.live_handles():
                with self.sequences.mutate(handle) as state:
                    for entry in state.page_table.entries:
                        meta = self.allocator.get_meta(entry.page)
                        if meta.allocation_state is not PageAllocationState.LIVE:
                            raise InvariantViolationError(
                                "committed page table refers to a non-live page"
                            )
                        if meta.valid_tokens < entry.valid_tokens:
                            raise InvariantViolationError(
                                "page-table and allocator token counts disagree"
                            )
                        sequence_page_refs[entry.page] = sequence_page_refs.get(entry.page, 0) + 1

                    if state.pending_transaction_index is not None:
                        tx = self._transactions.get(state.pending_transaction_index)
                        if tx is None or not any(
                            self._get_reservation(res).sequence == handle
                            for res in tx.reservation_handles
                        ):
                            raise InvariantViolationError(
                                "sequence points to a missing pending transaction"
                            )
                    if state.active_lease_index is not None:
                        lease = self._leases.get(state.active_lease_index)
                        if lease is None or not any(
                            self._get_reservation(res).sequence == handle
                            for res in lease.reservation_handles
                        ):
                            raise InvariantViolationError(
                                "sequence points to a missing active lease"
                            )

            for record in self._reservations.values():
                if record.attached_prefix_tokens != (
                    len(record.acquired_prefix_pages) * self.page_size
                ):
                    raise InvariantViolationError(
                        "reservation prefix refs disagree with its logical attachment"
                    )
                for page in () if record.committed else record.acquired_prefix_pages:
                    meta = self.allocator.get_meta(page)
                    if meta.allocation_state is not PageAllocationState.LIVE:
                        raise InvariantViolationError(
                            "tentative prefix ownership refers to a non-live page"
                        )
                    sequence_page_refs[page] = sequence_page_refs.get(page, 0) + 1

            if self._tier is not None:
                self._tier.assert_invariants()
                # A spill in flight holds an ordinary request reference on its
                # source page; it must be counted like any other owner.
                for page in self._tier.request_ref_pages():
                    sequence_page_refs[page] = sequence_page_refs.get(page, 0) + 1

            for page, expected_refs in sequence_page_refs.items():
                meta = self.allocator.get_meta(page)
                if meta.request_refs != expected_refs:
                    raise InvariantViolationError(
                        "physical page request refs do not match sequence ownership"
                    )

            reserved_pages = {
                page
                for record in self._reservations.values()
                if not record.committed
                for page in record.allocated_pages
            }
            if self._tier is not None:
                # Promotion destinations are reservations the tier owns
                # directly; no transaction ever sees them.
                reserved_pages.update(self._tier.reserved_pages())
            if len(reserved_pages) != self.allocator.snapshot().reserved_pages:
                raise InvariantViolationError(
                    "reservation registry does not match reserved-page accounting"
                )
            for index, (match_handle, match) in self._prefix_matches.items():
                if index != match_handle.index or match.matched_tokens <= 0:
                    raise InvariantViolationError("invalid scheduler-facing prefix match registry")
            padding = self.allocator.get_meta(self.padding_page)
            if padding.allocation_state is not PageAllocationState.PERMANENT:
                raise InvariantViolationError("padding page lost permanent ownership")

    def _plan_append(
        self,
        existing_entries: tuple[PageTableEntry, ...],
        allocated_pages: tuple[KVPageHandle, ...],
        *,
        base_committed_tokens: int,
        num_new_tokens: int,
    ) -> tuple[tuple[PageTableEntry, ...], tuple[KVWriteSlot, ...]]:
        """Build the post-append page table and per-token write slots.

        Walks each new token's logical position, appending a new reserved page
        exactly when the position crosses into a fresh block. Validates that
        the append starts at the committed tail (``page_offset == valid_tokens``)
        and consumes every reserved page (no gaps, no leftovers).

        Returns:
            ``(planned_entries, write_slots)`` for the full post-append table.
        """
        mutable_entries: list[list[object]] = [
            [entry.page, entry.valid_tokens] for entry in existing_entries
        ]
        allocated_iter = iter(allocated_pages)
        write_slots: list[KVWriteSlot] = []

        for logical_position in range(
            base_committed_tokens,
            base_committed_tokens + num_new_tokens,
        ):
            block_index = logical_position // self.page_size
            page_offset = logical_position % self.page_size
            if block_index == len(mutable_entries):
                try:
                    new_page = next(allocated_iter)
                except StopIteration as exc:
                    raise InvariantViolationError("append plan ran out of reserved pages") from exc
                mutable_entries.append([new_page, 0])
            if block_index >= len(mutable_entries):
                raise InvariantViolationError("append plan produced a page-table gap")

            page = mutable_entries[block_index][0]
            valid_tokens = mutable_entries[block_index][1]
            if not isinstance(page, KVPageHandle) or not isinstance(valid_tokens, int):
                raise InvariantViolationError("invalid mutable page-table entry")
            # The tail-fill rule: tokens always write at the first free offset
            # of their block, so the offset equals the current valid count.
            if page_offset != valid_tokens:
                raise InvariantViolationError(
                    "append offset does not follow committed tail position"
                )
            mutable_entries[block_index][1] = valid_tokens + 1
            physical_page = self.allocator.physical_id(page).value
            write_slots.append(
                KVWriteSlot(
                    logical_position=logical_position,
                    physical_page=physical_page,
                    page_offset=page_offset,
                    flat_slot=physical_page * self.page_size + page_offset,
                )
            )

        # Every reserved page must be used exactly once by the append.
        try:
            next(allocated_iter)
        except StopIteration:
            pass
        else:
            raise InvariantViolationError("append plan left reserved pages unused")

        planned_entries: list[PageTableEntry] = []
        for entry in mutable_entries:
            page = entry[0]
            valid_tokens = entry[1]
            if not isinstance(page, KVPageHandle) or not isinstance(valid_tokens, int):
                raise InvariantViolationError("invalid mutable page-table entry")
            planned_entries.append(PageTableEntry(page=page, valid_tokens=valid_tokens))
        planned = tuple(planned_entries)
        return planned, tuple(write_slots)

    def _rollback_transaction_locked(self, tx: TransactionRecord) -> None:
        """Undo layout-specific reservations through the shared transaction flow."""
        affected_sequences: list[SequenceHandle] = []

        def undo() -> None:
            for reservation_handle in tx.reservation_handles:
                record = self._get_reservation(reservation_handle)
                self.allocator.rollback_reserved(record.allocated_pages)
                self._release_tentative_prefix_pages(record.acquired_prefix_pages)
                with self.sequences.mutate(record.sequence) as state:
                    if state.pending_transaction_index == tx.handle.index:
                        state.pending_transaction_index = None
                    affected_sequences.append(record.sequence)
                    self._reservations.pop(reservation_handle.index)

        def finish() -> None:
            for sequence in affected_sequences:
                with self.sequences.mutate(sequence) as state:
                    if state.release_requested:
                        self._release_sequence_locked(
                            sequence,
                            safe_epoch=state.release_safe_epoch,
                        )

        TransactionOrchestrator.rollback(
            tx,
            undo=undo,
            finish=finish,
            transactions=self._transactions,
        )

    def _finish_lease_locked(
        self,
        lease: LeaseRecord,
        *,
        failure_safe_epoch: int | None = None,
    ) -> None:
        """Tear down a terminal lease under lock and honor deferred releases.

        Drops reservation records and the lease; for sequences that requested
        release while the lease was active, runs the release with an epoch that
        is at least as late as the failure's safe epoch.
        """
        affected_sequences: list[SequenceHandle] = []
        for reservation_handle in lease.reservation_handles:
            record = self._get_reservation(reservation_handle)
            affected_sequences.append(record.sequence)
            self._reservations.pop(reservation_handle.index)
        self._leases.pop(lease.handle.index)

        for sequence in affected_sequences:
            with self.sequences.mutate(sequence) as state:
                if state.release_requested:
                    safe_epoch = max(
                        state.release_safe_epoch,
                        failure_safe_epoch or self.current_epoch,
                    )
                    self._release_sequence_locked(sequence, safe_epoch=safe_epoch)

    def _release_sequence_locked(
        self,
        sequence: SequenceHandle,
        *,
        safe_epoch: int,
    ) -> None:
        """Drop every request ref of a sequence and recycle its arena slot.

        Requires the sequence to be idle (no transaction, no lease); pages are
        released with the given safe epoch and the slot is returned to the
        arena after clearing all committed state.
        """
        with self.sequences.mutate(sequence) as state:
            if state.pending_transaction_index is not None or state.active_lease_index is not None:
                raise SequenceBusyError("cannot immediately release a busy sequence")
            for entry in state.page_table.entries:
                self.allocator.release_request_ref(entry.page, safe_epoch=safe_epoch)
            state.page_table.entries.clear()
            state.committed_tokens = 0
            state.version += 1
            state.release_requested = False
            self.sequences.release(sequence)

    def _validate_lease_sequences(
        self,
        lease: LeaseRecord,
        records: Sequence[ReservationRecord],
    ) -> None:
        """Reject completion when any sequence changed during the lease."""
        for record in records:
            with self.sequences.mutate(record.sequence) as state:
                if (
                    state.active_lease_index != lease.handle.index
                    or state.version != record.base_sequence_version
                    or state.committed_tokens != record.base_committed_tokens
                ):
                    raise InvalidStateTransitionError(
                        "sequence changed while its execution lease was active"
                    )

    @staticmethod
    def _validate_written_counts(
        records: Sequence[ReservationRecord],
        written_tokens: Mapping[KVReservationHandle, int] | None,
    ) -> None:
        """Validate an optional written-token report against the reservations.

        Phase A1 requires every reserved token to be written, so partial writes
        are rejected structurally.
        """
        if written_tokens is None:
            return
        expected_handles = {record.handle for record in records}
        if set(written_tokens) != expected_handles:
            raise InvalidStateTransitionError(
                "written-token report does not match the step reservations"
            )
        for record in records:
            if written_tokens[record.handle] != record.num_new_tokens:
                raise InvalidStateTransitionError(
                    "Phase A1 requires every reserved token to be written"
                )

    def _lease_touched_pages(self, lease: LeaseRecord) -> tuple[KVPageHandle, ...]:
        """All unique pages a lease's steps may touch, deduplicated.

        ``dict.fromkeys`` keeps first-seen order so in-flight refs are counted
        once per page even when several reservations share a page.
        """
        pages: list[KVPageHandle] = []
        for reservation_handle in lease.reservation_handles:
            pages.extend(self._get_reservation(reservation_handle).touched_pages)
        return tuple(dict.fromkeys(pages))

    def _get_transaction(
        self,
        handle: MemoryTransactionHandle,
    ) -> TransactionRecord:
        """Resolve a transaction handle, rejecting stale generations."""
        record = self._transactions.get(handle.index)
        if record is None or record.handle != handle:
            raise InvalidHandleError(f"stale transaction handle: {handle}")
        return record

    def _consume_prefix_match_locked(
        self,
        handle: PrefixMatchHandle | None,
    ) -> PrefixMatch | None:
        """Consume a one-shot prefix-match handle under lock.

        The match is removed from the registry even on failure so it cannot be
        replayed; a stale or missing handle resolves to None.
        """
        if handle is None:
            return None
        if not isinstance(handle, PrefixMatchHandle):
            raise TypeError("prefix_match must be a PrefixMatchHandle or None")
        stored = self._prefix_matches.pop(handle.index, None)
        if stored is None or stored[0] != handle:
            return None
        return stored[1]

    def _release_tentative_prefix_pages(
        self,
        pages: Sequence[KVPageHandle],
        *,
        safe_epoch: int | None = None,
    ) -> None:
        """Release tentatively acquired prefix request refs (rollback path).

        Released in reverse order so nested acquires unwind safely.
        """
        epoch = self.current_epoch if safe_epoch is None else safe_epoch
        for page in reversed(tuple(pages)):
            self.allocator.release_request_ref(page, safe_epoch=epoch)

    def _get_reservation(self, handle: KVReservationHandle) -> ReservationRecord:
        """Resolve a reservation handle, rejecting stale generations."""
        record = self._reservations.get(handle.index)
        if record is None or record.handle != handle:
            raise InvalidHandleError(f"stale reservation handle: {handle}")
        return record

    def _get_lease(self, handle: StepMemoryLeaseHandle) -> LeaseRecord:
        """Resolve a lease handle, rejecting stale generations."""
        record = self._leases.get(handle.index)
        if record is None or record.handle != handle:
            raise InvalidHandleError(f"stale step-memory lease: {handle}")
        return record

    @staticmethod
    def _validate_storage(
        storage: object | None,
        *,
        total_pages: int,
        page_size: int,
    ) -> None:
        """Reject a storage whose geometry disagrees with the page allocator.

        A mismatched storage would silently alias pages; capacity and page size
        must match exactly.
        """
        if storage is None:
            return
        capacity = getattr(storage, "capacity_pages", None)
        storage_page_size = getattr(storage, "page_size", None)
        if capacity != total_pages or storage_page_size != page_size:
            raise ValueError("storage capacity/page_size must exactly match the page allocator")
