"""Fixed-size page allocator with generation checks, pin locks, and deferred reclamation.

Allocates generation-safe metadata identities into preallocated KV storage.
Deferred free defers recycling until a safe completion epoch, so a page can
never be handed out while a launched GPU step may still write to it.
"""

from __future__ import annotations

import heapq
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from threading import RLock

from ayaka.exceptions import (
    InvalidHandleError,
    InvalidStateTransitionError,
    InvariantViolationError,
)
from ayaka.handles import KVPageHandle, PhysicalPageId
from ayaka.kv.meta import PageMetadata
from ayaka.kv.state import PageAllocationState


@dataclass(frozen=True, slots=True)
class AllocatorSnapshot:
    """Immutable capacity and ownership counters for one allocator instance.

    The snapshot is a point-in-time report. It is computed while the
    allocator lock is held and can be safely inspected after the lock has been
    released. Page counts are counts of physical pages, not counts of
    references or tensor elements.

    Attributes:
        total_pages: Total number of physical page slots managed by the
            allocator, including permanent pages.
        usable_pages: ``total_pages`` minus permanent pages. These are the
            slots that can participate in normal allocation and reclamation.
        free_pages: Pages in ``FREE`` and ready to be allocated immediately.
        reserved_pages: Pages tentatively owned by a reservation and not yet
            committed to a request.
        live_pages: Pages in ``LIVE`` with committed request/cache ownership,
            or temporarily unowned while waiting for an in-flight step to
            finish.
        reclaim_pending_pages: Pages with no ownership references that are
            waiting for their safe reclamation epoch.
        permanent_pages: Pages reserved for allocator infrastructure, such as
            the padding page, and never returned to the free queue.
        request_owned_pages: Pages with at least one request reference.
        cache_owned_pages: Pages with at least one prefix-cache reference.
        pinned_pages: Pages with at least one explicit pin reference.
        shared_pages: Pages with more than one durable request/cache owner.
        inflight_pages: Pages with at least one launched execution step that
            may still access the page.
        reclaimed_pages_total: Cumulative number of pages returned to ``FREE``
            by deferred reclamation since allocator creation.
    """

    total_pages: int
    usable_pages: int
    free_pages: int
    reserved_pages: int
    live_pages: int
    reclaim_pending_pages: int
    permanent_pages: int

    request_owned_pages: int
    cache_owned_pages: int
    pinned_pages: int
    """Pages explicitly locked in memory (pin_refs > 0)."""
    shared_pages: int
    inflight_pages: int
    reclaimed_pages_total: int


class PageAllocator:
    """Thread-safe allocator for fixed-size pages in preallocated KV storage.

    The allocator owns metadata and page identities; it does not allocate or
    move the underlying K/V byte buffers. A :class:`~ayaka.handles.KVPageHandle`
    identifies a physical slot by ``(index, generation)``. The index addresses
    storage, while the generation changes whenever the slot is reused and
    makes handles from an older lifetime invalid.

    Normal page lifetimes follow this state machine::

        FREE -> RESERVED -> LIVE -> RECLAIM_PENDING -> FREE

    ``RESERVED`` is a tentative transaction state. ``commit_reserved`` changes
    it to ``LIVE``; ``rollback_reserved`` can return it directly to ``FREE``
    when no execution step is using it. If a reservation is abandoned after a
    launch, ``abandon_reserved`` defers reclamation until the supplied safe
    epoch. ``PERMANENT`` is a terminal side branch used for infrastructure
    pages and is excluded from usable capacity.

    Four durable/transient ownership dimensions are tracked independently:
    request references, cache references, reservation references, and
    in-flight references. Explicit pin references are also ownership for the
    purpose of reclamation. A page is never recycled while any reference is
    present or while its safe completion epoch has not been reached.

    All public operations that inspect or mutate allocator state are protected
    by one re-entrant lock. Handles should be treated as opaque by callers;
    use :meth:`get_meta` for a detached metadata snapshot rather than mutating
    allocator-owned metadata directly.
    """

    def __init__(self, *, total_pages: int, page_size: int) -> None:
        """Create an allocator with ``total_pages`` empty fixed-size slots.

        Args:
            total_pages: Number of physical page slots to manage. Every slot
                starts in ``FREE`` state and is immediately available for
                normal allocation.
            page_size: Number of KV tokens that fit in one page. The allocator
                uses this value to validate ``valid_tokens`` and to initialize
                a permanent padding page as completely valid.

        Raises:
            ValueError: If ``total_pages`` or ``page_size`` is not positive.

        Notes:
            Construction only creates metadata. The caller remains responsible
            for allocating storage buffers whose page indexing matches the
            resulting ``PhysicalPageId`` values.
        """
        if total_pages <= 0:
            raise ValueError("total_pages must be positive")
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        self._total_pages = total_pages
        self._page_size = page_size
        self._pages = [
            PageMetadata(physical_id=PhysicalPageId(index), generation=0)
            for index in range(total_pages)
        ]
        self._ready_free: deque[int] = deque(range(total_pages))
        self._reclaim_heap: list[tuple[int, int, int]] = []
        self._current_epoch = 0
        self._reclaimed_pages_total = 0
        self._lock = RLock()

    @property
    def total_pages(self) -> int:
        """Return the total number of physical slots, including permanent ones."""
        return self._total_pages

    @property
    def page_size(self) -> int:
        """Return the maximum number of valid tokens that fit in one page."""
        return self._page_size

    @property
    def current_epoch(self) -> int:
        """Return the greatest completed execution epoch observed so far."""
        return self._current_epoch

    def reserve_permanent_page(self) -> KVPageHandle:
        """Remove one free page permanently from normal allocator capacity.

        This is intended for infrastructure pages such as an all-padding KV
        page. The selected slot is assigned a new generation, marked
        ``PERMANENT``, and initialized with ``page_size`` valid tokens. A
        permanent page has no request/cache owner but is deliberately never
        reclaimed or returned by :meth:`allocate`.

        Returns:
            A generation-safe handle for the permanent page.

        Raises:
            InvalidStateTransitionError: If no page is currently in the ready
                free queue.
        """
        with self._lock:
            if not self._ready_free:
                raise InvalidStateTransitionError(
                    "no page is available for the permanent padding page"
                )
            index = self._ready_free.popleft()
            meta = self._pages[index]
            self._reset_for_new_generation(meta)
            meta.allocation_state = PageAllocationState.PERMANENT
            meta.valid_tokens = self._page_size
            return KVPageHandle(index=index, generation=meta.generation)

    def allocate(self, num_pages: int) -> tuple[KVPageHandle, ...] | None:
        """Reserve a batch of available page identities logically.

        Allocation is all-or-nothing: either ``num_pages`` pages are removed
        from the ready-free queue and returned as ``RESERVED`` handles, or no
        page is changed and ``None`` is returned. Each returned page starts a
        new generation and has exactly one reservation reference. Reclamation
        eligible at the allocator's current epoch is processed before
        capacity is checked.

        Args:
            num_pages: Number of page slots to reserve. Zero is a valid no-op
                and returns an empty tuple.

        Returns:
            A tuple of unique ``RESERVED`` page handles, or ``None`` when the
            requested batch does not fit in currently ready capacity.

        Raises:
            ValueError: If ``num_pages`` is negative.

        Notes:
            This method reserves metadata only. Callers must later call either
            :meth:`commit_reserved`, :meth:`rollback_reserved`, or
            :meth:`abandon_reserved` for every returned handle.
        """
        if num_pages < 0:
            raise ValueError("num_pages must be non-negative")
        if num_pages == 0:
            return ()
        with self._lock:
            self._reclaim_completed_locked(self._current_epoch)
            if len(self._ready_free) < num_pages:
                return None

            indices = [self._ready_free.popleft() for _ in range(num_pages)]
            handles: list[KVPageHandle] = []
            for index in indices:
                meta = self._pages[index]
                self._reset_for_new_generation(meta)
                meta.allocation_state = PageAllocationState.RESERVED
                meta.reservation_refs = 1
                handles.append(KVPageHandle(index=index, generation=meta.generation))
            return tuple(handles)

    def commit_reserved(self, pages: Sequence[KVPageHandle]) -> None:
        """Commit tentative reservations as live request-owned pages.

        Every page must still be ``RESERVED`` and must have exactly one
        reservation owner. The reservation reference is replaced by one
        request reference and the page transitions to ``LIVE``. The operation
        validates the complete batch before mutating any page, so an invalid
        handle or state leaves the batch unchanged.

        Args:
            pages: Reserved handles to commit. An empty sequence is a no-op.

        Raises:
            InvalidHandleError: If a handle is stale or outside the allocator.
            InvalidStateTransitionError: If any page is not ``RESERVED``.
            InvariantViolationError: If a reserved page does not have exactly
                one reservation reference.
        """
        with self._lock:
            metas = [self._require_state(page, PageAllocationState.RESERVED) for page in pages]
            for meta in metas:
                if meta.reservation_refs != 1:
                    raise InvariantViolationError(
                        "a reserved page must have exactly one reservation owner"
                    )
            for meta in metas:
                meta.reservation_refs = 0
                meta.request_refs += 1
                meta.allocation_state = PageAllocationState.LIVE

    def rollback_reserved(self, pages: Sequence[KVPageHandle]) -> None:
        """Cancel reservations and return their pages to the free queue.

        Rollback is the immediate, pre-launch cancellation path. Every page
        must still be ``RESERVED`` and have exactly one reservation reference;
        an in-flight page cannot be rolled back because a launched step may
        still access its storage. Valid pages are reset to ``FREE`` and their
        generation remains available for the next allocation lifetime.

        Args:
            pages: Reserved handles to cancel. An empty sequence is a no-op.

        Raises:
            InvalidHandleError: If a handle is stale or outside the allocator.
            InvalidStateTransitionError: If a page is not reserved or is
                already involved in an in-flight execution step.
            InvariantViolationError: If a reserved page has an invalid
                reservation reference count.
        """
        with self._lock:
            metas = [self._require_state(page, PageAllocationState.RESERVED) for page in pages]
            for meta in metas:
                if meta.inflight_refs:
                    raise InvalidStateTransitionError(
                        "an in-flight reserved page cannot be rolled back immediately"
                    )
                if meta.reservation_refs != 1:
                    raise InvariantViolationError("invalid reservation refcount")
            for meta in metas:
                meta.reservation_refs = 0
                self._make_free(meta)

    def abandon_reserved(
        self,
        pages: Sequence[KVPageHandle],
        *,
        safe_epoch: int,
    ) -> None:
        """Abandon reservations whose storage may still be in use by a launch.

        Unlike :meth:`rollback_reserved`, this path does not require the page
        to be freeable immediately. Reservation ownership is removed and an
        unowned page is placed on the deferred-reclamation path. If it has
        in-flight references, the page remains protected until both those
        references are released and ``safe_epoch`` has completed.

        Args:
            pages: Reserved handles to abandon. Each page must still have one
                reservation owner.
            safe_epoch: Earliest execution epoch at which the page's storage
                may be recycled. It must be non-negative.

        Raises:
            ValueError: If ``safe_epoch`` is negative.
            InvalidHandleError: If a handle is stale or outside the allocator.
            InvalidStateTransitionError: If a page is not ``RESERVED``.
            InvariantViolationError: If a reserved page has an invalid
                reservation reference count.
        """
        self._validate_epoch(safe_epoch)
        with self._lock:
            metas = [self._require_state(page, PageAllocationState.RESERVED) for page in pages]
            for meta in metas:
                if meta.reservation_refs != 1:
                    raise InvariantViolationError("invalid reservation refcount")
            for meta in metas:
                meta.reservation_refs = 0
                self._mark_for_reclaim_if_unowned(meta, safe_epoch=safe_epoch)

    def mark_inflight(self, pages: Iterable[KVPageHandle]) -> None:
        """Record that a launched execution step may access these pages.

        In-flight references are transient protection against premature reuse;
        they do not represent request or cache ownership. Duplicate handles in
        ``pages`` are collapsed so one step increments each page exactly once.
        Pages may be ``RESERVED`` or ``LIVE`` when the step is launched.

        Args:
            pages: Handles read or written by one execution step.

        Raises:
            InvalidHandleError: If a handle is stale or outside the allocator.
            InvalidStateTransitionError: If a page is not ``RESERVED`` or
                ``LIVE``.
        """
        unique = self._unique_pages(pages)
        with self._lock:
            metas = [self._require_live_or_reserved(page) for page in unique]
            for meta in metas:
                meta.inflight_refs += 1

    def unmark_inflight(self, pages: Iterable[KVPageHandle]) -> None:
        """Release one execution-step protection reference from each page.

        Duplicate handles are collapsed symmetrically with
        :meth:`mark_inflight`. When the last in-flight reference disappears
        from an otherwise unowned page, its pending safe epoch is converted
        into a reclaim-heap entry so a later epoch advancement can free it.

        Args:
            pages: Handles previously marked for the completing execution
                step.

        Raises:
            InvalidHandleError: If a handle is stale or outside the allocator.
            InvalidStateTransitionError: If a page is no longer ``RESERVED``
                or ``LIVE``.
            InvariantViolationError: If the in-flight reference count would
                become negative.
        """
        unique = self._unique_pages(pages)
        with self._lock:
            metas = [self._require_live_or_reserved(page) for page in unique]
            for meta in metas:
                if meta.inflight_refs <= 0:
                    raise InvariantViolationError("in-flight refcount underflow")
            for meta in metas:
                meta.inflight_refs -= 1
                if meta.ownership_refs == 0 and meta.pending_free_epoch is not None:
                    self._mark_for_reclaim_if_unowned(
                        meta,
                        safe_epoch=meta.pending_free_epoch,
                    )

    def set_valid_tokens(self, page: KVPageHandle, valid_tokens: int) -> None:
        """Set how many leading token positions in a page contain valid KV.

        The value is metadata only; this method does not write or clear the
        underlying K/V storage. A page can be updated while ``RESERVED`` or
        ``LIVE`` and must still be identified by a current generation-safe
        handle.

        Args:
            page: Current handle of the page to update.
            valid_tokens: Number of valid tokens, inclusive of zero and at
                most ``page_size``.

        Raises:
            ValueError: If ``valid_tokens`` lies outside ``[0, page_size]``.
            InvalidHandleError: If ``page`` is stale or outside the allocator.
            InvalidStateTransitionError: If the page is not ``RESERVED`` or
                ``LIVE``.
        """
        if not 0 <= valid_tokens <= self._page_size:
            raise ValueError("valid_tokens is outside the page boundary")
        with self._lock:
            meta = self._require_live_or_reserved(page)
            meta.valid_tokens = valid_tokens

    def acquire_request_ref(self, page: KVPageHandle) -> None:
        """Add one durable request/page-table owner to a live page.

        Args:
            page: Current ``LIVE`` page handle.

        Raises:
            InvalidHandleError: If ``page`` is stale or outside the allocator.
            InvalidStateTransitionError: If the page is not ``LIVE``.
        """
        with self._lock:
            meta = self._require_state(page, PageAllocationState.LIVE)
            meta.request_refs += 1

    def release_request_ref(self, page: KVPageHandle, *, safe_epoch: int) -> None:
        """Release one request owner and schedule safe reclamation if needed.

        If this was the final durable owner, the page is marked with the
        greatest required safe epoch. It is reclaimed immediately only when no
        in-flight reference remains; otherwise it waits for
        :meth:`unmark_inflight` and epoch advancement.

        Args:
            page: Current ``LIVE`` page handle.
            safe_epoch: Earliest epoch at which the page may be recycled.

        Raises:
            ValueError: If ``safe_epoch`` is negative.
            InvalidHandleError: If ``page`` is stale or outside the allocator.
            InvalidStateTransitionError: If the page is not ``LIVE``.
            InvariantViolationError: If no request reference exists to
                release.
        """
        self._validate_epoch(safe_epoch)
        with self._lock:
            meta = self._require_state(page, PageAllocationState.LIVE)
            if meta.request_refs <= 0:
                raise InvariantViolationError("request refcount underflow")
            meta.request_refs -= 1
            self._mark_for_reclaim_if_unowned(meta, safe_epoch=safe_epoch)

    def acquire_cache_ref(self, page: KVPageHandle) -> None:
        """Add one durable prefix-cache owner to a live page.

        Args:
            page: Current ``LIVE`` page handle.

        Raises:
            InvalidHandleError: If ``page`` is stale or outside the allocator.
            InvalidStateTransitionError: If the page is not ``LIVE``.
        """
        with self._lock:
            meta = self._require_state(page, PageAllocationState.LIVE)
            meta.cache_refs += 1

    def release_cache_ref(self, page: KVPageHandle, *, safe_epoch: int) -> None:
        """Release one cache owner and defer recycling until it is safe.

        Args:
            page: Current ``LIVE`` page handle.
            safe_epoch: Earliest epoch at which the page may be recycled.

        Raises:
            ValueError: If ``safe_epoch`` is negative.
            InvalidHandleError: If ``page`` is stale or outside the allocator.
            InvalidStateTransitionError: If the page is not ``LIVE``.
            InvariantViolationError: If no cache reference exists to release.
        """
        self._validate_epoch(safe_epoch)
        with self._lock:
            meta = self._require_state(page, PageAllocationState.LIVE)
            if meta.cache_refs <= 0:
                raise InvariantViolationError("cache refcount underflow")
            meta.cache_refs -= 1
            self._mark_for_reclaim_if_unowned(meta, safe_epoch=safe_epoch)

    def pin_page(self, page: KVPageHandle) -> None:
        """Add an explicit pin that prevents eviction and in-place mutation.

        Pins are counted, so callers that pin a page multiple times must call
        :meth:`unpin_page` the same number of times. A pinned page remains
        allocator-owned even when it has no request or cache references and
        therefore cannot enter deferred reclamation until all pins are gone.

        Args:
            page: Current ``LIVE`` page handle to pin.

        Raises:
            InvalidHandleError: If ``page`` is stale or outside the allocator.
            InvalidStateTransitionError: If the page is not ``LIVE``.
        """
        with self._lock:
            meta = self._require_state(page, PageAllocationState.LIVE)
            meta.pin()

    def unpin_page(self, page: KVPageHandle, *, safe_epoch: int) -> None:
        """Release one explicit pin and schedule reclamation when unowned.

        If dropping this pin leaves the page without request, cache,
        reservation, or pin ownership, it enters deferred reclamation for
        ``safe_epoch``. An in-flight reference can still keep the page in its
        current live state until the execution step completes.

        Args:
            page: Current ``LIVE`` page handle to unpin.
            safe_epoch: Earliest epoch at which the page may be recycled if
                this was its final ownership reference.

        Raises:
            ValueError: If ``safe_epoch`` is negative.
            InvalidHandleError: If ``page`` is stale or outside the allocator.
            InvalidStateTransitionError: If the page is not ``LIVE``.
            InvariantViolationError: If the page has no pin to release.
        """
        self._validate_epoch(safe_epoch)
        with self._lock:
            meta = self._require_state(page, PageAllocationState.LIVE)
            try:
                meta.unpin()
            except ValueError as err:
                raise InvariantViolationError(str(err)) from err
            self._mark_for_reclaim_if_unowned(meta, safe_epoch=safe_epoch)

    def advance_epoch(self, completed_epoch: int) -> int:
        """Advance the completed execution epoch and reclaim eligible pages.

        Epochs are monotonic completion markers supplied by the execution
        layer. Every reclaim candidate whose safe epoch is at most
        ``completed_epoch`` is returned to the ready-free queue, subject to a
        final generation, state, epoch, and reference check.

        Args:
            completed_epoch: New completed epoch. It may equal the current
                epoch but may not be lower.

        Returns:
            Number of pages reclaimed during this call.

        Raises:
            ValueError: If ``completed_epoch`` is negative or less than the
                current epoch.
        """
        self._validate_epoch(completed_epoch)
        with self._lock:
            if completed_epoch < self._current_epoch:
                raise ValueError("completed epoch must be monotonic")
            self._current_epoch = completed_epoch
            return self._reclaim_completed_locked(completed_epoch)

    def reclaim_completed(self) -> int:
        """Reclaim pages eligible at the current epoch without advancing it.

        Returns:
            Number of pages moved from ``RECLAIM_PENDING`` to ``FREE``.

        Raises:
            InvariantViolationError: If the reclaim heap contains a candidate
                that still has references.
        """
        with self._lock:
            return self._reclaim_completed_locked(self._current_epoch)

    def available_pages(self) -> int:
        """Return the number of pages immediately available for allocation.

        This count is the ready-free queue length. It excludes reserved, live,
        reclaim-pending, and permanent pages; it does not proactively advance
        the epoch or reclaim heap entries that have become eligible at the
        current epoch. :meth:`allocate` performs that reclamation as part of
        its own capacity check.

        Returns:
            Number of pages currently present in the ready-free queue.
        """
        with self._lock:
            return len(self._ready_free)

    def physical_id(self, handle: KVPageHandle) -> PhysicalPageId:
        """Resolve a generation-safe handle to its stable storage identity.

        The returned physical ID contains only the page index used by storage
        buffers and kernels. Generation validation happens before returning it,
        so a stale handle cannot be translated into a currently reused slot.

        Args:
            handle: Current or stale page handle to validate and resolve.

        Returns:
            Stable :class:`~ayaka.handles.PhysicalPageId` for the page slot.

        Raises:
            InvalidHandleError: If the index is outside the allocator, the
                generation does not match, or the page is ``FREE``.
        """
        with self._lock:
            return self._get_meta(handle).physical_id

    def get_meta(self, handle: KVPageHandle) -> PageMetadata:
        """Return a detached metadata snapshot for a current page handle.

        The returned :class:`~ayaka.kv.meta.PageMetadata` is a copy. Inspecting
        it does not require holding the allocator lock, and mutating the copy
        cannot mutate allocator state. The handle is validated while the lock
        is held before the copy is made.

        Args:
            handle: Current page handle to inspect.

        Returns:
            A snapshot of the page's generation, state, residency, refcounts,
            token validity, and epoch metadata.

        Raises:
            InvalidHandleError: If ``handle`` is stale or outside the
                allocator, or refers to a free page.
        """
        with self._lock:
            return self._get_meta(handle).snapshot()

    def snapshot(self) -> AllocatorSnapshot:
        """Return an O(N) point-in-time capacity and ownership report.

        The method scans each physical page once while holding the allocator
        lock. The report distinguishes lifecycle state counts from ownership
        counts: one page with several request references contributes one to
        ``request_owned_pages``, not several.

        Returns:
            An immutable :class:`AllocatorSnapshot` detached from subsequent
            allocator mutations.
        """
        with self._lock:
            free_c = reserved_c = live_c = reclaim_c = perm_c = 0
            req_owned = cache_owned = pinned_c = shared_c = inflight_c = 0

            for meta in self._pages:
                state = meta.allocation_state
                if state is PageAllocationState.FREE:
                    free_c += 1
                elif state is PageAllocationState.LIVE:
                    live_c += 1
                elif state is PageAllocationState.RESERVED:
                    reserved_c += 1
                elif state is PageAllocationState.RECLAIM_PENDING:
                    reclaim_c += 1
                elif state is PageAllocationState.PERMANENT:
                    perm_c += 1

                if meta.request_refs > 0:
                    req_owned += 1
                if meta.cache_refs > 0:
                    cache_owned += 1
                if meta.is_pinned:
                    pinned_c += 1
                if meta.is_shared:
                    shared_c += 1
                if meta.inflight_refs > 0:
                    inflight_c += 1

            return AllocatorSnapshot(
                total_pages=self._total_pages,
                usable_pages=self._total_pages - perm_c,
                free_pages=free_c,
                reserved_pages=reserved_c,
                live_pages=live_c,
                reclaim_pending_pages=reclaim_c,
                permanent_pages=perm_c,
                request_owned_pages=req_owned,
                cache_owned_pages=cache_owned,
                pinned_pages=pinned_c,
                shared_pages=shared_c,
                inflight_pages=inflight_c,
                reclaimed_pages_total=self._reclaimed_pages_total,
            )

    def assert_invariants(self) -> None:
        """Validate the allocator's internal state and ownership invariants.

        Checks include uniqueness and state consistency of the ready-free
        queue, non-negative refcounts, page-boundary token counts, valid
        reservation rules, deferred-reclamation requirements, and a complete
        partition of all physical pages across lifecycle states. This method
        is intended for tests, diagnostics, and debug assertions; it performs
        an O(N) scan and raises at the first violation.

        Raises:
            InvariantViolationError: If any allocator bookkeeping invariant is
                broken, including stale queue membership, invalid references,
                an unowned page without deferred reclamation, or a state
                partition that does not equal ``total_pages``.
        """
        with self._lock:
            free_indices = list(self._ready_free)
            if len(free_indices) != len(set(free_indices)):
                raise InvariantViolationError("duplicate page in ready-free queue")
            free_set = set(free_indices)

            for index, meta in enumerate(self._pages):
                counts = (
                    meta.request_refs,
                    meta.cache_refs,
                    meta.reservation_refs,
                    meta.inflight_refs,
                    meta.pin_refs,
                )
                if any(count < 0 for count in counts):
                    raise InvariantViolationError("negative page refcount")
                if not 0 <= meta.valid_tokens <= self._page_size:
                    raise InvariantViolationError("invalid page token count")

                if meta.allocation_state is PageAllocationState.FREE:
                    if index not in free_set:
                        raise InvariantViolationError("free page missing from free queue")
                    if meta.total_refs or meta.valid_tokens or meta.pending_free_epoch is not None:
                        raise InvariantViolationError("free page retains live metadata")
                else:
                    if index in free_set:
                        raise InvariantViolationError("allocated page appears in free queue")

                if meta.allocation_state is PageAllocationState.RESERVED:
                    if meta.reservation_refs not in (0, 1):
                        raise InvariantViolationError("invalid reserved-page owner count")
                    if meta.pin_refs != 0:
                        raise InvariantViolationError("reserved page cannot be pinned")
                    if meta.reservation_refs == 0 and not (
                        meta.inflight_refs > 0 and meta.pending_free_epoch is not None
                    ):
                        raise InvariantViolationError("unowned reserved page is not in flight")
                elif meta.allocation_state is PageAllocationState.LIVE:
                    if meta.reservation_refs:
                        raise InvariantViolationError("live page retains reservation owner")
                    if meta.ownership_refs == 0 and not (
                        meta.inflight_refs > 0 and meta.pending_free_epoch is not None
                    ):
                        raise InvariantViolationError("unowned live page is not deferred")
                elif meta.allocation_state is PageAllocationState.RECLAIM_PENDING:
                    if meta.total_refs:
                        raise InvariantViolationError("deferred page still has references")
                    if meta.pending_free_epoch is None:
                        raise InvariantViolationError("deferred page has no safe epoch")
                elif meta.allocation_state is PageAllocationState.PERMANENT:
                    if meta.pending_free_epoch is not None:
                        raise InvariantViolationError("permanent page cannot be reclaimed")

            snapshot = self.snapshot()
            partition = (
                snapshot.free_pages
                + snapshot.reserved_pages
                + snapshot.live_pages
                + snapshot.reclaim_pending_pages
                + snapshot.permanent_pages
            )
            if partition != self._total_pages:
                raise InvariantViolationError("page-state partition does not equal capacity")

    def _get_meta(self, handle: KVPageHandle) -> PageMetadata:
        """Return mutable metadata after validating a generation-safe handle.

        This is the central stale-handle check used by all allocator
        operations. Callers must already hold ``_lock``; the method deliberately
        returns the live internal object because its callers either inspect it
        under that lock or mutate it as part of an atomic public operation.

        Args:
            handle: Handle containing the physical index and expected
                generation.

        Returns:
            The allocator-owned metadata for the referenced page.

        Raises:
            InvalidHandleError: If the index is outside the allocator, the
                generation is stale, or the slot is currently ``FREE``.
        """
        if handle.index >= self._total_pages:
            raise InvalidHandleError(f"page index {handle.index} is out of range")
        meta = self._pages[handle.index]
        if (
            meta.generation != handle.generation
            or meta.allocation_state is PageAllocationState.FREE
        ):
            raise InvalidHandleError(f"stale page handle: {handle}")
        return meta

    def _require_state(
        self,
        handle: KVPageHandle,
        expected: PageAllocationState,
    ) -> PageMetadata:
        """Validate a handle and require one exact lifecycle state.

        Args:
            handle: Generation-safe page handle to validate.
            expected: The only lifecycle state accepted for the operation.

        Returns:
            Mutable allocator-owned metadata for the validated page.

        Raises:
            InvalidHandleError: If ``handle`` is stale, free, or out of range.
            InvalidStateTransitionError: If the page is valid but is not in
                ``expected`` state.
        """
        meta = self._get_meta(handle)
        if meta.allocation_state is not expected:
            raise InvalidStateTransitionError(
                f"page {handle} is {meta.allocation_state.name}, expected {expected.name}"
            )
        return meta

    def _require_live_or_reserved(self, handle: KVPageHandle) -> PageMetadata:
        """Validate a handle for an execution step that accepts active pages.

        ``mark_inflight`` and ``unmark_inflight`` may operate on both tentative
        reservations and committed live pages. Permanent, reclaim-pending, and
        free pages cannot be part of a launched step.

        Args:
            handle: Generation-safe page handle to validate.

        Returns:
            Mutable allocator-owned metadata for a ``RESERVED`` or ``LIVE``
            page.

        Raises:
            InvalidHandleError: If ``handle`` is stale, free, or out of range.
            InvalidStateTransitionError: If the page is not ``RESERVED`` or
                ``LIVE``.
        """
        meta = self._get_meta(handle)
        if meta.allocation_state not in (
            PageAllocationState.RESERVED,
            PageAllocationState.LIVE,
        ):
            raise InvalidStateTransitionError(
                f"page {handle} cannot participate in an execution step"
            )
        return meta

    def _reset_for_new_generation(self, meta: PageMetadata) -> None:
        """Start a new ownership lifetime for a page known to be free.

        The physical slot is reused in place, so all ownership, execution,
        pinning, validity, and pending-reclamation metadata must be cleared
        before the caller assigns the new lifecycle state. Incrementing the
        generation first invalidates every handle from the previous lifetime.

        Args:
            meta: Allocator-owned metadata whose state must be ``FREE``.

        Raises:
            InvariantViolationError: If the supplied metadata is not free.
        """
        if meta.allocation_state is not PageAllocationState.FREE:
            raise InvariantViolationError("only a free page can start a new generation")
        meta.generation += 1
        meta.request_refs = 0
        meta.cache_refs = 0
        meta.reservation_refs = 0
        meta.inflight_refs = 0
        meta.pin_refs = 0
        meta.valid_tokens = 0
        meta.pending_free_epoch = None
        meta.last_access_epoch = self._current_epoch

    def _make_free(self, meta: PageMetadata) -> None:
        """Return an unreferenced page to the ready-free queue.

        This helper performs the final reclamation transition. It does not
        increment the generation; that happens only when a subsequent
        allocation starts a new lifetime. The caller must hold ``_lock`` and
        must have already established that no durable or in-flight reference
        remains.

        Args:
            meta: Allocator-owned page metadata to transition to ``FREE``.

        Raises:
            InvariantViolationError: If any reference remains on the page.
        """
        index = meta.physical_id.value
        if meta.total_refs:
            raise InvariantViolationError("cannot free a referenced page")
        meta.allocation_state = PageAllocationState.FREE
        meta.valid_tokens = 0
        meta.pending_free_epoch = None
        meta.last_access_epoch = self._current_epoch
        self._ready_free.append(index)

    def _mark_for_reclaim_if_unowned(self, meta: PageMetadata, *, safe_epoch: int) -> None:
        """Record deferred reclamation for an otherwise unowned page.

        The required epoch is monotonic for a page lifetime: a later release
        cannot weaken an earlier safety requirement. If an in-flight reference
        remains, the page stays active and ``unmark_inflight`` will enqueue it
        once the last transient reference is gone. Otherwise the page is
        marked ``RECLAIM_PENDING`` immediately and a generation-tagged heap
        entry is created.

        Args:
            meta: Allocator-owned page metadata to inspect and possibly mark.
            safe_epoch: Earliest epoch at which storage reuse is safe. Public
                callers validate non-negativity before reaching this helper.

        Notes:
            This helper is intentionally idempotent with respect to repeated
            ownership-release notifications. The heap may contain stale or
            superseded entries; the reclaim path filters them by generation,
            state, and current pending epoch.
        """
        if meta.ownership_refs != 0:
            return
        previous = meta.pending_free_epoch
        meta.pending_free_epoch = max(previous or safe_epoch, safe_epoch)
        if meta.inflight_refs:
            return
        meta.allocation_state = PageAllocationState.RECLAIM_PENDING
        heapq.heappush(
            self._reclaim_heap,
            (meta.pending_free_epoch, meta.physical_id.value, meta.generation),
        )

    def _reclaim_completed_locked(self, completed_epoch: int) -> int:
        """Reclaim heap entries whose safe epoch has completed.

        The caller must hold ``_lock``. Each heap entry carries the epoch,
        physical index, and generation observed when it was queued. Those
        fields allow entries superseded by a later release or page reuse to be
        ignored safely instead of freeing the wrong lifetime.

        Args:
            completed_epoch: Completion boundary; entries at or below this
                epoch are candidates for reclamation.

        Returns:
            Number of pages actually transitioned to ``FREE``.

        Raises:
            InvariantViolationError: If an entry claims a reclaim-pending page
                still has references.
        """
        reclaimed = 0
        while self._reclaim_heap and self._reclaim_heap[0][0] <= completed_epoch:
            epoch, index, generation = heapq.heappop(self._reclaim_heap)
            meta = self._pages[index]
            if meta.generation != generation:
                continue
            if meta.allocation_state is not PageAllocationState.RECLAIM_PENDING:
                continue
            if meta.pending_free_epoch != epoch:
                continue
            if meta.total_refs:
                raise InvariantViolationError("reclaim queue contains a referenced page")
            self._make_free(meta)
            reclaimed += 1
            self._reclaimed_pages_total += 1
        return reclaimed

    @staticmethod
    def _unique_pages(pages: Iterable[KVPageHandle]) -> tuple[KVPageHandle, ...]:
        """Deduplicate page handles while preserving their input order.

        A single execution step may mention the same page more than once. The
        allocator treats that step as one in-flight owner for that page, so
        marking and unmarking must use the same deduplication rule.

        Args:
            pages: Iterable of page handles, possibly containing duplicates.

        Returns:
            A tuple containing the first occurrence of each distinct handle.
        """
        return tuple(dict.fromkeys(pages))

    @staticmethod
    def _validate_epoch(epoch: int) -> None:
        """Reject negative epoch values before they enter reclamation state.

        Args:
            epoch: Execution completion/safety epoch to validate.

        Raises:
            ValueError: If ``epoch`` is negative.
        """
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
