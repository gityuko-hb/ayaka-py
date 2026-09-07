from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import RLock

from ayaka.exceptions import (
    InvalidHandleError,
    InvalidStateTransitionError,
    SequenceCapacityError,
)
from ayaka.handles import KVPageHandle, SequenceHandle


@dataclass(frozen=True, slots=True)
class PageTableEntry:
    """One committed logical block: the page plus its valid token count.

    ``valid_tokens`` is the number of KV positions in the page that attention
    may read; non-tail committed pages must be full (equal to the page size).
    """

    page: KVPageHandle
    valid_tokens: int

    def __post_init__(self) -> None:
        if self.valid_tokens <= 0:
            raise ValueError("a committed page-table entry must contain tokens")

@dataclass(frozen=True, slots=True)
class GroupPageTableEntry:
    """One committed logical block inside one cache group's page table.

    ``logical_block`` is the block index *within the sequence* (shared across
    groups); ``valid_tokens`` is the number of valid KV positions in the page.
    """

    logical_block: int
    page: KVPageHandle
    valid_tokens: int

    def __post_init__(self) -> None:
        if self.logical_block < 0:
            raise ValueError("logical_block must be non-negative")
        if self.valid_tokens <= 0:
            raise ValueError("a group page-table entry must contain valid tokens")

@dataclass(slots=True)
class SequencePageTable:
    """Ordered committed page table of one sequence.

    ``entries[i]`` covers logical token positions
    ``[i * page_size, (i + 1) * page_size)``; the final entry may be partial.
    """

    entries: list[PageTableEntry] = field(default_factory=list)

    def snapshot(self) -> tuple[PageTableEntry, ...]:
        """Return an immutable copy for cross-thread inspection."""
        return tuple(self.entries)

    def validate(self, *, committed_tokens: int, page_size: int) -> None:
        """Verify the table is consistent with the committed token count.

        Checks: no duplicate pages, per-entry ``valid_tokens`` within the page
        boundary, non-tail entries full, and the sum of valid tokens matching
        ``committed_tokens`` (the authoritative-length invariant).
        """
        if committed_tokens < 0:
            raise ValueError("committed_tokens must be non-negative")
        if not self.entries:
            if committed_tokens != 0:
                raise InvalidStateTransitionError(
                    "an empty page table cannot represent committed tokens"
                )
            return

        seen: set[KVPageHandle] = set()
        total = 0
        for position, entry in enumerate(self.entries):
            if entry.page in seen:
                raise InvalidStateTransitionError("duplicate page in sequence table")
            seen.add(entry.page)
            if entry.valid_tokens > page_size:
                raise InvalidStateTransitionError("page valid_tokens exceeds page size")
            if position < len(self.entries) - 1 and entry.valid_tokens != page_size:
                raise InvalidStateTransitionError("non-tail committed pages must be full")
            total += entry.valid_tokens

        if total != committed_tokens:
            raise InvalidStateTransitionError(
                f"page table represents {total} tokens, expected {committed_tokens}"
            )

@dataclass(slots=True)
class SequenceMemoryState:
    """Mutable per-sequence state owned by the arena.

    ``committed_tokens`` is the authoritative length: the number of logical
    positions whose KV metadata is safe for attention. The version bumps on
    every committed change so tentative reservations can detect concurrent
    mutation. The pending transaction/lease indexes, release-request flag, and
    blocked epoch implement the single-transaction, deferred-release, and
    fail-after-launch blocking rules.
    """

    handle: SequenceHandle
    request_id: str
    committed_tokens: int = 0
    page_table: SequencePageTable = field(default_factory=SequencePageTable)
    version: int = 0

    pending_transaction_index: int | None = None
    """Open transaction index; at most one transaction per sequence at a time."""
    active_lease_index: int | None = None
    """Prepared lease index; at most one execution lease per sequence at a time."""
    release_requested: bool = False
    """True when release was requested while busy; honored after the step."""
    release_safe_epoch: int = 0
    """Earliest epoch at which the deferred release may drop request refs."""
    blocked_until_epoch: int = 0
    """Sequence is blocked from new work until this epoch (failed steps)."""

@dataclass(frozen=True, slots=True)
class SequenceMemorySnapshot:
    """Immutable scheduler-facing view of one sequence's KV state."""

    handle: SequenceHandle
    request_id: str
    committed_tokens: int
    page_table: tuple[PageTableEntry, ...]
    version: int
    busy: bool
    """True while the sequence participates in a transaction or lease."""
    release_requested: bool
    blocked_until_epoch: int

@dataclass(slots=True)
class GroupedSequenceMemoryState:
    """Mutable per-sequence state; ``group_tables`` holds one table per group.

    Mirrors ``SequenceMemoryState`` from the homogeneous manager: committed
    tokens are authoritative, the version guards tentative plans, and the
    pending transaction/lease indexes, release flag, and blocked epoch encode
    the single-owner, deferred-release, and fail-after-launch rules.
    """

    handle: SequenceHandle
    request_id: str
    group_tables: dict[str, list[GroupPageTableEntry]]
    committed_tokens: int = 0
    version: int = 0
    pending_transaction_index: int | None = None
    active_lease_index: int | None = None
    release_requested: bool = False
    release_safe_epoch: int = 0
    blocked_until_epoch: int = 0

@dataclass(frozen=True, slots=True)
class GroupedSequenceSnapshot:
    """Immutable scheduler-facing view of one grouped sequence."""

    handle: SequenceHandle
    request_id: str
    committed_tokens: int
    version: int
    group_page_tables: tuple[tuple[str, tuple[GroupPageTableEntry, ...]], ...]
    busy: bool
    release_requested: bool
    blocked_until_epoch: int

    def page_table_for(self, group_name: str) -> tuple[GroupPageTableEntry, ...]:
        """Return one group's committed page table.

        Raises:
            KeyError: if ``group_name`` is not a configured group.
        """
        for name, entries in self.group_page_tables:
            if name == group_name:
                return entries
        raise KeyError(group_name)

class SequenceArena:
    """Fixed-capacity arena that rejects stale sequence handles.

    Slots are recycled with monotonically increasing generations; releasing a
    slot requires its pages and committed tokens to be empty already, which the
    memory manager guarantees before it calls :meth:`release`.
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("sequence arena capacity must be positive")
        self._capacity = capacity
        self._generations = [0] * capacity
        self._states: list[SequenceMemoryState | None] = [None] * capacity
        # Reversed so pop() yields slot 0 first: deterministic slot reuse.
        self._free_indices = list(reversed(range(capacity)))
        self._lock = RLock()

    @property
    def capacity(self) -> int:
        """Maximum number of concurrent live sequences."""
        return self._capacity

    def allocate(self, request_id: str) -> SequenceHandle:
        """Allocate a fresh sequence slot and return its new handle.

        Raises:
            ValueError: if ``request_id`` is empty.
            SequenceCapacityError: if every slot is live.
        """
        if not request_id:
            raise ValueError("request_id must not be empty")
        with self._lock:
            if not self._free_indices:
                raise SequenceCapacityError("sequence arena is full")
            index = self._free_indices.pop()
            # Bumping the generation invalidates every stale handle for the slot.
            generation = self._generations[index] + 1
            self._generations[index] = generation
            handle = SequenceHandle(index=index, generation=generation)
            self._states[index] = SequenceMemoryState(
                handle=handle,
                request_id=request_id,
            )
            return handle

    def get(self, handle: SequenceHandle) -> SequenceMemorySnapshot:
        """Return an immutable snapshot of a live sequence.

        Raises:
            InvalidHandleError: if the handle is stale or out of range.
        """
        with self._lock:
            state = self._get_mutable(handle)
            return SequenceMemorySnapshot(
                handle=state.handle,
                request_id=state.request_id,
                committed_tokens=state.committed_tokens,
                page_table=state.page_table.snapshot(),
                version=state.version,
                busy=(
                    state.pending_transaction_index is not None
                    or state.active_lease_index is not None
                ),
                release_requested=state.release_requested,
                blocked_until_epoch=state.blocked_until_epoch,
            )

    @contextmanager
    def mutate(self, handle: SequenceHandle) -> Iterator[SequenceMemoryState]:
        """Yield one sequence's mutable state while holding the arena lock.

        This is the only supported way to mutate arena-owned state from
        outside the arena. :meth:`_get_mutable` deliberately takes no lock --
        it is the resolution step shared by the methods that already hold one
        -- so calling it directly leaves the arena's own synchronisation
        bypassed and makes this class thread-safe in name only.

        The lock is re-entrant, so a caller already inside :meth:`mutate` (or
        any other arena method) may nest further calls safely.

        Raises:
            InvalidHandleError: if the handle is stale or out of range.
        """
        with self._lock:
            yield self._get_mutable(handle)

    def release(self, handle: SequenceHandle) -> None:
        """Recycle the slot; the caller must have emptied pages and tokens first.

        Raises:
            InvalidHandleError: if the handle is stale or out of range.
            InvalidStateTransitionError: if the slot still holds KV state or is
                busy with a transaction/lease.
        """
        with self._lock:
            state = self._get_mutable(handle)
            if state.page_table.entries or state.committed_tokens:
                raise InvalidStateTransitionError(
                    "sequence pages must be released before its arena slot"
                )
            if state.pending_transaction_index is not None or state.active_lease_index is not None:
                raise InvalidStateTransitionError("cannot release a busy sequence slot")
            self._states[handle.index] = None
            self._free_indices.append(handle.index)

    def live_handles(self) -> tuple[SequenceHandle, ...]:
        """Return the handles of all live sequences (unsorted, slot order)."""
        with self._lock:
            return tuple(state.handle for state in self._states if state is not None)

    def _get_mutable(self, handle: SequenceHandle) -> SequenceMemoryState:
        """Resolve a handle to its mutable state, rejecting stale generations.

        Private and unsynchronised on purpose: every caller must already hold
        ``_lock``. Code outside the arena wants :meth:`mutate`, which takes the
        lock around exactly this resolution.
        """
        if handle.index >= self._capacity:
            raise InvalidHandleError(f"sequence index {handle.index} is out of range")
        state = self._states[handle.index]
        if state is None or state.handle.generation != handle.generation:
            raise InvalidHandleError(f"stale sequence handle: {handle}")
        return state

    def assert_invariants(self, *, page_size: int) -> None:
        """Debug-only check of free-list and per-slot invariants."""
        with self._lock:
            free_set = set(self._free_indices)
            if len(free_set) != len(self._free_indices):
                raise InvalidStateTransitionError("duplicate free sequence slot")
            for index, state in enumerate(self._states):
                if state is None:
                    if index not in free_set:
                        raise InvalidStateTransitionError(
                            "empty sequence slot missing from free list"
                        )
                    continue
                if index in free_set:
                    raise InvalidStateTransitionError("live sequence slot appears in free list")
                if state.handle.generation != self._generations[index]:
                    raise InvalidStateTransitionError(
                        "sequence generation does not match arena generation"
                    )
                state.page_table.validate(
                    committed_tokens=state.committed_tokens,
                    page_size=page_size,
                )
