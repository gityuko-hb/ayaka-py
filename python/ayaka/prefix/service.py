"""Resident prefix policy over the manager's single canonical ownership store."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from ayaka.exceptions import (
    InvalidHandleError,
    PrefixCapabilityStaleError,
)
from ayaka.handles import SequenceHandle
from ayaka.kvcache.grouped_manager import KVCacheGroupManager
from ayaka.kvcache.grouped_prefix import GroupedPrefixCache
from ayaka.memory.manager import RuntimeMemoryManager
from ayaka.memory.sequence import PageTableEntry
from ayaka.prefix.global_index import PrefixIndexPublisher
from ayaka.prefix.identity import PrefixCacheContext
from ayaka.prefix.interface import (
    CachedBlockInfo,
    GroupedValidResume,
    ValidResume,
)
from ayaka.utils.validation import require_int

if TYPE_CHECKING:
    from ayaka.kvcache.manager import LogicalKVManager
    from ayaka.prefix.transfer import PrefixTransfer


def common_resume_boundary(group_boundaries: tuple[tuple[int, ...], ...], limit: int) -> int:
    """Find an actual common checkpoint, never the minimum of longest matches.

    Each group lists positions it can independently restore, including its
    retention/checkpoint restrictions. Absence of a common positive boundary
    means recompute from zero. This helper does not certify recurrent support.
    """
    require_int(limit, "limit")
    if not group_boundaries:
        return 0
    common = set(group_boundaries[0])
    for values in group_boundaries:
        for value in values:
            require_int(value, "resume position")
        common.intersection_update(values)
    return max((x for x in common if x <= limit), default=0)


class PrefixService:
    """Policy and transfer lifecycle; durable page ownership stays in the backend.

    One service belongs to one LogicalKVManager. Both the homogeneous resident
    manager and the grouped manager serve canonical reuse: the homogeneous
    store keeps per-page ownership, the grouped store keeps all-group pinned
    boundaries. Partial-tail reuse is supported by both. A lookup is borrowed
    and may become stale; acquisition revalidates it and returns zero on
    eviction. Callers supply the complete execution identity and keep the
    final prompt query for logits.
    """

    def __init__(
        self,
        kv: LogicalKVManager,
        *,
        index_publisher: PrefixIndexPublisher | None = None,
    ) -> None:
        backend = kv.backend
        if isinstance(backend, RuntimeMemoryManager):
            if backend.tiering_enabled:
                raise ValueError("canonical prefix resume requires resident KV")
        elif not isinstance(backend, KVCacheGroupManager):
            raise ValueError("canonical prefix resume requires a resident KV backend")
        self.kv = kv
        self.backend = backend
        self._grouped = isinstance(backend, KVCacheGroupManager)
        self.store = backend.prefix_cache
        self.cache_id = self.store.cache_id
        self._max_entries: int | None = None
        self._transfers: set[PrefixTransfer] = set()
        self._index_publisher = index_publisher
        self.closed = False

    @property
    def max_entries(self) -> int | None:
        """Optional shared canonical-terminal ceiling; physical capacity is always bounded."""
        return self._max_entries

    @max_entries.setter
    def max_entries(self, value: int | None) -> None:
        """Fix the optional terminal ceiling; set-once for the shared service.

        The first owner to set a ceiling wins. A later owner must agree or the
        assignment fails: silently changing eviction policy would break the
        first consumer's capacity accounting.
        """
        if value is not None:
            require_int(value, "max_entries", minimum=1)
        if self._max_entries is not None and value != self._max_entries:
            raise ValueError(
                f"prefix service max_entries is fixed at {self._max_entries}; "
                "a second owner must not silently change eviction policy"
            )
        self._max_entries = value

    def require_open(self) -> None:
        if self.closed or self.kv.closed:
            raise ValueError("prefix service is closed")
        self.kv._validate_storage_bindings()

    def lookup(
        self, token_ids: Sequence[int], *, context: PrefixCacheContext
    ) -> ValidResume | GroupedValidResume | None:
        self.require_open()
        return self.store.match_resume(token_ids, context=context)

    def ready(self, match: ValidResume | GroupedValidResume) -> bool:
        """Avoid attaching another partial-tail consumer with no COW headroom.

        This is an engine-thread admission hint, not a reservation. A stale
        capability proceeds to acquire's generation-safe miss handling.
        """
        self.require_open()
        if isinstance(self.backend, RuntimeMemoryManager):
            return self.backend.resume_ready(match) if isinstance(match, ValidResume) else True
        return self.backend.resume_ready(match) if isinstance(match, GroupedValidResume) else True

    def acquire(
        self,
        sequence: SequenceHandle,
        match: ValidResume | GroupedValidResume,
        *,
        token_ids: Sequence[int],
        context: PrefixCacheContext,
    ) -> int:
        """Revalidate and attach a borrowed capability, or return zero on a miss.

        Capability revalidation failures (cache incarnation, entry identity,
        context, token identity, chain geometry) are clean misses. Sequence
        state errors from the backend — a busy sequence, a stale handle, an
        invariant violation — propagate so they are never masked as misses.
        """
        self.require_open()
        n = match.logical_position
        if n <= 0 or match.context != context or match.token_ids != tuple(token_ids[:n]):
            return 0
        try:
            if isinstance(self.backend, RuntimeMemoryManager):
                if not isinstance(match, ValidResume):
                    return 0
                return self.backend.attach_resume(sequence, match)
            if not isinstance(match, GroupedValidResume):
                return 0
            return self.backend.attach_resume(sequence, match)
        except (InvalidHandleError, PrefixCapabilityStaleError):
            return 0

    def publish(
        self, sequence: SequenceHandle, token_ids: Sequence[int], *, context: PrefixCacheContext
    ) -> ValidResume | GroupedValidResume | None:
        self.require_open()
        published = tuple(token_ids)
        result = self.backend.cache_resume(sequence, published, context=context)
        if self._grouped and result is not None and len(published) > 1:
            # Also certify the runtime's lookup boundary (the prompt without
            # its final query) when every group can still restore it. Sliding
            # history that already left the window makes this a clean no-op,
            # so an identical repeat is a miss there by policy; a longer
            # candidate still hits the primary boundary.
            self.backend.cache_resume(sequence, published[:-1], context=context)
        if result is not None and self._index_publisher is not None and not self._grouped:
            # The global index keys digests by one page size; a grouped entry
            # is pinned at the smallest group's granularity, so advertising
            # those digests would route peers to boundaries they cannot match.
            self._index_publisher.publish(
                published, context=context, page_size=self.backend.page_size
            )
        if self.max_entries is not None:
            while self.store.snapshot().terminal_entries > self.max_entries:
                self.evict()
        return result

    def pin(self, match: ValidResume) -> tuple[PageTableEntry, ...]:
        if isinstance(self.store, GroupedPrefixCache):
            raise RuntimeError("grouped prefix entries have no per-page transfer pins")
        self.require_open()
        return self.store.pin_resume(match)

    def evict(self, entry_id: int | None = None) -> bool:
        self.require_open()
        if isinstance(self.store, GroupedPrefixCache):
            return self.store.evict_entry(entry_id, safe_epoch=self.backend.current_epoch)
        publisher = self._index_publisher
        before = self.store.evictable_leaves() if publisher is not None else ()
        evicted = self.store.evict_entry(entry_id, safe_epoch=self.backend.current_epoch)
        if evicted and publisher is not None:
            after = {info.handle.index for info in self.store.evictable_leaves()}
            leaf = self._evicted_leaf(before, entry_id, remaining=after)
            if leaf is not None:
                publisher.withdraw_digest(leaf.identity.digest)
        return evicted

    @staticmethod
    def _evicted_leaf(
        before: tuple[CachedBlockInfo, ...],
        entry_id: int | None,
        *,
        remaining: set[int],
    ) -> CachedBlockInfo | None:
        """Identify the leaf that disappeared from the evictable set."""
        if entry_id is not None:
            return next((info for info in before if info.handle.index == entry_id), None)
        # Eviction without an explicit id pops the first evictable leaf;
        # pruning cannot create a new terminal, so the set difference is it.
        return next((info for info in before if info.handle.index not in remaining), None)

    def close(self) -> bool:
        """Drop cache ownership only after all transfers have retired."""
        if self.closed:
            return True
        if any(not transfer.retired for transfer in self._transfers):
            return False
        self.backend.clear_prefix_cache()
        self.backend.reclaim_deferred()
        if self._index_publisher is not None:
            self._index_publisher.flush_once()
        self.closed = True
        return True
