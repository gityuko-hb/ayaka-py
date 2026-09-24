"""Canonical resident prefix cache for grouped KV layouts.

A grouped resume is an *all-group* boundary: every cache group must retain
enough pages to reconstruct the prefix, and that boundary stays pinned until
its entry is evicted. Token identity reuses the shared chained-digest
machinery; physical ownership stays with the grouped manager's allocator pins,
so this store never duplicates page ownership or keeps a second page ledger.

Lookup never returns a boundary that a successful all-group pin did not
publish. A missing sliding-window history, an evicted entry or a changed
generation is a clean miss for the caller (recompute), never a partial hit.
Entries are deduplicated by ``(context, token_ids)``: publishing the same
token sequence twice reuses the first canonical pin.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import count
from threading import RLock
from typing import TYPE_CHECKING

from ayaka.exceptions import InvariantViolationError, PrefixCapabilityStaleError
from ayaka.handles import PrefixHandle
from ayaka.prefix.identity import (
    PrefixCacheContext,
    build_identities,
    full_token_blocks,
)
from ayaka.prefix.interface import (
    GroupedValidResume,
    PrefixCacheSnapshot,
)

if TYPE_CHECKING:
    from ayaka.kvcache.grouped_manager import GroupedPrefixSnapshot, KVCacheGroupManager

_CACHE_IDS = count(1)

__all__ = [
    "GroupedPrefixCache",
    "GroupedPrefixEntry",
]


@dataclass(slots=True)
class GroupedPrefixEntry:
    """One canonical all-group resume boundary and its pinned pages."""

    entry_id: int
    context: PrefixCacheContext
    token_ids: tuple[int, ...]
    snapshot: GroupedPrefixSnapshot | None
    page_count: int
    last_access: int = 0

    @property
    def logical_position(self) -> int:
        return len(self.token_ids)


@dataclass(slots=True)
class _EntryRadixNode:
    """One compressed edge spanning one or more index-block digests."""

    digests: tuple[bytes, ...]
    parent: _EntryRadixNode | None = None
    children: dict[bytes, _EntryRadixNode] = field(default_factory=dict)
    entry_id: int | None = None
    """Certified boundary at the end of this span (full index blocks)."""
    partials: dict[tuple[int, ...], int] = field(default_factory=dict)
    """Certified boundaries inside the next block, keyed by their token tuple."""

    def __post_init__(self) -> None:
        if len(self.digests) == 0 and self.parent is not None:
            raise ValueError("only a radix root may have an empty span")
        if self.entry_id is not None and not self.digests:
            raise ValueError("a certified boundary requires a full index block path")


class _EntryRadix:
    """Compressed token-digest trie over published grouped boundaries.

    Nodes carry no ownership; an ``entry_id`` marks a boundary whose pinned
    snapshot lives in the owning :class:`GroupedPrefixCache`.
    """

    def __init__(self) -> None:
        self._roots: dict[PrefixCacheContext, _EntryRadixNode] = {}

    def _root(self, context: PrefixCacheContext) -> _EntryRadixNode:
        root = self._roots.get(context)
        if root is None:
            root = _EntryRadixNode(digests=())
            self._roots[context] = root
        return root

    def insert(
        self,
        context: PrefixCacheContext,
        digests: tuple[bytes, ...],
        partial_tokens: tuple[int, ...] | None,
        entry_id: int,
    ) -> None:
        """Insert one certified boundary, splitting a compressed edge if needed."""
        if not digests and partial_tokens is None:
            raise ValueError("a grouped boundary must cover at least one token")
        current = self._root(context)
        offset = 0
        while offset < len(digests):
            child = current.children.get(digests[offset])
            if child is None:
                child = _EntryRadixNode(
                    digests=digests[offset:],
                    parent=current,
                )
                current.children[child.digests[0]] = child
                current = child
                break
            common = 0
            limit = min(len(child.digests), len(digests) - offset)
            while common < limit and child.digests[common] == digests[offset + common]:
                common += 1
            if common == 0:
                raise InvariantViolationError("radix child is indexed under the wrong digest")
            if common == len(child.digests):
                offset += common
                current = child
                continue
            # Split the compressed edge at the divergence point.
            old_suffix = _EntryRadixNode(
                digests=child.digests[common:],
                parent=child,
                children=child.children,
                entry_id=child.entry_id,
                partials=dict(child.partials),
            )
            for grandchild in old_suffix.children.values():
                grandchild.parent = old_suffix
            child.digests = child.digests[:common]
            child.children = {old_suffix.digests[0]: old_suffix}
            child.entry_id = None
            child.partials = {}
            offset += common
            if offset == len(digests):
                current = child
                break
            new_suffix = _EntryRadixNode(
                digests=digests[offset:],
                parent=child,
            )
            child.children[new_suffix.digests[0]] = new_suffix
            current = new_suffix
            break

        if partial_tokens is None:
            if current.entry_id not in (None, entry_id):
                raise InvariantViolationError("grouped boundary slot already certified")
            current.entry_id = entry_id
        else:
            existing = current.partials.get(partial_tokens)
            if existing not in (None, entry_id):
                raise InvariantViolationError("grouped partial boundary slot already certified")
            current.partials[partial_tokens] = entry_id

    def match(
        self,
        context: PrefixCacheContext,
        digests: tuple[bytes, ...],
        tokens: tuple[int, ...],
        index_block: int,
    ) -> tuple[int, int] | None:
        """Return ``(entry_id, position)`` of the longest certified prefix.

        Only boundaries a publisher actually certified are returned: a
        candidate that diverges inside a compressed span or inside a block
        falls back to the deepest certified boundary on the shared path.
        """
        root = self._roots.get(context)
        if root is None:
            return None
        checked: list[tuple[_EntryRadixNode, int]] = [(root, 0)]
        current = root
        offset = 0
        while offset < len(digests):
            child = current.children.get(digests[offset])
            if child is None:
                break
            common = 0
            limit = min(len(child.digests), len(digests) - offset)
            while common < limit and child.digests[common] == digests[offset + common]:
                common += 1
            if common != len(child.digests) or common != limit:
                break
            offset += common
            current = child
            checked.append((child, offset))

        best_entry: int | None = None
        best_position = 0
        for node, depth in checked:
            if node.entry_id is not None:
                position = depth * index_block
                if position > best_position:
                    best_entry, best_position = node.entry_id, position
            base = depth * index_block
            remaining = len(tokens) - base
            if remaining <= 0:
                continue
            for partial_tokens, entry_id in node.partials.items():
                length = len(partial_tokens)
                if length <= remaining and tokens[base : base + length] == partial_tokens:
                    position = base + length
                    if position > best_position:
                        best_entry, best_position = entry_id, position
        if best_entry is None:
            return None
        return best_entry, best_position

    def remove(
        self,
        context: PrefixCacheContext,
        digests: tuple[bytes, ...],
        partial_tokens: tuple[int, ...] | None,
        entry_id: int,
    ) -> None:
        """Remove one certified boundary and prune now-unshared nodes."""
        root = self._roots.get(context)
        if root is None:
            raise InvariantViolationError("grouped prefix context disappeared from the index")
        node = root
        offset = 0
        while offset < len(digests):
            child = node.children.get(digests[offset])
            if child is None:
                raise InvariantViolationError("grouped prefix entry path is missing")
            common = 0
            limit = min(len(child.digests), len(digests) - offset)
            while common < limit and child.digests[common] == digests[offset + common]:
                common += 1
            if common != len(child.digests) or common != limit:
                raise InvariantViolationError("grouped prefix entry path diverged")
            offset += common
            node = child
        if partial_tokens is None:
            if node.entry_id != entry_id:
                raise InvariantViolationError("grouped prefix entry slot changed before removal")
            node.entry_id = None
        else:
            if node.partials.get(partial_tokens) != entry_id:
                raise InvariantViolationError("grouped partial slot changed before removal")
            del node.partials[partial_tokens]
        while (
            node.parent is not None
            and node.entry_id is None
            and not node.partials
            and not node.children
        ):
            parent = node.parent
            parent.children.pop(node.digests[0], None)
            node = parent

    def clear(self) -> None:
        self._roots.clear()

    def referenced_entries(self) -> set[int]:
        """Every entry id still reachable from a certified boundary."""
        found: set[int] = set()

        def visit(node: _EntryRadixNode) -> None:
            if node.entry_id is not None:
                found.add(node.entry_id)
            found.update(node.partials.values())
            for child in node.children.values():
                visit(child)

        for root in self._roots.values():
            visit(root)
        return found


class GroupedPrefixCache:
    """Boundary registry over the grouped manager's pinned resume snapshots.

    The cache owns no pages itself: every entry holds exactly one
    ``GroupedPrefixSnapshot`` whose pages the manager pinned. Eviction is the
    only path that unpins, and it removes the entry before releasing pins so a
    stale capability can never resolve to an unpinned snapshot. Like the
    manager it serves, this cache is engine-thread serialized; its lock is not
    a cross-thread ordering boundary against the manager lock.
    """

    def __init__(self, manager: KVCacheGroupManager) -> None:
        self._manager = manager
        self.cache_id = next(_CACHE_IDS)
        self._index_block = manager.page_size
        self._index = _EntryRadix()
        self._entries: dict[int, GroupedPrefixEntry] = {}
        self._next_entry_id = 0
        self._clock = 0
        self._max_entries: int | None = None
        self._evicted_entries_total = 0
        self._evicted_pages_total = 0
        self._lock = RLock()

    @property
    def page_size(self) -> int:
        """Index block granularity: the smallest group page size."""
        return self._index_block

    @property
    def max_entries(self) -> int | None:
        return self._max_entries

    @max_entries.setter
    def max_entries(self, value: int | None) -> None:
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < 1
        ):
            raise ValueError("max_entries must be a positive integer or None")
        with self._lock:
            self._max_entries = value
            while value is not None and len(self._entries) > value:
                if not self.evict_entry(None, safe_epoch=self._manager.current_epoch):
                    break

    @property
    def evicted_entries_total(self) -> int:
        return self._evicted_entries_total

    @property
    def evicted_pages_total(self) -> int:
        return self._evicted_pages_total

    def match_resume(
        self, token_ids: Sequence[int], *, context: PrefixCacheContext
    ) -> GroupedValidResume | None:
        """Return the longest published boundary that prefixes ``token_ids``."""
        tokens = tuple(token_ids)
        if not tokens:
            return None
        with self._lock:
            if self._manager.tiering_enabled:
                # A host entry keeps the canonical token identity, but its
                # pages cannot back a borrowed resume until every group has
                # promoted and published. Queue a bounded promotion while
                # selecting the deepest currently device-readable boundary.
                candidates = sorted(
                    (
                        entry
                        for entry in self._entries.values()
                        if entry.context == context
                        and len(entry.token_ids) <= len(tokens)
                        and tokens[: len(entry.token_ids)] == entry.token_ids
                    ),
                    key=lambda entry: (-entry.logical_position, entry.entry_id),
                )
                for entry in candidates:
                    if entry.snapshot is None:
                        self._manager.promote_prefix_entry(entry.entry_id)
                        continue
                    if not self._manager.tier_entry_device_ready(entry.entry_id):
                        continue
                    self._touch(entry)
                    return GroupedValidResume(
                        self.cache_id, entry.entry_id, entry.context, entry.token_ids
                    )
                return None
            digests, _partial_tokens = self._path(tokens, context)
            found = self._index.match(context, digests, tokens, self._index_block)
            if found is None:
                return None
            entry_id, _position = found
            entry = self._entries.get(entry_id)
            if entry is None:
                raise InvariantViolationError("grouped prefix index refers to a missing entry")
            self._touch(entry)
            return GroupedValidResume(
                cache_id=self.cache_id,
                entry_id=entry.entry_id,
                context=entry.context,
                token_ids=entry.token_ids,
            )

    def find_exact(
        self, token_ids: Sequence[int], *, context: PrefixCacheContext
    ) -> GroupedValidResume | None:
        """Return the entry for exactly these tokens, or None."""
        tokens = tuple(token_ids)
        with self._lock:
            entry = next(
                (
                    value
                    for value in self._entries.values()
                    if value.context == context and value.token_ids == tokens
                ),
                None,
            )
            if entry is None:
                return None
            return GroupedValidResume(self.cache_id, entry.entry_id, context, tokens)

    def register(
        self,
        token_ids: Sequence[int],
        *,
        context: PrefixCacheContext,
        snapshot: GroupedPrefixSnapshot,
    ) -> GroupedValidResume:
        """Publish one all-group pinned boundary.

        The caller must have checked :meth:`find_exact` first: publishing the
        same ``(context, token_ids)`` twice would pin a second snapshot with no
        cache entry able to release it.
        """
        tokens = tuple(token_ids)
        if not tokens:
            raise ValueError("a grouped prefix entry must cover at least one token")
        if snapshot.manager_id != id(self._manager):
            raise ValueError("grouped prefix snapshot belongs to another manager")
        if snapshot.logical_position != len(tokens):
            raise ValueError("grouped prefix snapshot boundary disagrees with its tokens")
        digests, partial_tokens = self._path(tokens, context)
        page_count = sum(len(entries) for _, entries in snapshot.groups)
        with self._lock:
            existing = self._index.match(context, digests, tokens, self._index_block)
            if existing is not None and existing[1] == len(tokens):
                raise InvariantViolationError("grouped prefix entry already published")
            entry_id = self._next_entry_id
            self._next_entry_id += 1
            entry = GroupedPrefixEntry(
                entry_id=entry_id,
                context=context,
                token_ids=tokens,
                snapshot=snapshot,
                page_count=page_count,
            )
            self._index.insert(context, digests, partial_tokens, entry_id)
            self._entries[entry_id] = entry
            self._touch(entry)
            if self._max_entries is not None:
                while len(self._entries) > self._max_entries:
                    if not self.evict_entry(None, safe_epoch=self._manager.current_epoch):
                        break
            return GroupedValidResume(
                cache_id=self.cache_id,
                entry_id=entry_id,
                context=context,
                token_ids=tokens,
            )

    def resolve_resume(self, match: GroupedValidResume) -> GroupedPrefixSnapshot:
        """Revalidate a borrowed capability, or raise a stale-capability error."""
        with self._lock:
            if match.cache_id != self.cache_id:
                raise PrefixCapabilityStaleError(
                    "grouped resume belongs to another cache incarnation"
                )
            entry = self._entries.get(match.entry_id)
            if entry is None:
                raise PrefixCapabilityStaleError("grouped resume entry was evicted")
            if (
                entry.context != match.context
                or entry.token_ids != match.token_ids
                or entry.logical_position != match.logical_position
            ):
                raise PrefixCapabilityStaleError("grouped resume identity changed")
            if entry.snapshot is None or not self._manager.tier_entry_device_ready(entry.entry_id):
                raise PrefixCapabilityStaleError("grouped resume is not device-readable")
            self._touch(entry)
            return entry.snapshot

    def evict_entry(self, entry_id: int | None = None, *, safe_epoch: int) -> bool:
        """Evict one entry (LRU by default); returns whether anything was dropped."""
        with self._lock:
            entry = self._select(entry_id)
            if entry is None:
                return False
            if not self._manager.forget_tier_entry(entry.entry_id):
                return False
            digests, partial_tokens = self._path(entry.token_ids, entry.context)
            self._index.remove(entry.context, digests, partial_tokens, entry.entry_id)
            del self._entries[entry.entry_id]
            self._evicted_entries_total += 1
            snapshot = entry.snapshot
            self._evicted_pages_total += entry.page_count if snapshot is not None else 0
        # Pins release outside the cache lock; the manager lock is reentrant on
        # the engine thread and the entry is already unreachable.
        if snapshot is not None:
            self._manager.unpin_prefix(snapshot)
        return True

    def evict(self, target_pages: int, *, safe_epoch: int) -> int:
        """Evict LRU entries until ``target_pages`` worth of pins are released."""
        if target_pages <= 0:
            return 0
        released = 0
        with self._lock:
            while released < target_pages and self._entries:
                entry = self._select(None)
                if entry is None:
                    break
                pages = entry.page_count if entry.snapshot is not None else 0
                if not self.evict_entry(entry.entry_id, safe_epoch=safe_epoch):
                    break
                released += pages
        return released

    def clear(self, *, safe_epoch: int) -> int:
        """Drop every entry and release its pins; returns released page count."""
        with self._lock:
            released = 0
            while self._entries:
                entry = self._select(None)
                if entry is None:
                    break
                pages = entry.page_count if entry.snapshot is not None else 0
                if not self.evict_entry(entry.entry_id, safe_epoch=safe_epoch):
                    break
                released += pages
            if not self._entries:
                self._index.clear()
            return released

    def snapshot(self) -> PrefixCacheSnapshot:
        """Capacity accounting over the currently pinned boundaries."""
        with self._lock:
            entries = sorted(self._entries.values(), key=lambda item: item.entry_id)
            return PrefixCacheSnapshot(
                cached_blocks=sum(
                    entry.page_count for entry in entries if entry.snapshot is not None
                ),
                terminal_entries=len(entries),
                cached_tokens=sum(entry.logical_position for entry in entries),
                handles=tuple(PrefixHandle(entry.entry_id, 1) for entry in entries),
            )

    def assert_invariants(self) -> None:
        """Verify index/entry agreement, uniqueness, and page accounting."""
        with self._lock:
            referenced = self._index.referenced_entries()
            if referenced != set(self._entries):
                raise InvariantViolationError("grouped prefix index and entry registry disagree")
            seen_tokens: set[tuple[PrefixCacheContext, tuple[int, ...]]] = set()
            for entry in self._entries.values():
                if entry.logical_position <= 0:
                    raise InvariantViolationError("grouped prefix entry has no tokens")
                if entry.page_count <= 0:
                    raise InvariantViolationError("grouped prefix entry pins no pages")
                key = (entry.context, entry.token_ids)
                if key in seen_tokens:
                    raise InvariantViolationError("duplicate grouped prefix entry identity")
                seen_tokens.add(key)
                if entry.snapshot is not None:
                    if entry.snapshot.logical_position != entry.logical_position:
                        raise InvariantViolationError("grouped prefix snapshot boundary drifted")
                    if entry.snapshot.manager_id != id(self._manager):
                        raise InvariantViolationError("grouped prefix snapshot owner drifted")
                elif self._manager.tier_entry_device_ready(entry.entry_id):
                    raise InvariantViolationError("host grouped prefix entry is device-ready")
                digests, partial_tokens = self._path(entry.token_ids, entry.context)
                found = self._index.match(
                    entry.context, digests, entry.token_ids, self._index_block
                )
                if found is None or found[1] != entry.logical_position:
                    raise InvariantViolationError(
                        "grouped prefix entry is not its own deepest boundary"
                    )

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _path(
        self, tokens: tuple[int, ...], context: PrefixCacheContext
    ) -> tuple[tuple[bytes, ...], tuple[int, ...] | None]:
        """Split tokens into complete index blocks plus a partial tail.

        The returned digests are the chained identities of the complete
        blocks; a non-``None`` partial tail is the sub-block remainder that
        only a certified partial boundary may carry.
        """
        if not tokens:
            return (), None
        blocks = full_token_blocks(tokens, page_size=self._index_block)
        identities = build_identities(blocks, page_size=self._index_block, context=context)
        remainder = len(tokens) - len(blocks) * self._index_block
        if remainder:
            partial = tokens[len(blocks) * self._index_block :]
            return tuple(identity.digest for identity in identities), partial
        return tuple(identity.digest for identity in identities), None

    def _touch(self, entry: GroupedPrefixEntry) -> None:
        self._clock += 1
        entry.last_access = self._clock

    def _select(self, entry_id: int | None) -> GroupedPrefixEntry | None:
        if entry_id is not None:
            return self._entries.get(entry_id)
        if not self._entries:
            return None
        candidates = (
            entry
            for entry in self._entries.values()
            if self._manager.tier_entry_evictable(entry.entry_id)
        )
        return min(candidates, key=lambda item: (item.last_access, item.entry_id), default=None)
