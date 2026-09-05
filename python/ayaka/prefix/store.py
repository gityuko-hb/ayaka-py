"""Thread-safe orchestration of prefix identity, radix, and page ownership."""

from __future__ import annotations

from collections.abc import Sequence
from threading import RLock

from ayaka.exceptions import InvariantViolationError
from ayaka.handles import KVPageHandle, PrefixHandle
from ayaka.memory.allocator import PageAllocator
from ayaka.prefix.identity import PrefixCacheContext, build_identities, full_token_blocks
from ayaka.prefix.interface import CachedBlockInfo, PrefixCacheSnapshot, PrefixMatch
from ayaka.prefix.ownership import PrefixOwnershipNode, PrefixOwnershipTable
from ayaka.prefix.radix import PageRadixIndex, RadixPath


class RadixPagePrefixCache:
    """Compressed page-radix cache with allocator-backed physical ownership.

    ``identity`` computes compatibility-safe chained hashes; ``radix`` owns
    only logical lookup paths; ``ownership`` owns canonical page handles and
    allocator refs. This store serializes changes spanning those components.
    """

    def __init__(self, *, allocator: PageAllocator, page_size: int) -> None:
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        if allocator.page_size != page_size:
            raise ValueError("prefix cache and allocator page sizes must match")
        self._page_size = page_size
        self._ownership = PrefixOwnershipTable(
            allocator=allocator,
            page_size=page_size,
        )
        self._radix = PageRadixIndex()
        self._lock = RLock()

    @property
    def page_size(self) -> int:
        return self._page_size

    def match(
        self,
        token_ids: Sequence[int],
        *,
        context: PrefixCacheContext,
    ) -> PrefixMatch:
        """Return the longest reusable complete-page prefix."""

        blocks = full_token_blocks(token_ids, page_size=self._page_size)
        identities = build_identities(
            blocks,
            page_size=self._page_size,
            context=context,
        )
        if not blocks:
            return PrefixMatch.empty(context=context, page_size=self._page_size)

        with self._lock:
            handles = self._radix.match(
                context,
                tuple(identity.digest for identity in identities),
            )
            matched = [self._ownership.get(handle) for handle in handles]
            parent: PrefixHandle | None = None
            for node, block in zip(matched, blocks, strict=False):
                self._ownership.validate_node(
                    node,
                    expected_parent=parent,
                    expected_tokens=block,
                )
                parent = node.handle

            if not matched:
                return PrefixMatch.empty(context=context, page_size=self._page_size)
            self._ownership.touch(matched)
            pages = tuple(node.page for node in matched)
            return PrefixMatch(
                context=context,
                page_size=self._page_size,
                matched_tokens=len(pages) * self._page_size,
                pages=pages,
                terminal_handle=matched[-1].handle,
            )

    def insert(
        self,
        token_ids: Sequence[int],
        pages: Sequence[KVPageHandle],
        *,
        context: PrefixCacheContext,
    ) -> PrefixHandle | None:
        """Publish complete pages under one canonical compatible identity."""

        blocks = full_token_blocks(token_ids, page_size=self._page_size)
        identities = build_identities(
            blocks,
            page_size=self._page_size,
            context=context,
        )
        source_pages = tuple(pages)
        if len(source_pages) != len(blocks):
            raise ValueError("insert requires exactly one page per complete token block")
        if not blocks:
            return None

        with self._lock:
            self._ownership.validate_source_pages(source_pages)
            parent: PrefixHandle | None = None
            resolved: list[PrefixOwnershipNode] = []
            planned: list[PrefixOwnershipNode] = []

            for identity, block, source_page in zip(
                identities,
                blocks,
                source_pages,
                strict=True,
            ):
                node = self._ownership.find(identity)
                if node is None:
                    node = self._ownership.plan(
                        identity=identity,
                        block_token_ids=block,
                        page=source_page,
                        parent=parent,
                    )
                    planned.append(node)
                else:
                    self._ownership.validate_node(
                        node,
                        expected_parent=parent,
                        expected_tokens=block,
                    )
                resolved.append(node)
                parent = node.handle

            self._ownership.publish_new(planned)
            try:
                self._radix.insert(
                    context,
                    tuple(identity.digest for identity in identities),
                    tuple(node.handle for node in resolved),
                )
            except Exception:
                # New identities form a suffix. Before terminal retain that
                # suffix is removable as one unretained leaf chain.
                if planned:
                    self._ownership.prune_unretained(
                        planned[-1],
                        safe_epoch=self._ownership.current_epoch,
                    )
                self._rebuild_radix()
                raise

            terminal = resolved[-1]
            terminal.retain_terminal()
            self._ownership.touch(resolved)
            return terminal.handle

    def acquire_match(self, match: PrefixMatch) -> tuple[KVPageHandle, ...]:
        """Revalidate a match, then transactionally acquire request refs."""

        with self._lock:
            nodes = self._ownership.resolve_match(match)
            if not nodes:
                return ()
            pages = self._ownership.acquire_request_refs(nodes)
            self._ownership.touch(nodes)
            return pages

    def truncate_match(self, match: PrefixMatch, matched_tokens: int) -> PrefixMatch:
        """Return a shorter page-aligned capability after group intersection."""

        if matched_tokens < 0 or matched_tokens % self._page_size:
            raise ValueError("matched_tokens must be a non-negative page multiple")
        if matched_tokens > match.matched_tokens:
            raise ValueError("cannot extend a prefix match while truncating it")
        with self._lock:
            nodes = self._ownership.resolve_match(match)
            count = matched_tokens // self._page_size
            if count == 0:
                return PrefixMatch.empty(context=match.context, page_size=self._page_size)
            selected = nodes[:count]
            return PrefixMatch(
                context=match.context,
                page_size=self._page_size,
                matched_tokens=matched_tokens,
                pages=tuple(node.page for node in selected),
                terminal_handle=selected[-1].handle,
            )

    def release(self, prefix: PrefixHandle, *, safe_epoch: int) -> int:
        """Drop one terminal retainer and prune newly unshared ownership."""

        with self._lock:
            self._ownership.validate_safe_epoch(safe_epoch)
            node = self._ownership.get(prefix)
            if not node.release_terminal():
                return 0
            released = self._ownership.prune_unretained(node, safe_epoch=safe_epoch)
            # Rebuild even if the former terminal still has children: its
            # logical terminal marker changed without physical page removal.
            self._rebuild_radix()
            return released

    def evict(self, target_pages: int, *, safe_epoch: int) -> int:
        """Evict deterministic LRU terminal leaves until the target is met."""

        if target_pages < 0:
            raise ValueError("target_pages must be non-negative")
        if target_pages == 0:
            return 0

        with self._lock:
            self._ownership.validate_safe_epoch(safe_epoch)
            released = 0
            while released < target_pages and len(self._ownership):
                candidates = self._ownership.evictable_nodes()
                if not candidates:
                    raise InvariantViolationError(
                        "prefix cache contains no evictable terminal leaf"
                    )
                candidate = candidates[0]
                candidate.evict_terminal()
                pruned = self._ownership.prune_unretained(
                    candidate,
                    safe_epoch=safe_epoch,
                )
                if pruned <= 0:
                    raise InvariantViolationError(
                        "evicting a terminal leaf did not release a cache block"
                    )
                released += pruned
                self._rebuild_radix()
            return released

    def clear(self, *, safe_epoch: int) -> int:
        """Evict every cached ownership node."""

        with self._lock:
            self._ownership.validate_safe_epoch(safe_epoch)
            target = len(self._ownership)
        return self.evict(target, safe_epoch=safe_epoch) if target else 0

    def snapshot(self) -> PrefixCacheSnapshot:
        with self._lock:
            return self._ownership.snapshot()

    def cached_block_index(self) -> tuple[CachedBlockInfo, ...]:
        with self._lock:
            return self._ownership.cached_block_index()

    def evictable_leaves(self) -> tuple[CachedBlockInfo, ...]:
        with self._lock:
            return tuple(
                self._ownership.describe(node)
                for node in self._ownership.evictable_nodes()
            )

    def assert_invariants(self) -> None:
        """Verify ownership/allocator state and cross-check the radix index."""

        with self._lock:
            self._ownership.assert_invariants()
            self._radix.assert_invariants(
                identity_for=self._ownership.identity_for,
                all_handles=self._ownership.all_handles(),
                terminal_handles=self._ownership.terminal_handles(),
            )

    def _rebuild_radix(self) -> None:
        """Project retained ownership chains into the logical radix index."""

        self._radix.rebuild(
            RadixPath(
                context=chain[-1].identity.context,
                digests=tuple(node.identity.digest for node in chain),
                handles=tuple(node.handle for node in chain),
            )
            for chain in self._ownership.terminal_chains()
        )


# Compatibility name used before the radix index became canonical.
HashBlockPrefixCache = RadixPagePrefixCache
