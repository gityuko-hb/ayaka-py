"""Thread-safe orchestration of prefix identity, radix, and page ownership."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import count
from threading import RLock

from ayaka.exceptions import (
    InvalidHandleError,
    InvariantViolationError,
    PrefixCapabilityStaleError,
)
from ayaka.handles import KVPageHandle, PrefixHandle
from ayaka.memory.allocator import PageAllocator
from ayaka.memory.sequence import PageTableEntry
from ayaka.prefix.identity import (
    PrefixBlockIdentity,
    PrefixCacheContext,
    build_identities,
    full_token_blocks,
)
from ayaka.prefix.interface import CachedBlockInfo, PrefixCacheSnapshot, PrefixMatch, ValidResume
from ayaka.prefix.ownership import PrefixOwnershipNode, PrefixOwnershipTable
from ayaka.prefix.radix import PageRadixIndex, RadixPath

_CACHE_IDS = count(1)


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
        self.cache_id = next(_CACHE_IDS)
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
        return self._insert_blocks(blocks, tuple(pages), context=context, retain=True)

    def insert_resume(
        self,
        token_ids: Sequence[int],
        pages: Sequence[KVPageHandle],
        *,
        context: PrefixCacheContext,
    ) -> ValidResume | None:
        """Publish committed full pages and a partial tail under canonical ownership.

        Repeated publication is idempotent. Partial blocks are terminal leaves;
        complete-page callers continue to see only complete-page matches.
        """
        tokens = tuple(token_ids)
        full_token_blocks(tokens, page_size=self._page_size)  # Validate every token.
        blocks = tuple(
            tokens[i : i + self._page_size] for i in range(0, len(tokens), self._page_size)
        )
        with self._lock:
            handle = self._insert_blocks(blocks, tuple(pages), context=context, retain=False)
            if handle is None:
                return None
            return self._resume(self._ownership.chain(handle), len(tokens))

    def _insert_blocks(
        self,
        blocks: tuple[tuple[int, ...], ...],
        source_pages: tuple[KVPageHandle, ...],
        *,
        context: PrefixCacheContext,
        retain: bool,
    ) -> PrefixHandle | None:
        identities = build_identities(blocks, page_size=self._page_size, context=context)
        if len(source_pages) != len(blocks):
            raise ValueError("insert requires exactly one page per token block")
        if not blocks:
            return None

        with self._lock:
            self._ownership.validate_source_pages(source_pages, tuple(map(len, blocks)))
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
            if retain or not terminal.terminal:
                terminal.retain_terminal()
            self._ownership.touch(resolved)
            return terminal.handle

    def _resume(self, nodes: Sequence[PrefixOwnershipNode], n: int) -> ValidResume:
        return ValidResume(
            self.cache_id,
            nodes[-1].handle.index,
            nodes[-1].identity.context,
            tuple(token for node in nodes for token in node.block_token_ids)[:n],
            tuple(
                PageTableEntry(node.page, min(self._page_size, n - i * self._page_size))
                for i, node in enumerate(nodes)
            ),
        )

    def match_resume(
        self, token_ids: Sequence[int], *, context: PrefixCacheContext
    ) -> ValidResume | None:
        """Match full radix blocks, then the longest valid portion of the next block."""
        tokens = tuple(token_ids)
        with self._lock:
            full = self.match(tokens, context=context)
            nodes = (
                () if full.terminal_handle is None else self._ownership.chain(full.terminal_handle)
            )
            n = full.matched_tokens
            best = None
            common = 0
            for candidate in self._ownership.children(full.terminal_handle, context):
                matched = 0
                for a, b in zip(candidate.block_token_ids, tokens[n:], strict=False):
                    if a != b:
                        break
                    matched += 1
                if matched > common:
                    common, best = matched, candidate
            if best is not None:
                self._ownership.validate_node(
                    best, expected_parent=full.terminal_handle, expected_tokens=best.block_token_ids
                )
                nodes += (best,)
                n += common
            if not nodes:
                return None
            self._ownership.touch(nodes)
            return self._resume(nodes, n)

    def _resolve_resume(self, match: ValidResume) -> tuple[PrefixOwnershipNode, ...]:
        """Revalidate a borrowed capability or raise ``PrefixCapabilityStaleError``.

        Cache incarnation, entry identity, context, token identity, chain
        geometry and page generations must all agree with the store's current
        state. Structural corruption found along the chain still raises
        ``InvariantViolationError`` so it is never reported as a miss.
        """
        if match.cache_id != self.cache_id:
            raise PrefixCapabilityStaleError("resume belongs to another cache incarnation")
        nodes = self._ownership.chain(PrefixHandle(match.entry_id, 1))
        n = match.logical_position
        if (
            not (len(nodes) - 1) * self._page_size
            < n
            <= sum(len(node.block_token_ids) for node in nodes)
        ):
            raise PrefixCapabilityStaleError("resume length does not match its terminal block")
        if match != self._resume(nodes, n):
            raise PrefixCapabilityStaleError("resume identity or page generations changed")
        parent = None
        for node in nodes:
            self._ownership.validate_node(
                node, expected_parent=parent, expected_tokens=node.block_token_ids
            )
            parent = node.handle
        return nodes

    def acquire_resume(self, match: ValidResume) -> tuple[PageTableEntry, ...]:
        """Revalidate and acquire request refs under the same eviction lock.

        Stale capabilities raise ``PrefixCapabilityStaleError`` (a clean miss
        for ``acquire``); allocator failures during ref acquisition propagate
        as backend faults.
        """
        with self._lock:
            nodes = self._resolve_resume(match)
            self._ownership.acquire_request_refs(nodes)
            self._ownership.touch(nodes)
            return match.pages

    def chain_identities(self, match: ValidResume) -> tuple[PrefixBlockIdentity, ...] | None:
        """Validated identity chain of a borrowed resume, or None when stale.

        Read-only tier bookkeeping aid: it revalidates the capability exactly
        like ``acquire_resume`` but acquires no reference, so the tier can
        classify readiness before any ownership changes. A stale capability
        returns None instead of raising so callers fold it into their miss
        handling.
        """
        with self._lock:
            try:
                nodes = self._resolve_resume(match)
            except PrefixCapabilityStaleError:
                return None
            return tuple(node.identity for node in nodes)

    def acquire_resume_prefix(
        self, match: ValidResume, *, pages: int
    ) -> tuple[PageTableEntry, ...]:
        """Acquire request refs on only the first ``pages`` blocks of a resume.

        Tiered attachment trims a match to its device-readable boundary; the
        trimmed prefix is anchored at a whole block, so a later re-lookup or
        restore can continue the chain from there. Revalidation and the
        identity checks are identical to ``acquire_resume``: only the ref
        acquisition and the returned entries are truncated.
        """
        if pages < 1:
            raise ValueError("acquire_resume_prefix requires at least one page")
        with self._lock:
            nodes = self._resolve_resume(match)
            if pages > len(nodes):
                raise ValueError("resume has fewer blocks than the requested prefix")
            prefix_nodes = nodes[:pages]
            self._ownership.acquire_request_refs(prefix_nodes)
            self._ownership.touch(prefix_nodes)
            return match.pages[:pages]

    def pin_resume(self, match: ValidResume) -> tuple[PageTableEntry, ...]:
        """Keep spill sources alive independently of cache eviction."""
        with self._lock:
            nodes = self._resolve_resume(match)
            self._ownership.pin(nodes)
            return match.pages

    def evict_entry(self, entry_id: int | None = None, *, safe_epoch: int) -> bool:
        """Evict one canonical terminal, leaving consumer refs and pins intact."""
        with self._lock:
            self._ownership.validate_safe_epoch(safe_epoch)
            if entry_id is None:
                candidates = self._ownership.evictable_nodes()
                if not candidates:
                    return False
                node = candidates[0]
            else:
                try:
                    node = self._ownership.get(PrefixHandle(entry_id, 1))
                except InvalidHandleError:
                    return False
                if not node.terminal:
                    return False
            node.evict_terminal()
            self._ownership.prune_unretained(node, safe_epoch=safe_epoch)
            self._rebuild_radix()
            return True

    def evict_page(self, page: KVPageHandle, *, safe_epoch: int) -> int:
        """Drop every cached path depending on a page; consumer refs stay untouched."""
        with self._lock:
            self._ownership.validate_safe_epoch(safe_epoch)
            terminals = tuple(
                chain[-1].handle
                for chain in self._ownership.terminal_chains()
                if any(node.page == page for node in chain)
            )
            before = len(self._ownership)
            for handle in terminals:
                self.evict_entry(handle.index, safe_epoch=safe_epoch)
            return before - len(self._ownership)

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
                self._ownership.describe(node) for node in self._ownership.evictable_nodes()
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
