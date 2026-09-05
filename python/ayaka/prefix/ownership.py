"""Physical page ownership and lifetime rules for prefix caching."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from ayaka.exceptions import (
    InvalidHandleError,
    InvalidStateTransitionError,
    InvariantViolationError,
)
from ayaka.handles import KVPageHandle, PrefixHandle
from ayaka.memory.allocator import PageAllocator
from ayaka.memory.state import PageAllocationState
from ayaka.prefix.identity import PrefixBlockIdentity, context_seed, identity_for_block
from ayaka.prefix.interface import CachedBlockInfo, PrefixCacheSnapshot, PrefixMatch


@dataclass(slots=True)
class PrefixOwnershipNode:
    """One canonical cached block and its allocator-owned page reference."""

    handle: PrefixHandle
    identity: PrefixBlockIdentity
    block_token_ids: tuple[int, ...]
    page: KVPageHandle
    parent: PrefixHandle | None
    children: set[PrefixHandle] = field(default_factory=set)
    terminal_refs: int = 0
    last_access: int = 0

    @property
    def terminal(self) -> bool:
        return self.terminal_refs > 0

    def retain_terminal(self) -> None:
        self.terminal_refs += 1

    def release_terminal(self) -> bool:
        if self.terminal_refs <= 0:
            raise InvalidStateTransitionError(
                "prefix handle is not a retained terminal entry"
            )
        self.terminal_refs -= 1
        return self.terminal_refs == 0

    def evict_terminal(self) -> None:
        if self.terminal_refs <= 0:
            raise InvariantViolationError("cannot evict an unretained prefix terminal")
        self.terminal_refs = 0


class PrefixOwnershipTable:
    """Canonical identity registry plus allocator reference ownership.

    The table is not independently synchronized.  Its store owns the lock so
    that ownership publication and radix publication share one critical
    section.
    """

    def __init__(self, *, allocator: PageAllocator, page_size: int) -> None:
        self._allocator = allocator
        self._page_size = page_size
        self._nodes: dict[int, PrefixOwnershipNode] = {}
        self._identity_to_index: dict[PrefixBlockIdentity, int] = {}
        self._next_handle_index = 0
        self._clock = 0

    def __len__(self) -> int:
        return len(self._nodes)

    @property
    def current_epoch(self) -> int:
        """Allocator completion epoch used for rollback-only releases."""

        return self._allocator.current_epoch

    def values(self) -> tuple[PrefixOwnershipNode, ...]:
        return tuple(self._nodes.values())

    def get(self, handle: PrefixHandle) -> PrefixOwnershipNode:
        """Resolve a handle and reject stale generations."""

        node = self._nodes.get(handle.index)
        if node is None or node.handle != handle:
            raise InvalidHandleError(f"stale prefix handle: {handle}")
        return node

    def find(self, identity: PrefixBlockIdentity) -> PrefixOwnershipNode | None:
        index = self._identity_to_index.get(identity)
        if index is None:
            return None
        node = self._nodes.get(index)
        if node is None:
            raise InvariantViolationError("prefix identity index refers to a missing node")
        return node

    def plan(
        self,
        *,
        identity: PrefixBlockIdentity,
        block_token_ids: tuple[int, ...],
        page: KVPageHandle,
        parent: PrefixHandle | None,
    ) -> PrefixOwnershipNode:
        """Reserve a never-reused logical handle without publishing the node."""

        handle = PrefixHandle(index=self._next_handle_index, generation=1)
        self._next_handle_index += 1
        return PrefixOwnershipNode(
            handle=handle,
            identity=identity,
            block_token_ids=block_token_ids,
            page=page,
            parent=parent,
        )

    def publish_new(self, nodes: Sequence[PrefixOwnershipNode]) -> None:
        """Acquire cache refs, then atomically expose planned ownership nodes."""

        planned = tuple(nodes)
        seen_handles: set[PrefixHandle] = set()
        seen_identities: set[PrefixBlockIdentity] = set()
        known_handles = {node.handle for node in self._nodes.values()}
        for node in planned:
            if node.handle in seen_handles or node.handle.index in self._nodes:
                raise InvariantViolationError("duplicate prefix ownership handle")
            if node.identity in seen_identities or node.identity in self._identity_to_index:
                raise InvariantViolationError("duplicate canonical prefix identity")
            if node.parent is not None and node.parent not in known_handles | seen_handles:
                raise InvariantViolationError("planned prefix parent is missing")
            seen_handles.add(node.handle)
            seen_identities.add(node.identity)

        acquired: list[KVPageHandle] = []
        try:
            for node in planned:
                self._allocator.acquire_cache_ref(node.page)
                acquired.append(node.page)
                self.validate_cached_page(node.page)
        except Exception:
            for page in reversed(acquired):
                self._allocator.release_cache_ref(
                    page,
                    safe_epoch=self._allocator.current_epoch,
                )
            raise

        published: list[PrefixOwnershipNode] = []
        try:
            for node in planned:
                self._nodes[node.handle.index] = node
                self._identity_to_index[node.identity] = node.handle.index
                if node.parent is not None:
                    self.get(node.parent).children.add(node.handle)
                published.append(node)
        except Exception:
            for node in reversed(published):
                if node.parent is not None:
                    parent = self._nodes.get(node.parent.index)
                    if parent is not None:
                        parent.children.discard(node.handle)
                self._nodes.pop(node.handle.index, None)
                self._identity_to_index.pop(node.identity, None)
            for page in reversed(acquired):
                self._allocator.release_cache_ref(
                    page,
                    safe_epoch=self._allocator.current_epoch,
                )
            raise

    def validate_source_pages(self, pages: tuple[KVPageHandle, ...]) -> None:
        """Reject duplicate, partial, non-live, or request-unowned pages."""

        if len(set(pages)) != len(pages):
            raise ValueError("a prefix chain cannot contain duplicate pages")
        for page in pages:
            meta = self._allocator.get_meta(page)
            if (
                meta.allocation_state is not PageAllocationState.LIVE
                or meta.request_refs <= 0
                or meta.valid_tokens != self._page_size
            ):
                raise InvalidStateTransitionError(
                    "only request-owned complete live pages can enter the prefix cache"
                )

    def validate_node(
        self,
        node: PrefixOwnershipNode,
        *,
        expected_parent: PrefixHandle | None,
        expected_tokens: tuple[int, ...],
    ) -> None:
        if node.parent != expected_parent or node.block_token_ids != expected_tokens:
            raise InvariantViolationError("prefix digest collision or corrupted chain metadata")
        self.validate_cached_page(node.page)

    def validate_cached_page(self, page: KVPageHandle) -> None:
        try:
            meta = self._allocator.get_meta(page)
        except InvalidHandleError as exc:
            raise InvariantViolationError(
                "prefix cache refers to a stale physical page generation"
            ) from exc
        if (
            meta.allocation_state is not PageAllocationState.LIVE
            or meta.cache_refs <= 0
            or meta.valid_tokens != self._page_size
        ):
            raise InvariantViolationError(
                "prefix cache refers to a non-cache-owned complete live page"
            )

    def resolve_match(self, match: PrefixMatch) -> list[PrefixOwnershipNode]:
        """Revalidate geometry, context, length, pages, and parent chain."""

        if match.page_size != self._page_size:
            raise InvalidStateTransitionError("prefix match belongs to a different page geometry")
        if match.terminal_handle is None:
            if match.matched_tokens or match.pages:
                raise InvalidStateTransitionError("empty prefix match is inconsistent")
            return []

        reverse_chain: list[PrefixOwnershipNode] = []
        node = self.get(match.terminal_handle)
        while True:
            reverse_chain.append(node)
            if node.parent is None:
                break
            node = self.get(node.parent)
        nodes = list(reversed(reverse_chain))

        if nodes[-1].identity.context != match.context:
            raise InvalidStateTransitionError("prefix match compatibility context changed")
        pages = tuple(item.page for item in nodes)
        if pages != match.pages:
            raise InvalidStateTransitionError("prefix match page chain changed before attachment")
        if len(nodes) * self._page_size != match.matched_tokens:
            raise InvalidStateTransitionError("prefix match length changed before attachment")
        if len(set(pages)) != len(pages):
            raise InvariantViolationError("prefix match contains duplicate pages")

        parent: PrefixHandle | None = None
        for item in nodes:
            self.validate_node(
                item,
                expected_parent=parent,
                expected_tokens=item.block_token_ids,
            )
            parent = item.handle
        return nodes

    def acquire_request_refs(
        self,
        nodes: Sequence[PrefixOwnershipNode],
    ) -> tuple[KVPageHandle, ...]:
        """Attach a request transactionally, rolling back partial acquisition."""

        acquired: list[KVPageHandle] = []
        try:
            for node in nodes:
                self._allocator.acquire_request_ref(node.page)
                acquired.append(node.page)
        except Exception:
            for page in reversed(acquired):
                self._allocator.release_request_ref(
                    page,
                    safe_epoch=self._allocator.current_epoch,
                )
            raise
        return tuple(acquired)

    def touch(self, nodes: Sequence[PrefixOwnershipNode]) -> None:
        self._clock += 1
        for node in nodes:
            node.last_access = self._clock

    def prune_unretained(self, node: PrefixOwnershipNode, *, safe_epoch: int) -> int:
        """Release an unretained leaf and now-unshared ancestors."""

        released = 0
        current: PrefixOwnershipNode | None = node
        while current is not None and not current.terminal and not current.children:
            parent = self.get(current.parent) if current.parent is not None else None
            self._allocator.release_cache_ref(current.page, safe_epoch=safe_epoch)
            self._nodes.pop(current.handle.index)
            self._identity_to_index.pop(current.identity)
            if parent is not None:
                parent.children.remove(current.handle)
            released += 1
            current = parent
        return released

    def evictable_nodes(self) -> list[PrefixOwnershipNode]:
        candidates = [
            node for node in self._nodes.values() if node.terminal and not node.children
        ]
        candidates.sort(key=lambda node: (node.last_access, node.handle.index))
        return candidates

    def snapshot(self) -> PrefixCacheSnapshot:
        handles = tuple(
            node.handle
            for node in sorted(self._nodes.values(), key=lambda node: node.handle.index)
        )
        return PrefixCacheSnapshot(
            cached_blocks=len(handles),
            terminal_entries=sum(node.terminal for node in self._nodes.values()),
            cached_tokens=len(handles) * self._page_size,
            handles=handles,
        )

    def cached_block_index(self) -> tuple[CachedBlockInfo, ...]:
        return tuple(
            self.describe(node)
            for node in sorted(self._nodes.values(), key=lambda node: node.handle.index)
        )

    def describe(self, node: PrefixOwnershipNode) -> CachedBlockInfo:
        parent_identity = None if node.parent is None else self.get(node.parent).identity
        return CachedBlockInfo(
            handle=node.handle,
            identity=node.identity,
            parent_identity=parent_identity,
            block_token_ids=node.block_token_ids,
            page=node.page,
            terminal=node.terminal,
            children=len(node.children),
            last_access=node.last_access,
        )

    def terminal_chains(self) -> tuple[tuple[PrefixOwnershipNode, ...], ...]:
        """Return retained root-to-terminal ownership chains."""

        chains: list[tuple[PrefixOwnershipNode, ...]] = []
        terminals = sorted(
            (node for node in self._nodes.values() if node.terminal),
            key=lambda node: node.handle.index,
        )
        for terminal in terminals:
            chain: list[PrefixOwnershipNode] = []
            node = terminal
            while True:
                chain.append(node)
                if node.parent is None:
                    break
                node = self.get(node.parent)
            chain.reverse()
            chains.append(tuple(chain))
        return tuple(chains)

    def all_handles(self) -> set[PrefixHandle]:
        return {node.handle for node in self._nodes.values()}

    def terminal_handles(self) -> set[PrefixHandle]:
        return {node.handle for node in self._nodes.values() if node.terminal}

    def identity_for(self, handle: PrefixHandle) -> PrefixBlockIdentity:
        return self.get(handle).identity

    def validate_safe_epoch(self, safe_epoch: int) -> None:
        if safe_epoch < self._allocator.current_epoch:
            raise ValueError("safe_epoch cannot precede the completed epoch")

    def assert_invariants(self) -> None:
        """Verify chain identity, bidirectional links, and allocator refs."""

        if len(self._identity_to_index) != len(self._nodes):
            raise InvariantViolationError(
                "prefix identity index and node registry sizes differ"
            )

        expected_cache_refs: Counter[KVPageHandle] = Counter()
        for index, node in self._nodes.items():
            if index != node.handle.index:
                raise InvariantViolationError(
                    "prefix node is stored under the wrong handle index"
                )
            if self._identity_to_index.get(node.identity) != index:
                raise InvariantViolationError("prefix node is missing from the identity index")
            if len(node.block_token_ids) != self._page_size:
                raise InvariantViolationError("prefix node contains a partial token block")
            if node.terminal_refs < 0:
                raise InvariantViolationError("negative prefix terminal refcount")

            if node.parent is None:
                expected_block_index = 0
                parent_digest = context_seed(
                    node.identity.context,
                    page_size=self._page_size,
                )
            else:
                parent = self.get(node.parent)
                if node.handle not in parent.children:
                    raise InvariantViolationError("prefix parent is missing its child link")
                if parent.identity.context != node.identity.context:
                    raise InvariantViolationError("prefix chain changes compatibility context")
                expected_block_index = parent.identity.block_index + 1
                parent_digest = parent.identity.digest

            expected_identity = identity_for_block(
                context=node.identity.context,
                parent_digest=parent_digest,
                block_index=expected_block_index,
                block_token_ids=node.block_token_ids,
                page_size=self._page_size,
            )
            if node.identity != expected_identity:
                raise InvariantViolationError("prefix block identity does not match its chain")

            for child_handle in node.children:
                child = self.get(child_handle)
                if child.parent != node.handle:
                    raise InvariantViolationError(
                        "prefix child does not point back to its parent"
                    )
            if not node.children and not node.terminal:
                raise InvariantViolationError(
                    "an unretained prefix leaf should have been pruned"
                )

            self.validate_cached_page(node.page)
            expected_cache_refs[node.page] += 1

        if self._allocator.snapshot().cache_owned_pages != len(expected_cache_refs):
            raise InvariantViolationError(
                "allocator cache ownership is not represented by prefix nodes"
            )
        for page, expected_refs in expected_cache_refs.items():
            if self._allocator.get_meta(page).cache_refs != expected_refs:
                raise InvariantViolationError(
                    "prefix nodes and allocator cache refcounts disagree"
                )
