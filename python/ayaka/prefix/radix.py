"""Compressed page-radix index for prefix lookup.

The index is intentionally ignorant of page allocation and refcounts.  It
maps compatibility-context/digest paths to opaque ownership handles; physical
validation stays in :mod:`ownership`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from ayaka.exceptions import InvariantViolationError
from ayaka.handles import PrefixHandle
from ayaka.prefix.identity import PrefixBlockIdentity, PrefixCacheContext


@dataclass(slots=True)
class _RadixSpanNode:
    """One compressed edge spanning one or more complete-page digests."""

    digests: tuple[bytes, ...]
    handles: tuple[PrefixHandle, ...]
    parent: _RadixSpanNode | None = None
    children: dict[bytes, _RadixSpanNode] = field(default_factory=dict)
    terminal: bool = False

    def __post_init__(self) -> None:
        if len(self.digests) != len(self.handles):
            raise ValueError("radix span digests and handles must align")
        if not self.digests and self.parent is not None:
            raise ValueError("only a radix root may have an empty span")

@dataclass(frozen=True, slots=True)
class RadixPath:
    """One retained ownership chain used to rebuild the logical index."""

    context: PrefixCacheContext
    digests: tuple[bytes, ...]
    handles: tuple[PrefixHandle, ...]


class PageRadixIndex:
    """Compressed radix tree over full-page chained identities."""

    def __init__(self) -> None:
        self._roots: dict[PrefixCacheContext, _RadixSpanNode] = {}

    def match(
        self,
        context: PrefixCacheContext,
        digests: tuple[bytes, ...],
    ) -> tuple[PrefixHandle, ...]:
        """Return the longest page-aligned path, including mid-span ends."""

        root = self._roots.get(context)
        if root is None or not digests:
            return ()
        current = root
        offset = 0
        matched: list[PrefixHandle] = []
        while offset < len(digests):
            child = current.children.get(digests[offset])
            if child is None:
                break
            common = 0
            limit = min(len(child.digests), len(digests) - offset)
            while common < limit and child.digests[common] == digests[offset + common]:
                common += 1
            matched.extend(child.handles[:common])
            offset += common
            if common != len(child.digests):
                break
            current = child
        return tuple(matched)

    def insert(
        self,
        context: PrefixCacheContext,
        digests: tuple[bytes, ...],
        handles: tuple[PrefixHandle, ...],
    ) -> None:
        """Insert a page path, splitting a compressed edge on divergence."""

        if len(digests) != len(handles) or not digests:
            raise ValueError("radix insert needs aligned non-empty digests and handles")
        current = self._root(context)
        offset = 0
        while offset < len(digests):
            child = current.children.get(digests[offset])
            if child is None:
                child = _RadixSpanNode(
                    digests=digests[offset:],
                    handles=handles[offset:],
                    parent=current,
                    terminal=True,
                )
                current.children[child.digests[0]] = child
                return

            common = 0
            limit = min(len(child.digests), len(digests) - offset)
            while common < limit and child.digests[common] == digests[offset + common]:
                if child.handles[common] != handles[offset + common]:
                    raise InvariantViolationError(
                        "one radix identity resolves to two physical prefix handles"
                    )
                common += 1
            if common == 0:
                raise InvariantViolationError("radix child is indexed under the wrong digest")

            if common == len(child.digests):
                offset += common
                if offset == len(digests):
                    child.terminal = True
                    return
                current = child
                continue

            old_suffix = _RadixSpanNode(
                digests=child.digests[common:],
                handles=child.handles[common:],
                parent=child,
                children=child.children,
                terminal=child.terminal,
            )
            for grandchild in old_suffix.children.values():
                grandchild.parent = old_suffix
            child.digests = child.digests[:common]
            child.handles = child.handles[:common]
            child.children = {old_suffix.digests[0]: old_suffix}
            child.terminal = False
            offset += common
            if offset == len(digests):
                child.terminal = True
                return
            new_suffix = _RadixSpanNode(
                digests=digests[offset:],
                handles=handles[offset:],
                parent=child,
                terminal=True,
            )
            child.children[new_suffix.digests[0]] = new_suffix
            return

    def rebuild(self, paths: Iterable[RadixPath]) -> None:
        """Recreate compressed spans from retained ownership terminals."""

        self._roots.clear()
        for path in paths:
            self.insert(path.context, path.digests, path.handles)

    def clear(self) -> None:
        self._roots.clear()

    def assert_invariants(
        self,
        *,
        identity_for: Callable[[PrefixHandle], PrefixBlockIdentity],
        all_handles: set[PrefixHandle],
        terminal_handles: set[PrefixHandle],
    ) -> None:
        """Verify structural links and agreement with the ownership registry."""

        represented: set[PrefixHandle] = set()
        represented_terminals: set[PrefixHandle] = set()

        def visit(node: _RadixSpanNode) -> None:
            if node.parent is not None and not node.digests:
                raise InvariantViolationError("non-root radix node has an empty span")
            if len(node.digests) != len(node.handles):
                raise InvariantViolationError("radix span metadata is misaligned")
            for digest, handle in zip(node.digests, node.handles, strict=True):
                if identity_for(handle).digest != digest:
                    raise InvariantViolationError("radix digest and prefix owner disagree")
                represented.add(handle)
            if node.terminal:
                if not node.handles or node.handles[-1] not in terminal_handles:
                    raise InvariantViolationError("radix terminal lacks an ownership terminal")
                represented_terminals.add(node.handles[-1])
            for first_digest, child in node.children.items():
                if child.parent is not node or not child.digests:
                    raise InvariantViolationError("radix parent/child link is corrupt")
                if child.digests[0] != first_digest:
                    raise InvariantViolationError("radix child key disagrees with its span")
                visit(child)

        for root in self._roots.values():
            if root.parent is not None or root.digests or root.handles:
                raise InvariantViolationError("radix root carries a non-root span")
            visit(root)
        if represented != all_handles:
            raise InvariantViolationError("radix index does not represent every cached page")
        if represented_terminals != terminal_handles:
            raise InvariantViolationError("radix and ownership terminal sets disagree")

    def _root(self, context: PrefixCacheContext) -> _RadixSpanNode:
        root = self._roots.get(context)
        if root is None:
            root = _RadixSpanNode(digests=(), handles=())
            self._roots[context] = root
        return root
