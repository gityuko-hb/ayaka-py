"""Static speculative candidate trees, independent of any drafter or model.

A tree hangs below the verification block's base row. Nodes are numbered in
topological order (a parent precedes its children); ``parents[i] == -1`` means
node ``i`` is a child of the base row. Everything a verifier or attention
backend needs — depths, position offsets, ancestor masks, packed per-node
masks and gather indices for an accepted path — is derived here, so tree
topology never leaks into EAGLE- or MTP-specific code.

Serving does not consume trees yet: linear verification is the certified path
and every attention backend still refuses ``spec_tree_mask``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ayaka.utils.validation import require_int

__all__ = ["MAX_TREE_NODES", "SpecTree"]

#: Bound on tree nodes (draft rows) under one base row.
MAX_TREE_NODES = 256


@dataclass(frozen=True, slots=True)
class SpecTree:
    """Topology of one candidate tree.

    Attributes:
        parents: Parent node index per node; ``-1`` for children of the base
            row. Must satisfy ``parents[i] < i``.
    """

    parents: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.parents) is not tuple or not self.parents:
            raise ValueError("a speculative tree needs at least one node")
        if len(self.parents) > MAX_TREE_NODES:
            raise ValueError(f"speculative trees are limited to {MAX_TREE_NODES} nodes")
        for index, parent in enumerate(self.parents):
            require_int(parent, "tree parent", minimum=-1)
            if parent >= index:
                raise ValueError("tree nodes must be numbered in topological order")

    @classmethod
    def linear(cls, length: int) -> SpecTree:
        """A chain of ``length`` drafts: the linear special case."""
        require_int(length, "linear tree length", minimum=1)
        return cls(tuple(range(-1, length - 1)))

    @classmethod
    def from_choices(cls, choices: Sequence[Sequence[int]]) -> SpecTree:
        """Build from branch paths, e.g. ``[(0,), (1,), (0, 0)]``.

        Each choice names a node by the sibling ranks along its path from the
        base row; every prefix of a choice must also be a choice. Nodes are
        ordered by depth, then by the order they are listed.
        """
        keys = [tuple(choice) for choice in choices]
        if not keys or len(set(keys)) != len(keys):
            raise ValueError("tree choices must be non-empty and unique")
        for key in keys:
            if not key or any(type(rank) is not int or rank < 0 for rank in key):
                raise ValueError("each tree choice is a non-empty tuple of sibling ranks")
            if len(key) > 1 and key[:-1] not in keys:
                raise ValueError(f"tree choice {key} is missing its parent {key[:-1]}")
        ordered = sorted(keys, key=lambda key: (len(key), keys.index(key)))
        index = {key: position for position, key in enumerate(ordered)}
        return cls(tuple(-1 if len(key) == 1 else index[key[:-1]] for key in ordered))

    @property
    def num_nodes(self) -> int:
        return len(self.parents)

    @property
    def is_linear(self) -> bool:
        return self.parents == tuple(range(-1, self.num_nodes - 1))

    @property
    def depths(self) -> tuple[int, ...]:
        """Depth per node; children of the base row have depth 1."""
        depths: list[int] = []
        for parent in self.parents:
            depths.append(1 if parent < 0 else depths[parent] + 1)
        return tuple(depths)

    @property
    def position_offsets(self) -> tuple[int, ...]:
        """Logical position of each node relative to the base row's position."""
        return self.depths

    @property
    def max_depth(self) -> int:
        return max(self.depths)

    def children(self, node: int) -> tuple[int, ...]:
        """Children of ``node`` in index order; ``-1`` addresses the base row."""
        require_int(node, "tree node", minimum=-1)
        return tuple(i for i, parent in enumerate(self.parents) if parent == node)

    def ancestors(self, node: int) -> tuple[int, ...]:
        """Node path from the first-level ancestor down to ``node`` itself."""
        require_int(node, "tree node")
        if node >= self.num_nodes:
            raise IndexError(f"tree node {node} out of range")
        path = [node]
        while self.parents[path[-1]] >= 0:
            path.append(self.parents[path[-1]])
        return tuple(reversed(path))

    @property
    def leaves(self) -> tuple[int, ...]:
        parents = set(self.parents)
        return tuple(i for i in range(self.num_nodes) if i not in parents)

    @property
    def paths(self) -> tuple[tuple[int, ...], ...]:
        """Every base-to-leaf node path, in leaf index order."""
        return tuple(self.ancestors(leaf) for leaf in self.leaves)

    def draft_mask(self) -> tuple[tuple[bool, ...], ...]:
        """``[N, N]``: node ``r`` may attend node ``c`` iff ``c`` is ``r`` or its ancestor."""
        rows = []
        for node in range(self.num_nodes):
            visible = set(self.ancestors(node))
            rows.append(tuple(column in visible for column in range(self.num_nodes)))
        return tuple(rows)

    def verify_mask(self) -> tuple[tuple[bool, ...], ...]:
        """``[N+1, N+1]`` over the verification block ``[base, node_0 .. node_{N-1}]``.

        Every row sees the base row (and, outside this block, the committed
        prefix); node rows additionally see their ancestors and themselves.
        """
        draft = self.draft_mask()
        rows = [(True,) + (False,) * self.num_nodes]
        rows.extend((True, *row) for row in draft)
        return tuple(rows)

    def packed_mask(self) -> tuple[int, ...]:
        """Per-node bitmask; bit ``j`` set iff the node may attend node ``j``.

        This is the ``CommonAttentionMetadata.spec_tree_mask`` encoding for
        trees of at most 64 nodes.
        """
        if self.num_nodes > 64:
            raise ValueError("packed tree masks fit at most 64 nodes")
        return tuple(
            sum(1 << column for column in self.ancestors(node)) for node in range(self.num_nodes)
        )

    def gather_indices(self, path: Sequence[int]) -> tuple[int, ...]:
        """Verification-block rows of an accepted path, base row first.

        ``path`` must be a prefix of some root-to-node path (possibly empty).
        """
        nodes = tuple(path)
        if nodes and self.ancestors(nodes[-1]) != nodes:
            raise ValueError("accepted path is not a connected tree path")
        return (0, *(node + 1 for node in nodes))
