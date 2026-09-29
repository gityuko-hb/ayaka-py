"""Greedy acceptance over a static candidate tree (host oracle)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ayaka.speculative.tree import SpecTree

__all__ = ["TreeAcceptance", "accept_tree_greedy"]


@dataclass(frozen=True, slots=True)
class TreeAcceptance:
    """Accepted node path, emitted tokens and the block rows that stay in KV."""

    path: tuple[int, ...]
    tokens: tuple[int, ...]
    gather_rows: tuple[int, ...]


def accept_tree_greedy(
    tree: SpecTree, draft_tokens: Sequence[int], target_tokens: Sequence[int]
) -> TreeAcceptance:
    """Walk down from the base row while a child's draft equals the target.

    Args:
        tree: Candidate topology.
        draft_tokens: Draft id per node.
        target_tokens: Target choice per verification row ``[base, nodes...]``:
            ``target_tokens[0]`` follows the base row, ``target_tokens[i + 1]``
            follows node ``i``.

    Siblings are tried in index order, so duplicate sibling drafts resolve to
    the lowest index deterministically. The emitted tokens are the accepted
    drafts plus the target token after the deepest accepted row; a linear tree
    reproduces :func:`ayaka.speculative.acceptance.greedy.accept_greedy`.
    """
    if len(draft_tokens) != tree.num_nodes:
        raise ValueError("one draft token is required per tree node")
    if len(target_tokens) != tree.num_nodes + 1:
        raise ValueError("one target token is required per verification row")
    path: list[int] = []
    current = -1
    while True:
        expected = target_tokens[current + 1]
        match = next(
            (child for child in tree.children(current) if draft_tokens[child] == expected),
            None,
        )
        if match is None:
            break
        path.append(match)
        current = match
    tokens = (*(draft_tokens[node] for node in path), target_tokens[current + 1])
    return TreeAcceptance(tuple(path), tuple(tokens), tree.gather_indices(path))
