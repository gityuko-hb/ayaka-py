"""CSR trie and stateful matcher for allow-list constraints.

:class:`CsrTrie` is an immutable, compressed-sparse-row trie over token-id
sequences. Children of each node are stored sorted so transitions use binary
search. :class:`TrieMatcher` adds a cursor (a node stack) on top, exposing
the small interface :class:`GrammarMaskProducer` needs: accept, rollback,
termination check, and bitmask fill.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from ayaka.sampling.mask.producer import MaskRows

ROOT = 0
"""Index of the trie root node."""


class CsrTrie:
    """Immutable trie in compressed-sparse-row layout.

    Attributes:
        children_off: ``uint32`` offsets of shape ``(n_nodes + 1,)``; children
            of node ``n`` span ``[off[n], off[n+1])``.
        children_token: Sorted ``int32`` token ids, one per edge.
        children_next: ``uint32`` destination node per edge.
        terminal: Boolean array marking nodes that end a stored sequence.
        n_nodes: Number of nodes in the trie.
    """

    __slots__ = ("children_next", "children_off", "children_token", "n_nodes", "terminal")

    def __init__(self, children_off, children_token, children_next, terminal) -> None:
        """Store prebuilt CSR arrays without copying.

        Args:
            children_off: Edge offsets of shape ``(n_nodes + 1,)``.
            children_token: Sorted token id per edge.
            children_next: Destination node per edge.
            terminal: Per-node terminal flags.
        """
        self.children_off = children_off
        self.children_token = children_token
        self.children_next = children_next
        self.terminal = terminal
        self.n_nodes = len(terminal)

    @classmethod
    def build(cls, sequences: Iterable[Sequence[int]]) -> CsrTrie:
        """Build a trie from token-id sequences.

        Shared prefixes share nodes. An empty sequence marks the root as
        terminal. Duplicate sequences are merged.

        Args:
            sequences: Iterable of token-id sequences to insert.

        Returns:
            Immutable :class:`CsrTrie` with children sorted per node for
            binary search.
        """
        kids: list[dict[int, int]] = [{}]
        term: list[bool] = [False]
        for seq in sequences:
            node = ROOT
            for tok in seq:
                nxt = kids[node].get(tok)
                if nxt is None:
                    nxt = len(kids)
                    kids[node][tok] = nxt
                    kids.append({})
                    term.append(False)
                node = nxt
            term[node] = True

        n = len(kids)
        off = np.zeros(n + 1, dtype=np.uint32)
        for i, d in enumerate(kids):
            off[i + 1] = off[i] + len(d)
        toks = np.empty(int(off[-1]), dtype=np.int32)
        nxts = np.empty(int(off[-1]), dtype=np.uint32)
        for i, d in enumerate(kids):
            if not d:
                continue
            items = sorted(d.items())
            s = int(off[i])
            toks[s : s + len(items)] = [k for k, _ in items]
            nxts[s : s + len(items)] = [v for _, v in items]
        return cls(off, toks, nxts, np.asarray(term, dtype=bool))

    def allowed(self, node: int) -> np.ndarray:
        """Return the sorted allowed next-token ids from ``node``.

        Args:
            node: Source node index.

        Returns:
            1-D ``int32`` view of child token ids; empty when the node is
            a leaf.
        """
        s, e = int(self.children_off[node]), int(self.children_off[node + 1])
        sliced: np.ndarray = self.children_token[s:e]
        return sliced

    def step(self, node: int, token: int) -> int:
        """Follow the edge ``(node, token)`` if it exists.

        Args:
            node: Source node index.
            token: Token id to transition on.

        Returns:
            Destination node index, or ``-1`` when the node has no such
            child.
        """
        s, e = int(self.children_off[node]), int(self.children_off[node + 1])
        if s == e:
            return -1
        i = int(np.searchsorted(self.children_token[s:e], token))
        if i < (e - s) and int(self.children_token[s + i]) == token:
            return int(self.children_next[s + i])
        return -1

    def is_terminal(self, node: int) -> bool:
        """Check whether ``node`` ends a stored sequence.

        Args:
            node: Node index to test.

        Returns:
            True when the node was marked terminal at build time.
        """
        return bool(self.terminal[node])


class TrieMatcher:
    """Stateful cursor over a :class:`CsrTrie`.

    Keeps a stack of visited nodes starting at :data:`ROOT`. Each accepted
    token pushes one node; :meth:`rollback` pops back. The matcher itself
    owns no mask memory; :meth:`fill_bitmask` writes through the caller's
    :class:`MaskRows` handle.
    """

    __slots__ = ("_stack", "_trie")

    def __init__(self, trie: CsrTrie):
        """Create a matcher positioned at the trie root.

        Args:
            trie: Immutable trie to walk. Must outlive the matcher.
        """
        self._trie = trie
        self._stack: list[int] = [ROOT]

    @property
    def node(self) -> int:
        """Return the current node index (top of the stack)."""
        return self._stack[-1]

    def accept_token(self, token: int) -> bool:
        """Advance through ``token`` when the edge exists.

        Args:
            token: Token id to consume.

        Returns:
            True and push the destination node on success; False without
            changing state otherwise.
        """
        nxt = self._trie.step(self.node, token)
        if nxt < 0:
            return False
        self._stack.append(nxt)
        return True

    def rollback(self, k: int) -> None:
        """Pop the last ``k`` accepted tokens.

        Args:
            k: Number of steps to undo. ``0`` is a no-op.

        Raises:
            ValueError: If ``k`` is negative or exceeds the number of
                accepted tokens.
        """
        if k < 0 or k > len(self._stack) - 1:
            raise ValueError(f"rollback {k} vuot qua {len(self._stack) - 1} token da accept")
        if k:
            del self._stack[-k:]

    def is_terminated(self) -> bool:
        """Check whether the cursor is at a terminal leaf.

        Returns:
            True only when the current node ends a sequence and has no
            outgoing edges (no valid continuation).
        """
        return self._trie.is_terminal(self.node) and len(self._trie.allowed(self.node)) == 0

    def allowed_tokens(self) -> np.ndarray:
        """Return the sorted token ids allowed from the current node."""
        return self._trie.allowed(self.node)

    def allowed_count_hint(self) -> int | None:
        """Return the number of currently allowed tokens.

        Returns:
            Exact child count; always available for a trie, so never None.
            Used as a density hint for scheduling.
        """
        return len(self._trie.allowed(self.node))

    def fill_bitmask(self, rows: MaskRows, i: int) -> None:
        """Write the current allow-list into row ``i``.

        Args:
            rows: Destination bitmask view.
            i: Row index to overwrite with exactly the allowed tokens.
        """
        rows.allow_only(i, self.allowed_tokens())

    def num_accepted(self) -> int:
        """Return the number of tokens accepted since construction."""
        return len(self._stack) - 1
