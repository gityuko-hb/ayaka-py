"""CSR trie + TrieMatcher."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from ayaka.sampling.mask.producer import MaskRows

ROOT = 0


class CsrTrie:
    __slots__ = ("children_next", "children_off", "children_token", "n_nodes", "terminal")

    def __init__(self, children_off, children_token, children_next, terminal) -> None:
        self.children_off = children_off
        self.children_token = children_token
        self.children_next = children_next
        self.terminal = terminal
        self.n_nodes = len(terminal)

    @classmethod
    def build(cls, sequences: Iterable[Sequence[int]]) -> CsrTrie:
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
        s, e = int(self.children_off[node]), int(self.children_off[node + 1])
        sliced: np.ndarray = self.children_token[s:e]
        return sliced

    def step(self, node: int, token: int) -> int:
        s, e = int(self.children_off[node]), int(self.children_off[node + 1])
        if s == e:
            return -1
        i = int(np.searchsorted(self.children_token[s:e], token))
        if i < (e - s) and int(self.children_token[s + i]) == token:
            return int(self.children_next[s + i])
        return -1

    def is_terminal(self, node: int) -> bool:
        return bool(self.terminal[node])


class TrieMatcher:
    __slots__ = ("_stack", "_trie")

    def __init__(self, trie: CsrTrie):
        self._trie = trie
        self._stack: list[int] = [ROOT]

    @property
    def node(self) -> int:
        return self._stack[-1]

    def accept_token(self, token: int) -> bool:
        nxt = self._trie.step(self.node, token)
        if nxt < 0:
            return False
        self._stack.append(nxt)
        return True

    def rollback(self, k: int) -> None:
        if k < 0 or k > len(self._stack) - 1:
            raise ValueError(f"rollback {k} vuot qua {len(self._stack) - 1} token da accept")
        if k:
            del self._stack[-k:]

    def is_terminated(self) -> bool:
        return self._trie.is_terminal(self.node) and len(self._trie.allowed(self.node)) == 0

    def allowed_tokens(self) -> np.ndarray:
        return self._trie.allowed(self.node)

    def allowed_count_hint(self) -> int | None:
        return len(self._trie.allowed(self.node))

    def fill_bitmask(self, rows: MaskRows, i: int) -> None:
        rows.allow_only(i, self.allowed_tokens())

    def num_accepted(self) -> int:
        return len(self._stack) - 1
