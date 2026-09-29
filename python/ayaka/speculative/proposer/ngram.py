"""N-gram (prompt-lookup) proposer.

Private index: per request, the positions of every token in its committed
context. A lookup takes the most recent ``max_scan_occurrences`` earlier
occurrences of the last token, measures how far each matches the context
suffix (up to ``max_matching_ngram_size``) and drafts the tokens that followed
the chosen occurrence. Memory is one integer per committed token and is freed
when the request is released; host work per lookup is bounded by
``max_scan_occurrences * max_matching_ngram_size`` comparisons, independent of
context length.

Public pool (opt-in): continuations of n-gram patterns observed in any request
of the same isolation namespace (tenant + cache salt). It is bounded per
namespace and globally with LRU eviction, so it cannot grow without limit, and
one namespace's patterns are never proposed to another.

Both structures are caches derived from committed tokens only; tokens are
never retracted once published, so updating them during planning cannot
corrupt a later step even if the planned step is dropped.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

from ayaka.speculative.config import NGramConfig, NGramSelection
from ayaka.speculative.interface import (
    DraftBatch,
    DraftProposal,
    ProposalRequest,
    ProposerCapabilities,
    ProposerRuntimeState,
    SpeculativeProposer,
)
from ayaka.speculative.mode import SpeculativeMode

__all__ = ["NGramProposer", "PublicNGramPool"]


class _RequestIndex:
    __slots__ = ("epoch", "positions", "tokens")

    def __init__(self, epoch: int) -> None:
        self.epoch = epoch
        self.tokens: list[int] = []
        self.positions: dict[int, list[int]] = {}

    def sync(self, tokens: tuple[int, ...]) -> int:
        """Index committed tokens appended since the last call; return the old length."""
        old = len(self.tokens)
        if old > len(tokens) or (old and self.tokens[-1] != tokens[old - 1]):
            # Committed history never shrinks or changes; a mismatch means a
            # new incarnation reused the id, so the index starts over.
            self.tokens.clear()
            self.positions.clear()
            old = 0
        for position in range(old, len(tokens)):
            token = tokens[position]
            self.tokens.append(token)
            self.positions.setdefault(token, []).append(position)
        return old


def _truncate_at_stop(tokens: list[int], stops: frozenset[int]) -> tuple[int, ...]:
    """Drop everything after the first stop token: it could never publish."""
    for index, token in enumerate(tokens):
        if token in stops:
            return tuple(tokens[: index + 1])
    return tuple(tokens)


class PublicNGramPool:
    """Namespace-partitioned, LRU-bounded pattern -> continuation store."""

    def __init__(self, config: NGramConfig, continuation_tokens: int) -> None:
        self.config = config
        self.continuation_tokens = continuation_tokens
        self._pools: dict[str, OrderedDict[tuple[int, ...], list[tuple[int, ...]]]] = {}
        self.size = 0
        self.evictions = 0

    def record(self, namespace: str, tokens: Sequence[int], old_length: int) -> None:
        """Insert patterns whose continuation became complete since ``old_length``."""
        span = self.continuation_tokens
        first = max(old_length - span + 1, self.config.min_matching_ngram_size)
        for end in range(first, len(tokens) - span + 1):
            continuation = tuple(tokens[end : end + span])
            for n in range(
                self.config.min_matching_ngram_size, self.config.max_matching_ngram_size + 1
            ):
                if end - n < 0:
                    break
                self._insert(namespace, tuple(tokens[end - n : end]), continuation)

    def _insert(
        self, namespace: str, pattern: tuple[int, ...], continuation: tuple[int, ...]
    ) -> None:
        pool = self._pools.setdefault(namespace, OrderedDict())
        entries = pool.get(pattern)
        if entries is None:
            pool[pattern] = [continuation]
            self.size += 1
        else:
            pool.move_to_end(pattern)
            if continuation in entries:
                entries.remove(continuation)
            entries.append(continuation)
            limit = self.config.public_pool_max_continuations if self.config.keep_all else 1
            del entries[:-limit]
        while len(pool) > self.config.public_pool_namespace_max_patterns:
            self._evict(namespace)
        while self.size > self.config.public_pool_max_patterns:
            largest = max(self._pools, key=lambda name: len(self._pools[name]))
            self._evict(largest)

    def _evict(self, namespace: str) -> None:
        pool = self._pools[namespace]
        pool.popitem(last=False)
        self.size -= 1
        self.evictions += 1
        if not pool:
            del self._pools[namespace]

    def lookup(self, namespace: str, context: Sequence[int]) -> tuple[int, ...]:
        pool = self._pools.get(namespace)
        if not pool:
            return ()
        for n in range(
            self.config.max_matching_ngram_size, self.config.min_matching_ngram_size - 1, -1
        ):
            if len(context) < n:
                continue
            entries = pool.get(tuple(context[len(context) - n :]))
            if entries:
                pool.move_to_end(tuple(context[len(context) - n :]))
                if self.config.selection is NGramSelection.OLDEST:
                    return entries[0]
                if self.config.selection is NGramSelection.LONGEST:
                    return max(entries, key=len)
                return entries[-1]
        return ()

    def clear(self) -> None:
        self._pools.clear()
        self.size = 0


class NGramProposer(SpeculativeProposer):
    """Host prompt-lookup drafts from committed context (and an optional pool)."""

    def __init__(self, config: NGramConfig, max_draft_tokens: int) -> None:
        self.config = config
        self.max_draft_tokens = max_draft_tokens
        self._indexes: dict[str, _RequestIndex] = {}
        self.public = PublicNGramPool(config, max_draft_tokens) if config.public_pool else None
        self.lookups = 0
        self.private_hits = 0
        self.public_hits = 0
        self._capabilities = ProposerCapabilities(mode=SpeculativeMode.NGRAM, host_drafts=True)

    @property
    def capabilities(self) -> ProposerCapabilities:
        return self._capabilities

    def _index(self, request: ProposalRequest) -> tuple[_RequestIndex, int]:
        index = self._indexes.get(request.request_id)
        if index is None or index.epoch != request.sequence_epoch:
            index = self._indexes[request.request_id] = _RequestIndex(request.sequence_epoch)
        old = index.sync(request.token_ids)
        return index, old

    def _private(self, index: _RequestIndex, limit: int) -> tuple[int, ...]:
        tokens = index.tokens
        length = len(tokens)
        occurrences = index.positions.get(tokens[-1], ())
        best: list[tuple[int, int]] = []  # (match length, position)
        best_length = 0
        scanned = 0
        for cursor in range(len(occurrences) - 1, -1, -1):
            position = occurrences[cursor]
            if position >= length - 1:
                continue
            if scanned >= self.config.max_scan_occurrences:
                break
            scanned += 1
            matched = 1
            while (
                matched < self.config.max_matching_ngram_size
                and position - matched >= 0
                and tokens[position - matched] == tokens[length - 1 - matched]
            ):
                matched += 1
            if matched < self.config.min_matching_ngram_size or matched < best_length:
                continue
            if matched > best_length:
                best, best_length = [], matched
            best.append((matched, position))
        if not best:
            return ()
        positions = [position for _, position in best]  # most recent first
        selection = self.config.selection
        if selection is NGramSelection.MOST_RECENT:
            chosen = positions[0]
        elif selection is NGramSelection.OLDEST:
            chosen = positions[-1]
        else:
            full = [p for p in positions if length - (p + 1) >= limit]
            chosen = full[0] if full else positions[-1]
        return tuple(tokens[chosen + 1 : chosen + 1 + limit])

    def propose(
        self,
        batch: Sequence[ProposalRequest],
        *,
        max_draft_tokens: int,
        runtime_state: ProposerRuntimeState,
    ) -> DraftBatch:
        del runtime_state
        proposals: list[DraftProposal] = []
        for request in batch:
            index, old = self._index(request)
            if self.public is not None:
                self.public.record(request.namespace, index.tokens, old)
            limit = min(max_draft_tokens, request.max_draft_tokens, self.max_draft_tokens)
            if limit <= 0:
                continue
            self.lookups += 1
            drafts = self._private(index, limit)
            if drafts:
                self.private_hits += 1
            elif self.public is not None:
                drafts = self.public.lookup(request.namespace, index.tokens)[:limit]
                if drafts:
                    self.public_hits += 1
            drafts = _truncate_at_stop(list(drafts), request.stop_token_ids)
            if drafts:
                proposals.append(
                    DraftProposal.linear(request.request_id, drafts, len(request.token_ids))
                )
        return DraftBatch(tuple(proposals))

    def release_request(self, request_id: str) -> None:
        self._indexes.pop(request_id, None)

    def close(self) -> None:
        self._indexes.clear()
        if self.public is not None:
            self.public.clear()

    @property
    def live_requests(self) -> int:
        return len(self._indexes)

    def stats(self) -> dict[str, Any]:
        return {
            "live_requests": len(self._indexes),
            "indexed_tokens": sum(len(index.tokens) for index in self._indexes.values()),
            "lookups": self.lookups,
            "private_hits": self.private_hits,
            "public_hits": self.public_hits,
            "public_patterns": 0 if self.public is None else self.public.size,
            "public_evictions": 0 if self.public is None else self.public.evictions,
        }
