"""Sharded global prefix index: advisory digest→node membership for routing.

Each node keeps its canonical prefix ownership in its local
:class:`~ayaka.prefix.store.RadixPagePrefixCache`; this module only tracks
*which node* claims which block so a router can place requests near KV that is
already resident. The block-hash space is split into disjoint shards
(:func:`shard_for_digest`): every digest is stored in and answered by exactly
one shard, so a lookup touches only the shards the prompt's digests land on.

Entries are advisory and expire by TTL: a stale hit wastes a route and is
counted, never mis-attached — the destination store revalidates the capability
before acquisition. The digests are the chained SHA-256 identities of
:mod:`ayaka.prefix.identity`, the exact digests the local store matches on, so
an index hit and a local hit agree by construction.
"""

from __future__ import annotations

import heapq
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import count
from typing import Any, Protocol, runtime_checkable

from ayaka.prefix.identity import (
    PrefixCacheContext,
    build_prefix_block_identities,
    full_token_blocks,
)

__all__ = [
    "GlobalIndexShard",
    "GlobalPrefixIndex",
    "IndexEntry",
    "NodePrefixScore",
    "PrefixIndexPublisher",
    "ShardClient",
    "prompt_digest_pairs",
    "shard_for_digest",
]

_DEFAULT_SHARD_CAPACITY = 1 << 14
_DEFAULT_TTL_NS = 30_000_000_000


def shard_for_digest(digest: bytes, num_shards: int) -> int:
    """Map one block digest to its shard in ``[0, num_shards)``.

    Uses the digest's leading bytes so a given block always lands on the same
    shard for a fixed shard count; growing the shard count re-places entries
    as a whole rather than per node.
    """
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if len(digest) < 2:
        raise ValueError("digest must carry at least two bytes to shard")
    return int.from_bytes(digest[:2], "big") % num_shards


@dataclass(frozen=True, slots=True)
class IndexEntry:
    """One advisory membership claim: ``node_id`` holds this block digest."""

    digest: bytes
    node_id: str
    epoch: int
    token_count: int
    last_seen_ns: int


@dataclass(frozen=True, slots=True)
class NodePrefixScore:
    """Per-node longest-prefix estimate for one prompt lookup."""

    node_id: str
    matched_blocks: int
    matched_tokens: int


@runtime_checkable
class ShardClient(Protocol):
    """Transport seam for one shard: local dict today, RPC client later."""

    def upsert(self, entries: Sequence[IndexEntry]) -> None: ...

    def delete(self, digests: Sequence[bytes], *, node_id: str) -> None: ...

    def members(self, digests: Sequence[bytes], *, now_ns: int) -> dict[bytes, frozenset[str]]: ...


class GlobalIndexShard:
    """One in-process shard of the digest→node membership map.

    Each digest maps to the set of nodes claiming it. Capacity counts
    ``(digest, node)`` claims; eviction is oldest-last-seen with lazy heap
    invalidation. TTL expiry is applied lazily when a digest is queried.
    """

    def __init__(
        self,
        *,
        shard_id: int,
        capacity: int = _DEFAULT_SHARD_CAPACITY,
        ttl_ns: int = _DEFAULT_TTL_NS,
    ) -> None:
        if capacity <= 0:
            raise ValueError("shard capacity must be positive")
        if ttl_ns <= 0:
            raise ValueError("shard ttl must be positive")
        self.shard_id = int(shard_id)
        self.capacity = int(capacity)
        self.ttl_ns = int(ttl_ns)
        self._entries: dict[bytes, dict[str, IndexEntry]] = {}
        self._order: list[tuple[int, bytes, str]] = []
        self._size = 0
        self._lock = threading.RLock()
        self.upserts_total = 0
        self.deletes_total = 0
        self.evictions_total = 0
        self.expiries_total = 0

    def upsert(self, entries: Sequence[IndexEntry]) -> None:
        with self._lock:
            for entry in entries:
                self._insert(entry)
                self.upserts_total += 1
            self._evict_overflow()

    def _insert(self, entry: IndexEntry) -> None:
        per_node = self._entries.setdefault(entry.digest, {})
        if entry.node_id not in per_node:
            self._size += 1
        per_node[entry.node_id] = entry
        heapq.heappush(self._order, (entry.last_seen_ns, entry.digest, entry.node_id))

    def _evict_overflow(self) -> None:
        while self._size > self.capacity and self._order:
            _, digest, node_id = heapq.heappop(self._order)
            per_node = self._entries.get(digest)
            if per_node is None or node_id not in per_node:
                continue  # replaced or deleted after this record was pushed
            del per_node[node_id]
            self._size -= 1
            if not per_node:
                self._entries.pop(digest, None)
            self.evictions_total += 1

    def delete(self, digests: Sequence[bytes], *, node_id: str) -> None:
        with self._lock:
            for digest in digests:
                per_node = self._entries.get(digest)
                if per_node is None:
                    continue
                if node_id in per_node:
                    self._size -= 1
                per_node.pop(node_id, None)
                if not per_node:
                    self._entries.pop(digest, None)
                self.deletes_total += 1

    def members(self, digests: Sequence[bytes], *, now_ns: int) -> dict[bytes, frozenset[str]]:
        """Membership for exactly these digests; expired claims are dropped."""
        result: dict[bytes, frozenset[str]] = {}
        with self._lock:
            for digest in digests:
                per_node = self._entries.get(digest)
                if not per_node:
                    continue
                for node_id, entry in list(per_node.items()):
                    if now_ns - entry.last_seen_ns > self.ttl_ns:
                        del per_node[node_id]
                        self._size -= 1
                        self.expiries_total += 1
                if not per_node:
                    self._entries.pop(digest, None)
                if per_node:
                    result[digest] = frozenset(per_node)
            return result

    def node_digests(self, node_id: str) -> tuple[bytes, ...]:
        """Digests currently claimed by ``node_id`` (node wipe support)."""
        with self._lock:
            return tuple(
                digest for digest, per_node in self._entries.items() if node_id in per_node
            )

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._order.clear()
            self._size = 0

    def size(self) -> int:
        with self._lock:
            return self._size

    def assert_invariants(self) -> None:
        with self._lock:
            counted = 0
            for digest, per_node in self._entries.items():
                if not per_node:
                    raise ValueError(f"shard {self.shard_id} keeps an empty digest")
                for node_id, entry in per_node.items():
                    if entry.digest != digest or entry.node_id != node_id:
                        raise ValueError("shard membership is misindexed")
                counted += len(per_node)
            if counted != self._size:
                raise ValueError("shard size counter drifted")


class GlobalPrefixIndex:
    """Fan every digest to its owning shard; aggregate per-node prefix runs.

    Lookup walks the prompt's chained digests *in order* and counts, per node,
    the longest contiguous run of digests that node claims. Shard membership
    is queried only for the shards the prompt's digests land on; sharding is
    an access partition, and a prefix spanning several shards still scores as
    one run.
    """

    def __init__(
        self,
        *,
        num_shards: int = 256,
        shards: Sequence[ShardClient] | None = None,
    ) -> None:
        if num_shards <= 0:
            raise ValueError("num_shards must be positive")
        if shards is not None and len(shards) != num_shards:
            raise ValueError("shards must cover exactly num_shards slots")
        self.num_shards = int(num_shards)
        self._shards: tuple[ShardClient, ...] = (
            tuple(shards)
            if shards is not None
            else tuple(GlobalIndexShard(shard_id=i) for i in range(self.num_shards))
        )
        self._epoch = count(1)

    def shard_for(self, digest: bytes) -> ShardClient:
        return self._shards[shard_for_digest(digest, self.num_shards)]

    def publish_digests(
        self,
        digests: Sequence[tuple[bytes, int]],
        *,
        node_id: str,
        now_ns: int,
    ) -> int:
        """Upsert ``(digest, token_count)`` pairs; returns the count applied."""
        grouped: dict[int, list[IndexEntry]] = {}
        for digest, token_count in digests:
            entry = IndexEntry(
                digest=digest,
                node_id=node_id,
                epoch=next(self._epoch),
                token_count=token_count,
                last_seen_ns=now_ns,
            )
            grouped.setdefault(shard_for_digest(digest, self.num_shards), []).append(entry)
        for shard_id, entries in grouped.items():
            self._shards[shard_id].upsert(entries)
        return len(digests)

    def withdraw_digests(self, digests: Sequence[bytes], *, node_id: str) -> int:
        grouped: dict[int, list[bytes]] = {}
        for digest in digests:
            grouped.setdefault(shard_for_digest(digest, self.num_shards), []).append(digest)
        for shard_id, subset in grouped.items():
            self._shards[shard_id].delete(subset, node_id=node_id)
        return len(digests)

    def clear_node(self, node_id: str) -> int:
        """Drop every entry claimed by ``node_id`` (full wipe on shutdown)."""
        withdrawn = 0
        for shard in self._shards:
            digests = shard.node_digests(node_id) if isinstance(shard, GlobalIndexShard) else ()
            if digests:
                shard.delete(digests, node_id=node_id)
                withdrawn += len(digests)
        return withdrawn

    def lookup(
        self,
        digests: Sequence[bytes],
        *,
        now_ns: int,
        block_tokens: int,
    ) -> tuple[NodePrefixScore, ...]:
        """Longest per-node contiguous prefix over the ordered digests."""
        if block_tokens <= 0:
            raise ValueError("block_tokens must be positive")
        grouped: dict[int, list[bytes]] = {}
        for digest in digests:
            grouped.setdefault(shard_for_digest(digest, self.num_shards), []).append(digest)
        membership: dict[bytes, frozenset[str]] = {}
        for shard_id, subset in grouped.items():
            membership.update(self._shards[shard_id].members(subset, now_ns=now_ns))

        best: dict[str, int] = {}
        active: dict[str, int] = {}
        for digest in digests:
            live = set(membership.get(digest, frozenset()))
            for node_id in [node for node in active if node not in live]:
                del active[node_id]
            for node_id in live:
                run = active.get(node_id, 0) + 1
                active[node_id] = run
                if run > best.get(node_id, 0):
                    best[node_id] = run
        return tuple(
            sorted(
                (
                    NodePrefixScore(
                        node_id=node_id,
                        matched_blocks=run,
                        matched_tokens=run * block_tokens,
                    )
                    for node_id, run in best.items()
                ),
                key=lambda score: (-score.matched_tokens, score.node_id),
            )
        )

    def total_entries(self) -> int:
        total = 0
        for shard in self._shards:
            counter = getattr(shard, "size", None)
            if callable(counter):
                counted: Any = counter()
                total += int(counted)
        return total

    def assert_invariants(self) -> None:
        for shard in self._shards:
            checker = getattr(shard, "assert_invariants", None)
            if callable(checker):
                checker()


def prompt_digest_pairs(
    token_ids: Sequence[int],
    *,
    page_size: int,
    context: PrefixCacheContext,
) -> tuple[tuple[bytes, int], ...]:
    """Chained digests plus token counts for every complete prompt block.

    Shares the identity computation with the local prefix store so the index
    and the store agree on the same digests.
    """
    blocks = full_token_blocks(token_ids, page_size=page_size)
    identities = build_prefix_block_identities(token_ids, page_size=page_size, context=context)
    return tuple(
        (identity.digest, len(block)) for identity, block in zip(identities, blocks, strict=True)
    )


class PrefixIndexPublisher:
    """Coalescing, failure-isolated bridge between the prefix service and index.

    A single owner thread (only when :meth:`start` was called) drains a bounded
    work map keyed by digest, so bursts of publishes and evictions collapse to
    the latest action per block. Index faults are recorded in
    ``last_error``/``dropped_total`` and never raised: the index is advisory
    and must not break request serving.
    """

    def __init__(
        self,
        index: GlobalPrefixIndex,
        *,
        node_id: str,
        batch: int = 512,
        flush_interval_s: float = 0.05,
    ) -> None:
        self._index = index
        self._node_id = str(node_id)
        self._batch = max(1, int(batch))
        self._flush_interval_s = max(0.0, float(flush_interval_s))
        self._work: dict[bytes, tuple[str, int]] = {}
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.flushes_total = 0
        self.dropped_total = 0
        self.last_error = ""

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def pending(self) -> int:
        with self._lock:
            return len(self._work)

    def publish(
        self, token_ids: Sequence[int], *, context: PrefixCacheContext, page_size: int
    ) -> None:
        """Enqueue upserts for every complete block of ``token_ids``."""
        pairs = prompt_digest_pairs(token_ids, page_size=page_size, context=context)
        with self._lock:
            for digest, token_count in pairs:
                self._work[digest] = ("upsert", token_count)

    def withdraw(
        self,
        token_ids: Sequence[int],
        *,
        context: PrefixCacheContext,
        page_size: int,
    ) -> None:
        """Enqueue deletes for every complete block of ``token_ids``."""
        pairs = prompt_digest_pairs(token_ids, page_size=page_size, context=context)
        with self._lock:
            for digest, _ in pairs:
                self._work[digest] = ("withdraw", 0)

    def withdraw_digest(self, digest: bytes) -> None:
        with self._lock:
            self._work[digest] = ("withdraw", 0)

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("publisher thread already started")
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._drain_loop, name="ayaka-prefix-index-publisher", daemon=True
        )
        self._thread.start()

    def stop(self) -> int:
        thread = self._thread
        self._thread = None
        self._stop.set()
        flushed = self.flush_once()
        if thread is not None:
            thread.join(timeout=5.0)
        return flushed

    def flush_once(self) -> int:
        """Drain at most one coalesced batch; returns the actions applied."""
        with self._lock:
            items = list(self._work.items())[: self._batch]
            for digest, _ in items:
                self._work.pop(digest, None)
        if not items:
            return 0
        try:
            now_ns = time.monotonic_ns()
            upserts = [(d, n) for d, (a, n) in items if a == "upsert"]
            withdrawals = [d for d, (a, _) in items if a == "withdraw"]
            if upserts:
                self._index.publish_digests(upserts, node_id=self._node_id, now_ns=now_ns)
            if withdrawals:
                self._index.withdraw_digests(withdrawals, node_id=self._node_id)
            self.flushes_total += 1
        except Exception as exc:  # advisory index must never break serving
            self.last_error = str(exc)
            self.dropped_total += len(items)
        return len(items)

    def _drain_loop(self) -> None:
        while self._thread is not None:
            applied = self.flush_once()
            if applied == 0:
                if self._stop.wait(self._flush_interval_s):
                    return

    def close(self) -> int:
        flushed = 0
        if self._thread is not None:
            flushed += self.stop()
        return flushed + self.flush_once()
