"""KV-aware request routing over the sharded global prefix index.

The router is a *placement* decision made before admission: it queries the
global prefix index for the longest prompt prefix each node already holds and
either keeps the request local (with the hit as a cache hint) or points it at
the node that holds the KV. It never mutates scheduler or lifecycle state; the
only FSM effect it causes is parking a deferred request in
``WAITING_REMOTE_KV`` while its KV travels, driven through
:class:`RemoteKVPending`.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ayaka.prefix.global_index import GlobalPrefixIndex, NodePrefixScore, prompt_digest_pairs
from ayaka.prefix.identity import PrefixCacheContext
from ayaka.request.schema import Request
from ayaka.request.states import RequestState

if TYPE_CHECKING:
    from ayaka.request.lifecycle import LifecycleManager

__all__ = [
    "KVAwareRouter",
    "NodeLoadProbe",
    "RemoteKVPending",
    "RequestRouter",
    "RouteDecision",
]

WAITING_REMOTE_KV = RequestState.WAITING_REMOTE_KV


@dataclass(frozen=True, slots=True)
class RouteDecision:
    """Where a request goes and what the router believes it will reuse."""

    node_id: str
    prefix_tokens: int
    needs_remote_kv: bool
    local: bool

    def __post_init__(self) -> None:
        if self.prefix_tokens < 0:
            raise ValueError("prefix_tokens must be non-negative")
        if self.needs_remote_kv and self.local:
            raise ValueError("a local placement cannot also need a remote fetch")


@runtime_checkable
class NodeLoadProbe(Protocol):
    """Free KV capacity per node; higher is better. Advisory only."""

    def __call__(self) -> Mapping[str, int]: ...


@runtime_checkable
class RequestRouter(Protocol):
    """Placement boundary in front of admission."""

    def route(self, request: Request) -> RouteDecision: ...


def _default_context() -> PrefixCacheContext:
    return PrefixCacheContext("default", cache_dtype="float32")


class KVAwareRouter:
    """Route near KV: global-index hit first, load as the fallback order.

    Load first, then prefix (mirrors ``DataParallelRouter``): a prefix hit on
    a full node must not evict anyone. When the winner is this node the
    decision is a local cache hint; when it is a peer the request is parked in
    ``WAITING_REMOTE_KV`` until the KV transfer lands (route-only milestone:
    no transport is invoked here yet).
    """

    def __init__(
        self,
        index: GlobalPrefixIndex,
        *,
        node_id: str,
        page_size: int,
        context_for: Callable[[Request], PrefixCacheContext] | None = None,
        load: NodeLoadProbe | None = None,
        min_prefix_tokens: int = 0,
    ) -> None:
        self._index = index
        self._node_id = str(node_id)
        self._page_size = int(page_size)
        self._context_for = context_for
        self._load = load
        if min_prefix_tokens < 0:
            raise ValueError("min_prefix_tokens must be non-negative")
        self._min_prefix_tokens = int(min_prefix_tokens)
        self.route_misses = 0
        self.remote_routes = 0
        self.local_routes = 0

    @property
    def node_id(self) -> str:
        return self._node_id

    def route(self, request: Request) -> RouteDecision:
        tokens = getattr(request, "prompt_token_ids", None)
        if not tokens:
            return self._local(0)
        context = (
            self._context_for(request) if self._context_for is not None else _default_context()
        )
        digests = [
            digest
            for digest, _ in prompt_digest_pairs(tokens, page_size=self._page_size, context=context)
        ]
        if not digests:
            return self._local(0)
        scores = self._index.lookup(
            digests, now_ns=time.monotonic_ns(), block_tokens=self._page_size
        )
        eligible = tuple(
            score for score in scores if score.matched_tokens >= self._min_prefix_tokens
        )
        if not eligible:
            self.route_misses += 1
            return self._local(0)
        best = self._pick(eligible)
        if best.node_id == self._node_id:
            return self._local(best.matched_tokens)
        self.remote_routes += 1
        return RouteDecision(
            node_id=best.node_id,
            prefix_tokens=best.matched_tokens,
            needs_remote_kv=True,
            local=False,
        )

    def _pick(self, eligible: tuple[NodePrefixScore, ...]) -> NodePrefixScore:
        """Load-first ordering over the index's own best-first scores."""
        free = self._load() if self._load is not None else {}
        if not free:
            return eligible[0]

        def key(score: NodePrefixScore) -> tuple[int, int, str]:
            # Load first: a prefix hit on a full node must not evict anyone.
            # Missing nodes rank last (they cannot receive the request).
            return (
                -free.get(score.node_id, -1),
                -score.matched_tokens,
                score.node_id,
            )

        return min(eligible, key=key)

    def _local(self, prefix_tokens: int) -> RouteDecision:
        self.local_routes += 1
        return RouteDecision(
            node_id=self._node_id,
            prefix_tokens=prefix_tokens,
            needs_remote_kv=False,
            local=True,
        )


class RemoteKVPending:
    """Registry of requests parked in ``WAITING_REMOTE_KV``.

    The transport poller calls :meth:`release` when the KV bytes landed (the
    request re-runs the authoritative local lookup) or :meth:`abandon` when
    the fetch failed and the request should queue for local prefill instead.
    A parked request is never scheduled: ``WAITING_REMOTE_KV`` is outside the
    scheduler's schedulable state set.
    """

    def __init__(self, requests: LifecycleManager) -> None:
        self._requests = requests
        self._pending: set[str] = set()
        self._lock = threading.RLock()

    @property
    def pending_ids(self) -> frozenset[str]:
        with self._lock:
            return frozenset(self._pending)

    def park(self, request_id: str) -> None:
        lifecycle = self._requests.find(request_id)
        if lifecycle is None:
            raise KeyError(f"unknown request {request_id!r}")
        with self._lock:
            self._pending.add(request_id)

    def release(self, request_id: str, *, num_cached_tokens: int = 0) -> bool:
        """WAITING_REMOTE_KV → CACHE_LOOKUP → WAITING with the acquired prefix."""
        lifecycle = self._requests.find(request_id)
        if lifecycle is None or not self._unpark(request_id):
            return False
        machine = lifecycle.machine
        if machine.state is not WAITING_REMOTE_KV:
            return False
        machine.on_remote_kv_ready()
        machine.on_cache_looked_up(num_cached_tokens=num_cached_tokens)
        return True

    def abandon(self, request_id: str) -> bool:
        """WAITING_REMOTE_KV → WAITING with no cache progress."""
        lifecycle = self._requests.find(request_id)
        if lifecycle is None or not self._unpark(request_id):
            return False
        if lifecycle.machine.state is not WAITING_REMOTE_KV:
            return False
        lifecycle.machine.on_remote_kv_abandoned()
        return True

    def discard(self, request_id: str) -> None:
        """Forget the tracking entry (request aborted or finished elsewhere)."""
        with self._lock:
            self._pending.discard(request_id)

    def _unpark(self, request_id: str) -> bool:
        with self._lock:
            if request_id not in self._pending:
                return False
            self._pending.discard(request_id)
            return True
