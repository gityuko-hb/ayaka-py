"""Preemption adapters and deterministic victim selection."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from ayaka.handles import SequenceHandle
from ayaka.request.lifecycle import RequestLifecycle
from ayaka.sched.interfaces import PreemptionCallback

__all__ = [
    "CallbackPreemptionController",
    "DisabledPreemptionController",
    "select_preemption_victim",
]


@dataclass(frozen=True, slots=True)
class DisabledPreemptionController:
    def preempt(
        self,
        lifecycle: RequestLifecycle,
        sequence: SequenceHandle,
        *,
        mode: str,
    ) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class CallbackPreemptionController:
    """Adapter around the physical KV/runtime preemption transaction."""

    callback: PreemptionCallback

    def preempt(
        self,
        lifecycle: RequestLifecycle,
        sequence: SequenceHandle,
        *,
        mode: str,
    ) -> bool:
        return bool(self.callback(lifecycle, sequence, mode))


def select_preemption_victim(
    lifecycles: Iterable[RequestLifecycle],
    *,
    excluded_request_ids: set[str] | frozenset[str] = frozenset(),
) -> RequestLifecycle | None:
    """Choose a deterministic low-priority/newer victim.

    Older requests retain service among equal priorities, reducing starvation.
    Terminal and explicitly excluded requests are never returned.
    """

    victims = [
        lifecycle
        for lifecycle in lifecycles
        if not lifecycle.is_terminal
        and lifecycle.request_id not in excluded_request_ids
        and not lifecycle.token.is_cancelled
        and lifecycle.inflight_slice is None
    ]
    if not victims:
        return None
    victims.sort(
        key=lambda lifecycle: (
            int(getattr(lifecycle.request, "priority", 0) or 0),
            -int(getattr(lifecycle.request, "arrival_ns", 0) or 0),
            lifecycle.request_id,
        )
    )
    return victims[0]
