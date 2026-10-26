"""Serving-side weight update: drain, apply one transaction, publish (research R6).

``ServingService.update_weights`` runs on the engine owner thread:

``BEGIN``   refuse unsupported setups before anything changes; hold admission;
            optionally abort running requests (they end ``cancelled``).
``DRAIN``   wait, up to a finite deadline, until no owner can touch KV or the
            weights: no generation, unfinished request, ticket or transfer.
``LOAD``    ``ayaka.weights.update.apply_weight_transaction`` with a verified
            host backup: all-new, all-old (rolled back) or failed.
``COMMIT``  bump the weights revision used by prefix identity, flush the prefix
            cache of the old generation, resume admission.

A request admitted before ``BEGIN`` finishes on the old weights; one queued
after it runs on the new ones; none sees both. ``TIMEOUT`` and ``REFUSED`` never
touch the weights. ``FAILED`` (no verified restore) stops the engine.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import Future
from dataclasses import dataclass

from torch import nn

from ayaka.weights.update import WeightSource, WeightUpdateStatus

__all__ = ["WeightOperation", "WeightUpdateResult", "WeightUpdateStatus"]


@dataclass(frozen=True, slots=True)
class WeightUpdateResult:
    status: WeightUpdateStatus
    operation: int
    revision_before: str
    #: Revision serving after the operation (the new one only when committed).
    revision_after: str
    #: Weight generation serving after the operation; +1 per commit.
    generation: int
    digest_before: str = ""
    digest_after: str = ""
    mutated: bool = False
    backup_bytes: int = 0
    waited_seconds: float = 0.0
    load_seconds: float = 0.0
    #: Requests aborted at BEGIN because the caller asked for it.
    aborted_requests: tuple[str, ...] = ()
    #: Owners still busy when a TIMEOUT answer was given.
    blockers: tuple[str, ...] = ()
    detail: str = ""

    @property
    def committed(self) -> bool:
        return self.status is WeightUpdateStatus.COMMITTED

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "operation": self.operation,
            "revision_before": self.revision_before,
            "revision_after": self.revision_after,
            "generation": self.generation,
            "digest_before": self.digest_before,
            "digest_after": self.digest_after,
            "mutated": self.mutated,
            "backup_bytes": self.backup_bytes,
            "waited_seconds": self.waited_seconds,
            "load_seconds": self.load_seconds,
            "aborted_requests": list(self.aborted_requests),
            "blockers": list(self.blockers),
            "detail": self.detail,
        }


@dataclass(slots=True)
class WeightOperation:
    """The single parked weight update; engine thread only."""

    operation: int
    started_ns: int
    deadline_ns: int
    source: WeightSource
    revision: str
    backup: bool
    probe: Callable[[Mapping[str, nn.Module]], None] | None
    future: Future
    aborted: tuple[str, ...] = ()
