"""Batch mutation descriptors for sampling slot allocation and lifecycle management.

Defines delta structures representing request admissions, evictions, and slot
migrations exchanged between the engine scheduler and the sampling metadata store.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ayaka.sampling.params import SamplingParams


@dataclass(frozen=True, slots=True)
class AddedRequest:
    """Descriptor for a newly admitted request assigned to a sampling slot.

    Attributes:
        slot: Physical sampling slot allocated to the request.
        params: Sampling configuration parameters for the request.
        request_index: Monotonic per-request index used for RNG seed derivation.
    """

    slot: int
    params: SamplingParams
    request_index: int = 0


@dataclass(frozen=True, slots=True)
class RemovedRequest:
    """Descriptor for a completed or aborted request releasing its sampling slot.

    Attributes:
        slot: Physical sampling slot being deallocated.
    """

    slot: int


@dataclass(frozen=True, slots=True)
class MovedRequest:
    """Descriptor for a request undergoing slot migration or compaction.

    Attributes:
        src: Source physical sampling slot index.
        dst: Destination physical sampling slot index.
    """

    src: int
    dst: int


@dataclass(slots=True)
class BatchUpdate:
    """Batch mutation delta aggregating admissions, evictions, and slot moves.

    Attributes:
        batch_size: Target active batch size after applying all updates.
        removed: Requests to evict from their allocated slots.
        added: Newly admitted requests to populate in assigned slots.
        moved: Existing requests being migrated across slot indices.
    """

    batch_size: int
    removed: list[RemovedRequest] = field(default_factory=list)
    added: list[AddedRequest] = field(default_factory=list)
    moved: list[MovedRequest] = field(default_factory=list)

    def sort_removed(self) -> None:
        """Sort removed requests in descending slot order.

        Descending order allows safe in-place eviction or compaction without
        corrupting subsequent slot indices.
        """
        # Sort by slot descending to avoid index invalidation during sequential eviction.
        self.removed.sort(key=lambda r: r.slot, reverse=True)
