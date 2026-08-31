from __future__ import annotations

from dataclasses import dataclass


def _validate_index_generation(index: int, generation: int) -> None:
    """Validate the shared ``(index, generation)`` handle invariant.

    Indexes are non-negative slot identities; generations start at 1 so that a
    freshly initialized slot (generation 0) can never be confused with a live
    object.
    """
    if index < 0:
        raise ValueError("handle index must be non-negative")
    if generation <= 0:
        raise ValueError("handle generation must be positive")


@dataclass(frozen=True, slots=True, order=True)
class SequenceHandle:
    """Identity of one live sequence slot in the sequence arena.

    ``index`` selects the arena slot; ``generation`` is bumped each time the
    slot is reused, so stale handles from a released sequence are rejected.
    """

    index: int
    generation: int

    def __post_init__(self) -> None:
        _validate_index_generation(self.index, self.generation)


@dataclass(frozen=True, slots=True, order=True)
class KVPageHandle:
    """Identity of one physical page in the allocator.

    ``index`` is the physical page index shared by every layer's K/V buffer in
    the homogeneous MHA path; ``generation`` increments whenever the page is
    freed and reallocated, invalidating stale handles.
    """

    index: int
    generation: int

    def __post_init__(self) -> None:
        _validate_index_generation(self.index, self.generation)


@dataclass(frozen=True, slots=True, order=True)
class PrefixHandle:
    """Generation-safe identity for a cached full-page prefix node."""

    index: int
    generation: int

    def __post_init__(self) -> None:
        _validate_index_generation(self.index, self.generation)


@dataclass(frozen=True, slots=True, order=True)
class PrefixMatchHandle:
    """One-shot opaque identity for a scheduler-facing prefix lookup.

    The scheduler holds this while planning; it owns no page references and is
    consumed (or discarded) exactly once when the reservation is attempted.
    """

    index: int
    generation: int

    def __post_init__(self) -> None:
        _validate_index_generation(self.index, self.generation)


@dataclass(frozen=True, slots=True, order=True)
class PhysicalPageId:
    """Stable index of a physical page inside the KV storage buffers.

    Unlike ``KVPageHandle``, this carries no generation: it is the storage-side
    addressing identity consumed by tensors and kernels.
    """

    value: int

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError("physical page id must be non-negative")


@dataclass(frozen=True, slots=True, order=True)
class MemoryTransactionHandle:
    """Identity of one open tentative-planning transaction.

    Carries the scheduler ``step_id`` so reservations created inside the
    transaction inherit the step they plan for.
    """

    index: int
    generation: int
    step_id: int

    def __post_init__(self) -> None:
        _validate_index_generation(self.index, self.generation)
        if self.step_id < 0:
            raise ValueError("step_id must be non-negative")


@dataclass(frozen=True, slots=True, order=True)
class KVReservationHandle:
    """Opaque identity of one per-sequence reservation inside a transaction.

    The scheduler stores this in its schedule plan but never inspects its
    contents; the manager maps it back to the full reservation record.
    """

    index: int
    generation: int
    step_id: int

    def __post_init__(self) -> None:
        _validate_index_generation(self.index, self.generation)
        if self.step_id < 0:
            raise ValueError("step_id must be non-negative")


@dataclass(frozen=True, slots=True, order=True)
class StepMemoryLeaseHandle:
    """Identity of a frozen execution lease produced by ``prepare_step``.

    The lease freezes the transaction's reservations into an immutable,
    launch able step. Terminal (completed/aborted/failed) leases reject further
    mutation as stale.
    """

    index: int
    generation: int
    step_id: int

    def __post_init__(self) -> None:
        _validate_index_generation(self.index, self.generation)
        if self.step_id < 0:
            raise ValueError("step_id must be non-negative")
