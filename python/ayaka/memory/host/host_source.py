"""Exact-tier host allocation.  The source reports; the caller decides.

The race this closes: a source that answers ``pinned`` from
``torch.cuda.is_available()`` and then degrades a pinned request to pageable
inside ``alloc`` leaves its caller holding a ledger charge for HOST_PINNED and an
allocation that is HOST_PAGEABLE —

    probe says PINNED -> ledger charges PINNED -> allocation is PAGEABLE

— and nothing raises.  A driver that appears or disappears between the probe and
the allocation, a cgroup pinned limit reached by another process, a NUMA node
that runs out: each of those turns the ledger into fiction, and a fictional
ledger is worse than none because every later admission decision trusts it.

So the contract inverts.  :meth:`TorchExactHostSource.allocate_exact` returns the
tier it *actually* produced (``HostAllocation.actual_tier``) and is forbidden
from substituting one for another.  Whether a failed pinned allocation should be
retried as pageable is a *policy* question, and policy belongs to whichever
owner knows the whole pool must share one decision — not to this module.

Physical pinned allocation is shared with the byte sources through
:func:`ayaka.utils.torch_memory.empty_host_tensor`, so all three paths refuse a
pageable answer to a pinned request the same way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ayaka.exceptions import RuntimeMemoryError
from ayaka.types import MemoryTier
from ayaka.utils.import_utils import CapabilityError
from ayaka.utils.torch_memory import empty_host_tensor

__all__ = [
    "HOST_TIERS",
    "AllocationError",
    "HostAllocation",
    "HostMemorySource",
    "TorchExactHostSource",
]

HOST_TIERS: tuple[MemoryTier, ...] = (MemoryTier.HOST_PINNED, MemoryTier.HOST_PAGEABLE)


class AllocationError(RuntimeMemoryError):
    """The requested tier could not be allocated at the requested size.

    Distinct from returning a different tier, which is forbidden outright: a
    caller can decide what to do about a failure, but it cannot decide anything
    useful about a substitution it was never told happened.
    """


@dataclass(frozen=True, slots=True)
class HostAllocation:
    """One host buffer, tagged with the tier it really is.

    ``actual_tier`` is not a copy of the request.  It is the field the pool
    compares against the request, and the comparison is the whole point — a
    source that gets this wrong is caught at the seam rather than three layers
    later when the ledger and the machine disagree.
    """

    buffer: Any
    actual_tier: MemoryTier
    nbytes: int
    numa_node: int | None = None

    def __post_init__(self) -> None:
        if self.actual_tier not in HOST_TIERS:
            raise ValueError(f"{self.actual_tier} is not a host tier")
        if self.nbytes < 1:
            raise ValueError("a host allocation must have a positive size")

    @property
    def pinned(self) -> bool:
        return self.actual_tier is MemoryTier.HOST_PINNED

    def memoryview(self) -> memoryview:
        """A writable window over the buffer.

        ``memoryview`` and not the torch tensor: the reader that fills this
        buffer lives in ``ayaka.model_loader``, a peer package that must not
        learn what ``ayaka.weights`` allocates with.  A stdlib type is the
        widest interface that carries no dependency.
        """
        return memoryview(self.buffer.numpy())

    def close(self) -> None:
        """Drop the buffer.  Idempotent by construction — the dataclass is
        frozen, so this only exists to make the release point explicit at call
        sites that must not rely on refcounting."""


@runtime_checkable
class HostMemorySource(Protocol):
    """Allocates host memory at an exactly specified tier, or fails."""

    def supports(self, tier: MemoryTier) -> bool:
        """Whether this source can produce ``tier`` at all, right now.

        A prediction, and explicitly allowed to be wrong by the time
        ``allocate_exact`` runs — which is why the returned allocation carries
        its own tier and the caller checks it.
        """
        ...

    def allocate_exact(
        self, nbytes: int, *, tier: MemoryTier, numa_node: int | None = None
    ) -> HostAllocation: ...


class TorchExactHostSource:
    """Torch-backed host memory that refuses to substitute tiers.

    The behavioural difference from ``TorchHostSource`` is one line and the
    whole point: a pinned request with no CUDA driver raises
    :class:`AllocationError` instead of quietly returning pageable memory.
    """

    __slots__ = ("_torch",)

    def __init__(self) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - fast lane installs torch
            raise RuntimeMemoryError("TorchExactHostSource needs torch") from exc
        self._torch = torch

    def supports(self, tier: MemoryTier) -> bool:
        if tier is MemoryTier.HOST_PAGEABLE:
            return True
        if tier is MemoryTier.HOST_PINNED:
            return bool(self._torch.cuda.is_available())
        return False

    def allocate_exact(
        self, nbytes: int, *, tier: MemoryTier, numa_node: int | None = None
    ) -> HostAllocation:
        if tier not in HOST_TIERS:
            raise AllocationError(f"{tier} is not a host tier")
        if nbytes < 1:
            raise AllocationError("host allocation needs a positive size")
        want_pinned = tier is MemoryTier.HOST_PINNED
        if want_pinned and not self._torch.cuda.is_available():
            # The substitution a pageable fallback would perform silently.
            # Raising here is what lets the pool run its own fallback *and*
            # roll the pinned reservation back first.
            raise AllocationError(
                "pinned host memory needs a CUDA driver; refusing to substitute "
                "pageable memory for a pinned request"
            )
        try:
            buffer, pinned = empty_host_tensor((nbytes,), "uint8", pinned=want_pinned)
        except CapabilityError as exc:
            raise AllocationError(f"could not allocate {nbytes} B at {tier}: {exc}") from exc
        # Report what torch produced, never what was asked for.
        actual = MemoryTier.HOST_PINNED if pinned else MemoryTier.HOST_PAGEABLE
        return HostAllocation(buffer=buffer, actual_tier=actual, nbytes=nbytes, numa_node=numa_node)
