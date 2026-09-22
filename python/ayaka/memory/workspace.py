"""One growing scratch buffer per owner, sub-divided by offset each step.

The executor is handed a :class:`~ayaka.plan.MemoryPlan` listing the scratch
buffers this step needs and asks for exactly those; the planner, not the
executor, decides how much scratch there is.

**One buffer, sub-divided by offset.** N separate allocations per step means N
free-list lookups, N chances to land on a differently-sized block than last
step, and a peak that is the sum of every buffer's own peak rather than the peak
of their sum. One buffer sized to the largest step ever seen means: allocate
once, hand out offsets forever after, and the CUDA-graph capture sees the same
addresses on every replay — which it must, because a captured graph bakes in the
pointers it was traced with. If the address moves between capture and replay the
graph writes to the old one; nothing errors and the output is silently wrong.
So the workspace buffer grows monotonically, never shrinks, and
:meth:`WorkspaceManager.prepare` reports growth so every captured graph can be
discarded.

**Two owners, two growth policies.** ``WORKSPACE`` grows: it is backend scratch,
the planner sized it, and the ``grew`` flag already tells the caller to drop its
graphs. ``ACTIVATION`` does not grow after
:meth:`WorkspaceManager.initialize`: the activation figure is what the KV
capacity planner sizes pages against (measured through
:func:`ayaka.utils.torch_memory.peak_memory_bytes`), so growing it later would
make that decision retroactively wrong. Refusing is the only honest option, so
the arena raises instead of quietly reallocating.

Backing bytes come from a :class:`~ayaka.memory.caching.CachingAllocator`, so
the class-rounded slack between an arena's capacity and the pool's charge is
visible as :attr:`WorkspaceStats.internal_slack_bytes` rather than reading like
a leak.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from ayaka.memory.arena import Arena
from ayaka.memory.caching import CachingAllocator
from ayaka.memory.region import MemoryRegion
from ayaka.plan import MemoryPlan, WorkspaceRequest
from ayaka.types import MemoryOwner
from ayaka.utils.math_utils import align_up

__all__ = [
    "ActivationOverflowError",
    "WorkspaceCeilingError",
    "WorkspaceLease",
    "WorkspaceManager",
    "WorkspaceStats",
]

#: Owners this manager backs with a buffer, each with its own growth policy.
#: A request naming any other owner is refused — ``KV`` has its own allocator
#: and ``WEIGHT`` is not step-scoped, so routing either here would double-charge
#: the ledger against memory somebody else already owns.
_GROWABLE: dict[MemoryOwner, bool] = {
    MemoryOwner.WORKSPACE: True,
    MemoryOwner.ACTIVATION: False,
}


class ActivationOverflowError(RuntimeError):
    """A step needs more activation memory than the profiled peak.

    Distinct from ``ValueError`` because the recovery differs: this is not a bad
    argument, it is a profiling run that did not reach the real worst case, and
    the fix is upstream in whichever shape the profiling pass chose.
    """


class WorkspaceCeilingError(RuntimeError):
    """Workspace growth would exceed the budget frozen at bootstrap.

    The ceiling is reserved in the capacity snapshot; growing past it would
    invalidate the KV budget that was computed from it, so the step is refused
    rather than silently reallocated. Recovery is a rebuild with a larger
    frozen budget, never an in-place grow.
    """


@dataclass(frozen=True, slots=True)
class WorkspaceStats:
    """One arena's capacity accounting.

    ``capacity_bytes`` is what the arena can address; ``internal_slack_bytes``
    is what the pool holds beyond it (size-class rounding and any alignment
    offset).  The two reconcile the ledger against the arena:
    ``pool.live_bytes == capacity_bytes + internal_slack_bytes`` for the live
    buffers, while blocks sitting on the pool's free lists appear in
    ``CachingAllocator.stats().cached_bytes`` instead.
    """

    capacity_bytes: int
    used_bytes: int
    high_water_bytes: int
    internal_slack_bytes: int
    num_growths: int
    num_steps: int

    @property
    def headroom_bytes(self) -> int:
        return self.capacity_bytes - self.high_water_bytes

    @property
    def internal_slack_ratio(self) -> float:
        """``internal_slack_bytes`` as a fraction of capacity."""
        if not self.capacity_bytes:
            return 0.0
        return self.internal_slack_bytes / self.capacity_bytes


class WorkspaceLease:
    """One step's slice of the workspace, keyed by request name.

    Not a dict of tensors — a dict of regions. Materialising a tensor is the
    backend's job, and doing it here would put torch in the memory layer.
    ``close`` must be called when the step no longer reads the regions; the
    manager refuses to prepare the next step (and therefore to grow the buffer)
    while a lease is still live.
    """

    __slots__ = ("_closed", "_generation", "_manager", "_regions", "step_bytes")

    def __init__(
        self,
        regions: dict[str, MemoryRegion],
        generation: int,
        step_bytes: int,
        *,
        manager: WorkspaceManager | None = None,
    ) -> None:
        self._regions = regions
        self._generation = generation
        self.step_bytes = step_bytes
        self._manager = manager
        self._closed = False

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def closed(self) -> bool:
        return self._closed

    def __getitem__(self, name: str) -> MemoryRegion:
        region = self._regions.get(name)
        if region is None:
            raise KeyError(
                f"no workspace named {name!r} in this step; the plan requested "
                f"{sorted(self._regions)}"
            )
        return region

    def __contains__(self, name: object) -> bool:
        return name in self._regions

    def names(self) -> tuple[str, ...]:
        return tuple(self._regions)

    def close(self) -> None:
        """End this step's slice; idempotent."""
        if self._closed:
            return
        self._closed = True
        if self._manager is not None:
            self._manager._release(self)

    def __enter__(self) -> WorkspaceLease:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class WorkspaceManager:
    """Owns one growing buffer and sub-divides it per step."""

    __slots__ = (
        "_arenas",
        "_initial_bytes",
        "_live_lease",
        "_lock",
        "_num_steps",
    )

    def __init__(
        self,
        allocator: CachingAllocator,
        *,
        initial_bytes: int = 0,
        workspace_ceiling_bytes: int | None = None,
    ) -> None:
        self._initial_bytes = initial_bytes
        self._num_steps = 0
        self._live_lease: WorkspaceLease | None = None
        self._lock = threading.Lock()
        self._arenas: dict[MemoryOwner, _OwnedBuffer] = {
            owner: _OwnedBuffer(
                allocator,
                owner,
                may_grow=growable,
                ceiling_bytes=workspace_ceiling_bytes if growable else None,
            )
            for owner, growable in _GROWABLE.items()
        }
        if initial_bytes:
            self._arenas[MemoryOwner.WORKSPACE].grow_to(initial_bytes, floor=initial_bytes)

    # ── lifecycle ────────────────────────────────────────────────────────────

    def initialize(self, activation_bytes: int) -> None:
        """Fix the activation budget.  Callable once, before the first step.

        Separate from ``__init__`` because the number comes from the profiling
        pass, which needs a live device — and the manager is constructed in the
        parent process, before the CUDA fork boundary.

        After this returns, an activation request larger than ``activation_bytes``
        raises :class:`ActivationOverflowError`. That is the point: the KV page
        count has already been computed from this figure by the time a step runs.
        """
        arena = self._arenas[MemoryOwner.ACTIVATION]
        if arena.capacity_bytes:
            raise RuntimeError(
                f"activation budget already fixed at {arena.capacity_bytes} bytes; "
                "re-profiling after KV pages are handed out cannot take them back"
            )
        if activation_bytes:
            arena.grow_to(activation_bytes, floor=activation_bytes)

    # ── the step API ─────────────────────────────────────────────────────────

    def prepare(self, plan: MemoryPlan) -> tuple[WorkspaceLease, bool]:
        """Carve this step's buffers. Returns ``(lease, grew)``.

        ``grew`` true means the *workspace* buffer moved, so every captured CUDA
        graph is now pointing at a freed address and must be discarded. The
        caller is responsible for acting on it; silently reallocating under a
        live graph produces wrong output with no error.

        The activation buffer never contributes to ``grew``, because it never
        grows — a step that needs more raises instead. A previous lease must be
        closed first: growth is only allowed at a point with no consumer.
        """
        with self._lock:
            if self._live_lease is not None and not self._live_lease.closed:
                raise RuntimeError(
                    "a workspace lease is still live; release it before preparing the next step"
                )
            by_owner: dict[MemoryOwner, list[WorkspaceRequest]] = {o: [] for o in self._arenas}
            seen: set[str] = set()
            for request in plan.workspaces:
                if request.name in seen:
                    raise ValueError(f"duplicate workspace name {request.name!r} in plan")
                seen.add(request.name)
                bucket = by_owner.get(request.owner)
                if bucket is None:
                    raise ValueError(
                        f"{request.name!r} is charged to {request.owner.value}, which the "
                        f"workspace manager does not back; it holds "
                        f"{sorted(o.value for o in self._arenas)}"
                    )
                bucket.append(request)

            grew = False
            regions: dict[str, MemoryRegion] = {}
            generation = 0
            for owner, requests in by_owner.items():
                arena = self._arenas[owner]
                grew |= arena.reserve(_required_bytes(tuple(requests)))
                arena.reset()
                for request in requests:
                    regions[request.name] = arena.allocate(request)
                if owner is MemoryOwner.WORKSPACE:
                    generation = arena.generation

            self._num_steps += 1
            used = sum(a.used_bytes for a in self._arenas.values())
            lease = WorkspaceLease(regions, generation, used, manager=self)
            self._live_lease = lease
            return lease, grew

    def _release(self, lease: WorkspaceLease) -> None:
        """Drop a closed lease so a later prepare may reset/grow the arenas."""
        with self._lock:
            if self._live_lease is lease:
                self._live_lease = None

    # ── introspection ────────────────────────────────────────────────────────

    def close(self) -> None:
        with self._lock:
            if self._live_lease is not None and not self._live_lease.closed:
                raise RuntimeError("cannot close the workspace while a lease is still live")
            for arena in self._arenas.values():
                arena.close()

    @property
    def capacity_bytes(self) -> int:
        """Total across both owners.  Per-owner figures come from :meth:`stats_for`."""
        return sum(a.capacity_bytes for a in self._arenas.values())

    @property
    def activation_capacity_bytes(self) -> int:
        return self._arenas[MemoryOwner.ACTIVATION].capacity_bytes

    @property
    def workspace_generation(self) -> int:
        """Arena generation of the workspace buffer; bumps on every move."""
        return self._arenas[MemoryOwner.WORKSPACE].generation

    @property
    def activation_generation(self) -> int:
        """Arena generation of the fixed activation buffer."""
        return self._arenas[MemoryOwner.ACTIVATION].generation

    def stats_for(self, owner: MemoryOwner) -> WorkspaceStats:
        with self._lock:
            arena = self._arenas.get(owner)
            if arena is None:
                raise KeyError(f"workspace manager does not back {owner.value}")
            return arena.stats(self._num_steps)

    def stats(self) -> WorkspaceStats:
        """The workspace arena's figures.

        Deliberately *not* the sum: this is what the caller watches to decide
        whether growth is thrashing, and adding a fixed-size activation buffer
        into it would make the high-water mark stop moving and hide exactly that.
        """
        return self.stats_for(MemoryOwner.WORKSPACE)

    def __repr__(self) -> str:
        ws = self._arenas[MemoryOwner.WORKSPACE]
        act = self._arenas[MemoryOwner.ACTIVATION]
        return (
            f"<WorkspaceManager workspace={ws.high_water_bytes >> 10}/"
            f"{ws.capacity_bytes >> 10} KiB activation={act.capacity_bytes >> 10} KiB "
            f"growths={ws.num_growths} steps={self._num_steps}>"
        )


def _required_bytes(requests: tuple[WorkspaceRequest, ...]) -> int:
    """Sum with worst-case alignment padding between buffers.

    Padding must be counted, not assumed away: eight buffers at 256-byte
    alignment can each waste 255 bytes, and sizing to the bare sum makes the
    arena fail on the step that needs it most.
    """
    total = 0
    for request in requests:
        total = align_up(total, request.alignment)
        total += request.nbytes
    return total


class _OwnedBuffer:
    """One backing buffer, one ledger owner, one growth policy.

    Extracted from ``WorkspaceManager``. The manager used to *be*
    this, singular, hardcoded to ``MemoryOwner.WORKSPACE``; making it a part
    rather than the whole is what let ACTIVATION get its own charge and its own
    rule without a second class that would drift.
    """

    __slots__ = (
        "_allocator",
        "_arena",
        "_capacity",
        "_ceiling",
        "_high_water",
        "_may_grow",
        "_num_growths",
        "_owner",
        "_region",
    )

    def __init__(
        self,
        allocator: CachingAllocator,
        owner: MemoryOwner,
        *,
        may_grow: bool,
        ceiling_bytes: int | None = None,
    ) -> None:
        if ceiling_bytes is not None and ceiling_bytes < 0:
            raise ValueError("ceiling_bytes must be non-negative")
        self._allocator = allocator
        self._owner = owner
        self._may_grow = may_grow
        self._ceiling = ceiling_bytes
        self._region: MemoryRegion | None = None
        self._arena: Arena | None = None
        self._capacity = 0
        self._high_water = 0
        self._num_growths = 0

    def reserve(self, required: int) -> bool:
        """Ensure capacity.  Returns whether the buffer moved."""
        if required <= self._capacity:
            return False
        if not self._may_grow:
            if not self._capacity:
                raise ActivationOverflowError(
                    f"a step asked for {required} bytes of {self._owner.value} but that "
                    "buffer was never sized; call WorkspaceManager.initialize() with the "
                    "profiled activation peak before the first step"
                )
            raise ActivationOverflowError(
                f"step needs {required} bytes of {self._owner.value} but the budget "
                f"fixed at initialize() is {self._capacity}. The KV page count was "
                "computed from that figure and cannot be revised now — the profiling "
                "step did not reach the real worst case."
            )
        if self._ceiling is not None and required > self._ceiling:
            raise WorkspaceCeilingError(
                f"step needs {required} bytes of {self._owner.value} but the frozen "
                f"ceiling is {self._ceiling}; growth is refused so the KV budget "
                "computed at bootstrap stays valid. Rebuild with a larger budget."
            )
        self.grow_to(required, cap=self._ceiling)
        return True

    def grow_to(self, required: int, *, floor: int = 0, cap: int | None = None) -> None:
        # Grow in 2 MiB units with 50% slack. The slack buys a few steps of
        # headroom, which matters because every growth invalidates graphs — and
        # re-capturing them costs far more than the wasted bytes.
        #
        # The non-growable buffer takes this path exactly once, from
        # initialize(), and passes floor=required so it gets what was profiled
        # plus rounding rather than 50% it will never use. ``cap`` is the frozen
        # ceiling: the allocation never exceeds it, even to satisfy rounding.
        target = max(required if floor else required + required // 2, floor, 2 << 20)
        target = align_up(target, 2 << 20)
        if cap is not None:
            target = max(required, min(target, cap))
        new_region = self._allocator.allocate(target, owner=self._owner)
        old = self._region
        self._region = new_region
        self._arena = Arena(new_region)
        self._capacity = target
        self._num_growths += 1
        # Free after allocating, not before: freeing first would let the new
        # allocation reuse the block and then there is a window where a captured
        # graph's pointer is valid but points at reinitialised memory.
        if old is not None:
            self._allocator.free(old)

    def reset(self) -> None:
        if self._arena is not None:
            self._arena.reset()

    def allocate(self, request: WorkspaceRequest) -> MemoryRegion:
        arena = self._arena
        if arena is None:
            raise ActivationOverflowError(
                f"{request.name!r} is charged to {self._owner.value} but that buffer was "
                "never sized; call WorkspaceManager.initialize() with the profiled "
                "activation peak before the first step"
            )
        region = arena.allocate(request.nbytes, alignment=request.alignment)
        self._high_water = max(self._high_water, arena.used_bytes)
        return region

    def close(self) -> None:
        if self._region is not None:
            self._allocator.free(self._region)
            self._region = None
            self._arena = None
            self._capacity = 0

    @property
    def capacity_bytes(self) -> int:
        return self._capacity

    @property
    def high_water_bytes(self) -> int:
        return self._high_water

    @property
    def slack_bytes(self) -> int:
        """Bytes the pool holds for this buffer beyond its usable capacity.

        The backing block is class-rounded (and may start after an alignment
        offset), so the ledger charges more than ``capacity_bytes``. This is the
        difference. It is what lets a pool-held figure and an arena capacity
        reconcile instead of reading like a leak: for the live arenas,
        ``allocator.stats().live_bytes == sum(capacity) + sum(slack)``.
        """
        if self._region is None:
            return 0
        return max(self._region.class_bytes - self._capacity, 0)

    @property
    def num_growths(self) -> int:
        return self._num_growths

    @property
    def used_bytes(self) -> int:
        return self._arena.used_bytes if self._arena else 0

    @property
    def generation(self) -> int:
        return self._arena.generation if self._arena else 0

    def stats(self, num_steps: int) -> WorkspaceStats:
        return WorkspaceStats(
            capacity_bytes=self._capacity,
            used_bytes=self.used_bytes,
            high_water_bytes=self._high_water,
            internal_slack_bytes=self.slack_bytes,
            num_growths=self._num_growths,
            num_steps=num_steps,
        )
