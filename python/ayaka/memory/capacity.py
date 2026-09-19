"""Immutable capacity snapshot, owner table and resource generation.

One runtime generation owns one :class:`CapacitySnapshot`: it records what the
process may hold (claims per owner), the frozen budgets taken at bootstrap, and
a :class:`ResourceGeneration` that identifies the concrete backing (model
revision, KV storage fingerprint, workspace buffers, backend). A rebuild (for
example a runtime KV resize) mints a new generation and replaces the snapshot;
nothing may grow the previous one in place.

The ledger stays the accounting authority. This module only defines the frozen
statement the runtime publishes after materialization, the tier mapping a claim
must use on each lane, and the pre-admission reconciliation against the ledger.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from ayaka.memory.ledger import LedgerSnapshot, MemoryLedger
from ayaka.types import MemoryOwner, MemoryTier
from ayaka.utils.validation import require_int, require_text

__all__ = [
    "CapacityFreeze",
    "CapacitySnapshot",
    "MemoryLane",
    "OwnerClaim",
    "ResourceGeneration",
    "build_capacity_snapshot",
    "claim_tier",
    "mint_generation",
    "reconcile_actual_usage",
]

#: Process-local owner incarnation counter. Never reused within a process, so
#: an old generation can never compare equal to a later one.
_OWNER_INCARNATIONS = itertools.count(1)

#: Ledger attribution aliases. The pooled workspace allocator registers every
#: arena under ``MemoryOwner.WORKSPACE`` (one ledger entry per pool), while the
#: ACTIVATION arena is an in-process owner. Reconciliation therefore compares
#: the combined WORKSPACE+ACTIVATION budget against the pooled entry instead of
#: demanding a separate ledger claim the pool cannot produce.
_LEDGER_OWNER_ALIASES: dict[MemoryOwner, MemoryOwner] = {
    MemoryOwner.ACTIVATION: MemoryOwner.WORKSPACE,
}


def ledger_owner(owner: MemoryOwner) -> MemoryOwner:
    """Ledger attribution for one capacity owner (see the alias table)."""
    return _LEDGER_OWNER_ALIASES.get(owner, owner)


class MemoryLane(StrEnum):
    """Physical lane a runtime generation runs on."""

    CPU = "cpu"
    CUDA = "cuda"


def claim_tier(
    owner: MemoryOwner,
    *,
    lane: MemoryLane,
    pinned_host: bool = False,
) -> MemoryTier:
    """Physical tier a claim of ``owner`` must be charged against on ``lane``.

    A CPU diagnostic run materializes every backing into host-pageable memory,
    so all claims land on ``HOST_PAGEABLE`` -- charging ``DEVICE`` there would
    split the full budget into two virtual accounts. On CUDA every non-pinned
    claim lands on ``DEVICE``; ``pinned_host`` marks staging buffers that
    physically live in page-locked host memory and therefore charge
    ``HOST_PINNED`` while remaining owned by ``WORKSPACE``.

    R07 note: ``WorkspaceRequest.tier`` defaults to ``DEVICE`` and must be
    remapped through this helper on the CPU lane instead of used directly.
    """
    if not isinstance(owner, MemoryOwner):
        raise TypeError("owner must be a MemoryOwner")
    if lane is MemoryLane.CPU or lane == MemoryLane.CPU:
        return MemoryTier.HOST_PAGEABLE
    if lane is not MemoryLane.CUDA and lane != MemoryLane.CUDA:
        raise ValueError(f"unknown memory lane {lane!r}")
    if pinned_host:
        if owner is not MemoryOwner.WORKSPACE:
            raise ValueError("only WORKSPACE-owned staging may charge HOST_PINNED")
        return MemoryTier.HOST_PINNED
    return MemoryTier.DEVICE


@dataclass(frozen=True, slots=True)
class ResourceGeneration:
    """Identity of one runtime resource set, compared component by component.

    ``owner_incarnation`` is process-local and monotonic across rebuilds. The
    other components name the model, the materialized KV storage, the workspace
    and buffer generations and the backend. Equality of the whole tuple -- not
    of a single counter -- is what validates that a captured resource still
    belongs to the live owner.
    """

    owner_incarnation: int
    model_id: str
    model_revision: str
    weights_revision: str
    kv_storage: tuple[str, ...]
    workspace: int
    buffers: int
    backend: str

    def __post_init__(self) -> None:
        require_int(self.owner_incarnation, "owner_incarnation", minimum=1)
        for name in ("model_id", "model_revision", "weights_revision", "backend"):
            require_text(getattr(self, name), name)
        if type(self.kv_storage) is not tuple or not self.kv_storage:
            raise TypeError("kv_storage must be a non-empty tuple of fingerprints")
        if len(set(self.kv_storage)) != len(self.kv_storage):
            raise ValueError("kv_storage fingerprints must be unique")
        for index, value in enumerate(self.kv_storage):
            require_text(value, f"kv_storage[{index}]")
        require_int(self.workspace, "workspace")
        require_int(self.buffers, "buffers")


@dataclass(frozen=True, slots=True)
class OwnerClaim:
    """One owner's frozen budget on its mapped tier."""

    owner: MemoryOwner
    tier: MemoryTier
    budget_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.owner, MemoryOwner):
            raise TypeError("owner must be a MemoryOwner")
        if not isinstance(self.tier, MemoryTier) or not self.tier.allocatable:
            raise ValueError("claim tier must be an allocatable MemoryTier")
        require_int(self.budget_bytes, "budget_bytes")


@dataclass(frozen=True, slots=True)
class CapacitySnapshot:
    """Frozen statement of everything one runtime generation may own.

    The snapshot is the only place bootstrap decisions (geometry, ceilings,
    flights, owner budgets) are published. Growth beyond it is refused before
    admission; a legitimate change (page capacity, layout) requires a rebuild
    that mints a new snapshot, never an in-place mutation.
    """

    generation: ResourceGeneration
    lane: MemoryLane
    dtype: str
    kv_dtype: str
    page_size: int
    group_pages: tuple[tuple[str, int], ...]
    max_model_len: int
    max_num_seqs: int
    max_num_batched_tokens: int
    max_inflight: int
    activation_bytes: int
    workspace_ceiling_bytes: int
    graph_bytes: int
    staging_bytes: int
    budget_bytes: int
    kv_budget_bytes: int
    weights_bytes: int
    owners: tuple[OwnerClaim, ...]
    ledger: LedgerSnapshot
    #: Persistent per-flight runner buffer footprint charged to the
    #: WORKSPACE/DEVICE claim in addition to the growable workspace ceiling.
    runner_buffer_bytes: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.generation, ResourceGeneration):
            raise TypeError("generation must be a ResourceGeneration")
        if not isinstance(self.lane, MemoryLane):
            raise TypeError("lane must be a MemoryLane")
        for name in ("dtype", "kv_dtype"):
            require_text(getattr(self, name), name)
        for name in (
            "page_size",
            "max_model_len",
            "max_num_seqs",
            "max_num_batched_tokens",
            "max_inflight",
        ):
            require_int(getattr(self, name), name, minimum=1)
        for name in (
            "activation_bytes",
            "workspace_ceiling_bytes",
            "graph_bytes",
            "staging_bytes",
            "budget_bytes",
            "kv_budget_bytes",
            "weights_bytes",
            "runner_buffer_bytes",
        ):
            require_int(getattr(self, name), name)
        if type(self.group_pages) is not tuple or not self.group_pages:
            raise TypeError("group_pages must be a non-empty tuple")
        if len({name for name, _ in self.group_pages}) != len(self.group_pages):
            raise ValueError("group_pages names must be unique")
        for name, pages in self.group_pages:
            require_text(name, "group name")
            require_int(pages, f"{name} pages", minimum=1)
        if not isinstance(self.ledger, LedgerSnapshot):
            raise TypeError("ledger must be a LedgerSnapshot")
        if not self.owners:
            raise ValueError("a capacity snapshot requires an owner table")
        seen: set[tuple[MemoryOwner, MemoryTier]] = set()
        for claim in self.owners:
            if not isinstance(claim, OwnerClaim):
                raise TypeError("owners must contain OwnerClaim values")
            key = (claim.owner, claim.tier)
            if key in seen:
                raise ValueError(f"duplicate {claim.owner.value}/{claim.tier.name} claim")
            seen.add(key)
            if claim.budget_bytes > self.budget_bytes:
                raise ValueError(
                    f"{claim.owner.value} claim exceeds the {self.budget_bytes}-byte budget"
                )
        expected = {
            MemoryOwner.WEIGHT: claim_tier(MemoryOwner.WEIGHT, lane=self.lane),
            MemoryOwner.KV: claim_tier(MemoryOwner.KV, lane=self.lane),
            MemoryOwner.ACTIVATION: claim_tier(MemoryOwner.ACTIVATION, lane=self.lane),
            MemoryOwner.WORKSPACE: claim_tier(MemoryOwner.WORKSPACE, lane=self.lane),
            MemoryOwner.COMPILE: claim_tier(MemoryOwner.COMPILE, lane=self.lane),
        }
        for owner, tier in expected.items():
            if (owner, tier) not in seen:
                raise ValueError(f"{owner.value} claim must be charged to {tier.name} on this lane")
        if self.staging_bytes and self.lane is MemoryLane.CUDA:
            if (MemoryOwner.WORKSPACE, MemoryTier.HOST_PINNED) not in seen:
                raise ValueError("staging bytes require a WORKSPACE/HOST_PINNED claim on CUDA")

    @property
    def pages(self) -> int:
        return sum(pages for _, pages in self.group_pages)

    def owner_budget(self, owner: MemoryOwner) -> int:
        total = 0
        found = False
        for claim in self.owners:
            if claim.owner is owner:
                total += claim.budget_bytes
                found = True
        if not found:
            raise KeyError(f"no frozen budget for {owner.value}")
        return total

    def tier_budgets(self) -> dict[MemoryTier, int]:
        totals: dict[MemoryTier, int] = {}
        for claim in self.owners:
            totals[claim.tier] = totals.get(claim.tier, 0) + claim.budget_bytes
        return totals


def mint_generation(
    *,
    model_id: str,
    model_revision: str,
    weights_revision: str,
    kv_storage: Sequence[str],
    backend: str,
    workspace: int = 0,
    buffers: int = 0,
) -> ResourceGeneration:
    """Mint a fresh process-local owner incarnation for one runtime generation."""
    return ResourceGeneration(
        owner_incarnation=next(_OWNER_INCARNATIONS),
        model_id=model_id,
        model_revision=model_revision,
        weights_revision=weights_revision,
        kv_storage=tuple(kv_storage),
        workspace=workspace,
        buffers=buffers,
        backend=backend,
    )


def build_capacity_snapshot(
    *,
    generation: ResourceGeneration,
    lane: MemoryLane,
    dtype: str,
    kv_dtype: str,
    page_size: int,
    group_pages: Mapping[str, int],
    max_model_len: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    max_inflight: int,
    activation_bytes: int,
    workspace_ceiling_bytes: int,
    graph_bytes: int,
    staging_bytes: int,
    budget_bytes: int,
    kv_budget_bytes: int,
    weights_bytes: int,
    ledger: MemoryLedger,
    staging_pinned: bool = True,
    runner_buffer_bytes: int = 0,
) -> CapacitySnapshot:
    """Freeze one generation's owner table from the resolved budgets.

    The owner table maps every budget to the physical tier it must be charged
    against on ``lane`` (see :func:`claim_tier`). ``staging_bytes`` is a
    WORKSPACE-owned HOST_PINNED claim and may be zero. ``runner_buffer_bytes``
    widens the WORKSPACE claim by the persistent per-flight metadata footprint.
    """
    owners = [
        OwnerClaim(MemoryOwner.WEIGHT, claim_tier(MemoryOwner.WEIGHT, lane=lane), weights_bytes),
        OwnerClaim(MemoryOwner.KV, claim_tier(MemoryOwner.KV, lane=lane), kv_budget_bytes),
        OwnerClaim(
            MemoryOwner.ACTIVATION,
            claim_tier(MemoryOwner.ACTIVATION, lane=lane),
            activation_bytes,
        ),
        OwnerClaim(
            MemoryOwner.WORKSPACE,
            claim_tier(MemoryOwner.WORKSPACE, lane=lane),
            workspace_ceiling_bytes + runner_buffer_bytes,
        ),
        OwnerClaim(MemoryOwner.COMPILE, claim_tier(MemoryOwner.COMPILE, lane=lane), graph_bytes),
    ]
    if staging_bytes:
        owners.append(
            OwnerClaim(
                MemoryOwner.WORKSPACE,
                claim_tier(MemoryOwner.WORKSPACE, lane=lane, pinned_host=staging_pinned),
                staging_bytes,
            )
        )
    # On the CPU lane staging and workspace share HOST_PAGEABLE; merge so the
    # table keeps one claim per owner/tier pair instead of double-counting.
    merged: dict[tuple[MemoryOwner, MemoryTier], int] = {}
    for claim in owners:
        key = (claim.owner, claim.tier)
        merged[key] = merged.get(key, 0) + claim.budget_bytes
    return CapacitySnapshot(
        generation=generation,
        lane=lane,
        dtype=dtype,
        kv_dtype=kv_dtype,
        page_size=page_size,
        group_pages=tuple(sorted(group_pages.items())),
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_inflight=max_inflight,
        activation_bytes=activation_bytes,
        workspace_ceiling_bytes=workspace_ceiling_bytes,
        graph_bytes=graph_bytes,
        staging_bytes=staging_bytes,
        budget_bytes=budget_bytes,
        kv_budget_bytes=kv_budget_bytes,
        weights_bytes=weights_bytes,
        owners=tuple(
            OwnerClaim(owner, tier, budget)
            for (owner, tier), budget in sorted(merged.items(), key=lambda item: item[0])
        ),
        ledger=ledger.snapshot(),
        runner_buffer_bytes=runner_buffer_bytes,
    )


def reconcile_actual_usage(ledger: MemoryLedger, snapshot: CapacitySnapshot) -> None:
    """Fail before admission when the ledger disagrees with the frozen snapshot.

    Checks the capacity invariant, the lane rule that a CPU run never charges
    ``DEVICE``, and that no owner or tier holds more than the snapshot declared.
    This is the "actual usage" step between materialization and admission, not a
    device-level probe: it compares claims against the frozen budget only.
    """
    ledger.assert_within_capacity()
    if snapshot.lane is MemoryLane.CPU and ledger.committed(MemoryTier.DEVICE) > 0:
        raise MemoryError("CPU lane charged the DEVICE tier; every claim must be host-pageable")
    held = ledger.by_owner()
    declared: dict[MemoryOwner, int] = {}
    for claim in snapshot.owners:
        owner = ledger_owner(claim.owner)
        declared[owner] = declared.get(owner, 0) + claim.budget_bytes
    for owner, budget in declared.items():
        charged = held.get(owner, 0)
        if charged > budget:
            raise MemoryError(
                f"{owner.value} holds {charged} bytes against its frozen {budget}-byte budget"
            )
    tier_budgets = snapshot.tier_budgets()
    for tier in ledger.tiers:
        committed = ledger.committed(tier)
        if committed > tier_budgets.get(tier, 0):
            raise MemoryError(
                f"{tier.name} holds {committed} bytes but the snapshot declares "
                f"{tier_budgets.get(tier, 0)} bytes"
            )


class CapacityFreeze:
    """Owns the single frozen snapshot of one runtime generation.

    ``replace`` is the rebuild path only: the new generation must carry a
    strictly higher ``owner_incarnation``, so a stale snapshot can never be
    reinstated and no caller can silently grow the frozen budgets in place.
    """

    __slots__ = ("_snapshot",)

    def __init__(self, snapshot: CapacitySnapshot) -> None:
        if not isinstance(snapshot, CapacitySnapshot):
            raise TypeError("snapshot must be a CapacitySnapshot")
        self._snapshot = snapshot

    @property
    def current(self) -> CapacitySnapshot:
        return self._snapshot

    def replace(self, snapshot: CapacitySnapshot) -> None:
        if snapshot.generation.owner_incarnation <= self._snapshot.generation.owner_incarnation:
            raise ValueError("a replacement generation must advance owner_incarnation")
        self._snapshot = snapshot

    def assert_frozen(self) -> CapacitySnapshot:
        """Return the current snapshot; the freeze itself never mutates."""
        return self._snapshot
