"""Memory ledger

The ledger **records**; it does not allocate. That split is the point: an
allocator's job is to be fast, a ledger's job is to be true, and a component
that tries to be both ends up either slow or lying.

**One authority, four accounts.** Not one counter — DEVICE, HOST_PINNED,
HOST_PAGEABLE and DISK each carry their own capacity and totals, because they
are exhausted independently. A single number cannot express "the device is full
but the host tier has room", which is the entire premise of tiering.

**Transactional, for the same reason the KV manager is.** ``reserve`` charges
capacity without claiming that bytes exist, ``materialize`` records the actual
allocation, ``commit`` publishes it, and ``rollback`` undoes the operation. A
tier migration needs every stage: the destination is charged before allocation,
both physical copies are counted while the copy is in flight, and the source is
released only after completion.

Three things it exists to prevent:

1. **Two answers to "how much is left."** Every subsystem that takes memory
   registers here, so admission, the capacity planner and the metrics endpoint
   read one number per tier.
2. **Double counting against torch's caching pool.** Torch reserves in large
   chunks and reuses inside them, so ``memory_allocated`` (live tensors) and
   ``memory_reserved`` (held from the driver) differ by hundreds of MB under
   load. Summing individual tensors would count bytes torch already counts. The
   pool is therefore ONE entry whose size is *restated* -- see :meth:`update`.
3. **Drift.** A ledger nobody checks becomes a tidy dict that diverges from
   reality. :meth:`reconcile` compares a tier's total against what the system
   reports and fails loudly; without it the whole structure is decoration.

Honesty about what is verifiable: per-owner attribution is *intent*. Weights, KV
slabs and activations all come out of the same torch pool today, so the driver
cannot confirm the split -- only the total. The per-owner numbers answer "who
asked for what", which is what capacity planning and postmortems need; the
aggregate is what :meth:`reconcile` proves.

Reserved VA, admission charge and materialized bytes are separate from the
start. Alignment can already make charge differ from tensor payload; they
diverge further when KV moves to ``cuMemAddressReserve`` (A6 elastic KV): a huge
VA range is reserved once and physical pages are mapped and unmapped underneath
it. Adding all three columns now is free; retrofitting them later is not.
"""

from __future__ import annotations

import itertools
import threading
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Protocol

from ayaka.memory.region import MemoryRegion
from ayaka.types import MemoryOwner, MemoryTier

__all__ = [
    "DEFAULT_TOLERANCE_BYTES",
    "LedgerDrift",
    "LedgerSnapshot",
    "MemoryLedger",
    "Reservation",
    "Ticket",
    "TierAccount",
    "TierAccountSnapshot",
]

#: A CUDA context, the driver's own bookkeeping, cuBLAS/cuDNN handles and NCCL's
#: internal buffers together account for a few hundred MB that no ledger entry
#: describes. 256 MiB is loose enough not to false-positive on a fresh context
#: and tight enough that a genuinely lost gigabyte still trips.
DEFAULT_TOLERANCE_BYTES = 256 << 20

_TelemetryValue = str | int | float | bool | None
_MetricLabelValue = str | int | float | bool


class _DecisionSink(Protocol):
    def record(
        self,
        *,
        component: str,
        action: str,
        outcome: str,
        reason: str,
        request_id: str | None = None,
        attributes: Mapping[str, _TelemetryValue] | None = None,
    ) -> object: ...


class _MetricSink(Protocol):
    def set_gauge(
        self,
        name: str,
        value: float,
        *,
        labels: Mapping[str, _MetricLabelValue] | None = None,
    ) -> None: ...


class LedgerDrift(RuntimeError):
    """A tier's total no longer matches what the system reports."""


@dataclass(frozen=True, slots=True)
class TierAccount:
    """One tier's capacity.

    ``total_bytes`` is the physical size behind the tier and is used only by
    :meth:`MemoryLedger.reconcile` -- for DEVICE that is HBM, for HOST_PINNED it
    is the machine's RAM, and the gap between it and ``capacity_bytes`` is the
    safety margin the host policy carved out.
    """

    tier: MemoryTier
    capacity_bytes: int
    total_bytes: int = 0

    def __post_init__(self) -> None:
        if not self.tier.allocatable:
            raise ValueError(
                f"{self.tier.name} is not allocatable; only DEVICE, HOST_PINNED, "
                "HOST_PAGEABLE and DISK carry accounts"
            )
        if self.capacity_bytes < 0:
            raise ValueError(f"{self.tier.name}: capacity must be non-negative")
        if self.total_bytes and self.total_bytes < self.capacity_bytes:
            raise ValueError(
                f"{self.tier.name}: capacity {self.capacity_bytes} exceeds the "
                f"tier's physical size {self.total_bytes}"
            )


@dataclass(frozen=True, slots=True)
class Reservation:
    """One registered claim.

    ``label`` is the identity: unique across the ledger, stable, and descriptive
    enough to be useful in a postmortem (``"kv.layer_12.k"``,
    ``"torch_caching_pool"``), since that is the string an operator sees when the
    ledger says it ran out.

    ``reserved_bytes`` is address space, ``charged_bytes`` is admission capacity,
    and ``backed_bytes`` is physical memory known to exist. For an ordinary
    committed allocation all three are equal. They differ while a reservation is
    pending and once VMM reserves a large VA range with fewer mapped pages.
    """

    owner: MemoryOwner
    label: str
    reserved_bytes: int
    backed_bytes: int
    tier: MemoryTier = MemoryTier.DEVICE
    device_index: int = 0
    region: MemoryRegion | None = None
    charged_bytes: int | None = None

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("a reservation must carry a label")
        charged = self.backed_bytes if self.charged_bytes is None else self.charged_bytes
        object.__setattr__(self, "charged_bytes", charged)
        if self.reserved_bytes < 0 or self.backed_bytes < 0 or charged < 0:
            raise ValueError(f"{self.label}: byte counts must be non-negative")
        if self.backed_bytes > self.reserved_bytes:
            raise ValueError(
                f"{self.label}: backed_bytes={self.backed_bytes} exceeds "
                f"reserved_bytes={self.reserved_bytes} -- physical pages cannot "
                "exceed the address space mapping them"
            )
        if charged > self.reserved_bytes:
            raise ValueError(
                f"{self.label}: charged_bytes={charged} exceeds "
                f"reserved_bytes={self.reserved_bytes}"
            )
        if not self.tier.allocatable:
            raise ValueError(f"{self.label}: {self.tier.name} is not an allocatable tier")
        if self.region is not None:
            if self.region.owner is not self.owner:
                raise ValueError(
                    f"{self.label}: region owner {self.region.owner} disagrees with "
                    f"reservation owner {self.owner}"
                )
            if self.region.tier is not self.tier:
                raise ValueError(
                    f"{self.label}: region tier {self.region.tier.name} disagrees with "
                    f"reservation tier {self.tier.name}"
                )
            if self.region.device_index >= 0 and self.region.device_index != self.device_index:
                raise ValueError(
                    f"{self.label}: region device {self.region.device_index} disagrees with "
                    f"reservation device {self.device_index}"
                )
            if self.backed_bytes > self.region.nbytes:
                raise ValueError(
                    f"{self.label}: {self.backed_bytes} backed bytes exceed region size "
                    f"{self.region.nbytes}"
                )

    @property
    def materialized_bytes(self) -> int:
        """Physical bytes known to exist.

        ``backed_bytes`` is retained as the storage/VMM term used by the
        frozen memory-region contract. Ledger callers should use this name so
        it cannot be confused with admission capacity (``charged_bytes``).
        """
        return self.backed_bytes

    @property
    def capacity_charge_bytes(self) -> int:
        """Bytes held against admission capacity after normalization."""
        charged = self.charged_bytes
        if charged is None:  # Defensive; __post_init__ always normalizes it.
            raise AssertionError("reservation charge was not normalized")
        return charged

    @classmethod
    def backed(
        cls,
        owner: MemoryOwner,
        label: str,
        nbytes: int,
        *,
        tier: MemoryTier = MemoryTier.DEVICE,
        device_index: int = 0,
        region: MemoryRegion | None = None,
    ) -> Reservation:
        """The ordinary case: address space and physical pages are the same."""
        return cls(
            owner=owner,
            label=label,
            reserved_bytes=nbytes,
            backed_bytes=nbytes,
            tier=tier,
            device_index=device_index,
            region=region,
        )

    @classmethod
    def from_region(cls, region: MemoryRegion, label: str) -> Reservation:
        return cls(
            owner=region.owner,
            label=label,
            reserved_bytes=region.nbytes,
            backed_bytes=region.nbytes if region.backed else 0,
            tier=region.tier,
            device_index=region.device_index if region.device_index >= 0 else 0,
            region=region,
        )


@dataclass(frozen=True, slots=True)
class Ticket:
    """Handle on an uncommitted operation.

    ``from_tier`` is set only for a transfer, and it is what makes the migration
    window legible: while the ticket is open the bytes are committed on
    ``from_tier`` *and* pending on ``tier``, which is the truth -- during the copy
    both buffers exist.
    """

    ticket_id: int
    label: str
    tier: MemoryTier
    from_tier: MemoryTier | None = None

    @property
    def is_transfer(self) -> bool:
        return self.from_tier is not None


@dataclass(slots=True)
class _PendingOp:
    """Internal transaction state; a public ticket is only an opaque key."""

    ticket: Ticket
    destination: Reservation
    source_label: str | None = None
    source_snapshot: Reservation | None = None
    materialized: bool = False


@dataclass(frozen=True, slots=True)
class TierAccountSnapshot:
    """Byte accounting for one :class:`TierAccount` -- one allocatable tier.

    One of these exists per :class:`~ayaka.types.MemoryTier` the ledger carries
    (DEVICE, HOST_PINNED, HOST_PAGEABLE, DISK); :class:`LedgerSnapshot` holds
    them keyed by tier. Every field is a byte count.

    Not to be confused with
    :class:`~ayaka.memory.tiering.HostTierSnapshot`, which counts *pages* in
    the KV host-tier state machine rather than bytes against a capacity.
    """

    tier: MemoryTier
    capacity_bytes: int
    committed_bytes: int
    pending_bytes: int
    """Charged but not committed -- capacity is taken, nothing is published."""
    committed_materialized_bytes: int
    pending_materialized_bytes: int
    reserved_va_bytes: int
    num_entries: int

    @property
    def held_bytes(self) -> int:
        """What the tier owes: committed plus in-flight reservations."""
        return self.committed_bytes + self.pending_bytes

    @property
    def free_bytes(self) -> int:
        return max(self.capacity_bytes - self.held_bytes, 0)

    @property
    def materialized_bytes(self) -> int:
        return self.committed_materialized_bytes + self.pending_materialized_bytes

    @property
    def utilization(self) -> float:
        return self.held_bytes / self.capacity_bytes if self.capacity_bytes else 0.0


@dataclass(frozen=True, slots=True)
class LedgerSnapshot:
    device_index: int
    tiers: dict[MemoryTier, TierAccountSnapshot] = field(default_factory=dict)
    charged_by_owner: dict[MemoryOwner, int] = field(default_factory=dict)
    pending_charged_by_owner: dict[MemoryOwner, int] = field(default_factory=dict)
    materialized_by_owner: dict[MemoryOwner, int] = field(default_factory=dict)
    num_open_tickets: int = 0

    @property
    def committed_bytes(self) -> int:
        return sum(t.committed_bytes for t in self.tiers.values())

    def largest_owner(self) -> MemoryOwner | None:
        if not self.charged_by_owner:
            return None
        return max(self.charged_by_owner, key=lambda o: self.charged_by_owner[o])

    @property
    def backed_by_owner(self) -> dict[MemoryOwner, int]:
        """Compatibility view for physical, materialized ownership."""
        return self.materialized_by_owner

    @property
    def pending_by_owner(self) -> dict[MemoryOwner, int]:
        """Compatibility view for pending admission charges."""
        return self.pending_charged_by_owner

    @property
    def held_by_owner(self) -> dict[MemoryOwner, int]:
        """Committed plus pending admission charges, grouped by owner."""
        owners = set(self.charged_by_owner) | set(self.pending_charged_by_owner)
        return {
            owner: self.charged_by_owner.get(owner, 0) + self.pending_charged_by_owner.get(owner, 0)
            for owner in owners
        }


class MemoryLedger:
    """One accounting authority with one account per allocatable tier."""

    __slots__ = (
        "_accounts",
        "_claimed_labels",
        "_device_index",
        "_entries",
        "_ids",
        "_lock",
        "_locked_sources",
        "_pending",
        "_trace",
    )

    def __init__(
        self,
        accounts: Sequence[TierAccount],
        *,
        device_index: int = 0,
        trace: _DecisionSink | None = None,
    ) -> None:
        if not accounts:
            raise ValueError("a ledger needs at least one tier account")
        by_tier: dict[MemoryTier, TierAccount] = {}
        for account in accounts:
            if account.tier in by_tier:
                raise ValueError(f"duplicate account for {account.tier.name}")
            by_tier[account.tier] = account
        self._accounts = by_tier
        self._device_index = device_index
        self._entries: dict[str, Reservation] = {}
        self._pending: dict[int, _PendingOp] = {}
        self._claimed_labels: dict[str, int] = {}
        self._locked_sources: dict[str, int] = {}
        self._trace = trace
        self._ids = itertools.count(1)
        # Deliberately non-reentrant: an RLock would let a write path call a read
        # method and silently succeed, which is how double counting gets in.
        self._lock = threading.Lock()

    @classmethod
    def for_device(
        cls,
        *,
        device_budget_bytes: int,
        device_total_bytes: int,
        host_pinned_bytes: int = 0,
        host_pageable_bytes: int = 0,
        disk_bytes: int = 0,
        host_total_bytes: int = 0,
        device_index: int = 0,
        trace: _DecisionSink | None = None,
    ) -> MemoryLedger:
        """The ordinary construction: one device plus whichever host tiers are on."""
        accounts = [TierAccount(MemoryTier.DEVICE, device_budget_bytes, device_total_bytes)]
        if host_pinned_bytes:
            accounts.append(
                TierAccount(MemoryTier.HOST_PINNED, host_pinned_bytes, host_total_bytes)
            )
        if host_pageable_bytes:
            accounts.append(
                TierAccount(MemoryTier.HOST_PAGEABLE, host_pageable_bytes, host_total_bytes)
            )
        if disk_bytes:
            accounts.append(TierAccount(MemoryTier.DISK, disk_bytes, 0))
        return cls(accounts, device_index=device_index, trace=trace)

    # ── reads ────────────────────────────────────────────────────────────────

    @property
    def device_index(self) -> int:
        return self._device_index

    @property
    def tiers(self) -> tuple[MemoryTier, ...]:
        return tuple(sorted(self._accounts))

    def has_tier(self, tier: MemoryTier) -> bool:
        return tier in self._accounts

    def capacity(self, tier: MemoryTier) -> int:
        return self._account(tier).capacity_bytes

    def committed(self, tier: MemoryTier) -> int:
        with self._lock:
            return self._committed_locked(tier)

    def pending(self, tier: MemoryTier) -> int:
        with self._lock:
            return self._pending_locked(tier)

    def materialized(self, tier: MemoryTier) -> int:
        """Physical bytes known to exist, committed plus pending."""
        with self._lock:
            return self._committed_materialized_locked(tier) + self._pending_materialized_locked(
                tier
            )

    def held(self, tier: MemoryTier) -> int:
        with self._lock:
            return self._committed_locked(tier) + self._pending_locked(tier)

    def free(self, tier: MemoryTier) -> int:
        with self._lock:
            return max(
                self._account(tier).capacity_bytes
                - self._committed_locked(tier)
                - self._pending_locked(tier),
                0,
            )

    def by_owner(self, tier: MemoryTier | None = None) -> dict[MemoryOwner, int]:
        with self._lock:
            return self._by_owner_locked(tier)

    def pending_by_owner(self, tier: MemoryTier | None = None) -> dict[MemoryOwner, int]:
        with self._lock:
            return self._pending_by_owner_locked(tier)

    def materialized_by_owner(self, tier: MemoryTier | None = None) -> dict[MemoryOwner, int]:
        with self._lock:
            return self._materialized_by_owner_locked(tier)

    def get(self, label: str) -> Reservation | None:
        with self._lock:
            return self._entries.get(label)

    def __contains__(self, label: object) -> bool:
        with self._lock:
            return label in self._entries

    def __iter__(self) -> Iterator[Reservation]:
        with self._lock:
            return iter(tuple(self._entries.values()))

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    # ── the five verbs ───────────────────────────────────────────────────────

    def reserve(self, reservation: Reservation) -> Ticket:
        """Take capacity without publishing. Returns a ticket for commit/rollback.

        The label is claimed here, not at commit: two callers reserving the same
        label concurrently must not both succeed and find out at commit, when one
        of them has already allocated.
        """
        try:
            with self._lock:
                self._require_account(reservation.tier)
                self._check_capacity_locked(reservation)
                ticket = Ticket(next(self._ids), reservation.label, reservation.tier)
                self._claim_label_locked(reservation.label, ticket.ticket_id)
                destination = replace(reservation, backed_bytes=0, region=None)
                self._pending[ticket.ticket_id] = _PendingOp(ticket, destination)
        except Exception as exc:
            self._record_decision(
                "reserve",
                "rejected",
                str(exc),
                label=reservation.label,
                tier=reservation.tier.name,
                charged_bytes=reservation.capacity_charge_bytes,
            )
            raise
        self._record_decision(
            "reserve",
            "accepted",
            "capacity and label claimed",
            ticket_id=ticket.ticket_id,
            label=reservation.label,
            tier=reservation.tier.name,
            charged_bytes=reservation.capacity_charge_bytes,
        )
        return ticket

    def materialize(
        self,
        ticket: Ticket,
        *,
        actual_bytes: int,
        region: MemoryRegion | None = None,
    ) -> None:
        """Record that a pending destination now physically exists.

        Capacity was charged by :meth:`reserve` or :meth:`transfer` before the
        allocator ran. Reconciliation must not count those bytes until this
        method succeeds.
        """
        if actual_bytes < 0:
            raise ValueError("actual_bytes must be non-negative")
        with self._lock:
            op = self._require_ticket_locked(ticket)
            if op.materialized:
                raise RuntimeError(f"ticket {ticket.ticket_id} is already materialized")
            charged = max(op.destination.capacity_charge_bytes, actual_bytes)
            updated = replace(
                op.destination,
                backed_bytes=actual_bytes,
                charged_bytes=charged,
                region=region,
            )
            self._check_capacity_locked(updated, excluding_pending=ticket.ticket_id)
            op.destination = updated
            op.materialized = True
        self._record_decision(
            "materialize",
            "succeeded",
            "physical allocation recorded",
            ticket_id=ticket.ticket_id,
            label=ticket.label,
            tier=ticket.tier.name,
            materialized_bytes=actual_bytes,
        )

    def commit(self, ticket: Ticket) -> None:
        """Publish a reservation, or complete a transfer.

        For a transfer this is the moment the copy is known to have landed: the
        destination becomes committed and the source is dropped in one step, so
        there is never an instant when neither tier owns the bytes.
        """
        with self._lock:
            op = self._require_ticket_locked(ticket)
            if not op.materialized:
                raise RuntimeError(
                    f"ticket {ticket.ticket_id} has reserved capacity but no materialized bytes"
                )
            if op.source_label is not None:
                current = self._entries.get(op.source_label)
                if current != op.source_snapshot:
                    raise RuntimeError(
                        f"transfer source {op.source_label!r} changed while ticket "
                        f"{ticket.ticket_id} was open"
                    )
                del self._entries[op.source_label]
            self._entries[op.destination.label] = op.destination
            self._finish_ticket_locked(op)
        self._record_decision(
            "commit",
            "succeeded",
            "transaction published",
            ticket_id=ticket.ticket_id,
            label=op.destination.label,
            tier=op.destination.tier.name,
            transfer=op.source_label is not None,
        )

    def rollback(self, ticket: Ticket) -> None:
        """Drop a reservation that was never used.

        For a transfer this leaves the source exactly as it was -- a failed copy
        must not lose the only copy.
        """
        with self._lock:
            op = self._require_ticket_locked(ticket)
            self._finish_ticket_locked(op)
        self._record_decision(
            "rollback",
            "succeeded",
            "pending charge and label claim released",
            ticket_id=ticket.ticket_id,
            label=op.destination.label,
            tier=op.destination.tier.name,
            source_preserved=op.source_label is not None,
        )

    def release(self, label: str) -> Reservation | None:
        with self._lock:
            self._require_unlocked_source_locked(label)
            return self._entries.pop(label, None)

    def transfer(self, label: str, to_tier: MemoryTier, *, new_label: str | None = None) -> Ticket:
        """Begin a tier migration. Both tiers hold the bytes until commit.

        That double-booking is not slack in the accounting, it is the accounting
        being correct: an in-flight DEVICE -> HOST_PINNED copy has a live buffer
        on each side, and a ledger that charged only one of them would let a
        concurrent admission oversubscribe the destination.
        """
        try:
            with self._lock:
                current = self._entries.get(label)
                if current is None:
                    raise KeyError(f"no reservation labelled {label!r} to transfer")
                if current.tier is to_tier:
                    raise ValueError(f"{label!r} is already on {to_tier.name}")
                self._require_unlocked_source_locked(label)
                self._require_account(to_tier)
                destination_label = new_label or label
                destination = replace(
                    current,
                    tier=to_tier,
                    label=destination_label,
                    backed_bytes=0,
                    region=None,
                )
                self._check_capacity_locked(destination)
                ticket = Ticket(next(self._ids), label, to_tier, from_tier=current.tier)
                self._claim_label_locked(
                    destination_label,
                    ticket.ticket_id,
                    source_label=label,
                )
                self._locked_sources[label] = ticket.ticket_id
                self._pending[ticket.ticket_id] = _PendingOp(
                    ticket=ticket,
                    destination=destination,
                    source_label=label,
                    source_snapshot=current,
                )
        except Exception as exc:
            self._record_decision(
                "transfer",
                "rejected",
                str(exc),
                label=label,
                destination_label=new_label or label,
                to_tier=to_tier.name,
            )
            raise
        self._record_decision(
            "transfer",
            "accepted",
            "source locked and destination capacity charged",
            ticket_id=ticket.ticket_id,
            label=label,
            destination_label=destination_label,
            from_tier=current.tier.name,
            to_tier=to_tier.name,
        )
        return ticket

    def admit(self, reservation: Reservation) -> None:
        """Reserve and commit in one step.

        For allocations where nothing can fail between the two -- the bytes are
        already in hand and the entry is pure bookkeeping. Anything that can fail
        between deciding and having should use reserve/commit instead.
        """
        ticket = self.reserve(reservation)
        try:
            self.materialize(
                ticket,
                actual_bytes=reservation.materialized_bytes,
                region=reservation.region,
            )
            self.commit(ticket)
        except BaseException:
            with self._lock:
                still_open = ticket.ticket_id in self._pending
            if still_open:
                self.rollback(ticket)
            raise

    def update(
        self,
        label: str,
        *,
        backed_bytes: int,
        reserved_bytes: int | None = None,
        charged_bytes: int | None = None,
    ) -> None:
        """Restate a committed reservation's size.

        This is how the torch caching pool is tracked: one entry whose backed
        size follows ``torch.cuda.memory_reserved()``, restated after each step.
        Re-reserving would either duplicate the label or, worse, add the pool's
        size to itself every step.
        """
        with self._lock:
            self._require_unlocked_source_locked(label)
            current = self._entries.get(label)
            if current is None:
                raise KeyError(f"no reservation labelled {label!r}")
            reserved = reserved_bytes if reserved_bytes is not None else backed_bytes
            charged = charged_bytes if charged_bytes is not None else backed_bytes
            updated = replace(
                current,
                backed_bytes=backed_bytes,
                charged_bytes=charged,
                reserved_bytes=reserved,
            )
            self._check_capacity_locked(updated, excluding=label)
            self._entries[label] = updated

    def release_owner(self, owner: MemoryOwner, *, tier: MemoryTier | None = None) -> int:
        """Drop every committed reservation of one owner; returns bytes released.

        Used at teardown, in the reverse of bootstrap order.
        """
        with self._lock:
            pending = [
                op.ticket.ticket_id
                for op in self._pending.values()
                if op.destination.owner is owner and (tier is None or op.destination.tier is tier)
            ]
            if pending:
                raise RuntimeError(f"cannot release owner {owner.value}: open tickets {pending}")
            labels = [
                lbl
                for lbl, r in self._entries.items()
                if r.owner is owner and (tier is None or r.tier is tier)
            ]
            for label in labels:
                self._require_unlocked_source_locked(label)
            freed = sum(self._entries[lbl].capacity_charge_bytes for lbl in labels)
            for label in labels:
                del self._entries[label]
            return freed

    # ── the invariant that keeps this honest ─────────────────────────────────

    def reconcile(
        self,
        tier: MemoryTier,
        observed_free_bytes: int,
        *,
        tolerance_bytes: int = DEFAULT_TOLERANCE_BYTES,
    ) -> int:
        """Compare one tier against what the system reports. Returns the drift.

        ``drift = observed_held - ledger_held``, where ``observed_held`` is the
        tier's physical size minus what the system says is free. Positive drift
        means something holds memory nobody registered -- the dangerous
        direction, because the capacity planner believes those bytes are
        available. The usual causes are a library allocating behind our back
        (cuBLAS workspaces, NCCL buffers) and an allocation path that forgot to
        register.

        Call at the end of bootstrap, and periodically to catch a slow leak.
        """
        account = self._account(tier)
        if not account.total_bytes:
            raise ValueError(
                f"cannot reconcile {tier.name}: the account carries no total_bytes, "
                "so there is nothing to compare the observation against"
            )
        if not 0 <= observed_free_bytes <= account.total_bytes:
            raise ValueError(
                f"{tier.name} observed_free_bytes={observed_free_bytes} must be between "
                f"zero and total_bytes={account.total_bytes}"
            )
        if tolerance_bytes < 0:
            raise ValueError("tolerance_bytes must be non-negative")
        observed_held = account.total_bytes - observed_free_bytes
        with self._lock:
            ledger_held = self._committed_materialized_locked(
                tier
            ) + self._pending_materialized_locked(tier)
            owners = self._materialized_by_owner_locked(tier)
        drift = observed_held - ledger_held
        if abs(drift) > tolerance_bytes:
            direction = (
                "system holds bytes the ledger does not know about"
                if drift > 0
                else "ledger claims bytes the system reports free"
            )
            error = LedgerDrift(
                f"{tier.name} drift {drift / (1 << 20):.1f} MiB ({direction}); "
                f"observed {observed_held >> 20} MiB, ledger {ledger_held >> 20} MiB, "
                f"tolerance {tolerance_bytes >> 20} MiB. "
                f"By owner: {self._format_owners(owners)}"
            )
            self._record_decision(
                "reconcile",
                "drift",
                str(error),
                tier=tier.name,
                drift_bytes=drift,
                materialized_bytes=ledger_held,
                observed_held_bytes=observed_held,
            )
            raise error
        self._record_decision(
            "reconcile",
            "matched",
            "observed and materialized bytes are within tolerance",
            tier=tier.name,
            drift_bytes=drift,
            materialized_bytes=ledger_held,
            observed_held_bytes=observed_held,
        )
        return drift

    def assert_within_capacity(self) -> None:
        with self._lock:
            for tier, account in self._accounts.items():
                held = self._committed_locked(tier) + self._pending_locked(tier)
                if held > account.capacity_bytes:
                    raise MemoryError(
                        f"{tier.name} over capacity: {held >> 20} MiB held against "
                        f"{account.capacity_bytes >> 20} MiB. "
                        f"By owner: {self._format_owners(self._by_owner_locked(tier))}"
                    )

    # ── views ────────────────────────────────────────────────────────────────

    def snapshot(self) -> LedgerSnapshot:
        with self._lock:
            tiers: dict[MemoryTier, TierAccountSnapshot] = {}
            for tier, account in self._accounts.items():
                entries = [r for r in self._entries.values() if r.tier is tier]
                tiers[tier] = TierAccountSnapshot(
                    tier=tier,
                    capacity_bytes=account.capacity_bytes,
                    committed_bytes=sum(r.capacity_charge_bytes for r in entries),
                    pending_bytes=self._pending_locked(tier),
                    committed_materialized_bytes=sum(r.materialized_bytes for r in entries),
                    pending_materialized_bytes=self._pending_materialized_locked(tier),
                    reserved_va_bytes=sum(r.reserved_bytes for r in entries),
                    num_entries=len(entries),
                )
            return LedgerSnapshot(
                device_index=self._device_index,
                tiers=tiers,
                charged_by_owner=self._by_owner_locked(None),
                pending_charged_by_owner=self._pending_by_owner_locked(None),
                materialized_by_owner=self._materialized_by_owner_locked(None),
                num_open_tickets=len(self._pending),
            )

    def report(self) -> str:
        """Human-readable breakdown, for logs and for the exhaustion message."""
        snap = self.snapshot()
        lines: list[str] = []
        for tier in sorted(snap.tiers):
            t = snap.tiers[tier]
            in_flight = f", {t.pending_bytes >> 20} MiB in flight" if t.pending_bytes else ""
            physical = (
                f", {t.materialized_bytes >> 20} MiB materialized"
                if t.materialized_bytes != t.held_bytes
                else ""
            )
            lines.append(
                f"{tier.name:<14} {t.held_bytes >> 20:>8} / {t.capacity_bytes >> 20} MiB "
                f"({t.utilization:.1%}){in_flight}{physical}"
            )
            for owner, nbytes in sorted(self.by_owner(tier).items(), key=lambda kv: -kv[1]):
                lines.append(f"    {owner.value:<11} {nbytes >> 20:>7} MiB")
        return "\n".join(lines)

    def record_metrics(self, registry: _MetricSink) -> None:
        """Publish one internally consistent snapshot into a metric registry."""
        snapshot = self.snapshot()
        device = str(snapshot.device_index)
        for tier, account in snapshot.tiers.items():
            labels = {"device": device, "tier": tier.name.lower()}
            registry.set_gauge("ayaka_memory_capacity_bytes", account.capacity_bytes, labels=labels)
            registry.set_gauge("ayaka_memory_charged_bytes", account.held_bytes, labels=labels)
            registry.set_gauge(
                "ayaka_memory_materialized_bytes", account.materialized_bytes, labels=labels
            )
            registry.set_gauge(
                "ayaka_memory_reserved_va_bytes", account.reserved_va_bytes, labels=labels
            )
        registry.set_gauge(
            "ayaka_memory_open_tickets",
            snapshot.num_open_tickets,
            labels={"device": device},
        )
        held_by_owner = snapshot.held_by_owner
        owners = set(held_by_owner) | set(snapshot.materialized_by_owner)
        for owner in owners:
            labels = {"device": device, "owner": owner.value}
            registry.set_gauge(
                "ayaka_memory_owner_charged_bytes",
                held_by_owner.get(owner, 0),
                labels=labels,
            )
            registry.set_gauge(
                "ayaka_memory_owner_materialized_bytes",
                snapshot.materialized_by_owner.get(owner, 0),
                labels=labels,
            )

    # ── internals (caller holds the lock unless noted) ───────────────────────

    def _account(self, tier: MemoryTier) -> TierAccount:
        account = self._accounts.get(tier)
        if account is None:
            raise KeyError(
                f"no {tier.name} account on this ledger; it carries "
                f"{[t.name for t in sorted(self._accounts)]}"
            )
        return account

    def _require_account(self, tier: MemoryTier) -> None:
        self._account(tier)

    def _committed_locked(self, tier: MemoryTier) -> int:
        return sum(r.capacity_charge_bytes for r in self._entries.values() if r.tier is tier)

    def _committed_materialized_locked(self, tier: MemoryTier) -> int:
        return sum(r.materialized_bytes for r in self._entries.values() if r.tier is tier)

    def _pending_locked(self, tier: MemoryTier) -> int:
        return sum(
            op.destination.capacity_charge_bytes
            for op in self._pending.values()
            if op.destination.tier is tier
        )

    def _pending_materialized_locked(self, tier: MemoryTier) -> int:
        return sum(
            op.destination.materialized_bytes
            for op in self._pending.values()
            if op.destination.tier is tier and op.materialized
        )

    def _by_owner_locked(self, tier: MemoryTier | None) -> dict[MemoryOwner, int]:
        totals: dict[MemoryOwner, int] = {}
        for r in self._entries.values():
            if tier is not None and r.tier is not tier:
                continue
            totals[r.owner] = totals.get(r.owner, 0) + r.capacity_charge_bytes
        return totals

    def _pending_by_owner_locked(self, tier: MemoryTier | None) -> dict[MemoryOwner, int]:
        totals: dict[MemoryOwner, int] = {}
        for op in self._pending.values():
            reservation = op.destination
            if tier is not None and reservation.tier is not tier:
                continue
            totals[reservation.owner] = (
                totals.get(reservation.owner, 0) + reservation.capacity_charge_bytes
            )
        return totals

    def _materialized_by_owner_locked(self, tier: MemoryTier | None) -> dict[MemoryOwner, int]:
        totals: dict[MemoryOwner, int] = {}
        for reservation in self._entries.values():
            if tier is not None and reservation.tier is not tier:
                continue
            totals[reservation.owner] = (
                totals.get(reservation.owner, 0) + reservation.materialized_bytes
            )
        for op in self._pending.values():
            reservation = op.destination
            if not op.materialized or (tier is not None and reservation.tier is not tier):
                continue
            totals[reservation.owner] = (
                totals.get(reservation.owner, 0) + reservation.materialized_bytes
            )
        return totals

    def _check_capacity_locked(
        self,
        reservation: Reservation,
        *,
        excluding: str | None = None,
        excluding_pending: int | None = None,
    ) -> None:
        account = self._account(reservation.tier)
        committed = sum(
            r.capacity_charge_bytes
            for lbl, r in self._entries.items()
            if r.tier is reservation.tier and lbl != excluding
        )
        pending = sum(
            op.destination.capacity_charge_bytes
            for ticket_id, op in self._pending.items()
            if op.destination.tier is reservation.tier and ticket_id != excluding_pending
        )
        charged = reservation.capacity_charge_bytes
        if committed + pending + charged > account.capacity_bytes:
            raise MemoryError(
                f"cannot reserve {charged >> 20} MiB on "
                f"{reservation.tier.name} for {reservation.owner.value} "
                f"({reservation.label!r}): {(committed + pending) >> 20} MiB of a "
                f"{account.capacity_bytes >> 20} MiB capacity is already held, "
                f"leaving {(account.capacity_bytes - committed - pending) >> 20} MiB. "
                f"By owner: {self._format_owners(self._by_owner_locked(reservation.tier))}"
            )

    def _claim_label_locked(
        self,
        label: str,
        ticket_id: int,
        *,
        source_label: str | None = None,
    ) -> None:
        if label in self._claimed_labels or (label in self._entries and label != source_label):
            raise ValueError(
                f"duplicate reservation label {label!r}; release it first or pick a distinct label"
            )
        self._claimed_labels[label] = ticket_id

    def _require_ticket_locked(self, ticket: Ticket) -> _PendingOp:
        op = self._pending.get(ticket.ticket_id)
        if op is None:
            raise KeyError(f"ticket {ticket.ticket_id} is not open")
        if op.ticket is not ticket:
            raise ValueError(
                f"ticket {ticket.ticket_id} does not belong to this ledger transaction"
            )
        return op

    def _finish_ticket_locked(self, op: _PendingOp) -> None:
        self._pending.pop(op.ticket.ticket_id)
        self._claimed_labels.pop(op.destination.label, None)
        if op.source_label is not None:
            self._locked_sources.pop(op.source_label, None)

    def _require_unlocked_source_locked(self, label: str) -> None:
        ticket_id = self._locked_sources.get(label)
        if ticket_id is not None:
            raise RuntimeError(
                f"reservation {label!r} is the source of open transfer ticket {ticket_id}"
            )

    def _record_decision(
        self,
        action: str,
        outcome: str,
        reason: str,
        **attributes: _TelemetryValue,
    ) -> None:
        if self._trace is None:
            return
        safe_attributes: Mapping[str, _TelemetryValue] = attributes
        try:
            self._trace.record(
                component="memory.ledger",
                action=action,
                outcome=outcome,
                reason=reason,
                attributes=safe_attributes,
            )
        except Exception:
            # Telemetry is intentionally fail-open: it must never change the
            # result of a memory admission or lifecycle transaction.
            return

    @staticmethod
    def _format_owners(totals: dict[MemoryOwner, int]) -> str:
        if not totals:
            return "(nothing reserved)"
        return ", ".join(
            f"{owner.value}={nbytes >> 20}MiB"
            for owner, nbytes in sorted(totals.items(), key=lambda kv: -kv[1])
        )

    def __repr__(self) -> str:
        parts = [
            f"{tier.name}={self.held(tier) >> 20}/{acct.capacity_bytes >> 20}MiB"
            for tier, acct in sorted(self._accounts.items())
        ]
        return f"<MemoryLedger device={self._device_index} {' '.join(parts)}>"
