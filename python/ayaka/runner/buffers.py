"""Persistent per-flight runner buffers with ticket-scoped leases.

The paged runner used to build every attention-metadata, token and sampling
tensor fresh on each step. That makes the hot path allocation-bound and gives
CUDA-graph capture nothing stable to bind to. This module owns one fixed set of
backing tensors per flight slot, allocated once from the frozen capacity plan
and handed out only under a lease that follows the ticket from preparation to
retirement.

Three lifetime boundaries, and why each one exists:

* **Host staging.** Each slot carries pinned (or, on the CPU lane, alias)
  staging views. A stage write is legal only while the slot is leased, and the
  slot returns to the free list only after the stage-before-enqueue path is
  proven safe. Nothing mutates host staging while a H2D copy from it can still
  be in flight.
* **Device input.** Token ids, positions, offsets, tables, slots and sampling
  rows stay in the slot's device tensors until the model forward that reads
  them completes. The executor's completion fence is that proof: a slot is not
  reusable before executor-confirmed quiescence.
* **Output.** ``SampleOutputs`` remain ticket-owned device storage (see
  ``SampleOutputs``); they are materialized to host exactly once at the
  completion boundary and released at retire. This module does not pool them.

Staging deliberately avoids ``torch.tensor``/``.to``/``.clone`` on the hot
path: host values are written through NumPy views of the pre-allocated staging
tensors and uploaded with ``copy_(..., non_blocking=True)`` into pre-allocated
device views. Shrinking batches zero the device tail that the previous, larger
stage wrote, so a later request can never read stale metadata. Ceilings
(sequences, tokens, per-group block-table width) are validated before any
write; overflow is a hard error, never a truncation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import count
from typing import TYPE_CHECKING, Any

import torch

from ayaka.memory.capacity import MemoryLane, claim_tier
from ayaka.memory.ledger import Reservation
from ayaka.types import MemoryOwner

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ayaka.memory.ledger import MemoryLedger

__all__ = [
    "RunnerBufferCapacityError",
    "RunnerBufferExhausted",
    "RunnerBufferLease",
    "RunnerBufferSpec",
    "RunnerBufferStats",
    "RunnerBuffers",
    "StagedGroup",
]

#: Process-local pool generation. A rebuild mints a new one so graph identity
#: (``ResourceGeneration.buffers``) can never confuse old backing with new.
_BUFFER_GENERATIONS = count(1)


class RunnerBufferCapacityError(RuntimeError):
    """A step needs more metadata than the frozen buffer plan declares.

    Raised before any staging write. The plan is sized from the resolved
    scheduler ceilings, so this means the step bypassed those ceilings --
    rejecting is the only option that cannot silently truncate a block table.
    """


class RunnerBufferExhausted(RuntimeError):
    """Every flight slot is leased.

    Backpressure, not corruption: the caller must wait for completion/retire
    (which releases a lease) instead of allocating a buffer outside the plan.
    """


@dataclass(frozen=True, slots=True)
class RunnerBufferStats:
    capacity: int
    live: int
    generation: int
    acquire_count: int
    release_count: int
    exhausted_count: int
    alloc_events: int


@dataclass(frozen=True, slots=True)
class RunnerBufferSpec:
    """Frozen geometry of one pool, derived from scheduler/context ceilings.

    ``group_columns`` carries each attention group's block-table width ceiling,
    ``ceil(max_model_len / page_size)``, computed by the caller from the group
    geometry. A pool never grows to fit a step: a step that does not fit is
    rejected.
    """

    max_num_seqs: int
    max_num_batched_tokens: int
    max_inflight: int
    group_columns: tuple[tuple[str, int], ...]

    @classmethod
    def create(
        cls,
        *,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        max_inflight: int,
        group_columns: Mapping[str, int],
    ) -> RunnerBufferSpec:
        return cls(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            max_inflight=max_inflight,
            group_columns=tuple(sorted(group_columns.items())),
        )

    def __post_init__(self) -> None:
        for name in ("max_num_seqs", "max_num_batched_tokens", "max_inflight"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.group_columns) is not tuple or not self.group_columns:
            raise TypeError("group_columns must be a non-empty tuple of (name, columns)")
        if len({name for name, _ in self.group_columns}) != len(self.group_columns):
            raise ValueError("group_columns names must be unique")
        for name, columns in self.group_columns:
            if not isinstance(name, str) or not name:
                raise ValueError("group names must be non-empty strings")
            if not isinstance(columns, int) or isinstance(columns, bool) or columns < 1:
                raise ValueError(f"group {name!r} needs a positive column ceiling")

    def columns_for(self, name: str) -> int:
        for group_name, columns in self.group_columns:
            if group_name == name:
                return columns
        raise KeyError(f"runner buffers do not cover attention group {name!r}")

    @property
    def per_flight_device_bytes(self) -> int:
        """Bytes one flight's device tensors occupy."""
        tokens = self.max_num_batched_tokens
        seqs = self.max_num_seqs
        total = 2 * 8 * tokens  # token ids and model positions, int64
        total += 8 * seqs  # sampling rows, int64
        for _, columns in self.group_columns:
            total += 4 * (seqs + 1)  # query_start_loc
            total += 4 * seqs  # seq_lens
            total += 4 * seqs  # computed_lens
            total += 4 * seqs * columns  # block_table
            total += 4 * tokens  # slot_mapping
            total += 4 * tokens  # attention positions
        return total

    @property
    def device_bytes(self) -> int:
        return self.per_flight_device_bytes * self.max_inflight

    @property
    def staging_bytes(self) -> int:
        """Bytes of pinned host mirrors, equal in shape to the device tensors."""
        return self.device_bytes


class _Stage1D:
    """One pre-allocated array plus its staging mirror and dirty high-water."""

    __slots__ = ("device", "host", "high")

    def __init__(self, size: int, *, dtype: torch.dtype, device: torch.device, pin: bool) -> None:
        self.device = torch.zeros(size, dtype=dtype, device=device)
        self.host = torch.zeros(size, dtype=dtype, pin_memory=pin) if pin else self.device
        #: Highest index ever written on the device; everything at or beyond it
        #: is still the construction-time zero and safe to leave untouched.
        self.high = 0

    def fill(self, values: Sequence[int]) -> torch.Tensor:
        n = len(values)
        if n > self.device.numel():
            raise RunnerBufferCapacityError(
                f"metadata row needs {n} values but the buffer plan allows {self.device.numel()}"
            )
        if n:
            self.host.numpy()[:n] = values
        if n < self.high:
            self.device[n : self.high].zero_()
        if self.host is not self.device and n:
            self.device[:n].copy_(self.host[:n], non_blocking=True)
        self.high = n
        return self.device[:n]


class _Stage2D:
    """Two-dimensional variant of :class:`_Stage1D` with row/column tracking."""

    __slots__ = ("device", "host", "rows_high", "cols_high")

    def __init__(
        self, rows: int, columns: int, *, dtype: torch.dtype, device: torch.device, pin: bool
    ) -> None:
        self.device = torch.zeros((rows, columns), dtype=dtype, device=device)
        if pin:
            self.host = torch.zeros((rows, columns), dtype=dtype, pin_memory=True)
        else:
            self.host = self.device
        self.rows_high = 0
        self.cols_high = 0

    def fill(self, rows: Sequence[Sequence[int]], width: int) -> torch.Tensor:
        n = len(rows)
        if n > self.device.shape[0] or width > self.device.shape[1]:
            raise RunnerBufferCapacityError(
                f"block table needs {n}x{width} but the buffer plan allows "
                f"{self.device.shape[0]}x{self.device.shape[1]}"
            )
        if n and width:
            self.host.numpy()[:n, :width] = rows
        if n < self.rows_high:
            self.device[n : self.rows_high].zero_()
        if width < self.cols_high:
            self.device[:n, width : self.cols_high].zero_()
        if self.host is not self.device and n and width:
            self.device[:n, :width].copy_(self.host[:n, :width], non_blocking=True)
        self.rows_high = n
        self.cols_high = width
        return self.device[:n, :width]


class _GroupBuffers:
    """Device tensors and staging mirrors for one attention group in one slot."""

    __slots__ = (
        "name",
        "columns",
        "query_start_loc",
        "seq_lens",
        "computed_lens",
        "block_table",
        "slot_mapping",
        "positions",
        "query_start_loc_host",
        "seq_lens_host",
    )

    def __init__(
        self, name: str, columns: int, spec: RunnerBufferSpec, *, device, pin: bool
    ) -> None:
        seqs = spec.max_num_seqs
        tokens = spec.max_num_batched_tokens
        dtype = torch.int32
        self.name = name
        self.columns = columns
        self.query_start_loc = _Stage1D(seqs + 1, dtype=dtype, device=device, pin=pin)
        self.seq_lens = _Stage1D(seqs, dtype=dtype, device=device, pin=pin)
        self.computed_lens = _Stage1D(seqs, dtype=dtype, device=device, pin=pin)
        self.block_table = _Stage2D(seqs, columns, dtype=dtype, device=device, pin=pin)
        self.slot_mapping = _Stage1D(tokens, dtype=dtype, device=device, pin=pin)
        self.positions = _Stage1D(tokens, dtype=dtype, device=device, pin=pin)
        self.query_start_loc_host = self.query_start_loc.host
        self.seq_lens_host = self.seq_lens.host

    def tensors(self) -> tuple[torch.Tensor, ...]:
        """Every backing tensor, device first, distinct host mirrors included."""
        found: list[torch.Tensor] = []
        for stage in (
            self.query_start_loc,
            self.seq_lens,
            self.computed_lens,
            self.block_table,
            self.slot_mapping,
            self.positions,
        ):
            found.append(stage.device)
            if stage.host is not stage.device:
                found.append(stage.host)
        return tuple(found)


@dataclass(frozen=True, slots=True)
class StagedGroup:
    """Views over one group's slot buffers for the current step."""

    name: str
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    computed_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    positions: torch.Tensor
    query_start_loc_cpu: torch.Tensor
    seq_lens_cpu: torch.Tensor


class FlightBuffers:
    """One flight slot: token input, per-group attention metadata, sampling rows."""

    __slots__ = ("spec", "index", "generation", "_groups", "_tokens", "_positions", "_rows")

    def __init__(
        self,
        spec: RunnerBufferSpec,
        *,
        index: int,
        device: torch.device,
        pin_staging: bool,
        generation: int,
    ) -> None:
        self.spec = spec
        self.index = index
        self.generation = generation
        self._tokens = _Stage1D(
            spec.max_num_batched_tokens, dtype=torch.int64, device=device, pin=pin_staging
        )
        self._positions = _Stage1D(
            spec.max_num_batched_tokens, dtype=torch.int64, device=device, pin=pin_staging
        )
        self._rows = _Stage1D(spec.max_num_seqs, dtype=torch.int64, device=device, pin=pin_staging)
        self._groups = {
            name: _GroupBuffers(name, columns, spec, device=device, pin=pin_staging)
            for name, columns in spec.group_columns
        }

    @property
    def groups(self) -> Mapping[str, _GroupBuffers]:
        return self._groups

    def tensors(self) -> tuple[torch.Tensor, ...]:
        """Every backing tensor of this slot, for pointer-stability checks."""
        found: list[torch.Tensor] = []
        for stage in (self._tokens, self._positions, self._rows):
            found.append(stage.device)
            if stage.host is not stage.device:
                found.append(stage.host)
        for group in self._groups.values():
            found.extend(group.tensors())
        return tuple(found)

    def stage_tokens(
        self, token_ids: Sequence[int], positions: Sequence[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(token_ids) != len(positions):
            raise ValueError("token ids and positions must have the same length")
        return self._tokens.fill(token_ids), self._positions.fill(positions)

    def stage_sampling_rows(self, rows: Sequence[int]) -> torch.Tensor:
        return self._rows.fill(rows)

    def stage_group(
        self,
        name: str,
        *,
        starts: Sequence[int],
        lengths: Sequence[int],
        computed: Sequence[int],
        tables: Sequence[Sequence[int]],
        width: int,
        slots: Sequence[int],
        positions: Sequence[int],
    ) -> StagedGroup:
        """Validate, stage and upload one group's metadata; returns device views.

        All counts are validated against the frozen plan first. The active
        extents are written from pinned host staging on one non-blocking copy;
        any device tail left by a previous, larger stage is zeroed so it cannot
        leak into a later request.
        """
        group = self._groups.get(name)
        if group is None:
            raise KeyError(f"runner buffers do not cover attention group {name!r}")
        num_reqs = len(lengths)
        if num_reqs != len(starts) - 1:
            raise ValueError("query_start_loc must carry one more entry than seq_lens")
        if num_reqs != len(computed) or num_reqs != len(tables):
            raise ValueError("per-row metadata lengths must agree")
        num_tokens = len(slots)
        if num_tokens != len(positions):
            raise ValueError("slot mapping and positions must have the same length")
        if num_reqs > self.spec.max_num_seqs:
            raise RunnerBufferCapacityError(
                f"attention group {name!r} has {num_reqs} rows but the buffer plan allows "
                f"{self.spec.max_num_seqs}"
            )
        if num_tokens > self.spec.max_num_batched_tokens:
            raise RunnerBufferCapacityError(
                f"attention group {name!r} carries {num_tokens} tokens but the buffer plan "
                f"allows {self.spec.max_num_batched_tokens}"
            )
        if width > group.columns:
            raise RunnerBufferCapacityError(
                f"attention group {name!r} needs a {width}-column block table but the buffer "
                f"plan allows {group.columns}; the sequence ceiling and the plan diverged"
            )
        for table in tables:
            if len(table) != width:
                raise ValueError("every block-table row must be padded to the staged width")
        query_start_loc = group.query_start_loc.fill(starts)
        seq_lens = group.seq_lens.fill(lengths)
        computed_lens = group.computed_lens.fill(computed)
        block_table = group.block_table.fill(tables, width)
        slot_mapping = group.slot_mapping.fill(slots)
        positions_dev = group.positions.fill(positions)
        return StagedGroup(
            name=name,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            computed_lens=computed_lens,
            block_table=block_table,
            slot_mapping=slot_mapping,
            positions=positions_dev,
            query_start_loc_cpu=group.query_start_loc_host[: len(starts)],
            seq_lens_cpu=group.seq_lens_host[:num_reqs],
        )


class RunnerBufferLease:
    """One flight slot, bound to a ticket until quiescent completion."""

    __slots__ = ("_pool", "_released", "index", "step_id", "_ticket_id")

    def __init__(self, pool: RunnerBuffers, index: int, step_id: int) -> None:
        self._pool = pool
        self.index = index
        self.step_id = step_id
        self._ticket_id: Any = None
        self._released = False

    @property
    def buffers(self) -> FlightBuffers:
        if self._released:
            raise RuntimeError("runner buffer lease was released")
        return self._pool._slots[self.index]

    @property
    def generation(self) -> int:
        return self._pool.generation

    @property
    def ticket_id(self) -> Any:
        return self._ticket_id

    @property
    def released(self) -> bool:
        return self._released

    def bind(self, ticket_id: Any) -> None:
        """Attach the adopting ticket identity; once, by the owner only."""
        if self._released:
            raise RuntimeError("runner buffer lease was released before binding")
        if self._ticket_id is not None and self._ticket_id != ticket_id:
            raise ValueError("runner buffer lease already belongs to another ticket")
        self._ticket_id = ticket_id

    def release(self) -> None:
        """Return the slot to the pool; idempotent."""
        if self._released:
            return
        self._pool._release(self)
        self._released = True

    def __enter__(self) -> RunnerBufferLease:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()


class RunnerBuffers:
    """Fixed pool of :class:`FlightBuffers`, one per admitted flight.

    The pool is sized once from :class:`RunnerBufferSpec` and never grows on the
    hot path. Device metadata is claimed in the memory ledger as one
    WORKSPACE/DEVICE reservation; pinned staging is claimed separately unless
    the caller already reserved it (the serving runtime admits a
    ``staging_bytes`` claim up front, which the pool then fits inside).
    """

    __slots__ = (
        "_acquire_count",
        "_alloc_events",
        "_claims",
        "_closed",
        "_exhausted_count",
        "_free",
        "_label",
        "_leases",
        "_ledger",
        "_release_count",
        "_slots",
        "_stage_claim",
        "device",
        "generation",
        "spec",
    )

    def __init__(
        self,
        spec: RunnerBufferSpec,
        *,
        device: torch.device,
        pin_staging: bool = False,
        ledger: MemoryLedger | None = None,
        device_index: int = 0,
        label: str = "runner.buffers",
        reserve_staging: bool = True,
    ) -> None:
        if not isinstance(spec, RunnerBufferSpec):
            raise TypeError("spec must be a RunnerBufferSpec")
        self.spec = spec
        self.device = device
        self.generation = next(_BUFFER_GENERATIONS)
        self._label = label
        self._ledger = ledger
        self._closed = False
        self._slots = [
            FlightBuffers(
                spec,
                index=index,
                device=device,
                pin_staging=pin_staging,
                generation=self.generation,
            )
            for index in range(spec.max_inflight)
        ]
        self._free: deque[int] = deque(range(spec.max_inflight))
        self._leases: dict[int, RunnerBufferLease] = {}
        self._acquire_count = 0
        self._release_count = 0
        self._exhausted_count = 0
        self._stage_claim = False
        # One allocation event per backing tensor; a stage never adds any.
        self._alloc_events = len(self._slots[0].tensors()) * spec.max_inflight
        self._claims: tuple[str, ...] = ()
        if ledger is not None:
            self._claims = self._admit(ledger, device_index=device_index, staging=reserve_staging)
            self._stage_claim = any(claim.endswith(".staging") for claim in self._claims)

    def _admit(self, ledger: MemoryLedger, *, device_index: int, staging: bool) -> tuple[str, ...]:
        lane = MemoryLane.CUDA if self.device.type == "cuda" else MemoryLane.CPU
        claims = [
            (
                f"{self._label}.g{self.generation}.device",
                self.spec.device_bytes,
                claim_tier(MemoryOwner.WORKSPACE, lane=lane),
                device_index,
            )
        ]
        if staging and self.spec.staging_bytes:
            claims.append(
                (
                    f"{self._label}.g{self.generation}.staging",
                    self.spec.staging_bytes,
                    claim_tier(MemoryOwner.WORKSPACE, lane=lane, pinned_host=True),
                    0,
                )
            )
        written: list[str] = []
        for label, nbytes, tier, index in claims:
            ledger.admit(
                Reservation.backed(
                    MemoryOwner.WORKSPACE,
                    label,
                    nbytes,
                    tier=tier,
                    device_index=index,
                )
            )
            written.append(label)
        return tuple(written)

    # ── lease lifecycle ──────────────────────────────────────────────────────

    def acquire(self, *, step_id: int) -> RunnerBufferLease:
        """Lease a slot for one prepared step, or raise backpressure."""
        if self._closed:
            raise RuntimeError("runner buffer pool is closed")
        if not isinstance(step_id, int) or isinstance(step_id, bool) or step_id < 0:
            raise ValueError("step_id must be a non-negative integer")
        if not self._free:
            self._exhausted_count += 1
            raise RunnerBufferExhausted(
                f"all {len(self._slots)} runner buffer flight slots are leased; "
                "wait for a completion before preparing another step"
            )
        index = self._free.popleft()
        lease = RunnerBufferLease(self, index, step_id)
        self._leases[index] = lease
        self._acquire_count += 1
        return lease

    def _release(self, lease: RunnerBufferLease) -> None:
        if self._leases.get(lease.index) is not lease:
            raise ValueError("lease is not active in this runner buffer pool")
        del self._leases[lease.index]
        self._free.append(lease.index)
        self._release_count += 1

    # ── introspection ────────────────────────────────────────────────────────

    @property
    def live(self) -> int:
        return len(self._leases)

    @property
    def closed(self) -> bool:
        return self._closed

    def pointers(self) -> tuple[tuple[int, ...], ...]:
        """Stable device/staging ``data_ptr`` per slot; changes only on rebuild."""
        return tuple(tuple(tensor.data_ptr() for tensor in slot.tensors()) for slot in self._slots)

    def stats(self) -> RunnerBufferStats:
        return RunnerBufferStats(
            capacity=len(self._slots),
            live=self.live,
            generation=self.generation,
            acquire_count=self._acquire_count,
            release_count=self._release_count,
            exhausted_count=self._exhausted_count,
            alloc_events=self._alloc_events,
        )

    def rebuild(self, spec: RunnerBufferSpec) -> RunnerBuffers:
        """Replace the backing, allowed only with no live lease.

        The new pool is admitted before the old one is released so the ledger
        never observes a gap, and the generation advances so stale references
        fail identity checks instead of reading freed storage.
        """
        if self.live:
            raise RunnerBufferCapacityError(
                "runner buffers cannot be rebuilt while a flight lease is live"
            )
        replacement = RunnerBuffers(
            spec,
            device=self.device,
            pin_staging=self._pin_staging,
            ledger=self._ledger,
            label=self._label,
            reserve_staging=self._stage_claim,
        )
        self.close()
        return replacement

    @property
    def _pin_staging(self) -> bool:
        if not self._slots:
            return False
        stage = self._slots[0]._tokens
        return stage.host is not stage.device

    def close(self) -> None:
        """Release backing and claims; refuses while a lease is live."""
        if self._closed:
            return
        if self.live:
            raise RuntimeError(
                f"cannot close runner buffers with {self.live} live lease(s); "
                "drain the tickets that own them first"
            )
        if self._ledger is not None:
            for claim in self._claims:
                if self._ledger.get(claim) is not None:
                    self._ledger.release(claim)
        self._claims = ()
        self._slots = []
        self._free.clear()
        self._closed = True

    def __repr__(self) -> str:
        return (
            f"<RunnerBuffers flights={len(self._slots)} live={self.live} "
            f"generation={self.generation} device={self.spec.device_bytes >> 10} KiB>"
        )
