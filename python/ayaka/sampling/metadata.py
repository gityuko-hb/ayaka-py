"""SamplingMetadata -- struct-of-arrays for every per-request sampling parameter.

DESIGN DECISION: this is **not** a logits-processor stack.

temperature / top-p / min-p / penalty are pure functions of the logits and one
per-row number.  They have no state machine.  Wrapping each in an object with
``apply(logits)`` -- vLLM's ``MinPLogitsProcessor`` is the reference mistake --
forces one pass over ``[B, V]`` per processor and makes fusing them into the
sampling kernel impossible.  Here they are **columns**.

Invariants, each with a test:
  * Every tensor is allocated once with capacity ``max_batch_size``.  Never
    reallocated.  Slots are stable for the life of a request.
  * The active rows of a step are a *subset* of slots in packed order.  By
    default that subset is the identity ``[0, n_active)``; the engine slot
    registry calls ``set_active_rows`` when packed order differs from slot
    order (mixed PREFILL+DECODE steps, chunked prefill without sampling).
    Compacting slots instead would corrupt the penalty/RNG state of rows that
    are alive but not sampled this step.
  * Every device column has a parallel host staging column, page-locked when
    CUDA is present.
  * ``apply_batch_update`` is O(|delta|), not O(batch).
  * ``flush()`` copies only dirty columns and mirrors the active-row map.  A
    stable batch means zero dirty columns and therefore zero H2D.
  * ``all_greedy`` is computed on **staging** (host) so no device sync is
    needed to learn it -- which is what lets the planner pick a kernel variant
    and a graph bucket before the forward pass runs.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from ayaka.device.backend import get_backend
from ayaka.memory.ledger import MemoryLedger, Reservation
from ayaka.sampling.params import (
    ALL_COLUMNS,
    COLUMN_DTYPES,
    ColumnValues,
    SamplingParams,
    column_defaults,
    columns_of,
)
from ayaka.types import MemoryOwner, MemoryTier
from ayaka.utils.torch_memory import pinned_empty
from ayaka.utils.torch_utils import resolve_device

if TYPE_CHECKING:
    from ayaka.request.batch import BatchUpdate


@dataclass(frozen=True, slots=True)
class SamplingFootprint:
    """Aggregated memory allocation footprint across storage tiers.

    Attributes:
        device_bytes: Number of bytes allocated in device (accelerator) memory.
        host_pinned_bytes: Number of bytes allocated in host page-locked (pinned) memory.
        host_pageable_bytes: Number of bytes allocated in standard host pageable memory.
    """

    device_bytes: int = 0
    host_pinned_bytes: int = 0
    host_pageable_bytes: int = 0

    def __post_init__(self) -> None:
        # Enforce non-negative byte count invariants across all tiers.
        for name in ("device_bytes", "host_pinned_bytes", "host_pageable_bytes"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

    def __add__(self, other: SamplingFootprint) -> SamplingFootprint:
        """Combine two sampling footprints by summing tier-wise byte counts."""
        return SamplingFootprint(
            device_bytes=self.device_bytes + other.device_bytes,
            host_pinned_bytes=self.host_pinned_bytes + other.host_pinned_bytes,
            host_pageable_bytes=self.host_pageable_bytes + other.host_pageable_bytes,
        )

    @property
    def host_bytes(self) -> int:
        """Return total host memory bytes (pinned plus pageable)."""
        return self.host_pinned_bytes + self.host_pageable_bytes

    @property
    def pinning_succeeded(self) -> bool:
        """Indicate whether all host memory was successfully pinned without pageable fallback."""
        return self.host_pageable_bytes == 0


def measure(tensors: Iterable[torch.Tensor]) -> SamplingFootprint:
    """Measure the memory footprint of an iterable collection of PyTorch tensors.

    Categorizes each tensor by its placement (device, pinned host, or pageable host)
    and sums byte counts based on element size and element count.

    Args:
        tensors: Iterable collection of tensors to inspect.

    Returns:
        SamplingFootprint summarizing total bytes allocated across each tier.
    """
    device = pinned = pageable = 0
    for tensor in tensors:
        nbytes = tensor.numel() * tensor.element_size()
        # Classify allocation tier based on device placement and host pinning.
        if tensor.device.type != "cpu":
            device += nbytes
        elif tensor.is_pinned():
            pinned += nbytes
        else:
            pageable += nbytes
    return SamplingFootprint(
        device_bytes=device, host_pinned_bytes=pinned, host_pageable_bytes=pageable
    )


def charge_sampling_memory(
    ledger: MemoryLedger,
    footprint: SamplingFootprint,
    *,
    label: str,
    device_index: int = 0,
) -> tuple[str, ...]:
    """Record memory reservations for a sampling footprint in the memory ledger.

    Creates backing reservations under the `WORKSPACE` memory owner for each
    non-zero tier in the provided footprint.

    Args:
        ledger: Target MemoryLedger instance to record allocations.
        footprint: Byte measurements across device and host tiers.
        label: Prefix label identifying the reservation entries.
        device_index: Accelerator device index associated with device-tier allocations.

    Returns:
        Tuple of reservation entry names successfully admitted to the ledger.
    """
    written: list[str] = []
    for tier, nbytes, suffix in (
        (MemoryTier.DEVICE, footprint.device_bytes, "device"),
        (MemoryTier.HOST_PINNED, footprint.host_pinned_bytes, "host_pinned"),
        (MemoryTier.HOST_PAGEABLE, footprint.host_pageable_bytes, "host_pageable"),
    ):
        # Skip charging tiers with zero allocated bytes.
        if not nbytes:
            continue
        entry = f"{label}.{suffix}"
        ledger.admit(
            Reservation.backed(
                MemoryOwner.WORKSPACE,
                entry,
                nbytes,
                tier=tier,
                device_index=device_index if tier is MemoryTier.DEVICE else 0,
            )
        )
        written.append(entry)
    return tuple(written)


_DEFAULTS: ColumnValues = column_defaults()


class SamplingMetadata:
    """Column store for one persistent batch."""

    __slots__ = (
        "_dev",
        "_dirty",
        "_rows_dev",
        "_rows_dirty",
        "_rows_list",
        "_rows_stg",
        "_stg",
        "all_greedy",
        "any_bias",
        "any_logprobs",
        "any_penalty",
        "any_support_capture",
        "device",
        "max_batch_size",
        "n_active",
    )

    def __init__(self, max_batch_size: int, device: torch.device | None = None) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be > 0")
        self.max_batch_size = max_batch_size
        self.device = resolve_device(device)

        self._dev: dict[str, torch.Tensor] = {}
        self._stg: dict[str, torch.Tensor] = {}
        for name in ALL_COLUMNS:
            dtype = COLUMN_DTYPES[name]
            self._dev[name] = torch.empty(max_batch_size, dtype=dtype, device=self.device)
            self._stg[name], _ = pinned_empty((max_batch_size,), dtype)

        self._dirty: set[str] = set()
        self._rows_stg, _ = pinned_empty((max_batch_size,), torch.long)
        self._rows_dev = torch.empty(max_batch_size, dtype=torch.long, device=self.device)
        self._rows_list: list[int] | None = None
        self._rows_dirty = False
        self.n_active = 0
        self.all_greedy = True
        self.any_penalty = False
        self.any_logprobs = False
        self.any_support_capture = False
        self.any_bias = False

        self._reset_range(0, max_batch_size)
        self._dirty.clear()

    def column(self, name: str) -> torch.Tensor:
        try:
            return self._dev[name]
        except KeyError:
            raise KeyError(f"no sampling column {name!r}; columns are {ALL_COLUMNS}") from None

    def active(self, name: str) -> torch.Tensor:
        """Device column for this step's active rows, in packed sampling order."""
        column = self.column(name)
        if self._rows_list is None:
            return column.narrow(0, 0, self.n_active)
        rows = self._sync_rows()
        assert rows is not None
        return column.index_select(0, rows)

    def staging(self, name: str) -> torch.Tensor:
        try:
            staging = self._stg[name]
        except KeyError:
            raise KeyError(f"no sampling column {name!r}; columns are {ALL_COLUMNS}") from None
        rows = self._rows_cpu()
        if rows is None:
            return staging.narrow(0, 0, self.n_active)
        return staging.index_select(0, rows)

    @property
    def active_rows(self) -> tuple[int, ...] | None:
        """Explicit slot indices for the active rows; ``None`` means identity."""
        return None if self._rows_list is None else tuple(self._rows_list)

    def set_active_rows(self, rows: Sequence[int] | None) -> None:
        """Select this step's sampling rows as stable slot indices, packed order.

        ``None`` restores identity mode ``[0, n_active)``. Explicit rows must be
        unique and in range; they are staged on the host and mirrored to the
        device by ``flush``. A device read that happens before an explicit flush
        mirrors the pending rows first instead of using last step's map.
        """
        if rows is None:
            self._rows_list = None
            self._rows_dirty = False
            self._recompute_flags()
            return
        values = [int(row) for row in rows]
        if len(set(values)) != len(values):
            raise ValueError("active rows must be unique")
        for row in values:
            if not 0 <= row < self.max_batch_size:
                raise IndexError(f"active row {row} out of range [0, {self.max_batch_size})")
        self._rows_list = values
        if values:
            self._rows_stg[: len(values)] = torch.tensor(values, dtype=torch.long)
        self._rows_dirty = True
        self.n_active = len(values)
        self._recompute_flags()

    def _rows_cpu(self) -> torch.Tensor | None:
        if self._rows_list is None:
            return None
        return self._rows_stg[: self.n_active]

    def _sync_rows(self) -> torch.Tensor | None:
        """Return the active-row indices on the metadata device.

        Mirrors a pending row update instead of failing, so a device read can
        never observe a stale map. The copy is ordered on the caller's stream
        (``non_blocking``), exactly like ``flush``; callers that need a specific
        stream order still call ``flush`` explicitly.
        """
        rows = self._rows_cpu()
        if rows is None:
            return None
        if self._rows_dirty:
            self._rows_dev[: self.n_active].copy_(rows, non_blocking=self.device.type == "cuda")
            self._rows_dirty = False
        return self._rows_dev[: self.n_active]

    def _write(self, slot: int, values: ColumnValues) -> None:
        for name in ALL_COLUMNS:
            self._stg[name][slot] = values[name]
            self._dirty.add(name)

    def _reset_range(self, lo: int, hi: int) -> None:
        for name in ALL_COLUMNS:
            self._stg[name][lo:hi] = _DEFAULTS[name]
            self._dirty.add(name)

    def _move_slot(self, src: int, dst: int) -> None:
        if src == dst:
            return
        for name in ALL_COLUMNS:
            self._stg[name][dst] = self._stg[name][src]
            self._dirty.add(name)

    def apply_batch_update(self, upd: BatchUpdate) -> None:
        """Apply a batch mutation delta updating allocated and active slots.

        Processes deallocations, admissions, and slot migrations in O(|delta|)
        time, avoiding full batch scans. Updates `n_active` and recomputes batch flags.

        Args:
            upd: Batch update delta containing added, removed, and moved requests.

        Raises:
            ValueError: If `upd.batch_size` exceeds `max_batch_size`.
            IndexError: If an added slot index is out of bounds.
        """
        if upd.batch_size > self.max_batch_size:
            raise ValueError(
                f"batch_size {upd.batch_size} exceeds capacity {self.max_batch_size}; "
                "capacity is decided by the planner at a batch boundary, never grown "
                "in the hot path"
            )
        upd.sort_removed()
        for removed in upd.removed:
            self._reset_range(removed.slot, removed.slot + 1)
        for added in upd.added:
            if not 0 <= added.slot < self.max_batch_size:
                raise IndexError(f"slot {added.slot} out of range [0, {self.max_batch_size})")
            self._write(added.slot, columns_of(added.params, request_index=added.request_index))
        for moved in upd.moved:
            self._move_slot(moved.src, moved.dst)

        self.n_active = upd.batch_size
        self._recompute_flags()

    def write_slot(self, slot: int, params: SamplingParams, *, request_index: int = 0) -> None:
        """Write one slot's columns without touching ``n_active``.

        Engine slot ownership lives in ``ayaka.sampling.engine``;
        ``set_slot`` keeps the test-friendly "grow to fit" behavior.
        """
        if not 0 <= slot < self.max_batch_size:
            raise IndexError(f"slot {slot} out of range [0, {self.max_batch_size})")
        self._write(slot, columns_of(params, request_index=request_index))

    def reset_slot(self, slot: int) -> None:
        """Restore one slot to column defaults after its request is released."""
        if not 0 <= slot < self.max_batch_size:
            raise IndexError(f"slot {slot} out of range [0, {self.max_batch_size})")
        self._reset_range(slot, slot + 1)

    def set_slot(self, slot: int, params: SamplingParams, *, request_index: int = 0) -> None:
        """Configure parameters for a slot, automatically updating `n_active` if necessary."""
        if not 0 <= slot < self.max_batch_size:
            raise IndexError(f"slot {slot} out of range [0, {self.max_batch_size})")
        self._write(slot, columns_of(params, request_index=request_index))
        self.n_active = max(self.n_active, slot + 1)
        self._recompute_flags()

    def step_rng(self) -> None:
        """Increment the RNG offset by 1 for all currently active batch rows.

        Marks the offset column dirty to ensure updated offsets mirror to device
        on the next flush.
        """
        if not self.n_active:
            return
        rows = self._rows_cpu()
        if rows is None:
            self._stg["offset"][: self.n_active] += 1
        else:
            self._stg["offset"][rows] += 1
        self._dirty.add("offset")

    def _recompute_flags(self) -> None:
        n = self.n_active
        if n == 0:
            self.all_greedy = True
            self.any_penalty = False
            self.any_logprobs = False
            self.any_support_capture = False
            self.any_bias = False
            return
        rows = self._rows_cpu()

        def column(name: str) -> torch.Tensor:
            staging = self._stg[name]
            return staging[:n] if rows is None else staging[rows]

        temp = column("temperature")
        top_k = column("top_k")
        greedy_rows = (temp == 0.0) | (top_k == 1)
        self.all_greedy = bool(torch.all(greedy_rows).item())
        self.any_penalty = bool(
            torch.any(column("rep_penalty") != 1.0).item()
            or torch.any(column("freq_penalty") != 0.0).item()
            or torch.any(column("pres_penalty") != 0.0).item()
        )
        self.any_logprobs = bool(torch.any(column("logprobs_k") >= 0).item())
        self.any_support_capture = bool(torch.any(column("return_support") != 0).item())
        self.any_bias = bool(torch.any(column("bias_count") > 0).item())

    def host_scalar(self, name: str, slot: int) -> int | float:
        """Host staging value for one slot — no device sync, no active-row map.

        The runner reads reporting parameters (logprobs k, mode) straight from
        staging: they are decided at admission and only need host visibility.
        """
        if not 0 <= slot < self.max_batch_size:
            raise IndexError(f"slot {slot} out of range [0, {self.max_batch_size})")
        try:
            staging = self._stg[name]
        except KeyError:
            raise KeyError(f"no sampling column {name!r}; columns are {ALL_COLUMNS}") from None
        return staging[slot].item()

    def flush(self, stream: Any = None) -> int:
        """Synchronize dirty host staging columns and active-row indices to device memory.

        Executes asynchronous non-blocking copies on CUDA streams when available.

        Args:
            stream: Optional CUDA stream context to order copy operations.

        Returns:
            Number of distinct column buffers copied from host to device.
        """
        if self.n_active == 0:
            self._dirty.clear()
            self._rows_dirty = False
            return 0
        rows = self._rows_cpu()
        mirror_rows = rows is not None and self._rows_dirty
        if not self._dirty and not mirror_rows:
            return 0
        n = self.n_active
        non_blocking = self.device.type == "cuda"
        count = 0
        with get_backend().stream_context(stream):
            if mirror_rows:
                assert rows is not None
                self._rows_dev[:n].copy_(rows, non_blocking=non_blocking)
                self._rows_dirty = False
            if rows is None:
                for name in self._dirty:
                    self._dev[name].narrow(0, 0, n).copy_(
                        self._stg[name].narrow(0, 0, n),
                        non_blocking=non_blocking,
                    )
                    count += 1
            else:
                device_rows = self._rows_dev[:n]
                for name in self._dirty:
                    source = self._stg[name].index_select(0, rows)
                    if self.device.type != "cpu":
                        source = source.to(self.device, non_blocking=non_blocking)
                    self._dev[name].index_copy_(0, device_rows, source)
                    count += 1
        self._dirty.clear()
        return count

    def footprint(self) -> SamplingFootprint:
        """Measure the aggregate memory allocation footprint of all managed tensors."""
        return measure([*self._dev.values(), *self._stg.values(), self._rows_dev, self._rows_stg])

    def data_ptrs(self) -> dict[str, tuple[int, int]]:
        """Return memory buffer pointers for device and host tensors per column."""
        return {n: (self._dev[n].data_ptr(), self._stg[n].data_ptr()) for n in ALL_COLUMNS}

    def dirty_columns(self) -> frozenset[str]:
        """Return the set of column names currently marked dirty in host staging."""
        return frozenset(self._dirty)
