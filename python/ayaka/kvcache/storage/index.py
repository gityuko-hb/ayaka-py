"""Slot addressing and the scatter / gather / fill primitives.

Slot addressing is flat::

    slot = page_id * page_size + offset_in_page

so a plane of shape ``(pages, page_size, *tail)`` reshapes -- zero-copy,
because it is contiguous -- to ``(pages * page_size, *tail)`` and a slot
scatter is exactly ``index_copy_`` along dim 0.

**The performance fix in this module.** The previous implementation validated
every index tensor on the device::

    torch.any(indices < 0).item()            # sync 1
    torch.any(indices >= capacity).item()    # sync 2
    indices.unique().numel() != numel        # sync 3 (sort + dynamic shape)

Three device-to-host synchronizations per plane per layer -- 72 per step for a
24-layer model -- each of which drains the pipeline. Worse, ``.item()`` cannot
be captured into a CUDA graph, so the whole write path was structurally
graph-hostile.

The observation is that the input is almost always a *Python sequence* built by
the scheduler, and validating a Python sequence on the CPU costs no device
round trip at all: ``min()``, ``max()`` and ``len(set(...))`` are C loops over
memory that is already host-side. Only an input that is *already* a device
tensor needs device predicates, and for that case
:func:`slot_index` takes ``validate=False`` so a caller that built the tensor
itself (and therefore already knows it is valid) can opt out entirely and stay
graph-capturable.

A :class:`SlotIndex` carries proof of what was checked, so passing one down
never re-validates.
"""

from __future__ import annotations

from array import array
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from ayaka.kvcache.storage.dtypes import is_fp8_storage_dtype
from ayaka.kvcache.storage.errors import SlotAddressError
from ayaka.utils.torch_utils import dtype_name

__all__ = [
    "SlotIndex",
    "host_index_buffer",
    "index_copy_storage",
    "index_fill_storage",
    "index_select_storage",
    "page_index",
    "slot_index",
    "validated_page_ids",
]


@dataclass(frozen=True, slots=True)
class SlotIndex:
    """A validated 1-D ``int64`` index tensor plus proof of what was checked.

    Attributes:
        tensor: The device tensor to hand to ``index_copy_`` / ``index_select``.
        count: Number of addresses.
        capacity_slots: Bound the addresses were checked against.
        unique: Whether uniqueness was verified.

    Build one through :func:`slot_index`, then reuse it across every layer of a
    step. Re-validating the same addresses 24 times is pure overhead, and this
    type is what makes skipping it safe rather than merely fast.
    """

    tensor: Any
    count: int
    capacity_slots: int
    unique: bool

    def require_unique(self) -> None:
        """Assert this index was checked for duplicates.

        Raises:
            SlotAddressError: if it was not.
        """
        if not self.unique:
            raise SlotAddressError(
                "this index was built for gathering (duplicates allowed) and cannot be "
                "used to write: scattering to a repeated slot lets the second write "
                "silently overwrite the first"
            )

    def require_count(self, expected: int) -> None:
        """Assert the address count matches a value count.

        Raises:
            SlotAddressError: on mismatch.
        """
        if self.count != expected:
            raise SlotAddressError(f"slot count {self.count} does not match {expected} values")

    def require_capacity(self, capacity_slots: int) -> None:
        """Assert this index was bounds-checked against ``capacity_slots``."""
        if self.capacity_slots != capacity_slots:
            raise SlotAddressError(
                f"index was bounds-checked against {self.capacity_slots} slots "
                f"but is being used on storage with {capacity_slots}"
            )


def host_index_buffer(
    values: Sequence[int],
    *,
    capacity_slots: int,
    require_unique: bool,
) -> array:
    """Convert and validate a host-side sequence into an ``int64`` buffer.

    Deliberately torch-free, so the logic that replaced three device
    synchronizations is unit-testable on a machine with no PyTorch.

    Building the ``array`` first is doing double duty. ``array("q", values)``
    is one C-level pass that both converts and type-checks: floats, strings and
    ``None`` are rejected by the array constructor, and NumPy integers are
    accepted, which a naive ``isinstance(x, int)`` guard would wrongly refuse.
    It is also several times faster than ``torch.as_tensor(list)``, which walks
    the list element by element through the Python C-API -- worth having on a
    prefill chunk of a few thousand addresses.

    ``min``/``max`` over the array and ``len(set(...))`` are then single passes
    over native memory. Together they cost tens of microseconds for an 8k
    chunk, against the hundreds of microseconds and full pipeline drain of one
    device synchronization -- and, unlike the device predicates, they place no
    obstacle in front of CUDA graph capture.

    Raises:
        SlotAddressError: on a non-integer, out-of-range or duplicated address.
    """
    try:
        buffer = array("q", values)  # "q" == signed long long == int64
    except (TypeError, OverflowError, ValueError) as exc:
        raise SlotAddressError(f"slot addresses must be integers: {exc}") from exc
    if not buffer:
        return buffer
    lowest = min(buffer)
    highest = max(buffer)
    if lowest < 0:
        raise SlotAddressError(f"slot addresses must be non-negative, got {lowest}")
    if highest >= capacity_slots:
        raise SlotAddressError(
            f"slot address {highest} is outside storage (capacity {capacity_slots} slots)"
        )
    if require_unique and len(set(buffer)) != len(buffer):
        raise SlotAddressError(
            "slot addresses must be unique when writing: a repeated slot means the "
            "second scatter silently overwrites the first"
        )
    return buffer


def _tensor_from_buffer(torch: Any, buffer: array, *, device: Any) -> Any:
    """Wrap a validated host buffer as an ``int64`` device tensor.

    ``torch.frombuffer`` adopts the array's memory with no further copy and
    keeps the array object alive through the tensor's storage, so the local
    buffer outliving this frame is not a dangling reference.

    The host tensor is pageable, so the host-to-device copy is synchronous.
    Staging through a pinned buffer is the next improvement for the hot path,
    and belongs to whoever owns the step's pinned arena rather than here.
    """
    if not buffer:
        return torch.empty(0, dtype=torch.int64, device=device)
    return torch.frombuffer(buffer, dtype=torch.int64).to(device=device)


def slot_index(
    torch: Any,
    slots: Any,
    *,
    device: Any,
    capacity_slots: int,
    require_unique: bool,
    expected_count: int | None = None,
    validate: bool = True,
) -> SlotIndex:
    """Build a validated slot index from a sequence, a tensor, or a SlotIndex.

    Args:
        torch: The imported torch module.
        slots: A Python sequence of ints, a 1-D integer tensor, or an already
            built :class:`SlotIndex`.
        device: Device the resulting tensor must live on.
        capacity_slots: Exclusive upper bound, i.e. ``capacity_pages * page_size``.
        require_unique: True on write paths, False on gather paths. Retention
            eviction resolves evicted blocks to a shared padding page, so many
            logical positions legitimately gather the same slot.
        expected_count: When given, the number of value rows the addresses must
            match.
        validate: Set False only when ``slots`` is already a device tensor the
            caller built and knows to be valid. This is the CUDA-graph-safe
            path: it performs no ``.item()`` and therefore no synchronization.

    Returns:
        A :class:`SlotIndex` recording what was checked.

    Raises:
        SlotAddressError: on a malformed, out-of-range or duplicated address.
    """
    if isinstance(slots, SlotIndex):
        slots.require_capacity(capacity_slots)
        if require_unique:
            slots.require_unique()
        if expected_count is not None:
            slots.require_count(expected_count)
        return slots

    if isinstance(slots, torch.Tensor):
        indices = slots
        if indices.ndim != 1:
            raise SlotAddressError(f"slots must be one-dimensional, got {indices.ndim} dims")
        indices = indices.to(dtype=torch.int64, device=device)
        count = int(indices.shape[0])
        if validate and count:
            # One fused predicate instead of two: one synchronization, not two.
            out_of_range = ((indices < 0) | (indices >= capacity_slots)).any()
            if bool(out_of_range.item()):
                raise SlotAddressError(
                    f"a slot address is outside storage (capacity {capacity_slots} slots)"
                )
            if require_unique and int(indices.unique().numel()) != count:
                raise SlotAddressError("slot addresses must be unique when writing")
        if expected_count is not None and count != expected_count:
            raise SlotAddressError(f"slot count {count} does not match {expected_count} values")
        return SlotIndex(
            tensor=indices,
            count=count,
            capacity_slots=capacity_slots,
            unique=require_unique and validate,
        )

    values: Sequence[int]
    if isinstance(slots, Sequence) and not isinstance(slots, (str, bytes)):
        values = slots
    elif isinstance(slots, Iterable):
        values = tuple(slots)
    else:
        raise SlotAddressError(f"slots must be iterable, got {type(slots).__name__}")

    buffer = host_index_buffer(
        values,
        # A skipped validation still needs the conversion, so pass a bound that
        # cannot reject anything rather than branching around the whole call.
        capacity_slots=capacity_slots if validate else (1 << 62),
        require_unique=require_unique and validate,
    )
    if expected_count is not None and len(buffer) != expected_count:
        raise SlotAddressError(f"slot count {len(buffer)} does not match {expected_count} values")
    return SlotIndex(
        tensor=_tensor_from_buffer(torch, buffer, device=device),
        count=len(buffer),
        capacity_slots=capacity_slots,
        unique=require_unique and validate,
    )


def validated_page_ids(
    physical_pages: Iterable[int],
    *,
    capacity_pages: int,
) -> tuple[int, ...]:
    """Normalize physical page IDs: deduplicate in order, then bounds-check.

    ``dict.fromkeys`` preserves first-seen order while removing duplicates
    (dicts have been insertion-ordered since 3.7), so zeroing a page once is
    enough even if the caller repeats it, and the resulting order stays
    reproducible -- which a ``set`` would not give.
    """
    page_ids = tuple(dict.fromkeys(physical_pages))
    for page in page_ids:
        if not isinstance(page, int) or isinstance(page, bool):
            raise SlotAddressError(f"physical page IDs must be integers, got {page!r}")
        if not 0 <= page < capacity_pages:
            raise SlotAddressError(
                f"physical page ID {page} is outside storage (capacity {capacity_pages} pages)"
            )
    return page_ids


def page_index(torch: Any, page_ids: Sequence[int], *, device: Any) -> Any:
    """Build an ``int64`` device tensor from page IDs already validated on the host."""
    return _tensor_from_buffer(
        torch,
        host_index_buffer(page_ids, capacity_slots=1 << 62, require_unique=False),
        device=device,
    )


# ---------------------------------------------------------------------------
# Scatter / gather / fill, with the FP8 workaround applied uniformly.
#
# PyTorch ships no ``index_copy_`` or ``index_fill_`` kernel for
# ``float8_e4m3fn`` / ``float8_e5m2`` on *either* CPU or CUDA. Routing the
# workaround by device would look tidier and would be a trap: it raises
# ``NotImplementedError`` the moment a device clears the sm_89 floor and a real
# FP8 cache is written on it. So the rule is by dtype, everywhere.
#
# ``Tensor.view(torch.uint8)`` between equal-itemsize dtypes is a pure bitcast:
# same storage, same shape, same strides, not one bit changed. It requires unit
# stride in the last dimension, which every plane here satisfies because the
# allocations are contiguous and the reshapes that precede these calls are
# zero-copy views.
# ---------------------------------------------------------------------------
def _extract_index_tensor(indices: Any) -> Any:
    """Extract device tensor from SlotIndex or raw tensor."""
    if isinstance(indices, SlotIndex):
        return indices.tensor
    return indices


def index_copy_storage(torch: Any, target: Any, indices: Any, values: Any) -> None:
    """Scatter ``values`` into ``target`` rows addressed by ``indices``.

    ``values`` is converted to the target's dtype and device first; the caller
    is responsible for having applied any quantization scale beforehand.
    """
    idx_tensor = _extract_index_tensor(indices)
    if isinstance(indices, SlotIndex):
        indices.require_unique()

    if is_fp8_storage_dtype(dtype_name(target.dtype)):
        # 1. Move to device first to prevent NotImplementedError when casting FP8 on CPU
        val = values.to(device=target.device)

        # 2. Only cast if values isn't already raw bytes (uint8) or the exact target dtype
        if val.dtype != torch.uint8 and val.dtype != target.dtype:
            val = val.to(dtype=target.dtype)

        # 3. Ensure contiguous memory layout before bitcasting to uint8
        val_bytes = val.contiguous().view(torch.uint8)
        target.view(torch.uint8).index_copy_(0, idx_tensor, val_bytes)
        return

    converted = values.to(device=target.device).to(dtype=target.dtype)
    target.index_copy_(0, idx_tensor, converted)


def index_select_storage(torch: Any, source: Any, indices: Any) -> Any:
    """Gather rows from ``source`` at ``indices``.

    ``index_select`` *does* have FP8 kernels, so the ``uint8`` detour here is
    for symmetry rather than necessity. Keeping both directions on one rule
    means the FP8 gather cannot quietly start depending on which device it runs
    on, which is the failure mode the write path already has.
    """
    idx_tensor = _extract_index_tensor(indices)
    if is_fp8_storage_dtype(dtype_name(source.dtype)):
        return (
            source.view(torch.uint8)
            .index_select(0, idx_tensor)
            .view(source.dtype)
        )
    return source.index_select(0, idx_tensor)


def index_fill_storage(torch: Any, target: Any, dim: int, indices: Any) -> None:
    """Zero whole slices of ``target`` along ``dim`` at ``indices``.

    One kernel launch per allocation. The version this replaces looped in
    Python over layers and pages calling ``tensor[page].zero_()``, which is
    ``2 * num_layers * num_pages`` launches -- 3072 for 24 layers and 64 pages,
    or roughly 15-30 ms of pure launch overhead for an operation that moves
    12 MB.

    Zeroing an FP8 buffer through a ``uint8`` view is exact: the all-zero byte
    is ``+0.0`` in both ``e4m3fn`` and ``e5m2``.
    """
    idx_tensor = _extract_index_tensor(indices)
    if is_fp8_storage_dtype(dtype_name(target.dtype)):
        target.view(torch.uint8).index_fill_(dim, idx_tensor, 0)
        return
    target.index_fill_(dim, idx_tensor, 0)
