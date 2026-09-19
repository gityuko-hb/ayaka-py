"""Runtime KV cache resize: cost math, budget fit-check, rejection before free.

A resize decision is a two-phase protocol:

1. :func:`validate_resize` predicts the byte cost of the requested geometry and
   compares it with the headroom that actually exists while the old cache is
   still alive. On failure it raises :class:`CacheRebuildRejected` and touches
   nothing: the old caches keep serving, so the rejection is recoverable.
2. Only after a plan passes does the owning runtime run the destructive
   free-and-materialize step -- either alongside the old slab (``SWAP``) or
   after tearing it down (``REBUILD``).

Every storage family contributes its own cost through
:meth:`~ayaka.kvcache.storage.geometry.BaseKVStorageSpec.aligned_total_bytes`,
so the fit-check is one comparison over the planned per-group specs. Nothing in
this module allocates, frees or mutates a cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ayaka.exceptions import RuntimeMemoryError
from ayaka.kvcache.storage.geometry import BaseKVStorageSpec
from ayaka.kvcache.storage.layout import DEFAULT_KV_ALIGNMENT_BYTES
from ayaka.utils.torch_memory import device_memory

__all__ = [
    "LEDGER_OVERHEAD_BYTES",
    "CacheRebuildRejected",
    "CacheResizeFatal",
    "CacheResizeMode",
    "ResizePlan",
    "ResizeRejectionReason",
    "device_headroom",
    "minimum_pages",
    "storage_bytes",
    "validate_resize",
]

#: Bytes the memory ledger reserves on top of the aligned slab (bookkeeping).
LEDGER_OVERHEAD_BYTES = 1 << 20


class ResizeRejectionReason(StrEnum):
    """Why a resize was refused, with its recovery semantics.

    ``BUSY``, ``BUDGET``, ``INVALID_PAGES`` and ``UNSUPPORTED`` are all decided
    *before* any destructive free, so the old caches are intact and serving
    continues. ``ALLOCATION_FAILED`` is post-free but the runtime restored the
    previous geometry (or the old slab was never touched in ``SWAP`` mode).
    """

    BUSY = "busy"
    BUDGET = "budget"
    INVALID_PAGES = "invalid_pages"
    UNSUPPORTED = "unsupported"
    ALLOCATION_FAILED = "allocation_failed"


class CacheResizeMode(StrEnum):
    """How a validated resize may be executed safely.

    ``SWAP`` means the new slab fits in current headroom, so it is materialized
    before the old one is released and the destructive window is zero.
    ``REBUILD`` means headroom plus the old slab covers the new one: the old
    cache must be freed first, and the runtime is responsible for restoring it
    if the new materialization fails.
    """

    SWAP = "swap"
    REBUILD = "rebuild"


class CacheRebuildRejected(RuntimeMemoryError):
    """A runtime cache rebuild was rejected before any destructive free.

    The old caches are intact and serving continues -- this is recoverable.
    """

    def __init__(
        self,
        message: str,
        *,
        reason: ResizeRejectionReason,
        requested_pages: int | None = None,
        need_bytes: int | None = None,
        available_bytes: int | None = None,
        old_bytes: int | None = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.requested_pages = requested_pages
        self.need_bytes = need_bytes
        self.available_bytes = available_bytes
        self.old_bytes = old_bytes

    def describe(self) -> str:
        """One-line summary with the numbers a caller or UI can act on."""
        parts = [f"reason={self.reason.value}"]
        if self.requested_pages is not None:
            parts.append(f"pages={self.requested_pages}")
        if self.need_bytes is not None:
            parts.append(f"need={_mib(self.need_bytes)} MiB")
        if self.available_bytes is not None:
            parts.append(f"available={_mib(self.available_bytes)} MiB")
        if self.old_bytes is not None:
            parts.append(f"old={_mib(self.old_bytes)} MiB")
        return "; ".join(parts)


class CacheResizeFatal(RuntimeMemoryError):
    """Recovery failed after a destructive free; the engine must stop.

    Raised only when the old slab was already released, the new materialization
    failed, and restoring the previous geometry failed as well. The runtime
    surfaces this as an unavailable engine rather than continuing with a cache
    that does not match its allocator.
    """


@dataclass(frozen=True, slots=True)
class ResizePlan:
    """A validated, non-destructive description of one resize.

    Attributes:
        mode: ``SWAP`` when the new slab fits in current headroom, ``REBUILD``
            when the old slab must be released to make room.
        pages: Requested page capacity, known to satisfy the model-context
            floor.
        spec: Geometry of the new slab.
        current_pages: Capacity of the slab being replaced.
        need_bytes: Total device bytes the new slab (plus ledger overhead)
            requires.
        old_bytes: Total device bytes the old slab (plus overhead) holds.
        headroom_bytes: Measured bytes available at check time without
            releasing the old slab.
        safety_bytes: Reserve kept free during the operation.
    """

    mode: CacheResizeMode
    pages: int
    spec: BaseKVStorageSpec
    current_pages: int
    need_bytes: int
    old_bytes: int
    headroom_bytes: int
    safety_bytes: int

    @property
    def releases_old_first(self) -> bool:
        """Whether executing this plan opens a destructive window."""
        return self.mode is CacheResizeMode.REBUILD


def storage_bytes(
    spec: BaseKVStorageSpec,
    pages: int,
    *,
    alignment_bytes: int = DEFAULT_KV_ALIGNMENT_BYTES,
    ledger_overhead_bytes: int = LEDGER_OVERHEAD_BYTES,
) -> int:
    """Total device bytes a storage family occupies at ``pages`` capacity.

    Uses the same aligned byte math the materializer charges, so the fit-check
    and the eventual allocation cannot disagree about geometry.

    Args:
        spec: Family geometry, with ``capacity_pages`` ignored.
        pages: Requested page capacity.
        alignment_bytes: Per-allocation alignment used by the materializer.
        ledger_overhead_bytes: Fixed bookkeeping bytes charged on top.

    Raises:
        TypeError: If ``pages`` is not an integer.
        ValueError: If ``pages`` or ``alignment_bytes`` is not positive.
    """
    if not isinstance(pages, int) or isinstance(pages, bool):
        raise TypeError("pages must be an integer")
    if pages <= 0:
        raise ValueError("pages must be positive")
    if alignment_bytes <= 0:
        raise ValueError("alignment_bytes must be positive")
    if ledger_overhead_bytes < 0:
        raise ValueError("ledger_overhead_bytes must be non-negative")
    return spec.with_capacity_pages(pages).aligned_total_bytes(alignment_bytes) + (
        ledger_overhead_bytes
    )


def minimum_pages(page_size: int, max_sequence_tokens: int | None) -> int:
    """Smallest capacity that holds at least one full context plus padding.

    Mirrors the startup guard in :class:`~ayaka.runtime.serving.ServingRuntime`
    (``(pages - 1) * page_size >= max_sequence_tokens``).

    Args:
        page_size: Tokens per page.
        max_sequence_tokens: Model context length, or None when unbounded.

    Raises:
        ValueError: If ``page_size`` is not positive.
    """
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    if max_sequence_tokens is None:
        return 2
    context_pages = (max_sequence_tokens + page_size - 1) // page_size
    return max(2, context_pages + 1)


def device_headroom(device: object = None) -> int:
    """Bytes a device can still hand out, including cached-but-free blocks.

    Uses the caching allocator's own numbers rather than a policy fraction:
    ``driver_free`` plus ``reserved - allocated`` is exactly what can be
    allocated without releasing any live tensor.

    Raises:
        CapabilityError: If the resolved device is not CUDA. Callers that can
            run without a device budget (diagnostics, tests) must inject an
            explicit ``headroom_bytes`` instead.
    """
    memory = device_memory(device)
    return memory.driver_free + memory.allocator_overhead


def validate_resize(
    *,
    spec: BaseKVStorageSpec,
    requested_pages: int,
    page_size: int | None = None,
    max_sequence_tokens: int | None = None,
    headroom_bytes: int | None = None,
    device: object = None,
    safety_bytes: int = 0,
    kv_budget_bytes: int | None = None,
    alignment_bytes: int = DEFAULT_KV_ALIGNMENT_BYTES,
    ledger_overhead_bytes: int = LEDGER_OVERHEAD_BYTES,
) -> ResizePlan:
    """Check a requested capacity against the device budget without freeing.

    The function is side-effect free: it either returns a :class:`ResizePlan`
    the caller may execute, or raises :class:`CacheRebuildRejected` while every
    existing cache remains allocated and valid.

    Args:
        spec: Current storage geometry; ``spec.capacity_pages`` is the capacity
            being replaced.
        requested_pages: Desired page capacity.
        page_size: Tokens per page; defaults to ``spec.page_size``.
        max_sequence_tokens: Model context length used for the floor check.
        headroom_bytes: Measured free bytes; when None, queried from the device.
        device: Device passed to :func:`device_headroom`.
        safety_bytes: Reserve that must stay free after the operation.
        kv_budget_bytes: Frozen KV budget; when set, a request whose aligned
            cost exceeds it is rejected even if transient headroom allows it.
        alignment_bytes: Alignment used by the materializer.
        ledger_overhead_bytes: Fixed bookkeeping bytes charged on top.

    Returns:
        A plan whose ``mode`` says whether the new slab can be materialized
        before the old one is released.

    Raises:
        TypeError: If ``requested_pages`` is not an integer.
        ValueError: On negative ``safety_bytes``.
        CacheRebuildRejected: ``INVALID_PAGES`` below the context floor,
            ``UNSUPPORTED`` when headroom cannot be measured, or ``BUDGET``
            when even releasing the old slab (or the frozen budget) cannot fit
            the request.
    """
    if not isinstance(requested_pages, int) or isinstance(requested_pages, bool):
        raise TypeError("requested_pages must be an integer")
    if safety_bytes < 0:
        raise ValueError("safety_bytes must be non-negative")
    resolved_page_size = spec.page_size if page_size is None else page_size
    floor = minimum_pages(resolved_page_size, max_sequence_tokens)
    if requested_pages < floor:
        raise CacheRebuildRejected(
            f"KV cache resize to {requested_pages} pages is below the {floor}-page "
            f"context floor; old cache kept",
            reason=ResizeRejectionReason.INVALID_PAGES,
            requested_pages=requested_pages,
        )
    if headroom_bytes is None:
        try:
            headroom_bytes = device_headroom(device)
        except Exception as exc:
            raise CacheRebuildRejected(
                "KV cache resize requires a measurable device budget; old cache kept",
                reason=ResizeRejectionReason.UNSUPPORTED,
                requested_pages=requested_pages,
            ) from exc
    if headroom_bytes < 0:
        raise ValueError("headroom_bytes must be non-negative")

    current_pages = spec.capacity_pages
    need = storage_bytes(
        spec,
        requested_pages,
        alignment_bytes=alignment_bytes,
        ledger_overhead_bytes=ledger_overhead_bytes,
    )
    old = storage_bytes(
        spec,
        current_pages,
        alignment_bytes=alignment_bytes,
        ledger_overhead_bytes=ledger_overhead_bytes,
    )
    if kv_budget_bytes is not None:
        if kv_budget_bytes < 0:
            raise ValueError("kv_budget_bytes must be non-negative")
        if need > kv_budget_bytes:
            raise CacheRebuildRejected(
                f"KV cache resize to {requested_pages} pages needs {_mib(need)} MiB but the "
                f"frozen KV budget is {_mib(kv_budget_bytes)} MiB; old cache kept",
                reason=ResizeRejectionReason.BUDGET,
                requested_pages=requested_pages,
                need_bytes=need,
                available_bytes=kv_budget_bytes,
                old_bytes=old,
            )
    available = headroom_bytes + old - safety_bytes
    if need > available:
        raise CacheRebuildRejected(
            f"KV cache resize to {requested_pages} pages needs {_mib(need)} MiB but only "
            f"{_mib(available)} MiB can be made available; old cache kept",
            reason=ResizeRejectionReason.BUDGET,
            requested_pages=requested_pages,
            need_bytes=need,
            available_bytes=available,
            old_bytes=old,
        )
    mode = (
        CacheResizeMode.SWAP if need <= headroom_bytes - safety_bytes else CacheResizeMode.REBUILD
    )
    return ResizePlan(
        mode=mode,
        pages=requested_pages,
        spec=spec.with_capacity_pages(requested_pages),
        current_pages=current_pages,
        need_bytes=need,
        old_bytes=old,
        headroom_bytes=headroom_bytes,
        safety_bytes=safety_bytes,
    )


def _mib(value: int) -> str:
    """Format bytes as MiB with one decimal for rejection messages."""
    return f"{value / (1 << 20):.1f}"
