"""Physical step buffers with one ledger charge per allocation.

Views are borrowed: consumers must finish before the owner closes the buffer.
Allocation performs no tensor initialization or transfer and enqueues no work.
"""

from __future__ import annotations

from itertools import count
from typing import Any

from ayaka.memory.ledger import MemoryLedger, Reservation
from ayaka.memory.region import MemoryRegion
from ayaka.plan import WorkspaceRequest
from ayaka.types import MemoryTier
from ayaka.utils.torch_utils import require_torch

_BUFFER_IDS = count()


class BufferLease:
    """Own an aligned byte view, its backing tensor, and its ledger claim."""

    def __init__(
        self, ledger: MemoryLedger, label: str, tensor: Any, view: Any, region: MemoryRegion
    ) -> None:
        self._ledger = ledger
        self.label = label
        self._tensor = tensor
        self._view = view
        self.region = region
        self.closed = False

    @property
    def tensor(self) -> Any:
        """Borrow a byte tensor until the containing execution ticket retires."""
        if self.closed:
            raise RuntimeError("buffer lease is closed")
        return self._view

    def close(self) -> None:
        """Drop physical ownership after proven last use; idempotent."""
        if self.closed:
            return
        if self._ledger.get(self.label) is None:
            raise RuntimeError("buffer ledger claim disappeared")
        self._view = None
        self._tensor = None
        self._ledger.release(self.label)
        self.closed = True


class BufferAllocator:
    """Allocate CUDA, pinned-host, or pageable-host storage after reserving capacity.

    Zero-byte requests need no allocation or charge. Alignment slack is charged
    once, even when the returned view is smaller than the backing allocation.
    """

    def __init__(self, ledger: MemoryLedger) -> None:
        self.ledger = ledger

    def allocate(self, request: WorkspaceRequest, *, label: str) -> BufferLease | None:
        if request.tier not in (
            MemoryTier.DEVICE,
            MemoryTier.HOST_PINNED,
            MemoryTier.HOST_PAGEABLE,
        ):
            raise ValueError("step buffers require device or host memory")
        if not request.nbytes:
            return None
        size = request.nbytes + request.alignment - 1
        ticket = self.ledger.reserve(
            Reservation(
                request.owner,
                label,
                size,
                0,
                tier=request.tier,
                device_index=self.ledger.device_index,
                charged_bytes=size,
            )
        )
        tensor = view = None
        try:
            torch = require_torch(capability="runtime buffers")
            tensor = torch.empty(
                size,
                dtype=torch.uint8,
                device=(
                    f"cuda:{self.ledger.device_index}"
                    if request.tier is MemoryTier.DEVICE
                    else "cpu"
                ),
                pin_memory=request.tier is MemoryTier.HOST_PINNED,
            )
            offset = (-tensor.data_ptr()) % request.alignment
            view = tensor[offset : offset + request.nbytes]
            region = MemoryRegion(
                region_id=next(_BUFFER_IDS),
                base_ptr=view.data_ptr(),
                nbytes=request.nbytes,
                tier=request.tier,
                owner=request.owner,
                device_index=(
                    self.ledger.device_index if request.tier is MemoryTier.DEVICE else -1
                ),
                alignment=request.alignment,
            )
            self.ledger.materialize(ticket, actual_bytes=size)
            self.ledger.commit(ticket)
        except BaseException:
            view = tensor = None
            self.ledger.rollback(ticket)
            raise
        return BufferLease(self.ledger, label, tensor, view, region)
