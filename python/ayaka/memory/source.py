"""Raw byte sources for the caching allocator.

A source is the layer that actually asks the driver for memory.  The caching
allocator above it decides *when* to ask and when to reuse a block; the source
decides nothing, which is what makes it replaceable — torch's caching allocator
today, a raw ``cudaMalloc``/``cuMemMap`` arena later.

The one contract that matters upward: ``alloc`` must return a base pointer
aligned to at least :data:`SOURCE_ALIGNMENT`.  The caching allocator sizes its
over-allocation on that promise when a caller asks for a stronger boundary, and
it re-checks every fresh pointer rather than trusting this module.  A source
that cannot honour the floor fails closed.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ayaka.exceptions import RuntimeMemoryError
from ayaka.utils.import_utils import CapabilityError
from ayaka.utils.torch_memory import empty_host_tensor
from ayaka.utils.torch_utils import require_torch

__all__ = [
    "SOURCE_ALIGNMENT",
    "RawMemorySource",
    "TorchDeviceSource",
    "TorchHostByteSource",
]

SOURCE_ALIGNMENT = 256
"""Minimum base-pointer alignment every source must guarantee.

256 is the tensor-core floor and comfortably below what torch's own allocators
return in practice; raising it would turn every alignment request into
over-allocation without buying anything a kernel needs.
"""


@runtime_checkable
class RawMemorySource(Protocol):
    """Allocates raw bytes and reports the handle that releases them.

    ``alloc`` returns ``(keepalive, base_ptr)``.  The keepalive is deliberately
    opaque: for a torch source it is the tensor whose lifetime owns the block,
    for a raw CUDA source it could be a generic allocation handle, and the
    caching allocator only ever hands it back to :meth:`free`.
    """

    @property
    def name(self) -> str:
        """Stable identifier used in ledger labels and error messages."""
        ...

    def alloc(self, nbytes: int) -> tuple[Any, int]:
        """Take at least ``nbytes`` from the source and return how to free it."""
        ...

    def free(self, keepalive: Any) -> None:
        """Release one block previously returned by :meth:`alloc`."""
        ...


def _checked_base(tensor: Any, *, source: str) -> int:
    """Return ``tensor``'s data pointer, refusing one that breaks the contract.

    The tensor is dropped by the caller; the check exists because the caching
    allocator already sized a block on the alignment promise, so a violation
    must surface here and not as a misaligned write three layers up.
    """
    base_ptr = int(tensor.data_ptr())
    if base_ptr % SOURCE_ALIGNMENT:
        raise RuntimeMemoryError(
            f"{source} returned {base_ptr:#x}, not aligned to the {SOURCE_ALIGNMENT}-byte "
            "floor every raw source must honour"
        )
    return base_ptr


class TorchDeviceSource:
    """Device bytes from torch's caching allocator.

    ``free`` drops the reference and nothing else.  That is the point of
    stacking on torch's arena: torch already splits and coalesces inside it,
    and a free here never calls ``cudaFree``, which would synchronise the
    device in the middle of a step.
    """

    __slots__ = ("_device", "_device_index", "_torch")

    def __init__(self, *, device_index: int = 0) -> None:
        self._torch = require_torch(capability="caching allocator device source")
        self._device_index = device_index
        self._device = f"cuda:{device_index}"

    @property
    def name(self) -> str:
        return f"torch_device:{self._device_index}"

    def alloc(self, nbytes: int) -> tuple[Any, int]:
        if nbytes < 1:
            raise RuntimeMemoryError("a source allocation must be positive")
        tensor = self._torch.empty(nbytes, dtype=self._torch.uint8, device=self._device)
        try:
            base_ptr = _checked_base(tensor, source=self.name)
        except RuntimeMemoryError:
            tensor = None
            raise
        return tensor, base_ptr

    def free(self, keepalive: Any) -> None:
        # Only the allocator's reference is dropped; torch reclaims the block
        # when it decides to, which is what keeps this path non-blocking.
        del keepalive


class TorchHostByteSource:
    """Host bytes, pinned or pageable, from torch.

    A pinned request that torch silently satisfies with pageable memory is
    refused, in the same spirit as ``TorchExactHostSource``: the caller that
    charged the ledger for pinned bytes must not discover later that DMA from
    the buffer is impossible.  ``numa_node`` is recorded for the caller's
    bookkeeping but not enforced — torch exposes no portable API for it, and
    the host policy (``ayaka.memory.host.host_policy``) is where pinning is
    gated.
    """

    __slots__ = ("_numa_node", "_pinned", "_torch")

    def __init__(self, *, pinned: bool = True, numa_node: int | None = None) -> None:
        self._torch = require_torch(capability="caching allocator host source")
        if pinned and not self._torch.cuda.is_available():
            raise RuntimeMemoryError(
                "pinned host memory needs a CUDA driver; refusing to substitute "
                "pageable memory for a pinned source"
            )
        self._pinned = pinned
        self._numa_node = numa_node

    @property
    def name(self) -> str:
        return "torch_host:pinned" if self._pinned else "torch_host:pageable"

    @property
    def numa_node(self) -> int | None:
        return self._numa_node

    def alloc(self, nbytes: int) -> tuple[Any, int]:
        if nbytes < 1:
            raise RuntimeMemoryError("a source allocation must be positive")
        # Torch's host allocators only guarantee natural alignment, so a raw
        # pageable/pinned block can start at a pointer that misses this
        # module's floor. Over-allocate one alignment slack and hand the
        # allocator an aligned view; the base tensor stays alive through the
        # view (``_base``) until the block is freed.
        slack = SOURCE_ALIGNMENT - 1
        try:
            tensor, _ = empty_host_tensor(
                (nbytes + slack,), "uint8", pinned=self._pinned, allow_pageable=False
            )
        except CapabilityError as exc:
            raise RuntimeMemoryError(
                f"host source {self.name} could not allocate {nbytes} B: {exc}"
            ) from exc
        offset = -int(tensor.data_ptr()) % SOURCE_ALIGNMENT
        aligned = tensor.narrow(0, offset, nbytes)
        try:
            base_ptr = _checked_base(aligned, source=self.name)
        except RuntimeMemoryError:
            tensor = None
            raise
        return aligned, base_ptr

    def free(self, keepalive: Any) -> None:
        del keepalive
