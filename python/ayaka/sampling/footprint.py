"""Memory footprint accounting and ledger charging for sampling buffers.

Provides measurement utilities and ledger reservation helpers to track memory
allocations across accelerator device memory, page-locked (pinned) host memory,
and standard pageable host memory.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch

from ayaka.memory.ledger import MemoryLedger, Reservation
from ayaka.types import MemoryOwner, MemoryTier


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
