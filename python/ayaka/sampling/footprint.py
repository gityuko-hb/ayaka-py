from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch

from ayaka.memory.ledger import MemoryLedger, Reservation
from ayaka.types import MemoryOwner, MemoryTier


@dataclass(frozen=True, slots=True)
class SamplingFootprint:
    device_bytes: int = 0
    host_pinned_bytes: int = 0
    host_pageable_bytes: int = 0

    def __post_init__(self) -> None:
        for name in ("device_bytes", "host_pinned_bytes", "host_pageable_bytes"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

    def __add__(self, other: SamplingFootprint) -> SamplingFootprint:
        return SamplingFootprint(
            device_bytes=self.device_bytes + other.device_bytes,
            host_pinned_bytes=self.host_pinned_bytes + other.host_pinned_bytes,
            host_pageable_bytes=self.host_pageable_bytes + other.host_pageable_bytes,
        )

    @property
    def host_bytes(self) -> int:
        return self.host_pinned_bytes + self.host_pageable_bytes

    @property
    def pinning_succeeded(self) -> bool:
        return self.host_pageable_bytes == 0


def measure(tensors: Iterable[torch.Tensor]) -> SamplingFootprint:
    device = pinned = pageable = 0
    for tensor in tensors:
        nbytes = tensor.numel() * tensor.element_size()
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
    written: list[str] = []
    for tier, nbytes, suffix in (
        (MemoryTier.DEVICE, footprint.device_bytes, "device"),
        (MemoryTier.HOST_PINNED, footprint.host_pinned_bytes, "host_pinned"),
        (MemoryTier.HOST_PAGEABLE, footprint.host_pageable_bytes, "host_pageable"),
    ):
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
