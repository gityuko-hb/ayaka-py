from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


def _optional_torch() -> Any | None:
    """Return torch when importable, else None; multi-GPU code stays testable."""
    try:
        import torch
    except ImportError:  # pragma: no cover - exercised on torch-free hosts
        return None
    return torch


def _is_cuda_device(device: str) -> bool:
    return str(device).startswith("cuda")


@dataclass(frozen=True, slots=True)
class PeerTopology:
    """Measured peer-access map over a set of devices."""

    devices: tuple[str, ...]
    peer_capable: tuple[tuple[bool, ...], ...]
    """``peer_capable[i][j]`` is True when rank i can directly access rank j."""

    def __post_init__(self) -> None:
        size = len(self.devices)
        if len(self.peer_capable) != size or any(len(row) != size for row in self.peer_capable):
            raise ValueError("peer_capable must be a square matrix over devices")

    @property
    def world_size(self) -> int:
        return len(self.devices)

    @property
    def peer_available(self) -> bool:
        """Whether any ordered pair of distinct ranks can peer-access."""
        return any(
            self.peer_capable[i][j]
            for i in range(self.world_size)
            for j in range(self.world_size)
            if i != j
        )

    def can_access(self, source: int, destination: int) -> bool:
        """Whether ``source`` may directly read/write ``destination`` memory."""
        return self.peer_capable[source][destination]

    @classmethod
    def probe(cls, devices: Sequence[str]) -> PeerTopology:
        """Measure peer access for every ordered pair of ``devices``."""
        resolved = tuple(str(device) for device in devices)
        torch = _optional_torch()
        size = len(resolved)
        if torch is None or not torch.cuda.is_available():
            return cls(devices=resolved, peer_capable=tuple((False,) * size for _ in range(size)))

        def index_of(device: str) -> int:
            parsed = torch.device(device)
            return 0 if parsed.index is None else int(parsed.index)

        rows: list[tuple[bool, ...]] = []
        for i in range(size):
            row: list[bool] = []
            for j in range(size):
                if i == j or not (_is_cuda_device(resolved[i]) and _is_cuda_device(resolved[j])):
                    row.append(False)
                    continue
                row.append(
                    bool(
                        torch.cuda.can_device_access_peer(
                            index_of(resolved[i]), index_of(resolved[j])
                        )
                    )
                )
            rows.append(tuple(row))
        return cls(devices=resolved, peer_capable=tuple(rows))
