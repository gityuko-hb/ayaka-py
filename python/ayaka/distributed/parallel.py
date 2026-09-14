"""Explicit tensor/expert parallel views over runtime-owned communicators."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch

from ayaka.distributed.device import CommunicationBackend, DeviceGroup
from ayaka.utils.import_utils import CapabilityError
from ayaka.utils.validation import require_int


class ParallelCollectiveUnavailable(CapabilityError):
    def __init__(self, detail: str) -> None:
        super().__init__("layers.parallel", detail=detail, remedy="supply a communication backend")


class ParallelContext(Protocol):
    @property
    def rank(self) -> int: ...

    @property
    def world_size(self) -> int: ...

    def split_last_dim(self, tensor: torch.Tensor) -> torch.Tensor: ...

    def all_gather_last_dim(self, tensor: torch.Tensor) -> torch.Tensor: ...

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor: ...


def divide(value: int, divisor: int, *, name: str = "value") -> int:
    require_int(value, name)
    require_int(divisor, "divisor", minimum=1)
    if value % divisor:
        raise ValueError(f"{name}={value} must be divisible by {divisor}")
    return value // divisor


@dataclass(frozen=True)
class VirtualParallelContext:
    """Explicit rank simulation for tests; multi-rank collectives require callbacks."""

    rank: int = 0
    world_size: int = 1
    all_gather_fn: Callable[[torch.Tensor], torch.Tensor] | None = None
    all_reduce_fn: Callable[[torch.Tensor], torch.Tensor] | None = None

    def __post_init__(self) -> None:
        require_int(self.world_size, "world_size", minimum=1)
        require_int(self.rank, "rank")
        if self.rank >= self.world_size:
            raise ValueError("rank must be smaller than world_size")

    def split_last_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        width = divide(tensor.shape[-1], self.world_size, name="input width")
        return tensor.narrow(-1, self.rank * width, width).contiguous()

    def all_gather_last_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.world_size == 1:
            return tensor
        if self.all_gather_fn is None:
            raise ParallelCollectiveUnavailable("all_gather requires an explicit communicator")
        return self.all_gather_fn(tensor)

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.world_size == 1:
            return tensor
        if self.all_reduce_fn is None:
            raise ParallelCollectiveUnavailable("all_reduce requires an explicit communicator")
        return self.all_reduce_fn(tensor)


class LocalParallelContext(VirtualParallelContext):
    def __init__(self) -> None:
        super().__init__()


class RuntimeParallelContext:
    """Borrow a DeviceGroup and CommunicationBackend without owning their lifecycle.

    Collectives complete on the current stream. The gather output port is a tensor
    with shape ``[group.size, *input.shape]`` in group-rank order. This adapter
    returns a concatenation along the input's last dimension. Reductions may
    mutate the supplied tensor; callers must pass their owned GEMM output.
    """

    def __init__(self, group: DeviceGroup, communication: CommunicationBackend | None) -> None:
        if group.size < 1 or len(set(group.ranks)) != len(group.ranks):
            raise ValueError("parallel group must be nonempty with unique ranks")
        self.group = group
        self.communication = communication

    @property
    def rank(self) -> int:
        return self.group.local_rank

    @property
    def world_size(self) -> int:
        return self.group.size

    def _validate(self, tensor: torch.Tensor) -> None:
        local = self.group.devices[self.rank]
        if tensor.device.type != local.kind.value or (
            tensor.device.type != "cpu" and tensor.device.index != local.index
        ):
            raise ValueError(f"tensor device {tensor.device} does not match group device {local}")

    def split_last_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        self._validate(tensor)
        width = divide(tensor.shape[-1], self.world_size, name="input width")
        return tensor.narrow(-1, self.rank * width, width).contiguous()

    def all_gather_last_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        self._validate(tensor)
        if self.world_size == 1:
            return tensor
        if self.communication is None:
            raise ParallelCollectiveUnavailable("all_gather requires a communication backend")
        source = tensor.contiguous()
        output = torch.empty(
            (self.world_size, *source.shape), dtype=source.dtype, device=source.device
        )
        handle = self.communication.all_gather(output, source, self.group, async_op=False)
        if handle is not None:
            handle.wait()
        return torch.cat(output.unbind(0), dim=-1)

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        self._validate(tensor)
        if self.world_size == 1:
            return tensor
        if self.communication is None:
            raise ParallelCollectiveUnavailable("all_reduce requires a communication backend")
        output = tensor.contiguous()
        handle = self.communication.all_reduce(output, self.group, async_op=False)
        if handle is not None:
            handle.wait()
        return output


def get_default_parallel_context() -> LocalParallelContext:
    """No supplied group means singleton execution, even if distributed is initialized."""
    return LocalParallelContext()
