"""PyTorch communication using explicit, externally owned process groups."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.distributed as dist

from ayaka.distributed.device import AsyncHandle, CommOpType, DeviceGroup
from ayaka.distributed.parallel import RuntimeParallelContext


class TorchWorkHandle:
    """Adapt torch Work.wait() to Ayaka's completion contract and retain buffers."""

    def __init__(self, work: Any, buffers: tuple[torch.Tensor, ...]) -> None:
        self._work = work
        self._buffers = buffers

    def wait(self) -> None:
        if self._work.wait() is False:
            raise RuntimeError("distributed work did not complete successfully")

    def is_completed(self) -> bool:
        return bool(self._work.is_completed())


def _handle(work: Any, *buffers: torch.Tensor) -> AsyncHandle | None:
    return None if work is None else TorchWorkHandle(work, buffers)


class TorchCommunicationBackend:
    """Implement CommunicationBackend without initializing or destroying groups.

    Pass a mapping from each logical DeviceGroup to an initialized torch group.
    WORLD is allowed only when supplied explicitly. Collective buffers must be
    contiguous and remain alive until asynchronous handles have been joined.
    Gather/scatter buffers use ``[group.size, *local_tensor.shape]`` rank order.
    Broadcast/send/recv rank arguments are indices within the logical group.
    """

    name = "torch.distributed"

    def __init__(self, groups: Mapping[DeviceGroup, Any]) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized by the runtime")
        self._groups = dict(groups)
        for group, process_group in self._groups.items():
            if process_group is None or not group.ranks or group.size < 1:
                raise ValueError(
                    "explicit process groups and nonempty global rank lists are required"
                )
            if len(set(group.ranks)) != group.size:
                raise ValueError("group ranks must be unique")
            if tuple(dist.get_process_group_ranks(process_group)) != group.ranks:
                raise ValueError("DeviceGroup rank order differs from the torch process group")
            if dist.get_rank(process_group) != group.local_rank:
                raise ValueError("DeviceGroup.local_rank differs from the torch process group")

    def _group(self, group: DeviceGroup, *tensors: torch.Tensor) -> Any:
        if group not in self._groups:
            raise ValueError(f"unregistered communication group {group.name!r}")
        local = group.devices[group.local_rank]
        for tensor in tensors:
            if not tensor.is_contiguous():
                raise ValueError("collective tensors must be contiguous")
            if tensor.device.type != local.kind.value or (
                tensor.device.type != "cpu" and tensor.device.index != local.index
            ):
                raise ValueError("collective tensor placement differs from its group")
        return self._groups[group]

    @staticmethod
    def _op(op: CommOpType) -> Any:
        return {
            CommOpType.SUM: dist.ReduceOp.SUM,
            CommOpType.MAX: dist.ReduceOp.MAX,
            CommOpType.MIN: dist.ReduceOp.MIN,
            CommOpType.AVG: dist.ReduceOp.AVG,
            CommOpType.PROD: dist.ReduceOp.PRODUCT,
        }[op]

    @staticmethod
    def _rank(group: DeviceGroup, rank: int) -> int:
        if not 0 <= rank < group.size:
            raise ValueError("peer rank is outside the supplied group")
        return group.ranks[rank]

    def all_reduce(
        self,
        tensor: Any,
        group: DeviceGroup,
        op: CommOpType = CommOpType.SUM,
        async_op: bool = False,
    ) -> AsyncHandle | None:
        return _handle(
            dist.all_reduce(
                tensor, op=self._op(op), group=self._group(group, tensor), async_op=async_op
            ),
            tensor,
        )

    def all_gather(
        self, output: Any, tensor: Any, group: DeviceGroup, async_op: bool = False
    ) -> AsyncHandle | None:
        pg = self._group(group, output, tensor)
        if output.shape != (group.size, *tensor.shape) or output.dtype != tensor.dtype:
            raise ValueError(
                "all_gather output must be [group.size, *input.shape] with matching dtype"
            )
        return _handle(
            dist.all_gather(list(output.unbind(0)), tensor, group=pg, async_op=async_op),
            output,
            tensor,
        )

    def reduce_scatter(
        self,
        output: Any,
        tensor: Any,
        group: DeviceGroup,
        op: CommOpType = CommOpType.SUM,
        async_op: bool = False,
    ) -> AsyncHandle | None:
        pg = self._group(group, output, tensor)
        if tensor.shape != (group.size, *output.shape) or output.dtype != tensor.dtype:
            raise ValueError("reduce_scatter input must be [group.size, *output.shape]")
        return _handle(
            dist.reduce_scatter(
                output, list(tensor.unbind(0)), op=self._op(op), group=pg, async_op=async_op
            ),
            output,
            tensor,
        )

    def broadcast(
        self, tensor: Any, group: DeviceGroup, src_rank: int = 0, async_op: bool = False
    ) -> AsyncHandle | None:
        return _handle(
            dist.broadcast(
                tensor,
                src=self._rank(group, src_rank),
                group=self._group(group, tensor),
                async_op=async_op,
            ),
            tensor,
        )

    def all_to_all(
        self, output: Any, tensor: Any, group: DeviceGroup, async_op: bool = False
    ) -> AsyncHandle | None:
        pg = self._group(group, output, tensor)
        if (
            output.shape != tensor.shape
            or tensor.shape[0] % group.size
            or output.dtype != tensor.dtype
        ):
            raise ValueError(
                "all_to_all requires equally shaped buffers divisible along dimension 0"
            )
        return _handle(
            dist.all_to_all_single(output, tensor, group=pg, async_op=async_op), output, tensor
        )

    def send(self, tensor: Any, group: DeviceGroup, dst_rank: int) -> AsyncHandle | None:
        return _handle(
            dist.isend(tensor, dst=self._rank(group, dst_rank), group=self._group(group, tensor)),
            tensor,
        )

    def recv(self, tensor: Any, group: DeviceGroup, src_rank: int) -> AsyncHandle | None:
        return _handle(
            dist.irecv(tensor, src=self._rank(group, src_rank), group=self._group(group, tensor)),
            tensor,
        )

    def barrier(self, group: DeviceGroup) -> None:
        dist.barrier(group=self._group(group))


class TorchDistributedParallelContext(RuntimeParallelContext):
    """Convenience view with an explicit logical group and torch process group."""

    def __init__(self, group: DeviceGroup, process_group: Any) -> None:
        super().__init__(group, TorchCommunicationBackend({group: process_group}))
