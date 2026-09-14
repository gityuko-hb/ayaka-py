"""Linear setup adapters; collective implementations live in distributed."""

from __future__ import annotations

from typing import Any

import torch

from ayaka.distributed.device import DeviceGroup, DeviceRef
from ayaka.distributed.parallel import (
    LocalParallelContext,
    ParallelCollectiveUnavailable,
    ParallelContext,
    RuntimeParallelContext,
    VirtualParallelContext,
    divide,
    get_default_parallel_context,
)
from ayaka.distributed.torch_backend import TorchDistributedParallelContext
from ayaka.types import DeviceKind


def prepare_parallel_runtime(
    device: torch.device | str | None,
    context: ParallelContext | None,
    runtime: dict[str, Any],
    *,
    disabled: bool = False,
    expert: bool = False,
) -> tuple[torch.device, ParallelContext, dict[str, Any]]:
    """Resolve placement and group before allocation; never inspect ambient WORLD."""
    runtime = dict(runtime)
    key = "ep_group" if expert else "tp_group"
    if disabled:
        runtime.pop(key, None)
        context = None
    group = runtime.get(key)
    if group is not None and (group.size < 1 or len(set(group.ranks)) != len(group.ranks)):
        raise ValueError("parallel group must be nonempty with unique ranks")
    if context is not None:
        if group is not None or runtime.get("communication") is not None:
            raise ValueError("supply parallel_context or runtime group/communication, not both")
        if isinstance(context, RuntimeParallelContext):
            group = context.group
            runtime[key] = group
            runtime["communication"] = context.communication
        elif not isinstance(context, VirtualParallelContext):
            raise TypeError("parallel_context must be a runtime or explicit virtual context")
    if device is None:
        device_context = runtime.get("device_context")
        if device_context is not None:
            ref = device_context.ref
            device = "cpu" if device_context.backend.name == "null" else str(ref)
        elif group is not None:
            device = str(group.devices[group.local_rank])
        else:
            device = torch.get_default_device()
    resolved = torch.device(device)
    if resolved.type == "cpu":
        resolved = torch.device("cpu")
    if resolved.type == "cuda" and resolved.index is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    if isinstance(context, VirtualParallelContext) and context.world_size > 1:
        # This group describes the explicitly requested simulation, never a
        # process group. Callbacks remain responsible for its test collectives.
        ref = DeviceRef(DeviceKind(resolved.type), resolved.index or 0)
        runtime[key] = DeviceGroup(
            "virtual_ep" if expert else "virtual_tp",
            (ref,) * context.world_size,
            tuple(range(context.world_size)),
            context.rank,
        )
    if context is None:
        context = (
            RuntimeParallelContext(group, runtime.get("communication"))
            if group
            else LocalParallelContext()
        )
    return resolved, context, runtime
