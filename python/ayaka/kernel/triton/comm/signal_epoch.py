"""Signal/epoch two-rank SUM kernels for eager and graph

Correctness invariants this module encodes:

* **Ping-pong payload.** Epoch ``e`` uses slot ``e % 2``. One communication
  stream per workspace means a rank can be at most one epoch ahead, so the
  producer ever writes only the slot the consumer is not reading. A single
  slot would let a fast rank overwrite data another rank is still consuming.
* **One release per epoch.** A single program publishes the whole payload and
  then increments the peer signal exactly once; the consumer waits for
  ``seen >= epoch`` and reads the slot matching its own epoch.
* **Status is local.** The consuming program writes ``epoch`` on success or
  :data:`STATUS_TIMEOUT` when its finite spin budget is exhausted; the host
  never infers completion from a host return.

The eager op is registered through :mod:`ayaka.kernel.ops`, declares every mutated
buffer (local ``x``, the peer payload/signal written remotely, and the local
status pad) so a compiler cannot reorder or elide the signaling, and exposes a
fake implementation. It deliberately declares no reference implementation: a
collective has no meaningful single-process oracle, so
``AYAKA_FORCE_REFERENCE_OPS`` must route the group to Torch instead. The
graph launcher is separate: its device counter advances on every replay while
the eager op keeps its host-provided epoch and ``Cap.NONE``.
"""

from __future__ import annotations

from typing import Any, cast

import torch
import triton
import triton.language as tl

from ayaka.caps import Cap
from ayaka.kernel.comm_layout import (
    POINTER_TABLE_BYTES,
    SIGNAL_PAD_BYTES,
    STATUS_PAD_BYTES,
    STATUS_TIMEOUT,
    STATUS_UNSET,
    WorkspaceLayout,
)
from ayaka.kernel.ops import custom_op

__all__ = [
    "BLOCK_ELEMENTS",
    "DEFAULT_SPIN_BUDGET",
    "POINTER_TABLE_BYTES",
    "SIGNAL_PAD_BYTES",
    "STATUS_PAD_BYTES",
    "STATUS_TIMEOUT",
    "STATUS_UNSET",
    "SUPPORTED_DTYPES",
    "WorkspaceLayout",
    "available",
    "launch_signal_epoch_sum",
    "launch_signal_epoch_sum_graph",
    "signal_epoch_sum",
    "warmup_signal_epoch_sum",
]

#: Elements one program handles per loop iteration.
BLOCK_ELEMENTS = 1024
#: Default finite spin budget of the consuming program.
DEFAULT_SPIN_BUDGET = 1_000_000

#: Triton only resolves module globals instantiated as constexpr.
_STATUS_TIMEOUT = tl.constexpr(STATUS_TIMEOUT)

SUPPORTED_DTYPES: tuple[torch.dtype, ...] = (torch.float32, torch.float16, torch.bfloat16)


@triton.jit
def _signal_epoch_sum_kernel(
    x_ptr,
    local_payload_ptr,
    peer_payload_ptr,
    local_signal_ptr,
    peer_signal_ptr,
    status_ptr,
    n_elements,
    epoch,
    slot_stride,
    spin_budget,
    UPCAST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    peer_slot = peer_payload_ptr + (epoch % 2) * slot_stride
    local_slot = local_payload_ptr + (epoch % 2) * slot_stride
    blocks = tl.cdiv(n_elements, BLOCK)

    for block in range(0, blocks):
        offsets = block * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        value = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        tl.store(peer_slot + offsets, value, mask=mask)
    tl.atomic_add(peer_signal_ptr, 1, sem="release", scope="gpu")

    seen = 0
    spins = 0
    while (seen < epoch) & (spins < spin_budget):
        seen = tl.atomic_add(local_signal_ptr, 0, sem="acquire", scope="gpu")
        spins += 1

    if seen >= epoch:
        for block in range(0, blocks):
            offsets = block * BLOCK + tl.arange(0, BLOCK)
            mask = offsets < n_elements
            value = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            peer = tl.load(local_slot + offsets, mask=mask, other=0.0)
            if UPCAST:
                combined = (value.to(tl.float32) + peer.to(tl.float32)).to(x_ptr.dtype.element_ty)
            else:
                combined = value + peer
            tl.store(x_ptr + offsets, combined, mask=mask)
        tl.store(status_ptr, epoch)
    else:
        tl.store(status_ptr, _STATUS_TIMEOUT)


@triton.jit
def _signal_epoch_sum_graph_kernel(
    x_ptr,
    local_payload_ptr,
    peer_payload_ptr,
    local_signal_ptr,
    peer_signal_ptr,
    status_ptr,
    epoch_ptr,
    n_elements,
    slot_stride,
    spin_budget,
    epoch_limit,
    UPCAST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # The counter is device-owned. A captured host scalar would repeat the
    # capture epoch on every replay and accept stale peer payload.
    previous_epoch = tl.load(epoch_ptr)
    epoch = previous_epoch + 1
    if previous_epoch < epoch_limit:
        peer_slot = peer_payload_ptr + (epoch % 2) * slot_stride
        local_slot = local_payload_ptr + (epoch % 2) * slot_stride
        blocks = tl.cdiv(n_elements, BLOCK)
        for block in range(0, blocks):
            offsets = block * BLOCK + tl.arange(0, BLOCK)
            mask = offsets < n_elements
            value = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            tl.store(peer_slot + offsets, value, mask=mask)
        tl.atomic_add(peer_signal_ptr, 1, sem="release", scope="gpu")
        seen = 0
        spins = 0
        while (seen < epoch) & (spins < spin_budget):
            seen = tl.atomic_add(local_signal_ptr, 0, sem="acquire", scope="gpu")
            spins += 1
        if seen >= epoch:
            for block in range(0, blocks):
                offsets = block * BLOCK + tl.arange(0, BLOCK)
                mask = offsets < n_elements
                value = tl.load(x_ptr + offsets, mask=mask, other=0.0)
                peer = tl.load(local_slot + offsets, mask=mask, other=0.0)
                if UPCAST:
                    combined = (value.to(tl.float32) + peer.to(tl.float32)).to(
                        x_ptr.dtype.element_ty
                    )
                else:
                    combined = value + peer
                tl.store(x_ptr + offsets, combined, mask=mask)
            tl.store(status_ptr, epoch)
        else:
            tl.store(status_ptr, _STATUS_TIMEOUT)
        tl.store(epoch_ptr, epoch)
    else:
        tl.store(status_ptr, _STATUS_TIMEOUT)


def available() -> bool:
    """Whether the two-rank communication kernel can run here."""
    return bool(torch.cuda.is_available())


def _validate_region(tensor: Any, name: str, dtype: torch.dtype) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _validate_signal(tensor: Any, name: str) -> None:
    _validate_region(tensor, name, torch.int32)
    if tensor.numel() < 1:
        raise ValueError(f"{name} needs at least one element")


def validate_launch(
    *,
    x: torch.Tensor,
    local_payload: torch.Tensor,
    peer_payload: torch.Tensor,
    local_signal: torch.Tensor,
    peer_signal: torch.Tensor,
    status: torch.Tensor,
    epoch: int,
    slot_stride: int,
    spin_budget: int,
) -> None:
    """Validate one launch's buffers and scalars before any device work.

    Raises:
        TypeError: buffer dtype or element type is unsupported.
        ValueError: layout, alignment, size or scalar range is invalid.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if x.dtype not in SUPPORTED_DTYPES:
        raise TypeError(f"x must have dtype {SUPPORTED_DTYPES}, got {x.dtype}")
    if not x.is_cuda or not x.is_contiguous():
        raise ValueError("x must be a contiguous CUDA tensor")
    if x.numel() == 0:
        raise ValueError("x must be non-empty; empty collectives are a no-op")
    nbytes = x.numel() * x.element_size()
    if nbytes % 16 or x.data_ptr() % 16:
        raise ValueError("x must be 16-byte aligned in both pointer and byte length")
    _validate_region(local_payload, "local_payload", x.dtype)
    _validate_region(peer_payload, "peer_payload", x.dtype)
    _validate_signal(local_signal, "local_signal")
    _validate_signal(peer_signal, "peer_signal")
    _validate_signal(status, "status")
    if type(epoch) is not int or epoch < 1:
        raise ValueError("epoch must be a positive integer")
    if type(slot_stride) is not int or slot_stride < 1:
        raise ValueError("slot_stride must be a positive integer")
    if type(spin_budget) is not int or spin_budget < 1:
        raise ValueError("spin_budget must be a positive integer")
    if local_payload.numel() < 2 * slot_stride or peer_payload.numel() < 2 * slot_stride:
        raise ValueError("payload region must hold two slots of slot_stride elements")


def launch_signal_epoch_sum(
    *,
    x: torch.Tensor,
    local_payload: torch.Tensor,
    peer_payload: torch.Tensor,
    local_signal: torch.Tensor,
    peer_signal: torch.Tensor,
    status: torch.Tensor,
    epoch: int,
    slot_stride: int,
    spin_budget: int = DEFAULT_SPIN_BUDGET,
    block: int = BLOCK_ELEMENTS,
) -> None:
    """Launch one two-rank signal/epoch SUM on the current stream.

    Every rank executes the same call sequence; the launch itself performs the
    publish, the release, the bounded acquire wait and the in-place combine.
    """
    validate_launch(
        x=x,
        local_payload=local_payload,
        peer_payload=peer_payload,
        local_signal=local_signal,
        peer_signal=peer_signal,
        status=status,
        epoch=epoch,
        slot_stride=slot_stride,
        spin_budget=spin_budget,
    )
    upcast = x.dtype is not torch.float32
    cast(Any, _signal_epoch_sum_kernel)[(1,)](
        x,
        local_payload,
        peer_payload,
        local_signal,
        peer_signal,
        status,
        x.numel(),
        epoch,
        slot_stride,
        spin_budget,
        UPCAST=upcast,
        BLOCK=block,
    )


def launch_signal_epoch_sum_graph(
    *,
    x: torch.Tensor,
    local_payload: torch.Tensor,
    peer_payload: torch.Tensor,
    local_signal: torch.Tensor,
    peer_signal: torch.Tensor,
    status: torch.Tensor,
    epoch_counter: torch.Tensor,
    slot_stride: int,
    spin_budget: int,
    epoch_limit: int,
    block: int = BLOCK_ELEMENTS,
    warmup: bool = False,
) -> None:
    """Compile or launch a replayable SUM with a device-resident epoch counter."""
    validate_launch(
        x=x,
        local_payload=local_payload,
        peer_payload=peer_payload,
        local_signal=local_signal,
        peer_signal=peer_signal,
        status=status,
        epoch=1,
        slot_stride=slot_stride,
        spin_budget=spin_budget,
    )
    _validate_signal(epoch_counter, "epoch_counter")
    if type(epoch_limit) is not int or epoch_limit < 1 or epoch_limit >= 2**31:
        raise ValueError("epoch_limit must fit a positive signed int32")
    args = (
        x,
        local_payload,
        peer_payload,
        local_signal,
        peer_signal,
        status,
        epoch_counter,
        x.numel(),
        slot_stride,
        spin_budget,
        epoch_limit,
    )
    kwargs = {"UPCAST": x.dtype is not torch.float32, "BLOCK": block}
    kernel = cast(Any, _signal_epoch_sum_graph_kernel)
    if warmup:
        kernel.warmup(*args, **kwargs, grid=(1,))
    else:
        kernel[(1,)](*args, **kwargs)


@custom_op(
    name="signal_epoch_sum",
    namespace="ayaka",
    mutates_args=["x", "peer_payload", "peer_signal", "status"],
    caps=Cap.NONE,
)
def signal_epoch_sum(
    x: torch.Tensor,
    local_payload: torch.Tensor,
    peer_payload: torch.Tensor,
    local_signal: torch.Tensor,
    peer_signal: torch.Tensor,
    status: torch.Tensor,
    epoch: int,
    slot_stride: int,
    spin_budget: int,
) -> None:
    """Public custom op for the DC4 two-rank SUM (eager, CUDA only).

    ``x`` is combined in place with the peer payload staged in the local
    workspace. ``peer_payload`` and ``peer_signal`` are mutated through the
    imported peer mapping, and ``status`` records the consuming epoch or
    :data:`STATUS_TIMEOUT`. Not graph-certified: no CUDA graph support is
    claimed for DC4.
    """
    launch_signal_epoch_sum(
        x=x,
        local_payload=local_payload,
        peer_payload=peer_payload,
        local_signal=local_signal,
        peer_signal=peer_signal,
        status=status,
        epoch=epoch,
        slot_stride=slot_stride,
        spin_budget=spin_budget,
    )


def warmup_signal_epoch_sum(
    *,
    x: torch.Tensor,
    local_payload: torch.Tensor,
    peer_payload: torch.Tensor,
    local_signal: torch.Tensor,
    peer_signal: torch.Tensor,
    status: torch.Tensor,
    epoch: int,
    slot_stride: int,
    spin_budget: int = DEFAULT_SPIN_BUDGET,
    block: int = BLOCK_ELEMENTS,
) -> bool:
    """Compile the kernel for this specialization without launching it.

    Returns ``False`` when the installed Triton build cannot warm up
    out of band; the first real launch then pays the compile cost once.
    """
    warmup = getattr(_signal_epoch_sum_kernel, "warmup", None)
    if not callable(warmup):
        return False
    upcast = x.dtype is not torch.float32
    warmup(
        x,
        local_payload,
        peer_payload,
        local_signal,
        peer_signal,
        status,
        x.numel(),
        epoch,
        slot_stride,
        spin_budget,
        UPCAST=upcast,
        BLOCK=block,
        grid=(1,),
    )
    return True
