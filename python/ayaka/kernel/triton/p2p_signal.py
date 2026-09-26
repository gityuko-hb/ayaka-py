"""Peer-signal primitives used by the DC2 verified-P2P probe."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = [
    "TIMEOUT_VALUE",
    "available",
    "consume_epochs",
    "publish_epochs",
    "release_signal",
    "remote_atomic_add",
]

#: Consumer result written when the spin budget is exhausted before the epoch.
TIMEOUT_VALUE = -1

#: Triton only resolves module globals instantiated as constexpr.
_TIMEOUT_VALUE = tl.constexpr(TIMEOUT_VALUE)

_DEFAULT_SPIN_BUDGET = 5_000_000


@triton.jit
def _publish_epochs_kernel(payload_ptr, signal_ptr, epochs):
    for epoch in range(1, epochs + 1):
        tl.store(payload_ptr + epoch, epoch)
        tl.atomic_add(signal_ptr, 1, sem="release", scope="gpu")


@triton.jit
def _release_signal_kernel(signal_ptr):
    tl.atomic_add(signal_ptr, 1, sem="release", scope="gpu")


@triton.jit
def _consume_epochs_kernel(payload_ptr, signal_ptr, out_ptr, epochs, spin_budget):
    epoch = tl.program_id(0) + 1
    spins = 0
    seen = 0
    while (seen < epoch) & (spins < spin_budget):
        seen = tl.atomic_add(signal_ptr, 0, sem="acquire", scope="gpu")
        spins += 1
    value = tl.load(payload_ptr + epoch, volatile=True)
    if seen >= epoch:
        tl.store(out_ptr + epoch, value)
    else:
        tl.store(out_ptr + epoch, _TIMEOUT_VALUE)


@triton.jit
def _remote_atomic_add_kernel(counter_ptr, times):
    for _ in range(times):
        tl.atomic_add(counter_ptr, 1, sem="acq_rel", scope="gpu")


def available() -> bool:
    """Whether the peer-signal probe can run here (Triton import + CUDA)."""
    return bool(torch.cuda.is_available())


def _require_int32_cuda(tensor: torch.Tensor, name: str, minimum: int) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.dtype != torch.int32:
        raise TypeError(f"{name} must have dtype int32, got {tensor.dtype}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if tensor.numel() < minimum:
        raise ValueError(f"{name} needs at least {minimum} elements, got {tensor.numel()}")


def publish_epochs(payload: torch.Tensor, signal: torch.Tensor, epochs: int) -> None:
    """Write ``payload[1..epochs]`` with release increments of ``signal``.

    Payload and signal may live in a peer-mapped allocation; the caller is
    responsible for keeping that mapping alive past the launch.
    """
    if type(epochs) is not int or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    _require_int32_cuda(payload, "payload", epochs + 1)
    _require_int32_cuda(signal, "signal", 1)
    _publish_epochs_kernel[(1,)](payload, signal, epochs)


def release_signal(signal: torch.Tensor) -> None:
    """Release-increment ``signal`` once, publishing all prior writes."""
    _require_int32_cuda(signal, "signal", 1)
    _release_signal_kernel[(1,)](signal)


def consume_epochs(
    payload: torch.Tensor,
    signal: torch.Tensor,
    out: torch.Tensor,
    epochs: int,
    *,
    spin_budget: int = _DEFAULT_SPIN_BUDGET,
) -> None:
    """Acquire-wait each epoch and copy the payload value into ``out``.

    One Triton program per epoch spins on the acquire atomic with a finite
    budget; ``out[epoch]`` receives the payload value or :data:`TIMEOUT_VALUE`.
    """
    if type(epochs) is not int or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if type(spin_budget) is not int or spin_budget < 1:
        raise ValueError("spin_budget must be a positive integer")
    _require_int32_cuda(payload, "payload", epochs + 1)
    _require_int32_cuda(signal, "signal", 1)
    _require_int32_cuda(out, "out", epochs + 1)
    _consume_epochs_kernel[(epochs,)](payload, signal, out, epochs, spin_budget)


def remote_atomic_add(counter: torch.Tensor, times: int = 1) -> None:
    """Atomically add ``times`` to a counter through a peer mapping."""
    if type(times) is not int or times < 1:
        raise ValueError("times must be a positive integer")
    _require_int32_cuda(counter, "counter", 1)
    _remote_atomic_add_kernel[(1,)](counter, times)
