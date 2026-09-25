"""Semantic KV data operations; page ownership stays with the caller.

Slot indices must be bounds-checked before enqueue, and writes must be unique.
Storage validates SlotIndex; the paged runner uses executor-owned addressing.
CPU, missing Triton, and unsupported layouts select PyTorch before mutation.
A launched kernel failure is propagated, never retried against changed memory.
"""

from __future__ import annotations

import math
from importlib import import_module
from types import ModuleType

import torch
from ayaka.caps import Cap
from ayaka.kernel.ops import custom_op
from ayaka.utils.import_utils import has_module

_COMPUTE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def _triton(tensor: torch.Tensor) -> ModuleType | None:
    # The raw-address copy implementation and FP8 compatibility path are
    # validated on NVIDIA. Other devices keep the portable implementation.
    if tensor.is_cuda and torch.version.hip is None and has_module("triton"):
        return import_module("ayaka.kernel.triton.cache.paged_kv")
    return None


def _cache_dtype(tensor: torch.Tensor) -> str:
    if tensor.dtype == torch.float8_e4m3fn:
        return "fp8_e4m3"
    if tensor.dtype == torch.float8_e5m2:
        return "fp8_e5m2"
    return "auto"


def _write_reference(
    source: torch.Tensor,
    destination: torch.Tensor,
    slots: torch.Tensor,
    inverse_scale: float,
) -> None:
    payload = source.to(device=destination.device)
    if destination.dtype in _FP8_DTYPES:
        limit = torch.finfo(destination.dtype).max
        payload = payload.float().mul(inverse_scale).clamp(-limit, limit)
    payload = payload.to(destination.dtype)
    target = destination.flatten(0, 1)
    if destination.dtype in _FP8_DTYPES:
        target.view(torch.uint8).index_copy_(0, slots, payload.contiguous().view(torch.uint8))
    else:
        target.index_copy_(0, slots, payload)


@custom_op(
    name="cache_scatter",
    mutates_args=["destination"],
    reference=_write_reference,
    dispatch_key="CompositeExplicitAutograd",
    caps=Cap.CUDAGRAPH_SAFE,
)
def scatter_cache(
    source: torch.Tensor,
    destination: torch.Tensor,
    slots: torch.Tensor,
    inverse_scale: float,
) -> None:
    """Write one plane through validated slots, with static inverse FP8 scale."""
    kernels = _triton(destination)
    if (
        kernels is None
        or not destination.is_contiguous()
        or source.dtype not in _COMPUTE_DTYPES
        or source.device != destination.device
        or slots.device != destination.device
        or not slots.is_contiguous()
        or slots.dtype != torch.int64
        or not math.isfinite(inverse_scale)
        or inverse_scale <= 0
    ):
        _write_reference(source, destination, slots, inverse_scale)
        return
    width = math.prod(destination.shape[2:])
    payload = source.contiguous().view(source.size(0), width)
    kernels.scatter_cache_slots(
        payload,
        destination.view(-1, width),
        slots,
        inverse_scale,
        _cache_dtype(destination),
    )


def _write_kv_reference(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slots: torch.Tensor,
    k_inverse: float,
    v_inverse: float,
) -> None:
    _write_reference(key, key_cache, slots, k_inverse)
    _write_reference(value, value_cache, slots, v_inverse)


@custom_op(
    name="cache_write_kv",
    mutates_args=["key_cache", "value_cache"],
    reference=_write_kv_reference,
    dispatch_key="CompositeExplicitAutograd",
    caps=Cap.CUDAGRAPH_SAFE,
)
def write_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slots: torch.Tensor,
    k_inverse: float,
    v_inverse: float,
) -> None:
    """Fused NHD MHA/GQA write; called once before attention reads the cache."""
    kernels = _triton(key_cache)
    if (
        kernels is None
        or key_cache.ndim != 4
        or key.ndim != 3
        or key.shape != value.shape
        or key_cache.shape != value_cache.shape
        or key_cache.stride() != value_cache.stride()
        or key_cache.stride(-1) != 1
        or key.stride(-1) != 1
        or value.stride(-1) != 1
        or key.dtype not in _COMPUTE_DTYPES
        or value.dtype != key.dtype
        or key_cache.dtype != value_cache.dtype
        or (key_cache.dtype not in _FP8_DTYPES and key_cache.dtype != key.dtype)
        or any(t.device != key_cache.device for t in (key, value, value_cache, slots))
        or slots.dtype != torch.int64
        or not slots.is_contiguous()
        or any(not math.isfinite(s) or s <= 0 for s in (k_inverse, v_inverse))
    ):
        _write_kv_reference(key, value, key_cache, value_cache, slots, k_inverse, v_inverse)
        return
    kernels.reshape_and_cache_flash(
        key,
        value,
        key_cache,
        value_cache,
        slots,
        _cache_dtype(key_cache),
        k_inverse,
        v_inverse,
        inverse_scales=True,
    )


def _gather_reference(
    source: torch.Tensor,
    slots: torch.Tensor,
    destination: torch.Tensor,
    scale: float,
    dequantize: bool,
) -> None:
    flat = source.flatten(0, 1)
    if source.dtype in _FP8_DTYPES:
        result = flat.view(torch.uint8).index_select(0, slots).view(source.dtype)
    else:
        result = flat.index_select(0, slots)
    if dequantize:
        result = result.float().mul(scale)
    destination.copy_(result)


@custom_op(
    name="cache_gather",
    mutates_args=["destination"],
    reference=_gather_reference,
    dispatch_key="CompositeExplicitAutograd",
    caps=Cap.CUDAGRAPH_SAFE,
)
def gather_cache(
    source: torch.Tensor,
    slots: torch.Tensor,
    destination: torch.Tensor,
    scale: float,
    dequantize: bool,
) -> None:
    """Gather slots (including repeats), optionally dequantizing into the output."""
    kernels = _triton(source)
    same_dtype = source.dtype == destination.dtype
    if (
        kernels is None
        or not source.is_contiguous()
        or not destination.is_contiguous()
        or source.device != destination.device
        or slots.device != source.device
        or slots.dtype != torch.int64
        or not slots.is_contiguous()
        or (
            dequantize
            and (source.dtype not in _FP8_DTYPES or destination.dtype not in _COMPUTE_DTYPES)
        )
        or (
            not dequantize
            and not same_dtype
            and (source.dtype not in _COMPUTE_DTYPES or destination.dtype not in _COMPUTE_DTYPES)
        )
    ):
        _gather_reference(source, slots, destination, scale, dequantize)
        return
    width = math.prod(source.shape[2:])
    kernels.gather_cache_slots(
        source.view(source.size(0), source.size(1), width),
        destination.view(slots.numel(), width),
        slots,
        scale,
        _cache_dtype(source) if dequantize else "auto",
    )


def _copy_reference(
    sources: list[torch.Tensor],
    destinations: list[torch.Tensor],
    non_blocking: bool,
) -> None:
    for source, destination in zip(sources, destinations, strict=True):
        destination.copy_(source, non_blocking=non_blocking)


def _disjoint_copies(sources: list[torch.Tensor], destinations: list[torch.Tensor]) -> bool:
    """Batched memcpy cannot reproduce ordered overlapping copies."""
    reads = [(t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()) for t in sources]
    writes = [(t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()) for t in destinations]
    for i, (begin, end) in enumerate(writes):
        if any(begin < stop and start < end for start, stop in reads + writes[:i]):
            return False
    return True


@custom_op(
    name="cache_copy",
    mutates_args=["destinations"],
    reference=_copy_reference,
    dispatch_key="CompositeExplicitAutograd",
)
def copy_cache(
    sources: list[torch.Tensor],
    destinations: list[torch.Tensor],
    non_blocking: bool,
) -> None:
    """Copy whole pages or valid-prefix slices, preserving physical bytes.

    D2D batches use one raw-pointer launch. Host transfers retain DMA and the
    caller's stream/fence ownership; pageable or blocking copies stay in torch.
    This descriptor-building operation is intentionally outside CUDA graphs.
    """
    if len(sources) != len(destinations):
        raise ValueError("cache copy source/destination counts must match")
    if not sources:
        return
    gpu = next((t for t in sources + destinations if t.is_cuda), None)
    kernels = None if gpu is None else _triton(gpu)
    if (
        gpu is None
        or kernels is None
        or any(
            not s.is_contiguous()
            or not d.is_contiguous()
            or s.dtype != d.dtype
            or s.shape != d.shape
            for s, d in zip(sources, destinations, strict=True)
        )
    ):
        _copy_reference(sources, destinations, non_blocking)
        return
    if all(t.device == gpu.device for t in sources + destinations):
        if not _disjoint_copies(sources, destinations):
            _copy_reference(sources, destinations, non_blocking)
            return
        with torch.cuda.device(gpu.device):
            stream = torch.cuda.current_stream(gpu.device)
            for tensor in sources + destinations:
                tensor.record_stream(stream)
            kernels.swap_blocks_batch(
                torch.tensor([t.data_ptr() for t in sources], dtype=torch.int64),
                torch.tensor([t.data_ptr() for t in destinations], dtype=torch.int64),
                torch.tensor([t.numel() * t.element_size() for t in sources], dtype=torch.int64),
            )
        return
    host_dma = non_blocking and all(
        (s.device == gpu.device and d.device.type == "cpu" and d.is_pinned())
        or (d.device == gpu.device and s.device.type == "cpu" and s.is_pinned())
        for s, d in zip(sources, destinations, strict=True)
    )
    if host_dma:
        mapping = torch.tensor([[0, 0]], dtype=torch.int64)
        for source, destination in zip(sources, destinations, strict=True):
            kernels.swap_blocks(
                source, destination, source.numel() * source.element_size(), mapping
            )
        return
    _copy_reference(sources, destinations, non_blocking)
