"""Shared Triton KV-cache primitives and host-side helpers."""

from __future__ import annotations

from typing import Final, cast

import torch
import triton
import triton.language as tl
from ayaka.kernel.triton.fp8_compat import (
    e4m3_f32_to_u8,
    e4m3_native_cx,
    e4m3_u8_to_f32,
    e5m2_f32_to_u8,
    e5m2_native_cx,
    e5m2_u8_to_f32,
)
from ayaka.types import DType, KVCacheDtype
from ayaka.utils.math_utils import FP8_E4M3_MAX
from ayaka.utils.torch_utils import compute_torch_dtypes

__all__ = [
    "GATHER_BLOCK_T",
    "as_byte_view",
    "as_int_view",
    "cache_kernel_view",
    "dequantize_if_fp8",
    "div_rn",
    "next_power_of_2_f32",
    "normalize_kv_cache_dtype",
    "quantize_if_fp8",
    "require_kv_source_dtype",
    "resolve_token_locations",
    "upper_bound",
    "warps_for_tile",
]

# MLA FP8 DS layout (DeepSeek-V3):
#   nope_dim=512, rope_dim=64, tile_size=128, num_tiles=4, entry_bytes=656
# Reintroduce as constants when the MLA cache kernels are ported.

#: Elements one gather tile covers, shared so the gather kernel and its host
#: launcher cannot drift apart.
GATHER_BLOCK_T: Final[int] = 8

#: Same-width integer dtypes used to reinterpret payloads bit-exactly.
_RAW_INT_DTYPE: Final[dict[int, torch.dtype]] = {
    1: torch.uint8,
    2: torch.int16,
    4: torch.int32,
    8: torch.int64,
}

#: Saturation bounds, read from the dtype registry so a format change cannot
#: leave the kernels clamping to a stale constant. ``tl.constexpr`` because
#: Triton only resolves globals instantiated as constexpr.
_E4M3_MAX = tl.constexpr(FP8_E4M3_MAX)
_E5M2_MAX = tl.constexpr(cast(float, DType.FP8_E5M2.max_finite))

_KV_OFFLOAD_MAX_BATCH_DESCRIPTORS_ENV: Final[str] = "KV_OFFLOAD_MAX_BATCH_DESCRIPTORS"
_ROCM_DEFAULT_MAX_BATCH_DESCRIPTORS: Final[int] = 8192
_NVFP4_MSG = (
    "{what} is implemented in a separate CUDA translation unit "
    "(nvfp4 kernels) that is not part of cache_kernels.cu, so there is "
    "no Triton port of it here."
)


def normalize_kv_cache_dtype(value: KVCacheDtype | str) -> KVCacheDtype:
    """Normalize a configuration value into a :class:`KVCacheDtype`.

    Accepts the enum itself or its string value (``"auto"``, ``"fp8_e4m3"``,
    ``"fp8_e5m2"``). The vLLM aliases ``"fp8"`` and ``"fp8_ds_mla"`` are
    rejected: Ayaka's MLA cache uses its own storage geometry, so folding them
    into E4M3 would hide a layout mismatch.

    Raises:
        TypeError: if ``value`` is neither a :class:`KVCacheDtype` nor a string.
        ValueError: if the string is not a supported KV cache dtype.
    """
    if isinstance(value, KVCacheDtype):
        return value
    if not isinstance(value, str):
        raise TypeError(f"kv_cache_dtype must be a KVCacheDtype or str, got {type(value).__name__}")
    try:
        return KVCacheDtype(value)
    except ValueError:
        supported = ", ".join(member.value for member in KVCacheDtype)
        raise ValueError(
            f"unsupported kv_cache_dtype {value!r}; expected one of: {supported}"
        ) from None


def require_kv_source_dtype(dtype: torch.dtype) -> None:
    """Require a model-side dtype a KV cache can be copied from.

    Raises:
        ValueError: if ``dtype`` is not one of the compute dtypes.
    """
    if dtype not in compute_torch_dtypes():
        raise ValueError(
            f"unsupported KV source dtype {dtype}; expected float16, bfloat16, or float32"
        )


def as_byte_view(tensor: torch.Tensor) -> torch.Tensor:
    """Return a raw ``uint8`` view of a 1-byte (FP8) cache plane.

    ``view`` between equal-itemsize dtypes is a pure bitcast: same storage, same
    shape, same strides, not one bit changed. Unlike
    :func:`ayaka.kernel.triton.fp8_compat.fp8_kernel_view` this is
    unconditional, because a quantized cache is addressed as bytes on every
    device -- routing it by architecture would raise the moment a device clears
    the sm_89 floor.

    Raises:
        TypeError: if ``tensor`` is not a torch tensor.
        ValueError: if the element size is not one byte.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("cache must be a torch.Tensor")
    if tensor.element_size() != 1:
        raise ValueError(f"expected a 1-byte (fp8/uint8) cache, got {tensor.dtype}")
    return tensor if tensor.dtype is torch.uint8 else tensor.view(torch.uint8)


def cache_kernel_view(cache: torch.Tensor, kv_dtype: KVCacheDtype) -> torch.Tensor:
    """Byte view for a quantized cache; the cache itself for ``AUTO``."""
    return as_byte_view(cache) if kv_dtype.is_quantized else cache


def as_int_view(tensor: torch.Tensor) -> torch.Tensor:
    """Reinterpret a tensor as a same-width integer tensor (bit-exact copies).

    Raises:
        TypeError: if ``tensor`` is not a torch tensor.
        ValueError: if no same-width integer dtype exists.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor must be a torch.Tensor")
    integer_dtype = _RAW_INT_DTYPE.get(tensor.element_size())
    if integer_dtype is None:
        raise ValueError(f"no same-width integer dtype for {tensor.dtype}")
    return tensor.view(integer_dtype)


def warps_for_tile(n_elems: int) -> int:
    """Warps such that every thread owns ~8 elements of an ``n_elems`` tile."""
    return max(1, min(8, triton.next_power_of_2(max(1, n_elems // 256))))


@triton.jit
def div_rn(x, y):
    """IEEE-754 round-to-nearest fp32 division, matching CUDA's ``/``."""
    return tl.math.div_rn(x, y)


@triton.jit
def next_power_of_2_f32(x):
    """Exact ``exp2(ceil(log2(x)))`` for positive normal fp32 ``x``.

    Works on the exponent bits so it does not depend on the accuracy of the
    ``log2``/``exp2`` approximations.
    """
    bits = x.to(tl.int32, bitcast=True)
    exp = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF
    exp = exp + tl.where(mant != 0, 1, 0)
    return (exp << 23).to(tl.float32, bitcast=True)


@triton.jit
def quantize_if_fp8(x, scale, QUANT: tl.constexpr, IS_E5M2: tl.constexpr):
    """``CopyWithScaleOp``: identity for AUTO, else ``fp8(x / scale)`` as uint8.

    The native path clamps and keeps NaN before the hardware cast; the
    emulated path delegates to ``fp8_compat``'s integer RNE encoder. Both
    saturate the finite range and produce identical bytes for finite values.
    """
    if QUANT:
        scaled = div_rn(x.to(tl.float32), scale)
        if IS_E5M2:
            if e5m2_native_cx():
                y = tl.where(scaled != scaled, scaled, tl.clamp(scaled, -_E5M2_MAX, _E5M2_MAX))
                out = y.to(tl.float8e5).to(tl.uint8, bitcast=True)
            else:
                out = e5m2_f32_to_u8(scaled)
        else:
            if e4m3_native_cx():
                y = tl.where(scaled != scaled, scaled, tl.clamp(scaled, -_E4M3_MAX, _E4M3_MAX))
                out = y.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
            else:
                out = e4m3_f32_to_u8(scaled)
    else:
        out = x
    return out


@triton.jit
def dequantize_if_fp8(x, scale, QUANT: tl.constexpr, IS_E5M2: tl.constexpr):
    """Identity for AUTO, else ``float(fp8) * scale`` in fp32."""
    if QUANT:
        if IS_E5M2:
            out = e5m2_u8_to_f32(x) * scale
        else:
            out = e4m3_u8_to_f32(x) * scale
    else:
        out = x
    return out


@triton.jit
def upper_bound(cu_ptr, n_entries, x, active, num_iters):
    """Vectorized ``std::upper_bound``.

    For every element of ``x``, returns how many of the ascending entries
    ``cu[0:n_entries]`` are ``<= x``. ``num_iters`` must be at least
    ``n_entries.bit_length()``.
    """
    lo = tl.zeros_like(x)
    hi = tl.zeros_like(x) + n_entries
    for _ in range(num_iters):
        act = (lo < hi) & active
        mid = (lo + hi) // 2
        v = tl.load(cu_ptr + mid, mask=act, other=0)
        right = act & (v <= x)
        lo = tl.where(right, mid + 1, lo)
        hi = tl.where(act & (v > x), mid, hi)
    return lo


@triton.jit
def resolve_token_locations(
    tok,
    num_tokens,
    cu_ptr,
    seq_starts_ptr,
    num_reqs,
    num_iters,
    HAS_SEQ_STARTS: tl.constexpr,
    HAS_TERMINAL: tl.constexpr,
):
    """Token-parallel replacement of ``map_gather_page_task``.

    ``tok`` holds output token indices. Returns ``(ok, req, pos)`` where ``ok``
    says the token belongs to a request, ``req`` is that request and ``pos``
    the token's position inside the request's KV sequence
    (``seq_starts[req] + tok - cu[req]``).

    ``HAS_TERMINAL``: ``cu`` has ``num_reqs + 1`` entries (the last one is the
    end of the final request) versus ``num_reqs`` entries, in which case the
    final request extends to ``num_tokens``.
    """
    in_range = tok < num_tokens
    if HAS_TERMINAL:
        n_entries = num_reqs + 1
    else:
        n_entries = num_reqs
    req = upper_bound(cu_ptr, n_entries, tok, in_range, num_iters) - 1
    ok = in_range & (req >= 0) & (req < num_reqs)
    req = tl.where(ok, req, 0)
    out_begin = tl.load(cu_ptr + req, mask=ok, other=0)
    if HAS_SEQ_STARTS:
        src_begin = tl.load(seq_starts_ptr + req, mask=ok, other=0)
    else:
        src_begin = 0
    pos = src_begin + (tok - out_begin)
    return ok, req, pos
