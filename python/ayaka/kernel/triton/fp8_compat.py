"""FP8 (E4M3 & E5M2) emulation for CUDA GPUs below sm_89.

Triton refuses native FP8 types anywhere in a kernel compiled for sm < 89
(``dtype.to_ir``), so an FP8 *storage* format -- an e4m3 KV cache, a quantized
activation buffer -- cannot be read or written by a kernel on Ampere. This
module keeps the storage format and moves the arithmetic to fp16/fp32:

1. Compile-time constexpr branching flags for zero-cost dead-code elimination.
2. Host-side byte-view helpers, so a launch never hands Triton an FP8 pointer
   on an architecture that cannot compile one.
3. Bit-exact decode, single-step RNE round, and bit-exact encode in Triton JIT.

Formats:
  - E4M3 (1S, 4E, 3M, bias=7): scaled decode via an fp16 container (``* 256``),
    RNE round. Codes ``0x7F``/``0xFF`` decode to +/-480 rather than e4m3fn NaN:
    the emulation is for finite values, and saturating those two bytes is the
    same choice the reference implementation makes.
  - E5M2 (1S, 5E, 2M, bias=15): direct upper byte of fp16 (``<< 8``), RNE round.

Every primitive is verified against PyTorch's own FP8 casts in
``tests/test_fp8_compat.py``.
"""

from __future__ import annotations

import logging
import os
from functools import cache
from typing import Final

import torch
import triton
import triton.language as tl
from triton.language import target_info
from triton.runtime.jit import constexpr_function

from ayaka.distributed.env import local_rank
from ayaka.types import DType
from ayaka.utils.torch_utils import cuda_available, supports_fp8, torch_dtype

__all__ = [
    # Host-side inspection & environment
    "FORCE_EMU",
    "e4m3_native",
    "e5m2_native",
    "fp8_native",
    # Host-side tensor & dtype helpers
    "e4m3_act_dtype",
    "e4m3_kernel_view",
    "e5m2_act_dtype",
    "e5m2_kernel_view",
    "fp8_kernel_view",
    # Compile-time constexprs (Triton)
    "e4m3_native_cx",
    "e5m2_native_cx",
    "fp8_native_cx",
    # JIT primitives: E4M3
    "e4m3_f32_to_u8",
    "e4m3_u8_to_f32",
    "round_e4m3",
    # JIT primitives: E5M2
    "e5m2_f32_to_u8",
    "e5m2_u8_to_f16",
    "e5m2_u8_to_f32",
    "round_e5m2",
]

logger = logging.getLogger(__name__)

#: Environment flags that force the emulated path even where native FP8 exists.
_ENV_VARS: Final[tuple[str, ...]] = (
    "AYAKA_FORCE_FP8_EMU",
    "AYAKA_FORCE_E4M3_EMU",
    "AYAKA_FORCE_E5M2_EMU",
)
_TRUE: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})


def _max_finite(dtype: DType) -> float:
    """Read a format's largest finite magnitude from the dtype registry."""
    value = dtype.max_finite
    if value is None:  # pragma: no cover - the FP8 registry entries always declare one
        raise ValueError(f"{dtype.label} has no declared max finite")
    return value


#: Largest finite magnitudes, read from the registry so a format change cannot
#: leave the kernels clamping to a stale constant. ``tl.constexpr`` because
#: Triton only resolves globals instantiated as constexpr.
_E4M3_MAX = tl.constexpr(_max_finite(DType.FP8_E4M3))
_E5M2_MAX = tl.constexpr(_max_finite(DType.FP8_E5M2))


def _forced_by() -> str | None:
    """Return the environment variable forcing emulation, or None."""
    for var in _ENV_VARS:
        if os.environ.get(var, "").strip().lower() in _TRUE:
            return var
    return None


FORCE_EMU: Final[bool] = _forced_by() is not None

if FORCE_EMU:
    # The constexpr flags below are globals, not launch arguments, so Triton's
    # cache key cannot tell a forced launch from a native one on sm_89+. Give
    # the forced path its own cache directory instead of reusing stale binaries.
    if "TRITON_CACHE_DIR" not in os.environ:
        os.environ["TRITON_CACHE_DIR"] = os.path.join(
            os.path.expanduser("~/.triton"), "cache-fp8emu"
        )
    logger.warning(
        "FP8 emulation forced by %s: native FP8 tensor cores are bypassed "
        "even where the device supports them",
        _forced_by(),
    )


@cache
def _native(device: int | None) -> bool:
    if FORCE_EMU or not cuda_available():
        return False
    index = local_rank() if device is None else device
    return supports_fp8(index)


def fp8_native(device: int | None = None) -> bool:
    """Whether ``device`` can execute native FP8 instructions.

    Args:
        device: CUDA index. Defaults to :func:`ayaka.distributed.env.local_rank`,
            the device this process was assigned, so the answer does not depend
            on which device happens to be current.

    Raises:
        RuntimeError: if the force flags changed after import. The flags feed
            compile-time constants that are invisible to Triton's cache key, so
            a mid-process change would silently reuse binaries built for the
            other branch. Set them before process startup.
    """
    if (_forced_by() is not None) != FORCE_EMU:
        raise RuntimeError(
            "FP8 emulation environment flags changed after module import. "
            "Set them before process startup to preserve deterministic Triton cache keys."
        )
    return _native(device)


def e4m3_native() -> bool:
    """Whether E4M3 arithmetic runs natively on the assigned device."""
    return fp8_native()


def e5m2_native() -> bool:
    """Whether E5M2 arithmetic runs natively on the assigned device."""
    return fp8_native()


def fp8_kernel_view(t: torch.Tensor) -> torch.Tensor:
    """Return a raw ``uint8`` view of an FP8 tensor where emulation is active.

    ``view`` between equal-itemsize dtypes is a pure bitcast: same storage, same
    shape, same strides, not one bit changed. It is the same workaround the KV
    storage index applies for FP8 scatter/gather in
    ``ayaka.kvcache.storage.index``.
    """
    return t if fp8_native(t.device.index) else t.view(torch.uint8)


def e4m3_kernel_view(t: torch.Tensor) -> torch.Tensor:
    """Byte view of an E4M3 tensor for an emulated kernel launch."""
    return fp8_kernel_view(t)


def e5m2_kernel_view(t: torch.Tensor) -> torch.Tensor:
    """Byte view of an E5M2 tensor for an emulated kernel launch."""
    return fp8_kernel_view(t)


def e4m3_act_dtype() -> torch.dtype:
    """Buffer dtype for quantized E4M3 activations: fp8 if native, else bf16."""
    return torch_dtype(DType.FP8_E4M3) if e4m3_native() else torch_dtype(DType.BF16)


def e5m2_act_dtype() -> torch.dtype:
    """Buffer dtype for quantized E5M2 activations: fp8 if native, else fp16."""
    return torch_dtype(DType.FP8_E5M2) if e5m2_native() else torch_dtype(DType.FP16)


@constexpr_function
def fp8_native_cx() -> bool:
    """Compile-time constant: True only when the target has native FP8."""
    return not FORCE_EMU and target_info.cuda_capability_geq(8, 9)


@constexpr_function
def e4m3_native_cx() -> bool:
    """Compile-time constant for E4M3, resolving on the compilation target."""
    return fp8_native_cx()


@constexpr_function
def e5m2_native_cx() -> bool:
    """Compile-time constant for E5M2, resolving on the compilation target."""
    return fp8_native_cx()


@triton.jit
def e4m3_u8_to_f32(v):
    """Decode E4M3 bits (uint8) to fp32.

    Aligns sign to bit 15 and exp+mantissa to bits 7..13 of an fp16 container,
    then applies ``2^(15 - 7) = 256`` scaling to adjust the exponent bias from
    7 to 15. Exact for every finite code.
    """
    h = ((v & 0x80).to(tl.uint16) << 8) | ((v & 0x7F).to(tl.uint16) << 7)
    return h.to(tl.float16, bitcast=True).to(tl.float32) * 256.0


@triton.jit
def round_e4m3(x):
    """Round fp32 onto the E4M3 value grid using single-step RNE.

    Caller must clamp to [-448.0, 448.0] first; this function does not, so a
    caller that already clamped can reuse the result for other purposes.
    - Normal range (``|x| >= 2^-6``): integer round-half-to-even mantissa
      truncation.
    - Subnormal range (``|x| < 2^-6``): add-magic trick
      (``magic = 1.5 * 2^14 = 24576.0``) to align the ULP directly to ``2^-9``.
      The magic is applied to ``|x|`` and the sign is reapplied, because
      ``(x + magic) - magic`` loses the sign of a magnitude that rounds to zero
      and PyTorch's cast keeps it (``-0.0`` must stay ``0x80``).
    """
    b = x.to(tl.uint32, bitcast=True)
    ax = tl.abs(x)
    lsb = (b >> 20) & 1
    y_norm = ((b + 524287 + lsb) & 0xFFF00000).to(tl.float32, bitcast=True)
    y_sub = (ax + 24576.0) - 24576.0
    y_sub = (y_sub.to(tl.uint32, bitcast=True) | (b & 0x80000000)).to(tl.float32, bitcast=True)
    return tl.where(ax >= 0.015625, y_norm, y_sub)


@triton.jit
def e4m3_f32_to_u8(x):
    """Clamp, round and pack fp32 onto the E4M3 byte grid (RNE).

    ``round_e4m3`` yields an fp32 value *on* the grid; the pack step is exact
    because dividing by 256 moves the e4m3 exponent field into fp16's, so the
    byte is recovered by shifting the fp16 bits. Works for subnormals too.
    """
    grid = round_e4m3(tl.clamp(x, -_E4M3_MAX, _E4M3_MAX))
    h = (grid * 0.00390625).to(tl.float16).to(tl.uint16, bitcast=True)
    return (((h >> 8) & 0x80) | ((h >> 7) & 0x7F)).to(tl.uint8)


@triton.jit
def e5m2_u8_to_f16(v):
    """Decode E5M2 bits (uint8) to fp16.

    E5M2 has the same exponent width (5 bits) and bias (15) as IEEE fp16, so
    the byte is directly the most-significant 8 bits of float16.
    """
    h = v.to(tl.uint16) << 8
    return h.to(tl.float16, bitcast=True)


@triton.jit
def e5m2_u8_to_f32(v):
    """Decode E5M2 bits directly to fp32 via the fp16 view."""
    return e5m2_u8_to_f16(v).to(tl.float32)


@triton.jit
def round_e5m2(x):
    """Round fp32 onto the E5M2 value grid using single-step RNE.

    Clamps to the E5M2 max finite (+/-57344.0) internally.
    - Normal range (``|x| >= 2^-14``): truncate the 23-bit mantissa to 2 bits
      with RNE.
    - Subnormal range (``|x| < 2^-14``): add-magic trick
      (``magic = 1.5 * 2^7 = 192.0``) to align the ULP directly to ``2^-16``,
      applied to ``|x|`` with the sign reapplied for the same reason as
      :func:`round_e4m3`.
    """
    x = tl.clamp(x, -_E5M2_MAX, _E5M2_MAX)
    b = x.to(tl.uint32, bitcast=True)
    ax = tl.abs(x)
    lsb = (b >> 21) & 1
    y_norm = ((b + 1048575 + lsb) & 0xFFE00000).to(tl.float32, bitcast=True)
    y_sub = (ax + 192.0) - 192.0
    y_sub = (y_sub.to(tl.uint32, bitcast=True) | (b & 0x80000000)).to(tl.float32, bitcast=True)
    return tl.where(ax >= 0.00006103515625, y_norm, y_sub)


@triton.jit
def e5m2_f32_to_u8(x):
    """Clamp, round and pack fp32 onto the E5M2 byte grid (RNE).

    Every E5M2 value is exactly representable in fp16 (max 57344 < 65504), so
    the byte is the upper half of the fp16 bit pattern.
    """
    grid = round_e5m2(tl.clamp(x, -_E5M2_MAX, _E5M2_MAX))
    h = grid.to(tl.float16).to(tl.uint16, bitcast=True)
    return (h >> 8).to(tl.uint8)
