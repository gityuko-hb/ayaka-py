"""Math helpers shared by Triton kernels.

Two kinds of code live here:

* ``@triton.jit`` conversion helpers for quantized formats (E8M0/E4M3 scales,
  E2M1 codes, FP4 packing and L2-swizzled offsets) plus the grouped program-id
  swizzle used by tiled GEMM-like kernels. They must stay bit-compatible with
  the CUDA quant sources they were ported from: the midpoint comparisons,
  saturation rules and the E8M0 byte-0 quirk are deliberate.
* Host launch-geometry helpers (``block_size``, ``vocab_*``,
  ``norm_block_and_warps``) that kernel launchers share instead of copying the
  same ``min(cap, next_power_of_2(v))`` and ``num_warps`` ladders.

Importing this module imports Triton, so only kernel launchers may import it;
tests must guard with ``pytest.importorskip("triton")``.
"""

from __future__ import annotations

import triton
import triton.language as tl

from ayaka.utils.math_utils import next_power_of_2
from ayaka.utils.validation import require_int


def block_size(value: int, cap: int = 4096) -> int:
    """Smallest power of two >= ``value``, capped at ``cap``.

    The cap keeps pathological vocab/feature sizes from asking for a tile the
    kernel cannot afford; ``cap`` is expected to be a power of two.

    Raises:
        TypeError: if ``value`` is not an integer.
        ValueError: if ``value`` is not positive.
    """
    require_int(value, "value", minimum=1)
    return min(cap, next_power_of_2(value))


def vocab_block_size(vocab_size: int) -> int:
    """Tile width for vocab-wide sampling kernels (``min(4096, np2(V))``)."""
    return block_size(vocab_size, 4096)


def vocab_num_warps(vocab_size: int) -> int:
    """Warp count for vocab-wide sampling kernels: wider tiles want more warps."""
    return 8 if vocab_size >= 32768 else 4


def norm_block_and_warps(d: int) -> tuple[int, int]:
    """Block size (<= 8192) and warp count for an RMS-norm row of width ``d``."""
    require_int(d, "d", minimum=1)
    block = min(8192, next_power_of_2(d))
    if block >= 4096:
        warps = 16
    elif block >= 1024:
        warps = 8
    else:
        warps = 4
    return block, warps


@triton.jit
def grouped_pid(M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GROUP_M: tl.constexpr):
    """L2-friendly GEMM program-id swizzle used by all tiled kernels."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def e8m0_to_f32(e):
    """Source-faithful E8M0 decoder used by MXFP4.

    CUDA source implements ``__uint_as_float((uint32_t)e << 23)``.  In
    particular byte 0 decodes to +0.0 (not the mathematical 2**-127).
    """
    bits = e.to(tl.uint32) << 23
    return tl.cast(bits, tl.float32, bitcast=True)


@triton.jit
def positive_f32_to_e8m0_ceil(x):
    """Match mxfp4_quant.cu float_to_e8m0 for non-negative float32 x."""
    bits = tl.cast(x, tl.uint32, bitcast=True)
    biased = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF
    biased = biased + ((mant != 0) & (biased < 254)).to(tl.uint32)
    return tl.where(x <= 0.0, 0, biased).to(tl.uint8)


@triton.jit
def fp8_e4m3fn_bits_to_f32(raw):
    """Decode finite torch/NVIDIA E4M3FN bytes to float32."""
    r = raw.to(tl.uint32)
    sign = tl.where((r & 0x80) != 0, -1.0, 1.0)
    exp = (r >> 3) & 0xF
    mant = r & 0x7
    normal = (1.0 + mant.to(tl.float32) * 0.125) * tl.exp2(exp.to(tl.float32) - 7.0)
    sub = mant.to(tl.float32) * tl.exp2(-9.0)
    # 0x7f/0xff are NaN in E4M3FN. Quantized tensors here are expected finite;
    # map these payloads to NaN so corrupted inputs are not silently accepted.
    finite = ~((exp == 15) & (mant == 7))
    value = sign * tl.where(exp == 0, sub, normal)
    return tl.where(finite, value, float("nan"))


@triton.jit
def fp4_e2m1_bits_to_f32(code):
    c = code.to(tl.uint32)
    sign = tl.where((c & 0x8) != 0, -1.0, 1.0)
    mag = c & 0x7
    v = tl.where(
        mag == 0,
        0.0,
        tl.where(
            mag == 1,
            0.5,
            tl.where(
                mag == 2,
                1.0,
                tl.where(
                    mag == 3,
                    1.5,
                    tl.where(mag == 4, 2.0, tl.where(mag == 5, 3.0, tl.where(mag == 6, 4.0, 6.0))),
                ),
            ),
        ),
    )
    return sign * v


@triton.jit
def unpack_fp4_byte(byte, high):
    b = byte.to(tl.uint32)
    code = tl.where(high, (b >> 4) & 0xF, b & 0xF)
    return fp4_e2m1_bits_to_f32(code)


@triton.jit
def mxfp4_float_to_code(x):
    """MXFP4 software conversion from source (strict midpoint comparisons)."""
    ax = tl.abs(x)
    sign = tl.where(x < 0.0, 8, 0)
    code = tl.where(
        ax < 0.25,
        0,
        tl.where(
            ax < 0.75,
            1,
            tl.where(
                ax < 1.25,
                2,
                tl.where(
                    ax < 1.75,
                    3,
                    tl.where(ax < 2.5, 4, tl.where(ax < 3.5, 5, tl.where(ax < 5.0, 6, 7))),
                ),
            ),
        ),
    ).to(tl.uint8)
    return code | sign.to(tl.uint8)


@triton.jit
def nvfp4_float_to_code_rne(x):
    """NVFP4 software fallback: E2M1 round-to-nearest-even midpoints."""
    ax = tl.abs(x)
    sign = tl.where(x < 0.0, 8, 0)
    code = tl.where(
        ax <= 0.25,
        0,
        tl.where(
            ax < 0.75,
            1,
            tl.where(
                ax <= 1.25,
                2,
                tl.where(
                    ax < 1.75,
                    3,
                    tl.where(ax <= 2.5, 4, tl.where(ax < 3.5, 5, tl.where(ax <= 5.0, 6, 7))),
                ),
            ),
        ),
    ).to(tl.uint8)
    return code | sign.to(tl.uint8)


@triton.jit
def positive_f32_to_e4m3_bits_fallback(x):
    """Software E4M3 scale encoder from nvfp4_quant.cu, clamped to raw 126."""
    bits = tl.cast(x, tl.uint32, bitcast=True)
    exp = ((bits >> 23) & 0xFF).to(tl.int32) - 127
    mantissa = bits & 0x7FFFFF
    too_large = exp > 8
    exp = tl.where(too_large, 8, exp)
    mantissa = tl.where(too_large, 0x600000, mantissa)
    valid = (x > 0.0) & (exp >= -9)
    biased = tl.maximum(0, tl.minimum(15, exp + 7))
    mant3 = ((mantissa >> 20) & 0x7).to(tl.int32)
    rem = mantissa & 0xFFFFF
    round_up = (rem > 0x80000) | ((rem == 0x80000) & ((mant3 & 1) != 0))
    mant3 = mant3 + round_up.to(tl.int32)
    carry = mant3 > 7
    mant3 = tl.where(carry, 0, mant3)
    biased = biased + carry.to(tl.int32)
    overflow = biased > 15
    biased = tl.where(overflow, 15, biased)
    mant3 = tl.where(overflow, 7, mant3)
    raw = ((biased << 3) | mant3).to(tl.uint8)
    raw = tl.minimum(raw, 126).to(tl.uint8)
    return tl.where(valid, raw, 0).to(tl.uint8)


@triton.jit
def swizzled_128x4_offset(row, col, cols_padded):
    inner_k = col % 4
    inner_m = (row % 128) // 32
    outer_m = row % 32
    k_tile = col // 4
    m_tile = row // 128
    num_k_tiles = (cols_padded + 3) // 4
    return m_tile * num_k_tiles * 512 + k_tile * 512 + outer_m * 16 + inner_m * 4 + inner_k
