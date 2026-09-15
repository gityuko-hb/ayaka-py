"""Philox4x32-10 counter-based RNG for Triton sampling kernels.

Triton twin of ``ayaka.sampling.rng``: it must stay bit-identical to the
torch implementation for the same ``(seed, flat)`` address. The seed is the
2x u32 Philox key, ``flat >> 1`` is the 64-bit counter, and ``flat & 1``
selects one of the two uniform draws packed from each counter output.
"""

from __future__ import annotations

import triton
import triton.language as tl

_M0 = tl.constexpr(0xD2511F53)
_M1 = tl.constexpr(0xCD9E8D57)
_W0 = tl.constexpr(0x9E3779B9)
_W1 = tl.constexpr(0xBB67AE85)
_MASK32 = tl.constexpr(0xFFFFFFFF)


@triton.jit
def _mullo32(a, b):
    """Return the low 32 bits of a u32-by-u32 product.

    Args:
        a: First u32 operand held in an int64 lane.
        b: Second u32 operand held in an int64 lane.

    Returns:
        Low 32 bits of ``a * b`` as a u32 in an int64 lane.

    Note:
        Reconstructed from 16-bit halves so every intermediate stays below
        ``2**51`` and never overflows signed int64.
    """
    a_lo = a & 0xFFFF
    a_hi = a >> 16
    b_lo = b & 0xFFFF
    b_hi = b >> 16
    lh = a_lo * b_hi + a_hi * b_lo
    mid = (a_lo * b_lo) + (lh << 16)
    return mid & _MASK32


@triton.jit
def _mulhi32(a, b):
    """Return the high 32 bits of a u32-by-u32 product.

    Args:
        a: First u32 operand held in an int64 lane.
        b: Second u32 operand held in an int64 lane.

    Returns:
        High 32 bits of ``a * b`` as a u32 in an int64 lane.

    Note:
        Same 16-bit-half product as ``_mullo32``; only the carry handling
        differs, so Triton CSE can merge calls sharing operands.
    """
    a_lo = a & 0xFFFF
    a_hi = a >> 16
    b_lo = b & 0xFFFF
    b_hi = b >> 16
    lh = a_lo * b_hi + a_hi * b_lo
    hh = a_hi * b_hi
    mid = (a_lo * b_lo) + (lh << 16)
    return (hh + (mid >> 32)) & _MASK32


@triton.jit
def philox4x32_10(c0, c1, c2, c3, k0, k1):
    """Advance Philox4x32 for 10 rounds.

    Args:
        c0: Counter word 0 (u32 in an int64 lane).
        c1: Counter word 1 (u32 in an int64 lane).
        c2: Counter word 2 (u32 in an int64 lane).
        c3: Counter word 3 (u32 in an int64 lane).
        k0: Key word 0 (u32 in an int64 lane).
        k1: Key word 1 (u32 in an int64 lane).

    Returns:
        Tuple ``(c0, c1, c2, c3)`` of the four output words after 10
        rounds.

    Note:
        Bit-exact with ``philox4x32_10`` in ``ayaka.sampling.rng`` and the
        official Random123 KAT pinned by ``tests/test_sampling_rng.py``.
        Callers passing uint64 lanes are bitcast to int64 first so the
        loop-carried types agree; two's-complement wrap-around keeps
        bitwise and additive steps bit-identical across the two types.
        Round one uses the given key; later rounds bump it by ``(W0, W1)``.
    """
    # Some callers pass seed/offset as uint64 (topk_topp port) — bitcast to
    # int64 for consistent loop-carried types; two's-complement keeps
    # bitwise/add wrap-around bit-identical across the two types.
    c0 = tl.cast(c0, tl.int64, bitcast=True)
    c1 = tl.cast(c1, tl.int64, bitcast=True)
    c2 = tl.cast(c2, tl.int64, bitcast=True)
    c3 = tl.cast(c3, tl.int64, bitcast=True)
    k0 = tl.cast(k0, tl.int64, bitcast=True)
    k1 = tl.cast(k1, tl.int64, bitcast=True)
    for r in range(10):
        if r > 0:
            k0 = (k0 + _W0) & _MASK32
            k1 = (k1 + _W1) & _MASK32
        hi0 = _mulhi32(_M0, c0)
        lo0 = _mullo32(_M0, c0)
        hi1 = _mulhi32(_M1, c2)
        lo1 = _mullo32(_M1, c2)
        c0 = hi1 ^ c1 ^ k0
        c1 = lo1
        c2 = hi0 ^ c3 ^ k1
        c3 = lo0
    return c0, c1, c2, c3


@triton.jit
def _philox_u53_bits(seed, counter, odd):
    """Run 10 Philox rounds for one counter and select a draw.

    Args:
        seed: 64-bit Philox key split into ``(k0, k1)`` u32 halves.
        counter: 64-bit counter split into ``(c0, c1)`` u32 halves; the
            upper counter words are zero.
        odd: Draw selector; ``0`` packs ``(w0, w1)`` and ``1`` packs
            ``(w2, w3)``.

    Returns:
        Int64 mantissa bits in ``[0, 2**53)`` for the selected draw.
    """
    w0, w1, w2, w3 = philox4x32_10(  # pyright: ignore[reportGeneralTypeIssues]
        counter & _MASK32,
        (counter >> 32) & _MASK32,
        counter * 0,
        counter * 0,
        seed & _MASK32,
        (seed >> 32) & _MASK32,
    )
    v_even = (w0 << 32) | w1
    v_odd = (w2 << 32) | w3
    return tl.where(odd == 1, v_odd, v_even)


@triton.jit
def philox_u01(seed, flat):
    """Return a float64 uniform in ``[0, 1)`` for a stream address.

    Args:
        seed: Philox key identifying the per-row stream.
        flat: Flat stream address; ``flat >> 1`` is the counter and
            ``flat & 1`` is the draw index.

    Returns:
        53-bit-precision uniform as float64.

    Note:
        Must stay bit-identical to ``counter_uniform`` and
        ``counter_uniform_cols`` in ``ayaka.sampling.rng``.
    """
    counter = flat >> 1
    v = _philox_u53_bits(seed, counter, flat & 1)
    return ((v >> 11) & ((1 << 53) - 1)).to(tl.float64) * (1.0 / (1 << 53))


@triton.jit
def philox_u01_f32(seed, flat):
    """Return a float32 uniform in ``[0, 1)`` for a stream address.

    Args:
        seed: Philox key identifying the per-row stream.
        flat: Flat stream address; ``flat >> 1`` is the counter and
            ``flat & 1`` is the draw index.

    Returns:
        24-bit-precision uniform as float32.

    Note:
        CDF-scan path used by ``topk_topp.py``. Keeps the high 24 bits of
        the draw so ``u`` is always exactly representable in float32.
        Casting the 53-bit value down could round to ``1.0``: harmless for
        Gumbel-argmax but a semantics change for the CDF scan versus the
        ported FlashInfer behavior.
    """
    counter = flat >> 1
    v = _philox_u53_bits(seed, counter, flat & 1)
    return ((v >> 29) & 0x00FFFFFF).to(tl.float32) / 16777216.0
