"""Counter-based PRNG: Philox4x32-10 for tensor sampling and SplitMix64 for seed derivation.

This module provides deterministic, counter-based PRNG primitives for sampling kernels
and host-side seed derivation. The tensor primitives (`counter_uniform`,
`counter_uniform_cols`) implement the Philox4x32-10 algorithm from Random123
(Salmon et al., SC'11); the scalar seed derivation (`derive_seed`,
`splitmix64_scalar`) keeps SplitMix64.

Design Rationale:
-----------------
1. Counter-Based vs Stateful Generator:
   Traditional PRNGs (such as ``torch.Generator``) maintain internal mutable state. In batched
   inference, the random numbers drawn by row `i` would depend on how many tokens previous rows
   consumed and the dynamic batch ordering. Counter-based RNG evaluates a pure mathematical
   function of `(seed, offset)`, guaranteeing:
   - Batch-Invariance: Sampling output for request `i` is completely independent of batch size
     or concurrent requests.
   - CUDA Graph Compatibility: No mutable generator state needs to be synchronized or updated
     during graph capture and replay.

2. Philox4x32-10 as the canonical stream:
   Philox is the same keyed family used by cuRAND/rocRand and numpy's ``Philox``; with 10
   rounds it has a safety margin over the known-good 8-round minimum and passes BigCrush.
   One counter (4x u32) yields 4x u32 which we pack into TWO uniform float64 draws in
   [0, 1) at 53-bit precision. The `seed` becomes the 2x u32 Philox key; the stream
   position is a flat 64-bit address ``flat = offset * n_cols + col`` (linear addressing,
   batch-invariant); the counter is ``flat >> 1`` and the draw index is ``flat & 1``.
   SplitMix64 previously played this role — outputs for a given (seed, offset) CHANGED
   when Philox replaced it (no external contract pinned the old stream; see the
   philox battery test which pins the algorithm from here on).

3. Signed 64-Bit Representation:
   PyTorch lacks a native unsigned 64-bit integer (`uint64`) dtype. All tensor operations use
   sign `torch.int64`. Modular arithmetic and bitwise operations produce bit-exact two's-complement
   results, with zero-extended right shifts handled explicitly via bit masking. 32x32 -> 64
   multiplication is computed from 16-bit halves because a raw int64 product of two
   full-width u32 words overflows the signed range.

4. Canonical Core:
   The Triton twin lives in ``ayaka.kernel.triton.sampling.philox`` and MUST stay
   bit-identical to this module for the same (seed, address) — the gumbel kernel and its
   torch oracle are cross-checked bit-exactly.
"""

from __future__ import annotations

import torch

__all__ = [
    "counter_uniform",
    "counter_uniform_cols",
    "philox4x32_10",
    "splitmix64_scalar",
    "derive_seed",
]

_MASK64 = 0xFFFFFFFFFFFFFFFF
_U32 = 0xFFFFFFFF
_MASK53 = (1 << 53) - 1
_GOLDEN_HEX = 0x9E3779B97F4A7C15
_C1_HEX = 0xBF58476D1CE4E5B9
_C2_HEX = 0x94D049BB133111EB
# Philox4x32 constants (Random123: PHILOX_M4x32_0/1, PHILOX_W32_0/1).
_M0 = 0xD2511F53
_M1 = 0xCD9E8D57
_W0 = 0x9E3779B9
_W1 = 0xBB67AE85
_ROUNDS = 10


def _lsr(x: torch.Tensor, k: int) -> torch.Tensor:
    """Perform a logical shift right on a signed int64 tensor.

    PyTorch's `>>` operator performs arithmetic right shift (sign-extending negative values).
    This function masks out the propagated sign bits to emulate an unsigned logical right shift.

    Args:
        x: Int64 input tensor.
        k: Number of bit positions to shift right (0 <= k < 64).

    Returns:
        Zero-extended right-shifted tensor.
    """
    return (x >> k) & ((1 << (64 - k)) - 1)


def _mulhilo32(a: int | torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Full 32x32 -> 64-bit product for two u32 values held in int64 tensors.

    A raw ``a * b`` overflows signed int64 (max product ≈ 2^64), so the product is
    reconstructed from 16-bit halves; every intermediate stays below 2^51.

    Returns:
        ``(hi, lo)`` int64 tensors, each holding a u32 value.
    """
    a_lo = a & 0xFFFF
    a_hi = a >> 16
    b_lo = b & 0xFFFF
    b_hi = b >> 16
    ll = a_lo * b_lo
    lh = a_lo * b_hi + a_hi * b_lo
    hh = a_hi * b_hi
    # product = (hh << 32) + (lh << 16) + ll; mid folds the low two terms so a
    # single carry term (mid >> 32) accounts for everything above bit 32.
    mid = ll + (lh << 16)
    lo = mid & 0xFFFFFFFF
    hi = (hh + (mid >> 32)) & 0xFFFFFFFF
    return hi, lo


def _philox_round(
    c0: torch.Tensor,
    c1: torch.Tensor,
    c2: torch.Tensor,
    c3: torch.Tensor,
    k0: torch.Tensor,
    k1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One Philox4x32 round (Salmon et al. 2011, Fig. 4): two mulhilo pairs and
    a hi/lo cross-XOR. Bit-exact with the reference ``_philox4x32round``."""
    hi0, lo0 = _mulhilo32(_M0, c0)
    hi1, lo1 = _mulhilo32(_M1, c2)
    return (hi1 ^ c1 ^ k0), lo1, (hi0 ^ c3 ^ k1), lo0


def philox4x32_10(
    c0: torch.Tensor,
    c1: torch.Tensor,
    c2: torch.Tensor,
    c3: torch.Tensor,
    k0: torch.Tensor,
    k1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Philox4x32 with 10 rounds: pure function of (counter c0..c3, key k0..k1).

    All inputs are int64 tensors holding u32 values. Returns the final counter
    state — the 4 u32 output words. Round 1 uses the given key; every later round
    bumps the key by (W0, W1), matching ``philox4x32round``/``bumpkey`` in
    Random123's ``philox.h``. Pinned by the Random123 KAT battery test.
    """
    for r in range(_ROUNDS):
        if r > 0:
            k0 = (k0 + _W0) & 0xFFFFFFFF
            k1 = (k1 + _W1) & 0xFFFFFFFF
        c0, c1, c2, c3 = _philox_round(c0, c1, c2, c3, k0, k1)
    return c0, c1, c2, c3


def _u01_from_pair(r_hi: torch.Tensor, r_lo: torch.Tensor) -> torch.Tensor:
    """Pack (r_hi, r_lo) into u64, take the top 53 bits, scale to [0, 1)."""
    v = (r_hi << 32) | r_lo
    return _lsr(v, 11).to(torch.float64) * (1.0 / float(1 << 53))


def _philox_u01(seed: torch.Tensor, flat: torch.Tensor) -> torch.Tensor:
    """Two float64 draws per Philox counter: counter = flat >> 1, draw = flat & 1.

    Draw 0 packs output words (w0, w1), draw 1 packs (w2, w3) — each as a 53-bit
    uniform in [0, 1).
    """
    k0 = seed & 0xFFFFFFFF
    k1 = _lsr(seed, 32)
    counter = flat >> 1
    c0 = counter & 0xFFFFFFFF
    c1 = _lsr(counter, 32)
    c2 = torch.zeros_like(c0)
    c3 = torch.zeros_like(c0)
    r0, r1, r2, r3 = philox4x32_10(c0, c1, c2, c3, k0, k1)
    u_even = _u01_from_pair(r0, r1)
    u_odd = _u01_from_pair(r2, r3)
    return torch.where((flat & 1).bool(), u_odd, u_even)


def counter_uniform(seed: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    """Generate uniform random floats in [0, 1) using Philox4x32-10.

    Evaluates a pure function of `(seed, offset)` for each batch element. The seed
    is the 128-bit key (2x u32); the stream position is the flat address
    ``offset`` (linear addressing with one column), split into counter
    ``offset >> 1`` and draw ``offset & 1``.

    Unlike the previous SplitMix64 core, there is no all-zero degeneracy: Philox
    with counter (0,0,0,0) and key (0,0) is a fixed pseudo-random quadruple, so
    `(seed, offset) == (0, 0)` still yields a proper draw.

    Args:
        seed: Int64 tensor of shape `[B]` containing per-sequence RNG seeds.
        offset: Int64 tensor of shape `[B]` containing per-sequence step/counter offsets.

    Returns:
        Float64 tensor of shape `[B]` with uniform values in [0, 1) using 53 bits of precision.
    """
    return _philox_u01(seed, offset)


def counter_uniform_cols(seed: torch.Tensor, offset: torch.Tensor, n_cols: int) -> torch.Tensor:
    """Generate a 2D matrix of uniform random floats in [0, 1) of shape [B, n_cols].

    Extends `counter_uniform` across multiple independent random draws per batch row (e.g. for
    parallel candidate sampling or multi-column stochastic operations).

    The effective stream address for row `i` and column `col` is linear:
    `flat = offset[i] * n_cols + col`, then split into counter `flat >> 1` and
    draw index `flat & 1` (two float64 draws per Philox counter). This preserves
    batch-invariance and avoids collisions across distinct columns and steps.

    Args:
        seed: Int64 tensor of shape `[B]` containing per-sequence RNG seeds.
        offset: Int64 tensor of shape `[B]` containing base step offsets.
        n_cols: Number of independent random samples to draw per batch element.

    Returns:
        Float64 tensor of shape `[B, n_cols]` with uniform values in [0, 1).
    """
    b = seed.size(0)
    cols = torch.arange(n_cols, device=seed.device, dtype=torch.int64)
    offset_2d = offset.unsqueeze(1) * n_cols + cols.unsqueeze(0)
    seed_2d = seed.unsqueeze(1).expand(b, n_cols)
    return _philox_u01(seed_2d.reshape(-1), offset_2d.reshape(-1)).reshape(b, n_cols)


def _finalize_scalar(z: int) -> int:
    """Apply the SplitMix64 bit-avalanche finalizer to a Python scalar integer.

    Emulates 64-bit unsigned integer modular multiplication and shifts using `_MASK64`.

    Args:
        z: Scalar integer representing intermediate state.

    Returns:
        Mixed 64-bit unsigned integer in [0, 2^64 - 1].
    """
    z = ((z ^ (z >> 30)) * _C1_HEX) & _MASK64
    z = ((z ^ (z >> 27)) * _C2_HEX) & _MASK64
    z ^= z >> 31
    return z


def splitmix64_scalar(x: int) -> int:
    """Execute one standard SplitMix64 state advancement step on a scalar integer.

    Adds the golden ratio constant modulo 2^64 and applies the mixing finalizer.
    Kept ONLY for scalar seed derivation — tensor streams use Philox4x32-10.

    Args:
        x: Input scalar state (e.g. counter or request index).

    Returns:
        Deterministic 64-bit pseudo-random unsigned integer.
    """
    return _finalize_scalar((x + _GOLDEN_HEX) & _MASK64)


def derive_seed(request_index: int) -> int:
    """Derive a deterministic, unique seed for requests that do not specify one.

    The generated seed is masked to non-negative 63-bit range (`0x7FFFFFFFFFFFFFFF`)
    because PyTorch raises an error if an integer greater than `2^63 - 1` is assigned
    to a signed `torch.int64` tensor column, rather than wrapping naturally.

    Args:
        request_index: Monotonically increasing request index or unique identifier.

    Returns:
        Non-negative integer in [0, 2^63 - 1] suitable for storage in `torch.int64`.
    """
    return splitmix64_scalar(request_index) & 0x7FFFFFFFFFFFFFFF
