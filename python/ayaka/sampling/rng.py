"""Stateless counter-based pseudo-random number generator (SplitMix64).

This module provides deterministic, counter-based PRNG primitives for sampling kernels
and host-side seed derivation. It implements the SplitMix64 algorithm across both PyTorch
tensor operations and pure Python integers.

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

2. Signed 64-Bit Representation:
   PyTorch lacks a native unsigned 64-bit integer (`uint64`) dtype. All tensor operations use
   sign `torch.int64`. Modular arithmetic and bitwise operations produce bit-exact two's-complement
   results, with zero-extended right shifts handled explicitly via bit masking.

3. Canonical Core:
   Unifies tensor sampling (`counter_uniform`, `counter_uniform_cols`) and scalar request seed
   derivation (`derive_seed`, `splitmix64_scalar`) under identical SplitMix64 mixing constants.
"""

from __future__ import annotations

import torch

__all__ = [
    "GOLDEN",
    "C1",
    "C2",
    "i64",
    "counter_uniform",
    "counter_uniform_cols",
    "splitmix64_scalar",
    "derive_seed",
]

_MASK64 = 0xFFFFFFFFFFFFFFFF
_GOLDEN_HEX = 0x9E3779B97F4A7C15
_C1_HEX = 0xBF58476D1CE4E5B9
_C2_HEX = 0x94D049BB133111EB


def i64(x: int) -> int:
    """Convert an unsigned 64-bit integer into a signed two's-complement 64-bit int.

    Necessary because PyTorch `int64` tensors are signed, whereas standard SplitMix64
    constants are 64-bit unsigned integers.

    Args:
        x: An integer in the range [0, 2^64 - 1].

    Returns:
        Signed 64-bit integer in the range [-2^63, 2^63 - 1].
    """
    return x - (1 << 64) if x >= (1 << 63) else x


GOLDEN = i64(_GOLDEN_HEX)
C1 = i64(_C1_HEX)
C2 = i64(_C2_HEX)


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


def _finalize_tensor(z: torch.Tensor) -> torch.Tensor:
    """Apply the SplitMix64 bit-avalanche finalizer to an int64 tensor.

    Mixes the bits of `z` through three rounds of logical right shifts, XORs, and
    multiplications with constants C1 and C2.

    Args:
        z: Int64 tensor containing intermediate PRNG state.

    Returns:
        Scrambled int64 tensor with high avalanche quality.
    """
    z = (z ^ _lsr(z, 30)) * C1
    z = (z ^ _lsr(z, 27)) * C2
    z = z ^ _lsr(z, 31)
    return z


def counter_uniform(seed: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    """Generate uniform random floats in [0, 1) using counter-based SplitMix64.

    Evaluates a pure function of `(seed, offset)` for each batch element.

    Note:
        We use `(offset + 1)` rather than `offset` directly because SplitMix64 maps 0 to 0.
        If `seed == 0` and `offset == 0`, using raw `offset` would yield `u = 0.0`, which would
        unconditionally select the top-1 token in sampling kernels.

    Args:
        seed: Int64 tensor of shape `[B]` containing per-sequence RNG seeds.
        offset: Int64 tensor of shape `[B]` containing per-sequence step/counter offsets.

    Returns:
        Float64 tensor of shape `[B]` with uniform values in [0, 1) using 53 bits of precision.
    """
    z = seed + (offset + 1) * GOLDEN
    z = _finalize_tensor(z)
    return _lsr(z, 11).to(torch.float64) * (1.0 / float(1 << 53))


def counter_uniform_cols(seed: torch.Tensor, offset: torch.Tensor, n_cols: int) -> torch.Tensor:
    """Generate a 2D matrix of uniform random floats in [0, 1) of shape [B, n_cols].

    Extends `counter_uniform` across multiple independent random draws per batch row (e.g. for
    parallel candidate sampling or multi-column stochastic operations).

    The effective counter address for row `i` and column `col` is linear:
    `flat_offset = offset[i] * n_cols + col`. This preserves batch-invariance and avoids collisions
    across distinct columns.

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
    flat = counter_uniform(seed_2d.reshape(-1), offset_2d.reshape(-1))
    return flat.reshape(b, n_cols)


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
