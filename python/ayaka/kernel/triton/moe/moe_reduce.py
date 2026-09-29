"""MoE output reductions and row shuffles (Triton port of ``moe_sum.cu``, ``moe_sum_reduce.cu`` and the
``shuffle_rows`` / ``apply_shuffle_mul_sum`` part of ``prepare_moe_input.cu``).

Numerics
--------
* ``moe_sum_reduce`` / ``apply_shuffle_mul_sum``: fp32 accumulation, one final round-to-nearest-even
  to the output dtype (identical structure to the CUDA ``opmath_t`` kernels; the multiply-add is left
  to the compiler's FMA contraction just like ``nvcc -fmad=true``).
* ``moe_sum``: the CUDA kernel accumulates **in the tensor dtype** for ``topk in {2, 3, 4}``
  (``scalar_t x; x += ...`` -> every partial sum is rounded to fp16/bf16) and falls back to
  ``at::sum_out`` (fp32 accumulate, single rounding) for any other ``topk``.  The port reproduces
  both behaviours by default; pass ``accumulate_fp32=True`` to force fp32 accumulation everywhere.
* ``shuffle_rows`` is a pure byte copy (no dtype conversion), so fp8 / NaN payloads survive untouched.
"""

from __future__ import annotations

import struct

import torch
import triton
import triton.language as tl

from ayaka.kernel.triton.moe._common import (
    ROW_CONFIGS,
    SHUFFLE_CONFIGS,
    autotune,
    cdiv,
    check,
    check_tensor,
    device_guard,
    f32,
    next_pow2,
    same_device,
)

_FLOAT_DTYPES = (torch.float32, torch.float16, torch.bfloat16)

# 128-bit accesses need >= 16 B per thread: BLOCK_D / (32 * num_warps) >= 8 elements for 16-bit dtypes.
# Was a private copy; now the shared bank in _internal.tuning.
_ROW_CONFIGS = ROW_CONFIGS


@autotune(configs=_ROW_CONFIGS, key=["hidden_size", "TOPK", "ROUND_EACH_STEP", "token_bucket"])
@triton.jit
def moe_sum_kernel(
    out_ptr,
    in_ptr,
    hidden_size,
    token_bucket,
    TOPK: tl.constexpr,
    ROUND_EACH_STEP: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """out[t, :] = sum_k in[t, k, :]   (in: [T, TOPK, D] contiguous, out: [T, D] contiguous)."""
    token = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK_D), BLOCK_D)
    mask = offs < hidden_size
    in_base = in_ptr + token * TOPK * hidden_size
    if ROUND_EACH_STEP:
        # CUDA: `scalar_t x = 0; x += in[k]` -> the partial sum is rounded to scalar_t after each add
        acc = tl.zeros([BLOCK_D], dtype=in_ptr.dtype.element_ty)
        for k in tl.static_range(TOPK):
            x = tl.load(in_base + k * hidden_size + offs, mask=mask, other=0.0)
            acc = (acc.to(tl.float32) + x.to(tl.float32)).to(in_ptr.dtype.element_ty)
    else:
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        for k in tl.static_range(TOPK):
            x = tl.load(in_base + k * hidden_size + offs, mask=mask, other=0.0)
            acc += x.to(tl.float32)
    tl.store(out_ptr + token * hidden_size + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


def moe_sum(input: torch.Tensor, output: torch.Tensor, accumulate_fp32: bool = False) -> None:
    """``output[t] = input[t].sum(dim=0)`` for ``input: [num_tokens, topk, hidden]`` (contiguous).

    See the module docstring for the rounding behaviour of the 16-bit paths.
    """
    check_tensor(input, "input", dtypes=_FLOAT_DTYPES, ndim=3, contiguous=True)
    check_tensor(output, "output", dtypes=(input.dtype,), ndim=2, contiguous=True)
    same_device(input, output)
    hidden = input.size(-1)
    topk = input.size(1)
    check(hidden > 0 and topk > 0, "empty reduction")
    check(output.size(1) == hidden, "output hidden_size must match input hidden_size")
    check(input.size(0) == output.size(0), "input and output token dimensions must match")
    num_tokens = output.size(0)
    if num_tokens == 0:
        return
    round_each_step = (
        (not accumulate_fp32)
        and topk in (2, 3, 4)
        and input.dtype in (torch.float16, torch.bfloat16)
    )
    with device_guard(output):
        moe_sum_kernel[lambda meta: (num_tokens, cdiv(hidden, meta["BLOCK_D"]))](
            output,
            input,
            hidden,
            next_pow2(num_tokens),
            TOPK=topk,
            ROUND_EACH_STEP=round_each_step,
        )


# ======================================================================================
# moe_sum_reduce
# ======================================================================================
@autotune(configs=_ROW_CONFIGS, key=["hidden_dim", "TOPK", "token_bucket", "ACC_DTYPE"])
@triton.jit
def moe_sum_reduce_kernel(
    x_ptr,
    y_ptr,
    scale_bits,
    hidden_dim,
    token_bucket,
    stride_token,
    stride_topk,
    out_stride_token,
    TOPK: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """y[t, :] = (sum_k acc_t(x[t, k, :])) * scale  -> round-to-nearest-even to y's dtype.

    ``ACC_DTYPE`` mirrors ``at::opmath_type<scalar_t>``: float32 for fp16/bf16/fp32 input, float64 for
    float64 input (opmath_type<double> == double, i.e. no upcast — the CUDA op accumulates fp64 natively).
    ``scale_bits`` contains the IEEE-754 bits of the scale. Passing those bits as an integer scalar
    avoids a temporary device allocation and preserves the full float64 scale during CUDA Graph capture.

    One program = one token x BLOCK_D hidden columns (coalesced, up to 128-bit per thread).  Covers
    moe_sum_reduce_kernel / *_warp_token_topk / *_general / warp_per_token_vec_kernel of the CUDA file:
    they differ only in launch geometry, the arithmetic is identical.
    """
    token = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK_D), BLOCK_D)
    mask = offs < hidden_dim
    base = x_ptr + token * stride_token + offs
    if ACC_DTYPE == tl.float64:
        scale = tl.cast(scale_bits, tl.float64, bitcast=True)
    else:
        scale = tl.cast(scale_bits, tl.float32, bitcast=True)
    acc = tl.zeros([BLOCK_D], dtype=ACC_DTYPE)
    for k in tl.static_range(TOPK):
        v = tl.load(base + k * stride_topk, mask=mask, other=0.0)
        acc += v.to(ACC_DTYPE)
    acc = acc * scale
    tl.store(y_ptr + token * out_stride_token + offs, acc.to(y_ptr.dtype.element_ty), mask=mask)


def moe_sum_reduce(input: torch.Tensor, output: torch.Tensor, routed_scaling_factor: float) -> None:
    """``output = input.sum(dim=1) * routed_scaling_factor`` for ``input: [tokens, topk, hidden]``.

    fp32 accumulation (fp64 in, fp64 accumulation, matching ``at::opmath_type``), scaling in that same
    precision (``static_cast<opmath_t<scalar_t>>(routed_scaling_factor)``), one final rounding to the
    output dtype.
    """
    check_tensor(input, "input", dtypes=_FLOAT_DTYPES + (torch.float64,), ndim=3, contiguous=False)
    check_tensor(output, "output", dtypes=(input.dtype,), ndim=2, contiguous=False)
    same_device(input, output)
    check(input.size(0) == output.size(0), "token dim mismatch")
    check(input.size(2) == output.size(1), "hidden_dim mismatch")
    check(input.is_contiguous(), "expect input to be contiguous")
    check(output.is_contiguous(), "expect output to be contiguous")
    token_num, topk, hidden = input.shape
    if token_num == 0 or hidden == 0:
        return
    check(topk > 0, "topk must be positive")
    is_f64 = input.dtype == torch.float64
    acc_dtype = tl.float64 if is_f64 else tl.float32
    if is_f64:
        scale_bits = struct.unpack("q", struct.pack("d", float(routed_scaling_factor)))[0]
    else:
        scale_val = f32(routed_scaling_factor)
        scale_bits = struct.unpack("i", struct.pack("f", scale_val))[0]
    with device_guard(output):
        moe_sum_reduce_kernel[lambda meta: (token_num, cdiv(hidden, meta["BLOCK_D"]))](
            input,
            output,
            scale_bits,
            hidden,
            next_pow2(token_num),
            input.stride(0),
            input.stride(1),
            output.stride(0),
            TOPK=topk,
            ACC_DTYPE=acc_dtype,
        )


# ======================================================================================
# shuffle_rows  (prepare_moe_input.cu :: shuffleRowsKernel)
# ======================================================================================
@autotune(configs=SHUFFLE_CONFIGS, key=["row_bytes"])
@triton.jit
def shuffle_rows_kernel(
    input_bytes_ptr,
    output_bytes_ptr,
    permutation_ptr,
    num_rows,
    row_bytes,
    BLOCK: tl.constexpr,
):
    """Copy each selected source row as bytes, preserving all dtype bit patterns."""
    row = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    src_row = tl.load(permutation_ptr + row).to(tl.int64)
    values = tl.load(
        input_bytes_ptr + src_row * row_bytes + offs,
        mask=(row < num_rows) & (offs < row_bytes),
        other=0,
    )
    tl.store(output_bytes_ptr + row * row_bytes + offs, values, mask=offs < row_bytes)


def shuffle_rows(
    input: torch.Tensor,
    output: torch.Tensor,
    permutation: torch.Tensor,
) -> None:
    """Gather source rows into ``output`` using a bit-exact, dtype-agnostic byte copy.

    ``permutation[row]`` selects the source row for ``output[row]``. The output may have a
    different row count from the input, but both tensors must be contiguous 2D tensors of the same
    dtype and width. Input and output storage must be disjoint because different rows may be read in
    parallel.
    """
    check_tensor(input, "input", ndim=2, contiguous=True)
    check_tensor(output, "output", dtypes=(input.dtype,), ndim=2, contiguous=True)
    check_tensor(permutation, "permutation", dtypes=(torch.int32,), ndim=1, contiguous=True)
    same_device(input, output, permutation)
    check(input.size(1) == output.size(1), "input / output row width mismatch")
    check(output.size(0) == permutation.numel(), "permutation length must match output rows")
    check(input.size(0) > 0 or output.size(0) == 0, "cannot gather from an empty input")
    check(
        input.untyped_storage().data_ptr() != output.untyped_storage().data_ptr(),
        "input and output must not share storage",
    )
    row_bytes = input.size(1) * input.element_size()
    if output.size(0) == 0 or row_bytes == 0:
        return
    with device_guard(output):
        shuffle_rows_kernel[lambda meta: (output.size(0), cdiv(row_bytes, meta["BLOCK"]))](
            input.view(torch.uint8),
            output.view(torch.uint8),
            permutation,
            output.size(0),
            row_bytes,
        )


# ======================================================================================
# apply_shuffle_mul_sum  (prepare_moe_input.cu :: apply_shuffle_mul_sum_kernel)
# ======================================================================================
@autotune(configs=_ROW_CONFIGS, key=["row_stride", "TOPK", "HAS_FACTORS", "token_bucket"])
@triton.jit
def apply_shuffle_mul_sum_kernel(
    in_ptr,
    out_ptr,
    perm_ptr,
    factors_ptr,
    row_stride,
    token_bucket,
    TOPK: tl.constexpr,
    HAS_FACTORS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """out[i, :] = sum_j factors[i*TOPK + j] * in[perm[i*TOPK + j], :]   (fp32 accumulate, one rounding)."""
    i = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK_D), BLOCK_D)
    mask = offs < row_stride
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for j in tl.static_range(TOPK):
        idx = i * TOPK + j
        src = tl.load(perm_ptr + idx).to(tl.int64)
        v = tl.load(in_ptr + src * row_stride + offs, mask=mask, other=0.0).to(tl.float32)
        if HAS_FACTORS:
            f = tl.load(factors_ptr + idx).to(tl.float32)
            acc += f * v
        else:
            acc += v
    tl.store(out_ptr + i * row_stride + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


def apply_shuffle_mul_sum(
    input: torch.Tensor,
    output: torch.Tensor,
    permutation: torch.Tensor,
    factors: torch.Tensor | None = None,
) -> None:
    """Gather ``topk`` rows of ``input`` per output row (via ``permutation``), scale and sum them.

    ``input: [m * topk, n]``, ``output: [m, n]``, ``permutation: [m * topk]`` int32,
    ``factors: [m * topk]`` or ``[m, topk]`` (same dtype as ``output``, contiguous) or ``None`` (== all ones).
    """
    dtypes = (torch.float32, torch.float16, torch.bfloat16)
    check_tensor(input, "input", dtypes=dtypes, ndim=2, contiguous=True)
    check_tensor(output, "output", dtypes=(input.dtype,), ndim=2, contiguous=True)
    check_tensor(permutation, "permutation", dtypes=(torch.int32,), ndim=1, contiguous=True)
    same_device(input, output, permutation)
    m, n = output.shape
    check(input.size(1) == n, "input / output row width mismatch")
    check(input.size(0) == permutation.size(0), "input row count must match permutation length")
    if m == 0:
        # the CUDA host code computes `topk = permutation.numel() / m` *before* checking sizes, which is a
        # division by zero (undefined behaviour) when m == 0; a graceful no-op is the safe generalisation.
        check(permutation.size(0) == 0, "permutation length must be a multiple of the output rows")
        return
    check(permutation.size(0) % m == 0, "permutation length must be a multiple of the output rows")
    topk = permutation.size(0) // m
    if factors is not None:
        check_tensor(
            factors, "factors", dtypes=(output.dtype,), contiguous=True
        )  # any shape, [m * topk] or [m, topk], contiguous
        same_device(input, factors)
        check(factors.numel() == permutation.size(0), "Factors must have shape [m * topk]")
    if n == 0:
        return
    with device_guard(output):
        apply_shuffle_mul_sum_kernel[lambda meta: (m, cdiv(n, meta["BLOCK_D"]))](
            input,
            output,
            permutation,
            factors if factors is not None else input,  # dummy pointer, never dereferenced
            n,
            next_pow2(m),
            TOPK=topk,
            HAS_FACTORS=factors is not None,
        )
