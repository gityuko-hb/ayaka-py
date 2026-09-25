"""Generic paged KV-cache operations."""

from __future__ import annotations

import math
import os
import re
from typing import Any, cast

import torch
import triton
import triton.language as tl
from ayaka.kernel.triton.cache.cache_helpers import (
    GATHER_BLOCK_T,
    KV_OFFLOAD_MAX_BATCH_DESCRIPTORS_ENV,
    NVFP4_MSG,
    ROCM_DEFAULT_MAX_BATCH_DESCRIPTORS,
    as_byte_view,
    as_int_view,
    cache_kernel_view,
    dequantize_if_fp8,
    div_rn,
    next_power_of_2_f32,
    normalize_kv_cache_dtype,
    quantize_if_fp8,
    require_kv_source_dtype,
    resolve_token_locations,
    warps_for_tile,
)
from ayaka.types import KVCacheDtype
from ayaka.utils.math_utils import FP8_E4M3_MAX


@triton.jit
def swap_blocks_batch_kernel(
    src_addrs_ptr,  # int64 [n]  raw source addresses
    dst_addrs_ptr,  # int64 [n]  raw destination addresses
    sizes_ptr,  # int64 [n]  bytes to copy per descriptor
    BLOCK: tl.constexpr,
    WORD: tl.constexpr,  # copy 8-byte words (all addresses/sizes 8B aligned)
):
    """Batched memcpy: program (i, j) copies chunk j of descriptor i."""
    desc = tl.program_id(0)
    chunk = tl.program_id(1).to(tl.int64)
    src = tl.load(src_addrs_ptr + desc)
    dst = tl.load(dst_addrs_ptr + desc)
    nbytes = tl.load(sizes_ptr + desc)

    if WORD:
        n = nbytes // 8
        src_p = src.to(tl.pointer_type(tl.int64))
        dst_p = dst.to(tl.pointer_type(tl.int64))
    else:
        n = nbytes
        src_p = src.to(tl.pointer_type(tl.uint8))
        dst_p = dst.to(tl.pointer_type(tl.uint8))

    offs = chunk * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(dst_p + offs, tl.load(src_p + offs, mask=mask), mask=mask)


@triton.jit
def reshape_and_cache_kernel(
    key_ptr,  # [num_tokens, num_heads, head_size]
    value_ptr,  # [num_tokens, num_heads, head_size]
    key_cache_ptr,  # [num_blocks, num_heads, head_size/x, block_size, x]
    value_cache_ptr,  # [num_blocks, num_heads, head_size, block_size]
    slot_mapping_ptr,  # [num_tokens] int64
    k_scale_ptr,
    v_scale_ptr,
    key_stride_t,
    key_stride_h,
    value_stride_t,
    value_stride_h,
    kc_stride_b,
    kc_stride_h,
    kc_stride_c,  # stride of the head_size/x dimension
    kc_stride_s,  # stride of the block_size (slot) dimension
    vc_stride_b,
    vc_stride_h,
    vc_stride_d,  # stride of the head_size dimension
    block_size,
    HEAD_SIZE: tl.constexpr,
    X: tl.constexpr,
    BLOCK_D: tl.constexpr,
    KV_QUANTIZED: tl.constexpr,
    IS_E5M2: tl.constexpr,
):
    """Write K/V into the classic vLLM-style paged cache layout."""
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    slot = tl.load(slot_mapping_ptr + token)
    if slot < 0:
        return

    block_idx = slot // block_size
    block_off = slot % block_size
    d = tl.arange(0, BLOCK_D)
    mask = d < HEAD_SIZE

    k = tl.load(
        key_ptr + token * key_stride_t + head * key_stride_h + d,
        mask=mask,
        other=0.0,
    )
    v = tl.load(
        value_ptr + token * value_stride_t + head * value_stride_h + d,
        mask=mask,
        other=0.0,
    )

    if KV_QUANTIZED:
        k_scale = tl.load(k_scale_ptr)
        v_scale = tl.load(v_scale_ptr)
    else:
        k_scale = 1.0
        v_scale = 1.0

    k_out = quantize_if_fp8(k, k_scale, KV_QUANTIZED, IS_E5M2)
    v_out = quantize_if_fp8(v, v_scale, KV_QUANTIZED, IS_E5M2)

    # key_cache[block, head, d // x, slot, d % x]
    k_dst = (
        key_cache_ptr
        + block_idx * kc_stride_b
        + head * kc_stride_h
        + (d // X) * kc_stride_c
        + block_off * kc_stride_s
        + (d % X)
    )
    tl.store(k_dst, k_out, mask=mask)

    # value_cache[block, head, d, slot]
    v_dst = (
        value_cache_ptr + block_idx * vc_stride_b + head * vc_stride_h + d * vc_stride_d + block_off
    )
    tl.store(v_dst, v_out, mask=mask)


@triton.jit
def reshape_and_cache_flash_kernel(
    key_ptr,  # [num_tokens, num_heads, head_size]
    value_ptr,  # [num_tokens, num_heads, head_size]
    key_cache_ptr,  # [num_blocks, block_size, num_heads, head_size] (any strides)
    value_cache_ptr,
    slot_mapping_ptr,  # [num_tokens] int64
    k_scale_ptr,  # float32 [1] or [num_heads]
    v_scale_ptr,
    key_stride_t,
    key_stride_h,
    value_stride_t,
    value_stride_h,
    block_stride,
    page_stride,
    head_stride,
    block_size,
    num_heads,
    kv_scale_stride,  # 0 -> one scale for all heads, 1 -> per-head scales
    HEAD_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HEADS_PER_PROG: tl.constexpr,
    KV_QUANTIZED: tl.constexpr,
    IS_E5M2: tl.constexpr,
    SCALAR_SCALES: tl.constexpr,
    INVERSE_SCALES: tl.constexpr,
):
    """Write K/V into ``[block, token, head, dim]`` cache views."""
    token = tl.program_id(0).to(tl.int64)
    slot = tl.load(slot_mapping_ptr + token)
    if slot < 0:
        return

    block_idx = slot // block_size
    block_off = slot % block_size
    h = tl.program_id(1) * HEADS_PER_PROG + tl.arange(0, HEADS_PER_PROG)
    hmask = h < num_heads
    d = tl.arange(0, BLOCK_D)
    mask = hmask[:, None] & (d < HEAD_SIZE)[None, :]
    h64 = h.to(tl.int64)[:, None]

    k = tl.load(
        key_ptr + token * key_stride_t + h64 * key_stride_h + d[None, :],
        mask=mask,
        other=0.0,
    )
    v = tl.load(
        value_ptr + token * value_stride_t + h64 * value_stride_h + d[None, :],
        mask=mask,
        other=0.0,
    )

    if KV_QUANTIZED:
        if SCALAR_SCALES:
            k_scale = k_scale_ptr
            v_scale = v_scale_ptr
        else:
            k_scale = tl.load(k_scale_ptr + h * kv_scale_stride, mask=hmask, other=1.0)[:, None]
            v_scale = tl.load(v_scale_ptr + h * kv_scale_stride, mask=hmask, other=1.0)[:, None]
    else:
        k_scale = 1.0
        v_scale = 1.0

    if KV_QUANTIZED and INVERSE_SCALES:
        k = k.to(tl.float32) * k_scale
        v = v.to(tl.float32) * v_scale
        k_scale = 1.0
        v_scale = 1.0
    k_out = quantize_if_fp8(k, k_scale, KV_QUANTIZED, IS_E5M2)
    v_out = quantize_if_fp8(v, v_scale, KV_QUANTIZED, IS_E5M2)

    dst = block_idx * block_stride + block_off * page_stride + h64 * head_stride + d[None, :]
    tl.store(key_cache_ptr + dst, k_out, mask=mask)
    tl.store(value_cache_ptr + dst, v_out, mask=mask)


@triton.jit
def indexer_k_quant_and_cache_kernel(
    k_ptr,  # [num_tokens, head_dim]
    kv_cache_ptr,  # uint8 [num_blocks, block_size, head_dim + head_dim/qbs*4]
    slot_mapping_ptr,
    k_stride,
    cache_block_stride,  # in bytes
    cache_block_size,
    head_dim,
    QUANT_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_QBLOCKS: tl.constexpr,
    USE_UE8M0: tl.constexpr,
    SCALE_DIVISOR: tl.constexpr,
):
    """Program (token, j) quantises the j-th ``QUANT_BLOCK`` slice of a row.

    Cache block layout: ``block_size * head_dim`` fp8 bytes followed by
    ``block_size * head_dim / QUANT_BLOCK`` float32 scales.
    """
    token = tl.program_id(0).to(tl.int64)
    qb = tl.program_id(1)
    slot = tl.load(slot_mapping_ptr + token)
    if slot < 0:
        return

    block_idx = slot // cache_block_size
    block_off = slot % cache_block_size
    lane = tl.arange(0, BLOCK)
    mask = lane < QUANT_BLOCK
    offs = qb * QUANT_BLOCK + lane

    x = tl.load(k_ptr + token * k_stride + offs, mask=mask, other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=0)
    scale = div_rn(tl.maximum(amax, 1e-4), SCALE_DIVISOR)
    if USE_UE8M0:
        scale = next_power_of_2_f32(scale)

    # The sparse-indexer cache is intentionally fixed to E4M3.
    q = quantize_if_fp8(x, scale, True, False)

    base = kv_cache_ptr + block_idx * cache_block_stride
    tl.store(base + block_off * head_dim + offs, q, mask=mask)
    s_ptr = (base + cache_block_size * head_dim).to(tl.pointer_type(tl.float32))
    tl.store(s_ptr + (block_off * NUM_QBLOCKS + qb), scale)


@triton.jit
def cp_gather_indexer_k_quant_cache_kernel(
    kv_cache_ptr,  # uint8, layout of indexer_k_quant_and_cache_kernel
    dst_k_ptr,  # uint8 [num_tokens, head_dim]
    dst_scale_ptr,  # int32 view of the float32 scales [num_tokens, NUM_QBLOCKS]
    block_table_ptr,  # int32 [batch, block_table_cols]
    cu_seq_lens_ptr,  # int32 [batch + 1]
    batch_size,
    num_tokens,
    num_iters,
    block_table_stride,
    block_table_cols,
    cache_block_stride,
    cache_block_size,
    head_dim,
    dst_k_stride,
    dst_scale_stride,
    HEAD_DIM: tl.constexpr,
    NUM_QBLOCKS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    tok = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    ok, req, pos = resolve_token_locations(
        tok,
        num_tokens,
        cu_seq_lens_ptr,
        cu_seq_lens_ptr,
        batch_size,
        num_iters,
        False,
        True,
    )

    logical = pos // cache_block_size
    in_block = pos - logical * cache_block_size
    ok = ok & (logical < block_table_cols)
    phys = tl.load(
        block_table_ptr + req.to(tl.int64) * block_table_stride + logical,
        mask=ok,
        other=0,
    ).to(tl.int64)

    blk_base = kv_cache_ptr + phys * cache_block_stride
    row = blk_base + in_block.to(tl.int64) * head_dim
    dst_row = dst_k_ptr + tok.to(tl.int64) * dst_k_stride

    for d0 in tl.static_range(0, HEAD_DIM, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        m = ok[:, None] & (d < HEAD_DIM)[None, :]
        val = tl.load(row[:, None] + d[None, :], mask=m, other=0)
        tl.store(dst_row[:, None] + d[None, :], val, mask=m)

    s = tl.arange(0, BLOCK_S)
    sm = ok[:, None] & (s < NUM_QBLOCKS)[None, :]
    s_src = (blk_base + cache_block_size * head_dim).to(tl.pointer_type(tl.int32))
    sv = tl.load(
        s_src[:, None] + (in_block[:, None] * NUM_QBLOCKS + s[None, :]),
        mask=sm,
        other=0,
    )
    tl.store(
        dst_scale_ptr + tok.to(tl.int64)[:, None] * dst_scale_stride + s[None, :],
        sv,
        mask=sm,
    )


@triton.jit
def convert_fp8_kernel(
    src_ptr,
    dst_ptr,
    scale,
    numel,
    BLOCK: tl.constexpr,
    KV_QUANTIZED: tl.constexpr,
    IS_E5M2: tl.constexpr,
    TO_CACHE: tl.constexpr,
):
    """Flat cache conversion in either direction."""
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(src_ptr + offs, mask=mask, other=0)

    if TO_CACHE:
        y = quantize_if_fp8(x, scale, KV_QUANTIZED, IS_E5M2)
    else:
        y = dequantize_if_fp8(x, scale, KV_QUANTIZED, IS_E5M2)

    tl.store(dst_ptr + offs, y.to(dst_ptr.dtype.element_ty), mask=mask)


@triton.jit
def gather_cache_kernel(
    src_cache_ptr,  # [num_blocks, block_size, entry...]
    dst_ptr,  # [num_tokens, entry...]
    block_table_ptr,  # int32 [num_reqs, block_table_cols]
    cu_seq_lens_ptr,  # int32 [num_reqs + 1]
    seq_starts_ptr,  # int32 [num_reqs] (only read if HAS_SEQ_STARTS)
    scale_ptr,  # float32 [1] (only read if FP8 != 0)
    num_reqs,
    num_tokens,
    block_size,
    block_table_cols,
    num_iters,
    block_table_stride,
    cache_block_stride,
    cache_entry_stride,
    dst_entry_stride,
    ENTRY_SIZE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_E: tl.constexpr,
    KV_QUANTIZED: tl.constexpr,
    IS_E5M2: tl.constexpr,
    HAS_SEQ_STARTS: tl.constexpr,
    DIRECT_SLOTS: tl.constexpr,
    SCALAR_SCALE: tl.constexpr,
):
    """Gather paged cache entries and optionally dequantize FP8."""
    tok = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    if DIRECT_SLOTS:
        ok = tok < num_tokens
        slot = tl.load(block_table_ptr + tok, mask=ok, other=0).to(tl.int64)
        phys = slot // block_size
        in_block = slot % block_size
    else:
        ok, req, pos = resolve_token_locations(
            tok,
            num_tokens,
            cu_seq_lens_ptr,
            seq_starts_ptr,
            num_reqs,
            num_iters,
            HAS_SEQ_STARTS,
            True,
        )
        logical = pos // block_size
        in_block = pos - logical * block_size
        ok = ok & (logical < block_table_cols)
        phys = tl.load(
            block_table_ptr + req.to(tl.int64) * block_table_stride + logical,
            mask=ok,
            other=0,
        ).to(tl.int64)

    src_row = src_cache_ptr + phys * cache_block_stride + in_block.to(tl.int64) * cache_entry_stride
    dst_row = dst_ptr + tok.to(tl.int64) * dst_entry_stride
    if KV_QUANTIZED:
        if SCALAR_SCALE:
            scale = scale_ptr
        else:
            scale = tl.load(scale_ptr)
    else:
        scale = 1.0

    for e0 in tl.static_range(0, ENTRY_SIZE, BLOCK_E):
        e = e0 + tl.arange(0, BLOCK_E)
        m = ok[:, None] & (e < ENTRY_SIZE)[None, :]
        x = tl.load(src_row[:, None] + e[None, :], mask=m, other=0)
        y = dequantize_if_fp8(x, scale, KV_QUANTIZED, IS_E5M2)
        if DIRECT_SLOTS and KV_QUANTIZED and not IS_E5M2:
            # Public storage reads preserve E4M3 NaN codes as PyTorch does.
            y = tl.where((x & 0x7F) == 0x7F, float("nan"), y)
        tl.store(dst_row[:, None] + e[None, :], y.to(dst_ptr.dtype.element_ty), mask=m)


def _resolve_max_batch_desc() -> int:
    """Max descriptors per batched-copy launch (0 = unlimited)."""
    env = os.environ.get(KV_OFFLOAD_MAX_BATCH_DESCRIPTORS_ENV)
    if env:
        m = re.match(r"\s*[+-]?\d+", env)  # atoll semantics
        override = int(m.group()) if m else 0
        if override > 0:
            return override
    return ROCM_DEFAULT_MAX_BATCH_DESCRIPTORS if torch.version.hip is not None else 0


def _check(condition: bool, *message: object) -> None:
    if not condition:
        raise RuntimeError("".join(str(part) for part in message))


def _device_index(tensor: torch.Tensor) -> int:
    index = tensor.device.index
    return torch.cuda.current_device() if index is None else index


def _same_device(ref: torch.Tensor, ref_name: str, **others: torch.Tensor | None) -> None:
    for name, tensor in others.items():
        if tensor is not None:
            _check(
                ref.device == tensor.device, ref_name, " and ", name, " must be on the same device"
            )


def _dtype_flags(value: KVCacheDtype | str) -> tuple[KVCacheDtype, bool, bool]:
    dtype = normalize_kv_cache_dtype(value)
    quantized = bool(dtype.is_quantized)
    is_e5m2 = quantized and dtype.value == "fp8_e5m2"
    return dtype, quantized, is_e5m2


def _validate_quant_scales(
    quantized: bool,
    ref: torch.Tensor,
    **scales: torch.Tensor,
) -> None:
    if not quantized:
        return
    for name, scale in scales.items():
        _check(scale.dtype == torch.float32, name, " must be float32")
        _same_device(ref, "reference tensor", **{name: scale})


def _launch_batched_copy(src_addrs: torch.Tensor, dst_addrs: torch.Tensor, sizes: torch.Tensor):
    """Copy ``sizes[i]`` bytes from ``src_addrs[i]`` to ``dst_addrs[i]``.

    The three arguments are 1-D int64 CPU tensors.  All addresses must be
    dereferenceable from the current GPU (device memory, or pinned/mapped host
    memory through UVA).
    """
    n = src_addrs.numel()
    if n == 0:
        return
    max_size = int(sizes.max())
    if max_size <= 0:
        return
    word = bool((((src_addrs | dst_addrs | sizes) & 7) == 0).all())
    units = max_size // 8 if word else max_size
    block = 1024 if word else 2048
    while triton.cdiv(units, block) > 65535:  # grid.y limit
        block *= 2
    device = torch.device("cuda", torch.cuda.current_device())
    desc = torch.stack((src_addrs, dst_addrs, sizes)).to(device, non_blocking=True)
    step = _resolve_max_batch_desc()
    if step <= 0:
        step = n
    for off in range(0, n, step):
        cnt = min(step, n - off)
        cast(
            Any,
            swap_blocks_batch_kernel[(cnt, triton.cdiv(units, block))](
                desc[0, off : off + cnt],
                desc[1, off : off + cnt],
                desc[2, off : off + cnt],
                BLOCK=block,
                WORD=word,
                num_warps=4,
            ),
        )


def swap_blocks(
    src: torch.Tensor,
    dst: torch.Tensor,
    block_size_in_bytes: int,
    block_mapping: torch.Tensor,
) -> None:
    """Copy whole blocks ``src[m[i, 0]] -> dst[m[i, 1]]``.

    * GPU -> GPU (same device): one Triton launch for all pairs.
    * GPU <-> CPU: per-block ``copy_(non_blocking=True)`` on the current stream
      (copy engine / DMA, like ``cudaMemcpyAsync``); pinned host memory is
      needed for the copies to be asynchronous.
    """
    src_gpu = src.device.type == "cuda"
    dst_gpu = dst.device.type == "cuda"
    if src_gpu and dst_gpu:
        _check(src.device == dst.device, "src and dst must be on the same GPU")
    elif not (src_gpu or dst_gpu):
        raise RuntimeError("Invalid device combination: at least one tensor must be on a GPU")
    _check(block_mapping.device.type == "cpu", "block_mapping must be on CPU")
    _check(
        block_mapping.dim() == 2 and block_mapping.size(1) == 2,
        "block_mapping must be [num_pairs, 2] (src block, dst block)",
    )

    mapping = block_mapping.to(torch.int64)
    n = mapping.size(0)
    if n == 0:
        return

    block_bytes = int(block_size_in_bytes)
    gpu = src.device if src_gpu else dst.device
    with torch.cuda.device(gpu):
        if src_gpu and dst_gpu:
            src_addrs = mapping[:, 0] * block_bytes + src.data_ptr()
            dst_addrs = mapping[:, 1] * block_bytes + dst.data_ptr()
            sizes = torch.full((n,), block_bytes, dtype=torch.int64)
            _launch_batched_copy(src_addrs, dst_addrs, sizes)
            return

        _check(
            src.is_contiguous() and dst.is_contiguous(),
            "host/device block swap requires contiguous tensors",
        )
        src_bytes = src.view(-1).view(torch.uint8)
        dst_bytes = dst.view(-1).view(torch.uint8)
        for src_block, dst_block in mapping.tolist():
            dst_bytes[dst_block * block_bytes : (dst_block + 1) * block_bytes].copy_(
                src_bytes[src_block * block_bytes : (src_block + 1) * block_bytes],
                non_blocking=True,
            )


def swap_blocks_batch(
    src_ptrs: torch.Tensor,
    dst_ptrs: torch.Tensor,
    sizes: torch.Tensor,
    is_src_access_order_any: bool = False,
) -> None:
    """Batched raw-pointer memcpy (``cuMemcpyBatchAsync`` equivalent).

    ``src_ptrs``/``dst_ptrs``/``sizes`` are 1-D int64 CPU tensors holding raw
    addresses and byte counts.  The copy runs as a Triton kernel, so every
    address must be reachable from the GPU: device memory of the current GPU or
    pinned / registered host memory.  ``is_src_access_order_any`` only tunes the
    driver's batching hint in the CUDA version and is ignored here.
    """
    del is_src_access_order_any
    for name, t in (("src_ptrs", src_ptrs), ("dst_ptrs", dst_ptrs), ("sizes", sizes)):
        _check(t.device.type == "cpu", name, " must be on CPU")
        _check(t.dtype == torch.int64, name, " must be int64")
        _check(t.dim() == 1, name, " must be 1-D")
    _check(
        src_ptrs.numel() == dst_ptrs.numel() == sizes.numel(),
        "src_ptrs, dst_ptrs and sizes must have the same length",
    )
    _launch_batched_copy(src_ptrs, dst_ptrs, sizes)


def reshape_and_cache(
    key: torch.Tensor,  # [num_tokens, num_heads, head_size]
    value: torch.Tensor,  # [num_tokens, num_heads, head_size]
    key_cache: torch.Tensor,  # [num_blocks, num_heads, head_size/x, block_size, x]
    value_cache: torch.Tensor,  # [num_blocks, num_heads, head_size, block_size]
    slot_mapping: torch.Tensor,  # [num_tokens] int64
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    """Write token K/V into the classic paged-cache layout."""
    _check(key.is_cuda and value.is_cuda, "key and value must be CUDA tensors")
    _same_device(
        key,
        "key",
        value=value,
        key_cache=key_cache,
        value_cache=value_cache,
        slot_mapping=slot_mapping,
    )
    _check(
        key.dim() == 3 and value.shape == key.shape,
        "key/value must have shape [tokens, heads, head_size]",
    )
    _check(key.dtype == value.dtype, "key and value must have the same dtype")
    require_kv_source_dtype(key.dtype)
    _check(
        slot_mapping.dtype == torch.int64 and slot_mapping.dim() == 1,
        "slot_mapping must be 1-D int64",
    )
    _check(slot_mapping.numel() == key.size(0), "slot_mapping length must equal num_tokens")
    _check(key_cache.dim() == 5, "key_cache must be [blocks, heads, head_size/x, block_size, x]")
    _check(value_cache.dim() == 4, "value_cache must be [blocks, heads, head_size, block_size]")

    num_tokens, num_heads, head_size = key.shape
    block_size = key_cache.size(3)
    x = key_cache.size(4)
    _check(head_size % x == 0, "head_size must be a multiple of key_cache x")
    _check(key.stride(2) == 1 and value.stride(2) == 1, "key/value must be contiguous in head_size")
    _check(
        key_cache.stride(4) == 1 and value_cache.stride(3) == 1, "unexpected key/value cache layout"
    )

    if kv_cache_dtype.startswith("nvfp4"):
        raise NotImplementedError(NVFP4_MSG.format(what="nvfp4 reshape_and_cache"))

    kv_dtype, quantized, is_e5m2 = _dtype_flags(kv_cache_dtype)
    if quantized:
        _check(
            key_cache.element_size() == 1 and value_cache.element_size() == 1,
            "quantized caches must use 1-byte storage",
        )
        _validate_quant_scales(True, key, k_scale=k_scale, v_scale=v_scale)
        _check(k_scale.numel() == 1 and v_scale.numel() == 1, "classic cache scales must be scalar")
    else:
        _check(
            key_cache.dtype == key.dtype and value_cache.dtype == key.dtype,
            "AUTO cache dtype must match key/value dtype",
        )

    if num_tokens == 0 or num_heads == 0:
        return

    kc = cache_kernel_view(key_cache, kv_dtype)
    vc = cache_kernel_view(value_cache, kv_dtype)
    k_scale_ptr = k_scale if quantized else slot_mapping
    v_scale_ptr = v_scale if quantized else slot_mapping
    block_d = triton.next_power_of_2(head_size)

    with torch.cuda.device(_device_index(key)):
        cast(
            Any,
            reshape_and_cache_kernel[(num_tokens, num_heads)](
                key,
                value,
                kc,
                vc,
                slot_mapping,
                k_scale_ptr,
                v_scale_ptr,
                key.stride(0),
                key.stride(1),
                value.stride(0),
                value.stride(1),
                kc.stride(0),
                kc.stride(1),
                kc.stride(2),
                kc.stride(3),
                vc.stride(0),
                vc.stride(1),
                vc.stride(2),
                block_size,
                HEAD_SIZE=head_size,
                X=x,
                BLOCK_D=block_d,
                KV_QUANTIZED=quantized,
                IS_E5M2=is_e5m2,
                num_warps=warps_for_tile(block_d),
            ),
        )


def reshape_and_cache_flash(
    key: torch.Tensor,  # [num_tokens, num_heads, head_size]
    value: torch.Tensor,  # [num_tokens, num_heads, head_size]
    key_cache: torch.Tensor,  # [num_blocks, block_size, num_heads, head_size] (NHD or HND view)
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,  # [num_tokens] int64
    kv_cache_dtype: str,
    k_scale: torch.Tensor | float,  # float32 [1] or [num_heads]
    v_scale: torch.Tensor | float,
    *,
    inverse_scales: bool = False,
) -> None:
    """Write K/V to NHD/HND views; optionally use host scalar inverse scales.

    ``inverse_scales=True`` preserves storage's fp32 ``x * (1/scale)``
    rounding. Host scalars require no extra device allocation or synchronization.
    Slot bounds and uniqueness must have been checked by the caller.
    """
    _check(key.is_cuda and value.is_cuda, "key and value must be CUDA tensors")
    _same_device(
        key,
        "key",
        value=value,
        key_cache=key_cache,
        value_cache=value_cache,
        slot_mapping=slot_mapping,
    )
    _check(
        key.dim() == 3 and value.shape == key.shape,
        "key/value must have shape [tokens, heads, head_size]",
    )
    _check(key.dtype == value.dtype, "key and value must have the same dtype")
    require_kv_source_dtype(key.dtype)
    _check(
        slot_mapping.dtype == torch.int64 and slot_mapping.dim() == 1,
        "slot_mapping must be 1-D int64",
    )
    _check(slot_mapping.numel() == key.size(0), "slot_mapping length must equal num_tokens")
    _check(key_cache.dim() == 4 and value_cache.dim() == 4, "flash caches must be rank-4")
    _check(
        key_cache.shape == value_cache.shape, "key_cache and value_cache must have the same shape"
    )

    num_tokens, num_heads, head_size = key.shape
    block_size = key_cache.size(1)
    _check(key.stride(2) == 1 and value.stride(2) == 1, "key/value must be contiguous in head_size")
    _check(
        key_cache.stride(3) == 1 and value_cache.stride(3) == 1,
        "cache head_size must be contiguous",
    )
    _check(
        key_cache.stride() == value_cache.stride(),
        "key_cache and value_cache must have identical strides",
    )
    scalar_scales = isinstance(k_scale, (int, float)) and isinstance(v_scale, (int, float))
    if scalar_scales:
        _check(
            all(math.isfinite(float(s)) and float(s) > 0 for s in (k_scale, v_scale)),
            "scalar scales must be finite and positive",
        )
    else:
        _check(
            isinstance(k_scale, torch.Tensor) and isinstance(v_scale, torch.Tensor),
            "scales must both be tensors or both be scalars",
        )
        k_scale = cast(torch.Tensor, k_scale)
        v_scale = cast(torch.Tensor, v_scale)
        _check(k_scale.shape == v_scale.shape, "k_scale and v_scale must have the same shape")
        _check(k_scale.numel() in (1, num_heads), "scales must contain 1 or num_heads values")

    kv_dtype, quantized, is_e5m2 = _dtype_flags(kv_cache_dtype)
    if quantized:
        _check(
            key_cache.element_size() == 1 and value_cache.element_size() == 1,
            "quantized caches must use 1-byte storage",
        )
        if not scalar_scales:
            _validate_quant_scales(
                True,
                key,
                k_scale=cast(torch.Tensor, k_scale),
                v_scale=cast(torch.Tensor, v_scale),
            )
    else:
        _check(
            key_cache.dtype == key.dtype and value_cache.dtype == key.dtype,
            "AUTO cache dtype must match key/value dtype",
        )

    if num_tokens == 0 or num_heads == 0:
        return

    kc = cache_kernel_view(key_cache, kv_dtype)
    vc = cache_kernel_view(value_cache, kv_dtype)
    k_scale_ptr = k_scale if quantized else slot_mapping
    v_scale_ptr = v_scale if quantized else slot_mapping
    scale_stride = 0 if scalar_scales else int(cast(torch.Tensor, k_scale).numel() > 1)
    block_d = triton.next_power_of_2(head_size)
    heads_per_prog = min(triton.next_power_of_2(num_heads), max(1, 1024 // block_d))

    with torch.cuda.device(_device_index(key)):
        cast(
            Any,
            reshape_and_cache_flash_kernel[(num_tokens, triton.cdiv(num_heads, heads_per_prog))](
                key,
                value,
                kc,
                vc,
                slot_mapping,
                k_scale_ptr,
                v_scale_ptr,
                key.stride(0),
                key.stride(1),
                value.stride(0),
                value.stride(1),
                kc.stride(0),
                kc.stride(1),
                kc.stride(2),
                block_size,
                num_heads,
                scale_stride,
                HEAD_SIZE=head_size,
                BLOCK_D=block_d,
                HEADS_PER_PROG=heads_per_prog,
                KV_QUANTIZED=quantized,
                IS_E5M2=is_e5m2,
                SCALAR_SCALES=scalar_scales,
                INVERSE_SCALES=inverse_scales,
                num_warps=warps_for_tile(block_d * heads_per_prog),
            ),
        )


def _is_compute_dtype(dtype: torch.dtype) -> bool:
    try:
        require_kv_source_dtype(dtype)
    except ValueError:
        return False
    return True


def convert_fp8(
    dst_cache: torch.Tensor,
    src_cache: torch.Tensor,
    scale: float,
    kv_cache_dtype: KVCacheDtype | str,
) -> None:
    """Convert a complete cache between model dtype and Ayaka FP8 storage."""
    _check(src_cache.is_cuda and dst_cache.is_cuda, "src_cache and dst_cache must be CUDA tensors")
    _check(src_cache.device == dst_cache.device, "src_cache and dst_cache must be on the same GPU")
    _check(
        src_cache.numel() == dst_cache.numel(),
        "src_cache and dst_cache must have the same number of elements",
    )
    _check(scale > 0.0, "scale must be positive")

    kv_dtype, quantized, is_e5m2 = _dtype_flags(kv_cache_dtype)
    src_is_compute = _is_compute_dtype(src_cache.dtype)
    dst_is_compute = _is_compute_dtype(dst_cache.dtype)

    if quantized:
        _check(
            src_is_compute != dst_is_compute,
            "FP8 conversion requires exactly one model-dtype tensor and one 1-byte cache tensor",
        )
        to_cache = src_is_compute
        if to_cache:
            src = src_cache
            dst = as_byte_view(dst_cache)
        else:
            src = as_byte_view(src_cache)
            dst = dst_cache
    else:
        _check(
            src_is_compute and dst_is_compute,
            "AUTO conversion requires model/compute dtypes on both sides",
        )
        to_cache = True
        src = src_cache
        dst = dst_cache

    # Match the original cache op: process the physical block-stride footprint,
    # not merely logical numel when the block view is padded/strided.
    numel = src_cache.size(0) * src_cache.stride(0)
    if numel == 0:
        return

    block = 1024
    with torch.cuda.device(_device_index(src_cache)):
        cast(
            Any,
            convert_fp8_kernel[(triton.cdiv(numel, block),)](
                src,
                dst,
                float(scale),
                numel,
                BLOCK=block,
                KV_QUANTIZED=quantized,
                IS_E5M2=is_e5m2,
                TO_CACHE=to_cache,
                num_warps=4,
            ),
        )


def _gather_launch(
    src: torch.Tensor,
    dst: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    seq_starts: torch.Tensor | None,
    scale: torch.Tensor,
    num_reqs: int,
    num_tokens: int,
    entry_size: int,
    *,
    quantized: bool,
    is_e5m2: bool,
) -> None:
    block_e = min(256, triton.next_power_of_2(entry_size))
    grid = (triton.cdiv(num_tokens, GATHER_BLOCK_T),)

    with torch.cuda.device(_device_index(dst)):
        cast(
            Any,
            gather_cache_kernel[grid](
                src,
                dst,
                block_table,
                cu_seq_lens,
                seq_starts if seq_starts is not None else cu_seq_lens,
                scale if quantized else cu_seq_lens,
                num_reqs,
                num_tokens,
                src.size(1),
                block_table.size(1),
                (num_reqs + 1).bit_length(),
                block_table.stride(0),
                src.stride(0),
                src.stride(1),
                dst.stride(0),
                ENTRY_SIZE=entry_size,
                BLOCK_T=GATHER_BLOCK_T,
                BLOCK_E=block_e,
                KV_QUANTIZED=quantized,
                IS_E5M2=is_e5m2,
                HAS_SEQ_STARTS=seq_starts is not None,
                DIRECT_SLOTS=False,
                SCALAR_SCALE=False,
                num_warps=4,
            ),
        )


def gather_and_maybe_dequant_cache(
    src_cache: torch.Tensor,
    dst: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    token_to_seq: torch.Tensor,
    num_tokens: int,
    kv_cache_dtype: KVCacheDtype | str,
    scale: torch.Tensor,
    seq_starts: torch.Tensor | None = None,
) -> None:
    """Gather paged cache rows into ``dst``, dequantizing FP8 when requested."""
    del token_to_seq  # preserved only for API parity with V1

    _check(src_cache.is_cuda and dst.is_cuda, "src_cache and dst must be CUDA tensors")
    _check(block_table.dtype == torch.int32, "block_table must be int32")
    _check(cu_seq_lens.dtype == torch.int32, "cu_seq_lens must be int32")
    if seq_starts is not None:
        _check(seq_starts.dtype == torch.int32, "seq_starts must be int32")

    _same_device(
        src_cache,
        "src_cache",
        dst=dst,
        block_table=block_table,
        cu_seq_lens=cu_seq_lens,
        seq_starts=seq_starts,
    )
    _check(src_cache.stride(-1) == 1 and dst.stride(-1) == 1, "cache entries must be contiguous")
    _check(num_tokens >= 0 and num_tokens <= dst.size(0), "num_tokens is outside dst bounds")
    require_kv_source_dtype(dst.dtype)

    kv_dtype, quantized, is_e5m2 = _dtype_flags(kv_cache_dtype)
    if quantized:
        _check(src_cache.element_size() == 1, "quantized src_cache must use 1-byte storage")
        _check(
            scale.dtype == torch.float32 and scale.numel() == 1, "scale must be a float32 scalar"
        )
        _same_device(src_cache, "src_cache", scale=scale)
    else:
        _check(src_cache.dtype == dst.dtype, "AUTO cache dtype must match dst dtype")

    if num_tokens == 0:
        return
    num_reqs = cu_seq_lens.size(0) - 1
    if num_reqs <= 0:
        return

    entry_size = dst.size(-1)
    src = cache_kernel_view(src_cache, kv_dtype)
    _gather_launch(
        src,
        dst,
        block_table,
        cu_seq_lens,
        seq_starts,
        scale,
        num_reqs,
        num_tokens,
        entry_size,
        quantized=quantized,
        is_e5m2=is_e5m2,
    )


def cp_gather_cache(
    src_cache: torch.Tensor,
    dst: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    batch_size: int,
    seq_starts: torch.Tensor | None = None,
) -> None:
    """Bit-exact gather of paged cache entries with no dtype conversion."""
    _check(src_cache.is_cuda and dst.is_cuda, "src_cache and dst must be CUDA tensors")
    _check(block_table.dtype == torch.int32, "block_table must be int32")
    _check(cu_seq_lens.dtype == torch.int32, "cu_seq_lens must be int32")
    if seq_starts is not None:
        _check(seq_starts.dtype == torch.int32, "seq_starts must be int32")

    _same_device(
        src_cache,
        "src_cache",
        dst=dst,
        block_table=block_table,
        cu_seq_lens=cu_seq_lens,
        seq_starts=seq_starts,
    )
    _check(src_cache.dtype == dst.dtype, "src_cache and dst must have the same dtype")
    _check(batch_size == cu_seq_lens.size(0) - 1, "batch_size must match cu_seq_lens")
    _check(src_cache.stride(-1) == 1 and dst.stride(-1) == 1, "cache entries must be contiguous")

    total_tokens = dst.size(0)
    if total_tokens == 0 or batch_size == 0:
        return

    entry_size = 1
    for extent in src_cache.shape[2:]:
        entry_size *= extent

    src_raw = as_int_view(src_cache)
    dst_raw = as_int_view(dst)
    _gather_launch(
        src_raw,
        dst_raw,
        block_table,
        cu_seq_lens,
        seq_starts,
        cu_seq_lens,  # unused placeholder
        batch_size,
        total_tokens,
        entry_size,
        quantized=False,
        is_e5m2=False,
    )


def indexer_k_quant_and_cache(
    k: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str,
) -> None:
    """Quantize sparse-indexer K rows to E4M3 plus per-group float32 scales."""
    _check(k.is_cuda and kv_cache.is_cuda, "k and kv_cache must be CUDA tensors")
    _same_device(k, "k", kv_cache=kv_cache, slot_mapping=slot_mapping)
    require_kv_source_dtype(k.dtype)
    _check(k.dim() == 2, "k must be [num_tokens, head_dim]")
    _check(
        slot_mapping.dtype == torch.int64 and slot_mapping.dim() == 1,
        "slot_mapping must be 1-D int64",
    )
    _check(slot_mapping.numel() == k.size(0), "slot_mapping length must equal num_tokens")
    _check(quant_block_size > 0, "quant_block_size must be positive")
    _check(scale_fmt in ("float", "ue8m0"), "scale_fmt must be 'float' or 'ue8m0'")

    num_tokens, head_dim = k.shape
    cache_block_size = kv_cache.size(1)
    _check(head_dim % quant_block_size == 0, "head_dim must be divisible by quant_block_size")
    _check(k.stride(1) == 1, "k must be contiguous in head_dim")
    _check(kv_cache.element_size() == 1, "kv_cache must use a 1-byte dtype")

    num_qblocks = head_dim // quant_block_size
    _check(
        cache_block_size * (head_dim + 4 * num_qblocks) <= kv_cache.stride(0),
        "kv_cache blocks are too small for fp8 data plus scales",
    )
    if num_tokens == 0:
        return

    cache = as_byte_view(kv_cache)
    block = triton.next_power_of_2(quant_block_size)
    with torch.cuda.device(_device_index(k)):
        cast(
            Any,
            indexer_k_quant_and_cache_kernel[(num_tokens, num_qblocks)](
                k,
                cache,
                slot_mapping,
                k.stride(0),
                cache.stride(0),
                cache_block_size,
                head_dim,
                QUANT_BLOCK=quant_block_size,
                BLOCK=block,
                NUM_QBLOCKS=num_qblocks,
                USE_UE8M0=scale_fmt == "ue8m0",
                SCALE_DIVISOR=float(FP8_E4M3_MAX),
                num_warps=warps_for_tile(block),
            ),
        )


def cp_gather_indexer_k_quant_cache(
    kv_cache: torch.Tensor,
    dst_k: torch.Tensor,
    dst_scale: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
) -> None:
    """Gather sparse-indexer E4M3 bytes and their scale payloads."""
    _check(
        kv_cache.is_cuda and dst_k.is_cuda and dst_scale.is_cuda,
        "cache and destinations must be CUDA tensors",
    )
    _same_device(
        kv_cache,
        "kv_cache",
        dst_k=dst_k,
        dst_scale=dst_scale,
        block_table=block_table,
        cu_seq_lens=cu_seq_lens,
    )
    _check(block_table.dtype == torch.int32, "block_table must be int32")
    _check(cu_seq_lens.dtype == torch.int32, "cu_seq_lens must be int32")
    _check(
        kv_cache.element_size() == 1 and dst_k.element_size() == 1,
        "kv_cache and dst_k must use 1-byte dtypes",
    )
    _check(
        dst_k.dim() == 2 and dst_k.stride(1) == 1,
        "dst_k must be [num_tokens, head_dim] and contiguous in head_dim",
    )

    batch_size = block_table.size(0)
    num_tokens, head_dim = dst_k.shape
    cache_block_size = kv_cache.size(1)
    _check(
        dst_scale.dim() == 2 and dst_scale.size(0) == num_tokens,
        "dst_scale must be [num_tokens, scale_bytes]",
    )
    scale_bytes = dst_scale.size(1) * dst_scale.element_size()
    _check(scale_bytes % 4 == 0, "dst_scale rows must hold whole float32 scales")
    num_qblocks = scale_bytes // 4
    _check(
        num_qblocks > 0 and head_dim % num_qblocks == 0,
        "head_dim must be divisible by the inferred quant block size",
    )
    _check(
        cache_block_size * (head_dim + 4 * num_qblocks) <= kv_cache.stride(0),
        "kv_cache blocks are too small for fp8 data plus scales",
    )
    _check(dst_scale.stride(1) == 1, "dst_scale must be contiguous in the last dimension")
    _check(
        (dst_scale.stride(0) * dst_scale.element_size()) % 4 == 0,
        "dst_scale rows must be 4-byte aligned",
    )

    if num_tokens == 0 or batch_size == 0:
        return

    scale_i32 = dst_scale.view(torch.int32)
    cache = as_byte_view(kv_cache)
    dst_bytes = as_byte_view(dst_k)
    block_t = GATHER_BLOCK_T

    with torch.cuda.device(_device_index(dst_k)):
        cast(
            Any,
            cp_gather_indexer_k_quant_cache_kernel[(triton.cdiv(num_tokens, block_t),)](
                cache,
                dst_bytes,
                scale_i32,
                block_table,
                cu_seq_lens,
                batch_size,
                num_tokens,
                (batch_size + 1).bit_length(),
                block_table.stride(0),
                block_table.size(1),
                cache.stride(0),
                cache_block_size,
                head_dim,
                dst_bytes.stride(0),
                scale_i32.stride(0),
                HEAD_DIM=head_dim,
                NUM_QBLOCKS=num_qblocks,
                BLOCK_T=block_t,
                BLOCK_D=min(256, triton.next_power_of_2(head_dim)),
                BLOCK_S=triton.next_power_of_2(num_qblocks),
                num_warps=4,
            ),
        )


@triton.jit
def scatter_cache_slots_kernel(
    src_ptr,
    dst_ptr,
    slots_ptr,
    inverse_scale,
    src_stride,
    dst_stride,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    QUANT: tl.constexpr,
    IS_E5M2: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    slot = tl.load(slots_ptr + row)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(src_ptr + row * src_stride + col, mask=col < WIDTH, other=0)
    if QUANT:
        x = x.to(tl.float32) * inverse_scale
    y = quantize_if_fp8(x, 1.0, QUANT, IS_E5M2)
    tl.store(dst_ptr + slot * dst_stride + col, y, mask=col < WIDTH)


def scatter_cache_slots(
    src: torch.Tensor,
    dst: torch.Tensor,
    slots: torch.Tensor,
    inverse_scale: float,
    kv_cache_dtype: str,
) -> None:
    """Scatter contiguous row payloads using caller-validated unique slots.

    Used for planes with different widths (MLA), and dtype-converting writes.
    Inputs are flat ``[rows, width]`` views on one GPU.
    """
    kv_dtype, quantized, is_e5m2 = _dtype_flags(kv_cache_dtype)
    if slots.numel() == 0 or dst.size(1) == 0:
        return
    dst_view = cache_kernel_view(dst, kv_dtype)
    with torch.cuda.device(src.device):
        cast(Any, scatter_cache_slots_kernel)[(slots.numel(), triton.cdiv(dst.size(1), 256))](
            src,
            dst_view,
            slots,
            inverse_scale,
            src.stride(0),
            dst_view.stride(0),
            WIDTH=dst.size(1),
            BLOCK=256,
            QUANT=quantized,
            IS_E5M2=is_e5m2,
            num_warps=4,
        )


def gather_cache_slots(
    src: torch.Tensor,
    dst: torch.Tensor,
    slots: torch.Tensor,
    scale: float,
    kv_cache_dtype: str,
) -> None:
    """Gather prevalidated slots without constructing synthetic block tables.

    ``src`` is a contiguous ``[pages, page_size, width]`` view; ``dst`` is
    ``[tokens, width]``. AUTO copies same-dtype payloads bit-for-bit, including
    FP8 codes; explicit FP8 dtype dequantizes in fp32 before the output cast.
    Repeated slots are allowed. Bounds remain the storage/lease owner's job.
    """
    kv_dtype, quantized, is_e5m2 = _dtype_flags(kv_cache_dtype)
    if slots.numel() == 0 or dst.size(1) == 0:
        return
    source = cache_kernel_view(src, kv_dtype)
    output = dst
    if not quantized and src.dtype == dst.dtype:
        source, output = as_int_view(src), as_int_view(dst)
    with torch.cuda.device(src.device):
        cast(Any, gather_cache_kernel)[(triton.cdiv(slots.numel(), GATHER_BLOCK_T),)](
            source,
            output,
            slots,
            slots,
            slots,
            scale,
            0,
            slots.numel(),
            src.size(1),
            0,
            0,
            0,
            source.stride(0),
            source.stride(1),
            output.stride(0),
            ENTRY_SIZE=dst.size(1),
            BLOCK_T=GATHER_BLOCK_T,
            BLOCK_E=min(256, triton.next_power_of_2(dst.size(1))),
            KV_QUANTIZED=quantized,
            IS_E5M2=is_e5m2,
            HAS_SEQ_STARTS=False,
            DIRECT_SLOTS=True,
            SCALAR_SCALE=True,
            num_warps=4,
        )
