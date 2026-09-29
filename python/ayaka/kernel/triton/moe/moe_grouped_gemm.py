"""FP8 blockwise-scaled grouped GEMM for MoE, plus the metadata kernels that feed it
(Triton port of ``cutlass_moe_helper.cu``, ``fp8_blockwise_moe_kernel.cu`` and the ``compute_*`` kernels of
``prepare_moe_input.cu``).

Data layout (identical to the CUDA extension, which derives every address from ``data_ptr`` + arithmetic)
-----------------------------------------------------------------------------------------------------------
``a``          ``[M_total, K]``       fp8 e4m3fn, row-major, rows grouped by expert (``expert_offsets``)
``b``          memory ``[E][N][K]``   fp8 e4m3fn (passed either as ``[E, N, K]`` or as the transposed view ``[E, K, N]``)
``scales_a``   ``[M_total, K/128]``   fp32, one scale per row and 128-wide K block   (granularity 1 x 128)
``scales_b``   memory ``[E][N/128][K/128]`` fp32, one scale per 128 x 128 block of ``b``
``output``     ``[M_total, N]``       bf16 / fp16
``C[m, n] = sum_kb  scales_a[m, kb] * scales_b[n/128, kb] * dot_fp8(a[m, kb*128:+128], b[n, kb*128:+128])``

Numerics == CUTLASS blockwise "promotion" scheme: every 128-wide K block is multiplied on the tensor cores with a
fresh accumulator (fp32), the partial result is then scaled and added into an fp32 register accumulator, and the
final value is rounded to the output dtype once.  Results are not bit-identical to CUTLASS (summation order inside
the MMA and the association of the two scales differ) but follow the same error model.

Kernel structure
----------------
``get_group_gemm_starts_kernel`` builds per-expert device pointer tables (int64 addresses) exactly like the CUDA
helper, in the normal and in the "transposed" (swap-AB) flavour.  ``fp8_blockwise_grouped_gemm_kernel`` consumes them
the way the CUTLASS grouped-GEMM consumes its pointer arrays, in the CUTLASS operand convention
``D[m', n'] = X[m', :] . Y[n', :]`` (both operands K-contiguous).  The tile scheduler is *flexible*: the launch grid is
an upper bound computed on the host (no device sync), each program finds its expert with a vectorised prefix sum of the
per-expert tile counts and exits if it lies beyond the real number of tiles.

Swap-AB (``swap_ab``): for small ``M`` the weights become the ``M'`` operand and the (few) tokens the ``N'`` operand,
so the MMA tile is filled by the big dimension - the CUDA dispatcher does this for ``M_total <= 2048`` on SM90.

``layout_sfa`` / ``layout_sfb`` are ``[E, >=5]`` int32 descriptors ``[rows, k_blocks, 1, row_stride, batch_stride]``
of the scale tensors as seen by the GEMM (K-major).  CUTLASS' CuTe layout objects have no Triton meaning, so this is a
portable descriptor, not a binary CuTe layout.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ayaka.kernel.triton.moe._common import (
    autotune,
    cdiv,
    check,
    check_tensor,
    device_guard,
    next_pow2,
    same_device,
)
from ayaka.kernel.triton.moe.moe_align import launch_compute_arg_sorts, launch_expert_histogram

SCALE_BLOCK = 128  # granularity of the FP8 block scales (M/N/K)
_OUT_DTYPES = (torch.bfloat16, torch.float16)


# ======================================================================================
# cutlass_moe_helper.cu :: get_group_gemm_starts
# ======================================================================================
@triton.jit
def get_group_gemm_starts_kernel(
    expert_offsets_ptr,
    a_ptrs_ptr,
    b_ptrs_ptr,
    out_ptrs_ptr,
    a_scales_ptrs_ptr,
    b_scales_ptrs_ptr,
    a_base,
    b_base,
    out_base,
    a_scales_base,
    b_scales_base,
    layout_sfa_ptr,
    layout_sfb_ptr,
    layout_stride,
    problem_sizes_ptr,
    problem_sizes_transpose_ptr,
    num_experts,
    OUT_ELEM_BYTES: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """One lane per expert (CUDA: ``<<<1, num_experts>>>``).  ``*_base`` are device addresses (int64)."""
    e = tl.arange(0, BLOCK_E)
    ok = e < num_experts
    m = tl.load(problem_sizes_ptr + e * 3, mask=ok, other=0)
    n = tl.load(problem_sizes_ptr + e * 3 + 1, mask=ok, other=0)
    k = tl.load(problem_sizes_ptr + e * 3 + 2, mask=ok, other=0)
    e64 = e.to(tl.int64)
    n64 = n.to(tl.int64)
    k64 = k.to(tl.int64)
    off = tl.load(expert_offsets_ptr + e, mask=ok, other=0).to(tl.int64)

    if TRANSPOSE:
        tl.store(problem_sizes_transpose_ptr + e * 3, n, mask=ok)
        tl.store(problem_sizes_transpose_ptr + e * 3 + 1, m, mask=ok)
        tl.store(problem_sizes_transpose_ptr + e * 3 + 2, k, mask=ok)
        a_stride = e64 * k64 * n64
        b_stride = off * k64
        a_scale_stride = e64 * k64 * n64 // 128 // 128
        b_scale_stride = off * k64 // 128
    else:
        a_stride = off * k64
        b_stride = e64 * k64 * n64
        a_scale_stride = off * k64 // 128
        b_scale_stride = e64 * k64 * n64 // 128 // 128

    # pointer arithmetic of the CUDA code, in bytes: fp8 = 1 B, output = OUT_ELEM_BYTES, fp32 scale = 4 B
    tl.store(a_ptrs_ptr + e, a_base + a_stride, mask=ok)
    tl.store(b_ptrs_ptr + e, b_base + b_stride, mask=ok)
    tl.store(out_ptrs_ptr + e, out_base + off * n64 * OUT_ELEM_BYTES, mask=ok)
    tl.store(a_scales_ptrs_ptr + e, a_scales_base + a_scale_stride * 4, mask=ok)
    tl.store(b_scales_ptrs_ptr + e, b_scales_base + b_scale_stride * 4, mask=ok)

    # scale-layout descriptors [rows, k_blocks, L, row_stride, batch_stride] (K-major)
    kb = (k + 127) // 128
    nb = (n + 127) // 128
    if TRANSPOSE:
        sfa_rows = nb  # SFA covers the weights (128-row blocks)   [ScaleConfig<128, 1, 128>]
        sfb_rows = m  # SFB covers the tokens (one scale per row)
    else:
        sfa_rows = m  # SFA covers the tokens (one scale per row)   [ScaleConfig<1, 128, 128>]
        sfb_rows = nb  # SFB covers the weights (128-row blocks)
    sfa = layout_sfa_ptr + e * layout_stride
    sfb = layout_sfb_ptr + e * layout_stride
    tl.store(sfa, sfa_rows, mask=ok)
    tl.store(sfa + 1, kb, mask=ok)
    tl.store(sfa + 2, tl.zeros([BLOCK_E], dtype=tl.int32) + 1, mask=ok)
    tl.store(sfa + 3, kb, mask=ok)
    tl.store(sfa + 4, sfa_rows * kb, mask=ok)
    tl.store(sfb, sfb_rows, mask=ok)
    tl.store(sfb + 1, kb, mask=ok)
    tl.store(sfb + 2, tl.zeros([BLOCK_E], dtype=tl.int32) + 1, mask=ok)
    tl.store(sfb + 3, kb, mask=ok)
    tl.store(sfb + 4, sfb_rows * kb, mask=ok)


def get_group_gemm_starts(
    expert_offsets: torch.Tensor,
    a_ptrs: torch.Tensor,
    b_ptrs: torch.Tensor,
    out_ptrs: torch.Tensor,
    a_scales_ptrs: torch.Tensor,
    b_scales_ptrs: torch.Tensor,
    a_tensors: torch.Tensor,
    b_tensors: torch.Tensor,
    out_tensors: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    layout_sfa: torch.Tensor,
    layout_sfb: torch.Tensor,
    problem_sizes: torch.Tensor,
    problem_sizes_transpose: torch.Tensor,
    transpose: bool = False,
) -> None:
    """Fill the per-expert pointer tables (int64 device addresses), scale-layout descriptors and, when
    ``transpose``, the swapped problem sizes.  Mirrors ``run_get_group_gemm_starts`` argument for argument:
    in ``transpose`` mode the caller passes ``a_tensors`` = weights, ``b_tensors`` = activations,
    ``a_scales`` = weight scales and ``b_scales`` = activation scales (only their base addresses are used).
    """
    check_tensor(a_tensors, "a_tensors", dtypes=(torch.float8_e4m3fn,))
    check_tensor(b_tensors, "b_tensors", dtypes=(torch.float8_e4m3fn,))
    check_tensor(a_scales, "a_scales", dtypes=(torch.float32,))
    check_tensor(b_scales, "b_scales", dtypes=(torch.float32,))
    check(out_tensors.ndim == 2, "out_tensors must be 2D")
    check(a_tensors.ndim in (2, 3), "a_tensors must be 2D or 3D")
    check(b_tensors.ndim in (2, 3), "b_tensors must be 2D or 3D")
    check(
        out_tensors.size(1) % 128 == 0 or out_tensors.size(0) % 128 == 0,
        "output extent must be a multiple of 128",
    )
    check(
        a_tensors.size(1) % 128 == 0 or a_tensors.size(0) % 128 == 0,
        "a extent must be a multiple of 128",
    )
    check(out_tensors.dtype in _OUT_DTYPES, "Invalid output type (must be float16 or bfloat16)")
    for name, t in (
        ("a_ptrs", a_ptrs),
        ("b_ptrs", b_ptrs),
        ("out_ptrs", out_ptrs),
        ("a_scales_ptrs", a_scales_ptrs),
        ("b_scales_ptrs", b_scales_ptrs),
    ):
        check_tensor(t, name, dtypes=(torch.int64,), ndim=1, contiguous=True)
    check_tensor(expert_offsets, "expert_offsets", dtypes=(torch.int32,), ndim=1, contiguous=True)
    check_tensor(problem_sizes, "problem_sizes", dtypes=(torch.int32,), ndim=2, contiguous=True)
    check_tensor(layout_sfa, "layout_sfa", dtypes=(torch.int32,), ndim=2, contiguous=True)
    check_tensor(layout_sfb, "layout_sfb", dtypes=(torch.int32,), ndim=2, contiguous=True)
    same_device(
        expert_offsets,
        a_ptrs,
        b_ptrs,
        out_ptrs,
        a_scales_ptrs,
        b_scales_ptrs,
        a_tensors,
        b_tensors,
        out_tensors,
        a_scales,
        b_scales,
        layout_sfa,
        layout_sfb,
        problem_sizes,
        problem_sizes_transpose,
    )
    num_experts = expert_offsets.size(0)
    check(
        problem_sizes.size(0) == num_experts and problem_sizes.size(1) == 3,
        "problem_sizes must be [num_experts, 3]",
    )
    for name, t in (
        ("a_ptrs", a_ptrs),
        ("b_ptrs", b_ptrs),
        ("out_ptrs", out_ptrs),
        ("a_scales_ptrs", a_scales_ptrs),
        ("b_scales_ptrs", b_scales_ptrs),
    ):
        check(t.numel() >= num_experts, f"{name} must hold at least num_experts entries")
    check(
        layout_sfa.size(0) >= num_experts and layout_sfa.size(1) >= 5,
        "layout_sfa must be [num_experts, >=5] int32",
    )
    check(
        layout_sfb.size(0) >= num_experts and layout_sfb.size(1) >= 5,
        "layout_sfb must be [num_experts, >=5] int32",
    )
    check(
        layout_sfa.stride(0) == layout_sfb.stride(0),
        "layout_sfa and layout_sfb must have the same row stride",
    )
    if transpose:
        check_tensor(
            problem_sizes_transpose,
            "problem_sizes_transpose",
            dtypes=(torch.int32,),
            ndim=2,
            contiguous=True,
        )
        check(
            problem_sizes_transpose.size(1) == 3,
            "problem_sizes_transpose must have shape [num_experts, 3]",
        )
        check(
            problem_sizes_transpose.numel() >= 3 * num_experts,
            "problem_sizes_transpose must hold 3 * num_experts ints",
        )
        check(
            problem_sizes_transpose.dtype in (torch.int32,), "problem_sizes_transpose must be int32"
        )
    if num_experts == 0:
        return
    with device_guard(a_tensors):
        get_group_gemm_starts_kernel[(1,)](
            expert_offsets,
            a_ptrs,
            b_ptrs,
            out_ptrs,
            a_scales_ptrs,
            b_scales_ptrs,
            a_tensors.data_ptr(),
            b_tensors.data_ptr(),
            out_tensors.data_ptr(),
            a_scales.data_ptr(),
            b_scales.data_ptr(),
            layout_sfa,
            layout_sfb,
            layout_sfa.stride(0),
            problem_sizes,
            problem_sizes_transpose if transpose else problem_sizes,  # dummy when unused
            num_experts,
            OUT_ELEM_BYTES=out_tensors.element_size(),
            TRANSPOSE=bool(transpose),
            BLOCK_E=max(2, next_pow2(num_experts)),
            num_warps=4,
        )


# ======================================================================================
# prepare_moe_input.cu :: compute_problem_sizes / compute_expert_offsets / *_blockscale_offsets
# ======================================================================================
@triton.jit
def compute_problem_sizes_kernel(
    counts_ptr,
    problem_sizes1_ptr,
    problem_sizes2_ptr,
    num_experts,
    n,
    k,
    BLOCK_E: tl.constexpr,
):
    """problem_sizes1[e] = (count_e, 2n, k), problem_sizes2[e] = (count_e, k, n)."""
    e = tl.arange(0, BLOCK_E)
    ok = e < num_experts
    cnt = tl.load(counts_ptr + e, mask=ok, other=0)
    z = tl.zeros([BLOCK_E], dtype=tl.int32)
    tl.store(problem_sizes1_ptr + e * 3, cnt, mask=ok)
    tl.store(problem_sizes1_ptr + e * 3 + 1, z + 2 * n, mask=ok)
    tl.store(problem_sizes1_ptr + e * 3 + 2, z + k, mask=ok)
    tl.store(problem_sizes2_ptr + e * 3, cnt, mask=ok)
    tl.store(problem_sizes2_ptr + e * 3 + 1, z + k, mask=ok)
    tl.store(problem_sizes2_ptr + e * 3 + 2, z + n, mask=ok)


@triton.jit
def compute_expert_offsets_kernel(
    problem_sizes1_ptr,
    expert_offsets_ptr,
    atomic_buffer_ptr,
    num_experts,
    BLOCK_E: tl.constexpr,
):
    """expert_offsets = [0, cumsum(count)];  atomic_buffer[e] = exclusive prefix (cursor base for arg sorts)."""
    e = tl.arange(0, BLOCK_E)
    ok = e < num_experts
    cnt = tl.load(problem_sizes1_ptr + e * 3, mask=ok, other=0)
    incl = tl.cumsum(cnt, axis=0)
    tl.store(atomic_buffer_ptr + e, incl - cnt, mask=ok)
    tl.store(expert_offsets_ptr + e + 1, incl, mask=ok)
    tl.store(expert_offsets_ptr, 0)


@triton.jit
def compute_expert_blockscale_offsets_kernel(
    problem_sizes1_ptr,
    expert_offsets_ptr,
    blockscale_offsets_ptr,
    atomic_buffer_ptr,
    num_experts,
    BLOCK_E: tl.constexpr,
):
    """Same as above, plus ``blockscale_offsets = [0, cumsum(round_up(count, 128))]`` (128-aligned scale rows)."""
    e = tl.arange(0, BLOCK_E)
    ok = e < num_experts
    cnt = tl.load(problem_sizes1_ptr + e * 3, mask=ok, other=0)
    rounded = ((cnt + 127) // 128) * 128
    incl = tl.cumsum(cnt, axis=0)
    incl_r = tl.cumsum(rounded, axis=0)
    tl.store(atomic_buffer_ptr + e, incl - cnt, mask=ok)
    tl.store(expert_offsets_ptr + e + 1, incl, mask=ok)
    tl.store(blockscale_offsets_ptr + e + 1, incl_r, mask=ok)
    tl.store(expert_offsets_ptr, 0)
    tl.store(blockscale_offsets_ptr, 0)


def prepare_moe_input(
    topk_ids: torch.Tensor,
    expert_offsets: torch.Tensor,
    blockscale_offsets: torch.Tensor | None,
    problem_sizes1: torch.Tensor,
    problem_sizes2: torch.Tensor,
    input_permutation: torch.Tensor,
    output_permutation: torch.Tensor,
    num_experts: int,
    n: int,
    k: int,
) -> None:
    """Signature-compatible replacement for the CUDA ``prepare_moe_input`` op.

    ``topk_ids: [m, topk]`` int32 -> ``problem_sizes1/2: [E, 3]``, ``expert_offsets: [E + 1]``,
    optional ``blockscale_offsets: [E + 1]``, ``input_permutation`` (expert-major row -> source token) and
    ``output_permutation`` (token-major slot -> expert-major row), both ``[m * topk]`` int32.
    Rows of one expert appear in atomic-arrival order (as in CUDA); the two permutations stay mutually consistent.
    """
    check(topk_ids.dtype == torch.int32, "topk_ids must be int32")
    check_tensor(topk_ids, "topk_ids", dtypes=(torch.int32,), ndim=2, contiguous=True)
    check_tensor(expert_offsets, "expert_offsets", dtypes=(torch.int32,), ndim=1, contiguous=True)
    check_tensor(problem_sizes1, "problem_sizes1", dtypes=(torch.int32,), contiguous=True)
    check_tensor(problem_sizes2, "problem_sizes2", dtypes=(torch.int32,), contiguous=True)
    check_tensor(
        input_permutation, "input_permutation", dtypes=(torch.int32,), ndim=1, contiguous=True
    )
    check_tensor(
        output_permutation, "output_permutation", dtypes=(torch.int32,), ndim=1, contiguous=True
    )
    num_experts, n, k = int(num_experts), int(n), int(k)
    check(num_experts > 0, "num_experts must be positive")
    check(n > 0 and k > 0, "n and k must be positive")
    check(expert_offsets.numel() >= num_experts + 1, "expert_offsets needs num_experts + 1 entries")
    check(problem_sizes1.numel() >= 3 * num_experts, "problem_sizes1 needs 3 * num_experts entries")
    check(problem_sizes2.numel() >= 3 * num_experts, "problem_sizes2 needs 3 * num_experts entries")
    numel = topk_ids.numel()
    check(
        input_permutation.numel() >= numel and output_permutation.numel() >= numel,
        "permutation buffers too small",
    )
    if blockscale_offsets is not None:
        check_tensor(
            blockscale_offsets,
            "blockscale_offsets",
            dtypes=(torch.int32,),
            ndim=1,
            contiguous=True,
        )
        check(
            blockscale_offsets.numel() >= num_experts + 1,
            "blockscale_offsets needs num_experts + 1 entries",
        )
    same_device(
        topk_ids,
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        input_permutation,
        output_permutation,
        blockscale_offsets,
    )
    block_e = max(2, next_pow2(num_experts))
    with device_guard(topk_ids):
        atomic_buffer = torch.zeros(num_experts, dtype=torch.int32, device=topk_ids.device)
        launch_expert_histogram(topk_ids, atomic_buffer, num_experts, shift=0)
        compute_problem_sizes_kernel[(1,)](
            atomic_buffer,
            problem_sizes1,
            problem_sizes2,
            num_experts,
            n,
            k,
            BLOCK_E=block_e,
            num_warps=4,
        )
        if blockscale_offsets is not None:
            compute_expert_blockscale_offsets_kernel[(1,)](
                problem_sizes1,
                expert_offsets,
                blockscale_offsets,
                atomic_buffer,
                num_experts,
                BLOCK_E=block_e,
                num_warps=4,
            )
        else:
            compute_expert_offsets_kernel[(1,)](
                problem_sizes1,
                expert_offsets,
                atomic_buffer,
                num_experts,
                BLOCK_E=block_e,
                num_warps=4,
            )
        launch_compute_arg_sorts(
            topk_ids,
            atomic_buffer,
            input_permutation,
            output_permutation,
            topk_ids.size(1),
            num_experts,
        )


# ======================================================================================
# fp8_blockwise_moe_kernel.cu :: grouped GEMM
# ======================================================================================
# The tile shapes mirror the CUTLASS configs: swap-AB SmallM = 128 x 32 x 128, direct = 64/128 x 128 x 128.
_DIRECT_CONFIGS = [
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "GROUP_M": 8}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "GROUP_M": 8}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "GROUP_M": 8}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "GROUP_M": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "GROUP_M": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "GROUP_M": 8}, num_warps=4, num_stages=4),
]
_SWAP_CONFIGS = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "GROUP_M": 8}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 16, "GROUP_M": 8}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "GROUP_M": 8}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "GROUP_M": 8}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "GROUP_M": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "GROUP_M": 8}, num_warps=4, num_stages=4),
]


@triton.jit
def _fp8_blockwise_grouped_gemm(
    x_ptrs,
    y_ptrs,
    d_ptrs,
    sx_ptrs,
    sy_ptrs,
    ld_x_ptr,
    ld_y_ptr,
    ld_d_ptr,
    problem_sizes_ptr,
    num_experts,
    n_key,
    k_key,
    m_bucket,
    OUT_DTYPE: tl.constexpr,
    SWAP_AB: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """D_e[m', n'] = sum_kb  sx[m', kb] * sy[n', kb] * (X_e[m', kb-block] . Y_e[n', kb-block])   for every expert e.

    normal  (SWAP_AB=False): X = activations (per-row scales), Y = weights (128x128 block scales), D row-major [M_e, N]
    swap-AB (SWAP_AB=True) : X = weights (block scales), Y = activations (per-row scales), D stored column-major, i.e.
                             D[m', n'] lands at C[n', m'] of the row-major [M_e, N] output.
    """
    pid = tl.program_id(0)

    # ---- tile scheduler: flat tile id -> (expert, tile_m, tile_n) --------------------------------------------
    e_idx = tl.arange(0, BLOCK_E)
    e_ok = e_idx < num_experts
    Mv = tl.load(problem_sizes_ptr + e_idx * 3, mask=e_ok, other=0)
    Nv = tl.load(problem_sizes_ptr + e_idx * 3 + 1, mask=e_ok, other=0)
    tiles_m = (Mv + BLOCK_M - 1) // BLOCK_M
    tiles_n = (Nv + BLOCK_N - 1) // BLOCK_N
    tiles = tiles_m * tiles_n
    incl = tl.cumsum(tiles, axis=0)
    total_tiles = tl.sum(tiles, axis=0)
    if pid >= total_tiles:
        return
    e = tl.sum((incl <= pid).to(tl.int32), axis=0)  # experts whose tiles all precede this one
    is_e = e_idx == e
    first_tile = tl.sum(tl.where(is_e, incl - tiles, 0), axis=0)
    tm = tl.sum(tl.where(is_e, tiles_m, 0), axis=0)
    tn = tl.sum(tl.where(is_e, tiles_n, 0), axis=0)
    M = tl.sum(tl.where(is_e, Mv, 0), axis=0)
    N = tl.sum(tl.where(is_e, Nv, 0), axis=0)
    K = tl.load(problem_sizes_ptr + e * 3 + 2)
    local = pid - first_tile
    # grouped ordering inside the expert: consecutive programs share X / Y tiles in L2
    num_in_group = GROUP_M * tn
    group_id = local // num_in_group
    first_m = group_id * GROUP_M
    group_size_m = tl.minimum(tm - first_m, GROUP_M)
    pid_m = first_m + (local % num_in_group) % group_size_m
    pid_n = (local % num_in_group) // group_size_m

    # ---- per-expert operands (device pointer tables, as in the CUTLASS pointer-array GEMM) -----------------------
    x_base = tl.load(x_ptrs + e).to(tl.pointer_type(tl.float8e4nv))
    y_base = tl.load(y_ptrs + e).to(tl.pointer_type(tl.float8e4nv))
    d_base = tl.load(d_ptrs + e).to(tl.pointer_type(OUT_DTYPE))
    sx_base = tl.load(sx_ptrs + e).to(tl.pointer_type(tl.float32))
    sy_base = tl.load(sy_ptrs + e).to(tl.pointer_type(tl.float32))
    ld_x = tl.load(ld_x_ptr + e)
    ld_y = tl.load(ld_y_ptr + e)
    ld_d = tl.load(ld_d_ptr + e)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_ok = offs_m < M
    n_ok = offs_n < N
    x_tile = (
        x_base + offs_m.to(tl.int64)[:, None] * ld_x + offs_k[None, :]
    )  # [BM, BK]  (K contiguous)
    y_tile = (
        y_base + offs_n.to(tl.int64)[None, :] * ld_y + offs_k[:, None]
    )  # [BK, BN]  (K contiguous)

    SK = K // BLOCK_K  # number of 128-wide K blocks == number of scale columns
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for kb in range(0, SK):
        xt = tl.load(x_tile + kb * BLOCK_K, mask=m_ok[:, None], other=0.0)
        yt = tl.load(y_tile + kb * BLOCK_K, mask=n_ok[None, :], other=0.0)
        part = tl.dot(xt, yt)  # fp8 x fp8 -> fp32, fresh accumulator per 128-wide K block
        if SWAP_AB:
            sx = tl.load(
                sx_base + (offs_m // 128) * SK + kb, mask=m_ok, other=0.0
            )  # weights: 128-row blocks
            sy = tl.load(sy_base + offs_n * SK + kb, mask=n_ok, other=0.0)  # tokens: per row
        else:
            sx = tl.load(sx_base + offs_m * SK + kb, mask=m_ok, other=0.0)  # tokens: per row
            sy = tl.load(
                sy_base + (offs_n // 128) * SK + kb, mask=n_ok, other=0.0
            )  # weights: 128-row blocks
        acc += part * (
            sx[:, None] * sy[None, :]
        )  # promotion: scale in fp32, add into the fp32 accumulator

    if SWAP_AB:
        d_tile = d_base + offs_n.to(tl.int64)[None, :] * ld_d + offs_m[:, None]
    else:
        d_tile = d_base + offs_m.to(tl.int64)[:, None] * ld_d + offs_n[None, :]
    tl.store(d_tile, acc.to(OUT_DTYPE), mask=m_ok[:, None] & n_ok[None, :])


_GEMM_KEY = ["n_key", "k_key", "m_bucket", "BLOCK_E"]
fp8_blockwise_grouped_gemm_kernel = autotune(configs=_DIRECT_CONFIGS, key=_GEMM_KEY)(
    _fp8_blockwise_grouped_gemm
)
fp8_blockwise_grouped_gemm_swap_ab_kernel = autotune(configs=_SWAP_CONFIGS, key=_GEMM_KEY)(
    _fp8_blockwise_grouped_gemm
)


def _mem_layout_ok(t: torch.Tensor, e: int, rows: int, cols: int) -> bool:
    """True if ``t`` is [e, rows, cols] contiguous, or the transposed view [e, cols, rows] of such a tensor."""
    if tuple(t.shape) == (e, rows, cols):
        return t.is_contiguous()
    if tuple(t.shape) == (e, cols, rows):
        return tuple(t.stride()) == (rows * cols, 1, cols)
    return False


def fp8_blockwise_scaled_grouped_mm(
    output: torch.Tensor,
    a_ptrs: torch.Tensor,
    b_ptrs: torch.Tensor,
    out_ptrs: torch.Tensor,
    a_scales_ptrs: torch.Tensor,
    b_scales_ptrs: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    scales_a: torch.Tensor,
    scales_b: torch.Tensor,
    stride_a: torch.Tensor,
    stride_b: torch.Tensor,
    stride_c: torch.Tensor,
    layout_sfa: torch.Tensor,
    layout_sfb: torch.Tensor,
    problem_sizes: torch.Tensor,
    expert_offsets: torch.Tensor,
    workspace: torch.Tensor,
    *,
    swap_ab: bool | None = None,
) -> None:
    """Grouped blockwise-scaled FP8 GEMM, signature-compatible with the CUDA op (results written to ``output``).

    ``problem_sizes[e] = (M_e, N, K)`` (int32) and ``expert_offsets[e]`` (row offset of expert ``e``; length E, i.e. the
    CUDA op is called with ``expert_offsets[:-1]``).  ``stride_a/b/c`` hold the leading dimensions (K, K, N) as int64.
    ``a_ptrs .. b_scales_ptrs`` (int64 ``[E]``) and ``layout_sfa/sfb`` (int32 ``[E, >=5]``) are workspaces that this
    call fills (see ``get_group_gemm_starts``); ``workspace`` is accepted for signature compatibility only.
    ``swap_ab=None`` applies the CUDA SM90 rule (``M_total <= 2048``).
    Requires ``N % 128 == 0`` and ``K % 128 == 0`` (the scale-pointer arithmetic of the CUDA code assumes it) and an
    FP8-capable GPU (sm_89+).
    """
    # ---- the TORCH_CHECKs of the CUDA op ----------------------------------------------------------------------
    check(problem_sizes.dim() == 2, "problem_sizes must be 2D tensor")
    check(problem_sizes.size(1) == 3, "problem_sizes must have shape (num_experts, 3)")
    check(
        problem_sizes.size(0) == expert_offsets.size(0),
        "Number of experts in problem_sizes must match expert_offsets",
    )
    check(problem_sizes.dtype == torch.int32, "problem_sizes must be int32")
    check(a.dtype == torch.float8_e4m3fn, "a must be kFloat8_e4m3fn")
    check(b.dtype == torch.float8_e4m3fn, "b must be kFloat8_e4m3fn")
    check(output.dtype in _OUT_DTYPES, "output must be bfloat16 or half")
    check(scales_a.dtype == torch.float32, "scales_a must be float32")
    check(scales_b.dtype == torch.float32, "scales_b must be float32")
    for name, t in (("stride_a", stride_a), ("stride_b", stride_b), ("stride_c", stride_c)):
        check(t.dtype == torch.int64, f"{name} must be int64")
        check(t.dim() == 1, f"{name} must be 1D tensor")
    check(layout_sfa.dtype == torch.int32, "layout_sfa must be int32")
    check(layout_sfb.dtype == torch.int32, "layout_sfb must be int32")
    check(expert_offsets.dtype == torch.int32, "expert_offsets must be int32")
    check(output.dim() == 2, "output must be 2D tensor")
    check(a.dim() == 2, "a must be 2D tensor")
    check(b.dim() == 3, "b must be 3D tensor")
    check(scales_a.dim() == 2, "scales_a must be 2D tensor")
    check(scales_b.dim() == 3, "scales_b must be 3D tensor")
    check(layout_sfa.dim() == 2, "layout_sfa must be 2D tensor")
    check(layout_sfb.dim() == 2, "layout_sfb must be 2D tensor")
    for name, t in (
        ("a_ptrs", a_ptrs),
        ("b_ptrs", b_ptrs),
        ("out_ptrs", out_ptrs),
        ("a_scales_ptrs", a_scales_ptrs),
        ("b_scales_ptrs", b_scales_ptrs),
        ("workspace", workspace),
    ):
        check(t.dim() == 1, f"{name} must be 1D tensor")
    for t in (output, a, b, scales_a, scales_b, problem_sizes, expert_offsets):
        check(t.is_cuda, "all tensors must be CUDA tensors")
    same_device(
        output,
        a,
        b,
        scales_a,
        scales_b,
        problem_sizes,
        expert_offsets,
        a_ptrs,
        b_ptrs,
        out_ptrs,
        a_scales_ptrs,
        b_scales_ptrs,
        stride_a,
        stride_b,
        stride_c,
        layout_sfa,
        layout_sfb,
        workspace,
    )

    # ---- shape contract of the pointer arithmetic -------------------------------------------------------------
    num_experts = expert_offsets.size(0)
    m_total, k = a.shape
    n = output.size(1)
    for name, t in (
        ("a_ptrs", a_ptrs),
        ("b_ptrs", b_ptrs),
        ("out_ptrs", out_ptrs),
        ("a_scales_ptrs", a_scales_ptrs),
        ("b_scales_ptrs", b_scales_ptrs),
    ):
        check(t.dtype == torch.int64, f"{name} must be int64")
        check(t.is_contiguous(), f"{name} must be contiguous")
        check(t.numel() >= num_experts, f"{name} needs num_experts entries")
    for name, t in (
        ("stride_a", stride_a),
        ("stride_b", stride_b),
        ("stride_c", stride_c),
    ):
        check(t.is_contiguous(), f"{name} must be contiguous")
        check(t.numel() >= num_experts, f"{name} needs num_experts entries")
    check(problem_sizes.is_contiguous(), "problem_sizes must be contiguous")
    check(expert_offsets.is_contiguous(), "expert_offsets must be contiguous")
    check(layout_sfa.is_contiguous() and layout_sfb.is_contiguous(), "layouts must be contiguous")
    check(output.size(0) == m_total, "output and a must have the same number of rows")
    check(n % SCALE_BLOCK == 0 and k % SCALE_BLOCK == 0, "N and K must be multiples of 128")
    check(a.is_contiguous() and output.is_contiguous(), "a and output must be contiguous")
    check(
        b.size(0) == num_experts and _mem_layout_ok(b, num_experts, n, k),
        "b must have memory layout [E][N][K]",
    )
    check(
        scales_b.size(0) == num_experts
        and _mem_layout_ok(scales_b, num_experts, n // 128, k // 128),
        "scales_b must have memory layout [E][N/128][K/128]",
    )
    check(
        tuple(scales_a.shape) == (m_total, k // 128) and scales_a.is_contiguous(),
        "scales_a must be a contiguous [M_total, K/128] tensor",
    )
    cap = torch.cuda.get_device_capability(a.device)
    if (cap[0], cap[1]) < (8, 9):
        raise NotImplementedError(
            f"fp8_blockwise_scaled_grouped_mm needs FP8 tensor cores (sm_89+), got sm_{cap[0]}{cap[1]}"
        )
    if num_experts == 0 or m_total == 0:
        return

    swap = (m_total <= 2048) if swap_ab is None else bool(swap_ab)
    block_e = max(2, next_pow2(num_experts))
    with device_guard(a):
        problem_sizes_t = (
            torch.empty((num_experts, 3), dtype=torch.int32, device=a.device)
            if swap
            else problem_sizes
        )
        if swap:  # weights are the "A" operand of the swapped problem
            get_group_gemm_starts(
                expert_offsets,
                a_ptrs,
                b_ptrs,
                out_ptrs,
                a_scales_ptrs,
                b_scales_ptrs,
                b,
                a,
                output,
                scales_b,
                scales_a,
                layout_sfa,
                layout_sfb,
                problem_sizes,
                problem_sizes_t,
                True,
            )
            kernel = fp8_blockwise_grouped_gemm_swap_ab_kernel
            ld_x, ld_y = stride_b, stride_a
            sizes = problem_sizes_t

            def tiles_bound(meta: dict[str, int]) -> int:
                return cdiv(n, meta["BLOCK_M"]) * (cdiv(m_total, meta["BLOCK_N"]) + num_experts)
        else:
            get_group_gemm_starts(
                expert_offsets,
                a_ptrs,
                b_ptrs,
                out_ptrs,
                a_scales_ptrs,
                b_scales_ptrs,
                a,
                b,
                output,
                scales_a,
                scales_b,
                layout_sfa,
                layout_sfb,
                problem_sizes,
                problem_sizes,
                False,
            )
            kernel = fp8_blockwise_grouped_gemm_kernel
            ld_x, ld_y = stride_a, stride_b
            sizes = problem_sizes

            def tiles_bound(meta: dict[str, int]) -> int:
                return (cdiv(m_total, meta["BLOCK_M"]) + num_experts) * cdiv(n, meta["BLOCK_N"])

        kernel[lambda meta: (tiles_bound(meta),)](
            a_ptrs,
            b_ptrs,
            out_ptrs,
            a_scales_ptrs,
            b_scales_ptrs,
            ld_x,
            ld_y,
            stride_c,
            sizes,
            num_experts,
            n,
            k,
            next_pow2(m_total),
            OUT_DTYPE=tl.bfloat16 if output.dtype == torch.bfloat16 else tl.float16,
            SWAP_AB=swap,
            BLOCK_E=block_e,
            BLOCK_K=SCALE_BLOCK,
        )
