from __future__ import annotations

import triton
import triton.language as tl


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
def swizzled_128x4_offset(row, col, cols_padded):
    inner_k = col % 4
    inner_m = (row % 128) // 32
    outer_m = row % 32
    k_tile = col // 4
    m_tile = row // 128
    num_k_tiles = (cols_padded + 3) // 4
    return m_tile * num_k_tiles * 512 + k_tile * 512 + outer_m * 16 + inner_m * 4 + inner_k
