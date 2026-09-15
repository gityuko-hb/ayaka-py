"""Philox4x32-10 cho Triton — TWIN BIT-EXACT của ayaka.sampling.rng.

Hai module PHẢI cho cùng kết quả cho cùng (seed, address): torch int64
(ayaka.sampling.rng.philox4x32_10) và Triton int64 (module này) đều làm số học
two's-complement trên giá trị u32 đóng gói trong int64, nên bitwise/mask/shift
cho kết quả bit-identical. Khoảng cách giữa hai bản là cách bug trốn — battery
test (tests/test_sampling_kernel_gpu.py) bắt buộc phải so chéo.

Địa chỉ hoá: `flat = offset * n_cols + col` (linear addressing, batch-invariant).
Một counter Philox (4x u32) cho 4 word (w0..w3) → 2 uniform: draw chẵn từ
(w0, w1), draw lẻ từ (w2, w3); counter = flat >> 1, draw = flat & 1.

Ghi chú style: các helper jit dưới đây trả MỘT giá trị duy nhất (thay vì tuple)
— pyright coi call của JITFunction là NoReturn, nên unpack tuple trong jit code
sẽ thành lỗi typecheck giả; single-return cũng gọn cho Triton CSE.
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
    """Low 32 bit của tích u32×u32 — 16-bit chunks, mọi trung gian < 2^51."""
    a_lo = a & 0xFFFF
    a_hi = a >> 16
    b_lo = b & 0xFFFF
    b_hi = b >> 16
    lh = a_lo * b_hi + a_hi * b_lo
    mid = (a_lo * b_lo) + (lh << 16)
    return mid & _MASK32


@triton.jit
def _mulhi32(a, b):
    """High 32 bit của tích u32×u32 — cùng phép nhân với _mullo32, chỉ khác
    phần lấy carry; CSE của Triton gộp được hai lời gọi nếu cùng operand."""
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
    """Philox4x32, 10 rounds — bit-exact với philox4x32_10 trong rng.py và KAT
    Random123 (tests/test_sampling_rng.py pin 3 vector chính thức)."""
    # Một số caller nạp seed/offset dạng uint64 (topk_topp port) — bitcast về
    # int64 để kiểu loop-carried nhất quán; two's-complement làm bitwise/add
    # wrap-around bit-identical giữa hai kiểu.
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
    """Chạy trọn 10 rounds cho MỘT counter, trả u53-mantissa bits của draw
    được chọn (odd: 0 lấy (w0,w1), 1 lấy (w2,w3)). Trả int64 [0, 2^53)."""
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
    """Uniform float64 trong [0, 1) tại địa chỉ `flat` của stream `seed`.

    counter = flat >> 1, draw = flat & 1. Phải khớp bit
    counter_uniform/counter_uniform_cols trong ayaka.sampling.rng.
    """
    counter = flat >> 1
    v = _philox_u53_bits(seed, counter, flat & 1)
    return ((v >> 11) & ((1 << 53) - 1)).to(tl.float64) * (1.0 / (1 << 53))


@triton.jit
def philox_u01_f32(seed, flat):
    """Biến thể float32 24-bit cho đường CDF-scan (topk_topp.py).

    Lấy 24 bit cao của draw để u luôn biểu diễn được CHÍNH XÁC trong float32
    (53-bit cast xuống float32 có thể tròn thành 1.0 — vô hại ở gumbel-argmax
    nhưng đổi semantics CDF-scan so với bản FlashInfer port).
    """
    counter = flat >> 1
    v = _philox_u53_bits(seed, counter, flat & 1)
    return ((v >> 29) & 0x00FFFFFF).to(tl.float32) / 16777216.0
