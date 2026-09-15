"""Caps — các capability bitmask dùng chung giữa Tier-1 (mask producer) và
Tier-2 (custom op).

Tách khỏi ayaka.sampling.mask.producer để tránh cycle import khi tầng plan
(ayaka.plan.SamplingPlan) cần đọc Cap mà không kéo theo cả cây sampling.

Consumer và ý nghĩa (A4):
  * ARGMAX_INVARIANT — mask của producer không đổi kết quả argmax KHI
    winner unmasked vẫn allowed (nếu winner bị chặn thì sampler fallback
    về đường apply-mask đầy đủ). Cho phép Sampler greedy fast-path: argmax
    trên logits chưa mask, kiểm winner trong bitmask, chỉ apply mask khi có
    row vi phạm.
  * CUDAGRAPH_SAFE — op/producer chạy được bên trong CUDA graph capture
    (không sync, không allocation phụ thuộc shape động). SamplingPlan.
    graph_capturable đọc cap này thay vì cấm mọi custom op vô điều kiện.
  * SPEC_VERIFIABLE — producer chấp nhận draft tokens (speculative decode):
    emit(draft) trả số token accepted + rollback được. SamplingPlanner.build
    từ chối draft của producer thiếu cap này.
  * COMMUTATIVE — producer chạy được SONG SONG với producer khác trên cùng
    logits row (mask kết hợp bằng AND). Producer thiếu cap phải chạy tuần
    tự theo thứ tự khai báo.
"""

from __future__ import annotations

from enum import Flag, auto

__all__ = ["Cap"]


class Cap(Flag):
    """Capabilities advertised by Tier-1 mask producers and consumed by Tier-2.

    These flags form the contract between mask producers and the sampling
    planner/sampler. A capability must only be set when the producer satisfies
    the corresponding semantic and runtime guarantees.

    Attributes:
        NONE:
            No special capability is guaranteed.

        ARGMAX_INVARIANT:
            Applying the producer's mask does not change the argmax as long as
            the unmasked winner remains allowed. This permits the sampler to
            use a greedy fast-path: compute argmax on unmasked logits, inspect
            the winner, and apply the full mask only when that winner is
            rejected.

        CUDAGRAPH_SAFE:
            The producer is safe to execute inside CUDA graph capture. In
            particular, it must not perform synchronization or shape-dependent
            dynamic allocation that would make capture unsafe. SamplingPlan
            uses this capability when determining whether a plan is
            graph-capturable.

        SPEC_VERIFIABLE:
            The producer supports speculative decoding with draft tokens.
            Its ``emit(draft)`` operation must report the number of accepted
            tokens and support rollback. A SamplingPlanner must reject a
            speculative plan using a producer without this capability.

        COMMUTATIVE:
            The producer can run concurrently with other commutative producers
            on the same logits row, with their masks combined by AND.
            Producers without this capability must be applied sequentially in
            declaration order.
    """

    NONE = 0
    ARGMAX_INVARIANT = auto()
    CUDAGRAPH_SAFE = auto()
    SPEC_VERIFIABLE = auto()
    COMMUTATIVE = auto()
