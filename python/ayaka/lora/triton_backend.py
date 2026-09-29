"""Triton adapter between the LoRA backend contract and the public kernels.

The adapter owns no residency: it turns a validated ``LoRAExpandContext`` plus
the stable bank tensors into one public shrink call and one public expand call.
Algorithm choice is a pre-launch decision derived from the planned phase:
``prefill``/``mixed`` use the grouped SGMV pair when the grouped tables are
present, everything else uses BGMV. Unsupported dtype/shape/stride is rejected
before the first kernel enqueues, and a launched kernel failure is propagated
without a second torch attempt.
"""

from __future__ import annotations

import torch

from ayaka.kernel.triton.lora._common import validate_bgmv, validate_sgmv
from ayaka.lora.backend import (
    LoRAExecutionPlan,
    LoRAExpandContext,
    LoRAWorkspace,
    LoRAWorkspaceSpec,
    ReadFootprint,
)

__all__ = ["TritonLoRABackend"]

_PREFILL_PHASES = ("prefill", "mixed")


class TritonLoRABackend:
    name = "triton"
    version = "1"
    read_footprint = ReadFootprint.SELECTED_SLOTS

    def plan(
        self,
        *,
        tokens: int,
        rank: int,
        output: int,
        capacity: int,
        dtype: torch.dtype,
        phase: str = "reference",
    ) -> LoRAExecutionPlan:
        if tokens < 1 or rank < 1 or output < 1 or capacity < 1:
            raise ValueError("LoRA plan requires positive tokens/rank/output/capacity")
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("triton LoRA requires float16, bfloat16 or float32")
        algorithm = "sgmv" if phase in _PREFILL_PHASES else "bgmv"
        return LoRAExecutionPlan(
            backend=self.name,
            version=self.version,
            phase=phase,
            algorithm=algorithm,
            rank=rank,
            capacity=capacity,
            read_footprint=self.read_footprint,
            workspace=LoRAWorkspaceSpec(tokens, rank, output, dtype, include_mask=False),
        )

    @staticmethod
    def _latent(workspace: LoRAWorkspace, tokens: int, rank_capacity: int) -> torch.Tensor:
        if workspace.rank < rank_capacity:
            raise ValueError("latent workspace is smaller than the rank bank")
        return workspace.low[: tokens * rank_capacity].view(tokens, rank_capacity)

    @staticmethod
    def _grouped(context: LoRAExpandContext) -> bool:
        return (
            context.phase in _PREFILL_PHASES
            and context.slot_counts is not None
            and context.segment_offsets is not None
            and context.token_permutation is not None
        )

    def run_expand(
        self,
        context: LoRAExpandContext,
        x: torch.Tensor,
        output: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        ranks: torch.Tensor,
        offset: int,
    ) -> None:
        low = self._latent(context.workspace, int(x.shape[0]), int(a.shape[1]))
        if self._grouped(context):
            from ayaka.kernel.triton.lora import sgmv_expand, sgmv_shrink

            assert context.slot_counts is not None
            assert context.segment_offsets is not None
            assert context.token_permutation is not None
            validate_sgmv(
                x,
                a,
                ranks,
                context.slot_counts,
                context.segment_offsets,
                context.token_permutation,
                low,
                b,
                output,
                offset,
            )
            sgmv_shrink(
                x,
                a,
                ranks,
                context.slot_counts,
                context.segment_offsets,
                context.token_permutation,
                low,
            )
            sgmv_expand(
                low,
                b,
                ranks,
                context.slot_counts,
                context.segment_offsets,
                context.token_permutation,
                output,
                offset,
            )
            return
        from ayaka.kernel.triton.lora import bgmv_expand, bgmv_shrink

        validate_bgmv(x, a, ranks, context.rows, low, b, output, offset)
        bgmv_shrink(x, a, ranks, context.rows, low)
        bgmv_expand(low, b, ranks, context.rows, output, offset)

    def run_expand_slice(
        self,
        x: torch.Tensor,
        output: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        rows: torch.Tensor,
        ranks: torch.Tensor,
        workspace: LoRAWorkspace,
        offset: int,
    ) -> None:
        low = self._latent(workspace, int(x.shape[0]), int(a.shape[1]))
        validate_bgmv(x, a, ranks, rows, low, b, output, offset)
        from ayaka.kernel.triton.lora import bgmv_expand, bgmv_shrink

        bgmv_shrink(x, a, ranks, rows, low)
        bgmv_expand(low, b, ranks, rows, output, offset)
