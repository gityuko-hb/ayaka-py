from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from ayaka.distributed.parallel import ParallelContext
from ayaka.layers.linear.core import ColumnParallelLinear, ReplicatedLinear
from ayaka.layers.linear.methods import LinearMethodBase
from ayaka.layers.quantization.base import BaseQuantization


class LowRankColumnParallelLinear(nn.Module):
    """Replicated A projection followed by a column-parallel B projection."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        rank: int,
        *,
        norm: nn.Module | Callable[[torch.Tensor], torch.Tensor] | None = None,
        bias: bool = False,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: BaseQuantization | LinearMethodBase | str | None = None,
        prefix: str = "",
        return_bias: bool = True,
        disable_tp: bool = False,
        parallel_context: ParallelContext | None = None,
        device: torch.device | str | None = None,
        **runtime: Any,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.input_size = input_size
        self.output_size = output_size
        self.rank = rank
        self.norm = norm
        self.a_proj = ReplicatedLinear(
            input_size,
            rank,
            bias=False,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=f"{prefix}.a_proj" if prefix else "a_proj",
            return_bias=False,
            disable_tp=disable_tp,
            parallel_context=parallel_context,
            device=device,
            **runtime,
        )
        self.b_proj = ColumnParallelLinear(
            rank,
            output_size,
            bias=bias,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=f"{prefix}.b_proj" if prefix else "b_proj",
            return_bias=return_bias,
            disable_tp=disable_tp,
            parallel_context=parallel_context,
            device=device,
            **runtime,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        latent = self.a_proj(x)
        assert isinstance(latent, torch.Tensor)
        if self.norm is not None:
            latent = self.norm(latent)
        return self.b_proj(latent)
