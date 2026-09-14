from __future__ import annotations

from typing import Any

import torch

from ayaka.distributed.parallel import ParallelContext
from ayaka.layers.linear.core import PackedColumnParallelLinear
from ayaka.layers.linear.layout import ProjectionKind, ProjectionSpec
from ayaka.layers.linear.methods import LinearMethodBase
from ayaka.layers.quantization.base import BaseQuantization


class QKVParallelLinear(PackedColumnParallelLinear):
    """Packed Q/K/V projection with GQA/MQA head replication semantics."""

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: BaseQuantization | LinearMethodBase | str | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
        v_head_size: int | None = None,
        parallel_context: ParallelContext | None = None,
        device: torch.device | str | None = None,
        **runtime: Any,
    ) -> None:
        total_num_kv_heads = total_num_heads if total_num_kv_heads is None else total_num_kv_heads
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.v_head_size = head_size if v_head_size is None else v_head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        specs = [
            ProjectionSpec.heads(
                "q", num_heads=total_num_heads, head_dim=head_size, aliases=("q_proj",)
            ),
            ProjectionSpec.heads(
                "k",
                num_heads=total_num_kv_heads,
                head_dim=head_size,
                aliases=("k_proj",),
                replicate=True,
            ),
            ProjectionSpec.heads(
                "v",
                num_heads=total_num_kv_heads,
                head_dim=self.v_head_size,
                aliases=("v_proj",),
                replicate=True,
            ),
        ]
        self.output_sizes = [spec.output_size for spec in specs]
        super().__init__(
            hidden_size,
            specs,
            bias=bias,
            gather_output=False,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
            parallel_context=parallel_context,
            device=device,
            **runtime,
        )
        self.num_heads = self.layout["q"].local_num_heads or 0
        self.num_kv_heads = self.layout["k"].local_num_heads or 0
        self.num_kv_head_replicas = self.layout["k"].num_head_replicas


class SharedKVQParallelLinear(PackedColumnParallelLinear):
    """Q plus one replicated shared K=V head, as used by DeepSeek V4 MQA."""

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        *,
        shared_kv_size: int | None = None,
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
        shared_kv_size = head_size if shared_kv_size is None else shared_kv_size
        super().__init__(
            hidden_size,
            [
                ProjectionSpec.heads("q", num_heads=total_num_heads, head_dim=head_size),
                ProjectionSpec(
                    "shared_kv",
                    shared_kv_size,
                    ProjectionKind.REPLICATED,
                    aliases=("kv_proj",),
                ),
            ],
            bias=bias,
            gather_output=False,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
            parallel_context=parallel_context,
            device=device,
            **runtime,
        )
