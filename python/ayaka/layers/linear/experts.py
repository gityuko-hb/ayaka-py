from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ayaka.distributed.parallel import ParallelContext, divide
from ayaka.layers._common import check_dtype
from ayaka.layers.base import BaseLayer
from ayaka.layers.linear.parallel import prepare_parallel_runtime
from ayaka.layers.linear.weight_loading import WeightLoadError, set_weight_attrs
from ayaka.layers.quantization.base import QuantizationTarget
from ayaka.utils.validation import require_int


class ExpertParallelLinear(BaseLayer):
    """Native/reference expert linear with contiguous expert parallelism.

    Production routed-MoE kernels can replace this module, while this path stays
    useful for correctness tests, CPU execution, checkpoint validation, and small
    expert batches.
    """

    def __init__(
        self,
        num_experts: int,
        input_size: int,
        output_size: int,
        *,
        bias: bool = False,
        params_dtype: torch.dtype | None = None,
        disable_ep: bool = False,
        parallel_context: ParallelContext | None = None,
        device: torch.device | str | None = None,
        activation_dtype: torch.dtype | None = None,
        **runtime: Any,
    ) -> None:
        device, context, runtime = prepare_parallel_runtime(
            device,
            parallel_context,
            runtime,
            disabled=disable_ep,
            expert=True,
        )
        super().__init__(**runtime)
        for name, value in (
            ("num_experts", num_experts),
            ("input_size", input_size),
            ("output_size", output_size),
        ):
            require_int(value, name, minimum=1)
        self.parallel_context = context
        self.num_experts = num_experts
        self.input_size = input_size
        self.output_size = output_size
        self.local_num_experts = divide(num_experts, self.ep_size, name="num_experts")
        self.expert_start = self.ep_rank * self.local_num_experts
        self.expert_end = self.expert_start + self.local_num_experts
        dtype = params_dtype or torch.get_default_dtype()
        if self.quant_config is not None:
            self.create_weights(
                device=device,
                params_dtype=dtype,
                activation_dtype=activation_dtype,
                target=QuantizationTarget.EXPERT,
                num_experts=num_experts,
                local_num_experts=self.local_num_experts,
                input_size=input_size,
                output_size=output_size,
                weight_loader=self.weight_loader,
            )
        else:
            check_dtype(dtype)
            if activation_dtype is not None and activation_dtype != dtype:
                raise ValueError("dense expert activation and parameter dtypes must match")
            if device.type != "meta":
                device = self.runtime_context(
                    device, dtype, target=QuantizationTarget.EXPERT
                ).device
            self.weight = nn.Parameter(
                torch.empty(
                    self.local_num_experts, output_size, input_size, dtype=dtype, device=device
                ),
                requires_grad=False,
            )
            set_weight_attrs(self.weight, {"weight_loader": self.weight_loader})
        if bias:
            self.bias = nn.Parameter(
                torch.empty(
                    self.local_num_experts,
                    output_size,
                    dtype=dtype,
                    device=device,
                ),
                requires_grad=False,
            )
            set_weight_attrs(self.bias, {"weight_loader": self.weight_loader})
        else:
            self.register_parameter("bias", None)

    def weight_loader(
        self,
        parameter: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: object | None = None,
    ) -> None:
        if loaded_shard_id is not None:
            raise WeightLoadError("expert weights do not accept projection IDs")
        if parameter.is_meta or loaded_weight.is_meta:
            raise WeightLoadError("materialize weights before copying")
        if loaded_weight.shape == parameter.shape:
            source = loaded_weight
        elif loaded_weight.shape[0] == self.num_experts:
            source = loaded_weight.narrow(0, self.expert_start, self.local_num_experts)
        else:
            raise WeightLoadError(
                f"expert checkpoint shape {tuple(loaded_weight.shape)} is neither "
                f"local {tuple(parameter.shape)} nor globally expert-sharded"
            )
        if source.shape != parameter.shape:
            raise WeightLoadError(
                f"expert destination {tuple(parameter.shape)} != source {tuple(source.shape)}"
            )
        parameter.data.copy_(source.to(parameter))

    def process_weights_after_loading(self) -> None:
        if self.quant_config is not None:
            super().process_weights_after_loading()
        else:
            if any(t.is_meta for t in self.parameters()):
                raise RuntimeError("materialize expert weights before finalization")
            self.runtime_context(
                self.weight.device, self.weight.dtype, target=QuantizationTarget.EXPERT
            )

    def forward(
        self,
        x: torch.Tensor,
        expert_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.quant_config is not None:
            return self.apply_quantization(x, expert_ids)
        if expert_ids is None:
            if x.ndim < 2 or x.shape[0] != self.local_num_experts:
                raise ValueError("dense expert input must have leading dimension local_num_experts")
            output = self.run_kernel(torch.einsum, "e...i,eoi->e...o", x, self.weight)
            if self.bias is not None:
                shape = (self.local_num_experts,) + (1,) * (x.ndim - 2) + (self.output_size,)
                output = output + self.bias.reshape(shape)
            return output

        if x.ndim != 2 or expert_ids.ndim != 1 or expert_ids.shape[0] != x.shape[0]:
            raise ValueError("routed input requires x=[tokens,input] and expert_ids=[tokens]")
        if expert_ids.dtype not in (torch.int32, torch.int64) or expert_ids.device != x.device:
            raise ValueError("expert_ids must be int32/int64 on the input device")
        local_ids = expert_ids.to(torch.long) - self.expert_start
        if torch.any(local_ids < 0) or torch.any(local_ids >= self.local_num_experts):
            raise ValueError(
                f"expert_ids must belong to local range [{self.expert_start}, {self.expert_end})"
            )
        selected_weight = self.weight.index_select(0, local_ids)
        output = self.run_kernel(torch.bmm, selected_weight, x.unsqueeze(-1)).squeeze(-1)
        if self.bias is not None:
            output = output + self.bias.index_select(0, local_ids)
        return output

    def extra_repr(self) -> str:
        return (
            f"experts={self.local_num_experts}/{self.num_experts}, "
            f"in_features={self.input_size}, out_features={self.output_size}, "
            f"bias={self.bias is not None}, ep_size={self.ep_size}"
        )


class FusedExpertGateUpLinear(ExpertParallelLinear):
    def __init__(
        self,
        num_experts: int,
        input_size: int,
        intermediate_size: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_experts,
            input_size,
            2 * intermediate_size,
            **kwargs,
        )
        self.intermediate_size = intermediate_size

    def split_output(self, packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if packed.shape[-1] != 2 * self.intermediate_size:
            raise ValueError("packed expert output has an invalid width")
        gate, up = packed.split(self.intermediate_size, dim=-1)
        return gate, up
