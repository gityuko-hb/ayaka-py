from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import nn

from ayaka.distributed.parallel import ParallelContext, divide
from ayaka.layers._common import check_dtype
from ayaka.layers.base import BaseLayer
from ayaka.layers.linear.layout import ProjectionKind, ProjectionLayout, ProjectionSpec
from ayaka.layers.linear.methods import (
    LinearMethodBase,
    UnquantizedLinearMethod,
    resolve_quantization_config,
)
from ayaka.layers.linear.parallel import prepare_parallel_runtime
from ayaka.layers.linear.weight_loading import (
    WeightLoadError,
    load_packed_output_parameter,
    load_row_parameter,
    set_weight_attrs,
)
from ayaka.layers.quantization.base import BaseQuantization, QuantizeMethodBase
from ayaka.utils.validation import require_int


class LinearBase(BaseLayer):
    """Inference linear borrowing runtime groups, placement and quantization policy."""

    weight: nn.Parameter
    bias: nn.Parameter | None
    input_size_per_partition: int
    weight_loader: Callable[..., None]
    weight_loader_v2: Callable[..., None]

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: BaseQuantization | LinearMethodBase | str | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
        parallel_context: ParallelContext | None = None,
        device: torch.device | str | None = None,
        activation_dtype: torch.dtype | None = None,
        **runtime: Any,
    ) -> None:
        require_int(input_size, "input_size", minimum=1)
        require_int(output_size, "output_size", minimum=1)
        if skip_bias_add and bias and not return_bias:
            raise ValueError("skip_bias_add with bias requires return_bias=True")
        device, context, runtime = prepare_parallel_runtime(
            device,
            parallel_context,
            runtime,
            disabled=disable_tp,
        )
        super().__init__(
            prefix=prefix, quant_config=resolve_quantization_config(quant_config), **runtime
        )
        self.input_size = input_size
        self.output_size = output_size
        self.has_bias = bias
        self.skip_bias_add = skip_bias_add
        self.params_dtype = params_dtype if params_dtype is not None else torch.get_default_dtype()
        self.activation_dtype = (
            activation_dtype if activation_dtype is not None else self.params_dtype
        )
        self.return_bias = return_bias
        self.disable_tp = disable_tp
        self.parallel_context = context
        self.device = device
        if self.quant_config is None:
            check_dtype(self.params_dtype)
            if self.activation_dtype != self.params_dtype:
                raise ValueError("dense linear requires matching parameter and activation dtypes")
        if device.type != "meta":
            self.device = self.runtime_context(device, self.activation_dtype).device

    def select_weight_loader(self, method: QuantizeMethodBase) -> Callable[..., None]:
        """Choose the parameter loader entry point declared by the method object."""
        if getattr(method, "uses_weight_loader_v2", False):
            return self.weight_loader_v2
        return self.weight_loader

    def _create_linear_weights(self, **attrs: Any) -> None:
        attrs["device"] = self.device
        if self.quant_config is not None:
            self.create_weights(activation_dtype=self.activation_dtype, **attrs)
        else:
            self.quant_method = UnquantizedLinearMethod()
            attrs.setdefault("weight_loader", self.select_weight_loader(self.quant_method))
            self.quant_method.create_weights(self, **attrs)

    def _apply_linear(self, x: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        if x.ndim < 1 or x.shape[-1] != self.input_size_per_partition:
            raise ValueError(f"input must end in {self.input_size_per_partition} features")
        if self.quant_config is not None:
            return self.apply_quantization(x, bias)
        assert self.quant_method is not None
        return self.quant_method.apply(self, x, bias)

    def process_weights_after_loading(self) -> None:
        if self.quant_config is not None:
            super().process_weights_after_loading()
            return
        if any(t.is_meta for t in (*self.parameters(), *self.buffers())):
            raise RuntimeError("materialize meta weights before finalizing linear")
        self.runtime_context(self.weight.device, self.weight.dtype)

    def _return(
        self,
        output: torch.Tensor,
        deferred_bias: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        return (output, deferred_bias) if self.return_bias else output


class PackedColumnParallelLinear(LinearBase):
    """One GEMM whose output contains independently sharded logical matrices."""

    def __init__(
        self,
        input_size: int,
        specs: list[ProjectionSpec] | tuple[ProjectionSpec, ...],
        *,
        bias: bool = False,
        gather_output: bool = False,
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
        super().__init__(
            input_size=input_size,
            output_size=sum(spec.output_size for spec in specs),
            bias=bias,
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

        self.layout = ProjectionLayout(specs, self.parallel_context)
        self.output_partition_sizes = [part.local_size for part in self.layout.parts]
        self.input_size_per_partition = input_size
        self.output_size_per_partition = self.layout.local_output_size
        self.gather_output = bool(gather_output)
        if self.gather_output:
            for part in self.layout.parts:
                if (
                    part.spec.kind in (ProjectionKind.HEAD, ProjectionKind.KV_HEAD)
                    and part.num_head_replicas > 1
                ):
                    raise ValueError("gather_output is ambiguous for replicated head projections")

        self._create_linear_weights(
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            output_dim=0,
            device=self.device,
        )
        if bias:
            bias_parameter = nn.Parameter(
                torch.empty(
                    self.layout.local_output_size,
                    dtype=self.params_dtype,
                    device=self.device,
                ),
                requires_grad=False,
            )
            self.register_parameter("bias", bias_parameter)
            set_weight_attrs(
                bias_parameter,
                {"output_dim": 0, "weight_loader": self.weight_loader},
            )
        else:
            self.register_parameter("bias", None)

    def validate_shard_id(self, shard_id: Any) -> bool:
        if shard_id is None:
            return True
        ids = shard_id if isinstance(shard_id, tuple) else (shard_id,)
        for value in ids:
            self.layout[value]
        return True

    def weight_loader(
        self,
        parameter: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str | int | tuple[int, ...] | None = None,
    ) -> None:
        self.validate_shard_id(loaded_shard_id)
        load_packed_output_parameter(
            parameter,
            loaded_weight,
            self.layout,
            loaded_shard_id,
        )

    weight_loader_v2 = weight_loader

    def _gather_packed(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.tp_size == 1:
            return tensor
        pieces: list[torch.Tensor] = []
        local = self.layout.split_output(tensor)
        for part in self.layout.parts:
            piece = local[part.name]
            if part.spec.kind is ProjectionKind.REPLICATED:
                pieces.append(piece)
            else:
                pieces.append(self.parallel_context.all_gather_last_dim(piece))
        return torch.cat(pieces, dim=-1)

    def split_output(self, output: torch.Tensor) -> dict[str, torch.Tensor]:
        if output.shape[-1] == self.layout.local_output_size:
            return self.layout.split_output(output)
        if output.shape[-1] == self.layout.global_output_size:
            return {
                part.name: output.narrow(-1, part.global_offset, part.global_size)
                for part in self.layout.parts
            }
        raise ValueError(
            f"output width {output.shape[-1]} does not match local "
            f"{self.layout.local_output_size} or global {self.layout.global_output_size}"
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        bias = self.bias if not self.skip_bias_add else None
        output = self._apply_linear(x, bias)
        if self.gather_output:
            output = self._gather_packed(output)
        deferred_bias = self.bias if self.skip_bias_add else None
        if self.gather_output and deferred_bias is not None:
            deferred_bias = self._gather_packed(deferred_bias)
        return self._return(output, deferred_bias)

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[str]:
        for name, loaded_weight in weights:
            parameter = getattr(self, name, None)
            if parameter is None and name == "bias":
                continue
            if not isinstance(parameter, nn.Parameter):
                raise KeyError(f"{type(self).__name__} has no parameter {name!r}")
            shard_id = getattr(loaded_weight, "shard_id", None)
            loader = getattr(parameter, "weight_loader", self.weight_loader)
            loader(parameter, loaded_weight, shard_id)
            yield name

    def extra_repr(self) -> str:
        return (
            f"in_features={self.input_size}, "
            f"local_out_features={self.layout.local_output_size}, "
            f"global_out_features={self.layout.global_output_size}, "
            f"bias={self.bias is not None}, tp_size={self.tp_size}, "
            f"gather_output={self.gather_output}, layout={self.layout.extra_repr()}"
        )


class ReplicatedLinear(PackedColumnParallelLinear):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: BaseQuantization | LinearMethodBase | str | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
        parallel_context: ParallelContext | None = None,
        device: torch.device | str | None = None,
        **runtime: Any,
    ) -> None:
        super().__init__(
            input_size,
            [ProjectionSpec("output", output_size, ProjectionKind.REPLICATED)],
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


class ColumnParallelLinear(PackedColumnParallelLinear):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: BaseQuantization | LinearMethodBase | str | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
        parallel_context: ParallelContext | None = None,
        device: torch.device | str | None = None,
        **runtime: Any,
    ) -> None:
        super().__init__(
            input_size,
            [ProjectionSpec("output", output_size, ProjectionKind.COLUMN)],
            bias=bias,
            gather_output=gather_output,
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


class HeadParallelLinear(PackedColumnParallelLinear):
    """Column-parallel linear that preserves complete attention heads."""

    def __init__(
        self,
        input_size: int,
        num_heads: int,
        head_dim: int,
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: BaseQuantization | LinearMethodBase | str | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
        parallel_context: ParallelContext | None = None,
        device: torch.device | str | None = None,
        **runtime: Any,
    ) -> None:
        super().__init__(
            input_size,
            [ProjectionSpec.heads("output", num_heads=num_heads, head_dim=head_dim)],
            bias=bias,
            gather_output=gather_output,
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
        part = self.layout["output"]
        self.total_num_heads = num_heads
        self.head_dim = head_dim
        self.num_heads = part.local_num_heads or 0
        self.num_head_replicas = part.num_head_replicas


class MergedColumnParallelLinear(PackedColumnParallelLinear):
    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: BaseQuantization | LinearMethodBase | str | None = None,
        prefix: str = "",
        *,
        projection_names: list[str] | None = None,
        return_bias: bool = True,
        disable_tp: bool = False,
        parallel_context: ParallelContext | None = None,
        device: torch.device | str | None = None,
        **runtime: Any,
    ) -> None:
        if projection_names is None:
            projection_names = [str(index) for index in range(len(output_sizes))]
        if len(projection_names) != len(output_sizes):
            raise ValueError("projection_names and output_sizes must have equal length")
        self.output_sizes = list(output_sizes)
        self.projection_names = list(projection_names)
        specs = [
            ProjectionSpec(name, size, ProjectionKind.COLUMN)
            for name, size in zip(projection_names, output_sizes, strict=True)
        ]
        super().__init__(
            input_size,
            specs,
            bias=bias,
            gather_output=gather_output,
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


class FusedGateUpLinear(MergedColumnParallelLinear):
    """Packed gate/up projection used by SwiGLU, GeGLU and SiTU-GLU MLPs."""

    def __init__(
        self,
        input_size: int,
        intermediate_size: int,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("bias", False)
        super().__init__(
            input_size=input_size,
            output_sizes=[intermediate_size, intermediate_size],
            projection_names=["gate", "up"],
            **kwargs,
        )
        self.intermediate_size = intermediate_size


class RowParallelLinear(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        input_is_parallel: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        reduce_results: bool = True,
        quant_config: BaseQuantization | LinearMethodBase | str | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
        parallel_context: ParallelContext | None = None,
        device: torch.device | str | None = None,
        **runtime: Any,
    ) -> None:
        self.input_is_parallel = bool(input_is_parallel)
        self.reduce_results = bool(reduce_results)
        if not reduce_results and bias and not skip_bias_add:
            raise ValueError(
                "bias must be deferred when reduce_results=False to avoid adding it per rank"
            )

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            bias=bias,
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
        self.input_size_per_partition = divide(input_size, self.tp_size, name="input_size")
        self.output_size_per_partition = output_size
        self.output_partition_sizes = [output_size]
        self._create_linear_weights(
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=[output_size],
            input_size=input_size,
            output_size=output_size,
            params_dtype=self.params_dtype,
            input_dim=1,
            device=self.device,
        )
        if bias:
            bias_parameter = nn.Parameter(
                torch.empty(output_size, dtype=self.params_dtype, device=self.device),
                requires_grad=False,
            )
            self.register_parameter("bias", bias_parameter)
            set_weight_attrs(bias_parameter, {"weight_loader": self.weight_loader})
        else:
            self.register_parameter("bias", None)

    def weight_loader(
        self,
        parameter: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: object | None = None,
    ) -> None:
        if loaded_shard_id is not None:
            raise WeightLoadError("row weights do not accept logical projection IDs")
        if parameter.is_meta or loaded_weight.is_meta:
            raise WeightLoadError("materialize weights before copying")
        if getattr(parameter, "input_dim", None) is None:
            if parameter.shape != loaded_weight.shape:
                raise WeightLoadError(
                    f"bias shape {tuple(parameter.shape)} does not match "
                    f"checkpoint {tuple(loaded_weight.shape)}"
                )
            parameter.data.copy_(loaded_weight.to(parameter))
            return
        load_row_parameter(
            parameter,
            loaded_weight,
            rank=self.tp_rank,
            world_size=self.tp_size,
        )

    weight_loader_v2 = weight_loader

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        input_parallel = x if self.input_is_parallel else self.parallel_context.split_last_dim(x)
        gemm_bias = None
        if not self.skip_bias_add and self.bias is not None:
            if self.tp_size == 1 or self.tp_rank == 0:
                gemm_bias = self.bias
        output = self._apply_linear(input_parallel, gemm_bias)
        if self.reduce_results and self.tp_size > 1:
            output = self.parallel_context.all_reduce(output)
        deferred_bias = self.bias if self.skip_bias_add else None
        return self._return(output, deferred_bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.input_size_per_partition}, "
            f"output_features={self.output_size}, bias={self.bias is not None}, "
            f"tp_size={self.tp_size}, reduce_results={self.reduce_results}"
        )
