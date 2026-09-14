"""Linear storage methods using Ayaka's quantization and public kernel contracts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, ClassVar, Self, cast

import torch
import torch.nn.functional as F
from torch import nn

from ayaka.kernel.ops import custom_op
from ayaka.layers.base import BaseLayer
from ayaka.layers.quantization.base import (
    BaseQuantization,
    QuantizationCapabilities,
    QuantizeMethodBase,
)
from ayaka.utils.import_utils import resolve_qualname
from ayaka.utils.validation import require_int

from .weight_loading import set_weight_attrs


class LinearMethodBase(QuantizeMethodBase):
    """Linear specialization of the shared storage/execution contract."""

    uses_weight_loader_v2: ClassVar[bool] = False
    """Declare that parameters created by this method use the v2 loader entry point."""

    def get_capabilities(self) -> QuantizationCapabilities:
        """Dense callable defaults; packed methods should declare their capabilities."""
        return QuantizationCapabilities(
            supported_devices=frozenset({"cpu", "cuda"}),
            supported_act_dtypes=(torch.float16, torch.bfloat16, torch.float32),
        )


class UnquantizedLinearMethod(LinearMethodBase):
    uses_weight_loader_v2 = True

    def create_weights(self, layer: nn.Module, *weight_args: Any, **attrs: Any) -> None:
        if weight_args:
            raise TypeError("linear weight dimensions must be passed by name")
        weight = nn.Parameter(
            torch.empty(
                sum(attrs["output_partition_sizes"]),
                attrs["input_size_per_partition"],
                dtype=attrs["params_dtype"],
                device=attrs["device"],
            ),
            requires_grad=False,
        )
        layer.register_parameter("weight", weight)
        set_weight_attrs(
            weight,
            {
                key: value
                for key, value in attrs.items()
                if key
                not in {
                    "output_partition_sizes",
                    "input_size_per_partition",
                    "input_size",
                    "output_size",
                    "params_dtype",
                    "device",
                }
            },
        )

    def apply(self, layer: nn.Module, *args: Any, **kwargs: Any) -> torch.Tensor:
        x = args[0]
        bias = args[1] if len(args) > 1 else kwargs.get("bias")
        return BaseLayer.run_kernel(F.linear, x, cast(torch.Tensor, layer.weight), bias)


class OpaqueLinearMethod(UnquantizedLinearMethod):
    """Dense storage with an external public callable; finalize before execution."""

    def __init__(self, op: Callable[..., torch.Tensor] | str) -> None:
        self.op = resolve_qualname(op) if isinstance(op, str) else op
        if not callable(self.op):
            raise TypeError("op must be callable or resolve to a callable")

    def apply(self, layer: nn.Module, *args: Any, **kwargs: Any) -> torch.Tensor:
        x = args[0]
        bias = args[1] if len(args) > 1 else kwargs.get("bias")
        return BaseLayer.run_kernel(self.op, x, cast(torch.Tensor, layer.weight), bias)


def register_linear_kernel(
    fn: Callable[..., torch.Tensor] | None = None,
    *,
    name: str | None = None,
    namespace: str = "ayaka",
    reference: Callable[..., torch.Tensor] | None = None,
    dispatch_key: str = "CUDA",
    output_dtype: torch.dtype | None = None,
    output_size: int | None = None,
    fake_impl: Callable[..., torch.Tensor] | None = None,
) -> Any:
    """Register a functional GEMM through the existing custom-op dispatcher.

    The default fake implementation assumes dense [out, in] storage. A kernel
    packing the output dimension must supply its logical output_size or fake_impl.
    Mutating/out-buffer kernels use ayaka.kernel.ops.custom_op directly.
    """
    if output_size is not None:
        require_int(output_size, "output_size", minimum=1)
    if fake_impl is not None and (output_size is not None or output_dtype is not None):
        raise ValueError("custom fake_impl owns output shape and dtype")

    def decorator(kernel: Callable[..., torch.Tensor]) -> Any:
        def dense_fake(
            x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
        ) -> torch.Tensor:
            return torch.empty(
                (*x.shape[:-1], output_size if output_size is not None else weight.shape[0]),
                dtype=output_dtype or x.dtype,
                device=x.device,
            )

        return custom_op(
            kernel,
            name=name or kernel.__name__,
            namespace=namespace,
            fake_impl=fake_impl or dense_fake,
            reference=reference,
            dispatch_key=dispatch_key,
        )

    return decorator(fn) if fn is not None else decorator


class _MethodConfig(BaseQuantization):
    def __init__(self, method: LinearMethodBase) -> None:
        self.method = method

    @classmethod
    def get_name(cls) -> str:
        return "linear_method"

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> Self:
        raise TypeError("construct the method explicitly")

    def get_capabilities(self) -> QuantizationCapabilities:
        return self.method.get_capabilities()

    def get_quant_method(self, layer: nn.Module, prefix: str) -> QuantizeMethodBase:
        return self.method


#! TODO: INIT in configs/quant.py comm soon
def resolve_quantization_config(
    config: BaseQuantization | LinearMethodBase | str | None,
) -> BaseQuantization | None:
    """Adapt explicit methods to standard capability selection and finalization.

    Stateful instances must not be shared across layers. Use a BaseQuantization
    factory returning a fresh method per layer when per-layer state is needed.
    """
    if config is None or isinstance(config, BaseQuantization):
        return config
    if isinstance(config, str):
        resolved = resolve_qualname(config)
        config = resolved() if isinstance(resolved, type) else resolved
    if not isinstance(config, LinearMethodBase):
        raise TypeError("quant_config must be BaseQuantization or LinearMethodBase")
    return _MethodConfig(config)
