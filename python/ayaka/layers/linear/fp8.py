"""Dense FP8 linear quantization methods and checkpoint scale loading.

* :meth:`create_weights` allocates FP8 storage once, with the checkpoint's
  scale tensors as separate parameters named as the checkpoint ships them
  (``weight_scale``, ``weight_scale_inv``, ``input_scale``). They start as NaN
  sentinels so a missing checkpoint tensor cannot silently become a fake scale.
* :meth:`process_weights_after_loading` rejects unloaded/non-finite/zero/negative
  scales and records the host-side facts the execution path needs (piecewise
  constant weight scale, block geometry). It runs once per layer.
* :meth:`apply` dispatches to :mod:`ayaka.kernel.triton.gemm.fp8_linear`, which
  composes the registered ``ayaka::*`` custom ops so reference mode, capability
  gating and SM emulation are inherited rather than re-implemented.

``_scale_loader`` is the packed-projection scale loader: a fused QKV or
gate/up checkpoint stores one scale per logical projection, and the loader
writes it into the part of the packed scale parameter owned by that projection
(plus the TP row/K-block slice for real layouts).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, ClassVar, Literal, Self, cast

import torch
from torch import nn

from ayaka.kernel.triton.gemm.fp8_linear import (
    fp8_block_linear,
    fp8_pertensor_linear,
    fp8_rowwise_linear,
)
from ayaka.layers.linear.methods import LinearMethodBase
from ayaka.layers.linear.weight_loading import WeightLoadError, set_weight_attrs
from ayaka.layers.quantization.base import (
    BaseQuantization,
    QuantizationCapabilities,
    QuantizationTarget,
    QuantizeMethodBase,
)
from ayaka.utils.import_utils import CapabilityError
from ayaka.utils.math_utils import div_ceil
from ayaka.utils.torch_utils import compute_torch_dtypes
from ayaka.weights.spec import FP8Recipe

FP8 = torch.float8_e4m3fn

#: K width of one block-FP8 scale in the DeepSeek-V3 checkpoint contract.
BLOCK128 = 128

ActivationScheme = Literal["dynamic", "static", "weight_only"]
_ACTIVATION_SCHEMES: tuple[str, ...] = ("dynamic", "static", "weight_only")

#: Parameter attributes carried through from the layer's packed-weight attrs.
_WEIGHT_ATTR_EXCLUSIONS = frozenset(
    {
        "output_partition_sizes",
        "input_size_per_partition",
        "input_size",
        "output_size",
        "params_dtype",
        "device",
    }
)


def _not_loaded(name: str) -> ValueError:
    return ValueError(f"{name} not loaded from the checkpoint")


def _require_loaded_scale(name: str, tensor: torch.Tensor) -> None:
    """Reject a scale that is absent (NaN sentinel), non-finite, or non-positive."""
    if tensor.is_meta:
        raise _not_loaded(name)
    values = tensor.detach().float()
    if bool(torch.isnan(values).any().item()):
        raise _not_loaded(name)
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError(f"{name} must be finite; found an infinite checkpoint value")
    if bool((values <= 0).any().item()):
        raise ValueError(f"{name} must be positive; found a zero or negative checkpoint value")


def _layout_names(layer: nn.Module) -> tuple[str, ...]:
    layout = getattr(layer, "layout", None)
    if layout is None:
        return ()
    names = getattr(layout, "names", None)
    if names:
        return tuple(str(name) for name in names)
    return tuple(
        part if isinstance(part, str) else str(part.name) for part in getattr(layout, "parts", ())
    )


def _layout_parts_are_names(layer: nn.Module) -> bool:
    layout = getattr(layer, "layout", None)
    parts = getattr(layout, "parts", ()) if layout is not None else ()
    return bool(parts) and all(isinstance(part, str) for part in parts)


def _locate_projection(layer: nn.Module, shard_id: object) -> tuple[int, int, int]:
    """Return ``(index, row_start, row_stop)`` of a projection in the packed output."""
    layout = getattr(layer, "layout", None)
    if layout is None:
        raise WeightLoadError("scale loading requires the layer's projection layout")
    names = _layout_names(layer)
    projection = layout[shard_id]  # type: ignore[index]
    if isinstance(projection, str):
        name = projection
    else:
        name = str(projection.name)
    if name not in names:
        raise WeightLoadError(f"unknown projection {name!r}; expected one of {names}")
    index = names.index(name)
    if isinstance(projection, str):
        return index, index, index + 1
    start = int(projection.local_offset)
    stop = int(projection.local_offset + projection.local_size)
    return index, start, stop


def _assign_scale(destination: torch.Tensor, loaded_weight: torch.Tensor, label: str) -> None:
    if tuple(loaded_weight.shape) == tuple(destination.shape):
        destination.copy_(loaded_weight)
        return
    if loaded_weight.numel() == destination.numel():
        destination.copy_(loaded_weight.reshape(destination.shape))
        return
    if loaded_weight.numel() == 1:
        destination.fill_(loaded_weight.reshape(()))
        return
    raise WeightLoadError(
        f"{label}: checkpoint shape {tuple(loaded_weight.shape)} does not fit "
        f"destination shape {tuple(destination.shape)}"
    )


def _scale_loader(
    layer: nn.Module,
    parameter: torch.Tensor,
    loaded_weight: torch.Tensor,
    shard_id: object = None,
) -> None:
    """Load one projection's scale into a packed scale parameter.

    A genuine per-tensor checkpoint stores a scalar and broadcasts it; a fused
    projection stores one scalar per logical part, loaded with the part's
    ``shard_id``. The packed parameter keeps its own row/K-block layout, so the
    loader narrows the destination to the projection's slice before copying.
    """
    if not isinstance(parameter, torch.Tensor):
        raise TypeError("parameter must be a torch.Tensor")
    if loaded_weight.is_meta:
        raise WeightLoadError("scale loading requires a materialized checkpoint tensor")
    data = parameter.data
    if shard_id is None:
        _assign_scale(data, loaded_weight, "scale")
        return
    if tuple(loaded_weight.shape) == tuple(data.shape):
        data.copy_(loaded_weight)
        return
    names = _layout_names(layer)
    if not names:
        _assign_scale(data, loaded_weight, "scale")
        return
    index, start, stop = _locate_projection(layer, shard_id)
    if _layout_parts_are_names(layer) and data.ndim >= 1 and data.shape[0] == len(names):
        destination = data[index]
    else:
        rows_per_scale = int(getattr(layer, "fp8_scale_block_y", 1) or 1)
        if rows_per_scale > 1:
            row_start = start // rows_per_scale
            row_stop = div_ceil(stop, rows_per_scale)
        else:
            row_start, row_stop = start, stop
        if data.ndim < 1:
            raise WeightLoadError("packed scale storage must have at least one dimension")
        destination = data[row_start:row_stop]
    _assign_scale(destination, loaded_weight, "scale")


class _FP8LinearMethodBase(LinearMethodBase):
    """Shared FP8 storage creation, scale validation and capability declaration."""

    weight_scale_name: ClassVar[str] = "weight_scale"

    def __init__(self, *, activation_bits: int) -> None:
        self._activation_bits = activation_bits

    def get_capabilities(self) -> QuantizationCapabilities:
        return QuantizationCapabilities(
            supported_devices=frozenset({"cpu", "cuda"}),
            supported_act_dtypes=tuple(compute_torch_dtypes()),
            supported_targets=frozenset({QuantizationTarget.LINEAR}),
            storage_format="fp8_e4m3",
            weight_bits=8,
            activation_bits=self._activation_bits,
            supports_dynamic_activation=True,
            supports_static_activation=True,
        )

    def create_weights(self, layer: nn.Module, *weight_args: Any, **attrs: Any) -> None:
        if weight_args:
            raise TypeError("FP8 linear weight dimensions must be passed by name")
        output_partition_sizes = attrs.get("output_partition_sizes")
        input_size = attrs.get("input_size_per_partition")
        if not output_partition_sizes:
            raise ValueError("output_partition_sizes is required for FP8 linear weights")
        if not isinstance(input_size, int) or input_size < 1:
            raise ValueError("input_size_per_partition must be a positive integer")
        device = attrs.get("device")
        weight_loader = attrs.get("weight_loader")
        total_output = sum(int(size) for size in output_partition_sizes)
        weight = nn.Parameter(
            torch.empty(total_output, input_size, dtype=FP8, device=device),
            requires_grad=False,
        )
        layer.register_parameter("weight", weight)
        weight_attrs = {
            key: value for key, value in attrs.items() if key not in _WEIGHT_ATTR_EXCLUSIONS
        }
        if weight_loader is not None:
            weight_attrs["weight_loader"] = weight_loader
        set_weight_attrs(weight, weight_attrs)
        layer.fp8_total_output = total_output  # type: ignore
        layer.fp8_input_size = input_size  # type: ignore
        layer.fp8_scale_block_y = 1  # type: ignore
        self.create_scale_parameters(layer, total_output, input_size, device)

    def create_scale_parameters(
        self,
        layer: nn.Module,
        total_output: int,
        input_size: int,
        device: torch.device | str | None,
    ) -> None:
        raise NotImplementedError

    def apply(self, layer: nn.Module, *args: Any, **kwargs: Any) -> torch.Tensor:
        raise NotImplementedError

    def _bias(self, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> torch.Tensor | None:
        bias = args[1] if len(args) > 1 else kwargs.get("bias")
        return cast(torch.Tensor | None, bias)


class FP8PerTensorLinearMethod(_FP8LinearMethodBase):
    """Per-tensor/per-output-row FP8 with static, dynamic or weight-only activation.

    ``activation_scheme="static"`` consumes the checkpoint's calibrated
    ``input_scale`` and runs W8A8 at ``M > 1``; ``"dynamic"`` computes a fresh
    per-token activation scale at ``M > 1``; ``"weight_only"`` never quantizes
    the activation (W8A16). Decode (``M == 1``) always uses the W8A16 split-K
    GEMV over the raw FP8 weight.
    """

    def __init__(self, activation_scheme: ActivationScheme = "dynamic") -> None:
        scheme = str(activation_scheme)
        if scheme not in _ACTIVATION_SCHEMES:
            raise ValueError(f"activation_scheme must be one of {_ACTIVATION_SCHEMES}")
        super().__init__(activation_bits=16 if scheme == "weight_only" else 8)
        self.activation_scheme: ActivationScheme = cast(ActivationScheme, scheme)

    def create_scale_parameters(
        self,
        layer: nn.Module,
        total_output: int,
        input_size: int,
        device: torch.device | str | None,
    ) -> None:
        del input_size
        weight_scale = nn.Parameter(
            torch.full((total_output,), float("nan"), dtype=torch.float32, device=device),
            requires_grad=False,
        )
        layer.register_parameter("weight_scale", weight_scale)
        set_weight_attrs(weight_scale, {"weight_loader": _scale_loader})
        if self.activation_scheme == "static":
            input_scale = nn.Parameter(
                torch.full((), float("nan"), dtype=torch.float32, device=device),
                requires_grad=False,
            )
            layer.register_parameter("input_scale", input_scale)
            set_weight_attrs(input_scale, {"weight_loader": _scale_loader})
        else:
            layer.register_parameter("input_scale", None)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        _require_loaded_scale("weight_scale", cast(torch.Tensor, layer.weight_scale))
        if self.activation_scheme == "static":
            _require_loaded_scale("input_scale", cast(torch.Tensor, layer.input_scale))
        weight_scale = cast(torch.Tensor, layer.weight_scale)
        layer._fp8_uniform_scale = bool((weight_scale == weight_scale[0]).all().item())  # type: ignore

    def apply(self, layer: nn.Module, *args: Any, **kwargs: Any) -> torch.Tensor:
        x = args[0]
        weight = cast(torch.Tensor, layer.weight)
        weight_scale = cast(torch.Tensor, layer.weight_scale)
        bias = self._bias(args, kwargs)
        if self.activation_scheme == "weight_only":
            return fp8_pertensor_linear(x, weight, weight_scale, bias)
        if self.activation_scheme == "static":
            return fp8_pertensor_linear(
                x,
                weight,
                weight_scale,
                bias,
                input_scale=cast(torch.Tensor, layer.input_scale),
                uniform_scale=bool(getattr(layer, "_fp8_uniform_scale", False)),
            )
        return fp8_rowwise_linear(x, weight, weight_scale, bias)


class FP8RowwiseLinearMethod(_FP8LinearMethodBase):
    """Dynamic per-token activation quantization with per-output-row weight scales."""

    def __init__(self) -> None:
        super().__init__(activation_bits=8)

    def create_scale_parameters(
        self,
        layer: nn.Module,
        total_output: int,
        input_size: int,
        device: torch.device | str | None,
    ) -> None:
        del input_size
        weight_scale = nn.Parameter(
            torch.full((total_output,), float("nan"), dtype=torch.float32, device=device),
            requires_grad=False,
        )
        layer.register_parameter("weight_scale", weight_scale)
        set_weight_attrs(weight_scale, {"weight_loader": _scale_loader})

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        _require_loaded_scale("weight_scale", cast(torch.Tensor, layer.weight_scale))

    def apply(self, layer: nn.Module, *args: Any, **kwargs: Any) -> torch.Tensor:
        return fp8_rowwise_linear(
            args[0],
            cast(torch.Tensor, layer.weight),
            cast(torch.Tensor, layer.weight_scale),
            self._bias(args, kwargs),
        )


class FP8BlockLinearMethod(_FP8LinearMethodBase):
    """DeepSeek-V3-style 128x128 block scales with dynamic per-token-group activations."""

    weight_scale_name = "weight_scale_inv"

    def __init__(self, scale_dtype: torch.dtype = torch.bfloat16) -> None:
        if scale_dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("scale_dtype must be bfloat16 or float32")
        super().__init__(activation_bits=8)
        self.scale_dtype = scale_dtype

    def create_scale_parameters(
        self,
        layer: nn.Module,
        total_output: int,
        input_size: int,
        device: torch.device | str | None,
    ) -> None:
        scale = nn.Parameter(
            torch.full(
                (div_ceil(total_output, BLOCK128), div_ceil(input_size, BLOCK128)),
                float("nan"),
                dtype=self.scale_dtype,
                device=device,
            ),
            requires_grad=False,
        )
        layer.register_parameter("weight_scale_inv", scale)
        set_weight_attrs(scale, {"weight_loader": _scale_loader})
        layer.fp8_scale_block_y = BLOCK128  # type: ignore

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        _require_loaded_scale("weight_scale_inv", cast(torch.Tensor, layer.weight_scale_inv))

    def apply(self, layer: nn.Module, *args: Any, **kwargs: Any) -> torch.Tensor:
        return fp8_block_linear(
            args[0],
            cast(torch.Tensor, layer.weight),
            cast(torch.Tensor, layer.weight_scale_inv),
            self._bias(args, kwargs),
        )


class FP8W8A16LinearMethod(_FP8LinearMethodBase):
    """Weight-only E4M3 with FP32/BF16/E8M0 scales, per-channel or blocked.

    The activation is never quantized: ``M == 1`` uses the split-K GEMV and
    ``M > 1`` the tiled W8A16 GEMM over the raw FP8 weight.
    """

    def __init__(
        self,
        scale_dtype: torch.dtype = torch.float32,
        weight_block: tuple[int, int] | None = None,
    ) -> None:
        if scale_dtype not in (torch.float32, torch.bfloat16, torch.uint8):
            raise ValueError("scale_dtype must be float32, bfloat16, or uint8 (UE8M0)")
        if weight_block is not None:
            block_y, block_x = weight_block
            if block_y < 1 or block_x < 1:
                raise ValueError("weight_block sizes must be positive")
        super().__init__(activation_bits=16)
        self.scale_dtype = scale_dtype
        self.weight_block = weight_block

    def _scale_geometry(self, input_size: int) -> tuple[int, int]:
        if self.weight_block is None:
            return 1, input_size
        return self.weight_block

    def create_scale_parameters(
        self,
        layer: nn.Module,
        total_output: int,
        input_size: int,
        device: torch.device | str | None,
    ) -> None:
        block_y, block_x = self._scale_geometry(input_size)
        shape = (div_ceil(total_output, block_y), div_ceil(input_size, block_x))
        if self.scale_dtype is torch.uint8:
            sentinel: float | int = 0xFF
        else:
            sentinel = float("nan")
        scale = nn.Parameter(
            torch.full(shape, sentinel, dtype=self.scale_dtype, device=device),
            requires_grad=False,
        )
        layer.register_parameter("weight_scale", scale)
        set_weight_attrs(scale, {"weight_loader": _scale_loader})
        layer.fp8_scale_block_y = block_y  # type: ignore
        layer.fp8_weight_block = block_y, block_x  # type: ignore

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        scale = cast(torch.Tensor, layer.weight_scale)
        if self.scale_dtype is torch.uint8:
            if scale.is_meta or bool((scale == 0xFF).any().item()):
                raise _not_loaded("weight_scale")
        else:
            _require_loaded_scale("weight_scale", scale)

    def apply(self, layer: nn.Module, *args: Any, **kwargs: Any) -> torch.Tensor:
        from ayaka.kernel.triton.gemm.fp8_gemv import fp8_weight_only_gemv
        from ayaka.kernel.triton.gemm.fp8_w8a16_gemm import fp8_weight_only_gemm

        x = args[0]
        inner = int(x.shape[-1])
        block_y, block_x = self._scale_geometry(inner)
        scale_dtype = {
            torch.uint8: "e8m0",
            torch.float32: "float32",
            torch.bfloat16: "bfloat16",
        }[self.scale_dtype]
        weight = cast(torch.Tensor, layer.weight)
        scale = cast(torch.Tensor, layer.weight_scale)
        scale_row_stride = div_ceil(inner, block_x)
        if x.numel() == inner:
            result = fp8_weight_only_gemv(
                x.reshape(inner),
                weight,
                scale,
                scale_row_stride=scale_row_stride,
                block_size_y=block_y,
                block_size_x=block_x,
                scale_dtype=scale_dtype,
            )
        else:
            result = fp8_weight_only_gemm(
                x,
                weight,
                scale,
                scale_row_stride=scale_row_stride,
                block_size_y=block_y,
                block_size_x=block_x,
                scale_dtype=scale_dtype,
            )
        bias = self._bias(args, kwargs)
        if bias is not None:
            result = result + bias
        return result


class FP8Mxfp8LinearMethod(_FP8LinearMethodBase):
    """MXFP8: E4M3 activations and weights with UE8M0 1x32 scales."""

    MX_GROUP = 32

    def __init__(self) -> None:
        super().__init__(activation_bits=8)

    def create_scale_parameters(
        self,
        layer: nn.Module,
        total_output: int,
        input_size: int,
        device: torch.device | str | None,
    ) -> None:
        if input_size % self.MX_GROUP:
            raise ValueError("MXFP8 requires the weight K to be divisible by 32")
        scale = nn.Parameter(
            torch.full(
                (total_output, input_size // self.MX_GROUP),
                0xFF,
                dtype=torch.uint8,
                device=device,
            ),
            requires_grad=False,
        )
        layer.register_parameter("weight_scale", scale)
        set_weight_attrs(scale, {"weight_loader": _scale_loader})

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        scale = cast(torch.Tensor, layer.weight_scale)
        if scale.is_meta or bool((scale == 0xFF).any().item()):
            # 0xFF is the sentinel *and* the E8M0 NaN code, so it can never be a
            # valid loaded scale.
            raise _not_loaded("weight_scale")

    def apply(self, layer: nn.Module, *args: Any, **kwargs: Any) -> torch.Tensor:
        from ayaka.kernel.triton.gemm.fp8_mxfp8_gemm import mxfp8_scaled_mm
        from ayaka.kernel.triton.quant.fp8_quant import per_token_group_quant_mxfp8

        x = args[0]
        lead = tuple(x.shape[:-1])
        inner = int(x.shape[-1])
        if inner % self.MX_GROUP:
            raise ValueError("MXFP8 requires the activation K to be divisible by 32")
        x2d = x.reshape(-1, inner)
        if not x2d.is_contiguous():
            x2d = x2d.contiguous()
        weight = cast(torch.Tensor, layer.weight)
        quantized, activation_scale = per_token_group_quant_mxfp8(x2d, self.MX_GROUP)
        result = mxfp8_scaled_mm(
            quantized,
            activation_scale,
            weight,
            cast(torch.Tensor, layer.weight_scale),
            self._bias(args, kwargs),
            out=torch.empty((x2d.shape[0], int(weight.shape[0])), dtype=x.dtype, device=x.device),
        )
        return result.reshape(*lead, result.shape[-1])


class FP8Config(BaseQuantization):
    """Select a dense FP8 linear method from a validated recipe descriptor.

    The checkpoint adapter (:mod:`ayaka.model_loader.fp8`) builds this from a
    ``quantization_config``; ``get_quant_method`` returns a fresh, stateless
    method per layer so no per-layer state is shared across layers.
    """

    def __init__(
        self,
        recipe_name: str,
        *,
        scale_dtype: torch.dtype = torch.bfloat16,
        weight_block: tuple[int, int] | None = None,
        ignored_layers: tuple[str, ...] = (),
        method_factory: Callable[[], QuantizeMethodBase] | None = None,
    ) -> None:
        self.recipe_name = str(recipe_name)
        self.scale_dtype = scale_dtype
        self.weight_block = weight_block
        self.ignored_layers = tuple(ignored_layers)
        self._method_factory = method_factory

    @classmethod
    def get_name(cls) -> str:
        return "fp8"

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> Self:
        from ayaka.model_loader.fp8 import parse_fp8_config

        parsed = parse_fp8_config(config)
        return cls(
            parsed.recipe.recipe.value,
            scale_dtype=parsed.scale_dtype,
            weight_block=parsed.recipe.weight_block,
            ignored_layers=parsed.ignored_layers,
            method_factory=lambda: method_from_recipe(parsed),
        )

    def _make_method(self) -> QuantizeMethodBase:
        if self._method_factory is not None:
            return self._method_factory()
        recipe = FP8Recipe(self.recipe_name)
        if recipe is FP8Recipe.W8A8_STATIC_PER_TENSOR:
            return FP8PerTensorLinearMethod(activation_scheme="static")
        if recipe is FP8Recipe.W8A8_DYNAMIC_PER_TENSOR:
            return FP8PerTensorLinearMethod(activation_scheme="dynamic")
        if recipe is FP8Recipe.W8A8_ROWWISE:
            return FP8RowwiseLinearMethod()
        if recipe is FP8Recipe.W8A8_BLOCK128:
            return FP8BlockLinearMethod(scale_dtype=self.scale_dtype)
        if recipe is FP8Recipe.W8A16:
            return FP8W8A16LinearMethod(
                scale_dtype=self.scale_dtype, weight_block=self.weight_block
            )
        if recipe is FP8Recipe.MXFP8:
            return FP8Mxfp8LinearMethod()
        raise CapabilityError(
            "quantization.fp8_recipe",
            detail=f"recipe {recipe.value!r} has no dense linear method",
            remedy="use the KV-cache quantization owner for cache recipes",
        )

    def get_capabilities(self) -> QuantizationCapabilities:
        return cast(LinearMethodBase, self._make_method()).get_capabilities()

    def get_quant_method(self, layer: nn.Module, prefix: str) -> QuantizeMethodBase | None:
        del layer
        if any(
            prefix == ignored or prefix.startswith(f"{ignored}.") for ignored in self.ignored_layers
        ):
            return None
        return self._make_method()


def method_from_recipe(parsed: Any) -> QuantizeMethodBase:
    """Build the dense FP8 method for a parsed checkpoint recipe.

    ``parsed`` is a :class:`~ayaka.model_loader.fp8.ParsedFP8Config`; it is
    typed loosely here to keep the layer module free of a model-loader import.
    """
    recipe = parsed.recipe.recipe
    if recipe is FP8Recipe.W8A8_STATIC_PER_TENSOR:
        return FP8PerTensorLinearMethod(activation_scheme="static")
    if recipe is FP8Recipe.W8A8_DYNAMIC_PER_TENSOR:
        return FP8PerTensorLinearMethod(activation_scheme="dynamic")
    if recipe is FP8Recipe.W8A8_ROWWISE:
        return FP8RowwiseLinearMethod()
    if recipe is FP8Recipe.W8A8_BLOCK128:
        return FP8BlockLinearMethod(scale_dtype=parsed.scale_dtype)
    if recipe is FP8Recipe.W8A16:
        return FP8W8A16LinearMethod(
            scale_dtype=parsed.scale_dtype, weight_block=parsed.recipe.weight_block
        )
    if recipe is FP8Recipe.MXFP8:
        return FP8Mxfp8LinearMethod()
    raise CapabilityError(
        "quantization.fp8_recipe",
        detail=f"recipe {recipe.value!r} has no dense linear method",
        remedy="use the KV-cache quantization owner for cache recipes",
    )


__all__ = [
    "BLOCK128",
    "FP8BlockLinearMethod",
    "FP8Config",
    "FP8Mxfp8LinearMethod",
    "FP8PerTensorLinearMethod",
    "FP8RowwiseLinearMethod",
    "FP8W8A16LinearMethod",
    "_scale_loader",
    "method_from_recipe",
]
