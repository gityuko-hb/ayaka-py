from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar, Self

import torch
from torch import nn

from ayaka.utils.import_utils import CapabilityError, has_module

if TYPE_CHECKING:
    from ayaka.device.context import DeviceContext
    from ayaka.distributed.device import DeviceGroup


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be >= 1")


def _group_size(group: DeviceGroup | None) -> int:
    if group is None:
        return 1
    _positive_int(group.size, f"group {group.name!r} size")
    if len(set(group.ranks)) != len(group.ranks):
        raise ValueError(f"group {group.name!r} contains duplicate ranks")
    return group.size


class QuantizationTarget(StrEnum):
    """Runtime use site for a quantized method.

    A backend can support one target without being valid for the others. For
    example, a packed weight-only GEMM may support LINEAR and LM_HEAD but not
    EMBEDDING, while an FP8 cache implementation only supports KV_CACHE.
    """

    LINEAR = "linear"
    EMBEDDING = "embedding"
    LM_HEAD = "lm_head"
    EXPERT = "expert"
    KV_CACHE = "kv_cache"


@dataclass(frozen=True, slots=True)
class QuantizationContext:
    """Facts required to validate a quantization backend at runtime."""

    target: QuantizationTarget = QuantizationTarget.LINEAR
    device_type: str = "cpu"
    activation_dtype: torch.dtype | None = None
    compute_capability: int | None = None
    tensor_parallel_size: int = 1
    expert_parallel_size: int = 1
    device_index: int | None = None

    def __post_init__(self) -> None:
        _positive_int(self.tensor_parallel_size, "tensor_parallel_size")
        _positive_int(self.expert_parallel_size, "expert_parallel_size")
        object.__setattr__(self, "target", QuantizationTarget(self.target))
        device = (
            torch.device(self.device_type.lower())
            if self.device_index is None
            else torch.device(self.device_type.lower(), self.device_index)
        )
        object.__setattr__(self, "device_type", device.type)
        object.__setattr__(self, "device_index", device.index)
        if self.activation_dtype is not None and not isinstance(self.activation_dtype, torch.dtype):
            raise TypeError("activation_dtype must be a torch.dtype")
        if self.compute_capability is not None:
            _positive_int(self.compute_capability, "compute_capability")

    @property
    def device(self) -> torch.device:
        """Execution placement; unlike ``device_type``, preserves the CUDA index."""
        if self.device_index is None:
            return torch.device(self.device_type)
        return torch.device(self.device_type, self.device_index)

    @classmethod
    def from_device(
        cls,
        device: torch.device | str,
        *,
        target: QuantizationTarget = QuantizationTarget.LINEAR,
        activation_dtype: torch.dtype | None = None,
        compute_capability: int | None = None,
        tensor_parallel_size: int = 1,
        expert_parallel_size: int = 1,
    ) -> QuantizationContext:
        resolved = torch.device(device)
        if resolved.type == "cpu":
            resolved = torch.device("cpu")
        if resolved.type == "cuda" and resolved.index is None and torch.cuda.is_available():
            resolved = torch.device("cuda", torch.cuda.current_device())
        capability = compute_capability
        if resolved.type == "cuda" and capability is None and torch.cuda.is_available():
            index = resolved.index
            major, minor = torch.cuda.get_device_capability(index)
            capability = major * 10 + minor
        return cls(
            target=target,
            device_type=resolved.type,
            activation_dtype=activation_dtype,
            compute_capability=capability,
            tensor_parallel_size=tensor_parallel_size,
            expert_parallel_size=expert_parallel_size,
            device_index=resolved.index,
        )

    @classmethod
    def from_runtime(
        cls,
        device: torch.device | str,
        *,
        target: QuantizationTarget,
        activation_dtype: torch.dtype,
        device_context: DeviceContext | None = None,
        tp_group: DeviceGroup | None = None,
        ep_group: DeviceGroup | None = None,
    ) -> QuantizationContext:
        """Validate borrowed runtime facts without binding a device or creating groups.

        A null backend is a host simulation, even though its DeviceRef says CUDA.
        It must never supply simulated SM capabilities for real CUDA execution.
        Group ranks are group-local indices, not launcher/global ranks.
        """
        resolved = torch.device(device)
        capability = None
        if device_context is not None:
            if device_context.is_closed:
                raise RuntimeError("device context is closed")
            backend = device_context.backend  # Also checks use after fork.
            if backend.name == "null":
                if resolved.type != "cpu":
                    raise CapabilityError(
                        "quantization.device_context",
                        detail="a null device context only supports host simulation",
                        remedy="supply a real device context for accelerator execution",
                    )
            else:
                expected = device_context.ref
                if resolved.type != expected.kind.value or (
                    resolved.index is not None and resolved.index != expected.index
                ):
                    raise ValueError(f"device {resolved} does not match context {expected}")
                resolved = torch.device(expected.kind.value, expected.index)
                major, minor = device_context.capability.compute_capability
                capability = major * 10 + minor or None
        if resolved.type == "cuda" and resolved.index is None:
            resolved = torch.device("cuda", torch.cuda.current_device())
        tp_size, ep_size = _group_size(tp_group), _group_size(ep_group)
        for group in (tp_group, ep_group):
            if group is None:
                continue
            local = group.devices[group.local_rank]
            if local.kind.value != resolved.type or (
                resolved.type != "cpu" and local.index != resolved.index
            ):
                raise ValueError(
                    f"group {group.name!r} local device {local} does not match {resolved}"
                )
        return cls.from_device(
            resolved,
            target=target,
            activation_dtype=activation_dtype,
            compute_capability=capability,
            tensor_parallel_size=tp_size,
            expert_parallel_size=ep_size,
        )


@dataclass(frozen=True, slots=True)
class QuantizationCapabilities:
    """Static capability declaration for a quantization configuration."""

    supported_act_dtypes: tuple[torch.dtype, ...] = ()
    supported_devices: frozenset[str] = frozenset({"cpu", "cuda"})
    supported_targets: frozenset[QuantizationTarget] = frozenset({QuantizationTarget.LINEAR})
    min_cuda_capability: int | None = None
    required_modules: tuple[str, ...] = ()
    storage_format: str | None = None
    weight_bits: int | None = None
    activation_bits: int | None = None
    supports_tensor_parallel: bool = True
    supports_expert_parallel: bool = False
    supports_dynamic_activation: bool = False
    supports_static_activation: bool = True
    supports_online_quantization: bool = False

    def __post_init__(self) -> None:
        normalized_devices = frozenset(device.lower() for device in self.supported_devices)
        object.__setattr__(self, "supported_devices", normalized_devices)
        object.__setattr__(self, "supported_act_dtypes", tuple(self.supported_act_dtypes))
        object.__setattr__(self, "required_modules", tuple(self.required_modules))
        object.__setattr__(
            self,
            "supported_targets",
            frozenset(QuantizationTarget(t) for t in self.supported_targets),
        )
        if not normalized_devices:
            raise ValueError("supported_devices must not be empty")
        if not self.supported_targets:
            raise ValueError("supported_targets must not be empty")
        if self.min_cuda_capability is not None:
            _positive_int(self.min_cuda_capability, "min_cuda_capability")
        for field_name in ("weight_bits", "activation_bits"):
            value = getattr(self, field_name)
            if value is not None:
                _positive_int(value, field_name)
        if self.storage_format is not None:
            object.__setattr__(self, "storage_format", self.storage_format.lower())

    def supports_target(self, target: QuantizationTarget) -> bool:
        return target in self.supported_targets

    def validate(self, context: QuantizationContext) -> None:
        if context.target not in self.supported_targets:
            raise CapabilityError(
                "quantization.target",
                detail=(
                    f"target {context.target.value!r} is unsupported; supported targets are "
                    f"{sorted(target.value for target in self.supported_targets)}"
                ),
                remedy="select a quantization backend supporting this target",
            )
        if context.device_type not in self.supported_devices:
            raise CapabilityError(
                "quantization.device",
                detail=(
                    f"device {context.device_type!r} is unsupported; supported devices are "
                    f"{sorted(self.supported_devices)}"
                ),
                remedy="select a backend supporting the execution device",
            )
        if (
            context.activation_dtype is not None
            and self.supported_act_dtypes
            and context.activation_dtype not in self.supported_act_dtypes
        ):
            raise CapabilityError(
                "quantization.activation_dtype",
                detail=(
                    f"activation dtype {context.activation_dtype} is unsupported; supported "
                    f"dtypes are {[str(dtype) for dtype in self.supported_act_dtypes]}"
                ),
                remedy="use a supported activation dtype",
            )
        if context.device_type == "cuda" and self.min_cuda_capability is not None:
            if context.compute_capability is None:
                raise CapabilityError(
                    "quantization.compute_capability",
                    detail=(
                        "CUDA compute capability is unknown but the backend requires "
                        f"SM {self.min_cuda_capability} or newer"
                    ),
                    remedy="provide the execution device compute capability",
                )
            if context.compute_capability < self.min_cuda_capability:
                raise CapabilityError(
                    "quantization.compute_capability",
                    detail=(
                        f"SM {context.compute_capability} is below required "
                        f"SM {self.min_cuda_capability}"
                    ),
                    remedy="use a supported GPU or another quantization backend",
                )
        if context.tensor_parallel_size > 1 and not self.supports_tensor_parallel:
            raise CapabilityError(
                "quantization.tensor_parallel",
                detail=(f"tensor_parallel_size={context.tensor_parallel_size} is unsupported"),
                remedy="select a backend supporting tensor parallelism",
            )
        if context.expert_parallel_size > 1 and not self.supports_expert_parallel:
            raise CapabilityError(
                "quantization.expert_parallel",
                detail=(f"expert_parallel_size={context.expert_parallel_size} is unsupported"),
                remedy="select a backend supporting expert parallelism",
            )
        for module_name in self.required_modules:
            if not has_module(module_name):
                raise CapabilityError(
                    module_name,
                    detail=(
                        "the selected quantization backend requires an optional module "
                        "that is not importable"
                    ),
                    remedy=f"install {module_name}",
                )


class QuantizeMethodBase(ABC):
    """Storage and execution policy shared by quantized layer types."""

    uses_meta_device: ClassVar[bool] = False
    supported_targets: ClassVar[frozenset[QuantizationTarget]] = frozenset(
        {QuantizationTarget.LINEAR}
    )

    @abstractmethod
    def create_weights(
        self,
        layer: nn.Module,
        *weight_args: Any,
        **extra_weight_attrs: Any,
    ) -> None:
        """Create and register all parameters/buffers owned by ``layer``."""
        raise NotImplementedError

    @abstractmethod
    def apply(self, layer: nn.Module, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Execute the quantized operation after weights have been created."""
        raise NotImplementedError

    def supports(self, target: QuantizationTarget) -> bool:
        """Require both a target declaration and an implemented execution entry point."""
        if target not in self.supported_targets:
            return False
        entrypoint = {
            QuantizationTarget.EMBEDDING: "embedding",
            QuantizationTarget.EXPERT: "apply_expert",
            QuantizationTarget.KV_CACHE: "apply_kv_cache",
        }.get(target)
        return entrypoint is None or getattr(type(self), entrypoint) is not getattr(
            QuantizeMethodBase, entrypoint
        )

    def embedding(
        self,
        layer: nn.Module,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        del layer, args, kwargs
        raise CapabilityError(
            "quantization.embedding",
            detail=f"{type(self).__name__} does not implement embedding lookup",
            remedy="implement embedding() in the selected quantization method",
        )

    def apply_expert(
        self,
        layer: nn.Module,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        del layer, args, kwargs
        raise CapabilityError(
            "quantization.expert",
            detail=f"{type(self).__name__} does not implement expert execution",
            remedy="implement apply_expert() in the selected quantization method",
        )

    def apply_kv_cache(
        self,
        layer: nn.Module,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        del layer, args, kwargs
        raise CapabilityError(
            "quantization.kv_cache",
            detail=f"{type(self).__name__} does not implement KV-cache quantization",
            remedy="implement apply_kv_cache() in the selected quantization method",
        )

    def tie_weights(self, layer: nn.Module, embed_tokens: nn.Module) -> nn.Module:
        """Tie plain weights; packed methods must override to share scales/layout too."""
        if not isinstance(getattr(embed_tokens, "weight", None), nn.Parameter):
            raise TypeError("embed_tokens must expose a weight parameter")
        for module in (layer, embed_tokens):
            if any(name != "weight" for name, _ in module.named_parameters()) or any(
                True for _ in module.buffers()
            ):
                raise CapabilityError(
                    "quantization.tie_weights",
                    detail="tying packed storage requires sharing all parameters and scales",
                    remedy="override tie_weights() for this storage format",
                )
        layer.weight = embed_tokens.weight
        return layer

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        del layer
        return None

    def validate_layer(self, layer: nn.Module, *, prefix: str = "") -> None:
        del layer, prefix
        return None


class BaseQuantization(ABC):
    """Configuration and method selection, separate from per-layer weight storage.

    Instances may be shared across layers. ``get_quant_method`` must return a
    fresh method when that method holds per-layer state, and must not allocate or
    mutate weights. Selection validates the configuration and method target before
    the caller invokes ``create_weights``. This is not a checkpoint loader or a
    promise that every target/device has an implemented quantized kernel.
    """

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """Return the stable quantization name, e.g. the WeightSpec.quant method."""

    @classmethod
    @abstractmethod
    def from_config(cls, config: Mapping[str, Any]) -> Self:
        """Parse and validate checkpoint/configuration fields without loading weights."""

    @abstractmethod
    def get_capabilities(self) -> QuantizationCapabilities:
        """Return capabilities for this configuration, including optional dependencies."""

    @abstractmethod
    def get_quant_method(self, layer: nn.Module, prefix: str) -> QuantizeMethodBase | None:
        """Choose a method for the layer; None means the layer is unsupported/excluded."""

    def validate(self, context: QuantizationContext) -> None:
        """Raise CapabilityError before allocation when runtime facts are unsupported."""
        self.get_capabilities().validate(context)

    def select_method(
        self, layer: nn.Module, context: QuantizationContext, *, prefix: str = ""
    ) -> QuantizeMethodBase:
        """Select an executable target without silently falling back to dense weights."""
        self.validate(context)
        method = self.get_quant_method(layer, prefix)
        if method is None:
            raise CapabilityError(
                "quantization.layer",
                detail=f"{self.get_name()} has no method for {prefix or type(layer).__name__}",
                remedy="exclude this layer explicitly or implement its quantization method",
            )
        if not isinstance(method, QuantizeMethodBase):
            raise TypeError("get_quant_method() must return QuantizeMethodBase or None")
        if not method.supports(context.target):
            raise CapabilityError(
                "quantization.method_target",
                detail=f"{type(method).__name__} does not support {context.target.value}",
                remedy="select a method supporting this layer target",
            )
        return method
