"""Shared layer plumbing; runtime resources and per-forward scratch stay caller-owned."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from ayaka.distributed.device import AsyncHandle, CommOpType, CommunicationBackend, DeviceGroup
from ayaka.layers.quantization.base import (
    BaseQuantization,
    QuantizationContext,
    QuantizationTarget,
    QuantizeMethodBase,
)
from ayaka.utils.import_utils import CapabilityError

if TYPE_CHECKING:
    from ayaka.device.context import DeviceContext


class BaseLayer(nn.Module, ABC):
    """PyTorch layer with explicit runtime dependencies and a quantization lifecycle.

    Subclasses implement ``forward`` and use ``run_kernel`` with an OpHandle or
    another callable. The callable owns shape/dtype checks and mutation semantics;
    this wrapper preserves its return value, aliases, exceptions and current stream.
    Import backend modules during setup, before compilation/capture.

    Device contexts, TP/EP groups and communication backends are borrowed. This
    class never initializes process groups, closes a context, creates streams, or
    commits KV leases. Metadata, output buffers and scratch belong to each forward.
    ``nn.Module.__call__`` is untouched, so hooks and module registration work normally.

    Quantized subclasses call ``create_weights`` after defining their dimensions,
    load the registered weights, then call ``process_weights_after_loading`` once.
    Repacking failures are terminal for that instance: recreate it and reload.
    Moving/reloading packed weights after finalization is backend-specific; do it
    before finalization unless the concrete method explicitly supports it.
    """

    def __init__(
        self,
        *,
        prefix: str = "",
        device_context: DeviceContext | None = None,
        tp_group: DeviceGroup | None = None,
        ep_group: DeviceGroup | None = None,
        communication: CommunicationBackend | None = None,
        quant_config: BaseQuantization | None = None,
    ) -> None:
        super().__init__()
        for group in (tp_group, ep_group):
            if group is not None and group.size < 1:
                raise ValueError(f"group {group.name!r} must not be empty")
        self.prefix = prefix
        self.device_context = device_context
        self.tp_group = tp_group
        self.ep_group = ep_group
        self.communication = communication
        self.quant_config = quant_config
        self.quant_method: QuantizeMethodBase | None = None
        self.quant_context: QuantizationContext | None = None
        self._quantization_state = "uninitialized"

    @property
    def tp_size(self) -> int:
        return self.tp_group.size if self.tp_group is not None else 1

    @property
    def tp_rank(self) -> int:
        """Index inside the supplied TP group, independent of global/local env rank."""
        return self.tp_group.local_rank if self.tp_group is not None else 0

    @property
    def ep_size(self) -> int:
        return self.ep_group.size if self.ep_group is not None else 1

    @property
    def ep_rank(self) -> int:
        return self.ep_group.local_rank if self.ep_group is not None else 0

    def runtime_context(
        self,
        device: torch.device | str,
        activation_dtype: torch.dtype,
        *,
        target: QuantizationTarget = QuantizationTarget.LINEAR,
    ) -> QuantizationContext:
        """Validate execution placement and derive capability/TP/EP facts at setup."""
        return QuantizationContext.from_runtime(
            device,
            target=target,
            activation_dtype=activation_dtype,
            device_context=self.device_context,
            tp_group=self.tp_group,
            ep_group=self.ep_group,
        )

    @staticmethod
    def run_kernel[**P, R](kernel: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        """Call the public kernel handle once, preserving out= and in-place aliases.

        Never unwrap ``OpHandle.kernel``: that would bypass its dispatcher, fake
        implementation and explicit reference mode. There is no retry on failure.
        Plain callables are supported, but gain no compiler/capture guarantees here.
        """
        return kernel(*args, **kwargs)

    def all_reduce(
        self,
        tensor: torch.Tensor,
        *,
        group: DeviceGroup,
        op: CommOpType = CommOpType.SUM,
        async_op: bool = False,
    ) -> AsyncHandle | None:
        """Reduce in place on an explicit group; return the backend's completion handle.

        A singleton group is identity. For async work, the caller must join the
        returned handle before consuming its output or retiring associated storage.
        Other collectives are available through the borrowed ``communication`` port.
        """
        if group.size < 1:
            raise ValueError("collective group must not be empty")
        if len(set(group.ranks)) != len(group.ranks):
            raise ValueError("collective group contains duplicate ranks")
        local = group.devices[group.local_rank]
        if local.kind.value != tensor.device.type or (
            tensor.device.type != "cpu" and local.index != tensor.device.index
        ):
            raise ValueError(f"collective tensor device {tensor.device} does not match {local}")
        if group.is_trivial:
            return None
        if self.communication is None:
            raise CapabilityError(
                "layers.communication",
                detail=f"group {group.name!r} requires a communication backend",
                remedy="supply the runtime CommunicationBackend for this layer",
            )
        return self.communication.all_reduce(tensor, group, op=op, async_op=async_op)

    def create_weights(
        self,
        *weight_args: Any,
        device: torch.device | str,
        params_dtype: torch.dtype,
        activation_dtype: torch.dtype | None = None,
        target: QuantizationTarget = QuantizationTarget.LINEAR,
        **extra_weight_attrs: Any,
    ) -> None:
        """Select a quantization method and register weights after capability checks.

        ``params_dtype`` is the unquantized parameter dtype; packed storage dtype
        is chosen by the method. ``activation_dtype`` defaults to ``params_dtype``.
        ``device`` is the final execution
        placement. Methods with ``uses_meta_device`` receive meta for allocation
        and can read ``layer.quant_context.device`` for the final placement.
        Loaders must materialize meta storage before finalization.
        """
        if self._quantization_state != "uninitialized":
            raise RuntimeError("quantized weights have already been initialized or failed")
        if self.quant_config is None:
            raise RuntimeError("create_weights() requires a quant_config")
        context = self.runtime_context(
            device,
            activation_dtype if activation_dtype is not None else params_dtype,
            target=target,
        )
        method = self.quant_config.select_method(self, context, prefix=self.prefix)
        self.quant_context = context
        self.quant_method = method
        # Weight creation may partially mutate the module before raising. Do not retry.
        self._quantization_state = "failed"
        method.create_weights(
            self,
            *weight_args,
            device=torch.device("meta") if method.uses_meta_device else context.device,
            params_dtype=params_dtype,
            **extra_weight_attrs,
        )
        self._quantization_state = "created"

    def process_weights_after_loading(self) -> None:
        """Finalize and validate quantized storage once, after checkpoint assignment.

        Dense subclasses can override this hook. It is a no-op without quantization.
        Returning successfully does not itself prove asynchronous GPU work completed;
        the loader/runtime must order packing before the first forward.
        """
        if self.quant_config is None:
            return
        if self._quantization_state != "created" or self.quant_method is None:
            raise RuntimeError("quantized weights must be created and finalized exactly once")
        tensors = (*self.parameters(), *self.buffers())
        if any(tensor.is_meta for tensor in tensors):
            raise RuntimeError("materialize meta weights before finalizing quantization")
        context = self.quant_context
        assert context is not None and context.activation_dtype is not None
        # Recheck borrowed runtime resources after potentially lengthy checkpoint I/O.
        current = self.runtime_context(
            context.device, context.activation_dtype, target=context.target
        )
        self.quant_config.validate(current)
        self._quantization_state = "failed"
        self.quant_method.process_weights_after_loading(self)
        if any(tensor.is_meta for tensor in (*self.parameters(), *self.buffers())):
            raise RuntimeError("quantization finalization left meta storage")
        for name, parameter in self.named_parameters():
            if parameter.device != context.device:
                raise RuntimeError(
                    f"quantized parameter {name!r} is on {parameter.device}, "
                    f"expected {context.device}"
                )
        self.quant_method.validate_layer(self, prefix=self.prefix)
        self._quantization_state = "ready"

    def apply_quantization(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Execute the selected target only after successful weight finalization."""
        if self._quantization_state != "ready" or self.quant_method is None:
            raise RuntimeError("quantized weights are not ready; load and finalize them first")
        assert self.quant_context is not None
        target = self.quant_context.target
        if target is QuantizationTarget.EMBEDDING:
            return self.quant_method.embedding(self, *args, **kwargs)
        if target is QuantizationTarget.EXPERT:
            return self.quant_method.apply_expert(self, *args, **kwargs)
        if target is QuantizationTarget.KV_CACHE:
            return self.quant_method.apply_kv_cache(self, *args, **kwargs)
        return self.quant_method.apply(self, *args, **kwargs)

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Execute layer semantics using caller-supplied tensors and per-forward metadata."""
        raise NotImplementedError
