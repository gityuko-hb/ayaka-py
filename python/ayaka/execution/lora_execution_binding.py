"""Authoritative bounded LoRA storage, request leases and graph-safe row selection.

Supports unquantized, TP=1 linear projections, including packed projections in
their native output order. A/B weights are never registered as base parameters.
Changing adapter content cannot repoint captured storage. Management calls and
request admission use one lock; CUDA copies finish before a slot is published.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from threading import RLock

import torch
from torch import nn

from ayaka.execution.execution_variant import AdapterIdentity, ExecutionVariant
from ayaka.layers.linear.core import LinearBase
from ayaka.utils.validation import require_int


@dataclass(frozen=True, slots=True)
class LoRAConfig:
    modules: tuple[str, ...]
    max_rank: int = 16
    max_adapters: int = 4
    memory_bytes: int = 64 << 20

    def __post_init__(self) -> None:
        ExecutionVariant(self.modules, self.max_rank, self.max_adapters, "float32")
        require_int(self.memory_bytes, "LoRA memory_bytes", minimum=1)


@dataclass(frozen=True, slots=True)
class LoRAWeights:
    a: torch.Tensor
    b: torch.Tensor
    scale: float = 1.0


@dataclass(frozen=True, slots=True)
class AdapterBinding:
    identity: AdapterIdentity
    slot: int
    generation: int
    content_digest: str


class AdapterLease:
    def __init__(self, owner: LoRAExecutionBinding, binding: AdapterBinding) -> None:
        self.owner, self.binding, self.closed = owner, binding, False

    def validate(self) -> None:
        if self.closed:
            raise ValueError("adapter lease is retired")
        self.owner.validate(self.binding)

    def close(self) -> None:
        with self.owner._lock:
            if not self.closed:
                self.validate()
                self.owner._leases[self.binding.slot] -= 1
                self.closed = True


_selection: ContextVar[tuple[object, torch.Tensor] | None] = ContextVar("lora_rows", default=None)


class LoRAExecutionBinding:
    """One owner for adapter weights. Unload/reload refuses live request leases.

    Request leases deliberately last through cancellation drain and retirement,
    not just the Python forward call. Slot zero is explicitly the absent adapter.
    ``load`` takes already materialized native projection weights; HF adapter
    name conversion and quantized/sharded operators are outside this contract.
    """

    def __init__(self, model: nn.Module, config: LoRAConfig) -> None:
        if getattr(model, "_ayaka_lora_owner", None) is not None:
            raise ValueError("model already has a LoRA owner")
        self.config = config
        self.model = model
        parameter = next(model.parameters())
        self.device, self.dtype = parameter.device, parameter.dtype
        self.variant = ExecutionVariant(
            config.modules,
            config.max_rank,
            config.max_adapters,
            str(self.dtype).removeprefix("torch."),
        )
        modules = dict(model.named_modules())
        selected = {}
        self.bytes = 0
        for name in config.modules:
            module = modules.get(name)
            if not isinstance(module, (nn.Linear, LinearBase)):
                raise ValueError(f"unsupported LoRA operator {name!r}")
            if getattr(module, "tp_size", 1) != 1 or getattr(module, "quant_config", None):
                raise ValueError("LoRA requires unquantized TP=1 projections")
            weight = module.weight
            if weight.ndim != 2 or weight.device != self.device or weight.dtype != self.dtype:
                raise ValueError("LoRA projection weight must have native [out, in] layout")
            selected[name] = module
            self.bytes += (
                (config.max_adapters + 1)
                * config.max_rank
                * sum(weight.shape)
                * weight.element_size()
            )
        if self.bytes > config.memory_bytes:
            raise MemoryError("LoRA stable slots exceed memory reservation")
        self._lock = RLock()
        self._bindings: dict[AdapterIdentity, AdapterBinding] = {}
        self._generations = [0] * (config.max_adapters + 1)
        self._leases = [0] * (config.max_adapters + 1)
        self._readers = 0
        self._weights: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._hooks = []
        self.closed = False
        for name, module in selected.items():
            out_features, in_features = module.weight.shape
            a = torch.zeros(
                config.max_adapters + 1,
                config.max_rank,
                in_features,
                device=self.device,
                dtype=self.dtype,
            )
            b = torch.zeros(
                config.max_adapters + 1,
                out_features,
                config.max_rank,
                device=self.device,
                dtype=self.dtype,
            )
            self._weights[name] = a, b
            self._hooks.append(module.register_forward_hook(self._hook(name)))
        object.__setattr__(model, "_ayaka_lora_owner", self)

    def _hook(self, name: str):
        def apply(_module, args, output):
            selection = _selection.get()
            if selection is None or selection[0] is not self:
                return output
            rows = selection[1]
            x = args[0]
            tensor = output[0] if isinstance(output, tuple) else output
            if x.ndim != 2 or x.shape[0] != rows.numel():
                raise ValueError("LoRA rows must match packed token rows")
            a, b = self._weights[name]
            low = torch.bmm(a.index_select(0, rows), x.unsqueeze(-1))
            delta = torch.bmm(b.index_select(0, rows), low).squeeze(-1)
            result = tensor + delta
            return (result, *output[1:]) if isinstance(output, tuple) else result

        return apply

    @property
    def storage_identity(self) -> tuple:
        return (
            self.variant,
            tuple((n, a.data_ptr(), b.data_ptr()) for n, (a, b) in self._weights.items()),
        )

    def working_bytes(self, tokens: int) -> int:
        """Conservative transient gather/BMM reservation in addition to stable slots."""
        return self.bytes + max(
            tokens
            * (a.shape[1] * a.shape[2] + b.shape[1] * b.shape[2] + a.shape[1] + 2 * b.shape[1])
            * a.element_size()
            for a, b in self._weights.values()
        )

    def load(self, identity: AdapterIdentity, weights: Mapping[str, LoRAWeights]) -> AdapterBinding:
        """Validate everything before copying; refuse replacing leased content."""
        with self._lock, torch.inference_mode():
            if self.closed:
                raise RuntimeError("LoRA owner is closed")
            if not isinstance(identity, AdapterIdentity) or set(weights) != set(self._weights):
                raise ValueError("adapter identity/modules do not match the structural variant")
            if identity in self._bindings:
                raise ValueError("adapter revision already loaded; unload explicitly first")
            if any(key.name == identity.name for key in self._bindings):
                raise ValueError("unload the previous adapter revision before reloading its name")
            used = {value.slot for value in self._bindings.values()}
            slot = next((i for i in range(1, self.config.max_adapters + 1) if i not in used), None)
            if slot is None:
                raise MemoryError("adapter capacity exhausted")
            for name, value in weights.items():
                a, b = self._weights[name]
                rank = value.a.shape[0] if value.a.ndim == 2 else 0
                if (
                    not 1 <= rank <= self.config.max_rank
                    or value.a.shape != (rank, a.shape[2])
                    or value.b.shape != (b.shape[1], rank)
                    or value.a.dtype != self.dtype
                    or value.b.dtype != self.dtype
                    or not math.isfinite(value.scale)
                ):
                    raise ValueError(f"invalid LoRA rank/shape/dtype/scale for {name}")
                if (
                    not torch.isfinite(value.a).all()
                    or not torch.isfinite(value.b).all()
                    or not torch.isfinite(value.b * value.scale).all()
                ):
                    raise ValueError("nonfinite adapter weights")
            digest = hashlib.sha256(identity.prefix_key.encode())
            for name in self.config.modules:
                value = weights[name]
                digest.update(repr((name, value.a.shape, value.a.dtype, value.scale)).encode())
                for tensor in (value.a, value.b):
                    digest.update(
                        tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
                    )
            for name, value in weights.items():
                a, b = self._weights[name]
                rank = value.a.shape[0]
                a[slot].zero_()
                b[slot].zero_()
                a[slot, :rank].copy_(value.a)
                b[slot, :, :rank].copy_(value.b * value.scale)
            if self.device.type == "cuda":
                torch.cuda.current_stream(self.device).synchronize()
            self._generations[slot] += 1
            binding = AdapterBinding(identity, slot, self._generations[slot], digest.hexdigest())
            self._bindings[identity] = binding
            return binding

    def validate(self, binding: AdapterBinding) -> None:
        if self.closed or self._bindings.get(binding.identity) != binding:
            raise ValueError("stale adapter slot/revision")

    def acquire(self, identity: AdapterIdentity) -> AdapterLease:
        with self._lock:
            if self.closed or identity not in self._bindings:
                raise ValueError("adapter revision is not loaded")
            binding = self._bindings[identity]
            self._leases[binding.slot] += 1
            return AdapterLease(self, binding)

    def prefix_identity(self, identity: AdapterIdentity) -> str:
        """Hash content too: reusing a caller's revision label cannot alias old KV."""
        with self._lock:
            if self.closed or identity not in self._bindings:
                raise ValueError("adapter revision is not loaded")
            return self._bindings[identity].content_digest

    def retain_request(self) -> None:
        """Base rows also pin the zero slot and captured weight backing."""
        with self._lock:
            if self.closed:
                raise RuntimeError("LoRA owner is closed")
            self._readers += 1

    def release_request(self) -> None:
        with self._lock:
            if self._readers <= 0:
                raise RuntimeError("unbalanced LoRA request lifetime")
            self._readers -= 1

    def unload(self, identity: AdapterIdentity) -> None:
        with self._lock:
            binding = self._bindings[identity]
            if self._leases[binding.slot]:
                raise RuntimeError("adapter is leased; drain its requests before unload")
            del self._bindings[identity]

    @contextmanager
    def select(self, rows: torch.Tensor) -> Iterator[None]:
        if self.closed or rows.dtype != torch.long or rows.device != self.device or rows.ndim != 1:
            raise ValueError("invalid LoRA row binding")
        token = _selection.set((self, rows))
        try:
            yield
        finally:
            _selection.reset(token)

    def close(self) -> None:
        with self._lock:
            if self.closed:
                return
            if any(self._leases) or self._readers:
                raise RuntimeError("cannot close leased adapters")
            for hook in self._hooks:
                hook.remove()
            self._hooks.clear()
            self._weights.clear()
            self._bindings.clear()
            object.__setattr__(self.model, "_ayaka_lora_owner", None)
            self.closed = True
