"""Execution bridge over the single residency manager and native layer sidecars."""

from __future__ import annotations

from collections.abc import Generator, Mapping
from contextlib import contextmanager

import torch
from torch import nn

from ayaka.layers.linear.core import LinearBase
from ayaka.lora.backend import (
    BackendCapability,
    LoRAExecutionPlan,
    LoRAExpandContext,
    LoRAWorkspace,
    create_backend,
)
from ayaka.lora.cache import DeviceAdapterPool
from ayaka.lora.config import LoRAConfig
from ayaka.lora.layers import LinearLoRASidecar, LoRASelection, TorchLinearWithLoRA, selection
from ayaka.lora.loader import LoRALoader
from ayaka.lora.manager import AdapterBinding, AdapterLease, AdapterStatus, LoRAManager
from ayaka.lora.mapping import checkpoint_mapping, projection_targets
from ayaka.lora.resolver import AdapterRef, AdapterResolver
from ayaka.lora.variant import AdapterIdentity, ExecutionVariant
from ayaka.lora.weights import LoRAWeights
from ayaka.model_loader.module import WeightBinding

__all__ = ["AdapterBinding", "AdapterLease", "LoRAConfig", "LoRAExecutionBinding", "LoRAWeights"]


class LoRAExecutionBinding:
    def __init__(
        self,
        model: nn.Module,
        config: LoRAConfig,
        *,
        bindings: Mapping[str, WeightBinding] | None = None,
        base_model: str | None = None,
    ) -> None:
        if getattr(model, "_ayaka_lora_owner", None) is not None:
            raise ValueError("model already has a LoRA owner")
        self.config, self.model = config, model
        self.targets = projection_targets(model, config.modules)
        first = model.get_submodule(config.modules[0]).weight
        if not isinstance(first, torch.Tensor):
            raise ValueError("LoRA projection requires a tensor weight")
        self.device, self.dtype = first.device, first.dtype
        if self.device.type == "meta":
            raise ValueError("materialize base weights before LoRA bootstrap")
        if config.compute_dtype is not None:
            requested = getattr(torch, config.compute_dtype)
            if requested != self.dtype:
                raise ValueError(
                    "compute_dtype must match the base activation dtype until "
                    "quantized-base certification"
                )
        self.variant = ExecutionVariant(
            config.modules,
            config.max_rank,
            config.max_adapters,
            str(self.dtype).removeprefix("torch."),
        )
        for name in config.modules:
            module = model.get_submodule(name)
            if module.weight.device != self.device or module.weight.dtype != self.dtype:
                raise ValueError("LoRA layers require one device and runtime dtype per owner")
            if getattr(module, "lora_sidecar", None) is not None:
                raise ValueError(f"LoRA operator {name!r} already has a sidecar")
        self.loader = LoRALoader(
            config,
            self.targets,
            checkpoint_mapping(model, self.targets, bindings),
            self.dtype,
            base_model=base_model or config.base_model_name_or_path,
        )
        self.pool = DeviceAdapterPool(self.targets, config, self.device, self.dtype)
        self.manager = LoRAManager(config, self.pool)
        selection = create_backend(
            config.backend, capability=BackendCapability.probe(), device=self.device
        )
        self.backend = selection.backend
        self.backend_selection = selection
        if selection.fallback_reason is not None:
            self.manager.note_backend_fallback(selection.fallback_reason)
        self._originals: dict[str, nn.Module] = {}
        self._native: list[LinearBase] = []
        self._debug_workspace: LoRAWorkspace | None = None
        for name in config.modules:
            module = model.get_submodule(name)
            sidecar = LinearLoRASidecar(
                self, self.pool, self.backend, tuple(t for t in self.targets if t.module == name)
            )
            if isinstance(module, LinearBase):
                module.lora_sidecar = sidecar
                self._native.append(module)
            else:
                assert isinstance(module, nn.Linear)
                self._originals[name] = module
                model.set_submodule(name, TorchLinearWithLoRA(module, sidecar))
        object.__setattr__(model, "_ayaka_lora_owner", self)

    @property
    def closed(self) -> bool:
        return self.manager.closed

    @property
    def bytes(self) -> int:
        return self.pool.bytes

    @property
    def storage_identity(self) -> tuple:
        return self.variant, self.targets, self.pool.storage_identity

    @property
    def max_local_output(self) -> int:
        return max(t.local_output_size for t in self.targets)

    def execution_plan(self, tokens: int, *, phase: str = "reference") -> LoRAExecutionPlan:
        """Resolve the support decision before any enqueue or slot mutation."""
        if tokens < 1:
            raise ValueError("LoRA plan requires at least one token")
        return self.backend.plan(
            tokens=tokens,
            rank=self.config.max_rank,
            output=self.max_local_output,
            capacity=self.config.max_adapters,
            dtype=self.dtype,
            phase=phase,
        )

    def plan_identity(self, tokens: int, *, phase: str = "reference") -> tuple:
        return self.execution_plan(tokens, phase=phase).identity

    def working_bytes(self, tokens: int, flights: int = 1, *, routing_bytes: int = 0) -> int:
        if flights < 1:
            raise ValueError("flights must be positive")
        plan = self.execution_plan(tokens)
        return self.bytes + flights * (plan.workspace.per_flight_bytes + routing_bytes)

    def workspace(self, tokens: int) -> LoRAWorkspace:
        return self.execution_plan(tokens).workspace.allocate(self.device)

    def load(self, identity: AdapterIdentity, weights: Mapping[str, LoRAWeights]) -> AdapterBinding:
        result = self.manager.load(self.loader.from_native(identity, weights))
        assert result is not None
        return result

    load_adapter_from_tensors = load

    def load_adapter(self, ref: AdapterRef, *, device: bool = True) -> AdapterBinding | None:
        resolved = AdapterResolver().resolve(ref)
        return self.manager.load(self.loader.load(resolved), device=device)

    def acquire(self, identity: AdapterIdentity) -> AdapterLease:
        return self.manager.acquire(identity)

    def validate(self, binding: AdapterBinding) -> None:
        self.manager.validate(binding)

    def prefix_identity(self, identity: AdapterIdentity) -> str:
        return self.manager.prefix_identity(identity)

    def retain_request(self) -> None:
        self.manager.retain_request()

    def release_request(self) -> None:
        self.manager.release_request()

    def unload(self, identity: AdapterIdentity) -> None:
        self.manager.unload(identity)

    unload_adapter = unload

    def pin_adapter(self, identity: AdapterIdentity) -> None:
        self.manager.pin(identity)

    def unpin_adapter(self, identity: AdapterIdentity) -> None:
        self.manager.unpin(identity)

    def list_adapters(self) -> tuple[AdapterStatus, ...]:
        return self.manager.list_adapters()

    def get_adapter_status(self, identity: AdapterIdentity) -> AdapterStatus:
        return self.manager.get_adapter_status(identity)

    def stats(self) -> dict[str, int | float | str | None]:
        data = self.manager.stats()
        data["lora_backend"] = self.backend.name
        data["lora_backend_requested"] = self.config.backend
        data["lora_backend_version"] = self.backend.version
        data["lora_read_footprint"] = str(self.backend.read_footprint)
        data["lora_backend_fallback"] = self.backend_selection.fallback_reason
        return data

    @contextmanager
    def select(
        self,
        rows: torch.Tensor,
        workspace: LoRAWorkspace | None = None,
        *,
        sequence_slots: torch.Tensor | None = None,
        segment_offsets: torch.Tensor | None = None,
        token_permutation: torch.Tensor | None = None,
        token_inverse: torch.Tensor | None = None,
        projected_rows: torch.Tensor | None = None,
        slot_counts: torch.Tensor | None = None,
        phase: str = "reference",
    ) -> Generator[None]:
        """Caller keeps request leases until GPU completion.

        Serving passes its flight workspace and grouped routing tables; direct
        oracle calls provision scratch here, outside forward/capture. Shared
        oracle scratch is single-stream only. ``phase`` records the planned
        algorithm (decode BGMV vs prefill SGMV) for the captured kernel route.
        """
        if self.closed or rows.dtype != torch.long or rows.device != self.device or rows.ndim != 1:
            raise ValueError("invalid LoRA row binding")
        if workspace is None:
            if self._debug_workspace is None or self._debug_workspace.tokens < rows.numel():
                if self.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("provision LoRA workspace before CUDA Graph capture")
                self._debug_workspace = self.workspace(rows.numel())
            workspace = self._debug_workspace
        route = LoRAExpandContext(
            rows=rows,
            workspace=workspace,
            sequence_slots=sequence_slots,
            segment_offsets=segment_offsets,
            token_permutation=token_permutation,
            token_inverse=token_inverse,
            projected_rows=projected_rows,
            slot_counts=slot_counts,
            phase=phase,
        )
        with self.manager.execution_guard():
            token = selection.set(LoRASelection(self, route))
            try:
                yield
            finally:
                selection.reset(token)

    def close(self) -> None:
        with self.manager.lock:
            self.manager.close()
            for module in self._native:
                module.lora_sidecar = None
            for name, original in self._originals.items():
                self.model.set_submodule(name, original)
            self._native.clear()
            self._originals.clear()
            self._debug_workspace = None
            object.__setattr__(self.model, "_ayaka_lora_owner", None)
