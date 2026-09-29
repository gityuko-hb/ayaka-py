"""Explicit layer sidecars; no management, I/O, or forward hooks."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from ayaka.lora.backend import LoRABackend, LoRAExpandContext
from ayaka.lora.cache import DeviceAdapterPool
from ayaka.lora.mapping import ProjectionTarget


@dataclass(frozen=True, slots=True)
class LoRASelection:
    owner: object
    route: LoRAExpandContext


selection: ContextVar[LoRASelection | None] = ContextVar("ayaka_lora_selection", default=None)


class LinearLoRASidecar:
    def __init__(
        self,
        owner: object,
        pool: DeviceAdapterPool,
        backend: LoRABackend,
        targets: tuple[ProjectionTarget, ...],
    ) -> None:
        self.owner, self.pool, self.backend, self.targets = owner, pool, backend, targets

    def __call__(self, x: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        context = selection.get()
        if context is None or context.owner is not self.owner:
            return output
        for target in self.targets:
            a, b = self.pool.weights[target.key]
            self.backend.run_expand(
                context.route,
                x,
                output,
                a,
                b,
                self.pool.ranks[target.key],
                target.output_offset,
            )
        return output


class TorchLinearWithLoRA(nn.Linear):
    """Compatibility wrapper sharing original parameters and their state_dict keys."""

    def __init__(self, base: nn.Linear, sidecar: LinearLoRASidecar) -> None:
        nn.Module.__init__(self)
        self.in_features, self.out_features = base.in_features, base.out_features
        self.weight, self.bias = base.weight, base.bias
        self.sidecar = sidecar
        self.train(base.training)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.sidecar(input, F.linear(input, self.weight, self.bias))
