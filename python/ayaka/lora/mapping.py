"""Explicit logical projection descriptors derived from native layer layouts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from ayaka.layers.linear.core import PackedColumnParallelLinear, RowParallelLinear
from ayaka.model_loader.module import WeightBinding


@dataclass(frozen=True, slots=True)
class ProjectionTarget:
    key: str
    module: str
    projection: str | None
    input_size: int
    output_size: int
    local_input_size: int
    local_output_size: int
    input_offset: int = 0
    source_output_offset: int = 0
    output_offset: int = 0
    global_output_offset: int = 0
    tp_rank: int = 0
    tp_size: int = 1

    def localize(self, a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            a[:, self.input_offset : self.input_offset + self.local_input_size],
            b[self.source_output_offset : self.source_output_offset + self.local_output_size],
        )


def projection_targets(model: nn.Module, names: tuple[str, ...]) -> tuple[ProjectionTarget, ...]:
    modules = dict(model.named_modules())
    result: list[ProjectionTarget] = []
    for name in names:
        module = modules.get(name)
        if not isinstance(module, (nn.Linear, PackedColumnParallelLinear, RowParallelLinear)):
            raise ValueError(f"unsupported LoRA operator {name!r}")
        if getattr(module, "quant_config", None) is not None:
            raise ValueError("quantized-base LoRA requires P1 certification")
        if isinstance(module, PackedColumnParallelLinear):
            parts = module.layout.parts
            for part in parts:
                key = name if len(parts) == 1 else f"{name}::{part.name}"
                result.append(
                    ProjectionTarget(
                        key,
                        name,
                        part.name,
                        module.input_size,
                        part.global_size,
                        module.input_size_per_partition,
                        part.local_size,
                        source_output_offset=part.source_offset,
                        output_offset=part.local_offset,
                        global_output_offset=part.global_offset,
                        tp_rank=module.tp_rank,
                        tp_size=module.tp_size,
                    )
                )
        elif isinstance(module, RowParallelLinear):
            result.append(
                ProjectionTarget(
                    name,
                    name,
                    None,
                    module.input_size,
                    module.output_size,
                    module.input_size_per_partition,
                    module.output_size,
                    input_offset=module.tp_rank * module.input_size_per_partition,
                    tp_rank=module.tp_rank,
                    tp_size=module.tp_size,
                )
            )
        else:
            result.append(
                ProjectionTarget(
                    name,
                    name,
                    None,
                    module.in_features,
                    module.out_features,
                    module.in_features,
                    module.out_features,
                )
            )
    return tuple(result)


def checkpoint_mapping(
    model: nn.Module,
    targets: tuple[ProjectionTarget, ...],
    bindings: Mapping[str, WeightBinding] | None = None,
) -> dict[str, str]:
    """Reuse the architecture's checkpoint bindings; never guess model-family names."""
    if bindings is None:
        factory = getattr(model, "weight_bindings", None)
        bindings = cast(Mapping[str, WeightBinding], factory()) if callable(factory) else None
    result: dict[str, str] = {}
    if bindings is not None:
        for source, binding in bindings.items():
            if not source.endswith(".weight") or not binding.parameter.endswith(".weight"):
                continue
            module = binding.parameter.removesuffix(".weight")
            candidates = [t for t in targets if t.module == module]
            if binding.shard_id is not None:
                native = model.get_submodule(module)
                if not candidates:
                    continue
                if not isinstance(native, PackedColumnParallelLinear) or isinstance(
                    binding.shard_id, tuple
                ):
                    raise ValueError("adapter mapping requires one explicit packed projection ID")
                part = native.layout[binding.shard_id]
                candidates = [t for t in candidates if t.projection == part.name]
            if len(candidates) == 1:
                result[source.removesuffix(".weight")] = candidates[0].key
            elif candidates and binding.shard_id is None:
                result[source.removesuffix(".weight")] = module
    # Native keys are useful for custom models and explicit tensor injection.
    for target in targets:
        result.setdefault(target.key, target.key)
    return result
