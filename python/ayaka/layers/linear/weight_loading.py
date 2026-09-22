from __future__ import annotations

import itertools
from typing import Any

import torch
from torch.nn import Parameter

from ayaka.utils.math_utils import div_ceil
from ayaka.utils.validation import require_int

from .layout import ProjectionLayout


class WeightLoadError(ValueError):
    pass


def set_weight_attrs(parameter: Parameter, attrs: dict[str, Any]) -> None:
    for name, value in attrs.items():
        setattr(parameter, name, value)


def adjust_marlin_shard(
    parameter: Parameter,
    shard_size: int,
    shard_offset: int,
) -> tuple[int, int]:
    tile_size = getattr(parameter, "marlin_tile_size", None)
    if tile_size is None:
        return shard_size, shard_offset
    require_int(tile_size, "marlin_tile_size", minimum=1)
    return shard_size * tile_size, shard_offset * tile_size


def adjust_block_scale_shard(
    weight_block_size: tuple[int, ...] | None,
    shard_size: int,
    shard_offset: int,
) -> tuple[int, int]:
    if not weight_block_size:
        raise ValueError("weight_block_size is required for block scale parameters")
    block_n = int(weight_block_size[0])
    if block_n <= 0:
        raise ValueError("weight block size must be positive")
    return (
        div_ceil(shard_size, block_n),
        div_ceil(shard_offset, block_n),
    )


def adjust_bitsandbytes_4bit_shard(
    parameter: Parameter,
    shard_offsets: dict[str, tuple[int, int]],
    loaded_shard_id: str,
) -> tuple[int, int]:
    total, _ = shard_offsets["total"]
    original_offset, original_size = shard_offsets[loaded_shard_id]
    quantized_total = parameter.data.shape[0]
    return (
        original_size * quantized_total // total,
        original_offset * quantized_total // total,
    )


def adjust_scalar_to_fused_array(
    parameter_data: torch.Tensor,
    loaded_weight: torch.Tensor,
    shard_id: int | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    named = {"q": 0, "k": 1, "v": 2}
    if isinstance(shard_id, str):
        if shard_id not in named:
            raise ValueError(f"unknown scalar shard id {shard_id!r}")
        shard_id = named[shard_id]
    if loaded_weight.ndim != 0:
        if loaded_weight.shape[0] != 1:
            raise WeightLoadError("fused scalar checkpoint must contain one value")
        loaded_weight = loaded_weight[0]
    return parameter_data[shard_id], loaded_weight


def _adjust_packed_slice(
    parameter: Parameter,
    size: int,
    offset: int,
    output_dim: int,
) -> tuple[int, int]:
    packed_dim = getattr(parameter, "packed_dim", None)
    if packed_dim != output_dim:
        return size, offset
    factor = getattr(parameter, "packed_factor", 1)
    require_int(factor, "packed_factor", minimum=1)
    if size % factor or offset % factor:
        raise WeightLoadError(
            f"packed slice ({offset}, {size}) is not divisible by factor {factor}"
        )
    size //= factor
    offset //= factor
    return adjust_marlin_shard(parameter, size, offset)


def _narrow(tensor: torch.Tensor, dim: int, offset: int, size: int) -> torch.Tensor:
    if offset < 0 or size < 0 or offset + size > tensor.shape[dim]:
        raise WeightLoadError(
            f"slice ({offset}, {size}) exceeds dimension {dim} of shape {tuple(tensor.shape)}"
        )
    return tensor.narrow(dim, offset, size)


def _prepare_copy(
    destination: torch.Tensor,
    source: torch.Tensor,
    *,
    label: str,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    if destination.is_meta or source.is_meta:
        raise WeightLoadError("materialize meta storage before copying checkpoint weights")
    if source.ndim == 0 and destination.numel() == 1:
        source = source.reshape(destination.shape)
    if destination.shape != source.shape:
        raise WeightLoadError(
            f"{label}: destination shape {tuple(destination.shape)} does not match "
            f"source shape {tuple(source.shape)}"
        )
    return destination, source, label


def _commit_copy_plan(
    plan: list[tuple[torch.Tensor, torch.Tensor, str]],
) -> None:
    # Every destination/source shape is checked before this function runs, so a
    # bad later shard cannot leave an earlier shard partially updated.
    for destination, source, _label in plan:
        with torch.no_grad():
            destination.copy_(source.to(device=destination.device, dtype=destination.dtype))


def load_packed_output_parameter(
    parameter: Parameter,
    loaded_weight: torch.Tensor,
    layout: ProjectionLayout,
    loaded_shard_id: str | int | tuple[int, ...] | None = None,
) -> None:
    """Shape-validate every shard before copying full/local/logical packed storage.

    Packing metadata describes physical checkpoint and parameter dimensions. Copy
    or device failures are terminal; this is not a GPU transaction/rollback API.
    """

    output_dim = getattr(parameter, "output_dim", 0)
    require_int(output_dim, "output_dim")
    data = parameter.data
    if output_dim >= data.ndim or loaded_weight.ndim != data.ndim:
        raise WeightLoadError("checkpoint rank and output dimension must match the parameter")
    expected_width = sum(
        _adjust_packed_slice(parameter, part.local_size, part.local_offset, output_dim)[0]
        for part in layout.parts
    )
    if data.shape[output_dim] != expected_width:
        raise WeightLoadError(
            "parameter storage width does not match its declared projection layout"
        )
    if loaded_shard_id is None:
        parts = list(layout.parts)
    else:
        ids = loaded_shard_id if isinstance(loaded_shard_id, tuple) else (loaded_shard_id,)
        if not ids:
            raise WeightLoadError("shard IDs must not be empty")
        parts = [layout[name] for name in ids]
        canonical = [layout.parts.index(part) for part in parts]
        if any(right != left + 1 for left, right in zip(canonical, canonical[1:], strict=False)):
            raise WeightLoadError("tuple shard IDs must be consecutive and unique")

    destinations = [
        _adjust_packed_slice(parameter, part.local_size, part.local_offset, output_dim)
        for part in parts
    ]
    globals_ = [
        _adjust_packed_slice(parameter, part.global_size, 0, output_dim)[0] for part in parts
    ]
    local_width = sum(size for size, _ in destinations)
    global_width = sum(globals_)
    width = loaded_weight.shape[output_dim]
    if width not in (local_width, global_width):
        raise WeightLoadError(
            f"checkpoint storage width {width} must be local {local_width} or global {global_width}"
        )
    local_source = width == local_width
    source_cursor = 0
    plan: list[tuple[torch.Tensor, torch.Tensor, str]] = []
    for part, (size, offset), global_size in zip(parts, destinations, globals_, strict=True):
        destination = _narrow(data, output_dim, offset, size)
        if local_source:
            source_size, source_offset = size, source_cursor
            source_cursor += size
        else:
            source_size, relative_offset = _adjust_packed_slice(
                parameter,
                part.source_size,
                part.source_offset,
                output_dim,
            )
            source_offset = source_cursor + relative_offset
            source_cursor += global_size
        source = _narrow(loaded_weight, output_dim, source_offset, source_size)
        plan.append(_prepare_copy(destination, source, label=part.name))
    _commit_copy_plan(plan)


def load_row_parameter(
    parameter: Parameter,
    loaded_weight: torch.Tensor,
    *,
    rank: int,
    world_size: int,
) -> None:
    require_int(world_size, "world_size", minimum=1)
    require_int(rank, "rank")
    if rank >= world_size:
        raise WeightLoadError("rank must be smaller than world_size")
    input_dim = getattr(parameter, "input_dim", 1)
    require_int(input_dim, "input_dim")
    if input_dim >= parameter.ndim or loaded_weight.ndim != parameter.ndim:
        raise WeightLoadError("checkpoint rank and input dimension must match the parameter")
    data = parameter.data
    if data.shape == loaded_weight.shape:
        _commit_copy_plan([_prepare_copy(data, loaded_weight, label="row parameter")])
        return
    if loaded_weight.shape[input_dim] % world_size:
        raise WeightLoadError(
            f"row checkpoint input width {loaded_weight.shape[input_dim]} must be "
            f"divisible by world_size={world_size}"
        )
    shard_size = loaded_weight.shape[input_dim] // world_size
    source = _narrow(loaded_weight, input_dim, rank * shard_size, shard_size)
    _commit_copy_plan([_prepare_copy(data, source, label="row parameter")])


def build_original_offsets(sizes: list[int]) -> dict[str, tuple[int, int]]:
    offsets = list(itertools.accumulate([0] + sizes))
    result = {str(i): (offsets[i], size) for i, size in enumerate(sizes)}
    result["total"] = (sum(sizes), 0)
    return result
