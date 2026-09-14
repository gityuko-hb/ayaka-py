"""Bind checkpoint tensors to module parameters using explicit architecture mappings.

This adapter owns no runtime memory budget. It reads at most one selected tensor
at a time through BoundedCheckpointReader and writes registered module storage.
The composition root supplies mappings and optional physical checkpoint slices.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass

import torch
from torch import nn

from ayaka.layers.base import BaseLayer
from ayaka.layers.quantization.base import QuantizationTarget
from ayaka.model_loader.reader import BoundedCheckpointReader
from ayaka.model_loader.validate import verify_consumed_names
from ayaka.types import DType
from ayaka.utils.torch_utils import torch_dtype
from ayaka.weights.plan import CheckpointManifest, FileReadPlan, ManifestEntry, TensorSlice


@dataclass(frozen=True)
class WeightBinding:
    """Checkpoint key -> module tensor path and optional logical projection ID."""

    parameter: str
    shard_id: str | int | tuple[str | int, ...] | None = None


def _tensors(module: nn.Module) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = dict(module.named_parameters(remove_duplicate=False))
    for path, owner in module.named_modules():
        for name, value in owner.named_buffers(recurse=False):
            if name not in owner._non_persistent_buffers_set:
                tensors[f"{path}.{name}" if path else name] = value
    return tensors


def materialize_module_weights(
    module: nn.Module, *, device: torch.device | str | None = None
) -> None:
    """Materialize meta tensors while preserving loader metadata and tied aliases.

    Quantized tensors use their owner's validated final placement. Dense meta
    tensors require an explicit device. Already materialized tensors are unchanged.
    Call before loading, never after quantization finalization.
    """
    replacements: dict[int, torch.Tensor] = {}
    for owner in module.modules():
        quant_context = getattr(owner, "quant_context", None)
        placement = quant_context.device if quant_context is not None else device
        for name, tensor in (*owner._parameters.items(), *owner._buffers.items()):
            if tensor is None or not tensor.is_meta:
                continue
            if placement is None or torch.device(placement).type == "meta":
                raise ValueError("a real execution device is required to materialize meta weights")
            if getattr(owner, "_quantization_state", None) in ("ready", "failed"):
                raise RuntimeError("cannot materialize finalized or failed quantized storage")
            if isinstance(owner, BaseLayer):
                context = owner.runtime_context(
                    placement,
                    quant_context.activation_dtype if quant_context is not None else tensor.dtype,
                    target=quant_context.target
                    if quant_context is not None
                    else QuantizationTarget.LINEAR,
                )
                placement = context.device
            replacement = replacements.get(id(tensor))
            if replacement is None:
                replacement = torch.empty_like(tensor, device=placement)
                if isinstance(tensor, nn.Parameter):
                    replacement = nn.Parameter(replacement, requires_grad=tensor.requires_grad)
                for attr, value in tensor.__dict__.items():
                    setattr(replacement, attr, value)
                replacements[id(tensor)] = replacement
            elif (
                replacement.device != torch.device(placement)
                and torch.device(placement).index is not None
            ):
                raise ValueError("tied meta tensors have conflicting placements")
            setattr(owner, name, replacement)


def _bindings(
    module: nn.Module,
    bindings: Mapping[str, WeightBinding] | None,
) -> tuple[dict[str, torch.Tensor], dict[str, WeightBinding]]:
    tensors = _tensors(module)
    if bindings is None:
        # One checkpoint key per tied tensor unless the architecture maps aliases explicitly.
        seen: set[int] = set()
        result: dict[str, WeightBinding] = {}
        for name, tensor in tensors.items():
            if id(tensor) not in seen:
                result[name] = WeightBinding(name)
                seen.add(id(tensor))
    else:
        result = dict(bindings)
    coverage: dict[int, set[str] | None] = {}
    for binding in result.values():
        if binding.parameter not in tensors:
            raise KeyError(f"unknown module tensor {binding.parameter!r}")
        tensor = tensors[binding.parameter]
        path, _, _name = binding.parameter.rpartition(".")
        owner = module.get_submodule(path) if path else module
        layout = getattr(owner, "layout", None)
        if binding.shard_id is None:
            if id(tensor) in coverage:
                raise ValueError(f"overlapping bindings for {binding.parameter!r}")
            coverage[id(tensor)] = None
        else:
            if layout is None:
                raise ValueError("logical shard bindings require a projection layout")
            ids = binding.shard_id if isinstance(binding.shard_id, tuple) else (binding.shard_id,)
            names = {layout[key].name for key in ids}
            if len(names) != len(ids) or not names:
                raise ValueError("shard bindings must be nonempty and unique")
            if id(tensor) in coverage and coverage[id(tensor)] is None:
                raise ValueError("full and partial bindings overlap")
            current = coverage.setdefault(id(tensor), set())
            assert current is not None
            if current & names:
                raise ValueError("logical shard bindings overlap")
            current.update(names)
    for name, tensor in tensors.items():
        if id(tensor) not in coverage:
            raise ValueError(f"no checkpoint binding for {name!r}")
        parts = coverage[id(tensor)]
        if parts is not None:
            path, _, _ = name.rpartition(".")
            owner = module.get_submodule(path) if path else module
            layout = getattr(owner, "layout", None)
            if layout is None or parts != set(layout.names):
                raise ValueError(f"incomplete logical projection bindings for {name!r}")
    return tensors, result


def load_module_weights(
    module: nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    bindings: Mapping[str, WeightBinding] | None = None,
    device: torch.device | str | None = None,
    finalize: bool = True,
) -> frozenset[str]:
    """Load every required parameter/buffer and optionally finalize layer methods.

    Unknown, missing, duplicate and overlapping bindings raise. A custom parameter
    weight_loader owns storage/shard interpretation; otherwise shapes must match.
    Pre-sharded tensors remain local when their shape matches destination storage.
    Model-wide I/O/copy failures do not roll back earlier writes: recreate/reload
    the model after failure. CUDA copies/packing are ordered on the current stream;
    the runtime must join that stream before publishing to other execution streams.
    """
    _bindings(module, bindings)  # Fail on incomplete architecture maps before allocation.
    materialize_module_weights(module, device=device)
    tensors, mapping = _bindings(module, bindings)
    for owner in module.modules():
        if isinstance(owner, BaseLayer) and owner.quant_config is not None:
            if owner._quantization_state != "created":
                raise RuntimeError(
                    "quantized weights must be loaded before finalization, exactly once"
                )
    consumed: set[str] = set()
    for key, source in weights:
        if key not in mapping:
            raise KeyError(f"unexpected checkpoint tensor {key!r}")
        if key in consumed:
            raise ValueError(f"duplicate checkpoint tensor {key!r}")
        binding = mapping[key]
        destination = tensors[binding.parameter]
        if source.is_meta or destination.is_meta:
            raise ValueError("checkpoint copies require materialized tensors")
        loader = getattr(destination, "weight_loader", None)
        if loader is not None:
            loader(destination, source, binding.shard_id)
        else:
            if binding.shard_id is not None or source.shape != destination.shape:
                raise ValueError(f"checkpoint shape does not match {binding.parameter!r}")
            with torch.no_grad():
                destination.copy_(source.to(destination))
        consumed.add(key)
    verify_consumed_names(frozenset(mapping), consumed, context="module weight loading")
    if finalize:
        for owner in module.modules():
            if isinstance(owner, BaseLayer):
                owner.process_weights_after_loading()
    return frozenset(consumed)


def _read_plan(
    entry: ManifestEntry, selection: TensorSlice | None
) -> tuple[FileReadPlan | None, tuple[int, ...]]:
    if entry.dtype.sub_byte:
        raise ValueError("sub-byte checkpoint entries require a format-specific unpacking adapter")
    shape = entry.shape
    if selection is None:
        plan = (
            FileReadPlan(entry.file_uri, entry.byte_offset, entry.nbytes) if entry.nbytes else None
        )
        return plan, shape
    dim, start, stop = selection.dim, selection.start, selection.stop
    if dim >= len(shape) or stop > shape[dim]:
        raise ValueError(f"checkpoint slice is outside {entry.tensor_key!r}")
    outer = math.prod(shape[:dim])
    inner_bytes = math.prod(shape[dim + 1 :]) * (entry.dtype.bits // 8)
    run_bytes = (stop - start) * inner_bytes
    selected = (*shape[:dim], stop - start, *shape[dim + 1 :])
    if not run_bytes or not outer:
        return None, selected
    offset = entry.byte_offset + start * inner_bytes
    if outer == 1 or stop - start == shape[dim]:
        return FileReadPlan(entry.file_uri, offset, outer * run_bytes), selected
    return FileReadPlan(
        entry.file_uri,
        offset,
        outer * run_bytes,
        run_bytes=run_bytes,
        stride_bytes=shape[dim] * inner_bytes,
        run_count=outer,
    ), selected


def iter_checkpoint_tensors(
    manifest: CheckpointManifest,
    *,
    selections: Mapping[str, TensorSlice] | None = None,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Read validated manifest entries, optionally slicing physical storage before I/O.

    Slices are explicit: architecture/planning owns rank/head semantics. Integer
    packed storage (e.g. int32) is read verbatim; it is never implicitly dequantized.
    """
    selections = selections or {}
    if set(selections) - manifest.tensor_keys:
        raise KeyError("checkpoint selections contain unknown keys")
    with BoundedCheckpointReader(workers=1, max_inflight=1) as reader:
        for entry in manifest.entries:
            plan, shape = _read_plan(entry, selections.get(entry.tensor_key))
            dtype = torch_dtype(
                {
                    DType.FP64: "float64",
                    DType.FP32: "float32",
                    DType.FP16: "float16",
                    DType.BF16: "bfloat16",
                    DType.FP8_E4M3: "float8_e4m3fn",
                    DType.FP8_E5M2: "float8_e5m2",
                    DType.INT64: "int64",
                    DType.INT32: "int32",
                    DType.INT8: "int8",
                    DType.UINT8: "uint8",
                    DType.BOOL: "bool",
                }[entry.dtype]
            )
            if plan is None:
                tensor = torch.empty(shape, dtype=dtype)
            else:
                payload = bytearray(reader.read_one(plan))
                tensor = torch.frombuffer(payload, dtype=dtype).reshape(shape)
            yield entry.tensor_key, tensor


def load_checkpoint(
    module: nn.Module,
    manifest: CheckpointManifest,
    *,
    bindings: Mapping[str, WeightBinding] | None = None,
    selections: Mapping[str, TensorSlice] | None = None,
    device: torch.device | str | None = None,
) -> frozenset[str]:
    """Validate names before I/O, load selected storage, then finalize the module."""
    _, mapping = _bindings(module, bindings)
    verify_consumed_names(frozenset(mapping), manifest.tensor_keys, context="checkpoint manifest")
    return load_module_weights(
        module,
        iter_checkpoint_tensors(manifest, selections=selections),
        bindings=mapping,
        device=device,
    )
