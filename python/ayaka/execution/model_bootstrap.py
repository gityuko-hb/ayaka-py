"""Weight readiness proof from the loader, ordered before any model execution."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from uuid import uuid4

import torch
from torch import nn

from ayaka.execution.input_buffers import TensorBinding
from ayaka.layers.base import BaseLayer


class BootstrapStage(StrEnum):
    MODEL_LOADED = "model_loaded"
    WEIGHTS_FINALIZED = "weights_finalized"
    RESOURCES_BOUND = "resources_bound"
    CAPTURE_READY = "capture_ready"
    WORKER_READY = "worker_ready"
    FAILED = "failed"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class WeightReadiness:
    """Loader completion evidence; retains a producer event, never weight storage."""

    finalized: bool
    event: torch.cuda.Event | None
    binding: str
    revision: str


def weight_binding(model: nn.Module) -> str:
    """Include storage/layout/version, so reload or replacement cannot replay stale weights."""
    parts = []
    for name, tensor in (*model.named_parameters(), *model.named_buffers()):
        if tensor.is_meta:
            raise ValueError(f"model tensor {name!r} is still on meta")
        try:
            version = tensor._version
        except RuntimeError:  # tensors created under inference_mode have no counter
            version = None
        parts.append((name, TensorBinding.of(tensor), version))
    return hashlib.sha256(repr(parts).encode()).hexdigest()[:24]


def finalize_loaded_weights(model: nn.Module, *, finalized: bool = True) -> WeightReadiness:
    """Attest a completed external load or the canonical loader's successful result.

    External loaders must join their packing streams onto the current stream
    first. A device-wide bootstrap barrier additionally covers asynchronous
    packing implementations whose hooks do not expose a completion event.
    No such synchronization is performed per replay.
    """
    tensors = (*model.parameters(), *model.buffers())
    if not tensors or any(t.is_meta for t in tensors):
        raise ValueError("weights must be fully materialized before finalization")
    devices = {t.device for t in tensors}
    if len(devices) != 1:
        raise ValueError("EP0 requires all model tensors on one device")
    for module in model.modules():
        if (
            isinstance(module, BaseLayer)
            and module.quant_config is not None
            and module._quantization_state != "ready"
            and finalized
        ):
            raise ValueError("quantized weights have not been finalized")
    device = next(iter(devices))
    event = None
    if device.type == "cuda" and finalized:
        torch.cuda.synchronize(device)
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(device))
    proof = WeightReadiness(finalized, event, weight_binding(model), uuid4().hex)
    object.__setattr__(model, "_ayaka_weight_readiness", proof)
    return proof


def require_ready_weights(model: nn.Module) -> WeightReadiness:
    """Reject missing/partial/failed loads; order the current execution stream."""
    proof = getattr(model, "_ayaka_weight_readiness", None)
    if not isinstance(proof, WeightReadiness) or not proof.finalized:
        raise ValueError("load and finalize model weights before execution bootstrap")
    if proof.binding != weight_binding(model):
        raise ValueError("model weights changed after finalization; finalize the new binding")
    if proof.event is not None:
        tensor = next(iter((*model.parameters(), *model.buffers())))
        torch.cuda.current_stream(tensor.device).wait_event(proof.event)
    return proof


def adopt_model_weights(model: nn.Module) -> WeightReadiness:
    """Honor ServingRuntime's existing externally materialized model contract.

    Passing a model to a NEW runtime is the external loader's handoff. Models
    created/initialized by callers, or moved before bootstrap, receive a fresh
    placement proof here. A failed/unfinished canonical load is distinguishable
    and can never be rescued by this compatibility bridge. Live graph owners
    use strict binding checks; this is not a hot reload operation.
    """
    if hasattr(model, "_ayaka_weight_readiness"):
        proof = model._ayaka_weight_readiness
        if not isinstance(proof, WeightReadiness) or not proof.finalized:
            raise ValueError("cannot adopt failed or unfinalized model weights")
        if proof.binding == weight_binding(model):
            return require_ready_weights(model)
    return finalize_loaded_weights(model)
