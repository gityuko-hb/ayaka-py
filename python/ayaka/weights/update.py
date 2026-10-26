"""Weight-update transaction: all-new, all-old or failed (research C05 / R6).

The SGLang baseline updates the target before the draft, may flush or not, and
rebuilds its rollback iterator from an already-mutated config, so a partial
failure can leave a READY server serving a mix of generations. This module
applies one update to a set of live modules ("roles": target, draft, ...)
under a strict contract:

* the caller has drained every owner (no request, ticket or transfer); this
  module never checks that, it is the serving owner's job;
* every module is backed up (host copies) and digested before anything is
  touched; the old identity is captured before the source is opened;
* a source that raises or a probe that rejects the result triggers a restore
  of every role from the backup, which is accepted only if the restored digest
  equals the old one; the old readiness proof is then re-attested;
* without a verified backup, a failure after mutation can only be ``FAILED``:
  the weights are unknown and the caller must fail-stop, never report READY.

In-place loading keeps parameter storage (and so pointers) stable but changes
tensor versions, which invalidates any captured graph's weight binding; owners
with captured graphs must refuse before calling this.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch
from torch import nn

from ayaka.layers.base import BaseLayer
from ayaka.model_loader.readiness import finalize_loaded_weights

__all__ = [
    "CheckpointWeightSource",
    "StateDictWeightSource",
    "WeightSource",
    "WeightTransactionOutcome",
    "WeightUpdateStatus",
    "apply_weight_transaction",
    "reload_refusal",
    "weights_digest",
]


class WeightUpdateStatus(StrEnum):
    """Outcome of one update. Only ``COMMITTED`` serves new weights."""

    #: Every role carries the new weights and a fresh readiness proof.
    COMMITTED = "committed"
    #: The update failed and every role was restored and verified bit-identical.
    ROLLED_BACK = "rolled_back"
    #: Refused before draining or touching anything (capability or input).
    REFUSED = "refused"
    #: The owners never became quiescent before the deadline; nothing was touched.
    TIMEOUT = "timeout"
    #: Weights may be partially updated; the server must stop (fail-closed).
    FAILED = "failed"


@runtime_checkable
class WeightSource(Protocol):
    """Where new weights come from; ``apply`` mutates the module in place."""

    identity: str

    def apply(self, module: nn.Module) -> None: ...


@dataclass(frozen=True, slots=True)
class StateDictWeightSource:
    """Copy tensors by name; names, shapes and dtypes must match exactly.

    Validation runs before the first copy, so a mismatch never mutates.
    """

    state: Mapping[str, torch.Tensor]
    identity: str = "state_dict"

    def apply(self, module: nn.Module) -> None:
        targets = dict((*module.named_parameters(), *module.named_buffers()))
        missing = sorted(set(targets) - set(self.state))
        unexpected = sorted(set(self.state) - set(targets))
        if missing or unexpected:
            raise ValueError(f"state dict mismatch: missing={missing} unexpected={unexpected}")
        for name, source in self.state.items():
            target = targets[name]
            if source.shape != target.shape or source.dtype != target.dtype:
                raise ValueError(
                    f"{name}: source {tuple(source.shape)}/{source.dtype} does not match "
                    f"{tuple(target.shape)}/{target.dtype}"
                )
        with torch.no_grad():
            for name, source in self.state.items():
                targets[name].copy_(source)


@dataclass(frozen=True, slots=True)
class CheckpointWeightSource:
    """Load a safetensors checkpoint through a model family's validating loader.

    ``loader`` is e.g. ``ayaka.models.qwen2.load_qwen2_weights``; it validates
    the manifest against the model schema before reading payloads, but a read
    error can still happen after some tensors were copied, which is what the
    transaction's backup is for.
    """

    path: Path
    loader: Callable[[Any, Path], object]
    identity: str = ""

    def apply(self, module: nn.Module) -> None:
        self.loader(module, self.path)


def _named_tensors(module: nn.Module) -> list[tuple[str, torch.Tensor]]:
    return [*module.named_parameters(), *module.named_buffers()]


def weights_digest(module: nn.Module) -> str:
    """SHA-256 over every parameter/buffer name, dtype, shape and byte.

    Copies device tensors to host: an administrative cost, never a hot path.
    """
    digest = hashlib.sha256()
    for name, tensor in _named_tensors(module):
        data = tensor.detach().to("cpu").contiguous()
        digest.update(f"{name}|{data.dtype}|{tuple(data.shape)}\0".encode())
        digest.update(data.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _backup(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().to("cpu", copy=True) for name, tensor in _named_tensors(module)}


def _restore(module: nn.Module, backup: Mapping[str, torch.Tensor]) -> None:
    targets = dict(_named_tensors(module))
    if set(targets) != set(backup):
        raise RuntimeError("module structure changed during the update; cannot restore")
    with torch.no_grad():
        for name, saved in backup.items():
            targets[name].copy_(saved)


def reload_refusal(module: nn.Module) -> str | None:
    """Why weights of ``module`` cannot be reloaded in place, or ``None``."""
    for owner in module.modules():
        if isinstance(owner, BaseLayer) and owner.quant_config is not None:
            return "quantized layers are packed once at load; in-place reload is not supported"
    if any(tensor.is_meta for _, tensor in _named_tensors(module)):
        return "module has unmaterialized (meta) tensors"
    return None


@dataclass(frozen=True, slots=True)
class WeightTransactionOutcome:
    status: WeightUpdateStatus
    #: Per role: digest before and after the transaction.
    old_digests: Mapping[str, str] = field(default_factory=dict)
    new_digests: Mapping[str, str] = field(default_factory=dict)
    #: Whether any role's bytes differed from the old ones after the failure
    #: (False: the source failed before touching anything).
    mutated: bool = False
    backup_bytes: int = 0
    load_seconds: float = 0.0
    detail: str = ""


def apply_weight_transaction(
    roles: Mapping[str, nn.Module],
    sources: Mapping[str, WeightSource],
    *,
    backup: bool = True,
    probe: Callable[[Mapping[str, nn.Module]], None] | None = None,
) -> WeightTransactionOutcome:
    """Apply ``sources`` to ``roles`` (same keys) as one all-or-nothing update.

    Roles load in mapping order. ``probe`` (e.g. golden logits) runs after every
    role loaded and may raise to reject the result. The caller must hold every
    role quiescent: no forward may run until this returns.
    """
    if set(roles) != set(sources) or not roles:
        raise ValueError("every role needs exactly one weight source")
    for role, module in roles.items():
        reason = reload_refusal(module)
        if reason is not None:
            return WeightTransactionOutcome(WeightUpdateStatus.REFUSED, detail=f"{role}: {reason}")
    old_digests = {role: weights_digest(module) for role, module in roles.items()}
    saved = {role: _backup(module) for role, module in roles.items()} if backup else None
    backup_bytes = (
        sum(t.numel() * t.element_size() for tensors in saved.values() for t in tensors.values())
        if saved is not None
        else 0
    )
    started = time.perf_counter()
    try:
        for role, module in roles.items():
            sources[role].apply(module)
        if probe is not None:
            probe(roles)
        for module in roles.values():
            finalize_loaded_weights(module)
    except Exception as exc:  # noqa: BLE001 - every failure must reach the outcome
        elapsed = time.perf_counter() - started
        current = {role: weights_digest(module) for role, module in roles.items()}
        mutated = current != old_digests
        detail = f"{type(exc).__name__}: {exc}"
        if saved is None:
            if not mutated:
                _reattest(roles)
                return WeightTransactionOutcome(
                    WeightUpdateStatus.ROLLED_BACK,
                    old_digests,
                    current,
                    mutated=False,
                    load_seconds=elapsed,
                    detail=detail,
                )
            return WeightTransactionOutcome(
                WeightUpdateStatus.FAILED,
                old_digests,
                current,
                mutated=True,
                load_seconds=elapsed,
                detail=f"{detail}; no backup to restore from",
            )
        try:
            for role, module in roles.items():
                _restore(module, saved[role])
            restored = {role: weights_digest(module) for role, module in roles.items()}
        except Exception as restore_error:  # noqa: BLE001
            return WeightTransactionOutcome(
                WeightUpdateStatus.FAILED,
                old_digests,
                current,
                mutated=mutated,
                backup_bytes=backup_bytes,
                load_seconds=elapsed,
                detail=f"{detail}; restore failed: {restore_error}",
            )
        if restored != old_digests:
            return WeightTransactionOutcome(
                WeightUpdateStatus.FAILED,
                old_digests,
                restored,
                mutated=mutated,
                backup_bytes=backup_bytes,
                load_seconds=elapsed,
                detail=f"{detail}; restored weights do not match the old digest",
            )
        _reattest(roles)
        return WeightTransactionOutcome(
            WeightUpdateStatus.ROLLED_BACK,
            old_digests,
            restored,
            mutated=mutated,
            backup_bytes=backup_bytes,
            load_seconds=elapsed,
            detail=detail,
        )
    new_digests = {role: weights_digest(module) for role, module in roles.items()}
    return WeightTransactionOutcome(
        WeightUpdateStatus.COMMITTED,
        old_digests,
        new_digests,
        mutated=new_digests != old_digests,
        backup_bytes=backup_bytes,
        load_seconds=time.perf_counter() - started,
    )


def _reattest(roles: Mapping[str, nn.Module]) -> None:
    """Restore a readiness proof for verified old weights (the loader cleared it)."""
    for module in roles.values():
        finalize_loaded_weights(module)
