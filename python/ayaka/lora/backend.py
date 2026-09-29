"""Backend contract: execution plans, routing context and the reference kernel.

The reference backend is the permanent correctness oracle. It reads every
stable slot before masking, so its read footprint is ``all_slots``: management
copies must drain all reader streams before writing any slot. Optimized
backends declare ``selected_slots`` once their kernels only touch routed slots.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import torch

from ayaka.utils.import_utils import has_module

__all__ = [
    "BackendCapability",
    "BackendSelection",
    "BackendUnavailableError",
    "LoRAExecutionPlan",
    "LoRAExpandContext",
    "LoRABackend",
    "LoRAWorkspace",
    "LoRAWorkspaceSpec",
    "ReadFootprint",
    "TorchLoRABackend",
    "create_backend",
]


class ReadFootprint(StrEnum):
    """Which stable slots a backend may read for one forward."""

    ALL_SLOTS = "all_slots"
    SELECTED_SLOTS = "selected_slots"


class BackendUnavailableError(RuntimeError):
    """The requested backend cannot serve this build, hardware or phase."""


class LoRAWorkspace:
    def __init__(
        self,
        tokens: int,
        rank: int,
        output: int,
        device: torch.device,
        dtype: torch.dtype,
        *,
        include_mask: bool = True,
    ) -> None:
        self.tokens, self.rank, self.output = tokens, rank, output
        self.low = torch.empty(tokens * rank, dtype=dtype, device=device)
        self.delta = torch.empty(tokens * output, dtype=dtype, device=device)
        self.mask = torch.empty(tokens, dtype=torch.bool, device=device) if include_mask else None

    @property
    def storage_identity(self) -> tuple[int, ...]:
        tensors = (self.low, self.delta) if self.mask is None else (self.low, self.delta, self.mask)
        return tuple(t.data_ptr() for t in tensors)

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.low, self.delta)) + (
            0 if self.mask is None else self.mask.numel() * self.mask.element_size()
        )


@dataclass(frozen=True, slots=True)
class LoRAWorkspaceSpec:
    """Capture-safe scratch sizing; all backing is allocated before capture."""

    tokens: int
    rank: int
    output: int
    dtype: torch.dtype
    include_mask: bool = True

    @property
    def identity(self) -> tuple[int, int, int, str, bool]:
        return (self.tokens, self.rank, self.output, str(self.dtype), self.include_mask)

    @property
    def per_flight_bytes(self) -> int:
        element = torch.empty((), dtype=self.dtype).element_size()
        bulk = self.tokens * (self.rank + self.output) * element
        return bulk + (self.tokens if self.include_mask else 0)

    def allocate(self, device: torch.device) -> LoRAWorkspace:
        return LoRAWorkspace(
            self.tokens, self.rank, self.output, device, self.dtype, include_mask=self.include_mask
        )


@dataclass(frozen=True, slots=True)
class LoRAExecutionPlan:
    """Pre-launch support decision; must not contain adapter name or revision."""

    backend: str
    version: str
    phase: str
    algorithm: str
    rank: int
    capacity: int
    read_footprint: ReadFootprint
    workspace: LoRAWorkspaceSpec

    @property
    def identity(self) -> tuple[object, ...]:
        return (
            self.backend,
            self.version,
            self.phase,
            self.algorithm,
            self.rank,
            self.capacity,
            str(self.read_footprint),
            self.workspace.identity,
        )


@dataclass(frozen=True, slots=True)
class LoRAExpandContext:
    """Per-flight routing contract handed to the backend inside one forward.

    Only ``rows`` and ``workspace`` are required by the reference backend.
    Grouped kernels additionally read the stable permutation/offset tables and
    the per-slot counts; ``phase`` records which planned algorithm this forward
    was staged for, so a captured graph never mixes topologies.
    """

    rows: torch.Tensor
    workspace: LoRAWorkspace
    sequence_slots: torch.Tensor | None = None
    segment_offsets: torch.Tensor | None = None
    token_permutation: torch.Tensor | None = None
    token_inverse: torch.Tensor | None = None
    projected_rows: torch.Tensor | None = None
    slot_counts: torch.Tensor | None = None
    phase: str = "reference"


class LoRABackend(Protocol):
    name: str
    version: str
    read_footprint: ReadFootprint

    def plan(
        self,
        *,
        tokens: int,
        rank: int,
        output: int,
        capacity: int,
        dtype: torch.dtype,
        phase: str = "reference",
    ) -> LoRAExecutionPlan: ...

    def run_expand(
        self,
        context: LoRAExpandContext,
        x: torch.Tensor,
        output: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        ranks: torch.Tensor,
        offset: int,
    ) -> None: ...

    def run_expand_slice(
        self,
        x: torch.Tensor,
        output: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        rows: torch.Tensor,
        ranks: torch.Tensor,
        workspace: LoRAWorkspace,
        offset: int,
    ) -> None: ...


class TorchLoRABackend:
    name = "torch_reference"
    version = "1"
    read_footprint = ReadFootprint.ALL_SLOTS

    def plan(
        self,
        *,
        tokens: int,
        rank: int,
        output: int,
        capacity: int,
        dtype: torch.dtype,
        phase: str = "reference",
    ) -> LoRAExecutionPlan:
        if tokens < 1 or rank < 1 or output < 1 or capacity < 1:
            raise ValueError("LoRA plan requires positive tokens/rank/output/capacity")
        return LoRAExecutionPlan(
            backend=self.name,
            version=self.version,
            phase=phase,
            algorithm="slot_loop",
            rank=rank,
            capacity=capacity,
            read_footprint=self.read_footprint,
            workspace=LoRAWorkspaceSpec(tokens, rank, output, dtype),
        )

    def run_expand(
        self,
        context: LoRAExpandContext,
        x: torch.Tensor,
        output: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        ranks: torch.Tensor,
        offset: int,
    ) -> None:
        self.run_expand_slice(x, output, a, b, context.rows, ranks, context.workspace, offset)

    @torch.no_grad()
    def run_expand_slice(
        self,
        x: torch.Tensor,
        output: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        rows: torch.Tensor,
        ranks: torch.Tensor,
        workspace: LoRAWorkspace,
        offset: int,
    ) -> None:
        count = x.shape[0]
        if (
            x.ndim != 2
            or count != rows.numel()
            or count > workspace.tokens
            or x.shape[1] != a.shape[2]
            or x.dtype != a.dtype
            or output.dtype != b.dtype
        ):
            raise ValueError("LoRA rows/dtype/shape must match packed token rows and workspace")
        rank, width = a.shape[1], b.shape[1]
        low = workspace.low[: count * rank].view(count, rank)
        delta = workspace.delta[: count * width].view(count, width)
        mask = workspace.mask
        if mask is None:
            raise ValueError("torch reference LoRA requires a mask workspace")
        mask = mask[:count]
        destination = output[:, offset : offset + width]
        for slot in range(1, a.shape[0]):
            torch.mm(x, a[slot].T, out=low)
            torch.mm(low, b[slot].T, out=delta)
            torch.ne(rows, slot, out=mask)
            # Mask by assignment: NaN in an unused/quarantined slot must not
            # contaminate base or other adapters through NaN * 0.
            delta.masked_fill_(mask.unsqueeze(1), 0)
            destination.add_(delta)


@dataclass(frozen=True, slots=True)
class BackendCapability:
    """Hardware/build verdict for backend selection, with a human-readable reason."""

    cuda: bool
    triton: bool

    @classmethod
    def probe(cls) -> BackendCapability:
        return cls(cuda=torch.cuda.is_available(), triton=has_module("triton"))

    def triton_reason(self) -> str | None:
        if not self.triton:
            return "triton module is not installed"
        if not self.cuda:
            return "CUDA device is not available"
        return None


@dataclass(frozen=True, slots=True)
class BackendSelection:
    backend: LoRABackend
    requested: str
    fallback_reason: str | None = None


def _triton_backend(
    capability: BackendCapability, device: torch.device | None = None
) -> LoRABackend:
    reason = capability.triton_reason()
    if reason is None and device is not None and device.type != "cuda":
        reason = "triton LoRA backend requires a CUDA device"
    if reason is not None:
        raise BackendUnavailableError(f"triton LoRA backend unavailable: {reason}")
    from ayaka.lora.triton_backend import TritonLoRABackend

    return TritonLoRABackend()


def create_backend(
    name: str,
    *,
    capability: BackendCapability | None = None,
    device: torch.device | None = None,
) -> BackendSelection:
    """Resolve the backend before any kernel enqueue or slot mutation.

    ``triton`` raises when unsupported; ``auto`` falls back to the reference
    backend and reports the reason for accounting. Backend availability never
    changes mid-capture because it is resolved once at bootstrap.
    """
    if name not in ("torch_reference", "triton", "auto"):
        raise ValueError(f"unknown LoRA backend {name!r}")
    capability = capability or BackendCapability.probe()
    if name == "torch_reference":
        return BackendSelection(TorchLoRABackend(), name)
    if name == "triton":
        return BackendSelection(_triton_backend(capability, device), name)
    try:
        return BackendSelection(_triton_backend(capability, device), name)
    except BackendUnavailableError as exc:
        return BackendSelection(TorchLoRABackend(), name, str(exc))
