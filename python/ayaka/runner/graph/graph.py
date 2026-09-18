"""Capture/replay protocol for CUDA graph backends.

Defines the contract every graph backend, and every graph-capturable
attention/metadata implementation, must satisfy. Per Ayaka's protocol
invariant, this module imports nothing else from `ayaka` — concrete
buffer storage (ayaka.device.graph.arena.GraphBufferArena) only has to
satisfy `BufferSource` structurally; it is never imported here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class ShapeKey:
    """Identifies one captured CUDA-graph shape.

    size: per-phase capture size (decode: bs, prefill: num_tokens).
    stream_idx: pdmux-style stream group; None for single-stream runners.
    variant_label: e.g. "lora" / "nolora" when a runner records a
        separate graph per variant to avoid an in-graph branch.
    quant_mode: kernel-variant selector (e.g. "w4a16_marlin", "fp8",
        "mxfp4") for when the chosen GEMM/attention kernel changes
        graph structure at the same (size, stream_idx). Reserved from
        day one so the dict-key type this backs never has to change
        later — see ayaka-marlin's W4A16 / FP8 / MXFP4 / MXFP8 paths.
    """

    size: int
    stream_idx: int | None = None
    variant_label: str | None = None
    quant_mode: str | None = None


class CanRun(Enum):
    """Result of a capability check for one (batch, shape_key) pair.

    RUNNABLE and NOT_CAPTURED are both *expected* routing outcomes —
    callers fall back to eager silently for either. CAPABILITY_MISMATCH
    means the caller asked for something this backend was never
    captured to serve (e.g. a hidden_mode or quant_mode richer than
    what this shape_key was captured with). Per invariant I4, that
    must not be swallowed into a silent fallback: raise
    GraphCapabilityError instead of returning NOT_CAPTURED.
    """

    RUNNABLE = auto()
    NOT_CAPTURED = auto()
    CAPABILITY_MISMATCH = auto()


class GraphCapabilityError(RuntimeError):
    """A batch requires something the graph subsystem cannot serve for
    a reason other than "not captured yet". Never caught silently —
    see CanRun.CAPABILITY_MISMATCH and invariant I4."""


@runtime_checkable
class BufferSource(Protocol):
    """Structural stand-in for GraphBufferArena. Anything with a
    ``.slot(name)`` satisfies this — no import of the concrete arena
    needed here."""

    def slot(self, name: str) -> Any: ...


class MetadataPlanner(Protocol):
    """Two-step contract an attention/metadata implementation must
    satisfy to be graph-capturable.

    plan_out_of_graph runs BEFORE capture_session opens (and, at
    replay time, before backend.replay()) and may do arbitrary
    CPU-side / dynamic-shape work — radix-trie block lookup,
    prefix-cache hits, Python-level branching. Its only obligation is
    to write results into the *stable* buffers owned by the arena,
    never to return fresh tensors.

    bind_in_graph runs INSIDE the captured region (recorded once at
    capture, replayed unchanged thereafter) and may only perform pure
    GPU ops that read those stable buffers. If an attention
    implementation cannot express its metadata prep this way — e.g.
    the number of kernel launches depends on a runtime value — it does
    not belong behind a Full-style backend; route it to a
    segmented/breakable backend instead.
    """

    def plan_out_of_graph(self, arena: BufferSource, shape_key: ShapeKey) -> None: ...

    def bind_in_graph(self, arena: BufferSource, shape_key: ShapeKey) -> None: ...


class GraphBackend(ABC):
    """Pure interface: no state, no defaults. Each concrete backend
    owns its own captured artifacts and binds whatever handles it
    needs from the arena passed at construction time.
    """

    @abstractmethod
    def capture_session(self, stream: Any) -> AbstractContextManager[None]: ...

    @abstractmethod
    def capture_one(
        self,
        shape_key: ShapeKey,
        forward_fn: Callable[[], Any],
        post_warmup_hook: Callable[[], None] | None = None,
    ) -> None: ...

    @abstractmethod
    def can_run(
        self, shape_key: ShapeKey, *, requested_hidden_mode: str | None = None
    ) -> CanRun: ...

    @abstractmethod
    def replay_session(self) -> AbstractContextManager[None]: ...

    @abstractmethod
    def replay(self, shape_key: ShapeKey, **kwargs: Any) -> Any: ...

    @abstractmethod
    def cleanup(self) -> None: ...
