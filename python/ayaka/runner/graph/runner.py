"""Graph-buffer arena, shape-bucket selection, and the reference
decode-runner orchestration that drives ayaka.executor.graph.backends.

Division of labor, preserved from the original files:
  - GraphBufferArena / ArenaSlot: static, address-stable buffers —
    every graph-capturable forward's input must come from here.
  - pad_to_bucket / build_buckets: bucket selection, shared by any
    phase (decode, prefill, ...).
  - GraphBackendKind / resolve_backend: the one place a new backend
    kind gets wired in.
  - DecodeGraphRunner: reference orchestration for the decode phase.
    Deliberately one concrete class, not an abstract base with only
    one subclass — the abstraction for a second phase should be
    extracted once that phase's actual needs are known.
"""

from __future__ import annotations

import bisect
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import torch

from ayaka.runner.graph.backend import BreakableGraphBackend, FullGraphBackend, slice_rows
from ayaka.runner.graph.graph import CanRun, GraphBackend, GraphCapabilityError, ShapeKey

logger = logging.getLogger(__name__)


@dataclass
class ArenaSlot:
    storage: torch.Tensor  # full-capacity static buffer; data_ptr is load-bearing

    def slice_for(self, n: int) -> torch.Tensor:
        """View onto the first n rows — same address, smaller logical size."""
        return self.storage[:n]


class GraphBufferArena:
    """Owns one static, address-stable tensor per named field.

    register() is meant to be called once per field, during runner
    construction, before any capture_session opens. slot() is what
    MetadataPlanner implementations and GraphBackend implementations
    read/write against afterwards — via `BufferSource` in
    ayaka.protocol.graph, never by importing this class directly.
    """

    def __init__(self) -> None:
        self._slots: dict[str, ArenaSlot] = {}

    def register(self, name: str, storage: torch.Tensor) -> None:
        if name in self._slots:
            raise ValueError(
                f"arena slot {name!r} already registered — re-registering "
                "would move the address a captured graph depends on"
            )
        self._slots[name] = ArenaSlot(storage)

    def slot(self, name: str) -> ArenaSlot:
        try:
            return self._slots[name]
        except KeyError as e:
            raise KeyError(f"no arena slot {name!r} — register() before any capture_session") from e


def pad_to_bucket(raw_size: int, buckets: Sequence[int]) -> int:
    """Smallest bucket >= raw_size. `buckets` must be sorted ascending
    and non-empty; the caller's admission check (can_run / can-
    schedule) must reject raw_size > buckets[-1] before this is ever
    called — that contract is enforced here as a real error with the
    actual sizes named, not just an out-of-range IndexError.
    """
    if not buckets:
        raise ValueError("buckets must be non-empty")
    if raw_size > buckets[-1]:
        raise ValueError(
            f"raw_size {raw_size} exceeds the largest captured bucket "
            f"{buckets[-1]} — the caller's admission check should have "
            f"rejected this batch before it reached load_batch()"
        )
    index = bisect.bisect_left(buckets, raw_size)
    return buckets[index]


def build_buckets(
    sizes: Sequence[int],
    *,
    max_size: int,
    multiple_of: int = 1,
) -> list[int]:
    """Filter/normalize a configured bucket list: keep only sizes that
    are a multiple of `multiple_of` (e.g. tensor-parallel alignment)
    and <= max_size, always include the largest valid multiple of
    `multiple_of` not exceeding max_size so nothing above every
    configured bucket is silently unreachable, dedupe, sort.
    """
    buckets = {s for s in sizes if 0 < s <= max_size and s % multiple_of == 0}
    largest_aligned = max_size - (max_size % multiple_of)
    if largest_aligned > 0:
        buckets.add(largest_aligned)
    if not buckets:
        raise ValueError(
            f"no valid capture bucket <= {max_size} that's a multiple of "
            f"{multiple_of} — check the cuda_graph bucket config"
        )
    return sorted(buckets)


class GraphBackendKind(Enum):
    FULL = auto()
    BREAKABLE = auto()


def resolve_backend(
    kind: GraphBackendKind,
    *,
    device: torch.device,
    barrier_fn: Callable[[], None] | None = None,
) -> GraphBackend:
    """The one place a new backend kind gets wired in — nothing in
    DecodeGraphRunner needs to change to add one, per the Strategy +
    Factory split this whole module follows."""
    if kind is GraphBackendKind.FULL:
        return FullGraphBackend(device=device, barrier_fn=barrier_fn)
    if kind is GraphBackendKind.BREAKABLE:
        return BreakableGraphBackend(device=device, barrier_fn=barrier_fn)
    raise ValueError(f"unknown GraphBackendKind: {kind}")


class DecodeGraphRunner:
    """Owns one GraphBufferArena, one GraphBackend, and the bucket
    list for the decode phase.

    build_forward_fn is supplied by the caller — it closes over the
    real model and the arena's static slots, so this class never has
    to import model code. Its return type must match the chosen
    backend: a plain zero-arg callable for FULL, a zero-arg callable
    returning `breakable_backend.Spans` for BREAKABLE. Called once per
    bucket during capture() and again on every execute().

    The runner is bound to one resource generation: ``bind_generation``
    makes ``can_run`` refuse a bucket whose captured pointers belong to a
    replaced owner, so a stale graph can never be replayed. The optional
    ``prepare`` hook stages capture-time input into the captured backing
    *outside* the captured region; replay-time staging is passed per call
    so a request-dependent branch is never recorded into a graph.
    """

    def __init__(
        self,
        *,
        buckets: list[int],
        device: torch.device,
        arena: GraphBufferArena,
        build_forward_fn: Callable[[int], Any],
        backend_kind: GraphBackendKind = GraphBackendKind.FULL,
        barrier_fn: Callable[[], None] | None = None,
        generation_provider: Callable[[], Any] | None = None,
    ) -> None:
        if not buckets or list(buckets) != sorted(buckets):
            raise ValueError("buckets must be a non-empty, ascending, deduped list")
        self._buckets = buckets
        self._device = device
        self._arena = arena
        self._build_forward_fn = build_forward_fn
        self._backend = resolve_backend(backend_kind, device=device, barrier_fn=barrier_fn)
        self._generation_provider = generation_provider
        self._generation: Any = None
        self._captured = False

    def bind_generation(self, generation: Any) -> None:
        """Pin the captured graph's owner; replay is refused for any other."""
        self._generation = generation

    @property
    def generation(self) -> Any:
        return self._generation

    def _generation_current(self) -> bool:
        if self._generation is None:
            return True
        if self._generation_provider is None:
            return True
        live = self._generation_provider()
        return live is not None and live == self._generation

    def capture(
        self,
        *,
        prepare: Callable[[int], None] | None = None,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        if self._captured:
            raise RuntimeError("capture() already ran for this runner instance")
        stream = stream if stream is not None else torch.cuda.Stream(device=self._device)
        with self._backend.capture_session(stream):
            # Largest bucket first: later (smaller) captures reuse address
            # space the pool freed from the larger one, so peak memory
            # tracks the largest bucket, not the sum of all of them — see
            # GraphMemoryPool.
            for size in sorted(self._buckets, reverse=True):
                shape_key = ShapeKey(size=size)
                logger.info("capturing decode graph, size=%d", size)
                if prepare is not None:
                    # Stage the capture-placeholder input OUTSIDE the captured
                    # region: the recorded graph must only contain device work.
                    prepare(size)
                self._backend.capture_one(shape_key, self._build_forward_fn(size))
        self._captured = True

    def can_run(self, raw_bs: int) -> bool:
        if not self._captured or raw_bs > self._buckets[-1]:
            return False
        if not self._generation_current():
            raise GraphCapabilityError(
                "decode graphs were captured under a replaced runtime generation; "
                "recapture before replaying"
            )
        bucket = pad_to_bucket(raw_bs, self._buckets)
        result = self._backend.can_run(ShapeKey(size=bucket))
        if result is CanRun.CAPABILITY_MISMATCH:
            # I4: a real capability gap must not look like "just fall back
            # to eager" to the caller — surface it instead of returning
            # False, which the caller would read as an ordinary routing
            # decision.
            raise GraphCapabilityError(
                f"decode graph at bucket {bucket} exists but can't serve "
                f"this batch's requested capability"
            )
        return result is CanRun.RUNNABLE

    def execute(self, raw_bs: int, *, forward_fn: Callable[[], Any] | None = None) -> Any:
        if not self.can_run(raw_bs):
            raise RuntimeError("execute() called without a prior successful can_run()")
        bucket = pad_to_bucket(raw_bs, self._buckets)
        shape_key = ShapeKey(size=bucket)
        with self._backend.replay_session():
            # A per-call forward_fn restages the live step into the captured
            # backing before replay; a full-graph backend ignores it at replay,
            # a breakable backend runs its eager spans for real.
            stage = forward_fn if forward_fn is not None else self._build_forward_fn(bucket)
            raw_output = self._backend.replay(shape_key, forward_fn=stage)
        return slice_rows(raw_output, raw_bs)

    def cleanup(self) -> None:
        self._backend.cleanup()
        self._captured = False
