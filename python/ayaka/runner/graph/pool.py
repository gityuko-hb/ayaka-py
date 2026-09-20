"""Per-flight decode-graph capture pool.

One captured graph binds the exact backing pointers of one flight slot, so the
pool captures every configured ``(bucket, slot)`` pair at bootstrap and never
lets a graph replay against another slot's buffers. All captured instances share
one process-global graph memory pool (``GraphMemoryPool``); each instance's
output stays private to its slot, which is what makes two in-flight tickets
unable to overwrite each other's logits.

Capture is fail-closed: a partial capture tears down whatever it built and
re-raises, so serving never starts with a half-captured pool. The actual
graph-private footprint is measured and reconciled against the frozen reserve
before admission.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from ayaka.runner.graph.graph import GraphCapabilityError
from ayaka.runner.graph.planner import DecodeGraphPlanner
from ayaka.runner.graph.runner import DecodeGraphRunner, GraphBufferArena

if TYPE_CHECKING:
    from ayaka.memory.capacity import ResourceGeneration
    from ayaka.sched.plan import BatchStepPlan
    from ayaka.types import AttentionCudaGraphSupport

logger = logging.getLogger(__name__)

__all__ = ["CaptureProgram", "DecodeGraphConfig", "DecodeGraphPool", "DecodeGraphPoolError"]


class DecodeGraphPoolError(RuntimeError):
    """Bootstrap capture/reconcile failed; the runtime must not start with it."""


@dataclass(frozen=True, slots=True)
class DecodeGraphConfig:
    """Everything the runner needs to capture pure-decode graphs.

    ``padding_page``/``padding_slot`` name the reserved padding page dummy
    decode lanes write to; ``reserve_bytes`` is the frozen graph budget the
    measured footprint is reconciled against before admission.
    """

    buckets: tuple[int, ...]
    max_model_len: int
    padding_page: int
    padding_slot: int
    reserve_bytes: int = 0
    enable_gc_freeze: bool = True
    barrier_fn: Callable[[], None] | None = None

    def __post_init__(self) -> None:
        if not self.buckets or tuple(sorted(set(self.buckets))) != tuple(self.buckets):
            raise ValueError("buckets must be a non-empty ascending deduplicated tuple")
        if any(bucket < 1 for bucket in self.buckets):
            raise ValueError("buckets must be positive")
        if self.max_model_len < 1:
            raise ValueError("max_model_len must be positive")
        if self.padding_page < 0 or self.padding_slot < 0:
            raise ValueError("padding page/slot must be non-negative")
        if self.reserve_bytes < 0:
            raise ValueError("reserve_bytes must be non-negative")


@dataclass(frozen=True, slots=True)
class CaptureProgram:
    """Caller-supplied hook binding the pool to a concrete model and flight.

    ``prepare(slot, bucket)`` stages capture-placeholder input and backend
    metadata OUTSIDE the captured region and returns the zero-arg forward
    callable that contains only device work — that is what gets recorded.
    """

    prepare: Callable[[int, int], Callable[[], Any]]


class DecodeGraphPool:
    """Owns one :class:`DecodeGraphRunner` per flight slot and routes replays."""

    def __init__(
        self,
        *,
        device: torch.device,
        backend: str,
        dtype: str,
        buckets: Sequence[int],
        support: AttentionCudaGraphSupport,
        slots: Sequence[int],
        generation: Callable[[], ResourceGeneration | None],
        metrics: Any | None = None,
        barrier_fn: Callable[[], None] | None = None,
        eager_only_reason: Callable[[BatchStepPlan], str | None] | None = None,
        enable_gc_freeze: bool = True,
    ) -> None:
        if device.type != "cuda":
            raise DecodeGraphPoolError("decode graphs require a CUDA device")
        resolved = tuple(sorted(set(int(b) for b in buckets)))
        if not resolved or resolved[0] < 1:
            raise DecodeGraphPoolError("decode graph buckets must be positive")
        slot_list = tuple(sorted(set(int(s) for s in slots)))
        if not slot_list or slot_list[0] < 0:
            raise DecodeGraphPoolError("decode graph flight slots must be non-negative")
        self._device = device
        self._buckets = resolved
        self._slots = slot_list
        self._generation = generation
        self._barrier_fn = barrier_fn
        self._enable_gc_freeze = bool(enable_gc_freeze)
        self._runners: dict[int, DecodeGraphRunner] = {}
        self._actual_bytes: int | None = None
        self._planner = DecodeGraphPlanner(
            buckets=resolved,
            support=support,
            backend=backend,
            dtype=dtype,
            generation=generation,
            captured=self._captured_buckets,
            eager_only_reason=eager_only_reason,
            metrics=metrics,
        )

    # ── state ─────────────────────────────────────────────────────────────

    @property
    def planner(self) -> DecodeGraphPlanner:
        return self._planner

    @property
    def buckets(self) -> tuple[int, ...]:
        return self._buckets

    @property
    def slots(self) -> tuple[int, ...]:
        return self._slots

    @property
    def actual_bytes(self) -> int | None:
        """Measured graph-private bytes held after capture, when captured."""
        return self._actual_bytes

    def _captured_buckets(self) -> frozenset[int]:
        if not self._runners:
            return frozenset()
        return frozenset(self._buckets)

    def has_instance(self, bucket: int, slot: int) -> bool:
        runner = self._runners.get(slot)
        if runner is None:
            return False
        return runner.can_run(bucket)

    # ── capture ───────────────────────────────────────────────────────────

    def capture(
        self,
        program: CaptureProgram,
        *,
        stream: torch.cuda.Stream | None = None,
        persistent_bytes: int = 0,
    ) -> int:
        """Warm up and record every ``(bucket, slot)``; fail closed on error.

        ``persistent_bytes`` covers backend buffers allocated before capture
        (the Triton graph state), which are part of the real footprint the
        frozen reserve must cover. Returns the measured graph-private footprint.
        """
        if self._runners:
            raise DecodeGraphPoolError("this pool already captured; rebuild instead of recapturing")
        before = self._reserved_bytes()
        try:
            for slot in self._slots:
                arena = GraphBufferArena()
                forward_by_size: dict[int, Callable[[], Any]] = {}

                def build_forward(size: int, _cache: dict = forward_by_size) -> Callable[[], Any]:
                    return _cache[size]

                runner = DecodeGraphRunner(
                    buckets=list(self._buckets),
                    device=self._device,
                    arena=arena,
                    build_forward_fn=build_forward,
                    barrier_fn=self._barrier_fn,
                    generation_provider=self._generation,
                    enable_gc_freeze=self._enable_gc_freeze,
                )

                def prepare_size(
                    size: int,
                    _slot: int = slot,
                    _cache: dict = forward_by_size,
                ) -> None:
                    _cache[size] = program.prepare(_slot, size)

                runner.capture(prepare=prepare_size, stream=stream)
                self._runners[slot] = runner
        except BaseException:
            self.cleanup()
            raise
        torch.cuda.synchronize(self._device)
        after = self._reserved_bytes()
        self._actual_bytes = max(0, after - before) + max(0, int(persistent_bytes))
        generation = self._generation()
        for runner in self._runners.values():
            runner.bind_generation(generation)
        self._planner.note_capture(self._buckets)
        logger.info(
            "captured decode graphs: buckets=%s slots=%s bytes=%d",
            self._buckets,
            self._slots,
            self._actual_bytes,
        )
        return self._actual_bytes

    def reconcile(self, reserved_bytes: int) -> int:
        """Refuse to serve when the measured pool exceeds the frozen reserve."""
        actual = self._actual_bytes
        if actual is None:
            raise DecodeGraphPoolError("reconcile() requires a completed capture")
        if reserved_bytes < 0:
            raise DecodeGraphPoolError("reserved graph bytes must be non-negative")
        if actual > reserved_bytes:
            raise DecodeGraphPoolError(
                f"decode graphs hold {actual} bytes but only {reserved_bytes} bytes were "
                f"reserved (graph_pool_bytes); lower the largest bucket, raise the reserve, "
                f"or reduce max_num_seqs"
            )
        return actual

    # ── replay ────────────────────────────────────────────────────────────

    def should_replay(self, step: BatchStepPlan, slot: int) -> bool:
        """Planner gate plus the slot's captured instance presence."""
        if not self._planner.should_replay(step):
            return False
        return self.has_instance(step.graph.bucket, slot)

    def execute(
        self,
        step: BatchStepPlan,
        *,
        slot: int,
        rows: int,
        stage: Callable[[], Any],
    ) -> Any:
        """Restage the live step into the captured backing, then replay.

        ``stage`` runs on the engine stream immediately before replay; it must
        be copy-only (no planning that allocates, no host sync). ``rows`` is the
        real (unpadded) row count whose output prefix is returned.
        """
        runner = self._runners.get(slot)
        if runner is None:
            raise GraphCapabilityError(f"no captured decode graph for flight slot {slot}")
        bucket = step.graph.bucket
        if not self._planner.should_replay(step):
            raise GraphCapabilityError("decode graph replay refused after the plan was validated")
        if rows > bucket:
            raise GraphCapabilityError(f"{rows} real rows exceed the captured bucket {bucket}")
        self._planner.note_replay(bucket)
        return runner.execute(rows, forward_fn=stage)

    # ── lifecycle ─────────────────────────────────────────────────────────

    def invalidate(self, reason: Any) -> None:
        self._planner.invalidate(reason)

    def note_workspace_growth(self) -> None:
        self._planner.note_workspace_growth()

    def cleanup(self) -> None:
        """Drop every captured graph; caller must have drained all flights."""
        for runner in self._runners.values():
            runner.cleanup()
        self._runners.clear()
        self._actual_bytes = None

    def close(self) -> None:
        """Alias for cleanup with a name the owning runtime can call."""
        self.cleanup()

    def _reserved_bytes(self) -> int:
        stats = torch.cuda.memory_stats(self._device)
        return int(stats.get("reserved_bytes.all", 0))
