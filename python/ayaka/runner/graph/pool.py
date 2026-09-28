"""Per-flight decode-graph capture pool.

One captured graph binds the exact backing pointers of one flight slot. The
pool captures every configured ``(bucket, slot)`` pair at bootstrap. Each slot
has a private graph memory pool: other slots and independent runtime owners
cannot overwrite its retained logits or intermediates. Buckets within a slot
share scratch and must replay serially, after its prior consumers retire.

Capture is fail-closed: a partial capture tears down whatever it built and
re-raises, so serving never starts with a half-captured pool. The actual
graph-private footprint is measured and reconciled against the frozen reserve
before admission.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from threading import RLock
from typing import TYPE_CHECKING, Any

import torch

from ayaka.runner.graph.graph import GraphCapabilityError
from ayaka.runner.graph.planner import DecodeGraphPlanner
from ayaka.runner.graph.runner import DecodeGraphRunner, GraphBackendKind, GraphBufferArena

if TYPE_CHECKING:
    from ayaka.memory.capacity import ResourceGeneration
    from ayaka.sched.plan import BatchStepPlan
    from ayaka.types import AttentionCudaGraphSupport

logger = logging.getLogger(__name__)

# Capture is a process/device bootstrap operation. Serializing all devices is
# conservative; this gate is never held by ordinary serving replay.
_capture_gate = RLock()


@contextmanager
def execution_enqueue_gate():
    """Prevent cooperating workers from enqueueing during bootstrap capture.

    This serializes host enqueue scopes only, not completion. Capture performs
    the stream/device joins after acquiring it; workers never wait for GPU
    completion while holding this gate.
    """
    with _capture_gate:
        yield


class GraphOwnerState(StrEnum):
    NEW = "new"
    RESOURCES_BOUND = "resources_bound"
    WARMING = "warming"
    CAPTURING = "capturing"
    VALIDATING = "validating"
    READY = "ready"
    INVALIDATING = "invalidating"
    DRAINING = "draining"
    FAILED = "failed"
    CLOSED = "closed"


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
    graph_backend: str = "full"
    compile_seconds: int = 120

    def __post_init__(self) -> None:
        from ayaka.execution.shape_key import validate_buckets
        from ayaka.utils.validation import require_int

        validate_buckets(self.buckets)
        if self.graph_backend not in ("full", "breakable", "torch_compile_piecewise"):
            raise ValueError("unknown graph backend")
        for name in ("max_model_len", "padding_page", "padding_slot", "reserve_bytes"):
            require_int(getattr(self, name), name, minimum=1 if name == "max_model_len" else 0)
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
        kernel_binding: Callable[[], str],
        metrics: Any | None = None,
        barrier_fn: Callable[[], None] | None = None,
        eager_only_reason: Callable[[BatchStepPlan], str | None] | None = None,
        enable_gc_freeze: bool = True,
        graph_backend: str = "full",
    ) -> None:
        if device.type != "cuda":
            raise DecodeGraphPoolError("decode graphs require a CUDA device")
        if not callable(kernel_binding):
            raise DecodeGraphPoolError("kernel_binding must be a zero-argument callable")
        from ayaka.execution.shape_key import validate_buckets

        resolved = tuple(buckets)
        validate_buckets(resolved)
        slot_list = tuple(slots)
        if (
            not slot_list
            or any(type(s) is not int or s < 0 for s in slot_list)
            or tuple(sorted(set(slot_list))) != slot_list
        ):
            raise DecodeGraphPoolError("decode graph flight slots must be non-negative")
        self.state = GraphOwnerState.RESOURCES_BOUND
        self._device = device
        self._buckets = resolved
        self._slots = slot_list
        self._generation = generation
        self._kernel_binding = kernel_binding
        self._barrier_fn = barrier_fn
        self._enable_gc_freeze = bool(enable_gc_freeze)
        self._graph_backend = graph_backend
        self._runners: dict[int, DecodeGraphRunner] = {}
        self._actual_bytes: int | None = None
        self.peak_capture_bytes = 0
        self._planner = DecodeGraphPlanner(
            buckets=resolved,
            support=support,
            backend=backend,
            dtype=dtype,
            generation=generation,
            captured=self._captured_buckets,
            kernel_binding=kernel_binding,
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
        if self.state is not GraphOwnerState.READY or not self._runners:
            return frozenset()
        return frozenset(self._buckets)

    def has_instance(self, bucket: int, slot: int) -> bool:
        if self.state is not GraphOwnerState.READY:
            return False
        runner = self._runners.get(slot)
        if runner is None:
            return False
        return runner.can_run(bucket)

    def compilation_report(self) -> tuple[dict, ...]:
        from ayaka.execution.graph_program import compiler_report

        return tuple(
            {"slot": slot, "bucket": bucket, **report}
            for slot, runner in self._runners.items()
            for bucket in self._buckets
            if (report := compiler_report(runner._build_forward_fn(bucket))) is not None
        )

    # ── capture ───────────────────────────────────────────────────────────

    def capture(
        self,
        program: CaptureProgram,
        *,
        stream: torch.cuda.Stream | None = None,
        persistent_bytes: int = 0,
    ) -> int:
        """Exclusive, RNG-preserving bootstrap transaction; publish only a full set."""
        if self.state is not GraphOwnerState.RESOURCES_BOUND:
            raise DecodeGraphPoolError("capture requires a fresh resource binding")
        with _capture_gate, torch.random.fork_rng(devices=[self._device.index or 0]):
            try:
                return self._capture(program, stream=stream, persistent_bytes=persistent_bytes)
            except BaseException:
                self.state = GraphOwnerState.FAILED
                # Cover partial warmup/capture work before dropping references.
                # If this fails the owner and its partial set remain reachable.
                torch.cuda.synchronize(self._device)
                self.cleanup()
                self.state = GraphOwnerState.FAILED
                raise

    def _capture(
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
        allocated_before = torch.cuda.memory_allocated(self._device)
        torch.cuda.reset_peak_memory_stats(self._device)
        generation_before = self._generation()
        binding_before = self._kernel_binding()
        started = time.perf_counter()
        self.state = GraphOwnerState.WARMING
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
                    backend_kind=(
                        GraphBackendKind.FULL
                        if self._graph_backend == "full"
                        else GraphBackendKind.BREAKABLE
                    ),
                )

                def prepare_size(
                    size: int,
                    _slot: int = slot,
                    _cache: dict = forward_by_size,
                ) -> None:
                    _cache[size] = program.prepare(_slot, size)

                # Keep the currently failing slot reachable for cleanup too.
                self._runners[slot] = runner
                self.state = GraphOwnerState.CAPTURING
                runner.capture(prepare=prepare_size, stream=stream)
        except BaseException:
            raise
        self.state = GraphOwnerState.VALIDATING
        torch.cuda.synchronize(self._device)
        capture_seconds = time.perf_counter() - started
        after = self._reserved_bytes()
        self.peak_capture_bytes = max(
            0,
            torch.cuda.max_memory_reserved(self._device) - before,
            torch.cuda.max_memory_allocated(self._device) - allocated_before,
        ) + max(0, int(persistent_bytes))
        self._actual_bytes = max(
            0, after - before, torch.cuda.memory_allocated(self._device) - allocated_before
        ) + max(0, int(persistent_bytes))
        generation = self._generation()
        if generation != generation_before or self._kernel_binding() != binding_before:
            raise DecodeGraphPoolError("resource binding changed during capture")
        for runner in self._runners.values():
            runner.bind_generation(generation)
        self._planner.note_capture(
            self._buckets,
            actual_bytes=self._actual_bytes,
            capture_seconds=capture_seconds,
        )
        self.state = GraphOwnerState.READY
        logger.info(
            "captured decode graphs: buckets=%s slots=%s bytes=%d seconds=%.3f",
            self._buckets,
            self._slots,
            self._actual_bytes,
            capture_seconds,
        )
        return self._actual_bytes

    def reconcile(self, reserved_bytes: int) -> int:
        """Refuse to serve when the measured pool exceeds the frozen reserve."""
        actual = self._actual_bytes
        if actual is None:
            raise DecodeGraphPoolError("reconcile() requires a completed capture")
        if reserved_bytes < 0:
            raise DecodeGraphPoolError("reserved graph bytes must be non-negative")
        if max(actual, self.peak_capture_bytes) > reserved_bytes:
            self.invalidate("capture_budget")
            raise DecodeGraphPoolError(
                f"decode graphs hold {actual} bytes (peak {self.peak_capture_bytes}) "
                f"but only {reserved_bytes} bytes were "
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
        if self._graph_backend == "full":
            output = runner.execute(rows, forward_fn=stage, captured_bucket=bucket)
        else:
            stage()
            output = runner.execute(rows, captured_bucket=bucket)
        # Only a launched replay counts as a hit; a refused/failed launch above
        # must not inflate the hit counter.
        self._planner.note_replay(bucket)
        return output

    # ── lifecycle ─────────────────────────────────────────────────────────

    def invalidate(self, reason: Any) -> None:
        self.state = GraphOwnerState.INVALIDATING
        self._planner.invalidate(reason)

    def note_workspace_growth(self) -> None:
        self.state = GraphOwnerState.INVALIDATING
        self._planner.note_workspace_growth()

    def cleanup(self) -> None:
        """Drop every captured graph; caller must have drained all flights."""
        if self.state is GraphOwnerState.CLOSED:
            return
        self.state = GraphOwnerState.DRAINING
        for runner in self._runners.values():
            runner.cleanup()
        self._runners.clear()
        self._actual_bytes = None
        self.state = GraphOwnerState.CLOSED

    def close(self) -> None:
        """Alias for cleanup with a name the owning runtime can call."""
        self.cleanup()

    def _reserved_bytes(self) -> int:
        return torch.cuda.memory_reserved(self._device)
