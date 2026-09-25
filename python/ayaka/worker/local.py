"""Local single-device worker: owns context, streams, runner, COW and fences.

The executor adapter never imports CUDA stream or event APIs: every device
operation lives here. Ordering is fixed: COW pages move on the KV stream at
the batch boundary, compute waits for them on the compute stream, and the
completion fence is recorded after the last compute consumer. A drain proof
covers transfer plus compute, including partially submitted work.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import nullcontext

import torch

from ayaka.device.backend import DeviceBackend, NullBackend
from ayaka.device.context import DeviceContext
from ayaka.device.stream import StreamPool
from ayaka.executor.ticket import CompletionFence
from ayaka.kvcache.manager import LogicalKVManager
from ayaka.obs import RuntimeMetrics
from ayaka.runner.sampling_runner import SampleRunner
from ayaka.types import StreamRole
from ayaka.utils.validation import require_int
from ayaka.worker.base import StepWorker, WorkerOutcome, WorkerStep
from ayaka.worker.fences import FlightFence, ImmediateFence, RecordingFence
from ayaka.worker.lifecycle import FlightKey, WorkerLifecycle, WorkerState
from ayaka.worker.resources import WorkerResources

__all__ = ["LocalWorker"]

_CUDA_ERRORS: tuple[type[BaseException], ...] = tuple(
    error
    for error in (
        getattr(torch.cuda, "CudaError", None),
        getattr(torch.cuda, "OutOfMemoryError", None),
    )
    if isinstance(error, type)
)


def _is_ambiguous_device_failure(error: BaseException) -> bool:
    """CUDA faults may have poisoned the context; never reuse it optimistically."""
    if _CUDA_ERRORS and isinstance(error, _CUDA_ERRORS):
        return True
    # PyTorch also surfaces asynchronous device faults as plain RuntimeError.
    message = str(error).lower()
    return isinstance(error, RuntimeError) and any(
        marker in message
        for marker in ("cuda error:", "device-side assert", "illegal memory access")
    )


class LocalWorker(StepWorker):
    """One resident local worker owning every device resource of a step.

    The KV binding is read-only for physical page addressing; leases and
    commit/retire rights stay with the ticket's ``ExecutionResources``. The
    runner is closed here before streams are destroyed, and close refuses to
    run while any flight has not proved quiescence.
    """

    def __init__(
        self,
        kv: LogicalKVManager,
        runner: SampleRunner,
        *,
        max_inflight: int = 1,
        device: torch.device | None = None,
        backend: DeviceBackend | None = None,
        pool: StreamPool | None = None,
        metrics: RuntimeMetrics | None = None,
        fence_factory: Callable[[], CompletionFence] | None = None,
        resources: WorkerResources | None = None,
    ) -> None:
        require_int(max_inflight, "max_inflight", minimum=1)
        devices = {
            tensor.device
            for lease in kv.storages.values()
            for family in lease.storage.buffers()
            for tensor in family
        }
        if len(devices) != 1:
            raise ValueError("resident worker requires KV on exactly one device")
        resolved = next(iter(devices)) if device is None else device
        if not isinstance(resolved, torch.device):
            raise TypeError("device must be a torch.device")
        if resolved.type not in ("cpu", "cuda"):
            raise ValueError("resident worker supports only CPU or CUDA")
        if resolved != next(iter(devices)):
            raise ValueError("worker device must match the KV storage device")
        self.kv = kv
        if resources is not None and resources.kv is not kv:
            raise ValueError("worker resources must own the worker KV binding")
        self.resources = resources
        self.runner = runner
        self._device = resolved
        self.max_inflight = max_inflight
        self.metrics = metrics or RuntimeMetrics()
        self.fence_factory = fence_factory
        self._lifecycle = WorkerLifecycle(device_index=resolved.index or 0)
        self._backend = backend
        self._pool = pool
        self._owns_pool = pool is None
        self._flight_fences: dict[FlightKey, list[CompletionFence]] = {}

    # ------------------------------------------------------------------
    # public state
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def state(self) -> WorkerState:
        return self._lifecycle.state

    @property
    def accepting(self) -> bool:
        return self._lifecycle.accepting

    @property
    def incarnation(self) -> int:
        return self._lifecycle.generation

    @property
    def failure(self) -> str | None:
        return self._lifecycle.failure

    @property
    def active_flights(self) -> tuple[FlightKey, ...]:
        return self._lifecycle.active_flights

    @property
    def num_flights(self) -> int:
        return self._lifecycle.num_flights

    def fail(self, error: object) -> None:
        """Report an externally observed device failure; admission stops."""
        self.metrics.increment("worker_failures")
        self._lifecycle.fail(error)

    @property
    def resource_generation(self):
        return self.kv.generation

    @property
    def stream(self):
        """The compute stream, when this worker owns real device queues."""
        pool = self._open_pool()
        return None if pool is None else pool.get(StreamRole.COMPUTE)

    @property
    def kv_stream(self):
        """The transfer stream used for COW copies."""
        pool = self._open_pool()
        return None if pool is None else pool.get(StreamRole.KV)

    def _open_pool(self) -> StreamPool | None:
        pool = self._pool
        if pool is None or pool.is_closed or self._lifecycle.state is WorkerState.CLOSED:
            return None
        return pool

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        self._lifecycle.begin_initializing()
        try:
            if self.device.type == "cuda":
                context = DeviceContext(self.device.index or 0)
                context.bind()
                backend = self._backend or context.backend
                if self._pool is None:
                    self._pool = StreamPool(backend, self.device.index or 0)
                self._backend = backend
                # Adopt whatever the caller already ordered onto the current
                # stream; the worker never synchronizes globally.
                current = torch.cuda.current_stream(self.device)
                self._pool.get(StreamRole.COMPUTE).wait_stream(current)
            elif self._backend is None:
                self._backend = NullBackend()
            self._lifecycle.mark_ready()
        except BaseException:
            self._lifecycle.fail("worker initialization failed")
            self._close_pool()
            if self.resources is not None:
                self.resources.close()
            raise

    def execute(self, step: WorkerStep) -> WorkerOutcome:
        self._require_accepting()
        self._lifecycle.validate_runtime()
        self._validate_generation(step)
        if self.num_flights >= self.max_inflight:
            raise RuntimeError("worker is at its max_inflight capacity")
        self._lifecycle.begin_flight(step.ticket_id, tag=str(step.step_id))
        self.metrics.increment("worker_flights_started")
        try:
            context = (
                nullcontext() if self._pool is None else self._pool.context(StreamRole.COMPUTE)
            )
            with context, torch.inference_mode():
                self._apply_cow(step)
                note_growth = getattr(self.runner, "on_workspace_growth", None)
                if note_growth is not None and step.workspace_grew:
                    # The step grew the shared workspace; any captured graph that
                    # bound the old addresses must not replay. The runner falls
                    # back to eager for this step and recaptures before the next
                    # replay.
                    note_growth()
                samples = self.runner(step.prepared)
                if samples.token_ids.device != self.device:
                    raise ValueError("runner samples must reside on the KV device")
                fence = self._register_fence(step.ticket_id, self._device_fence())
            return WorkerOutcome(samples, fence)
        except BaseException as exc:
            if _is_ambiguous_device_failure(exc):
                self._lifecycle.fail(f"ambiguous device failure during execute: {exc}")
                self.metrics.increment("worker_failures")
            # The flight stays registered: an exception here may hide partially
            # submitted work, so only a whole-ticket drain proof can retire it.
            raise

    def drain(self, step: WorkerStep) -> CompletionFence:
        """Return a proof covering transfer plus compute, tracked or not."""
        self._lifecycle.validate_runtime()
        if self._lifecycle.state is WorkerState.CLOSED:
            raise RuntimeError("worker is closed and cannot produce a drain proof")
        if self._pool is not None:
            try:
                # Join any partial transfer before proving quiescence.
                self._pool.order(after=StreamRole.COMPUTE, before=StreamRole.KV)
            except BaseException as exc:
                self._lifecycle.fail(f"cannot order transfer drain: {exc}")
                self.metrics.increment("worker_failures")
                raise
        try:
            fence = self._register_fence(step.ticket_id, self._device_fence())
        except BaseException as exc:
            self._lifecycle.fail(f"cannot establish drain proof: {exc}")
            self.metrics.increment("worker_failures")
            raise
        return fence

    def request_closing(self) -> WorkerState:
        """Stop admission without ending the incarnation; idempotent."""
        return self._lifecycle.request_closing()

    def shutdown(self) -> bool:
        """Stop admission; close only once every flight proved quiescence."""
        self._lifecycle.request_closing()
        if self._lifecycle.num_flights:
            return False
        self.close()
        return True

    def close(self) -> None:
        if self._lifecycle.state is WorkerState.CLOSED:
            return
        self._lifecycle.validate_runtime()
        if self._lifecycle.num_flights:
            raise RuntimeError("cannot close a worker with active flights")
        self.request_closing()
        try:
            self.runner.close()
            self._release_flights()
            self._close_pool()
            if self.resources is not None:
                self.resources.close()
        except BaseException as exc:
            self.fail(f"worker cleanup failed: {exc}")
            raise
        self._lifecycle.mark_closed()

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _require_accepting(self) -> None:
        state = self._lifecycle.state
        if not self._lifecycle.accepting:
            raise RuntimeError(f"worker is {state.value} and cannot accept work")

    def _validate_generation(self, step: WorkerStep) -> None:
        captured = step.generation
        current = self.kv.generation
        if current != captured:
            raise ValueError("worker step belongs to a replaced runtime generation")

    def _register_fence(self, key: FlightKey, fence: CompletionFence) -> FlightFence:
        self._flight_fences.setdefault(key, []).append(fence)
        return FlightFence(fence, lambda: self._release_flight(key))

    def _release_flight(self, key: FlightKey) -> None:
        try:
            for fence in self._flight_fences.get(key, ()):
                release = getattr(fence, "release", None)
                if release is not None:
                    release()
        except BaseException as exc:
            self._note_fence_error(exc)
            raise
        self._flight_fences.pop(key, None)
        if self._lifecycle.end_flight(key):
            self.metrics.increment("worker_flights_completed")

    def _release_flights(self) -> None:
        for key in tuple(self._flight_fences):
            self._release_flight(key)

    def _note_fence_error(self, error: BaseException) -> None:
        self.metrics.increment("worker_fence_errors")
        self._lifecycle.fail(f"completion query failed: {error}")

    def _device_fence(self) -> CompletionFence:
        """Factory-aware fence used for submission and drain proofs."""
        if self.fence_factory is not None:
            return self.fence_factory()
        return self._default_fence()

    def _default_fence(self) -> CompletionFence:
        if self._pool is None or self._backend is None:
            return ImmediateFence()
        event = self._pool.events.record_on(self._pool.get(StreamRole.COMPUTE))
        return RecordingFence(
            self._backend,
            event,
            self._pool.events,
            on_error=self._note_fence_error,
        )

    def _close_pool(self) -> None:
        if not self._owns_pool:
            return
        pool = self._pool
        if pool is not None:
            pool.close()
            self._pool = None

    def _copy_pages(self, step: WorkerStep) -> None:
        from ayaka.kernel.triton.cache.cache_ops import copy_cache

        sources, destinations = [], []
        for copy in step.prepared.memory_view.copies:
            source = self.kv.physical_page(copy.group_name, copy.source)
            destination = self.kv.physical_page(copy.group_name, copy.destination)
            storage = self.kv.storages[copy.group_name].storage
            if not 0 < copy.valid_tokens <= storage.page_size:
                raise ValueError("invalid COW copy length")
            # Slice valid tokens only; never overwrite a destination's tail.
            for family in storage.buffers():
                for tensor in family:
                    sources.append(tensor[source, : copy.valid_tokens])
                    destinations.append(tensor[destination, : copy.valid_tokens])
        if sources:
            copy_cache(sources, destinations, False)

    def _apply_cow(self, step: WorkerStep) -> None:
        """Run lease-owned KV copies on the transfer stream at the batch boundary."""
        if self._pool is None:
            self._copy_pages(step)
            return
        self._pool.order(after=StreamRole.KV, before=StreamRole.COMPUTE)
        try:
            with self._pool.context(StreamRole.KV):
                self._copy_pages(step)
        finally:
            # Cover partial-copy failure too, before the compute drain fence.
            self._pool.order(after=StreamRole.COMPUTE, before=StreamRole.KV)

    def __repr__(self) -> str:
        return (
            f"<LocalWorker device={self.device} state={self.state.value} "
            f"flights={self._lifecycle.num_flights} pid={os.getpid()}>"
        )
