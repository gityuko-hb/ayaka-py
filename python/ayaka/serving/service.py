"""One owner thread serializes admission, engine execution and terminal cleanup."""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from collections import deque
from concurrent.futures import Future

from ayaka.configs.serving import ServingConfig
from ayaka.kvcache.resize import (
    CacheRebuildRejected,
    CacheResizeFatal,
    ResizeRejectionReason,
)
from ayaka.kvcache.status import CacheStatus
from ayaka.request.schema import Request
from ayaka.request.states import RequestState
from ayaka.serving.errors import (
    DeadlineExceededError,
    EngineUnavailableError,
    InvalidRequestError,
    OverloadedError,
)
from ayaka.serving.events import (
    ContentDelta,
    GenerationFailed,
    GenerationFinished,
    ReasoningDelta,
    ServingEvent,
    ToolCallArgumentsDelta,
    ToolCallEnd,
    ToolCallStart,
    UsageUpdate,
)
from ayaka.serving.migration import MigrationController
from ayaka.serving.parsers import OutputParser
from ayaka.serving.prepare import GenerationSpec, PrepareTimings
from ayaka.serving.stats import ServingStats, UsageRecord

_LOG = logging.getLogger(__name__)


def _wake(future: asyncio.Future) -> None:
    if not future.done():
        future.set_result(None)


def _event_bytes(event: ServingEvent) -> int:
    """UTF-8 payload estimate for the stream byte ceiling (control events are ~0)."""
    if isinstance(event, (ContentDelta, ReasoningDelta)):
        return len(event.text.encode())
    if isinstance(event, ToolCallArgumentsDelta):
        return len(event.fragment.encode())
    if isinstance(event, ToolCallEnd):
        return len(event.arguments.encode()) + len(event.name)
    if isinstance(event, ToolCallStart):
        return len(event.name)
    return 0


class _EngineQuarantined(RuntimeError):
    """A ticket's device completion is unknown; resources are retained."""


class _WorkerFailed(RuntimeError):
    """The device worker left READY; admission must stop."""


class GenerationHandle:
    """Bounded, nonblocking producer with a reserved terminal slot."""

    def __init__(self, request: Request, spec: GenerationSpec, config: ServingConfig):
        self.request = request
        self.cancelled = threading.Event()
        self._lock = threading.Lock()
        self._events: deque[ServingEvent] = deque()
        self._waiter: asyncio.Future | None = None
        self._limit = config.max_stream_events
        self._byte_limit = config.max_stream_bytes
        self._queued_bytes = 0
        self._terminal_sent = False
        self.overflow: str | None = None
        self.cursor = 0
        self.frontend_released = False
        self.started = time.monotonic()
        self.first_token: float | None = None
        #: Absolute monotonic boundaries for the SLO record (0 = unmeasured).
        self.ingress_ns = 0
        self.validation_done_ns = 0
        self.tokenize_done_ns = 0
        self.admitted_ns = 0
        self.first_published_ns = 0
        self.last_published_ns = 0
        self.first_socket_write_ns = 0
        self.error: str | None = None
        self.parser = OutputParser(
            reasoning=config.reasoning_parser == "think"
            and request.constraint is None
            and spec.messages is not None,
            tools=spec.tools if spec.tool_choice != "none" else (),
            constrained_tools=request.constraint is not None
            and request.constraint.kind == "tool_calls",
            max_buffer=config.max_parser_bytes,
        )

    def cancel(self) -> None:
        self.cancelled.set()

    def put(self, event: ServingEvent, *, terminal=False) -> bool:
        """Queue one event; returns False when the consumer is over budget.

        The terminal event has a reserved slot and is delivered exactly once,
        regardless of how full the queue is. Non-terminal events must fit both
        the event-count and UTF-8 byte ceilings; ``overflow`` names the budget
        that rejected the event so the caller can record a bounded reason.
        """
        size = _event_bytes(event)
        with self._lock:
            if terminal:
                if self._terminal_sent:
                    return False
                self._terminal_sent = True
            elif len(self._events) >= self._limit:
                self.overflow = "events"
                return False
            elif self._queued_bytes + size > self._byte_limit:
                self.overflow = "bytes"
                return False
            self._events.append(event)
            self._queued_bytes += size
            waiter, self._waiter = self._waiter, None
        if waiter is not None:
            try:
                waiter.get_loop().call_soon_threadsafe(_wake, waiter)
            except RuntimeError:
                self.cancel()
        return True

    async def next_event(self) -> ServingEvent:
        while True:
            with self._lock:
                if self._events:
                    event = self._events.popleft()
                    self._queued_bytes -= _event_bytes(event)
                    return event
                waiter = asyncio.get_running_loop().create_future()
                self._waiter = waiter
            try:
                await waiter
            finally:
                with self._lock:
                    if self._waiter is waiter:
                        self._waiter = None


class ServingService:
    """Transport never reads or mutates live scheduler/lifecycle dictionaries."""

    def __init__(
        self,
        engine,
        config: ServingConfig,
        *,
        constraints=None,
        stats: ServingStats | None = None,
        controller=None,
        router=None,
        migration: MigrationController | None = None,
    ):
        self.engine, self.config = engine, config
        self.stats = stats or ServingStats()
        self._commands: queue.Queue = queue.Queue(maxsize=config.max_pending_commands)
        self._controls: queue.Queue = queue.Queue(maxsize=config.max_pending_commands)
        self._handles: dict[str, GenerationHandle] = {}
        self._closing = threading.Event()
        self._stopped = threading.Event()
        self._signal = threading.Event()
        self._lock = threading.Lock()
        self._fatal: BaseException | None = None
        self._drained = False
        #: Engine-thread owner of KV rebuilds; ``apply_resize`` is only safe here.
        self._controller = controller
        #: Optional KV-aware placement decision before admission (route-only).
        self._router = router
        #: Optional chờ-KV migration: drives parked requests out of the park.
        self._migration = migration
        runner = engine.executor.runner
        runner.set_request_source(engine.requests.get, constraints=constraints)
        self._thread = threading.Thread(target=self._run, name="ayaka-engine", daemon=True)
        self._thread.start()

    @property
    def ready(self) -> bool:
        """Readiness requires a live owner and an executor that accepts work.

        A quarantined ticket or a worker that left READY makes this False even
        before the owner loop observes it, so admission stops immediately.
        """
        if self._closing.is_set() or self._fatal is not None or self._stopped.is_set():
            return False
        kv = getattr(self.engine, "kv", None)
        if kv is not None and getattr(kv, "closed", False):
            return False
        executor = getattr(self.engine, "executor", None)
        if executor is None:
            return True
        return bool(getattr(executor, "accepting_work", True))

    def resize(self, pages: int, *, timeout: float = 30.0) -> CacheStatus:
        """Queue a cache resize on the engine owner thread and wait for it.

        The engine thread rejects the request while any generation handle is
        active, before touching the cache, so the caller can retry when idle.
        """
        if not isinstance(pages, int) or isinstance(pages, bool):
            raise TypeError("pages must be an integer")
        if not self.ready:
            raise EngineUnavailableError("engine is unavailable")
        future: Future = Future()
        with self._lock:
            try:
                self._controls.put_nowait(("resize", pages, future))
            except queue.Full as exc:
                self.stats.record_rejection("control")
                raise EngineUnavailableError("engine control queue is full") from exc
        self._signal.set()
        try:
            result = future.result(timeout)
        except TimeoutError as exc:
            raise EngineUnavailableError("cache resize did not complete in time") from exc
        if not isinstance(result, CacheStatus):
            raise EngineUnavailableError("cache resize returned no status")
        return result

    def cache_status(self, *, timeout: float = 10.0) -> CacheStatus:
        """Read cache status on the engine owner thread.

        Going through the owner thread avoids racing a concurrent resize over
        the runtime's storage/manager attributes.
        """
        if not self.ready:
            raise EngineUnavailableError("engine is unavailable")
        future: Future = Future()
        with self._lock:
            try:
                self._controls.put_nowait(("status", None, future))
            except queue.Full as exc:
                self.stats.record_rejection("control")
                raise EngineUnavailableError("engine control queue is full") from exc
        self._signal.set()
        try:
            result = future.result(timeout)
        except TimeoutError as exc:
            raise EngineUnavailableError("cache status did not complete in time") from exc
        if not isinstance(result, CacheStatus):
            raise EngineUnavailableError("cache status returned no status")
        return result

    def _apply_controls(self) -> None:
        """Run queued cache controls on the engine thread between steps.

        Raises:
            CacheResizeFatal: When a rebuild cannot be recovered; the run loop
                treats it like any fatal engine error and drains resources.
        """
        while True:
            try:
                command, payload, future = self._controls.get_nowait()
            except queue.Empty:
                return
            if future.cancelled():
                continue
            controller = self._controller
            if controller is None or self._closing.is_set() or self._fatal is not None:
                future.set_exception(EngineUnavailableError("engine is unavailable"))
                continue
            try:
                if command == "status":
                    result = controller.cache_status()
                elif self._handles:
                    raise CacheRebuildRejected(
                        "engine is busy; resize requires an idle server; old cache kept",
                        reason=ResizeRejectionReason.BUSY,
                        requested_pages=payload,
                    )
                else:
                    result = controller.apply_resize(payload)
            except BaseException as exc:
                self.engine = controller.engine
                if not future.done():
                    future.set_exception(exc)
                if isinstance(exc, CacheResizeFatal):
                    raise
            else:
                self.engine = controller.engine
                future.set_result(result)

    def submit(
        self,
        request: Request,
        spec: GenerationSpec,
        *,
        timings: PrepareTimings | None = None,
    ) -> Future[GenerationHandle]:
        future: Future[GenerationHandle] = Future()
        with self._lock:
            if not self.ready:
                raise EngineUnavailableError("engine is unavailable")
            self.stats.conservation_submitted.inc()
            try:
                self._commands.put_nowait((request, spec, future, timings))
            except queue.Full as exc:
                self.stats.record_rejection("ingress")
                raise OverloadedError("engine ingress queue is full") from exc
        self._signal.set()
        return future

    @property
    def outstanding(self) -> int:
        """Requests offered but not settled: queued commands plus live handles."""
        return len(self._handles) + self._commands.qsize()

    def _admit(self, request, spec, future, timings=None):
        if not future.set_running_or_notify_cancel():
            return
        if not self.ready:
            future.set_exception(EngineUnavailableError("engine is closing"))
            return
        admitted_limit = self.config.max_admitted_requests
        if admitted_limit and len(self._handles) >= admitted_limit:
            # Refuse before engine admission: no lifecycle, sequence or KV is
            # ever allocated for a request the service cannot track.
            self.stats.record_rejection("admitted")
            future.set_exception(OverloadedError("too many admitted requests"))
            return
        try:
            decision = self._router.route(request) if self._router is not None else None
            defer = bool(decision is not None and decision.local and decision.needs_remote_kv)
            if decision is not None and not decision.local:
                # Remote placement has no executor path yet: this node keeps
                # the request (route-only milestone). A local decision with a
                # remote KV hit stages a chờ-KV fetch instead.
                _LOG.debug("router picked remote node %s; served locally", decision.node_id)
            self.engine.submit(request, defer_to_remote_kv=defer)
            self.stats.conservation_admitted.inc()
            if defer and self._migration is not None and decision is not None:
                self._migration.stage(
                    str(request.request_id),
                    decision.node_id,
                    prefix_tokens=decision.prefix_tokens,
                )
        except Exception as exc:
            from ayaka.sched.interfaces import DeadlineExceededError as SchedulerDeadline
            from ayaka.sched.interfaces import OverloadedError as SchedulerOverloaded

            if isinstance(exc, SchedulerDeadline):
                error = DeadlineExceededError(str(exc))
            elif isinstance(exc, SchedulerOverloaded):
                self.stats.record_rejection("scheduler")
                error = OverloadedError(str(exc))
            elif isinstance(exc, (ValueError, TypeError)):
                error = InvalidRequestError(str(exc))
            else:
                _LOG.exception("engine admission failed")
                error = EngineUnavailableError("engine admission failed")
            future.set_exception(error)
            return
        handle = GenerationHandle(request, spec, self.config)
        handle.admitted_ns = time.monotonic_ns()
        if timings is not None:
            handle.ingress_ns = timings.ingress_ns
            handle.validation_done_ns = timings.validation_done_ns
            handle.tokenize_done_ns = timings.tokenize_done_ns
        self._handles[str(request.request_id)] = handle
        future.set_result(handle)

    def _collect(self):
        for request_id, handle in tuple(self._handles.items()):
            lifecycle = self.engine.requests.get(request_id)
            if handle.cancelled.is_set() and not lifecycle.is_terminal:
                self.engine.abort(request_id)
            events = self.engine.events(request_id)
            for event in events[handle.cursor :]:
                handle.cursor += 1
                if event.usage.completion_tokens:
                    now_ns = time.monotonic_ns()
                    if handle.first_token is None:
                        handle.first_token = time.monotonic() - handle.started
                        handle.first_published_ns = now_ns
                    handle.last_published_ns = now_ns
                if handle.error or handle.cancelled.is_set():
                    continue
                try:
                    for parsed in handle.parser.feed(event.delta):
                        if not handle.put(parsed):
                            raise OverloadedError("output consumer is too slow")
                except Exception as exc:
                    if handle.overflow is not None:
                        self.stats.stream_overflows.labels(handle.overflow).inc()
                    handle.error = str(exc)
                    self.engine.abort(request_id)
            if not lifecycle.is_terminal:
                continue
            reason = lifecycle.finish_reason
            cancelled = lifecycle.state is RequestState.CANCELLED
            if lifecycle.state is RequestState.FAILED:
                handle.error = handle.error or "model execution failed"
            status = reason.value if reason is not None else ("cancelled" if cancelled else "error")
            if handle.cancelled.is_set():
                status = "cancelled"
            if not handle.error and status not in ("cancelled", "abort", "error", "timeout"):
                try:
                    for parsed in handle.parser.finish():
                        if not handle.put(parsed):
                            raise OverloadedError("output consumer is too slow")
                except Exception as exc:
                    if handle.overflow is not None:
                        self.stats.stream_overflows.labels(handle.overflow).inc()
                    handle.error = str(exc)
            usage = UsageUpdate(
                prompt_tokens=lifecycle.request.prompt_len,
                completion_tokens=len(lifecycle.output_token_ids),
                cached_tokens=lifecycle.machine.num_cached_tokens,
            )
            if handle.error or status == "error":
                status = "error"
                terminal = GenerationFailed(
                    "generation_failed", handle.error or "generation failed"
                )
            else:
                finish = (
                    "cancelled"
                    if status in ("abort", "cancelled")
                    else "tool_call"
                    if handle.parser.had_tools
                    else "timeout"
                    if status == "timeout"
                    else "length"
                    if status == "length"
                    else "stop"
                )
                terminal = GenerationFinished(finish, usage)
            terminal_ns = time.monotonic_ns()
            self.engine.forget(request_id)
            self.engine.executor.runner.forget_request(request_id)
            self.engine.requests.forget(request_id)
            del self._handles[request_id]
            # Deliver the terminal before accounting so a metrics fault can
            # never strand the only terminal event; the handle guard also makes
            # a repeated settlement a no-op instead of a duplicate terminal.
            handle.put(terminal, terminal=True)
            lm_timing = lifecycle.machine.timing
            itl_p50_ns, itl_p95_ns = lm_timing.inter_token_percentiles_ns()
            self.stats.record(
                UsageRecord(
                    request_id=request_id,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    cached_tokens=usage.cached_tokens,
                    status=status,
                    duration_seconds=time.monotonic() - handle.started,
                    priority=lifecycle.request.priority,
                    sla_class=lifecycle.request.sla_class,
                    prefix_hit=usage.cached_tokens > 0,
                    ingress_ns=handle.ingress_ns,
                    validation_done_ns=handle.validation_done_ns,
                    tokenize_done_ns=handle.tokenize_done_ns,
                    admitted_ns=handle.admitted_ns,
                    first_scheduled_ns=lm_timing.first_scheduled_ns,
                    first_model_token_ns=lm_timing.first_token_ns,
                    first_published_ns=handle.first_published_ns,
                    first_socket_write_ns=handle.first_socket_write_ns,
                    last_published_ns=handle.last_published_ns,
                    terminal_ns=terminal_ns,
                    cleanup_done_ns=time.monotonic_ns(),
                    itl_p50_ns=int(itl_p50_ns),
                    itl_p95_ns=int(itl_p95_ns),
                ),
                first_token=handle.first_token,
                itl_samples_ns=tuple(lm_timing.inter_token_intervals_ns),
            )
            _LOG.info(
                "request completed id=%s status=%s prompt_tokens=%d completion_tokens=%d",
                request_id,
                status,
                usage.prompt_tokens,
                usage.completion_tokens,
            )

    def _gauges(self):
        self.stats.running.set(self.engine.scheduler.num_running)
        self.stats.waiting.set(self.engine.scheduler.num_waiting)
        if self.engine.kv.closed:
            self.stats.kv_free.set(0)
            self.stats.kv_total.set(0)
            self.stats.clear_tiering()
            return
        snapshot = self.engine.kv.snapshot()
        self.stats.kv_free.set(snapshot.free_pages)
        self.stats.kv_total.set(snapshot.total_pages)
        tier = self.engine.kv.tier_metrics()
        if tier is not None:
            self.stats.update_tiering(tier)
        self.stats.update_pressure(self.engine.kv.pressure_metrics)

    def _raise_if_admission_closed(self) -> None:
        """Surface quarantine/worker failure on the owner thread.

        A quarantined ticket does not count as pending completion, and a FAILED
        worker refuses new flights, so without this check affected clients would
        wait forever. Raising routes both through the fail-closed drain path:
        clients receive one terminal, admission stops, and resources stay
        retained unless the drain proves quiescence.
        """
        executor = getattr(self.engine, "executor", None)
        if executor is None:
            return
        quarantined = getattr(executor, "quarantined", ())
        if quarantined:
            raise _EngineQuarantined(
                f"{len(quarantined)} quarantined ticket(s) retain device resources"
            )
        worker = getattr(executor, "worker", None)
        if worker is not None and not worker.accepting:
            raise _WorkerFailed(worker.failure or "device worker is not accepting work")

    def _fail_engine(self, exc: BaseException, *, code: str, message: str) -> None:
        """Drain if provable, fail every live client once, and retain on doubt."""
        _LOG.exception("engine owner failed; attempting drain before resource release")
        self._fatal = exc
        try:
            for handle in self._handles.values():
                handle.error = message
            result = self.engine.close()
            if result.closed:
                self._collect()
                self._gauges()
                self._drained = True
        except BaseException:
            _LOG.exception("fatal engine drain failed; resources retained")
        for handle in self._handles.values():
            handle.put(GenerationFailed(code, message), terminal=True)

    def _run(self):
        try:
            while True:
                for _ in range(32):
                    try:
                        request, spec, future, timings = self._commands.get_nowait()
                    except queue.Empty:
                        break
                    self._admit(request, spec, future, timings)
                self._apply_controls()
                if self._closing.is_set():
                    result = self.engine.close()
                    self._collect()
                    self._gauges()
                    if result.closed:
                        self._drained = True
                        return
                else:
                    self.engine.step()
                    self._collect()
                    self._gauges()
                    if self._migration is not None:
                        self._migration.poll()
                    self._raise_if_admission_closed()
                self._signal.wait(0.001 if self._handles else 0.05)
                self._signal.clear()
        except BaseException as exc:
            if isinstance(exc, _EngineQuarantined):
                code, message = "engine_quarantined", str(exc)
            elif isinstance(exc, _WorkerFailed):
                code, message = "worker_failed", str(exc)
            else:
                code, message = "engine_unavailable", "engine owner failed"
            self._fail_engine(exc, code=code, message=message)
        finally:
            with self._lock:
                self._stopped.set()
                while True:
                    try:
                        _, _, future, _ = self._commands.get_nowait()
                    except queue.Empty:
                        break
                    if not future.done():
                        future.set_exception(EngineUnavailableError("engine owner stopped"))
                while True:
                    try:
                        _, _, future = self._controls.get_nowait()
                    except queue.Empty:
                        break
                    if not future.done():
                        future.set_exception(EngineUnavailableError("engine owner stopped"))

    def close(self, timeout=10.0) -> bool:
        """False or an exception retains resources; only proven drain permits release."""
        with self._lock:
            self._closing.set()
        self._signal.set()
        self._thread.join(timeout)
        if self._fatal is not None and not self._drained:
            raise RuntimeError("engine owner failed; resources remain retained") from self._fatal
        return self._stopped.is_set()
