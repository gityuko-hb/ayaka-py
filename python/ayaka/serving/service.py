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
from ayaka.serving.errors import EngineUnavailableError, InvalidRequestError, OverloadedError
from ayaka.serving.events import GenerationFailed, GenerationFinished, ServingEvent, UsageUpdate
from ayaka.serving.parsers import OutputParser
from ayaka.serving.prepare import GenerationSpec
from ayaka.serving.stats import ServingStats, UsageRecord

_LOG = logging.getLogger(__name__)


def _wake(future: asyncio.Future) -> None:
    if not future.done():
        future.set_result(None)


class GenerationHandle:
    """Bounded, nonblocking producer with a reserved terminal slot."""

    def __init__(self, request: Request, spec: GenerationSpec, config: ServingConfig):
        self.request = request
        self.cancelled = threading.Event()
        self._lock = threading.Lock()
        self._events: deque[ServingEvent] = deque()
        self._waiter: asyncio.Future | None = None
        self._limit = config.max_stream_events
        self.cursor = 0
        self.frontend_released = False
        self.started = time.monotonic()
        self.first_token: float | None = None
        self.error: str | None = None
        self.parser = OutputParser(
            reasoning=config.reasoning_parser == "think"
            and request.constraint is None
            and spec.messages is not None,
            tools=spec.tools if spec.tool_choice != "none" else (),
            constrained_tools=request.constraint is not None
            and request.constraint.kind == "tool_calls",
        )

    def cancel(self) -> None:
        self.cancelled.set()

    def put(self, event: ServingEvent, *, terminal=False) -> bool:
        with self._lock:
            if not terminal and len(self._events) >= self._limit:
                return False
            self._events.append(event)
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
                    return self._events.popleft()
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
    ):
        self.engine, self.config = engine, config
        self.stats = stats or ServingStats()
        self._commands: queue.Queue = queue.Queue(maxsize=config.max_pending_commands)
        self._controls: queue.Queue = queue.Queue()
        self._handles: dict[str, GenerationHandle] = {}
        self._closing = threading.Event()
        self._stopped = threading.Event()
        self._signal = threading.Event()
        self._lock = threading.Lock()
        self._fatal: BaseException | None = None
        self._drained = False
        #: Engine-thread owner of KV rebuilds; ``apply_resize`` is only safe here.
        self._controller = controller
        runner = engine.executor.runner
        runner.set_request_source(engine.requests.get, constraints=constraints)
        self._thread = threading.Thread(target=self._run, name="ayaka-engine", daemon=True)
        self._thread.start()

    @property
    def ready(self) -> bool:
        return not self._closing.is_set() and self._fatal is None and not self._stopped.is_set()

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
            self._controls.put_nowait((pages, future))
        self._signal.set()
        try:
            result = future.result(timeout)
        except TimeoutError as exc:
            raise EngineUnavailableError("cache resize did not complete in time") from exc
        if not isinstance(result, CacheStatus):
            raise EngineUnavailableError("cache resize returned no status")
        return result

    def _apply_controls(self) -> None:
        """Run queued cache controls on the engine thread between steps.

        Raises:
            CacheResizeFatal: When a rebuild cannot be recovered; the run loop
                treats it like any fatal engine error and drains resources.
        """
        while True:
            try:
                pages, future = self._controls.get_nowait()
            except queue.Empty:
                return
            if future.cancelled():
                continue
            controller = self._controller
            if controller is None or self._closing.is_set() or self._fatal is not None:
                future.set_exception(EngineUnavailableError("engine is unavailable"))
                continue
            try:
                if self._handles:
                    raise CacheRebuildRejected(
                        "engine is busy; resize requires an idle server; old cache kept",
                        reason=ResizeRejectionReason.BUSY,
                        requested_pages=pages,
                    )
                result = controller.apply_resize(pages)
            except BaseException as exc:
                self.engine = controller.engine
                if not future.done():
                    future.set_exception(exc)
                if isinstance(exc, CacheResizeFatal):
                    raise
            else:
                self.engine = controller.engine
                future.set_result(result)

    def submit(self, request: Request, spec: GenerationSpec) -> Future[GenerationHandle]:
        future: Future[GenerationHandle] = Future()
        with self._lock:
            if not self.ready:
                raise EngineUnavailableError("engine is unavailable")
            try:
                self._commands.put_nowait((request, spec, future))
            except queue.Full as exc:
                raise OverloadedError("engine ingress queue is full") from exc
        self._signal.set()
        return future

    def _admit(self, request, spec, future):
        if not future.set_running_or_notify_cancel():
            return
        if not self.ready:
            future.set_exception(EngineUnavailableError("engine is closing"))
            return
        try:
            self.engine.submit(request)
        except Exception as exc:
            from ayaka.sched.interfaces import OverloadedError as SchedulerOverloaded

            if isinstance(exc, SchedulerOverloaded):
                error = OverloadedError(str(exc))
            elif isinstance(exc, (ValueError, TypeError)):
                error = InvalidRequestError(str(exc))
            else:
                _LOG.exception("engine admission failed")
                error = EngineUnavailableError("engine admission failed")
            future.set_exception(error)
            return
        handle = GenerationHandle(request, spec, self.config)
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
                if handle.first_token is None and event.usage.completion_tokens:
                    handle.first_token = time.monotonic() - handle.started
                if handle.error or handle.cancelled.is_set():
                    continue
                try:
                    for parsed in handle.parser.feed(event.delta):
                        if not handle.put(parsed):
                            raise OverloadedError("output consumer is too slow")
                except Exception as exc:
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
            if not handle.error and status not in ("cancelled", "abort", "error"):
                try:
                    for parsed in handle.parser.finish():
                        if not handle.put(parsed):
                            raise OverloadedError("output consumer is too slow")
                except Exception as exc:
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
                    else "length"
                    if status == "length"
                    else "stop"
                )
                terminal = GenerationFinished(finish, usage)
            self.engine.forget(request_id)
            self.engine.executor.runner.forget_request(request_id)
            self.engine.requests.forget(request_id)
            del self._handles[request_id]
            self.stats.record(
                UsageRecord(
                    request_id,
                    usage.prompt_tokens,
                    usage.completion_tokens,
                    usage.cached_tokens,
                    status,
                    time.monotonic() - handle.started,
                ),
                first_token=handle.first_token,
            )
            _LOG.info(
                "request completed id=%s status=%s prompt_tokens=%d completion_tokens=%d",
                request_id,
                status,
                usage.prompt_tokens,
                usage.completion_tokens,
            )
            handle.put(terminal, terminal=True)

    def _gauges(self):
        self.stats.running.set(self.engine.scheduler.num_running)
        self.stats.waiting.set(self.engine.scheduler.num_waiting)
        snapshot = self.engine.kv.snapshot()
        self.stats.kv_free.set(snapshot.free_pages)
        self.stats.kv_total.set(snapshot.total_pages)

    def _run(self):
        try:
            while True:
                for _ in range(32):
                    try:
                        request, spec, future = self._commands.get_nowait()
                    except queue.Empty:
                        break
                    self._admit(request, spec, future)
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
                self._signal.wait(0.001 if self._handles else 0.05)
                self._signal.clear()
        except BaseException as exc:
            _LOG.exception("engine owner failed; attempting drain before resource release")
            self._fatal = exc
            try:
                for handle in self._handles.values():
                    handle.error = "engine owner failed"
                result = self.engine.close()
                if result.closed:
                    self._collect()
                    self._gauges()
                    self._drained = True
            except BaseException:
                _LOG.exception("fatal engine drain failed; resources retained")
            for handle in self._handles.values():
                handle.put(
                    GenerationFailed("engine_unavailable", "engine owner failed"), terminal=True
                )
        finally:
            with self._lock:
                self._stopped.set()
                while True:
                    try:
                        _, _, future = self._commands.get_nowait()
                    except queue.Empty:
                        break
                    if not future.done():
                        future.set_exception(EngineUnavailableError("engine owner stopped"))
                while True:
                    try:
                        _, future = self._controls.get_nowait()
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
