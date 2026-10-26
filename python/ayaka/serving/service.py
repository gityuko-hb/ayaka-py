"""One owner thread serializes admission, engine execution and terminal cleanup."""

from __future__ import annotations

import asyncio
import itertools
import logging
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, InvalidStateError

from ayaka.configs.serving import ServingConfig
from ayaka.executor.base import TicketAge
from ayaka.kvcache.resize import (
    CacheRebuildRejected,
    CacheResizeFatal,
    ResizeRejectionReason,
)
from ayaka.kvcache.status import CacheStatus
from ayaka.request.schema import Request
from ayaka.request.states import RequestState
from ayaka.request.stream import OutputHistoryExpired
from ayaka.sched.forecast import OutputLengthForecaster
from ayaka.sched.outcome import FinishReason
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
    TokenLogprobs,
    ToolCallArgumentsDelta,
    ToolCallEnd,
    ToolCallStart,
    UsageUpdate,
)
from ayaka.serving.flush import (
    FlushOperation,
    FlushResult,
    FlushStatus,
    validate_flush_timeout,
)
from ayaka.serving.load import LoadSnapshot
from ayaka.serving.migration import MigrationController
from ayaka.serving.parsers import OutputParser
from ayaka.serving.prepare import GenerationSpec, PrepareTimings
from ayaka.serving.profiling import (
    ProfilerBusyError,
    ProfileSession,
    ProfileSessionResult,
    ProfileSessionStatus,
    validate_profile_options,
)
from ayaka.serving.stats import ServingStats, UsageRecord
from ayaka.serving.weights import WeightOperation, WeightUpdateResult, WeightUpdateStatus
from ayaka.weights.update import WeightSource

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


#: Estimated wire bytes of one reported token or alternative (text, bytes
#: array, logprob and JSON framing); token text is rendered after the queue.
LOGPROB_ENTRY_BYTES = 64


def _logprob_bytes(event: ServingEvent) -> int:
    """Estimated payload of a logprob report, charged to its own byte ceiling."""
    if isinstance(event, TokenLogprobs):
        return LOGPROB_ENTRY_BYTES * (1 + len(event.top))
    return 0


class _EngineQuarantined(RuntimeError):
    """A ticket's device completion is unknown; resources are retained."""


class _WorkerFailed(RuntimeError):
    """The device worker left READY; admission must stop."""


class _FlushFailed(RuntimeError):
    """A prefix-cache clear raised mid-way; state may be partial, admission stops."""


class _WeightUpdateFailed(RuntimeError):
    """A weight update failed without a verified restore; admission stops."""


#: Finishes that end a request on its own terms (scored by the R2 forecast).
_NATURAL_FINISHES = frozenset({"stop", "eos", "length"})

#: How long a flush caller waits past its own timeout for the owner's answer.
_FLUSH_REPLY_GRACE_SECONDS = 10.0


def _cancel_reason_label(lifecycle) -> str:
    """Map an internal cancellation cause to a bounded telemetry label."""
    finish = lifecycle.finish_reason
    if finish is FinishReason.TIMEOUT:
        return "deadline"
    if finish is FinishReason.QUEUE_TIMEOUT:
        return "queue_timeout"
    reason = lifecycle.token.reason
    if reason is None:
        return "unknown"
    text = str(reason).lower()
    if "disconnect" in text:
        return "disconnect"
    if "client" in text:
        return "client"
    if "deadline" in text:
        return "deadline"
    if "queue timeout" in text:
        return "queue_timeout"
    if "shutdown" in text or "closing" in text or "stopping" in text:
        return "shutdown"
    if "cancel" in text:
        return "cancel"
    return "other"


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
        self._logprob_byte_limit = config.max_stream_logprob_bytes
        self._queued_logprob_bytes = 0
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
        #: R2 shadow forecast made at admission (None when the probe is off).
        self.forecast: int | None = None
        self.error: str | None = None
        #: Set by the service on admission: wakes the owner loop on cancel.
        self._wake: Callable[[], None] | None = None
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
        wake = self._wake
        if wake is not None:
            wake()  # the owner aborts promptly instead of at its next poll

    def put(self, event: ServingEvent, *, terminal=False) -> bool:
        """Queue one event; returns False when the consumer is over budget.

        The terminal event has a reserved slot and is delivered exactly once,
        regardless of how full the queue is. Non-terminal events must fit the
        event-count ceiling and their byte ceiling: UTF-8 text for content,
        a separate estimate for logprob reports. ``overflow`` names the budget
        that rejected the event so the caller can record a bounded reason.
        """
        size = _event_bytes(event)
        logprob_size = _logprob_bytes(event)
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
            elif self._queued_logprob_bytes + logprob_size > self._logprob_byte_limit:
                self.overflow = "logprob_bytes"
                return False
            self._events.append(event)
            self._queued_bytes += size
            self._queued_logprob_bytes += logprob_size
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
                    self._queued_logprob_bytes -= _logprob_bytes(event)
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

    #: R2 shadow forecaster (instance attribute when enabled); off by default.
    forecaster: OutputLengthForecaster | None = None

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
        forecaster: OutputLengthForecaster | None = None,
        clock: Callable[[], int] | None = None,
    ):
        self.engine, self.config = engine, config
        self._clock = clock or time.monotonic_ns
        self.stats = stats or ServingStats(clock=self._clock, history_size=config.slo_history_size)
        self._commands: queue.Queue = queue.Queue(maxsize=config.max_pending_commands)
        control_limit = config.max_pending_controls or config.max_pending_commands
        self._controls: queue.Queue = queue.Queue(maxsize=control_limit)
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
        #: Optional R2 shadow forecaster: scored at terminal, never consulted.
        self.forecaster = forecaster
        #: The one parked cache flush (engine thread only); see ``flush_cache``.
        self._flush: FlushOperation | None = None
        self._flush_ids = itertools.count(1)
        #: The one parked weight update (engine thread only); see ``update_weights``.
        self._weights: WeightOperation | None = None
        self._weight_ids = itertools.count(1)
        #: The active profiling window (engine thread only) and the last result.
        self._profile: ProfileSession | None = None
        self._profile_ids = itertools.count(1)
        self._last_profile: ProfileSessionResult | None = None
        #: Prompt tokens of queued, not yet admitted commands (under ``_lock``).
        self._queued_tokens = 0
        #: The ticket that tripped the completion watchdog, while it stays stuck.
        self._stall: TicketAge | None = None
        self._load_versions = itertools.count(1)
        self._load = LoadSnapshot(
            version=0,
            taken_ns=self._clock(),
            ready=False,
            live_generations=0,
            queued_commands=0,
            queued_prompt_tokens=0,
            running=0,
            waiting=0,
            free_pages=0,
            total_pages=0,
        )
        runner = engine.executor.runner
        runner.set_request_source(engine.requests.get, constraints=constraints)
        self._thread = threading.Thread(target=self._run, name="ayaka-engine", daemon=True)
        self._thread.start()

    @property
    def ready(self) -> bool:
        """Readiness requires a live owner and an executor that accepts work.

        A quarantined ticket or a worker that left READY makes this False even
        before the owner loop observes it, so admission stops immediately. So
        does a ticket the completion watchdog found stuck, until it settles.
        """
        if self._closing.is_set() or self._fatal is not None or self._stopped.is_set():
            return False
        if self._stall is not None:
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

    def flush_cache(self, *, timeout: float = 0.0, drain: bool = False) -> FlushResult:
        """Drop the prefix cache once no owner can read KV (engine owner thread).

        ``timeout`` is how long, in finite seconds, a busy engine may take to
        become quiescent; ``0`` answers ``BUSY`` at once. ``drain=True`` also
        holds new admissions while the flush waits, so a steady stream of
        requests cannot starve it; held requests keep their queue deadlines.
        A flush issued while another waits joins it and gets the same outcome
        (its own ``timeout``/``drain`` are not applied). ``FAILED`` means the
        clear raised; the engine then stops admitting work, fail-closed.

        Raises:
            TypeError, ValueError: For a non-numeric, NaN, infinite, negative or
                oversized timeout, or a non-boolean ``drain``.
            EngineUnavailableError: When the engine is closing, has no cache
                controller, or did not answer in time.
        """
        seconds = validate_flush_timeout(timeout)
        if type(drain) is not bool:
            raise TypeError("drain must be a boolean")
        if not self.ready:
            raise EngineUnavailableError("engine is unavailable")
        future: Future = Future()
        with self._lock:
            try:
                self._controls.put_nowait(("flush", (seconds, drain), future))
            except queue.Full as exc:
                self.stats.record_rejection("control")
                raise EngineUnavailableError("engine control queue is full") from exc
        self._signal.set()
        try:
            result = future.result(seconds + _FLUSH_REPLY_GRACE_SECONDS)
        except TimeoutError as exc:
            future.cancel()  # the owner drops a cancelled caller, never answers it
            raise EngineUnavailableError("cache flush did not answer in time") from exc
        if not isinstance(result, FlushResult):
            raise EngineUnavailableError("cache flush returned no result")
        return result

    def update_weights(
        self,
        source: WeightSource,
        *,
        revision: str,
        timeout: float = 30.0,
        abort_running: bool = False,
        backup: bool = True,
        probe=None,
    ) -> WeightUpdateResult:
        """Replace the serving weights as one transaction (see ``ayaka.serving.weights``).

        Admission is held from the moment the owner picks the command up. With
        ``abort_running`` live requests are aborted; otherwise they finish on
        the old weights within ``timeout`` (finite seconds) or the update times
        out untouched. ``revision`` must differ from the serving one: it becomes
        part of the prefix-cache identity. ``backup=False`` saves the host copy
        but turns any failure after mutation into a fail-stop.

        Raises:
            TypeError, ValueError: For an invalid timeout, revision or flags.
            EngineUnavailableError: When the engine is closing, has no weight
                controller, or did not answer in time.
        """
        seconds = validate_flush_timeout(timeout)
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError("revision must be a nonempty string")
        if type(abort_running) is not bool or type(backup) is not bool:
            raise TypeError("abort_running and backup must be booleans")
        if not isinstance(source, WeightSource):
            raise TypeError("source must implement WeightSource")
        if not self.ready:
            raise EngineUnavailableError("engine is unavailable")
        future: Future = Future()
        payload = (source, revision, seconds, abort_running, backup, probe)
        with self._lock:
            try:
                self._controls.put_nowait(("weights", payload, future))
            except queue.Full as exc:
                self.stats.record_rejection("control")
                raise EngineUnavailableError("engine control queue is full") from exc
        self._signal.set()
        try:
            result = future.result(seconds + _FLUSH_REPLY_GRACE_SECONDS)
        except TimeoutError as exc:
            future.cancel()
            raise EngineUnavailableError("weight update did not answer in time") from exc
        if not isinstance(result, WeightUpdateResult):
            raise EngineUnavailableError("weight update returned no result")
        return result

    def start_profile(
        self,
        *,
        max_steps: int | None = None,
        trace_path: str | None = None,
        activities: tuple[str, ...] = ("cpu",),
        recorder_factory=None,
        timeout: float = 10.0,
    ) -> int:
        """Start one bounded profiling window on the owner thread; returns its id.

        Raises:
            ValueError: For invalid options, before any state changes.
            ProfilerBusyError: When a session is already active.
            RuntimeError: When the recorder failed to start (the session is
                recorded as FAILED in :meth:`last_profile`; nothing keeps running).
            EngineUnavailableError: When the engine is closing or did not answer.
        """
        options = validate_profile_options(max_steps, trace_path, activities)
        return self._control("profile_start", (options, recorder_factory), timeout)

    def stop_profile(self, *, timeout: float = 30.0) -> ProfileSessionResult:
        """Stop the active window and return its result.

        Raises:
            ProfilerBusyError: When no session is active.
        """
        return self._control("profile_stop", None, timeout)

    def last_profile(self) -> ProfileSessionResult | None:
        """The most recently finished session (stopped, auto-stopped or failed)."""
        return self._last_profile

    def _control(self, command: str, payload, timeout: float):
        if not self.ready:
            raise EngineUnavailableError("engine is unavailable")
        future: Future = Future()
        with self._lock:
            try:
                self._controls.put_nowait((command, payload, future))
            except queue.Full as exc:
                self.stats.record_rejection("control")
                raise EngineUnavailableError("engine control queue is full") from exc
        self._signal.set()
        try:
            return future.result(timeout)
        except TimeoutError as exc:
            future.cancel()
            raise EngineUnavailableError(f"{command} did not answer in time") from exc

    def _start_profile(self, payload, future: Future) -> None:
        (max_steps, trace_path, activities), recorder_factory = payload
        if self._profile is not None:
            future.set_exception(
                ProfilerBusyError(f"profiling session {self._profile.session} is active")
            )
            return
        session = ProfileSession(
            session=next(self._profile_ids),
            max_steps=max_steps,
            trace_path=trace_path,
            activities=activities,
            **({"recorder_factory": recorder_factory} if recorder_factory is not None else {}),
        )
        try:
            session.start()
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller as FAILED
            self._last_profile = ProfileSessionResult(
                session.session, ProfileSessionStatus.FAILED, 0, 0.0, reason=f"start: {exc}"
            )
            future.set_exception(RuntimeError(f"profiler failed to start: {exc}"))
            return
        self._profile = session
        future.set_result(session.session)

    def _end_profile(self, reason: str) -> ProfileSessionResult | None:
        session, self._profile = self._profile, None
        if session is None:
            return None
        result = session.finish(reason)
        self._last_profile = result
        return result

    def _weights_result(
        self, operation: int, status: WeightUpdateStatus, *, started_ns: int, **fields
    ) -> WeightUpdateResult:
        controller = self._controller
        revision, generation = controller.weights_identity() if controller else ("", 0)
        return WeightUpdateResult(
            status=status,
            operation=operation,
            revision_before=fields.pop("revision_before", revision),
            revision_after=revision,
            generation=generation,
            waited_seconds=max(0, self._clock() - started_ns) / 1e9,
            **fields,
        )

    def _start_weight_update(self, payload, future: Future) -> None:
        source, revision, seconds, abort_running, backup, probe = payload
        now = self._clock()
        operation = next(self._weight_ids)
        controller = self._controller
        assert controller is not None  # _apply_controls refuses without one
        refusal = (
            "another weight update is in progress"
            if self._weights is not None
            else controller.weight_update_refusal(revision)
        )
        if refusal is not None:
            future.set_result(
                self._weights_result(
                    operation, WeightUpdateStatus.REFUSED, started_ns=now, detail=refusal
                )
            )
            self.stats.weight_updates.labels(WeightUpdateStatus.REFUSED.value).inc()
            return
        aborted: tuple[str, ...] = ()
        if abort_running:
            aborted = tuple(self._handles)
            for request_id in aborted:
                self.engine.abort(request_id, reason="weight update")
        self._weights = WeightOperation(
            operation=operation,
            started_ns=now,
            deadline_ns=now + int(seconds * 1e9),
            source=source,
            revision=revision,
            backup=backup,
            probe=probe,
            future=future,
            aborted=aborted,
        )

    def _poll_weights(self) -> None:
        """Run the parked update once quiescent; time it out otherwise."""
        operation = self._weights
        if operation is None:
            return
        if operation.future.cancelled():
            self._weights = None  # the caller gave up before anything was touched
            return
        blockers = self._quiescence_blockers()
        if blockers and self._clock() < operation.deadline_ns:
            return
        self._weights = None
        if blockers:
            self._answer_weights(
                operation,
                self._weights_result(
                    operation.operation,
                    WeightUpdateStatus.TIMEOUT,
                    started_ns=operation.started_ns,
                    aborted_requests=operation.aborted,
                    blockers=blockers,
                ),
            )
            return
        controller = self._controller
        assert controller is not None
        revision_before, _ = controller.weights_identity()
        outcome = controller.apply_weight_update(
            operation.source,
            revision=operation.revision,
            backup=operation.backup,
            probe=operation.probe,
        )
        result = self._weights_result(
            operation.operation,
            outcome.status,
            started_ns=operation.started_ns,
            revision_before=revision_before,
            digest_before=outcome.old_digests.get("target", ""),
            digest_after=outcome.new_digests.get("target", ""),
            mutated=outcome.mutated,
            backup_bytes=outcome.backup_bytes,
            load_seconds=outcome.load_seconds,
            aborted_requests=operation.aborted,
            detail=outcome.detail,
        )
        self._answer_weights(operation, result)
        if outcome.status is WeightUpdateStatus.FAILED:
            raise _WeightUpdateFailed(f"weight update failed: {outcome.detail}")

    def _answer_weights(self, operation: WeightOperation, result: WeightUpdateResult) -> None:
        self.stats.weight_updates.labels(result.status.value).inc()
        if not operation.future.cancelled():
            try:
                operation.future.set_result(result)
            except InvalidStateError:  # cancelled concurrently
                pass

    def _quiescence_blockers(self) -> tuple[str, ...]:
        """Every owner that may still read or write KV; empty means flush-safe.

        Queued, not yet admitted commands hold nothing and do not block.
        """
        blockers: list[str] = []
        if self._handles:
            blockers.append(f"{len(self._handles)} live generations")
        engine = self.engine
        if engine.has_unfinished:
            blockers.append("scheduler has unfinished requests")
        executor = getattr(engine, "executor", None)
        if executor is not None:
            tickets = len(tuple(getattr(executor, "tickets", ())))
            if tickets:
                blockers.append(f"{tickets} unretired tickets")
            if getattr(executor, "pending_completion", False):
                blockers.append("a completion fence is pending")
            if getattr(executor, "quarantined", ()):
                blockers.append("quarantined tickets retain resources")
        controller = self._controller
        if controller is not None:
            blockers.extend(controller.flush_blockers())
        return tuple(blockers)

    def _start_flush(self, seconds: float, drain: bool, future: Future) -> None:
        pending = self._flush
        if pending is not None:
            pending.waiters.append(future)  # coalesce: one destructive operation
            return
        now = self._clock()
        operation = FlushOperation(
            operation=next(self._flush_ids),
            started_ns=now,
            deadline_ns=now + int(seconds * 1e9),
            drain=drain,
            waiters=[future],
        )
        blockers = self._quiescence_blockers()
        if not blockers:
            self._run_flush(operation)
        elif seconds == 0:
            self._answer_flush(operation, FlushStatus.BUSY, blockers=blockers)
        else:
            self._flush = operation

    def _poll_flush(self) -> None:
        """Resolve the parked flush: quiescent wins, then the deadline."""
        operation = self._flush
        if operation is None:
            return
        if not operation.live_waiters():
            # Every caller gave up: nothing to flush and the fence lifts.
            self._flush = None
            self.stats.cache_flushes.labels("abandoned").inc()
            return
        blockers = self._quiescence_blockers()
        if not blockers:
            self._flush = None
            self._run_flush(operation)
        elif self._clock() >= operation.deadline_ns:
            self._flush = None
            self._answer_flush(operation, FlushStatus.TIMEOUT, blockers=blockers)

    def _run_flush(self, operation: FlushOperation) -> None:
        controller = self._controller
        assert controller is not None
        before = self.engine.kv.snapshot().free_pages
        try:
            released = controller.flush_prefix_cache()
        except BaseException as exc:
            self._answer_flush(operation, FlushStatus.FAILED, detail=str(exc))
            raise _FlushFailed(f"prefix cache flush failed: {exc}") from exc
        after = self.engine.kv.snapshot().free_pages
        self._answer_flush(
            operation,
            FlushStatus.FLUSHED,
            released=released,
            free_before=before,
            free_after=after,
        )

    def _answer_flush(
        self,
        operation: FlushOperation,
        status: FlushStatus,
        *,
        blockers: tuple[str, ...] = (),
        released: int = 0,
        free_before: int = 0,
        free_after: int = 0,
        detail: str = "",
    ) -> None:
        callers = len(operation.live_waiters())
        result = FlushResult(
            status=status,
            operation=operation.operation,
            released_refs=released,
            free_pages_before=free_before,
            free_pages_after=free_after,
            callers=callers,
            waited_seconds=max(0, self._clock() - operation.started_ns) / 1e9,
            blockers=blockers,
            detail=detail,
        )
        operation.answer(result)
        self.stats.cache_flushes.labels(status.value).inc()

    def _apply_controls(self) -> None:
        """Run queued cache controls on the engine thread between steps.

        Raises:
            CacheResizeFatal: When a rebuild cannot be recovered; the run loop
                treats it like any fatal engine error and drains resources.
            _FlushFailed: When a cache flush raised; handled the same way.
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
            if command == "flush":
                seconds, drain = payload
                self._start_flush(seconds, drain, future)
                continue
            if command == "weights":
                self._start_weight_update(payload, future)
                continue
            if command == "profile_start":
                self._start_profile(payload, future)
                continue
            if command == "profile_stop":
                result = self._end_profile("stop")
                if result is None:
                    future.set_exception(ProfilerBusyError("no active profiling session"))
                else:
                    future.set_result(result)
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
        tokens = request.prompt_len
        token_limit = self.config.max_pending_prompt_tokens
        with self._lock:
            if not self.ready:
                raise EngineUnavailableError("engine is unavailable")
            self.stats.conservation_submitted.inc()
            if token_limit and self._queued_tokens and self._queued_tokens + tokens > token_limit:
                # Weigh ingress by prompt tokens, not only by command count, so a
                # few huge prompts cannot pin unbounded host memory in the queue.
                self.stats.record_rejection("ingress_tokens")
                raise OverloadedError("engine ingress token budget is full")
            try:
                self._commands.put_nowait((request, spec, future, timings))
            except queue.Full as exc:
                self.stats.record_rejection("ingress")
                raise OverloadedError("engine ingress queue is full") from exc
            self._queued_tokens += tokens
        self._signal.set()
        return future

    def _dequeued(self, request: Request) -> None:
        """Return a command's prompt tokens to the ingress budget."""
        with self._lock:
            self._queued_tokens -= request.prompt_len

    @property
    def outstanding(self) -> int:
        """Requests offered but not settled: queued commands plus live handles."""
        return len(self._handles) + self._commands.qsize()

    @property
    def stall(self) -> TicketAge | None:
        """The ticket holding the service not ready, while it stays unsettled."""
        return self._stall

    def load_snapshot(self) -> LoadSnapshot:
        """The latest detached load snapshot; safe from any thread, never blocks."""
        return self._load

    def load_report(self) -> dict[str, object]:
        """``load_snapshot`` as a dict, aged on this service's clock."""
        return self._load.as_dict(now_ns=self._clock())

    def _admit(self, request, spec, future, timings=None):
        if not future.set_running_or_notify_cancel():
            # The submitter withdrew before the owner picked the command up;
            # the request was already counted as submitted.
            self.stats.record_rejection("withdrawn")
            return
        if not self.ready:
            self.stats.record_rejection("engine")
            future.set_exception(EngineUnavailableError("engine is closing"))
            return
        admitted_limit = self.config.max_admitted_requests
        if admitted_limit and len(self._handles) >= admitted_limit:
            # Refuse before engine admission: no lifecycle, sequence or KV is
            # ever allocated for a request the service cannot track.
            self.stats.record_rejection("admitted")
            future.set_exception(OverloadedError("too many admitted requests"))
            return
        queue_deadline = request.queue_deadline_ns
        if queue_deadline is not None and queue_deadline <= self._clock():
            # The request waited past its admission bound in the service queue:
            # refuse before the engine can allocate anything.
            self.stats.record_rejection("queue_timeout")
            future.set_exception(DeadlineExceededError("queue timeout elapsed before admission"))
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
                self.stats.record_rejection("deadline")
                error = DeadlineExceededError(str(exc))
            elif isinstance(exc, SchedulerOverloaded):
                self.stats.record_rejection("scheduler")
                error = OverloadedError(str(exc))
            elif isinstance(exc, (ValueError, TypeError)):
                self.stats.record_rejection("invalid")
                error = InvalidRequestError(str(exc))
            else:
                self.stats.record_rejection("engine")
                _LOG.exception("engine admission failed")
                error = EngineUnavailableError("engine admission failed")
            future.set_exception(error)
            return
        handle = GenerationHandle(request, spec, self.config)
        handle.admitted_ns = self._clock()
        if self.forecaster is not None:
            handle.forecast = self.forecaster.predict(request.prompt_len, request.stop.max_tokens)
        if timings is not None:
            handle.ingress_ns = timings.ingress_ns
            handle.validation_done_ns = timings.validation_done_ns
            handle.tokenize_done_ns = timings.tokenize_done_ns
        handle._wake = self._signal.set
        self._handles[str(request.request_id)] = handle
        future.set_result(handle)

    def _withhold(self, handle: GenerationHandle) -> None:
        """Count one published token that will never reach its stream."""
        reason = handle.overflow or ("cancelled" if handle.cancelled.is_set() else "error")
        self.stats.stream_tokens_withheld.labels(reason).inc()

    def _collect(self):
        for request_id, handle in tuple(self._handles.items()):
            lifecycle = self.engine.requests.get(request_id)
            if handle.cancelled.is_set() and not lifecycle.is_terminal:
                self.engine.abort(request_id, reason="client cancelled")
            try:
                events = self.engine.events_since(request_id, handle.cursor)
            except OutputHistoryExpired as exc:
                # The consumer fell behind the bounded output history, so the
                # skipped tokens can never be replayed. Fail only this request,
                # like any other slow consumer. Resuming at the retained base keeps
                # the cursor in step; with the error set, the loop below only
                # records token timing and queues nothing.
                if not handle.error:
                    handle.overflow = "history"
                    handle.error = str(exc)
                    self.stats.stream_overflows.labels(handle.overflow).inc()
                    self.engine.abort(request_id)
                skipped = exc.base - handle.cursor
                self.stats.stream_tokens_published.inc(skipped)
                self.stats.stream_tokens_withheld.labels("history").inc(skipped)
                handle.cursor = exc.base
                events = self.engine.events_since(request_id, handle.cursor)
            for event in events:
                handle.cursor += 1
                self.stats.stream_tokens_published.inc()
                if event.usage.completion_tokens:
                    now_ns = self._clock()
                    if handle.first_token is None:
                        handle.first_token = time.monotonic() - handle.started
                        handle.first_published_ns = now_ns
                    handle.last_published_ns = now_ns
                if handle.error or handle.cancelled.is_set():
                    self._withhold(handle)
                    continue
                try:
                    # The token's report precedes the text it produced, so a
                    # stream adapter can attach it to that text's chunk.
                    if event.logprobs is not None and not handle.put(event.logprobs):
                        raise OverloadedError("output consumer is too slow")
                    for parsed in handle.parser.feed(event.delta):
                        if not handle.put(parsed):
                            raise OverloadedError("output consumer is too slow")
                except Exception as exc:
                    if handle.overflow is not None:
                        self.stats.stream_overflows.labels(handle.overflow).inc()
                    handle.error = str(exc)
                    self.engine.abort(request_id)
                    self._withhold(handle)
                else:
                    self.stats.stream_tokens_forwarded.inc()
            if not lifecycle.is_terminal:
                continue
            if handle.cursor != len(lifecycle.output_token_ids):
                # Every published token emits exactly one output event; a gap
                # here means the stream ledger and the lifecycle disagree.
                self.stats.stream_token_mismatches.inc()
                _LOG.warning(
                    "token conservation mismatch id=%s events=%d usage=%d",
                    request_id,
                    handle.cursor,
                    len(lifecycle.output_token_ids),
                )
            reason = lifecycle.finish_reason
            cancelled = lifecycle.state is RequestState.CANCELLED
            if lifecycle.state is RequestState.FAILED:
                handle.error = handle.error or "model execution failed"
            status = reason.value if reason is not None else ("cancelled" if cancelled else "error")
            if handle.cancelled.is_set():
                status = "cancelled"
            if not handle.error and status not in (
                "cancelled",
                "abort",
                "error",
                "timeout",
                "queue_timeout",
            ):
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
                    if status in ("timeout", "queue_timeout")
                    else "length"
                    if status == "length"
                    else "stop"
                )
                terminal = GenerationFinished(finish, usage)
            cancel_reason = ""
            if status in ("cancelled", "abort"):
                cancel_reason = _cancel_reason_label(lifecycle)
                self.stats.record_cancellation(cancel_reason)
            terminal_ns = self._clock()
            if (
                self.forecaster is not None
                and handle.forecast is not None
                and status in _NATURAL_FINISHES
            ):
                # Natural finishes only: cancels, timeouts and errors cut a
                # request short and say nothing about its output length.
                self.forecaster.observe(
                    usage.prompt_tokens,
                    usage.completion_tokens,
                    censored=status == "length",
                    forecast=handle.forecast,
                )
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
                    cleanup_done_ns=self._clock(),
                    itl_p50_ns=int(itl_p50_ns),
                    itl_p95_ns=int(itl_p95_ns),
                    cancel_reason=cancel_reason,
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
        scheduler_stats = getattr(self.engine.scheduler, "stats", None)
        if scheduler_stats is not None:
            self.stats.update_scheduler_stats(scheduler_stats)
        runner = getattr(self.engine.executor, "runner", None)
        graph_snapshot = getattr(runner, "graph_stats", None)
        self.stats.update_graph_stats(graph_snapshot() if graph_snapshot is not None else None)
        oldest = self._watch_completion()
        if self.engine.kv.closed:
            self.stats.kv_free.set(0)
            self.stats.kv_total.set(0)
            self.stats.clear_tiering()
            self._publish_load(oldest, free_pages=0, total_pages=0)
            return
        snapshot = self.engine.kv.snapshot()
        self.stats.kv_free.set(snapshot.free_pages)
        self.stats.kv_total.set(snapshot.total_pages)
        tier = self.engine.kv.tier_metrics()
        if tier is not None:
            self.stats.update_tiering(tier)
        self.stats.update_pressure(self.engine.kv.pressure_metrics)
        self._publish_load(oldest, free_pages=snapshot.free_pages, total_pages=snapshot.total_pages)

    def _watch_completion(self) -> TicketAge | None:
        """Track the oldest unsettled ticket; hold readiness while one is stuck.

        A trip records which ticket, step and requests are waiting and stops
        admission; resources stay retained (a slow device is not a failed one).
        Readiness returns once that ticket settles or ages back under the limit.
        """
        executor = getattr(self.engine, "executor", None)
        oldest_of = getattr(executor, "oldest_unsettled", None)
        oldest = oldest_of() if oldest_of is not None else None
        self.stats.oldest_ticket_age.set(0.0 if oldest is None else oldest.age_ns / 1e9)
        limit = self.config.completion_watchdog_seconds
        if limit is None:
            return oldest
        stuck = oldest is not None and oldest.age_ns > limit * 1e9
        if stuck and self._stall is None:
            self.stats.completion_stalls.inc()
            _LOG.warning("completion watchdog: %s", oldest.as_dict() if oldest else None)
        elif not stuck and self._stall is not None:
            _LOG.warning("completion watchdog cleared: %s", self._stall.as_dict())
        self._stall = oldest if stuck else None
        return oldest

    def _publish_load(self, oldest: TicketAge | None, *, free_pages: int, total_pages: int) -> None:
        """Swap in a fresh immutable load snapshot (one reference assignment)."""
        with self._lock:
            queued_tokens = self._queued_tokens
        scheduler = self.engine.scheduler
        self._load = LoadSnapshot(
            version=next(self._load_versions),
            taken_ns=self._clock(),
            ready=self.ready,
            live_generations=len(self._handles),
            queued_commands=self._commands.qsize(),
            queued_prompt_tokens=queued_tokens,
            running=scheduler.num_running,
            waiting=scheduler.num_waiting,
            free_pages=free_pages,
            total_pages=total_pages,
            oldest_ticket_age_seconds=None if oldest is None else oldest.age_ns / 1e9,
            stalled=self._stall is not None,
        )

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

    def _idle_wait_seconds(self, progressed: bool = True) -> float:
        """How long the owner may sleep before its next iteration.

        Work in flight polls every millisecond, as before. When live requests
        exist but the step moved nothing and nothing needs polling (no
        completion fence, tier transfer or migration), the owner sleeps up to
        50 ms, waking at the nearest request deadline; arrivals, controls and
        cancels set the signal and wake it at once. Parked flush/update
        deadlines bound every sleep.
        """
        if not self._handles:
            wait = 0.05
        elif progressed or self._needs_polling():
            wait = 0.001
        else:
            wait = 0.05
            now = self._clock()
            for lifecycle in self.engine.requests:
                for deadline in (
                    lifecycle.request.deadline_ns,
                    lifecycle.request.queue_deadline_ns,
                ):
                    if deadline is not None:
                        wait = min(wait, max(deadline - now, 0) / 1e9)
        for operation in (self._flush, self._weights):
            if operation is not None:
                remaining = (operation.deadline_ns - self._clock()) / 1e9
                wait = min(wait, max(remaining, 0.0))
        return wait

    def _needs_polling(self) -> bool:
        """Progress that no signal announces: fences, tier copies, migrations."""
        executor = getattr(self.engine, "executor", None)
        if executor is not None and getattr(executor, "pending_completion", False):
            return True
        if self._migration is not None:
            return True
        kv = getattr(self.engine, "kv", None)
        if kv is not None and not getattr(kv, "closed", True):
            tier = kv.tier_metrics()
            if tier is not None and tier.pending_transfers:
                return True
        return False

    def _fail_parked(self, error: BaseException) -> None:
        """Answer every parked maintenance caller instead of stranding it."""
        flush, self._flush = self._flush, None
        if flush is not None:
            flush.fail(error)
        weights, self._weights = self._weights, None
        if weights is not None and not weights.future.done():
            try:
                weights.future.set_exception(error)
            except InvalidStateError:
                pass

    def _run(self):
        try:
            while True:
                # A draining flush or a weight update fences admission: queued
                # requests wait (and keep their queue deadlines) until it resolves.
                fenced = self._weights is not None or (
                    self._flush is not None and self._flush.drain
                )
                for _ in range(0 if fenced else 32):
                    try:
                        request, spec, future, timings = self._commands.get_nowait()
                    except queue.Empty:
                        break
                    self._dequeued(request)
                    self._admit(request, spec, future, timings)
                self._apply_controls()
                progressed = True
                if self._closing.is_set():
                    self._fail_parked(EngineUnavailableError("engine is closing"))
                    self._end_profile("shutdown")
                    result = self.engine.close()
                    self._collect()
                    self._gauges()
                    if result.closed:
                        self._drained = True
                        return
                else:
                    progressed = self.engine.step()
                    if self._profile is not None and self._profile.on_step():
                        self._end_profile("max_steps")
                    self._collect()
                    self._poll_flush()
                    self._poll_weights()
                    self._gauges()
                    if self._migration is not None:
                        self._migration.poll()
                    self._raise_if_admission_closed()
                self._signal.wait(self._idle_wait_seconds(progressed))
                self._signal.clear()
        except BaseException as exc:
            if isinstance(exc, _EngineQuarantined):
                code, message = "engine_quarantined", str(exc)
            elif isinstance(exc, _WorkerFailed):
                code, message = "worker_failed", str(exc)
            elif isinstance(exc, _FlushFailed):
                code, message = "cache_flush_failed", str(exc)
            elif isinstance(exc, _WeightUpdateFailed):
                code, message = "weight_update_failed", str(exc)
            else:
                code, message = "engine_unavailable", "engine owner failed"
            self._fail_engine(exc, code=code, message=message)
        finally:
            self._fail_parked(EngineUnavailableError("engine owner stopped"))
            self._end_profile("shutdown")
            with self._lock:
                self._stopped.set()
                while True:
                    try:
                        request, _, future, _ = self._commands.get_nowait()
                    except queue.Empty:
                        break
                    self._queued_tokens -= request.prompt_len  # _lock is already held
                    if not future.done():
                        self.stats.record_rejection("engine")
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
