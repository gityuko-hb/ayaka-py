"""Push-based façade over one serialized Engine with a dedicated owner thread."""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from collections.abc import AsyncIterator, Callable
from concurrent.futures import Future

from ayaka.request.parallel import expand_parallel_request
from ayaka.request.schema import Request
from ayaka.request.stream import OutputEvent
from ayaka.runtime.collection import OutputCollector, OutputListener, OutputStream
from ayaka.sched.interfaces import OverloadedError

__all__ = ["AyakaLLM", "AsyncAyakaLLM"]

_LOG = logging.getLogger(__name__)

#: Commands admitted per owner-loop pass.  Bounded so a sustained submit burst
#: cannot starve ``engine.step()`` while the queue drains over later passes.
_COMMANDS_PER_PASS = 32


class AyakaLLM:
    """Sync façade: submit returns a future to a push output stream."""

    def __init__(
        self,
        engine,
        *,
        stream_limit: int = 1024,
        idle_interval: float = 0.05,
        max_pending_commands: int = 256,
        on_settled: Callable[[str, str | None], None] | None = None,
        collector_downstream: OutputListener | None = None,
    ) -> None:
        if not isinstance(max_pending_commands, int) or isinstance(max_pending_commands, bool):
            raise TypeError("max_pending_commands must be an integer")
        if max_pending_commands < 1:
            raise ValueError("max_pending_commands must be positive")
        self.engine = engine
        self._collector = OutputCollector(limit=stream_limit, downstream=collector_downstream)
        engine.output.set_listener(self._collector)
        self._commands: queue.Queue = queue.Queue(maxsize=max_pending_commands)
        self._closing = threading.Event()
        self._stopped = threading.Event()
        self._signal = threading.Event()
        self._lock = threading.Lock()
        self._fatal: BaseException | None = None
        self._drained = False
        self._idle_interval = idle_interval
        #: Settled-observer: (request_id, finish reason or None); runs on the
        #  owner thread before state is forgotten — the engine-core wire uses
        #  it to signal stream closure across a process boundary.
        self._on_settled = on_settled
        #: Parallel-sampling families this façade opened streams for.
        self._parallel: dict[str, tuple[str, ...]] = {}
        self._thread = threading.Thread(target=self._run, name="ayaka-llm", daemon=True)
        self._thread.start()

    def submit(self, request: Request) -> Future[OutputStream]:
        """Queue admission on the owner thread; resolves to the request's stream.

        Raises immediately when the engine already closed or the bounded
        command queue is full; per-request validation errors are surfaced
        through the future instead.
        """
        future: Future[OutputStream] = Future()
        with self._lock:
            if self._stopped.is_set():
                raise RuntimeError("engine owner is stopped")
            if self._closing.is_set():
                raise RuntimeError("engine is closing")
            try:
                self._commands.put_nowait((request, future))
            except queue.Full as exc:
                raise OverloadedError("engine SDK command queue is full") from exc
        self._signal.set()
        return future

    def cancel(self, request_id: str) -> bool:
        """Thread-safe cancel signal; the owner loop converts it to an abort.

        Parallel parents fan the cancel token out to every child.
        """
        children = self._parallel.get(request_id)
        if children is not None:
            did = False
            for child in children:
                did = self.engine.requests.cancel(child, "cancelled by client") or did
            return did
        return self.engine.requests.cancel(request_id, "cancelled by client")

    def close(self, timeout: float = 10.0) -> bool:
        """Drain the engine; False or an exception retains resources."""
        with self._lock:
            self._closing.set()
        self._signal.set()
        self._thread.join(timeout)
        if self._fatal is not None and not self._drained:
            raise RuntimeError("engine owner failed; resources remain retained") from self._fatal
        return self._stopped.is_set()

    # ------------------------------------------------------------------
    # Owner-thread internals
    # ------------------------------------------------------------------

    def _run(self) -> None:
        try:
            while True:
                self._admit_pending()
                if self._closing.is_set():
                    self._abort_active()
                    result = self.engine.close()
                    self._settle()
                    if result.closed:
                        self._drained = True
                        self._fail_stragglers()
                        return
                else:
                    self.engine.step()
                    self._settle()
                self._signal.wait(0.001 if self._collector.active_ids else self._idle_interval)
                self._signal.clear()
        except BaseException as exc:
            _LOG.exception("engine owner failed; attempting drain before release")
            self._fatal = exc
            try:
                if self.engine.close().closed:
                    self._drained = True
            except BaseException:
                _LOG.exception("fatal engine drain failed; resources may be retained")
            self._settle()
            self._fail_stragglers()
        finally:
            with self._lock:
                self._stopped.set()
                while True:
                    try:
                        _, future = self._commands.get_nowait()
                    except queue.Empty:
                        break
                    if not future.done():
                        future.set_exception(RuntimeError("engine owner stopped"))

    def _abort_active(self) -> None:
        """Abort every open stream so closing never strands a consumer."""
        for request_id in self._collector.active_ids:
            try:
                self.engine.abort(request_id)
            except BaseException:
                _LOG.exception("failed to abort %s while closing", request_id)

    def _fail_stragglers(self) -> None:
        """Force-close streams that settlement could not finish normally."""
        for request_id in self._collector.active_ids:
            stream = self._collector.get(request_id)
            if stream is not None:
                stream.close()
            self._collector.drop(request_id)
            self._notify_settled(request_id)
            self._forget_family(request_id)

    def _admit_pending(self) -> None:
        for _ in range(_COMMANDS_PER_PASS):
            try:
                request, future = self._commands.get_nowait()
            except queue.Empty:
                return
            if not future.set_running_or_notify_cancel():
                continue
            request_id = str(request.request_id)
            try:
                if request.sampling.n > 1:
                    children = expand_parallel_request(request)
                    child_ids = tuple(str(child.request_id) for child in children)
                    self._parallel[request_id] = child_ids
                    stream = self._collector.attach_parallel(request_id, child_ids)
                else:
                    stream = self._collector.attach(request_id)
            except BaseException as exc:
                self._collector.drop(request_id)
                self._parallel.pop(request_id, None)
                future.set_exception(exc)
                continue
            try:
                self.engine.submit(request)
            except BaseException as exc:
                self._collector.drop(request_id)
                self._parallel.pop(request_id, None)
                future.set_exception(exc)
                continue
            future.set_result(stream)

    def _settle(self) -> None:
        """Close finished streams, abort overflowed ones, free output state."""
        for request_id in self._collector.overflowed_requests():
            stream = self._collector.get(request_id)
            self.engine.abort(request_id)
            if stream is not None:
                stream.close()
            self._collector.drop(request_id)
            self._notify_settled(request_id)
            self._forget_family(request_id)
        for request_id in self._collector.active_ids:
            children = self._parallel.get(request_id)
            if children is not None:
                if all(self._child_terminal(child) for child in children):
                    stream = self._collector.get(request_id)
                    if stream is not None and not stream.overflowed:
                        # Aborted/failed children never published a terminal event.
                        stream.close()
                    self._collector.drop(request_id)
                    self._notify_settled(request_id)
                    self._forget_family(request_id)
                continue
            lifecycle = self.engine.requests.find(request_id)
            if lifecycle is None or not lifecycle.is_terminal:
                continue
            stream = self._collector.get(request_id)
            if stream is not None and not stream.overflowed:
                # Aborted/failed incarnations never published a terminal event.
                stream.close()
            self._collector.drop(request_id)
            self._notify_settled(request_id)
            self._forget(request_id)

    def _finish_reason_of(self, request_id: str) -> str | None:
        lifecycle = self.engine.requests.find(request_id)
        if lifecycle is None or lifecycle.finish_reason is None:
            return None
        return lifecycle.finish_reason.value

    def _notify_settled(self, request_id: str) -> None:
        if self._on_settled is None:
            return
        children = self._parallel.get(request_id)
        if children is not None:
            # Family: the first settled child's reason stands in for the parent.
            reason = next(
                (reason for reason in map(self._finish_reason_of, children) if reason is not None),
                None,
            )
        else:
            reason = self._finish_reason_of(request_id)
        try:
            self._on_settled(request_id, reason)
        except BaseException:
            _LOG.exception("settled-observer failed for %s", request_id)

    def _child_terminal(self, child_id: str) -> bool:
        lifecycle = self.engine.requests.find(child_id)
        return lifecycle is None or lifecycle.is_terminal

    def _forget_family(self, request_id: str) -> None:
        """Drop engine-side state for a family; every event reached its stream."""
        children = self._parallel.pop(request_id, None)
        for rid in children if children is not None else (request_id,):
            self._forget(rid)

    def _forget(self, request_id: str) -> None:
        """Drop engine-side state; every event already reached the stream."""
        try:
            self.engine.forget(request_id)
        except KeyError:
            pass
        try:
            self.engine.requests.forget(request_id)
        except ValueError:
            pass


class AsyncAyakaLLM:
    """asyncio façade: ``generate`` streams ``OutputEvent`` values."""

    def __init__(
        self,
        engine,
        *,
        stream_limit: int = 1024,
        idle_interval: float = 0.05,
        max_pending_commands: int = 256,
    ) -> None:
        self._llm = AyakaLLM(
            engine,
            stream_limit=stream_limit,
            idle_interval=idle_interval,
            max_pending_commands=max_pending_commands,
        )

    async def generate(self, request: Request) -> AsyncIterator[OutputEvent]:
        """Yield output events until terminal; aborts when consumed partially.

        The request id must be unique per submission; reusing one makes the
        future fail with the engine's duplicate-id error.
        """
        stream = await asyncio.wrap_future(self._llm.submit(request))
        request_id = str(request.request_id)
        try:
            while True:
                try:
                    yield await stream.next_event()
                except StopAsyncIteration:
                    return
        finally:
            # Terminal streams ignore the cancel token; partial consumption
            # stops the engine from generating tokens nobody reads.
            self._llm.cancel(request_id)

    def cancel(self, request_id: str) -> bool:
        return self._llm.cancel(request_id)

    def close(self, timeout: float = 10.0) -> bool:
        return self._llm.close(timeout)

    @property
    def engine(self):
        return self._llm.engine
