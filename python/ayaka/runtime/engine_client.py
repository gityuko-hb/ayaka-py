"""Engine-core process split: wire protocol, server thread, and clients.

Mirrors vLLM V1's ``EngineCore``/``EngineCoreClient`` split at Ayaka's scale.
The engine core owns the serialized engine (one owner thread inside
:class:`~ayaka.runtime.llm.AyakaLLM`) and serves it over two ZMQ channels:

- control: REQ/REP, two frames — a JSON header plus an optional pickled
  payload (submit/cancel/close). Trusted local IPC only;
- output: PUSH/PULL, one pickled message per frame — ``CoreSubmitResult``,
  ``CoreOutputEvent`` (transport-neutral ``OutputEvent``), and
  ``CoreOutputClosed``.

Ordering contract: the client attaches its host-side stream *before* sending
the wire submit, so no published event can be missed; admission results
resolve through ``CoreSubmitResult`` on the output channel, keeping the REP
reply fast. ``InprocEngineCoreClient`` skips the wire entirely and delegates
to :class:`AyakaLLM` — the parity anchor every wire path is tested against.

The child-side entry point (:func:`run_engine_core_process`) is module-level
so ``multiprocessing`` spawn can pickle it together with a picklable
``engine_factory`` callable.
"""

from __future__ import annotations

import json
import pickle
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass

import zmq

from ayaka.request.parallel import expand_parallel_request
from ayaka.request.schema import Request
from ayaka.request.stream import OutputEvent
from ayaka.runtime.collection import OutputCollector, OutputStream
from ayaka.runtime.llm import AyakaLLM

__all__ = [
    "CoreOutputClosed",
    "CoreSubmitResult",
    "EngineCoreClient",
    "EngineCoreServer",
    "InprocEngineCoreClient",
    "ZmqEngineCoreClient",
    "start_threaded_engine_core",
    "start_multiproc_engine_core",
    "run_engine_core_process",
]


@dataclass(frozen=True, slots=True)
class CoreSubmitResult:
    """Admission outcome for one submitted request id."""

    request_id: str
    error: str | None = None


@dataclass(frozen=True, slots=True)
class CoreOutputClosed:
    """Stream closure: no further events will arrive for this request."""

    request_id: str
    finish_reason: str | None = None


class EngineCoreClient:
    """Host-side contract: submit/cancel/close like the local façade."""

    def submit(self, request: Request) -> Future[OutputStream]:  # pragma: no cover
        raise NotImplementedError

    def cancel(self, request_id: str) -> bool:  # pragma: no cover
        raise NotImplementedError

    def close(self, timeout: float = 10.0) -> bool:  # pragma: no cover
        raise NotImplementedError


class InprocEngineCoreClient(EngineCoreClient):
    """Same-process client: a thin delegate to :class:`AyakaLLM`."""

    def __init__(
        self,
        engine,
        *,
        stream_limit: int = 1024,
        idle_interval: float = 0.05,
    ) -> None:
        self._llm = AyakaLLM(engine, stream_limit=stream_limit, idle_interval=idle_interval)

    def submit(self, request: Request) -> Future[OutputStream]:
        return self._llm.submit(request)

    def cancel(self, request_id: str) -> bool:
        return self._llm.cancel(request_id)

    def close(self, timeout: float = 10.0) -> bool:
        return self._llm.close(timeout)

    @property
    def llm(self) -> AyakaLLM:
        return self._llm


class EngineCoreServer:
    """One engine served over ZMQ control/output channels from one thread.

    Socket roles are direction-parameterized: ``bind=True`` (thread mode)
    binds REP/PUSH and exposes the endpoints; ``bind=False`` (child process)
    connects them to the parent's already-bound addresses, so no ports need
    to be reported back through the process start-up race.
    """

    def __init__(
        self,
        engine,
        *,
        idle_interval: float = 0.05,
        stream_limit: int = 1024,
        context: zmq.Context | None = None,
        bind: bool = True,
        control_endpoint: str | None = None,
        output_endpoint: str | None = None,
    ) -> None:
        self._context = context if context is not None else zmq.Context.instance()
        if bind:
            self._rep = self._context.socket(zmq.REP)
            port = self._rep.bind_to_random_port("tcp://127.0.0.1")
            self.control_endpoint = f"tcp://127.0.0.1:{port}"
        else:
            if control_endpoint is None:
                raise ValueError("connect mode requires the control endpoint")
            self._rep = self._context.socket(zmq.REP)
            self._rep.connect(control_endpoint)
            self.control_endpoint = control_endpoint
        self._push = self._context.socket(zmq.PUSH)
        self._push.set_hwm(1 << 16)
        if bind:
            port = self._push.bind_to_random_port("tcp://127.0.0.1")
            self.output_endpoint = f"tcp://127.0.0.1:{port}"
        else:
            if output_endpoint is None:
                raise ValueError("connect mode requires the output endpoint")
            self._push.connect(output_endpoint)
            self.output_endpoint = output_endpoint
        self._push_lock = threading.Lock()
        self._closing = threading.Event()
        self._llm = AyakaLLM(
            engine,
            stream_limit=stream_limit,
            idle_interval=idle_interval,
            on_settled=self._on_settled,
            collector_downstream=self,
        )
        self._poller = zmq.Poller()
        self._poller.register(self._rep, zmq.POLLIN)
        self._thread = threading.Thread(target=self._run, name="ayaka-engine-core", daemon=True)
        self._thread.start()

    def _on_settled(self, request_id: str, finish_reason: str | None) -> None:
        self._send_wire(pickle.dumps(CoreOutputClosed(request_id, finish_reason)))

    def on_output_event(self, event: OutputEvent) -> None:
        """Collector downstream: re-emit every published event over the wire.

        Runs on the façade owner thread mid-step; the push lock keeps the
        socket single-writer.
        """
        self._send_wire(pickle.dumps(event))

    def _run(self) -> None:
        try:
            while not self._closing.is_set() and not self._llm._stopped.is_set():
                if self._poller.poll(0):
                    self._handle_control()
                self._llm._signal.wait(self._llm._idle_interval)
        finally:
            self._llm._thread.join(timeout=10.0)
            with self._push_lock:
                self._push.close(0)
            self._rep.close(0)

    def _handle_control(self) -> None:
        header_raw, payload_raw = self._rep.recv_multipart()
        header = json.loads(header_raw)
        op = header["op"]
        if op == "hello":
            self._reply({"ok": True})
        elif op == "submit":
            request: Request = pickle.loads(payload_raw)
            self._reply({"ok": True})
            try:
                future = self._llm.submit(request)
            except BaseException as exc:
                self._send_wire(pickle.dumps(CoreSubmitResult(str(request.request_id), str(exc))))
                return
            future.add_done_callback(
                lambda done, rid=str(request.request_id): self._admission_done(rid, done)
            )
        elif op == "cancel":
            self._reply({"ok": self._llm.cancel(header["request_id"])})
        elif op == "close":
            try:
                closed = self._llm.close(timeout=header.get("timeout", 10.0))
                self._reply({"ok": bool(closed)})
            finally:
                self._closing.set()
        else:
            self._reply({"ok": False, "error": f"unknown op {op!r}"})

    def _admission_done(self, request_id: str, future: Future) -> None:
        if future.cancelled():
            self._send_wire(pickle.dumps(CoreSubmitResult(request_id, "submission cancelled")))
            return
        exc = future.exception()
        error = None if exc is None else str(exc)
        self._send_wire(pickle.dumps(CoreSubmitResult(request_id, error)))

    def _send_wire(self, raw: bytes) -> None:
        with self._push_lock:
            self._push.send(raw)

    def _reply(self, body: dict) -> None:
        self._rep.send_multipart([json.dumps(body).encode(), b""])


class ZmqEngineCoreClient(EngineCoreClient):
    """Wire client: control REQ + output PULL pump feeding host-side streams.

    The pump thread mirrors the local façade's settlement semantics: events
    push into the collector, ``CoreSubmitResult`` resolves admission futures
    (including per-request errors), and ``CoreOutputClosed`` closes streams.
    Partial consumption cancels over the control channel exactly like
    :class:`AyakaLLM.cancel`.
    """

    def __init__(
        self,
        control_endpoint: str,
        output_endpoint: str,
        *,
        stream_limit: int = 1024,
        context: zmq.Context | None = None,
    ) -> None:
        self._context = context if context is not None else zmq.Context.instance()
        self._req = self._context.socket(zmq.REQ)
        self._req.setsockopt(zmq.RCVTIMEO, 15_000)
        self._req.connect(control_endpoint)
        self._pull = self._context.socket(zmq.PULL)
        self._pull.connect(output_endpoint)
        self._control_lock = threading.Lock()
        self._collector = OutputCollector(limit=stream_limit)
        self._parallel: dict[str, tuple[str, ...]] = {}
        self._admissions: dict[str, Future[CoreSubmitResult]] = {}
        self._lock = threading.Lock()
        self._closing = threading.Event()
        #: Spawned-child handles, set by :func:`start_multiproc_engine_core`.
        self._process: object | None = None
        self._spawn_timeout: float | None = None
        self._poller = zmq.Poller()
        self._poller.register(self._pull, zmq.POLLIN)
        self._thread = threading.Thread(target=self._pump, name="ayaka-core-pump", daemon=True)
        self._thread.start()

    def submit(self, request: Request) -> Future[OutputStream]:
        """Submit over the wire; resolves to the host stream when admitted."""
        request_id = str(request.request_id)
        future: Future[OutputStream] = Future()
        admission: Future[CoreSubmitResult] = Future()
        with self._lock:
            if self._closing.is_set():
                raise RuntimeError("engine core client is closing")
            self._admissions[request_id] = admission
        try:
            if request.sampling.n > 1:
                children = expand_parallel_request(request)
                child_ids = tuple(str(child.request_id) for child in children)
                self._parallel[request_id] = child_ids
                self._collector.attach_parallel(request_id, child_ids)
            else:
                self._collector.attach(request_id)
        except BaseException as exc:
            with self._lock:
                self._admissions.pop(request_id, None)
            self._collector.drop(request_id)
            self._parallel.pop(request_id, None)
            future.set_exception(exc)
            return future
        header = json.dumps({"op": "submit"}).encode()
        with self._control_lock:
            self._req.send_multipart([header, pickle.dumps(request)])
            self._req.recv_multipart()
        admission.add_done_callback(
            lambda done, out=future: self._admission_resolved(request_id, done, out)
        )
        return future

    def cancel(self, request_id: str) -> bool:
        header = json.dumps({"op": "cancel", "request_id": request_id}).encode()
        with self._control_lock:
            self._req.send_multipart([header, b""])
            frames = self._req.recv_multipart()
        reply = json.loads(frames[0])
        return bool(reply.get("ok"))

    def close(self, timeout: float = 10.0) -> bool:
        header = json.dumps({"op": "close", "timeout": timeout}).encode()
        with self._control_lock:
            self._req.send_multipart([header, b""])
            try:
                frames = self._req.recv_multipart()
                reply = json.loads(frames[0])
            except zmq.error.Again:
                reply = {"ok": False}
        self._closing.set()
        self._thread.join(timeout + 2.0)
        with self._lock:
            self._admissions.clear()
        for stream_id in self._collector.active_ids:
            stream = self._collector.get(stream_id)
            if stream is not None:
                stream.close()
            self._collector.drop(stream_id)
        self._parallel.clear()
        self._pull.close(0)
        self._req.close(0)
        self._join_child(timeout)
        return bool(reply.get("ok"))

    def _join_child(self, timeout: float) -> None:
        process = getattr(self, "_process", None)
        if process is None:
            return
        process.join(getattr(self, "_spawn_timeout", None) or timeout)

    # ------------------------------------------------------------------
    # Pump thread: wire → host streams
    # ------------------------------------------------------------------

    def _pump(self) -> None:
        while not self._closing.is_set():
            events = dict(self._poller.poll(50))
            if events.get(self._pull) != zmq.POLLIN:
                continue
            message = pickle.loads(self._pull.recv())
            if isinstance(message, CoreSubmitResult):
                self._resolve_admission(message)
            elif isinstance(message, CoreOutputClosed):
                stream = self._collector.get(message.request_id)
                if stream is not None and not stream.overflowed:
                    stream.close()
                self._collector.drop(message.request_id)
                self._parallel.pop(message.request_id, None)
            elif isinstance(message, OutputEvent):
                self._collector.on_output_event(message)
            else:
                raise TypeError(f"unexpected wire message {type(message)!r}")

    def _resolve_admission(self, result: CoreSubmitResult) -> None:
        with self._lock:
            admission = self._admissions.pop(result.request_id, None)
        if admission is None or admission.done():
            return
        if result.error is not None:
            self._collector.drop(result.request_id)
            self._parallel.pop(result.request_id, None)
            admission.set_exception(RuntimeError(result.error))
        else:
            admission.set_result(result)

    def _admission_resolved(
        self, request_id: str, admission: Future[CoreSubmitResult], future: Future[OutputStream]
    ) -> None:
        if future.done():
            return
        if admission.cancelled():
            return
        exc = admission.exception()
        if exc is not None:
            future.set_exception(exc)
            return
        result = admission.result()
        if result.error is not None:
            future.set_exception(RuntimeError(result.error))
            return
        stream = self._collector.get(request_id)
        if stream is None:
            future.set_exception(RuntimeError(f"stream {request_id!r} vanished before admission"))
            return
        future.set_result(stream)


def start_threaded_engine_core(
    engine,
    *,
    idle_interval: float = 0.05,
    stream_limit: int = 1024,
    context: zmq.Context | None = None,
) -> ZmqEngineCoreClient:
    """Run the engine core on a local thread and return a wired client."""
    server = EngineCoreServer(
        engine,
        idle_interval=idle_interval,
        stream_limit=stream_limit,
        context=context,
    )
    return ZmqEngineCoreClient(
        server.control_endpoint,
        server.output_endpoint,
        stream_limit=stream_limit,
        context=server._context,
    )


def run_engine_core_process(
    engine_factory: Callable[[], object],
    bootstrap_endpoint: str,
    *,
    idle_interval: float = 0.05,
    stream_limit: int = 1024,
    diagnostics_path: str | None = None,
) -> None:
    """Child-process entry: build the engine, bind, report endpoints, serve.

    The child binds its own control/output sockets on random ports and
    reports the endpoints back over the bootstrap PULL, so the parent never
    sits between the client and the server. ``diagnostics_path`` appends
    progress markers to a file — for start-up debugging only.
    """

    def diag(marker: str) -> None:
        if diagnostics_path:
            with open(diagnostics_path, "a", encoding="utf-8") as handle:
                handle.write(f"{marker}\n")

    diag("entry")
    engine = engine_factory()
    diag("engine built")
    server = EngineCoreServer(engine, idle_interval=idle_interval, stream_limit=stream_limit)
    diag("bound")
    context = zmq.Context.instance()
    push = context.socket(zmq.PUSH)
    # Windows quirk: a send issued before the TCP handshake completes is
    # queued but dropped at close even with linger. Wait for POLLOUT so the
    # endpoints report is provably flushable before sending.
    push.connect(bootstrap_endpoint)
    try:
        deadline = time.monotonic() + 30.0
        while not push.getsockopt(zmq.EVENTS) & zmq.POLLOUT:
            if time.monotonic() > deadline:
                diag("pollout timeout")
                raise TimeoutError("bootstrap connect never became writable")
            time.sleep(0.01)
        diag("push writable")
        push.send(
            json.dumps(
                {
                    "control": server.control_endpoint,
                    "output": server.output_endpoint,
                }
            ).encode()
        )
        diag("push sent")
    finally:
        push.close(0)
    server._thread.join()


def _engine_core_child(
    _rank: int,
    engine_factory: Callable[[], object],
    bootstrap_endpoint: str,
    idle_interval: float,
    stream_limit: int,
    diagnostics_path: str | None,
) -> None:
    """torch.multiprocessing entry (rank prepended)."""
    run_engine_core_process(
        engine_factory,
        bootstrap_endpoint,
        idle_interval=idle_interval,
        stream_limit=stream_limit,
        diagnostics_path=diagnostics_path,
    )


def _spawn_engine_core_process(
    engine_factory: Callable[[], object],
    bootstrap_endpoint: str,
    *,
    idle_interval: float,
    stream_limit: int,
    diagnostics_path: str | None = None,
):
    """Spawn the child via ``torch.multiprocessing.spawn``.

    Plain ``multiprocessing`` spawn re-runs the parent's ``__main__`` in the
    child; under ``uv run pytest`` that is a zipimporter over ``pytest.exe``
    and the child stalls before its entry. ``torch.multiprocessing.spawn``
    carries its own preparation (proven by the repo's process tests), so it
    is the supported path for the engine-core child.
    """
    import torch.multiprocessing as torch_mp

    spawn_fn = getattr(torch_mp, "spawn")
    return spawn_fn(
        _engine_core_child,
        args=(
            engine_factory,
            bootstrap_endpoint,
            idle_interval,
            stream_limit,
            diagnostics_path,
        ),
        nprocs=1,
        join=False,
        daemon=True,
    )


def start_multiproc_engine_core(
    engine_factory: Callable[[], object],
    *,
    idle_interval: float = 0.05,
    stream_limit: int = 1024,
    start_timeout: float = 100.0,
    spawn_timeout: float | None = None,
    diagnostics_path: str | None = None,
) -> ZmqEngineCoreClient:
    """Spawn a child process owning the engine; return a wired client.

    The child binds its own control/output sockets and reports the endpoints
    back over a bootstrap PULL the parent binds — no port-reporting race, and
    the client talks straight to the child (no relay in the middle).
    ``spawn_timeout`` overrides the child join timeout at close.
    """

    context = zmq.Context.instance()
    bootstrap = context.socket(zmq.PULL)
    try:
        port = bootstrap.bind_to_random_port("tcp://127.0.0.1")
        bootstrap_endpoint = f"tcp://127.0.0.1:{port}"
        bootstrap.setsockopt(zmq.RCVTIMEO, int(start_timeout * 1000))
        process = _spawn_engine_core_process(
            engine_factory,
            bootstrap_endpoint,
            idle_interval=idle_interval,
            stream_limit=stream_limit,
            diagnostics_path=diagnostics_path,
        )
        try:
            raw = bootstrap.recv()
        except zmq.error.Again as exc:
            raise TimeoutError("engine core process did not start") from exc
        endpoints = json.loads(raw)
        client = ZmqEngineCoreClient(
            endpoints["control"],
            endpoints["output"],
            stream_limit=stream_limit,
            context=context,
        )
        client._process = process
        client._spawn_timeout = spawn_timeout
        return client
    except BaseException:
        bootstrap.close(0)
        raise
