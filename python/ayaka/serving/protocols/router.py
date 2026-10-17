"""FastAPI router for the ported protocols; models are accessed only through ports."""

from __future__ import annotations

import asyncio
import json
import queue
from collections.abc import AsyncGenerator
from contextlib import aclosing
from dataclasses import dataclass
from typing import Any

import anyio
from fastapi import APIRouter, FastAPI, Request, WebSocket, WebSocketDisconnect
from starlette.responses import JSONResponse, StreamingResponse

from . import anthropic, chat, responses
from .common import Journal, dumps, invalid_constant, obj, sse
from .error import ApiError, classify
from .media import MAX_HTTP_BODY_BYTES
from .source import (
    AdaptedRequest,
    Admitted,
    Completed,
    Failed,
    GenerationEvent,
    GenerationStream,
    ModelInvocation,
    ModelLoadProgress,
    Output,
    Progress,
    ServedModels,
)


@dataclass(slots=True)
class _Session:
    invocation: ModelInvocation | None = None

    def observe_write(self) -> None:
        callback = getattr(self.invocation, "observe_write", None)
        if callback is not None:
            callback()


async def _source_events(
    models: ServedModels,
    adapted: AdaptedRequest,
    tenant: str,
    progress: bool,
    session: _Session,
) -> GenerationStream:
    pending: queue.Queue[ModelLoadProgress] = queue.Queue(maxsize=32)
    wake = asyncio.Event()
    loop = asyncio.get_running_loop()

    def loading(value: ModelLoadProgress) -> None:
        try:
            pending.put_nowait(value)
        except queue.Full:
            return
        loop.call_soon_threadsafe(wake.set)

    task = asyncio.create_task(
        models.invoke(adapted.model, loading if progress else None, tenant=tenant)
    )
    try:
        while True:
            wake.clear()
            while not pending.empty():
                value = pending.get_nowait()
                yield Progress(
                    {
                        "type": "model_loading",
                        "stage": value.stage,
                        "fraction": max(0.0, min(1.0, value.fraction)),
                    }
                )
            if task.done():
                break
            waiter = asyncio.create_task(wake.wait())
            try:
                await asyncio.wait((task, waiter), return_when=asyncio.FIRST_COMPLETED)
            finally:
                waiter.cancel()
                await asyncio.gather(waiter, return_exceptions=True)
        session.invocation = await task
        stream = session.invocation.generate(adapted.request)
        admitted = False
        async with aclosing(stream):
            async for event in stream:
                if isinstance(event, Admitted):
                    if admitted:
                        raise ApiError.server("duplicate generation admission")
                    admitted = True
                elif isinstance(event, (Output, Completed)) and not admitted:
                    raise ApiError.server("semantic output arrived before admission")
                yield event
                if isinstance(event, (Completed, Failed)):
                    return
    except Exception as exc:
        yield Failed(classify(exc))
    finally:
        # ASGI may cancel a whole AnyIO scope repeatedly during disconnect.
        # Shield just ownership cleanup, while cancelling the device request.
        with anyio.CancelScope(shield=True):
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            if (
                session.invocation is None
                and task.done()
                and not task.cancelled()
                and task.exception() is None
            ):
                session.invocation = task.result()
            if session.invocation is not None:
                await session.invocation.aclose()


async def _while_connected(request: Request, operation):
    async def disconnected():
        while True:
            if (await request.receive())["type"] == "http.disconnect":
                return

    work = asyncio.create_task(operation)
    watcher = asyncio.create_task(disconnected())
    try:
        done, _ = await asyncio.wait((work, watcher), return_when=asyncio.FIRST_COMPLETED)
        if work in done:
            return await work
        raise ApiError("client disconnected", status=499, code="request_cancelled")
    finally:
        work.cancel()
        watcher.cancel()
        await asyncio.gather(work, watcher, return_exceptions=True)


class _GenerationResponse(StreamingResponse):
    """Close the binding even if headers fail before the iterator starts."""

    def __init__(
        self, body: AsyncGenerator[str], *, source: GenerationStream, session: _Session, **kwargs
    ):
        super().__init__(body, **kwargs)
        self.source, self.session, self.iterator = source, session, body

    async def __call__(self, scope, receive, send):
        async def observed(message):
            if message.get("type") == "http.response.body" and message.get("body"):
                self.session.observe_write()
            await send(message)

        try:
            return await super().__call__(scope, receive, observed)
        finally:
            with anyio.CancelScope(shield=True):
                await self.iterator.aclose()
                await self.source.aclose()


class _GenerationJSONResponse(JSONResponse):
    def __init__(self, content, *, session: _Session, **kwargs):
        super().__init__(content, **kwargs)
        self.session = session

    async def __call__(self, scope, receive, send):
        async def observed(message):
            if message.get("type") == "http.response.body" and message.get("body"):
                self.session.observe_write()
            await send(message)

        return await super().__call__(scope, receive, observed)


def create_router(
    models: ServedModels,
    *,
    max_request_bytes: int = MAX_HTTP_BODY_BYTES,
    max_request_chunks: int = 4096,
) -> APIRouter:
    """Mount with include_router. The containing app owns auth and engine lifespan."""
    router = APIRouter()

    async def body(request: Request) -> dict[str, Any]:
        chunks, size, count = [], 0, 0
        async for chunk in request.stream():
            size += len(chunk)
            count += 1
            if size > max_request_bytes or count > max_request_chunks:
                raise ApiError("request body too large", status=413, code="request_too_large")
            chunks.append(chunk)
        try:
            value = json.loads(b"".join(chunks).decode("utf-8"), parse_constant=invalid_constant)
        except (ValueError, UnicodeError) as exc:
            raise ApiError("body must be valid JSON") from exc
        return obj(value, "body")

    async def generate(request: Request, protocol: str):
        native = protocol == "anthropic"
        journal = Journal(
            "", {"chat": "chatcmpl_", "responses": "resp_", "anthropic": "msg_"}[protocol]
        )
        source = None
        handed_off = False
        try:
            if native:
                anthropic.validate_version(request.headers.get("anthropic-version"))
            adapted = {
                "chat": chat.adapt,
                "responses": responses.adapt,
                "anthropic": anthropic.adapt,
            }[protocol](await body(request))
            journal.model = adapted.model
            if native:
                journal.model = (
                    request.headers.get("magnitude-gateway-model")
                    or request.headers.get("ayaka-gateway-model")
                    or adapted.model
                )
            progress = not native and (
                request.headers.get("magnitude-include-progress", "").lower() == "true"
                or request.headers.get("ayaka-include-progress", "").lower() == "true"
            )
            session = _Session()
            source = _source_events(
                models, adapted, getattr(request.state, "tenant", "anonymous"), progress, session
            )
            before: list[GenerationEvent] = []
            if not (adapted.stream and progress):

                async def preflight():
                    async for event in source:
                        if isinstance(event, Failed):
                            raise event.error
                        before.append(event)
                        if isinstance(event, Admitted):
                            return
                    raise ApiError.server("generation ended without admission")

                await _while_connected(request, preflight())

            async def events() -> GenerationStream:
                for event in before:
                    yield event
                async for event in source:
                    yield event

            if adapted.stream:

                async def framed():
                    async with aclosing(events()) as stream:
                        if protocol == "responses":
                            async with aclosing(
                                responses.project(stream, journal, adapted, progress=progress)
                            ) as values:
                                async for value in values:
                                    yield sse(value, value["type"])
                        else:
                            adapter = (
                                anthropic.project(stream, journal)
                                if native
                                else chat.project(stream, journal, adapted, progress=progress)
                            )
                            async with aclosing(adapter):
                                async for value in adapter:
                                    yield value

                handed_off = True
                return _GenerationResponse(
                    framed(),
                    source=source,
                    session=session,
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                        "x-request-id": journal.id,
                        **({"request-id": journal.id} if native else {}),
                    },
                )

            async def collect():
                async with aclosing(events()) as stream:
                    async for event in stream:
                        if isinstance(event, Admitted):
                            journal.prompt_tokens = event.prompt_tokens
                        elif isinstance(event, Output):
                            journal.observe(event)
                        elif isinstance(event, Completed):
                            journal.finish(event)
                            return
                        elif isinstance(event, Failed):
                            raise event.error
                    raise ApiError.server("generation ended without a terminal event")

            await _while_connected(request, collect())
            payload = (
                responses.response(journal, adapted)
                if protocol == "responses"
                else anthropic.response(journal)
                if native
                else chat.response(journal)
            )
            return _GenerationJSONResponse(
                payload,
                session=session,
                headers={
                    "x-request-id": journal.id,
                    **({"request-id": journal.id} if native else {}),
                },
            )
        except Exception as exc:
            error = classify(exc)
            if error.code == "context_length_exceeded" and error.param is None:
                error.param = "input" if protocol == "responses" else "messages"
            return error.response(anthropic=native, request_id=journal.id)
        finally:
            if source is not None and not handed_off:
                await source.aclose()

    @router.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        return await generate(request, "chat")

    @router.post("/v1/responses")
    async def create_response(request: Request):
        return await generate(request, "responses")

    @router.post("/anthropic/v1/messages")
    @router.post("/v1/messages")
    async def messages(request: Request):
        return await generate(request, "anthropic")

    @router.post("/anthropic/v1/messages/count_tokens")
    @router.post("/v1/messages/count_tokens")
    async def count_tokens(request: Request):
        try:
            anthropic.validate_version(request.headers.get("anthropic-version"))
            adapted = anthropic.adapt(await body(request), count_only=True)
            host = await models.host(adapted.model)
            return {"input_tokens": await asyncio.to_thread(host.count, adapted.request.input)}
        except Exception as exc:
            return classify(exc).response(anthropic=True)

    @router.post("/api/v1/chat/templates/apply")
    async def apply_template(request: Request):
        try:
            value = await body(request)
            allowed = {
                "model",
                "messages",
                "tools",
                "tool_choice",
                "parallel_tool_calls",
                "response_format",
                "chat_template_kwargs",
            }
            if set(value) - allowed:
                raise ApiError("unknown template application field")
            adapted = chat.adapt(value)
            host = await models.host(adapted.model)
            return await asyncio.to_thread(host.apply_template, adapted.request.input)
        except Exception as exc:
            return classify(exc).response()

    @router.post("/api/v1/models/{model_id:path}/properties")
    async def properties(model_id: str):
        try:
            host = await models.host(model_id)
            return await asyncio.to_thread(host.properties)
        except Exception as exc:
            return classify(exc).response()

    @router.websocket("/v1/responses")
    async def websocket(socket: WebSocket):
        await socket.accept()
        inbox: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1)
        gone = asyncio.Event()
        history = responses.WebsocketHistory(max_request_bytes)

        async def reader():
            try:
                while True:
                    frame = await socket.receive()
                    if frame["type"] == "websocket.disconnect":
                        return
                    raw = frame.get("text")
                    if raw is None:
                        raw = frame.get("bytes", b"").decode("utf-8")
                    if len(raw.encode()) > max_request_bytes:
                        await socket.close(code=1009)
                        return
                    try:
                        inbox.put_nowait({"raw": raw})
                    except asyncio.QueueFull:
                        await socket.close(code=1013)
                        return
            except (WebSocketDisconnect, UnicodeError, RuntimeError):
                return
            finally:
                gone.set()

        async def process(raw: str):
            source = None
            previous = None
            try:
                value = json.loads(raw, parse_constant=invalid_constant)
                value, input, previous, generate = history.resolve(value)
                adapted = responses.adapt(value)
                journal = Journal(adapted.model, "resp_")
                if not generate:
                    await models.host(adapted.model)
                    initial = responses.base(journal, adapted)
                    await socket.send_json(
                        {"type": "response.created", "sequence_number": 0, "response": initial}
                    )
                    journal.finish(Completed())
                    await socket.send_json(
                        {
                            "type": "response.completed",
                            "sequence_number": 1,
                            "response": responses.response(journal, adapted),
                        }
                    )
                    history.record(journal.id, input, [])
                    return
                progress = (
                    socket.headers.get("magnitude-include-progress", "").lower() == "true"
                    or socket.headers.get("ayaka-include-progress", "").lower() == "true"
                )
                session = _Session()
                source = _source_events(
                    models, adapted, getattr(socket.state, "tenant", "anonymous"), progress, session
                )
                async with aclosing(
                    responses.project(source, journal, adapted, progress=progress)
                ) as events:
                    async for event in events:
                        session.observe_write()
                        await socket.send_text(dumps(event))
                if journal.completion is not None:
                    history.record(
                        journal.id,
                        input,
                        responses.response(journal, adapted, ordered=True)["output"],
                    )
                else:
                    history.evict(previous)
            except (ValueError, ApiError) as exc:
                history.evict(previous)
                error = (
                    exc
                    if isinstance(exc, ApiError)
                    else ApiError("WebSocket message must be valid JSON")
                )
                await socket.send_json(
                    {
                        "type": "error",
                        "sequence_number": 0,
                        "status": error.status,
                        "error": error.body(),
                    }
                )
            except Exception as exc:
                history.evict(previous)
                error = classify(exc)
                await socket.send_json(
                    {
                        "type": "error",
                        "sequence_number": 0,
                        "status": error.status,
                        "error": error.body(),
                    }
                )
            finally:
                if source is not None:
                    await source.aclose()

        pump = asyncio.create_task(reader())
        disconnected = asyncio.create_task(gone.wait())
        work = None
        take = None
        try:
            while not gone.is_set():
                take = asyncio.create_task(inbox.get())
                done, _ = await asyncio.wait(
                    (take, disconnected), return_when=asyncio.FIRST_COMPLETED
                )
                if disconnected in done:
                    take.cancel()
                    await asyncio.gather(take, return_exceptions=True)
                    return
                work = asyncio.create_task(process(take.result()["raw"]))
                done, _ = await asyncio.wait(
                    (work, disconnected), return_when=asyncio.FIRST_COMPLETED
                )
                if disconnected in done:
                    return
                await work
        except asyncio.CancelledError:
            # Disconnect and server shutdown both cancel this connection's
            # work; no response remains to be delivered to the socket.
            pass
        finally:
            with anyio.CancelScope(shield=True):
                tasks: list[asyncio.Task[Any]] = [
                    t for t in (pump, disconnected, work, take) if t is not None
                ]
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

    return router


def create_app(models: ServedModels, **router_options) -> FastAPI:
    """Standalone protocol app. Use Ayaka's HTTP factory for production auth/lifecycle."""
    app = FastAPI(title="Ayaka Protocols")
    app.include_router(create_router(models, **router_options))
    return app
