"""FastAPI application factory with a shared generation pipeline and lifecycle."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import time
from contextlib import aclosing, asynccontextmanager

from fastapi import FastAPI, Request
from starlette.responses import JSONResponse, Response, StreamingResponse

from ayaka.kvcache.resize import CacheRebuildRejected
from ayaka.serving.errors import (
    InvalidRequestError,
    ServingError,
    cache_resize_error,
)
from ayaka.serving.generation import GenerationPipeline
from ayaka.serving.protocol import (
    Result,
    anthropic_stream,
    normalize,
    openai_stream,
    responses_stream,
)


def error_response(exc: ServingError, path: str):
    error = {"message": exc.message, "type": exc.code, "code": exc.code, "param": exc.param}
    if exc.detail:
        error["detail"] = exc.detail
    payload = {"error": error}
    if path.startswith("/v1/messages"):
        error_type = {
            400: "invalid_request_error",
            401: "authentication_error",
            403: "permission_error",
            404: "not_found_error",
            429: "rate_limit_error",
            503: "overloaded_error",
        }.get(exc.status_code, "api_error")
        payload = {"type": "error", "error": {"type": error_type, "message": exc.message}}
    headers = {"Retry-After": "1"} if exc.status_code == 429 else {}
    return JSONResponse(payload, status_code=exc.status_code, headers=headers)


class AccessMiddleware:
    """Authenticate before parsing; cap actual body bytes including chunked requests."""

    def __init__(self, app, config, stats=None):
        self.app, self.config, self.stats = app, config, stats

    def _reject(self, reason):
        if self.stats is not None:
            self.stats.record_rejection(reason)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path = scope["path"]
        headers = dict(scope["headers"])
        tenant = "anonymous"
        if self.config.api_keys and path not in ("/health", "/healthz", "/readyz", "/ping"):
            authorization = headers.get(b"authorization", b"").decode("latin-1")
            supplied = authorization[7:] if authorization.lower().startswith("bearer ") else ""
            # Anthropic SDK normally uses x-api-key.
            if not supplied and path.startswith("/v1/messages"):
                supplied = headers.get(b"x-api-key", b"").decode("latin-1")
            valid = False
            for key in self.config.api_keys:
                valid |= hmac.compare_digest(supplied.encode(), key.encode())
            if not valid:
                from ayaka.serving.errors import AuthenticationError

                self._reject("auth")
                response = error_response(AuthenticationError("invalid API key"), path)
                return await response(scope, receive, send)
            tenant = hashlib.sha256(supplied.encode()).hexdigest()
        scope.setdefault("state", {})["tenant"] = tenant
        messages, size = [], 0
        if scope["method"] == "POST":
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                size += len(message.get("body", b""))
                if size > self.config.max_request_bytes:
                    self._reject("request_bytes")
                    response = JSONResponse(
                        {
                            "error": {
                                "type": "request_too_large",
                                "message": "request body too large",
                            }
                        },
                        status_code=413,
                    )
                    return await response(scope, receive, send)
                messages.append(message)
                if not message.get("more_body", False):
                    break

        async def replay():
            return messages.pop(0) if messages else await receive()

        await self.app(scope, replay, send)


class GenerationResponse(StreamingResponse):
    """Release admission even when the connection disappears during HTTP headers."""

    def __init__(self, *args, pipeline, handle, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipeline, self.handle = pipeline, handle

    async def __call__(self, scope, receive, send):
        async def stamp_first_write(message):
            if (
                not self.handle.first_socket_write_ns
                and message.get("type") == "http.response.body"
                and message.get("body")
            ):
                self.handle.first_socket_write_ns = time.monotonic_ns()
            await send(message)

        try:
            return await super().__call__(scope, receive, stamp_first_write)
        finally:
            self.pipeline.release(self.handle)


async def _body(request):
    try:
        body = await request.json()
    except (ValueError, UnicodeError) as exc:
        raise InvalidRequestError("body must be valid JSON") from exc
    if not isinstance(body, dict):
        raise InvalidRequestError("body must be a JSON object")
    return body


def _normalize(body, protocol):
    try:
        return normalize(body, protocol)
    except (TypeError, AttributeError, KeyError) as exc:
        raise InvalidRequestError("malformed request fields") from exc


async def _while_connected(request, operation):
    """Race work with the ASGI disconnect event without polling cancel scopes."""

    async def disconnected():
        while True:
            message = await request.receive()
            if message["type"] == "http.disconnect":
                return

    worker = asyncio.create_task(operation)
    watcher = asyncio.create_task(disconnected())
    try:
        done, _ = await asyncio.wait((worker, watcher), return_when=asyncio.FIRST_COMPLETED)
        if worker in done:
            return await worker
        from ayaka.serving.errors import RequestCancelledError

        raise RequestCancelledError("client disconnected")
    finally:
        worker.cancel()
        watcher.cancel()
        await asyncio.gather(worker, watcher, return_exceptions=True)


def create_app(service, processor, *, close=None, admin=None) -> FastAPI:
    pipeline = GenerationPipeline(service, processor)

    @asynccontextmanager
    async def lifespan(app):
        yield
        closer = close or service.close
        if not await asyncio.to_thread(closer):
            raise RuntimeError("serving shutdown is still draining; resources retained")

    app = FastAPI(title="Ayaka Serving", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.pipeline, app.state.service = pipeline, service
    app.add_middleware(
        AccessMiddleware, config=processor.config, stats=getattr(service, "stats", None)
    )

    @app.exception_handler(ServingError)
    async def handle_error(request, exc):
        return error_response(exc, request.url.path)

    @app.get("/health")
    @app.get("/healthz")
    @app.get("/ping")
    async def health():
        return {"status": "ok"}

    @app.get("/readyz")
    async def ready():
        return JSONResponse(
            {"status": "ready" if service.ready else "unavailable"},
            status_code=200 if service.ready else 503,
        )

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [
                {
                    "id": processor.config.model,
                    "object": "model",
                    "created": 0,
                    "owned_by": "ayaka",
                }
            ],
        }

    if processor.config.expose_metrics:

        @app.get("/metrics")
        async def metrics():
            return Response(service.stats.render(), media_type="text/plain; version=0.0.4")

    @app.get("/stats")
    async def stats():
        return {
            "ready": service.ready,
            "frontend_active": pipeline.active,
            "outstanding": service.outstanding,
            "conservation": service.stats.conservation(outstanding=service.outstanding),
            "slo": service.stats.slo_summary(),
            "recent_requests": service.stats.recent(),
        }

    if admin is not None:

        @app.get("/v1/cache/status")
        async def cache_status():
            # Owner-thread dispatch: never read storage/manager while a resize
            # may be rebuilding the tail.
            return await asyncio.to_thread(service.cache_status)

        @app.post("/v1/cache/resize")
        async def cache_resize(request: Request):
            body = await _body(request)
            pages = body.get("pages")
            if not isinstance(pages, int) or isinstance(pages, bool):
                raise InvalidRequestError("pages must be an integer")
            if pages < 2:
                raise InvalidRequestError("pages must be at least 2")
            try:
                status = await asyncio.to_thread(admin.resize, pages)
            except CacheRebuildRejected as exc:
                raise cache_resize_error(exc) from exc
            return status.as_dict()

    async def generate(request: Request, protocol: str):
        body = await _body(request)
        spec = _normalize(body, protocol)
        handle = await _while_connected(request, pipeline.prepare(spec, request.state.tenant))
        result = Result(spec.model)

        async def events():
            async with aclosing(pipeline.generate_events(handle)) as source:
                async for event in source:
                    yield event

        if spec.stream:

            async def stream():
                async with aclosing(events()) as source:
                    if protocol == "anthropic":
                        adapter = anthropic_stream(source, result, handle.request.prompt_len)
                    elif protocol == "responses":
                        adapter = responses_stream(source, result)
                    else:
                        adapter = openai_stream(source, result, spec, protocol)
                    async with aclosing(adapter):
                        async for chunk in adapter:
                            yield chunk

            return GenerationResponse(
                stream(),
                pipeline=pipeline,
                handle=handle,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "X-Request-ID": str(handle.request.request_id),
                },
            )
        try:
            async with aclosing(events()) as source:
                # Also watch disconnect during a non-streaming generation.
                async def collect():
                    async for event in source:
                        result.add(event)

                await _while_connected(request, collect())
            return JSONResponse(
                result.response(protocol), headers={"X-Request-ID": str(handle.request.request_id)}
            )
        finally:
            pipeline.release(handle)

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        return await generate(request, "chat")

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await generate(request, "completion")

    @app.post("/v1/messages")
    async def messages(request: Request):
        return await generate(request, "anthropic")

    @app.post("/v1/responses")
    async def responses(request: Request):
        return await generate(request, "responses")

    @app.post("/v1/messages/count_tokens")
    async def count_tokens(request: Request):
        body = await _body(request)
        # Count the exact rendered prompt, without scheduling model execution.
        body["max_tokens"] = 1
        spec = _normalize(body, "anthropic")
        prepared = await asyncio.to_thread(processor.prepare, spec, request.state.tenant)
        return {"input_tokens": prepared.prompt_len}

    return app
