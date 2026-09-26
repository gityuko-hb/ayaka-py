"""One asynchronous generation path for every protocol and both delivery modes."""

from __future__ import annotations

import asyncio
import time

from ayaka.request.cancel import CancellationToken
from ayaka.serving.errors import EngineUnavailableError, OverloadedError
from ayaka.serving.events import GenerationFailed, GenerationFinished
from ayaka.serving.prepare import PrepareTimings


class GenerationPipeline:
    """The admission counter covers preprocessing through stream finalization."""

    def __init__(self, service, processor):
        self.service, self.processor = service, processor
        self.config = processor.config
        self.active = 0

    def reserve(self):
        if not self.service.ready:
            raise EngineUnavailableError("engine is not ready")
        limit = self.config.max_concurrent_requests
        if limit and self.active >= limit:
            self.service.stats.record_rejection("concurrent")
            raise OverloadedError("too many concurrent requests")
        self.active += 1

    async def prepare(self, spec, tenant):
        # Called before SSE headers, so all validation/admission errors have HTTP status.
        timings = PrepareTimings(ingress_ns=time.monotonic_ns())
        self.reserve()
        token = CancellationToken()
        future = None
        try:
            request = await asyncio.to_thread(self.processor.prepare, spec, tenant, token, timings)
            future = self.service.submit(request, spec, timings=timings)
            return await asyncio.shield(asyncio.wrap_future(future))
        except BaseException:
            token.cancel("preparation cancelled")
            if future is not None:

                def cancel_late(completed):
                    if not completed.cancelled() and completed.exception() is None:
                        completed.result().cancel()

                future.add_done_callback(cancel_late)
                future.cancel()
            self.active -= 1
            raise

    def release(self, handle):
        """Idempotent cleanup, including a disconnect before SSE iteration starts."""
        handle.cancel()
        if not handle.frontend_released:
            handle.frontend_released = True
            self.active -= 1

    async def generate_events(self, handle):
        try:
            while True:
                event = await handle.next_event()
                yield event
                if isinstance(event, (GenerationFinished, GenerationFailed)):
                    return
        finally:
            self.release(handle)
