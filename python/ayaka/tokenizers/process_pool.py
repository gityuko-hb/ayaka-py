"""Bounded spawn-based encoding and affinity-preserving detokenizer processes."""

from __future__ import annotations

import asyncio
import multiprocessing
import threading
from concurrent.futures import Future, ProcessPoolExecutor
from uuid import uuid4

from ayaka.configs.tokenizer import DetokUpdate, TokenizerConfig
from ayaka.tokenizers.contracts import (
    ChatInput,
    EncodeOptions,
    TemplateInput,
    TextInput,
    TokenIdsInput,
)
from ayaka.tokenizers.service import TokenizerOverloaded, TokenizerService
from ayaka.utils.validation import require_int

_service: TokenizerService | None = None
_DEFAULT_OPTIONS = EncodeOptions()
_decoders = {}


def _initialize(config, max_model_len, model_vocab_size, factory):
    global _service
    _service = (
        factory()
        if factory is not None
        else TokenizerService(
            config, max_model_len=max_model_len, model_vocab_size=model_vocab_size
        )
    )


def _encode(value, options):
    if _service is None:
        raise RuntimeError("tokenizer worker is not initialized")
    return _service.encode(value, options)


def _encode_media(value, options, markers, placeholder):
    if _service is None:
        raise RuntimeError("tokenizer worker is not initialized")
    from ayaka.tokenizers.multimodal import encode_media_chat

    return encode_media_chat(_service, value, options, markers, placeholder)


def _decoder(operation, identity, *args):
    if operation == "create":
        if _service is None:
            raise RuntimeError("tokenizer worker is not initialized")
        params, prompt = args
        _decoders[identity] = _service.new_detokenizer(params, prompt_token_ids=prompt)
        return None
    if operation == "forget":
        _decoders.pop(identity, None)
        return None
    decoder = _decoders[identity]
    if operation == "update":
        tokens, check_stops = args
        return decoder.update(tokens, check_stops=check_stops)
    if operation == "finish":
        try:
            return decoder.finish()
        finally:
            del _decoders[identity]
    raise ValueError("unknown decoder operation")


class ProcessDetokenizer:
    """One request decoder pinned to one worker until finish or close."""

    def __init__(self, pool, worker: int, identity: str) -> None:
        self._pool = pool
        self._worker = worker
        self._identity = identity
        self._finished = False

    def update(self, tokens, *, check_stops=True):
        if self._finished:
            raise RuntimeError("detokenizer is terminal")
        return self._pool._submit(
            self._worker, _decoder, "update", self._identity, tuple(tokens), check_stops
        ).result()

    def finish(self):
        if self._finished:
            return DetokUpdate()
        result = self._pool._submit(self._worker, _decoder, "finish", self._identity).result()
        self._finished = True
        return result

    def close(self) -> None:
        if not self._finished:
            self._pool._submit(self._worker, _decoder, "forget", self._identity).result()
            self._finished = True


class ProcessTokenizerPool:
    """Separate processes for both encode and stateful incremental decoding.

    factory is an optional importable zero-argument adapter factory (useful for
    offline deployments). Work is bounded across queued and running jobs. A
    cancelled future keeps its charge until the child actually stops using it.
    """

    def __init__(
        self,
        config: TokenizerConfig,
        *,
        max_model_len: int,
        model_vocab_size: int,
        workers: int = 2,
        max_pending: int = 64,
        eos_token_ids: tuple[int, ...] = (),
        factory=None,
    ) -> None:
        require_int(workers, "workers", minimum=1)
        require_int(max_pending, "max_pending", minimum=1)
        if config.encode_pool_workers:
            raise ValueError("process workers require inline child tokenizers")
        self.max_model_len = max_model_len
        self.model_vocab_size = model_vocab_size
        self.eos_token_ids = eos_token_ids
        self._capacity = threading.BoundedSemaphore(max_pending)
        self._lock = threading.Lock()
        self._next = 0
        self._pending_bytes = 0
        self._max_pending_bytes = config.encode_max_pending_bytes
        self._closed = False
        self._workers = [
            ProcessPoolExecutor(
                max_workers=1,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize,
                initargs=(config, max_model_len, model_vocab_size, factory),
            )
            for _ in range(workers)
        ]

    def _worker(self) -> int:
        with self._lock:
            worker = self._next % len(self._workers)
            self._next += 1
            return worker

    def _submit(self, worker: int, function, *args, payload_bytes=0) -> Future:
        if not self._capacity.acquire(blocking=False):
            raise TokenizerOverloaded("process tokenizer capacity exhausted")
        try:
            with self._lock:
                if self._closed:
                    raise RuntimeError("process tokenizer pool is closed")
                if self._pending_bytes + payload_bytes > self._max_pending_bytes:
                    raise TokenizerOverloaded("process tokenizer byte capacity exhausted")
                future = self._workers[worker].submit(function, *args)
                self._pending_bytes += payload_bytes
        except BaseException:
            self._capacity.release()
            raise

        def release(_):
            with self._lock:
                self._pending_bytes -= payload_bytes
            self._capacity.release()

        future.add_done_callback(release)
        return future

    def submit(self, value, options=_DEFAULT_OPTIONS) -> Future:
        if isinstance(value, TextInput):
            weight = len(value.text.encode("utf-8"))
        elif isinstance(value, TokenIdsInput):
            weight = 8 * len(value.token_ids)
        elif isinstance(value, ChatInput):
            weight = sum(len(m.content.encode("utf-8")) + len(m.role) for m in value.messages)
            weight += len((value.chat_template or "").encode("utf-8"))
        elif isinstance(value, TemplateInput):
            weight = len(value.messages_json.encode("utf-8")) + len(
                value.tools_json.encode("utf-8")
            )
        else:
            raise TypeError("unsupported encode input")
        weight += len(options.cache_namespace.encode("utf-8"))
        return self._submit(self._worker(), _encode, value, options, payload_bytes=weight)

    def encode(self, value, options=_DEFAULT_OPTIONS):
        return self.submit(value, options).result()

    async def encode_async(self, value, options=_DEFAULT_OPTIONS):
        return await asyncio.wrap_future(self.submit(value, options))

    def encode_multimodal(self, value, options, markers, placeholder):
        weight = sum(len(m.content.encode("utf-8")) + len(m.role) for m in value.messages)
        weight += len((value.chat_template or "").encode("utf-8"))
        weight += len(options.cache_namespace.encode("utf-8")) + sum(8 * n for _, n in markers)
        return self._submit(
            self._worker(),
            _encode_media,
            value,
            options,
            markers,
            placeholder,
            payload_bytes=weight,
        ).result()

    def new_detokenizer(self, params, *, prompt_token_ids=()):
        worker, identity = self._worker(), uuid4().hex
        self._submit(worker, _decoder, "create", identity, params, tuple(prompt_token_ids)).result()
        return ProcessDetokenizer(self, worker, identity)

    def close(self) -> None:
        with self._lock:
            self._closed = True
        for worker in self._workers:
            worker.shutdown(wait=True, cancel_futures=True)
