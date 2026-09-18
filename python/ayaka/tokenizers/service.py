"""Bounded preprocessing service; no scheduler, GPU state or request publication.

A zero-worker service is synchronous. Positive workers coalesce submitted jobs
within a bounded window. Adapter calls and cache access are serialized to protect
slow/custom tokenizers. Cancellation cannot interrupt native encode: its result
is discarded and its charge stays held until native work finishes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections import OrderedDict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

from ayaka.configs.tokenizer import DetokenizeParams, EncodeResult, TokenizerConfig
from ayaka.request.cancel import CancellationToken, CancelledError
from ayaka.tokenizers.contracts import (
    ChatInput,
    EncodeInput,
    EncodeOptions,
    TemplateInput,
    TextInput,
    TokenIdsInput,
)
from ayaka.tokenizers.detokenizer import IncrementalDetokenizer
from ayaka.tokenizers.factory import (
    DefaultTokenizerFactory,
    LoadedTokenizer,
    validate_loading_config,
)
from ayaka.utils.validation import require_int


class TokenizerOverloaded(RuntimeError):
    """Queued plus running input exceeds a configured service budget."""


_DEFAULT_OPTIONS = EncodeOptions()
_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class _Job:
    value: EncodeInput
    options: EncodeOptions
    token: CancellationToken
    future: Future[EncodeResult]
    weight: int


class TokenizerService:
    def __init__(
        self,
        config: TokenizerConfig,
        *,
        max_model_len: int,
        model_vocab_size: int,
        loaded: LoadedTokenizer | None = None,
    ) -> None:
        require_int(max_model_len, "max_model_len", minimum=1)
        require_int(model_vocab_size, "model_vocab_size", minimum=1)
        validate_loading_config(config)
        self.config = config
        self.max_model_len = max_model_len
        self.model_vocab_size = model_vocab_size
        self.loaded = loaded if loaded is not None else DefaultTokenizerFactory.from_config(config)
        if not isinstance(self.loaded, LoadedTokenizer):
            raise TypeError("loaded must be LoadedTokenizer")
        caps = self.loaded.capabilities
        if config.skip_tokenizer_init != (caps.backend == "disabled"):
            raise ValueError("skip_tokenizer_init and loaded backend disagree")
        if config.mode in ("hf", "slow") and not config.skip_tokenizer_init:
            if config.mode != caps.backend:
                raise ValueError("requested and loaded backend disagree")
        if caps.chunked_encode:
            raise ValueError("P1 service does not implement chunked encoding")
        if config.detokenize_backend == "bytes" and not caps.byte_decode:
            raise ValueError("loaded adapter cannot use byte decoding")
        if config.detokenize_backend == "stream" and not caps.decode_stream:
            raise ValueError("loaded adapter cannot use stream decoding")
        self.tokenizer = self.loaded.tokenizer
        self._valid_ids = (
            frozenset(self.tokenizer.get_vocab().values()) if self.tokenizer is not None else None
        )
        if self._valid_ids is not None and any(
            type(t) is not int or t < 0 or t >= model_vocab_size for t in self._valid_ids
        ):
            raise ValueError("tokenizer vocabulary exceeds model embedding/logit capacity")
        self._fingerprint = (
            self.tokenizer.fingerprint() if self.tokenizer is not None else "disabled"
        )
        self._cv = threading.Condition()
        self._adapter_lock = threading.RLock()
        self._queue: deque[_Job] = deque()
        self._jobs: dict[int, _Job] = {}
        self._pending = 0
        self._pending_bytes = 0
        self._closed = False
        self._cache: OrderedDict[tuple[str, bool, str], tuple[int, ...]] = OrderedDict()
        self._cache_bytes = 0
        self._pool = (
            ThreadPoolExecutor(
                max_workers=config.encode_pool_workers, thread_name_prefix="ayaka-encode"
            )
            if config.encode_pool_workers
            else None
        )
        self._local = threading.local()
        self._dispatcher = None
        if self._pool is not None:
            self._dispatcher = threading.Thread(
                target=self._dispatch, daemon=True, name="ayaka-encode-dispatch"
            )
            self._dispatcher.start()

    @property
    def pending(self) -> tuple[int, int]:
        """Charged request count and logical payload bytes, including running work."""
        with self._cv:
            return self._pending, self._pending_bytes

    @property
    def eos_token_ids(self) -> tuple[int, ...]:
        """EOS ids for pre-sample masking; empty when the tokenizer is disabled."""
        if self.tokenizer is None or self.tokenizer.eos_token_id is None:
            return ()
        return (int(self.tokenizer.eos_token_id),)

    def _weight(self, value: EncodeInput) -> int:
        if isinstance(value, TextInput):
            return len(value.text.encode("utf-8"))
        if isinstance(value, ChatInput):
            return sum(len(m.content.encode("utf-8")) + len(m.role) for m in value.messages) + len(
                (value.chat_template or "").encode("utf-8")
            )
        if isinstance(value, TemplateInput):
            return len(value.messages_json.encode("utf-8")) + len(value.tools_json.encode("utf-8"))
        if isinstance(value, TokenIdsInput):
            return 8 * len(value.token_ids)
        raise TypeError("expected TextInput, ChatInput or TokenIdsInput")

    def submit(
        self,
        value: EncodeInput,
        options: EncodeOptions = _DEFAULT_OPTIONS,
        *,
        cancellation: CancellationToken | None = None,
    ) -> Future[EncodeResult]:
        """Admit or reject immediately; zero workers also execute inline."""
        if not isinstance(options, EncodeOptions):
            raise TypeError("options must be EncodeOptions")
        if cancellation is not None and not isinstance(cancellation, CancellationToken):
            raise TypeError("cancellation must be CancellationToken")
        token = cancellation if cancellation is not None else CancellationToken()
        token.raise_if_cancelled()
        weight = self._weight(value) + len(options.cache_namespace.encode("utf-8"))
        if self.tokenizer is None and not isinstance(value, TokenIdsInput):
            raise ValueError("tokenizer disabled: only token-ID input is available")
        if (
            isinstance(value, (TextInput, ChatInput, TemplateInput))
            and not self.loaded.capabilities.text_encode
        ):
            raise ValueError("loaded tokenizer does not support text encoding")
        if isinstance(value, (ChatInput, TemplateInput)) and not self.loaded.capabilities.chat:
            raise ValueError("loaded tokenizer does not support chat templates")
        if options.max_output_tokens >= self.max_model_len:
            raise ValueError("output reservation leaves no room for a prompt")
        job = _Job(value, options, token, Future(), weight)
        with self._cv:
            if self._closed:
                raise RuntimeError("tokenizer service is closed")
            if (
                self._pending >= self.config.encode_max_pending
                or self._pending_bytes + weight > self.config.encode_max_pending_bytes
            ):
                raise TokenizerOverloaded("tokenizer input capacity exhausted")
            self._pending += 1
            self._pending_bytes += weight
            self._jobs[id(job)] = job
            if self._pool is not None:
                self._queue.append(job)
                self._cv.notify_all()
        if self._pool is None:
            self._run_batch([job])
        return job.future

    def encode(
        self,
        value: EncodeInput,
        options: EncodeOptions = _DEFAULT_OPTIONS,
        *,
        cancellation: CancellationToken | None = None,
    ) -> EncodeResult:
        """Blocking convenience over the same submit path."""
        if self._pool is not None and getattr(self._local, "in_worker", False):
            raise RuntimeError("blocking encode must not run from a completion callback")
        return self.submit(value, options, cancellation=cancellation).result()

    async def encode_async(
        self,
        value: EncodeInput,
        options: EncodeOptions = _DEFAULT_OPTIONS,
        *,
        cancellation: CancellationToken | None = None,
    ) -> EncodeResult:
        """Await background encoding; use positive workers to avoid blocking the loop."""
        if self._pool is None:
            raise ValueError("encode_async requires encode_pool_workers > 0")
        token = cancellation if cancellation is not None else CancellationToken()
        try:
            return await asyncio.wrap_future(self.submit(value, options, cancellation=token))
        except asyncio.CancelledError:
            token.cancel("async encode cancelled")
            raise

    def _dispatch(self) -> None:
        assert self._pool is not None
        while True:
            with self._cv:
                self._cv.wait_for(lambda: self._queue or self._closed)
                if self._closed and not self._queue:
                    return
                deadline = time.monotonic() + self.config.encode_batch_window_ms / 1000
                while not self._closed and len(self._queue) < self.config.encode_max_batch:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._cv.wait(remaining)
                batch = [
                    self._queue.popleft()
                    for _ in range(min(len(self._queue), self.config.encode_max_batch))
                ]
            try:
                self._pool.submit(self._run_batch, batch)
            except Exception as exc:
                for job in batch:
                    job.future.set_running_or_notify_cancel()
                    self._finish_job(job, exc)

    def _finish_job(self, job: _Job, result: EncodeResult | BaseException) -> None:
        with self._cv:
            self._pending -= 1
            self._pending_bytes -= job.weight
            del self._jobs[id(job)]
            self._cv.notify_all()
        previous = getattr(self._local, "in_worker", False)
        self._local.in_worker = True
        try:
            if not job.future.cancelled():
                if job.token.is_cancelled:
                    job.future.set_exception(CancelledError("encode cancelled"))
                elif isinstance(result, BaseException):
                    job.future.set_exception(result)
                else:
                    job.future.set_result(result)
        except BaseException:
            # Future invokes user callbacks inline after committing its result.
            # A fatal callback must not strand the other jobs in this batch.
            _LOGGER.exception("encode completion callback failed")
        finally:
            self._local.in_worker = previous

    def _run_batch(self, batch: list[_Job]) -> None:
        previous = getattr(self._local, "in_worker", False)
        self._local.in_worker = True
        active = []
        for job in batch:
            if job.future.set_running_or_notify_cancel():
                active.append(job)
            else:
                self._finish_job(job, CancelledError("encode cancelled"))
        results: list[EncodeResult | BaseException]
        try:
            with self._adapter_lock:
                results = self._encode_batch(active)
        except BaseException as exc:
            results = [exc] * len(active)
        try:
            for job, result in zip(active, results, strict=True):
                self._finish_job(job, result)
        finally:
            self._local.in_worker = previous

    def _render(self, value: TextInput | ChatInput | TemplateInput) -> tuple[str, bool]:
        if isinstance(value, TextInput):
            return value.text, value.add_special_tokens
        assert self.tokenizer is not None
        if isinstance(value, TemplateInput):
            return self.tokenizer.apply_chat_template(
                json.loads(value.messages_json), tools=json.loads(value.tools_json)
            ), False
        return self.tokenizer.apply_chat_template(
            [{"role": m.role, "content": m.content} for m in value.messages],
            add_generation_prompt=value.add_generation_prompt,
            chat_template=value.chat_template,
            continue_final_message=value.continue_final_message,
        ), False

    def _cache_put(self, key: tuple[str, bool, str], ids: tuple[int, ...]) -> None:
        payload = len(key[0].encode("utf-8")) + len(key[2].encode("utf-8"))
        size = payload + 8 * len(ids)
        if size > self.config.session_cache_max_bytes:
            return
        old = self._cache.pop(key, None)
        if old is not None:
            self._cache_bytes -= payload + 8 * len(old)
        self._cache[key] = ids
        self._cache_bytes += size
        while (
            len(self._cache) > self.config.session_cache_capacity
            or self._cache_bytes > self.config.session_cache_max_bytes
        ):
            removed, value = self._cache.popitem(last=False)
            self._cache_bytes -= (
                len(removed[0].encode("utf-8")) + len(removed[2].encode("utf-8")) + 8 * len(value)
            )

    def _encode_batch(self, jobs: list[_Job]) -> list[EncodeResult | BaseException]:
        results: list[EncodeResult | BaseException] = [RuntimeError("unprocessed encode")] * len(
            jobs
        )
        groups: dict[bool, list[tuple[int, str, tuple[str, bool, str]]]] = {}
        rendered_bytes = 0
        for i, job in enumerate(jobs):
            try:
                job.token.raise_if_cancelled()
                if isinstance(job.value, TokenIdsInput):
                    results[i] = self._result(job.value.token_ids, job.options, reused=False)
                    continue
                text, special = self._render(job.value)
                size = len(text.encode("utf-8"))
                if rendered_bytes + size > self.config.encode_max_pending_bytes:
                    raise TokenizerOverloaded("rendered batch exceeds input budget")
                rendered_bytes += size
                key = (text, special, job.options.cache_namespace)
                cached = self._cache.get(key) if self.config.enable_session_cache else None
                if cached is not None:
                    assert self.tokenizer is not None
                    if self.config.verify_session_cache:
                        actual = tuple(self.tokenizer.encode(text, add_special_tokens=special))
                        if actual != cached:
                            raise ValueError("encode-cache verification mismatch")
                    self._cache.move_to_end(key)
                    results[i] = self._result(cached, job.options, reused=True)
                else:
                    groups.setdefault(special, []).append((i, text, key))
            except Exception as exc:
                results[i] = exc
        for special, group in groups.items():
            assert self.tokenizer is not None
            try:
                encoded = self.tokenizer.encode_batch(
                    [text for _, text, _ in group], add_special_tokens=special
                )
                if len(encoded) != len(group):
                    raise ValueError("encode_batch cardinality mismatch")
            except Exception as exc:
                for i, _, _ in group:
                    results[i] = exc
                continue
            for (i, _, key), ids in zip(group, encoded, strict=True):
                try:
                    jobs[i].token.raise_if_cancelled()
                    owned = tuple(ids)
                    results[i] = self._result(owned, jobs[i].options, reused=False)
                    if self.config.enable_session_cache:
                        self._cache_put(key, owned)
                except Exception as exc:
                    results[i] = exc
        return results

    def _result(
        self, ids: tuple[int, ...], options: EncodeOptions, *, reused: bool
    ) -> EncodeResult:
        for token in ids:
            require_int(token, "encoded token id")
            if token >= self.model_vocab_size or (
                self._valid_ids is not None and token not in self._valid_ids
            ):
                raise ValueError(f"token ID is not supported by model/tokenizer: {token}")
        limit = self.max_model_len - options.max_output_tokens
        if options.max_input_tokens is not None:
            limit = min(limit, options.max_input_tokens)
        truncated = len(ids) > limit
        if truncated:
            if not options.truncate:
                raise ValueError("prompt plus output reservation exceeds context capacity")
            ids = ids[-limit:] if self.config.truncation_side == "left" else ids[:limit]
        if not ids:
            raise ValueError("empty encoded prompt")
        return EncodeResult(
            ids,
            len(ids) if reused else 0,
            truncated,
            extra=(("tokenizer_fingerprint", self._fingerprint),),
        )

    def new_detokenizer(
        self, params: DetokenizeParams, *, prompt_token_ids: tuple[int, ...] = ()
    ) -> IncrementalDetokenizer:
        if self.tokenizer is None or not self.loaded.capabilities.text_decode:
            raise ValueError("tokenizer disabled: text output is unavailable")
        with self._adapter_lock:
            with self._cv:
                if self._closed:
                    raise RuntimeError("tokenizer service is closed")
            decoder = IncrementalDetokenizer.create(
                self.tokenizer,
                params,
                prompt_token_ids=prompt_token_ids,
                backend=self.config.detokenize_backend,
            )
            # The per-request decoder and encoder share the same adapter resource.
            decoder._lock = self._adapter_lock
            return decoder

    def close(self, *, cancel_pending: bool = True) -> None:
        """Stop admission and join native work; safe to call repeatedly."""
        if getattr(self._local, "in_worker", False):
            raise RuntimeError("close must not run from an encode completion callback")
        with self._cv:
            self._closed = True
            cancelled = [
                job for job in self._jobs.values() if cancel_pending and not job.future.running()
            ]
            self._cv.notify_all()
        for job in cancelled:
            job.token.cancel("tokenizer service closed")
        if self._dispatcher is not None:
            self._dispatcher.join()
        if self._pool is not None:
            self._pool.shutdown(wait=True)
        with self._cv:
            self._cv.wait_for(lambda: self._pending == 0)
        with self._adapter_lock:
            self._cache.clear()
            self._cache_bytes = 0

    def __enter__(self) -> TokenizerService:
        return self

    def __exit__(self, *_args) -> None:
        self.close()
