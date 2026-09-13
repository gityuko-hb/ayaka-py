"""Append-only incremental decoding with explicit stop and finalization state.

Unsupported decoder prefix rewrites fail instead of silently corrupting output.
The window fallback uses reference decoding for correctness (quadratic in output
length); bytes/DecodeStream avoid reference decoding on every token.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from threading import RLock

from ayaka.configs.tokenizer import DetokenizeParams, DetokUpdate
from ayaka.tokenizers.ports import TokenizerLike
from ayaka.tokenizers.stop_checker import StopChecker
from ayaka.utils.validation import require_int


class IncrementalDetokenizer(ABC):
    def __init__(self, params: DetokenizeParams) -> None:
        if not isinstance(params, DetokenizeParams):
            raise TypeError("params must be DetokenizeParams")
        self._params = params
        self._stop = StopChecker(params.stop, params.include_stop_str_in_output)
        self._text: list[str] = []
        self._n_chars = 0
        self._finished = False
        self._stopped = False
        self._checking_stops: bool | None = None
        self._lock = RLock()

    @staticmethod
    def create(
        tokenizer: TokenizerLike,
        params: DetokenizeParams,
        *,
        prompt_token_ids: Sequence[int] = (),
        backend: str = "auto",
    ) -> IncrementalDetokenizer:
        """Select a supported path; explicit unsupported choices never fall back."""
        if backend not in ("auto", "bytes", "stream", "window"):
            raise ValueError(f"unknown detokenizer backend: {backend}")
        if not isinstance(params, DetokenizeParams):
            raise TypeError("params must be DetokenizeParams")
        byte_ok = getattr(tokenizer, "byte_path_verified", False)
        # Byte tables have no policy for inserting spaces around visible specials.
        if backend in ("auto", "bytes") and byte_ok and params.skip_special_tokens:
            from ayaka.tokenizers.byte_detok import ByteIncrementalDetokenizer

            return ByteIncrementalDetokenizer(tokenizer, params, prompt_token_ids=prompt_token_ids)
        if backend == "bytes":
            raise ValueError("bytes requires verified bytes and skip_special_tokens=True")
        if backend in ("auto", "stream"):
            stream = tokenizer.new_decode_stream(skip_special_tokens=params.skip_special_tokens)
            if stream is not None:
                return FastIncrementalDetokenizer(
                    stream, params, tokenizer=tokenizer, prompt_token_ids=prompt_token_ids
                )
            if backend == "stream":
                raise ValueError("tokenizer does not provide DecodeStream")
        return SlowIncrementalDetokenizer(tokenizer, params, prompt_token_ids=prompt_token_ids)

    def _record(self, text: str) -> str:
        self._n_chars += len(text)
        if text and self._params.accumulate_text:
            self._text.append(text)
        return text

    def update(self, new_token_ids: Sequence[int], *, check_stops: bool = True) -> DetokUpdate:
        """Consume tokens until the first stop; reject updates after termination.

        check_stops=False is for tokens before min_tokens. Switching is legal
        only before any stop matching has begun; no stop spans that boundary.
        """
        with self._lock:
            return self._update(new_token_ids, check_stops=check_stops)

    def _update(self, new_token_ids: Sequence[int], *, check_stops: bool) -> DetokUpdate:
        if self._finished or self._stopped:
            raise RuntimeError("detokenizer is terminal")
        if type(check_stops) is not bool:
            raise TypeError("check_stops must be bool")
        ids = tuple(new_token_ids)
        for token in ids:
            require_int(token, "token id")
            self._validate_token(token)
        if ids:
            if not check_stops and self._checking_stops is True:
                raise ValueError("stop matching cannot be disabled after it starts")
            self._checking_stops = check_stops
        emitted: list[str] = []
        matched = None
        consumed = 0
        for token in ids:
            try:
                raw = self._decode_next(token)
            except Exception:
                self._finished = True
                raise
            consumed += 1
            text, matched = self._stop.feed(raw) if check_stops else (raw, None)
            emitted.append(self._record(text))
            if matched is not None:
                self._stopped = True
                break
        delta = "".join(emitted)
        return DetokUpdate(
            delta=delta,
            stop_matched=matched,
            stalled=bool(consumed) and not delta,
            consumed_tokens=consumed,
        )

    def finish(self) -> DetokUpdate:
        """Flush decoder AND stop buffer exactly once; repeated finish is empty."""
        with self._lock:
            return self._finish()

    def _finish(self) -> DetokUpdate:
        if self._finished:
            return DetokUpdate()
        self._finished = True
        if self._stopped:
            return DetokUpdate()
        tail = self._decode_tail()
        emit, matched = self._stop.feed(tail) if self._checking_stops is not False else (tail, None)
        # Even a non-empty decoder tail can remain entirely in the stop buffer.
        emit += self._stop.flush()
        self._stopped = matched is not None
        return DetokUpdate(delta=self._record(emit), stop_matched=matched)

    @property
    def output_text(self) -> str:
        with self._lock:
            if not self._params.accumulate_text:
                raise RuntimeError("output_text requires accumulate_text=True")
            return "".join(self._text)

    @property
    def n_chars(self) -> int:
        with self._lock:
            return self._n_chars

    @abstractmethod
    def _validate_token(self, token_id: int) -> None: ...

    @abstractmethod
    def _decode_next(self, token_id: int) -> str: ...

    def _decode_tail(self) -> str:
        return ""


class _ReferenceDecoder(IncrementalDetokenizer):
    def __init__(
        self,
        tokenizer: TokenizerLike,
        params: DetokenizeParams,
        *,
        prompt_token_ids: Sequence[int] = (),
    ) -> None:
        super().__init__(params)
        self._tok = tokenizer
        self._valid_ids = frozenset(tokenizer.get_vocab().values())
        self._ids = list(prompt_token_ids)
        for token in self._ids:
            require_int(token, "prompt token id")
            self._validate_token(token)
        self._prefix = self._decode()
        self._raw_text = ""

    def _validate_token(self, token_id: int) -> None:
        if token_id not in self._valid_ids:
            raise ValueError(f"unknown token id: {token_id}")

    def _decode(self) -> str:
        return self._tok.decode(
            self._ids,
            skip_special_tokens=self._params.skip_special_tokens,
            spaces_between_special_tokens=self._params.spaces_between_special_tokens,
        )

    def _reference_output(self) -> str:
        full = self._decode()
        if not full.startswith(self._prefix):
            raise ValueError("decoder rewrote the prompt boundary; cannot stream append-only")
        return full[len(self._prefix) :]

    def _decode_tail(self) -> str:
        reference = self._reference_output()
        if not reference.startswith(self._raw_text):
            raise ValueError("decoder rewrote emitted text; cannot stream append-only")
        return reference[len(self._raw_text) :]


class FastIncrementalDetokenizer(_ReferenceDecoder):
    """DecodeStream with final reference verification; stream faults propagate.

    No implicit reset: resetting without replay loses decoder context. The caller
    must fail the request if a stream faults after emission.
    """

    def __init__(
        self,
        stream,
        params: DetokenizeParams,
        *,
        tokenizer: TokenizerLike,
        prompt_token_ids: Sequence[int] = (),
    ) -> None:
        super().__init__(tokenizer, params, prompt_token_ids=prompt_token_ids)
        self._stream = stream
        self._pieces: list[str] = []
        for token in self._ids:
            self._stream.step(token)

    def _decode_next(self, token_id: int) -> str:
        self._ids.append(token_id)
        text = self._stream.step(token_id) or ""
        self._pieces.append(text)
        return text

    def _decode_tail(self) -> str:
        self._raw_text = "".join(self._pieces)
        return super()._decode_tail()


class SlowIncrementalDetokenizer(_ReferenceDecoder):
    """Correctness-first reference fallback; buffers incomplete UTF-8 tails."""

    def _decode_next(self, token_id: int) -> str:
        self._ids.append(token_id)
        reference = self._reference_output()
        if reference.endswith("\ufffd"):
            return ""
        if not reference.startswith(self._raw_text):
            raise ValueError("decoder rewrote emitted text; cannot stream append-only")
        delta = reference[len(self._raw_text) :]
        self._raw_text = reference
        return delta
