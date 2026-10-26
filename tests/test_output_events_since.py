"""Cursor-based output retrieval and bounded history (research tasks O01 / R5).

Before this change the HTTP collector asked ``Engine.events(request_id)`` for the
whole retained history on every pass and sliced off the part it had already seen.
Copying a tuple of ``H`` references per pass makes one request that emits ``E``
events cost ``1 + 2 + ... + E`` reference copies when it is polled once per
event, and the history itself was retained without bound.

These tests pin the replacement contract:

* ``events_since`` touches only the delta, with an absolute monotonic cursor;
* the legacy ``events()`` stays a compatibility view of the retained window;
* ``history_limit`` bounds retained events and an expired cursor is reported
  explicitly instead of silently skipping tokens;
* ``ServingService._collect`` consumes through the delta path and fails only the
  request whose consumer fell behind the history.

CPU-only: the real ``OutputProcessor`` and the real ``_collect`` body run; the
engine below them is a minimal fake.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest
from ayaka.configs.serving import ServingConfig
from ayaka.request.schema import Request, RequestId, StopCriteria
from ayaka.request.states import RequestState
from ayaka.request.stream import OutputEvent, OutputHistoryExpired
from ayaka.runtime.engine import Engine
from ayaka.runtime.output import OutputProcessor
from ayaka.serving.events import ContentDelta, GenerationFailed
from ayaka.serving.service import GenerationHandle, ServingService
from ayaka.serving.stats import ServingStats

PROMPT = 3


def _request(request_id: str = "r1", *, max_tokens: int = 100_000) -> Request:
    return Request(
        request_id=RequestId(request_id),
        prompt_token_ids=(1, 2, 3),
        stop=StopCriteria(max_tokens=max_tokens),
    )


def _publish(output: OutputProcessor, request_id: str, count: int, *, start: int = 0) -> None:
    """Publish ``count`` samples; ``start`` is how many were published before."""
    for offset in range(count):
        generated = start + offset + 1
        output.on_published(
            request_id,
            1000 + generated,
            sequence_epoch=1,
            generated_tokens=generated,
            prompt_tokens=PROMPT,
        )


class _Decoder:
    """Detokenizer stand-in: every token decodes to ``<id>``."""

    def update(self, ids, *, check_stops: bool):
        return SimpleNamespace(delta="".join(f"<{i}>" for i in ids), stop_matched=None)

    def finish(self):
        return SimpleNamespace(delta="")

    def close(self) -> None:
        pass


class _Tokenizer:
    def new_detokenizer(self, params, *, prompt_token_ids):
        return _Decoder()


def _processor(history_limit: int | None = None, *, text: bool = False) -> OutputProcessor:
    tokenizer = _Tokenizer() if text else None
    return OutputProcessor(tokenizer, history_limit=history_limit)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# OutputProcessor.events_since
# ---------------------------------------------------------------------------


def test_events_since_equals_the_matching_slice_of_the_full_history() -> None:
    output = _processor()
    output.register(_request())
    _publish(output, "r1", 25)
    full = output.events("r1")
    assert [e.event_index for e in full] == list(range(25))
    for cursor in range(26):
        assert output.events_since("r1", cursor) == full[cursor:]


def test_cursor_defaults_to_the_start() -> None:
    output = _processor()
    output.register(_request())
    _publish(output, "r1", 3)
    assert output.events_since("r1") == output.events("r1")


def test_limit_pages_the_history_without_gaps_or_duplicates() -> None:
    output = _processor()
    output.register(_request())
    _publish(output, "r1", 23)
    cursor, pages = 0, []
    while True:
        page = output.events_since("r1", cursor, 5)
        if not page:
            break
        assert len(page) <= 5
        pages.extend(page)
        cursor = page[-1].event_index + 1
    assert pages == list(output.events("r1"))
    assert output.events_since("r1", 3, 0) == ()


def test_cursor_beyond_what_was_published_is_a_caller_error() -> None:
    output = _processor()
    output.register(_request())
    _publish(output, "r1", 4)
    assert output.events_since("r1", 4) == ()  # caught up
    with pytest.raises(ValueError, match="ahead"):
        output.events_since("r1", 5)


@pytest.mark.parametrize("cursor", [-1, True, 1.5, "0"])
def test_cursor_must_be_a_nonnegative_integer(cursor: object) -> None:
    output = _processor()
    output.register(_request())
    with pytest.raises((TypeError, ValueError)):
        output.events_since("r1", cursor)  # type: ignore[arg-type]


@pytest.mark.parametrize("limit", [-1, True, 1.5])
def test_limit_must_be_a_nonnegative_integer_or_none(limit: object) -> None:
    output = _processor()
    output.register(_request())
    with pytest.raises((TypeError, ValueError)):
        output.events_since("r1", 0, limit)  # type: ignore[arg-type]


def test_unknown_or_forgotten_request_raises_keyerror_like_events() -> None:
    output = _processor()
    with pytest.raises(KeyError):
        output.events_since("nope")
    output.register(_request())
    _publish(output, "r1", 2)
    output.forget("r1")
    with pytest.raises(KeyError):
        output.events_since("r1")


class _CountingList(list):
    """Counts the references the processor reads out of the retained history."""

    visited = 0

    def __getitem__(self, index):
        out = super().__getitem__(index)
        if isinstance(index, slice):
            type(self).visited += len(out)
        return out

    def __iter__(self):
        for item in super().__iter__():
            type(self).visited += 1
            yield item


def _instrumented() -> OutputProcessor:
    output = _processor()
    output.register(_request())
    state = output._states["r1"]
    state.events = _CountingList(state.events)
    return output


def test_polling_after_every_event_copies_each_event_exactly_once() -> None:
    """The O(E^2) -> O(E) claim as an operation count, not a stopwatch.

    ``_CountingList`` sits under the processor's history, so this measures the
    references actually read, not just the length of what was returned.
    """
    events = 400

    output = _instrumented()
    cursor = 0
    _CountingList.visited = 0
    for index in range(events):
        _publish(output, "r1", 1, start=index)
        delta = output.events_since("r1", cursor)
        assert len(delta) == 1
        cursor += len(delta)
    assert _CountingList.visited == events  # linear: each event read exactly once

    legacy = _instrumented()  # the old collector: full copy on every pass
    _CountingList.visited = 0
    for index in range(events):
        _publish(legacy, "r1", 1, start=index)
        legacy.events("r1")
    assert _CountingList.visited == events * (events + 1) // 2  # quadratic


def test_idle_polls_copy_nothing() -> None:
    output = _processor()
    output.register(_request())
    _publish(output, "r1", 50)
    assert all(output.events_since("r1", 50) == () for _ in range(100))


# ---------------------------------------------------------------------------
# Bounded history
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0, -3])
def test_history_limit_must_be_positive(bad: int) -> None:
    with pytest.raises(ValueError):
        _processor(bad)


@pytest.mark.parametrize("bad", [True, 2.5, "8"])
def test_history_limit_must_be_an_integer(bad: object) -> None:
    with pytest.raises(TypeError):
        _processor(bad)  # type: ignore[arg-type]


def test_history_is_trimmed_to_a_bounded_window_with_absolute_indices() -> None:
    limit = 8
    output = _processor(limit)
    output.register(_request())
    for produced in range(1, 301):
        _publish(output, "r1", 1, start=produced - 1)
        window = output.events("r1")
        assert limit <= len(window) < 2 * limit or produced < 2 * limit
        # The window is always the newest contiguous run, ending at the head.
        assert [e.event_index for e in window] == list(range(produced - len(window), produced))
    assert window[-1].event_index == 299  # indices never restart after a trim


def test_the_newest_history_limit_events_always_stay_retrievable() -> None:
    limit = 8
    output = _processor(limit)
    output.register(_request())
    for produced in range(1, 120):
        _publish(output, "r1", 1, start=produced - 1)
        tail = output.events_since("r1", max(0, produced - limit))
        assert [e.event_index for e in tail] == list(range(max(0, produced - limit), produced))


def test_a_cursor_behind_the_window_is_reported_not_silently_skipped() -> None:
    output = _processor(4)
    output.register(_request())
    _publish(output, "r1", 40)
    with pytest.raises(OutputHistoryExpired) as caught:
        output.events_since("r1", 0)
    err = caught.value
    assert err.request_id == "r1"
    assert err.cursor == 0
    assert 0 < err.base <= 36
    assert isinstance(err, LookupError)
    # Resynchronising at the reported base yields a gap-free suffix.
    resumed = output.events_since("r1", err.base)
    assert [e.event_index for e in resumed] == list(range(err.base, 40))


def test_unbounded_history_never_expires_and_matches_the_old_events_contract() -> None:
    output = _processor()  # history_limit=None is the default
    output.register(_request())
    _publish(output, "r1", 5_000)
    assert len(output.events("r1")) == 5_000
    assert output.events_since("r1", 0)[0].event_index == 0


def test_trimming_never_touches_accumulated_text_or_finish_state() -> None:
    output = _processor(4, text=True)
    output.register(_request(max_tokens=50))
    _publish(output, "r1", 50)
    expected = "".join(f"<{1000 + n}>" for n in range(1, 51))
    assert output.text("r1") == expected
    last = output.events("r1")[-1]
    assert last.event_index == 49
    assert last.finish_reason is not None  # LENGTH at the 50th token survives the trim


def test_registering_another_request_does_not_share_the_window() -> None:
    output = _processor(4)
    output.register(_request("a"))
    output.register(_request("b"))
    _publish(output, "a", 40)
    _publish(output, "b", 3)
    assert output.events_since("b", 0)[0].event_index == 0
    with pytest.raises(OutputHistoryExpired):
        output.events_since("a", 0)


# ---------------------------------------------------------------------------
# Engine passthrough
# ---------------------------------------------------------------------------


def test_engine_events_since_delegates_to_the_output_owner() -> None:
    output = _processor()
    output.register(_request())
    _publish(output, "r1", 6)
    engine = object.__new__(Engine)
    engine._output = output  # type: ignore[attr-defined]
    assert engine.events_since("r1", 2, 3) == output.events_since("r1", 2, 3)
    assert engine.events_since("r1") == engine.events("r1")


# ---------------------------------------------------------------------------
# ServingService._collect (real body, no owner thread, fake engine)
# ---------------------------------------------------------------------------


class _FakeEngine:
    def __init__(self, output: OutputProcessor) -> None:
        self.output = output
        self.lifecycles: dict[str, SimpleNamespace] = {}
        self.aborted: list[str] = []
        self.calls: list[tuple[str, int, int]] = []
        self.requests = SimpleNamespace(get=self.lifecycles.__getitem__, forget=lambda rid: None)
        self.executor = SimpleNamespace(runner=SimpleNamespace(forget_request=lambda rid: None))

    def add(self, request: Request) -> SimpleNamespace:
        rid = str(request.request_id)
        self.output.register(request)
        lifecycle = SimpleNamespace(
            is_terminal=False,
            finish_reason=None,
            state=RequestState.DECODING,
            request=SimpleNamespace(prompt_len=PROMPT, priority=0, sla_class="default"),
            output_token_ids=[],
            machine=SimpleNamespace(
                num_cached_tokens=0,
                timing=SimpleNamespace(
                    first_scheduled_ns=0,
                    first_token_ns=0,
                    inter_token_percentiles_ns=lambda: (0, 0),
                    inter_token_intervals_ns=(),
                ),
            ),
            token=SimpleNamespace(reason=None),
        )
        self.lifecycles[rid] = lifecycle
        return lifecycle

    def events_since(self, request_id: str, cursor: int = 0, limit: int | None = None):
        out = self.output.events_since(request_id, cursor, limit)
        self.calls.append((request_id, cursor, len(out)))
        return out

    def events(self, request_id: str):  # the legacy full-history copy
        raise AssertionError("_collect must not copy the whole retained history")

    def abort(self, request_id: str, **_: object) -> bool:
        self.aborted.append(request_id)
        lifecycle = self.lifecycles[request_id]
        lifecycle.is_terminal = True
        lifecycle.state = RequestState.CANCELLED
        return True

    def forget(self, request_id: str) -> None:
        self.output.forget(request_id)


class _Harness(ServingService):
    """Runs the real ``_collect`` without the owner thread or admission machinery."""

    def __init__(self, engine: _FakeEngine, config: ServingConfig) -> None:  # noqa: D107
        self.engine, self.config = engine, config
        self._clock = time.monotonic_ns
        self.stats = ServingStats(clock=self._clock)
        self._handles: dict[str, GenerationHandle] = {}

    def open(self, request: Request) -> GenerationHandle:
        spec = SimpleNamespace(messages=None, tools=(), tool_choice="none")
        handle = GenerationHandle(request, spec, self.config)  # type: ignore[arg-type]
        self._handles[str(request.request_id)] = handle
        return handle


def _drain_handle(handle: GenerationHandle, count: int) -> list[object]:
    async def go() -> list[object]:
        return [await asyncio.wait_for(handle.next_event(), 2.0) for _ in range(count)]

    return asyncio.run(go())


def _service(history_limit: int | None = None, **config: int) -> tuple[_Harness, _FakeEngine]:
    output = _processor(history_limit, text=True)
    engine = _FakeEngine(output)
    return _Harness(engine, ServingConfig(**{"max_stream_events": 4096, **config})), engine


def test_collect_reads_only_new_events_and_never_the_full_history() -> None:
    service, engine = _service()
    request = _request()
    engine.add(request)
    handle = service.open(request)
    for index in range(30):
        _publish(engine.output, "r1", 1, start=index)
        service._collect()
        service._collect()  # an idle second pass must not re-read anything
    assert sum(n for _, _, n in engine.calls) == 30
    assert max(n for _, _, n in engine.calls) == 1
    assert handle.cursor == 30
    texts = [e.text for e in _drain_handle(handle, 30)]  # type: ignore[attr-defined]
    assert texts == [f"<{1000 + n}>" for n in range(1, 31)]


def test_collect_handles_a_multi_event_burst_in_one_call() -> None:
    service, engine = _service()
    request = _request()
    engine.add(request)
    handle = service.open(request)
    _publish(engine.output, "r1", 7)
    service._collect()
    assert engine.calls == [("r1", 0, 7)]
    assert handle.cursor == 7
    assert all(isinstance(e, ContentDelta) for e in _drain_handle(handle, 7))


def test_a_consumer_that_polls_every_step_survives_a_tight_history_quota() -> None:
    limit = 8
    service, engine = _service(limit)
    request = _request()
    engine.add(request)
    handle = service.open(request)
    produced = 0
    for _ in range(100):  # bursts the size of a speculative step, one pass per step
        _publish(engine.output, "r1", 5, start=produced)
        produced += 5
        service._collect()
        assert len(engine.output.events("r1")) < 2 * limit  # memory stays bounded
    assert handle.error is None
    assert engine.aborted == []
    assert handle.cursor == produced
    assert len(_drain_handle(handle, 500)) == 500


def test_history_expiry_fails_only_the_lagging_request() -> None:
    service, engine = _service(history_limit=4)
    slow, fast = _request("slow"), _request("fast")
    slow_lifecycle = engine.add(slow)
    engine.add(fast)
    slow_handle = service.open(slow)
    fast_handle = service.open(fast)

    _publish(engine.output, "slow", 40)  # the consumer did not run meanwhile
    _publish(engine.output, "fast", 3)
    service._collect()

    assert engine.aborted == ["slow"]
    assert slow_lifecycle.is_terminal
    assert "slow" not in service._handles  # settled in the same pass
    assert "fast" in service._handles
    assert fast_handle.error is None
    assert [e.text for e in _drain_handle(fast_handle, 3)] == [  # type: ignore[attr-defined]
        "<1001>",
        "<1002>",
        "<1003>",
    ]
    assert slow_handle.error is not None
    assert "history" in slow_handle.error
    terminal = _drain_handle(slow_handle, 1)[0]
    assert isinstance(terminal, GenerationFailed)
    assert service.stats.stream_overflows.labels("history")._value.get() == 1  # type: ignore[attr-defined]


def test_expiry_is_handled_once_not_on_every_later_pass() -> None:
    service, engine = _service(history_limit=4)
    request = _request()
    lifecycle = engine.add(request)
    handle = service.open(request)
    lifecycle.state = RequestState.DECODING
    # Keep the lifecycle non-terminal after the abort so later passes happen.
    engine.abort = lambda rid, **_: engine.aborted.append(rid) or True  # type: ignore[method-assign]

    _publish(engine.output, "r1", 40)
    service._collect()
    # Resumed at the retained base and consumed the surviving window, so the
    # cursor is back in step with production instead of failing on every pass.
    assert handle.cursor == 40
    calls_after_expiry = len(engine.calls)
    service._collect()
    _publish(engine.output, "r1", 3, start=40)
    service._collect()
    assert handle.cursor == 43
    assert len(engine.calls) == calls_after_expiry + 2
    assert engine.aborted == ["r1"]
    assert handle.error is not None
    assert service.stats.stream_overflows.labels("history")._value.get() == 1  # type: ignore[attr-defined]


def test_slow_consumer_fails_only_its_request_and_still_gets_one_terminal() -> None:
    """Case E04: a full frontend queue aborts that request, never its neighbours."""
    service, engine = _service(max_stream_events=2)
    slow, fast = _request("slow"), _request("fast")
    engine.add(slow)
    engine.add(fast)
    slow_handle = service.open(slow)
    fast_handle = service.open(fast)

    _publish(engine.output, "slow", 6)  # the socket did not drain while this burst arrived
    _publish(engine.output, "fast", 2)
    service._collect()

    assert engine.aborted == ["slow"]
    assert slow_handle.error == "output consumer is too slow"
    assert service.stats.stream_overflows.labels("events")._value.get() == 1  # type: ignore[attr-defined]
    delivered = _drain_handle(slow_handle, 3)  # two accepted deltas, then the reserved terminal
    assert [type(e) for e in delivered] == [ContentDelta, ContentDelta, GenerationFailed]
    assert "slow" not in service._handles

    assert fast_handle.error is None
    assert "fast" in service._handles
    assert [e.text for e in _drain_handle(fast_handle, 2)] == ["<1001>", "<1002>"]  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# ServingConfig knob
# ---------------------------------------------------------------------------


def test_config_defaults_to_unbounded_history() -> None:
    assert ServingConfig().max_history_events == 0


def test_config_accepts_unbounded_or_a_quota_covering_one_speculative_step() -> None:
    from ayaka.speculative.config import MAX_DRAFT_TOKENS

    floor = MAX_DRAFT_TOKENS + 1
    assert ServingConfig(max_history_events=floor).max_history_events == floor
    assert ServingConfig(max_history_events=1 << 20).max_history_events == 1 << 20
    for bad in (-1, 1, floor - 1):
        with pytest.raises(ValueError, match="max_history_events"):
            ServingConfig(max_history_events=bad)


def test_event_type_is_unchanged_by_the_cursor_api() -> None:
    output = _processor()
    output.register(_request())
    _publish(output, "r1", 1)
    (event,) = output.events_since("r1")
    assert isinstance(event, OutputEvent)
    assert (event.request_id, event.sequence_epoch, event.event_index) == ("r1", 1, 0)
