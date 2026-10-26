"""Bounded HTTP delivery contract of ``GenerationHandle`` (research case E04).

The handle is the HTTP-side counterpart of the SDK ``OutputStream``.  It must
bound both the event count and the UTF-8 bytes a slow socket can pin, name which
budget was hit, keep one terminal slot that is deliverable even when the queue is
full, and deliver that terminal exactly once.
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest
from ayaka.configs.serving import ServingConfig
from ayaka.request.schema import Request, RequestId
from ayaka.serving.events import (
    ContentDelta,
    GenerationFailed,
    GenerationFinished,
    TokenLogprobs,
    UsageUpdate,
)
from ayaka.serving.service import LOGPROB_ENTRY_BYTES, GenerationHandle


def _handle(max_events: int = 3, max_bytes: int = 16) -> GenerationHandle:
    config = ServingConfig(max_stream_events=max_events, max_stream_bytes=max_bytes)
    request = Request(request_id=RequestId("r1"), prompt_token_ids=(1, 2, 3))
    spec = SimpleNamespace(messages=None, tools=(), tool_choice="none")
    return GenerationHandle(request, spec, config)  # type: ignore[arg-type]


def _next(handle: GenerationHandle, timeout: float = 2.0):
    async def go():
        return await asyncio.wait_for(handle.next_event(), timeout)

    return asyncio.run(go())


def _terminal() -> GenerationFinished:
    return GenerationFinished("stop", UsageUpdate(prompt_tokens=3, completion_tokens=2))


def test_event_budget_refuses_the_overflowing_event_and_names_itself() -> None:
    handle = _handle(max_events=3, max_bytes=1 << 20)
    assert [handle.put(ContentDelta("a")) for _ in range(3)] == [True, True, True]
    assert handle.overflow is None
    assert handle.put(ContentDelta("a")) is False
    assert handle.overflow == "events"


def test_byte_budget_is_independent_of_the_event_budget() -> None:
    handle = _handle(max_events=100, max_bytes=16)
    assert handle.put(ContentDelta("x" * 10)) is True
    assert handle.put(ContentDelta("y" * 7)) is False  # 17 > 16 bytes
    assert handle.overflow == "bytes"


def test_one_event_larger_than_the_byte_budget_is_refused_even_when_empty() -> None:
    handle = _handle(max_events=100, max_bytes=8)
    assert handle.put(ContentDelta("z" * 9)) is False
    assert handle.overflow == "bytes"


def test_bytes_are_counted_as_utf8_not_characters() -> None:
    handle = _handle(max_events=100, max_bytes=8)
    assert handle.put(ContentDelta("é" * 4)) is True  # 4 chars, 8 bytes
    assert handle.put(ContentDelta("a")) is False
    assert handle.overflow == "bytes"


def test_the_terminal_slot_is_reserved_when_the_queue_is_full() -> None:
    handle = _handle(max_events=2, max_bytes=1 << 20)
    handle.put(ContentDelta("a"))
    handle.put(ContentDelta("b"))
    assert handle.put(ContentDelta("c")) is False  # full

    terminal = GenerationFailed("generation_failed", "output consumer is too slow")
    assert handle.put(terminal, terminal=True) is True  # still deliverable
    delivered = [_next(handle) for _ in range(3)]
    assert [type(e) for e in delivered] == [ContentDelta, ContentDelta, GenerationFailed]


def test_the_terminal_is_delivered_exactly_once() -> None:
    handle = _handle()
    assert handle.put(_terminal(), terminal=True) is True
    assert handle.put(_terminal(), terminal=True) is False  # a repeated settlement is a no-op
    assert isinstance(_next(handle), GenerationFinished)
    with pytest.raises(TimeoutError):
        _next(handle, timeout=0.2)  # no duplicate terminal is waiting


def test_terminal_does_not_consume_the_data_budgets() -> None:
    handle = _handle(max_events=1, max_bytes=4)
    assert handle.put(ContentDelta("abcd")) is True
    assert handle.put(_terminal(), terminal=True) is True
    assert isinstance(_next(handle), ContentDelta)
    assert isinstance(_next(handle), GenerationFinished)


def test_draining_releases_budget_so_a_keeping_up_consumer_never_overflows() -> None:
    handle = _handle(max_events=2, max_bytes=8)
    for _ in range(50):
        assert handle.put(ContentDelta("abcd")) is True
        assert _next(handle) == ContentDelta("abcd")
    assert handle.overflow is None


def test_events_are_delivered_in_order_then_the_terminal() -> None:
    handle = _handle(max_events=8)
    for text in ("a", "b", "c"):
        handle.put(ContentDelta(text))
    handle.put(_terminal(), terminal=True)
    texts = [_next(handle) for _ in range(3)]
    assert [e.text for e in texts] == ["a", "b", "c"]  # type: ignore[attr-defined]
    assert isinstance(_next(handle), GenerationFinished)


def test_put_from_the_owner_thread_wakes_a_parked_consumer() -> None:
    handle = _handle()

    async def go():
        consumer = asyncio.ensure_future(handle.next_event())
        await asyncio.sleep(0.05)
        threading.Thread(target=handle.put, args=(ContentDelta("hi"),)).start()
        return await asyncio.wait_for(consumer, 2.0)

    assert asyncio.run(go()) == ContentDelta("hi")


def test_a_vanished_consumer_loop_cancels_the_request_instead_of_raising() -> None:
    handle = _handle()
    loop = asyncio.new_event_loop()
    parked = loop.create_future()
    handle._waiter = parked
    loop.close()  # the event loop is gone; waking it raises RuntimeError internally
    assert handle.put(ContentDelta("x")) is True
    assert handle.cancelled.is_set()


# ---------------------------------------------------------------------------
# Logprob reports have their own byte ceiling (research A-03)
# ---------------------------------------------------------------------------


def _logprob_handle(max_logprob_bytes: int, max_bytes: int = 8) -> GenerationHandle:
    config = ServingConfig(
        max_stream_events=100,
        max_stream_bytes=max_bytes,
        max_stream_logprob_bytes=max_logprob_bytes,
    )
    request = Request(request_id=RequestId("r1"), prompt_token_ids=(1, 2, 3))
    spec = SimpleNamespace(messages=None, tools=(), tool_choice="none")
    return GenerationHandle(request, spec, config)  # type: ignore[arg-type]


def _report(alternatives: int) -> TokenLogprobs:
    return TokenLogprobs(7, -0.5, tuple((i, -1.0 - i) for i in range(alternatives)))


def test_logprob_reports_use_their_own_ceiling_not_the_text_one() -> None:
    handle = _logprob_handle(max_logprob_bytes=3 * LOGPROB_ENTRY_BYTES)
    assert handle.put(ContentDelta("x" * 8)) is True  # the text ceiling is full
    assert handle.put(_report(2)) is True  # three entries still fit the logprob one
    assert handle.put(_report(0)) is False
    assert handle.overflow == "logprob_bytes"


def test_draining_a_report_releases_its_logprob_bytes() -> None:
    handle = _logprob_handle(max_logprob_bytes=2 * LOGPROB_ENTRY_BYTES)
    for _ in range(20):
        assert handle.put(_report(1)) is True
        assert _next(handle) == _report(1)
    assert handle.overflow is None


def test_a_full_logprob_ceiling_still_delivers_the_terminal() -> None:
    handle = _logprob_handle(max_logprob_bytes=LOGPROB_ENTRY_BYTES)
    assert handle.put(_report(0)) is True
    assert handle.put(_report(0)) is False
    assert handle.put(_terminal(), terminal=True) is True
    assert isinstance(_next(handle), TokenLogprobs)
    assert isinstance(_next(handle), GenerationFinished)
