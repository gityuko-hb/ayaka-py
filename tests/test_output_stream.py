"""Terminal-monotonicity contract of the SDK output stream (research task C01).

``OutputStream`` used to encode "closed" only as a ``None`` marker at the tail of
its deque.  Once the consumer popped the marker the stream forgot it had been
closed, so a late ``put`` re-queued an event and a repeated ``close`` queued a
second marker.  These tests pin the monotonic contract: closed is a permanent
bit under the stream lock, independent of what is still queued.

Everything here is CPU-only and drives the real ``OutputStream`` /
``OutputCollector`` / ``AyakaLLM`` classes; only the engine below the façade is
scripted.
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest
from ayaka.request.schema import Request, RequestId
from ayaka.request.stream import OutputEvent, TokenUsage
from ayaka.runtime.collection import OutputCollector, OutputStream
from ayaka.runtime.llm import AsyncAyakaLLM, AyakaLLM
from ayaka.sched.outcome import FinishReason
from hypothesis import given, settings
from hypothesis import strategies as st


def _event(index: int, *, finish: FinishReason | None = None, request_id: str = "r1"):
    return OutputEvent(
        request_id=request_id,
        sequence_epoch=1,
        event_index=index,
        delta=f"t{index}",
        usage=TokenUsage(prompt_tokens=3, completion_tokens=index + 1),
        finish_reason=finish,
    )


def _next(stream: OutputStream, *, timeout: float = 2.0) -> OutputEvent:
    """Consume one event; a hang fails fast instead of eating the test timeout."""

    async def go() -> OutputEvent:
        return await asyncio.wait_for(stream.next_event(), timeout)

    return asyncio.run(go())


def _drain(stream: OutputStream) -> list[OutputEvent]:
    out: list[OutputEvent] = []
    while True:
        try:
            out.append(_next(stream))
        except StopAsyncIteration:
            return out


# ---------------------------------------------------------------------------
# The two counterexamples found at ayaka-py@b3ab895 (CPU contract tests).
# ---------------------------------------------------------------------------


def test_late_put_after_consumed_terminal_is_dropped() -> None:
    stream = OutputStream("r1", limit=4)
    stream.put(_event(0))
    stream.close()
    assert _next(stream).event_index == 0
    with pytest.raises(StopAsyncIteration):
        _next(stream)

    assert stream.put(_event(1)) is True  # dropped as late, not reported as overflow
    assert stream.pending() == 0
    assert stream.overflowed is False
    with pytest.raises(StopAsyncIteration):
        _next(stream)


def test_repeated_close_after_consumed_terminal_adds_no_marker() -> None:
    stream = OutputStream("r1", limit=4)
    stream.close()
    with pytest.raises(StopAsyncIteration):
        _next(stream)

    stream.close()
    assert stream.pending() == 0


# ---------------------------------------------------------------------------
# Surrounding contract: what must keep working.
# ---------------------------------------------------------------------------


def test_repeated_close_before_consume_queues_a_single_marker() -> None:
    stream = OutputStream("r1", limit=4)
    stream.put(_event(0))
    stream.close()
    stream.close()
    assert stream.pending() == 2  # one event + one marker
    assert [e.event_index for e in _drain(stream)] == [0]
    assert stream.pending() == 0


def test_next_event_stays_terminal_instead_of_parking_a_waiter() -> None:
    stream = OutputStream("r1")
    stream.close()
    for _ in range(3):  # every later call terminates immediately, none hangs
        with pytest.raises(StopAsyncIteration):
            _next(stream, timeout=0.5)
    assert stream._waiter is None


def test_closed_is_observable_and_survives_marker_consumption() -> None:
    stream = OutputStream("r1")
    assert stream.closed is False
    stream.close()
    assert stream.closed is True
    with pytest.raises(StopAsyncIteration):
        _next(stream)
    assert stream.closed is True


def test_terminal_slot_is_reserved_when_the_queue_is_full() -> None:
    stream = OutputStream("r1", limit=2)
    assert stream.put(_event(0)) is True
    assert stream.put(_event(1)) is True
    assert stream.put(_event(2)) is False  # nonterminal overflow
    assert stream.overflowed is True

    stream.close()  # never refused, even though the data queue is full
    assert stream.put(_event(3)) is True  # late: dropped, not an overflow report
    assert [e.event_index for e in _drain(stream)] == [0, 1]
    assert stream.pending() == 0


def test_events_queued_before_close_are_delivered_in_order_then_terminate() -> None:
    stream = OutputStream("r1", limit=8)
    for index in range(5):
        stream.put(_event(index))
    stream.close()
    assert [e.event_index for e in _drain(stream)] == [0, 1, 2, 3, 4]


def test_close_wakes_a_parked_consumer_from_another_thread() -> None:
    stream = OutputStream("r1")

    async def go() -> None:
        consumer = asyncio.ensure_future(stream.next_event())
        await asyncio.sleep(0.05)  # let the consumer park on its waiter
        assert not consumer.done()
        threading.Thread(target=stream.close).start()
        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(consumer, 2.0)
        # The wake raced a close that already happened: still terminal afterwards.
        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(stream.next_event(), 0.5)

    asyncio.run(go())


def test_put_wakes_a_parked_consumer_from_another_thread() -> None:
    stream = OutputStream("r1")

    async def go() -> OutputEvent:
        consumer = asyncio.ensure_future(stream.next_event())
        await asyncio.sleep(0.05)
        threading.Thread(target=stream.put, args=(_event(0),)).start()
        return await asyncio.wait_for(consumer, 2.0)

    assert asyncio.run(go()).event_index == 0


def test_limit_must_be_positive() -> None:
    with pytest.raises(ValueError):
        OutputStream("r1", limit=0)


# ---------------------------------------------------------------------------
# Collector level: the "settle closes again after the consumer finished" path.
# ---------------------------------------------------------------------------


def test_settle_style_second_close_after_terminal_event_adds_nothing() -> None:
    """``AyakaLLM._settle`` closes every terminal request's stream again.

    The collector already closed it when the terminal event arrived; if the
    consumer popped that marker in between, the second close used to queue a new
    one (pending 0 -> 1).
    """
    collector = OutputCollector(limit=4)
    stream = collector.attach("r1")
    collector.on_output_event(_event(0))
    collector.on_output_event(_event(1, finish=FinishReason.LENGTH))
    assert [e.event_index for e in _drain(stream)] == [0, 1]
    assert stream.pending() == 0

    stream.close()  # what the owner's settlement scan does for a terminal request
    assert stream.pending() == 0
    assert stream.closed is True


def test_collector_drops_events_that_arrive_after_the_terminal() -> None:
    collector = OutputCollector(limit=4)
    stream = collector.attach("r1")
    collector.on_output_event(_event(0, finish=FinishReason.EOS))
    assert [e.event_index for e in _drain(stream)] == [0]

    collector.on_output_event(_event(1))  # stale duplicate after terminal
    assert stream.pending() == 0
    assert collector.overflowed_requests() == ()


def test_parallel_family_children_do_not_close_the_parent_stream() -> None:
    collector = OutputCollector(limit=8)
    stream = collector.attach_parallel("p", ("p-0", "p-1"))
    for child, name in enumerate(("p-0", "p-1")):
        collector.on_output_event(
            OutputEvent(
                request_id=name,
                sequence_epoch=1,
                event_index=0,
                delta="x",
                usage=TokenUsage(1, 1),
                finish_reason=FinishReason.LENGTH,
                child_index=child,
            )
        )
    assert stream.closed is False  # the façade closes it once every child is terminal
    assert stream.pending() == 2
    stream.close()
    assert [e.child_index for e in _drain(stream)] == [0, 1]


# ---------------------------------------------------------------------------
# Model-based property: any interleaving of put/close/consume matches the spec.
# ---------------------------------------------------------------------------

_TERMINAL = object()


@settings(max_examples=200, deadline=None)
@given(
    limit=st.integers(min_value=1, max_value=4),
    ops=st.lists(st.sampled_from(["put", "close", "pop"]), max_size=40),
)
def test_stream_matches_the_monotonic_model(limit: int, ops: list[str]) -> None:
    stream = OutputStream("r1", limit=limit)
    queue: list[object] = []  # events plus, once closed, the terminal sentinel
    closed = False
    saw_terminal = False
    next_index = 0
    delivered: list[int] = []

    for op in ops:
        if op == "put":
            event = _event(next_index)
            next_index += 1
            accepted = stream.put(event)
            if closed:
                assert accepted is True  # late events are dropped, never queued
            elif len(queue) >= limit:
                assert accepted is False
            else:
                assert accepted is True
                queue.append(event)
        elif op == "close":
            stream.close()
            if not closed:
                closed = True
                queue.append(_TERMINAL)
        elif queue or closed:  # pop only when the model says it cannot park
            try:
                got = _next(stream, timeout=1.0)
            except StopAsyncIteration:
                assert closed
                if queue:
                    assert queue.pop(0) is _TERMINAL
                    assert not queue  # nothing may follow the marker
                saw_terminal = True
            else:
                assert not saw_terminal  # no event after the terminal was observed
                assert got is queue.pop(0)
                delivered.append(got.event_index)

        assert stream.closed is closed
        assert stream.pending() == len(queue)

    assert delivered == sorted(delivered)
    if saw_terminal:
        assert stream.pending() == 0
        with pytest.raises(StopAsyncIteration):  # permanent, not a one-shot marker
            _next(stream, timeout=1.0)


# ---------------------------------------------------------------------------
# Façade level: AsyncAyakaLLM over a scripted engine (no model, no device).
# ---------------------------------------------------------------------------


class _Lifecycle:
    def __init__(self) -> None:
        self.is_terminal = False
        self.finish_reason: FinishReason | None = None


class _ScriptedEngine:
    """Publishes ``tokens`` events per request through the real output listener."""

    def __init__(self, tokens: int, *, publish_terminal: bool = True) -> None:
        self.tokens = tokens
        self.publish_terminal = publish_terminal
        self.listener = None
        self.lifecycles: dict[str, _Lifecycle] = {}
        self.sent: dict[str, int] = {}
        self.aborted: list[str] = []
        self.forgotten: list[str] = []
        self.output = SimpleNamespace(set_listener=self._set_listener)
        self.requests = SimpleNamespace(
            find=self.lifecycles.get,
            cancel=lambda rid, reason: rid in self.lifecycles,
            forget=lambda rid: None,
        )

    def _set_listener(self, listener) -> None:
        self.listener = listener

    def submit(self, request: Request) -> None:
        rid = str(request.request_id)
        self.lifecycles[rid] = _Lifecycle()
        self.sent[rid] = 0

    def step(self) -> bool:
        did = False
        for rid, lifecycle in self.lifecycles.items():
            if lifecycle.is_terminal:
                continue
            index = self.sent[rid]
            last = index + 1 >= self.tokens
            finish = FinishReason.LENGTH if last and self.publish_terminal else None
            if self.listener is not None:
                self.listener.on_output_event(_event(index, finish=finish, request_id=rid))
            self.sent[rid] = index + 1
            if last:
                lifecycle.is_terminal = True
                lifecycle.finish_reason = FinishReason.LENGTH
            did = True
        return did

    def abort(self, request_id: str, **_: object) -> bool:
        self.aborted.append(request_id)
        lifecycle = self.lifecycles.get(request_id)
        if lifecycle is not None:
            lifecycle.is_terminal = True
        return True

    def forget(self, request_id: str) -> None:
        self.forgotten.append(request_id)

    def close(self):
        return SimpleNamespace(closed=True)


def _request(request_id: str) -> Request:
    return Request(request_id=RequestId(request_id), prompt_token_ids=(1, 2, 3))


def test_async_generate_yields_every_event_once_and_completes() -> None:
    engine = _ScriptedEngine(tokens=6)
    llm = AsyncAyakaLLM(engine, stream_limit=16, idle_interval=0.005)

    async def go() -> list[int]:
        out = [e.event_index async for e in llm.generate(_request("r1"))]
        return out

    try:
        assert asyncio.run(asyncio.wait_for(go(), 10.0)) == [0, 1, 2, 3, 4, 5]
    finally:
        assert llm.close(timeout=5.0) is True


def test_owner_settle_does_not_queue_a_second_marker_after_the_consumer_finished() -> None:
    """Drive ``AyakaLLM`` by hand: consume to the end, then let settlement run again."""
    engine = _ScriptedEngine(tokens=3)
    llm = AyakaLLM(engine, stream_limit=16, idle_interval=0.005)
    try:
        stream = llm.submit(_request("r1")).result(timeout=5.0)
        events = _drain(stream)
        assert [e.event_index for e in events] == [0, 1, 2]
        assert events[-1].finish_reason is FinishReason.LENGTH

        # Settlement dropped the stream from the collector; a repeated close from
        # any owner path (shutdown straggler sweep, overflow path) stays a no-op.
        stream.close()
        stream.close()
        assert stream.pending() == 0
        assert stream.put(_event(99)) is True
        assert stream.pending() == 0
        with pytest.raises(StopAsyncIteration):
            _next(stream, timeout=0.5)
    finally:
        assert llm.close(timeout=5.0) is True


def test_abnormal_end_without_terminal_event_still_terminates_the_consumer() -> None:
    """Aborted/failed requests never publish a terminal event; settlement closes."""
    engine = _ScriptedEngine(tokens=100, publish_terminal=False)
    llm = AyakaLLM(engine, stream_limit=1024, idle_interval=0.005)
    try:
        stream = llm.submit(_request("r1")).result(timeout=5.0)
        first = _next(stream)
        assert first.event_index == 0
        llm.cancel("r1")
        engine.abort("r1")  # the engine side of the cancel: lifecycle goes terminal
        tail = _drain(stream)
        assert all(e.finish_reason is None for e in tail)
        assert stream.closed is True
    finally:
        assert llm.close(timeout=5.0) is True
