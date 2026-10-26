"""Deferred prefix-cache flush with all-owner quiescence (research C04, flush_wrapper).

SGLang's deferred flush keeps one pending request with a deadline, rejects a
concurrent flush, never fences admission and does not validate NaN/infinite
timeouts. Ayaka's flush is a typed command on the serving owner thread:

* it runs only when no owner can touch KV (generations, scheduler, tickets,
  pages, transactions, leases, tier and prefix transfers), never on an empty
  queue alone;
* ``timeout == 0`` answers BUSY at once; a positive timeout parks one operation,
  quiescence seen in the poll that reaches the deadline still wins, and a busy
  engine at the deadline answers TIMEOUT with the blockers;
* duplicates coalesce onto the parked operation; cancelled callers are dropped
  and an abandoned operation flushes nothing;
* ``drain=True`` holds new admissions while waiting;
* a clear that raises answers FAILED and stops the engine (fail-closed);
  shutdown answers parked callers instead of stranding them.

State-machine tests drive the real ``ServingService`` methods on a bare instance
with fake owners and a manual clock; the rest run the CPU serving stack.
"""

from __future__ import annotations

import asyncio
import itertools
import math
import threading
import time
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
from ayaka.serving.errors import EngineUnavailableError
from ayaka.serving.flush import (
    MAX_FLUSH_TIMEOUT_SECONDS,
    FlushOperation,
    FlushResult,
    FlushStatus,
    validate_flush_timeout,
)
from ayaka.serving.service import ServingService, _FlushFailed
from ayaka.serving.stats import ServingStats
from cpu_e2e import CpuServing, prompt_of

LONG = prompt_of(*range(40, 80))  # 40 tokens: two full shareable pages
SECOND = 1_000_000_000


# ---------------------------------------------------------------------------
# State machine on a bare service: fake owners, manual clock
# ---------------------------------------------------------------------------


class _Owners:
    """Engine + controller stand-in whose quiescence the test flips."""

    def __init__(self) -> None:
        self.busy: list[str] = []
        self.flushes = 0
        self.fail = False
        self.has_unfinished = False
        self.executor = SimpleNamespace(tickets=(), pending_completion=False, quarantined=())
        self.kv = SimpleNamespace(snapshot=lambda: SimpleNamespace(free_pages=10 + self.flushes))

    def flush_blockers(self) -> tuple[str, ...]:
        return tuple(self.busy)

    def flush_prefix_cache(self) -> int:
        if self.fail:
            raise RuntimeError("clear exploded")
        self.flushes += 1
        return 7


class _Clock:
    def __init__(self) -> None:
        self.now = 1_000 * SECOND

    def __call__(self) -> int:
        return self.now


def _bare_service() -> tuple[ServingService, _Owners, _Clock]:
    service = ServingService.__new__(ServingService)
    owners, clock = _Owners(), _Clock()
    service.engine = owners
    service._controller = owners
    service._handles = {}
    service._clock = clock
    service._flush = None
    service._flush_ids = itertools.count(1)
    service.stats = ServingStats()
    return service, owners, clock


def _count(service: ServingService, status: str) -> float:
    value = service.stats.registry.get_sample_value("ayaka_cache_flushes_total", {"status": status})
    return value or 0.0


def test_an_idle_engine_flushes_at_once() -> None:
    service, owners, _ = _bare_service()
    caller: Future = Future()
    service._start_flush(0.0, False, caller)
    result = caller.result(0)
    assert result.status is FlushStatus.FLUSHED and result.flushed
    assert (result.released_refs, result.free_pages_before, result.free_pages_after) == (7, 10, 11)
    assert owners.flushes == 1 and service._flush is None


def test_busy_with_zero_timeout_answers_busy_and_flushes_nothing() -> None:
    service, owners, _ = _bare_service()
    owners.busy = ["2 live sequences"]
    service._handles = {"r": object()}  # type: ignore[dict-item]
    caller: Future = Future()
    service._start_flush(0.0, False, caller)
    result = caller.result(0)
    assert result.status is FlushStatus.BUSY
    assert result.blockers == ("1 live generations", "2 live sequences")
    assert owners.flushes == 0 and service._flush is None


def test_a_parked_flush_runs_when_the_owners_go_quiet() -> None:
    service, owners, clock = _bare_service()
    owners.busy = ["1 in-flight pages"]
    caller: Future = Future()
    service._start_flush(5.0, False, caller)
    service._poll_flush()
    assert not caller.done() and service._flush is not None
    clock.now += 2 * SECOND
    owners.busy = []
    service._poll_flush()
    result = caller.result(0)
    assert result.status is FlushStatus.FLUSHED and result.waited_seconds == 2.0
    assert owners.flushes == 1 and service._flush is None


def test_busy_exactly_at_the_deadline_times_out_with_its_blockers() -> None:
    service, owners, clock = _bare_service()
    owners.busy = ["1 unretired tickets"]
    caller: Future = Future()
    service._start_flush(5.0, False, caller)
    clock.now += 5 * SECOND - 1
    service._poll_flush()
    assert not caller.done()
    clock.now += 1  # the deadline itself is expired
    service._poll_flush()
    result = caller.result(0)
    assert result.status is FlushStatus.TIMEOUT
    assert result.blockers == ("1 unretired tickets",) and owners.flushes == 0


def test_quiescence_seen_at_the_deadline_still_flushes() -> None:
    """The deadline bounds the wait; idle is checked first (SGLang order)."""
    service, owners, clock = _bare_service()
    owners.busy = ["1 live sequences"]
    caller: Future = Future()
    service._start_flush(5.0, False, caller)
    clock.now += 60 * SECOND  # the owner loop was slow to come back
    owners.busy = []
    service._poll_flush()
    assert caller.result(0).status is FlushStatus.FLUSHED


def test_duplicate_flushes_coalesce_onto_one_operation() -> None:
    service, owners, _ = _bare_service()
    owners.busy = ["1 live sequences"]
    first: Future = Future()
    second: Future = Future()
    service._start_flush(5.0, False, first)
    service._start_flush(0.0, True, second)  # joins; its own policy is not applied
    assert not second.done()  # a zero timeout did not turn into a BUSY answer
    owners.busy = []
    service._poll_flush()
    one, two = first.result(0), second.result(0)
    assert one == two and one.callers == 2 and one.operation == 1
    assert owners.flushes == 1 and _count(service, "flushed") == 1.0


def test_cancelled_callers_are_dropped_and_an_abandoned_flush_does_nothing() -> None:
    service, owners, _ = _bare_service()
    owners.busy = ["1 live sequences"]
    caller: Future = Future()
    service._start_flush(5.0, True, caller)
    assert caller.cancel()
    owners.busy = []
    service._poll_flush()
    assert owners.flushes == 0 and service._flush is None
    assert _count(service, "abandoned") == 1.0


def test_a_failing_clear_answers_failed_and_raises_to_stop_the_owner() -> None:
    service, owners, _ = _bare_service()
    owners.fail = True
    caller: Future = Future()
    with pytest.raises(_FlushFailed, match="clear exploded"):
        service._start_flush(0.0, False, caller)
    result = caller.result(0)
    assert result.status is FlushStatus.FAILED and "clear exploded" in result.detail


def test_an_operation_never_answers_a_cancelled_caller() -> None:
    gone, waiting = Future(), Future()
    operation = FlushOperation(1, 0, 0, False, [gone, waiting])
    gone.cancel()
    result = FlushResult(FlushStatus.BUSY, 1)
    assert operation.answer(result) == 1
    assert gone.cancelled() and waiting.result(0) is result
    assert operation.waiters == [waiting]


@pytest.mark.parametrize(
    ("timeout", "error"),
    [
        (math.nan, ValueError),
        (math.inf, ValueError),
        (-0.5, ValueError),
        (MAX_FLUSH_TIMEOUT_SECONDS + 1, ValueError),
        (True, TypeError),
        ("1", TypeError),
        (None, TypeError),
    ],
)
def test_nonfinite_negative_or_oversized_timeouts_are_refused(timeout: object, error) -> None:
    with pytest.raises(error):
        validate_flush_timeout(timeout)


@pytest.mark.parametrize("timeout", [0, 0.0, 2, 1.5, MAX_FLUSH_TIMEOUT_SECONDS])
def test_finite_nonnegative_timeouts_are_accepted(timeout: float) -> None:
    assert validate_flush_timeout(timeout) == float(timeout)


# ---------------------------------------------------------------------------
# The CPU serving stack
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def serving():
    with CpuServing() as stack:
        yield stack


def _cached(stack: CpuServing) -> int:
    return stack.memory().cache_owned_pages


def test_idle_flush_drops_the_prefix_cache_without_changing_outputs(serving: CpuServing) -> None:
    first = serving.generate(LONG, max_tokens=8)
    assert _cached(serving) > 0
    result = serving.service.flush_cache()
    assert result.status is FlushStatus.FLUSHED and result.released_refs > 0
    assert result.free_pages_after - result.free_pages_before == result.released_refs
    assert _cached(serving) == 0
    serving.assert_quiescent_accounting()
    again = serving.generate(LONG, max_tokens=8)
    assert again.text == first.text
    assert again.finish is not None and again.finish.usage.cached_tokens == 0  # a real miss


def test_busy_flush_names_the_owners_and_keeps_the_cache(serving: CpuServing) -> None:
    serving.generate(LONG, max_tokens=4)
    cached = _cached(serving)

    async def scenario():
        running = asyncio.ensure_future(serving.agenerate(LONG, max_tokens=120))
        await asyncio.sleep(0.05)
        busy = await asyncio.to_thread(serving.service.flush_cache)
        return busy, await running

    busy, generation = asyncio.run(asyncio.wait_for(scenario(), 60))
    assert busy.status is FlushStatus.BUSY
    assert "1 live generations" in busy.blockers
    assert any("live sequences" in b for b in busy.blockers)
    assert generation.finish_reason == "length" and _cached(serving) >= cached


def test_a_parked_flush_waits_for_the_running_generation(serving: CpuServing) -> None:
    expected = serving.generate(LONG, max_tokens=60).text

    async def scenario():
        running = asyncio.ensure_future(serving.agenerate(LONG, max_tokens=60))
        await asyncio.sleep(0.05)
        flushed = await asyncio.to_thread(serving.service.flush_cache, timeout=30)
        finished_first = running.done()
        return flushed, finished_first, await running

    flushed, finished_first, generation = asyncio.run(asyncio.wait_for(scenario(), 60))
    assert flushed.status is FlushStatus.FLUSHED and flushed.waited_seconds > 0
    assert finished_first  # the flush answered only after the generation settled
    assert generation.text == expected
    assert _cached(serving) == 0
    serving.assert_quiescent_accounting()


@pytest.mark.parametrize("drain", [False, True])
def test_drain_fences_new_admissions_until_the_flush(serving: CpuServing, drain: bool) -> None:
    """B shares A's prefix: it hits the cache unless the fence held it back."""
    serving.service.flush_cache()

    async def scenario():
        a = asyncio.ensure_future(serving.agenerate(LONG, max_tokens=60))
        await asyncio.sleep(0.05)
        flush = asyncio.ensure_future(
            asyncio.to_thread(serving.service.flush_cache, timeout=30, drain=drain)
        )
        await asyncio.sleep(0.05)  # the flush is parked before B arrives
        b = await serving.agenerate(LONG, max_tokens=4)
        return await flush, b, await a

    flushed, b, a = asyncio.run(asyncio.wait_for(scenario(), 60))
    assert flushed.status is FlushStatus.FLUSHED
    assert a.finish_reason == b.finish_reason == "length"
    assert b.finish is not None
    if drain:
        assert b.finish.usage.cached_tokens == 0  # admitted after the flush
    else:
        assert b.finish.usage.cached_tokens >= 32  # admitted beside A, then flushed
    serving.assert_quiescent_accounting()


def test_a_live_prefix_transfer_blocks_the_flush_with_empty_queues(serving: CpuServing) -> None:
    serving.generate(LONG, max_tokens=4)
    assert serving.runtime.kv is not None
    prefix = serving.runtime.kv.prefix_service

    class _LiveTransfer:
        retired = False

    transfer = _LiveTransfer()
    prefix._transfers.add(transfer)  # type: ignore[arg-type]
    try:
        busy = serving.service.flush_cache()
        assert busy.status is FlushStatus.BUSY and busy.blockers == ("1 prefix transfers",)
        assert _cached(serving) > 0
    finally:
        prefix._transfers.discard(transfer)  # type: ignore[arg-type]
    assert serving.service.flush_cache().status is FlushStatus.FLUSHED


def test_a_flush_withdraws_every_digest_this_node_advertised(serving: CpuServing) -> None:
    """Peers must stop routing to blocks that no longer exist here."""
    from ayaka.prefix.global_index import prompt_digest_pairs

    class _Publisher:
        def __init__(self) -> None:
            self.advertised: set[bytes] = set()
            self.withdrawn: set[bytes] = set()

        def publish(self, token_ids, *, context, page_size) -> None:
            pairs = prompt_digest_pairs(token_ids, page_size=page_size, context=context)
            self.advertised.update(digest for digest, _ in pairs)

        def withdraw_digest(self, digest: bytes) -> None:
            self.withdrawn.add(digest)

    serving.service.flush_cache()
    assert serving.runtime.kv is not None
    prefix = serving.runtime.kv.prefix_service
    publisher = _Publisher()
    prefix._index_publisher = publisher  # type: ignore[assignment]
    try:
        serving.generate(LONG, max_tokens=4)
        assert publisher.advertised and not publisher.withdrawn
        assert serving.service.flush_cache().status is FlushStatus.FLUSHED
    finally:
        prefix._index_publisher = None
    # Every advertised (complete) block is withdrawn; the partial tail block was
    # never advertised, and withdrawing it is a no-op for the index.
    assert publisher.advertised <= publisher.withdrawn


def test_an_idle_owner_wakes_for_the_flush_deadline(serving: CpuServing) -> None:
    """With no generation the owner sleeps 50 ms per poll; a parked flush caps it.

    Only a KV-side owner (here a live prefix transfer) can park a flush while no
    generation runs, so that is the case where the deadline must wake the loop.
    """
    assert serving.runtime.kv is not None
    prefix = serving.runtime.kv.prefix_service

    class _LiveTransfer:
        retired = False

    transfer = _LiveTransfer()
    prefix._transfers.add(transfer)  # type: ignore[arg-type]
    try:
        elapsed = []
        for _ in range(5):
            started = time.monotonic()
            result = serving.service.flush_cache(timeout=0.005)
            elapsed.append(time.monotonic() - started)
            assert result.status is FlushStatus.TIMEOUT
    finally:
        prefix._transfers.discard(transfer)  # type: ignore[arg-type]
    assert min(elapsed) < 0.04  # an uncapped idle sleep alone is 0.05 s


def test_a_busy_engine_at_the_deadline_answers_timeout(serving: CpuServing) -> None:
    async def scenario():
        running = asyncio.ensure_future(serving.agenerate(LONG, max_tokens=200))
        await asyncio.sleep(0.05)
        started = time.monotonic()
        result = await asyncio.to_thread(serving.service.flush_cache, timeout=0.2)
        return result, time.monotonic() - started, await running

    result, elapsed, generation = asyncio.run(asyncio.wait_for(scenario(), 60))
    assert result.status is FlushStatus.TIMEOUT and result.blockers
    assert 0.2 <= elapsed < 5.0  # answered at the deadline, not at idle
    assert generation.finish_reason == "length"  # the generation was not disturbed


def test_shutdown_answers_a_parked_flush_instead_of_stranding_it() -> None:
    stack = CpuServing()
    outcome: list[BaseException | FlushResult] = []

    def flush() -> None:
        try:
            outcome.append(stack.service.flush_cache(timeout=60))
        except BaseException as exc:  # noqa: BLE001 - recorded for the assertion
            outcome.append(exc)

    async def scenario():
        running = asyncio.ensure_future(stack.agenerate(LONG, max_tokens=200))
        await asyncio.sleep(0.05)
        caller = threading.Thread(target=flush)
        caller.start()
        await asyncio.sleep(0.1)  # parked behind the running generation
        await asyncio.to_thread(stack.close)
        await asyncio.to_thread(caller.join, 10)
        return await running

    asyncio.run(asyncio.wait_for(scenario(), 60))
    assert len(outcome) == 1 and isinstance(outcome[0], EngineUnavailableError)


def test_a_failing_clear_stops_the_engine_fail_closed() -> None:
    with CpuServing() as stack:
        stack.generate(LONG, max_tokens=4)

        def boom() -> int:
            raise RuntimeError("allocator invariant violated")

        stack.runtime.flush_prefix_cache = boom  # type: ignore[method-assign]
        result = stack.service.flush_cache()
        assert result.status is FlushStatus.FAILED
        assert "allocator invariant violated" in result.detail
        deadline = time.monotonic() + 10
        while stack.service.ready and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not stack.service.ready  # admission stopped
        with pytest.raises(EngineUnavailableError):
            stack.service.flush_cache()


@pytest.mark.parametrize(
    ("body", "status"),
    [
        ({}, 200),
        ({"timeout": 1.5, "drain": True}, 200),
        ({"timeout": -1}, 400),
        ({"timeout": "soon"}, 400),
        ({"drain": "yes"}, 400),
        ({"timeout": 1e9}, 400),
        ({"pages": 3}, 400),
    ],
)
def test_http_flush_route(serving: CpuServing, body: dict, status: int) -> None:
    response = serving.http().post("/v1/cache/flush", json=body)
    assert response.status_code == status, response.text
    if status == 200:
        assert response.json()["status"] == "flushed"


def test_http_flush_reports_busy_as_conflict(serving: CpuServing) -> None:
    client = serving.http()

    async def scenario():
        running = asyncio.ensure_future(serving.agenerate(LONG, max_tokens=120))
        await asyncio.sleep(0.05)
        response = await asyncio.to_thread(client.post, "/v1/cache/flush", json={})
        await running
        return response

    response = asyncio.run(asyncio.wait_for(scenario(), 60))
    assert response.status_code == 409
    payload = response.json()
    assert payload["status"] == "busy" and payload["blockers"]
