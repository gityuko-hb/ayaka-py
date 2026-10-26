"""Serving observability that cannot lie (research P1: watchdog, load snapshot, CT-O03).

* Completion watchdog: a ticket unsettled past ``completion_watchdog_seconds``
  makes the service not ready and names the ticket, step and requests; nothing is
  freed, and readiness returns when the ticket settles. Off by default.
* Detached load snapshot: immutable, versioned, timestamped, readable from any
  thread without touching live state, aged so a stale view is visible.
* Ingress token quota: queued prompt tokens are bounded, not only the command
  count; one request always fits an empty queue.
* Token conservation: every published token is forwarded to its stream or
  withheld with a reason; at quiescence published equals reported usage.
* ITL: a speculative burst of tokens published at one instant adds no
  fabricated zero-length inter-token samples.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from types import SimpleNamespace

import pytest
from ayaka.configs.serving import ServingConfig
from ayaka.request.schema import Request, RequestId
from ayaka.serving.errors import EngineUnavailableError, OverloadedError
from ayaka.serving.load import LoadSnapshot
from ayaka.serving.service import ServingService
from ayaka.serving.stats import ServingStats
from ayaka.speculative.config import SpeculativeDecodingConfig
from ayaka.utils.timing import RequestTiming
from cpu_e2e import MODEL, CpuServing, build_model, prompt_of

PROMPTS = [prompt_of(*[(i * 11 + k * 5) % 120 for k in range(4 + i)]) for i in range(6)]


def _wait_for(predicate, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached in time")
        time.sleep(0.005)


class _FrozenCompletion:
    """Make completion polling report nothing, as a stuck device fence would."""

    def __init__(self, executor) -> None:
        self._executor = executor
        self._poll = executor._poll
        self.frozen = threading.Event()

    def __enter__(self) -> _FrozenCompletion:
        def poll(ticket) -> None:
            if not self.frozen.is_set():
                self._poll(ticket)

        self._executor._poll = poll
        return self

    def __exit__(self, *_: object) -> None:
        self.frozen.clear()
        del self._executor._poll


# ---------------------------------------------------------------------------
# Completion watchdog
# ---------------------------------------------------------------------------


def _watched(seconds: float | None) -> CpuServing:
    config = ServingConfig(
        model=MODEL, structured_outputs=False, completion_watchdog_seconds=seconds
    )
    return CpuServing(config=config)


def test_a_stuck_ticket_trips_the_watchdog_and_recovery_restores_readiness() -> None:
    with _watched(0.2) as stack:
        expected = stack.generate(PROMPTS[0], max_tokens=24).text
        service = stack.service
        with _FrozenCompletion(service.engine.executor) as freeze:

            async def scenario():
                freeze.frozen.set()  # the first ticket of the next request gets stuck
                running = asyncio.ensure_future(stack.agenerate(PROMPTS[0], max_tokens=24))
                await asyncio.to_thread(_wait_for, lambda: service.stall is not None)
                stall = service.stall
                assert not service.ready  # admission stops while the ticket is stuck
                with pytest.raises(EngineUnavailableError):
                    await stack.agenerate(PROMPTS[1], max_tokens=2)
                load = service.load_snapshot()
                stats = stack.http().get("/stats").json()["watchdog"]
                freeze.frozen.clear()  # the fence completes after all
                await asyncio.to_thread(_wait_for, lambda: service.ready)
                return stall, load, stats, await running

            stall, load, stats, generation = asyncio.run(asyncio.wait_for(scenario(), 60))
        assert stall is not None and stall.age_ns > 0.2e9
        assert len(stall.request_ids) == 1 and stall.state.value in ("submitted", "draining")
        assert load.stalled and not load.ready and load.oldest_ticket_age_seconds > 0.2
        assert stats["threshold_seconds"] == 0.2
        assert stats["stalled"]["request_ids"] == list(stall.request_ids)
        assert generation.text == expected  # nothing was freed or replayed
        registry = service.stats.registry
        assert registry.get_sample_value("ayaka_completion_stalls_total") == 1.0
        stack.assert_quiescent_accounting()


def test_without_a_threshold_a_slow_ticket_only_ages_the_gauge() -> None:
    with _watched(None) as stack:
        service = stack.service
        with _FrozenCompletion(service.engine.executor) as freeze:

            async def scenario():
                freeze.frozen.set()
                running = asyncio.ensure_future(stack.agenerate(PROMPTS[0], max_tokens=8))
                await asyncio.to_thread(
                    _wait_for,
                    lambda: (service.load_snapshot().oldest_ticket_age_seconds or 0) >= 0.25,
                )
                ready, load = service.ready, service.load_snapshot()
                freeze.frozen.clear()
                return ready, load, await running

            ready, load, generation = asyncio.run(asyncio.wait_for(scenario(), 60))
        assert ready and service.stall is None  # the default never changes readiness
        assert load.oldest_ticket_age_seconds is not None
        assert load.oldest_ticket_age_seconds >= 0.25
        assert generation.finish_reason == "length"


@pytest.mark.parametrize("seconds", [0, -1.0, float("nan"), float("inf"), True])
def test_watchdog_threshold_must_be_positive_and_finite(seconds: object) -> None:
    with pytest.raises(ValueError):
        ServingConfig(completion_watchdog_seconds=seconds)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Detached load snapshot
# ---------------------------------------------------------------------------


def test_the_load_snapshot_is_versioned_detached_and_tracks_work() -> None:
    with CpuServing() as stack:
        service = stack.service
        seen: list[LoadSnapshot] = []
        stop = threading.Event()

        def reader() -> None:  # another thread, no locks, no live state
            while not stop.is_set():
                seen.append(service.load_snapshot())
                time.sleep(0.001)

        thread = threading.Thread(target=reader)
        thread.start()
        try:
            stack.generate_concurrently(PROMPTS[:3], max_tokens=24)
        finally:
            stop.set()
            thread.join()
        versions = [snapshot.version for snapshot in seen]
        assert versions == sorted(versions) and versions[-1] > versions[0]
        assert max(s.live_generations for s in seen) >= 1
        assert max(s.running for s in seen) >= 1
        assert all(s.total_pages == seen[-1].total_pages > 0 for s in seen if s.version)
        _wait_for(lambda: service.load_snapshot().live_generations == 0)
        final = service.load_snapshot()
        assert (
            final.ready
            and final.queued_prompt_tokens == 0
            and final.oldest_ticket_age_seconds is None
        )
        report = stack.http().get("/load").json()
        assert report["version"] >= final.version and 0 <= report["age_seconds"] < 5


def test_snapshot_age_and_staleness_are_explicit() -> None:
    snapshot = LoadSnapshot(1, 1_000, True, 0, 0, 0, 0, 0, 10, 10)
    assert snapshot.age_seconds(1_000 + 2_000_000_000) == 2.0
    assert snapshot.is_stale(1.0, 1_000 + 2_000_000_000)
    assert not snapshot.is_stale(5.0, 1_000 + 2_000_000_000)
    assert snapshot.age_seconds(0) == 0.0  # a reader clock behind the writer is not negative
    assert snapshot.as_dict(now_ns=1_000)["age_seconds"] == 0.0


# ---------------------------------------------------------------------------
# Ingress token quota (bare service: the owner thread never drains it)
# ---------------------------------------------------------------------------


def _ingress(limit: int) -> ServingService:
    service = ServingService.__new__(ServingService)
    service.config = ServingConfig(max_pending_prompt_tokens=limit)
    service.stats = ServingStats()
    service._lock = threading.Lock()
    service._commands = queue.Queue(maxsize=16)
    service._queued_tokens = 0
    service._signal = threading.Event()
    service._closing = threading.Event()
    service._stopped = threading.Event()
    service._fatal = None
    service._stall = None
    service.engine = SimpleNamespace(
        kv=SimpleNamespace(closed=False), executor=SimpleNamespace(accepting_work=True)
    )
    return service


def _request(tokens: int) -> Request:
    return Request(request_id=RequestId(f"r{tokens}"), prompt_token_ids=tuple(range(1, tokens + 1)))


def test_queued_prompt_tokens_are_bounded_and_returned_on_dequeue() -> None:
    service = _ingress(limit=10)
    spec = SimpleNamespace()
    first = _request(6)
    service.submit(first, spec)  # type: ignore[arg-type]
    with pytest.raises(OverloadedError, match="token budget"):
        service.submit(_request(5), spec)  # type: ignore[arg-type]
    assert service._queued_tokens == 6
    rejected = service.stats.registry.get_sample_value(
        "ayaka_rejected_requests_total", {"reason": "ingress_tokens"}
    )
    assert rejected == 1.0
    assert service.stats.conservation(outstanding=1)["submitted"] == 2
    service._dequeued(first)
    service.submit(_request(9), spec)  # type: ignore[arg-type]
    assert service._queued_tokens == 9


def test_one_oversized_prompt_still_enters_an_empty_queue() -> None:
    service = _ingress(limit=4)
    service.submit(_request(20), SimpleNamespace())  # type: ignore[arg-type]
    assert service._queued_tokens == 20
    with pytest.raises(OverloadedError):
        service.submit(_request(1), SimpleNamespace())  # type: ignore[arg-type]


def test_a_zero_quota_never_limits_tokens() -> None:
    service = _ingress(limit=0)
    for size in (50, 60, 70):
        service.submit(_request(size), SimpleNamespace())  # type: ignore[arg-type]
    assert service._queued_tokens == 180


def test_the_quota_serves_normally_and_drains_to_zero() -> None:
    config = ServingConfig(model=MODEL, structured_outputs=False, max_pending_prompt_tokens=64)
    with CpuServing(config=config) as stack:
        results = stack.generate_concurrently(PROMPTS[:4], max_tokens=6)
        assert {g.finish_reason for g in results} == {"length"}
        assert stack.service._queued_tokens == 0


# ---------------------------------------------------------------------------
# Token conservation (CT-O03)
# ---------------------------------------------------------------------------


def test_every_published_token_is_forwarded_or_withheld_with_a_reason() -> None:
    config = ServingConfig(model=MODEL, structured_outputs=False, max_stream_events=2)
    with CpuServing(config=config) as stack:

        async def slow_reader():
            handle = await stack.pipeline.prepare(
                stack.spec(PROMPTS[3], max_tokens=24), tenant="default"
            )
            await asyncio.sleep(0.4)  # the queue overflows meanwhile
            return [event async for event in stack.pipeline.generate_events(handle)]

        async def scenario():
            return await asyncio.gather(
                stack.agenerate(PROMPTS[0], max_tokens=12),
                stack.agenerate(PROMPTS[1], max_tokens=200, cancel_after=3),
                stack.agenerate(PROMPTS[2], max_tokens=200, timeout=0.05),
                slow_reader(),
            )

        normal, cancelled, timed_out, overflowed = asyncio.run(asyncio.wait_for(scenario(), 60))
        assert normal.finish_reason == "length" and cancelled.finish_reason == "cancelled"
        assert timed_out.finish_reason == "timeout"
        assert "too slow" in overflowed[-1].message
        _wait_for(lambda: not stack.service._handles)
        ledger = stack.service.stats.token_conservation()
        withheld = ledger["withheld"]
        assert isinstance(withheld, dict)
        assert ledger["published"] == ledger["forwarded"] + sum(withheld.values())
        assert ledger["published"] == ledger["usage_completion"]  # quiescent: all settled
        assert ledger["mismatches"] == 0
        assert withheld.get("events", 0) >= 1  # the overflowing stream
        stack.assert_quiescent_accounting()
        stats = stack.http().get("/stats").json()["token_conservation"]
        assert stats["published"] == ledger["published"]


# ---------------------------------------------------------------------------
# Inter-token latency without fabricated samples
# ---------------------------------------------------------------------------


def test_a_burst_published_at_one_instant_adds_one_interval() -> None:
    timing = RequestTiming()
    for now in (100, 200, 200, 200, 350):
        timing.mark_streamed(now)
    timing.num_output_tokens = 5
    assert timing.inter_token_intervals_ns == [100, 150]
    assert timing.tpot_ns() == (350 - 100) // 4  # TPOT still counts every token


def test_speculative_bursts_record_no_zero_intervals() -> None:
    config = ServingConfig(
        model=MODEL,
        structured_outputs=False,
        speculative_decoding=SpeculativeDecodingConfig(max_draft_tokens=4),
    )
    with CpuServing(model=build_model(scale=1.0), config=config) as stack:
        metrics = stack.service.engine.speculation.metrics
        result = stack.generate(PROMPTS[0], max_tokens=32)
        assert metrics.draft_tokens_accepted > 0  # at least one multi-token burst
        histogram = stack.service.stats.registry
        samples = histogram.get_sample_value("ayaka_inter_token_latency_seconds_count")
        zeros = histogram.get_sample_value(
            "ayaka_inter_token_latency_seconds_bucket", {"le": "0.0"}
        )
        assert samples is not None and samples < result.completion_tokens - 1
        assert not zeros  # no zero-length interval was recorded
