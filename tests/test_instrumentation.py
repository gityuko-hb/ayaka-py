"""P2 instrumentation that measures instead of guessing (research A-04, A-07, CT-P, CT-O01).

* Enqueue gate: wait/held host time per owner, outermost acquisition only, so
  re-entrant scopes count once and contention is visible (A-04: measure first).
* Kernel profiler: raw device events give measured per-invocation min/max.
* Profiling sessions: one bounded window at a time on the owner thread, with
  atomic option validation, no second recorder, cleanup on every failure.
* Owner wake: cancel signals the owner; an idle owner sleeps to the nearest
  deadline instead of spinning, but keeps polling while work needs it.
* Cache events: bounded pull log, oldest dropped and counted, never blocking.
* Step cost: an analytical FLOPs/bytes model checked against a FLOP counter,
  labelled as an estimate.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from ayaka.configs.serving import ServingConfig
from ayaka.kernel.profiler.kernel_profiler import _parse_profiler_events
from ayaka.metrics.estimate import (
    DenseModelShape,
    SliceShape,
    StepCostEstimate,
    estimate_step_cost,
)
from ayaka.prefix.events import CacheEventKind, CacheEventLog
from ayaka.runner.graph.pool import (
    enqueue_gate_stats,
    execution_enqueue_gate,
    reset_enqueue_gate_stats,
)
from ayaka.serving.profiling import (
    ProfilerBusyError,
    ProfileSessionStatus,
    validate_profile_options,
)
from ayaka.serving.service import GenerationHandle, ServingService
from cpu_e2e import MODEL, CpuServing, build_model, prompt_of
from torch.utils.flop_counter import FlopCounterMode

PROMPTS = [prompt_of(*[(i * 11 + k * 5) % 120 for k in range(4 + i)]) for i in range(6)]
LONG = prompt_of(*range(40, 80))


# ---------------------------------------------------------------------------
# A-04: enqueue gate telemetry
# ---------------------------------------------------------------------------


def _gate(owner: str):
    return next((s for s in enqueue_gate_stats() if s.owner == owner), None)


def test_uncontended_scopes_count_once_and_measure_holding() -> None:
    reset_enqueue_gate_stats()
    with execution_enqueue_gate("solo"):
        time.sleep(0.01)
        with execution_enqueue_gate("solo"):  # re-entrant: no wait, not counted again
            pass
    stats = _gate("solo")
    assert stats is not None
    assert (stats.acquisitions, stats.contended) == (1, 0)
    assert stats.held_ns_total >= 9_000_000 and stats.wait_ns_max < 5_000_000


def test_contention_is_counted_with_its_wait() -> None:
    reset_enqueue_gate_stats()
    holding = threading.Event()
    release = threading.Event()

    def holder() -> None:
        with execution_enqueue_gate("holder"):
            holding.set()
            release.wait(5)

    thread = threading.Thread(target=holder)
    thread.start()
    holding.wait(5)
    threading.Timer(0.05, release.set).start()
    with execution_enqueue_gate("waiter"):
        pass
    thread.join()
    waiter = _gate("waiter")
    assert waiter is not None and waiter.contended == 1
    assert waiter.wait_ns_total >= 30_000_000  # it really waited for the holder


def test_an_exception_inside_the_gate_still_records_and_releases() -> None:
    reset_enqueue_gate_stats()
    with pytest.raises(RuntimeError), execution_enqueue_gate("boom"):
        raise RuntimeError("enqueue failed")
    with execution_enqueue_gate("boom"):  # not left held
        pass
    stats = _gate("boom")
    assert stats is not None and stats.acquisitions == 2


def test_serving_workers_report_their_gate_time() -> None:
    reset_enqueue_gate_stats()
    with CpuServing() as stack:
        stack.generate(PROMPTS[0], max_tokens=8)
        report = stack.http().get("/stats").json()["enqueue_gate"]
    assert "worker:cpu" in report and report["worker:cpu"]["acquisitions"] >= 8


# ---------------------------------------------------------------------------
# A-07: per-invocation kernel extrema from raw events
# ---------------------------------------------------------------------------


def _raw(*events: tuple[str, float, str]) -> SimpleNamespace:
    return SimpleNamespace(
        events=lambda: [
            SimpleNamespace(key=name, self_device_time_total=us, device_type=kind)
            for name, us, kind in events
        ],
        key_averages=lambda: [],
    )


def test_raw_device_events_give_measured_min_and_max() -> None:
    stats = _parse_profiler_events(
        _raw(
            ("gemm", 10.0, "DeviceType.CUDA"),
            ("gemm", 30.0, "DeviceType.CUDA"),
            ("gemm", 20.0, "DeviceType.CUDA"),
            ("norm", 5.0, "DeviceType.CUDA"),
            ("aten::mm", 60.0, "DeviceType.CPU"),  # a CPU op's self-CUDA time repeats kernels
        )
    )
    gemm, norm = stats
    assert (gemm.name, gemm.calls, gemm.min_us, gemm.max_us, gemm.avg_us) == (
        "gemm",
        3,
        10.0,
        30.0,
        20.0,
    )
    assert gemm.pct == round(60 / 65 * 100, 2) and norm.min_us == norm.max_us == 5.0


def test_without_raw_device_events_the_aggregate_path_reports_none() -> None:
    aggregate = SimpleNamespace(
        events=lambda: [],
        key_averages=lambda: [SimpleNamespace(key="k", count=2, self_cuda_time_total=8.0)],
    )
    (stats,) = _parse_profiler_events(aggregate)
    assert stats.min_us is None and stats.max_us is None and stats.avg_us == 4.0


# ---------------------------------------------------------------------------
# CT-P: profiling sessions on the owner thread
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "options",
    [
        {"max_steps": 0},
        {"max_steps": -3},
        {"max_steps": 2.5},
        {"max_steps": True},
        {"activities": ("gpu",)},
        {"activities": ()},
        {"trace_path": "/nonexistent-dir/trace.json"},
    ],
)
def test_invalid_profile_options_are_refused_before_any_state(options: dict) -> None:
    arguments = {"max_steps": None, "trace_path": None, "activities": ("cpu",)} | options
    with pytest.raises(ValueError):
        validate_profile_options(**arguments)


def test_cuda_profiling_is_refused_on_a_cpu_host() -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA is available here")
    with pytest.raises(ValueError, match="CUDA is not available"):
        validate_profile_options(None, None, ("cpu", "cuda"))


@pytest.fixture(scope="module")
def serving():
    with CpuServing() as stack:
        yield stack


def test_a_session_records_steps_and_exports_a_trace(serving: CpuServing, tmp_path: Path) -> None:
    trace = tmp_path / "trace.json"
    expected = serving.generate(PROMPTS[1], max_tokens=8).text
    session = serving.service.start_profile(trace_path=str(trace))
    with pytest.raises(ProfilerBusyError):
        serving.service.start_profile()  # never a second recorder
    profiled = serving.generate(PROMPTS[1], max_tokens=8).text
    result = serving.service.stop_profile()
    assert result.session == session and result.status is ProfileSessionStatus.COMPLETED
    assert result.steps > 0 and result.wall_seconds > 0 and result.reason == "stop"
    assert json.loads(trace.read_text())["traceEvents"]
    assert profiled == expected  # profiling does not change outputs
    with pytest.raises(ProfilerBusyError):
        serving.service.stop_profile()  # nothing active any more
    assert serving.service.start_profile() == session + 1  # ids keep increasing
    serving.service.stop_profile()


def test_max_steps_stops_the_session_by_itself(serving: CpuServing) -> None:
    serving.service.start_profile(max_steps=3)
    serving.generate(PROMPTS[2], max_tokens=12)
    last = serving.service.last_profile()
    assert last is not None and last.status is ProfileSessionStatus.COMPLETED
    assert (last.steps, last.reason) == (3, "max_steps")
    with pytest.raises(ProfilerBusyError):
        serving.service.stop_profile()


class _Recorder:
    def __init__(self, *, fail_start: bool = False, fail_stop: bool = False) -> None:
        self.fail_start, self.fail_stop = fail_start, fail_stop
        self.stops = 0

    def start(self) -> None:
        if self.fail_start:
            raise RuntimeError("kineto init failed")

    def stop(self) -> None:
        self.stops += 1
        if self.fail_stop:
            raise OSError("trace buffer lost")

    def events(self):
        return []

    def key_averages(self):
        return []


def test_a_recorder_that_fails_to_start_leaves_nothing_running(serving: CpuServing) -> None:
    recorder = _Recorder(fail_start=True)
    with pytest.raises(RuntimeError, match="failed to start"):
        serving.service.start_profile(recorder_factory=lambda _: recorder)
    assert recorder.stops == 1  # torn down
    last = serving.service.last_profile()
    assert last is not None and last.status is ProfileSessionStatus.FAILED
    serving.service.start_profile()  # the owner is free again
    serving.service.stop_profile()


def test_a_recorder_that_fails_to_stop_is_reported_failed(serving: CpuServing) -> None:
    recorder = _Recorder(fail_stop=True)
    serving.service.start_profile(recorder_factory=lambda _: recorder)
    result = serving.service.stop_profile()
    assert result.status is ProfileSessionStatus.FAILED and "trace buffer lost" in result.reason
    serving.service.start_profile()
    serving.service.stop_profile()


def test_shutdown_ends_an_active_session() -> None:
    stack = CpuServing()
    stack.service.start_profile()
    stack.generate(PROMPTS[0], max_tokens=4)
    stack.close()
    last = stack.service.last_profile()
    assert last is not None and last.reason == "shutdown"
    assert last.status is ProfileSessionStatus.COMPLETED


def test_profile_routes(serving: CpuServing) -> None:
    client = serving.http()
    started = client.post("/v1/profile/start", json={"max_steps": 1000})
    assert started.status_code == 200
    assert client.post("/v1/profile/start", json={}).status_code == 409
    assert client.post("/v1/profile/start", json={"max_steps": 0}).status_code == 400
    stopped = client.post("/v1/profile/stop")
    assert stopped.status_code == 200 and stopped.json()["status"] == "completed"
    assert client.post("/v1/profile/stop").status_code == 409


# ---------------------------------------------------------------------------
# Owner wake: cancel signal and deadline-aware idle sleep
# ---------------------------------------------------------------------------


def test_cancel_wakes_the_owner() -> None:
    request = SimpleNamespace(constraint=None)
    spec = SimpleNamespace(messages=None, tools=(), tool_choice="none")
    handle = GenerationHandle(request, spec, ServingConfig())  # type: ignore[arg-type]
    woken = threading.Event()
    handle._wake = woken.set
    handle.cancel()
    assert handle.cancelled.is_set() and woken.is_set()


def _owner(handles: int, *, pending: bool = False, deadlines=()) -> ServingService:
    service = ServingService.__new__(ServingService)
    service._handles = {f"r{i}": object() for i in range(handles)}  # type: ignore[assignment]
    service._flush = None
    service._weights = None
    service._migration = None
    service._clock = lambda: 1_000_000_000
    lifecycles = [
        SimpleNamespace(request=SimpleNamespace(deadline_ns=d, queue_deadline_ns=None))
        for d in deadlines
    ]
    service.engine = SimpleNamespace(
        executor=SimpleNamespace(pending_completion=pending),
        kv=SimpleNamespace(closed=False, tier_metrics=lambda: None),
        requests=lifecycles,
    )
    return service


def test_the_idle_sleep_policy() -> None:
    assert _owner(0)._idle_wait_seconds(False) == 0.05  # nothing live: long sleep
    assert _owner(2)._idle_wait_seconds(True) == 0.001  # progress: keep stepping
    assert _owner(2, pending=True)._idle_wait_seconds(False) == 0.001  # a fence to poll
    stuck = _owner(1, deadlines=(1_000_000_000 + 10_000_000,))
    assert stuck._idle_wait_seconds(False) == pytest.approx(0.01)  # wake for the deadline
    assert _owner(1)._idle_wait_seconds(False) == 0.05  # no deadline: bounded long sleep


def test_deadlines_still_fire_under_the_new_sleep(serving: CpuServing) -> None:
    result = serving.generate(PROMPTS[3], max_tokens=200, timeout=0.05)
    assert result.finish_reason == "timeout"


def test_a_cancel_reaches_the_owner_promptly(serving: CpuServing) -> None:
    async def scenario():
        started = time.monotonic()
        generation = await serving.agenerate(PROMPTS[4], max_tokens=200, cancel_after=1)
        return generation, time.monotonic() - started

    generation, elapsed = asyncio.run(asyncio.wait_for(scenario(), 60))
    assert generation.finish_reason == "cancelled" and elapsed < 5


# ---------------------------------------------------------------------------
# Bounded cache-event log
# ---------------------------------------------------------------------------


def test_the_log_drops_the_oldest_and_shows_the_gap() -> None:
    log = CacheEventLog(capacity=2)
    for index in range(5):
        log.record(CacheEventKind.STORED, bytes([index]), 16)
    assert (log.recorded, log.dropped, log.pending) == (5, 3, 2)
    events = log.drain()
    assert [event.sequence for event in events] == [4, 5]  # 1..3 visibly lost
    assert log.drain() == () and log.pending == 0


def test_serving_records_publish_and_flush_events_without_waiting_for_consumers() -> None:
    config = ServingConfig(model=MODEL, structured_outputs=False, cache_event_buffer=8)
    with CpuServing(config=config) as stack:
        log = stack.runtime.cache_events
        assert log is not None
        for start in range(0, 100, 20):  # five prompts with two full pages each
            stack.generate(prompt_of(*range(start, start + 40)), max_tokens=4)
        assert stack.service.flush_cache().flushed  # nobody drained meanwhile
        assert log.pending <= 8 and log.dropped > 0  # bounded, loss counted
        events = log.drain()
        assert events[-1].kind is CacheEventKind.CLEARED
        sequences = [event.sequence for event in events]
        assert sequences == sorted(sequences)
        stored = {event.digest for event in events if event.kind is CacheEventKind.STORED}
        assert stored, "the long prompt's complete blocks were recorded"
        response = stack.http().get("/v1/cache/events")
        assert response.status_code == 200 and response.json()["events"] == []


def test_stored_digests_are_the_index_digests() -> None:
    config = ServingConfig(model=MODEL, structured_outputs=False, cache_event_buffer=64)
    with CpuServing(config=config) as stack:
        log = stack.runtime.cache_events
        assert log is not None
        stack.generate(LONG, max_tokens=4)
        stored = [e for e in log.drain() if e.kind is CacheEventKind.STORED]
        prefix = stack.runtime.kv.prefix_service  # type: ignore[union-attr]
        blocks = {info.identity.digest: info for info in prefix.store.cached_block_index()}
        assert [e.token_count for e in stored] == [16, 32]  # tokens covered, cumulative
        for event in stored:
            assert blocks[event.digest].identity.token_count == event.token_count


def test_cache_events_are_off_by_default() -> None:
    with CpuServing() as stack:
        assert stack.runtime.cache_events is None
        assert stack.http().get("/v1/cache/events").status_code == 400


# ---------------------------------------------------------------------------
# CT-O01: analytical step cost, labelled as an estimate
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny():
    model = build_model()
    return model, DenseModelShape.from_config(model.config)


@pytest.mark.parametrize("tokens", [1, 7, 33])
def test_weight_flops_match_a_flop_counter_on_the_dense_forward(tiny, tokens: int) -> None:
    """forward_dense computes logits for every row, so every row is a logits row."""
    model, shape = tiny
    with FlopCounterMode(display=False) as counter:
        model.forward_dense(torch.arange(tokens) + 4)
    estimate = estimate_step_cost(
        shape,
        [SliceShape(tokens, 0, logits_rows=tokens)],
        weight_bytes_per_value=4,
        kv_bytes_per_value=4,
    )
    assert estimate.linear_flops + estimate.lm_head_flops == counter.get_total_flops()


def test_causal_attention_counts_the_triangle_on_top_of_the_prefix(tiny) -> None:
    _, shape = tiny
    per_key = 4 * shape.num_heads * shape.head_dim * shape.num_layers

    def attention(*slices: SliceShape) -> int:
        return estimate_step_cost(
            shape, slices, weight_bytes_per_value=2, kv_bytes_per_value=2
        ).attention_flops

    assert attention(SliceShape(1, 100)) == per_key * 101  # decode reads every key once
    assert attention(SliceShape(4, 0)) == per_key * (1 + 2 + 3 + 4)
    assert attention(SliceShape(64), SliceShape(64)) == 2 * attention(SliceShape(64))
    prefix_heavy = attention(SliceShape(8, 1000))
    assert prefix_heavy == per_key * (8 * 1000 + 36)


def test_bytes_follow_ideal_reuse_and_the_empty_step_is_free(tiny) -> None:
    _, shape = tiny
    cost = estimate_step_cost(
        shape,
        [SliceShape(1, 9), SliceShape(3, 0, logits_rows=0)],
        weight_bytes_per_value=2,
        kv_bytes_per_value=1,
    )
    per_token = shape.kv_values_per_token
    assert cost.kv_read_bytes == (10 + 3) * per_token
    assert cost.kv_write_bytes == 4 * per_token
    weights = shape.linear_params_per_layer * shape.num_layers + shape.lm_head_params
    assert cost.weight_bytes == weights * 2  # once per step, however many slices
    assert cost.estimated and "estimate" in cost.provenance
    empty = estimate_step_cost(shape, [], weight_bytes_per_value=2, kv_bytes_per_value=2)
    assert empty == StepCostEstimate(0, 0, 0, 0, 0, 0) and empty.total_flops == 0


def test_slice_shapes_are_validated() -> None:
    with pytest.raises(ValueError):
        SliceShape(1, 0, logits_rows=2)
    with pytest.raises(ValueError):
        DenseModelShape(0, 64, 4, 2, 16, 128, 128)
