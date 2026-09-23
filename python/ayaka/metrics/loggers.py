"""Stat logger implementations: interval logging and Prometheus export."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Protocol

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from ayaka.metrics.stats import IterationStats

__all__ = ["StatLogger", "LoggingStatLogger", "PrometheusStatLogger"]

_LOG = logging.getLogger("ayaka.engine.stats")


class StatLogger(Protocol):
    """What a stat logger must implement; ``record`` fans out every step."""

    def record(self, stats: IterationStats) -> None: ...

    def log(self) -> None: ...


class LoggingStatLogger:
    """Aggregate between log intervals and emit one throughput line.

    Throughput is wall-clock tokens/second across the whole interval, matching
    vLLM's engine-level generation throughput semantics.
    """

    def __init__(self, *, clock: Callable[[], float] | None = None) -> None:
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._reset()

    def _reset(self) -> None:
        self._started = self._clock()
        self._new_tokens = 0
        self._scheduled_tokens = 0
        self._settled = 0
        self._preempted = 0
        self._finished = 0
        self._aborted = 0
        self._failed = 0
        self._last_running = 0
        self._last_waiting = 0

    def record(self, stats: IterationStats) -> None:
        with self._lock:
            self._new_tokens += stats.num_new_tokens
            self._scheduled_tokens += stats.num_scheduled_tokens
            self._settled += stats.num_settled
            self._preempted += stats.num_preempted
            self._finished += stats.num_finished
            self._aborted += stats.num_aborted
            self._failed += stats.num_failed
            self._last_running = stats.num_running
            self._last_waiting = stats.num_waiting

    def log(self) -> None:
        with self._lock:
            now = self._clock()
            elapsed = now - self._started
            if elapsed <= 0:
                return
            new_rate = self._new_tokens / elapsed
            scheduled_rate = self._scheduled_tokens / elapsed
            line = (
                "Engine stats: running=%d waiting=%d "
                "output_tokens=%d (%.1f tok/s) scheduled_tokens=%d (%.1f tok/s) "
                "settled=%d finished=%d aborted=%d failed=%d preempted=%d over %.1fs"
            )
            args = (
                self._last_running,
                self._last_waiting,
                self._new_tokens,
                new_rate,
                self._scheduled_tokens,
                scheduled_rate,
                self._settled,
                self._finished,
                self._aborted,
                self._failed,
                self._preempted,
                elapsed,
            )
            self._reset()
        _LOG.info(line, *args)


class PrometheusStatLogger:
    """Bounded-label engine metrics on a dedicated registry."""

    def __init__(self, *, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry if registry is not None else CollectorRegistry()
        self._running = Gauge(
            "ayaka_engine_running_requests",
            "Scheduler running requests",
            registry=self.registry,
        )
        self._waiting = Gauge(
            "ayaka_engine_waiting_requests",
            "Scheduler waiting requests",
            registry=self.registry,
        )
        self._new_tokens = Counter(
            "ayaka_engine_output_tokens",
            "Published output tokens",
            registry=self.registry,
        )
        self._scheduled_tokens = Counter(
            "ayaka_engine_scheduled_tokens",
            "Tokens sent to the executor (prefill + decode queries)",
            registry=self.registry,
        )
        self._requests = Counter(
            "ayaka_engine_terminal_requests",
            "Requests reaching a terminal outcome",
            ["reason"],
            registry=self.registry,
        )
        self._preemptions = Counter(
            "ayaka_engine_preemptions_total",
            "Preemption events",
            registry=self.registry,
        )
        self._step = Histogram(
            "ayaka_engine_step_seconds",
            "Engine step duration",
            buckets=(0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0),
            registry=self.registry,
        )

    def record(self, stats: IterationStats) -> None:
        self._running.set(stats.num_running)
        self._waiting.set(stats.num_waiting)
        if stats.num_new_tokens:
            self._new_tokens.inc(stats.num_new_tokens)
        if stats.num_scheduled_tokens:
            self._scheduled_tokens.inc(stats.num_scheduled_tokens)
        if stats.num_preempted:
            self._preemptions.inc(stats.num_preempted)
        if stats.num_finished:
            self._requests.labels("finished").inc(stats.num_finished)
        if stats.num_aborted:
            self._requests.labels("aborted").inc(stats.num_aborted)
        if stats.num_failed:
            self._requests.labels("failed").inc(stats.num_failed)
        self._step.observe(stats.step_seconds)

    def log(self) -> None:
        return None  # Prometheus scrape pulls lazily; nothing to emit
