"""Per-runtime bounded accounting and Prometheus metrics without request labels."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import asdict, dataclass

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest


@dataclass(frozen=True, slots=True)
class UsageRecord:
    request_id: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    status: str
    duration_seconds: float


class ServingStats:
    def __init__(self, *, history_size=128):
        self.registry = CollectorRegistry()
        self._lock = threading.Lock()
        self._history: deque[UsageRecord] = deque(maxlen=history_size)
        self.requests = Counter(
            "ayaka_requests", "Finished admitted requests", ["status"], registry=self.registry
        )
        self.tokens = Counter(
            "ayaka_tokens", "Engine token counts", ["kind"], registry=self.registry
        )
        self.rejected = Counter(
            "ayaka_rejected_requests", "Frontend admission rejections", registry=self.registry
        )
        self.running = Gauge(
            "ayaka_running_requests", "Scheduler running requests", registry=self.registry
        )
        self.waiting = Gauge(
            "ayaka_waiting_requests", "Scheduler waiting requests", registry=self.registry
        )
        self.kv_free = Gauge(
            "ayaka_kv_free_pages", "Unallocated resident KV pages", registry=self.registry
        )
        self.kv_total = Gauge(
            "ayaka_kv_total_pages", "Resident KV page capacity", registry=self.registry
        )
        buckets = (0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 30, 60, 300)
        self.ttft = Histogram(
            "ayaka_time_to_first_token_seconds",
            "Admission to first published token",
            buckets=buckets,
            registry=self.registry,
        )
        self.tpot = Histogram(
            "ayaka_time_per_output_token_seconds",
            "Mean time between published output tokens",
            buckets=buckets,
            registry=self.registry,
        )
        self.duration = Histogram(
            "ayaka_request_duration_seconds",
            "Admission to terminal result",
            buckets=buckets,
            registry=self.registry,
        )

    def record(self, record: UsageRecord, *, first_token: float | None) -> None:
        # Exactly one invocation from the serialized engine owner per lifecycle.
        with self._lock:
            self._history.append(record)
        self.requests.labels(record.status).inc()
        self.tokens.labels("prompt").inc(record.prompt_tokens)
        self.tokens.labels("completion").inc(record.completion_tokens)
        self.tokens.labels("cached").inc(record.cached_tokens)
        self.duration.observe(record.duration_seconds)
        if first_token is not None:
            self.ttft.observe(first_token)
            if record.completion_tokens > 1:
                self.tpot.observe(
                    max(0, record.duration_seconds - first_token) / (record.completion_tokens - 1)
                )

    def recent(self) -> list[dict]:
        with self._lock:
            return [asdict(record) for record in self._history]

    def render(self) -> bytes:
        return generate_latest(self.registry)
