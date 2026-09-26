"""Per-runtime bounded accounting and Prometheus metrics without request labels."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from ayaka.utils.math_utils import percentiles

if TYPE_CHECKING:
    from ayaka.memory.pressure import MemoryPressureMetrics
    from ayaka.memory.tiering import TieringMetrics

#: Prompt-size buckets reported in the SLO summary and fairness test lanes.
SMALL_PROMPT_TOKENS = 128
MEDIUM_PROMPT_TOKENS = 1024


def _latency_ns(start: int, end: int) -> int | None:
    """Signed elapsed time in one monotonic domain, or None when unmeasured."""
    if start <= 0 or end < start:
        return None
    return end - start


@dataclass(frozen=True, slots=True)
class UsageRecord:
    """One settled request with absolute monotonic boundaries.

    The timestamps are the raw evidence behind ``/stats``: service ingress,
    tokenizer, admission, model side (scheduled/first token) and the stream
    write.  Derived accessors keep the formulas in one place; unset boundaries
    return ``None`` instead of a fabricated zero.
    """

    request_id: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    status: str
    duration_seconds: float
    priority: int = 0
    sla_class: str = "default"
    prefix_hit: bool = False
    ingress_ns: int = 0
    validation_done_ns: int = 0
    tokenize_done_ns: int = 0
    admitted_ns: int = 0
    first_scheduled_ns: int = 0
    first_model_token_ns: int = 0
    first_published_ns: int = 0
    first_socket_write_ns: int = 0
    last_published_ns: int = 0
    terminal_ns: int = 0
    cleanup_done_ns: int = 0
    itl_p50_ns: int = 0
    itl_p95_ns: int = 0

    @property
    def prompt_bucket(self) -> str:
        if self.prompt_tokens <= SMALL_PROMPT_TOKENS:
            return "small"
        if self.prompt_tokens <= MEDIUM_PROMPT_TOKENS:
            return "medium"
        return "large"

    @property
    def queue_latency_ns(self) -> int | None:
        return _latency_ns(self.ingress_ns, self.admitted_ns)

    @property
    def tokenizer_latency_ns(self) -> int | None:
        if self.tokenize_done_ns:
            return _latency_ns(self.ingress_ns, self.tokenize_done_ns)
        return _latency_ns(self.ingress_ns, self.admitted_ns)

    @property
    def model_ttft_ns(self) -> int | None:
        return _latency_ns(self.admitted_ns, self.first_model_token_ns)

    @property
    def service_ttft_ns(self) -> int | None:
        return _latency_ns(self.ingress_ns, self.first_published_ns)

    @property
    def client_ttft_ns(self) -> int | None:
        return _latency_ns(self.ingress_ns, self.first_socket_write_ns)

    @property
    def tpot_ns(self) -> int | None:
        if self.completion_tokens < 2:
            return None
        span = _latency_ns(self.first_published_ns, self.last_published_ns)
        if span is None:
            return None
        return span // (self.completion_tokens - 1)

    def as_dict(self) -> dict:
        data = asdict(self)
        data["prompt_bucket"] = self.prompt_bucket
        data["queue_latency_ns"] = self.queue_latency_ns
        data["service_ttft_ns"] = self.service_ttft_ns
        data["tpot_ns"] = self.tpot_ns
        return data


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
            "ayaka_rejected_requests",
            "Frontend admission rejections by reason",
            ["reason"],
            registry=self.registry,
        )
        self.stream_overflows = Counter(
            "ayaka_stream_overflows",
            "Generation streams aborted because the consumer fell behind",
            ["reason"],
            registry=self.registry,
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
        # Host-tier observability. Gauges (not Counters) because each scrape
        # reads a cumulative value from the memory owner on one thread and set
        # is idempotent; a Counter could only ever be incremented.
        self.host_tier_pages = Gauge(
            "ayaka_kv_host_tier_pages",
            "Host KV mirror pages by kind",
            ["kind"],
            registry=self.registry,
        )
        self.host_tier_bytes = Gauge(
            "ayaka_kv_host_tier_bytes", "Host KV mirror payload bytes", registry=self.registry
        )
        self.host_pinned = Gauge(
            "ayaka_kv_host_pinned",
            "1 when the host mirror is page-locked, else 0",
            registry=self.registry,
        )
        self.transfer_bytes = Gauge(
            "ayaka_kv_transfer_bytes_total",
            "Cumulative tier transfer bytes by direction",
            ["direction"],
            registry=self.registry,
        )
        self.transfer_pending = Gauge(
            "ayaka_kv_transfer_pending",
            "Tier transfers submitted but not settled",
            registry=self.registry,
        )
        self.transfer_failed = Gauge(
            "ayaka_kv_transfer_failures_total",
            "Failed tier transfers",
            registry=self.registry,
        )
        self.transfer_aborts = Gauge(
            "ayaka_kv_transfer_aborts_total",
            "Failed or cancelled tier copies by kind",
            ["kind"],
            registry=self.registry,
        )
        self.quarantined_blocks = Gauge(
            "ayaka_kv_quarantined_blocks",
            "Tier blocks held under quarantine",
            registry=self.registry,
        )
        self.pressure_actions = Gauge(
            "ayaka_kv_pressure_actions_total",
            "Pressure actions by action and outcome",
            ["action", "outcome"],
            registry=self.registry,
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
        self.queue_latency = Histogram(
            "ayaka_queue_latency_seconds",
            "Service ingress to engine admission",
            buckets=buckets,
            registry=self.registry,
        )
        self.tokenizer_duration = Histogram(
            "ayaka_tokenizer_duration_seconds",
            "Service ingress to tokenization completion",
            buckets=buckets,
            registry=self.registry,
        )
        self.itl = Histogram(
            "ayaka_inter_token_latency_seconds",
            "Published inter-token gaps (bounded samples per request)",
            buckets=buckets,
            registry=self.registry,
        )
        self.goodput = Counter(
            "ayaka_goodput_requests_total",
            "Requests that finished with a usable completion",
            registry=self.registry,
        )
        self.conservation_submitted = Counter(
            "ayaka_submitted_requests_total",
            "Requests offered to admission",
            registry=self.registry,
        )
        self.conservation_admitted = Counter(
            "ayaka_admitted_requests_total",
            "Requests admitted by the engine",
            registry=self.registry,
        )

    def record_rejection(self, reason: str) -> None:
        """Count one finite admission refusal with a bounded reason label."""
        self.rejected.labels(reason).inc()

    def record(
        self,
        record: UsageRecord,
        *,
        first_token: float | None = None,
        itl_samples_ns: tuple[int, ...] = (),
    ) -> None:
        # Exactly one invocation from the serialized engine owner per lifecycle.
        with self._lock:
            self._history.append(record)
        self.requests.labels(record.status).inc()
        self.tokens.labels("prompt").inc(record.prompt_tokens)
        self.tokens.labels("completion").inc(record.completion_tokens)
        self.tokens.labels("cached").inc(record.cached_tokens)
        self.duration.observe(record.duration_seconds)
        if record.status not in ("cancelled", "abort", "error", "timeout"):
            self.goodput.inc()
        queue = record.queue_latency_ns
        if queue is not None:
            self.queue_latency.observe(queue / 1e9)
        tokenizer = record.tokenizer_latency_ns
        if tokenizer is not None:
            self.tokenizer_duration.observe(tokenizer / 1e9)
        for sample in itl_samples_ns:
            self.itl.observe(sample / 1e9)
        if first_token is not None:
            self.ttft.observe(first_token)
            if record.completion_tokens > 1:
                self.tpot.observe(
                    max(0, record.duration_seconds - first_token) / (record.completion_tokens - 1)
                )

    def rejection_counts(self) -> dict[str, float]:
        """Current rejection counter per reason (bounded labels only)."""
        return {
            sample.labels["reason"]: sample.value
            for metric in self.rejected.collect()
            for sample in metric.samples
            if sample.name == "ayaka_rejected_requests_total"
        }

    def conservation(self, *, outstanding: int) -> dict[str, int]:
        """Submitted = rejected + admitted; admitted = terminals + outstanding.

        Only admission-stream rejections (ingress/admitted/scheduler) are part
        of the submitted equation; frontend refusals (auth, body size,
        concurrency) happen before ``submit`` and are reported separately.
        Counters and terminals come from the same owner-thread settlement
        stream, so a passed check is an observed equation, not an estimate.
        """
        submitted = int(self.conservation_submitted._value.get())
        admitted = int(self.conservation_admitted._value.get())
        counts = self.rejection_counts()
        stream_rejected = int(
            counts.get("ingress", 0.0)
            + counts.get("admitted", 0.0)
            + counts.get("scheduler", 0.0)
        )
        frontend_rejected = int(sum(counts.values()) - stream_rejected)
        completed = int(self.requests.labels("stop")._value.get())
        completed += int(self.requests.labels("length")._value.get())
        completed += int(self.requests.labels("tool_call")._value.get())
        cancelled = int(self.requests.labels("cancelled")._value.get())
        cancelled += int(self.requests.labels("abort")._value.get())
        failed = int(self.requests.labels("error")._value.get())
        timed_out = int(self.requests.labels("timeout")._value.get())
        return {
            "submitted": submitted,
            "rejected": stream_rejected,
            "frontend_rejected": frontend_rejected,
            "admitted": admitted,
            "completed": completed,
            "cancelled": cancelled,
            "timed_out": timed_out,
            "failed": failed,
            "outstanding": outstanding,
        }

    def slo_summary(self) -> dict[str, float | int]:
        """p50/p95 latency summary over the bounded recent-request history.

        Every value comes from raw monotonic boundaries in ``UsageRecord``;
        requests without a measured boundary are excluded instead of counted
        as zero.  Buckets are not mixed: TTFT starts at service ingress.
        """
        records = list(self._history)
        if not records:
            return {"sample_count": 0}
        queue = [r.queue_latency_ns for r in records if r.queue_latency_ns is not None]
        tokenizer = [r.tokenizer_latency_ns for r in records if r.tokenizer_latency_ns is not None]
        service_ttft = [r.service_ttft_ns for r in records if r.service_ttft_ns is not None]
        itl = [r.itl_p50_ns for r in records if r.itl_p50_ns]
        tpot = [r.tpot_ns for r in records if r.tpot_ns is not None]
        e2e = [r.duration_seconds * 1e9 for r in records]
        summary: dict[str, float | int] = {
            "sample_count": len(records),
            "goodput": int(self.goodput._value.get()),
        }
        for name, values in (
            ("queue_latency", queue),
            ("tokenizer", tokenizer),
            ("service_ttft", service_ttft),
            ("itl", itl),
            ("tpot", tpot),
            ("e2e", e2e),
        ):
            if not values:
                continue
            stats = percentiles([float(value) for value in values])
            summary[f"{name}_p50_ms"] = stats["p50"] / 1e6
            summary[f"{name}_p95_ms"] = stats["p95"] / 1e6
            summary[f"{name}_p99_ms"] = stats["p99"] / 1e6
        return summary

    def update_tiering(self, metrics: TieringMetrics) -> None:
        """Publish one host-tier reading; never reserves or mutates memory."""
        self.host_tier_pages.labels("capacity").set(metrics.host_capacity_pages)
        self.host_tier_pages.labels("used").set(metrics.host_used_slots)
        self.host_tier_bytes.set(metrics.host_mirror_bytes)
        self.host_pinned.set(1 if metrics.host_pinned else 0)
        self.transfer_bytes.labels("to_host").set(metrics.bytes_to_host_total)
        self.transfer_bytes.labels("to_device").set(metrics.bytes_to_device_total)
        self.transfer_pending.set(metrics.pending_transfers)
        self.transfer_failed.set(metrics.failed_total)
        self.transfer_aborts.labels("spill").set(metrics.spill_aborts_total)
        self.transfer_aborts.labels("promotion").set(metrics.promotion_aborts_total)
        self.quarantined_blocks.set(metrics.quarantined_blocks)

    def update_pressure(self, metrics: MemoryPressureMetrics) -> None:
        """Publish cumulative pressure-path attempts and progress."""
        self.pressure_actions.labels("reclaim", "attempts").set(
            metrics.pressure_reclaim_attempts_total
        )
        self.pressure_actions.labels("reclaim", "progress").set(
            metrics.pressure_reclaim_progress_total
        )
        self.pressure_actions.labels("evict_prefix", "attempts").set(
            metrics.pressure_prefix_eviction_attempts_total
        )
        self.pressure_actions.labels("evict_prefix", "progress").set(
            metrics.pressure_prefix_eviction_progress_total
        )
        self.pressure_actions.labels("preempt", "progress").set(metrics.pressure_preemptions_total)
        self.pressure_actions.labels("reclaim", "pages").set(metrics.kv_reclaims_total)
        self.pressure_actions.labels("evict_prefix", "pages").set(metrics.kv_prefix_evictions_total)

    def clear_tiering(self) -> None:
        """Zero every host-tier series after the cache owner is gone."""
        self.host_tier_pages.labels("capacity").set(0)
        self.host_tier_pages.labels("used").set(0)
        self.host_tier_bytes.set(0)
        self.host_pinned.set(0)
        self.transfer_bytes.labels("to_host").set(0)
        self.transfer_bytes.labels("to_device").set(0)
        self.transfer_pending.set(0)
        self.transfer_failed.set(0)
        self.transfer_aborts.labels("spill").set(0)
        self.transfer_aborts.labels("promotion").set(0)
        self.quarantined_blocks.set(0)

    def recent(self) -> list[dict]:
        with self._lock:
            return [record.as_dict() for record in self._history]

    def render(self) -> bytes:
        return generate_latest(self.registry)
