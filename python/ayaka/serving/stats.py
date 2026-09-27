"""Per-runtime bounded accounting and Prometheus metrics without request labels."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from ayaka.utils.math_utils import percentiles

if TYPE_CHECKING:
    from ayaka.configs.serving import SloTargets
    from ayaka.memory.pressure import MemoryPressureMetrics
    from ayaka.memory.tiering import TieringMetrics

#: Prompt-size buckets reported in the SLO summary and fairness test lanes.
SMALL_PROMPT_TOKENS = 128
MEDIUM_PROMPT_TOKENS = 1024

#: Terminal statuses counted in the conservation equation.  ``eos`` is the
#: model's natural stop and ``tool_call`` is accepted for a tool-parser finish;
#: every settlement status falls in exactly one bucket, and an unknown status
#: is conservatively bucketed as failed so the equation cannot silently drift.
_COMPLETED_STATUSES = frozenset({"stop", "length", "eos", "tool_call"})
_CANCELLED_STATUSES = frozenset({"cancelled", "abort"})
_TIMED_OUT_STATUSES = frozenset({"timeout", "queue_timeout"})
_GOODPUT_EXCLUDED_STATUSES = _CANCELLED_STATUSES | _TIMED_OUT_STATUSES | {"error"}

#: Rejection reasons that are counted outside ``submit``: these are refused by
#: the frontend before a request ever enters the submitted equation.  Every
#: other reason is an admission-stream refusal by construction.
_FRONTEND_REJECTION_REASONS = frozenset(
    {"auth", "request_bytes", "request_chunks", "concurrent", "control"}
)


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
    cancel_reason: str = ""

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

    @property
    def e2e_ns(self) -> int | None:
        """Service ingress to terminal settlement; the request-visible span."""
        return _latency_ns(self.ingress_ns, self.terminal_ns)

    def as_dict(self) -> dict:
        data = asdict(self)
        data["prompt_bucket"] = self.prompt_bucket
        data["queue_latency_ns"] = self.queue_latency_ns
        data["tokenizer_latency_ns"] = self.tokenizer_latency_ns
        data["model_ttft_ns"] = self.model_ttft_ns
        data["service_ttft_ns"] = self.service_ttft_ns
        data["client_ttft_ns"] = self.client_ttft_ns
        data["e2e_ns"] = self.e2e_ns
        data["tpot_ns"] = self.tpot_ns
        return data


class ServingStats:
    def __init__(
        self,
        *,
        history_size: int = 128,
        clock: Callable[[], int] | None = None,
    ):
        self.registry = CollectorRegistry()
        self._lock = threading.Lock()
        self._clock = clock or time.monotonic_ns
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
        self.client_ttft = Histogram(
            "ayaka_client_ttft_seconds",
            "Service ingress to the first socket write",
            buckets=buckets,
            registry=self.registry,
        )
        self.goodput = Counter(
            "ayaka_goodput_requests_total",
            "Requests that finished with a usable completion",
            registry=self.registry,
        )
        self.prefix_hits = Counter(
            "ayaka_prefix_cache_hits_total",
            "Settled requests that consumed a cached prefix",
            registry=self.registry,
        )
        self.prefix_misses = Counter(
            "ayaka_prefix_cache_misses_total",
            "Settled requests that forwarded a full prompt",
            registry=self.registry,
        )
        self.cancellations = Counter(
            "ayaka_cancellations_total",
            "Terminal cancellations by bounded reason",
            ["reason"],
            registry=self.registry,
        )
        # Engine-side cumulative counters are published as gauges (one owner
        # thread sets them idempotently, exactly like the tiering readings).
        self.preemptions = Gauge(
            "ayaka_preemptions_total", "Recompute preemptions observed", registry=self.registry
        )
        self.recomputed_tokens = Gauge(
            "ayaka_recomputed_tokens_total",
            "Prompt tokens forwarded again after preemption",
            registry=self.registry,
        )
        self.wasted_compute_tokens = Gauge(
            "ayaka_wasted_compute_tokens_total",
            "Forwarded tokens discarded before publication",
            registry=self.registry,
        )
        self.graph_hits = Gauge(
            "ayaka_graph_hits_total", "Decode-graph replay hits", registry=self.registry
        )
        self.graph_misses = Gauge(
            "ayaka_graph_misses_total", "Decode-graph replay misses", registry=self.registry
        )
        self.graph_captures = Gauge(
            "ayaka_graph_captures_total", "Decode-graph bucket captures", registry=self.registry
        )
        self.graph_eager_fallbacks = Gauge(
            "ayaka_graph_eager_fallbacks_total",
            "Decode steps routed back to eager",
            registry=self.registry,
        )
        self.graph_capture_bytes = Gauge(
            "ayaka_graph_capture_bytes", "Bytes reserved by captured graphs", registry=self.registry
        )
        self._graph: dict = {}
        self._scheduler: dict = {}
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

    def record_cancellation(self, reason: str) -> None:
        """Count one terminal cancellation with a bounded reason label."""
        self.cancellations.labels(reason).inc()

    def observe_client_ttft(self, ingress_ns: int) -> None:
        """First socket write for a request whose ingress boundary is known."""
        if ingress_ns <= 0:
            return
        elapsed = self._clock() - ingress_ns
        if elapsed >= 0:
            self.client_ttft.observe(elapsed / 1e9)

    def update_scheduler_stats(self, stats) -> None:
        """Publish cumulative scheduler counters; never mutates scheduling."""
        self.preemptions.set(int(getattr(stats, "preemptions", 0)))
        self.recomputed_tokens.set(int(getattr(stats, "recomputed_tokens", 0)))
        self.wasted_compute_tokens.set(int(getattr(stats, "wasted_compute_tokens", 0)))
        self._scheduler = {
            "preemptions": int(getattr(stats, "preemptions", 0)),
            "recomputed_tokens": int(getattr(stats, "recomputed_tokens", 0)),
            "wasted_compute_tokens": int(getattr(stats, "wasted_compute_tokens", 0)),
            "transient_prepare_failures": int(getattr(stats, "transient_prepare_failures", 0)),
            "admission_rejections": int(getattr(stats, "admission_rejections", 0)),
            "bypasses": int(getattr(stats, "bypasses", 0)),
        }

    def update_graph_stats(self, stats) -> None:
        """Publish a runner graph snapshot; the runner keeps the authority."""
        if stats is None:
            self._graph = {}
            return
        self.graph_hits.set(int(stats.hits))
        self.graph_misses.set(int(stats.misses))
        self.graph_captures.set(int(stats.captures))
        self.graph_eager_fallbacks.set(int(stats.eager_fallbacks))
        self.graph_capture_bytes.set(int(stats.capture_bytes))
        self._graph = stats.as_dict()

    def graph_snapshot(self) -> dict:
        """Latest runner graph view, or empty while graphs are disabled."""
        return dict(self._graph)

    def scheduler_snapshot(self) -> dict:
        """Latest cumulative scheduler counters."""
        return dict(self._scheduler)

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
        if record.status not in _GOODPUT_EXCLUDED_STATUSES:
            self.goodput.inc()
        if record.prefix_hit:
            self.prefix_hits.inc()
        else:
            self.prefix_misses.inc()
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

        Only admission-stream rejections (ingress/admitted/scheduler/deadline/
        invalid/engine/withdrawn) are part of the submitted equation; frontend
        refusals (auth, body size, concurrency, control) happen before
        ``submit`` and are reported separately.  Counters and terminals come
        from the same owner-thread settlement stream, so a passed check is an
        observed equation, not an estimate.
        """
        submitted = int(self.conservation_submitted._value.get())
        admitted = int(self.conservation_admitted._value.get())
        rejection_counts = self.rejection_counts()
        total_rejected = int(sum(rejection_counts.values()))
        frontend_rejected = int(
            sum(rejection_counts.get(reason, 0.0) for reason in _FRONTEND_REJECTION_REASONS)
        )
        stream_rejected = total_rejected - frontend_rejected
        status_counts = self._terminal_counts()
        completed = sum(status_counts.get(status, 0) for status in _COMPLETED_STATUSES)
        cancelled = sum(status_counts.get(status, 0) for status in _CANCELLED_STATUSES)
        timed_out = sum(status_counts.get(status, 0) for status in _TIMED_OUT_STATUSES)
        failed = status_counts.get("error", 0)
        classified = completed + cancelled + timed_out + failed
        failed += sum(status_counts.values()) - classified
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

    def _terminal_counts(self) -> dict[str, int]:
        """Settled terminal counts by status, read from the metric registry."""
        counts: dict[str, int] = {}
        for metric in self.requests.collect():
            for sample in metric.samples:
                if sample.name == "ayaka_requests_total":
                    counts[sample.labels["status"]] = int(sample.value)
        return counts

    @staticmethod
    def _percentile_summary(records: list[UsageRecord]) -> dict[str, float | int]:
        """Latency percentiles over one record slice; unmeasured keys are absent."""
        if not records:
            return {"sample_count": 0}
        groups = (
            (
                "queue_latency",
                [r.queue_latency_ns for r in records if r.queue_latency_ns is not None],
            ),
            (
                "tokenizer",
                [r.tokenizer_latency_ns for r in records if r.tokenizer_latency_ns is not None],
            ),
            (
                "service_ttft",
                [r.service_ttft_ns for r in records if r.service_ttft_ns is not None],
            ),
            ("model_ttft", [r.model_ttft_ns for r in records if r.model_ttft_ns is not None]),
            ("client_ttft", [r.client_ttft_ns for r in records if r.client_ttft_ns is not None]),
            ("itl", [r.itl_p50_ns for r in records if r.itl_p50_ns]),
            ("tpot", [r.tpot_ns for r in records if r.tpot_ns is not None]),
            ("e2e", [r.duration_seconds * 1e9 for r in records]),
        )
        summary: dict[str, float | int] = {"sample_count": len(records)}
        for name, values in groups:
            if not values:
                continue
            stats = percentiles([float(value) for value in values])
            summary[f"{name}_p50_ms"] = stats["p50"] / 1e6
            summary[f"{name}_p95_ms"] = stats["p95"] / 1e6
            summary[f"{name}_p99_ms"] = stats["p99"] / 1e6
        return summary

    @staticmethod
    def _evaluate_targets(summary: dict, targets: SloTargets) -> None:
        """Record PASS/FAIL_PERFORMANCE for measured p95s; never gates serving."""
        outcomes: list[bool] = []
        for name in ("queue_latency", "tokenizer", "service_ttft", "itl", "tpot", "e2e"):
            target_ms = getattr(targets, f"{name}_p95_ms", None)
            value = summary.get(f"{name}_p95_ms")
            if target_ms is None or value is None:
                continue
            meets = float(value) <= float(target_ms)
            summary[f"{name}_meets_target"] = meets
            outcomes.append(meets)
        summary["performance_status"] = (
            "PASS"
            if outcomes and all(outcomes)
            else "FAIL_PERFORMANCE"
            if outcomes
            else "NO_TARGET"
        )

    def slo_summary(
        self,
        *,
        skip_records: int = 0,
        targets: SloTargets | None = None,
    ) -> dict[str, float | int | bool | str]:
        """p50/p95 latency summary over the bounded recent-request history.

        Every value comes from raw monotonic boundaries in ``UsageRecord``;
        requests without a measured boundary are excluded instead of counted
        as zero.  Buckets are not mixed: TTFT starts at service ingress.
        ``skip_records`` drops the first N settled requests so a warmup or
        capture phase is never averaged into steady-state latency.  ``targets``
        only classifies performance: a breach is never a correctness failure.
        """
        records = list(self._history)[max(0, skip_records) :]
        summary: dict[str, float | int | bool | str] = {}
        for key, value in self._percentile_summary(records).items():
            summary[key] = value
        if not records:
            return summary
        summary["goodput"] = int(self.goodput._value.get())
        hits = int(self.prefix_hits._value.get())
        misses = int(self.prefix_misses._value.get())
        if hits + misses:
            summary["prefix_hit_ratio"] = hits / (hits + misses)
        prompt_total = sum(r.prompt_tokens for r in records)
        completion_total = sum(r.completion_tokens for r in records)
        summary["prompt_tokens"] = prompt_total
        summary["completion_tokens"] = completion_total
        measured = [r for r in records if r.ingress_ns and r.terminal_ns >= r.ingress_ns]
        if measured:
            start = min(r.ingress_ns for r in measured)
            end = max(r.terminal_ns for r in measured)
            span_seconds = (end - start) / 1e9
            if span_seconds > 0:
                summary["prompt_tokens_per_second"] = prompt_total / span_seconds
                summary["generation_tokens_per_second"] = completion_total / span_seconds
        if targets is not None:
            self._evaluate_targets(summary, targets)
        return summary

    def slo_groups(self) -> dict[str, dict[str, dict[str, float | int]]]:
        """Latency percentiles grouped by finite keys: priority, bucket, prefix.

        Grouping is advisory reporting over the same bounded history; it never
        changes scheduling.  Priority is bucketed into high/default/low because
        the raw value is an open integer and metric labels must stay finite.
        """
        records = list(self._history)
        buckets: dict[str, dict[str, list[UsageRecord]]] = {
            "priority": {"high": [], "default": [], "low": []},
            "prompt_bucket": {"small": [], "medium": [], "large": []},
            "prefix": {"hit": [], "miss": []},
        }
        for record in records:
            priority = (
                "high" if record.priority > 0 else "low" if record.priority < 0 else "default"
            )
            buckets["priority"][priority].append(record)
            buckets["prompt_bucket"][record.prompt_bucket].append(record)
            buckets["prefix"]["hit" if record.prefix_hit else "miss"].append(record)
        return {
            kind: {name: self._percentile_summary(group) for name, group in groups.items()}
            for kind, groups in buckets.items()
        }

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
