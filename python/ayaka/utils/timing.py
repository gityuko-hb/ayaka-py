from __future__ import annotations

import time
from dataclasses import dataclass, field

from ayaka.utils.math_utils import percentiles

MAX_INTER_TOKEN_SAMPLES = 512


@dataclass(frozen=True, slots=True, order=True)
class Timestamp:
    nanos: int = 0

    def as_nanos(self) -> int:
        return self.nanos

    def as_micros_f64(self) -> float:
        return self.nanos / 1_000.0

    def as_mills_f64(self) -> float:
        return self.nanos / 1_000_000.0


class Clock:
    """Monotonic clock shared by request state machines.

    Timestamps are absolute values from one monotonic clock domain so they can
    safely be compared across requests and with an explicitly supplied
    ``arrival_ns``.
    """

    __slots__ = ()

    def now(self) -> Timestamp:
        return Timestamp(time.monotonic_ns())


@dataclass(slots=True)
class RequestTiming:
    arrival_ns: int = 0
    first_scheduled_ns: int = 0
    first_token_ns: int = 0
    last_token_ns: int = 0
    finished_ns: int = 0
    num_output_tokens: int = 0
    inter_token_intervals_ns: list[int] = field(default_factory=list)
    """Bounded per-token publication gaps, in arrival order.

    The first published token establishes ``first_token_ns``; later tokens
    append ``now - previous`` while the sample cap lasts. The buffer is an
    inter-token-latency distribution for benchmarks and never feeds scheduling.
    """

    def queue_time_ns(self) -> int | None:
        if self.first_scheduled_ns >= self.arrival_ns and self.first_scheduled_ns != 0:
            return self.first_scheduled_ns - self.arrival_ns
        return None

    def ttft_ns(self) -> int | None:
        if self.first_token_ns >= self.arrival_ns and self.first_token_ns != 0:
            return self.first_token_ns - self.arrival_ns
        return None

    def tpot_ns(self) -> int | None:
        if self.num_output_tokens >= 2 and self.last_token_ns > self.first_token_ns:
            return (self.last_token_ns - self.first_token_ns) // (self.num_output_tokens - 1)
        return None

    def e2e_ns(self) -> int | None:
        end_ns = self.finished_ns or self.last_token_ns
        if end_ns >= self.arrival_ns and end_ns != 0:
            return end_ns - self.arrival_ns
        return None

    def mark_scheduled(self, now_ns: int) -> None:
        if not self.first_scheduled_ns:
            self.first_scheduled_ns = now_ns

    def inter_token_stats_ns(self) -> dict[str, float]:
        """min/mean/p50/p90/p95/p99/max of the sampled publication gaps."""
        return percentiles([float(value) for value in self.inter_token_intervals_ns])

    def inter_token_percentiles_ns(self) -> tuple[int, int]:
        """Cheap p50/p95 for the settlement path, without a numpy conversion."""
        samples = sorted(self.inter_token_intervals_ns)
        if not samples:
            return (0, 0)
        last = len(samples) - 1

        def pick(quantile: float) -> int:
            return samples[min(last, max(0, int(round(last * quantile))))]

        return (pick(0.5), pick(0.95))

    def inter_token_p95_ns(self) -> float:
        return self.inter_token_stats_ns()["p95"]

    def mark_streamed(self, now_ns: int) -> None:
        if not self.first_token_ns:
            self.first_token_ns = now_ns
            self.last_token_ns = now_ns
            return
        if now_ns < self.last_token_ns:
            return
        if len(self.inter_token_intervals_ns) < MAX_INTER_TOKEN_SAMPLES:
            self.inter_token_intervals_ns.append(now_ns - self.last_token_ns)
        self.last_token_ns = now_ns

    def mark_finished(self, now_ns: int) -> None:
        self.finished_ns = now_ns
