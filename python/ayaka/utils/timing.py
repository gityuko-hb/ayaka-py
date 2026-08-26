from __future__ import annotations

import time
from dataclasses import dataclass


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
    __slots__ = ("_start_ns",)

    def __init__(self) -> None:
        self._start_ns = time.perf_counter_ns()

    def now(self) -> Timestamp:
        return Timestamp(time.perf_counter_ns() - self._start_ns)


@dataclass(slots=True)
class RequestTiming:
    arrival_ns: int = 0
    first_scheduled_ns: int = 0
    first_token_ns: int = 0
    last_token_ns: int = 0
    num_output_tokens: int = 0

    def ttft_ns(self) -> int | None:
        if self.first_token_ns >= self.arrival_ns and self.first_token_ns != 0:
            return self.first_token_ns - self.arrival_ns
        return None

    def tpot_ns(self) -> int | None:
        if self.num_output_tokens >= 2 and self.last_token_ns > self.first_token_ns:
            return (self.last_token_ns - self.first_token_ns) // (
                self.num_output_tokens - 1
            )
        return None

    def e2e_ns(self) -> int | None:
        if self.last_token_ns >= self.arrival_ns and self.last_token_ns != 0:
            return self.last_token_ns - self.arrival_ns
        return None
