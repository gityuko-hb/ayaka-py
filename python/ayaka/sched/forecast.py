"""Shadow output-length forecast, separated from allocation authority (research R2).

SGLang's ``new_token_ratio_tracker`` is one global ratio with linear decay; it
describes no per-request distribution. This forecaster keeps a bounded
histogram of completed output lengths per prompt-length bucket and answers
quantile questions ("90 % of requests like this finish within N more tokens"),
bounded by the request's remaining ``max_tokens``.

It is a **shadow** component: nothing in the scheduler or allocator reads it.
The safety condition stays the physical reservation acquired for the step about
to launch; a forecast only describes how much KV a request is likely to need,
and its calibration (did the actual length fall under the forecast?) is
reported so a later policy can be judged on data. Requests that hit
``max_tokens`` are right-censored: their true length is unknown, so they count
toward the histogram at the cap and are reported separately in calibration.
"""

from __future__ import annotations

import bisect
import threading
from dataclasses import dataclass, field

from ayaka.utils.math_utils import div_ceil
from ayaka.utils.validation import require_int

__all__ = ["ForecastCalibration", "OutputLengthForecaster"]

#: Prompt-length bucket edges (tokens); a prompt of length L uses the first edge >= L.
_PROMPT_BUCKETS = (16, 64, 256, 1024, 4096, 16384, 65536)
#: Output-length histogram bin upper edges (tokens), roughly geometric.
_OUTPUT_BINS = (
    1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024,
    1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 32768, 65536, 131072,
)  # fmt: skip


@dataclass(slots=True)
class _Histogram:
    counts: list[int] = field(default_factory=lambda: [0] * len(_OUTPUT_BINS))
    total: int = 0

    def add(self, length: int) -> None:
        index = min(bisect.bisect_left(_OUTPUT_BINS, length), len(_OUTPUT_BINS) - 1)
        self.counts[index] += 1
        self.total += 1

    def quantile(self, q: float) -> int:
        """Upper edge of the bin holding the q-quantile (a conservative estimate)."""
        target = q * self.total
        running = 0
        for edge, count in zip(_OUTPUT_BINS, self.counts, strict=True):
            running += count
            if running >= target:
                return edge
        return _OUTPUT_BINS[-1]


@dataclass(frozen=True, slots=True)
class ForecastCalibration:
    """How forecasts made at admission compared with what happened."""

    observed: int
    #: Uncensored requests whose actual output stayed within the forecast.
    covered: int
    #: Uncensored requests (finished by stop/EOS before ``max_tokens``).
    uncensored: int
    #: Requests that hit ``max_tokens``; their true length is unknown.
    censored: int
    #: Sum over uncensored requests of ``forecast - actual`` (over-reservation).
    slack_tokens: int

    @property
    def coverage(self) -> float | None:
        return None if not self.uncensored else self.covered / self.uncensored

    def as_dict(self) -> dict[str, object]:
        return {
            "observed": self.observed,
            "covered": self.covered,
            "uncensored": self.uncensored,
            "censored": self.censored,
            "coverage": self.coverage,
            "slack_tokens": self.slack_tokens,
        }


class OutputLengthForecaster:
    """Quantile forecasts of output length per prompt bucket; thread-safe, bounded.

    Args:
        quantile: The forecast quantile in (0, 1).
        min_samples: Observations a bucket needs before it is trusted; below
            that the forecast is the conservative remaining budget.
    """

    def __init__(self, *, quantile: float = 0.9, min_samples: int = 16) -> None:
        if not 0.0 < quantile < 1.0:
            raise ValueError("quantile must be in (0, 1)")
        require_int(min_samples, "min_samples", minimum=1)
        self.quantile = quantile
        self.min_samples = min_samples
        self._buckets = [_Histogram() for _ in range(len(_PROMPT_BUCKETS) + 1)]
        self._lock = threading.Lock()
        self._observed = self._covered = self._uncensored = self._censored = 0
        self._slack = 0

    @staticmethod
    def _bucket(prompt_tokens: int) -> int:
        return bisect.bisect_left(_PROMPT_BUCKETS, prompt_tokens)

    def predict(self, prompt_tokens: int, max_tokens: int, generated: int = 0) -> int:
        """Forecast remaining output tokens, never above the remaining budget."""
        require_int(prompt_tokens, "prompt_tokens")
        require_int(max_tokens, "max_tokens", minimum=1)
        require_int(generated, "generated")
        remaining = max(0, max_tokens - generated)
        with self._lock:
            histogram = self._buckets[self._bucket(prompt_tokens)]
            if histogram.total < self.min_samples:
                return remaining  # no evidence: assume the whole budget
            total_forecast = histogram.quantile(self.quantile)
        return min(remaining, max(0, total_forecast - generated))

    def predicted_pages(
        self,
        prompt_tokens: int,
        max_tokens: int,
        *,
        page_size: int,
        generated: int = 0,
        resident_tokens: int | None = None,
    ) -> int:
        """Pages the request is forecast to add on top of the KV it already has.

        ``resident_tokens`` (default: prompt + generated) is the KV already
        present for the request, computed or shared from the prefix cache (at
        admission: the cached prefix). A shared partial tail page that will need
        a private copy is not counted. A forecast, never a reservation.
        """
        require_int(page_size, "page_size", minimum=1)
        remaining = self.predict(prompt_tokens, max_tokens, generated)
        resident = prompt_tokens + generated if resident_tokens is None else resident_tokens
        require_int(resident, "resident_tokens")
        final = prompt_tokens + generated + remaining
        return max(0, div_ceil(final, page_size) - div_ceil(resident, page_size))

    def observe(
        self, prompt_tokens: int, output_tokens: int, *, censored: bool, forecast: int
    ) -> None:
        """Record one finished request and score the forecast made at admission."""
        require_int(prompt_tokens, "prompt_tokens")
        require_int(output_tokens, "output_tokens")
        require_int(forecast, "forecast")
        with self._lock:
            self._buckets[self._bucket(prompt_tokens)].add(output_tokens)
            self._observed += 1
            if censored:
                self._censored += 1
                return
            self._uncensored += 1
            if output_tokens <= forecast:
                self._covered += 1
                self._slack += forecast - output_tokens

    def calibration(self) -> ForecastCalibration:
        with self._lock:
            return ForecastCalibration(
                self._observed, self._covered, self._uncensored, self._censored, self._slack
            )
