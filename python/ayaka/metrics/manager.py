"""Fan-out hub: record every step, log on the interval (vLLM manager analog)."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

from ayaka.metrics.loggers import StatLogger
from ayaka.metrics.stats import IterationStats

__all__ = ["StatLoggerManager"]


class StatLoggerManager:
    """Feed every stat logger on each step; trigger interval logging.

    ``record`` is cheap and always called by the engine; ``maybe_log`` fires
    the loggers' ``log`` only when ``log_interval`` of wall clock has passed
    since the last emission. A ``clock`` is injectable for tests.
    """

    def __init__(
        self,
        *,
        loggers: Sequence[StatLogger] = (),
        log_interval: float = 10.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if log_interval <= 0:
            raise ValueError("log_interval must be positive")
        self._loggers = tuple(loggers)
        self._log_interval = log_interval
        self._clock = clock or time.monotonic
        self._last_log = self._clock()

    @property
    def loggers(self) -> tuple[StatLogger, ...]:
        return self._loggers

    def record(self, stats: IterationStats) -> None:
        for logger in self._loggers:
            logger.record(stats)

    def maybe_log(self) -> None:
        now = self._clock()
        if now - self._last_log < self._log_interval:
            return
        self.log()

    def log(self) -> None:
        for logger in self._loggers:
            logger.log()
        self._last_log = self._clock()
