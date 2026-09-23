"""Engine-level metrics: vLLM-style iteration stats and stat loggers.

Two delivery targets, one source of truth — :class:`~ayaka.metrics.IterationStats`
built per engine step from the scheduler report plus engine-loop counters:

- :class:`LoggingStatLogger` aggregates between intervals and emits a compact
  throughput line (vLLM's ``do_log_stats`` analog);
- :class:`PrometheusStatLogger` exports bounded-label gauges/counters/
  histograms on its own registry (request-id labels are forbidden, matching
  the ``obs.py`` policy).

The serving layer keeps its own request-centric stats (TTFT/TPOT); this module
covers the engine loop itself.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["IterationStats"]


@dataclass(frozen=True, slots=True)
class IterationStats:
    """One engine step's observations, ready for logger fan-out.

    Counts are per step (not cumulative); loggers aggregate. ``step_id`` is the
    scheduler report identity (-1 when the step produced no scheduler report).
    """

    step_id: int
    num_running: int = 0
    num_waiting: int = 0
    num_preempted: int = 0
    num_scheduled_tokens: int = 0
    num_new_tokens: int = 0
    num_finished: int = 0
    num_aborted: int = 0
    num_failed: int = 0
    num_settled: int = 0
    step_seconds: float = 0.0
