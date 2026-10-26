"""Truthful extrema in the kernel profiler (research finding A-07).

``torch.profiler`` key averages expose only per-kernel totals.  The parser used
to fill ``min_us`` / ``max_us`` with the average, which reads as a measured
latency tail.  The aggregate path now reports those fields as ``None`` and the
exports leave them empty; totals, averages and the percentage breakdown, which
*are* derived from data, are unchanged.

CPU-only: the profiler object is a stand-in with the ``key_averages()`` shape the
parser consumes, so no CUDA or ``torch.profiler`` run is involved.
"""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

from ayaka.kernel.profiler.kernel_profiler import (
    KernelStats,
    ProfileResult,
    _parse_profiler_events,
)


def _prof(*rows: tuple[str, int, float]) -> SimpleNamespace:
    events = [
        SimpleNamespace(key=name, count=count, self_cuda_time_total=total)
        for name, count, total in rows
    ]
    return SimpleNamespace(key_averages=lambda: events)


def test_aggregate_path_does_not_fabricate_extrema() -> None:
    stats = _parse_profiler_events(_prof(("gemm", 4, 400.0), ("norm", 2, 100.0)))
    by_name = {s.name: s for s in stats}
    assert by_name["gemm"].min_us is None
    assert by_name["gemm"].max_us is None
    assert by_name["norm"].min_us is None
    assert by_name["norm"].max_us is None


def test_totals_averages_and_percentages_are_still_derived_from_data() -> None:
    stats = _parse_profiler_events(_prof(("gemm", 4, 400.0), ("norm", 2, 100.0)))
    assert [s.name for s in stats] == ["gemm", "norm"]  # sorted by total time
    gemm, norm = stats
    assert (gemm.calls, gemm.total_us, gemm.avg_us, gemm.pct) == (4, 400.0, 100.0, 80.0)
    assert (norm.calls, norm.total_us, norm.avg_us, norm.pct) == (2, 100.0, 50.0, 20.0)


def test_events_without_cuda_time_are_skipped_and_empty_input_is_empty() -> None:
    assert _parse_profiler_events(_prof(("cpu_only", 3, 0.0))) == []
    assert _parse_profiler_events(_prof()) == []


def test_csv_keeps_its_columns_and_leaves_unmeasured_extrema_empty(tmp_path: Path) -> None:
    result = ProfileResult(stats=_parse_profiler_events(_prof(("gemm", 4, 400.0))))
    target = tmp_path / "nested" / "stats.csv"
    result.to_csv(target)
    rows = list(csv.reader(target.read_text(encoding="utf-8").splitlines()))
    assert rows[0] == ["kernel", "calls", "total_us", "avg_us", "min_us", "max_us", "pct"]
    assert rows[1] == ["gemm", "4", "400.0", "100.0", "", "", "100.0"]


def test_csv_still_writes_extrema_when_a_caller_supplies_measured_values(
    tmp_path: Path,
) -> None:
    measured = KernelStats("k", 3, 30.0, 10.0, 8.5, 12.25, 100.0)
    target = tmp_path / "measured.csv"
    ProfileResult(stats=[measured]).to_csv(target)
    rows = list(csv.reader(target.read_text(encoding="utf-8").splitlines()))
    assert rows[1][4:6] == ["8.5", "12.25"]


def test_to_dicts_exposes_none_not_a_number() -> None:
    result = ProfileResult(stats=_parse_profiler_events(_prof(("gemm", 4, 400.0))))
    (row,) = result.to_dicts()
    assert row["min_us"] is None and row["max_us"] is None
    assert row["avg_us"] == 100.0


def test_table_never_prints_extrema() -> None:
    result = ProfileResult(stats=_parse_profiler_events(_prof(("gemm", 4, 400.0))))
    text = result.table(use_color=False)
    assert "Avg" in text
    assert "Min" not in text and "Max" not in text and "None" not in text
