"""Kernel-level GPU profiler for Ayaka.

Wraps ``torch.profiler`` to produce per-kernel breakdown tables showing call
count, total GPU time, average latency, and percentage of total GPU time.
Outputs Chrome/Perfetto-compatible traces and CSV for further analysis.
"""

from __future__ import annotations

import contextlib
import csv
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ayaka.utils.logging import should_use_color

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

logger = logging.getLogger(__name__)

__all__ = [
    "KernelProfiler",
    "KernelStats",
    "ProfileResult",
    "profile_region",
]

# ANSI escape sequences (same values as ayaka.utils.logging._COLORS)
_RED = "\033[31m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _resolve_color(use_color: bool | None) -> bool:
    """Decide whether to emit ANSI escapes, respecting NO_COLOR/FORCE_COLOR."""
    if use_color is not None:
        return use_color
    return should_use_color()


def _color_for_pct(pct: float) -> str:
    """Return an ANSI color code based on GPU % threshold.

    - >30 %  → Red   (bottleneck — optimize first)
    - 10–30% → Yellow (notable)
    - <10 %  → Green  (acceptable)
    """
    if pct > 30:
        return _RED
    if pct >= 10:
        return _YELLOW
    return _GREEN


@dataclass(frozen=True, slots=True)
class KernelStats:
    """Profiling statistics for a single CUDA kernel."""

    name: str
    """Demangled kernel name."""
    calls: int
    """Number of invocations."""
    total_us: float
    """Total self CUDA time in microseconds."""
    avg_us: float
    """Average self CUDA time per call in microseconds."""
    min_us: float
    """Minimum self CUDA time in microseconds."""
    max_us: float
    """Maximum self CUDA time in microseconds."""
    pct: float
    """Percentage of total GPU time across all kernels."""


@dataclass(slots=True)
class ProfileResult:
    """Container for a profiling session's results."""

    stats: list[KernelStats] = field(default_factory=list)
    """Per-kernel statistics, sorted by total GPU time descending."""
    total_gpu_us: float = 0.0
    """Sum of self CUDA time across all kernels."""
    wall_time_ms: float = 0.0
    """Wall-clock time of the profiled region in milliseconds."""
    trace_path: str | None = None
    """Path to exported Chrome/Perfetto trace, if any."""

    def table(self, *, row_limit: int = 30, use_color: bool | None = None) -> str:
        """Format a human-readable breakdown table with optional ANSI color."""
        return _format_table(self.stats, row_limit=row_limit, use_color=use_color)

    def to_csv(self, path: str | Path) -> None:
        """Write per-kernel stats to a CSV file."""
        _export_csv(self.stats, path)

    def to_dicts(self) -> list[dict[str, Any]]:
        """Serialize stats to a list of plain dicts."""
        return [asdict(s) for s in self.stats]


def _parse_profiler_events(prof: Any) -> list[KernelStats]:
    """Extract per-kernel stats from ``torch.profiler`` key averages.

    Reads ``self_cuda_time_total`` (microseconds) from each event that has
    non-zero CUDA time, groups by kernel name, and computes percentages.
    """
    events = prof.key_averages()
    raw: list[tuple[str, int, float]] = []
    for evt in events:
        cuda_us: float = getattr(evt, "self_cuda_time_total", 0.0)
        if cuda_us <= 0:
            continue
        name: str = evt.key or "(unknown)"
        count: int = getattr(evt, "count", 1)
        raw.append((name, count, cuda_us))

    if not raw:
        return []

    total_us = sum(r[2] for r in raw)
    if total_us <= 0:
        return []

    # Build stats with min/max approximated as avg (torch.profiler key_averages
    # only exposes totals; per-invocation min/max would require event-level
    # iteration which is expensive).  The values are still useful because the
    # primary purpose is the percentage breakdown.
    stats: list[KernelStats] = []
    for name, count, cuda_us in raw:
        avg = cuda_us / max(count, 1)
        stats.append(
            KernelStats(
                name=name,
                calls=count,
                total_us=round(cuda_us, 1),
                avg_us=round(avg, 1),
                min_us=round(avg, 1),
                max_us=round(avg, 1),
                pct=round(cuda_us / total_us * 100, 2),
            )
        )

    stats.sort(key=lambda s: s.total_us, reverse=True)
    return stats


def _format_table(
    stats: list[KernelStats], *, row_limit: int = 30, use_color: bool | None = None
) -> str:
    """Render a fixed-width text table of kernel stats with optional ANSI color.

    Color thresholds for GPU %:
    - Red (>30%): bottleneck — optimize first
    - Yellow (10–30%): notable
    - Green (<10%): acceptable
    """
    if not stats:
        return "(no CUDA kernels recorded)"

    color = _resolve_color(use_color)
    rows = stats[:row_limit]

    # Column widths (measured without ANSI escapes)
    w_name = max(len("Kernel"), max(len(_trunc(s.name, 48)) for s in rows))
    w_calls = max(len("Calls"), max(len(f"{s.calls:,}") for s in rows))
    w_total = max(len("GPU Total µs"), max(len(f"{s.total_us:,.1f}") for s in rows))
    w_avg = max(len("Avg µs"), max(len(f"{s.avg_us:,.1f}") for s in rows))
    w_pct = max(len("GPU %"), max(len(f"{s.pct:.1f}%") for s in rows))

    # Header
    hdr_plain = (
        f"{'Kernel':<{w_name}}  "
        f"{'Calls':>{w_calls}}  "
        f"{'GPU Total µs':>{w_total}}  "
        f"{'Avg µs':>{w_avg}}  "
        f"{'GPU %':>{w_pct}}"
    )
    sep = "-" * len(hdr_plain)

    if color:
        hdr = (
            f"{_BOLD}{'Kernel':<{w_name}}{_RESET}  "
            f"{_BOLD}{'Calls':>{w_calls}}{_RESET}  "
            f"{_BOLD}{'GPU Total µs':>{w_total}}{_RESET}  "
            f"{_BOLD}{'Avg µs':>{w_avg}}{_RESET}  "
            f"{_BOLD}{'GPU %':>{w_pct}}{_RESET}"
        )
    else:
        hdr = hdr_plain

    lines = [sep, hdr, sep]

    for s in rows:
        pct_str = f"{s.pct:.1f}%"
        if color:
            pct_color = _color_for_pct(s.pct)
            pct_cell = f"{pct_color}{pct_str:>{w_pct}}{_RESET}"
            name_cell = f"{pct_color}{_trunc(s.name, 48):<{w_name}}{_RESET}"
        else:
            pct_cell = f"{pct_str:>{w_pct}}"
            name_cell = f"{_trunc(s.name, 48):<{w_name}}"

        lines.append(
            f"{name_cell}  "
            f"{s.calls:>{w_calls},}  "
            f"{s.total_us:>{w_total},.1f}  "
            f"{s.avg_us:>{w_avg},.1f}  "
            f"{pct_cell}"
        )

    lines.append(sep)

    total_us = sum(s.total_us for s in stats)
    summary = f"Total GPU time: {total_us:,.1f} µs across {len(stats)} kernels"
    if color:
        summary = f"{_CYAN}{summary}{_RESET}"
    lines.append(summary)

    if len(stats) > row_limit:
        lines.append(f"(showing top {row_limit} of {len(stats)})")
    return "\n".join(lines)


def _trunc(s: str, maxlen: int) -> str:
    return s if len(s) <= maxlen else s[: maxlen - 3] + "..."


def _export_csv(stats: list[KernelStats], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["kernel", "calls", "total_us", "avg_us", "min_us", "max_us", "pct"])
        for s in stats:
            writer.writerow([s.name, s.calls, s.total_us, s.avg_us, s.min_us, s.max_us, s.pct])


class KernelProfiler:
    """Profile CUDA kernels launched by an arbitrary callable.

    Example::

        profiler = KernelProfiler(device="cuda")
        result = profiler.profile(lambda: my_kernel(q, k, v), warmup=5, repeat=10)
        print(result.table())
        result.to_csv("kernel_stats.csv")
    """

    def __init__(self, *, device: str = "cuda") -> None:
        self.device = device

    def profile(
        self,
        fn: Callable[..., Any],
        *,
        warmup: int = 5,
        repeat: int = 10,
        export_trace: str | Path | None = None,
    ) -> ProfileResult:
        """Run *fn* under ``torch.profiler`` and return per-kernel breakdown.

        Parameters
        ----------
        fn:
            Zero-argument callable to profile.
        warmup:
            Number of warm-up iterations (not profiled).
        repeat:
            Number of measured iterations.
        export_trace:
            If set, export a Chrome/Perfetto JSON trace to this path.
        """
        import torch
        from torch.profiler import ProfilerActivity, profile

        # Warm up
        for _ in range(warmup):
            fn()
        if self.device == "cuda":
            torch.cuda.synchronize()

        activities = [ProfilerActivity.CPU]
        if self.device == "cuda":
            activities.append(ProfilerActivity.CUDA)

        wall_start = time.perf_counter()
        with profile(activities=activities, record_shapes=True) as prof:
            for _ in range(repeat):
                fn()
            if self.device == "cuda":
                torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - wall_start) * 1000

        stats = _parse_profiler_events(prof)
        total_us = sum(s.total_us for s in stats)

        trace_path: str | None = None
        if export_trace is not None:
            p = Path(export_trace)
            p.parent.mkdir(parents=True, exist_ok=True)
            prof.export_chrome_trace(str(p))
            trace_path = str(p)
            logger.info("Trace exported to %s", trace_path)

        return ProfileResult(
            stats=stats,
            total_gpu_us=total_us,
            wall_time_ms=round(wall_ms, 3),
            trace_path=trace_path,
        )


@contextlib.contextmanager
def profile_region(
    label: str = "region",
    *,
    device: str = "cuda",
    export_trace: str | Path | None = None,
    print_table: bool = True,
    row_limit: int = 30,
) -> Generator[ProfileResult]:
    """Context manager that profiles all CUDA kernels in its body.

    The yielded :class:`ProfileResult` is populated on exit.  If
    *print_table* is ``True`` the breakdown table is printed automatically.

    Example::

        with profile_region("attention", export_trace="attn.json") as result:
            for _ in range(10):
                paged_attention(q, k, v, ...)
                torch.cuda.synchronize()
        # result.stats is now populated
    """
    import torch
    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)

    result = ProfileResult()

    wall_start = time.perf_counter()
    with profile(activities=activities, record_shapes=True) as prof:
        yield result

    if device == "cuda":
        torch.cuda.synchronize()

    wall_ms = (time.perf_counter() - wall_start) * 1000

    stats = _parse_profiler_events(prof)
    total_us = sum(s.total_us for s in stats)

    result.stats = stats
    result.total_gpu_us = total_us
    result.wall_time_ms = round(wall_ms, 3)

    if export_trace is not None:
        p = Path(export_trace)
        p.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(p))
        result.trace_path = str(p)
        logger.info("Trace exported to %s", result.trace_path)

    if print_table:
        color = _resolve_color(None)
        if color:
            header = f"\n{_BOLD}{_CYAN}=== Kernel Breakdown: {label} ==={_RESET}"
            wall = f"{_CYAN}Wall time: {wall_ms:,.1f} ms{_RESET}\n"
        else:
            header = f"\n=== Kernel Breakdown: {label} ==="
            wall = f"Wall time: {wall_ms:,.1f} ms\n"
        print(header)
        print(result.table(row_limit=row_limit))
        print(wall)
