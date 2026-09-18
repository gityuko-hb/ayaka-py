"""Micro-benchmark helper for comparing kernel implementations.

Provides :func:`bench_kernel` which handles warmup, CUDA synchronization,
and timing via ``triton.testing.do_bench`` when available, falling back to
``torch.cuda.Event`` based measurement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

__all__ = [
    "BenchResult",
    "bench_kernel",
    "compare_kernels",
]


@dataclass(frozen=True, slots=True)
class BenchResult:
    """Result of a micro-benchmark run."""

    label: str
    """Human-readable label for the kernel."""
    ms: float
    """Median execution time in milliseconds."""
    warmup: int
    """Number of warmup iterations used."""
    rep: int
    """Number of measured repetitions."""


def bench_kernel(
    fn: Callable[..., Any],
    *,
    label: str = "kernel",
    warmup: int = 25,
    rep: int = 100,
    device: str = "cuda",
) -> BenchResult:
    """Time a zero-argument callable with proper CUDA synchronization.

    Tries ``triton.testing.do_bench`` first (most accurate for GPU kernels),
    then falls back to ``torch.cuda.Event`` measurement.

    Parameters
    ----------
    fn:
        Zero-argument callable to benchmark.
    label:
        Descriptive name for the kernel.
    warmup:
        Warm-up iterations.
    rep:
        Measured repetitions.
    device:
        ``"cuda"`` or ``"cpu"``.

    Returns
    -------
    BenchResult:
        Contains the median time in milliseconds.
    """
    if device == "cuda":
        ms = _bench_cuda(fn, warmup=warmup, rep=rep)
    else:
        ms = _bench_cpu(fn, warmup=warmup, rep=rep)

    return BenchResult(label=label, ms=round(ms, 4), warmup=warmup, rep=rep)


def compare_kernels(
    kernels: dict[str, Callable[..., Any]],
    *,
    warmup: int = 25,
    rep: int = 100,
    device: str = "cuda",
    baseline: str | None = None,
    use_color: bool | None = None,
) -> str:
    """Benchmark multiple kernel variants and print a comparison table.

    Parameters
    ----------
    kernels:
        Mapping from label to zero-argument callable.
    baseline:
        Label of the kernel to use as the 1.00x baseline.  Defaults to the
        first entry.
    use_color:
        Explicit color override. ``None`` auto-detects from TTY / env.

    Returns
    -------
    str:
        Formatted comparison table.
    """
    from ayaka.utils.logging import should_use_color

    results: list[BenchResult] = []
    for label, fn in kernels.items():
        r = bench_kernel(fn, label=label, warmup=warmup, rep=rep, device=device)
        results.append(r)

    if not results:
        return "(no kernels to compare)"

    color = use_color if use_color is not None else should_use_color()

    base_label = baseline or results[0].label
    base_ms = next((r.ms for r in results if r.label == base_label), results[0].ms)

    w_label = max(len("Kernel"), max(len(r.label) for r in results))
    w_ms = max(len("Time (ms)"), max(len(f"{r.ms:.4f}") for r in results))
    w_speed = len("Speedup")

    hdr_plain = f"{'Kernel':<{w_label}}  {'Time (ms)':>{w_ms}}  {'Speedup':>{w_speed}}"
    sep = "-" * len(hdr_plain)

    if color:
        hdr = (
            f"\033[1m{'Kernel':<{w_label}}\033[0m  "
            f"\033[1m{'Time (ms)':>{w_ms}}\033[0m  "
            f"\033[1m{'Speedup':>{w_speed}}\033[0m"
        )
    else:
        hdr = hdr_plain

    lines = [sep, hdr, sep]
    for r in results:
        speedup = base_ms / r.ms if r.ms > 0 else float("inf")
        marker = " (baseline)" if r.label == base_label else ""
        speedup_str = f"{speedup:.2f}x{marker}"

        if color and r.label != base_label:
            if speedup >= 1.2:
                speedup_str = f"\033[32m{speedup_str}\033[0m"  # Green — faster
            elif speedup < 1.0:
                speedup_str = f"\033[31m{speedup_str}\033[0m"  # Red — regression

        lines.append(f"{r.label:<{w_label}}  {r.ms:>{w_ms}.4f}  {speedup_str}")
    lines.append(sep)
    table = "\n".join(lines)
    print(table)
    return table


def _bench_cuda(fn: Callable[..., Any], *, warmup: int, rep: int) -> float:
    """Benchmark using Triton's do_bench or torch.cuda.Event fallback."""
    try:
        from triton.testing import do_bench

        ms = float(do_bench(fn, warmup=warmup, rep=rep))  # type: ignore[arg-type]
        return ms
    except ImportError:
        logger.debug("triton.testing.do_bench unavailable, using torch.cuda.Event fallback")

    return _bench_cuda_events(fn, warmup=warmup, rep=rep)


def _bench_cuda_events(fn: Callable[..., Any], *, warmup: int, rep: int) -> float:
    """Fallback CUDA timing using torch.cuda.Event."""
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(rep):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    mid = len(times) // 2
    return times[mid] if len(times) % 2 == 1 else (times[mid - 1] + times[mid]) / 2


def _bench_cpu(fn: Callable[..., Any], *, warmup: int, rep: int) -> float:
    """CPU-only timing using time.perf_counter."""
    import time

    for _ in range(warmup):
        fn()

    times: list[float] = []
    for _ in range(rep):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times.sort()
    mid = len(times) // 2
    return times[mid] if len(times) % 2 == 1 else (times[mid - 1] + times[mid]) / 2
