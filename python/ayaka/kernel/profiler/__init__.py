"""Ayaka kernel profiler — per-kernel GPU time breakdown and micro-benchmarks.

Public API
----------
.. autoclass:: KernelProfiler
.. autoclass:: KernelStats
.. autoclass:: ProfileResult
.. autofunction:: profile_region
.. autofunction:: bench_kernel
.. autofunction:: compare_kernels
"""

from ayaka.kernel.profiler.bench import BenchResult, bench_kernel, compare_kernels
from ayaka.kernel.profiler.kernel_profiler import (
    KernelProfiler,
    KernelStats,
    ProfileResult,
    profile_region,
)

__all__ = [
    "BenchResult",
    "KernelProfiler",
    "KernelStats",
    "ProfileResult",
    "bench_kernel",
    "compare_kernels",
    "profile_region",
]
