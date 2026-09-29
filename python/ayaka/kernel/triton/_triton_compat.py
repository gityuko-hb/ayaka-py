"""Triton-specific launch adapters.

Keep these optional-dependency helpers out of :mod:`._host`, which provides
Torch-only contracts for CPU installs and reference paths.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import triton

_INTERPRET = os.environ.get("TRITON_INTERPRET", "0") == "1"


class _FirstConfigKernel:
    """Stand-in for an autotuned kernel when interpreting.

    Keeps the decorated function launchable but ignores the config search,
    running ``configs[0]`` instead.
    """

    def __init__(self, fn: Any, config: triton.Config) -> None:
        self.fn = fn
        self.config = config

    def __getitem__(self, grid: Any) -> Callable[..., Any]:
        def launch(*args: Any, **kwargs: Any) -> Any:
            merged = dict(self.config.kwargs)
            merged.update(kwargs)
            return self.fn[grid](*args, **merged)

        return launch


def autotune(configs: list[triton.Config], key: list[str], **kwargs: Any) -> Any:
    """Wrap Triton's autotune decorator with a deterministic interpreter path."""
    if _INTERPRET:
        if not configs:
            raise ValueError("configs must contain at least one entry in interpreter mode")
        return lambda fn: _FirstConfigKernel(fn, configs[0])
    return triton.autotune(configs=configs, key=key, **kwargs)
