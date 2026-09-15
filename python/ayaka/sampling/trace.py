"""End-to-end stage tracing for sampling execution.

Analogous to `_trace_e2e_sampler` in SGLang, this module provides low-overhead
stage tracing enabled via the `AYAKA_TRACE_SAMPLER_E2E=1` environment variable.
Each executed sampling stage emits a single structured line to stdout:

    AYAKA_TRACE_SAMPLER ws=1 rank=0 local=0 stage=forward_enter rows=8 vocab=32000 ...

Notes:
    The environment variable is evaluated dynamically without caching to allow
    toggling trace output during active debugging sessions without restarting the
    runtime process. When disabled, the runtime overhead is limited to a single
    dictionary lookup (`os.environ.get`), which is negligible relative to kernel
    execution time.
"""

from __future__ import annotations

import os
from typing import Any

__all__ = ["trace_sampler"]


def _trace_enabled() -> bool:
    """Check whether end-to-end sampler tracing is enabled in the environment."""
    return os.environ.get("AYAKA_TRACE_SAMPLER_E2E", "0").lower() in ("1", "true", "yes")


def _rank_prefix() -> str:
    """Construct distributed rank prefix string for formatted trace output."""
    try:
        from ayaka.distributed.env import local_rank, rank, world_size

        return f"ws={world_size()} rank={rank()} local={local_rank()}"
    except Exception:  # pragma: no cover - defensive, prefix only
        return "rank=unknown"


def trace_sampler(stage: str, **fields: Any) -> None:
    """Emit a structured trace log entry for a sampling pipeline stage.

    When `AYAKA_TRACE_SAMPLER_E2E` is enabled, formats and prints the stage name,
    distributed rank information, and key-value fields to stdout with immediate flush.
    Acts as a no-op when tracing is disabled.

    Args:
        stage: Identifier string of the current sampling execution stage.
        **fields: Arbitrary key-value attributes associated with the stage event.
    """
    # Fast exit when tracing is disabled.
    if not _trace_enabled():
        return
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    suffix = f" {details}" if details else ""
    print(f"AYAKA_TRACE_SAMPLER {_rank_prefix()} stage={stage}{suffix}", flush=True)
