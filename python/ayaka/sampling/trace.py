"""E2E stage tracing cho sampler (analog của ``_trace_e2e_sampler`` SGLang).

Bật bằng ``AYAKA_TRACE_SAMPLER_E2E=1``; mỗi stage in MỘT dòng ra stdout::

    AYAKA_TRACE_SAMPLER ws=1 rank=0 local=0 stage=forward_enter rows=8 vocab=32000 ...

Env đọc DYNAMIC (không cache) — theo idiom ``_force_reference_env`` bên
``kernel/ops.py`` — vì trace thường bật/tắt giữa chừng khi debug. Khi tắt,
chi phí là một ``os.environ.get`` — không đáng kể so với forward pass.
"""

from __future__ import annotations

import os
from typing import Any

__all__ = ["trace_sampler"]


def _trace_enabled() -> bool:
    return os.environ.get("AYAKA_TRACE_SAMPLER_E2E", "0").lower() in ("1", "true", "yes")


def _rank_prefix() -> str:
    try:
        from ayaka.distributed.env import local_rank, rank, world_size

        return f"ws={world_size()} rank={rank()} local={local_rank()}"
    except Exception:  # pragma: no cover - defensive, prefix only
        return "rank=unknown"


def trace_sampler(stage: str, **fields: Any) -> None:
    """In một dòng trace cho stage hiện tại; no-op khi env tắt."""
    if not _trace_enabled():
        return
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    suffix = f" {details}" if details else ""
    print(f"AYAKA_TRACE_SAMPLER {_rank_prefix()} stage={stage}{suffix}", flush=True)
