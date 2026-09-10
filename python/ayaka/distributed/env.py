from __future__ import annotations

import os
from functools import lru_cache

__all__ = [
    "describe",
    "is_distributed",
    "is_local_master",
    "is_master",
    "local_rank",
    "local_world_size",
    "rank",
    "reset_cache",
    "world_size",
]


def _parse_int_env(name: str, default: int, *, minimum: int | None = None) -> int:
    """Parse an integer from an environment variable with validation."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        val = int(raw.strip())
    except ValueError:
        raise ValueError(
            f"Environment variable {name}={raw!r} is not a valid integer"
        ) from None
    if minimum is not None and val < minimum:
        raise ValueError(f"Environment variable {name}={val} must be >= {minimum}")
    return val


@lru_cache(maxsize=1)
def rank() -> int:
    """Global rank of this process, parsed from ``RANK``. Defaults to 0.

    Guaranteed to be non-negative.
    """
    return _parse_int_env("RANK", 0, minimum=0)


@lru_cache(maxsize=1)
def local_rank() -> int:
    """Rank within this node, from ``LOCAL_RANK``, falling back to ``RANK``.

    Distinct from `rank` on multi-node jobs, and it is the one that selects
    a device: ``cuda:{local_rank}`` is correct where ``cuda:{rank}`` indexes
    past the end of the node on rank 8 of a two-node job.
    """
    if "LOCAL_RANK" in os.environ:
        return _parse_int_env("LOCAL_RANK", 0, minimum=0)
    return rank()


@lru_cache(maxsize=1)
def world_size() -> int:
    """Total number of processes in the job, from ``WORLD_SIZE``. Defaults to 1.

    Guaranteed to be positive (>= 1).
    """
    return _parse_int_env("WORLD_SIZE", 1, minimum=1)


@lru_cache(maxsize=1)
def local_world_size() -> int:
    """Processes on this node, from ``LOCAL_WORLD_SIZE``.

    Falls back to `world_size`, which is correct for the single-node case
    and wrong for multi-node -- but the launcher sets the variable in
    every multi-node setup that matters, and silently guessing would be
    worse than being wrong only where the launcher is broken.
    """
    if "LOCAL_WORLD_SIZE" in os.environ:
        return _parse_int_env("LOCAL_WORLD_SIZE", 1, minimum=1)
    return world_size()


def is_master() -> bool:
    """True on global rank 0. The rank that logs, serves, and reports metrics."""
    return rank() == 0


def is_local_master() -> bool:
    """True on local rank 0. The rank that should touch node-local state.

    Downloading weights into a shared cache, creating a directory,
    compiling into an on-disk kernel cache -- one process per *node*, not
    one per job, or eight ranks race on the same files.
    """
    return local_rank() == 0


def is_distributed() -> bool:
    """True when more than one process participates in the job."""
    return world_size() > 1


def reset_cache() -> None:
    """Forget cached values.

    Needed in exactly two places: tests, and any child process that
    inherits a parent's cached rank across ``fork`` and must re-read its own.
    """
    rank.cache_clear()
    local_rank.cache_clear()
    world_size.cache_clear()
    local_world_size.cache_clear()


def describe() -> str:
    """Short identity string for log prefixes, diagnostics, and process titles.

    Examples:
        - Single process: ``"single"``
        - Single node (8 ranks, rank 0): ``"rank0/8"``
        - Multi-node (16 ranks, rank 8 on node 2, local rank 0): ``"rank8/16 local0"``
    """
    if not is_distributed():
        return "single"
    if local_world_size() == world_size():
        return f"rank{rank()}/{world_size()}"
    return f"rank{rank()}/{world_size()} local{local_rank()}"
