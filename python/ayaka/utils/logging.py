from __future__ import annotations

import json
import logging
import time
from typing import Any

from ayaka.distributed.env import describe, is_distributed, is_master, rank

__all__ = [
    "JSONFormatter",
    "TextFormatter",
    "log_master",
    "log_once",
    "reset_once_state",
]


class _OnceFilter:
    """A logging filter that only allows a message to be emitted once per process.

    Bounded by capacity to prevent unbounded memory growth in long-running
    inference engines.
    """

    __slots__ = ("_capacity", "_overflow", "_seen")

    def __init__(self, capacity: int = 4096) -> None:
        self._seen: set[Any] = set()
        self._capacity = capacity
        self._overflow = 0

    def should_emit(self, key: Any) -> bool:
        if key in self._seen:
            return False
        if len(self._seen) >= self._capacity:
            self._overflow += 1
            return True
        self._seen.add(key)
        return True

    @property
    def suppressed_overflow(self) -> int:
        return self._overflow

    def reset(self) -> None:
        self._seen.clear()
        self._overflow = 0


_once = _OnceFilter()


def log_once(
    logger: logging.Logger,
    level: int,
    message: str,
    *args: Any,
    master_only: bool = True,
    **kwargs: Any,
) -> None:
    """Emit `message` at most once per process.

    Prevents repetitive warnings or logs from spamming disk I/O and stalling
    high-throughput token decode loops.
    """
    key = (logger.name, level, message, args)
    if master_only and not is_master():
        return
    if _once.should_emit(key):
        logger.log(level, message, *args, **kwargs)


def log_master(
    logger: logging.Logger,
    level: int,
    message: str,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Emit log only on global rank 0.

    Suppresses duplicate lines across all tensor/pipeline parallel ranks.
    """
    if is_master():
        logger.log(level, message, *args, **kwargs)


def reset_once_state() -> None:
    """Clear the deduplication state. For tests."""
    _once.reset()


class JSONFormatter(logging.Formatter):
    """A structured JSON log formatter.

    Outputs a single JSON line per log record, containing:
    - ts: ISO 8601 timestamp with millisecond precision
    - level: Log level name (INFO, WARNING, etc.)
    - logger: Logger name
    - rank: Process global rank from ``ayaka.distributed.rank()``
    - message: The formatted message
    - exception: Formatted traceback (if present)
    - Custom extra metadata attributes starting with ``ayaka_`` (stripped prefix).
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created))
            + f".{int(record.msecs):03d}",
            "level": record.levelname,
            "logger": record.name,
            "rank": rank(),
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        # Any custom extra attributes passed via extra={...} starting with "ayaka_"
        for key, value in record.__dict__.items():
            if key.startswith("ayaka_"):
                payload[key[6:]] = value
        return json.dumps(payload, ensure_ascii=False, default=str)


class TextFormatter(logging.Formatter):
    """Human-readable text log formatter with rank context.

    Only prepends a rank prefix (e.g. ``[rank0/8]``) when running in distributed
    mode (world_size > 1). Keeps single-process local development logs clean.
    """

    def __init__(self) -> None:
        prefix = f"[{describe()}] " if is_distributed() else ""
        super().__init__(
            fmt=f"%(asctime)s {prefix}%(levelname)-7s %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
