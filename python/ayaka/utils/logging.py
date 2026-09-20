from __future__ import annotations

import json
import logging
import os
import sys
import time
from functools import partial
from typing import Any, cast

from ayaka.distributed.env import describe, is_distributed, is_master, rank

__all__ = [
    "AyakaLogger",
    "ColorFormatter",
    "DisabledTqdm",
    "JSONFormatter",
    "TextFormatter",
    "init_logger",
    "log_master",
    "log_once",
    "quiet_http_loggers",
    "reset_once_state",
    "should_use_color",
]

# ANSI color escape sequences
_COLORS = {
    "DEBUG": "\033[36m",  # Cyan
    "INFO": "\033[32m",  # Green
    "WARNING": "\033[33m",  # Yellow
    "WARN": "\033[33m",  # Yellow
    "ERROR": "\033[31m",  # Red
    "CRITICAL": "\033[35m",  # Magenta
}
_RESET = "\033[0m"
_BOLD = "\033[1m"


def should_use_color(stream: Any = None) -> bool:
    """Determine whether color output should be enabled.

    Honors NO_COLOR, FORCE_COLOR, TERM=dumb, and stream TTY status.
    """
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR") not in (None, "", "0"):
        return True
    if os.environ.get("TERM") == "dumb":
        return False
    target_stream = stream if stream is not None else sys.stdout
    return bool(getattr(target_stream, "isatty", lambda: False)())


# HTTP client loggers emit one INFO line per request — a hub download is
# dozens of them, interleaved with progress bars and ayaka's own log lines.
# WARNING hides the per-request chatter while keeping genuine failures (timeouts,
# connection refusals) visible.  An operator who wants the chatter back sets
# AYAKA_HTTPX_LOGGING=1, or configures the logger's level explicitly before
# this runs — a level set on purpose is never overridden.
_QUIET_LOGGERS = ("httpx", "httpcore", "huggingface_hub", "urllib3")


def quiet_http_loggers() -> None:
    """Raise third-party HTTP loggers to WARNING, once, unless told not to.

    Called before a hub download so the operator sees ayaka's progress lines,
    not httpx's request log.  Idempotent; respects an explicit level (anything
    set before this call stays as it is) and the ``AYAKA_HTTPX_LOGGING`` escape
    hatch.
    """
    if os.environ.get("AYAKA_HTTPX_LOGGING"):
        return
    for name in _QUIET_LOGGERS:
        client_logger = logging.getLogger(name)
        if client_logger.level == logging.NOTSET:
            client_logger.setLevel(logging.WARNING)


from tqdm.auto import tqdm as _BaseTqdm


class DisabledTqdm(_BaseTqdm):
    """A tqdm subclass with progress output permanently disabled.

    Used when fetching lightweight files (e.g. metadata JSONs) or running in
    headless/server mode so progress updates do not spam the console or server logs.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.pop("name", None)
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)


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


class ColorFormatter(logging.Formatter):
    """Console formatter with ANSI colors, timestamps, and rank info."""

    COLORS = _COLORS
    RESET = _RESET
    BOLD = _BOLD

    def __init__(
        self,
        suffix: str = "",
        *,
        strip_file: bool = True,
        use_pid: bool | None = None,
        use_tp_rank: bool | None = None,
        use_color: bool | None = None,
    ) -> None:
        super().__init__()
        if strip_file and suffix:
            suffix = os.path.basename(suffix)
        self.suffix = f"|{suffix}" if suffix else ""

        if use_pid is None:
            use_pid = os.getenv("LOG_PID", "0").lower() in ("1", "true", "yes")
        if use_pid:
            self.suffix = f"|pid={os.getpid()}{self.suffix}"

        self.use_tp_rank = use_tp_rank
        self.use_color = should_use_color() if use_color is None else use_color

    def format(self, record: logging.LogRecord) -> str:
        # SGLang timestamp format: [YYYY-MM-DD|HH:MM:SS]
        timestamp = self.formatTime(record, "%Y-%m-%d|%H:%M:%S")

        rank_str = ""
        if self.use_tp_rank is not False and is_distributed():
            rank_str = f"|core|{describe()}"
        elif self.use_tp_rank is True:
            rank_str = f"|core|rank={rank()}"

        full_tag = f"[{timestamp}{self.suffix}{rank_str}]"
        levelname = record.levelname
        level_padded = f"{levelname:<8}"
        message = record.getMessage()

        if self.use_color:
            level_color = self.COLORS.get(levelname, "")
            colored_level = f"{level_color}{level_padded}{self.RESET}"
            header = f"{self.BOLD}{full_tag}{self.RESET} {colored_level}"
        else:
            header = f"{full_tag} {level_padded}"

        formatted = f"{header} {message}"

        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            if not formatted.endswith("\n"):
                formatted += "\n"
            formatted += record.exc_text
        if record.stack_info:
            if not formatted.endswith("\n"):
                formatted += "\n"
            formatted += self.formatStack(record.stack_info)

        return formatted


class AyakaLogger(logging.Logger):
    """Custom logger type providing convenience helpers for distributed rank 0 logging."""

    def info_rank0(self, msg: object, *args: object, **kwargs: Any) -> None:
        """Log at INFO level only on rank 0."""
        ...

    def warning_rank0(self, msg: object, *args: object, **kwargs: Any) -> None:
        """Log at WARNING level only on rank 0."""
        ...

    def debug_rank0(self, msg: object, *args: object, **kwargs: Any) -> None:
        """Log at DEBUG level only on rank 0."""
        ...

    def error_rank0(self, msg: object, *args: object, **kwargs: Any) -> None:
        """Log at ERROR level only on rank 0."""
        ...

    def critical_rank0(self, msg: object, *args: object, **kwargs: Any) -> None:
        """Log at CRITICAL level only on rank 0."""
        ...


_LEVEL_MAP: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "WARN": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def _resolve_log_level(level: int | str | None) -> int:
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        level_upper = level.strip().upper()
        if level_upper in _LEVEL_MAP:
            return _LEVEL_MAP[level_upper]
    env_level = os.getenv("LOG_LEVEL", "").strip().upper()
    return _LEVEL_MAP.get(env_level, logging.INFO)


def init_logger(
    name: str = "ayaka",
    suffix: str = "",
    *,
    strip_file: bool = True,
    level: int | str | None = None,
    use_pid: bool | None = None,
    use_tp_rank: bool | None = None,
    use_color: bool | None = None,
    stream: Any = None,
) -> AyakaLogger:
    """Initialize a logger with SGLang-style colors and pretty formatting.

    Args:
        name: Logger name (e.g. ``"ayaka"`` or ``__name__``).
        suffix: Optional suffix to append to the log prefix (e.g. filename).
        strip_file: If True and suffix is a file path, strip down to basename.
        level: Explicit log level (int or string). Defaults to ``$LOG_LEVEL`` or INFO.
        use_pid: Whether to include process ID in prefix. Defaults to ``$LOG_PID``.
        use_tp_rank: Whether to include distributed rank in prefix.
        use_color: Explicit boolean to force enable/disable ANSI colors.
        stream: Target stream for logging. Defaults to ``sys.stdout``.

    Returns:
        An :class:`AyakaLogger` instance configured with SGLang formatting and rank-0 helpers.
    """
    resolved_level = _resolve_log_level(level)
    target_stream = stream if stream is not None else sys.stdout

    logger = logging.getLogger(name)
    logger.setLevel(resolved_level)
    logger.handlers.clear()

    formatter = ColorFormatter(
        suffix=suffix,
        strip_file=strip_file,
        use_pid=use_pid,
        use_tp_rank=use_tp_rank,
        use_color=use_color if use_color is not None else should_use_color(target_stream),
    )

    handler = logging.StreamHandler(target_stream)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

    def _call_rank0(level_name: str, msg: object, *args: object, **kwargs: Any) -> None:
        if is_master():
            getattr(logger, level_name)(msg, *args, **kwargs)

    wrapper = cast(Any, logger)
    wrapper.info_rank0 = partial(_call_rank0, "info")
    wrapper.warning_rank0 = partial(_call_rank0, "warning")
    wrapper.debug_rank0 = partial(_call_rank0, "debug")
    wrapper.error_rank0 = partial(_call_rank0, "error")
    wrapper.critical_rank0 = partial(_call_rank0, "critical")

    return cast(AyakaLogger, logger)
