from __future__ import annotations

import errno
import json
import logging
import os
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

from ayaka.configs.model_source import ModelSourceConfig
from ayaka.exceptions import CheckpointCorruptError, ModelLoadError
from ayaka.model_loader._logging import logger
from ayaka.model_loader.source import (
    CONFIG_FILENAME,
    INDEX_FILENAME,
    SINGLE_FILENAME,
    SnapshotFetcher,
)
from ayaka.utils.import_utils import import_module
from ayaka.utils.logging import log_master, quiet_http_loggers

if sys.platform == "win32":
    # The Xet content backend opens many concurrent connections from its own
    # Rust network stack and is the repeat offender behind WinError 10037
    # (WSAEALREADY, "operation already in progress") on Windows.  The plain
    # HTTPS CDN path is slower per connection but stable, so default it on
    # unless the operator asked for Xet explicitly before we got here.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# Files fetched in pass one: enough to decide what pass two needs, and small
# enough that fetching them for a model you then reject costs nothing.
CONTROL_FILES: tuple[str, ...] = (
    CONFIG_FILENAME,
    INDEX_FILENAME,
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
)

# Windows winsock races and CDN hiccups are transient by nature; five attempts
# with exponential backoff rides out roughly half a minute of flakiness before
# the load is declared a network failure.
_RETRY_ATTEMPTS = 5
_RETRY_BASE_DELAY_S = 1.0

# A modest pool: snapshot_download defaults to eight concurrent files, and each
# one fans out further inside the HTTP layer — the exact shape of concurrency
# that trips non-blocking-connect races on Windows.  Two workers keep the
# bandwidth while the connections stay serialized enough to be reliable.
_DOWNLOAD_WORKERS = 2

# The default is 10s, which times out on slow links before the retry layer ever
# sees a chance to help.
_ETAG_TIMEOUT_S = 30.0


class HubUnavailableError(ModelLoadError):
    """``huggingface_hub`` is not installed, or the hub could not be reached.

    A ``ModelLoadError`` so bootstrap's single handler covers it, with a message
    that names the extra to install — an ``ImportError`` traceback from a
    transitive dependency tells the reader nothing about what to do next.
    """


class SnapshotDownloader(Protocol):
    """The one function this module needs from ``huggingface_hub``.

    Declared structurally so the real ``snapshot_download`` and a test double
    are interchangeable without either importing the other.
    """

    def __call__(
        self,
        repo_id: str,
        *,
        revision: str | None = ...,
        cache_dir: str | None = ...,
        local_files_only: bool = ...,
        allow_patterns: Sequence[str] | None = ...,
        max_workers: int = ...,
        etag_timeout: float = ...,
    ) -> str: ...


def _is_transient(exc: Exception) -> bool:
    """Whether a fetch failure is the kind a retry can fix.

    Connection resets, read timeouts and Windows winsock races (10037,
    WSAEALREADY — non-blocking connect colliding with one already in flight)
    are transient.  A 404, a validation error or an auth refusal never becomes
    true by waiting, so those propagate on the first attempt.
    """
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    if isinstance(exc, OSError):
        # BlockingIOError(10037) surfaces errno, not winerror, depending on how
        # the socket layer raised; check both.
        if getattr(exc, "winerror", None) == 10037 or exc.errno == 10037:
            return True
        return exc.errno in (errno.ECONNRESET, errno.ECONNABORTED, errno.ETIMEDOUT, errno.EAGAIN)
    # requests' ConnectionError is an OSError subclass in practice, but some
    # urllib3 wrappers are not; treat any exception whose type name mentions a
    # connection or timeout as transient rather than importing urllib3 here.
    name = type(exc).__name__.lower()
    return "connection" in name or "timeout" in name


def _with_retry(operation: Callable[[], Path], *, model: str, revision: str) -> Path:
    """Run a fetch with exponential backoff: 1s, 2s, 4s, 8s, 16s.

    ``snapshot_download`` resumes partial files, so a retry never re-downloads
    bytes already on disk — the cost of an attempt is only the transfer that
    had not finished when the connection died.
    """
    where = f"{model!r}" + (f" at revision {revision!r}" if revision else "")
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            return operation()
        except Exception as exc:
            if not _is_transient(exc) or attempt == _RETRY_ATTEMPTS - 1:
                raise
            delay = _RETRY_BASE_DELAY_S * (2**attempt)
            log_master(
                logger,
                logging.WARNING,
                "network error fetching %s (%s: %s), retrying %d/%d in %.0fs...",
                where,
                type(exc).__name__,
                exc,
                attempt + 1,
                _RETRY_ATTEMPTS,
                delay,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover - loop raises or returns


def _import_snapshot_download() -> SnapshotDownloader:
    try:
        # Imported by name rather than with a plain `import` so that mypy on a
        # base install — where the extra is absent — does not fail the type
        # gate over an optional dependency.  The Protocol above is the contract
        # that matters; the real callable is checked structurally at the call.
        module = import_module("huggingface_hub")
    except ImportError as exc:
        raise HubUnavailableError(
            "downloading from the Hugging Face Hub needs the optional dependency: "
            "pip install 'ayaka[hub]' (or huggingface_hub). A local checkpoint "
            "path needs no extras."
        ) from exc
    download: SnapshotDownloader = module.snapshot_download
    return download


def shard_filenames_from_index(index_path: Path) -> tuple[str, ...]:
    """Which shard files the index actually references.

    Deliberately *not* ``read_weight_map``: that function validates the map
    against shard headers, and at this point the shards have not been
    downloaded yet.

    The duplicate-key refusal matches ``read_weight_map`` and the shard-header
    parser: ``json.loads`` alone keeps the last value for a repeated key, and a
    tensor dropped from the index here is a shard that pass two never fetches.
    """
    try:
        raw = index_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CheckpointCorruptError(f"{index_path}: cannot read index: {exc}") from exc

    def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        seen: set[str] = set()
        duplicates: set[str] = set()
        for key, _ in pairs:
            if key in seen:
                duplicates.add(key)
            else:
                seen.add(key)
        if duplicates:
            raise CheckpointCorruptError(
                f"{index_path}: duplicate keys in checkpoint index: {sorted(duplicates)} — "
                "json would have kept only the last and dropped the rest silently"
            )
        return dict(pairs)

    try:
        obj: Any = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise CheckpointCorruptError(f"{index_path}: index is not valid JSON: {exc}") from exc
    weight_map = obj.get("weight_map") if isinstance(obj, dict) else None
    if not isinstance(weight_map, dict):
        raise CheckpointCorruptError(f"{index_path}: index has no usable `weight_map`")
    names = {str(v) for v in weight_map.values()}
    if not names:
        raise CheckpointCorruptError(f"{index_path}: `weight_map` names no shard files")
    return tuple(sorted(names))


def hub_fetcher(
    source: ModelSourceConfig,
    *,
    download: SnapshotDownloader | None = None,
) -> SnapshotFetcher:
    """Build the fetcher ``resolve_source`` will call.

    Returns a closure rather than doing the work, because ``resolve_source``
    must check the local filesystem first: constructing a fetcher has to be free
    so it can be passed in and never used.
    """

    def fetch(model: str, revision: str) -> str:
        fn = download or _import_snapshot_download()
        rev = revision or None
        cache = source.cache_dir or None

        def call(patterns: Sequence[str]) -> Path:
            return Path(
                fn(
                    model,
                    revision=rev,
                    cache_dir=cache,
                    local_files_only=source.offline,
                    allow_patterns=list(patterns),
                    max_workers=_DOWNLOAD_WORKERS,
                    etag_timeout=_ETAG_TIMEOUT_S,
                )
            )

        where = f"{model!r}" + (f" @ {revision}" if revision else "")
        # One INFO line per HTTP request turns a download into dozens of lines
        # between ayaka's own progress messages; keep request-level chatter off
        # unless the operator opted back in.
        quiet_http_loggers()
        try:
            log_master(logger, logging.INFO, "fetching %s (pass 1: metadata and config)", where)
            # Pass one: control files only.  If this repo turns out to be
            # quantized or an architecture we do not serve, we stop here having
            # spent kilobytes instead of gigabytes.
            root = _with_retry(lambda: call(CONTROL_FILES), model=model, revision=revision)

            index = root / INDEX_FILENAME
            if index.is_file():
                wanted = list(shard_filenames_from_index(index))
            else:
                wanted = [SINGLE_FILENAME, "*.safetensors"]

            log_master(
                logger,
                logging.INFO,
                "fetching %s (pass 2: %d weight file(s))",
                where,
                len(wanted),
            )
            # Pass two: exactly the weight files, and nothing else in the repo.
            # tqdm progress bars per file come from huggingface_hub itself.
            started = time.monotonic()
            root = _with_retry(
                lambda: call([*CONTROL_FILES, *wanted]), model=model, revision=revision
            )
            elapsed = time.monotonic() - started
            log_master(
                logger,
                logging.INFO,
                "fetched %s in %.1fs%s",
                where,
                elapsed,
                "" if source.offline else " — files already cached are skipped by the hub",
            )
        except (HubUnavailableError, CheckpointCorruptError):
            raise
        except Exception as exc:
            # huggingface_hub raises RepositoryNotFoundError, RevisionNotFound,
            # HFValidationError, requests' ConnectionError, OSError...
            raise HubUnavailableError(
                f"could not fetch {model!r}"
                + (f" at revision {revision!r}" if revision else "")
                + f" after {_RETRY_ATTEMPTS} attempts: {type(exc).__name__}: {exc}"
                + " — if the model is already cached, try offline mode; otherwise "
                "check the network and retry"
            ) from exc
        return str(root)

    return fetch
