from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from ayaka.configs.model_source import ModelSourceConfig
from ayaka.exceptions import CheckpointCorruptError, ModelLoadError
from ayaka.model_loader.source import (
    CONFIG_FILENAME,
    INDEX_FILENAME,
    SINGLE_FILENAME,
    SnapshotFetcher,
)
from ayaka.utils.import_utils import import_module

# Files fetched in pass one: enough to decide what pass two needs, and small
# enough that fetching them for a model you then reject costs nothing.
CONTROL_FILES: tuple[str, ...] = (
    CONFIG_FILENAME,
    INDEX_FILENAME,
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
)

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
    ) -> str: ...

def shard_filenames_from_index(index_path: Path) -> tuple[str, ...]:
    """Which shard files the index actually references.

    Deliberately *not* ``read_weight_map``: that function validates the map
    against shard headers, and at this point the shards have not been
    downloaded yet.
    """
    try:
        obj: Any = json.loads(index_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointCorruptError(f"{index_path}: cannot read index: {exc}") from exc
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
        try:
            # Pass one: control files only.  If this repo turns out to be
            # quantized or an architecture we do not serve, we stop here having
            # spent kilobytes instead of gigabytes.
            root = Path(
                fn(
                    model,
                    revision=rev,
                    cache_dir=cache,
                    local_files_only=source.offline,
                    allow_patterns=list(CONTROL_FILES),
                )
            )

            index = root / INDEX_FILENAME
            if index.is_file():
                wanted = list(shard_filenames_from_index(index))
            else:
                wanted = [SINGLE_FILENAME, "*.safetensors"]

            # Pass two: exactly the weight files, and nothing else in the repo.
            root = Path(
                fn(
                    model,
                    revision=rev,
                    cache_dir=cache,
                    local_files_only=source.offline,
                    allow_patterns=[*CONTROL_FILES, *wanted],
                )
            )
        except (HubUnavailableError, CheckpointCorruptError):
            raise
        except Exception as exc:
            # huggingface_hub raises RepositoryNotFoundError, RevisionNotFound,
            # HFValidationError, requests' ConnectionError, OSError...
            raise HubUnavailableError(
                f"could not fetch {model!r}"
                + (f" at revision {revision!r}" if revision else "")
                + f": {type(exc).__name__}: {exc}"
            ) from exc
        return str(root)
    return fetch
