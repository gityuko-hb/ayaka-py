from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ayaka.configs.model_source import ModelSourceConfig, RequestedFormat
from ayaka.exceptions import CheckpointSecurityError, ModelLoadError
from ayaka.utils.import_utils import import_module
from ayaka.utils.logging import DisabledTqdm
from ayaka.weights.plan import CheckpointFormat

CONFIG_FILENAME = "config.json"
GENERATION_CONFIG_FILENAME = "generation_config.json"
INDEX_FILENAME = "model.safetensors.index.json"
SINGLE_FILENAME = "model.safetensors"


def hub_cache_dir(model: str, *, cache_dir: str = "") -> Path:
    """Where a hub id lands on disk, per the HF cache layout.

    Layout knowledge only; nothing here touches the network or requires
    ``huggingface_hub`` to be installed.

    Note: ids are slash-separated but the cache flattens ``/`` to ``--``, so
    ``"a/b"`` and the (unusual) literal id ``"a--b"`` share one directory.  The
    Hub itself has this same aliasing, so fixing it here would disagree with
    where ``snapshot_download`` puts the files; documented rather than fixed.
    """
    base = Path(cache_dir or os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    if base.name != "hub":
        base = base / "hub"
    return base / ("models--" + model.replace("/", "--"))


# Returns the local snapshot directory for a hub id.
SnapshotFetcher = Callable[[str, str], str]


class SourceNotFoundError(ModelLoadError):
    """Nothing at the requested path, and no local hub snapshot for the id."""


@dataclass(frozen=True, slots=True)
class ResolvedSource:
    """A checkpoint located and contained.  No bytes read beyond a directory listing."""

    root: str  # canonical absolute directory
    format: CheckpointFormat
    config_path: str
    weight_files: tuple[str, ...] = ()  # absolute, sorted, all inside `root`
    index_path: str = ""
    revision: str = ""
    generation_config_path: str = ""

    @property
    def is_sharded(self) -> bool:
        return self.format is CheckpointFormat.SAFETENSORS_SHARDED

    def contained(self, path: str) -> str:
        """Re-check containment for a path built later.  Cheap; call it."""
        return str(safe_join(Path(self.root), path))


def safe_join(root: Path, relative: str) -> Path:
    """Join and prove the result stays under ``root``.

    ``Path.resolve()`` on both sides, then ``relative_to``.  Doing the check on
    the resolved paths is what makes it catch symlinks: a path-string check
    passes ``snapshots/abc/w.safetensors`` happily while the file itself points
    at ``/etc/shadow``.

    An absolute ``relative`` is refused outright rather than silently replacing
    the root, which is what ``os.path.join`` would do.
    """
    if not relative:
        raise CheckpointSecurityError("empty relative path")
    candidate = Path(relative)
    if candidate.is_absolute() or (os.name == "nt" and candidate.drive):
        raise CheckpointSecurityError(
            f"absolute path {relative!r} in a checkpoint index — refusing to escape the snapshot"
        )
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        # Allow Hugging Face Hub blob symlinks within the hub cache directory
        in_hub_blobs = False
        try:
            if resolved_root.parent.name == "snapshots":
                hub_dir = resolved_root.parent.parent.parent
                if (hub_dir / "blobs").is_dir() and resolved.is_relative_to(hub_dir):
                    in_hub_blobs = True
        except (ValueError, AttributeError):
            pass
        if not in_hub_blobs:
            raise CheckpointSecurityError(
                f"{relative!r} resolves to {resolved} which is outside the snapshot {resolved_root}"
            ) from exc
    return resolved


def _latest_snapshot(cache: Path, revision: str) -> Path | None:
    snapshots = cache / "snapshots"
    if not snapshots.is_dir():
        return None
    if revision:
        exact = snapshots / revision
        if exact.is_dir():
            return exact
        ref = cache / "refs" / revision
        if ref.is_file():
            sha = ref.read_text(encoding="utf-8").strip()
            if (snapshots / sha).is_dir():
                return snapshots / sha
        return None
    # Newest download wins, not the lexicographically-largest hash: snapshot
    # dirs are named for commit shas, so name order is arbitrary.  mtime is
    # the only signal of "the one the user downloaded last"; the name breaks
    # ties so the choice stays deterministic across identical mtimes.
    candidates: list[tuple[int, str, Path]] = []
    for p in snapshots.iterdir():
        if not p.is_dir():
            continue
        try:
            mtime = p.stat().st_mtime_ns
        except OSError:
            continue  # vanished between iterdir and stat; not a candidate
        candidates.append((mtime, p.name, p))
    candidates.sort(key=lambda t: (t[0], t[1]))
    return candidates[-1][2] if candidates else None


def _classify(root: Path, requested: RequestedFormat) -> tuple[CheckpointFormat, Path | None]:
    """Decide the format from what is on disk, honouring an explicit request.

    Sharded is checked first: a sharded checkpoint sometimes also ships a stray
    ``model.safetensors``, and treating that as the whole model loads one shard
    and reports every other weight missing.
    """
    index = root / INDEX_FILENAME
    single = root / SINGLE_FILENAME
    if requested is RequestedFormat.DUMMY:
        return CheckpointFormat.DUMMY, None
    if requested is RequestedFormat.PT:
        return CheckpointFormat.PYTORCH_BIN, None
    if index.is_file():
        return CheckpointFormat.SAFETENSORS_SHARDED, index
    if single.is_file():
        return CheckpointFormat.SAFETENSORS, None
    shards = sorted(root.glob("*.safetensors"))
    if shards:
        # Shards with no index.  Legal but unusual; the manifest builder
        # will have to read every header instead of a weight_map.
        return CheckpointFormat.SAFETENSORS, None
    raise SourceNotFoundError(
        f"no safetensors checkpoint in {root}: expected {INDEX_FILENAME} or "
        f"{SINGLE_FILENAME} or *.safetensors"
    )


def resolve_source(
    source: ModelSourceConfig,
    *,
    fetch_snapshot: SnapshotFetcher | None = None,
) -> ResolvedSource:
    """Locate the checkpoint.  Local filesystem first, always.

    A local directory named ``org/name`` beats a hub lookup for the same string.
    The reverse order is how a typo silently downloads a stranger's model.

    Reproducible mode is satisfied by any local path without a revision — the
    bytes on disk are already pinned and cannot move the way a floating ref
    can.  A hub id must resolve to the exact commit the revision names: the
    snapshot directory in the cache is named for the commit it holds, so a
    branch or tag that floats to different bytes next week is refused here
    rather than silently reused.
    """
    path = Path(source.model).expanduser()
    single_file: Path | None = None
    revision = source.revision

    if path.is_dir():
        root = path.resolve()
        revision = ""  # a directory has no revision; do not claim one
    elif path.is_file() and path.suffix == ".safetensors":
        # A single file, addressed directly.  The root is its parent, and only
        # this one file is in scope — globbing the parent would silently pull
        # in every other checkpoint sitting in the same downloads folder, and
        # the parent's index (if any) describes whatever multi-shard
        # checkpoint lives there, not this file.
        single_file = path.resolve()
        root = single_file.parent
        revision = ""
    else:
        if source.reproducible and not source.revision:
            raise CheckpointSecurityError(
                "reproducible mode needs a pinned revision; a floating ref resolves "
                "to different bytes on different days"
            )
        cache = hub_cache_dir(source.model, cache_dir=source.cache_dir)
        snapshot = _latest_snapshot(cache, source.revision)
        # A snapshot left half-fetched — pass one died mid-download — has a
        # directory but no usable contents.  Trusting it turns "network hiccup,
        # retry" into "no config.json", so an incomplete snapshot is treated
        # exactly like no snapshot at all: fetch.
        usable = snapshot is not None and (snapshot / CONFIG_FILENAME).is_file()
        if snapshot is not None and not usable and source.offline:
            raise SourceNotFoundError(
                f"offline mode: local snapshot for {source.model!r} is incomplete "
                f"(no {CONFIG_FILENAME} in {snapshot}) — re-fetch with network access"
            )
        if snapshot is None or not usable:
            if source.offline:
                raise SourceNotFoundError(
                    f"offline mode and no local snapshot for {source.model!r} (looked in {cache})"
                )
            if fetch_snapshot is None:
                raise SourceNotFoundError(
                    f"{source.model!r} is neither a local path nor a cached snapshot, "
                    "and no hub fetcher was supplied (node 1.14)"
                )
            snapshot = Path(fetch_snapshot(source.model, source.revision)).resolve()
        root = snapshot.resolve()
        revision = source.revision or root.name
        # Strict reproducibility for hub ids: the snapshot directory is named
        # for the commit it contains, so the resolved revision must *be* that
        # commit.  "main" pins nothing — next week's "main" is different bytes.
        if source.reproducible and revision != root.name:
            raise CheckpointSecurityError(
                f"reproducible mode asked for revision {source.revision!r}, which "
                f"resolves to commit {root.name!r}; a branch or tag is not a pin — "
                "pass the commit sha"
            )

    if single_file is None:
        fmt, index = _classify(root, source.format)
    else:
        fmt, index = CheckpointFormat.SAFETENSORS, None
    config = root / CONFIG_FILENAME
    if not config.is_file() and fmt is not CheckpointFormat.DUMMY:
        raise SourceNotFoundError(f"no {CONFIG_FILENAME} in {root}")

    if single_file is not None:
        weight_files = (str(single_file),)
    else:
        weight_files = tuple(str(p) for p in sorted(root.glob("*.safetensors")))
        for name in weight_files:
            safe_join(root, Path(name).name)  # symlink escape check, per file

    gen_config = root / GENERATION_CONFIG_FILENAME
    generation_config_path = str(gen_config) if gen_config.is_file() else ""

    return ResolvedSource(
        root=str(root),
        format=fmt,
        config_path=str(config),
        weight_files=weight_files,
        index_path=str(index) if index else "",
        revision=revision,
        generation_config_path=generation_config_path,
    )


def optional_hf_file(
    model_path: str | Path,
    filename: str,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
) -> str | None:
    """Local path of ``filename`` in a checkpoint dir or Hub repo; None when unavailable.

    Checks:
    1. Local filesystem path if ``model_path`` is an existing directory.
    2. Local Hugging Face Hub cache directory.
    3. Hub download via ``huggingface_hub.hf_hub_download`` using ``DisabledTqdm``.
    """
    path_obj = Path(model_path).expanduser()
    if path_obj.is_dir():
        file_path = path_obj / filename
        return str(file_path) if file_path.is_file() else None

    # Try looking into the local HF cache before hitting the network
    cache = hub_cache_dir(str(model_path), cache_dir=cache_dir or "")
    snapshot = _latest_snapshot(cache, revision or "")
    if snapshot is not None:
        file_path = snapshot / filename
        if file_path.is_file():
            return str(file_path)

    # Optional network download via huggingface_hub
    try:
        hf_hub = import_module("huggingface_hub")
        return hf_hub.hf_hub_download(
            repo_id=str(model_path),
            filename=filename,
            revision=revision or None,
            cache_dir=cache_dir or None,
            tqdm_class=DisabledTqdm,
        )
    except Exception:
        return None


def load_generation_config(
    source: str | Path | ResolvedSource | ModelSourceConfig,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
) -> dict[str, Any]:
    """Load and parse ``generation_config.json`` into a dictionary.

    Returns an empty dictionary if the configuration is not present or cannot be parsed.
    """
    if isinstance(source, ResolvedSource):
        if source.generation_config_path and os.path.isfile(source.generation_config_path):
            try:
                with open(source.generation_config_path, encoding="utf-8") as f:
                    data = json.load(f)
                return data if isinstance(data, dict) else {}
            except (OSError, json.JSONDecodeError):
                return {}
        return {}

    if isinstance(source, ModelSourceConfig):
        model_str = source.model
        rev = revision or source.revision or None
        cache = cache_dir or source.cache_dir or None
    else:
        model_str = str(source)
        rev = revision
        cache = cache_dir

    path = Path(model_str).expanduser()
    if path.is_file() and path.name == GENERATION_CONFIG_FILENAME:
        target_path: str | None = str(path)
    elif path.is_dir():
        candidate = path / GENERATION_CONFIG_FILENAME
        target_path = str(candidate) if candidate.is_file() else None
    else:
        target_path = optional_hf_file(
            model_str, GENERATION_CONFIG_FILENAME, revision=rev, cache_dir=cache
        )

    if target_path is None or not os.path.isfile(target_path):
        return {}

    try:
        with open(target_path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def parse_eos_token_ids(
    gen_config: Mapping[str, Any],
    extra_eos_token_ids: int | Sequence[int] | None = None,
) -> frozenset[int]:
    """Extract stop/EOS token IDs from a generation config dictionary and union with extra IDs.

    Many models (e.g. Gemma, Qwen) specify multiple stop token IDs in ``generation_config.json``
    (e.g. ``<end_of_turn>`` or ``<|im_end|>``). This extracts all integer IDs safely.
    """
    ids: set[int] = set()
    if extra_eos_token_ids is not None:
        if isinstance(extra_eos_token_ids, int):
            ids.add(int(extra_eos_token_ids))
        else:
            ids.update(int(x) for x in extra_eos_token_ids if x is not None)

    gen_eos = gen_config.get("eos_token_id")
    if isinstance(gen_eos, int):
        ids.add(int(gen_eos))
    elif isinstance(gen_eos, (list, tuple, set)):
        ids.update(int(x) for x in gen_eos if x is not None)

    return frozenset(ids)


def load_eos_token_ids(
    source: str | Path | ResolvedSource | ModelSourceConfig,
    extra_eos_token_ids: int | Sequence[int] | None = None,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
) -> frozenset[int]:
    """Read all EOS token IDs for a model source."""
    cfg = load_generation_config(source, revision=revision, cache_dir=cache_dir)
    return parse_eos_token_ids(cfg, extra_eos_token_ids)


def parse_generation_sampling(gen_config: Mapping[str, Any]) -> dict[str, Any]:
    """Extract recommended generation sampling defaults from ``generation_config.json``.

    If ``do_sample`` is False, returns greedy sampling (``{"temperature": 0.0}``).
    Otherwise extracts ``temperature``, ``top_k``, and ``top_p`` when present.
    """
    if gen_config.get("do_sample") is False:
        return {"temperature": 0.0}

    out: dict[str, Any] = {}
    for key in ("temperature", "top_k", "top_p"):
        if (val := gen_config.get(key)) is not None:
            if key in ("temperature", "top_p"):
                out[key] = float(val)
            elif key == "top_k":
                out[key] = int(val)
            else:
                out[key] = val
    return out


def load_generation_sampling(
    source: str | Path | ResolvedSource | ModelSourceConfig,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
) -> dict[str, Any]:
    """Read recommended sampling defaults from ``generation_config.json``."""
    cfg = load_generation_config(source, revision=revision, cache_dir=cache_dir)
    return parse_generation_sampling(cfg)
