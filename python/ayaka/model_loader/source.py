from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ayaka.configs.model_source import ModelSourceConfig, RequestedFormat
from ayaka.exceptions import CheckpointSecurityError, ModelLoadError
from ayaka.weights.plan import CheckpointFormat

CONFIG_FILENAME = "config.json"
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
        if snapshot is None:
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

    return ResolvedSource(
        root=str(root),
        format=fmt,
        config_path=str(config),
        weight_files=weight_files,
        index_path=str(index) if index else "",
        revision=revision,
    )
