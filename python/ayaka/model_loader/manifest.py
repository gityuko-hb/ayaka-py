from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ayaka.exceptions import CheckpointCorruptError, UnsupportedWeightFormatError
from ayaka.model_loader.source import ResolvedSource, safe_join
from ayaka.model_loader.st import SafetensorsHeader, read_safetensors_header
from ayaka.weights.plan import CheckpointFormat, CheckpointManifest, ManifestEntry


def _headers_for(root: Path, filenames: list[str]) -> dict[str, SafetensorsHeader]:
    """Read and validate the safetensors headers named by an index.

    Args:
        root: Canonical snapshot directory containing the shard files.
        filenames: Relative shard paths taken from the checkpoint index.

    Returns:
        A mapping from each relative shard name to its parsed
        :class:`~ayaka.model_loader.st.SafetensorsHeader`.

    Raises:
        CheckpointCorruptError: If a shard path escapes ``root``, does not
            exist, or has an invalid safetensors header.
    """
    headers: dict[str, SafetensorsHeader] = {}
    for name in filenames:
        # Containment re-checked here and not only at resolution: `name` came
        # out of the index, which is checkpoint-controlled data (node 1.09).
        path = safe_join(root, name)
        if not path.is_file():
            raise CheckpointCorruptError(
                f"index names shard {name!r} but {path} does not exist — "
                "an interrupted or partial download"
            )
        headers[name] = read_safetensors_header(path)
    return headers


def _group_by_shard(weight_map: dict[str, str]) -> dict[str, list[str]]:
    """Invert a weight map into shard names and their tensor keys.

    Grouping before reading headers ensures each shard is parsed once. Tensor
    names in every group are sorted so manifest construction is deterministic
    regardless of the JSON object's insertion order.

    Args:
        weight_map: Mapping from tensor names to relative shard filenames as
            read from ``model.safetensors.index.json``.

    Returns:
        A mapping from each shard filename to its sorted tensor names. The
        returned lists are newly allocated and do not alias the input mapping.
    """
    grouped: dict[str, list[str]] = {}
    for tensor, shard in weight_map.items():
        grouped.setdefault(shard, []).append(tensor)
    for names in grouped.values():
        names.sort()
    return grouped


def _from_index(source: ResolvedSource) -> tuple[tuple[ManifestEntry, ...], tuple[str, ...]]:
    """Build manifest entries from an indexed safetensors checkpoint.

    The index is checked in both directions: every mapped tensor must occur in
    its declared shard, and every tensor found in a shard must be present in
    the index. This prevents a stale or incomplete index from silently
    dropping weights. The optional ``metadata.total_size`` is also checked
    against the bytes declared by the shard headers.

    Args:
        source: Resolved checkpoint source with a non-empty ``index_path``.

    Returns:
        A tuple containing the manifest entries discovered from the index and
        the absolute shard paths used to produce them.

    Raises:
        CheckpointCorruptError: If the index, shard headers, tensor ownership,
            or declared total size is inconsistent.
    """
    root = Path(source.root)
    weight_map, metadata = read_weight_map(source.index_path)
    tensors_by_shard = _group_by_shard(weight_map)
    shard_names = sorted(tensors_by_shard)
    headers = _headers_for(root, shard_names)

    # One pass per shard, in a stable order.  Each header is parsed exactly once
    # (asserted by the call-count test) and indexed once; resolution is then a
    # dict lookup per tensor.  The version this replaced scanned a shard's
    # entries for every weight_map key — invisible on a 291-tensor dense
    # checkpoint, tens of seconds on Qwen2-57B-A14B's ~5.4k or DeepSeek-V3's ~47k.
    entries: list[ManifestEntry] = []
    for shard in shard_names:
        by_key = {e.tensor_key: e for e in headers[shard].entries}
        for key in tensors_by_shard[shard]:
            entry = by_key.get(key)
            if entry is None:
                raise CheckpointCorruptError(
                    f"index maps {key!r} to {shard} but that shard's header does not "
                    f"contain it — the index and the shards disagree"
                )
            entries.append(entry)

    # The reverse direction.  A tensor present in a shard but absent from the
    # map would never be read, and "the checkpoint had it and we ignored it" is
    # indistinguishable at runtime from "the checkpoint never had it".
    mapped = set(weight_map)
    for shard, header in headers.items():
        stray = sorted(header.tensor_keys - mapped)
        if stray:
            raise CheckpointCorruptError(
                f"{shard} contains {len(stray)} tensor(s) the index never maps, "
                f"starting with {stray[0]!r} — they would silently never be read"
            )

    declared_total = metadata.get("total_size")
    if isinstance(declared_total, int):
        actual = sum(e.nbytes for e in entries)
        if declared_total != actual:
            raise CheckpointCorruptError(
                f"index metadata says total_size={declared_total} but the shard "
                f"headers sum to {actual}"
            )
    return tuple(entries), tuple(str(safe_join(root, n)) for n in shard_names)


def _from_headers(source: ResolvedSource) -> tuple[tuple[ManifestEntry, ...], tuple[str, ...]]:
    """Build manifest entries by reading unindexed safetensors files.

    Args:
        source: Resolved checkpoint source whose ``weight_files`` contain the
            safetensors files to inspect.

    Returns:
        A tuple containing all parsed manifest entries and the source files in
        their resolved order.

    Raises:
        CheckpointCorruptError: If a file has an invalid header or the same
            tensor key appears in more than one file.
    """
    entries: list[ManifestEntry] = []
    owner: dict[str, str] = {}
    for path in source.weight_files:
        header = read_safetensors_header(path)
        for entry in header.entries:
            previous = owner.get(entry.tensor_key)
            if previous is not None:
                raise CheckpointCorruptError(
                    f"tensor {entry.tensor_key!r} appears in both {previous} and "
                    f"{path} — with no index there is no way to tell which is current"
                )
            owner[entry.tensor_key] = path
            entries.append(entry)
    return tuple(entries), tuple(source.weight_files)


def read_weight_map(index_path: str | Path) -> tuple[dict[str, str], dict[str, Any]]:
    """Read and validate a ``model.safetensors.index.json`` file.

    The tensor map and metadata are returned separately because metadata such
    as ``total_size`` is useful as a cross-check and must not be discarded when
    the tensor map is consumed.

    Args:
        index_path: Path to the JSON index file.

    Returns:
        A pair ``(weight_map, metadata)``. ``weight_map`` maps tensor names to
        shard filenames. ``metadata`` contains the original metadata object
        when present, or an empty dictionary otherwise.

    Raises:
        CheckpointCorruptError: If the file cannot be read, is not valid JSON,
            is not a JSON object, or does not contain a non-empty string-to-
            string ``weight_map``.
    """

    p = Path(index_path)
    try:
        raw = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise CheckpointCorruptError(f"{p}: cannot read index: {exc}") from exc

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CheckpointCorruptError(f"{p}: index is not valid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise CheckpointCorruptError(f"{p}: index is {type(obj).__name__}, not an object")

    weight_map = obj.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise CheckpointCorruptError(f"{p}: index has no usable `weight_map`")
    for key, value in weight_map.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise CheckpointCorruptError(
                f"{p}: weight_map entry {key!r} -> {value!r} is not string -> string"
            )
    raw_metadata = obj.get("metadata")
    metadata: dict[str, Any] = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
    return dict(weight_map), metadata


def build_manifest_from_source(source: ResolvedSource) -> CheckpointManifest:
    """Build a complete, physically ordered manifest for a resolved source.

    Indexed checkpoints are resolved through their index; unindex
    checkpoints are resolved by reading each safetensors header. The returned
    manifest is sorted by ``(file_uri, byte_offset)`` so equivalent sources
    produce byte-identical ordering across machines.

    Args:
        source: A source already resolved to a local directory and checkpoint
            format.

    Returns:
        A validated :class:`~ayaka.weights.plan.CheckpointManifest`. Dummy
        sources produce an empty dummy manifest.

    Raises:
        CheckpointCorruptError: If the checkpoint contains no tensors or its
            index/header data is inconsistent.
        UnsupportedWeightFormatError: If the source is a PyTorch ``.bin``
            checkpoint, which would require unsafe pickle loading.
    """
    if source.format is CheckpointFormat.DUMMY:
        return CheckpointManifest(
            format=CheckpointFormat.DUMMY, root_uri=source.root, revision=source.revision
        )
    if source.format is CheckpointFormat.PYTORCH_BIN:
        raise UnsupportedWeightFormatError(
            "Reads Safetensors only; a .bin checkpoint requires unpickling, "
            "which executes arbitrary code from the checkpoint"
        )

    if source.index_path:
        entries, files = _from_index(source)
    else:
        entries, files = _from_headers(source)

    if not entries:
        raise CheckpointCorruptError(f"{source.root}: checkpoint contains no tensors")

    # Physical order, so two machines produce byte-identical manifests.  The
    # protocol type re-asserts this in __post_init__ — sorting here and
    # validating there is deliberate belt-and-braces, because the sort is what
    # makes the invariant true and the assert is what keeps a future caller from
    # constructing one by hand in dict order.
    ordered = tuple(sorted(entries, key=lambda e: (e.file_uri, e.byte_offset)))
    return CheckpointManifest(
        format=source.format,
        root_uri=source.root,
        entries=ordered,
        files=files,
        revision=source.revision,
    )
