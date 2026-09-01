"""Safetensors header, parsed defensively.

Layout, in full, because every check below refers to it:

    [0, 8)          u64 little-endian: header length N
    [8, 8+N)        UTF-8 JSON: {tensor_key: {dtype, shape, data_offsets}, ...}
    [8+N, EOF)      tensor payload; data_offsets are relative to 8+N

The whole file is memory-mappable and the format is trivial, which is exactly
why it gets parsed carelessly.  Every field is attacker-controlled: N can be
larger than the file, ``data_offsets`` can be ``[0, 2**64-1]``, two tensors can
claim the same bytes, a shape can multiply to something that disagrees with the
byte count.  None of those are caught by ``json.loads``.

Nothing here reads a tensor payload.  The point of the node is to answer "what
is in this file and is it self-consistent" while touching only the first few
kilobytes — so that discovery of a 400 GB sharded checkpoint costs a few
hundred reads, not a full traversal.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ayaka.exceptions import CheckpointCorruptError, UnsupportedWeightFormatError
from ayaka.types import DType
from ayaka.weights.plan import ManifestEntry

# Safetensors dtype tokens.  Explicit map: an unknown token must be a refusal,
# because guessing an item size turns a corrupt file into a silently misread one.
_SAFETENSORS_DTYPES: dict[str, DType] = {
    "F64": DType.FP64,  # widened at read; there is no fp64 compute path
    "F32": DType.FP32,
    "F16": DType.FP16,
    "BF16": DType.BF16,
    "F8_E4M3": DType.FP8_E4M3,
    "F8_E5M2": DType.FP8_E5M2,
    "I64": DType.INT64,
    "I32": DType.INT32,
    "I8": DType.INT8,
    "U8": DType.UINT8,
    "BOOL": DType.BOOL,
}

# The largest unsigned integer that can be represented in 64 bits.
_U64_MAX = (1 << 64) - 1
# A real header is a few hundred KB at most (Llama-405B's index is ~5 MB of JSON
# spread over shards; a single shard header is far smaller).  The cap turns a
# malicious 2**63 length into an immediate refusal instead of an allocation that
# takes the machine down.
_HEADER_LENGTH_LIMIT = 64 << 20

@dataclass(frozen=True, slots=True)
class SafetensorsHeader:
    """One file's header, validated and normalized."""

    file_uri: str
    data_start: int # 8 + header_length
    file_size: int 
    entries: tuple[ManifestEntry,...] = ()
    metadata: tuple[tuple[str, str], ...] = ()
    
    @property
    def tensor_keys(self) -> frozenset[str]:
        return frozenset(e.tensor_key for e in self.entries)
    
    @property
    def payload_bytes(self) -> int:
        return sum(e.nbytes for e in self.entries)
    
def _dtype(token: object, key: str) -> DType:
    """Convert a safetensors dtype token to Ayaka's normalized dtype.

    Args:
        token: Raw ``dtype`` value decoded from the safetensors JSON header.
            It must be a string token supported by ``_SAFETENSORS_DTYPES``.
        key: Tensor name used to identify the field in validation errors.

    Returns:
        The corresponding :class:`~ayaka.types.DType` value.

    Raises:
        CheckpointCorruptError: If ``token`` is not a string or names an
            unknown safetensors dtype.
        UnsupportedWeightFormatError: If ``token`` is a recognized sub-byte
            dtype (``I4``, ``U4``, or ``F4``) for which A1 has no unpacking
            path.
    """
    if not isinstance(token, str):
        raise CheckpointCorruptError(f"{key}: dtype is {type(token).__name__}, not a string")
    try:
        return _SAFETENSORS_DTYPES[token]
    except KeyError as exc:
        # The only sub-byte dtype is int4, which has no unpacking path 
        #!TODO: Implement unpacking path for sub-byte dtypes
        if token in ("I4", "U4", "F4"):
            raise UnsupportedWeightFormatError(
                f"{key}: sub-byte dtype {token} — A1 has no unpacking path"
            ) from exc
        raise CheckpointCorruptError(
            f"{key}: unknown safetensors dtype {token!r}; known: {sorted(_SAFETENSORS_DTYPES)}"
        ) from exc
        
def _shape(raw: object, key: str) -> tuple[int, ...]:
    """Validate and normalize a safetensors tensor shape.

    Args:
        raw: Raw ``shape`` value decoded from the safetensors JSON header. It
            must be a list of non-negative integers; JSON booleans are rejected
            even though Python considers ``bool`` a subclass of ``int``.
        key: Tensor name used to identify the field in validation errors.

    Returns:
        The shape as an immutable tuple. An empty list is preserved as an
        empty tuple and represents a scalar tensor with one element.

    Raises:
        CheckpointCorruptError: If ``raw`` is not a list of integers, contains
            a negative dimension, or its element count exceeds the unsigned
            64-bit range.
    """
    if not isinstance(raw, list) or any(not isinstance(d, int) or isinstance(d, bool) for d in raw):
        raise CheckpointCorruptError(f"{key}: shape must be a list of ints, got {raw!r}")
    shape = tuple(int(d) for d in raw)
    if any(d < 0 for d in shape):
        raise CheckpointCorruptError(f"{key}: negative dim in shape {shape}")
    numel = math.prod(shape) if shape else 1
    if numel > _U64_MAX:
        raise CheckpointCorruptError(f"{key}: shape {shape} overflows a 64-bit element count")
    return shape

def _offsets(raw: object, key: str) -> tuple[int, int]:
    """Validate a tensor's half-open byte range in the safetensors payload.

    Safetensors stores ``data_offsets`` as ``[begin, end]`` relative to the
    beginning of the payload, which is the byte immediately after the eight
    byte header-length field and the encoded JSON header. This helper validates
    the range itself; checking it against the file size and checking for
    overlap with other tensors are responsibilities of the manifest builder.

    Args:
        raw: Raw ``data_offsets`` value decoded from the safetensors JSON
            header. It must be a two-item list of integer endpoints.
        key: Tensor name used to identify the field in validation errors.

    Returns:
        A ``(begin, end)`` tuple describing the half-open byte interval
        ``[begin, end)``. Both endpoints are normalized to plain ``int``
        values.

    Raises:
        CheckpointCorruptError: If the value is not a two-item integer list,
            an endpoint is negative, ``begin`` is greater than ``end``, or
            ``end`` does not fit in an unsigned 64-bit integer.
    """
    if not isinstance(raw, list) or len(raw) != 2:
        raise CheckpointCorruptError(f"{key}: data_offsets must be [begin, end], got {raw!r}")
    begin, end = raw
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in (begin, end)):
        raise CheckpointCorruptError(f"{key}: data_offsets must be integers, got {raw!r}")
    if begin < 0 or end < 0:
        raise CheckpointCorruptError(f"{key}: negative data_offset in {raw!r}")
    if begin > end:
        raise CheckpointCorruptError(f"{key}: data_offsets [{begin}, {end}) run backwards")
    if end > _U64_MAX:
        raise CheckpointCorruptError(f"{key}: data_offset {end} overflows u64")
    return int(begin), int(end)

def parse_safetensors_header(
    raw_header: bytes,
    *,
    file_uri: str,
    data_start: int,
    file_size: int,
) -> SafetensorsHeader:
    """Validate one header's JSON against the file it came from.

    Split from the reader so the whole check runs on a bytes literal — every
    corruption case in the test suite is a two-line construction rather than a
    file on disk.
    """
    
    try: 
        decoded = raw_header.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CheckpointCorruptError(f"{file_uri}: header is not valid UTF-8") from exc

    # object_pairs_hook, not the default: json silently keeps the last value for
    # a duplicated key, so `{"w": A, "w": B}` parses clean and one tensor
    # vanishes.  This is the only way to see it.
    seen: list[str] = []
    
    def _pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        seen.extend(k for k, _ in pairs)
        return dict(pairs)

    try: 
        header = json.loads(decoded, object_pairs_hook=_pairs_hook)
    except json.JSONDecodeError as exc:
        raise CheckpointCorruptError(f"{file_uri}: header is not valid JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise CheckpointCorruptError(
            f"{file_uri}: header is {type(header).__name__}, not an object"
        )
    
    metadata: tuple[tuple[str, str], ...] = ()
    meta = header.pop("__metadata__", None)
    if isinstance(meta, dict):
        metadata = tuple(sorted((str(k), str(v)) for k, v in meta.items()))

    # One pass, two sets.  `top_level.count(k)` inside a comprehension over the
    # same list is a scan per key — O(tensors^2), invisible on a 291-tensor
    # dense shard and 36M scans on a 6k-tensor MoE one.
    #
    # This runs on the key list captured by `object_pairs_hook` above, which is
    # the only place the duplicates are still visible: `json.loads` keeps the
    # last value for a repeated key, so by the time there is a dict the evidence
    # is gone.
    interesting = {k for k in header} | {"__metadata__"}
    first_seen: set[str] = set()
    duplicates: set[str] = set()
    duplicates: set[str] = set()
    for key in seen:
        if key not in interesting:
            continue
        if key in first_seen:
            duplicates.add(key)
        else:
            first_seen.add(key)
    dupes = sorted(duplicates)
    if dupes:
        raise CheckpointCorruptError(
            f"{file_uri}: duplicate tensor keys in header: {dupes} — json would have "
            "kept only the last and dropped the rest silently"
        )
        
    # Build the manifest entries.  The offsets are checked against the file size
    # and for overlap in the manifest builder, not here.
    entries: list[ManifestEntry] = []
    for key, value in header.items():
        if not isinstance(key, str):
            raise CheckpointCorruptError(f"{file_uri}: entry {key!r} is not a string")
        if not isinstance(value, dict):
            raise CheckpointCorruptError(f"{file_uri}: entry {key!r} is not an object")
        dtype = _dtype(value.get("dtype"), key)
        shape = _shape(value.get("shape"), key)
        begin_offset, end_offset = _offsets(value.get("data_offsets"), key)
        nbytes = end_offset - begin_offset
        declared_byte = dtype.nbytes(math.prod(shape) if shape else 1)
        if nbytes != declared_byte:
            raise CheckpointCorruptError(
                f"{file_uri}: entry {key!r} has {nbytes} bytes in data_offsets, "
                f"but shape {shape} and dtype {dtype} require {declared_byte}"
            )
        abs = data_start + begin_offset
        if abs + nbytes > file_size:
            raise CheckpointCorruptError(
                f"{file_uri}: entry {key!r} ends at {abs+nbytes}, which is beyond the "
                f"file size of {file_size}"
            )
        entries.append(
            ManifestEntry(
                tensor_key=key,
                file_uri=file_uri,
                dtype=dtype,
                shape=shape,
                byte_offset=abs,
                nbytes=nbytes,
            )
        )
    entries.sort(key=lambda e: e.byte_offset)
    prev: ManifestEntry | None = None
    for entry in entries: 
        if prev is not None and entry.byte_offset < prev.end_offset:
            raise CheckpointCorruptError(
                f"{file_uri}: {prev.tensor_key} [{prev.byte_offset}:"
                f"{prev.end_offset}) overlaps {entry.tensor_key} "
                f"[{entry.byte_offset}:{entry.end_offset})"
            )
        prev = entry
    
    return SafetensorsHeader(
        file_uri=file_uri,
        data_start=data_start,
        file_size=file_size,
        entries=tuple(entries),
        metadata=metadata,
    )

def read_safetensors_header(path: str | Path) -> SafetensorsHeader:
    """Read and validate one file's header.  Touches the first 8+N bytes only."""
    p = Path(path)
    try:
        size = p.stat().st_size
    except OSError as exc:
        raise CheckpointCorruptError(f"{p}: cannot stat: {exc}") from exc
    if size < 8:
        raise CheckpointCorruptError(f"{p}: {size} B is too small to hold a header length")

    try:
        with p.open("rb") as fh:
            raw_len = fh.read(8)
            (header_len,) = struct.unpack("<Q", raw_len)
            if header_len == 0:
                raise CheckpointCorruptError(f"{p}: header length is 0")
            if header_len > _HEADER_LENGTH_LIMIT:
                raise CheckpointCorruptError(
                    f"{p}: header claims {header_len} B (limit {_HEADER_LENGTH_LIMIT}) — "
                    "refusing to allocate on a forged length"
                )
            if 8 + header_len > size:
                raise CheckpointCorruptError(
                    f"{p}: header claims {header_len} B but the file is {size} B"
                )
            raw_header = fh.read(header_len)
    except OSError as exc:
        raise CheckpointCorruptError(f"{p}: cannot read header: {exc}") from exc

    if len(raw_header) != header_len:
        raise CheckpointCorruptError(
            f"{p}: header truncated, read {len(raw_header)} of {header_len} B"
        )
    return parse_safetensors_header(
        raw_header, file_uri=str(p), data_start=8 + header_len, file_size=size
    )
