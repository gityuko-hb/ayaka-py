from __future__ import annotations

import json
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ayaka.types import DType

__all__ = [
    "DUMMY_ARCH_JSON",
    "TensorSeed",
    "break_offsets",
    "duplicate_tensor_key",
    "overlap_offsets",
    "truncate_header",
    "truncate_payload",
    "write_config",
    "write_sharded_checkpoint",
    "write_single_checkpoint",
    "wrong_shape",
]

_DTYPE_TOKENS: dict[DType, str] = {
    DType.FP32: "F32",
    DType.FP16: "F16",
    DType.BF16: "BF16",
    DType.INT64: "I64",
    DType.INT32: "I32",
    DType.INT8: "I8",
    DType.UINT8: "U8",
    DType.BOOL: "BOOL",
}

# A tiny but *structurally real* Qwen2: 2 layers, GQA 4:2, tied embeddings.
# Real enough that the expected-schema generator produces the same shape of
# output it does for the 0.5B, small enough that a checkpoint is ~100 KB.
DUMMY_ARCH_JSON: dict[str, Any] = {
    "architectures": ["Qwen2ForCausalLM"],
    "model_type": "qwen2",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "vocab_size": 128,
    "max_position_embeddings": 512,
    "rms_norm_eps": 1e-06,
    "rope_theta": 10000.0,
    "tie_word_embeddings": True,
    "torch_dtype": "bfloat16",
}


@dataclass(frozen=True, slots=True)
class TensorSeed:
    """One tensor to write.  Content is deterministic from ``fill``.

    Deterministic rather than random so a materialization test can assert the
    exact bytes that arrived on the device, which is what turns "the load
    succeeded" into "the load was correct".
    """

    name: str
    shape: tuple[int, ...]
    dtype: DType = DType.BF16
    fill: int = 0

    @property
    def numel(self) -> int:
        return math.prod(self.shape) if self.shape else 1

    @property
    def nbytes(self) -> int:
        return self.dtype.nbytes(self.numel)

    def payload(self) -> bytes:
        """A repeating byte pattern keyed to ``fill``.

        Not zeros: a zero-filled buffer is indistinguishable from an allocation
        that was never written, so a transform bug that drops the copy would
        pass a zero-based check.
        """
        pattern = bytes(((self.fill + i) & 0xFF) or 0x5A for i in range(7))
        reps = -(-self.nbytes // len(pattern))
        return (pattern * reps)[: self.nbytes]


def _header_json(seeds: Sequence[TensorSeed], metadata: Mapping[str, str] | None) -> bytes:
    header: dict[str, Any] = {}
    if metadata:
        header["__metadata__"] = dict(metadata)
    offset = 0
    for seed in seeds:
        token = _DTYPE_TOKENS.get(seed.dtype)
        if token is None:
            raise ValueError(f"dummy writer has no safetensors token for {seed.dtype.label}")
        header[seed.name] = {
            "dtype": token,
            "shape": list(seed.shape),
            "data_offsets": [offset, offset + seed.nbytes],
        }
        offset += seed.nbytes
    return json.dumps(header, separators=(",", ":")).encode("utf-8")


def _write_file(
    path: Path, seeds: Sequence[TensorSeed], metadata: Mapping[str, str] | None
) -> Path:
    raw = _header_json(seeds, metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(raw)))
        fh.write(raw)
        for seed in seeds:
            fh.write(seed.payload())
    return path


def write_config(root: Path, overrides: Mapping[str, Any] | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    cfg = {**DUMMY_ARCH_JSON, **(overrides or {})}
    path = root / "config.json"
    path.write_text(json.dumps(cfg, indent=2))
    return path


def write_single_checkpoint(
    root: Path,
    seeds: Sequence[TensorSeed],
    *,
    filename: str = "model.safetensors",
    metadata: Mapping[str, str] | None = None,
) -> Path:
    return _write_file(root / filename, seeds, metadata)


def write_sharded_checkpoint(
    root: Path,
    shards: Sequence[Sequence[TensorSeed]],
    *,
    metadata: Mapping[str, Any] | None = None,
    weight_map_override: Mapping[str, str] | None = None,
) -> Path:
    """Write N shards plus the index that maps every tensor to its file.

    ``weight_map_override`` exists so a test can point a weight at
    ``../../../etc/shadow`` without hand-writing the whole index — the dummy writer is not a
    security tool, it just needs to be able to simulate a weight that is not in the checkpoint.
    """
    root.mkdir(parents=True, exist_ok=True)
    weight_map: dict[str, str] = {}
    total = 0
    for i, shard in enumerate(shards):
        name = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        _write_file(root / name, shard, None)
        for seed in shard:
            weight_map[seed.name] = name
            total += seed.nbytes
    index = {
        "metadata": {"total_size": total, **(metadata or {})},
        "weight_map": dict(weight_map_override) if weight_map_override else weight_map,
    }
    path = root / "model.safetensors.index.json"
    path.write_text(json.dumps(index, indent=2))
    return path


@dataclass(frozen=True, slots=True)
class _Parts:
    header_len: int
    header: bytes
    payload: bytes
    obj: dict[str, Any] = field(default_factory=dict)


def _split(path: Path) -> _Parts:
    raw = path.read_bytes()
    (n,) = struct.unpack("<Q", raw[:8])
    header = raw[8 : 8 + n]
    return _Parts(n, header, raw[8 + n :], json.loads(header))


def _reassemble(path: Path, obj: dict[str, Any], payload: bytes) -> Path:
    raw = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
    return path


def truncate_header(path: Path, *, claim: int | None = None) -> Path:
    """Header length says more than the file holds.

    The default is derived from the *file* size, not the header size: a header
    length of 4× the real header still lands inside a checkpoint whose payload
    is megabytes, and the parser would then read payload bytes as JSON and
    report a parse error instead of the bounds violation under test.
    """
    parts = _split(path)
    total = 8 + parts.header_len + len(parts.payload)
    claimed = claim if claim is not None else total + 1024
    path.write_bytes(struct.pack("<Q", claimed) + parts.header + parts.payload)
    return path


def truncate_payload(path: Path, *, keep: int = 8) -> Path:
    """The classic interrupted download: header intact, bytes missing."""
    parts = _split(path)
    raw = json.dumps(parts.obj, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + parts.payload[:keep])
    return path


def break_offsets(path: Path, tensor: str, *, begin: int, end: int) -> Path:
    parts = _split(path)
    parts.obj[tensor]["data_offsets"] = [begin, end]
    return _reassemble(path, parts.obj, parts.payload)


def overlap_offsets(path: Path, first: str, second: str) -> Path:
    """Point ``second`` at bytes ``first`` already owns."""
    parts = _split(path)
    parts.obj[second]["data_offsets"] = list(parts.obj[first]["data_offsets"])
    parts.obj[second]["shape"] = list(parts.obj[first]["shape"])
    parts.obj[second]["dtype"] = parts.obj[first]["dtype"]
    return _reassemble(path, parts.obj, parts.payload)


def wrong_shape(path: Path, tensor: str, shape: Sequence[int]) -> Path:
    """Shape that no longer multiplies out to the declared byte span."""
    parts = _split(path)
    parts.obj[tensor]["shape"] = list(shape)
    return _reassemble(path, parts.obj, parts.payload)


def duplicate_tensor_key(path: Path, tensor: str) -> Path:
    """Two entries with the same key — legal JSON, and ``json.loads`` keeps one.

    Written textually because there is no way to express it through a dict.
    """
    parts = _split(path)
    text = json.dumps(parts.obj, separators=(",", ":"))
    entry = json.dumps({tensor: parts.obj[tensor]}, separators=(",", ":"))[1:-1]
    injected = text[:-1] + "," + entry + "}"
    raw = injected.encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + parts.payload)
    return path
