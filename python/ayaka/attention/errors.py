from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

__all__ = [
    "AttentionErrorCode",
    "AttentionError",
    "BackendSelectionError",
    "BackendCapabilityError",
    "AttentionMetadataError",
    "GraphStateError",
    "BackendRejection",
]


class AttentionErrorCode(StrEnum):
    # --- selection ---
    BACKEND_UNKNOWN = "BACKEND_UNKNOWN"
    BACKEND_NO_CANDIDATE = "BACKEND_NO_CANDIDATE"
    BACKEND_DUPLICATE_REGISTRATION = "BACKEND_DUPLICATE_REGISTRATION"
    # --- capability (why one candidate lost) ---
    UNSUPPORTED_ATTN_TYPE = "UNSUPPORTED_ATTN_TYPE"
    UNSUPPORTED_PAGE_SIZE = "UNSUPPORTED_PAGE_SIZE"
    UNSUPPORTED_HEAD_DIM = "UNSUPPORTED_HEAD_DIM"
    UNSUPPORTED_KV_LAYOUT = "UNSUPPORTED_KV_LAYOUT"
    UNSUPPORTED_SLIDING_WINDOW = "UNSUPPORTED_SLIDING_WINDOW"
    UNSUPPORTED_LOGITS_SOFT_CAP = "UNSUPPORTED_LOGITS_SOFT_CAP"
    UNSUPPORTED_SINKS = "UNSUPPORTED_SINKS"
    UNSUPPORTED_DTYPE = "UNSUPPORTED_DTYPE"
    UNSUPPORTED_KV_CACHE_DTYPE = "UNSUPPORTED_KV_CACHE_DTYPE"
    UNSUPPORTED_NON_CAUSAL = "UNSUPPORTED_NON_CAUSAL"
    UNSUPPORTED_TREE_MASK = "UNSUPPORTED_TREE_MASK"
    MISSING_MLA_EXTRAS = "MISSING_MLA_EXTRAS"
    UNSUPPORTED_ARCH = "UNSUPPORTED_ARCH"
    MISSING_DEPENDENCY = "MISSING_DEPENDENCY"
    UNSUPPORTED_SPEC_DECODE = "UNSUPPORTED_SPEC_DECODE"
    # --- metadata ---
    METADATA_GROUP_MISSING = "METADATA_GROUP_MISSING"
    METADATA_TYPE_MISMATCH = "METADATA_TYPE_MISMATCH"
    METADATA_DEVICE_MISMATCH = "METADATA_DEVICE_MISMATCH"
    METADATA_SHAPE_MISMATCH = "METADATA_SHAPE_MISMATCH"
    MISSING_KV_SCALE = "MISSING_KV_SCALE"
    SPEC_TREE_TOO_WIDE = "SPEC_TREE_TOO_WIDE"
    # --- cuda graph ---
    GRAPH_NOT_INITIALIZED = "GRAPH_NOT_INITIALIZED"
    GRAPH_ALREADY_INITIALIZED = "GRAPH_ALREADY_INITIALIZED"
    GRAPH_BATCH_SIZE_UNSUPPORTED = "GRAPH_BATCH_SIZE_UNSUPPORTED"
    GRAPH_MODE_UNSUPPORTED = "GRAPH_MODE_UNSUPPORTED"
    GRAPH_STAGING_WIDTH_EXCEEDED = "GRAPH_STAGING_WIDTH_EXCEEDED"


class AttentionError(RuntimeError):
    """Base for every refusal this package raises.

    ``context`` is structured so a caller (or a test) can assert on the cause without
    string-matching the message. ``remedy`` mirrors ``CapabilityError``: every refusal in
    this codebase tells the operator what to do next, not just what went wrong.
    """

    def __init__(
        self,
        code: AttentionErrorCode,
        message: str,
        *,
        remedy: str = "",
        **context: Any,
    ) -> None:
        self.code = code
        self.remedy = remedy
        self.context: dict[str, Any] = context
        detail = " ".join(f"{k}={v!r}" for k, v in context.items())
        super().__init__(
            f"[{code.value}] {message}"
            + (f" ({detail})" if detail else "")
            + (f" -- {remedy}" if remedy else "")
        )


class BackendSelectionError(AttentionError):
    """No backend can serve a group, or a named backend does not exist."""


class BackendCapabilityError(AttentionError):
    """A specific backend cannot serve a specific group/spec."""


class AttentionMetadataError(AttentionError):
    """A Attention metadata bundle is missing a group, or holds the wrong concrete type."""


class GraphStateError(AttentionError):
    """CUDA-graph lifecycle misuse or a staging buffer that cannot hold this step."""


@dataclass(frozen=True)
class BackendRejection:
    """Why one candidate backend lost, as data.

    The selector collects these for every candidate it walks and attaches the whole list to
    ``BACKEND_NO_CANDIDATE``. An operator debugging "why did it pick triton on my H100"
    gets the full decision trace instead of having to reconstruct it from log lines.
    """

    backend: str
    code: AttentionErrorCode
    detail: str
    context: dict[str, Any] = field(default_factory=dict)
    remedy: str = ""

    def __str__(self) -> str:
        tail = f" [{self.remedy}]" if self.remedy else ""
        return f"{self.backend}: {self.code.value} -- {self.detail}{tail}"


def rejections_to_lines(rejections: tuple[BackendRejection, ...]) -> str:
    return "\n".join(f"  - {r}" for r in rejections)
