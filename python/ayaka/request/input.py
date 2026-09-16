from __future__ import annotations

import json
import math
from dataclasses import dataclass

from ayaka.utils.validation import require_frozen, require_int, require_text


@dataclass(frozen=True, slots=True)
class ConstraintSpec:
    """Canonical JSON constraint; mutable matcher state belongs to the runner."""

    kind: str
    schema_json: str = ""

    def __post_init__(self) -> None:
        if self.kind not in ("json_object", "json_schema", "tool_calls"):
            raise ValueError("unknown constraint kind")
        if self.schema_json:
            value = json.loads(self.schema_json)
            if not isinstance(value, dict):
                raise ValueError("constraint schema must be a JSON object")
        elif self.kind != "json_object":
            raise ValueError("a schema is required")


@dataclass(frozen=True, slots=True)
class MultimodalEmbedding:
    """Owned embedding rows replacing prompt placeholders at absolute positions.

    Image/audio encoders run before admission. The immutable digest participates
    in prefix identity; the engine never downloads media or loads an encoder.
    """

    modality: str
    start: int
    rows: tuple[tuple[float, ...], ...]
    digest: str

    def __post_init__(self) -> None:
        require_frozen(self, "multimodal embedding")
        require_int(self.start, "embedding start")
        require_text(self.digest, "embedding digest")
        if self.modality not in ("image", "audio"):
            raise ValueError("embedding modality must be image or audio")
        if not self.rows or not self.rows[0]:
            raise ValueError("embedding rows must be nonempty")
        width = len(self.rows[0])
        if any(len(row) != width or any(not math.isfinite(x) for x in row) for row in self.rows):
            raise ValueError("embedding rows must have equal width and finite values")
