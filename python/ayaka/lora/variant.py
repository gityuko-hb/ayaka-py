"""Bounded model structure and immutable adapter semantics (EP3)."""

import hashlib
import json
from dataclasses import dataclass

from ayaka.utils.validation import require_int, require_text


@dataclass(frozen=True, slots=True)
class AdapterIdentity:
    """Content revision, not a display label; included in prefix identity."""

    name: str
    revision: str

    def __post_init__(self) -> None:
        require_text(self.name, "adapter name")
        require_text(self.revision, "adapter revision")

    @property
    def prefix_key(self) -> str:
        return hashlib.sha256(json.dumps((self.name, self.revision)).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class ExecutionVariant:
    """One padded-rank route, shared by all adapter selections in stable slots."""

    modules: tuple[str, ...]
    rank_capacity: int
    adapter_capacity: int
    dtype: str
    route: str = "dense_bmm"

    def __post_init__(self) -> None:
        require_int(self.rank_capacity, "rank_capacity", minimum=1)
        require_int(self.adapter_capacity, "adapter_capacity", minimum=1)
        if not self.modules or tuple(sorted(set(self.modules))) != self.modules:
            raise ValueError("variant modules must be nonempty, sorted and unique")
        if self.rank_capacity > 64 or self.adapter_capacity > 16:
            raise ValueError("EP3 LoRA supports ranks 1..64 and at most 16 adapters")
        if self.dtype not in ("float32", "float16", "bfloat16") or self.route != "dense_bmm":
            raise ValueError("unsupported LoRA dtype or kernel route")
