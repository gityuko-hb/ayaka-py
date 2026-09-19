"""Identity of one captured decode graph (R08).

A captured graph may only replay while the exact resource incarnation it was
captured against is still live: owner incarnation, model, KV storage,
workspace, buffer pool and backend. The identity is deliberately content-free —
block-table values, sequence lengths and prefix hits may all change between
replays, because those are staged into the same backing pointers rather than
baked into the capture.

``slot_index`` names the flight slot whose backing the graph bound. A graph
captured against one slot's buffers must never replay against another's, so
the slot is part of the identity rather than a detail of the pool that owns it.
"""

from __future__ import annotations

from dataclasses import dataclass

from ayaka.memory.capacity import ResourceGeneration
from ayaka.types import ForwardMode
from ayaka.utils.validation import require_int, require_text

__all__ = ["GraphIdentity", "graph_identity_key"]


def graph_identity_key(
    *,
    generation: ResourceGeneration,
    bucket: int,
    mode: ForwardMode,
    dtype: str,
    backend: str,
) -> str:
    """Content-free key for one decode bucket under one resource generation.

    Stable across replays with different table values/lengths/prefix hits;
    changes whenever any generation component or the padded bucket changes.
    """
    require_int(bucket, "bucket", minimum=1)
    if not isinstance(mode, ForwardMode):
        raise TypeError("mode must be a ForwardMode")
    require_text(dtype, "dtype")
    require_text(backend, "backend")
    if not isinstance(generation, ResourceGeneration):
        raise TypeError("generation must be a ResourceGeneration")
    return (
        f"decode:{backend}:{dtype}:b{bucket}:{mode.name.lower()}:"
        f"own{generation.owner_incarnation}:ws{generation.workspace}:"
        f"buf{generation.buffers}"
    )


@dataclass(frozen=True, slots=True)
class GraphIdentity:
    """One captured ``(bucket, flight slot)`` under one resource generation."""

    generation: ResourceGeneration
    bucket: int
    mode: ForwardMode
    dtype: str
    backend: str
    slot_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.generation, ResourceGeneration):
            raise TypeError("generation must be a ResourceGeneration")
        require_int(self.bucket, "bucket", minimum=1)
        if not isinstance(self.mode, ForwardMode):
            raise TypeError("mode must be a ForwardMode")
        if self.mode is not ForwardMode.DECODE:
            raise ValueError("captured decode graphs serve pure decode only")
        require_text(self.dtype, "dtype")
        require_text(self.backend, "backend")
        require_int(self.slot_index, "slot_index")

    @property
    def key(self) -> str:
        """Identity key including the flight slot the capture bound."""
        base = graph_identity_key(
            generation=self.generation,
            bucket=self.bucket,
            mode=self.mode,
            dtype=self.dtype,
            backend=self.backend,
        )
        return f"{base}:slot{self.slot_index}"

    def same_binding(self, other: GraphIdentity) -> bool:
        """Whether both identities name the same generation/bucket/slot."""
        if not isinstance(other, GraphIdentity):
            return NotImplemented
        return (
            self.generation == other.generation
            and self.bucket == other.bucket
            and self.mode is other.mode
            and self.dtype == other.dtype
            and self.backend == other.backend
            and self.slot_index == other.slot_index
        )
