"""Scoped attention binding for one synchronous Python forward owner."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from types import MappingProxyType

import torch

from ayaka.attention.base import BaseAttentionBackend
from ayaka.attention.metadata import BaseAttentionMetadata


@dataclass(frozen=True, slots=True)
class ForwardContext:
    """References to the same cache-bound backends/builders the runner owns."""

    backends: Mapping[str, BaseAttentionBackend]
    layers: Mapping[int, tuple[str, int]]
    metadata: Mapping[str, BaseAttentionMetadata]

    def __post_init__(self) -> None:
        for name in ("backends", "layers", "metadata"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))
        if set(self.backends) != set(self.metadata):
            raise ValueError("context metadata must match the attention binding")
        if any(group not in self.backends for group, _ in self.layers.values()):
            raise ValueError("context layer names an unbound attention group")

    def attention(
        self, layer: int, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        name, local = self.layers[layer]
        return self.backends[name].forward(local, query, key, value, self.metadata[name])


_current: ContextVar[ForwardContext | None] = ContextVar("ayaka_forward_context", default=None)


def get_forward_context() -> ForwardContext:
    context = _current.get()
    if context is None:
        raise RuntimeError("no active forward context")
    return context


@contextmanager
def forward_context(context: ForwardContext) -> Iterator[ForwardContext]:
    """Task/thread-local lookup; mutable backend isolation remains the lane's job."""
    token = _current.set(context)
    try:
        yield context
    finally:
        _current.reset(token)
