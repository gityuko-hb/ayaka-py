from __future__ import annotations

import abc


class Executor(abc.ABC):
    """Rule 1 — executes plans, decides nothing."""

    @abc.abstractmethod
    def execute(self, plan) -> None: ...

    @abc.abstractmethod
    def initialize(self) -> None:
        """Create contexts, streams, pools.  Separate from ``__init__`` so a
        rank can be constructed in the parent process and initialised in the
        child, after the CUDA fork boundary."""

    @abc.abstractmethod
    def shutdown(self) -> None:
        """Teardown order is: drain streams → destroy graphs → free pools →
        destroy context.  Freeing a pool while a graph still references its
        buffers is a segfault at interpreter exit, not an error."""

    @abc.abstractmethod
    def profile_memory(self) -> int:
        """Peak activation bytes for a worst-case step.  This is what turns into
        the KV block count at bootstrap."""