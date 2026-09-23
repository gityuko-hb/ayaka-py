"""Parallel-sampling expansion shared by the scheduler and the engine façade.

"Expand at admission": an n>1 request becomes n independent children with
distinct ids and derived seeds — never forked mid-flight. Prompt KV is
duplicated per child (prefix sharing for children is a paged-runtime backlog
item). Both the scheduler's admission fan-out and the engine's output
registration must derive the exact same child set, so the helpers live here.
"""

from __future__ import annotations

from dataclasses import replace

from ayaka.request.schema import Request, RequestId

__all__ = ["child_request_id", "expand_parallel_request", "ParentRequestRegistry"]

#: Separator for child ids; a child id is ``f"{parent}#c{index}"``. Never
#: parsed back out of a string — the parent registry is the authority.
_CHILD_MARK = "#c"


def child_request_id(parent_id: str, index: int) -> str:
    return f"{parent_id}{_CHILD_MARK}{index}"


def expand_parallel_request(request: Request) -> tuple[Request, ...]:
    """Expand one n>1 request into n n=1 children with derived seeds.

    Every other field (cache hints, tenant, priority, constraint, ...) is
    preserved so children schedule and cache identically to the parent.
    """
    n = request.sampling.n
    if n <= 1:
        raise ValueError("expand_parallel_request requires sampling.n > 1")
    return tuple(
        replace(
            request,
            request_id=RequestId(child_request_id(str(request.request_id), index)),
            sampling=replace(
                request.sampling,
                n=1,
                seed=None if request.sampling.seed is None else request.sampling.seed + index,
            ),
        )
        for index in range(n)
    )


class ParentRequestRegistry:
    """Parent→children bookkeeping for admission fan-out.

    Owned by the scheduler core (single thread); entries are removed when the
    last child leaves scheduler tracking, and can be discarded explicitly for
    admission rollback.
    """

    def __init__(self) -> None:
        self._children: dict[str, tuple[str, ...]] = {}
        self._remaining: dict[str, set[str]] = {}
        self._parent_of: dict[str, str] = {}

    def register(self, parent_id: str, children: tuple[str, ...]) -> None:
        if not children:
            raise ValueError("a parallel family needs at least one child")
        if parent_id in self._children:
            raise ValueError(f"parallel family {parent_id!r} already registered")
        self._children[parent_id] = children
        self._remaining[parent_id] = set(children)
        for child_id in children:
            self._parent_of[child_id] = parent_id

    def unregister(self, parent_id: str) -> None:
        for child_id in self._children.pop(parent_id, ()):
            existing = self._parent_of.get(child_id)
            if existing == parent_id:
                del self._parent_of[child_id]
        self._remaining.pop(parent_id, None)

    def child_finished(self, child_id: str) -> str | None:
        """Record one settled child; return the parent once all are done.

        The registry is NOT auto-unregistered: the caller unregisters when it
        has consumed the family, so late events for settled children still
        resolve.
        """
        parent_id = self._parent_of.get(child_id)
        if parent_id is None:
            return None
        remaining = self._remaining.get(parent_id)
        if remaining is None or remaining.discard(child_id) is None and remaining:
            return None
        return parent_id

    def children_of(self, parent_id: str) -> tuple[str, ...] | None:
        return self._children.get(parent_id)

    def parent_of(self, child_id: str) -> str | None:
        return self._parent_of.get(child_id)
