"""Per-role schedulers built on the shared continuous-batching machinery.

Roles change *what gets batched*, not *how policy works*: waiting order comes
from :mod:`ayaka.sched.policy` and preemption from
:mod:`ayaka.sched.preemption` unchanged. ``PrefillScheduler`` never schedules
a decode step; ``DecodeScheduler`` only builds decode steps (mixed batches
off, prefill of remote-KV prompts happens through the normal preparer gate).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

from ayaka.request.states import RequestState
from ayaka.sched.budget import BatchBudget
from ayaka.sched.continuous import ContinuousScheduler

if TYPE_CHECKING:
    from ayaka.sched.plan import BatchStepPlan

__all__ = ["DecodeScheduler", "PrefillScheduler"]


class PrefillScheduler(ContinuousScheduler):
    """Prefill-node role: prefill-only batches; decode never scheduled here.

    A completed prompt stays parked in the running set until the fabric
    hands its KV to the decode node (WS5 transport); the decode ring is
    deliberately left unscheduled so the node never spends compute on decode.
    """

    def _candidate_plans(self) -> Iterator[BatchStepPlan]:
        plan = self._build_prefill_only()
        if plan is not None:
            yield plan

    @property
    def prefill_only(self) -> bool:
        return True


class DecodeScheduler(ContinuousScheduler):
    """Decode-node role: decode-only batches; mixed batches disabled.

    New prompts reach the decode ring only when their KV is fully resident —
    transferred or fully cached — through the preparer gate; this node never
    spends steps on partial prompt chunks.
    """

    def _candidate_plans(self) -> Iterator[BatchStepPlan]:
        plan = self._build_decode_only()
        if plan is not None:
            yield plan

    def _append_decodes(
        self,
        budget: BatchBudget,
        slices: list,
        inputs: list,
    ) -> None:
        # Fold transferred prompts into the decode ring once: the whole
        # prompt KV landed remotely and the prefill node's first sample was
        # published through the output plane, so a decode step is first work.
        for entry in self._iter_waiting():
            lifecycle = entry.lifecycle
            if lifecycle.state is not RequestState.ADMITTED:
                continue
            if lifecycle.request_id in self._inflight_ids_all:
                continue
            snapshot = lifecycle.snapshot()
            if snapshot.computed_tokens < snapshot.prompt_tokens:
                continue  # prompt compute belongs to the prefill node
            if len(snapshot.known_tokens) <= snapshot.prompt_tokens:
                continue  # the transferred first sample has not arrived yet
            self._running_add(lifecycle.request_id, lifecycle)
        super()._append_decodes(budget, slices, inputs)

    @property
    def decode_only(self) -> bool:
        return True
