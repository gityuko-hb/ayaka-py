"""Minimal in-repo sampling runner (K06 P0).

The paged model runner is not in this tree yet, so this module provides the
smallest complete bridge between ``BatchStepPlan`` and ``ayaka.sampling``:

  * a ``LogitsProvider`` supplies raw ``[num_sampling_rows, vocab]`` logits in
    packed sampling order -- the order of ``BatchStepPlan.sampling_rows``;
  * ``SamplingRunner`` flushes the coordinator (dirty columns + active-row map)
    and samples one token per row, returning ids in that same order;
  * ``SamplingExecutor`` is a minimal ``Executor`` whose backend hook runs the
    runner and publishes the packed samples via ``set_samples``.

A future GPU runner replaces the provider with the model forward; the
flush-before-read ordering and the ``set_samples`` contract stay identical.
"""

from __future__ import annotations

from typing import Protocol

import torch

from ayaka.executor.base import Executor
from ayaka.executor.ticket import ExecutionTicket, FenceResult, WorkState
from ayaka.sampling.engine import SamplingCoordinator
from ayaka.sched.plan import BatchStepPlan, PreparedStep

__all__ = ["LogitsProvider", "SamplingExecutor", "SamplingRunner"]


class LogitsProvider(Protocol):
    """Raw logits for the step's sampling rows, in packed sampling order."""

    def __call__(self, step: BatchStepPlan) -> torch.Tensor: ...


class SamplingRunner:
    """Sample one token per sampling row for a prepared step."""

    __slots__ = ("_closed", "_coordinator", "_force_reference", "_provider")

    def __init__(
        self,
        coordinator: SamplingCoordinator,
        logits_provider: LogitsProvider,
        *,
        force_reference: bool = False,
    ) -> None:
        self._coordinator = coordinator
        self._provider = logits_provider
        self._force_reference = force_reference
        self._closed = False

    def __call__(self, prepared: PreparedStep) -> tuple[int, ...]:
        if self._closed:
            raise RuntimeError("sampling runner is closed")
        plan = prepared.step.sampling
        logits = self._provider(prepared.step)
        if not isinstance(logits, torch.Tensor):
            raise TypeError("logits provider must return a torch.Tensor")
        if logits.dim() != 2 or logits.size(0) != plan.num_rows:
            raise ValueError(
                f"logits must be [num_sampling_rows={plan.num_rows}, vocab], "
                f"got {tuple(logits.shape)}"
            )
        self._coordinator.flush()
        return self._coordinator.sample(logits, plan, force_reference=self._force_reference)

    def close(self) -> None:
        self._closed = True


class _DoneFence:
    """Immediate success fence: P0 sampling completes synchronously on host."""

    __slots__ = ()

    def query(self) -> FenceResult:
        return FenceResult(WorkState.SUCCEEDED, quiescent=True)


class SamplingExecutor(Executor):
    """Minimal executor that runs ``SamplingRunner`` inside ``_enqueue``.

    Deliberately not a model executor: it exists so wired sampling can complete
    real tickets until the paged runtime lands, and so tests exercise the same
    executor contract (``set_samples`` + fences) as production.
    """

    __slots__ = ("runner",)

    def __init__(self, runner: SamplingRunner, *, max_inflight: int = 1) -> None:
        super().__init__(max_inflight=max_inflight)
        self.runner = runner

    def _enqueue(self, ticket: ExecutionTicket) -> None:
        samples = self.runner(ticket.prepared)
        self.set_samples(ticket, samples)
        self.track_fence(ticket, _DoneFence())

    def _begin_drain(self, ticket: ExecutionTicket) -> _DoneFence:
        return _DoneFence()

    def _close(self) -> None:
        self.runner.close()
