"""Minimal in-repo sampling runner (K06 P0).

The paged model runner is not in this tree yet, so this module provides the
smallest complete bridge between ``BatchStepPlan`` and ``ayaka.sampling``:

  * a ``LogitsProvider`` supplies raw ``[num_sampling_rows, vocab]`` logits in
    packed sampling order -- the order of ``BatchStepPlan.sampling_rows``;
  * ``SamplingRunner`` flushes the coordinator (dirty columns + active-row map)
    and samples one token per row, returning device-resident ``SampleOutputs``;
  * ``SamplingExecutor`` is a minimal ``Executor`` whose backend hook runs the
    runner and publishes the packed samples via ``set_samples``.

A future GPU runner replaces the provider with the model forward; the
flush-before-read ordering and the ``set_samples`` contract stay identical.
Host materialization happens only at the completion boundary.
"""

from __future__ import annotations

from typing import Protocol

import torch

from ayaka.executor.base import Executor
from ayaka.executor.ticket import ExecutionTicket, FenceResult, SampleOutputs, WorkState
from ayaka.sampling.bans import BanApplier, BansProvider, LogitFillBanApplier
from ayaka.sampling.engine import SamplingCoordinator
from ayaka.sampling.logprobs import LogprobMode
from ayaka.sched.plan import BatchStepPlan, PreparedStep
from ayaka.utils.import_utils import CapabilityError

__all__ = ["LogitsProvider", "SampleRunner", "SamplingExecutor", "SamplingRunner"]


class LogitsProvider(Protocol):
    """Raw logits for the step's sampling rows, in packed sampling order."""

    def __call__(self, step: BatchStepPlan) -> torch.Tensor: ...


class SampleRunner(Protocol):
    """Executor-facing runner contract shared by provider and model runners."""

    def __call__(self, prepared: PreparedStep) -> SampleOutputs: ...

    def close(self) -> None: ...


class SamplingRunner:
    """Sample one token per sampling row for a prepared step."""

    __slots__ = (
        "_ban_applier",
        "_bans",
        "_closed",
        "_coordinator",
        "_force_reference",
        "_provider",
    )

    def __init__(
        self,
        coordinator: SamplingCoordinator,
        logits_provider: LogitsProvider,
        *,
        force_reference: bool = False,
        bans: BansProvider | None = None,
        ban_applier: BanApplier | None = None,
    ) -> None:
        self._coordinator = coordinator
        self._provider = logits_provider
        self._force_reference = force_reference
        self._bans = bans
        self._ban_applier = ban_applier if ban_applier is not None else LogitFillBanApplier()
        self._closed = False

    def set_bans(self, bans: BansProvider | None) -> None:
        """Install the per-step ban provider (the engine's sampling masks)."""
        self._bans = bans

    def _apply_bans(self, step: BatchStepPlan, logits: torch.Tensor, rows: int) -> None:
        if self._bans is None:
            return
        bans = self._bans(step)
        if len(bans) != rows:
            raise ValueError(f"ban provider returned {len(bans)} sets for {rows} sampling rows")
        self._ban_applier.apply(logits, bans)

    def __call__(self, prepared: PreparedStep) -> SampleOutputs:
        if self._closed:
            raise RuntimeError("sampling runner is closed")
        step = prepared.step
        plan = step.sampling
        if step.prompt_logprobs:
            raise CapabilityError(
                "prompt_logprobs",
                detail=(
                    "the provider-based sampling runner cannot produce prompt "
                    "logprobs; it never sees model hidden states"
                ),
                remedy="wire the model runner (ModelSamplingRunner) for prompt scoring",
            )
        for scheduled in step.slices:
            if not scheduled.sample_last_query:
                continue
            if (
                self._coordinator.logprob_k_for(scheduled.request_id) is not None
                and self._coordinator.logprob_mode_for(scheduled.request_id) is LogprobMode.SAMPLING
            ):
                raise CapabilityError(
                    "logprob_mode",
                    detail=(
                        "the provider-based sampling runner cannot report "
                        "sampling-distribution logprobs; it never sees the model's "
                        "pre-transform logits"
                    ),
                    remedy="wire the model runner (ModelSamplingRunner) for logprobs",
                )
            if self._coordinator.token_ids_logprobs_for(scheduled.request_id) is not None:
                raise CapabilityError(
                    "token_ids_logprobs",
                    detail=(
                        "the provider-based sampling runner cannot score specific "
                        "token logprobs; it never sees the model's pre-transform logits"
                    ),
                    remedy="wire the model runner (ModelSamplingRunner) for logprobs",
                )
        logits = self._provider(prepared.step)
        if not isinstance(logits, torch.Tensor):
            raise TypeError("logits provider must return a torch.Tensor")
        if logits.dim() != 2 or logits.size(0) != plan.num_rows:
            raise ValueError(
                f"logits must be [num_sampling_rows={plan.num_rows}, vocab], "
                f"got {tuple(logits.shape)}"
            )
        self._apply_bans(prepared.step, logits, plan.num_rows)
        self._coordinator.flush()
        token_ids, support = self._coordinator.sample_with_support(
            logits, plan, force_reference=self._force_reference
        )
        return SampleOutputs(token_ids=token_ids, sampling_support=support)

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

    def __init__(self, runner: SampleRunner, *, max_inflight: int = 1) -> None:
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
