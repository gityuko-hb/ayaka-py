"""In-process pipeline (PP) microbatch planning.

The planner is host-only: it freezes how one adopted batch step splits into
per-microbatch plans and which SEND/RECV messages cross each stage boundary,
so a pipeline executor cannot drift from the scheduler's batch geometry.
Executing the stage loop and moving boundary tensors is the pipeline
runtime's milestone; this module only owns the plan geometry and the
identity/communication annotations ``BatchStepPlan`` already declares.
"""

from __future__ import annotations

from dataclasses import replace

from ayaka.configs.parallel import ResolvedParallelPlan
from ayaka.plan import (
    EMPTY_COMMUNICATION_PLAN,
    CommOp,
    CommOpKind,
    CommunicationPlan,
    SamplingPlan,
)
from ayaka.sched.plan import (
    BatchStepPlan,
    DistributedStepIdentity,
    KVRequirement,
)
from ayaka.types import DType, StreamRole
from ayaka.utils.validation import require_int

__all__ = ["PpMicrobatchPlanner", "stage_spans"]


def stage_spans(parallel: ResolvedParallelPlan) -> tuple[tuple[int, int], ...]:
    """Half-open ``(layer_start, layer_end)`` per PP rank, stage order."""
    return tuple(parallel.layer_range(rank) for rank in range(parallel.pp_size))


class PpMicrobatchPlanner:
    """Split one adopted step into this rank's pipeline microbatch plans.

    Round-robin keeps microbatch widths balanced. Each microbatch renumbers
    its slices/inputs (packed order is preserved), recomputes token padding,
    sampling rows, prompt-logprob slice indices and per-group KV attribution,
    and carries a :class:`DistributedStepIdentity` plus this stage's boundary
    SEND/RECV geometry. Step ids stay increasing: microbatch ``m`` of base
    step ``k`` uses ``k * microbatches + m``.
    """

    def __init__(
        self,
        *,
        parallel: ResolvedParallelPlan,
        pp_rank: int,
        microbatches: int,
        dtype: DType = DType.BF16,
        token_padding_multiple: int = 1,
        bytes_per_boundary_token: int = 0,
    ) -> None:
        if not 0 <= pp_rank < parallel.pp_size:
            raise ValueError(f"pp_rank {pp_rank} outside pp_size {parallel.pp_size}")
        require_int(microbatches, "microbatches", minimum=1)
        require_int(token_padding_multiple, "token_padding_multiple", minimum=1)
        require_int(bytes_per_boundary_token, "bytes_per_boundary_token")
        self._parallel = parallel
        self._pp_rank = pp_rank
        self._microbatches = microbatches
        self._dtype = dtype
        self._token_padding_multiple = token_padding_multiple
        self._bytes_per_boundary_token = bytes_per_boundary_token
        self._spans = stage_spans(parallel)

    @property
    def pp_rank(self) -> int:
        return self._pp_rank

    @property
    def microbatches(self) -> int:
        return self._microbatches

    @property
    def layer_range(self) -> tuple[int, int]:
        """This stage's half-open layer span."""
        return self._spans[self._pp_rank]

    def split(self, step: BatchStepPlan) -> tuple[BatchStepPlan, ...]:
        """Split ``step`` into at most ``microbatches`` plans for this stage."""
        if step.distributed is not None or step.communication != EMPTY_COMMUNICATION_PLAN:
            raise ValueError("step already carries pipeline identity/communication")
        if step.sampling.num_mask_rows > 0:
            raise ValueError("grammar masks per microbatch are not planned yet")
        pairs = list(zip(step.slices, step.inputs, strict=True))
        if not pairs:
            raise ValueError("an empty step cannot be split into microbatches")

        count = min(self._microbatches, len(pairs))
        base = step.step_id * self._microbatches
        ranks = tuple(range(self._parallel.pp_size))
        plans = tuple(
            self._microbatch_plan(
                step,
                tuple(slice_ for slice_, _ in pairs[microbatch::count]),
                tuple(value for _, value in pairs[microbatch::count]),
                step_id=base + microbatch,
                ranks=ranks,
            )
            for microbatch in range(count)
        )
        return plans

    def _microbatch_plan(
        self,
        step: BatchStepPlan,
        slices: tuple,
        inputs: tuple,
        *,
        step_id: int,
        ranks: tuple[int, ...],
    ) -> BatchStepPlan:
        total = 0
        sampling_rows: list[int] = []
        attribution: list[tuple[str, int]] = []
        for scheduled in slices:
            total += scheduled.query_count
            attribution.append((scheduled.request_id, scheduled.query_count))
            if scheduled.sample_last_query:
                sampling_rows.append(total - 1)

        logprob_by_slice = {entry.slice_index: entry for entry in step.prompt_logprobs}
        prompt_logprobs = tuple(
            replace(logprob_by_slice[new_index], slice_index=new_index)
            for new_index, scheduled in enumerate(slices)
            if new_index in logprob_by_slice
        )
        sampling = SamplingPlan(
            num_rows=len(sampling_rows),
            all_greedy=step.sampling.all_greedy,
            any_penalty=step.sampling.any_penalty,
            any_bias=step.sampling.any_bias,
        )
        kv_requirements = tuple(
            KVRequirement(requirement.group_id, total, request_tokens=tuple(attribution))
            for requirement in step.kv_requirements
        )
        return replace(
            step,
            step_id=step_id,
            slices=slices,
            inputs=inputs,
            padded_num_tokens=self._pad(total),
            sampling_rows=tuple(sampling_rows),
            sampling=sampling,
            prompt_logprobs=prompt_logprobs,
            kv_requirements=kv_requirements,
            distributed=DistributedStepIdentity(
                participating_ranks=ranks,
                collective_sequence=step_id,
                worker_generation=0,
            ),
            communication=self._boundary_plan(self._pad(total)),
        )

    def _pad(self, total: int) -> int:
        multiple = self._token_padding_multiple
        return ((total + multiple - 1) // multiple) * multiple

    def _boundary_plan(self, padded_tokens: int) -> CommunicationPlan:
        if self._parallel.pp_size == 1:
            return EMPTY_COMMUNICATION_PLAN
        nbytes = padded_tokens * self._bytes_per_boundary_token
        ops: list[CommOp] = []
        if self._pp_rank > 0:
            ops.append(
                CommOp(
                    kind=CommOpKind.RECV,
                    group="pp",
                    nbytes=nbytes,
                    dtype=self._dtype,
                    stream=StreamRole.COMM,
                    peer_rank=self._pp_rank - 1,
                )
            )
        if self._pp_rank < self._parallel.pp_size - 1:
            ops.append(
                CommOp(
                    kind=CommOpKind.SEND,
                    group="pp",
                    nbytes=nbytes,
                    dtype=self._dtype,
                    stream=StreamRole.COMM,
                    peer_rank=self._pp_rank + 1,
                )
            )
        if not ops:
            return EMPTY_COMMUNICATION_PLAN
        return CommunicationPlan(ops=tuple(ops), overlap_with_compute=False)
