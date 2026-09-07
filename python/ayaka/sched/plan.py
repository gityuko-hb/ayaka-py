from dataclasses import dataclass, field
from enum import StrEnum

from ayaka.handles import SequenceHandle
from ayaka.memory.views import ExecutionMemoryView, GroupedExecutionMemoryView
from ayaka.plan import (
    CommunicationPlan,
    ExecutionPlan,
    GraphMode,
    GraphPlan,
    MemoryPlan,
    SamplingPlan,
    WeightResidencyPlan,
)
from ayaka.utils.validation import require_frozen, require_int, require_text

__all__ = [
    "BatchStepPlan", "DistributedStepIdentity", "KVRequirement", "Phase",
    "PreparedStep", "RequestStepInput", "ScheduledSlice", "StepDependency",
]


class Phase(StrEnum):
    """Request semantics; a one-query prefill remains PREFILL."""

    PREFILL = "prefill"
    DECODE = "decode"


@dataclass(frozen=True, slots=True)
class ScheduledSlice:
    """One contiguous logical range, guarded by request epoch and KV version.

    sequence_epoch is the request execution incarnation, not an allocator
    reclamation epoch. expected_state_version is the KV sequence version
    observed before reservation; a successful KV commit advances that version.
    Bounds against actual known tokens are checked by RequestStepInput.
    """

    request_id: str
    sequence_epoch: int
    expected_state_version: int
    query_start: int
    query_count: int
    phase: Phase
    sample_last_query: bool = False

    def __post_init__(self) -> None:
        require_text(self.request_id, "request_id")
        require_int(self.sequence_epoch, "sequence_epoch", minimum=1)
        require_int(self.expected_state_version, "expected_state_version")
        require_int(self.query_start, "query_start")
        require_int(self.query_count, "query_count", minimum=1)
        if not isinstance(self.phase, Phase):
            raise TypeError("phase must be Phase")
        if type(self.sample_last_query) is not bool:
            raise TypeError("sample_last_query must be bool")
        if self.phase is Phase.DECODE and self.query_count != 1:
            raise ValueError("decode must contain exactly one query")

    @property
    def query_end(self) -> int:
        return self.query_start + self.query_count


@dataclass(frozen=True, slots=True)
class RequestStepInput:
    """Owned token snapshot and sequence binding used to validate one slice.

    known_tokens contains prompt and already published output tokens, never
    tentative samples. The snapshot may outlive the mutable request. A prefix
    lookup alone is not sufficient: computed_tokens must describe validated,
    acquired KV/state. Later preparation must revalidate this binding.
    """

    request_id: str
    sequence: SequenceHandle
    sequence_epoch: int
    state_version: int
    prompt_tokens: int
    known_tokens: tuple[int, ...]
    computed_tokens: int
    max_output_tokens: int

    def __post_init__(self) -> None:
        require_frozen(self, "request input")
        require_text(self.request_id, "request_id")
        if not isinstance(self.sequence, SequenceHandle):
            raise TypeError("sequence must be SequenceHandle")
        require_int(self.sequence_epoch, "sequence_epoch", minimum=1)
        require_int(self.state_version, "state_version")
        require_int(self.prompt_tokens, "prompt_tokens", minimum=1)
        require_int(self.computed_tokens, "computed_tokens")
        require_int(self.max_output_tokens, "max_output_tokens", minimum=1)
        for token in self.known_tokens:
            require_int(token, "token id")
        if not self.prompt_tokens <= len(self.known_tokens) <= (
            self.prompt_tokens + self.max_output_tokens
        ):
            raise ValueError("known token count disagrees with prompt/output limits")
        if self.computed_tokens > len(self.known_tokens):
            raise ValueError("computed_tokens exceeds known tokens")

    def validate_slice(self, scheduled: ScheduledSlice) -> None:
        """Raise without mutation on stale identity, invalid bounds or sampling."""
        if (
            scheduled.request_id != self.request_id
            or scheduled.sequence_epoch != self.sequence_epoch
            or scheduled.expected_state_version != self.state_version
        ):
            raise ValueError("slice identity or KV state version is stale")
        if scheduled.query_start != self.computed_tokens:
            raise ValueError("slice must start at computed_tokens")
        if scheduled.query_end > len(self.known_tokens):
            raise ValueError("slice extends beyond known tokens")
        if scheduled.phase is Phase.PREFILL:
            if scheduled.query_start >= self.prompt_tokens or scheduled.query_end > self.prompt_tokens:
                raise ValueError("prefill must stay within the prompt")
        elif scheduled.query_start < self.prompt_tokens:
            raise ValueError("decode cannot replace an unfinished prefill")
        if scheduled.sample_last_query:
            if scheduled.query_end != len(self.known_tokens):
                raise ValueError("sampling is allowed only at the known-token boundary")
            if len(self.known_tokens) - self.prompt_tokens >= self.max_output_tokens:
                raise ValueError("output token budget is exhausted")


@dataclass(frozen=True, slots=True)
class KVRequirement:
    """Per-group capacity requirement, not a physical allocation or page table."""

    group_id: int
    append_tokens: int
    cow_pages: int = 0
    restore_bytes: int = 0
    growth_bytes: int = 0

    def __post_init__(self) -> None:
        for name in ("group_id", "append_tokens", "cow_pages", "restore_bytes", "growth_bytes"):
            require_int(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class StepDependency:
    """Opaque producer identity to be resolved by the later runtime."""

    producer_id: str
    kind: str

    def __post_init__(self) -> None:
        require_text(self.producer_id, "producer_id")
        require_text(self.kind, "dependency kind")


@dataclass(frozen=True, slots=True)
class DistributedStepIdentity:
    """Rank participation and collective ordering, without launching collectives."""

    participating_ranks: tuple[int, ...]
    collective_sequence: int
    worker_generation: int

    def __post_init__(self) -> None:
        require_frozen(self, "distributed identity")
        require_int(self.collective_sequence, "collective_sequence")
        require_int(self.worker_generation, "worker_generation")
        if not self.participating_ranks or len(set(self.participating_ranks)) != len(
            self.participating_ranks
        ):
            raise ValueError("participating ranks must be non-empty and unique")
        for rank in self.participating_ranks:
            require_int(rank, "rank")


@dataclass(frozen=True, slots=True)
class BatchStepPlan:
    """Dynamic decisions for one iteration, with one slice per request.

    slices and inputs share packed order. Query rows are compact and padding
    follows all real rows. sampling_rows must exactly match the last rows of
    slices explicitly requesting a sample; rows cannot be inferred from phase
    or query length. No live request, tensor, allocator or page table is held.
    """

    step_id: int
    execution_plan_id: str
    slices: tuple[ScheduledSlice, ...]
    inputs: tuple[RequestStepInput, ...]
    padded_num_tokens: int
    sampling_rows: tuple[int, ...] = ()
    sampling: SamplingPlan = field(default_factory=SamplingPlan)
    memory: MemoryPlan = field(default_factory=MemoryPlan)
    kv_requirements: tuple[KVRequirement, ...] = ()
    dependencies: tuple[StepDependency, ...] = ()
    distributed: DistributedStepIdentity | None = None
    communication: CommunicationPlan = field(default_factory=CommunicationPlan)
    weight_residency: WeightResidencyPlan = field(default_factory=WeightResidencyPlan)
    graph: GraphPlan = field(default_factory=GraphPlan)
    trace_ids: tuple[str, ...] = ()
    created_ns: int = 0

    def __post_init__(self) -> None:
        require_frozen(self, "batch step")
        require_int(self.step_id, "step_id")
        require_text(self.execution_plan_id, "execution_plan_id")
        require_int(self.padded_num_tokens, "padded_num_tokens")
        require_int(self.created_ns, "created_ns")
        if len(self.inputs) != len(self.slices):
            raise ValueError("one input snapshot is required for each slice")
        if len(set(self.request_order)) != len(self.slices):
            raise ValueError("a request may have only one slice per step")
        if len({value.sequence for value in self.inputs}) != len(self.inputs):
            raise ValueError("different requests cannot bind the same sequence")
        offset = 0
        expected_rows = []
        for scheduled, value in zip(self.slices, self.inputs, strict=True):
            value.validate_slice(scheduled)
            offset += scheduled.query_count
            if scheduled.sample_last_query:
                expected_rows.append(offset - 1)
        if self.padded_num_tokens < offset:
            raise ValueError("padding cannot remove real query rows")
        for row in self.sampling_rows:
            require_int(row, "sampling row")
        if self.sampling_rows != tuple(expected_rows):
            raise ValueError("sampling rows must exactly match scheduled sample boundaries")
        if self.sampling.num_rows != len(self.sampling_rows):
            raise ValueError("sampling row count disagrees with explicit sampling rows")
        groups = [requirement.group_id for requirement in self.kv_requirements]
        if len(set(groups)) != len(groups):
            raise ValueError("KV requirements must be unique by group")
        if self.graph.mode is GraphMode.REPLAY:
            if not self.is_pure_decode:
                raise ValueError("graph replay requires explicit pure decode")
            if not self.sampling.graph_capturable:
                raise ValueError("graph replay cannot contain custom sampling operations")

    @property
    def request_order(self) -> tuple[str, ...]:
        return tuple(scheduled.request_id for scheduled in self.slices)

    @property
    def num_tokens(self) -> int:
        return sum(scheduled.query_count for scheduled in self.slices)

    @property
    def is_pure_decode(self) -> bool:
        return bool(self.slices) and all(s.phase is Phase.DECODE for s in self.slices)

    @property
    def token_ids(self) -> tuple[int, ...]:
        """Packed input tokens, excluding any padding."""
        return tuple(
            token
            for scheduled, value in zip(self.slices, self.inputs, strict=True)
            for token in value.known_tokens[scheduled.query_start : scheduled.query_end]
        )

    @property
    def positions(self) -> tuple[int, ...]:
        return tuple(
            position for scheduled in self.slices
            for position in range(scheduled.query_start, scheduled.query_end)
        )


@dataclass(frozen=True, slots=True)
class PreparedStep:
    """Host preparation contract binding a step to a frozen KV execution view.

    Only data is validated here: construction does not acquire a lease, mark
    it inflight, enqueue work or prove completion. The runtime must retain
    the lease and revalidate allocator generations before any consumer uses
    these physical addresses. Backend metadata/workspace binding is added
    by the later preparation milestone, not fabricated by this P0 contract.
    """

    execution: ExecutionPlan
    step: BatchStepPlan
    memory_view: ExecutionMemoryView | GroupedExecutionMemoryView

    def __post_init__(self) -> None:
        require_frozen(self, "prepared step")
        if not isinstance(self.memory_view, (ExecutionMemoryView, GroupedExecutionMemoryView)):
            raise TypeError("memory_view must be an execution memory snapshot")
        if self.execution.plan_id != self.step.execution_plan_id:
            raise ValueError("resolved execution identity disagrees with step")
        if not self.step.slices:
            raise ValueError("an empty batch cannot be prepared")
        if self.step.padded_num_tokens > self.execution.compute.max_num_batched_tokens:
            raise ValueError("step exceeds resolved token capacity")
        if (
            self.memory_view.step_id != self.step.step_id
            or self.memory_view.lease.step_id != self.step.step_id
        ):
            raise ValueError("KV lease and step identities disagree")
        if tuple(view.sequence for view in self.memory_view.sequences) != tuple(
            value.sequence for value in self.step.inputs
        ):
            raise ValueError("KV execution views must follow packed request order")
        for scheduled, view in zip(self.step.slices, self.memory_view.sequences, strict=True):
            if view.reservation.step_id != self.step.step_id:
                raise ValueError("KV reservation belongs to another step")
            if (
                view.base_committed_tokens != scheduled.query_start
                or view.num_reserved_tokens != scheduled.query_count
            ):
                raise ValueError("KV reservation disagrees with scheduled range")
            slot_groups = (
                (view.write_slots,) if isinstance(self.memory_view, ExecutionMemoryView)
                else tuple(group.write_slots for group in view.groups)
            )
            if not slot_groups:
                raise ValueError("prepared sequence requires at least one KV group")
            for slots in slot_groups:
                if tuple(slot.logical_position for slot in slots) != tuple(
                    range(scheduled.query_start, scheduled.query_end)
                ):
                    raise ValueError("write slots must cover each scheduled logical position")
