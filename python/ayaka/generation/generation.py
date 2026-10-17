"""Request-local progress, bounded publication and two-phase reconciliation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from enum import StrEnum

from ayaka.generation.contracts import (
    I32_MAX,
    Constraint,
    Demand,
    FeatureReader,
    FeatureRef,
    ForwardOperation,
    HeadOperation,
    HeadOutcome,
    InputLayout,
    Priming,
    Sampling,
    SelectSpec,
    Shaping,
    WorkKind,
    bounded,
    checked_add,
)
from ayaka.generation.controls import Controlled, EndOfGeneration, ReasoningBudget
from ayaka.generation.draft_control import DraftOutlook, TimedLaunch
from ayaka.generation.method import (
    Method,
    MethodCheckpoint,
    MethodChoice,
    MethodEffects,
    MethodKind,
    MethodState,
    Verification,
)
from ayaka.generation.plain import Plain
from ayaka.generation.round import MethodUpdate, RoundForward, RoundState
from ayaka.generation.shaping import selection_history, verification_selects


@dataclass(frozen=True, slots=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True, slots=True)
class DraftStats:
    proposed: int = 0
    accepted: int = 0

    def record(self, proposed: int, accepted: int) -> DraftStats:
        bounded(proposed, "proposed count")
        bounded(accepted, "accepted count", proposed)
        return DraftStats(
            checked_add(self.proposed, proposed), checked_add(self.accepted, accepted)
        )


@dataclass(frozen=True, slots=True)
class DetailedUsage:
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    draft_n: int
    draft_n_accepted: int


@dataclass(frozen=True, slots=True)
class Options:
    max_tokens: int
    output_capacity: int
    context_limit: int
    vocabulary: int
    stop_tokens: frozenset[int] = frozenset()
    suppressed_tokens: frozenset[int] = frozenset()
    sampling: Sampling = Sampling.GREEDY
    shaping: Shaping = Shaping()
    seed: int = 0
    forced_quantum: int = 256
    method: MethodChoice = MethodChoice()
    end_of_generation: EndOfGeneration = EndOfGeneration.STOP
    reasoning_budget: ReasoningBudget | None = None

    def __post_init__(self) -> None:
        bounded(self.max_tokens, "max_tokens", I32_MAX)
        bounded(self.output_capacity, "output_capacity", minimum=1)
        bounded(self.context_limit, "context_limit", I32_MAX, 1)
        bounded(self.vocabulary, "vocabulary", I32_MAX, 1)
        bounded(self.seed, "seed")
        bounded(self.forced_quantum, "forced_quantum", 256)
        object.__setattr__(self, "stop_tokens", frozenset(self.stop_tokens))
        object.__setattr__(self, "suppressed_tokens", frozenset(self.suppressed_tokens))
        object.__setattr__(self, "sampling", Sampling(self.sampling))
        object.__setattr__(self, "end_of_generation", EndOfGeneration(self.end_of_generation))
        self.shaping.validate()
        if (self.shaping.temperature == 0) != (self.sampling is Sampling.GREEDY):
            raise ValueError("sampling mode disagrees with shaping temperature")
        for token in self.stop_tokens | self.suppressed_tokens:
            bounded(token, "control token", self.vocabulary - 1)


class FinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"
    CONTEXT = "context"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class CausalProgress:
    accepted_position: int
    resident_position: int


@dataclass(frozen=True, slots=True)
class PendingReconciliation:
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class OutputToken:
    index: int
    token: int


@dataclass(frozen=True, slots=True)
class ReconcileDecision:
    accepted_rows: int


@dataclass(frozen=True, slots=True)
class PreparedMethodTransition:
    request: str
    decision: ReconcileDecision
    _owner: object
    _version: int
    _method: MethodState


@dataclass(frozen=True, slots=True)
class PreparedGenerationTransition:
    request: str
    decision: ReconcileDecision
    _owner: object
    _round: object
    _version: int
    _method: MethodState
    _effects: MethodEffects
    _constraint: Constraint | None
    _generated: tuple[int, ...]
    _output: tuple[OutputToken, ...]
    _stats: DraftStats
    _resident: int
    _accepted: int
    _finish: FinishReason | None


@dataclass(frozen=True, slots=True)
class MethodWork:
    """Reconcile every returned head operation, then start the target round again."""

    generation: Generation
    operations: tuple[HeadOperation, ...]


class GenerationSeed:
    """Device-free intent; create live method state at the numerical owner."""

    def __init__(
        self,
        prompt: tuple[int, ...],
        layout: InputLayout,
        options: Options,
        constraint: Constraint | None = None,
    ) -> None:
        prompt = tuple(prompt)
        if not prompt or len(prompt) != layout.count or len(prompt) > options.context_limit:
            raise ValueError("invalid generation prompt or input layout")
        for token in prompt:
            bounded(token, "prompt token", options.vocabulary - 1)
        if constraint is not None and constraint.position() != 0:
            raise ValueError("initial constraint position must be zero")
        self.prompt, self.layout, self.options = prompt, layout, options
        self.constraint = constraint
        self._consumed = False

    def selection_mask(self) -> tuple[int, ...] | None:
        return None if self.constraint is None else tuple(self.constraint.mask())

    def into_generation(self, method_factory: Method | None = None) -> Generation:
        if self._consumed:
            raise ValueError("generation seed was already consumed")
        generation = Generation(
            self.prompt, self.layout, self.options, self.constraint, method_factory
        )
        self._consumed = True
        return generation


class Generation:
    """Single-owner logical state; physical KV mutation is the executor's job.

    Starting a target round locks mutating APIs until commit/discard. Prepared
    transitions carry owner, round and version identities and are single-use.
    """

    def __init__(
        self,
        prompt: tuple[int, ...],
        layout: InputLayout,
        options: Options,
        constraint: Constraint | None = None,
        method_factory: Method | None = None,
    ) -> None:
        seed = GenerationSeed(prompt, layout, options, constraint)
        factory = method_factory if method_factory is not None else Plain()
        if not method_matches_choice(options.method, factory):
            raise ValueError("generation method factory does not match request policy")
        if factory.embeds_prompt_tokens() and any(
            not span.language_history for span in layout.spans
        ):
            raise ValueError("this drafter embeds prompt tokens and cannot enter media rows")
        self.prompt, self.layout, self.options = seed.prompt, layout, options
        self.constraint = Controlled.compose(
            constraint,
            options.end_of_generation,
            options.stop_tokens,
            options.suppressed_tokens,
            options.reasoning_budget,
            options.vocabulary,
        )
        self.method_factory, self._method = factory, factory.create()
        self._generated: tuple[int, ...] = ()
        self._output: deque[OutputToken] = deque()
        self._published = 0
        self._resident_position = self._accepted_position = self._reconciliation_target = 0
        self._resident = False
        self._reusable_prefixes = True
        self._finish = FinishReason.LENGTH if options.max_tokens == 0 else None
        self._cached_tokens = 0
        self._stats = DraftStats()
        self._identity = object()
        self._version = 0
        self._active: object | None = None

    @property
    def generated(self) -> tuple[int, ...]:
        return self._generated

    @property
    def resident_position(self) -> int:
        return self._resident_position

    @property
    def accepted_position(self) -> int:
        return self._accepted_position

    @property
    def finish_reason(self) -> FinishReason | None:
        return self._finish

    @property
    def output_len(self) -> int:
        return len(self._output)

    @property
    def is_resident(self) -> bool:
        return self._resident

    def _idle(self) -> None:
        if self._active is not None:
            raise ValueError("generation is owned by a started target round")

    def causal_progress(self) -> CausalProgress:
        return CausalProgress(self.accepted_position, self.resident_position)

    def usage(self) -> Usage:
        return Usage(len(self.prompt), len(self.generated))

    def detailed_usage(self) -> DetailedUsage:
        return DetailedUsage(
            len(self.prompt),
            len(self.generated),
            self._cached_tokens,
            self._stats.proposed,
            self._stats.accepted,
        )

    def method_identity(self) -> str:
        return self.method_factory.identity()

    def method_requirements(self):
        return self.method_factory.requires()

    def method_reclaimable(self) -> int:
        return self._method.reclaimable()

    def set_prefix_reuse(self, reusable: bool) -> None:
        self._idle()
        if self.is_resident or self.resident_position or self.generated:
            raise ValueError("prefix reuse must be declared before generation starts")
        self._reusable_prefixes = reusable

    def _priming_horizon(self, boundary: int) -> int:
        return boundary if self._reusable_prefixes else len(self.prompt) + len(self.generated)

    def method_checkpoint(self) -> MethodCheckpoint:
        self._idle()
        if not self._reusable_prefixes or not self.is_resident:
            raise ValueError("method checkpoint requires resident reusable state")
        return self._method.checkpoint()

    def constraint_position(self) -> int | None:
        return None if self.constraint is None else self.constraint.position()

    def selection_mask(self) -> tuple[int, ...] | None:
        return None if self.constraint is None else tuple(self.constraint.mask())

    def _forced_tokens(self, limit: int) -> tuple[int, ...]:
        limit = min(
            limit,
            self.options.forced_quantum,
            self.options.max_tokens - len(self.generated),
            self.options.output_capacity - self.output_len,
        )
        if limit <= 0 or self.constraint is None:
            return ()
        forced = tuple(self.constraint.forced(limit))
        if len(forced) > limit:
            raise ValueError("constraint returned an invalid forced run")
        for index, token in enumerate(forced):
            bounded(token, "forced token", self.options.vocabulary - 1)
            if token in self.options.stop_tokens:
                return forced[: index + 1]
        return forced

    def _plan(self, allowance: int) -> tuple[str, int | tuple[int, ...]]:
        self._idle()
        bounded(allowance, "service allowance", minimum=1)
        if self.finish_reason is not None or not self.is_resident:
            raise ValueError("round requires unfinished resident generation")
        if self.output_len >= self.options.output_capacity:
            raise ValueError("generation output is full")
        position = self.resident_position
        if position < self._reconciliation_target:
            target = self._reconciliation_target
            if position < len(self.prompt):
                target = min(target, len(self.prompt))
            return "replay", self.layout.chunk_end(position, target, allowance)
        if position < len(self.prompt):
            return "prefill", self.layout.chunk_end(position, len(self.prompt), allowance)
        if (
            not self.generated
            or position != len(self.prompt) + len(self.generated) - 1
            or position >= self.options.context_limit
        ):
            raise ValueError("generation history and numerical continuation disagree")
        remaining = self.options.max_tokens - len(self.generated)
        credit = self.options.output_capacity - self.output_len
        forced = self._forced_tokens(
            min(allowance, self.options.context_limit - position, remaining, credit)
        )
        if forced:
            return "forced", forced
        limit = RoundState.proposal_limit(
            min(allowance, self.options.context_limit - position),
            remaining,
            credit,
            self.options.method.kind is MethodKind.PLAIN,
        )
        return "propose", min(limit, self.options.method.proposals)

    def publication_bound(self, allowance: int) -> int:
        kind, value = self._plan(allowance)
        if kind == "replay":
            return 0
        if kind == "prefill":
            return int(value == len(self.prompt))
        if isinstance(value, tuple):
            return len(value)
        return value + 1

    def _prefill_round(self, start: int, end: int, drafter: int | None) -> RoundState:
        requirements = self.method_factory.requires()
        successor = self.method_factory.priming_requires_successor()
        prime = None
        if (
            drafter is not None
            and requirements.head
            and requirements.prefill_demand & Demand.FEATURES
        ):
            tokens = (
                self.prompt[drafter + 1 : end]
                if successor
                else self.prompt[start + 1 : min(end + 1, len(self.prompt))]
            )
            if successor or tokens:
                prime = Priming(tokens, drafter, self._priming_horizon(end), successor)
        forced = self._forced_tokens(1) if end == len(self.prompt) else ()
        select = None
        if end == len(self.prompt) and not forced:
            select = verification_selects(
                len(self.generated),
                self.generated,
                (),
                self.constraint,
                self.options.sampling,
                self.options.shaping,
                self.options.seed,
            )[0]
        return RoundState.progress(
            WorkKind.PREFILL, self.prompt[start:end], forced, select, prime, requirements
        )

    def _proposal_selects(self, count: int) -> tuple[SelectSpec, ...]:
        history = selection_history(self.generated) if self.options.shaping.uses_history() else None
        first_mask = self.selection_mask()
        return tuple(
            SelectSpec(
                sampling=self.options.sampling,
                seed=self.options.seed,
                position=checked_add(len(self.generated), index),
                mask=first_mask if index == 0 else None,
                shaping=self.options.shaping,
                history=history,
            )
            for index in range(count)
        )

    def start_round(self, request: str, allowance: int) -> StartedRound | MethodWork:
        kind, value = self._plan(allowance)
        requirements = self.method_factory.requires()
        if kind == "replay":
            assert isinstance(value, int)
            known = self.prompt + self.generated
            state = RoundState.progress(
                WorkKind.REPLAY, known[self.resident_position : value], requirements=requirements
            )
        elif kind == "prefill":
            assert isinstance(value, int)
            state = self._prefill_round(
                self.resident_position, value, self._method.priming_position()
            )
        elif kind == "forced":
            assert isinstance(value, tuple)
            state = RoundState.forced(self.generated[-1], value, requirements)
        else:
            assert isinstance(value, int)
            method = self._method.fork_transition()
            proposed = method.propose(request, self._proposal_selects(value))
            if proposed.operations:
                self._validate_method_operations(request, proposed.operations)
                self._method = method
                self._version += 1
                return MethodWork(self, proposed.operations)
            proposal = proposed.tokens
            if len(proposal) > value:
                raise ValueError("method returned an invalid proposal")
            for index, token in enumerate(proposal):
                bounded(token, "proposal token", self.options.vocabulary - 1)
                if token in self.options.stop_tokens:
                    proposal = proposal[:index]
                    break
            state = RoundState.verification(
                self.generated[-1],
                proposal,
                value,
                len(self.generated),
                self.generated,
                self.constraint,
                self.options.sampling,
                self.options.shaping,
                self.options.seed,
                requirements,
            )
            self._method = method
        self._version += 1
        started = StartedRound(self, state, request)
        self._active = started._identity
        return started

    def pending_reconciliation(self) -> PendingReconciliation | None:
        if self.is_resident and self.resident_position < self.accepted_position:
            return PendingReconciliation(self.resident_position, self.accepted_position)
        return None

    def start_reconciliation(self, allowance: int, request: str = "request") -> StartedRound:
        self._idle()
        bounded(allowance, "reconciliation allowance", minimum=1)
        pending = self.pending_reconciliation()
        if pending is None:
            raise ValueError("generation has no pending causal reconciliation")
        end = self.layout.chunk_end(pending.start, pending.end, allowance)
        state = RoundState.progress(
            WorkKind.REPLAY,
            (self.prompt + self.generated)[pending.start : end],
            requirements=self.method_factory.requires(),
        )
        self._reconciliation_target = pending.end
        self._version += 1
        started = StartedRound(self, state, request)
        self._active = started._identity
        return started

    def take(self, count: int) -> tuple[OutputToken, ...]:
        self._idle()
        bounded(count, "output collection count", minimum=1)
        tokens = tuple(self._output.popleft() for _ in range(min(count, self.output_len)))
        self._published += len(tokens)
        self._version += 1
        return tokens

    def _validate_method_operations(
        self, request: str, operations: tuple[HeadOperation, ...]
    ) -> None:
        for operation in operations:
            if (
                not isinstance(operation, HeadOperation)
                or operation.request != request
                or not self.method_factory.requires().head
            ):
                raise ValueError("method returned effects outside its executor ownership")
            operation.validate()

    def prepare_method_transition(
        self, operation: HeadOperation, outcome: HeadOutcome
    ) -> PreparedMethodTransition:
        self._idle()
        if not self.is_resident:
            raise ValueError("method transition requires resident state")
        self._validate_method_operations(operation.request, (operation,))
        for selected in outcome.proposals:
            bounded(selected.status, "head selection status", 2)
            if selected.status == 0:
                bounded(selected.token, "head proposal", self.options.vocabulary - 1)
        method = self._method.fork_transition()
        method.reconcile(operation, outcome)
        return PreparedMethodTransition(
            operation.request,
            ReconcileDecision(len(operation.tokens)),
            self._identity,
            self._version,
            method,
        )

    def commit_method_transition(self, transition: PreparedMethodTransition) -> None:
        self._idle()
        if transition._owner is not self._identity or transition._version != self._version:
            raise ValueError("stale or foreign method transition")
        self._method = transition._method
        self._version += 1

    def timed(self, launch: TimedLaunch, seconds: float, sequences: int, rows: int) -> None:
        self._method.timed(launch, seconds, sequences, rows)

    def draft_outlook(self) -> DraftOutlook | None:
        return self._method.outlook()

    def cancel(self) -> None:
        self._idle()
        self._finish = self._finish or FinishReason.CANCELLED
        self._version += 1

    def fail(self) -> None:
        self._idle()
        self._finish = self._finish or FinishReason.FAILED
        self._version += 1

    def discard_output(self) -> None:
        self._idle()
        self._published += self.output_len
        self._output.clear()
        self._version += 1

    def evicted(self) -> None:
        self._idle()
        self._reconciliation_target = self.resident_position
        self._method.evict()
        self._resident = False
        self._version += 1

    def resume_bound(self) -> int:
        return max(len(self.prompt), self.accepted_position)

    def resume_at(self, prefix: tuple[int, MethodCheckpoint] | None = None) -> None:
        self._idle()
        if self.is_resident:
            raise ValueError("only a non-resident generation can resume")
        position = 0
        method = self._method
        if prefix is not None:
            position, checkpoint = prefix
            bounded(position, "resumed prefix", self.resume_bound() - 1, 1)
            if position < len(self.prompt) and not self.layout.boundary(position):
                raise ValueError("resumed prefix is not an exact input boundary")
            if checkpoint.kind is not self.options.method.kind:
                raise ValueError("retained checkpoint does not match generation method")
            if checkpoint.draft is not None and checkpoint.draft.target_rows() != position:
                raise ValueError("retained method checkpoint has incorrect target rows")
            method = self.method_factory.create(checkpoint)
        self._method = method
        if position > self._reconciliation_target:
            self._reconciliation_target = position
            self._accepted_position = max(self.accepted_position, position)
            self._cached_tokens = max(self._cached_tokens, position)
        self._resident_position = position
        self._resident = True
        self._version += 1


class StartedRound:
    def __init__(self, generation: Generation, state: RoundState, request: str) -> None:
        self.generation, self.state, self.request = generation, state, request
        self._identity = object()

    def _live(self) -> None:
        if self.generation._active is not self._identity:
            raise ValueError("target round has already ended")

    def round_forward(self) -> RoundForward:
        return self.state.forward

    def round_publication_bound(self) -> int:
        return self.state.publication_bound()

    def timed(self, launch: TimedLaunch, seconds: float, sequences: int, rows: int) -> None:
        self._live()
        self.generation.timed(launch, seconds, sequences, rows)

    def planned_prefill(self, request: str, allowance: int) -> ForwardOperation | None:
        self._live()
        gen, forward = self.generation, self.state.forward
        if (
            forward.kind is not WorkKind.PREFILL
            or forward.selects
            or forward.committed != len(forward.tokens)
            or gen.layout.spans
            or gen.finish_reason is not None
            or allowance <= 0
        ):
            return None
        position = gen.resident_position + len(forward.tokens)
        if position < gen._reconciliation_target or position >= len(gen.prompt):
            return None
        drafter = (
            None if forward.prime is None else forward.prime.position + len(forward.prime.tokens)
        )
        end = gen.layout.chunk_end(position, len(gen.prompt), allowance)
        return gen._prefill_round(position, end, drafter).forward.into_operation(request, position)

    def prepare_round_transition(
        self,
        request: str,
        samples: tuple[int, ...],
        features: FeatureRef | None = None,
        reader: FeatureReader | None = None,
    ) -> PreparedGenerationTransition:
        self._live()
        if request != self.request:
            raise ValueError("target outcome belongs to a different request")
        gen, forward = self.generation, self.state.forward
        acceptance = self.state.reconcile(tuple(samples), gen.options.stop_tokens)
        causal = (
            gen.finish_reason is not None
            and forward.kind is WorkKind.REPLAY
            and not forward.selects
        )
        if gen.finish_reason is not None and not causal:
            acceptance = replace(
                acceptance,
                emitted=(),
                committed_rows=0,
                proposed=0,
                accepted_proposals=0,
                method_update=MethodUpdate.NONE,
            )
        if (
            len(acceptance.emitted) > gen.options.output_capacity - gen.output_len
            or len(acceptance.emitted) > gen.options.max_tokens - len(gen.generated)
            or (
                not causal
                and acceptance.committed_rows > gen.options.context_limit - gen.resident_position
            )
        ):
            raise ValueError("round exceeded a generation bound")
        for token in acceptance.emitted:
            bounded(token, "selected token", gen.options.vocabulary - 1)
        constraint = None
        if acceptance.emitted and gen.constraint is not None:
            if gen.constraint.position() != len(gen.generated):
                raise ValueError("constraint progress differs from accepted tokens")
            constraint = gen.constraint.stage(acceptance.emitted)
            if constraint.position() != len(gen.generated) + len(acceptance.emitted):
                raise ValueError("constraint successor has incorrect position")
        method = gen._method.fork_transition()
        effects = MethodEffects()
        if acceptance.method_update is MethodUpdate.PRIME:
            if forward.demand & Demand.FEATURES and features is None:
                raise ValueError("prompt chunk did not return demanded target features")
            effects = method.prime(
                request,
                acceptance.inputs,
                acceptance.emitted[0] if acceptance.emitted else None,
                features,
                gen._priming_horizon(gen.resident_position + acceptance.committed_rows),
                None if forward.prime is None else len(forward.prime.tokens),
                reader,
            )
        elif acceptance.method_update is MethodUpdate.OBSERVE:
            if not acceptance.emitted:
                raise ValueError("resolved verification emitted no successor token")
            effects = method.observe(
                request,
                Verification(
                    acceptance.inputs, acceptance.committed_rows, acceptance.emitted[-1], features
                ),
                reader,
            )
        gen._validate_method_operations(request, effects.operations)
        stats = gen._stats.record(acceptance.proposed, acceptance.accepted_proposals)
        resident = checked_add(gen.resident_position, acceptance.committed_rows)
        generated, output, finish = gen.generated, tuple(gen._output), gen.finish_reason
        for token in acceptance.emitted:
            generated += (token,)
            if token in gen.options.stop_tokens:
                finish = FinishReason.STOP
                break
            output += (OutputToken(checked_add(gen._published, len(output)), token),)
        if finish is None and len(generated) >= gen.options.max_tokens:
            finish = FinishReason.LENGTH
        accepted = max(checked_add(len(gen.prompt), len(generated)), resident)
        if finish is None and resident >= gen.options.context_limit:
            finish = FinishReason.CONTEXT
        return PreparedGenerationTransition(
            request,
            ReconcileDecision(acceptance.committed_rows),
            gen._identity,
            self._identity,
            gen._version,
            method,
            effects,
            constraint,
            generated,
            output,
            stats,
            resident,
            accepted,
            finish,
        )

    def commit(self, transition: PreparedGenerationTransition) -> tuple[Generation, MethodEffects]:
        """Call only after the executor reconciles transition.decision.accepted_rows."""
        self._live()
        gen = self.generation
        if (
            transition._owner is not gen._identity
            or transition._round is not self._identity
            or transition._version != gen._version
            or transition.request != self.request
        ):
            raise ValueError("stale or foreign generation transition")
        gen._method = transition._method
        if transition._constraint is not None:
            gen.constraint = transition._constraint
        gen._generated, gen._output = transition._generated, deque(transition._output)
        gen._stats = transition._stats
        gen._resident_position, gen._accepted_position = transition._resident, transition._accepted
        gen._finish = transition._finish
        gen._active = None
        gen._version += 1
        gen._method.publish_learning()
        return gen, transition._effects

    def cancel(self) -> Generation:
        self._live()
        self.generation._active = None
        self.generation.cancel()
        return self.generation

    def fail(self) -> Generation:
        self._live()
        self.generation._active = None
        self.generation.fail()
        return self.generation

    def evicted(self) -> Generation:
        self._live()
        self.generation._active = None
        self.generation.evicted()
        return self.generation


def method_matches_choice(choice: MethodChoice, method: Method) -> bool:
    requirements = method.requires()
    if choice.kind is MethodKind.PLAIN:
        from ayaka.generation.method import MethodRequirements

        return method.identity() == "plain" and requirements == MethodRequirements()
    return (
        method.identity().startswith(f"{choice.kind.value}:")
        and requirements.head
        and bool((requirements.prefill_demand | requirements.closing_demand) & Demand.FEATURES)
        and bool(requirements.verify_demand & Demand.FEATURES)
        and 0 < choice.proposals <= method.proposals()
    )
