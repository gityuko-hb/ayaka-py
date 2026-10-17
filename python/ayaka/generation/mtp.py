"""MTP and the target-conditioned state machine shared with block drafters."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace

from ayaka.generation.contracts import (
    Demand,
    DraftForm,
    FeatureReader,
    FeatureRef,
    FeatureRows,
    HeadOperation,
    HeadOutcome,
    HeadPhase,
    SelectSpec,
    bounded,
    checked_add,
)
from ayaka.generation.draft_budget import DraftBudget
from ayaka.generation.draft_control import DraftController, DraftLoad, DraftOutlook, TimedLaunch
from ayaka.generation.draft_transition import DraftHistory, DraftRoundState
from ayaka.generation.method import (
    DraftCheckpoint,
    MethodCheckpoint,
    MethodCheckpointError,
    MethodEffects,
    MethodKind,
    MethodRequirements,
    OpenFeature,
    PendingRows,
    Propose,
    Verification,
)


@dataclass(frozen=True, slots=True)
class Mtp:
    artifact: str
    width: int
    reads_target: bool = False
    budget: DraftBudget | None = None
    load: DraftLoad | None = None

    def __post_init__(self) -> None:
        if not self.artifact.strip():
            raise ValueError("MTP identity must be nonempty")
        bounded(self.width, "MTP proposal width", 255, 1)
        if self.budget is not None:
            self.budget.validate(self.width)

    @classmethod
    def reading_target(cls, artifact: str, proposals: int) -> Mtp:
        return cls(artifact, proposals, reads_target=True)

    def with_budget(self, budget: DraftBudget) -> Mtp:
        return replace(self, budget=budget)

    def adaptive(self) -> Mtp:
        return replace(self, load=DraftLoad())

    def identity(self) -> str:
        suffix = ":adaptive" if self.load is not None else ""
        if self.budget is not None:
            suffix += f":{self.budget.generated}:{self.budget.verified}"
        return f"mtp:{self.artifact}:{self.width}{suffix}"

    def proposals(self) -> int:
        return self.width

    def priming_requires_successor(self) -> bool:
        return not self.reads_target

    def embeds_prompt_tokens(self) -> bool:
        return not self.reads_target

    def requires(self) -> MethodRequirements:
        return MethodRequirements(
            Demand.NONE if self.reads_target else Demand.FEATURES,
            Demand.FEATURES if self.reads_target else Demand.NONE,
            Demand.FEATURES,
            True,
        )

    def create(self, checkpoint: MethodCheckpoint | None = None) -> DrafterState:
        return DrafterState(
            MethodKind.MTP,
            self.width,
            self.budget or DraftBudget(self.width, self.width),
            self.load,
            checkpoint,
            self.reads_target,
        )


class DrafterState:
    def __init__(
        self,
        kind: MethodKind,
        proposals: int,
        budget: DraftBudget,
        load: DraftLoad | None = None,
        checkpoint: MethodCheckpoint | None = None,
        reads_target: bool = False,
    ) -> None:
        if checkpoint is not None and checkpoint.kind is not kind:
            raise ValueError("another method's checkpoint cannot restore drafter state")
        state = checkpoint.draft if checkpoint is not None else DraftCheckpoint()
        assert state is not None
        if reads_target and (state.pending is not None or state.open is not None):
            raise ValueError("a head reading target history restores no feature rows")
        self.kind, self.width, self.budget = kind, proposals, budget
        self.reads_target = reads_target
        self.position, self.pending, self.open = state.position, state.pending, state.open
        self.head: HeadOperation | None = None
        self.proposal: tuple[int, ...] | None = None
        self.control = None if load is None else DraftController(proposals, load)
        self.generated: int | None = None
        self.verified: int | None = None

    @property
    def form(self) -> DraftForm:
        return DraftForm.BLOCK if self.kind is MethodKind.DFLASH else DraftForm.CHAINED

    def fork_transition(self) -> DrafterState:
        successor = copy.copy(self)
        successor.control = None if self.control is None else self.control.fork()
        return successor

    def _round_state(self) -> DraftRoundState:
        return DraftRoundState(
            DraftHistory.TARGET if self.reads_target else DraftHistory.OWN,
            self.width + 1,
            self.position,
            0 if self.pending is None else len(self.pending.tokens),
            self.open is not None,
        )

    def _ensure_idle(self) -> None:
        if self.head is not None or self.proposal is not None:
            raise ValueError("drafter method already has unresolved work")

    @staticmethod
    def _read(
        reader: FeatureReader | None, features: FeatureRef, start: int, count: int
    ) -> FeatureRows:
        if reader is None or start < 0 or count <= 0 or start + count > features.rows:
            raise ValueError("target features require a valid reader/span")
        rows = reader.read(features, start, count)
        if rows.rows != count:
            raise ValueError("feature reader returned the wrong row count")
        return rows

    def _append(self, tokens: tuple[int, ...], features: FeatureRows) -> None:
        if len(tokens) != features.rows:
            raise ValueError("drafter pairs differ from their feature rows")
        if self.pending is not None:
            tokens = self.pending.tokens + tokens
            features = self.pending.features.concat(features)
        self.pending = PendingRows(tokens, features)

    def _flush(
        self, request: str, keep: int, phase: HeadPhase, draft_from: int | None = None
    ) -> MethodEffects:
        call, _ = self._round_state().flushing(keep)
        if call is None:
            return MethodEffects()
        pending = self.pending
        assert pending is not None
        entered = call.entry_rows
        operation = HeadOperation(
            request,
            phase,
            pending.tokens[:entered],
            pending.features.slice(0, entered),
            call.position,
            form=self.form,
            draft_from=draft_from,
        )
        operation.validate()
        self.pending = (
            PendingRows(pending.tokens[entered:], pending.features.slice(entered, keep))
            if keep
            else None
        )
        self.head = operation
        return MethodEffects((operation,))

    def priming_position(self) -> int | None:
        if (
            not self.reads_target
            and self.pending is None
            and (self.open is None or self.open.host is None)
            and self.head is None
            and self.proposal is None
        ):
            return self.position
        return None

    def prime(
        self,
        request: str,
        tokens: tuple[int, ...],
        next_token: int | None,
        features: FeatureRef | None,
        draft_from: int,
        primed: int | None,
        reader: FeatureReader | None,
    ) -> MethodEffects:
        if not tokens:
            raise ValueError("drafter cannot prime an empty target chunk")
        if self.reads_target:
            if primed is not None:
                raise ValueError("a head reading target history primes nothing")
            self._ensure_idle()
            self.pending = self.open = None
            self.position = checked_add(self.position, len(tokens))
            if next_token is not None:
                if features is None:
                    raise ValueError("closing prompt chunk requires its final target feature")
                self._append((next_token,), self._read(reader, features, len(tokens) - 1, 1))
            return MethodEffects()
        if features is None:
            raise ValueError("drafter priming requires target features")
        if primed is not None:
            if self.priming_position() != self.position or primed > len(tokens):
                raise ValueError("drafter state cannot take a device-primed chunk")
            self.position = checked_add(self.position, primed)
            self.open = None
            if self.kind is MethodKind.MTP:
                boundary = features.boundary
                if boundary is None:
                    raise ValueError("MTP priming requires an independently owned boundary feature")
                if next_token is None:
                    self.open = OpenFeature(device=boundary.downgrade())
                else:
                    self._append((next_token,), self._read(reader, boundary, 0, 1))
            elif next_token is not None:
                self._append((next_token,), self._read(reader, features, len(tokens) - 1, 1))
            return MethodEffects()
        self._ensure_idle()
        rows = self._read(reader, features, 0, len(tokens))
        if self.open is not None:
            self._append(tokens[:1], self.open.read(reader))
            self.open = None
        if len(tokens) > 1:
            self._append(tokens[1:], rows.slice(0, len(tokens) - 1))
        last = rows.slice(len(tokens) - 1, 1)
        if next_token is not None:
            self._append((next_token,), last)
            return self._flush(request, 1, HeadPhase.PRIMING, draft_from)
        self.open = OpenFeature(host=last)
        return self._flush(request, 0, HeadPhase.PRIMING, draft_from)

    def propose(self, request: str, selects: tuple[SelectSpec, ...]) -> Propose:
        if self.proposal is not None:
            count = self.verified
            if count is None:
                count = (
                    self.control.verify(len(self.proposal), len(selects))
                    if self.control is not None
                    else self.budget.verified
                )
            count = min(count, len(self.proposal), len(selects))
            self.verified = count
            return Propose(tokens=self.proposal[:count])
        if (
            self.pending is None
            or self.head is not None
            or self.open is not None
            or not selects
            or len(self.pending.tokens) > self.width + 1
        ):
            return Propose()
        generated = self.generated
        if generated is None:
            generated = (
                self.control.generate(len(selects), self._round_state().target_position())
                if self.control is not None
                else self.budget.generated
            )
        self.generated = generated
        if generated == 0 or generated > len(selects):
            return Propose()
        call, _ = self._round_state().proposing(generated)
        assert call is not None
        operation = HeadOperation(
            request,
            HeadPhase.GENERATION,
            self.pending.tokens,
            self.pending.features,
            call.position,
            selects[:generated],
            self.form,
        )
        operation.validate()
        self.head = operation
        return Propose(operations=(operation,))

    def observe(
        self, request: str, verification: Verification, reader: FeatureReader | None
    ) -> MethodEffects:
        if self.head is not None:
            raise ValueError("cannot observe while a drafter transaction is unresolved")
        features, accepted = verification.features, verification.accepted
        if features is None or not 0 < accepted <= len(verification.inputs):
            raise ValueError("drafter verification requires features and a valid prefix")
        transition = self._round_state().observed(accepted)
        self.proposal, self.generated, self.verified = None, None, None
        if self.control is not None:
            self.control.committed(
                (*verification.inputs[1:accepted], verification.next), publish=False
            )
        if self.open is not None:
            self._append(verification.inputs[:1], self.open.read(reader))
            self.open = None
        if self.pending is not None and self.pending.tokens[-1] != verification.inputs[0]:
            raise ValueError("pending anchor differs from the verified anchor")
        if self.reads_target:
            last = self._read(reader, features, accepted - 1, 1)
            self.pending = None
            self.position = transition.appended.position
            self._append((verification.next,), last)
        else:
            rows = self._read(reader, features, 0, accepted)
            self._append((*verification.inputs[1:accepted], verification.next), rows)
        if self._round_state() != transition.appended:
            raise ValueError("drafter target transition disagrees with feature pairs")
        if transition.maintenance is not None:
            return self._flush(request, 1, HeadPhase.GENERATION)
        return MethodEffects()

    def reconcile(self, operation: HeadOperation, outcome: HeadOutcome) -> None:
        if self.head is None or self.head != operation:
            raise ValueError("drafter outcome differs from its pending transaction")
        if not isinstance(outcome, HeadOutcome) or len(outcome.proposals) != len(
            operation.proposals
        ):
            raise ValueError("drafter outcome has the wrong kind or proposal count")
        if any(selected.status not in (0, 1, 2) for selected in outcome.proposals):
            raise ValueError("drafter outcome has an invalid selection status")
        if not self.reads_target:
            self.position = checked_add(self.position, len(operation.tokens))
        self.head = None
        if operation.proposals:
            self.pending = None
            tokens = []
            for selected in outcome.proposals:
                if selected.status != 0:
                    break
                tokens.append(selected.token)
            self.proposal = tuple(tokens)
            if self.control is not None:
                self.control.drafted(len(operation.proposals), self.proposal)

    def timed(self, launch: TimedLaunch, seconds: float, sequences: int, rows: int) -> None:
        if self.control is not None:
            self.control.timed(launch, seconds, sequences, rows)

    def outlook(self) -> DraftOutlook | None:
        if self.control is not None and self.proposal is not None and self.verified is None:
            return self.control.outlook()
        return None

    def checkpoint(self) -> MethodCheckpoint:
        if self.head is not None or self.proposal is not None:
            raise MethodCheckpointError("cannot checkpoint unresolved method work")
        pending, opened = self.pending, self.open
        if self.reads_target:
            pending, opened = None, None
        elif opened is None and pending is not None:
            last = len(pending.tokens) - 1
            opened = OpenFeature(host=pending.features.slice(last, 1))
            pending = (
                PendingRows(pending.tokens[:last], pending.features.slice(0, last))
                if last
                else None
            )
        return MethodCheckpoint(self.kind, DraftCheckpoint(self.position, pending, opened))

    def evict(self) -> None:
        self.position = 0
        self.pending, self.open, self.head, self.proposal = None, None, None, None
        self.generated, self.verified = None, None

    def reclaimable(self) -> int:
        return DraftCheckpoint(self.position, self.pending, self.open).retained_bytes()

    def publish_learning(self) -> None:
        if self.control is not None:
            self.control.publish()
