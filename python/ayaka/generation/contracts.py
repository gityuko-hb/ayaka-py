"""Device-free contracts replacing the Rust magnitude executor/artifact imports.

These values describe work; they never execute a model or own a KV lease.
A numerical backend must honor the prepare/reconcile/commit protocol.
"""

from __future__ import annotations

import math
import weakref
from dataclasses import dataclass
from enum import IntFlag, StrEnum
from typing import Protocol

from ayaka.utils.validation import require_int

TokenId = int
RequestId = str
I32_MAX = (1 << 31) - 1
U64_MAX = (1 << 64) - 1


def bounded(value: int, name: str, maximum: int = U64_MAX, minimum: int = 0) -> None:
    require_int(value, name, minimum=minimum)
    if value > maximum:
        raise ValueError(f"{name} exceeds {maximum}")


def checked_add(left: int, right: int) -> int:
    value = left + right
    bounded(value, "position/counter")
    return value


class Sampling(StrEnum):
    GREEDY = "greedy"
    RANDOM = "random"


@dataclass(frozen=True, slots=True)
class Shaping:
    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    def validate(self) -> None:
        if not all(
            math.isfinite(x)
            for x in (
                self.temperature,
                self.top_p,
                self.min_p,
                self.repetition_penalty,
                self.frequency_penalty,
                self.presence_penalty,
            )
        ):
            raise ValueError("shaping values must be finite")
        if (
            self.temperature < 0
            or not 0 < self.top_p <= 1
            or not 0 <= self.min_p <= 1
            or self.repetition_penalty <= 0
            or type(self.top_k) is not int
            or self.top_k == 0
            or self.top_k < -1
        ):
            raise ValueError("invalid shaping parameters")

    def uses_history(self) -> bool:
        return (
            self.repetition_penalty != 1
            or self.frequency_penalty != 0
            or self.presence_penalty != 0
        )


class Constraint(Protocol):
    """Independent staged successor; neither fork nor stage may mutate self."""

    def position(self) -> int: ...
    def fork(self) -> Constraint: ...
    def stage(self, tokens: tuple[int, ...]) -> Constraint: ...
    def forced(self, limit: int) -> tuple[int, ...]: ...
    def mask(self) -> tuple[int, ...]: ...


@dataclass(frozen=True, slots=True)
class SelectSpec:
    sampling: Sampling = Sampling.GREEDY
    seed: int = 0
    position: int = 0
    domain: int = 0
    mask: tuple[int, ...] | None = None
    shaping: Shaping = Shaping()
    history: tuple[int, ...] | None = None


class Demand(IntFlag):
    NONE = 0
    SELECT = 1
    FEATURES = 2


class WorkKind(StrEnum):
    PREFILL = "prefill"
    REPLAY = "replay"
    DECODE = "decode"
    VERIFY = "verify"


class BoundaryRule(StrEnum):
    ANY = "any"
    ATOMIC = "atomic"


@dataclass(frozen=True, slots=True)
class InputSpan:
    start: int
    end: int
    language_history: bool = True
    rule: BoundaryRule = BoundaryRule.ATOMIC

    def __post_init__(self) -> None:
        bounded(self.start, "span start")
        bounded(self.end, "span end", minimum=self.start + 1)


@dataclass(frozen=True, slots=True)
class InputLayout:
    count: int
    spans: tuple[InputSpan, ...] = ()

    def __post_init__(self) -> None:
        bounded(self.count, "layout count", I32_MAX, 1)
        object.__setattr__(self, "spans", tuple(self.spans))
        end = 0
        for span in self.spans:
            if span.start < end or span.end > self.count:
                raise ValueError("input spans must be disjoint and inside the prompt")
            end = span.end

    def boundary(self, position: int) -> bool:
        return all(
            span.rule is not BoundaryRule.ATOMIC or not span.start < position < span.end
            for span in self.spans
        )

    def chunk_end(self, start: int, target: int, allowance: int) -> int:
        bounded(start, "chunk start")
        bounded(target, "chunk target", minimum=start + 1)
        bounded(allowance, "chunk allowance", minimum=1)
        if not self.boundary(start) or not self.boundary(target):
            raise ValueError("chunk must start and end at input boundaries")
        end = min(target, checked_add(start, allowance))
        for span in self.spans:
            if span.rule is BoundaryRule.ATOMIC and span.start < end < span.end:
                end = span.start
                break
        if end <= start:
            raise ValueError("allowance cannot hold the next atomic input span")
        return end


@dataclass(frozen=True, slots=True)
class FeatureRows:
    """Owned host bytes with explicit row geometry; no tensor arena is retained."""

    data: bytes
    rows: int
    row_bytes: int

    def __post_init__(self) -> None:
        bounded(self.rows, "feature rows")
        bounded(self.row_bytes, "feature row bytes", minimum=1)
        if type(self.data) is not bytes or len(self.data) != self.rows * self.row_bytes:
            raise ValueError("feature bytes disagree with row geometry")

    def slice(self, start: int, count: int) -> FeatureRows:
        bounded(start, "feature start")
        bounded(count, "feature count")
        if start + count > self.rows:
            raise ValueError("feature span exceeds owned rows")
        return FeatureRows(
            self.data[start * self.row_bytes : (start + count) * self.row_bytes],
            count,
            self.row_bytes,
        )

    def concat(self, other: FeatureRows) -> FeatureRows:
        if other.row_bytes != self.row_bytes:
            raise ValueError("feature rows have incompatible geometry")
        return FeatureRows(self.data + other.data, self.rows + other.rows, self.row_bytes)


@dataclass(frozen=True, slots=True, weakref_slot=True)
class FeatureRef:
    """Opaque backend handle; boundary must be independently owned by the backend."""

    key: str
    rows: int
    boundary: FeatureRef | None = None

    def downgrade(self) -> weakref.ReferenceType[FeatureRef]:
        return weakref.ref(self)


class FeatureReader(Protocol):
    def read(self, features: FeatureRef, start: int, count: int) -> FeatureRows: ...


@dataclass(frozen=True, slots=True)
class Priming:
    tokens: tuple[int, ...]
    position: int
    draft_from: int
    boundary: bool = False


class DraftForm(StrEnum):
    CHAINED = "chained"
    BLOCK = "block"


class HeadPhase(StrEnum):
    PRIMING = "priming"
    GENERATION = "generation"


@dataclass(frozen=True, slots=True)
class ForwardOperation:
    request: str
    kind: WorkKind
    tokens: tuple[int, ...]
    position: int
    demand: Demand
    selects: tuple[SelectSpec, ...]
    committed: int
    prime: Priming | None = None
    conditioning: str | None = None

    def validate(self) -> None:
        bounded(self.position, "forward position")
        if not self.tokens or not 0 <= self.committed <= len(self.tokens):
            raise ValueError("invalid forward input/commit range")
        if bool(self.selects) != bool(self.demand & Demand.SELECT):
            raise ValueError("selection demand disagrees with forward rows")
        if self.kind in (WorkKind.PREFILL, WorkKind.REPLAY):
            if len(self.selects) > 1 or self.committed != len(self.tokens):
                raise ValueError("invalid progress selection/commit range")
            if self.kind is WorkKind.REPLAY and self.selects:
                raise ValueError("replay cannot select")
        elif self.selects and len(self.selects) != len(self.tokens):
            raise ValueError("decode/verify needs one selection per row")


@dataclass(frozen=True, slots=True)
class HeadOperation:
    request: str
    phase: HeadPhase
    tokens: tuple[int, ...]
    conditioning: FeatureRows
    position: int
    proposals: tuple[SelectSpec, ...] = ()
    form: DraftForm = DraftForm.CHAINED
    draft_from: int | None = None

    def validate(self) -> None:
        bounded(self.position, "head position")
        if not self.tokens or len(self.tokens) != self.conditioning.rows:
            raise ValueError("head inputs disagree with conditioning rows")
        if len(self.proposals) > 255:
            raise ValueError("head proposal width exceeds 255")


Operation = ForwardOperation | HeadOperation


@dataclass(frozen=True, slots=True)
class SelectedToken:
    token: int
    status: int = 0


@dataclass(frozen=True, slots=True)
class HeadOutcome:
    proposals: tuple[SelectedToken, ...] = ()
