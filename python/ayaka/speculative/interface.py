"""Stable proposer contract shared by every speculative method.

A proposer turns committed token snapshots into tentative drafts. It never
mutates request lifecycles, KV page tables or sampling state, and it never
receives tentative tokens as if they were committed: ``ProposalRequest.token_ids``
is the scheduler's ``known_tokens`` snapshot (prompt + published output).
Capability-specific payloads (probabilities, hidden states, trees) are optional
fields, so a proposer that has none does not manufacture placeholder tensors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ayaka.speculative.config import MAX_DRAFT_TOKENS
from ayaka.speculative.mode import SpeculativeMode
from ayaka.speculative.tree import SpecTree
from ayaka.utils.validation import require_int, require_text

if TYPE_CHECKING:
    import torch

    from ayaka.speculative.metadata import SpeculativeVerification

__all__ = [
    "DraftBatch",
    "DraftProposal",
    "ProposalRequest",
    "ProposerCapabilities",
    "ProposerRuntimeState",
    "SpeculativeProposer",
]


@dataclass(frozen=True, slots=True)
class ProposalRequest:
    """One request's committed context and limits for a proposal.

    Attributes:
        request_id: Request identity.
        sequence_epoch: Request incarnation; proposals are never reused across
            an epoch change.
        token_ids: Committed prompt + published output tokens, never drafts.
        max_draft_tokens: Hard upper bound on this proposal's length.
        namespace: Isolation key (tenant + cache salt) for any state shared
            across requests.
        stop_token_ids: Tokens that end generation; a proposal stops after the
            first one because nothing it proposes afterwards can be published.
    """

    request_id: str
    sequence_epoch: int
    token_ids: tuple[int, ...]
    max_draft_tokens: int
    namespace: str = ""
    stop_token_ids: frozenset[int] = frozenset()

    def __post_init__(self) -> None:
        require_text(self.request_id, "proposal request_id")
        require_int(self.sequence_epoch, "proposal sequence_epoch", minimum=1)
        if type(self.token_ids) is not tuple or not self.token_ids:
            raise ValueError("proposal context must be a non-empty tuple of token ids")
        require_int(self.max_draft_tokens, "proposal max_draft_tokens")
        if self.max_draft_tokens > MAX_DRAFT_TOKENS:
            raise ValueError(f"proposals are limited to {MAX_DRAFT_TOKENS} tokens")
        if not isinstance(self.namespace, str):
            raise TypeError("proposal namespace must be a string")
        if not isinstance(self.stop_token_ids, frozenset):
            raise TypeError("stop_token_ids must be a frozenset")


@dataclass(frozen=True, slots=True)
class ProposerRuntimeState:
    """Read-only step facts a proposer may use to size its work."""

    batch_size: int
    step_id: int = 0

    def __post_init__(self) -> None:
        require_int(self.batch_size, "proposer batch_size")
        require_int(self.step_id, "proposer step_id")


@dataclass(frozen=True, slots=True)
class DraftProposal:
    """Tentative candidates for one request.

    Attributes:
        request_id: Owning request.
        token_ids: Draft ids for logical positions ``positions``.
        positions: Logical position of each draft token.
        valid: Per-token validity; invalid drafts are never verified.
        probabilities: Optional ``[k, V]`` draft distribution ``q`` for
            rejection sampling.
        hidden_states: Optional drafter hidden rows for methods that chain
            proposals.
        tree: Optional candidate topology; ``None`` means a linear chain.
    """

    request_id: str
    token_ids: tuple[int, ...]
    positions: tuple[int, ...]
    valid: tuple[bool, ...]
    probabilities: torch.Tensor | None = field(default=None, compare=False)
    hidden_states: torch.Tensor | None = field(default=None, compare=False)
    tree: SpecTree | None = None

    def __post_init__(self) -> None:
        require_text(self.request_id, "draft request_id")
        if type(self.token_ids) is not tuple:
            raise TypeError("draft token_ids must be a tuple")
        if len(self.token_ids) > MAX_DRAFT_TOKENS:
            raise ValueError(f"drafts are limited to {MAX_DRAFT_TOKENS} tokens")
        for token in self.token_ids:
            require_int(token, "draft token id")
        if len(self.positions) != len(self.token_ids) or len(self.valid) != len(self.token_ids):
            raise ValueError("positions and validity must cover every draft token")
        for position in self.positions:
            require_int(position, "draft position")
        if any(type(flag) is not bool for flag in self.valid):
            raise TypeError("draft validity flags must be bool")
        if self.tree is None:
            expected = tuple(range(self.positions[0], self.positions[0] + len(self.positions)))
            if self.positions and self.positions != expected:
                raise ValueError("a linear draft must occupy consecutive positions")
            if any(not flag for flag in self.valid):
                # A linear chain cannot skip a token: everything after an
                # invalid draft is unverifiable, so it must be truncated instead.
                raise ValueError("truncate a linear draft instead of marking tokens invalid")
        elif self.tree.num_nodes != len(self.token_ids):
            raise ValueError("tree drafts need one token per tree node")
        if self.probabilities is not None and self.probabilities.shape[0] != len(self.token_ids):
            raise ValueError("draft probabilities need one row per draft token")

    @classmethod
    def linear(cls, request_id: str, token_ids: tuple[int, ...], start: int) -> DraftProposal:
        """A host chain for positions ``start .. start + len(token_ids) - 1``."""
        return cls(
            request_id,
            token_ids,
            tuple(range(start, start + len(token_ids))),
            (True,) * len(token_ids),
        )

    @property
    def length(self) -> int:
        return len(self.token_ids)


@dataclass(frozen=True, slots=True)
class DraftBatch:
    """Proposals for one step, in request order of the proposal batch."""

    proposals: tuple[DraftProposal, ...]

    def __post_init__(self) -> None:
        ids = [proposal.request_id for proposal in self.proposals]
        if len(set(ids)) != len(ids):
            raise ValueError("a draft batch holds at most one proposal per request")

    @property
    def request_ids(self) -> tuple[str, ...]:
        return tuple(proposal.request_id for proposal in self.proposals)

    @property
    def lengths(self) -> tuple[int, ...]:
        return tuple(proposal.length for proposal in self.proposals)

    def get(self, request_id: str) -> DraftProposal | None:
        return next((p for p in self.proposals if p.request_id == request_id), None)


@dataclass(frozen=True, slots=True)
class ProposerCapabilities:
    """What a proposer produces, declared once.

    ``host_drafts`` proposals exist on the host at plan time and can be placed
    into the immutable step plan; device-resident proposers publish their
    drafts through completion before the step that verifies them.
    """

    mode: SpeculativeMode
    host_drafts: bool
    produces_probabilities: bool = False
    produces_hidden_states: bool = False
    supports_tree: bool = False


class SpeculativeProposer(ABC):
    """Draft source; one instance per serving runtime, host-serialized."""

    @property
    @abstractmethod
    def capabilities(self) -> ProposerCapabilities: ...

    def prepare_request(self, request: ProposalRequest) -> None:
        """Optional warm-up when a request first becomes eligible."""
        del request

    @abstractmethod
    def propose(
        self,
        batch: Sequence[ProposalRequest],
        *,
        max_draft_tokens: int,
        runtime_state: ProposerRuntimeState,
    ) -> DraftBatch:
        """Propose drafts for ``batch``; each proposal honours both length limits.

        Implementations must not mutate request state. They may update caches
        derived from committed tokens, since those never change once published.
        """

    def update_after_verify(self, verification: SpeculativeVerification) -> None:
        """Feedback after completion settled a speculative slice."""
        del verification

    @abstractmethod
    def release_request(self, request_id: str) -> None:
        """Drop all per-request state; idempotent."""

    def close(self) -> None:
        """Release shared state at runtime shutdown."""
        return None

    def stats(self) -> dict[str, Any]:
        """Bounded, request-free diagnostic counters."""
        return {}
