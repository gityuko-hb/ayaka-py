"""Speculative verification payloads on both sides of the completion boundary.

``SpeculativeSampleOutputs`` is device-resident and travels inside
``SampleOutputs`` until completion materializes it; validation here is
shape/dtype only, so building it never synchronizes the device.
``SpeculativeVerification`` is the host record completion produces once the
accepted counts are known and KV has been committed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ayaka.speculative.mode import SpeculativeMode
from ayaka.utils.validation import require_int, require_text

if TYPE_CHECKING:
    from ayaka.sched.plan import BatchStepPlan

__all__ = ["SpeculativeSampleOutputs", "SpeculativeVerification", "VerifyLayout"]


@dataclass(frozen=True, slots=True)
class SpeculativeSampleOutputs:
    """Device acceptance results for the speculative rows of one ticket.

    Attributes:
        accepted: ``[R]`` integer tensor, accepted draft count per speculative
            slice, in the order of ``rows``.
        rows: Sampling-row indices (packed sampling order) of speculative
            slices; strictly increasing.
        draft_counts: Verified draft length per speculative slice.
    """

    accepted: torch.Tensor
    rows: tuple[int, ...]
    draft_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.accepted, torch.Tensor):
            raise TypeError("accepted must be a torch.Tensor")
        if self.accepted.dim() != 1 or self.accepted.size(0) != len(self.rows):
            raise ValueError("accepted must be [R] for R speculative rows")
        if self.accepted.is_floating_point() or self.accepted.is_complex():
            raise ValueError("accepted counts must use an integer dtype")
        if type(self.rows) is not tuple or type(self.draft_counts) is not tuple:
            raise TypeError("rows and draft_counts must be tuples")
        if len(self.draft_counts) != len(self.rows):
            raise ValueError("one draft count is required per speculative row")
        previous = -1
        for row in self.rows:
            require_int(row, "speculative sampling row")
            if row <= previous:
                raise ValueError("speculative sampling rows must be strictly increasing")
            previous = row
        for count in self.draft_counts:
            require_int(count, "speculative draft count", minimum=1)

    def validate_rows(self, sampling_rows: int) -> None:
        """Reject rows outside a ticket with ``sampling_rows`` sampled rows."""
        if self.rows and self.rows[-1] >= sampling_rows:
            raise IndexError(
                f"speculative row {self.rows[-1]} outside the {sampling_rows} sampling rows"
            )


@dataclass(frozen=True, slots=True)
class SpeculativeVerification:
    """Settled outcome of one speculative slice, after the partial KV commit.

    ``published`` counts tokens appended to the request (accepted drafts plus
    the target token, truncated at a stop or budget). ``committed_kv`` equals
    ``published`` for a live request. A discarded request (stale epoch,
    cancelled, terminal) publishes nothing; its committed rows are released with
    its sequence at retirement.
    """

    request_id: str
    sequence_epoch: int
    mode: SpeculativeMode
    proposed: int
    accepted: int
    published: int
    reserved_kv: int
    committed_kv: int
    discarded: bool = False

    def __post_init__(self) -> None:
        require_text(self.request_id, "request_id")
        require_int(self.sequence_epoch, "sequence_epoch", minimum=1)
        if not isinstance(self.mode, SpeculativeMode):
            raise TypeError("mode must be SpeculativeMode")
        require_int(self.proposed, "proposed", minimum=1)
        require_int(self.accepted, "accepted")
        require_int(self.published, "published")
        require_int(self.reserved_kv, "reserved_kv", minimum=2)
        require_int(self.committed_kv, "committed_kv", minimum=1)
        if type(self.discarded) is not bool:
            raise TypeError("discarded must be bool")
        if self.accepted > self.proposed:
            raise ValueError("accepted drafts exceed proposed drafts")
        if self.reserved_kv != self.proposed + 1:
            raise ValueError("reserved KV must cover the base row and every draft")
        if self.committed_kv > self.reserved_kv:
            raise ValueError("committed KV exceeds the reservation")
        if self.discarded:
            if self.published:
                raise ValueError("a discarded slice publishes nothing")
        elif not 1 <= self.published <= self.accepted + 1 or self.committed_kv != self.published:
            raise ValueError("a live slice commits exactly the rows it publishes")

    @property
    def rejected(self) -> int:
        return self.proposed - self.accepted

    @property
    def reclaimed_kv(self) -> int:
        return self.reserved_kv - self.committed_kv


@dataclass(frozen=True, slots=True)
class VerifyLayout:
    """Host row layout of the speculative part of one prepared step.

    ``sampling_rows`` and ``extension_rows`` index the packed execution rows
    (base rows followed by each slice's draft rows). ``spec_positions[j]`` is
    the index in ``sampling_rows`` of speculative slice ``j``; its extension rows
    are ``extension_rows[offsets[j] : offsets[j + 1]]``.
    """

    sampling_rows: tuple[int, ...]
    extension_rows: tuple[int, ...]
    spec_positions: tuple[int, ...]
    offsets: tuple[int, ...]
    draft_token_ids: tuple[tuple[int, ...], ...]
    request_ids: tuple[str, ...]

    @classmethod
    def from_step(cls, step: BatchStepPlan) -> VerifyLayout:
        positions: list[int] = []
        offsets = [0]
        sample_index = {}
        index = 0
        for slice_index, scheduled in enumerate(step.slices):
            if scheduled.sample_last_query:
                sample_index[slice_index] = index
                index += 1
        for plan in step.speculative:
            positions.append(sample_index[plan.slice_index])
            offsets.append(offsets[-1] + plan.effective_k)
        return cls(
            sampling_rows=step.sampling_rows,
            extension_rows=step.extension_rows,
            spec_positions=tuple(positions),
            offsets=tuple(offsets),
            draft_token_ids=tuple(plan.draft_token_ids for plan in step.speculative),
            request_ids=tuple(plan.request_id for plan in step.speculative),
        )

    @property
    def num_speculative(self) -> int:
        return len(self.spec_positions)

    @property
    def draft_counts(self) -> tuple[int, ...]:
        return tuple(b - a for a, b in zip(self.offsets, self.offsets[1:], strict=False))

    @property
    def max_k(self) -> int:
        return max(self.draft_counts, default=0)

    def padded_drafts(self, pad: int = -1) -> list[list[int]]:
        """Draft ids padded to ``max_k``; ``pad`` must never equal a token id."""
        width = self.max_k
        return [list(ids) + [pad] * (width - len(ids)) for ids in self.draft_token_ids]
