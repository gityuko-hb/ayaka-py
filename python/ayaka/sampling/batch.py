from __future__ import annotations

from dataclasses import dataclass, field

from ayaka.sampling.params import SamplingParams


@dataclass(frozen=True, slots=True)
class AddedRequest:
    slot: int
    params: SamplingParams
    request_index: int = 0


@dataclass(frozen=True, slots=True)
class RemovedRequest:
    slot: int


@dataclass(frozen=True, slots=True)
class MovedRequest:
    src: int
    dst: int


@dataclass(slots=True)
class BatchUpdate:
    batch_size: int
    removed: list[RemovedRequest] = field(default_factory=list)
    added: list[AddedRequest] = field(default_factory=list)
    moved: list[MovedRequest] = field(default_factory=list)

    def sort_removed(self) -> None:
        self.removed.sort(key=lambda r: r.slot, reverse=True)
