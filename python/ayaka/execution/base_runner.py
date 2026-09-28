"""Typed phase runner interface; no scheduler, allocator or CUDA fence ownership."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from threading import get_ident
from typing import TYPE_CHECKING, Protocol

import torch

from ayaka.execution.execution_batch import ExecutionBatch
from ayaka.execution.execution_result import ExecutionResult
from ayaka.execution.input_buffers import InputBuffers

if TYPE_CHECKING:
    from ayaka.runner.paged_runner import PagedModelRunner, StagedEagerInputs


class FallbackReason(StrEnum):
    DISABLED = "disabled"
    WRONG_PHASE = "wrong_phase"
    UNSUPPORTED_BACKEND = "unsupported_backend"
    BUCKET_CEILING = "bucket_ceiling"
    MISSING_GRAPH = "missing_graph"
    OUTPUT_MODE = "output_mode"
    STALE_GENERATION = "stale_generation"
    WORKSPACE_GROWTH = "workspace_growth"
    SLOT_UNAVAILABLE = "slot_unavailable"


@dataclass(frozen=True, slots=True)
class RunnerSupport:
    reason: FallbackReason | None = None

    @property
    def supported(self) -> bool:
        return self.reason is None


@dataclass(slots=True)
class PreparedInvocation:
    """Single-use staging permission, borrowed from the adopting ticket."""

    batch: ExecutionBatch
    runner: BaseRunner
    buffers: InputBuffers | None
    eager_inputs: StagedEagerInputs | None = None
    binding: str | None = None
    producer: tuple[int, int | None] | None = None
    consumed: bool = False

    def claim(self, runner: BaseRunner) -> None:
        if self.runner is not runner or self.consumed:
            raise RuntimeError("invocation belongs to another runner or was already submitted")
        self.batch.validate()
        if self.buffers is not None:
            self.buffers.validate()
        # Even a partial launch consumes it. Never retry eager after this point.
        self.consumed = True


def _producer(owner: PagedModelRunner) -> tuple[int, int | None]:
    stream = (
        torch.cuda.current_stream(owner.device).cuda_stream if owner.device.type == "cuda" else None
    )
    return get_ident(), stream


@contextmanager
def load_invocation(
    owner: PagedModelRunner, runner: BaseRunner, batch: ExecutionBatch
) -> Iterator[PreparedInvocation]:
    """One loaded invocation per synchronous lane; snapshots remain independent.

    Prevent restaging pinned inputs or shared attention metadata between load
    and enqueue. A failed load consumes its attempt and the worker must drain.
    """
    previous = owner._pending_execution
    if previous is not None and not previous.consumed:
        lease = previous.batch.prepared.buffers
        if lease is None or not lease.released:
            raise RuntimeError("execute or drain the loaded invocation before loading another")
    lease = batch.prepared.buffers
    invocation = PreparedInvocation(
        batch,
        runner,
        None if lease is None else InputBuffers.borrow(lease),
        producer=_producer(owner),
    )
    owner._pending_execution = invocation
    try:
        yield invocation
    except BaseException:
        invocation.consumed = True
        owner._pending_execution = None
        raise


def claim_invocation(
    owner: PagedModelRunner, runner: BaseRunner, invocation: PreparedInvocation
) -> None:
    if invocation.producer != _producer(owner):
        raise RuntimeError("execute must use the same Python owner and stream as load_batch")
    invocation.claim(runner)
    if owner._pending_execution is not invocation:
        raise RuntimeError("invocation is not the lane's loaded batch")
    owner._pending_execution = None
    if owner.kv is None or owner.kv.generation != invocation.batch.generation:
        raise ValueError("execution generation changed after load_batch")


class BaseRunner(Protocol):
    def supports(self, batch: ExecutionBatch) -> RunnerSupport: ...

    def load_batch(self, batch: ExecutionBatch) -> PreparedInvocation: ...

    def execute(self, invocation: PreparedInvocation) -> ExecutionResult: ...

    def close(self) -> None: ...
