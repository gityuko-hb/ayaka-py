"""In-process fake ranks for the distributed step protocol (CPU, no device collectives).

``LoopbackGroup`` gives each rank thread a ``DistributedControlPlane`` whose
collectives are ``threading.Barrier`` exchanges with deadlines. A broken
barrier raises ``DistributedCollectiveTimeout`` on every rank, as a lost peer
does for NCCL/Gloo, and a broken group stays broken: a communicator failure
has no transparent recovery. Ranks that enter different collectives (one votes
while another broadcasts) fail on every rank instead of reading each other's
payloads, which is what a real communicator would hang or corrupt on.

Every rank is a real ``ResidentKVEngine`` (scheduler, KV allocator,
``LocalWorker``, executor, completion coordinator) wrapped in a
``DistributedStepWorker``. The logits come from a deterministic function of
the previous token, so ranks agree exactly and a fault is the only thing that
can make them diverge. This proves host-side protocol contracts, never
collective performance or device safety.

Request incarnations (``sequence_epoch``) come from a process-wide counter, so
threads in one process would allocate different epochs for the same request.
``rank_local_epochs`` gives every thread its own counter, as separate rank
processes have.
"""

from __future__ import annotations

import itertools
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
from ayaka.configs.cache import CacheConfig
from ayaka.configs.model import ArchitectureConfig
from ayaka.configs.scheduler import SchedulerConfig
from ayaka.distributed.execution import DistributedControlPlane
from ayaka.distributed.process_group import DistributedCollectiveTimeout
from ayaka.request.schema import Request, StopCriteria
from ayaka.runner.sampling_runner import SamplingRunner
from ayaka.runtime.resident import ResidentKVEngine
from ayaka.sampling.engine import SamplingCoordinator
from ayaka.types import DType
from ayaka.worker.distributed import DistributedStepWorker
from ayaka.worker.local import LocalWorker

VOCAB = 64


def next_token(previous: int) -> int:
    return (previous * 7 + 3) % (VOCAB - 4) + 4


@contextmanager
def rank_local_epochs() -> Iterator[None]:
    """Give each thread its own request-incarnation counter while active."""
    import ayaka.request.lifecycle as lifecycle_module
    import ayaka.request.machine as machine_module

    local = threading.local()
    original = machine_module.next_request_epoch

    def next_epoch() -> int:
        counter = getattr(local, "counter", None)
        if counter is None:
            counter = local.counter = itertools.count(1)
        return next(counter)

    machine_module.next_request_epoch = next_epoch
    lifecycle_module.next_request_epoch = next_epoch
    try:
        yield
    finally:
        machine_module.next_request_epoch = original
        lifecycle_module.next_request_epoch = original


class LoopbackGroup:
    """Shared state of one fake process group."""

    def __init__(self, world_size: int, *, timeout_s: float = 5.0) -> None:
        self.world_size = world_size
        self.timeout_s = timeout_s
        self._barrier = threading.Barrier(world_size)
        self._slots: list[Any] = [None] * world_size
        self.collectives = [0] * world_size

    def plane(self, rank: int) -> LoopbackPlane:
        return LoopbackPlane(self, rank)

    def exchange(self, rank: int, kind: str, value: Any, timeout_s: float | None) -> list[Any]:
        timeout = self.timeout_s if timeout_s is None else timeout_s
        try:
            self._slots[rank] = (kind, value)
            self._barrier.wait(timeout)
            entries = list(self._slots)
            self._barrier.wait(timeout)  # nobody overwrites a slot before all read
        except threading.BrokenBarrierError as exc:
            raise DistributedCollectiveTimeout(
                f"loopback {kind} broke or timed out on rank {rank}"
            ) from exc
        kinds = [entry[0] for entry in entries]
        if len(set(kinds)) != 1:
            self._barrier.abort()  # the communicator is unusable from here on
            raise DistributedCollectiveTimeout(f"collective sequence diverged: {kinds}")
        self.collectives[rank] += 1
        return [entry[1] for entry in entries]


class LoopbackPlane(DistributedControlPlane):
    def __init__(self, group: LoopbackGroup, rank: int) -> None:
        self._group = group
        self._rank = rank

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._group.world_size

    def all_grant(self, local_grant: bool, *, timeout_s: float | None = None) -> bool:
        return all(self._group.exchange(self._rank, "all_grant", bool(local_grant), timeout_s))

    def broadcast_text(
        self, text: str | None, *, source_rank: int, timeout_s: float | None = None
    ) -> str:
        received = self._group.exchange(self._rank, "broadcast", text, timeout_s)[source_rank]
        assert isinstance(received, str)
        return received

    def barrier(self, *, timeout_s: float | None = None) -> None:
        self._group.exchange(self._rank, "barrier", None, timeout_s)

    def shutdown(self, local_accepting: bool, *, timeout_s: float | None = None) -> bool:
        agreed = self.all_grant(local_accepting, timeout_s=timeout_s)
        if agreed:
            self.barrier(timeout_s=timeout_s)
        return agreed


@dataclass
class ScriptedLogits:
    """Deterministic logits plus a log of every token it made the runner sample.

    ``fault`` runs before the logits of each call with the call index, so a
    test can raise or stall on one rank at one step.
    """

    fault: Callable[[int], None] | None = None
    sampled: list[tuple[int, tuple[int, ...]]] = field(default_factory=list)

    def __call__(self, step: Any) -> torch.Tensor:
        if self.fault is not None:
            self.fault(len(self.sampled))
        logits = torch.zeros(len(step.sampling_rows), VOCAB)
        chosen = []
        for index, row in enumerate(step.sampling_rows):
            token = next_token(int(step.token_ids[row]))
            logits[index, token] = 10.0
            chosen.append(token)
        self.sampled.append((step.step_id, tuple(chosen)))
        return logits


def build_engine(
    plane: DistributedControlPlane | None = None,
    *,
    logits: ScriptedLogits | None = None,
    weights_revision: str = "w",
    step_timeout_s: float = 2.0,
) -> ResidentKVEngine:
    """One rank's engine; ``plane=None`` builds the single-rank reference."""
    sampling = SamplingCoordinator(4, vocab_size=VOCAB)
    runner = SamplingRunner(sampling, logits or ScriptedLogits(), force_reference=True)
    factory = None
    if plane is not None:

        def factory(resources, kv, runner, max_inflight):
            local = LocalWorker(kv, runner, max_inflight=max_inflight, resources=resources)
            return DistributedStepWorker(
                plane, local, step_timeout_s=step_timeout_s, shutdown_timeout_s=step_timeout_s
            )

    return ResidentKVEngine.from_configs(
        ArchitectureConfig(
            hidden_size=64,
            num_layers=2,
            num_attention_heads=4,
            num_kv_heads=2,
            vocab_size=VOCAB,
            intermediate_size=128,
        ),
        CacheConfig(block_size=16, kv_dtype=DType.BF16),
        runner=runner,
        device_total_bytes=64 << 20,
        plan_id="fake-ranks",
        model_id="scripted",
        model_revision="r",
        weights_revision=weights_revision,
        scheduler_config=SchedulerConfig(
            max_num_seqs=4, max_num_requests=4, max_num_batched_tokens=64
        ),
        compute_dtype=DType.BF16,
        sampling=sampling,
        max_model_len=128,
        worker_factory=factory,
    )


def requests(count: int = 3, *, max_tokens: int = 6, extra: int = 0) -> list[Request]:
    """``extra`` lengthens the last prompt: a way to make one rank diverge."""
    return [
        Request(
            request_id=f"r{index}",
            prompt_token_ids=tuple(range(5, 9 + 3 * index + (extra if index == count - 1 else 0))),
            stop=StopCriteria(max_tokens=max_tokens),
        )
        for index in range(count)
    ]


@dataclass
class RankResult:
    #: (request_id, completion_tokens, finish_reason) per published event.
    events: list[tuple[str, int, str | None]] = field(default_factory=list)
    #: Terminal state of every submitted request, by id.
    states: dict[str, str] = field(default_factory=dict)
    step_error: BaseException | None = None
    close_error: BaseException | None = None
    closed: bool = False
    steps: int = 0


class _Recorder:
    def __init__(self, result: RankResult) -> None:
        self.result = result

    def on_output_event(self, event: Any) -> None:
        reason = None if event.finish_reason is None else str(event.finish_reason.value)
        self.result.events.append((event.request_id, event.usage.completion_tokens, reason))


def drive(
    engine: ResidentKVEngine, batch: Sequence[Request], *, max_steps: int = 200
) -> RankResult:
    """Submit, step until idle, then close; never raises (the result records it)."""
    result = RankResult()
    engine.output.set_listener(_Recorder(result))
    lifecycles = [engine.submit(request) for request in batch]
    try:
        while engine.has_unfinished and result.steps < max_steps:
            engine.step()
            result.steps += 1
    except BaseException as exc:
        result.step_error = exc
    result.states = {lc.request_id: lc.state.value for lc in lifecycles}
    try:
        result.closed = bool(engine.close().closed)
    except BaseException as exc:
        result.close_error = exc
    return result


def run_ranks(
    engines: Sequence[ResidentKVEngine],
    batches: Sequence[Sequence[Request]],
    *,
    timeout_s: float = 25.0,
) -> list[RankResult]:
    """Drive every rank on its own thread (each with its own epoch counter)."""
    results: list[RankResult | None] = [None] * len(engines)

    def run(rank: int) -> None:
        results[rank] = drive(engines[rank], batches[rank])

    threads = [threading.Thread(target=run, args=(rank,)) for rank in range(len(engines))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout_s)
        assert not thread.is_alive(), "a fake rank never finished"
    assert all(result is not None for result in results)
    return [result for result in results if result is not None]


def _main() -> None:
    """One real rank process over Gloo: ``fake_ranks.py RANK WORLD STORE OUT [FAULT_CALL]``."""
    import json
    import sys

    import torch.distributed as dist
    from ayaka.distributed.process_group import TorchDistributedKVProcessGroup

    rank, world, store, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
    fault_call = int(sys.argv[5]) if len(sys.argv) > 5 else None
    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=world)
    plane = TorchDistributedKVProcessGroup(default_timeout_s=5.0, owns_process_group=True)
    logits = ScriptedLogits()
    if fault_call is not None and rank == world - 1:

        def fault(index: int) -> None:
            if index == fault_call:
                raise RuntimeError("injected device failure")

        logits.fault = fault
    result = drive(build_engine(plane, logits=logits, step_timeout_s=5.0), requests())
    if result.closed:
        plane.destroy()  # only after the coordinated shutdown handshake
    else:
        plane.destroy(force=True)
    with open(out, "w") as handle:
        json.dump(
            {
                "events": result.events,
                "sampled": logits.sampled,
                "states": result.states,
                "closed": result.closed,
                "close_error": None
                if result.close_error is None
                else type(result.close_error).__name__,
                "backend": plane.backend,
            },
            handle,
        )


if __name__ == "__main__":
    _main()
