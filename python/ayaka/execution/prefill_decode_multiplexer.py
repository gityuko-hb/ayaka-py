"""PDMux dispatch and page hazards; requests and KV remain owned by the engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ayaka.execution.execution_lane import ExecutionLane, ExecutionLaneConfig
from ayaka.memory.views import ExecutionMemoryView


@dataclass(slots=True)
class _Flight:
    lane: ExecutionLane
    requests: frozenset[str]
    reads: frozenset[int]
    writes: frozenset[int]
    event: torch.cuda.Event | None = None


class PrefillDecodeMultiplexer:
    """Host scopes are synchronous; device work overlaps on independent streams.

    Page conflicts get producer-event dependencies. Same-request overlap is
    rejected because a stream wait alone cannot validate logical progress.
    A failed lane poisons worker admission until every flight has drained.
    """

    def __init__(self, decode: Any, prefill: Any, config: ExecutionLaneConfig) -> None:
        if decode is prefill or any(
            decode.backends[n] is prefill.backends[n] for n in decode.backends
        ):
            raise ValueError("concurrent lanes require separate runners and attention backends")
        if decode.buffers is not prefill.buffers or decode.kv is not prefill.kv:
            raise ValueError("lanes must borrow the same canonical ticket/KV owners")
        self.config = config
        self.lanes = {
            "decode": ExecutionLane("decode", decode, config),
            "prefill": ExecutionLane("prefill", prefill, config),
        }
        self.effective = config.policy if decode.device.type == "cuda" else "serialized"
        self.reason = "requested" if decode.device.type == "cuda" else "CPU has no CUDA streams"
        self.flights: dict[object, _Flight] = {}
        self.closed = False
        prefill._request_ir = decode._request_ir
        for lane in self.lanes.values():
            lane.runner.plain_greedy = True
            # Reuse the scheduler's existing separate-phase admission path.
            lane.runner.supports_mixed_batches = False

    def admit(self, step) -> ExecutionLane:
        if self.closed:
            raise RuntimeError("multiplexer is closed")
        view = step.prepared.memory_view
        if not isinstance(view, ExecutionMemoryView):
            raise ValueError("PDMux grouped KV is not certified")
        phase = "decode" if step.prepared.step.is_pure_decode else "prefill"
        lane = self.lanes[phase]
        if lane.active >= self.config.max_inflight_per_lane:
            raise RuntimeError("lane in-flight quota exhausted")
        requests = frozenset(step.prepared.step.request_order)
        reads = {
            p for sequence in view.sequences for p in sequence.block_table if p != view.padding_page
        }
        writes = {
            slot.physical_page for sequence in view.sequences for slot in sequence.write_slots
        }
        for copy in view.copies:
            reads.add(lane.runner.kv.physical_page(copy.group_name, copy.source))
            writes.add(lane.runner.kv.physical_page(copy.group_name, copy.destination))
        for previous in self.flights.values():
            if requests & previous.requests:
                raise ValueError("cannot concurrently append the same request on two lanes")
            hazard = writes & (previous.reads | previous.writes) or reads & previous.writes
            if previous.lane is not lane and (hazard or self.effective == "serialized"):
                if lane.stream is not None:
                    if previous.event is None:
                        raise RuntimeError("previous lane has no producer fence")
                    lane.stream.wait_event(previous.event)
                lane.dependency_waits += 1
        self.flights[step.ticket_id] = _Flight(lane, requests, frozenset(reads), frozenset(writes))
        lane.active += 1
        lane.submitted += 1
        return lane

    def finish_enqueue(self, ticket_id: object) -> None:
        flight = self.flights[ticket_id]
        if flight.lane.stream is not None:
            flight.event = torch.cuda.Event()
            flight.event.record(flight.lane.stream)

    def retire(self, ticket_id: object) -> None:
        flight = self.flights.pop(ticket_id, None)
        if flight is not None:
            flight.lane.active -= 1

    def lane_for(self, ticket_id: object) -> ExecutionLane | None:
        flight = self.flights.get(ticket_id)
        return None if flight is None else flight.lane

    def report(self) -> dict:
        return {
            "requested": self.config.policy,
            "effective": self.effective,
            "reason": self.reason,
            "lanes": {
                name: {
                    "generation": lane.generation,
                    "submitted": lane.submitted,
                    "active": lane.active,
                    "dependency_waits": lane.dependency_waits,
                }
                for name, lane in self.lanes.items()
            },
        }

    def close(self) -> None:
        if self.closed:
            return
        if self.flights:
            raise RuntimeError("multiplexer requires drain of every lane")
        for lane in self.lanes.values():
            lane.runner.close()
        self.closed = True
