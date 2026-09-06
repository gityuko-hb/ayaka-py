from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ayaka.exceptions import InvariantViolationError


@dataclass(frozen=True, slots=True)
class DistributedKVMetadata:
    """Canonical, checksum execution metadata broadcast to every TP rank.

    Tensor payloads remain rank-local. This envelope carries only immutable
    page tables/write slots and therefore preserves the control/data-plane
    split: no allocator object or storage address crosses the process boundary.
    """

    step_id: int
    canonical_json: str
    sha256: str

    @classmethod
    def from_execution_view(cls, view: Any) -> DistributedKVMetadata:
        def write_slot_payload(slot: Any) -> dict[str, int]:
            payload = {
                "logical_position": int(slot.logical_position),
                "physical_page": int(slot.physical_page),
                "page_offset": int(slot.page_offset),
                "flat_slot": int(slot.flat_slot),
            }
            logical_block = getattr(slot, "logical_block", None)
            if logical_block is not None:
                payload["logical_block"] = int(logical_block)
            return payload

        sequences = []
        for sequence_view in view.sequences:
            sequence_payload: dict[str, Any] = {
                "sequence": {
                    "index": int(sequence_view.sequence.index),
                    "generation": int(sequence_view.sequence.generation),
                },
                "reservation": {
                    "index": int(sequence_view.reservation.index),
                    "generation": int(sequence_view.reservation.generation),
                    "step_id": int(sequence_view.reservation.step_id),
                },
                "base_committed_tokens": int(sequence_view.base_committed_tokens),
                "num_reserved_tokens": int(sequence_view.num_reserved_tokens),
            }
            if hasattr(sequence_view, "groups"):
                sequence_payload["groups"] = [
                    {
                        "group_name": group.group_name,
                        "storage_kind": group.storage_kind,
                        "dtype": group.dtype,
                        "page_size": int(group.page_size),
                        "logical_blocks": [int(block) for block in group.logical_blocks],
                        "block_table": [int(page) for page in group.block_table],
                        "write_slots": [
                            write_slot_payload(slot) for slot in group.write_slots
                        ],
                        "attention_token_start": int(group.attention_token_start),
                        "attention_token_stop": int(group.attention_token_stop),
                        "retained_token_start": int(group.retained_token_start),
                        "retained_token_stop": int(group.retained_token_stop),
                        "padding_page": int(group.padding_page),
                        "padding_slot": int(group.padding_slot),
                    }
                    for group in sequence_view.groups
                ]
                state_slot = getattr(sequence_view, "recurrent_state_slot", None)
                sequence_payload["recurrent_state_slot"] = (
                    None if state_slot is None else int(state_slot)
                )
            else:
                sequence_payload["block_table"] = [
                    int(page) for page in sequence_view.block_table
                ]
                sequence_payload["write_slots"] = [
                    write_slot_payload(slot) for slot in sequence_view.write_slots
                ]
            sequences.append(sequence_payload)

        grouped = not hasattr(view, "page_size")
        payload = {
            "version": 2,
            "layout": "grouped" if grouped else "homogeneous",
            "step_id": int(view.step_id),
            "lease": {
                "index": int(view.lease.index),
                "generation": int(view.lease.generation),
                "step_id": int(view.lease.step_id),
            },
            "sequences": sequences,
        }
        if not grouped:
            payload.update(
                {
                    "page_size": int(view.page_size),
                    "padding_page": int(view.padding_page),
                    "padding_slot": int(view.padding_slot),
                }
            )
        return cls.from_payload(payload)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> DistributedKVMetadata:
        canonical = json.dumps(
            dict(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        step_id = int(payload["step_id"])
        return cls(step_id=step_id, canonical_json=canonical, sha256=digest)

    @classmethod
    def decode(cls, encoded: str) -> DistributedKVMetadata:
        outer = json.loads(encoded)
        if not isinstance(outer, dict):
            raise InvariantViolationError(
                "distributed KV metadata envelope must decode to an object"
            )
        metadata = cls(
            step_id=int(outer["step_id"]),
            canonical_json=str(outer["canonical_json"]),
            sha256=str(outer["sha256"]),
        )
        metadata.verify()
        return metadata

    def encode(self) -> str:
        self.verify()
        return json.dumps(
            {
                "step_id": self.step_id,
                "canonical_json": self.canonical_json,
                "sha256": self.sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def payload(self) -> Mapping[str, Any]:
        self.verify()
        value = json.loads(self.canonical_json)
        if not isinstance(value, dict):
            raise InvariantViolationError("distributed KV metadata must decode to an object")
        return value

    def verify(self) -> None:
        digest = hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()
        if digest != self.sha256:
            raise InvariantViolationError("distributed KV metadata checksum mismatch")
        payload = json.loads(self.canonical_json)
        if not isinstance(payload, dict):
            raise InvariantViolationError(
                "distributed KV metadata payload must decode to an object"
            )
        if int(payload.get("version", -1)) != 2:
            raise InvariantViolationError("unsupported distributed KV metadata version")
        if int(payload.get("step_id", -1)) != self.step_id:
            raise InvariantViolationError("distributed KV metadata step mismatch")
