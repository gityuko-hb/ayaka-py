"""Canonical, checksummed execution envelope broadcast to every step rank.

R11 requires one immutable envelope per ``WorkerStep`` that every rank can
validate without asking rank 0 for mutable state: step identity, execution
plan identity, request/sequence incarnation, mode/query ranges, token and
sampling metadata, KV requirements, graph selection and the collective plan
identity. The payload is canonical JSON with a SHA-256 digest and therefore
carries no raw GPU pointers from the source rank — tensors stay rank-local,
only addresses described as logical values (page ids, slots, positions) move.

The KV part reuses the v2 ``DistributedKVMetadata`` payload unchanged, so a
rank can keep reasoning about block tables through the existing type while
the execution contract travels in one broadcast.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ayaka.distributed.metadata import DistributedKVMetadata
from ayaka.exceptions import InvariantViolationError

if TYPE_CHECKING:
    from ayaka.sched.plan import PreparedStep

__all__ = ["DistributedStepEnvelope", "ENVELOPE_VERSION"]

ENVELOPE_VERSION = 3

#: Field names that would smuggle a device address through the control plane.
_FORBIDDEN_KEYS = frozenset({"pointer", "ptr", "address", "data_ptr", "cuda_ptr", "device_ptr"})

_PLAN_STRING_FIELDS = (
    "execution_plan_id",
    "model_id",
    "model_revision",
    "weights_revision",
    "dtype",
    "kv_dtype",
)
_PLAN_INT_FIELDS = ("layer_start", "layer_stop")
_PARALLEL_FIELDS = ("tp_size", "tp_rank", "pp_size", "pp_rank", "dp_size", "dp_rank")
_SLICE_FIELDS = (
    "request_id",
    "sequence_epoch",
    "state_version",
    "phase",
    "query_start",
    "query_count",
    "sample_last_query",
)


def _require_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise InvariantViolationError(f"envelope field {name} must be a boolean")
    return value


def _require_int(value: Any, name: str) -> int:
    if type(value) is not int or isinstance(value, bool):
        raise InvariantViolationError(f"envelope field {name} must be an integer")
    return value


def _require_text(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise InvariantViolationError(f"envelope field {name} must be a string")
    return value


@dataclass(frozen=True, slots=True)
class DistributedStepEnvelope:
    """One step's execution contract, broadcast and validated on every rank.

    The envelope proves its own integrity: :meth:`verify` re-hashes the
    canonical bytes and checks the structural contract before any caller may
    read :attr:`payload`. Construction happens on the source rank from the
    prepared host plan; every other rank decodes the broadcast bytes and must
    reach the identical checksum.
    """

    step_id: int
    canonical_json: str
    sha256: str

    @classmethod
    def from_prepared(cls, prepared: PreparedStep) -> DistributedStepEnvelope:
        """Build the canonical payload from a prepared host plan."""
        step = prepared.step
        execution = prepared.execution
        kv_view = DistributedKVMetadata.from_execution_view(prepared.memory_view)
        payload = {
            "version": ENVELOPE_VERSION,
            "step_id": int(step.step_id),
            "plan": {
                "execution_plan_id": str(execution.plan_id),
                "model_id": str(execution.model_id),
                "model_revision": str(execution.model_revision),
                "weights_revision": str(execution.weights_revision),
                "dtype": execution.compute.dtype.label,
                "kv_dtype": execution.compute.kv_dtype.label,
                "layer_start": int(execution.compute.layer_range[0]),
                "layer_stop": int(execution.compute.layer_range[1]),
                "parallel": {
                    "tp_size": int(execution.parallel.tp_size),
                    "tp_rank": int(execution.parallel.tp_rank),
                    "pp_size": int(execution.parallel.pp_size),
                    "pp_rank": int(execution.parallel.pp_rank),
                    "dp_size": int(execution.parallel.dp_size),
                    "dp_rank": int(execution.parallel.dp_rank),
                },
            },
            "distributed": (
                None
                if step.distributed is None
                else {
                    "participating_ranks": [
                        int(rank) for rank in step.distributed.participating_ranks
                    ],
                    "collective_sequence": int(step.distributed.collective_sequence),
                    "worker_generation": int(step.distributed.worker_generation),
                }
            ),
            "communication": {
                "overlap_with_compute": bool(step.communication.overlap_with_compute),
                "ops": [
                    {
                        "kind": str(op.kind.value),
                        "group": str(op.group),
                        "nbytes": int(op.nbytes),
                        "dtype": op.dtype.label,
                        "stream": str(op.stream.value),
                        "peer_rank": None if op.peer_rank is None else int(op.peer_rank),
                        "layer_index": (None if op.layer_index is None else int(op.layer_index)),
                    }
                    for op in step.communication.ops
                ],
            },
            "graph": {
                "mode": str(step.graph.mode.value),
                "bucket": int(step.graph.bucket),
                "graph_key": str(step.graph.graph_key),
            },
            "sampling": {
                "num_rows": int(step.sampling.num_rows),
                "num_mask_rows": int(step.sampling.num_mask_rows),
                "all_greedy": bool(step.sampling.all_greedy),
                "any_penalty": bool(step.sampling.any_penalty),
                "any_bias": bool(step.sampling.any_bias),
                "custom_ops": [str(op) for op in step.sampling.custom_ops],
                "argmax_invariant": bool(step.sampling.argmax_invariant),
            },
            "sequences": [
                {
                    "request_id": str(scheduled.request_id),
                    "sequence_epoch": int(scheduled.sequence_epoch),
                    "state_version": int(scheduled.expected_state_version),
                    "phase": str(scheduled.phase.value),
                    "query_start": int(scheduled.query_start),
                    "query_count": int(scheduled.query_count),
                    "sample_last_query": bool(scheduled.sample_last_query),
                }
                for scheduled in step.slices
            ],
            "kv_requirements": [
                {
                    "group_id": int(requirement.group_id),
                    "append_tokens": int(requirement.append_tokens),
                    "cow_pages": int(requirement.cow_pages),
                    "restore_bytes": int(requirement.restore_bytes),
                    "growth_bytes": int(requirement.growth_bytes),
                    "request_tokens": [
                        [str(request_id), int(tokens)]
                        for request_id, tokens in requirement.request_tokens
                    ],
                }
                for requirement in step.kv_requirements
            ],
            "kv": kv_view.payload,
        }
        return cls.from_payload(payload)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> DistributedStepEnvelope:
        """Hash and freeze a raw payload mapping."""
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
    def decode(cls, encoded: str) -> DistributedStepEnvelope:
        """Decode broadcast bytes and verify before first use."""
        outer = json.loads(encoded)
        if not isinstance(outer, dict):
            raise InvariantViolationError("distributed step envelope must decode to an object")
        envelope = cls(
            step_id=int(outer["step_id"]),
            canonical_json=str(outer["canonical_json"]),
            sha256=str(outer["sha256"]),
        )
        envelope.verify()
        return envelope

    def encode(self) -> str:
        """Canonical transport bytes; the checksum must verify first."""
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
        """The validated execution payload; raises when integrity fails."""
        self.verify()
        value = json.loads(self.canonical_json)
        if not isinstance(value, dict):
            raise InvariantViolationError("distributed step envelope must decode to an object")
        return value

    @property
    def kv_metadata(self) -> DistributedKVMetadata:
        """The embedded v2 KV control-plane payload as its own typed view."""
        return DistributedKVMetadata.from_payload(dict(self.payload["kv"]))

    def verify(self) -> None:
        """Re-hash the bytes, then check the structural contract."""
        digest = hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()
        if digest != self.sha256:
            raise InvariantViolationError("distributed step envelope checksum mismatch")
        payload = json.loads(self.canonical_json)
        if not isinstance(payload, dict):
            raise InvariantViolationError(
                "distributed step envelope payload must decode to an object"
            )
        if int(payload.get("version", -1)) != ENVELOPE_VERSION:
            raise InvariantViolationError("unsupported distributed step envelope version")
        if int(payload.get("step_id", -1)) != self.step_id:
            raise InvariantViolationError("distributed step envelope step mismatch")
        _reject_pointers(payload)
        self._validate_plan(payload["plan"])
        self._validate_distributed(payload["distributed"])
        self._validate_communication(payload["communication"])
        self._validate_graph(payload["graph"])
        self._validate_sampling(payload["sampling"])
        self._validate_sequences(payload["sequences"])
        self._validate_kv_requirements(payload["kv_requirements"])
        kv = payload["kv"]
        if not isinstance(kv, dict) or int(kv.get("version", -1)) != 2:
            raise InvariantViolationError("distributed step envelope embeds an invalid KV view")
        if int(kv.get("step_id", -1)) != self.step_id:
            raise InvariantViolationError("embedded KV view belongs to another step")

    def _validate_plan(self, plan: Any) -> None:
        if not isinstance(plan, dict):
            raise InvariantViolationError("envelope plan must be an object")
        for field_name in _PLAN_STRING_FIELDS:
            if not isinstance(plan.get(field_name), str) or not plan[field_name]:
                raise InvariantViolationError(
                    f"envelope plan.{field_name} must be a non-empty string"
                )
        for field_name in _PLAN_INT_FIELDS:
            _require_int(plan.get(field_name), f"plan.{field_name}")
        if plan["layer_stop"] <= plan["layer_start"]:
            raise InvariantViolationError("envelope plan layer range must be non-empty")
        parallel = plan.get("parallel")
        if not isinstance(parallel, dict):
            raise InvariantViolationError("envelope plan.parallel must be an object")
        for field_name in _PARALLEL_FIELDS:
            _require_int(parallel.get(field_name), f"plan.parallel.{field_name}")
        if parallel["tp_size"] < 1 or not 0 <= parallel["tp_rank"] < parallel["tp_size"]:
            raise InvariantViolationError("envelope plan.parallel TP placement is invalid")

    def _validate_distributed(self, distributed: Any) -> None:
        if distributed is None:
            return
        if not isinstance(distributed, dict):
            raise InvariantViolationError("envelope distributed must be an object or null")
        ranks = distributed.get("participating_ranks")
        if not isinstance(ranks, list) or not ranks:
            raise InvariantViolationError("envelope distributed needs participating ranks")
        for rank in ranks:
            _require_int(rank, "distributed rank")
        if len(set(ranks)) != len(ranks):
            raise InvariantViolationError("envelope participating ranks must be unique")
        _require_int(distributed.get("collective_sequence"), "distributed.collective_sequence")
        generation = _require_int(
            distributed.get("worker_generation"), "distributed.worker_generation"
        )
        if generation < 0:
            raise InvariantViolationError("envelope worker generation must be non-negative")

    def _validate_communication(self, communication: Any) -> None:
        if not isinstance(communication, dict):
            raise InvariantViolationError("envelope communication must be an object")
        _require_bool(
            communication.get("overlap_with_compute"), "communication.overlap_with_compute"
        )
        ops = communication.get("ops")
        if not isinstance(ops, list):
            raise InvariantViolationError("envelope communication.ops must be a list")
        for op in ops:
            if not isinstance(op, dict):
                raise InvariantViolationError("envelope communication ops must be objects")
            if not isinstance(op.get("kind"), str) or not op["kind"]:
                raise InvariantViolationError("envelope communication op kind must be a string")
            if not isinstance(op.get("group"), str) or not op["group"]:
                raise InvariantViolationError("envelope communication op group must be a string")
            if _require_int(op.get("nbytes"), "communication op nbytes") < 0:
                raise InvariantViolationError(
                    "envelope communication op nbytes must be non-negative"
                )
            if not isinstance(op.get("dtype"), str) or not op["dtype"]:
                raise InvariantViolationError("envelope communication op dtype must be a string")
            for optional in ("peer_rank", "layer_index"):
                if op.get(optional) is not None:
                    _require_int(op.get(optional), f"communication op {optional}")

    def _validate_graph(self, graph: Any) -> None:
        if not isinstance(graph, dict):
            raise InvariantViolationError("envelope graph must be an object")
        if graph.get("mode") not in ("eager", "capture", "replay"):
            raise InvariantViolationError("envelope graph mode is unknown")
        if _require_int(graph.get("bucket"), "graph.bucket") < 0:
            raise InvariantViolationError("envelope graph bucket must be non-negative")
        if not isinstance(graph.get("graph_key"), str):
            raise InvariantViolationError("envelope graph key must be a string")

    def _validate_sampling(self, sampling: Any) -> None:
        if not isinstance(sampling, dict):
            raise InvariantViolationError("envelope sampling must be an object")
        for field_name in ("num_rows", "num_mask_rows"):
            if _require_int(sampling.get(field_name), f"sampling.{field_name}") < 0:
                raise InvariantViolationError(
                    f"envelope sampling.{field_name} must be non-negative"
                )
        for field_name in ("all_greedy", "any_penalty", "any_bias", "argmax_invariant"):
            _require_bool(sampling.get(field_name), f"sampling.{field_name}")
        if not isinstance(sampling.get("custom_ops"), list):
            raise InvariantViolationError("envelope sampling custom ops must be a list")

    def _validate_sequences(self, sequences: Any) -> None:
        if not isinstance(sequences, list) or not sequences:
            raise InvariantViolationError("envelope sequences must be a non-empty list")
        request_ids: set[str] = set()
        for sequence in sequences:
            if not isinstance(sequence, dict):
                raise InvariantViolationError("envelope sequences must be objects")
            for field_name in _SLICE_FIELDS:
                if field_name not in sequence:
                    raise InvariantViolationError(f"envelope sequence missing {field_name}")
            _require_text(sequence["request_id"], "sequence.request_id")
            if sequence["request_id"] in request_ids:
                raise InvariantViolationError("envelope sequences must be unique by request")
            request_ids.add(sequence["request_id"])
            if _require_int(sequence["sequence_epoch"], "sequence_epoch") < 1:
                raise InvariantViolationError("envelope sequence epoch must be positive")
            _require_int(sequence["state_version"], "sequence.state_version")
            if sequence["phase"] not in ("prefill", "decode"):
                raise InvariantViolationError("envelope sequence phase is unknown")
            _require_int(sequence["query_start"], "sequence.query_start")
            if _require_int(sequence["query_count"], "sequence.query_count") < 1:
                raise InvariantViolationError("envelope sequence query count must be positive")
            _require_bool(sequence["sample_last_query"], "sequence.sample_last_query")

    def _validate_kv_requirements(self, requirements: Any) -> None:
        if not isinstance(requirements, list):
            raise InvariantViolationError("envelope kv_requirements must be a list")
        group_ids: set[int] = set()
        for requirement in requirements:
            if not isinstance(requirement, dict):
                raise InvariantViolationError("envelope kv requirements must be objects")
            for field_name in (
                "group_id",
                "append_tokens",
                "cow_pages",
                "restore_bytes",
                "growth_bytes",
            ):
                _require_int(requirement.get(field_name), f"kv requirement {field_name}")
            group_id = requirement["group_id"]
            if group_id in group_ids:
                raise InvariantViolationError("envelope kv requirements must be unique by group")
            group_ids.add(group_id)
            attributed = 0
            request_tokens = requirement.get("request_tokens")
            if not isinstance(request_tokens, list):
                raise InvariantViolationError("envelope kv request tokens must be a list")
            for request_id, tokens in request_tokens:
                _require_text(request_id, "kv requirement request id")
                if _require_int(tokens, "kv requirement tokens") < 1:
                    raise InvariantViolationError("envelope kv requirement tokens must be positive")
                attributed += tokens
            if request_tokens and attributed != requirement["append_tokens"]:
                raise InvariantViolationError(
                    "envelope kv request attribution must sum to append_tokens"
                )


def _reject_pointers(node: Any) -> None:
    """Walk the payload and reject anything that smells like a device address."""
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(key, str) and key.lower() in _FORBIDDEN_KEYS:
                raise InvariantViolationError(
                    f"distributed step envelope field {key!r} may not carry a raw pointer"
                )
            _reject_pointers(value)
    elif isinstance(node, (list, tuple)):
        for item in node:
            _reject_pointers(item)
