"""Capabilities are refused explicitly, before any resource is touched (research F02, E11/E12/E15).

The research found several implementations that exist in source but are *not*
wired into the serving path: internal logprob reports (the HTTP layer has no
payload for them), auxiliary distributed/communication/dependency plans (the
resident KV runtime has no adopter) and grouped COW.  The contract is that such
requests fail with a structured error at the boundary rather than being
accepted and silently dropped.  These tests pin that current truth, so the day a
capability is wired the matching test must be updated deliberately.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from ayaka.handles import SequenceHandle
from ayaka.plan import (
    EMPTY_GRAPH_PLAN,
    EMPTY_MEMORY_PLAN,
    CommOp,
    CommOpKind,
    CommunicationPlan,
    GraphMode,
    GraphPlan,
    MemoryPlan,
    SamplingPlan,
    WeightResidency,
    WeightResidencyPlan,
)
from ayaka.runtime.kv import KVStepRuntime
from ayaka.sched.plan import (
    BatchStepPlan,
    DistributedStepIdentity,
    Phase,
    RequestStepInput,
    ScheduledSlice,
    StepDependency,
)
from ayaka.serving.errors import InvalidRequestError, UnsupportedFeatureError
from ayaka.serving.protocol import normalize
from ayaka.types import DType, MemoryTier

# ---------------------------------------------------------------------------
# HTTP: options with no response payload are rejected, never silently dropped
# ---------------------------------------------------------------------------

_BODIES = {
    "completion": {"model": "ayaka", "prompt": "hi"},
    "chat": {"model": "ayaka", "messages": [{"role": "user", "content": "hi"}]},
}


@pytest.mark.parametrize("protocol", ["completion", "chat"])
@pytest.mark.parametrize("field", ["logprobs", "top_logprobs", "echo", "suffix"])
def test_options_without_a_response_payload_are_rejected(protocol: str, field: str) -> None:
    body = {**_BODIES[protocol], field: 5 if field != "suffix" else "tail"}
    with pytest.raises(UnsupportedFeatureError) as caught:
        normalize(body, protocol)
    assert field in caught.value.message
    assert caught.value.code == "unsupported_feature"
    assert caught.value.status_code == 400


@pytest.mark.parametrize("protocol", ["completion", "chat"])
@pytest.mark.parametrize("field", ["logprobs", "top_logprobs", "echo", "suffix"])
@pytest.mark.parametrize("unset", [None, False])
def test_explicitly_unset_options_are_accepted(protocol: str, field: str, unset: object) -> None:
    spec = normalize({**_BODIES[protocol], field: unset}, protocol)
    assert spec.model == "ayaka"


@pytest.mark.parametrize("value", [2, 0, True, 1.0, "1"])
def test_only_exactly_one_completion_is_supported(value: object) -> None:
    for field in ("n", "best_of"):
        with pytest.raises(UnsupportedFeatureError):
            normalize({**_BODIES["completion"], field: value}, "completion")


def test_one_completion_is_accepted() -> None:
    spec = normalize({**_BODIES["completion"], "n": 1, "best_of": 1}, "completion")
    assert spec.prompt == "hi"


def test_parallel_tool_calls_false_is_rejected_but_true_is_accepted() -> None:
    with pytest.raises(UnsupportedFeatureError):
        normalize({**_BODIES["chat"], "parallel_tool_calls": False}, "chat")
    assert normalize({**_BODIES["chat"], "parallel_tool_calls": True}, "chat").messages


def test_unknown_fields_are_rejected_rather_than_ignored() -> None:
    with pytest.raises(UnsupportedFeatureError, match="unsupported field"):
        normalize({**_BODIES["chat"], "frobnicate": 1}, "chat")


def test_protocol_mixups_are_invalid_requests() -> None:
    with pytest.raises(InvalidRequestError):
        normalize({"model": "ayaka", "prompt": "x", "messages": []}, "chat")
    with pytest.raises(InvalidRequestError):
        normalize({"model": "ayaka", "messages": [{"role": "user", "content": "x"}]}, "completion")


# ---------------------------------------------------------------------------
# Resident KV runtime: unsupported auxiliary plans fail before the allocator
# ---------------------------------------------------------------------------


class _Tripwire:
    """Any use means the runtime reached for a physical resource it must not touch."""

    def __getattr__(self, name: str):
        raise AssertionError(f"allocator touched via .{name} before the capability check")


def _runtime() -> KVStepRuntime:
    executor = SimpleNamespace(closed=False, has_submission_capacity=lambda: True)
    return KVStepRuntime(
        SimpleNamespace(plan_id="plan"),  # type: ignore[arg-type]
        _Tripwire(),  # type: ignore[arg-type]
        SimpleNamespace(executor=executor),  # type: ignore[arg-type]
    )


def _step(**overrides) -> BatchStepPlan:
    sequence = SequenceHandle(1, 1)
    fields = dict(
        step_id=0,
        execution_plan_id="plan",
        slices=(ScheduledSlice("a", 1, 0, 0, 3, Phase.PREFILL, True),),
        inputs=(RequestStepInput("a", sequence, 1, 0, 3, (1, 2, 3), 0, 4),),
        padded_num_tokens=3,
        sampling_rows=(2,),
        sampling=SamplingPlan(num_rows=1),
    )
    fields.update(overrides)
    return BatchStepPlan(**fields)  # type: ignore[arg-type]


_UNSUPPORTED = {
    "dependencies": {"dependencies": (StepDependency("producer", "kv_transfer"),)},
    "distributed": {"distributed": DistributedStepIdentity((0, 1), 1, 1)},
    "communication": {
        "communication": CommunicationPlan(
            ops=(CommOp(CommOpKind.ALL_REDUCE, "tp", 8, DType.FP16),)
        )
    },
    "weight_residency": {
        "weight_residency": WeightResidencyPlan(
            residency=(WeightResidency("w", MemoryTier.DEVICE, MemoryTier.HOST_PINNED, 8),)
        )
    },
    "memory_without_workspace": {
        "memory": MemoryPlan(activation_bytes=1, peak_bytes_estimate=1),
    },
    "graph_without_planner": {"graph": GraphPlan(GraphMode.CAPTURE, 4, "bucket-4")},
}


@pytest.mark.parametrize("name", sorted(_UNSUPPORTED))
def test_unsupported_auxiliary_plans_are_refused_before_any_reservation(name: str) -> None:
    step = _step(**_UNSUPPORTED[name])
    with pytest.raises(ValueError, match="does not support these auxiliary plans"):
        _runtime().prepare(step)  # the tripwire allocator proves nothing was acquired


def test_empty_auxiliary_plans_pass_the_gate_and_reach_the_allocator() -> None:
    assert _step().memory == EMPTY_MEMORY_PLAN and _step().graph == EMPTY_GRAPH_PLAN
    with pytest.raises(AssertionError, match="allocator touched"):
        _runtime().prepare(_step())  # the gate did not refuse; reservation was attempted


def test_a_mismatched_execution_plan_is_refused_before_any_reservation() -> None:
    with pytest.raises(ValueError, match="mismatched execution plan"):
        _runtime().prepare(_step(execution_plan_id="other"))
