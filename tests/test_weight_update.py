"""Weight updates as one transaction: all-new, all-old or failed (research C05 / R6, CT-W).

The SGLang baseline updates target then draft, can skip the cache flush and
rolls back through an already-mutated config, so a partial failure can leave a
READY server on mixed weights. Ayaka's update drains every owner, backs the
weights up, loads, and either commits (new revision in the prefix identity,
old-generation prefix cache dropped) or restores and verifies the old bytes;
without a verified restore it fails the engine closed.

Unit tests drive ``apply_weight_transaction`` on small modules; the rest load
real safetensors checkpoints into the CPU serving stack and compare served text
with a fresh model loaded from the same checkpoint (golden oracle).
"""

from __future__ import annotations

import asyncio
import tempfile
import threading
import time
from pathlib import Path

import pytest
import torch
from ayaka.model_loader.dummy import truncate_payload, wrong_shape
from ayaka.model_loader.readiness import finalize_loaded_weights, require_ready_weights
from ayaka.models.qwen2 import (
    Qwen2Config,
    Qwen2ForCausalLM,
    load_qwen2_weights,
    write_qwen2_checkpoint,
)
from ayaka.serving.errors import EngineUnavailableError
from ayaka.weights.update import (
    CheckpointWeightSource,
    StateDictWeightSource,
    WeightUpdateStatus,
    apply_weight_transaction,
    weights_digest,
)
from cpu_e2e import CpuServing, prompt_of, tiny_config
from safetensors.torch import load_file, save_file
from torch import nn

PROMPT = prompt_of(*range(10, 30))
OTHER = prompt_of(*range(60, 64))


# ---------------------------------------------------------------------------
# The transaction on small modules
# ---------------------------------------------------------------------------


def _module(seed: int) -> nn.Module:
    torch.manual_seed(seed)
    module = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    finalize_loaded_weights(module)
    return module


def _state(seed: int) -> dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in _module(seed).state_dict().items()}


class _FailsAfter:
    """Copy ``count`` tensors of a good state dict, then raise."""

    identity = "fails-after"

    def __init__(self, state: dict[str, torch.Tensor], count: int) -> None:
        self.state, self.count = state, count

    def apply(self, module: nn.Module) -> None:
        targets = dict(module.named_parameters())
        with torch.no_grad():
            for index, (name, value) in enumerate(self.state.items()):
                if index == self.count:
                    raise OSError("shard read failed")
                targets[name].copy_(value)


def test_a_valid_update_commits_and_attests_the_new_binding() -> None:
    module = _module(0)
    old = weights_digest(module)
    outcome = apply_weight_transaction(
        {"target": module}, {"target": StateDictWeightSource(_state(1))}
    )
    assert outcome.status is WeightUpdateStatus.COMMITTED and outcome.mutated
    assert outcome.old_digests["target"] == old != outcome.new_digests["target"]
    assert outcome.new_digests["target"] == weights_digest(_module(1))
    require_ready_weights(module)  # a fresh readiness proof for the new binding
    assert outcome.backup_bytes == sum(p.numel() * 4 for p in module.parameters())


def test_a_mismatched_state_dict_is_rejected_before_any_copy() -> None:
    module = _module(0)
    old = weights_digest(module)
    state = _state(1)
    state.pop("1.bias")
    outcome = apply_weight_transaction({"target": module}, {"target": StateDictWeightSource(state)})
    assert outcome.status is WeightUpdateStatus.ROLLED_BACK and not outcome.mutated
    assert "missing=['1.bias']" in outcome.detail
    assert weights_digest(module) == old
    require_ready_weights(module)


def test_a_failure_mid_load_is_restored_bit_identically() -> None:
    module = _module(0)
    old = weights_digest(module)
    outcome = apply_weight_transaction({"target": module}, {"target": _FailsAfter(_state(1), 2)})
    assert outcome.status is WeightUpdateStatus.ROLLED_BACK and outcome.mutated
    assert "shard read failed" in outcome.detail
    assert weights_digest(module) == old == outcome.new_digests["target"]
    require_ready_weights(module)


def test_without_a_backup_a_mutating_failure_can_only_fail() -> None:
    module = _module(0)
    outcome = apply_weight_transaction(
        {"target": module}, {"target": _FailsAfter(_state(1), 2)}, backup=False
    )
    assert outcome.status is WeightUpdateStatus.FAILED and outcome.mutated
    assert "no backup" in outcome.detail


def test_without_a_backup_a_failure_before_mutation_is_still_the_old_generation() -> None:
    module = _module(0)
    old = weights_digest(module)
    outcome = apply_weight_transaction(
        {"target": module}, {"target": _FailsAfter(_state(1), 0)}, backup=False
    )
    assert outcome.status is WeightUpdateStatus.ROLLED_BACK and not outcome.mutated
    assert weights_digest(module) == old


def test_target_success_with_draft_failure_restores_both_roles() -> None:
    """CT-W01: never a READY target on G+1 with a draft on G."""
    target, draft = _module(0), _module(10)
    old_target, old_draft = weights_digest(target), weights_digest(draft)
    outcome = apply_weight_transaction(
        {"target": target, "draft": draft},
        {"target": StateDictWeightSource(_state(1)), "draft": _FailsAfter(_state(11), 1)},
    )
    assert outcome.status is WeightUpdateStatus.ROLLED_BACK
    assert (weights_digest(target), weights_digest(draft)) == (old_target, old_draft)


def test_a_probe_rejecting_the_new_weights_rolls_back() -> None:
    module = _module(0)
    old = weights_digest(module)

    def golden(roles) -> None:
        raise AssertionError("logits differ from the golden fixture")

    outcome = apply_weight_transaction(
        {"target": module}, {"target": StateDictWeightSource(_state(1))}, probe=golden
    )
    assert outcome.status is WeightUpdateStatus.ROLLED_BACK and "golden" in outcome.detail
    assert weights_digest(module) == old


def test_a_restore_that_cannot_be_verified_is_failed() -> None:
    module = _module(0)

    class Restructures:
        identity = "restructure"

        def apply(self, target: nn.Module) -> None:
            target.register_parameter("extra", nn.Parameter(torch.zeros(1)))
            raise RuntimeError("loader replaced the module layout")

    outcome = apply_weight_transaction({"target": module}, {"target": Restructures()})
    assert outcome.status is WeightUpdateStatus.FAILED
    assert "restore failed" in outcome.detail


def test_a_restore_whose_digest_differs_is_failed_not_rolled_back() -> None:
    """Copying the backup back is not proof; only the old digest is."""
    module = _module(0)

    class ChangesDtype:
        identity = "dtype"

        def apply(self, target: nn.Module) -> None:
            layer = target[0]
            assert isinstance(layer, nn.Linear)
            layer.weight.data = layer.weight.data.double()  # storage swapped under us
            raise RuntimeError("loader died after re-materializing a tensor")

    outcome = apply_weight_transaction({"target": module}, {"target": ChangesDtype()})
    assert outcome.status is WeightUpdateStatus.FAILED
    assert "do not match the old digest" in outcome.detail


def test_unmaterialized_modules_are_refused_untouched() -> None:
    meta = nn.Linear(2, 2, device="meta")
    outcome = apply_weight_transaction({"target": meta}, {"target": StateDictWeightSource({})})
    assert outcome.status is WeightUpdateStatus.REFUSED and "meta" in outcome.detail


def test_roles_and_sources_must_match() -> None:
    with pytest.raises(ValueError):
        apply_weight_transaction({"target": _module(0)}, {})


# ---------------------------------------------------------------------------
# Real checkpoints through the CPU serving stack
# ---------------------------------------------------------------------------


def _checkpoint(root: Path, seed: int, *, scale: float = 25.0) -> Path:
    """A real HF-layout safetensors checkpoint whose weights give varied outputs."""
    config = Qwen2Config.from_dict(tiny_config())
    write_qwen2_checkpoint(root, config, seed=seed)
    path = root / "model.safetensors"
    tensors = load_file(str(path))
    for name, value in tensors.items():
        if not name.endswith("norm.weight"):
            tensors[name] = value * scale
    save_file(tensors, str(path), metadata={"format": "pt"})
    return root


def _oracle(stack: CpuServing, checkpoint: Path, prompt: str, tokens: int) -> str:
    """Greedy text of a fresh model loaded from ``checkpoint`` (no cache, no scheduler)."""
    model = Qwen2ForCausalLM(
        Qwen2Config.from_dict(tiny_config()), device="cpu", dtype=torch.float32, backend="torch"
    )
    load_qwen2_weights(model, checkpoint)
    sequence = stack.encode(prompt)
    generated = []
    with torch.inference_mode():
        for _ in range(tokens):
            token = int(torch.argmax(model.forward_dense(torch.tensor(sequence))[-1]))
            generated.append(token)
            sequence.append(token)
    return stack.decode(generated)


@pytest.fixture(scope="module")
def checkpoints():
    with tempfile.TemporaryDirectory(prefix="ayaka-weights-") as tmp:
        root = Path(tmp)
        yield {name: _checkpoint(root / name, seed) for name, seed in (("b", 7), ("c", 8))}


@pytest.fixture
def stack():
    with CpuServing() as serving:
        yield serving


def _broken(checkpoints, tmp_path: Path, breaker) -> Path:
    import shutil

    target = tmp_path / "broken"
    shutil.copytree(checkpoints["b"], target)
    breaker(target / "model.safetensors")
    return target


def test_a_checkpoint_update_serves_exactly_the_new_weights(stack: CpuServing, checkpoints) -> None:
    runtime = stack.runtime
    before = stack.generate(PROMPT, max_tokens=12)
    stack.generate(PROMPT, max_tokens=12)  # the prompt is now prefix-cached
    assert stack.memory().cache_owned_pages > 0
    result = runtime.update_weights(runtime.checkpoint_source(checkpoints["b"]), revision="b")
    assert result.committed and result.mutated
    assert (result.revision_before, result.revision_after, result.generation) == ("local", "b", 1)
    assert result.digest_before != result.digest_after
    assert stack.memory().cache_owned_pages == 0  # the old generation's prefixes are gone
    after = stack.generate(PROMPT, max_tokens=12)
    assert after.text.strip() == _oracle(stack, checkpoints["b"], PROMPT, 12)
    assert after.text != before.text
    assert after.finish is not None and after.finish.usage.cached_tokens == 0  # no old KV
    assert stack.service.ready
    stack.assert_quiescent_accounting()


def test_a_truncated_checkpoint_rolls_back_to_the_old_weights(
    stack: CpuServing, checkpoints, tmp_path: Path
) -> None:
    runtime = stack.runtime
    before = stack.generate(PROMPT, max_tokens=12).text
    broken = _broken(checkpoints, tmp_path, lambda path: truncate_payload(path, keep=64))
    result = runtime.update_weights(runtime.checkpoint_source(broken), revision="broken")
    assert result.status is WeightUpdateStatus.ROLLED_BACK
    assert result.revision_after == "local" and result.generation == 0
    assert result.digest_after == result.digest_before
    assert stack.service.ready
    assert stack.generate(PROMPT, max_tokens=12).text == before
    stack.assert_quiescent_accounting()


def test_a_wrong_shape_checkpoint_is_rejected_before_any_tensor_moves(
    stack: CpuServing, checkpoints, tmp_path: Path
) -> None:
    runtime = stack.runtime
    broken = _broken(
        checkpoints, tmp_path, lambda path: wrong_shape(path, "model.norm.weight", (63,))
    )
    result = runtime.update_weights(runtime.checkpoint_source(broken), revision="broken")
    assert result.status is WeightUpdateStatus.ROLLED_BACK and not result.mutated
    assert stack.service.ready


def test_a_late_failure_without_backup_stops_the_engine(stack: CpuServing, checkpoints) -> None:
    def load_then_fail(module: nn.Module, path: Path) -> None:
        load_qwen2_weights(module, path)  # every tensor copied ...
        raise OSError("checksum of the last shard did not verify")  # ... then the verdict

    source = CheckpointWeightSource(checkpoints["b"], load_then_fail, identity="late")
    result = stack.runtime.update_weights(source, revision="b", backup=False)
    assert result.status is WeightUpdateStatus.FAILED and result.mutated
    deadline = time.monotonic() + 10
    while stack.service.ready and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not stack.service.ready  # never READY on unknown weights
    with pytest.raises(EngineUnavailableError):
        stack.generate(PROMPT, max_tokens=2)


def test_no_request_ever_sees_two_generations(stack: CpuServing, checkpoints) -> None:
    """Admitted before BEGIN: old weights to the end. Queued after: new weights."""
    runtime = stack.runtime
    old_text = stack.generate(PROMPT, max_tokens=60).text
    new_text = _oracle(stack, checkpoints["b"], OTHER, 12)

    async def scenario():
        early = asyncio.ensure_future(stack.agenerate(PROMPT, max_tokens=60))
        await asyncio.sleep(0.05)
        update = asyncio.ensure_future(
            asyncio.to_thread(
                runtime.update_weights,
                runtime.checkpoint_source(checkpoints["b"]),
                revision="b",
            )
        )
        await asyncio.sleep(0.05)  # the update is parked, admission is held
        late = await stack.agenerate(OTHER, max_tokens=12)
        return await early, await update, late

    early, result, late = asyncio.run(asyncio.wait_for(scenario(), 60))
    assert result.committed and result.waited_seconds > 0
    assert early.text == old_text  # finished entirely on the old weights
    assert late.text.strip() == new_text  # ran entirely on the new weights
    stack.assert_quiescent_accounting()


def test_abort_running_cancels_live_requests_and_updates_at_once(
    stack: CpuServing, checkpoints
) -> None:
    runtime = stack.runtime

    async def scenario():
        running = asyncio.ensure_future(stack.agenerate(PROMPT, max_tokens=200))
        await asyncio.sleep(0.05)
        result = await asyncio.to_thread(
            runtime.update_weights,
            runtime.checkpoint_source(checkpoints["b"]),
            revision="b",
            abort_running=True,
        )
        return result, await running

    result, running = asyncio.run(asyncio.wait_for(scenario(), 60))
    assert result.committed and len(result.aborted_requests) == 1
    assert running.finish_reason == "cancelled"
    stack.assert_quiescent_accounting()


def test_a_drain_timeout_leaves_the_weights_untouched(stack: CpuServing, checkpoints) -> None:
    runtime = stack.runtime
    expected = stack.generate(PROMPT, max_tokens=200).text

    async def scenario():
        running = asyncio.ensure_future(stack.agenerate(PROMPT, max_tokens=200))
        await asyncio.sleep(0.05)
        result = await asyncio.to_thread(
            runtime.update_weights,
            runtime.checkpoint_source(checkpoints["b"]),
            revision="b",
            timeout=0.1,
        )
        return result, await running

    result, running = asyncio.run(asyncio.wait_for(scenario(), 60))
    assert result.status is WeightUpdateStatus.TIMEOUT and result.blockers
    assert not result.mutated and result.revision_after == "local"
    assert running.text == expected  # never disturbed
    assert stack.generate(PROMPT, max_tokens=200).text == expected  # still the old weights


def test_unsupported_setups_and_reused_revisions_are_refused_untouched(
    stack: CpuServing, checkpoints
) -> None:
    runtime = stack.runtime
    source = runtime.checkpoint_source(checkpoints["b"])
    same = runtime.update_weights(source, revision="local")
    assert same.status is WeightUpdateStatus.REFUSED and "revision must change" in same.detail
    runtime._decode_graph = True  # as if decode graphs were captured
    try:
        graphs = runtime.update_weights(source, revision="b")
    finally:
        runtime._decode_graph = False
    assert graphs.status is WeightUpdateStatus.REFUSED and "graphs" in graphs.detail
    assert runtime.weights_identity() == ("local", 0)
    with pytest.raises(ValueError):
        runtime.update_weights(source, revision=" ")
    with pytest.raises(ValueError):
        runtime.update_weights(source, revision="b", timeout=float("inf"))


def test_a_second_update_while_one_is_parked_is_refused(stack: CpuServing, checkpoints) -> None:
    runtime = stack.runtime

    async def scenario():
        running = asyncio.ensure_future(stack.agenerate(PROMPT, max_tokens=80))
        await asyncio.sleep(0.05)
        first = asyncio.ensure_future(
            asyncio.to_thread(
                runtime.update_weights, runtime.checkpoint_source(checkpoints["b"]), revision="b"
            )
        )
        await asyncio.sleep(0.05)
        second = await asyncio.to_thread(
            runtime.update_weights, runtime.checkpoint_source(checkpoints["c"]), revision="c"
        )
        return await first, second, await running

    first, second, _ = asyncio.run(asyncio.wait_for(scenario(), 60))
    assert first.committed and first.revision_after == "b"
    assert second.status is WeightUpdateStatus.REFUSED and "in progress" in second.detail


def test_shutdown_answers_a_parked_update(checkpoints) -> None:
    serving = CpuServing()
    outcome: list[object] = []

    def update() -> None:
        try:
            outcome.append(
                serving.runtime.update_weights(
                    serving.runtime.checkpoint_source(checkpoints["b"]), revision="b", timeout=60
                )
            )
        except BaseException as exc:  # noqa: BLE001 - recorded for the assertion
            outcome.append(exc)

    async def scenario():
        running = asyncio.ensure_future(serving.agenerate(PROMPT, max_tokens=200))
        await asyncio.sleep(0.05)
        caller = threading.Thread(target=update)
        caller.start()
        await asyncio.sleep(0.1)
        await asyncio.to_thread(serving.close)
        await asyncio.to_thread(caller.join, 10)
        return await running

    asyncio.run(asyncio.wait_for(scenario(), 60))
    assert len(outcome) == 1 and isinstance(outcome[0], EngineUnavailableError)


def test_http_weight_update_route(stack: CpuServing, checkpoints, tmp_path: Path) -> None:
    client = stack.http()
    ok = client.post(
        "/v1/weights/update", json={"checkpoint": str(checkpoints["b"]), "revision": "b"}
    )
    assert ok.status_code == 200, ok.text
    assert ok.json()["status"] == "committed" and ok.json()["generation"] == 1
    same = client.post(
        "/v1/weights/update", json={"checkpoint": str(checkpoints["c"]), "revision": "b"}
    )
    assert same.status_code == 409 and same.json()["status"] == "refused"
    broken = _broken(checkpoints, tmp_path, lambda path: truncate_payload(path, keep=64))
    rolled = client.post("/v1/weights/update", json={"checkpoint": str(broken), "revision": "x"})
    assert rolled.status_code == 422 and rolled.json()["status"] == "rolled_back"
    assert client.post("/v1/weights/update", json={"revision": "x"}).status_code == 400
    bad = client.post(
        "/v1/weights/update",
        json={"checkpoint": str(checkpoints["c"]), "revision": "c", "timeout": -1},
    )
    assert bad.status_code == 400
    assert stack.runtime.weights_identity() == ("b", 1)
