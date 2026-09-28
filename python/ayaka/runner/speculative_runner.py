"""Greedy speculative lookahead through the existing one-token ticket protocol.

Draft/verify use isolated dense causal state. Verified target KV rows remain
private until a normal prepared ticket reserves the corresponding single input.
Only that row is copied into live KV; the executor commits it after its fence.
Rejected suffixes never enter the canonical page tables. This correctness-first
baseline recomputes prefixes and makes no throughput claim.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from ayaka.configs.speculative import SpeculativeConfig
from ayaka.memory.views import ExecutionMemoryView
from ayaka.model_loader.readiness import adopt_model_weights, weight_binding
from ayaka.request.schema import Request
from ayaka.runner.execution_result import ForwardResult
from ayaka.runner.graph.backend import FullGraphBackend
from ayaka.runner.graph.graph import ShapeKey
from ayaka.runner.graph.pool import execution_enqueue_gate
from ayaka.runner.speculative_batch import SpeculativeExecutionBatch, SpeculativeRole
from ayaka.sampling.acceptance import accept_greedy
from ayaka.sched.plan import PreparedStep, RequestStepInput
from ayaka.types import MaskKind


def require_plain_greedy(request: Request) -> None:
    """Reject unsupported sampling before admission, not after a mutating forward."""
    p = request.sampling
    if (
        not p.is_greedy
        or p.repetition_penalty != 1
        or p.frequency_penalty != 0
        or p.presence_penalty != 0
        or p.logit_bias
        or p.logprobs is not None
        or p.prompt_logprobs is not None
        or p.token_ids_logprobs is not None
        or p.return_sampling_support
        or request.constraint is not None
        or request.multimodal
    ):
        raise ValueError("EP3 speculative/PDMux serving requires plain greedy text requests")


class _DenseRole:
    def __init__(self, model: Any, role: SpeculativeRole, config: SpeculativeConfig) -> None:
        self.model, self.role, self.config = model, role, config
        adopt_model_weights(model)
        self.identity = weight_binding(model)
        self.device = next(model.parameters()).device
        self.instances: dict[int, tuple[Any, ...]] = {}
        self.hits = self.calls = 0

    def forward(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        positions = torch.arange(tokens.numel(), device=tokens.device)
        keys, values = {}, {}

        def attention(layer, query, key, value):
            keys[layer], values[layer] = key.clone(), value.clone()
            q, k, v = (x.transpose(0, 1).unsqueeze(0) for x in (query, key, value))
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                enable_gqa=q.shape[1] != k.shape[1],
                scale=self.model.config.scaling,
            )
            return out.squeeze(0).transpose(0, 1)

        hidden = self.model.forward_hidden(tokens, positions, attention)
        logits = self.model.logits_from_hidden(hidden)
        return (
            logits,
            tuple(keys[i] for i in sorted(keys)),
            tuple(values[i] for i in sorted(values)),
        )

    def capture(self, backend: str) -> None:
        if backend == "eager":
            return
        if self.device.type != "cuda":
            raise ValueError("speculative full graphs require CUDA")
        for bucket in reversed(self.config.token_buckets):
            tokens = torch.zeros(bucket, dtype=torch.long, device=self.device)
            graph = FullGraphBackend(device=self.device)

            def program(t=tokens):
                return self.forward(t)

            self.instances[bucket] = graph, tokens, program
            with graph.capture_session(torch.cuda.Stream(device=self.device)):
                graph.capture_one(
                    ShapeKey(bucket, variant_label=f"{self.role}:{self.config.width}"), program
                )

    def run(
        self, ids: tuple[int, ...]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        if self.identity != weight_binding(self.model):
            raise RuntimeError("speculative model binding changed; drain and rebuild")
        self.calls += 1
        bucket = next((b for b in sorted(self.instances) if b >= len(ids)), None)
        if bucket is None:
            return self.forward(torch.tensor(ids, dtype=torch.long, device=self.device))
        graph, tokens, program = self.instances[bucket]
        tokens.zero_()
        tokens[: len(ids)].copy_(torch.tensor(ids, dtype=torch.long, device=self.device))
        with graph.replay_session():
            output = graph.replay(
                ShapeKey(bucket, variant_label=f"{self.role}:{self.config.width}"),
                forward_fn=program,
            )
        self.hits += 1
        return output

    def close(self) -> None:
        for graph, _, _ in self.instances.values():
            graph.cleanup()
        self.instances.clear()


@dataclass(slots=True)
class TentativeKV:
    """Accepted private rows; cursor follows scheduler-confirmed known tokens."""

    prefix: tuple[int, ...]
    outputs: tuple[int, ...]
    epoch: int
    logits: torch.Tensor
    keys: tuple[torch.Tensor, ...]
    values: tuple[torch.Tensor, ...]

    def row(self, known: tuple[int, ...], epoch: int) -> int | None:
        offset = len(known) - len(self.prefix)
        if (
            epoch != self.epoch
            or not 0 <= offset < len(self.outputs)
            or known != (*self.prefix, *self.outputs[:offset])
        ):
            return None
        return offset


class SpeculativeRunner:
    """Request-scoped lookahead; no direct scheduler or allocator mutation rights."""

    def __init__(self, owner: Any, draft: Any, config: SpeculativeConfig) -> None:
        self.owner, self.config = owner, config
        from ayaka.models.llama import LlamaForCausalLM
        from ayaka.models.qwen import QwenForCausalLM
        from ayaka.models.qwen2 import Qwen2ForCausalLM

        for model in (owner._model, draft):
            if type(model) not in (LlamaForCausalLM, QwenForCausalLM, Qwen2ForCausalLM):
                raise ValueError("speculative baseline supports native Llama/Qwen/Qwen2 only")
            if getattr(model.config, "use_dynamic_ntk", False):
                raise ValueError("speculative dense graphs do not support dynamic NTK")
            if any(getattr(module, "tp_size", 1) != 1 for module in model.modules()):
                raise ValueError("speculative distributed execution is not certified")
            if (
                type(model) in (LlamaForCausalLM, Qwen2ForCausalLM)
                and next(model.parameters()).dtype != torch.float32
            ):
                raise ValueError("Llama/Qwen2 dense speculative verification requires FP32")
        for backend in owner.backends.values():
            spec = backend.group.spec
            if spec.mask is not MaskKind.CAUSAL or spec.logits_soft_cap or spec.has_sinks:
                raise ValueError("speculative dense baseline requires full causal attention")
        self.target = _DenseRole(owner._model, SpeculativeRole.VERIFY, config)
        self.draft = _DenseRole(draft, SpeculativeRole.DRAFT, config)
        if self.draft.device != self.target.device:
            raise ValueError("draft and target must reside on the same device")
        if draft.config.vocab_size != owner._model.config.vocab_size:
            raise ValueError("draft and target vocabulary sizes differ")
        if draft is owner._model or {p.data_ptr() for p in draft.parameters()} & {
            p.data_ptr() for p in owner._model.parameters()
        }:
            raise ValueError("draft and target weights must have independent storage")
        if any(
            b > min(draft.config.max_position_embeddings, owner._max_model_len)
            for b in config.token_buckets
        ):
            raise ValueError("speculative graph bucket exceeds model context")
        self.pending: dict[str, TentativeKV] = {}
        self.proposed = self.accepted = self.rollbacks = self.cache_hits = 0
        self.closed = False
        # Bound retained lookahead plus dense logits/KV temporary work before capture.
        target = owner._model.config
        itemsize = next(owner._model.parameters()).element_size()
        kv_per_token = (
            sum(
                2
                * cache.kv_cache.num_layers
                * cache.kv_cache.key_cache(0).shape[-2]
                * cache.kv_cache.key_cache(0).shape[-1]
                for cache in owner.backends.values()
            )
            * itemsize
        )
        per_token = kv_per_token + target.vocab_size * itemsize
        draft_config = draft.config
        draft_per_token = (
            2
            * draft_config.num_hidden_layers
            * getattr(draft_config, "num_kv_heads", draft_config.num_attention_heads)
            * draft_config.head_dim
            + draft_config.vocab_size
        ) * next(draft.parameters()).element_size()
        slots = owner._coordinator.max_batch_size
        self.reserved_bytes = (
            config.width + 1
        ) * slots * per_token + owner._max_model_len * 2 * max(per_token, draft_per_token)
        if self.reserved_bytes > config.memory_bytes:
            raise MemoryError("speculative state exceeds memory reservation")
        device = self.target.device
        before = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
        started = time.monotonic()
        try:
            with execution_enqueue_gate(), torch.inference_mode():
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                self.draft.capture(config.draft_backend)
                self.target.capture(config.verify_backend)
                if time.monotonic() - started > config.capture_seconds:
                    raise TimeoutError("speculative bootstrap capture time budget exceeded")
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                allocated = max(0, torch.cuda.memory_allocated(device) - before)
                if allocated + self.reserved_bytes > config.memory_bytes:
                    raise MemoryError("speculative captures exceed memory reservation")
                if torch.cuda.max_memory_allocated(device) - before > config.memory_bytes:
                    raise MemoryError("speculative bootstrap peak exceeds memory reservation")
        except BaseException:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            self.close()
            raise

    def _verify(self, value: RequestStepInput) -> TentativeKV:
        known = value.known_tokens
        remaining = value.max_output_tokens - (len(known) - value.prompt_tokens)
        width = min(
            self.config.width,
            remaining - 1,
            self.owner._max_model_len - len(known),
            self.draft.model.config.max_position_embeddings - len(known),
        )
        candidates: list[int] = []
        for _ in range(max(0, width)):
            logits, _, _ = self.draft.run((*known, *candidates))
            row = logits[len(known) + len(candidates) - 1].clone()
            if self.owner._valid_token_mask is not None:
                row.masked_fill_(~self.owner._valid_token_mask, -torch.inf)
            candidates.append(int(row.argmax().item()))
        batch = SpeculativeExecutionBatch(
            self.draft.identity,
            self.target.identity,
            known,
            tuple(candidates),
            self.owner._model.config.vocab_size,
        )
        logits, keys, values = self.target.run((*known, *batch.candidates))
        start = len(known) - 1
        selected = logits[start : start + len(candidates) + 1].clone()
        decision_logits = selected.clone()
        if self.owner._valid_token_mask is not None:
            decision_logits.masked_fill_(~self.owner._valid_token_mask, -torch.inf)
        result = accept_greedy(batch.candidates, tuple(decision_logits.argmax(-1).tolist()))
        self.proposed += len(candidates)
        self.accepted += result.accepted
        self.rollbacks += int(result.accepted != len(candidates))
        count = len(result.tokens)
        return TentativeKV(
            known,
            result.tokens,
            value.sequence_epoch,
            selected[:count].clone(),
            tuple(t[start : start + count].clone() for t in keys),
            tuple(t[start : start + count].clone() for t in values),
        )

    def validate_binding(self) -> None:
        if self.closed:
            raise RuntimeError("speculative runner is closed")
        if self.target.identity != weight_binding(
            self.target.model
        ) or self.draft.identity != weight_binding(self.draft.model):
            raise RuntimeError("speculative model binding changed; drain and rebuild")

    def execute(self, prepared: PreparedStep) -> ForwardResult:
        self.validate_binding()
        if not prepared.step.is_pure_decode or not isinstance(
            prepared.memory_view, ExecutionMemoryView
        ):
            raise ValueError("speculative serving requires homogeneous one-token decode tickets")
        rows = []
        for scheduled, value, view in zip(
            prepared.step.slices, prepared.step.inputs, prepared.memory_view.sequences, strict=True
        ):
            request = self.owner._request_ir[scheduled.request_id]
            require_plain_greedy(request)
            state = self.pending.get(scheduled.request_id)
            offset = None if state is None else state.row(value.known_tokens, value.sequence_epoch)
            if offset is None:
                state = self._verify(value)
                self.pending[scheduled.request_id] = state
                offset = 0
            else:
                self.cache_hits += 1
            assert state is not None
            slot = torch.tensor(
                [view.write_slots[0].flat_slot], dtype=torch.long, device=self.owner.device
            )
            for layer, (name, local) in self.owner.layers.items():
                self.owner.backends[name].kv_cache.store_kv(
                    state.keys[layer][offset : offset + 1],
                    state.values[layer][offset : offset + 1],
                    slot,
                    local,
                )
            if scheduled.sample_last_query:
                rows.append(state.logits[offset : offset + 1].clone())
        return ForwardResult(torch.cat(rows), len(rows))

    def forget(self, request_id: str) -> None:
        self.pending.pop(request_id, None)

    def report(self) -> dict:
        return {
            "algorithm": "greedy_linear",
            "publication": "one_token_per_ticket",
            "proposed": self.proposed,
            "accepted": self.accepted,
            "rollbacks": self.rollbacks,
            "lookahead_hits": self.cache_hits,
            "target_calls": self.target.calls,
            "draft_calls": self.draft.calls,
            "target_graph_hits": self.target.hits,
            "draft_graph_hits": self.draft.hits,
            "pending_requests": len(self.pending),
        }

    def close(self) -> None:
        if self.closed:
            return
        self.pending.clear()
        self.draft.close()
        self.target.close()
        self.closed = True
