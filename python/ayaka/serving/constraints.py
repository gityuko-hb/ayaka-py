"""Request-local grammar state advanced only from published output tokens."""

from __future__ import annotations

import json

import torch

from ayaka.request.input import ConstraintSpec


class GrammarConstraints:
    """Compile schemas once, own one matcher per admitted request."""

    def __init__(self, tokenizer_info, *, cache_limit_bytes=64 << 20):
        import xgrammar as xgr

        self._xgr = xgr
        self._compiler = xgr.GrammarCompiler(
            tokenizer_info, max_threads=1, cache_limit_bytes=cache_limit_bytes
        )
        self._states = {}
        self.vocab_size = tokenizer_info.vocab_size

    def register(self, request_id: str, constraint: ConstraintSpec) -> None:
        if request_id in self._states:
            raise ValueError("duplicate grammar request")
        schema = constraint.schema_json or json.dumps({"type": "object"})
        try:
            compiled = self._compiler.compile_json_schema(
                schema, any_whitespace=False, separators=(",", ":")
            )
        except (RuntimeError, ValueError) as exc:
            raise ValueError("schema is unsupported by the grammar compiler") from exc
        matcher = self._xgr.GrammarMatcher(compiled)
        bits = self._xgr.allocate_token_bitmask(1, self.vocab_size)
        self._states[request_id] = [matcher, bits, ()]

    def apply(self, request_id: str, published: tuple[int, ...], logits: torch.Tensor) -> None:
        matcher, bits, committed = self._states[request_id]
        if published[: len(committed)] != committed:
            raise ValueError("grammar publication history changed")
        for token in published[len(committed) :]:
            if not matcher.accept_token(token):
                raise ValueError("published token violates grammar")
        self._states[request_id][2] = published
        matcher.fill_next_token_bitmask(bits)
        ids = torch.arange(self.vocab_size, device=logits.device)
        words = bits.to(device=logits.device, dtype=torch.int64)[0]
        allowed = ((words[ids // 32] >> (ids % 32)) & 1).bool()
        logits.masked_fill_(~allowed, float("-inf"))
        if not bool(torch.isfinite(logits).any()):
            raise ValueError("grammar and sampling constraints leave no valid token")

    def forget(self, request_id: str) -> None:
        self._states.pop(request_id, None)
