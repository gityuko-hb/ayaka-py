"""Compatibility adapter; prefix state and page ownership live in PrefixService."""

from __future__ import annotations

from ayaka.prefix.identity import PrefixCacheContext, build_prefix_context
from ayaka.prefix.interface import ValidResume
from ayaka.prefix.service import common_resume_boundary as common_resume_boundary
from ayaka.runtime.kv import prefix_prompt_candidate, prompt_logprobs_requires_full_prompt
from ayaka.utils.validation import require_int


class PrefixReuse:
    """Legacy runner API delegating to the runner's canonical PrefixService.

    This adapter has no entry registry, LRU, or durable pins. New runtime code
    should use LogicalKVManager.prefix_service with an explicit context provider.
    The adapter never closes the shared service: the LogicalKVManager owns its
    lifecycle, so closing the adapter cannot break another consumer.
    """

    def __init__(self, runner, *, max_entries: int = 32):
        require_int(max_entries, "max_entries", minimum=1)
        self.runner = runner
        self.kv = runner.attention.resources.kv
        self.service = self.kv.prefix_service
        self.service.require_open()
        # Set-once policy: a second adapter must agree with the first owner.
        self.service.max_entries = max_entries
        self.backend = self.service.backend
        self.cache_id = self.service.cache_id
        self.hits = self.misses = self.reused_tokens = 0

    @property
    def max_entries(self):
        return self.service.max_entries

    @property
    def closed(self):
        return self.service.closed

    def validate_runner(self, runner):
        self.service.require_open()
        if runner is not self.runner:
            raise ValueError("prefix cache belongs to another runner")

    def context(self, request) -> PrefixCacheContext:
        """Canonical execution identity from checkpoint config and KV geometry."""
        self.validate_runner(self.runner)
        weights = self.runner.weights
        return build_prefix_context(
            model_id=weights.model_id,
            model_revision=weights.revision,
            config=weights.config,
            storage_spec=self.kv.storages["default"].storage.spec,
            cache_salt=request.cache.cache_salt,
        )

    def publish(self, sequence, request, known_tokens):
        if not request.cache.store_kv:
            return None
        n = min(self.backend.get_sequence(sequence).committed_tokens, request.prompt_len)
        tokens = tuple(known_tokens[:n])
        if len(tokens) != n or tokens != request.prompt_token_ids[:n]:
            raise ValueError("prefix publication token identity differs from request")
        return self.service.publish(sequence, tokens, context=self.context(request))

    def lookup(self, request) -> ValidResume | None:
        if not request.cache.prefix_cache or prompt_logprobs_requires_full_prompt(request):
            return None
        return self.service.lookup(prefix_prompt_candidate(request), context=self.context(request))

    def acquire(self, sequence, request, match: ValidResume) -> int:
        if not request.cache.prefix_cache or prompt_logprobs_requires_full_prompt(request):
            return 0
        attached = self.service.acquire(
            sequence,
            match,
            token_ids=prefix_prompt_candidate(request),
            context=self.context(request),
        )
        if attached:
            self.hits += 1
            self.reused_tokens += attached
        return attached

    def attach(self, sequence, request):
        match = self.lookup(request)
        tokens = 0 if match is None else self.acquire(sequence, request, match)
        if not tokens:
            self.misses += 1
        return tokens

    def evict(self, entry_id: int | None = None) -> bool:
        return self.service.evict(entry_id)

    def close(self) -> bool:
        """Retire the adapter only; the shared service stays open for its owner."""
        return True
