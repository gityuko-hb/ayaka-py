"""Generation-safe resident resume capabilities, including immutable partial tails."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
from hashlib import sha256
from itertools import count

from ayaka.exceptions import InvalidHandleError, InvalidStateTransitionError
from ayaka.memory.manager import RuntimeMemoryManager
from ayaka.memory.sequence import PageTableEntry
from ayaka.prefix.identity import PrefixCacheContext
from ayaka.utils.validation import require_int


def common_resume_boundary(group_boundaries: tuple[tuple[int, ...], ...], limit: int) -> int:
    """Find an actual common checkpoint, never the minimum of longest matches.

    Each group lists positions it can independently restore, including its
    retention/checkpoint restrictions. Absence of a common positive boundary
    means recompute from zero. This helper does not certify recurrent support.
    """
    require_int(limit, "limit")
    if not group_boundaries:
        return 0
    common = set(group_boundaries[0])
    for values in group_boundaries:
        for value in values:
            require_int(value, "resume position")
        common.intersection_update(values)
    return max((x for x in common if x <= limit), default=0)


@dataclass(frozen=True, slots=True)
class ValidResume:
    """Borrowed lookup result; only its originating cache may acquire it.

    ``logical_position`` is an all-layer valid boundary. Page handles include
    allocator generations, while cache/entry IDs prevent an evicted lookup from
    acquiring replacement pages. A lookup conveys no ownership. This baseline
    advertises only full-retention MHA; recurrent and grouped resume are rejected.
    """

    cache_id: int
    entry_id: int
    context: PrefixCacheContext
    token_ids: tuple[int, ...]
    pages: tuple[PageTableEntry, ...]

    @property
    def logical_position(self) -> int:
        return len(self.token_ids)


_CACHE_IDS = count(1)


class PrefixReuse:
    """Bounded prefix registry owning explicit immutable page pins.

    Bind to one runner/ledger/manager. Publish only after successful retirement;
    lookup never pins, and acquisition rechecks the original entry and identity.
    A full prompt match replays its final token, with COW if that token shares a
    physical tail. Entries can be evicted while consumers run: their request and
    execution refs independently protect last use. Caller closes the cache after
    its schedulers; the cache does not own model weights or the KV slab.
    """

    def __init__(self, runner, *, max_entries: int = 32):
        require_int(max_entries, "max_entries", minimum=1)
        self.runner = runner
        self.kv = runner.attention.resources.kv
        if not isinstance(self.kv.backend, RuntimeMemoryManager):
            raise ValueError("P6 prefix resume currently requires homogeneous full-retention MHA")
        self.backend = self.kv.backend
        self.max_entries = max_entries
        self.cache_id = next(_CACHE_IDS)
        self._ids = count(1)
        self._entries: OrderedDict[int, ValidResume] = OrderedDict()
        self._transfers = set()
        self.closed = False
        self.hits = self.misses = self.reused_tokens = 0

    def validate_runner(self, runner):
        if self.closed or runner is not self.runner:
            raise ValueError("prefix cache belongs to another or closed runner")

    def context(self, request) -> PrefixCacheContext:
        """Include every supported execution identity field and tenant salt."""
        self.validate_runner(self.runner)
        c = self.runner.weights.config
        return PrefixCacheContext(
            model_id=self.runner.weights.model_id,
            model_revision=self.runner.weights.revision,
            cache_salt=request.cache.cache_salt,
            rope_config_hash=sha256(repr((c.rope_theta, c.head_dim)).encode()).hexdigest(),
            cache_dtype=str(self.runner.weights.dtype).split(".")[-1],
            cache_layout_version="mha-nhd-v1",
        )

    def _insert(self, context, tokens, pages):
        """Take ownership of already pinned pages; no device work."""
        duplicate = next(
            (
                key
                for key, value in self._entries.items()
                if value.context == context and value.token_ids == tokens
            ),
            None,
        )
        if duplicate is not None:
            self.evict(duplicate)
        result = ValidResume(self.cache_id, next(self._ids), context, tokens, pages)
        self._entries[result.entry_id] = result
        while len(self._entries) > self.max_entries:
            self.evict()
        return result

    def publish(self, sequence, request, known_tokens):
        """Publish a completed prompt boundary; sampled-but-uncomputed tokens are excluded."""
        context = self.context(request)
        state = self.backend.get_sequence(sequence)
        n = min(state.committed_tokens, request.prompt_len)
        if n <= 0:
            return None
        tokens = tuple(known_tokens[:n])
        if tokens != request.prompt_token_ids[:n]:
            raise ValueError("prefix publication token identity differs from request")
        existing = next(
            (v for v in self._entries.values() if v.context == context and v.token_ids == tokens),
            None,
        )
        if existing is not None:
            return existing
        pages = self.backend.pin_prefix(sequence, n)
        try:
            return self._insert(context, tokens, pages)
        except BaseException:
            self.backend.unpin_prefix(pages)
            raise

    @staticmethod
    def _truncate(value, n, page_size):
        pages = tuple(
            PageTableEntry(e.page, min(page_size, n - i * page_size))
            for i, e in enumerate(value.pages[: (n + page_size - 1) // page_size])
        )
        return replace(value, token_ids=value.token_ids[:n], pages=pages)

    def lookup(self, request) -> ValidResume | None:
        """Return the longest matching prefix strictly before the final prompt token."""
        context = self.context(request)
        best = None
        for value in self._entries.values():
            if value.context != context:
                continue
            n = 0
            for a, b in zip(value.token_ids, request.prompt_token_ids[:-1], strict=False):
                if a != b:
                    break
                n += 1
            if n and (best is None or n > best.logical_position):
                best = self._truncate(value, n, self.backend.page_size)
        return best

    def acquire(self, sequence, request, match: ValidResume) -> int:
        """Atomically attach after eviction/identity/generation revalidation; miss is zero."""
        context = self.context(request)
        current = self._entries.get(match.entry_id)
        n = match.logical_position
        if (
            match.cache_id != self.cache_id
            or current is None
            or match.context != context
            or n <= 0
            or n >= request.prompt_len
            or match.token_ids != request.prompt_token_ids[:n]
            or match != self._truncate(current, n, self.backend.page_size)
        ):
            return 0
        try:
            attached = self.backend.attach_pinned_prefix(sequence, match.pages)
        except (InvalidHandleError, InvalidStateTransitionError):
            return 0
        self._entries.move_to_end(match.entry_id)
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
        """Drop one publisher ownership; generation-safe consumer leases remain live."""
        if not self._entries:
            return False
        key = next(iter(self._entries)) if entry_id is None else entry_id
        value = self._entries.pop(key, None)
        if value is None:
            return False
        self.backend.unpin_prefix(value.pages)
        return True

    def close(self) -> bool:
        """Close after transfer last-use; never free pending copy buffers."""
        if any(not transfer.retired for transfer in self._transfers):
            return False
        while self.evict():
            pass
        self.closed = True
        return True
