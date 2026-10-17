"""Tokenizer binding and transactional per-request llguidance state."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .compiler import Grammar
from .gbnf import GrammarError


class TokenTable(Protocol):
    """Exact model token bytes, including partial UTF-8; never use lossy decode.

    None marks an unusable token. Stop ids must be in range and usable.
    encode must tokenize text without BOS/EOS insertion or normalization.
    """

    def __len__(self) -> int: ...
    def bytes(self, token: int) -> bytes | None: ...
    @property
    def stop_tokens(self) -> frozenset[int]: ...
    def encode(self, text: str) -> Sequence[int]: ...


@dataclass(frozen=True, slots=True)
class ByteTokenTable:
    """Immutable explicit token-byte table for a model vocabulary."""

    tokens: tuple[bytes | None, ...]
    stop_tokens: frozenset[int]
    encoder: Callable[[str], Sequence[int]]

    def __len__(self) -> int:
        return len(self.tokens)

    def bytes(self, token: int) -> bytes | None:
        return self.tokens[token]

    def encode(self, text: str) -> Sequence[int]:
        return self.encoder(text)


@dataclass(frozen=True, slots=True)
class CacheLimits:
    entries: int = 16
    bytes: int = 16 << 20

    def __post_init__(self) -> None:
        if self.entries < 0 or self.bytes < 0:
            raise ValueError("cache limits must be nonnegative")


def _check(matcher: Any) -> None:
    if matcher.is_error():
        raise GrammarError("binding", matcher.get_error())


class _Tokenizer:
    def __init__(self, table: TokenTable, tokens: list[bytes]) -> None:
        self.tokens = tokens
        self.eos_token_id = min(table.stop_tokens)
        self.bos_token_id = None
        self.special_token_ids = sorted(table.stop_tokens)
        self.table = table

    def __call__(self, text: str | bytes) -> list[int]:
        if isinstance(text, bytes):
            text = text.decode("utf-8")
        tokens = list(self.table.encode(text))
        if any(
            not isinstance(t, int)
            or t < 0
            or t >= len(self.table)
            or t in self.table.stop_tokens
            or self.table.bytes(t) is None
            for t in tokens
        ):
            raise GrammarError("binding", "encoder returned an unusable token")
        if b"".join(self.table.bytes(t) or b"" for t in tokens) != text.encode("utf-8"):
            raise GrammarError("binding", "encoder does not round-trip exact token bytes")
        return tokens


class Vocabulary:
    """Model-width tokenizer factory and bounded LRU of prefix-bound matchers.

    Use on one engine thread. Each bind/fork owns a deep copy of native state.
    Cache bytes charge serialized inputs and masks, excluding native memory.
    """

    def __init__(
        self, table: TokenTable, projection: int, limits: CacheLimits | None = None
    ) -> None:
        import llguidance as llg

        limits = limits if limits is not None else CacheLimits()
        stops = frozenset(table.stop_tokens)
        if not len(table) <= projection <= 0xFFFFFFFF or not stops:
            raise GrammarError("binding", "need model-sized projection and explicit stop tokens")
        if any(t < 0 or t >= len(table) or table.bytes(t) is None for t in stops):
            raise GrammarError("binding", "stop tokens must be in range and usable")
        self.table, self.projection, self.limits = table, projection, limits
        self.stops = stops
        words = []
        usable = [0] * ((projection + 31) // 32)
        for token in range(projection):
            data = table.bytes(token) if token < len(table) else None
            if data is not None and not isinstance(data, bytes):
                raise GrammarError("binding", "token spellings must be bytes or None")
            if data is not None and (data or token in stops):
                usable[token // 32] |= 1 << (token % 32)
            words.append(b"" if token in stops or data is None else data)
        self.usable = tuple(usable)
        self.tokenizer = llg.LLTokenizer(
            llg.TokenizerWrapper(_Tokenizer(table, words)),
            n_vocab=projection,
            eos_token=sorted(stops),
            slices=[],
        )
        self._cache: OrderedDict[tuple[Grammar, tuple[int, ...]], tuple[Any, tuple[int, ...], int]]
        self._cache = OrderedDict()
        self.cached_bytes = 0
        self.cache_hits = 0

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer: Any,
        projection: int,
        *,
        stop_tokens: Sequence[int],
        limits: CacheLimits | None = None,
    ) -> Vocabulary:
        """Bind an Ayaka TokenizerLike using its exact decoded byte vocabulary.

        Stop tokens come from the model's generation policy, including all EOS
        aliases. Non-stop special ids and empty vocabulary gaps are disabled.
        """
        stops = frozenset(stop_tokens)
        special = frozenset(tokenizer.all_special_ids)
        raw = tokenizer.get_decoded_vocab()
        tokens = tuple(
            data if token in stops else (data if data and token not in special else None)
            for token, data in enumerate(raw)
        )
        table = ByteTokenTable(
            tokens, stops, lambda text: tokenizer.encode(text, add_special_tokens=False)
        )
        return cls(table, projection, limits)

    @property
    def cached_entries(self) -> int:
        return len(self._cache)

    def bind(self, grammar: Grammar, prefix: Sequence[int] = ()) -> GrammarConstraint:
        """Consume only the grammar's prefix, not the entire conversation prompt.

        Bound prefix tokens do not count toward the generated-token position.
        """
        import llguidance as llg

        prefix = tuple(prefix)
        key = (grammar, prefix)
        if key in self._cache:
            self.cache_hits += 1
            self._cache.move_to_end(key)
            matcher, mask, _ = self._cache[key]
            return GrammarConstraint(self, matcher.deep_copy(), mask)
        if any(not self.is_usable(t) or t in self.stops for t in prefix):
            raise GrammarError("binding", "prefix contains unusable or stop tokens")
        matcher = llg.LLMatcher(
            self.tokenizer,
            grammar.lark,
            log_level=0,
            limits=llg.LLParserLimits(
                initial_lexer_fuel=2_000_000,
                step_lexer_fuel=2_000_000,
                verbose_errors=False,
            ),
        )
        _check(matcher)
        if matcher.get_grammar_warnings():
            raise GrammarError("binding", "; ".join(matcher.get_grammar_warnings()))
        if prefix:
            if matcher.validate_tokens(list(prefix)) != len(prefix):
                raise GrammarError("binding", "grammar rejects its prefix")
            matcher.consume_tokens(list(prefix))
            _check(matcher)
        state = GrammarConstraint(self, matcher)
        mask = state.mask()
        charge = len(grammar.lark.encode()) + len(prefix) * 4 + len(mask) * 4
        if self.limits.entries and charge <= self.limits.bytes:
            while self._cache and (
                len(self._cache) >= self.limits.entries
                or self.cached_bytes + charge > self.limits.bytes
            ):
                _, (_, _, size) = self._cache.popitem(last=False)
                self.cached_bytes -= size
            self._cache[key] = (matcher.deep_copy(), mask, charge)
            self.cached_bytes += charge
        return state

    def is_usable(self, token: int) -> bool:
        return 0 <= token < self.projection and bool(self.usable[token // 32] & (1 << (token % 32)))


class GrammarConstraint:
    """Immutable-style transitions: stage/advance never mutate the parent.

    accepting means EOS is legal; stopped means EOS was explicitly consumed.
    """

    def __init__(
        self,
        vocabulary: Vocabulary,
        matcher: Any,
        mask: tuple[int, ...] | None = None,
        position: int = 0,
        terminal: bool = False,
    ) -> None:
        self.vocabulary, self._matcher = vocabulary, matcher
        self._mask = mask
        self.position, self._terminal = position, terminal

    def fork(self) -> GrammarConstraint:
        return GrammarConstraint(
            self.vocabulary, self._matcher.deep_copy(), self._mask, self.position, self._terminal
        )

    def accepting(self) -> bool:
        result = self._matcher.is_accepting()
        _check(self._matcher)
        return bool(result)

    def stopped(self) -> bool:
        return self._terminal

    def mask(self) -> tuple[int, ...]:
        """Return packed little-endian uint32 allow bits, clipped to model width."""
        if self._mask is None:
            raw = self._matcher.compute_bitmask()
            _check(self._matcher)
            if len(raw) < len(self.vocabulary.usable) * 4:
                raise GrammarError("binding", "matcher mask does not cover model projection")
            words = [int.from_bytes(raw[i : i + 4], "little") for i in range(0, len(raw), 4)]
            self._mask = tuple(a & b for a, b in zip(words, self.vocabulary.usable, strict=False))
        return self._mask

    def allows(self, token: int) -> bool:
        return self.vocabulary.is_usable(token) and bool(
            self.mask()[token // 32] & (1 << (token % 32))
        )

    def advance(self, tokens: Sequence[int]) -> GrammarConstraint:
        """Atomically stage a nonempty accepted token sequence, including optional final EOS."""
        if self._terminal or not tokens:
            raise GrammarError("violation", "transition needs tokens and a live matcher")
        state = self.fork()
        for i, token in enumerate(tokens):
            if not self.vocabulary.is_usable(token):
                raise GrammarError("violation", f"unusable token {token}")
            if token in self.vocabulary.stops:
                if i != len(tokens) - 1 or not state.accepting() or not state.allows(token):
                    raise GrammarError("violation", "stop token must end an accepting transition")
                if not state._matcher.is_stopped():
                    state._matcher.consume_token(token)
                state._terminal = True
            else:
                # Validation accepts legal noncanonical segmentations too;
                # the sampling mask may prefer a canonical fast-forward token.
                if state._matcher.validate_tokens([token]) != 1:
                    raise GrammarError("violation", f"grammar rejects token {token}")
                state._matcher.consume_token(token)
            _check(state._matcher)
            state._mask = None
            state.position += 1
        return state

    stage = advance

    def forced(self, limit: int) -> tuple[int, ...]:
        """Return up to limit backend-forced tokens, excluding stop tokens."""
        if limit <= 0:
            raise ValueError("forced-token allowance must be positive")
        if self._terminal:
            return ()
        result = []
        for token in self._matcher.compute_ff_tokens()[:limit]:
            if token in self.vocabulary.stops:
                break
            result.append(token)
        _check(self._matcher)
        return tuple(result)
