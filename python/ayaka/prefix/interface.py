from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ayaka.handles import KVPageHandle, PrefixHandle, PrefixMatchHandle
from ayaka.prefix.identity import PrefixBlockIdentity, PrefixCacheContext


@dataclass(frozen=True, slots=True)
class PrefixMatch:
    """Full-page match that must be revalidated before attachment."""

    context: PrefixCacheContext
    page_size: int
    matched_tokens: int
    pages: tuple[KVPageHandle, ...]
    terminal_handle: PrefixHandle | None

    def __post_init__(self) -> None:
        if self.page_size <= 0:
            raise ValueError("page_size must be positive")
        if self.matched_tokens < 0 or self.matched_tokens % self.page_size:
            raise ValueError("matched_tokens must contain complete pages")
        if len(self.pages) != self.matched_tokens // self.page_size:
            raise ValueError("prefix pages do not match matched_tokens")
        if (self.terminal_handle is None) != (self.matched_tokens == 0):
            raise ValueError("only an empty prefix match may omit its terminal handle")

    @classmethod
    def empty(cls, *, context: PrefixCacheContext, page_size: int) -> PrefixMatch:
        return cls(
            context=context,
            page_size=page_size,
            matched_tokens=0,
            pages=(),
            terminal_handle=None,
        )

@dataclass(frozen=True, slots=True)
class PrefixLookupResult:
    """Logical match length plus an opaque attachment capability."""

    matched_tokens: int
    handle: PrefixMatchHandle | None

    def __post_init__(self) -> None:
        if self.matched_tokens < 0:
            raise ValueError("matched_tokens must be non-negative")
        if (self.handle is None) != (self.matched_tokens == 0):
            raise ValueError("only an empty prefix lookup may omit its handle")

    @classmethod
    def empty(cls) -> PrefixLookupResult:
        return cls(matched_tokens=0, handle=None)


@dataclass(frozen=True, slots=True)
class PrefixCacheSnapshot:
    """Capacity accounting for the prefix cache."""

    cached_blocks: int
    terminal_entries: int
    cached_tokens: int
    handles: tuple[PrefixHandle, ...]


@dataclass(frozen=True, slots=True)
class CachedBlockInfo:
    """Read-only projection of a cache-owned block for tier bookkeeping."""

    handle: PrefixHandle
    identity: PrefixBlockIdentity
    parent_identity: PrefixBlockIdentity | None
    block_token_ids: tuple[int, ...]
    page: KVPageHandle
    terminal: bool
    children: int
    last_access: int


@runtime_checkable
class PrefixCache(Protocol):
    """Backend-neutral cache contract for complete KV pages."""

    @property
    def page_size(self) -> int: ...

    def match(
        self,
        token_ids: Sequence[int],
        *,
        context: PrefixCacheContext,
    ) -> PrefixMatch: ...

    def insert(
        self,
        token_ids: Sequence[int],
        pages: Sequence[KVPageHandle],
        *,
        context: PrefixCacheContext,
    ) -> PrefixHandle | None: ...

    def release(self, prefix: PrefixHandle, *, safe_epoch: int) -> int: ...

    def evict(self, target_pages: int, *, safe_epoch: int) -> int: ...
