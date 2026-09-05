"""Compatibility identity for complete-page prefix reuse.

This module is deliberately allocator-free.  It turns token pages plus the
execution context into chained SHA-256 identities; it does not decide where
pages live or who owns them.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

_MAX_TOKEN_ID = (1 << 63) - 1
_HASH_DOMAIN = b"ayaka-prefix-block-v1"

@dataclass(frozen=True, slots=True)
class PrefixCacheContext:
    """Execution state that must agree before cached KV can be reused."""

    model_id: str
    model_revision: str | None = None
    adapter_id: str | None = None
    cache_salt: str | None = None
    """Tenant/session namespace. Different salts can never share KV pages."""
    multimodal_hash: str | None = None
    rope_config_hash: str | None = None
    cache_dtype: str = "float16"
    cache_layout_version: str = "mha-nhd-v1"

    def __post_init__(self) -> None:
        required = {
            "model_id": self.model_id,
            "cache_dtype": self.cache_dtype,
            "cache_layout_version": self.cache_layout_version,
        }
        for name, value in required.items():
            if not value:
                raise ValueError(f"{name} must not be empty")
        optional: dict[str, str | None] = {
            "model_revision": self.model_revision,
            "adapter_id": self.adapter_id,
            "cache_salt": self.cache_salt,
            "multimodal_hash": self.multimodal_hash,
            "rope_config_hash": self.rope_config_hash,
        }
        for name, value in optional.items():
            if value == "":
                raise ValueError(f"{name} must be None or a non-empty string")
    def components(self) -> tuple[str | None, ...]:
        """Ordered fields fed into the context seed."""

        return (
            self.model_id,
            self.model_revision,
            self.adapter_id,
            self.cache_salt,
            self.multimodal_hash,
            self.rope_config_hash,
            self.cache_dtype,
            self.cache_layout_version,
        )

@dataclass(frozen=True, slots=True)
class PrefixBlockIdentity:
    """Collision-resistant identity for one complete prefix block."""

    context: PrefixCacheContext
    digest: bytes
    block_index: int
    token_count: int

    def __post_init__(self) -> None:
        if len(self.digest) != hashlib.sha256().digest_size:
            raise ValueError("prefix digest must be a SHA-256 digest")
        if self.block_index < 0:
            raise ValueError("block_index must be non-negative")
        if self.token_count <= 0:
            raise ValueError("token_count must be positive")

    @property
    def digest_hex(self) -> str:
        """Hex form for diagnostics and storage keys."""

        return self.digest.hex()

def build_prefix_block_identities(
    token_ids: Sequence[int],
    *,
    page_size: int,
    context: PrefixCacheContext,
) -> tuple[PrefixBlockIdentity, ...]:
    """Hash complete pages, chaining each block to its entire prefix."""

    blocks = full_token_blocks(token_ids, page_size=page_size)
    return build_identities(blocks, page_size=page_size, context=context)

def full_token_blocks(
    token_ids: Sequence[int],
    *,
    page_size: int,
) -> tuple[tuple[int, ...], ...]:
    """Split token IDs into complete page-size blocks; drop a partial tail."""

    if page_size <= 0:
        raise ValueError("page_size must be positive")
    normalized: list[int] = []
    for token_id in token_ids:
        if (
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or not 0 <= token_id <= _MAX_TOKEN_ID
        ):
            raise ValueError(f"token ids must be integers in [0, {_MAX_TOKEN_ID}]")
        normalized.append(token_id)
    full_token_count = len(normalized) // page_size * page_size
    return tuple(
        tuple(normalized[offset : offset + page_size])
        for offset in range(0, full_token_count, page_size)
    )


def build_identities(
    blocks: Sequence[tuple[int, ...]],
    *,
    page_size: int,
    context: PrefixCacheContext,
) -> tuple[PrefixBlockIdentity, ...]:
    """Build chained identities for already-normalized complete blocks."""

    parent_digest = context_seed(context, page_size=page_size)
    identities: list[PrefixBlockIdentity] = []
    for block_index, block in enumerate(blocks):
        identity = identity_for_block(
            context=context,
            parent_digest=parent_digest,
            block_index=block_index,
            block_token_ids=block,
            page_size=page_size,
        )
        identities.append(identity)
        parent_digest = identity.digest
    return tuple(identities)


def identity_for_block(
    *,
    context: PrefixCacheContext,
    parent_digest: bytes,
    block_index: int,
    block_token_ids: tuple[int, ...],
    page_size: int,
) -> PrefixBlockIdentity:
    """Hash domain + parent digest + position + one token page."""

    digest = hashlib.sha256()
    digest.update(_HASH_DOMAIN)
    digest.update(parent_digest)
    digest.update(block_index.to_bytes(8, "big", signed=False))
    for token_id in block_token_ids:
        digest.update(token_id.to_bytes(8, "big", signed=False))
    return PrefixBlockIdentity(
        context=context,
        digest=digest.digest(),
        block_index=block_index,
        token_count=(block_index + 1) * page_size,
    )


def context_seed(context: PrefixCacheContext, *, page_size: int) -> bytes:
    """Hash compatibility context and page size into the chain root."""

    digest = hashlib.sha256()
    digest.update(_HASH_DOMAIN)
    digest.update(page_size.to_bytes(8, "big", signed=False))
    for component in context.components():
        if component is None:
            digest.update(b"\x00")
            continue
        encoded = component.encode("utf-8")
        digest.update(b"\x01")
        digest.update(len(encoded).to_bytes(8, "big", signed=False))
        digest.update(encoded)
    return digest.digest()
