from __future__ import annotations

import enum


class EvictionPolicy(enum.StrEnum):
    """Eviction strategies for physical KV-cache block managers and memory tiers.

    Defines the ordering and heuristic policies used by the cache allocator when
    reclaiming blocks under GPU VRAM pressure or migrating blocks between
    :class:`MemoryTier` levels.
    """

    LRU = "lru"
    """Least Recently Used: evicts blocks with the oldest access timestamp.
    Standard baseline for flat, unindex block allocators."""

    LFU = "lfu"
    """Least Frequently Used: evicts blocks with the lowest total access counts.
    Protects long-lived, high-frequency system prompts."""

    SLRU = "slru"
    """Segmented LRU: partitions memory into probationary and protected segments.
    Prevents cache pollution from large, single-use prompt bursts."""

    PREFIX_AWARE = "prefix_aware"
    """Radix/Trie structure-aware eviction: evicts leaf nodes before branching roots.
    Maximizes prefix-sharing hit rates across concurrent conversational sessions."""

    COST_AWARE = "cost_aware"
    """Cost-sensitive eviction: factors in prompt recomputation FLOPs and PCIe transfer
    latencies to minimize overall token generation delay."""

    PRIORITY = "priority"
    """Strict priority-driven eviction: reclaims memory according to explicit request SLA
    classes and user-defined scheduling priorities."""

    @property
    def requires_tree_metadata(self) -> bool:
        """Whether this policy requires Radix/Trie hierarchy metadata from the block table."""
        return self is EvictionPolicy.PREFIX_AWARE

    @property
    def requires_frequency_tracking(self) -> bool:
        """Whether the cache allocator must maintain hit-frequency counters."""
        return self in (EvictionPolicy.LFU, EvictionPolicy.SLRU)

class PrefixHashAlgorithm(enum.StrEnum):
    """Hashing algorithms for Radix Tree and prefix-caching block indexers.

    Prefix caching computes deterministic fingerprints over prompt token sequences,
    multimodal tokens, and adapter configurations to identify reusable KV blocks.
    """

    SHA256 = "sha256"
    """Standard SHA-256 hash. Strong collision resistance for multi-tenant serving."""

    CBOR_SHA256 = "cbor_sha256"
    """Deterministic CBOR serialization (RFC 8949) hashed with SHA-256.
    Ensures stable keys for composite metadata (token IDs + LoRA adapter + soft prompts)."""

    XXHASH = "xxhash"
    """High-throughput non-cryptographic hash (xxHash64/xxH3).
    Minimizes CPU latency during token tree traversals on the critical scheduling path."""

    @property
    def is_cryptographic(self) -> bool:
        """Whether this hashing algorithm provides cryptographic collision resistance."""
        return self in (PrefixHashAlgorithm.SHA256, PrefixHashAlgorithm.CBOR_SHA256)

class PrefixReuseMode(enum.StrEnum):
    """Granularity and scoping strategy for KV-cache prefix reuse.

    Controls how the scheduler and block table lookup match cached KV blocks
    against model attention head architectures.
    """

    FULL_ATTENTION_ONLY = "full_attention_only"
    """Requires exact KV cache matching across all attention heads and layers.
    Standard baseline for dense multi-head attention (MHA) models."""

    PER_GROUP = "per_group"
    """Enables prefix matching per KV-head group.
    Optimized for Grouped-Query Attention (GQA) and Multi-Query Attention (MQA),
    allowing partial reuse across head groups sharing identical context."""

class CacheLayerKind(enum.StrEnum):
    """Architectural classification of stateful layer caches in LLM backbones.

    Directs the memory allocator and kernel dispatcher on how to allocate, shape,
    and update intermediate layer states during prefill and autoregressive decoding.
    """

    FULL_ATTENTION = "full_attention"
    """Standard causal Multi-Head, Grouped-Query, or Multi-Query Attention (MHA/GQA/MQA).
    KV cache grows monotonically O(N) with sequence length."""

    SLIDING_WINDOW = "sliding_window"
    """Bounded local window attention (e.g., Mistral, Gemma 2).
    KV cache memory is bounded at O(W) using rolling ring buffers."""

    MLA = "mla"
    """Multi-Head Latent Attention (DeepSeek-V2 / DeepSeek-V3).
    Caches compressed latent key-value projections and decoupled RoPE keys ($c_t^{KV} + k_t^R$)."""

    MAMBA = "mamba"
    """Selective State Space Models (Mamba, Mamba-2 / SSD, Jamba hybrid layers).
    Maintains fixed-size O(1) recurrent hidden states and 1D convolution states."""

    @property
    def is_fixed_size(self) -> bool:
        """Whether the cache footprint per layer remains constant regardless of context length."""
        return self is CacheLayerKind.MAMBA

    @property
    def is_bounded(self) -> bool:
        """Whether the cache footprint per layer is bounded by a fixed maximum token limit."""
        return self in (CacheLayerKind.SLIDING_WINDOW, CacheLayerKind.MAMBA)

    @property
    def is_compressed_latent(self) -> bool:
        """Whether the layer stores low-rank compressed latent
        representations rather than explicit KV heads."""
        return self is CacheLayerKind.MLA

    @property
    def supports_prefix_caching(self) -> bool:
        """Whether the layer natively supports Radix-tree / hash-based prompt prefix sharing."""
        return self in (CacheLayerKind.FULL_ATTENTION, CacheLayerKind.MLA)
