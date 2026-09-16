"""Runtime memory: KV paging, ledger accounting, and scratch byte pools.

Modules, roughly bottom-up:

- :mod:`~ayaka.memory.region`, :mod:`~ayaka.memory.state`,
  :mod:`~ayaka.memory.metadata` — contract vocabulary: byte ranges, page
  lifecycle enums, allocator-private page metadata.
- :mod:`~ayaka.memory.ledger` — the single accounting authority per tier.
- :mod:`~ayaka.memory.allocator` — generation-safe KV page identities
  (metadata only; the K/V bytes live in the storage layer).
- :mod:`~ayaka.memory.source` and :mod:`~ayaka.memory.caching` — raw byte
  sources and the size-class caching allocator for scratch.
- :mod:`~ayaka.memory.arena`, :mod:`~ayaka.memory.buffer`,
  :mod:`~ayaka.memory.workspace` — the carving and leasing layers built on a
  byte pool.
- :mod:`~ayaka.memory.sequence`, :mod:`~ayaka.memory.transaction`,
  :mod:`~ayaka.memory.views`, :mod:`~ayaka.memory.manager`,
  :mod:`~ayaka.memory.pressure` — the paged KV lifecycle the scheduler drives.
- :mod:`~ayaka.memory.tiering` — the opt-in host KV tier.
- :mod:`~ayaka.memory.host` — host memory policy and exact-tier host sources.

Nothing in this package imports torch at module import time; optional
dependencies are resolved lazily at the call that needs them.
"""
