"""Configuration types for locating, authenticating, and resolving model sources.

This module defines:

* :class:`RequestedFormat` — operator-requested weight checkpoint formats
  (:term:`safetensors`, PyTorch, auto-discovery, dummy weights).
* :class:`TrustPolicy` — security policy for executing checkpoint-supplied
  Python code.
* :class:`ModelSourceConfig` — immutable configuration identifying *where*
  to find model artifacts (local path or HuggingFace Hub) and under what
  integrity and network rules.

Resolution flow::

    ModelSourceConfig(model="meta-llama/Llama-3-8B", ...)
        │
        ├─ .looks_like_hub_id() ──→ True (heuristic hub check)
        ├─ .tokenizer_path      ──→ "meta-llama/Llama-3-8B"
        ├─ .allows_remote_code  ──→ False (STRICT trust)
        │
        ▼  resolve_source(source)
    ResolvedSource(root=..., format=..., weight_files=...)
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from ayaka.configs.base import ConfigMixin


class RequestedFormat(enum.StrEnum):
    """What the operator asked for, not what discovery found.

    Deliberately a different enum from
    ``ayaka.model_loader.weight_plan.CheckpointFormat``: ``AUTO`` is a request that
    no checkpoint can be, and ``SAFETENSORS`` here means "refuse anything else",
    while ``CheckpointFormat.SAFETENSORS`` means "single-file, as opposed to
    sharded" — a distinction discovery makes and the operator does not.
    Collapsing them would make ``format=safetensors`` silently reject every
    sharded checkpoint, which is most of them above 5 B parameters.
    """

    AUTO = "auto"
    """Auto-detect format: inspect available checkpoint files and pick the best match."""

    SAFETENSORS = "safetensors"
    """Require SafeTensors format (single-file or sharded); reject legacy formats."""

    PT = "pt"
    """Legacy PyTorch pickle-based weights (*.bin / *.pt)."""

    DUMMY = "dummy"
    """Random weights of the right shape; used for shape, memory, and plumbing tests."""


class TrustPolicy(enum.StrEnum):
    """Whether executing checkpoint-supplied Python code is permitted.

    ``STRICT`` is the default and the only value A1 core honours.  The other
    member exists so the refusal can name what was asked for; a resolver that
    sees ``ALLOW_REMOTE_CODE`` raises ``CheckpointSecurityError`` rather than
    silently downgrading, because a config that says "trust this" and a runtime
    that quietly does not is worse than either.
    """

    STRICT = "strict"
    """Strict security: refuse execution of any arbitrary code from the model checkpoint."""

    ALLOW_REMOTE_CODE = "allow_remote_code"
    """Allow execution of Python code bundled within the model checkpoint repository."""


@dataclass(frozen=True, slots=True)
class ModelSourceConfig(ConfigMixin):
    """Immutable configuration specifying model checkpoint origins and security policies.

    Identifies whether the model resides on the local filesystem or the HuggingFace
    Hub, which revision to pin, whether offline resolution is enforced, and
    whether remote custom code execution is permitted.

    Attributes:
        model: Local directory path (e.g. ``"/models/llama-3-8b"``) or HuggingFace
            Hub repo identifier (e.g. ``"meta-llama/Llama-3-8B"``).
        revision: Specific Git commit SHA, branch name, or tag on the HuggingFace
            Hub. An empty string (``""``) defaults to whatever the HEAD ref points to.
        tokenizer: Optional hub repo ID or directory path containing tokenizer
            files. When empty (default), falls back to :attr:`model`.
        format: Desired checkpoint serialization format. See :class:`RequestedFormat`.
            Default is :attr:`RequestedFormat.AUTO`.
        trust: Remote code execution security policy. See :class:`TrustPolicy`.
            Default is :attr:`TrustPolicy.STRICT`.
        offline: When ``True``, never touches the network and resolves exclusively
            from the local hub cache directory.
        reproducible: When ``True``, requires an explicit immutable revision
            (commit SHA) and rejects floating branch names or HEAD.
        cache_dir: Override path for HuggingFace Hub downloads. An empty string
            defaults to ``HF_HOME`` or the standard library cache.
    """

    model: str  # local path or hub id
    revision: str = ""  # branch, tag or commit sha; "" = whatever HEAD is
    tokenizer: str = ""  # defaults to `model`
    format: RequestedFormat = RequestedFormat.AUTO
    trust: TrustPolicy = TrustPolicy.STRICT

    # Offline: never touch the network, resolve from the local hub cache only.
    offline: bool = False

    # Reproducible: refuse a revision that can move.  A branch name resolves to
    # a different commit next week, and "the model changed under us" is a class
    # of bug that costs days precisely because nothing in the config recorded
    # that it could happen.
    reproducible: bool = False

    cache_dir: str = ""  # "" = HF_HOME / default hub cache

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model path or hub id must not be empty")
        if self.reproducible and not self.revision:
            raise ValueError(
                "reproducible=True requires an explicit revision — a floating ref "
                "cannot be reproduced, and pretending otherwise defeats the flag"
            )

    @property
    def tokenizer_path(self) -> str:
        """Effective path or repo ID for the tokenizer, falling back to ``model``."""
        return self.tokenizer or self.model

    @property
    def is_dummy(self) -> bool:
        """Whether this configuration requests synthetic dummy weights for testing."""
        return self.format is RequestedFormat.DUMMY

    @property
    def allows_remote_code(self) -> bool:
        """Whether executing checkpoint-supplied custom Python code is allowed."""
        return self.trust is TrustPolicy.ALLOW_REMOTE_CODE

    def looks_like_hub_id(self) -> bool:
        """A hub id is ``org/name`` with no path separators beyond the one.

        Heuristic on purpose, and only a hint: the resolver checks the local
        filesystem first regardless, because a local directory named ``org/name``
        must win over a hub lookup or a typo silently downloads a stranger's
        model.

        Returns:
            ``True`` if ``model`` conforms to the ``org/name`` hub naming pattern.
        """
        s = self.model
        return s.count("/") == 1 and not s.startswith((".", "/", "~")) and "\\" not in s
