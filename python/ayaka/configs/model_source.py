from __future__ import annotations

import enum
from dataclasses import dataclass


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
    SAFETENSORS = "safetensors"
    PT = "pt"
    DUMMY = "dummy"  # random weights of the right shape; for shape/plumbing tests

class TrustPolicy(enum.StrEnum):
    """Whether executing checkpoint-supplied Python is permitted.

    ``STRICT`` is the default and the only value A1 core honours.  The other
    member exists so the refusal can name what was asked for; a resolver that
    sees ``ALLOW_REMOTE_CODE`` raises ``CheckpointSecurityError`` rather than
    silently downgrading, because a config that says "trust this" and a runtime
    that quietly does not is worse than either.
    """

    STRICT = "strict"
    ALLOW_REMOTE_CODE = "allow_remote_code"

@dataclass(frozen=True, slots=True)
class ModelSourceConfig:
    """Where to get the model from, and under what rules."""

    model: str # local path or hub id
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
        return self.tokenizer or self.model

    @property
    def is_dummy(self) -> bool:
        return self.format is RequestedFormat.DUMMY

    @property
    def allows_remote_code(self) -> bool:
        return self.trust is TrustPolicy.ALLOW_REMOTE_CODE

    def looks_like_hub_id(self) -> bool:
        """A hub id is ``org/name`` with no path separators beyond the one.

        Heuristic on purpose, and only a hint: the resolver checks the local
        filesystem first regardless, because a local directory named ``org/name``
        must win over a hub lookup or a typo silently downloads a stranger's
        model.
        """
        s = self.model
        return s.count("/") == 1 and not s.startswith((".", "/", "~")) and "\\" not in s
