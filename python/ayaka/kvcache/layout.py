"""Layout feature capability statements for advanced KV cache layouts.

This module is intentionally independent of grouping construction so callers
can ask whether a resolved layout supports cross-layout features without
pulling in grouped-manager lifecycle state.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from ayaka.kvcache.retention.policy import FullRetention, RetentionPolicy
from ayaka.kvcache.storage.geometry import BaseKVStorageSpec
from ayaka.kvcache.storage.layout import KVStorageKind


class _ResolvedKVCacheGroup(Protocol):
    """Structural subset needed to evaluate layout feature support."""

    retention: RetentionPolicy
    storage_spec: BaseKVStorageSpec

class LayoutFeature(StrEnum):
    """Cross-layout features the A9 grouping may or may not support."""

    PREFIX_CACHE = "prefix_cache"
    """A6 full-page prefix reuse."""
    RECURRENT_STATE = "recurrent_state"
    """Recurrent model state lifecycle."""


class LayoutFeatureIssueCode(StrEnum):
    """Structured reasons a layout feature has no safe implementation."""

    MULTI_GROUP_PREFIX_UNSUPPORTED = "multi_group_prefix_unsupported"
    """Prefix reuse has no canonical page chain across multiple groups."""
    RETENTION_PREFIX_UNSUPPORTED = "retention_prefix_unsupported"
    """Prefix sharing requires full retention in the current implementation."""
    LAYOUT_PREFIX_UNSUPPORTED = "layout_prefix_unsupported"
    """Prefix sharing is implemented only for homogeneous MHA/GQA storage."""
    PREFIX_CACHE_NOT_IMPLEMENTED = "prefix_cache_not_implemented"
    """Layout is compatible but no grouped manager implements the cache."""
    NO_VALIDATED_RECURRENT_BACKEND = "no_validated_recurrent_backend"
    """No recurrent model/backend contract exists for lifecycle validation."""

@dataclass(frozen=True, slots=True)
class LayoutFeatureIssue:
    """One structured reason a feature cannot be enabled."""

    code: LayoutFeatureIssueCode
    message: str


@dataclass(frozen=True, slots=True)
class LayoutFeatureCapability:
    """Structured capability statement for one layout feature.

    ``supported`` is true exactly when there are no issues; callers either
    branch on ``supported`` or call :meth:`require_supported` to fail loudly
    before any schedule or launch.
    """

    feature: LayoutFeature
    issues: tuple[LayoutFeatureIssue, ...]

    @property
    def supported(self) -> bool:
        return not self.issues

    def require_supported(self) -> None:
        """Raise a ``LayoutFeatureCompatibilityError`` when unsupported."""
        if self.issues:
            raise LayoutFeatureCompatibilityError(self)


class LayoutFeatureCompatibilityError(ValueError):
    """A requested cross-layout feature has no safe A9 implementation.

    Raised before any schedule or launch so unsupported layouts fail closed
    with an explicit reason instead of silently falling back.
    """

    def __init__(self, capability: LayoutFeatureCapability) -> None:
        self.capability = capability
        detail = "; ".join(issue.message for issue in capability.issues)
        super().__init__(f"{capability.feature.value} is incompatible: {detail}")


def prefix_cache_capability(
    groups: tuple[_ResolvedKVCacheGroup, ...],
) -> LayoutFeatureCapability:
    """Only the stable homogeneous full-retention MHA path reuses A6 pages.

    Evaluates the *layout-level* statement: a single MHA full-retention group
    is compatible with prefix sharing; multi-group, retention-restricted, or
    MLA layouts produce explicit issue codes instead of silent fallback.

    Args:
        groups: The resolved cache groups to evaluate.

    Returns:
        A structured capability; ``supported`` only for a single homogeneous
        full-retention MHA group.
    """

    normalized = tuple(groups)
    issues: list[LayoutFeatureIssue] = []
    if len(normalized) != 1:
        issues.append(
            LayoutFeatureIssue(
                LayoutFeatureIssueCode.MULTI_GROUP_PREFIX_UNSUPPORTED,
                "A9 multi-group prefix ownership has no canonical page chain",
            )
        )
    if any(not isinstance(group.retention, FullRetention) for group in normalized):
        issues.append(
            LayoutFeatureIssue(
                LayoutFeatureIssueCode.RETENTION_PREFIX_UNSUPPORTED,
                "prefix sharing requires full retention in the current implementation",
            )
        )
    if any(group.storage_spec.kind is not KVStorageKind.MHA for group in normalized):
        issues.append(
            LayoutFeatureIssue(
                LayoutFeatureIssueCode.LAYOUT_PREFIX_UNSUPPORTED,
                "prefix sharing is implemented only for homogeneous MHA/GQA storage",
            )
        )
    return LayoutFeatureCapability(
        feature=LayoutFeature.PREFIX_CACHE,
        issues=tuple(issues),
    )


def recurrent_state_storage_capability() -> LayoutFeatureCapability:
    """Declare recurrent-state storage explicitly unsupported.

    Ayaka has no validated recurrent model/backend contract, so the capability
    always carries ``NO_VALIDATED_RECURRENT_BACKEND`` instead of silently
    modeling recurrent state as K/V tensors.
    """
    return LayoutFeatureCapability(
        feature=LayoutFeature.RECURRENT_STATE,
        issues=(
            LayoutFeatureIssue(
                LayoutFeatureIssueCode.NO_VALIDATED_RECURRENT_BACKEND,
                "no recurrent model/backend contract is available for lifecycle validation",
            ),
        ),
    )
