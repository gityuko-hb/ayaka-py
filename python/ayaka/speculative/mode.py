"""Speculative decoding modes and the capabilities this build implements.

A mode is data. Every capability that the scheduler, runner, sampler or graph
planner consults is a field of one frozen :class:`ModeCapabilities` record, so
no owner grows its own ``if mode is ...`` chain. The record describes what this
build *implements and certifies*, not what the published algorithm could do:
a capability flips only together with the code and tests that back it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

__all__ = [
    "Certification",
    "ModeCapabilities",
    "SpeculativeMode",
    "mode_capabilities",
    "resolve_mode",
]


class Certification(StrEnum):
    """How far a mode is proven in this build.

    ``PRODUCTION`` has an end-to-end serving path with golden parity tests.
    ``EXPERIMENTAL`` is implemented and unit tested but needs an explicit
    operator opt-in. ``UNAVAILABLE`` is a contract only; construction refuses.
    """

    PRODUCTION = "production"
    EXPERIMENTAL = "experimental"
    UNAVAILABLE = "unavailable"


class SpeculativeMode(StrEnum):
    """Canonical speculative method identity.

    Neural drafters are resources of the target worker; there is no separate
    draft engine or scheduler. ``AUTO`` must be resolved with
    :func:`resolve_mode` before any capability is read.
    """

    NONE = "none"
    AUTO = "auto"
    NGRAM = "ngram"
    DRAFT_MODEL = "draft_model"
    MTP = "mtp"
    EAGLE3 = "eagle3"

    @property
    def capabilities(self) -> ModeCapabilities:
        return mode_capabilities(self)

    @property
    def uses_neural_drafter(self) -> bool:
        return self.capabilities.uses_neural_drafter

    @property
    def uses_target_hidden_states(self) -> bool:
        return self.capabilities.uses_target_hidden_states

    @property
    def needs_draft_kv(self) -> bool:
        return self.capabilities.needs_draft_kv

    @property
    def needs_draft_kv_catchup(self) -> bool:
        return self.capabilities.needs_draft_kv_catchup

    @property
    def supports_tree(self) -> bool:
        return self.capabilities.supports_tree

    @property
    def supports_dynamic_k(self) -> bool:
        return self.capabilities.supports_dynamic_k

    @property
    def supports_cuda_graph(self) -> bool:
        return self.capabilities.supports_cuda_graph

    @property
    def supports_advanced_sampling(self) -> bool:
        return self.capabilities.supports_advanced_sampling

    @property
    def supports_guided_decoding(self) -> bool:
        return self.capabilities.supports_guided_decoding


@dataclass(frozen=True, slots=True, kw_only=True)
class ModeCapabilities:
    """Implemented behavior of one mode in this build.

    Attributes:
        uses_neural_drafter: Proposals come from a model forward.
        uses_target_hidden_states: The drafter consumes target hidden rows.
        needs_draft_kv: The drafter owns KV separate from the target's.
        needs_draft_kv_catchup: Draft KV must absorb accepted tokens after
            verification before the next proposal.
        supports_tree: Verification of branching candidates is implemented.
        supports_dynamic_k: The per-step draft length may change without
            rebuilding runtime state.
        supports_cuda_graph: Verification may replay a captured graph.
        supports_advanced_sampling: Non-greedy requests may speculate.
        supports_guided_decoding: Grammar-constrained requests may speculate.
        certification: Proof level; see :class:`Certification`.
    """

    uses_neural_drafter: bool
    uses_target_hidden_states: bool
    needs_draft_kv: bool
    needs_draft_kv_catchup: bool
    supports_tree: bool
    supports_dynamic_k: bool
    supports_cuda_graph: bool
    supports_advanced_sampling: bool
    supports_guided_decoding: bool
    certification: Certification

    def __post_init__(self) -> None:
        for name in (
            "uses_neural_drafter",
            "uses_target_hidden_states",
            "needs_draft_kv",
            "needs_draft_kv_catchup",
            "supports_tree",
            "supports_dynamic_k",
            "supports_cuda_graph",
            "supports_advanced_sampling",
            "supports_guided_decoding",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool")
        if not isinstance(self.certification, Certification):
            raise TypeError("certification must be Certification")
        if self.needs_draft_kv_catchup and not self.needs_draft_kv:
            raise ValueError("draft KV catch-up requires draft KV")
        if self.uses_target_hidden_states and not self.uses_neural_drafter:
            raise ValueError("target hidden-state consumers are neural drafters")


_OFF = dict(
    uses_neural_drafter=False,
    uses_target_hidden_states=False,
    needs_draft_kv=False,
    needs_draft_kv_catchup=False,
    supports_tree=False,
    supports_dynamic_k=False,
    supports_cuda_graph=False,
    supports_advanced_sampling=False,
    supports_guided_decoding=False,
)

_CAPABILITIES = MappingProxyType(
    {
        SpeculativeMode.NONE: ModeCapabilities(**_OFF, certification=Certification.PRODUCTION),
        # Host n-gram drafts verified by one ragged target forward: greedy
        # golden parity (tiny native families, SmolLM2, Qwen2.5; CPU FP32 and
        # CUDA fp16/bf16 Triton) on one device. Sampled requests (temperature,
        # top-k/top-p/min-p) verify by per-row target sampling, which is exact
        # rejection sampling for deterministic drafts; certified by chi-square
        # tests against the exact joint distribution and end to end against
        # target-only sampling. Trees, captured verification and tensor
        # parallelism are separate gates.
        SpeculativeMode.NGRAM: ModeCapabilities(
            **{**_OFF, "supports_dynamic_k": True, "supports_advanced_sampling": True},
            certification=Certification.PRODUCTION,
        ),
        SpeculativeMode.DRAFT_MODEL: ModeCapabilities(
            **{
                **_OFF,
                "uses_neural_drafter": True,
                "needs_draft_kv": True,
                "needs_draft_kv_catchup": True,
                "supports_dynamic_k": True,
            },
            certification=Certification.UNAVAILABLE,
        ),
        SpeculativeMode.MTP: ModeCapabilities(
            **{
                **_OFF,
                "uses_neural_drafter": True,
                "uses_target_hidden_states": True,
                "needs_draft_kv": True,
                "needs_draft_kv_catchup": True,
            },
            certification=Certification.UNAVAILABLE,
        ),
        SpeculativeMode.EAGLE3: ModeCapabilities(
            **{
                **_OFF,
                "uses_neural_drafter": True,
                "uses_target_hidden_states": True,
                "needs_draft_kv": True,
                "needs_draft_kv_catchup": True,
            },
            certification=Certification.UNAVAILABLE,
        ),
    }
)


def mode_capabilities(mode: SpeculativeMode) -> ModeCapabilities:
    """Return the implemented capability record of a concrete mode.

    Raises:
        TypeError: If ``mode`` is not a :class:`SpeculativeMode`.
        ValueError: If ``mode`` is ``AUTO`` (resolve it first).
    """
    if not isinstance(mode, SpeculativeMode):
        raise TypeError("mode must be SpeculativeMode")
    if mode is SpeculativeMode.AUTO:
        raise ValueError("AUTO has no capabilities; resolve it with resolve_mode first")
    return _CAPABILITIES[mode]


def resolve_mode(
    requested: SpeculativeMode,
    *,
    has_draft_model: bool = False,
    mtp_layers: int = 0,
    has_eagle_heads: bool = False,
) -> SpeculativeMode:
    """Resolve ``AUTO`` to the strongest *production* mode the inputs allow.

    Explicit modes are returned unchanged; their certification is checked by
    the configuration owner. ``AUTO`` never selects an experimental or
    unavailable mode, so an operator must opt into those by name.
    """
    if not isinstance(requested, SpeculativeMode):
        raise TypeError("requested must be SpeculativeMode")
    if requested is not SpeculativeMode.AUTO:
        return requested
    if type(mtp_layers) is not int or mtp_layers < 0:
        raise ValueError("mtp_layers must be a non-negative integer")
    candidates = []
    if has_eagle_heads:
        candidates.append(SpeculativeMode.EAGLE3)
    if mtp_layers:
        candidates.append(SpeculativeMode.MTP)
    if has_draft_model:
        candidates.append(SpeculativeMode.DRAFT_MODEL)
    candidates.append(SpeculativeMode.NGRAM)
    for mode in candidates:
        if _CAPABILITIES[mode].certification is Certification.PRODUCTION:
            return mode
    return SpeculativeMode.NONE
