"""Device-side verification of linear drafts: greedy and sampled.

The base row of a speculative slice is sampled by the unchanged sampling
pipeline; its token ``t0`` is the target's choice after zero accepted drafts.
Each draft row ``i`` then decides the token after ``i`` accepted drafts. A draft
row applies the same transformations in the same order as the regular sampling
path: the vocabulary validity mask, the pre-sample bans for ``generated + i``
tokens, and the temperature ``logits / temperature.clamp_min(1e-6)``. Features
whose transformations depend on tentative history (penalties, grammars) never
reach this module; the coordinator keeps those requests at ``k = 0``.

Greedy rows pick ``argmax`` and match target-only greedy decoding exactly.

Sampled rows draw ``t_i`` from the request's filtered target distribution
(top-k, top-p, min-p) and the longest prefix with ``d_{i+1} == t_i`` is
accepted. For a deterministic drafter (``q`` one-hot on ``d``) this *is*
rejection sampling: ``d`` survives with probability ``min(1, p(d) / q(d)) =
p(d)``, and on rejection ``t`` is distributed as ``p`` conditioned on ``t != d``,
which is the normalized residual ``max(p - q, 0)``. The guarantee is
distributional; it needs every draw to be independent of every other draw and
of the draft. The base row draws from ``(seed, offset)`` exactly as without
speculation; draft row ``i`` draws from ``(draft_row_seed(seed, i), offset)``,
a Philox key derived from the request seed that no base-row draw uses. The
request offset advances once per step, so no ``(key, offset)`` pair repeats.

Nothing here reads a device value on the host: acceptance counts and final
tokens stay on the device until the completion boundary materializes them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from ayaka.sampling.bans import BanApplier
from ayaka.sampling.ops.sampling import topk_topp_sample
from ayaka.sampling.rng import philox4x32_10
from ayaka.speculative.acceptance.greedy import greedy_accept
from ayaka.speculative.metadata import VerifyLayout

__all__ = [
    "RowSampling",
    "draft_row_seeds",
    "greedy_verify",
    "sampled_verify",
    "stage_host_tensor",
]

# Philox counter word naming the draft-row key derivation, so derived keys
# cannot coincide with any other keyed stream the sampler draws from.
_DRAFT_STREAM_TAG = 0x53504543  # "SPEC"


def stage_host_tensor(values: Sequence[int] | Sequence[Sequence[int]], device: torch.device):
    """Host ints -> device int64 without a blocking copy on CUDA."""
    host = torch.tensor(values, dtype=torch.long, pin_memory=device.type == "cuda")
    return host.to(device, non_blocking=True)


def draft_row_seeds(seeds: torch.Tensor, draft_index: torch.Tensor) -> torch.Tensor:
    """Independent non-negative Philox keys for draft rows.

    The key is one Philox4x32-10 block keyed by the request seed at counter
    ``(draft_index, TAG, 0, 0)``: a pure, bijective-in-the-counter function, so
    draft rows of one request get distinct keys and no key equals the request
    seed except with ~2^-63 probability.

    Args:
        seeds: ``[N]`` int64 request seeds.
        draft_index: ``[N]`` int64 draft row index within its slice (``>= 1``).

    Returns:
        ``[N]`` int64 keys in ``[0, 2^63)``.
    """
    if seeds.shape != draft_index.shape:
        raise ValueError("one draft index is required per seed")
    seeds = seeds.to(torch.long)
    k0 = seeds & 0xFFFFFFFF
    k1 = (seeds >> 32) & 0xFFFFFFFF
    c0 = draft_index.to(torch.long) & 0xFFFFFFFF
    c1 = torch.full_like(c0, _DRAFT_STREAM_TAG)
    zeros = torch.zeros_like(c0)
    r0, r1, _, _ = philox4x32_10(c0, c1, zeros, zeros, k0, k1)
    return ((r0 & 0x7FFFFFFF) << 32) | r1


@dataclass(frozen=True, slots=True)
class RowSampling:
    """Sampling columns of the ``R`` speculative slices, as the sampler read them.

    Device tensors are ``[R]`` in slice order. ``any_min_p`` is the host-known
    flag for the same rows, so dispatch never synchronizes.
    """

    temperature: torch.Tensor
    top_k: torch.Tensor
    top_p: torch.Tensor
    min_p: torch.Tensor
    seed: torch.Tensor
    offset: torch.Tensor
    any_min_p: bool

    def __post_init__(self) -> None:
        rows = self.temperature.shape
        for name in ("top_k", "top_p", "min_p", "seed", "offset"):
            if getattr(self, name).shape != rows:
                raise ValueError(f"{name} needs one entry per speculative slice")
        if type(self.any_min_p) is not bool:
            raise TypeError("any_min_p must be bool")


def _prepare_rows(
    extension_logits: torch.Tensor,
    base_tokens: torch.Tensor,
    layout: VerifyLayout,
    temperatures: torch.Tensor,
    valid_mask: torch.Tensor | None,
    bans: Sequence[frozenset[int]],
    ban_applier: BanApplier,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate, mask and ban draft rows; return ``(repeats, scaled logits)``."""
    rows = layout.num_speculative
    if extension_logits.dim() != 2 or extension_logits.size(0) != sum(layout.draft_counts):
        raise ValueError("extension logits must hold one row per draft token")
    if base_tokens.shape != (rows,) or temperatures.shape != (rows,):
        raise ValueError("base tokens and temperatures need one entry per speculative slice")
    if len(bans) != extension_logits.size(0):
        raise ValueError("one ban set is required per draft row")
    if valid_mask is not None:
        extension_logits.masked_fill_(~valid_mask, float("-inf"))
    ban_applier.apply(extension_logits, bans)
    repeats = stage_host_tensor(layout.draft_counts, extension_logits.device)
    row_temperatures = _per_row(temperatures, repeats, layout)
    scaled = extension_logits / row_temperatures.clamp_min(1e-6).unsqueeze(1)
    return repeats, scaled


def _per_row(column: torch.Tensor, repeats: torch.Tensor, layout: VerifyLayout) -> torch.Tensor:
    return column.repeat_interleave(repeats, output_size=sum(layout.draft_counts))


def _accept(
    choices: torch.Tensor, base_tokens: torch.Tensor, layout: VerifyLayout
) -> tuple[torch.Tensor, torch.Tensor]:
    """Longest draft prefix matching the target choices; ``(final, accepted)``."""
    device = choices.device
    counts = layout.draft_counts
    targets = torch.full(
        (layout.num_speculative, layout.max_k + 1), -1, dtype=torch.long, device=device
    )
    targets[:, 0] = base_tokens.to(torch.long)
    owner = [slice_ for slice_, count in enumerate(counts) for _ in range(count)]
    column = [offset + 1 for count in counts for offset in range(count)]
    targets[stage_host_tensor(owner, device), stage_host_tensor(column, device)] = choices.to(
        torch.long
    )
    drafts = stage_host_tensor(layout.padded_drafts(), device)
    accepted, final = greedy_accept(drafts, targets)
    return final, accepted


def greedy_verify(
    extension_logits: torch.Tensor,
    base_tokens: torch.Tensor,
    layout: VerifyLayout,
    *,
    temperatures: torch.Tensor,
    valid_mask: torch.Tensor | None,
    bans: Sequence[frozenset[int]],
    ban_applier: BanApplier,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Greedy target choices for draft rows and the resulting acceptance.

    Args:
        extension_logits: ``[E, V]`` logits of every draft row in layout order;
            transformed in place.
        base_tokens: ``[R]`` tokens the regular sampler chose for the base rows
            of the ``R`` speculative slices.
        layout: Host row layout of the step.
        temperatures: ``[R]`` temperature column of the speculative slices, as
            the sampler read it.
        valid_mask: Vocabulary validity mask, or ``None``.
        bans: One ban set per draft row.
        ban_applier: The runner's ban applier (same fill as sampling rows).

    Returns:
        ``(final [R], accepted [R])`` on the logits' device.
    """
    _, scaled = _prepare_rows(
        extension_logits, base_tokens, layout, temperatures, valid_mask, bans, ban_applier
    )
    return _accept(scaled.argmax(dim=-1), base_tokens, layout)


def sampled_verify(
    extension_logits: torch.Tensor,
    base_tokens: torch.Tensor,
    layout: VerifyLayout,
    *,
    sampling: RowSampling,
    valid_mask: torch.Tensor | None,
    bans: Sequence[frozenset[int]],
    ban_applier: BanApplier,
    force_reference: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sampled target choices for draft rows and the resulting acceptance.

    Each draft row draws from the request's filtered target distribution with
    its own derived Philox key (see the module docstring). Rows whose request
    is greedy (``temperature == 0`` or ``top_k == 1``) take ``argmax`` exactly
    like the sampler's per-row greedy override, so mixed batches stay exact for
    their greedy members.

    Args:
        extension_logits: ``[E, V]`` logits of every draft row in layout order;
            transformed in place.
        base_tokens: ``[R]`` tokens the regular sampler chose for the base rows.
        layout: Host row layout of the step.
        sampling: Sampling columns of the speculative slices; ``offset`` must be
            the offset the base rows drew with (before this step's advance).
        valid_mask: Vocabulary validity mask, or ``None``.
        bans: One ban set per draft row.
        ban_applier: The runner's ban applier.
        force_reference: Use the torch sampling oracle, like the base sampler.

    Returns:
        ``(final [R], accepted [R])`` on the logits' device.
    """
    repeats, scaled = _prepare_rows(
        extension_logits,
        base_tokens,
        layout,
        sampling.temperature,
        valid_mask,
        bans,
        ban_applier,
    )
    device = scaled.device
    index = stage_host_tensor(
        [offset + 1 for count in layout.draft_counts for offset in range(count)], device
    )
    top_k = _per_row(sampling.top_k, repeats, layout)
    seeds = draft_row_seeds(_per_row(sampling.seed, repeats, layout), index)
    drawn = topk_topp_sample(
        scaled,
        top_k,
        _per_row(sampling.top_p, repeats, layout),
        _per_row(sampling.min_p, repeats, layout),
        seeds,
        _per_row(sampling.offset, repeats, layout),
        force_reference=force_reference,
        any_min_p=sampling.any_min_p,
    )
    greedy_rows = (_per_row(sampling.temperature, repeats, layout) == 0.0) | (top_k == 1)
    choices = torch.where(greedy_rows, scaled.argmax(dim=-1).to(drawn.dtype), drawn)
    return _accept(choices, base_tokens, layout)
