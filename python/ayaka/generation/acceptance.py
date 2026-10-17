"""Target-sampled prefix acceptance, shared by the host generation planes."""

from collections.abc import Collection, Sequence


def accept_prefix(
    proposal: Sequence[int], samples: Sequence[int], stops: Collection[int] = ()
) -> int:
    """Count matching draft tokens, stopping before any stop token.

    Stop tokens are emitted only as the target correction/bonus. ``samples``
    must cover every proposal; a round additionally requires one bonus row.
    """
    if len(samples) < len(proposal):
        raise ValueError("verification samples do not cover every proposal token")
    accepted = 0
    for proposed, sampled in zip(proposal, samples, strict=False):
        if proposed != sampled or proposed in stops:
            break
        accepted += 1
    return accepted
