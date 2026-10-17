"""Packed repetition history and independent constraint previews."""

from collections.abc import Sequence

from ayaka.generation.contracts import (
    I32_MAX,
    Constraint,
    Sampling,
    SelectSpec,
    Shaping,
    bounded,
    checked_add,
)

HISTORY_WIDTH = 64
_DEFAULT_SHAPING = Shaping()


def selection_history(
    accepted: Sequence[int], proposal_prefix: Sequence[int] = ()
) -> tuple[int, ...]:
    keep_proposal = min(len(proposal_prefix), HISTORY_WIDTH)
    keep_accepted = min(len(accepted), HISTORY_WIDTH - keep_proposal)
    tokens = tuple(accepted[len(accepted) - keep_accepted :]) + tuple(
        proposal_prefix[len(proposal_prefix) - keep_proposal :]
    )
    for token in tokens:
        bounded(token, "selection history token", I32_MAX)
    return (-1,) * (HISTORY_WIDTH - len(tokens)) + tokens


def verification_selects(
    generated_position: int,
    accepted: Sequence[int],
    proposal: Sequence[int],
    constraint: Constraint | None = None,
    sampling: Sampling = Sampling.GREEDY,
    shaping: Shaping = _DEFAULT_SHAPING,
    seed: int = 0,
) -> tuple[SelectSpec, ...]:
    preview = constraint.fork() if constraint is not None else None
    poisoned = False
    selects = []
    for row in range(len(proposal) + 1):
        selects.append(
            SelectSpec(
                sampling=sampling,
                seed=seed,
                position=checked_add(generated_position, row),
                mask=None if preview is None else tuple(preview.mask()),
                shaping=shaping,
                history=selection_history(accepted, proposal[:row])
                if shaping.uses_history()
                else None,
            )
        )
        if row < len(proposal) and not poisoned and preview is not None:
            try:
                successor = preview.stage((proposal[row],))
            except ValueError:
                poisoned = True
            else:
                if successor.position() != preview.position() + 1:
                    raise ValueError("constraint preview advanced to an invalid position")
                preview = successor
    return tuple(selects)
