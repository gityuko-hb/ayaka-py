from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from ayaka.exceptions import WeightMismatchError
from ayaka.types import DType
from ayaka.weights.plan import CheckpointManifest, ManifestEntry
from ayaka.weights.spec import WeightSpec


@dataclass(frozen=True, slots=True)
class WeightMismatch:
    """One weight whose checkpoint entry disagrees with the expectation."""

    name: str
    reason: str
    expected: str
    actual: str

    def __str__(self) -> str:
        return (
            f"{self.name}: {self.reason} (expected {self.expected}, checkpoint has {self.actual})"
        )

@dataclass(frozen=True, slots=True)
class WeightDiff:
    """Every disagreement between the model and the checkpoint, at once."""

    missing: tuple[str, ...] = ()
    unexpected: tuple[str, ...] = ()
    incompatible: tuple[WeightMismatch, ...] = ()
    bad_tied: tuple[WeightMismatch, ...] = ()
    matched: tuple[str, ...] = ()
    optional_absent: tuple[str, ...] = field(default=())

    @property
    def ok(self) -> bool:
        return not (self.missing or self.unexpected or self.incompatible or self.bad_tied)

    def summary(self) -> str:
        parts = [
            f"{len(self.matched)} matched",
            f"{len(self.missing)} missing",
            f"{len(self.unexpected)} unexpected",
            f"{len(self.incompatible)} incompatible",
        ]
        if self.bad_tied:
            parts.append(f"{len(self.bad_tied)} bad tied alias")
        if self.optional_absent:
            parts.append(f"{len(self.optional_absent)} optional absent")
        return ", ".join(parts)

    def report(self, *, limit: int = 8) -> str:
        """Human-readable, truncated per category rather than overall.

        Per-category truncation matters: a run with 291 missing and 3
        incompatible would, under a global limit, print 8 missing names and hide
        the three that actually explain the failure.
        """
        lines = [self.summary()]
        for label, names in (("missing", self.missing), ("unexpected", self.unexpected)):
            if names:
                shown = ", ".join(names[:limit])
                more = f" (+{len(names) - limit} more)" if len(names) > limit else ""
                lines.append(f"  {label}: {shown}{more}")
        for label, items in (("incompatible", self.incompatible), ("tied", self.bad_tied)):
            if items:
                for item in items[:limit]:
                    lines.append(f"  {label}: {item}")
                if len(items) > limit:
                    lines.append(f"  {label}: (+{len(items) - limit} more)")
        return "\n".join(lines)

    def raise_if_bad(self, *, context: str = "") -> None:
        if self.ok:
            return
        where = f"{context}: " if context else ""
        raise WeightMismatchError(f"{where}checkpoint does not match the model\n{self.report()}")

def _dtype_compatible(expected: DType, actual: DType) -> bool:
    """Whether a checkpoint dtype can feed an expected one.

    Narrowing fp32 → bf16 is the deliberate downcast ,so
    it is allowed here too.  Everything else must match exactly: reading an fp16
    checkpoint as bf16 without a convert is a five-order-of-magnitude error in
    every value, and it produces no shape complaint at all.
    """

    if expected is actual:
        return True
    return actual is DType.FP32 and expected in (DType.BF16, DType.FP16)


def get_diff_weights(
    expected: Sequence[WeightSpec],
    manifest: CheckpointManifest
) -> WeightDiff:
    """Compare the model's expectation against what the checkpoint provides."""
    provided: dict[str, ManifestEntry] = {e.tensor_key: e for e in manifest.entries}
    expected_by_name = {w.name: w for w in expected}

    missing: list[str] = []
    optional_absent: list[str] = []
    incompatible: list[WeightMismatch] = []
    bad_tied: list[WeightMismatch] = []
    matched: list[str] = []

    for spec in expected:
        if spec.is_tied:
            # A tied weight must NOT be in the checkpoint — it shares storage
            # with its target.  A checkpoint that ships both is either untied
            # (and the config lied) or has a stale duplicate; either way,
            # binding the alias would leave one of the two copies unused and
            # silently wrong.
            target = expected_by_name.get(spec.tied_alias)
            if target is None:
                bad_tied.append(
                    WeightMismatch(
                        spec.name,
                        "tied to a weight the model never declares",
                        spec.tied_alias,
                        "<absent>",
                    )
                )
            elif spec.name in provided:
                bad_tied.append(
                    WeightMismatch(
                        spec.name,
                        "declared tied but the checkpoint ships it separately",
                        f"alias of {spec.tied_alias}",
                        "a real tensor",
                    )
                )
            elif target.full_shape != spec.full_shape:
                bad_tied.append(
                    WeightMismatch(
                        spec.name,
                        "tied to a weight of a different shape",
                        str(spec.full_shape),
                        str(target.full_shape),
                    )
                )
            else:
                matched.append(spec.name)
            continue

        entry = provided.get(spec.name)
        if entry is None:
            (optional_absent if spec.optional else missing).append(spec.name)
            continue
        if entry.shape != spec.full_shape:
            incompatible.append(
                WeightMismatch(spec.name, "shape", str(spec.full_shape), str(entry.shape))
            )
        elif not _dtype_compatible(spec.spec.dtype, entry.dtype):
            incompatible.append(
                WeightMismatch(spec.name, "dtype", spec.spec.dtype.label, entry.dtype.label)
            )
        else:
            matched.append(spec.name)

    # Tied names are legitimately absent from the checkpoint, so they must not
    # count towards `unexpected` in the reverse direction either.
    unexpected = sorted(set(provided) - set(expected_by_name))

    return WeightDiff(
        missing=tuple(sorted(missing)),
        unexpected=tuple(unexpected),
        incompatible=tuple(incompatible),
        bad_tied=tuple(bad_tied),
        matched=tuple(sorted(matched)),
        optional_absent=tuple(sorted(optional_absent)),
    )

def verify_consumed(
    expected: Sequence[WeightSpec],
    consumed: Iterable[str],
    *,
    context: str = "",
) -> None:
    """What the model actually bound.

    A weight that was read, transformed, copied to the device and then never
    bound is the most expensive failure in the list — it costs the full load
    time and leaves the parameter it should have filled holding whatever
    ``__init__`` put there.  Nothing else in the pipeline notices: the bytes
    moved, the ledger balanced, the shapes agreed.
    """
    verify_consumed_names(
        frozenset(w.name for w in expected if not w.optional),
        consumed,
        offered=frozenset(w.name for w in expected),
        context=context,
    )

def verify_consumed_names(
    required: frozenset[str],
    consumed: Iterable[str],
    *,
    offered: frozenset[str] | None = None,
    context: str = "",
) -> None:
    """The same check, when only names are in hand.

    The composition root holds a ``ParameterStore``, not a weight schema — under
    EP a rank owns only its own experts, so the store's local binding set is the
    correct expectation and there are no specs to pass.  Sharing the
    implementation rather than adapting one side to the other keeps one
    definition of "the model consumed what it was given".
    """
    consumed_set = set(consumed)
    allowed = offered if offered is not None else required
    unbound = sorted(required - consumed_set)
    stray = sorted(consumed_set - allowed)
    if not unbound and not stray:
        return
    where = f"{context}: " if context else ""
    lines = [f"{where}model did not consume the weights it was given"]
    if unbound:
        lines.append(
            f"  loaded but never bound: {unbound[:8]}" + (" ..." if len(unbound) > 8 else "")
        )
    if stray:
        lines.append(
            f"  bound but never expected: {stray[:8]}" + (" ..." if len(stray) > 8 else "")
        )
    raise WeightMismatchError("\n".join(lines))
