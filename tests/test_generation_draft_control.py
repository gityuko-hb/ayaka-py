"""Acceptance-depth and launch-cost controller behavior, without device work."""

from dataclasses import replace

import pytest
from ayaka.generation import (
    DraftController,
    DraftLoad,
    DraftOutlook,
    FeatureRef,
    FeatureRows,
    HeadOutcome,
    Mtp,
    SelectedToken,
    SelectSpec,
    TimedLaunch,
    Verification,
    share_rows,
)


def test_unmeasured_load_starts_widest_and_scores_unverified_depths_later():
    control = DraftController(3)
    assert control.generate(3, 5) == 3
    control.drafted(3, (1, 2, 3))
    assert control.outlook().right == (1.0, 1.0, 1.0)
    assert control.verify(3, 1) == 1
    control.committed((1, 2))
    assert control.blocks[0].matched == 2
    control.committed((9,))
    assert control.right() == (1.0, 1.0, 0.0)
    assert not control.blocks


def test_short_draft_is_wrong_from_where_it_ends():
    control = DraftController(4)
    control.generate(4, 4)
    control.drafted(4, (1, 2))
    assert control.right() == (1.0, 1.0, 0.0, 0.0)
    control.committed((9,))
    assert control.right() == (0.0, 0.0, 0.0, 0.0)


def test_timing_is_counted_once_and_shared_by_context_and_batch():
    control = DraftController(3)
    control.generate(3, 5)
    control.timed(TimedLaunch.DRAFT, 0.01, 2, 8)
    control.timed(TimedLaunch.DRAFT, 99, 2, 8)
    assert control.measured().draft == {3: 0.01}
    control.drafted(3, (1, 2, 3))
    control.verify(3, 3)
    control.timed(TimedLaunch.TARGET, 0.02, 2, 8)
    assert control.measured().verify == {8: 0.02}
    assert (8, 2) in control.load.costs


def test_shared_load_seeds_new_requests_and_forks_keep_one_measurement_owner():
    load = DraftLoad()
    first = DraftController(3, load)
    first.generate(3, 2)
    first.drafted(3, (1, 2, 3))
    first.committed((1, 8))
    second = DraftController(3, load)
    assert second.right() == first.right()
    fork = second.fork()
    assert fork.load is load
    fork._right[0] = 0
    assert second.right()[0] == 1


def test_row_sharing_prefers_the_most_likely_prefixes_and_reserves_anchors():
    load = DraftLoad()
    outlooks = (
        DraftOutlook(load, 8, (0.9, 0.1), 0),
        DraftOutlook(load, 8, (0.8, 0.7), 0),
    )
    assert share_rows(outlooks, 5) == (2, 3)
    assert share_rows(outlooks, 2) == (1, 1)
    assert share_rows(outlooks, 1) is None
    assert share_rows(outlooks[:1], 5) is None


@pytest.mark.parametrize("limit", [0, 1, 2, 3, 6, 20])
def test_controller_bounds_every_generated_verified_width(limit):
    control = DraftController(6)
    generated = control.generate(limit, 1)
    assert 0 <= generated <= min(limit, 6)
    control.drafted(generated, (1,) * generated)
    verified = control.verify(generated, limit)
    assert 0 <= verified <= generated


def test_invalid_durations_do_not_pollute_shared_costs():
    control = DraftController(3)
    control.generate(3, 1)
    for seconds in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            control.timed(TimedLaunch.DRAFT, seconds, 1, 3)
    assert not control.load.costs


def test_fallible_method_preview_does_not_publish_adaptive_learning():
    class Reader:
        def read(self, feature, start, count):
            return FeatureRows(bytes(count), count, 1)

    factory = Mtp.reading_target("x", 2).adaptive()
    state = factory.create()
    state.prime("r", (1,), 2, FeatureRef("p", 1), 1, None, Reader())
    operation = state.propose("r", (SelectSpec(),) * 2).operations[0]
    state.reconcile(operation, HeadOutcome((SelectedToken(3), SelectedToken(4))))
    state.propose("r", (SelectSpec(),) * 2)
    preview = state.fork_transition()
    verification = Verification((2, 3, 4), 2, 9, FeatureRef("v", 3))
    with pytest.raises(ValueError):
        preview.observe("r", replace(verification, features=None), Reader())
    assert factory.load.right is None
    preview = state.fork_transition()
    preview.observe("r", verification, Reader())
    assert factory.load.right is None
    preview.publish_learning()
    assert factory.load.right == [1.0, 0.0]
