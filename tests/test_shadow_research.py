"""Research prototypes R1-R3, all shadow-only and off by default.

* R1 prepare-ahead: counts iterations that hit the inflight ceiling while
  requests owning no ticket waited, so raising ``max_inflight`` can be judged
  on data. It observes launches and never changes them.
* R2 forecast: bounded per-prompt-bucket histograms give output-length
  quantiles, capped by the remaining budget, with calibration over uncensored
  finishes. Nothing in the scheduler or allocator reads it.
* R3 cost-to-ready: restore a tiered prefix only if it is ready sooner by a
  margin, fits the pinned/transfer budgets and stops at a boundary every group
  can restore. A decision function only; the tiering path does not call it.

The parity test is the shadow contract: enabling the probes changes no output.
"""

from __future__ import annotations

import math

import pytest
from ayaka.configs.serving import ServingConfig
from ayaka.prefix.restore_cost import RestoreChoice, RestoreInputs, decide_restore
from ayaka.sched.forecast import _OUTPUT_BINS, OutputLengthForecaster
from cpu_e2e import MODEL, CpuServing, prompt_of

PROMPTS = [prompt_of(*[(i * 11 + k * 5) % 120 for k in range(4 + i)]) for i in range(6)]


# ---------------------------------------------------------------------------
# R2: output-length forecast
# ---------------------------------------------------------------------------


def _trained(lengths, *, prompt: int = 10, min_samples: int = 16) -> OutputLengthForecaster:
    forecaster = OutputLengthForecaster(quantile=0.9, min_samples=min_samples)
    for length in lengths:
        forecaster.observe(prompt, length, censored=False, forecast=10**6)
    return forecaster


def test_without_evidence_the_forecast_is_the_whole_remaining_budget() -> None:
    forecaster = OutputLengthForecaster(min_samples=4)
    assert forecaster.predict(10, 200) == 200
    assert forecaster.predict(10, 200, generated=50) == 150
    forecaster.observe(10, 5, censored=False, forecast=200)
    assert forecaster.predict(10, 200) == 200  # one sample is not evidence yet


def test_the_forecast_is_a_conservative_quantile_capped_by_the_budget() -> None:
    forecaster = _trained(range(1, 101))
    # The 90th percentile is 90 tokens; the estimate is the holding bin's upper edge.
    assert forecaster.predict(10, 500) == 96
    assert forecaster.predict(10, 50) == 50  # never above max_tokens
    assert forecaster.predict(10, 500, generated=20) == 76
    assert forecaster.predict(10, 500, generated=200) == 0


def test_prompt_buckets_are_independent() -> None:
    forecaster = _trained(range(1, 101), prompt=10)
    assert forecaster.predict(2000, 300) == 300  # another bucket has no evidence


def test_predicted_pages_count_only_new_pages() -> None:
    forecaster = _trained([40] * 20)  # every request produced 40 tokens
    assert forecaster.predict(10, 500) == 48
    # 10 prompt + 48 forecast = 58 tokens: 4 pages of 16 in total.
    assert forecaster.predicted_pages(10, 500, page_size=16, resident_tokens=0) == 4
    # Running request with 10 + 30 tokens resident (3 pages): 18 more -> 4 pages.
    assert forecaster.predicted_pages(10, 500, page_size=16, generated=30) == 1


def test_calibration_scores_only_uncensored_finishes() -> None:
    forecaster = OutputLengthForecaster()
    forecaster.observe(10, 30, censored=False, forecast=40)
    forecaster.observe(10, 50, censored=False, forecast=40)
    forecaster.observe(10, 64, censored=True, forecast=40)  # hit max_tokens: unknown
    calibration = forecaster.calibration()
    assert (calibration.observed, calibration.uncensored, calibration.censored) == (3, 2, 1)
    assert calibration.covered == 1 and calibration.coverage == 0.5
    assert calibration.slack_tokens == 10


def test_memory_is_bounded_however_many_requests_are_observed() -> None:
    forecaster = _trained(range(1, 100_000, 7))
    for histogram in forecaster._buckets:
        assert len(histogram.counts) == len(_OUTPUT_BINS)


def test_invalid_forecaster_options_are_refused() -> None:
    with pytest.raises(ValueError):
        OutputLengthForecaster(quantile=1.0)
    with pytest.raises(ValueError):
        OutputLengthForecaster(min_samples=0)


# ---------------------------------------------------------------------------
# Shadow contract on the serving stack
# ---------------------------------------------------------------------------


def _shadow_config() -> ServingConfig:
    return ServingConfig(
        model=MODEL, structured_outputs=False, shadow_forecast=True, shadow_prepare_ahead=True
    )


def test_enabling_the_shadow_probes_changes_no_output() -> None:
    with CpuServing() as plain:
        expected = [g.text for g in plain.generate_concurrently(PROMPTS[:4], max_tokens=16)]
        assert plain.runtime.prepare_ahead is None and plain.service.forecaster is None
    with CpuServing(config=_shadow_config()) as shadowed:
        got = [g.text for g in shadowed.generate_concurrently(PROMPTS[:4], max_tokens=16)]
        report = shadowed.http().get("/stats").json()["shadow"]
    assert got == expected
    assert report["forecast"]["observed"] == 4
    assert report["forecast"]["censored"] == 4  # all hit max_tokens=16
    assert report["prepare_ahead"]["iterations"] > 0


def test_prepare_ahead_sees_waiting_work_behind_the_inflight_ceiling() -> None:
    options = {"max_inflight": 1, "max_requests": 2, "max_pending_requests": 8}
    with CpuServing(config=_shadow_config(), **options) as stack:
        stack.generate_concurrently(PROMPTS, max_tokens=12)  # 6 requests, 2 sequences
        snapshot = stack.runtime.prepare_ahead.snapshot()  # type: ignore[union-attr]
        stack.assert_quiescent_accounting()
    assert snapshot.capacity_blocked > 0 and snapshot.blocked_ready_total > 0
    # The one ticket in flight holds both sequences; at most 6 - 2 wait outside it.
    assert snapshot.blocked_ready_max <= len(PROMPTS) - 2


def test_the_forecast_is_scored_against_natural_finishes_only() -> None:
    with CpuServing(config=_shadow_config()) as stack:
        stack.generate(PROMPTS[0], max_tokens=8)  # length: censored
        stack.generate(PROMPTS[1], max_tokens=200, timeout=0.02)  # cut short: ignored
        stack.generate(PROMPTS[2], max_tokens=200, cancel_after=2)  # cancelled: ignored
        calibration = stack.service.forecaster.calibration()  # type: ignore[union-attr]
    assert (calibration.observed, calibration.censored) == (1, 1)


# ---------------------------------------------------------------------------
# R3: cost-to-ready restore decision
# ---------------------------------------------------------------------------


def _inputs(**overrides) -> RestoreInputs:
    values = {
        "group_boundaries": ((16, 32, 48, 64),),
        "limit": 64,
        "bytes_per_token": 1024,
        "restore_bytes_per_second": 10e9,
        "restore_overhead_seconds": 50e-6,
        "prefill_tokens_per_second": 10_000.0,
        "margin": 0.2,
        "pinned_bytes_available": 1 << 20,
        "transfer_slots_available": 1,
    } | overrides
    return RestoreInputs(**values)


def test_a_fast_copy_path_restores() -> None:
    decision = decide_restore(_inputs())
    assert decision.choice is RestoreChoice.RESTORE and decision.boundary == 64
    assert decision.restore_seconds < decision.recompute_seconds


def test_a_slow_copy_path_recomputes() -> None:
    decision = decide_restore(_inputs(restore_bytes_per_second=1e6))
    assert decision.choice is RestoreChoice.RECOMPUTE
    assert decision.reason == "not faster by the margin"


def test_a_win_inside_the_uncertainty_margin_is_not_taken() -> None:
    # Restore 5.5 ms vs recompute 6.4 ms: faster, but not by 20 %.
    decision = decide_restore(_inputs(restore_overhead_seconds=5.5e-3 - 6.5536e-6))
    assert decision.restore_seconds < decision.recompute_seconds
    assert decision.choice is RestoreChoice.RECOMPUTE


@pytest.mark.parametrize(
    ("budget", "reason"),
    [
        ({"pinned_bytes_available": 1024}, "pinned staging budget"),
        ({"transfer_slots_available": 0}, "no free transfer slot"),
    ],
)
def test_hard_budgets_override_a_faster_restore(budget: dict, reason: str) -> None:
    decision = decide_restore(_inputs(**budget))
    assert decision.choice is RestoreChoice.RECOMPUTE and decision.reason == reason


def test_a_partial_restore_stops_where_every_group_can_resume() -> None:
    decision = decide_restore(_inputs(group_boundaries=((16, 32, 48), (32, 64))))
    assert decision.boundary == 32  # not 48 (group 2 cannot) nor 64 (group 1 cannot)
    none = decide_restore(_inputs(group_boundaries=((16,), (32,))))
    assert none.choice is RestoreChoice.RECOMPUTE and none.reason == "no common boundary"


@pytest.mark.parametrize(
    "bad",
    [
        {"restore_bytes_per_second": 0.0},
        {"prefill_tokens_per_second": math.nan},
        {"restore_overhead_seconds": -1.0},
        {"margin": math.inf},
        {"bytes_per_token": 0},
    ],
)
def test_unmeasured_or_invalid_inputs_are_refused(bad: dict) -> None:
    with pytest.raises(ValueError):
        _inputs(**bad)
