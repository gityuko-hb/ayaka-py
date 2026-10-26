"""Stop / length / EOS precedence and the speculative publication limit (research C02).

The SGLang snapshot had branches where a speculative token cap and an EOS in the
same step disagreed about which won.  Ayaka routes both decisions through
``ResolvedStopPolicy`` and the pure ``OutputProcessor.token_stop_limit`` oracle
the completion coordinator applies *before* committing speculative KV rows.  The
coordinator's truncation (``keep``) is only correct if that oracle agrees with
what sequential publication would decide, so the agreement is pinned here, both
on hand-picked precedence cases and as a property over random token streams.
"""

from __future__ import annotations

import pytest
from ayaka.request.schema import Request, RequestId, StopCriteria
from ayaka.request.stream import ResolvedStopPolicy
from ayaka.runtime.output import OutputProcessor
from ayaka.sched.outcome import FinishReason
from hypothesis import given, settings
from hypothesis import strategies as st

STOP, EOS_ID = 7, 9
PROMPT = 3


def _policy(**kwargs) -> ResolvedStopPolicy:
    defaults = dict(max_tokens=5, stop_token_ids=(STOP,), eos_token_ids=(EOS_ID,))
    return ResolvedStopPolicy(**{**defaults, **kwargs})


def _processor(**stop) -> OutputProcessor:
    output = OutputProcessor(eos_token_ids=(EOS_ID,))
    output.register(
        Request(
            request_id=RequestId("r"),
            prompt_token_ids=(1, 2, 3),
            stop=StopCriteria(**{"max_tokens": 8, "stop_token_ids": (STOP,), **stop}),
        )
    )
    return output


# ---------------------------------------------------------------------------
# ResolvedStopPolicy.evaluate precedence
# ---------------------------------------------------------------------------


def test_a_stop_token_on_the_last_allowed_position_is_stop_not_length() -> None:
    assert _policy().evaluate(STOP, generated_tokens=5) is FinishReason.STOP


def test_eos_on_the_last_allowed_position_is_eos_not_length() -> None:
    assert _policy().evaluate(EOS_ID, generated_tokens=5) is FinishReason.EOS


def test_an_ordinary_token_on_the_last_allowed_position_is_length() -> None:
    assert _policy().evaluate(1, generated_tokens=5) is FinishReason.LENGTH
    assert _policy().evaluate(1, generated_tokens=4) is None


def test_an_explicit_stop_id_wins_over_eos_when_the_same_token_is_both() -> None:
    policy = _policy(stop_token_ids=(EOS_ID,))
    assert policy.evaluate(EOS_ID, generated_tokens=2) is FinishReason.STOP


def test_a_text_stop_match_ends_generation_and_wins_over_length() -> None:
    policy = _policy(stop_strings=("END",))
    assert policy.evaluate(1, generated_tokens=5, stop_matched="END") is FinishReason.STOP
    assert policy.evaluate(1, generated_tokens=2, stop_matched="END") is FinishReason.STOP


def test_an_unknown_text_stop_or_overlong_generation_is_a_contract_violation() -> None:
    with pytest.raises(ValueError, match="unknown text stop"):
        _policy().evaluate(1, generated_tokens=1, stop_matched="nope")
    with pytest.raises(ValueError, match="exceeded its token limit"):
        _policy().evaluate(1, generated_tokens=6)


def test_min_tokens_defers_stops_until_the_minimum_is_generated() -> None:
    policy = _policy(min_tokens=3)
    assert policy.evaluate(STOP, generated_tokens=3) is None  # still below the minimum
    assert policy.evaluate(EOS_ID, generated_tokens=3) is None
    assert policy.evaluate(STOP, generated_tokens=4) is FinishReason.STOP
    assert policy.evaluate(EOS_ID, generated_tokens=4) is FinishReason.EOS


def test_ignore_eos_never_stops_on_eos_but_still_honours_explicit_stops() -> None:
    policy = _policy(ignore_eos=True)
    assert policy.evaluate(EOS_ID, generated_tokens=2) is None
    assert policy.evaluate(STOP, generated_tokens=2) is FinishReason.STOP
    assert policy.masked_token_ids(0) == frozenset()  # nothing to mask with min_tokens == 0


@pytest.mark.parametrize("min_tokens", [0, 1, 3, 5])
@pytest.mark.parametrize("ignore_eos", [False, True])
def test_the_sampler_mask_and_the_post_sample_check_agree_on_min_tokens(
    min_tokens: int, ignore_eos: bool
) -> None:
    """A token banned before sampling must be exactly one that could not stop afterwards."""
    policy = _policy(min_tokens=min_tokens, ignore_eos=ignore_eos)
    for before in range(policy.max_tokens):
        masked = policy.masked_token_ids(before)
        after = before + 1
        for token in (STOP, EOS_ID):
            can_stop = policy.evaluate(token, generated_tokens=after) in (
                FinishReason.STOP,
                FinishReason.EOS,
            )
            if token == EOS_ID and ignore_eos:
                assert not can_stop and token not in masked
            elif before < min_tokens:
                assert token in masked and not can_stop
            else:
                assert token not in masked and can_stop


# ---------------------------------------------------------------------------
# OutputProcessor.token_stop_limit: the speculative publication oracle
# ---------------------------------------------------------------------------


def test_the_limit_includes_the_first_stopping_token_and_nothing_after_it() -> None:
    output = _processor()
    assert output.token_stop_limit("r", (1, 2, STOP, 4, STOP), 0) == 3
    assert output.token_stop_limit("r", (STOP, 2), 0) == 1
    assert output.token_stop_limit("r", (1, 2, EOS_ID, STOP), 0) == 3  # EOS first
    assert output.token_stop_limit("r", (1, 2, 4), 0) == 3  # no stop: everything publishes


def test_the_limit_honours_min_tokens_using_the_already_generated_count() -> None:
    """With ``min_tokens=2`` the 1st and 2nd generated tokens cannot stop; the 3rd can."""
    output = _processor(min_tokens=2)
    assert output.token_stop_limit("r", (STOP, STOP, STOP, 1), 0) == 3  # two too early
    assert output.token_stop_limit("r", (STOP, STOP, 1), 1) == 2  # 2nd too early, 3rd stops
    assert output.token_stop_limit("r", (STOP, 1), 2) == 1  # already past the minimum
    assert output.token_stop_limit("r", (STOP, 1), 0) == 2  # no token may stop: all publish


def test_ignore_eos_requests_never_stop_on_eos_in_the_limit() -> None:
    output = _processor(ignore_eos=True)
    assert output.token_stop_limit("r", (EOS_ID, 1, EOS_ID), 0) == 3


def test_the_limit_is_pure() -> None:
    output = _processor()
    before = (output.events("r"), output.text("r"))
    first = output.token_stop_limit("r", (1, STOP, 2), 0)
    assert output.token_stop_limit("r", (1, STOP, 2), 0) == first
    assert (output.events("r"), output.text("r")) == before == ((), "")


def test_text_stops_cannot_be_evaluated_before_publication() -> None:
    class _Tokenizer:
        def new_detokenizer(self, params, *, prompt_token_ids):
            raise AssertionError("not reached")

    output = OutputProcessor(eos_token_ids=(EOS_ID,))
    # Without a tokenizer a text-stop request cannot even register ...
    with pytest.raises(Exception, match="tokenizer"):
        output.register(
            Request(
                request_id=RequestId("t"),
                prompt_token_ids=(1, 2, 3),
                stop=StopCriteria(stop_strings=("x",)),
            )
        )
    # ... so the speculative planner is never handed one.
    assert output.text_stops_supported is False


def test_publishing_decides_the_same_finish_the_limit_predicted() -> None:
    output = _processor(max_tokens=3)
    tokens = (1, STOP, 2)
    limit = output.token_stop_limit("r", tokens, 0)
    decisions = [
        output.on_published("r", t, sequence_epoch=1, generated_tokens=i, prompt_tokens=PROMPT)
        for i, t in enumerate(tokens[:limit], start=1)
    ]
    assert decisions[0] is None
    assert decisions[1] is not None and decisions[1].reason is FinishReason.STOP
    assert limit == 2


@settings(max_examples=300, deadline=None)
@given(
    tokens=st.lists(st.sampled_from([1, 2, STOP, EOS_ID]), min_size=1, max_size=8),
    min_tokens=st.integers(min_value=0, max_value=4),
    ignore_eos=st.booleans(),
    already=st.integers(min_value=0, max_value=3),
)
def test_the_limit_matches_sequential_publication(
    tokens: list[int], min_tokens: int, ignore_eos: bool, already: int
) -> None:
    """Property: the speculative oracle truncates exactly where real publication stops."""
    # Leave room so LENGTH does not interfere; min_tokens may not exceed max_tokens.
    max_tokens = max(already + len(tokens) + 1, min_tokens)
    output = _processor(max_tokens=max_tokens, min_tokens=min_tokens, ignore_eos=ignore_eos)
    limit = output.token_stop_limit("r", tuple(tokens), already)

    stopped_at = len(tokens)
    for index, token in enumerate(tokens, start=1):
        decision = output.on_published(
            "r",
            token,
            sequence_epoch=1,
            generated_tokens=already + index,
            prompt_tokens=PROMPT,
        )
        if decision is not None:
            assert decision.reason in (FinishReason.STOP, FinishReason.EOS)
            stopped_at = index
            break
    assert limit == stopped_at
