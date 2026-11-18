"""Sampling coverage limits (instrumentation-overhead-acceptance-v1) over synthetic summaries."""

import pytest
from agentwatch_agent.resources import interval_summary
from agentwatch_agent.sampling_limits import LIMITS, judge_sampling, short_steps

DECLARED = 0.25
NS = 1_000_000_000


def ticks(*gaps_seconds: float) -> list[int]:
    """Round ticks starting at 0 with the given gaps between consecutive rounds."""
    result = [0]
    for gap in gaps_seconds:
        result.append(result[-1] + round(gap * NS))
    return result


def summary(*gaps_seconds: float) -> dict:
    return interval_summary(ticks(*gaps_seconds), DECLARED)


def test_on_cadence_sampling_meets_every_limit():
    judged = judge_sampling([summary(*[0.25] * 20)], DECLARED)
    assert judged["verdict"] == "met"
    assert judged["limits_version"] == LIMITS["version"]
    checks = judged["checks"]
    assert checks["interval_median"]["value"] == pytest.approx(0.25)
    assert checks["interval_median"]["limit"] == [pytest.approx(0.225), pytest.approx(0.275)]
    assert checks["interval_max"]["value"] == pytest.approx(0.25)
    assert checks["missed_fraction"]["value"] == 0
    assert all(check["met"] for check in checks.values())


def test_a_slow_median_interval_is_not_met():
    judged = judge_sampling([summary(*[0.3] * 20)], DECLARED)
    assert judged["verdict"] == "not_met"
    assert not judged["checks"]["interval_median"]["met"]
    # The declared cadence expects more rounds than a slow sampler takes, so they count as missed.
    assert judged["failed_checks"] == ["interval_median", "missed_fraction"]


def test_one_gap_longer_than_twice_the_interval_is_not_met():
    judged = judge_sampling([summary(*[0.25] * 20, 0.6)], DECLARED)
    assert judged["checks"]["interval_max"]["value"] == pytest.approx(0.6)
    assert "interval_max" in judged["failed_checks"]


def test_a_gap_of_exactly_twice_the_interval_is_within_the_limit():
    judged = judge_sampling([summary(*[0.25] * 20, 0.5)], DECLARED)
    assert judged["checks"]["interval_max"]["met"]


def test_missed_rounds_above_one_percent_are_not_met():
    # 21 rounds over 5.25 s, where the cadence expects 22: one missed round, about 4.5%.
    judged = judge_sampling([summary(*[0.25] * 19, 0.5)], DECLARED)
    missed = judged["checks"]["missed_fraction"]
    assert missed["missed_rounds"] == 1 and missed["expected_rounds"] == 22
    assert missed["value"] == pytest.approx(1 / 22)
    assert not missed["met"]


def test_late_rounds_are_reported_without_a_limit():
    judged = judge_sampling([summary(*[0.25] * 200, 0.4)], DECLARED)
    assert judged["late_rounds"] == 1
    assert judged["verdict"] == "met"


def test_samplers_are_pooled_across_repetitions():
    judged = judge_sampling(
        [summary(*[0.25] * 4), summary(*[0.26] * 4), summary(*[0.3] * 4)], DECLARED
    )
    assert judged["samplers"] == 3 and judged["measured_gaps"] == 12
    # The median of the three samplers' own medians, not of their pooled gaps.
    assert judged["checks"]["interval_median"]["value"] == pytest.approx(0.26)
    assert judged["checks"]["interval_median"]["statistic"] == "median_of_sampler_medians"
    assert judged["checks"]["interval_max"]["value"] == pytest.approx(0.3)


def test_samplers_with_a_single_round_have_no_interval_and_are_counted():
    judged = judge_sampling([summary(), summary(*[0.25] * 12)], DECLARED)
    assert judged["samplers_without_interval"] == 1
    assert judged["verdict"] == "met"


def test_no_measured_interval_is_not_evaluated_rather_than_met():
    for summaries in ([], [summary()], [interval_summary([], DECLARED)]):
        judged = judge_sampling(summaries, DECLARED)
        assert judged["verdict"] == "not_evaluated"
        assert judged["checks"] is None


def test_the_declared_interval_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        judge_sampling([summary(0.25)], 0)


def test_short_steps_are_counted_among_executed_steps_only():
    steps = [
        {"window": {"clock_id": "c"}, "shorter_than_interval": True},
        {"window": {"clock_id": "c"}, "shorter_than_interval": False},
        {"window": {"clock_id": "c"}, "shorter_than_interval": True},
        {"window": None},  # never executed: not a short step and not in the share
    ]
    assert short_steps(steps) == {"executed": 3, "shorter_than_interval": 2, "share": 2 / 3}
    assert short_steps([{"window": None}]) == {
        "executed": 0,
        "shorter_than_interval": 0,
        "share": None,
    }


# --- Review regressions (#94) ----------------------------------------------------------------


def test_fewer_than_ten_gaps_is_not_evaluated_rather_than_judged():
    judged = judge_sampling([summary(*[0.25] * 9)], DECLARED)
    assert judged["verdict"] == "not_evaluated"
    assert judged["measured_gaps"] == 9
    assert judged["reason"] == "9 measured gaps, fewer than 10"
    assert judge_sampling([summary(*[0.25] * 10)], DECLARED)["verdict"] == "met"


def test_a_configured_resource_without_samples_is_not_met():
    judged = judge_sampling([summary(*[0.25] * 20)], DECLARED, ["gpu-1"])
    assert judged["verdict"] == "not_met"
    assert judged["unsampled_resources"] == ["gpu-1"]
    assert judged["failed_checks"] == ["unsampled_resources"]
    # Even with no gap measured anywhere, an unsampled probe is a failure, not missing data.
    assert judge_sampling([], DECLARED, ["gpu-1"])["verdict"] == "not_met"


def test_a_summary_with_a_median_but_no_maximum_is_rejected():
    broken = summary(*[0.25] * 20) | {"observed_interval_max_seconds": None}
    with pytest.raises(ValueError, match="lacks its maximum"):
        judge_sampling([broken], DECLARED)


@pytest.mark.parametrize("field", ["missed_rounds", "expected_rounds", "late_rounds"])
def test_a_missing_field_is_an_error_not_a_measured_zero(field):
    incomplete = summary(*[0.25] * 20)
    del incomplete[field]
    with pytest.raises(KeyError, match=field):
        judge_sampling([incomplete], DECLARED)
