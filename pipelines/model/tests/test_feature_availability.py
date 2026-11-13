"""Feature availability on synthetic training and held-out rows; no row here is recorded evidence."""

import math

import pytest
from agentwatch_model.feature_availability import (
    AvailabilityError,
    availability_report,
    describe_feature,
    gain_shares,
    markdown_table,
    range_table,
)
from agentwatch_model.features import FEATURES, FeatureError
from training_fixtures import featured_row


def held_out_row(index: int, configuration: str, **overrides) -> dict:
    return featured_row("inference", index) | {"configuration": configuration} | overrides


def even_gain() -> dict[str, float]:
    return dict.fromkeys(FEATURES, 1 / len(FEATURES))


class FakeBooster:
    def __init__(self, names, gains):
        self.names, self.gains = list(names), list(gains)

    def feature_name(self):
        return self.names

    def feature_importance(self, importance_type):
        assert importance_type == "gain"
        return self.gains


def test_describe_counts_missing_and_distinct_values():
    rows = [{"queue_seconds": value} for value in (0.5, None, 0.5, 2.0)]
    described = describe_feature(rows, "queue_seconds")
    assert {key: described[key] for key in ("rows", "missing", "missing_fraction")} == {
        "rows": 4,
        "missing": 1,
        "missing_fraction": 0.25,
    }
    assert described["distinct_values"] == 2
    assert described["constant"] is False and described["all_missing"] is False


def test_a_single_present_value_is_constant():
    rows = [{"cgroup_available": value} for value in (True, True, None)]
    described = describe_feature(rows, "cgroup_available")
    assert described["constant"] is True and described["all_missing"] is False
    assert described["distinct_values"] == 1


def test_every_value_missing_is_all_missing_not_zero():
    rows = [{"worker_capacity": None}, {"worker_capacity": None}]
    described = describe_feature(rows, "worker_capacity")
    assert described["missing_fraction"] == 1.0
    assert described["all_missing"] is True and described["constant"] is False
    assert described["distinct_values"] == 0


def test_a_measured_zero_is_present_not_missing():
    described = describe_feature([{"queue_seconds": 0.0}, {"queue_seconds": 0}], "queue_seconds")
    assert described["missing"] == 0 and described["constant"] is True


def test_describe_refuses_no_rows_and_bad_values():
    with pytest.raises(AvailabilityError, match="no rows"):
        describe_feature([], "queue_seconds")
    with pytest.raises(FeatureError, match="queue_seconds"):
        describe_feature([{"queue_seconds": "slow"}], "queue_seconds")
    with pytest.raises(FeatureError, match="missing feature queue_seconds"):
        describe_feature([{}], "queue_seconds")


def test_gain_shares_sum_to_one_by_feature_name():
    names = list(reversed(FEATURES))
    gains = [float(index) for index in range(len(names))]
    shares = gain_shares(FakeBooster(names, gains))
    assert list(shares) == list(FEATURES)
    present = [share for share in shares.values() if share is not None]
    assert len(present) == len(FEATURES) and math.isclose(sum(present), 1.0)
    assert shares[names[-1]] == pytest.approx(gains[-1] / sum(gains))


def test_gain_shares_refuse_other_features_and_no_gain():
    with pytest.raises(AvailabilityError, match="features"):
        gain_shares(FakeBooster(["other"], [1.0]))
    assert gain_shares(FakeBooster(FEATURES, [0.0] * len(FEATURES))) == dict.fromkeys(FEATURES)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -1.0])
def test_gain_shares_refuse_non_finite_or_negative_gain(bad):
    gains = [1.0] * (len(FEATURES) - 1) + [bad]
    with pytest.raises(AvailabilityError, match="gain"):
        gain_shares(FakeBooster(FEATURES, gains))


def test_report_compares_train_with_held_out_overall_and_per_configuration():
    train = [featured_row("none", index) for index in range(4)]
    held_out = [
        held_out_row(0, "cfg-default", cpu_pressure_max_avg10=None),
        held_out_row(1, "cfg-default", cpu_pressure_max_avg10=None),
        held_out_row(2, "cfg-hardware", cpu_pressure_max_avg10=None),
    ]
    train = [row | {"cpu_pressure_max_avg10": 0.1 * index} for index, row in enumerate(train)]
    report = availability_report(train, held_out, even_gain())

    assert report["row_sets"] == {
        "train": 4,
        "train_not_classified": 0,
        "held_out": 3,
        "held_out_by_configuration": {"cfg-default": 2, "cfg-hardware": 1},
    }
    (pressure,) = [
        item for item in report["features"] if item["feature"] == "cpu_pressure_max_avg10"
    ]
    assert pressure["train"]["missing_fraction"] == 0.0
    assert pressure["held_out"]["missing_fraction"] == 1.0
    assert pressure["held_out"]["missing_fraction_gap"] == 1.0
    assert pressure["held_out"]["flags"] == ["missing_on_held_out_only"]
    by_configuration = pressure["held_out_by_configuration"]
    assert set(by_configuration) == {"cfg-default", "cfg-hardware"}
    assert by_configuration["cfg-hardware"]["flags"] == ["missing_on_held_out_only"]


def test_constant_on_held_out_only_is_flagged():
    train = [featured_row("none", index, queue_seconds=0.1 * index) for index in range(3)]
    held_out = [held_out_row(index, "cfg-default", queue_seconds=0.4) for index in range(3)]
    report = availability_report(train, held_out, even_gain())
    (queue,) = [item for item in report["features"] if item["feature"] == "queue_seconds"]
    assert queue["held_out"]["flags"] == ["constant_on_held_out_only"]


def test_a_feature_missing_everywhere_is_not_flagged():
    train = [featured_row("none", index, worker_capacity=None) for index in range(2)]
    held_out = [held_out_row(index, "cfg-default", worker_capacity=None) for index in range(2)]
    report = availability_report(train, held_out, even_gain())
    (capacity,) = [item for item in report["features"] if item["feature"] == "worker_capacity"]
    assert capacity["train"]["all_missing"] and capacity["held_out"]["all_missing"]
    assert capacity["held_out"]["flags"] == []
    assert capacity["held_out"]["missing_fraction_gap"] == 0.0


def test_features_are_ordered_by_gain_share_then_name():
    gain = dict.fromkeys(FEATURES, 0.0) | {"telemetry_coverage": 0.7, "queue_seconds": 0.3}
    report = availability_report([featured_row("none", 0)], [held_out_row(0, "cfg-default")], gain)
    names = [item["feature"] for item in report["features"]]
    assert names[:2] == ["telemetry_coverage", "queue_seconds"]
    assert names[2:] == sorted(names[2:])
    assert report["features"][0]["gain_share"] == 0.7


def test_unknown_gain_sorts_last():
    gain = dict.fromkeys(FEATURES)
    report = availability_report([featured_row("none", 0)], [held_out_row(0, "cfg-default")], gain)
    assert [item["feature"] for item in report["features"]] == sorted(FEATURES)
    assert all(item["gain_share"] is None for item in report["features"])


def test_report_refuses_rows_without_a_configuration():
    with pytest.raises(AvailabilityError, match="configuration"):
        availability_report([featured_row("none", 0)], [featured_row("none", 1)], even_gain())


def test_report_refuses_gain_for_other_features():
    with pytest.raises(AvailabilityError, match="gain"):
        availability_report(
            [featured_row("none", 0)], [held_out_row(0, "cfg-default")], {"queue_seconds": 1.0}
        )


def test_report_states_it_is_descriptive():
    report = availability_report(
        [featured_row("none", 0)], [held_out_row(0, "cfg-default")], even_gain()
    )
    assert report["evaluation"] is False
    assert any("not a cause" in line for line in report["limitations"])


def test_markdown_table_has_one_line_per_feature_with_percentages():
    train = [featured_row("none", 0, cpu_pressure_max_avg10=0.2)]
    held_out = [held_out_row(0, "cfg-default", cpu_pressure_max_avg10=None)]
    table = markdown_table(availability_report(train, held_out, even_gain()))
    lines = table.strip().splitlines()
    assert lines[0].startswith("| Feature | Gain share | Train missing | Held-out missing |")
    assert "cfg-default missing" in lines[0]
    assert len(lines) == 2 + len(FEATURES)
    (pressure,) = [line for line in lines if line.startswith("| `cpu_pressure_max_avg10`")]
    assert "| 0.0% | 100.0% |" in pressure and "missing_on_held_out_only" in pressure


def test_markdown_table_keeps_almost_every_row_apart_from_every_row():
    train = [featured_row("none", 0, cpu_pressure_max_avg10=0.2)]
    held_out = [
        held_out_row(index, "cfg-default", cpu_pressure_max_avg10=None if index else 0.1)
        for index in range(1000)
    ]
    table = markdown_table(availability_report(train, held_out, even_gain()))
    (pressure,) = [line for line in table.splitlines() if "`cpu_pressure_max_avg10`" in line]
    assert "| 99.9% |" in pressure and "100.0%" not in pressure


def test_markdown_table_shows_unknown_gain_as_not_available():
    table = markdown_table(
        availability_report(
            [featured_row("none", 0)], [held_out_row(0, "cfg-default")], dict.fromkeys(FEATURES)
        )
    )
    assert "| n/a |" in table


def test_describe_reports_quantiles_and_range_of_present_values():
    rows = [{"queue_seconds": value} for value in (None, *range(1, 102))]
    described = describe_feature(rows, "queue_seconds")
    assert described["min"] == 1.0 and described["max"] == 101.0
    assert described["quantiles"] == {"p05": 6.0, "p50": 51.0, "p95": 96.0}


def test_all_missing_has_no_quantiles_or_range():
    described = describe_feature([{"cpu_seconds": None}], "cpu_seconds")
    assert described["quantiles"] is None
    assert described["min"] is None and described["max"] is None


def test_held_out_values_outside_the_train_range_are_counted_as_shares_of_present_values():
    train = [
        featured_row("none", index, queue_seconds=value) for index, value in enumerate((1, 2, 3))
    ]
    held_out = [
        held_out_row(index, "cfg-default", queue_seconds=value)
        for index, value in enumerate((0.5, 2.0, 4.0, 5.0, None))
    ]
    report = availability_report(train, held_out, even_gain())
    (queue,) = [item for item in report["features"] if item["feature"] == "queue_seconds"]
    assert queue["held_out"]["outside_train_range"] == {
        "below_train_min": 0.25,
        "above_train_max": 0.5,
    }
    by_configuration = queue["held_out_by_configuration"]["cfg-default"]
    assert by_configuration["outside_train_range"] == queue["held_out"]["outside_train_range"]


def test_outside_train_range_is_null_without_present_values_on_both_sides():
    train = [featured_row("none", 0, cpu_seconds=None, queue_seconds=1.0)]
    held_out = [held_out_row(0, "cfg-default", cpu_seconds=0.2, queue_seconds=None)]
    report = availability_report(train, held_out, even_gain())
    by_name = {item["feature"]: item for item in report["features"]}
    assert by_name["cpu_seconds"]["held_out"]["outside_train_range"] is None
    assert by_name["queue_seconds"]["held_out"]["outside_train_range"] is None


def test_range_table_lists_only_features_present_on_both_sides():
    train = [featured_row("none", index, queue_seconds=float(index + 1)) for index in range(3)]
    held_out = [held_out_row(index, "cfg-default", queue_seconds=9.0) for index in range(2)]
    table = range_table(availability_report(train, held_out, even_gain()))
    lines = table.strip().splitlines()
    assert lines[0].startswith("| Feature | Gain share | Train p5 / p50 / p95 |")
    (queue,) = [line for line in lines if line.startswith("| `queue_seconds`")]
    assert "| 1.1 / 2 / 2.9 |" in queue and "| 9 / 9 / 9 |" in queue
    assert queue.endswith("| 0.0% | 100.0% |")
    assert not [line for line in lines if "`cpu_pressure_max_avg10`" in line]
