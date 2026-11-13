"""Feature availability on synthetic rows; none of it is recorded evidence."""

import json
import math

from agentwatch_model.features import FEATURES
from agentwatch_model.held_out_features import feature_availability
from training_fixtures import featured_row


def rows(count, configuration="cfg-default", **overrides):
    return [
        featured_row("none", index, configuration=configuration) | overrides
        for index in range(count)
    ]


def test_every_feature_is_described_in_feature_order():
    result = feature_availability(rows(2), rows(3))

    assert list(result) == list(FEATURES)
    assert set(result[FEATURES[0]]) == {"train", "held_out", "held_out_by_configuration"}


def test_a_feature_missing_only_in_held_out_is_shown_against_train():
    result = feature_availability(rows(4, cpu_seconds=None), rows(5))["cpu_seconds"]

    assert result["train"] == {
        "rows": 5,
        "missing_fraction": 0.0,
        "distinct_values": 1,
        "most_common_fraction": 1.0,
    }
    assert result["held_out"] == {
        "rows": 4,
        "missing_fraction": 1.0,
        "distinct_values": 0,
        "most_common_fraction": None,
    }


def test_a_constant_feature_has_one_distinct_value():
    held_out = rows(3)
    held_out[0]["execution_seconds"] = 9.0

    result = feature_availability(held_out, rows(2))

    assert result["worker_capacity"]["held_out"]["distinct_values"] == 1
    assert result["worker_capacity"]["held_out"]["most_common_fraction"] == 1.0
    execution = result["execution_seconds"]["held_out"]
    assert execution["distinct_values"] == 2
    assert execution["most_common_fraction"] == 2 / 3


def test_null_and_nan_are_both_missing():
    held_out = rows(2, queue_fraction=None)
    held_out[1]["queue_fraction"] = math.nan

    result = feature_availability(held_out, rows(1))["queue_fraction"]["held_out"]

    assert result["missing_fraction"] == 1.0
    assert result["distinct_values"] == 0 and result["most_common_fraction"] is None


def test_a_partly_missing_feature_counts_shares_of_present_values():
    held_out = rows(4)
    held_out[0]["cpu_seconds"] = None
    held_out[1]["cpu_seconds"] = 0.0  # a measured zero is present, not missing

    result = feature_availability(held_out, rows(1))["cpu_seconds"]["held_out"]

    assert result["missing_fraction"] == 0.25
    assert result["distinct_values"] == 2
    assert result["most_common_fraction"] == 2 / 3


def test_held_out_rows_are_split_by_configuration_in_sorted_order():
    held_out = rows(1, "cfg-hardware", cpu_seconds=None) + rows(3, "cfg-default")

    result = feature_availability(held_out, rows(2))["cpu_seconds"]["held_out_by_configuration"]

    assert list(result) == ["cfg-default", "cfg-hardware"]
    assert result["cfg-default"]["rows"] == 3 and result["cfg-default"]["missing_fraction"] == 0.0
    assert result["cfg-hardware"]["rows"] == 1 and result["cfg-hardware"]["missing_fraction"] == 1.0


def test_a_group_without_rows_has_no_fractions():
    result = feature_availability(rows(1), [])["cpu_seconds"]

    assert result["train"] == {
        "rows": 0,
        "missing_fraction": None,
        "distinct_values": 0,
        "most_common_fraction": None,
    }


def test_the_result_is_json_safe_and_deterministic():
    held_out = rows(2, queue_fraction=math.nan) + rows(1, "cfg-hardware")

    result = feature_availability(held_out, rows(3))

    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert feature_availability(held_out, rows(3)) == result
