import math

import pytest
from agentwatch_data.schema import feature_columns
from agentwatch_model.features import FEATURES, FeatureError, feature_matrix, preprocessing_contract
from training_fixtures import featured_row


def test_features_are_exactly_the_schema_feature_columns_in_order():
    assert FEATURES == tuple(feature_columns())
    assert "label_resource" not in FEATURES and "condition" not in FEATURES


def test_missing_readings_stay_nan_and_booleans_become_numbers():
    row = featured_row("none", 0) | {"cpu_pressure_max_avg10": None, "cgroup_available": True}
    [values] = feature_matrix([row]).tolist()
    by_name = dict(zip(FEATURES, values, strict=True))
    assert math.isnan(by_name["cpu_pressure_max_avg10"])
    assert by_name["cgroup_available"] == 1.0 and by_name["cpu_pressure_available"] == 0.0
    assert by_name["queue_fraction"] == pytest.approx(0.03)


def test_a_row_missing_a_feature_column_is_refused():
    row = {k: v for k, v in featured_row("none", 0).items() if k != "queue_seconds"}
    with pytest.raises(FeatureError, match="queue_seconds"):
        feature_matrix([row])


def test_a_non_numeric_feature_is_refused():
    with pytest.raises(FeatureError, match="queue_fraction"):
        feature_matrix([featured_row("none", 0, queue_fraction="high")])


def test_preprocessing_contract_records_what_serving_must_reproduce():
    contract = preprocessing_contract(
        ("none", "cpu_sandbox"), "training-v1+abc", {"dataset_version": "d"}
    )
    assert contract["features"] == list(FEATURES)
    assert contract["class_order"] == ["none", "cpu_sandbox"]
    assert contract["missing_values"].startswith("NaN")
    assert contract["booleans"] == "true is 1.0, false is 0.0"
    assert contract["training_version"] == "training-v1+abc" and contract["dataset_version"] == "d"
