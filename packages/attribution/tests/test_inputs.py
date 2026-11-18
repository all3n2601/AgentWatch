import math

import pytest
from agentwatch_attribution.inputs import MODEL_INPUTS, FeatureError, feature_matrix

CONTRACT_ORDER = (
    "queue_seconds",
    "execution_seconds",
    "duration_seconds",
    "cpu_seconds",
    "queue_fraction",
    "workload_iterations",
    "tests_per_step",
    "worker_capacity",
    "cpu_pressure_max_avg10",
    "cpu_pressure_available",
    "cgroup_throttled_usec_delta",
    "cgroup_available",
    "telemetry_sample_count",
    "telemetry_coverage",
)


def complete_row() -> dict[str, object]:
    return {name: float(index + 1) for index, name in enumerate(MODEL_INPUTS)}


def test_none_is_nan_and_zero_stays_zero() -> None:
    row = complete_row() | {"cpu_seconds": None, "queue_seconds": 0.0}
    [values] = feature_matrix([row]).tolist()
    by_name = dict(zip(MODEL_INPUTS, values, strict=True))
    assert math.isnan(by_name["cpu_seconds"])
    assert by_name["queue_seconds"] == 0.0


def test_booleans_encode_as_one_and_zero() -> None:
    row = complete_row() | {"cpu_pressure_available": True, "cgroup_available": False}
    [values] = feature_matrix([row]).tolist()
    by_name = dict(zip(MODEL_INPUTS, values, strict=True))
    assert by_name["cpu_pressure_available"] == 1.0
    assert by_name["cgroup_available"] == 0.0


def test_missing_column_is_refused_by_name() -> None:
    row = {k: v for k, v in complete_row().items() if k != "telemetry_coverage"}
    with pytest.raises(FeatureError, match="telemetry_coverage"):
        feature_matrix([row])


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan, "1.5", [1.0]])
def test_non_finite_value_is_refused(value: object) -> None:
    with pytest.raises(FeatureError, match="queue_seconds"):
        feature_matrix([complete_row() | {"queue_seconds": value}])


def test_column_order_is_the_contract_order() -> None:
    assert MODEL_INPUTS == CONTRACT_ORDER
    [values] = feature_matrix([complete_row()]).tolist()
    assert values == [float(index + 1) for index in range(len(CONTRACT_ORDER))]
