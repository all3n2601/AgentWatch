"""One step's attribution record, built from synthetic feature rows and the synthetic model pair.

Every row and model here is a made-up test fixture, never recorded-run evidence.
"""

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import get_args
from uuid import UUID

import pytest
from agentwatch_attribution.inputs import MODEL_INPUTS
from agentwatch_attribution.rules import Rules, load_rules
from agentwatch_attribution.served_model import ModelPrediction, ServedModel
from agentwatch_features import FEATURE_NAMES, FeatureRow
from agentwatch_features.live import TelemetryStatus as FeatureTelemetryStatus
from agentwatch_sdk.attribution import AttributionRecord, ModelOutcome, StepState, TelemetryStatus
from agentwatch_worker.attribution import (
    NOT_LIVE_INPUTS,
    STEP_STATES,
    build_attribution,
    model_input_row,
)
from agentwatch_worker.attribution_model import ModelService, ModelStatus
from pair_fixtures import CALIBRATION_VERSION, TRAINING_VERSION, write_synthetic_pair

RUN = UUID("00000000-0000-4000-8000-0000000000b3")
CANDIDATES = "AGENTWATCH_MODEL_CANDIDATES_DIR"
CALIBRATION_DIR = "AGENTWATCH_MODEL_CALIBRATION_DIR"
ALLOW_UNPROMOTED = "AGENTWATCH_ALLOW_UNPROMOTED_MODEL"
OUTCOMES = {"predicted", "abstained", "not_scored_step_state", "unavailable"}
MODEL_KEYS = {
    "state",
    "reason",
    "training_version",
    "calibration_version",
    "not_promoted",
    "outcome",
    "predicted_class",
    "confidence",
    "abstention_threshold",
    "abstained",
    "calibrated_probabilities",
    "excess_fraction_estimate",
}
PREDICTION_KEYS = (
    "predicted_class",
    "confidence",
    "abstention_threshold",
    "abstained",
    "calibrated_probabilities",
    "excess_fraction_estimate",
)
# Well above the synthetic pair's training threshold for cpu_sandbox (50%).
HIGH_PRESSURE_AVG10 = 90.0
# Below every rules-v1 threshold: PSI 20%, queue fraction 0.5, throttled fraction 0.1.
QUIET_PRESSURE_AVG10 = 5.0


def feature_row(
    *,
    queue_seconds: float | None = 0.5,
    execution_seconds: float | None = 1.5,
    cpu_seconds: float | None = 1.0,
    pressure: float | None = QUIET_PRESSURE_AVG10,
    throttled_usec: float | None = 1000.0,
    coverage: float | None = 1.0,
    executed: bool = True,
) -> FeatureRow:
    """A consistent step-features-v1 row; resource readings imply present telemetry."""
    duration = (
        queue_seconds + execution_seconds
        if queue_seconds is not None and execution_seconds is not None
        else None
    )
    present = executed and (pressure is not None or throttled_usec is not None)
    sample_count = 3 if present else 0
    if not executed:
        status = "missing"
    else:
        status = "present" if present else "empty"
    if not present:
        coverage = None  # coverage needs a sampler stream for the step's target
    timings = {
        "queue_seconds": queue_seconds,
        "execution_seconds": execution_seconds,
        "duration_seconds": duration,
        "cpu_seconds": cpu_seconds,
    }
    return FeatureRow.model_validate(
        {
            "contract_version": "live-contracts-v1",
            "record_type": "feature_row",
            "run_id": str(RUN),
            "task_id": "fixture-task-v1",
            "step_id": "tool-1",
            "step_type": "tool",
            "feature_version": "step-features-v1",
            "source_event_ids": ["00000000-0000-4000-8000-000000000e01"],
            "source_sample_ids": ["00000000-0000-4000-8000-000000000501"] if present else [],
            "window": {
                "clock_id": "clock-1",
                "started_tick_ns": 1_000_000_000 if executed else None,
                "finished_tick_ns": 3_000_000_000 if executed else None,
            },
            "timings": timings,
            "telemetry": {
                "status": status,
                "sample_count": sample_count,
                "coverage": coverage,
                "declared_interval_seconds": 0.5 if executed else None,
                "observed_interval_seconds": 0.5 if present else None,
            },
            "features": timings
            | {
                "queue_fraction": queue_seconds / duration if duration else None,
                "cpu_pressure_max_avg10": pressure if present else None,
                "cpu_pressure_available": present and pressure is not None,
                "cgroup_throttled_usec_delta": throttled_usec if present else None,
                "cgroup_available": present and throttled_usec is not None,
                "telemetry_sample_count": sample_count,
                "telemetry_coverage": coverage,
            },
        }
    )


def never_executed_row() -> FeatureRow:
    """A step cancelled before it started: no window, no execution, missing telemetry."""
    return feature_row(
        queue_seconds=None,
        execution_seconds=None,
        cpu_seconds=None,
        pressure=None,
        throttled_usec=None,
        executed=False,
    )


def service_for(
    root: Path,
    *,
    allow_unpromoted: bool = True,
    blocks_promotion: bool = True,
    abstention_threshold: float = 0.6,
) -> ModelService:
    candidates, calibration = write_synthetic_pair(
        root, abstention_threshold=abstention_threshold, blocks_promotion=blocks_promotion
    )
    environ = {CANDIDATES: str(candidates), CALIBRATION_DIR: str(calibration)}
    if allow_unpromoted:
        environ[ALLOW_UNPROMOTED] = "1"
    return ModelService.from_environ(environ)


@pytest.fixture
def rules() -> Rules:
    return load_rules()


@pytest.fixture
def unpromoted(tmp_path: Path) -> ModelService:
    service = service_for(tmp_path, blocks_promotion=True)
    assert service.status.state == "ready_not_promoted"
    return service


def record_for(
    row: FeatureRow, step_state: str, rules: Rules, service: ModelService
) -> dict[str, object]:
    record = build_attribution(row, step_state, rules, service.status, service.served)
    # Everything stored and returned must survive strict JSON: no NaN or infinity anywhere.
    json.dumps(record, allow_nan=False)
    return record


def model_of(record: dict[str, object]) -> dict[str, object]:
    model = record["model"]
    assert isinstance(model, dict)
    assert set(model) == MODEL_KEYS
    assert model["outcome"] in OUTCOMES
    return model


def test_model_input_row_has_every_input_and_passes_non_live_inputs_as_missing() -> None:
    row = feature_row()
    inputs = model_input_row(row)
    assert set(inputs) == set(MODEL_INPUTS)
    assert NOT_LIVE_INPUTS == ("workload_iterations", "tests_per_step", "worker_capacity")
    assert all(inputs[name] is None for name in NOT_LIVE_INPUTS)
    assert {name: inputs[name] for name in FEATURE_NAMES} == row.features.model_dump()


def test_completed_step_gets_rules_and_model(rules: Rules, unpromoted: ModelService) -> None:
    row = feature_row(pressure=HIGH_PRESSURE_AVG10)

    record = record_for(row, "completed", rules, unpromoted)

    assert record["run_id"] == str(RUN)
    assert record["step_id"] == "tool-1"
    assert record["step_state"] == "completed"
    assert record["feature_version"] == "step-features-v1"
    assert record["feature_row"] == row.model_dump(mode="json")
    assert record["telemetry_status"] == "present"
    assert record["missing_inputs"] == list(NOT_LIVE_INPUTS)
    assert record["rules"] == {"version": rules.version, "verdict": "cpu_sandbox"}
    model = model_of(record)
    assert model["state"] == "ready_not_promoted"
    assert model["reason"] is None
    assert model["training_version"] == TRAINING_VERSION
    assert model["calibration_version"] == CALIBRATION_VERSION
    assert model["outcome"] == "predicted"
    assert model["abstained"] is False
    assert model["predicted_class"] == "cpu_sandbox"
    probabilities = model["calibrated_probabilities"]
    assert isinstance(probabilities, dict)
    assert list(probabilities) == ["none", "cpu_sandbox", "inference", "retrieval"]
    assert math.isclose(sum(probabilities.values()), 1.0)
    assert model["confidence"] == max(probabilities.values())
    assert model["abstention_threshold"] == 0.6
    assert unpromoted.served is not None
    [expected] = unpromoted.served.predict([model_input_row(row)])
    assert model["excess_fraction_estimate"] == expected.excess_fraction


@pytest.mark.parametrize("step_state", ["failed", "cancelled"])
def test_failed_and_cancelled_steps_are_not_scored_by_the_model_but_keep_rules(
    step_state: str, rules: Rules, unpromoted: ModelService, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(self: ServedModel, rows: object) -> list[ModelPrediction]:
        raise AssertionError("the model must not score a step that did not complete")

    monkeypatch.setattr(ServedModel, "predict", refuse)

    record = record_for(feature_row(queue_seconds=3.0), step_state, rules, unpromoted)

    assert record["step_state"] == step_state
    assert record["rules"] == {"version": rules.version, "verdict": "cpu_sandbox"}
    model = model_of(record)
    assert model["outcome"] == "not_scored_step_state"
    assert model["state"] == "ready_not_promoted"
    assert model["training_version"] == TRAINING_VERSION
    assert all(model[key] is None for key in PREDICTION_KEYS)


def test_step_that_never_executed_still_gets_both_and_lists_every_missing_input(
    rules: Rules, unpromoted: ModelService
) -> None:
    row = never_executed_row()
    assert row.window.started_tick_ns is None

    record = record_for(row, "cancelled", rules, unpromoted)

    assert record["telemetry_status"] == "missing"
    assert record["missing_inputs"] == [
        "queue_seconds",
        "execution_seconds",
        "duration_seconds",
        "cpu_seconds",
        "queue_fraction",
        "workload_iterations",
        "tests_per_step",
        "worker_capacity",
        "cpu_pressure_max_avg10",
        "cgroup_throttled_usec_delta",
        "telemetry_coverage",
    ]
    assert record["rules"] == {"version": rules.version, "verdict": "none"}
    assert model_of(record)["outcome"] == "not_scored_step_state"


@pytest.mark.parametrize("step_state", ["running", "queued", "", "Completed"])
def test_a_non_terminal_or_unknown_step_state_is_refused(
    step_state: str, rules: Rules, unpromoted: ModelService
) -> None:
    with pytest.raises(ValueError, match="step_state"):
        build_attribution(feature_row(), step_state, rules, unpromoted.status, unpromoted.served)


def test_measured_zero_is_not_listed_missing(rules: Rules, unpromoted: ModelService) -> None:
    row = feature_row(execution_seconds=0.0, cpu_seconds=0.0, pressure=0.0, throttled_usec=None)
    assert row.features.execution_seconds == 0.0
    assert row.features.cgroup_available is False

    record = record_for(row, "completed", rules, unpromoted)

    missing = record["missing_inputs"]
    assert isinstance(missing, list)
    assert "execution_seconds" not in missing
    assert "cpu_seconds" not in missing
    assert "cpu_pressure_max_avg10" not in missing
    assert "cgroup_available" not in missing
    assert "cpu_pressure_available" not in missing
    # The delta was not measured: that absence is listed, the False flag beside it is a value.
    assert missing == [*NOT_LIVE_INPUTS, "cgroup_throttled_usec_delta"]
    inputs = model_input_row(row)
    assert inputs["execution_seconds"] == 0.0
    assert inputs["cgroup_available"] is False


def test_abstention_is_not_none(rules: Rules, tmp_path: Path) -> None:
    # A threshold of 1.0 is above any calibrated confidence the four-class pair produces.
    service = service_for(tmp_path, abstention_threshold=1.0)

    record = record_for(feature_row(), "completed", rules, service)

    model = model_of(record)
    assert model["outcome"] == "abstained"
    assert model["abstained"] is True
    assert model["predicted_class"] is None
    assert model["predicted_class"] != "none"
    assert isinstance(model["confidence"], float)
    assert model["confidence"] < 1.0
    assert model["abstention_threshold"] == 1.0
    assert isinstance(model["calibrated_probabilities"], dict)
    assert isinstance(model["excess_fraction_estimate"], float)


def test_model_unavailable_keeps_the_rules_verdict_and_names_the_reason(
    rules: Rules, tmp_path: Path
) -> None:
    refused = service_for(tmp_path / "refused", allow_unpromoted=False, blocks_promotion=True)
    assert refused.status.state == "not_promoted_refused"
    services = {
        "not_configured": ModelService.from_environ({}),
        "not_promoted_refused": refused,
        "load_failed": ModelService(ModelStatus("load_failed", reason="hash_mismatch")),
    }
    expected_reasons = {
        "not_configured": "not_configured",
        "not_promoted_refused": "not_promoted_refused",
        "load_failed": "hash_mismatch",
    }
    row = feature_row(queue_seconds=3.0)

    for state, service in services.items():
        record = record_for(row, "completed", rules, service)
        assert record["rules"] == {"version": rules.version, "verdict": "cpu_sandbox"}
        model = model_of(record)
        assert model["state"] == state
        assert model["outcome"] == "unavailable"
        assert model["reason"] == expected_reasons[state]
        assert all(model[key] is None for key in PREDICTION_KEYS)

    refused_model = model_of(record_for(row, "completed", rules, refused))
    assert refused_model["training_version"] == TRAINING_VERSION
    assert refused_model["calibration_version"] == CALIBRATION_VERSION
    assert refused_model["not_promoted"] is True
    unconfigured = model_of(record_for(row, "completed", rules, services["not_configured"]))
    assert unconfigured["training_version"] is None
    assert unconfigured["calibration_version"] is None
    assert unconfigured["not_promoted"] is None


def test_unpromoted_model_is_labelled(
    rules: Rules, unpromoted: ModelService, tmp_path: Path
) -> None:
    promoted = service_for(tmp_path / "promoted", blocks_promotion=False)
    assert promoted.status.state == "ready_promoted"

    labelled = model_of(record_for(feature_row(), "completed", rules, unpromoted))
    unlabelled = model_of(record_for(feature_row(), "completed", rules, promoted))

    assert labelled["state"] == "ready_not_promoted"
    assert labelled["not_promoted"] is True
    assert unlabelled["state"] == "ready_promoted"
    assert unlabelled["not_promoted"] is False


def test_excess_fraction_is_named_an_estimate_and_kept_apart_from_queue_seconds(
    rules: Rules, unpromoted: ModelService
) -> None:
    row = feature_row(queue_seconds=1.25)

    record = record_for(row, "completed", rules, unpromoted)

    model = model_of(record)
    assert "excess_fraction" not in model
    estimate = model["excess_fraction_estimate"]
    assert isinstance(estimate, float)
    assert 0.0 <= estimate <= 1.0
    assert "queue_seconds" not in model
    feature_row_json = record["feature_row"]
    assert isinstance(feature_row_json, dict)
    assert feature_row_json["features"]["queue_seconds"] == 1.25
    assert "excess_fraction_estimate" not in feature_row_json["features"]


def test_rules_verdict_is_cpu_sandbox_above_the_queue_fraction_threshold(
    rules: Rules, unpromoted: ModelService
) -> None:
    above = feature_row(queue_seconds=3.0, execution_seconds=1.0)
    assert above.features.queue_fraction is not None
    assert above.features.queue_fraction > rules.min_queue_fraction

    record = record_for(above, "completed", rules, unpromoted)

    assert record["rules"] == {"version": rules.version, "verdict": "cpu_sandbox"}


def test_rules_verdict_is_none_below_every_threshold(
    rules: Rules, unpromoted: ModelService
) -> None:
    quiet = feature_row(queue_seconds=0.5, execution_seconds=1.5)
    assert quiet.features.queue_fraction is not None
    assert quiet.features.queue_fraction < rules.min_queue_fraction

    record = record_for(quiet, "completed", rules, unpromoted)

    assert record["rules"] == {"version": rules.version, "verdict": "none"}


@pytest.mark.parametrize("field", ["confidence", "excess_fraction", "calibrated_probabilities"])
@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_non_finite_prediction_is_unavailable(
    field: str,
    bad: float,
    rules: Rules,
    unpromoted: ModelService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_predict = ServedModel.predict

    def corrupted(self: ServedModel, rows: list[dict[str, object]]) -> list[ModelPrediction]:
        [prediction] = real_predict(self, rows)
        if field == "calibrated_probabilities":
            probabilities = dict(prediction.calibrated_probabilities, inference=bad)
            return [replace(prediction, calibrated_probabilities=probabilities)]
        return [replace(prediction, **{field: bad})]

    monkeypatch.setattr(ServedModel, "predict", corrupted)

    record = record_for(feature_row(queue_seconds=3.0), "completed", rules, unpromoted)

    assert record["rules"] == {"version": rules.version, "verdict": "cpu_sandbox"}
    model = model_of(record)
    assert model["outcome"] == "unavailable"
    assert model["reason"] == "non_finite_prediction"
    assert model["state"] == "ready_not_promoted"
    assert model["training_version"] == TRAINING_VERSION
    assert all(model[key] is None for key in PREDICTION_KEYS)


def test_a_record_of_every_outcome_matches_the_published_record_type(
    rules: Rules, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The typed contract mirrors exactly what is built: nothing dropped, nothing coerced."""
    unpromoted = service_for(tmp_path / "unpromoted")
    abstaining = service_for(tmp_path / "abstaining", abstention_threshold=1.0)
    refused = service_for(tmp_path / "refused", allow_unpromoted=False)
    load_failed = ModelService(ModelStatus("load_failed", reason="hash_mismatch"))
    row = feature_row(pressure=HIGH_PRESSURE_AVG10)
    records = {
        "predicted": record_for(row, "completed", rules, unpromoted),
        "abstained": record_for(row, "completed", rules, abstaining),
        "not_scored_step_state": record_for(row, "failed", rules, unpromoted),
        "never_executed": record_for(never_executed_row(), "cancelled", rules, unpromoted),
        "not_configured": record_for(row, "completed", rules, ModelService.from_environ({})),
        "not_promoted_refused": record_for(row, "completed", rules, refused),
        "load_failed": record_for(row, "completed", rules, load_failed),
    }
    real_predict = ServedModel.predict

    def non_finite(self: ServedModel, rows: list[dict[str, object]]) -> list[ModelPrediction]:
        [prediction] = real_predict(self, rows)
        return [replace(prediction, confidence=math.nan)]

    monkeypatch.setattr(ServedModel, "predict", non_finite)
    records["non_finite_prediction"] = record_for(row, "completed", rules, unpromoted)

    outcomes = {name: model_of(record)["outcome"] for name, record in records.items()}
    assert set(outcomes.values()) == OUTCOMES
    unavailable = {"not_configured", "not_promoted_refused", "load_failed", "non_finite_prediction"}
    assert {name: model_of(records[name])["reason"] for name in unavailable} == {
        "not_configured": "not_configured",
        "not_promoted_refused": "not_promoted_refused",
        "load_failed": "hash_mismatch",
        "non_finite_prediction": "non_finite_prediction",
    }
    for name, record in records.items():
        typed = AttributionRecord.model_validate(record)
        assert typed.model_dump(mode="json") == record, name
        # The stored form is JSON: it validates the same way after a round trip.
        assert AttributionRecord.model_validate_json(json.dumps(record)) == typed, name


def test_record_literals_match_the_values_the_builder_and_features_use() -> None:
    assert set(get_args(StepState)) == STEP_STATES
    assert set(get_args(ModelOutcome)) == OUTCOMES
    assert get_args(TelemetryStatus) == get_args(FeatureTelemetryStatus)
