"""Score held-out rows with the frozen M03 classifier and its M04 calibration.

Nothing here fits or refits: the booster is read from M03's saved file, and the temperature and
abstention threshold from M04's calibration.json. Both are accepted only when the files agree
with each other and with the frozen split and evaluation contract.

Two checks then prove the loaded pair is the one that was evaluated, step by step, so neither
can pass on an empty or partial record. `check_reproduces_validation` scores the validation
rows and must match every M03 validation prediction; the record must hold exactly the number of
steps M04 fitted on (its report's `fit.steps`). `check_reproduces_calibration` scores the test
rows M04 recorded, through the same temperature step as held-out scoring, and must match every
recorded prediction, abstention, calibrated confidence and class probability; the record must
hold exactly the steps the report counts (`scored_steps` plus `joint_only_steps_not_scored`:
M04 records a prediction for every test step but leaves joint-only ones out of its metrics),
each a test-split step.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from agentwatch_data.files import file_sha256

from agentwatch_model.calibrate import read_json, score_test
from agentwatch_model.calibration import CalibrationError
from agentwatch_model.calibration_report import Fit, Scored
from agentwatch_model.candidates import check_lightgbm
from agentwatch_model.features import FEATURES, feature_matrix
from agentwatch_model.held_out_scorers import ScoredStep
from agentwatch_model.metrics import is_number
from agentwatch_model.rule_baseline import split_identity
from agentwatch_model.train_candidates import (
    MODEL_FILES,
    PREPROCESSING,
    REPORT,
    VALIDATION_PREDICTIONS,
)

if TYPE_CHECKING:
    import lightgbm

CLASSIFIER_SCORER = "lightgbm-classifier-m03-m04"
CALIBRATION = "calibration.json"
CALIBRATION_REPORT = "report.json"
CALIBRATED_PREDICTIONS = "predictions.jsonl"
REPRODUCTION_TOLERANCE = 1e-9
ACCEPTED_CONTRACT = {"version": "eval-contract-v3", "hash": "1cab0cbe", "status": "accepted"}


class HeldOutModelError(ValueError):
    """The frozen classifier or calibration cannot be used for held-out scoring."""


@dataclass(frozen=True)
class FrozenClassifier:
    booster: "lightgbm.Booster"
    class_order: tuple[str, ...]
    temperature: float
    threshold: float
    # Steps M04 fitted on (its report's fit.steps) and recorded on test (scored_steps plus
    # joint_only_steps_not_scored); the reproduction checks require exactly these counts.
    validation_steps: int
    calibrated_test_steps: int
    identity: dict


def read_object(path: Path, what: str) -> dict:
    try:
        return read_json(path, what)
    except CalibrationError as error:
        raise HeldOutModelError(str(error)) from error


def digest(path: Path) -> str:
    try:
        return file_sha256(path)
    except OSError as error:
        raise HeldOutModelError(f"Cannot read {path}: {error}") from error


def field(container: object, *keys: str, where: str) -> object:
    """container[keys...], or a HeldOutModelError naming the file and field."""
    value = container
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise HeldOutModelError(f"{where} has no {'.'.join(keys)}")
        value = value[key]
    return value


def require_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise HeldOutModelError(f"{label} is {actual!r}, expected {expected!r}")


def positive_finite(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise HeldOutModelError(f"{label} must be a finite number above 0, got {value!r}")
    return float(value)


def positive_count(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HeldOutModelError(f"{label} must be a positive integer, got {value!r}")
    return value


def non_negative_count(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise HeldOutModelError(f"{label} must be a non-negative integer, got {value!r}")
    return value


def check_report(report: dict, split: Path, where: str) -> None:
    require_equal(field(report, "status", where=where), "evaluated", f"{where} status")
    require_equal(field(report, "split", where=where), split_identity(split), f"{where} split")
    require_equal(field(report, "contract", where=where), ACCEPTED_CONTRACT, f"{where} contract")


def check_calibration(calibration: dict, preprocessing: dict, where: str) -> tuple[float, float]:
    require_equal(field(calibration, "fitted", where=where), True, f"{where} fitted")
    require_equal(field(calibration, "status", where=where), "evaluated", f"{where} status")
    require_equal(
        field(calibration, "training_version", where=where),
        field(preprocessing, "training_version", where=PREPROCESSING),
        f"{where} training_version",
    )
    temperature = positive_finite(
        field(calibration, "temperature", where=where), f"{where} temperature"
    )
    threshold = positive_finite(
        field(calibration, "abstention_threshold", where=where), f"{where} abstention_threshold"
    )
    return temperature, threshold


def load_frozen_classifier(
    candidates_dir: Path, calibrated_dir: Path, split: Path
) -> FrozenClassifier:
    """The M03 classifier with its M04 temperature and threshold, after every identity check."""
    check_lightgbm()
    import lightgbm

    report_path = candidates_dir / REPORT
    preprocessing_path = candidates_dir / PREPROCESSING
    validation_path = candidates_dir / VALIDATION_PREDICTIONS
    classifier_path = candidates_dir / MODEL_FILES["classifier"]
    calibration_path = calibrated_dir / CALIBRATION
    calibration_report_path = calibrated_dir / CALIBRATION_REPORT
    report = read_object(report_path, "M03 report")
    check_report(report, split, str(report_path))
    preprocessing = read_object(preprocessing_path, "M03 preprocessing")
    calibration = read_object(calibration_path, "M04 calibration")
    calibration_report = read_object(calibration_report_path, "M04 report")
    where = str(calibration_report_path)
    validation_steps = positive_count(
        field(calibration_report, "fit", "steps", where=where), f"{where} fit.steps"
    )
    calibrated_test_steps = positive_count(
        field(calibration_report, "scored_steps", where=where), f"{where} scored_steps"
    ) + non_negative_count(
        field(calibration_report, "joint_only_steps_not_scored", where=where),
        f"{where} joint_only_steps_not_scored",
    )

    classifier_sha = digest(classifier_path)
    require_equal(
        classifier_sha,
        field(report, "model_files", "classifier", where=str(report_path)),
        f"{classifier_path} sha256 against {report_path} model_files.classifier",
    )
    require_equal(
        classifier_sha,
        field(calibration, "model_files", "classifier", "sha256", where=str(calibration_path)),
        f"{classifier_path} sha256 against {calibration_path} model_files.classifier.sha256",
    )
    preprocessing_sha = digest(preprocessing_path)
    require_equal(
        preprocessing_sha,
        field(report, "preprocessing", "sha256", where=str(report_path)),
        f"{preprocessing_path} sha256 against {report_path} preprocessing.sha256",
    )
    require_equal(
        digest(validation_path),
        field(report, "validation_predictions", "sha256", where=str(report_path)),
        f"{validation_path} sha256 against {report_path} validation_predictions.sha256",
    )
    where = str(preprocessing_path)
    require_equal(
        field(preprocessing, "features", where=where), list(FEATURES), f"{where} features"
    )
    class_order = field(preprocessing, "class_order", where=where)
    require_equal(
        field(preprocessing, "probability_columns", where=where),
        class_order,
        f"{where} probability_columns",
    )
    temperature, threshold = check_calibration(calibration, preprocessing, str(calibration_path))
    try:
        booster = lightgbm.Booster(model_file=str(classifier_path))
    except lightgbm.basic.LightGBMError as error:
        raise HeldOutModelError(f"Cannot load {classifier_path}: {error}") from error
    identity = {
        "training_version": preprocessing["training_version"],
        "calibration_version": calibration.get("version"),
        "contract": dict(ACCEPTED_CONTRACT),
        "split": split_identity(split),
        "classifier_sha256": classifier_sha,
        "preprocessing_sha256": preprocessing_sha,
        "calibration_sha256": digest(calibration_path),
        "temperature": temperature,
        "threshold": threshold,
        "validation_steps": validation_steps,
    }
    return FrozenClassifier(
        booster,
        tuple(class_order),
        temperature,
        threshold,
        validation_steps,
        calibrated_test_steps,
        identity,
    )


def class_probabilities(model: FrozenClassifier, rows: list[dict]) -> np.ndarray:
    """Probabilities in class_order; M03 saved the best iteration, so all iterations are used."""
    if not rows:
        return np.empty((0, len(model.class_order)))
    return np.asarray(model.booster.predict(feature_matrix(rows)))


def top_class(model: FrozenClassifier, scores: np.ndarray) -> tuple[str, dict[str, float]]:
    by_class = dict(zip(model.class_order, map(float, scores), strict=True))
    return max(by_class, key=by_class.__getitem__), by_class


def recorded_steps(path: Path) -> dict[tuple[str, str], dict]:
    """A predictions file's records by (run_id, step_id); one that cannot be read, records no
    step, or records a step twice is refused."""
    try:
        records = [json.loads(line) for line in path.read_text().splitlines()]
        recorded = {(record["run_id"], record["step_id"]): record for record in records}
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise HeldOutModelError(f"Cannot read {path}: {error!r}") from error
    if not records:
        raise HeldOutModelError(f"{path} records no steps; there is nothing to reproduce")
    if len(records) != len(recorded):
        raise HeldOutModelError(f"{path} records a (run_id, step_id) step more than once")
    return recorded


def close(value: float, saved: object) -> bool:
    return is_number(saved) and abs(value - saved) <= REPRODUCTION_TOLERANCE


def check_probabilities(
    path: Path, key: tuple[str, str], probabilities: dict[str, float], saved: object
) -> None:
    if not isinstance(saved, dict) or set(saved) != set(probabilities):
        raise HeldOutModelError(f"{path} probabilities do not match the class order at {key}")
    for name, value in probabilities.items():
        if not close(value, saved[name]):
            raise HeldOutModelError(
                f"{path} probability {name} differs from the loaded model at {key}"
            )


def check_reproduces_validation(
    model: FrozenClassifier, validation_rows: list[dict], candidates_dir: Path
) -> None:
    """The loaded booster must reproduce every one of M03's recorded validation predictions,
    and the record must hold the steps M04 fitted on."""
    path = candidates_dir / VALIDATION_PREDICTIONS
    recorded = recorded_steps(path)
    if len(recorded) != model.validation_steps:
        raise HeldOutModelError(
            f"{path} records {len(recorded)} steps, but M04 fitted on {model.validation_steps} "
            "(its report's fit.steps)"
        )
    keys = [(row["run_id"], row["step_id"]) for row in validation_rows]
    if set(keys) != set(recorded) or len(keys) != len(set(keys)):
        raise HeldOutModelError(
            f"{path} and the validation rows do not cover the same (run_id, step_id) steps"
        )
    for key, scores in zip(keys, class_probabilities(model, validation_rows), strict=True):
        record = recorded[key]
        label, probabilities = top_class(model, scores)
        if label != record.get("prediction"):
            raise HeldOutModelError(f"{path} prediction differs from the loaded model at {key}")
        check_probabilities(path, key, probabilities, record.get("probabilities"))


def calibrated_scores(model: FrozenClassifier, rows: list[dict]) -> tuple[list[dict], Scored]:
    """M03's top class and probabilities per row, and M04's temperature step applied to them
    by `calibrate.score_test`, the function M04 itself scored the test split with."""
    records = []
    for scores in class_probabilities(model, rows):
        label, probabilities = top_class(model, scores)
        records.append({"prediction": label, "probabilities": probabilities})
    fit = Fit(
        fitted=True,
        temperature=model.temperature,
        threshold=model.threshold,
        reason=None,
        validation_steps=0,
        usable_steps=0,
        temperature_at_bound=False,
        selection=None,
        early_stopping_on_fit_split=False,
    )
    return records, score_test(rows, records, model.class_order, fit)


def check_reproduces_calibration(
    model: FrozenClassifier, test_rows: list[dict], calibrated_dir: Path
) -> None:
    """The loaded pair must reproduce every one of M04's recorded test predictions: label,
    abstention, calibrated confidence and each calibrated class probability.

    Every recorded step must be a test-split row, and the record must hold exactly the steps
    the M04 report counts: `scored_steps` plus `joint_only_steps_not_scored`, as M04 records a
    prediction for each joint-only step too but leaves it out of its metrics.
    """
    path = calibrated_dir / CALIBRATED_PREDICTIONS
    recorded = recorded_steps(path)
    if len(recorded) != model.calibrated_test_steps:
        raise HeldOutModelError(
            f"{path} records {len(recorded)} steps, but the M04 report's scored_steps plus "
            f"joint_only_steps_not_scored is {model.calibrated_test_steps}"
        )
    test_keys = {(row["run_id"], row["step_id"]) for row in test_rows}
    outside = sorted(set(recorded) - test_keys)
    if outside:
        raise HeldOutModelError(
            f"{path} records {len(outside)} steps that are not test-split rows, first {outside[0]}"
        )
    rows = [row for row in test_rows if (row["run_id"], row["step_id"]) in recorded]
    records, scored = calibrated_scores(model, rows)
    for row, record, confidence, abstained, probabilities in zip(
        rows, records, scored.confidences, scored.abstained, scored.probabilities, strict=True
    ):
        key = (row["run_id"], row["step_id"])
        saved = recorded[key]
        if record["prediction"] != saved.get("prediction"):
            raise HeldOutModelError(f"{path} prediction differs from the loaded model at {key}")
        if saved.get("abstained") is not abstained:
            raise HeldOutModelError(f"{path} abstained differs from the loaded model at {key}")
        if not close(confidence, saved.get("confidence")):
            raise HeldOutModelError(f"{path} confidence differs from the loaded model at {key}")
        check_probabilities(path, key, probabilities, saved.get("probabilities"))


def frozen_scores(model: FrozenClassifier, rows: list[dict]) -> list[ScoredStep]:
    """Top class, calibrated confidence and abstention flag per row; missing features are NaN."""
    records, scored = calibrated_scores(model, rows)
    return [
        ScoredStep(CLASSIFIER_SCORER, record["prediction"], confidence, abstained)
        for record, confidence, abstained in zip(
            records, scored.confidences, scored.abstained, strict=True
        )
    ]
