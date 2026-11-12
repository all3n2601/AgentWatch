"""Frozen M03/M04 pair loading on synthetic fixtures; nothing here is recorded evidence."""

import json

import pytest
from frozen_model_fixtures import CALIBRATION, TEMPERATURE, THRESHOLD, write_frozen_pair
from training_fixtures import dataset, require_lightgbm

require_lightgbm()

from agentwatch_data.files import file_sha256
from agentwatch_model.candidates import predict
from agentwatch_model.features import feature_matrix
from agentwatch_model.held_out_model import (
    CLASSIFIER_SCORER,
    REPRODUCTION_TOLERANCE,
    FrozenClassifier,
    HeldOutModelError,
    check_reproduces_validation,
    frozen_scores,
    load_frozen_classifier,
)
from agentwatch_model.train_candidates import PREPROCESSING, REPORT, VALIDATION_PREDICTIONS


@pytest.fixture
def split(tmp_path):
    path = tmp_path / "split-test.json"
    path.write_text('{"synthetic": true}\n')
    return path


@pytest.fixture
def pair(tmp_path, split):
    return write_frozen_pair(tmp_path, split)


def edit(path, change):
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))


def rebind_preprocessing(pair):
    """Keep report.json's hash for preprocessing.json true after a test edits the file."""
    sha = file_sha256(pair.candidates_dir / PREPROCESSING)
    edit(pair.candidates_dir / REPORT, lambda r: r["preprocessing"].update(sha256=sha))


def test_a_matching_pair_loads_and_records_its_identity(pair, split):
    model = load_frozen_classifier(pair.candidates_dir, pair.calibrated_dir, split)

    assert isinstance(model, FrozenClassifier)
    assert model.class_order == pair.candidate.class_order
    assert model.temperature == TEMPERATURE and model.threshold == THRESHOLD
    identity = model.identity
    assert identity["training_version"] == pair.candidate.training_version
    assert identity["calibration_version"] == "calibration-v1+synthetic"
    assert identity["contract"] == {
        "version": "eval-contract-v3",
        "hash": "1cab0cbe",
        "status": "accepted",
    }
    assert identity["split"] == {"file": split.name, "sha256": file_sha256(split)}
    assert identity["classifier_sha256"] == file_sha256(pair.candidates_dir / "classifier.txt")
    assert identity["preprocessing_sha256"] == file_sha256(pair.candidates_dir / PREPROCESSING)
    assert identity["calibration_sha256"] == file_sha256(pair.calibrated_dir / CALIBRATION)
    assert identity["temperature"] == TEMPERATURE and identity["threshold"] == THRESHOLD
    check_reproduces_validation(model, pair.validation_rows, pair.candidates_dir)


def test_loaded_scores_equal_the_trained_candidate(pair, split):
    model = load_frozen_classifier(pair.candidates_dir, pair.calibrated_dir, split)
    rows = dataset(6, "held")

    expected = predict(pair.candidate, rows)
    actual = model.booster.predict(feature_matrix(rows))

    assert len(actual) == len(expected) == len(rows)
    for scores, trained in zip(actual, expected, strict=True):
        for name, value in zip(model.class_order, scores, strict=True):
            assert abs(value - trained.probabilities[name]) <= REPRODUCTION_TOLERANCE
    steps = frozen_scores(model, rows)
    assert [s.label for s in steps] == [p.label for p in expected]
    assert all(s.scorer == CLASSIFIER_SCORER for s in steps)
    assert all(s.confidence is not None and 0 < s.confidence <= 1 for s in steps)


def test_missing_feature_values_score_as_nan_not_zero(pair, split):
    model = load_frozen_classifier(pair.candidates_dir, pair.calibrated_dir, split)
    rows = dataset(2, "held")
    rows[0]["cpu_seconds"] = None

    assert len(frozen_scores(model, rows)) == len(rows)


@pytest.mark.parametrize(("threshold", "abstains"), [(0.99, True), (1e-6, False)])
def test_abstains_below_the_calibrated_threshold(pair, split, threshold, abstains):
    # A high temperature flattens the calibrated confidences, so 0.99 is out of reach for every
    # step.
    edit(
        pair.calibrated_dir / CALIBRATION,
        lambda c: c.update(abstention_threshold=threshold, temperature=50.0),
    )
    model = load_frozen_classifier(pair.candidates_dir, pair.calibrated_dir, split)

    steps = frozen_scores(model, dataset(6, "held"))

    assert steps
    assert all(s.abstained is abstains for s in steps)


def split_other(pair, split):
    other = split.parent / "split-other.json"
    other.write_text('{"synthetic": "other"}\n')
    return other


def contract_proposed(pair, split):
    edit(pair.candidates_dir / REPORT, lambda r: r["contract"].update(status="proposed"))


def classifier_hash_differs(pair, split):
    edit(
        pair.calibrated_dir / CALIBRATION,
        lambda c: c["model_files"]["classifier"].update(sha256="0" * 64),
    )


def not_fitted(pair, split):
    edit(pair.calibrated_dir / CALIBRATION, lambda c: c.update(fitted=False))


def features_reordered(pair, split):
    edit(pair.candidates_dir / PREPROCESSING, lambda p: p["features"].reverse())
    rebind_preprocessing(pair)


def temperature_not_finite(pair, split):
    path = pair.calibrated_dir / CALIBRATION
    path.write_text(path.read_text().replace(f"{TEMPERATURE}", "NaN"))


def training_version_differs(pair, split):
    edit(
        pair.calibrated_dir / CALIBRATION, lambda c: c.update(training_version="training-v1+other")
    )


def preprocessing_edited_without_rebinding(pair, split):
    edit(pair.candidates_dir / PREPROCESSING, lambda p: p.update(note="changed"))


def validation_predictions_edited(pair, split):
    path = pair.candidates_dir / VALIDATION_PREDICTIONS
    path.write_text(path.read_text() + "\n")


def report_not_evaluated(pair, split):
    edit(pair.candidates_dir / REPORT, lambda r: r.update(status="proposed"))


def classifier_differs_from_report(pair, split):
    edit(
        pair.candidates_dir / REPORT,
        lambda r: r["model_files"].update(classifier="0" * 64),
    )


def probability_columns_reordered(pair, split):
    edit(pair.candidates_dir / PREPROCESSING, lambda p: p["probability_columns"].reverse())
    rebind_preprocessing(pair)


def calibration_not_evaluated(pair, split):
    edit(pair.calibrated_dir / CALIBRATION, lambda c: c.update(status="proposed"))


def threshold_zero(pair, split):
    edit(pair.calibrated_dir / CALIBRATION, lambda c: c.update(abstention_threshold=0))


def threshold_not_finite(pair, split):
    path = pair.calibrated_dir / CALIBRATION
    path.write_text(path.read_text().replace(f"{THRESHOLD}", "NaN"))


REFUSALS = [
    (split_other, "split"),
    (contract_proposed, "contract"),
    (classifier_hash_differs, "model_files.classifier.sha256"),
    (not_fitted, "fitted"),
    (features_reordered, "features"),
    (temperature_not_finite, "temperature"),
    (training_version_differs, "training_version"),
    (preprocessing_edited_without_rebinding, "preprocessing.sha256"),
    (validation_predictions_edited, "validation_predictions.sha256"),
    (report_not_evaluated, r"report\.json status"),
    (classifier_differs_from_report, r"report\.json model_files\.classifier"),
    (probability_columns_reordered, "probability_columns"),
    (calibration_not_evaluated, r"calibration\.json status"),
    (threshold_zero, "abstention_threshold"),
    (threshold_not_finite, "abstention_threshold"),
]


@pytest.mark.parametrize(("break_it", "field"), REFUSALS, ids=[r[0].__name__ for r in REFUSALS])
def test_a_pair_that_disagrees_is_refused_naming_the_field(pair, split, break_it, field):
    chosen = break_it(pair, split) or split

    with pytest.raises(HeldOutModelError, match=field):
        load_frozen_classifier(pair.candidates_dir, pair.calibrated_dir, chosen)


def test_a_missing_calibration_is_refused(pair, split):
    (pair.calibrated_dir / CALIBRATION).unlink()

    with pytest.raises(HeldOutModelError, match="calibration"):
        load_frozen_classifier(pair.candidates_dir, pair.calibrated_dir, split)


def test_a_validation_prediction_that_differs_is_refused(pair, split):
    model = load_frozen_classifier(pair.candidates_dir, pair.calibrated_dir, split)
    path = pair.candidates_dir / VALIDATION_PREDICTIONS
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    name = model.class_order[0]
    lines[0]["probabilities"][name] += 1e-6
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))

    with pytest.raises(HeldOutModelError, match=name):
        check_reproduces_validation(model, pair.validation_rows, pair.candidates_dir)


def test_a_changed_validation_label_is_refused(pair, split):
    model = load_frozen_classifier(pair.candidates_dir, pair.calibrated_dir, split)
    path = pair.candidates_dir / VALIDATION_PREDICTIONS
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    lines[0]["prediction"] = "not-a-class"
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))

    with pytest.raises(HeldOutModelError, match="prediction"):
        check_reproduces_validation(model, pair.validation_rows, pair.candidates_dir)


def test_validation_rows_missing_on_either_side_are_refused(pair, split):
    model = load_frozen_classifier(pair.candidates_dir, pair.calibrated_dir, split)

    with pytest.raises(HeldOutModelError, match="step"):
        check_reproduces_validation(model, pair.validation_rows[1:], pair.candidates_dir)
    path = pair.candidates_dir / VALIDATION_PREDICTIONS
    path.write_text("".join(path.read_text().splitlines(keepends=True)[1:]))
    with pytest.raises(HeldOutModelError, match="step"):
        check_reproduces_validation(model, pair.validation_rows, pair.candidates_dir)


def test_a_duplicate_validation_prediction_line_is_refused(pair, split):
    model = load_frozen_classifier(pair.candidates_dir, pair.calibrated_dir, split)
    path = pair.candidates_dir / VALIDATION_PREDICTIONS
    lines = path.read_text().splitlines(keepends=True)
    path.write_text("".join([*lines, lines[0]]))

    with pytest.raises(HeldOutModelError, match="more than once"):
        check_reproduces_validation(model, pair.validation_rows, pair.candidates_dir)
