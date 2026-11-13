"""Frozen M03/M04 pair loading on synthetic fixtures; nothing here is recorded evidence."""

import json

import pytest
from frozen_model_fixtures import (
    CALIBRATED_PREDICTIONS,
    CALIBRATION,
    SYNTHETIC_VERSIONS,
    TEMPERATURE,
    THRESHOLD,
    calibrated_records,
    jsonl_text,
    write_frozen_pair,
)
from training_fixtures import dataset, require_lightgbm

require_lightgbm()

from agentwatch_data.files import file_sha256
from agentwatch_model.candidates import predict
from agentwatch_model.features import feature_matrix
from agentwatch_model.held_out_model import (
    CLASSIFIER_SCORER,
    FROZEN_CALIBRATION_VERSION,
    FROZEN_TRAINING_VERSION,
    FROZEN_VERSIONS,
    REPRODUCTION_TOLERANCE,
    FrozenClassifier,
    FrozenVersions,
    HeldOutModelError,
    check_reproduces_calibration,
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


def load(pair, split, versions=SYNTHETIC_VERSIONS):
    """The synthetic pair, loaded against the versions it records unless a test says otherwise."""
    return load_frozen_classifier(pair.candidates_dir, pair.calibrated_dir, split, versions)


def edit(path, change):
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))


def write_calibrated(pair, records):
    """Replace M04's test predictions and keep its report's count and hash true to them."""
    path = pair.calibrated_dir / CALIBRATED_PREDICTIONS
    path.write_text(jsonl_text(records))

    def rebind(report):
        report["scored_steps"] = len(records) - report["joint_only_steps_not_scored"]
        report["predictions"]["sha256"] = file_sha256(path)

    edit(pair.calibrated_dir / REPORT, rebind)


def calibrated_lines(pair):
    path = pair.calibrated_dir / CALIBRATED_PREDICTIONS
    return [json.loads(line) for line in path.read_text().splitlines()]


def rebind_preprocessing(pair):
    """Keep report.json's hash for preprocessing.json true after a test edits the file."""
    sha = file_sha256(pair.candidates_dir / PREPROCESSING)
    edit(pair.candidates_dir / REPORT, lambda r: r["preprocessing"].update(sha256=sha))


def test_a_matching_pair_loads_and_records_its_identity(pair, split):
    model = load(pair, split)

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
    assert identity["validation_steps"] == len(pair.validation_rows)
    assert identity["calibration_report_sha256"] == file_sha256(pair.calibrated_dir / REPORT)
    assert identity["calibration_predictions_sha256"] == file_sha256(
        pair.calibrated_dir / CALIBRATED_PREDICTIONS
    )
    check_reproduces_validation(model, pair.validation_rows, pair.candidates_dir)


def test_loaded_scores_equal_the_trained_candidate(pair, split):
    model = load(pair, split)
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
    model = load(pair, split)
    rows = dataset(2, "held")
    rows[0]["cpu_seconds"] = None

    assert len(frozen_scores(model, rows)) == len(rows)


@pytest.mark.parametrize(("threshold", "abstains"), [(0.99, True), (1e-6, False)])
def test_abstains_below_the_calibrated_threshold(pair, split, threshold, abstains):
    # A high temperature flattens the calibrated confidences, so 0.99 is out of reach for every
    # step.
    fitted = {"abstention_threshold": threshold, "temperature": 50.0}
    edit(pair.calibrated_dir / CALIBRATION, lambda c: c.update(fitted))
    edit(pair.calibrated_dir / REPORT, lambda r: r["calibration"].update(fitted))
    model = load(pair, split)

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


def preprocessing_training_version_differs(pair, split):
    edit(pair.candidates_dir / PREPROCESSING, lambda p: p.update(training_version="training-v1+x"))
    rebind_preprocessing(pair)


def m03_model_version_differs(pair, split):
    edit(pair.candidates_dir / REPORT, lambda r: r["model"].update(version="training-v1+x"))


def calibration_version_differs(pair, split):
    edit(pair.calibrated_dir / CALIBRATION, lambda c: c.update(version="calibration-v1+x"))


def m04_report_version_differs(pair, split):
    edit(pair.calibrated_dir / REPORT, lambda r: r["calibration"].update(version="other"))


def m04_report_temperature_differs(pair, split):
    edit(pair.calibrated_dir / REPORT, lambda r: r["calibration"].update(temperature=1.5))


def m04_report_threshold_differs(pair, split):
    edit(pair.calibrated_dir / REPORT, lambda r: r["calibration"].update(abstention_threshold=0.1))


def m04_report_split_differs(pair, split):
    edit(pair.calibrated_dir / REPORT, lambda r: r["split"].update(sha256="0" * 64))


def m04_predictions_edited_without_rebinding(pair, split):
    path = pair.calibrated_dir / CALIBRATED_PREDICTIONS
    path.write_text(path.read_text() + "\n")


def m03_report_edited_after_calibration(pair, split):
    edit(pair.candidates_dir / REPORT, lambda r: r.update(note="changed"))


def m04_report_missing(pair, split):
    (pair.calibrated_dir / REPORT).unlink()


M04_REPORT = r"calibration[\\/]report\.json"
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
    (preprocessing_training_version_differs, r"preprocessing\.json training_version"),
    (m03_model_version_differs, r"report\.json model\.version"),
    (calibration_version_differs, r"calibration\.json version"),
    (m04_report_version_differs, f"{M04_REPORT} calibration\\.version"),
    (m04_report_temperature_differs, f"{M04_REPORT} calibration\\.temperature"),
    (m04_report_threshold_differs, f"{M04_REPORT} calibration\\.abstention_threshold"),
    (m04_report_split_differs, f"{M04_REPORT} split"),
    (m04_predictions_edited_without_rebinding, f"{M04_REPORT} predictions\\.sha256"),
    (m03_report_edited_after_calibration, "candidates\\.report_sha256"),
    (m04_report_missing, "M04 report"),
]


@pytest.mark.parametrize(("break_it", "field"), REFUSALS, ids=[r[0].__name__ for r in REFUSALS])
def test_a_pair_that_disagrees_is_refused_naming_the_field(pair, split, break_it, field):
    chosen = break_it(pair, split) or split

    with pytest.raises(HeldOutModelError, match=field):
        load(pair, chosen)


def test_a_missing_calibration_is_refused(pair, split):
    (pair.calibrated_dir / CALIBRATION).unlink()

    with pytest.raises(HeldOutModelError, match="calibration"):
        load(pair, split)


def test_a_validation_prediction_that_differs_is_refused(pair, split):
    model = load(pair, split)
    path = pair.candidates_dir / VALIDATION_PREDICTIONS
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    name = model.class_order[0]
    lines[0]["probabilities"][name] += 1e-6
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))

    with pytest.raises(HeldOutModelError, match=name):
        check_reproduces_validation(model, pair.validation_rows, pair.candidates_dir)


def test_a_changed_validation_label_is_refused(pair, split):
    model = load(pair, split)
    path = pair.candidates_dir / VALIDATION_PREDICTIONS
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    lines[0]["prediction"] = "not-a-class"
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))

    with pytest.raises(HeldOutModelError, match="prediction"):
        check_reproduces_validation(model, pair.validation_rows, pair.candidates_dir)


def test_validation_rows_missing_on_either_side_are_refused(pair, split):
    model = load(pair, split)

    with pytest.raises(HeldOutModelError, match="step"):
        check_reproduces_validation(model, pair.validation_rows[1:], pair.candidates_dir)
    path = pair.candidates_dir / VALIDATION_PREDICTIONS
    path.write_text("".join(path.read_text().splitlines(keepends=True)[1:]))
    with pytest.raises(HeldOutModelError, match="step"):
        check_reproduces_validation(model, pair.validation_rows, pair.candidates_dir)


def test_a_duplicate_validation_prediction_line_is_refused(pair, split):
    model = load(pair, split)
    path = pair.candidates_dir / VALIDATION_PREDICTIONS
    lines = path.read_text().splitlines(keepends=True)
    path.write_text("".join([*lines, lines[0]]))

    with pytest.raises(HeldOutModelError, match="more than once"):
        check_reproduces_validation(model, pair.validation_rows, pair.candidates_dir)


def test_an_empty_validation_record_is_refused(pair, split):
    model = load(pair, split)
    (pair.candidates_dir / VALIDATION_PREDICTIONS).write_text("")

    with pytest.raises(HeldOutModelError, match="no steps"):
        check_reproduces_validation(model, [], pair.candidates_dir)


def test_a_validation_count_that_disagrees_with_m04_fit_steps_is_refused(pair, split):
    model = load(pair, split)
    assert model.validation_steps == len(pair.validation_rows)
    # The file and the rows still agree with each other; only M04's fitted count disagrees.
    path = pair.candidates_dir / VALIDATION_PREDICTIONS
    path.write_text("".join(path.read_text().splitlines(keepends=True)[1:]))

    with pytest.raises(HeldOutModelError, match="fit.steps"):
        check_reproduces_validation(model, pair.validation_rows[1:], pair.candidates_dir)


def test_m04_test_predictions_are_reproduced(pair, split):
    model = load(pair, split)

    check_reproduces_calibration(model, pair.test_rows, pair.calibrated_dir)


def test_m04_predictions_of_a_subset_of_test_rows_are_reproduced(pair, split):
    write_calibrated(pair, calibrated_lines(pair)[1:])
    model = load(pair, split)

    check_reproduces_calibration(model, pair.test_rows, pair.calibrated_dir)


def test_m04_joint_only_steps_count_towards_the_recorded_steps(pair, split):
    # M04 records a prediction for a joint-only step but leaves it out of scored_steps.
    def one_joint_only(report):
        report["scored_steps"] -= 1
        report["joint_only_steps_not_scored"] = 1

    edit(pair.calibrated_dir / REPORT, one_joint_only)
    model = load(pair, split)
    assert model.calibrated_test_steps == len(pair.test_rows)

    check_reproduces_calibration(model, pair.test_rows, pair.calibrated_dir)


def test_m04_predictions_at_another_temperature_are_refused(pair, split):
    write_calibrated(
        pair, calibrated_records(pair.candidate, pair.test_rows, TEMPERATURE * 2, THRESHOLD)
    )
    model = load(pair, split)

    with pytest.raises(HeldOutModelError, match="differs from the loaded model"):
        check_reproduces_calibration(model, pair.test_rows, pair.calibrated_dir)


def change_confidence(record):
    record["confidence"] += 1e-6


def change_probability(record):
    name = min(record["probabilities"])
    record["probabilities"][name] += 1e-6


def drop_probability(record):
    record["probabilities"].popitem()


def change_abstained(record):
    record["abstained"] = not record["abstained"]


def change_prediction(record):
    record["prediction"] = "not-a-class"


def null_confidence(record):
    record["confidence"] = None


CALIBRATION_CHANGES = [
    (change_confidence, "confidence"),
    (change_probability, "probability"),
    (drop_probability, "class order"),
    (change_abstained, "abstained"),
    (change_prediction, "prediction"),
    (null_confidence, "confidence"),
]


@pytest.mark.parametrize(
    ("change", "message"), CALIBRATION_CHANGES, ids=[c[0].__name__ for c in CALIBRATION_CHANGES]
)
def test_an_m04_test_prediction_that_differs_is_refused(pair, split, change, message):
    records = calibrated_lines(pair)
    change(records[-1])
    write_calibrated(pair, records)
    model = load(pair, split)

    with pytest.raises(HeldOutModelError, match=message):
        check_reproduces_calibration(model, pair.test_rows, pair.calibrated_dir)


def test_an_m04_step_missing_from_the_test_rows_is_refused(pair, split):
    model = load(pair, split)

    with pytest.raises(HeldOutModelError, match="not test-split rows"):
        check_reproduces_calibration(model, pair.test_rows[1:], pair.calibrated_dir)


def test_an_m04_count_that_disagrees_with_scored_steps_is_refused(pair, split):
    edit(pair.calibrated_dir / REPORT, lambda r: r.update(scored_steps=r["scored_steps"] + 1))
    model = load(pair, split)

    with pytest.raises(HeldOutModelError, match="scored_steps plus joint_only_steps_not_scored"):
        check_reproduces_calibration(model, pair.test_rows, pair.calibrated_dir)


def test_empty_or_repeated_m04_predictions_are_refused(pair, split):
    records = calibrated_lines(pair)
    write_calibrated(pair, [*records, records[0]])
    model = load(pair, split)
    with pytest.raises(HeldOutModelError, match="more than once"):
        check_reproduces_calibration(model, pair.test_rows, pair.calibrated_dir)

    (pair.calibrated_dir / CALIBRATED_PREDICTIONS).write_text("")
    with pytest.raises(HeldOutModelError, match="no steps"):
        check_reproduces_calibration(model, pair.test_rows, pair.calibrated_dir)


def test_unreadable_m04_predictions_are_refused(pair, split):
    model = load(pair, split)
    (pair.calibrated_dir / CALIBRATED_PREDICTIONS).write_text("{not json\n")

    with pytest.raises(HeldOutModelError, match="Cannot read"):
        check_reproduces_calibration(model, pair.test_rows, pair.calibrated_dir)


@pytest.mark.parametrize("value", [0, -1, 2.5, True, None])
@pytest.mark.parametrize("key", ["fit.steps", "scored_steps"])
def test_m04_step_counts_must_be_positive_integers(pair, split, key, value):
    def change(report):
        if key == "fit.steps":
            report["fit"]["steps"] = value
        else:
            report["scored_steps"] = value

    edit(pair.calibrated_dir / REPORT, change)

    with pytest.raises(HeldOutModelError, match=f"{key} must be a positive integer"):
        load(pair, split)


@pytest.mark.parametrize("value", [-1, 0.5, True, None])
def test_m04_joint_only_count_must_be_a_non_negative_integer(pair, split, value):
    edit(pair.calibrated_dir / REPORT, lambda r: r.update(joint_only_steps_not_scored=value))

    with pytest.raises(HeldOutModelError, match="joint_only_steps_not_scored must be a non-neg"):
        load(pair, split)


def test_the_frozen_versions_are_the_evaluated_m03_m04_pair():
    assert FROZEN_TRAINING_VERSION == "training-v1+a34ea0be"
    assert FROZEN_CALIBRATION_VERSION == "calibration-v1+08119e78"
    assert FROZEN_VERSIONS == FrozenVersions(FROZEN_TRAINING_VERSION, FROZEN_CALIBRATION_VERSION)


def test_a_synthetic_pair_is_refused_by_the_pinned_versions(pair, split):
    # The synthetic training version is the packaged config's, so the calibration pin refuses.
    with pytest.raises(HeldOutModelError, match=r"calibration\.json version .*08119e78"):
        load_frozen_classifier(pair.candidates_dir, pair.calibrated_dir, split)


def test_a_pair_of_another_training_version_is_refused(pair, split):
    versions = FrozenVersions("training-v1+other", SYNTHETIC_VERSIONS.calibration)

    with pytest.raises(HeldOutModelError, match=r"training_version .*training-v1\+other"):
        load(pair, split, versions)
