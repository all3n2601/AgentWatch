import pytest
from training_fixtures import dataset, featured_row, require_lightgbm

require_lightgbm()

from agentwatch_model.calibration import load_calibration_config
from agentwatch_model.candidates import fit_candidate, load_training_config
from agentwatch_model.contract import DEFAULT_CONTRACT, load_contract
from agentwatch_model.held_out_scorers import (
    CLASSIFIER_SCORER,
    RULES_SCORER,
    classifier_scores,
    rule_scores,
    seen_task_groups,
    train_classifier,
)
from agentwatch_model.rule_baseline import RowError
from agentwatch_model.rules import load_rules

CLASSES = ("none", "cpu_sandbox", "inference", "retrieval")


@pytest.fixture(scope="module")
def config():
    return load_training_config()


@pytest.fixture(scope="module")
def contract():
    return load_contract(DEFAULT_CONTRACT)


@pytest.fixture(scope="module")
def calibration():
    return load_calibration_config()


def unlabeled(rows):
    return [{k: v for k, v in row.items() if k != "label_resource"} for row in rows]


def test_only_train_and_validation_are_loaded(config, tmp_path):
    names = []
    splits = {"train": dataset(30, "train"), "validation": dataset(8, "valid")}

    def loader(path, name, contract_path, frozen_schema, allow_unchecked_leakage=False):
        names.append(name)
        assert allow_unchecked_leakage is False
        return splits[name]

    candidate, validation, groups = train_classifier(
        tmp_path / "split.json", DEFAULT_CONTRACT, tmp_path / "schema.pbtxt", config, loader
    )
    assert names == ["train", "validation"]
    assert validation == splits["validation"]
    assert candidate.class_order == CLASSES
    assert groups == {f"train-{i}" for i in range(5)}


def test_rules_v1_scores_never_abstain():
    rows = [featured_row("cpu_sandbox", 0), featured_row("none", 1)]
    scores = rule_scores(load_rules(), rows)
    assert [s.scorer for s in scores] == [RULES_SCORER] * 2 == ["rules-v1"] * 2
    assert [s.label for s in scores] == [load_rules().predict(r) for r in rows]
    assert all(s.confidence is None and s.abstained is False for s in scores)


def test_classifier_abstention_applies_the_validation_threshold(config, contract, calibration):
    candidate = fit_candidate(dataset(30, "train"), dataset(30, "valid"), config)
    validation = dataset(30, "valid")
    rows = unlabeled(dataset(10, "held"))
    scores, fit = classifier_scores(candidate, validation, rows, contract, calibration)
    assert fit["fitted"] is True
    assert "abstention" not in fit
    assert fit["temperature"] > 0 and fit["threshold"] is not None
    assert len(scores) == len(rows)
    assert {s.scorer for s in scores} == {CLASSIFIER_SCORER}
    assert CLASSIFIER_SCORER == "lightgbm-classifier-training-v1"
    for score in scores:
        assert score.confidence is not None
        assert score.abstained == (score.confidence < fit["threshold"])
        assert score.label in CLASSES


def test_unfittable_calibration_reports_abstention_unavailable(config, contract, calibration):
    candidate = fit_candidate(dataset(30, "train"), dataset(8, "valid"), config)
    rows = unlabeled(dataset(3, "held"))
    scores, fit = classifier_scores(candidate, dataset(8, "valid"), rows, contract, calibration)
    assert fit["fitted"] is False
    assert fit["abstention"] == "unavailable"
    assert "validation steps" in fit["reason"]
    assert all(s.confidence is None and s.abstained is False for s in scores)
    assert all(s.label in CLASSES for s in scores)


def test_classes_absent_from_train_are_reported(config, tmp_path):
    train = dataset(30, "train", classes=("none", "cpu_sandbox", "inference"))
    splits = {"train": train, "validation": dataset(8, "valid")}

    def loader(path, name, contract_path, frozen_schema, allow_unchecked_leakage=False):
        return splits[name]

    candidate, _, _ = train_classifier(
        tmp_path / "s.json", DEFAULT_CONTRACT, tmp_path / "f.pbtxt", config, loader
    )
    assert candidate.classes_absent_from_train == ("retrieval",)
    assert "retrieval" not in candidate.class_order


def test_seen_task_groups_come_from_train_rows_only():
    train = [featured_row("none", 0, task_id="fix-mean-empty-r2")]
    validation = [featured_row("none", 1, task_id="other")]
    assert seen_task_groups(train) == {"fix-mean-empty"}
    assert "other" not in seen_task_groups(train) and validation


def _train_with(config, tmp_path, train, validation):
    splits = {"train": train, "validation": validation}

    def loader(path, name, contract_path, frozen_schema, allow_unchecked_leakage=False):
        return splits[name]

    return train_classifier(
        tmp_path / "s.json", DEFAULT_CONTRACT, tmp_path / "f.pbtxt", config, loader
    )


def test_a_validation_row_without_a_label_is_rejected(config, tmp_path):
    validation = dataset(8, "valid")
    del validation[0]["label_resource"]
    with pytest.raises(RowError, match="validation row 0 is missing label_resource"):
        _train_with(config, tmp_path, dataset(30, "train"), validation)


def test_mismatched_feature_versions_across_train_and_validation_are_rejected(config, tmp_path):
    train = [row | {"feature_version": "steps-v1.3"} for row in dataset(30, "train")]
    validation = [row | {"feature_version": "steps-v1.2"} for row in dataset(8, "valid")]
    with pytest.raises(RowError, match="Incompatible feature definitions"):
        _train_with(config, tmp_path, train, validation)
