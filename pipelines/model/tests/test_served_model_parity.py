"""The served model scores as the held-out scorer does, on one synthetic pair loaded by both.

Nothing here is recorded evidence: the pair is frozen_model_fixtures' synthetic one.
"""

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from frozen_model_fixtures import CALIBRATION, SYNTHETIC_VERSIONS, FrozenPair, write_frozen_pair
from training_fixtures import require_lightgbm

require_lightgbm()

from agentwatch_attribution.served_model import ServedModel, load_served_model
from agentwatch_data.files import file_sha256
from agentwatch_model.held_out_model import (
    REPRODUCTION_TOLERANCE,
    FrozenClassifier,
    frozen_scores,
    load_frozen_classifier,
)
from agentwatch_model.train_candidates import PREPROCESSING, REPORT

EXCESS_FRACTION_CLIP = [0.0, 1.0]


def edit(path: Path, change: Callable[[dict], None]) -> None:
    document = json.loads(path.read_text())
    change(document)
    path.write_text(json.dumps(document))


def complete_to_real_layout(pair: FrozenPair) -> None:
    """Add the fields the real writers record and frozen_model_fixtures trims.

    `make train-candidates` writes preprocessing.json with `status` and `excess_fraction_clip`,
    and `make calibrate` writes calibration.json with `blocks_promotion`, `probability_columns`
    and both model hashes; the served loader requires them. Each edit is rebound through M03's
    report and calibration.json's hash of it, so the held-out loader accepts the same files.
    """
    preprocessing_path = pair.candidates_dir / PREPROCESSING
    report_path = pair.candidates_dir / REPORT
    preprocessing = json.loads(preprocessing_path.read_text())

    def evaluated(document: dict) -> None:
        document["status"] = "evaluated"
        document["excess_fraction_clip"] = EXCESS_FRACTION_CLIP

    def rebind_preprocessing(report: dict) -> None:
        report["preprocessing"]["sha256"] = file_sha256(preprocessing_path)

    def calibrated(document: dict) -> None:
        document["blocks_promotion"] = True
        document["probability_columns"] = preprocessing["probability_columns"]
        document["model_files"] = preprocessing["model_files"]
        document["candidates"]["report_sha256"] = file_sha256(report_path)

    edit(preprocessing_path, evaluated)
    edit(report_path, rebind_preprocessing)
    edit(pair.calibrated_dir / CALIBRATION, calibrated)


def set_threshold(pair: FrozenPair, threshold: float) -> None:
    """Rewrite the abstention threshold in calibration.json and in M04's report, which the
    held-out loader requires to agree."""

    def in_calibration(document: dict) -> None:
        document["abstention_threshold"] = threshold

    def in_report(report: dict) -> None:
        report["calibration"]["abstention_threshold"] = threshold

    edit(pair.calibrated_dir / CALIBRATION, in_calibration)
    edit(pair.calibrated_dir / REPORT, in_report)


@pytest.fixture
def split(tmp_path: Path) -> Path:
    path = tmp_path / "split-test.json"
    path.write_text('{"synthetic": true}\n')
    return path


@pytest.fixture
def pair(tmp_path: Path, split: Path) -> FrozenPair:
    frozen = write_frozen_pair(tmp_path, split)
    complete_to_real_layout(frozen)
    return frozen


def loaded(pair: FrozenPair, split: Path) -> tuple[FrozenClassifier, ServedModel]:
    """The same files through the held-out loader and through the served loader."""
    held_out = load_frozen_classifier(
        pair.candidates_dir, pair.calibrated_dir, split, SYNTHETIC_VERSIONS
    )
    return held_out, load_served_model(pair.candidates_dir, pair.calibrated_dir)


def scored_rows(pair: FrozenPair) -> list[dict]:
    return pair.validation_rows + pair.test_rows


def assert_same_scores(pair: FrozenPair, split: Path) -> list[bool]:
    """Class, confidence and abstention agree on every row; returns the abstention flags."""
    rows = scored_rows(pair)
    held_out, served = loaded(pair, split)

    expected = frozen_scores(held_out, rows)
    predictions = served.predict(rows)

    assert served.identity.class_order == held_out.class_order
    assert served.identity.abstention_threshold == held_out.threshold
    assert len(predictions) == len(expected) == len(rows)
    for step, prediction in zip(expected, predictions, strict=True):
        assert step.confidence is not None
        assert abs(prediction.confidence - step.confidence) <= REPRODUCTION_TOLERANCE
        assert prediction.abstained is step.abstained
        # frozen_scores reports the top class even when abstaining; serving withholds it.
        probabilities = prediction.calibrated_probabilities
        assert max(probabilities, key=probabilities.__getitem__) == step.label
        assert prediction.predicted_class == (None if step.abstained else step.label)
    return [prediction.abstained for prediction in predictions]


def test_served_predictions_equal_frozen_scores(pair: FrozenPair, split: Path) -> None:
    assert_same_scores(pair, split)


def test_served_abstentions_equal_frozen_scores(pair: FrozenPair, split: Path) -> None:
    held_out, _ = loaded(pair, split)
    confidences = sorted(
        step.confidence
        for step in frozen_scores(held_out, scored_rows(pair))
        if step.confidence is not None
    )
    # The median confidence: the rows below it abstain, the rest, it included, do not.
    set_threshold(pair, confidences[len(confidences) // 2])

    abstained = assert_same_scores(pair, split)

    assert any(abstained)
    assert not all(abstained)
