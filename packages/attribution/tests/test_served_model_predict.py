"""ServedModel.predict on synthetic pairs; nothing here is recorded evidence."""

import hashlib
import json
import math
from dataclasses import FrozenInstanceError
from pathlib import Path

import lightgbm
import numpy as np
import pytest
from agentwatch_attribution.calibration import scale
from agentwatch_attribution.classes import RESOURCE_CLASSES
from agentwatch_attribution.inputs import MODEL_INPUTS, FeatureError, feature_matrix
from agentwatch_attribution.served_model import ModelPrediction, ServedModel, load_served_model
from pair_fixtures import (
    CALIBRATION,
    MODEL_FILES,
    TEMPERATURE,
    synthetic_rows,
    write_synthetic_pair,
)

Pair = tuple[Path, Path]
# A seed other than the fixture's training seed, so the rows are not the ones it learned from.
SCORED_SEED = 11
PRESSURE = "cpu_pressure_max_avg10"


@pytest.fixture
def pair(tmp_path: Path) -> Pair:
    return write_synthetic_pair(tmp_path)


def booster(pair: Pair, key: str) -> lightgbm.Booster:
    return lightgbm.Booster(model_file=str(pair[0] / MODEL_FILES[key]))


def with_threshold(pair: Pair, threshold: float) -> ServedModel:
    """The pair reloaded after calibration.json is rewritten to carry this threshold."""
    path = pair[1] / CALIBRATION
    document = json.loads(path.read_text())
    document["abstention_threshold"] = threshold
    path.write_text(json.dumps(document))
    return load_served_model(*pair)


def top_class(prediction: ModelPrediction) -> str:
    probabilities = prediction.calibrated_probabilities
    return max(probabilities, key=probabilities.__getitem__)


def test_predict_matches_manual_booster_and_scale(pair: Pair) -> None:
    rows = synthetic_rows(20, seed=SCORED_SEED)
    predictions = load_served_model(*pair).predict(rows)

    features = feature_matrix(rows)
    raw = booster(pair, "classifier").predict(features)
    calibrated = scale(raw, TEMPERATURE)
    excess = np.clip(booster(pair, "regressor").predict(features), 0.0, 1.0)
    assert len(predictions) == len(rows)
    for index, prediction in enumerate(predictions):
        assert list(prediction.raw_probabilities) == list(RESOURCE_CLASSES)
        assert list(prediction.calibrated_probabilities) == list(RESOURCE_CLASSES)
        assert list(prediction.raw_probabilities.values()) == raw[index].tolist()
        assert list(prediction.calibrated_probabilities.values()) == calibrated[index].tolist()
        assert prediction.confidence == calibrated[index].max()
        assert prediction.excess_fraction == excess[index]
        assert not prediction.abstained
        assert prediction.predicted_class == RESOURCE_CLASSES[int(calibrated[index].argmax())]
        values = [
            *prediction.raw_probabilities.values(),
            *prediction.calibrated_probabilities.values(),
            prediction.confidence,
            prediction.excess_fraction,
        ]
        assert all(type(value) is float for value in values)


def test_predict_of_no_rows_is_empty(pair: Pair) -> None:
    assert load_served_model(*pair).predict([]) == []


def test_prediction_is_frozen(pair: Pair) -> None:
    prediction = load_served_model(*pair).predict(synthetic_rows(1, seed=SCORED_SEED))[0]
    with pytest.raises(FrozenInstanceError):
        prediction.abstained = True


def test_abstains_below_threshold_and_has_no_class(pair: Pair) -> None:
    rows = synthetic_rows(1, seed=SCORED_SEED)
    confidence = load_served_model(*pair).predict(rows)[0].confidence

    [prediction] = with_threshold(pair, float(np.nextafter(confidence, 1.0))).predict(rows)

    assert prediction.abstained
    assert prediction.predicted_class is None
    assert prediction.confidence == confidence
    # The calibrated scores are still reported; only the attribution is withheld.
    assert top_class(prediction) in RESOURCE_CLASSES


def test_exactly_at_threshold_does_not_abstain(pair: Pair) -> None:
    rows = synthetic_rows(1, seed=SCORED_SEED)
    confidence = load_served_model(*pair).predict(rows)[0].confidence

    model = with_threshold(pair, confidence)

    calibration_bytes = (pair[1] / CALIBRATION).read_bytes()
    assert model.identity.abstention_threshold == confidence
    assert model.identity.calibration_sha256 == hashlib.sha256(calibration_bytes).hexdigest()
    [prediction] = model.predict(rows)
    assert prediction.confidence == confidence
    assert not prediction.abstained
    assert prediction.predicted_class == top_class(prediction)


def test_missing_inputs_pass_as_nan_not_zero(pair: Pair) -> None:
    [row] = synthetic_rows(1, seed=SCORED_SEED)
    missing, zero = row | {PRESSURE: None}, row | {PRESSURE: 0.0}

    missing_prediction, zero_prediction = load_served_model(*pair).predict([missing, zero])

    nan_features = feature_matrix([zero])
    nan_features[0, MODEL_INPUTS.index(PRESSURE)] = math.nan
    nan_raw = booster(pair, "classifier").predict(nan_features)[0]
    assert list(missing_prediction.raw_probabilities.values()) == nan_raw.tolist()
    assert missing_prediction.raw_probabilities != zero_prediction.raw_probabilities
    assert missing_prediction.predicted_class != zero_prediction.predicted_class


@pytest.mark.parametrize(("offset", "bound"), [(2.0, 1.0), (-2.0, 0.0)])
def test_excess_fraction_is_clipped_to_the_declared_range(
    tmp_path: Path, offset: float, bound: float
) -> None:
    pair = write_synthetic_pair(tmp_path, regressor_offset=offset)
    rows = synthetic_rows(5, seed=SCORED_SEED)

    predictions = load_served_model(*pair).predict(rows)

    unclipped = booster(pair, "regressor").predict(feature_matrix(rows))
    assert ((unclipped > 1.0) | (unclipped < 0.0)).all()
    assert [prediction.excess_fraction for prediction in predictions] == [bound] * len(rows)


def test_row_missing_a_column_raises_feature_error(pair: Pair) -> None:
    rows = synthetic_rows(2, seed=SCORED_SEED)
    del rows[1]["telemetry_coverage"]

    with pytest.raises(FeatureError, match="Row 1 is missing feature telemetry_coverage"):
        load_served_model(*pair).predict(rows)
