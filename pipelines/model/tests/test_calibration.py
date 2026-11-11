"""Temperature scaling, top-label ECE and the abstention threshold, on synthetic scores.

The probabilities and labels here are made up to have known answers. They are not model output
or recorded evidence.
"""

import numpy as np
import pytest
from agentwatch_model.calibration import (
    DEFAULT_CALIBRATION,
    CalibrationError,
    abstaining_scores,
    abstention_rates,
    choose_threshold,
    fit_temperature,
    load_calibration_config,
    scale,
    top_label_ece,
)
from agentwatch_model.contract import DEFAULT_CONTRACT, ContractError, load_contract

CLASSES = ("none", "cpu_sandbox", "inference", "retrieval")


@pytest.fixture(scope="module")
def config():
    return load_calibration_config(DEFAULT_CALIBRATION)


def two_class(confidence: float, steps: int, correct: int) -> tuple[np.ndarray, np.ndarray]:
    """Every step predicts class 0 with the same confidence; the first `correct` are right."""
    probabilities = np.tile([confidence, 1 - confidence], (steps, 1))
    targets = np.array([0] * correct + [1] * (steps - correct))
    return probabilities, targets


def test_temperature_one_leaves_probabilities_unchanged():
    probabilities = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    assert np.allclose(scale(probabilities, 1.0), probabilities)


def test_a_higher_temperature_softens_confidence_without_changing_the_predicted_class():
    probabilities = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    softened = scale(probabilities, 2.0)
    assert np.allclose(softened.sum(axis=1), 1.0)
    assert (softened.max(axis=1) < probabilities.max(axis=1)).all()
    assert (softened.argmax(axis=1) == probabilities.argmax(axis=1)).all()


def test_a_zero_probability_is_scaled_without_error():
    softened = scale(np.array([[1.0, 0.0]]), 2.0)
    assert np.isfinite(softened).all() and softened[0, 0] < 1.0


def test_well_calibrated_scores_keep_a_temperature_near_one(config):
    probabilities, targets = two_class(0.9, steps=100, correct=90)
    assert fit_temperature(probabilities, targets, config) == pytest.approx(1.0, abs=0.01)


def test_overconfident_scores_get_a_temperature_above_one_that_matches_accuracy(config):
    probabilities, targets = two_class(0.95, steps=100, correct=60)
    temperature = fit_temperature(probabilities, targets, config)
    assert temperature > 1
    assert scale(probabilities, temperature)[:, 0].mean() == pytest.approx(0.6, abs=0.005)


def test_underconfident_scores_get_a_temperature_below_one(config):
    probabilities, targets = two_class(0.6, steps=100, correct=95)
    assert fit_temperature(probabilities, targets, config) < 1


def test_the_temperature_stays_within_the_configured_bounds(config):
    # Every step right at 0.51 confidence: the likelihood keeps improving as T shrinks.
    probabilities, targets = two_class(0.51, steps=20, correct=20)
    assert fit_temperature(probabilities, targets, config) == pytest.approx(
        config.min_temperature, rel=0.01
    )


def test_top_label_ece_is_the_weighted_gap_between_confidence_and_accuracy():
    assert top_label_ece([0.9] * 10, [True] * 6 + [False] * 4, bins=10) == pytest.approx(0.3)
    # Two bins: 0.95 with 2/2 right (gap 0.05) and 0.55 with 1/2 right (gap 0.05).
    value = top_label_ece([0.95, 0.95, 0.55, 0.55], [True, True, True, False], bins=10)
    assert value == pytest.approx(0.05)


def test_a_confidence_of_one_falls_in_the_last_bin():
    assert top_label_ece([1.0, 1.0], [True, True], bins=10) == pytest.approx(0.0)


def test_ece_of_no_steps_is_unmeasured_not_zero():
    assert top_label_ece([], [], bins=10) is None


def test_an_abstention_is_a_miss_for_its_true_class_and_a_prediction_of_no_class():
    labels = ["none", "none", "cpu_sandbox", "cpu_sandbox"]
    scores = abstaining_scores(labels, ["none", None, "cpu_sandbox", "none"], CLASSES)
    assert scores["none"]["support"] == 2 and scores["none"]["predicted"] == 2
    assert scores["none"]["recall"] == 0.5 and scores["none"]["precision"] == 0.5
    assert scores["cpu_sandbox"]["recall"] == 0.5 and scores["cpu_sandbox"]["precision"] == 1.0
    assert scores["inference"]["f1"] is None


def test_abstention_rates_are_reported_overall_and_per_true_class():
    labels = ["none", "none", "cpu_sandbox", "cpu_sandbox"]
    rates = abstention_rates(labels, [True, False, False, False], CLASSES)
    assert rates["overall"] == {"steps": 4, "abstained": 1, "rate": 0.25}
    assert rates["per_true_class"]["none"]["rate"] == 0.5
    assert rates["per_true_class"]["cpu_sandbox"]["rate"] == 0.0
    assert rates["per_true_class"]["retrieval"] == {"steps": 0, "abstained": 0, "rate": None}


def test_the_threshold_abstains_on_low_confidence_mistakes_when_the_limit_allows():
    labels = ["none"] * 8 + ["cpu_sandbox"] * 2
    predictions = ["none"] * 8 + ["none", "cpu_sandbox"]
    confidences = [0.9] * 8 + [0.4, 0.9]
    chosen = choose_threshold(labels, predictions, confidences, CLASSES, max_rate=0.5)
    # Abstaining below 0.9 drops only the wrong 0.4 step, lifting none's precision to 1.
    assert chosen["threshold"] == pytest.approx(0.9)
    assert chosen["validation_abstention_rate"] == pytest.approx(0.1)
    assert chosen["validation_macro_f1_defined_classes"] == pytest.approx((1.0 + 2 / 3) / 2)


def test_the_threshold_never_exceeds_the_abstention_limit_overall_or_for_a_class():
    labels = ["none"] * 8 + ["cpu_sandbox"] * 2
    predictions = ["none"] * 8 + ["none", "cpu_sandbox"]
    confidences = [0.9] * 8 + [0.4, 0.9]
    # Abstaining on the 0.4 step is 1 of 2 cpu_sandbox steps: over a 20% per-class limit.
    chosen = choose_threshold(labels, predictions, confidences, CLASSES, max_rate=0.2)
    assert chosen["threshold"] == 0.0 and chosen["validation_abstention_rate"] == 0.0


def test_when_abstaining_never_helps_the_tie_keeps_the_zero_threshold():
    # Thresholds 0 and 0.6 both abstain on nothing and tie; the lower one is kept.
    labels = ["none", "cpu_sandbox"]
    chosen = choose_threshold(labels, labels, [0.6, 0.7], CLASSES, max_rate=0.5)
    assert chosen["threshold"] == 0.0 and chosen["validation_abstention_rate"] == 0.0


def test_the_configuration_is_versioned_by_its_contents(config, tmp_path):
    assert config.version.startswith("calibration-v1+") and len(config.version) == 23
    changed = tmp_path / "calibration.toml"
    text = DEFAULT_CALIBRATION.read_text()
    changed.write_text(text.replace("min_validation_steps = 100", "min_validation_steps = 99"))
    assert load_calibration_config(changed).version != config.version


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("min_temperature = 0.05", "min_temperature = 0", "positive"),
        ("max_temperature = 20.0", "max_temperature = 0.01", "greater than min_temperature"),
        ("min_validation_steps = 100", "min_validation_steps = 1.5", "positive integer"),
        ('version = "calibration-v1"', 'version = "other"', "calibration-v1"),
    ],
)
def test_a_bad_configuration_is_rejected(tmp_path, old, new, message):
    text = DEFAULT_CALIBRATION.read_text()
    assert old in text
    broken = tmp_path / "calibration.toml"
    broken.write_text(text.replace(old, new, 1))
    with pytest.raises(CalibrationError, match=message):
        load_calibration_config(broken)


def test_the_contract_supplies_the_calibration_gates():
    contract = load_contract(DEFAULT_CONTRACT)
    assert contract.calibration_fit_split == "validation"
    assert contract.ece_bins == 10 and contract.max_ece == 0.05
    assert contract.max_abstention_rate == 0.20
    assert contract.abstention_rate_scopes == ("overall", "per_true_class", "mixed_contention")


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('fit_split = "validation"', 'fit_split = "test"', "validation"),
        ("ece_bins = 10", "ece_bins = 0", "positive integer"),
        ("max_ece = 0.05", "max_ece = 2", "from 0 to 1"),
        ('"mixed_contention"]', '"per_task"]', "abstention_rate_scopes"),
    ],
)
def test_bad_calibration_contract_values_are_rejected(tmp_path, old, new, message):
    text = DEFAULT_CONTRACT.read_text()
    assert old in text
    broken = tmp_path / "contract.toml"
    broken.write_text(text.replace(old, new, 1))
    with pytest.raises(ContractError, match=message):
        load_contract(broken)


def test_the_threshold_respects_the_abstention_limit_on_mixed_steps():
    # Abstaining below 0.9 would drop the wrong 0.5 step but also both mixed steps (0.3).
    labels = ["none"] * 8 + ["cpu_sandbox"] * 2 + ["mixed"] * 2
    predictions = ["none"] * 8 + ["none", "cpu_sandbox"] + ["cpu_sandbox", "none"]
    confidences = [0.9] * 8 + [0.5, 0.9] + [0.3, 0.3]
    flags = [False] * 10 + [True] * 2
    chosen = choose_threshold(labels, predictions, confidences, CLASSES, 0.5, mixed=flags)
    assert chosen["threshold"] == 0.0
    assert chosen["validation_mixed_abstention_rate"] == 0.0


def test_the_threshold_limits_only_the_contract_abstention_scopes():
    # The same steps as above; a contract without the mixed scope lets 0.9 abstain on both.
    labels = ["none"] * 8 + ["cpu_sandbox"] * 2 + ["mixed"] * 2
    predictions = ["none"] * 8 + ["none", "cpu_sandbox"] + ["cpu_sandbox", "none"]
    confidences = [0.9] * 8 + [0.5, 0.9] + [0.3, 0.3]
    flags = [False] * 10 + [True] * 2
    chosen = choose_threshold(
        labels,
        predictions,
        confidences,
        CLASSES,
        0.5,
        mixed=flags,
        scopes=("overall", "per_true_class"),
    )
    assert chosen["threshold"] == pytest.approx(0.9)
    assert chosen["validation_mixed_abstention_rate"] == 1.0


def test_mixed_steps_never_enter_the_validation_macro_f1():
    # The same single-class steps, with and without two mixed steps beside them.
    labels = ["none"] * 8 + ["cpu_sandbox"] * 2
    predictions = ["none"] * 8 + ["none", "cpu_sandbox"]
    confidences = [0.9] * 8 + [0.5, 0.9]
    alone = choose_threshold(labels, predictions, confidences, CLASSES, 0.5)
    beside = choose_threshold(
        labels + ["mixed"] * 2,
        predictions + ["cpu_sandbox", "cpu_sandbox"],
        confidences + [0.95, 0.95],
        CLASSES,
        0.5,
        mixed=[False] * 10 + [True] * 2,
    )
    assert beside["threshold"] == alone["threshold"] == pytest.approx(0.9)
    key = "validation_macro_f1_defined_classes"
    assert beside[key] == pytest.approx(alone[key])
