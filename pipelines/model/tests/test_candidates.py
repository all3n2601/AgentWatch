import pytest
from training_fixtures import dataset, featured_row, require_lightgbm

require_lightgbm()

from agentwatch_model.candidates import (
    DEFAULT_TRAINING,
    CandidateError,
    TrainingError,
    fit_candidate,
    load_training_config,
    predict,
)


@pytest.fixture(scope="module")
def config():
    return load_training_config(DEFAULT_TRAINING)


@pytest.fixture(scope="module")
def candidate(config):
    return fit_candidate(dataset(30, "train"), dataset(8, "valid"), config)


def test_training_config_is_versioned_by_its_parsed_contents(config, tmp_path):
    assert (
        config.version.startswith("training-v1+") and len(config.version) == len("training-v1+") + 8
    )
    edited = tmp_path / "training.toml"
    edited.write_text(
        DEFAULT_TRAINING.read_text().replace("num_boost_round = 300", "num_boost_round = 301")
    )
    assert load_training_config(edited).version != config.version


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("num_boost_round = 300", "num_boost_round = 0", "positive integer"),
        ("learning_rate = 0.05", "learning_rate = nan", "finite"),
        ("num_leaves = 15", 'num_leaves = 15\nboosting = "dart"', "boosting"),
    ],
)
def test_invalid_training_settings_are_rejected(tmp_path, old, new, message):
    bad = tmp_path / "training.toml"
    bad.write_text(DEFAULT_TRAINING.read_text().replace(old, new, 1))
    with pytest.raises(TrainingError, match=message):
        load_training_config(bad)


def test_candidate_learns_every_class_present_in_training(candidate):
    assert candidate.class_order == ("none", "cpu_sandbox", "inference", "retrieval")
    test = dataset(6, "test")
    predictions = predict(candidate, test)
    accuracy = sum(
        p.label == r["label_resource"] for p, r in zip(predictions, test, strict=True)
    ) / len(test)
    assert accuracy >= 0.9


def test_probabilities_cover_the_class_order_and_sum_to_one(candidate):
    [prediction] = predict(candidate, [featured_row("cpu_sandbox", 1)])
    assert set(prediction.probabilities) == set(candidate.class_order)
    assert sum(prediction.probabilities.values()) == pytest.approx(1.0)
    assert prediction.label == max(prediction.probabilities, key=prediction.probabilities.get)


def test_excess_fractions_are_clipped_to_zero_and_one(candidate):
    fractions = [p.excess_fraction for p in predict(candidate, dataset(4, "test"))]
    assert all(0.0 <= f <= 1.0 for f in fractions)


def test_missing_readings_are_accepted_without_imputation(candidate):
    gap = featured_row("none", 2, queue_seconds=None, queue_fraction=None)
    assert predict(candidate, [gap])[0].label in candidate.class_order


def test_training_is_deterministic(config):
    first = fit_candidate(dataset(20, "train"), dataset(5, "valid"), config)
    second = fit_candidate(dataset(20, "train"), dataset(5, "valid"), config)
    rows = dataset(3, "test")
    assert [p.probabilities for p in predict(first, rows)] == [
        p.probabilities for p in predict(second, rows)
    ]


def test_early_stopping_uses_validation_only(candidate, config):
    assert candidate.early_stopping == "validation"
    assert 1 <= candidate.best_iterations["classifier"] <= config.num_boost_round


def test_without_usable_validation_rows_training_uses_the_fixed_rounds(config):
    candidate = fit_candidate(dataset(20, "train", classes=("none", "cpu_sandbox")), [], config)
    assert candidate.early_stopping == "not_used"
    assert candidate.best_iterations["classifier"] == config.num_boost_round


def test_training_needs_at_least_two_classes(config):
    with pytest.raises(CandidateError, match="at least two classes"):
        fit_candidate(dataset(10, "train", classes=("none",)), [], config)


def test_coverage_gaps_are_recorded_not_hidden(config):
    train = dataset(20, "train", classes=("none", "cpu_sandbox"))
    validation = dataset(4, "valid", classes=("none", "inference"))
    candidate = fit_candidate(train, validation, config)
    assert candidate.validation_rows_dropped == 4
    assert candidate.classes_absent_from_train == ("inference", "retrieval")
