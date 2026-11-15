"""LightGBM attribution candidates: a resource classifier and an excess-latency regressor.

Both train on the training split only; the validation split is used only for early stopping.
Settings come from training.toml; earlier versions are kept as training-v<N>.toml. LightGBM is
imported where it is used, so the configuration and errors stay importable on hosts where its
OpenMP runtime is missing.
"""

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from agentwatch_data.extract import canonical_sha256
from agentwatch_data.slices import balancing_weights

from agentwatch_model.features import FEATURES, feature_matrix
from agentwatch_model.metrics import CLASSES, is_number

if TYPE_CHECKING:
    import lightgbm

DEFAULT_TRAINING = Path(__file__).with_name("training.toml")
# The settings' version, `training-v<N>`; tune-candidates proposes the next one.
VERSION_PATTERN = re.compile(r"training-v[1-9][0-9]*")
HASH_DIGITS = 8  # matches the rules and contract identifiers
TUNABLE = ("learning_rate", "num_leaves", "min_data_in_leaf", "feature_fraction", "lambda_l2")
# Training rows can be re-weighted so every value of one of these attributes carries the same total
# weight (the data slices' balancing weights); unset, every row weighs 1.
WEIGHT_ATTRIBUTES = ("condition",)
# One thread, a fixed seed and LightGBM's deterministic mode make retraining reproducible.
FIXED = {"deterministic": True, "num_threads": 1, "seed": 0, "verbose": -1, "force_col_wise": True}

Validation = tuple[np.ndarray, np.ndarray] | None


class TrainingError(ValueError):
    """A training configuration that cannot produce a reproducible candidate."""


class CandidateError(ValueError):
    """Training data that cannot produce a candidate."""


@dataclass(frozen=True)
class TrainingConfig:
    version: str
    num_boost_round: int
    early_stopping_rounds: int
    classifier: dict
    regressor: dict
    weight_by: str | None = None


@dataclass(frozen=True)
class Prediction:
    label: str
    probabilities: dict[str, float]
    excess_fraction: float


@dataclass(frozen=True)
class Candidate:
    class_order: tuple[str, ...]
    classifier: "lightgbm.Booster"
    regressor: "lightgbm.Booster"
    best_iterations: dict[str, int]
    # How each model stopped: "validation" or "not_used". early_stopping is the classifier's
    # (single-class validation rows of trained classes); the regressor uses every validation row.
    early_stopping: str
    training_version: str
    # Coverage gaps, reported rather than hidden: validation rows of classes unseen in training
    # cannot be scored for early stopping, and absent classes can never be predicted.
    validation_rows_dropped: int = 0
    regressor_early_stopping: str = "not_used"
    # Mixed training steps: the regressor learns from them, the classifier cannot.
    train_rows_not_classified: int = 0
    classes_absent_from_train: tuple[str, ...] = ()


def check_lightgbm() -> None:
    """Raise TrainingError when LightGBM cannot load, such as without its OpenMP runtime."""
    try:
        import lightgbm  # noqa: F401
    except (ImportError, OSError) as error:
        raise TrainingError(
            f"LightGBM could not load ({type(error).__name__}: {error}); it needs OpenMP, "
            "libgomp on Linux or libomp on macOS"
        ) from error


def positive_int(parsed: dict, key: str) -> int:
    value = parsed.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TrainingError(f"{key} must be a positive integer")
    return value


def booster_settings(parsed: dict, table: str) -> dict:
    settings = parsed.get(table)
    if not isinstance(settings, dict):
        raise TrainingError(f"Missing [{table}]")
    for key, value in settings.items():
        if key not in TUNABLE:
            raise TrainingError(f"[{table}] {key} is not an allowed setting")
        if not is_number(value):
            raise TrainingError(f"[{table}] {key} must be a finite number")
    return dict(settings)


def load_training_config(path: Path = DEFAULT_TRAINING) -> TrainingConfig:
    parsed = tomllib.loads(path.read_text())
    name = parsed.get("version")
    if not isinstance(name, str) or not VERSION_PATTERN.fullmatch(name):
        raise TrainingError(f"{path.name}: version must be training-v<N>, got {name!r}")
    # Validate before hashing: canonical JSON refuses NaN, and the error should name the value.
    rounds = positive_int(parsed, "num_boost_round")
    patience = positive_int(parsed, "early_stopping_rounds")
    classifier = booster_settings(parsed, "classifier")
    regressor = booster_settings(parsed, "regressor")
    weight_by = parsed.get("weight_by")
    if weight_by is not None and weight_by not in WEIGHT_ATTRIBUTES:
        raise TrainingError(
            f"weight_by must be one of {list(WEIGHT_ATTRIBUTES)}, got {weight_by!r}"
        )
    return TrainingConfig(
        version=f"{name}+{canonical_sha256(parsed)[:HASH_DIGITS]}",
        num_boost_round=rounds,
        early_stopping_rounds=patience,
        classifier=classifier,
        regressor=regressor,
        weight_by=weight_by,
    )


def row_weights(rows: list[dict], attribute: str | None) -> np.ndarray | None:
    """Balancing weights by one attribute's value, or None for unweighted training."""
    if attribute is None:
        return None
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row[attribute])] = counts.get(str(row[attribute]), 0) + 1
    by_value = balancing_weights({value: {"rows": n} for value, n in counts.items()}, len(rows))
    return np.array([by_value[str(row[attribute])] for row in rows], dtype=np.float64)


def class_targets(rows: list[dict], index: dict[str, int]) -> np.ndarray:
    return np.array([index[row["label_resource"]] for row in rows])


def fraction_targets(rows: list[dict]) -> np.ndarray:
    return np.array([row["label_excess_fraction"] for row in rows], dtype=np.float64)


def train_booster(
    params: dict,
    features: np.ndarray,
    target: np.ndarray,
    validation: Validation,
    config: TrainingConfig,
    weight: np.ndarray | None = None,
) -> "lightgbm.Booster":
    import lightgbm

    data = lightgbm.Dataset(
        features, target, weight=weight, feature_name=list(FEATURES), free_raw_data=False
    )
    if validation is None:
        return lightgbm.train(params, data, num_boost_round=config.num_boost_round)
    valid = lightgbm.Dataset(*validation, reference=data)
    stop = lightgbm.early_stopping(config.early_stopping_rounds, verbose=False)
    return lightgbm.train(
        params, data, num_boost_round=config.num_boost_round, valid_sets=[valid], callbacks=[stop]
    )


def best_iteration(booster: "lightgbm.Booster") -> int:
    return booster.best_iteration or booster.current_iteration()


def fit_candidate(train: list[dict], validation: list[dict], config: TrainingConfig) -> Candidate:
    """Train both models on the training split; stop early on validation rows when usable."""
    present = {row["label_resource"] for row in train}
    class_order = tuple(name for name in CLASSES if name in present)
    if len(class_order) < 2:
        raise CandidateError(
            f"Training needs at least two classes; the training split has {list(class_order)}"
        )
    index = {name: position for position, name in enumerate(class_order)}
    # A mixed step has no single class to learn, so the classifier trains on single-class steps
    # only; the regressor learns, and stops early on, every step's excess fraction. Validation
    # rows of a class unseen in training, and mixed ones, cannot be scored by this classifier.
    classified = [row for row in train if row["label_resource"] in index]
    # Weights are computed over each model's own training rows.
    classifier_weights = row_weights(classified, config.weight_by)
    regressor_weights = row_weights(train, config.weight_by)
    usable = [row for row in validation if row["label_resource"] in index]
    held_out = feature_matrix(usable) if usable else None
    features = feature_matrix(train)
    classifier = train_booster(
        FIXED | config.classifier | {"objective": "multiclass", "num_class": len(class_order)},
        feature_matrix(classified),
        class_targets(classified, index),
        None if held_out is None else (held_out, class_targets(usable, index)),
        config,
        classifier_weights,
    )
    regressor = train_booster(
        FIXED | config.regressor | {"objective": "regression_l1"},
        features,
        fraction_targets(train),
        (feature_matrix(validation), fraction_targets(validation)) if validation else None,
        config,
        regressor_weights,
    )
    return Candidate(
        class_order=class_order,
        classifier=classifier,
        regressor=regressor,
        best_iterations={
            "classifier": best_iteration(classifier),
            "regressor": best_iteration(regressor),
        },
        early_stopping="validation" if usable else "not_used",
        regressor_early_stopping="validation" if validation else "not_used",
        training_version=config.version,
        validation_rows_dropped=len(validation) - len(usable),
        train_rows_not_classified=len(train) - len(classified),
        classes_absent_from_train=tuple(name for name in CLASSES if name not in present),
    )


def predict(candidate: Candidate, rows: list[dict]) -> list[Prediction]:
    """Class probabilities, the most probable class and the clipped excess fraction per row."""
    if not rows:
        return []
    features = feature_matrix(rows)
    iterations = candidate.best_iterations
    probabilities = np.asarray(
        candidate.classifier.predict(features, num_iteration=iterations["classifier"])
    )
    fractions = np.clip(
        candidate.regressor.predict(features, num_iteration=iterations["regressor"]), 0.0, 1.0
    )
    predictions = []
    for scores, fraction in zip(probabilities, fractions, strict=True):
        by_class = {
            name: float(score) for name, score in zip(candidate.class_order, scores, strict=True)
        }
        predictions.append(
            Prediction(
                label=candidate.class_order[int(np.argmax(scores))],
                probabilities=by_class,
                excess_fraction=float(fraction),
            )
        )
    return predictions
