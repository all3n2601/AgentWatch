"""LightGBM attribution candidates: a resource classifier and an excess-latency regressor.

Both train on the training split only; the validation split is used only for early stopping.
Settings come from training.toml (training-v1). LightGBM is imported where it is used, so the
configuration and errors stay importable on hosts where its OpenMP runtime is missing.
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from agentwatch_data.extract import canonical_sha256

from agentwatch_model.features import FEATURES, feature_matrix
from agentwatch_model.metrics import CLASSES, is_number

if TYPE_CHECKING:
    import lightgbm

DEFAULT_TRAINING = Path(__file__).with_name("training.toml")
VERSION = "training-v1"
HASH_DIGITS = 8  # matches the rules and contract identifiers
TUNABLE = ("learning_rate", "num_leaves", "min_data_in_leaf", "feature_fraction", "lambda_l2")
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
    early_stopping: str
    training_version: str
    # Coverage gaps, reported rather than hidden: validation rows of classes unseen in training
    # cannot be scored for early stopping, and absent classes can never be predicted.
    validation_rows_dropped: int = 0
    classes_absent_from_train: tuple[str, ...] = ()


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
    if parsed.get("version") != VERSION:
        raise TrainingError(f"{path.name}: version must be {VERSION!r}")
    # Validate before hashing: canonical JSON refuses NaN, and the error should name the value.
    rounds = positive_int(parsed, "num_boost_round")
    patience = positive_int(parsed, "early_stopping_rounds")
    classifier = booster_settings(parsed, "classifier")
    regressor = booster_settings(parsed, "regressor")
    return TrainingConfig(
        version=f"{VERSION}+{canonical_sha256(parsed)[:HASH_DIGITS]}",
        num_boost_round=rounds,
        early_stopping_rounds=patience,
        classifier=classifier,
        regressor=regressor,
    )


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
) -> "lightgbm.Booster":
    import lightgbm

    data = lightgbm.Dataset(features, target, feature_name=list(FEATURES), free_raw_data=False)
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
    # Validation rows of a class unseen in training cannot be scored by this classifier.
    usable = [row for row in validation if row["label_resource"] in index]
    held_out = feature_matrix(usable) if usable else None
    features = feature_matrix(train)
    classifier = train_booster(
        FIXED | config.classifier | {"objective": "multiclass", "num_class": len(class_order)},
        features,
        class_targets(train, index),
        None if held_out is None else (held_out, class_targets(usable, index)),
        config,
    )
    regressor = train_booster(
        FIXED | config.regressor | {"objective": "regression_l1"},
        features,
        fraction_targets(train),
        None if held_out is None else (held_out, fraction_targets(usable)),
        config,
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
        training_version=config.version,
        validation_rows_dropped=len(validation) - len(usable),
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
