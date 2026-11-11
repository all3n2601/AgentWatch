"""Calibrated confidence and abstention for the LightGBM classifier, fitted on validation only.

Temperature scaling divides the log class scores by one fitted temperature, so confidence can
be corrected without changing which class is predicted. A step abstains when its calibrated
top-label confidence is below the threshold chosen on validation. The contract scores an
abstention as a miss for the step's true class, never as a prediction of any class.
"""

import math
import tomllib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from agentwatch_data.extract import canonical_sha256

from agentwatch_model.contract import ABSTENTION_SCOPES
from agentwatch_model.metrics import f1_score, is_number, macro_f1

DEFAULT_CALIBRATION = Path(__file__).with_name("calibration.toml")
VERSION = "calibration-v1"
HASH_DIGITS = 8  # matches the rules, training and contract identifiers
# Log of a zero probability would be minus infinity; this floor keeps scaling finite.
PROBABILITY_FLOOR = 1e-12
GOLDEN_RATIO = (math.sqrt(5) - 1) / 2


class CalibrationError(ValueError):
    """A calibration configuration or input that cannot produce a reproducible calibration."""


@dataclass(frozen=True)
class CalibrationConfig:
    version: str
    min_temperature: float
    max_temperature: float
    tolerance: float
    min_validation_steps: int


def positive_number(parsed: dict, key: str) -> float:
    value = parsed.get(key)
    if not is_number(value) or value <= 0:
        raise CalibrationError(f"Calibration {key} must be a positive number")
    return float(value)


def load_calibration_config(path: Path = DEFAULT_CALIBRATION) -> CalibrationConfig:
    parsed = tomllib.loads(path.read_text())
    if parsed.get("version") != VERSION:
        raise CalibrationError(f"Calibration version must be {VERSION!r}")
    low = positive_number(parsed, "min_temperature")
    high = positive_number(parsed, "max_temperature")
    if high <= low:
        raise CalibrationError("Calibration max_temperature must be greater than min_temperature")
    steps = parsed.get("min_validation_steps")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise CalibrationError("Calibration min_validation_steps must be a positive integer")
    return CalibrationConfig(
        version=f"{VERSION}+{canonical_sha256(parsed)[:HASH_DIGITS]}",
        min_temperature=low,
        max_temperature=high,
        tolerance=positive_number(parsed, "tolerance"),
        min_validation_steps=steps,
    )


def scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Class probabilities after dividing their log scores by the temperature."""
    logits = np.log(np.clip(probabilities, PROBABILITY_FLOOR, 1.0)) / temperature
    shifted = np.exp(logits - logits.max(axis=1, keepdims=True))
    return shifted / shifted.sum(axis=1, keepdims=True)


def log_loss(probabilities: np.ndarray, targets: np.ndarray, temperature: float) -> float:
    scaled = scale(probabilities, temperature)
    chosen = scaled[np.arange(len(targets)), targets]
    return float(-np.log(np.clip(chosen, PROBABILITY_FLOOR, 1.0)).mean())


def fit_temperature(
    probabilities: np.ndarray, targets: np.ndarray, config: CalibrationConfig
) -> float:
    """The temperature within the configured bounds that minimizes validation log loss.

    Log loss is convex in the inverse temperature (a log-sum-exp minus a linear term), so a
    golden-section search over the inverse temperature finds the bounded minimum.
    """
    low, high = 1 / config.max_temperature, 1 / config.min_temperature

    def loss(inverse: float) -> float:
        return log_loss(probabilities, targets, 1 / inverse)

    left = high - GOLDEN_RATIO * (high - low)
    right = low + GOLDEN_RATIO * (high - low)
    left_loss, right_loss = loss(left), loss(right)
    while high - low > config.tolerance:
        if left_loss <= right_loss:
            high, right, right_loss = right, left, left_loss
            left = high - GOLDEN_RATIO * (high - low)
            left_loss = loss(left)
        else:
            low, left, left_loss = left, right, right_loss
            right = low + GOLDEN_RATIO * (high - low)
            right_loss = loss(right)
    return 1 / ((low + high) / 2)


def top_label_ece(confidences: list[float], correct: list[bool], bins: int) -> float | None:
    """Expected calibration error of the predicted class over equal-width confidence bins."""
    if not confidences:
        return None
    groups: dict[int, list[tuple[float, bool]]] = {}
    for confidence, hit in zip(confidences, correct, strict=True):
        # A confidence of exactly 1.0 belongs in the last bin, not a bin of its own.
        groups.setdefault(min(int(confidence * bins), bins - 1), []).append((confidence, hit))
    # Each bin's weight times its accuracy-confidence gap is |hits - total confidence| / steps.
    gaps = [
        abs(sum(hit for _, hit in members) - sum(confidence for confidence, _ in members))
        for members in groups.values()
    ]
    return sum(gaps) / len(confidences)


def abstaining_scores(
    labels: list[str], predictions: list[str | None], classes: tuple[str, ...]
) -> dict[str, dict]:
    """Per-class scores where None is an abstention: a miss for its true class only."""
    support = Counter(labels)
    predicted = Counter(p for p in predictions if p is not None)
    hits = Counter(label for label, p in zip(labels, predictions, strict=True) if label == p)
    scores = {}
    for name in classes:
        precision = hits[name] / predicted[name] if predicted[name] else None
        recall = hits[name] / support[name] if support[name] else None
        scores[name] = {
            "support": support[name],
            "predicted": predicted[name],
            "precision": precision,
            "recall": recall,
            "f1": f1_score(hits[name], precision, recall),
        }
    return scores


def rate(abstained: list[bool]) -> dict:
    count = sum(abstained)
    return {
        "steps": len(abstained),
        "abstained": count,
        "rate": count / len(abstained) if abstained else None,
    }


def abstention_rates(labels: list[str], abstained: list[bool], classes: tuple[str, ...]) -> dict:
    """Share of steps abstained overall and per true class; None where a class has no steps."""
    return {
        "overall": rate(abstained),
        "per_true_class": {
            name: rate([a for label, a in zip(labels, abstained, strict=True) if label == name])
            for name in classes
        },
    }


def within_limit(scopes: list[dict], max_rate: float) -> bool:
    return all(scope["rate"] is None or scope["rate"] <= max_rate for scope in scopes)


def choose_threshold(
    labels: list[str],
    predictions: list[str],
    confidences: list[float],
    classes: tuple[str, ...],
    max_rate: float,
    mixed: list[bool] | None = None,
    scopes: tuple[str, ...] = ABSTENTION_SCOPES,
) -> dict:
    """The confidence threshold with the best validation macro F1 within the abstention limit.

    Candidates are zero (never abstain) and every distinct validation confidence; a step
    abstains when its confidence is below the threshold. The limit applies to each of `scopes`
    (the contract's abstention_rate_scopes): overall, per true class on single-class steps, and
    mixed_contention on the steps `mixed` marks. Macro F1 is over single-class steps only
    (mixed-scoring-v1) and the classes with a defined F1, as validation can lack classes. Ties
    keep the lowest threshold, so abstention is added only when it measurably helps. Threshold
    zero abstains on nothing, so it is always within the limit.
    """
    flags = mixed or [False] * len(labels)
    single = [index for index, flag in enumerate(flags) if not flag]
    single_labels = [labels[index] for index in single]

    def measure(abstained: list[bool]) -> dict:
        return {
            "overall": rate(abstained),
            "mixed": rate([a for a, flag in zip(abstained, flags, strict=True) if flag]),
            "per_true_class": abstention_rates(
                single_labels, [abstained[index] for index in single], classes
            )["per_true_class"],
        }

    def limited(measured: dict) -> list[dict]:
        rates = {
            "overall": [measured["overall"]],
            "per_true_class": list(measured["per_true_class"].values()),
            "mixed_contention": [measured["mixed"]],
        }
        return [scope_rate for scope in scopes for scope_rate in rates[scope]]

    best = {"threshold": 0.0, "validation_macro_f1_defined_classes": None}
    best_value = -math.inf
    for threshold in [0.0, *sorted(set(confidences))]:
        abstained = [confidence < threshold for confidence in confidences]
        if not within_limit(limited(measure(abstained)), max_rate):
            continue
        kept = [None if abstained[index] else predictions[index] for index in single]
        value = macro_f1(abstaining_scores(single_labels, kept, classes))
        if value is not None and value > best_value:
            best_value = value
            best = {"threshold": threshold, "validation_macro_f1_defined_classes": value}
    chosen = measure([confidence < best["threshold"] for confidence in confidences])
    return best | {
        "validation_abstention_rate": chosen["overall"]["rate"],
        "validation_mixed_abstention_rate": chosen["mixed"]["rate"],
        "max_abstention_rate": max_rate,
    }
