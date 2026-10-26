"""Classification and regression metrics for attribution, with undefined values kept null.

A value that cannot be measured (precision of a class never predicted, F1 of a class absent
from the test split) is None, never zero, so reports cannot read a missing measurement as a
measured failure or success.
"""

import math
import random
from typing import TypeGuard

from agentwatch_data.schema import RESOURCE_CLASSES

CLASSES = RESOURCE_CLASSES
NONE = "none"
MIN_BOOTSTRAP_GROUPS = 2  # fewer groups cannot show how results vary across tasks


def is_number(value: object) -> TypeGuard[int | float]:
    """A finite int or float; booleans are not numbers here."""
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def confusion_matrix(labels: list[str], predictions: list[str]) -> dict[str, dict[str, int]]:
    """Counts of (true class -> predicted class) over every contract class."""
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must have the same length")
    matrix = {truth: dict.fromkeys(CLASSES, 0) for truth in CLASSES}
    for truth, predicted in zip(labels, predictions, strict=True):
        if truth not in matrix or predicted not in matrix:
            raise ValueError(f"Unknown class in ({truth!r}, {predicted!r})")
        matrix[truth][predicted] += 1
    return matrix


def class_scores(matrix: dict[str, dict[str, int]]) -> dict[str, dict]:
    """Support, predicted count, precision, recall and F1 per class."""
    scores = {}
    for name in CLASSES:
        hits = matrix[name][name]
        support = sum(matrix[name].values())
        predicted = sum(row[name] for row in matrix.values())
        precision = hits / predicted if predicted else None
        recall = hits / support if support else None
        scores[name] = {
            "support": support,
            "predicted": predicted,
            "precision": precision,
            "recall": recall,
            "f1": f1_score(hits, precision, recall),
        }
    return scores


def f1_score(hits: int, precision: float | None, recall: float | None) -> float | None:
    if recall is None:
        # Absent from the split: F1 is undefined even when the class was predicted, so false
        # positives on it count against precision elsewhere, not as a measured zero here.
        return None
    if hits == 0 or precision is None:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def macro_f1(scores: dict[str, dict]) -> float | None:
    """Unweighted mean F1 over the classes whose F1 is defined."""
    defined = [score["f1"] for score in scores.values() if score["f1"] is not None]
    return sum(defined) / len(defined) if defined else None


def macro_f1_all_classes(scores: dict[str, dict], classes: tuple[str, ...]) -> float | None:
    """The contract's macro F1 over every class; None while any class's F1 is undefined."""
    values = [scores[name]["f1"] for name in classes]
    if any(value is None for value in values):
        return None
    return sum(values) / len(values)


def none_rate(rows: list[dict], predictions: list[str], conditions: set[str]) -> dict:
    """Share of steps in the given conditions predicted none; any other answer is a miss."""
    selected = [p for r, p in zip(rows, predictions, strict=True) if r["condition"] in conditions]
    hits = sum(1 for prediction in selected if prediction == NONE)
    return {
        "steps": len(selected),
        "none": hits,
        "rate": hits / len(selected) if selected else None,
    }


def condition_slices(rows: list[dict], predictions: list[str]) -> dict[str, dict]:
    """Step count and confusion matrix for each experiment condition present."""
    slices = {}
    for condition in sorted({row["condition"] for row in rows}):
        pairs = [
            (r["label_resource"], p)
            for r, p in zip(rows, predictions, strict=True)
            if r["condition"] == condition
        ]
        labels, predicted = [label for label, _ in pairs], [p for _, p in pairs]
        slices[condition] = {
            "steps": len(pairs),
            "confusion_matrix": confusion_matrix(labels, predicted),
        }
    return slices


def excess_fractions(rows: list[dict]) -> list[float]:
    values = [row["label_excess_fraction"] for row in rows]
    if not all(is_number(value) for value in values):
        raise ValueError("label_excess_fraction must be a finite number on every row")
    return values


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def constant_reference_mae(train: list[dict], test: list[dict]) -> dict:
    """MAE of the two constant predictors the contract requires a model to beat."""
    reference = mean(excess_fractions(train))
    truth = excess_fractions(test)
    return {
        "training_mean": reference,
        "mae_training_mean": (
            mean([abs(v - reference) for v in truth]) if reference is not None and truth else None
        ),
        "mae_zero": mean([abs(v) for v in truth]) if truth else None,
    }


def bootstrap_macro_f1(
    rows: list[dict],
    predictions: list[str],
    resamples: int,
    seed: int,
    level: float = 0.95,
    min_groups: int = MIN_BOOTSTRAP_GROUPS,
) -> dict:
    """Percentile interval for defined-class macro F1, resampling whole tasks together."""
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(row["task_id"], []).append(index)
    interval = {
        "level": level,
        "low": None,
        "high": None,
        "resamples": resamples,
        "valid_resamples": 0,
        "seed": seed,
        "groups": len(groups),
        "min_groups": min_groups,
    }
    if len(groups) < max(min_groups, MIN_BOOTSTRAP_GROUPS):
        return interval
    keys = sorted(groups)
    generator = random.Random(seed)
    values = []
    for _ in range(resamples):
        drawn = [i for key in generator.choices(keys, k=len(keys)) for i in groups[key]]
        labels = [rows[i]["label_resource"] for i in drawn]
        value = macro_f1(class_scores(confusion_matrix(labels, [predictions[i] for i in drawn])))
        if value is not None:
            values.append(value)
    interval["valid_resamples"] = len(values)
    if not values:
        return interval
    values.sort()
    tail = (1 - level) / 2
    interval["low"] = values[math.floor(tail * len(values))]
    interval["high"] = values[math.ceil((1 - tail) * len(values)) - 1]
    return interval
