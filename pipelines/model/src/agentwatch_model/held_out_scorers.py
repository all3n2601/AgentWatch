"""Score held-out capacity-sweep rows with rules-v1 and the calibrated LightGBM classifier.

The classifier is trained on the release's train split and calibrated on its validation split;
the release's test split is never loaded here (it mixes two feature definitions). Held-out rows
carry no labels, so they are only scored.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from agentwatch_data.replay_labels import task_group
from agentwatch_data.splits import load_split

from agentwatch_model.calibrate import fit_on_validation, score_test
from agentwatch_model.calibration import CalibrationConfig
from agentwatch_model.candidates import (
    Candidate,
    TrainingConfig,
    check_lightgbm,
    fit_candidate,
    predict,
)
from agentwatch_model.contract import Contract
from agentwatch_model.report import dataset_version
from agentwatch_model.rule_baseline import validate_dataset_version, validate_rows
from agentwatch_model.rules import Rules
from agentwatch_model.train_candidates import label_version

RULES_SCORER = "rules-v1"
CLASSIFIER_SCORER = "lightgbm-classifier-training-v1"
# The only splits this command reads; "test" is deliberately absent.
TRAINING_SPLITS = ("train", "validation")

Loader = Callable[..., list[dict]]


@dataclass(frozen=True)
class ScoredStep:
    scorer: str
    label: str
    confidence: float | None
    abstained: bool


def rule_scores(rules: Rules, rows: list[dict]) -> list[ScoredStep]:
    """rules-v1 labels; it has no confidence and never abstains."""
    return [ScoredStep(RULES_SCORER, rules.predict(row), None, False) for row in rows]


def seen_task_groups(train: list[dict]) -> frozenset[str]:
    """Task groups the classifier trained on; validation groups are not training exposure."""
    return frozenset(task_group(row["task_id"]) for row in train)


def train_classifier(
    split: Path,
    contract_path: Path,
    frozen_schema: Path,
    config: TrainingConfig,
    loader: Loader = load_split,
) -> tuple[Candidate, list[dict], frozenset[str], dict]:
    """Fit on train with validation early stopping; return the candidate, validation rows, the
    train task groups and the release identity (dataset and label version).

    The identity comes from the train rows: a release's validation split can be empty, and
    validate_dataset_version has already required one version across both splits.
    """
    check_lightgbm()
    train, validation = (
        loader(split, name, contract_path, frozen_schema) for name in TRAINING_SPLITS
    )
    validate_rows(train, "train")
    validate_rows(validation, "validation")
    validate_dataset_version(train + validation)
    release = {
        "dataset_version": dataset_version(train),
        "label_version": label_version(train),
    }
    return fit_candidate(train, validation, config), validation, seen_task_groups(train), release


def classifier_scores(
    candidate: Candidate,
    validation: list[dict],
    rows: list[dict],
    contract: Contract,
    calibration: CalibrationConfig,
) -> tuple[list[ScoredStep], dict]:
    """Calibrated labels for rows, abstaining below the threshold chosen on validation.

    When too few validation steps remain to fit, confidence is None, nothing abstains, and the
    returned record says abstention is unavailable and why.
    """
    columns = candidate.class_order

    def records(batch: list[dict]) -> list[dict]:
        return [
            {"prediction": p.label, "probabilities": p.probabilities}
            for p in predict(candidate, batch)
        ]

    fit = fit_on_validation(
        validation, records(validation), columns, contract, calibration, candidate.early_stopping
    )
    scored = score_test(rows, records(rows), columns, fit)
    steps = [
        ScoredStep(
            CLASSIFIER_SCORER,
            record["prediction"],
            confidence if fit.fitted else None,
            abstained,
        )
        for record, confidence, abstained in zip(
            scored.records, scored.confidences, scored.abstained, strict=True
        )
    ]
    summary = {
        "fitted": fit.fitted,
        "temperature": fit.temperature,
        "threshold": fit.threshold,
        "reason": fit.reason,
    }
    if not fit.fitted:
        summary["abstention"] = "unavailable"
    return steps, summary
