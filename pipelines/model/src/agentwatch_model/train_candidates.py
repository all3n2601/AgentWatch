"""Train the LightGBM candidates, compare them with rules-v1 on identical rows, and track the run.

Splits are read only through load_split. A blocked split or any failure during the attempt still
produces a report and an MLflow run, so failed and inconclusive attempts are kept. --explore
skips only the leakage check, scores the validation split, writes to its own directory, and is
never an evaluation.
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

from agentwatch_data.files import file_sha256, json_text, write_atomic
from agentwatch_data.splits import SplitError, load_split

from agentwatch_model.candidates import (
    DEFAULT_TRAINING,
    Candidate,
    Prediction,
    TrainingConfig,
    TrainingError,
    fit_candidate,
    load_training_config,
    predict,
)
from agentwatch_model.comparison import build_model_report
from agentwatch_model.contract import DEFAULT_CONTRACT, Contract, ContractError, load_contract
from agentwatch_model.features import preprocessing_contract
from agentwatch_model.report import blocked_report, dataset_version, identity
from agentwatch_model.rule_baseline import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    DEFAULT_SCHEMA,
    EXIT_BLOCKED,
    EXIT_ERROR,
    split_identity,
    validate_rows,
)
from agentwatch_model.rules import DEFAULT_RULES, Rules, RulesError, load_rules
from agentwatch_model.tracking import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_TRACKING_URI,
    log_run,
    tracking_errors,
)

DEFAULT_OUTPUT = Path(".data/candidates")
EXPLORATORY_DIRECTORY = "exploratory"  # never shares files with an evaluated attempt
TASK = "M03"
REPORT = "report.json"
PREDICTIONS = "predictions.jsonl"
VALIDATION_PREDICTIONS = "validation_predictions.jsonl"
PREPROCESSING = "preprocessing.json"
TRACKING = "tracking.json"
MODEL_FILES = {"classifier": "classifier.txt", "regressor": "regressor.txt"}
ARTIFACTS = (
    *MODEL_FILES.values(),
    PREPROCESSING,
    PREDICTIONS,
    VALIDATION_PREDICTIONS,
    REPORT,
    TRACKING,
)
STOPPED = ("blocked", "failed")
EXIT_UNTRACKED = 3
OBJECTIVES = {"classifier": "multiclass", "regressor": "regression_l1"}
FRACTION_RANGE = (0.0, 1.0)

Loader = Callable[..., list[dict]]


@dataclass(frozen=True)
class Outcome:
    status: str
    run_id: str | None


def attempt_errors() -> tuple[type[BaseException], ...]:
    """Data and training failures that end an attempt as a recorded failed run.

    RowError, FeatureError and CandidateError are ValueErrors. OSError and ImportError cover a
    missing OpenMP runtime; LightGBMError is added when LightGBM itself can load.
    """
    errors: list[type[BaseException]] = [ValueError, OSError, ImportError]
    try:
        from lightgbm.basic import LightGBMError

        errors.append(LightGBMError)
    except (ImportError, OSError):
        pass
    return tuple(errors)


def failed_report(contract: Contract, rules: Rules, model: dict, reason: str) -> dict:
    return identity(contract, rules) | {
        "status": "failed",
        "evaluation": False,
        "reason": reason,
        "model": model,
        "metrics": None,
        "gates": {},
        "summary": {"outcomes": {}, "blocks_promotion": True},
    }


def prediction_records(
    rows: list[dict],
    predictions: list[Prediction],
    rule_labels: list[str],
    split: str,
    evaluation: bool,
) -> str:
    lines = []
    for row, prediction, rule_label in zip(rows, predictions, rule_labels, strict=True):
        record = {
            "run_id": row["run_id"],
            "step_id": row["step_id"],
            "task_id": row["task_id"],
            "condition": row["condition"],
            "split": split,
            "dataset_version": row["dataset_version"],
            "label_version": row["label_version"],
            "label_resource": row["label_resource"],
            "label_excess_fraction": row["label_excess_fraction"],
            "prediction": prediction.label,
            "probabilities": prediction.probabilities,
            "excess_fraction": prediction.excess_fraction,
            "rule_prediction": rule_label,
            "evaluation": evaluation,
        }
        lines.append(json.dumps(record, sort_keys=True, allow_nan=False))
    return "".join(f"{line}\n" for line in lines)


def label_version(rows: list[dict]) -> str | None:
    versions = {row["label_version"] for row in rows}
    return versions.pop() if len(versions) == 1 else None


def clear_artifacts(directory: Path) -> None:
    """Remove an earlier attempt's files so none can be mistaken for this attempt's result."""
    for name in ARTIFACTS:
        (directory / name).unlink(missing_ok=True)


def save_models(candidate: Candidate, directory: Path) -> dict[str, str]:
    """Save each model atomically at its best iteration; return the files' SHA-256."""
    directory.mkdir(parents=True, exist_ok=True)
    boosters = {"classifier": candidate.classifier, "regressor": candidate.regressor}
    hashes = {}
    for key, booster in boosters.items():
        target = directory / MODEL_FILES[key]
        descriptor, staging = tempfile.mkstemp(prefix=f".{target.name}.", dir=directory)
        os.close(descriptor)
        booster.save_model(staging, num_iteration=candidate.best_iterations[key])
        os.replace(staging, target)
        hashes[key] = file_sha256(target)
    return hashes


def model_identity(config: TrainingConfig, candidate: Candidate, labels: str | None) -> dict:
    return {
        "version": config.version,
        "label_version": labels,
        "class_order": list(candidate.class_order),
        "classes_absent_from_train": list(candidate.classes_absent_from_train),
        "validation_rows_dropped": candidate.validation_rows_dropped,
        "best_iterations": candidate.best_iterations,
        "early_stopping": candidate.early_stopping,
        "settings": {
            "num_boost_round": config.num_boost_round,
            "early_stopping_rounds": config.early_stopping_rounds,
            "classifier": config.classifier,
            "regressor": config.regressor,
        },
        "lightgbm_version": version("lightgbm"),
        "numpy_version": version("numpy"),
    }


def serving_contract(candidate: Candidate, model: dict, report: dict, hashes: dict) -> dict:
    lineage = {
        "dataset_version": report["dataset_version"],
        "label_version": model["label_version"],
        "status": report["status"],
        "evaluation": report["evaluation"],
        "model_files": {key: {"file": MODEL_FILES[key], "sha256": h} for key, h in hashes.items()},
        "objectives": OBJECTIVES,
        "best_iterations": candidate.best_iterations,
        "probability_columns": list(candidate.class_order),
        "excess_fraction_clip": list(FRACTION_RANGE),
        "lightgbm_version": model["lightgbm_version"],
    }
    return preprocessing_contract(candidate.class_order, candidate.training_version, lineage)


def attempt(
    contract: Contract,
    rules: Rules,
    config: TrainingConfig,
    splits: dict[str, list[dict]],
    scored_split: str,
    directory: Path,
    *,
    evaluation: bool,
    resamples: int,
    seed: int,
) -> dict:
    """Validate, train, score and save one attempt; return its report."""
    train, validation, scored = splits["train"], splits["validation"], splits[scored_split]
    for name, rows in splits.items():
        validate_rows(rows, name)
    dataset_version(train + validation + scored)  # refuses rows from mixed releases
    candidate = fit_candidate(train, validation, config)
    model = model_identity(config, candidate, label_version(train + validation + scored))
    predictions = predict(candidate, scored)
    rule_labels = [rules.predict(row) for row in scored]
    report = build_model_report(
        contract,
        rules,
        model,
        train,
        scored,
        [p.label for p in predictions],
        [p.excess_fraction for p in predictions],
        rule_labels,
        evaluation=evaluation,
        resamples=resamples,
        seed=seed,
    )
    records = prediction_records(scored, predictions, rule_labels, scored_split, evaluation)
    hashes = save_models(candidate, directory)
    write_atomic(
        directory / PREPROCESSING, json_text(serving_contract(candidate, model, report, hashes))
    )
    write_atomic(directory / PREDICTIONS, records)
    if scored_split != "validation":
        # Probabilities for M04 calibration. Validation also drove early stopping, so
        # calibration fitted on it is slightly optimistic; M04 records that.
        validation_predictions = predict(candidate, validation)
        write_atomic(
            directory / VALIDATION_PREDICTIONS,
            prediction_records(
                validation,
                validation_predictions,
                [rules.predict(row) for row in validation],
                "validation",
                evaluation,
            ),
        )
    return report | {
        "scored_steps": len(scored),
        "predictions": {
            "file": PREDICTIONS,
            "sha256": hashlib.sha256(records.encode()).hexdigest(),
        },
        "model_files": hashes,
    }


def finish(report: dict, directory: Path, tracking_uri: str, artifact_root: Path) -> Outcome:
    # The report is written last: its presence marks a complete attempt.
    write_atomic(directory / REPORT, json_text(report))
    try:
        run_id = log_run(
            report, directory, tracking_uri=tracking_uri, artifact_root=artifact_root, task=TASK
        )
    except tracking_errors() as error:
        write_atomic(directory / TRACKING, json_text({"logged": False, "error": str(error)}))
        print(f"Tracking error, report kept: {error}", file=sys.stderr)
        return Outcome(report["status"], None)
    write_atomic(directory / TRACKING, json_text({"logged": True, "run_id": run_id}))
    return Outcome(report["status"], run_id)


def run(
    split: Path,
    output: Path,
    *,
    contract_path: Path = DEFAULT_CONTRACT,
    frozen_schema: Path = DEFAULT_SCHEMA,
    rules_path: Path = DEFAULT_RULES,
    training_path: Path = DEFAULT_TRAINING,
    explore: bool = False,
    loader: Loader | None = None,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> Outcome:
    """Train, score and track one attempt."""
    contract, rules = load_contract(contract_path), load_rules(rules_path)
    config = load_training_config(training_path)
    scored_split = "validation" if explore else "test"
    directory = output / EXPLORATORY_DIRECTORY if explore else output
    directory.mkdir(parents=True, exist_ok=True)
    clear_artifacts(directory)
    base = {"split": split_identity(split), "scored_split": scored_split}
    # Resolved at call time, so tests can replace the module's load_split.
    load = loader or load_split
    try:
        splits = {
            name: load(split, name, contract_path, frozen_schema, allow_unchecked_leakage=explore)
            for name in ("train", "validation", scored_split)
        }
    except SplitError as error:
        report = blocked_report(contract, rules, str(error)) | {
            "model": {"version": config.version}
        }
        return finish(report | base, directory, tracking_uri, artifact_root)
    try:
        report = attempt(
            contract,
            rules,
            config,
            splits,
            scored_split,
            directory,
            evaluation=not explore,
            resamples=resamples,
            seed=seed,
        )
    except attempt_errors() as error:
        clear_artifacts(directory)
        model = {"version": config.version}
        reason = f"{type(error).__name__}: {error}"
        report = failed_report(contract, rules, model, reason)
    return finish(report | base, directory, tracking_uri, artifact_root)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LightGBM candidates and compare them with rules-v1 on identical rows."
    )
    parser.add_argument("--split", type=Path, required=True, help="Frozen split file")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--frozen-schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--training", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument(
        "--explore",
        action="store_true",
        help="Skip only the leakage check and score the validation split; not an evaluation",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_errors = (ContractError, RulesError, TrainingError, tomllib.TOMLDecodeError, OSError)
    try:
        outcome = run(
            args.split,
            args.output,
            contract_path=args.contract,
            frozen_schema=args.frozen_schema,
            rules_path=args.rules,
            training_path=args.training,
            explore=args.explore,
            tracking_uri=args.tracking_uri,
            artifact_root=args.artifact_root,
        )
    except config_errors as error:
        print(f"Training error: {error}", file=sys.stderr)
        return EXIT_ERROR
    print(f"Candidates {outcome.status}; MLflow run {outcome.run_id or 'not logged'}")
    if outcome.status in STOPPED:
        return EXIT_BLOCKED
    return 0 if outcome.run_id else EXIT_UNTRACKED


if __name__ == "__main__":
    sys.exit(main())
