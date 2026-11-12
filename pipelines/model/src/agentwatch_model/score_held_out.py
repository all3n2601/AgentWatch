"""Score rules-v1 and the calibrated classifier on frozen held-out capacity-sweep truths.

    python -m agentwatch_model.score_held_out --truth DIR [DIR ...] --sweep-root DIR \
        --split FILE --output DIR

Each attempt writes `<output>/<attempt id>/` with report.json, predictions.jsonl and models/,
and is logged to the MLflow experiment `a06-held-out-scoring`, failed attempts included. The
attempt id is the UTC start time plus the first digits of the hash of every input, so it names
both when and on what the attempt ran. Outputs are scores, not run evidence: they never belong in
`make add-evidence`, `make dataset` or `data/held-out/`.
"""

import argparse
import json
import sys
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from agentwatch_data.extract import canonical_sha256
from agentwatch_data.files import file_sha256, json_text, write_atomic
from agentwatch_data.replay_evidence import EVIDENCE_FILE, JOURNAL_FILE
from agentwatch_data.replay_rows import FeatureJoinError
from agentwatch_data.splits import SplitError, load_split

from agentwatch_model.calibration import (
    DEFAULT_CALIBRATION,
    CalibrationConfig,
    CalibrationError,
    load_calibration_config,
)
from agentwatch_model.candidates import (
    DEFAULT_TRAINING,
    TrainingError,
    check_lightgbm,
)
from agentwatch_model.contract import DEFAULT_CONTRACT, ContractError
from agentwatch_model.held_out_report import build_report
from agentwatch_model.held_out_rows import (
    HeldOutError,
    held_out_rows,
    load_truths,
    run_directory,
)
from agentwatch_model.held_out_scorers import (
    CLASSIFIER_SCORER,
    RULES_SCORER,
    ScoredStep,
    classifier_scores,
    rule_scores,
    train_classifier,
)
from agentwatch_model.report import identity
from agentwatch_model.rule_baseline import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    DEFAULT_SCHEMA,
    EXIT_ERROR,
    split_identity,
)
from agentwatch_model.rules import DEFAULT_RULES, RulesError
from agentwatch_model.tracking import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_TRACKING_URI,
    log_run,
    tracking_errors,
)
from agentwatch_model.train_candidates import (
    Loader,
    attempt_errors,
    failed_report,
    load_configuration,
    model_identity,
    save_models,
)

DEFAULT_OUTPUT = Path(".data/a06-scoring/scores")
EXPERIMENT = "a06-held-out-scoring"
TASK = "A06"
REPORT = "report.json"
PREDICTIONS = "predictions.jsonl"
TRACKING = "tracking.json"
MODELS = "models"
LOGGED_ARTIFACTS = (REPORT, PREDICTIONS)
FORBIDDEN_OUTPUT = ("data", "held-out")
ATTEMPT_ID_HASH_DIGITS = 6
EXIT_FAILED = 1
EXIT_UNTRACKED = 3
TRUTH_FILES = ("truth.json", "truth.jsonl")


class OutputError(ValueError):
    """An output location that must not receive scores."""


@dataclass(frozen=True)
class Outcome:
    status: str
    directory: Path
    run_id: str | None


def check_output(output: Path) -> None:
    """Refuse any path inside a `data/held-out` folder; parts are compared case-insensitively
    so Windows spellings are covered, and the resolved path catches `..` and symlinks."""
    for candidate in (output, output.resolve()):
        parts = [part.lower() for part in candidate.parts]
        for index in range(len(parts) - 1):
            if tuple(parts[index : index + 2]) == FORBIDDEN_OUTPUT:
                raise OutputError(f"{output} is inside data/held-out; scores are never evidence")


def hash_of(path: Path) -> str | None:
    return file_sha256(path) if path.is_file() else None


def input_hashes(truth_dirs: list[Path], sweep_root: Path, split: Path) -> dict:
    """SHA-256 of the split, every truth file and each run's evidence and journal.

    Truth files are keyed by the sweep folder their summary names, or by the truth folder as
    given when it cannot be read; a key already taken gets the folder's argument position, so
    no two truth folders share a key. A truth folder that cannot be read contributes only its
    own files; the attempt itself then fails on it and records why. A sweep folder named twice
    is hashed once (its run files are the same files). A missing file is None, so it stays
    distinct from any hash.
    """
    hashes: dict = {"split": hash_of(split), "truth": {}, "runs": {}}
    labels: set[str] = set()
    hashed_sweeps: set[str] = set()
    for position, truth_dir in enumerate(truth_dirs):
        try:
            summary, records = load_truths(truth_dir)
        except HeldOutError:
            summary, records = None, []
        label = summary["sweep_folder"] if summary else truth_dir.as_posix()
        if label in labels:
            label = f"{label} (truth argument {position})"
        labels.add(label)
        for name in TRUTH_FILES:
            hashes["truth"][f"{label}/{name}"] = hash_of(truth_dir / name)
        if summary is None or summary["sweep_folder"] in hashed_sweeps:
            continue
        hashed_sweeps.add(summary["sweep_folder"])
        sweep_dir = sweep_root / summary["sweep_folder"]
        for run_folder in sorted({record.get("run_folder") for record in records}, key=str):
            try:
                run_dir = run_directory(sweep_dir, run_folder)
            except FeatureJoinError:
                continue
            for name in (EVIDENCE_FILE, JOURNAL_FILE):
                hashes["runs"][f"{summary['sweep_folder']}/{run_folder}/{name}"] = hash_of(
                    run_dir / name
                )
    return hashes


def attempt_id(hashes: dict, started: datetime) -> str:
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{canonical_sha256(hashes)[:ATTEMPT_ID_HASH_DIGITS]}"


def prediction_lines(rows: list[dict], scores: dict[str, list[ScoredStep]]) -> str:
    lines = []
    for scored in scores.values():
        for row, score in zip(rows, scored, strict=True):
            record = {
                "run_id": row["run_id"],
                "step_id": row["step_id"],
                "scorer": score.scorer,
                "label": score.label,
                "confidence": score.confidence,
                "abstained": score.abstained,
                "truths": row["truths"],
            }
            lines.append(json.dumps(record, sort_keys=True, allow_nan=False))
    return "".join(f"{line}\n" for line in lines)


def score_rows(
    truth_dirs: list[Path],
    sweep_root: Path,
    split: Path,
    directory: Path,
    *,
    contract_path: Path,
    frozen_schema: Path,
    rules_path: Path,
    training_path: Path,
    calibration: CalibrationConfig,
    loader: Loader,
    resamples: int,
    seed: int,
) -> dict:
    """Build rows, train, score with both scorers and save the models; return the report body."""
    contract, rules, config = load_configuration(contract_path, rules_path, training_path)
    rows, refused, summaries, _ = held_out_rows(truth_dirs, sweep_root)
    if not rows:
        raise HeldOutError(f"no step could be scored; refused steps by code: {refused}")
    candidate, validation, seen_groups, release = train_classifier(
        split, contract_path, frozen_schema, config, loader
    )
    classifier, fit = classifier_scores(candidate, validation, rows, contract, calibration)
    scores = {RULES_SCORER: rule_scores(rules, rows), CLASSIFIER_SCORER: classifier}
    model_hashes = save_models(candidate, directory / MODELS)
    model = model_identity(config, candidate, release["label_version"]) | {
        "training_config_sha256": file_sha256(training_path),
        "calibration": fit,
        "calibration_version": calibration.version,
        "model_files": model_hashes,
        "split": split_identity(split),
        "dataset_version": release["dataset_version"],
    }
    write_atomic(directory / PREDICTIONS, prediction_lines(rows, scores))
    body = build_report(
        rows,
        scores,
        {
            "seen_task_groups": seen_groups,
            "refused": refused,
            "summaries": summaries,
            "model": model,
            "resamples": resamples,
            "seed": seed,
            "abstention_unavailable": fit.get("abstention") == "unavailable",
        },
    )
    return (
        identity(contract, rules)
        | body
        | {
            "status": "completed",
            "evaluation": False,
            "dataset_version": model["dataset_version"],
            "scored_split": "held-out",
            "split": model["split"],
            "scored_steps": len(rows),
        }
    )


def finish(report: dict, directory: Path, tracking_uri: str, artifact_root: Path) -> Outcome:
    # The report is written last: its presence marks a complete attempt.
    write_atomic(directory / REPORT, json_text(report))
    try:
        run_id = log_run(
            report,
            directory,
            artifacts=LOGGED_ARTIFACTS,
            tracking_uri=tracking_uri,
            artifact_root=artifact_root,
            task=TASK,
            experiment=EXPERIMENT,
        )
    except tracking_errors() as error:
        write_atomic(directory / TRACKING, json_text({"logged": False, "error": str(error)}))
        print(f"Tracking error, report kept: {error}", file=sys.stderr)
        return Outcome(report["status"], directory, None)
    write_atomic(directory / TRACKING, json_text({"logged": True, "run_id": run_id}))
    return Outcome(report["status"], directory, run_id)


def run_attempt(
    truth_dirs: list[Path],
    sweep_root: Path,
    split: Path,
    output: Path,
    *,
    contract_path: Path = DEFAULT_CONTRACT,
    frozen_schema: Path = DEFAULT_SCHEMA,
    rules_path: Path = DEFAULT_RULES,
    training_path: Path = DEFAULT_TRAINING,
    calibration_path: Path = DEFAULT_CALIBRATION,
    loader: Loader | None = None,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> Outcome:
    """One scoring attempt; data and training failures end as a recorded failed attempt."""
    check_output(output)
    contract, rules, config = load_configuration(contract_path, rules_path, training_path)
    calibration = load_calibration_config(calibration_path)
    # An environment that cannot train is a setup error, not a failed attempt.
    check_lightgbm()
    recorded_failures = (*attempt_errors(), HeldOutError, SplitError, CalibrationError)
    hashes = input_hashes(truth_dirs, sweep_root, split)
    directory = output / attempt_id(hashes, datetime.now(UTC))
    directory.mkdir(parents=True, exist_ok=True)
    # A rerun with identical inputs in the same second reuses the folder; drop its old files.
    for name in (REPORT, PREDICTIONS, TRACKING):
        (directory / name).unlink(missing_ok=True)
    try:
        report = score_rows(
            truth_dirs,
            sweep_root,
            split,
            directory,
            contract_path=contract_path,
            frozen_schema=frozen_schema,
            rules_path=rules_path,
            training_path=training_path,
            calibration=calibration,
            loader=loader or load_split,
            resamples=resamples,
            seed=seed,
        )
    except recorded_failures as error:
        (directory / PREDICTIONS).unlink(missing_ok=True)
        reason = f"{type(error).__name__}: {error}"
        report = failed_report(contract, rules, {"version": config.version}, reason) | {
            "held_out": True,
            "split": split_identity(split),
        }
    return finish(report | {"inputs": hashes}, directory, tracking_uri, artifact_root)


def run(truth_dirs: list[Path], sweep_root: Path, split: Path, output: Path, **options) -> str:
    """Score and track one attempt; return "completed" or "failed"."""
    return run_attempt(truth_dirs, sweep_root, split, output, **options).status


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score rules-v1 and the calibrated classifier on held-out sweep truths."
    )
    parser.add_argument("--truth", type=Path, nargs="+", required=True)
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True, help="Frozen split file")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--frozen-schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--training", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_errors = (
        ContractError,
        RulesError,
        TrainingError,
        CalibrationError,
        OutputError,
        tomllib.TOMLDecodeError,
    )
    try:
        outcome = run_attempt(
            args.truth,
            args.sweep_root,
            args.split,
            args.output,
            contract_path=args.contract,
            frozen_schema=args.frozen_schema,
            rules_path=args.rules,
            training_path=args.training,
            calibration_path=args.calibration,
            tracking_uri=args.tracking_uri,
            artifact_root=args.artifact_root,
        )
    except config_errors as error:
        print(f"Scoring error: {error}", file=sys.stderr)
        return EXIT_ERROR
    print(
        f"Held-out scoring {outcome.status} in {outcome.directory}; "
        f"MLflow run {outcome.run_id or 'not logged'}"
    )
    if not outcome.run_id:
        return EXIT_UNTRACKED
    return EXIT_FAILED if outcome.status == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())
