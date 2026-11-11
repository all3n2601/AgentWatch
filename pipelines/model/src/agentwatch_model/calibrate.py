"""Calibrate the M03 classifier's confidence on validation, choose its abstention threshold, and
score both once on the test split.

Reads an evaluated M03 attempt (make train-candidates) and the split it was trained on, which is
read again through load_split. The attempt's test predictions must match the hash its report
recorded, the split file must be the one it names, the contract must be unchanged, and every
prediction record must belong to exactly one split row. Candidates that were not evaluated, or a
split load_split refuses, give a blocked report. Every attempt with a report is tracked.
"""

import argparse
import hashlib
import json
import math
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path

import numpy as np
from agentwatch_data.files import file_sha256, json_text, write_atomic
from agentwatch_data.splits import SplitError, load_split

from agentwatch_model.calibration import (
    DEFAULT_CALIBRATION,
    CalibrationConfig,
    CalibrationError,
    choose_threshold,
    fit_temperature,
    load_calibration_config,
    scale,
)
from agentwatch_model.calibration_report import (
    METHOD,
    Fit,
    Scored,
    build_calibration_report,
    contract_identity,
)
from agentwatch_model.contract import DEFAULT_CONTRACT, Contract, ContractError, load_contract
from agentwatch_model.metrics import CLASSES, is_number
from agentwatch_model.mixed import is_joint_only, is_mixed
from agentwatch_model.report import dataset_version, summary
from agentwatch_model.rule_baseline import (
    DEFAULT_SCHEMA,
    EXIT_BLOCKED,
    EXIT_ERROR,
    RowError,
    validate_dataset_version,
    validate_rows,
)
from agentwatch_model.tracking import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_TRACKING_URI,
    log_run,
    tracking_errors,
)
from agentwatch_model.train_candidates import DEFAULT_OUTPUT as DEFAULT_CANDIDATES
from agentwatch_model.train_candidates import (
    EXIT_UNTRACKED,
    PREDICTIONS,
    REPORT,
    TRACKING,
    Outcome,
)

DEFAULT_OUTPUT = Path(".data/calibration")
TASK = "M04"
CALIBRATION = "calibration.json"
# Written in this order; the report is last, so its presence marks a complete attempt.
OUTPUTS = (PREDICTIONS, CALIBRATION, REPORT)
ABSTAIN_WHEN = "calibrated top-label confidence is below abstention_threshold"
# M03 writes softmax outputs as float64; their sum can differ from 1 by rounding only.
PROBABILITY_SUM_TOLERANCE = 1e-6
BOUND_TOLERANCE = 1e-3  # relative; a fitted temperature this close to a bound is reported

Loader = Callable[..., list[dict]]


def read_json(path: Path, what: str) -> dict:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as error:
        raise CalibrationError(f"No {what} at {path}; run make train-candidates first") from error
    except (OSError, json.JSONDecodeError) as error:
        raise CalibrationError(f"Cannot read the {what} {path}: {error}") from error
    if not isinstance(value, dict):
        raise CalibrationError(f"The {what} {path} is not a JSON object")
    return value


def require(report: dict, *keys: str) -> None:
    missing = [key for key in keys if key not in report]
    if missing:
        raise CalibrationError(f"The candidates report is missing {', '.join(missing)}")


def bound_text(directory: Path, candidates: dict, key: str) -> tuple[str, str]:
    """A file the candidates report names under key, after checking its recorded hash."""
    identity = candidates[key]
    name = identity.get("file") if isinstance(identity, dict) else None
    if not isinstance(name, str) or Path(name).name != name:
        raise CalibrationError(f"The candidates report {key} entry does not name a file")
    try:
        text = (directory / name).read_text()
    except OSError as error:
        raise CalibrationError(f"Cannot read candidate file {directory / name}: {error}") from error
    if hashlib.sha256(text.encode()).hexdigest() != identity.get("sha256"):
        raise CalibrationError(
            f"{name} contents do not match the candidates report; rerun make train-candidates"
        )
    return name, text


def read_records(directory: Path, candidates: dict, key: str) -> list[dict]:
    name, text = bound_text(directory, candidates, key)
    try:
        records = [json.loads(line) for line in text.splitlines()]
    except json.JSONDecodeError as error:
        raise CalibrationError(f"{name} has a line that is not JSON: {error}") from error
    if not all(isinstance(record, dict) for record in records):
        raise CalibrationError(f"{name} has a line that is not a JSON object")
    return records


def read_preprocessing(directory: Path, candidates: dict) -> dict:
    """The serving contract, bound by hash, whose model files match the candidates report."""
    name, text = bound_text(directory, candidates, "preprocessing")
    try:
        preprocessing = json.loads(text)
    except json.JSONDecodeError as error:
        raise CalibrationError(f"{name} is not JSON: {error}") from error
    files = preprocessing.get("model_files") if isinstance(preprocessing, dict) else None
    recorded = candidates["model_files"]
    if (
        not isinstance(files, dict)
        or {key: (value or {}).get("sha256") for key, value in files.items()} != recorded
    ):
        raise CalibrationError(f"{name} model files do not match the candidates report")
    return preprocessing


def probability_columns(preprocessing: dict) -> tuple[str, ...]:
    columns = preprocessing.get("probability_columns")
    if (
        not isinstance(columns, list)
        or len(columns) < 2
        or len(columns) != len(set(columns))
        or not set(columns) <= set(CLASSES)
    ):
        raise CalibrationError(
            "preprocessing.json probability_columns must name at least two distinct classes "
            f"from {list(CLASSES)}"
        )
    return tuple(columns)


def probability_matrix(records: list[dict], columns: tuple[str, ...], split: str) -> np.ndarray:
    matrix = np.empty((len(records), len(columns)), dtype=np.float64)
    for index, record in enumerate(records):
        scores = record.get("probabilities")
        if not isinstance(scores, dict) or set(scores) != set(columns):
            raise CalibrationError(f"{split} record {index} does not score exactly {list(columns)}")
        values = [scores[name] for name in columns]
        if not all(is_number(v) and 0 <= v <= 1 for v in values) or not math.isclose(
            sum(values), 1.0, abs_tol=PROBABILITY_SUM_TOLERANCE
        ):
            raise CalibrationError(f"{split} record {index} probabilities are not a distribution")
        if record.get("prediction") not in columns:
            raise CalibrationError(f"{split} record {index} predicts an unknown class")
        matrix[index] = values
    return matrix


def matched_records(records: list[dict], rows: list[dict], split: str) -> list[dict]:
    """The records in split-row order; each row must have exactly one, with the same label."""
    by_key: dict[tuple, dict] = {}
    for record in records:
        key = (record.get("run_id"), record.get("step_id"))
        if key in by_key:
            raise CalibrationError(f"{split} predictions repeat run {key[0]} step {key[1]}")
        by_key[key] = record
    keys = [(row["run_id"], row["step_id"]) for row in rows]
    if set(keys) != set(by_key):
        raise CalibrationError(
            f"{split} predictions ({len(by_key)} steps) do not cover exactly the {len(keys)} "
            f"{split} split rows; rerun make train-candidates on this split"
        )
    matched = [by_key[key] for key in keys]
    for row, record in zip(rows, matched, strict=True):
        if record.get("label_resource") != row["label_resource"]:
            raise CalibrationError(
                f"{split} prediction for run {row['run_id']} step {row['step_id']} records a "
                "different label than the split"
            )
    return matched


def check_candidates(contract: Contract, candidates: dict, split: Path) -> None:
    if candidates["scored_split"] != "test" or candidates["evaluation"] is not True:
        raise CalibrationError("The candidates report is not an evaluation on the test split")
    if candidates["contract"].get("hash") != contract.hash:
        raise CalibrationError(
            "The evaluation contract changed since the candidates were trained "
            f"({candidates['contract'].get('hash')} then, {contract.hash} now); retrain them"
        )
    actual = file_sha256(split) if split.is_file() else None
    if actual != (candidates["split"] or {}).get("sha256"):
        raise CalibrationError(
            f"Split {split.name} is not the split the candidates were trained on"
        )


def at_bound(temperature: float, config: CalibrationConfig) -> bool:
    return any(
        math.isclose(temperature, bound, rel_tol=BOUND_TOLERANCE)
        for bound in (config.min_temperature, config.max_temperature)
    )


def fit_on_validation(
    rows: list[dict],
    records: list[dict],
    columns: tuple[str, ...],
    contract: Contract,
    config: CalibrationConfig,
    early_stopping: str | None,
) -> Fit:
    """Temperature and threshold from validation; unfitted below the configured minimum."""
    labels = [row["label_resource"] for row in rows]
    # A class unseen in training has no probability column, so it cannot inform the fit.
    usable = [index for index, label in enumerate(labels) if label in columns]
    common = {
        "validation_steps": len(rows),
        "usable_steps": len(usable),
        "early_stopping_on_fit_split": early_stopping == "validation",
    }
    # Validated before the size check, so malformed records are refused even when too few
    # steps remain to fit on.
    matrix = probability_matrix(records, columns, "validation")
    if len(usable) < config.min_validation_steps:
        reason = (
            f"{len(usable)} usable validation steps; {config.version} needs at least "
            f"{config.min_validation_steps} to fit calibration"
        )
        return Fit(False, None, None, reason, temperature_at_bound=False, selection=None, **common)
    targets = np.array([columns.index(labels[index]) for index in usable])
    temperature = fit_temperature(matrix[usable], targets, config)
    confidences = scale(matrix, temperature).max(axis=1).tolist()
    # Macro F1 is judged on single-class steps, as a mixed step has no single class, but the
    # abstention limit also covers the scored mixed steps, the contract's mixed_contention
    # scope. Joint-only steps are never scored, so they take no part.
    scored = [index for index, row in enumerate(rows) if not is_joint_only(row)]
    selection = choose_threshold(
        [labels[index] for index in scored],
        [records[index]["prediction"] for index in scored],
        [confidences[index] for index in scored],
        contract.classes,
        contract.max_abstention_rate,
        mixed=[is_mixed(rows[index]) for index in scored],
    )
    return Fit(
        True,
        temperature,
        selection["threshold"],
        None,
        temperature_at_bound=at_bound(temperature, config),
        selection=selection,
        **common,
    )


def score_test(rows: list[dict], records: list[dict], columns: tuple[str, ...], fit: Fit) -> Scored:
    matrix = probability_matrix(records, columns, "test")
    calibrated = scale(matrix, fit.temperature) if fit.temperature is not None else matrix
    confidences = calibrated.max(axis=1).tolist() if rows else []
    threshold = fit.threshold
    return Scored(
        rows=rows,
        records=records,
        raw_confidences=matrix.max(axis=1).tolist() if rows else [],
        confidences=confidences,
        abstained=[threshold is not None and c < threshold for c in confidences],
        probabilities=[dict(zip(columns, map(float, row), strict=True)) for row in calibrated],
        calibrated=fit.fitted,
    )


def prediction_lines(scored: Scored, fit: Fit, config: CalibrationConfig) -> str:
    lines = []
    for index, (row, record) in enumerate(zip(scored.rows, scored.records, strict=True)):
        abstained = scored.abstained[index]
        # Unfitted, there is no calibrated confidence: the fields are null, never raw scores.
        confidence = scored.confidences[index] if fit.fitted else None
        probabilities = scored.probabilities[index] if fit.fitted else None
        lines.append(
            json.dumps(
                {
                    "run_id": row["run_id"],
                    "step_id": row["step_id"],
                    "task_id": row["task_id"],
                    "condition": row["condition"],
                    "split": "test",
                    "dataset_version": row["dataset_version"],
                    "label_version": row["label_version"],
                    "label_resource": row["label_resource"],
                    "prediction": record["prediction"],
                    "attribution": None if abstained else record["prediction"],
                    "abstained": abstained,
                    "confidence": confidence,
                    "raw_confidence": scored.raw_confidences[index],
                    "probabilities": probabilities,
                    "calibrated": fit.fitted,
                    "calibration_version": config.version,
                    "evaluation": True,
                },
                sort_keys=True,
                allow_nan=False,
            )
        )
    return "".join(f"{line}\n" for line in lines)


def calibration_document(
    fit: Fit,
    config: CalibrationConfig,
    columns: tuple[str, ...],
    preprocessing: dict,
    report: dict,
) -> dict:
    """What serving needs to reproduce calibrated confidence and abstention.

    The gate outcomes travel with the parameters, so a calibration whose test-split gates did
    not pass is never mistaken for one that may be served.
    """
    return {
        "version": config.version,
        "method": METHOD,
        "status": report["status"],
        "contract": report["contract"],
        "gates": {name: report["gates"][name]["outcome"] for name in ("calibration", "abstention")},
        "blocks_promotion": report["summary"]["blocks_promotion"],
        "fitted": fit.fitted,
        "reason": fit.reason,
        "temperature": fit.temperature,
        "abstention_threshold": fit.threshold,
        "abstain_when": ABSTAIN_WHEN,
        "fit_split": "validation",
        "probability_columns": list(columns),
        "training_version": preprocessing.get("training_version"),
        "model_files": preprocessing["model_files"],
        "candidates": report["candidates"],
    }


def candidates_identity(directory: Path, candidates: dict) -> dict:
    tracked = directory / TRACKING
    run_id = None
    if tracked.is_file():
        run_id = read_json(tracked, "candidates tracking record").get("run_id")
    return {
        "report_sha256": file_sha256(directory / REPORT),
        "predictions_sha256": (candidates.get("predictions") or {}).get("sha256"),
        "run_id": run_id,
    }


def blocked_report(
    contract: Contract, candidates: dict, config: CalibrationConfig, reason: str, identity: dict
) -> dict:
    return {
        "rules": candidates["rules"],
        "contract": contract_identity(contract),
        "status": "blocked",
        "evaluation": False,
        "reason": reason,
        "candidates": identity,
        "model": candidates.get("model"),
        "calibration": {"version": config.version, "method": METHOD, "fitted": False},
        "metrics": None,
        "gates": {},
        "summary": summary({}) | {"blocks_promotion": True},
    }


def replace_outputs(directory: Path, report: dict, files: dict[str, str] | None = None) -> None:
    """Replace an earlier attempt's outputs once this attempt has a report; report last."""
    directory.mkdir(parents=True, exist_ok=True)
    for name in (*OUTPUTS, TRACKING):
        (directory / name).unlink(missing_ok=True)
    for name, text in (files or {}).items():
        write_atomic(directory / name, text)
    write_atomic(directory / REPORT, json_text(report))


def finish(report: dict, directory: Path, tracking_uri: str, artifact_root: Path) -> Outcome:
    try:
        run_id = log_run(
            report,
            directory,
            artifacts=OUTPUTS,
            tracking_uri=tracking_uri,
            artifact_root=artifact_root,
            task=TASK,
        )
    except tracking_errors() as error:
        write_atomic(directory / TRACKING, json_text({"logged": False, "error": str(error)}))
        print(f"Tracking error, report kept: {error}", file=sys.stderr)
        return Outcome(report["status"], None)
    write_atomic(directory / TRACKING, json_text({"logged": True, "run_id": run_id}))
    return Outcome(report["status"], run_id)


def load_configuration(
    contract_path: Path, calibration_path: Path
) -> tuple[Contract, CalibrationConfig]:
    try:
        return load_contract(contract_path), load_calibration_config(calibration_path)
    except OSError as error:
        raise CalibrationError(f"Cannot read configuration: {error}") from error


def load_rows(
    load: Loader, split: Path, contract_path: Path, frozen_schema: Path, candidates: dict
) -> dict[str, list[dict]]:
    rows = {
        name: load(split, name, contract_path, frozen_schema) for name in ("validation", "test")
    }
    for name, split_rows in rows.items():
        validate_rows(split_rows, name)
    combined = rows["validation"] + rows["test"]
    validate_dataset_version(combined)
    if combined and dataset_version(combined) != candidates.get("dataset_version"):
        raise CalibrationError("The split rows come from a different release than the candidates")
    return rows


def calibrated_attempt(
    contract: Contract,
    config: CalibrationConfig,
    candidates: dict,
    identity: dict,
    directory: Path,
    rows: dict[str, list[dict]],
) -> tuple[dict, dict[str, str]]:
    """The report and the files to write beside it for one evaluated attempt."""
    preprocessing = read_preprocessing(directory, candidates)
    columns = probability_columns(preprocessing)
    test_records = matched_records(
        read_records(directory, candidates, "predictions"), rows["test"], "test"
    )
    validation_records = matched_records(
        read_records(directory, candidates, "validation_predictions"),
        rows["validation"],
        "validation",
    )
    early_stopping = candidates["model"].get("early_stopping")
    fit = fit_on_validation(
        rows["validation"], validation_records, columns, contract, config, early_stopping
    )
    scored = score_test(rows["test"], test_records, columns, fit)
    report = build_calibration_report(contract, candidates, identity, fit, scored, config)
    lines = prediction_lines(scored, fit, config)
    report |= {
        "predictions": {"file": PREDICTIONS, "sha256": hashlib.sha256(lines.encode()).hexdigest()}
    }
    document = calibration_document(fit, config, columns, preprocessing, report)
    return report, {PREDICTIONS: lines, CALIBRATION: json_text(document)}


def run(
    split: Path,
    candidates_directory: Path,
    output: Path,
    *,
    contract_path: Path = DEFAULT_CONTRACT,
    frozen_schema: Path = DEFAULT_SCHEMA,
    calibration_path: Path = DEFAULT_CALIBRATION,
    loader: Loader | None = None,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> Outcome:
    """Calibrate one evaluated M03 attempt, write its outputs, and track the attempt."""
    contract, config = load_configuration(contract_path, calibration_path)
    candidates = read_json(candidates_directory / REPORT, "candidates report")
    require(candidates, "status", "rules", "contract")
    identity = candidates_identity(candidates_directory, candidates)
    if candidates["status"] != "evaluated":
        reason = (
            f"The candidates attempt is {candidates['status']}, not evaluated: "
            f"{candidates.get('reason', '')}"
        )
        report = blocked_report(contract, candidates, config, reason, identity)
        replace_outputs(output, report)
        return finish(report, output, tracking_uri, artifact_root)
    require(
        candidates,
        "split",
        "scored_split",
        "evaluation",
        "predictions",
        "validation_predictions",
        "preprocessing",
        "model_files",
        "gates",
        "model",
    )
    check_candidates(contract, candidates, split)
    # Resolved at call time, so tests can replace the module's load_split.
    load = loader or load_split
    try:
        rows = load_rows(load, split, contract_path, frozen_schema, candidates)
    except SplitError as error:
        report = blocked_report(contract, candidates, config, str(error), identity)
        replace_outputs(output, report)
        return finish(report, output, tracking_uri, artifact_root)
    report, files = calibrated_attempt(
        contract, config, candidates, identity, candidates_directory, rows
    )
    replace_outputs(output, report, files)
    return finish(report, output, tracking_uri, artifact_root)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate the M03 classifier on validation and score abstention on test."
    )
    parser.add_argument("--split", type=Path, required=True, help="The split M03 trained on")
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--frozen-schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Input and configuration problems exit 2; an I/O failure writing results crashes loudly.
    input_errors = (ContractError, CalibrationError, RowError, tomllib.TOMLDecodeError)
    try:
        outcome = run(
            args.split,
            args.candidates,
            args.output,
            contract_path=args.contract,
            frozen_schema=args.frozen_schema,
            calibration_path=args.calibration,
            tracking_uri=args.tracking_uri,
            artifact_root=args.artifact_root,
        )
    except input_errors as error:
        print(f"Calibration error: {error}", file=sys.stderr)
        return EXIT_ERROR
    print(f"Calibration {outcome.status}; MLflow run {outcome.run_id or 'not logged'}")
    # Untracked comes first, as for the candidates: an attempt that never reached MLflow must
    # be distinguishable from one that was recorded.
    if not outcome.run_id:
        return EXIT_UNTRACKED
    return EXIT_BLOCKED if outcome.status == "blocked" else 0


if __name__ == "__main__":
    sys.exit(main())
