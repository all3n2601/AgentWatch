"""Score rules-v1 and the frozen M03/M04 classifier on held-out capacity-sweep truths.

    python -m agentwatch_model.score_held_out --truth DIR [DIR ...] --sweep-root DIR \
        --split FILE [--candidates DIR] [--calibrated DIR] [--output DIR]

`--truth` folders are `sweep-truth-v2` outputs, one per sweep. `--candidates` is the M03
`make train-candidates` folder and `--calibrated` the M04 `make calibrate` folder; nothing is
trained or fitted here. The pair is loaded and checked against the split first: a pair that
does not match is a setup error (exit 2) and no attempt is written. Then the split's rows are
loaded with the leakage check on. The model must reproduce every M03 validation prediction, in
a record of exactly the `fit.steps` M04 fitted on, and every M04 test prediction (label,
abstention, calibrated confidence and class probabilities), in a record of exactly the
test-split steps the M04 report counts; no scored run or plan may appear in the split. Each
of those failures ends as a recorded failed attempt.

Each attempt writes `<output>/<attempt id>/` with report.json and predictions.jsonl, and is
logged to the MLflow experiment `a06-held-out-scoring`, failed attempts included. The attempt id
is the UTC start time plus the first digits of the hash of every input, so it names both when
and on what the attempt ran. Outputs are scores, not run evidence: they never belong in
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

from agentwatch_model.candidates import TrainingError
from agentwatch_model.contract import DEFAULT_CONTRACT, Contract, ContractError, load_contract
from agentwatch_model.held_out_model import (
    CALIBRATION,
    CLASSIFIER_SCORER,
    FrozenClassifier,
    HeldOutModelError,
    check_reproduces_calibration,
    check_reproduces_validation,
    frozen_scores,
    load_frozen_classifier,
)
from agentwatch_model.held_out_report import GATE, build_report, unscored_steps
from agentwatch_model.held_out_rows import (
    HeldOutError,
    held_out_rows,
    load_truths,
    run_directory,
)
from agentwatch_model.held_out_scorers import (
    RULES_SCORER,
    ScoredStep,
    rule_scores,
    seen_task_groups,
)
from agentwatch_model.report import dataset_version, identity
from agentwatch_model.rule_baseline import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    DEFAULT_SCHEMA,
    EXIT_ERROR,
    split_identity,
)
from agentwatch_model.rules import DEFAULT_RULES, Rules, RulesError, load_rules
from agentwatch_model.tracking import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_TRACKING_URI,
    log_run,
    tracking_errors,
)
from agentwatch_model.train_candidates import (
    MODEL_FILES,
    PREPROCESSING,
    VALIDATION_PREDICTIONS,
    Loader,
    attempt_errors,
    label_version,
)
from agentwatch_model.train_candidates import (
    REPORT as CANDIDATES_REPORT,
)

DEFAULT_CANDIDATES = Path(".data/candidates")
DEFAULT_CALIBRATED = Path(".data/calibration")
DEFAULT_OUTPUT = Path(".data/a06-scoring/scores")
EXPERIMENT = "a06-held-out-scoring"
TASK = "A06"
REPORT = "report.json"
PREDICTIONS = "predictions.jsonl"
TRACKING = "tracking.json"
SPLIT_NAMES = ("train", "validation", "test")
LOGGED_ARTIFACTS = (REPORT, PREDICTIONS)
FORBIDDEN_OUTPUT = ("data", "held-out")
ATTEMPT_ID_HASH_DIGITS = 6
EXIT_FAILED = 1
EXIT_UNTRACKED = 3
TRUTH_FILES = ("truth.json", "truth.jsonl")
TRUTH_IDENTITY_FIELDS = ("label_version", "steps_sha256", "steps_summary_sha256", "truth_sha256")


class OutputError(ValueError):
    """An output location that must not receive scores."""


class NoStepScored(HeldOutError):
    """Every truth record was unjudged or refused; carries the unjudged records for the report."""

    def __init__(self, refused: dict[str, int], unjudged: list[dict]) -> None:
        super().__init__(f"no step could be scored; refused steps by code: {refused}")
        self.unjudged = unjudged


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


def model_hashes(candidates_dir: Path, calibrated_dir: Path) -> dict:
    """SHA-256 of each frozen M03/M04 file the attempt depends on; a missing file is None."""
    files = (
        candidates_dir / MODEL_FILES["classifier"],
        candidates_dir / PREPROCESSING,
        candidates_dir / CANDIDATES_REPORT,
        candidates_dir / VALIDATION_PREDICTIONS,
        calibrated_dir / CALIBRATION,
    )
    return {path.name: hash_of(path) for path in files}


def input_hashes(
    truth_dirs: list[Path],
    sweep_root: Path,
    split: Path,
    candidates_dir: Path | None = None,
    calibrated_dir: Path | None = None,
) -> dict:
    """SHA-256 of the split, every truth file and each run's evidence and journal, and, when
    the frozen model folders are given, the model files.

    Truth files are keyed by the sweep folder their summary names, or by the truth folder as
    given when it cannot be read; a key already taken gets the folder's argument position, so
    no two truth folders share a key. A truth folder that cannot be read contributes only its
    own files; the attempt itself then fails on it and records why. A sweep folder named twice
    is hashed once (its run files are the same files). A missing file is None, so it stays
    distinct from any hash.
    """
    hashes: dict = {"split": hash_of(split), "truth": {}, "runs": {}}
    if candidates_dir is not None and calibrated_dir is not None:
        hashes["model"] = model_hashes(candidates_dir, calibrated_dir)
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


def scored_overlap(rows: list[dict], split_rows: list[dict]) -> list[str]:
    """Every scored run_id, and every scored non-null plan_sha256, found in any split row."""
    split_runs = {row["run_id"] for row in split_rows}
    split_plans = {row.get("plan_sha256") for row in split_rows} - {None}
    runs = sorted({row["run_id"] for row in rows} & split_runs)
    plans = sorted(({row.get("plan_sha256") for row in rows} - {None}) & split_plans)
    return [f"run_id {run}" for run in runs] + [f"plan_sha256 {plan}" for plan in plans]


def score_rows(
    truth_dirs: list[Path],
    sweep_root: Path,
    split: Path,
    candidates_dir: Path,
    calibrated_dir: Path,
    model: FrozenClassifier,
    rules: Rules,
    contract: Contract,
    *,
    contract_path: Path,
    frozen_schema: Path,
    loader: Loader,
    resamples: int,
    seed: int,
) -> tuple[dict, str]:
    """Join rows, check the model and the split, score with both scorers; return the report
    body and the predictions text. Nothing is written, so a failed check leaves no scores.

    The checks, in order: the model reproduces M03's validation predictions and M04's test
    predictions, then no scored run or plan appears in any split row."""
    rows, refused, summaries, unjudged = held_out_rows(truth_dirs, sweep_root)
    if not rows:
        raise NoStepScored(refused, unjudged)
    split_rows = {name: loader(split, name, contract_path, frozen_schema) for name in SPLIT_NAMES}
    check_reproduces_validation(model, split_rows["validation"], candidates_dir)
    check_reproduces_calibration(model, split_rows["test"], calibrated_dir)
    overlap = scored_overlap(rows, [row for part in split_rows.values() for row in part])
    if overlap:
        raise HeldOutError(f"scored steps appear in the split: {', '.join(overlap)}")
    train = split_rows["train"]
    scores = {RULES_SCORER: rule_scores(rules, rows), CLASSIFIER_SCORER: frozen_scores(model, rows)}
    body = build_report(
        rows,
        scores,
        {
            "seen_task_groups": seen_task_groups(train),
            "refused": refused,
            "summaries": summaries,
            "unjudged": unjudged,
            "model": {"frozen": model.identity, "label_version": label_version(train)},
            "resamples": resamples,
            "seed": seed,
        },
    )
    report = (
        identity(contract, rules)
        | body
        | {
            "status": "completed",
            "evaluation": False,
            "dataset_version": dataset_version(train),
            "scored_split": "held-out",
            "split": split_identity(split),
            "scored_steps": len(rows),
            "truths": {
                summary["sweep_folder"]: {name: summary[name] for name in TRUTH_IDENTITY_FIELDS}
                for summary in summaries
            },
        }
    )
    return report, prediction_lines(rows, scores)


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


def load_configuration(contract_path: Path, rules_path: Path) -> tuple[Contract, Rules]:
    """The evaluation contract and rules; one that cannot be read is a configuration error."""
    try:
        return load_contract(contract_path), load_rules(rules_path)
    except OSError as error:
        raise ContractError(f"Cannot read configuration: {error}") from error


def run_attempt(
    truth_dirs: list[Path],
    sweep_root: Path,
    split: Path,
    output: Path,
    *,
    candidates_dir: Path = DEFAULT_CANDIDATES,
    calibrated_dir: Path = DEFAULT_CALIBRATED,
    contract_path: Path = DEFAULT_CONTRACT,
    frozen_schema: Path = DEFAULT_SCHEMA,
    rules_path: Path = DEFAULT_RULES,
    loader: Loader | None = None,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> Outcome:
    """One scoring attempt; data and model-check failures end as a recorded failed attempt.

    A frozen pair that cannot be loaded or does not match the split is a setup error, raised
    before any attempt folder exists. A recorded failed report carries the rules identity and
    the frozen identity under `model.frozen`; nothing was trained, so there is no training
    config to name.
    """
    check_output(output)
    contract, rules = load_configuration(contract_path, rules_path)
    model = load_frozen_classifier(candidates_dir, calibrated_dir, split)
    recorded_failures = (*attempt_errors(), HeldOutError, HeldOutModelError, SplitError)
    hashes = input_hashes(truth_dirs, sweep_root, split, candidates_dir, calibrated_dir)
    directory = output / attempt_id(hashes, datetime.now(UTC))
    directory.mkdir(parents=True, exist_ok=True)
    # A rerun with identical inputs in the same second reuses the folder; drop its old files.
    for name in (REPORT, PREDICTIONS, TRACKING):
        (directory / name).unlink(missing_ok=True)
    try:
        report, predictions = score_rows(
            truth_dirs,
            sweep_root,
            split,
            candidates_dir,
            calibrated_dir,
            model,
            rules,
            contract,
            contract_path=contract_path,
            frozen_schema=frozen_schema,
            loader=loader or load_split,
            resamples=resamples,
            seed=seed,
        )
    except recorded_failures as error:
        reason = f"{type(error).__name__}: {error}"
        report = identity(contract, rules) | {
            "status": "failed",
            "evaluation": False,
            "reason": reason,
            "model": {"frozen": model.identity},
            "gate": GATE,
            "held_out": True,
            "split": split_identity(split),
        }
        if isinstance(error, NoStepScored):
            report["unscored"] = {"steps": unscored_steps([], error.unjudged)}
    else:
        write_atomic(directory / PREDICTIONS, predictions)
    return finish(report | {"inputs": hashes}, directory, tracking_uri, artifact_root)


def run(truth_dirs: list[Path], sweep_root: Path, split: Path, output: Path, **options) -> str:
    """Score and track one attempt; return "completed" or "failed"."""
    return run_attempt(truth_dirs, sweep_root, split, output, **options).status


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score rules-v1 and the frozen M03/M04 classifier on held-out sweep truths."
    )
    parser.add_argument("--truth", type=Path, nargs="+", required=True)
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True, help="Frozen split file")
    parser.add_argument(
        "--candidates", type=Path, default=DEFAULT_CANDIDATES, help="M03 train-candidates folder"
    )
    parser.add_argument(
        "--calibrated", type=Path, default=DEFAULT_CALIBRATED, help="M04 calibrate folder"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--frozen-schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_errors = (
        ContractError,
        RulesError,
        HeldOutModelError,
        # An environment that cannot load LightGBM is a setup error: no attempt exists.
        TrainingError,
        OutputError,
        tomllib.TOMLDecodeError,
    )
    try:
        outcome = run_attempt(
            args.truth,
            args.sweep_root,
            args.split,
            args.output,
            candidates_dir=args.candidates,
            calibrated_dir=args.calibrated,
            contract_path=args.contract,
            frozen_schema=args.frozen_schema,
            rules_path=args.rules,
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
