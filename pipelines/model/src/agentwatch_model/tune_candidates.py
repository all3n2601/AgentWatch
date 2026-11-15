"""Tune the LightGBM candidates on the validation split and propose the next training version.

The search space and the selection rule are declared in tuning.toml before any run. Every grid
point trains both boosters on the training split (stopping early on validation, as
train-candidates does) and is scored on the validation split only: the test split is never
read. The classifier and the regressor train independently, so the best classifier settings
and the best regressor settings are chosen separately from the same grid. A setting replaces
the base (training-v1.toml) only when it beats the base by the declared margin.

Every point is a nested MLflow run under one parent run, failed points included. The proposal
(training-v2.toml) is committed as the new training.toml before the test split is scored.
"""

import argparse
import itertools
import re
import statistics
import sys
import tomllib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from agentwatch_data.extract import canonical_sha256
from agentwatch_data.files import json_text, write_atomic
from agentwatch_data.splits import SplitError, load_split

from agentwatch_model.candidates import (
    HASH_DIGITS,
    TUNABLE,
    CandidateError,
    TrainingConfig,
    TrainingError,
    check_lightgbm,
    fit_candidate,
    load_training_config,
    predict,
)
from agentwatch_model.contract import DEFAULT_CONTRACT, ContractError, load_contract
from agentwatch_model.features import FeatureError
from agentwatch_model.metrics import class_scores, confusion_matrix, is_number, macro_f1_all_classes
from agentwatch_model.rule_baseline import (
    DEFAULT_SCHEMA,
    EXIT_BLOCKED,
    EXIT_ERROR,
    RowError,
    split_identity,
    validate_dataset_version,
    validate_rows,
)
from agentwatch_model.tracking import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_TRACKING_URI,
    SQLITE_PREFIX,
    experiment_id,
    tracking_errors,
)

PACKAGE = Path(__file__).parent
DEFAULT_TUNING = PACKAGE / "tuning.toml"
DEFAULT_BASE = PACKAGE / "training-v1.toml"
DEFAULT_OUTPUT = Path(".data/tuning")
VERSION = "tuning-v1"
EXPERIMENT = "agentwatch-tuning"
TASK = "M03-tuning"
REPORT = "report.json"
TRACKING = "tracking.json"
EXIT_UNTRACKED = 3
SPLITS = ("train", "validation")
PROBABILITY_FLOOR = 1e-15  # log loss of a zero probability is unbounded
CLASSIFIER_METRIC = ("classifier", "log_loss")
REGRESSOR_METRIC = ("regressor", "mae")
# Simpler first: fewer leaves, more data per leaf, more L2, fewer features, slower learning.
SIMPLICITY = (
    ("num_leaves", 1),
    ("min_data_in_leaf", -1),
    ("lambda_l2", -1),
    ("feature_fraction", 1),
    ("learning_rate", 1),
)
FIXED_LIMITATIONS = (
    (
        "The validation split also stops training early, so every point's validation value is "
        "slightly optimistic, equally for all points."
    ),
    (
        "The test split is not read; the chosen settings are evaluated on it once, after the "
        "proposal is committed as training-v2."
    ),
)


def limitations(validation: list[dict]) -> list[str]:
    """The fixed limitations, after one describing the validation split actually used."""
    experiments = {row.get("experiment") for row in validation} - {None}
    tasks = sorted({row["task_id"] for row in validation})
    classes = Counter(row["label_resource"] for row in validation)
    counts = ", ".join(f"{classes[name]} {name}" for name in sorted(classes))
    first = (
        f"Settings are chosen on the validation split: {len(validation)} steps from "
        f"{len(experiments)} experiments of {len(tasks)} task(s), by class {counts}. Small "
        "differences between points are within its noise; the declared margin guards against "
        "adopting them."
    )
    return [first, *FIXED_LIMITATIONS]


class TuningError(ValueError):
    """A tuning configuration that cannot be searched as declared."""


@dataclass(frozen=True)
class TuningConfig:
    version: str
    search: dict[str, tuple[float, ...]]
    min_relative_improvement: float
    base_path: Path


@dataclass(frozen=True)
class Outcome:
    status: str
    run_id: str | None


def search_values(name: str, values: object) -> tuple[float, ...]:
    if name not in TUNABLE:
        raise TuningError(f"[search] {name} is not a tunable setting ({', '.join(TUNABLE)})")
    if not isinstance(values, list) or not values:
        raise TuningError(f"[search] {name} needs at least one value")
    if not all(is_number(value) for value in values):
        raise TuningError(f"[search] {name} values must each be a finite number")
    if len(set(values)) != len(values):
        raise TuningError(f"[search] {name} repeats a value")
    return tuple(sorted(values))


def load_tuning_config(path: Path = DEFAULT_TUNING, base_path: Path = DEFAULT_BASE) -> TuningConfig:
    try:
        parsed = tomllib.loads(path.read_text())
    except OSError as error:
        raise TuningError(f"Cannot read {path}: {error}") from error
    if parsed.get("version") != VERSION:
        raise TuningError(f"{path.name}: version must be {VERSION!r}")
    margin = (parsed.get("selection") or {}).get("min_relative_improvement")
    if not is_number(margin) or not 0 <= margin < 1:
        raise TuningError("[selection] min_relative_improvement must be a number in [0, 1)")
    search_table = parsed.get("search")
    if not isinstance(search_table, dict) or not search_table:
        raise TuningError("Missing [search]")
    search = {name: search_values(name, values) for name, values in sorted(search_table.items())}
    try:
        base = load_training_config(base_path)
    except (TrainingError, OSError) as error:
        raise TuningError(f"Base settings {base_path.name}: {error}") from error
    points = grid(search)
    for model, settings in (("classifier", base.classifier), ("regressor", base.regressor)):
        if settings not in points:
            raise TuningError(f"The base settings ({model}, {base.version}) are not a grid point")
    return TuningConfig(
        version=f"{VERSION}+{canonical_sha256(parsed)[:HASH_DIGITS]}",
        search=search,
        min_relative_improvement=float(margin),
        base_path=base_path,
    )


def grid(search: Mapping[str, Sequence[float]]) -> list[dict]:
    """Every combination of the search values, settings in name order, values ascending."""
    names = sorted(search)
    values = [sorted(search[name]) for name in names]
    return [
        dict(zip(names, combination, strict=True)) for combination in itertools.product(*values)
    ]


def log_loss(probabilities: list[dict[str, float]], labels: list[str]) -> float:
    chosen = [
        max(p[label], PROBABILITY_FLOOR) for p, label in zip(probabilities, labels, strict=True)
    ]
    return float(-np.mean(np.log(chosen)))


def validation_metrics(candidate, validation: list[dict]) -> tuple[dict | None, dict]:
    """The classifier's log loss and macro F1 on single-class validation steps of trained
    classes, and the regressor's MAE on every validation step."""
    predictions = predict(candidate, validation)
    fractions = [p.excess_fraction for p in predictions]
    targets = [row["label_excess_fraction"] for row in validation]
    regressor = {
        "mae": float(np.mean(np.abs(np.array(fractions) - np.array(targets)))),
        "steps": len(validation),
    }
    usable = [
        (row, p)
        for row, p in zip(validation, predictions, strict=True)
        if row["label_resource"] in candidate.class_order
    ]
    if not usable:
        return None, regressor
    labels = [row["label_resource"] for row, _ in usable]
    scores = class_scores(confusion_matrix(labels, [p.label for _, p in usable]))
    classifier = {
        "log_loss": log_loss([p.probabilities for _, p in usable], labels),
        "macro_f1": macro_f1_all_classes(scores, candidate.class_order),
        "steps": len(usable),
    }
    return classifier, regressor


def evaluate_point(
    settings: dict, base: TrainingConfig, train: list[dict], validation: list[dict], errors
) -> dict:
    config = replace(base, classifier=dict(settings), regressor=dict(settings))
    try:
        candidate = fit_candidate(train, validation, config)
        classifier, regressor = validation_metrics(candidate, validation)
    except errors as error:
        return {
            "settings": settings,
            "status": "failed",
            "reason": f"{type(error).__name__}: {error}",
            "classifier": None,
            "regressor": None,
        }
    return {
        "settings": settings,
        "status": "trained",
        "best_iterations": candidate.best_iterations,
        "classifier": classifier,
        "regressor": regressor,
    }


def measured(result: dict, metric: tuple[str, str]) -> float | None:
    model, name = metric
    value = (result.get(model) or {}).get(name)
    return float(value) if is_number(value) else None


def simplicity(settings: dict) -> tuple:
    return tuple(sign * settings.get(name, 0) for name, sign in SIMPLICITY)


def select(results: list[dict], metric: tuple[str, str], base: dict, margin: float) -> dict:
    """The best measured point, adopted only when it beats the base by the relative margin."""
    scored = [(measured(r, metric), r) for r in results]
    scored = [(value, r) for value, r in scored if value is not None]
    base_value = next((value for value, r in scored if r["settings"] == base), None)
    if not scored:
        return {
            "settings": base,
            "adopted": False,
            "value": None,
            "base_value": base_value,
            "best_settings": None,
            "relative_improvement": None,
            "measured_points": 0,
            "base_measured": False,
        }
    value, best = min(scored, key=lambda item: (item[0], simplicity(item[1]["settings"])))
    # Without a measured base there is nothing to beat by the margin, so the base is kept.
    improvement = None if not base_value else (base_value - value) / base_value
    adopted = improvement is not None and improvement >= margin
    return {
        "settings": best["settings"] if adopted else base,
        "adopted": adopted,
        "value": value if adopted else base_value,
        "base_value": base_value,
        "best_settings": best["settings"],
        "best_value": value,
        "relative_improvement": improvement,
        "measured_points": len(scored),
        "base_measured": base_value is not None,
    }


def sensitivity(results: list[dict], metric: tuple[str, str]) -> dict:
    """For each setting and value: the spread of the metric over every point with that value."""
    summary: dict[str, dict[str, list[float]]] = {}
    for result in results:
        value = measured(result, metric)
        if value is None:
            continue
        for name, setting in result["settings"].items():
            summary.setdefault(name, {}).setdefault(str(setting), []).append(value)
    return {
        name: {
            setting: {
                "points": len(values),
                "min": min(values),
                "median": statistics.median(values),
                "max": max(values),
            }
            for setting, values in sorted(by_value.items(), key=lambda item: float(item[0]))
        }
        for name, by_value in sorted(summary.items())
    }


def next_version(base_version: str) -> str:
    """The training version after the base's: training-v1+... proposes training-v2."""
    matched = re.fullmatch(r"training-v([0-9]+)\+[0-9a-f]+", base_version)
    if matched is None:
        raise TuningError(f"Cannot derive the next training version from {base_version!r}")
    return f"training-v{int(matched.group(1)) + 1}"


def toml_value(value: float) -> str:
    return repr(float(value)) if isinstance(value, float) else str(value)


MODEL_COMMENTS = {
    "classifier": "Resource classifier (multiclass log loss).",
    "regressor": "Excess-latency fraction regressor (L1 loss, matching the contract's MAE).",
}


def proposal_text(tuning: TuningConfig, base: TrainingConfig, selection: dict) -> str:
    version = next_version(base.version)
    lines = [
        f"# {version}: LightGBM settings for the attribution candidates, chosen by",
        f"# tune-candidates ({tuning.version}) on the validation split only, against the base",
        f"# {base.version}. The test split was not read. Any change changes the version hash.",
        "# Determinism settings (one thread, fixed seed, deterministic mode) are fixed in code.",
        f'version = "{version}"',
        "",
        "# Rounds are an upper bound; training stops early when the validation loss stops improving.",
        f"num_boost_round = {base.num_boost_round}",
        f"early_stopping_rounds = {base.early_stopping_rounds}",
    ]
    for model in ("classifier", "regressor"):
        chosen = selection[model]
        how = "best grid point" if chosen["adopted"] else f"kept from {base.version}"
        lines += ["", f"# {MODEL_COMMENTS[model]}", f"# Settings: {how}.", f"[{model}]"]
        lines += [f"{name} = {toml_value(v)}" for name, v in sorted(chosen["settings"].items())]
    return "\n".join(lines) + "\n"


def child_metrics(result: dict) -> dict[str, float]:
    metrics = {}
    for model in ("classifier", "regressor"):
        for name, value in (result.get(model) or {}).items():
            if name != "steps" and is_number(value):
                metrics[f"{model}_{name}"] = float(value)
    return metrics


def log_tuning(report: dict, directory: Path, tracking_uri: str, artifact_root: Path) -> str:
    """One parent run with the report and proposal, and one nested run per grid point."""
    import mlflow

    if tracking_uri.startswith(SQLITE_PREFIX):
        Path(tracking_uri.removeprefix(SQLITE_PREFIX)).parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    experiment = experiment_id(artifact_root, EXPERIMENT)
    with mlflow.start_run(experiment_id=experiment) as parent:
        mlflow.set_tags({"task": TASK, "status": report["status"], "evaluation": "False"})
        mlflow.log_params(
            {
                "tuning_version": report["tuning"]["version"],
                "base_version": report["base"]["version"],
                "split_sha256": str(report["split"]["sha256"]),
                "dataset_version": str(report.get("dataset_version")),
                "points": str(report.get("points", 0)),
            }
        )
        for index, result in enumerate(report.get("results", [])):
            with mlflow.start_run(
                experiment_id=experiment, run_name=f"point-{index:03d}", nested=True
            ):
                mlflow.set_tags({"task": TASK, "status": result["status"]})
                mlflow.log_params({name: str(v) for name, v in result["settings"].items()})
                metrics = child_metrics(result)
                if metrics:
                    mlflow.log_metrics(metrics)
        for name in (REPORT, (report.get("proposal") or {}).get("file")):
            if name and (directory / name).is_file():
                mlflow.log_artifact(str(directory / name))
        return parent.info.run_id


def finish(report: dict, directory: Path, tracking_uri: str, artifact_root: Path) -> Outcome:
    write_atomic(directory / REPORT, json_text(report))
    try:
        run_id = log_tuning(report, directory, tracking_uri, artifact_root)
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
    tuning_path: Path = DEFAULT_TUNING,
    base_path: Path = DEFAULT_BASE,
    contract_path: Path = DEFAULT_CONTRACT,
    frozen_schema: Path = DEFAULT_SCHEMA,
    loader=None,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> Outcome:
    """Search the declared grid on train and validation; write the report and the proposal."""
    tuning = load_tuning_config(tuning_path, base_path)
    base = load_training_config(base_path)
    try:
        contract = load_contract(contract_path)
    except OSError as error:
        raise TuningError(f"Cannot read {contract_path}: {error}") from error
    proposal = f"{next_version(base.version)}.toml"
    check_lightgbm()
    from lightgbm.basic import LightGBMError

    errors = (CandidateError, FeatureError, LightGBMError)
    output.mkdir(parents=True, exist_ok=True)
    # An earlier attempt's files must not be mistaken for this attempt's.
    for stale in (output / REPORT, output / TRACKING, *output.glob("training-v*.toml")):
        stale.unlink(missing_ok=True)
    report = {
        "status": "tuned",
        "evaluation": False,
        "test_split_read": False,
        "tuning": {
            "version": tuning.version,
            "search": {name: list(values) for name, values in tuning.search.items()},
            "min_relative_improvement": tuning.min_relative_improvement,
            "selection_metrics": {"classifier": "log_loss", "regressor": "mae"},
        },
        "base": {
            "version": base.version,
            "classifier": base.classifier,
            "regressor": base.regressor,
        },
        "contract": {"version": contract.version, "hash": contract.hash},
        "split": split_identity(split),
        "limitations": list(FIXED_LIMITATIONS),
    }
    load = loader or load_split
    try:
        splits = {name: load(split, name, contract_path, frozen_schema) for name in SPLITS}
    except SplitError as error:
        return finish(
            report | {"status": "blocked", "reason": str(error)},
            output,
            tracking_uri,
            artifact_root,
        )
    try:
        for name, rows in splits.items():
            validate_rows(rows, name)
        validate_dataset_version(splits["train"] + splits["validation"])
    except RowError as error:
        reason = f"{type(error).__name__}: {error}"
        return finish(
            report | {"status": "failed", "reason": reason}, output, tracking_uri, artifact_root
        )
    results = [
        evaluate_point(settings, base, splits["train"], splits["validation"], errors)
        for settings in grid(tuning.search)
    ]
    report["limitations"] = limitations(splits["validation"])
    margin = tuning.min_relative_improvement
    selection = {
        "classifier": select(results, CLASSIFIER_METRIC, base.classifier, margin),
        "regressor": select(results, REGRESSOR_METRIC, base.regressor, margin),
    }
    report |= {
        "dataset_version": splits["train"][0]["dataset_version"] if splits["train"] else None,
        "rows": {name: len(rows) for name, rows in splits.items()},
        "points": len(results),
        "failed_points": sum(1 for r in results if r["status"] == "failed"),
        "results": results,
        "selection": selection,
        "sensitivity": {
            "classifier": sensitivity(results, CLASSIFIER_METRIC),
            "regressor": sensitivity(results, REGRESSOR_METRIC),
        },
    }
    # Without a measured base, or with no measured point at all, nothing was tuned: no proposal.
    unmeasured = [
        model
        for model, chosen in selection.items()
        if not chosen["base_measured"] or chosen["measured_points"] == 0
    ]
    if unmeasured:
        reason = (
            f"the base point or every point could not be measured for {', '.join(unmeasured)}; "
            "no settings are proposed"
        )
        return finish(
            report | {"status": "failed", "reason": reason}, output, tracking_uri, artifact_root
        )
    report["proposal"] = {"version": next_version(base.version), "file": proposal}
    write_atomic(output / proposal, proposal_text(tuning, base, selection))
    return finish(report, output, tracking_uri, artifact_root)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune the LightGBM candidates on the validation split (never test)."
    )
    parser.add_argument("--split", type=Path, required=True, help="Frozen split file")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tuning", type=Path, default=DEFAULT_TUNING)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--frozen-schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Only configuration problems exit 2; an I/O failure while writing results crashes loudly,
    # as in train_candidates. Unreadable configuration files are TuningErrors.
    config_errors = (TuningError, TrainingError, ContractError, tomllib.TOMLDecodeError)
    try:
        outcome = run(
            args.split,
            args.output,
            tuning_path=args.tuning,
            base_path=args.base,
            contract_path=args.contract,
            frozen_schema=args.frozen_schema,
            tracking_uri=args.tracking_uri,
            artifact_root=args.artifact_root,
        )
    except config_errors as error:
        print(f"Tuning error: {error}", file=sys.stderr)
        return EXIT_ERROR
    print(f"Tuning {outcome.status}; MLflow run {outcome.run_id or 'not logged'}")
    if outcome.run_id is None:
        return EXIT_UNTRACKED
    return 0 if outcome.status == "tuned" else EXIT_BLOCKED


if __name__ == "__main__":
    sys.exit(main())
