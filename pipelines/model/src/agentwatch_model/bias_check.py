"""Model-level bias check on the evaluation slices, with a measured mitigation.

The dataset has no demographic attributes; its slices are the conditions a model could serve
unevenly (the data slices of `agentwatch_data.slices`, #48, at the data level). For a model, each
slice's correct-attribution rate is compared with the rate over every scored step:

- condition: baseline, clean or contended run;
- telemetry_readings: whether any of the data slices' nullable readings (telemetry coverage, CPU
  PSI, cgroup throttling) is missing, or all are present;
- contention: on contended steps with a resource label, single-class or mixed.

A step is correct when the prediction is its class; a mixed step when the prediction names one of
its confirmed resources (mixed-scoring-v1); joint-only mixed steps are never scored. Thresholds
and the mitigation trade-off are declared in bias.toml before the check runs.

The mitigation trains the same settings with condition-balancing weights. Both models are
checked on validation, which decides; test is checked and reported for both, never used to
choose. The check runs on the packaged training settings and writes no model.
"""

import argparse
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

from agentwatch_data.extract import canonical_sha256
from agentwatch_data.files import file_sha256, json_text, write_atomic
from agentwatch_data.slices import READINGS
from agentwatch_data.splits import SplitError, load_split

from agentwatch_model.candidates import (
    HASH_DIGITS,
    WEIGHT_ATTRIBUTES,
    CandidateError,
    TrainingError,
    check_lightgbm,
    fit_candidate,
    load_training_config,
    predict,
)
from agentwatch_model.contract import DEFAULT_CONTRACT
from agentwatch_model.features import FeatureError
from agentwatch_model.metrics import NONE, is_number
from agentwatch_model.mixed import is_joint_only, is_mixed, members
from agentwatch_model.model_analysis import pyplot, save
from agentwatch_model.rule_baseline import (
    DEFAULT_SCHEMA,
    EXIT_ERROR,
    RowError,
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
from agentwatch_model.tune_candidates import validation_metrics

PACKAGE = Path(__file__).parent
DEFAULT_BIAS = PACKAGE / "bias.toml"
# The unweighted settings the mitigation is decided on: training-v2, which training-v3 weights.
DEFAULT_TRAINING = PACKAGE / "training-v2.toml"
DEFAULT_OUTPUT = Path(".data/bias")
VERSION = "bias-v1"
REPORT = "report.json"
PROPOSAL = "training-weighted.toml"
TRACKING = "tracking.json"
EXPERIMENT = "agentwatch-bias"
TASK = "M04-bias"
EXIT_UNTRACKED = 3
ATTRIBUTES = ("condition", "telemetry_readings", "contention")
CONTENDED = "contended"
MODELS = ("unweighted", "weighted")
LIMITATIONS = (
    (
        "Validation, which decides the mitigation, has 273 steps from 3 experiments of one task; "
        "its contended, mixed and complete-telemetry slices are small."
    ),
    (
        "Test is one unseen task. Slice rates are descriptive, with no interval, and a disparity "
        "is a difference in observed rates, not a cause."
    ),
    (
        "Condition slices differ in difficulty by design: contended steps carry the resource "
        "labels, clean and baseline steps are almost all `none`."
    ),
)


def weighted_settings_text(base_text: str, base_version: str, weight_by: str) -> str:
    """The base training file with the next training version and condition weighting."""
    matched = re.fullmatch(r"training-v([0-9]+)\+[0-9a-f]+", base_version)
    if matched is None:
        raise BiasError(f"Cannot derive the next training version from {base_version!r}")
    number = int(matched.group(1)) + 1
    text = re.sub(
        r'^version = "training-v[0-9]+"',
        f'version = "training-v{number}"',
        base_text,
        count=1,
        flags=re.MULTILINE,
    )
    header = (
        f"# training-v{number}: {base_version}'s settings trained with {weight_by}-balancing\n"
        "# weights, written by bias-check (bias-v1) as the mitigation candidate.\n"
    )
    return header + text.replace(
        "\nnum_boost_round =", f'\nweight_by = "{weight_by}"\nnum_boost_round =', 1
    )


class BiasError(ValueError):
    """A bias configuration that cannot be checked as declared."""


@dataclass(frozen=True)
class BiasConfig:
    version: str
    max_disparity: float
    min_slice_steps: int
    weight_by: str
    min_disparity_reduction: float
    max_log_loss_increase: float


@dataclass(frozen=True)
class Outcome:
    status: str
    run_id: str | None


def fraction(table: dict, key: str, low: float, high: float) -> float:
    value = table.get(key)
    if not is_number(value) or not low <= value <= high:
        raise BiasError(f"{key} must be a number in [{low}, {high}]")
    return float(value)


def load_bias_config(path: Path = DEFAULT_BIAS) -> BiasConfig:
    parsed = tomllib.loads(path.read_text())
    if parsed.get("version") != VERSION:
        raise BiasError(f"{path.name}: version must be {VERSION!r}")
    thresholds = parsed.get("thresholds") or {}
    mitigation = parsed.get("mitigation") or {}
    steps = thresholds.get("min_slice_steps")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise BiasError("min_slice_steps must be a positive integer")
    weight_by = mitigation.get("weight_by")
    if weight_by not in WEIGHT_ATTRIBUTES:
        raise BiasError(f"weight_by must be one of {list(WEIGHT_ATTRIBUTES)}, got {weight_by!r}")
    return BiasConfig(
        version=f"{VERSION}+{canonical_sha256(parsed)[:HASH_DIGITS]}",
        max_disparity=fraction(thresholds, "max_disparity", 0, 1),
        min_slice_steps=steps,
        weight_by=weight_by,
        min_disparity_reduction=fraction(mitigation, "min_disparity_reduction", 0, 1),
        max_log_loss_increase=fraction(mitigation, "max_log_loss_increase", 0, 10),
    )


def slice_values(row: dict) -> dict[str, str | None]:
    """The step's value on each slice attribute; None when the attribute does not apply."""
    readings = "missing_any" if any(row.get(name) is None for name in READINGS) else "complete"
    contention = None
    if row["condition"] == CONTENDED and row["label_resource"] != NONE:
        contention = "mixed" if is_mixed(row) else "single_class"
    return {
        "condition": str(row["condition"]),
        "telemetry_readings": readings,
        "contention": contention,
    }


def correct(row: dict, prediction: str, classes: tuple[str, ...]) -> bool | None:
    """Whether the prediction attributes the step correctly; None for joint-only steps."""
    if is_joint_only(row):
        return None
    if is_mixed(row):
        return prediction in members(row, classes)
    return prediction == row["label_resource"]


def rate(outcomes: list[bool]) -> float:
    return sum(outcomes) / len(outcomes)


def slice_report(
    rows: list[dict], predictions: list[str], classes: tuple[str, ...], config: BiasConfig
) -> dict:
    """The overall correct rate and, per slice, its rate, disparity and outcome."""
    scored = [
        (row, ok)
        for row, prediction in zip(rows, predictions, strict=True)
        if (ok := correct(row, prediction, classes)) is not None
    ]
    if not scored:
        raise BiasError("No scored step to check")
    overall = rate([ok for _, ok in scored])
    groups: dict[str, dict[str, list[bool]]] = {name: {} for name in ATTRIBUTES}
    for row, ok in scored:
        for attribute, value in slice_values(row).items():
            if value is not None:
                groups[attribute].setdefault(value, []).append(ok)
    attributes: dict[str, dict] = {}
    flags = []
    disparities = []
    for attribute, by_value in groups.items():
        attributes[attribute] = {}
        for value, outcomes in sorted(by_value.items()):
            slice_rate = rate(outcomes)
            disparity = overall - slice_rate
            if len(outcomes) < config.min_slice_steps:
                verdict = "insufficient"
            else:
                disparities.append(disparity)
                verdict = "flagged" if disparity > config.max_disparity else "pass"
            if verdict == "flagged":
                flags.append(
                    {"attribute": attribute, "value": value, "disparity": round(disparity, 6)}
                )
            attributes[attribute][value] = {
                "steps": len(outcomes),
                "correct_rate": slice_rate,
                "disparity": disparity,
                "outcome": verdict,
            }
    return {
        "overall": {"steps": len(scored), "correct_rate": overall},
        "attributes": attributes,
        "flags": flags,
        "worst_disparity": max(disparities) if disparities else None,
        "outcome": "fail" if flags else "pass",
    }


def choose(
    unweighted: dict,
    weighted: dict,
    unweighted_loss: float,
    weighted_loss: float,
    config: BiasConfig,
) -> dict:
    """The declared validation decision between the unweighted and the weighted model."""
    decision = {"decided_on": "validation", "adopted": False, "selected": "unweighted"}
    if unweighted["outcome"] == "pass":
        return decision | {"reason": "no slice is flagged on validation; no mitigation needed"}
    reduction = unweighted["worst_disparity"] - weighted["worst_disparity"]
    increase = (weighted_loss - unweighted_loss) / unweighted_loss
    decision |= {"disparity_reduction": reduction, "log_loss_increase": increase}
    if reduction < config.min_disparity_reduction:
        return decision | {"reason": "weighting does not lower the worst disparity enough"}
    if increase > config.max_log_loss_increase:
        return decision | {"reason": "weighting raises validation log loss past the trade-off"}
    return decision | {
        "adopted": True,
        "selected": "weighted",
        "reason": "weighting lowers the worst disparity within the declared log-loss trade-off",
    }


def bias_figure(reports: dict[str, dict], split: str, config: BiasConfig, path: Path) -> None:
    plt = pyplot()
    figure, axes = plt.subplots(1, len(ATTRIBUTES), figsize=(12, 3.8), sharey=True)
    width = 0.38
    for axis, attribute in zip(axes, ATTRIBUTES, strict=True):
        values = sorted({v for r in reports.values() for v in r["attributes"][attribute]})
        for offset, (name, report) in enumerate(reports.items()):
            slices = report["attributes"][attribute]
            heights = [slices[v]["correct_rate"] if v in slices else 0 for v in values]
            positions = [i + (offset - 0.5) * width for i in range(len(values))]
            axis.bar(positions, heights, width, label=name)
            line = report["overall"]["correct_rate"] - config.max_disparity
            axis.axhline(line, linestyle="--", linewidth=0.8, color=f"C{offset}")
        axis.set_xticks(range(len(values)), values, rotation=20)
        axis.set_title(attribute, fontsize=10)
        axis.set_ylim(0, 1.05)
    axes[0].set_ylabel("Correct-attribution rate")
    axes[-1].legend(fontsize=8, loc="lower right")
    figure.suptitle(
        f"{split}: slice rates; dashed lines are each model's overall rate minus "
        f"{config.max_disparity}",
        fontsize=10,
    )
    save(figure, path)


def log_bias(report: dict, directory: Path, tracking_uri: str, artifact_root: Path) -> str:
    import mlflow

    if tracking_uri.startswith(SQLITE_PREFIX):
        Path(tracking_uri.removeprefix(SQLITE_PREFIX)).parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run(experiment_id=experiment_id(artifact_root, EXPERIMENT)) as active:
        mlflow.set_tags({"task": TASK, "status": report["status"], "evaluation": "False"})
        mlflow.log_params(
            {
                "bias_version": report["bias"]["version"],
                "training_version": report["models"]["unweighted"]["training_version"],
                "selected": report["decision"]["selected"],
            }
        )
        metrics = {}
        for name, model in report["models"].items():
            for split in ("validation", "test"):
                metrics[f"{name}_{split}_correct_rate"] = model[split]["overall"]["correct_rate"]
                worst = model[split]["worst_disparity"]
                if worst is not None:
                    metrics[f"{name}_{split}_worst_disparity"] = worst
        mlflow.log_metrics(metrics)
        for name in (REPORT, "bias-validation.png", "bias-test.png"):
            if (directory / name).is_file():
                mlflow.log_artifact(str(directory / name))
        return active.info.run_id


def run(
    split: Path,
    output: Path,
    *,
    bias_path: Path = DEFAULT_BIAS,
    training_path: Path = DEFAULT_TRAINING,
    contract_path: Path = DEFAULT_CONTRACT,
    frozen_schema: Path = DEFAULT_SCHEMA,
    loader=None,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> Outcome:
    """Train both models, check both on validation and test, decide on validation."""
    config = load_bias_config(bias_path)
    base = load_training_config(training_path)
    check_lightgbm()
    load = loader or load_split
    splits = {
        name: load(split, name, contract_path, frozen_schema)
        for name in ("train", "validation", "test")
    }
    for name, rows in splits.items():
        validate_rows(rows, name)
    validate_dataset_version(splits["train"] + splits["validation"] + splits["test"])
    if base.weight_by is not None:
        raise BiasError(f"{training_path.name} is already weighted; check unweighted settings")
    output.mkdir(parents=True, exist_ok=True)
    proposal = output / PROPOSAL
    write_atomic(
        proposal, weighted_settings_text(training_path.read_text(), base.version, config.weight_by)
    )
    variants = {"unweighted": base, "weighted": load_training_config(proposal)}
    models = {}
    for name, settings in variants.items():
        candidate = fit_candidate(splits["train"], splits["validation"], settings)
        classifier, _ = validation_metrics(candidate, splits["validation"])
        models[name] = {
            "training_version": settings.version,
            "weight_by": settings.weight_by,
            "validation_log_loss": classifier["log_loss"] if classifier else None,
        }
        for split_name in ("validation", "test"):
            labels = [p.label for p in predict(candidate, splits[split_name])]
            models[name][split_name] = slice_report(
                splits[split_name], labels, candidate.class_order, config
            )
    unweighted, weighted = models["unweighted"], models["weighted"]
    decision = choose(
        unweighted["validation"],
        weighted["validation"],
        unweighted["validation_log_loss"],
        weighted["validation_log_loss"],
        config,
    )
    for name in ("validation", "test"):
        reports = {model: models[model][name] for model in MODELS}
        bias_figure(reports, name, config, output / f"bias-{name}.png")
    report = {
        "status": "checked",
        "evaluation": False,
        "bias": {
            "version": config.version,
            "max_disparity": config.max_disparity,
            "min_slice_steps": config.min_slice_steps,
            "weight_by": config.weight_by,
            "min_disparity_reduction": config.min_disparity_reduction,
            "max_log_loss_increase": config.max_log_loss_increase,
        },
        "split": {"file": split.name, "sha256": file_sha256(split) if split.is_file() else None},
        "models": models,
        "decision": decision | {"settings_file": PROPOSAL if decision["adopted"] else None},
        "figures": {
            name: file_sha256(output / name) for name in ("bias-validation.png", "bias-test.png")
        },
        "limitations": list(LIMITATIONS),
    }
    write_atomic(output / REPORT, json_text(report))
    try:
        run_id = log_bias(report, output, tracking_uri, artifact_root)
    except tracking_errors() as error:
        write_atomic(output / TRACKING, json_text({"logged": False, "error": str(error)}))
        print(f"Tracking error, report kept: {error}", file=sys.stderr)
        return Outcome(report["status"], None)
    write_atomic(output / TRACKING, json_text({"logged": True, "run_id": run_id}))
    return Outcome(report["status"], run_id)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Model-level bias check with a measured mitigation."
    )
    parser.add_argument("--split", type=Path, required=True, help="Frozen split file")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bias", type=Path, default=DEFAULT_BIAS)
    parser.add_argument("--training", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--frozen-schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    errors = (
        BiasError,
        TrainingError,
        CandidateError,
        FeatureError,
        RowError,
        SplitError,
        tomllib.TOMLDecodeError,
        OSError,
    )
    try:
        outcome = run(
            args.split,
            args.output,
            bias_path=args.bias,
            training_path=args.training,
            contract_path=args.contract,
            frozen_schema=args.frozen_schema,
            tracking_uri=args.tracking_uri,
            artifact_root=args.artifact_root,
        )
    except errors as error:
        print(f"Bias check error: {error}", file=sys.stderr)
        return EXIT_ERROR
    print(f"Bias check {outcome.status}; MLflow run {outcome.run_id or 'not logged'}")
    return 0 if outcome.run_id else EXIT_UNTRACKED


if __name__ == "__main__":
    sys.exit(main())
