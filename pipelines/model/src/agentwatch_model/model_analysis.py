"""Result figures and explanations for evaluated candidates.

From one or more train-candidates folders (`name=folder`) this draws a confusion matrix per model
and for rules-v1, a bar chart comparing them, the chosen model's SHAP feature importance on the
test rows, and, from a tune-candidates report, each setting's validation sensitivity. It trains,
fits and selects nothing: it reads reports, predictions and saved boosters.

SHAP values come from LightGBM's own TreeSHAP (`predict(pred_contrib=True)`), exact for tree
models; per class they add up to the raw score. Importance is the mean absolute SHAP value of each
feature over the test rows, per class and averaged over classes. It describes what drives the
model's scores, not causes of contention.
"""

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import numpy as np
from agentwatch_data.files import file_sha256, json_text, write_atomic
from agentwatch_data.splits import SplitError, load_split

from agentwatch_model.candidates import TrainingError, check_lightgbm
from agentwatch_model.contract import DEFAULT_CONTRACT
from agentwatch_model.features import FEATURES, FeatureError, feature_matrix
from agentwatch_model.metrics import CLASSES, NONE
from agentwatch_model.rule_baseline import DEFAULT_SCHEMA, EXIT_ERROR, split_identity

DEFAULT_OUTPUT = Path(".data/analysis")
ANALYSIS = "analysis.json"
REPORT = "report.json"
CLASSIFIER = "classifier.txt"
PREDICTIONS = "predictions.jsonl"
RULES = "rules-v1"
DPI = 120
# PNG metadata without the Matplotlib version or a timestamp, so equal inputs give equal files.
PNG_METADATA = {"Software": None}
SENSITIVITY_METRICS = (("classifier", "log loss"), ("regressor", "MAE"))


class AnalysisError(ValueError):
    """Inputs that cannot be analysed as given."""


def parse_named(values: list[str]) -> dict[str, Path]:
    named: dict[str, Path] = {}
    for value in values:
        name, separator, folder = value.partition("=")
        if not separator or not name or not folder:
            raise AnalysisError(f"--candidates takes name=folder, got {value!r}")
        if name in named:
            raise AnalysisError(f"--candidates names {name!r} twice")
        named[name] = Path(folder)
    return named


def load_report(folder: Path) -> dict:
    path = folder / REPORT
    try:
        report = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise AnalysisError(f"Cannot read the candidates report {path}: {error}") from error
    if report.get("status") != "evaluated" or report.get("scored_split") != "test":
        raise AnalysisError(f"{path} is not an evaluated test report")
    return report


def check_inputs(
    candidates: dict[str, Path], reports: dict[str, dict], split: Path, rows: list[dict]
) -> None:
    """Every folder must be scored on --split's rows, with the files its report recorded."""
    expected = split_identity(split)
    for name, folder in candidates.items():
        report = reports[name]
        if report.get("split") != expected:
            raise AnalysisError(f"{name}: scored on {report.get('split')}, not {expected}")
        recorded = {
            CLASSIFIER: report["model_files"]["classifier"],
            PREDICTIONS: report["predictions"]["sha256"],
        }
        for file, sha in recorded.items():
            path = folder / file
            if not path.is_file() or file_sha256(path) != sha:
                raise AnalysisError(f"{name}: {path} does not match the hash its report recorded")
        if report.get("scored_steps") != len(rows):
            raise AnalysisError(
                f"{name}: scored {report.get('scored_steps')} test steps, the split has {len(rows)}"
            )


def ordered_classes(matrix: dict) -> list[str]:
    seen = set(matrix) | {label for row in matrix.values() for label in row}
    return [name for name in CLASSES if name in seen]


def clean_none_rate(source: dict) -> float | None:
    matrix = source["slices"]["by_condition"].get("clean", {}).get("confusion_matrix", {})
    row = matrix.get(NONE, {})
    total = sum(row.values())
    return row.get(NONE, 0) / total if total else None


def model_rows(reports: dict[str, dict]) -> dict[str, dict]:
    """Macro F1, the clean-step none rate and MAE per model, and rules-v1's from the first report."""
    first = next(iter(reports.values()))
    rows = {
        RULES: {
            "macro_f1": first["metrics"]["rules"]["macro_f1"]["all_classes"],
            "clean_none_rate": clean_none_rate(first["metrics"]["rules"]),
            "mae": None,
        }
    }
    for name, report in reports.items():
        rows[name] = {
            "training_version": report["model"]["version"],
            "macro_f1": report["metrics"]["model"]["macro_f1"]["all_classes"],
            "clean_none_rate": clean_none_rate(report["metrics"]["model"]),
            "mae": report["gates"]["regression_error"].get("value"),
        }
    return rows


def pyplot():
    # A notebook kernel exports MPLBACKEND for its inline backend, which a subprocess such as
    # `make model-analysis` run from a cell cannot load; these figures are files, so use Agg.
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save(figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=DPI, metadata=PNG_METADATA)
    pyplot().close(figure)


def confusion_figure(matrix: dict, title: str, path: Path) -> None:
    plt = pyplot()
    classes = ordered_classes(matrix)
    counts = np.array([[matrix.get(t, {}).get(p, 0) for p in classes] for t in classes])
    totals = counts.sum(axis=1, keepdims=True)
    shares = np.divide(counts, totals, out=np.zeros(counts.shape), where=totals > 0)
    figure, axis = plt.subplots(figsize=(5.2, 4.4))
    axis.imshow(shares, cmap="Blues", vmin=0, vmax=1)
    for i, j in np.ndindex(counts.shape):
        colour = "white" if shares[i, j] > 0.6 else "black"
        axis.text(
            j,
            i,
            f"{counts[i, j]}\n{shares[i, j]:.0%}",
            ha="center",
            va="center",
            fontsize=8,
            color=colour,
        )
    axis.set_xticks(range(len(classes)), classes, rotation=30, ha="right")
    axis.set_yticks(range(len(classes)), classes)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Label")
    axis.set_title(title, fontsize=10)
    save(figure, path)


def comparison_figure(rows: dict[str, dict], path: Path) -> None:
    plt = pyplot()
    measures = (("macro_f1", "Macro F1"), ("clean_none_rate", "Clean steps predicted none"))
    names = list(rows)
    figure, axes = plt.subplots(1, len(measures), figsize=(8, 3.6))
    for axis, (key, label) in zip(axes, measures, strict=True):
        values = [rows[name][key] for name in names]
        heights = [value if value is not None else 0 for value in values]
        bars = axis.bar(names, heights, color=["#9e9e9e"] + ["#1f77b4"] * (len(names) - 1))
        for bar, value in zip(bars, values, strict=True):
            text = "n/a" if value is None else f"{value:.3f}"
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                text,
                ha="center",
                va="bottom",
                fontsize=8,
            )
        axis.set_ylim(0, 1.08)
        axis.set_title(label, fontsize=10)
    figure.suptitle("Test split, identical rows", fontsize=10)
    save(figure, path)


def shap_importance(booster, rows: list[dict], class_order: tuple[str, ...]) -> dict:
    """Mean |SHAP| per feature over the rows, per class and averaged over classes."""
    matrix = feature_matrix(rows)
    contributions = np.asarray(booster.predict(matrix, pred_contrib=True))
    per_class = contributions.reshape(len(rows), len(class_order), len(FEATURES) + 1)[:, :, :-1]
    mean_abs = np.abs(per_class).mean(axis=0)
    by_class = {
        name: {feature: float(mean_abs[k, f]) for f, feature in enumerate(FEATURES)}
        for k, name in enumerate(class_order)
    }
    overall = {feature: float(mean_abs[:, f].mean()) for f, feature in enumerate(FEATURES)}
    return {"features": list(FEATURES), "rows": len(rows), "overall": overall, "by_class": by_class}


def shap_figure(importance: dict, title: str, path: Path) -> None:
    plt = pyplot()
    ranked = sorted(importance["overall"].items(), key=lambda item: item[1])
    figure, axis = plt.subplots(figsize=(6.4, 4.6))
    axis.barh([name for name, _ in ranked], [value for _, value in ranked], color="#1f77b4")
    axis.set_xlabel("Mean |SHAP value| (raw score), averaged over classes")
    axis.set_title(title, fontsize=10)
    save(figure, path)


def numeric_order(spread: dict) -> list[str]:
    return sorted(spread, key=float)


def sensitivity_figure(tuning: dict, path: Path) -> None:
    plt = pyplot()
    settings = sorted(
        {name for model, _ in SENSITIVITY_METRICS for name in tuning["sensitivity"][model]}
    )
    figure, axes = plt.subplots(
        len(SENSITIVITY_METRICS), len(settings), figsize=(13, 5.2), squeeze=False
    )
    for row, (model, label) in enumerate(SENSITIVITY_METRICS):
        for column, setting in enumerate(settings):
            axis = axes[row][column]
            spread = tuning["sensitivity"][model].get(setting)
            if not spread:
                axis.set_axis_off()  # this model's grid has no such setting
                continue
            # JSON keys come back sorted as text (10 before 5); plot them in numeric order.
            values = numeric_order(spread)
            medians = [spread[v]["median"] for v in values]
            low = [spread[v]["median"] - spread[v]["min"] for v in values]
            high = [spread[v]["max"] - spread[v]["median"] for v in values]
            axis.errorbar(range(len(values)), medians, yerr=[low, high], fmt="o-", capsize=3)
            axis.set_xticks(range(len(values)), values)
            axis.tick_params(labelsize=7)
            if row == len(SENSITIVITY_METRICS) - 1:
                axis.set_xlabel(setting, fontsize=8)
            if column == 0:
                axis.set_ylabel(f"{model} {label}", fontsize=8)
    figure.suptitle(
        f"Validation sensitivity ({tuning['tuning']['version']}): median, min to max over the grid",
        fontsize=10,
    )
    save(figure, path)


def read_tuning(path: Path | None) -> dict | None:
    if path is None:
        return None
    try:
        tuning = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise AnalysisError(f"Cannot read the tuning report {path}: {error}") from error
    if tuning.get("status") != "tuned" or "sensitivity" not in tuning:
        raise AnalysisError(f"{path} is not a completed tune-candidates report")
    return tuning


def run(
    candidates: dict[str, Path],
    split: Path,
    output: Path,
    *,
    explain: str,
    tuning_report: Path | None = None,
    contract_path: Path = DEFAULT_CONTRACT,
    frozen_schema: Path = DEFAULT_SCHEMA,
    loader=None,
) -> Path:
    """Write the figures and analysis.json; return its path. Every input is read first."""
    if explain not in candidates:
        raise AnalysisError(f"--explain {explain!r} is not one of {sorted(candidates)}")
    reports = {name: load_report(folder) for name, folder in candidates.items()}
    tuning = read_tuning(tuning_report)
    check_lightgbm()
    import lightgbm

    rows = (loader or load_split)(split, "test", contract_path, frozen_schema)
    check_inputs(candidates, reports, split, rows)
    booster = lightgbm.Booster(model_file=str(candidates[explain] / CLASSIFIER))
    class_order = tuple(reports[explain]["model"]["class_order"])
    importance = shap_importance(booster, rows, class_order)
    output.mkdir(parents=True, exist_ok=True)
    first = next(iter(reports.values()))
    figures = {"confusion-rules.png": None}
    confusion_figure(
        first["metrics"]["rules"]["confusion_matrix"],
        f"{RULES}, test",
        output / "confusion-rules.png",
    )
    for name, report in reports.items():
        file = f"confusion-{name}.png"
        title = f"{name} ({report['model']['version']}), test"
        confusion_figure(report["metrics"]["model"]["confusion_matrix"], title, output / file)
        figures[file] = None
    models = model_rows(reports)
    comparison_figure(models, output / "comparison.png")
    shap_file = f"shap-{explain}.png"
    shap_figure(importance, f"{explain} classifier SHAP importance, test rows", output / shap_file)
    figures |= {"comparison.png": None, shap_file: None}
    sensitivity = None
    if tuning is not None:
        sensitivity_figure(tuning, output / "sensitivity.png")
        figures["sensitivity.png"] = None
        sensitivity = {
            "tuning_version": tuning["tuning"]["version"],
            "selection": tuning["selection"],
            "median_spread": {
                model: {
                    setting: max(v["median"] for v in by_value.values())
                    - min(v["median"] for v in by_value.values())
                    for setting, by_value in tuning["sensitivity"][model].items()
                }
                for model in ("classifier", "regressor")
            },
        }
    ranked = sorted(importance["overall"], key=importance["overall"].get, reverse=True)
    analysis = {
        "evaluation": False,
        "split": {"file": split.name, "sha256": file_sha256(split) if split.is_file() else None},
        "models": models,
        "explained": {"name": explain, "training_version": reports[explain]["model"]["version"]},
        "shap": importance
        | {"ranking": ranked, "median_overall": statistics.median(importance["overall"].values())},
        "sensitivity": sensitivity,
        "figures": [{"file": f, "sha256": file_sha256(output / f)} for f in sorted(figures)],
    }
    write_atomic(output / ANALYSIS, json_text(analysis))
    return output / ANALYSIS


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Figures and SHAP for evaluated candidates.")
    parser.add_argument("--split", type=Path, required=True, help="Frozen split file")
    parser.add_argument(
        "--candidates", nargs="+", required=True, help="name=train-candidates folder"
    )
    parser.add_argument("--explain", required=True, help="Which named candidates to explain")
    parser.add_argument("--tuning-report", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--frozen-schema", type=Path, default=DEFAULT_SCHEMA)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        path = run(
            parse_named(args.candidates),
            args.split,
            args.output,
            explain=args.explain,
            tuning_report=args.tuning_report,
            contract_path=args.contract,
            frozen_schema=args.frozen_schema,
        )
    except (AnalysisError, TrainingError, SplitError, FeatureError) as error:
        print(f"Analysis error: {error}", file=sys.stderr)
        return EXIT_ERROR
    print(f"Analysis written to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
