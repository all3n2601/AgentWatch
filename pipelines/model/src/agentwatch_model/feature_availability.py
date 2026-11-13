"""How the frozen classifier's inputs differ between its training rows and held-out rows.

For every model feature this reports, on split-v3's train rows and on the held-out sweep rows
that `score-held-out` scores (overall and per configuration), how many readings are missing, how
many distinct values the rest take, their quantiles and range, and the share of held-out values
outside the train range, beside the feature's share of the classifier's split gain. It reads
the frozen M03/M04 pair through the same checked loader as `score-held-out`.

The availability counts (rows, missing fraction, distinct values, most common share) come from
`held_out_features.availability`, the same counting `score-held-out` reports; this module adds
the classifier's own train rows, gain shares, two flags, quantiles and the train range.

Descriptive only: nothing is scored, trained, fitted or tuned, and no threshold is applied.
A feature that is missing or constant on held-out rows is a possible reason for held-out
errors, not a measured cause of them.
"""

import argparse
import math
import sys
import tomllib
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from agentwatch_data.files import json_text, write_atomic
from agentwatch_data.splits import SplitError, load_split

from agentwatch_model.candidates import TrainingError
from agentwatch_model.contract import DEFAULT_CONTRACT, ContractError
from agentwatch_model.features import FEATURES, FeatureError, encode
from agentwatch_model.held_out_features import availability
from agentwatch_model.held_out_model import (
    FROZEN_VERSIONS,
    FrozenVersions,
    HeldOutModelError,
    check_reproduces_calibration,
    check_reproduces_validation,
    load_frozen_classifier,
)
from agentwatch_model.held_out_rows import HeldOutError, held_out_rows
from agentwatch_model.rule_baseline import DEFAULT_SCHEMA, EXIT_ERROR, split_identity
from agentwatch_model.score_held_out import (
    DEFAULT_CALIBRATED,
    DEFAULT_CANDIDATES,
    SPLIT_NAMES,
    TRUTH_IDENTITY_FIELDS,
    OutputError,
    check_output,
    scored_overlap,
)

DEFAULT_OUTPUT = Path(".data/a06-scoring/feature-availability")
DEFAULT_SWEEP_ROOT = Path("data/held-out/a06-20261111/capacity-sweeps")
REPORT_FILE = "feature-availability.json"
TABLE_FILE = "feature-availability.md"
GAIN_IMPORTANCE = "gain"
MISSING_ON_HELD_OUT_ONLY = "missing_on_held_out_only"
CONSTANT_ON_HELD_OUT_ONLY = "constant_on_held_out_only"
# Quantiles of the present values, by name; numpy's default linear interpolation.
QUANTILES = {"p05": 0.05, "p50": 0.5, "p95": 0.95}
METHOD = {
    "train": "split-v3 train rows the classifier was fitted on: single-class steps of its class "
    "order; the others (mixed steps) are counted in train_not_classified",
    "missing": "a null reading, encoded as NaN exactly as the classifier receives it; a "
    "measured zero is present",
    "distinct_values": "distinct present values after the classifier's encoding",
    "constant": "exactly one distinct present value",
    "gain_share": "the feature's share of the frozen classifier's total split gain",
    "quantiles": "p05, p50 and p95 of the present values (linear interpolation); null when "
    "every value is missing",
    "outside_train_range": "shares of the present held-out values below the train minimum and "
    "above the train maximum; null unless both sides have a present value",
    MISSING_ON_HELD_OUT_ONLY: "some train value is present and every held-out value is missing",
    CONSTANT_ON_HELD_OUT_ONLY: "train values vary and held-out values are constant",
}
LIMITATIONS = (
    "Descriptive only: nothing is scored, trained, fitted or tuned on these rows.",
    "A missing or constant feature is a possible reason for held-out errors, not a cause.",
    (
        "Gain share describes what the classifier used in training, not what changes its "
        "held-out predictions."
    ),
    (
        "Train rows are dataset v1 (mostly Ollama); held-out rows are the A06 vLLM sweeps on "
        "one laptop."
    ),
)


class AvailabilityError(ValueError):
    """Rows or gain shares that cannot be described as recorded."""


def present_values(rows: list[dict], name: str) -> tuple[list[float], int]:
    """The present values of one feature, encoded as the classifier receives them, and the
    missing count."""
    if not rows:
        raise AvailabilityError(f"Cannot describe {name}: no rows")
    present = []
    for index, row in enumerate(rows):
        if name not in row:
            raise FeatureError(f"Row {index} is missing feature {name}")
        value = encode(row[name], name, index)
        if not math.isnan(value):
            present.append(value)
    return present, len(rows) - len(present)


def describe_feature(rows: list[dict], name: str) -> dict:
    """Missing count and fraction, distinct present values, quantiles and range, and the
    constant and all-missing flags of one feature over rows, using the classifier's own
    encoding."""
    present, missing = present_values(rows, name)
    counted = availability(rows, name)
    distinct = counted["distinct_values"]
    quantiles = None
    if present:
        values = np.quantile(np.array(present), list(QUANTILES.values()))
        quantiles = {key: float(value) for key, value in zip(QUANTILES, values, strict=True)}
    return counted | {
        "missing": missing,
        "constant": distinct == 1,
        "all_missing": not present,
        "quantiles": quantiles,
        "min": min(present) if present else None,
        "max": max(present) if present else None,
    }


def outside_train_range(train: dict, present: list[float]) -> dict | None:
    """Shares of the present held-out values below the train minimum and above its maximum."""
    if train["min"] is None or not present:
        return None
    return {
        "below_train_min": sum(value < train["min"] for value in present) / len(present),
        "above_train_max": sum(value > train["max"] for value in present) / len(present),
    }


def gain_shares(booster) -> dict[str, float | None]:
    """Each feature's share of the booster's total split gain, in FEATURES order; every share
    is None when the booster recorded no gain."""
    names = list(booster.feature_name())
    if sorted(names) != sorted(FEATURES):
        raise AvailabilityError(f"The classifier's features {names} are not {list(FEATURES)}")
    values = [float(value) for value in booster.feature_importance(importance_type=GAIN_IMPORTANCE)]
    gains = dict(zip(names, values, strict=True))
    for name, value in gains.items():
        if not math.isfinite(value) or value < 0:
            raise AvailabilityError(f"The classifier's gain for {name} is {value!r}, not >= 0")
    total = sum(gains.values())
    if total <= 0:
        return dict.fromkeys(FEATURES)
    return {name: float(gains[name]) / total for name in FEATURES}


def held_out_flags(train: dict, held_out: dict) -> list[str]:
    flags = []
    if not train["all_missing"] and held_out["all_missing"]:
        flags.append(MISSING_ON_HELD_OUT_ONLY)
    if not train["all_missing"] and not train["constant"] and held_out["constant"]:
        flags.append(CONSTANT_ON_HELD_OUT_ONLY)
    return flags


def compared(train: dict, rows: list[dict], name: str) -> dict:
    described = describe_feature(rows, name)
    present, _ = present_values(rows, name)
    return described | {
        "missing_fraction_gap": described["missing_fraction"] - train["missing_fraction"],
        "outside_train_range": outside_train_range(train, present),
        "flags": held_out_flags(train, described),
    }


def by_configuration(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for index, row in enumerate(rows):
        configuration = row.get("configuration")
        if not isinstance(configuration, str) or not configuration:
            raise AvailabilityError(f"Held-out row {index} names no configuration")
        groups[configuration].append(row)
    return dict(sorted(groups.items()))


def gain_order(item: dict) -> tuple:
    share = item["gain_share"]
    return (share is None, -(share or 0.0), item["feature"])


def availability_report(
    train: list[dict],
    held_out: list[dict],
    gain: Mapping[str, float | None],
    *,
    train_not_classified: int = 0,
) -> dict:
    """The comparison body: row counts and one entry per feature, by gain share."""
    if set(gain) != set(FEATURES):
        raise AvailabilityError(f"The gain shares name {sorted(gain)}, not the model's features")
    groups = by_configuration(held_out)
    features = []
    for name in FEATURES:
        described = describe_feature(train, name)
        features.append(
            {
                "feature": name,
                "gain_share": gain[name],
                "train": described,
                "held_out": compared(described, held_out, name),
                "held_out_by_configuration": {
                    configuration: compared(described, rows, name)
                    for configuration, rows in groups.items()
                },
            }
        )
    return {
        "evaluation": False,
        "held_out": True,
        "row_sets": {
            "train": len(train),
            "train_not_classified": train_not_classified,
            "held_out": len(held_out),
            "held_out_by_configuration": {name: len(rows) for name, rows in groups.items()},
        },
        "features": sorted(features, key=gain_order),
        "method": METHOD,
        "limitations": list(LIMITATIONS),
    }


def percent(fraction: float) -> str:
    return f"{fraction:.1%}"


def markdown_table(report: dict) -> str:
    configurations = list(report["row_sets"]["held_out_by_configuration"])
    header = ["Feature", "Gain share", "Train missing", "Held-out missing"]
    header += [f"{name} missing" for name in configurations]
    header += ["Train distinct", "Held-out distinct", "Held-out flags"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for item in report["features"]:
        share = item["gain_share"]
        flags = sorted(
            set(item["held_out"]["flags"])
            | {
                f"{flag} ({name})"
                for name, described in item["held_out_by_configuration"].items()
                for flag in described["flags"]
                if flag not in item["held_out"]["flags"]
            }
        )
        cells = [
            f"`{item['feature']}`",
            "n/a" if share is None else f"{share:.3f}",
            percent(item["train"]["missing_fraction"]),
            percent(item["held_out"]["missing_fraction"]),
        ]
        cells += [
            percent(item["held_out_by_configuration"][name]["missing_fraction"])
            for name in configurations
        ]
        cells += [
            str(item["train"]["distinct_values"]),
            str(item["held_out"]["distinct_values"]),
            ", ".join(flags) or "none",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def number(value: float) -> str:
    return f"{value:.4g}"


def spread(described: dict) -> str:
    quantiles = described["quantiles"]
    return " / ".join(number(quantiles[key]) for key in QUANTILES)


def range_table(report: dict) -> str:
    """Quantiles of each feature present on both sides, and the held-out shares outside the
    train range."""
    header = [
        "Feature",
        "Gain share",
        "Train p5 / p50 / p95",
        "Held-out p5 / p50 / p95",
        "Held-out below train min",
        "Held-out above train max",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for item in report["features"]:
        outside = item["held_out"]["outside_train_range"]
        if outside is None:
            continue
        share = item["gain_share"]
        cells = [
            f"`{item['feature']}`",
            "n/a" if share is None else f"{share:.3f}",
            spread(item["train"]),
            spread(item["held_out"]),
            percent(outside["below_train_min"]),
            percent(outside["above_train_max"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def run(
    truth_dirs: list[Path],
    sweep_root: Path,
    split: Path,
    output: Path,
    *,
    candidates_dir: Path = DEFAULT_CANDIDATES,
    calibrated_dir: Path = DEFAULT_CALIBRATED,
    contract_path: Path = DEFAULT_CONTRACT,
    frozen_schema: Path = DEFAULT_SCHEMA,
    loader=load_split,
    versions: FrozenVersions = FROZEN_VERSIONS,
) -> Path:
    """Describe the frozen classifier's features on train and held-out rows; return the
    report's path. Every check runs before anything is written. `versions` defaults to the
    pinned frozen pair, as in score-held-out; only tests of synthetic pairs pass others."""
    check_output(output)
    model = load_frozen_classifier(candidates_dir, calibrated_dir, split, versions)
    rows, refused, summaries, unjudged = held_out_rows(truth_dirs, sweep_root)
    if not rows:
        raise AvailabilityError("no held-out step has a truth to describe")
    # The same model and row checks as score-held-out, so the report describes only rows that
    # it would score.
    splits = {name: loader(split, name, contract_path, frozen_schema) for name in SPLIT_NAMES}
    check_reproduces_validation(model, splits["validation"], candidates_dir)
    check_reproduces_calibration(model, splits["test"], calibrated_dir)
    overlap = scored_overlap(rows, [row for part in splits.values() for row in part])
    if overlap:
        raise AvailabilityError(f"held-out steps appear in the split: {', '.join(overlap)}")
    train = [row for row in splits["train"] if row.get("label_resource") in model.class_order]
    gain = gain_shares(model.booster)
    not_classified = len(splits["train"]) - len(train)
    report = availability_report(train, rows, gain, train_not_classified=not_classified) | {
        "model": {"frozen": model.identity},
        "split": split_identity(split),
        "truths": {
            summary["sweep_folder"]: {name: summary[name] for name in TRUTH_IDENTITY_FIELDS}
            for summary in summaries
        },
        "refused": refused,
        "unjudged_steps": len(unjudged),
    }
    table = markdown_table(report) + "\n" + range_table(report)
    output.mkdir(parents=True, exist_ok=True)
    write_atomic(output / TABLE_FILE, table)
    write_atomic(output / REPORT_FILE, json_text(report))
    return output / REPORT_FILE


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the frozen classifier's feature availability on train and "
        "held-out rows (descriptive; nothing is scored or fitted)."
    )
    parser.add_argument("--truth", type=Path, nargs="+", required=True)
    parser.add_argument("--sweep-root", type=Path, default=DEFAULT_SWEEP_ROOT)
    parser.add_argument("--split", type=Path, required=True, help="Frozen split file")
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--calibrated", type=Path, default=DEFAULT_CALIBRATED)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--frozen-schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    errors = (
        AvailabilityError,
        ContractError,
        FeatureError,
        HeldOutError,
        HeldOutModelError,
        OutputError,
        SplitError,
        # An environment that cannot load LightGBM cannot read the frozen classifier.
        TrainingError,
        tomllib.TOMLDecodeError,
    )
    try:
        path = run(
            args.truth,
            args.sweep_root,
            args.split,
            args.output,
            candidates_dir=args.candidates,
            calibrated_dir=args.calibrated,
            contract_path=args.contract,
            frozen_schema=args.frozen_schema,
            loader=load_split,
            versions=FROZEN_VERSIONS,
        )
    except errors as error:
        print(f"Feature availability error: {error}", file=sys.stderr)
        return EXIT_ERROR
    print((path.parent / TABLE_FILE).read_text(), end="")
    print(f"Report: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
