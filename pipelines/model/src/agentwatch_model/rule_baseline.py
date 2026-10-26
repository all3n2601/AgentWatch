"""Evaluate rules-v1 on a frozen split and save the report and every scored prediction.

Splits are read only through agentwatch_data.splits.load_split, which refuses a split whose
quality verdict or leakage check does not pass. A refused split produces a blocked report
with no metrics. --explore skips only the leakage check, scores the validation split rather
than the test split so test results cannot steer later changes, and marks the report as not
an evaluation.
"""

import argparse
import hashlib
import json
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path

from agentwatch_data.files import file_sha256, json_text, write_atomic
from agentwatch_data.splits import SplitError, load_split

from agentwatch_model.contract import DEFAULT_CONTRACT, ContractError, load_contract
from agentwatch_model.metrics import CLASSES, is_number
from agentwatch_model.report import blocked_report, build_report
from agentwatch_model.rules import DEFAULT_RULES, Rules, RulesError, load_rules

# pipelines/validation/schema/steps.pbtxt beside this package's source tree. This default holds
# for the workspace (editable) install; the Makefile always passes the path explicitly.
DEFAULT_SCHEMA = Path(__file__).resolve().parents[3] / "validation" / "schema" / "steps.pbtxt"
DEFAULT_OUTPUT = Path(".data/rule-baseline")
REPORT = "report.json"
PREDICTIONS = "predictions.jsonl"
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 0  # recorded in the report so the interval can be reproduced
REQUIRED_FIELDS = (
    "run_id",
    "step_id",
    "task_id",
    "condition",
    "label_resource",
    "label_excess_fraction",
    "dataset_version",
    "label_version",
)
EXIT_BLOCKED = 1
EXIT_ERROR = 2

Loader = Callable[..., list[dict]]


class RowError(ValueError):
    """A split row the evaluation cannot use as recorded."""


def validate_rows(rows: list[dict], split: str) -> None:
    seen = set()
    for index, row in enumerate(rows):
        missing = [field for field in REQUIRED_FIELDS if field not in row]
        if missing:
            raise RowError(f"{split} row {index} is missing {', '.join(missing)}")
        key = (row["run_id"], row["step_id"])
        if key in seen:
            raise RowError(f"{split} has more than one row for run {key[0]} step {key[1]}")
        seen.add(key)
        if row["label_resource"] not in CLASSES:
            raise RowError(f"{split} row {index} has unknown label {row['label_resource']!r}")
        fraction = row["label_excess_fraction"]
        if not is_number(fraction) or not 0 <= fraction <= 1:
            raise RowError(f"{split} row {index} label_excess_fraction is not a number from 0 to 1")


def validate_dataset_version(rows: list[dict]) -> None:
    versions = sorted({row["dataset_version"] for row in rows})
    if len(versions) > 1:
        raise RowError(f"Rows come from more than one dataset version: {versions}")


def prediction_records(
    rules: Rules, rows: list[dict], predictions: list[str], split: str, evaluation: bool
) -> str:
    lines = [
        json.dumps(
            {
                "run_id": row["run_id"],
                "step_id": row["step_id"],
                "task_id": row["task_id"],
                "condition": row["condition"],
                "split": split,
                "dataset_version": row["dataset_version"],
                "label_version": row["label_version"],
                "label_resource": row["label_resource"],
                "prediction": prediction,
                "rules_version": rules.version,
                "evaluation": evaluation,
            },
            sort_keys=True,
            allow_nan=False,
        )
        for row, prediction in zip(rows, predictions, strict=True)
    ]
    return "".join(f"{line}\n" for line in lines)


def split_identity(path: Path) -> dict:
    return {"file": path.name, "sha256": file_sha256(path) if path.is_file() else None}


def run(
    split: Path,
    output: Path,
    *,
    contract_path: Path = DEFAULT_CONTRACT,
    frozen_schema: Path = DEFAULT_SCHEMA,
    rules_path: Path = DEFAULT_RULES,
    explore: bool = False,
    loader: Loader | None = None,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> str:
    """Write the report (and predictions unless blocked); return the report status."""
    contract = load_contract(contract_path)
    rules = load_rules(rules_path)
    scored_split = "validation" if explore else "test"
    # Resolved at call time, so tests can replace the module's load_split.
    load = loader or load_split
    try:
        train, scored = (
            load(split, name, contract_path, frozen_schema, allow_unchecked_leakage=explore)
            for name in ("train", scored_split)
        )
    except SplitError as error:
        report = blocked_report(contract, rules, str(error)) | {"split": split_identity(split)}
        # A stale predictions file from an earlier run must not sit beside a blocked report.
        (output / PREDICTIONS).unlink(missing_ok=True)
        write_atomic(output / REPORT, json_text(report))
        return report["status"]
    validate_rows(train, "train")
    validate_rows(scored, scored_split)
    validate_dataset_version(train + scored)
    predictions = [rules.predict(row) for row in scored]
    report = build_report(
        contract,
        rules,
        train,
        scored,
        predictions,
        evaluation=not explore,
        resamples=resamples,
        seed=seed,
    )
    records = prediction_records(rules, scored, predictions, scored_split, not explore)
    report |= {
        "split": split_identity(split),
        "scored_split": scored_split,
        "scored_steps": len(scored),
        "predictions": {
            "file": PREDICTIONS,
            "sha256": hashlib.sha256(records.encode()).hexdigest(),
        },
    }
    # Predictions first: the report names their hash, so a crash between the two writes leaves
    # a report that does not match, which the hash exposes.
    write_atomic(output / PREDICTIONS, records)
    write_atomic(output / REPORT, json_text(report))
    return report["status"]


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate rules-v1 on a frozen split against the evaluation contract."
    )
    parser.add_argument("--split", type=Path, required=True, help="Frozen split file")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--frozen-schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument(
        "--explore",
        action="store_true",
        help="Skip only the leakage check and score the validation split; not an evaluation",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        status = run(
            args.split,
            args.output,
            contract_path=args.contract,
            frozen_schema=args.frozen_schema,
            rules_path=args.rules,
            explore=args.explore,
        )
    except (ContractError, RulesError, RowError, tomllib.TOMLDecodeError, OSError) as error:
        print(f"Rule baseline error: {error}", file=sys.stderr)
        return EXIT_ERROR
    print(f"Rule baseline {status}: {args.output / REPORT}")
    return EXIT_BLOCKED if status == "blocked" else 0


if __name__ == "__main__":
    sys.exit(main())
