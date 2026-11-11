"""Synthetic M03 candidate outputs and split rows for calibration tests.

The scores and rows are made up so calibration has a known direction: the default scorer is
right 70% of the time but always 97% sure. They are not model output or recorded evidence.
"""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

from agentwatch_data.files import json_text
from agentwatch_model.comparison import MODEL_NOT_EVALUATED
from agentwatch_model.contract import DEFAULT_CONTRACT, load_contract
from agentwatch_model.report import identity
from agentwatch_model.rules import DEFAULT_RULES, load_rules
from training_fixtures import featured_row

COLUMNS = ("none", "cpu_sandbox", "inference", "retrieval")
MODEL_HASH = "0" * 64  # a stand-in model file hash; no model file exists in these tests
Scorer = Callable[[dict, int], dict[str, float]]


def steps(per_class: int, prefix: str, classes=COLUMNS) -> list[dict]:
    return [
        featured_row(label, index, task_id=f"{prefix}-{index % 5}")
        for label in classes
        for index in range(per_class)
    ]


def overconfident(row: dict, index: int) -> dict[str, float]:
    label = row["label_resource"]
    wrong = COLUMNS[(COLUMNS.index(label) + 1) % len(COLUMNS)]
    predicted = label if index % 10 < 7 else wrong
    return {name: 0.97 if name == predicted else 0.01 for name in COLUMNS}


def unsure(row: dict, index: int) -> dict[str, float]:
    """Right, but barely more confident than chance on every step."""
    return {name: 0.28 if name == row["label_resource"] else 0.24 for name in COLUMNS}


def record_lines(rows: list[dict], split: str, scorer: Scorer) -> str:
    lines = []
    for index, row in enumerate(rows):
        probabilities = scorer(row, index)
        lines.append(
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
                    "label_excess_fraction": row["label_excess_fraction"],
                    "prediction": max(probabilities, key=lambda name: probabilities[name]),
                    "probabilities": probabilities,
                    "excess_fraction": 0.1,
                    "rule_prediction": "none",
                    "evaluation": True,
                },
                sort_keys=True,
            )
        )
    return "".join(f"{line}\n" for line in lines)


def write_candidates(
    directory: Path,
    split_file: Path,
    validation: list[dict],
    test: list[dict],
    *,
    validation_scorer: Scorer = overconfident,
    test_scorer: Scorer = overconfident,
    status: str = "evaluated",
    early_stopping: str = "validation",
) -> None:
    """The files an M03 attempt leaves in its output directory, as calibration reads them."""
    directory.mkdir(parents=True, exist_ok=True)
    contract, rules = load_contract(DEFAULT_CONTRACT), load_rules(DEFAULT_RULES)
    model_files = {"classifier": {"file": "classifier.txt", "sha256": MODEL_HASH}}
    files = {
        "predictions": ("predictions.jsonl", record_lines(test, "test", test_scorer)),
        "validation_predictions": (
            "validation_predictions.jsonl",
            record_lines(validation, "validation", validation_scorer),
        ),
        "preprocessing": (
            "preprocessing.json",
            json_text(
                {
                    "probability_columns": list(COLUMNS),
                    "training_version": "training-v1+synthetic",
                    "model_files": model_files,
                }
            ),
        ),
    }
    for name, text in files.values():
        (directory / name).write_text(text)
    gates = {"regression_error": {"outcome": "pass", "value": 0.05, "threshold": 0.1}} | {
        name: {"outcome": "not_evaluated", "reason": reason}
        for name, reason in MODEL_NOT_EVALUATED.items()
    }
    report = identity(contract, rules) | {
        "status": status,
        "evaluation": status == "evaluated",
        "reason": "" if status == "evaluated" else "Split x.json: its leakage check is blocked",
        "split": {"file": split_file.name, "sha256": file_hash(split_file)},
        "scored_split": "test",
        "dataset_version": "steps-v1.2+synthetic",
        **{
            key: {"file": name, "sha256": hashlib.sha256(text.encode()).hexdigest()}
            for key, (name, text) in files.items()
        },
        "model_files": {"classifier": MODEL_HASH},
        "model": {
            "version": "training-v1+synthetic",
            "class_order": list(COLUMNS),
            "classes_absent_from_train": [],
            "early_stopping": early_stopping,
        },
        "gates": gates if status == "evaluated" else {},
    }
    (directory / "report.json").write_text(json_text(report))
    (directory / "tracking.json").write_text(json_text({"logged": True, "run_id": "m03-run"}))


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_loader(splits: dict[str, list[dict]], calls: list | None = None):
    def load(path, name, contract_path, frozen_schema, allow_unchecked_leakage=False):
        if calls is not None:
            calls.append((name, allow_unchecked_leakage))
        return splits[name]

    return load
