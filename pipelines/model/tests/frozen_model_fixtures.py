"""Synthetic frozen M03/M04 folders for tests of the held-out scorer.

Everything here is made up: a small LightGBM candidate trained on training_fixtures rows, with
reports and a calibration written to match it. It is not recorded evidence and is never a
substitute for the files `make train-candidates` and `make calibrate` write.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from agentwatch_data.files import file_sha256, json_text
from agentwatch_model.calibration import scale
from agentwatch_model.candidates import (
    DEFAULT_TRAINING,
    Candidate,
    fit_candidate,
    load_training_config,
    predict,
)
from agentwatch_model.features import FEATURES
from agentwatch_model.held_out_model import FrozenVersions
from agentwatch_model.rule_baseline import split_identity
from agentwatch_model.train_candidates import (
    MODEL_FILES,
    PREPROCESSING,
    REPORT,
    VALIDATION_PREDICTIONS,
    save_models,
)
from training_fixtures import dataset

CONTRACT = {"version": "eval-contract-v3", "hash": "1cab0cbe", "status": "accepted"}
CALIBRATION = "calibration.json"
CALIBRATED_PREDICTIONS = "predictions.jsonl"
TEMPERATURE = 0.9
THRESHOLD = 0.6
SYNTHETIC_CALIBRATION_VERSION = "calibration-v1+synthetic"
# The versions a synthetic pair records. Its training version is the packaged training
# config's, as for a real M03 run; its calibration version is never a real one, so the pinned
# FROZEN_VERSIONS refuse every synthetic pair.
SYNTHETIC_VERSIONS = FrozenVersions(
    training=load_training_config(DEFAULT_TRAINING).version,
    calibration=SYNTHETIC_CALIBRATION_VERSION,
)


@dataclass(frozen=True)
class FrozenPair:
    candidates_dir: Path
    calibrated_dir: Path
    candidate: Candidate
    validation_rows: list[dict]
    test_rows: list[dict]


def write_json(path: Path, value: dict) -> None:
    path.write_text(json_text(value))


def jsonl_text(records: list[dict]) -> str:
    return "".join(f"{json.dumps(record, sort_keys=True)}\n" for record in records)


def calibrated_records(
    candidate: Candidate, rows: list[dict], temperature: float, threshold: float
) -> list[dict]:
    """M04-style test prediction records: the trained candidate's scores, temperature-scaled."""
    predictions = predict(candidate, rows)
    columns = list(candidate.class_order)
    raw = np.array([[p.probabilities[name] for name in columns] for p in predictions])
    records = []
    for row, prediction, raw_scores, scores in zip(
        rows, predictions, raw, scale(raw, temperature), strict=True
    ):
        confidence = float(scores.max())
        records.append(
            {
                "run_id": row["run_id"],
                "step_id": row["step_id"],
                "prediction": prediction.label,
                "abstained": confidence < threshold,
                "confidence": confidence,
                "raw_confidence": float(raw_scores.max()),
                "probabilities": dict(zip(columns, map(float, scores), strict=True)),
            }
        )
    return records


def write_frozen_pair(directory: Path, split: Path) -> FrozenPair:
    """Write a matching candidates folder and calibration folder under directory.

    Returns the folders, the in-memory candidate and the validation and test rows it was scored
    on, so a test can compare the loaded model with the trained one and call
    check_reproduces_validation and check_reproduces_calibration. Every call makes new step ids,
    so a test's split must use these rows, not another call of dataset().
    """
    candidates_dir, calibrated_dir = directory / "candidates", directory / "calibration"
    calibrated_dir.mkdir(parents=True, exist_ok=True)
    train, validation, test = dataset(30, "train"), dataset(8, "valid"), dataset(4, "test")
    candidate = fit_candidate(train, validation, load_training_config(DEFAULT_TRAINING))
    hashes = save_models(candidate, candidates_dir)
    columns = list(candidate.class_order)
    write_json(
        candidates_dir / PREPROCESSING,
        {
            "features": list(FEATURES),
            "class_order": columns,
            "probability_columns": columns,
            "training_version": candidate.training_version,
            "model_files": {
                key: {"file": MODEL_FILES[key], "sha256": value} for key, value in hashes.items()
            },
        },
    )
    lines = [
        json.dumps(
            {
                "run_id": row["run_id"],
                "step_id": row["step_id"],
                "prediction": prediction.label,
                "probabilities": prediction.probabilities,
            },
            sort_keys=True,
        )
        for row, prediction in zip(validation, predict(candidate, validation), strict=True)
    ]
    (candidates_dir / VALIDATION_PREDICTIONS).write_text("".join(f"{line}\n" for line in lines))
    write_json(
        candidates_dir / REPORT,
        {
            "status": "evaluated",
            "split": split_identity(split),
            "contract": CONTRACT,
            "model": {"version": candidate.training_version},
            "model_files": hashes,
            "preprocessing": {
                "file": PREPROCESSING,
                "sha256": file_sha256(candidates_dir / PREPROCESSING),
            },
            "validation_predictions": {
                "file": VALIDATION_PREDICTIONS,
                "sha256": file_sha256(candidates_dir / VALIDATION_PREDICTIONS),
            },
        },
    )
    write_json(
        calibrated_dir / CALIBRATION,
        {
            "fitted": True,
            "status": "evaluated",
            "temperature": TEMPERATURE,
            "abstention_threshold": THRESHOLD,
            "training_version": candidate.training_version,
            "version": SYNTHETIC_CALIBRATION_VERSION,
            "contract": CONTRACT,
            "model_files": {
                "classifier": {"file": MODEL_FILES["classifier"], "sha256": hashes["classifier"]}
            },
            "candidates": {"report_sha256": file_sha256(candidates_dir / REPORT)},
        },
    )
    (calibrated_dir / CALIBRATED_PREDICTIONS).write_text(
        jsonl_text(calibrated_records(candidate, test, TEMPERATURE, THRESHOLD))
    )
    write_json(
        calibrated_dir / REPORT,
        {
            "calibration": {
                "version": SYNTHETIC_CALIBRATION_VERSION,
                "temperature": TEMPERATURE,
                "abstention_threshold": THRESHOLD,
            },
            "split": split_identity(split),
            "fit": {"steps": len(validation)},
            "scored_steps": len(test),
            "joint_only_steps_not_scored": 0,
            "predictions": {
                "file": CALIBRATED_PREDICTIONS,
                "sha256": file_sha256(calibrated_dir / CALIBRATED_PREDICTIONS),
            },
        },
    )
    return FrozenPair(candidates_dir, calibrated_dir, candidate, validation, test)
