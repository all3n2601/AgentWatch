"""Synthetic frozen M03/M04 folders for tests of the held-out scorer.

Everything here is made up: a small LightGBM candidate trained on training_fixtures rows, with
reports and a calibration written to match it. It is not recorded evidence and is never a
substitute for the files `make train-candidates` and `make calibrate` write.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from agentwatch_data.files import file_sha256, json_text
from agentwatch_model.candidates import (
    DEFAULT_TRAINING,
    Candidate,
    fit_candidate,
    load_training_config,
    predict,
)
from agentwatch_model.features import FEATURES
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
TEMPERATURE = 0.9
THRESHOLD = 0.6


@dataclass(frozen=True)
class FrozenPair:
    candidates_dir: Path
    calibrated_dir: Path
    candidate: Candidate
    validation_rows: list[dict]


def write_json(path: Path, value: dict) -> None:
    path.write_text(json_text(value))


def write_frozen_pair(directory: Path, split: Path) -> FrozenPair:
    """Write a matching candidates folder and calibration folder under directory.

    Returns the folders, the in-memory candidate and the validation rows it was scored on, so a
    test can compare the loaded model with the trained one and call check_reproduces_validation.
    """
    candidates_dir, calibrated_dir = directory / "candidates", directory / "calibration"
    calibrated_dir.mkdir(parents=True, exist_ok=True)
    train, validation = dataset(30, "train"), dataset(8, "valid")
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
            "version": "calibration-v1+synthetic",
            "contract": CONTRACT,
            "model_files": {
                "classifier": {"file": MODEL_FILES["classifier"], "sha256": hashes["classifier"]}
            },
        },
    )
    return FrozenPair(candidates_dir, calibrated_dir, candidate, validation)
