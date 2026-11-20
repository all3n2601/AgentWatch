"""A synthetic M03/M04 model pair for the served-model tests.

Everything here is made up: a tiny LightGBM classifier and regressor trained on generated rows
over MODEL_INPUTS, with preprocessing.json and calibration.json written in the field layout
`make train-candidates` and `make calibrate` produce. It is not recorded evidence and is never a
substitute for a real pair. It imports neither agentwatch_model nor agentwatch_data, so the
attribution package stays testable on its own; the layout is copied from
pipelines/model/src/agentwatch_model/train_candidates.py and calibrate.py.
"""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import lightgbm
import numpy as np
from agentwatch_attribution.classes import RESOURCE_CLASSES
from agentwatch_attribution.inputs import MODEL_INPUTS, feature_matrix

PREPROCESSING = "preprocessing.json"
CALIBRATION = "calibration.json"
MODEL_FILES = {"classifier": "classifier.txt", "regressor": "regressor.txt"}
OBJECTIVES = {"classifier": "multiclass", "regressor": "regression_l1"}
EXCESS_FRACTION_CLIP = [0.0, 1.0]
# Never real versions, so a synthetic pair cannot be mistaken for a trained one.
TRAINING_VERSION = "training-v1+synthetic"
CALIBRATION_VERSION = "calibration-v1+synthetic"
TEMPERATURE = 0.9
SEED = 7
TRAINING_ROWS = 240
NUM_BOOST_ROUND = 15
PRESSURE_CPU_THRESHOLD = 50.0
QUEUE_INFERENCE_THRESHOLD = 0.5
FIXED = {"deterministic": True, "num_threads": 1, "seed": 0, "verbose": -1, "force_col_wise": True}
SETTINGS = {"learning_rate": 0.3, "num_leaves": 7, "min_data_in_leaf": 5}


def synthetic_rows(count: int, seed: int = SEED) -> list[dict[str, object]]:
    """Generated step rows over MODEL_INPUTS; a quarter have no CPU-pressure reading."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for _ in range(count):
        queue = float(rng.uniform(0.0, 5.0))
        execution = float(rng.uniform(0.1, 5.0))
        draw = rng.uniform()
        # Missing, measured zero and measured positive pressure all occur, so the classifier
        # learns that a missing reading is not a zero one.
        pressure = None if draw < 0.25 else 0.0 if draw < 0.5 else float(rng.uniform(0, 100))
        throttled = None if rng.uniform() < 0.3 else int(rng.integers(0, 2_000_000))
        rows.append(
            {
                "queue_seconds": queue,
                "execution_seconds": execution,
                "duration_seconds": queue + execution,
                "cpu_seconds": float(rng.uniform(0.0, execution)),
                "queue_fraction": queue / (queue + execution),
                "workload_iterations": int(rng.integers(1, 10)),
                "tests_per_step": int(rng.integers(1, 5)),
                "worker_capacity": int(rng.integers(1, 4)),
                "cpu_pressure_max_avg10": pressure,
                "cpu_pressure_available": pressure is not None,
                "cgroup_throttled_usec_delta": throttled,
                "cgroup_available": throttled is not None,
                "telemetry_sample_count": int(rng.integers(0, 50)),
                "telemetry_coverage": float(rng.uniform(0.0, 1.0)),
            }
        )
    return rows


def synthetic_label(row: dict[str, object]) -> str:
    pressure = row["cpu_pressure_max_avg10"]
    queue_fraction = row["queue_fraction"]
    assert isinstance(queue_fraction, float)
    if pressure is None:
        return "retrieval"
    assert isinstance(pressure, float)
    if pressure >= PRESSURE_CPU_THRESHOLD:
        return "cpu_sandbox"
    return "inference" if queue_fraction >= QUEUE_INFERENCE_THRESHOLD else "none"


def train(features: np.ndarray, targets: np.ndarray, params: dict[str, object]) -> lightgbm.Booster:
    data = lightgbm.Dataset(features, label=targets, params={"verbose": -1})
    return lightgbm.train(FIXED | SETTINGS | params, data, num_boost_round=NUM_BOOST_ROUND)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict[str, object]) -> None:
    # The same canonical text agentwatch_data.files.json_text writes for the real artifacts.
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def preprocessing_document(model_files: dict[str, dict[str, str]]) -> dict[str, object]:
    """train_candidates.serving_contract's layout, with synthetic lineage."""
    columns = list(RESOURCE_CLASSES)
    return {
        "features": list(MODEL_INPUTS),
        "missing_values": "NaN passed to LightGBM, which routes it natively; never imputed",
        "booleans": "true is 1.0, false is 0.0",
        "class_order": columns,
        "training_version": TRAINING_VERSION,
        "dataset_version": "dataset-synthetic",
        "label_version": "labels-synthetic",
        "label_versions": ["labels-synthetic"],
        "feature_versions": [],
        "status": "evaluated",
        "evaluation": True,
        "model_files": model_files,
        "objectives": OBJECTIVES,
        "best_iterations": {"classifier": NUM_BOOST_ROUND, "regressor": NUM_BOOST_ROUND},
        "probability_columns": columns,
        "excess_fraction_clip": EXCESS_FRACTION_CLIP,
        "lightgbm_version": version("lightgbm"),
    }


def calibration_document(
    model_files: dict[str, dict[str, str]], abstention_threshold: float, blocks_promotion: bool
) -> dict[str, object]:
    """calibrate.calibration_document's layout, with a synthetic contract and gates."""
    outcome = "fail" if blocks_promotion else "pass"
    return {
        "version": CALIBRATION_VERSION,
        "method": "temperature_scaling",
        "status": "evaluated",
        "contract": {
            "version": "eval-contract-synthetic",
            "hash": "00000000",
            "status": "accepted",
        },
        "gates": {"calibration": outcome, "abstention": "pass"},
        "blocks_promotion": blocks_promotion,
        "fitted": True,
        "reason": None,
        "temperature": TEMPERATURE,
        "abstention_threshold": abstention_threshold,
        "abstain_when": "calibrated top-label confidence is below abstention_threshold",
        "fit_split": "validation",
        "probability_columns": list(RESOURCE_CLASSES),
        "training_version": TRAINING_VERSION,
        "model_files": model_files,
        "candidates": {"report_sha256": "0" * 64, "predictions_sha256": None, "run_id": None},
    }


def write_synthetic_pair(
    root: Path,
    *,
    abstention_threshold: float = 0.6,
    blocks_promotion: bool = True,
    regressor_offset: float = 0.0,
) -> tuple[Path, Path]:
    """Write a matching candidates folder and calibration folder under root; return both.

    regressor_offset is added to every regressor target, so a positive offset makes the
    regressor predict above the [0, 1] excess-fraction clip.
    """
    candidates_dir, calibration_dir = root / "candidates", root / "calibration"
    candidates_dir.mkdir(parents=True)
    calibration_dir.mkdir(parents=True)
    rows = synthetic_rows(TRAINING_ROWS)
    features = feature_matrix(rows)
    index = {name: position for position, name in enumerate(RESOURCE_CLASSES)}
    classes = np.array([index[synthetic_label(row)] for row in rows], dtype=np.float64)
    queue_fraction = features[:, MODEL_INPUTS.index("queue_fraction")]
    boosters = {
        "classifier": train(
            features,
            classes,
            {"objective": OBJECTIVES["classifier"], "num_class": len(RESOURCE_CLASSES)},
        ),
        "regressor": train(
            features, queue_fraction + regressor_offset, {"objective": OBJECTIVES["regressor"]}
        ),
    }
    model_files = {}
    for key, booster in boosters.items():
        path = candidates_dir / MODEL_FILES[key]
        booster.save_model(str(path), num_iteration=NUM_BOOST_ROUND)
        model_files[key] = {"file": MODEL_FILES[key], "sha256": sha256_of(path)}
    write_json(candidates_dir / PREPROCESSING, preprocessing_document(model_files))
    write_json(
        calibration_dir / CALIBRATION,
        calibration_document(model_files, abstention_threshold, blocks_promotion),
    )
    return candidates_dir, calibration_dir
