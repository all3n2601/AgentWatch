"""Calibration over a real M03 attempt: LightGBM output read exactly as train_candidates writes it."""

import json

from training_fixtures import dataset, require_lightgbm

require_lightgbm()

from agentwatch_model.calibrate import run as calibrate
from agentwatch_model.train_candidates import run as train

SPLITS = {
    "train": dataset(25, "train"),
    "validation": dataset(30, "valid"),
    "test": dataset(30, "test"),
}


def load(path, name, contract_path, frozen_schema, allow_unchecked_leakage=False):
    return SPLITS[name]


def test_a_trained_attempt_calibrates_end_to_end(tmp_path):
    split = tmp_path / "split.json"
    split.write_text("{}\n")
    tracking = {
        "tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
        "artifact_root": tmp_path / "artifacts",
    }
    candidates = tmp_path / "candidates"
    assert train(split, candidates, loader=load, **tracking).status == "evaluated"
    outcome = calibrate(split, candidates, tmp_path / "out", loader=load, **tracking)
    assert outcome.status == "evaluated" and outcome.run_id
    report = json.loads((tmp_path / "out" / "report.json").read_text())
    assert report["calibration"]["fitted"] is True
    assert (
        report["candidates"]["run_id"]
        == json.loads((candidates / "tracking.json").read_text())["run_id"]
    )
    lines = (tmp_path / "out" / "predictions.jsonl").read_text().splitlines()
    assert len(lines) == len(SPLITS["test"])
