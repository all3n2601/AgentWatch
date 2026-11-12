"""score-held-out end to end on a synthetic sweep and a fake split loader.

Nothing here is recorded evidence: the sweep is the data tests' synthetic journal copied into
a sweep layout, and the dataset v1 split is replaced by synthetic rows.
"""

import hashlib
import json

import pytest
from agentwatch_data.splits import SplitError
from training_fixtures import dataset, require_lightgbm

require_lightgbm()

import mlflow
from agentwatch_data import evidence
from agentwatch_model.score_held_out import EXPERIMENT, OutputError, main, run, run_attempt
from test_held_out_rows import sweep, write_truths  # noqa: F401

SPLITS = {"train": dataset(30, "train"), "validation": dataset(30, "valid")}


def loader(path, name, contract_path, frozen_schema, allow_unchecked_leakage=False):
    assert allow_unchecked_leakage is False
    assert name in SPLITS, f"{name} must never be loaded"
    return SPLITS[name]


def refusing(path, name, contract_path, frozen_schema, allow_unchecked_leakage=False):
    raise SplitError("Split x.json: its leakage check is blocked")


@pytest.fixture
def tracking(tmp_path):
    previous = mlflow.get_tracking_uri()
    yield {
        "tracking_uri": f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
        "artifact_root": tmp_path / "artifacts",
    }
    mlflow.set_tracking_uri(previous)


@pytest.fixture
def prepared(sweep, tmp_path):  # noqa: F811
    """Truth records with the position fields the report slices by."""
    records = [
        record
        | {
            "configuration": "cfg-default",
            "n": 3,
            "phase": "live",
            "batch_id": "batch-1",
            "truths": {"queue": "inference", "slower": "none", "protocol_clean": None},
            "truth_reasons": {"queue": None, "slower": None, "protocol_clean": "not_applicable"},
        }
        for record in sweep["records"]
    ]
    write_truths(sweep["truth"], records)
    split = tmp_path / "split.json"
    split.write_text("{}")
    return sweep | {"split": split, "output": tmp_path / "scores"}


def attempt_directory(output):
    (directory,) = [path for path in output.iterdir() if path.is_dir()]
    return directory


def run_synthetic(prepared, tracking, **overrides):
    return run_attempt(
        [prepared["truth"]],
        prepared["root"],
        prepared["split"],
        prepared["output"],
        loader=overrides.pop("loader", loader),
        resamples=50,
        **tracking,
        **overrides,
    )


def test_run_writes_report_predictions_and_models_and_logs_the_named_experiment(prepared, tracking):
    status = run(
        [prepared["truth"]],
        prepared["root"],
        prepared["split"],
        prepared["output"],
        loader=loader,
        resamples=50,
        **tracking,
    )
    assert status == "completed"
    directory = attempt_directory(prepared["output"])
    stamp, digest = directory.name.rsplit("-", 1)
    assert stamp.endswith("Z") and len(digest) == 6
    report = json.loads((directory / "report.json").read_text())
    assert report["status"] == "completed" and report["held_out"] is True
    assert report["gate"] == "none (descriptive baseline)"
    assert report["scored_steps"] == 3
    model = report["model"]
    assert model["version"].startswith("training-v1+") and model["lightgbm_version"]
    assert set(model["model_files"]) == {"classifier", "regressor"}
    for key, file_name in (("classifier", "classifier.txt"), ("regressor", "regressor.txt")):
        saved = directory / "models" / file_name
        assert model["model_files"][key] == hashlib.sha256(saved.read_bytes()).hexdigest()
    assert model["split"]["sha256"] == hashlib.sha256(b"{}").hexdigest()
    inputs = report["inputs"]
    assert inputs["split"] == model["split"]["sha256"]
    assert all(inputs["truth"].values())
    assert len(inputs["runs"]) == 2 and all(inputs["runs"].values())
    lines = [json.loads(x) for x in (directory / "predictions.jsonl").read_text().splitlines()]
    assert len(lines) == 6  # three steps, two scorers
    assert {line["scorer"] for line in lines} == {"rules-v1", "lightgbm-classifier-training-v1"}
    assert set(lines[0]) == {
        "run_id",
        "step_id",
        "scorer",
        "label",
        "confidence",
        "abstained",
        "truths",
    }
    assert lines[0]["truths"]["queue"] == "inference"
    mlflow.set_tracking_uri(tracking["tracking_uri"])
    experiment = mlflow.get_experiment_by_name(EXPERIMENT)
    assert experiment is not None
    (logged,) = mlflow.search_runs([experiment.experiment_id], output_format="list")
    assert logged.data.tags["status"] == "completed"
    assert mlflow.get_experiment_by_name("agentwatch-attribution") is None
    assert json.loads((directory / "tracking.json").read_text())["logged"] is True


def test_release_identity_survives_an_empty_validation_split(prepared, tracking):
    splits = {"train": dataset(30, "train"), "validation": []}

    def empty_validation(path, name, contract_path, frozen_schema, allow_unchecked_leakage=False):
        return splits[name]

    outcome = run_synthetic(prepared, tracking, loader=empty_validation)
    assert outcome.status == "completed"
    report = json.loads((outcome.directory / "report.json").read_text())
    assert report["dataset_version"] == splits["train"][0]["dataset_version"]
    assert report["model"]["dataset_version"] == report["dataset_version"]
    assert report["model"]["label_version"] == splits["train"][0]["label_version"]
    assert report["model"]["calibration"]["abstention"] == "unavailable"
    assert report["model"]["calibration"]["reason"]
    classifier = [
        e
        for e in report["rollups"]["by_scorer_truth"]
        if e["scorer"].startswith("lightgbm") and e["truth"] == "queue"
    ]
    assert classifier and classifier[0]["abstention_rate"] is None


def test_a_failed_attempt_is_reported_and_tracked(prepared, tracking):
    outcome = run_synthetic(prepared, tracking, loader=refusing)
    assert outcome.status == "failed" and outcome.run_id
    report = json.loads((outcome.directory / "report.json").read_text())
    assert report["status"] == "failed" and report["held_out"] is True
    assert "SplitError" in report["reason"] and "leakage" in report["reason"]
    assert report["inputs"]["split"]
    assert not (outcome.directory / "predictions.jsonl").exists()
    mlflow.set_tracking_uri(tracking["tracking_uri"])
    experiment = mlflow.get_experiment_by_name(EXPERIMENT)
    (logged,) = mlflow.search_runs([experiment.experiment_id], output_format="list")
    assert logged.data.tags["status"] == "failed"


def test_a_truth_file_without_held_out_fails_the_attempt(prepared, tracking):
    summary = json.loads((prepared["truth"] / "truth.json").read_text())
    (prepared["truth"] / "truth.json").write_text(json.dumps(summary | {"held_out": False}))
    outcome = run_synthetic(prepared, tracking)
    assert outcome.status == "failed"
    report = json.loads((outcome.directory / "report.json").read_text())
    assert "held_out" in report["reason"]


@pytest.mark.parametrize(
    "parts", [("data", "held-out", "a"), ("Data", "HELD-OUT"), ("x", "DATA", "Held-Out", "y")]
)
def test_outputs_inside_data_held_out_are_refused(prepared, tracking, tmp_path, parts):
    target = tmp_path.joinpath(*parts)
    with pytest.raises(OutputError):
        run_synthetic(prepared | {"output": target}, tracking)
    assert not target.exists()
    arguments = [
        "--truth",
        str(prepared["truth"]),
        "--sweep-root",
        str(prepared["root"]),
        "--split",
        str(prepared["split"]),
        "--output",
        str(target),
    ]
    assert main(arguments) == 2


def test_scores_output_is_not_accepted_as_evidence(prepared, tracking, tmp_path, capsys):
    outcome = run_synthetic(prepared, tracking)
    store = tmp_path / "store"
    for folder in (outcome.directory, prepared["truth"]):
        assert evidence.main([str(folder), "--store", str(store)]) == 1
    assert "refused" in capsys.readouterr().err
    assert not store.exists() or not any(store.iterdir())
