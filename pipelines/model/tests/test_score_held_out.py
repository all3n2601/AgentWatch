"""score-held-out end to end on a synthetic sweep, a synthetic frozen M03/M04 pair and a fake
split loader.

Nothing here is recorded evidence: the sweep is the data tests' synthetic journal copied into
a sweep layout, the frozen model is a small LightGBM candidate trained on made-up rows, and the
dataset v1 split is replaced by synthetic rows.
"""

import hashlib
import json

import pytest
from agentwatch_data.splits import SplitError
from training_fixtures import dataset, featured_row, require_lightgbm

require_lightgbm()

import mlflow
from agentwatch_data import evidence
from agentwatch_model.candidates import TrainingError
from agentwatch_model.held_out_model import CLASSIFIER_SCORER
from agentwatch_model.score_held_out import (
    EXPERIMENT,
    OutputError,
    input_hashes,
    main,
    run,
    run_attempt,
    scored_overlap,
)
from frozen_model_fixtures import write_frozen_pair
from test_held_out_rows import sweep, write_summary, write_truths  # noqa: F401


def split_loader(validation, extra_train_rows=()):
    """A load_split stand-in: the synthetic split, plus rows a test wants to plant in it."""
    splits = {
        "train": dataset(30, "train") + list(extra_train_rows),
        "validation": validation,
        "test": dataset(4, "test"),
    }

    def loader(path, name, contract_path, frozen_schema, allow_unchecked_leakage=False):
        assert allow_unchecked_leakage is False
        return splits[name]

    return loader


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
    """Truth records with the position fields the report slices by, and a frozen model pair."""
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
    pair = write_frozen_pair(tmp_path / "frozen", split)
    return sweep | {
        "split": split,
        "output": tmp_path / "scores",
        "pair": pair,
        "loader": split_loader(pair.validation_rows),
    }


def attempt_directory(output):
    (directory,) = [path for path in output.iterdir() if path.is_dir()]
    return directory


def run_synthetic(prepared, tracking, **overrides):
    return run_attempt(
        [prepared["truth"]],
        prepared["root"],
        prepared["split"],
        prepared["output"],
        candidates_dir=prepared["pair"].candidates_dir,
        calibrated_dir=prepared["pair"].calibrated_dir,
        loader=overrides.pop("loader", prepared["loader"]),
        resamples=50,
        **tracking,
        **overrides,
    )


def cli_arguments(prepared, tracking):
    return [
        "--truth",
        str(prepared["truth"]),
        "--sweep-root",
        str(prepared["root"]),
        "--split",
        str(prepared["split"]),
        "--candidates",
        str(prepared["pair"].candidates_dir),
        "--calibrated",
        str(prepared["pair"].calibrated_dir),
        "--output",
        str(prepared["output"]),
        "--tracking-uri",
        tracking["tracking_uri"],
    ]


def test_run_scores_both_scorers_and_writes_no_models(prepared, tracking):
    status = run(
        [prepared["truth"]],
        prepared["root"],
        prepared["split"],
        prepared["output"],
        candidates_dir=prepared["pair"].candidates_dir,
        calibrated_dir=prepared["pair"].calibrated_dir,
        loader=prepared["loader"],
        resamples=50,
        **tracking,
    )
    assert status == "completed"
    directory = attempt_directory(prepared["output"])
    stamp, digest = directory.name.rsplit("-", 1)
    assert stamp.endswith("Z") and len(digest) == 6
    assert not (directory / "models").exists()
    report = json.loads((directory / "report.json").read_text())
    assert report["status"] == "completed" and report["held_out"] is True
    assert report["gate"] == "none (descriptive baseline)"
    assert report["scored_steps"] == 3
    frozen = report["model"]["frozen"]
    classifier_file = prepared["pair"].candidates_dir / "classifier.txt"
    assert frozen["classifier_sha256"] == hashlib.sha256(classifier_file.read_bytes()).hexdigest()
    assert frozen["temperature"] == 0.9 and frozen["threshold"] == 0.6
    assert report["split"]["sha256"] == hashlib.sha256(b"{}").hexdigest()
    (truth,) = report["truths"].values()
    assert set(truth) == {"label_version", "steps_sha256", "steps_summary_sha256", "truth_sha256"}
    inputs = report["inputs"]
    assert inputs["split"] == report["split"]["sha256"]
    assert all(inputs["truth"].values())
    assert len(inputs["runs"]) == 2 and all(inputs["runs"].values())
    assert set(inputs["model"]) == {
        "classifier.txt",
        "preprocessing.json",
        "report.json",
        "validation_predictions.jsonl",
        "calibration.json",
    }
    assert all(inputs["model"].values())
    lines = [json.loads(x) for x in (directory / "predictions.jsonl").read_text().splitlines()]
    assert len(lines) == 6  # three steps, two scorers
    assert {line["scorer"] for line in lines} == {"rules-v1", CLASSIFIER_SCORER}
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


@pytest.mark.parametrize("option", ["--training", "--calibration"])
def test_the_training_and_calibration_options_are_gone(prepared, tracking, option):
    with pytest.raises(SystemExit) as raised:
        main([*cli_arguments(prepared, tracking), option, "x"])
    assert raised.value.code == 2
    assert not prepared["output"].exists()


def test_a_scored_run_in_the_split_fails_the_attempt(prepared, tracking):
    planted = featured_row("none", 0, run_id=prepared["run_id"])
    loader = split_loader(prepared["pair"].validation_rows, [planted])
    outcome = run_synthetic(prepared, tracking, loader=loader)
    assert outcome.status == "failed" and outcome.run_id
    report = json.loads((outcome.directory / "report.json").read_text())
    assert report["status"] == "failed" and prepared["run_id"] in report["reason"]
    assert report["model"]["frozen"]["classifier_sha256"]
    assert not (outcome.directory / "predictions.jsonl").exists()


def test_scored_overlap_names_run_ids_and_plan_hashes_found_in_the_split():
    scored = [
        {"run_id": "run-a", "plan_sha256": "plan-1"},
        {"run_id": "run-b", "plan_sha256": None},
        {"run_id": "run-c", "plan_sha256": "plan-9"},
    ]
    split_rows = [
        {"run_id": "run-b", "plan_sha256": None},
        {"run_id": "other", "plan_sha256": "plan-1"},
        {"run_id": "another", "plan_sha256": None},
    ]
    assert scored_overlap(scored, split_rows) == ["run_id run-b", "plan_sha256 plan-1"]
    assert scored_overlap(scored, [{"run_id": "x", "plan_sha256": None}]) == []


def test_a_mismatched_model_pair_exits_2_and_writes_no_attempt(prepared, tracking):
    calibration = prepared["pair"].calibrated_dir / "calibration.json"
    body = json.loads(calibration.read_text())
    body["model_files"]["classifier"]["sha256"] = "0" * 64
    calibration.write_text(json.dumps(body))
    assert main(cli_arguments(prepared, tracking)) == 2
    assert not prepared["output"].exists() or not any(prepared["output"].iterdir())


def test_a_missing_lightgbm_exits_2_and_writes_no_attempt(prepared, tracking, monkeypatch):
    def unavailable():
        raise TrainingError("LightGBM cannot be loaded")

    monkeypatch.setattr("agentwatch_model.held_out_model.check_lightgbm", unavailable)
    assert main(cli_arguments(prepared, tracking)) == 2
    assert not prepared["output"].exists() or not any(prepared["output"].iterdir())


def test_a_model_that_does_not_reproduce_validation_is_a_recorded_failure(prepared, tracking):
    candidates = prepared["pair"].candidates_dir
    predictions = candidates / "validation_predictions.jsonl"
    lines = predictions.read_text().splitlines()
    first = json.loads(lines[0])
    first["probabilities"] = {name: 0.25 for name in first["probabilities"]}
    predictions.write_text("".join(f"{line}\n" for line in [json.dumps(first), *lines[1:]]))
    report_path = candidates / "report.json"
    saved = json.loads(report_path.read_text())
    saved["validation_predictions"]["sha256"] = hashlib.sha256(predictions.read_bytes()).hexdigest()
    report_path.write_text(json.dumps(saved))
    outcome = run_synthetic(prepared, tracking)
    assert outcome.status == "failed" and outcome.run_id
    report = json.loads((outcome.directory / "report.json").read_text())
    assert "HeldOutModelError" in report["reason"]
    assert not (outcome.directory / "predictions.jsonl").exists()


def test_only_unjudged_steps_fail_the_attempt_with_counts(prepared, tracking):
    records = [
        record
        | {
            "truths": {"queue": None, "slower": None, "protocol_clean": None},
            "truth_reasons": {
                "queue": "root_step",
                "slower": "root_step",
                "protocol_clean": "root_step",
            },
        }
        for record in prepared["records"]
    ]
    write_truths(prepared["truth"], records)
    outcome = run_synthetic(prepared, tracking)
    assert outcome.status == "failed"
    report = json.loads((outcome.directory / "report.json").read_text())
    assert "no step could be scored" in report["reason"]
    assert report["unscored"]["steps"]["queue"] == {"root_step": 3}
    assert not (outcome.directory / "predictions.jsonl").exists()


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


def test_a_malformed_truth_record_fails_the_attempt_with_a_tracked_report(prepared, tracking):
    lines = (prepared["truth"] / "truth.jsonl").read_text().splitlines()
    malformed = [json.loads(line) for line in lines]
    del malformed[0]["run_id"]
    write_truths(prepared["truth"], malformed)
    outcome = run_synthetic(prepared, tracking)
    assert outcome.status == "failed" and outcome.run_id
    report = json.loads((outcome.directory / "report.json").read_text())
    assert report["status"] == "failed" and report["held_out"] is True
    assert "HeldOutError" in report["reason"] and "run_id must be a string" in report["reason"]
    assert all(report["inputs"]["truth"].values()) and report["inputs"]["runs"] == {}
    assert not (outcome.directory / "predictions.jsonl").exists()
    assert json.loads((outcome.directory / "tracking.json").read_text())["logged"] is True


def test_truth_hashes_of_folders_with_one_name_do_not_collide(prepared, tmp_path):
    other = tmp_path / "elsewhere" / prepared["truth"].name
    other.mkdir(parents=True)
    (other / "truth.jsonl").write_text((prepared["truth"] / "truth.jsonl").read_text())
    write_summary(other, sweep_id="another", sweep_folder="cfg-default-elsewhere")
    hashes = input_hashes([prepared["truth"], other], prepared["root"], prepared["split"])
    assert len(hashes["truth"]) == 4
    assert (
        hashes["truth"]["cfg-default-synthetic/truth.json"]
        != hashes["truth"]["cfg-default-elsewhere/truth.json"]
    )


def test_truth_hashes_keep_a_folder_passed_twice(prepared):
    hashes = input_hashes([prepared["truth"]] * 2, prepared["root"], prepared["split"])
    assert len(hashes["truth"]) == 4


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
