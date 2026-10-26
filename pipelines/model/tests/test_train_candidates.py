import json
import sys
from pathlib import Path

import pytest
from agentwatch_data.files import file_sha256
from agentwatch_data.splits import SplitError
from training_fixtures import dataset, featured_row, require_lightgbm

require_lightgbm()

import mlflow
from agentwatch_model.train_candidates import main, run

SPLITS = {
    "train": dataset(25, "train"),
    "validation": dataset(6, "valid"),
    "test": dataset(6, "test"),
}


def loader_for(calls, data):
    def load(path, name, contract_path, frozen_schema, allow_unchecked_leakage=False):
        calls.append((name, allow_unchecked_leakage))
        return data[name]

    return load


def blocked(path, name, contract_path, frozen_schema, allow_unchecked_leakage=False):
    raise SplitError("Split x.json: its leakage check is blocked; the contract blocks evaluation")


def tracking(tmp_path):
    return {
        "tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
        "artifact_root": tmp_path / "artifacts",
    }


def logged_runs(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    return mlflow.search_runs(search_all_experiments=True, output_format="list")


def read(path):
    return json.loads(path.read_text())


def test_run_trains_scores_test_and_saves_every_artifact(tmp_path):
    calls = []
    output = tmp_path / "out"
    outcome = run(
        tmp_path / "split.json", output, loader=loader_for(calls, SPLITS), **tracking(tmp_path)
    )
    assert outcome.status == "evaluated" and outcome.run_id
    assert calls == [("train", False), ("validation", False), ("test", False)]
    for name in (
        "classifier.txt",
        "regressor.txt",
        "preprocessing.json",
        "predictions.jsonl",
        "validation_predictions.jsonl",
        "report.json",
        "tracking.json",
    ):
        assert (output / name).is_file(), name
    report = read(output / "report.json")
    assert report["comparison"]["rows"] == len(SPLITS["test"])
    assert report["model"]["validation_rows_dropped"] == 0
    first = json.loads((output / "predictions.jsonl").read_text().splitlines()[0])
    assert set(first) >= {
        "run_id",
        "step_id",
        "prediction",
        "probabilities",
        "excess_fraction",
        "rule_prediction",
    }
    [logged] = logged_runs(tmp_path)
    assert logged.info.run_id == outcome.run_id
    assert logged.data.tags["status"] == "evaluated" and logged.data.tags["task"] == "M03"
    assert logged.data.params["dataset_version"] == "steps-v1.2+synthetic"
    assert logged.data.params["classifier.learning_rate"] == "0.05"
    assert "model_macro_f1_defined_classes" in logged.data.metrics


def test_preprocessing_contract_carries_what_serving_and_calibration_need(tmp_path):
    output = tmp_path / "out"
    run(tmp_path / "split.json", output, loader=loader_for([], SPLITS), **tracking(tmp_path))
    contract = read(output / "preprocessing.json")
    assert contract["status"] == "evaluated" and contract["evaluation"] is True
    assert contract["probability_columns"] == ["none", "cpu_sandbox", "inference", "retrieval"]
    assert contract["excess_fraction_clip"] == [0.0, 1.0]
    assert contract["model_files"]["classifier"]["sha256"] == file_sha256(output / "classifier.txt")
    assert contract["objectives"] == {"classifier": "multiclass", "regressor": "regression_l1"}
    validation = (output / "validation_predictions.jsonl").read_text().splitlines()
    assert (
        len(validation) == len(SPLITS["validation"])
        and json.loads(validation[0])["split"] == "validation"
    )


def test_a_blocked_split_is_still_logged_and_trains_nothing(tmp_path):
    output = tmp_path / "out"
    outcome = run(tmp_path / "split.json", output, loader=blocked, **tracking(tmp_path))
    assert outcome.status == "blocked"
    assert not (output / "classifier.txt").exists()
    assert read(output / "report.json")["status"] == "blocked"
    [logged] = logged_runs(tmp_path)
    assert logged.data.tags["status"] == "blocked" and "leakage" in logged.data.tags["reason"]


def test_a_failed_training_is_kept_as_a_failed_run(tmp_path):
    one_class = {
        "train": dataset(10, "train", classes=("none",)),
        "validation": [],
        "test": dataset(2, "test"),
    }
    output = tmp_path / "out"
    outcome = run(
        tmp_path / "split.json", output, loader=loader_for([], one_class), **tracking(tmp_path)
    )
    assert outcome.status == "failed"
    assert "at least two classes" in read(output / "report.json")["reason"]
    [logged] = logged_runs(tmp_path)
    assert logged.data.tags["status"] == "failed"


def test_bad_rows_are_kept_as_a_failed_run_not_a_crash(tmp_path):
    broken = dict(SPLITS) | {
        "train": SPLITS["train"] + [featured_row("none", 0) | {"label_excess_fraction": None}]
    }
    output = tmp_path / "out"
    outcome = run(
        tmp_path / "split.json", output, loader=loader_for([], broken), **tracking(tmp_path)
    )
    assert outcome.status == "failed"
    assert "RowError" in read(output / "report.json")["reason"]
    [logged] = logged_runs(tmp_path)
    assert logged.data.tags["status"] == "failed"


def test_rows_from_mixed_dataset_versions_are_a_failed_run(tmp_path):
    other = featured_row("none", 0) | {"dataset_version": "steps-v1.3+other"}
    mixed = dict(SPLITS) | {"test": SPLITS["test"] + [other]}
    output = tmp_path / "out"
    outcome = run(
        tmp_path / "split.json", output, loader=loader_for([], mixed), **tracking(tmp_path)
    )
    assert outcome.status == "failed"
    assert "more than one dataset version" in read(output / "report.json")["reason"]


def test_an_unexpected_error_crashes_instead_of_becoming_a_failed_run(tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError("a programming error, not a data problem")

    monkeypatch.setattr("agentwatch_model.train_candidates.fit_candidate", broken)
    with pytest.raises(ValueError, match="programming error"):
        run(
            tmp_path / "split.json",
            tmp_path / "out",
            loader=loader_for([], SPLITS),
            **tracking(tmp_path),
        )


def test_a_failure_removes_an_earlier_attempts_models(tmp_path):
    output = tmp_path / "out"
    run(tmp_path / "split.json", output, loader=loader_for([], SPLITS), **tracking(tmp_path))
    assert (output / "classifier.txt").exists()
    run(tmp_path / "split.json", output, loader=blocked, **tracking(tmp_path))
    assert not (output / "classifier.txt").exists() and not (output / "predictions.jsonl").exists()


def test_explore_writes_apart_from_evaluated_results_and_marks_the_run(tmp_path):
    output = tmp_path / "out"
    run(tmp_path / "split.json", output, loader=loader_for([], SPLITS), **tracking(tmp_path))
    evaluated = file_sha256(output / "classifier.txt")
    calls = []
    run(
        tmp_path / "split.json",
        output,
        explore=True,
        loader=loader_for(calls, SPLITS),
        **tracking(tmp_path),
    )
    assert [name for name, _ in calls] == ["train", "validation", "validation"]
    assert all(unchecked for _, unchecked in calls)
    assert file_sha256(output / "classifier.txt") == evaluated
    exploratory = output / "exploratory"
    assert read(exploratory / "preprocessing.json")["evaluation"] is False
    statuses = sorted(r.data.tags["status"] for r in logged_runs(tmp_path))
    assert statuses == ["evaluated", "exploratory"]


def test_an_evaluated_run_logs_none_of_the_exploratory_files(tmp_path):
    output = tmp_path / "out"
    loader = loader_for([], SPLITS)
    run(tmp_path / "split.json", output, explore=True, loader=loader, **tracking(tmp_path))
    outcome = run(tmp_path / "split.json", output, loader=loader, **tracking(tmp_path))
    logged_runs(tmp_path)  # points mlflow at this test's tracking database
    names = {item.path for item in mlflow.MlflowClient().list_artifacts(outcome.run_id)}
    assert "report.json" in names and "classifier.txt" in names
    assert not any(name.startswith("exploratory") for name in names)


def test_only_the_attempts_named_files_are_logged(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    (output / "notes.txt").write_text("not an artifact\n")
    outcome = run(
        tmp_path / "split.json", output, loader=loader_for([], SPLITS), **tracking(tmp_path)
    )
    logged_runs(tmp_path)  # points mlflow at this test's tracking database
    names = {item.path for item in mlflow.MlflowClient().list_artifacts(outcome.run_id)}
    assert "notes.txt" not in names and "tracking.json" not in names
    assert {"report.json", "classifier.txt", "predictions.jsonl"} <= names


def test_a_failed_model_save_leaves_no_staging_file(tmp_path, monkeypatch):
    import lightgbm
    from lightgbm.basic import LightGBMError

    def failing_save(self, filename, **kwargs):
        Path(filename).write_text("partial")
        raise LightGBMError("could not write the model")

    monkeypatch.setattr(lightgbm.Booster, "save_model", failing_save)
    output = tmp_path / "out"
    outcome = run(
        tmp_path / "split.json", output, loader=loader_for([], SPLITS), **tracking(tmp_path)
    )
    assert outcome.status == "failed"
    assert not [path.name for path in output.iterdir() if path.name.startswith(".")]


def test_a_blocked_run_that_was_not_tracked_exits_three(tmp_path, monkeypatch):
    from mlflow.exceptions import MlflowException

    def broken_tracking(*args, **kwargs):
        raise MlflowException("tracking server unavailable")

    monkeypatch.setattr("agentwatch_model.train_candidates.log_run", broken_tracking)
    monkeypatch.setattr("agentwatch_model.train_candidates.load_split", blocked)
    output = tmp_path / "out"
    assert main(["--split", str(tmp_path / "split.json"), "--output", str(output)]) == 3
    assert read(output / "report.json")["status"] == "blocked"


def test_a_tracking_failure_keeps_the_report_and_exits_three(tmp_path, monkeypatch):
    from mlflow.exceptions import MlflowException

    def broken_tracking(*args, **kwargs):
        raise MlflowException("tracking server unavailable")

    monkeypatch.setattr("agentwatch_model.train_candidates.log_run", broken_tracking)
    monkeypatch.setattr("agentwatch_model.train_candidates.load_split", loader_for([], SPLITS))
    output = tmp_path / "out"
    assert main(["--split", str(tmp_path / "split.json"), "--output", str(output)]) == 3
    assert read(output / "report.json")["status"] == "evaluated"
    assert read(output / "tracking.json") == {
        "logged": False,
        "error": "tracking server unavailable",
    }


def test_main_exits_one_for_blocked_and_failed_runs(tmp_path, monkeypatch):
    monkeypatch.setattr("agentwatch_model.train_candidates.load_split", blocked)
    args = [
        "--split",
        str(tmp_path / "split.json"),
        "--output",
        str(tmp_path / "out"),
        "--tracking-uri",
        f"sqlite:///{tmp_path / 'mlflow.db'}",
        "--artifact-root",
        str(tmp_path / "artifacts"),
    ]
    assert main(args) == 1


def test_main_exits_two_when_lightgbm_cannot_load(tmp_path, monkeypatch, capsys):
    # A None entry makes `import lightgbm` raise ImportError, as a missing OpenMP runtime does.
    monkeypatch.setitem(sys.modules, "lightgbm", None)
    args = [
        "--split",
        str(tmp_path / "split.json"),
        "--output",
        str(tmp_path / "out"),
        "--tracking-uri",
        f"sqlite:///{tmp_path / 'mlflow.db'}",
    ]
    assert main(args) == 2
    assert "LightGBM could not load" in capsys.readouterr().err
    assert not (tmp_path / "out").exists() and not (tmp_path / "mlflow.db").exists()


def test_main_exits_two_on_a_configuration_error(tmp_path, capsys):
    args = ["--split", str(tmp_path / "split.json"), "--training", str(tmp_path / "missing.toml")]
    assert main(args) == 2
    assert "Training error" in capsys.readouterr().err
