import hashlib
import json

import pytest
from agentwatch_data.splits import SplitError
from agentwatch_model.rule_baseline import RowError, main, run
from rule_fixtures import row

TRAIN = [row(excess=0.1), row(excess=0.3)]
VALIDATION = [row(condition="clean", task_id="task-v")]
TEST = [row(condition="clean"), row("cpu_sandbox", "contended", queue_fraction=0.8, excess=0.7)]


def fake_loader(calls, splits=None):
    data = splits or {"train": TRAIN, "validation": VALIDATION, "test": TEST}

    def load(path, name, contract_path, frozen_schema, allow_unchecked_leakage=False):
        calls.append((name, allow_unchecked_leakage))
        return data[name]

    return load


def blocked_loader(path, name, contract_path, frozen_schema, allow_unchecked_leakage=False):
    raise SplitError("Split x.json: its leakage check is failed; the contract blocks evaluation")


def read_report(output):
    return json.loads((output / "report.json").read_text())


def test_run_writes_the_report_and_one_prediction_per_test_row(tmp_path):
    calls = []
    status = run(tmp_path / "split.json", tmp_path / "out", loader=fake_loader(calls))
    report = read_report(tmp_path / "out")
    text = (tmp_path / "out" / "predictions.jsonl").read_text()
    predictions = [json.loads(line) for line in text.splitlines()]
    assert status == "evaluated" and report["status"] == "evaluated"
    assert calls == [("train", False), ("test", False)]
    assert [p["prediction"] for p in predictions] == ["none", "cpu_sandbox"]
    assert predictions[1] | {"step_id": None} == {
        "run_id": "run-1",
        "step_id": None,
        "task_id": "task-a",
        "condition": "contended",
        "split": "test",
        "dataset_version": "steps-v1.2+synthetic",
        "label_version": "labels-v1.1+synthetic",
        "label_resource": "cpu_sandbox",
        "prediction": "cpu_sandbox",
        "rules_version": report["rules"]["version"],
        "evaluation": True,
    }
    assert report["split"]["file"] == "split.json" and report["scored_steps"] == 2
    assert report["scored_split"] == "test"
    assert report["predictions"]["sha256"] == hashlib.sha256(text.encode()).hexdigest()


def test_a_blocked_split_writes_a_blocked_report_and_no_predictions(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    (output / "predictions.jsonl").write_text("stale\n")
    status = run(tmp_path / "split.json", output, loader=blocked_loader)
    report = read_report(output)
    assert status == "blocked" and report["status"] == "blocked"
    assert "leakage check is failed" in report["reason"]
    assert not (output / "predictions.jsonl").exists()


def test_explore_scores_validation_not_test_and_marks_every_output(tmp_path):
    calls = []
    run(tmp_path / "split.json", tmp_path / "out", explore=True, loader=fake_loader(calls))
    report = read_report(tmp_path / "out" / "exploratory")
    line = (tmp_path / "out" / "exploratory" / "predictions.jsonl").read_text().splitlines()[0]
    assert calls == [("train", True), ("validation", True)]
    assert report["evaluation"] is False and report["scored_split"] == "validation"
    assert json.loads(line)["evaluation"] is False and json.loads(line)["split"] == "validation"


def test_an_exploratory_run_leaves_the_evaluated_result_in_place(tmp_path):
    output = tmp_path / "out"
    run(tmp_path / "split.json", output, loader=fake_loader([]))
    evaluated = (output / "report.json").read_text()
    run(tmp_path / "split.json", output, explore=True, loader=fake_loader([]))
    assert (output / "report.json").read_text() == evaluated
    assert read_report(output / "exploratory")["status"] == "exploratory"


def test_a_run_that_stops_on_bad_rows_leaves_no_earlier_outputs(tmp_path):
    output = tmp_path / "out"
    run(tmp_path / "split.json", output, loader=fake_loader([]))
    loader = fake_loader([], {"train": TRAIN, "test": [row(label="gpu")]})
    with pytest.raises(RowError):
        run(tmp_path / "split.json", output, loader=loader)
    assert not (output / "report.json").exists()
    assert not (output / "predictions.jsonl").exists()


def test_rows_missing_a_required_field_are_refused(tmp_path):
    broken = [row() | {}, {k: v for k, v in row().items() if k != "label_version"}]
    loader = fake_loader([], {"train": TRAIN, "test": broken})
    with pytest.raises(RowError, match="test row 1 is missing label_version"):
        run(tmp_path / "split.json", tmp_path / "out", loader=loader)


def test_duplicate_rows_are_refused(tmp_path):
    duplicate = row()
    loader = fake_loader([], {"train": TRAIN, "test": [duplicate, dict(duplicate)]})
    with pytest.raises(RowError, match="more than one row"):
        run(tmp_path / "split.json", tmp_path / "out", loader=loader)


def test_main_exits_one_when_the_split_is_blocked(tmp_path, monkeypatch):
    monkeypatch.setattr("agentwatch_model.rule_baseline.load_split", blocked_loader)
    args = ["--split", str(tmp_path / "split.json"), "--output", str(tmp_path / "out")]
    assert main(args) == 1
    assert read_report(tmp_path / "out")["status"] == "blocked"


def test_main_exits_two_on_a_configuration_error(tmp_path, capsys):
    args = [
        "--split",
        str(tmp_path / "split.json"),
        "--output",
        str(tmp_path / "out"),
        "--rules",
        str(tmp_path / "missing.toml"),
    ]
    assert main(args) == 2
    assert "Rule baseline error" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"label_excess_fraction": None}, "not a number from 0 to 1"),
        ({"label_excess_fraction": "0.2"}, "not a number from 0 to 1"),
        ({"label_excess_fraction": 1.5}, "not a number from 0 to 1"),
        ({"label_resource": "gpu"}, "unknown label"),
        ({"dataset_version": "steps-v1.3+other"}, "more than one dataset version"),
    ],
)
def test_rows_with_unusable_labels_or_versions_are_refused(tmp_path, change, message):
    loader = fake_loader([], {"train": TRAIN, "test": [row() | change]})
    with pytest.raises(RowError, match=message):
        run(tmp_path / "split.json", tmp_path / "out", loader=loader)


def test_main_exits_two_on_unusable_rows(tmp_path, monkeypatch, capsys):
    loader = fake_loader([], {"train": TRAIN, "test": [row(label="gpu")]})
    monkeypatch.setattr("agentwatch_model.rule_baseline.load_split", loader)
    args = ["--split", str(tmp_path / "split.json"), "--output", str(tmp_path / "out")]
    assert main(args) == 2
    assert "unknown label 'gpu'" in capsys.readouterr().err
