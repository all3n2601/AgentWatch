import json

import mlflow
import pytest
from agentwatch_data.splits import SplitError
from agentwatch_model.calibrate import main, run
from agentwatch_model.calibration import CalibrationError
from agentwatch_model.comparison import MODEL_NOT_EVALUATED
from agentwatch_model.contract import DEFAULT_CONTRACT
from agentwatch_model.features import FEATURES
from agentwatch_model.rule_baseline import RowError
from calibration_fixtures import (
    COLUMNS,
    overconfident,
    split_loader,
    steps,
    unsure,
    write_candidates,
)
from training_fixtures import featured_row

VALIDATION = steps(30, "valid")
TEST = steps(30, "test")


def mistakes_unsure(row: dict, index: int) -> dict[str, float]:
    """Sure and right on 8 of 10 steps; on the rest, wrong and unsure."""
    label = row["label_resource"]
    if index % 10 < 8:
        return {name: 0.97 if name == label else 0.01 for name in COLUMNS}
    wrong = COLUMNS[(COLUMNS.index(label) + 1) % len(COLUMNS)]
    return {name: 0.4 if name == wrong else 0.2 for name in COLUMNS}


def always_unsure_and_wrong(row: dict, index: int) -> dict[str, float]:
    wrong = COLUMNS[(COLUMNS.index(row["label_resource"]) + 1) % len(COLUMNS)]
    return {name: 0.4 if name == wrong else 0.2 for name in COLUMNS}


def mixed_steps(
    count: int, members: str = "cpu_sandbox,inference", basis: str = "mixed_confirmed", prefix="mix"
) -> list[dict]:
    """Synthetic mixed-contention steps, confirmed on `members`, for mixed-scoring-v1 tests."""
    return [
        featured_row("cpu_sandbox", index, task_id=f"{prefix}-{index % 5}")
        | {
            "label_resource": "mixed",
            "label_resources": members,
            "label_basis": basis,
            "condition": "mixed",
            "label_excess_fraction": 0.7,
        }
        for index in range(count)
    ]


def names_cpu_sandbox(row: dict, index: int) -> dict[str, float]:
    """Sure of cpu_sandbox on a mixed step (a hit when cpu_sandbox is a confirmed member)."""
    if row["label_resource"] == "mixed":
        return {name: 0.97 if name == "cpu_sandbox" else 0.01 for name in COLUMNS}
    return overconfident(row, index)


def unsure_on_mixed(row: dict, index: int) -> dict[str, float]:
    if row["label_resource"] == "mixed":
        return {name: 0.4 if name == "cpu_sandbox" else 0.2 for name in COLUMNS}
    return mistakes_unsure(row, index)


def prepare(tmp_path, validation=VALIDATION, test=TEST, **options):
    tmp_path.mkdir(parents=True, exist_ok=True)
    split = tmp_path / "split.json"
    split.write_text("{}\n")
    candidates = tmp_path / "candidates"
    write_candidates(candidates, split, validation, test, **options)
    return split, candidates


def tracking(tmp_path):
    return {
        "tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
        "artifact_root": tmp_path / "artifacts",
    }


def calibrate(tmp_path, split, candidates, splits=None, calls=None, **kwargs):
    splits = splits or {"validation": VALIDATION, "test": TEST}
    return run(
        split,
        candidates,
        tmp_path / "out",
        loader=split_loader(splits, calls),
        **tracking(tmp_path),
        **kwargs,
    )


def read(path):
    return json.loads(path.read_text())


def predictions(tmp_path):
    return [json.loads(line) for line in (tmp_path / "out" / "predictions.jsonl").open()]


def test_overconfident_scores_are_softened_and_calibrated_on_validation(tmp_path):
    split, candidates = prepare(tmp_path)
    calls = []
    outcome = calibrate(tmp_path, split, candidates, calls=calls)
    assert outcome.status == "evaluated" and outcome.run_id
    assert calls == [("validation", False), ("test", False)]
    report = read(tmp_path / "out" / "report.json")
    assert report["calibration"]["fitted"] is True and report["calibration"]["temperature"] > 1
    ece = report["metrics"]["calibration"]
    assert ece["ece_uncalibrated"] == pytest.approx(0.27)
    limit = report["gates"]["calibration"]["threshold"]
    assert ece["ece_calibrated"] < limit < ece["ece_uncalibrated"]
    assert report["gates"]["calibration"]["outcome"] == "pass"
    saved = read(tmp_path / "out" / "calibration.json")
    assert saved["fitted"] is True
    assert saved["temperature"] == report["calibration"]["temperature"]
    assert saved["probability_columns"] == list(COLUMNS)
    assert saved["model_files"]["classifier"]["sha256"] == "0" * 64
    # The gate outcomes travel with the parameters: mixed contention is unmeasured, so this
    # calibration still blocks promotion even though its ECE passed.
    assert saved["gates"] == {"calibration": "pass", "abstention": "inconclusive"}
    assert saved["blocks_promotion"] is True
    assert saved["contract"]["hash"] == report["contract"]["hash"]
    lines = predictions(tmp_path)
    assert len(lines) == len(TEST)
    for line in lines:
        assert line["confidence"] < line["raw_confidence"]
        assert line["calibrated"] is True
        assert (line["attribution"] is None) == line["abstained"]


def test_the_calibration_depends_on_validation_scores_only(tmp_path):
    split, candidates = prepare(tmp_path, test_scorer=overconfident)
    first = calibrate(tmp_path, split, candidates)
    chosen = read(tmp_path / "out" / "calibration.json")
    split, candidates = prepare(tmp_path / "again", test_scorer=unsure)
    second = calibrate(tmp_path / "again", split, candidates)
    again = read(tmp_path / "again" / "out" / "calibration.json")
    assert first.status == second.status == "evaluated"
    assert again["temperature"] == chosen["temperature"]
    assert again["abstention_threshold"] == chosen["abstention_threshold"]


@pytest.mark.parametrize("per_class", [0, 2])
def test_too_few_validation_steps_leave_the_model_uncalibrated(tmp_path, per_class):
    validation = steps(per_class, "valid")
    split, candidates = prepare(tmp_path, validation=validation)
    outcome = calibrate(tmp_path, split, candidates, {"validation": validation, "test": TEST})
    assert outcome.status == "evaluated"
    report = read(tmp_path / "out" / "report.json")
    assert report["calibration"]["fitted"] is False
    assert f"{len(validation)} usable validation steps" in report["calibration"]["reason"]
    for name in ("calibration", "abstention"):
        assert report["gates"][name]["outcome"] == "inconclusive"
        assert "usable validation steps" in report["gates"][name]["reason"]
    saved = read(tmp_path / "out" / "calibration.json")
    assert saved["fitted"] is False and saved["temperature"] is None
    assert saved["abstention_threshold"] is None
    for line in predictions(tmp_path):
        assert line["calibrated"] is False and line["abstained"] is False
        # No calibrated confidence exists, so the field is null rather than the raw score.
        assert line["confidence"] is None and line["probabilities"] is None
        assert line["raw_confidence"] is not None
    # The report agrees with the predictions: calibrated metrics are unavailable, raw ones kept.
    metrics = report["metrics"]["calibration"]
    assert metrics["ece_calibrated"] is None and metrics["mean_confidence_calibrated"] is None
    assert metrics["ece_uncalibrated"] == pytest.approx(0.27)
    assert metrics["mean_confidence_uncalibrated"] == pytest.approx(0.97)
    for group in report["metrics"]["slices"]["by_condition"].values():
        assert group["ece_calibrated"] is None and group["ece_uncalibrated"] is not None
    missing = report["metrics"]["slices"]["missing_features"]
    assert missing["missing_any"]["ece_calibrated"] is None


def zero_scores(row: dict, index: int) -> dict[str, float]:
    return dict.fromkeys(COLUMNS, 0.0)


def test_malformed_validation_scores_are_refused_even_below_the_fit_minimum(tmp_path):
    validation = steps(1, "valid")
    split, candidates = prepare(tmp_path, validation=validation, validation_scorer=zero_scores)
    with pytest.raises(CalibrationError, match="not a distribution"):
        calibrate(tmp_path, split, candidates, {"validation": validation, "test": TEST})


def test_low_confidence_steps_abstain_and_count_as_misses_on_clean_steps(tmp_path):
    split, candidates = prepare(
        tmp_path, validation_scorer=mistakes_unsure, test_scorer=mistakes_unsure
    )
    calibrate(tmp_path, split, candidates)
    report = read(tmp_path / "out" / "report.json")
    assert report["calibration"]["abstention_threshold"] > 0
    lines = predictions(tmp_path)
    clean = [line for line in lines if line["condition"] == "clean"]
    assert sum(line["abstained"] for line in clean) == 6
    assert all(line["attribution"] is None for line in clean if line["abstained"])
    # 24 of 30 clean steps are named none; the 6 abstentions are misses, not passes.
    assert report["gates"]["clean_none_rate"]["value"] == pytest.approx(0.8)
    abstention = report["gates"]["abstention"]
    assert abstention["value"] == pytest.approx(0.2)
    assert abstention["scopes"]["mixed_contention"] == {"steps": 0, "abstained": 0, "rate": None}
    # Overall and per-class rates are within the limit, but no mixed step was scored.
    assert abstention["outcome"] == "inconclusive"
    assert "mixed" in abstention["reason"].lower()


def test_abstaining_on_more_test_steps_than_the_limit_fails_the_gate(tmp_path):
    split, candidates = prepare(
        tmp_path, validation_scorer=mistakes_unsure, test_scorer=always_unsure_and_wrong
    )
    calibrate(tmp_path, split, candidates)
    gate = read(tmp_path / "out" / "report.json")["gates"]["abstention"]
    assert gate["value"] == 1.0 and gate["outcome"] == "fail"


def test_missing_features_are_reported_as_their_own_slice(tmp_path):
    # The fixture leaves some readings unmeasured; measure them all, then drop one on purpose.
    measured = [row | {name: 0.0 for name in FEATURES if row[name] is None} for row in TEST]
    test = [
        row | {"queue_seconds": None} if index % 4 == 0 else row
        for index, row in enumerate(measured)
    ]
    split, candidates = prepare(tmp_path, test=test)
    calibrate(tmp_path, split, candidates, {"validation": VALIDATION, "test": test})
    missing = read(tmp_path / "out" / "report.json")["metrics"]["slices"]["missing_features"]
    assert missing["missing_any"]["steps"] == 30 and missing["complete"]["steps"] == 90
    assert missing["steps_missing_each_feature"]["queue_seconds"] == 30
    assert missing["missing_any"]["ece_calibrated"] is not None


def test_early_stopping_on_the_fit_split_is_recorded_as_a_limitation(tmp_path):
    split, candidates = prepare(tmp_path)
    calibrate(tmp_path, split, candidates)
    fit = read(tmp_path / "out" / "report.json")["fit"]
    assert fit["early_stopping_on_fit_split"] is True and "optimistic" in fit["limitation"]
    split, candidates = prepare(tmp_path / "unused", early_stopping="not_used")
    calibrate(tmp_path / "unused", split, candidates)
    fit = read(tmp_path / "unused" / "out" / "report.json")["fit"]
    assert fit["early_stopping_on_fit_split"] is False and fit["limitation"] is None


def test_gates_calibration_cannot_change_are_carried_from_the_candidates(tmp_path):
    split, candidates = prepare(tmp_path)
    calibrate(tmp_path, split, candidates)
    gates = read(tmp_path / "out" / "report.json")["gates"]
    assert gates["regression_error"]["outcome"] == "pass"
    assert "M03" in gates["regression_error"]["source"]
    assert gates["unseen_tasks"] == {
        "outcome": "not_evaluated",
        "reason": MODEL_NOT_EVALUATED["unseen_tasks"],
    }
    assert gates["calibration"]["outcome"] != "not_evaluated"
    assert {"gain_over_rules", "clean_none_rate", "baseline_and_clean_none_rate"} <= set(gates)


def test_candidates_that_were_not_evaluated_give_a_blocked_report(tmp_path):
    split, candidates = prepare(tmp_path, status="blocked")
    outcome = calibrate(tmp_path, split, candidates)
    assert outcome.status == "blocked"
    report = read(tmp_path / "out" / "report.json")
    assert "not evaluated" in report["reason"] and report["metrics"] is None
    assert not (tmp_path / "out" / "predictions.jsonl").exists()
    assert not (tmp_path / "out" / "calibration.json").exists()


def test_a_blocked_split_gives_a_blocked_report(tmp_path):
    split, candidates = prepare(tmp_path)

    def blocked(*args, **kwargs):
        raise SplitError("Split x.json: its leakage check is blocked")

    outcome = run(split, candidates, tmp_path / "out", loader=blocked, **tracking(tmp_path))
    assert outcome.status == "blocked"
    assert "leakage" in read(tmp_path / "out" / "report.json")["reason"]


def test_changed_test_predictions_are_refused_and_the_earlier_result_kept(tmp_path):
    split, candidates = prepare(tmp_path)
    calibrate(tmp_path, split, candidates)
    with (candidates / "predictions.jsonl").open("a") as handle:
        handle.write("{}\n")
    with pytest.raises(CalibrationError, match="do not match"):
        calibrate(tmp_path, split, candidates)
    assert read(tmp_path / "out" / "report.json")["status"] == "evaluated"


@pytest.mark.parametrize("name", ["validation_predictions.jsonl", "preprocessing.json"])
def test_an_edited_candidate_file_is_refused(tmp_path, name):
    split, candidates = prepare(tmp_path)
    path = candidates / name
    path.write_text(path.read_text().replace("0.97", "0.96", 1).replace("training-v1", "x", 1))
    with pytest.raises(CalibrationError, match="do not match"):
        calibrate(tmp_path, split, candidates)


def test_model_files_that_differ_from_the_candidates_report_are_refused(tmp_path):
    split, candidates = prepare(tmp_path)
    report = read(candidates / "report.json")
    (candidates / "report.json").write_text(
        json.dumps(report | {"model_files": {"classifier": "f" * 64}})
    )
    with pytest.raises(CalibrationError, match="model files"):
        calibrate(tmp_path, split, candidates)


def test_candidates_that_did_not_score_the_test_split_are_refused(tmp_path):
    split, candidates = prepare(tmp_path)
    report = read(candidates / "report.json")
    (candidates / "report.json").write_text(json.dumps(report | {"scored_split": "validation"}))
    with pytest.raises(CalibrationError, match="test split"):
        calibrate(tmp_path, split, candidates)


def test_a_measured_failure_is_reported_even_while_a_class_lacks_steps(tmp_path):
    test = steps(30, "test", classes=COLUMNS[:3]) + steps(2, "few", classes=COLUMNS[3:])
    split, candidates = prepare(
        tmp_path, test=test, validation_scorer=mistakes_unsure, test_scorer=always_unsure_and_wrong
    )
    calibrate(tmp_path, split, candidates, {"validation": VALIDATION, "test": test})
    gates = read(tmp_path / "out" / "report.json")["gates"]
    assert gates["abstention"]["outcome"] == "fail"
    assert gates["calibration"]["outcome"] == "fail"


def test_a_split_other_than_the_candidates_split_is_refused(tmp_path):
    split, candidates = prepare(tmp_path)
    split.write_text('{"changed": true}\n')
    with pytest.raises(CalibrationError, match="split"):
        calibrate(tmp_path, split, candidates)


def test_predictions_that_do_not_cover_the_split_rows_are_refused(tmp_path):
    split, candidates = prepare(tmp_path)
    extra = steps(1, "extra", classes=("none",))
    with pytest.raises(CalibrationError, match="test"):
        calibrate(tmp_path, split, candidates, {"validation": VALIDATION, "test": TEST + extra})


def test_a_contract_changed_since_training_is_refused(tmp_path):
    split, candidates = prepare(tmp_path)
    changed = tmp_path / "contract.toml"
    changed.write_text(DEFAULT_CONTRACT.read_text().replace("max_mae = 0.10", "max_mae = 0.11"))
    with pytest.raises(CalibrationError, match="contract"):
        calibrate(tmp_path, split, candidates, contract_path=changed)


def test_rows_with_an_unknown_label_are_refused(tmp_path):
    test = TEST[:-1] + [TEST[-1] | {"label_resource": "gpu"}]
    split, candidates = prepare(tmp_path)
    with pytest.raises(RowError, match="unknown label"):
        calibrate(tmp_path, split, candidates, {"validation": VALIDATION, "test": test})


def test_the_attempt_is_logged_to_mlflow_with_its_calibration(tmp_path):
    split, candidates = prepare(tmp_path)
    outcome = calibrate(tmp_path, split, candidates)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    [logged] = mlflow.search_runs(search_all_experiments=True, output_format="list")
    assert logged.info.run_id == outcome.run_id
    assert logged.data.tags["task"] == "M04" and logged.data.tags["status"] == "evaluated"
    assert logged.data.params["calibration_version"].startswith("calibration-v1+")
    assert logged.data.params["candidates_run_id"] == "m03-run"
    assert "ece_calibrated" in logged.data.metrics
    names = {item.path for item in mlflow.MlflowClient().list_artifacts(outcome.run_id)}
    assert {"report.json", "predictions.jsonl", "calibration.json"} <= names


def test_a_tracking_failure_keeps_the_report_and_exits_three(tmp_path, monkeypatch):
    from mlflow.exceptions import MlflowException

    def broken_tracking(*args, **kwargs):
        raise MlflowException("tracking server unavailable")

    split, candidates = prepare(tmp_path)
    monkeypatch.setattr("agentwatch_model.calibrate.log_run", broken_tracking)
    monkeypatch.setattr(
        "agentwatch_model.calibrate.load_split",
        split_loader({"validation": VALIDATION, "test": TEST}),
    )
    output = tmp_path / "out"
    args = ["--split", str(split), "--candidates", str(candidates), "--output", str(output)]
    assert main(args) == 3
    assert read(output / "report.json")["status"] == "evaluated"
    assert read(output / "tracking.json") == {
        "logged": False,
        "error": "tracking server unavailable",
    }


def test_main_exits_zero_when_evaluated_and_one_when_blocked(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "agentwatch_model.calibrate.load_split",
        split_loader({"validation": VALIDATION, "test": TEST}),
    )
    for status, code in (("evaluated", 0), ("blocked", 1)):
        directory = tmp_path / status
        split, candidates = prepare(directory, status=status)
        args = [
            "--split",
            str(split),
            "--candidates",
            str(candidates),
            "--output",
            str(directory / "out"),
            "--tracking-uri",
            f"sqlite:///{directory / 'mlflow.db'}",
            "--artifact-root",
            str(directory / "artifacts"),
        ]
        assert main(args) == code, status


def test_main_exits_two_without_a_candidates_report(tmp_path, capsys):
    args = ["--split", str(tmp_path / "split.json"), "--candidates", str(tmp_path / "missing")]
    assert main(args) == 2
    assert "make train-candidates" in capsys.readouterr().err


def test_mixed_steps_are_scored_by_hit_and_joint_only_steps_are_not_scored(tmp_path):
    joint = mixed_steps(4, "cpu_sandbox,retrieval", "mixed_joint_only", "joint")
    test = TEST + mixed_steps(32) + joint
    split, candidates = prepare(tmp_path, test=test, test_scorer=names_cpu_sandbox)
    calibrate(tmp_path, split, candidates, {"validation": VALIDATION, "test": test})
    report = read(tmp_path / "out" / "report.json")
    assert report["scored_steps"] == len(TEST) + 32
    assert report["joint_only_steps_not_scored"] == 4
    mixed = report["metrics"]["slices"]["mixed_contention"]
    # Naming a confirmed resource is a hit; an abstention counts as a miss.
    assert mixed["steps"] == 32
    assert mixed["hit_rate"] == pytest.approx(1 - mixed["abstention_rate"])
    assert mixed["joint_only"]["steps"] == 4 and mixed["joint_only"]["scored"] is False
    assert mixed["joint_only"]["named_a_member"] + mixed["joint_only"]["abstained"] == 4
    # ECE counts the 32 hits as correct and leaves the joint-only steps out: (84 + 32) / 152.
    calibration = report["metrics"]["calibration"]
    assert calibration["accuracy_before_abstention"] == pytest.approx(116 / 152)
    assert report["gates"]["abstention"]["scopes"]["mixed_contention"]["steps"] == 32
    # The four-class metrics cover single-class steps only: mixed steps change none of them.
    base_path = tmp_path / "base"
    base_split, base_candidates = prepare(base_path, test_scorer=names_cpu_sandbox)
    calibrate(base_path, base_split, base_candidates)
    base = read(base_path / "out" / "report.json")
    assert report["metrics"]["model"]["class_scores"] == base["metrics"]["model"]["class_scores"]


def test_abstaining_on_mixed_steps_above_the_limit_fails_the_abstention_gate(tmp_path):
    test = TEST + mixed_steps(32)
    split, candidates = prepare(
        tmp_path, test=test, validation_scorer=mistakes_unsure, test_scorer=unsure_on_mixed
    )
    calibrate(tmp_path, split, candidates, {"validation": VALIDATION, "test": test})
    abstention = read(tmp_path / "out" / "report.json")["gates"]["abstention"]
    assert abstention["scopes"]["mixed_contention"]["rate"] == 1.0
    assert abstention["outcome"] == "fail"
    assert "mixed_contention" in abstention["reason"]


def test_mixed_validation_steps_do_not_change_the_fit(tmp_path):
    base_split, base_candidates = prepare(tmp_path / "base", validation_scorer=mistakes_unsure)
    calibrate(tmp_path / "base", base_split, base_candidates)
    validation = VALIDATION + mixed_steps(20)
    split, candidates = prepare(
        tmp_path / "mixed", validation=validation, validation_scorer=unsure_on_mixed
    )
    calibrate(tmp_path / "mixed", split, candidates, {"validation": validation, "test": TEST})
    fit = read(tmp_path / "mixed" / "out" / "report.json")["calibration"]
    base = read(tmp_path / "base" / "out" / "report.json")["calibration"]
    assert fit["temperature"] == pytest.approx(base["temperature"])
    assert fit["abstention_threshold"] == pytest.approx(base["abstention_threshold"])
