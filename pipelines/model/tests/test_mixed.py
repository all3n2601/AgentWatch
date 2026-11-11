"""mixed-scoring-v1 on synthetic rows: naming any confirmed resource of a mixed step is a hit,
and every score is measured against the recorded labels."""

from dataclasses import replace

import pytest
from agentwatch_model.contract import DEFAULT_CONTRACT, load_contract
from agentwatch_model.mixed import (
    hit_rate,
    mixed_macro_f1,
    mixed_summary,
    per_resource_scores,
    single_class,
    split_joint_only,
)
from agentwatch_model.report import build_report
from agentwatch_model.rule_baseline import RowError, validate_rows
from agentwatch_model.rules import DEFAULT_RULES, load_rules
from rule_fixtures import row

CLASSES = ("none", "cpu_sandbox", "inference", "retrieval")


def mixed(*members: str, basis: str = "mixed_confirmed", **options) -> dict:
    return row("mixed", "mixed", **options) | {
        "label_resources": ",".join(sorted(members)),
        "label_basis": basis,
    }


@pytest.fixture(scope="module")
def contract():
    return replace(
        load_contract(DEFAULT_CONTRACT),
        min_test_steps_per_class=1,
        min_clean_steps=1,
        min_mixed_steps=2,
    )


def test_naming_any_confirmed_resource_of_a_mixed_step_is_a_hit():
    rows = [mixed("cpu_sandbox", "inference")] * 3
    assert hit_rate(rows, ["cpu_sandbox", "inference", "none"], CLASSES) == pytest.approx(2 / 3)


def test_hit_rate_of_no_mixed_steps_is_unmeasured():
    assert hit_rate([], [], CLASSES) is None


def test_four_class_metrics_cover_single_class_steps_only():
    rows = [mixed("cpu_sandbox", "inference"), row(), row("cpu_sandbox", "contended")]
    kept, predictions = single_class(rows, ["cpu_sandbox", "none", "none"])
    assert [r["label_resource"] for r in kept] == ["none", "cpu_sandbox"]
    assert predictions == ["none", "none"]


def test_joint_only_steps_are_split_out_of_scoring():
    rows = [
        mixed("cpu_sandbox", "inference"),
        mixed("cpu_sandbox", "retrieval", basis="mixed_joint_only"),
        row(),
    ]
    kept, kept_predictions, joint, joint_predictions = split_joint_only(
        rows, ["cpu_sandbox", "retrieval", "none"]
    )
    assert [r["label_basis"] for r in joint] == ["mixed_joint_only"]
    assert joint_predictions == ["retrieval"] and kept_predictions == ["cpu_sandbox", "none"]
    assert len(kept) == 2


def test_per_resource_scores_count_every_confirmed_resource_as_a_positive():
    rows = [mixed("cpu_sandbox", "inference"), row(), row("inference", "contended")]
    scores = per_resource_scores(rows, ["cpu_sandbox", "none", "none"], CLASSES)
    assert scores["cpu_sandbox"] == {
        "support": 1,
        "predicted": 1,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
    }
    # Two steps were slowed by inference and neither was named inference.
    assert scores["inference"]["support"] == 2 and scores["inference"]["recall"] == 0.0
    assert scores["none"]["precision"] == 0.5 and scores["retrieval"]["f1"] is None


def test_mixed_macro_f1_is_over_the_resources_the_labels_confirm():
    rows = [mixed("cpu_sandbox", "inference"), mixed("cpu_sandbox", "inference")]
    value, resources = mixed_macro_f1(rows, ["cpu_sandbox", "inference"], CLASSES)
    assert resources == ("cpu_sandbox", "inference")
    # Each resource is named on one of its two steps: precision 1, recall 0.5, F1 2/3.
    assert value == pytest.approx(2 / 3)


def test_naming_one_resource_every_time_does_not_score_well():
    # 9 of 10 steps include inference; always naming it hits 90% but misses cpu and retrieval.
    rows = [mixed("cpu_sandbox", "inference")] * 5 + [mixed("inference", "retrieval")] * 4
    rows += [mixed("cpu_sandbox", "retrieval")]
    always = ["inference"] * len(rows)
    assert hit_rate(rows, always, CLASSES) == pytest.approx(0.9)
    value, _ = mixed_macro_f1(rows, always, CLASSES)
    assert value < 0.4


def test_the_resources_the_macro_covers_do_not_depend_on_the_prediction():
    rows = [mixed("cpu_sandbox", "inference"), mixed("inference", "retrieval")]
    _, first = mixed_macro_f1(rows, ["none", "none"], CLASSES)
    _, second = mixed_macro_f1(rows, ["retrieval", "cpu_sandbox"], CLASSES)
    assert first == second == ("cpu_sandbox", "inference", "retrieval")


def test_the_mixed_summary_reports_hits_and_joint_only_steps_apart(contract):
    rows = [
        mixed("cpu_sandbox", "inference"),
        mixed("cpu_sandbox", "inference"),
        mixed("cpu_sandbox", "retrieval", basis="mixed_joint_only"),
        row(),
    ]
    summary = mixed_summary(rows, ["inference", "none", "retrieval", "none"], contract)
    assert summary["steps"] == 2 and summary["hit_rate"] == 0.5
    assert summary["status"] == "reported" and summary["min_steps"] == 2
    assert summary["joint_only"] == {"steps": 1, "named_a_member": 1, "scored": False}
    assert summary["macro_f1_resources"] == ["cpu_sandbox", "inference"]
    assert summary["per_resource"]["cpu_sandbox"]["support"] == 2


def test_too_few_mixed_steps_leave_the_slice_inconclusive(contract):
    summary = mixed_summary([mixed("cpu_sandbox", "inference")], ["inference"], contract)
    assert summary["status"] == "inconclusive" and summary["hit_rate"] == 1.0


def test_the_rule_report_scores_single_class_and_mixed_steps_apart(contract):
    rules = load_rules(DEFAULT_RULES)
    test = [
        row(condition="clean"),
        row("cpu_sandbox", "contended", queue_fraction=0.9),
        mixed("cpu_sandbox", "inference", queue_fraction=0.9),
        mixed("inference", "retrieval"),
        mixed("cpu_sandbox", "retrieval", basis="mixed_joint_only", queue_fraction=0.9, excess=0.9),
    ]
    predictions = [rules.predict(r) for r in test]
    report = build_report(
        contract, rules, [row()], test, predictions, evaluation=True, resamples=10, seed=1
    )
    metrics = report["metrics"]
    assert report["mixed_scoring"] == "mixed-scoring-v1"
    # Only the two single-class steps enter the four-class confusion matrix.
    assert sum(sum(r.values()) for r in metrics["confusion_matrix"].values()) == 2
    assert metrics["slices"]["mixed_contention"]["hit_rate"] == 0.5
    assert metrics["slices"]["mixed_contention"]["joint_only"]["steps"] == 1
    assert metrics["multi_resource"]["classes"]["inference"]["support"] == 2
    assert "label_resources" in metrics["multi_resource"]["semantics"]
    # The joint-only step's excess fraction is not scored either.
    assert metrics["constant_references"]["mae_zero"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"label_resources": "cpu_sandbox"}, "two or more"),
        ({"label_resources": "cpu_sandbox,gpu"}, "unknown resource"),
        ({"label_resources": "none,cpu_sandbox"}, "unknown resource"),
        ({"label_basis": "mixed_joint"}, "basis 'mixed_joint'"),
        ({"label_basis": None}, "basis None"),
    ],
)
def test_a_mixed_row_must_name_two_or_more_resources_and_a_known_basis(overrides, message):
    with pytest.raises(RowError, match=message):
        validate_rows([mixed("cpu_sandbox", "inference") | overrides], "test")


def test_a_mixed_row_without_its_resources_is_refused():
    step = mixed("cpu_sandbox", "inference")
    del step["label_resources"]
    with pytest.raises(RowError, match="label_resources"):
        validate_rows([step], "test")


def test_a_well_formed_mixed_row_is_accepted():
    validate_rows([mixed("cpu_sandbox", "inference"), row()], "test")
