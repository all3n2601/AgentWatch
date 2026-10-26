from dataclasses import replace

import pytest
from agentwatch_model.comparison import (
    MODEL_NOT_EVALUATED,
    build_model_report,
    gain_gate,
    regression_gate,
)
from agentwatch_model.contract import DEFAULT_CONTRACT, load_contract
from agentwatch_model.rules import DEFAULT_RULES, load_rules
from rule_fixtures import row


@pytest.fixture(scope="module")
def contract():
    return replace(load_contract(DEFAULT_CONTRACT), min_test_steps_per_class=1, min_clean_steps=2)


@pytest.fixture(scope="module")
def rules():
    return load_rules(DEFAULT_RULES)


def test_gain_gate_passes_only_when_the_model_is_not_worse_than_rules(contract):
    assert gain_gate(0.8, 0.7, [], contract)["outcome"] == "pass"
    assert gain_gate(0.7, 0.7, [], contract)["outcome"] == "pass"
    assert gain_gate(0.6, 0.7, [], contract)["outcome"] == "fail"


def test_gain_gate_is_inconclusive_with_too_few_steps_or_no_score(contract):
    assert gain_gate(0.9, 0.1, ["retrieval"], contract)["outcome"] == "inconclusive"
    assert gain_gate(None, 0.5, [], contract)["outcome"] == "inconclusive"


def test_regression_gate_needs_the_limit_and_both_constant_references(contract):
    references = {"mae_training_mean": 0.3, "mae_zero": 0.25}
    assert regression_gate(0.05, references, contract)["outcome"] == "pass"
    assert regression_gate(0.2, references, contract)["outcome"] == "fail"  # above 0.10
    assert (
        regression_gate(0.05, {"mae_training_mean": 0.04, "mae_zero": 0.3}, contract)["outcome"]
        == "fail"
    )
    assert regression_gate(None, references, contract)["outcome"] == "inconclusive"
    assert (
        regression_gate(0.05, {"mae_training_mean": None, "mae_zero": 0.3}, contract)["outcome"]
        == "inconclusive"
    )


def report(contract, rules, test, model_labels, fractions, evaluation=True):
    rule_labels = [rules.predict(r) for r in test]
    return build_model_report(
        contract,
        rules,
        {"version": "training-v1+test", "class_order": ["none", "cpu_sandbox"]},
        [row(excess=0.2)],
        test,
        model_labels,
        fractions,
        rule_labels,
        evaluation=evaluation,
        resamples=50,
        seed=0,
    )


def test_model_and_rules_are_scored_on_identical_rows(contract, rules):
    test = [
        row(condition="clean"),
        row(condition="clean"),
        row("cpu_sandbox", "contended", excess=0.8),
    ]
    result = report(contract, rules, test, ["none", "none", "cpu_sandbox"], [0.0, 0.0, 0.8])
    model, baseline = result["metrics"]["model"], result["metrics"]["rules"]
    assert sum(sum(r.values()) for r in model["confusion_matrix"].values()) == 3
    assert sum(sum(r.values()) for r in baseline["confusion_matrix"].values()) == 3
    assert result["comparison"]["rows"] == 3
    assert result["metrics"]["regression"]["mae"] == pytest.approx(0.0)


def test_calibration_and_abstention_are_left_to_m04(contract, rules):
    gates = report(contract, rules, [row(condition="clean")] * 2, ["none", "none"], [0.0, 0.0])[
        "gates"
    ]
    assert gates["calibration"] == {
        "outcome": "not_evaluated",
        "reason": MODEL_NOT_EVALUATED["calibration"],
    }
    assert "M04" in MODEL_NOT_EVALUATED["abstention"]
    assert gates["clean_none_rate"]["outcome"] == "pass"


def test_exploratory_model_reports_evaluate_no_gate(contract, rules):
    result = report(
        contract,
        rules,
        [row(condition="clean")] * 2,
        ["none", "none"],
        [0.0, 0.0],
        evaluation=False,
    )
    assert result["status"] == "exploratory"
    assert {g["outcome"] for g in result["gates"].values()} == {"not_evaluated"}
    assert result["summary"]["blocks_promotion"] is True


def test_gain_gate_is_inconclusive_when_training_lacked_a_class(contract):
    gate = gain_gate(0.9, 0.5, [], contract, absent=("retrieval",))
    assert gate["outcome"] == "inconclusive" and "absent from the training split" in gate["reason"]


def test_regression_gate_is_inconclusive_while_classes_lack_test_steps(contract):
    references = {"mae_training_mean": 0.3, "mae_zero": 0.25}
    assert regression_gate(0.05, references, contract, ["inference"])["outcome"] == "inconclusive"
