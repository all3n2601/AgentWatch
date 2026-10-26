import hashlib
import json
import tomllib
from dataclasses import replace

import pytest
from agentwatch_model.contract import DEFAULT_CONTRACT, ContractError, load_contract
from agentwatch_model.report import NOT_EVALUATED_FOR_RULES, blocked_report, build_report
from agentwatch_model.rules import DEFAULT_RULES, load_rules
from rule_fixtures import row


@pytest.fixture(scope="module")
def rules():
    return load_rules(DEFAULT_RULES)


@pytest.fixture(scope="module")
def contract():
    # Small minimums so a handful of synthetic rows can reach a decision.
    return replace(load_contract(DEFAULT_CONTRACT), min_test_steps_per_class=1, min_clean_steps=4)


def predict(rules, rows):
    return [rules.predict(r) for r in rows]


def report_for(rules, contract, test, train=(), evaluation=True):
    return build_report(
        contract,
        rules,
        list(train),
        test,
        predict(rules, test),
        evaluation=evaluation,
        resamples=100,
        seed=3,
    )


def test_contract_is_identified_by_its_parsed_contents():
    contract = load_contract(DEFAULT_CONTRACT)
    parsed = tomllib.loads(DEFAULT_CONTRACT.read_text())
    canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert contract.hash == hashlib.sha256(canonical.encode()).hexdigest()[:8]
    assert contract.version == parsed["version"] and contract.status == parsed["status"]
    assert contract.classes == ("none", "cpu_sandbox", "inference", "retrieval")
    assert contract.min_none_rate_clean == 0.95


def test_a_contract_missing_a_required_value_is_rejected(tmp_path):
    broken = tmp_path / "contract.toml"
    broken.write_text(DEFAULT_CONTRACT.read_text().replace("min_none_rate_clean", "renamed"))
    with pytest.raises(ContractError, match="min_none_rate_clean"):
        load_contract(broken)


def test_clean_steps_predicted_none_pass_the_clean_gate(rules, contract):
    test = [row(condition="clean") for _ in range(4)] + [
        row("cpu_sandbox", "contended", "b", queue_fraction=0.9)
    ]
    report = report_for(rules, contract, test)
    assert report["status"] == "evaluated" and report["evaluation"] is True
    gate = report["gates"]["clean_none_rate"]
    assert gate["outcome"] == "pass" and gate["value"] == 1.0 and gate["threshold"] == 0.95


def test_a_clean_step_blamed_on_cpu_fails_the_clean_gate(rules, contract):
    test = [row(condition="clean") for _ in range(3)] + [row(condition="clean", queue_fraction=0.9)]
    gate = report_for(rules, contract, test)["gates"]["clean_none_rate"]
    assert gate["outcome"] == "fail" and gate["value"] == 0.75


def test_too_few_clean_steps_make_the_clean_gates_inconclusive(rules, contract):
    report = report_for(rules, contract, [row(condition="clean") for _ in range(3)])
    assert report["gates"]["clean_none_rate"]["outcome"] == "inconclusive"
    assert report["gates"]["baseline_and_clean_none_rate"]["outcome"] == "inconclusive"


def test_the_baseline_and_clean_gate_records_both_populations(rules, contract):
    test = [row(condition="clean") for _ in range(3)] + [
        row(condition="baseline") for _ in range(2)
    ]
    gate = report_for(rules, contract, test)["gates"]["baseline_and_clean_none_rate"]
    # The rate covers all five steps; sufficiency is judged on the three clean ones only.
    assert gate["steps"] == 5 and gate["value"] == 1.0
    assert gate["clean_steps"] == 3 and gate["min_clean_steps"] == contract.min_clean_steps
    assert gate["outcome"] == "inconclusive"


def test_macro_f1_is_inconclusive_while_a_class_lacks_test_steps(rules, contract):
    test = [row(condition="clean") for _ in range(4)] + [
        row("cpu_sandbox", "contended", queue_fraction=0.9)
    ]
    macro = report_for(rules, contract, test)["metrics"]["macro_f1"]
    assert macro["status"] == "inconclusive"
    assert macro["classes_below_minimum"] == ["inference", "retrieval"]


def test_macro_f1_is_reported_once_every_class_has_enough_test_steps(rules, contract):
    test = [row(condition="clean") for _ in range(4)]
    test += [
        row(label, "contended", f"task-{label}")
        for label in ("cpu_sandbox", "inference", "retrieval")
    ]
    macro = report_for(rules, contract, test)["metrics"]["macro_f1"]
    assert macro["status"] == "reported" and macro["classes_below_minimum"] == []


def test_gates_that_do_not_apply_to_rules_are_never_reported_as_passed(rules, contract):
    gates = report_for(rules, contract, [row() for _ in range(4)])["gates"]
    for name, reason in NOT_EVALUATED_FOR_RULES.items():
        assert gates[name] == {"outcome": "not_evaluated", "reason": reason}
    assert {"mixed_contention_gain", "regression_error", "calibration"} <= set(
        NOT_EVALUATED_FOR_RULES
    )


def test_report_records_rules_contract_and_uncovered_classes(rules, contract):
    report = report_for(rules, contract, [row() for _ in range(4)], train=[row(excess=0.2)])
    assert report["rules"] == {
        "version": rules.version,
        "covered_classes": ["none", "cpu_sandbox"],
        "uncovered_classes": ["inference", "retrieval"],
    }
    assert report["contract"] == {
        "version": contract.version,
        "status": contract.status,
        "hash": contract.hash,
    }
    assert report["dataset_version"] == "steps-v1.2+synthetic"
    assert report["metrics"]["constant_references"]["training_mean"] == pytest.approx(0.2)


def test_exploratory_reports_keep_metrics_but_evaluate_no_gate(rules, contract):
    report = report_for(
        rules, contract, [row(condition="clean") for _ in range(4)], evaluation=False
    )
    assert report["status"] == "exploratory" and report["evaluation"] is False
    assert report["metrics"]["confusion_matrix"]["none"]["none"] == 4
    assert {gate["outcome"] for gate in report["gates"].values()} == {"not_evaluated"}


def test_a_blocked_split_reports_the_reason_and_no_metrics(rules, contract):
    report = blocked_report(contract, rules, "leakage check is failed")
    assert report["status"] == "blocked" and report["evaluation"] is False
    assert report["reason"] == "leakage check is failed"
    assert report["metrics"] is None and report["gates"] == {}
    assert report["summary"] == {
        "outcomes": {"pass": 0, "fail": 0, "inconclusive": 0, "not_evaluated": 0},
        "blocks_promotion": True,
    }


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("min_clean_test_steps = 100", 'min_clean_test_steps = "100"', "non-negative integer"),
        ("min_none_rate_clean = 0.95", "min_none_rate_clean = 1.5", "from 0 to 1"),
        ("max_mae = 0.10", "max_mae = true", "from 0 to 1"),
        ('classes = ["none"', 'classes = [1, "none"', "list of names"),
        ('"retrieval"]', '"gpu"]', "dataset's classes"),
        ('"retrieval"]', '"retrieval", "none"]', "dataset's classes"),
        ("[clean_runs]", "clean_runs = 1\n[unused]", r"\[clean_runs\] table"),
    ],
)
def test_contract_values_of_the_wrong_type_are_rejected(tmp_path, old, new, message):
    text = DEFAULT_CONTRACT.read_text()
    assert old in text
    broken = tmp_path / "contract.toml"
    broken.write_text(text.replace(old, new, 1))
    with pytest.raises(ContractError, match=message):
        load_contract(broken)


def test_four_class_macro_f1_stays_null_while_classes_are_missing(rules, contract):
    test = [row(condition="clean") for _ in range(4)] + [
        row("cpu_sandbox", "contended", queue_fraction=0.9)
    ]
    macro = report_for(rules, contract, test)["metrics"]["macro_f1"]
    assert macro["all_classes"] is None and macro["defined_classes"] == 1.0
    assert macro["interval_of"] == "defined_classes"


def test_mixed_contention_slice_is_reported_as_unavailable_with_its_reason(rules, contract):
    metrics = report_for(rules, contract, [row(condition="clean") for _ in range(4)])["metrics"]
    mixed = metrics["slices"]["mixed_contention"]
    assert mixed["status"] == "unavailable" and "mixed contention" in mixed["reason"]
    assert mixed["min_steps"] == contract.min_mixed_steps
    assert metrics["slices"]["by_condition"]["clean"]["steps"] == 4


def test_regression_references_carry_the_contract_limit(rules, contract):
    metrics = report_for(rules, contract, [row() for _ in range(4)], train=[row(excess=0.2)])[
        "metrics"
    ]
    assert metrics["constant_references"]["contract_max_mae"] == contract.max_mae


def test_any_gate_short_of_pass_blocks_promotion(rules, contract):
    report = report_for(rules, contract, [row(condition="clean") for _ in range(4)])
    outcomes = report["summary"]["outcomes"]
    assert outcomes["pass"] == 2 and outcomes["not_evaluated"] == len(NOT_EVALUATED_FOR_RULES)
    assert report["summary"]["blocks_promotion"] is True


def test_no_clean_steps_is_inconclusive_even_with_a_zero_minimum(rules, contract):
    lenient = replace(contract, min_clean_steps=0)
    gate = report_for(rules, lenient, [row("cpu_sandbox", "contended")])["gates"]["clean_none_rate"]
    assert gate["outcome"] == "inconclusive" and gate["value"] is None
