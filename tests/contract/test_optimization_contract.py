"""opt-contract-v1 stays well-formed and in step with every document and gate it restates."""

import math
import tomllib
from pathlib import Path

import pytest
from agentwatch_agent.experiment import STABLE_CV_MAX

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = tomllib.loads((ROOT / "pipelines/model/optimization-contract.toml").read_text())
EVALUATION = tomllib.loads((ROOT / "pipelines/model/evaluation-contract.toml").read_text())
DOC = (ROOT / "docs/optimization-contract.md").read_text()
PLAN = (ROOT / "docs/optimization-plan.md").read_text()
GATES = CONTRACT["gates"]


def is_number(value) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def test_version_and_status_are_declared():
    assert CONTRACT["version"] == "opt-contract-v1"
    assert CONTRACT["status"] in {"proposed", "accepted"}


@pytest.mark.parametrize(
    "key",
    [
        "min_median_batch_time_reduction",
        "min_throughput_ratio",
        "max_median_admission_overhead",
        "max_median_instrumentation_overhead",
        "max_baseline_root_step_cv",
        "interval_level",
    ],
)
def test_share_thresholds_are_numbers_from_zero_to_one(key):
    assert is_number(GATES[key]) and 0 <= GATES[key] <= 1, key


def test_ratio_and_count_thresholds_are_well_formed():
    assert (
        is_number(GATES["max_p95_task_latency_ratio"]) and GATES["max_p95_task_latency_ratio"] >= 1
    )
    for key in (
        "max_new_failures",
        "max_new_timeouts",
        "max_tasks_newly_failing_acceptance",
        "max_starved_admissions",
        "max_leaked_permits",
    ):
        assert type(GATES[key]) is int and GATES[key] >= 0, key
    assert type(GATES["bootstrap_resamples"]) is int and GATES["bootstrap_resamples"] > 0


def test_limits_are_positive_integers_and_unlimited_is_an_absent_key():
    policy = CONTRACT["policy"]
    assert type(policy["min_limit"]) is int and policy["min_limit"] >= 1
    assert policy["baseline"] == "unlimited" and policy["unlimited_encoding"] == "absent"
    candidates = CONTRACT["workload"]["candidate_limits"]
    assert candidates and all(type(c) is int and c >= policy["min_limit"] for c in candidates)


def test_repetitions_and_outcomes_are_well_formed():
    repetitions = CONTRACT["repetitions"]
    assert type(repetitions["min_paired_batches"]) is int and repetitions["min_paired_batches"] >= 2
    assert type(repetitions["min_completed_tasks_per_configuration"]) is int
    assert CONTRACT["outcomes"]["values"] == ["pass", "fail", "inconclusive", "not_evaluated"]
    assert CONTRACT["outcomes"]["max_promoted_candidates"] == 1


def test_shared_gates_match_the_evaluation_contract_and_the_agent():
    assert (
        GATES["max_median_instrumentation_overhead"]
        == EVALUATION["overhead"]["max_median_wall_time_increase"]
    )
    assert GATES["max_baseline_root_step_cv"] == STABLE_CV_MAX


@pytest.mark.parametrize(
    "text",
    [
        "at least 10% lower than baseline",
        "at least 95% of baseline",
        "at most 105% of baseline",
        "Median wall-time increase at most 5%",
        "at most 15%",
        "1, 2 and 4",
        "At least 5 pairs and 20 completed tasks",
        "10,000 resamples",
    ],
)
def test_the_contract_page_states_each_threshold(text):
    assert text in DOC


@pytest.mark.parametrize(
    "text",
    [
        "at least 10% lower median batch",
        "at least 95% of baseline",
        "at most 105% of baseline",
        "less than 15%",
    ],
)
def test_the_plan_states_the_same_proposals(text):
    assert text in PLAN


def test_numbers_in_the_toml_match_the_restated_values():
    assert GATES["min_median_batch_time_reduction"] == 0.10
    assert GATES["min_throughput_ratio"] == 0.95
    assert GATES["max_p95_task_latency_ratio"] == 1.05
    assert GATES["max_median_admission_overhead"] == 0.05
    assert GATES["bootstrap_resamples"] == 10_000
    assert CONTRACT["workload"]["candidate_limits"] == [1, 2, 4]
    assert CONTRACT["repetitions"]["min_paired_batches"] == 5
    assert CONTRACT["repetitions"]["min_completed_tasks_per_configuration"] == 20
