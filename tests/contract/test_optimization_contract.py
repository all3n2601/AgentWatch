"""opt-contract-v1 stays well-formed and in step with every document and gate it restates."""

import math
import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = tomllib.loads((ROOT / "pipelines/model/optimization-contract.toml").read_text())
EVALUATION = tomllib.loads((ROOT / "pipelines/model/evaluation-contract.toml").read_text())
DOC = (ROOT / "docs/optimization-contract.md").read_text()
PLAN = " ".join((ROOT / "docs/optimization-plan.md").read_text().split())
GATES = CONTRACT["gates"]
TEAMS = ("Agent & experiments", "Frontend & ML", "Data & backend")
ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def is_number(value) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def percent(share: float) -> str:
    return f"{share * 100:g}%"


def spoken_list(values: list[int]) -> str:
    return ", ".join(str(value) for value in values[:-1]) + f" and {values[-1]}"


def table_rows(text: str) -> dict[str, list[str]]:
    """Cells of every Markdown table row, keyed by the row's first cell."""
    rows = {}
    for line in text.splitlines():
        if line.startswith("|") and not line.startswith("|---"):
            cells = [" ".join(cell.split()) for cell in line.strip().strip("|").split("|")]
            rows[cells[0]] = cells
    return rows


ROWS = table_rows(DOC)


def test_version_and_status_are_declared():
    assert CONTRACT["version"] == "opt-contract-v1"
    assert CONTRACT["status"] in {"proposed", "accepted"}


def test_accepted_status_requires_every_team_to_have_accepted():
    accepted = [ROWS[team][2] != "" and ISO_DATE.fullmatch(ROWS[team][3]) for team in TEAMS]
    assert all(accepted) if CONTRACT["status"] == "accepted" else not any(accepted)


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
    # Imported here so a missing agent package fails this test, not collection of the module.
    from agentwatch_agent.experiment import STABLE_CV_MAX

    assert (
        GATES["max_median_instrumentation_overhead"]
        == EVALUATION["overhead"]["max_median_wall_time_increase"]
    )
    assert GATES["max_baseline_root_step_cv"] == STABLE_CV_MAX


REPETITIONS = CONTRACT["repetitions"]
WORKLOAD = CONTRACT["workload"]


@pytest.mark.parametrize(
    ("row", "text"),
    [
        ("Improvement", f"at least {percent(GATES['min_median_batch_time_reduction'])} lower"),
        ("Improvement uncertainty", f"{percent(GATES['interval_level'])} percentile paired"),
        ("Improvement uncertainty", f"{GATES['bootstrap_resamples']:,} resamples"),
        ("Throughput", f"at least {percent(GATES['min_throughput_ratio'])} of baseline"),
        ("Tail latency", f"at most {percent(GATES['max_p95_task_latency_ratio'])} of baseline"),
        ("Admission overhead", f"at most {percent(GATES['max_median_admission_overhead'])}"),
        (
            "Instrumentation overhead",
            f"at most {percent(GATES['max_median_instrumentation_overhead'])}",
        ),
        (
            "Baseline stability (`max_baseline_root_step_cv`)",
            f"at most {percent(GATES['max_baseline_root_step_cv'])}",
        ),
        ("Candidate limits", spoken_list(WORKLOAD["candidate_limits"])),
        (
            "Repetitions",
            (
                f"At least {REPETITIONS['min_paired_batches']} pairs and "
                f"{REPETITIONS['min_completed_tasks_per_configuration']} completed tasks"
            ),
        ),
        ("Minimum frozen evaluation tasks", f"{WORKLOAD['min_evaluation_tasks']}, a placeholder"),
    ],
)
def test_each_contract_page_row_states_the_toml_value(row, text):
    assert text in ROWS[row][1]


@pytest.mark.parametrize(
    "text",
    [
        f"at least {percent(GATES['min_median_batch_time_reduction'])} lower median batch",
        f"throughput at least {percent(GATES['min_throughput_ratio'])} of baseline",
        f"latency at most {percent(GATES['max_p95_task_latency_ratio'])} of baseline",
        f"coefficient of variation of at most {percent(GATES['max_baseline_root_step_cv'])}",
    ],
)
def test_the_plan_states_the_same_proposals(text):
    assert text in PLAN
