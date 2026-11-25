"""The contract's gates over synthetic optimization experiments (O06). Never recorded evidence.

Experiments come from `optimization_fixtures`; each task also gets a run journal whose root step
lasts as the test declares, for the baseline stability gate.
"""

import dataclasses
import json
from pathlib import Path

import pytest
from agentwatch_data.extract import canonical_sha256
from agentwatch_data.optimization_gates import (
    FAIL,
    INCONCLUSIVE,
    NOT_EVALUATED,
    PASS,
    GateInputError,
    candidate_outcome,
    evaluate,
    improvement_gates,
    load_thresholds,
    main,
    promotion,
    root_step_seconds,
)
from agentwatch_data.optimization_manifest import CANDIDATE
from optimization_fixtures import (
    CONTRACT_PATH,
    HARDWARE,
    MODEL,
    SECOND_NS,
    TASKS,
    contract,
    default_timing,
    manifest_document,
    write_experiment,
)

CONTRACT = contract()
HOST = HARDWARE["host_id"]


def journal(root: Path, record: dict, seconds) -> None:
    """A root-step event per task: `seconds(task_id, slot_index)` long, or None for no journal."""
    for task in record["tasks"]:
        duration = seconds(task["task_id"], record["index"])
        if duration is None:
            continue
        line = {
            "kind": "record",
            "payload": {
                "record_type": "step_event",
                "run_id": task["run_id"],
                "step_id": "agent-run",
                "sequence": 3,
                "submitted_tick_ns": 0,
                "finished_tick_ns": int(duration * SECOND_NS),
            },
        }
        path = root / task["evidence_dir"] / "live-journal.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not json\n" + json.dumps(line) + "\n")


def experiment(root, *, seconds=lambda task, index: 9.0, change=None, timing=default_timing):
    def edit(slot, record):
        record = change(slot, record) if change is not None else record
        if record is not None:
            journal(root, record, seconds)
        return record

    return write_experiment(root, CONTRACT, timing=timing, change=edit)


def by_condition(report, condition="contended", candidate="inference-1"):
    entry = next(item for item in report["candidates"] if item["candidate"] == candidate)
    conditions = {item["condition"]: item for item in entry["conditions"]}
    return entry, {item["gate"]: item for item in conditions[condition]["gates"]}


def test_every_measured_gate_passes_but_unmeasured_overhead_keeps_it_inconclusive(tmp_path):
    experiment(tmp_path)
    report = evaluate(tmp_path, CONTRACT_PATH)
    entry, gates = by_condition(report)
    assert {name: item["status"] for name, item in gates.items()} == dict.fromkeys(
        [
            "improvement",
            "improvement_uncertainty",
            "throughput",
            "tail_latency",
            "starvation_and_leaks",
            "baseline_stability",
            "failures_and_timeouts",
            "quality",
        ],
        PASS,
    )
    assert gates["improvement"]["observed"] == pytest.approx(0.2)
    _entry, clean = by_condition(report, "clean")
    # Improvement is gated under contention only.
    assert "improvement" not in clean and clean["throughput"]["status"] == PASS
    shared = {item["gate"]: item["status"] for item in report["experiment_gates"]}
    assert shared == {
        "admission_overhead": NOT_EVALUATED,
        "instrumentation_overhead": NOT_EVALUATED,
    }
    assert entry["outcome"] == INCONCLUSIVE
    assert "experiment: admission_overhead not_evaluated" in entry["reasons"]
    assert report["promotion"]["promoted"] is None
    assert "proposed" in report["promotion"]["reason"]
    assert report["thresholds"]["min_median_batch_time_reduction"] == 0.10


def test_a_reduction_below_the_threshold_fails_the_candidate(tmp_path):
    def timing(slot, task_id):
        submitted, finished = default_timing(slot, task_id)
        if slot.configuration == CANDIDATE and slot.condition == "contended":
            # 9.5 s against 10 s: a 5% reduction, consistently, so the interval excludes zero.
            finished = finished + int(1.5 * SECOND_NS)
        return submitted, finished

    experiment(tmp_path, timing=timing)
    entry, gates = by_condition(evaluate(tmp_path, CONTRACT_PATH))
    assert gates["improvement"]["status"] == FAIL
    assert gates["improvement"]["observed"] == pytest.approx(0.05)
    assert gates["improvement_uncertainty"]["status"] == PASS
    assert entry["outcome"] == FAIL and "contended: improvement" in entry["reasons"]


def test_an_interval_that_includes_zero_is_inconclusive(tmp_path):
    def timing(slot, task_id):
        submitted, finished = default_timing(slot, task_id)
        # Half the pairs' candidates are slower than their baselines.
        if (
            slot.configuration == CANDIDATE
            and slot.condition == "contended"
            and slot.pair_index % 2
        ):
            finished += 4 * SECOND_NS
        return submitted, finished

    experiment(tmp_path, timing=timing)
    _entry, gates = by_condition(evaluate(tmp_path, CONTRACT_PATH))
    assert gates["improvement_uncertainty"]["status"] == INCONCLUSIVE
    assert "includes zero" in gates["improvement_uncertainty"]["reason"]


def test_an_unstable_baseline_makes_every_comparison_on_it_inconclusive(tmp_path):
    # Root steps alternate between 5 s and 15 s across baseline batches: CV far above 15%.
    experiment(tmp_path, seconds=lambda task, index: 5.0 if index % 2 else 15.0)
    entry, gates = by_condition(evaluate(tmp_path, CONTRACT_PATH))
    assert gates["baseline_stability"]["status"] == FAIL
    assert entry["outcome"] == INCONCLUSIVE
    assert any("baseline fail" in reason for reason in entry["reasons"])


def test_missing_journals_leave_baseline_stability_undecided(tmp_path):
    experiment(tmp_path, seconds=lambda task, index: None if task == TASKS[0][0] else 9.0)
    _entry, gates = by_condition(evaluate(tmp_path, CONTRACT_PATH))
    assert gates["baseline_stability"]["status"] == INCONCLUSIVE
    assert "unreadable" in gates["baseline_stability"]["reason"]


def test_a_new_failure_fails_the_guardrail(tmp_path):
    def change(slot, record):
        if slot.configuration == CANDIDATE and slot.condition == "clean" and slot.pair_index == 0:
            record["tasks"][0].update(state="failed", acceptance="failed")
        return record

    experiment(tmp_path, change=change)
    entry, gates = by_condition(evaluate(tmp_path, CONTRACT_PATH), "clean")
    assert gates["failures_and_timeouts"]["status"] == FAIL
    assert gates["quality"]["status"] == FAIL
    # The failed task also leaves too few completed tasks, but more evidence cannot undo it.
    clean = next(item for item in entry["conditions"] if item["condition"] == "clean")
    assert any("completed candidate tasks" in reason for reason in clean["insufficient_reasons"])
    assert entry["outcome"] == FAIL
    assert entry["reasons"] == ["clean: failures_and_timeouts", "clean: quality"]


def test_missing_evidence_keeps_the_baseline(tmp_path):
    def change(slot, record):
        return None if slot.index == 3 else record

    experiment(tmp_path, change=change)
    entry, _gates = by_condition(evaluate(tmp_path, CONTRACT_PATH))
    assert entry["outcome"] == INCONCLUSIVE
    assert any("scheduled batches have no record" in reason for reason in entry["reasons"])


LIMITS = load_thresholds(CONTRACT_PATH, CONTRACT)


def verdict_document(*groups, contract_sha256=None, budget=None) -> dict:
    """An overhead verdict report as `overhead_verdict` writes it, judged under this contract."""
    return {
        "contract": {"sha256": (contract_sha256 or CONTRACT.sha256)[:8]},
        "budget": budget
        or {
            "max_median_instrumentation_overhead": LIMITS.max_median_instrumentation_overhead,
            "max_median_admission_overhead": LIMITS.max_median_admission_overhead,
        },
        "groups": list(groups),
    }


def overhead_verdict(tmp_path, *groups):
    path = tmp_path / "overhead-verdict.json"
    path.write_text(json.dumps(verdict_document(*groups)))
    return path


@pytest.mark.parametrize(
    ("groups", "status"),
    [
        (
            [
                {
                    "settings": {"model": MODEL["name"], "host_label": HOST},
                    "verdict": "within_budget",
                }
            ],
            PASS,
        ),
        (
            [{"settings": {"model": MODEL["name"], "host_label": HOST}, "verdict": "over_budget"}],
            FAIL,
        ),
        (
            [{"settings": {"model": MODEL["name"], "host_label": HOST}, "verdict": "inconclusive"}],
            INCONCLUSIVE,
        ),
        (
            [
                {
                    "settings": {"model": "another-model", "host_label": HOST},
                    "verdict": "within_budget",
                }
            ],
            NOT_EVALUATED,
        ),
        (
            [{"settings": {"model": MODEL["name"], "host_label": HOST}, "verdict": "within_budget"}]
            * 2,
            NOT_EVALUATED,
        ),
    ],
)
def test_the_overhead_verdict_for_this_model_is_the_instrumentation_gate(tmp_path, groups, status):
    experiment(tmp_path / "experiment")
    report = evaluate(tmp_path / "experiment", CONTRACT_PATH, overhead_verdict(tmp_path, *groups))
    shared = {item["gate"]: item for item in report["experiment_gates"]}
    assert shared["instrumentation_overhead"]["status"] == status


def test_an_unknown_overhead_verdict_is_refused(tmp_path):
    experiment(tmp_path / "experiment")
    verdict = overhead_verdict(
        tmp_path, {"settings": {"model": MODEL["name"], "host_label": HOST}, "verdict": "ok"}
    )
    with pytest.raises(GateInputError, match="Unknown overhead verdict"):
        evaluate(tmp_path / "experiment", CONTRACT_PATH, verdict)


def test_thresholds_come_from_the_compared_contract_file(tmp_path):
    limits = load_thresholds(CONTRACT_PATH, CONTRACT)
    assert (limits.min_throughput_ratio, limits.max_p95_task_latency_ratio) == (0.95, 1.05)
    assert limits.improvement_conditions == ("contended",)
    edited = tmp_path / "contract.toml"
    edited.write_text(
        CONTRACT_PATH.read_text().replace(
            "min_throughput_ratio = 0.95", "min_throughput_ratio = 0.5"
        )
    )
    with pytest.raises(GateInputError, match="contract changed"):
        load_thresholds(edited, CONTRACT)


def passing(name, seconds, limits):
    return {
        "candidate": name,
        "limits": limits,
        "outcome": PASS,
        "contended_median_batch_seconds": seconds,
    }


def test_at_most_one_candidate_is_promoted_and_only_under_an_accepted_contract():
    candidates = [
        passing("inference-2", 8.0, {"inference_requests": 2}),
        passing("inference-1", 8.0, {"inference_requests": 1}),
        passing("test-jobs-1", 9.0, {"test_jobs": 1}),
        {**passing("inference-4", 7.0, {"inference_requests": 4}), "outcome": FAIL},
    ]
    accepted = dataclasses.replace(CONTRACT, status="accepted")
    result = promotion(candidates, accepted)
    # The fastest passing candidate; the smaller limit wins the tie. A failed one never does.
    assert result["promoted"] == "inference-1" and result["candidates_tried"] == 4
    assert promotion(candidates, CONTRACT)["promoted"] is None
    assert promotion([candidates[3]], accepted)["promoted"] is None


def test_an_insufficient_comparison_outranks_a_failed_gate():
    failed = {"gate": "throughput", "status": FAIL}
    conditions = [
        {"condition": "clean", "insufficient_reasons": ["4 complete pairs"], "gates": [failed]}
    ]
    assert candidate_outcome(conditions, []) == (INCONCLUSIVE, ["clean: 4 complete pairs"])


def test_the_command_writes_the_report_and_refuses_a_bad_manifest(tmp_path, capsys):
    experiment(tmp_path / "experiment")
    output = tmp_path / "reports"
    assert (
        main(
            [
                str(tmp_path / "experiment"),
                "--contract",
                str(CONTRACT_PATH),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    written = json.loads((output / "gates.json").read_text())
    assert written["method"] == "opt-gates-v1"
    assert "inference-1: inconclusive" in capsys.readouterr().out
    (tmp_path / "experiment" / "manifest.json").write_text("{}")
    assert main([str(tmp_path / "experiment"), "--contract", str(CONTRACT_PATH)]) == 1


ADMISSION = {
    "settings": {"protocol": "admission-overhead-v2", "model": MODEL["name"], "host_label": HOST},
    "verdict": "within_budget",
}
INSTRUMENTATION = {
    "settings": {
        "protocol": "resource-sampling-overhead-v2",
        "model": MODEL["name"],
        "host_label": HOST,
    },
    "verdict": "within_budget",
}


def write_verdict(path, *groups, **changes):
    path.write_text(json.dumps(verdict_document(*groups, **changes)))
    return path


def test_with_both_overhead_verdicts_every_gate_can_pass(tmp_path):
    experiment(tmp_path / "experiment")
    report = evaluate(
        tmp_path / "experiment",
        CONTRACT_PATH,
        overhead_verdict=write_verdict(tmp_path / "a07.json", INSTRUMENTATION),
        admission_verdict=write_verdict(tmp_path / "admission.json", ADMISSION),
    )
    shared = {item["gate"]: item["status"] for item in report["experiment_gates"]}
    assert shared == {"admission_overhead": PASS, "instrumentation_overhead": PASS}
    entry, _gates = by_condition(report)
    assert (entry["outcome"], entry["reasons"]) == (PASS, [])
    # A pass under a proposed contract still promotes nothing.
    assert report["promotion"]["promoted"] is None


def test_an_admission_overhead_v1_verdict_judges_neither_gate(tmp_path):
    # v1 paced its two conditions differently, so its verdict cannot judge the admission gate,
    # and no admission protocol ever stands in for instrumentation overhead.
    experiment(tmp_path / "experiment")
    v1 = ADMISSION | {"settings": ADMISSION["settings"] | {"protocol": "admission-overhead-v1"}}
    report = evaluate(
        tmp_path / "experiment",
        CONTRACT_PATH,
        overhead_verdict=write_verdict(tmp_path / "a07.json", v1),
        admission_verdict=write_verdict(tmp_path / "admission.json", v1),
    )
    shared = {item["gate"]: item for item in report["experiment_gates"]}
    for name in ("admission_overhead", "instrumentation_overhead"):
        assert shared[name]["status"] == NOT_EVALUATED
        assert "0 matching groups" in shared[name]["reason"]


def test_a_verdict_given_for_the_other_gate_is_not_used(tmp_path):
    experiment(tmp_path / "experiment")
    # Each file holds only the other protocol's group, so neither gate finds its own.
    report = evaluate(
        tmp_path / "experiment",
        CONTRACT_PATH,
        overhead_verdict=write_verdict(tmp_path / "a07.json", ADMISSION),
        admission_verdict=write_verdict(tmp_path / "admission.json", INSTRUMENTATION),
    )
    shared = {item["gate"]: item for item in report["experiment_gates"]}
    for name in ("admission_overhead", "instrumentation_overhead"):
        assert shared[name]["status"] == NOT_EVALUATED
        assert "0 matching groups" in shared[name]["reason"]


def test_an_over_budget_admission_verdict_fails_the_candidate(tmp_path):
    experiment(tmp_path / "experiment")
    over = ADMISSION | {"verdict": "over_budget"}
    report = evaluate(
        tmp_path / "experiment",
        CONTRACT_PATH,
        admission_verdict=write_verdict(tmp_path / "admission.json", over),
    )
    entry, _gates = by_condition(report)
    assert entry["outcome"] == FAIL and "experiment: admission_overhead" in entry["reasons"]


def test_a_verdict_measured_on_another_host_is_not_used(tmp_path):
    # Overhead depends on the host: a CPU-container measurement of the same model name must
    # not stand in for the experiment's GPU host.
    experiment(tmp_path / "experiment")
    elsewhere = {"host_label": "some-other-host"}
    report = evaluate(
        tmp_path / "experiment",
        CONTRACT_PATH,
        overhead_verdict=write_verdict(
            tmp_path / "a07.json",
            INSTRUMENTATION | {"settings": INSTRUMENTATION["settings"] | elsewhere},
        ),
        admission_verdict=write_verdict(
            tmp_path / "admission.json",
            ADMISSION | {"settings": ADMISSION["settings"] | elsewhere},
        ),
    )
    shared = {item["gate"]: item for item in report["experiment_gates"]}
    for name in ("admission_overhead", "instrumentation_overhead"):
        assert shared[name]["status"] == NOT_EVALUATED
        assert (
            f"0 matching groups for model {MODEL['name']} on host {HOST}"
            in (shared[name]["reason"])
        )
        assert shared[name]["source"]["settings"] is None
    entry, _gates = by_condition(report)
    assert entry["outcome"] != PASS


def test_each_overhead_gate_records_the_verdict_and_group_it_used(tmp_path):
    experiment(tmp_path / "experiment")
    admission = write_verdict(tmp_path / "admission.json", ADMISSION)
    report = evaluate(tmp_path / "experiment", CONTRACT_PATH, admission_verdict=admission)
    shared = {item["gate"]: item for item in report["experiment_gates"]}
    source = shared["admission_overhead"]["source"]
    assert source["settings"] == ADMISSION["settings"]
    assert source["verdict_sha256"] == canonical_sha256(json.loads(admission.read_text()))
    assert "source" not in shared["instrumentation_overhead"]  # no verdict was given


def test_root_step_duration_comes_from_the_tasks_own_run(tmp_path):
    # A journal holding a second attempt's root step, at a higher sequence, as an interrupted
    # and rerun batch could leave it: only the recorded run's own event counts.
    path = tmp_path / "run" / "live-journal.jsonl"
    path.parent.mkdir()
    own, other = "11111111-1111-4111-8111-111111111111", "22222222-2222-4222-8222-222222222222"
    events = [
        {"run_id": own, "sequence": 3, "finished_tick_ns": 9 * SECOND_NS},
        {"run_id": other, "sequence": 4, "finished_tick_ns": 30 * SECOND_NS},
    ]
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "kind": "record",
                    "payload": {
                        "record_type": "step_event",
                        "step_id": "agent-run",
                        "submitted_tick_ns": 0,
                        **event,
                    },
                }
            )
            for event in events
        )
        + "\n"
    )
    assert root_step_seconds(tmp_path, "run", own) == 9.0
    assert root_step_seconds(tmp_path, "run", None) is None


def test_an_interval_below_zero_fails_the_uncertainty_gate():
    limits = load_thresholds(CONTRACT_PATH, CONTRACT)
    slower = {
        "median_batch_time_reduction": -0.5,
        "median_batch_time_reduction_interval": {"low": -0.6, "high": -0.4},
    }
    improvement, uncertainty = improvement_gates(slower, limits)
    assert (improvement["status"], uncertainty["status"]) == (FAIL, FAIL)
    assert "below zero" in uncertainty["reason"]
    faster = {
        "median_batch_time_reduction": 0.3,
        "median_batch_time_reduction_interval": {"low": 0.2, "high": 0.4},
    }
    assert improvement_gates(faster, limits)[1]["status"] == PASS


def test_a_verdict_judged_under_another_contract_is_not_used(tmp_path):
    experiment(tmp_path / "experiment")
    stale = write_verdict(tmp_path / "admission.json", ADMISSION, contract_sha256="0" * 64)
    report = evaluate(tmp_path / "experiment", CONTRACT_PATH, admission_verdict=stale)
    shared = {item["gate"]: item for item in report["experiment_gates"]}
    assert shared["admission_overhead"]["status"] == NOT_EVALUATED
    assert "judged under contract '00000000'" in shared["admission_overhead"]["reason"]


@pytest.mark.parametrize("budget_in", ["group", "report"])
def test_a_verdict_judged_against_another_budget_is_not_used(tmp_path, budget_in):
    experiment(tmp_path / "experiment")
    if budget_in == "group":
        path = write_verdict(tmp_path / "admission.json", ADMISSION | {"budget": 0.2})
    else:
        looser = {"max_median_admission_overhead": 0.2, "max_median_instrumentation_overhead": 0.2}
        path = write_verdict(tmp_path / "admission.json", ADMISSION, budget=looser)
    report = evaluate(tmp_path / "experiment", CONTRACT_PATH, admission_verdict=path)
    shared = {item["gate"]: item for item in report["experiment_gates"]}
    assert shared["admission_overhead"]["status"] == NOT_EVALUATED
    assert "budget of 0.2" in shared["admission_overhead"]["reason"]
    assert shared["admission_overhead"]["source"]["budget"] == 0.2


def test_the_overhead_gate_records_the_task_and_budget_it_used(tmp_path):
    experiment(tmp_path / "experiment")
    group = ADMISSION | {"settings": ADMISSION["settings"] | {"task_id": "fix-mean-empty-r1"}}
    path = write_verdict(tmp_path / "admission.json", group)
    report = evaluate(tmp_path / "experiment", CONTRACT_PATH, admission_verdict=path)
    source = {item["gate"]: item for item in report["experiment_gates"]}["admission_overhead"][
        "source"
    ]
    assert source["task_id"] == "fix-mean-empty-r1"
    assert source["budget"] == LIMITS.max_median_admission_overhead
    assert source["contract_sha256"] == CONTRACT.sha256[:8]


def test_a_journal_line_that_is_not_an_object_is_skipped(tmp_path):
    run_id = "11111111-1111-4111-8111-111111111111"
    path = tmp_path / "run" / "live-journal.jsonl"
    path.parent.mkdir()
    event = {
        "kind": "record",
        "payload": {
            "record_type": "step_event",
            "step_id": "agent-run",
            "run_id": run_id,
            "sequence": 3,
            "submitted_tick_ns": 0,
            "finished_tick_ns": 4 * SECOND_NS,
        },
    }
    path.write_text("[1, 2]\n" + json.dumps(event) + "\n")
    assert root_step_seconds(tmp_path, "run", run_id) == 4.0


def test_a_slower_clean_candidate_fails_throughput_and_tail_latency(tmp_path):
    def timing(slot, task_id):
        submitted, finished = default_timing(slot, task_id)
        if slot.configuration == CANDIDATE and slot.condition == "clean":
            finished += 3 * SECOND_NS  # 13 s against 10 s
        return submitted, finished

    experiment(tmp_path, timing=timing)
    entry, gates = by_condition(evaluate(tmp_path, CONTRACT_PATH), "clean")
    assert gates["throughput"]["status"] == FAIL
    assert gates["tail_latency"]["status"] == FAIL
    assert entry["outcome"] == FAIL
    assert {"clean: throughput", "clean: tail_latency"} <= set(entry["reasons"])


@pytest.mark.parametrize(
    ("field", "reason"),
    [("starved_admissions", "waited past its bound"), ("leaked_permits", "held after its batch")],
)
def test_starvation_or_a_leaked_permit_fails_the_candidate(tmp_path, field, reason):
    def change(slot, record):
        if slot.configuration == CANDIDATE and slot.condition == "clean" and slot.pair_index == 0:
            record["admission"][field] = 1
        return record

    experiment(tmp_path, change=change)
    entry, gates = by_condition(evaluate(tmp_path, CONTRACT_PATH), "clean")
    assert gates["starvation_and_leaks"]["status"] == FAIL
    assert reason in gates["starvation_and_leaks"]["reason"]
    assert entry["outcome"] == FAIL


def test_a_new_timeout_fails_the_guardrail(tmp_path):
    def change(slot, record):
        if slot.configuration == CANDIDATE and slot.condition == "clean" and slot.pair_index == 0:
            record["tasks"][0].update(state="timed_out", acceptance=None)
        return record

    experiment(tmp_path, change=change)
    entry, gates = by_condition(evaluate(tmp_path, CONTRACT_PATH), "clean")
    assert gates["failures_and_timeouts"]["status"] == FAIL
    assert gates["failures_and_timeouts"]["observed"]["new_timeouts"] == 1
    assert entry["outcome"] == FAIL


def test_unknown_acceptance_leaves_quality_inconclusive(tmp_path):
    def change(slot, record):
        if slot.configuration == CANDIDATE and slot.condition == "clean" and slot.pair_index == 0:
            record["tasks"][0]["acceptance"] = None
        return record

    experiment(tmp_path, change=change)
    _entry, gates = by_condition(evaluate(tmp_path, CONTRACT_PATH), "clean")
    assert gates["quality"]["status"] == INCONCLUSIVE
    assert "no recorded acceptance" in gates["quality"]["reason"]


def test_a_development_experiment_is_never_gated_to_a_pass(tmp_path):
    document = manifest_document(CONTRACT, purpose="development")
    del document["workload"]["development_tasks"]
    write_experiment(
        tmp_path,
        CONTRACT,
        document,
        change=lambda slot, record: journal(tmp_path, record, lambda t, i: 9.0) or record,
    )
    report = evaluate(tmp_path, CONTRACT_PATH)
    entry, _gates = by_condition(report)
    assert report["purpose"] == "development" and entry["outcome"] == INCONCLUSIVE
    assert any("never gated" in reason for reason in entry["reasons"])
