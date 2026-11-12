"""Held-out report measures, checked against hand-counted confusions on synthetic rows.

The rows and scores here are made up for these tests; none of it is recorded evidence.
"""

import pytest
from agentwatch_model.held_out_model import CLASSIFIER_SCORER
from agentwatch_model.held_out_report import (
    LIMITATIONS,
    NO_PLAN_REASON,
    build_report,
    measures,
)
from agentwatch_model.held_out_scorers import RULES_SCORER, ScoredStep


def step(truth, batch="b1", task="fix-mean-empty-r1", **overrides):
    base = {
        "task_id": task,
        "configuration": "cfg-default",
        "n": 3,
        "phase": "live",
        "step_type": "model_call",
        "batch_id": batch,
        "truths": {"queue": truth, "slower": truth, "protocol_clean": None},
        "truth_reasons": {"queue": None, "slower": None, "protocol_clean": "not_applicable"},
    }
    return base | overrides


def scored(label, abstained=False, scorer=CLASSIFIER_SCORER):
    return ScoredStep(scorer, label, None if scorer == RULES_SCORER else 0.9, abstained)


def run_measures(rows, scores, truth="queue"):
    return measures(rows, scores, truth, resamples=200, seed=0)


def test_measures_match_a_hand_counted_confusion():
    rows = [step("inference")] * 3 + [step("none")] * 3
    scores = [
        scored("inference"),  # caught
        scored("none"),  # missed
        scored("inference", abstained=True),  # abstained: counted as missed
        scored("none"),
        scored("none"),
        scored("cpu_sandbox"),  # false alarm
    ]
    result = run_measures(rows, scores)
    assert result["steps"] == 6
    assert result["inference_steps"] == 3 and result["none_steps"] == 3
    assert result["miss_rate"] == pytest.approx(2 / 3)
    assert result["answered_miss_rate"] == pytest.approx(1 / 2)
    assert result["false_alarm_rate"] == pytest.approx(1 / 3)
    assert result["answered_false_alarm_rate"] == pytest.approx(1 / 3)
    assert result["abstention_rate"] == pytest.approx(1 / 6)
    assert result["confusion"] == {
        "inference": {"inference": 1, "none": 1, "abstained": 1},
        "none": {"none": 2, "cpu_sandbox": 1},
    }


def test_zero_denominators_are_none_not_zero():
    result = run_measures([step("none")], [scored("none")])
    assert result["inference_steps"] == 0
    assert result["miss_rate"] is None and result["answered_miss_rate"] is None
    assert result["false_alarm_rate"] == 0.0
    only_inference = run_measures([step("inference")], [scored("none", abstained=True)])
    assert only_inference["none_steps"] == 0
    assert only_inference["false_alarm_rate"] is None
    assert only_inference["answered_miss_rate"] is None  # nothing answered
    assert only_inference["miss_rate"] == 1.0


def test_steps_without_a_truth_are_left_out():
    rows = [step("inference"), step(None)]
    result = run_measures(rows, [scored("none"), scored("inference")])
    assert result["steps"] == 1
    assert result["confusion"] == {"inference": {"none": 1}}
    assert run_measures(rows, [scored("none")] * 2, truth="protocol_clean")["steps"] == 0


def test_intervals_group_by_batch():
    # Batch b1 always misses and b2 always catches, so any draw of whole batches gives a miss
    # rate of 0, 0.5 or 1; steps drawn independently would give values in between.
    rows = [step("inference", batch="b1")] * 4 + [step("inference", batch="b2")] * 4
    scores = [scored("none")] * 4 + [scored("inference")] * 4
    result = run_measures(rows, scores)
    interval = result["intervals"]["miss_rate"]
    assert interval["groups"] == 2 and interval["valid_resamples"] == 200
    assert interval["low"] == 0.0 and interval["high"] == 1.0
    assert result["miss_rate"] == 0.5
    assert result["intervals"]["false_alarm_rate"]["valid_resamples"] == 0


def test_a_cell_with_one_batch_has_no_interval():
    result = run_measures([step("inference")] * 3, [scored("none")] * 3)
    interval = result["intervals"]["miss_rate"]
    assert interval["low"] is None and interval["groups"] == 1


def test_interval_groups_count_only_batches_with_the_rates_denominator():
    rows = [step("inference", batch="b1"), step("none", batch="b2")]
    result = run_measures(rows, [scored("none"), scored("none")])
    miss = result["intervals"]["miss_rate"]
    assert miss["groups"] == 1 and miss["low"] is None and miss["high"] is None
    alarm = result["intervals"]["false_alarm_rate"]
    assert alarm["groups"] == 1 and alarm["low"] is None


def context(**overrides):
    return {
        "seen_task_groups": {"fix-mean-empty"},
        "refused": {"feature_timing_mismatch": 2},
        "summaries": [
            {
                "configuration": "cfg-default",
                "protocol_sha256": "a" * 64,
                "runs_without_plan": 4,
                "runs_without_evidence": 1,
                "excluded_batches": ["ramp-1"],
            }
        ],
        "unjudged": [],
        "model": {"version": "training-v1"},
        "resamples": 50,
        "seed": 0,
        "abstention_unavailable": False,
    } | overrides


def rules_scores(count, label="none"):
    return {RULES_SCORER: [scored(label, scorer=RULES_SCORER)] * count}


def test_unscored_steps_are_counted_by_reason():
    reasons = {
        "queue": "span_not_measured",
        "slower": "no_baseline",
        "protocol_clean": "not_applicable",
    }
    rows = [step("inference"), step(None, truth_reasons=reasons), step(None, truth_reasons=reasons)]
    report = build_report(rows, rules_scores(3), context())
    steps = report["unscored"]["steps"]
    assert steps["queue"] == {"span_not_measured": 2}
    assert steps["slower"] == {"no_baseline": 2}
    assert steps["protocol_clean"] == {"not_applicable": 3}  # one from the scored row's default
    runs = report["unscored"]["runs"]["cfg-default"]
    assert runs["runs_without_plan"] == {"count": 4, "reason": NO_PLAN_REASON}
    assert runs["runs_without_evidence"] == 1 and runs["excluded_batches"] == 1
    assert report["refused"] == {"feature_timing_mismatch": 2}


def test_unjudged_steps_are_counted_by_reason():
    root_reasons = dict.fromkeys(("queue", "slower", "protocol_clean"), "root_step")
    tool_reasons = dict.fromkeys(("queue", "slower", "protocol_clean"), "tool_step")
    unjudged = [
        step(None, truth_reasons=root_reasons),
        step(None, truth_reasons=tool_reasons),
        step(None, truth_reasons=tool_reasons),
    ]
    rows = [step("inference")]
    report = build_report(rows, rules_scores(1), context(unjudged=unjudged))
    steps = report["unscored"]["steps"]
    assert steps["queue"] == {"root_step": 1, "tool_step": 2}
    assert steps["slower"] == {"root_step": 1, "tool_step": 2}
    assert steps["protocol_clean"] == {"not_applicable": 1, "root_step": 1, "tool_step": 2}
    assert report["rollups"]["by_scorer_truth"][0]["steps"] == 1


def test_cells_split_seen_and_unseen_tasks():
    rows = [
        step("inference", task="fix-mean-empty-r1"),
        step("none", task="fix-mean-empty-r1"),
        step("inference", task="fix-median-even-r1"),
    ]
    report = build_report(rows, rules_scores(3), context())
    by_seen = {
        entry["task_seen_in_training"]: entry
        for entry in report["rollups"]["by_task_seen"]
        if entry["truth"] == "queue"
    }
    assert set(by_seen) == {True, False}
    assert by_seen[True]["steps"] == 2 and by_seen[False]["steps"] == 1
    assert by_seen[False]["inference_steps"] == 1
    assert report["seen_task_groups"] == ["fix-mean-empty"]
    assert {entry["task_seen_in_training"] for entry in report["cells"]} == {True, False}
    assert {"by_scorer_truth", "by_configuration", "by_configuration_task_seen"} <= set(
        report["rollups"]
    )
    overall = [e for e in report["rollups"]["by_scorer_truth"] if e["truth"] == "queue"]
    assert len(overall) == 1 and overall[0]["steps"] == 3


def test_report_states_its_fixed_claims():
    report = build_report([step("inference")], rules_scores(1), context())
    assert report["held_out"] is True
    assert report["gate"] == "none (descriptive baseline)"
    assert report["protocol_sha256"] == {"cfg-default": "a" * 64}
    assert report["limitations"] == list(LIMITATIONS)
    assert report["model"] == {"version": "training-v1"}


def test_unavailable_abstention_leaves_classifier_rates_none_not_zero():
    rows = [step("inference"), step("none")]
    scores = rules_scores(2) | {CLASSIFIER_SCORER: [scored("none"), scored("none")]}
    report = build_report(rows, scores, context(abstention_unavailable=True))
    overall = {
        e["scorer"]: e for e in report["rollups"]["by_scorer_truth"] if e["truth"] == "queue"
    }
    classifier = overall[CLASSIFIER_SCORER]
    assert classifier["abstention_rate"] is None
    assert classifier["answered_miss_rate"] is None
    assert classifier["answered_false_alarm_rate"] is None
    assert classifier["miss_rate"] == 1.0
    assert overall[RULES_SCORER]["abstention_rate"] == 0.0
