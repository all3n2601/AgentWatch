"""Tests for the sweep-truth-v1 rule. All step records here are synthetic fixtures."""

import re
from importlib.resources import files
from pathlib import Path

import pytest
from agentwatch_data.labels import LabelingError
from agentwatch_data.sweep_truth import (
    TRUTH_NAMES,
    SweepTruthRule,
    judge,
    load_truth_rule,
    noise_threshold,
)

RULE_FILE = "sweep-truth-v1.toml"
BASELINE = {"count": 3, "mean_seconds": 1.0, "stdev_seconds": 0.1}


@pytest.fixture(scope="module")
def rule() -> SweepTruthRule:
    return load_truth_rule()


def packaged_text() -> str:
    return files("agentwatch_data").joinpath(RULE_FILE).read_text(encoding="utf-8")


def record(**changes) -> dict:
    base = {
        "step_type": "model_call",
        "parent_step_id": "root",
        "phase": "injected",
        "n": 4,
        "state": "completed",
        "duration_seconds": 1.0,
        "span_status": "measured",
        "vllm_queue_seconds": 0.0,
        "baseline": dict(BASELINE),
    }
    base.update(changes)
    return base


def assert_class_or_reason(result: dict) -> None:
    for name in TRUTH_NAMES:
        truth, reason = result["truths"][name], result["truth_reasons"][name]
        assert (truth is None) != (reason is None), (name, truth, reason)


def judged(rule: SweepTruthRule, **changes) -> dict:
    result = judge(record(**changes), rule)
    assert_class_or_reason(result)
    return result


def test_packaged_rule_values(rule):
    assert rule.version == "sweep-truth-v1"
    assert rule.stdev_multiplier == 3.0
    assert rule.min_seconds == 0.01
    assert rule.min_baseline_runs == 3
    assert rule.scored_step_types == ("model_call",)
    assert rule.clean_phases == ("ramp", "control")
    assert rule.clean_level == 1
    assert rule.baseline_phase == "ramp"
    assert rule.baseline_level == 1
    assert rule.protocols == {
        "cfg-default": "22e26dc7536fe5e88e3d9ea9f01445a754788ce587b1626d6a4eb055ec4524b9",
        "cfg-hardware": "1608deee2a3bcb5245b07d73e13e73dcd1e82702ef14aa683ecd94144e0f18cd",
    }
    assert re.fullmatch(r"sweep-truth-v1\+[0-9a-f]{8}", rule.label_version)


def test_comment_does_not_change_the_identifier(tmp_path, rule):
    path = tmp_path / RULE_FILE
    path.write_text("# a new comment\n" + packaged_text(), encoding="utf-8")
    assert load_truth_rule(path).label_version == rule.label_version


def test_a_value_changes_the_identifier(tmp_path, rule):
    path = tmp_path / RULE_FILE
    path.write_text(packaged_text().replace("min_seconds = 0.01", "min_seconds = 0.02"))
    changed = load_truth_rule(path)
    assert changed.min_seconds == 0.02
    assert changed.label_version != rule.label_version


def bad_text(old: str, new: str) -> str:
    text = packaged_text()
    assert old in text
    return text.replace(old, new)


MALFORMED = {
    "missing_noise": lambda: bad_text("[noise]", "[nois]"),
    "zero_multiplier": lambda: bad_text("stdev_multiplier = 3.0", "stdev_multiplier = 0"),
    "empty_scored_types": lambda: bad_text('scored_types = ["model_call"]', "scored_types = []"),
    "short_protocol_hash": lambda: bad_text(
        "22e26dc7536fe5e88e3d9ea9f01445a754788ce587b1626d6a4eb055ec4524b9", "22e26dc7536f"
    ),
}


@pytest.mark.parametrize("name", MALFORMED)
def test_malformed_rule_is_refused(tmp_path: Path, name):
    path = tmp_path / RULE_FILE
    path.write_text(MALFORMED[name](), encoding="utf-8")
    with pytest.raises(LabelingError):
        load_truth_rule(path)


def test_model_call_queue_above_and_at_threshold(rule):
    threshold = noise_threshold(BASELINE, rule)
    assert threshold == pytest.approx(0.3)
    above = judged(rule, vllm_queue_seconds=2.0)
    at = judged(rule, vllm_queue_seconds=threshold)
    assert above["truths"]["queue"] == "inference"
    assert at["truths"]["queue"] == "none"
    assert at["noise_threshold_seconds"] == threshold


def test_model_call_slower_above_and_below(rule):
    above = judged(rule, duration_seconds=1.0 + 0.31)
    below = judged(rule, duration_seconds=1.0 + 0.29)
    assert above["truths"]["slower"] == "inference"
    assert below["truths"]["slower"] == "none"
    assert above["excess_seconds"] == pytest.approx(0.31)


def test_slower_unset_on_ramp_n1(rule):
    result = judged(rule, phase="ramp", n=1, vllm_queue_seconds=2.0)
    assert result["truths"]["slower"] is None
    assert result["truth_reasons"]["slower"] == "not_applicable"
    assert result["truths"]["queue"] == "inference"


def test_slower_judged_on_control_n1(rule):
    result = judged(rule, phase="control", n=1, duration_seconds=3.0)
    assert result["truths"]["slower"] == "inference"


def test_protocol_clean_only_in_clean_batches(rule):
    for phase in ("ramp", "control"):
        assert judged(rule, phase=phase, n=1)["truths"]["protocol_clean"] == "none"
    for phase, n in (("injected", 1), ("ramp", 2), ("live", 3)):
        result = judged(rule, phase=phase, n=n)
        assert result["truths"]["protocol_clean"] is None
        assert result["truth_reasons"]["protocol_clean"] == "not_applicable"


def test_protocol_clean_ignores_state(rule):
    result = judged(rule, phase="control", n=1, state="failed")
    assert result["truths"]["protocol_clean"] == "none"
    assert result["truth_reasons"]["queue"] == "not_completed"


def test_root_step_is_root_step_everywhere(rule):
    result = judged(rule, parent_step_id=None, step_type="agent_run")
    assert result["truths"] == dict.fromkeys(TRUTH_NAMES)
    assert set(result["truth_reasons"].values()) == {"root_step"}
    assert result["noise_threshold_seconds"] is None
    assert result["excess_seconds"] is None


@pytest.mark.parametrize("step_type", ["tool_call", "retrieval", "other"])
def test_tool_retrieval_other_are_no_queue_reference(rule, step_type):
    result = judged(rule, step_type=step_type, phase="control", n=1)
    assert result["truths"] == dict.fromkeys(TRUTH_NAMES)
    assert set(result["truth_reasons"].values()) == {"no_queue_reference"}
    assert result["noise_threshold_seconds"] is None
    assert result["excess_seconds"] is None


def test_root_step_wins_over_model_call_type(rule):
    result = judged(rule, parent_step_id=None, step_type="model_call")
    assert set(result["truth_reasons"].values()) == {"root_step"}


def test_not_completed_before_malformed_before_no_baseline(rule):
    not_completed = judged(rule, state="failed", duration_seconds=-1.0, baseline=None)
    malformed = judged(rule, duration_seconds=-1.0, baseline=None)
    no_baseline = judged(rule, baseline=None)
    assert not_completed["truth_reasons"]["queue"] == "not_completed"
    assert not_completed["truth_reasons"]["slower"] == "not_completed"
    assert malformed["truth_reasons"]["queue"] == "malformed"
    assert malformed["truth_reasons"]["slower"] == "malformed"
    assert no_baseline["truth_reasons"]["queue"] == "no_baseline"
    assert no_baseline["truth_reasons"]["slower"] == "no_baseline"
    assert no_baseline["noise_threshold_seconds"] is None
    assert no_baseline["excess_seconds"] is None


def test_completed_without_duration_is_not_completed(rule):
    result = judged(rule, duration_seconds=None)
    assert result["truth_reasons"]["queue"] == "not_completed"
    assert result["truth_reasons"]["slower"] == "not_completed"


def test_baseline_below_min_runs_is_no_baseline(rule):
    result = judged(rule, baseline={**BASELINE, "count": 2})
    assert result["truth_reasons"]["queue"] == "no_baseline"
    assert result["truth_reasons"]["slower"] == "no_baseline"


def test_unmeasured_span_leaves_queue_unset_but_judges_slower(rule):
    result = judged(rule, span_status="unavailable", vllm_queue_seconds=None, duration_seconds=2.0)
    assert result["truths"]["queue"] is None
    assert result["truth_reasons"]["queue"] == "span_not_measured"
    assert result["truths"]["slower"] == "inference"


def test_measured_span_without_queue_value_is_span_not_measured(rule):
    result = judged(rule, span_status="measured", vllm_queue_seconds=None)
    assert result["truths"]["queue"] is None
    assert result["truth_reasons"]["queue"] == "span_not_measured"


def test_null_stdev_uses_the_floor(rule):
    baseline = {"count": 3, "mean_seconds": 1.0, "stdev_seconds": None}
    assert noise_threshold(baseline, rule) == 0.01
    result = judged(rule, baseline=baseline)
    assert result["noise_threshold_seconds"] == 0.01


def test_a_truth_is_a_class_or_a_reason_never_both(rule):
    no_baseline = {**BASELINE, "count": 2}
    cases = [
        {},
        {"vllm_queue_seconds": 2.0, "duration_seconds": 3.0},
        {"phase": "ramp", "n": 1},
        {"phase": "control", "n": 1},
        {"phase": "control", "n": 1, "state": "failed"},
        {"phase": "live", "n": 3},
        {"parent_step_id": None, "step_type": "agent_run"},
        {"parent_step_id": None},
        {"step_type": "tool_call"},
        {"step_type": "retrieval"},
        {"step_type": "other"},
        {"state": "failed"},
        {"duration_seconds": None},
        {"duration_seconds": -1.0},
        {"baseline": None},
        {"baseline": no_baseline},
        {"baseline": {**BASELINE, "stdev_seconds": None}},
        {"span_status": "unavailable", "vllm_queue_seconds": None},
        {"span_status": "measured", "vllm_queue_seconds": None},
    ]
    for changes in cases:
        assert_class_or_reason(judge(record(**changes), rule))
