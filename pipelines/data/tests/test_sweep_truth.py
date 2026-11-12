"""Tests for the sweep-truth-v2 rule. All step records here are synthetic fixtures."""

import hashlib
import json
import re
from importlib.resources import files
from pathlib import Path

import pytest
from agentwatch_data.labels import LabelingError
from agentwatch_data.sweep_truth import (
    TRUTH_FILE,
    TRUTH_NAMES,
    TRUTH_SUMMARY_FILE,
    SweepTruthRule,
    judge,
    load_truth_rule,
    main,
    noise_threshold,
    write_truths,
)
from test_sweep_steps_input import valid_line, write_input

RULE_FILE = "sweep-truth-v2.toml"
# The full label_version of the packaged sweep-truth rule and the code that applies it, pinned so
# no change to either reaches a release unnoticed. When this fails: if the change can alter any
# truth the rule produces, or whether a step is judged at all, bump `version` in
# sweep-truth-v2.toml. Otherwise (a comment or refactor) keep the version, and say why in the PR.
# Either way, update the pin and the docs (docs/data-card.md and docs/project-checklist.md) in
# the same PR.
PINNED_SWEEP_TRUTH_LABEL_VERSION = "sweep-truth-v2+f96f57be"
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
    assert rule.version == "sweep-truth-v2"
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
    assert re.fullmatch(r"sweep-truth-v2\+[0-9a-f]{8}", rule.label_version)


def test_sweep_truth_label_version_is_pinned(rule):
    assert rule.label_version == PINNED_SWEEP_TRUTH_LABEL_VERSION, (
        "sweep_truth.py, sweep_steps_input.py or sweep-truth-v2.toml changed; see the comment "
        "above PINNED_SWEEP_TRUTH_LABEL_VERSION"
    )


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


@pytest.mark.parametrize(
    "changes",
    [{"state": "failed"}, {"state": "completed", "duration_seconds": None}],
    ids=["failed", "completed-without-duration"],
)
def test_protocol_clean_is_not_completed_for_an_unfinished_call_in_a_clean_batch(rule, changes):
    result = judged(rule, phase="control", n=1, **changes)
    assert result["truths"]["protocol_clean"] is None
    assert result["truth_reasons"]["protocol_clean"] == "not_completed"
    assert result["truth_reasons"]["queue"] == "not_completed"


def test_protocol_clean_stays_none_for_a_completed_call_in_a_clean_batch(rule):
    result = judged(rule, phase="control", n=1)
    assert result["truths"]["protocol_clean"] == "none"
    assert result["truth_reasons"]["protocol_clean"] is None


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


def test_no_baseline_comes_before_span_not_measured(rule):
    result = judged(rule, baseline=None, span_status="unavailable", vllm_queue_seconds=None)
    assert result["truths"]["queue"] is None
    assert result["truth_reasons"]["queue"] == "no_baseline"
    assert result["truths"]["slower"] is None
    assert result["truth_reasons"]["slower"] == "no_baseline"


def test_ramp_n1_slower_is_not_applicable_even_when_not_completed(rule):
    result = judged(rule, phase="ramp", n=1, state="failed")
    assert result["truths"]["slower"] is None
    assert result["truth_reasons"]["slower"] == "not_applicable"
    assert result["truths"]["queue"] is None
    assert result["truth_reasons"]["queue"] == "not_completed"
    assert result["truths"]["protocol_clean"] is None
    assert result["truth_reasons"]["protocol_clean"] == "not_completed"


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


def steps_input(tmp_path: Path, rule: SweepTruthRule, name: str = "steps") -> Path:
    """Synthetic recorder output covering a classed, an unmeasured and a root step."""
    lines = [
        valid_line(step_id="s1", phase="injected", n=4, vllm_queue_seconds=2.0),
        valid_line(step_id="s2", phase="ramp", n=1, vllm_queue_seconds=0.0),
        valid_line(
            step_id="s3",
            span_status="unavailable",
            vllm_queue_seconds=None,
            vllm_service_seconds=None,
        ),
        valid_line(step_id="root", parent_step_id=None, step_type="agent_run", baseline=None),
    ]
    folder = tmp_path / name
    folder.mkdir()
    return write_input(folder, lines, protocol_sha256=rule.protocols["cfg-default"])


def read_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_records_and_summary_fields(tmp_path, rule):
    source = steps_input(tmp_path, rule)
    output = write_truths(source, tmp_path / "out", rule)
    records = read_lines(output / TRUTH_FILE)
    summary = json.loads((output / TRUTH_SUMMARY_FILE).read_text(encoding="utf-8"))
    first = records[0]
    for dropped in ("span_status", "vllm_service_seconds", "baseline"):
        assert dropped not in first
    assert first["vllm_queue_seconds"] == 2.0
    assert first["baseline_mean_seconds"] == 1.0
    assert first["baseline_stdev_seconds"] == 0.1
    assert first["baseline_count"] == 3
    assert first["truths"]["queue"] == "inference"
    assert first["noise_threshold_seconds"] == pytest.approx(0.3)
    assert records[3]["baseline_count"] is None
    assert records[3]["truth_reasons"]["queue"] == "root_step"
    assert summary["label_version"] == rule.label_version
    assert summary["held_out"] is True
    assert summary["steps"] == 4
    assert summary["configuration"] == "cfg-default"
    assert summary["sweep_folder"] == "sweep-1"
    assert summary["steps_sha256"] and summary["steps_summary_sha256"]
    assert summary["rule"] == rule.values
    assert summary["rule"]["noise"] == {"stdev_multiplier": 3.0, "min_seconds": 0.01}
    truth_bytes = (output / TRUTH_FILE).read_bytes()
    assert summary["truth_sha256"] == hashlib.sha256(truth_bytes).hexdigest()
    assert set(summary["counts"]) == set(TRUTH_NAMES)


def test_a_rule_given_by_file_is_recorded_with_its_values(tmp_path, rule):
    path = tmp_path / RULE_FILE
    path.write_text(packaged_text().replace("min_seconds = 0.01", "min_seconds = 0.02"))
    output = tmp_path / "out"
    assert (
        main([str(steps_input(tmp_path, rule)), "--output", str(output), "--rules", str(path)]) == 0
    )
    summary = json.loads((output / TRUTH_SUMMARY_FILE).read_text(encoding="utf-8"))
    assert summary["label_version"] != rule.label_version
    assert summary["rule"]["noise"]["min_seconds"] == 0.02


def test_counts_match_records(tmp_path, rule):
    output = write_truths(steps_input(tmp_path, rule), tmp_path / "out", rule)
    records = read_lines(output / TRUTH_FILE)
    counts = json.loads((output / TRUTH_SUMMARY_FILE).read_text(encoding="utf-8"))["counts"]
    for name in TRUTH_NAMES:
        classes, reasons = {}, {}
        for item in records:
            truth, reason = item["truths"][name], item["truth_reasons"][name]
            if truth is not None:
                classes[truth] = classes.get(truth, 0) + 1
            if reason is not None:
                reasons[reason] = reasons.get(reason, 0) + 1
        assert counts[name] == {"classes": classes, "reasons": reasons}
    assert counts["queue"]["classes"] == {"inference": 1, "none": 1}
    assert counts["queue"]["reasons"] == {"root_step": 1, "span_not_measured": 1}


def test_same_input_gives_identical_bytes(tmp_path, rule):
    source = steps_input(tmp_path, rule)
    first = write_truths(source, tmp_path / "one", rule)
    second = write_truths(source, tmp_path / "two", rule)
    for name in (TRUTH_FILE, TRUTH_SUMMARY_FILE):
        assert (first / name).read_bytes() == (second / name).read_bytes()
        assert b"\r" not in (first / name).read_bytes()


def test_refused_input_writes_nothing(tmp_path, rule):
    source = steps_input(tmp_path, rule)
    (source / "steps.json").write_text("{}", encoding="utf-8")
    with pytest.raises(LabelingError):
        write_truths(source, tmp_path / "out", rule)
    assert not (tmp_path / "out").exists()


def test_output_inside_held_out_is_refused(tmp_path, rule):
    source = steps_input(tmp_path, rule)
    target = tmp_path / "data" / "held-out" / "out"
    with pytest.raises(LabelingError, match="held-out"):
        write_truths(source, target, rule)
    assert not target.exists()


def test_output_resolving_into_held_out_is_refused(tmp_path, rule):
    source = steps_input(tmp_path, rule)
    held_out = tmp_path / "data" / "held-out"
    held_out.mkdir(parents=True)
    link = tmp_path / "link"
    try:
        link.symlink_to(held_out, target_is_directory=True)
    except OSError:
        pytest.skip("symbolic links are not available here")
    with pytest.raises(LabelingError, match="held-out"):
        write_truths(source, link / "out", rule)
    assert not (held_out / "out").exists()


def test_main_exit_codes(tmp_path, rule, capsys):
    source = steps_input(tmp_path, rule)
    output = tmp_path / "out"
    assert main([str(source), "--output", str(output)]) == 0
    assert str(output / TRUTH_FILE) in capsys.readouterr().out
    assert (output / TRUTH_SUMMARY_FILE).is_file()

    bad = tmp_path / "bad"
    bad.mkdir()
    assert main([str(bad), "--output", str(tmp_path / "bad-out")]) == 2
    assert capsys.readouterr().err.startswith("error: ")

    rules = tmp_path / "rules.toml"
    rules.write_text("version = ", encoding="utf-8")
    assert main([str(source), "--output", str(tmp_path / "r"), "--rules", str(rules)]) == 2
    assert capsys.readouterr().err.startswith("error: ")

    held = tmp_path / "data" / "held-out" / "out"
    assert main([str(source), "--output", str(held)]) == 2
    assert "held-out" in capsys.readouterr().err
