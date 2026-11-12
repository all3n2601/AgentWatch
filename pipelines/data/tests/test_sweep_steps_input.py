"""Tests for reading capacity-sweep step records.

Every fixture here is synthetic recorder output written by the tests, not a recorded run.
"""

import hashlib
import json
import uuid
from pathlib import Path

import pytest
from agentwatch_data.files import file_sha256
from agentwatch_data.labels import LabelingError
from agentwatch_data.sweep_steps_input import (
    STEPS_FILE,
    SUMMARY_FILE,
    TIMING_TOLERANCE_SECONDS,
    SweepStepsError,
    read_steps,
)

HASH = "a" * 64
OTHER_HASH = "b" * 64
PROTOCOLS = {"cfg-default": HASH, "cfg-hardware": OTHER_HASH}


def valid_line(**changes) -> dict:
    line = {
        "run_id": str(uuid.UUID(int=1)),
        "run_folder": "batch-1/run-1",
        "step_id": "s1",
        "step_type": "model_call",
        "parent_step_id": "root",
        "plan_sha256": "c" * 64,
        "task_id": "task-1",
        "configuration": "cfg-default",
        "n": 2,
        "phase": "live",
        "batch_sequence": 0,
        "batch_id": "batch-1",
        "state": "completed",
        "duration_seconds": 1.5,
        "queue_seconds": 0.1,
        "execution_seconds": 1.4,
        "span_status": "measured",
        "vllm_queue_seconds": 0.2,
        "vllm_service_seconds": 1.0,
        "baseline": {"count": 3, "mean_seconds": 1.0, "stdev_seconds": 0.1},
    }
    line.update(changes)
    return line


def write_input(tmp_path: Path, lines: list, newline: str = "\n", **summary_changes) -> Path:
    texts = [line if isinstance(line, str) else json.dumps(line) for line in lines]
    written = "".join(text + newline for text in texts).encode()
    (tmp_path / STEPS_FILE).write_bytes(written)
    summary = {
        "format": "sweep-steps-v1",
        "sweep_folder": "sweep-1",
        "sweep_id": "id-1",
        "configuration": "cfg-default",
        "protocol_version": "natural-overload-sweep-v1",
        "protocol_sha256": HASH,
        "sweep_json_sha256": "d" * 64,
        "recorder_sha256": "c" * 64,
        "held_out": True,
        "steps": len(texts),
        "steps_sha256": hashlib.sha256(written).hexdigest(),
        "excluded_batches": [{"sequence": 1, "level": 2, "causes": []}],
        "runs_without_evidence": 0,
        "runs_without_plan": 0,
    }
    summary.update(summary_changes)
    summary = {key: value for key, value in summary.items() if value is not ...}
    (tmp_path / SUMMARY_FILE).write_text(json.dumps(summary), encoding="utf-8")
    return tmp_path


def test_valid_input_reads_back(tmp_path):
    second = valid_line(step_id="s2", parent_step_id=None, baseline=None, task_id=None)
    write_input(tmp_path, [valid_line(), second])
    result = read_steps(tmp_path, PROTOCOLS)
    assert list(result.records) == [valid_line(), second]
    assert result.summary["steps"] == 2
    assert result.steps_sha256 == file_sha256(tmp_path / STEPS_FILE)
    assert result.summary_sha256 == file_sha256(tmp_path / SUMMARY_FILE)


def test_crlf_and_trailing_newline_read_the_same(tmp_path):
    lines = [valid_line(), valid_line(step_id="s2")]
    write_input(tmp_path, lines)
    lf = read_steps(tmp_path, PROTOCOLS)
    write_input(tmp_path, lines, newline="\r\n")
    crlf = read_steps(tmp_path, PROTOCOLS)
    assert crlf.records == lf.records


def test_empty_lines_are_ignored(tmp_path):
    write_input(tmp_path, [valid_line(), "", valid_line(step_id="s2")], steps=2)
    assert len(read_steps(tmp_path, PROTOCOLS).records) == 2


@pytest.mark.parametrize(
    ("changes", "field"),
    [
        ({"format": "sweep-steps-v0"}, "format"),
        ({"held_out": False}, "held_out"),
        ({"held_out": ...}, "held_out"),
        ({"protocol_version": "natural-overload-sweep-v2"}, "protocol_version"),
        ({"protocol_sha256": "e" * 64}, "protocol_sha256"),
        ({"protocol_sha256": OTHER_HASH}, "protocol_sha256"),
        ({"configuration": "cfg-unknown"}, "configuration"),
        ({"steps": 5}, "steps"),
        ({"steps": True}, "steps"),
        ({"excluded_batches": {}}, "excluded_batches"),
        ({"runs_without_plan": -1}, "runs_without_plan"),
        ({"sweep_id": 3}, "sweep_id"),
        ({"sweep_json_sha256": "d" * 63}, "sweep_json_sha256"),
        ({"sweep_json_sha256": "D" * 64}, "sweep_json_sha256"),
        ({"recorder_sha256": ...}, "recorder_sha256"),
        ({"recorder_sha256": "c" * 63}, "recorder_sha256"),
        ({"recorder_sha256": "C" * 64}, "recorder_sha256"),
        ({"recorder_sha256": None}, "recorder_sha256"),
        ({"steps_sha256": ...}, "steps_sha256"),
        ({"steps_sha256": "not-a-hash"}, "steps_sha256"),
    ],
)
def test_summary_is_refused(tmp_path, changes, field):
    write_input(tmp_path, [valid_line()], **changes)
    with pytest.raises(SweepStepsError, match=field) as raised:
        read_steps(tmp_path, PROTOCOLS)
    assert SUMMARY_FILE in str(raised.value)
    assert isinstance(raised.value, LabelingError)


def without(key: str) -> dict:
    line = valid_line()
    del line[key]
    return line


@pytest.mark.parametrize(
    ("line", "field"),
    [
        (without("state"), "state"),
        ({**valid_line(), "extra": 1}, "extra"),
        (valid_line(run_id="not-a-uuid"), "run_id"),
        (valid_line(n=0), "n"),
        (valid_line(n=True), "n"),
        (valid_line(batch_sequence=-1), "batch_sequence"),
        (valid_line(phase="warmup"), "phase"),
        (valid_line(duration_seconds=-0.1), "duration_seconds"),
        (valid_line(duration_seconds=float("nan")), "duration_seconds"),
        (valid_line(duration_seconds="1.0"), "duration_seconds"),
        (valid_line(queue_seconds=True), "queue_seconds"),
        (valid_line(duration_seconds=1.6), "duration_seconds"),
        (valid_line(execution_seconds=1.0), "duration_seconds"),
        (valid_line(span_status="reported"), "span_status"),
        (valid_line(span_status="unavailable"), "vllm_queue_seconds"),
        (
            valid_line(span_status="malformed", vllm_queue_seconds=None),
            "vllm_service_seconds",
        ),
        (valid_line(baseline={"count": 3, "mean_seconds": 1.0}), "baseline"),
        (
            valid_line(baseline={"count": 0, "mean_seconds": 1.0, "stdev_seconds": None}),
            "baseline.count",
        ),
        (
            valid_line(baseline={"count": 2, "mean_seconds": -1.0, "stdev_seconds": 0.1}),
            "baseline.mean_seconds",
        ),
        (valid_line(run_folder="../x"), "run_folder"),
        (valid_line(run_folder="/abs"), "run_folder"),
        (valid_line(step_id=""), "step_id"),
        (valid_line(parent_step_id=3), "parent_step_id"),
        ("{not json", "JSON"),
        ("[1]", "object"),
    ],
)
def test_line_is_refused(tmp_path, line, field):
    write_input(tmp_path, [valid_line(step_id="ok"), line])
    with pytest.raises(SweepStepsError, match=field) as raised:
        read_steps(tmp_path, PROTOCOLS)
    assert f"{STEPS_FILE} line 2" in str(raised.value)


def test_steps_from_another_write_are_refused(tmp_path):
    write_input(tmp_path, [valid_line()], steps_sha256="e" * 64)
    with pytest.raises(SweepStepsError, match="steps_sha256") as raised:
        read_steps(tmp_path, PROTOCOLS)
    assert "not from the same write" in str(raised.value)


def test_duration_breakdown_within_tolerance_is_accepted(tmp_path):
    near = valid_line(duration_seconds=1.5 + TIMING_TOLERANCE_SECONDS / 2)
    write_input(tmp_path, [near])
    assert (
        read_steps(tmp_path, PROTOCOLS).records[0]["duration_seconds"] == near["duration_seconds"]
    )


@pytest.mark.parametrize("field", ["duration_seconds", "queue_seconds", "execution_seconds"])
def test_duration_breakdown_is_not_checked_when_a_part_is_missing(tmp_path, field):
    write_input(tmp_path, [valid_line(**{field: None})])
    assert read_steps(tmp_path, PROTOCOLS).records[0][field] is None


def test_repeated_run_and_step_is_refused(tmp_path):
    write_input(tmp_path, [valid_line(), valid_line()])
    with pytest.raises(SweepStepsError, match="step_id") as raised:
        read_steps(tmp_path, PROTOCOLS)
    assert "line 2" in str(raised.value)


def test_unavailable_span_with_null_values_is_accepted(tmp_path):
    line = valid_line(span_status="unavailable", vllm_queue_seconds=None, vllm_service_seconds=None)
    write_input(tmp_path, [line])
    assert read_steps(tmp_path, PROTOCOLS).records[0]["span_status"] == "unavailable"


def test_missing_summary_is_refused(tmp_path):
    write_input(tmp_path, [valid_line()])
    (tmp_path / SUMMARY_FILE).unlink()
    with pytest.raises(SweepStepsError, match=SUMMARY_FILE):
        read_steps(tmp_path, PROTOCOLS)


def test_missing_steps_file_is_refused(tmp_path):
    write_input(tmp_path, [valid_line()])
    (tmp_path / STEPS_FILE).unlink()
    with pytest.raises(SweepStepsError, match=STEPS_FILE):
        read_steps(tmp_path, PROTOCOLS)
