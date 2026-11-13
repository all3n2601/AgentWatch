"""Held-out feature rows: truth records joined to features from synthetic sweep run folders.

The sweeps here are synthetic (the data tests' replay journals copied into a sweep layout with
invented truth values); none of it is recorded held-out evidence.
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest
from agentwatch_data.schema import feature_columns
from agentwatch_data.sweep_truth import load_truth_rule

# The synthetic live journals are written by the data tests' fixtures.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "data" / "tests"))

from agentwatch_model.held_out_rows import (
    ACCEPTED_PROTOCOL_SHA256,
    ACCEPTED_TRUTH_LABEL_VERSION,
    HeldOutError,
    held_out_rows,
    load_truths,
)
from replay_fixtures import live_timing, write_experiment

SWEEP_FOLDER = "cfg-default-synthetic"
LEAF_STEPS = ("model-call-1", "tool-call-1")
STEP_SECONDS = {"agent-run": 1.0, "model-call-1": 0.6, "tool-call-1": 0.2}


def record(run_id: str, run_folder: str, step_id: str, **overrides) -> dict:
    timing = live_timing(step_id, STEP_SECONDS[step_id])
    base = {
        "run_id": run_id,
        "step_id": step_id,
        "step_type": timing["step_type"],
        "task_id": "fix-a-r1",
        "parent_step_id": None if step_id == "agent-run" else "agent-run",
        "plan_sha256": None,
        "configuration": "cfg-default",
        "n": 1,
        "phase": "ramp",
        "batch_sequence": 1,
        "batch_id": "batch-1",
        "state": "ok",
        "duration_seconds": timing["total_seconds"],
        "queue_seconds": timing["queue_seconds"],
        "execution_seconds": timing["execution_seconds"],
        "vllm_queue_seconds": None,
        "baseline_mean_seconds": None,
        "baseline_stdev_seconds": None,
        "baseline_count": 0,
        "noise_threshold_seconds": None,
        "excess_seconds": None,
        "truths": {"queue": "none", "slower": None, "protocol_clean": "none"},
        "truth_reasons": {"queue": None, "slower": "not_applicable", "protocol_clean": None},
        "run_folder": run_folder,
    }
    return base | overrides


STEPS_SHA256 = "a1" * 32
STEPS_SUMMARY_SHA256 = "b2" * 32
RECORDER_SHA256 = "c3" * 32


def write_summary(directory: Path, **overrides) -> dict:
    truth_file = directory / "truth.jsonl"
    truth_bytes = truth_file.read_bytes() if truth_file.exists() else b""
    summary = {
        "label_version": ACCEPTED_TRUTH_LABEL_VERSION,
        "truth_sha256": hashlib.sha256(truth_bytes).hexdigest(),
        "steps_sha256": STEPS_SHA256,
        "steps_summary_sha256": STEPS_SUMMARY_SHA256,
        "recorder_sha256": RECORDER_SHA256,
        "sweep_id": "synthetic",
        "configuration": "cfg-default",
        "protocol_version": "synthetic",
        "protocol_sha256": ACCEPTED_PROTOCOL_SHA256["cfg-default"],
        "held_out": True,
        "sweep_folder": SWEEP_FOLDER,
        "steps": len(truth_bytes.splitlines()),
        "runs_without_plan": 0,
        "runs_without_evidence": 0,
        "excluded_batches": [],
    } | overrides
    (directory / "truth.json").write_text(json.dumps(summary))
    return summary


def write_truths(directory: Path, records: list[dict]) -> None:
    """Write truth.jsonl and, when a summary exists, keep its truth_sha256 and steps in step."""
    (directory / "truth.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    summary_file = directory / "truth.json"
    if summary_file.exists():
        summary = json.loads(summary_file.read_text())
        kept = {k: v for k, v in summary.items() if k not in {"truth_sha256", "steps"}}
        write_summary(directory, **kept)


@pytest.fixture
def sweep(tmp_path):
    """sweep_root with one run folder (runs/a) and a truth folder naming its three steps."""
    synthetic = write_experiment(tmp_path / "source", live=True)
    run_dir = tmp_path / "sweeps" / SWEEP_FOLDER / "runs" / "a"
    shutil.copytree(synthetic / "replays" / "01", run_dir)
    evidence = json.loads((run_dir / "agent-evidence.json").read_text())
    truth_dir = tmp_path / "truth"
    truth_dir.mkdir()
    records = [record(evidence["run_id"], "runs/a", step) for step in STEP_SECONDS]
    write_truths(truth_dir, records)
    write_summary(truth_dir)
    return {
        "root": tmp_path / "sweeps",
        "truth": truth_dir,
        "run_dir": run_dir,
        "run_id": evidence["run_id"],
        "records": records,
    }


def test_held_out_rows_join_truth_and_features(sweep):
    rows, refused, summaries, unjudged = held_out_rows([sweep["truth"]], sweep["root"])
    assert refused == {}
    assert unjudged == []
    assert len(summaries) == 1
    assert [row["step_id"] for row in rows] == list(STEP_SECONDS)
    for row in rows:
        assert set(feature_columns()) <= set(row)
        assert row["truths"]["queue"] == "none"
        assert row["telemetry_status"]
        assert row["worker_capacity"] is None
        assert row["queue_seconds"] == pytest.approx(0.001)
    assert rows[0]["execution_seconds"] == pytest.approx(0.999)


def test_a_timing_mismatch_refuses_only_that_step(sweep):
    records = sweep["records"]
    records[1] = records[1] | {"duration_seconds": 9.0}
    write_truths(sweep["truth"], records)
    rows, refused, _, _ = held_out_rows([sweep["truth"]], sweep["root"])
    assert refused == {"feature_timing_mismatch": 1}
    assert [row["step_id"] for row in rows] == ["agent-run", "tool-call-1"]


def test_a_journal_of_another_run_refuses_that_run(sweep):
    other = "00000000-0000-4000-8000-0000000000ff"
    write_truths(sweep["truth"], [r | {"run_id": other} for r in sweep["records"]])
    rows, refused, _, _ = held_out_rows([sweep["truth"]], sweep["root"])
    assert rows == []
    assert refused == {"feature_row_missing": 3}


def test_a_missing_journal_refuses_that_run(sweep):
    (sweep["run_dir"] / "live-journal.jsonl").unlink()
    rows, refused, _, _ = held_out_rows([sweep["truth"]], sweep["root"])
    assert rows == []
    assert refused == {"feature_evidence_unreadable": 3}


@pytest.mark.parametrize("run_folder", ["/etc", "C:/x", "../escape", "runs/../../x"])
def test_a_run_folder_outside_the_sweep_is_refused(sweep, run_folder):
    write_truths(sweep["truth"], [r | {"run_folder": run_folder} for r in sweep["records"]])
    rows, refused, _, _ = held_out_rows([sweep["truth"]], sweep["root"])
    assert rows == []
    assert refused == {"feature_evidence_unreadable": 3}


@pytest.mark.parametrize(
    "change",
    [
        {"held_out": False},
        {"held_out": None},
        {"label_version": "sweep-truth-v0"},
        {"protocol_sha256": "0" * 64},
    ],
)
def test_truths_without_held_out_are_refused(sweep, change):
    write_summary(sweep["truth"], **change)
    with pytest.raises(HeldOutError):
        load_truths(sweep["truth"])


def test_the_accepted_summary_loads(sweep):
    summary, records = load_truths(sweep["truth"])
    assert summary["sweep_folder"] == SWEEP_FOLDER
    assert len(records) == 3


MISSING = object()


def changed(record: dict, path: str, value: object) -> dict:
    """The record with one field, or one `truths`/`truth_reasons` entry, replaced or removed."""
    copy = json.loads(json.dumps(record))
    *parents, name = path.split(".")
    target = copy
    for parent in parents:
        target = target[parent]
    if value is MISSING:
        del target[name]
    else:
        target[name] = value
    return copy


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("run_id", MISSING, "run_id must be a string"),
        ("run_id", None, "run_id must be a string"),
        ("step_id", 7, "step_id must be a string"),
        ("run_folder", None, "run_folder must be a string"),
        ("batch_id", MISSING, "batch_id must be a string"),
        ("step_type", None, "step_type must be a string"),
        ("task_id", None, "task_id must be a string"),
        ("n", "1", "n must be an integer"),
        ("n", True, "n must be an integer"),
        ("n", MISSING, "n must be an integer"),
        ("phase", "warmup", "phase must be one of"),
        ("phase", ["ramp"], "phase must be one of"),
        ("configuration", "cfg-hardware", "the summary's is 'cfg-default'"),
        ("configuration", MISSING, "configuration is None"),
        ("truths", None, "truths must be an object"),
        ("truths.slower", MISSING, "truths must be an object"),
        ("truths.extra", "none", "truths must be an object"),
        ("truth_reasons", MISSING, "truth_reasons must be an object"),
        ("truth_reasons.extra", None, "truth_reasons must be an object"),
        ("truths.queue", "queue_free", "truths.queue is 'queue_free'"),
        ("truths.queue", ["none"], "truths.queue is"),
        ("truth_reasons.slower", None, "truths.slower is null without a reason"),
        ("truth_reasons.slower", "", "truths.slower is null without a reason"),
        ("truth_reasons.queue", "not_applicable", "truth_reasons.queue must be null"),
        ("duration_seconds", float("nan"), "duration_seconds must be a finite number"),
        ("queue_seconds", float("inf"), "queue_seconds must be a finite number"),
        ("execution_seconds", "0.5", "execution_seconds must be a finite number"),
        ("execution_seconds", True, "execution_seconds must be a finite number"),
        ("duration_seconds", MISSING, "duration_seconds must be a finite number"),
        ("plan_sha256", MISSING, "plan_sha256 is missing"),
        ("plan_sha256", "ab" * 31, "plan_sha256 must be null or 64 lowercase hex"),
        ("plan_sha256", "AB" * 32, "plan_sha256 must be null or 64 lowercase hex"),
        ("plan_sha256", 7, "plan_sha256 must be null or 64 lowercase hex"),
    ],
)
def test_a_malformed_truth_record_is_refused_naming_file_line_and_field(
    sweep, path, value, message
):
    records = sweep["records"]
    records[1] = changed(records[1], path, value)
    write_truths(sweep["truth"], records)
    with pytest.raises(HeldOutError, match=message) as refused:
        load_truths(sweep["truth"])
    assert "truth.jsonl" in str(refused.value) and "line 2" in str(refused.value)
    with pytest.raises(HeldOutError):
        held_out_rows([sweep["truth"]], sweep["root"])


@pytest.mark.parametrize("plan", [None, "0f" * 32])
def test_a_null_or_hex_plan_sha256_is_accepted(sweep, plan):
    records = sweep["records"]
    records[0] = records[0] | {"plan_sha256": plan}
    write_truths(sweep["truth"], records)
    _, loaded = load_truths(sweep["truth"])
    assert loaded[0]["plan_sha256"] == plan


def test_a_null_timing_is_accepted_as_unmeasured(sweep):
    records = sweep["records"]
    records[0] = records[0] | {"queue_seconds": None}
    write_truths(sweep["truth"], records)
    _, loaded = load_truths(sweep["truth"])
    assert loaded[0]["queue_seconds"] is None


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"configuration": "cfg-hardware"}, "not the accepted cfg-hardware protocol hash"),
        ({"configuration": "cfg-unknown"}, "not an accepted held-out configuration"),
        ({"configuration": None}, "not an accepted held-out configuration"),
        ({"configuration": ["cfg-default"]}, "not an accepted held-out configuration"),
        ({"sweep_id": None}, "sweep_id must be a non-empty string"),
        ({"steps": 4}, "steps is 4 but truth.jsonl holds 3 records"),
        ({"steps": None}, "steps is None"),
        ({"steps": True}, "steps is True"),
        ({"runs_without_plan": None}, "runs_without_plan must be a non-negative integer"),
        ({"runs_without_plan": -1}, "runs_without_plan must be a non-negative integer"),
        ({"runs_without_plan": True}, "runs_without_plan must be a non-negative integer"),
        ({"runs_without_evidence": "0"}, "runs_without_evidence must be a non-negative integer"),
        ({"excluded_batches": 0}, "excluded_batches must be a list"),
    ],
)
def test_a_malformed_summary_is_refused(sweep, change, message):
    write_summary(sweep["truth"], **change)
    with pytest.raises(HeldOutError, match=message):
        load_truths(sweep["truth"])


@pytest.mark.parametrize("name", ["steps", "runs_without_plan", "runs_without_evidence"])
def test_a_summary_without_a_counter_is_refused(sweep, name):
    summary = json.loads((sweep["truth"] / "truth.json").read_text())
    del summary[name]
    (sweep["truth"] / "truth.json").write_text(json.dumps(summary))
    with pytest.raises(HeldOutError, match=name):
        load_truths(sweep["truth"])


def test_a_summary_without_excluded_batches_is_refused(sweep):
    summary = json.loads((sweep["truth"] / "truth.json").read_text())
    del summary["excluded_batches"]
    (sweep["truth"] / "truth.json").write_text(json.dumps(summary))
    with pytest.raises(HeldOutError, match="excluded_batches must be a list"):
        load_truths(sweep["truth"])


def copy_truth(sweep, name: str, **summary_changes) -> Path:
    other = sweep["truth"].parent / name
    shutil.copytree(sweep["truth"], other)
    if summary_changes:
        write_summary(other, **summary_changes)
    return other


def test_the_same_truth_folder_twice_is_refused(sweep):
    with pytest.raises(HeldOutError, match="sweep_id 'synthetic'"):
        held_out_rows([sweep["truth"], sweep["truth"]], sweep["root"])


def test_two_truth_folders_of_one_sweep_folder_are_refused(sweep):
    other = copy_truth(sweep, "other", sweep_id="another")
    with pytest.raises(HeldOutError, match=f"sweep_folder '{SWEEP_FOLDER}'"):
        held_out_rows([sweep["truth"], other], sweep["root"])


def test_two_truth_folders_of_one_sweep_id_are_refused(sweep):
    other = copy_truth(sweep, "other", sweep_folder="cfg-default-elsewhere")
    with pytest.raises(HeldOutError, match="sweep_id 'synthetic'"):
        held_out_rows([sweep["truth"], other], sweep["root"])


def test_a_repeated_run_and_step_is_refused(sweep):
    records = sweep["records"]
    write_truths(sweep["truth"], [*records, records[1]])
    write_summary(sweep["truth"], steps=len(records) + 1)
    with pytest.raises(HeldOutError, match="model-call-1 appears more than once"):
        held_out_rows([sweep["truth"]], sweep["root"])


def test_a_run_and_step_repeated_across_sweeps_is_refused(sweep):
    other = copy_truth(sweep, "other", sweep_id="another", sweep_folder="cfg-default-elsewhere")
    with pytest.raises(HeldOutError, match="appears more than once"):
        held_out_rows([sweep["truth"], other], sweep["root"])


def test_the_accepted_label_version_is_the_packaged_rules():
    assert ACCEPTED_TRUTH_LABEL_VERSION == load_truth_rule().label_version


def test_a_summary_of_another_label_version_is_refused(sweep):
    write_summary(sweep["truth"], label_version="sweep-truth-v2+00000000")
    with pytest.raises(HeldOutError, match="label_version"):
        load_truths(sweep["truth"])


def test_a_summary_with_a_stale_truth_sha256_is_refused(sweep):
    # Rewritten after the summary, so the summary's hash describes the earlier bytes.
    (sweep["truth"] / "truth.jsonl").write_text(
        "".join(json.dumps(r | {"n": 2}) + "\n" for r in sweep["records"])
    )
    with pytest.raises(HeldOutError, match="truth_sha256"):
        load_truths(sweep["truth"])


def test_a_summary_without_a_truth_sha256_is_refused(sweep):
    summary = json.loads((sweep["truth"] / "truth.json").read_text())
    del summary["truth_sha256"]
    (sweep["truth"] / "truth.json").write_text(json.dumps(summary))
    with pytest.raises(HeldOutError, match="truth_sha256"):
        load_truths(sweep["truth"])


@pytest.mark.parametrize("field", ["steps_sha256", "steps_summary_sha256", "recorder_sha256"])
def test_a_summary_without_a_step_or_recorder_hash_is_refused(sweep, field):
    summary = json.loads((sweep["truth"] / "truth.json").read_text())
    del summary[field]
    (sweep["truth"] / "truth.json").write_text(json.dumps(summary))
    with pytest.raises(HeldOutError, match=field):
        load_truths(sweep["truth"])


@pytest.mark.parametrize("field", ["steps_sha256", "steps_summary_sha256", "recorder_sha256"])
@pytest.mark.parametrize(
    "malformed",
    [None, 7, "", "ab" * 31, "ab" * 33, "AB" * 32, "zz" * 32],
    ids=["null", "number", "empty", "short", "long", "uppercase", "not-hex"],
)
def test_a_summary_with_a_malformed_step_or_recorder_hash_is_refused(sweep, field, malformed):
    write_summary(sweep["truth"], **{field: malformed})
    with pytest.raises(HeldOutError, match=field):
        load_truths(sweep["truth"])


def test_unjudged_steps_are_kept_out_of_the_join(sweep):
    root = record(sweep["run_id"], "runs/no-journal", "agent-run") | {
        "step_id": "root-1",
        "truths": {"queue": None, "slower": None, "protocol_clean": None},
        "truth_reasons": {
            "queue": "root_step",
            "slower": "root_step",
            "protocol_clean": "root_step",
        },
    }
    write_truths(sweep["truth"], [*sweep["records"], root])
    result = held_out_rows([sweep["truth"]], sweep["root"])
    assert result.unjudged == [root]
    assert [row["step_id"] for row in result.rows] == list(STEP_SECONDS)
    assert result.refused == {}
