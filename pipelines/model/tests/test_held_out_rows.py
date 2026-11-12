"""Held-out feature rows: truth records joined to features from synthetic sweep run folders.

The sweeps here are synthetic (the data tests' replay journals copied into a sweep layout with
invented truth values); none of it is recorded held-out evidence.
"""

import json
import shutil
import sys
from pathlib import Path

import pytest
from agentwatch_data.schema import feature_columns

# The synthetic live journals are written by the data tests' fixtures.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "data" / "tests"))

from agentwatch_model.held_out_rows import (
    ACCEPTED_PROTOCOL_SHA256,
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
        "duration_seconds": timing["total_seconds"],
        "queue_seconds": timing["queue_seconds"],
        "execution_seconds": timing["execution_seconds"],
        "truths": {"queue": "queue_free", "slower": "not_slower", "protocol_clean": "clean"},
        "run_folder": run_folder,
    }
    return base | overrides


def write_summary(directory: Path, **overrides) -> dict:
    summary = {
        "method": "sweep-truth-v1",
        "sweep_id": "synthetic",
        "configuration": "cfg-default",
        "protocol_version": "synthetic",
        "protocol_sha256": ACCEPTED_PROTOCOL_SHA256["cfg-default"],
        "held_out": True,
        "sweep_folder": SWEEP_FOLDER,
    } | overrides
    (directory / "truth.json").write_text(json.dumps(summary))
    return summary


def write_truths(directory: Path, records: list[dict]) -> None:
    (directory / "truth.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))


@pytest.fixture
def sweep(tmp_path):
    """sweep_root with one run folder (runs/a) and a truth folder naming its three steps."""
    synthetic = write_experiment(tmp_path / "source", live=True)
    run_dir = tmp_path / "sweeps" / SWEEP_FOLDER / "runs" / "a"
    shutil.copytree(synthetic / "replays" / "01", run_dir)
    evidence = json.loads((run_dir / "agent-evidence.json").read_text())
    truth_dir = tmp_path / "truth"
    truth_dir.mkdir()
    write_summary(truth_dir)
    records = [record(evidence["run_id"], "runs/a", step) for step in STEP_SECONDS]
    write_truths(truth_dir, records)
    return {
        "root": tmp_path / "sweeps",
        "truth": truth_dir,
        "run_dir": run_dir,
        "run_id": evidence["run_id"],
        "records": records,
    }


def test_held_out_rows_join_truth_and_features(sweep):
    rows, refused, summaries = held_out_rows([sweep["truth"]], sweep["root"])
    assert refused == {}
    assert len(summaries) == 1
    assert [row["step_id"] for row in rows] == list(STEP_SECONDS)
    for row in rows:
        assert set(feature_columns()) <= set(row)
        assert row["truths"]["queue"] == "queue_free"
        assert row["telemetry_status"]
        assert row["worker_capacity"] is None
        assert row["queue_seconds"] == pytest.approx(0.001)
    assert rows[0]["execution_seconds"] == pytest.approx(0.999)


def test_a_timing_mismatch_refuses_only_that_step(sweep):
    records = sweep["records"]
    records[1] = records[1] | {"duration_seconds": 9.0}
    write_truths(sweep["truth"], records)
    rows, refused, _ = held_out_rows([sweep["truth"]], sweep["root"])
    assert refused == {"feature_timing_mismatch": 1}
    assert [row["step_id"] for row in rows] == ["agent-run", "tool-call-1"]


def test_a_journal_of_another_run_refuses_that_run(sweep):
    other = "00000000-0000-4000-8000-0000000000ff"
    write_truths(sweep["truth"], [r | {"run_id": other} for r in sweep["records"]])
    rows, refused, _ = held_out_rows([sweep["truth"]], sweep["root"])
    assert rows == []
    assert refused == {"feature_row_missing": 3}


def test_a_missing_journal_refuses_that_run(sweep):
    (sweep["run_dir"] / "live-journal.jsonl").unlink()
    rows, refused, _ = held_out_rows([sweep["truth"]], sweep["root"])
    assert rows == []
    assert refused == {"feature_evidence_unreadable": 3}


@pytest.mark.parametrize("run_folder", ["/etc", "C:/x", "../escape", "runs/../../x"])
def test_a_run_folder_outside_the_sweep_is_refused(sweep, run_folder):
    write_truths(sweep["truth"], [r | {"run_folder": run_folder} for r in sweep["records"]])
    rows, refused, _ = held_out_rows([sweep["truth"]], sweep["root"])
    assert rows == []
    assert refused == {"feature_evidence_unreadable": 3}


@pytest.mark.parametrize(
    "change",
    [
        {"held_out": False},
        {"held_out": None},
        {"method": "sweep-truth-v0"},
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
