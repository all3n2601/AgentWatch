"""Per-step truths for a capacity sweep. Every sweep here is synthetic: the stub driver's folder
with its run folders rewritten as fixtures. None of it is recorded evidence."""

import json
import shutil
from collections import Counter

import pytest
from agentwatch_agent import sweep
from agentwatch_agent.sweep_offline import SweepFolderError
from agentwatch_agent.sweep_truth import (
    ACCEPTED_PROTOCOL_SHA256,
    SUMMARY_FILE,
    TRUTH_FILE,
    TRUTH_METHOD,
    sweep_truths,
    write_truths,
)
from test_sweep import Harness
from test_sweep_labels import MODEL_STEP, PLAN, span_call, write_run

BASELINE_SECONDS = (0.9, 1.0, 1.1)  # mean 1.0, stdev 0.1 at N = 1
SLOW_SECONDS = 3.0
SLOW_QUEUE_SECONDS = 2.0
RECORD_KEYS = {
    "run_id",
    "run_folder",
    "step_id",
    "step_type",
    "plan_sha256",
    "task_id",
    "configuration",
    "n",
    "phase",
    "batch_sequence",
    "batch_id",
    "state",
    "duration_seconds",
    "queue_seconds",
    "execution_seconds",
    "baseline_mean_seconds",
    "noise_threshold_seconds",
    "excess_seconds",
    "vllm_queue_seconds",
    "truths",
    "truth_reasons",
}


def read_record(sweep_dir):
    return json.loads((sweep_dir / sweep.SWEEP_FILE).read_text())


def write_record(sweep_dir, change):
    record = read_record(sweep_dir)
    change(record)
    (sweep_dir / sweep.SWEEP_FILE).write_text(json.dumps(record))


def synthetic_sweep(tmp_path, monkeypatch, name="sweep"):
    """A stub-driver sweep whose every run folder holds one synthetic model step.

    N = 1 ramp runs take 0.9, 1.0, 1.1 s with no queue (the baseline); every other run takes
    3 s with 2 s of measured vLLM queue, except injected and control runs at N = 1, which take
    the baseline mean.
    """
    # The declared 2 s injector ramp is a real wait; tests do not sleep.
    monkeypatch.setattr(sweep, "INJECTOR_RAMP_SECONDS", 0.0)
    harness = Harness(tmp_path, output_dir=tmp_path / name)
    harness.run()
    sweep_dir = harness.output_dir
    # A synthetic sweep carries today's protocol hash, not a frozen held-out sweep's.
    monkeypatch.setitem(
        ACCEPTED_PROTOCOL_SHA256, "cfg-default", read_record(sweep_dir)["protocol_sha256"]
    )
    ramp_one = 0
    for entry in read_record(sweep_dir)["batches"]:
        for run in entry["runs"]:
            run_dir = sweep_dir / entry["folder"] / "runs" / f"{run['index']:02d}"
            shutil.rmtree(run_dir)
            baseline_like = entry["level"] == 1
            if entry["phase"] == "ramp" and baseline_like:
                seconds = BASELINE_SECONDS[ramp_one % 3]
                ramp_one += 1
            else:
                seconds = 1.0 if baseline_like else SLOW_SECONDS
            queue = SLOW_QUEUE_SECONDS if seconds == SLOW_SECONDS else 0.0
            call = span_call(MODEL_STEP, queue=queue)
            write_run(run_dir, [(MODEL_STEP, "model_call", seconds, call)])
    return sweep_dir


def test_truths_cover_every_scored_phase_and_skip_warmup(tmp_path, monkeypatch):
    sweep_dir = synthetic_sweep(tmp_path, monkeypatch)
    summary, records = sweep_truths(sweep_dir)
    assert {record["phase"] for record in records} == {"ramp", "live", "injected", "control"}
    assert all(set(record) == RECORD_KEYS for record in records)
    assert summary["method"] == TRUTH_METHOD
    assert summary["held_out"] is True
    assert summary["sweep_folder"] == sweep_dir.name
    assert summary["steps"] == len(records)
    assert summary["sweep_id"] == read_record(sweep_dir)["sweep_id"]
    assert summary["protocol_sha256"] == read_record(sweep_dir)["protocol_sha256"]
    first = records[0]
    assert (first["configuration"], first["plan_sha256"], first["step_id"]) == (
        "cfg-default",
        PLAN,
        MODEL_STEP,
    )
    assert first["run_folder"].startswith("batches/") and "/runs/" in first["run_folder"]
    # The warm-up batch is skipped, not counted as excluded.
    assert summary["excluded_batches"] == []
    assert min(record["batch_sequence"] for record in records) > 0


def test_truth_classes_and_reasons_follow_the_label_rule(tmp_path, monkeypatch):
    summary, records = sweep_truths(synthetic_sweep(tmp_path, monkeypatch))
    slow = next(r for r in records if r["phase"] == "ramp" and r["n"] == 2)
    assert slow["truths"] == {"queue": "inference", "slower": "inference", "protocol_clean": None}
    assert slow["truth_reasons"] == {
        "queue": None,
        "slower": None,
        "protocol_clean": "not_applicable",
    }
    assert slow["vllm_queue_seconds"] == SLOW_QUEUE_SECONDS
    quiet = next(r for r in records if r["phase"] == "control")
    assert quiet["truths"]["queue"] == quiet["truths"]["slower"] == "none"
    assert summary["counts"]["queue"]["classes"] == dict(
        Counter(r["truths"]["queue"] for r in records)
    )
    assert summary["counts"]["protocol_clean"]["reasons"] == {
        "not_applicable": sum(1 for r in records if r["truths"]["protocol_clean"] is None)
    }


def test_a_step_without_a_baseline_keeps_its_reason_not_a_class(tmp_path, monkeypatch):
    sweep_dir = synthetic_sweep(tmp_path, monkeypatch)
    entry = next(e for e in read_record(sweep_dir)["batches"] if e["phase"] == "live")
    run_dir = sweep_dir / entry["folder"] / "runs" / "01"
    shutil.rmtree(run_dir)
    write_run(run_dir, [("other-step", "model_call", 1.0, span_call("other-step"))], plan="b" * 64)
    _summary, records = sweep_truths(sweep_dir)
    (record,) = [r for r in records if r["step_id"] == "other-step"]
    assert record["truths"]["queue"] is None and record["truths"]["slower"] is None
    assert record["truth_reasons"]["queue"] == record["truth_reasons"]["slower"] == "no_baseline"
    assert record["baseline_mean_seconds"] is None


def test_protocol_clean_is_none_only_for_one_agent_ramp_and_control(tmp_path, monkeypatch):
    _summary, records = sweep_truths(synthetic_sweep(tmp_path, monkeypatch))
    by_condition = {}
    for record in records:
        by_condition.setdefault((record["phase"], record["n"]), set()).add(
            (record["truths"]["protocol_clean"], record["truth_reasons"]["protocol_clean"])
        )
    assert by_condition[("ramp", 1)] == {("none", None)}
    assert by_condition[("control", 1)] == {("none", None)}
    assert by_condition[("injected", 1)] == {(None, "not_applicable")}
    assert by_condition[("ramp", 2)] == {(None, "not_applicable")}
    assert by_condition[("live", 2)] == {(None, "not_applicable")}


def test_control_at_a_higher_level_is_not_protocol_clean(tmp_path, monkeypatch):
    sweep_dir = synthetic_sweep(tmp_path, monkeypatch)
    write_record(
        sweep_dir,
        lambda r: [b.update(level=4) for b in r["batches"] if b["phase"] == "control"],
    )
    _summary, records = sweep_truths(sweep_dir)
    control = [r for r in records if r["phase"] == "control"]
    assert {r["n"] for r in control} == {4}
    assert {r["truths"]["protocol_clean"] for r in control} == {None}


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r["protocol"].update(version="other"),
        lambda r: r.update(protocol_sha256="0" * 64),
        lambda r: r.update(held_out=False),
        lambda r: r.pop("held_out"),
        lambda r: r["protocol"]["configuration"].update(name="cfg-unknown"),
    ],
)
def test_a_refused_sweep_writes_nothing(tmp_path, monkeypatch, change):
    sweep_dir = synthetic_sweep(tmp_path, monkeypatch)
    write_record(sweep_dir, change)
    output = tmp_path / "out"
    with pytest.raises(SweepFolderError):
        sweep_truths(sweep_dir)
    with pytest.raises(SweepFolderError):
        write_truths(sweep_dir, output)
    assert not output.exists()


def test_runs_resolve_inside_a_copied_sweep(tmp_path, monkeypatch):
    sweep_dir = synthetic_sweep(tmp_path, monkeypatch)
    _before, expected = sweep_truths(sweep_dir)

    def relative(record):
        for entry in record["batches"]:
            for run in entry["runs"]:
                run["output_dir"] = f".data/elsewhere/{entry['folder']}/runs/{run['index']:02d}"

    write_record(sweep_dir, relative)
    copy = tmp_path / "copies" / "renamed"
    shutil.copytree(sweep_dir, copy)
    shutil.rmtree(sweep_dir)
    elsewhere = tmp_path / "cwd"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    summary, records = sweep_truths(copy)
    assert summary["sweep_folder"] == "renamed"
    assert summary["runs_without_evidence"] == 0
    assert [r["run_folder"] for r in records] == [r["run_folder"] for r in expected]
    assert all((copy / r["run_folder"] / "agent-evidence.json").is_file() for r in records)


def test_excluded_ramp_batches_are_counted_not_scored(tmp_path, monkeypatch):
    sweep_dir = synthetic_sweep(tmp_path, monkeypatch)
    _all, full = sweep_truths(sweep_dir)
    cause = {"state": "failed", "reason": "synthetic exclusion"}

    def exclude(record):
        entry = next(b for b in record["batches"] if b["phase"] == "ramp" and b["level"] == 2)
        entry["metrics_exclusion"] = {"causes": [cause]}

    write_record(sweep_dir, exclude)
    summary, records = sweep_truths(sweep_dir)
    (excluded,) = summary["excluded_batches"]
    assert (excluded["level"], excluded["causes"]) == (2, [cause])
    assert excluded["sequence"] not in {r["batch_sequence"] for r in records}
    assert len(records) == len(full) - 2


def test_runs_without_evidence_are_counted_and_not_recorded(tmp_path, monkeypatch):
    sweep_dir = synthetic_sweep(tmp_path, monkeypatch)
    _all, full = sweep_truths(sweep_dir)
    entry = next(b for b in read_record(sweep_dir)["batches"] if b["phase"] == "live")
    (sweep_dir / entry["folder"] / "runs" / "01" / "live-journal.jsonl").unlink()
    shutil.rmtree(sweep_dir / entry["folder"] / "runs" / "02")
    summary, records = sweep_truths(sweep_dir)
    assert summary["runs_without_evidence"] == 2
    assert len(records) == len(full) - 2


def test_unreadable_evidence_refuses_the_sweep(tmp_path, monkeypatch):
    sweep_dir = synthetic_sweep(tmp_path, monkeypatch)
    entry = next(b for b in read_record(sweep_dir)["batches"] if b["phase"] == "live")
    (sweep_dir / entry["folder"] / "runs" / "01" / "agent-evidence.json").write_text("{broken")
    with pytest.raises(SweepFolderError, match="batches/28-live-n2-r0/runs/01"):
        sweep_truths(sweep_dir)


def test_a_scored_batch_outside_the_sweep_folder_is_refused(tmp_path, monkeypatch):
    sweep_dir = synthetic_sweep(tmp_path, monkeypatch)
    write_record(
        sweep_dir,
        lambda r: next(b for b in r["batches"] if b["phase"] == "live").update(folder="../x"),
    )
    with pytest.raises(SweepFolderError, match="outside the sweep"):
        sweep_truths(sweep_dir)


def test_output_inside_data_held_out_is_refused(tmp_path, monkeypatch):
    sweep_dir = synthetic_sweep(tmp_path, monkeypatch)
    output = tmp_path / "data" / "held-out" / "truth"
    with pytest.raises(SweepFolderError, match="held-out"):
        write_truths(sweep_dir, output)
    assert not output.exists()


def test_both_files_are_lf_sorted_and_reproducible(tmp_path, monkeypatch):
    sweep_dir = synthetic_sweep(tmp_path, monkeypatch)
    first = write_truths(sweep_dir, tmp_path / "out-1")
    second = write_truths(sweep_dir, tmp_path / "out-2")
    _summary, records = sweep_truths(sweep_dir)
    for name in (TRUTH_FILE, SUMMARY_FILE):
        raw = (first / name).read_bytes()
        assert raw == (second / name).read_bytes()
        assert b"\r" not in raw and raw.endswith(b"\n")
    lines = (first / TRUTH_FILE).read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == records
    assert all(
        line == json.dumps(json.loads(line), sort_keys=True, allow_nan=False) for line in lines
    )
    assert json.loads((first / SUMMARY_FILE).read_text(encoding="utf-8"))["steps"] == len(records)


def test_the_accepted_hashes_are_the_frozen_held_out_protocols():
    assert ACCEPTED_PROTOCOL_SHA256 == {
        "cfg-default": "22e26dc7536fe5e88e3d9ea9f01445a754788ce587b1626d6a4eb055ec4524b9",
        "cfg-hardware": "1608deee2a3bcb5245b07d73e13e73dcd1e82702ef14aa683ecd94144e0f18cd",
    }


def test_baseline_runs_keep_queue_but_have_no_slower_truth(tmp_path, monkeypatch):
    _summary, records = sweep_truths(synthetic_sweep(tmp_path, monkeypatch))
    baseline = [r for r in records if r["phase"] == "ramp" and r["n"] == 1]
    assert baseline
    for record in baseline:
        assert record["truths"]["slower"] is None
        assert record["truth_reasons"]["slower"] == "not_applicable"
        assert record["truths"]["queue"] == "none"
        assert record["truths"]["protocol_clean"] == "none"
    control = next(r for r in records if r["phase"] == "control" and r["n"] == 1)
    assert control["truths"]["slower"] == "none" and control["truth_reasons"]["slower"] is None
    injected = next(r for r in records if r["phase"] == "injected")
    assert injected["truths"]["slower"] is not None
