"""Per-step measurements of a capacity sweep. Every sweep here is synthetic: the stub driver's
folder with its run folders rewritten as fixtures. None of it is recorded evidence."""

import hashlib
import json
import re
import shutil
import statistics
from pathlib import Path

import pytest
from agentwatch_agent import sweep
from agentwatch_agent import sweep_steps as steps_module
from agentwatch_agent.sweep_offline import SweepFolderError
from agentwatch_agent.sweep_steps import (
    ACCEPTED_PROTOCOL_SHA256,
    RECORD_KEYS,
    RECORDER_MODULES,
    SPAN_STATUSES,
    STEPS_FILE,
    STEPS_FORMAT,
    SUMMARY_FILE,
    SUMMARY_KEYS,
    hash_recorder_sources,
    recorder_sha256,
    sweep_steps,
    write_steps,
)
from test_sweep import Harness
from test_sweep_labels import MODEL_STEP, PLAN, span_call, write_run

ROOT_STEP = "agent-run"
TOOL_STEP = "tool-1"
BASELINE_SECONDS = (0.9, 1.0, 1.1)  # mean 1.0, stdev 0.1 at N = 1
SLOW_SECONDS = 3.0
SLOW_QUEUE_SECONDS = 2.0


def read_record(sweep_dir):
    return json.loads((sweep_dir / sweep.SWEEP_FILE).read_text())


def write_record(sweep_dir, change):
    record = read_record(sweep_dir)
    change(record)
    (sweep_dir / sweep.SWEEP_FILE).write_text(json.dumps(record))


def set_parent(run_dir, step_id, parent):
    """Name `parent` as `step_id`'s parent in every journaled event of that step."""
    path = run_dir / "live-journal.jsonl"
    lines = []
    for line in path.read_text().splitlines():
        entry = json.loads(line)
        payload = entry["payload"]
        if payload.get("step_id") == step_id:
            payload["parent_step_id"] = parent
        lines.append(json.dumps(entry))
    path.write_text("\n".join(lines) + "\n")


def set_evidence_run_id(run_dir, run_id):
    """Make the run folder's own evidence name `run_id` (None removes it)."""
    path = run_dir / "agent-evidence.json"
    evidence = json.loads(path.read_text())
    if run_id is None:
        evidence.pop("run_id", None)
    else:
        evidence["run_id"] = run_id
    path.write_text(json.dumps(evidence))


def synthetic_sweep(tmp_path, monkeypatch, name="sweep"):
    """A stub-driver sweep whose every run folder holds a root step, a model step and a tool step.

    N = 1 ramp runs take 0.9, 1.0, 1.1 s (the baseline); every other run takes 3 s with 2 s of
    measured vLLM queue, except N = 1 runs outside the ramp, which take 1 s with none.
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
            write_run(
                run_dir,
                [
                    (ROOT_STEP, "other", seconds + 0.5, None),
                    (MODEL_STEP, "model_call", seconds, span_call(MODEL_STEP, queue=queue)),
                    (TOOL_STEP, "tool", 0.2, None),
                ],
            )
            set_parent(run_dir, MODEL_STEP, ROOT_STEP)
            set_parent(run_dir, TOOL_STEP, ROOT_STEP)
            set_evidence_run_id(run_dir, run["run_id"])
    return sweep_dir


def live_run_dir(sweep_dir):
    entry = next(e for e in read_record(sweep_dir)["batches"] if e["phase"] == "live")
    return sweep_dir / entry["folder"] / "runs" / "01"


def rewrite_live_run(sweep_dir, steps, plan=PLAN):
    """Replace the live run's folder with `steps`, keeping the run id `sweep.json` names."""
    entry = next(e for e in read_record(sweep_dir)["batches"] if e["phase"] == "live")
    run_dir = live_run_dir(sweep_dir)
    shutil.rmtree(run_dir)
    write_run(run_dir, steps, plan=plan)
    set_evidence_run_id(run_dir, entry["runs"][0]["run_id"])
    return run_dir


def steps_of(sweep_dir, run_dir):
    """The recorded steps of one run, by step id."""
    folder = run_dir.relative_to(sweep_dir).as_posix()
    _summary, records = sweep_steps(sweep_dir)
    return {r["step_id"]: r for r in records if r["run_folder"] == folder}


def test_lines_carry_exactly_the_record_keys(tmp_path, monkeypatch):
    _summary, records = sweep_steps(synthetic_sweep(tmp_path, monkeypatch))
    assert all(set(record) == RECORD_KEYS for record in records)
    assert {record["phase"] for record in records} == {"ramp", "live", "injected", "control"}


def test_root_step_has_null_parent_and_children_name_it(tmp_path, monkeypatch):
    sweep_dir = synthetic_sweep(tmp_path, monkeypatch)
    by_step = steps_of(sweep_dir, live_run_dir(sweep_dir))
    assert by_step[ROOT_STEP]["parent_step_id"] is None
    assert by_step[MODEL_STEP]["parent_step_id"] == ROOT_STEP
    assert by_step[TOOL_STEP]["parent_step_id"] == ROOT_STEP


def test_measured_span_keeps_vllm_seconds(tmp_path, monkeypatch):
    sweep_dir = synthetic_sweep(tmp_path, monkeypatch)
    model = steps_of(sweep_dir, live_run_dir(sweep_dir))[MODEL_STEP]
    assert model["span_status"] == "measured"
    assert model["vllm_queue_seconds"] == SLOW_QUEUE_SECONDS
    assert model["vllm_service_seconds"] == pytest.approx(0.1 + 0.2)


def test_unmeasured_span_has_null_vllm_seconds(tmp_path, monkeypatch):
    sweep_dir = synthetic_sweep(tmp_path, monkeypatch)
    run_dir = rewrite_live_run(
        sweep_dir,
        [
            (MODEL_STEP, "model_call", 1.0, span_call(MODEL_STEP, status="unavailable")),
            (TOOL_STEP, "tool", 0.2, None),
        ],
    )
    by_step = steps_of(sweep_dir, run_dir)
    model, tool = by_step[MODEL_STEP], by_step[TOOL_STEP]
    assert model["span_status"] == "unavailable"
    assert model["vllm_queue_seconds"] is None and model["vllm_service_seconds"] is None
    assert tool["span_status"] == "not_recorded"
    assert tool["vllm_queue_seconds"] is None and tool["vllm_service_seconds"] is None
    _summary, records = sweep_steps(sweep_dir)
    assert {r["span_status"] for r in records} <= set(SPAN_STATUSES)


def test_baseline_is_count_mean_stdev_or_null(tmp_path, monkeypatch):
    sweep_dir = synthetic_sweep(tmp_path, monkeypatch)
    rewrite_live_run(
        sweep_dir, [("other-step", "model_call", 1.0, span_call("other-step"))], plan="b" * 64
    )
    _summary, records = sweep_steps(sweep_dir)
    model = next(r for r in records if r["step_id"] == MODEL_STEP and r["n"] == 2)
    # Each N = 1 ramp batch holds two runs, so the three durations repeat twice.
    durations = BASELINE_SECONDS * 2
    assert model["baseline"] == {
        "count": len(durations),
        "mean_seconds": pytest.approx(statistics.fmean(durations)),
        "stdev_seconds": pytest.approx(statistics.stdev(durations)),
    }
    (other,) = [r for r in records if r["step_id"] == "other-step"]
    assert other["baseline"] is None


def test_no_threshold_or_truth_fields(tmp_path, monkeypatch):
    summary, records = sweep_steps(synthetic_sweep(tmp_path, monkeypatch))
    banned = ("threshold", "truth", "slower", "queued")
    keys = {key for record in records for key in record} | set(summary)
    assert not [key for key in keys if any(word in key for word in banned)]


def test_warmup_is_skipped_and_live_runs_count_as_without_plan(tmp_path, monkeypatch):
    sweep_dir = synthetic_sweep(tmp_path, monkeypatch)
    run_dir = rewrite_live_run(sweep_dir, [(MODEL_STEP, "model_call", 1.0, None)], plan=None)
    summary, records = sweep_steps(sweep_dir)
    assert "warmup" not in {r["phase"] for r in records}
    assert min(r["batch_sequence"] for r in records) > 0
    assert summary["runs_without_plan"] == 1
    assert summary["excluded_batches"] == []
    assert steps_of(sweep_dir, run_dir) == {}


def test_summary_fields(tmp_path, monkeypatch):
    sweep_dir = synthetic_sweep(tmp_path, monkeypatch)
    summary, records = sweep_steps(sweep_dir)
    record = read_record(sweep_dir)
    # write_steps adds steps_sha256 once the records are serialised.
    assert set(summary) == SUMMARY_KEYS - {"steps_sha256"}
    assert SUMMARY_KEYS == {
        "format",
        "sweep_folder",
        "sweep_id",
        "configuration",
        "protocol_version",
        "protocol_sha256",
        "sweep_json_sha256",
        "recorder_sha256",
        "held_out",
        "steps",
        "steps_sha256",
        "excluded_batches",
        "runs_without_evidence",
        "runs_without_plan",
    }
    assert summary["format"] == STEPS_FORMAT == "sweep-steps-v1"
    assert summary["held_out"] is True
    assert summary["steps"] == len(records)
    assert summary["sweep_folder"] == sweep_dir.name
    assert summary["sweep_id"] == record["sweep_id"]
    assert summary["configuration"] == "cfg-default"
    assert summary["protocol_sha256"] == record["protocol_sha256"]
    assert (
        summary["sweep_json_sha256"]
        == hashlib.sha256((sweep_dir / sweep.SWEEP_FILE).read_bytes()).hexdigest()
    )
    first = records[0]
    assert first["plan_sha256"] == PLAN
    assert first["run_folder"].startswith("batches/") and "/runs/" in first["run_folder"]
    assert summary["recorder_sha256"] == recorder_sha256()


def test_recorder_sha256_changes_with_the_recorder_source():
    sources = {"sweep_steps.py": "a = 1\n", "sweep_labels.py": "b = 2\n"}
    digest = hash_recorder_sources(sources)
    assert re.fullmatch(r"[0-9a-f]{64}", digest)
    assert hash_recorder_sources(dict(sources)) == digest
    changed = {**sources, "sweep_labels.py": "b = 3\n"}
    assert hash_recorder_sources(changed) != digest
    crlf = {name: text.replace("\n", "\r\n") for name, text in sources.items()}
    assert hash_recorder_sources(crlf) == digest


def test_recorder_sha256_hashes_the_two_recorder_modules():
    assert RECORDER_MODULES == ("sweep_steps.py", "sweep_labels.py")
    package = Path(steps_module.__file__).parent
    sources = {name: (package / name).read_text(encoding="utf-8") for name in RECORDER_MODULES}
    assert recorder_sha256() == hash_recorder_sources(sources)


def test_written_files_are_lf_sorted_and_reproducible(tmp_path, monkeypatch):
    sweep_dir = synthetic_sweep(tmp_path, monkeypatch)
    first = write_steps(sweep_dir, tmp_path / "out-1")
    second = write_steps(sweep_dir, tmp_path / "out-2")
    _summary, records = sweep_steps(sweep_dir)
    for name in (STEPS_FILE, SUMMARY_FILE):
        raw = (first / name).read_bytes()
        assert raw == (second / name).read_bytes()
        assert b"\r" not in raw and raw.endswith(b"\n")
    lines = (first / STEPS_FILE).read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == records
    assert all(
        line == json.dumps(json.loads(line), sort_keys=True, allow_nan=False) for line in lines
    )
    written = json.loads((first / SUMMARY_FILE).read_text(encoding="utf-8"))
    assert set(written) == SUMMARY_KEYS
    assert written["steps"] == len(records)
    assert written["steps_sha256"] == hashlib.sha256((first / STEPS_FILE).read_bytes()).hexdigest()


def test_the_accepted_hashes_are_the_frozen_held_out_protocols():
    assert steps_module.ACCEPTED_PROTOCOL_SHA256 == {
        "cfg-default": "22e26dc7536fe5e88e3d9ea9f01445a754788ce587b1626d6a4eb055ec4524b9",
        "cfg-hardware": "1608deee2a3bcb5245b07d73e13e73dcd1e82702ef14aa683ecd94144e0f18cd",
    }


def break_evidence(sweep_dir):
    (live_run_dir(sweep_dir) / "agent-evidence.json").write_text("{broken")


def escape_batch(sweep_dir):
    write_record(
        sweep_dir,
        lambda r: next(b for b in r["batches"] if b["phase"] == "live").update(folder="../x"),
    )


def drop_batch_runs(sweep_dir):
    write_record(
        sweep_dir,
        lambda r: next(b for b in r["batches"] if b["phase"] == "live").pop("runs"),
    )


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r["protocol"].update(version="other"),
        lambda r: r["protocol"]["configuration"].update(name="cfg-unknown"),
        lambda r: r.update(protocol_sha256="0" * 64),
        lambda r: r["protocol"].update(label_rule={"changed": True}),
        lambda r: r.update(held_out=False),
        lambda r: r.pop("held_out"),
    ],
    ids=[
        "other-version",
        "unknown-configuration",
        "unknown-hash",
        "hash-not-of-protocol",
        "held-out-false",
        "held-out-missing",
    ],
)
def test_a_refused_sweep_writes_nothing(tmp_path, monkeypatch, change):
    sweep_dir = synthetic_sweep(tmp_path, monkeypatch)
    write_record(sweep_dir, change)
    output = tmp_path / "out"
    with pytest.raises(SweepFolderError):
        write_steps(sweep_dir, output)
    assert not output.exists()


@pytest.mark.parametrize("damage", [break_evidence, escape_batch, drop_batch_runs])
def test_a_damaged_sweep_writes_nothing(tmp_path, monkeypatch, damage):
    sweep_dir = synthetic_sweep(tmp_path, monkeypatch)
    damage(sweep_dir)
    output = tmp_path / "out"
    with pytest.raises(SweepFolderError):
        write_steps(sweep_dir, output)
    assert not output.exists()


@pytest.mark.parametrize("evidence_run_id", ["00000000-0000-4000-8000-0000000000ff", None])
def test_a_run_id_that_disagrees_with_its_evidence_writes_nothing(
    tmp_path, monkeypatch, evidence_run_id
):
    sweep_dir = synthetic_sweep(tmp_path, monkeypatch)
    run_dir = live_run_dir(sweep_dir)
    set_evidence_run_id(run_dir, evidence_run_id)
    output = tmp_path / "out"
    with pytest.raises(SweepFolderError, match=run_dir.relative_to(sweep_dir).as_posix()):
        write_steps(sweep_dir, output)
    assert not output.exists()


def test_the_sweep_folder_is_the_real_name_when_given_as_a_dot(tmp_path, monkeypatch):
    sweep_dir = synthetic_sweep(tmp_path, monkeypatch, name="the-real-sweep")
    monkeypatch.chdir(sweep_dir)
    summary, _records = sweep_steps(Path("."))
    assert summary["sweep_folder"] == "the-real-sweep"


def test_output_inside_held_out_is_refused(tmp_path, monkeypatch):
    sweep_dir = synthetic_sweep(tmp_path, monkeypatch)
    output = tmp_path / "data" / "held-out" / "steps"
    with pytest.raises(SweepFolderError, match="held-out"):
        write_steps(sweep_dir, output)
    assert not output.exists()


def test_output_resolving_into_held_out_is_refused(tmp_path, monkeypatch):
    sweep_dir = synthetic_sweep(tmp_path, monkeypatch)
    output = tmp_path / "x" / ".." / "data" / "held-out" / "out"
    with pytest.raises(SweepFolderError, match="held-out"):
        write_steps(sweep_dir, output)
    assert not (tmp_path / "data").exists()
