"""The offline sweep analyser over sweep folders the stub driver recorded; never real evidence."""

import json
import shutil

import pytest
from agentwatch_agent import cli, sweep, sweep_analysis
from agentwatch_agent.sweep_offline import (
    ANALYSIS_FILE,
    SweepFolderError,
    analyse_sweep_folder,
)
from test_sweep import CACHE_CONFIG, Harness, cancel_sweep, read


def json_round_trip(value):
    return json.loads(json.dumps(value))


def test_the_offline_analysis_reproduces_the_recorded_one_beside_it(tmp_path):
    harness = Harness(tmp_path)
    record = harness.run()
    before = harness.sweep_file.read_bytes()
    document = analyse_sweep_folder(harness.output_dir)
    # sweep.json is never modified.
    assert harness.sweep_file.read_bytes() == before
    assert read(harness.output_dir / ANALYSIS_FILE) == json_round_trip(document)
    assert document["sweep_id"] == record["sweep_id"]
    assert document["protocol_sha256_matches"] is True
    assert document["analysis_scope"] == "full_ramp"
    # Recomputed from the batch folders, it matches what the sweep recorded.
    for key in ("level_summaries", "capacity_decision", "excluded_batches", "label_validation"):
        assert json_round_trip(document[key]) == json_round_trip(record[key])
    # Level keys are ints while analysing: N = 1 is never validated against itself.
    assert set(document["level_summaries"]) == set(record["level_summaries"])
    assert 1 not in document["label_validation"]["levels"]


def test_a_failed_sweep_is_analysed_offline_with_its_cut_short_batch_excluded(tmp_path):
    harness = Harness(tmp_path)

    def runner_error(_stub, entries, wall):
        return wall, [ValueError("fixture: runner bug")] + [None] * (len(entries) - 1)

    harness.hooks["ramp-n2-r1"] = runner_error
    with pytest.raises(ValueError, match="runner bug"):
        harness.run()
    recorded = read(harness.sweep_file)
    assert recorded["state"] == "failed" and recorded["capacity_decision"] is None
    document = analyse_sweep_folder(harness.output_dir)
    assert document["sweep_state"] == "failed"
    assert document["analysis_scope"] == "partial_ramp"
    (excluded,) = document["excluded_batches"]
    assert (excluded["sequence"], excluded["causes"][0]["state"]) == (8, "failed")
    decision = document["capacity_decision"]
    assert decision["result"] == "incomplete" and decision["capacity"] == 1
    assert set(document["label_validation"]["levels"]) == {2}


def test_a_batch_whose_analysis_raises_is_recorded_offline(tmp_path, monkeypatch):
    harness = Harness(tmp_path)
    harness.run()
    real = sweep_analysis.batch_metrics

    def broken(batch_dir, run_timeout_seconds, **options):
        if batch_dir.name == "07-ramp-n2-r0":
            raise ZeroDivisionError("fixture: analysis bug")
        return real(batch_dir, run_timeout_seconds, **options)

    monkeypatch.setattr(sweep_analysis, "batch_metrics", broken)
    document = analyse_sweep_folder(harness.output_dir)
    assert document["level_summaries"][2]["analysis_errors"] == 1
    assert document["capacity_decision"]["criteria"][2]["readable"]["passed"] is False


def test_a_cancelled_sweep_is_analysed_as_a_partial_ramp(tmp_path):
    harness = Harness(tmp_path)
    harness.hooks["ramp-n3-r0"] = lambda *_: cancel_sweep(harness)
    harness.run()
    document = analyse_sweep_folder(harness.output_dir)
    assert document["analysis_scope"] == "partial_ramp"
    assert [entry["sequence"] for entry in document["excluded_batches"]] == [10]


@pytest.mark.parametrize(
    "content", [None, "{not json", "[]", '{"method": "other"}', '{"method": "capacity-sweep-v1"}']
)
def test_a_folder_without_a_usable_sweep_record_is_refused(tmp_path, content):
    if content is not None:
        (tmp_path / sweep.SWEEP_FILE).write_text(content)
    with pytest.raises(SweepFolderError):
        analyse_sweep_folder(tmp_path)
    assert not (tmp_path / ANALYSIS_FILE).exists()


def test_a_copied_sweep_is_analysed_from_its_own_folder_only(tmp_path):
    """Bug hunt: a copy read run folders from the original sweep, which still existed."""
    harness = Harness(tmp_path)
    harness.run()
    copy = tmp_path / "copy"
    shutil.copytree(harness.output_dir, copy)
    entry = next(e for e in read(copy / sweep.SWEEP_FILE)["batches"] if e["sequence"] == 7)
    removed = copy / entry["folder"] / "runs" / "01"
    shutil.rmtree(removed)
    assert (harness.output_dir / entry["folder"] / "runs" / "01").is_dir()
    document = analyse_sweep_folder(copy)
    level = document["level_summaries"][2]
    # The removed run is missing from the copy, never read from the original.
    assert level["run_duration_seconds"]["missing"] == 1
    assert level["runs_without_evidence"] == 1
    criterion = document["capacity_decision"]["criteria"][2]["evidence_complete"]
    assert criterion["passed"] is False


def write_record(harness, change):
    record = read(harness.sweep_file)
    change(record)
    harness.sweep_file.write_text(json.dumps(record))


def first_ramp(record):
    return next(entry for entry in record["batches"] if entry["phase"] == "ramp")


@pytest.mark.parametrize(
    "change",
    [
        # The re-review's probe: a ramp entry whose server reading is null.
        lambda r: first_ramp(r).update(server=None),
        lambda r: first_ramp(r).update(server="read"),
        lambda r: first_ramp(r)["server"].update(changes=[1, 2]),
        lambda r: first_ramp(r).update(metrics_exclusion={"why": "x"}),
        lambda r: first_ramp(r).update(metrics_exclusion={"causes": "x"}),
        lambda r: first_ramp(r).update(metrics_exclusion={"causes": [{"reason": "x"}]}),
        lambda r: r["hard_stops"].append({"level": 3, "phase": "ramp", "code": None}),
        lambda r: r.update(phases=None),
        lambda r: r["phases"].update(warmup="ran"),
    ],
)
def test_a_malformed_record_is_refused_with_a_clear_error(tmp_path, change, capsys):
    harness = Harness(tmp_path)
    harness.run()
    write_record(harness, change)
    with pytest.raises(SweepFolderError):
        analyse_sweep_folder(harness.output_dir)
    assert not (harness.output_dir / ANALYSIS_FILE).exists()
    # The command exits 2 with the message, never a traceback.
    assert cli.main(["sweep-analyse", str(harness.output_dir)]) == cli.EXIT_USAGE
    assert "error:" in capsys.readouterr().err


@pytest.mark.parametrize("how", ["refused", "models_unreadable"])
def test_a_sweep_that_never_reached_the_ramp_has_no_analysis_scope(tmp_path, how):
    """Re-review: offline called a sweep that never ran a batch a partial ramp."""
    if how == "refused":
        harness = Harness(tmp_path, cache_config=CACHE_CONFIG | {"gpu_memory_utilization": "0.9"})
    else:
        harness = Harness(tmp_path)

        def unreachable():
            raise OSError("fixture: connection refused")

        harness.models = unreachable
    record = harness.run()
    assert record["analysis_scope"] == "none"
    document = analyse_sweep_folder(harness.output_dir)
    assert document["analysis_scope"] == "none"
