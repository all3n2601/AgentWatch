"""The offline sweep analyser over sweep folders the stub driver recorded; never real evidence."""

import json

import pytest
from agentwatch_agent import sweep, sweep_analysis
from agentwatch_agent.sweep_offline import (
    ANALYSIS_FILE,
    SweepFolderError,
    analyse_sweep_folder,
)
from test_sweep import Harness, cancel_sweep, read


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

    def broken(batch_dir, run_timeout_seconds):
        if batch_dir.name == "07-ramp-n2-r0":
            raise ZeroDivisionError("fixture: analysis bug")
        return real(batch_dir, run_timeout_seconds)

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
