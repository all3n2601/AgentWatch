import json
from pathlib import Path

import pytest
from agentwatch_data import replay_evidence
from agentwatch_data.replay_evidence import ReplayEvidenceError, read_experiment
from replay_fixtures import write_experiment

MIXED_STEPS = {"agent-run": 1.5, "model-call-1": 1.0, "tool-call-1": 0.2}


def synthetic(tmp_path, **kwargs):
    kwargs.setdefault("contended", {"cpu": MIXED_STEPS, "mixed": MIXED_STEPS})
    return write_experiment(tmp_path, **kwargs)


def edit_manifest(directory, change):
    path = directory / "experiment.json"
    manifest = json.loads(path.read_text())
    change(manifest)
    path.write_text(json.dumps(manifest))


def test_reads_synthetic_experiment(tmp_path):
    experiment = read_experiment(synthetic(tmp_path))

    assert experiment.name == "synthetic-exp"
    assert experiment.task_id == "fix-a-r1"
    assert experiment.status == "completed"
    roles = [replay.role for replay in experiment.replays]
    assert roles == ["warmup"] + ["baseline"] * 3 + ["contended", "clean_control"] * 2
    mixed = experiment.replays[-2]
    assert mixed.condition == "mixed"
    assert {injector.resource_class for injector in mixed.injectors} == {"cpu_sandbox", "inference"}
    assert experiment.replays[-1].injectors == ()
    assert mixed.steps["agent-run"].is_leaf is False
    assert mixed.steps["model-call-1"].is_leaf is True
    assert mixed.steps["model-call-1"].total_seconds == 1.0


def test_relative_dot_path_keeps_the_directory_name(tmp_path, monkeypatch):
    directory = synthetic(tmp_path)
    monkeypatch.chdir(directory)

    assert read_experiment(Path(".")).name == directory.name


def test_matches_source_comes_from_agent_evidence(tmp_path):
    experiment = read_experiment(synthetic(tmp_path, mismatched=[(4, "model-call-1")]))

    assert experiment.replays[4].steps["model-call-1"].matches_source is False
    assert experiment.replays[4].steps["tool-call-1"].matches_source is True
    assert experiment.replays[3].steps["model-call-1"].matches_source is True
    # The root step is neither a model output nor a tool result.
    assert experiment.replays[4].steps["agent-run"].matches_source is None


@pytest.mark.parametrize("evidence_dir", ["../outside", "/tmp/elsewhere"])
def test_paths_outside_experiment_are_rejected(tmp_path, evidence_dir):
    directory = synthetic(tmp_path)

    def change(manifest):
        manifest["replays"][1]["evidence_dir"] = evidence_dir

    edit_manifest(directory, change)
    with pytest.raises(ReplayEvidenceError, match="outside the experiment"):
        read_experiment(directory)


def test_injection_log_outside_experiment_is_rejected(tmp_path):
    directory = synthetic(tmp_path)

    def change(manifest):
        manifest["replays"][4]["injection_log"] = "../x.json"

    edit_manifest(directory, change)
    with pytest.raises(ReplayEvidenceError, match="outside the experiment"):
        read_experiment(directory)


def test_duplicate_keys_and_oversize_are_rejected(tmp_path, monkeypatch):
    directory = synthetic(tmp_path)
    path = directory / "experiment.json"
    original = path.read_text()
    path.write_text(original.replace('"status"', '"status": "x", "status"', 1))
    with pytest.raises(ReplayEvidenceError, match="experiment.json"):
        read_experiment(directory)

    path.write_text(original)
    monkeypatch.setattr(replay_evidence, "MAX_EVIDENCE_BYTES", 10)
    with pytest.raises(ReplayEvidenceError, match="larger than"):
        read_experiment(directory)


def test_invalid_journal_line_is_rejected(tmp_path):
    directory = synthetic(tmp_path)
    with (directory / "replays" / "01" / "live-journal.jsonl").open("a") as journal:
        journal.write("not json\n")
    with pytest.raises(ReplayEvidenceError, match="live-journal.jsonl"):
        read_experiment(directory)


@pytest.mark.parametrize(("field", "value"), [("role", "warmup2"), ("condition", "gpu")])
def test_unknown_role_or_condition_is_rejected(tmp_path, field, value):
    directory = synthetic(tmp_path)

    def change(manifest):
        manifest["replays"][2][field] = value

    edit_manifest(directory, change)
    with pytest.raises(ReplayEvidenceError, match=value):
        read_experiment(directory)


def test_nonfinite_durations_are_kept_as_none_or_rejected(tmp_path):
    directory = synthetic(tmp_path)
    path = directory / "experiment.json"
    original = path.read_text()

    def set_total(manifest, value):
        manifest["replays"][1]["steps"]["agent-run"]["total_seconds"] = value

    edit_manifest(directory, lambda manifest: set_total(manifest, None))
    experiment = read_experiment(directory)
    assert experiment.replays[1].steps["agent-run"].total_seconds is None

    path.write_text(original)
    edit_manifest(directory, lambda manifest: set_total(manifest, 0.5))
    path.write_text(path.read_text().replace("0.5", "1e999", 1))
    with pytest.raises(ReplayEvidenceError, match="finite"):
        read_experiment(directory)


def test_unreadable_steps_stay_empty(tmp_path):
    directory = synthetic(tmp_path)

    def change(manifest):
        manifest["replays"][1]["steps"] = None

    edit_manifest(directory, change)
    assert read_experiment(directory).replays[1].steps == {}


def test_negative_durations_are_rejected_and_zero_is_kept(tmp_path):
    directory = synthetic(tmp_path)

    def set_queue(value):
        def change(manifest):
            manifest["replays"][1]["steps"]["agent-run"]["queue_seconds"] = value

        edit_manifest(directory, change)

    set_queue(0)
    assert read_experiment(directory).replays[1].steps["agent-run"].queue_seconds == 0.0
    set_queue(-5)
    with pytest.raises(ReplayEvidenceError, match="negative"):
        read_experiment(directory)


@pytest.mark.parametrize("line", ["[1]", "3"])
def test_non_object_journal_row_is_rejected(tmp_path, line):
    directory = synthetic(tmp_path)
    with (directory / "replays" / "01" / "live-journal.jsonl").open("a") as journal:
        journal.write(line + "\n")
    with pytest.raises(ReplayEvidenceError, match="must be an object"):
        read_experiment(directory)


@pytest.mark.parametrize(
    "change",
    [
        lambda m: m["replays"][2].update(index=m["replays"][1]["index"]),
        lambda m: m["replays"][0].update(index=-1),
    ],
    ids=["duplicate", "negative"],
)
def test_duplicate_or_negative_replay_index_is_rejected(tmp_path, change):
    directory = synthetic(tmp_path)
    edit_manifest(directory, change)
    with pytest.raises(ReplayEvidenceError, match="Replay index"):
        read_experiment(directory)
