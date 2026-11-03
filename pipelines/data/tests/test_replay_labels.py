import json
import shutil

import pytest
from agentwatch_data import replay_labels
from agentwatch_data.labels import LabelingError
from agentwatch_data.replay_labels import (
    ROW_KEYS,
    label_directory,
    load_replay_config,
    task_group,
)
from replay_fixtures import PASSING_CHECKS, write_experiment

VALID_RULES = """
version = "labels-v2"
[noise]
stdev_multiplier = {multiplier}
min_seconds = 0.01
[condition_resources]
cpu = "cpu_sandbox"
inference = "inference"
retrieval = "retrieval"
"""


def write_rules(tmp_path, text=None, multiplier="3.0"):
    path = tmp_path / "labels-v2.toml"
    path.write_text(text if text is not None else VALID_RULES.format(multiplier=multiplier))
    return path


@pytest.fixture
def config():
    return load_replay_config()


def test_packaged_replay_rules_load(config):
    assert (config.version, config.stdev_multiplier, config.min_seconds) == ("labels-v2", 3.0, 0.01)
    assert config.condition_resources == {
        "cpu": "cpu_sandbox",
        "inference": "inference",
        "retrieval": "retrieval",
    }
    assert config.label_version.startswith("labels-v2+")


# The full label_version of the packaged replay rules and labeling code, pinned so no
# change to either reaches a release unnoticed. When this fails: if the change can alter
# the label of any row the pipeline produces, or whether such a run is labeled at all,
# bump `version` in labels-v2.toml, which also needs an evaluation-contract change.
# Otherwise (a comment or refactor) keep the version, and say why in the PR. Either way,
# update the pin in the same PR.
PINNED_REPLAY_LABEL_VERSION = "labels-v2+0c15744b"


def test_replay_label_version_is_pinned(config):
    assert config.label_version == PINNED_REPLAY_LABEL_VERSION, (
        "replay_labels.py or labels-v2.toml changed; see the comment above "
        "PINNED_REPLAY_LABEL_VERSION"
    )


def test_formatting_does_not_change_fingerprint(config, tmp_path):
    text = "# extra comment\n" + VALID_RULES.format(multiplier="3")
    for variant in (text, text.replace("\n", "\r\n")):
        assert load_replay_config(write_rules(tmp_path, variant)).fingerprint == (
            load_replay_config(write_rules(tmp_path)).fingerprint
        )


def test_changed_values_change_fingerprint(tmp_path):
    base = load_replay_config(write_rules(tmp_path)).fingerprint
    assert load_replay_config(write_rules(tmp_path, multiplier="2.5")).fingerprint != base


@pytest.mark.parametrize(
    "text",
    [
        VALID_RULES.format(multiplier="3.0") + "extra = 1\n",
        'version = "labels-v2"\n[noise]\nstdev_multiplier = 3.0\nmin_seconds = 0.01\n',
        VALID_RULES.format(multiplier="nan"),
        VALID_RULES.format(multiplier="3.0") + 'gpu = "cpu_sandbox"\n',
        VALID_RULES.format(multiplier="3.0").replace('"retrieval"\n', '"none"\n', 1),
        VALID_RULES.format(multiplier="3.0").replace('cpu = "cpu_sandbox"', 'cpu = "gpu"'),
        "not toml [",
    ],
)
def test_invalid_rules_are_rejected(tmp_path, text):
    with pytest.raises(LabelingError):
        load_replay_config(write_rules(tmp_path, text))


def test_missing_rules_file_is_rejected(tmp_path):
    with pytest.raises(LabelingError):
        load_replay_config(tmp_path / "absent.toml")


# --- Labeling synthetic replay experiments ----------------------------------------
# Baseline seconds [1.00, 1.01, 0.99] give B = 1.00 and threshold max(0.01, 3 x 0.01) = 0.03.

BASELINES = [
    {"agent-run": 2.0, "model-call-1": 1.00, "tool-call-1": 0.20},
    {"agent-run": 2.0, "model-call-1": 1.01, "tool-call-1": 0.20},
    {"agent-run": 2.0, "model-call-1": 0.99, "tool-call-1": 0.20},
]


EXPECTED_ROW_KEYS = (
    "experiment",
    "task_id",
    "task_group",
    "plan_sha256",
    "agent_model",
    "run_id",
    "replay_index",
    "role",
    "condition",
    "step_id",
    "step_type",
    "duration_seconds",
    "queue_seconds",
    "execution_seconds",
    "label_resource",
    "label_resources",
    "label_excess_fraction",
    "label_basis",
    "injection_method",
    "baseline_mean_seconds",
    "noise_threshold_seconds",
    "control_exceeded_noise",
    "failed_checks",
    "label_version",
)


def steps(model_call):
    return {"agent-run": 2.5, "model-call-1": model_call, "tool-call-1": 0.20}


def experiment(tmp_path, contended=None, **kwargs):
    kwargs.setdefault("baseline", BASELINES)
    return write_experiment(tmp_path, contended=contended or {"cpu": steps(1.5)}, **kwargs)


def edit_manifest(directory, change):
    path = directory / "experiment.json"
    manifest = json.loads(path.read_text())
    change(manifest)
    path.write_text(json.dumps(manifest))


def label(directory, config):
    return label_directory(directory, config)


def row_for(rows, condition, step_id="model-call-1", role="contended"):
    (row,) = [
        r
        for r in rows
        if r["condition"] == condition and r["role"] == role and r["step_id"] == step_id
    ]
    return row


def codes(rejections):
    return [rejection.reason.split(": ", 1)[0] for rejection in rejections]


def test_solo_exceeded_and_below(tmp_path, config):
    rows, rejections = label(experiment(tmp_path), config)
    assert rejections == []
    cpu = row_for(rows, "cpu")
    assert (cpu["label_resource"], cpu["label_resources"]) == ("cpu_sandbox", "cpu_sandbox")
    assert cpu["label_basis"] == "contention_exceeded_threshold"
    assert cpu["label_excess_fraction"] == pytest.approx(1 / 3)
    assert cpu["baseline_mean_seconds"] == pytest.approx(1.0)
    assert cpu["noise_threshold_seconds"] == pytest.approx(0.03)
    assert cpu["injection_method"] == "cpu-spin-processes-v1"
    assert cpu["label_version"] == config.label_version
    assert tuple(cpu) == ROW_KEYS == EXPECTED_ROW_KEYS

    rows, _ = label(experiment(tmp_path / "below", {"cpu": steps(1.02)}), config)
    below = row_for(rows, "cpu")
    assert (below["label_resource"], below["label_resources"]) == ("none", "")
    assert below["label_basis"] == "contention_below_threshold"


def test_threshold_tie_is_not_exceeded(tmp_path, config):
    # Exact binary fractions: B = 1.0, stdev = 0.0625, threshold = 0.1875, T = B + threshold.
    baseline = [steps(1.0), steps(1.0625), steps(0.9375)]
    rows, _ = label(experiment(tmp_path, {"cpu": steps(1.1875)}, baseline=baseline), config)
    tie = row_for(rows, "cpu")
    assert tie["noise_threshold_seconds"] == 0.1875
    assert tie["label_resource"] == "none"
    assert tie["label_basis"] == "contention_below_threshold"


def mixed_experiment(tmp_path, cpu, inference, mixed):
    contended = {"cpu": steps(cpu), "inference": steps(inference), "mixed": steps(mixed)}
    return experiment(tmp_path, contended)


def test_mixed_confirmed(tmp_path, config):
    rows, rejections = label(mixed_experiment(tmp_path, 1.5, 1.4, 1.9), config)
    assert rejections == []
    mixed = row_for(rows, "mixed")
    assert mixed["label_resources"] == "cpu_sandbox,inference"
    assert mixed["label_resource"] == "mixed"
    assert mixed["label_basis"] == "mixed_confirmed"
    assert mixed["injection_method"] == "competing-ollama-generate-v1+cpu-spin-processes-v1"
    assert row_for(rows, "inference")["label_resource"] == "inference"


def test_mixed_partial_single_member(tmp_path, config):
    rows, _ = label(mixed_experiment(tmp_path, 1.5, 1.01, 1.9), config)
    mixed = row_for(rows, "mixed")
    assert mixed["label_resources"] == "cpu_sandbox"
    assert mixed["label_resource"] == "cpu_sandbox"
    assert mixed["label_basis"] == "mixed_partial"


def test_mixed_joint_only(tmp_path, config):
    rows, _ = label(mixed_experiment(tmp_path, 1.01, 1.01, 1.9), config)
    mixed = row_for(rows, "mixed")
    assert mixed["label_resources"] == "cpu_sandbox,inference"
    assert mixed["label_resource"] == "mixed"
    assert mixed["label_basis"] == "mixed_joint_only"


def test_mixed_below_threshold(tmp_path, config):
    rows, _ = label(mixed_experiment(tmp_path, 1.5, 1.4, 1.01), config)
    mixed = row_for(rows, "mixed")
    assert (mixed["label_resource"], mixed["label_resources"]) == ("none", "")
    assert mixed["label_basis"] == "mixed_below_threshold"


def test_mixed_missing_solo_member(tmp_path, config):
    directory = experiment(tmp_path, {"cpu": steps(1.5), "mixed": steps(1.9)})
    rows, rejections = label(directory, config)
    assert [r for r in rows if r["condition"] == "mixed"] == []
    assert set(codes(rejections)) == {"mixed_solo_evidence_missing"}
    assert {r.origin for r in rejections} == {
        "experiment synthetic-exp replay 06 step model-call-1",
        "experiment synthetic-exp replay 06 step tool-call-1",
    }
    assert row_for(rows, "cpu")["label_resource"] == "cpu_sandbox"


def test_mixed_with_rejected_solo_member_is_rejected(tmp_path, config):
    directory = mixed_experiment(tmp_path, 1.5, 1.4, 1.9)
    edit_manifest(directory, lambda m: m["replays"][6].update(task_passed=False))
    rows, rejections = label(directory, config)
    assert "replay_not_succeeded" in codes(rejections)
    assert "mixed_solo_evidence_missing" in codes(rejections)
    assert [r for r in rows if r["condition"] == "mixed"] == []


def test_mixed_with_single_injector_is_rejected(tmp_path, config):
    directory = mixed_experiment(tmp_path, 1.5, 1.4, 1.9)
    log = directory / "injections" / "08.json"
    content = json.loads(log.read_text())
    content["injections"] = content["injections"][:1]
    log.write_text(json.dumps(content))
    rows, rejections = label(directory, config)
    assert codes(rejections) == ["resource_class_mismatch"]
    assert rejections[0].origin == "experiment synthetic-exp replay 08"
    assert [r for r in rows if r["condition"] == "mixed"] == []


def test_solo_injector_class_must_match_condition(tmp_path, config):
    directory = experiment(tmp_path)
    log = directory / "injections" / "04.json"
    content = json.loads(log.read_text())
    content["injections"][0]["contended_resource_class"] = "inference"
    log.write_text(json.dumps(content))
    rows, rejections = label(directory, config)
    assert codes(rejections) == ["resource_class_mismatch"]
    assert [r for r in rows if r["role"] == "contended"] == []


def test_clean_rows_and_noisy_control(tmp_path, config):
    directory = experiment(tmp_path, controls={"cpu": steps(1.5)})
    rows, _ = label(directory, config)
    baselines = [r for r in rows if r["role"] == "baseline"]
    assert len(baselines) == 6
    assert {(r["label_basis"], r["label_resource"], r["label_resources"]) for r in baselines} == {
        ("baseline_reference", "none", "")
    }
    assert {r["injection_method"] for r in baselines} == {"none"}
    control = row_for(rows, "clean", role="clean_control")
    assert (control["label_basis"], control["label_resource"]) == ("clean_exceeded_noise", "none")
    assert control["control_exceeded_noise"] is None
    contended = row_for(rows, "cpu")
    assert contended["control_exceeded_noise"] is True
    assert contended["label_resource"] == "cpu_sandbox"
    assert row_for(rows, "cpu", "tool-call-1")["control_exceeded_noise"] is False
    within = row_for(rows, "clean", "tool-call-1", role="clean_control")
    assert within["label_basis"] == "clean_within_noise"


def test_unusable_control_leaves_control_exceeded_noise_missing(tmp_path, config):
    directory = experiment(tmp_path)
    edit_manifest(directory, lambda m: m["replays"][5].update(state="failed"))
    rows, rejections = label(directory, config)
    assert codes(rejections) == ["replay_not_succeeded"]
    assert row_for(rows, "cpu")["control_exceeded_noise"] is None


def test_only_leaf_steps_are_labeled(tmp_path, config):
    rows, _ = label(experiment(tmp_path), config)
    assert rows
    assert all(r["step_id"] != "agent-run" for r in rows)
    assert [(r["replay_index"], r["step_id"]) for r in rows] == sorted(
        (r["replay_index"], r["step_id"]) for r in rows
    )
    assert 0 not in {r["replay_index"] for r in rows}  # the warm-up is never labeled


def test_differing_baseline_outputs_shrink_reference(tmp_path, config):
    mismatched = [(1, "model-call-1"), (2, "model-call-1"), (4, "tool-call-1")]
    rows, rejections = label(experiment(tmp_path, mismatched=mismatched), config)
    assert [r for r in rows if r["step_id"] == "model-call-1"] == []
    by_origin = {r.origin: r.reason.split(": ", 1)[0] for r in rejections}
    for index in (1, 2, 3, 4, 5):
        origin = f"experiment synthetic-exp replay {index:02d} step model-call-1"
        assert by_origin[origin] == "baseline_reference_missing"
    assert by_origin["experiment synthetic-exp replay 04 step tool-call-1"] == (
        "output_differs_from_source"
    )
    assert len(rejections) == 6
    assert {r["step_id"] for r in rows} == {"tool-call-1"}


def test_incomplete_step_is_rejected(tmp_path, config):
    directory = experiment(tmp_path)

    def change(manifest):
        manifest["replays"][4]["steps"]["model-call-1"]["total_seconds"] = None
        manifest["replays"][5]["steps"]["model-call-1"]["total_seconds"] = 0
        manifest["replays"][3]["steps"]["tool-call-1"]["state"] = "failed"

    edit_manifest(directory, change)
    rows, rejections = label(directory, config)
    assert codes(rejections) == ["step_not_completed"] * 3
    assert all(r.reason.startswith("step_not_completed: ") for r in rejections)
    # The failed baseline copy leaves two usable copies, so tool-call-1 keeps its reference.
    assert row_for(rows, "cpu", "tool-call-1")["label_basis"] == "contention_below_threshold"


def test_incomplete_experiment_yields_no_rows(tmp_path, config):
    rows, rejections = label(experiment(tmp_path, status="cancelled"), config)
    assert rows == []
    assert codes(rejections) == ["experiment_not_completed"]
    assert rejections[0].origin == "experiment synthetic-exp"

    checks = PASSING_CHECKS | {"stable_baseline": False}
    rows, rejections = label(experiment(tmp_path / "unstable", checks=checks), config)
    assert rows == []
    assert codes(rejections) == ["protocol_checks_failed"]
    assert "stable_baseline" in rejections[0].reason


def test_missing_blocking_check_is_not_a_pass(tmp_path, config):
    checks = {k: v for k, v in PASSING_CHECKS.items() if k != "injectors_cleaned_up"}
    rows, rejections = label(experiment(tmp_path, checks=checks), config)
    assert rows == []
    assert codes(rejections) == ["protocol_checks_failed"]
    assert "injectors_cleaned_up" in rejections[0].reason


def test_unsucceeded_or_inactive_replay_is_rejected(tmp_path, config):
    directory = experiment(tmp_path)
    edit_manifest(directory, lambda m: m["replays"][4].update(task_passed=False))
    rows, rejections = label(directory, config)
    assert codes(rejections) == ["replay_not_succeeded"]
    assert rejections[0].origin == "experiment synthetic-exp replay 04"
    assert [r for r in rows if r["role"] == "contended"] == []

    directory = experiment(tmp_path / "inactive")
    edit_manifest(directory, lambda m: m["replays"][4].update(injection_active=False))
    rows, rejections = label(directory, config)
    assert codes(rejections) == ["injection_inactive"]
    assert [r for r in rows if r["role"] == "contended"] == []


def test_non_blocking_failed_checks_are_recorded(tmp_path, config):
    checks = PASSING_CHECKS | {"evidence_delivered": False, "injectors_active": True}
    rows, rejections = label(experiment(tmp_path, checks=checks), config)
    assert rejections == []
    assert {r["failed_checks"] for r in rows} == {"evidence_delivered"}
    rows, _ = label(experiment(tmp_path / "clean"), config)
    assert {r["failed_checks"] for r in rows} == {None}


def test_row_identity_comes_from_the_experiment(tmp_path, config):
    rows, _ = label(experiment(tmp_path, task_id="fix-mean-empty-r2"), config)
    cpu = row_for(rows, "cpu")
    assert (cpu["experiment"], cpu["task_id"], cpu["task_group"]) == (
        "synthetic-exp",
        "fix-mean-empty-r2",
        "fix-mean-empty",
    )
    assert (cpu["plan_sha256"], cpu["agent_model"]) == ("0" * 64, "synthetic-model")
    assert (cpu["replay_index"], cpu["run_id"]) == (4, "00000000-0000-4000-8000-000000000004")
    assert (cpu["step_type"], cpu["duration_seconds"]) == ("model_call", 1.5)
    assert (cpu["queue_seconds"], cpu["execution_seconds"]) == (0.001, pytest.approx(1.499))


def test_labels_reproduce(tmp_path, config):
    directory = mixed_experiment(tmp_path / "a", 1.5, 1.01, 1.9)
    first, first_rejections = label(directory, config)
    second, second_rejections = label(directory, load_replay_config())
    copy = tmp_path / "b" / "copied-exp"
    shutil.copytree(directory, copy)
    copied, copied_rejections = label(copy, config)
    assert first == second
    assert first_rejections == second_rejections == copied_rejections == []
    assert [r | {"experiment": None} for r in copied] == [r | {"experiment": None} for r in first]
    assert {r["label_version"] for r in first + copied} == {config.label_version}


def test_task_group():
    assert task_group("fix-mean-empty-r1") == "fix-mean-empty"
    assert task_group("fix-window-count-r12") == "fix-window-count"
    assert task_group("a-r2x") == "a-r2x"
    assert task_group("plain") == "plain"


def test_cli_prints_rows(tmp_path, capsys):
    directory = experiment(tmp_path)
    assert replay_labels.main([str(directory)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["rows"] and output["rejections"] == []

    assert replay_labels.main([str(tmp_path / "missing")]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "experiment.json" in captured.err
