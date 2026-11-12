"""split-v3: the frozen assignment that holds out the A06 evidence and fills validation.

Every row here is synthetic; experiment names are chosen by their split-v3 hash position
(exp-0 is 0.023, below the validation share; exp-1 is 0.812, above it).
"""

from pathlib import Path

import pytest
from agentwatch_data.split_assignment import (
    AssignmentError,
    experiment_position,
    held_out_protocol,
    load_assignment,
    partition_v3,
    validation_share,
)

FRACTIONS = {"train": 0.7, "validation": 0.15, "test": 0.15}
FEATURES = "step-features-v1"
PROTOCOL = "natural-overload-sweep-v1"
SWEEP_DEFAULT = "22e26dc7536fe5e88e3d9ea9f01445a754788ce587b1626d6a4eb055ec4524b9"
SWEEP_HARDWARE = "1608deee2a3bcb5245b07d73e13e73dcd1e82702ef14aa683ecd94144e0f18cd"
SWEEP_DRY_RUN = "a16b2a1469fbccb94b715ab9718b482b3539776d994efa7e1c4fb4964ddad25e"
PACKAGED = Path(__file__).resolve().parents[1] / "src/agentwatch_data/split-assignment-v3.toml"


def row(group, run, experiment=None, step=0, method="none", plan=None, **extra):
    return {
        "task_group": group,
        "task_id": f"{group}-r1",
        "run_id": run,
        "step_id": f"s{step}",
        "experiment": experiment,
        "plan_sha256": plan,
        "feature_version": FEATURES,
        "condition": "baseline",
        "replay_condition": "clean",
        "label_resource": "none",
        "injection_method": method,
    } | extra


def v3_text(**overrides) -> str:
    values = {
        "version": '"split-v3"',
        "feature_versions": f'["{FEATURES}"]',
        "held_out_injection_methods": '["worker_queue_blocker", "prefill-flood-openai-chat-v1"]',
        "test_task_groups": '["fix-window-count", "fix-median-even"]',
    } | overrides
    lines = [f"{key} = {value}" for key, value in values.items() if value is not None]
    protocol = f'[held_out_protocol]\nversion = "{PROTOCOL}"\nsha256 = ["{SWEEP_DEFAULT}"]\n'
    return "\n".join(lines) + "\n\n" + protocol


def loaded(tmp_path: Path, text: str | None = None):
    path = tmp_path / "assignment.toml"
    path.write_text(text if text is not None else v3_text(), encoding="utf-8")
    return load_assignment(path)


def test_the_packaged_v3_assignment_holds_out_the_a06_evidence():
    assignment = load_assignment(PACKAGED)
    assert assignment.version == "split-v3"
    assert assignment.feature_versions == (FEATURES,)
    assert assignment.held_out_injection_methods == (
        "worker_queue_blocker",
        "prefill-flood-openai-chat-v1",
    )
    assert assignment.test_task_groups == ("fix-window-count", "fix-median-even")
    assert assignment.held_out_protocol_version == PROTOCOL
    assert assignment.held_out_protocol_sha256 == (SWEEP_DEFAULT, SWEEP_HARDWARE, SWEEP_DRY_RUN)


def test_the_packaged_v2_assignment_is_unchanged():
    assignment = load_assignment()
    assert assignment.version == "split-v2"
    assert assignment.feature_versions == () and assignment.held_out_protocol_sha256 == ()


@pytest.mark.parametrize(
    "text",
    [
        v3_text(feature_versions=None),
        v3_text(feature_versions="[]"),
        v3_text(feature_versions='["a", "a"]'),
        v3_text(extra="1"),
        v3_text().replace(f'["{SWEEP_DEFAULT}"]', "[]"),
        v3_text().replace(f'["{SWEEP_DEFAULT}"]', '["abc"]'),
        v3_text().replace(f'["{SWEEP_DEFAULT}"]', f'["{SWEEP_DEFAULT.upper()}"]'),
        v3_text().replace(f'["{SWEEP_DEFAULT}"]', f'["{SWEEP_DEFAULT}", "{SWEEP_DEFAULT}"]'),
        v3_text().replace(f'version = "{PROTOCOL}"', 'version = ""'),
        v3_text().replace("[held_out_protocol]", "[other]"),
        v3_text().split("[held_out_protocol]")[0],
    ],
)
def test_an_invalid_v3_assignment_is_rejected(tmp_path, text):
    with pytest.raises(AssignmentError):
        loaded(tmp_path, text)


def test_a_v2_file_with_v3_keys_is_rejected(tmp_path):
    with pytest.raises(AssignmentError, match="Split assignment keys"):
        loaded(tmp_path, v3_text(version='"split-v2"'))


def test_the_validation_share_keeps_the_contract_ratio():
    assert validation_share(FRACTIONS) == pytest.approx(0.15 / 0.85)


def test_training_experiments_below_the_share_go_to_validation(tmp_path):
    rows = [
        row("fix-mean-empty", "a", "exp-0", plan="p0"),
        row("fix-mean-empty", "b", "exp-1", plan="p1"),
    ]
    groups, members, _ = partition_v3(rows, loaded(tmp_path), FRACTIONS)
    assert groups == {"fix-mean-empty": "train"}
    assert [r["run_id"] for r in members["validation"]] == ["a"]
    assert [r["run_id"] for r in members["train"]] == ["b"]


@pytest.mark.parametrize("experiment", [None, ""])
def test_a_split_row_without_an_experiment_is_refused(tmp_path, experiment):
    # A capacity-sweep run has no experiment; untagged, it would otherwise land in train.
    rows = [row("fix-mean-empty", "a", "exp-1", plan="p1"), row("fix-mean-empty", "n", experiment)]
    with pytest.raises(AssignmentError, match="names no experiment"):
        partition_v3(rows, loaded(tmp_path), FRACTIONS)


def test_a_test_task_row_without_an_experiment_is_refused_too(tmp_path):
    # Fail closed on every split row, not only training ones, so tagging cannot be skipped.
    rows = [row("fix-mean-empty", "a", "exp-1", plan="p1"), row("fix-window-count", "w", None)]
    with pytest.raises(AssignmentError, match="names no experiment"):
        partition_v3(rows, loaded(tmp_path), FRACTIONS)


def test_a_held_out_row_needs_no_experiment(tmp_path):
    overload = {"protocol_version": PROTOCOL, "protocol_sha256": SWEEP_DEFAULT}
    rows = [row("fix-mean-empty", "a", "exp-1"), row("fix-mean-empty", "n", None, **overload)]
    _, _, checks = partition_v3(rows, loaded(tmp_path), FRACTIONS)
    assert checks["held_out_evaluation"]["rows"] == 1


def test_declared_test_groups_go_to_test_whole(tmp_path):
    rows = [
        row("fix-window-count", "w", "exp-0", plan="pw"),
        row("fix-median-even", "m", "exp-0", plan="pm", method="prefill-flood-openai-chat-v1"),
    ]
    groups, members, _ = partition_v3(rows, loaded(tmp_path), FRACTIONS)
    assert groups == {"fix-median-even": "test", "fix-window-count": "test"}
    assert not members["validation"] and len(members["test"]) == 2


@pytest.mark.parametrize(
    "method", ["prefill-flood-openai-chat-v1", "cpu-spin-processes-v1+worker_queue_blocker"]
)
def test_a_held_out_method_on_a_training_task_is_refused(tmp_path, method):
    # split-v2 would move the whole task to test, emptying training; split-v3 refuses instead.
    rows = [
        row("fix-mean-empty", "a", "exp-1", plan="p1"),
        row("fix-mean-empty", "b", "exp-2", plan="p2", method=method),
    ]
    with pytest.raises(AssignmentError, match="not declare as a test task"):
        partition_v3(rows, loaded(tmp_path), FRACTIONS)


def test_a_v2_assignment_is_refused(tmp_path):
    with pytest.raises(AssignmentError, match="needs a split-v3 assignment"):
        partition_v3([row("fix-mean-empty", "a")], load_assignment(), FRACTIONS)


def test_the_validation_experiments_are_recorded(tmp_path):
    rows = [row("fix-mean-empty", "a", "exp-0", plan="p0"), row("fix-mean-empty", "b", "exp-1")]
    _, _, checks = partition_v3(rows, loaded(tmp_path), FRACTIONS)
    assert checks["validation_experiments"] == ["exp-0"]


def test_rows_of_another_feature_definition_are_left_out(tmp_path):
    rows = [
        row("fix-mean-empty", "a", "exp-1"),
        row("cpu-task", "c", feature_version="steps-v1.2"),  # left out, so no experiment needed
        row("cpu-task", "d", feature_version=None),
    ]
    groups, members, checks = partition_v3(rows, loaded(tmp_path), FRACTIONS)
    assert "cpu-task" not in groups
    assert sum(map(len, members.values())) == 1
    assert checks["excluded_feature_versions"] == {
        "<missing>": {"rows": 1, "runs": 1},
        "steps-v1.2": {"rows": 1, "runs": 1},
    }


def test_natural_overload_rows_stay_out_of_every_split(tmp_path):
    # A replay of the training task under natural overload is held out, never trained on.
    overload = {"protocol_version": PROTOCOL, "protocol_sha256": SWEEP_DEFAULT}
    rows = [row("fix-mean-empty", "a", "exp-1"), row("fix-mean-empty", "n", "sweep", **overload)]
    _, members, checks = partition_v3(rows, loaded(tmp_path), FRACTIONS)
    assert [r["run_id"] for r in members["train"]] == ["a"]
    assert not members["validation"] and not members["test"]
    held_out = checks["held_out_evaluation"]
    assert (held_out["rows"], held_out["runs"]) == (1, 1)
    assert held_out["protocol_sha256"] == [SWEEP_DEFAULT]


def test_a_listed_protocol_hash_is_held_out_whatever_its_version(tmp_path):
    tagged = {"protocol_version": "another-protocol", "protocol_sha256": SWEEP_DEFAULT}
    assert held_out_protocol(row("g", "r", **tagged), loaded(tmp_path))


def test_the_packaged_assignment_holds_out_the_preflight_dry_run():
    # The dry run (sweep 3a8dd5b2-...) uses the protocol's version with its own hash.
    dry = {"protocol_version": PROTOCOL, "protocol_sha256": SWEEP_DRY_RUN}
    assert held_out_protocol(row("fix-mean-empty", "r", **dry), load_assignment(PACKAGED))


@pytest.mark.parametrize("sha256", ["f" * 64, None])
def test_an_unlisted_natural_overload_sweep_is_refused(tmp_path, sha256):
    unknown = {"protocol_version": PROTOCOL, "protocol_sha256": sha256}
    with pytest.raises(AssignmentError, match="not listed"):
        partition_v3([row("fix-mean-empty", "n", "sweep", **unknown)], loaded(tmp_path), FRACTIONS)


@pytest.mark.parametrize("key", ["run_id", "plan_sha256"])
def test_the_held_out_set_shares_no_run_or_plan_with_the_splits(tmp_path, key):
    overload = {"protocol_version": PROTOCOL, "protocol_sha256": SWEEP_DEFAULT}
    rows = [
        row("fix-mean-empty", "a", "exp-1", plan="p1"),
        row("fix-mean-empty", "n", "sweep", plan="pn", **overload),
    ]
    rows[1][key] = rows[0][key]
    with pytest.raises(AssignmentError, match=f"held-out evaluation set shares {key}"):
        partition_v3(rows, loaded(tmp_path), FRACTIONS)


def test_train_and_validation_may_share_a_task_but_not_a_plan(tmp_path):
    rows = [
        row("fix-mean-empty", "a", "exp-0", plan="p0"),
        row("fix-mean-empty", "b", "exp-1", plan="p1"),
    ]
    _, _, checks = partition_v3(rows, loaded(tmp_path), FRACTIONS)
    assert checks["independence"]["status"] == "passed"
    for key, value in (("plan_sha256", "p0"), ("run_id", "a")):
        shared = [rows[0], rows[1] | {key: value}]
        with pytest.raises(AssignmentError, match="not independent"):
            partition_v3(shared, loaded(tmp_path), FRACTIONS)


def test_experiment_positions_are_pinned():
    # A literal guard: changing the salt or the hash would move every validation experiment.
    assert experiment_position("exp-0") == pytest.approx(0.023, abs=5e-4)
    assert experiment_position("exp-1") == pytest.approx(0.812, abs=5e-4)


def test_split_v3_partition_is_deterministic(tmp_path):
    rows = [row("fix-mean-empty", f"r{n}", f"exp-{n}", plan=f"p{n}") for n in range(12)]
    first = partition_v3(rows, loaded(tmp_path), FRACTIONS)
    assert partition_v3(list(reversed(rows)), loaded(tmp_path), FRACTIONS) == first
