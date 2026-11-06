import random
from pathlib import Path

import pytest
from agentwatch_data import splits
from agentwatch_data.split_assignment import (
    Assignment,
    AssignmentError,
    assign_groups,
    held_out_status,
    independence,
    load_assignment,
    mixed_in_test,
    partition_v2,
    row_methods,
)

FRACTIONS = {"train": 0.7, "validation": 0.15, "test": 0.15}
HELD_OUT = "worker_queue_blocker"
CPU = "cpu-spin-processes-v1"


def row(group, run, step=0, method="none", resource="none", condition="baseline", plan=None):
    return {
        "task_group": group,
        "task_id": f"{group}-r1",
        "run_id": run,
        "step_id": f"s{step}",
        "plan_sha256": plan,
        "condition": condition,
        "label_resource": resource,
        "injection_method": method,
    }


def assignment(**overrides):
    values = {
        "version": "split-v2",
        "held_out_injection_methods": (HELD_OUT,),
        "test_task_groups": ("fix-window-count",),
        "sha256": "00000000",
    }
    return Assignment(**(values | overrides))


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "assignment.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_packaged_assignment_loads():
    loaded = load_assignment()
    assert loaded.version == "split-v2"
    assert loaded.held_out_injection_methods == (HELD_OUT,)
    assert loaded.test_task_groups == ("fix-window-count",)
    assert len(loaded.sha256) == 8


@pytest.mark.parametrize(
    "text",
    [
        'version = "split-v2"\nheld_out_injection_methods = []\ntest_task_groups = []\nextra = 1\n',
        'version = "split-v2"\nheld_out_injection_methods = [""]\ntest_task_groups = []\n',
        'version = "split-v2"\nheld_out_injection_methods = ["none"]\ntest_task_groups = []\n',
        'version = "split-v2"\nheld_out_injection_methods = ["a", "a"]\ntest_task_groups = []\n',
        'version = "split-v2"\nheld_out_injection_methods = []\ntest_task_groups = ["g", "g"]\n',
        'version = "split-v20"\nheld_out_injection_methods = []\ntest_task_groups = []\n',
        'version = "split-v1"\nheld_out_injection_methods = []\ntest_task_groups = []\n',
        'version = "split-v2"\nheld_out_injection_methods = []\n',
        "not toml [",
    ],
)
def test_invalid_assignment_is_rejected(tmp_path, text):
    with pytest.raises(AssignmentError):
        load_assignment(write(tmp_path, text))


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(AssignmentError):
        load_assignment(tmp_path / "absent.toml")


def test_row_methods_splits_mixed_and_drops_none():
    assert row_methods(row("g", "r", method="none")) == set()
    assert row_methods(row("g", "r", method=f"{CPU}+{HELD_OUT}")) == {CPU, HELD_OUT}


def test_declared_test_group_goes_to_test():
    assert splits.assign("fix-window-count", FRACTIONS) == "train"
    rows = [row("fix-window-count", "r1")]
    assert assign_groups(rows, assignment(), FRACTIONS) == {"fix-window-count": "test"}


def test_held_out_method_moves_whole_group():
    rows = [row("fix-a", "r1"), row("fix-a", "r2", method=HELD_OUT, resource="worker_queue")]
    groups, members, _ = partition_v2(rows, assignment(), FRACTIONS)
    assert groups == {"fix-a": "test"}
    assert len(members["test"]) == 2


def test_mixed_method_containing_held_out_moves_group():
    rows = [row("fix-a", "r1", method=f"{CPU}+{HELD_OUT}", condition="mixed", resource="mixed")]
    assert assign_groups(rows, assignment(), FRACTIONS) == {"fix-a": "test"}


def test_other_groups_use_split_v1_hash():
    rows = [row(f"task-{n}", f"r{n}") for n in range(20)]
    groups = assign_groups(rows, assignment(), FRACTIONS)
    assert groups == {f"task-{n}": splits.assign(f"task-{n}", FRACTIONS) for n in range(20)}


def test_rows_without_task_group_group_by_task_id():
    bare = {k: v for k, v in row("cpu", "r1").items() if k != "task_group"} | {"task_id": "t-9"}
    assert assign_groups([bare], assignment(), FRACTIONS) == {
        "t-9": splits.assign("t-9", FRACTIONS)
    }


def test_revisions_share_a_split():
    rows = [
        row("fix-a", "r1") | {"task_id": "fix-a-r1"},
        row("fix-a", "r2") | {"task_id": "fix-a-r2"},
    ]
    _, members, _ = partition_v2(rows, assignment(), FRACTIONS)
    assert sum(bool(members[name]) for name in splits.SPLITS) == 1


def find_groups_in_different_splits():
    names = [f"task-{n}" for n in range(50)]
    first = names[0]
    other = next(n for n in names if splits.assign(n, FRACTIONS) != splits.assign(first, FRACTIONS))
    return first, other


@pytest.mark.parametrize("key", ["plan_sha256", "run_id"])
def test_shared_plan_or_run_fails_independence(key):
    first, other = find_groups_in_different_splits()
    a, b = row(first, "run-a", plan="p1"), row(other, "run-b", plan="p2")
    a[key] = b[key] = "shared"
    with pytest.raises(AssignmentError, match=key):
        partition_v2([a, b], assignment(), FRACTIONS)


def test_independence_reports_shared_values_and_ignores_missing_plan():
    members = {
        "train": [row("g1", "r1", plan=None)],
        "validation": [],
        "test": [row("g2", "r2", plan=None)],
    }
    assert independence(members)["status"] == "passed"
    members["test"].append(row("g1", "r3"))
    result = independence(members)
    assert result["status"] == "failed"
    assert result["shared"]["task_group"] == ["g1"]
    assert result["shared"]["run_id"] == []


def held_members(with_contrast):
    train = [row("g1", "r1")]
    if with_contrast:
        train.append(row("g1", "r2", method=CPU, resource="cpu_sandbox"))
    test = [row("g2", "r3", method=HELD_OUT, resource="cpu_sandbox")]
    return {"train": train, "validation": [], "test": test}


def test_held_out_needs_contrasting_training_method():
    missing = held_out_status(held_members(False), assignment())
    assert missing["status"] == "not_met" and missing["reason"]
    met = held_out_status(held_members(True), assignment())
    assert met["status"] == "met" and met["reason"] is None
    assert met["only_in_test"] == [HELD_OUT]


def test_held_out_method_in_training_is_not_met():
    members = held_members(True)
    members["train"].append(row("g1", "r9", method=HELD_OUT, resource="cpu_sandbox"))
    assert held_out_status(members, assignment())["status"] == "not_met"


def test_held_out_without_training_rows_is_not_met():
    members = {
        "train": [],
        "validation": [],
        "test": [row("g2", "r3", method=HELD_OUT, resource="none")],
    }
    result = held_out_status(members, assignment())
    assert result["status"] == "not_met"
    assert "no contended label" in result["reason"]


def test_held_out_rows_without_contended_label_are_not_met():
    members = held_members(True)
    members["test"] = [
        row("g2", "r3", method=HELD_OUT, resource="none"),
        row("g2", "r4", method=HELD_OUT, resource="mixed", condition="mixed"),
    ]
    result = held_out_status(members, assignment())
    assert result["status"] == "not_met"
    assert "no contended label" in result["reason"]


def test_mixed_in_test_reported():
    members = {"train": [row("g", "r", condition="mixed")], "validation": [], "test": []}
    assert mixed_in_test(members) == {"status": "not_met", "rows": 0}
    members["test"].append(row("g2", "r2", condition="mixed"))
    assert mixed_in_test(members) == {"status": "met", "rows": 1}


def test_mixed_in_test_reads_replay_condition_on_release_rows():
    """Release rows keep the role in condition and the injected condition in replay_condition."""
    replay = row("g", "r1", condition="contended") | {"replay_condition": "mixed"}
    run = row("cpu", "r2", condition="contended") | {"replay_condition": None}
    solo = row("g", "r3", condition="contended") | {"replay_condition": "cpu"}
    members = {"train": [], "validation": [], "test": [replay, run, solo]}
    assert mixed_in_test(members) == {"status": "met", "rows": 1}


def test_split_positions_are_the_split_v1_hash():
    from agentwatch_data import split_positions

    assert split_positions.HASH_METHOD == "split-v1"
    assert split_positions.SPLITS == splits.SPLITS == ("train", "validation", "test")
    assert splits.assign is split_positions.assign


def step_keys(rows):
    return [(r["run_id"], r["step_id"]) for r in rows]


def test_partition_is_deterministic():
    rows = [row(f"task-{n}", f"r{n}", step=s) for n in range(12) for s in range(2)]
    rows.append(row("fix-window-count", "rw"))
    baseline = partition_v2(rows, assignment(), FRACTIONS)
    shuffled = list(rows)
    random.Random(7).shuffle(shuffled)
    again = partition_v2(shuffled, assignment(), FRACTIONS)
    assert baseline[0] == again[0]
    for name in splits.SPLITS:
        assert step_keys(baseline[1][name]) == step_keys(again[1][name])
