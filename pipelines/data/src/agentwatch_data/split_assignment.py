"""Frozen `split-v2` assignment and the independence checks that guard it.

D02 provides these functions over labeled rows; D03 wires them into make_split and load_split
once releases carry `task_group`. eval-contract-v1 names no `split-v2`, so `split-v1` is
unchanged. The assignment file is split-assignment.toml: its declared test task groups and
held-out injection methods go to test, and every other group keeps its `split-v1` hash
position, so new tasks never move frozen ones.

Rows group by `task_group` (revisions of a task share it) or, for rows without one such as the
CPU and coding release rows, by `task_id`.
"""

import tomllib
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from agentwatch_data.extract import canonical_sha256
from agentwatch_data.labels import NO_INJECTION
from agentwatch_data.splits import SPLITS, assign

ASSIGNMENT_KEYS = {"version", "held_out_injection_methods", "test_task_groups"}
# This module implements split-v2; a later assignment needs its own code and test split.
VERSION = "split-v2"
MIXED = "mixed"
INDEPENDENCE_KEYS = ("task_group", "run_id", "plan_sha256")


class AssignmentError(ValueError):
    """The assignment file is invalid, or the split violates independence."""


@dataclass(frozen=True)
class Assignment:
    version: str
    held_out_injection_methods: tuple[str, ...]
    test_task_groups: tuple[str, ...]
    sha256: str  # first 8 hex digits of the canonical hash of the parsed file


def names(parsed: dict, key: str, forbidden: set[str]) -> tuple[str, ...]:
    value = parsed.get(key)
    if not isinstance(value, list):
        raise AssignmentError(f"{key} must be a list of names")
    for name in value:
        if not isinstance(name, str) or not name.strip():
            raise AssignmentError(f"{key} entries must be nonempty strings, got {name!r}")
        if name in forbidden or "+" in name:
            raise AssignmentError(f"{key} entry {name!r} is not allowed")
    if len(set(value)) != len(value):
        raise AssignmentError(f"{key} has duplicate entries")
    return tuple(value)


def load_assignment(path: Path | None = None) -> Assignment:
    source = (
        path if path is not None else files("agentwatch_data").joinpath("split-assignment.toml")
    )
    try:
        parsed = tomllib.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as error:
        raise AssignmentError(f"Split assignment {source} cannot be read: {error}") from error
    except tomllib.TOMLDecodeError as error:
        raise AssignmentError(f"Split assignment is not valid TOML: {error}") from error
    if set(parsed) != ASSIGNMENT_KEYS:
        raise AssignmentError(
            f"Split assignment keys must be {sorted(ASSIGNMENT_KEYS)}, got {sorted(parsed)}"
        )
    version = parsed["version"]
    if not isinstance(version, str) or version != VERSION:
        raise AssignmentError(f"version must be {VERSION!r}, got {version!r}")
    return Assignment(
        version=version,
        held_out_injection_methods=names(parsed, "held_out_injection_methods", {NO_INJECTION}),
        test_task_groups=names(parsed, "test_task_groups", set()),
        sha256=canonical_sha256(parsed)[:8],
    )


def row_methods(row: dict) -> set[str]:
    """The injection methods of a row: a solo method, or the parts of a mixed one."""
    return set(str(row["injection_method"]).split("+")) - {NO_INJECTION}


def group_of(row: dict) -> str:
    return row.get("task_group") or row["task_id"]


def assign_groups(
    rows: list[dict], assignment: Assignment, fractions: dict[str, float]
) -> dict[str, str]:
    """Map each group to its split: declared and held-out groups to test, others by hash."""
    held_out = set(assignment.held_out_injection_methods)
    groups: dict[str, str] = {}
    for group in sorted({group_of(row) for row in rows}):
        groups[group] = "test" if group in assignment.test_task_groups else assign(group, fractions)
    for row in rows:
        if row_methods(row) & held_out:
            groups[group_of(row)] = "test"
    return groups


def independence(members: dict[str, list[dict]]) -> dict:
    """Fail when a task group, run, or plan appears in more than one split.

    Inside partition_v2 the task_group check cannot fail: assign_groups maps each group to one
    split. It protects only when D03 recomputes this check from a stored split. Sharing of
    run_id and plan_sha256 are the live checks there.
    """
    shared: dict[str, list[str]] = {}
    for key in INDEPENDENCE_KEYS:
        seen: dict[str, set[str]] = {}
        for split, rows in members.items():
            for row in rows:
                # A missing plan hash is unknown, not a shared value.
                value = group_of(row) if key == "task_group" else row.get(key)
                if value is not None:
                    seen.setdefault(value, set()).add(split)
        shared[key] = sorted(value for value, where in seen.items() if len(where) > 1)
    return {"status": "failed" if any(shared.values()) else "passed", "shared": shared}


def held_out_status(members: dict[str, list[dict]], assignment: Assignment) -> dict:
    """Whether each held-out method appears only in test while training can learn its class."""
    held_out = set(assignment.held_out_injection_methods)

    def methods(split: str) -> set[str]:
        return set().union(*(row_methods(row) for row in members.get(split, [])))

    outside_test = methods("train") | methods("validation")
    only_in_test = sorted((held_out & methods("test")) - outside_test)
    reason = None
    for method in sorted(held_out):
        if method in outside_test:
            reason = f"held-out method {method} appears outside the test split"
        elif method not in only_in_test:
            reason = f"held-out method {method} appears in no test row"
        else:
            classes = {
                row["label_resource"]
                for row in members.get("test", [])
                if method in row_methods(row)
            } - {NO_INJECTION, MIXED}
            trained = {
                row["label_resource"]
                for row in members.get("train", [])
                if row_methods(row) - held_out
            }
            missing = sorted(classes - trained)
            if not classes:
                # Nothing to contrast with training, so the hold-out is untested, not met.
                reason = (
                    f"held-out rows for {method} in test carry no contended label "
                    "to contrast with training"
                )
            elif missing:
                reason = f"training has no other method for {method}'s classes {missing}"
        if reason:
            break
    return {
        "status": "met" if reason is None else "not_met",
        "reason": reason,
        "only_in_test": only_in_test,
    }


def mixed_in_test(members: dict[str, list[dict]]) -> dict:
    count = sum(row["condition"] == MIXED for row in members.get("test", []))
    return {"status": "met" if count else "not_met", "rows": count}


def partition_v2(
    rows: list[dict], assignment: Assignment, fractions: dict[str, float]
) -> tuple[dict[str, str], dict[str, list[dict]], dict]:
    """Assign rows to splits; returns (groups, split -> rows, checks).

    Raises AssignmentError when a group, run, or plan is shared between splits. For a group
    that can never happen here (each group maps to one split); run_id and plan_sha256 sharing
    are the checks that can fire.
    """
    groups = assign_groups(rows, assignment, fractions)
    members: dict[str, list[dict]] = {name: [] for name in SPLITS}
    for row in sorted(rows, key=lambda r: (r["run_id"], str(r.get("step_id")))):
        members[groups[group_of(row)]].append(row)
    checks = {
        "independence": independence(members),
        "held_out": held_out_status(members, assignment),
        "mixed_in_test": mixed_in_test(members),
    }
    if checks["independence"]["status"] != "passed":
        shared = {k: v for k, v in checks["independence"]["shared"].items() if v}
        raise AssignmentError(f"Splits are not independent; shared values: {shared}")
    return groups, members, checks
