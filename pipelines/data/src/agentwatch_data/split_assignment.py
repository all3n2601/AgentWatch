"""Frozen `split-v2` and `split-v3` assignments and the independence checks that guard them.

splits.make_split and splits.load_split partition with partition_v2 when the evaluation
contract selects `split-v2` (eval-contract-v2) and with partition_v3 for `split-v3`
(eval-contract-v3); eval-contract-v1 names no method, so its `split-v1` splits are unchanged.
In both, the assignment file's declared test task groups and held-out injection methods go to
test, and every other group keeps its `split-v1` hash position (split_positions), so new tasks
never move frozen ones.

split-v3 (split-assignment-v3.toml) adds three rules, fixed before any label was seen:
- only rows of the listed feature definitions are split, so a model never mixes them;
- rows of a listed natural-overload sweep (by protocol hash) are held out of every split, as a
  separate evaluation set, and a natural-overload row of an unlisted sweep is refused;
- within a training task, whole experiments whose `split-v3:<experiment>` hash position is
  below the contract's validation share go to validation. Experiments keep their baselines
  and plan together, and test stays held out by task.

Rows group by `task_group` (revisions of a task share it) or, for rows without one such as the
CPU and coding release rows, by `task_id`.
"""

import hashlib
import re
import tomllib
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from agentwatch_data.extract import canonical_sha256
from agentwatch_data.labels import NO_INJECTION
from agentwatch_data.split_positions import SPLITS, assign

ASSIGNMENT_KEYS = {"version", "held_out_injection_methods", "test_task_groups"}
VERSION = "split-v2"
VERSION_V3 = "split-v3"
ASSIGNMENT_KEYS_V3 = ASSIGNMENT_KEYS | {"feature_versions", "held_out_protocol"}
PROTOCOL_KEYS = {"version", "sha256"}
SHA256 = re.compile(r"[0-9a-f]{64}")
MIXED = "mixed"
INDEPENDENCE_KEYS = ("task_group", "run_id", "plan_sha256")
# split-v3 validation takes experiments of training tasks, so those two may share a task.
TRAINING_SPLITS = ("train", "validation")


class AssignmentError(ValueError):
    """The assignment file is invalid, or the split violates independence."""


@dataclass(frozen=True)
class Assignment:
    version: str
    held_out_injection_methods: tuple[str, ...]
    test_task_groups: tuple[str, ...]
    sha256: str  # first 8 hex digits of the canonical hash of the parsed file
    # split-v3 only; empty under split-v2, which splits every row.
    feature_versions: tuple[str, ...] = ()
    held_out_protocol_version: str | None = None
    held_out_protocol_sha256: tuple[str, ...] = ()


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
    version = parsed.get("version")
    keys = ASSIGNMENT_KEYS_V3 if version == VERSION_V3 else ASSIGNMENT_KEYS
    if set(parsed) != keys:
        raise AssignmentError(f"Split assignment keys must be {sorted(keys)}, got {sorted(parsed)}")
    if version not in (VERSION, VERSION_V3):
        raise AssignmentError(f"version must be {VERSION!r} or {VERSION_V3!r}, got {version!r}")
    common = {
        "version": version,
        "held_out_injection_methods": names(parsed, "held_out_injection_methods", {NO_INJECTION}),
        "test_task_groups": names(parsed, "test_task_groups", set()),
        "sha256": canonical_sha256(parsed)[:8],
    }
    if version == VERSION:
        return Assignment(**common)
    feature_versions = names(parsed, "feature_versions", set())
    if not feature_versions:
        raise AssignmentError("feature_versions must name at least one feature definition")
    protocol_version, protocol_hashes = held_out_protocol_table(parsed["held_out_protocol"])
    return Assignment(
        **common,
        feature_versions=feature_versions,
        held_out_protocol_version=protocol_version,
        held_out_protocol_sha256=protocol_hashes,
    )


def held_out_protocol_table(table: object) -> tuple[str, tuple[str, ...]]:
    """The natural-overload protocol version and the sweep hashes held out under it."""
    if not isinstance(table, dict) or set(table) != PROTOCOL_KEYS:
        raise AssignmentError(f"[held_out_protocol] keys must be {sorted(PROTOCOL_KEYS)}")
    version, hashes = table["version"], table["sha256"]
    if not isinstance(version, str) or not version.strip():
        raise AssignmentError("[held_out_protocol] version must be a nonempty string")
    if not isinstance(hashes, list) or not hashes:
        raise AssignmentError("[held_out_protocol] sha256 must list at least one sweep hash")
    if not all(isinstance(value, str) and SHA256.fullmatch(value) for value in hashes):
        raise AssignmentError("[held_out_protocol] sha256 entries must be 64 lowercase hex digits")
    if len(set(hashes)) != len(hashes):
        raise AssignmentError("[held_out_protocol] sha256 has duplicate entries")
    return version, tuple(hashes)


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


def independence(members: dict[str, list[dict]], together: tuple[str, ...] = ()) -> dict:
    """Fail when a task group, run, or plan appears in more than one split.

    Inside partition_v2 the task_group check cannot fail: assign_groups maps each group to one
    split. It protects only when the check is recomputed from split members, as splits does
    after excluding untested sources. Sharing of run_id and plan_sha256 are the live checks.
    Splits named in `together` may share a task group (split-v3: train and validation), never
    a run or a plan.
    """
    shared: dict[str, list[str]] = {}
    for key in INDEPENDENCE_KEYS:
        seen: dict[str, set[str]] = {}
        for split, rows in members.items():
            where = "+".join(together) if key == "task_group" and split in together else split
            for row in rows:
                # A missing plan hash is unknown, not a shared value.
                value = group_of(row) if key == "task_group" else row.get(key)
                if value is not None:
                    seen.setdefault(value, set()).add(where)
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
    """Test rows recorded under the mixed injection condition, whatever their label.

    Release rows keep the replay role in `condition` and the injected condition in
    `replay_condition` (null on run rows); D02 label rows carry it in `condition`.
    """

    def condition(row: dict):
        return row["replay_condition"] if "replay_condition" in row else row["condition"]

    count = sum(condition(row) == MIXED for row in members.get("test", []))
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
    for row in sorted(rows, key=row_order):
        members[groups[group_of(row)]].append(row)
    return groups, members, independent(split_checks(members, assignment))


def row_order(row: dict) -> tuple[str, str]:
    return row["run_id"], str(row.get("step_id"))


def split_checks(members: dict[str, list[dict]], assignment: Assignment) -> dict:
    """The independence, hold-out and mixed-in-test checks on the rows each split holds."""
    together = TRAINING_SPLITS if assignment.version == VERSION_V3 else ()
    return {
        "independence": independence(members, together),
        "held_out": held_out_status(members, assignment),
        "mixed_in_test": mixed_in_test(members),
    }


def independent(checks: dict) -> dict:
    if checks["independence"]["status"] != "passed":
        shared = {k: v for k, v in checks["independence"]["shared"].items() if v}
        raise AssignmentError(f"Splits are not independent; shared values: {shared}")
    return checks


def validation_share(fractions: dict[str, float]) -> float:
    """The share of a training task's experiments that split-v3 sends to validation."""
    return fractions["validation"] / (fractions["train"] + fractions["validation"])


def experiment_position(experiment: str) -> float:
    digest = hashlib.sha256(f"{VERSION_V3}:{experiment}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def held_out_protocol(row: dict, assignment: Assignment) -> bool:
    """Whether a row is from a held-out natural-overload sweep; an unlisted one is refused.

    A listed hash is held out whatever its protocol version (the preflight dry run is listed
    but is not the protocol). A row of the protocol with an unlisted hash cannot be placed
    safely, so it is refused rather than split.
    """
    sha256 = row.get("protocol_sha256")
    if sha256 is not None and sha256 in assignment.held_out_protocol_sha256:
        return True
    if row.get("protocol_version") == assignment.held_out_protocol_version:
        raise AssignmentError(
            f"Row of run {row['run_id']} is from a {assignment.held_out_protocol_version} sweep "
            f"whose protocol hash {sha256!r} is not listed in {assignment.version}"
        )
    return False


MISSING_FEATURE_VERSION = "<missing>"


def excluded_counts(rows: list[dict]) -> dict[str, dict[str, int]]:
    """Rows and runs left out of every split, per feature definition."""
    by_version: dict[str, list[dict]] = {}
    for row in rows:
        version = row.get("feature_version")
        key = MISSING_FEATURE_VERSION if version is None else str(version)
        by_version.setdefault(key, []).append(row)
    return {
        version: {"rows": len(members), "runs": len({row["run_id"] for row in members})}
        for version, members in sorted(by_version.items())
    }


def held_out_summary(rows: list[dict]) -> dict:
    """The separate natural-overload evaluation set; never trained, calibrated or tuned on."""
    return {
        "rows": len(rows),
        "runs": len({row["run_id"] for row in rows}),
        "task_groups": sorted({group_of(row) for row in rows}),
        "protocol_sha256": sorted({row["protocol_sha256"] for row in rows}),
    }


def require_held_out_apart(held_out: list[dict], members: dict[str, list[dict]]) -> None:
    """The held-out set may share a task with any split, never a run or a plan."""
    for key in ("run_id", "plan_sha256"):
        split_values = {row.get(key) for rows in members.values() for row in rows} - {None}
        shared = sorted({row.get(key) for row in held_out} & split_values)
        if shared:
            raise AssignmentError(
                f"The held-out evaluation set shares {key} with the splits: {shared[:5]}"
            )


def require_held_out_methods_in_test(rows: list[dict], assignment: Assignment) -> None:
    """split-v3 never moves a training task to test for a held-out method; it refuses instead.

    Moving the only training task to test (as split-v2 would) would empty training. Rows of a
    held-out method on a non-test task need a decision before they reach a release.
    """
    held_out = set(assignment.held_out_injection_methods)
    for row in rows:
        method = row_methods(row) & held_out
        if method and group_of(row) not in assignment.test_task_groups:
            raise AssignmentError(
                f"Row of run {row['run_id']} uses held-out method {sorted(method)} on task "
                f"{group_of(row)!r}, which {assignment.version} does not declare as a test task"
            )


def partition_v3(
    rows: list[dict], assignment: Assignment, fractions: dict[str, float]
) -> tuple[dict[str, str], dict[str, list[dict]], dict]:
    """Assign rows under split-v3; returns (groups, split -> rows, checks).

    `groups` is task-level: a training task maps to train even though some of its experiments
    are in validation, which the checks list as `validation_experiments`. Natural-overload rows
    and rows of other feature definitions are in no split; the checks record them. Raises
    AssignmentError when a run or plan is shared between splits or with the held-out set, a
    task group between test and the training splits, or a held-out method appears on a task
    that is not a declared test task.
    """
    if assignment.version != VERSION_V3:
        raise AssignmentError(
            f"partition_v3 needs a {VERSION_V3} assignment, not {assignment.version}"
        )
    held_out, other_features, kept = [], [], []
    for row in rows:
        if held_out_protocol(row, assignment):
            held_out.append(row)
        elif row.get("feature_version") not in assignment.feature_versions:
            other_features.append(row)
        else:
            kept.append(row)
    groups = assign_groups(kept, assignment, fractions)
    require_held_out_methods_in_test(kept, assignment)
    share = validation_share(fractions)
    members: dict[str, list[dict]] = {name: [] for name in SPLITS}
    for row in sorted(kept, key=row_order):
        split = groups[group_of(row)]
        experiment = row.get("experiment")
        if split == "train" and experiment and experiment_position(experiment) < share:
            split = "validation"
        members[split].append(row)
    checks = independent(split_checks(members, assignment))
    require_held_out_apart(held_out, members)
    return (
        groups,
        members,
        checks
        | {
            "validation_experiments": sorted(
                {row["experiment"] for row in members["validation"] if row.get("experiment")}
            ),
            "excluded_feature_versions": excluded_counts(other_features),
            "held_out_evaluation": held_out_summary(held_out),
        },
    )
