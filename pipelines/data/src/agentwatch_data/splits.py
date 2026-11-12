"""Freeze the train, validation, and test splits of a dataset release and check them for leakage.

    python -m agentwatch_data.splits --release data/datasets/<version>

Only a release whose latest quality-gate verdict is a pass, for exactly the files it holds now
and the reviewed frozen schema, can be split. The split method, fractions, group key, test
sources, class minimums, and leakage bound come from the evaluation contract
(pipelines/model/evaluation-contract.toml), so the contract stays the single source of those
values.

Under `split-v1` every group (task) is assigned by hashing its key (split_positions), so under
the same fractions a task keeps its split in every later release: a task once tested is never
trained on, and new tasks never move old ones. The fractions are part of the split's identity;
changing them moves tasks near the boundaries, so it needs a new contract version and test
split. `split-v2` (eval-contract-v2) groups by task_group and partitions with
split_assignment.partition_v2: the frozen assignment file's declared test groups and held-out
injection methods go to test, every other group keeps its `split-v1` hash position, and a run
or plan shared between splits refuses the split. `split-v3` (eval-contract-v3) partitions with
split_assignment.partition_v3: the same rules, plus only the assignment's feature definitions
are split, rows of listed natural-overload sweeps are held out of every split, and whole
experiments of a training task go to validation by their `split-v3` hash position. Under every
method, runs whose source the contract does not allow in the test split are excluded there, not
moved, so a task's replays never cross splits.

The split is written to <datasets>/splits/<release>/<split id>.json; the id is prefixed with
the split method. Its content is deterministic and the split id is derived from it, so
rerunning reuses the same file. It records the task assignment, a hash of each split's rows
(the "test-split hash" evaluation reports record), counts, held-out injection methods, and the
contract's leakage check on the training split. A `split-v2` or `split-v3` file also records the assignment
file's version and hash, the D02 independence, hold-out, and mixed-in-test checks, the test
split's class balance against the contract minimums, missing feature readings per feature
version, and a report-only design-encoding check. A blocked or inconclusive leakage check does
not fail the split, which is kept as evidence, but load_split refuses to hand its rows to
training or evaluation. load_split also rechecks the release's latest quality-gate verdict and
rebuilds the split on every read, so neither a split made before a later failed validation, an
edited split file, nor a changed assignment file can be used to bypass the gate or the checks.
"""

import argparse
import hashlib
import json
import math
import re
import sys
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from agentwatch_data.build import (
    DATASET_FILE,
    MANIFEST_FILE,
    RELEASE_FORMAT,
    SCHEMA_FILE,
    counts,
)
from agentwatch_data.extract import canonical_sha256
from agentwatch_data.files import file_sha256, json_text, write_atomic
from agentwatch_data.labels import NO_INJECTION
from agentwatch_data.schema import RESOURCE_CLASSES, SOURCES, feature_columns
from agentwatch_data.split_assignment import (
    MIXED,
    Assignment,
    AssignmentError,
    load_assignment,
    mixed_in_test,
    partition_v2,
    partition_v3,
    split_checks,
)
from agentwatch_data.split_assignment import VERSION as SPLIT_V2
from agentwatch_data.split_assignment import VERSION_V3 as SPLIT_V3
from agentwatch_data.split_positions import HASH_METHOD, SPLITS, assign

SPLIT_METHOD = HASH_METHOD  # split-v1: groups are assigned by their hash position alone
# split-v2 and split-v3 partition by a frozen assignment file instead of hashing alone
# (split_assignment).
METHODS = (SPLIT_METHOD, SPLIT_V2, SPLIT_V3)
PARTITIONS = {SPLIT_V2: partition_v2, SPLIT_V3: partition_v3}
NO_RESOURCE = RESOURCE_CLASSES[0]  # "none": the class of a step no resource slowed
# Replay roles the design-encoding report contrasts: injected load against its clean control.
CONTENDED_ROLE, CONTROL_ROLE = "contended", "clean_control"
SPLITS_DIR, VALIDATIONS_DIR, RESULT_FILE = "splits", "validations", "result.json"
FEATURES = tuple(feature_columns())  # the schema decides what may be a model input
# Grouping must keep a task's baselines and replays together, so only task-level keys qualify.
GROUP_KEYS = ("task_id", "task_group")
VERSION = re.compile(r"(?P<name>[a-z]+)-v(?P<major>\d+)\.(?P<minor>\d+)")
PASSED, BLOCKED, INCONCLUSIVE = "passed", "blocked", "inconclusive"


class SplitError(RuntimeError):
    """The release cannot be split under the contract; nothing was written."""


@dataclass(frozen=True)
class Contract:
    version: str
    status: str
    sha256: str  # the contract identifier: first 8 hex digits, as evaluation reports record it
    schema_version: str
    accept_compatible_minor_versions: bool
    label_rules: tuple[str, ...]  # label-rule versions evaluated, each pinned exactly
    classes: tuple[str, ...]
    target: str
    test_sources: tuple[str, ...]
    method: str
    group_key: str
    # The frozen split-assignment file, resolved against the contract's directory; split-v2 only.
    assignment: Path | None
    fractions: dict[str, float]
    held_out_injection_methods: int
    min_test_steps_per_class: int
    min_clean_test_steps: int
    min_mixed_test_steps: int
    max_single_feature_auc: float


@dataclass(frozen=True)
class SplitResult:
    path: Path
    split: dict
    reused: bool


def require(table: dict, key: str, kind: type, where: str):
    value = table.get(key)
    # bool is an int subclass, and an integer is not a fraction, so types are matched exactly.
    if type(value) is not kind:
        raise SplitError(f"Contract {where}.{key} must be {kind.__name__}")
    return value


def count(table: dict, key: str, where: str) -> int:
    value = require(table, key, int, where)
    if value < 0:
        raise SplitError(f"Contract {where}.{key} must be >= 0")
    return value


def label_rules(dataset: dict) -> tuple[str, ...]:
    """eval-contract-v1 names one rule as a string; later contracts list several."""
    value = dataset.get("label_rules")
    rules = [value] if isinstance(value, str) else value
    if not isinstance(rules, list) or not rules or not all(isinstance(r, str) and r for r in rules):
        raise SplitError("Contract dataset.label_rules must be a name or a non-empty list of names")
    return tuple(rules)


def split_method(splits: dict, path: Path) -> tuple[str, Path | None]:
    """The split method and its assignment file; contracts before split-v2 name no method."""
    method = require(splits, "method", str, "splits") if "method" in splits else SPLIT_METHOD
    if method not in METHODS:
        raise SplitError(f"Contract split method {method!r} is not one of {list(METHODS)}")
    if method == SPLIT_METHOD:
        if "assignment" in splits:
            raise SplitError(
                f"Contract method {method} hashes groups and must not name an assignment"
            )
        return method, None
    if "assignment" not in splits:
        raise SplitError(f"Contract method {method} needs an assignment file in [splits]")
    return method, (path.parent / require(splits, "assignment", str, "splits")).resolve()


def load_contract(path: Path) -> Contract:
    try:
        parsed = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise SplitError(f"Evaluation contract {path} is unreadable: {error}") from error
    except tomllib.TOMLDecodeError as error:
        raise SplitError(f"Evaluation contract {path} is not valid TOML: {error}") from error
    tables = {}
    for name in ("dataset", "splits", "leakage"):
        tables[name] = parsed.get(name)
        if not isinstance(tables[name], dict):
            raise SplitError(f"Contract is missing its [{name}] table")
    dataset, splits, leakage = tables["dataset"], tables["splits"], tables["leakage"]
    classes = tuple(require(dataset, "classes", list, "dataset"))
    if classes != RESOURCE_CLASSES:
        raise SplitError(f"Contract classes {list(classes)} differ from {list(RESOURCE_CLASSES)}")
    test_sources = tuple(require(dataset, "test_sources", list, "dataset"))
    if not test_sources or not set(test_sources) <= set(SOURCES):
        raise SplitError(f"Contract dataset.test_sources must list sources from {list(SOURCES)}")
    target = require(dataset, "classification_target", str, "dataset")
    if target != "label_resource":
        raise SplitError(f"Contract classification target {target!r} is not label_resource")
    group_key = require(splits, "group_key", str, "splits")
    if group_key not in GROUP_KEYS:
        raise SplitError(f"Contract group key {group_key!r} is not one of {list(GROUP_KEYS)}")
    fractions = {name: require(splits, f"{name}_fraction", float, "splits") for name in SPLITS}
    if not all(0 < value < 1 for value in fractions.values()) or not math.isclose(
        sum(fractions.values()), 1.0
    ):
        raise SplitError(f"Contract split fractions must be in (0, 1) and sum to 1: {fractions}")
    method, assignment = split_method(splits, path)
    if method in PARTITIONS and group_key != "task_group":
        raise SplitError(f"{method} requires group_key = task_group")
    held_out = require(splits, "held_out_injection_methods", int, "splits")
    bound = require(leakage, "max_single_feature_auc", float, "leakage")
    if held_out < 0 or not 0.5 <= bound <= 1:
        raise SplitError("Contract held-out methods must be >= 0 and the AUC bound in [0.5, 1]")
    return Contract(
        version=require(parsed, "version", str, "contract"),
        status=require(parsed, "status", str, "contract"),
        sha256=canonical_sha256(parsed)[:8],
        schema_version=require(dataset, "schema_version", str, "dataset"),
        accept_compatible_minor_versions=require(
            dataset, "accept_compatible_minor_versions", bool, "dataset"
        ),
        label_rules=label_rules(dataset),
        classes=classes,
        target=target,
        test_sources=test_sources,
        method=method,
        group_key=group_key,
        assignment=assignment,
        fractions=fractions,
        held_out_injection_methods=held_out,
        min_test_steps_per_class=count(splits, "min_test_steps_per_class", "splits"),
        min_clean_test_steps=count(splits, "min_clean_test_steps", "splits"),
        min_mixed_test_steps=count(splits, "min_mixed_contention_test_steps", "splits"),
        max_single_feature_auc=bound,
    )


def check_compatibility(contract: Contract, manifest: dict) -> None:
    """The release's schema and label rules must be ones the contract evaluates."""
    schema, label = manifest.get("schema_version"), manifest.get("label_version")
    wanted, found = VERSION.fullmatch(contract.schema_version), VERSION.fullmatch(str(schema))
    if wanted is None:
        raise SplitError(f"Contract schema version {contract.schema_version!r} is malformed")
    compatible = schema == contract.schema_version or (
        contract.accept_compatible_minor_versions
        and found is not None
        and found["name"] == wanted["name"]
        and found["major"] == wanted["major"]
        and int(found["minor"]) >= int(wanted["minor"])
    )
    if not compatible:
        raise SplitError(f"Schema {schema} is not accepted by {contract.version}")
    # A release holding several label versions names them comma-joined. Each must be one the
    # contract pins exactly; only the fingerprint after "+" may vary.
    bases = {part.split("+", 1)[0] for part in str(label).split(",")}
    if not bases <= set(contract.label_rules):
        raise SplitError(f"Labels {label} are not accepted by {contract.version}")


def read_object(path: Path, what: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError) as error:  # RecursionError: deep nesting
        raise SplitError(f"{what} {path.name} is unreadable: {type(error).__name__}") from error
    if not isinstance(value, dict):
        raise SplitError(f"{what} {path.name} is not a JSON object")
    return value


def verify_release(path: Path) -> dict:
    """A release's manifest, after verifying its files and name against it."""
    if not (path / DATASET_FILE).is_file():
        raise SplitError(f"{path} is not a dataset release")
    manifest = read_object(path / MANIFEST_FILE, "Manifest")
    intact = (
        manifest.get("release_format") in (2, RELEASE_FORMAT)
        and manifest.get("dataset_version") == path.name
        and manifest.get("parquet_sha256") == file_sha256(path / DATASET_FILE)
        and (path / SCHEMA_FILE).is_file()
        and manifest.get("schema_sha256") == file_sha256(path / SCHEMA_FILE)
    )
    if not intact:
        raise SplitError(f"Release {path.name} does not match its manifest; rebuild it")
    return manifest


def load_release(path: Path) -> tuple[dict, list[dict]]:
    """A release's manifest and rows, after verifying its files against the manifest."""
    manifest = verify_release(path)
    try:
        return manifest, pq.read_table(path / DATASET_FILE).to_pylist()
    # Hash-consistent but not Parquet, or a timestamp beyond what Python can represent.
    except (pa.ArrowException, OSError, ValueError, OverflowError) as error:
        raise SplitError(f"Release {path.name}: {DATASET_FILE} is not readable Parquet") from error


def text_sha256(path: Path) -> str:
    """The frozen schema's hash as the quality gate records it: line endings normalized."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise SplitError(f"Frozen schema {path} is unreadable: {error}") from error
    return hashlib.sha256(text.replace("\r\n", "\n").encode()).hexdigest()


def validated_time(result: dict) -> datetime | None:
    try:
        moment = datetime.fromisoformat(result["validated_at"])
    except (KeyError, TypeError, ValueError):
        return None
    return moment if moment.tzinfo is not None else None


def latest_verdict(release: Path, now: datetime | None = None) -> dict | None:
    """The release's latest committed quality-gate attempt, by the gate's own rules.

    The gate commits each attempt as validations/<release>/<attempt>/result.json through a
    rename; staging directories are hidden. A committed attempt that is missing its result,
    unreadable, malformed, or validated in the future makes the verdict unknown, so an older
    attempt can never stand in for it. The gate's own reader (committed_attempts in gate.py)
    runs in another environment; both suites run pipelines/validation/tests/verdict_cases.json,
    so the two cannot drift apart.
    """
    now = now or datetime.now(UTC)
    attempts, damaged = [], []
    base = release.parent / VALIDATIONS_DIR / release.name
    for directory in sorted(base.iterdir()) if base.is_dir() else []:
        if directory.name.startswith(".") or not directory.is_dir():
            continue
        try:
            result = json.loads((directory / RESULT_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError, RecursionError):  # RecursionError: deep nesting
            result = None
        moment = validated_time(result) if isinstance(result, dict) else None
        if (
            moment is None
            or result.get("attempt") != directory.name
            or result.get("release") != release.name
            or result.get("status") not in ("passed", "failed")
        ):
            damaged.append(directory.name)
        elif moment > now:
            damaged.append(f"{directory.name} (validated in the future)")
        else:
            attempts.append((moment, result["attempt"], result))
    if damaged:
        raise SplitError(
            f"Quality-gate verdict of {release.name} is unknown: damaged attempts {damaged}"
        )
    return max(attempts, key=lambda item: item[:2])[2] if attempts else None


def require_passing_verdict(release: Path, manifest: dict, frozen_schema: Path) -> None:
    verdict = latest_verdict(release)
    rerun = "run make validate-dataset"
    if verdict is None:
        raise SplitError(f"Release {release.name} has no quality-gate verdict; {rerun}")
    if verdict.get("status") != "passed":
        raise SplitError(
            f"Release {release.name} failed its latest quality gate ({verdict['attempt']})"
        )
    if (
        verdict.get("release_parquet_sha256") != manifest["parquet_sha256"]
        or verdict.get("release_contract_sha256") != manifest["schema_sha256"]
    ):
        raise SplitError(f"Release {release.name} changed since it was validated; {rerun}")
    if verdict.get("schema_sha256") != text_sha256(frozen_schema):
        raise SplitError(
            f"Release {release.name} was validated against a different schema than "
            f"{frozen_schema.name}; {rerun}"
        )


def exclude_untested(members: dict[str, list[dict]], contract: Contract) -> list[dict]:
    """Drop test rows whose source the contract does not allow there; returns their runs.

    The rows are excluded, not moved, so a task's replays never cross splits.
    """
    excluded = {}
    kept = []
    for row in members["test"]:
        if row["source"] in contract.test_sources:
            kept.append(row)
            continue
        excluded[row["run_id"]] = {
            "run_id": row["run_id"],
            "source": row["source"],
            "reason": f"source {row['source']} is not allowed in the test split",
        }
    members["test"] = kept
    return [excluded[run] for run in sorted(excluded)]


def partition(rows: list[dict], contract: Contract) -> tuple[dict, dict, list[dict]]:
    """Assign rows to splits by the hash of their group (split-v1).

    Returns (group -> split, split -> rows, excluded runs). A run must belong to one group,
    or its steps could land in different splits.
    """
    run_groups: dict[str, set] = {}
    for row in rows:
        run_groups.setdefault(row["run_id"], set()).add(row[contract.group_key])
    straddling = sorted(run for run, groups in run_groups.items() if len(groups) > 1)
    if straddling:
        raise SplitError(f"Runs with more than one {contract.group_key}: {straddling[:5]}")
    groups = {
        group: assign(group, contract.fractions) for group in {r[contract.group_key] for r in rows}
    }
    members: dict[str, list[dict]] = {name: [] for name in SPLITS}
    for row in rows:
        members[groups[row[contract.group_key]]].append(row)
    excluded = exclude_untested(members, contract)
    return dict(sorted(groups.items())), members, excluded


def require_group_key(rows: list[dict], contract: Contract) -> None:
    # task_group exists from steps-v1.3; an older release cannot be grouped by it.
    if any(contract.group_key not in row for row in rows):
        raise SplitError(
            f"Release rows have no {contract.group_key} column, which {contract.version} groups by"
        )


def assignment_for(contract: Contract) -> Assignment | None:
    """The contract's frozen split assignment; split-v1 has none."""
    if contract.method == SPLIT_METHOD:
        return None
    if contract.assignment is None:
        raise SplitError(f"Contract method {contract.method} needs an assignment file")
    try:
        assignment = load_assignment(contract.assignment)
    except AssignmentError as error:
        raise SplitError(str(error)) from error
    if assignment.version != contract.method:
        raise SplitError(
            f"Contract {contract.version} names a {assignment.version} assignment, "
            f"not {contract.method}"
        )
    return assignment


def members_for(
    rows: list[dict], contract: Contract, assignment: Assignment | None = None
) -> tuple[dict, dict[str, list[dict]], list[dict], dict | None]:
    """(group -> split, split -> rows, excluded runs, assignment checks) under the contract.

    split-v1 hashes groups (partition) and has no assignment checks. split-v2 and split-v3
    partition with the frozen assignment file (assignment, or the contract's file when None),
    exclude untested sources the same way, then record the independence, hold-out, and
    mixed-in-test checks on the rows each split keeps; split-v3 also records the rows it left
    out of every split.
    """
    require_group_key(rows, contract)
    if contract.method == SPLIT_METHOD:
        return *partition(rows, contract), None
    assignment = assignment or assignment_for(contract)
    try:
        groups, members, checks = PARTITIONS[contract.method](rows, assignment, contract.fractions)
    except AssignmentError as error:
        raise SplitError(str(error)) from error
    excluded = exclude_untested(members, contract)
    # Recomputed after untested sources are excluded; these, not the partition's, are recorded.
    return groups, members, excluded, checks | split_checks(members, assignment)


def rows_sha256(dataset_version: str, rows: list[dict]) -> str:
    """Identify a split's rows: the release plus each row's (run_id, step_id)."""
    keys = sorted([row["run_id"], row["step_id"]] for row in rows)
    return canonical_sha256({"dataset_version": dataset_version, "rows": keys})


def missing_features(rows: list[dict]) -> dict[str, dict[str, int]]:
    """Null count of each feature per feature_version: the definitions differ, so not pooled."""
    versions = sorted({row["feature_version"] for row in rows})
    return {
        version: {
            feature: sum(row[feature] is None for row in rows if row["feature_version"] == version)
            for feature in FEATURES
        }
        for version in versions
    }


def summary(rows: list[dict], group_key: str) -> dict:
    return {
        "rows": len(rows),
        "runs": len({row["run_id"] for row in rows}),
        "groups": len({row[group_key] for row in rows}),
        "label_resource": counts(rows, "label_resource"),
        "condition": counts(rows, "condition"),
        "source": counts(rows, "source"),
        "injection_method": counts(rows, "injection_method"),
    }


def separation(values: list, positive: list[bool]) -> float:
    """One-vs-rest AUC of a single feature, folded so 1.0 means perfect separation either way.

    Mann-Whitney U with average ranks for ties. A missing value ranks below every reading,
    so missingness that tracks the label is detected too.
    """

    def key(value):
        return (0, 0) if value is None else (1, value)

    order = sorted(range(len(values)), key=lambda index: key(values[index]))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and key(values[order[end + 1]]) == key(values[order[start]]):
            end += 1
        for index in order[start : end + 1]:
            ranks[index] = (start + end) / 2 + 1
        start = end + 1
    positives = sum(positive)
    negatives = len(values) - positives
    rank_sum = sum(rank for rank, is_positive in zip(ranks, positive, strict=True) if is_positive)
    auc = (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
    return max(auc, 1 - auc)


def is_member(row: dict, name: str) -> bool:
    """Whether a row belongs to a class, by its label_resources when it has them.

    A mixed step (label_resources "cpu_sandbox,inference") belongs to each of its resources,
    and "none" means no resource ("" in label_resources). Rows without label_resources (before
    steps-v1.3) and CPU and coding rows, whose label_resources equal label_resource, are
    members of their label_resource only, so split-v1 results do not change.
    """
    resources = row.get("label_resources")
    if resources is None:
        return row["label_resource"] == name
    if name == NO_RESOURCE:
        return resources == ""
    return name in resources.split(",")


def leakage_check(train: list[dict], contract: Contract) -> dict:
    """The contract's check: no single feature may nearly separate a class on training data.

    Each class is one-vs-rest on membership (is_member), so a mixed step is a positive for
    every resource it was slowed by rather than label noise against them.
    """
    bound = contract.max_single_feature_auc
    membership = {name: [is_member(row, name) for row in train] for name in contract.classes}
    classes = [name for name in contract.classes if 0 < sum(membership[name]) < len(train)]
    check = {"max_single_feature_auc": bound, "classes": classes, "features": []}
    if not classes:
        reason = "the training split is empty" if not train else "it has only one class"
        return check | {"status": INCONCLUSIVE, "reason": reason}
    leaking = []
    for feature in FEATURES:
        values = [row[feature] for row in train]
        scores = {name: separation(values, membership[name]) for name in classes}
        worst = max(classes, key=lambda name: scores[name])
        if scores[worst] > bound:  # compared before rounding, so nothing just above passes
            leaking.append(feature)
        check["features"].append(
            {"feature": feature, "class": worst, "auc": round(scores[worst], 6)}
        )
    check["features"].sort(key=lambda item: (-item["auc"], item["feature"]))
    leaking.sort()
    if leaking:
        return check | {"status": BLOCKED, "reason": f"features above {bound}: {leaking}"}
    return check | {"status": PASSED, "reason": None}


def held_out_methods(members: dict[str, list[dict]], required: int) -> dict:
    """Whether enough injection methods appear only in the test split.

    A method counts as held out only while training has another method to learn from;
    otherwise there is no contention in training at all, not a held-out method.
    """

    def methods(rows):
        return {row["injection_method"] for row in rows} - {NO_INJECTION}

    trained = methods(members["train"])
    only_in_test = sorted(methods(members["test"]) - trained - methods(members["validation"]))
    if len(only_in_test) < required:
        reason = f"{len(only_in_test)} of {required} required methods appear only in test"
    elif required and not trained:
        reason = "the training split has no injection method to contrast with"
    else:
        reason = None
    return {
        "required": required,
        "methods": sorted(trained | methods(members["validation"]) | methods(members["test"])),
        "only_in_test": only_in_test,
        "status": "met" if reason is None else "not_met",
        "reason": reason,
    }


def class_balance(test: list[dict], contract: Contract) -> dict:
    """The test split's class counts against the contract minimums.

    A class counts the steps whose label_resource is that class. `mixed` counts steps labeled
    mixed (confirmed slowed by two or more resources), not every step of a mixed-condition
    replay; those rows are recorded beside it as mixed_condition_rows.
    """
    per_class = {
        name: sum(row["label_resource"] == name for row in test) for name in contract.classes
    }
    clean = sum(row["condition"] == "clean" for row in test)
    mixed = sum(row["label_resource"] == MIXED for row in test)
    minimums = {
        "per_class": contract.min_test_steps_per_class,
        "clean": contract.min_clean_test_steps,
        "mixed": contract.min_mixed_test_steps,
    }
    below = [name for name, found in per_class.items() if found < minimums["per_class"]]
    below += [name for name, found in (("clean", clean), (MIXED, mixed)) if found < minimums[name]]
    return {
        "per_class": per_class,
        "clean": clean,
        "mixed": mixed,
        "mixed_condition_rows": mixed_in_test({"test": test})["rows"],
        "minimums": minimums,
        "below": below,
        "status": "met" if not below else "not_met",
    }


def design_encoding(train: list[dict]) -> dict:
    """Report-only: how well each feature tells contended from clean-control replay steps.

    Only training replay steps labeled none are compared, so a high AUC means a feature encodes
    the experiment design (injected load) rather than the label. It is not a gate: live
    telemetry legitimately sees injected load on steps the load did not slow.
    """
    rows = [
        row
        for row in train
        if row["kind"] == "agent_replay"
        and row["label_resource"] == NO_RESOURCE
        and row["replay_role"] in (CONTENDED_ROLE, CONTROL_ROLE)
    ]
    contended = [row["replay_role"] == CONTENDED_ROLE for row in rows]
    if not any(contended) or all(contended):
        return {
            "status": "unavailable",
            "reason": "training has no contended and clean-control replay steps labeled none "
            "to compare",
        }
    features = [
        {"feature": feature, "auc": round(separation([row[feature] for row in rows], contended), 6)}
        for feature in FEATURES
    ]
    features.sort(key=lambda item: (-item["auc"], item["feature"]))
    return {"status": "report_only", "rows": len(rows), "features": features}


def checks_key(method: str) -> str:
    """The split document key of an assignment's checks: split_v2_checks, split_v3_checks."""
    return f"{method.replace('-', '_')}_checks"


def describe(
    release: Path,
    manifest: dict,
    rows: list[dict],
    contract: Contract,
    schema_sha256: str,
    assignment: Assignment | None = None,
) -> dict:
    """The split document; split-v2 and later read the contract's assignment unless given one."""
    assignment = assignment or assignment_for(contract)
    groups, members, excluded, assignment_checks = members_for(rows, contract, assignment)
    version = manifest["dataset_version"]
    by_assignment = assignment_checks is not None  # split-v2 and later use an assignment
    document = {
        "split_method": contract.method,
        "release": release.name,
        "dataset_version": version,
        "label_version": manifest["label_version"],
        "release_parquet_sha256": manifest["parquet_sha256"],
        "frozen_schema_sha256": schema_sha256,
        "contract": {
            "version": contract.version,
            "status": contract.status,
            "sha256": contract.sha256,
            "group_key": contract.group_key,
            "fractions": contract.fractions,
            "test_sources": list(contract.test_sources),
        },
        "groups": groups,
        "excluded_runs": excluded,
        "splits": {
            name: summary(members[name], contract.group_key)
            | {"rows_sha256": rows_sha256(version, members[name])}
            | ({"missing": missing_features(members[name])} if by_assignment else {})
            for name in SPLITS
        },
        "held_out_injection_methods": held_out_methods(
            members, contract.held_out_injection_methods
        ),
        "leakage": leakage_check(members["train"], contract),
    }
    if by_assignment:
        # The assignment file is outside the contract's hash, so its own hash is recorded:
        # a changed assignment changes the rebuilt document, and load_split refuses the file.
        document |= {
            "assignment": {"version": assignment.version, "sha256": assignment.sha256},
            checks_key(contract.method): assignment_checks,
            "class_balance": class_balance(members["test"], contract),
            "design_encoding": design_encoding(members["train"]),
        }
    split_id = f"{contract.method}+{canonical_sha256(document)[:12]}"
    return {"split_id": split_id} | document


def make_split(release: Path, contract_path: Path, frozen_schema: Path) -> SplitResult:
    """Split a release that passed the quality gate, or reuse its identical existing split."""
    contract = load_contract(contract_path)
    manifest, rows = load_release(release)
    check_compatibility(contract, manifest)
    require_passing_verdict(release, manifest, frozen_schema)
    split = describe(release, manifest, rows, contract, text_sha256(frozen_schema))
    path = release.parent / SPLITS_DIR / release.name / f"{split['split_id']}.json"
    text = json_text(split)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise SplitError(f"Existing split {path.name} does not match its content")
        return SplitResult(path, split, reused=True)
    write_atomic(path, text)
    return SplitResult(path, split, reused=False)


def read_split_file(path: Path) -> dict:
    """A split file whose id is the hash of its content and matches its name and folder.

    This proves the file is whole and in place, not that it is the right split: load_split
    also rebuilds it from the release and contract.
    """
    split = read_object(path, "Split")
    method = split.get("split_method")
    if method not in METHODS:
        raise SplitError(f"Split {path.name} names unknown split method {method!r}")
    content = {key: value for key, value in split.items() if key != "split_id"}
    try:
        expected = f"{method}+{canonical_sha256(content)[:12]}"
    except ValueError as error:  # json accepts NaN and Infinity, which canonical JSON refuses
        raise SplitError(f"Split {path.name} holds a non-finite number") from error
    if (
        split.get("split_id") != expected
        or path.stem != split["split_id"]
        or path.parent.name != split.get("release")
    ):
        raise SplitError(f"Split {path.name} does not match its content or location")
    return split


def load_split(
    path: Path,
    name: str,
    contract_path: Path,
    frozen_schema: Path,
    allow_unchecked_leakage: bool = False,
) -> list[dict]:
    """The rows of one split, after proving the split file is exactly what it should be.

    Training and evaluation should read splits through this. The release's latest
    quality-gate verdict must still be a pass against frozen_schema, and the split is rebuilt
    from the release, the contract, and that schema: any difference from the file, whether an
    edited assignment, exclusion, leakage status, or contract field, is refused, even when the
    file's id was recomputed to match. There is no way to skip these checks. A split whose
    leakage check did not pass is refused too, as the contract blocks evaluation on it;
    exploration may pass allow_unchecked_leakage=True, which skips only that check, and its
    results must not be reported as an evaluation.
    """
    if name not in SPLITS:
        raise SplitError(f"Unknown split {name!r}; expected one of {list(SPLITS)}")
    split = read_split_file(path)
    contract = load_contract(contract_path)
    require_contract_of(split, path, contract, contract_path)
    release = path.parents[2] / path.parent.name
    manifest, rows = load_release(release)
    check_compatibility(contract, manifest)
    # The verdict can change after the split was made, so it is checked on every read.
    require_passing_verdict(release, manifest, frozen_schema)
    # Read once, so the rows returned are partitioned by the assignment the rebuild verified.
    assignment = assignment_for(contract)
    expected = describe(release, manifest, rows, contract, text_sha256(frozen_schema), assignment)
    if json.loads(json_text(expected)) != split:
        raise SplitError(
            f"Split {path.name} is not what {release.name}, {contract.version}, and "
            f"{frozen_schema.name} produce; run make split-dataset"
        )
    leakage = expected["leakage"]["status"]
    if leakage != PASSED and not allow_unchecked_leakage:
        raise SplitError(
            f"Split {path.name}: its leakage check is {leakage}; the contract blocks evaluation"
        )
    return members_for(rows, contract, assignment)[1][name]


def require_contract_of(split: dict, path: Path, contract: Contract, contract_path: Path) -> None:
    """Refuse a split made under another contract, naming the contract to pass instead.

    Checked before the rebuild, so a split-v2 file read with the current contract is not met
    with advice to rebuild it, which would make a split of the current contract instead.
    """
    made_with = split.get("contract", {}).get("version")
    if split["split_method"] == contract.method and made_with == contract.version:
        return
    raise SplitError(
        f"Split {path.name} was made with {made_with} ({split['split_method']}), but "
        f"{contract_path} is {contract.version} ({contract.method}); pass the {made_with} file "
        f"(earlier contracts are kept as pipelines/model/evaluation-contract-v<N>.toml)"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument(
        "--contract", type=Path, default=Path("pipelines/model/evaluation-contract.toml")
    )
    parser.add_argument(
        "--frozen-schema",
        type=Path,
        default=Path("pipelines/validation/schema/steps-v1.3.pbtxt"),
    )
    args = parser.parse_args(argv)
    try:
        result = make_split(args.release, args.contract, args.frozen_schema)
    except SplitError as error:
        print(f"Split failed: {error}", file=sys.stderr)
        return 1
    split = result.split
    print(f"split: {result.path}{' (reused)' if result.reused else ''}")
    for name in SPLITS:
        counts = split["splits"][name]
        print(f"  {name}: {counts['rows']} rows, {counts['runs']} runs, {counts['groups']} groups")
    leakage = split["leakage"]
    print(
        f"leakage: {leakage['status']}" + (f" ({leakage['reason']})" if leakage["reason"] else "")
    )
    if leakage["status"] != PASSED:
        print("Evaluation must not use this split until the leakage check passes", file=sys.stderr)
    for line in assignment_lines(split):
        print(line)
    return 0


def assignment_lines(split: dict) -> list[str]:
    """The class balance and assignment check statuses; a split-v1 document has neither."""
    lines = []
    balance = split.get("class_balance")
    if balance is not None:
        minimums, per_class = balance["minimums"], balance["per_class"]
        below = [
            f"{name} {per_class[name]} < {minimums['per_class']}"
            if name in per_class
            else f"{name} {balance[name]} < {minimums[name]}"
            for name in balance["below"]
        ]
        lines.append(
            f"class balance: {balance['status']}"
            + (f" (below the minimum: {', '.join(below)})" if below else "")
        )
    method = split["split_method"]
    checks = split.get(checks_key(method))
    if checks is not None:
        independent, held_out, mixed = (
            checks["independence"],
            checks["held_out"],
            checks["mixed_in_test"],
        )
        shared = [f"{key} {len(found)}" for key, found in independent["shared"].items() if found]
        lines += [
            f"independence: {independent['status']}"
            + (f" (values in more than one split: {', '.join(shared)})" if shared else ""),
            f"held-out methods ({method}): {held_out['status']}"
            + (f" ({held_out['reason']})" if held_out["reason"] else ""),
            f"mixed in test: {mixed['status']} ({mixed['rows']} rows)",
        ]
    if checks is not None and "held_out_evaluation" in checks:
        left_out = checks["excluded_feature_versions"]
        lines += [
            "left out (other feature definitions): "
            + (", ".join(f"{name} {n['rows']} rows" for name, n in left_out.items()) or "none"),
            f"held-out evaluation set: {checks['held_out_evaluation']['rows']} rows",
        ]
    return lines


if __name__ == "__main__":
    raise SystemExit(main())
