"""Freeze the train, validation, and test splits of a dataset release and check them for leakage.

    python -m agentwatch_data.splits --release data/datasets/<version>

Only a release whose latest quality-gate verdict is a pass, for exactly the files it holds now
and the reviewed frozen schema, can be split. The fractions, group key, test sources, and
leakage bound come from the evaluation contract (pipelines/model/evaluation-contract.toml), so
the contract stays the single source of those values.

Every group (task) is assigned by hashing its key with SPLIT_METHOD, so under the same
fractions a task keeps its split in every later release: a task once tested is never trained
on, and new tasks never move old ones. The fractions are part of the split's identity; changing
them moves tasks near the boundaries, so it needs a new contract version and test split. Runs whose source the contract does not allow in the test split are excluded there, not
moved, so a task's replays never cross splits.

The split is written to <datasets>/splits/<release>/<split id>.json. Its content is
deterministic and the split id is derived from it, so rerunning reuses the same file. It
records the task assignment, a hash of each split's rows (the "test-split hash" evaluation
reports record), counts, held-out injection methods, and the contract's leakage check on the
training split. A blocked or inconclusive leakage check does not fail the split, which is kept
as evidence, but load_split refuses to hand its rows to training or evaluation. load_split
also rechecks the release's latest quality-gate verdict and rebuilds the split on every read,
so neither a split made before a later failed validation nor an edited split file can be used
to bypass the gate or the leakage check.
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

SPLIT_METHOD = "split-v1"  # hashed with each group key; a new value redraws every split
SPLITS = ("train", "validation", "test")
SPLITS_DIR, VALIDATIONS_DIR, RESULT_FILE = "splits", "validations", "result.json"
FEATURES = tuple(feature_columns())  # the schema decides what may be a model input
# Grouping must keep a task's baselines and replays together, so only task-level keys qualify.
GROUP_KEYS = ("task_id",)
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
    label_rules: str
    classes: tuple[str, ...]
    target: str
    test_sources: tuple[str, ...]
    group_key: str
    fractions: dict[str, float]
    held_out_injection_methods: int
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
        label_rules=require(dataset, "label_rules", str, "dataset"),
        classes=classes,
        target=target,
        test_sources=test_sources,
        group_key=group_key,
        fractions=fractions,
        held_out_injection_methods=held_out,
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
    # Label versions are pinned exactly; only the fingerprint after "+" may vary.
    if str(label).split("+", 1)[0] != contract.label_rules:
        raise SplitError(f"Labels {label} are not accepted by {contract.version}")


def read_object(path: Path, what: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SplitError(f"{what} {path.name} is unreadable: {error}") from error
    if not isinstance(value, dict):
        raise SplitError(f"{what} {path.name} is not a JSON object")
    return value


def verify_release(path: Path) -> dict:
    """A release's manifest, after verifying its files and name against it."""
    if not (path / DATASET_FILE).is_file():
        raise SplitError(f"{path} is not a dataset release")
    manifest = read_object(path / MANIFEST_FILE, "Manifest")
    intact = (
        manifest.get("release_format") == RELEASE_FORMAT
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
    except (pa.ArrowException, OSError, ValueError) as error:  # hash-consistent but not Parquet
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
        except (OSError, ValueError):
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


def assign(group: str, fractions: dict[str, float]) -> str:
    """The split of one group, from a hash of its key: stable across releases."""
    digest = hashlib.sha256(f"{SPLIT_METHOD}:{group}".encode()).digest()
    position = int.from_bytes(digest[:8], "big") / 2**64
    cumulative = 0.0
    for name in SPLITS[:-1]:
        cumulative += fractions[name]
        if position < cumulative:
            return name
    return SPLITS[-1]


def partition(rows: list[dict], contract: Contract) -> tuple[dict, dict, list[dict]]:
    """Assign rows to splits by group.

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
    excluded = {}
    for row in rows:
        split = groups[row[contract.group_key]]
        if split == "test" and row["source"] not in contract.test_sources:
            excluded[row["run_id"]] = {
                "run_id": row["run_id"],
                "source": row["source"],
                "reason": f"source {row['source']} is not allowed in the test split",
            }
            continue
        members[split].append(row)
    return dict(sorted(groups.items())), members, [excluded[run] for run in sorted(excluded)]


def rows_sha256(dataset_version: str, rows: list[dict]) -> str:
    """Identify a split's rows: the release plus each row's (run_id, step_id)."""
    keys = sorted([row["run_id"], row["step_id"]] for row in rows)
    return canonical_sha256({"dataset_version": dataset_version, "rows": keys})


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


def leakage_check(train: list[dict], contract: Contract) -> dict:
    """The contract's check: no single feature may nearly separate a class on training data."""
    bound = contract.max_single_feature_auc
    labels = [row[contract.target] for row in train]
    classes = [name for name in contract.classes if 0 < labels.count(name) < len(labels)]
    check = {"max_single_feature_auc": bound, "classes": classes, "features": []}
    if not classes:
        reason = "the training split is empty" if not train else "it has only one class"
        return check | {"status": INCONCLUSIVE, "reason": reason}
    leaking = []
    for feature in FEATURES:
        values = [row[feature] for row in train]
        scores = {name: separation(values, [label == name for label in labels]) for name in classes}
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


def describe(
    release: Path, manifest: dict, rows: list[dict], contract: Contract, schema_sha256: str
) -> dict:
    groups, members, excluded = partition(rows, contract)
    version = manifest["dataset_version"]
    document = {
        "split_method": SPLIT_METHOD,
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
            for name in SPLITS
        },
        "held_out_injection_methods": held_out_methods(
            members, contract.held_out_injection_methods
        ),
        "leakage": leakage_check(members["train"], contract),
    }
    split_id = f"{SPLIT_METHOD}+{canonical_sha256(document)[:12]}"
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
    content = {key: value for key, value in split.items() if key != "split_id"}
    try:
        expected = f"{SPLIT_METHOD}+{canonical_sha256(content)[:12]}"
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
    release = path.parents[2] / path.parent.name
    manifest, rows = load_release(release)
    check_compatibility(contract, manifest)
    # The verdict can change after the split was made, so it is checked on every read.
    require_passing_verdict(release, manifest, frozen_schema)
    expected = describe(release, manifest, rows, contract, text_sha256(frozen_schema))
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
    return partition(rows, contract)[1][name]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument(
        "--contract", type=Path, default=Path("pipelines/model/evaluation-contract.toml")
    )
    parser.add_argument(
        "--frozen-schema", type=Path, default=Path("pipelines/validation/schema/steps.pbtxt")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
