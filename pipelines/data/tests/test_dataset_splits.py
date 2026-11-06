import dataclasses
import hashlib
import json
import shutil
import stat
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from agentwatch_data import splits as splits_module
from agentwatch_data.build import DATASET_FILE, MANIFEST_FILE, SCHEMA_FILE, Source, build
from agentwatch_data.evidence import add
from agentwatch_data.extract import canonical_sha256
from agentwatch_data.schema import SCHEMA_VERSION
from agentwatch_data.splits import (
    BLOCKED,
    FEATURES,
    INCONCLUSIVE,
    PASSED,
    SPLITS,
    SplitError,
    assign,
    check_compatibility,
    class_balance,
    describe,
    design_encoding,
    held_out_methods,
    is_member,
    leakage_check,
    load_contract,
    load_split,
    main,
    make_split,
    members_for,
    partition,
    read_split_file,
    separation,
)
from dataset_fixtures import make_result, write_run
from replay_fixtures import write_experiment

REPOSITORY = Path(__file__).resolve().parents[3]
# The committed contract is eval-contract-v2 (split-v2); split-v1 behavior is tested with the
# kept eval-contract-v1 file and the steps.pbtxt frozen schema it was validated against.
CONTRACT = REPOSITORY / "pipelines" / "model" / "evaluation-contract.toml"
V1 = REPOSITORY / "pipelines" / "model" / "evaluation-contract-v1.toml"
FROZEN = REPOSITORY / "pipelines" / "validation" / "schema" / "steps.pbtxt"
FROZEN_V1_3 = REPOSITORY / "pipelines" / "validation" / "schema" / "steps-v1.3.pbtxt"
GOLDEN = Path(__file__).parent / "fixtures"
NOW = datetime(2026, 10, 14, tzinfo=UTC)
CPU_WORKLOAD = "fixed SHA-256 CPU task"


def cpu_task(iterations: int) -> str:
    return f"cpu:{CPU_WORKLOAD}:{iterations}"


def iterations_in(split: str, count: int) -> list[int]:
    """Work sizes whose interim task_id hashes into the given split."""
    fractions = load_contract(V1).fractions
    found = [n for n in range(1, 500) if assign(cpu_task(n), fractions) == split]
    return found[:count]


@pytest.fixture
def evidence(tmp_path):
    """Two CPU runs per split, each a different task."""
    directory = tmp_path / "evidence"
    for split in SPLITS:
        for iterations in iterations_in(split, 2):
            result = make_result()
            result["experiment"]["iterations"] = iterations
            write_run(directory / f"cpu-{iterations}", result)
    return directory


def write_verdict(
    release: Path, /, attempt="20261014T000000000000Z-a", frozen: Path = FROZEN, **fields
):
    """A quality-gate attempt as the gate commits it: validations/<release>/<attempt>/.

    fields override the result; result_attempt makes it name a different attempt.
    """
    named = fields.pop("result_attempt", attempt)
    manifest = json.loads((release / MANIFEST_FILE).read_text())
    text = frozen.read_text().replace("\r\n", "\n")
    result = {
        "attempt": named,
        "release": release.name,
        "status": "passed",
        "validated_at": "2026-10-01T00:00:00+00:00",
        "release_parquet_sha256": manifest["parquet_sha256"],
        "release_contract_sha256": manifest["schema_sha256"],
        "schema_sha256": hashlib.sha256(text.encode()).hexdigest(),
    } | fields
    directory = release.parent / "validations" / release.name / attempt
    directory.mkdir(parents=True)
    (directory / "result.json").write_text(json.dumps(result))


@pytest.fixture
def release(evidence, tmp_path):
    path = build([Source("local", str(evidence))], tmp_path / "datasets", now=NOW).release
    write_verdict(path)
    return path


def split_of(release: Path, contract: Path = V1, frozen: Path = FROZEN):
    return make_split(release, contract, frozen)


def edited_contract(tmp_path, *edits: tuple[str, str], base: Path = V1) -> Path:
    text = base.read_text()
    for old, new in edits:
        assert old in text
        text = text.replace(old, new)
    path = tmp_path / f"contract-{len(list(tmp_path.glob('contract-*')))}.toml"
    path.write_text(text)
    return path


def resign(release: Path) -> None:
    manifest = json.loads((release / MANIFEST_FILE).read_text())
    manifest["parquet_sha256"] = hashlib.sha256((release / DATASET_FILE).read_bytes()).hexdigest()
    (release / MANIFEST_FILE).write_text(json.dumps(manifest))


def rows_of(path: Path, name: str) -> list[dict]:
    """The synthetic release's queue timing separates its classes, so its leakage check is
    blocked; tests that read its rows opt in explicitly."""
    return load_split(path, name, V1, FROZEN, allow_unchecked_leakage=True)


def resigned(directory: Path, split: dict) -> Path:
    """Write an edited split under the id its content now hashes to, as a forger could."""
    content = {key: value for key, value in split.items() if key != "split_id"}
    split = split | {"split_id": f"{split['split_method']}+{canonical_sha256(content)[:12]}"}
    path = directory / f"{split['split_id']}.json"
    path.write_text(json.dumps(split))
    return path


# Assignment


def test_every_task_lands_in_one_split_with_all_its_rows(release):
    result = split_of(release)
    for split in SPLITS:
        rows = rows_of(result.path, split)
        assert result.split["splits"][split]["groups"] == 2 and len(rows) == 10
        assert {result.split["groups"][row["task_id"]] for row in rows} == {split}
    assert sum(result.split["splits"][s]["rows"] for s in SPLITS) == 30


def test_assignment_is_stable_as_tasks_are_added(evidence, release, tmp_path):
    before = split_of(release).split["groups"]
    for iterations in iterations_in("train", 4)[2:]:
        result = make_result()
        result["experiment"]["iterations"] = iterations
        write_run(evidence / f"cpu-{iterations}", result)
    later = build([Source("local", str(evidence))], tmp_path / "datasets", now=NOW).release
    write_verdict(later)
    after = split_of(later).split["groups"]
    assert len(after) == len(before) + 2
    assert {task: after[task] for task in before} == before  # no task ever moves


def test_fractions_hold_across_many_tasks():
    fractions = load_contract(V1).fractions
    drawn = [assign(f"task-{n}", fractions) for n in range(20_000)]
    for split, fraction in fractions.items():
        assert drawn.count(split) / len(drawn) == pytest.approx(fraction, abs=0.01)


def test_a_run_with_two_tasks_is_refused():
    rows = [
        {"run_id": "r", "task_id": "a", "source": "measured"},
        {"run_id": "r", "task_id": "b", "source": "measured"},
    ]
    with pytest.raises(SplitError, match="more than one task_id"):
        partition(rows, load_contract(V1))


def test_sources_not_allowed_in_test_are_excluded_not_moved():
    contract = load_contract(V1)
    [test_task] = [cpu_task(n) for n in iterations_in("test", 1)]
    [train_task] = [cpu_task(n) for n in iterations_in("train", 1)]
    rows = [
        {"run_id": "measured", "task_id": test_task, "source": "measured"},
        {"run_id": "simulated", "task_id": test_task, "source": "simulated"},
        {"run_id": "trained", "task_id": train_task, "source": "simulated"},
    ]
    groups, members, excluded = partition(rows, contract)
    assert [row["run_id"] for row in members["test"]] == ["measured"]
    assert [row["run_id"] for row in members["train"]] == ["trained"]  # allowed outside test
    assert excluded == [
        {
            "run_id": "simulated",
            "source": "simulated",
            "reason": "source simulated is not allowed in the test split",
        }
    ]
    assert groups == {test_task: "test", train_task: "train"}


# Leakage


def row(label: str, **features) -> dict:
    return {name: 0 for name in FEATURES} | {"label_resource": label} | features


def test_separation_is_mann_whitney_auc_folded():
    assert separation([1, 2, 3, 4], [False, False, True, True]) == 1.0
    assert separation([1, 2, 3, 4], [True, True, False, False]) == 1.0  # inverse also leaks
    assert separation([1, 2, 3, 4], [False, True, False, True]) == 0.75
    assert separation([5, 5, 5, 5], [False, True, False, True]) == 0.5  # ties
    assert separation([None, None, 1, 2], [True, True, False, False]) == 1.0  # missingness leaks


def test_a_separating_feature_blocks_evaluation():
    train = [row("none", queue_fraction=0.01 * n) for n in range(5)]
    train += [row("cpu_sandbox", queue_fraction=0.9 + 0.01 * n) for n in range(5)]
    check = leakage_check(train, load_contract(V1))
    assert check["status"] == BLOCKED and "queue_fraction" in check["reason"]
    assert check["features"][0] == {"feature": "queue_fraction", "class": "none", "auc": 1.0}
    assert check["classes"] == ["none", "cpu_sandbox"]


def test_an_auc_just_above_the_bound_is_not_rounded_away():
    """5/6 = 0.8333333... rounds to the bound 0.833333 but is still above it."""
    contract = dataclasses.replace(load_contract(V1), max_single_feature_auc=0.833333)
    train = [row("none", queue_fraction=v) for v in (0, 1, 2, 3, 4, 10)]
    train.append(row("cpu_sandbox", queue_fraction=5))
    check = leakage_check(train, contract)
    assert check["features"][0] == {"feature": "queue_fraction", "class": "none", "auc": 0.833333}
    assert check["status"] == BLOCKED


def test_overlapping_features_pass():
    train = [row("none", queue_fraction=v) for v in (0.1, 0.4, 0.5, 0.8)]
    train += [row("cpu_sandbox", queue_fraction=v) for v in (0.2, 0.3, 0.6, 0.7)]
    check = leakage_check(train, load_contract(V1))
    assert check["status"] == PASSED and check["reason"] is None
    assert {item["feature"] for item in check["features"]} == set(FEATURES)
    assert max(item["auc"] for item in check["features"]) <= 0.98


@pytest.mark.parametrize(
    ("train", "reason"),
    [([], "the training split is empty"), ([row("none"), row("none")], "only one class")],
)
def test_leakage_is_inconclusive_without_two_classes(train, reason):
    check = leakage_check(train, load_contract(V1))
    assert check["status"] == INCONCLUSIVE and reason in check["reason"]


def test_the_real_split_records_its_leakage_check(release):
    leakage = split_of(release).split["leakage"]
    # Contention creates queue waiting directly, so queue timing separates the classes.
    assert leakage["status"] == BLOCKED
    assert leakage["features"][0]["auc"] == 1.0


# Held-out injection methods


def method_rows(**splits):
    return {
        name: [{"injection_method": method} for method in splits.get(name, [])] for name in SPLITS
    }


@pytest.mark.parametrize(
    ("members", "status", "only_in_test"),
    [
        (method_rows(train=["a", "none"], test=["b"]), "met", ["b"]),
        (method_rows(train=["a"], validation=["b"], test=["b"]), "not_met", []),
        (method_rows(train=["a"], test=["a"]), "not_met", []),
        (method_rows(test=["a"]), "not_met", ["a"]),  # nothing in training to contrast with
    ],
)
def test_held_out_injection_methods(members, status, only_in_test):
    found = held_out_methods(members, required=1)
    assert (found["status"], found["only_in_test"]) == (status, only_in_test)
    assert (found["reason"] is None) == (status == "met")


def test_no_held_out_method_required_is_met():
    assert held_out_methods(method_rows(), required=0)["status"] == "met"


# The split file


def test_split_file_is_deterministic_and_reused(release):
    first = split_of(release)
    assert not first.reused and first.path.parent == release.parent / "splits" / release.name
    assert first.path.name == f"{first.split['split_id']}.json"
    assert stat.S_IMODE(first.path.stat().st_mode) == 0o644
    second = split_of(release)
    assert second.reused and second.path == first.path and second.split == first.split
    assert [p.name for p in first.path.parent.iterdir()] == [first.path.name]  # no staging left


def test_split_records_its_provenance(release):
    split = split_of(release).split
    manifest = json.loads((release / MANIFEST_FILE).read_text())
    contract = load_contract(V1)
    assert split["dataset_version"] == release.name == split["release"]
    assert split["release_parquet_sha256"] == manifest["parquet_sha256"]
    assert split["label_version"] == manifest["label_version"]
    assert split["contract"]["version"] == contract.version == "eval-contract-v1"
    assert split["contract"]["fractions"] == {"train": 0.7, "validation": 0.15, "test": 0.15}
    assert split["frozen_schema_sha256"] == hashlib.sha256(FROZEN.read_bytes()).hexdigest()


@pytest.mark.parametrize("path", [CONTRACT, V1], ids=["v2", "v1"])
def test_contract_hash_follows_the_contract_definition(path):
    parsed = tomllib.loads(path.read_text())
    expected = hashlib.sha256(
        json.dumps(parsed, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:8]
    assert load_contract(path).sha256 == expected


def test_split_rows_hash_identifies_the_rows(release):
    result = split_of(release)
    test_rows = rows_of(result.path, "test")
    keys = sorted([row["run_id"], row["step_id"]] for row in test_rows)
    expected = canonical_sha256({"dataset_version": release.name, "rows": keys})
    assert result.split["splits"]["test"]["rows_sha256"] == expected


def test_a_changed_split_file_is_refused(release):
    result = split_of(release)
    split = json.loads(result.path.read_text())
    split["groups"] = {task: "train" for task in split["groups"]}
    result.path.write_text(json.dumps(split))
    # Not re-signed, so its id no longer hashes its content: refused before any rebuild.
    with pytest.raises(SplitError, match="does not match its content or location"):
        rows_of(result.path, "train")
    with pytest.raises(SplitError, match="Existing split .* does not match its content"):
        split_of(release)


REBUILT = "is not what .* produce; run make split-dataset"


def edit(split: dict, change) -> dict:
    change(split)
    return split


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s.update(groups={task: "train" for task in s["groups"]}),  # moves tasks
        lambda s: s["splits"]["test"].update(rows_sha256="0" * 64),
        lambda s: s["leakage"].update(status=PASSED, reason=None),  # unblocks leakage
        lambda s: s["excluded_runs"].append(
            {"run_id": "x", "source": "measured", "reason": "cherry-picked"}
        ),
        lambda s: s["contract"].update(group_key="run_id"),
        lambda s: s["contract"].update(test_sources=["simulated"]),
        lambda s: s.update(label_version="labels-v1.1+00000000"),
    ],
    ids=["groups", "rows", "leakage", "excluded", "group-key", "test-sources", "labels"],
)
def test_a_resigned_split_that_differs_from_its_rebuild_is_refused(release, change):
    """Its id was recomputed to match, so only rebuilding from the release can catch it."""
    result = split_of(release)
    split = edit(json.loads(result.path.read_text()), change)
    with pytest.raises(SplitError, match=REBUILT):
        load_split(resigned(result.path.parent, split), "train", V1, FROZEN, True)


def test_a_split_made_under_another_contract_is_refused(release, tmp_path):
    result = split_of(release)
    contract = edited_contract(
        tmp_path, ("max_single_feature_auc = 0.98", "max_single_feature_auc = 0.99")
    )
    with pytest.raises(SplitError, match=REBUILT):
        load_split(result.path, "train", contract, FROZEN, True)


def test_a_split_file_must_keep_its_name_and_folder(release):
    result = split_of(release)
    renamed = result.path.with_name("split-v1+000000000000.json")
    renamed.write_bytes(result.path.read_bytes())
    elsewhere = release.parent / "splits" / "other-release" / result.path.name
    elsewhere.parent.mkdir()
    elsewhere.write_bytes(result.path.read_bytes())
    for path in (renamed, elsewhere):
        with pytest.raises(SplitError, match="does not match its content or location"):
            rows_of(path, "test")


def test_a_split_of_a_different_release_is_refused(release):
    result = split_of(release)
    table = pq.read_table(release / DATASET_FILE)
    pq.write_table(table.slice(0, 20), release / DATASET_FILE)
    resign(release)
    with pytest.raises(SplitError, match="changed since it was validated"):
        rows_of(result.path, "train")


def test_splits_without_a_passing_leakage_check_are_refused_by_default(release):
    result = split_of(release)
    assert result.split["leakage"]["status"] == BLOCKED
    with pytest.raises(SplitError, match="leakage check is blocked; the contract blocks"):
        load_split(result.path, "train", V1, FROZEN)
    assert len(rows_of(result.path, "train")) == 10  # explicit opt-in for exploration


def test_splits_with_a_passing_leakage_check_load(release, monkeypatch):
    passed = {"status": PASSED, "reason": None}
    monkeypatch.setattr(splits_module, "leakage_check", lambda train, contract: passed)
    result = split_of(release)
    assert len(load_split(result.path, "validation", V1, FROZEN)) == 10


@pytest.fixture
def usable_split(release, monkeypatch):
    """A split that loads with default options: its leakage check passed."""
    passed = {"status": PASSED, "reason": None}
    monkeypatch.setattr(splits_module, "leakage_check", lambda train, contract: passed)
    path = split_of(release).path
    assert len(load_split(path, "train", V1, FROZEN)) == 10
    return path


def test_an_existing_split_is_refused_after_a_later_failed_verdict(release, usable_split):
    write_verdict(
        release,
        attempt="20261002T000000000000Z-b",
        status="failed",
        validated_at="2026-10-02T00:00:00+00:00",
    )
    for allow in (False, True):  # the leakage opt-in never bypasses the gate
        with pytest.raises(SplitError, match="failed its latest quality gate"):
            load_split(usable_split, "train", V1, FROZEN, allow_unchecked_leakage=allow)


def test_an_existing_split_is_refused_once_its_verdict_is_gone(release, usable_split):
    shutil.rmtree(release.parent / "validations")
    with pytest.raises(SplitError, match="has no quality-gate verdict"):
        load_split(usable_split, "test", V1, FROZEN)


def test_an_existing_split_is_refused_under_a_changed_frozen_schema(
    release, usable_split, tmp_path
):
    changed = tmp_path / "steps.pbtxt"
    changed.write_text(FROZEN.read_text() + "# reviewed change\n")
    with pytest.raises(SplitError, match="validated against a different schema"):
        load_split(usable_split, "train", V1, changed)
    # Revalidated under the new schema, the release passes, but this split predates it.
    write_verdict(
        release,
        attempt="20261002T000000000000Z-c",
        frozen=changed,
        validated_at="2026-10-02T00:00:00+00:00",
    )
    with pytest.raises(SplitError, match=REBUILT):
        load_split(usable_split, "train", V1, changed)


def test_unknown_split_names_are_refused(release):
    with pytest.raises(SplitError, match="Unknown split"):
        load_split(split_of(release).path, "holdout", V1, FROZEN)


def test_a_failed_write_leaves_nothing(release, monkeypatch):
    def fail(source, target):
        raise OSError("disk full")

    from agentwatch_data import files

    monkeypatch.setattr(files.os, "replace", fail)  # split files use the shared writer
    with pytest.raises(OSError, match="disk full"):
        split_of(release)
    assert list((release.parent / "splits" / release.name).iterdir()) == []


# The quality-gate verdict


def test_an_unvalidated_release_is_refused(release):
    shutil.rmtree(release.parent / "validations")
    with pytest.raises(SplitError, match="has no quality-gate verdict"):
        split_of(release)


def test_a_failed_latest_verdict_is_refused(release):
    write_verdict(
        release,
        attempt="20261002T000000000000Z-b",
        status="failed",
        validated_at="2026-10-02T00:00:00+00:00",
    )
    with pytest.raises(SplitError, match="failed its latest quality gate"):
        split_of(release)


def test_a_later_pass_supersedes_an_earlier_failure(release):
    shutil.rmtree(release.parent / "validations")
    write_verdict(release, attempt="a1", status="failed", validated_at="2026-09-30T00:00:00+00:00")
    write_verdict(release, attempt="a2")
    assert split_of(release).split["release"] == release.name


def test_staging_attempts_are_ignored(release):
    write_verdict(
        release,
        attempt=".attempt-staging",
        status="failed",
        validated_at="2026-10-02T00:00:00+00:00",
    )
    assert split_of(release).split["release"] == release.name


@pytest.mark.parametrize(
    "fields",
    [
        {"result_attempt": "other"},  # names another attempt
        {"release": "elsewhere"},
        {"validated_at": None},
        {"validated_at": "2026-10-02"},  # no time zone
        {"status": "maybe"},
        {"validated_at": "2999-01-01T00:00:00+00:00"},  # in the future
    ],
)
def test_damaged_or_impossible_attempts_make_the_verdict_unknown(release, fields):
    """The gate commits attempts by rename, so a bad one must not let an older pass count."""
    write_verdict(release, attempt="20261002T000000000000Z-b", **{"status": "failed"} | fields)
    with pytest.raises(SplitError, match="verdict of .* is unknown"):
        split_of(release)


@pytest.mark.parametrize("content", ["{not json", None])
def test_unreadable_attempts_make_the_verdict_unknown(release, content):
    damaged = release.parent / "validations" / release.name / "20261002T000000000000Z-b"
    damaged.mkdir()
    if content is not None:
        (damaged / "result.json").write_text(content)
    with pytest.raises(SplitError, match="verdict of .* is unknown"):
        split_of(release)


def test_a_release_changed_since_validation_is_refused(release):
    shutil.rmtree(release.parent / "validations")
    write_verdict(release, release_parquet_sha256="0" * 64)
    with pytest.raises(SplitError, match="changed since it was validated"):
        split_of(release)


def test_a_verdict_against_another_schema_is_refused(release, tmp_path):
    other = tmp_path / "other.pbtxt"
    other.write_text("feature { name: 'x' }\n")
    with pytest.raises(SplitError, match="different schema than other.pbtxt"):
        split_of(release, frozen=other)


def test_an_unreadable_frozen_schema_is_refused(release, tmp_path):
    with pytest.raises(SplitError, match="Frozen schema .* is unreadable"):
        split_of(release, frozen=tmp_path / "missing.pbtxt")


def test_crlf_checkouts_of_the_frozen_schema_match_the_verdict(release, tmp_path):
    crlf = tmp_path / "steps.pbtxt"
    crlf.write_bytes(FROZEN.read_bytes().replace(b"\n", b"\r\n"))
    assert split_of(release, frozen=crlf).split["release"] == release.name


# Releases the split stage refuses


def test_a_tampered_release_is_refused(release):
    (release / DATASET_FILE).write_bytes(b"tampered")
    with pytest.raises(SplitError, match="does not match its manifest"):
        split_of(release)


@pytest.mark.parametrize("missing", [SCHEMA_FILE, MANIFEST_FILE])
def test_a_release_missing_its_metadata_is_refused(release, missing):
    (release / missing).unlink()
    with pytest.raises(SplitError, match="does not match its manifest|is unreadable"):
        split_of(release)


def test_a_non_object_manifest_is_refused(release):
    (release / MANIFEST_FILE).write_text("[1]")
    with pytest.raises(SplitError, match="is not a JSON object"):
        split_of(release)


def test_a_directory_without_data_is_not_a_release(tmp_path):
    with pytest.raises(SplitError, match="is not a dataset release"):
        split_of(tmp_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", "steps-v2.0", "Schema steps-v2.0 is not accepted"),
        ("schema_version", "steps-v1.0", "Schema steps-v1.0 is not accepted"),
        ("schema_version", "rows-v1.2", "Schema rows-v1.2 is not accepted"),
        ("schema_version", None, "Schema None is not accepted"),
        ("label_version", "labels-v2+abc", "Labels labels-v2\\+abc are not accepted"),
    ],
)
def test_releases_the_contract_does_not_evaluate_are_refused(release, field, value, message):
    manifest = json.loads((release / MANIFEST_FILE).read_text())
    manifest[field] = value
    (release / MANIFEST_FILE).write_text(json.dumps(manifest))
    with pytest.raises(SplitError, match=message):
        split_of(release)


def test_exact_schema_versions_are_required_without_minor_compatibility(release, tmp_path):
    no_minor = (
        "accept_compatible_minor_versions = true",
        "accept_compatible_minor_versions = false",
    )
    with pytest.raises(SplitError, match=f"Schema {SCHEMA_VERSION} is not accepted"):
        split_of(release, contract=edited_contract(tmp_path, no_minor))
    exact = edited_contract(
        tmp_path,
        no_minor,
        ('schema_version = "steps-v1.1"', f'schema_version = "{SCHEMA_VERSION}"'),
    )
    assert split_of(release, contract=exact).split["release"] == release.name


# The evaluation contract


def test_the_v1_contract_loads_as_split_v1():
    contract = load_contract(V1)
    assert (contract.version, contract.status) == ("eval-contract-v1", "proposed")
    assert contract.method == "split-v1" and contract.assignment is None  # no method key: v1
    assert contract.label_rules == ("labels-v1.1",)  # a TOML string becomes a 1-tuple
    assert contract.group_key == "task_id" and contract.test_sources == ("measured",)
    assert contract.max_single_feature_auc == 0.98 and contract.held_out_injection_methods == 1
    assert (
        contract.min_test_steps_per_class,
        contract.min_clean_test_steps,
        contract.min_mixed_test_steps,
    ) == (30, 100, 30)


def test_v1_contract_file_keeps_the_v1_identifier():
    # 3022770b identifies pipelines/model/evaluation-contract.toml at commit c6b4333, the
    # eval-contract-v1 that split-v1+58e921fa846a and the #45/#46 results record.
    assert load_contract(V1).sha256 == "3022770b"


def test_v2_contract_selects_split_v2_with_both_label_rules():
    c = load_contract(CONTRACT)
    assert (c.version, c.status, c.schema_version) == ("eval-contract-v2", "accepted", "steps-v1.3")
    assert (c.method, c.group_key, c.label_rules) == (
        "split-v2",
        "task_group",
        ("labels-v1.1", "labels-v2"),
    )
    assert c.assignment.name == "split-assignment.toml" and c.assignment.is_file()
    assert c.assignment == REPOSITORY / "pipelines/data/src/agentwatch_data/split-assignment.toml"


def test_v2_changes_only_the_accepted_items():
    """Thresholds, fractions, sources and every other value stay as eval-contract-v1 set them."""
    v1, v2 = tomllib.loads(V1.read_text()), tomllib.loads(CONTRACT.read_text())
    changed = {
        ("version",): "eval-contract-v2",
        ("status",): "accepted",
        ("dataset", "schema_version"): "steps-v1.3",
        ("dataset", "label_rules"): ["labels-v1.1", "labels-v2"],
        ("splits", "method"): "split-v2",
        ("splits", "group_key"): "task_group",
        ("splits", "assignment"): "../data/src/agentwatch_data/split-assignment.toml",
    }
    for keys, value in changed.items():
        table_v1, table_v2 = v1, v2
        for key in keys[:-1]:
            table_v1, table_v2 = table_v1[key], table_v2[key]
        assert table_v2.pop(keys[-1]) == value
        table_v1.pop(keys[-1], None)
    assert v2 == v1


def test_the_assignment_resolves_against_the_contract_directory(tmp_path):
    copied = tmp_path / "contract.toml"
    copied.write_text(
        CONTRACT.read_text().replace(
            'assignment = "../data/src/agentwatch_data/split-assignment.toml"',
            'assignment = "assignment.toml"',
        )
    )
    assert load_contract(copied).assignment == tmp_path / "assignment.toml"


@pytest.mark.parametrize(
    ("label_version", "accepted"),
    [
        ("labels-v1.1+2c495a72", True),
        ("labels-v2+d803a9c6", True),
        ("labels-v1.1+2c495a72,labels-v2+d803a9c6", True),
        ("labels-v1.1+2c495a72,labels-v3+00000000", False),
        ("labels-v3+00000000", False),
        ("labels-v1.1+2c495a72,", False),  # an empty part names no accepted rule
        (None, False),
    ],
)
def test_release_with_an_unlisted_label_rule_is_refused(label_version, accepted):
    contract = load_contract(CONTRACT)
    manifest = {"schema_version": "steps-v1.3", "label_version": label_version}
    if accepted:
        check_compatibility(contract, manifest)
    else:
        with pytest.raises(SplitError, match="are not accepted by eval-contract-v2"):
            check_compatibility(contract, manifest)


def test_the_v1_contract_still_refuses_labels_v2():
    manifest = {"schema_version": "steps-v1.3", "label_version": "labels-v1.1+a,labels-v2+b"}
    with pytest.raises(SplitError, match="are not accepted by eval-contract-v1"):
        check_compatibility(load_contract(V1), manifest)


def test_split_v1_contract_with_an_assignment_is_refused(tmp_path):
    contract = edited_contract(
        tmp_path, ('group_key = "task_id"', 'group_key = "task_id"\nassignment = "a.toml"')
    )
    with pytest.raises(SplitError, match="split-v1 .*must not name an assignment"):
        load_contract(contract)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('method = "split-v2"', 'method = "split-v3"', "method 'split-v3' is not one of"),
        ('method = "split-v2"', "method = 2", "must be str"),
        ('assignment = "../data/src/agentwatch_data/split-assignment.toml"', "", "needs an"),
        (
            'assignment = "../data/src/agentwatch_data/split-assignment.toml"',
            "assignment = 1",
            "must be str",
        ),
        ('label_rules = ["labels-v1.1", "labels-v2"]', "label_rules = []", "label_rules"),
        ('label_rules = ["labels-v1.1", "labels-v2"]', "label_rules = [1]", "label_rules"),
        ('group_key = "task_group"', 'group_key = "run_id"', "is not one of"),
        ("min_clean_test_steps = 100", "min_clean_test_steps = -1", "min_clean_test_steps"),
        ("min_test_steps_per_class = 30", "min_test_steps_per_class = 3.0", "must be int"),
        ("min_mixed_contention_test_steps = 30", "", "min_mixed_contention_test_steps"),
    ],
)
def test_invalid_v2_contracts_are_refused(tmp_path, old, new, message):
    contract = edited_contract(tmp_path, (old, new), base=CONTRACT)
    with pytest.raises(SplitError, match=message):
        load_contract(contract)


SPLIT_V1_SUMMARY_KEYS = [
    "condition",
    "groups",
    "injection_method",
    "label_resource",
    "rows",
    "rows_sha256",
    "runs",
    "source",
]


def test_v1_3_release_splits_as_split_v1_with_the_v1_files(release):
    """A split made with eval-contract-v1 and steps.pbtxt is the split-v1 document as before.

    The fixture is a steps-v1.3 release of labels-v1.1 CPU rows, which eval-contract-v1 accepts
    as a compatible minor version. Split ids hash the whole document, so pinning the
    contract-derived parts and the exact set of fields proves neither the new contract fields
    nor split-v2's additions (class balance, design encoding, missingness) reach a split-v1 id.
    """
    made = split_of(release, contract=V1, frozen=FROZEN)
    split = made.split
    content = {key: value for key, value in split.items() if key != "split_id"}
    assert split["split_id"] == f"split-v1+{canonical_sha256(content)[:12]}"
    assert split["split_method"] == "split-v1"
    assert sorted(content) == [
        "contract",
        "dataset_version",
        "excluded_runs",
        "frozen_schema_sha256",
        "groups",
        "held_out_injection_methods",
        "label_version",
        "leakage",
        "release",
        "release_parquet_sha256",
        "split_method",
        "splits",
    ]
    for name in SPLITS:
        assert sorted(split["splits"][name]) == SPLIT_V1_SUMMARY_KEYS
    assert split["contract"] == {
        "version": "eval-contract-v1",
        "status": "proposed",
        "sha256": "3022770b",
        "group_key": "task_id",
        "fractions": {"train": 0.7, "validation": 0.15, "test": 0.15},
        "test_sources": ["measured"],
    }
    assert split_of(release, contract=V1, frozen=FROZEN).reused  # same id on a rebuild
    for name in SPLITS:
        rows = load_split(made.path, name, V1, FROZEN, allow_unchecked_leakage=True)
        assert split["splits"][name]["rows_sha256"] == canonical_sha256(
            {
                "dataset_version": release.name,
                "rows": sorted([row["run_id"], row["step_id"]] for row in rows),
            }
        )


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("train_fraction = 0.70", "train_fraction = 0.60", "sum to 1"),
        ("train_fraction = 0.70", "train_fraction = 1", "must be float"),
        ("test_fraction = 0.15", "test_fraction = 0.0", "sum to 1"),
        ('group_key = "task_id"', 'group_key = "queue_seconds"', "is not one of"),
        ('group_key = "task_id"', 'group_key = "run_id"', "is not one of"),  # splits a task
        ('group_key = "task_id"', "group_key = 7", "must be str"),
        ('classes = ["none",', 'classes = ["gpu", "none",', "differ from"),
        ('test_sources = ["measured"]', "test_sources = []", "must list sources"),
        ('test_sources = ["measured"]', "test_sources = [1]", "must list sources"),
        ('test_sources = ["measured"]', 'test_sources = ["measurd"]', "must list sources"),
        ('classification_target = "label_resource"', 'classification_target = "rule"', "target"),
        ("max_single_feature_auc = 0.98", "max_single_feature_auc = 0.4", "AUC bound"),
        ("held_out_injection_methods = 1", "held_out_injection_methods = -1", "held-out"),
        ("held_out_injection_methods = 1", "held_out_injection_methods = true", "must be int"),
        ("[leakage]", "[not_leakage]", r"missing its \[leakage\] table"),
        ('schema_version = "steps-v1.1"', 'schema_version = "steps"', None),
        ("[splits]", "[splits", "not valid TOML"),
    ],
)
def test_invalid_contracts_are_refused(release, tmp_path, old, new, message):
    contract = edited_contract(tmp_path, (old, new))
    with pytest.raises(SplitError, match=message or "schema version 'steps' is malformed"):
        split_of(release, contract=contract)


def test_an_unreadable_contract_is_refused(release, tmp_path):
    with pytest.raises(SplitError, match="Evaluation contract .* is unreadable"):
        split_of(release, contract=tmp_path / "missing.toml")


# The recorded run, end to end


def test_the_recorded_run_splits_honestly(tmp_path):
    """One recorded task lands wholly in one split, so the leakage check cannot run yet."""
    release = build([Source("local", str(GOLDEN))], tmp_path / "datasets", now=NOW).release
    write_verdict(release)
    split = split_of(release).split
    assert sum(split["splits"][name]["groups"] for name in SPLITS) == 1
    assert split["leakage"]["status"] == INCONCLUSIVE
    assert split["held_out_injection_methods"]["status"] == "not_met"


def test_the_recorded_run_keeps_its_split_v1_id(tmp_path):
    """A literal guard: the id the recorded run split to before eval-contract-v2 existed.

    Every split-v1 field is hashed into the id, so any change to the split-v1 document, the
    hash assignment, or the leakage check under eval-contract-v1 changes this value.
    """
    release = build([Source("local", str(GOLDEN))], tmp_path / "datasets", now=NOW).release
    write_verdict(release)
    assert split_of(release, contract=V1, frozen=FROZEN).split["split_id"] == (
        "split-v1+eee68b0e3d47"
    )


def test_a_split_v1_contract_needs_its_group_key_column():
    """A release without the contract's group key column (task_group before steps-v1.3)."""
    contract = dataclasses.replace(load_contract(V1), group_key="task_group")
    rows = [{"run_id": "r", "step_id": "s", "task_id": "t", "source": "measured"}]
    with pytest.raises(SplitError, match="no task_group column"):
        members_for(rows, contract)


# Command line


def test_cli_prints_the_split_and_its_leakage_status(release, capsys):
    args = ["--release", str(release), "--contract", str(V1), "--frozen-schema", str(FROZEN)]
    assert main(args) == 0
    captured = capsys.readouterr()
    assert (
        captured.out.startswith("split: ") and "  train: 10 rows, 2 runs, 2 groups" in captured.out
    )
    assert "leakage: blocked (features above 0.98" in captured.out
    assert "Evaluation must not use this split" in captured.err
    assert main(args) == 0 and "(reused)" in capsys.readouterr().out


def test_cli_reports_a_passing_leakage_check_quietly(release, capsys, monkeypatch):
    passed = {"status": PASSED, "reason": None}
    monkeypatch.setattr(splits_module, "leakage_check", lambda train, contract: passed)
    args = ["--release", str(release), "--contract", str(V1), "--frozen-schema", str(FROZEN)]
    assert main(args) == 0
    captured = capsys.readouterr()
    assert "leakage: passed\n" in captured.out and captured.err == ""


def test_cli_defaults_to_the_committed_contract_and_the_steps_v1_3_schema(tmp_path, monkeypatch):
    seen = []

    def refuse(release, contract, frozen_schema):
        seen.append((contract, frozen_schema))
        raise SplitError("stop")

    monkeypatch.setattr(splits_module, "make_split", refuse)
    assert main(["--release", str(tmp_path)]) == 1
    assert seen == [
        (
            Path("pipelines/model/evaluation-contract.toml"),
            Path("pipelines/validation/schema/steps-v1.3.pbtxt"),
        )
    ]
    assert all((REPOSITORY / path).is_file() for path in seen[0])


def test_cli_exits_one_when_the_split_fails(tmp_path, capsys):
    assert main(["--release", str(tmp_path)]) == 1
    assert "Split failed:" in capsys.readouterr().err


def test_module_runs_as_a_command(release, monkeypatch):
    import runpy
    import sys

    argv = ["agentwatch_data.splits", "--release", str(release), "--contract", str(V1)]
    monkeypatch.setattr(sys, "argv", argv + ["--frozen-schema", str(FROZEN)])
    monkeypatch.delitem(sys.modules, "agentwatch_data.splits", raising=False)
    with pytest.raises(SystemExit) as exited:
        runpy.run_module("agentwatch_data.splits", run_name="__main__")
    assert exited.value.code == 0


def test_hash_consistent_bytes_that_are_not_parquet_are_refused(release):
    from agentwatch_data.splits import load_release

    (release / DATASET_FILE).write_bytes(b"not parquet at all")
    resign(release)
    with pytest.raises(SplitError, match="steps.parquet is not readable Parquet"):
        load_release(release)


def test_timestamps_python_cannot_represent_are_refused(release):
    import pyarrow as pa
    from agentwatch_data.splits import load_release

    table = pq.read_table(release / DATASET_FILE)
    index = table.column_names.index("recorded_at")
    field = table.schema.field(index)
    far = pa.array([2**62] * table.num_rows, pa.int64()).cast(field.type)
    pq.write_table(table.set_column(index, field, far), release / DATASET_FILE)
    resign(release)
    with pytest.raises(SplitError, match="steps.parquet is not readable Parquet"):
        load_release(release)


# Verdict selection shared with the quality gate


VERDICT_CASES = json.loads(
    (Path(__file__).resolve().parents[2] / "validation/tests/verdict_cases.json").read_text()
)


@pytest.mark.parametrize("case", VERDICT_CASES["cases"], ids=lambda case: case["name"])
def test_verdict_selection_matches_the_shared_cases(tmp_path, case):
    """The gate's own reader (gate.latest_result) runs the same cases; the two must agree."""
    from agentwatch_data.splits import latest_verdict

    release, now = VERDICT_CASES["release"], datetime.fromisoformat(VERDICT_CASES["now"])
    base = tmp_path / "validations" / release
    base.mkdir(parents=True)
    for attempt in case["attempts"]:
        content = attempt["result"]
        if attempt.get("file"):
            (base / attempt["name"]).write_text(content)
            continue
        (base / attempt["name"]).mkdir()
        if "nesting" in attempt:
            content = "[" * attempt["nesting"] + "]" * attempt["nesting"]
        if content is not None:
            text = content if isinstance(content, str) else json.dumps(content)
            (base / attempt["name"] / "result.json").write_text(text)
    try:
        result = latest_verdict(tmp_path / release, now)
        found = None if result is None else {k: result[k] for k in ("attempt", "status")}
    except SplitError:
        found = "unknown"
    assert found == case["verdict"]


# split-v2 under eval-contract-v2. The release is the golden recorded run plus two synthetic
# (replay_fixtures) live experiments, one per agent task; every replay value is invented.

GOLDEN_RUN = GOLDEN / "coding-macos-2026-10-05"
GOLDEN_TASK = "coding:fixed Python project: run three tests:600000"
REPLAY_BASELINES = [
    {"agent-run": 2.0, "model-call-1": model, "tool-call-1": tool}
    for model, tool in ((1.0, 0.2), (1.01, 0.21), (0.99, 0.19))
]
# Solo contention slows each step's own resource; the mixed replay confirms a mixed model call.
REPLAY_CONTENDED = {
    "cpu": {"agent-run": 2.5, "model-call-1": 1.5, "tool-call-1": 0.5},
    "inference": {"agent-run": 2.5, "model-call-1": 1.4, "tool-call-1": 0.2},
    "mixed": {"agent-run": 3.0, "model-call-1": 1.9, "tool-call-1": 0.6},
}
EXPERIMENTS = {"fix-window-count": "window-exp", "fix-mean-empty": "mean-exp"}


def write_store(tmp_path: Path) -> Path:
    store = tmp_path / "store"
    add(GOLDEN_RUN, store)
    for namespace, (group, name) in enumerate(EXPERIMENTS.items(), start=1):
        experiment = write_experiment(
            tmp_path / "in",
            name=name,
            task_id=f"{group}-r1",
            live=True,
            run_namespace=namespace,
            plan_sha256=f"{namespace}" * 64,
            baseline=REPLAY_BASELINES,
            contended=REPLAY_CONTENDED,
        )
        add(experiment, store)
    return store


def validated_release(store: Path, datasets: Path) -> Path:
    release = build([Source("local", str(store))], datasets, now=NOW).release
    write_verdict(release, frozen=FROZEN_V1_3)
    return release


@pytest.fixture
def v1_3_release(tmp_path):
    """A steps-v1.3 release with run and replay rows that passed the quality gate."""
    return validated_release(write_store(tmp_path), tmp_path / "datasets")


def split_v2(release: Path, contract: Path = CONTRACT):
    return make_split(release, contract, FROZEN_V1_3)


def test_split_v2_puts_the_declared_task_and_held_out_method_in_test(v1_3_release):
    split = split_v2(v1_3_release).split
    assert split["split_id"].startswith("split-v2+")
    assert split["split_method"] == "split-v2"
    # fix-window-count is declared; the golden task holds worker_queue_blocker, held out.
    assert split["groups"] == {
        GOLDEN_TASK: "test",
        "fix-mean-empty": "train",
        "fix-window-count": "test",
    }
    checks = split["split_v2_checks"]
    assert checks["independence"]["status"] == "passed"
    assert checks["held_out"]["status"] == "met"
    assert checks["held_out"]["only_in_test"] == ["worker_queue_blocker"]
    assert checks["mixed_in_test"] == {"status": "met", "rows": 2}  # by replay_condition
    assert split["class_balance"]["status"] in ("met", "not_met")
    assert split["contract"]["group_key"] == "task_group"


def test_split_v2_records_its_assignment_and_class_balance(v1_3_release):
    split = split_v2(v1_3_release).split
    assignment = load_contract(CONTRACT).assignment
    canonical = canonical_sha256(tomllib.loads(assignment.read_text()))[:8]
    assert split["assignment"] == {"version": "split-v2", "sha256": canonical}
    balance = split["class_balance"]
    test = load_split(
        split_v2(v1_3_release).path, "test", CONTRACT, FROZEN_V1_3, allow_unchecked_leakage=True
    )
    assert balance["per_class"] == {
        name: sum(row["label_resource"] == name for row in test)
        for name in ("none", "cpu_sandbox", "inference", "retrieval")
    }
    assert balance["mixed"] == sum(row["label_resource"] == "mixed" for row in test) == 1
    assert balance["clean"] == sum(row["condition"] == "clean" for row in test)
    assert balance["mixed_condition_rows"] == 2
    assert balance["minimums"] == {"per_class": 30, "clean": 100, "mixed": 30}
    # A handful of synthetic steps meets no minimum.
    assert balance["below"] == ["none", "cpu_sandbox", "inference", "retrieval", "clean", "mixed"]
    assert balance["status"] == "not_met"


def test_split_v2_summaries_report_missing_features_per_feature_version(v1_3_release):
    split = split_v2(v1_3_release).split
    test = load_split(
        split_v2(v1_3_release).path, "test", CONTRACT, FROZEN_V1_3, allow_unchecked_leakage=True
    )
    missing = split["splits"]["test"]["missing"]
    assert set(missing) == {"steps-v1.2", "step-features-v1"}
    for version, counts in missing.items():
        rows = [row for row in test if row["feature_version"] == version]
        assert counts == {name: sum(row[name] is None for row in rows) for name in FEATURES}
    assert split["splits"]["validation"]["missing"] == {}  # no rows, no feature version


def test_split_v2_reports_design_encoding_on_training_replays(v1_3_release):
    report = split_v2(v1_3_release).split["design_encoding"]
    assert report["status"] == "report_only"
    assert report["rows"] > 0
    assert {item["feature"] for item in report["features"]} == set(FEATURES)
    aucs = [(-item["auc"], item["feature"]) for item in report["features"]]
    assert aucs == sorted(aucs)


def test_load_split_v2_rebuilds_and_refuses_an_edited_file(v1_3_release):
    result = split_v2(v1_3_release)
    train = load_split(result.path, "train", CONTRACT, FROZEN_V1_3, allow_unchecked_leakage=True)
    assert train and {row["task_group"] for row in train} == {"fix-mean-empty"}
    assert result.split["splits"]["train"]["rows_sha256"] == canonical_sha256(
        {
            "dataset_version": v1_3_release.name,
            "rows": sorted([row["run_id"], row["step_id"]] for row in train),
        }
    )
    for change in (
        lambda s: s["groups"].update({"fix-window-count": "train"}),
        lambda s: s["class_balance"].update(status="met", below=[]),
        lambda s: s["split_v2_checks"]["mixed_in_test"].update(rows=99),
        lambda s: s["assignment"].update(sha256="00000000"),
        lambda s: s["design_encoding"].update(status="unavailable"),
    ):
        split = edit(json.loads(result.path.read_text()), change)
        forged = resigned(result.path.parent, split)
        with pytest.raises(SplitError, match=REBUILT):
            load_split(forged, "train", CONTRACT, FROZEN_V1_3, True)
        forged.unlink()
    split = json.loads(result.path.read_text())
    split["groups"]["fix-window-count"] = "train"
    result.path.write_text(json.dumps(split))  # not re-signed
    with pytest.raises(SplitError, match="does not match its content or location"):
        load_split(result.path, "train", CONTRACT, FROZEN_V1_3, True)


def copied_contract(tmp_path: Path) -> Path:
    """eval-contract-v2 and its assignment file, copied so the assignment can be edited."""
    directory = tmp_path / "contract"
    directory.mkdir()
    shutil.copyfile(load_contract(CONTRACT).assignment, directory / "assignment.toml")
    contract = directory / "evaluation-contract.toml"
    contract.write_text(
        CONTRACT.read_text().replace(
            'assignment = "../data/src/agentwatch_data/split-assignment.toml"',
            'assignment = "assignment.toml"',
        )
    )
    return contract


def test_changed_assignment_file_is_refused_by_load_split(v1_3_release, tmp_path):
    contract = copied_contract(tmp_path)
    result = split_v2(v1_3_release, contract)
    assert load_split(result.path, "test", contract, FROZEN_V1_3, True)
    assignment = contract.parent / "assignment.toml"
    # Moves no task (no such group exists) but is a different frozen assignment.
    assignment.write_text(
        assignment.read_text().replace(
            'test_task_groups = ["fix-window-count"]',
            'test_task_groups = ["fix-window-count", "fix-other"]',
        )
    )
    with pytest.raises(SplitError, match=REBUILT):
        load_split(result.path, "test", contract, FROZEN_V1_3, True)


def test_an_unreadable_assignment_is_a_split_error(v1_3_release, tmp_path):
    contract = copied_contract(tmp_path)
    (contract.parent / "assignment.toml").write_text('version = "split-v3"\n')
    with pytest.raises(SplitError, match="Split assignment keys"):
        split_v2(v1_3_release, contract)
    (contract.parent / "assignment.toml").unlink()
    with pytest.raises(SplitError, match="cannot be read"):
        split_v2(v1_3_release, contract)


def replay_row(group: str, run: str, **fields) -> dict:
    return {
        "run_id": run,
        "step_id": "s",
        "task_id": f"{group}-r1",
        "task_group": group,
        "source": "measured",
        "injection_method": "none",
        "label_resource": "none",
        "condition": "baseline",
        "replay_condition": "clean",
        "plan_sha256": None,
    } | fields


def test_a_run_shared_between_splits_is_a_split_error():
    rows = [replay_row("fix-window-count", "shared"), replay_row("fix-mean-empty", "shared")]
    with pytest.raises(SplitError, match="not independent.*run_id"):
        members_for(rows, load_contract(CONTRACT))


def test_split_v2_excludes_untested_sources_from_test_and_rechecks():
    rows = [
        replay_row("fix-window-count", "measured", replay_condition="mixed"),
        replay_row("fix-window-count", "simulated", source="simulated", replay_condition="mixed"),
        replay_row("fix-mean-empty", "trained", source="simulated"),
    ]
    groups, members, excluded, checks = members_for(rows, load_contract(CONTRACT))
    assert groups == {"fix-mean-empty": "train", "fix-window-count": "test"}
    assert [row["run_id"] for row in members["test"]] == ["measured"]
    assert [row["run_id"] for row in members["train"]] == ["trained"]
    assert excluded == [
        {
            "run_id": "simulated",
            "source": "simulated",
            "reason": "source simulated is not allowed in the test split",
        }
    ]
    assert checks["mixed_in_test"] == {"status": "met", "rows": 1}  # the excluded run not counted


def test_split_v1_members_carry_no_split_v2_checks(release):
    rows = splits_module.load_release(release)[1]
    groups, members, excluded, checks = members_for(rows, load_contract(V1))
    assert checks is None
    assert (groups, members, excluded) == partition(rows, load_contract(V1))


def test_split_twice_reuses_the_same_file_and_rows_sha256(v1_3_release, tmp_path):
    first = split_v2(v1_3_release)
    again = split_v2(v1_3_release)
    assert again.reused and again.path == first.path and again.split == first.split
    # Building and splitting again into an empty store reproduces the release and split.
    rebuilt = validated_release(tmp_path / "store", tmp_path / "rebuilt")
    other = split_v2(rebuilt)
    assert rebuilt.name == v1_3_release.name
    assert other.split["split_id"] == first.split["split_id"]
    assert other.path.read_bytes() == first.path.read_bytes()
    for name in SPLITS:
        assert (
            other.split["splits"][name]["rows_sha256"] == first.split["splits"][name]["rows_sha256"]
        )


def test_a_split_file_with_an_unknown_method_is_refused(v1_3_release):
    result = split_v2(v1_3_release)
    split = json.loads(result.path.read_text()) | {"split_method": "split-v9"}
    content = {key: value for key, value in split.items() if key != "split_id"}
    split["split_id"] = f"split-v9+{canonical_sha256(content)[:12]}"
    forged = result.path.parent / f"{split['split_id']}.json"
    forged.write_text(json.dumps(split))
    with pytest.raises(SplitError, match="unknown split method 'split-v9'"):
        read_split_file(forged)


def test_split_v1_documents_have_no_split_v2_sections(release):
    split = describe(
        release,
        json.loads((release / MANIFEST_FILE).read_text()),
        splits_module.load_release(release)[1],
        load_contract(V1),
        "0" * 64,
    )
    assert not {"assignment", "split_v2_checks", "class_balance", "design_encoding"} & set(split)
    assert all("missing" not in split["splits"][name] for name in SPLITS)


# Membership, class balance, and design encoding


def test_membership_reads_label_resources_when_present():
    mixed = {"label_resource": "mixed", "label_resources": "cpu_sandbox,inference"}
    assert is_member(mixed, "cpu_sandbox") and is_member(mixed, "inference")
    assert not is_member(mixed, "none") and not is_member(mixed, "retrieval")
    none = {"label_resource": "none", "label_resources": ""}
    assert is_member(none, "none") and not is_member(none, "cpu_sandbox")
    # Rows before steps-v1.3 (or a null label_resources) fall back to label_resource.
    assert is_member({"label_resource": "inference"}, "inference")
    assert is_member({"label_resource": "none", "label_resources": None}, "none")


def test_leakage_counts_a_mixed_step_for_each_of_its_resources():
    train = [row("none", label_resources="", queue_fraction=0.01 * n) for n in range(5)]
    train += [
        row("mixed", label_resources="cpu_sandbox,inference", queue_fraction=0.9 + 0.01 * n)
        for n in range(5)
    ]
    check = leakage_check(train, load_contract(CONTRACT))
    assert check["classes"] == ["none", "cpu_sandbox", "inference"]
    assert check["status"] == BLOCKED and "queue_fraction" in check["reason"]
    assert check["features"][0]["auc"] == 1.0


def balance_row(label: str, condition: str = "contended", **fields) -> dict:
    return {"label_resource": label, "condition": condition, "replay_condition": None} | fields


def test_class_balance_counts_test_rows_against_the_contract_minimums():
    test = [balance_row(name) for name in ("none", "cpu_sandbox", "inference", "retrieval")]
    test += [balance_row("none", "clean"), balance_row("mixed", replay_condition="mixed")]
    test.append(balance_row("cpu_sandbox", replay_condition="mixed"))
    contract = dataclasses.replace(
        load_contract(CONTRACT),
        min_test_steps_per_class=1,
        min_clean_test_steps=1,
        min_mixed_test_steps=1,
    )
    balance = class_balance(test, contract)
    assert balance == {
        "per_class": {"none": 2, "cpu_sandbox": 2, "inference": 1, "retrieval": 1},
        "clean": 1,
        "mixed": 1,
        "mixed_condition_rows": 2,
        "minimums": {"per_class": 1, "clean": 1, "mixed": 1},
        "below": [],
        "status": "met",
    }
    contract = dataclasses.replace(contract, min_test_steps_per_class=2, min_mixed_test_steps=2)
    balance = class_balance(test, contract)
    assert balance["below"] == ["inference", "retrieval", "mixed"]
    assert balance["status"] == "not_met"


def encoding_row(role: str, label: str = "none", kind: str = "agent_replay", **features) -> dict:
    return row(label, kind=kind, replay_role=role) | features


def test_design_encoding_contrasts_contended_and_clean_none_replays():
    train = [encoding_row("contended", queue_fraction=0.9 + 0.01 * n) for n in range(3)]
    train += [encoding_row("clean_control", queue_fraction=0.01 * n) for n in range(3)]
    train += [
        encoding_row("contended", label="cpu_sandbox"),  # not labeled none
        encoding_row("baseline"),  # not a contended/control pair
        encoding_row(None, kind="cpu"),  # run rows are not replays
    ]
    report = design_encoding(train)
    assert report["status"] == "report_only" and report["rows"] == 6
    assert report["features"][0] == {"feature": "queue_fraction", "auc": 1.0}
    assert {item["feature"] for item in report["features"]} == set(FEATURES)


@pytest.mark.parametrize("role", ["contended", "clean_control"])
def test_design_encoding_is_unavailable_without_both_roles(role):
    report = design_encoding([encoding_row(role)])
    assert report["status"] == "unavailable" and report["reason"]
    assert design_encoding([])["status"] == "unavailable"


def test_import_order_has_no_cycle():
    for modules in (
        "agentwatch_data.split_assignment, agentwatch_data.splits",
        "agentwatch_data.splits, agentwatch_data.split_assignment",
    ):
        subprocess.run([sys.executable, "-I", "-c", f"import {modules}"], check=True, timeout=60)
