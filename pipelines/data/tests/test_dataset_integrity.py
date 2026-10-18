import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest
from agentwatch_data.build import Source, build
from agentwatch_data.evidence import add_run
from agentwatch_data.extract import canonical_sha256
from agentwatch_data.integrity import main, verify
from dataset_fixtures import make_result, write_run

GOLDEN = next((Path(__file__).parent / "fixtures").iterdir())
NOW = datetime(2026, 10, 18, tzinfo=UTC)


@pytest.fixture
def stores(tmp_path):
    """An evidence store holding the recorded run and a dataset store built from it."""
    evidence, datasets = tmp_path / "evidence", tmp_path / "datasets"
    recorded = tmp_path / "recorded" / "coding-run"
    shutil.copytree(GOLDEN, recorded)
    add_run(recorded, evidence)
    release = build([Source("local", str(evidence))], datasets, now=NOW).release
    return evidence, datasets, release


def problems(evidence, datasets, status=None):
    return verify(evidence, datasets, status).problems


def write_split(datasets: Path, release: str, content: dict) -> Path:
    """A split file whose id hashes its content, as split-dataset writes it."""
    document = {"release": release} | content
    split_id = f"split-v1+{canonical_sha256(document)[:12]}"
    path = datasets / "splits" / release / f"{split_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"split_id": split_id} | document))
    return path


def write_attempt(datasets: Path, release: str, name: str, result=None, statistics="datasets {}"):
    """A committed gate attempt: result.json bound to the statistics written beside it."""
    directory = datasets / "validations" / release / name
    directory.mkdir(parents=True)
    if result is None:
        result = {
            "attempt": name,
            "release": release,
            "status": "passed",
            "validated_at": "2026-10-18T00:00:00+00:00",
            "statistics_sha256": hashlib.sha256((statistics or "").encode()).hexdigest(),
        }
    text = result if isinstance(result, str) else json.dumps(result)
    (directory / "result.json").write_text(text)
    if statistics is not None:
        (directory / "statistics.pbtxt").write_text(statistics)
    return directory


def test_intact_stores_are_counted(stores):
    evidence, datasets, release = stores
    write_attempt(datasets, release.name, "a1")
    write_split(datasets, release.name, {"groups": {}})
    (datasets / ".building-x").mkdir()  # a build in progress is not part of a version
    (evidence / ".adding-y").mkdir()
    entry = next(p for p in evidence.iterdir() if not p.name.startswith("."))
    (entry / ".DS_Store").write_text("")  # hidden tool state inside an entry is skipped too
    report = verify(evidence, datasets)
    assert report.problems == []
    counts = (report.runs, report.releases, report.reports, report.attempts, report.splits)
    assert counts == (1, 1, 1, 1, 1)
    assert report.hidden == 3


def test_missing_stores_are_problems(tmp_path):
    assert problems(tmp_path / "no-evidence", tmp_path / "no-datasets") == [
        f"evidence store {tmp_path / 'no-evidence'} does not exist",
        f"datasets store {tmp_path / 'no-datasets'} does not exist",
    ]


# Evidence


def test_damaged_evidence_is_reported(stores):
    evidence, datasets, _ = stores
    [entry] = list(evidence.iterdir())
    shutil.copytree(entry, evidence / "coding-00000000-0000-0000-0000-000000000000")  # misnamed
    (evidence / "notes.txt").write_text("not a run")
    (evidence / "cpu-linked").symlink_to(entry, target_is_directory=True)
    broken = evidence / "cpu-broken"
    broken.mkdir()
    (broken / "results.json").write_text("{damaged")
    assert problems(evidence, datasets) == [
        "evidence/coding-00000000-0000-0000-0000-000000000000: stored under the wrong name",
        (
            "evidence/cpu-broken: Stored evidence cpu-broken is unreadable: JSONDecodeError: "
            "Expecting property name enclosed in double quotes: line 1 column 2 (char 1)"
        ),
        "evidence/cpu-linked: not a run directory",
        "evidence/notes.txt: not a run directory",
    ]


def test_an_entry_holding_more_than_its_evidence_is_reported(stores):
    """The build reads any run directory nested in an entry, so it must not be versioned."""
    evidence, datasets, _ = stores
    [entry] = list(evidence.iterdir())
    write_run(entry / "cpu-nested", make_result())
    (entry / "notes.txt").write_text("")
    run_id = entry.name.split("-", 1)[1]
    assert problems(evidence, datasets) == [
        f"evidence/{entry.name}: holds more than run evidence: ['cpu-nested', 'notes.txt']",
        # and so the release built from it can no longer be rebuilt from the store
        (
            f"datasets/{next(datasets.glob('steps-*')).name}: built from evidence that is not "
            f"in the evidence store (runs ['{run_id}']), so it cannot be rebuilt"
        ),
    ]


def test_one_run_stored_under_two_kinds_is_reported(stores, tmp_path):
    evidence, datasets, _ = stores
    result = make_result(kind="coding", agent_model="qwen3:1.7b")
    write_run(tmp_path / "recorded" / "coding-agent", result)
    add_run(tmp_path / "recorded" / "coding-agent", evidence)
    shutil.copytree(evidence / f"coding-{result['run_id']}", evidence / f"llm-{result['run_id']}")
    assert problems(evidence, datasets) == [
        f"evidence/llm-{result['run_id']}: run {result['run_id']} is also coding-{result['run_id']}"
    ]


# Releases


def test_damaged_releases_and_reports_are_reported(stores):
    evidence, datasets, release = stores
    (release / "steps.parquet").write_bytes(b"tampered")
    (datasets / "stray.txt").write_text("")
    (datasets / "steps-v1.2+aaaaaaaaaaaa").symlink_to(release, target_is_directory=True)
    (datasets / "builds" / "broken.json").write_text("{damaged")
    (datasets / "builds" / "notes.txt").write_text("{}")
    (datasets / "builds" / "listed.json").write_text(json.dumps({"release": ["x"]}))
    assert problems(evidence, datasets) == [
        f"datasets/{release.name}: Release {release.name} does not match its manifest; rebuild it",
        "datasets/steps-v1.2+aaaaaaaaaaaa: unexpected entry",
        "datasets/stray.txt: unexpected entry",
        "datasets/builds/broken.json: unreadable build report",
        "datasets/builds/listed.json: names release ['x'], which is missing",
        "datasets/builds/notes.txt: unreadable build report",
    ]


def test_a_deleted_release_leaves_its_report_naming_it(stores):
    evidence, datasets, release = stores
    shutil.rmtree(release)
    [report] = list((datasets / "builds").iterdir())
    assert problems(evidence, datasets) == [
        f"datasets/builds/{report.name}: names release {release.name!r}, which is missing"
    ]


def test_a_renamed_release_is_reported_once(stores):
    evidence, datasets, release = stores
    release.rename(datasets / "steps-v1.2+bbbbbbbbbbbb")
    [report] = list((datasets / "builds").iterdir())
    assert problems(evidence, datasets) == [
        (
            "datasets/steps-v1.2+bbbbbbbbbbbb: Release steps-v1.2+bbbbbbbbbbbb does not match "
            "its manifest; rebuild it"
        ),
        f"datasets/builds/{report.name}: names release {release.name!r}, which is missing",
    ]


def format_1(release: Path) -> None:
    """Turn a release into the superseded format: no schema.json, no format field."""
    manifest = json.loads((release / "manifest.json").read_text())
    del manifest["release_format"], manifest["schema_sha256"]
    (release / "manifest.json").write_text(json.dumps(manifest))
    (release / "schema.json").unlink()


def test_superseded_releases_are_accepted_by_their_own_format(stores):
    evidence, datasets, release = stores
    format_1(release)
    report = verify(evidence, datasets)
    assert report.problems == [] and (report.releases, report.superseded) == (0, 1)
    (release / "steps.parquet").write_bytes(b"tampered")
    assert problems(evidence, datasets) == [
        f"datasets/{release.name}: Release {release.name} does not match its manifest (no format)"
    ]


@pytest.mark.parametrize(
    ("content", "message"), [("{damaged", "is unreadable"), ("[1]", "is not a JSON object")]
)
def test_unreadable_manifests_are_reported(stores, content, message):
    evidence, datasets, release = stores
    (release / "manifest.json").write_text(content)
    [problem] = problems(evidence, datasets)
    assert problem.startswith(f"datasets/{release.name}: Manifest") and message in problem


def test_unknown_release_formats_are_reported(stores):
    evidence, datasets, release = stores
    manifest = json.loads((release / "manifest.json").read_text())
    (release / "manifest.json").write_text(json.dumps(manifest | {"release_format": 9}))
    assert problems(evidence, datasets) == [
        f"datasets/{release.name}: Release {release.name} does not match its manifest; rebuild it"
    ]


def test_a_release_built_from_evidence_outside_the_store_is_reported(stores, tmp_path):
    """It could not be rebuilt from versioned evidence."""
    evidence, datasets, _ = stores
    elsewhere = tmp_path / "scratch"
    for index in range(4):
        write_run(elsewhere / f"cpu-{index}", make_result())
    other = build([Source("local", str(elsewhere))], datasets, now=NOW).release
    [problem] = problems(evidence, datasets)
    assert problem.startswith(f"datasets/{other.name}: built from evidence that is not in the")
    assert problem.endswith("]...), so it cannot be rebuilt")  # the first three runs, then ...


def test_edited_evidence_no_longer_reproduces_its_release(stores):
    evidence, datasets, release = stores
    [entry] = list(evidence.iterdir())
    result = json.loads((entry / "results.json").read_text())
    result["contended"]["cpu_seconds"] = 0.123  # still valid evidence, but not the same
    (entry / "results.json").write_text(json.dumps(result))
    run_id = entry.name.split("-", 1)[1]
    assert problems(evidence, datasets) == [
        (
            f"datasets/{release.name}: built from evidence that is not in the evidence store "
            f"(runs ['{run_id}']), so it cannot be rebuilt"
        )
    ]


# Quality-gate attempts


def test_damaged_gate_attempts_are_reported(stores):
    evidence, datasets, release = stores
    write_attempt(datasets, release.name, "a1", result="{damaged")
    write_attempt(datasets, "steps-v1.2+cccccccccccc", "a1")
    found = problems(evidence, datasets)
    assert found[0].startswith(f"datasets/validations/{release.name}: Quality-gate verdict of ")
    assert found[1:] == ["datasets/validations/steps-v1.2+cccccccccccc: no such release"]


def test_gate_attempts_must_hold_their_matching_statistics(stores):
    evidence, datasets, release = stores
    good = write_attempt(datasets, release.name, "a1")
    (good / ".DS_Store").write_text("")  # hidden, skipped
    write_attempt(datasets, release.name, "a2", statistics=None)
    mismatched = write_attempt(datasets, release.name, "a3")
    (mismatched / "statistics.pbtxt").write_text("datasets { num_examples: 1 }")
    extra = write_attempt(datasets, release.name, "a4")
    (extra / "notes.txt").write_text("")
    (datasets / "validations" / release.name / "stray.txt").write_text("")
    label = f"datasets/validations/{release.name}"
    assert problems(evidence, datasets) == [
        f"{label}/a2: holds ['result.json'], not ['result.json', 'statistics.pbtxt']",
        f"{label}/a3: statistics do not match the hash their result recorded",
        (
            f"{label}/a4: holds ['notes.txt', 'result.json', 'statistics.pbtxt'], "
            "not ['result.json', 'statistics.pbtxt']"
        ),
        f"{label}/stray.txt: not a gate attempt",
    ]


def test_a_validations_file_is_not_a_release(stores):
    evidence, datasets, _ = stores
    (datasets / "validations").mkdir()
    (datasets / "validations" / "loose.json").write_text("{}")
    assert problems(evidence, datasets) == ["datasets/validations/loose.json: no such release"]


# Splits


def test_damaged_splits_are_reported(stores):
    evidence, datasets, release = stores
    good = write_split(datasets, release.name, {"groups": {}})
    content = json.loads(good.read_text())
    content["groups"] = {"task": "train"}  # edited without a new id
    good.write_text(json.dumps(content))
    nan = datasets / "splits" / release.name / "split-v1+000000000000.json"
    nan.write_text(f'{{"split_id": "{nan.stem}", "release": "{release.name}", "x": NaN}}')
    write_split(datasets, "steps-v1.2+dddddddddddd", {"groups": {}})
    (datasets / "splits" / "loose.json").write_text("{}")
    assert problems(evidence, datasets) == [  # entries are checked in sorted order
        "datasets/splits/loose.json: no such release",
        f"datasets/splits/{release.name}: Split {nan.name} holds a non-finite number",
        f"datasets/splits/{release.name}: Split {good.name} does not match its content or location",
        "datasets/splits/steps-v1.2+dddddddddddd: no such release",
    ]


# Append-only, against what DVC already versioned


def test_deleting_or_editing_versioned_files_is_refused(stores, tmp_path):
    evidence, datasets, release = stores
    deleted = datasets / "validations" / release.name / "a2" / "result.json"
    edited = evidence / "x" / "results.json"
    status = {
        "uncommitted": {
            "deleted": [str(deleted)],
            "modified": [str(datasets) + "/", str(edited)],
            "added": [str(datasets / "new.json")],
        },
        "committed": {"deleted": [str(tmp_path / "elsewhere.txt")]},  # outside the stores
    }
    assert problems(evidence, datasets, status) == [
        f"{deleted}: versioned file was deleted; the stores are append-only",
        f"{edited}: versioned file was modified; the stores are append-only",
    ]


def test_relative_status_paths_are_resolved_from_the_working_directory(stores, monkeypatch):
    _, datasets, release = stores
    monkeypatch.chdir(datasets.parent)
    deleted = f"datasets/{release.name}/steps.parquet"
    status = {"committed": {"deleted": [deleted]}}
    assert problems(Path("evidence"), Path("datasets"), status) == [
        f"{deleted}: versioned file was deleted; the stores are append-only"
    ]


# Command line


def test_cli_exits_zero_with_a_summary(stores, capsys):
    evidence, datasets, _ = stores
    assert main(["--evidence", str(evidence), "--datasets", str(datasets)]) == 0
    assert capsys.readouterr().out == (
        "verified: 1 runs, 1 releases (+0 superseded), 1 build reports, 0 gate attempts, "
        "0 splits (0 hidden entries excluded)\n"
    )


def test_cli_refuses_to_version_damage(stores, capsys):
    evidence, datasets, release = stores
    (release / "manifest.json").write_text('{"release_format": 2}')
    assert main(["--evidence", str(evidence), "--datasets", str(datasets)]) == 1
    err = capsys.readouterr().err
    assert f"problem: datasets/{release.name}" in err and "Not versioning: 1 problem(s)" in err


def test_cli_reads_dvc_status(stores, tmp_path, capsys):
    evidence, datasets, release = stores
    status = tmp_path / "status.json"
    args = ["--evidence", str(evidence), "--datasets", str(datasets), "--dvc-status", str(status)]
    status.write_text("{}")
    assert main(args) == 0
    status.write_text(json.dumps({"uncommitted": {"deleted": [str(release / "steps.parquet")]}}))
    assert main(args) == 1 and "append-only" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (None, "cannot read DVC status"),
        ("{x", "cannot read DVC status"),
        ("[]", "not a JSON object"),
    ],
)
def test_cli_refuses_unreadable_dvc_status(stores, tmp_path, capsys, content, message):
    evidence, datasets, _ = stores
    status = tmp_path / "status.json"
    if content is not None:
        status.write_text(content)
    args = ["--evidence", str(evidence), "--datasets", str(datasets), "--dvc-status", str(status)]
    assert main(args) == 1 and message in capsys.readouterr().err


def test_module_runs_as_a_command(stores, monkeypatch):
    import runpy
    import sys

    evidence, datasets, _ = stores
    argv = ["agentwatch_data.integrity", "--evidence", str(evidence), "--datasets", str(datasets)]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.delitem(sys.modules, "agentwatch_data.integrity", raising=False)
    with pytest.raises(SystemExit) as exited:
        runpy.run_module("agentwatch_data.integrity", run_name="__main__")
    assert exited.value.code == 0
