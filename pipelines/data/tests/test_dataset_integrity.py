import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest
from agentwatch_data.build import Source, build
from agentwatch_data.evidence import add_run
from agentwatch_data.extract import canonical_sha256
from agentwatch_data.integrity import main, verify

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


def problems(evidence, datasets):
    return verify(evidence, datasets).problems


def write_split(datasets: Path, release: str, content: dict) -> Path:
    """A split file whose id hashes its content, as split-dataset writes it."""
    document = {"release": release} | content
    split_id = f"split-v1+{canonical_sha256(document)[:12]}"
    path = datasets / "splits" / release / f"{split_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"split_id": split_id} | document))
    return path


def write_attempt(datasets: Path, release: str, name: str, content) -> None:
    directory = datasets / "validations" / release / name
    directory.mkdir(parents=True)
    text = content if isinstance(content, str) else json.dumps(content)
    (directory / "result.json").write_text(text)


def passing(release: str, attempt: str) -> dict:
    return {
        "attempt": attempt,
        "release": release,
        "status": "passed",
        "validated_at": "2026-10-18T00:00:00+00:00",
    }


def test_intact_stores_are_counted(stores):
    evidence, datasets, release = stores
    write_attempt(datasets, release.name, "a1", passing(release.name, "a1"))
    write_split(datasets, release.name, {"groups": {}})
    (datasets / ".building-x").mkdir()  # a build in progress is not part of a version
    (evidence / ".adding-y").mkdir()
    report = verify(evidence, datasets)
    assert report.problems == []
    assert (report.runs, report.releases, report.reports, report.splits) == (1, 1, 1, 1)
    assert report.staging == 2


def test_missing_stores_are_problems(tmp_path):
    assert problems(tmp_path / "no-evidence", tmp_path / "no-datasets") == [
        f"evidence store {tmp_path / 'no-evidence'} does not exist",
        f"datasets store {tmp_path / 'no-datasets'} does not exist",
    ]


def test_damaged_evidence_is_reported(stores, tmp_path):
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
            "evidence/cpu-broken: Stored evidence cpu-broken is unreadable: "
            "JSONDecodeError: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)"
        ),
        "evidence/cpu-linked: not a run directory",
        "evidence/notes.txt: not a run directory",
    ]


def test_damaged_releases_and_reports_are_reported(stores):
    evidence, datasets, release = stores
    (release / "steps.parquet").write_bytes(b"tampered")
    (datasets / "stray.txt").write_text("")
    (datasets / "steps-v1.2+aaaaaaaaaaaa").symlink_to(release, target_is_directory=True)
    (datasets / "builds" / "broken.json").write_text("{damaged")
    (datasets / "builds" / "notes.txt").write_text("{}")
    assert problems(evidence, datasets) == [
        f"datasets/{release.name}: Release {release.name} does not match its manifest; rebuild it",
        "datasets/steps-v1.2+aaaaaaaaaaaa: unexpected entry",
        "datasets/stray.txt: unexpected entry",
        "datasets/builds/broken.json: unreadable build report",
        "datasets/builds/notes.txt: unreadable build report",
    ]


def test_a_renamed_release_is_reported(stores):
    evidence, datasets, release = stores
    release.rename(datasets / "steps-v1.2+bbbbbbbbbbbb")
    assert problems(evidence, datasets) == [
        (
            "datasets/steps-v1.2+bbbbbbbbbbbb: Release steps-v1.2+bbbbbbbbbbbb does not match its "
            "manifest; rebuild it"
        )
    ]


def test_damaged_validations_are_reported(stores):
    evidence, datasets, release = stores
    write_attempt(datasets, release.name, "a1", "{damaged")
    write_attempt(datasets, "steps-v1.2+cccccccccccc", "a1", passing("x", "a1"))
    found = problems(evidence, datasets)
    assert found[0].startswith(f"datasets/validations/{release.name}: Quality-gate verdict of ")
    assert found[1:] == ["datasets/validations/steps-v1.2+cccccccccccc: no such release"]


def test_damaged_splits_are_reported(stores):
    evidence, datasets, release = stores
    good = write_split(datasets, release.name, {"groups": {}})
    content = json.loads(good.read_text())
    content["groups"] = {"task": "train"}  # edited without a new id
    good.write_text(json.dumps(content))
    write_split(datasets, "steps-v1.2+dddddddddddd", {"groups": {}})
    (datasets / "splits" / "loose.json").write_text("{}")
    assert problems(evidence, datasets) == [  # entries are checked in sorted order
        "datasets/splits/loose.json: no such release",
        f"datasets/splits/{release.name}: Split {good.name} does not match its content or location",
        "datasets/splits/steps-v1.2+dddddddddddd: no such release",
    ]


def test_cli_exits_zero_with_a_summary(stores, capsys):
    evidence, datasets, _ = stores
    assert main(["--evidence", str(evidence), "--datasets", str(datasets)]) == 0
    assert capsys.readouterr().out == (
        "verified: 1 runs, 1 releases, 1 build reports, 0 splits "
        "(0 hidden staging entries excluded)\n"
    )


def test_cli_refuses_to_version_damage(stores, capsys):
    evidence, datasets, release = stores
    (release / "manifest.json").write_text("{}")
    assert main(["--evidence", str(evidence), "--datasets", str(datasets)]) == 1
    err = capsys.readouterr().err
    assert f"problem: datasets/{release.name}" in err and "Not versioning: 1 problem(s)" in err


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
