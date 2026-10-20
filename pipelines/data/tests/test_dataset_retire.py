import json
import runpy
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from agentwatch_data.build import Source, build
from agentwatch_data.evidence import add_run
from agentwatch_data.integrity import verify
from agentwatch_data.retire import RetireError, main, retire

GOLDEN = next((Path(__file__).parent / "fixtures").iterdir())
NOW = datetime(2026, 10, 18, tzinfo=UTC)


@pytest.fixture
def stores(tmp_path):
    """An evidence store and a release built from it, with an attempt and a split."""
    evidence, datasets = tmp_path / "evidence", tmp_path / "datasets"
    shutil.copytree(GOLDEN, tmp_path / "recorded" / "coding-run")
    add_run(tmp_path / "recorded" / "coding-run", evidence)
    release = build([Source("local", str(evidence))], datasets, now=NOW).release
    attempt = datasets / "validations" / release.name / "a1"
    attempt.mkdir(parents=True)
    (attempt / "result.json").write_text("{}")
    split = datasets / "splits" / release.name
    split.mkdir(parents=True)
    (split / "split-v1+000000000000.json").write_text("{}")
    return evidence, datasets, release


def test_a_release_and_everything_naming_it_are_removed(stores):
    evidence, datasets, release = stores
    [report] = (datasets / "builds").iterdir()
    other = datasets / "builds" / "other.json"
    other.write_text(json.dumps({"release": "steps-v1.2+bbbbbbbbbbbb"}))
    (datasets / "builds" / "damaged.json").write_text("{")
    removed = retire(datasets, release.name, {})
    assert removed == [
        datasets / "validations" / release.name,
        datasets / "splits" / release.name,
        report,
        release,
    ]
    assert not release.exists() and other.exists()  # reports of other releases stay
    assert verify(evidence, datasets).problems == [
        "datasets/builds/damaged.json: unreadable build report",
        f"datasets/builds/{other.name}: names release 'steps-v1.2+bbbbbbbbbbbb', which is missing",
    ]


def test_an_interrupted_retire_is_finished_by_running_it_again(stores):
    _, datasets, release = stores
    shutil.rmtree(datasets / "validations" / release.name)
    [report] = (datasets / "builds").iterdir()
    report.unlink()
    assert retire(datasets, release.name, {}) == [datasets / "splits" / release.name, release]
    with pytest.raises(RetireError, match="No release"):
        retire(datasets, release.name, {})


@pytest.mark.parametrize(
    "where",
    [
        ("unchanged",),
        ("committed", "added"),
        ("committed", "modified"),
        ("uncommitted", "deleted"),
    ],
)
def test_anything_dvc_already_holds_is_never_removed(stores, where):
    _, datasets, release = stores
    held = str(datasets / "splits" / release.name / "split-v1+000000000000.json")
    status = {where[0]: [held]} if len(where) == 1 else {where[0]: {where[1]: [held]}}
    with pytest.raises(RetireError, match="is already versioned"):
        retire(datasets, release.name, status)
    assert release.exists() and (datasets / "validations" / release.name).exists()


def test_files_added_since_the_last_snapshot_may_be_removed(stores):
    _, datasets, release = stores
    status = {"uncommitted": {"added": [str(release / "steps.parquet")]}}
    assert release in retire(datasets, release.name, status)


def test_only_release_names_are_accepted(stores):
    _, datasets, _ = stores
    with pytest.raises(RetireError, match="is not a release name"):
        retire(datasets, "../evidence", {})


def test_a_linked_release_is_removed_as_a_link(stores, tmp_path):
    _, datasets, release = stores
    target = tmp_path / "elsewhere"
    release.rename(target)
    release.symlink_to(target)
    assert release in retire(datasets, release.name, {})
    assert target.exists() and not release.is_symlink()


def test_cli_removes_and_reports(stores, tmp_path, capsys):
    _, datasets, release = stores
    status = tmp_path / "status.json"
    status.write_text("{}")
    args = ["--datasets", str(datasets), "--dvc-status", str(status), release.name]
    assert main(args) == 0
    assert capsys.readouterr().out.endswith(f"removed: {release}\n")
    assert main(args) == 1
    assert "Not retiring: No release" in capsys.readouterr().err


def test_cli_refuses_without_a_readable_status(stores, tmp_path, capsys):
    _, datasets, release = stores
    args = ["--datasets", str(datasets), "--dvc-status", str(tmp_path / "missing.json")]
    assert main([*args, release.name]) == 1
    assert "Not retiring: cannot read DVC status" in capsys.readouterr().err
    assert release.exists()


def test_module_runs_as_a_command(stores, tmp_path, monkeypatch):
    _, datasets, release = stores
    status = tmp_path / "status.json"
    status.write_text("{}")
    argv = ["retire", "--datasets", str(datasets), "--dvc-status", str(status), release.name]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.delitem(sys.modules, "agentwatch_data.retire", raising=False)
    with pytest.raises(SystemExit) as exited:
        runpy.run_module("agentwatch_data.retire", run_name="__main__")
    assert exited.value.code == 0
