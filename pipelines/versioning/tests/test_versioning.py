"""End-to-end versioning with the repository's own DVC configuration and a local remote.

The acceptance check for dataset versioning: a release versioned with its evidence can be
pulled into a fresh clone and rebuilt there, byte for byte, under the same dataset version.
A local remote stands in for GCS; DVC treats both as the same content-addressed store.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
GOLDEN = next((ROOT / "pipelines/data/tests/fixtures").iterdir())
sys.path.insert(0, str(ROOT / "pipelines/data/tests"))
from dataset_fixtures import make_result, write_run

TIMEOUT_SECONDS = 300
GIT_IDENTITY = ["-c", "user.name=AgentWatch tests", "-c", "user.email=tests@agentwatch.invalid"]


def run(command: list[str], cwd: Path) -> str:
    # The data commands run in the shared workspace environment, not this one.
    env = {key: value for key, value in os.environ.items() if key != "VIRTUAL_ENV"}
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_SECONDS,
        check=False,  # failures are reported below with the command's own output
    )
    if completed.returncode != 0:
        raise AssertionError(f"{command} failed:\n{completed.stdout}\n{completed.stderr}")
    return completed.stdout


def dvc(cwd: Path, *args: str) -> str:
    return run([sys.executable, "-m", "dvc", *args], cwd)


def git(cwd: Path, *args: str) -> str:
    return run(["git", *GIT_IDENTITY, *args], cwd)


def data(cwd: Path, module: str, *args: str) -> str:
    uv = ["uv", "run", "--project", str(ROOT), "--package", "agentwatch-data", "--frozen"]
    return run([*uv, "python", "-m", f"agentwatch_data.{module}", *args], cwd)


def files(directory: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


@pytest.fixture
def repo(tmp_path) -> Path:
    """A git repository carrying this repository's DVC configuration and data layout."""
    repo = tmp_path / "repo"
    (repo / ".dvc").mkdir(parents=True)
    (repo / "data").mkdir()
    for name in (".dvc/config", ".dvc/.gitignore", ".dvcignore", "data/.gitignore"):
        shutil.copyfile(ROOT / name, repo / name)
    git(repo, "init", "-q")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "versioning layout")
    return repo


@pytest.fixture
def remote(tmp_path) -> Path:
    return tmp_path / "remote"


def version(repo: Path, remote: Path, message: str) -> str:
    """What make version-dataset and make push-dataset do, then commit the pointers."""
    data(repo, "integrity", "--evidence", "data/evidence", "--datasets", "data/datasets")
    dvc(repo, "add", "data/evidence", "data/datasets")
    if "store" not in dvc(repo, "remote", "list"):
        dvc(repo, "remote", "add", "--local", "-d", "store", str(remote))
    dvc(repo, "push")
    git(repo, "add", "data/evidence.dvc", "data/datasets.dvc", "data/.gitignore")
    git(repo, "commit", "-q", "-m", message)
    return git(repo, "rev-parse", "HEAD").strip()


def clone(repo: Path, remote: Path, target: Path) -> Path:
    git(repo.parent, "clone", "-q", str(repo), str(target))
    dvc(target, "remote", "add", "--local", "-d", "store", str(remote))
    return target


def build(repo: Path, output: str) -> Path:
    out = data(repo, "build", "--source", "local:data/evidence", "--output", output)
    return repo / out.split("release: ", 1)[1].splitlines()[0].strip()


def test_committed_config_keeps_analytics_off_and_holds_no_remote():
    config = (ROOT / ".dvc/config").read_text()
    assert "analytics = false" in config and "check_update = false" in config
    assert "remote" not in config  # remotes and credentials are configured locally
    assert (ROOT / "data/.gitignore").read_text().split() == ["/evidence", "/datasets"]


def test_a_versioned_release_is_rebuilt_byte_for_byte_from_versioned_evidence(
    repo, remote, tmp_path
):
    shutil.copytree(GOLDEN, tmp_path / "recorded" / "coding-run")
    data(repo, "evidence", "--store", "data/evidence", str(tmp_path / "recorded" / "coding-run"))
    release = build(repo, "data/datasets")
    version(repo, remote, "version the recorded run")

    copy = clone(repo, remote, tmp_path / "clone")
    assert not (copy / "data/datasets").exists()  # git holds only the pointers
    dvc(copy, "pull")
    for store in ("data/evidence", "data/datasets"):
        assert files(copy / store) == files(repo / store)

    rebuilt = build(copy, str(tmp_path / "rebuilt"))
    assert rebuilt.name == release.name
    assert files(rebuilt) == files(copy / "data/datasets" / release.name)


def test_one_release_can_be_pulled_alone(repo, remote, tmp_path):
    shutil.copytree(GOLDEN, tmp_path / "recorded" / "coding-run")
    data(repo, "evidence", "--store", "data/evidence", str(tmp_path / "recorded" / "coding-run"))
    release = build(repo, "data/datasets")
    version(repo, remote, "version")
    copy = clone(repo, remote, tmp_path / "clone")
    dvc(copy, "pull", f"data/datasets/{release.name}")
    assert sorted(path.name for path in (copy / "data/datasets").iterdir()) == [release.name]
    assert not (copy / "data/evidence").exists()


def test_staging_is_never_versioned(repo, remote, tmp_path):
    shutil.copytree(GOLDEN, tmp_path / "recorded" / "coding-run")
    data(repo, "evidence", "--store", "data/evidence", str(tmp_path / "recorded" / "coding-run"))
    build(repo, "data/datasets")
    (repo / "data/datasets/.building-x").mkdir()
    (repo / "data/datasets/.building-x/steps.parquet").write_text("half written")
    (repo / "data/evidence/.adding-y").mkdir()
    (repo / "data/datasets/builds/.report.json.tmp").write_text("{")
    version(repo, remote, "version during a build")
    copy = clone(repo, remote, tmp_path / "clone")
    dvc(copy, "pull")
    pulled = [*files(copy / "data/datasets"), *files(copy / "data/evidence")]
    assert not [name for name in pulled if "/." in f"/{name}"]


def test_every_committed_version_stays_retrievable(repo, remote, tmp_path):
    shutil.copytree(GOLDEN, tmp_path / "recorded" / "coding-run")
    data(repo, "evidence", "--store", "data/evidence", str(tmp_path / "recorded" / "coding-run"))
    first_release = build(repo, "data/datasets")
    first = version(repo, remote, "first")

    result = make_result()
    write_run(tmp_path / "recorded" / "cpu-run", result)
    data(repo, "evidence", "--store", "data/evidence", str(tmp_path / "recorded" / "cpu-run"))
    second_release = build(repo, "data/datasets")
    version(repo, remote, "second")
    assert second_release != first_release

    copy = clone(repo, remote, tmp_path / "clone")
    git(copy, "checkout", "-q", first)
    dvc(copy, "pull")
    assert sorted(p.name for p in (copy / "data/evidence").iterdir()) == [
        next((repo / "data/evidence").glob("coding-*")).name
    ]
    releases = [p.name for p in (copy / "data/datasets").iterdir() if p.name.startswith("steps")]
    assert releases == [first_release.name]  # the store exactly as it was at that commit
    report = json.loads(next((copy / "data/datasets/builds").iterdir()).read_text())
    assert report["release"] == first_release.name
