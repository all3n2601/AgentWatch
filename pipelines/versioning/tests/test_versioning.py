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
GIT_IDENTITY = [
    "-c",
    "user.name=AgentWatch tests",
    "-c",
    "user.email=tests@agentwatch.invalid",
    "-c",
    "commit.gpgsign=false",
]


@pytest.fixture(autouse=True)
def isolated_tools(tmp_path, monkeypatch):
    """Keep the developer's own git and DVC settings out of the tests.

    Without this, a global commit-signing setup signs every test commit with a real key (or
    blocks on a passphrase), and DVC reads global config and leaves site caches behind.
    """
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for variable in ("DVC_GLOBAL_CONFIG_DIR", "DVC_SYSTEM_CONFIG_DIR", "DVC_SITE_CACHE_DIR"):
        directory = tmp_path / "dvc-home" / variable.lower()
        directory.mkdir(parents=True)
        monkeypatch.setenv(variable, str(directory))


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
    for name in (
        ".dvc/config",
        ".dvc/.gitignore",
        ".dvcignore",
        ".gitattributes",
        "data/.gitignore",
    ):
        shutil.copyfile(ROOT / name, repo / name)
    git(repo, "init", "-q")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "versioning layout")
    return repo


@pytest.fixture
def remote(tmp_path) -> Path:
    return tmp_path / "remote"


def snapshot(repo: Path) -> str:
    """What make version-dataset does; returns the integrity check's summary."""
    status = repo / ".dvc/tmp/version-dataset-status.json"
    status.parent.mkdir(parents=True, exist_ok=True)
    status.write_text(dvc(repo, "data", "status", "--granular", "--unchanged", "--json"))
    stores = ["--evidence", "data/evidence", "--datasets", "data/datasets"]
    summary = data(repo, "integrity", *stores, "--dvc-status", str(status))
    dvc(repo, "add", "data/evidence", "data/datasets")
    return summary


def version(repo: Path, remote: Path, message: str) -> str:
    """make version-dataset and make push-dataset, then commit the pointers."""
    snapshot(repo)
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
    # Only the new release is rebuilt from evidence; DVC reports the first one unchanged.
    assert "; 1 releases rebuilt from evidence" in snapshot(repo)
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


def write_attempt(repo: Path, release: str, name: str, status: str, validated_at: str) -> None:
    """A committed quality-gate attempt, as the gate writes it."""
    import hashlib

    directory = repo / "data/datasets/validations" / release / name
    directory.mkdir(parents=True)
    statistics = f"datasets {{ name: '{name}' }}"
    (directory / "statistics.pbtxt").write_text(statistics)
    result = {
        "attempt": name,
        "release": release,
        "status": status,
        "validated_at": validated_at,
        "statistics_sha256": hashlib.sha256(statistics.encode()).hexdigest(),
    }
    (directory / "result.json").write_text(json.dumps(result))


def test_a_versioned_failed_verdict_cannot_be_deleted(repo, remote, tmp_path):
    """Deleting the later failed attempt would turn the release's verdict back into a pass."""
    shutil.copytree(GOLDEN, tmp_path / "recorded" / "coding-run")
    data(repo, "evidence", "--store", "data/evidence", str(tmp_path / "recorded" / "coding-run"))
    release = build(repo, "data/datasets")
    write_attempt(repo, release.name, "a1-pass", "passed", "2026-10-18T10:00:00+00:00")
    write_attempt(repo, release.name, "a2-fail", "failed", "2026-10-18T11:00:00+00:00")
    version(repo, remote, "version with a failed verdict")
    shutil.rmtree(repo / "data/datasets/validations" / release.name / "a2-fail")
    with pytest.raises(
        AssertionError, match="versioned file was deleted; the stores are append-only"
    ):
        snapshot(repo)


def test_hidden_entries_are_never_versioned_or_built(repo, remote, tmp_path):
    shutil.copytree(GOLDEN, tmp_path / "recorded" / "coding-run")
    data(repo, "evidence", "--store", "data/evidence", str(tmp_path / "recorded" / "coding-run"))
    release = build(repo, "data/datasets")
    [entry] = (repo / "data/evidence").iterdir()
    for directory in (repo / "data/evidence", entry, repo / "data/datasets", release):
        (directory / ".DS_Store").write_text("Finder state")
    write_run(repo / "data/evidence/.old/cpu-archived", make_result())
    version(repo, remote, "version beside hidden entries")
    copy = clone(repo, remote, tmp_path / "clone")
    dvc(copy, "pull")
    pulled = [*files(copy / "data/evidence"), *files(copy / "data/datasets")]
    assert not [name for name in pulled if "/." in f"/{name}"]
    assert build(copy, str(tmp_path / "rebuilt")).name == release.name


def merge_driver(clone_dir: Path) -> None:
    """What make dataset-merge-driver configures, with this environment's DVC."""
    git(clone_dir, "config", "merge.dvc.name", "DVC merge driver")
    driver = f"{sys.executable} -m dvc git-hook merge-driver --ancestor %O --our %A --their %B"
    git(clone_dir, "config", "merge.dvc.driver", driver)


def test_concurrent_versions_merge_into_their_union(repo, remote, tmp_path):
    """Two people version different runs from the same commit; git merges both versions."""
    origin = tmp_path / "origin.git"
    git(tmp_path, "clone", "-q", "--bare", str(repo), str(origin))
    branch = git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip()
    first, second = (
        clone(origin, remote, tmp_path / "first"),
        clone(origin, remote, tmp_path / "second"),
    )
    for person in (first, second):
        merge_driver(person)

    shutil.copytree(GOLDEN, tmp_path / "recorded" / "coding-run")
    data(first, "evidence", "--store", "data/evidence", str(tmp_path / "recorded" / "coding-run"))
    first_release = build(first, "data/datasets")
    version(first, remote, "first versions the recorded run")
    git(first, "push", "-q", "origin", f"HEAD:{branch}")

    write_run(tmp_path / "recorded" / "cpu-run", make_result())
    data(second, "evidence", "--store", "data/evidence", str(tmp_path / "recorded" / "cpu-run"))
    second_release = build(second, "data/datasets")
    version(second, remote, "second versions a CPU run")

    git(second, "fetch", "-q", "origin")
    dvc(second, "fetch", "--all-commits")  # make fetch-dataset: the driver reads both listings
    git(second, "merge", "-q", "-m", "merge", f"origin/{branch}")  # no conflict to resolve
    dvc(second, "pull")
    releases = sorted(p.name for p in (second / "data/datasets").glob("steps-*"))
    assert releases == sorted([first_release.name, second_release.name])
    assert len(list((second / "data/evidence").iterdir())) == 2
    snapshot(second)  # intact, reproducible, and nothing versioned was lost in the merge
    assert dvc(second, "data", "status", "--granular", "--json").strip() == "{}"
