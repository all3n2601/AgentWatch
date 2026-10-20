import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
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


def format_1(release: Path) -> Path:
    """A genuine superseded release beside release: what the builder wrote before schema.json.

    Its version came from the rows alone, and its manifest had no format or contract hash.
    """
    manifest = json.loads((release / "manifest.json").read_text())
    name = f"{manifest['schema_version']}+{manifest['rows_sha256'][:12]}"
    old = release.parent / name
    old.mkdir()
    table = pq.read_table(release / "steps.parquet")
    index = table.column_names.index("dataset_version")
    versions = pa.array([name] * table.num_rows, pa.string())
    pq.write_table(
        table.set_column(index, table.schema.field(index), versions), old / "steps.parquet"
    )
    del manifest["release_format"], manifest["schema_sha256"]
    manifest |= {"dataset_version": name, "parquet_sha256": sha256(old / "steps.parquet")}
    (old / "manifest.json").write_text(json.dumps(manifest))
    return old


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_superseded_releases_are_accepted_by_their_own_format(stores):
    evidence, datasets, release = stores
    old = format_1(release)
    report = verify(evidence, datasets)
    assert report.problems == [] and (report.releases, report.superseded) == (1, 1)
    (old / "steps.parquet").write_bytes(b"tampered")
    assert problems(evidence, datasets) == [
        f"datasets/{old.name}: Release {old.name} does not match its manifest (no format)"
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
        "0 splits; 1 releases rebuilt from evidence (0 hidden entries excluded)\n"
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


# Releases must be exactly what their content produces (review of #28)


def rewrite_column(release: Path, column: str, change) -> None:
    """Edit one column of steps.parquet, keeping its Arrow type."""
    table = pq.read_table(release / "steps.parquet")
    index = table.column_names.index(column)
    values = [change(value) for value in table.column(column).to_pylist()]
    array = pa.array(values, type=table.schema.field(index).type)
    pq.write_table(
        table.set_column(index, table.schema.field(index), array), release / "steps.parquet"
    )


def resign(release: Path, **fields) -> None:
    """Update the manifest so every file hash matches again, plus any fields given."""
    manifest = json.loads((release / "manifest.json").read_text())
    manifest["parquet_sha256"] = sha256(release / "steps.parquet")
    if (release / "schema.json").exists():
        manifest["schema_sha256"] = sha256(release / "schema.json")
    (release / "manifest.json").write_text(json.dumps(manifest | fields))


def test_edited_rows_with_a_resigned_manifest_are_refused(stores):
    """The review's reproduction: change queue_seconds, update only parquet_sha256."""
    evidence, datasets, release = stores
    rewrite_column(release, "queue_seconds", lambda value: value + 1.0)
    resign(release)
    [problem] = problems(evidence, datasets)
    assert problem.startswith(f"datasets/{release.name}: its content produces steps-v1.2+")
    assert problem.endswith("not its name; it was changed after it was built")


def test_edited_rows_with_a_resigned_row_digest_are_refused_too(stores):
    from agentwatch_data.build import rows_digest

    evidence, datasets, release = stores
    rewrite_column(release, "queue_seconds", lambda value: value + 1.0)
    rows = pq.read_table(release / "steps.parquet").to_pylist()
    digest = rows_digest([{k: v for k, v in row.items() if k != "dataset_version"} for row in rows])
    resign(release, rows_sha256=digest)  # every hash consistent; only the name is old
    assert "not its name" in problems(evidence, datasets)[0]


@pytest.mark.parametrize(
    "fields",
    [
        {"rows": 99},
        {"label_resource": {"cpu_sandbox": 7}},
        {"telemetry_status": {}},
        {"rows_sha256": "0" * 64},
        {"unexpected": 1},
    ],
)
def test_manifest_summaries_must_match_the_rows(stores, fields):
    evidence, datasets, release = stores
    resign(release, **fields)
    [problem] = problems(evidence, datasets)
    assert (
        problem
        == f"datasets/{release.name}: manifest fields {sorted(fields)} do not match its rows"
    )


def test_the_builder_version_is_not_content(stores):
    evidence, datasets, release = stores
    resign(release, builder_version="9.9.9")  # recorded by the builder, not derived from rows
    assert problems(evidence, datasets) == []


@pytest.mark.parametrize("column", ["dataset_version", "label_version", "schema_version"])
def test_rows_must_carry_the_manifest_versions(stores, column):
    evidence, datasets, release = stores
    first = []

    def relabel(value):
        first.append(value)
        return "other" if len(first) == 1 else value

    rewrite_column(release, column, relabel)
    resign(release)
    [problem] = problems(evidence, datasets)
    assert f"rows do not all carry the manifest's {column}" in problem


def test_an_edited_contract_is_refused(stores):
    evidence, datasets, release = stores
    contract = json.loads((release / "schema.json").read_text())
    contract["columns"][0]["description"] = "edited"
    (release / "schema.json").write_text(json.dumps(contract))
    resign(release)
    assert problems(evidence, datasets) == [
        f"datasets/{release.name}: schema.json is not the frozen steps-v1.2 column contract"
    ]


def test_a_reformatted_contract_is_refused_too(stores):
    """Same JSON, other bytes: a rebuild would not reproduce the release byte for byte."""
    evidence, datasets, release = stores
    contract = json.loads((release / "schema.json").read_text())
    (release / "schema.json").write_text(json.dumps(contract))
    resign(release)
    assert "schema.json is not the frozen steps-v1.2" in problems(evidence, datasets)[0]


def test_an_edited_superseded_release_is_refused(stores):
    evidence, datasets, release = stores
    old = format_1(release)
    rewrite_column(old, "queue_seconds", lambda value: value + 1.0)
    resign(old)
    [problem] = problems(evidence, datasets)
    assert problem.startswith(f"datasets/{old.name}: its content produces")


def test_identity_is_recomputed_for_rows_with_saved_times(tmp_path, monkeypatch):
    """API runs carry a recorded_at timestamp, which round-trips through Parquet."""
    from agentwatch_data import build as build_module
    from agentwatch_data.extract import Extraction, validate
    from agentwatch_data.integrity import identity_problem, read_rows, release_columns

    saved = datetime(2026, 10, 5, 12, 30, tzinfo=UTC)
    runs = [validate(make_result(), [], "cpu", saved, origin="api") for _ in range(2)]
    monkeypatch.setattr(build_module, "extract", lambda source: Extraction(runs=list(runs)))
    release = build([Source("api", "http://api")], tmp_path / "datasets", now=NOW).release
    manifest = json.loads((release / "manifest.json").read_text())
    rows = read_rows(release, release_columns(release, manifest, True))
    assert identity_problem(release, manifest, True, rows) is None


# A release holds exactly its declared files (review of #28)


def test_a_release_holding_extra_files_is_refused(stores):
    evidence, datasets, release = stores
    (release / "notes.txt").write_text("not part of the release")
    assert problems(evidence, datasets) == [
        (
            f"datasets/{release.name}: holds ['manifest.json', 'notes.txt', 'schema.json', "
            "'steps.parquet'], not ['manifest.json', 'schema.json', 'steps.parquet']"
        )
    ]


def test_a_release_missing_a_file_or_holding_links_or_folders_is_refused(stores, tmp_path):
    evidence, datasets, release = stores
    (release / "nested").mkdir()
    assert "holds ['manifest.json', 'nested'" in problems(evidence, datasets)[0]
    (release / "nested").rmdir()
    elsewhere = tmp_path / "steps.parquet"
    (release / "steps.parquet").rename(elsewhere)
    (release / "steps.parquet").symlink_to(elsewhere)
    assert (
        "holds ['manifest.json', 'schema.json', 'steps.parquet'], not"
        in problems(evidence, datasets)[0]
    )
    (release / "steps.parquet").unlink()
    elsewhere.rename(release / "steps.parquet")
    (release / "schema.json").unlink()
    assert "holds ['manifest.json', 'steps.parquet'], not" in problems(evidence, datasets)[0]


def test_hidden_files_in_a_release_are_skipped(stores):
    evidence, datasets, release = stores
    (release / ".DS_Store").write_text("Finder state")
    report = verify(evidence, datasets)
    assert report.problems == [] and report.hidden == 1


def test_a_superseded_release_holds_no_contract(stores):
    evidence, datasets, release = stores
    old = format_1(release)
    (old / "schema.json").write_text("{}")
    assert problems(evidence, datasets) == [
        (
            f"datasets/{old.name}: holds ['manifest.json', 'schema.json', 'steps.parquet'], "
            "not ['manifest.json', 'steps.parquet']"
        )
    ]


# Every row names its exact evidence, and malformed files are refused (re-review of #28)


def consistent_release_from(rows: list[dict], datasets: Path, label_version: str) -> Path:
    """Write rows as a release whose name, hashes, and manifest all agree with them."""
    from agentwatch_data.build import release_digest, rows_digest, write_release

    labeled = [{k: v for k, v in row.items() if k != "dataset_version"} for row in rows]
    digest = rows_digest(labeled)
    version = f"steps-v1.2+{release_digest(digest)[:12]}"
    final = [row | {"dataset_version": version} for row in labeled]
    write_release(datasets / version, final, version, digest, label_version)
    return datasets / version


def test_a_row_naming_different_evidence_is_refused_even_when_consistent(stores):
    """The review's reproduction: one row's source_sha256 changed, everything re-derived."""
    evidence, datasets, release = stores
    rows = pq.read_table(release / "steps.parquet").to_pylist()
    rows[0]["source_sha256"] = "0" * 64  # attributed to a source artifact that is not stored
    manifest = json.loads((release / "manifest.json").read_text())
    forged = consistent_release_from(rows, datasets, manifest["label_version"])
    run_id = rows[0]["run_id"]
    assert problems(evidence, datasets) == [
        (
            f"datasets/{forged.name}: rows of one run name different evidence hashes "
            f"(runs ['{run_id}'])"
        )
    ]


def test_hash_consistent_bytes_that_are_not_parquet_are_refused(stores):
    evidence, datasets, release = stores
    (release / "steps.parquet").write_bytes(b"not parquet at all")
    resign(release)
    [problem] = problems(evidence, datasets)
    assert problem.startswith(f"datasets/{release.name}: steps.parquet is not readable Parquet")


def test_readable_parquet_missing_a_release_column_is_refused(stores):
    evidence, datasets, release = stores
    table = pq.read_table(release / "steps.parquet")
    pq.write_table(table.drop(["recorded_at"]), release / "steps.parquet")
    resign(release)
    [problem] = problems(evidence, datasets)
    assert problem == (
        f"datasets/{release.name}: steps.parquet does not have its contract's columns and types"
    )


# Rows are held to their version's frozen contract, and stores hold only real files
# (re-review of #28 at 0a12934)


def forge(datasets: Path, table: pa.Table, label_version: str) -> Path:
    """Write table as a release whose name, file hashes, digest, and manifest all agree.

    Unlike consistent_release_from, the table keeps whatever columns and types it has.
    """
    from agentwatch_data.build import contract_text, release_digest, release_manifest, rows_digest

    rows = [{k: v for k, v in row.items() if k != "dataset_version"} for row in table.to_pylist()]
    digest = rows_digest(rows)
    version = f"steps-v1.2+{release_digest(digest)[:12]}"
    release = datasets / version
    release.mkdir()
    index = table.column_names.index("dataset_version")
    versions = pa.array([version] * table.num_rows, pa.string())
    table = table.set_column(index, table.schema.field(index), versions)
    pq.write_table(table, release / "steps.parquet")
    (release / "schema.json").write_text(contract_text())
    manifest = release_manifest(
        [row | {"dataset_version": version} for row in rows],
        version,
        digest,
        label_version,
        sha256(release / "steps.parquet"),
        sha256(release / "schema.json"),
    )
    (release / "manifest.json").write_text(json.dumps(manifest))
    return release


def test_a_consistent_release_without_evidence_hashes_is_refused(stores):
    """The review's reproduction: source_sha256 removed and every hash and name re-derived."""
    evidence, datasets, release = stores
    table = pq.read_table(release / "steps.parquet").drop(["source_sha256"])
    label_version = json.loads((release / "manifest.json").read_text())["label_version"]
    forged = forge(datasets, table, label_version)
    assert problems(evidence, datasets) == [
        f"datasets/{forged.name}: steps.parquet does not have its contract's columns and types"
    ]


def test_a_consistent_release_with_a_retyped_column_is_refused(stores):
    evidence, datasets, release = stores
    table = pq.read_table(release / "steps.parquet")
    index = table.column_names.index("queue_seconds")
    as_text = table.column(index).cast(pa.string())
    table = table.set_column(index, "queue_seconds", as_text)
    label_version = json.loads((release / "manifest.json").read_text())["label_version"]
    forged = forge(datasets, table, label_version)
    assert "does not have its contract's columns and types" in problems(evidence, datasets)[0]
    assert forged.name in problems(evidence, datasets)[0]


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("source_sha256", "", "source_sha256 must be a nonempty string"),
        ("queue_seconds", -1.0, "queue_seconds must be at least 0"),
        ("label_resource", "disk", "label_resource must be one of"),
    ],
)
def test_a_consistent_release_breaking_the_contract_is_refused(stores, column, value, message):
    evidence, datasets, release = stores
    table = pq.read_table(release / "steps.parquet")
    index = table.column_names.index(column)
    field = table.schema.field(index)
    values = pa.array([value] * table.num_rows, field.type)
    table = table.set_column(index, field, values)
    label_version = json.loads((release / "manifest.json").read_text())["label_version"]
    forged = forge(datasets, table, label_version)
    [problem] = problems(evidence, datasets)
    assert problem.startswith(f"datasets/{forged.name}: rows break their column contract: Row 0")
    assert message in problem


def test_timestamps_python_cannot_represent_are_refused(stores):
    evidence, datasets, release = stores
    table = pq.read_table(release / "steps.parquet")
    index = table.column_names.index("recorded_at")
    field = table.schema.field(index)
    far = pa.array([2**62] * table.num_rows, pa.int64()).cast(field.type)
    pq.write_table(table.set_column(index, field, far), release / "steps.parquet")
    resign(release)
    [problem] = problems(evidence, datasets)
    assert problem.startswith(f"datasets/{release.name}: steps.parquet is not readable Parquet")


@pytest.mark.parametrize("version", ["steps-v9.9", None, ["steps-v1.2"]])
def test_a_release_of_an_unknown_schema_version_is_refused(stores, version):
    evidence, datasets, release = stores
    resign(release, schema_version=version)
    assert problems(evidence, datasets) == [
        f"datasets/{release.name}: schema version {version!r} has no frozen column contract"
    ]


def test_a_superseded_release_is_held_to_its_versions_contract(stores):
    evidence, datasets, release = stores
    old = format_1(release)
    rewrite_column(old, "queue_seconds", lambda value: -1.0)
    resign(old)
    [problem] = problems(evidence, datasets)
    assert problem.startswith(f"datasets/{old.name}: rows break their column contract")


def test_gate_attempt_files_must_be_real_files(stores, tmp_path):
    """The review's reproduction: both files linked to correctly hashed files outside."""
    evidence, datasets, release = stores
    linked = write_attempt(datasets, release.name, "a1")
    outside = tmp_path / "outside"
    outside.mkdir()
    for name in ("result.json", "statistics.pbtxt"):
        (linked / name).rename(outside / name)
        (linked / name).symlink_to(outside / name)
    folder = write_attempt(datasets, release.name, "a2")
    (folder / "statistics.pbtxt").unlink()
    (folder / "statistics.pbtxt").mkdir()
    label = f"datasets/validations/{release.name}"
    expected = "holds ['result.json', 'statistics.pbtxt'], not ['result.json', 'statistics.pbtxt']"
    assert problems(evidence, datasets) == [f"{label}/a1: {expected}", f"{label}/a2: {expected}"]


def test_a_linked_gate_attempt_is_refused(stores, tmp_path):
    evidence, datasets, release = stores
    real = write_attempt(datasets, release.name, "a1")
    real.rename(tmp_path / "a1")
    real.symlink_to(tmp_path / "a1")
    assert problems(evidence, datasets) == [
        f"datasets/validations/{release.name}/a1: not a gate attempt"
    ]


def test_unreadable_gate_statistics_are_reported(stores):
    evidence, datasets, release = stores
    attempt = write_attempt(datasets, release.name, "a1")
    (attempt / "statistics.pbtxt").write_bytes(b"\xff\xfe not utf-8")
    [problem] = problems(evidence, datasets)
    assert problem.startswith(f"datasets/validations/{release.name}/a1: unreadable: ")


def test_linked_splits_reports_and_store_folders_are_refused(stores, tmp_path):
    evidence, datasets, release = stores
    split = write_split(datasets, release.name, {"rows": 1})
    outside = tmp_path / "outside"
    outside.mkdir()
    split.rename(outside / split.name)
    split.symlink_to(outside / split.name)
    [report] = sorted((datasets / "builds").iterdir())
    report.rename(outside / report.name)
    report.symlink_to(outside / report.name)
    assert problems(evidence, datasets) == [
        f"datasets/builds/{report.name}: unreadable build report",
        f"datasets/splits/{release.name}/{split.name}: not a file",
    ]
    report.unlink()
    (datasets / "splits").rename(outside / "splits")
    (datasets / "splits").symlink_to(outside / "splits")
    assert problems(evidence, datasets) == ["datasets/splits: unexpected entry"]


def test_a_linked_store_folder_of_a_release_is_refused(stores, tmp_path):
    evidence, datasets, release = stores
    write_split(datasets, release.name, {"rows": 1})
    folder = datasets / "splits" / release.name
    folder.rename(tmp_path / "linked")
    folder.symlink_to(tmp_path / "linked")
    validations = write_attempt(datasets, release.name, "a1").parent
    validations.rename(tmp_path / "validations")
    validations.symlink_to(tmp_path / "validations")
    assert problems(evidence, datasets) == [
        f"datasets/validations/{release.name}: no such release",
        f"datasets/splits/{release.name}: no such release",
    ]


# Releases not yet versioned must be what their stored evidence builds (re-review at 6d46efa)


def unchanged(release: Path) -> dict:
    """DVC status reporting release as versioned and unchanged (--unchanged)."""
    return {"unchanged": [str(release / "steps.parquet"), str(release / "manifest.json")]}


def edited_copy(release: Path, datasets: Path, edit) -> Path:
    """A release of edited rows, with every hash, digest, and name re-derived to agree."""
    rows = pq.read_table(release / "steps.parquet").to_pylist()
    manifest = json.loads((release / "manifest.json").read_text())
    return consistent_release_from(edit(rows), datasets, manifest["label_version"])


def test_a_consistent_release_with_edited_values_is_refused(stores):
    """The review's reproduction: one queue_seconds moved by 0.125, authentic evidence hash."""
    evidence, datasets, release = stores

    def nudge(rows):
        rows[0]["queue_seconds"] += 0.125
        return rows

    forged = edited_copy(release, datasets, nudge)
    report = verify(evidence, datasets)
    assert report.problems == [
        (
            f"datasets/{forged.name}: its rows are not what its stored evidence builds; "
            "it was edited or built elsewhere"
        )
    ]
    assert report.rebuilt == 2


def test_a_consistent_release_missing_a_row_is_refused(stores):
    evidence, datasets, release = stores
    forged = edited_copy(release, datasets, lambda rows: rows[1:])
    assert problems(evidence, datasets) == [
        (
            f"datasets/{forged.name}: its rows are not what its stored evidence builds; "
            "it was edited or built elsewhere"
        )
    ]


def test_a_re_encoded_release_is_refused(stores):
    """Same rows, other Parquet bytes: not what a rebuild writes."""
    evidence, datasets, release = stores
    table = pq.read_table(release / "steps.parquet")
    pq.write_table(table, release / "steps.parquet", compression="none")
    resign(release)
    assert problems(evidence, datasets) == [
        f"datasets/{release.name}: its steps.parquet is not the bytes its rows build"
    ]
    assert problems(evidence, datasets, unchanged(release)) == []


def test_superseded_releases_are_rebuilt_by_their_rows(stores):
    evidence, datasets, release = stores
    old = format_1(release)
    pq.write_table(pq.read_table(old / "steps.parquet"), old / "steps.parquet", compression="none")
    resign(old)  # format 1 recorded no encoding, so only its rows must rebuild
    report = verify(evidence, datasets)
    assert report.problems == [] and report.rebuilt == 2


def test_releases_of_other_label_rules_must_already_be_versioned(stores, monkeypatch):
    import dataclasses

    from agentwatch_data import integrity

    evidence, datasets, release = stores
    retuned = dataclasses.replace(integrity.load_config(), version="labels-v9")
    monkeypatch.setattr(integrity, "load_config", lambda: retuned)
    [problem] = problems(evidence, datasets)
    assert problem.startswith(f"datasets/{release.name}: it was labeled under labels-v1.1+")
    assert problem.endswith(
        f"but this pipeline labels under {retuned.label_version}, so it cannot be rebuilt "
        "from its evidence; build it again"
    )
    report = verify(evidence, datasets, unchanged(release))
    assert report.problems == [] and report.rebuilt == 0


def test_evidence_that_no_longer_builds_is_reported(stores, monkeypatch):
    from agentwatch_data import integrity
    from agentwatch_data.extract import Rejection

    evidence, datasets, release = stores
    rejection = Rejection("run x", "baseline missing")
    monkeypatch.setattr(integrity, "derive_rows", lambda runs, rules: ([], {"label": [rejection]}))
    assert problems(evidence, datasets) == [
        f"datasets/{release.name}: its evidence no longer builds (label: baseline missing)"
    ]


@pytest.mark.parametrize("listed", [None, "steps.parquet", [1, None]])
def test_a_malformed_unchanged_list_versions_nothing(stores, listed):
    evidence, datasets, _ = stores
    report = verify(evidence, datasets, {"unchanged": listed})
    assert report.problems == [] and report.rebuilt == 1


def test_entries_that_are_not_paths_are_ignored(stores):
    evidence, datasets, release = stores
    status = {"unchanged": [1, None, *unchanged(release)["unchanged"]]}
    assert verify(evidence, datasets, status).rebuilt == 0


def test_a_partly_versioned_release_is_rebuilt(stores):
    evidence, datasets, release = stores
    status = {"unchanged": [str(release / "manifest.json")]}
    assert verify(evidence, datasets, status).rebuilt == 1
