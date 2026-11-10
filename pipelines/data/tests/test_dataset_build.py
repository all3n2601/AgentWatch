import hashlib
import json
import shutil
import stat
import urllib.error
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from agentwatch_data import build as build_module
from agentwatch_data import extract
from agentwatch_data.build import (
    ARROW_SCHEMA,
    DATASET_FILE,
    MANIFEST_FILE,
    MANIFEST_WITHOUT_STATISTICS,
    RELEASE_FORMAT,
    SCHEMA_FILE,
    BuildError,
    Redactor,
    Source,
    build,
    derive_rows,
    experiment_directories,
    main,
    published_releases,
    release_digest,
    release_label_version,
    release_manifest,
)
from agentwatch_data.evidence import add
from agentwatch_data.labels import load_config
from agentwatch_data.releases import main as releases_main
from agentwatch_data.replay_labels import load_replay_config
from agentwatch_data.schema import COLUMNS, SCHEMA_VERSION, contract
from dataset_fixtures import make_result, write_run
from replay_fixtures import write_experiment

FIXTURES = Path(__file__).parent / "fixtures"
NOW = datetime(2026, 10, 11, tzinfo=UTC)


@pytest.fixture
def evidence(tmp_path):
    """A copy of the recorded run, so tests can add bad runs beside it."""
    directory = tmp_path / "evidence"
    shutil.copytree(FIXTURES, directory)
    return directory


def result_from(directory, output, **kwargs):
    return build([Source("local", str(directory))], output, now=NOW, **kwargs)


def build_from(directory, output, **kwargs):
    return result_from(directory, output, **kwargs).release


def report_of(result):
    return json.loads(result.report.read_text())


def manifest_of(release):
    return json.loads((release / MANIFEST_FILE).read_text())


def leftover_staging(output):
    return [path for path in output.iterdir() if path.name.startswith(".building-")]


def releases_in(output):
    return [path for path in output.iterdir() if path.name.startswith(f"{SCHEMA_VERSION}+")]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_format_1_release(release, output):
    """Publish release as the builder wrote it before schema.json (release format 1).

    The version came from the rows alone, and the manifest had no format or contract hash.
    """
    manifest = manifest_of(release)
    name = f"{SCHEMA_VERSION}+{manifest['rows_sha256'][:12]}"
    legacy = output / name
    legacy.mkdir(parents=True)
    table = pq.read_table(release / DATASET_FILE)
    index = table.column_names.index("dataset_version")
    versions = pa.array([name] * table.num_rows, pa.string())
    pq.write_table(
        table.set_column(index, table.schema.field(index), versions), legacy / DATASET_FILE
    )
    del manifest["release_format"], manifest["schema_sha256"]
    manifest |= {"dataset_version": name, "parquet_sha256": sha256(legacy / DATASET_FILE)}
    (legacy / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    report = {"build_id": "format-1", "built_at": "2026-10-01T00:00:00+00:00", "release": name}
    (output / "builds").mkdir()
    (output / "builds" / "20261001T000000000000Z-format-1.json").write_text(json.dumps(report))
    return legacy


def nothing_published(output):
    """No release, no staging or withdrawn directory, and no report or staged report."""
    if not output.exists():
        return True
    leftovers = [p for p in output.iterdir() if p.name != "builds"]
    reports = list((output / "builds").iterdir()) if (output / "builds").exists() else []
    return leftovers == [] and reports == []


def test_release_holds_the_dataset_and_manifest(evidence, tmp_path):
    release = build_from(evidence, tmp_path / "out")
    assert release.name.startswith(f"{SCHEMA_VERSION}+")
    table = pq.read_table(release / DATASET_FILE)
    assert table.schema.equals(ARROW_SCHEMA)
    assert table.num_rows == 7
    assert set(table.column("dataset_version").to_pylist()) == {release.name}
    assert table.column("condition").to_pylist() == ["baseline"] * 5 + ["clean", "contended"]
    manifest = manifest_of(release)
    assert manifest["dataset_version"] == release.name
    assert (manifest["rows"], manifest["runs"], manifest["tasks"]) == (7, 1, 1)
    assert manifest["label_resource"] == {"cpu_sandbox": 1, "none": 6}
    assert manifest["telemetry_status"] == {"present": 7}
    assert not {"rejections", "sources", "duplicate_copies"} & manifest.keys()  # content only
    assert manifest["label_version"].startswith("labels-v1.1+")
    shipped = json.loads((release / "schema.json").read_text())
    assert shipped == contract() and shipped["schema_version"] == SCHEMA_VERSION
    assert manifest["schema_sha256"] == sha256(release / "schema.json")
    assert manifest["release_format"] == RELEASE_FORMAT


def test_release_files_are_readable_by_other_users(evidence, tmp_path):
    release = build_from(evidence, tmp_path / "out")
    assert stat.S_IMODE(release.stat().st_mode) == 0o755
    assert leftover_staging(tmp_path / "out") == []


def test_rebuilding_the_same_evidence_reproduces_the_release(evidence, tmp_path):
    first = build_from(evidence, tmp_path / "a")
    second = build_from(evidence, tmp_path / "b")
    assert first.name == second.name
    for name in (DATASET_FILE, MANIFEST_FILE):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_building_an_existing_release_verifies_it(evidence, tmp_path):
    output = tmp_path / "out"
    release = build_from(evidence, output)
    before = (release / DATASET_FILE).stat().st_mtime_ns
    assert build_from(evidence, output) == release
    assert (release / DATASET_FILE).stat().st_mtime_ns == before


def test_a_tampered_release_is_refused(evidence, tmp_path):
    output = tmp_path / "out"
    release = build_from(evidence, output)
    (release / DATASET_FILE).write_bytes(b"tampered")
    with pytest.raises(BuildError, match="does not match its manifest"):
        build_from(evidence, output)


def test_a_tampered_contract_is_refused_on_reuse(evidence, tmp_path):
    output = tmp_path / "out"
    release = build_from(evidence, output)
    (release / SCHEMA_FILE).write_text("{}\n")
    with pytest.raises(BuildError, match="does not match its manifest"):
        build_from(evidence, output)


def test_a_changed_contract_produces_a_new_version(evidence, tmp_path, monkeypatch):
    first = build_from(evidence, tmp_path / "out")
    changed = contract() | {"columns": contract()["columns"][::-1]}
    monkeypatch.setattr(build_module, "contract", lambda: changed)
    second = build_from(evidence, tmp_path / "out")
    assert second != first
    assert json.loads((second / SCHEMA_FILE).read_text()) == changed
    assert manifest_of(first)["rows_sha256"] == manifest_of(second)["rows_sha256"]


def test_a_format_1_release_is_superseded_not_reused(evidence, tmp_path, capsys):
    """A release from before schema.json is kept as it was, and a rebuild publishes a new one."""
    current = build_from(evidence, tmp_path / "current")
    output = tmp_path / "out"
    legacy = write_format_1_release(current, output)
    before = {path.name: path.read_bytes() for path in legacy.iterdir()}
    result = result_from(evidence, output)
    assert not result.reused and result.release != legacy
    assert {path.name: path.read_bytes() for path in legacy.iterdir()} == before
    assert (result.release / DATASET_FILE).read_bytes() == (current / DATASET_FILE).read_bytes()
    manifest = manifest_of(result.release)
    assert manifest["schema_sha256"] == sha256(result.release / SCHEMA_FILE)
    assert published_releases(output) == [legacy, result.release]
    assert releases_main(["--output", str(output)]) == 0
    assert capsys.readouterr().out.strip() == str(result.release)
    assert result_from(evidence, output).reused  # and the upgraded release is reused from now on


@pytest.mark.parametrize("damage", ["release_format", "schema_sha256"])
def test_an_existing_release_with_incomplete_metadata_is_refused(evidence, tmp_path, damage):
    output = tmp_path / "out"
    release = build_from(evidence, output)
    manifest = manifest_of(release)
    del manifest[damage]
    (release / MANIFEST_FILE).write_text(json.dumps(manifest))
    with pytest.raises(BuildError, match="is unreadable"):
        build_from(evidence, output)


def test_an_existing_release_with_a_non_object_manifest_is_refused(evidence, tmp_path):
    output = tmp_path / "out"
    release = build_from(evidence, output)
    (release / MANIFEST_FILE).write_text("[1, 2]")
    with pytest.raises(BuildError, match="is unreadable"):
        build_from(evidence, output)


def test_new_evidence_produces_a_new_version(evidence, tmp_path):
    first = build_from(evidence, tmp_path / "out")
    write_run(evidence / "cpu-extra", make_result())
    second = build_from(evidence, tmp_path / "out")
    assert second != first
    assert manifest_of(second)["runs"] == 2


def test_rejected_runs_are_reported_with_redacted_origins(evidence, tmp_path):
    bad = make_result()
    bad["contended"]["step_id"] = "baseline-9"
    write_run(evidence / "cpu-renamed", bad)
    invalid = make_result()
    invalid["clean"]["queue_seconds"] = 5.0
    write_run(evidence / "cpu-invalid", invalid)
    result = result_from(evidence, tmp_path / "out")
    report = report_of(result)
    stages = sorted(item["stage"] for item in report["rejections"])
    assert stages == ["extract", "flatten"]
    text = result.report.read_text()
    assert str(tmp_path) not in text and str(Path.home()) not in text
    assert "<local:evidence>/cpu-invalid" in text
    assert manifest_of(result.release)["rows"] == 7  # the good run still builds


def test_unreadable_source_writes_nothing(tmp_path):
    output = tmp_path / "out"
    with pytest.raises(BuildError, match="could not be read"):
        build_from(tmp_path / "absent", output)
    assert nothing_published(output)


def test_api_failure_writes_nothing(evidence, tmp_path, monkeypatch):
    def unreachable(url, timeout):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(extract, "get_json", unreachable)
    monkeypatch.setattr(extract, "DEFAULT_RETRY", extract.RetryPolicy(sleep=lambda _: None))
    sources = [Source("local", str(evidence)), Source("api", "http://127.0.0.1:9")]
    output = tmp_path / "out"
    with pytest.raises(BuildError, match="api:http://127.0.0.1:9 could not be read"):
        build(sources, output, now=NOW)
    assert nothing_published(output)


def test_no_usable_rows_fails_the_build(tmp_path):
    only_bad = tmp_path / "evidence"
    bad = make_result()
    bad["checks"] = {"same_workload_output": False, "passed": False}
    write_run(only_bad / "cpu-bad", bad)
    with pytest.raises(BuildError, match="No usable rows"):
        build_from(only_bad, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_invalid_rules_fail_the_build(evidence, tmp_path):
    rules = tmp_path / "labels.toml"
    rules.write_text('version = "x"\n')
    with pytest.raises(BuildError, match="Label rules are invalid"):
        build_from(evidence, tmp_path / "out", rules=rules)


def test_no_usable_rows_after_schema_rejections_fails_the_build(evidence, tmp_path, monkeypatch):
    real = build_module.label_rows

    def broken(rows, config):
        labeled, rejections = real(rows, config)
        return [row | {"queue_fraction": 2.0} for row in labeled], rejections

    monkeypatch.setattr(build_module, "label_rows", broken)
    with pytest.raises(BuildError, match="No usable rows"):
        build_from(evidence, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda result: result["contended"].update(rule_version=""),  # contract allows ""
        lambda result: result["experiment"].update(workload=" "),
    ],
    ids=["empty-rule-version", "blank-workload"],
)
def test_a_run_the_contract_accepts_but_the_schema_does_not_is_rejected_alone(
    evidence, tmp_path, corrupt
):
    """One such run used to stop every build until someone deleted it by hand."""
    bad = make_result()
    corrupt(bad)
    write_run(evidence / "cpu-loose", bad)
    result = result_from(evidence, tmp_path / "out")
    [rejection] = [item for item in report_of(result)["rejections"] if item["stage"] == "schema"]
    assert (
        rejection["origin"].endswith("cpu-loose")
        and "must be a nonempty string" in rejection["reason"]
    )
    assert manifest_of(result.release)["runs"] == 1  # the recorded run still builds


def test_psi_above_100_is_rejected_alone(evidence, tmp_path):
    result = make_result()
    window = result["clean"]["submitted_ns"] + 1
    samples = [
        {
            "timestamp_ns": window,
            "cpu_pressure": {"some": {"avg10": 150.0, "avg60": 0.0, "avg300": 0.0, "total": 1}},
            "cgroup_cpu_stat": None,
            "unavailable": [],
        }
    ]
    write_run(evidence / "cpu-psi", result, samples)
    built = result_from(evidence, tmp_path / "out")
    reasons = [item["reason"] for item in report_of(built)["rejections"]]
    assert any("cpu_pressure_max_avg10 must be at most 100" in reason for reason in reasons)


def test_a_builder_defect_still_fails_the_whole_build(evidence, tmp_path, monkeypatch):
    real = build_module.validate_rows

    def fail_release_check(rows, expected=None):
        if expected == build_module.ALL_COLUMNS:
            raise build_module.SchemaError("dataset_version malformed")
        return real(rows, expected)

    monkeypatch.setattr(build_module, "validate_rows", fail_release_check)
    with pytest.raises(BuildError, match="Release rows violate the dataset schema"):
        build_from(evidence, tmp_path / "out")


def test_a_failed_write_leaves_no_partial_release(evidence, tmp_path, monkeypatch):
    def disk_full(*args, **kwargs):
        raise OSError("No space left on device")

    monkeypatch.setattr(build_module.pq, "write_table", disk_full)
    output = tmp_path / "out"
    with pytest.raises(OSError, match="No space left"):
        build_from(evidence, output)
    assert nothing_published(output)


def test_at_least_one_source_is_required(tmp_path):
    with pytest.raises(BuildError, match="At least one source"):
        build([], tmp_path / "out")


@pytest.mark.parametrize("text", ["local:", "s3:bucket", "data", "api: "])
def test_source_specs_are_validated(text):
    with pytest.raises(ValueError, match="local:<directory> or api:<url>"):
        Source.parse(text)


def test_source_specs_round_trip():
    assert str(Source.parse("api:http://127.0.0.1:8090")) == "api:http://127.0.0.1:8090"


def test_cli_prints_the_release_and_exits_zero(evidence, tmp_path, capsys):
    code = main(["--source", f"local:{evidence}", "--output", str(tmp_path / "out")])
    assert code == 0
    printed = dict(line.split(": ", 1) for line in capsys.readouterr().out.strip().splitlines())
    release, report = Path(printed["release"]), Path(printed["report"])
    assert release.parent == tmp_path / "out" and (release / DATASET_FILE).exists()
    assert report.parent == tmp_path / "out" / "builds" and report.exists()


@pytest.mark.parametrize(
    "args", [["--source", "bogus"], ["--source", "local:/definitely/absent/agentwatch"]]
)
def test_cli_reports_failures_with_exit_one(args, tmp_path, capsys):
    assert main([*args, "--output", str(tmp_path / "out")]) == 1
    assert "Dataset build failed" in capsys.readouterr().err


def test_sources_with_the_same_name_get_distinct_labels(tmp_path):
    for parent in ("a", "b"):
        write_run(tmp_path / parent / "runs" / "cpu-1", make_result())
    sources = [
        Source("local", str(tmp_path / "a" / "runs")),
        Source("local", str(tmp_path / "b" / "runs")),
    ]
    result = build(sources, tmp_path / "out", now=NOW)
    labels = [item["source"] for item in report_of(result)["sources"]]
    assert labels == ["<local:runs#1>", "<local:runs#2>"]
    assert str(tmp_path) not in result.report.read_text()


def test_relative_local_sources_are_redacted(evidence, tmp_path, monkeypatch):
    bad = make_result()
    bad["clean"]["queue_seconds"] = 5.0
    write_run(evidence / "cpu-invalid", bad)
    monkeypatch.chdir(evidence.parent)
    result = build([Source("local", "evidence")], tmp_path / "out", now=NOW)
    text = result.report.read_text()
    assert str(tmp_path) not in text and "<local:evidence>/cpu-invalid" in text


def test_a_concurrent_build_that_finishes_first_is_verified(evidence, tmp_path, monkeypatch):
    real_replace = build_module.os.replace

    def other_builder_won(source, target):
        if not Path(source).is_dir():  # the build report's own atomic rename
            return real_replace(source, target)
        shutil.copytree(source, target)  # the other build's identical release appears first
        raise OSError("Directory not empty")

    monkeypatch.setattr(build_module.os, "replace", other_builder_won)
    output = tmp_path / "out"
    result = result_from(evidence, output)
    assert result.reused and (result.release / DATASET_FILE).exists()
    assert leftover_staging(output) == []
    monkeypatch.setattr(build_module.os, "replace", real_replace)


def test_a_failed_rename_without_a_competing_release_is_raised(evidence, tmp_path, monkeypatch):
    def rename_fails(source, target):
        raise OSError("Read-only file system")

    monkeypatch.setattr(build_module.os, "replace", rename_fails)
    with pytest.raises(OSError, match="Read-only"):
        build_from(evidence, tmp_path / "out")
    assert nothing_published(tmp_path / "out")


def test_an_existing_release_without_a_manifest_is_refused(evidence, tmp_path):
    output = tmp_path / "out"
    release = build_from(evidence, output)
    (release / MANIFEST_FILE).unlink()
    with pytest.raises(BuildError, match="is unreadable"):
        build_from(evidence, output)


def test_module_runs_as_a_command(evidence, tmp_path, monkeypatch, capsys):
    import runpy
    import sys

    argv = ["agentwatch_data.build", "--source", f"local:{evidence}", "--output", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.delitem(sys.modules, "agentwatch_data.build")  # run a fresh copy, as -m does
    with pytest.raises(SystemExit) as exited:
        runpy.run_module("agentwatch_data.build", run_name="__main__")
    assert exited.value.code == 0


# Per-build reports (Allen's #14 review)


def test_every_build_writes_its_own_report(evidence, tmp_path):
    first = result_from(evidence, tmp_path / "out")
    second = result_from(evidence, tmp_path / "out")
    assert first.release == second.release and first.report != second.report
    assert (first.reused, second.reused) == (False, True)
    reports = sorted((tmp_path / "out" / "builds").iterdir())
    assert len(reports) == 2
    assert stat.S_IMODE(second.report.stat().st_mode) == 0o644
    report = report_of(second)
    assert report["dataset_version"] == second.release.name and report["reused_release"]
    assert (
        report_of(first)["release"] == first.release.name and not report_of(first)["reused_release"]
    )


def test_rebuild_after_new_rejected_evidence_records_the_rejection(evidence, tmp_path):
    """Allen's repro: the accepted rows are unchanged, so the release is reused, but the
    newly rejected run must still be reported."""
    first = result_from(evidence, tmp_path / "out")
    invalid = make_result()
    invalid["clean"]["queue_seconds"] = 5.0
    write_run(evidence / "cpu-invalid", invalid)
    second = result_from(evidence, tmp_path / "out")
    assert second.release == first.release and second.reused
    report = report_of(second)
    assert [item["stage"] for item in report["rejections"]] == ["extract"]
    assert "Duration does not match" in report["rejections"][0]["reason"]
    assert report["sources"] == [{"source": "<local:evidence>", "runs": 1, "rejections": 1}]
    assert report_of(first)["rejections"] == []


def test_unchanged_data_from_changed_sources_reports_the_new_sources(evidence, tmp_path):
    first = result_from(evidence, tmp_path / "out")
    mirror = tmp_path / "mirror"
    shutil.copytree(evidence, mirror)
    sources = [Source("local", str(evidence)), Source("local", str(mirror))]
    second = build(sources, tmp_path / "out", now=NOW)
    assert second.release == first.release and second.reused
    report = report_of(second)
    assert [item["source"] for item in report["sources"]] == ["<local:evidence>", "<local:mirror>"]
    assert report["duplicate_copies"] == 1 and report["rejections"] == []


# Publication is the report commit; builds never delete releases (Allen's #14 reviews)


def fail_report_commit(monkeypatch, message="No space left on device"):
    real_replace = build_module.os.replace

    def replace(source, target):
        if Path(target).suffix == ".json":
            raise OSError(message)
        return real_replace(source, target)

    monkeypatch.setattr(build_module.os, "replace", replace)


def test_failed_report_commit_leaves_an_unpublished_release(evidence, tmp_path, monkeypatch):
    output = tmp_path / "out"
    fail_report_commit(monkeypatch)
    with pytest.raises(OSError, match="No space left"):
        build_from(evidence, output)
    [release] = releases_in(output)
    assert published_releases(output) == []  # not published: no report names it
    assert list((output / "builds").iterdir()) == []  # no report, no staged report
    monkeypatch.undo()
    retry = result_from(evidence, output)  # recovery: verify and publish it
    assert retry.reused and retry.release == release
    assert published_releases(output) == [release]


def test_failed_report_staging_publishes_nothing(evidence, tmp_path, monkeypatch):
    real_write = Path.write_text

    def write_text(self, data, *args, **kwargs):
        if self.name.endswith(".json.tmp"):
            raise OSError("Read-only file system")
        return real_write(self, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", write_text)
    output = tmp_path / "out"
    with pytest.raises(OSError, match="Read-only"):
        build_from(evidence, output)
    assert nothing_published(output)


def test_failed_report_on_reuse_keeps_the_existing_release(evidence, tmp_path, monkeypatch):
    output = tmp_path / "out"
    first = result_from(evidence, output)
    fail_report_commit(monkeypatch)
    with pytest.raises(OSError):
        build_from(evidence, output)
    assert sorted((output / "builds").iterdir()) == [first.report]
    monkeypatch.undo()
    assert result_from(evidence, output).reused
    assert published_releases(output) == [first.release]


def test_a_concurrent_successful_build_keeps_its_dataset(evidence, tmp_path, monkeypatch):
    """Allen's interleaving: A publishes the release and its report commit fails, while B
    verifies that release and commits its report in between. B's dataset must survive."""
    output = tmp_path / "out"
    real_replace = build_module.os.replace
    outcome = {}

    def replace(source, target):
        if Path(target).suffix == ".json" and "b" not in outcome:
            monkeypatch.setattr(build_module.os, "replace", real_replace)
            outcome["b"] = result_from(evidence, output)  # B runs to completion here
            raise OSError("No space left on device")  # then A's report commit fails
        return real_replace(source, target)

    monkeypatch.setattr(build_module.os, "replace", replace)
    with pytest.raises(OSError, match="No space left"):
        build_from(evidence, output)  # builder A
    b = outcome["b"]
    assert b.reused and b.report.exists()
    assert (b.release / DATASET_FILE).exists() and (b.release / MANIFEST_FILE).exists()
    assert published_releases(output) == [b.release]


def test_failed_report_after_a_concurrent_release_keeps_it(evidence, tmp_path, monkeypatch):
    def replace(source, target):
        if Path(source).is_dir():
            shutil.copytree(source, target)  # the other build's identical release appears first
            raise OSError("Directory not empty")
        raise OSError("No space left on device")  # and then our report commit fails

    monkeypatch.setattr(build_module.os, "replace", replace)
    output = tmp_path / "out"
    with pytest.raises(OSError, match="No space left"):
        build_from(evidence, output)
    [release] = releases_in(output)
    assert (release / MANIFEST_FILE).exists()
    assert list((output / "builds").iterdir()) == []


def test_published_releases_follow_valid_reports_only(evidence, tmp_path):
    output = tmp_path / "out"
    first = result_from(evidence, output)
    write_run(evidence / "cpu-extra", make_result())
    later = datetime(2026, 10, 12, tzinfo=UTC)
    second = build([Source("local", str(evidence))], output, now=later)
    reuse = build([Source("local", str(evidence))], output, now=datetime(2026, 10, 13, tzinfo=UTC))
    assert reuse.reused and reuse.release == second.release
    builds = output / "builds"
    (builds / "19990101T000000000000Z-broken.json").write_text("{not json")
    (builds / "19990101T000000000001Z-list.json").write_text("[1, 2]")
    (builds / "19990101T000000000002Z-gone.json").write_text('{"release": "steps-v1.2+gone"}')
    (builds / "19990101T000000000003Z-escape.json").write_text('{"release": "../evidence"}')
    (builds / "19990101T000000000004Z-number.json").write_text('{"release": 7}')
    assert published_releases(output) == [first.release, second.release]
    assert published_releases(output) == published_releases(output)  # stable
    assert published_releases(tmp_path / "absent") == []


def test_latest_published_release_is_printed(evidence, tmp_path, capsys):
    output = tmp_path / "out"
    assert releases_main(["--output", str(output)]) == 1
    assert "No published release" in capsys.readouterr().err
    first = result_from(evidence, output)
    write_run(evidence / "cpu-extra", make_result())
    second = build([Source("local", str(evidence))], output, now=datetime(2026, 10, 12, tzinfo=UTC))
    assert releases_main(["--output", str(output)]) == 0
    assert capsys.readouterr().out.strip() == str(second.release)
    assert first.release != second.release


def test_releases_module_runs_as_a_command(evidence, tmp_path, monkeypatch):
    import runpy
    import sys

    result_from(evidence, tmp_path / "out")
    monkeypatch.setattr(
        sys, "argv", ["agentwatch_data.releases", "--output", str(tmp_path / "out")]
    )
    monkeypatch.delitem(sys.modules, "agentwatch_data.releases", raising=False)
    with pytest.raises(SystemExit) as exited:
        runpy.run_module("agentwatch_data.releases", run_name="__main__")
    assert exited.value.code == 0


# Audit fixes


def test_the_newest_published_release_is_the_one_last_built(evidence, tmp_path, capsys):
    """Rebuilding older evidence must make that release the one later steps act on."""
    output = tmp_path / "out"
    first = result_from(evidence, output)
    write_run(evidence / "cpu-extra", make_result())
    second = build([Source("local", str(evidence))], output, now=datetime(2026, 10, 12, tzinfo=UTC))
    shutil.rmtree(evidence / "cpu-extra")
    again = build([Source("local", str(evidence))], output, now=datetime(2026, 10, 13, tzinfo=UTC))
    assert again.reused and again.release == first.release
    assert published_releases(output) == [second.release, first.release]
    assert releases_main(["--output", str(output)]) == 0
    assert capsys.readouterr().out.strip() == str(first.release)


def test_reports_naming_anything_but_a_real_release_publish_nothing(evidence, tmp_path):
    output = tmp_path / "out"
    real = result_from(evidence, output).release
    builds = output / "builds"
    lookalike = output / f"{SCHEMA_VERSION}+{'a' * 12}"
    lookalike.mkdir()
    (lookalike / MANIFEST_FILE).write_text("{}")  # its manifest names no version
    linked = output / f"{SCHEMA_VERSION}+{'b' * 12}"
    linked.symlink_to(real, target_is_directory=True)
    bare = output / f"{SCHEMA_VERSION}+{'c' * 12}"
    bare.mkdir()  # no manifest at all
    names = ["x" * 300, lookalike.name, linked.name, bare.name, "steps-v1.2+ABCDEF"]
    for index, name in enumerate(names):
        report = {"release": name, "built_at": "2099-01-01T00:00:00+00:00"}
        (builds / f"2099010{index}T000000000000Z-bad.json").write_text(json.dumps(report))
    assert published_releases(output) == [real]  # and the 300-character name does not crash


def test_source_order_does_not_change_the_version(tmp_path):
    result = make_result()
    result["baseline"]["execution_cv"] = 0.0
    loose = json.loads(json.dumps(result))
    loose["baseline"]["execution_cv"] = 0  # same evidence, different raw bytes
    write_run(tmp_path / "x" / "cpu-1", result)
    write_run(tmp_path / "y" / "cpu-1", loose)
    sources = [Source("local", str(tmp_path / "x")), Source("local", str(tmp_path / "y"))]
    forward = build(sources, tmp_path / "a", now=NOW).release
    backward = build(sources[::-1], tmp_path / "b", now=NOW).release
    assert forward.name == backward.name


def test_the_recorded_time_zone_does_not_change_the_version():
    from datetime import timedelta, timezone

    row = {"recorded_at": datetime(2026, 10, 5, 12, tzinfo=UTC), "run_id": "r"}
    eastern = row | {"recorded_at": row["recorded_at"].astimezone(timezone(timedelta(hours=-4)))}
    assert build_module.rows_digest([row]) == build_module.rows_digest([eastern])


@pytest.mark.parametrize("home", [None, "/"])
def test_redaction_survives_a_container_without_a_usable_home(tmp_path, monkeypatch, home):
    def no_home():
        raise RuntimeError("Could not determine home directory.")

    monkeypatch.setattr(build_module.Path, "home", no_home if home is None else lambda: Path(home))
    redact = build_module.Redactor([Source("local", str(tmp_path / "ev"))])
    assert redact("/var/data/cpu-1 failed") == "/var/data/cpu-1 failed"  # no stray "~"


def test_parquet_is_written_with_declared_options(evidence, tmp_path):
    release = build_from(evidence, tmp_path / "out")
    metadata = pq.ParquetFile(release / DATASET_FILE).metadata
    assert metadata.row_group(0).column(0).compression == "SNAPPY"
    assert metadata.format_version == "2.6"


# steps-v1.3: replay experiments beside runs, and manifest statistics (D03). Every experiment
# here is synthetic (replay_fixtures); the run is the golden recorded fixture.

GOLDEN_RUN = FIXTURES / "coding-macos-2026-10-05"
REPLAY_KIND = "agent_replay"


@pytest.fixture
def store(tmp_path):
    """An evidence store holding the golden run and one synthetic live replay experiment."""
    directory = tmp_path / "store"
    add(GOLDEN_RUN, directory)
    add(write_experiment(tmp_path / "in", live=True), directory)
    return directory


def rows_of(release):
    return pq.read_table(release / DATASET_FILE).to_pylist()


def test_release_holds_run_rows_then_replay_rows(store, tmp_path):
    result = build([Source("local", str(store))], tmp_path / "out")
    rows = rows_of(result.release)
    kinds = [r["kind"] for r in rows]
    assert kinds.index(REPLAY_KIND) > max(i for i, k in enumerate(kinds) if k != REPLAY_KIND)
    manifest = manifest_of(result.release)
    assert manifest["label_version"].count(",") == 1
    assert set(manifest["label_versions"]) == {r["label_version"] for r in rows}
    assert manifest["missing"]["cpu_pressure_max_avg10"] == sum(
        r["cpu_pressure_max_avg10"] is None for r in rows
    )
    report = report_of(result)
    assert report["label_version"] == manifest["label_version"]
    assert report["experiments"] == 1 and report["rejections"] == []


def test_manifest_statistics_describe_every_row_kind(store, tmp_path):
    release = build_from(store, tmp_path / "out")
    rows, manifest = rows_of(release), manifest_of(release)
    replay = [row for row in rows if row["kind"] == REPLAY_KIND]
    assert manifest["kind"] == {"coding": 7, REPLAY_KIND: len(replay)}
    assert manifest["feature_version"] == {
        version: sum(row["feature_version"] == version for row in rows)
        for version in {row["feature_version"] for row in rows}
    }
    assert sum(manifest["label_versions"].values()) == len(rows)
    assert set(manifest["missing"]) == {column.name for column in COLUMNS}
    assert manifest["missing"]["dataset_version"] == 0  # zeros are kept, not omitted
    assert manifest["missing"]["experiment"] == 7  # run rows have no experiment
    assert manifest["missing"]["rule_prediction"] == len(replay)


def test_label_resources_are_counted_per_resource_with_empty_as_none(evidence, tmp_path):
    release = build_from(evidence, tmp_path / "out")
    rows = rows_of(release)
    assert [row["label_resources"] for row in rows] == [""] * 6 + ["cpu_sandbox"]
    rows[0] = rows[0] | {"label_resource": "mixed", "label_resources": "cpu_sandbox,inference"}
    manifest = release_manifest(rows, release.name, "digest", "labels", "p", "s")
    assert manifest["label_resources"] == {"cpu_sandbox": 2, "inference": 1, "none": 5}


def test_rebuilding_into_empty_stores_reproduces_version_and_bytes(store, tmp_path):
    first = build_from(store, tmp_path / "a")
    second = build_from(store, tmp_path / "b")
    assert first.name == second.name
    for name in (DATASET_FILE, SCHEMA_FILE, MANIFEST_FILE):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_failed_experiment_is_reported_not_dropped(evidence, tmp_path):
    checks = {
        "baseline_replays_complete": True,
        "stable_baseline": False,
        "same_steps_every_replay": True,
        "injectors_cleaned_up": True,
    }
    write_experiment(evidence, name="failed-exp", live=True, checks=checks)
    result = result_from(evidence, tmp_path / "out")
    rejections = report_of(result)["rejections"]
    assert [(r["stage"], r["origin"]) for r in rejections] == [
        ("replay_label", "experiment failed-exp")
    ]
    assert rejections[0]["reason"].startswith("protocol_checks_failed")
    assert {row["kind"] for row in rows_of(result.release)} == {"coding"}  # runs still build


def test_unreadable_experiment_is_a_replay_label_rejection(evidence, tmp_path):
    broken = write_experiment(evidence, name="broken-exp", live=True)
    (broken / "experiment.json").write_text("{not json")
    result = result_from(evidence, tmp_path / "out")
    [rejection] = report_of(result)["rejections"]
    assert rejection["stage"] == "replay_label"
    assert rejection["origin"] == "<local:evidence>/broken-exp"
    assert rejection["reason"]
    assert str(tmp_path) not in result.report.read_text()


def test_v1_2_manifest_shape_is_unchanged():
    rows = [{column.name: None for column in COLUMNS} | {"run_id": "r", "task_id": "t"}]
    manifest = release_manifest(rows, "v", "d", "l", "p", "s", schema_version="steps-v1.2")
    assert "steps-v1.2" in MANIFEST_WITHOUT_STATISTICS
    assert set(manifest) == {
        "release_format",
        "dataset_version",
        "schema_version",
        "label_version",
        "builder_version",
        "rows",
        "runs",
        "tasks",
        "rows_sha256",
        "parquet_sha256",
        "schema_sha256",
        "label_resource",
        "label_basis",
        "condition",
        "telemetry_status",
    }


def test_release_label_version_joins_the_distinct_versions_sorted():
    rows = [{"label_version": v} for v in ("labels-v2+b", "labels-v1.1+a", "labels-v2+b")]
    assert release_label_version(rows) == "labels-v1.1+a,labels-v2+b"


def test_replay_rows_follow_run_rows_ordered_by_experiment(tmp_path):
    later = write_experiment(tmp_path, name="b-exp", live=True, run_namespace=1)
    earlier = write_experiment(tmp_path, name="a-exp", live=True)
    runs = extract.extract_local(FIXTURES).runs
    rows, rejections = derive_rows(runs, load_config(), [later, earlier], load_replay_config())
    assert set(rejections) == {
        "flatten",
        "label",
        "schema",
        "replay_label",
        "replay_features",
        "replay_schema",
    }
    assert not any(rejections.values())
    replay = [row for row in rows if row["kind"] == REPLAY_KIND]
    assert rows[: len(rows) - len(replay)] == [r for r in rows if r["kind"] != REPLAY_KIND]
    keys = [(r["experiment"], r["replay_index"], r["step_id"]) for r in replay]
    assert keys == sorted(keys) and keys[0][0] == "a-exp" and keys[-1][0] == "b-exp"


def test_derive_rows_without_experiments_matches_runs_only(evidence):
    runs = extract.extract_local(evidence).runs
    rows, rejections = derive_rows(runs, load_config())
    assert {row["kind"] for row in rows} == {"coding"}
    assert not any(rejections.values())


def test_experiment_directories_are_found_sorted_skipping_hidden_and_links(tmp_path):
    source = tmp_path / "source"
    write_experiment(source / "nested", name="b-exp")
    write_experiment(source, name="a-exp")
    write_experiment(source / ".staging", name="hidden-exp")
    (source / "linked").symlink_to(write_experiment(tmp_path / "elsewhere", name="c-exp"))
    sources = [Source("local", str(source))]
    found, rejections, duplicates = experiment_directories(sources, Redactor(sources))
    assert [path.name for path in found] == ["a-exp", "b-exp"]
    assert (rejections, duplicates) == ([], 0)


def test_identical_experiment_copies_in_two_sources_are_kept_once(tmp_path):
    for parent in ("x", "y"):
        write_experiment(tmp_path / parent, name="same-exp")
    sources = [Source("local", str(tmp_path / "x")), Source("local", str(tmp_path / "y"))]
    found, rejections, duplicates = experiment_directories(sources, Redactor(sources))
    assert [path.name for path in found] == ["same-exp"]
    assert (rejections, duplicates) == ([], 1)


def test_conflicting_experiment_copies_are_all_rejected(tmp_path):
    write_experiment(tmp_path / "x", name="same-exp")
    write_experiment(tmp_path / "y", name="same-exp", task_id="fix-b-r1")
    sources = [Source("local", str(tmp_path / "x")), Source("local", str(tmp_path / "y"))]
    found, rejections, duplicates = experiment_directories(sources, Redactor(sources))
    assert (found, duplicates) == ([], 0)
    assert sorted(r["origin"] for r in rejections) == ["<local:x>/same-exp", "<local:y>/same-exp"]
    assert all("conflicting" in r["reason"] for r in rejections)
    assert str(tmp_path) not in json.dumps(rejections)


def test_identical_experiments_from_two_sources_build_one_copy(store, tmp_path):
    mirror = tmp_path / "mirror"
    shutil.copytree(store, mirror)
    sources = [Source("local", str(store)), Source("local", str(mirror))]
    result = build(sources, tmp_path / "out", now=NOW)
    single = build_from(store, tmp_path / "single")
    assert result.release.name == single.name
    report = report_of(result)
    assert report["duplicate_copies"] == 2 and report["rejections"] == []  # one run, one replay


def test_api_sources_have_no_experiments(tmp_path):
    sources = [Source("api", "http://127.0.0.1:1")]
    assert experiment_directories(sources, Redactor(sources)) == ([], [], 0)


def test_invalid_replay_rules_fail_the_build(evidence, tmp_path):
    rules = tmp_path / "bad.toml"
    rules.write_text("not = [valid")
    with pytest.raises(BuildError, match="Replay label rules"):
        result_from(evidence, tmp_path / "out", replay_rules=rules)
    assert not (tmp_path / "out").exists()


def test_cli_accepts_replay_rules(evidence, tmp_path, capsys):
    rules = tmp_path / "bad.toml"
    rules.write_text("not = [valid")
    args = ["--source", f"local:{evidence}", "--output", str(tmp_path / "out")]
    assert main([*args, "--replay-rules", str(rules)]) == 1
    assert "Replay label rules" in capsys.readouterr().err


def test_missingness_by_kind_includes_all_columns_and_measured_zeros():
    rows = [
        {column.name: None for column in COLUMNS}
        | {"kind": kind, "run_id": kind, "task_id": kind, "telemetry_sample_count": 0}
        for kind in ("cpu", "coding", "agent_replay")
    ]
    manifest = release_manifest(rows, "v", "d", "l", "p", "s")
    assert list(manifest["missing_by_kind"]) == ["agent_replay", "coding", "cpu"]
    for counts in manifest["missing_by_kind"].values():
        assert set(counts) == {column.name for column in COLUMNS}
        assert counts["telemetry_sample_count"] == 0
        assert counts["cpu_pressure_max_avg10"] == 1
    for column in COLUMNS:
        assert (
            sum(counts[column.name] for counts in manifest["missing_by_kind"].values())
            == manifest["missing"][column.name]
        )
    old = release_manifest(rows, "v", "d", "l", "p", "s", release_format=2)
    assert "missing_by_kind" not in old
    assert release_digest("d", release_format=2) != release_digest("d")
