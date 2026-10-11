import json
import shutil
import stat
import urllib.error
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from agentwatch_data import build as build_module
from agentwatch_data import extract
from agentwatch_data.build import (
    ARROW_SCHEMA,
    DATASET_FILE,
    MANIFEST_FILE,
    BuildError,
    Source,
    build,
    main,
)
from agentwatch_data.schema import SCHEMA_VERSION
from dataset_fixtures import make_result, write_run

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


def test_schema_violations_fail_the_build(evidence, tmp_path, monkeypatch):
    real = build_module.label_rows

    def broken(rows, config):
        labeled, rejections = real(rows, config)
        return [row | {"queue_fraction": 2.0} for row in labeled], rejections

    monkeypatch.setattr(build_module, "label_rows", broken)
    with pytest.raises(BuildError, match="violate the dataset schema"):
        build_from(evidence, tmp_path / "out")
    assert not (tmp_path / "out").exists()


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


# Report commit failures (Allen's second #14 review)


def fail_report_commit(monkeypatch, message="No space left on device"):
    real_replace = build_module.os.replace

    def replace(source, target):
        if Path(target).suffix == ".json":
            raise OSError(message)
        return real_replace(source, target)

    monkeypatch.setattr(build_module.os, "replace", replace)


def test_failed_report_commit_withdraws_the_new_release(evidence, tmp_path, monkeypatch):
    fail_report_commit(monkeypatch)
    output = tmp_path / "out"
    with pytest.raises(OSError, match="No space left"):
        build_from(evidence, output)
    assert nothing_published(output)


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
    assert first.release.exists()
    assert sorted((output / "builds").iterdir()) == [first.report]  # no new or staged report
    monkeypatch.undo()
    assert result_from(evidence, output).reused  # the kept release still verifies


def test_a_release_another_build_reported_is_never_withdrawn(evidence, tmp_path, monkeypatch):
    """If a concurrent build reused this release and committed its report, keep it."""
    output = tmp_path / "out"
    real_replace = build_module.os.replace

    def replace(source, target):
        if Path(target).suffix == ".json":
            other = Path(target).with_name("20991231T000000000000Z-other.json")
            other.write_text(json.dumps({"release": releases_in(output)[0].name}))
            raise OSError("No space left on device")
        return real_replace(source, target)

    monkeypatch.setattr(build_module.os, "replace", replace)
    with pytest.raises(OSError):
        build_from(evidence, output)
    [release] = releases_in(output)
    assert (release / DATASET_FILE).exists()


def test_unrelated_or_unreadable_reports_do_not_stop_a_withdrawal(evidence, tmp_path, monkeypatch):
    output = tmp_path / "out"
    (output / "builds").mkdir(parents=True)
    (output / "builds" / "20000101T000000000000Z-broken.json").write_text("{not json")
    (output / "builds" / "20000101T000000000001Z-list.json").write_text("[1, 2]")
    (output / "builds" / "20000101T000000000002Z-other.json").write_text(
        '{"release": "steps-v0+x"}'
    )
    fail_report_commit(monkeypatch)
    with pytest.raises(OSError):
        build_from(evidence, output)
    assert releases_in(output) == []


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


def test_an_unremovable_release_is_logged_not_hidden(evidence, tmp_path, monkeypatch, caplog):
    fail_report_commit(monkeypatch)

    def cannot_remove(path, ignore_errors=False):
        if not ignore_errors:
            raise PermissionError(f"Permission denied: {path}")

    monkeypatch.setattr(build_module.shutil, "rmtree", cannot_remove)
    with pytest.raises(OSError, match="No space left"), caplog.at_level("ERROR"):
        build_from(evidence, tmp_path / "out")
    assert "Could not withdraw unreported release" in caplog.text
