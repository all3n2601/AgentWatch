"""Quality gate tests on a real release: steps-v1.2 built from the recorded macOS coding run."""

import hashlib
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import tensorflow_data_validation as tfdv
from agentwatch_validation import gate
from agentwatch_validation.__main__ import main
from agentwatch_validation.gate import (
    FAILED,
    PASSED,
    RESULT_FILE,
    STATISTICS_FILE,
    VALIDATIONS_DIR,
    GateError,
    committed_attempts,
    freeze_schema,
    latest_result,
    load_release,
    previous_passing,
    validate,
    write_schema,
)

HERE = Path(__file__).parent
[FIXTURE] = sorted((HERE / "fixtures").iterdir())
FROZEN = HERE.parent / "schema" / "steps.pbtxt"
NOW = datetime(2026, 10, 12, tzinfo=UTC)


@pytest.fixture
def datasets(tmp_path):
    root = tmp_path / "datasets"
    shutil.copytree(FIXTURE, root / FIXTURE.name)
    return root


@pytest.fixture
def release(datasets):
    return datasets / FIXTURE.name


@pytest.fixture
def schema(release, tmp_path):
    path = tmp_path / "steps.pbtxt"
    loaded = load_release(release)
    write_schema(freeze_schema(loaded), path, loaded.contract["schema_version"])
    return path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resign(release: Path) -> None:
    """Update the manifest after deliberately changing the release, so integrity checks pass."""
    manifest = json.loads((release / "manifest.json").read_text())
    manifest["parquet_sha256"] = sha256(release / "steps.parquet")
    manifest["schema_sha256"] = sha256(release / "schema.json")
    (release / "manifest.json").write_text(json.dumps(manifest))


def change_column(release: Path, name: str, values, as_type=None) -> None:
    table = pq.read_table(release / "steps.parquet")
    field = table.schema.field(name)
    if as_type is not None:  # also relax nullability, so a null can reach the gate
        field = pa.field(name, as_type)
    array = pa.array(values, type=field.type)
    pq.write_table(
        table.set_column(table.column_names.index(name), field, array), release / "steps.parquet"
    )
    resign(release)


def retype(release: Path, name: str, values, arrow_type) -> None:
    """Store a column with a different physical type than the contract declares."""
    table = pq.read_table(release / "steps.parquet")
    index = table.column_names.index(name)
    array = pa.array(values, type=arrow_type)
    pq.write_table(
        table.set_column(index, pa.field(name, arrow_type), array), release / "steps.parquet"
    )
    resign(release)


def copy_release(datasets: Path, name: str) -> Path:
    target = datasets / name
    shutil.copytree(FIXTURE, target)
    return target


# The committed frozen schema


def test_committed_schema_accepts_the_committed_release(release):
    assert validate(release, FROZEN, now=NOW).status == PASSED


def test_frozen_schema_uses_contract_domains_not_observed_values(schema):
    features = {feature.name: feature for feature in tfdv.load_schema_text(str(schema)).feature}
    # Only none and cpu_sandbox occur in the data, but all four classes are allowed.
    assert list(features["label_resource"].string_domain.value) == [
        "none",
        "cpu_sandbox",
        "inference",
        "retrieval",
    ]
    assert features["queue_fraction"].float_domain.min == 0.0
    assert features["queue_fraction"].float_domain.max == 1.0
    assert features["workload_iterations"].int_domain.min == 1
    assert features["cpu_pressure_available"].int_domain.max == 1
    assert features["telemetry_coverage"].presence.min_fraction == 0.0  # nullable
    assert features["run_id"].presence.min_fraction == 1.0
    assert features["workload"].WhichOneof("domain_info") is None  # free text: new tasks are fine
    assert features["label_resource"].drift_comparator.infinity_norm.threshold == 0.25


def test_frozen_schema_file_is_marked_as_reviewed_input(schema):
    assert schema.read_text().startswith("# Frozen TFDV schema for steps-v1.2.")


# Passing and failing releases


def test_valid_release_passes_and_records_its_result(release, schema, datasets):
    outcome = validate(release, schema, now=NOW)
    assert outcome.status == PASSED
    result = json.loads(outcome.result_path.read_text())
    assert result["release"] == release.name and result["examples"] == 7
    assert result["anomalies"] == [] and result["drift"] == []
    assert result["baseline_release"] is None
    assert result["validated_at"] == NOW.isoformat()
    attempt = outcome.result_path.parent
    assert attempt.parent == datasets / VALIDATIONS_DIR / release.name
    assert result["attempt"] == attempt.name
    assert result["statistics_sha256"] == sha256(attempt / STATISTICS_FILE)
    manifest = json.loads((release / "manifest.json").read_text())
    assert result["release_parquet_sha256"] == manifest["parquet_sha256"]
    assert result["release_contract_sha256"] == manifest["schema_sha256"]
    assert sorted(path.name for path in attempt.parent.iterdir()) == [attempt.name]  # no staging


@pytest.mark.parametrize(
    ("column", "values", "anomaly", "as_type"),
    [
        ("label_resource", ["gpu"] + ["none"] * 6, "ENUM_TYPE_UNEXPECTED_STRING_VALUES", None),
        ("telemetry_status", ["partial"] * 7, "ENUM_TYPE_UNEXPECTED_STRING_VALUES", None),
        ("queue_fraction", [1.5] + [0.0] * 6, "FLOAT_TYPE_BIG_FLOAT", None),
        ("duration_seconds", [-1.0] + [0.3] * 6, "FLOAT_TYPE_SMALL_FLOAT", None),
        ("workload_iterations", [0] * 7, "INT_TYPE_SMALL_INT", None),
        ("run_id", [None] + ["run"] * 6, "FEATURE_TYPE_LOW_FRACTION_PRESENT", pa.string()),
    ],
)
def test_anomalies_fail_the_gate(release, schema, column, values, anomaly, as_type):
    change_column(release, column, values, as_type)
    outcome = validate(release, schema, now=NOW)
    assert outcome.status == FAILED
    [found] = outcome.result["anomalies"]
    assert found["feature"] == column and anomaly in found["types"]


@pytest.mark.parametrize(
    ("column", "values", "arrow_type"),
    [
        ("worker_capacity", ["1"] * 7, pa.string()),  # would convert to a valid integer
        ("cpu_pressure_available", ["1"] * 7, pa.string()),  # would convert to a valid boolean
        ("worker_capacity", [1.0] * 7, pa.float64()),
        ("workload", [1.5] * 7, pa.float64()),
        ("recorded_at", [1] * 7, pa.int64()),
    ],
)
def test_parquet_types_that_differ_from_the_contract_are_refused(
    release, schema, column, values, arrow_type
):
    retype(release, column, values, arrow_type)
    with pytest.raises(GateError, match=f"Parquet column {column} is .* contract declares"):
        validate(release, schema)


def test_a_declared_type_change_fails_the_frozen_schema(release, schema):
    """Changing the contract with the data keeps the release consistent; TFDV still blocks it."""
    retype(release, "worker_capacity", ["1"] * 7, pa.string())
    contract = json.loads((release / "schema.json").read_text())
    for column in contract["columns"]:
        if column["name"] == "worker_capacity":
            column.update(dtype="string", minimum=None)
    (release / "schema.json").write_text(json.dumps(contract))
    resign(release)
    outcome = validate(release, schema, now=NOW)
    assert outcome.status == FAILED
    [found] = outcome.result["anomalies"]
    assert found["feature"] == "worker_capacity"


def test_an_unsupported_contract_dtype_is_refused(release, schema):
    contract = json.loads((release / "schema.json").read_text())
    contract["columns"][0]["dtype"] = "decimal128"
    (release / "schema.json").write_text(json.dumps(contract))
    resign(release)
    with pytest.raises(GateError, match="unsupported contract dtype 'decimal128'"):
        validate(release, schema)


# Drift against the previous passing release


def test_drift_is_reported_against_the_previous_passing_release(datasets, release, schema):
    assert validate(release, schema, now=NOW).status == PASSED
    later = copy_release(datasets, "steps-v1.2+later")
    change_column(later, "label_resource", ["cpu_sandbox"] * 7)
    outcome = validate(later, schema, now=datetime(2026, 10, 13, tzinfo=UTC))
    assert outcome.status == PASSED  # drift alone never fails the gate
    assert outcome.result["baseline_release"] == release.name
    [drift] = outcome.result["drift"]
    assert drift["feature"] == "label_resource"
    assert drift["types"] == ["COMPARATOR_L_INFTY_HIGH"]


def passing(release: str, attempt: str, **fields) -> dict:
    return {
        "attempt": attempt,
        "release": release,
        "status": PASSED,
        "validated_at": "9999",
        "statistics_sha256": hashlib.sha256(b"").hexdigest(),
    } | fields


def test_baseline_ignores_failed_unreadable_and_uncommitted_attempts(datasets, release, schema):
    validations = datasets / VALIDATIONS_DIR
    hidden = [
        ("failed", "a1", passing("failed", "a1", status=FAILED)),
        ("broken", "a1", "{not json"),
        ("list", "a1", [1, 2]),
        ("staged", ".attempt-x", passing("staged", ".attempt-x")),  # not committed yet
        ("renamed", "a1", passing("renamed", "other")),  # result names another attempt
        ("moved", "a1", passing("elsewhere", "a1")),  # result names another release
        ("untimed", "a1", passing("untimed", "a1", validated_at=None)),
        ("no-stats", "a1", passing("no-stats", "a1")),
        ("tampered", "a1", passing("tampered", "a1")),  # statistics do not match the result
    ]
    for name, attempt, content in hidden:
        (validations / name / attempt).mkdir(parents=True)
        text = content if isinstance(content, str) else json.dumps(content)
        (validations / name / attempt / RESULT_FILE).write_text(text)
    (validations / "tampered" / "a1" / STATISTICS_FILE).write_text("datasets {}")
    (validations / "stray-file").write_text("")
    assert previous_passing(datasets, release.name) == (None, None)
    validate(release, schema, now=NOW)
    assert previous_passing(datasets, release.name) == (None, None)  # never itself
    baseline, stats = previous_passing(datasets, "another")
    assert baseline == release.name and stats.datasets[0].num_examples == 7


def test_no_validations_means_no_baseline_and_no_verdict(datasets, release):
    assert previous_passing(datasets, release.name) == (None, None)
    assert latest_result(datasets, release.name) == (None, None)


# Validation attempts


def minimum(stats, feature: str) -> float:
    [found] = [f for f in stats.datasets[0].features if f.path.step == [feature]]
    return found.num_stats.min


def test_revalidation_keeps_earlier_evidence_but_its_verdict_is_the_latest(
    datasets, release, schema
):
    first = validate(release, schema, now=NOW)
    change_column(release, "duration_seconds", [-1.0] * 7)
    second = validate(release, schema, now=datetime(2026, 10, 13, tzinfo=UTC))
    assert second.status == FAILED
    attempts = committed_attempts(datasets, release.name)
    assert [result["status"] for result, _ in attempts] == [PASSED, FAILED]
    assert json.loads(first.result_path.read_text())["status"] == PASSED  # evidence preserved
    assert latest_result(datasets, release.name)[0]["status"] == FAILED
    assert previous_passing(datasets, "another") == (None, None)  # no longer a baseline


def test_a_failed_commit_leaves_the_previous_attempt_paired_with_its_statistics(
    datasets, release, schema, monkeypatch
):
    validate(release, schema, now=NOW)
    change_column(release, "duration_seconds", [-1.0] * 7)

    def fail(source, target):
        raise OSError("disk full")

    monkeypatch.setattr(gate.os, "replace", fail)
    with pytest.raises(OSError, match="disk full"):
        validate(release, schema, now=datetime(2026, 10, 13, tzinfo=UTC))
    [(result, attempt)] = committed_attempts(datasets, release.name)
    assert sorted(path.name for path in attempt.parent.iterdir()) == [attempt.name]
    baseline, stats = previous_passing(datasets, "another")
    assert baseline == release.name and result["status"] == PASSED
    assert minimum(stats, "duration_seconds") >= 0  # the statistics that passed, not the new ones


def test_concurrent_attempts_commit_separately(datasets, release, schema, monkeypatch):
    """A second attempt that completes while the first is committing does not disturb it."""
    replace, interleaved = os.replace, []

    def commit_after_another_attempt(source, target):
        if not interleaved:
            interleaved.append(None)  # the other attempt commits normally
            interleaved[0] = validate(release, schema, now=NOW)
        replace(source, target)

    monkeypatch.setattr(gate.os, "replace", commit_after_another_attempt)
    outcome = validate(release, schema, now=NOW)
    attempts = committed_attempts(datasets, release.name)
    assert len(attempts) == 2
    assert {directory for _, directory in attempts} == {
        outcome.result_path.parent,
        interleaved[0].result_path.parent,
    }
    for result, directory in attempts:
        assert result["statistics_sha256"] == sha256(directory / STATISTICS_FILE)
    assert sorted(p.name for p in attempts[0][1].parent.iterdir()) == sorted(
        d.name for _, d in attempts
    )


# Releases and schemas the gate refuses


def test_tampered_data_is_refused(release, schema):
    table = pq.read_table(release / "steps.parquet")
    pq.write_table(table.slice(0, 3), release / "steps.parquet")  # manifest not updated
    with pytest.raises(GateError, match="does not match its manifest"):
        validate(release, schema)


def test_tampered_contract_is_refused(release, schema):
    contract = json.loads((release / "schema.json").read_text())
    contract["columns"][0]["nullable"] = True
    (release / "schema.json").write_text(json.dumps(contract))
    with pytest.raises(GateError, match="schema.json does not match"):
        validate(release, schema)


@pytest.mark.parametrize("missing", ["manifest.json", "schema.json"])
def test_releases_without_their_metadata_are_refused(release, schema, missing):
    (release / missing).unlink()
    with pytest.raises(GateError, match="is missing; rebuild"):
        validate(release, schema)


@pytest.mark.parametrize("content", ["{not json", "[1, 2]"])
def test_unreadable_metadata_is_refused(release, schema, content):
    (release / "manifest.json").write_text(content)
    with pytest.raises(GateError, match="Manifest"):
        validate(release, schema)


def test_a_directory_without_data_is_not_a_release(tmp_path, schema):
    with pytest.raises(GateError, match="is not a dataset release"):
        validate(tmp_path, schema)


def test_columns_must_match_the_contract(release, schema):
    table = pq.read_table(release / "steps.parquet")
    pq.write_table(table.drop(["queue_seconds"]), release / "steps.parquet")
    resign(release)
    with pytest.raises(GateError, match="do not match the column contract"):
        validate(release, schema)


@pytest.mark.parametrize("content", [None, "feature { name: "])
def test_missing_or_broken_frozen_schema_is_refused(release, tmp_path, content):
    path = tmp_path / "steps.pbtxt"
    if content is not None:
        path.write_text(content)
    with pytest.raises(GateError, match="Frozen schema"):
        validate(release, path)


def test_freezing_rejects_columns_outside_the_contract(release):
    loaded = load_release(release)
    contract = dict(loaded.contract, columns=loaded.contract["columns"][1:])
    with pytest.raises(GateError, match="outside the contract"):
        freeze_schema(gate.Release(loaded.path, loaded.manifest, contract, loaded.frame))


# Representation for TFDV


def test_frame_types_survive_missing_values(release):
    frame = load_release(release).frame
    assert str(frame["cgroup_throttled_usec_delta"].dtype) == "Int64"  # all null, still integer
    assert str(frame["cpu_pressure_available"].dtype) == "Int64"
    assert str(frame["recorded_at"].dtype) == "Int64"


def test_timestamps_become_epoch_microseconds():
    contract = {"columns": [{"name": "recorded_at", "dtype": "timestamp[us, tz=UTC]"}]}
    import pandas as pd

    stamps = pd.DataFrame({"recorded_at": [pd.Timestamp("2026-10-11T00:00:00Z"), pd.NaT]})
    prepared = gate.to_frame(stamps, contract)
    assert prepared["recorded_at"].iloc[0] == 1791676800000000
    assert pd.isna(prepared["recorded_at"].iloc[1])


def test_atomic_writes_leave_nothing_behind_on_failure(tmp_path, monkeypatch):
    def fail(source, target):
        raise OSError("disk full")

    monkeypatch.setattr(gate.os, "replace", fail)
    with pytest.raises(OSError):
        gate.write_atomic(tmp_path / "out" / "result.json", "{}")
    assert list((tmp_path / "out").iterdir()) == []


# Command line


def test_cli_validates_and_exits_zero(release, schema, capsys):
    assert main(["validate", "--release", str(release), "--schema", str(schema)]) == 0
    assert capsys.readouterr().out.startswith(f"passed: {release.name}")


def test_cli_exits_one_on_anomalies(release, schema, capsys):
    change_column(release, "label_resource", ["gpu"] * 7)
    assert main(["validate", "--release", str(release), "--schema", str(schema)]) == 1
    captured = capsys.readouterr()
    assert captured.out.startswith("failed:") and "anomaly label_resource" in captured.err


def test_cli_exits_one_when_the_gate_cannot_run(tmp_path, schema, capsys):
    assert main(["validate", "--release", str(tmp_path), "--schema", str(schema)]) == 1
    assert "Quality gate could not run" in capsys.readouterr().err


def test_cli_reports_drift(datasets, release, schema, capsys):
    validate(release, schema, now=NOW)
    later = copy_release(datasets, "steps-v1.2+later")
    change_column(later, "label_resource", ["cpu_sandbox"] * 7)
    assert main(["validate", "--release", str(later), "--schema", str(schema)]) == 0
    assert "drift label_resource" in capsys.readouterr().out


def test_cli_freezes_a_schema(release, tmp_path, capsys):
    path = tmp_path / "frozen.pbtxt"
    assert main(["freeze-schema", "--release", str(release), "--schema", str(path)]) == 0
    assert path.is_file() and capsys.readouterr().out == f"schema: {path}\n"


def test_dataset_level_anomalies_block_the_gate(release, schema, tmp_path):
    """No current rule produces one, but a future dataset constraint must not pass silently."""
    frozen = tfdv.load_schema_text(str(schema))
    frozen.dataset_constraints.min_examples_count = 1000
    strict = tmp_path / "strict.pbtxt"
    write_schema(frozen, strict, "steps-v1.2")
    outcome = validate(release, strict, now=NOW)
    assert outcome.status == FAILED
    [found] = outcome.result["anomalies"]
    assert found["feature"] == "<dataset>"


def test_integer_maximums_from_the_contract_are_enforced(release):
    loaded = load_release(release)
    columns = [
        dict(column, maximum=5) if column["name"] == "worker_capacity" else column
        for column in loaded.contract["columns"]
    ]
    contract = dict(loaded.contract, columns=columns)
    frozen = freeze_schema(gate.Release(loaded.path, loaded.manifest, contract, loaded.frame))
    [feature] = [f for f in frozen.feature if f.name == "worker_capacity"]
    assert (feature.int_domain.min, feature.int_domain.max) == (1, 5)


def test_module_runs_as_a_command(release, schema, monkeypatch):
    import runpy
    import sys

    argv = ["agentwatch_validation", "validate", "--release", str(release), "--schema", str(schema)]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.delitem(sys.modules, "agentwatch_validation.__main__", raising=False)
    with pytest.raises(SystemExit) as exited:
        runpy.run_module("agentwatch_validation", run_name="__main__")
    assert exited.value.code == 0
