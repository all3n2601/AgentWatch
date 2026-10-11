"""Golden test: a real recorded run through extract, flatten, label, and schema validation.

The fixture is an unmodified `make demo-coding` run recorded on macOS on 2026-10-05, so
CPU pressure and cgroup readings are unavailable, as they are on every macOS host.
"""

from pathlib import Path

import pytest
from agentwatch_data.extract import extract_local
from agentwatch_data.flatten import flatten_runs
from agentwatch_data.labels import label_rows, load_config
from agentwatch_data.schema import COLUMNS, RELEASE_COLUMNS, SCHEMA_VERSION, validate_rows

FIXTURES = Path(__file__).parent / "fixtures"


def build_rows():
    extraction = extract_local(FIXTURES)
    assert extraction.rejections == []
    rows, flatten_rejections = flatten_runs(extraction.runs)
    labeled, label_rejections = label_rows(rows, load_config())
    assert flatten_rejections == [] and label_rejections == []
    return labeled


@pytest.fixture(scope="module")
def rows():
    return build_rows()


def test_every_row_satisfies_the_full_schema(rows):
    validate_rows(rows)
    produced = set().union(*rows)
    assert produced == {column.name for column in COLUMNS} - RELEASE_COLUMNS


def test_recorded_run_produces_the_expected_rows(rows):
    assert [row["condition"] for row in rows] == ["baseline"] * 5 + ["clean", "contended"]
    assert [row["label_resource"] for row in rows] == ["none"] * 6 + ["cpu_sandbox"]
    assert [row["label_basis"] for row in rows] == ["baseline_reference"] * 5 + [
        "clean_within_noise",
        "contention_exceeded_threshold",
    ]
    assert [row["telemetry_sample_count"] for row in rows] == [1, 2, 1, 2, 1, 2, 11]
    contended = rows[-1]
    assert contended["queue_seconds"] == pytest.approx(2.488, abs=1e-3)
    assert contended["label_excess_fraction"] == pytest.approx(0.871, abs=1e-3)
    assert contended["injection_method"] == "worker_queue_blocker"


def test_macos_telemetry_is_present_but_unavailable(rows):
    assert {row["telemetry_status"] for row in rows} == {"present"}
    assert not any(row["cpu_pressure_available"] or row["cgroup_available"] for row in rows)
    assert all(row["cpu_pressure_max_avg10"] is None for row in rows)
    assert all(row["telemetry_coverage"] == 1.0 for row in rows)
    assert {row["sampling_interval_seconds"] for row in rows} == {0.25}
    [measured] = {row["measured_sampling_interval_seconds"] for row in rows}
    assert measured == pytest.approx(0.255, abs=0.01)


def test_lineage_identifies_versions_and_evidence(rows):
    assert {row["schema_version"] for row in rows} == {SCHEMA_VERSION}
    assert {row["label_version"] for row in rows} == {load_config().label_version}
    assert len({row["source_sha256"] for row in rows}) == 1
    assert {(row["run_passed"], row["failed_checks"]) for row in rows} == {(True, None)}


def test_rebuilding_from_the_same_evidence_is_identical(rows):
    assert build_rows() == rows
