"""Replay release rows: labels joined to step-features-v1, tested with synthetic experiments only."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from agentwatch_data.replay_evidence import ReplayEvidenceError
from agentwatch_data.replay_labels import load_replay_config
from agentwatch_data.replay_rows import (
    FEATURE_TIMING_TOLERANCE_SECONDS,
    REPLAY_STAGES,
    experiment_rows,
    main,
    replay_source_sha256,
)
from agentwatch_data.schema import SCHEMA_VERSION, validate_rows
from replay_fixtures import (
    DEFAULT_STEPS,
    configuration_digest,
    live_id,
    rejected_line,
    write_experiment,
)

# write_experiment's default schedule: 00 warm-up, 01-03 baselines, 04 cpu, 05 clean control.
CONTENDED, CONTROL = 4, 5
LEAF_STEPS = ("model-call-1", "tool-call-1")
LABELED_REPLAYS = (1, 2, 3, CONTENDED, CONTROL)


@pytest.fixture
def config():
    return load_replay_config()


def experiment(tmp_path: Path, **kwargs) -> Path:
    return write_experiment(tmp_path, live=True, **kwargs)


def journal_path(directory: Path, index: int) -> Path:
    return directory / "replays" / f"{index:02d}" / "live-journal.jsonl"


def read_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_lines(path: Path, lines: list[dict]) -> None:
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))


def edit_manifest(directory: Path, change) -> None:
    path = directory / "experiment.json"
    manifest = json.loads(path.read_text())
    change(manifest)
    path.write_text(json.dumps(manifest))


def step_events(lines: list[dict], step_id: str) -> list[dict]:
    return [
        line["payload"]
        for line in lines
        if line["kind"] == "record"
        and line["payload"]["record_type"] == "step_event"
        and line["payload"]["step_id"] == step_id
    ]


def origin(index: int, step_id: str) -> str:
    return f"experiment synthetic-exp replay {index:02d} step {step_id}"


def row_for(rows: list[dict], index: int, step_id: str) -> dict:
    return next(r for r in rows if r["replay_index"] == index and r["step_id"] == step_id)


def test_labeled_steps_get_features_from_their_own_replay(tmp_path, config):
    directory = experiment(tmp_path)
    rows, rejections = experiment_rows(directory, config)

    assert rows and all(r["feature_version"] == "step-features-v1" for r in rows)
    assert {r["kind"] for r in rows} == {"agent_replay"}
    assert [(r["replay_index"], r["step_id"]) for r in rows] == [
        (index, step_id) for index in LABELED_REPLAYS for step_id in LEAF_STEPS
    ]
    assert set(rejections) == set(REPLAY_STAGES)
    assert all(not found for found in rejections.values())
    tool = next(
        r for r in rows if r["step_id"] == "tool-call-1" and r["replay_role"] == "contended"
    )
    assert tool["telemetry_sample_count"] > 0 and tool["cpu_pressure_available"] is True
    assert tool["condition"] == "contended" and tool["rule_prediction"] is None
    assert tool["replay_condition"] == "cpu" and tool["run_id"].endswith(f"{CONTENDED:012d}")
    assert tool["recorded_at"] == datetime(2026, 11, 5, tzinfo=UTC)
    assert tool["workload"] == "agent replay fix-a-r1" and tool["task_group"] == "fix-a"
    assert tool["run_passed"] is True and tool["source"] == "measured"
    assert tool["sampling_interval_seconds"] == 0.25
    assert tool["configuration_sha256"] == configuration_digest()
    assert tool["source_sha256"] == replay_source_sha256(directory, f"replays/{CONTENDED:02d}")
    assert tool["schema_version"] == SCHEMA_VERSION
    assert (tool["workload_iterations"], tool["worker_capacity"], tool["tests_per_step"]) == (
        None,
        None,
        None,
    )
    assert {r["condition"] for r in rows if r["replay_role"] == "baseline"} == {"baseline"}
    assert {r["condition"] for r in rows if r["replay_role"] == "clean_control"} == {"clean"}
    # Each replay's rows carry that replay's own evidence digest.
    assert len({r["source_sha256"] for r in rows}) == len(LABELED_REPLAYS)
    validate_rows(rows)  # every column present and valid


def test_null_reading_stays_null_not_zero(tmp_path, config):
    rows, _ = experiment_rows(experiment(tmp_path), config)
    model = row_for(rows, CONTENDED, "model-call-1")  # its resource has no samples
    assert model["telemetry_status"] == "empty"
    assert model["telemetry_sample_count"] == 0
    assert model["cpu_pressure_max_avg10"] is None and model["cpu_pressure_available"] is False
    assert model["cgroup_throttled_usec_delta"] is None and model["cgroup_available"] is False
    assert model["telemetry_coverage"] is None
    assert model["measured_sampling_interval_seconds"] is None
    assert model["cpu_seconds"] is None  # the journal recorded no process CPU time


def test_missing_declared_interval_leaves_interval_and_coverage_null(tmp_path, config):
    directory = experiment(tmp_path)
    path = directory / "replays" / f"{CONTENDED:02d}" / "agent-evidence.json"
    evidence = json.loads(path.read_text())
    del evidence["configuration"]
    path.write_text(json.dumps(evidence))

    rows, _ = experiment_rows(directory, config)
    tool = row_for(rows, CONTENDED, "tool-call-1")
    assert tool["sampling_interval_seconds"] is None
    assert tool["telemetry_coverage"] is None
    assert tool["telemetry_sample_count"] > 0
    assert row_for(rows, CONTROL, "tool-call-1")["sampling_interval_seconds"] == 0.25


def all_labeled_steps() -> list[str]:
    return [origin(index, step) for index in LABELED_REPLAYS for step in LEAF_STEPS]


def edit_evidence(directory: Path, index: int, change) -> None:
    path = directory / "replays" / f"{index:02d}" / "agent-evidence.json"
    evidence = json.loads(path.read_text())
    change(evidence)
    path.write_text(json.dumps(evidence))


@pytest.mark.parametrize(
    "change",
    [
        lambda evidence: evidence["configuration"].update(sampling_interval_seconds="0.25"),
        lambda evidence: evidence["configuration"].update(sampling_interval_seconds=True),
        lambda evidence: evidence.update(configuration=[0.25]),
        *[
            lambda evidence, value=value: evidence["configuration"].update(
                sampling_interval_seconds=value
            )
            for value in (
                0,
                -0.25,
                1e-10,
                5e-324,
                1e308,
                10**400,
                float("nan"),
                float("inf"),
                -float("inf"),
            )
        ],
    ],
    ids=[
        "string",
        "boolean",
        "not-an-object",
        "zero",
        "negative",
        "sub-ns",
        "tiny",
        "overflow",
        "huge-integer",
        "nan",
        "positive-infinity",
        "negative-infinity",
    ],
)
def test_invalid_declared_interval_rejects_the_whole_experiment(tmp_path, config, change):
    """A read defect in one replay rejects its experiment, not just that replay (D03 spec)."""
    directory = experiment(tmp_path)
    edit_evidence(directory, CONTENDED, change)

    rows, rejections = experiment_rows(directory, config)
    assert rows == []
    assert [r.origin for r in rejections["replay_features"]] == all_labeled_steps()
    assert all(
        r.reason.startswith(f"declared_interval_invalid: replay {CONTENDED:02d} ")
        for r in rejections["replay_features"]
    )
    assert not rejections["replay_label"] and not rejections["replay_schema"]


@pytest.mark.parametrize(
    ("offset_seconds", "rejected"),
    [(1e-3, True), (FEATURE_TIMING_TOLERANCE_SECONDS / 2, False)],
)
def test_timing_mismatch_rejects_only_that_step(tmp_path, config, offset_seconds, rejected):
    directory = experiment(tmp_path)

    def shift(manifest):
        manifest["replays"][CONTENDED]["steps"]["tool-call-1"]["total_seconds"] += offset_seconds

    edit_manifest(directory, shift)
    rows, rejections = experiment_rows(directory, config)
    if not rejected:
        assert len(rows) == 2 * len(LABELED_REPLAYS) and not rejections["replay_features"]
        return
    assert len(rows) == 2 * len(LABELED_REPLAYS) - 1
    (rejection,) = rejections["replay_features"]
    assert rejection.origin == origin(CONTENDED, "tool-call-1")
    assert rejection.reason.startswith("feature_timing_mismatch:")
    assert "duration_seconds" in rejection.reason
    assert not rejections["replay_label"] and not rejections["replay_schema"]


def test_missing_feature_row_rejects_only_that_step(tmp_path, config):
    directory = experiment(tmp_path)
    path = journal_path(directory, CONTENDED)
    write_lines(
        path,
        [
            line
            for line in read_lines(path)
            if not (
                line["kind"] == "record"
                and line["payload"].get("step_id") == "tool-call-1"
                and line["payload"].get("state") == "completed"
            )
        ],
    )

    rows, rejections = experiment_rows(directory, config)
    assert len(rows) == 2 * len(LABELED_REPLAYS) - 1
    (rejection,) = rejections["replay_features"]
    assert rejection.origin == origin(CONTENDED, "tool-call-1")
    assert rejection.reason.startswith("feature_row_missing:")


def test_swapped_journals_never_attribute_features_to_the_wrong_replay(tmp_path, config):
    directory = experiment(tmp_path)
    contended, control = journal_path(directory, CONTENDED), journal_path(directory, CONTROL)
    contended_text, control_text = contended.read_text(), control.read_text()
    contended.write_text(control_text)
    control.write_text(contended_text)

    rows, rejections = experiment_rows(directory, config)
    assert {r["replay_index"] for r in rows} == {1, 2, 3}
    assert sorted(r.origin for r in rejections["replay_features"]) == sorted(
        origin(index, step) for index in (CONTENDED, CONTROL) for step in LEAF_STEPS
    )
    for rejection in rejections["replay_features"]:
        assert rejection.reason.startswith("feature_row_missing:")
        assert "run_id" in rejection.reason


def unknown_journal_line(directory: Path) -> None:
    path = journal_path(directory, CONTENDED)
    write_lines(path, [*read_lines(path), {"kind": "surprise", "payload": {}}])


def out_of_contract_sample(directory: Path) -> None:
    path = journal_path(directory, CONTENDED)
    lines = read_lines(path)
    lines[-2]["payload"]["value"] = 150.0  # pressure is a percentage
    write_lines(path, lines)


@pytest.mark.parametrize(
    ("damage", "detail"),
    [(unknown_journal_line, "unknown kind"), (out_of_contract_sample, "resource_sample")],
    ids=["unknown-kind", "out-of-contract-record"],
)
def test_unreadable_feature_evidence_rejects_the_whole_experiment(tmp_path, config, damage, detail):
    directory = experiment(tmp_path)
    damage(directory)

    rows, rejections = experiment_rows(directory, config)
    assert rows == []
    assert [r.origin for r in rejections["replay_features"]] == all_labeled_steps()
    for rejection in rejections["replay_features"]:
        assert rejection.reason.startswith(f"feature_evidence_unreadable: replay {CONTENDED:02d} ")
        assert detail in rejection.reason


def test_main_counts_an_experiment_rejected_for_unreadable_evidence(tmp_path, capsys):
    directory = experiment(tmp_path)
    unknown_journal_line(directory)
    assert main([str(directory)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["rows"] == 0
    assert printed["rejections"]["replay_features"] == {
        "feature_evidence_unreadable": len(all_labeled_steps())
    }


def test_records_the_live_api_refused_still_feed_their_rows(tmp_path, config):
    """A refused delivery is not a measurement defect: the rows keep the journal's readings."""
    directory = experiment(tmp_path)
    before, _ = experiment_rows(directory, config)
    path = journal_path(directory, CONTENDED)
    lines = read_lines(path)
    samples = [
        line["payload"]["sample_id"]
        for line in lines
        if line["kind"] == "record" and line["payload"]["record_type"] == "resource_sample"
    ]
    write_lines(path, [*lines[:-1], rejected_line(samples), lines[-1]])

    rows, rejections = experiment_rows(directory, config)
    assert all(not found for found in rejections.values())
    digest = ("source_sha256",)  # the journal's bytes changed, so only its digest differs
    assert [{k: v for k, v in r.items() if k not in digest} for r in rows] == [
        {k: v for k, v in r.items() if k not in digest} for r in before
    ]
    assert row_for(rows, CONTENDED, "tool-call-1")["telemetry_sample_count"] > 0


def test_zero_duration_step_is_rejected_at_replay_schema(tmp_path, config):
    """A step whose journal window is empty has no queue fraction; only it is rejected."""
    directory = experiment(tmp_path)
    path = journal_path(directory, CONTENDED)
    lines = read_lines(path)
    for event in step_events(lines, "tool-call-1"):
        started_tick_ns = event["submitted_tick_ns"] + 1_000_000
        event["submitted_tick_ns"] = started_tick_ns
        if event["started_tick_ns"] is not None:
            event["started_tick_ns"] = started_tick_ns
        if event["finished_tick_ns"] is not None:
            event["finished_tick_ns"] = started_tick_ns
    write_lines(path, lines)
    # The manifest's own timing stays positive (so labeling accepts the step) and within the
    # timing tolerance of the journal's zero-length window.
    near_zero_seconds = FEATURE_TIMING_TOLERANCE_SECONDS / 2

    def shrink(manifest):
        manifest["replays"][CONTENDED]["steps"]["tool-call-1"].update(
            total_seconds=near_zero_seconds, queue_seconds=0.0, execution_seconds=near_zero_seconds
        )

    edit_manifest(directory, shrink)

    rows, rejections = experiment_rows(directory, config)
    assert len(rows) == 2 * len(LABELED_REPLAYS) - 1
    assert not rejections["replay_features"] and not rejections["replay_label"]
    (rejection,) = rejections["replay_schema"]
    assert rejection.origin == origin(CONTENDED, "tool-call-1")
    assert "queue_fraction is null" in rejection.reason


def throttled_samples(lines: list[dict], step_usec: float) -> list[dict]:
    """A cumulative throttling counter on the tool host at every pressure sample's tick."""
    pressure = [
        line["payload"]
        for line in lines
        if line["kind"] == "record" and line["payload"]["record_type"] == "resource_sample"
    ]
    return [
        {
            "kind": "record",
            "payload": sample
            | {
                "sample_id": live_id(sample["run_id"], "throttled", sample["tick_ns"]),
                "metric": "cpu.cgroup.throttled_usec.v1",
                "unit": "microseconds",
                "value": step_usec * number,
            },
        }
        for number, sample in enumerate(pressure)
    ]


def test_integral_throttling_delta_is_an_integer(tmp_path, config):
    long_tool = {"cpu": {**DEFAULT_STEPS, "tool-call-1": 1.0}}
    directory = experiment(tmp_path, contended=long_tool)
    path = journal_path(directory, CONTENDED)
    lines = read_lines(path)
    write_lines(path, [*lines, *throttled_samples(lines, 100.0)])

    rows, rejections = experiment_rows(directory, config)
    delta = row_for(rows, CONTENDED, "tool-call-1")["cgroup_throttled_usec_delta"]
    assert type(delta) is int and delta > 0
    assert not rejections["replay_schema"]


def test_fractional_throttling_delta_is_rejected_at_replay_schema(tmp_path, config):
    long_tool = {"cpu": {**DEFAULT_STEPS, "tool-call-1": 1.0}}
    directory = experiment(tmp_path, contended=long_tool)
    path = journal_path(directory, CONTENDED)
    lines = read_lines(path)
    write_lines(path, [*lines, *throttled_samples(lines, 100.5)])

    rows, rejections = experiment_rows(directory, config)
    assert len(rows) == 2 * len(LABELED_REPLAYS) - 1
    (rejection,) = rejections["replay_schema"]
    assert rejection.origin == origin(CONTENDED, "tool-call-1")
    assert "cgroup_throttled_usec_delta" in rejection.reason


def test_source_sha256_changes_when_any_of_the_three_files_changes(tmp_path):
    directory = experiment(tmp_path)
    evidence_dir = f"replays/{CONTENDED:02d}"
    seen = {replay_source_sha256(directory, evidence_dir)}
    for relative in (
        "experiment.json",
        f"{evidence_dir}/agent-evidence.json",
        f"{evidence_dir}/live-journal.jsonl",
    ):
        path = directory / relative
        path.write_bytes(path.read_bytes() + b"\n")
        seen.add(replay_source_sha256(directory, evidence_dir))
    assert len(seen) == 4
    # Another replay's files are not part of this replay's source.
    other = journal_path(directory, CONTROL)
    before = replay_source_sha256(directory, evidence_dir)
    other.write_bytes(other.read_bytes() + b"\n")
    assert replay_source_sha256(directory, evidence_dir) == before


def test_source_sha256_refuses_paths_outside_the_experiment(tmp_path):
    directory = experiment(tmp_path)
    for evidence_dir in ("../outside", "/replays/04"):
        with pytest.raises(ReplayEvidenceError):
            replay_source_sha256(directory, evidence_dir)


def test_missing_passed_is_null_and_rejected_never_false(tmp_path, config):
    rows, rejections = experiment_rows(experiment(tmp_path, passed=None), config)
    assert rows == []
    assert not rejections["replay_label"] and not rejections["replay_features"]
    assert len(rejections["replay_schema"]) == 2 * len(LABELED_REPLAYS)
    assert all("run_passed is null" in r.reason for r in rejections["replay_schema"])


def test_label_rejections_pass_through(tmp_path, config):
    directory = experiment(tmp_path, mismatched=[(CONTENDED, "tool-call-1")])
    rows, rejections = experiment_rows(directory, config)
    (rejection,) = rejections["replay_label"]
    assert rejection.origin == origin(CONTENDED, "tool-call-1")
    assert rejection.reason.startswith("output_differs_from_source:")
    assert len(rows) == 2 * len(LABELED_REPLAYS) - 1


def test_unreadable_manifest_propagates(tmp_path, config):
    directory = experiment(tmp_path)
    (directory / "experiment.json").write_text("{not json")
    with pytest.raises(ReplayEvidenceError):
        experiment_rows(directory, config)


def test_main_prints_row_and_rejection_counts(tmp_path, capsys):
    directory = experiment(tmp_path, mismatched=[(CONTENDED, "tool-call-1")])
    assert main([str(directory)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["rows"] == 2 * len(LABELED_REPLAYS) - 1
    assert output["rejections"] == {
        "replay_label": {"output_differs_from_source": 1},
        "replay_features": {},
        "replay_schema": {},
    }


def test_one_nanosecond_interval_is_valid(tmp_path, config):
    directory = experiment(tmp_path)
    edit_evidence(
        directory,
        CONTENDED,
        lambda evidence: evidence["configuration"].update(sampling_interval_seconds=1e-9),
    )
    rows, rejections = experiment_rows(directory, config)
    assert rows
    assert not rejections["replay_features"]


def test_renamed_replay_copy_cannot_duplicate_release_rows(tmp_path):
    import shutil

    from agentwatch_data.build import derive_replay_rows

    original = experiment(tmp_path / "first")
    copied = shutil.copytree(original, tmp_path / "renamed")
    rows, rejections = derive_replay_rows([original, copied], None)
    assert rows == []
    assert any("run identity" in r.reason for items in rejections.values() for r in items)


def test_duplicate_replay_rejection_keeps_independent_experiments(tmp_path):
    import shutil

    from agentwatch_data.build import derive_replay_rows

    original = experiment(tmp_path / "first")
    copied = shutil.copytree(original, tmp_path / "renamed")
    independent = experiment(tmp_path / "independent", name="independent", run_namespace=1)
    rows, rejections = derive_replay_rows([original, copied, independent], None)
    assert len(rows) == 10
    assert {r["experiment"] for r in rows} == {"independent"}
    assert sum(len(items) for items in rejections.values()) == 2
