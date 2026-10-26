import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from agentwatch_data.build import DATASET_FILE
from agentwatch_data.slices import (
    ABSENT_FROM_SPLIT,
    EXCLUDED,
    MISSING_READINGS_GAP,
    MITIGATIONS,
    POSITIVE_RATE_GAP,
    SINGLE_CLASS,
    UNASSIGNED,
    UNDER_REPRESENTED,
    UNRECORDED,
    Assignment,
    SliceError,
    analyze,
    load_assignment,
    load_rows,
    main,
    markdown,
    slice_release,
)

RELEASE = "steps-v1.2+0123456789ab"
CPU_TASK, CODING_TASK = "cpu:task:1", "coding:task:1"


def step(index: int, kind: str, label: str, **overrides: object) -> dict[str, object]:
    """One synthetic step row (test fixture, not recorded evidence) with every sliced column."""
    contended = label != "none"
    row: dict[str, object] = {
        "run_id": f"{kind}-run-{index // 5}",
        "task_id": CPU_TASK if kind == "cpu" else CODING_TASK,
        "kind": kind,
        "label_resource": label,
        "rule_prediction": label,
        "condition": "contended" if contended else "clean",
        "injection_method": "worker_queue_blocker" if contended else "none",
        "agent_model": None,
        "source": "measured",
        "telemetry_status": "present",
        "cpu_pressure_available": True,
        "cgroup_available": True,
        "telemetry_coverage": 0.9,
        "cpu_pressure_max_avg10": 1.5,
        "cgroup_throttled_usec_delta": 0,
    }
    return row | overrides


def balanced(per_group: int = 30) -> list[dict[str, object]]:
    """Both kinds, each half positive, every reading present: nothing to flag."""
    return [
        step(index, kind, label)
        for kind in ("cpu", "coding")
        for label in ("none", "cpu_sandbox")
        for index in range(per_group)
    ]


def kinds(report: dict, attribute: str | None = None) -> list[tuple[str, object]]:
    return [
        (flag["kind"], flag["value"])
        for flag in report["flags"]
        if attribute is None or flag["attribute"] == attribute
    ]


def write_release(tmp_path, rows: list[dict[str, object]]):
    release = tmp_path / "datasets" / RELEASE
    release.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), release / DATASET_FILE)
    return release


def write_split(tmp_path, groups: dict[str, str], excluded: list[str] = (), release=RELEASE):
    path = tmp_path / "split.json"
    document = {
        "release": release,
        "groups": groups,
        "excluded_runs": [{"run_id": run, "source": "simulated"} for run in excluded],
    }
    path.write_text(json.dumps(document))
    return path


def test_balanced_data_has_no_flags():
    report = analyze(balanced(), RELEASE, None)
    assert report["flags"] == []
    assert report["rows"] == 120
    assert report["runs"] == 12  # six runs per kind, each with steps of both labels
    assert report["split_rows"] is None
    kind = report["attributes"]["kind"]["slices"]["cpu"]
    assert kind["rows"] == 60
    assert kind["share"] == 0.5
    assert kind["labels"] == {"cpu_sandbox": 30, "none": 30}
    assert kind["positive_rate"] == 0.5
    assert kind["rule_agreement"] == 1.0
    assert "splits" not in kind


def test_a_null_attribute_is_its_own_unrecorded_slice_and_booleans_read_as_words():
    report = analyze(balanced(), RELEASE, None)
    assert list(report["attributes"]["agent_model"]["slices"]) == [UNRECORDED]
    assert list(report["attributes"]["cgroup_available"]["slices"]) == ["true"]


def test_a_small_slice_is_under_represented():
    rows = balanced() + [
        step(index, "coding", "none", task_id="coding:other:1") for index in range(4)
    ]
    rows += [step(index, "coding", "cpu_sandbox", task_id="coding:other:1") for index in range(4)]
    report = analyze(rows, RELEASE, None)
    assert kinds(report, "task_id") == [(UNDER_REPRESENTED, "coding:other:1")]
    (flag,) = [f for f in report["flags"] if f["attribute"] == "task_id"]
    assert "8 rows" in flag["detail"]
    assert flag["mitigation"] == MITIGATIONS[UNDER_REPRESENTED]


def test_a_slice_with_one_class_is_flagged_but_not_for_label_defining_attributes():
    rows = [r for r in balanced() if not (r["kind"] == "coding" and r["label_resource"] != "none")]
    report = analyze(rows, RELEASE, None)
    assert (SINGLE_CLASS, "coding") in kinds(report, "kind")
    for attribute in ("label_resource", "condition", "injection_method"):
        assert SINGLE_CLASS not in [kind for kind, _ in kinds(report, attribute)]


def test_a_positive_rate_gap_across_an_attribute_is_flagged():
    rows = balanced()
    # Relabel 20 coding positives as negatives: cpu stays at 50%, coding falls to ~17%.
    relabeled = 0
    for row in rows:
        if row["kind"] == "coding" and row["label_resource"] != "none" and relabeled < 20:
            row.update(label_resource="none", rule_prediction="none")
            relabeled += 1
    report = analyze(rows, RELEASE, None)
    assert (POSITIVE_RATE_GAP, None) in kinds(report, "kind")
    # The label and the attributes that define it are never compared for a label-rate gap.
    for attribute in ("label_resource", "condition", "injection_method"):
        assert POSITIVE_RATE_GAP not in [kind for kind, _ in kinds(report, attribute)]


def test_rule_agreement_counts_rows_where_the_rule_matches_the_label():
    rows = balanced()
    for row in rows[:12]:
        row["rule_prediction"] = "cpu_sandbox"  # the first 12 rows are cpu negatives
    report = analyze(rows, RELEASE, None)
    assert report["attributes"]["kind"]["slices"]["cpu"]["rule_agreement"] == 0.8


def test_a_missing_readings_gap_is_flagged_and_missing_is_not_zero():
    rows = balanced()
    for row in rows:
        if row["kind"] == "cpu":
            row["telemetry_coverage"] = None
    report = analyze(rows, RELEASE, None)
    flags = [f for f in report["flags"] if f["attribute"] == "kind"]
    assert [(f["kind"], f["value"]) for f in flags] == [(MISSING_READINGS_GAP, None)]
    assert "telemetry_coverage is missing on 0.0%–100.0%" in flags[0]["detail"]
    missing = report["attributes"]["kind"]["slices"]["coding"]["missing_rate"]
    assert missing["cgroup_throttled_usec_delta"] == 0.0  # a measured zero is not missing


def test_an_availability_flag_is_not_compared_with_its_own_reading():
    rows = balanced()
    for row in rows[:60]:
        row.update(cgroup_available=False, cgroup_throttled_usec_delta=None)
    report = analyze(rows, RELEASE, None)
    details = [f["detail"] for f in report["flags"] if f["attribute"] == "cgroup_available"]
    assert not any("cgroup_throttled_usec_delta" in detail for detail in details)
    assert any(
        "cgroup_throttled_usec_delta" in detail
        for f in report["flags"]
        if f["attribute"] == "kind"
        for detail in [f["detail"]]
    )


def test_slices_absent_from_the_train_or_test_split_are_flagged():
    assignment = Assignment({CPU_TASK: "train", CODING_TASK: "test"}, frozenset())
    report = analyze(balanced(), RELEASE, assignment)
    assert report["split_rows"] == {"test": 60, "train": 60}
    assert kinds(report, "kind") == [(ABSENT_FROM_SPLIT, "coding"), (ABSENT_FROM_SPLIT, "cpu")]
    cpu = report["attributes"]["kind"]["slices"]["cpu"]["splits"]
    assert cpu == {"train": 60, "validation": 0, "test": 0, EXCLUDED: 0, UNASSIGNED: 0}


def test_excluded_runs_and_unplaced_tasks_are_not_counted_in_a_split():
    excluded = frozenset({"coding-run-0"})
    report = analyze(balanced(), RELEASE, Assignment({CODING_TASK: "test"}, excluded))
    coding = report["attributes"]["kind"]["slices"]["coding"]["splits"]
    assert coding[EXCLUDED] == 10  # run 0's steps in both label groups
    assert coding["test"] == 50
    assert report["attributes"]["kind"]["slices"]["cpu"]["splits"][UNASSIGNED] == 60
    # No train rows at all, so being absent from train is not a per-slice finding.
    assert (ABSENT_FROM_SPLIT, "cpu") in kinds(report, "kind")
    assert all("train" not in f["detail"] for f in report["flags"])


def test_balancing_weights_give_every_value_the_same_total_weight():
    rows = balanced() + [step(index, "coding", "none") for index in range(60)]
    report = analyze(rows, RELEASE, None)
    weights = report["attributes"]["kind"]["balancing_weights"]
    slices = report["attributes"]["kind"]["slices"]
    totals = {value: weights[value] * slices[value]["rows"] for value in slices}
    assert totals["cpu"] == pytest.approx(totals["coding"])
    assert weights == {"coding": 0.75, "cpu": 1.5}


def test_the_markdown_lists_flags_and_one_table_per_attribute():
    assignment = Assignment({CPU_TASK: "train", CODING_TASK: "test"}, frozenset())
    text = markdown(analyze(balanced(), RELEASE, assignment))
    assert text.startswith(f"# Slices of {RELEASE}")
    assert "Rows per split: test 60, train 60." in text
    assert "- **absent_from_split** — kind = coding: no rows in the train split." in text
    assert "| cpu | 60 | 50.0% | 50.0% | 100.0% |" in text
    assert text.count("\n## ") == 1 + 10  # flags, then each attribute
    assert "Train / test" in text


def test_the_markdown_says_when_nothing_was_flagged():
    text = markdown(analyze(balanced(), RELEASE, None))
    assert "No slice crossed a declared threshold." in text
    assert "Train / test" not in text
    assert "Rows per split" not in text


def test_slice_release_writes_both_reports(tmp_path):
    release = write_release(tmp_path, balanced())
    split = write_split(tmp_path, {CPU_TASK: "train", CODING_TASK: "test"})
    result = slice_release(release, split, tmp_path / "reports")
    assert json.loads(result.json_path.read_text()) == result.report
    assert result.markdown_path.read_text() == markdown(result.report)


def test_an_unreadable_release_is_refused(tmp_path):
    release = tmp_path / RELEASE
    release.mkdir()
    (release / DATASET_FILE).write_text("not parquet")
    with pytest.raises(SliceError, match="unreadable"):
        load_rows(release)


def test_a_release_missing_sliced_columns_is_refused(tmp_path):
    rows = [{k: v for k, v in row.items() if k != "kind"} for row in balanced()]
    with pytest.raises(SliceError, match=r"lacks columns the slices need: \['kind'\]"):
        load_rows(write_release(tmp_path, rows))


def test_an_empty_release_is_refused(tmp_path):
    release = write_release(tmp_path, balanced())
    table = pq.read_table(release / DATASET_FILE)
    pq.write_table(table.slice(0, 0), release / DATASET_FILE)
    with pytest.raises(SliceError, match="no rows"):
        load_rows(release)


def test_a_split_of_another_release_is_refused(tmp_path):
    split = write_split(tmp_path, {CPU_TASK: "train"}, release="steps-v1.2+ffffffffffff")
    with pytest.raises(SliceError, match="not steps-v1.2"):
        load_assignment(split, tmp_path / RELEASE)


@pytest.mark.parametrize("content", ["not json", "{}", '{"release": 1}'])
def test_an_unreadable_split_is_refused(tmp_path, content):
    split = tmp_path / "split.json"
    split.write_text(content)
    with pytest.raises(SliceError, match="unreadable"):
        load_assignment(split, tmp_path / RELEASE)


@pytest.mark.parametrize("groups", [[], {CPU_TASK: "holdout"}])
def test_a_split_with_an_invalid_assignment_is_refused(tmp_path, groups):
    with pytest.raises(SliceError, match="invalid task assignment"):
        load_assignment(write_split(tmp_path, groups), tmp_path / RELEASE)


def test_load_assignment_reads_groups_and_excluded_runs(tmp_path):
    split = write_split(tmp_path, {CPU_TASK: "train"}, excluded=["cpu-run-0"])
    assignment = load_assignment(split, tmp_path / RELEASE)
    assert assignment == Assignment({CPU_TASK: "train"}, frozenset({"cpu-run-0"}))


def test_main_writes_the_reports_and_prints_each_flag(tmp_path, capsys):
    release = write_release(tmp_path, balanced())
    split = write_split(tmp_path, {CPU_TASK: "train", CODING_TASK: "test"})
    output = tmp_path / "reports"
    argv = ["--release", str(release), "--split", str(split), "--output", str(output)]
    assert main(argv) == 0
    printed = capsys.readouterr().out
    assert f"slices: {output / 'slices.json'}" in printed
    assert f"summary: {output / 'slices.md'}" in printed
    assert "  flag absent_from_split: kind=coding: no rows in the train split" in printed


def test_main_defaults_to_a_report_directory_outside_the_datasets_store(
    tmp_path, capsys, monkeypatch
):
    release = write_release(tmp_path, balanced())
    monkeypatch.chdir(tmp_path)
    assert main(["--release", str(release)]) == 0
    assert (tmp_path / ".data" / "reports" / RELEASE / "slices.json").is_file()
    assert "flag" not in capsys.readouterr().out


def test_main_reports_a_failure_without_writing(tmp_path, capsys):
    release = tmp_path / RELEASE
    release.mkdir()
    output = tmp_path / "reports"
    assert main(["--release", str(release), "--output", str(output)]) == 1
    assert "Slicing failed:" in capsys.readouterr().err
    assert not output.exists()
