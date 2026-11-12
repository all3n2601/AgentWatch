"""The capacity record's tables over sweep folders the stub driver recorded; never real evidence."""

import json
from pathlib import Path

import pytest
from agentwatch_agent import cli, sweep, sweep_record
from agentwatch_agent.sweep_offline import SweepFolderError
from agentwatch_agent.sweep_record import render_coverage_tables, render_sweep_tables
from test_sweep import Harness, read


@pytest.fixture
def recorded(tmp_path):
    harness = Harness(tmp_path)
    return harness, harness.run()


def rows(markdown: str, heading: str) -> list[list[str]]:
    """The cells of the first table after `heading`, without its header and divider."""
    lines = markdown.split(heading + "\n", 1)[1].splitlines()
    start = next(index for index, line in enumerate(lines) if line.startswith("| "))
    table = []
    for line in lines[start + 2 :]:
        if not line.startswith("| "):
            break
        table.append([cell.strip() for cell in line[2:-2].split(" | ")])
    return table


def rewrite(harness, change):
    record = read(harness.sweep_file)
    change(record)
    harness.sweep_file.write_text(json.dumps(record))


def test_the_tables_show_the_recorded_decision_and_every_level(recorded):
    harness, record = recorded
    markdown = render_sweep_tables(harness.output_dir)
    assert f"Sweep `{record['sweep_id']}`" in markdown
    decision = record["capacity_decision"]
    assert f"- Result: **{decision['result']}**, capacity **{decision['capacity']}**." in markdown
    ramp = rows(markdown, "#### Ramp")
    assert [row[0] for row in ramp] == [str(level) for level in record["level_summaries"]]
    # Each level's batches and runs are the recorded counts.
    for row, summary in zip(ramp, record["level_summaries"].values(), strict=True):
        assert row[1] == f"{summary['batches']} / {summary['runs']}"
    criteria = rows(markdown, "#### Capacity decision")
    assert [row[0] for row in criteria] == [str(level) for level in decision["criteria"]]
    labels = rows(markdown, "#### Label validation")
    assert labels[-1][0] == "all N ≥ 2"
    assert labels[-1][-1] == record["label_validation"]["overall"]["verdict"]


def test_a_value_that_was_not_recorded_prints_as_not_available_never_zero(recorded):
    harness, _record = recorded

    def clear(record):
        first = next(iter(record["level_summaries"].values()))
        first["throughput_per_minute"] = None
        first["device"]["metrics"]["gpu_power_watts"] = None

    rewrite(harness, clear)
    first = rows(render_sweep_tables(harness.output_dir), "#### Ramp")[0]
    assert first[3] == "n/a ± n/a (n/a)"
    assert first[14] == "n/a"


def test_the_flood_activity_comes_from_the_injection_log(recorded):
    harness, record = recorded
    injected = next(entry for entry in record["batches"] if entry["phase"] == "injected")
    log = read(harness.output_dir / injected["injection_log"])
    activity = log["injections"][0]["activity"]
    table = rows(render_sweep_tables(harness.output_dir), "#### Injected condition")
    flood = next(row for row in table if row[1] == "injected")[-1]
    assert flood.startswith(
        f"{activity['requests_completed']} done / {activity['requests_failed']} failed"
    )
    assert all(row[-1] == "no flood" for row in table if row[1] == "control")


@pytest.mark.parametrize(
    "change",
    [
        lambda record: record.pop("level_summaries"),
        lambda record: record["capacity_decision"].pop("criteria"),
        lambda record: record["configuration"].pop("verified"),
    ],
    ids=["no level summaries", "no decision criteria", "no verified configuration"],
)
def test_a_record_without_a_field_the_tables_read_is_refused(recorded, change):
    harness, _record = recorded
    rewrite(harness, change)
    with pytest.raises(SweepFolderError, match="lacks a field the record tables read"):
        render_sweep_tables(harness.output_dir)


def test_an_injection_log_outside_the_sweep_folder_is_never_read(recorded, tmp_path):
    harness, _record = recorded
    outside = tmp_path / "elsewhere.json"
    outside.write_text("{}")

    def escape(record):
        entry = next(e for e in record["batches"] if e["phase"] == "injected")
        entry["injection_log"] = "../elsewhere.json"

    rewrite(harness, escape)
    with pytest.raises(SweepFolderError, match="outside the sweep folder"):
        render_sweep_tables(harness.output_dir)


def fake_coverage(run_dir: Path) -> dict:
    return {
        "run_id": f"{run_dir.parent.parent.name}-{run_dir.name}",
        "sampling": [
            {
                "service_id": "nvidia-nvml",
                "sample_rounds": 9,
                "expected_rounds": 8,
                "late_rounds": 0,
                "missed_rounds": 1,
                "observed_interval_median_seconds": 0.25,
                "observed_interval_p90_seconds": 0.2512,
                "observed_interval_max_seconds": None,
            }
        ],
    }


def test_coverage_reads_the_first_run_of_the_first_ramp_batch_at_each_level(recorded, monkeypatch):
    harness, record = recorded
    read_dirs = []

    def coverage(run_dir):
        read_dirs.append(run_dir)
        return fake_coverage(run_dir)

    monkeypatch.setattr(sweep_record, "coverage_report", coverage)
    markdown, reports = render_coverage_tables(harness.output_dir)
    first_ramp = {}
    for entry in record["batches"]:
        if entry["phase"] == "ramp" and entry["runs"]:
            first_ramp.setdefault(entry["level"], entry)
    expected = [
        harness.output_dir / entry["folder"] / "runs" / Path(entry["runs"][0]["output_dir"]).name
        for entry in first_ramp.values()
    ]
    # Read under this sweep folder, never at the path the runner recorded.
    assert read_dirs == expected
    assert [report["level"] for report in reports] == list(first_ramp)
    table = rows(markdown, "### `cfg-default`")
    # A missing interval prints as n/a, and intervals keep three decimals.
    assert table[0][3:] == ["9 (8)", "0", "1", "0.250 / 0.251 / n/a"]


def test_sweep_record_prints_the_tables_and_writes_the_coverage_reports(
    recorded, monkeypatch, tmp_path, capsys
):
    harness, record = recorded
    monkeypatch.setattr(sweep_record, "coverage_report", fake_coverage)
    out = tmp_path / "coverage.jsonl"
    argv = ["sweep-record", str(harness.output_dir), "--coverage-jsonl", str(out)]
    assert cli.main(argv) == cli.EXIT_PASSED
    printed = capsys.readouterr().out
    assert printed.startswith("### cfg-default")
    assert "\n### `cfg-default`\n" in printed
    lines = out.read_bytes().split(b"\n")
    assert lines[-1] == b"" and b"\r" not in out.read_bytes()
    assert [json.loads(line)["sweep"] for line in lines[:-1]] == [harness.output_dir.name] * len(
        lines[:-1]
    )
    assert record["sweep_id"] in printed


def test_sweep_record_without_a_usable_record_exits_2(tmp_path, capsys):
    (tmp_path / sweep.SWEEP_FILE).write_text("not json")
    assert cli.main(["sweep-record", str(tmp_path)]) == cli.EXIT_USAGE
    assert capsys.readouterr().err.startswith("error: cannot read")
