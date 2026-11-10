"""Replay experiments in the evidence store, tested with synthetic experiments only."""

import hashlib
import json
import shutil
import stat
from pathlib import Path

import pytest
from agentwatch_data import evidence as evidence_module
from agentwatch_data.evidence import (
    EXPERIMENT_FILE,
    EvidenceError,
    Stored,
    add,
    experiment_entry_name,
    main,
    read_stored_experiment,
)
from agentwatch_data.replay_evidence import ReplayEvidenceError, read_experiment
from agentwatch_data.replay_journal import read_journal
from agentwatch_features import compute_feature_rows
from replay_fixtures import (
    REFUSED_DETAIL,
    rejected_line,
    remote_closure_line,
    write_experiment,
)


def experiment(tmp_path: Path, **kwargs) -> Path:
    kwargs.setdefault("name", "attempt2")
    return write_experiment(tmp_path / "in", live=True, **kwargs)


def entry_files(directory: Path) -> dict[str, bytes]:
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def edit_lines(path: Path, change) -> None:
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    change(lines)
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))


def test_experiment_is_stored_byte_for_byte_under_its_content_name(tmp_path):
    exp = experiment(tmp_path)
    store = tmp_path / "store"
    stored = add(exp, store)
    digest = hashlib.sha256((exp / "experiment.json").read_bytes()).hexdigest()[:6]
    assert stored == Stored(f"replay-fix-a-r1-20261105T000000Z-{digest}", added=True)
    entry = store / stored.name
    assert entry_files(entry) == entry_files(exp)
    assert [p.name for p in store.iterdir()] == [stored.name]  # no staging left behind
    assert {stat.S_IMODE(p.stat().st_mode) for p in entry.rglob("*") if p.is_dir()} == {0o755}
    assert {stat.S_IMODE(p.stat().st_mode) for p in entry.rglob("*") if p.is_file()} == {0o644}
    assert stat.S_IMODE(entry.stat().st_mode) == 0o755


def test_stored_experiment_reads_back_with_every_file_hash(tmp_path):
    exp = experiment(tmp_path)
    stored = add(exp, tmp_path / "store")
    read, hashes = read_stored_experiment(tmp_path / "store" / stored.name)
    assert read.name == stored.name
    assert hashes == {
        name: hashlib.sha256(data).hexdigest() for name, data in entry_files(exp).items()
    }
    assert all(EXPERIMENT_FILE.fullmatch(name) for name in hashes)
    assert experiment_entry_name((exp / "experiment.json").read_bytes(), read) == stored.name


def test_entry_name_uses_utc_start_time(tmp_path):
    exp = experiment(tmp_path, started_at="2026-11-05T12:15:21.7+05:30")
    stored = add(exp, tmp_path / "store")
    assert stored.name.startswith("replay-fix-a-r1-20261105T064521Z-")


def test_same_experiment_under_another_folder_name_is_already_stored(tmp_path, capsys):
    exp = experiment(tmp_path)
    store = tmp_path / "store"
    first = add(exp, store)
    copy = shutil.copytree(exp, tmp_path / "redelivered" / "attempt2-copy")
    assert add(copy, store) == Stored(first.name, added=False)
    assert main(["--store", str(store), str(copy)]) == 0
    assert capsys.readouterr().out == f"already stored: {store / first.name}\n"
    assert [p.name for p in store.iterdir()] == [first.name]


def test_protocol_failed_experiment_is_stored(tmp_path):
    """Labeling rejects it with a recorded reason; the failed run is still evidence."""
    checks = {
        "baseline_replays_complete": True,
        "stable_baseline": False,
        "same_steps_every_replay": True,
        "injectors_cleaned_up": True,
    }
    exp = experiment(tmp_path, checks=checks, passed=False)
    stored = add(exp, tmp_path / "store")
    assert stored.added is True
    read, _ = read_stored_experiment(tmp_path / "store" / stored.name)
    assert (read.status, read.checks["stable_baseline"], read.passed) == ("completed", False, False)


def add_extra_file(exp: Path, tmp_path: Path) -> None:
    (exp / "replays" / "01" / "vllm-metrics.txt").write_text("vllm:num_requests_running 1\n")


def symlink_journal(exp: Path, tmp_path: Path) -> None:
    journal = exp / "replays" / "01" / "live-journal.jsonl"
    elsewhere = tmp_path / "elsewhere.jsonl"
    elsewhere.write_bytes(journal.read_bytes())
    journal.unlink()
    journal.symlink_to(elsewhere)


def truncate_last_journal_line(exp: Path, tmp_path: Path) -> None:
    journal = exp / "replays" / "01" / "live-journal.jsonl"
    journal.write_bytes(journal.read_bytes()[:-20])


def record_for_another_run(exp: Path, tmp_path: Path) -> None:
    def change(lines):
        lines[-2]["payload"]["run_id"] = "00000000-0000-4000-8000-999999999999"

    edit_lines(exp / "replays" / "01" / "live-journal.jsonl", change)


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        (add_extra_file, "replays/01/vllm-metrics.txt is not an experiment evidence file"),
        (symlink_journal, "replays/01/live-journal.jsonl is a symlink"),
        (truncate_last_journal_line, r"live-journal.jsonl line \d+"),
        (record_for_another_run, r"live-journal.jsonl line \d+: .*another run"),
    ],
    ids=["extra-file", "symlink", "truncated-line", "other-run-record"],
)
def test_extra_file_symlink_or_bad_journal_line_is_refused_and_nothing_staged(
    tmp_path, damage, message
):
    exp = experiment(tmp_path)
    damage(exp, tmp_path)
    store = tmp_path / "store"
    with pytest.raises(EvidenceError, match=message):
        add(exp, store)
    assert not store.exists() or list(store.iterdir()) == []  # no entry, no hidden staging


def test_different_content_under_a_stored_name_is_refused(tmp_path):
    exp = experiment(tmp_path)
    store = tmp_path / "store"
    stored = add(exp, store)
    journal = store / stored.name / "source" / "live-journal.jsonl"

    def change(lines):
        lines[-2]["payload"]["value"] = 11.0  # still a valid reading, but not the evidence

    edit_lines(journal, change)
    changed = journal.read_bytes()
    with pytest.raises(EvidenceError, match="already stored with different evidence"):
        add(exp, store)
    assert journal.read_bytes() == changed  # the stored entry is reported, never overwritten


def test_a_damaged_stored_experiment_is_reported_as_unreadable(tmp_path):
    exp = experiment(tmp_path)
    stored = add(exp, tmp_path / "store")
    (tmp_path / "store" / stored.name / "replays" / "02" / "live-journal.jsonl").write_text("{")
    with pytest.raises(EvidenceError, match=f"Stored evidence {stored.name} is unreadable"):
        add(exp, tmp_path / "store")


def test_stored_experiment_with_an_extra_file_is_refused(tmp_path):
    exp = experiment(tmp_path)
    stored = add(exp, tmp_path / "store")
    (tmp_path / "store" / stored.name / "notes.txt").write_text("not evidence")
    with pytest.raises(EvidenceError, match="notes.txt is not an experiment evidence file"):
        read_stored_experiment(tmp_path / "store" / stored.name)


def add_hidden_files(directory: Path) -> None:
    (directory / ".DS_Store").write_text("")
    (directory / "replays" / "02" / "._live-journal.jsonl").write_text("resource fork")
    (directory / ".git").mkdir()
    (directory / ".git" / "HEAD").write_text("ref: refs/heads/main\n")


def test_a_stored_experiment_skips_hidden_files(tmp_path):
    exp = experiment(tmp_path)
    stored = add(exp, tmp_path / "store")
    entry = tmp_path / "store" / stored.name
    expected = read_stored_experiment(entry)
    add_hidden_files(entry)
    assert read_stored_experiment(entry) == expected
    assert add(exp, tmp_path / "store") == Stored(stored.name, added=False)


def test_a_stored_manifest_naming_a_hidden_directory_is_refused(tmp_path):
    """A hidden directory is skipped, so evidence the manifest reads from it has no hash."""
    exp = experiment(tmp_path)
    stored = add(exp, tmp_path / "store")
    entry = tmp_path / "store" / stored.name
    shutil.move(entry / "replays" / "02", entry / "replays" / ".02")
    manifest = json.loads((entry / "experiment.json").read_text())
    manifest["replays"][2]["evidence_dir"] = "replays/.02"
    (entry / "experiment.json").write_text(json.dumps(manifest))
    with pytest.raises(EvidenceError, match=r"names hidden path 'replays/\.02'"):
        read_stored_experiment(entry)


def test_hidden_files_in_a_delivery_folder_are_never_stored(tmp_path):
    exp = experiment(tmp_path)
    clean = entry_files(exp)
    add_hidden_files(exp)
    stored = add(exp, tmp_path / "store")
    assert stored.added is True
    entry = tmp_path / "store" / stored.name
    assert entry_files(entry) == clean  # byte for byte, and nothing hidden was copied
    assert not [path for path in entry.rglob(".*")]


def test_a_copy_that_does_not_match_what_was_read_is_never_published(tmp_path, monkeypatch):
    real = evidence_module.fsync_file

    def corrupt(path):
        if path.name == "live-journal.jsonl" and path.parent.name == "source":
            # Still a valid journal, so only the byte comparison can catch it.
            edit_lines(path, lambda lines: lines[-2]["payload"].update(value=11.0))
        real(path)

    monkeypatch.setattr(evidence_module, "fsync_file", corrupt)
    with pytest.raises(EvidenceError, match="does not match the evidence read"):
        add(experiment(tmp_path), tmp_path / "store")
    assert list((tmp_path / "store").iterdir()) == []


def test_a_concurrent_add_of_the_same_experiment_is_accepted(tmp_path, monkeypatch):
    real_replace = evidence_module.os.replace

    def another_add_finishes_first(source, target):
        real_replace(source, target)  # the other command's identical copy lands first
        raise OSError("target exists")

    monkeypatch.setattr(evidence_module.os, "replace", another_add_finishes_first)
    stored = add(experiment(tmp_path), tmp_path / "store")
    assert stored.added is False
    assert [p.name for p in (tmp_path / "store").iterdir()] == [stored.name]


def test_missing_started_at_is_refused(tmp_path):
    exp = experiment(tmp_path)
    manifest = json.loads((exp / "experiment.json").read_text())
    del manifest["started_at"]
    (exp / "experiment.json").write_text(json.dumps(manifest))
    with pytest.raises(EvidenceError, match="started_at"):
        add(exp, tmp_path / "store")
    assert list((tmp_path / "store").iterdir()) == []


@pytest.mark.parametrize("started_at", ["2026-11-05T00:00:00", "yesterday", 5])
def test_started_at_must_be_a_timezone_aware_timestamp(tmp_path, started_at):
    exp = experiment(tmp_path)
    manifest = json.loads((exp / "experiment.json").read_text())
    manifest["started_at"] = started_at
    (exp / "experiment.json").write_text(json.dumps(manifest))
    with pytest.raises(ReplayEvidenceError, match="started_at"):
        read_experiment(exp)


def test_passed_stays_null_when_not_recorded(tmp_path):
    read = read_experiment(experiment(tmp_path, passed=None))
    assert read.passed is None  # null is never coerced to False


@pytest.mark.parametrize("task_id", ["../escape", "a/b", ".hidden"])
def test_task_id_that_is_not_a_safe_entry_name_is_refused(tmp_path, task_id):
    exp = experiment(tmp_path, task_id=task_id)
    with pytest.raises(EvidenceError, match="task_id"):
        add(exp, tmp_path / "store")
    assert list((tmp_path / "store").iterdir()) == []


def test_a_run_directory_still_goes_to_add_run(tmp_path):
    with pytest.raises(EvidenceError, match="is not usable evidence"):
        add(tmp_path, tmp_path / "store")


def test_live_journal_reads_as_sdk_records_with_its_configuration(tmp_path):
    exp = experiment(tmp_path)
    journal = read_journal(exp / "replays" / "00" / "live-journal.jsonl")
    evidence = json.loads((exp / "replays" / "00" / "agent-evidence.json").read_text())
    assert journal.configuration_sha256 == evidence["configuration_digest"]
    assert {event.state for event in journal.events} == {"queued", "running", "completed"}
    assert {sample.metric for sample in journal.samples} == {"cpu.psi.some.avg10.v1"}
    assert (exp / "source" / "live-journal.jsonl").read_bytes() == (
        exp / "replays" / "00" / "live-journal.jsonl"
    ).read_bytes()


def test_live_journal_ticks_reproduce_the_manifest_timings(tmp_path):
    exp = experiment(tmp_path, contended={"cpu": {"agent-run": 1.5, "tool-call-1": 0.3}})
    read = read_experiment(exp)
    for replay in read.replays:
        directory = exp / "replays" / f"{replay.index:02d}"
        journal = read_journal(directory / "live-journal.jsonl")
        evidence = json.loads((directory / "agent-evidence.json").read_text())
        interval = evidence["configuration"]["sampling_interval_seconds"]
        rows = compute_feature_rows(journal.events, journal.samples, interval)
        assert rows.incomplete_step_ids == []
        by_step = {row.step_id: row for row in rows.rows}
        assert by_step.keys() == replay.steps.keys()
        for step_id, timing in replay.steps.items():
            row = by_step[step_id]
            assert str(row.run_id) == replay.run_id
            assert row.timings.duration_seconds == timing.total_seconds
            assert row.timings.queue_seconds == timing.queue_seconds
            assert row.timings.execution_seconds == timing.execution_seconds
        assert by_step["tool-call-1"].telemetry.status == "present"


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        ([{"kind": "heartbeat", "payload": {}}], "line 1: unknown kind 'heartbeat'"),
        (
            ["", {"kind": "record", "payload": {"record_type": "feature_row"}}],
            "line 2: unknown record_type 'feature_row'",
        ),
        ([["not", "an", "object"]], "line 1: must be an object"),
        ([{"kind": "run_created", "payload": None}], "line 1: must be an object"),
        ([{"kind": "registration", "payload": {}}], "line 1: registration is invalid"),
    ],
    ids=["kind", "record-type", "not-object", "no-payload", "bad-registration"],
)
def test_journal_refuses_unknown_or_malformed_lines(tmp_path, lines, message):
    journal = tmp_path / "live-journal.jsonl"
    journal.write_text("".join((json.dumps(line) if line else "") + "\n" for line in lines))
    with pytest.raises(ReplayEvidenceError, match=message):
        read_journal(journal)


def test_journal_refuses_a_second_registration(tmp_path):
    path = experiment(tmp_path) / "replays" / "00" / "live-journal.jsonl"
    edit_lines(path, lambda lines: lines.append(lines[0]))
    with pytest.raises(ReplayEvidenceError, match=r"line \d+: the run is registered twice"):
        read_journal(path)


def test_journal_refuses_a_record_that_fails_the_sdk_contract(tmp_path):
    exp = experiment(tmp_path)
    path = exp / "replays" / "00" / "live-journal.jsonl"

    def change(lines):
        lines[-2]["payload"]["value"] = 150.0  # pressure is a percentage

    edit_lines(path, change)
    with pytest.raises(ReplayEvidenceError, match=r"line \d+: resource_sample"):
        read_journal(path)


def record_ids(lines: list[dict]) -> list[str]:
    return [
        line["payload"].get("event_id") or line["payload"]["sample_id"]
        for line in lines
        if line["kind"] == "record"
    ]


def test_delivery_outcome_lines_are_read_and_their_records_still_count(tmp_path):
    """A refused delivery is a delivery outcome; the journal still recorded the measurement."""
    path = experiment(tmp_path) / "replays" / "04" / "live-journal.jsonl"
    before = read_journal(path)

    def change(lines):
        refused = record_ids(lines)[-3:]
        lines.insert(-1, rejected_line(refused[:2]))
        lines.insert(-1, rejected_line(refused[2:]))
        lines.append(remote_closure_line("cancelled", "Cancelled from the dashboard"))
        lines.append(remote_closure_line("completed", None))

    edit_lines(path, change)
    after = read_journal(path)
    assert after == before  # every record line still feeds the features


def test_an_experiment_with_refused_deliveries_is_added(tmp_path):
    exp = experiment(tmp_path)
    for index in ("00", "04"):
        edit_lines(
            exp / "replays" / index / "live-journal.jsonl",
            lambda lines: lines.insert(-1, rejected_line(record_ids(lines)[-1:])),
        )
    stored = add(exp, tmp_path / "store")
    assert stored.added is True
    assert read_stored_experiment(tmp_path / "store" / stored.name)[1] == {
        name: hashlib.sha256(data).hexdigest() for name, data in entry_files(exp).items()
    }


ONE_ID = "00000000-0000-4000-8000-000000000001"


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ({"kind": "rejected", "payload": {"status": 422, "detail": "x"}}, "record_ids"),
        (rejected_line([]), "record_ids"),
        (rejected_line(["not-a-uuid"]), "record_ids"),
        (rejected_line([ONE_ID.replace("-", "")]), "record_ids"),
        (rejected_line([ONE_ID, ONE_ID]), "record_ids"),
        (rejected_line([7]), "record_ids"),
        (rejected_line([ONE_ID]) | {"replayed": True}, "unexpected keys"),
        (rejected_line([ONE_ID], status=True), "status"),
        (rejected_line([ONE_ID], status=99), "status"),
        (
            {
                "kind": "rejected",
                "payload": {"status": "422", "detail": "x"},
                "record_ids": [ONE_ID],
            },
            "status",
        ),
        (
            {"kind": "rejected", "payload": {"status": 422, "detail": 5}, "record_ids": [ONE_ID]},
            "detail",
        ),
        (
            {
                "kind": "rejected",
                "payload": {"status": 422, "detail": REFUSED_DETAIL, "extra": 1},
                "record_ids": [ONE_ID],
            },
            "payload keys",
        ),
        (remote_closure_line("closed", None), "state"),
        (remote_closure_line("cancelled", 5), "reason"),
        ({"kind": "remote_closure", "payload": {"state": "failed"}}, "payload keys"),
        (remote_closure_line() | {"record_ids": [ONE_ID]}, "unexpected keys"),
    ],
    ids=[
        "no-record-ids",
        "empty-record-ids",
        "record-id-not-uuid",
        "record-id-not-canonical",
        "repeated-record-id",
        "record-id-not-string",
        "rejected-extra-key",
        "status-bool",
        "status-out-of-range",
        "status-string",
        "detail-not-string",
        "rejected-payload-extra",
        "closure-unknown-state",
        "closure-reason-not-string",
        "closure-missing-reason",
        "closure-extra-key",
    ],
)
def test_malformed_delivery_outcome_lines_are_refused(tmp_path, line, message):
    path = experiment(tmp_path) / "replays" / "00" / "live-journal.jsonl"
    edit_lines(path, lambda lines: lines.insert(-1, line))
    with pytest.raises(ReplayEvidenceError, match=rf"line \d+: {line['kind']} .*{message}"):
        read_journal(path)


def test_edited_manifest_cannot_store_the_same_replay_runs_twice(tmp_path):
    exp = experiment(tmp_path)
    store = tmp_path / "store"
    first = add(exp, store)
    manifest = exp / "experiment.json"
    manifest.write_text(manifest.read_text() + "\n")
    with pytest.raises(EvidenceError, match="already stored"):
        add(exp, store)
    assert [p.name for p in store.iterdir()] == [first.name]


def test_concurrent_publication_fails_without_leaving_staged_evidence(tmp_path):
    exp = experiment(tmp_path)
    store = tmp_path / "store"
    store.mkdir()
    lock = store / ".replay-publication-lock"
    lock.mkdir()
    with pytest.raises(EvidenceError, match="retry later"):
        add(exp, store)
    assert list(store.iterdir()) == [lock]
