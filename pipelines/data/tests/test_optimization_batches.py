"""Batch records: incompatible identities and unscheduled or conflicting records are rejected."""

import json
import os

import pytest
from agentwatch_data.optimization_batches import read_batches
from agentwatch_data.optimization_manifest import CANDIDATE, ManifestError, parse_manifest, schedule
from optimization_fixtures import (
    TASKS,
    contract,
    first_slot,
    manifest_document,
    write_experiment,
    write_json,
)

CONTRACT = contract()


def read(root, manifest):
    return read_batches(root.resolve(), manifest, schedule(manifest), CONTRACT)


def rejected_codes(rejections) -> dict[str, str]:
    return {rejection.file: rejection.code for rejection in rejections}


def only_changing(target_index, edit):
    def change(slot, record):
        if slot.index == target_index:
            edit(record)
        return record

    return change


def test_a_complete_experiment_is_accepted_with_every_identity_kept(tmp_path):
    manifest = write_experiment(tmp_path, CONTRACT)
    records, rejections = read(tmp_path, manifest)
    assert rejections == []
    assert len(records) == len(schedule(manifest))
    assert all(len(record.file_sha256) == 64 for record in records)
    run_ids = [task.run_id for record in records for task in record.tasks]
    assert len(set(run_ids)) == len(run_ids)


def test_a_manifest_edited_after_the_batches_ran_matches_none_of_them(tmp_path):
    write_experiment(tmp_path, CONTRACT)
    document = manifest_document(CONTRACT, created_at="2026-11-05T16:00:00Z")
    write_json(tmp_path / "manifest.json", document)
    records, rejections = read(tmp_path, parse_manifest(document, CONTRACT))
    assert records == []
    assert set(rejected_codes(rejections).values()) == {"manifest_mismatch"}


@pytest.mark.parametrize(
    ("section", "key", "value", "code"),
    [
        ("hardware", "host_id", "another-laptop", "hardware_mismatch"),
        ("hardware", "gpus", [], "hardware_mismatch"),
        ("model", "digest", "0000aaaa", "model_mismatch"),
        ("model", "backend", "vllm", "model_mismatch"),
        ("power", "source", "battery", "power_mismatch"),
        ("state", "retrieval", "docs-index-r2", "state_mismatch"),
    ],
)
def test_incompatible_identities_are_rejected(tmp_path, section, key, value, code):
    target = 3

    def edit(record):
        record[section][key] = value

    manifest = write_experiment(tmp_path, CONTRACT, change=only_changing(target, edit))
    records, rejections = read(tmp_path, manifest)
    assert rejected_codes(rejections) == {f"{target:03d}.json": code}
    assert target not in {record.slot.index for record in records}


def test_records_outside_their_schedule_slot_are_rejected(tmp_path):
    def edit(record):
        record["configuration"] = "baseline" if record["configuration"] == CANDIDATE else CANDIDATE

    manifest = write_experiment(tmp_path, CONTRACT, change=only_changing(5, edit))
    write_json(
        tmp_path / "batches" / "999.json",
        json.loads((tmp_path / "batches" / "004.json").read_text()) | {"index": 999},
    )
    _, rejections = read(tmp_path, manifest)
    assert rejected_codes(rejections) == {
        "005.json": "schedule_mismatch",
        "999.json": "schedule_mismatch",
    }


def test_a_batch_must_apply_its_slot_policy(tmp_path):
    manifest = write_experiment(tmp_path, CONTRACT)
    candidate = first_slot(manifest, "contended", CANDIDATE)
    path = tmp_path / "batches" / f"{candidate.index:03d}.json"
    record = json.loads(path.read_text())
    record["policy"] = {"sha256": manifest.baseline.sha256, "limits": {}, "version": 0}
    write_json(path, record)
    forged = first_slot(manifest, "clean", CANDIDATE)
    forged_path = tmp_path / "batches" / f"{forged.index:03d}.json"
    record = json.loads(forged_path.read_text())
    record["policy"]["limits"] = {"inference_requests": 2}  # hash still names limit 1
    write_json(forged_path, record)
    _, rejections = read(tmp_path, manifest)
    assert rejected_codes(rejections) == {
        path.name: "policy_mismatch",
        forged_path.name: "policy_mismatch",
    }


def test_every_frozen_task_runs_exactly_once_per_batch(tmp_path):
    def edit(record):
        record["tasks"] = record["tasks"][:1]

    manifest = write_experiment(tmp_path, CONTRACT, change=only_changing(2, edit))
    _, rejections = read(tmp_path, manifest)
    assert rejected_codes(rejections) == {"002.json": "task_mismatch"}


def agent_evidence(run_id, task_id, content, digest) -> dict:
    return {
        "run_id": run_id,
        "task": {"task_id": task_id, "content_sha256": content},
        "configuration_digest": digest,
    }


def test_linked_agent_evidence_must_be_the_frozen_tasks_run(tmp_path):
    task_id, content, digest = TASKS[0]

    def link(record):
        task = record["tasks"][0]
        task["evidence_dir"] = f"runs/{record['index']}"
        content_seen = "9" * 64 if record["index"] == 4 else content
        write_json(
            tmp_path / "runs" / str(record["index"]) / "agent-evidence.json",
            agent_evidence(task["run_id"], task_id, content_seen, digest),
        )
        return record

    manifest = write_experiment(
        tmp_path,
        CONTRACT,
        change=lambda slot, record: link(record) if slot.index in (3, 4) else record,
    )
    records, rejections = read(tmp_path, manifest)
    assert rejected_codes(rejections) == {"004.json": "task_identity_mismatch"}
    linked = next(record for record in records if record.slot.index == 3)
    assert linked.tasks[0].evidence_linked


def test_linked_evidence_must_stay_inside_the_experiment(tmp_path):
    outside = tmp_path / "outside"
    write_json(outside / "agent-evidence.json", {})
    root = tmp_path / "experiment"

    def link(record):
        record["tasks"][0]["evidence_dir"] = "linked"
        return record

    root.mkdir()
    os.symlink(outside, root / "linked")
    manifest = write_experiment(
        root, CONTRACT, change=lambda slot, record: link(record) if slot.index == 1 else record
    )
    _, rejections = read(root, manifest)
    assert rejected_codes(rejections) == {"001.json": "task_identity_mismatch"}


def test_records_sharing_a_batch_or_run_identity_are_all_rejected(tmp_path):
    def edit(slot, record):
        if slot.index in (1, 2):
            record["batch_id"] = "00000000-0000-4000-8000-000000000001"
        if slot.index in (6, 9):
            record["tasks"][0]["run_id"] = "00000000-0000-4000-8000-000000000002"
        return record

    manifest = write_experiment(tmp_path, CONTRACT, change=edit)
    records, rejections = read(tmp_path, manifest)
    assert rejected_codes(rejections) == {
        "001.json": "duplicate_identity",
        "002.json": "duplicate_identity",
        "006.json": "duplicate_identity",
        "009.json": "duplicate_identity",
    }
    assert {record.slot.index for record in records}.isdisjoint({1, 2, 6, 9})


def test_unreadable_or_linked_records_are_rejected_not_skipped(tmp_path):
    manifest = write_experiment(tmp_path, CONTRACT)
    batches = tmp_path / "batches"
    (batches / "003.json").unlink()
    os.symlink(batches / "004.json", batches / "003.json")
    (batches / "007.json").write_text("{not json")
    (batches / "notes.txt").write_text("hello")
    _, rejections = read(tmp_path, manifest)
    assert rejected_codes(rejections) == {
        "003.json": "malformed",
        "007.json": "malformed",
        "notes.txt": "malformed",
    }


@pytest.mark.parametrize(
    ("edit", "code"),
    [
        (
            lambda record: record["tasks"][0].update(state="not_started", finished_tick_ns=None),
            "inconsistent_status",
        ),
        (lambda record: record.update(status="failed"), "inconsistent_status"),
        (
            lambda record: record["tasks"][1].update(finished_tick_ns=0, submitted_tick_ns=5),
            "malformed",
        ),
        (lambda record: record["tasks"][0].update(acceptance="unknown"), "malformed"),
        (lambda record: record["admission"].update(leaked_permits=-1), "malformed"),
        (lambda record: record.update(clock_id=None), "malformed"),
    ],
)
def test_internally_inconsistent_records_are_rejected(tmp_path, edit, code):
    manifest = write_experiment(tmp_path, CONTRACT, change=only_changing(8, edit))
    _, rejections = read(tmp_path, manifest)
    assert rejected_codes(rejections) == {"008.json": code}


def test_a_failed_batch_is_kept_with_its_reason(tmp_path):
    def edit(record):
        record.update(status="failed", reason="Injector cleanup was not confirmed")
        record["tasks"][1]["state"] = "failed"

    manifest = write_experiment(tmp_path, CONTRACT, change=only_changing(8, edit))
    records, rejections = read(tmp_path, manifest)
    assert rejections == []
    failed = next(record for record in records if record.slot.index == 8)
    assert (failed.status, failed.reason) == ("failed", "Injector cleanup was not confirmed")


def test_the_batches_directory_must_be_a_real_directory(tmp_path):
    manifest = write_experiment(tmp_path / "plain", CONTRACT)
    empty = tmp_path / "empty"
    write_json(empty / "manifest.json", {})
    assert read(empty, manifest) == ([], [])
    (tmp_path / "file").mkdir()
    (tmp_path / "file" / "batches").write_text("not a directory")
    with pytest.raises(ManifestError, match="must be a directory"):
        read(tmp_path / "file", manifest)
    linked = tmp_path / "linked"
    linked.mkdir()
    os.symlink(tmp_path / "plain" / "batches", linked / "batches")
    with pytest.raises(ManifestError, match="symlink"):
        read(linked, manifest)


@pytest.mark.parametrize(
    "edit",
    [
        lambda record: record.update(hardware={"host_id": "only"}),
        lambda record: record["policy"].update(limits={"inference_requests": 0}),
        lambda record: record.update(batch_id="not-a-uuid"),
        lambda record: record["tasks"][0].update(run_id=7),
        lambda record: record.update(finished_at="yesterday"),
        lambda record: record["tasks"][0].update(admission_wait_seconds=float("inf")),
    ],
)
def test_malformed_fields_reject_only_their_record(tmp_path, edit):
    manifest = write_experiment(tmp_path, CONTRACT, change=only_changing(6, edit))
    records, rejections = read(tmp_path, manifest)
    assert rejected_codes(rejections) == {"006.json": "malformed"}
    assert len(records) == len(schedule(manifest)) - 1


def test_a_batch_that_never_started_has_no_started_task(tmp_path):
    def edit(record):
        record.update(status="not_started", reason="Runner could not register")
        record["tasks"][0].update(state="not_started", finished_tick_ns=None)

    manifest = write_experiment(tmp_path, CONTRACT, change=only_changing(6, edit))
    _, rejections = read(tmp_path, manifest)
    assert rejected_codes(rejections) == {"006.json": "inconsistent_status"}


def test_linked_evidence_without_a_run_id_is_not_the_tasks_run(tmp_path):
    task_id, content, digest = TASKS[0]

    def link(record):
        task = record["tasks"][0]
        task.update(evidence_dir="runs/one", run_id=None)
        write_json(
            tmp_path / "runs" / "one" / "agent-evidence.json",
            agent_evidence(None, task_id, content, digest),
        )
        return record

    manifest = write_experiment(
        tmp_path, CONTRACT, change=lambda slot, record: link(record) if slot.index == 2 else record
    )
    _, rejections = read(tmp_path, manifest)
    assert rejected_codes(rejections) == {"002.json": "task_identity_mismatch"}
    write_json(tmp_path / "runs" / "one" / "agent-evidence.json", [])
    _, rejections = read(tmp_path, manifest)
    assert rejected_codes(rejections) == {"002.json": "task_identity_mismatch"}
