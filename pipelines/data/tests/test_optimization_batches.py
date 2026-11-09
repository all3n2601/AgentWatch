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
    reading = read_batches(root.resolve(), manifest, schedule(manifest), CONTRACT)
    return reading.records, reading.rejections


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


def agent_evidence(run_id, task_id, content, digest, task_passed=True) -> dict:
    return {
        "run_id": run_id,
        "task": {"task_id": task_id, "content_sha256": content},
        "configuration_digest": digest,
        "task_passed": task_passed,
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
        lambda record: record["tasks"][0]["admission_waits"].update(test_jobs=float("inf")),
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


@pytest.mark.parametrize(
    ("condition", "injection"),
    [
        ("contended", {"active": False, "cleanup_confirmed": True}),
        ("contended", {"active": True, "cleanup_confirmed": None}),
        ("contended", {"active": None, "cleanup_confirmed": None}),
        ("clean", {"active": True, "cleanup_confirmed": True}),
    ],
)
def test_injection_must_match_the_condition(tmp_path, condition, injection):
    slot = first_slot(parse_manifest(manifest_document(CONTRACT), CONTRACT), condition)
    manifest = write_experiment(
        tmp_path,
        CONTRACT,
        change=only_changing(slot.index, lambda record: record["injection"].update(injection)),
    )
    _, rejections = read(tmp_path, manifest)
    assert rejected_codes(rejections) == {f"{slot.index:03d}.json": "injection_mismatch"}


def test_injection_flags_are_booleans_or_null(tmp_path):
    def edit(record):
        record["injection"]["cleanup_confirmed"] = "banana"

    manifest = write_experiment(tmp_path, CONTRACT, change=only_changing(2, edit))
    _, rejections = read(tmp_path, manifest)
    assert rejected_codes(rejections) == {"002.json": "malformed"}


def test_submissions_follow_the_preregistered_arrivals(tmp_path):
    def edit(record):
        record["tasks"][1]["submitted_tick_ns"] = 0  # preregistered at +1.5 s

    manifest = write_experiment(tmp_path, CONTRACT, change=only_changing(4, edit))
    _, rejections = read(tmp_path, manifest)
    assert rejected_codes(rejections) == {"004.json": "arrival_mismatch"}


def test_submissions_within_the_declared_tolerance_are_accepted(tmp_path):
    def edit(record):
        record["tasks"][1]["submitted_tick_ns"] += 200_000_000  # 0.2 s late; tolerance 0.25 s

    manifest = write_experiment(tmp_path, CONTRACT, change=only_changing(4, edit))
    assert read(tmp_path, manifest)[1] == []


@pytest.mark.parametrize(
    "edit",
    [
        lambda record: record.update(started_at="2026-99-99T99:99:99Z"),
        lambda record: record.update(started_at=None),
        lambda record: record.update(finished_at="2026-11-05T15:00:00Z"),
    ],
)
def test_batch_times_must_be_valid_and_ordered(tmp_path, edit):
    manifest = write_experiment(tmp_path, CONTRACT, change=only_changing(5, edit))
    _, rejections = read(tmp_path, manifest)
    assert set(rejected_codes(rejections)) == {"005.json"}
    assert rejected_codes(rejections)["005.json"] in ("malformed", "inconsistent_status")


def test_a_batch_overlapping_the_previous_one_is_rejected(tmp_path):
    def edit(record):
        record["started_at"] = "2026-11-05T15:11:25Z"  # batch 4 runs 15:11:20-15:11:30
        record["finished_at"] = "2026-11-05T15:11:45Z"

    manifest = write_experiment(tmp_path, CONTRACT, change=only_changing(5, edit))
    records, rejections = read(tmp_path, manifest)
    assert rejected_codes(rejections) == {"005.json": "order_mismatch"}
    assert 6 in {record.slot.index for record in records}  # later batches still compare


def test_a_never_started_batch_has_no_times(tmp_path):
    def edit(record):
        record.update(status="not_started", reason="Registration failed")
        for task in record["tasks"]:
            task.update(state="not_started", finished_tick_ns=None)

    manifest = write_experiment(tmp_path, CONTRACT, change=only_changing(6, edit))
    _, rejections = read(tmp_path, manifest)
    assert rejected_codes(rejections) == {"006.json": "inconsistent_status"}


def test_two_tasks_of_one_batch_cannot_share_a_run(tmp_path):
    def edit(record):
        record["tasks"][1]["run_id"] = record["tasks"][0]["run_id"]

    manifest = write_experiment(tmp_path, CONTRACT, change=only_changing(3, edit))
    _, rejections = read(tmp_path, manifest)
    assert rejected_codes(rejections) == {"003.json": "duplicate_identity"}


@pytest.mark.parametrize("pair_index", [True, 1.0, "1"])
def test_pair_index_is_a_strict_integer(tmp_path, pair_index):
    manifest = parse_manifest(manifest_document(CONTRACT), CONTRACT)
    slot = next(slot for slot in schedule(manifest) if slot.pair_index == 1)
    manifest = write_experiment(
        tmp_path,
        CONTRACT,
        change=only_changing(slot.index, lambda record: record.update(pair_index=pair_index)),
    )
    _, rejections = read(tmp_path, manifest)
    assert rejected_codes(rejections)[f"{slot.index:03d}.json"] in (
        "malformed",
        "schedule_mismatch",
    )


def test_linked_evidence_cannot_escape_with_dot_dot(tmp_path):
    write_json(tmp_path / "outside" / "agent-evidence.json", {})
    root = tmp_path / "experiment"

    def link(record):
        record["tasks"][0]["evidence_dir"] = "../outside"

    manifest = write_experiment(root, CONTRACT, change=only_changing(1, link))
    _, rejections = read(root, manifest)
    assert rejected_codes(rejections) == {"001.json": "task_identity_mismatch"}


def test_the_record_hash_is_of_the_bytes_that_were_parsed(tmp_path):
    import hashlib

    manifest = write_experiment(tmp_path, CONTRACT)
    records, _ = read(tmp_path, manifest)
    first = next(record for record in records if record.slot.index == 0)
    raw = (tmp_path / "batches" / "000.json").read_bytes()
    assert first.file_sha256 == hashlib.sha256(raw).hexdigest()


# --- The applied treatment must be the preregistered one ---------------------------------------


def contended_slot():
    return first_slot(parse_manifest(manifest_document(CONTRACT), CONTRACT), "contended")


@pytest.mark.parametrize(
    "applied",
    [
        [{"kind": "cpu", "method": "other-cpu-burner-v1", "intensity": {"workers": 4}}],
        [{"kind": "cpu", "method": "cpu-spin-processes-v1", "intensity": {"workers": 8}}],
        [],
        None,
    ],
)
def test_a_contended_batch_must_apply_the_preregistered_injectors(tmp_path, applied):
    slot = contended_slot()

    def edit(record):
        record["injection"]["injectors"] = applied

    manifest = write_experiment(tmp_path, CONTRACT, change=only_changing(slot.index, edit))
    _, rejections = read(tmp_path, manifest)
    assert rejected_codes(rejections) == {f"{slot.index:03d}.json": "injection_mismatch"}


def test_a_clean_batch_applies_no_injectors(tmp_path):
    slot = first_slot(parse_manifest(manifest_document(CONTRACT), CONTRACT), "clean")
    cpu = {"kind": "cpu", "method": "cpu-spin-processes-v1", "intensity": {"workers": 4}}

    def edit(record):
        record["injection"]["injectors"] = [cpu]

    manifest = write_experiment(tmp_path, CONTRACT, change=only_changing(slot.index, edit))
    _, rejections = read(tmp_path, manifest)
    assert rejected_codes(rejections) == {f"{slot.index:03d}.json": "injection_mismatch"}


def test_the_injector_set_matches_whatever_order_it_is_listed_in(tmp_path):
    cpu = {"kind": "cpu", "method": "cpu-spin-processes-v1", "intensity": {"workers": 4}}
    retrieval = {
        "kind": "retrieval",
        "method": "in-process-index-query-flood-v1",
        "intensity": {"threads": 2},
    }
    document = manifest_document(CONTRACT)
    document["conditions"][1]["injectors"] = [retrieval, cpu]
    slot = first_slot(parse_manifest(document, CONTRACT), "contended")

    def edit(scheduled, record):
        if scheduled.index == slot.index:
            record["injection"]["injectors"] = [cpu, retrieval]
        return record

    manifest = write_experiment(tmp_path, CONTRACT, document=document, change=edit)
    assert read(tmp_path, manifest)[1] == []


def test_a_batch_that_did_not_complete_need_not_record_its_injectors(tmp_path):
    slot = contended_slot()

    def edit(record):
        record.update(status="failed", reason="Injector failed to start")
        record["injection"].update(active=False, injectors=None)

    manifest = write_experiment(tmp_path, CONTRACT, change=only_changing(slot.index, edit))
    records, rejections = read(tmp_path, manifest)
    assert rejections == []
    assert next(record for record in records if record.slot.index == slot.index).status == "failed"


def test_malformed_applied_injectors_are_rejected(tmp_path):
    slot = contended_slot()

    def edit(record):
        record["injection"]["injectors"] = [{"kind": "gpu", "method": "x", "intensity": {"a": 1}}]

    manifest = write_experiment(tmp_path, CONTRACT, change=only_changing(slot.index, edit))
    _, rejections = read(tmp_path, manifest)
    assert rejected_codes(rejections) == {f"{slot.index:03d}.json": "malformed"}


# --- Every started task proves which task and configuration ran --------------------------------


def test_a_started_task_without_linked_evidence_is_rejected(tmp_path):
    def edit(record):
        record["tasks"][1]["evidence_dir"] = None

    manifest = write_experiment(tmp_path, CONTRACT, change=only_changing(3, edit))
    _, rejections = read(tmp_path, manifest)
    assert rejected_codes(rejections) == {"003.json": "provenance_missing"}


def test_a_task_that_never_started_needs_no_evidence(tmp_path):
    def edit(record):
        record.update(status="aborted", reason="Runner stopped before the second task")
        record["tasks"][1].update(
            state="not_started",
            acceptance=None,
            run_id=None,
            submitted_tick_ns=None,
            finished_tick_ns=None,
            evidence_dir=None,
        )

    manifest = write_experiment(tmp_path, CONTRACT, change=only_changing(3, edit))
    assert read(tmp_path, manifest)[1] == []


@pytest.mark.parametrize(
    ("evidence", "detail"),
    [
        ({"configuration_digest": "5" * 64}, "configuration_digest"),
        ({"task": {"task_id": TASKS[0][0], "content_sha256": "6" * 64}}, "content_sha256"),
        ({"task_passed": False}, "task_passed is False"),
        ({"task_passed": None}, "task_passed is None"),
    ],
)
def test_linked_evidence_must_agree_with_the_record(tmp_path, evidence, detail):
    task_id, content, digest = TASKS[0]

    def link(record):
        task = record["tasks"][0]
        write_json(
            tmp_path / task["evidence_dir"] / "agent-evidence.json",
            agent_evidence(task["run_id"], task_id, content, digest) | evidence,
        )

    manifest = write_experiment(tmp_path, CONTRACT, change=only_changing(5, link))
    _, rejections = read(tmp_path, manifest)
    assert rejected_codes(rejections) == {"005.json": "task_identity_mismatch"}
    assert detail in rejections[0].detail


def test_unknown_acceptance_must_match_a_run_that_recorded_no_outcome(tmp_path):
    def edit(record):
        record["tasks"][0]["acceptance"] = None  # the fixture's evidence then omits task_passed

    manifest = write_experiment(tmp_path, CONTRACT, change=only_changing(5, edit))
    assert read(tmp_path, manifest)[1] == []
