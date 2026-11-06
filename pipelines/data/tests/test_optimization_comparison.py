"""Comparison records: only complete pairs are compared, every attempt is kept, and missing
measurements stay missing."""

import json

import pytest
from agentwatch_data import optimization_comparison as comparison_module
from agentwatch_data.files import json_text
from agentwatch_data.optimization_comparison import (
    bootstrap_median,
    compare,
    main,
    nearest_rank,
)
from agentwatch_data.optimization_manifest import (
    BASELINE,
    CANDIDATE,
    DESIGN_FIELDS,
    parse_manifest,
    schedule,
)
from agentwatch_data.schema import COLUMNS, REPLAY_LABEL_COLUMNS, Role
from agentwatch_features.live import FEATURE_NAMES
from optimization_fixtures import (
    CONTRACT_PATH,
    SECOND_NS,
    TASKS,
    contract,
    default_timing,
    first_slot,
    manifest_document,
    write_experiment,
)

CONTRACT = contract()


def planned():
    """The fixture manifest, parsed without writing any batch."""
    return parse_manifest(manifest_document(CONTRACT), CONTRACT)


def entry(report, condition) -> dict:
    return next(item for item in report["comparisons"] if item["condition"] == condition)


def test_paired_effects_on_a_complete_experiment(tmp_path):
    write_experiment(tmp_path, CONTRACT)
    report = compare(tmp_path, CONTRACT)
    assert report["missing_batches"] == []
    assert all(batch["accepted"] for batch in report["batches"])
    contended = entry(report, "contended")
    assert contended["pairs"] == {"scheduled": 10, "complete": 10, "unpaired": []}
    assert contended["sufficient"] and contended["insufficient_reasons"] == []
    effects = contended["effects"]
    # 10 s baseline batches against 8 s candidate batches in every pair.
    assert effects["median_reduction"] == pytest.approx(0.2)
    assert effects["reduction_interval"]["low"] == pytest.approx(0.2)
    assert effects["reduction_interval"]["high"] == pytest.approx(0.2)
    assert effects["reduction_interval"]["seed"] == 20261105
    assert effects["throughput_ratio"] == pytest.approx(1.25)
    assert effects["baseline"]["completed_tasks"] == 20
    assert effects["new_failures"] == effects["newly_failing_acceptance"] == 0
    clean = entry(report, "clean")
    assert clean["effects"]["median_reduction"] == pytest.approx(0)
    assert clean["effects"]["throughput_ratio"] == pytest.approx(1)
    for item in report["comparisons"]:
        assert item["gates"]["status"] == "not_evaluated"
        assert item["candidate_policy_sha256"] != item["baseline_policy_sha256"]


def test_the_record_is_deterministic_and_names_its_lineage(tmp_path):
    manifest = write_experiment(tmp_path, CONTRACT)
    first, second = compare(tmp_path, CONTRACT), compare(tmp_path, CONTRACT)
    assert json_text(first) == json_text(second)
    assert first["manifest_sha256"] == manifest.sha256
    assert first["contract"] == {
        "version": "opt-contract-v1",
        "status": CONTRACT.status,
        "sha256": CONTRACT.sha256[:8],
    }
    assert first["experiment_id"] == manifest.experiment_id
    batch_ids = [batch["batch_id"] for batch in first["batches"]]
    assert len(set(batch_ids)) == len(batch_ids) == len(schedule(manifest))


def test_a_failed_candidate_batch_leaves_its_pair_unpaired_but_listed(tmp_path):
    manifest = planned()
    slot = first_slot(manifest, "contended", CANDIDATE)

    def change(scheduled, record):
        if scheduled.index == slot.index:
            record.update(status="failed", reason="Model server stopped responding")
            record["tasks"][0]["state"] = "failed"
        return record

    write_experiment(tmp_path, CONTRACT, change=change)
    report = compare(tmp_path, CONTRACT)
    contended = entry(report, "contended")
    assert contended["pairs"]["complete"] == 9
    (unpaired,) = contended["pairs"]["unpaired"]
    assert unpaired["pair_id"] == slot.pair_id
    assert f"candidate batch {slot.index} is failed" in unpaired["reason"]
    kept = next(batch for batch in report["batches"] if batch["index"] == slot.index)
    assert (kept["status"], kept["reason"]) == ("failed", "Model server stopped responding")
    # 9 pairs of 2 tasks complete 18 per configuration: below the contract's 20.
    assert not contended["sufficient"]
    assert any("18 completed" in reason for reason in contended["insufficient_reasons"])


def test_missing_and_rejected_batches_make_pairs_unpaired(tmp_path):
    manifest = planned()
    missing = first_slot(manifest, "clean", BASELINE)
    rejected = first_slot(manifest, "contended", BASELINE)

    def change(scheduled, record):
        if scheduled.index == missing.index:
            return None
        if scheduled.index == rejected.index:
            record["hardware"]["host_id"] = "another-laptop"
        return record

    write_experiment(tmp_path, CONTRACT, change=change)
    report = compare(tmp_path, CONTRACT)
    assert report["missing_batches"] == [missing.index]
    (clean,) = entry(report, "clean")["pairs"]["unpaired"]
    assert clean["reason"] == f"baseline batch {missing.index} has no record"
    (contended,) = entry(report, "contended")["pairs"]["unpaired"]
    assert "rejected (hardware_mismatch)" in contended["reason"]
    rejected_entry = next(
        batch for batch in report["batches"] if batch["file"] == f"{rejected.index:03d}.json"
    )
    assert rejected_entry == {
        "file": f"{rejected.index:03d}.json",
        "sha256": rejected_entry["sha256"],
        "accepted": False,
        "code": "hardware_mismatch",
        "detail": "hardware differs from the manifest in ['host_id']",
    }


def test_missing_task_ticks_never_become_a_batch_time(tmp_path):
    manifest = planned()
    slot = first_slot(manifest, "contended", CANDIDATE)

    def timing(scheduled, task_id):
        if scheduled.index == slot.index and task_id == TASKS[1][0]:
            return None
        return default_timing(scheduled, task_id)

    write_experiment(tmp_path, CONTRACT, timing=timing)
    report = compare(tmp_path, CONTRACT)
    (unpaired,) = entry(report, "contended")["pairs"]["unpaired"]
    assert "no positive measured batch time" in unpaired["reason"]


def test_unobserved_admission_statistics_stay_unavailable(tmp_path):
    write_experiment(tmp_path, CONTRACT)
    effects = entry(compare(tmp_path, CONTRACT), "contended")["effects"]
    assert effects["baseline"]["starved_admissions"] is None  # not recorded, not zero
    assert effects["candidate"]["starved_admissions"] == 0
    assert effects["baseline"]["tasks_without_admission_wait"] == 20
    assert effects["baseline"]["median_admission_wait_seconds"] is None


def test_cold_requests_are_excluded_from_warm_tail_latency(tmp_path):
    manifest = planned()
    slot = first_slot(manifest, "contended", BASELINE)

    def timing(scheduled, task_id):
        if scheduled.index == slot.index and task_id == TASKS[0][0]:
            return 0, 60 * SECOND_NS  # one slow cold start
        return default_timing(scheduled, task_id)

    def change(scheduled, record):
        if scheduled.index == slot.index:
            record["tasks"][0]["cold"] = True
        return record

    write_experiment(tmp_path, CONTRACT, timing=timing, change=change)
    baseline = entry(compare(tmp_path, CONTRACT), "contended")["effects"]["baseline"]
    assert baseline["cold_tasks"] == 1
    assert baseline["p95_warm_task_latency_seconds"] == pytest.approx(9.0)


def test_new_failures_timeouts_and_acceptance_regressions_are_counted(tmp_path):
    manifest = planned()
    slots = [slot for slot in schedule(manifest) if slot.condition == "contended"]
    candidates = [slot.index for slot in slots if slot.configuration == CANDIDATE][:3]

    def change(scheduled, record):
        if scheduled.index == candidates[0]:
            record["tasks"][0].update(state="failed", acceptance="failed")
        if scheduled.index == candidates[1]:
            record["tasks"][1]["state"] = "timed_out"
        if scheduled.index == candidates[2]:
            record["tasks"][1]["acceptance"] = None
        return record

    write_experiment(tmp_path, CONTRACT, change=change)
    effects = entry(compare(tmp_path, CONTRACT), "contended")["effects"]
    assert effects["new_failures"] == 1
    assert effects["new_timeouts"] == 1
    assert effects["newly_failing_acceptance"] == 1
    assert effects["acceptance_unknown"] == 1
    assert effects["candidate"]["completed_tasks"] == 18


def test_development_experiments_are_never_sufficient_for_gating(tmp_path):
    document = manifest_document(CONTRACT, purpose="development")
    del document["workload"]["development_tasks"]
    write_experiment(tmp_path, CONTRACT, document=document)
    contended = entry(compare(tmp_path, CONTRACT), "contended")
    assert not contended["sufficient"]
    assert contended["insufficient_reasons"] == [
        "development experiments choose settings and are never gated"
    ]


def test_nearest_rank_and_bootstrap_are_reproducible():
    assert nearest_rank([], 0.95) is None
    assert nearest_rank([3.0, 1.0, 2.0], 0.95) == 3.0
    assert nearest_rank([float(value) for value in range(1, 21)], 0.95) == 19.0
    values = [0.05, 0.12, 0.2, 0.18, 0.09]
    assert bootstrap_median(values, 2000, 0.95, 1) == bootstrap_median(values, 2000, 0.95, 1)
    low, high = bootstrap_median(values, 2000, 0.95, 1)
    assert 0.05 <= low <= high <= 0.2
    assert bootstrap_median([0.1], 2000, 0.95, 1) is None  # one pair has no interval


def test_cli_writes_the_record_and_refuses_an_invalid_manifest(tmp_path, capsys):
    experiment = tmp_path / "experiment"
    write_experiment(experiment, CONTRACT)
    output = tmp_path / "reports"
    assert (
        main(
            ["compare", str(experiment), "--contract", str(CONTRACT_PATH), "--output", str(output)]
        )
        == 0
    )
    written = json.loads((output / "comparison.json").read_text())
    assert written == compare(experiment, CONTRACT)
    assert "contended inference-1: 10/10 pairs, sufficient" in capsys.readouterr().out
    assert (
        main(
            [
                "schedule",
                "--manifest",
                str(experiment / "manifest.json"),
                "--contract",
                str(CONTRACT_PATH),
            ]
        )
        == 0
    )
    planned = json.loads(capsys.readouterr().out)
    assert len(planned["batches"]) == 42
    assert {batch["limits"].get("inference_requests") for batch in planned["batches"]} == {None, 1}
    document = json.loads((experiment / "manifest.json").read_text())
    document["contract"]["sha256"] = "0" * 64
    (experiment / "manifest.json").write_text(json.dumps(document))
    assert (
        main(
            ["compare", str(experiment), "--contract", str(CONTRACT_PATH), "--output", str(output)]
        )
        == 1
    )
    assert "does not match" in capsys.readouterr().err


def test_default_output_is_keyed_by_the_manifest_hash(tmp_path, monkeypatch):
    manifest = write_experiment(tmp_path / "experiment", CONTRACT)
    monkeypatch.setattr(comparison_module, "DEFAULT_REPORTS", tmp_path / "reports")
    assert main(["compare", str(tmp_path / "experiment"), "--contract", str(CONTRACT_PATH)]) == 0
    assert (tmp_path / "reports" / manifest.sha256[:12] / "comparison.json").is_file()


def test_policy_and_injection_metadata_are_never_attribution_features():
    features = {
        column.name for column in (*COLUMNS, *REPLAY_LABEL_COLUMNS) if column.role == Role.FEATURE
    }
    assert DESIGN_FIELDS.isdisjoint(features)
    assert DESIGN_FIELDS.isdisjoint(FEATURE_NAMES)
