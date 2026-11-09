"""Comparison records: only complete pairs are compared, every attempt is kept, failures in
broken batches are still counted, and missing measurements stay missing."""

import json

import pytest
from agentwatch_data import optimization_comparison as comparison_module
from agentwatch_data.files import json_text
from agentwatch_data.optimization_comparison import (
    bootstrap,
    compare,
    main,
    median_time_reduction,
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


def planned(**changes):
    """The fixture manifest, parsed without writing any batch."""
    return parse_manifest(manifest_document(CONTRACT, **changes), CONTRACT)


def entry(report, condition) -> dict:
    return next(item for item in report["comparisons"] if item["condition"] == condition)


def editing(index, edit):
    def change(slot, record):
        if slot.index == index:
            edit(record)
        return record

    return change


def test_paired_effects_on_a_complete_experiment(tmp_path):
    write_experiment(tmp_path, CONTRACT)
    report = compare(tmp_path, CONTRACT)
    assert report["missing_batches"] == [] and report["hidden_files"] == []
    assert all(batch["accepted"] for batch in report["batches"])
    contended = entry(report, "contended")
    assert contended["pairs"] == {"scheduled": 10, "complete": 10, "unpaired": []}
    assert contended["sufficient"] and contended["insufficient_reasons"] == []
    effects = contended["effects"]
    # 10 s baseline batches against 8 s candidate batches in every pair.
    assert effects["median_batch_time_reduction"] == pytest.approx(0.2)
    assert effects["median_pair_reduction"] == pytest.approx(0.2)
    interval = effects["median_batch_time_reduction_interval"]
    assert (interval["low"], interval["high"]) == (pytest.approx(0.2), pytest.approx(0.2))
    assert (interval["seed"], interval["resamples"]) == (20261105, CONTRACT.bootstrap_resamples)
    assert effects["throughput_ratio"] == pytest.approx(1.25)
    assert effects["baseline"]["completed_tasks"] == 20
    assert contended["outcome_changes"] == {
        "pairs_compared": 10,
        "new_failures": 0,
        "new_timeouts": 0,
        "newly_failing_acceptance": 0,
        "acceptance_unknown": 0,
    }
    clean = entry(report, "clean")
    assert clean["effects"]["median_batch_time_reduction"] == pytest.approx(0)
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


def test_an_aborted_candidate_batch_still_counts_the_failures_it_caused(tmp_path):
    slot = first_slot(planned(), "contended", CANDIDATE)

    def abort(record):
        record.update(status="aborted", reason="Candidate deadlocked under its limit")
        for task in record["tasks"]:
            task["state"] = "timed_out"
            task["acceptance"] = "failed"

    write_experiment(tmp_path, CONTRACT, change=editing(slot.index, abort))
    contended = entry(compare(tmp_path, CONTRACT), "contended")
    assert contended["pairs"]["complete"] == 9
    assert contended["outcome_changes"]["pairs_compared"] == 10
    assert contended["outcome_changes"]["new_timeouts"] == 2
    assert contended["outcome_changes"]["newly_failing_acceptance"] == 2
    assert contended["attempts"][CANDIDATE]["batches_by_status"] == {"aborted": 1, "completed": 9}
    assert contended["attempts"][CANDIDATE]["tasks_by_state"] == {"completed": 18, "timed_out": 2}
    assert not contended["sufficient"]
    assert "1 scheduled pairs are unpaired" in contended["insufficient_reasons"]


def test_a_failed_candidate_batch_leaves_its_pair_unpaired_but_listed(tmp_path):
    slot = first_slot(planned(), "contended", CANDIDATE)

    def fail(record):
        record.update(status="failed", reason="Model server stopped responding")
        record["tasks"][0]["state"] = "failed"

    write_experiment(tmp_path, CONTRACT, change=editing(slot.index, fail))
    report = compare(tmp_path, CONTRACT)
    contended = entry(report, "contended")
    (unpaired,) = contended["pairs"]["unpaired"]
    assert unpaired["pair_id"] == slot.pair_id
    assert f"candidate batch {slot.index} is failed" in unpaired["reason"]
    assert contended["outcome_changes"]["new_failures"] == 1
    kept = next(batch for batch in report["batches"] if batch["index"] == slot.index)
    assert (kept["status"], kept["reason"]) == ("failed", "Model server stopped responding")
    # 9 pairs of 2 tasks complete 18 per configuration: below the contract's 20.
    assert any("18 completed" in reason for reason in contended["insufficient_reasons"])


def test_rejected_or_missing_evidence_keeps_every_comparison_insufficient(tmp_path):
    # Two extra pairs, so the remaining pairs still reach the contract's minimums.
    repetitions = {"pairs": 12, "warmup_batches": 1, "seed": 3}
    manifest = planned(repetitions=repetitions)
    missing = first_slot(manifest, "clean", BASELINE)
    rejected = first_slot(manifest, "contended", BASELINE)

    def change(scheduled, record):
        if scheduled.index == missing.index:
            return None
        if scheduled.index == rejected.index:
            record["hardware"]["host_id"] = "another-laptop"
        return record

    document = manifest_document(CONTRACT, repetitions=repetitions)
    write_experiment(tmp_path, CONTRACT, document=document, change=change)
    report = compare(tmp_path, CONTRACT)
    assert report["missing_batches"] == [missing.index]
    for condition in ("clean", "contended"):
        item = entry(report, condition)
        assert item["pairs"]["complete"] == 11
        assert not item["sufficient"]
        reasons = item["insufficient_reasons"]
        assert "the experiment has rejected batch records (1 hardware_mismatch)" in reasons
        assert "1 scheduled batches have no record" in reasons
    (clean,) = entry(report, "clean")["pairs"]["unpaired"]
    assert clean["reason"] == f"baseline batch {missing.index} has no record"
    (contended,) = entry(report, "contended")["pairs"]["unpaired"]
    assert "rejected (hardware_mismatch)" in contended["reason"]
    file = f"{rejected.index:03d}.json"
    rejected_entry = next(batch for batch in report["batches"] if batch["file"] == file)
    assert rejected_entry == {
        "file": file,
        "sha256": rejected_entry["sha256"],
        "accepted": False,
        "code": "hardware_mismatch",
        "detail": "hardware differs from the manifest in ['host_id']",
    }
    assert len(rejected_entry["sha256"]) == 64


def test_missing_task_ticks_never_become_a_batch_time(tmp_path):
    slot = first_slot(planned(), "contended", CANDIDATE)

    def timing(scheduled, task_id):
        if scheduled.index == slot.index and task_id == TASKS[1][0]:
            return None
        return default_timing(scheduled, task_id)

    write_experiment(tmp_path, CONTRACT, timing=timing)
    (unpaired,) = entry(compare(tmp_path, CONTRACT), "contended")["pairs"]["unpaired"]
    assert "no positive measured batch time" in unpaired["reason"]


def test_admission_waits_are_judged_against_the_declared_bounds(tmp_path):
    slot = first_slot(planned(), "contended", CANDIDATE)

    def starve(record):
        record["tasks"][1]["admission_waits"]["inference_requests"] = 200.0  # bound is 120 s

    write_experiment(tmp_path, CONTRACT, change=editing(slot.index, starve))
    effects = entry(compare(tmp_path, CONTRACT), "contended")["effects"]
    inference = effects["candidate"]["admission_waits"]["inference_requests"]
    assert inference["over_declared_max_wait"] == 1
    assert inference["declared_max_wait_seconds"] == 120
    assert inference["max_seconds"] == 200.0 and inference["unobserved"] == 0
    # Waits the runner did not observe stay unobserved: never zero, never "within bounds".
    assert effects["candidate"]["admission_waits"]["test_jobs"] == {
        "median_seconds": None,
        "max_seconds": None,
        "declared_max_wait_seconds": 60,
        "over_declared_max_wait": 0,
        "unobserved": 20,
    }
    assert effects["baseline"]["starved_admissions"] is None  # not recorded, not zero
    assert effects["candidate"]["starved_admissions"] == 0


def test_cold_requests_are_excluded_from_warm_tail_latency(tmp_path):
    slot = first_slot(planned(), "contended", BASELINE)

    def timing(scheduled, task_id):
        if scheduled.index == slot.index and task_id == TASKS[0][0]:
            return 0, 60 * SECOND_NS  # one slow cold start
        return default_timing(scheduled, task_id)

    write_experiment(
        tmp_path,
        CONTRACT,
        timing=timing,
        change=editing(slot.index, lambda record: record["tasks"][0].update(cold=True)),
    )
    baseline = entry(compare(tmp_path, CONTRACT), "contended")["effects"]["baseline"]
    assert baseline["cold_tasks"] == 1
    assert baseline["p95_warm_task_latency_seconds"] == pytest.approx(9.0)


def test_new_failures_timeouts_and_acceptance_regressions_are_counted(tmp_path):
    slots = [slot for slot in schedule(planned()) if slot.condition == "contended"]
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
    contended = entry(compare(tmp_path, CONTRACT), "contended")
    assert contended["outcome_changes"] == {
        "pairs_compared": 10,
        "new_failures": 1,
        "new_timeouts": 1,
        "newly_failing_acceptance": 1,
        "acceptance_unknown": 1,
    }
    assert contended["effects"]["candidate"]["completed_tasks"] == 18


def test_development_experiments_are_never_sufficient_for_gating(tmp_path):
    document = manifest_document(CONTRACT, purpose="development")
    del document["workload"]["development_tasks"]
    write_experiment(tmp_path, CONTRACT, document=document)
    contended = entry(compare(tmp_path, CONTRACT), "contended")
    assert contended["insufficient_reasons"] == [
        "development experiments choose settings and are never gated"
    ]


def test_hidden_and_oddly_named_files_are_reported_not_crashed_on(tmp_path):
    write_experiment(tmp_path, CONTRACT)
    (tmp_path / "batches" / ".001.json.tmp").write_text("{}")
    (tmp_path / "batches" / "²²².json").write_text("{}")  # digits, but not ASCII
    report = compare(tmp_path, CONTRACT)
    assert report["hidden_files"] == [".001.json.tmp"]
    (rejected,) = [batch for batch in report["batches"] if not batch["accepted"]]
    assert (rejected["file"], rejected["code"]) == ("²²².json", "malformed")


def test_the_improvement_statistic_is_the_contracts_median_batch_time_reduction():
    # The median of per-pair reductions is 0.10 here, but the contract's statistic, the relative
    # reduction of the median batch time, is 1 - 11/12.
    pairs = [(10.0, 9.0), (12.0, 12.0), (14.0, 11.0)]
    assert median_time_reduction(pairs) == pytest.approx(1 - 11 / 12)
    interval = bootstrap(pairs, median_time_reduction, 2000, 0.95, seed=1)
    assert interval == bootstrap(pairs, median_time_reduction, 2000, 0.95, seed=1)
    low, high = interval
    assert low <= median_time_reduction(pairs) <= high and low < high
    assert bootstrap(pairs[:1], median_time_reduction, 2000, 0.95, seed=1) is None


def test_nearest_rank_percentile():
    assert nearest_rank([], 0.95) is None
    assert nearest_rank([3.0, 1.0, 2.0], 0.95) == 3.0
    assert nearest_rank([float(value) for value in range(1, 21)], 0.95) == 19.0


def test_cli_writes_the_record_and_refuses_an_invalid_manifest(tmp_path, capsys):
    experiment = tmp_path / "experiment"
    write_experiment(experiment, CONTRACT)
    output = tmp_path / "reports"
    command = [
        "compare",
        str(experiment),
        "--contract",
        str(CONTRACT_PATH),
        "--output",
        str(output),
    ]
    assert main(command) == 0
    written = json.loads((output / "comparison.json").read_text())
    assert written == compare(experiment, CONTRACT)
    assert "contended inference-1: 10/10 pairs, sufficient" in capsys.readouterr().out
    manifest = str(experiment / "manifest.json")
    assert main(["schedule", "--manifest", manifest, "--contract", str(CONTRACT_PATH)]) == 0
    planned_batches = json.loads(capsys.readouterr().out)["batches"]
    assert len(planned_batches) == 42
    assert {batch["limits"].get("inference_requests") for batch in planned_batches} == {None, 1}
    document = json.loads((experiment / "manifest.json").read_text())
    document["contract"]["sha256"] = "0" * 64
    (experiment / "manifest.json").write_text(json.dumps(document))
    assert main(command) == 1
    assert "does not match" in capsys.readouterr().err


def test_a_manifest_with_nan_is_refused_not_crashed_on(tmp_path, capsys):
    experiment = tmp_path / "experiment"
    write_experiment(experiment, CONTRACT)
    text = (experiment / "manifest.json").read_text()
    assert '"workers": 4' in text
    (experiment / "manifest.json").write_text(text.replace('"workers": 4', '"workers": NaN'))
    assert main(["compare", str(experiment), "--contract", str(CONTRACT_PATH)]) == 1
    assert "not canonical JSON" in capsys.readouterr().err


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


def test_batches_without_provenance_never_make_a_comparison_sufficient(tmp_path):
    # The owner's reproduction: every evidence link left null, task IDs only.
    def unlink(scheduled, record):
        for task in record["tasks"]:
            task["evidence_dir"] = None
        return record

    write_experiment(tmp_path, CONTRACT, change=unlink)
    report = compare(tmp_path, CONTRACT)
    assert {batch["code"] for batch in report["batches"]} == {"provenance_missing"}
    for item in report["comparisons"]:
        assert not item["sufficient"] and item["effects"] is None


def test_another_contention_treatment_never_makes_a_comparison_sufficient(tmp_path):
    # The owner's reproduction: the manifest's injector unchanged, another treatment applied.
    def other_treatment(scheduled, record):
        if scheduled.condition == "contended":
            record["injection"]["injectors"][0]["intensity"] = {"workers": 1}
        return record

    write_experiment(tmp_path, CONTRACT, change=other_treatment)
    contended = entry(compare(tmp_path, CONTRACT), "contended")
    assert not contended["sufficient"]
    assert contended["pairs"]["complete"] == 0
    assert any("injection_mismatch" in reason for reason in contended["insufficient_reasons"])
