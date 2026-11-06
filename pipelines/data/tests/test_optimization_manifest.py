"""Optimization manifests: preregistration checked against opt-contract-v1, and the seeded order."""

import json
import tomllib
from collections import Counter

import pytest
from agentwatch_data.extract import canonical_sha256
from agentwatch_data.optimization_manifest import (
    BASELINE,
    CANDIDATE,
    DESIGN_FIELDS,
    PAIR,
    WARMUP,
    ManifestError,
    load_optimization_contract,
    parse_manifest,
    policy_sha256,
    read_manifest,
    schedule,
)
from optimization_fixtures import CONTRACT_PATH, contract, manifest_document

CONTRACT = contract()


def broken(**changes) -> dict:
    return manifest_document(CONTRACT, **changes)


def edited(path: str, value) -> dict:
    """The fixture manifest with one nested value replaced, by a dotted path."""
    document = manifest_document(CONTRACT)
    *parents, last = path.split(".")
    target = document
    for key in parents:
        target = target[int(key)] if isinstance(target, list) else target[key]
    if isinstance(target, list):
        target[int(last)] = value
    else:
        target[last] = value
    return document


def test_contract_is_identified_by_its_parsed_contents():
    parsed = tomllib.loads(CONTRACT_PATH.read_text())
    assert CONTRACT.sha256 == canonical_sha256(parsed)
    assert (CONTRACT.version, CONTRACT.controls) == (
        "opt-contract-v1",
        ("inference_requests", "test_jobs"),
    )
    assert CONTRACT.candidate_limits == (1, 2, 4) and CONTRACT.min_paired_batches == 5


def test_contract_with_another_version_is_refused(tmp_path):
    path = tmp_path / "contract.toml"
    path.write_text(CONTRACT_PATH.read_text().replace('"opt-contract-v1"', '"opt-contract-v0"'))
    with pytest.raises(ManifestError, match="version"):
        load_optimization_contract(path)


def test_valid_manifest_parses_with_a_formatting_independent_hash(tmp_path):
    document = manifest_document(CONTRACT)
    manifest = parse_manifest(document, CONTRACT)
    assert manifest.sha256 == canonical_sha256(document)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(dict(reversed(document.items())), indent=4))
    assert read_manifest(path, CONTRACT).sha256 == manifest.sha256
    assert [task.task_id for task in manifest.tasks] == ["fix-mean-empty-r1", "fix-window-count-r1"]
    assert manifest.arrivals == {"fix-mean-empty-r1": 0.0, "fix-window-count-r1": 1.5}


@pytest.mark.parametrize(
    ("document", "message"),
    [
        (edited("contract.sha256", "0" * 64), "does not match"),
        (edited("contract.version", "opt-contract-v0"), "does not match"),
        (broken(format="opt-experiment-manifest-v0"), "format"),
        (broken(surprise=True), "unknown"),
        (edited("experiment_id", "not-a-uuid"), "UUID"),
        (edited("created_at", "2026-11-05 15:00"), "RFC 3339"),
        (edited("power.source", "unknown"), "power.source"),
        (edited("model.digest", ""), "model.digest"),
        (edited("hardware.logical_cpus", True), "logical_cpus"),
        (edited("workload.tasks.0.content_sha256", "abc"), "content_sha256"),
        (edited("workload.tasks.1.task_id", "fix-mean-empty-r1"), "repeat"),
    ],
)
def test_malformed_or_stale_manifests_are_refused(document, message):
    with pytest.raises(ManifestError, match=message):
        parse_manifest(document, CONTRACT)


@pytest.mark.parametrize(
    ("limits", "message"),
    [
        ({"inference_requests": 1, "test_jobs": 1}, "exactly one control"),
        ({}, "exactly one control"),
        ({"inference_requests": 3}, "preregistered candidate limit"),
        ({"inference_requests": 0}, "at least 1"),
        ({"inference_requests": -1}, "at least 1"),
        ({"inference_requests": None}, "at least 1"),
        ({"inference_requests": True}, "at least 1"),
        ({"inference_requests": 2.0}, "at least 1"),
        ({"gpu_requests": 1}, "unknown controls"),
    ],
)
def test_candidates_change_one_control_to_a_preregistered_limit(limits, message):
    document = edited("policies.candidates.0.limits", limits)
    with pytest.raises(ManifestError, match=message):
        parse_manifest(document, CONTRACT)


def test_baseline_is_unlimited_and_candidates_are_distinct():
    with pytest.raises(ManifestError, match="baseline.limits must be empty"):
        parse_manifest(edited("policies.baseline.limits", {"test_jobs": 4}), CONTRACT)
    twice = [
        {"name": "inference-1", "limits": {"inference_requests": 1}},
        {"name": "again", "limits": {"inference_requests": 1}},
    ]
    with pytest.raises(ManifestError, match="limits must not repeat"):
        parse_manifest(edited("policies.candidates", twice), CONTRACT)


def test_every_control_declares_queue_bounds():
    document = manifest_document(CONTRACT)
    del document["queue_bounds"]["test_jobs"]
    with pytest.raises(ManifestError, match="queue_bounds keys"):
        parse_manifest(document, CONTRACT)
    with pytest.raises(ManifestError, match="positive"):
        parse_manifest(edited("queue_bounds.test_jobs.max_wait_seconds", 0), CONTRACT)


@pytest.mark.parametrize(
    ("repetitions", "message"),
    [
        ({"pairs": 4, "warmup_batches": 1, "seed": 1}, "at least 5"),
        # 5 pairs of 2 tasks complete only 10 tasks per configuration; the contract needs 20.
        ({"pairs": 5, "warmup_batches": 1, "seed": 1}, "cannot reach"),
        ({"pairs": 10, "seed": 1}, "warmup_batches"),
        ({"pairs": 10, "warmup_batches": 1, "seed": -1}, "seed"),
    ],
)
def test_repetitions_meet_the_contract_minimums_before_any_batch_runs(repetitions, message):
    with pytest.raises(ManifestError, match=message):
        parse_manifest(broken(repetitions=repetitions), CONTRACT)


def test_conditions_preregister_contention_only_where_the_contract_gates_it():
    with pytest.raises(ManifestError, match="preregister its injector set"):
        parse_manifest(edited("conditions.1.injectors", []), CONTRACT)
    injector = {"kind": "cpu", "method": "cpu-spin-processes-v1", "intensity": {"workers": 2}}
    with pytest.raises(ManifestError, match="uncontended"):
        parse_manifest(edited("conditions.0.injectors", [injector]), CONTRACT)
    with pytest.raises(ManifestError, match="cover exactly"):
        parse_manifest(edited("conditions", [{"name": "clean", "injectors": []}]), CONTRACT)
    with pytest.raises(ManifestError, match="intensity"):
        parse_manifest(edited("conditions.1.injectors.0.intensity", {}), CONTRACT)


def test_evaluation_tasks_are_frozen_and_disjoint_from_development_tasks():
    with pytest.raises(ManifestError, match="disjoint"):
        parse_manifest(edited("workload.development_tasks", ["fix-mean-empty-r1"]), CONTRACT)
    with pytest.raises(ManifestError, match="must name its development_tasks"):
        parse_manifest(edited("workload.development_tasks", []), CONTRACT)
    development = manifest_document(CONTRACT, purpose="development")
    with pytest.raises(ManifestError, match="Only an evaluation"):
        parse_manifest(development, CONTRACT)
    del development["workload"]["development_tasks"]
    assert parse_manifest(development, CONTRACT).purpose == "development"


def test_an_evaluation_needs_the_contract_minimum_of_tasks():
    document = manifest_document(
        CONTRACT, repetitions={"pairs": 20, "warmup_batches": 0, "seed": 1}
    )
    document["workload"]["tasks"] = document["workload"]["tasks"][:1]
    document["workload"]["arrivals"]["offsets"] = document["workload"]["arrivals"]["offsets"][:1]
    with pytest.raises(ManifestError, match="at least 2 frozen tasks"):
        parse_manifest(document, CONTRACT)


def test_arrivals_schedule_every_task_once_from_zero():
    with pytest.raises(ManifestError, match="start at offset 0"):
        parse_manifest(edited("workload.arrivals.offsets.0.offset_seconds", 0.5), CONTRACT)
    with pytest.raises(ManifestError, match="every task exactly once"):
        parse_manifest(
            edited(
                "workload.arrivals.offsets", [{"task_id": "fix-mean-empty-r1", "offset_seconds": 0}]
            ),
            CONTRACT,
        )


def test_policy_hash_identifies_what_is_enforced_only():
    assert policy_sha256({"inference_requests": 1}, CONTRACT) == policy_sha256(
        {"inference_requests": 1}, CONTRACT
    )
    hashes = {
        policy_sha256(limits, CONTRACT)
        for limits in ({}, {"inference_requests": 1}, {"inference_requests": 2}, {"test_jobs": 1})
    }
    assert len(hashes) == 4


def test_schedule_is_reproducible_paired_and_randomized_by_the_seed():
    manifest = parse_manifest(manifest_document(CONTRACT), CONTRACT)
    slots = schedule(manifest)
    assert slots == schedule(manifest)
    assert [slot.index for slot in slots] == list(range(len(slots)))
    assert len(slots) == 2 * (1 + 10 * 2)
    by_pair: dict[str, list] = {}
    for slot in slots:
        if slot.kind == PAIR:
            by_pair.setdefault(slot.pair_id, []).append(slot)
    for both in by_pair.values():
        assert sorted(slot.configuration for slot in both) == [BASELINE, CANDIDATE]
        assert abs(both[0].index - both[1].index) == 1  # back to back
    # Warm-ups run first in each condition, at baseline limits.
    clean = [slot for slot in slots if slot.condition == "clean"]
    assert clean[0].kind == WARMUP and clean[0].policy_sha256 == manifest.baseline.sha256
    firsts = Counter(both[0].configuration for both in by_pair.values())
    assert firsts[BASELINE] and firsts[CANDIDATE]  # the order is not fixed
    reseeded = parse_manifest(
        manifest_document(CONTRACT, repetitions={"pairs": 10, "warmup_batches": 1, "seed": 7}),
        CONTRACT,
    )
    assert schedule(reseeded) != slots


def test_design_fields_name_the_policy_and_injection_metadata():
    assert {"limits", "policy", "condition", "injectors", "candidate"} <= DESIGN_FIELDS


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('status = "proposed"', 'status = "draft"', "status"),
        ("[repetitions]", "[repeats]", r"no \[repetitions\] table"),
        ("min_paired_batches = 5", "minimum_pairs = 5", "lacks 'min_paired_batches'"),
        ('version = "opt-contract-v1"', 'version = "opt-contract-v1', "cannot be read"),
    ],
)
def test_an_unusable_contract_is_refused(tmp_path, old, new, message):
    path = tmp_path / "contract.toml"
    text = CONTRACT_PATH.read_text()
    assert old in text
    path.write_text(text.replace(old, new, 1))
    with pytest.raises(ManifestError, match=message):
        load_optimization_contract(path)


@pytest.mark.parametrize(
    ("document", "message"),
    [
        (broken(model=[]), "model must be an object"),
        (edited("workload.arrivals.offsets.1.offset_seconds", "1.5"), "finite number"),
        (edited("workload.arrivals.offsets.1.offset_seconds", -1), "nonnegative"),
        (edited("workload.tasks", {}), "list"),
        (edited("created_at", "2026-13-05T15:00:00Z"), "not a valid timestamp"),
        (edited("repetitions.seed", 2**63), "at most"),
    ],
)
def test_wrongly_typed_values_are_refused(document, message):
    with pytest.raises(ManifestError, match=message):
        parse_manifest(document, CONTRACT)


def test_a_manifest_file_must_be_bounded_strict_json(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text('{"format": 1, "format": 2}')
    with pytest.raises(ManifestError, match="manifest.json"):
        read_manifest(path, CONTRACT)


def test_revisions_of_a_development_task_are_not_new_evaluation_tasks():
    document = edited("workload.development_tasks", ["fix-mean-empty-r2"])
    with pytest.raises(ManifestError, match=r"shared \['fix-mean-empty'\]"):
        parse_manifest(document, CONTRACT)


@pytest.mark.parametrize(
    ("repetitions", "message"),
    [
        ({"pairs": 10**7, "warmup_batches": 1, "seed": 1}, "at most 1000"),
        ({"pairs": 10, "warmup_batches": 10**9, "seed": 1}, "at most 100"),
    ],
)
def test_repetitions_are_bounded(repetitions, message):
    with pytest.raises(ManifestError, match=message):
        parse_manifest(broken(repetitions=repetitions), CONTRACT)


def test_arrivals_declare_a_positive_tolerance():
    document = manifest_document(CONTRACT)
    del document["workload"]["arrivals"]["tolerance_seconds"]
    with pytest.raises(ManifestError, match="tolerance_seconds"):
        parse_manifest(document, CONTRACT)
    with pytest.raises(ManifestError, match="positive"):
        parse_manifest(edited("workload.arrivals.tolerance_seconds", 0), CONTRACT)


def test_queue_bounds_are_kept_for_the_comparison():
    manifest = parse_manifest(manifest_document(CONTRACT), CONTRACT)
    assert manifest.queue_bounds == {"inference_requests": (8, 120.0), "test_jobs": (4, 60.0)}
    assert manifest.arrival_tolerance_seconds == 0.25
