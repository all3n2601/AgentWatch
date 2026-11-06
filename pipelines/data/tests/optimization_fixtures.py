"""Synthetic optimization experiments for tests. Never recorded evidence.

Writes `manifest.json` and `batches/<index>.json` in the O04 formats, checked against the real
opt-contract-v1 file, so the manifest and batch readers are tested against the contract they
enforce. Task hashes, hardware and timings are invented.
"""

import copy
import json
import uuid
from collections.abc import Callable
from pathlib import Path

from agentwatch_data.optimization_manifest import (
    BASELINE,
    CANDIDATE,
    Manifest,
    OptimizationContract,
    ScheduledBatch,
    load_optimization_contract,
    parse_manifest,
    schedule,
)

ROOT = Path(__file__).resolve().parents[3]
CONTRACT_PATH = ROOT / "pipelines/model/optimization-contract.toml"
TASKS = (
    ("fix-mean-empty-r1", "1" * 64, "2" * 64),
    ("fix-window-count-r1", "3" * 64, "4" * 64),
)
HARDWARE = {
    "host_id": "synthetic-laptop-01",
    "cpu_model": "Synthetic CPU 8C",
    "logical_cpus": 8,
    "memory_bytes": 34_359_738_368,
    "os": "Ubuntu 24.04 (WSL2)",
    "kernel": "6.6.87.2-microsoft-standard-WSL2",
    "gpus": [{"name": "Synthetic GPU", "driver_version": "580.00", "memory_total_bytes": 8 << 30}],
}
MODEL = {"backend": "ollama", "name": "qwen3:1.7b", "digest": "8f68893c685c"}
POWER = {"source": "ac", "profile": "balanced"}
STATE = {"tests": "fixture-tests-r1", "retrieval": "docs-index-r1"}
SECOND_NS = 1_000_000_000

# Baseline batches take 10 s; the candidate takes 8 s under contention and 10 s when clean.
Timing = Callable[[ScheduledBatch, str], tuple[int, int] | None]


def contract() -> OptimizationContract:
    return load_optimization_contract(CONTRACT_PATH)


def manifest_document(contract: OptimizationContract, **changes) -> dict:
    document = {
        "format": "opt-experiment-manifest-v1",
        "experiment_id": "6f1d3c2a-8b8e-4f4e-9f61-0f0e9d4c2b11",
        "created_at": "2026-11-05T15:00:00Z",
        "contract": {"version": contract.version, "sha256": contract.sha256},
        "purpose": "evaluation",
        "workload": {
            "unit": "batch",
            "tasks": [
                {"task_id": task_id, "content_sha256": content, "configuration_digest": digest}
                for task_id, content, digest in TASKS
            ],
            "development_tasks": ["fix-median-odd-r1"],
            "arrivals": {
                "method": "fixed_offsets_v1",
                "offsets": [
                    {"task_id": TASKS[0][0], "offset_seconds": 0},
                    {"task_id": TASKS[1][0], "offset_seconds": 1.5},
                ],
            },
        },
        "runner": {"version": "agent-runner-v3"},
        "model": copy.deepcopy(MODEL),
        "hardware": copy.deepcopy(HARDWARE),
        "power": copy.deepcopy(POWER),
        "state": copy.deepcopy(STATE),
        "conditions": [
            {"name": "clean", "injectors": []},
            {
                "name": "contended",
                "injectors": [
                    {"kind": "cpu", "method": "cpu-spin-processes-v1", "intensity": {"workers": 4}}
                ],
            },
        ],
        "policies": {
            "baseline": {"limits": {}},
            "candidates": [{"name": "inference-1", "limits": {"inference_requests": 1}}],
        },
        "queue_bounds": {
            "inference_requests": {"max_length": 8, "max_wait_seconds": 120},
            "test_jobs": {"max_length": 4, "max_wait_seconds": 60},
        },
        "repetitions": {"pairs": 10, "warmup_batches": 1, "seed": 20261105},
    }
    document.update(changes)
    return document


def default_timing(slot: ScheduledBatch, task_id: str) -> tuple[int, int]:
    """(submitted, finished) ticks: the last task finishes when the batch does."""
    faster = slot.configuration == CANDIDATE and slot.condition == "contended"
    batch_ns = (8 if faster else 10) * SECOND_NS
    submitted = 0 if task_id == TASKS[0][0] else int(1.5 * SECOND_NS)
    finished = batch_ns if task_id == TASKS[1][0] else batch_ns - SECOND_NS
    return submitted, finished


def batch_record(
    manifest: Manifest,
    slot: ScheduledBatch,
    contract: OptimizationContract,
    timing: Timing = default_timing,
) -> dict:
    policy = manifest.policy(slot.candidate if slot.configuration == CANDIDATE else None)
    tasks = []
    for task in manifest.tasks:
        ticks = timing(slot, task.task_id)
        tasks.append(
            {
                "task_id": task.task_id,
                "run_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{slot.index}/{task.task_id}")),
                "state": "completed",
                "acceptance": "passed",
                "cold": False,
                "submitted_tick_ns": None if ticks is None else ticks[0],
                "finished_tick_ns": None if ticks is None else ticks[1],
                "admission_wait_seconds": 0.0 if slot.configuration == CANDIDATE else None,
                "evidence_dir": None,
            }
        )
    return {
        "format": "opt-batch-v1",
        "manifest_sha256": manifest.sha256,
        "batch_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"batch/{slot.index}")),
        "index": slot.index,
        "kind": slot.kind,
        "condition": slot.condition,
        "configuration": slot.configuration,
        "candidate": slot.candidate,
        "pair_index": slot.pair_index,
        "policy": {"sha256": policy.sha256, "limits": policy.limits, "version": None},
        "status": "completed",
        "reason": None,
        "started_at": "2026-11-05T15:10:00Z",
        "finished_at": "2026-11-05T15:10:10Z",
        "clock_id": f"agent-runner-batch-{slot.index}",
        "model": copy.deepcopy(MODEL),
        "hardware": copy.deepcopy(HARDWARE),
        "power": copy.deepcopy(POWER),
        "state": copy.deepcopy(STATE),
        "injection": {
            "active": True if slot.condition == "contended" else None,
            "cleanup_confirmed": True if slot.condition == "contended" else None,
        },
        "admission": {"starved_admissions": 0, "leaked_permits": 0}
        if slot.configuration == CANDIDATE
        else {"starved_admissions": None, "leaked_permits": None},
        "tasks": tasks,
    }


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


def write_experiment(
    root: Path,
    contract: OptimizationContract,
    document: dict | None = None,
    timing: Timing = default_timing,
    change: Callable[[ScheduledBatch, dict], dict | None] | None = None,
) -> Manifest:
    """Write a complete experiment; `change` may edit a record, or return None to omit it."""
    document = manifest_document(contract) if document is None else document
    write_json(root / "manifest.json", document)
    manifest = parse_manifest(document, contract)
    for slot in schedule(manifest):
        record = batch_record(manifest, slot, contract, timing)
        if change is not None:
            record = change(slot, record)
        if record is not None:
            write_json(root / "batches" / f"{slot.index:03d}.json", record)
    return manifest


def first_slot(manifest: Manifest, condition: str, configuration: str = BASELINE, kind="pair"):
    return next(
        slot
        for slot in schedule(manifest)
        if slot.condition == condition and slot.configuration == configuration and slot.kind == kind
    )
