"""Shared run evidence builders for the dataset pipeline tests."""

import json
from uuid import uuid4

BASE_NS = 1_700_000_000_000_000_000


def make_step(name, offset, queue_ns=1_000_000):
    submitted = BASE_NS + offset
    started = submitted + queue_ns
    finished = started + 999_000_000
    duration = (finished - submitted) / 1e9
    return {
        "step_id": name,
        "submitted_ns": submitted,
        "started_ns": started,
        "finished_ns": finished,
        "duration_seconds": duration,
        "queue_seconds": queue_ns / 1e9,
        "execution_seconds": 0.999,
        "cpu_seconds": 0.9,
        "queue_fraction": queue_ns / 1e9 / duration,
        "resource": "none",
        "checksum": "fixed",
        "evidence": "direct worker enqueue/start timestamps",
        "rule_version": "cpu-queue-v1",
        "threshold_seconds": 0.01,
    }


def make_result():
    baseline = [make_step(f"baseline-{i}", i * 2_000_000_000) for i in range(3)]
    contended = make_step("contended", 8_000_000_000, queue_ns=2_000_000_000) | {
        "resource": "cpu_sandbox"
    }
    return {
        "run_id": str(uuid4()),
        "experiment": {
            "worker_capacity": 1,
            "iterations": 10,
            "baseline_repeats": 3,
            "workload": "fixed SHA-256 CPU task",
        },
        "baseline": {"execution_cv": 0.01, "queue_threshold_seconds": 0.01, "steps": baseline},
        "clean": make_step("clean", 6_000_000_000),
        "contended": contended,
        "checks": {"contention_detected": True, "passed": True},
        "limitations": [],
    }


SAMPLE = {
    "timestamp_ns": 1,
    "cpu_pressure": None,
    "cgroup_cpu_stat": None,
    "unavailable": ["cpu_pressure: unavailable"],
}


def write_run(directory, result, samples=(SAMPLE,)):
    directory.mkdir(parents=True)
    (directory / "results.json").write_text(json.dumps(result))
    (directory / "samples.jsonl").write_text("".join(json.dumps(s) + "\n" for s in samples))
