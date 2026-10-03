import os
from datetime import UTC, datetime
from uuid import uuid4

import psycopg
import pytest
from agentwatch_worker.app import app
from fastapi.testclient import TestClient
from psycopg import sql
from psycopg.conninfo import make_conninfo


@pytest.fixture
def evidence():
    run_id = str(uuid4())

    def step(name, offset):
        submitted = 1_700_000_000_000_000_000 + offset
        return {
            "step_id": name,
            "submitted_ns": submitted,
            "started_ns": submitted + 1_000_000,
            "finished_ns": submitted + 1_000_000_000,
            "duration_seconds": 1,
            "queue_seconds": 0.001,
            "execution_seconds": 0.999,
            "cpu_seconds": 0.9,
            "queue_fraction": 0.001,
            "resource": "none",
            "checksum": "fixed",
            "evidence": "worker timestamps",
            "rule_version": "cpu-queue-v1",
            "threshold_seconds": 0.01,
        }

    clean = step("clean", 4_000_000_000)
    span = {
        "name": "sandbox.run_tests",
        "context": {"trace_id": "0x" + "a" * 32, "span_id": "0x" + "b" * 16},
        "start_time": datetime.fromtimestamp(clean["submitted_ns"] / 1e9, UTC).isoformat(),
        "end_time": datetime.fromtimestamp(clean["finished_ns"] / 1e9, UTC).isoformat(),
        "attributes": {
            "agentwatch.run_id": run_id,
            "agentwatch.step_id": "clean",
            "agentwatch.resource_class": "cpu_sandbox",
        },
    }
    return {
        "kind": "coding",
        "result": {
            "run_id": run_id,
            "experiment": {
                "worker_capacity": 1,
                "iterations": 1000,
                "baseline_repeats": 3,
                "workload": "tests",
                "tests_per_step": 3,
            },
            "baseline": {
                "execution_cv": 0.02,
                "queue_threshold_seconds": 0.01,
                "steps": [step(f"baseline-{i}", i * 1_000_000_000) for i in range(3)],
            },
            "clean": clean,
            "contended": step("contended", 5_000_000_000),
            "checks": {"passed": True, "stable_baseline": True},
            "limitations": [],
        },
        "spans": [span],
        "samples": [
            {
                "timestamp_ns": clean["submitted_ns"],
                "cpu_pressure": None,
                "cgroup_cpu_stat": None,
                "unavailable": ["unsupported"],
            }
        ],
    }


@pytest.fixture
def client(monkeypatch):
    dsn = os.environ.get("AGENTWATCH_TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("Set AGENTWATCH_TEST_DATABASE_URL for isolated PostgreSQL API tests")
    schema = "aw_test_" + uuid4().hex
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    monkeypatch.setenv("DATABASE_URL", make_conninfo(dsn, options=f"-csearch_path={schema}"))
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
