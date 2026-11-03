"""Versioned concurrency policy storage against isolated PostgreSQL schemas."""

import threading
from importlib.resources import files
from uuid import UUID, uuid4

import psycopg
import pytest
from agentwatch_sdk.live import RunCreate
from agentwatch_sdk.policy import RUNNER_SERVICE_ID, PolicyChange
from agentwatch_worker.live_store import LiveStore, MissingRunError
from agentwatch_worker.policy_store import (
    InvalidRestoreError,
    PolicyStore,
    UnsupportedTargetError,
)
from agentwatch_worker.store import ConflictError, RunStore
from psycopg import sql
from psycopg.conninfo import make_conninfo

TARGET = "coding-agent"


def change(expected_version, action="set", **fields):
    payload = {
        "request_id": str(uuid4()),
        "expected_version": expected_version,
        "action": action,
        "operator": "fixture-operator",
        "reason": "fixture policy change",
    } | fields
    return PolicyChange.model_validate(payload)


def run_registration(service_id):
    return {
        "contract_version": "live-contracts-v1",
        "request_id": str(uuid4()),
        "task_id": "policy-fixture",
        "configuration_digest": "a" * 64,
        "clock_ids": ["fixture-clock"],
        "resources": [
            {
                "resource_id": "fixture-resource",
                "service_id": service_id,
                "resource_class": "cpu_sandbox",
            }
        ],
    }


def create_run(client, service_id=RUNNER_SERVICE_ID):
    response = client.post("/v1/live/runs", json=run_registration(service_id))
    assert response.status_code == 201
    return UUID(response.json()["run_id"])


def link_count(client):
    with client.app.state.store.connect() as conn:
        return conn.execute("SELECT count(*) AS n FROM live_run_policy_links").fetchone()["n"]


@pytest.fixture
def policies(client):
    return PolicyStore(client.app.state.store)


def test_active_defaults_to_baseline(policies):
    active = policies.active(TARGET)
    assert active["version"] == 0
    assert active["limits"] == {}
    assert active["action"] == "baseline"
    assert active["operator"] is None and active["request_id"] is None
    assert active["controlled_services"]["test_jobs"]["service_id"] == RUNNER_SERVICE_ID
    assert active["enforcement_scope"] == "runner_process"


def test_set_creates_next_version(policies):
    result = policies.change(TARGET, change(0, limits={"inference_requests": 2}))
    assert result["created"] is True
    assert result["policy"]["version"] == 1
    assert result["policy"]["limits"] == {"inference_requests": 2}
    assert result["policy"]["operator"] == "fixture-operator"
    assert result["policy"]["created_at"] is not None
    assert policies.active(TARGET) == result["policy"]


def test_stale_expected_version_conflicts(policies):
    policies.change(TARGET, change(0, limits={"test_jobs": 1}))
    with pytest.raises(ConflictError):
        policies.change(TARGET, change(0, limits={"test_jobs": 2}))
    page = policies.history(TARGET, None, 50)
    assert [v["version"] for v in page["versions"]] == [1, 0]
    assert page["next_before"] is None


def test_retry_is_idempotent(policies):
    first = change(0, limits={"test_jobs": 3})
    created = policies.change(TARGET, first)
    # A later version must not turn a faithful retry of an older request into a conflict.
    policies.change(TARGET, change(1, action="disable"))
    retried = policies.change(TARGET, first)
    assert retried == {"created": False, "policy": created["policy"]}
    altered = first.model_copy(update={"reason": "different reason"})
    with pytest.raises(ConflictError):
        policies.change(TARGET, altered)
    assert [v["version"] for v in policies.history(TARGET, None, 50)["versions"]] == [2, 1, 0]


def test_disable_and_restore(policies):
    policies.change(TARGET, change(0, limits={"test_jobs": 1}))
    disabled = policies.change(TARGET, change(1, action="disable"))["policy"]
    assert disabled["limits"] == {} and disabled["action"] == "disable"
    restored = policies.change(TARGET, change(2, action="restore", restored_from=1))["policy"]
    assert restored["limits"] == {"test_jobs": 1}
    assert restored["restored_from"] == 1
    baseline = policies.change(TARGET, change(3, action="restore", restored_from=0))["policy"]
    assert baseline["limits"] == {} and baseline["restored_from"] == 0
    assert baseline["version"] == 4


def test_restore_missing_version_rejected(policies):
    policies.change(TARGET, change(0, limits={"test_jobs": 1}))
    with pytest.raises(InvalidRestoreError):
        policies.change(TARGET, change(1, action="restore", restored_from=7))
    assert policies.active(TARGET)["version"] == 1
    assert len(policies.history(TARGET, None, 50)["versions"]) == 2


def test_unsupported_target(policies):
    with pytest.raises(UnsupportedTargetError):
        policies.active("cpu-demo")
    with pytest.raises(UnsupportedTargetError):
        policies.change("cpu-demo", change(0, limits={"test_jobs": 1}))
    with pytest.raises(UnsupportedTargetError):
        policies.history("cpu-demo", None, 50)


def test_history_pages_newest_first_with_baseline_last(policies):
    for version in range(3):
        policies.change(TARGET, change(version, limits={"test_jobs": version + 1}))
    first = policies.history(TARGET, None, 2)
    assert [v["version"] for v in first["versions"]] == [3, 2]
    assert first["next_before"] == 2
    second = policies.history(TARGET, first["next_before"], 2)
    assert [v["version"] for v in second["versions"]] == [1, 0]
    assert second["versions"][-1]["action"] == "baseline"
    assert second["next_before"] is None
    assert policies.history(TARGET, 0, 2) == {"versions": [], "next_before": None}


def test_concurrent_writers_one_wins(policies):
    barrier = threading.Barrier(2)
    outcomes = []

    def write(limit):
        barrier.wait()
        try:
            outcomes.append(
                policies.change(TARGET, change(0, limits={"test_jobs": limit}))["created"]
            )
        except ConflictError:
            outcomes.append("conflict")

    threads = [threading.Thread(target=write, args=(limit,)) for limit in (1, 2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(outcomes, key=str) == [True, "conflict"]
    assert [v["version"] for v in policies.history(TARGET, None, 50)["versions"]] == [1, 0]


def test_new_version_links_only_active_agent_runs(client, policies):
    agent = create_run(client)
    other = create_run(client, service_id="fixture-service")
    finished = create_run(client)
    with client.app.state.store.connect() as conn:
        conn.execute("UPDATE live_runs SET state='completed' WHERE run_id=%s", (finished,))
    policies.change(TARGET, change(0, limits={"inference_requests": 1}))

    agent_policy = policies.run_policy(agent)
    assert agent_policy["applicable"] is True
    assert agent_policy["target"] == TARGET
    assert [v["version"] for v in agent_policy["versions"]] == [1]
    assert agent_policy["versions"][0]["limits"] == {"inference_requests": 1}
    assert policies.run_policy(other) == {
        "run_id": str(other),
        "target": TARGET,
        "applicable": False,
        "versions": [],
    }
    assert policies.run_policy(finished)["versions"] == []
    with pytest.raises(MissingRunError):
        policies.run_policy(uuid4())


def test_link_new_run_records_active_version(client, policies):
    agent = create_run(client)
    other = create_run(client, service_id="fixture-service")
    store = client.app.state.store
    with store.connect() as conn:
        PolicyStore.link_new_run(conn, agent, run_registration(RUNNER_SERVICE_ID))
        PolicyStore.link_new_run(conn, other, run_registration("fixture-service"))
    baseline = policies.run_policy(agent)["versions"]
    assert [v["version"] for v in baseline] == [0]
    assert baseline[0]["action"] == "baseline"
    assert policies.run_policy(other)["versions"] == []
    policies.change(TARGET, change(0, limits={"test_jobs": 2}))
    assert [v["version"] for v in policies.run_policy(agent)["versions"]] == [0, 1]


def test_failed_change_rolls_back(client, policies, monkeypatch):
    create_run(client)
    original = PolicyStore.link_active_runs

    def link_then_fail(conn, target, version):
        original(conn, target, version)
        raise RuntimeError("injected failure after version insert")

    monkeypatch.setattr(PolicyStore, "link_active_runs", staticmethod(link_then_fail))
    with pytest.raises(RuntimeError):
        policies.change(TARGET, change(0, limits={"test_jobs": 1}))
    assert policies.active(TARGET)["version"] == 0
    assert link_count(client) == 0


def test_migration_upgrades_populated_0003_schema(client):
    schema = "aw_upgrade_" + uuid4().hex
    dsn = client.app.state.store.dsn
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    upgraded = RunStore(make_conninfo(dsn, options=f"-csearch_path={schema}"))
    legacy = [
        "0001_run_history.sql",
        "0002_live_lifecycle.sql",
        "0003_remove_duplicate_live_step_index.sql",
    ]
    try:
        with upgraded.connect() as conn:
            conn.execute("CREATE TABLE schema_migrations(version TEXT PRIMARY KEY)")
            for name in legacy:
                conn.execute(
                    files("agentwatch_worker").joinpath("migrations", name).read_text(),
                    prepare=False,
                )
                conn.execute("INSERT INTO schema_migrations VALUES (%s)", (name,))
        run = LiveStore(upgraded).create(
            RunCreate.model_validate(run_registration(RUNNER_SERVICE_ID))
        )
        upgraded.migrate()
        assert LiveStore(upgraded).get(UUID(run["run_id"]))["state"] == "queued"
        upgraded_policies = PolicyStore(upgraded)
        assert upgraded_policies.active(TARGET)["version"] == 0
        assert upgraded_policies.run_policy(UUID(run["run_id"]))["applicable"] is True
    finally:
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
