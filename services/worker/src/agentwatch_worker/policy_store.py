"""Append-only versioned concurrency policy with run linkage (O02).

Version 0 is a synthesized baseline (no limits) and is never stored. Stored versions are
never updated or deleted; disable and restore append new versions.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from uuid import UUID

from agentwatch_sdk.policy import SUPPORTED_TARGETS, PolicyChange, PolicyVersion
from psycopg import Connection
from psycopg.types.json import Jsonb

from agentwatch_worker.live_store import MissingRunError
from agentwatch_worker.store import ConflictError, RunStore

# Lock order is always 730502 (live run creation) before 730503 (policy) to avoid deadlock.
POLICY_LOCK_KEY = 730503
ACTIVE_RUN_STATES = ("queued", "running")
VERSION_COLUMNS = (
    "target,version,action,limits,restored_from,operator,reason,request_id,created_at,"
    "controlled_services,enforcement_scope"
)


class UnsupportedTargetError(Exception):
    pass


class InvalidRestoreError(ValueError):
    pass


def target_definition(target: str) -> dict[str, Any]:
    definition = SUPPORTED_TARGETS.get(target)
    if definition is None:
        raise UnsupportedTargetError("Unsupported policy target")
    return definition


def baseline(target: str) -> dict[str, Any]:
    definition = target_definition(target)
    return {
        "target": target,
        "version": 0,
        "action": "baseline",
        "limits": {},
        "restored_from": None,
        "operator": None,
        "reason": None,
        "request_id": None,
        "created_at": None,
        "controlled_services": definition["controlled_services"],
        "enforcement_scope": definition["enforcement_scope"],
    }


def present(row: dict[str, Any]) -> dict[str, Any]:
    """Validate a stored or synthesized version and render it as JSON-compatible data."""
    version = PolicyVersion.model_validate(row)
    rendered = version.model_dump(mode="json")
    # Absent keys mean "no limit"; never render them as null limits.
    rendered["limits"] = version.limits.model_dump(exclude_none=True)
    return rendered


def runner_resources(target: str) -> Jsonb:
    return Jsonb([{"service_id": target_definition(target)["runner_service_id"]}])


class PolicyStore:
    def __init__(self, store: RunStore):
        self.store = store

    @contextmanager
    def connect(self) -> Iterator[Connection]:
        with self.store.connect() as conn:
            conn.execute("SET LOCAL statement_timeout = '5s'")
            conn.execute("SET LOCAL lock_timeout = '3s'")
            yield conn

    @staticmethod
    def stored_version(conn: Connection, target: str, version: int) -> dict[str, Any] | None:
        return conn.execute(
            f"SELECT {VERSION_COLUMNS} FROM concurrency_policy_versions "
            "WHERE target=%s AND version=%s",
            (target, version),
        ).fetchone()

    @staticmethod
    def current(conn: Connection, target: str) -> dict[str, Any]:
        row = conn.execute(
            f"SELECT {VERSION_COLUMNS} FROM concurrency_policy_versions "
            "WHERE target=%s ORDER BY version DESC LIMIT 1",
            (target,),
        ).fetchone()
        return row if row is not None else baseline(target)

    def active(self, target: str) -> dict[str, Any]:
        target_definition(target)
        with self.connect() as conn:
            return present(self.current(conn, target))

    def history(self, target: str, before: int | None, limit: int) -> dict[str, Any]:
        target_definition(target)
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if before is not None and before <= 0:
            return {"versions": [], "next_before": None}
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT {VERSION_COLUMNS} FROM concurrency_policy_versions "
                "WHERE target=%s AND (%s::integer IS NULL OR version < %s) "
                "ORDER BY version DESC LIMIT %s",
                (target, before, before, limit + 1),
            ).fetchall()
        # Version 0 is older than every stored version, so it closes the final page.
        versions = [*rows, baseline(target)]
        page = versions[:limit]
        return {
            "versions": [present(row) for row in page],
            "next_before": page[-1]["version"] if len(versions) > limit else None,
        }

    @staticmethod
    def link_active_runs(conn: Connection, target: str, version: int) -> None:
        """Link a new version to every unfinished run of the target's workload."""
        conn.execute(
            "INSERT INTO live_run_policy_links(run_id,target,version) "
            "SELECT run_id,%s,%s FROM live_runs WHERE state = ANY(%s) "
            "AND registration->'resources' @> %s ON CONFLICT DO NOTHING",
            (target, version, list(ACTIVE_RUN_STATES), runner_resources(target)),
        )

    def change(self, target: str, change: PolicyChange) -> dict[str, Any]:
        definition = target_definition(target)
        request = change.model_dump(mode="json")
        with self.connect() as conn:
            conn.execute("SELECT pg_advisory_xact_lock(%s)", (POLICY_LOCK_KEY,))
            # Idempotency precedes the revision check so a faithful retry succeeds even
            # after later versions were created.
            existing = conn.execute(
                f"SELECT {VERSION_COLUMNS},request=%s AS identical "
                "FROM concurrency_policy_versions WHERE target=%s AND request_id=%s",
                (Jsonb(request), target, change.request_id),
            ).fetchone()
            if existing is not None:
                if not existing.pop("identical"):
                    raise ConflictError("Policy request reused with a different body")
                return {"created": False, "policy": present(existing)}
            current = self.current(conn, target)["version"]
            if change.expected_version != current:
                raise ConflictError(
                    f"Policy changed: expected version {change.expected_version}, "
                    f"active version is {current}"
                )
            # PolicyChange guarantees set carries limits and restore carries restored_from.
            if change.action == "set" and change.limits is not None:
                limits = change.limits.model_dump(exclude_none=True)
            elif change.action == "disable" or change.restored_from == 0:
                limits = {}
            else:
                source = self.stored_version(conn, target, change.restored_from)
                if source is None:
                    raise InvalidRestoreError(
                        f"Policy version {change.restored_from} does not exist"
                    )
                limits = source["limits"]
            version = current + 1
            row = conn.execute(
                "INSERT INTO concurrency_policy_versions(target,version,action,limits,"
                "restored_from,operator,reason,request_id,request,controlled_services,"
                "enforcement_scope) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
                f"RETURNING {VERSION_COLUMNS}",
                (
                    target,
                    version,
                    change.action,
                    Jsonb(limits),
                    change.restored_from,
                    change.operator,
                    change.reason,
                    change.request_id,
                    Jsonb(request),
                    Jsonb(definition["controlled_services"]),
                    definition["enforcement_scope"],
                ),
            ).fetchone()
            self.link_active_runs(conn, target, version)
            return {"created": True, "policy": present(row)}

    @staticmethod
    def link_new_run(conn: Connection, run_id: UUID, registration: dict[str, Any]) -> None:
        """Link the active version of each applicable target inside the caller's transaction.

        Callers creating runs already hold lock 730502; taking 730503 here keeps the
        documented lock order and stops a concurrent change from missing this run.
        """
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (POLICY_LOCK_KEY,))
        services = {resource["service_id"] for resource in registration["resources"]}
        for target, definition in SUPPORTED_TARGETS.items():
            if definition["runner_service_id"] not in services:
                continue
            version = PolicyStore.current(conn, target)["version"]
            conn.execute(
                "INSERT INTO live_run_policy_links(run_id,target,version) VALUES (%s,%s,%s) "
                "ON CONFLICT DO NOTHING",
                (run_id, target, version),
            )

    def run_policy(self, run_id: UUID) -> dict[str, Any]:
        # O02 has a single target; a run's applicability is decided by its registration.
        (target,) = SUPPORTED_TARGETS
        with self.connect() as conn:
            run = conn.execute(
                "SELECT registration->'resources' @> %s AS applicable FROM live_runs "
                "WHERE run_id=%s",
                (runner_resources(target), run_id),
            ).fetchone()
            if run is None:
                raise MissingRunError("Live run not found")
            links = conn.execute(
                "SELECT version FROM live_run_policy_links WHERE run_id=%s AND target=%s "
                "ORDER BY version",
                (run_id, target),
            ).fetchall()
            versions = []
            for link in links:
                row = (
                    baseline(target)
                    if link["version"] == 0
                    else self.stored_version(conn, target, link["version"])
                )
                if row is None:
                    raise ConflictError("Run is linked to a missing policy version")
                versions.append(present(row))
        return {
            "run_id": str(run_id),
            "target": target,
            "applicable": bool(run["applicable"]),
            "versions": versions,
        }
