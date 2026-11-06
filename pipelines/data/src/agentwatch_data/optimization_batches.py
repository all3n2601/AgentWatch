"""Batch records of an optimization experiment, checked against its manifest (O04).

Each attempted batch, including warm-ups and failed or aborted ones, has one record in
`batches/<index>.json`, written by the experiment runner. A record must name the manifest's
hash, sit in its preregistered schedule slot, apply that slot's policy, and report the same
model, hardware, power and test/retrieval state the manifest froze. A record that does not is
rejected with a code and kept in the report: nothing is dropped or repaired.

Tasks may link their agent run evidence; when they do, the run ID, task content hash and
configuration digest must match. Missing measurements stay null: an admission statistic the
runner could not observe is unavailable, not zero.
"""

import dataclasses
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from agentwatch_data.extract import TIMESTAMP, parse_json
from agentwatch_data.files import confined_path, file_sha256, read_bounded_text
from agentwatch_data.optimization_manifest import (
    SHA256,
    TASK_ID,
    Manifest,
    ManifestError,
    OptimizationContract,
    ScheduledBatch,
    hardware_identity,
    model_identity,
    policy_limits,
    policy_sha256,
    power_identity,
    state_identity,
)

BATCH_FORMAT = "opt-batch-v1"
BATCHES_DIR = "batches"
EVIDENCE_FILE = "agent-evidence.json"
BATCH_FILE = re.compile(r"(\d{3,6})\.json")
MAX_BATCH_BYTES = 4 * 1024 * 1024
MAX_EVIDENCE_BYTES = 64 * 1024 * 1024
MAX_TICK_NS = 2**53 - 1  # the live contract's monotonic tick bound
BATCH_STATUSES = ("completed", "failed", "aborted", "not_started")
TASK_STATES = ("completed", "failed", "timed_out", "cancelled", "not_started")
TERMINAL_TASK_STATES = ("completed", "failed", "timed_out", "cancelled")
ACCEPTANCE = ("passed", "failed")
REJECTION_CODES = (
    "malformed",
    "manifest_mismatch",
    "schedule_mismatch",
    "policy_mismatch",
    "model_mismatch",
    "hardware_mismatch",
    "power_mismatch",
    "state_mismatch",
    "task_mismatch",
    "task_identity_mismatch",
    "inconsistent_status",
    "duplicate_identity",
)


class BatchRejected(Exception):
    """One batch record cannot be used; its code says why."""

    def __init__(self, code: str, detail: str) -> None:
        if code not in REJECTION_CODES:
            raise ValueError(f"Unknown batch rejection code {code!r}")
        super().__init__(f"{code}: {detail}")
        self.code, self.detail = code, detail


@dataclass(frozen=True)
class TaskResult:
    task_id: str
    run_id: str | None
    state: str
    acceptance: str | None  # None: the acceptance outcome was not recorded
    cold: bool
    submitted_tick_ns: int | None
    finished_tick_ns: int | None
    admission_wait_seconds: float | None  # None until the runner records admission (O03)
    evidence_linked: bool

    @property
    def latency_seconds(self) -> float | None:
        if self.submitted_tick_ns is None or self.finished_tick_ns is None:
            return None
        return (self.finished_tick_ns - self.submitted_tick_ns) / 1e9


@dataclass(frozen=True)
class BatchRecord:
    file: str
    file_sha256: str
    batch_id: str
    slot: ScheduledBatch
    status: str
    reason: str | None
    policy_version: int | None  # the stored policy version applied, when the runner recorded it
    injection_active: bool | None
    starved_admissions: int | None
    leaked_permits: int | None
    tasks: tuple[TaskResult, ...]

    @property
    def batch_seconds(self) -> float | None:
        """First submission to last terminal task, or None when any task's ticks are missing."""
        ticks = [(task.submitted_tick_ns, task.finished_tick_ns) for task in self.tasks]
        if not ticks or any(None in pair for pair in ticks):
            return None
        return (max(finished for _, finished in ticks) - min(start for start, _ in ticks)) / 1e9


@dataclass(frozen=True)
class BatchRejection:
    file: str
    file_sha256: str | None
    code: str
    detail: str


# --- Field readers ----------------------------------------------------------------------------


def need(condition: bool, detail: str, code: str = "malformed") -> None:
    if not condition:
        raise BatchRejected(code, detail)


def fields(value, name: str, keys: set[str]) -> dict:
    need(isinstance(value, dict), f"{name} must be an object")
    missing, unknown = sorted(keys - set(value)), sorted(set(value) - keys)
    need(not missing and not unknown, f"{name} keys: missing {missing}, unknown {unknown}")
    return value


def optional_count(value, name: str) -> int | None:
    if value is None:
        return None
    need(
        not isinstance(value, bool) and isinstance(value, int) and value >= 0,
        f"{name} must be a nonnegative integer or null",
    )
    return value


def optional_tick(value, name: str) -> int | None:
    if value is None:
        return None
    need(
        not isinstance(value, bool) and isinstance(value, int) and 0 <= value <= MAX_TICK_NS,
        f"{name} must be an integer tick from 0 to {MAX_TICK_NS} or null",
    )
    return value


def optional_bool(value, name: str) -> bool | None:
    need(value is None or isinstance(value, bool), f"{name} must be true, false or null")
    return value


def optional_text(value, name: str) -> str | None:
    if value is None:
        return None
    need(isinstance(value, str) and bool(value.strip()), f"{name} must be a non-empty string")
    return value


def optional_uuid(value, name: str) -> str | None:
    text = optional_text(value, name)
    if text is None:
        return None
    try:
        return str(UUID(text))
    except ValueError:
        raise BatchRejected("malformed", f"{name} must be a UUID") from None


def optional_timestamp(value, name: str) -> str | None:
    text = optional_text(value, name)
    need(text is None or bool(TIMESTAMP.fullmatch(text)), f"{name} must be RFC 3339 with a zone")
    return text


def identity(read, value, name: str, expected: dict, code: str) -> None:
    """An observed identity section, valid and identical to the manifest's."""
    try:
        observed = read(value, name)
    except ManifestError as error:
        raise BatchRejected("malformed", str(error)) from error
    if observed != expected:
        differing = sorted(key for key in expected if observed.get(key) != expected[key])
        raise BatchRejected(code, f"{name} differs from the manifest in {differing}")


# --- Tasks ------------------------------------------------------------------------------------


TASK_KEYS = {
    "task_id",
    "run_id",
    "state",
    "acceptance",
    "cold",
    "submitted_tick_ns",
    "finished_tick_ns",
    "admission_wait_seconds",
    "evidence_dir",
}


def read_task(value, index: int) -> tuple[TaskResult, str | None]:
    name = f"tasks[{index}]"
    entry = fields(value, name, TASK_KEYS)
    task_id = entry["task_id"]
    need(isinstance(task_id, str) and bool(TASK_ID.fullmatch(task_id)), f"{name}.task_id invalid")
    need(entry["state"] in TASK_STATES, f"{name}.state must be one of {TASK_STATES}")
    need(
        entry["acceptance"] is None or entry["acceptance"] in ACCEPTANCE,
        f"{name}.acceptance must be one of {ACCEPTANCE} or null",
    )
    need(isinstance(entry["cold"], bool), f"{name}.cold must be true or false")
    submitted = optional_tick(entry["submitted_tick_ns"], f"{name}.submitted_tick_ns")
    finished = optional_tick(entry["finished_tick_ns"], f"{name}.finished_tick_ns")
    need(
        submitted is None or finished is None or finished >= submitted,
        f"{name} finished before it was submitted",
    )
    wait = entry["admission_wait_seconds"]
    need(
        wait is None
        or (
            not isinstance(wait, bool)
            and isinstance(wait, int | float)
            and math.isfinite(wait)
            and wait >= 0
        ),
        f"{name}.admission_wait_seconds must be a nonnegative number or null",
    )
    if entry["state"] == "not_started":
        need(finished is None, f"{name} never started but has a finish tick")
    evidence_dir = optional_text(entry["evidence_dir"], f"{name}.evidence_dir")
    result = TaskResult(
        task_id=task_id,
        run_id=optional_uuid(entry["run_id"], f"{name}.run_id"),
        state=entry["state"],
        acceptance=entry["acceptance"],
        cold=entry["cold"],
        submitted_tick_ns=submitted,
        finished_tick_ns=finished,
        admission_wait_seconds=None if wait is None else float(wait),
        evidence_linked=evidence_dir is not None,
    )
    return result, evidence_dir


def check_task_evidence(
    root: Path, evidence_dir: str, task: TaskResult, manifest: Manifest
) -> None:
    """The linked agent run is this task's run, of the frozen task revision and configuration."""
    try:
        directory = confined_path(root, evidence_dir, ManifestError)
        evidence = parse_json(
            read_bounded_text(directory / EVIDENCE_FILE, MAX_EVIDENCE_BYTES, ManifestError)
        )
    except ValueError as error:
        raise BatchRejected("task_identity_mismatch", f"{task.task_id}: {error}") from error
    expected = next(item for item in manifest.tasks if item.task_id == task.task_id)
    if not isinstance(evidence, dict) or not isinstance(evidence.get("task"), dict):
        raise BatchRejected("task_identity_mismatch", f"{task.task_id}: evidence has no task")
    checks = {
        "run_id": (evidence.get("run_id"), task.run_id),
        "task_id": (evidence["task"].get("task_id"), task.task_id),
        "content_sha256": (evidence["task"].get("content_sha256"), expected.content_sha256),
        "configuration_digest": (
            evidence.get("configuration_digest"),
            expected.configuration_digest,
        ),
    }
    differing = sorted(key for key, (found, wanted) in checks.items() if found != wanted)
    if task.run_id is None or differing:
        raise BatchRejected(
            "task_identity_mismatch",
            f"{task.task_id}: linked evidence differs in {differing or ['run_id']}",
        )


def read_tasks(entry: dict, root: Path, manifest: Manifest) -> tuple[TaskResult, ...]:
    raw = entry["tasks"]
    need(isinstance(raw, list) and len(raw) > 0, "tasks must be a non-empty list")
    parsed = [read_task(value, index) for index, value in enumerate(raw)]
    task_ids = [task.task_id for task, _ in parsed]
    if len(set(task_ids)) != len(task_ids) or set(task_ids) != {t.task_id for t in manifest.tasks}:
        raise BatchRejected("task_mismatch", "a batch must run every frozen task exactly once")
    for task, evidence_dir in parsed:
        if evidence_dir is not None:
            check_task_evidence(root, evidence_dir, task, manifest)
    return tuple(sorted((task for task, _ in parsed), key=lambda task: task.task_id))


# --- Batches ----------------------------------------------------------------------------------


BATCH_KEYS = {
    "format",
    "manifest_sha256",
    "batch_id",
    "index",
    "kind",
    "condition",
    "configuration",
    "candidate",
    "pair_index",
    "policy",
    "status",
    "reason",
    "started_at",
    "finished_at",
    "clock_id",
    "model",
    "hardware",
    "power",
    "state",
    "injection",
    "admission",
    "tasks",
}


def check_slot(entry: dict, slot: ScheduledBatch | None) -> ScheduledBatch:
    if slot is None:
        raise BatchRejected("schedule_mismatch", f"index {entry['index']!r} is not scheduled")
    declared = {
        "kind": entry["kind"],
        "condition": entry["condition"],
        "configuration": entry["configuration"],
        "candidate": entry["candidate"],
        "pair_index": entry["pair_index"],
    }
    planned = {
        "kind": slot.kind,
        "condition": slot.condition,
        "configuration": slot.configuration,
        "candidate": slot.candidate,
        "pair_index": slot.pair_index,
    }
    differing = sorted(key for key in planned if declared[key] != planned[key])
    if differing:
        raise BatchRejected("schedule_mismatch", f"slot {slot.index} differs in {differing}")
    return slot


def check_policy(entry: dict, slot: ScheduledBatch, contract: OptimizationContract) -> int | None:
    policy = fields(entry["policy"], "policy", {"sha256", "limits", "version"})
    try:
        limits = policy_limits(policy["limits"], "policy.limits", contract)
    except ManifestError as error:
        raise BatchRejected("malformed", str(error)) from error
    version = optional_count(policy["version"], "policy.version")
    if policy["sha256"] != policy_sha256(limits, contract):
        raise BatchRejected("policy_mismatch", "policy.sha256 does not identify policy.limits")
    if policy["sha256"] != slot.policy_sha256:
        raise BatchRejected("policy_mismatch", f"slot {slot.index} schedules another policy")
    return version


def check_status(entry: dict, tasks: tuple[TaskResult, ...]) -> None:
    status = entry["status"]
    need(status in BATCH_STATUSES, f"status must be one of {BATCH_STATUSES}")
    if status == "completed":
        need(
            all(task.state in TERMINAL_TASK_STATES for task in tasks),
            "a completed batch has a task that never started",
            "inconsistent_status",
        )
        need(
            entry["finished_at"] is not None,
            "a completed batch needs finished_at",
            "inconsistent_status",
        )
    if status == "not_started":
        need(
            all(task.state == "not_started" for task in tasks),
            "a batch that never started has started tasks",
            "inconsistent_status",
        )
    if status != "completed":
        need(
            entry["reason"] is not None,
            f"a {status} batch must record its reason",
            "inconsistent_status",
        )
    if any(task.submitted_tick_ns is not None for task in tasks):
        need(entry["clock_id"] is not None, "ticks need the clock_id they were read from")


def parse_batch(
    entry,
    file: str,
    root: Path,
    manifest: Manifest,
    slots: dict[int, ScheduledBatch],
    contract: OptimizationContract,
) -> BatchRecord:
    entry = fields(entry, "batch", BATCH_KEYS)
    need(entry["format"] == BATCH_FORMAT, f"format must be {BATCH_FORMAT!r}")
    need(
        isinstance(entry["manifest_sha256"], str)
        and bool(SHA256.fullmatch(entry["manifest_sha256"])),
        "manifest_sha256 must be a SHA-256 hex digest",
    )
    if entry["manifest_sha256"] != manifest.sha256:
        raise BatchRejected("manifest_mismatch", "batch was recorded for another manifest")
    batch_id = optional_uuid(entry["batch_id"], "batch_id")
    need(batch_id is not None, "batch_id is required")
    index = entry["index"]
    need(not isinstance(index, bool) and isinstance(index, int), "index must be an integer")
    need(file == f"{index:03d}.json", f"index {index} must be stored as {index:03d}.json")
    slot = check_slot(entry, slots.get(index))
    version = check_policy(entry, slot, contract)
    optional_text(entry["reason"], "reason")
    optional_text(entry["clock_id"], "clock_id")
    optional_timestamp(entry["started_at"], "started_at")
    optional_timestamp(entry["finished_at"], "finished_at")
    identity(model_identity, entry["model"], "model", manifest.model, "model_mismatch")
    identity(
        hardware_identity, entry["hardware"], "hardware", manifest.hardware, "hardware_mismatch"
    )
    identity(power_identity, entry["power"], "power", manifest.power, "power_mismatch")
    identity(state_identity, entry["state"], "state", manifest.state, "state_mismatch")
    injection = fields(entry["injection"], "injection", {"active", "cleanup_confirmed"})
    admission = fields(entry["admission"], "admission", {"starved_admissions", "leaked_permits"})
    tasks = read_tasks(entry, root, manifest)
    check_status(entry, tasks)
    return BatchRecord(
        file=file,
        file_sha256="",  # filled by the caller, which hashed the bytes it parsed
        batch_id=batch_id,
        slot=slot,
        status=entry["status"],
        reason=entry["reason"],
        policy_version=version,
        injection_active=optional_bool(injection["active"], "injection.active"),
        starved_admissions=optional_count(admission["starved_admissions"], "starved_admissions"),
        leaked_permits=optional_count(admission["leaked_permits"], "leaked_permits"),
        tasks=tasks,
    )


def batch_files(root: Path) -> list[Path]:
    directory = root / BATCHES_DIR
    if directory.is_symlink():
        raise ManifestError(f"{BATCHES_DIR}/ must not be a symlink")
    if not directory.exists():
        return []
    if not directory.is_dir():
        raise ManifestError(f"{BATCHES_DIR} must be a directory")
    return sorted(path for path in directory.iterdir() if not path.name.startswith("."))


def read_batches(
    root: Path,
    manifest: Manifest,
    slots: tuple[ScheduledBatch, ...],
    contract: OptimizationContract,
) -> tuple[list[BatchRecord], list[BatchRejection]]:
    """Every batch record, valid or rejected with its reason; nothing is dropped."""
    by_index = {slot.index: slot for slot in slots}
    records: list[BatchRecord] = []
    rejections: list[BatchRejection] = []
    for path in batch_files(root):
        digest = None if path.is_symlink() or not path.is_file() else file_sha256(path)
        try:
            if not BATCH_FILE.fullmatch(path.name):
                raise BatchRejected("malformed", "batch files are named <index>.json")
            try:
                document = parse_json(read_bounded_text(path, MAX_BATCH_BYTES, ManifestError))
            except ValueError as error:
                raise BatchRejected("malformed", str(error)) from error
            record = parse_batch(document, path.name, root, manifest, by_index, contract)
            records.append(dataclasses.replace(record, file_sha256=digest))
        except BatchRejected as rejected:
            rejections.append(BatchRejection(path.name, digest, rejected.code, rejected.detail))
    return reject_shared_identities(records, rejections)


def reject_shared_identities(
    records: list[BatchRecord], rejections: list[BatchRejection]
) -> tuple[list[BatchRecord], list[BatchRejection]]:
    """Reject every record that shares a batch ID, slot or run ID with another.

    Which copy is genuine cannot be decided from the records, so none is kept.
    """
    owners: dict[tuple[str, object], set[str]] = defaultdict(set)
    for record in records:
        owners[("batch_id", record.batch_id)].add(record.file)
        owners[("index", record.slot.index)].add(record.file)
        for task in record.tasks:
            if task.run_id is not None:
                owners[("run_id", task.run_id)].add(record.file)
    shared: dict[str, list[str]] = defaultdict(list)
    for (kind, value), files in owners.items():
        if len(files) > 1:
            for file in files:
                shared[file].append(f"{kind} {value}")
    kept = [record for record in records if record.file not in shared]
    rejections = rejections + [
        BatchRejection(
            record.file,
            record.file_sha256,
            "duplicate_identity",
            "shares " + ", ".join(sorted(shared[record.file])),
        )
        for record in records
        if record.file in shared
    ]
    return kept, sorted(rejections, key=lambda rejection: rejection.file)
