"""Read one replay experiment directory into validated, typed evidence.

Everything on disk is untrusted input: files are size-bounded and parsed with the same strict
JSON as `extract.py` (unique keys, bounded depth), paths named by the manifest must stay
inside the experiment directory, and roles, conditions and durations are checked before use.
The condition and injection data read here are labeling metadata; they must never become
model features.

Missing and unreadable evidence is kept distinct from measured values: a step duration that
was not recorded is `None`, a replay whose steps could not be read has no steps, and a step
the replay evidence says nothing about has `matches_source = None`.
"""

import math
from dataclasses import dataclass
from pathlib import Path, PurePath

from agentwatch_data.extract import parse_json, parse_jsonl

MAX_EVIDENCE_BYTES = 64 * 1024 * 1024
MANIFEST_FILE = "experiment.json"
EVIDENCE_FILE = "agent-evidence.json"
JOURNAL_FILE = "live-journal.jsonl"
ROLES = ("warmup", "baseline", "contended", "clean_control")
CONDITIONS = ("clean", "cpu", "inference", "retrieval", "mixed")


class ReplayEvidenceError(ValueError):
    """The experiment directory cannot be read as trustworthy replay evidence."""


@dataclass(frozen=True)
class StepTiming:
    step_id: str
    step_type: str
    state: str
    total_seconds: float | None
    queue_seconds: float | None
    execution_seconds: float | None
    is_leaf: bool
    matches_source: bool | None  # None when the replay evidence has no entry


@dataclass(frozen=True)
class Injector:
    method: str
    resource_class: str


@dataclass(frozen=True)
class Replay:
    index: int
    role: str
    condition: str
    state: str
    task_passed: bool | None
    run_id: str | None
    injection_active: bool | None
    injectors: tuple[Injector, ...]  # empty for clean replays
    steps: dict[str, StepTiming]  # empty when unreadable


@dataclass(frozen=True)
class Experiment:
    name: str  # directory name
    status: str
    checks: dict[str, bool]
    task_id: str
    plan_sha256: str
    model: str
    replays: tuple[Replay, ...]


def read_experiment(directory: Path) -> Experiment:
    root = directory.resolve()
    manifest = _object(_read_json(root / MANIFEST_FILE), MANIFEST_FILE)
    plan = _object(manifest.get("plan"), "plan")
    checks = _object(manifest.get("checks"), "checks")
    if not all(isinstance(passed, bool) for passed in checks.values()):
        raise ReplayEvidenceError("checks must map each check name to true or false")
    entries = manifest.get("replays")
    if not isinstance(entries, list):
        raise ReplayEvidenceError("replays must be a list")
    return Experiment(
        name=directory.name,
        status=_text(manifest, "status"),
        checks=dict(checks),
        task_id=_text(plan, "task_id"),
        plan_sha256=_text(plan, "plan_sha256"),
        model=_text(plan, "model"),
        replays=tuple(_read_replay(root, _object(entry, "replay")) for entry in entries),
    )


def _read_replay(root: Path, entry: dict) -> Replay:
    role, condition = _text(entry, "role"), _text(entry, "condition")
    if role not in ROLES:
        raise ReplayEvidenceError(f"Unknown replay role {role!r}")
    if condition not in CONDITIONS:
        raise ReplayEvidenceError(f"Unknown replay condition {condition!r}")
    index = entry.get("index")
    if not isinstance(index, int) or isinstance(index, bool):
        raise ReplayEvidenceError("Replay index must be an integer")
    raw_steps = entry.get("steps")
    steps: dict[str, StepTiming] = {}
    if raw_steps is not None:
        steps = _read_steps(root, _text(entry, "evidence_dir"), _object(raw_steps, "steps"))
    return Replay(
        index=index,
        role=role,
        condition=condition,
        state=_text(entry, "state"),
        task_passed=_optional_bool(entry, "task_passed"),
        run_id=_optional_text(entry, "run_id"),
        injection_active=_optional_bool(entry, "injection_active"),
        injectors=_read_injectors(root, _optional_text(entry, "injection_log")),
        steps=steps,
    )


def _read_steps(root: Path, evidence_dir: str, raw_steps: dict) -> dict[str, StepTiming]:
    replay_dir = _confined(root, evidence_dir)
    matches = _matches_source(_object(_read_json(replay_dir / EVIDENCE_FILE), EVIDENCE_FILE))
    parents = _parent_step_ids(replay_dir / JOURNAL_FILE)
    steps = {}
    for step_id, raw in raw_steps.items():
        timing = _object(raw, f"step {step_id}")
        steps[step_id] = StepTiming(
            step_id=step_id,
            step_type=_text(timing, "step_type"),
            state=_text(timing, "state"),
            total_seconds=_duration(timing, "total_seconds"),
            queue_seconds=_duration(timing, "queue_seconds"),
            execution_seconds=_duration(timing, "execution_seconds"),
            is_leaf=step_id not in parents,
            matches_source=matches.get(step_id),
        )
    return steps


def _matches_source(evidence: dict) -> dict[str, bool]:
    replay = _object(evidence.get("replay"), "replay")
    matches: dict[str, bool] = {}
    for key in ("model_outputs", "tool_results"):
        items = replay.get(key)
        if not isinstance(items, list):
            raise ReplayEvidenceError(f"{EVIDENCE_FILE}: replay.{key} must be a list")
        for item in items:
            item = _object(item, f"replay.{key} entry")
            matched = item.get("matches_source")
            if not isinstance(matched, bool):
                raise ReplayEvidenceError(f"{EVIDENCE_FILE}: matches_source must be true or false")
            matches[_text(item, "step_id")] = matched
    return matches


def _parent_step_ids(journal: Path) -> set[str]:
    try:
        rows = parse_jsonl(_read_text(journal), journal.name)
    except ValueError as error:
        raise ReplayEvidenceError(f"{journal.name}: {error}") from error
    parents = set()
    for number, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ReplayEvidenceError(f"{journal.name} row {number} must be an object")
        payload = row.get("payload") if row.get("kind") == "record" else None
        if isinstance(payload, dict) and payload.get("record_type") == "step_event":
            parent = payload.get("parent_step_id")
            if isinstance(parent, str):
                parents.add(parent)
    return parents


def _read_injectors(root: Path, log_path: str | None) -> tuple[Injector, ...]:
    if log_path is None:
        return ()
    log = _object(_read_json(_confined(root, log_path)), "injection log")
    injections = log.get("injections")
    if not isinstance(injections, list):
        raise ReplayEvidenceError("injection log injections must be a list")
    return tuple(
        Injector(
            method=_text(injection, "method"),
            resource_class=_text(injection, "contended_resource_class"),
        )
        for injection in (_object(item, "injection") for item in injections)
    )


def _confined(root: Path, relative: str) -> Path:
    """Resolve a manifest path inside the experiment; reject absolute and escaping paths."""
    if PurePath(relative).is_absolute():
        raise ReplayEvidenceError(f"Path {relative!r} is outside the experiment directory")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ReplayEvidenceError(f"Path {relative!r} is outside the experiment directory")
    return resolved


def _read_text(path: Path) -> str:
    try:
        # Read one byte past the limit so an oversize file is detected without loading it all.
        with path.open("rb") as handle:
            raw = handle.read(MAX_EVIDENCE_BYTES + 1)
    except OSError as error:
        raise ReplayEvidenceError(f"{path.name} cannot be read: {error}") from error
    if len(raw) > MAX_EVIDENCE_BYTES:
        raise ReplayEvidenceError(f"{path.name} is larger than {MAX_EVIDENCE_BYTES} bytes")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ReplayEvidenceError(f"{path.name} is not valid UTF-8: {error}") from error


def _read_json(path: Path):
    text = _read_text(path)
    try:
        return parse_json(text)
    except ValueError as error:
        raise ReplayEvidenceError(f"{path.name}: {error}") from error


def _object(value, name: str) -> dict:
    if not isinstance(value, dict):
        raise ReplayEvidenceError(f"{name} must be an object")
    return value


def _text(container: dict, key: str) -> str:
    value = container.get(key)
    if not isinstance(value, str) or not value:
        raise ReplayEvidenceError(f"{key} must be a non-empty string")
    return value


def _optional_text(container: dict, key: str) -> str | None:
    return None if container.get(key) is None else _text(container, key)


def _optional_bool(container: dict, key: str) -> bool | None:
    value = container.get(key)
    if value is not None and not isinstance(value, bool):
        raise ReplayEvidenceError(f"{key} must be true, false or null")
    return value


def _duration(container: dict, key: str) -> float | None:
    """A finite, nonnegative number of seconds, or None when the evidence did not record one."""
    value = container.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ReplayEvidenceError(f"{key} must be a finite number or null, got {value!r}")
    if value < 0:
        raise ReplayEvidenceError(f"{key} must not be negative, got {value!r}")
    return float(value)
