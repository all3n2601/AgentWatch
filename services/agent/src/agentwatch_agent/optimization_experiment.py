"""The experiment directory an optimization run reads (O06): manifest, schedule and records.

Only what the batch runner needs is read here. The data pipeline owns the manifest format, derives
the schedule from it and validates every record (`docs/optimization-evidence.md`), so nothing is
re-derived: the schedule must name the manifest's canonical hash, and slots run in its order.
"""

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from agentwatch_agent.admission import QueueBounds
from agentwatch_agent.replay import canonical_sha256

MANIFEST_FORMAT = "opt-experiment-manifest-v1"
MANIFEST_FILE = "manifest.json"
SCHEDULE_FILE = "schedule.json"
BATCHES_DIR = "batches"
MAX_DOCUMENT_BYTES = 1024 * 1024
CONTROLS = ("inference_requests", "test_jobs")
TASK_ID = re.compile(r"(?P<name>[a-z0-9][a-z0-9-]*)-r(?P<revision>[1-9][0-9]*)")


class ExperimentError(ValueError):
    """The experiment directory cannot be run as it is; nothing was attempted."""


@dataclass(frozen=True)
class FrozenTask:
    task_id: str
    name: str
    revision: int
    content_sha256: str
    configuration_digest: str
    offset_seconds: float


@dataclass(frozen=True)
class Slot:
    """One scheduled batch, exactly as `schedule.json` lists it."""

    index: int
    kind: str
    condition: str
    configuration: str
    candidate: str | None
    pair_index: int | None
    policy_sha256: str
    limits: dict[str, int]

    def head(self) -> dict:
        return {
            "index": self.index,
            "kind": self.kind,
            "condition": self.condition,
            "configuration": self.configuration,
            "candidate": self.candidate,
            "pair_index": self.pair_index,
        }


@dataclass(frozen=True)
class Experiment:
    root: Path
    experiment_id: str
    manifest_sha256: str
    tasks: tuple[FrozenTask, ...]
    injectors: dict[str, tuple[dict, ...]]  # per condition, in the manifest's shape
    queue_bounds: dict[str, QueueBounds]
    declared: dict  # the manifest's model, hardware, power and state blocks
    slots: tuple[Slot, ...]


def _read_json(path: Path) -> object:
    try:
        if path.stat().st_size > MAX_DOCUMENT_BYTES:
            raise ExperimentError(f"{path.name} is larger than {MAX_DOCUMENT_BYTES} bytes")
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ExperimentError(f"{path.name} cannot be read: {error}") from error

    def refuse(constant: str) -> None:
        raise ExperimentError(f"{path.name} holds {constant}, which has no canonical form")

    try:
        return json.loads(text, parse_constant=refuse)
    except json.JSONDecodeError as error:
        raise ExperimentError(f"{path.name} is not JSON: {error}") from error


def _object(value, name: str) -> dict:
    if not isinstance(value, dict):
        raise ExperimentError(f"{name} must be an object")
    return value


def _seconds(value, name: str, *, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ExperimentError(f"{name} must be a number of seconds")
    if value < 0 or (positive and value == 0):
        raise ExperimentError(f"{name} must be {'positive' if positive else 'non-negative'}")
    return float(value)


def _count(value, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ExperimentError(f"{name} must be an integer of at least {minimum}")
    return value


def read_experiment(root: Path) -> Experiment:
    """The parts of the manifest and schedule this runner needs; the data pipeline validates
    the rest when it compares the records."""
    manifest = _object(_read_json(root / MANIFEST_FILE), "manifest")
    if manifest.get("format") != MANIFEST_FORMAT:
        raise ExperimentError(f"manifest.format must be {MANIFEST_FORMAT!r}")
    digest = canonical_sha256(manifest)
    schedule = _object(_read_json(root / SCHEDULE_FILE), "schedule")
    if schedule.get("manifest_sha256") != digest:
        raise ExperimentError(
            "schedule.json was derived from another manifest; run make optimization-schedule again"
        )
    workload = _object(manifest.get("workload"), "workload")
    arrivals = _object(workload.get("arrivals"), "workload.arrivals")
    offsets = {}
    for entry in arrivals.get("offsets") or []:
        entry = _object(entry, "workload.arrivals.offsets[]")
        offsets[entry.get("task_id")] = _seconds(
            entry.get("offset_seconds"), "offset_seconds", positive=False
        )
    tasks = []
    for entry in workload.get("tasks") or []:
        entry = _object(entry, "workload.tasks[]")
        task_id = entry.get("task_id")
        match = TASK_ID.fullmatch(task_id) if isinstance(task_id, str) else None
        if match is None or task_id not in offsets:
            raise ExperimentError(f"workload task {task_id!r} needs a <name>-r<rev> ID and offset")
        tasks.append(
            FrozenTask(
                task_id=task_id,
                name=match["name"],
                revision=int(match["revision"]),
                content_sha256=entry.get("content_sha256"),
                configuration_digest=entry.get("configuration_digest"),
                offset_seconds=offsets[task_id],
            )
        )
    if not tasks:
        raise ExperimentError("The manifest freezes no tasks")
    injectors = {}
    for condition in manifest.get("conditions") or []:
        condition = _object(condition, "conditions[]")
        injectors[condition.get("name")] = tuple(condition.get("injectors") or ())
    bounds = _object(manifest.get("queue_bounds"), "queue_bounds")
    queue_bounds = {}
    for control in CONTROLS:
        declared = _object(bounds.get(control), f"queue_bounds.{control}")
        queue_bounds[control] = QueueBounds(
            _count(declared.get("max_length"), f"{control}.max_length", 1),
            _seconds(
                declared.get("max_wait_seconds"), f"{control}.max_wait_seconds", positive=True
            ),
        )
    slots = []
    for position, entry in enumerate(schedule.get("batches") or []):
        entry = _object(entry, "schedule.batches[]")
        if entry.get("index") != position:
            raise ExperimentError("schedule.batches must list slots 0, 1, 2, ... in order")
        if entry.get("condition") not in injectors:
            raise ExperimentError(f"slot {position} names an unknown condition")
        limits = _object(entry.get("limits"), f"slot {position} limits")
        slots.append(
            Slot(
                index=position,
                kind=entry.get("kind"),
                condition=entry["condition"],
                configuration=entry.get("configuration"),
                candidate=entry.get("candidate"),
                pair_index=entry.get("pair_index"),
                policy_sha256=entry.get("policy_sha256"),
                limits={control: _count(limits[control], control, 1) for control in limits},
            )
        )
    if not slots:
        raise ExperimentError("schedule.json lists no batches")
    return Experiment(
        root=root,
        experiment_id=str(manifest.get("experiment_id")),
        manifest_sha256=digest,
        tasks=tuple(tasks),
        injectors=injectors,
        queue_bounds=queue_bounds,
        declared={key: manifest.get(key) for key in ("model", "hardware", "power", "state")},
        slots=tuple(slots),
    )


def recorded_slots(root: Path) -> int:
    """How many slots already have records; they must be the schedule's first ones."""
    directory = root / BATCHES_DIR
    if not directory.exists():
        return 0
    indexes = sorted(int(path.stem) for path in directory.glob("*.json") if path.stem.isdigit())
    if indexes != list(range(len(indexes))):
        raise ExperimentError(
            f"batches/ holds records {indexes}, which are not the schedule's first slots"
        )
    return len(indexes)
