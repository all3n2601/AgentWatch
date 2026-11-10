"""Read one replay's `live-journal.jsonl` into validated live-contracts-v1 records.

The journal is the agent runner's append-only log of what it sent to the live ingestion API:
its run registration, the run's creation and state changes, every step event and resource
sample, and two delivery outcomes: batches the API refused (`rejected`) and closures another
actor made (`remote_closure`). It is untrusted input, read strictly: the file is size-bounded,
every line is strict JSON, every line kind is known, and every record must pass the same SDK
contract the ingestion service enforces. One bad line refuses the whole journal, naming the
line, because features computed from a partly readable journal would describe a run that was
not recorded.

Delivery outcomes are validated in the exact shape the runner writes them but add nothing to
the events and samples. Records named in a `rejected` line still count: the journal is the
agent's own measurement, and the live API refusing to ingest a record is a delivery outcome,
not a measurement defect.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import get_args
from uuid import UUID

from agentwatch_sdk.live import ResourceSample, RunCreate, State, StepEvent
from pydantic import ValidationError

from agentwatch_data.extract import parse_jsonl, summarize
from agentwatch_data.replay_evidence import ReplayEvidenceError, read_evidence_text

LINE_KINDS = ("registration", "run_created", "run_change", "record", "rejected", "remote_closure")
# The exact keys of each delivery-outcome line and payload, as agentwatch_agent writes them:
# LiveDelivery._settle journals `rejected` with the refused record IDs beside the payload, and
# RunSession.adopt_remote_closure journals the state it observed and its reason.
OUTCOME_LINE_KEYS = {
    "rejected": {"kind", "payload", "record_ids"},
    "remote_closure": {"kind", "payload"},
}
OUTCOME_PAYLOAD_KEYS = {"rejected": {"status", "detail"}, "remote_closure": {"state", "reason"}}
REFUSAL_STATUSES = range(400, 600)  # LiveRejectedError carries the HTTP error status
RUN_STATES = get_args(State)
RECORD_MODELS = {"step_event": StepEvent, "resource_sample": ResourceSample}


@dataclass(frozen=True)
class Journal:
    events: tuple[StepEvent, ...]
    samples: tuple[ResourceSample, ...]
    configuration_sha256: str | None  # the registration's digest; None without a registration


def read_journal(path: Path) -> Journal:
    text = read_evidence_text(path)  # bounded by MAX_EVIDENCE_BYTES, regular file, UTF-8
    try:
        rows = parse_jsonl(text, path.name)
    except ValueError as error:
        raise ReplayEvidenceError(str(error)) from error
    # parse_jsonl skips blank lines; keep each row's line number in the file for errors.
    numbers = [number for number, line in enumerate(text.splitlines(), 1) if line.strip()]
    events: list[StepEvent] = []
    samples: list[ResourceSample] = []
    configuration_sha256 = None
    run_id = None
    for number, row in zip(numbers, rows, strict=True):
        where = f"{path.name} line {number}"
        if not isinstance(row, dict) or not isinstance(row.get("payload"), dict):
            raise ReplayEvidenceError(f"{where}: must be an object with an object payload")
        kind, payload = row.get("kind"), row["payload"]
        if kind not in LINE_KINDS:
            raise ReplayEvidenceError(f"{where}: unknown kind {kind!r}")
        if kind == "registration":
            if configuration_sha256 is not None:
                raise ReplayEvidenceError(f"{where}: the run is registered twice")
            configuration_sha256 = _validated(
                RunCreate, payload, f"{where}: registration"
            ).configuration_digest
        if kind in OUTCOME_LINE_KEYS:
            check_outcome(row, f"{where}: {kind}")
            continue
        if kind != "record":
            continue
        record_type = payload.get("record_type")
        model = RECORD_MODELS.get(record_type) if isinstance(record_type, str) else None
        if model is None:
            raise ReplayEvidenceError(f"{where}: unknown record_type {record_type!r}")
        record = _validated(model, payload, f"{where}: {record_type}")
        if run_id is None:
            run_id = record.run_id
        elif record.run_id != run_id:
            raise ReplayEvidenceError(f"{where}: record for another run {record.run_id}")
        (events if isinstance(record, StepEvent) else samples).append(record)
    return Journal(tuple(events), tuple(samples), configuration_sha256)


def check_outcome(row: dict, what: str) -> None:
    """Refuse a delivery-outcome line that is not exactly what the agent runner writes."""
    kind, payload = row["kind"], row["payload"]
    keys, expected = set(row), OUTCOME_LINE_KEYS[kind]
    if keys - expected:
        raise ReplayEvidenceError(f"{what} has unexpected keys {sorted(keys - expected)}")
    if expected - keys:
        raise ReplayEvidenceError(f"{what} lacks {sorted(expected - keys)}")
    if set(payload) != OUTCOME_PAYLOAD_KEYS[kind]:
        raise ReplayEvidenceError(
            f"{what} payload keys must be {sorted(OUTCOME_PAYLOAD_KEYS[kind])}, "
            f"got {sorted(payload)}"
        )
    if kind == "rejected":
        status = payload["status"]
        if (
            isinstance(status, bool)
            or not isinstance(status, int)
            or status not in REFUSAL_STATUSES
        ):
            raise ReplayEvidenceError(f"{what} status must be an HTTP error status, got {status!r}")
        if not isinstance(payload["detail"], str):
            raise ReplayEvidenceError(f"{what} detail must be a string")
        ids = row["record_ids"]
        if (
            not isinstance(ids, list)
            or not ids
            or not all(isinstance(value, str) and is_uuid(value) for value in ids)
            or len(set(ids)) != len(ids)
        ):
            raise ReplayEvidenceError(f"{what} record_ids must be a non-empty list of unique UUIDs")
        return
    if payload["state"] not in RUN_STATES:
        raise ReplayEvidenceError(f"{what} state must be one of {list(RUN_STATES)}")
    if payload["reason"] is not None and not isinstance(payload["reason"], str):
        raise ReplayEvidenceError(f"{what} reason must be a string or null")


def is_uuid(value: str) -> bool:
    """A UUID in the canonical lowercase hyphenated form `str(UUID)` writes."""
    try:
        return str(UUID(value)) == value
    except ValueError:
        return False


def _validated[Model: RunCreate | StepEvent | ResourceSample](
    model: type[Model], payload: dict, what: str
) -> Model:
    try:
        return model.model_validate(payload)
    except ValidationError as error:
        raise ReplayEvidenceError(f"{what} is invalid: {summarize(error)}") from error
