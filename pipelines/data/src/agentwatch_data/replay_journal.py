"""Read one replay's `live-journal.jsonl` into validated live-contracts-v1 records.

The journal is the agent runner's append-only log of what it sent to the live ingestion API:
its run registration, the run's creation and state changes, and every step event and
resource sample. It is untrusted input, read strictly: the file is size-bounded, every line
is strict JSON, every line kind is known, and every record must pass the same SDK contract the
ingestion service enforces. One bad line refuses the whole journal, naming the line, because
features computed from a partly readable journal would describe a run that was not recorded.
"""

from dataclasses import dataclass
from pathlib import Path

from agentwatch_sdk.live import ResourceSample, RunCreate, StepEvent
from pydantic import ValidationError

from agentwatch_data.extract import parse_jsonl, summarize
from agentwatch_data.replay_evidence import ReplayEvidenceError, read_evidence_text

LINE_KINDS = ("registration", "run_created", "run_change", "record")
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


def _validated[Model: RunCreate | StepEvent | ResourceSample](
    model: type[Model], payload: dict, what: str
) -> Model:
    try:
        return model.model_validate(payload)
    except ValidationError as error:
        raise ReplayEvidenceError(f"{what} is invalid: {summarize(error)}") from error
