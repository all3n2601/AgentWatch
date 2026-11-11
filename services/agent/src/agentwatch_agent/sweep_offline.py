"""Analyse a recorded capacity sweep again from its folder, without vLLM, a GPU or the live API.

`sweep-analyse DIR` reads `DIR/sweep.json` and the batch folders it names, recomputes every ramp
batch's metrics from disk, then the level summaries, the capacity decision (excluding the
batches the driver marked, as the sweep itself does) and label validation, and writes
`DIR/analysis.json`. `sweep.json` is never modified, so the recorded and the offline analysis
can be compared. Use it when a sweep failed before or during its analysis, or to re-check one.

Every run folder is read under `DIR` only, never from the path a runner recorded, so a copied
sweep is analysed from its own files. A record whose fields the analysis reads are malformed is
refused with `SweepFolderError` (the command exits 2), never a traceback.
"""

import json
import math
from datetime import UTC, datetime
from pathlib import Path

from agentwatch_agent.runner import write_json_atomic
from agentwatch_agent.sweep import SWEEP_FILE, SWEEP_METHOD, analysis_scope
from agentwatch_agent.sweep_analysis import _is_number, batch_metrics_or_error, ramp_analysis
from agentwatch_agent.sweep_labels import ramp_label_results
from agentwatch_agent.sweep_protocol import protocol_sha256

ANALYSIS_FILE = "analysis.json"
OFFLINE_METHOD = "capacity-sweep-offline-analysis-v1"


class SweepFolderError(ValueError):
    """The folder holds no `sweep.json` that can be analysed."""


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _ramp_entry_problem(entry: dict) -> str | None:
    """What is malformed in a ramp batch entry the analysis reads, or None."""
    folder, runs = entry.get("folder"), entry.get("runs")
    if not (
        isinstance(folder, str)
        and _is_int(entry.get("level"))
        and _is_int(entry.get("sequence"))
        and isinstance(runs, list)
        and all(isinstance(run, dict) for run in runs)
    ):
        return "a ramp batch entry without its folder, level, sequence or runs"
    # Folders are recorded relative to the sweep folder; never read outside it.
    if Path(folder).is_absolute() or ".." in Path(folder).parts:
        return f"a batch folder outside the sweep: {folder}"
    # An unfinished batch has no `server` key; a finished one has its readings as an object.
    if "server" in entry:
        server = entry["server"]
        if not isinstance(server, dict):
            return f"batch {entry['sequence']}: server is not an object"
        if not isinstance(server.get("changes"), (dict, type(None))):
            return f"batch {entry['sequence']}: server changes are not an object"
    exclusion = entry.get("metrics_exclusion")
    if exclusion:
        causes = exclusion.get("causes") if isinstance(exclusion, dict) else None
        if not isinstance(causes, list) or not all(
            isinstance(cause, dict) and isinstance(cause.get("state"), str) for cause in causes
        ):
            return f"batch {entry['sequence']}: metrics_exclusion has no list of causes"
    return None


def _read_record(sweep_dir: Path) -> dict:
    """`sweep.json`, with the fields the analysis reads checked at this boundary."""
    path = sweep_dir / SWEEP_FILE
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SweepFolderError(f"cannot read {path}: {error}") from error
    if not isinstance(record, dict) or record.get("method") != SWEEP_METHOD:
        raise SweepFolderError(f"{path} is not a {SWEEP_METHOD} record")
    protocol = record.get("protocol")
    configuration = protocol.get("configuration") if isinstance(protocol, dict) else None
    timeout = protocol.get("run_timeout_seconds") if isinstance(protocol, dict) else None
    max_num_seqs = configuration.get("max_num_seqs") if isinstance(configuration, dict) else None
    if not (_is_number(timeout) and math.isfinite(timeout) and timeout > 0):
        raise SweepFolderError(f"{path} declares no usable run_timeout_seconds")
    if not isinstance(max_num_seqs, int) or isinstance(max_num_seqs, bool):
        raise SweepFolderError(f"{path} declares no configuration max_num_seqs")
    batches, hard_stops = record.get("batches"), record.get("hard_stops")
    if not isinstance(batches, list) or not all(isinstance(entry, dict) for entry in batches):
        raise SweepFolderError(f"{path} has no list of batch entries")
    if not isinstance(hard_stops, list) or not all(
        isinstance(event, dict)
        and _is_int(event.get("level"))
        and isinstance(event.get("code"), str)
        and isinstance(event.get("phase"), str)
        for event in hard_stops
    ):
        raise SweepFolderError(f"{path} has no list of hard-stop events with levels and codes")
    phases = record.get("phases")
    warmup = phases.get("warmup") if isinstance(phases, dict) else None
    if not isinstance(warmup, dict) or not isinstance(warmup.get("status"), str):
        raise SweepFolderError(f"{path} records no warm-up phase status")
    for entry in batches:
        if entry.get("phase") != "ramp":
            continue
        problem = _ramp_entry_problem(entry)
        if problem is not None:
            raise SweepFolderError(f"{path} has {problem}")
    return record


def analyse_sweep_folder(sweep_dir: Path) -> dict:
    """Rebuild the ramp analysis from `sweep_dir` and write it to `analysis.json` beside it."""
    record = _read_record(sweep_dir)
    protocol = record["protocol"]
    run_timeout_seconds = float(protocol["run_timeout_seconds"])
    analysis = ramp_analysis(
        record["batches"],
        record["hard_stops"],
        run_timeout_seconds,
        lambda entry: batch_metrics_or_error(
            sweep_dir / entry["folder"], run_timeout_seconds, confine_to_batch=True
        ),
    )
    labels = ramp_label_results(
        sweep_dir,
        record["batches"],
        protocol["configuration"]["max_num_seqs"],
        confine_to_batch=True,
    )
    ramp_end = record.get("ramp_end")
    document = {
        "method": OFFLINE_METHOD,
        "analysed_at": datetime.now(UTC).isoformat(),
        "source": SWEEP_FILE,
        "sweep_id": record.get("sweep_id"),
        "sweep_state": record.get("state"),
        "ramp_end": ramp_end,
        "protocol_sha256": record.get("protocol_sha256"),
        # False when sweep.json's protocol was edited after it was recorded.
        "protocol_sha256_matches": protocol_sha256(protocol) == record.get("protocol_sha256"),
        # The driver's own mapping: `none` for a sweep that never ran a batch.
        "analysis_scope": analysis_scope(record),
        **analysis,
        **labels,
    }
    write_json_atomic(sweep_dir / ANALYSIS_FILE, document)
    return document
