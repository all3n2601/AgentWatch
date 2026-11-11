"""Analyse a recorded capacity sweep again from its folder, without vLLM, a GPU or the live API.

`sweep-analyse DIR` reads `DIR/sweep.json` and the batch folders it names, recomputes every ramp
batch's metrics from disk, then the level summaries, the capacity decision (excluding the
batches the driver marked, as the sweep itself does) and label validation, and writes
`DIR/analysis.json`. `sweep.json` is never modified, so the recorded and the offline analysis
can be compared. Use it when a sweep failed before or during its analysis, or to re-check one.
"""

import json
import math
from datetime import UTC, datetime
from pathlib import Path

from agentwatch_agent.runner import write_json_atomic
from agentwatch_agent.sweep import (
    FULL_RAMP,
    PARTIAL_RAMP,
    RAMP_CEILING,
    RAMP_HARD_STOP,
    SWEEP_FILE,
    SWEEP_METHOD,
)
from agentwatch_agent.sweep_analysis import _is_number, batch_metrics_or_error, ramp_analysis
from agentwatch_agent.sweep_labels import ramp_label_results
from agentwatch_agent.sweep_protocol import protocol_sha256

ANALYSIS_FILE = "analysis.json"
OFFLINE_METHOD = "capacity-sweep-offline-analysis-v1"


class SweepFolderError(ValueError):
    """The folder holds no `sweep.json` that can be analysed."""


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
        isinstance(event, dict) and isinstance(event.get("level"), int) for event in hard_stops
    ):
        raise SweepFolderError(f"{path} has no list of hard-stop events with levels")
    for entry in batches:
        if entry.get("phase") != "ramp":
            continue
        folder, level, sequence = entry.get("folder"), entry.get("level"), entry.get("sequence")
        runs = entry.get("runs")
        if not (
            isinstance(folder, str)
            and isinstance(level, int)
            and isinstance(sequence, int)
            and isinstance(runs, list)
            and all(isinstance(run, dict) for run in runs)
        ):
            raise SweepFolderError(f"{path} has a ramp batch entry without its folder or level")
        # Folders are recorded relative to the sweep folder; never read outside it.
        if Path(folder).is_absolute() or ".." in Path(folder).parts:
            raise SweepFolderError(f"{path} names a batch folder outside the sweep: {folder}")
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
        lambda entry: batch_metrics_or_error(sweep_dir / entry["folder"], run_timeout_seconds),
    )
    labels = ramp_label_results(
        sweep_dir, record["batches"], protocol["configuration"]["max_num_seqs"]
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
        "analysis_scope": FULL_RAMP if ramp_end in (RAMP_CEILING, RAMP_HARD_STOP) else PARTIAL_RAMP,
        **analysis,
        **labels,
    }
    write_json_atomic(sweep_dir / ANALYSIS_FILE, document)
    return document
