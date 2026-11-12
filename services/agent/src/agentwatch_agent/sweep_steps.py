"""Per-step measurements of a frozen capacity sweep, recorded with no judgement.

`capacity-sweep-steps DIR` reads `DIR/sweep.json` and every scored run folder under `DIR` (never
the path a runner recorded) and writes one record per journaled step to `steps.jsonl`, with a
summary in `steps.json`. Each record carries what was measured: the step's runner-tick durations,
its own measured vLLM queue and service time, its parent step, where it ran in the sweep, and the
N = 1 baseline of its `(plan, step)`. It carries no threshold and no truth: a later step in
`pipelines/data` judges these records, so the judgement can change without re-reading the sweep.

A reading that was not measured is null, never zero. `span_status` says why a vLLM reading is
absent (`unavailable`, `malformed`, `incomplete`, or `not_recorded` for a step with no model
call). Durations are runner ticks and vLLM span durations are their own measurements; they are
never aligned (see `sweep_labels`).

Only sweeps with the held-out protocol hashes below are accepted. Phases read: `ramp` (the
batches `split_ramp_entries` keeps), `live`, `injected` and `control`; `warmup` is skipped. Live
runs have no replay plan and are counted in `runs_without_plan`. Each record carries its run's
folder relative to the sweep folder, so a consumer finds the run at
`<sweep root>/<summary sweep_folder>/<run_folder>`.
"""

import hashlib
import json
import os
from pathlib import Path

from agentwatch_agent.coverage import read_journal
from agentwatch_agent.runner import JOURNAL_FILE
from agentwatch_agent.sweep import SWEEP_FILE
from agentwatch_agent.sweep_analysis import _resolve_run_dir, split_ramp_entries
from agentwatch_agent.sweep_labels import (
    BASELINE_LEVEL,
    Baselines,
    baselines,
    ramp_run_dirs,
    read_run,
)
from agentwatch_agent.sweep_offline import SweepFolderError, _is_int, _read_record
from agentwatch_agent.sweep_protocol import PROTOCOL_VERSION, protocol_sha256

STEPS_FORMAT = "sweep-steps-v1"
STEPS_FILE = "steps.jsonl"
SUMMARY_FILE = "steps.json"
EVIDENCE_FILE = "agent-evidence.json"

# The held-out sweeps' declared protocols, by configuration name (A06, 2026-11-11).
ACCEPTED_PROTOCOL_SHA256: dict[str, str] = {
    "cfg-default": "22e26dc7536fe5e88e3d9ea9f01445a754788ce587b1626d6a4eb055ec4524b9",
    "cfg-hardware": "1608deee2a3bcb5245b07d73e13e73dcd1e82702ef14aa683ecd94144e0f18cd",
}
SCORED_PHASES = ("ramp", "live", "injected", "control")
HELD_OUT_PARTS = ("data", "held-out")
SPAN_STATUSES = ("measured", "unavailable", "malformed", "incomplete", "not_recorded")
RECORD_KEYS = frozenset(
    {
        "run_id",
        "run_folder",
        "step_id",
        "step_type",
        "parent_step_id",
        "plan_sha256",
        "task_id",
        "configuration",
        "n",
        "phase",
        "batch_sequence",
        "batch_id",
        "state",
        "duration_seconds",
        "queue_seconds",
        "execution_seconds",
        "span_status",
        "vllm_queue_seconds",
        "vllm_service_seconds",
        "baseline",
    }
)


def _check_protocol(record: dict, path: Path) -> str:
    """The configuration name of an accepted held-out sweep; refuses any other."""
    protocol = record["protocol"]
    name = protocol["configuration"].get("name")
    if protocol.get("version") != PROTOCOL_VERSION:
        raise SweepFolderError(f"{path} is not a {PROTOCOL_VERSION} sweep")
    if record.get("held_out") is not True:
        raise SweepFolderError(f"{path} is not marked held_out")
    accepted = ACCEPTED_PROTOCOL_SHA256.get(name) if isinstance(name, str) else None
    if accepted is None:
        raise SweepFolderError(f"{path} has no accepted configuration (found {name!r})")
    # The recorded hash must be both the accepted one and the hash of the recorded protocol.
    if record.get("protocol_sha256") != accepted or protocol_sha256(protocol) != accepted:
        raise SweepFolderError(f"{path} does not carry the accepted {name} protocol hash")
    return name


def _scored_entry_problem(entry: dict) -> str | None:
    folder, runs = entry.get("folder"), entry.get("runs")
    if not (
        isinstance(folder, str)
        and _is_int(entry.get("level"))
        and _is_int(entry.get("sequence"))
        and isinstance(entry.get("batch_id"), str)
        and isinstance(runs, list)
        and all(isinstance(run, dict) for run in runs)
    ):
        return "a batch entry without its folder, level, sequence, batch_id or runs"
    if Path(folder).is_absolute() or ".." in Path(folder).parts:
        return f"a batch folder outside the sweep: {folder}"
    return None


def _scored_entries(record: dict, path: Path) -> tuple[list[dict], list[dict]]:
    """The batches whose runs are recorded in sequence order, and the excluded ramp batches."""
    batches = record["batches"]
    for entry in batches:
        if entry.get("phase") in SCORED_PHASES:
            problem = _scored_entry_problem(entry)
            if problem is not None:
                raise SweepFolderError(f"{path} has {problem}")
    usable_ramp, excluded = split_ramp_entries(batches)
    others = [entry for entry in batches if entry.get("phase") in SCORED_PHASES[1:]]
    return sorted(usable_ramp + others, key=lambda entry: entry["sequence"]), excluded


def _parent_step_ids(run_dir: Path) -> dict[str, str | None]:
    """Each step's parent from its latest journaled `step_event`.

    `step_timings` does not carry the parent and its output is recorded by replay experiments,
    so the journal is read here.
    """
    _registration, records, _unusable = read_journal(run_dir / JOURNAL_FILE)
    latest: dict[str, dict] = {}
    for record in records:
        if record.get("record_type") != "step_event":
            continue
        current = latest.get(record["step_id"])
        if current is None or record["sequence"] > current["sequence"]:
            latest[record["step_id"]] = record
    return {step_id: event.get("parent_step_id") for step_id, event in latest.items()}


def _check_run_identity(run_dir: Path, run: dict, run_folder: str, path: Path) -> None:
    """Refuse a run whose `sweep.json` id is not the id in its own folder's evidence."""
    evidence = json.loads((run_dir / EVIDENCE_FILE).read_text(encoding="utf-8"))
    recorded = evidence.get("run_id") if isinstance(evidence, dict) else None
    if not isinstance(recorded, str) or recorded != run.get("run_id"):
        raise SweepFolderError(
            f"{path} names run {run.get('run_id')!r} for {run_folder}, "
            f"whose evidence names {recorded!r}"
        )


def _baseline_of(reference: Baselines, plan: str, step_id: str) -> dict | None:
    found = reference.get((plan, step_id))
    if found is None:
        return None
    return {
        "count": found["count"],
        "mean_seconds": found["mean_seconds"],
        "stdev_seconds": found["stdev_seconds"],
    }


def _run_records(
    run_dir: Path,
    reference: Baselines,
    run: dict,
    entry: dict,
    run_folder: str,
    configuration: str,
) -> list[dict] | None:
    """The run's step records, or None when the run has no replay plan."""
    plan, timings, spans = read_run(run_dir)
    if plan is None:
        return None
    parents = _parent_step_ids(run_dir)
    records = []
    for step_id, timing in timings.items():
        span = spans.get(step_id, {"status": "not_recorded"})
        measured = span["status"] == "measured"
        records.append(
            {
                "run_id": run.get("run_id"),
                "run_folder": run_folder,
                "step_id": step_id,
                "step_type": timing["step_type"],
                "parent_step_id": parents.get(step_id),
                "plan_sha256": plan,
                "task_id": run.get("task_id"),
                "configuration": configuration,
                "n": entry["level"],
                "phase": entry["phase"],
                "batch_sequence": entry["sequence"],
                "batch_id": entry["batch_id"],
                "state": timing["state"],
                "duration_seconds": timing["total_seconds"],
                "queue_seconds": timing["queue_seconds"],
                "execution_seconds": timing["execution_seconds"],
                "span_status": span["status"],
                "vllm_queue_seconds": span["queue_seconds"] if measured else None,
                "vllm_service_seconds": span["service_seconds"] if measured else None,
                "baseline": _baseline_of(reference, plan, step_id),
            }
        )
    return records


def sweep_steps(sweep_dir: Path) -> tuple[dict, list[dict]]:
    """The summary and one record per journaled step of the sweep's scored runs.

    Raises `SweepFolderError` for a sweep that is not an accepted held-out sweep, has a malformed
    scored batch, has a run whose id disagrees with its evidence, or has a run whose files cannot
    be read: the records are never partial.
    """
    path = sweep_dir / SWEEP_FILE
    record = _read_record(sweep_dir)
    configuration = _check_protocol(record, path)
    entries, excluded = _scored_entries(record, path)
    ramp_runs, _ramp_missing = ramp_run_dirs(sweep_dir, record["batches"], confine_to_batch=True)
    records: list[dict] = []
    without_evidence = without_plan = 0
    run_folder = ""
    try:
        reference = baselines(ramp_runs.get(BASELINE_LEVEL, []))
        for entry in entries:
            batch_dir = sweep_dir / entry["folder"]
            for run in entry["runs"]:
                run_dir, _found_by = _resolve_run_dir(
                    batch_dir, run.get("output_dir"), confine_to_batch=True
                )
                if run_dir is None or not all(
                    (run_dir / name).is_file() for name in (EVIDENCE_FILE, JOURNAL_FILE)
                ):
                    without_evidence += 1
                    continue
                run_folder = run_dir.relative_to(sweep_dir).as_posix()
                _check_run_identity(run_dir, run, run_folder, path)
                run_records = _run_records(
                    run_dir, reference, run, entry, run_folder, configuration
                )
                if run_records is None:
                    without_plan += 1
                    continue
                records += run_records
    # CoverageError and json.JSONDecodeError are ValueErrors.
    except (OSError, ValueError, TypeError, KeyError) as error:
        where = f" in {run_folder}" if run_folder else ""
        raise SweepFolderError(f"cannot read the runs of {path}{where}: {error}") from error
    summary = {
        "format": STEPS_FORMAT,
        "sweep_folder": sweep_dir.resolve().name,
        "sweep_id": record.get("sweep_id"),
        "configuration": configuration,
        "protocol_version": record["protocol"]["version"],
        "protocol_sha256": record["protocol_sha256"],
        "sweep_json_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "held_out": True,
        "steps": len(records),
        "excluded_batches": excluded,
        "runs_without_evidence": without_evidence,
        "runs_without_plan": without_plan,
    }
    return summary, records


def _inside_held_out(output_dir: Path) -> bool:
    parts = output_dir.resolve().parts
    width = len(HELD_OUT_PARTS)
    return any(parts[i : i + width] == HELD_OUT_PARTS for i in range(len(parts) - width + 1))


def _write_lf(path: Path, text: str) -> None:
    """Write through a temporary file so a reader never sees a half-written result."""
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    os.replace(temporary, path)


def write_steps(sweep_dir: Path, output_dir: Path) -> Path:
    """Write `steps.jsonl` and `steps.json` into `output_dir`; refuses a held-out folder.

    The sweep is read completely first, so a refused sweep leaves nothing behind. The summary is
    written last: it marks the records as complete.
    """
    if _inside_held_out(output_dir):
        raise SweepFolderError(
            f"{output_dir} is inside a data/held-out folder; evidence is read only"
        )
    summary, records = sweep_steps(sweep_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, sort_keys=True, allow_nan=False) for record in records]
    _write_lf(output_dir / STEPS_FILE, "".join(line + "\n" for line in lines))
    _write_lf(
        output_dir / SUMMARY_FILE,
        json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n",
    )
    return output_dir
