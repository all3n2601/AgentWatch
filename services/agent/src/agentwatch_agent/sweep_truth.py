"""Per-step truths for a frozen capacity sweep: queue, slower and protocol-clean.

`capacity-sweep-truth DIR` reads `DIR/sweep.json` and every scored run folder under `DIR` (never
the path a runner recorded) and writes one record per journaled step to `truth.jsonl`, with a
summary in `truth.json`. The truths are what a later scoring step compares predictions with:

- `queue`: the step's own measured vLLM queue time exceeds its noise threshold.
- `slower`: the step's duration exceeds its N = 1 baseline mean by more than that threshold.
  The N = 1 ramp runs form the baseline, so `slower` is circular there: it is null with reason
  `not_applicable` for them (N = 1 `control` and `injected` runs keep it).
- `protocol_clean`: the batch ran one agent with nothing injected (`ramp` or `control` at N = 1),
  so the step should read as no overload at all.

`queue` and `slower` map true to `inference` and false to `none`. A step that cannot be judged
has a null truth and a reason (`span_not_measured`, `no_baseline`, `not_completed`,
`malformed`, `not_applicable`), never a measured zero. Durations are runner ticks and vLLM span
durations are their own measurements; they are never aligned (see `sweep_labels`).

Only sweeps with the held-out protocol hashes below are accepted. Phases read: `ramp` (the
batches `split_ramp_entries` keeps), `live`, `injected` and `control`; `warmup` is skipped. Each
record carries its run's folder relative to the sweep folder, so a consumer finds the run at
`<sweep root>/<summary sweep_folder>/<run_folder>`.
"""

import hashlib
import json
import os
from pathlib import Path

from agentwatch_agent.runner import JOURNAL_FILE
from agentwatch_agent.sweep import SWEEP_FILE
from agentwatch_agent.sweep_analysis import _resolve_run_dir, split_ramp_entries
from agentwatch_agent.sweep_labels import (
    BASELINE_LEVEL,
    StepTruth,
    baselines,
    ramp_run_dirs,
    step_truths,
)
from agentwatch_agent.sweep_offline import SweepFolderError, _is_int, _read_record
from agentwatch_agent.sweep_protocol import PROTOCOL_VERSION, protocol_sha256

TRUTH_METHOD = "sweep-truth-v1"
TRUTH_FILE = "truth.jsonl"
SUMMARY_FILE = "truth.json"
EVIDENCE_FILE = "agent-evidence.json"

# The held-out sweeps' declared protocols, by configuration name (A06, 2026-11-11).
ACCEPTED_PROTOCOL_SHA256: dict[str, str] = {
    "cfg-default": "22e26dc7536fe5e88e3d9ea9f01445a754788ce587b1626d6a4eb055ec4524b9",
    "cfg-hardware": "1608deee2a3bcb5245b07d73e13e73dcd1e82702ef14aa683ecd94144e0f18cd",
}
SCORED_PHASES = ("ramp", "live", "injected", "control")
PROTOCOL_CLEAN_PHASES = ("ramp", "control")
PROTOCOL_CLEAN_LEVEL = 1
TRUTH_NAMES = ("queue", "slower", "protocol_clean")
NOT_APPLICABLE = "not_applicable"
HELD_OUT_PARTS = ("data", "held-out")


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
    """The batches whose runs are scored in sequence order, and the excluded ramp batches."""
    batches = record["batches"]
    for entry in batches:
        if entry.get("phase") in SCORED_PHASES:
            problem = _scored_entry_problem(entry)
            if problem is not None:
                raise SweepFolderError(f"{path} has {problem}")
    usable_ramp, excluded = split_ramp_entries(batches)
    others = [entry for entry in batches if entry.get("phase") in SCORED_PHASES[1:]]
    return sorted(usable_ramp + others, key=lambda entry: entry["sequence"]), excluded


def _truth_class(value: bool | None) -> str | None:
    return None if value is None else ("inference" if value else "none")


def _truths_of(step: StepTruth, entry: dict) -> tuple[dict, dict]:
    """The three truths and, for each one that is null, why."""
    clean = entry["phase"] in PROTOCOL_CLEAN_PHASES and entry["level"] == PROTOCOL_CLEAN_LEVEL
    is_baseline = entry["phase"] == "ramp" and entry["level"] == BASELINE_LEVEL
    truths = {
        "queue": _truth_class(step.queued),
        "slower": None if is_baseline else _truth_class(step.slower),
        "protocol_clean": "none" if clean else None,
    }
    reasons = {
        "queue": step.reason if step.queued is None else None,
        "slower": (
            NOT_APPLICABLE if is_baseline else (step.reason if step.slower is None else None)
        ),
        "protocol_clean": None if clean else NOT_APPLICABLE,
    }
    return truths, reasons


def _record(
    step: StepTruth, plan: str, run: dict, entry: dict, run_folder: str, configuration: str
) -> dict:
    truths, reasons = _truths_of(step, entry)
    return {
        "run_id": run.get("run_id"),
        "run_folder": run_folder,
        "step_id": step.step_id,
        "step_type": step.step_type,
        "plan_sha256": plan,
        "task_id": run.get("task_id"),
        "configuration": configuration,
        "n": entry["level"],
        "phase": entry["phase"],
        "batch_sequence": entry["sequence"],
        "batch_id": entry["batch_id"],
        "state": step.state,
        "duration_seconds": step.duration_seconds,
        "queue_seconds": step.queue_seconds,
        "execution_seconds": step.execution_seconds,
        "baseline_mean_seconds": step.baseline_mean_seconds,
        "noise_threshold_seconds": step.noise_threshold_seconds,
        "excess_seconds": step.excess_seconds,
        "vllm_queue_seconds": step.vllm_queue_seconds,
        "truths": truths,
        "truth_reasons": reasons,
    }


def _tally(records: list[dict]) -> dict:
    counts: dict[str, dict[str, dict[str, int]]] = {
        name: {"classes": {}, "reasons": {}} for name in TRUTH_NAMES
    }
    for record in records:
        for name in TRUTH_NAMES:
            for group, value in (
                ("classes", record["truths"][name]),
                ("reasons", record["truth_reasons"][name]),
            ):
                if value is not None:
                    counts[name][group][value] = counts[name][group].get(value, 0) + 1
    return counts


def sweep_truths(sweep_dir: Path) -> tuple[dict, list[dict]]:
    """The summary and one truth record per journaled step of the sweep's scored runs.

    Raises `SweepFolderError` for a sweep that is not an accepted held-out sweep, has a malformed
    scored batch, or has a run whose files cannot be read: the truths are never partial.
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
                plan, steps = step_truths(run_dir, reference)
                if plan is None:
                    without_plan += 1
                    continue
                records += [
                    _record(step, plan, run, entry, run_folder, configuration) for step in steps
                ]
    # CoverageError and json.JSONDecodeError are ValueErrors.
    except (OSError, ValueError, TypeError, KeyError) as error:
        where = f" in {run_folder}" if run_folder else ""
        raise SweepFolderError(f"cannot read the runs of {path}{where}: {error}") from error
    summary = {
        "method": TRUTH_METHOD,
        "sweep_folder": sweep_dir.name,
        "sweep_id": record.get("sweep_id"),
        "configuration": configuration,
        "protocol_version": record["protocol"]["version"],
        "protocol_sha256": record["protocol_sha256"],
        "sweep_json_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "held_out": True,
        "steps": len(records),
        "counts": _tally(records),
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


def write_truths(sweep_dir: Path, output_dir: Path) -> Path:
    """Write `truth.jsonl` and `truth.json` into `output_dir`; refuses a held-out folder.

    The sweep is read completely first, so a refused sweep leaves nothing behind. The summary is
    written last: it marks the records as complete.
    """
    if _inside_held_out(output_dir):
        raise SweepFolderError(
            f"{output_dir} is inside a data/held-out folder; evidence is read only"
        )
    summary, records = sweep_truths(sweep_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, sort_keys=True, allow_nan=False) for record in records]
    _write_lf(output_dir / TRUTH_FILE, "".join(line + "\n" for line in lines))
    _write_lf(
        output_dir / SUMMARY_FILE,
        json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n",
    )
    return output_dir
