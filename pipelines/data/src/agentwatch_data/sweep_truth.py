"""The sweep-truth-v2 label rule for held-out capacity-sweep steps.

Values live in sweep-truth-v2.toml, fixed before any scoring with them. `judge` decides, for one
validated step record from `sweep_steps_input`, each truth as a class or as null with a
reason, never both. Missing readings stay null; they are never compared as zero.
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tomllib
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from agentwatch_data.extract import canonical_sha256
from agentwatch_data.labels import LabelingError, require_keys, require_number, require_text
from agentwatch_data.sweep_steps_input import SweepSteps, read_steps

TRUTH_NAMES = ("queue", "slower", "protocol_clean")
CLASS_INFERENCE, CLASS_NONE = "inference", "none"
SHA256_HEX = re.compile(r"[0-9a-f]{64}")
PROTOCOL_CONFIGURATIONS = ("cfg-default", "cfg-hardware")
TRUTH_FILE = "truth.jsonl"
TRUTH_SUMMARY_FILE = "truth.json"
HELD_OUT_PARTS = ("data", "held-out")
DEFAULT_OUTPUT_ROOT = Path(".data/a06-scoring/truth")
DROPPED_FIELDS = ("span_status", "vllm_service_seconds", "baseline")
# Summary fields copied from the recorder's steps.json; the rest of the summary is derived here.
CARRIED_SUMMARY_FIELDS = (
    "sweep_folder",
    "sweep_id",
    "configuration",
    "protocol_version",
    "protocol_sha256",
    "sweep_json_sha256",
    "recorder_sha256",
    "excluded_batches",
    "runs_without_evidence",
    "runs_without_plan",
)
LOGIC_MODULES = ("sweep_truth.py", "sweep_steps_input.py")


@dataclass(frozen=True)
class SweepTruthRule:
    version: str
    stdev_multiplier: float
    min_seconds: float
    min_baseline_runs: int
    scored_step_types: tuple[str, ...]
    clean_phases: tuple[str, ...]
    clean_level: int
    baseline_phase: str
    baseline_level: int
    protocols: dict[str, str]
    # The validated values the fingerprint covers, recorded in truth.json so a rule loaded with
    # --rules can be reconstructed from its output.
    values: dict
    fingerprint: str

    @property
    def label_version(self) -> str:
        return f"{self.version}+{self.fingerprint[:8]}"


def logic_source() -> dict[str, str]:
    """Fingerprint the truth decisions together with the input validation they rely on."""
    return {
        name: files("agentwatch_data")
        .joinpath(name)
        .read_text(encoding="utf-8")
        .replace("\r\n", "\n")
        for name in LOGIC_MODULES
    }


def _require_int(value, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise LabelingError(f"{name} must be an integer of at least {minimum}, got {value!r}")
    return value


def _require_texts(value, name: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise LabelingError(f"{name} must be a nonempty list of strings")
    return [require_text(item, f"{name} item") for item in value]


def _require_protocols(table) -> dict[str, str]:
    require_keys(table, "protocols", set(PROTOCOL_CONFIGURATIONS))
    protocols = {}
    for configuration in PROTOCOL_CONFIGURATIONS:
        digest = table[configuration]
        if not isinstance(digest, str) or not SHA256_HEX.fullmatch(digest):
            raise LabelingError(f"protocols.{configuration} must be 64 lowercase hex digits")
        protocols[configuration] = digest
    return protocols


def load_truth_rule(path: Path | None = None) -> SweepTruthRule:
    source = path if path is not None else files("agentwatch_data").joinpath("sweep-truth-v2.toml")
    try:
        raw = source.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise LabelingError(f"Sweep truth rule {source} cannot be read: {error}") from error
    try:
        rules = tomllib.loads(raw)
    except tomllib.TOMLDecodeError as error:
        raise LabelingError(f"Sweep truth rule is not valid TOML: {error}") from error
    require_keys(
        rules,
        "sweep truth rule",
        {"version", "noise", "baseline", "steps", "protocol_clean", "slower", "protocols"},
    )
    noise = require_keys(rules["noise"], "noise", {"stdev_multiplier", "min_seconds"})
    baseline = require_keys(rules["baseline"], "baseline", {"min_runs"})
    steps = require_keys(rules["steps"], "steps", {"scored_types"})
    clean = require_keys(rules["protocol_clean"], "protocol_clean", {"phases", "level"})
    slower = require_keys(rules["slower"], "slower", {"baseline_phase", "baseline_level"})
    multiplier = require_number(noise["stdev_multiplier"], "noise.stdev_multiplier", positive=True)
    floor = require_number(noise["min_seconds"], "noise.min_seconds", positive=False)
    min_runs = _require_int(baseline["min_runs"], "baseline.min_runs", minimum=1)
    scored_types = _require_texts(steps["scored_types"], "steps.scored_types")
    clean_phases = _require_texts(clean["phases"], "protocol_clean.phases")
    clean_level = _require_int(clean["level"], "protocol_clean.level", minimum=1)
    baseline_phase = require_text(slower["baseline_phase"], "slower.baseline_phase")
    baseline_level = _require_int(slower["baseline_level"], "slower.baseline_level", minimum=1)
    version = require_text(rules["version"], "version")
    protocols = _require_protocols(rules["protocols"])
    # Fingerprint the validated values, so equivalent spellings such as 3 and 3.0 agree.
    validated = {
        "version": version,
        "noise": {"stdev_multiplier": multiplier, "min_seconds": floor},
        "baseline": {"min_runs": min_runs},
        "steps": {"scored_types": scored_types},
        "protocol_clean": {"phases": clean_phases, "level": clean_level},
        "slower": {"baseline_phase": baseline_phase, "baseline_level": baseline_level},
        "protocols": protocols,
    }
    return SweepTruthRule(
        version=version,
        stdev_multiplier=multiplier,
        min_seconds=floor,
        min_baseline_runs=min_runs,
        scored_step_types=tuple(scored_types),
        clean_phases=tuple(clean_phases),
        clean_level=clean_level,
        baseline_phase=baseline_phase,
        baseline_level=baseline_level,
        protocols=protocols,
        values=validated,
        fingerprint=canonical_sha256({"rules": validated, "logic": logic_source()}),
    )


def noise_threshold(baseline: dict, rule: SweepTruthRule) -> float:
    """max(floor, multiplier x stdev); a baseline without a stdev uses the floor alone."""
    stdev = baseline["stdev_seconds"]
    if stdev is None:
        return rule.min_seconds
    return max(rule.min_seconds, rule.stdev_multiplier * stdev)


def _unjudged(reason: str) -> dict:
    return {
        "truths": dict.fromkeys(TRUTH_NAMES),
        "truth_reasons": dict.fromkeys(TRUTH_NAMES, reason),
        "noise_threshold_seconds": None,
        "excess_seconds": None,
    }


def _unusable_reason(record: dict, rule: SweepTruthRule) -> str | None:
    """Why the step has no usable duration or baseline, in the rule's order, or None."""
    duration = record["duration_seconds"]
    if record["state"] != "completed" or duration is None:
        return "not_completed"
    if not math.isfinite(duration) or duration < 0:
        return "malformed"
    baseline = record["baseline"]
    if baseline is None or baseline["count"] < rule.min_baseline_runs:
        return "no_baseline"
    return None


def _judge_model_call(record: dict, rule: SweepTruthRule) -> dict:
    truths: dict[str, str | None] = dict.fromkeys(TRUTH_NAMES)
    reasons: dict[str, str | None] = dict.fromkeys(TRUTH_NAMES)
    threshold = excess = None

    if record["phase"] in rule.clean_phases and record["n"] == rule.clean_level:
        # A call that never finished cannot be called clean: its duration is not a measurement.
        if record["state"] == "completed" and record["duration_seconds"] is not None:
            truths["protocol_clean"] = CLASS_NONE
        else:
            reasons["protocol_clean"] = "not_completed"
    else:
        reasons["protocol_clean"] = "not_applicable"
    # The baseline batches are not judged against their own mean.
    is_baseline_batch = (
        record["phase"] == rule.baseline_phase and record["n"] == rule.baseline_level
    )

    problem = _unusable_reason(record, rule)
    if problem is not None:
        reasons["queue"] = problem
        reasons["slower"] = "not_applicable" if is_baseline_batch else problem
    else:
        baseline = record["baseline"]
        threshold = noise_threshold(baseline, rule)
        excess = record["duration_seconds"] - baseline["mean_seconds"]
        queue = record["vllm_queue_seconds"]
        if record["span_status"] == "measured" and queue is not None:
            truths["queue"] = CLASS_INFERENCE if queue > threshold else CLASS_NONE
        else:
            reasons["queue"] = "span_not_measured"
        if is_baseline_batch:
            reasons["slower"] = "not_applicable"
        else:
            truths["slower"] = CLASS_INFERENCE if excess > threshold else CLASS_NONE
    return {
        "truths": truths,
        "truth_reasons": reasons,
        "noise_threshold_seconds": threshold,
        "excess_seconds": excess,
    }


def judge(record: dict, rule: SweepTruthRule) -> dict:
    """The truths, reasons, threshold and excess of one validated step record."""
    if record["parent_step_id"] is None:
        return _unjudged("root_step")
    if record["step_type"] not in rule.scored_step_types:
        return _unjudged("no_queue_reference")
    return _judge_model_call(record, rule)


def _truth_record(record: dict, rule: SweepTruthRule) -> dict:
    baseline = record["baseline"]
    result = {key: value for key, value in record.items() if key not in DROPPED_FIELDS}
    result["baseline_mean_seconds"] = None if baseline is None else baseline["mean_seconds"]
    result["baseline_stdev_seconds"] = None if baseline is None else baseline["stdev_seconds"]
    result["baseline_count"] = None if baseline is None else baseline["count"]
    result.update(judge(record, rule))
    return result


def _counts(records: list[dict]) -> dict:
    counts: dict = {name: {"classes": {}, "reasons": {}} for name in TRUTH_NAMES}
    for record in records:
        for name in TRUTH_NAMES:
            for group, value in (
                ("classes", record["truths"][name]),
                ("reasons", record["truth_reasons"][name]),
            ):
                if value is not None:
                    tally = counts[name][group]
                    tally[value] = tally.get(value, 0) + 1
    return counts


def sweep_truths(steps: SweepSteps, rule: SweepTruthRule) -> tuple[dict, list[dict]]:
    """The truth summary and one truth record per step, in the recorder's order.

    `write_truths` adds `truth_sha256`, the hash of the written `truth.jsonl`.
    """
    records = [_truth_record(record, rule) for record in steps.records]
    summary = {
        "label_version": rule.label_version,
        "rule": rule.values,
        **{field: steps.summary[field] for field in CARRIED_SUMMARY_FIELDS},
        "steps_sha256": steps.steps_sha256,
        "steps_summary_sha256": steps.summary_sha256,
        "held_out": True,
        "steps": len(records),
        "counts": _counts(records),
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


def write_truths(steps_dir: Path, output_dir: Path, rule: SweepTruthRule | None = None) -> Path:
    """Write `truth.jsonl` and `truth.json` into `output_dir`.

    Input, rule, output location and all JSON are checked or built before the folder exists, so
    a refusal leaves nothing behind. The summary is written last: it marks the records complete.
    """
    rule = rule if rule is not None else load_truth_rule()
    if _inside_held_out(output_dir):
        raise LabelingError(f"{output_dir} is inside a data/held-out folder; evidence is read only")
    steps = read_steps(steps_dir, rule.protocols)
    summary, records = sweep_truths(steps, rule)
    try:
        lines = "".join(
            json.dumps(record, sort_keys=True, allow_nan=False) + "\n" for record in records
        )
        # The records file is replaced before the summary; this hash lets a reader detect a
        # summary left beside records from another write.
        summary["truth_sha256"] = hashlib.sha256(lines.encode("utf-8")).hexdigest()
        summary_text = json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n"
    except ValueError as error:
        raise LabelingError(f"Truths cannot be serialised: {error}") from error
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_lf(output_dir / TRUTH_FILE, lines)
    _write_lf(output_dir / TRUTH_SUMMARY_FILE, summary_text)
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("steps_dir", type=Path, help="folder with steps.jsonl and steps.json")
    parser.add_argument("--output", type=Path, help="default: .data/a06-scoring/truth/<folder>")
    parser.add_argument("--rules", type=Path, help="rule TOML; default: the packaged rule")
    args = parser.parse_args(argv)
    output = args.output if args.output is not None else DEFAULT_OUTPUT_ROOT / args.steps_dir.name
    try:
        rule = load_truth_rule(args.rules)
        written = write_truths(args.steps_dir, output, rule)
    except (LabelingError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(written / TRUTH_FILE)
    print(written / TRUTH_SUMMARY_FILE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
