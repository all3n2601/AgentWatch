"""The sweep-truth-v1 label rule for held-out capacity-sweep steps.

Values live in sweep-truth-v1.toml, frozen before any evaluation. `judge` decides, for one
validated step record from `sweep_steps_input`, each truth as a class or as null with a
reason, never both. Missing readings stay null; they are never compared as zero.
"""

import math
import re
import tomllib
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from agentwatch_data.extract import canonical_sha256
from agentwatch_data.labels import LabelingError, require_keys, require_number, require_text

TRUTH_NAMES = ("queue", "slower", "protocol_clean")
CLASS_INFERENCE, CLASS_NONE = "inference", "none"
SHA256_HEX = re.compile(r"[0-9a-f]{64}")
PROTOCOL_CONFIGURATIONS = ("cfg-default", "cfg-hardware")
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
    source = path if path is not None else files("agentwatch_data").joinpath("sweep-truth-v1.toml")
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
        truths["protocol_clean"] = CLASS_NONE
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
