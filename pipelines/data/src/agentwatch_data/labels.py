"""Experiment-derived step labels and the excess-latency regression target.

Labels come from the experiment design (which resource was contended) and a frozen
noise threshold, never from the rule baseline, so model and rule can be compared fairly.
"""

import logging
import math
import statistics
import tomllib
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from agentwatch_data.extract import Rejection, canonical_sha256
from agentwatch_data.schema import RESOURCE_CLASSES

logger = logging.getLogger(__name__)

BASELINE, CLEAN, CONTENDED = "baseline", "clean", "contended"
NO_INJECTION = "none"
# A run whose contended step produced different output did different work, so its
# slowdown cannot be attributed to contention. Other failed checks are recorded in
# failed_checks but do not block labels: dropping runs because the rule disagreed would
# select the dataset by the rule's own answer.
DESIGN_INVALIDATING_CHECKS = ("same_workload_output",)


class LabelingError(ValueError):
    """Evidence or rules that cannot produce trustworthy labels."""


@dataclass(frozen=True)
class Contention:
    resource: str
    injection_method: str


@dataclass(frozen=True)
class LabelConfig:
    version: str
    stdev_multiplier: float
    min_seconds: float
    contention: dict[str, Contention]
    fingerprint: str

    @property
    def label_version(self) -> str:
        return f"{self.version}+{self.fingerprint[:8]}"


def logic_source() -> str:
    """This module's source with normalized line endings, so checkouts agree."""
    return Path(__file__).read_text(encoding="utf-8").replace("\r\n", "\n")


def fingerprint(rules: dict) -> str:
    """Identify the parsed rules and the labeling code that applies them.

    Comments, whitespace, and line endings in the rule file do not change it; any change
    to a rule value or to this module does.
    """
    return canonical_sha256({"rules": rules, "logic": logic_source()})


def require_keys(table, name: str, required: set[str]) -> dict:
    if not isinstance(table, dict):
        raise LabelingError(f"{name} must be a table")
    missing, unknown = required - table.keys(), table.keys() - required
    if missing or unknown:
        raise LabelingError(f"{name}: missing {sorted(missing)}, unknown {sorted(unknown)}")
    return table


def require_number(value, name: str, *, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise LabelingError(f"{name} must be a finite number, got {value!r}")
    if value < 0 or (positive and value == 0):
        bound = "positive" if positive else "nonnegative"
        raise LabelingError(f"{name} must be {bound}, got {value!r}")
    return float(value)


def require_text(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LabelingError(f"{name} must be a nonempty string, got {value!r}")
    return value


def load_config(path: Path | None = None) -> LabelConfig:
    source = path if path is not None else files("agentwatch_data").joinpath("labels.toml")
    try:
        raw = source.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise LabelingError(f"Label rules {source} cannot be read: {error}") from error
    try:
        rules = tomllib.loads(raw)
    except tomllib.TOMLDecodeError as error:
        raise LabelingError(f"Label rules are not valid TOML: {error}") from error
    require_keys(rules, "label rules", {"version", "noise", "contention"})
    noise = require_keys(rules["noise"], "noise", {"stdev_multiplier", "min_seconds"})
    if not isinstance(rules["contention"], dict) or not rules["contention"]:
        raise LabelingError("contention must declare at least one experiment kind")
    contended_classes = set(RESOURCE_CLASSES) - {"none"}
    contention = {}
    for kind, table in rules["contention"].items():
        name = f"contention.{kind}"
        require_keys(table, name, {"resource", "injection_method"})
        resource = require_text(table["resource"], f"{name}.resource")
        if resource not in contended_classes:
            raise LabelingError(f"{name}.resource must be one of {sorted(contended_classes)}")
        method = require_text(table["injection_method"], f"{name}.injection_method")
        if method == NO_INJECTION:
            raise LabelingError(f"{name}.injection_method cannot be {NO_INJECTION!r}")
        contention[kind] = Contention(resource, method)
    version = require_text(rules["version"], "version")
    multiplier = require_number(noise["stdev_multiplier"], "noise.stdev_multiplier", positive=True)
    floor = require_number(noise["min_seconds"], "noise.min_seconds", positive=False)
    # Fingerprint the validated values, so equivalent spellings such as 3 and 3.0 agree.
    validated = {
        "version": version,
        "noise": {"stdev_multiplier": multiplier, "min_seconds": floor},
        "contention": {kind: vars(c) for kind, c in sorted(contention.items())},
    }
    return LabelConfig(version, multiplier, floor, contention, fingerprint(validated))


def require_positive_finite(value: float, name: str) -> float:
    """Reject NaN, infinity, and booleans explicitly: comparisons with NaN are always False,
    so they slip past sign checks, and clamping would turn them into valid-looking targets."""
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise LabelingError(f"{name} must be a finite positive number, got {value!r}")
    return value


def excess_fraction(duration_seconds: float, baseline_mean_seconds: float) -> float:
    """max(0, min(1, (T - B) / T)): the share of the step beyond the clean baseline."""
    require_positive_finite(duration_seconds, "Step duration")
    require_positive_finite(baseline_mean_seconds, "Baseline mean duration")
    return max(0.0, min(1.0, (duration_seconds - baseline_mean_seconds) / duration_seconds))


def check_run(rows: list[dict]) -> None:
    """Reject evidence that would make one run's labels untrustworthy."""
    if len({row["run_id"] for row in rows}) != 1:
        raise LabelingError("label_run expects the steps of exactly one run")
    run_id = rows[0]["run_id"]
    if len({row["kind"] for row in rows}) != 1:
        raise LabelingError(f"Run {run_id} mixes experiment kinds")
    step_ids = [row["step_id"] for row in rows]
    if len(step_ids) != len(set(step_ids)):
        raise LabelingError(f"Run {run_id} repeats step IDs")
    for row in rows:
        require_positive_finite(row["duration_seconds"], f"{row['condition']} step duration")
    unknown = {row["condition"] for row in rows} - {BASELINE, CLEAN, CONTENDED}
    if unknown:
        raise LabelingError(f"Labels are not defined for conditions {sorted(unknown)}")
    if sum(row["condition"] == BASELINE for row in rows) < 2:
        raise LabelingError(f"Run {run_id} needs at least two baseline steps")
    failed = {name for row in rows for name in (row.get("failed_checks") or "").split(",") if name}
    invalidating = failed.intersection(DESIGN_INVALIDATING_CHECKS)
    if invalidating:
        raise LabelingError(
            f"Run {run_id} failed {sorted(invalidating)}; contention is not attributable"
        )


def label_run(rows: list[dict], config: LabelConfig) -> list[dict]:
    """Label every step of one run against that run's own clean baseline."""
    check_run(rows)
    kind = rows[0]["kind"]
    baseline = [row["duration_seconds"] for row in rows if row["condition"] == BASELINE]
    baseline_mean = statistics.fmean(baseline)
    threshold = max(config.min_seconds, config.stdev_multiplier * statistics.stdev(baseline))
    labeled = []
    for row in rows:
        exceeded = row["duration_seconds"] - baseline_mean > threshold
        resource, basis, method = "none", "baseline_reference", NO_INJECTION
        if row["condition"] == CLEAN:
            basis = "clean_exceeded_noise" if exceeded else "clean_within_noise"
        elif row["condition"] == CONTENDED:
            if kind not in config.contention:
                raise LabelingError(f"No contended resource is declared for kind {kind!r}")
            contention = config.contention[kind]
            method = contention.injection_method
            resource = contention.resource if exceeded else "none"
            basis = "contention_exceeded_threshold" if exceeded else "contention_below_threshold"
        labeled.append(
            row
            | {
                "label_resource": resource,
                "label_excess_fraction": excess_fraction(row["duration_seconds"], baseline_mean),
                "label_basis": basis,
                "injection_method": method,
                "baseline_mean_seconds": baseline_mean,
                "noise_threshold_seconds": threshold,
                "label_version": config.label_version,
            }
        )
    return labeled


def label_rows(rows: list[dict], config: LabelConfig) -> tuple[list[dict], list[Rejection]]:
    """Label rows from any number of runs, keeping the order of the runs that succeed.

    A run whose evidence cannot be labeled is rejected with its reason and the other runs
    continue, so one bad run cannot block every later build. Invalid rules are already
    rejected by load_config before any run is labeled.
    """
    positions: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        positions.setdefault(row["run_id"], []).append(index)
    labeled: dict[int, dict] = {}
    rejections = []
    for run_id, indexes in positions.items():
        try:
            run_rows = label_run([rows[i] for i in indexes], config)
        except LabelingError as error:
            logger.warning("Rejected run %s at labeling: %s", run_id, error)
            rejections.append(Rejection(f"run {run_id}", str(error)))
            continue
        labeled.update(zip(indexes, run_rows))
    return [labeled[index] for index in sorted(labeled)], rejections
