"""Experiment-derived step labels and the excess-latency regression target.

Labels come from the experiment design (which resource was contended) and a frozen
noise threshold, never from the rule baseline, so model and rule can be compared fairly.
"""

import hashlib
import statistics
import tomllib
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from agentwatch_data.schema import RESOURCE_CLASSES

BASELINE, CLEAN, CONTENDED = "baseline", "clean", "contended"


class LabelingError(ValueError):
    """Evidence that cannot be labeled under the current rules."""


@dataclass(frozen=True)
class LabelConfig:
    version: str
    stdev_multiplier: float
    min_seconds: float
    contention: dict[str, str]
    sha256: str

    @property
    def label_version(self) -> str:
        return f"{self.version}+{self.sha256[:8]}"


def load_config(path: Path | None = None) -> LabelConfig:
    raw = (
        path.read_bytes()
        if path is not None
        else files("agentwatch_data").joinpath("labels.toml").read_bytes()
    )
    data = tomllib.loads(raw.decode())
    config = LabelConfig(
        version=data["version"],
        stdev_multiplier=float(data["noise"]["stdev_multiplier"]),
        min_seconds=float(data["noise"]["min_seconds"]),
        contention=dict(data["contention"]),
        sha256=hashlib.sha256(raw).hexdigest(),
    )
    if config.stdev_multiplier <= 0 or config.min_seconds < 0:
        raise LabelingError("Noise threshold settings must be positive")
    contended_classes = set(RESOURCE_CLASSES) - {"none"}
    unknown = set(config.contention.values()) - contended_classes
    if unknown:
        raise LabelingError(f"Contention resources must be real resource classes: {unknown}")
    return config


def excess_fraction(duration_seconds: float, baseline_mean_seconds: float) -> float:
    """max(0, min(1, (T - B) / T)): the share of the step beyond the clean baseline."""
    if duration_seconds <= 0:
        raise LabelingError("Step duration must be positive")
    return max(0.0, min(1.0, (duration_seconds - baseline_mean_seconds) / duration_seconds))


def label_run(rows: list[dict], config: LabelConfig) -> list[dict]:
    """Label every step of one run against that run's own clean baseline."""
    if len({row["run_id"] for row in rows}) != 1:
        raise LabelingError("label_run expects the steps of exactly one run")
    kind = rows[0]["kind"]
    baseline = [row["duration_seconds"] for row in rows if row["condition"] == BASELINE]
    if len(baseline) < 2:
        raise LabelingError(f"Run {rows[0]['run_id']} needs at least two baseline steps")
    unknown = {row["condition"] for row in rows} - {BASELINE, CLEAN, CONTENDED}
    if unknown:
        raise LabelingError(f"Labels are not defined for conditions {sorted(unknown)}")

    baseline_mean = statistics.fmean(baseline)
    threshold = max(config.min_seconds, config.stdev_multiplier * statistics.stdev(baseline))
    labeled = []
    for row in rows:
        exceeded = row["duration_seconds"] - baseline_mean > threshold
        resource, basis = "none", "baseline_reference"
        if row["condition"] == CLEAN:
            basis = "clean_exceeded_noise" if exceeded else "clean_within_noise"
        elif row["condition"] == CONTENDED:
            if kind not in config.contention:
                raise LabelingError(f"No contended resource is declared for kind {kind!r}")
            resource = config.contention[kind] if exceeded else "none"
            basis = "contention_exceeded_threshold" if exceeded else "contention_below_threshold"
        labeled.append(
            row
            | {
                "label_resource": resource,
                "label_excess_fraction": excess_fraction(row["duration_seconds"], baseline_mean),
                "label_basis": basis,
                "baseline_mean_seconds": baseline_mean,
                "noise_threshold_seconds": threshold,
                "label_version": config.label_version,
            }
        )
    return labeled


def label_rows(rows: list[dict], config: LabelConfig) -> list[dict]:
    """Label rows from any number of runs, keeping their original order."""
    positions: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        positions.setdefault(row["run_id"], []).append(index)
    labeled: list[dict] = [{} for _ in rows]
    for indexes in positions.values():
        for index, row in zip(indexes, label_run([rows[i] for i in indexes], config)):
            labeled[index] = row
    return labeled
