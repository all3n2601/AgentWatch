"""rules-v1: transparent threshold rules for the resource a step waited on.

The rules are the baseline the learned model must beat. Their thresholds live in rules.toml,
declared before any evaluation. Only resources with a measured signal in steps-v1.2 are
covered; the rules never predict the others.
"""

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

from agentwatch_data.extract import canonical_sha256

from agentwatch_model.metrics import NONE, is_number

DEFAULT_RULES = Path(__file__).with_name("rules.toml")
VERSION = "rules-v1"
HASH_DIGITS = 8  # matches label_version and the evaluation contract hash
CPU = "cpu_sandbox"
USEC_PER_SECOND = 1_000_000
# Upper bound of each threshold: fractions are shares, PSI avg10 is a percentage.
THRESHOLD_BOUNDS = {
    "min_queue_fraction": 1.0,
    "min_cpu_pressure_avg10": 100.0,
    "min_throttled_fraction": 1.0,
}


class RulesError(ValueError):
    """A rules file that cannot produce honest predictions."""


@dataclass(frozen=True)
class Rules:
    version: str
    min_queue_fraction: float
    min_cpu_pressure_avg10: float
    min_throttled_fraction: float
    covered_classes: tuple[str, ...] = (NONE, CPU)

    def predict(self, row: dict) -> str:
        return CPU if self.cpu_signal(row) else NONE

    def cpu_signal(self, row: dict) -> bool:
        queue = row.get("queue_fraction")
        if is_number(queue) and queue >= self.min_queue_fraction:
            return True
        pressure = row.get("cpu_pressure_max_avg10")
        if (
            row.get("cpu_pressure_available") is True
            and is_number(pressure)
            and pressure >= self.min_cpu_pressure_avg10
        ):
            return True
        throttled = throttled_fraction(row)
        return throttled is not None and throttled >= self.min_throttled_fraction


def throttled_fraction(row: dict) -> float | None:
    """cgroup throttled time over step duration, or None when either was not measured."""
    delta = row.get("cgroup_throttled_usec_delta")
    duration = row.get("duration_seconds")
    if row.get("cgroup_available") is not True or not is_number(delta) or not is_number(duration):
        return None
    if duration <= 0:
        return None
    return delta / (duration * USEC_PER_SECOND)


def threshold(section: dict, name: str) -> float:
    value = section.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RulesError(f"cpu_sandbox.{name} must be a number")
    if not math.isfinite(value):
        raise RulesError(f"cpu_sandbox.{name} must be finite")
    upper = THRESHOLD_BOUNDS[name]
    if not 0 <= value <= upper:
        raise RulesError(f"cpu_sandbox.{name} must be between 0 and {upper:g}")
    return float(value)


def load_rules(path: Path = DEFAULT_RULES) -> Rules:
    """Rules from a TOML file, identified by version plus a hash of the parsed contents."""
    parsed = tomllib.loads(path.read_text())
    if parsed.get("version") != VERSION:
        raise RulesError(f"{path.name}: version must be {VERSION!r}")
    section = parsed.get("cpu_sandbox")
    if not isinstance(section, dict):
        raise RulesError(f"{path.name}: missing [cpu_sandbox]")
    # Validate before hashing: canonical JSON refuses NaN, and the error should name the value.
    queue = threshold(section, "min_queue_fraction")
    pressure = threshold(section, "min_cpu_pressure_avg10")
    throttled = threshold(section, "min_throttled_fraction")
    digest = canonical_sha256(parsed)[:HASH_DIGITS]
    return Rules(
        version=f"{VERSION}+{digest}",
        min_queue_fraction=queue,
        min_cpu_pressure_avg10=pressure,
        min_throttled_fraction=throttled,
    )
