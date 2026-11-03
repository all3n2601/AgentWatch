"""Labels for replay experiments, where each injected condition contends one resource.

Rules live in labels-v2.toml, frozen before any evaluation. Conditions come from the
experiment design and are used for labeling only; condition data is never a feature.
"""

import tomllib
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from agentwatch_data.extract import canonical_sha256
from agentwatch_data.labels import LabelingError, require_keys, require_number, require_text
from agentwatch_data.schema import RESOURCE_CLASSES

# --- Frozen rules -----------------------------------------------------------------

REPLAY_CONDITIONS = ("cpu", "inference", "retrieval")


@dataclass(frozen=True)
class ReplayLabelConfig:
    version: str
    stdev_multiplier: float
    min_seconds: float
    condition_resources: dict[str, str]
    fingerprint: str

    @property
    def label_version(self) -> str:
        return f"{self.version}+{self.fingerprint[:8]}"


def logic_source() -> str:
    """This module's source with normalized line endings, so checkouts agree."""
    return Path(__file__).read_text(encoding="utf-8").replace("\r\n", "\n")


def fingerprint(rules: dict) -> str:
    """Identify the validated rules and the labeling code that applies them."""
    return canonical_sha256({"rules": rules, "logic": logic_source()})


def load_replay_config(path: Path | None = None) -> ReplayLabelConfig:
    source = path if path is not None else files("agentwatch_data").joinpath("labels-v2.toml")
    try:
        raw = source.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise LabelingError(f"Replay label rules {source} cannot be read: {error}") from error
    try:
        rules = tomllib.loads(raw)
    except tomllib.TOMLDecodeError as error:
        raise LabelingError(f"Replay label rules are not valid TOML: {error}") from error
    require_keys(rules, "replay label rules", {"version", "noise", "condition_resources"})
    noise = require_keys(rules["noise"], "noise", {"stdev_multiplier", "min_seconds"})
    table = require_keys(
        rules["condition_resources"], "condition_resources", set(REPLAY_CONDITIONS)
    )
    contended_classes = set(RESOURCE_CLASSES) - {"none"}
    condition_resources = {}
    for condition in REPLAY_CONDITIONS:
        name = f"condition_resources.{condition}"
        resource = require_text(table[condition], name)
        if resource not in contended_classes:
            raise LabelingError(f"{name} must be one of {sorted(contended_classes)}")
        condition_resources[condition] = resource
    version = require_text(rules["version"], "version")
    multiplier = require_number(noise["stdev_multiplier"], "noise.stdev_multiplier", positive=True)
    floor = require_number(noise["min_seconds"], "noise.min_seconds", positive=False)
    # Fingerprint the validated values, so equivalent spellings such as 3 and 3.0 agree.
    validated = {
        "version": version,
        "noise": {"stdev_multiplier": multiplier, "min_seconds": floor},
        "condition_resources": condition_resources,
    }
    return ReplayLabelConfig(
        version, multiplier, floor, condition_resources, fingerprint(validated)
    )
