"""The evaluation-contract values the rule baseline is scored against."""

import tomllib
from dataclasses import dataclass
from pathlib import Path

from agentwatch_data.extract import canonical_sha256
from agentwatch_data.schema import RESOURCE_CLASSES

from agentwatch_model.metrics import is_number

# pipelines/model/evaluation-contract.toml beside this package's source tree. This default
# holds for the workspace (editable) install; other installs must pass the path explicitly.
DEFAULT_CONTRACT = Path(__file__).resolve().parents[2] / "evaluation-contract.toml"
HASH_DIGITS = 8  # the contract's documented report identifier
# Calibration is fitted on validation only; any other fit split would let test results steer it.
CALIBRATION_FIT_SPLIT = "validation"
ABSTENTION_SCOPES = ("overall", "per_true_class", "mixed_contention")


class ContractError(ValueError):
    """A contract missing a value the evaluation needs, or holding one of the wrong type."""


@dataclass(frozen=True)
class Contract:
    version: str
    status: str
    hash: str
    classes: tuple[str, ...]
    min_test_steps_per_class: int
    min_clean_steps: int
    min_mixed_steps: int
    min_none_rate_clean: float
    min_none_rate_baseline_and_clean: float
    max_mae: float
    min_gain_over_rules_overall: float
    calibration_fit_split: str
    ece_bins: int
    max_ece: float
    max_abstention_rate: float
    abstention_rate_scopes: tuple[str, ...]


def section(parsed: dict, name: str) -> dict:
    value = parsed.get(name)
    if not isinstance(value, dict):
        raise ContractError(f"Evaluation contract is missing the [{name}] table")
    return value


def count(parsed: dict, table: str, key: str) -> int:
    value = section(parsed, table).get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError(f"Evaluation contract [{table}] {key} must be a non-negative integer")
    return value


def share(parsed: dict, table: str, key: str) -> float:
    value = section(parsed, table).get(key)
    if not is_number(value) or not 0 <= value <= 1:
        raise ContractError(f"Evaluation contract [{table}] {key} must be a number from 0 to 1")
    return float(value)


def positive_count(parsed: dict, table: str, key: str) -> int:
    value = section(parsed, table).get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ContractError(f"Evaluation contract [{table}] {key} must be a positive integer")
    return value


def calibration_fit_split(parsed: dict) -> str:
    value = section(parsed, "calibration").get("fit_split")
    if value != CALIBRATION_FIT_SPLIT:
        raise ContractError(
            f"Evaluation contract [calibration] fit_split must be {CALIBRATION_FIT_SPLIT!r}"
        )
    return value


def abstention_scopes(parsed: dict) -> tuple[str, ...]:
    value = section(parsed, "calibration").get("abstention_rate_scopes")
    if (
        not isinstance(value, list)
        or not value
        or len(value) != len(set(value))
        or not set(value) <= set(ABSTENTION_SCOPES)
    ):
        raise ContractError(
            "Evaluation contract [calibration] abstention_rate_scopes must list distinct scopes "
            f"from {list(ABSTENTION_SCOPES)}"
        )
    return tuple(value)


def classes(parsed: dict) -> tuple[str, ...]:
    value = section(parsed, "dataset").get("classes")
    if not isinstance(value, list) or not value or not all(isinstance(c, str) for c in value):
        raise ContractError(
            "Evaluation contract [dataset] classes must be a non-empty list of names"
        )
    # Metrics are computed over the dataset's classes, so the two lists must name the same set.
    if len(value) != len(set(value)) or set(value) != set(RESOURCE_CLASSES):
        raise ContractError(
            f"Evaluation contract [dataset] classes {value} must name each of the dataset's "
            f"classes {list(RESOURCE_CLASSES)} once"
        )
    return tuple(value)


def load_contract(path: Path = DEFAULT_CONTRACT) -> Contract:
    parsed = tomllib.loads(path.read_text())
    for key in ("version", "status"):
        if not isinstance(parsed.get(key), str):
            raise ContractError(f"Evaluation contract is missing {key}")
    return Contract(
        version=parsed["version"],
        status=parsed["status"],
        hash=canonical_sha256(parsed)[:HASH_DIGITS],
        classes=classes(parsed),
        min_test_steps_per_class=count(parsed, "splits", "min_test_steps_per_class"),
        min_clean_steps=count(parsed, "splits", "min_clean_test_steps"),
        min_mixed_steps=count(parsed, "splits", "min_mixed_contention_test_steps"),
        min_none_rate_clean=share(parsed, "clean_runs", "min_none_rate_clean"),
        min_none_rate_baseline_and_clean=share(
            parsed, "clean_runs", "min_none_rate_baseline_and_clean"
        ),
        max_mae=share(parsed, "regression", "max_mae"),
        min_gain_over_rules_overall=share(parsed, "classification", "min_gain_over_rules_overall"),
        calibration_fit_split=calibration_fit_split(parsed),
        ece_bins=positive_count(parsed, "calibration", "ece_bins"),
        max_ece=share(parsed, "calibration", "max_ece"),
        max_abstention_rate=share(parsed, "calibration", "max_abstention_rate"),
        abstention_rate_scopes=abstention_scopes(parsed),
    )
