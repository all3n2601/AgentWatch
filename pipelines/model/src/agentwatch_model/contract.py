"""The evaluation-contract values the rule baseline is scored against."""

import tomllib
from dataclasses import dataclass
from pathlib import Path

from agentwatch_data.extract import canonical_sha256

from agentwatch_model.metrics import is_number

# pipelines/model/evaluation-contract.toml beside this package's source tree. This default
# holds for the workspace (editable) install; other installs must pass the path explicitly.
DEFAULT_CONTRACT = Path(__file__).resolve().parents[2] / "evaluation-contract.toml"
HASH_DIGITS = 8  # the contract's documented report identifier


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
    min_gain_over_rules_overall: float = 0.0


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


def classes(parsed: dict) -> tuple[str, ...]:
    value = section(parsed, "dataset").get("classes")
    if not isinstance(value, list) or not value or not all(isinstance(c, str) for c in value):
        raise ContractError(
            "Evaluation contract [dataset] classes must be a non-empty list of names"
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
    )
