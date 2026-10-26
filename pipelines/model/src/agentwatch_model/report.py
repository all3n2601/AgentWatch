"""The rule-baseline report: metrics plus each evaluation-contract gate's outcome.

Every gate is pass, fail, inconclusive or not_evaluated. Gates the rules cannot be judged on
are listed as not_evaluated with the reason, so a reader never mistakes them for passes. Any
gate other than pass blocks promotion, as the contract requires.
"""

from collections import Counter

from agentwatch_model.contract import Contract
from agentwatch_model.metrics import (
    bootstrap_macro_f1,
    class_scores,
    condition_slices,
    confusion_matrix,
    constant_reference_mae,
    macro_f1,
    macro_f1_all_classes,
    none_rate,
)
from agentwatch_model.rules import Rules

# Proposed. Fewer tasks than this give an interval too unstable to report.
BOOTSTRAP_MIN_GROUPS = 5
MIXED_CONTENTION_REASON = "labels-v1.1 does not define mixed contention or ties."
NOT_EVALUATED_FOR_RULES = {
    "gain_over_rules": "The rules are the reference a learned model is compared against.",
    "mixed_contention_gain": MIXED_CONTENTION_REASON,
    "regression_error": (
        "rules-v1 names a resource, not an excess-latency fraction; the constant references "
        "are reported instead."
    ),
    "calibration": "rules-v1 gives no confidence score.",
    "abstention": "rules-v1 never abstains.",
    "unseen_tasks": "task_id is still an interim placeholder, not real task identity.",
    "held_out_injection_method": "Needs at least two injection methods in the dataset.",
    "instrumentation_overhead": "Measured on live runs, not on a dataset split.",
    "attribution_latency": "Measured on the live service, not on a dataset split.",
}
EXPLORATORY_REASON = (
    "Exploratory run on the validation split with the leakage check skipped; no gate is evaluated."
)


def none_rate_gate(rate: dict, threshold: float, clean_steps: int, minimum: int) -> dict:
    gate = {"value": rate["rate"], "threshold": threshold, "steps": rate["steps"]}
    if clean_steps < minimum or rate["rate"] is None:
        reason = f"{clean_steps} clean steps; the contract needs at least {max(minimum, 1)}"
        return gate | {"outcome": "inconclusive", "reason": reason}
    return gate | {"outcome": "pass" if rate["rate"] >= threshold else "fail"}


def clean_gates(contract: Contract, rows: list[dict], predictions: list[str]) -> dict:
    # Both gates require the contract's minimum of clean steps, which baseline steps
    # cannot substitute for.
    clean = none_rate(rows, predictions, {"clean"})
    both = none_rate(rows, predictions, {"baseline", "clean"})
    minimum = contract.min_clean_steps
    return {
        "clean_none_rate": none_rate_gate(
            clean, contract.min_none_rate_clean, clean["steps"], minimum
        ),
        "baseline_and_clean_none_rate": none_rate_gate(
            both, contract.min_none_rate_baseline_and_clean, clean["steps"], minimum
        ),
    }


def identity(contract: Contract, rules: Rules) -> dict:
    covered = list(rules.covered_classes)
    return {
        "rules": {
            "version": rules.version,
            "covered_classes": covered,
            "uncovered_classes": [c for c in contract.classes if c not in covered],
        },
        "contract": {"version": contract.version, "status": contract.status, "hash": contract.hash},
    }


def dataset_version(rows: list[dict]) -> str | None:
    versions = {row["dataset_version"] for row in rows}
    if len(versions) > 1:
        raise ValueError(f"Rows from more than one dataset version: {sorted(versions)}")
    return versions.pop() if versions else None


def macro_f1_block(
    contract: Contract,
    scores: dict,
    rows: list[dict],
    predictions: list[str],
    resamples: int,
    seed: int,
) -> dict:
    below = [
        c for c in contract.classes if scores[c]["support"] < contract.min_test_steps_per_class
    ]
    return {
        "all_classes": macro_f1_all_classes(scores, contract.classes),
        "defined_classes": macro_f1(scores),
        "status": "inconclusive" if below else "reported",
        "classes_below_minimum": below,
        "interval_of": "defined_classes",
        "interval": bootstrap_macro_f1(
            rows, predictions, resamples, seed, min_groups=BOOTSTRAP_MIN_GROUPS
        ),
    }


def slices(contract: Contract, rows: list[dict], predictions: list[str]) -> dict:
    return {
        "by_condition": condition_slices(rows, predictions),
        "mixed_contention": {
            "status": "unavailable",
            "reason": MIXED_CONTENTION_REASON,
            "min_steps": contract.min_mixed_steps,
        },
    }


def summary(gates: dict) -> dict:
    outcomes = Counter(gate["outcome"] for gate in gates.values())
    return {
        "outcomes": {
            name: outcomes[name] for name in ("pass", "fail", "inconclusive", "not_evaluated")
        },
        "blocks_promotion": any(gate["outcome"] != "pass" for gate in gates.values()),
    }


def build_report(
    contract: Contract,
    rules: Rules,
    train: list[dict],
    scored: list[dict],
    predictions: list[str],
    *,
    evaluation: bool,
    resamples: int,
    seed: int,
) -> dict:
    """Metrics for the rules' predictions on the scored split and each contract gate's outcome."""
    matrix = confusion_matrix([row["label_resource"] for row in scored], predictions)
    scores = class_scores(matrix)
    gates = clean_gates(contract, scored, predictions) | {
        name: {"outcome": "not_evaluated", "reason": why}
        for name, why in NOT_EVALUATED_FOR_RULES.items()
    }
    if not evaluation:
        gates = {name: {"outcome": "not_evaluated", "reason": EXPLORATORY_REASON} for name in gates}
    return identity(contract, rules) | {
        "status": "evaluated" if evaluation else "exploratory",
        "evaluation": evaluation,
        "dataset_version": dataset_version(scored + train),
        "metrics": {
            "confusion_matrix": matrix,
            "class_scores": scores,
            "macro_f1": macro_f1_block(contract, scores, scored, predictions, resamples, seed),
            "constant_references": constant_reference_mae(train, scored)
            | {"contract_max_mae": contract.max_mae},
            "slices": slices(contract, scored, predictions),
        },
        "gates": gates,
        "summary": summary(gates),
    }


def blocked_report(contract: Contract, rules: Rules, reason: str) -> dict:
    """A split the contract blocks: the reason, and no metrics that could be misread."""
    return identity(contract, rules) | {
        "status": "blocked",
        "evaluation": False,
        "reason": reason,
        "metrics": None,
        "gates": {},
        "summary": {"outcomes": {}, "blocks_promotion": True},
    }
