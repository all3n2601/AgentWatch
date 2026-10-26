"""The candidate report: model and rules scored on identical rows against the contract.

The model and rules-v1 predictions cover the same scored rows, so every comparison is like for
like. Gates are pass, fail, inconclusive or not_evaluated; calibration and abstention are left
to M04 rather than read from raw LightGBM scores.
"""

from collections.abc import Sequence

from agentwatch_model.contract import Contract
from agentwatch_model.metrics import (
    class_scores,
    confusion_matrix,
    constant_reference_mae,
    mean,
)
from agentwatch_model.report import (
    MIXED_CONTENTION_REASON,
    NOT_EVALUATED_FOR_RULES,
    clean_gates,
    dataset_version,
    identity,
    macro_f1_block,
    slices,
    summary,
)
from agentwatch_model.rules import Rules

MODEL_NOT_EVALUATED = {
    "mixed_contention_gain": MIXED_CONTENTION_REASON,
    "calibration": (
        "M04 calibrates confidence on the validation split; raw LightGBM scores are not "
        "calibrated confidence."
    ),
    "abstention": "M04 chooses the abstention threshold on the validation split.",
    "unseen_tasks": NOT_EVALUATED_FOR_RULES["unseen_tasks"],
    "held_out_injection_method": NOT_EVALUATED_FOR_RULES["held_out_injection_method"],
    "instrumentation_overhead": NOT_EVALUATED_FOR_RULES["instrumentation_overhead"],
    "attribution_latency": NOT_EVALUATED_FOR_RULES["attribution_latency"],
}
EXPLORATORY_REASON = (
    "Exploratory run on the validation split with the leakage check skipped; no gate is evaluated."
)


def gain_gate(
    model_macro: float | None,
    rules_macro: float | None,
    below: list[str],
    contract: Contract,
    absent: tuple[str, ...] = (),
) -> dict:
    """The contract's overall gain over rules, on the four-class macro F1 point estimates."""
    gate = {
        "model": model_macro,
        "rules": rules_macro,
        "min_gain": contract.min_gain_over_rules_overall,
    }
    if absent:
        # A class the model never saw scores zero F1: a data-coverage gap, not a model result.
        return gate | {
            "outcome": "inconclusive",
            "reason": f"Classes absent from the training split: {', '.join(absent)}",
        }
    if below:
        return gate | {
            "outcome": "inconclusive",
            "reason": f"Classes below the contract's minimum test steps: {', '.join(below)}",
        }
    if model_macro is None or rules_macro is None:
        return gate | {
            "outcome": "inconclusive",
            "reason": "The four-class macro F1 is undefined for the model or the rules.",
        }
    passed = model_macro - rules_macro >= contract.min_gain_over_rules_overall
    return gate | {"outcome": "pass" if passed else "fail"}


def regression_gate(
    mae: float | None, references: dict, contract: Contract, below: Sequence[str] = ()
) -> dict:
    """MAE within the contract limit and below both constant references."""
    constants = [references["mae_training_mean"], references["mae_zero"]]
    gate = {"value": mae, "threshold": contract.max_mae, "constant_references": constants}
    if below:
        return gate | {
            "outcome": "inconclusive",
            "reason": f"Classes below the contract's minimum test steps: {', '.join(below)}",
        }
    if mae is None or any(value is None for value in constants):
        return gate | {
            "outcome": "inconclusive",
            "reason": "The model MAE or a constant reference could not be measured.",
        }
    passed = mae <= contract.max_mae and all(mae < value for value in constants)
    return gate | {"outcome": "pass" if passed else "fail"}


def classification(
    contract: Contract,
    rows: list[dict],
    predictions: list[str],
    resamples: int,
    seed: int,
) -> dict:
    matrix = confusion_matrix([row["label_resource"] for row in rows], predictions)
    scores = class_scores(matrix)
    return {
        "confusion_matrix": matrix,
        "class_scores": scores,
        "macro_f1": macro_f1_block(contract, scores, rows, predictions, resamples, seed),
        "slices": slices(contract, rows, predictions),
    }


def build_model_report(
    contract: Contract,
    rules: Rules,
    model: dict,
    train: list[dict],
    scored: list[dict],
    model_labels: list[str],
    fractions: list[float],
    rule_labels: list[str],
    *,
    evaluation: bool,
    resamples: int,
    seed: int,
) -> dict:
    """Metrics for the model and the rules on the same rows, and each contract gate."""
    model_metrics = classification(contract, scored, model_labels, resamples, seed)
    rules_metrics = classification(contract, scored, rule_labels, resamples, seed)
    references = constant_reference_mae(train, scored)
    truth = [row["label_excess_fraction"] for row in scored]
    mae = mean([abs(f - y) for f, y in zip(fractions, truth, strict=True)])
    model_macro, rules_macro = model_metrics["macro_f1"], rules_metrics["macro_f1"]
    gates = clean_gates(contract, scored, model_labels) | {
        "gain_over_rules": gain_gate(
            model_macro["all_classes"],
            rules_macro["all_classes"],
            model_macro["classes_below_minimum"],
            contract,
            tuple(model.get("classes_absent_from_train", ())),
        ),
        "regression_error": regression_gate(
            mae, references, contract, model_macro["classes_below_minimum"]
        ),
    }
    gates |= {
        name: {"outcome": "not_evaluated", "reason": why}
        for name, why in MODEL_NOT_EVALUATED.items()
    }
    if not evaluation:
        gates = {name: {"outcome": "not_evaluated", "reason": EXPLORATORY_REASON} for name in gates}
    return identity(contract, rules) | {
        "status": "evaluated" if evaluation else "exploratory",
        "evaluation": evaluation,
        "dataset_version": dataset_version(scored + train),
        "model": model,
        "metrics": {
            "model": model_metrics,
            "rules": rules_metrics,
            "regression": {
                "mae": mae,
                "constant_references": references | {"contract_max_mae": contract.max_mae},
            },
        },
        "comparison": {
            "rows": len(scored),
            "macro_f1_all_classes": {
                "model": model_macro["all_classes"],
                "rules": rules_macro["all_classes"],
            },
            "macro_f1_defined_classes": {
                "model": model_macro["defined_classes"],
                "rules": rules_macro["defined_classes"],
            },
        },
        "gates": gates,
        "summary": summary(gates),
    }
