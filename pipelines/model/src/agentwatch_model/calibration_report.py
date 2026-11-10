"""The M04 report: calibrated confidence and abstention scored once on the test split.

The temperature and abstention threshold come from validation only (calibration.py). Gates
calibration cannot change, such as the regression error, are carried from the M03 candidates
report, so this report holds every contract gate for promotion checks. An abstention counts as
a miss everywhere the contract scores predictions.
"""

from dataclasses import dataclass

from agentwatch_model.calibration import (
    CalibrationConfig,
    abstaining_scores,
    abstention_rates,
    top_label_ece,
)
from agentwatch_model.comparison import gain_gate
from agentwatch_model.contract import Contract
from agentwatch_model.features import FEATURES
from agentwatch_model.metrics import class_scores, confusion_matrix, macro_f1, macro_f1_all_classes
from agentwatch_model.report import MIXED_CONTENTION_REASON, clean_gates, summary

METHOD = "temperature_scaling"
# Stands in for an abstained step's prediction in the clean gates, where it counts as a miss.
ABSTAINED = "abstained"
CARRIED_SOURCE = "M03 candidates report; calibration does not change this result"
EARLY_STOPPING_LIMITATION = (
    "The validation split also chose the classifier's early-stopping iteration, so a "
    "calibration fitted on it is slightly optimistic; the test-split ECE gate is the check."
)
COMPUTED_GATES = (
    "calibration",
    "abstention",
    "clean_none_rate",
    "baseline_and_clean_none_rate",
    "gain_over_rules",
)


@dataclass(frozen=True)
class Fit:
    """The calibration chosen on validation, or the reason none was fitted."""

    fitted: bool
    temperature: float | None
    threshold: float | None
    reason: str | None
    validation_steps: int
    usable_steps: int
    temperature_at_bound: bool
    selection: dict | None
    early_stopping_on_fit_split: bool


@dataclass(frozen=True)
class Scored:
    """Test-split rows with their M03 records, confidences and abstentions, in row order."""

    rows: list[dict]
    records: list[dict]
    raw_confidences: list[float]
    confidences: list[float]
    abstained: list[bool]
    probabilities: list[dict[str, float]]

    def correct(self) -> list[bool]:
        return [
            record["prediction"] == row["label_resource"]
            for row, record in zip(self.rows, self.records, strict=True)
        ]


def contract_identity(contract: Contract) -> dict:
    return {"version": contract.version, "status": contract.status, "hash": contract.hash}


def share(count: int, steps: int) -> float | None:
    return count / steps if steps else None


def group_summary(scored: Scored, indices: list[int], bins: int) -> dict:
    correct = scored.correct()
    return {
        "steps": len(indices),
        "accuracy_before_abstention": share(sum(correct[i] for i in indices), len(indices)),
        "abstention_rate": share(sum(scored.abstained[i] for i in indices), len(indices)),
        "ece_calibrated": top_label_ece(
            [scored.confidences[i] for i in indices], [correct[i] for i in indices], bins
        ),
    }


def slices(contract: Contract, scored: Scored) -> dict:
    """By condition, by whether any feature was missing, and mixed contention (unavailable)."""
    bins = contract.ece_bins
    indices = range(len(scored.rows))
    missing = {i for i in indices if any(scored.rows[i][name] is None for name in FEATURES)}
    conditions = sorted({row["condition"] for row in scored.rows})
    return {
        "by_condition": {
            condition: group_summary(
                scored, [i for i in indices if scored.rows[i]["condition"] == condition], bins
            )
            for condition in conditions
        },
        "missing_features": {
            "complete": group_summary(scored, [i for i in indices if i not in missing], bins),
            "missing_any": group_summary(scored, sorted(missing), bins),
            "steps_missing_each_feature": {
                name: sum(1 for row in scored.rows if row[name] is None) for name in FEATURES
            },
        },
        "mixed_contention": {
            "status": "unavailable",
            "reason": MIXED_CONTENTION_REASON,
            "min_steps": contract.min_mixed_steps,
        },
    }


def below_minimum_reason(below: list[str]) -> str:
    return f"Classes below the contract's minimum test steps: {', '.join(below)}"


def calibration_gate(ece: float | None, contract: Contract, fit: Fit, below: list[str]) -> dict:
    gate = {
        "value": ece,
        "threshold": contract.max_ece,
        "bins": contract.ece_bins,
        "calibrated": fit.fitted,
    }
    if not fit.fitted:
        return gate | {"outcome": "inconclusive", "reason": fit.reason}
    # ECE is measured over every test step, so a value over the limit is a failure whatever
    # the per-class support; only a pass needs every class to reach the minimum.
    if ece is not None and ece > contract.max_ece:
        return gate | {"outcome": "fail"}
    if below:
        return gate | {"outcome": "inconclusive", "reason": below_minimum_reason(below)}
    if ece is None:
        return gate | {"outcome": "inconclusive", "reason": "No test steps to measure ECE on"}
    return gate | {"outcome": "pass"}


def abstention_scopes(rates: dict, contract: Contract) -> dict:
    available = {
        "overall": rates["overall"],
        "per_true_class": rates["per_true_class"],
        "mixed_contention": {"status": "unavailable", "reason": MIXED_CONTENTION_REASON},
    }
    return {scope: available[scope] for scope in contract.abstention_rate_scopes}


def measured_rates(scopes: dict, min_steps: int) -> dict[str, float]:
    """Rates measured on enough steps to judge: overall, and classes at the contract minimum."""
    measured = {}
    if "overall" in scopes and scopes["overall"]["rate"] is not None:
        measured["overall"] = scopes["overall"]["rate"]
    for name, value in scopes.get("per_true_class", {}).items():
        if value["rate"] is not None and value["steps"] >= min_steps:
            measured[name] = value["rate"]
    return measured


def abstention_gate(rates: dict, contract: Contract, fit: Fit, below: list[str]) -> dict:
    """The contract's abstention limit, applied to each scope it lists."""
    scopes = abstention_scopes(rates, contract)
    limit = contract.max_abstention_rate
    gate = {"value": rates["overall"]["rate"], "threshold": limit, "scopes": scopes}
    if not fit.fitted:
        return gate | {"outcome": "inconclusive", "reason": fit.reason}
    # A measured rate over the limit fails the gate even while another class lacks steps.
    measured = measured_rates(scopes, contract.min_test_steps_per_class)
    exceeded = [name for name, value in measured.items() if value > limit]
    if exceeded:
        reason = f"Abstention above the limit for: {', '.join(exceeded)}"
        return gate | {"outcome": "fail", "reason": reason}
    if below:
        return gate | {"outcome": "inconclusive", "reason": below_minimum_reason(below)}
    if "mixed_contention" in scopes:
        reason = f"Mixed-contention abstention is unmeasured: {MIXED_CONTENTION_REASON}"
        return gate | {"outcome": "inconclusive", "reason": reason}
    return gate | {"outcome": "pass"}


def carried_gates(candidate_gates: dict) -> dict:
    return {
        name: gate | {"source": CARRIED_SOURCE} if name == "regression_error" else gate
        for name, gate in candidate_gates.items()
        if name not in COMPUTED_GATES
    }


def fit_summary(fit: Fit) -> dict:
    return {
        "split": "validation",
        "steps": fit.validation_steps,
        "usable_steps": fit.usable_steps,
        "early_stopping_on_fit_split": fit.early_stopping_on_fit_split,
        "limitation": EARLY_STOPPING_LIMITATION if fit.early_stopping_on_fit_split else None,
        "threshold_selection": fit.selection,
    }


def calibration_summary(fit: Fit, config: CalibrationConfig) -> dict:
    return {
        "version": config.version,
        "method": METHOD,
        "fitted": fit.fitted,
        "temperature": fit.temperature,
        "temperature_at_bound": fit.temperature_at_bound,
        "abstention_threshold": fit.threshold,
        "reason": fit.reason,
    }


def calibration_metrics(contract: Contract, scored: Scored) -> dict:
    correct, steps = scored.correct(), len(scored.rows)
    return {
        "bins": contract.ece_bins,
        "ece_uncalibrated": top_label_ece(scored.raw_confidences, correct, contract.ece_bins),
        "ece_calibrated": top_label_ece(scored.confidences, correct, contract.ece_bins),
        "accuracy_before_abstention": share(sum(correct), steps),
        "mean_confidence_calibrated": sum(scored.confidences) / steps if steps else None,
    }


def build_calibration_report(
    contract: Contract,
    candidates: dict,
    candidates_identity: dict,
    fit: Fit,
    scored: Scored,
    config: CalibrationConfig,
) -> dict:
    """Calibration metrics on the test split, every contract gate, and the summary."""
    labels = [row["label_resource"] for row in scored.rows]
    kept = [
        None if abstained else record["prediction"]
        for record, abstained in zip(scored.records, scored.abstained, strict=True)
    ]
    scores = abstaining_scores(labels, kept, contract.classes)
    minimum = contract.min_test_steps_per_class
    below = [name for name in contract.classes if scores[name]["support"] < minimum]
    rules = class_scores(confusion_matrix(labels, [r["rule_prediction"] for r in scored.records]))
    model_macro = macro_f1_all_classes(scores, contract.classes)
    rules_macro = macro_f1_all_classes(rules, contract.classes)
    rates = abstention_rates(labels, scored.abstained, contract.classes)
    metrics = calibration_metrics(contract, scored)
    absent = tuple((candidates.get("model") or {}).get("classes_absent_from_train", ()))
    gates = carried_gates(candidates["gates"]) | {
        "calibration": calibration_gate(metrics["ece_calibrated"], contract, fit, below),
        "abstention": abstention_gate(rates, contract, fit, below),
        **clean_gates(contract, scored.rows, [ABSTAINED if p is None else p for p in kept]),
        "gain_over_rules": gain_gate(model_macro, rules_macro, below, contract, absent),
    }
    return {
        "rules": candidates["rules"],
        "contract": contract_identity(contract),
        "status": "evaluated",
        "evaluation": True,
        "dataset_version": candidates.get("dataset_version"),
        "split": candidates["split"],
        "scored_split": "test",
        "scored_steps": len(scored.rows),
        "candidates": candidates_identity,
        "model": candidates.get("model"),
        "calibration": calibration_summary(fit, config),
        "fit": fit_summary(fit),
        "metrics": {
            "model": {
                "class_scores": scores,
                "macro_f1": {
                    "all_classes": model_macro,
                    "defined_classes": macro_f1(scores),
                    "classes_below_minimum": below,
                    "abstention_scoring": "incorrect",
                },
                "abstention": rates,
            },
            "rules": {"macro_f1": {"all_classes": rules_macro, "defined_classes": macro_f1(rules)}},
            "calibration": metrics,
            "slices": slices(contract, scored),
        },
        "gates": gates,
        "summary": summary(gates),
    }
