"""mixed-scoring-v1: how predictions are scored on steps labeled `mixed`.

Rohit's M02 decision for eval-contract-v2 item 2, fixed before any evaluation:

- a step labeled `mixed` was slowed by two or more confirmed resources (`label_resources`); a
  four-class prediction on it is correct when it names any of them;
- steps whose basis is `mixed_joint_only` slowed only when the contentions ran together, so no
  single resource is confirmed; they are reported separately and never scored;
- per-resource F1 on `label_resources` is reported beside macro F1.

Every score here uses the recorded labels as fixed truth, never the prediction, so the model
and the rules are measured against the same answer on the same rows:

- the contract's four-class metrics (confusion matrix, per-class scores, macro F1 and its
  support minimums) cover single-class steps only, as a mixed step has no single class;
- mixed steps are scored by their hit rate under the any-member rule;
- per-resource scores are one-vs-rest on `label_resources`: a step is a positive for every
  confirmed resource, so a single prediction on a mixed step leaves its other members as misses
  and per-resource recall on mixed steps stays below 1;
- the mixed-contention macro F1 is the mean per-resource F1 on scored mixed steps, over the
  resources confirmed on at least one of them.
"""

from agentwatch_data.schema import MIXED
from agentwatch_data.splits import is_member

from agentwatch_model.contract import Contract
from agentwatch_model.metrics import NONE, f1_score

VERSION = "mixed-scoring-v1"
JOINT_ONLY = "mixed_joint_only"
MIXED_BASES = ("mixed_confirmed", "mixed_partial", JOINT_ONLY)
PER_RESOURCE_SEMANTICS = (
    "One-vs-rest on label_resources: a step is a positive for every confirmed resource, so "
    "recall on mixed steps stays below 1 for a single prediction."
)


def members(row: dict, classes: tuple[str, ...]) -> tuple[str, ...]:
    """A mixed step's confirmed resources, in contract class order."""
    named = set(row["label_resources"].split(","))
    return tuple(name for name in classes if name in named)


def is_mixed(row: dict) -> bool:
    return row["label_resource"] == MIXED


def is_joint_only(row: dict) -> bool:
    return is_mixed(row) and row["label_basis"] == JOINT_ONLY


def split_joint_only(
    rows: list[dict], predictions: list[str]
) -> tuple[list[dict], list[str], list[dict], list[str]]:
    """(scored rows, their predictions, joint-only rows, their predictions)."""
    pairs = list(zip(rows, predictions, strict=True))
    kept = [(r, p) for r, p in pairs if not is_joint_only(r)]
    joint = [(r, p) for r, p in pairs if is_joint_only(r)]
    return (
        [r for r, _ in kept],
        [p for _, p in kept],
        [r for r, _ in joint],
        [p for _, p in joint],
    )


def scored_rows(rows: list[dict], predictions: list[str]) -> tuple[list[dict], list[str]]:
    """Every step except the joint-only ones, which are never scored."""
    kept, kept_predictions, _, _ = split_joint_only(rows, predictions)
    return kept, kept_predictions


def single_class(rows: list[dict], predictions: list[str]) -> tuple[list[dict], list[str]]:
    """The steps the contract's four-class metrics cover: those with one recorded class."""
    pairs = [(r, p) for r, p in zip(rows, predictions, strict=True) if not is_mixed(r)]
    return [r for r, _ in pairs], [p for _, p in pairs]


def hit_rate(rows: list[dict], predictions: list[str], classes: tuple[str, ...]) -> float | None:
    """Share of mixed steps whose prediction names a confirmed resource."""
    hits = sum(1 for r, p in zip(rows, predictions, strict=True) if p in members(r, classes))
    return hits / len(rows) if rows else None


def per_resource_scores(
    rows: list[dict], predictions: list[str], classes: tuple[str, ...]
) -> dict[str, dict]:
    """One-vs-rest scores per class on label_resources (see PER_RESOURCE_SEMANTICS)."""
    scores = {}
    for name in classes:
        positive = [is_member(row, name) for row in rows]
        named = [prediction == name for prediction in predictions]
        hits = sum(p and n for p, n in zip(positive, named, strict=True))
        support, predicted = sum(positive), sum(named)
        precision = hits / predicted if predicted else None
        recall = hits / support if support else None
        scores[name] = {
            "support": support,
            "predicted": predicted,
            "precision": precision,
            "recall": recall,
            "f1": f1_score(hits, precision, recall),
        }
    return scores


def mixed_macro_f1(
    rows: list[dict], predictions: list[str], classes: tuple[str, ...]
) -> tuple[float | None, tuple[str, ...]]:
    """Mean per-resource F1 on mixed steps over the resources confirmed on them, and those
    resources. The set depends on the labels only, so two predictors share it."""
    confirmed = tuple(
        name for name in classes if name != NONE and any(is_member(r, name) for r in rows)
    )
    if not confirmed:
        return None, confirmed
    scores = per_resource_scores(rows, predictions, classes)
    return sum(scores[name]["f1"] for name in confirmed) / len(confirmed), confirmed


def mixed_summary(rows: list[dict], predictions: list[str], contract: Contract) -> dict:
    """The mixed-contention slice: scored mixed steps and, apart, the joint-only ones."""
    kept, kept_predictions, joint, joint_predictions = split_joint_only(rows, predictions)
    pairs = [(r, p) for r, p in zip(kept, kept_predictions, strict=True) if is_mixed(r)]
    mixed_rows, mixed_predictions = [r for r, _ in pairs], [p for _, p in pairs]
    macro, confirmed = mixed_macro_f1(mixed_rows, mixed_predictions, contract.classes)
    return {
        "status": "reported" if len(pairs) >= contract.min_mixed_steps else "inconclusive",
        "scoring": VERSION,
        "steps": len(pairs),
        "min_steps": contract.min_mixed_steps,
        "hit_rate": hit_rate(mixed_rows, mixed_predictions, contract.classes),
        "macro_f1": macro,
        "macro_f1_resources": list(confirmed),
        "per_resource": per_resource_scores(mixed_rows, mixed_predictions, contract.classes),
        "joint_only": {
            "steps": len(joint),
            "named_a_member": sum(
                1
                for r, p in zip(joint, joint_predictions, strict=True)
                if p in members(r, contract.classes)
            ),
            "scored": False,
        },
    }
