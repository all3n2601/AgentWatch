"""Scoring types and the rules-v1 scorer for held-out capacity-sweep rows.

Nothing here trains, calibrates or loads a model: the frozen M03/M04 classifier is scored in
`held_out_model`. Held-out rows carry no labels, so they are only scored.
"""

from dataclasses import dataclass

from agentwatch_data.replay_labels import task_group

from agentwatch_model.rules import Rules

RULES_SCORER = "rules-v1"


@dataclass(frozen=True)
class ScoredStep:
    scorer: str
    label: str
    confidence: float | None
    abstained: bool


def rule_scores(rules: Rules, rows: list[dict]) -> list[ScoredStep]:
    """rules-v1 labels; it has no confidence and never abstains."""
    return [ScoredStep(RULES_SCORER, rules.predict(row), None, False) for row in rows]


def seen_task_groups(train: list[dict]) -> frozenset[str]:
    """Task groups of the given train rows; validation groups are not training exposure."""
    return frozenset(task_group(row["task_id"]) for row in train)
