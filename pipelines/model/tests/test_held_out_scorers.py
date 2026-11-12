"""Held-out scorers on synthetic rows; nothing here is recorded evidence."""

from agentwatch_model.held_out_scorers import RULES_SCORER, rule_scores, seen_task_groups
from agentwatch_model.rules import load_rules
from training_fixtures import dataset, featured_row


def test_rules_v1_scores_never_abstain():
    rows = [featured_row("cpu_sandbox", 0), featured_row("none", 1)]
    scores = rule_scores(load_rules(), rows)
    assert [s.scorer for s in scores] == [RULES_SCORER] * 2 == ["rules-v1"] * 2
    assert [s.label for s in scores] == [load_rules().predict(r) for r in rows]
    assert all(s.confidence is None and s.abstained is False for s in scores)


def test_seen_task_groups_are_the_groups_of_the_rows_given():
    assert seen_task_groups([featured_row("none", 0, task_id="fix-mean-empty-r2")]) == {
        "fix-mean-empty"
    }
    train, validation = dataset(30, "train"), dataset(8, "valid")
    seen = seen_task_groups(train)
    assert seen == {f"train-{i}" for i in range(5)}
    assert not seen & seen_task_groups(validation)
