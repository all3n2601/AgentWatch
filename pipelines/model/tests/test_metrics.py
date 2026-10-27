import pytest
from agentwatch_model.metrics import (
    CLASSES,
    bootstrap_macro_f1,
    class_scores,
    condition_slices,
    confusion_matrix,
    constant_reference_mae,
    is_number,
    macro_f1,
    macro_f1_all_classes,
    none_rate,
)
from rule_fixtures import row


def test_confusion_matrix_counts_every_true_and_predicted_pair():
    labels = ["cpu_sandbox", "cpu_sandbox", "none", "none"]
    predictions = ["cpu_sandbox", "none", "none", "cpu_sandbox"]
    matrix = confusion_matrix(labels, predictions)
    assert matrix["cpu_sandbox"] == {"none": 1, "cpu_sandbox": 1, "inference": 0, "retrieval": 0}
    assert matrix["none"]["none"] == 1 and matrix["none"]["cpu_sandbox"] == 1
    assert matrix["inference"] == dict.fromkeys(CLASSES, 0)


def test_class_scores_leave_undefined_values_null_instead_of_zero():
    labels = ["cpu_sandbox", "cpu_sandbox", "none", "none", "retrieval"]
    predictions = ["cpu_sandbox", "none", "none", "cpu_sandbox", "none"]
    scores = class_scores(confusion_matrix(labels, predictions))
    assert scores["cpu_sandbox"] == pytest.approx(
        {"support": 2, "predicted": 2, "precision": 0.5, "recall": 0.5, "f1": 0.5}
    )
    # Never predicted, but present: recall and F1 are measured zeros, precision is undefined.
    assert scores["retrieval"] == {
        "support": 1,
        "predicted": 0,
        "precision": None,
        "recall": 0.0,
        "f1": 0.0,
    }
    # Neither present nor predicted: nothing is measured.
    assert scores["inference"] == {
        "support": 0,
        "predicted": 0,
        "precision": None,
        "recall": None,
        "f1": None,
    }


def test_a_class_absent_from_the_split_has_no_f1_even_when_predicted():
    scores = class_scores(confusion_matrix(["none", "none"], ["none", "inference"]))
    assert scores["inference"] == {
        "support": 0,
        "predicted": 1,
        "precision": 0.0,
        "recall": None,
        "f1": None,
    }
    # The false positive lowers none's recall instead of adding a zero to the average.
    assert macro_f1(scores) == pytest.approx(scores["none"]["f1"])


def test_macro_f1_averages_only_classes_with_a_defined_f1():
    labels = ["cpu_sandbox", "cpu_sandbox", "none", "none"]
    predictions = ["cpu_sandbox", "none", "none", "cpu_sandbox"]
    assert macro_f1(class_scores(confusion_matrix(labels, predictions))) == pytest.approx(0.5)
    assert macro_f1(class_scores(confusion_matrix([], []))) is None


def test_none_rate_is_the_share_of_selected_steps_predicted_none():
    rows = [row(condition="clean"), row(condition="clean"), row(condition="baseline")]
    predictions = ["none", "cpu_sandbox", "none"]
    assert none_rate(rows, predictions, {"clean"}) == {"steps": 2, "none": 1, "rate": 0.5}
    assert none_rate(rows, predictions, {"clean", "baseline"})["rate"] == pytest.approx(2 / 3)
    assert none_rate(rows, predictions, {"contended"}) == {"steps": 0, "none": 0, "rate": None}


def test_constant_references_use_the_training_mean_and_zero():
    train = [row(excess=0.0), row(excess=0.5)]
    test = [row(excess=0.0), row(excess=1.0)]
    assert constant_reference_mae(train, test) == pytest.approx(
        {"training_mean": 0.25, "mae_training_mean": 0.5, "mae_zero": 0.5}
    )
    assert constant_reference_mae([], test)["mae_training_mean"] is None
    assert constant_reference_mae(train, [])["mae_zero"] is None


def test_bootstrap_interval_resamples_whole_tasks_and_is_reproducible():
    rows = [row(task_id=f"t{i}", label="cpu_sandbox") for i in range(6)]
    rows += [row(task_id=f"t{i}") for i in range(6)]
    perfect = [r["label_resource"] for r in rows]
    assert bootstrap_macro_f1(rows, perfect, resamples=200, seed=7) == {
        "level": 0.95,
        "low": 1.0,
        "high": 1.0,
        "resamples": 200,
        "valid_resamples": 200,
        "seed": 7,
        "groups": 6,
        "min_groups": 2,
    }
    noisy = ["none"] * len(rows)
    first = bootstrap_macro_f1(rows, noisy, resamples=200, seed=7)
    assert first == bootstrap_macro_f1(rows, noisy, resamples=200, seed=7)
    assert first["low"] <= first["high"]


def test_bootstrap_needs_at_least_two_tasks():
    rows = [row(task_id="only"), row(task_id="only", label="cpu_sandbox")]
    interval = bootstrap_macro_f1(rows, ["none", "none"], resamples=50, seed=1)
    assert interval["low"] is None and interval["high"] is None and interval["groups"] == 1


def test_bootstrap_reports_no_interval_below_the_declared_minimum_of_tasks():
    rows = [row(task_id=f"t{i}") for i in range(4)]
    interval = bootstrap_macro_f1(rows, ["none"] * 4, resamples=50, seed=1, min_groups=5)
    assert interval["low"] is None and interval["valid_resamples"] == 0
    assert interval["groups"] == 4 and interval["min_groups"] == 5


def test_four_class_macro_f1_is_null_while_any_class_is_undefined():
    labels = ["cpu_sandbox", "none", "inference", "retrieval"]
    scores = class_scores(confusion_matrix(labels, labels))
    assert macro_f1_all_classes(scores, CLASSES) == 1.0
    partial = class_scores(confusion_matrix(["none", "cpu_sandbox"], ["none", "cpu_sandbox"]))
    assert macro_f1_all_classes(partial, CLASSES) is None
    assert macro_f1(partial) == 1.0


def test_condition_slices_count_each_condition_separately():
    rows = [row(condition="clean"), row("cpu_sandbox", "contended"), row(condition="clean")]
    slices = condition_slices(rows, ["none", "cpu_sandbox", "cpu_sandbox"])
    assert set(slices) == {"clean", "contended"}
    assert slices["clean"]["steps"] == 2
    assert slices["clean"]["confusion_matrix"]["none"] == {
        "none": 1,
        "cpu_sandbox": 1,
        "inference": 0,
        "retrieval": 0,
    }
    assert slices["contended"]["confusion_matrix"]["cpu_sandbox"]["cpu_sandbox"] == 1


def test_booleans_are_not_numbers():
    assert is_number(0.5) and is_number(3)
    assert not is_number(True) and not is_number(float("nan")) and not is_number(None)
