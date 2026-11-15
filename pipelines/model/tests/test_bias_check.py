"""bias-check on synthetic splits and hand-built rows; none of it is recorded evidence."""

import json

import pytest
from training_fixtures import dataset, featured_row, require_lightgbm

require_lightgbm()

import mlflow
from agentwatch_model.bias_check import (
    REPORT,
    BiasError,
    choose,
    correct,
    load_bias_config,
    main,
    run,
    slice_report,
    slice_values,
)
from agentwatch_model.candidates import (
    DEFAULT_TRAINING,
    fit_candidate,
    load_training_config,
    row_weights,
)

CLASSES = ("none", "cpu_sandbox", "inference", "retrieval")
SPLITS = {
    "train": dataset(25, "train"),
    "validation": dataset(8, "valid"),
    "test": dataset(8, "test"),
}


def mixed(index, basis="mixed_confirmed"):
    return featured_row("inference", index) | {
        "label_resource": "mixed",
        "label_resources": "cpu_sandbox,inference",
        "label_basis": basis,
    }


def test_slice_values_place_each_step_on_every_attribute():
    clean = featured_row("none", 0, cpu_pressure_max_avg10=None)
    assert slice_values(clean) == {
        "condition": "clean",
        "telemetry_readings": "missing_any",
        "contention": None,
    }
    full = featured_row("cpu_sandbox", 1, cpu_pressure_max_avg10=3.0, cgroup_throttled_usec_delta=0)
    assert slice_values(full)["telemetry_readings"] == "complete"
    assert slice_values(full)["contention"] == "single_class"
    assert slice_values(mixed(2))["contention"] == "mixed"


def test_correct_scores_mixed_steps_by_hit_and_never_joint_only_ones():
    assert correct(featured_row("inference", 0), "inference", CLASSES) is True
    assert correct(featured_row("inference", 0), "none", CLASSES) is False
    assert correct(mixed(1), "cpu_sandbox", CLASSES) is True
    assert correct(mixed(1), "retrieval", CLASSES) is False
    assert correct(mixed(1, basis="mixed_joint_only"), "inference", CLASSES) is None


def config(tmp_path, **changes):
    values = {
        "max_disparity": 0.10,
        "min_slice_steps": 3,
        "weight_by": '"condition"',
        "min_disparity_reduction": 0.02,
        "max_log_loss_increase": 0.05,
    } | changes
    path = tmp_path / "bias.toml"
    path.write_text(
        'version = "bias-v1"\n\n[thresholds]\n'
        f"max_disparity = {values['max_disparity']}\nmin_slice_steps = {values['min_slice_steps']}\n"
        f"\n[mitigation]\nweight_by = {values['weight_by']}\n"
        f"min_disparity_reduction = {values['min_disparity_reduction']}\n"
        f"max_log_loss_increase = {values['max_log_loss_increase']}\n"
    )
    return path


def test_slice_report_flags_a_slice_far_below_the_overall_rate(tmp_path):
    settings = load_bias_config(config(tmp_path))
    rows = [featured_row("none", i) for i in range(6)] + [
        featured_row("cpu_sandbox", i) for i in range(4)
    ]
    predictions = ["none"] * 6 + ["cpu_sandbox", "none", "none", "none"]
    report = slice_report(rows, predictions, CLASSES, settings)
    assert report["overall"] == {"steps": 10, "correct_rate": 0.7}
    contended = report["attributes"]["condition"]["contended"]
    assert contended["correct_rate"] == 0.25
    assert contended["disparity"] == pytest.approx(0.45)
    assert contended["outcome"] == "flagged"
    assert report["attributes"]["condition"]["clean"]["outcome"] == "pass"
    assert report["outcome"] == "fail"
    assert report["worst_disparity"] == pytest.approx(0.45)
    # Here the contended steps are also the single-class contention slice.
    assert report["flags"] == [
        {"attribute": "condition", "value": "contended", "disparity": 0.45},
        {"attribute": "contention", "value": "single_class", "disparity": 0.45},
    ]


def test_small_slices_are_insufficient_not_flagged(tmp_path):
    settings = load_bias_config(config(tmp_path, min_slice_steps=5))
    rows = [featured_row("none", i) for i in range(6)] + [featured_row("cpu_sandbox", 0)]
    report = slice_report(rows, ["none"] * 7, CLASSES, settings)
    assert report["attributes"]["condition"]["contended"]["outcome"] == "insufficient"
    assert report["outcome"] == "pass" and report["flags"] == []


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"max_disparity": 1.5}, "max_disparity"),
        ({"min_slice_steps": 0}, "min_slice_steps"),
        ({"weight_by": '"task_id"'}, "weight_by"),
        ({"max_log_loss_increase": -0.1}, "max_log_loss_increase"),
    ],
)
def test_bad_bias_configurations_are_refused(tmp_path, changes, message):
    with pytest.raises(BiasError, match=message):
        load_bias_config(config(tmp_path, **changes))


def test_the_packaged_configuration_is_declared():
    settings = load_bias_config()
    assert settings.version.startswith("bias-v1+")
    assert (settings.max_disparity, settings.min_slice_steps) == (0.10, 30)


def outcome(worst, flagged=True):
    return {"worst_disparity": worst, "outcome": "fail" if flagged else "pass"}


def test_choose_adopts_mitigation_only_within_the_declared_trade_off(tmp_path):
    settings = load_bias_config(config(tmp_path))
    adopted = choose(outcome(0.30), outcome(0.20), 0.10, 0.104, settings)
    assert adopted["selected"] == "weighted" and adopted["adopted"] is True
    too_costly = choose(outcome(0.30), outcome(0.20), 0.10, 0.11, settings)
    assert too_costly["selected"] == "unweighted" and "log loss" in too_costly["reason"]
    too_small = choose(outcome(0.30), outcome(0.29), 0.10, 0.10, settings)
    assert too_small["selected"] == "unweighted" and "disparity" in too_small["reason"]
    unneeded = choose(outcome(0.05, flagged=False), outcome(0.0), 0.10, 0.10, settings)
    assert unneeded["selected"] == "unweighted" and "no slice" in unneeded["reason"]


def test_choose_keeps_the_unweighted_model_without_mitigation_evidence(tmp_path):
    settings = load_bias_config(config(tmp_path))
    no_slices = choose(
        outcome(0.30), {"worst_disparity": None, "outcome": "pass"}, 0.1, 0.1, settings
    )
    assert no_slices["selected"] == "unweighted" and "insufficient" in no_slices["reason"]
    no_loss = choose(outcome(0.30), outcome(0.10), None, 0.1, settings)
    assert no_loss["selected"] == "unweighted" and "log loss" in no_loss["reason"]
    zero_loss = choose(outcome(0.30), outcome(0.10), 0.0, 0.1, settings)
    assert zero_loss["selected"] == "unweighted"


def test_row_weights_give_each_condition_the_same_total_weight():
    rows = [featured_row("none", i) for i in range(6)] + [featured_row("inference", 0)] * 2
    weights = row_weights(rows, "condition")
    clean = sum(w for w, r in zip(weights, rows, strict=True) if r["condition"] == "clean")
    contended = sum(w for w, r in zip(weights, rows, strict=True) if r["condition"] != "clean")
    assert clean == pytest.approx(contended) == pytest.approx(len(rows) / 2)
    assert row_weights(rows, None) is None


def test_weighting_changes_the_model_and_unweighted_training_is_unchanged(tmp_path):
    unweighted = DEFAULT_TRAINING.with_name("training-v2.toml")
    base = load_training_config(unweighted)
    weighted_path = tmp_path / "training.toml"
    weighted_path.write_text(
        unweighted.read_text().replace(
            "num_boost_round =", 'weight_by = "condition"\nnum_boost_round =', 1
        )
    )
    weighted = load_training_config(weighted_path)
    assert weighted.weight_by == "condition" and weighted.version != base.version
    train = dataset(25, "train")[:80]
    plain = fit_candidate(train, SPLITS["validation"], base)
    again = fit_candidate(train, SPLITS["validation"], base)
    assert plain.classifier.model_to_string() == again.classifier.model_to_string()
    reweighted = fit_candidate(train, SPLITS["validation"], weighted)
    assert reweighted.classifier.model_to_string() != plain.classifier.model_to_string()


def loader(path, name, contract_path, frozen_schema, allow_unchecked_leakage=False):
    return SPLITS[name]


def test_run_reports_both_models_on_validation_and_test_with_figures(tmp_path):
    output = tmp_path / "out"
    tracking = {"tracking_uri": f"sqlite:///{tmp_path / 'm.db'}", "artifact_root": tmp_path / "a"}
    outcome_ = run(
        tmp_path / "split.json", output, bias_path=config(tmp_path), loader=loader, **tracking
    )
    assert outcome_.run_id
    report = json.loads((output / REPORT).read_text())
    assert set(report["models"]) == {"unweighted", "weighted"}
    for model in report["models"].values():
        assert set(model) >= {"validation", "test", "validation_log_loss", "training_version"}
        assert model["validation"]["overall"]["steps"] > 0
    assert report["decision"]["selected"] in {"unweighted", "weighted"}
    assert report["decision"]["decided_on"] == "validation"
    versions = {name: model["training_version"] for name, model in report["models"].items()}
    assert versions["unweighted"].startswith("training-v2+")
    assert versions["weighted"].startswith("training-v3+")
    proposed = load_training_config(output / "training-weighted.toml")
    assert proposed.weight_by == "condition" and proposed.version == versions["weighted"]
    assert proposed.version == load_training_config().version  # the packaged training-v3
    for name in ("bias-validation.png", "bias-test.png"):
        assert (output / name).read_bytes().startswith(b"\x89PNG")
    mlflow.set_tracking_uri(tracking["tracking_uri"])
    [logged] = mlflow.search_runs(search_all_experiments=True, output_format="list")
    assert logged.info.run_id == outcome_.run_id


def test_main_exits_2_on_a_bad_bias_file(tmp_path, capsys):
    bad = tmp_path / "bias.toml"
    bad.write_text('version = "bias-v9"\n')
    assert main(["--split", str(tmp_path / "s.json"), "--bias", str(bad)]) == 2
    assert "Bias check error" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("unweighted", "weighted", "base_loss", "weighted_loss", "adopted"),
    [
        (0.30, 0.28, 0.10, 0.10, True),  # reduction exactly the 0.02 margin
        (0.30, 0.20, 0.10, 0.105, True),  # log loss up exactly 5%
        (0.30, 0.2801, 0.10, 0.10, False),  # just short of the margin
        (0.30, 0.20, 0.10, 0.1051, False),  # just past the trade-off
    ],
)
def test_choose_counts_values_on_a_declared_boundary_as_on_it(
    tmp_path, unweighted, weighted, base_loss, weighted_loss, adopted
):
    settings = load_bias_config(config(tmp_path))
    decision = choose(outcome(unweighted), outcome(weighted), base_loss, weighted_loss, settings)
    assert decision["adopted"] is adopted


def test_a_disparity_exactly_at_the_threshold_is_not_flagged(tmp_path):
    settings = load_bias_config(config(tmp_path, max_disparity=0.25))
    rows = [featured_row("none", i) for i in range(4)] + [
        featured_row("cpu_sandbox", i) for i in range(4)
    ]
    predictions = ["none"] * 4 + ["cpu_sandbox", "cpu_sandbox", "none", "none"]
    report = slice_report(rows, predictions, CLASSES, settings)
    contended = report["attributes"]["condition"]["contended"]
    assert contended["disparity"] == pytest.approx(0.25) and contended["outcome"] == "pass"


def test_a_training_file_the_weighting_cannot_be_added_to_is_refused(tmp_path):
    odd = tmp_path / "training.toml"
    text = DEFAULT_TRAINING.with_name("training-v2.toml").read_text()
    odd.write_text(text.replace("num_boost_round = 300", "num_boost_round=300"))
    with pytest.raises(BiasError, match="weight_by"):
        run(
            tmp_path / "split.json",
            tmp_path / "out",
            bias_path=config(tmp_path),
            training_path=odd,
            loader=loader,
            tracking_uri=f"sqlite:///{tmp_path / 'm.db'}",
            artifact_root=tmp_path / "a",
        )
