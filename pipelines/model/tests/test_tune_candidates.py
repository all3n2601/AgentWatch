"""tune-candidates on synthetic splits. Every row here is a made-up fixture, not evidence."""

import json
import tomllib

import pytest
from agentwatch_data.splits import SplitError
from training_fixtures import dataset, require_lightgbm

require_lightgbm()

import mlflow
from agentwatch_model.candidates import load_training_config
from agentwatch_model.tune_candidates import (
    EXIT_ERROR,
    PROPOSAL,
    REPORT,
    TuningError,
    grid,
    load_tuning_config,
    main,
    run,
    select,
    sensitivity,
)

SPLITS = {"train": dataset(25, "train"), "validation": dataset(6, "valid")}
BASE_SETTINGS = {
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_data_in_leaf": 5,
    "lambda_l2": 1.0,
    "feature_fraction": 1.0,
}


def write_tuning(tmp_path, search=None, margin=0.02):
    search = search or {
        "learning_rate": [0.05, 0.1],
        "num_leaves": [7, 15],
        "min_data_in_leaf": [5],
        "lambda_l2": [1.0],
        "feature_fraction": [1.0],
    }
    lines = ['version = "tuning-v1"', "", "[selection]", f"min_relative_improvement = {margin}"]
    lines += ["", "[search]"] + [f"{key} = {json.dumps(values)}" for key, values in search.items()]
    path = tmp_path / "tuning.toml"
    path.write_text("\n".join(lines) + "\n")
    return path


def loader_for(calls):
    def load(path, name, contract_path, frozen_schema, allow_unchecked_leakage=False):
        calls.append((name, allow_unchecked_leakage))
        return SPLITS[name]

    return load


def blocked(path, name, contract_path, frozen_schema, allow_unchecked_leakage=False):
    raise SplitError("Split x.json: its leakage check is blocked")


def tracking(tmp_path):
    return {
        "tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
        "artifact_root": tmp_path / "artifacts",
    }


def tune(tmp_path, **overrides):
    calls = overrides.pop("calls", [])
    return run(
        tmp_path / "split.json",
        tmp_path / "out",
        tuning_path=overrides.pop("tuning_path", write_tuning(tmp_path)),
        loader=overrides.pop("loader", loader_for(calls)),
        **tracking(tmp_path),
        **overrides,
    )


def read_report(tmp_path):
    return json.loads((tmp_path / "out" / REPORT).read_text())


def test_the_packaged_search_space_contains_training_v1_and_is_declared():
    config = load_tuning_config()
    assert config.version.startswith("tuning-v1+")
    assert config.min_relative_improvement == 0.02
    base = load_training_config(config.base_path)
    assert base.version.startswith("training-v1+")
    points = grid(config.search)
    assert len(points) == 162
    assert base.classifier in points and base.regressor in points


def test_grid_is_the_full_product_in_a_fixed_order():
    points = grid({"num_leaves": [7, 15], "learning_rate": [0.1, 0.05]})
    assert points == [
        {"learning_rate": 0.05, "num_leaves": 7},
        {"learning_rate": 0.05, "num_leaves": 15},
        {"learning_rate": 0.1, "num_leaves": 7},
        {"learning_rate": 0.1, "num_leaves": 15},
    ]


@pytest.mark.parametrize(
    ("search", "message"),
    [
        ({"num_leaves": []}, "at least one value"),
        ({"num_leaves": [7, 7]}, "repeats"),
        ({"bagging": [1]}, "not a tunable setting"),
        ({"num_leaves": ["7"]}, "finite number"),
    ],
)
def test_bad_search_spaces_are_refused(tmp_path, search, message):
    full = {key: [value] for key, value in BASE_SETTINGS.items()} | search
    with pytest.raises(TuningError, match=message):
        load_tuning_config(write_tuning(tmp_path, full))


def test_a_search_space_without_the_base_settings_is_refused(tmp_path):
    search = {key: [value] for key, value in BASE_SETTINGS.items()} | {"num_leaves": [31]}
    with pytest.raises(TuningError, match="base settings"):
        load_tuning_config(write_tuning(tmp_path, search))


@pytest.mark.parametrize("margin", [-0.1, 1.0])
def test_the_improvement_margin_must_be_a_fraction(tmp_path, margin):
    with pytest.raises(TuningError, match="min_relative_improvement"):
        load_tuning_config(write_tuning(tmp_path, margin=margin))


def point(settings, value):
    return {"settings": settings, "classifier": {"log_loss": value}, "status": "trained"}


def test_select_adopts_the_best_only_when_it_beats_the_base_by_the_margin():
    base = {"num_leaves": 15}
    results = [point(base, 0.50), point({"num_leaves": 7}, 0.40), point({"num_leaves": 31}, 0.45)]
    chosen = select(results, ("classifier", "log_loss"), base, 0.02)
    assert chosen["adopted"] is True and chosen["settings"] == {"num_leaves": 7}
    assert chosen["value"] == 0.40 and chosen["base_value"] == 0.50
    assert chosen["relative_improvement"] == pytest.approx(0.2)
    close = [point(base, 0.50), point({"num_leaves": 7}, 0.495)]
    kept = select(close, ("classifier", "log_loss"), base, 0.02)
    assert kept["adopted"] is False and kept["settings"] == base
    assert kept["best_settings"] == {"num_leaves": 7}


def test_select_breaks_ties_toward_the_simpler_setting():
    base = {"num_leaves": 31, "min_data_in_leaf": 5}
    results = [
        point(base, 0.9),
        point({"num_leaves": 15, "min_data_in_leaf": 5}, 0.4),
        point({"num_leaves": 7, "min_data_in_leaf": 5}, 0.4),
    ]
    chosen = select(results, ("classifier", "log_loss"), base, 0.02)
    assert chosen["settings"] == {"num_leaves": 7, "min_data_in_leaf": 5}


def test_an_unmeasured_base_is_kept_rather_than_bypassing_the_margin():
    base = {"num_leaves": 15}
    results = [
        {"settings": base, "classifier": None, "status": "failed"},
        point({"num_leaves": 7}, 0.1),
    ]
    chosen = select(results, ("classifier", "log_loss"), base, 0.02)
    assert chosen["adopted"] is False and chosen["settings"] == base
    assert chosen["base_measured"] is False and chosen["best_settings"] == {"num_leaves": 7}


def test_select_ignores_points_that_could_not_be_measured():
    base = {"num_leaves": 15}
    results = [
        point(base, 0.5),
        {"settings": {"num_leaves": 7}, "classifier": None, "status": "failed"},
    ]
    chosen = select(results, ("classifier", "log_loss"), base, 0.02)
    assert chosen["adopted"] is False and chosen["measured_points"] == 1


def test_sensitivity_summarizes_each_value_of_each_setting():
    results = [
        point({"num_leaves": 7, "learning_rate": 0.1}, 0.4),
        point({"num_leaves": 7, "learning_rate": 0.05}, 0.6),
        point({"num_leaves": 15, "learning_rate": 0.1}, 0.5),
    ]
    summary = sensitivity(results, ("classifier", "log_loss"))
    assert summary["num_leaves"]["7"] == {"points": 2, "min": 0.4, "median": 0.5, "max": 0.6}
    assert summary["learning_rate"]["0.05"]["points"] == 1


def test_run_reads_only_train_and_validation_and_writes_a_training_v2_proposal(tmp_path):
    calls = []
    outcome = tune(tmp_path, calls=calls)
    assert outcome.status == "tuned" and outcome.run_id
    assert calls == [("train", False), ("validation", False)]
    report = read_report(tmp_path)
    assert report["test_split_read"] is False
    assert report["points"] == 4 and len(report["results"]) == 4
    for result in report["results"]:
        assert result["status"] == "trained"
        assert result["classifier"]["log_loss"] > 0
        assert result["classifier"]["steps"] == len(SPLITS["validation"])
        assert result["regressor"]["mae"] >= 0
    for model in ("classifier", "regressor"):
        assert set(report["selection"][model]) >= {"settings", "adopted", "base_value"}
    assert set(report["sensitivity"]) == {"classifier", "regressor"}
    first = report["limitations"][0]
    assert first.startswith(
        f"Settings are chosen on the validation split: {len(SPLITS['validation'])} steps"
    )
    proposal = tomllib.loads((tmp_path / "out" / PROPOSAL).read_text())
    assert proposal["version"] == "training-v2"
    assert proposal["classifier"] == report["selection"]["classifier"]["settings"]
    assert proposal["regressor"] == report["selection"]["regressor"]["settings"]
    proposed = tmp_path / "proposed.toml"
    proposed.write_text((tmp_path / "out" / PROPOSAL).read_text())
    assert load_training_config(proposed).version.startswith("training-v2+")


def test_every_point_is_a_nested_mlflow_run_under_one_parent(tmp_path):
    outcome = tune(tmp_path)
    mlflow.set_tracking_uri(tracking(tmp_path)["tracking_uri"])
    runs = mlflow.search_runs(search_all_experiments=True, output_format="list")
    parents = [r for r in runs if "mlflow.parentRunId" not in r.data.tags]
    children = [r for r in runs if r.data.tags.get("mlflow.parentRunId") == outcome.run_id]
    assert [p.info.run_id for p in parents] == [outcome.run_id]
    assert len(children) == 4
    assert all("classifier_log_loss" in child.data.metrics for child in children)
    assert all(child.data.params["num_leaves"] in {"7", "15"} for child in children)


def test_a_rerun_gives_the_same_report(tmp_path):
    tune(tmp_path)
    first = read_report(tmp_path)["results"]
    tune(tmp_path)
    assert read_report(tmp_path)["results"] == first


def test_a_blocked_split_is_a_recorded_attempt_without_a_proposal(tmp_path):
    outcome = tune(tmp_path, loader=blocked)
    assert outcome.status == "blocked" and outcome.run_id
    report = read_report(tmp_path)
    assert report["status"] == "blocked" and "leakage" in report["reason"]
    assert not (tmp_path / "out" / PROPOSAL).exists()


def test_main_exits_2_on_a_bad_tuning_file(tmp_path, capsys):
    bad = tmp_path / "bad.toml"
    bad.write_text('version = "tuning-v9"\n')
    assert main(["--split", str(tmp_path / "s.json"), "--tuning", str(bad)]) == EXIT_ERROR
    assert "Tuning error" in capsys.readouterr().err
