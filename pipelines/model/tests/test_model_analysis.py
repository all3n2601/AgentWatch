"""model-analysis on candidates trained from synthetic splits; nothing here is evidence."""

import json

import numpy as np
import pytest
from training_fixtures import dataset, require_lightgbm

require_lightgbm()

from agentwatch_model.features import FEATURES, feature_matrix
from agentwatch_model.model_analysis import (
    EXIT_ERROR,
    AnalysisError,
    main,
    numeric_order,
    parse_named,
    pyplot,
    run,
    shap_importance,
)
from agentwatch_model.train_candidates import run as train
from agentwatch_model.tune_candidates import run as tune

SPLITS = {
    "train": dataset(25, "train"),
    "validation": dataset(6, "valid"),
    "test": dataset(6, "test"),
}
PNG = b"\x89PNG\r\n\x1a\n"


def load(path, name, contract_path, frozen_schema, allow_unchecked_leakage=False):
    return SPLITS[name]


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("analysis")
    tracking = {"tracking_uri": f"sqlite:///{root / 'mlflow.db'}", "artifact_root": root / "art"}
    candidates = root / "candidates"
    assert train(root / "split.json", candidates, loader=load, **tracking).status == "evaluated"
    tuning = root / "tuning.toml"
    tuning.write_text(
        'version = "tuning-v1"\n\n[selection]\nmin_relative_improvement = 0.02\n\n[search]\n'
        "learning_rate = [0.05, 0.1]\nnum_leaves = [7, 15]\nmin_data_in_leaf = [5]\n"
        "lambda_l2 = [1.0]\nfeature_fraction = [1.0]\n"
    )
    tuned = root / "tuned"
    assert tune(root / "split.json", tuned, tuning_path=tuning, loader=load, **tracking).run_id
    return {"root": root, "candidates": candidates, "tuning": tuned / "report.json"}


def analyse(trained, output, **overrides):
    return run(
        overrides.pop("candidates", {"v1": trained["candidates"], "v2": trained["candidates"]}),
        trained["root"] / "split.json",
        output,
        explain=overrides.pop("explain", "v2"),
        tuning_report=overrides.pop("tuning_report", trained["tuning"]),
        loader=load,
        **overrides,
    )


def test_parse_named_reads_name_equals_directory(tmp_path):
    assert parse_named([f"v1={tmp_path}", f"v2={tmp_path}"]) == {"v1": tmp_path, "v2": tmp_path}
    for bad in (["v1"], [f"={tmp_path}"], [f"v1={tmp_path}", f"v1={tmp_path}"]):
        with pytest.raises(AnalysisError):
            parse_named(bad)


def test_run_writes_every_figure_and_the_analysis(trained, tmp_path):
    path = analyse(trained, tmp_path / "out")
    analysis = json.loads(path.read_text())
    names = {figure["file"] for figure in analysis["figures"]}
    assert names == {
        "confusion-rules.png",
        "confusion-v1.png",
        "confusion-v2.png",
        "comparison.png",
        "shap-v2.png",
        "sensitivity.png",
    }
    for figure in analysis["figures"]:
        data = (tmp_path / "out" / figure["file"]).read_bytes()
        assert data.startswith(PNG) and figure["sha256"]
    assert set(analysis["models"]) == {"rules-v1", "v1", "v2"}
    for row in analysis["models"].values():
        assert set(row) >= {"macro_f1", "clean_none_rate"}
    assert analysis["evaluation"] is False and analysis["shap"]["rows"] == len(SPLITS["test"])
    assert analysis["sensitivity"]["tuning_version"].startswith("tuning-v1+")


def test_the_same_inputs_give_the_same_analysis(trained, tmp_path):
    first = analyse(trained, tmp_path / "a").read_text()
    second = analyse(trained, tmp_path / "b").read_text()
    assert first == second


def test_shap_values_add_up_to_the_raw_score(trained):
    import lightgbm

    booster = lightgbm.Booster(model_file=str(trained["candidates"] / "classifier.txt"))
    rows = SPLITS["test"]
    importance = shap_importance(booster, rows, ("none", "cpu_sandbox", "inference", "retrieval"))
    assert importance["features"] == list(FEATURES)
    assert all(value >= 0 for value in importance["overall"].values())
    matrix = feature_matrix(rows)
    contributions = booster.predict(matrix, pred_contrib=True).reshape(len(rows), 4, -1)
    raw = booster.predict(matrix, raw_score=True)
    assert np.allclose(contributions.sum(axis=2), raw)


def test_without_a_tuning_report_there_is_no_sensitivity_figure(trained, tmp_path):
    analysis = json.loads(analyse(trained, tmp_path / "out", tuning_report=None).read_text())
    assert "sensitivity.png" not in {figure["file"] for figure in analysis["figures"]}
    assert analysis["sensitivity"] is None


def test_explain_must_name_a_given_candidates_folder(trained, tmp_path):
    with pytest.raises(AnalysisError, match="explain"):
        analyse(trained, tmp_path / "out", explain="v9")


def test_a_folder_without_an_evaluated_report_is_refused(trained, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(AnalysisError, match="report"):
        analyse(trained, tmp_path / "out", candidates={"v1": empty}, explain="v1")


def test_main_exits_2_on_a_bad_candidates_argument(tmp_path, capsys):
    argv = ["--split", str(tmp_path / "s.json"), "--candidates", "v1", "--explain", "v1"]
    assert main(argv) == EXIT_ERROR
    assert "Analysis error" in capsys.readouterr().err


def test_sensitivity_values_are_plotted_in_numeric_order():
    assert numeric_order({"10": {}, "20": {}, "5": {}}) == ["5", "10", "20"]
    assert numeric_order({"0.1": {}, "0.03": {}, "0.05": {}}) == ["0.03", "0.05", "0.1"]


def test_figures_draw_under_a_notebook_kernel_backend(monkeypatch):
    # Jupyter exports this for its inline backend; a `make` subprocess inherits it.
    monkeypatch.setenv("MPLBACKEND", "module://matplotlib_inline.backend_inline")
    plt = pyplot()
    assert plt.get_backend().lower() == "agg"
