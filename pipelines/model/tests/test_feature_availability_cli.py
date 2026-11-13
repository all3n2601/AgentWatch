"""held-out-feature-availability end to end on the synthetic sweep, frozen pair and split loader
of the held-out scorer's tests. Nothing here is recorded evidence."""

import json

from training_fixtures import require_lightgbm

require_lightgbm()

from agentwatch_model.feature_availability import (
    EXIT_ERROR,
    REPORT_FILE,
    TABLE_FILE,
    main,
    run,
)
from agentwatch_model.features import FEATURES
from agentwatch_model.held_out_rows import held_out_rows
from frozen_model_fixtures import SYNTHETIC_VERSIONS
from test_held_out_rows import sweep, write_truths  # noqa: F401
from test_score_held_out import prepared, split_loader  # noqa: F401
from training_fixtures import dataset, featured_row

TRAIN_ROWS = len(dataset(30, "train"))


def run_prepared(prepared, **overrides):  # noqa: F811
    return run(
        [prepared["truth"]],
        prepared["root"],
        prepared["split"],
        overrides.pop("output", prepared["output"]),
        candidates_dir=prepared["pair"].candidates_dir,
        calibrated_dir=prepared["pair"].calibrated_dir,
        loader=overrides.pop("loader", prepared["loader"]),
        versions=SYNTHETIC_VERSIONS,
        **overrides,
    )


def cli_arguments(prepared, output=None):  # noqa: F811
    return [
        "--truth",
        str(prepared["truth"]),
        "--sweep-root",
        str(prepared["root"]),
        "--split",
        str(prepared["split"]),
        "--candidates",
        str(prepared["pair"].candidates_dir),
        "--calibrated",
        str(prepared["pair"].calibrated_dir),
        "--output",
        str(output or prepared["output"]),
    ]


def test_run_writes_the_report_and_table_for_the_scored_rows(prepared):  # noqa: F811
    path = run_prepared(prepared)
    output = prepared["output"]
    assert path == output / REPORT_FILE
    assert sorted(item.name for item in output.iterdir()) == sorted([REPORT_FILE, TABLE_FILE])
    report = json.loads(path.read_text())
    rows = held_out_rows([prepared["truth"]], prepared["root"]).rows
    assert report["row_sets"]["held_out"] == len(rows) > 0
    assert report["row_sets"]["train"] == TRAIN_ROWS
    assert report["row_sets"]["train_not_classified"] == 0
    assert report["row_sets"]["held_out_by_configuration"] == {"cfg-default": len(rows)}
    assert [item["feature"] for item in report["features"]] != []
    assert {item["feature"] for item in report["features"]} == set(FEATURES)
    shares = [item["gain_share"] for item in report["features"]]
    assert abs(sum(shares) - 1.0) < 1e-9
    assert report["model"]["frozen"]["classifier_sha256"]
    assert report["split"]["file"] == "split.json"
    (truth,) = report["truths"].values()
    assert set(truth) == {"label_version", "truth_sha256", "steps_sha256", "steps_summary_sha256"}
    assert report["evaluation"] is False and report["held_out"] is True
    table = (output / TABLE_FILE).read_text()
    assert table.startswith("| Feature |")
    assert "| Feature | Gain share | Train p5 / p50 / p95 |" in table


def test_run_is_byte_for_byte_reproducible(prepared, tmp_path):  # noqa: F811
    first = run_prepared(prepared, output=tmp_path / "first").read_bytes()
    second = run_prepared(prepared, output=tmp_path / "second").read_bytes()
    assert first == second


def pin_synthetic_pair(monkeypatch):
    """The CLI always loads the pinned frozen pair; main() tests stand the synthetic pair in."""
    monkeypatch.setattr("agentwatch_model.feature_availability.FROZEN_VERSIONS", SYNTHETIC_VERSIONS)


def test_the_cli_refuses_any_pair_but_the_frozen_one(prepared, capsys, monkeypatch):  # noqa: F811
    monkeypatch.setattr("agentwatch_model.feature_availability.load_split", prepared["loader"])
    assert main(cli_arguments(prepared)) == EXIT_ERROR
    assert "calibration-v1+08119e78" in capsys.readouterr().err
    assert not prepared["output"].exists()


def test_main_prints_the_table_and_returns_zero(prepared, capsys, monkeypatch):  # noqa: F811
    pin_synthetic_pair(monkeypatch)
    monkeypatch.setattr("agentwatch_model.feature_availability.load_split", prepared["loader"])
    assert main(cli_arguments(prepared)) == 0
    printed = capsys.readouterr().out
    assert printed.startswith("| Feature |")
    assert str(prepared["output"] / REPORT_FILE) in printed


def test_output_inside_held_out_evidence_is_refused(prepared, tmp_path, capsys, monkeypatch):  # noqa: F811
    pin_synthetic_pair(monkeypatch)
    forbidden = tmp_path / "data" / "held-out" / "availability"
    assert main(cli_arguments(prepared, output=forbidden)) == EXIT_ERROR
    assert "data/held-out" in capsys.readouterr().err
    assert not forbidden.exists()


def test_a_mismatched_frozen_pair_is_refused_before_writing(prepared, capsys, monkeypatch):  # noqa: F811
    pin_synthetic_pair(monkeypatch)
    (prepared["pair"].calibrated_dir / "calibration.json").unlink()
    assert main(cli_arguments(prepared)) == EXIT_ERROR
    assert "Feature availability error" in capsys.readouterr().err
    assert not prepared["output"].exists()


def test_no_judged_step_is_refused(prepared, capsys, monkeypatch):  # noqa: F811
    pin_synthetic_pair(monkeypatch)
    lines = (prepared["truth"] / "truth.jsonl").read_text().splitlines()
    unjudged = [
        json.loads(line)
        | {
            "truths": {"queue": None, "slower": None, "protocol_clean": None},
            "truth_reasons": {
                "queue": "root_step",
                "slower": "root_step",
                "protocol_clean": "root_step",
            },
        }
        for line in lines
    ]
    write_truths(prepared["truth"], unjudged)
    monkeypatch.setattr("agentwatch_model.feature_availability.load_split", prepared["loader"])
    assert main(cli_arguments(prepared)) == EXIT_ERROR
    assert "no held-out step" in capsys.readouterr().err
    assert not prepared["output"].exists()


def test_a_refusing_split_is_an_error(prepared, capsys, monkeypatch):  # noqa: F811
    pin_synthetic_pair(monkeypatch)
    from test_score_held_out import refusing

    monkeypatch.setattr("agentwatch_model.feature_availability.load_split", refusing)
    assert main(cli_arguments(prepared)) == EXIT_ERROR
    assert "leakage check is blocked" in capsys.readouterr().err


def test_train_rows_are_the_classifiers_single_class_rows(prepared):  # noqa: F811
    mixed = featured_row("none", 99, run_id="run-mixed") | {"label_resource": "mixed"}
    loader = split_loader(prepared["pair"], planted={"train": [mixed]})
    report = json.loads(run_prepared(prepared, loader=loader).read_text())
    assert report["row_sets"]["train"] == TRAIN_ROWS
    assert report["row_sets"]["train_not_classified"] == 1


def test_a_held_out_run_found_in_the_split_is_refused(prepared, capsys, monkeypatch):  # noqa: F811
    pin_synthetic_pair(monkeypatch)
    planted = {"test": [featured_row("none", 0, run_id=prepared["run_id"])]}
    monkeypatch.setattr(
        "agentwatch_model.feature_availability.load_split",
        split_loader(prepared["pair"], planted=planted),
    )
    assert main(cli_arguments(prepared)) == EXIT_ERROR
    err = capsys.readouterr().err
    assert "appear in the split" in err and prepared["run_id"] in err
    assert not prepared["output"].exists()


def test_a_booster_that_does_not_reproduce_validation_is_refused(prepared, capsys, monkeypatch):  # noqa: F811
    pin_synthetic_pair(monkeypatch)
    monkeypatch.setattr(
        "agentwatch_model.feature_availability.load_split",
        split_loader(prepared["pair"], validation=dataset(8, "other")),
    )
    assert main(cli_arguments(prepared)) == EXIT_ERROR
    assert "validation rows" in capsys.readouterr().err
    assert not prepared["output"].exists()
