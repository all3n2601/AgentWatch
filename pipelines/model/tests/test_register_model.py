"""register-model against a temporary MLflow registry; the models are synthetic fixtures."""

import json
import shutil

import pytest
from training_fixtures import dataset, require_lightgbm

require_lightgbm()

import mlflow
from agentwatch_model.register_model import (
    CANDIDATE,
    CHAMPION,
    RegistryError,
    load_registry_config,
    main,
    register,
    rollback,
)
from agentwatch_model.train_candidates import run as train
from mlflow import MlflowClient

SPLITS = {
    "train": dataset(25, "train"),
    "validation": dataset(6, "valid"),
    "test": dataset(6, "test"),
}


def load(path, name, contract_path, frozen_schema, allow_unchecked_leakage=False):
    return SPLITS[name]


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("registry")
    tracking = {"tracking_uri": f"sqlite:///{root / 'train.db'}", "artifact_root": root / "art"}
    candidates = root / "candidates"
    assert train(root / "split.json", candidates, loader=load, **tracking).status == "evaluated"
    return candidates


def calibrated(directory, training_version, **changes):
    directory.mkdir(parents=True, exist_ok=True)
    calibration = {
        "version": "calibration-v1+synthetic",
        "training_version": training_version,
        "fitted": True,
        # Promotable here so the champion rule can be exercised; real pairs still block.
        "blocks_promotion": False,
        "gates": {"calibration": "pass", "abstention": "pass"},
    } | changes
    (directory / "calibration.json").write_text(json.dumps(calibration))
    (directory / "report.json").write_text(json.dumps({"status": "evaluated"}))
    return directory


def candidates_copy(trained, directory, macro_f1=None, gate=None):
    shutil.copytree(trained, directory)
    report = json.loads((directory / "report.json").read_text())
    if macro_f1 is not None:
        report["comparison"]["macro_f1_all_classes"]["model"] = macro_f1
    if gate is not None:
        report["gates"]["clean_none_rate"]["outcome"] = gate
    (directory / "report.json").write_text(json.dumps(report))
    return directory, report["model"]["version"]


@pytest.fixture
def registry(tmp_path):
    return {"tracking_uri": f"sqlite:///{tmp_path / 'reg.db'}", "artifact_root": tmp_path / "ra"}


def push(trained, tmp_path, registry, name, **changes):
    bias = changes.pop("bias", None)
    # A distinct calibration version per push gives each pushed model its own identity.
    calibration_changes = {"version": f"calibration-v1+{name}"} | changes.pop("calibration", {})
    folder, version = candidates_copy(trained, tmp_path / name, **changes)
    calibration = calibrated(tmp_path / f"{name}-cal", version, **calibration_changes)
    return register(folder, calibration, bias, output=tmp_path / f"{name}-out", **registry)


def aliases(registry):
    mlflow.set_tracking_uri(registry["tracking_uri"])
    model = MlflowClient().get_registered_model(load_registry_config().model_name)
    return {alias: int(version) for alias, version in model.aliases.items()}


def test_the_first_eligible_version_becomes_candidate_and_champion(trained, tmp_path, registry):
    outcome = push(trained, tmp_path, registry, "a", macro_f1=0.80)
    assert outcome["version"] == 1 and outcome["champion"] == 1 and outcome["eligible"] is True
    assert aliases(registry) == {CANDIDATE: 1, CHAMPION: 1}
    record = json.loads((tmp_path / "a-out" / "registration.json").read_text())
    assert record["tags"]["blocks_promotion"] == "false"
    assert record["tags"]["bias_test_outcome"] == "unavailable"
    assert record["tags"]["split_sha256"] and record["tags"]["identity_sha256"]
    assert record["tags"]["training_version"].startswith("training-")


def test_a_worse_model_is_registered_but_the_champion_is_kept(trained, tmp_path, registry):
    push(trained, tmp_path, registry, "a", macro_f1=0.80)
    outcome = push(trained, tmp_path, registry, "b", macro_f1=0.70)
    assert outcome["version"] == 2 and outcome["champion"] == 1
    assert "macro F1" in outcome["reason"]
    assert aliases(registry) == {CANDIDATE: 2, CHAMPION: 1}


def test_an_equal_or_better_model_takes_over_and_can_be_rolled_back(trained, tmp_path, registry):
    push(trained, tmp_path, registry, "a", macro_f1=0.80)
    outcome = push(trained, tmp_path, registry, "b", macro_f1=0.80)
    assert outcome["champion"] == 2
    assert aliases(registry) == {CANDIDATE: 2, CHAMPION: 2}
    back = rollback(**registry)
    assert back == {"champion": 1, "replaced": 2}
    assert aliases(registry)[CHAMPION] == 1


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"gate": "fail"}, "failed gate"),
        ({"calibration": {"gates": {"calibration": "fail"}}}, "failed gate"),
        ({"bias": "fail"}, "bias check"),
    ],
)
def test_failed_gates_or_bias_keep_a_model_out_of_champion(
    trained, tmp_path, registry, changes, reason
):
    if changes.get("bias") == "fail":
        bias = tmp_path / "bias.json"
        version = json.loads((trained / "report.json").read_text())["model"]["version"]
        bias.write_text(
            json.dumps(
                {"models": {"x": {"training_version": version, "test": {"outcome": "fail"}}}}
            )
        )
        changes = {"bias": bias}
    outcome = push(trained, tmp_path, registry, "a", **changes)
    assert outcome["eligible"] is False and reason in outcome["reason"]
    assert aliases(registry) == {CANDIDATE: 1}


def test_unevaluated_candidates_or_an_unfitted_calibration_are_refused(trained, tmp_path, registry):
    with pytest.raises(RegistryError, match="fitted"):
        push(trained, tmp_path, registry, "a", calibration={"fitted": False})
    folder, version = candidates_copy(trained, tmp_path / "b")
    report = json.loads((folder / "report.json").read_text()) | {"status": "failed"}
    (folder / "report.json").write_text(json.dumps(report))
    with pytest.raises(RegistryError, match="evaluated"):
        register(
            folder, calibrated(tmp_path / "b-cal", version), None, output=tmp_path / "o", **registry
        )


def test_a_calibration_of_another_model_is_refused(trained, tmp_path, registry):
    folder, _ = candidates_copy(trained, tmp_path / "a")
    calibration = calibrated(tmp_path / "a-cal", "training-v9+00000000")
    with pytest.raises(RegistryError, match="training_version"):
        register(folder, calibration, None, output=tmp_path / "o", **registry)


def test_rollback_without_a_previous_champion_is_refused(trained, tmp_path, registry):
    push(trained, tmp_path, registry, "a", macro_f1=0.80)
    with pytest.raises(RegistryError, match="previous champion"):
        rollback(**registry)


def test_main_exits_2_on_a_refused_registration(tmp_path, capsys):
    argv = ["--candidates", str(tmp_path / "none"), "--calibrated", str(tmp_path / "none")]
    assert main(argv) == 2
    assert "Registry error" in capsys.readouterr().err


def test_a_model_that_blocks_promotion_never_becomes_champion(trained, tmp_path, registry):
    outcome = push(trained, tmp_path, registry, "a", calibration={"blocks_promotion": True})
    assert outcome["eligible"] is False and "blocks promotion" in outcome["reason"]
    assert aliases(registry) == {CANDIDATE: 1}


def test_a_rerun_of_the_same_model_changes_nothing(trained, tmp_path, registry):
    push(trained, tmp_path, registry, "a", macro_f1=0.80)
    again = register(
        tmp_path / "a", tmp_path / "a-cal", None, output=tmp_path / "again", **registry
    )
    assert again["version"] == 1 and "already registered" in again["reason"]
    assert aliases(registry) == {CANDIDATE: 1, CHAMPION: 1}
    with pytest.raises(RegistryError, match="previous champion"):
        rollback(**registry)


def test_a_model_scored_on_another_split_does_not_displace_the_champion(
    trained, tmp_path, registry
):
    push(trained, tmp_path, registry, "a", macro_f1=0.80)
    folder, version = candidates_copy(trained, tmp_path / "b", macro_f1=0.95)
    report = json.loads((folder / "report.json").read_text())
    report["split"]["sha256"] = "f" * 64
    (folder / "report.json").write_text(json.dumps(report))
    calibration = calibrated(tmp_path / "b-cal", version, version="calibration-v1+b")
    outcome = register(folder, calibration, None, output=tmp_path / "b-out", **registry)
    assert outcome["champion"] == 1 and "different split" in outcome["reason"]


def test_main_requires_a_bias_report_or_an_explicit_opt_out(trained, tmp_path, registry, capsys):
    folder, version = candidates_copy(trained, tmp_path / "a")
    calibration = calibrated(tmp_path / "a-cal", version)
    common = [
        "--candidates",
        str(folder),
        "--calibrated",
        str(calibration),
        "--output",
        str(tmp_path / "out"),
        "--tracking-uri",
        registry["tracking_uri"],
        "--artifact-root",
        str(registry["artifact_root"]),
    ]
    assert main(common) == 2 and "--no-bias-report" in capsys.readouterr().err
    assert main([*common, "--no-bias-report"]) == 0


def test_rollback_through_main(trained, tmp_path, registry):
    push(trained, tmp_path, registry, "a", macro_f1=0.80)
    push(trained, tmp_path, registry, "b", macro_f1=0.85)
    argv = [
        "--rollback",
        "--tracking-uri",
        registry["tracking_uri"],
        "--artifact-root",
        str(registry["artifact_root"]),
    ]
    assert main(argv) == 0
    assert aliases(registry)[CHAMPION] == 1
