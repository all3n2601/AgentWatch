import hashlib
import json
import sys
from collections.abc import Callable
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from agentwatch_attribution.classes import RESOURCE_CLASSES
from agentwatch_attribution.inputs import MODEL_INPUTS
from agentwatch_attribution.served_model import (
    CHECK_CODES,
    ServedModel,
    ServedModelError,
    load_served_model,
)
from pair_fixtures import (
    CALIBRATION,
    CALIBRATION_VERSION,
    MODEL_FILES,
    PREPROCESSING,
    TEMPERATURE,
    TRAINING_VERSION,
    write_synthetic_pair,
)

Pair = tuple[Path, Path]


@pytest.fixture
def pair(tmp_path: Path) -> Pair:
    return write_synthetic_pair(tmp_path)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rewrite(path: Path, change: Callable[[dict], None]) -> None:
    document = json.loads(path.read_text())
    change(document)
    path.write_text(json.dumps(document))


def refusal(pair: Pair) -> ServedModelError:
    with pytest.raises(ServedModelError) as caught:
        load_served_model(*pair)
    assert caught.value.check in CHECK_CODES
    return caught.value


def record_model_hash(pair: Pair, key: str) -> None:
    """Make both JSON files name the current hash of a model file, as a rewrite would."""
    candidates_dir, calibration_dir = pair
    digest = sha256_of(candidates_dir / MODEL_FILES[key])

    def change(document: dict) -> None:
        document["model_files"][key]["sha256"] = digest

    rewrite(candidates_dir / PREPROCESSING, change)
    rewrite(calibration_dir / CALIBRATION, change)


def test_good_pair_loads_and_reports_identity(pair: Pair) -> None:
    candidates_dir, calibration_dir = pair
    served = load_served_model(candidates_dir, calibration_dir)
    assert isinstance(served, ServedModel)
    identity = served.identity
    assert identity.training_version == TRAINING_VERSION
    assert identity.calibration_version == CALIBRATION_VERSION
    assert identity.classifier_sha256 == sha256_of(candidates_dir / "classifier.txt")
    assert identity.regressor_sha256 == sha256_of(candidates_dir / "regressor.txt")
    assert identity.preprocessing_sha256 == sha256_of(candidates_dir / PREPROCESSING)
    assert identity.calibration_sha256 == sha256_of(calibration_dir / CALIBRATION)
    assert identity.temperature == TEMPERATURE
    assert identity.abstention_threshold == 0.6
    assert identity.class_order == RESOURCE_CLASSES
    assert identity.blocks_promotion is True
    assert (served.excess_fraction_low, served.excess_fraction_high) == (0.0, 1.0)
    assert served.classifier.num_feature() == len(MODEL_INPUTS)
    with pytest.raises(FrozenInstanceError):
        identity.training_version = "other"


def test_identity_is_fixed_at_load(pair: Pair) -> None:
    served = load_served_model(*pair)
    rewrite(pair[1] / CALIBRATION, lambda document: document.update(abstention_threshold=0.9))
    (pair[0] / "classifier.txt").write_text("replaced after load")
    assert served.identity.abstention_threshold == 0.6
    assert served.classifier.num_feature() == len(MODEL_INPUTS)


def test_unblocked_pair_reports_it(tmp_path: Path) -> None:
    served = load_served_model(*write_synthetic_pair(tmp_path, blocks_promotion=False))
    assert served.identity.blocks_promotion is False


@pytest.mark.parametrize(
    "missing",
    [
        "candidates/classifier.txt",
        "candidates/regressor.txt",
        "candidates/preprocessing.json",
        "calibration/calibration.json",
    ],
)
def test_missing_file_is_artifact_unreadable(pair: Pair, missing: str) -> None:
    path = pair[0].parent / missing
    path.unlink()
    error = refusal(pair)
    assert error.check == "artifact_unreadable"
    assert str(path) in str(error)


@pytest.mark.parametrize("text", ["{not json", "[1, 2]", '"text"'])
def test_unparsable_or_non_object_json_is_artifact_unreadable(pair: Pair, text: str) -> None:
    (pair[1] / CALIBRATION).write_text(text)
    error = refusal(pair)
    assert error.check == "artifact_unreadable"
    assert CALIBRATION in str(error)


@pytest.mark.parametrize(
    ("file", "field"),
    [
        (PREPROCESSING, "features"),
        (PREPROCESSING, "class_order"),
        (PREPROCESSING, "excess_fraction_clip"),
        (PREPROCESSING, "model_files"),
        (CALIBRATION, "temperature"),
        (CALIBRATION, "blocks_promotion"),
        (CALIBRATION, "version"),
        (CALIBRATION, "probability_columns"),
    ],
)
def test_missing_field_is_artifact_unreadable(pair: Pair, file: str, field: str) -> None:
    directory = pair[0] if file == PREPROCESSING else pair[1]
    rewrite(directory / file, lambda document: document.pop(field))
    error = refusal(pair)
    assert error.check == "artifact_unreadable"
    assert file in str(error) and field in str(error)


@pytest.mark.parametrize("field", ["temperature", "abstention_threshold"])
@pytest.mark.parametrize("value", [True, "0.9", None, 0, -0.5, float("nan"), float("inf")])
def test_calibration_parameter_must_be_a_positive_finite_number(
    pair: Pair, field: str, value: object
) -> None:
    path = pair[1] / CALIBRATION
    document = json.loads(path.read_text())
    document[field] = value
    path.write_text(json.dumps(document))  # allow_nan: the file may say NaN or Infinity
    error = refusal(pair)
    assert error.check == "artifact_unreadable"
    assert field in str(error)


@pytest.mark.parametrize("value", [1, "true", None])
def test_blocks_promotion_must_be_a_boolean(pair: Pair, value: object) -> None:
    rewrite(pair[1] / CALIBRATION, lambda document: document.update(blocks_promotion=value))
    error = refusal(pair)
    assert error.check == "artifact_unreadable"
    assert "blocks_promotion" in str(error)


@pytest.mark.parametrize(
    "clip", [[0.0], [0.0, 1.0, 2.0], [1.0, 0.0], [0.0, "1"], [True, 1.0], "0,1", [0.0, None]]
)
def test_excess_fraction_clip_must_be_an_ordered_number_pair(pair: Pair, clip: object) -> None:
    rewrite(pair[0] / PREPROCESSING, lambda document: document.update(excess_fraction_clip=clip))
    error = refusal(pair)
    assert error.check == "artifact_unreadable"
    assert "excess_fraction_clip" in str(error)


def test_tampered_classifier_is_hash_mismatch(pair: Pair) -> None:
    path = pair[0] / "classifier.txt"
    path.write_text(path.read_text() + "\n")
    error = refusal(pair)
    assert error.check == "hash_mismatch"
    assert "classifier.txt" in str(error) and PREPROCESSING in str(error)


def test_tampered_regressor_is_hash_mismatch(pair: Pair) -> None:
    path = pair[0] / "regressor.txt"
    path.write_text(path.read_text() + "\n")
    error = refusal(pair)
    assert error.check == "hash_mismatch"
    assert "regressor.txt" in str(error)


def test_classifier_hash_must_match_in_calibration_json_too(pair: Pair) -> None:
    def change(document: dict) -> None:
        document["model_files"]["classifier"]["sha256"] = "0" * 64

    rewrite(pair[1] / CALIBRATION, change)
    error = refusal(pair)
    assert error.check == "hash_mismatch"
    assert "classifier.txt" in str(error) and CALIBRATION in str(error)


def test_regressor_hash_must_match_in_calibration_json_too(pair: Pair) -> None:
    def change(document: dict) -> None:
        document["model_files"]["regressor"]["sha256"] = "0" * 64

    rewrite(pair[1] / CALIBRATION, change)
    error = refusal(pair)
    assert error.check == "hash_mismatch"
    assert "regressor.txt" in str(error) and CALIBRATION in str(error)


def test_training_version_disagreement_is_version_mismatch(pair: Pair) -> None:
    rewrite(pair[1] / CALIBRATION, lambda document: document.update(training_version="other"))
    error = refusal(pair)
    assert error.check == "version_mismatch"
    assert "training_version" in str(error)


@pytest.mark.parametrize("file", [PREPROCESSING, CALIBRATION])
def test_status_not_evaluated_is_refused(pair: Pair, file: str) -> None:
    directory = pair[0] if file == PREPROCESSING else pair[1]
    rewrite(directory / file, lambda document: document.update(status="blocked"))
    error = refusal(pair)
    assert error.check == "not_evaluated"
    assert file in str(error) and "status" in str(error)


@pytest.mark.parametrize("fitted", [False, "true", 1])
def test_unfitted_calibration_is_refused(pair: Pair, fitted: object) -> None:
    rewrite(pair[1] / CALIBRATION, lambda document: document.update(fitted=fitted))
    error = refusal(pair)
    assert error.check == "not_fitted"
    assert "fitted" in str(error)


def test_calibration_columns_disagreeing_with_preprocessing_is_columns_mismatch(
    pair: Pair,
) -> None:
    columns = list(reversed(RESOURCE_CLASSES))
    rewrite(pair[1] / CALIBRATION, lambda document: document.update(probability_columns=columns))
    error = refusal(pair)
    assert error.check == "columns_mismatch"
    assert CALIBRATION in str(error) and "probability_columns" in str(error)


def test_probability_columns_disagreeing_with_class_order_is_columns_mismatch(pair: Pair) -> None:
    columns = list(reversed(RESOURCE_CLASSES))

    def change(document: dict) -> None:
        document["probability_columns"] = columns

    rewrite(pair[0] / PREPROCESSING, change)
    rewrite(pair[1] / CALIBRATION, change)
    error = refusal(pair)
    assert error.check == "columns_mismatch"
    assert PREPROCESSING in str(error) and "class_order" in str(error)


def test_reordered_feature_list_is_features_mismatch(pair: Pair) -> None:
    features = list(MODEL_INPUTS)
    features[0], features[1] = features[1], features[0]
    rewrite(pair[0] / PREPROCESSING, lambda document: document.update(features=features))
    error = refusal(pair)
    assert error.check == "features_mismatch"
    assert PREPROCESSING in str(error) and "features" in str(error)


@pytest.mark.parametrize("key", ["classifier", "regressor"])
def test_corrupt_model_text_is_model_load_failed(pair: Pair, key: str) -> None:
    (pair[0] / MODEL_FILES[key]).write_text("not a LightGBM model\n")
    record_model_hash(pair, key)
    error = refusal(pair)
    assert error.check == "model_load_failed"
    assert MODEL_FILES[key] in str(error)


def test_classifier_with_the_wrong_class_count_is_model_load_failed(pair: Pair) -> None:
    rewrite(
        pair[0] / PREPROCESSING,
        lambda document: document.update(
            class_order=["none", "cpu_sandbox", "inference"],
            probability_columns=["none", "cpu_sandbox", "inference"],
        ),
    )
    rewrite(
        pair[1] / CALIBRATION,
        lambda document: document.update(probability_columns=["none", "cpu_sandbox", "inference"]),
    )
    error = refusal(pair)
    assert error.check == "model_load_failed"
    assert "classifier.txt" in str(error)


def test_missing_lightgbm_is_lightgbm_unavailable(
    pair: Pair, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A None entry makes `import lightgbm` raise ImportError, as an absent package would.
    monkeypatch.setitem(sys.modules, "lightgbm", None)
    error = refusal(pair)
    assert error.check == "lightgbm_unavailable"


def test_lightgbm_failing_to_load_its_library_is_lightgbm_unavailable(
    pair: Pair, monkeypatch: pytest.MonkeyPatch
) -> None:
    import builtins

    real_import = builtins.__import__

    def missing_openmp(name: str, *args: object, **kwargs: object) -> object:
        if name == "lightgbm":
            raise OSError("libomp.dylib not found")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_openmp)
    error = refusal(pair)
    assert error.check == "lightgbm_unavailable"
    assert "libomp" in str(error)


def test_error_rejects_unknown_check_codes() -> None:
    with pytest.raises(ValueError, match="unknown check"):
        ServedModelError("made_up", "message")
