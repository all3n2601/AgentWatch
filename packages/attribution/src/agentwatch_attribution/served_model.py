"""Load an M03/M04 model pair for serving, only after the serving contract's checks pass.

docs/model-serving-contract.md ("Artifacts") lists the checks: both model files hash to the
`model_files` SHA-256 values in preprocessing.json and in calibration.json, the training
versions agree, both files are `evaluated`, the calibration is `fitted`, the feature list is
MODEL_INPUTS, and the probability columns equal the class order in both files. Any failure
raises ServedModelError naming the failed check, the file and the field.

Each file is read once, at load: the hashes are of the bytes that are then parsed or loaded, so
a file replaced after its check is never served under the checked identity, and the identity
cannot change while the pair is served.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

from agentwatch_attribution.classes import is_number
from agentwatch_attribution.inputs import MODEL_INPUTS

if TYPE_CHECKING:
    import lightgbm

CHECK_CODES: tuple[str, ...] = (
    "artifact_unreadable",
    "hash_mismatch",
    "version_mismatch",
    "not_evaluated",
    "not_fitted",
    "columns_mismatch",
    "features_mismatch",
    "lightgbm_unavailable",
    "model_load_failed",
)
PREPROCESSING = "preprocessing.json"
CALIBRATION = "calibration.json"
# Fixed names, as M03 writes them; the `file` entries in model_files are never followed.
MODEL_FILES = {"classifier": "classifier.txt", "regressor": "regressor.txt"}
EVALUATED = "evaluated"
CLIP_BOUNDS = 2


class ServedModelError(ValueError):
    """A model pair that must not be served; `check` is the CHECK_CODES entry that failed."""

    def __init__(self, check: str, message: str) -> None:
        if check not in CHECK_CODES:
            raise ValueError(f"unknown check code {check!r}")
        super().__init__(message)
        self.check = check


@dataclass(frozen=True)
class ModelIdentity:
    """What was loaded, fixed at load: the versions, file hashes and calibration parameters."""

    training_version: str
    calibration_version: str
    classifier_sha256: str
    regressor_sha256: str
    preprocessing_sha256: str
    calibration_sha256: str
    temperature: float
    abstention_threshold: float
    class_order: tuple[str, ...]
    blocks_promotion: bool


@dataclass(frozen=True, eq=False)
class ServedModel:
    """A checked pair: both boosters, their identity and the excess-fraction clip range."""

    identity: ModelIdentity
    classifier: "lightgbm.Booster"
    regressor: "lightgbm.Booster"
    excess_fraction_low: float
    excess_fraction_high: float


@dataclass(frozen=True)
class Artifact:
    path: Path
    content: bytes
    sha256: str


def read_artifact(path: Path) -> Artifact:
    try:
        content = path.read_bytes()
    except OSError as error:
        raise ServedModelError("artifact_unreadable", f"Cannot read {path}: {error}") from error
    return Artifact(path, content, hashlib.sha256(content).hexdigest())


def json_object(artifact: Artifact) -> dict[str, object]:
    try:
        document = json.loads(artifact.content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ServedModelError(
            "artifact_unreadable", f"{artifact.path} is not valid JSON: {error}"
        ) from error
    if not isinstance(document, dict):
        raise ServedModelError("artifact_unreadable", f"{artifact.path} is not a JSON object")
    return document


def field(document: dict[str, object], *keys: str, where: Path) -> object:
    """document[keys...], or an artifact_unreadable error naming the file and field."""
    value: object = document
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise ServedModelError("artifact_unreadable", f"{where} has no {'.'.join(keys)}")
        value = value[key]
    return value


def text_field(document: dict[str, object], key: str, where: Path) -> str:
    value = field(document, key, where=where)
    if not isinstance(value, str) or not value:
        raise ServedModelError(
            "artifact_unreadable", f"{where} {key} must be a non-empty string, got {value!r}"
        )
    return value


def positive_finite(document: dict[str, object], key: str, where: Path) -> float:
    value = field(document, key, where=where)
    if not is_number(value) or value <= 0:
        raise ServedModelError(
            "artifact_unreadable", f"{where} {key} must be a finite number above 0, got {value!r}"
        )
    return float(value)


def class_order(document: dict[str, object], where: Path) -> tuple[str, ...]:
    value = field(document, "class_order", where=where)
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(name, str) for name in value)
        or len(set(value)) != len(value)
    ):
        raise ServedModelError(
            "artifact_unreadable",
            f"{where} class_order must be a non-empty list of distinct names, got {value!r}",
        )
    return tuple(value)


def excess_fraction_clip(document: dict[str, object], where: Path) -> tuple[float, float]:
    value = field(document, "excess_fraction_clip", where=where)
    if (
        not isinstance(value, list)
        or len(value) != CLIP_BOUNDS
        or not all(is_number(bound) for bound in value)
        or value[0] > value[1]
    ):
        raise ServedModelError(
            "artifact_unreadable",
            f"{where} excess_fraction_clip must be two finite numbers, low then high, "
            f"got {value!r}",
        )
    return float(value[0]), float(value[1])


def require_evaluated(document: dict[str, object], where: Path) -> None:
    status = field(document, "status", where=where)
    if status != EVALUATED:
        raise ServedModelError(
            "not_evaluated", f"{where} status is {status!r}, expected {EVALUATED!r}"
        )


def require_model_hash(model: Artifact, key: str, document: dict[str, object], where: Path) -> None:
    recorded = field(document, "model_files", key, "sha256", where=where)
    if recorded != model.sha256:
        raise ServedModelError(
            "hash_mismatch",
            f"{model.path} sha256 {model.sha256} does not match {where} "
            f"model_files.{key}.sha256 {recorded!r}",
        )


def import_lightgbm() -> ModuleType:
    """LightGBM, imported only when a pair is loaded: a missing package or OpenMP library is a
    load failure, not an import failure of every module that imports this one."""
    try:
        import lightgbm
    except (ImportError, OSError) as error:
        raise ServedModelError(
            "lightgbm_unavailable", f"Cannot import lightgbm: {error}"
        ) from error
    return lightgbm


def load_booster(library: ModuleType, model: Artifact, classes: int) -> "lightgbm.Booster":
    """The booster from the hashed bytes, refused unless it takes MODEL_INPUTS and has one
    output per class."""
    try:
        booster = library.Booster(model_str=model.content.decode("utf-8"))
    # ValueError covers text that is not UTF-8 and a malformed `pandas_categorical:` last line,
    # which LightGBM parses with json.loads and so fails as a JSONDecodeError.
    except (ValueError, library.basic.LightGBMError) as error:
        raise ServedModelError("model_load_failed", f"Cannot load {model.path}: {error}") from error
    if booster.num_feature() != len(MODEL_INPUTS):
        raise ServedModelError(
            "model_load_failed",
            f"{model.path} takes {booster.num_feature()} features, expected {len(MODEL_INPUTS)}",
        )
    if booster.num_model_per_iteration() != classes:
        raise ServedModelError(
            "model_load_failed",
            f"{model.path} has {booster.num_model_per_iteration()} outputs per row, "
            f"expected {classes}",
        )
    return booster


def load_served_model(candidates_dir: Path, calibration_dir: Path) -> ServedModel:
    """The M03 boosters with their M04 calibration, after every serving contract check."""
    library = import_lightgbm()
    preprocessing_file = read_artifact(candidates_dir / PREPROCESSING)
    calibration_file = read_artifact(calibration_dir / CALIBRATION)
    models = {key: read_artifact(candidates_dir / name) for key, name in MODEL_FILES.items()}
    preprocessing, pre_path = json_object(preprocessing_file), preprocessing_file.path
    calibration, cal_path = json_object(calibration_file), calibration_file.path

    require_evaluated(preprocessing, pre_path)
    require_evaluated(calibration, cal_path)
    fitted = field(calibration, "fitted", where=cal_path)
    if fitted is not True:
        raise ServedModelError("not_fitted", f"{cal_path} fitted is {fitted!r}, expected true")
    training_version = text_field(preprocessing, "training_version", pre_path)
    calibrated_version = text_field(calibration, "training_version", cal_path)
    if calibrated_version != training_version:
        raise ServedModelError(
            "version_mismatch",
            f"{cal_path} training_version {calibrated_version!r} does not match "
            f"{pre_path} training_version {training_version!r}",
        )
    calibration_version = text_field(calibration, "version", cal_path)

    features = field(preprocessing, "features", where=pre_path)
    if features != list(MODEL_INPUTS):
        raise ServedModelError(
            "features_mismatch",
            f"{pre_path} features {features!r} do not equal MODEL_INPUTS {list(MODEL_INPUTS)!r}",
        )
    classes = class_order(preprocessing, pre_path)
    columns = field(preprocessing, "probability_columns", where=pre_path)
    if columns != list(classes):
        raise ServedModelError(
            "columns_mismatch",
            f"{pre_path} probability_columns {columns!r} do not equal its class_order "
            f"{list(classes)!r}",
        )
    calibrated_columns = field(calibration, "probability_columns", where=cal_path)
    if calibrated_columns != columns:
        raise ServedModelError(
            "columns_mismatch",
            f"{cal_path} probability_columns {calibrated_columns!r} do not equal "
            f"{pre_path} probability_columns {columns!r}",
        )

    temperature = positive_finite(calibration, "temperature", cal_path)
    threshold = positive_finite(calibration, "abstention_threshold", cal_path)
    blocks_promotion = field(calibration, "blocks_promotion", where=cal_path)
    if not isinstance(blocks_promotion, bool):
        raise ServedModelError(
            "artifact_unreadable",
            f"{cal_path} blocks_promotion must be true or false, got {blocks_promotion!r}",
        )
    clip_low, clip_high = excess_fraction_clip(preprocessing, pre_path)

    for key, model in models.items():
        require_model_hash(model, key, preprocessing, pre_path)
        require_model_hash(model, key, calibration, cal_path)
    classifier = load_booster(library, models["classifier"], len(classes))
    regressor = load_booster(library, models["regressor"], 1)

    identity = ModelIdentity(
        training_version=training_version,
        calibration_version=calibration_version,
        classifier_sha256=models["classifier"].sha256,
        regressor_sha256=models["regressor"].sha256,
        preprocessing_sha256=preprocessing_file.sha256,
        calibration_sha256=calibration_file.sha256,
        temperature=temperature,
        abstention_threshold=threshold,
        class_order=classes,
        blocks_promotion=blocks_promotion,
    )
    return ServedModel(identity, classifier, regressor, clip_low, clip_high)
