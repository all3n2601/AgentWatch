import sys
from pathlib import Path

import pytest
from agentwatch_attribution.served_model import CHECK_CODES, ServedModel
from agentwatch_worker.attribution_model import ModelService
from pair_fixtures import CALIBRATION, MODEL_FILES, TRAINING_VERSION, write_synthetic_pair

CANDIDATES = "AGENTWATCH_MODEL_CANDIDATES_DIR"
CALIBRATION_DIR = "AGENTWATCH_MODEL_CALIBRATION_DIR"
ALLOW_UNPROMOTED = "AGENTWATCH_ALLOW_UNPROMOTED_MODEL"


def environment(pair: tuple[Path, Path], **extra: str) -> dict[str, str]:
    candidates, calibration = pair
    return {CANDIDATES: str(candidates), CALIBRATION_DIR: str(calibration), **extra}


@pytest.fixture
def blocked_pair(tmp_path: Path) -> tuple[Path, Path]:
    return write_synthetic_pair(tmp_path, blocks_promotion=True)


@pytest.fixture
def promoted_pair(tmp_path: Path) -> tuple[Path, Path]:
    return write_synthetic_pair(tmp_path, blocks_promotion=False)


@pytest.mark.parametrize("environ", [{}, {CANDIDATES: "", CALIBRATION_DIR: ""}])
def test_unconfigured_is_not_configured(environ: dict[str, str]) -> None:
    service = ModelService.from_environ(environ)
    assert service.status.state == "not_configured"
    assert service.status.reason is None
    assert service.status.identity is None
    assert service.served is None


@pytest.mark.parametrize("variable", [CANDIDATES, CALIBRATION_DIR])
def test_one_directory_only_is_load_failed(variable: str, promoted_pair: tuple[Path, Path]) -> None:
    only_one = {variable: environment(promoted_pair)[variable]}
    service = ModelService.from_environ(only_one)
    assert service.status.state == "load_failed"
    assert service.status.reason == "artifact_unreadable"
    assert service.status.identity is None
    assert service.served is None


def test_blocked_pair_without_the_flag_is_refused_and_not_loaded(
    blocked_pair: tuple[Path, Path],
) -> None:
    service = ModelService.from_environ(environment(blocked_pair))
    assert service.status.state == "not_promoted_refused"
    assert service.served is None
    identity = service.status.identity
    assert identity is not None
    assert identity.blocks_promotion is True
    assert identity.training_version == TRAINING_VERSION


@pytest.mark.parametrize("value", ["1", "true", "TRUE", " True "])
def test_blocked_pair_with_the_flag_is_ready_not_promoted(
    value: str, blocked_pair: tuple[Path, Path]
) -> None:
    service = ModelService.from_environ(environment(blocked_pair, **{ALLOW_UNPROMOTED: value}))
    assert service.status.state == "ready_not_promoted"
    assert isinstance(service.served, ServedModel)
    assert service.status.identity is service.served.identity
    assert service.status.identity.blocks_promotion is True


def test_pair_with_blocks_promotion_false_is_ready_promoted(
    promoted_pair: tuple[Path, Path],
) -> None:
    service = ModelService.from_environ(environment(promoted_pair))
    assert service.status.state == "ready_promoted"
    assert service.status.reason is None
    assert isinstance(service.served, ServedModel)
    assert service.status.identity is service.served.identity
    assert service.status.identity.blocks_promotion is False


def test_tampered_pair_is_load_failed_with_the_check_code(
    promoted_pair: tuple[Path, Path],
) -> None:
    candidates, _ = promoted_pair
    classifier = candidates / MODEL_FILES["classifier"]
    classifier.write_text(classifier.read_text() + "\n# tampered\n")
    service = ModelService.from_environ(environment(promoted_pair))
    assert service.status.state == "load_failed"
    assert service.status.reason == "hash_mismatch"
    assert service.status.reason in CHECK_CODES
    assert service.status.identity is None
    assert service.served is None


def test_unavailable_lightgbm_is_load_failed_with_its_check_code(
    promoted_pair: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    # A None entry makes `import lightgbm` raise ImportError, as an absent package would.
    monkeypatch.setitem(sys.modules, "lightgbm", None)
    service = ModelService.from_environ(environment(promoted_pair))
    assert service.status.state == "load_failed"
    assert service.status.reason == "lightgbm_unavailable"
    assert service.served is None


def test_files_changed_after_startup_do_not_change_status_or_identity(
    promoted_pair: tuple[Path, Path],
) -> None:
    candidates, calibration = promoted_pair
    service = ModelService.from_environ(environment(promoted_pair))
    status, identity = service.status, service.status.identity
    assert status.state == "ready_promoted"

    (candidates / MODEL_FILES["classifier"]).write_text("tampered")
    (calibration / CALIBRATION).unlink()

    assert service.status == status
    assert service.status.identity == identity
    assert service.status.state == "ready_promoted"
    assert service.served is not None
    assert service.served.identity == identity


@pytest.mark.parametrize("value", ["yes", "0", "", "false", "2", "on"])
def test_flag_values_other_than_one_or_true_do_not_enable(
    value: str, blocked_pair: tuple[Path, Path]
) -> None:
    service = ModelService.from_environ(environment(blocked_pair, **{ALLOW_UNPROMOTED: value}))
    assert service.status.state == "not_promoted_refused"
    assert service.served is None
