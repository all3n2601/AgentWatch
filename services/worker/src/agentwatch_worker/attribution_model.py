"""Which attribution model pair the worker serves, and why it serves none.

The pair is chosen by environment, loaded once when the service is built, and never re-read: the
status and identity reported here are those of the bytes that were checked and loaded, even if
the files change later. A pair whose calibration `blocks_promotion` (its gates did not pass) is
served only when AGENTWATCH_ALLOW_UNPROMOTED_MODEL is `1` or `true`, and is then labelled not
promoted. When a pair is not served, `status.reason` and `status.identity` say why.

The outcome is logged once at startup. A load failure is logged with the loader's whole message,
which names the local artifact file and field that failed; the status keeps only the check code.
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from agentwatch_attribution.served_model import (
    ModelIdentity,
    ServedModel,
    ServedModelError,
    load_served_model,
)
from agentwatch_sdk.attribution import ModelState

CANDIDATES_DIR_VARIABLE = "AGENTWATCH_MODEL_CANDIDATES_DIR"
CALIBRATION_DIR_VARIABLE = "AGENTWATCH_MODEL_CALIBRATION_DIR"
ALLOW_UNPROMOTED_VARIABLE = "AGENTWATCH_ALLOW_UNPROMOTED_MODEL"
ALLOW_UNPROMOTED_VALUES = frozenset({"1", "true"})
# The same code the loader uses for an artifact it cannot read.
UNREADABLE = "artifact_unreadable"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelStatus:
    """`reason` is a CHECK_CODES value when `state` is load_failed, otherwise None. `identity`
    is set whenever a pair was loaded, including a refused one."""

    state: ModelState
    reason: str | None = None
    identity: ModelIdentity | None = None


class ModelService:
    """The configured model pair, if any. `served` is set only in the two ready states."""

    def __init__(self, status: ModelStatus, served: ServedModel | None = None) -> None:
        self.status = status
        self.served = served

    @classmethod
    def from_environ(cls, environ: Mapping[str, str]) -> "ModelService":
        candidates = environ.get(CANDIDATES_DIR_VARIABLE, "")
        calibration = environ.get(CALIBRATION_DIR_VARIABLE, "")
        if not candidates and not calibration:
            logger.info("Attribution model state=not_configured: serving the rules verdict only")
            return cls(ModelStatus("not_configured"))
        if not candidates or not calibration:
            set_variable, unset_variable = (
                (CANDIDATES_DIR_VARIABLE, CALIBRATION_DIR_VARIABLE)
                if candidates
                else (CALIBRATION_DIR_VARIABLE, CANDIDATES_DIR_VARIABLE)
            )
            logger.warning(
                "Attribution model state=load_failed reason=%s: %s is set but %s is not",
                UNREADABLE,
                set_variable,
                unset_variable,
            )
            return cls(ModelStatus("load_failed", reason=UNREADABLE))
        try:
            model = load_served_model(Path(candidates), Path(calibration))
        except ServedModelError as error:
            # The message names local configuration paths and artifact fields, never producer data.
            logger.warning("Attribution model state=load_failed reason=%s: %s", error.check, error)
            return cls(ModelStatus("load_failed", reason=error.check))
        identity = model.identity
        if not identity.blocks_promotion:
            return cls(log_loaded(ModelStatus("ready_promoted", identity=identity)), model)
        if allows_unpromoted(environ):
            return cls(log_loaded(ModelStatus("ready_not_promoted", identity=identity)), model)
        return cls(log_loaded(ModelStatus("not_promoted_refused", identity=identity)))


def allows_unpromoted(environ: Mapping[str, str]) -> bool:
    return environ.get(ALLOW_UNPROMOTED_VARIABLE, "").strip().lower() in ALLOW_UNPROMOTED_VALUES


def log_loaded(status: ModelStatus) -> ModelStatus:
    """Log the state and versions of a pair that loaded, served or refused; return `status`."""
    identity = status.identity
    if identity is None:
        raise ValueError(f"A loaded pair's status needs its identity, got state {status.state}")
    if status.state == "not_promoted_refused":
        logger.warning(
            "Attribution model state=not_promoted_refused training_version=%s "
            "calibration_version=%s: the pair blocks promotion and %s is not set; serving the "
            "rules verdict only",
            identity.training_version,
            identity.calibration_version,
            ALLOW_UNPROMOTED_VARIABLE,
        )
    else:
        logger.info(
            "Attribution model state=%s training_version=%s calibration_version=%s not_promoted=%s",
            status.state,
            identity.training_version,
            identity.calibration_version,
            identity.blocks_promotion,
        )
    return status
