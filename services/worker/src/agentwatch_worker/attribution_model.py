"""Which attribution model pair the worker serves, and why it serves none.

The pair is chosen by environment, loaded once when the service is built, and never re-read: the
status and identity reported here are those of the bytes that were checked and loaded, even if
the files change later. A pair whose calibration `blocks_promotion` (its gates did not pass) is
served only when AGENTWATCH_ALLOW_UNPROMOTED_MODEL is `1` or `true`, and is then labelled not
promoted. When a pair is not served, `status.reason` and `status.identity` say why.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from agentwatch_attribution.served_model import (
    ModelIdentity,
    ServedModel,
    ServedModelError,
    load_served_model,
)

CANDIDATES_DIR_VARIABLE = "AGENTWATCH_MODEL_CANDIDATES_DIR"
CALIBRATION_DIR_VARIABLE = "AGENTWATCH_MODEL_CALIBRATION_DIR"
ALLOW_UNPROMOTED_VARIABLE = "AGENTWATCH_ALLOW_UNPROMOTED_MODEL"
ALLOW_UNPROMOTED_VALUES = frozenset({"1", "true"})
# The same code the loader uses for an artifact it cannot read.
UNREADABLE = "artifact_unreadable"

ModelState = Literal[
    "not_configured",
    "not_promoted_refused",
    "ready_not_promoted",
    "ready_promoted",
    "load_failed",
]


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
            return cls(ModelStatus("not_configured"))
        if not candidates or not calibration:
            return cls(ModelStatus("load_failed", reason=UNREADABLE))
        try:
            model = load_served_model(Path(candidates), Path(calibration))
        except ServedModelError as error:
            return cls(ModelStatus("load_failed", reason=error.check))
        identity = model.identity
        if not identity.blocks_promotion:
            return cls(ModelStatus("ready_promoted", identity=identity), model)
        if allows_unpromoted(environ):
            return cls(ModelStatus("ready_not_promoted", identity=identity), model)
        return cls(ModelStatus("not_promoted_refused", identity=identity))


def allows_unpromoted(environ: Mapping[str, str]) -> bool:
    return environ.get(ALLOW_UNPROMOTED_VARIABLE, "").strip().lower() in ALLOW_UNPROMOTED_VALUES
