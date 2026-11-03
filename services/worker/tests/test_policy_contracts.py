"""SDK concurrency policy contract: strict limits and action shapes."""

from uuid import uuid4

import pytest
from agentwatch_sdk.policy import PolicyChange, PolicyLimits
from pydantic import ValidationError


def change(**overrides):
    record = {
        "request_id": str(uuid4()),
        "expected_version": 0,
        "action": "set",
        "limits": {"test_jobs": 2},
        "operator": "alice",
        "reason": "cap runner contention",
    }
    record.update(overrides)
    return {key: value for key, value in record.items() if value is not ...}


def test_limits_accept_positive_integers_and_absence():
    assert PolicyLimits.model_validate({"test_jobs": 2}).model_dump(exclude_none=True) == {
        "test_jobs": 2
    }
    assert PolicyLimits.model_validate({}).model_dump(exclude_none=True) == {}


@pytest.mark.parametrize(
    "limits",
    [
        {"test_jobs": 0},
        {"test_jobs": -1},
        {"test_jobs": None},
        {"inference_requests": None},
        {"test_jobs": 2.0},
        {"test_jobs": True},
        {"test_jobs": "2"},
        {"test_jobs": 1025},
        {"gpu": 1},
    ],
)
def test_limits_reject_invalid(limits):
    with pytest.raises(ValidationError):
        PolicyLimits.model_validate(limits)


@pytest.mark.parametrize(
    "record",
    [
        change(limits=...),
        change(limits={}),
        change(restored_from=1),
        change(action="disable"),
        change(action="disable", limits=..., restored_from=1),
        change(action="restore", limits=...),
        change(action="restore", limits={"test_jobs": 2}, restored_from=1),
    ],
)
def test_change_rejects_wrong_action_shape(record):
    with pytest.raises(ValidationError):
        PolicyChange.model_validate(record)


def test_set_without_limits_points_to_disable():
    with pytest.raises(ValidationError, match="disable"):
        PolicyChange.model_validate(change(limits={}))


@pytest.mark.parametrize(
    "record",
    [
        change(),
        change(action="disable", limits=...),
        change(action="restore", limits=..., restored_from=1),
    ],
)
def test_change_accepts_each_valid_action_shape(record):
    assert PolicyChange.model_validate(record).action == record["action"]


@pytest.mark.parametrize("field,value", [("operator", " "), ("reason", "")])
def test_operator_and_reason_reject_blank(field, value):
    with pytest.raises(ValidationError):
        PolicyChange.model_validate(change(**{field: value}))
