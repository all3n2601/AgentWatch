"""Versioned manual concurrency policy contract (O02)."""

from datetime import datetime
from typing import Annotated, Any, Literal, Self
from uuid import UUID

from pydantic import Field, SerializerFunctionWrapHandler, model_serializer, model_validator
from pydantic.json_schema import SkipJsonSchema

from agentwatch_sdk.live import Identifier, LiveModel, Reason, ResourceClass

PolicyTarget = Literal["coding-agent"]
RUNNER_SERVICE_ID = "agentwatch-agent-runner"
SUPPORTED_TARGETS: dict[str, dict[str, Any]] = {
    "coding-agent": {
        "controlled_services": {
            "inference_requests": {"service_id": "ollama", "resource_class": "inference"},
            "test_jobs": {"service_id": RUNNER_SERVICE_ID, "resource_class": "cpu_sandbox"},
        },
        "enforcement_scope": "runner_process",
        "runner_service_id": RUNNER_SERVICE_ID,
    }
}

Limit = Annotated[int, Field(strict=True, ge=1, le=1024)]
Version = Annotated[int, Field(strict=True, ge=0)]
PolicyAction = Literal["set", "disable", "restore"]


class PolicyLimits(LiveModel):
    """Per-control limits; an omitted control means no limit."""

    # None is only the in-memory "absent" marker: input rejects null and output drops it, so
    # the published schema advertises integers only.
    inference_requests: Limit | SkipJsonSchema[None] = None
    test_jobs: Limit | SkipJsonSchema[None] = None

    @model_validator(mode="before")
    @classmethod
    def reject_null_limits(cls, data: Any) -> Any:
        if isinstance(data, dict) and any(value is None for value in data.values()):
            raise ValueError("Omit a limit for no limit; null is not accepted")
        return data

    @model_serializer(mode="wrap")
    def omit_absent_limits(self, handler: SerializerFunctionWrapHandler):
        # The wire contract mirrors input: "no limit" is an absent key, never null. The return
        # type is left unannotated so the published schema keeps the declared properties.
        return {key: value for key, value in handler(self).items() if value is not None}


class PolicyChange(LiveModel):
    request_id: UUID
    expected_version: Version
    action: PolicyAction
    limits: PolicyLimits | None = None
    restored_from: Version | None = None
    operator: Identifier
    reason: Reason

    @model_validator(mode="after")
    def check_action_shape(self) -> Self:
        has_limits = self.limits is not None and bool(self.limits.model_dump(exclude_none=True))
        if self.action == "set":
            if not has_limits:
                raise ValueError("set needs at least one limit; use disable to remove all limits")
            if self.restored_from is not None:
                raise ValueError("set does not accept restored_from")
        elif self.action == "disable":
            if self.limits is not None or self.restored_from is not None:
                raise ValueError("disable accepts neither limits nor restored_from")
        else:
            if self.restored_from is None:
                raise ValueError("restore needs restored_from")
            if self.limits is not None:
                raise ValueError("restore does not accept limits")
        return self


class ControlledService(LiveModel):
    service_id: str
    resource_class: ResourceClass


class PolicyVersion(LiveModel):
    target: str
    version: int
    action: Literal["baseline", "set", "disable", "restore"]
    limits: PolicyLimits
    restored_from: int | None
    operator: str | None
    reason: str | None
    request_id: UUID | None
    created_at: datetime | None
    controlled_services: dict[str, ControlledService]
    enforcement_scope: Literal["runner_process"]


class PolicyChangeResult(LiveModel):
    created: bool
    policy: PolicyVersion


class PolicyPage(LiveModel):
    versions: list[PolicyVersion]
    next_before: int | None


class RunPolicy(LiveModel):
    """Policy versions linked to one live run.

    `versions` lists, oldest first, every policy version that was active while the run was
    open (queued or running): the version active at registration plus each later change.
    This records what was configured, not evidence that any limit was enforced; O02 stores
    policy only. `applicable` is true when the run registered the target's runner service
    (a coding-agent run). `versions` is empty for non-applicable runs and for runs registered
    before policy tracking existed.
    """

    run_id: UUID
    target: str
    applicable: bool
    versions: list[PolicyVersion]
