"""SDK and published API schemas must describe the final live router."""

import json
from pathlib import Path
from uuid import uuid4

import pytest
from agentwatch_sdk.client import LiveClient
from agentwatch_sdk.live import LiveBatch, ResourceSample, StepEvent
from agentwatch_worker.export_live_contracts import openapi_document, typescript_document
from pydantic import ValidationError
from test_live_api import event, sample

ROOT = Path(__file__).resolve().parents[3]


def test_published_live_contracts_match_router():
    document = openapi_document()
    assert json.loads((ROOT / "api/openapi/live.json").read_text()) == document
    assert (ROOT / "packages/sdk-typescript/src/live.ts").read_text() == typescript_document(
        document
    )


def test_nonfinite_and_strict_producer_records():
    record = event(str(uuid4()), 2, "completed", cpu_seconds=float("inf"), cpu_scope="process")
    with pytest.raises(ValidationError):
        StepEvent.model_validate(record)
    with pytest.raises(ValidationError):
        LiveBatch.model_validate({"records": [record]})


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), True, False])
def test_sdk_rejects_unbounded_timeouts(timeout):
    with pytest.raises(ValueError):
        LiveClient("http://localhost:8090", timeout)


def test_sdk_validates_response_and_propagates_transport_failure(monkeypatch):
    from io import BytesIO
    from urllib.error import HTTPError

    from agentwatch_sdk import client as sdk_client
    from agentwatch_sdk.live import RunCreate

    registration = RunCreate.model_validate(
        {
            "contract_version": "live-contracts-v1",
            "request_id": str(uuid4()),
            "task_id": "fixture",
            "configuration_digest": "a" * 64,
            "clock_ids": ["clock"],
            "resources": [],
        }
    )
    run_id = str(uuid4())
    sent = []

    def response(request, timeout):
        sent.append((request.full_url, json.loads(request.data), timeout))
        return BytesIO(json.dumps({"run_id": run_id, "created": True}).encode())

    monkeypatch.setattr(sdk_client, "urlopen", response)
    sdk = LiveClient("http://localhost:8090", 2)
    assert str(sdk.create_run(registration).run_id) == run_id
    assert sent == [("http://localhost:8090/v1/live/runs", registration.model_dump(mode="json"), 2)]

    def malformed(request, timeout):
        return BytesIO(b'{"created":true}')

    monkeypatch.setattr(sdk_client, "urlopen", malformed)
    with pytest.raises(ValidationError):
        sdk.create_run(registration)

    def failure(request, timeout):
        raise HTTPError(request.full_url, 409, "Conflict", {}, None)

    monkeypatch.setattr(sdk_client, "urlopen", failure)
    with pytest.raises(HTTPError) as caught:
        sdk.create_run(registration)
    assert caught.value.code == 409


def test_sdk_rejects_path_injection_before_network():
    sdk = LiveClient("http://localhost:8090")
    with pytest.raises(ValueError):
        sdk.get_run("../other")


GPU_METRICS = {
    "gpu.nvml.utilization.gpu.v1": ("percent", 100),
    "gpu.nvml.utilization.memory.v1": ("percent", 100),
    "gpu.nvml.memory.used_bytes.v1": ("bytes", None),
    "gpu.nvml.power.usage_watts.v1": ("watts", None),
    "gpu.nvml.clock.sm_mhz.v1": ("megahertz", None),
}


def gpu_sample(metric, **changes):
    unit, _ = GPU_METRICS[metric]
    return (
        sample(str(uuid4()), resource_class="inference", metric=metric, unit=unit, value=0)
        | changes
    )


CPU_COUNTERS = (
    "cpu.psi.some.total_usec.v1",
    "cpu.cgroup.usage_usec.v1",
    "cpu.cgroup.psi.some.total_usec.v1",
)


@pytest.mark.parametrize("metric", CPU_COUNTERS)
def test_cpu_counters_are_nonnegative_microseconds_for_cpu_resources(metric):
    counter = {"metric": metric, "unit": "microseconds", "value": 499_359_020}
    assert ResourceSample.model_validate(sample(str(uuid4()), **counter)).value == 499_359_020
    with pytest.raises(ValidationError, match="outside registered bounds"):
        ResourceSample.model_validate(sample(str(uuid4()), **counter | {"value": -1}))
    with pytest.raises(ValidationError, match="class/unit mismatch"):
        ResourceSample.model_validate(sample(str(uuid4()), **counter | {"unit": "percent"}))
    with pytest.raises(ValidationError, match="class/unit mismatch"):
        ResourceSample.model_validate(
            sample(str(uuid4()), **counter | {"resource_class": "inference"})
        )


@pytest.mark.parametrize("metric", GPU_METRICS)
def test_gpu_metrics_are_registered_for_inference_resources(metric):
    ResourceSample.model_validate(gpu_sample(metric))
    unsupported = gpu_sample(metric, value=None, availability="unsupported", reason="No NVML")
    assert ResourceSample.model_validate(unsupported).value is None
    with pytest.raises(ValidationError, match="class/unit mismatch"):
        ResourceSample.model_validate(gpu_sample(metric, resource_class="cpu_sandbox"))
    with pytest.raises(ValidationError, match="class/unit mismatch"):
        ResourceSample.model_validate(gpu_sample(metric, unit="milliwatts"))
    with pytest.raises(ValidationError, match="outside registered bounds"):
        ResourceSample.model_validate(gpu_sample(metric, value=-1))


@pytest.mark.parametrize(
    "metric", [m for m, (_, maximum) in GPU_METRICS.items() if maximum is not None]
)
def test_gpu_percentages_are_bounded(metric):
    with pytest.raises(ValidationError, match="outside registered bounds"):
        ResourceSample.model_validate(gpu_sample(metric, value=100.5))
