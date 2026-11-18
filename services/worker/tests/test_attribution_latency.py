"""The step-completion-to-visible-attribution latency measurement, over a stub transport.

No network and no real sleeps: the stub answers the live API's routes in memory, and a fake
clock advances only when the measurement sleeps. Every run, step and attribution here is a
synthetic fixture, never recorded evidence.
"""

import json
import math
import tomllib
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest
from agentwatch_attribution.rules import load_rules
from agentwatch_worker.attribution_latency import (
    LIMIT_P95_SECONDS,
    measure,
    p95_within_limit,
    summarize,
)
from agentwatch_worker.attribution_model import ModelService
from agentwatch_worker.attribution_store import AttributionStore
from fastapi.testclient import TestClient
from pair_fixtures import write_synthetic_pair

ROOT = Path(__file__).resolve().parents[3]
SERVING_KEY = "c" * 64
TERMINAL_RECEIVED_AT = "2026-11-18T00:00:00.000000Z"
COMMITTED_AT = "2026-11-18T00:00:00.040000Z"
REVISION = {"git_sha": "0" * 40, "dirty": False}


class FakeClock:
    """A monotonic clock that only moves when the measurement sleeps, plus a small step per read
    so request ordering is visible."""

    def __init__(self, read_step_seconds: float = 0.001) -> None:
        self.now = 100.0
        self.read_step_seconds = read_step_seconds
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        self.now += self.read_step_seconds
        return self.now

    def sleep(self, seconds: float) -> None:
        assert seconds >= 0
        self.sleeps.append(seconds)
        self.now += seconds


def model_status(state: str = "ready_not_promoted", not_promoted: bool | None = True) -> dict:
    identity = None
    if not_promoted is not None:
        identity = {
            "training_version": "training-v1+synthetic",
            "calibration_version": "calibration-v1+synthetic",
            "classifier_sha256": "1" * 64,
            "regressor_sha256": "2" * 64,
            "preprocessing_sha256": "3" * 64,
            "calibration_sha256": "4" * 64,
            "temperature": 0.9,
            "abstention_threshold": 0.6,
            "class_order": ["cpu_sandbox", "inference", "retrieval", "none"],
            "blocks_promotion": not_promoted,
        }
    return {
        "state": state,
        "reason": None,
        "identity": identity,
        "not_promoted": not_promoted,
        "serving_key": SERVING_KEY,
    }


class StubLiveApi:
    """Answers the routes `measure` calls. A terminal step becomes visible on the poll after it
    arrives unless its id is in `hidden`."""

    def __init__(self, hidden: frozenset[str] = frozenset(), status: dict | None = None) -> None:
        self.run_id = str(uuid4())
        self.hidden = hidden
        self.status = status or model_status()
        self.terminal: list[str] = []
        self.rows: list[dict] = []
        self.requests: list[tuple[str, str]] = []
        self.states: list[str] = []

    def __call__(self, method: str, path: str, payload: bytes | None) -> bytes:
        self.requests.append((method, path))
        route = path.split("?", 1)[0]
        if (method, route) == ("POST", "/v1/live/runs"):
            return self.reply({"run_id": self.run_id, "created": True})
        if (method, route) == ("GET", "/v1/live/attribution/model"):
            return self.reply(self.status)
        run_path = f"/v1/live/runs/{self.run_id}"
        if (method, route) == ("POST", f"{run_path}/state"):
            change = json.loads(payload or b"{}")
            self.states.append(change["state"])
            revision = change["expected_revision"] + 1
            return self.reply({"run_id": self.run_id, "revision": revision, "changed": True})
        if (method, route) == ("POST", f"{run_path}/records"):
            return self.reply(self.ingest(json.loads(payload or b"{}")["records"]))
        if (method, route) == ("GET", f"{run_path}/attributions"):
            return self.reply(self.page(int(path.split("after=", 1)[1].split("&", 1)[0])))
        raise AssertionError(f"unexpected request {method} {path}")

    @staticmethod
    def reply(body: dict) -> bytes:
        return json.dumps(body).encode()

    def ingest(self, records: list[dict]) -> dict:
        acknowledgements = []
        for record in records:
            if record["record_type"] == "step_event":
                acknowledgements.append({"record_id": record["event_id"], "disposition": "applied"})
                if record["state"] == "completed":
                    self.terminal.append(record["step_id"])
            else:
                acknowledgements.append(
                    {"record_id": record["sample_id"], "disposition": "applied"}
                )
        return {
            "run_id": self.run_id,
            "added": len(records),
            "acknowledgements": acknowledgements,
            "gaps": [],
        }

    def page(self, after: int) -> dict:
        written = {row["record"]["step_id"] for row in self.rows}
        for step_id in self.terminal:
            if step_id not in written and step_id not in self.hidden:
                self.rows.append(
                    {
                        "sequence": len(self.rows) + 1,
                        "serving_key": SERVING_KEY,
                        "committed_at": COMMITTED_AT,
                        "step_terminal_received_at": TERMINAL_RECEIVED_AT,
                        "evidence_changed": False,
                        "record": {
                            "run_id": self.run_id,
                            "step_id": step_id,
                            "model": {"outcome": "predicted"},
                        },
                    }
                )
        return {
            "run_id": self.run_id,
            "model": self.status,
            "attributions": [row for row in self.rows if row["sequence"] > after],
            "next_cursor": None,
        }


def run(stub: StubLiveApi, clock: FakeClock, steps: int = 3, **changes: float) -> dict:
    settings = {
        "poll_interval_seconds": 1.0,
        # Fake time: the sampling sleeps advance the fake clock and never block.
        "step_duration_seconds": 0.004,
        "visible_timeout_seconds": 3.0,
    } | changes
    return measure(
        "http://stub.invalid",
        steps,
        transport=stub,
        clock=clock,
        sleep=clock.sleep,
        revision=lambda: dict(REVISION),
        **settings,
    )


def keys(value: object) -> Iterator[str]:
    if isinstance(value, dict):
        for key, nested in value.items():
            yield key
            yield from keys(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from keys(nested)


def test_summarize_uses_nearest_rank_percentiles() -> None:
    latencies = [float(value) for value in range(20, 0, -1)]

    assert summarize(latencies) == {"count": 20, "p50": 10.0, "p95": 19.0, "max": 20.0}
    # Nearest rank picks an observed value, never an interpolated one: ceil(0.95 * 3) = 3.
    assert summarize([0.5, 2.0, 1.0]) == {"count": 3, "p50": 1.0, "p95": 2.0, "max": 2.0}
    assert summarize([4.25]) == {"count": 1, "p50": 4.25, "p95": 4.25, "max": 4.25}


def test_summarize_of_nothing_is_none_not_zero() -> None:
    assert summarize([]) == {"count": 0, "p50": None, "p95": None, "max": None}


def test_summarize_rejects_non_finite_latencies() -> None:
    for invalid in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError):
            summarize([1.0, invalid])


def test_unseen_steps_are_counted_and_make_the_gate_inconclusive() -> None:
    stub = StubLiveApi(hidden=frozenset({"latency-step-0001"}))
    clock = FakeClock()

    report = run(stub, clock, steps=3)

    never = report["steps_never_visible"]
    assert never == {"count": 1, "step_ids": ["latency-step-0001"]}
    # The unseen step is a failure, not a dropped sample: two visible, three measured.
    assert report["client_latency_seconds"]["count"] == 2
    assert report["steps"] == 3
    assert report["p95_within_limit"] is None
    unseen = [step for step in report["per_step"] if step["step_id"] == "latency-step-0001"]
    assert unseen[0]["client_latency_seconds"] is None
    assert unseen[0]["visible"] is False
    # It was polled until the declared timeout, one interval apart, then given up.
    assert unseen[0]["polls"] == 3
    # Only fake sleeps happen, none longer than one poll interval.
    assert all(seconds <= 1.0 for seconds in clock.sleeps)
    # The run was closed after the steps.
    assert stub.requests[-1] == ("POST", f"/v1/live/runs/{stub.run_id}/state")
    assert stub.states == ["running", "completed"]


def test_visible_steps_are_timed_from_the_terminal_request_to_the_poll_that_shows_them() -> None:
    stub = StubLiveApi()
    clock = FakeClock()

    report = run(stub, clock, steps=2, poll_interval_seconds=1.0)

    latencies = [step["client_latency_seconds"] for step in report["per_step"]]
    # The first poll is sent one interval after the terminal request starts (a poller that
    # had just polled), so each latency includes the full interval.
    assert all(1.0 <= latency < 1.1 for latency in latencies)
    assert report["client_latency_seconds"]["count"] == 2
    assert report["server_side_seconds"] == {"count": 2, "p50": 0.04, "p95": 0.04, "max": 0.04}
    assert report["steps_never_visible"] == {"count": 0, "step_ids": []}
    assert report["p95_within_limit"] is True
    assert report["model_outcomes"] == {"predicted": 2}


def test_report_names_workload_as_synthetic_and_never_claims_promotion() -> None:
    stub = StubLiveApi(status=model_status("ready_not_promoted", not_promoted=True))

    report = run(stub, FakeClock(), steps=1)

    assert report["synthetic_workload"] is True
    assert "synthetic" in report["workload"]["description"]
    assert report["model_status"]["state"] == "ready_not_promoted"
    assert report["model_status"]["not_promoted"] is True
    assert report["limit_p95_seconds"] == 15.0
    assert "does not change any promotion state" in report["note"]
    claims = {"promoted", "promotion_passed", "gates_passed", "promotion_ready"}
    assert claims.isdisjoint(keys(report))
    text = json.dumps(report).lower()
    for wording in ("model is promoted", "promotion passed", "good model", "ready for promotion"):
        assert wording not in text
    assert report["code_revision"] == REVISION
    for timestamp in (report["started_at"], report["finished_at"]):
        assert timestamp.endswith("+00:00")
    assert {"platform", "python", "cpu_count"} <= set(report["machine"])


def test_p95_within_limit_true_and_false_cases() -> None:
    assert LIMIT_P95_SECONDS == 15.0
    assert p95_within_limit(summarize([1.0, 15.0]), never_visible=0) is True
    # Rank 19 of 20 is 15.0 exactly: one slower step in 20 still passes, two do not.
    assert p95_within_limit(summarize([15.0] * 19 + [99.0]), never_visible=0) is True
    assert p95_within_limit(summarize([15.0] * 18 + [15.000001] * 2), never_visible=0) is False
    assert p95_within_limit(summarize([15.000001]), never_visible=0) is False
    # Inconclusive: nothing visible, or any step that never became visible.
    assert p95_within_limit(summarize([]), never_visible=0) is None
    assert p95_within_limit(summarize([]), never_visible=4) is None
    assert p95_within_limit(summarize([1.0, 2.0]), never_visible=1) is None


def test_declared_limit_matches_the_evaluation_contract() -> None:
    contract = tomllib.loads((ROOT / "pipelines/model/evaluation-contract.toml").read_text())
    assert contract["version"] == "eval-contract-v3"
    assert contract["overhead"]["max_attribution_latency_p95_seconds"] == LIMIT_P95_SECONDS


@pytest.mark.parametrize(
    "changes",
    [
        {"steps": 0},
        {"steps": 1001},
        {"poll_interval_seconds": 0.0},
        {"poll_interval_seconds": math.inf},
        {"step_duration_seconds": -1.0},
        {"visible_timeout_seconds": 0.0},
    ],
)
def test_invalid_workload_settings_are_rejected_before_any_request(changes: dict) -> None:
    stub = StubLiveApi()
    steps = int(changes.pop("steps", 1))
    with pytest.raises(ValueError):
        run(stub, FakeClock(), steps=steps, **changes)
    assert stub.requests == []


def test_a_failed_request_cancels_the_run_and_propagates() -> None:
    stub = StubLiveApi()
    ingests: list[int] = []

    def failing(method: str, path: str, payload: bytes | None) -> bytes:
        if path.endswith("/records"):
            ingests.append(1)
            if len(ingests) == 3:  # The first step's terminal record.
                raise ConnectionRefusedError("worker stopped")
        return stub(method, path, payload)

    clock = FakeClock()
    with pytest.raises(ConnectionRefusedError):
        measure(
            "http://stub.invalid",
            2,
            poll_interval_seconds=1.0,
            step_duration_seconds=0.004,
            visible_timeout_seconds=3.0,
            transport=failing,
            clock=clock,
            sleep=clock.sleep,
            revision=lambda: dict(REVISION),
        )
    assert stub.requests[-1] == ("POST", f"/v1/live/runs/{stub.run_id}/state")
    assert stub.states == ["running", "cancelled"]


def test_measurement_drives_the_real_live_api(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The records, commands and reads `measure` sends are accepted by the worker's own API.

    TestClient runs the background attribution pass before a POST returns, so this checks
    shapes and bookkeeping, not latency. The model pair is the synthetic test pair.
    """
    candidates, calibration = write_synthetic_pair(tmp_path, blocks_promotion=True)
    model = ModelService.from_environ(
        {
            "AGENTWATCH_MODEL_CANDIDATES_DIR": str(candidates),
            "AGENTWATCH_MODEL_CALIBRATION_DIR": str(calibration),
            "AGENTWATCH_ALLOW_UNPROMOTED_MODEL": "1",
        }
    )
    store = AttributionStore(client.app.state.store, load_rules(), model)
    monkeypatch.setattr(client.app.state, "attribution", store)

    def transport(method: str, path: str, payload: bytes | None) -> bytes:
        headers = {} if payload is None else {"Content-Type": "application/json"}
        response = client.request(method, path, content=payload, headers=headers)
        assert response.status_code in {200, 201}, (path, response.text)
        return response.content

    clock = FakeClock()
    report = measure(
        "http://testserver",
        3,
        poll_interval_seconds=2.0,
        step_duration_seconds=0.004,
        visible_timeout_seconds=30.0,
        transport=transport,
        clock=clock,
        sleep=clock.sleep,
        revision=lambda: dict(REVISION),
    )

    assert report["steps_never_visible"] == {"count": 0, "step_ids": []}
    assert report["client_latency_seconds"]["count"] == 3
    assert report["p95_within_limit"] is True
    assert report["model_status"]["state"] == "ready_not_promoted"
    assert report["model_states_seen"] == ["ready_not_promoted"]
    assert report["serving_key_changed"] is False
    assert set(report["model_outcomes"]) <= {"predicted", "abstained"}
    assert sum(report["model_outcomes"].values()) == 3
    assert all(step["terminal_disposition"] == "applied" for step in report["per_step"])
    assert all(step["server_side_seconds"] >= 0 for step in report["per_step"])
    page = client.get(f"/v1/live/runs/{report['run_id']}/attributions").json()
    # The samples sent inside each step's window are the stored feature row's sources.
    for item in page["attributions"]:
        assert len(item["record"]["feature_row"]["source_sample_ids"]) == 8
        assert "telemetry_coverage" not in item["record"]["missing_inputs"]
    run = client.get(f"/v1/live/runs/{report['run_id']}").json()
    assert run["state"] == "completed"
    assert run["registration"]["task_id"] == "synthetic-latency-v1"
