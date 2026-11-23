"""Admission overhead measurement (admission-overhead-v1): alternating requests without and with
runner-process admission, under unchanged limits.

The model server and the policy service are test doubles; nothing here is recorded evidence or
a measured overhead.
"""

import threading
from urllib.error import URLError

from agent_test_doubles import FakePolicySource, chat_response, policy_version
from agentwatch_agent.admission_overhead import CONDITIONS, PROTOCOL, measure_admission_overhead
from agentwatch_agent.delivery import RetryPolicy
from agentwatch_agent.tasks import load_task

TASK = load_task("fix-mean-empty")
NO_RETRY = RetryPolicy(attempts=1, initial_backoff_seconds=0, max_backoff_seconds=0)


class CountingModel:
    """Replies the same way every time and counts requests; never a real model."""

    def __init__(self):
        self.requests = 0
        self.lock = threading.Lock()

    def chat(self, payload):
        with self.lock:
            self.requests += 1
        return chat_response("I could not find the defect.", total_duration=2_000_000)


def measure(source, model=None, **overrides):
    model = model or CountingModel()
    settings = {
        "inference_service_id": "ollama",
        "repetitions": 3,
        "warmup": 2,
        "policy_poll_seconds": 0.01,
        "retry": NO_RETRY,
        "phase_seed": 5,
        # No untimed phase wait by default; the phase test below uses a real one.
        "pause": lambda _seconds: None,
    } | overrides
    return measure_admission_overhead(TASK, "test-model", model.chat, source, **settings), model


def test_requests_alternate_and_each_admitted_one_holds_a_permit():
    report, model = measure(FakePolicySource(policy_version(1, inference=2)))
    assert report["status"] == "completed" and report["error"] is None
    assert report["protocol"]["version"] == PROTOCOL["version"] == "admission-overhead-v1"
    assert model.requests == 2 + 6
    assert [m["condition"] for m in report["measurements"]] == list(CONDITIONS) * 3
    assert [m["index"] for m in report["measurements"]] == list(range(6))
    assert len(report["warmups"]) == 2
    # The limit applies to this runner's model server, so the permits enforced it.
    assert report["policy"] == {
        "version": 1,
        "limits": {"inference_requests": 2},
        "inference_service": "ollama",
        "enforced": True,
        "not_enforced_reason": None,
    }
    assert len(report["admissions"]) == 3
    for admission in report["admissions"]:
        assert admission["outcome"] == "admitted" and admission["released"] is True
        assert admission["limit"] == 2 and admission["policy_version"] == 1
    assert report["controller"]["open_permits"] == 0
    assert len(report["policy_watchers"]) == 3
    assert all(watcher["stopped_by_error"] is None for watcher in report["policy_watchers"])
    assert report["conditions"]["admission_on"]["wall_seconds"]["count"] == 3
    assert report["median_wall_difference_fraction"] is not None


def test_the_baseline_policy_still_runs_the_admission_path():
    report, _model = measure(FakePolicySource(policy_version(0)))
    assert report["status"] == "completed"
    assert report["policy"]["limits"] == {} and report["policy"]["enforced"] is False
    assert all(admission["outcome"] == "admitted" for admission in report["admissions"])


class ChangingSource(FakePolicySource):
    """Serves version 1, then version 2 from the given read on: someone changed the policy."""

    def __init__(self, change_on_read):
        super().__init__(policy_version(1, inference=2))
        self.change_on_read = change_on_read
        self.changed = policy_version(2, inference=4)  # one stored version, read many times

    def get_policy(self, target="coding-agent"):
        policy = super().get_policy(target)
        return self.changed if self.reads >= self.change_on_read else policy


def test_a_policy_change_fails_the_measurement_and_keeps_what_was_measured():
    # Requests here are instant, so no watcher polls: the final read must catch the change.
    report, _model = measure(ChangingSource(change_on_read=2), policy_poll_seconds=60)
    assert report["status"] == "failed" and "version 1 to 2" in report["error"]
    assert "PolicyChanged" in report["error"]
    # Every request measured before the change is detected is kept.
    assert len(report["measurements"]) == 6


def test_a_change_the_watcher_sees_fails_at_once():
    report, _model = measure(
        ChangingSource(change_on_read=2), SlowModel(0.05), policy_poll_seconds=0.005
    )
    assert report["status"] == "failed" and "PolicyChanged" in report["error"]
    assert len(report["measurements"]) < 6


def test_an_unreadable_policy_sends_no_request():
    source = FakePolicySource(policy_version(1))
    source.failures = [URLError("policy service down")]
    report, model = measure(source)
    assert report["status"] == "failed" and "URLError" in report["error"]
    assert model.requests == 0 and report["policy"] is None and report["measurements"] == []


class SlowModel(CountingModel):
    """Each request takes `seconds`, so watcher polls can fall inside it."""

    def __init__(self, seconds):
        super().__init__()
        self.seconds = seconds

    def chat(self, payload):
        threading.Event().wait(self.seconds)
        return super().chat(payload)


def test_each_admitted_request_starts_at_a_recorded_random_phase_of_the_poll_cycle():
    waited = []
    report, _model = measure(
        FakePolicySource(policy_version(1, inference=2)),
        policy_poll_seconds=0.5,
        phase_seed=11,
        pause=waited.append,
    )
    admitted = [m for m in report["measurements"] if m["condition"] == "admission_on"]
    phases = [m["poll_phase_seconds"] for m in admitted]
    # The untimed wait before each timed request is its recorded phase, within one interval.
    assert waited == phases and all(0 <= phase < 0.5 for phase in phases)
    assert len(set(phases)) == 3
    assert report["poll_phase_seed"] == 11 and report["policy_poll_seconds"] == 0.5
    again, _model = measure(
        FakePolicySource(policy_version(1, inference=2)),
        policy_poll_seconds=0.5,
        phase_seed=11,
    )
    replayed = [m["poll_phase_seconds"] for m in again["measurements"] if "poll_phase_seconds" in m]
    assert replayed == phases  # the seed reproduces the phases
    assert all("poll_phase_seconds" not in m for m in report["measurements"][0::2])


def test_polls_that_fall_inside_a_timed_request_are_counted():
    # Requests of 0.05 s against a 0.01 s poll: every admitted request overlaps a poll.
    report, _model = measure(
        FakePolicySource(policy_version(1, inference=2)),
        SlowModel(0.05),
        pause=threading.Event().wait,
    )
    admitted = [m for m in report["measurements"] if m["condition"] == "admission_on"]
    assert all(m["polls_during_request"] >= 1 for m in admitted)
    assert report["conditions"]["admission_on"]["requests_overlapping_a_poll"] == 3


def test_a_policy_without_an_inference_entry_is_saved_as_failed():
    # `apply` refuses it; the report must still be built, never raise on the missing entry.
    complete = policy_version(1, inference=2)
    partial = complete.model_copy(
        update={
            "controlled_services": {
                control: service
                for control, service in complete.controlled_services.items()
                if control != "inference_requests"
            }
        }
    )
    report, model = measure(FakePolicySource(partial))
    assert report["status"] == "failed" and report["error"]
    assert report["policy"]["inference_service"] is None
    assert model.requests == 0
