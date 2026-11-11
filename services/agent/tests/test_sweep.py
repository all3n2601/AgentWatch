"""The capacity-sweep driver with test doubles: stub batches, a scripted server and fake time.

Nothing here runs a model, a GPU or a real injector, and nothing sleeps. Batch records, run
folders, server readings and injector activity are synthetic fixtures, never sweep evidence.
"""

import json
import urllib.error
from dataclasses import dataclass
from pathlib import Path

import pytest
from agentwatch_agent import sweep
from agentwatch_agent.errors import RunNotStarted
from agentwatch_agent.replay import ReplayPlan
from agentwatch_agent.runner import VERIFICATION_FAILED
from agentwatch_agent.sweep import SWEEP_FILE, CapacitySweep
from agentwatch_agent.sweep_protocol import (
    CONFIGURATIONS,
    IDLE_WAIT_SECONDS,
    INJECTOR_BOUND_MARGIN_SECONDS,
    MODEL_ID,
    MODEL_REVISION,
    PACING_SECONDS,
    RAMP_LEVELS,
)
from agentwatch_agent.vllm_server import ServerState, ServerStateUnavailable
from agentwatch_injectors.base import BoundedInjector
from test_sweep_analysis import write_run

RUN_LIMIT_SECONDS = 900.0
SERVED = f"{MODEL_ID}@{MODEL_REVISION}"
# The label set vLLM v0.32 reported on 2026-10-27 (synthetic copy).
CACHE_CONFIG = {
    "block_size": "16",
    "enable_prefix_caching": "True",
    "engine": "0",
    "gpu_memory_utilization": "0.6",
    "kv_cache_max_concurrency": "1.80078125",
    "kv_cache_size_tokens": "7376",
    "num_gpu_blocks": "461",
}
SOURCE_TASKS = ("fix-mean-empty", "fix-window-count")


class FakeTime:
    """Monotonic seconds that only advance when the driver sleeps."""

    def __init__(self, events):
        self.now = 0.0
        self.events = events
        self.on_sleep = None  # called with the seconds slept, after the clock advances

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.events.append(("sleep", seconds))
        self.now += seconds
        if self.on_sleep is not None:
            self.on_sleep(seconds)


class FakeServer:
    """vLLM's state as the driver reads it: busy until a time, failing reads, a restart."""

    def __init__(self, time, cache_config=None):
        self.time = time
        self.start_time = 1_793_140_000.0
        self.cache_config = dict(cache_config or CACHE_CONFIG)
        self.requests = 0.0
        self.busy_until = None
        self.fail_at = set()  # read numbers (1-based) that raise
        self.reads = 0
        self.on_read = None  # called with the read number before the reading is returned

    def fail_read(self, after):
        """Make the read `after` reads from now fail (1 is the next read)."""
        self.fail_at.add(self.reads + after)

    def __call__(self):
        self.reads += 1
        if self.on_read is not None:
            self.on_read(self.reads)
        if self.reads in self.fail_at:
            raise ServerStateUnavailable("fixture: metrics unavailable")
        busy = self.busy_until is not None and self.time.now < self.busy_until
        return ServerState(
            start_time_seconds=self.start_time,
            cache_config=dict(self.cache_config),
            counters={
                "vllm:prefix_cache_queries_total": 10.0 * self.requests,
                "vllm:prefix_cache_hits_total": 5.0 * self.requests,
                "vllm:num_preemptions_total": 0.0,
                "vllm:request_success_total": self.requests,
            },
            running=1.0 if busy else 0.0,
            waiting=0.0,
        )


def passed_entry(index, run_dir_name):
    return {
        "index": index,
        "task_id": "fixture",
        "output_dir": f"runs/{run_dir_name}",
        "run_id": f"00000000-0000-0000-0000-{index:012d}",
        "state": "completed",
        "reason": None,
        "task_passed": True,
        "delivery_complete": True,
        "resend_needed": False,
        "records_refused": 0,
        "completion_refused": False,
        "not_started": None,
        "error": None,
    }


@dataclass
class FakeBatchRun:
    entry: dict
    error: BaseException | None = None

    def evidence(self):
        return self.entry


def default_wall_seconds(level):
    # Throughput doubles from N = 1 to N = 2 and is flat after: the knee is at N = 2.
    return 60.0 * max(1.0, level / 2)


class StubBatch:
    """Writes a fixed batch.json and synthetic run folders; hooks change one batch's outcome."""

    def __init__(self, harness, name, level, batch_dir, clock):
        self.harness = harness
        self.name = name
        self.level = level
        self.output_dir = batch_dir
        self.path = batch_dir / "batch.json"
        self.clock = clock
        self.batch_id = f"batch-{name}"
        self.cancel_requests = []
        self.ran = False
        self.runs = []  # like AgentBatch.runs: the runs, with whatever evidence they have

    def request_cancel_from_signal(self, source, reason):
        self.cancel_requests.append((source, reason))

    def run(self):
        harness = self.harness
        self.ran = True
        harness.events.append(("batch", self.name))
        # Which batches sweep.json held, and whether each was already finished.
        saved = read(harness.sweep_file)["batches"]
        harness.sweep_json_batches.append(["server" in entry for entry in saved])
        entries = []
        for index in range(1, self.level + 1):
            run_name = f"{index:02d}"
            write_run(self.output_dir / "runs" / run_name, duration_seconds=30.0)
            entries.append(passed_entry(index, run_name))
        self.runs = [FakeBatchRun(entry) for entry in entries]
        wall = default_wall_seconds(self.level)
        errors = [None] * len(entries)
        hook = harness.hooks.get(self.name)
        if hook is not None:
            wall, errors = hook(self, entries, wall) or (wall, errors)
        for run, error in zip(self.runs, errors, strict=True):
            run.error = error
        self.output_dir.mkdir(parents=True, exist_ok=True)
        document = {"batch_id": self.batch_id, "batch_wall_seconds": wall, "runs": entries}
        self.path.write_text(json.dumps(document))
        harness.server.requests += len(entries)
        return self.runs


class FakeInjector(BoundedInjector):
    """Test double: applies no load; reports one completed prefill request."""

    kind = "prefill"
    method = "test-double"
    resource_class = "inference"

    def __init__(self, clock, bound_seconds, confirm=True):
        super().__init__(clock, bound_seconds)
        self.confirm = confirm

    def bounds(self):
        return {}

    def target(self):
        return {}

    def _launch(self):
        pass

    def _halt(self):
        activity = {
            "report": "received",
            "stopped_by_backstop": False,
            "requests_failed": 0,
            "requests_completed": 1,
            "prompt_tokens_target": 2048,
            "prompt_tokens": {"count": 1, "min": 2083, "median": 2083, "max": 2083},
        }
        return activity, {"confirmed": self.confirm}


class Harness:
    def __init__(self, tmp_path, configuration="cfg-default", cache_config=None):
        self.events = []
        self.time = FakeTime(self.events)
        self.server = FakeServer(self.time, cache_config)
        self.hooks = {}
        self.batches = []
        self.calls = []
        self.injector_calls = []
        self.sweep_json_batches = []
        self.confirm_cleanup = True
        self.on_make_batch = {}  # batch name -> called inside make_batch
        self.on_make_injectors = None
        self.models = lambda: {"data": [{"id": SERVED, "root": MODEL_ID, "max_model_len": 4096}]}
        self.output_dir = tmp_path / "sweep"
        self.sweep_file = self.output_dir / SWEEP_FILE
        self.sweep = CapacitySweep(
            configuration=CONFIGURATIONS[configuration],
            sources=[plan(task) for task in SOURCE_TASKS],
            output_dir=self.output_dir,
            run_timeout_seconds=RUN_LIMIT_SECONDS,
            make_batch=self.make_batch,
            make_live_batch=self.make_live_batch,
            make_injectors=self.make_injectors,
            read_state=self.server,
            models=lambda: self.models(),
            sleep=self.time.sleep,
            monotonic=self.time.monotonic,
            host={"gpu": "fixture"},
            start_attempts=[{"gpu_memory_utilization": 0.6, "started": True}],
        )

    def _stub(self, name, level, batch_dir, clock):
        stub = StubBatch(self, name, level, batch_dir, clock)
        self.batches.append(stub)
        return stub

    def make_batch(self, batch, batch_dir, clock):
        self.calls.append((batch.phase, batch.level, batch.repeat, batch.sources))
        name = f"{batch.phase}-n{batch.level}-r{batch.repeat}"
        if name in self.on_make_batch:
            self.on_make_batch[name]()
        return self._stub(name, batch.level, batch_dir, clock)

    def make_live_batch(self, level, batch_dir, clock):
        self.calls.append(("live", level, None, None))
        repeat = sum(1 for call in self.calls if call[0] == "live") - 1
        return self._stub(f"live-n{level}-r{repeat}", level, batch_dir, clock)

    def make_injectors(self, clock, bound_seconds):
        self.injector_calls.append((clock, bound_seconds))
        if self.on_make_injectors is not None:
            self.on_make_injectors()
        return [FakeInjector(clock, bound_seconds, confirm=self.confirm_cleanup)]

    def run(self):
        return self.sweep.run()

    def phases_run(self):
        return [(c[0], c[1]) for c in self.calls]


def plan(task_id):
    return ReplayPlan(
        source_run_id=f"source-{task_id}",
        source_evidence_sha256="0" * 64,
        task_id=task_id,
        task_content_sha256="1" * 64,
        model=SERVED,
        chat_options={"temperature": 0},
        tools=(),
        turns=(),
        model_backend="vllm",
    )


def read(path: Path):
    return json.loads(path.read_text())


@pytest.fixture(autouse=True)
def no_injector_ramp(monkeypatch):
    # The declared 2 s ramp is a real wait inside InjectionSet; tests do not sleep.
    monkeypatch.setattr(sweep, "INJECTOR_RAMP_SECONDS", 0.0)


def set_runs(**fields):
    def hook(_stub, entries, wall):
        for entry in entries:
            entry.update(fields)
        return wall, [None] * len(entries)

    return hook


def expected_ramp(last_level=RAMP_LEVELS[-1]):
    calls = [("warmup", 1)]
    for level in RAMP_LEVELS:
        if level > last_level:
            break
        calls += [("ramp", level)] * (6 if level == 1 else 3)
    return calls


def post_ramp(capacity=2, injected=1):
    return [("live", capacity)] * 3 + [("injected", injected), ("control", injected)] * 3


def test_phases_run_in_declared_order(tmp_path):
    harness = Harness(tmp_path)
    record = harness.run()
    assert record["state"] == "completed" and record["ramp_end"] == "ceiling"
    assert harness.phases_run() == expected_ramp() + post_ramp(capacity=2, injected=1)
    # N = 1 runs three batches per source; sources alternate within larger batches.
    n1 = [c[3] for c in harness.calls if c[:2] == ("ramp", 1)]
    assert n1 == [(0,), (1,), (0,), (1,), (0,), (1,)]
    assert next(c[3] for c in harness.calls if c[:2] == ("ramp", 3)) == (0, 1, 0)
    decision = record["capacity_decision"]
    assert decision["capacity"] == 2 and decision["result"] == "measured"
    assert decision["first_failing_level"] == 3
    assert record["phases"]["live"] == {"status": "completed", "level": 2}
    assert record["phases"]["injected"] == {"status": "completed", "level": 1}
    # Each batch has its own clock domain, shared with that batch's injection set.
    clock_ids = [entry["clock_id"] for entry in record["batches"]]
    assert len(set(clock_ids)) == len(clock_ids)
    assert [entry["folder"] for entry in record["batches"][:2]] == [
        "batches/00-warmup-n1-r0",
        "batches/01-ramp-n1-r0",
    ]
    injected = [entry for entry in record["batches"] if entry["phase"] == "injected"]
    assert len(harness.injector_calls) == 3
    assert {bound for _clock, bound in harness.injector_calls} == {
        RUN_LIMIT_SECONDS + INJECTOR_BOUND_MARGIN_SECONDS
    }
    for entry, (clock, _bound) in zip(injected, harness.injector_calls, strict=True):
        assert entry["clock_id"] == clock.clock_id
        log = read(harness.output_dir / entry["injection_log"])
        assert log["clock_id"] == clock.clock_id and log["cleanup_confirmed"] is True
        assert entry["injection_active"] is True
    assert record["injection_logs"] == [entry["injection_log"] for entry in injected]
    assert all(e["injection_log"] is None for e in record["batches"] if e["phase"] == "control")
    # The warm-up is recorded, never analysed.
    warmup = record["batches"][0]
    assert warmup["analysed"] is False and "metrics" not in warmup
    assert record["level_summaries"][1]["batches"] == 6
    # Counter changes are stored per batch from the before and after readings.
    first_ramp = record["batches"][1]
    assert first_ramp["server"]["changes"]["request_successes"] == 1
    assert first_ramp["server"]["before"]["counters"]["vllm:request_success_total"] == 1.0
    assert record["held_out"] is True
    assert record["protocol_sha256"] == sweep.protocol_sha256(record["protocol"])
    assert record["start_attempts"] == [{"gpu_memory_utilization": 0.6, "started": True}]
    assert record["sources"][1]["plan_sha256"] == plan("fix-window-count").sha256()
    assert record["admission"] == {"status": "not_configured"}
    assert record["label_validation"]["levels_skipped"] == []
    assert 1 not in record["label_validation"]["levels"]
    assert record["max_running_check"]["status"] == "not_sampled"
    assert record["configuration"]["verified"]["settings"]["max_num_seqs"]["status"] == (
        "declared_not_verified"
    )
    assert read(harness.sweep_file) == json.loads(json.dumps(record))


HARD_STOP_HOOKS = {
    "infrastructure_failure": set_runs(state="failed", reason="run_timeout"),
    "delivery_incomplete": set_runs(delivery_complete=False),
}


@pytest.mark.parametrize(
    "code",
    [
        "infrastructure_failure",
        "delivery_incomplete",
        "server_busy_after_idle_wait",
        "server_state_unavailable",
    ],
)
def test_each_hard_stop_ends_the_ramp(tmp_path, code):
    harness = Harness(tmp_path)

    def trigger(stub, entries, wall):
        if code in HARD_STOP_HOOKS:
            return HARD_STOP_HOOKS[code](stub, entries, wall)
        if code == "server_busy_after_idle_wait":
            # Busy beyond the idle wait, then idle again before the live phase.
            harness.server.busy_until = harness.time.now + IDLE_WAIT_SECONDS + 15
        else:
            harness.server.fail_read(after=1)  # the reading right after the batch
        return None

    harness.hooks["ramp-n3-r0"] = trigger
    record = harness.run()
    assert record["state"] == "completed" and record["ramp_end"] == "hard_stop"
    # The ramp ends at the first batch with the event; later phases still run.
    assert harness.phases_run() == [*expected_ramp(2), ("ramp", 3), *post_ramp(2, 1)]
    # Batch 10: the warm-up is 0, N = 1 is 1-6, N = 2 is 7-9.
    assert [(h["code"], h["phase"], h["level"], h["batch"]) for h in record["hard_stops"]] == [
        (code, "ramp", 3, 10)
    ]
    assert record["capacity_decision"]["capacity"] == 2
    assert record["capacity_decision"]["criteria"][3]["no_hard_stop"]["passed"] is False


def test_a_vllm_restart_stops_the_whole_sweep(tmp_path):
    harness = Harness(tmp_path)

    def restart(_stub, _entries, _wall):
        harness.server.start_time += 60.0

    harness.hooks["ramp-n2-r0"] = restart
    record = harness.run()
    assert record["state"] == "stopped_vllm_restart" and record["ramp_end"] == "stopped"
    assert harness.phases_run()[-1] == ("ramp", 2)
    assert [h["code"] for h in record["hard_stops"]] == ["vllm_restart"]
    assert record["analysis_scope"] == "partial_ramp"
    assert record["phases"]["live"]["status"] == "not_run"
    assert record["phases"]["injected"]["status"] == "not_run"
    # No idle wait or pacing once the sweep has stopped.
    assert harness.events[-1] == ("batch", "ramp-n2-r0")


def test_unconfirmed_injector_cleanup_stops_the_sweep(tmp_path):
    harness = Harness(tmp_path)
    harness.confirm_cleanup = False
    record = harness.run()
    assert record["state"] == "stopped_injector_cleanup_unconfirmed"
    assert record["ramp_end"] == "ceiling"
    assert harness.phases_run()[-1] == ("injected", 1)
    last = record["batches"][-1]
    assert last["phase"] == "injected" and last["injection_cleanup_confirmed"] is False
    assert "cleanup was not confirmed" in last["injection_error"]
    assert read(harness.output_dir / last["injection_log"])["cleanup_confirmed"] is False
    assert [h["code"] for h in record["hard_stops"]] == ["injector_cleanup_unconfirmed"]
    # The batch's runs are still recorded.
    assert last["runs"][0]["state"] == "completed"


def test_unreadable_server_state_is_a_hard_stop(tmp_path):
    harness = Harness(tmp_path)

    def fail_before_level_three(_stub, _entries, _wall):
        # After this batch: the reading after it, one idle poll, then the reading before
        # the next batch, which is level 3's first and fails.
        harness.server.fail_read(after=3)

    harness.hooks["ramp-n2-r2"] = fail_before_level_three
    record = harness.run()
    assert record["state"] == "completed" and record["ramp_end"] == "hard_stop"
    # Level 3 never ran: the unreadable server stopped the ramp before its first batch.
    assert harness.phases_run() == [*expected_ramp(2), *post_ramp(2, 1)]
    assert [(h["code"], h["level"], h["batch"]) for h in record["hard_stops"]] == [
        ("server_state_unavailable", 3, None)
    ]
    criteria = record["capacity_decision"]["criteria"][3]
    assert criteria["summary_present"]["passed"] is False
    assert record["capacity_decision"]["capacity"] == 2


def test_unreadable_server_state_after_the_ramp_stops_the_sweep(tmp_path):
    harness = Harness(tmp_path)

    def fail_before_live(_stub, _entries, _wall):
        harness.server.fail_read(after=3)  # after, idle poll, then before the first live batch

    harness.hooks["ramp-n16-r2"] = fail_before_live
    record = harness.run()
    assert record["state"] == "stopped_server_state_unavailable"
    assert record["ramp_end"] == "ceiling"
    assert record["capacity_decision"]["capacity"] == 2
    assert not any(call[0] == "live" for call in harness.calls)
    assert record["hard_stops"][-1]["phase"] == "live"


@pytest.mark.parametrize("how", ["runs_not_started", "batch_raises"])
def test_a_batch_whose_runs_never_started_stops_the_sweep(tmp_path, how):
    harness = Harness(tmp_path)

    def not_started(_stub, entries, wall):
        for entry in entries:
            entry |= {
                "state": "not_started",
                "run_id": None,
                "task_passed": None,
                "delivery_complete": None,
                "resend_needed": None,
                "not_started": {"stage": "registration", "error": "fixture"},
            }
        if how == "batch_raises":
            raise RunNotStarted("fixture: policy unavailable", "policy")
        return wall, [None] * len(entries)

    harness.hooks["ramp-n2-r1"] = not_started
    record = harness.run()
    assert record["state"] == "stopped_run_not_started" and record["ramp_end"] == "stopped"
    assert harness.phases_run()[-1] == ("ramp", 2)
    assert len(harness.calls) == len(expected_ramp(1)) + 2
    assert record["batches"][-1]["phase"] == "ramp"
    assert "not started" in record["stop_reason"]
    assert record["phases"]["live"]["status"] == "not_run"
    # Without a returned run list, the batch's own runs keep their evidence.
    assert [run["state"] for run in record["batches"][-1]["runs"]] == ["not_started"] * 2


def test_the_idle_wait_times_out_as_server_busy(tmp_path):
    harness = Harness(tmp_path)

    def stay_busy(_stub, _entries, _wall):
        harness.server.busy_until = float("inf")

    harness.hooks["ramp-n4-r1"] = stay_busy
    record = harness.run()
    busy = [h for h in record["hard_stops"] if h["code"] == "server_busy_after_idle_wait"]
    assert [(h["phase"], h["level"]) for h in busy] == [("ramp", 4), ("live", 2)]
    # Polled at most IDLE_WAIT_SECONDS, with no pacing pause after a busy server.
    after = harness.events.index(("batch", "ramp-n4-r1"))
    sleeps = [seconds for kind, seconds in harness.events[after + 1 :] if kind == "sleep"]
    assert sum(sleeps) == 2 * IDLE_WAIT_SECONDS and PACING_SECONDS not in sleeps
    # The capacity decision ran; the live phase could not start on a busy server.
    assert record["capacity_decision"]["capacity"] == 2
    assert record["state"] == "stopped_server_busy_after_idle_wait"
    assert not any(call[0] == "live" for call in harness.calls)


def test_pacing_pauses_ten_seconds_after_the_server_is_idle(tmp_path):
    harness = Harness(tmp_path)

    def busy_three_seconds(_stub, _entries, _wall):
        harness.server.busy_until = harness.time.now + 3.0

    harness.hooks["warmup-n1-r0"] = busy_three_seconds
    harness.run()
    first, second, third = [i for i, e in enumerate(harness.events) if e[0] == "batch"][:3]
    assert harness.events[first + 1 : second] == [
        ("sleep", 1.0),
        ("sleep", 1.0),
        ("sleep", 1.0),
        ("sleep", PACING_SECONDS),
    ]
    # Every later batch follows an idle server and a 10 s pause.
    assert harness.events[second + 1 : third] == [("sleep", PACING_SECONDS)]


def test_sweep_json_is_rewritten_after_every_batch(tmp_path):
    harness = Harness(tmp_path)
    record = harness.run()
    # Each batch saw sweep.json holding every finished batch before it, and itself as started.
    assert harness.sweep_json_batches == [
        [True] * count + [False] for count in range(len(record["batches"]))
    ]
    assert read(harness.sweep_file)["state"] == "completed"


def test_a_contradicted_configuration_refuses_the_sweep(tmp_path):
    harness = Harness(tmp_path, cache_config=CACHE_CONFIG | {"gpu_memory_utilization": "0.9"})
    record = harness.run()
    assert record["state"] == "refused_configuration" and record["ramp_end"] is None
    assert harness.calls == [] and record["batches"] == []
    verified = read(harness.sweep_file)["configuration"]["verified"]
    assert verified["contradicted"] == ["gpu_memory_utilization"]
    assert verified["kv_cache_size_tokens"] == 7376


def test_cancellation_runs_nothing_further(tmp_path):
    harness = Harness(tmp_path)

    def cancel(stub, entries, wall):
        harness.sweep.request_cancel_from_signal("signal", "Cancelled by operator signal SIGINT")
        for entry in entries:
            entry |= {"state": "cancelled", "reason": "Cancelled by operator signal SIGINT"}
        return wall, [None] * len(entries)

    harness.hooks["ramp-n2-r0"] = cancel
    record = harness.run()
    assert record["state"] == "cancelled" and record["ramp_end"] == "cancelled"
    assert harness.phases_run()[-1] == ("ramp", 2)
    # The running batch was told; no idle wait or pause followed.
    assert harness.batches[-1].cancel_requests == [
        ("signal", "Cancelled by operator signal SIGINT")
    ]
    assert harness.events[-1] == ("batch", "ramp-n2-r0")
    assert record["analysis_scope"] == "partial_ramp"
    assert record["cancellation"] == {
        "source": "signal",
        "reason": "Cancelled by operator signal SIGINT",
    }


def test_capacity_below_one_skips_live_and_injected_phases(tmp_path):
    harness = Harness(tmp_path)
    for level in RAMP_LEVELS:
        for repeat in range(6 if level == 1 else 3):
            harness.hooks[f"ramp-n{level}-r{repeat}"] = set_runs(
                state="failed", reason=VERIFICATION_FAILED, task_passed=False
            )
    record = harness.run()
    # Task failures are recorded, never a hard stop: the ramp reaches the ceiling.
    assert record["ramp_end"] == "ceiling" and record["hard_stops"] == []
    assert record["capacity_decision"]["result"] == "below_1"
    assert record["phases"]["live"]["status"] == "skipped"
    assert "below 1" in record["phases"]["live"]["reason"]
    assert record["phases"]["injected"]["status"] == "skipped"
    assert not any(call[0] in ("live", "injected", "control") for call in harness.calls)
    assert record["state"] == "completed"


def test_a_resend_needed_run_stops_the_sweep(tmp_path):
    harness = Harness(tmp_path)
    harness.hooks["ramp-n2-r0"] = set_runs(delivery_complete=False, resend_needed=True)
    record = harness.run()
    assert record["state"] == "stopped_resend_needed" and record["ramp_end"] == "stopped"
    assert harness.phases_run()[-1] == ("ramp", 2)
    assert [h["code"] for h in record["hard_stops"]] == ["delivery_incomplete"]


def test_an_unexpected_error_is_recorded_and_re_raised(tmp_path):
    harness = Harness(tmp_path)

    def explode(_stub, _entries, _wall):
        raise RuntimeError("fixture: driver bug")

    harness.hooks["ramp-n2-r0"] = explode
    with pytest.raises(RuntimeError, match="driver bug"):
        harness.run()
    record = read(harness.sweep_file)
    assert record["state"] == "failed"
    assert record["error"] == "RuntimeError: fixture: driver bug"
    assert harness.batches[-1].cancel_requests[0][0] == "sweep"
    # The batch's own runs keep their evidence although run() raised.
    assert [run["state"] for run in record["batches"][-1]["runs"]] == ["completed"] * 2
    # Analysis was skipped, and the record says why.
    assert record["analysis_scope"] == "none"
    assert "failed" in record["analysis_note"]


def test_a_runner_error_is_re_raised_once_the_batch_is_recorded(tmp_path):
    harness = Harness(tmp_path)

    def runner_error(_stub, entries, wall):
        return wall, [ValueError("fixture: runner bug")] + [None] * (len(entries) - 1)

    harness.hooks["ramp-n2-r0"] = runner_error
    with pytest.raises(ValueError, match="runner bug"):
        harness.run()
    record = read(harness.sweep_file)
    assert record["state"] == "failed"
    assert record["batches"][-1]["phase"] == "ramp" and record["batches"][-1]["level"] == 2


def test_a_hard_stop_in_the_warmup_stops_the_sweep(tmp_path):
    harness = Harness(tmp_path)
    harness.hooks["warmup-n1-r0"] = set_runs(state="failed", reason="run_timeout")
    record = harness.run()
    assert harness.phases_run() == [("warmup", 1)]
    # Never "completed": the warm-up failed, so the ramp never started.
    assert record["state"] == "stopped_warmup_failed" and record["ramp_end"] == "stopped"
    assert [(h["code"], h["phase"], h["batch"]) for h in record["hard_stops"]] == [
        ("infrastructure_failure", "warmup", 0)
    ]
    assert record["phases"]["ramp"]["status"] == "not_run"
    assert record["phases"]["live"]["status"] == "not_run"


def cancel_sweep(harness):
    harness.sweep.request_cancel_from_signal("signal", "Cancelled by operator signal SIGINT")


@pytest.mark.parametrize("how", ["runs_not_started", "vllm_restart", "resend_needed", "cancelled"])
def test_a_sweep_stopped_mid_ramp_still_analyses_the_recorded_ramp(tmp_path, how):
    harness = Harness(tmp_path)

    def stop(_stub, entries, wall):
        if how == "runs_not_started":
            for entry in entries:
                entry |= {"state": "not_started", "delivery_complete": None}
        elif how == "vllm_restart":
            harness.server.start_time += 60.0
        elif how == "resend_needed":
            entries[0] |= {"delivery_complete": False, "resend_needed": True}
        else:
            cancel_sweep(harness)
        return wall, [None] * len(entries)

    harness.hooks["ramp-n2-r1"] = stop
    record = harness.run()
    assert record["state"] != "completed" and record["ramp_end"] in ("stopped", "cancelled")
    # The ramp batches it recorded are analysed and marked with the stop.
    assert set(record["level_summaries"]) == {1, 2}
    assert record["level_summaries"][2]["batches"] == 2
    assert record["capacity_decision"] is not None
    assert record["analysis_scope"] == "partial_ramp"
    assert record["state"] in record["analysis_note"]
    assert record["stop_reason"] in record["analysis_note"]
    assert set(record["label_validation"]["levels"]) == {2}
    assert record["max_running_check"]["status"] == "not_sampled"
    saved = read(harness.sweep_file)
    assert saved["analysis_scope"] == "partial_ramp" and "2" in saved["level_summaries"]


@pytest.mark.parametrize("where", ["make_batch", "read_state"])
def test_cancellation_just_before_a_batch_runs_stops_it(tmp_path, where):
    harness = Harness(tmp_path)
    if where == "make_batch":
        harness.on_make_batch["ramp-n2-r0"] = lambda: cancel_sweep(harness)
    else:

        def arm(_stub, _entries, _wall):
            # The reading after this batch, one idle poll, then the reading before the next.
            target = harness.server.reads + 3

            def on_read(number):
                if number == target:
                    cancel_sweep(harness)

            harness.server.on_read = on_read

        harness.hooks["ramp-n1-r5"] = arm
    record = harness.run()
    assert record["state"] == "cancelled" and record["ramp_end"] == "cancelled"
    assert harness.calls[-1][:2] == ("ramp", 2)
    # The batch was made but never run.
    assert harness.batches[-1].ran is False
    assert ("batch", "ramp-n2-r0") not in harness.events


def test_cancellation_during_the_injector_ramp_stops_the_sweep(tmp_path):
    harness = Harness(tmp_path)
    harness.on_make_injectors = lambda: cancel_sweep(harness)
    record = harness.run()
    assert record["state"] == "cancelled"
    assert harness.batches[-1].ran is False
    last = record["batches"][-1]
    assert last["phase"] == "injected"
    log = read(harness.output_dir / last["injection_log"])
    assert log["ramp_completed_tick_ns"] is None
    assert [injection["stop_reason"] for injection in log["injections"]] == ["cancelled"]
    assert last["injection_cleanup_confirmed"] is True


def test_cancellation_between_batches_runs_nothing_further(tmp_path):
    harness = Harness(tmp_path)

    def cancel_while_pacing(_stub, _entries, _wall):
        def on_sleep(seconds):
            if seconds == PACING_SECONDS:
                cancel_sweep(harness)

        harness.time.on_sleep = on_sleep

    harness.hooks["ramp-n2-r0"] = cancel_while_pacing
    record = harness.run()
    assert record["state"] == "cancelled" and record["ramp_end"] == "cancelled"
    assert harness.phases_run()[-1] == ("ramp", 2)
    assert harness.events[-2:] == [("batch", "ramp-n2-r0"), ("sleep", PACING_SECONDS)]


def test_resend_needed_outranks_cancellation_and_restart(tmp_path):
    harness = Harness(tmp_path)

    def everything(_stub, entries, wall):
        cancel_sweep(harness)
        harness.server.start_time += 60.0
        entries[0] |= {"delivery_complete": False, "resend_needed": True}
        entries[1] |= {"state": "cancelled"}
        return wall, [None] * len(entries)

    harness.hooks["ramp-n2-r0"] = everything
    record = harness.run()
    assert record["state"] == "stopped_resend_needed"
    # The other causes are recorded too.
    assert [cause["state"] for cause in record["stop_causes"]] == [
        "stopped_vllm_restart",
        "cancelled",
        "stopped_resend_needed",
    ]
    assert {h["code"] for h in record["hard_stops"]} == {"vllm_restart", "delivery_incomplete"}


def test_a_server_without_a_start_time_is_refused(tmp_path):
    harness = Harness(tmp_path)
    harness.server.start_time = None
    record = harness.run()
    assert record["state"] == "refused_configuration"
    assert "restart" in record["stop_reason"]
    assert harness.calls == []


def test_an_unreadable_model_list_stops_before_anything_runs(tmp_path):
    harness = Harness(tmp_path)

    def unreachable():
        raise urllib.error.URLError("fixture: connection refused")

    harness.models = unreachable
    record = harness.run()
    assert record["state"] == "stopped_server_state_unavailable"
    assert "models" in record["stop_reason"]
    assert harness.calls == [] and record["analysis_scope"] == "none"


def test_a_runner_error_with_unconfirmed_cleanup_records_the_cleanup_first(tmp_path):
    harness = Harness(tmp_path)
    harness.confirm_cleanup = False

    def runner_error(_stub, entries, wall):
        return wall, [ValueError("fixture: runner bug")] + [None] * (len(entries) - 1)

    harness.hooks["injected-n1-r0"] = runner_error
    with pytest.raises(ValueError, match="runner bug"):
        harness.run()
    record = read(harness.sweep_file)
    assert record["state"] == "failed"
    assert [h["code"] for h in record["hard_stops"]] == ["injector_cleanup_unconfirmed"]
    assert record["batches"][-1]["injection_cleanup_confirmed"] is False
