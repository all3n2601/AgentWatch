"""The capacity sweep end to end over real batches and runners, against test doubles only.

`CapacitySweep` drives real `AgentBatch`es of real `ReplayRunner`s (and `AgentRunner`s for the
live phase) against the loopback fake vLLM (chat, `/metrics`, request spans), the in-memory
live server and a fake GPU. Only the declared ramp is shortened to N = 1 and 2, and the
injector is the stub flood from `test_sweep`. Everything recorded here is synthetic.
"""

import functools
import itertools
from pathlib import Path

import pytest
from agent_test_doubles import FakeGpu
from agentwatch_agent import sweep, sweep_analysis, sweep_protocol
from agentwatch_agent.batch import AgentBatch
from agentwatch_agent.config import RunLimits
from agentwatch_agent.delivery import RetryPolicy
from agentwatch_agent.model import VllmChatModel
from agentwatch_agent.replay import ReplayRunner, load_replay_plan
from agentwatch_agent.runner import AgentRunner
from agentwatch_agent.sweep import CapacitySweep
from agentwatch_agent.sweep_offline import analyse_sweep_folder
from agentwatch_agent.sweep_protocol import CONFIGURATIONS, declared_batches
from agentwatch_agent.tasks import load_task
from agentwatch_agent.vllm_server import (
    SERVED_MODEL_NAME,
    ServerStateUnavailable,
    read_server_state,
)
from test_runner_vllm import SOLVE_MEAN_VLLM
from test_sweep import FakeInjector
from test_vllm_server import exposition, models
from vllm_test_server import MODEL, FakeVllm, text_reply, tool_reply

RUN_LIMIT_SECONDS = 120.0  # fits the sample budget at the fixture's sampling interval
SHORT_RAMP = (1, 2)
NO_SLEEP = RetryPolicy(attempts=2, initial_backoff_seconds=0, max_backoff_seconds=0)
SOLVE_WINDOW_VLLM = [
    tool_reply("read_file", '{"path": "windows.py"}'),
    tool_reply("apply_patch", '{"patch_id": "include-final-window"}'),
    tool_reply("run_tests", "{}"),
    text_reply("Applied include-final-window; the tests pass."),
]
# The runner's own vLLM probe reads these too; the extra series mirror vLLM's names.
PROBE_SERIES = f"""\
vllm:kv_cache_usage_perc{{engine="0",model_name="{MODEL}"}} 0.1
vllm:request_queue_time_seconds_sum{{engine="0",model_name="{MODEL}"}} 0.001
vllm:request_queue_time_seconds_count{{engine="0",model_name="{MODEL}"}} 3.0
"""


@pytest.fixture
def short_ramp(monkeypatch):
    for module in (sweep, sweep_analysis, sweep_protocol):
        monkeypatch.setattr(module, "RAMP_LEVELS", SHORT_RAMP)
    # The declared 2 s injector ramp is a real wait inside InjectionSet.
    monkeypatch.setattr(sweep, "INJECTOR_RAMP_SECONDS", 0.0)


def settings(server, fake, output_dir: Path, traces: Path, clock=None) -> dict:
    return {
        "model": VllmChatModel(fake.url, MODEL, num_ctx=4096),
        "transport": server,
        "output_dir": output_dir,
        "limits": RunLimits(run_timeout_seconds=RUN_LIMIT_SECONDS),
        "sampling_interval_seconds": 0.06,
        "cancel_poll_seconds": 0.05,
        "flush_interval_seconds": 0.05,
        "retry": NO_SLEEP,
        "proc": output_dir.parent / "no-procfs",
        "open_gpu": FakeGpu,
        "vllm_traces_file": traces,
        "trace_wait_seconds": 2.0,
        **({"clock": clock} if clock is not None else {}),
    }


def flaky_once(read, failing_read=5):
    """`read`, except that its `failing_read`-th call fails as an unreachable /metrics would."""
    calls = itertools.count(1)

    def reading():
        if next(calls) == failing_read:
            raise ServerStateUnavailable("fixture: one failed /metrics read")
        return read()

    return reading


def record_source(server, tmp_path, task, replies) -> Path:
    traces = tmp_path / "source-traces.jsonl"
    source = tmp_path / "sources" / task.name
    with FakeVllm(replies, traces_file=traces) as fake:
        outcome = AgentRunner(task=task, **settings(server, fake, source, traces)).run()
    assert (outcome.state, outcome.task_passed) == ("completed", True)
    return source


def test_a_sweep_over_real_batches_resolves_and_analyses_every_run(
    tmp_path, monkeypatch, server, short_ramp
):
    tasks = [load_task("fix-mean-empty"), load_task("fix-window-count")]
    scripts = [SOLVE_MEAN_VLLM, SOLVE_WINDOW_VLLM]
    plans = [
        load_replay_plan(record_source(server, tmp_path, task, script), task)
        for task, script in zip(tasks, scripts, strict=True)
    ]
    # The CLI's default output folder is relative to the working directory.
    monkeypatch.chdir(tmp_path)
    traces = tmp_path / "sweep-traces.jsonl"
    # Replays force the source's calls, so any reply serves; live agents answer at once.
    with FakeVllm([text_reply("Done.")] * 400, traces_file=traces) as fake:
        # vLLM labels its series with the model it serves: here the fake's, as the runners use.
        fake.metrics_text = exposition(extra=PROBE_SERIES).replace(SERVED_MODEL_NAME, MODEL)

        def make_batch(batch, path, clock):
            runners = []
            for index, source in enumerate(batch.sources, 1):
                task, plan = tasks[source], plans[source]
                folder = path / "runs" / f"{index:02d}-{task.name}"
                runners.append(
                    ReplayRunner(
                        plan=plan, task=task, **settings(server, fake, folder, traces, clock)
                    )
                )
            return AgentBatch(runners, None, path)

        def make_live_batch(batch, path, clock):
            runners = []
            for index, source in enumerate(batch.sources, 1):
                task = tasks[source]
                folder = path / "runs" / f"{index:02d}-{task.name}"
                runners.append(
                    AgentRunner(task=task, **settings(server, fake, folder, traces, clock))
                )
            return AgentBatch(runners, None, path)

        capacity_sweep = CapacitySweep(
            configuration=CONFIGURATIONS["cfg-default"],
            sources=plans,
            output_dir=Path("sweeps/e2e"),
            run_timeout_seconds=RUN_LIMIT_SECONDS,
            make_batch=make_batch,
            make_live_batch=make_live_batch,
            make_injectors=lambda clock, bound: [FakeInjector(clock, bound)],
            read_state=flaky_once(functools.partial(read_server_state, fake.url + "/metrics")),
            # The fake serves its own model name; the configuration check reads the declared one.
            models=models,
            host={"gpu": "fake"},
            sleep=lambda _seconds: None,
        )
        record = capacity_sweep.run()

    assert record["state"] == "completed", record["stop_reason"]
    ramp = [entry for entry in record["batches"] if entry["phase"] == "ramp"]
    assert [entry["level"] for entry in ramp] == [1] * declared_batches(1) + [2] * 3
    for entry in ramp:
        # The real runners record absolute folders: the batch's own runs/ folders, which the
        # analysis reads first.
        batch_runs = (capacity_sweep.output_dir / entry["folder"] / "runs").resolve()
        assert all(Path(run["output_dir"]).parent == batch_runs for run in entry["runs"])
        metrics = entry["metrics"]
        assert metrics["status"] == "readable"
        assert {folder["found_by"] for folder in metrics["run_folders"]} == {"batch_relative"}
        assert metrics["run_durations_missing"] == 0 and metrics["runs_without_evidence"] == 0
        assert metrics["passed"] == metrics["runs"]
        assert entry["sweep_process"]["wall_seconds"] > 0
    for level, batches in ((1, declared_batches(1)), (2, 3)):
        summary = record["level_summaries"][level]
        assert summary["batches"] == batches and summary["excluded_batches"] == 0
        assert summary["spans"]["calls_by_status"]["measured"] > 0
        # The fake span's e2e (0.033 s) exceeds a loopback call's wall time, so the runner
        # records no time outside the server: counted unavailable, never 0.
        outside = summary["client_outside_server_seconds"]
        assert outside["count"] + outside["unavailable"] == sum(
            summary["spans"]["calls_by_status"].values()
        )
        assert outside["unavailable"] > 0
        assert summary["device"]["metrics"]["gpu_sm_clock_mhz"]["runs_measured"] > 0
    decision = record["capacity_decision"]
    assert decision["result"] in ("measured", "at_least_ceiling")
    assert decision["capacity"] >= 1
    assert record["phases"]["live"]["status"] == "completed"
    # One failed /metrics read was retried within the declared bound, never a hard stop.
    assert record["hard_stops"] == [] and record["server"]["read_failures"] == []
    # Live batches run the tasks of their declared sources, recorded on each batch.
    live = [entry for entry in record["batches"] if entry["phase"] == "live"]
    for entry in live:
        tasks_run = [run["task_id"] for run in entry["runs"]]
        assert tasks_run == [source["task_id"] for source in entry["sources"]]
    assert record["phases"]["injected"]["status"] == "completed"
    labels = record["label_validation"]
    assert labels["runs_without_evidence"] == 0 and set(labels["levels"]) == {2}
    assert labels["label_rule_baselines"] > 0
    assert record["max_running_check"]["status"] == "consistent"
    # The offline analyser reaches the same decision from the folder alone.
    offline = analyse_sweep_folder(capacity_sweep.output_dir)
    assert offline["capacity_decision"]["capacity"] == decision["capacity"]
    assert offline["capacity_decision"]["result"] == decision["result"]
