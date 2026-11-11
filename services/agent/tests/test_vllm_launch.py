"""Reading vLLM's launch arguments from `docker inspect`, with a fake subprocess runner only.

No test here runs Docker: every `docker inspect` reply is a synthetic copy of the shape Docker
prints for a container (a JSON array of one object), built from the Compose service's command.
"""

import json
import subprocess

import pytest
from agentwatch_agent.sweep_protocol import CONFIGURATIONS
from agentwatch_agent.vllm_launch import (
    DEFAULT_VLLM_CONTAINER,
    DOCKER_EXECUTABLES,
    DOCKER_INSPECT_TIMEOUT_SECONDS,
    read_launch_arguments,
)
from agentwatch_agent.vllm_server import (
    SERVED_MODEL_NAME,
    ServerState,
    verify_configuration,
)


def compose_command(max_num_seqs="2", utilization="0.6", style="equals"):
    """The Compose service's `command:` with its variables interpolated, as Docker stores it."""
    flags = {"--gpu-memory-utilization": utilization, "--max-num-seqs": max_num_seqs}
    command = [
        "--model=Qwen/Qwen3-1.7B",
        "--served-model-name=Qwen/Qwen3-1.7B@70d244cc86cc",
        "--max-model-len=4096",
    ]
    for flag, value in flags.items():
        if value is None:
            continue
        command += [f"{flag}={value}"] if style == "equals" else [flag, value]
    return command + ["--enable-auto-tool-choice", "--tool-call-parser=hermes"]


def inspect_document(command, running=True):
    return [
        {
            "Name": f"/{DEFAULT_VLLM_CONTAINER}",
            "Config": {
                "Image": "vllm/vllm-openai@sha256:fixture",
                "Entrypoint": ["vllm", "serve"],
                "Cmd": command,
            },
            "State": {"Running": running, "Status": "running" if running else "exited"},
        }
    ]


def inspect_runner(flags=None, *, stdout=None, missing=(), failing=()):
    """A fake `subprocess.run` for `docker inspect`, and the calls it received."""
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        executable = command[0]
        if executable in missing:
            raise FileNotFoundError(2, "No such file or directory", executable)
        if executable in failing:
            raise subprocess.CalledProcessError(
                1, command, output="", stderr="Error: No such container: agentwatch-vllm-1"
            )
        text = (
            stdout
            if stdout is not None
            else json.dumps(inspect_document(compose_command(**(flags or {}))))
        )
        return subprocess.CompletedProcess(command, 0, stdout=text, stderr="")

    return run, calls


@pytest.mark.parametrize("style", ["equals", "separate"])
def test_the_launch_arguments_are_parsed_from_the_container(style):
    run, calls = inspect_runner({"max_num_seqs": "16", "utilization": "0.8", "style": style})
    launch = read_launch_arguments(DEFAULT_VLLM_CONTAINER, run=run)
    assert launch["status"] == "read"
    assert (launch["max_num_seqs"], launch["gpu_memory_utilization"]) == (16, 0.8)
    assert launch["arguments"][:2] == ["vllm", "serve"]
    assert launch["running"] is True and launch["executable"] == "docker"
    # Bounded, list arguments, never a shell.
    ((command, kwargs),) = calls
    assert command == ["docker", "inspect", "--type", "container", DEFAULT_VLLM_CONTAINER]
    assert kwargs["timeout"] == DOCKER_INSPECT_TIMEOUT_SECONDS
    assert kwargs.get("shell", False) is False
    assert kwargs["check"] is True


def test_docker_exe_is_tried_when_docker_is_missing():
    run, calls = inspect_runner(missing=("docker",))
    launch = read_launch_arguments(DEFAULT_VLLM_CONTAINER, run=run)
    assert launch["status"] == "read" and launch["executable"] == "docker.exe"
    assert [command[0] for command, _kwargs in calls] == list(DOCKER_EXECUTABLES)


@pytest.mark.parametrize(
    ("runner", "reason"),
    [
        (inspect_runner(missing=DOCKER_EXECUTABLES)[0], "FileNotFoundError"),
        (inspect_runner(failing=DOCKER_EXECUTABLES)[0], "No such container"),
    ],
)
def test_docker_unavailable_is_recorded_with_every_reason(runner, reason):
    launch = read_launch_arguments(DEFAULT_VLLM_CONTAINER, run=runner)
    assert launch["status"] == "unavailable"
    assert reason in launch["reason"]
    assert "docker:" in launch["reason"] and "docker.exe:" in launch["reason"]


def test_a_timed_out_inspect_is_unavailable():
    def slow(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    launch = read_launch_arguments(DEFAULT_VLLM_CONTAINER, run=slow)
    assert launch["status"] == "unavailable" and "TimeoutExpired" in launch["reason"]


@pytest.mark.parametrize(
    "stdout",
    [
        "not json",
        "[]",
        json.dumps([{"Config": {"Entrypoint": None, "Cmd": "vllm serve"}}]),
        json.dumps([{"Config": {"Entrypoint": None, "Cmd": [1, 2]}}]),
    ],
)
def test_a_malformed_inspect_reply_is_unavailable_not_guessed(stdout):
    run, _calls = inspect_runner(stdout=stdout)
    launch = read_launch_arguments(DEFAULT_VLLM_CONTAINER, run=run)
    assert launch["status"] == "unavailable" and "docker inspect" in launch["reason"]


def test_an_unset_or_unparsable_flag_is_none_never_a_default():
    run, _calls = inspect_runner({"max_num_seqs": None, "utilization": "lots"})
    launch = read_launch_arguments(DEFAULT_VLLM_CONTAINER, run=run)
    assert launch["status"] == "read"
    assert launch["max_num_seqs"] is None and launch["gpu_memory_utilization"] is None


def state(utilization="0.6"):
    return ServerState(
        start_time_seconds=1.0,
        cache_config={"gpu_memory_utilization": utilization, "enable_prefix_caching": "True"},
        counters={},
        running=0.0,
        waiting=0.0,
    )


MODELS = {"data": [{"id": SERVED_MODEL_NAME, "max_model_len": 4096}]}


def launch(**flags):
    run, _calls = inspect_runner(flags)
    return read_launch_arguments(DEFAULT_VLLM_CONTAINER, run=run)


def test_a_stopped_container_does_not_verify_the_running_server():
    run, _calls = inspect_runner(stdout=json.dumps(inspect_document(compose_command(), False)))
    stopped = read_launch_arguments(DEFAULT_VLLM_CONTAINER, run=run)
    verified = verify_configuration(CONFIGURATIONS["cfg-default"], MODELS, state(), stopped)
    setting = verified["settings"]["max_num_seqs"]
    assert setting["status"] == "declared_not_verified" and "not running" in setting["reason"]


def test_launch_utilization_must_match_what_vllm_reports():
    hardware = CONFIGURATIONS["cfg-hardware"]
    verified = verify_configuration(
        hardware, MODELS, state("0.75"), launch(max_num_seqs="16", utilization="0.8")
    )
    # 0.8 is declared, but vLLM reports 0.75: the launch arguments describe another server.
    assert verified["contradicted"] == ["launch_gpu_memory_utilization"]
    matching = verify_configuration(
        hardware, MODELS, state("0.75"), launch(max_num_seqs="16", utilization="0.75")
    )
    assert matching["contradicted"] == []
    assert matching["settings"]["max_num_seqs"]["status"] == "verified"
