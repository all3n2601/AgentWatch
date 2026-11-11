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


# The Compose service publishes vLLM's port 8000 on the host's loopback address only.
COMPOSE_PORTS = {"8000/tcp": [{"HostIp": "127.0.0.1", "HostPort": "8000"}]}


def inspect_document(command, running=True, ports=COMPOSE_PORTS):
    return [
        {
            "Name": f"/{DEFAULT_VLLM_CONTAINER}",
            "Config": {
                "Image": "vllm/vllm-openai@sha256:fixture",
                "Entrypoint": ["vllm", "serve"],
                "Cmd": command,
            },
            "State": {"Running": running, "Status": "running" if running else "exited"},
            "NetworkSettings": {"Ports": ports},
        }
    ]


def inspect_runner(flags=None, *, stdout=None, missing=(), failing=(), ports=COMPOSE_PORTS):
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
            else json.dumps(inspect_document(compose_command(**(flags or {})), ports=ports))
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


def test_undecodable_inspect_output_is_unavailable_not_a_crash():
    # `subprocess.run(text=True)` raises UnicodeDecodeError (a ValueError) for non-UTF-8 output.
    def undecodable(command, **kwargs):
        raise UnicodeDecodeError("utf-8", bytes([255]), 0, 1, "invalid start byte")

    launch = read_launch_arguments(DEFAULT_VLLM_CONTAINER, run=undecodable)
    assert launch["status"] == "unavailable"
    assert "UnicodeDecodeError" in launch["reason"]
    assert "docker:" in launch["reason"] and "docker.exe:" in launch["reason"]


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


def launch(endpoint_port=8000, ports=COMPOSE_PORTS, **flags):
    run, _calls = inspect_runner(flags, ports=ports)
    return read_launch_arguments(DEFAULT_VLLM_CONTAINER, run=run, endpoint_port=endpoint_port)


def test_a_stopped_container_does_not_verify_the_running_server():
    run, _calls = inspect_runner(stdout=json.dumps(inspect_document(compose_command(), False)))
    stopped = read_launch_arguments(DEFAULT_VLLM_CONTAINER, run=run, endpoint_port=8000)
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


def test_the_published_ports_and_the_served_endpoint_are_recorded():
    served = launch(endpoint_port=8000)
    assert served["published_ports"] == [
        {"container_port": "8000/tcp", "host_ip": "127.0.0.1", "host_port": 8000}
    ]
    assert served["endpoint_port"] == 8000 and served["serves_endpoint"] is True
    # No endpoint given: whether the container serves it is unknown, never assumed.
    assert launch(endpoint_port=None)["serves_endpoint"] is None


@pytest.mark.parametrize(
    "ports",
    [
        {"8000/tcp": [{"HostIp": "127.0.0.1", "HostPort": "8001"}]},  # another host port
        {"8000/tcp": [{"HostIp": "10.0.0.5", "HostPort": "8000"}]},  # not a loopback address
        {"8000/tcp": None},  # exposed, never published
        {},
        None,
    ],
)
def test_a_container_that_does_not_publish_the_endpoint_does_not_serve_it(ports):
    assert launch(endpoint_port=8000, ports=ports)["serves_endpoint"] is False


@pytest.mark.parametrize("host_ip", ["", "0.0.0.0", "::", "::1"])
def test_every_address_that_reaches_loopback_serves_the_endpoint(host_ip):
    ports = {"8000/tcp": [{"HostIp": host_ip, "HostPort": "8000"}]}
    assert launch(endpoint_port=8000, ports=ports)["serves_endpoint"] is True


def test_launch_evidence_from_another_container_never_verifies_the_endpoint():
    # Review of #65: the default container runs 2 sequences on port 8000, while the sweep
    # measures another local vLLM on port 8001 whose /metrics report the same settings.
    other = launch(endpoint_port=8001, max_num_seqs="2", utilization="0.6")
    verified = verify_configuration(CONFIGURATIONS["cfg-default"], MODELS, state(), other)
    for name in ("max_num_seqs", "launch_gpu_memory_utilization"):
        setting = verified["settings"][name]
        assert setting["status"] == "declared_not_verified"
        assert "does not publish" in setting["reason"] and "8001" in setting["reason"]
    assert verified["contradicted"] == []
    # The container that does publish port 8001 verifies the server there.
    second = {"8000/tcp": [{"HostIp": "127.0.0.1", "HostPort": "8001"}]}
    own = launch(endpoint_port=8001, ports=second, max_num_seqs="16", utilization="0.8")
    hardware = verify_configuration(CONFIGURATIONS["cfg-hardware"], MODELS, state("0.8"), own)
    assert hardware["settings"]["max_num_seqs"]["status"] == "verified"
    assert hardware["settings"]["max_num_seqs"]["reported"] == 16


def test_launch_evidence_without_an_endpoint_never_verifies():
    unbound = launch(endpoint_port=None)
    verified = verify_configuration(CONFIGURATIONS["cfg-default"], MODELS, state(), unbound)
    setting = verified["settings"]["max_num_seqs"]
    assert setting["status"] == "declared_not_verified" and "endpoint" in setting["reason"]
