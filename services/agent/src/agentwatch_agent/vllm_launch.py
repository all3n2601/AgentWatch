"""Read the vLLM container's launch arguments with `docker inspect`, when Docker is reachable.

vLLM v0.32 does not report `max_num_seqs` on `/metrics`, so the capacity sweep reads the flag
the Compose service started vLLM with. `docker` is tried first, then `docker.exe` (Docker
Desktop through WSL interop). Each attempt is one bounded subprocess with list arguments and no
shell. A missing Docker, a failed or timed-out inspect, or a reply that is not one container's
JSON is recorded `unavailable` with every reason, never guessed; a flag that is absent or not a
number is None, never vLLM's default.

Launch flags describe the vLLM being measured only if this container serves its endpoint:
the container must publish the endpoint's port on an address that loopback reaches
(`serves_endpoint`). Two containers cannot publish one host port, so that ties the
container to the endpoint; otherwise the flags verify nothing (`vllm_server`).
"""

import json
import math
import subprocess
from collections.abc import Callable

# Compose names the container <project>-<service>-<index>; the project is `agentwatch`.
DEFAULT_VLLM_CONTAINER = "agentwatch-vllm-1"
DOCKER_EXECUTABLES = ("docker", "docker.exe")
DOCKER_INSPECT_TIMEOUT_SECONDS = 10.0
MAX_NUM_SEQS_FLAG = "--max-num-seqs"
GPU_MEMORY_UTILIZATION_FLAG = "--gpu-memory-utilization"
MAX_REASON_TEXT = 300
# Published host addresses a loopback endpoint reaches: loopback itself or every interface.
LOOPBACK_REACHABLE_HOST_IPS = frozenset({"", "0.0.0.0", "::", "127.0.0.1", "::1"})

READ = "read"
UNAVAILABLE = "unavailable"

Runner = Callable[..., subprocess.CompletedProcess]


def _failure_text(executable: str, error: BaseException) -> str:
    detail = f"{type(error).__name__}: {error}"
    if isinstance(error, subprocess.CalledProcessError) and error.stderr:
        detail += f" ({str(error.stderr).strip()})"
    return f"{executable}: {detail}"[:MAX_REASON_TEXT]


def _flag_value(arguments: list[str], flag: str) -> str | None:
    """The flag's last value (`--flag value` or `--flag=value`), as argparse would take it."""
    value = None
    for position, argument in enumerate(arguments):
        if argument == flag and position + 1 < len(arguments):
            value = arguments[position + 1]
        elif argument.startswith(flag + "="):
            value = argument.split("=", 1)[1]
    return value


def _int(text: str | None) -> int | None:
    try:
        return int(text) if text is not None else None
    except ValueError:
        return None


def _float(text: str | None) -> float | None:
    try:
        value = float(text) if text is not None else None
    except ValueError:
        return None
    return value if value is not None and math.isfinite(value) else None


def _published_ports(document: dict) -> list[dict]:
    """Each published port binding; a malformed entry is left out, never guessed."""
    network = document.get("NetworkSettings")
    ports = network.get("Ports") if isinstance(network, dict) else None
    if not isinstance(ports, dict):
        return []
    published = []
    for container_port, bindings in ports.items():
        for binding in bindings if isinstance(bindings, list) else []:
            if not isinstance(binding, dict) or not isinstance(binding.get("HostIp"), str):
                continue
            host_port = _int(binding.get("HostPort"))
            if host_port is not None:
                published.append(
                    {
                        "container_port": container_port,
                        "host_ip": binding["HostIp"],
                        "host_port": host_port,
                    }
                )
    return published


def _serves(published: list[dict], endpoint_port: int | None) -> bool | None:
    """Whether a binding publishes `endpoint_port` where loopback reaches it; None if unknown."""
    if endpoint_port is None:
        return None
    return any(
        binding["host_port"] == endpoint_port and binding["host_ip"] in LOOPBACK_REACHABLE_HOST_IPS
        for binding in published
    )


def parse_inspect(
    container: str, executable: str, stdout: str, endpoint_port: int | None = None
) -> dict:
    """The launch arguments in one container's `docker inspect` JSON, or why they cannot be read."""
    unusable = {
        "status": UNAVAILABLE,
        "container": container,
        "reason": f"{executable}: docker inspect did not return one container's launch arguments",
    }
    try:
        document = json.loads(stdout)
    except ValueError:
        return unusable
    if not isinstance(document, list) or len(document) != 1 or not isinstance(document[0], dict):
        return unusable
    config = document[0].get("Config")
    state = document[0].get("State")
    if not isinstance(config, dict):
        return unusable
    parts = [config.get("Entrypoint") or [], config.get("Cmd") or []]
    if not all(isinstance(part, list) and all(isinstance(a, str) for a in part) for part in parts):
        return unusable
    arguments = [*parts[0], *parts[1]]
    running = state.get("Running") if isinstance(state, dict) else None
    published = _published_ports(document[0])
    return {
        "status": READ,
        "container": container,
        "executable": executable,
        "image": config.get("Image") if isinstance(config.get("Image"), str) else None,
        "running": running if isinstance(running, bool) else None,
        "arguments": arguments,
        "max_num_seqs": _int(_flag_value(arguments, MAX_NUM_SEQS_FLAG)),
        "gpu_memory_utilization": _float(_flag_value(arguments, GPU_MEMORY_UTILIZATION_FLAG)),
        "published_ports": published,
        "endpoint_port": endpoint_port,
        "serves_endpoint": _serves(published, endpoint_port),
    }


def read_launch_arguments(
    container: str, run: Runner = subprocess.run, *, endpoint_port: int | None = None
) -> dict:
    """`docker inspect` the container with each executable in turn; the first reply is used.

    `endpoint_port` is the port of the vLLM endpoint being measured; without it the
    container cannot be tied to that endpoint (`serves_endpoint` is None).
    """
    failures = []
    for executable in DOCKER_EXECUTABLES:
        command = [executable, "inspect", "--type", "container", container]
        try:
            completed = run(
                command,
                capture_output=True,
                text=True,
                timeout=DOCKER_INSPECT_TIMEOUT_SECONDS,
                check=True,
            )
        # A missing executable is an OSError; a failed or timed-out inspect a SubprocessError;
        # output that is not UTF-8 raises UnicodeDecodeError, a ValueError.
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            failures.append(_failure_text(executable, error))
            continue
        return parse_inspect(container, executable, completed.stdout, endpoint_port)
    return {"status": UNAVAILABLE, "container": container, "reason": "; ".join(failures)}
