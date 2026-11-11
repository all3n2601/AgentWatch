"""Read the vLLM container's launch arguments with `docker inspect`, when Docker is reachable.

vLLM v0.32 does not report `max_num_seqs` on `/metrics`, so the capacity sweep reads the flag
the Compose service started vLLM with. `docker` is tried first, then `docker.exe` (Docker
Desktop through WSL interop). Each attempt is one bounded subprocess with list arguments and no
shell. A missing Docker, a failed or timed-out inspect, or a reply that is not one container's
JSON is recorded `unavailable` with every reason, never guessed; a flag that is absent or not a
number is None, never vLLM's default.
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


def parse_inspect(container: str, executable: str, stdout: str) -> dict:
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
    return {
        "status": READ,
        "container": container,
        "executable": executable,
        "image": config.get("Image") if isinstance(config.get("Image"), str) else None,
        "running": running if isinstance(running, bool) else None,
        "arguments": arguments,
        "max_num_seqs": _int(_flag_value(arguments, MAX_NUM_SEQS_FLAG)),
        "gpu_memory_utilization": _float(_flag_value(arguments, GPU_MEMORY_UTILIZATION_FLAG)),
    }


def read_launch_arguments(container: str, run: Runner = subprocess.run) -> dict:
    """`docker inspect` the container with each executable in turn; the first reply is used."""
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
        # A missing executable is an OSError; a failed or timed-out inspect a SubprocessError.
        except (OSError, subprocess.SubprocessError) as error:
            failures.append(_failure_text(executable, error))
            continue
        return parse_inspect(container, executable, completed.stdout)
    return {"status": UNAVAILABLE, "container": container, "reason": "; ".join(failures)}
