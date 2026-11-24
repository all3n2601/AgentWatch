"""Observed identity of the host an optimization experiment runs on (O06).

The experiment manifest freezes the hardware and power state every batch must run under
(`docs/optimization-evidence.md`). These readers measure what can be measured before each
batch, so a batch records the state it actually ran in and the comparison rejects one that
differs. The host label is the only declared value: it names the machine without any user or
host name (A07 found a user name in a host label), so the operator chooses it.

A reading that fails is raised, never replaced by a default: an unknown power state cannot show
that both batches of a pair ran under the same one.
"""

import os
import platform
import re
import subprocess
from collections.abc import Callable
from pathlib import Path

from agentwatch_worker.nvml import NvmlDevice, NvmlError, NvmlUnavailable

# Windows reports AC power through .NET; WSL reaches it through interop. Both are bounded.
POWERSHELL_LINE_STATUS = (
    "Add-Type -AssemblyName System.Windows.Forms; "
    "[System.Windows.Forms.SystemInformation]::PowerStatus.PowerLineStatus"
)
WINDOWS_COMMAND_TIMEOUT_SECONDS = 20.0
HOST_LABEL = re.compile(r"[a-z0-9][a-z0-9-]{1,62}")
ACTIVE_SCHEME = re.compile(r"\(([^()]+)\)\s*$")


class HostIdentityError(RuntimeError):
    """A part of the host's identity could not be read."""


def host_label(value: str) -> str:
    """A neutral, operator-chosen label such as `rtx5070-laptop-wsl2`."""
    if not HOST_LABEL.fullmatch(value):
        raise ValueError(
            "The host label is 2-63 lowercase letters, digits and hyphens, e.g. rtx5070-laptop"
        )
    return value


def _field(path: Path, pattern: str, name: str) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        raise HostIdentityError(f"{name} unreadable from {path}: {error}") from error
    match = re.search(pattern, text, re.MULTILINE)
    if match is None:
        raise HostIdentityError(f"{name} not found in {path}")
    return match.group(1).strip()


def hardware_identity(
    label: str,
    *,
    proc: Path = Path("/proc"),
    os_release: Path = Path("/etc/os-release"),
    open_gpu: Callable[[], NvmlDevice] | None = NvmlDevice.open,
) -> dict:
    """The manifest's `hardware` block, measured; GPUs are those NVML reports (none on CPU hosts)."""
    memory_kib = int(_field(proc / "meminfo", r"^MemTotal:\s+(\d+)\s+kB", "MemTotal"))
    logical_cpus = os.cpu_count()
    if not logical_cpus:
        raise HostIdentityError("The logical CPU count is unknown")
    return {
        "host_id": host_label(label),
        "cpu_model": _field(proc / "cpuinfo", r"^model name\s*:\s*(.+)$", "CPU model"),
        "logical_cpus": logical_cpus,
        "memory_bytes": memory_kib * 1024,
        "os": _field(os_release, r'^PRETTY_NAME="?([^"\n]+)"?$', "OS name"),
        "kernel": platform.release(),
        "gpus": gpu_identities(open_gpu),
    }


def gpu_identities(open_gpu: Callable[[], NvmlDevice] | None) -> list[dict]:
    if open_gpu is None:
        return []
    try:
        device = open_gpu()
    except NvmlUnavailable:
        # No GPU this runner can use or measure. On a host whose manifest froze a GPU, the
        # empty list makes the batch a hardware mismatch instead of hiding the difference.
        return []
    except NvmlError as error:
        raise HostIdentityError(f"NVML could not identify the GPU: {error}") from error
    try:
        identity = device.identity
    finally:
        device.close()
    if identity.device_count != 1:
        # The runner samples device 0 only; a second GPU would make "the GPU" ambiguous.
        raise HostIdentityError(f"Expected one GPU, NVML reports {identity.device_count}")
    return [
        {
            "name": identity.name,
            "driver_version": identity.driver_version,
            "memory_total_bytes": identity.memory_total_bytes,
        }
    ]


def _windows(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=WINDOWS_COMMAND_TIMEOUT_SECONDS,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise HostIdentityError(f"{command[0]} failed: {error}") from error
    return result.stdout.replace("\r", "").strip()


def power_identity(run: Callable[[list[str]], str] = _windows) -> dict:
    """The manifest's `power` block from Windows (native or through WSL interop).

    `source` is `ac` or `battery`; `profile` is the active power plan's name.
    """
    line = run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", POWERSHELL_LINE_STATUS]
    )
    sources = {"Online": "ac", "Offline": "battery"}
    if line not in sources:
        raise HostIdentityError(f"Windows reports power line status {line!r}")
    scheme = ACTIVE_SCHEME.search(run(["powercfg.exe", "/getactivescheme"]))
    if scheme is None:
        raise HostIdentityError("The active Windows power plan could not be read")
    return {"source": sources[line], "profile": scheme.group(1).strip()}
