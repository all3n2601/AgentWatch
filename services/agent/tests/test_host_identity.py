"""Host identity for optimization batches: measured readings, never defaults.

`/proc`, NVML and Windows command output are test doubles; no host is read.
"""

import pytest
from agent_test_doubles import FakeGpu
from agentwatch_agent.host_identity import (
    HostIdentityError,
    hardware_identity,
    host_label,
    power_identity,
)
from agentwatch_worker.nvml import GpuIdentity, NvmlError, NvmlUnavailable

SCHEME = "Power Scheme GUID: 381b4222-f694-41f0-9685-ff5bb260df2e  (Balanced)\r\n"


def fake_proc(tmp_path, cpu="model name\t: Test CPU 8C\n", memory="MemTotal:  1024 kB\n"):
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "cpuinfo").write_text(f"processor\t: 0\n{cpu}")
    (proc / "meminfo").write_text(memory)
    os_release = tmp_path / "os-release"
    os_release.write_text('NAME="Ubuntu"\nPRETTY_NAME="Ubuntu 26.04.1 LTS"\n')
    return proc, os_release


def windows(line_status="Online", scheme=SCHEME):
    def run(command):
        if command[0] == "powershell.exe":
            return line_status
        return scheme.replace("\r", "").strip()

    return run


def test_hardware_identity_is_measured_and_labelled_by_the_operator(tmp_path):
    proc, os_release = fake_proc(tmp_path)
    identity = hardware_identity(
        "rtx5070-laptop-wsl2", proc=proc, os_release=os_release, open_gpu=FakeGpu
    )
    assert identity["host_id"] == "rtx5070-laptop-wsl2"
    assert identity["cpu_model"] == "Test CPU 8C"
    assert identity["memory_bytes"] == 1024 * 1024
    assert identity["os"] == "Ubuntu 26.04.1 LTS"
    assert identity["logical_cpus"] >= 1 and identity["kernel"]
    assert identity["gpus"] == [
        {"name": "Test GPU", "driver_version": "0.0", "memory_total_bytes": 8 * 2**30}
    ]


def test_a_host_without_nvml_has_no_gpus_and_an_nvml_error_is_not_hidden(tmp_path):
    proc, os_release = fake_proc(tmp_path)

    def unavailable():
        raise NvmlUnavailable("libnvidia-ml.so.1 not found")

    assert (
        hardware_identity("cpu-host", proc=proc, os_release=os_release, open_gpu=unavailable)[
            "gpus"
        ]
        == []
    )

    def broken():
        raise NvmlError("nvmlInit", 999)

    with pytest.raises(HostIdentityError, match="NVML could not identify"):
        hardware_identity("cpu-host", proc=proc, os_release=os_release, open_gpu=broken)


def test_a_second_gpu_makes_the_device_ambiguous(tmp_path):
    proc, os_release = fake_proc(tmp_path)

    class TwoGpus(FakeGpu):
        identity = GpuIdentity(0, 2, "Test GPU", "0.0", 2**33)

    with pytest.raises(HostIdentityError, match="Expected one GPU"):
        hardware_identity("gpu-host", proc=proc, os_release=os_release, open_gpu=TwoGpus)


@pytest.mark.parametrize(
    ("cpu", "memory", "message"),
    [
        ("", "MemTotal:  1024 kB\n", "CPU model not found"),
        ("model name\t: Test CPU\n", "MemFree: 10 kB\n", "MemTotal not found"),
    ],
)
def test_unreadable_hardware_is_an_error_not_a_default(tmp_path, cpu, memory, message):
    proc, os_release = fake_proc(tmp_path, cpu=cpu, memory=memory)
    with pytest.raises(HostIdentityError, match=message):
        hardware_identity("host-a", proc=proc, os_release=os_release, open_gpu=None)


@pytest.mark.parametrize("label", ["A", "Benoy-Laptop", "x", "host name", "-lead"])
def test_host_labels_are_neutral_lowercase_names(label):
    with pytest.raises(ValueError, match="lowercase"):
        host_label(label)


def test_power_reads_the_line_status_and_active_plan():
    assert power_identity(windows()) == {"source": "ac", "profile": "Balanced"}
    assert power_identity(windows("Offline"))["source"] == "battery"


@pytest.mark.parametrize(
    ("run", "message"),
    [
        (windows("Unknown"), "power line status 'Unknown'"),
        (windows(scheme="Power Scheme GUID: 381b4222"), "power plan could not be read"),
    ],
)
def test_an_unknown_power_state_is_an_error(run, message):
    with pytest.raises(HostIdentityError, match=message):
        power_identity(run)
