# 0008: Read GPU profiling activity through a Windows-side NVML GPM exporter on WSL2 hosts

Date: October 28, 2026. Status: proposed for M2 review. Extends
[ADR 0007](0007-vllm-request-traces-and-server-metrics.md); depends on its service probes.

DCGM's profiling fields (SM activity and occupancy, tensor and DRAM activity) read N/A on the GPU
laptop. Calling NVML's GPU Performance Monitoring (GPM) API directly located the cause: the RTX
5070 supports GPM, and the Windows driver serves samples, but WSL2 refuses them ("GPU access
blocked by the operating system", then "in use by another client"), with or without DCGM
([the record](../gpu-measurement-access.md#m2-gpm-profiling-through-a-windows-side-exporter-october-28-2026)).
The runner, vLLM and DCGM all live in WSL2 or Docker on that laptop, so none of them can sample
GPM there.

**A Windows process samples GPM.** `agentwatch_worker.gpm_exporter` (`make gpm-exporter`, run on
Windows) binds the GPM entry points through the existing ctypes NVML binding
(`agentwatch_worker.gpm`) and computes nine metrics every 250 ms between two samples: graphics
engine active, SM utilization and occupancy, any tensor-core activity, DRAM bandwidth, FP32 and
FP16 pipe utilization (percent), and PCIe TX and RX (MiB/s). The driver names each metric ID; a
name that differs from the expected one is reported as refused, never under our name.

**A file, not a socket, crosses the boundary.** WSL2 cannot reach a plain Windows `localhost`
port on this host, and opening one to the WSL network would need a firewall rule. The exporter
instead replaces `.data/gpm/gpm.prom` atomically with a Prometheus text exposition, and the
runner reads it through `/mnt/c`. The file carries a random `instance` per exporter process, a
`sequence`, the window's length, each metric's NVML status and the exporter's own CPU time. A metric NVML refused is omitted, never
written as zero.

**Each window is used once.** The runner's `GpmFileProbe` reads the file on the HTTP-probe thread
(ADR 0007) as nine `gpu.gpm.*.v1` metrics on their own resource (service `nvidia-nvml-gpm`).
When the instance and sequence have not changed since the previous round, every reading is
`unavailable` ("no new GPM window"), so no window is re-reported at a new tick; a restarted
exporter, which counts from 1 again, is told apart by its instance. The first window a run reads
is only a baseline, also `unavailable`: a file left by a stopped exporter looks the same as a live
one, and seeing the file change needs no clock shared with Windows. A missing or unreadable file,
a file without an instance or sequence, and a refused metric are `unavailable` with their reason;
an out-of-range value is `malformed`.

**Windows and runner clocks are not mapped.** A reading describes the exporter's latest window on
its own monotonic clock, ending before the runner's tick by up to one interval plus file latency.
At 250 ms this describes a run and its longer calls, not short steps: a step's window often holds
the activity of the step before it.

Consequences: a WSL2 run that passes `--gpm-file` samples 9 more metrics per round. GPM metrics
are whole-device, every process on it, and not per process. The exporter needs GPU performance
counters open to the user running it (`RmProfilingAdminOnly=0` was set when this was measured;
whether Windows requires it was not tested). A reader can hold the file while the exporter
replaces it; after three attempts that window is dropped and counted (`write_failures_total`).
On a native Linux host DCGM's own profiling fields apply instead, and this exporter is not needed.
Not established: GPM on any other device or driver, the exporter's effect on vLLM timing beyond the
measured sampling overhead, and a clock mapping that would align windows with steps.
