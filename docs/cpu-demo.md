# First proof: CPU sandbox queue attribution

Run `make demo-cpu` from the repository root with `uv` installed. The Python workspace installs
the required dependencies automatically. Open `.data/cpu-demo/report.html` afterward.

## What happens

1. Start one worker process and warm it up.
2. Execute a fixed SHA-256 workload five times to measure baseline timing and dispatch noise.
3. Execute a clean evaluation task.
4. Submit an eight-times-larger competing CPU task; confirm it started before submitting the
   same evaluation task again.
5. Record enqueue, worker start, worker finish, process CPU time, and output checksum.
6. Attribute delays exceeding the queue threshold to `cpu_sandbox` and export the report.

The threshold is the larger of 10 milliseconds and three times the largest baseline queue delay.
It is a simple rule baseline, not a trained model. The attribution function receives only the
step measurements and threshold, not whether an injector was active.

## Evidence and acceptance

- `results.json`: baseline repetitions, clean/contended measurements, attribution, checks, and
  experiment configuration. Injection configuration is kept separate from step features.
- `spans.jsonl`: one OpenTelemetry span per measured task, carrying run/step IDs, resource class,
  workload size, queue/execution/CPU measurements, and worker start/finish events.
- `report.html`: visual comparison of queue time and execution wall time.
- `samples.jsonl`: timestamped Linux CPU pressure and cgroup v2 CPU counters, or explicit
  unavailability records on unsupported systems. Summaries are joined to each evaluation step.

The process exits with code 1 if the baseline coefficient of variation reaches 15%, outputs differ,
the clean task is falsely classified, or the competing task fails to produce the required queue
increase. Inspect the saved results before rerunning a noisy experiment. The default workload
runs for a few seconds and uses one child worker. It shuts the worker down after the experiment.

To change workload size or baseline repeats:

```sh
uv run --package agentwatch-worker python -m agentwatch_worker.demo --iterations 1000000 --repeats 7
```

## Boundaries and next milestone

This validates a local CPU worker queue and the path from measurements to spans, attribution,
and a report. It is not yet a coding agent, a sandbox security boundary, a hardware saturation
profiler, or a classifier spanning CPU, inference, and retrieval.

Queue time is directly observed here. Execution wall time can contain OS scheduling delays and
must not be presented as pure useful work. The classifier names the worker resource class,
not a proven root cause of host CPU saturation.

## Coding tool and resource sampler

`make demo-coding` invokes a fixed project's Python tests in a subprocess through the same
worker. Its three checks cover empty inputs, ordinary aggregation, and CPU-heavy processing.
Subprocess failures propagate rather than producing a successful measurement. Child-process
CPU usage is included in the step's CPU time on supported Unix systems.

The read-only sampler reads `/proc/pressure/cpu` and the profiler's cgroup v2 `cpu.stat`
every 250 ms. Summaries include the maximum observed PSI `some avg10` and, when two valid
counter samples exist inside the span window, an observed throttled-time delta. Short spans
may have no samples. The ten-second PSI average is contextual evidence, not instantaneous
per-step pressure; cgroup metrics cover the shared cgroup, not only the measured subprocess.
Queue rules do not yet use these signals. macOS records unavailable metrics explicitly.

This coding fixture is a trusted local tool, not an LLM agent or a secure execution environment.

Next, integrate an agent's tool dispatcher, then compare a hardware-saturated CPU workload
with a long task running without contention on Linux. Only then build the replay dataset and
learned attribution model.
