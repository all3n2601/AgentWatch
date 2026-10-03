"""Bounded local CPU sandbox queue experiment; no cloud or GPU required."""

import argparse
import hashlib
import html
import json
import multiprocessing
import os
import statistics
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from agentwatch_sandbox.coding_task import coding_task
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from agentwatch_worker.profiling import StepMeasurement, attribute_cpu_queue
from agentwatch_worker.sampler import CpuSampler


def cpu_task(iterations: int, marker: str | None = None) -> dict:
    started_ns = time.time_ns()
    cpu_start = time.process_time()
    if marker is not None:
        Path(marker).touch()
    value = b"agentwatch-fixed-workload"
    for _ in range(iterations):
        value = hashlib.sha256(value).digest()
    return {
        "started_ns": started_ns,
        "finished_ns": time.time_ns(),
        "cpu_seconds": time.process_time() - cpu_start,
        "checksum": value.hex(),
    }


def record_step(
    pool, tracer, iterations: int, step_id: str, run_id: str, task=cpu_task
) -> StepMeasurement:
    submitted_ns = time.time_ns()
    name = "sandbox.run_tests" if task is coding_task else "sandbox.cpu_task"
    with tracer.start_as_current_span(name, start_time=submitted_ns) as span:
        result = pool.submit(task, iterations).result(timeout=60)
        step = StepMeasurement(step_id=step_id, submitted_ns=submitted_ns, **result)
        span.set_attributes(
            {
                "agentwatch.run_id": run_id,
                "agentwatch.step_id": step_id,
                "agentwatch.resource_class": "cpu_sandbox",
                "agentwatch.workload.iterations": iterations,
                "agentwatch.queue_seconds": step.queue_seconds,
                "agentwatch.execution_seconds": step.execution_seconds,
                "agentwatch.cpu_seconds": step.cpu_seconds,
                "agentwatch.tool": name,
            }
        )
        span.add_event("worker.started", timestamp=step.started_ns)
        span.add_event("worker.finished", timestamp=step.finished_ns)
        return step


def write_report(result: dict, path: Path) -> None:
    cards = []
    for label, step in [("Clean run", result["clean"]), ("Busy worker", result["contended"])]:
        width = step["queue_fraction"] * 100
        telemetry = step.get("telemetry", {})
        pressure = telemetry.get("max_cpu_pressure_avg10_percent")
        pressure_text = f"{pressure:.2f}%" if pressure is not None else "unavailable"
        cards.append(f"""<article><h2>{label}</h2>
<p class="resource">{html.escape(step["resource"])}</p>
<div class="bar"><div class="queue" style="width:{width:.2f}%"></div></div>
<p><b>{step["duration_seconds"]:.3f}s</b> total ·
{step["queue_seconds"]:.3f}s queued · {step["execution_seconds"]:.3f}s executing</p>
<p>{width:.1f}% of this step was spent in the worker queue.</p>
<p class="muted">CPU pressure: {pressure_text} ·
{telemetry.get("sample_count", 0)} samples in this step</p></article>""")
    passed = result["checks"]["passed"]
    path.write_text(
        f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AgentWatch — CPU queue experiment</title><style>
body{{font:17px system-ui;background:#101827;color:#e6edf7;margin:0;padding:48px;
max-width:1000px;margin:auto}}h1{{font-size:40px}}.intro,p{{line-height:1.6}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:24px}}
article{{background:#1c293c;padding:24px;border-radius:16px}}.resource{{color:#80ded1}}
.bar{{height:28px;background:#45b7a8;border-radius:8px;overflow:hidden}}
.queue{{height:100%;background:#f1b654}}a{{color:#80ded1}}.muted{{color:#b8c5d9}}
</style><h1>AgentWatch</h1><p class="intro">A real task. One CPU worker.
Measured waiting when that worker is busy.</p>
<p>Workload: {html.escape(result["experiment"]["workload"])}</p><p><b>
{"Experiment passed" if passed else "Inconclusive — inspect the evidence"}</b></p>
<div class="cards">{"".join(cards)}</div>
<p>Bar: <span style="color:#f1b654">queue time</span> +
<span style="color:#45b7a8">execution wall time</span>.</p>
<p class="muted">Attribution uses measured enqueue/start timestamps. This demonstrates
sandbox worker capacity contention. It does not measure host CPU saturation or distinguish
all working time from scheduling delays. No learned model or calibrated confidence is used.</p>
<p class="muted">Baseline execution coefficient of variation:
{result["baseline"]["execution_cv"]:.1%}; queue threshold:
{result["baseline"]["queue_threshold_seconds"]:.3f}s.</p>
<p><a href="results.json">Experiment evidence</a> ·
<a href="spans.jsonl">OpenTelemetry spans</a> ·
<a href="samples.jsonl">CPU samples</a></p></html>""",
        encoding="utf-8",
    )


def run_demo(
    output: Path, iterations: int = 600_000, repeats: int = 5, workload: str = "cpu"
) -> dict:
    if iterations < 1 or repeats < 3:
        raise ValueError("Use positive iterations and at least three baseline repeats")
    output.mkdir(parents=True, exist_ok=True)
    run_id = str(uuid.uuid4())
    if workload not in ("cpu", "coding"):
        raise ValueError("Unknown workload")
    task = coding_task if workload == "coding" else cpu_task
    sampler = CpuSampler()
    provider = TracerProvider()
    with (output / "spans.jsonl").open("w", encoding="utf-8") as stream:
        provider.add_span_processor(
            SimpleSpanProcessor(
                ConsoleSpanExporter(
                    out=stream, formatter=lambda span: json.dumps(json.loads(span.to_json())) + "\n"
                )
            )
        )
        tracer = provider.get_tracer("agentwatch.cpu-demo", "0.1.0")
        # Spawn works on macOS and Linux. Warm up before measuring process startup.
        try:
            with (
                sampler,
                ProcessPoolExecutor(
                    max_workers=1, mp_context=multiprocessing.get_context("spawn")
                ) as pool,
            ):
                pool.submit(task, iterations).result(timeout=60)
                baselines = [
                    record_step(pool, tracer, iterations, f"baseline-{i}", run_id, task)
                    for i in range(repeats)
                ]
                threshold = max(0.01, max(s.queue_seconds for s in baselines) * 3)
                clean = record_step(pool, tracer, iterations, "clean", run_id, task)
                marker = output / f"busy-{run_id}"
                try:
                    blocker = pool.submit(cpu_task, iterations * 8, str(marker.resolve()))
                    deadline = time.monotonic() + 15
                    while not marker.exists():
                        if blocker.done():
                            blocker.result()
                            raise RuntimeError("Competing task ended before start was observed")
                        if time.monotonic() >= deadline:
                            raise TimeoutError("Competing CPU task did not start")
                        time.sleep(0.005)
                    contended = record_step(pool, tracer, iterations, "contended", run_id, task)
                    blocker.result(timeout=60)
                finally:
                    marker.unlink(missing_ok=True)
        finally:
            provider.shutdown()

    durations = [s.execution_seconds for s in baselines]
    cv = statistics.stdev(durations) / statistics.mean(durations)
    clean_result = attribute_cpu_queue(clean, threshold)
    contended_result = attribute_cpu_queue(contended, threshold)
    for step in (clean_result, contended_result):
        step["telemetry"] = sampler.summarize(step["submitted_ns"], step["finished_ns"])
    with (output / "samples.jsonl").open("w", encoding="utf-8") as stream:
        for sample in sampler.samples:
            stream.write(json.dumps(sample) + "\n")
    checks = {
        "same_workload_output": clean.checksum == contended.checksum,
        "stable_baseline": cv < 0.15,
        "clean_classified_none": clean_result["resource"] == "none",
        "contention_detected": contended_result["resource"] == "cpu_sandbox",
        "queue_increased": contended.queue_seconds > clean.queue_seconds + threshold,
    }
    result = {
        "run_id": run_id,
        "experiment": {
            "worker_capacity": 1,
            "iterations": iterations,
            "competing_iterations": iterations * 8,
            "baseline_repeats": repeats,
            "workload": "fixed Python project: run three tests"
            if workload == "coding"
            else "fixed SHA-256 CPU task",
            "tests_per_step": 3 if workload == "coding" else None,
        },
        "baseline": {
            "execution_cv": cv,
            "queue_threshold_seconds": threshold,
            "steps": [attribute_cpu_queue(s, threshold) for s in baselines],
        },
        "clean": clean_result,
        "contended": contended_result,
        "checks": {**checks, "passed": all(checks.values())},
        "limitations": [
            "CPU worker queue only; not host saturation",
            "Execution wall time is not pure working time",
            "Rule baseline only; no ML accuracy claim",
            "Linux CPU pressure and cgroup metrics are unavailable on macOS",
            "Coding workload is a trusted test tool fixture, not an LLM agent",
        ],
    }
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_report(result, output / "report.html")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".data/cpu-demo"))
    parser.add_argument("--iterations", type=int, default=600_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--workload", choices=["cpu", "coding"], default="cpu")
    parser.add_argument(
        "--publish", action="store_true", help="Save run history through the ingestion API"
    )
    args = parser.parse_args()
    result = run_demo(args.output.resolve(), args.iterations, args.repeats, args.workload)
    for name in ("clean", "contended"):
        step = result[name]
        print(
            f"{name:10s} total={step['duration_seconds']:.3f}s "
            f"queue={step['queue_seconds']:.3f}s resource={step['resource']}"
        )
    print(f"Report: {(args.output / 'report.html').resolve()}")
    if args.publish:
        from agentwatch_worker.publish import publish

        saved = publish(
            args.output.resolve(),
            args.workload,
            os.environ.get("AGENTWATCH_API_URL", "http://127.0.0.1:8090"),
        )
        print(f"Saved run history: {saved['run_id']}")
    if not result["checks"]["passed"]:
        print("Experiment inconclusive: inspect checks in results.json")
        raise SystemExit(1)
    print("Experiment passed: same task, stable baseline, CPU queue contention detected.")


if __name__ == "__main__":
    main()
