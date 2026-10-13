"""Real Ollama tool-calling smoke test using the trusted coding experiment."""

import argparse
import http.client
import json
import os
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from agentwatch_worker.demo import run_demo
from agentwatch_worker.publish import publish

TOOL = {
    "type": "function",
    "function": {
        "name": "run_tests",
        "description": (
            "Run three real Python tests under clean and busy-worker conditions. "
            "Return measured queue and execution times and verification checks."
        ),
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
}
INITIAL_MESSAGES = (
    {
        "role": "system",
        "content": "Use run_tests exactly once with empty arguments {}. It takes no parameters. "
        "Report its actual checks and timings. "
        "Do not invent results. After the tool returns, give a concise final summary.",
    },
    {
        "role": "user",
        "content": "Run the project's tests and check whether AgentWatch detects waiting "
        "when its single coding worker is busy. Use the run_tests tool now.",
    },
)
CHAT_OPTIONS = {"temperature": 0, "num_ctx": 4096, "num_predict": 256}


def ollama_chat(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url.rstrip("/") + "/api/chat",
        data=json.dumps(payload, allow_nan=False).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        result = json.load(response)
    if result.get("error") or not isinstance(result.get("message"), dict):
        raise RuntimeError(f"Invalid Ollama response: {result}")
    return result


def ollama_loaded_models(url: str) -> dict:
    with urllib.request.urlopen(url.rstrip("/") + "/api/ps", timeout=10) as response:
        return json.load(response)


def model_placement(loaded: dict, model: str) -> dict:
    """Summarize where Ollama reports the model's memory, from one /api/ps response.

    Ollama's own "100% GPU" display is size_vram / size. This is model placement,
    not GPU utilization or saturation.
    """
    source = {"source": "ollama /api/ps"}
    models = loaded.get("models") if isinstance(loaded, dict) else None
    if not isinstance(models, list):
        return source | {"status": "malformed", "reason": "response has no models list"}
    entry = next(
        (
            item
            for item in models
            if isinstance(item, dict) and model in (item.get("name"), item.get("model"))
        ),
        None,
    )
    if entry is None:
        return source | {"status": "unavailable", "reason": "model is not currently loaded"}
    size, size_vram = entry.get("size"), entry.get("size_vram")
    valid = all(
        isinstance(value, int) and not isinstance(value, bool) for value in (size, size_vram)
    )
    if not valid or size <= 0 or not 0 <= size_vram <= size:
        return source | {
            "status": "malformed",
            "reason": "size/size_vram are not consistent byte counts",
        }
    details = entry.get("details") if isinstance(entry.get("details"), dict) else {}
    return source | {
        "status": "measured",
        "size_bytes": size,
        "size_vram_bytes": size_vram,
        "gpu_memory_fraction": size_vram / size,
        "digest": entry.get("digest"),
        "quantization_level": details.get("quantization_level"),
        "context_length": entry.get("context_length"),
    }


def observe_model_placement(fetch_loaded: Callable[[], dict], model: str) -> dict:
    observed_at = time.time_ns()
    try:
        placement = model_placement(fetch_loaded(), model)
    # Transport and decoding failures only; the measured tool loop has already finished,
    # so a missing placement reading must not turn a completed run into a failure.
    except (OSError, http.client.HTTPException, json.JSONDecodeError) as error:
        placement = {
            "source": "ollama /api/ps",
            "status": "unavailable",
            "reason": f"placement request failed: {error}",
        }
    return placement | {"observed_at_unix_ns": observed_at}


def agent_loop(chat, execute, tracer, model: str, evidence: dict) -> dict:
    messages = evidence["messages"] = [dict(message) for message in INITIAL_MESSAGES]
    result = None
    for index in range(3):
        started = time.time_ns()
        with tracer.start_as_current_span("llm.chat") as span:
            span.set_attributes(
                {"gen_ai.request.model": model, "agentwatch.resource_class": "inference"}
            )
            response = chat(
                {
                    "model": model,
                    "messages": messages,
                    "tools": [TOOL],
                    "stream": False,
                    "think": False,
                    "options": dict(CHAT_OPTIONS),
                }
            )
            metrics = {
                key: response[key]
                for key in (
                    "total_duration",
                    "load_duration",
                    "prompt_eval_count",
                    "prompt_eval_duration",
                    "eval_count",
                    "eval_duration",
                    "done_reason",
                )
                if key in response
            }
            span.set_attributes({f"ollama.{key}": value for key, value in metrics.items()})
            evidence.setdefault("inference", []).append(
                {
                    "request_index": index,
                    "wall_seconds": (time.time_ns() - started) / 1e9,
                    "metrics": metrics,
                }
            )
        message = response["message"]
        messages.append(message)
        calls = message.get("tool_calls", [])
        if calls:
            if result is not None or len(calls) != 1:
                raise ValueError("Expected exactly one tool call across the run")
            function = calls[0].get("function", {})
            if function.get("name") != "run_tests" or function.get("arguments") != {}:
                evidence.setdefault("rejected_calls", []).append(function)
                messages.append(
                    {
                        "role": "tool",
                        "tool_name": function.get("name", "unknown"),
                        "content": json.dumps(
                            {
                                "error": "No tool executed. Only run_tests exists. "
                                "Call run_tests with empty arguments {} and no parameters."
                            }
                        ),
                    }
                )
                continue
            print(
                "Model requested run_tests; executing the measured coding experiment.", flush=True
            )
            with tracer.start_as_current_span("tool.run_tests") as tool_span:
                result = execute()
                tool_span.set_attributes(
                    {
                        "agentwatch.run_id": result["run_id"],
                        "agentwatch.resource_class": "cpu_sandbox",
                    }
                )
            span_summary = {
                "run_id": result["run_id"],
                "tests_per_step": 3,
                "checks": result["checks"],
                "clean": {
                    key: result["clean"][key]
                    for key in (
                        "queue_seconds",
                        "execution_seconds",
                        "resource",
                    )
                },
                "contended": {
                    key: result["contended"][key]
                    for key in (
                        "queue_seconds",
                        "execution_seconds",
                        "resource",
                    )
                },
            }
            evidence["tool_result"] = span_summary
            messages.append(
                {"role": "tool", "tool_name": "run_tests", "content": json.dumps(span_summary)}
            )
        else:
            if result is None or not message.get("content", "").strip():
                raise ValueError("Model did not complete the tool-call/result/final-answer loop")
            evidence["final_answer"] = message["content"]
            evidence["checks"] = {
                "real_tool_requested": True,
                "tool_result_returned": True,
                "final_answer_received": True,
                "coding_experiment_passed": result["checks"]["passed"],
            }
            return result
    raise ValueError("Model exceeded the bounded agent loop")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("AGENTWATCH_LLM_MODEL", "qwen3:1.7b"))
    parser.add_argument("--url", default=os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434"))
    parser.add_argument("--output", type=Path, default=Path(".data/llm-demo"))
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    evidence = {
        "model": args.model,
        "endpoint": args.url,
        "status": "running",
        "limitations": [
            "Model placement is Ollama's reported memory split, not GPU utilization or saturation",
            "No GPU samples or inference queue attribution are recorded by this demo",
            "Inference wall time includes transport and server work; no inferred queue split",
            "Fixed trusted tool, not arbitrary agent code execution",
        ],
    }
    provider = TracerProvider()
    try:
        with (output / "agent-spans.jsonl").open("w") as stream:
            provider.add_span_processor(
                SimpleSpanProcessor(
                    ConsoleSpanExporter(
                        out=stream,
                        formatter=lambda span: json.dumps(json.loads(span.to_json())) + "\n",
                    )
                )
            )
            tracer = provider.get_tracer("agentwatch.ollama-demo")
            try:
                with tracer.start_as_current_span("agent.tool_loop") as span:
                    result = agent_loop(
                        lambda payload: ollama_chat(args.url, payload),
                        lambda: run_demo(output, workload="coding"),
                        tracer,
                        args.model,
                        evidence,
                    )
                    span.set_attribute("agentwatch.run_id", result["run_id"])
                    evidence["run_id"] = result["run_id"]
                    evidence["status"] = (
                        "passed" if all(evidence["checks"].values()) else "inconclusive"
                    )
                    result["limitations"] = [
                        item for item in result["limitations"] if "not an LLM agent" not in item
                    ] + evidence["limitations"]
                    result["agent"] = evidence
            finally:
                provider.shutdown()
        # Read while the model is still within Ollama's keep-alive window.
        evidence["model_placement"] = observe_model_placement(
            lambda: ollama_loaded_models(args.url), args.model
        )
        evidence["spans"] = [
            json.loads(line)
            for line in (output / "agent-spans.jsonl").read_text().splitlines()
            if line.strip()
        ]
        (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        if args.publish:
            saved = publish(
                output, "coding", os.environ.get("AGENTWATCH_API_URL", "http://127.0.0.1:8090")
            )
            evidence["persisted_run_id"] = saved["run_id"]
            print(f"Saved dashboard run: {saved['run_id']}")
    except Exception as error:
        evidence["status"] = "failed"
        evidence["error"] = str(error)
        raise
    finally:
        (output / "agent.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(evidence["final_answer"])
    print(f"Agent evidence: {output / 'agent.json'}")
    if evidence["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
