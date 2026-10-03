# Docker LLM tool-calling test

This smoke test runs Qwen3 1.7B through Ollama in Docker. The model requests the
`run_tests` function, AgentWatch executes its existing clean/busy coding experiment,
and the actual tool result is sent back to the model for a final answer.
The model server is containerized; the dispatcher and trusted test tool run on the host.

## Run

Start Docker Desktop, then:

```sh
make llm-up
make demo-llm
```

To retain the coding measurements in the existing dashboard, start PostgreSQL and
run the ingestion API and web app in separate terminals:

```sh
make infra-up
make dev-worker
make dev-web
```

Then run:

```sh
make record-llm
```

Visit http://localhost:3000 and select the latest coding run. Downloads of the saved
`results.json` include the agent conversation, tool result, model metrics, and checks.
The dashboard charts show the coding experiment's worker queue measurements.
Open the **Agent** tab for inference timings, token counts, the model/tool timeline,
and expandable conversation evidence. New runs persist their agent spans inside the saved
result; the original coding-step spans remain in the separate coding trace artifact.

`make llm-down` stops the model server without deleting its downloaded weights.
`make infra-down` stops both Compose services and preserves their named volumes.
The optional `llm` profile prevents ordinary `make infra-up` from downloading an LLM.

## Evidence

Each execution replaces `.data/llm-demo/`:

- `agent.json`: exact messages, tool result, final answer, request wall times,
  Ollama token counts and nanosecond timing fields, status and failure details.
- `agent-spans.jsonl`: OpenTelemetry agent loop, tool invocation, and inference spans;
  these are also embedded in `results.json` so the dashboard can display them from history.
- `results.json`, `spans.jsonl`, `samples.jsonl`, `report.html`: existing measured
  coding experiment, linked to the agent by run ID and trace context.

The loop rejects missing tool calls, unknown functions, nonempty arguments, and
repeated calls. It never executes model-provided shell commands. It allows at most
three model requests and bounds each request to 256 generated tokens and a five-minute
timeout. A pass requires an actual tool request, tool result feedback, a nonempty final
answer, and the coding experiment's existing checks. It does not grade the model's
summary for semantic accuracy; inspect the preserved transcript for that.

Invalid arguments receive a tool error so the model can correct its call within the
same three-request budget. Repeated execution is still rejected.

Docker is limited to 5GB for Ollama and binds its API only to localhost. The image is
pinned by digest. The default model is approximately 1.4GB; its observed model ID is
recorded in the verification notes below. Use `AGENTWATCH_LLM_MODEL` to select another
installed tool-capable model and `OLLAMA_URL` to change the server. Make targets and
the Python runner read exported environment variables; they do not load `.env` automatically.

## Limits

This is CPU inference on this Mac's Docker runtime. It does not validate GPU inference,
vLLM queue attribution, retrieval contention, or production model quality. Request
wall time includes transport and inference; Ollama's server durations are kept as reported
and are not converted into an invented queue split. The fixed coding tool is a trusted
fixture, not a container security boundary. General live-agent ingestion remains future work.

## Verified locally — October 2, 2026

Ollama 0.35.1, Qwen3 model ID `8f68893c685c`, ARM64 Docker CPU.
Two complete tool loops were published successfully. Latest run:
`99ca88a6-41c7-4861-a8d4-54fce55fa58c`.

Clean queue: 0.000151 s; busy queue: 1.094709 s. All six coding checks passed.
The model issued one actual tool call, received its result, and produced a final summary.
The web history endpoint returned the run, and the saved artifact retained its conversation.
34 Python tests (including PostgreSQL integration) and 4 web tests passed.
The first attempted call invented a parameter and was rejected before execution;
the dispatcher now supports corrective tool errors and has a regression test for recovery.

Dashboard integration follow-up: run `b5ebc2bd-d964-4cee-9a14-000d4a22cef8` passed,
with four persisted agent spans sharing one trace ID. Its complete loop took 15.49 s:
5.05 s first inference request, 2.59 s test tool, and 7.86 s second inference request.
The Agent view displays all three phases and expandable tool feedback. Earlier runs
without span timestamps retain their request durations and conversation view.
34 Python tests and 8 web tests passed; web lint, type checking, and production build passed.

## References

- [Ollama Docker setup](https://docs.ollama.com/docker)
- [Ollama tool calling](https://docs.ollama.com/capabilities/tool-calling)
- [Qwen3 1.7B model](https://ollama.com/library/qwen3:1.7b)
