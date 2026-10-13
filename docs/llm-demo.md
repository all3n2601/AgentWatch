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

### Optional NVIDIA GPU and port

`AGENTWATCH_LLM_GPU=1 make llm-up` adds `deploy/compose/compose.gpu.yaml`, which reserves the
host's NVIDIA GPUs for the Ollama service. It needs Docker Desktop with WSL2 GPU support or the
NVIDIA Container Toolkit; without the variable the CPU-only service is unchanged. Only `0`, `1`,
or unset are accepted; other values stop `make llm-up` instead of silently running on CPU.
Docker is expected to refuse the GPU reservation on a host without NVIDIA support; that failure
path has not been exercised because the team's only GPU machine was used for verification.

If another Ollama already listens on 11434, set `OLLAMA_HOST_PORT` (for example 11435). Make
targets derive `OLLAMA_URL` from it; when running the Python module directly, export
`OLLAMA_URL=http://127.0.0.1:11435` as well. The URL must be `http://` or `https://` with a host.

Each run records Ollama's reported model placement from `/api/ps`: after every inference request
(`inference[].model_placement`) and once more after the loop, or when the run fails
(`model_placement`). Each reading is taken after its `llm.chat` span closes, so inference timings
exclude it, but the `agent.tool_loop` span includes these short requests; `read_seconds`
records each one's duration (bounded by a 10 s timeout). Model names are compared in Ollama's
canonical form, so `qwen3` matches the listed `qwen3:latest`.
The Windows GPU laptop setup, its measured signals, and their limits are in
[the GPU measurement access record](gpu-measurement-access.md).

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
  Ollama token counts and nanosecond timing fields, model placement, status and failure details.
  Placement is `measured`, `unavailable` (model unloaded or `/api/ps` unreachable), or
  `malformed` (inconsistent sizes, or a response that is not valid UTF-8 JSON), with a
  UTC `observed_at` timestamp. It never defaults to CPU or GPU, and a failed placement read
  does not change a run's status or replace its recorded error.
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

The demo runs on Docker CPU by default and on an NVIDIA GPU with the optional override. It does
not sample GPU resources, attribute inference contention, or validate vLLM queues, retrieval
contention, L4/GCP performance, or production model quality. Model placement shows where Ollama
loaded the model, not GPU utilization or saturation. The first request after creating the
GPU container was 53–64 s on the verified laptop, for an unexplained reason. Request
wall time includes transport and inference; Ollama's server durations are kept as reported
and are not converted into an invented queue split. The fixed coding tool is a trusted
fixture, not a container security boundary. General live-agent ingestion remains future work.

## Verified on an NVIDIA GPU — October 13, 2026

RTX 5070 Laptop GPU (8 GiB) through Docker Desktop and WSL2, same image and model ID.
Run `7afc596d-131f-4ba6-a374-4ffad0ccce01` used the committed override with port 11435 and passed;
Ollama reported 100% of the model in VRAM. Later reruns completed the tool loop with the same
placement but were inconclusive on the coding baseline gate (`dd25e410-ad19-48ff-af93-461d9ae0ca6f`,
CV 19.3%; `ec908384-e9bc-45ad-9788-b22a3ac0d668`, CV 17.3%). One rerun failed when the model
repeated `run_tests` and the dispatcher rejected it; it still recorded per-request and failure
placement. None were published to dashboard history. Details and unavailable signals are in
[the access record](gpu-measurement-access.md).

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
