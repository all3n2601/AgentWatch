# Docker LLM

Ollama in Docker serves Qwen3 1.7B to the instrumented agent runner
([`services/agent`](../services/agent/README.md)). The model server is containerized; the
runner, its dispatcher and the trusted test tool run on the host.

## Run

Start Docker Desktop, then start PostgreSQL and the ingestion API (the runner reports its
steps live), and the model server:

```sh
make infra-up
make dev-worker      # in its own terminal
make llm-up
make agent-run
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

The runner records Ollama's reported model placement from `/api/ps` after every model call and
after a failed call (`model_calls[].model_placement` in `agent-evidence.json`). Each reading is
taken after its model-call step closes, so step timings exclude it; it is skipped or abandoned
(recorded as `unavailable`) once cancellation is requested or the run deadline passes. Model
names are compared in Ollama's canonical form, so `qwen3` matches the listed `qwen3:latest`.
The Windows GPU laptop setup, its measured signals, and their limits are in
[the GPU measurement access record](gpu-measurement-access.md).

`make llm-down` stops the model server without deleting its downloaded weights.
`make infra-down` stops both Compose services and preserves their named volumes.
The optional `llm` profile prevents ordinary `make infra-up` from downloading an LLM.

## Removed tool-calling demo

Until 2026-10-20, `make demo-llm` and `make record-llm` ran a single-tool smoke test
(`agentwatch_worker.llm_demo`) that invoked the fixed coding experiment. The agent runner
replaced it, and its shared Ollama helpers moved to `agentwatch_worker.ollama`. Runs it saved
remain in dashboard history, and the **Agent** tab still displays their conversation,
inference timings and placement. New agent runs report to the live API instead; their
dashboard view is task F02. The verification notes below describe the demo as it was.

## Configuration

Docker is limited to 5GB for Ollama and binds its API only to localhost. The image is
pinned by digest. The default model is approximately 1.4GB; its observed model ID is
recorded in the verification notes below. Use `AGENTWATCH_LLM_MODEL` to select another
installed tool-capable model and `OLLAMA_URL` to change the server. Make targets and
the runner read exported environment variables; they do not load `.env` automatically.

## Limits

The model server runs on Docker CPU by default and on an NVIDIA GPU with the optional override. It does
not sample GPU resources, attribute inference contention, or validate vLLM queues, retrieval
contention, L4/GCP performance, or production model quality. Model placement shows where Ollama
loaded the model, not GPU utilization or saturation. The first request after creating the
GPU container was 53–64 s on the verified laptop, for an unexplained reason. Request
wall time includes transport and inference; Ollama's server durations are kept as reported
and are not converted into an invented queue split. The test tool is a trusted fixture, not
a container security boundary.

## Historical: demo verified on an NVIDIA GPU — October 13, 2026

RTX 5070 Laptop GPU (8 GiB) through Docker Desktop and WSL2, same image and model ID.
Run `7afc596d-131f-4ba6-a374-4ffad0ccce01` used the committed override with port 11435 and passed;
Ollama reported 100% of the model in VRAM. Later reruns completed the tool loop with the same
placement but were inconclusive on the coding baseline gate (`dd25e410-ad19-48ff-af93-461d9ae0ca6f`,
CV 19.3%; `ec908384-e9bc-45ad-9788-b22a3ac0d668`, CV 17.3%). One rerun failed when the model
repeated `run_tests` and the dispatcher rejected it; it still recorded per-request and failure
placement. None were published to dashboard history. Details and unavailable signals are in
[the access record](gpu-measurement-access.md).

## Historical: demo verified locally — October 2, 2026

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
