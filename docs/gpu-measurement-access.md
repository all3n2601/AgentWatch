# GPU and Linux measurement access (A01)

Status: verified on one laptop, October 13, 2026. Task A01, owner Benoy, reviewer Akshayaa.
This records what the team's only GPU-equipped machine can measure today. It does not establish
NVIDIA L4, GKE, or GCP readiness, and it does not add GPU telemetry to AgentWatch ingestion.
The AgentWatch GPU sampler is planned separately under A04.

## Hardware and runtime identity

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU, 8,151 MiB, CUDA compute capability 12.0 |
| Driver | 617.42 on Windows (Ollama reports CUDA driver 13.4) |
| Host OS | Windows 11 Pro 10.0.26200; Docker Desktop with Engine 29.8.2, Compose 5.5.1 |
| Linux environment | WSL2 distro `Ubuntu` (Ubuntu 26.04.1 LTS), kernel `6.18.54.1-microsoft-standard-WSL2`, 16 vCPUs |
| Inference server | Repository-pinned `ollama/ollama@sha256:292ee794…` (0.35.1), CUDA library `cuda_v13` |
| Model | `qwen3:1.7b`, digest `8f68893c685c…`, `Q4_K_M`, context 4,096; 29/29 layers on GPU |

Docker GPU passthrough works (`docker run --gpus all … nvidia-smi`). vLLM was not installed or
tested; its support for this GPU remains unverified. The plan's 7B FP16 model would not fit in
8 GiB; a quantized or smaller model is required on this machine.

## Reproducible setup

Run Docker commands from Windows and experiments inside `wsl -d Ubuntu`. The default WSL distro on
this machine is `docker-desktop`, which has no repository tooling. A native Windows Ollama app
already owns `127.0.0.1:11434` and is not reachable from WSL, so the Docker service is moved to
11435. Leave that app and any existing `.env` unchanged.

```powershell
# Windows PowerShell, from the repository root
$env:AGENTWATCH_LLM_GPU = "1"; $env:OLLAMA_HOST_PORT = "11435"
make llm-up
```

```sh
# Ubuntu (wsl -d Ubuntu), from the repository root
export OLLAMA_URL=http://127.0.0.1:11435
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/agentwatch" UV_LINK_MODE=copy
bash deploy/diagnostics/measurement-access.sh
# Since 2026-10-20 the probe drives the agent runner, which reports to the live API: start
# PostgreSQL (`make infra-up` from Windows) and, in another Ubuntu terminal, the worker API:
#   uv run --package agentwatch-worker uvicorn agentwatch_worker.app:app --port 8090
# The probe checks GET /readyz first. Set AGENTWATCH_API_URL if the API is elsewhere.
bash deploy/diagnostics/gpu-sampled-llm-run.sh a01-my-run
python3 -I deploy/diagnostics/summarize_gpu_run.py .data/a01-my-run
```

`UV_PROJECT_ENVIRONMENT` matters on a Windows checkout mounted in WSL: without it, `uv` in Ubuntu
deletes and recreates the repository's Windows `.venv`. The diagnostic scripts set it when unset.
Without `AGENTWATCH_LLM_GPU=1`, `make llm-up` starts the unchanged CPU-only service on 11434.
`make` is not installed in this Ubuntu distro; run `uv` commands directly there. On Windows,
`make` runs recipes through `cmd.exe` from PowerShell, so its targets avoid shell-only defaults.

## Supported metrics

| Source | Signal | Observed |
|---|---|---|
| `nvidia-smi` in WSL (`/usr/lib/wsl/lib/nvidia-smi`, not on `PATH`) | utilization.gpu, utilization.memory (%), memory.used/total (MiB), power.draw (W), temperature (°C), SM clock (MHz), pstate, clock event reasons | All present in every sample; idle 408 MiB, 4.2 W, 180 MHz, P8 |
| Ollama `/api/ps` | `size`, `size_vram`, digest, quantization, context length | 1,702,593,821 of 1,702,593,821 bytes in VRAM |
| Ollama `/api/chat` response | load, prompt-eval, eval, and total durations (ns); prompt and eval token counts | Present on every request |
| `/proc/pressure/{cpu,memory,io}` | PSI `some`/`full` avg10/60/300 and totals | Present |
| cgroup v2 (`/sys/fs/cgroup`) | controllers cpuset, cpu, io, memory, hugetlb, pids, rdma; `cpu.stat`, `cpu.pressure`, `memory.current` | Readable for the process cgroup |

`deploy/diagnostics/measurement-access.sh` prints the same inventory on another machine.

## Unavailable signals and measurement boundaries

- DCGM (`dcgmi`, `nv-hostengine`) is not installed; DCGM support on WSL2 is unconfirmed.
- vLLM queue and request metrics are unavailable because vLLM is not installed.
- Per-process GPU accounting is unavailable. In WSL, `--query-compute-apps` lists a PID with
  process name `[Not Found]` and memory `[N/A]`; Windows `nvidia-smi` names host processes but
  also reports their GPU memory as `[N/A]`.
- `nvidia-smi` timestamps are local time without an offset; the probe records the offset (−04:00).
- No systemd: every process is in the single `/non-systemd` cgroup, so there are no per-service
  cgroups. `cpu.max` is `max 100000` (no limit), so throttling counters stay zero.
- PSI and cgroups describe the WSL2 VM, not the Windows host. Windows load can affect timings
  without appearing as VM pressure; the CPU baseline probe records Windows load separately.
- NVML utilization is averaged over its sampling period, so it cannot resolve short requests and
  does not establish GPU saturation. Ollama placement is a memory split, not utilization.
- The LLM demo's CPU sampler covers only the tool step (20–22 samples per run), not model calls.
- Before this change the demo hard-coded "Docker CPU inference" limitations and the span attribute
  `agentwatch.resource_class=inference_cpu`. The two preserved runs below carry those incorrect
  labels; Ollama logs and `ollama ps` showed 100% GPU placement for both. New runs record placement
  from `/api/ps` after each request and after the loop or a failure (see
  [the LLM guide](llm-demo.md)). That describes the demo as it was on 2026-10-13; since
  2026-10-20 the agent runner records placement after each model call and after a failed call. The two committed-config runs predate the per-request reading
  and store a single `model_placement` with `observed_at_unix_ns` instead of `observed_at`.

## Recorded runs

All runs used the same model and request settings. The GPU sampler targeted 250 ms; the CPU
experiment applies its unchanged 15% baseline execution-variability gate. The final-code run
repeated verification after the last `llm_demo.py` edits: its tool loop and placement evidence
worked, and its coding experiment failed the variability gate. The two review-fix runs verified
the per-request and failure-path placement readings. In the first, the model requested
`run_tests` again after receiving a failed result, and the dispatcher rejected the repeat; that
run is recorded as failed, with placement measured for both requests and after the failure.
No run was repeated to obtain a pass.

| Run | Run ID | Result | First request | Second request | GPU samples, interval | Queue clean → busy |
|---|---|---|---|---|---|---|
| Cold start, scratch override (22:19 UTC) | `6df93f49-389e-4593-893c-8939826a0823` | passed, CV 11.4% | 53.46 s: load 26.19 s, prompt 20.33 s/208 tok, eval 6.93 s/16 tok | 1.50 s: 231 tok in 1.34 s | 248, median 266 ms, max 347 ms | 0.000907 → 1.116 s |
| Model loaded, scratch override (22:21 UTC) | `5e70a263-cdad-429c-9608-9d6c35586ef7` | **inconclusive, CV 21.6%** | 2.36 s: prompt 2.05 s/208 tok | 1.37 s: 192 tok in 1.18 s | 60, median 269 ms, max 307 ms | 0.000432 → 1.134 s |
| Cold start, committed config (22:46 UTC) | `7afc596d-131f-4ba6-a374-4ffad0ccce01` | passed, CV 11.5% | 64.25 s: load 30.78 s, prompt 24.70 s/208 tok, eval 8.76 s/16 tok | 0.61 s: 63 tok in 0.48 s | 287, median 268 ms, max 313 ms | 0.000370 → 1.142 s |
| Model loaded, final code (23:01 UTC) | `dd25e410-ad19-48ff-af93-461d9ae0ca6f` | **inconclusive, CV 19.3%** | 0.36 s: prompt 0.19 s/208 tok | 1.17 s: 175 tok in 1.03 s | 50, median 269 ms, max 300 ms | 0.000780 → 1.028 s |
| Runner reload, review fixes (23:30 UTC) | coding run `ae1ea19a-1a7c-4e65-a19d-e0ca3dec8890` | **failed: repeated tool call rejected**; coding CV 25.0% | 3.60 s: load 3.25 s, prompt 0.14 s/208 tok | 0.21 s: repeated `run_tests` call | 60, median 270 ms, max 291 ms | 0.000642 → 1.005 s |
| Model loaded, review fixes (23:31 UTC) | `ec908384-e9bc-45ad-9788-b22a3ac0d668` | **inconclusive, CV 17.3%** | 2.54 s: prompt 2.27 s/208 tok | 2.68 s: 189 tok in 2.54 s | 64, median 273 ms, max 373 ms | 0.000935 → 1.070 s |

No sample gap exceeded 500 ms. During the five second requests that produced an answer, peak
GPU utilization was 59–81%, with up to 87 W and a 2,662 MHz SM clock; the rejected repeated call
(0.21 s) had one sample at 31%. Memory rose from 378–408 MiB idle to 2,192–2,221 MiB with the
model loaded, depending on the container instance.
These runs were not published to dashboard history. The model's final summaries are not graded:
two of them claim contended execution time increased, while the measurements show it decreased.

## First-request latency diagnostic

`deploy/diagnostics/first_request_timing.py` sent the demo's exact first request with its
imported messages and options. Both protocols were declared in the script before evaluation;
all 36 requests are kept. Since 2026-10-20 the script sends the agent runner's first request
instead (its system prompt, task message, four tool definitions and `num_predict` 512), so
the results below, recorded with the removed demo's single-tool request, are not comparable
with later runs.

- **Tool definitions are not the cause.** After unloading with `keep_alive=0`, cold requests took
  3.44–3.89 s with tools (load 3.05–3.58 s, prompt 0.17–0.20 s for 208 tokens) and 3.66–4.61 s
  without (load 2.57–3.43 s, prompt 0.14–0.19 s for 85 tokens). Warm repeats took 0.10–0.15 s
  with tools and 0.79–1.04 s without, because the untooled answer generates about 120 tokens.
- **Idle time and prompt-cache contents add little.** With the model loaded, requests after
  10 s or 60 s idle took 0.24–0.40 s (prompt 0.12–0.18 s) versus 0.12–0.14 s immediately.
  A different prompt cost 0.99 s once, then 0.15–0.16 s; the demo prompt after it took 0.15–0.19 s.
- **Open limitation.** The 53–64 s first requests occurred only in a newly created container,
  with GPU utilization averaging under 1% and mostly low-clock pstates during that request.
  Restarting only the model runner does not reproduce it, and the cause is unknown. A slow
  first request with the model already loaded occurred twice: 2.05 s and 2.27 s of prompt
  evaluation for 208 tokens, each at the start of an agent run about 10–30 s after a previous
  run. During both, the GPU stayed near idle: mean utilization 6.9% and 0%, SM clock
  180–202 MHz, in P8 apart from one P2 sample. No idle or cache condition in the diagnostic reproduced it, and its cause is also
  unknown; low-resolution utilization does not show where that time was spent.
  Exclude first requests after container creation, and treat first requests of a new run as
  possibly slow, in timing comparisons until explained.

## CPU baseline variability

`deploy/diagnostics/cpu-baseline-attempts.sh 5` ran five declared coding-experiment attempts
at 22:56 UTC on AC power, Windows Balanced power scheme, with Docker Desktop, the idle GPU
Ollama container, and the Windows Ollama app running. Windows CPU before attempts was 12–21%;
WSL load average was about 0.7.

| Attempt | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Execution CV | 42.2% | 12.1% | 14.2% | 22.0% | 15.8% |
| Gate | fail | pass | pass | fail | fail |

A second batch of five ran at 23:29 UTC under the same power scheme and AC power, with Windows
CPU 9–28% before attempts (read through invariant WMI counters). It started unintentionally:
a script check passed an empty attempt count, which selects the documented default of five.
It is kept like any other attempt.

| Attempt | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Execution CV | 24.4% | 10.9% | 14.5% | 18.6% | 21.0% |
| Gate | fail | pass | pass | fail | fail |

Including the six agent runs, 6 of 16 coding experiments on this machine passed the gate
(CV 10.9–14.5% when passing, 15.8–42.2% when failing). Every failure was `stable_baseline`;
clean/busy queue detection was consistent in all sixteen (clean < 0.001 s, busy 0.95–1.27 s).
CPU baseline validation is **inconclusive on this configuration**. Windows background activity
is a hypothesis, not an established cause.

## Evidence locations

Raw evidence is kept out of Git on the GPU laptop, in `.data/` and copied to
`C:\Users\benoy\AgentWatch-evidence\a01-2026-10-13\`:

- `a01-gpu-probe/`, `a01-gpu-probe-warm/`, `a01-verify-committed-config/`,
  `a01-verify-final-code/`, `a01-verify-review-fixes/`, `a01-verify-review-fixes-2/`:
  `agent.json`, agent and coding spans, CPU samples, `gpu-samples.csv`, `probe-meta.txt`
- `a01-first-request-cold-tools-20261013T224933Z/`, `a01-first-request-warm-state-20261013T225150Z/`
- `a01-first-request-cold-tools-20261013T233549Z/`: abort-path check with an unknown model
  (`aborted.json` records the 404; no requests), not part of either protocol's results
- `a01-cpu-baseline-20261013T225613Z/`, `a01-cpu-baseline-20261013T232905Z/`: per-attempt
  results and recorded conditions
- `session-scripts/`: the original ad-hoc probe scripts and scratch Compose override

Request a copy from Benoy; prompts are the fixed demo prompts and contain no secrets.
