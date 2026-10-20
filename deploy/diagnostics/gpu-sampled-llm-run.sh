#!/usr/bin/env bash
# A01 access probe: run the instrumented agent runner (services/agent) while nvidia-smi samples
# the GPU at a 250 ms target interval. This is a diagnostic for measurement access, not the
# AgentWatch GPU sampler (planned under A04); its GPU samples are not ingested or
# step-correlated. The runner reports its own steps to the live API, which must be running.
#
# Usage: OLLAMA_URL=http://127.0.0.1:11435 deploy/diagnostics/gpu-sampled-llm-run.sh [NAME]
# Optional: AGENTWATCH_API_URL (default http://127.0.0.1:8090), AGENT_TASK (default
# fix-mean-empty). Evidence is written to .data/NAME (default a01-gpu-probe) and never
# overwrites a prior run.
set -u
repo=$(cd "$(dirname "$0")/../.." && pwd)
out="$repo/.data/${1:-a01-gpu-probe}"
interval_ms=250
# The runner bounds itself at 900 s; the extra time lets it close the run and clean up.
agent_timeout_s=960
api_url="${AGENTWATCH_API_URL:-http://127.0.0.1:8090}"
task="${AGENT_TASK:-fix-mean-empty}"
fields=timestamp,index,name,pstate,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm,clocks_event_reasons.active

: "${OLLAMA_URL:?Set OLLAMA_URL to the Docker Ollama endpoint, e.g. http://127.0.0.1:11435}"
if ! curl -fsS --max-time 5 "$api_url/readyz" >/dev/null; then
  echo "The live API at $api_url is not ready (GET /readyz failed); start the worker API and PostgreSQL first." >&2
  exit 2
fi
if [ -e "$out" ]; then
  echo "Refusing to overwrite existing evidence: $out" >&2
  exit 2
fi
# On a Windows checkout mounted in WSL, uv would otherwise replace the Windows .venv.
if grep -qi microsoft /proc/version 2>/dev/null && [ -z "${UV_PROJECT_ENVIRONMENT:-}" ]; then
  export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/agentwatch" UV_LINK_MODE=copy
fi
smi=$(command -v nvidia-smi || true)
[ -z "$smi" ] && [ -x /usr/lib/wsl/lib/nvidia-smi ] && smi=/usr/lib/wsl/lib/nvidia-smi

mkdir -p "$out"
cd "$repo" || exit 1
{
  echo "probe_started_utc=$(date -u +%FT%T.%NZ)"
  echo "nvidia_smi_timestamp_utc_offset=$(date +%z) (nvidia-smi timestamps are local time without an offset)"
  echo "sampling_target_interval_ms=$interval_ms"
  echo "ollama_url=$OLLAMA_URL"
  echo "api_url=$api_url"
  echo "agent_task=$task"
  echo "kernel=$(uname -r)"
  echo "nvidia_smi=${smi:-unavailable}"
} > "$out/probe-meta.txt"

sampler_pid=
if [ -n "$smi" ]; then
  "$smi" --query-gpu="$fields" --format=csv -lms "$interval_ms" \
    > "$out/gpu-samples.csv" 2> "$out/gpu-sampler.err" &
  sampler_pid=$!
fi
cleanup() { [ -n "$sampler_pid" ] && kill "$sampler_pid" 2>/dev/null && wait "$sampler_pid" 2>/dev/null; }
trap cleanup EXIT

# Idle GPU readings before the first request and after the loop bracket the run.
sleep 3
timeout "$agent_timeout_s" uv run --package agentwatch-agent \
  python -m agentwatch_agent run --task "$task" --ollama-url "$OLLAMA_URL" \
  --api-url "$api_url" --output-dir "$out/agent-runs"
agent_status=$?
sleep 3

# A sampler that exited early leaves a gap at the end of gpu-samples.csv; say so explicitly.
if [ -z "$sampler_pid" ]; then
  sampler_state="not started (nvidia-smi unavailable)"
elif kill -0 "$sampler_pid" 2>/dev/null; then
  sampler_state="running at end of run"
else
  wait "$sampler_pid"
  sampler_state="exited early with status $? (see gpu-sampler.err)"
  sampler_pid=
fi
{
  echo "probe_finished_utc=$(date -u +%FT%T.%NZ)"
  echo "nvidia_smi_timestamp_utc_offset_end=$(date +%z)"
  echo "gpu_sampler=$sampler_state"
  echo "agent_run_exit_status=$agent_status"
} >> "$out/probe-meta.txt"
exit "$agent_status"
