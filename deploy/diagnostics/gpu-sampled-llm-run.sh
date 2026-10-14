#!/usr/bin/env bash
# A01 access probe: run the existing LLM tool loop while nvidia-smi samples the GPU at a
# 250 ms target interval. This is a diagnostic for measurement access, not the AgentWatch
# GPU sampler (planned under A04); its samples are not ingested or step-correlated.
#
# Usage: OLLAMA_URL=http://127.0.0.1:11435 deploy/diagnostics/gpu-sampled-llm-run.sh [NAME]
# Evidence is written to .data/NAME (default a01-gpu-probe) and never overwrites a prior run.
set -u
repo=$(cd "$(dirname "$0")/../.." && pwd)
out="$repo/.data/${1:-a01-gpu-probe}"
interval_ms=250
demo_timeout_s=900
fields=timestamp,index,name,pstate,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm,clocks_event_reasons.active

: "${OLLAMA_URL:?Set OLLAMA_URL to the Docker Ollama endpoint, e.g. http://127.0.0.1:11435}"
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
timeout "$demo_timeout_s" uv run --package agentwatch-worker \
  python -m agentwatch_worker.llm_demo --url "$OLLAMA_URL" --output "$out/llm-demo"
demo_status=$?
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
  echo "llm_demo_exit_status=$demo_status"
} >> "$out/probe-meta.txt"
exit "$demo_status"
