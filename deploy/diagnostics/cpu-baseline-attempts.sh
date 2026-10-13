#!/usr/bin/env bash
# A01 CPU baseline check: run the coding experiment a declared number of times and keep
# every attempt. Each attempt applies the experiment's unchanged 15% execution-variability
# gate; this script reports how many attempts passed and never replaces a failed attempt.
# On WSL it also records Windows power and CPU load, which the VM's PSI cannot see.
#
# Usage: deploy/diagnostics/cpu-baseline-attempts.sh [ATTEMPTS]   (default 5)
set -u
attempts="${1:-5}"
repo=$(cd "$(dirname "$0")/../.." && pwd)
out="$repo/.data/a01-cpu-baseline-$(date -u +%Y%m%dT%H%M%SZ)"
attempt_timeout_s=300
on_wsl=$(grep -qi microsoft /proc/version 2>/dev/null && echo yes || echo no)
if [ "$on_wsl" = yes ] && [ -z "${UV_PROJECT_ENVIRONMENT:-}" ]; then
  # On a Windows checkout mounted in WSL, uv would otherwise replace the Windows .venv.
  export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/agentwatch" UV_LINK_MODE=copy
fi

conditions() {
  echo "observed_utc=$(date -u +%FT%T.%NZ)"
  echo "wsl_loadavg=$(cut -d' ' -f1-3 /proc/loadavg)"
  echo "wsl_cpu_pressure=$(head -1 /proc/pressure/cpu 2>/dev/null || echo unavailable)"
  if [ "$on_wsl" = yes ] && command -v powershell.exe >/dev/null; then
    timeout 30 powershell.exe -NoProfile -NonInteractive -Command '
      $battery = Get-CimInstance Win32_Battery
      $status = if ($battery) { $battery.BatteryStatus } else { "no battery" }
      $load = (Get-Counter "\Processor(_Total)\% Processor Time" -SampleInterval 1 -MaxSamples 3).CounterSamples |
        Measure-Object CookedValue -Average
      "windows_battery_status=$status (2 = on AC power, 1 = discharging)"
      "windows_power_scheme=$(powercfg /getactivescheme)"
      "windows_cpu_percent_3s_mean={0:N1}" -f $load.Average' 2>&1 | tr -d '\r'
  else
    echo "windows_conditions=unavailable (not WSL)"
  fi
}

mkdir -p "$out"
cd "$repo" || exit 1
{
  echo "attempts_declared=$attempts"
  echo "workload=coding (python -m agentwatch_worker.demo --workload coding)"
  echo "gate=execution CV < 15% per attempt, as implemented by the experiment"
  echo "kernel=$(uname -r) cpus=$(nproc)"
} > "$out/protocol.txt"

passed=0
for attempt in $(seq 1 "$attempts"); do
  dir="$out/attempt-$attempt"
  mkdir -p "$dir"
  conditions > "$dir/conditions-before.txt"
  timeout "$attempt_timeout_s" uv run --package agentwatch-worker \
    python -m agentwatch_worker.demo --workload coding --output "$dir" > "$dir/stdout.txt" 2>&1
  status=$?
  echo "exit_status=$status" >> "$dir/conditions-before.txt"
  cv=$(python3 -I - "$dir/results.json" 2>&1 <<'EOF'
import json, sys
result = json.load(open(sys.argv[1]))
checks = result["checks"]
print(f"{result['baseline']['execution_cv']:.4f} stable={checks['stable_baseline']} passed={checks['passed']}")
EOF
)
  echo "attempt $attempt: exit=$status execution_cv=$cv" | tee -a "$out/summary.txt"
  [ "$status" -eq 0 ] && passed=$((passed + 1))
done
echo "passed $passed of $attempts declared attempts" | tee -a "$out/summary.txt"
echo "Evidence: $out"
