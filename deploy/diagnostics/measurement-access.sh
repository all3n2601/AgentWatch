#!/usr/bin/env bash
# Read-only report of the Linux and NVIDIA measurement signals visible to this shell.
# It records availability; it is not a sampler. Run inside the Linux environment that
# will execute experiments (on Windows: `wsl -d Ubuntu`, not the docker-desktop distro).
set -u

section() { printf '\n--- %s\n' "$1"; }
show() { printf '[%s]\n' "$1"; cat "$1" 2>&1 | head -"${2:-8}"; }

section "host"
uname -r
if [ -r /etc/os-release ]; then . /etc/os-release && echo "$PRETTY_NAME"; fi
echo "cpus=$(nproc)"
grep -qi microsoft /proc/version 2>/dev/null && echo "wsl2=yes (PSI and cgroups describe the WSL VM, not the Windows host)"
echo "local_utc_offset=$(date +%z)"

section "pressure stall information"
for resource in cpu memory io; do show "/proc/pressure/$resource"; done

section "cgroup"
mount | grep -E '^cgroup2? ' || echo "no cgroup mount found"
show /sys/fs/cgroup/cgroup.controllers
# Only the unified (cgroup v2) entry, `0::<path>`, names files under /sys/fs/cgroup; hybrid
# hosts also list v1 controller lines first.
if group=$(grep -m1 '^0::' /proc/self/cgroup 2>/dev/null); then
  group=${group#0::}
  echo "self cgroup v2 path: $group"
  for file in cpu.stat cpu.max cpu.pressure memory.current memory.max; do
    show "/sys/fs/cgroup${group%/}/$file" 6
  done
else
  echo "self cgroup v2 path: unavailable (no 0:: entry in /proc/self/cgroup)"
fi

section "nvidia"
smi=$(command -v nvidia-smi || true)
[ -z "$smi" ] && [ -x /usr/lib/wsl/lib/nvidia-smi ] && smi=/usr/lib/wsl/lib/nvidia-smi
if [ -n "$smi" ]; then
  echo "nvidia-smi: $smi"
  ls -l /dev/dxg 2>/dev/null
  "$smi" --query-gpu=name,memory.total,driver_version,compute_cap,pstate,utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.sm,clocks_event_reasons.active --format=csv 2>&1
  "$smi" --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1
else
  echo "nvidia-smi: unavailable"
fi

section "tooling"
for tool in dcgmi nv-hostengine docker vllm ollama uv python3; do
  printf '%s: ' "$tool"; command -v "$tool" || echo unavailable
done
