#!/bin/bash
# Lightweight memory monitor for LTX pipeline runs
# Usage: bash memory_monitor.sh <container_name> <log_file>
# Launches in background, kills via PID file

CONTAINER_NAME="$1"
LOG_FILE="$2"
PID_FILE="/tmp/memory_monitor_${CONTAINER_NAME}.pid"

echo "$$" > "$PID_FILE"

INTERVAL=10

log_memory() {
  local label="$1"
  local ts
  ts=$(date '+%H:%M:%S')

  local host_mem docker_mem gpu_mem
  host_mem=$(free -m | awk '/^Mem:/ {printf "%dM/%dM (%.0f%%)", $3, $2, $3/$2*100}')
  docker_mem=$(docker stats "$CONTAINER_NAME" --no-stream --format "{{.MemUsage}}" 2>/dev/null || echo "N/A")
  gpu_mem=$(rocm-smi --showmeminfo vram 2>/dev/null | awk '/VRAM Usage/ {print $5" "$6; exit}' || echo "N/A")

  local cache
  cache=$(awk '/^Cached:/ {printf "%.0fM", $2/1024}' /proc/meminfo 2>/dev/null || echo "N/A")

  echo "$ts | $label | host: $host_mem | cache: $cache | docker: $docker_mem | gpu: $gpu_mem" >> "$LOG_FILE"
}

log_memory "start"

while kill -0 "$(cat "$PID_FILE" 2>/dev/null)" 2>/dev/null; do
  sleep "$INTERVAL"
  log_memory "running"
done
