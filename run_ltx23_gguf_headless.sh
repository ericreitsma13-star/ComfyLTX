#!/bin/bash
set -euo pipefail

HOST_MOUNT="/home/eric-reitsma/comfyui_toolbox"
MODEL_DIR="/home/eric-reitsma/comfy-models"
OUTPUT_DIR="/home/eric-reitsma/comfyui_toolbox/outputs"
WORKFLOW="$HOST_MOUNT/workflow_ltx23_gguf_t2v.json"
CONTAINER_NAME="comfyui-ltx23-gguf"
HOST_PORT="${9:-8288}"
DOCKER_PORT=8188

PROMPT="${1:-}"
SEED="${2:-42}"
WIDTH="${3:-384}"
HEIGHT="${4:-256}"
FRAMES="${5:-193}"
STEPS="${6:-4}"
CFG="${7:-1.0}"
NEGATIVE="${8:-blurry, out of focus, low quality, artifacts, distorted, low resolution, bad composition}"

if [ -z "$PROMPT" ]; then
  echo "Usage: bash run_ltx23_gguf_headless.sh \"prompt\" [seed] [width] [height] [frames] [steps] [cfg] [negative] [port]"
  PROMPT="A serene Japanese garden at sunset with cherry blossom petals falling, cinematic quality."
fi

mkdir -p "$OUTPUT_DIR"

echo "=== LTX-2.3 GGUF Headless ==="
echo "Size: ${WIDTH}x${HEIGHT} x ${FRAMES}frames  Steps: $STEPS  CFG: $CFG"

COMFY_FLAGS="--bf16-vae --disable-mmap --cache-ram 2 --disable-smart-memory --use-pytorch-cross-attention"
if [ "${GGUF_HIGHVRAM:-0}" = "1" ]; then
  COMFY_FLAGS="$COMFY_FLAGS --highvram --disable-async-offload"
fi

# Start container if not running
if ! docker ps --filter "name=$CONTAINER_NAME" --format "{{.Names}}" 2>/dev/null | grep -q "$CONTAINER_NAME"; then
  echo "Launching ComfyUI on port $HOST_PORT..."
  docker run -d \
    --name "$CONTAINER_NAME" \
    --device=/dev/kfd --device=/dev/dri \
    --group-add=video --group-add=render \
    -e HSA_ENABLE_SDMA=0 -e HSA_USE_SVM=0 -e AMD_SERIALIZE_KERNEL=3 \
    -e TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
    -e PYTORCH_HIP_ALLOC_CONF="backend:native,expandable_segments:True,garbage_collection_threshold:0.6,max_split_size_mb:256" \
    -e PYTHONMALLOC=malloc -e MALLOC_TRIM_THRESHOLD_=100000 \
    -v "$MODEL_DIR:/root/comfy-models" \
    -v "$HOST_MOUNT:/opt/comfy-toolbox" \
    -v "$OUTPUT_DIR:/opt/ComfyUI/output" \
    -p "$HOST_PORT:$DOCKER_PORT" \
    kyuz0/amd-strix-halo-comfyui:latest \
    bash -c '
set -e
source /etc/profile.d/venv.sh
/opt/set_extra_paths.sh

pip install protobuf -q

# Clear MIOpen kernel compilation cache (can grow to 40GB+ on Strix Halo)
rm -rf /root/.cache/miopen

cd /opt/ComfyUI
for node in ComfyUI-GGUF ComfyUI-KJNodes ComfyUI-LTXVideo ComfyUI-ROCM-Optimized-VAE; do
  [ -d "/opt/comfy-toolbox/custom_nodes/$node" ] && [ ! -d "custom_nodes/$node" ] && cp -r "/opt/comfy-toolbox/custom_nodes/$node" "custom_nodes/$node"
done
exec python main.py --listen 0.0.0.0 --port '"$DOCKER_PORT"' '"$COMFY_FLAGS"'
'
fi

API_BASE="http://127.0.0.1:$HOST_PORT"

# Wait for API
echo -n "Waiting for ComfyUI"
for i in $(seq 1 120); do
  if curl -s "$API_BASE/system_stats" > /dev/null 2>&1; then echo " ready!"; break; fi
  sleep 2; echo -n "."
done
echo ""

# Build and queue workflow
TMP_WF=$(mktemp)
python3 -c "
import json
with open('$WORKFLOW') as f:
    wf = json.load(f)
wf['5']['inputs']['text'] = '''$PROMPT'''
wf['6']['inputs']['text'] = '''$NEGATIVE'''
wf['14']['inputs']['noise_seed'] = $SEED
wf['7']['inputs']['width'] = $WIDTH
wf['7']['inputs']['height'] = $HEIGHT
wf['7']['inputs']['length'] = $FRAMES
json.dump({'prompt': wf}, open('$TMP_WF', 'w'))
"

RESPONSE=$(curl -s -X POST "$API_BASE/prompt" -H 'Content-Type: application/json' -d @"$TMP_WF")
rm -f "$TMP_WF"

PROMPT_ID=$(echo "$RESPONSE" | python3 -c "import json,sys; print(json.load(sys.stdin).get('prompt_id','error'))" 2>/dev/null || echo "error")
if [ "$PROMPT_ID" = "error" ]; then echo "ERROR: $RESPONSE"; exit 1; fi

cleanup() { kill "$MONITOR_PID" 2>/dev/null; rm -f "/tmp/memory_monitor_${CONTAINER_NAME}.pid"; }
trap cleanup EXIT INT TERM
echo "Prompt ID: $PROMPT_ID"

MEM_LOG="$OUTPUT_DIR/memory_$(date +%Y%m%d_%H%M%S).log"
bash "$HOST_MOUNT/memory_monitor.sh" "$CONTAINER_NAME" "$MEM_LOG"
MONITOR_PID=$(cat /tmp/memory_monitor_${CONTAINER_NAME}.pid)
echo "Memory log: $MEM_LOG (monitor PID $MONITOR_PID)"

# Poll for completion
echo -n "Generating"
while true; do
  STATUS=$(curl -s "$API_BASE/history/$PROMPT_ID" | python3 -c "
import json, sys
data = json.load(sys.stdin)
h = data.get('$PROMPT_ID', {})
s = h.get('status', {})
if s.get('completed'): print('completed')
elif s.get('failed'): print('failed')
else: print('running')
" 2>/dev/null || echo "unknown")
  case "$STATUS" in
    completed)
      echo " completed!"
      ts=$(date '+%H:%M:%S'); echo "$ts | done | $(free -m | awk '/^Mem:/ {printf "host: %dM/%dM (%.0f%%)", $3, $2, $3/$2*100}') | docker: $(docker stats "$CONTAINER_NAME" --no-stream --format "{{.MemUsage}}" 2>/dev/null || echo N/A)" >> "$MEM_LOG"
      ls -lt "$OUTPUT_DIR"/ltx23_gguf* 2>/dev/null | head -3
      break
      ;;
    failed)
      echo " FAILED!"
      ts=$(date '+%H:%M:%S'); echo "$ts | failed | $(free -m | awk '/^Mem:/ {printf "host: %dM/%dM (%.0f%%)", $3, $2, $3/$2*100}') | docker: $(docker stats "$CONTAINER_NAME" --no-stream --format "{{.MemUsage}}" 2>/dev/null || echo N/A)" >> "$MEM_LOG"
      docker logs "$CONTAINER_NAME" 2>/dev/null | tail -20
      break
      ;;
    *) echo -n "."; sleep 5 ;;
  esac
done
echo "Outputs: $OUTPUT_DIR"
