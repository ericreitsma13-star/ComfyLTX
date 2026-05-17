#!/bin/bash
set -euo pipefail

HOST_MOUNT="/home/eric-reitsma/comfyui_toolbox"
MODEL_DIR="/home/eric-reitsma/comfy-models"
OUTPUT_DIR="/home/eric-reitsma/comfyui_toolbox/outputs"
mkdir -p "$OUTPUT_DIR"

echo "=== ComfyUI Toolbox — Qwen 2512 4-Step Warm Test ==="
echo "Outputs dir: $OUTPUT_DIR"

docker run --rm \
  --name comfyui-toolbox \
  --device=/dev/kfd --device=/dev/dri \
  --group-add=video --group-add=render \
  -e HSA_ENABLE_SDMA=0 \
  -e HSA_USE_SVM=0 \
  -e AMD_SERIALIZE_KERNEL=3 \
  -e TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
  -e PYTORCH_HIP_ALLOC_CONF="backend:native,expandable_segments:True,garbage_collection_threshold:0.7,max_split_size_mb:256" \
  -e PYTHONMALLOC=malloc \
  -e MALLOC_TRIM_THRESHOLD_=100000 \
  -v "$MODEL_DIR:/root/comfy-models" \
  -v "$HOST_MOUNT:/opt/comfy-toolbox" \
  -v "$OUTPUT_DIR:/opt/ComfyUI/output" \
  -p 8188:8188 \
  kyuz0/amd-strix-halo-comfyui:latest \
  bash -c '
set -euo pipefail
source /etc/profile.d/venv.sh
/opt/set_extra_paths.sh
cp /opt/comfy-toolbox/workflow_4step_fp8.json /opt/comfy-workflows/

cd /opt/ComfyUI
python main.py --listen 0.0.0.0 --port 8188 \
  --bf16-vae --disable-mmap --cache-ram 2 --disable-smart-memory --highvram --disable-async-offload --use-pytorch-cross-attention \
  --dont-print-server &
COMFY_PID=$!

echo "Waiting for ComfyUI server..."
for i in $(seq 1 120); do
  if curl -s http://127.0.0.1:8188/queue >/dev/null 2>&1; then
    echo "Server ready!"
    break
  fi
  sleep 2
done

python /opt/comfy-toolbox/runner_sequential.py

kill $COMFY_PID 2>/dev/null; wait $COMFY_PID 2>/dev/null
echo "=== Warm test complete ==="
'
