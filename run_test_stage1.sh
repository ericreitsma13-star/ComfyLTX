#!/bin/bash
# run_test_stage1.sh — Single-stage Docker run for memory testing
set -euo pipefail

HOST_MOUNT="/home/eric-reitsma/comfyui_toolbox"
LTX_REPO="/home/eric-reitsma/LTX-2"
MODEL_DIR="/home/eric-reitsma/comfy-models"
OUTPUT_DIR="/home/eric-reitsma/comfyui_toolbox/outputs"
HF_CACHE="/home/eric-reitsma/.cache/huggingface"
TEMP_DIR="/opt/comfy-toolbox/tmp/ltx23_test"

rm -rf "$TEMP_DIR" 2>/dev/null || true

echo "=== LTX-2.3 19B Stage 1 Test ==="
echo "Resolution: 512×288, 25 frames"
echo "Temp dir: $TEMP_DIR"

docker run --rm \
  --shm-size=4g \
  --device=/dev/kfd --device=/dev/dri \
  --group-add=video --group-add=render \
  -e HSA_ENABLE_SDMA=0 \
  -e AMD_SERIALIZE_KERNEL=3 \
  -e TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
  -e 'PYTORCH_HIP_ALLOC_CONF=backend:native,garbage_collection_threshold:0.7,max_split_size_mb:256' \
  -e PYTHONMALLOC=malloc \
  -e MALLOC_TRIM_THRESHOLD_=100000 \
  -e HIP_VISIBLE_DEVICES=0 \
  -v "$MODEL_DIR:/root/comfy-models" \
  -v "$HOST_MOUNT:/opt/comfy-toolbox" \
  -v "$LTX_REPO:/opt/LTX-2" \
  -v "$OUTPUT_DIR:/opt/ComfyUI/output" \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  kyuz0/amd-strix-halo-comfyui:latest \
  bash -c '
set -euo pipefail
source /etc/profile.d/venv.sh
pip install -q einops av openimageio psutil safetensors 2>/dev/null
export PYTHONPATH="/opt/LTX-2/packages/ltx-core/src:/opt/LTX-2/packages/ltx-pipelines/src:${PYTHONPATH:-}"

echo "=== Starting stage 1 ==="
python /opt/comfy-toolbox/ltx23_19b_stage1.py \
  --temp-dir /opt/comfy-toolbox/tmp/ltx23_test \
  --width 512 --height 288 --frames 25 --fps 24 \
  --seed 42 \
  --prompt "A serene Japanese garden at sunset with cherry blossom petals falling, cinematic quality."
echo "=== Stage 1 exited ==="
'
