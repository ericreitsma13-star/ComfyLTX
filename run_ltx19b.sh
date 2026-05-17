#!/bin/bash
set -euo pipefail

HOST_MOUNT="/home/eric-reitsma/comfyui_toolbox"
LTX_REPO="/home/eric-reitsma/LTX-2"
MODEL_DIR="/home/eric-reitsma/comfy-models"
OUTPUT_DIR="/home/eric-reitsma/comfyui_toolbox/outputs"
HF_CACHE="/home/eric-reitsma/.cache/huggingface"
mkdir -p "$OUTPUT_DIR/ltx19b"

CHECKPOINT="$MODEL_DIR/diffusion_models/ltx-2-19b-distilled-fp8.safetensors"
GEMMA_ROOT="$HOST_MOUNT/gemma_tokenizer"

echo "=== LTX-2 19B Distilled FP8 ==="
echo "Checkpoint: $CHECKPOINT"
echo "Gemma root: $GEMMA_ROOT"
echo "Output:     $OUTPUT_DIR/ltx19b"

docker run --rm \
  --name ltx19b-pipeline \
  --shm-size=16g \
  --device=/dev/kfd --device=/dev/dri \
  --group-add=video --group-add=render \
  -e HSA_ENABLE_SDMA=0 \
  -e AMD_SERIALIZE_KERNEL=3 \
  -e TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
  -e PYTORCH_HIP_ALLOC_CONF="backend:native,expandable_segments:True,garbage_collection_threshold:0.7,max_split_size_mb:512" \
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

echo "Installing LTX-2 dependencies..."
pip install -q "diffusers>=0.37.0" einops av openimageio psutil 2>/dev/null

export PYTHONPATH="/opt/LTX-2/packages/ltx-core/src:/opt/LTX-2/packages/ltx-pipelines/src:${PYTHONPATH:-}"

echo "Running LTX-2 19B pipeline..."
python /opt/comfy-toolbox/ltx19b_infer.py "$@"

echo "=== LTX-2 19B pipeline complete ==="
' "ltx19b" "$@"
