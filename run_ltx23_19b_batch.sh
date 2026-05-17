#!/bin/bash
# run_ltx23_19b_batch.sh — One-shot pipeline that keeps the transformer hot on GPU.
#
# No subprocesses, no streaming builder, no OffloadMode.CPU.
# Builds the transformer ONCE (~19 GB FP8 on GPU), processes all prompts,
# then frees the transformer. Designed for batch processing.
#
# Usage:
#   bash run_ltx23_19b_batch.sh --prompt "A cat video"
#   bash run_ltx23_19b_batch.sh --prompts-file prompts.txt
set -euo pipefail

HOST_MOUNT="/home/eric-reitsma/comfyui_toolbox"
LTX_REPO="/home/eric-reitsma/LTX-2"
MODEL_DIR="/home/eric-reitsma/comfy-models"
OUTPUT_DIR="/home/eric-reitsma/comfyui_toolbox/outputs"
HF_CACHE="/home/eric-reitsma/.cache/huggingface"

mkdir -p "$OUTPUT_DIR/ltx23_19b"

echo "=== LTX-2.3 19B Batch Pipeline ==="
echo "Transformer stays hot on GPU for all prompts"
echo "Args: $*"

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

python /opt/comfy-toolbox/ltx23_19b_batch.py "$@"
' "_" "$@"
