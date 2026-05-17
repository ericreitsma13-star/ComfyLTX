#!/bin/bash
set -euo pipefail

HOST_MOUNT="/home/eric-reitsma/comfyui_toolbox"
MODEL_DIR="/home/eric-reitsma/comfy-models"
OUTPUT_DIR="/home/eric-reitsma/comfyui_toolbox/outputs/sdxl"
HF_CACHE="/home/eric-reitsma/.cache/huggingface"
mkdir -p "$OUTPUT_DIR"

PACK="${1:-}"
EXTRA="${2:-}"

echo "=== SDXL ==="
echo "Pack: ${PACK:-none}"
echo "Outputs: $OUTPUT_DIR"

docker run --rm \
  --name sdxl-run \
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
  -v "$HF_CACHE:/root/.cache/huggingface" \
  kyuz0/amd-strix-halo-comfyui:latest \
  bash -c "
set -euo pipefail
source /etc/profile.d/venv.sh
python /opt/comfy-toolbox/test_sdxl.py \
  ${PACK:+--pack $PACK} \
  ${EXTRA:+--prompt \"$EXTRA\"}
echo '=== SDXL complete ==='
"
