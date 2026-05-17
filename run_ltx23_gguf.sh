#!/bin/bash
set -euo pipefail

HOST_MOUNT="/home/eric-reitsma/comfyui_toolbox"
MODEL_DIR="/home/eric-reitsma/comfy-models"
OUTPUT_DIR="/home/eric-reitsma/comfyui_toolbox/outputs"
CUSTOM_NODES_DIR="/home/eric-reitsma/comfyui_toolbox/custom_nodes"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$CUSTOM_NODES_DIR"

echo "=== LTX-2.3 GGUF ComfyUI Launcher (No Downloads) ==="
echo "Models:  $MODEL_DIR"
echo "Outputs: $OUTPUT_DIR"
echo "Nodes:   $CUSTOM_NODES_DIR"
echo "UI:      http://127.0.0.1:8188"

COMFY_FLAGS="--bf16-vae --disable-mmap --cache-ram 2 --disable-smart-memory --use-pytorch-cross-attention"
if [ "${GGUF_HIGHVRAM:-0}" = "1" ]; then
  COMFY_FLAGS="$COMFY_FLAGS --highvram --disable-async-offload"
fi

docker run --rm \
  --name comfyui-ltx23-gguf \
  --device=/dev/kfd --device=/dev/dri \
  --group-add=video --group-add=render \
  -e HSA_ENABLE_SDMA=0 \
  -e HSA_USE_SVM=0 \
  -e AMD_SERIALIZE_KERNEL=3 \
  -e TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
  -e PYTORCH_HIP_ALLOC_CONF="backend:native,expandable_segments:True,garbage_collection_threshold:0.6,max_split_size_mb:256" \
  -e PYTHONMALLOC=malloc \
  -e MALLOC_TRIM_THRESHOLD_=100000 \
  -v "$MODEL_DIR:/root/comfy-models" \
  -v "$HOST_MOUNT:/opt/comfy-toolbox" \
  -v "$OUTPUT_DIR:/opt/ComfyUI/output" \
  -v "$CUSTOM_NODES_DIR:/opt/ComfyUI/custom_nodes" \
  -p 8188:8188 \
  kyuz0/amd-strix-halo-comfyui:latest \
  bash -c '
set -euo pipefail
source /etc/profile.d/venv.sh
/opt/set_extra_paths.sh

rm -rf /root/.cache/miopen

pip install protobuf -q

if [ ! -d /opt/ComfyUI/custom_nodes/ComfyUI-GGUF ]; then
  echo "Missing custom node: /opt/ComfyUI/custom_nodes/ComfyUI-GGUF"
  echo "Install it once in your image/container; this launcher does not download anything."
  exit 1
fi
if [ ! -d /opt/ComfyUI/custom_nodes/ComfyUI-KJNodes ]; then
  echo "Missing custom node: /opt/ComfyUI/custom_nodes/ComfyUI-KJNodes"
  echo "Install it once in your image/container; this launcher does not download anything."
  exit 1
fi
if [ ! -d /opt/ComfyUI/custom_nodes/ComfyUI-LTXVideo ]; then
  echo "Missing custom node: /opt/ComfyUI/custom_nodes/ComfyUI-LTXVideo"
  echo "LTX-2.3 22B GGUF workflows may fail without LTXVideo-specific VAE/audio nodes."
  echo "Install it once in your host custom_nodes mount; this launcher does not download anything."
fi

cd /opt/ComfyUI
python main.py --listen 0.0.0.0 --port 8188 '"$COMFY_FLAGS"'
'
