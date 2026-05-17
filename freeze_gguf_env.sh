#!/bin/bash
# Record the exact versions of all pipeline components
set -euo pipefail

NAMESPACE="comfyui-ltx23-gguf"

if ! docker ps --filter "name=$NAMESPACE" --format "{{.Names}}" | grep -q "$NAMESPACE"; then
  echo "ERROR: Container $NAMESPACE not running"
  exit 1
fi

OUTPUT=$(cat << EOF
# GGUF Pipeline Version Lock
Generated: $(date -Is)

## ComfyUI
Commit: $(docker exec $NAMESPACE bash -c 'cd /opt/ComfyUI && git log --oneline -1')
Tag:   $(docker exec $NAMESPACE bash -c 'cd /opt/ComfyUI && git describe --tags --always 2>/dev/null || echo "no-tag"')

## ComfyUI-GGUF
Commit: $(docker exec $NAMESPACE bash -c 'cd /opt/ComfyUI/custom_nodes/ComfyUI-GGUF && git log --oneline -1')

## ComfyUI-LTXVideo
Commit: $(docker exec $NAMESPACE bash -c 'cd /opt/ComfyUI/custom_nodes/ComfyUI-LTXVideo && git log --oneline -1')

## ComfyUI-KJNodes
Commit: $(docker exec $NAMESPACE bash -c 'cd /opt/ComfyUI/custom_nodes/ComfyUI-KJNodes && git log --oneline -1')

## Docker Image
Image: kyuz0/amd-strix-halo-comfyui:latest
Digest: $(docker inspect kyuz0/amd-strix-halo-comfyui:latest --format '{{.Id}}' 2>/dev/null || echo "unknown")

## Model Files
EOF
)

# Add model file info
for f in /home/eric-reitsma/comfy-models/unet/*.gguf /home/eric-reitsma/comfy-models/vae/*.safetensors /home/eric-reitsma/comfy-models/text_encoders/*.gguf /home/eric-reitsma/comfy-models/text_encoders/*.safetensors; do
  [ -f "$f" ] && OUTPUT+="\n- $(basename $f) ($(du -h "$f" | cut -f1))"
done

echo -e "$OUTPUT" | tee /home/eric-reitsma/comfyui_toolbox/GGUF_LOCK.md
echo "Lock written to GGUF_LOCK.md"
