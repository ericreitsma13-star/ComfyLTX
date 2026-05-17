#!/bin/bash
set -euo pipefail

MODEL_DIR="/home/eric-reitsma/comfy-models"

FILES=(
  "$MODEL_DIR/unet/ltx-2.3-22b-distilled-1.1-UD-Q4_K_M.gguf"
  "$MODEL_DIR/vae/ltx-2.3-22b-distilled_video_vae.safetensors"
  "$MODEL_DIR/vae/ltx-2.3-22b-distilled_audio_vae.safetensors"
  "$MODEL_DIR/text_encoders/ltx-2.3-22b-distilled_embeddings_connectors.safetensors"
  "$MODEL_DIR/text_encoders/gemma-3-12b-it-qat-UD-Q4_K_XL.gguf"
  "$MODEL_DIR/text_encoders/mmproj-BF16.gguf"
)

echo "=== Warming LTX-2.3 GGUF file cache ==="
for f in "${FILES[@]}"; do
  if [ ! -f "$f" ]; then
    echo "Missing: $f"
    exit 1
  fi
  echo "Warm: $f"
  dd if="$f" of=/dev/null bs=8M status=none

done

echo "Cache warm complete."
