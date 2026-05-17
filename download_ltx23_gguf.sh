#!/bin/bash
set -euo pipefail

MODEL_DIR="/home/eric-reitsma/comfy-models"

mkdir -p \
  "$MODEL_DIR/unet" \
  "$MODEL_DIR/vae" \
  "$MODEL_DIR/text_encoders" \
  "$MODEL_DIR/loras" \
  "$MODEL_DIR/latent_upscale_models"

A2="aria2c -c -x 16 -s 16 -k 1M --file-allocation=none"

# Main distilled GGUF (you already started this one)
$A2 "https://huggingface.co/unsloth/LTX-2.3-GGUF/resolve/main/distilled-1.1/ltx-2.3-22b-distilled-1.1-UD-Q4_K_M.gguf?download=true" \
  -d "$MODEL_DIR/unet" -o "ltx-2.3-22b-distilled-1.1-UD-Q4_K_M.gguf"

# Distilled companion files required by Comfy workflows
$A2 "https://huggingface.co/unsloth/LTX-2.3-GGUF/resolve/main/vae/ltx-2.3-22b-distilled_video_vae.safetensors?download=true" \
  -d "$MODEL_DIR/vae" -o "ltx-2.3-22b-distilled_video_vae.safetensors"
$A2 "https://huggingface.co/unsloth/LTX-2.3-GGUF/resolve/main/vae/ltx-2.3-22b-distilled_audio_vae.safetensors?download=true" \
  -d "$MODEL_DIR/vae" -o "ltx-2.3-22b-distilled_audio_vae.safetensors"
$A2 "https://huggingface.co/unsloth/LTX-2.3-GGUF/resolve/main/text_encoders/ltx-2.3-22b-distilled_embeddings_connectors.safetensors?download=true" \
  -d "$MODEL_DIR/text_encoders" -o "ltx-2.3-22b-distilled_embeddings_connectors.safetensors"

# Gemma GGUF text encoder + mmproj
$A2 "https://huggingface.co/unsloth/gemma-3-12b-it-qat-GGUF/resolve/main/gemma-3-12b-it-qat-UD-Q4_K_XL.gguf?download=true" \
  -d "$MODEL_DIR/text_encoders" -o "gemma-3-12b-it-qat-UD-Q4_K_XL.gguf"
$A2 "https://huggingface.co/unsloth/gemma-3-12b-it-qat-GGUF/resolve/main/mmproj-BF16.gguf?download=true" \
  -d "$MODEL_DIR/text_encoders" -o "mmproj-BF16.gguf"

# Optional but recommended extras for common LTX-2.3 workflows
$A2 "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-distilled-lora-384.safetensors?download=true" \
  -d "$MODEL_DIR/loras" -o "ltx-2.3-22b-distilled-lora-384.safetensors"
$A2 "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.0.safetensors?download=true" \
  -d "$MODEL_DIR/latent_upscale_models" -o "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
$A2 "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-temporal-upscaler-x2-1.0.safetensors?download=true" \
  -d "$MODEL_DIR/latent_upscale_models" -o "ltx-2.3-temporal-upscaler-x2-1.0.safetensors"

echo "LTX-2.3 GGUF downloads complete."
echo "Model root: $MODEL_DIR"
