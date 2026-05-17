#!/bin/bash
# run_ltx23_19b_split.sh — Subprocess pipeline for LTX-2.3 19B on unified memory.
#
# Splits generation into 4 separate Python processes. Each process exits
# completely before the next starts, guaranteeing ALL GPU+CPU memory is
# reclaimed by the OS between stages. The only reliable approach on
# Strix Halo unified memory.
#
# Usage:
#   bash run_ltx23_19b_split.sh --prompt "Your prompt" [--width 768] [...]
#   bash run_ltx23_19b_split.sh --skip-stage 3   # resume from stage 3
set -euo pipefail

HOST_MOUNT="/home/eric-reitsma/comfyui_toolbox"
LTX_REPO="/home/eric-reitsma/LTX-2"
MODEL_DIR="/home/eric-reitsma/comfy-models"
OUTPUT_DIR="/home/eric-reitsma/comfyui_toolbox/outputs"
HF_CACHE="/home/eric-reitsma/.cache/huggingface"
TEMP_DIR="/opt/comfy-toolbox/tmp/ltx23_19b_split"

# Parse --skip-stage
SKIP_STAGE=""
PASSTHROUGH=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-stage) SKIP_STAGE="$2"; shift 2 ;;
        *) PASSTHROUGH+=("$1"); shift ;;
    esac
done

mkdir -p "$OUTPUT_DIR/ltx23_19b"

echo "================================================"
echo "LTX-2.3 19B Split Pipeline"
echo "Temp:  $TEMP_DIR"
echo "Skip:  ${SKIP_STAGE:-none}"
echo "Args:  ${PASSTHROUGH[*]:-(defaults)}"
echo "================================================"

# Fresh start: clean temp dir
if [ -z "$SKIP_STAGE" ]; then
    rm -rf "$TEMP_DIR" 2>/dev/null || true
fi

# Build the inline script that will run inside Docker.
# COMMON_ARGS is defined at the top using bash @Q quoting so every arg
# is properly escaped, then piped into `docker run bash`.
INNER_SCRIPT=$(cat << 'INNER'
#!/bin/bash
set -euo pipefail
source /etc/profile.d/venv.sh
pip install -q einops av openimageio psutil safetensors 2>/dev/null
export PYTHONPATH="/opt/LTX-2/packages/ltx-core/src:/opt/LTX-2/packages/ltx-pipelines/src:${PYTHONPATH:-}"

SKIP="${SKIP:-}"
# Temp files on persistent mount so --skip-stage works across Docker restarts
mkdir -p /opt/comfy-toolbox/tmp/ltx23_19b_split

run_stage() {
    local label=$1 script=$2 num=$3
    if [ -n "$SKIP" ] && [ "$num" -lt "$SKIP" ]; then
        echo "[$label] Already completed (skip), continuing"; return
    fi
    if [ -n "$SKIP" ] && [ "$num" -eq "$SKIP" ]; then
        echo "[$label] Resume point"
    fi
    echo ""
    echo "============================================"
    echo "  Stage $num: $label"
    echo "============================================"
    $script "${COMMON_ARGS[@]}"
    echo "[$label] Process exited -- memory reclaimed."
}

run_stage "Stage 1 -- half-res denoising" "python /opt/comfy-toolbox/ltx23_19b_stage1.py" 1
run_stage "Upscale -- spatial x2"         "python /opt/comfy-toolbox/ltx23_19b_upscale.py" 2
run_stage "Stage 2 -- full-res refine"    "python /opt/comfy-toolbox/ltx23_19b_stage2.py" 3
run_stage "Decode -- VAE + MP4 export"    "python /opt/comfy-toolbox/ltx23_19b_decode.py" 4

echo ""
echo "================================================"
echo "Split pipeline complete!"
echo "Output files in /opt/ComfyUI/output/"
echo "================================================"
INNER
)

# Prepend the COMMON_ARGS with persistent temp-dir + user passthrough args
# ${array[@]@Q} quotes each element for safe re-entry into shell
FULL_SCRIPT="COMMON_ARGS=(--temp-dir /opt/comfy-toolbox/tmp/ltx23_19b_split ${PASSTHROUGH[@]@Q})
$INNER_SCRIPT"

# Pipe the full script into Docker's bash via stdin
echo "$FULL_SCRIPT" | docker run --rm -i \
  --name ltx23-19b-split \
  --shm-size=16g \
  --device=/dev/kfd --device=/dev/dri \
  --group-add=video --group-add=render \
  -e HSA_ENABLE_SDMA=0 \
  -e AMD_SERIALIZE_KERNEL=3 \
  -e TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
  -e 'PYTORCH_HIP_ALLOC_CONF=backend:native,garbage_collection_threshold:0.7,max_split_size_mb:256' \
  -e PYTHONMALLOC=malloc \
  -e MALLOC_TRIM_THRESHOLD_=100000 \
  -e HIP_VISIBLE_DEVICES=0 \
  -e "SKIP=${SKIP_STAGE}" \
  -v "$MODEL_DIR:/root/comfy-models" \
  -v "$HOST_MOUNT:/opt/comfy-toolbox" \
  -v "$LTX_REPO:/opt/LTX-2" \
  -v "$OUTPUT_DIR:/opt/ComfyUI/output" \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  kyuz0/amd-strix-halo-comfyui:latest \
  bash -s -- "$@"
