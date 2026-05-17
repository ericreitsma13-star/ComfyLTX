#!/bin/bash
set -euo pipefail

WORKFLOW="/home/eric-reitsma/comfyui_toolbox/workflow_ltx23_gguf_t2v.json"
API_BASE="${9:-http://127.0.0.1:8288}"

PROMPT="${1:-A serene Japanese garden at sunset with cherry blossom petals falling, gentle water flowing in a stream, soft koto music, cinematic quality.}"
SEED="${2:-42}"
WIDTH="${3:-640}"
HEIGHT="${4:-384}"
FRAMES="${5:-97}"
NEGATIVE="${8:-blurry, out of focus, low quality, artifacts, distorted, low resolution, bad composition}"

if [ ! -f "$WORKFLOW" ]; then
  echo "Missing workflow: $WORKFLOW"
  exit 1
fi

TMP_REQ=$(mktemp)
python3 - <<PY
import json
wf = json.load(open("$WORKFLOW"))
wf["5"]["inputs"]["text"] = "$PROMPT"
wf["6"]["inputs"]["text"] = "$NEGATIVE"
wf["14"]["inputs"]["noise_seed"] = int("$SEED")
wf["7"]["inputs"]["width"] = int("$WIDTH")
wf["7"]["inputs"]["height"] = int("$HEIGHT")
wf["7"]["inputs"]["length"] = int("$FRAMES")
json.dump({"prompt": wf}, open("$TMP_REQ", "w"))
PY

RESP=$(curl -s -X POST "$API_BASE/prompt" -H 'Content-Type: application/json' -d @"$TMP_REQ")
rm -f "$TMP_REQ"
echo "$RESP"
PID=$(echo "$RESP" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("prompt_id",""))' 2>/dev/null || true)
[ -n "$PID" ] && echo "Queued: $PID"
