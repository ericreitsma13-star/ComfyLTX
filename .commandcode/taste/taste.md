# Taste (Continuously Learned by [CommandCode][cmd])

[cmd]: https://commandcode.ai/

# workflow
- Prefer simpler single-quantization approaches (like loading a single quantized model) over complex multi-stage pipelines with streaming/offloading/replacement logic. Confidence: 0.80
- Prefer running ComfyUI headlessly (via workflow JSON + --headless flag) instead of through the web UI for automated/production runs. Confidence: 0.65

# workflow
- For LTX-2.3 pipeline: Use the 19B model instead of 22B to reduce memory usage. Confidence: 0.20
- Pre-cache (pre-download) all pipeline model files upfront so no downloads happen on first run. Confidence: 0.65
- Show full raw reasoning/thought process step-by-step, not just concise "crafting" or "working" summaries. Confidence: 0.75

# download
- Use aria2c (`aria2c -x 16 -s 16 -k 1M`) for downloading large HuggingFace models instead of HF native tools. Confidence: 0.65

# cli
- Prefer one-line commands over multi-line scripts for quick operations. Confidence: 0.65
- When user asks for a command (e.g., "give me [command]"), provide the command text only — do not execute it automatically. Confidence: 0.65

# documentation
- Maintain a "map of objects" (architecture/object relationship diagram) in project documentation to clarify system structure. Confidence: 0.60

# communication
- When asked for a status update, give a concise summary first before diving into technical details. Deep debugging loops without answering the direct question frustrate the user. Confidence: 0.80

