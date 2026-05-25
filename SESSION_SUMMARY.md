# Session Summary — May 24, 2026

## Current State

### Working
- **Clean I2V + audio pipeline** via API (`workflow_clean_i2v_audio.json`, `gen_1min_mv.py`)
- **Z-Image Turbo** generates images via API (verified: `CLIPLoader(qwen_image)` + `UNETLoader` + `EmptyLatentImage` + `SamplerCustom` + `FlowMatchEulerDiscreteScheduler (Custom)`)
- **LTX I2V + audio conditioning** (lip sync works)
- **Multi-scene stitch** with crossfade
- **VRGDG workflow loads** in ComfyUI UI with all 93 nodes
- **FlowMatchEulerDiscreteScheduler (Custom)** shim installed

### All Models Downloaded (16 total)
| Model | File | Size |
|---|---|---|
| LTX GGUF Q6_K | `diffusion_models/LTX-2.3-22B-distilled-1.1-Q6_K.gguf` | 19.5 GB |
| LTX GGUF Q3_K_M | `diffusion_models/LTX-2.3-22B-distilled-1.1-Q3_K_M.gguf` | 13.7 GB |
| LTX full | `checkpoints/ltx-2.3-22b-distilled-1.1.safetensors` | 43 GB |
| Video VAE | `vae/ltx-2.3-22b-distilled_video_vae.safetensors` | 1.5 GB |
| Audio VAE | `vae/ltx-2.3-22b-distilled_audio_vae.safetensors` | 0.4 GB |
| Gemma CLIP | `text_encoders/gemma_3_12B_it_fp4_mixed.safetensors` | 7.4 GB |
| LTX LoRA 384 | `loras/ltx-2.3-22b-distilled-lora-384-1.1.safetensors` | 7.6 GB |
| Spatial Upscaler | `upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | 1.0 GB |
| Z-Image Turbo | `diffusion_models/z_image_turbo_bf16.safetensors` | 12.3 GB |
| Z-Image Qwen | `text_encoders/qwen_3_4b.safetensors` | 8.0 GB |
| Z-Image VAE | `vae/ae.safetensors` | 0.3 GB |
| SuperGemma LLM | `LLM/supergemma4-26b-uncensored-fast-v2-Q4_K_M.gguf` | 16.8 GB |
| MMProj | `LLM/mmproj-BF16.gguf` | 1.2 GB |
| SDXL base | `checkpoints/sd_xl_base_1.0.safetensors` | 7.0 GB |
| Sulphur LoRA | `loras/sulphur_final.safetensors` | 10.3 GB |
| IC-LoRA lipdub | `loras/ltx-2.3-22b-ic-lora-lipdub.safetensors` | 2.5 GB |

### Workflow Files
| File | Purpose |
|---|---|
| `workflow_clean_i2v_audio.json` | Clean I2V + audio (no subgraphs, works anywhere) |
| `workflow_local_lite.json` | Same as clean, alias for local testing |
| `workflow_a_zimage_refs.json` | Step A: Z-Image generates scene references |
| `workflow_i2v_audio_api.json` | API-format workflow JSON |
| `gen_1min_mv.py` | 1-minute full pipeline (SDXL refs → LTX → stitch) |
| `gen_mv_step_a.py` | Step A: generate per-scene references (Z-Image) |
| `gen_mv_step_b.py` | Step B: LTX I2V from references + audio |
| `pipeline_final.py` | Alternative full pipeline |
| `test_lora.py` / `test_fixed.py` / etc. | Various test scripts |

### Config File
`~/.config/opencode/ltx_video_pipeline_config.md`

## What Works Locally (16GB Laptop)
- **Starting to get usable results** — last 1-minute test showed lip sync working, proper front-facing composition
- **Quality bottleneck**: LTX at 832×480 with 500kbps is soft, Z-Image + LTX simultaneously exceeds VRAM
- **VRGDG workflow** loads and runs through Z-Image phase but gets stuck on SRT/LLM issues (all now fixed)

## What's Needed for Cloud (48GB VRAM / 64GB RAM)
The VRGDG workflow `LTX2.3_Music_Video_Creator_I2V_V5.1.json` will run fully on MimicPC because:
- No VRAM swapping (everything fits in 48GB)
- UI workflow resolves subgraph UUIDs correctly
- SuperGemma LLM loads with llama-cpp-python
- Spatial upscaler works (latent upscale before decode)
- Two-pass LoRA (half/full strength)
- Z-Image + LTX simultaneously

## Next Steps
1. **Step A** currently generating SDXL references (Z-Image version ready to test)
2. **Step B** LTX pipeline tested and working
3. On cloud: load VRGDG workflow directly, no split needed

### For Tomorrow
Check `tail -50 /tmp/comfyui.log | grep -E "Prompt executed|error"` for overnight runs
`ls -lt /home/ericr/ComfyUI/output/*.mp4 | head -5`

## Key Commands
```bash
# Start ComfyUI (from correct CWD)
cd /home/ericr/ComfyUI && python3 -c "
import subprocess
proc = subprocess.Popen(
    ['/home/ericr/ComfyUI/venv/bin/python', 'main.py', '--port', '8188', '--listen', '127.0.0.1', '--disable-auto-launch'],
    cwd='/home/ericr/ComfyUI',
    stdout=open('/tmp/comfyui.log','w'),
    stderr=subprocess.STDOUT)
print(f'PID: {proc.pid}')
"
```
