# ComfyLTX Music Video Pipeline

## Machine: ROG Strix (RTX 4090 Mobile, 16GB VRAM, 64GB RAM)

### Running
```
cd /home/ericr/ComfyUI && setsid /home/ericr/ComfyUI/venv/bin/python main.py --port 8188 --listen 127.0.0.1 --disable-auto-launch > /tmp/comfyui.log 2>&1 &
```

### Quick Quality Test
```
python3 pipeline_zimage_ltx.py               # 3 scenes, Z-Image + LTX
python3 gen_music_video_pro.py               # Lyrics + MusicGen + Z-Image + LTX
python3 gen_mv_step_a.py && gen_mv_step_b.py # 15 scenes, full music video
```

---

## Model Inventory (Local, 79 GB total)

### Core LTX Video Models
| Model | Size | Location | Used By |
|---|---|---|---|
| `LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf` (symlink→Q3_K_M) | 17 GB (→14 GB) | `unet/` | LTX UNet |
| `LTX-2.3-22B-distilled-1.1-Q6_K.gguf` | 20 GB | `diffusion_models/` | Alt (too big for 16GB) |
| `ltx-2.3-22b-distilled-1.1.safetensors` | 43 GB | `checkpoints/` | Audio VAE + text proj extraction |
| `gemma-3-12b-it-qat-UD-Q4_K_XL.gguf` | 7 GB | `text_encoders/`, `clip/` | DualCLIPLoaderGGUF |
| `ltx-2.3_text_projection_bf16.safetensors` | 2.2 GB | `text_encoders/`, `clip/` | DualCLIPLoaderGGUF |

### Z-Image Models
| Model | Size | Location | Notes |
|---|---|---|---|
| `z-image-turbo-Q6_K.gguf` | 5.6 GB | `unet/` | GGUF Z-Image UNet |
| `qwen_3_4b.safetensors` | 7.5 GB | `text_encoders/` | Z-Image CLIP |
| `ae.safetensors` | 320 MB | `vae/` | Z-Image VAE |

### VAEs & Accessories
| Model | Size | Location |
|---|---|---|
| `ltx-2.3-22b-distilled_video_vae.safetensors` | 1.4 GB | `vae/` |
| `ltx-2.3-22b-distilled_audio_vae.safetensors` | 348 MB | `vae/` |
| `ltx-2.3-22b-distilled-lora-384-1.1.safetensors` | 7.1 GB | `loras/` |
| `ltx-2.3-22b-ic-lora-lipdub.safetensors` | 2.3 GB | `loras/` |
| `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | 1 GB | `latent_upscale_models/` |

### LLM (optional, for prompt/lyrics gen)
| Model | Size | Location |
|---|---|---|
| `supergemma4-26b-uncensored-fast-v2-Q4_K_M.gguf` | 16 GB | `LLM/` |
| `mmproj-BF16.gguf` | 1.2 GB | `LLM/` |

### Music Models
| Model | Size | Location |
|---|---|---|
| `facebook/musicgen-medium` | 7.5 GB | HF cache |
| `HeartMuLa-RL-oss-3B` | 19 GB | `models/HeartMuLa/` |

### Docker / Misc
| Model | Size | Location |
|---|---|---|
| `qwen-image-Q4_K_S.gguf` | 12 GB | `unet/` |
| `sulphur_prompt_enhancer_model-q8_0.gguf` | 3.5 GB | `prompt_enhancers/` |
| `umt5-xxl-encoder-Q3_K_M.gguf` | 2.9 GB | `clip/` |

---

## Custom Nodes Installed (22 packs)

| Node Pack | Purpose |
|---|---|
| `ComfyUI-GGUF` | UnetLoaderGGUF, DualCLIPLoaderGGUF |
| `ComfyUI-LTXVideo` | LTX I2V pipeline nodes, LowVRAM loaders |
| `ComfyUI-VRGameDevGirl` | VRGDG workflow, prompt creator, audio split |
| `ComfyUI-KJNodes` | VAELoaderKJ, misc helpers |
| `ComfyUI-VideoHelperSuite` | VHS_VideoCombine for outputs |
| `ComfyUI-Erosdiffusion-LTX2` | FlowMatchEulerDiscreteScheduler |
| `ComfyUI-Manager` | Manager for updates |
| `ComfyUI_HeartMuLa` | Music generation with vocals |
| `ComfyUI-WanVideoWrapper` | Wan video support (unused) |
| `ComfyUI-LTX-2-3-LipSync` | IC-LoRA lip sync |
| `ComfyUI-SoundTracks` | Audio-driven motion tracks |
| `comfyui_memory_cleanup` | VRAM management |
| `comfy_mtb` | Utility nodes |
| `comfyui-custom-scripts` | Better combos, misc |
| `ComfyUI_Comfyroll_CustomNodes` | 175 misc nodes |
| `rgthree-comfy` | 48 utility nodes |
| `10S-Comfy-nodes` | Misc nodes |
| `WhatDreamsCost-ComfyUI` | Misc nodes |
| `ComfyUI-ErosDiffusion-ControlnetMaps` | Controlnet support |

---

## Pipeline Scripts

| File | Purpose |
|---|---|
| `pipeline_zimage_ltx.py` | 3 scenes, Z-Image → LTX, /free cleanup |
| `gen_mv_step_a.py` | 15 scenes, Z-Image GGUF refs, /free |
| `gen_mv_step_b.py` | 15 scenes, LTX video with audio, /free |
| `gen_music_video_pro.py` | End-to-end: lyrics → music → refs → video → stitch |
| `gen_mv_final.py` | IC-LoRA lip sync test (4 scenes) |
| `gen_1min_mv.py` | 1 minute multi-scene pipeline |

## Workflow JSONs

| File | Purpose |
|---|---|
| `workflow_local_lite_gguf.json` | Single scene LTX I2V (proven working) |
| `workflow_local_gguf_dual.json` | Same but DualCLIPLoaderGGUF (no sentencepiece) |
| `workflow_I2V_V5.1_lite.json` | Patched VRGDG I2V (GGUF models) |
| `workflow_T2V_V5.1_lite.json` | Patched VRGDG T2V (GGUF models) |

---

## Key Details

### VRAM Cleanup
POST `/free` with `{"free_memory": true}` between scenes to prevent OOM.

### Text Encoder on CPU
All pipelines use `device: "cpu"` for text encoder to save ~4 GB VRAM.

### Sentencepiece Bug
The `gemma_3_12B_it_fp4_mixed.safetensors` from inflatebot has incompatible tokenizer → use `workflow_local_gguf_dual.json` with DualCLIPLoaderGGUF instead.

### DualCLIPLoaderGGUF Paths
- `gemma-3-12b-it-qat-UD-Q4_K_XL.gguf` → symlinked from `text_encoders/` to `clip/`
- `ltx-2.3_text_projection_bf16.safetensors` → symlinked from `text_encoders/` to `clip/`

### Backup Drive
Mounted at `/media/ericr/backup/ltxmodel/` — contains full model inventory copy. Not needed for local operation.
