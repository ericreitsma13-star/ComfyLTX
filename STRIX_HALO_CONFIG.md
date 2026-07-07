# LTX-2.3 Music Video Creator V5.1 — Strix Halo Config

## Backup Drive Layout

```
/media/ericr/backup/ltxmodel/
├── checkpoints/              # Combined checkpoints
│   └── ltx-2.3-22b-distilled-1.1.safetensors        43 GB  ✅
├── unet/                    # Diffusion models (UNet)
│   ├── LTX-2.3-22B-distilled-1.1-Q6_K.gguf          20 GB  ✅
│   └── z_image_turbo_bf16.safetensors                12 GB  ✅
├── vae/                     # VAEs
│   ├── ae.safetensors                                320 MB ✅
│   ├── ltx-2.3-22b-distilled_audio_vae.safetensors   348 MB ✅
│   └── ltx-2.3-22b-distilled_video_vae.safetensors   1.4 GB ✅
├── text_encoders/           # Text encoders & CLIP
│   ├── gemma_3_12B_it_fp4_mixed.safetensors          8.8 GB ✅
│   ├── gemma-3-12b-it-qat-UD-Q4_K_XL.gguf            7.0 GB ✅
│   ├── gemma-3-12b-it-abliterated-sikaworld-...     14.1 GB ⏳
│   ├── ltx-2.3_text_projection_bf16.safetensors      2.2 GB ✅
│   └── qwen_3_4b.safetensors                         7.5 GB ✅
├── latent_upscale_models/   # Upscalers
│   ├── ltx-2.3-spatial-upscaler-x2-1.0.safetensors   950 MB ✅
│   └── ltx-2.3-spatial-upscaler-x2-1.1.safetensors   950 MB ✅
├── LLM/                     # LLM for prompt generation
│   ├── supergemma4-26b-uncensored-fast-v2-Q4_K_M.gguf  16 GB ✅
│   └── mmproj-BF16.gguf                               1.2 GB ✅
├── loras/                   # LoRAs
│   ├── ltx-2.3-22b-distilled-lora-384-1.1.safetensors  7.1 GB ✅
│   └── ltx-2.3-22b-ic-lora-lipdub.safetensors         2.3 GB ✅
├── gemma/                   # Full Gemma model (for training)
│   └── model-0000{1..5}-of-00005.safetensors         ~24 GB ✅
├── Z-Image-Turbo/           # Z-Image Turbo diffusers
│   └── (transformer, vae, text_encoder, tokenizer)    ~31 GB ✅
└── MODEL_INVENTORY.md       # This file
```

### Download Status (background curl, PIDs on original machine)

| Model | From | Size | PID | Progress |
|---|---|---|---|---|
| `gemma-3-12b-it-abliterated-sikaworld-...` | Sikaworld1990 HF | 14.1 GB | 4121272 | ~6.5 GB done |
| `ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors` | Kijai HF | 25.2 GB | 4130477 | ~1.2 GB done |

Logs: `/tmp/download_gemma.log`, `/tmp/download_fp8.log`

---

## Strix Halo Setup

### 1. Clone the repo
```bash
git clone https://github.com/ericreitsma13-star/ComfyLTX.git /home/ericr/ComfyUI
cd /home/ericr/ComfyUI
git remote add origin https://github.com/comfyanonymous/ComfyUI.git
# Keep eric remote for pushing custom code
```

### 2. Mount backup drive and link models
Copy `extra_model_paths.yaml` from repo root to ComfyUI root, then edit `base_path`:
```yaml
backup:
    base_path: /media/ericr/backup/ltxmodel/
    checkpoints: checkpoints/
    diffusion_models: |
        unet/
        checkpoints/
    vae: vae/
    text_encoders: text_encoders/
    clip: text_encoders/
    loras: loras/
    latent_upscale_models: latent_upscale_models/
    LLM: LLM/
```

### 3. Use the edited workflows (NOT the original VRGDG ones)
The original VRGDG workflow references `ltx-av-step-1751000_vocoder_24K.safetensors` — this file does NOT exist publicly. Instead use:

| Workflow | Description |
|---|---|
| `workflow_local_lite.json` | Simple LTX-2.3 music video |
| `workflow_i2v_audio_full.json` | Full I2V with audio |
| `workflow_clean_i2v_audio.json` | Clean I2V variant |
| `generate_music_video.py` | Python automation script |

---

## Recommended Model Config (96 GB Unified VRAM)

| Component | Model | Path on backup | VRAM |
|---|---|---|---|
| **UNet (primary)** | `LTX-2.3-22B-distilled-1.1-Q6_K.gguf` | `unet/` | ~14 GB |
| **Text Encoder** | `gemma_3_12B_it_fp4_mixed.safetensors` | `text_encoders/` | ~6 GB |
| **Text Projection** | `ltx-2.3_text_projection_bf16.safetensors` | `text_encoders/` | ~2 GB |
| **Video VAE** | `ltx-2.3-22b-distilled_video_vae.safetensors` | `vae/` | ~2 GB |
| **Audio VAE** | `ltx-2.3-22b-distilled_audio_vae.safetensors` | `vae/` | ~1 GB |
| **Z-Image Turbo** | `z_image_turbo_bf16.safetensors` | `unet/` | ~8 GB |
| **Upscaler** | `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | `latent_upscale_models/` | ~1 GB |
| **LLM** | `supergemma4-26b-uncensored-fast-v2-Q4_K_M.gguf` | `LLM/` | ~12 GB |
| **System overhead** | — | — | ~8 GB |
| **Total** | | | **~52 GB** ✅ |

### Alternative: if you want the Sikaworld text encoder instead
Once download finishes, replace `gemma_3_12B_it_fp4_mixed.safetensors` with `gemma-3-12b-it-abliterated-sikaworld-high-fidelity-edition.safetensors` in the DualCLIPLoader. Same VRAM.

### Do NOT use
- `ltx-2.3-22b-distilled-1.1.safetensors` (43 GB) — too large, unneeded with GGUF
- The original VRGDG workflow JSONs as-is (they reference a missing checkpoint)

---

## Key Paths Summary

| Resource | Location |
|---|---|
| Backup drive mount | `/media/ericr/backup/` |
| Models | `/media/ericr/backup/ltxmodel/` |
| Git repo | `https://github.com/ericreitsma13-star/ComfyLTX.git` |
| Custom scripts | `ComfyUI/gen_*.py`, `run_*.py`, `pipeline_*.py` |
| Workflow JSONs | `ComfyUI/workflow_*.json` |
| Custom nodes | `ComfyUI/custom_nodes/ComfyUI-VRGameDevGirl/` |
| Model config | `ComfyUI/extra_model_paths.yaml` |
