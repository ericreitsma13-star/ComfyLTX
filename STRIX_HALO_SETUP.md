# Strix Halo (Mimic PC) Cloud Config

## Hardware
- 48 GB unified VRAM
- 90 GB disk budget
- 64 GB+ system RAM

## Model Selection (82 GB)

```
models/unet/
  LTX-2.3-22B-distilled-1.1-Q6_K.gguf          20 GB  ← main UNet

models/text_encoders/
  gemma-3-12b-it-abliterated-sikaworld-*.safetensors  14 GB  ← workflow default
  ltx-2.3_text_projection_bf16.safetensors      2 GB

models/diffusion_models/
  z_image_turbo_bf16.safetensors                12 GB  ← Z-Image ref gen

models/text_encoders/
  qwen_3_4b.safetensors                          7 GB  ← Z-Image CLIP

models/vae/
  ae.safetensors                               0.3 GB
  ltx-2.3-22b-distilled_audio_vae.safetensors  0.3 GB
  ltx-2.3-22b-distilled_video_vae.safetensors  1.4 GB

models/latent_upscale_models/
  ltx-2.3-spatial-upscaler-x2-1.1.safetensors    1 GB

models/loras/
  ltx-2.3-22b-distilled-lora-384-1.1.safetensors 7 GB

models/LLM/
  supergemma4-26b-uncensored-fast-v2-Q4_K_M.gguf 16 GB
  mmproj-BF16.gguf                                1 GB
```

## Avoid 43 GB Checkpoint

Symlinks replace the 43 GB checkpoint:
```bash
cd ComfyUI/models/checkpoints
ln -s ../vae/ltx-2.3-22b-distilled_audio_vae.safetensors .
ln -s ../text_encoders/ltx-2.3_text_projection_bf16.safetensors .
```

## Workflow
Original I2V workflow — no lite version needed:
```
custom_nodes/ComfyUI-VRGameDevGirl/Workflows/
  LTX-2_Workflows/LTX 2.3 Music Video Creator V5.1/
    LTX2.3_Music_Video_Creator_I2V_V5.1.json
```

## Model Paths
Copy `extra_model_paths.yaml` from this repo root. Adjust `base_path` for Strix Halo mount point.
