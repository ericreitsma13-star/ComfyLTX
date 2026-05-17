# ComfyUI Toolbox — Status

## Hardware
- **Device:** Bosgame M5 / AMD Strix Halo (gfx1151, Radeon 8060S)
- **Memory:** 96GB unified (80GB visible to ROCm)
- **OS:** Ubuntu (host) + Docker (kyuz0/amd-strix-halo-comfyui)

---

## ROCm / PyTorch

### Host (current)
- **ROCm 7.13 nightly** (`2.10.0+rocm7.13.0a20260513`)
- **Status:** GPU detected but kernel images broken — `Kernels.so` has gfx1100/1150 but **NOT gfx1151**
- **Fix needed:** Roll back to `2.9.1+rocm7.11.0a20260106` (same build from [bkpaine1's Strix Halo guide](https://github.com/bkpaine1/AMD-Strix-Halo-AI-Guide))
  ```
  source ~/Video_summary/.venv/bin/activate
  pip uninstall torch torchvision torchaudio -y
  pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ \
      torch==2.9.1+rocm7.11.0a20260106 torchvision torchaudio --force-reinstall
  ```

### Docker (kyuz0/amd-strix-halo-comfyui)
- **ROCm 7.2** (stable, known working on gfx1151)
- Used for all pipeline runs below

---

## Focus: LTX-2.3 19B Pipeline

The 22B model (46 GB) was abandoned due to memory constraints. Switching to the **19B model** with the LTX-2.3 custom DistilledPipeline — gives audio support, upscalers, and better quality at half the model size.

### Pre-cached files (nothing downloads on first run)

| File | Size | Location | Status |
|------|------|----------|--------|
| `ltx-2-19b-distilled-fp8.safetensors` | 27 GB | `comfy-models/diffusion_models/` | ✓ Complete |
| `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | 950 MB | `comfy-models/diffusion_models/` | ✓ Complete |
| `ltx-2.3-temporal-upscaler-x2-1.0.safetensors` | 250 MB | `comfy-models/diffusion_models/` | ✓ Complete |
| Gemma3 12B Q4 text encoder | 23 GB | HF cache (`google/gemma-3-12b-it-qat-q4_0-unquantized`) | ✓ Complete |
| Gemma tokenizer | 38 MB | `gemma_tokenizer/` | ✓ Complete |

### How it works
The 19B model (`ltx-2-19b-distilled-fp8.safetensors`) is a **full multi-component checkpoint** (6404 tensors) with:
- `AVTransformer3DModel` (48 layers, same architecture as 22B) — **not** just transformer weights
- Audio VAE decoder/encoder
- Video VAE decoder/encoder
- Vocoder
- All loaded by the custom `DistilledPipeline` from `/opt/LTX-2/`

The DistilledPipeline does two-stage generation:
1. Stage 1: Generate at half resolution (8 denoising steps)
2. Upsample via spatial upscaler
3. Stage 2: Refine at full resolution (4 denoising steps, no LoRA since 22B LoRA is incompatible)

### Scripts
- **`ltx23_19b_infer.py`** — Python inference script using DistilledPipeline + 19B checkpoint
- **`run_ltx23_19b.sh`** — Docker launcher: `bash run_ltx23_19b.sh --prompt "..." [options]`
- **`ltx23_19b_stage1.py`** — Stage 1 subprocess (half-res denoising, saves latents to disk)
- **`ltx23_19b_upscale.py`** — Upscale subprocess (spatial x2, reads/writes disk)
- **`ltx23_19b_stage2.py`** — Stage 2 subprocess (full-res refinement, reads/writes disk)
- **`ltx23_19b_decode.py`** — Decode subprocess (VAE decode + MP4 export)
- **`run_ltx23_19b_split.sh`** — **Docker launcher for subprocess pipeline** (4 separate Pythons)

---

## Map of Objects (Architecture)

### Object Relationship Diagram

```
checkpoint.safetensors (27 GB disk, 6404 tensors, FP8)
  ├─ AVTransformer3DModel (48 layers, 19B params)  ← the model
  ├─ Video VAE encoder/decoder                       ← pixel↔latent
  ├─ Audio VAE encoder/decoder                       ← audio↔latent
  └─ Vocoder                                          ← latent→waveform

DistilledPipeline (ltx_pipelines/distilled.py)
  ├─ PromptEncoder    ── Gemma3 12B Q4 text encoder  (6 GB in memory)
  ├─ ImageConditioner  ── VideoEncoder (for I2V encoding)
  ├─ DiffusionStage    ── AVTransformer3DModel (~19 GB FP8)
  ├─ VideoUpsampler    ── VideoEncoder + LatentUpsampler (~1 GB)
  ├─ VideoDecoder      ── VideoDecoder (~2 GB)
  └─ AudioDecoder      ── AudioDecoder + Vocoder (~1.5 GB)

Pipeline Call Flow (single process):
  prompt_encoder([prompt])
    └─ Gemma loaded → encode → freed ───┐
                                         ├─→ video_context, audio_context
    ┌────────────────────────────────────┘
    ▼
  Stage 1 (@ half-res, 8 steps):
    DiffusionStage.__call__()
      └─ BlockStreaming: each of 48 layers streamed CPU→GPU one at a time
         ├─ FP8 weights (in CPU RAM) → cast to BF16 on GPU
         └─ KV cache + activations on GPU (freed per-step with streaming)
      └─ Returns (video_state, audio_state) — denoised latents
    │
    ▼
  Upscaler:
    VideoUpsampler(video_state.latent)
      └─ VideoEncoder + LatentUpsampler loaded → upscale → freed
    │
    ▼
  Stage 2 (@ full-res, 4 steps):
    DiffusionStage.__call__()   ← same transformer, larger KV cache
      └─ KV cache ~4× larger than stage 1
      └─ Returns refined (video_state, audio_state)
    │
    ▼
  VideoDecoder(video_state.latent) → decoded video iterator
  AudioDecoder(audio_state.latent) → decoded audio
    │
    ▼
  encode_video() → MP4
```

### Block Lifecycle (from ltx_pipelines/utils/blocks.py)

Each block is a **context manager** that loads its model on `__call__` and frees it on exit via `gpu_model()`:

```python
class gpu_model:
    with gpu_model(build_model()) as model:
        ...  # use model
    # model.to("meta") + cleanup_memory() called automatically
    # cleanup_memory() = gc.collect() + torch.cuda.empty_cache()
```

So model weights ARE freed between stages in the monolithic pipeline too. The OOM is NOT from model weights piling up — it's from **ROCm not returning freed GPU memory to the OS** when using `expandable_segments:True`.

### Every Block Used

| Block | Creates | Lifetime | Cleanup |
|-------|---------|----------|---------|
| PromptEncoder | Gemma3 + EmbeddingsProcessor | One call | `gpu_model()` on exit |
| ImageConditioner | VideoEncoder | One call | `gpu_model()` on exit |
| DiffusionStage | AVTransformer3DModel | One call or streaming | `streaming.teardown()` or `gpu_model()` |
| VideoUpsampler | VideoEncoder + LatentUpsampler | One call | `gpu_model()` on both |
| VideoDecoder | VideoDecoder | One call (lazy iterator) | `_cleanup_iter` on exhaustion |
| AudioDecoder | AudioDecoder + Vocoder | One call | `gpu_model()` on both |

### Why the Monolithic Pipeline OOMs (Root Cause)

**Strix Halo has unified memory** — the 93 GB pool serves both CPU and GPU. `OffloadMode.CPU` keeps 19B weights (~19 GB FP8) in "CPU RAM", but on Strix Halo that's the SAME pool GPU allocations come from. This does NOT free memory — it just relabels it.

The real OOM chain:
```
1. 19B weights @ FP8:  ~19 GB  (in "CPU" == unified pool)
2. Gemma3 Q4:          ~6 GB   (loaded in unified pool)
3. VAE + Vocoder:      ~3 GB   (loaded in unified pool)
4. Python overhead:    ~2 GB
5. ROCm HIP allocator: ~5 GB   (fragmentation + expansion)

Stage 1 starts:
6. KV cache @ half-res: ~5 GB  (GPU == same pool)
7. Activations:        ~10 GB  (GPU == same pool)
  → Total: ~50 GB (fits)

Stage 1 ends → memory freed via cleanup_memory()
  But ROCm with expandable_segments:True HOLDS the allocated segments
  Even though tensors are freed, the HIP arena stays expanded

Stage 2 starts:
8. KV cache @ full-res: ~22 GB (needs fresh allocation)
9. Activations:        ~20 GB  (needs fresh allocation)
  → ROCm expands arena to ~90 GB
  → Fragmentation pushes arena past 93 GB → OOM

On a discrete GPU this wouldn't happen: CPU RAM (64 GB) holds weights,
GPU VRAM (24 GB) holds KV cache + activations. On Strix Halo, they
compete for the same 93 GB.
```

### Why Subprocess Fixes It

Each stage runs in its own Python process. When `python stageX.py` exits:
1. All Python objects destroyed (gc.collect runs automatically)
2. All torch tensors freed (refcount drops to zero)
3. ROCm/CUDA context destroyed (driver releases all GPU memory)
4. Python process exits → OS reclaims ALL memory (both CPU and GPU)
5. Next stage starts with a clean 93 GB pool

Peak memory per stage (vs ~90 GB for monolithic):
- Stage 1: ~45 GB (weights + KV cache @ half-res)
- Upscale: ~25 GB (encoder + upsampler only)
- Stage 2: ~50 GB (weights + KV cache @ full-res)
- Decode: ~25 GB (VAE decoders only)

---

## Models (other)

#### Image Generation
| Model | File | Approach | Speed |
|-------|------|----------|-------|
| **Qwen-Image-2512 + fp8** | `qwen_image_2512_fp8_e4m3fn.safetensors` (20GB) | 1024→1328 upscale, Lightning 4-step LoRA | cold 46s, warm 15s |

Script: `warm_test_pipeline.py` — loads bf16 model from HF + streams fp8 unet weights

#### LTX-2.3 22B (archived)
| File | Size | Status |
|------|------|--------|
| `ltx-2.3-22b-distilled-1.1.safetensors` | 46 GB | ✓ Downloaded but not used — too memory-heavy |
| `ltx-2.3-22b-distilled-lora-384-1.1.safetensors` | ~8 GB | ✗ Not downloaded (incompatible with 19B anyway) |

---

## Environment Variables (Strix Halo)

From [bkpaine1's guide](https://github.com/bkpaine1/AMD-Strix-Halo-AI-Guide) + Reddit:

```bash
HSA_ENABLE_SDMA=0            # Prevents GPU hangs on gfx1151
HSA_USE_SVM=0                # Fixes VRAM crashes
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1  # Faster attention
AMD_SERIALIZE_KERNEL=3       # Detailed error logging

# Memory management
PYTORCH_HIP_ALLOC_CONF="backend:native,expandable_segments:True,garbage_collection_threshold:0.7,max_split_size_mb:256"
PYTHONMALLOC=malloc
MALLOC_TRIM_THRESHOLD_=100000
```

Already applied in `run_ltx2.sh`.

---

## Optimization Tips (from Reddit)

| Source | Tip |
|--------|-----|
| [bkpaine1 guide](https://github.com/bkpaine1/AMD-Strix-Halo-AI-Guide) | TheROCk 7.11 is **dramatically faster** than ROCm 7.2 |
| [StrixHalo post](https://www.reddit.com/r/StrixHalo/comments/1qib8ch/) | LTX-2 audio-video sync, I2V, talking characters all work |
| [Windows guide](https://www.reddit.com/r/comfyui/comments/1rlx2og/) | PYTORCH_HIP_ALLOC_CONF + memory tweaks |
| [User report](https://www.reddit.com/r/StableDiffusion/comments/1qmlc1t/) | ROCm 7.2 → 5s video in 5 min (4x boost over older drivers) |
| [Docker post](https://www.reddit.com/r/StrixHalo/comments/1t18l5c/) | Alternative image: `ignatberesnev/comfyui-gfx1151` (self-managed) |
| [LTX-2 audio sync workflow](https://www.reddit.com/r/StableDiffusion/comments/1qd525f/) | **Distill LoRA strength 0.6** best for realistic people. Detail LoRA at 0.3. Include lyrics/transcript in prompt for lipsync. Resolution must be ÷32. |
| [LTX-2 audio sync v1](https://www.reddit.com/r/StableDiffusion/comments/1qcc81m/) | Uses **dev-fp8 + distilled LoRA** (not distilled-only model) for adjustable LoRA strength. Gemma fp8 text encoder available. Tiled VAE. Low-res preview before upscale. |
| ["Bad LTX2 results? You're using it wrong"](https://www.reddit.com/r/StableDiffusion/comments/1qqewis/) | Common mistakes and fixes. LTX botched release making it hard to get working. |
| [LTX-2 Mastering Guide](https://www.reddit.com/r/StableDiffusion/comments/1rptnsg/) | Advanced prompt engineering, 4K/50FPS workflow, multi-shot sequencing. |
| [Stop using T2V — I2V Best Practices](https://www.reddit.com/r/StableDiffusion/comments/1q8dxon/) | I2V >> T2V for LTX. Image conditioning beats pure text prompts. |
| [LTX-2 on Jetson Thor](https://www.reddit.com/r/StableDiffusion/comments/1r03u80/) | 1080p pipeline, audio, camera control LoRAs. GitHub: `divhanthelion/ltx2` |

### Key LTX-2 Quality Tips
- **Distill LoRA strength 0.6** — better realistic people than default 1.0
- **Detail LoRA at 0.3** — optional quality boost, costs VRAM
- **Two-stage pipeline** (dev model + LoRA) > distilled-only model (no LoRA control)
- **Gemma fp8** text encoder available from `GitMylo/LTX-2-comfy_gemma_fp8_e4m3fn` (~4GB vs 24GB)
- **Resolution must be ÷32**, starting at 480x832 portrait
- **Include lyrics/transcript** in prompt for lipsync
- **ComfyUI v0.9.1+** for better memory management
- 4090 (24GB VRAM): 20s 1280p clips in 6-8 min

### LTX-2.3 specific (from GitHub readme)
- 22B params, improved quality over 19B
- Distilled 1.1 version available (46GB), 8-step inference
- Spatial upscaler x2 and x1.5 available (~1GB each)
- Temporal upscaler available (262MB)
- `fp8-cast` quantization for on-the-fly memory savings (pure PyTorch, no CUDA deps)

### Docker images available
- `kyuz0/amd-strix-halo-comfyui:latest` — Fedora Toolbox, ROCm nightly, curated. **Current.**
- `ignatberesnev/comfyui-gfx1151` — Standard Docker, ROCm 7.2 stable, update-yourself.

### LTX-2 Prompting
From [LTX-2 GitHub](https://github.com/Lightricks/LTX-2): chronological action descriptions, camera angles, under 200 words. Avoid keyword soup.

---

## What to Use Now

### LTX-2.3 GGUF Mode (No Downloads During Run)

This toolbox now has a GGUF path intended to avoid the 22B safetensors OOM issue.

Scripts:
- `download_ltx23_gguf.sh` — one-time aria2 downloads (manual)
- `warm_ltx23_gguf_cache.sh` — pre-reads model files from disk to warm cache
- `run_ltx23_gguf.sh` — launches ComfyUI only; performs **no downloads**

Run order:
```bash
bash /home/eric-reitsma/comfyui_toolbox/download_ltx23_gguf.sh
bash /home/eric-reitsma/comfyui_toolbox/warm_ltx23_gguf_cache.sh
bash /home/eric-reitsma/comfyui_toolbox/run_ltx23_gguf.sh
```

Required files for distilled-1.1 GGUF:
- `unet/ltx-2.3-22b-distilled-1.1-UD-Q4_K_M.gguf`
- `vae/ltx-2.3-22b-distilled_video_vae.safetensors`
- `vae/ltx-2.3-22b-distilled_audio_vae.safetensors`
- `text_encoders/ltx-2.3-22b-distilled_embeddings_connectors.safetensors`
- `text_encoders/gemma-3-12b-it-qat-UD-Q4_K_XL.gguf`
- `text_encoders/mmproj-BF16.gguf`

Optional extras:
- `loras/ltx-2.3-22b-distilled-lora-384.safetensors`
- `latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.0.safetensors`
- `latent_upscale_models/ltx-2.3-temporal-upscaler-x2-1.0.safetensors`

Compatibility notes (toolbox + ROCm):
- GGUF is **not** used by `ltx23_infer.py` / `ltx23_19b_*.py` (those are safetensors DistilledPipeline scripts).
- GGUF runs through ComfyUI workflows with `ComfyUI-GGUF` and `ComfyUI-KJNodes`.
- On this system (Docker ROCm 7.2 + Strix Halo), this route is expected to work and is the preferred 22B path vs monolithic safetensors.

### Primary: LTX-2.3 19B Video (Subprocess Pipeline — RECOMMENDED)
```bash
bash run_ltx23_19b_split.sh --prompt "Your prompt here"
```
Runs 4 separate Python processes (stage1 → upscale → stage2 → decode), exiting between each. This is the ONLY reliable approach on Strix Halo unified memory — each process exit returns ALL memory to the OS.

Options: `--width 768 --height 512 --frames 97 --fps 24 --seed 42 --image /path/to/i2v.png`
Resume: `--skip-stage 3` picks up from stage 3 if stages 1-2 completed.

### Legacy: LTX-2.3 19B Video (Single Process — MAY OOM)
```bash
bash run_ltx23_19b.sh --prompt "Your prompt here"
```
Same underlying pipeline but all in one process. Can OOM on 93 GB unified memory due to ROCm memory fragmentation. Prefer the split pipeline above.

### Also available
- **Qwen Image gen** — `bash run_pipeline_test.sh` (or `warm_test_pipeline.py` directly)
- **SDXL** — `bash run_sdxl.sh` (or `bash run_sdxl.sh <pack-name>` for content packs)
- **Content packs** — `bash run_sdxl.sh architectural-watercolor` etc. Packs live in `content_packs/`

### Needs TheROCk fix (host)
- TheROCk 7.13 has broken kernel images for gfx1151. Docker runs fine on ROCm 7.2.
- Roll back to `2.9.1+rocm7.11.0a20260106` for host-side ROCm.

---

## SDXL Compatibility

**Verdict: Runs fine on existing Docker image.** No changes needed.

| Component | Status |
|-----------|--------|
| Docker image | `kyuz0/amd-strix-halo-comfyui` has `StableDiffusionXLPipeline` in diffusers 0.36.0 + full ComfyUI SDXL model classes |
| ROCm | ROCm 7.2 (Docker) — same env vars work (`HSA_ENABLE_SDMA=0`, etc.) |
| Memory | SDXL needs ~8-12GB at 1024². Strix Halo has 86GB unified — massive overkill |
| Model needed | ~7GB download (`stabilityai/stable-diffusion-xl-base-1.0` from HF, or safetensors mirror) |
| ComfyUI workflows | Not yet created, but diffusers approach works immediately |

Minimal test script would mirror `warm_test_pipeline.py` but with `StableDiffusionXLPipeline` instead of Qwen pipeline.

---

## Content Packs — Market Research

### Top underserved niches for commercial LoRA packs (SDXL base, OpenRAIL-M license)

| Niche | Why | Price |
|-------|-----|-------|
| **Architecture / interior design** | Pros pay, almost zero supply. Styles: Japandi, Art Deco, Brutalism, Maximalist | $15-50/bundle |
| **Product photography** | E-com sellers need consistent product shots. White background, beverage, jewelry, macro | $15-50/bundle |
| **Fashion design** | Technical flats, fabric draping, specific designer mimicry | $10-30 |
| **Tattoo design** | American traditional, Japanese, tribal — tiny market, loyal, professionals | $10-20 |
| **Watercolor / ink wash** | Very few good ones exist | $10-20 |
| **Medical / scientific illustration** | Anatomical diagrams, cellular visualization — untapped | $20-50 |
| **Industrial design** | Product concept sketches, CAD render mimicry | $15-40 |

### Marketplaces
- **CivitAI** — free + platform generation fees (Buzz). Top LoRAs get 100-200K downloads
- **Gumroad / Ko-fi** — direct sales. $3-15 per LoRA, $15-50 for bundles
- **Patreon** — $5-25/mo subs
- **Tensor.Art** — growing, strong in semi-real/anime

### Commercial model license matrix
| Model | Sell LoRAs? | Notes |
|-------|-------------|-------|
| **SDXL** | Yes (OpenRAIL-M) | Safest, largest ecosystem |
| **SD 1.5** | Yes (OpenRAIL-M) | Legacy but works |
| **SD3.5** | Yes | Needs Stability membership ($20/mo) |
| **Flux Dev** | Complicated | Recent policy restrictions, avoid |
| **Illustrious** | Yes | Best for anime/semi-real |
| **Wan2.2** | Yes (Apache 2.0) | Video LoRAs |
| **Qwen-Image-2512** | Yes (Tongyi Qianwen) | Small LoRA ecosystem |

### Format
- `.safetensors` files are universal. Bundle: LoRA + trigger word guide + example prompts + ComfyUI workflow (bonus).

### Strategy
SDXL base → train on professional/industry-specific styles → sell bundles on Gumroad → cross-post free on CivitAI for Buzz. Architecture/interior design is the single best gap.
