# Agent Memory — ComfyUI Ideogram + SUPIR

## Nettie — Stedin Character (Jul 2026)

### Overview
Anthropomorphic female energy creature mascot for Stedin (Dutch grid operator).
Hybrid lightning elemental with humanoid features, wearing Stedin work uniform.

### Stedin Brand Colors
| Color | Hex | Role |
|-------|-----|------|
| **Yellow/Gold** | `#F3D400` | Primary |
| **White** | `#FFFFFF` | Secondary |
| **Dark Grey** | `#4F5150` | Text/outline |

### Logo
- File: `/home/ericr/Downloads/Stedin-logo-300x150.png` (RGBA, transparent bg)
- Overlay using `CR Overlay Transparent Image` node
- Position: x=400, y=350, scale=0.5 (on chest area)

### Chosen Model: Krea 2 Turbo
Best results for hybrid character design. Prompt:
```
anthropomorphic female energy creature, humanoid lightning elemental,
glowing golden-yellow eyes, lightning bolt shaped horns and ears,
crackling electricity patterns across translucent skin,
white and gold Stedin work uniform with #F3D400 yellow accents,
Stedin company logo on chest, safety helmet with lightning emblem,
electricity and energy theme, glowing energy core visible in chest,
semi-realistic stylized character, friendly expression,
warm golden lighting, professional portrait, 960x544
```

### Reference Images Generated
- **Krea 2**: 12 images (`nettie_krea2_*`) — best hybrid look
- **Z-Image**: 8 images (`nettie_zimage_*`) — fast, good variation
- **Ideogram 4**: 9 images (`nettie_ideogram4_*`) — structured prompts
- **Logo overlays**: 3 images (`nettie_logo_*`) — with Stedin logo on chest

### Scripts
- `nettie_refs.py` — Generate Nettie across all 4 models
- `nettie_logo_overlay.py` — Add Stedin logo to images

### Next Steps
1. Select best Krea 2 reference for animation
2. Add Stedin logo overlay to selected refs
3. Animate with LTX Video (same proven pipeline)
4. Test character consistency across clips

## Markthal Rotterdam Scene (Jul 2026)

### Best Approach Found
**Krea 2 img2img** with aerial Markthal photo + denoise=0.6 produced the best result.
- Used Wikimedia Commons aerial photo (`markthal_aerial_2.jpg`)
- Prompt: `running on the curved grey roof of a large horseshoe-shaped market hall building, glass facade, yellow cube apartments in background`
- Denoise=0.6 preserved Markthal structure while adding Nettie

### Issues to Fix
1. **Nettie too large/monster** — need smaller scale or different pose
2. **Artifacts** — small duplicate Nettie appearing on street (denoise too high)
3. **Background removal** — Krea 2 output has golden background that's hard to remove cleanly
4. **No perfect starting frame yet** — still iterating

### Files
- `markthal_aerial_2.jpg` — Best aerial Markthal photo (Wikimedia Commons)
- `step1_nettie_on_roof_342_00001_.png` — Best starting frame (Nettie on roof, denoise=0.6)
- `nettie_greenscreen_00001_.png` — Nettie with green screen background
- `markthal_refs.py` — Markthal reference generator
- `nettie_refs.py` — Nettie reference generator

### Workflow (from user)
1. Krea 2 img2img → perfect starting frame
2. LTX-Video I2V → animate (moderate denoise 0.55-0.70)
3. Krea 2 upscale → add lifelike texture
- Use architecture shapes in prompt, not "Markthal" name
- Keep CFG 3.0-4.0, Steps 40-50
- Denoise 0.55-0.70 for I2V

### Aerial Photos Downloaded
- `markthal_aerial_1.jpg` — WTC rooftop panorama (2953x1969)
- `markthal_aerial_2.jpg` — Aerial view with roof (4500x3000) — **BEST**
- `markthal_aerial_3.jpg` — Aerial view alternate (4500x3000)
- `markthal_aerial_4.jpg` — From Sint-Laurenskerk (2592x1944)
- `markthal_aerial_5.jpg` — 2015 aerial (3264x2448)

## Krea 2 Status (as of Jun 30 2026)

**Krea 2 Turbo running locally** on RTX 4090 16GB VRAM.

### Models Downloaded
| Model | File | Size | Location |
|-------|------|------|----------|
| Krea 2 Turbo FP8 | `krea2_turbo_fp8_scaled.safetensors` | 13GB | `diffusion_models/` |
| Krea 2 RAW FP8 | `krea2_raw_fp8_scaled.safetensors` | 13GB | `diffusion_models/` |
| Krea 2 RAW BF16 | `krea2_raw_bf16.safetensors` | 13GB | `diffusion_models/` |
| Qwen3-VL-4B FP8 | `qwen3vl_4b_fp8_scaled.safetensors` | 4.9GB | `text_encoders/` |
| Qwen Image VAE | `qwen_image_vae.safetensors` | 243MB | `vae/` |

### Key Fixes Applied
- ComfyUI updated to latest (native Krea 2 support via `comfy/text_encoders/krea2.py`)
- `comfy-kitchen` updated from 0.2.10 to 0.2.15 (fixed FP8 text encoder loading)

### Workflow
- Template: `workflows/krea2_turbo_t2i.json`
- Uncensored workflow: `workflows/krea2_turbo_uncensored.json`
- Test script: `test_krea2.py`
- Uncensored test script: `test_krea2_uncensored.py`
- Node chain: `UNETLoader → CLIPLoader(krea2) → TextEncodeKrea2 → ConditioningZeroOut → EmptyLatentImage → KSampler(8 steps, euler, simple, cfg=1) → VAEDecode → SaveImage`
- Uncensored chain: `UNETLoader → CLIPLoader(krea2) → LoraLoaderModelOnly(MysticXXX) → CLIPTextEncode → ConditioningKrea2Rebalance(balanced, renormalize) → KSampler(12 steps, euler, beta, cfg=1) → VAEDecode → SaveImage`

### MysticXXX LoRA
- `MysticXXX_KREA2_v1.safetensors` (228MB) in `models/loras/`
- From CivitAI (model ID 3067313)
- Strength 1.0, applied to model only (not clip)

### ConditioningKrea2Rebalance
- Custom node: `ComfyUI-ConditioningKrea2Rebalance` (fork of nova452)
- Repo: `huwhitememes/comfyui-krea2-conditioning`
- Quality-preserving per-layer conditioning control
- Default: `balanced` preset, `renormalize=true`, `multiplier=1.0`
- Boosts deep detail layers without inflating conditioning magnitude

### FP8 Text Encoder Limitation
- FP8 Qwen3-VL works for **text-only** generation
- For **image references** (vision path), need BF16 text encoder
- Error: "add_stub not implemented for Float8_e4m3fn"

### Singer LoRA Training v1 (Completed)
- **Trained**: `singer_krea2_v1.safetensors` (218MB) in `models/loras/`
- **Trigger word**: `svsinger`
- **Training data**: 15 images in `ai-toolkit/training_data/singer_sdxl/`
- **Trained on**: Modal A100 80GB, ~41 min, 1500 steps
- **Model**: Krea 2 RAW BF16 (trained), Krea 2 Turbo FP8 (inference)

### Singer LoRA Training v2 (Setup Ready — Not Yet Trained)
Standalone clone at `/home/ericr/ai-toolkit_v2/` (separate from existing `~/ai-toolkit/`).

**Source**: AI_Characters training configs from Reddit/CivitAI
- Pinned commit: `f63221e577053e86c2a673adec20e43d7b81988d`
- 4 replacement files applied (custom flowmatch sampler, base_trigger_preservation, SDTrainer)
- **Key improvement**: `timestep_type: sigmoid_linear_shift` (sigmoid-linear-blend distribution)

**Dataset**: 30 images with SCALISG captions at `training_data/singer_v2/`
- 15 from original `singer_sdxl/` + 9 from `singer/` + 6 from `input/singer_*_v2.png`
- All captions LLM-generated in natural-language paragraphs (80-220 words)
- Caption rules: `[trigger]` throughout, never describe eyes/hair/skin/age, describe mouth open/closed + shot type

**3 Configs (Krea 2 focus — others for reference)**:
| Config | arch | CFG | Steps | Model |
|--------|------|-----|-------|-------|
| `config/train_singer_krea2_v2.yaml` | krea2 | 1.0 | 8 | local `krea2/` diffusers dir |
| `config/train_singer_ideogram4_v2.yaml` | ideogram4 | 7.0 | 48 | local `ideogram4_fp8_scaled.safetensors` |
| `config/train_singer_klein9b_v2.yaml` | flux2_klein_9b | 4.0 | 50 | local `models/FLUX.2-klein-base-9B/` (48GB, downloaded via aria2c) |

**All share**: rank 8, LR 6e-05 AdamW constant, `sigmoid_linear_shift`, `base_trigger_preservation: true`, caption_dropout 0.05, buckets [768, 1024], 3000 steps

**To train** (ComfyUI must be stopped):
```bash
cd /home/ericr/ai-toolkit_v2 && source venv/bin/activate
# Krea 2 (best model — focus here)
python run.py config/train_singer_krea2_v2.yaml
# Others if needed:
python run.py config/train_singer_ideogram4_v2.yaml
python run.py config/train_singer_klein9b_v2.yaml
```

**Runner shortcut**: `bash run_all_training.sh`

**Checkpoint testing**: Start at 750 steps, then every 150 steps. Best typically 750-3000.

**CRITICAL**: Singer LoRA is Krea 2 architecture — CANNOT be applied to LTX Video (22B transformer). Different architectures. LoRA → generate static reference images → LTX (distilled LoRA only).

### Working LTX Pipeline (Jul 3 2026)
- **Pipeline**: `ltx_lipsync_fixed.py` with `--lora 0.8 --i2v 0.6 --cfg 3.5`
- **Model**: Q6_K GGUF + distilled LoRA only (no singer LoRA, no IC-LoRA)
- **References**: Generated with Krea 2 + singer LoRA, then fed to LTX
- **Resolution**: 960×544 @ 24fps
- **Duration**: 7.5s per clip (177 frames)
- **Sampler**: euler, linear_quadratic, 15 steps

**Working reference images** (all in `input/`):
| Ref | Setting | Status |
|-----|---------|--------|
| `ref_c_stage.png` | Stage, concert lighting | ✅ Good identity |
| `ref_d_bar.png` | Bar, neon signs | ✅ Good identity |
| `ref_e_golden_hour.png` | Outdoors, golden hour | ✅ Good identity |
| `ref_g_look_camera.png` | Looking at camera | ✅ Good identity |
| `ref_h_bokeh_wall.png` | Leaning on wall, bokeh | ✅ Good identity |
| `singer_studio_v2.png` | Studio, warm amber | ✅ Good identity |
| `singer_medium_v2.png` | Neon street, medium | ✅ Good identity |
| `ref_f_alley.png` | Dark alley | ⚠️ Slightly off identity |
| `singer_closeup_v2.png` | Close-up | ❌ Too zoomed, no lip sync |
| `singer_rain_v2.png` | Rain scene | ❌ Too artistic, no lip sync |

**Batch rendering**:
```bash
python3 -u ~/ComfyUI/batch_song.py
```
- Cycles through 7 good refs (stage, bar, golden hour, look camera, bokeh, studio, medium)
- 25 segments (segment_000 to segment_024)
- Output: `output/ltx_lipsync/ltx_lipsync_001XX-audio.mp4`
- ~5 min per clip, ~2 hours total

### Custom Node Fix (Required)
- `ComfyUI-LTXVideo` `pyramid_blending.py` fails to import due to kornia API change
- Error: `cannot import name 'pad' from 'kornia.geometry.transform.pyramid'`
- **Fix**: Apply git stash or manually patch `pyramid_blending.py`
- Without this fix, ALL LTXVideo nodes fail to register (LTXVSetAudioVideoMaskByTime, LTXVSpatioTemporalTiledVAEDecode, etc.)

### LoRA Training Plan
- Full plan: `plan_heatwave_lora_training.md`
- Train on Krea 2 RAW, infer on Krea 2 Turbo
- Config: `ai-toolkit/config/train_lora_krea2_16gb_v2.yaml`
- 15 training images in `ai-toolkit/training_data/singer_sdxl/`
- **Running on Modal** (A100 80GB, ~$1.50/hr): `modal_train_krea2.py`
- Monitor: `modal app list && modal app logs <app-id>`
- Download: `modal volume get krea2-lora-output / /tmp/output/`

### Mimic PC Workflow Candidates
| Workflow | Size | Status |
|----------|------|--------|
| HyperLoRA: Character Portrait Generation | 23.8 GB | ❌ NOT a training tool — uses pre-trained SDXL models, not Krea 2 |
| InstaLoRAm: Your Virtual Influencer Generator | 8.2 GB | Not yet investigated |
| AI InfluencerV2 | 22.7 GB | Not yet investigated |

## Workflow Files

| File | Description |
|------|-------------|
| `/home/ericr/Downloads/c0a2671af7c4_supir.json` | KJNodes-based Ideogram v4 + native SUPIR upscaling (active) |
| `/home/ericr/Downloads/c0a2671af7c4.json` | Original download (LLM prompt template version) |
| `/home/ericr/ComfyUI/user/default/workflows/c0a2671af7c4.json` | Base KJNodes workflow (source for KJNodes version) |

## Key Architecture

### Ideogram v4 Pipeline
- UNET: `ideogram4_fp8_scaled.safetensors` (diffusion_models/)
- Uncond UNET: `ideogram4_unconditional_fp8_scaled.safetensors` (diffusion_models/)
- CLIP: `qwen3vl_8b_fp8_scaled.safetensors` (text_encoders/)
- VAE: `flux2-vae.safetensors` (vae/)
- Prompt source: KJNodes `Ideogram4PromptBuilderKJ` (visual bbox editor)

### SUPIR Upscaling Branch
- SDXL checkpoint: `sd_xl_base_1.0.safetensors` (checkpoints/)
- SUPIR model: `SUPIR-v0Q_fp16.safetensors` (model_patches/, downloaded via aria2c)
- VAE: `ae.safetensors` (vae/, SDXL VAE from checkpoint)
- Nodes: CheckpointLoaderSimple → ModelPatchLoader → SUPIRApply → KSampler → VAEDecode → SaveImage

## CFG Tweaks
- **Recommended: Turbo preset, CFG 3.0** — best balance of quality/speed across all subjects
- Turbo cfg3 is clean and natural-looking on landscapes, portraits, animals, and still life
- Higher CFG (5, 7) overcooks portraits — too pronounced, harsh contrast/artifacts
- **Absurd/surreal prompts: CFG 5 wins** — more prompt adherence helps bizarre concepts, turbo cfg5 also acceptable
- Default preset (20 steps) marginal quality gain over Turbo (12 steps) not worth 1.7x time
- DualModelGuider CFG: 7 (unchanged, internal to Ideogram 4)

## Runtime Notes
- Launched with: `--lowvram --use-flash-attention` (required for 16GB VRAM)
- Flash attention confirmed active
- comfy-aimdo + DynamicVRAM detected

## Resolution
- KJNodes: 896×1152 (portrait, 5:7.2 ratio)
- EmptyLatentImage: matches at 896×1152 (same-res SUPIR)
- For upscaling: add ImageScale node before SUPIRApply, set both it and EmptyLatentImage to target

## Dimension Error Fix
- Error: `tensor a (192) != tensor b (112) at dim 3` — hint latent spatial dims must match target
- Hint latent encodes at VAE 8x compression: image_px / 8 = latent_dim
- Target latent matches EmptyLatentImage / 8
- Fix: ensure both hint image and empty latent are the SAME pixel resolution

## Models
- `SUPIR-v0Q_fp16.safetensors` (2.5GB, from Kijai/SUPIR_pruned via aria2c)
- `sd_xl_base_1.0.safetensors` (6.5GB, pre-existing)
- `ae.safetensors` (320MB, pre-existing SDXL VAE)

## Ideogram 4.0 Parameter Sweep Experiment

### Overview
Automated experiment that runs 36 prompts × 4 settings = **144 image generations** through the Ideogram 4 workflow, then builds a contact sheet for comparison.

### Pipeline
```
Natural language prompt
  → LLM (OpenCode mimo-v2.5) → Ideogram 4 structured JSON
  → ComfyUI API → Image generation
  → Contact sheet (PNG + HTML)
```

### Files

| File | Purpose |
|------|---------|
| `~/ComfyUI/experiment.py` | Main experiment script |
| `~/ComfyUI/prompt_builder.py` | LLM prompt→JSON transform (reusable) |
| `~/ComfyUI/experiment_template.json` | ComfyUI prompt format template (from history) |
| `~/ComfyUI/output/experiment/prompts.json` | Cached LLM transforms (raw + structured JSON) |
| `~/ComfyUI/output/experiment/*.png` | Generated images |
| `~/ComfyUI/output/experiment/contact_sheet.png` | PNG contact sheet |
| `~/ComfyUI/output/experiment/contact_sheet.html` | HTML contact sheet (click-to-copy prompts) |

### Prompt Format (Ideogram 4 JSON Schema)
```json
{
  "high_level_description": "one or two sentence summary",
  "style_description": {
    "aesthetics": "style keywords",
    "lighting": "lighting description",
    "photo": "camera/lens details",
    "medium": "photograph",
    "color_palette": ["#RRGGBB"]
  },
  "compositional_deconstruction": {
    "background": "scene shell description",
    "elements": [
      {"type": "obj", "bbox": [y_min, x_min, y_max, x_max], "desc": "element description"}
    ]
  }
}
```
- `compositional_deconstruction` is **required**
- Bbox: normalized 0-1000, `[y_min, x_min, y_max, x_max]`, origin top-left
- Style key order: `aesthetics`, `lighting`, `photo`/`art_style`, `medium`, `color_palette`
- Reference: https://github.com/ideogram-oss/ideogram4/blob/main/docs/prompting.md

### Experiment Grid

**6 subjects × 6 variations = 36 prompts:**

| Subject | Variations |
|---------|-----------|
| landscape | ocean, mountain, desert, forest, arctic, urban |
| portrait | studio, outdoor, elderly, child, group, profile |
| animals | wildlife, pet, bird, marine, insect, fantasy |
| absurd | surreal, scale, mashup, physics, time, dreamscape |
| still_life | food, flowers, vintage, organic, tech, mineral |
| architecture | modern, ancient, interior, exterior, ruin, futuristic |

**4 settings:**

| Name | Preset | CFG | Steps | Mu | Std |
|------|--------|-----|-------|----|-----|
| turbo_cfg3 | Turbo | 3.0 | 12 | 0.5 | 1.75 |
| turbo_cfg5 | Turbo | 5.0 | 12 | 0.5 | 1.75 |
| default_cfg3 | Default | 3.0 | 20 | 0.0 | 1.75 |
| default_cfg7 | Default | 7.0 | 20 | 0.0 | 1.75 |

### Key ComfyUI Nodes (Prompt Format)

| Node ID | Type | What it controls |
|---------|------|-----------------|
| `209` | Ideogram4PromptBuilderKJ | HLD, style, background, elements, bboxes |
| `134:115` | PrimitiveStringMultiline | Raw user prompt (clear when using HLD directly) |
| `98:157` | CFGOverride | CFG value |
| `98:156` | CustomCombo | Preset (Quality/Default/Turbo) |
| `98:17` | Ideogram4Scheduler | Steps, mu, std (wired from preset via JSON extraction) |
| `98:18` | RandomNoise | Seed |
| `98:24` | CLIPTextEncode | Positive prompt ( fed from node 209) |
| `158` | SaveImage | Output |

### Workflow Format
ComfyUI API `/prompt` expects **prompt format** (flat dict), NOT workflow format:
```json
{
  "node_id": {
    "inputs": {"key": value, "key2": ["other_node_id", slot]},
    "class_type": "NodeType",
    "_meta": {"title": "Node Title"}
  }
}
```
Subgraph nodes use `"subgraph_id:node_id"` notation (e.g., `"98:157"`).

### Running
```bash
# Use conda flash env (has flash-attn 2.8.3)
conda run -n flash python3 experiment.py
```
- Conda env at: `/home/ericr/miniconda3/envs/flash/` (PyTorch 2.12.0+cu130, flash-attn 2.8.3)
- First run: LLM transforms all 36 prompts, saves to `prompts.json`
- Subsequent runs: loads cached prompts, fills missing via LLM, skips existing images
- To re-transform: delete `prompts.json`
- Sequential processing: one generation at a time (~30s each, ~72 min total)

### Headless Single Image
```python
import json, sys
sys.path.insert(0, "/home/ericr/ComfyUI")
from prompt_builder import query_opencode, validate_and_fix

raw = query_opencode("A golden retriever on a skateboard")
caption = validate_and_fix(json.loads(raw))
# Patch experiment_template.json node "209" with caption, POST to /prompt
```

### Safety Filter Notes
- Ideogram 4 has built-in safety filter (not a ComfyUI node)
- False positives higher with non-JSON prompts
- Using structured JSON reduces false positive rate
- Reference: prompting.md safety section

## Ideogram 4.0 Text Rendering Experiment

### Overview
Tests literal text rendering in images — posters, banners, signs, packaging, covers, and real-life scenes with text. Single setting: **Turbo, CFG 3.0**. 40 prompts total.

### Files

| File | Purpose |
|------|---------|
| `~/ComfyUI/experiment_text.py` | Main experiment script |
| `~/ComfyUI/experiment_template.json` | ComfyUI prompt format template (shared) |
| `~/ComfyUI/output/experiment_text/prompts.json` | Cached LLM transforms |
| `~/ComfyUI/output/experiment_text/*.png` | Generated images |
| `~/ComfyUI/output/experiment_text/contact_sheet.png` | PNG contact sheet |
| `~/ComfyUI/output/experiment_text/contact_sheet.html` | HTML contact sheet (click-to-copy prompts) |

### Running
```bash
/home/ericr/miniconda3/envs/flash/bin/python3 ~/ComfyUI/experiment_text.py
```
- Same conda flash env as parameter sweep experiment
- Cached prompts: loads existing, transforms missing, skips existing images
- Sequential processing (~30s per image, ~20 min for 10 new prompts)
- ComfyUI must be running: `main.py --lowvram --use-flash-attention`

### Text Rendering in Ideogram 4 JSON Schema

Text uses `"type": "text"` elements with a `"text"` field for **literal text** to render:

```json
{
  "compositional_deconstruction": {
    "elements": [
      {
        "type": "text",
        "bbox": [50, 100, 150, 900],
        "text": "THE EXACT TEXT TO RENDER",
        "desc": "bold white sans-serif font, centered, large size",
        "color_palette": ["#FFFFFF", "#000000"]
      }
    ]
  }
}
```

Key rules:
- `"type": "text"` — tells the model this is text, not an object
- `"text"` field — the literal string rendered on the image (spell exactly right)
- `"desc"` — describes font style, color, material, size
- `"bbox"` — normalized 0-1000, `[y_min, x_min, y_max, x_max]`
- `"color_palette"` — per-element hex colors for precise text color control
- **Use `"medium": "graphic_design"`** for posters/banners/signs (not `"photograph"`)
- **Use `"art_style"` instead of `"photo"`** for non-photographic text layouts
- Combine text elements with `"type": "obj"` elements for background imagery

### Prompt Categories (40 total)

| Category | Count | Examples |
|----------|-------|---------|
| Movie/event posters | 5 | sci-fi, horror, comedy, concert, festival |
| Book/album covers | 5 | thriller novel, fantasy novel, hip-hop album, electronic album, magazine |
| Signs/neon | 5 | bar sign, diner sign, bakery, street sign, protest sign |
| Packaging/products | 5 | wine label, coffee bag, soda can, cereal box, tech box |
| Tickets/passes/cards | 4 | concert ticket, boarding pass, business card, movie stub |
| Banners/signage | 3 | grand opening, marathon finish, welcome mat |
| Digital/UI | 3 | app icon, game cover, t-shirt graphic |
| **Real-life scenes** | **10** | Tokyo street, diner, construction site, farmer's market, record store, bus stop, gym, laundromat, bookstore, airport gate |

### Findings
- **Turbo CFG 3 works well for text** — clean rendering, good prompt adherence
- **Simple text (1-3 words) renders reliably** — signs, titles, single phrases
- **Long text (4+ lines)** works but occasionally garbles characters
- **Graphic design medium** is key — using `"medium": "graphic_design"` + `"art_style"` for posters/banners/signs
- **Real-life scenes** are more compelling than isolated product shots — text appears naturally in context
- **Horror prompts** tend to trigger safety filter (false positives)
- **Kanji/CJK characters** render correctly when specified in the `"text"` field
- **Multiple text elements** in one image work — the model handles spatial layout

### LLM Transform for Text
The text experiment uses a specialized `text_render_transform()` function with a system prompt that:
- Instructs the LLM to use `"type": "text"` elements (not just `"obj"`)
- Emphasizes `"medium": "graphic_design"` for non-photo text layouts
- Specifies key order: `aesthetics`, `lighting`, `medium`, `art_style`, `color_palette`
- Returns raw JSON string for direct patching of node 209

### Known Bugs & Fixes

**DynamicCombo API format (node 209):**
- `Ideogram4PromptBuilderKJ.style` is a v3 `DynamicCombo` input
- API format: `"style": "photo"` (plain string) + `"style.photo": "85mm f/1.4"` (dotted sub-input key)
- NOT a dict — passing `{"style": "photo", "photo": "..."}` fails validation
- The API framework wraps string values into dicts via `build_nested_inputs()` using `dynamic_paths`

**LLM JSON parsing failures (text experiment):**
- Some LLM outputs fail JSON parse: `Expecting ',' delimiter` at high char positions
- This happens when the LLM produces complex nested JSON with many text elements
- Fallback: sends raw prompt text as `high_level_description` (no structured elements)
- Affects ~30% of prompts on first try; re-running deletes `prompts.json` to re-transform
- The raw prompt fallback still works but loses text-specific element control

**SUPIR template contamination:**
- `experiment_template.json` originally contained a separate SUPIR upscaling branch (nodes 200-208)
- SUPIR had a hardcoded prompt (`"high quality, detailed, sharp, 8k, masterpiece"`)
- The experiment's image download loop grabbed SUPIR output (node 208) instead of Ideogram 4 (node 158)
- Fix: removed SUPIR nodes, explicitly target node 158 in download logic

## LTX 2.3 GGUF Lip-Sync Pipeline

### Overview
Lip-synced video generation using LTX 2.3 + distilled LoRA on 16GB VRAM. Reference image + vocals audio → 7.5s lip-synced video at 960×544.

**Key insight**: Singer LoRA (Krea 2 architecture) CANNOT be applied to LTX Video (22B transformer). Identity comes from reference images generated with Krea 2 + singer LoRA, then animated with LTX (distilled LoRA only).

### Files

| File | Purpose |
|------|---------|
| `~/ComfyUI/ltx_lipsync_fixed.py` | Main lip-sync script (CLI tool, current active) |
| `~/ComfyUI/batch_song.py` | Batch renderer (25 segments, alternating refs) |
| `~/ComfyUI/generate_singer_refs.py` | Reference image generator (Krea 2 + singer LoRA) |
| `~/ComfyUI/input/ref_*.png` | Working reference images (7 verified) |
| `~/ComfyUI/input/segment_*.wav` | Audio segments (25 × 7.5s) |
| `~/ComfyUI/output/ltx_lipsync/ltx_lipsync_*.mp4` | Generated video clips |

### Running (Single Clip)
```bash
/home/ericr/miniconda3/envs/flash/bin/python3 ~/ComfyUI/ltx_lipsync_fixed.py \
  --image ref_c_stage.png \
  --audio segment_010.wav \
  --prompt "svsinger, female singer on neon-lit city street, subtle movement, dramatic neon lighting, cinematic portrait, photorealistic, detailed face, natural skin, 35mm film" \
  --width 960 --height 544 --duration 7.5 --seed 50 \
  --lora 0.8 --i2v 0.6 --cfg 3.5
```

### Running (Batch - Full Song)
```bash
python3 -u ~/ComfyUI/batch_song.py
```
- Cycles through 7 verified reference images
- 25 segments (segment_000 to segment_024)
- Output: `output/ltx_lipsync/ltx_lipsync_001XX-audio.mp4`
- ~5 min per clip, ~2 hours total

### Key Parameters
| Param | Value | Notes |
|-------|-------|-------|
| `--width` | 960 | Must be divisible by 32 |
| `--height` | 544 | Must be divisible by 32 |
| `--duration` | 7.5 | 7.5s = 177 frames @ 24fps |
| `--lora` | 0.8 | Distilled LoRA strength |
| `--i2v` | 0.6 | Image-to-video strength |
| `--cfg` | 3.5 | Classifier-free guidance |
| `--seed` | varies | `50 + segment_index * 13` |

### Model

| Component | File | Size | Notes |
|-----------|------|------|-------|
| UNET (Q6_K) | `LTX-2.3-22B-distilled-1.1-Q6_K.gguf` | ~13 GB | Main model |
| Distilled LoRA | `ltx-2.3-22b-distilled-lora-384-1.1.safetensors` | 7.1 GB | Quality enhancement |
| Video VAE | `ltx-2.3-22b-distilled_video_vae.safetensors` | 1.4 GB | Video decode |
| Audio VAE | `ltx-2.3-22b-distilled_audio_vae.safetensors` | 348 MB | Audio decode |
| Text Encoder | `gemma_3_12B_it_fp8_e4m3fn.safetensors` | 13 GB | Text encoding |
| Text Projection | `ltx-2-3-22b-text_encoder.safetensors` | 2.2 GB | Text projection |

### Workflow Architecture (ltx_lipsync_fixed.py)

```
UnetLoaderGGUF (Q6_K) → LoraLoaderModelOnly (distilled)
LTXAVTextEncoderLoader → clip
VAELoader → video_vae
LTXVAudioVAELoader → audio_vae

LoadImage → image
LoadAudio → LTXVAudioVAEEncode → audio_latent
EmptyLTXVLatentVideo → latent (960×544)
LTXVImgToVideoInplace (strength=0.6) → video_latent

LTXVConcatAVLatent(video + audio) → av_latent
LTXVSetAudioVideoMaskByTime (mask_video:1, mask_audio:0, SOTAI) → masked_av

LTXVConditioning → conditioning
RandomNoise + KSamplerSelect(euler) + BasicScheduler(linear_quadratic, 15 steps) + CFGGuider
SamplerCustomAdvanced → sampled_av

LTXVSeparateAVLatent → video + audio latents
LTXVSpatioTemporalTiledVAEDecode(spatial_tiles=2, temporal_tile_length=16)
LTXVAudioVAEDecode → audio
VHS_VideoCombine → .mp4
```

### Architecture Fixes (vs original ltx_lipsync_test.py)

| Fix | Problem | Solution |
|-----|---------|----------|
| Resolution | 720×1280 not divisible by 32 | 704×1280 (704/32=22) |
| Audio mask | `SolidMask+SetLatentNoiseMask` wrong shape for 4D audio | `LTXVSetAudioVideoMaskByTime` (SOTAI: mask_audio=False, preserve signal) |
| Character drift | No identity preservation | IC-LoRA guide with lipdub LoRA |
| Sampling | Wrong sigmas/sampler | `euler_ancestral_cfg_pp` + 8-step schedule |
| VAE decode | STD VAEDecodeTiled had boundary artifacts | `LTXVSpatioTemporalTiledVAEDecode` with proper normalization |
| End frame | Coherence decay at tail | `last_frame_fix=True`, temporal_overlap=2 |

### Experiment Summary (12 iterations)

| # | Model | I2V | IC-LoRA | LoRA | Duration | VAE | Result |
|---|-------|-----|---------|------|----------|-----|--------|
| 01 | Q6_K | 0.6 | frame0@1.0 | 0.3 | 6s | VAEDecodeTiled | OOM |
| 02 | Q6_K | 0.6 | frame0@1.0 | 0.3 | 5s | SpatioTemporal | end weird, lip sync good |
| 03 | Q6_K | 0.6 | dual(f0@1.0+f80@0.5) | 0.3 | 5s | SpatioTemporal | static, too constrained |
| 04 | Q6_K | 0.6 | dual(f0@1.0+f100@0.3) | 0.3 | 5s | SpatioTemporal | end still off, static |
| 05 | Q6_K | 0.6 | single f0@1.0 | 0.3 | 4s | VAEDecodeTiled | **best so far** |
| 06 | Q6_K | 0.6 | single f0@1.0 | 0.3 | 4s | VAEDecodeTiled | 4-step refine sigmas = trash |
| 07 | Q6_K | 0.6 | single f0@1.0 | 0.3 | 4s | VAEDecodeTiled | good, end slightly off |
| 08 | **Q4_K_M** | 0.6 | single f0@1.0 | 0.3 | 4s | VAEDecodeTiled | better, "few only now" |
| 09 | Q4_K_M | 0.6 | dual(f0@0.7+f88@0.2) | 0.3 | 4s | VAEDecodeTiled | stiff |
| 10 | Q4_K_M | **0.5** | single f0@**0.5** | **0.2** | 4s | VAEDecodeTiled | expressive, end "like negative" |
| 11 | Q4_K_M | 0.5 | single f0@0.5 | 0.2 | 4s | **SpatioTemporal** | expressive, end still bad |
| 12 | Q4_K_M | 0.5 | **single f88@0.35** (end only) | 0.2 | 4s | SpatioTemporal | ? (latest) |

### Remaining Issues

- **End frame weirdness**: Last 1-3 frames have artifacts (color inversion/"negative" or static/distorted). Present in all iterations. Suspected causes:
  1. LTX temporal coherence budget exhausted at ~4s
  2. SOTAI mask boundary at end_time=4.0s exceeding actual fc=89 frames (3.708s)
  3. VAE decode tile boundary at sequence end

- **IC-LoRA motion trade-off**: Higher strength = identity preserved but motion constrained. Lower strength = expressive but identity drifts. End-only guide may help but not fully tested.

### Untried Approaches
- **Two-stage latent upscale**: Official ComfyUI workflow approach (half-res gen → latent upscaler 2x → refinement pass). Would need restructuring, requires ~7GB extra VRAM (now available with Q4)
- **Post-process last frame**: Crop last 2 frames and blend/duplicate previous frame
- **Q5_K_M model**: Available from unsloth/LTX-2.3-GGUF, about 16GB file
- **LTX 2.3 native workflows**: Pre-built templates at ComfyUI → Workflow Templates → filter "LTX 2.3"
- **MSR LoRA approach**: liconstudio/ComfyUI-Licon-MSR for multi-subject reference with MSR LoRA

### ComfyUI Launch
```bash
systemctl --user start comfyui
# Flags: --lowvram --use-flash-attention (systemd service)
```

## Music Video Pipeline (LTX 2.3 I2V Only) — Heatwave

> **Active build plan: `plan_heatwave_music_video.md`**
> Lyric-driven broll spec, 12 unique Ideogram 4 refs, per-clip LTX params.
> Step 0+ not yet executed. Read that file first if you are picking up this work.

### Goal
Longform music video for Heatwave.wav (180s, 123 BPM, single singer, realism).
Use the **same proven LTX 2.3 + IC-LoRA pipeline** for both singer and broll
scenes (no SVI / Video-Infinity — see "Long-Video Path Abandoned" below).

### Current Architecture (as of Jun 24 2026)
- **All scenes use LTX 2.3 Q6_K** with the same prompt/LoRA/encoder stack as
  the 00026 winner
- Singer scenes: `ltx_lipsync_fixed.py` (audio-driven lip sync) — **singer is good, do not touch**
- B-roll scenes: `broll_generate.py` (same model, no audio path) — **being replaced by lyric-driven v2 (see plan)**
- Driver scripts: `singer_heatwave.py` and `broll_heatwave.py` orchestrators
- Workflow template exported from the 00026 MP4: `workflow_ltx_00026_*.json`

**Active working pipeline (laptop, NVIDIA RTX 4090 16GB):**
- Singer scenes: `ltx_lipsync_fixed.py` → LTX 2.3 Q6_K + distilled LoRA + audio VAE
  - 7.5s clips at 960×544, LoRA 0.8, I2V 0.6, CFG 3.5, euler, linear_quadratic, 15 steps
  - Ref: `heatwave_refs/portrait_v2.png` (Z-Image natural-skin)
  - Audio: `segment_NNN.wav` (7.5s chunks)
  - Outputs: `output/ltx_lipsync_00018-audio.mp4` through `00055-audio.mp4` (~7.4s each, ~1-2.5MB)
  - 00026 = best result (used segment_010.wav, seed 50)
- B-roll scenes: `broll_generate.py` → same model, no audio path
  - Refs: `broll_refs/{city_aerial,neon_bokeh,dance_floor,feet_puddle,skyline}.png`
  - 7.5s clips at 960×544, seed varies per clip
  - Driven by `broll_heatwave_serial.py` (one clip at a time, restart ComfyUI between)
  - 12 clips total at ~5 min each = ~70-90 min

### Long-Video Path Abandoned
- **SVI (Stable Video Infinity)** — error-recycling LoRA, 22B Q6_K OOMs on 16GB
- **Video-Infinity** (Yuanshi9815) — multi-GPU distributed, won't work on single GPU
- **Wan 2.2 I2V** — too large for laptop, only on cloud
- Strix Halo (96GB unified) could run SVI but user is on RTX 4090
- **Conclusion: long-video generation methods aren't viable for this hardware.
  Stick with the proven LTX 2.3 I2V + stitching approach.**

### Critical Pattern: Restart ComfyUI Between Clips
- 22B Q6_K model OOMs on 16GB if ComfyUI runs back-to-back clips (state leak)
- **Solution**: Restart ComfyUI between every broll render:
  ```bash
  systemctl --user restart comfyui
  # wait for /system_stats to return
  ```
- Built into `broll_heatwave_serial.py` (--no-restart flag for testing)
- Without restart, all clips after the first OOM with "Allocation on device 0 would exceed allowed memory"

### Files (Final, Jun 24 2026)

| File | Purpose |
|------|---------|
| `ltx_lipsync_fixed.py` | **Singer scenes** — proven 00026 winner, audio-driven lip sync |
| `singer_heatwave.py` | Singer scene orchestrator (18 clips via ltx_lipsync_fixed.py) |
| `broll_generate.py` | **B-roll scenes** — LTX 2.3 I2V, no audio path |
| `broll_heatwave.py` | B-roll orchestrator (batch mode, 12 clips in queue) |
| `broll_heatwave_serial.py` | **B-roll serial mode** — one clip at a time with ComfyUI restart |
| `beat_stitch.py` | Final assembly with crossfades and color matching |
| `workflows_export.py` | Export 00026 winning workflow as reusable API JSON templates |
| `workflow_ltx_00026_lipsync.json` | Exported 00026 winning workflow (24 nodes, audio) |
| `workflow_ltx_00026_broll.json` | Exported broll variant (17 nodes, audio stripped) |

**Removed (SVI/abandoned):** `audio_analyzer.py`, `lyrics_parser.py`, `svi_workflow.py`,
`svi_runner.py`, `broll_director.py`, `music_video_director.py`, `test_svi_chorus.py`,
`prompt_stream_generator.py`

### Models (all in /home/ericr/ComfyUI/models/)

| Model | Path | Size | Used by |
|-------|------|------|---------|
| `LTX-2.3-22B-distilled-1.1-Q6_K.gguf` | `diffusion_models/` | ~13GB | Singer + B-roll UNet |
| `ltx-2.3-22b-distilled-1.1.safetensors` | `diffusion_models/` | ~20GB | Singer audio VAE |
| `ltx-2.3-22b-distilled_video_vae.safetensors` | `vae/` | ~250MB | Singer + B-roll video VAE |
| `gemma_3_12B_it_fp8_e4m3fn.safetensors` | `text_encoders/` | ~6GB | Text encoder (T5) |
| `ltx-2-3-22b-text_encoder.safetensors` | `text_encoders/` | ~2GB | Text projection |
| `ltx-2.3-22b-distilled-lora-384-1.1.safetensors` | `loras/` | ~7GB | Distilled LoRA |
| `wan2.1-i2v-14b-480p-Q4_K_M.gguf` | `diffusion_models/` | ~7GB | SVI (unused, kept) |
| `Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors` | `diffusion_models/` | 17GB | SVI (unused, kept) |
| `svi_*.safetensors` | `loras/svi_wan21/version-1.0/` | 2.4GB each | SVI (unused, kept) |

### Music Video Plan for Heatwave (180s, 123 BPM)

| Scene | Section | Type | Duration | Model | Ref |
|-------|---------|------|----------|-------|-----|
| 0 | intro | broll | 12s | LTX 2.3 I2V | city_aerial |
| 1 | verse1 | singer | 24s | LTX 2.3 + IC-LoRA | singer_01 (portrait_v2) |
| 2 | prechorus1 | singer | 12s | LTX 2.3 + IC-LoRA | singer_01 |
| 3 | chorus1 | broll | 24s | LTX 2.3 I2V | feet_puddle |
| 4 | verse2 | singer | 24s | LTX 2.3 + IC-LoRA | singer_01 |
| 5 | prechorus2 | singer | 12s | LTX 2.3 + IC-LoRA | singer_01 |
| 6 | chorus2 | broll | 24s | LTX 2.3 I2V | dance_floor |
| 7 | bridge | broll | 18s | LTX 2.3 I2V | skyline |
| 8 | solo | broll | 6s | LTX 2.3 I2V | neon_bokeh |
| 9 | chorus3 | singer | 18s | LTX 2.3 + IC-LoRA | singer_01 |
| 10 | outro | broll | 6.3s | LTX 2.3 I2V | city_aerial |

### Broll Concept Bank

| Concept | Mood | Motion | File |
|---------|------|--------|------|
| city_aerial | atmospheric, mysterious | slow aerial push-in over neon city | `broll_refs/city_aerial.png` |
| neon_bokeh | intimate, moody | shallow DOF pan over bokeh | `broll_refs/neon_bokeh.png` |
| dance_floor | rising tension | camera pushes toward dance floor | `broll_refs/dance_floor.png` |
| feet_puddle | triumphant, kinetic | low-angle feet walking through neon puddle | `broll_refs/feet_puddle.png` |
| skyline | expansive, contemplative | slow cinematic reveal | `broll_refs/skyline.png` |

### Settings (SVI on Wan 2.1 I2V 480P)

| Param | Value | Notes |
|-------|-------|-------|
| `width` | 832 | Native SVI training res |
| `height` | 480 | Native SVI training res |
| `num_frames` | 81 | 3.24s per clip @ 25fps |
| `cfg` | 5.0 | Per SVI FAQ Q3 |
| `lora_strength` | 1.0 | Full SVI strength |
| `vram_blocks_to_swap` | 20 | For 16GB; reduce to 5-10 for 48GB |
| `motion_latent_count` | 5 (Film) / 1 (Shot) | Per SVI FAQ Q5 |
| `steps` | 30 | Default |
| `sampler` | euler / uni_pc | Both work |
| `scheduler` | normal | |
| `flow_shift` | 5.0 | For 480p |
| `seed` | 42 + i*137 | Different per clip (CRITICAL) |

### Test Commands

```bash
# Smoke test: render SVI-Film on chorus 1 (dry run)
python test_svi_chorus.py --dry-run

# Real render (after Wan 2.1 I2V download completes)
python test_svi_chorus.py

# Full MV orchestration
python music_video_director.py \
  --audio input/Heatwave.wav \
  --lyrics input/Heatwave_lyrics.txt \
  --singer-ref input/heatwave_singer.png \
  --broll-refs-dir output/broll_refs \
  --output-dir output/heatwave_v2 \
  --url http://<cloud-url>:8188

# Just broll, skip singer (faster iteration)
python music_video_director.py \
  --audio input/Heatwave.wav --lyrics input/Heatwave_lyrics.txt \
  --skip-singer --output-dir output/heatwave_broll_test
```

### Test Results (Jun 23 2026)

✅ **Pipeline works end-to-end** on laptop with Strix Halo 16GB VRAM:
- `audio_analyzer.py` — BPM 123.05, 16 sections detected (after noise-section filtering)
- `lyrics_parser.py` — clean parsing of [Section] tags
- `svi_workflow.py` — valid ComfyUI workflow with VACE + I2V + SVI LoRA pattern
- `svi_runner.py` — queues to ComfyUI, runs SVI-Film, downloads output
- `broll_director.py` — scene plan + prompt stream generation
- `music_video_director.py` — full orchestration (LTX singer + SVI broll)

**Hardware notes:**
- Wan 2.1 14B I2V fp8 (17GB) **OOMs on 16GB VRAM** even with full block swap
- **Use the GGUF version** `wan2.1-i2v-14b-480p-Q4_K_M.gguf` (~7GB) for 16GB VRAM
- On cloud 48GB VRAM: use fp8 safetensors (17GB) for better quality, or GGUF for faster
- **Laptop test (Strix Halo)**: ~25-30 min per 81-frame clip with GGUF + 20 block swap
- **Cloud estimate**: 3-5 min per clip on 48GB VRAM

**Key workflow fix (svi_workflow.py):**
- Use `WanVideoModelLoader` (not `UnetLoaderGGUF`) — auto-detects format
- For GGUF: `quantization: "disabled"` (not `fp8_e4m3fn`)
- Pattern: `LoadImage → WanVideoVACEStartToEndFrame → WanVideoImageToVideoEncode → WanVideoSampler`
- Required sampler field: `riflex_freq_index: 0`
- Valid schedulers: `euler`, `unipc`, `dpm++`, `lcm`, etc. (NOT `normal`)

### SVI LoRA Strengths Found

Tested with `svi_wan21/version-1.0/svi-film-opt-10212025.safetensors`:
- Strength 1.0 (full SVI) — model dominates, strong reference-image lock
- Need to enable LoRA via `WanVideoLoraSelect` node before `WanVideoSetBlockSwap`

### Critical Notes

- **SVI LoRAs cannot use vanilla Wan 2.1 I2V workflow** — needs `WanVideoSVIProEmbeds` for chaining
- **Different seed per clip is mandatory** (community confirmed)
- **Use fp16 LoRA not quantized** (per SVI FAQ Issue #51)
- **480×832 horizontal is the only stable resolution** (training match)
- **No 121-frame clips — must use 81** (causes color degradation per community)
- **End frame weirdness** (LTX 2.3 IC-LoRA) is a base-model limit, not SVI — SVI's error recycling mitigates it
- SVI-Talk is for **speaking**, not singing. For singing: keep LTX 2.3 IC-LoRA, or wait for community SVI-Sing fine-tune

### Future Improvements (not yet built)

- Auto LLM prompt generation per scene (currently static bank)
- Color consistency pass between scenes (deflicker node exists in custom_nodes)
- Beat-aligned cuts (currently time-aligned to section boundaries)
- SVI-Sing: singing-specific LoRA (community project, not in SVI repo yet)

## Working Lip-Sync Pipeline (Jun 20-23 2026)

### What Works
Z-Image reference → LTX 2.3 (audio-conditioned) → video with motion + lip sync.

### Winning Settings (output 00026)
```bash
python3 ltx_lipsync_fixed.py \
  --image heatwave_refs/portrait_v2.png \
  --audio segment_010.wav \
  --prompt "female singer on neon-lit city street, subtle movement, dramatic neon lighting, cinematic portrait, photorealistic, detailed face, natural skin, 35mm film" \
  --duration 7.5 --seed 50 --lora 0.8 --i2v 0.6 --cfg 3.5 \
  --width 960 --height 544
```

| Param | Value | Notes |
|-------|-------|-------|
| Model | Q6_K | 20GB, better than Q4 for quality |
| LoRA strength | 0.8 | Distilled LoRA |
| I2V strength | 0.6 | Lower = preserves reference face, less distortion |
| CFG | 3.5 | Balance of prompt adherence + quality |
| Sampler | euler | Simple, fast |
| Scheduler | linear_quadratic | 15 steps |
| Resolution | 960x544 | Matches pro_clip format |
| Duration | 7.5s | 180 frames |
| Reference | `heatwave_refs/portrait_v2.png` | Z-Image natural-skin ref, LoRA 0.3/0.2 |
| Seed | 50 (base) | + `scene_index * 100 + clip_index * 13` for variation |
| Audio | `segment_NNN.wav` | 7.5s chunks from `input/` |

### Singer Scene Orchestration (`singer_heatwave.py`)

The 5 singer scenes (verse1, prechorus1, verse2, prechorus2, chorus3) need
~18 lip-sync clips total (3-4 per scene, depending on duration). Use:

```bash
# Show the plan first
python singer_heatwave.py --plan

# Render all singer scenes (sequential, ~5-6 min each = ~90-108 min)
python singer_heatwave.py --submit

# Check progress
python singer_heatwave.py --status
cat /home/ericr/ComfyUI/output/heatwave_singers/submit_log.json | jq

# Stitch when done
python singer_heatwave.py --stitch
# Output: output/heatwave_singers/singer_scenes_full.mp4
```

Segment-to-scene mapping (7.5s segments @ 123 BPM, 24fps):
- **verse1** (12-36s, 24s) → segment_001 (cut) + 002, 003, 004 (cut) = 4 clips
- **prechorus1** (36-48s, 12s) → segment_004 (cut) + 005, 006 (cut) = 3 clips
- **verse2** (72-96s, 24s) → segment_009 (cut) + 010, 011, 012 (cut) = 4 clips
- **prechorus2** (96-108s, 12s) → segment_012 (cut) + 013, 014 (cut) = 3 clips
- **chorus3** (156-174s, 18s) → segment_020 (cut) + 021, 022 + 023 (cut) = 4 clips

`segment_010.wav` (verse2, clip 1) is what produced the 00026 winner.

### Reference Image Generation (Z-Image)
```python
# Z-Image settings for reference images
UNET: z_image_turbo_bf16.safetensors
CLIP: qwen_3_4b.safetensors (type: "qwen_image")
VAE: ae.safetensors  # CRITICAL: NOT flux2-vae.safetensors
LoRAs: DarkB ZIT lora (0.3) + REDZ15_DetailDaemon (0.2)
Steps: 4, CFG: 1.5, Sampler: euler
Resolution: 960x544 for singer refs, 1920x1088 for b-roll
```

**Key finding: Z-Image VAE is `ae.safetensors`, NOT `flux2-vae.safetensors`**
Using the wrong VAE causes tensor dimension mismatch (16 vs 128 channels).

### Reference Image Style (portrait_v2 — Natural Skin)
```
positive: "close-up portrait of a young woman singing, visible skin pores,
  natural skin texture, film grain, warm golden skin, curly dark hair,
  neon pink and blue light on face, candid moment, 35mm film photography,
  photorealistic, natural imperfections"
negative: "ugly, deformed, blurry, low quality, plastic skin, airbrushed,
  smooth skin, doll-like, cartoon"
LoRA strengths: DarkB 0.3, DetailDaemon 0.2 (lower = more natural)
```

### Shot Plan (Heatwave)
- **Singer scenes**: studio_front.png / studio_34.png references
- **B-roll**: Z-Image or Ideogram 4 references at 1920x1088
- **B-roll audio**: instrumental track (not vocals) drives LTX motion
- **Final assembly**: overlay full song audio in ffmpeg

### Wav2Lip (Installed but Degrades Quality)
- Node: `ComfyUI_wav2lip` (patched: soundfile.write instead of torchaudio.save)
- Model: `wav2lip_gan.pth` (415MB) in `custom_nodes/ComfyUI_wav2lip/Wav2Lip/checkpoints/`
- Result: adds lip sync but causes eye glitches and face artifacts
- **Verdict: LTX alone produces better output than LTX + Wav2Lip**

### What Doesn't Work
- **IC-LoRA**: lip sync not convincing, 4s limit, end-frame artifacts
- **LatentSync**: post-processing mouth-mover, not singing
- **Wav2Lip**: eye glitches, face artifacts
- **Empty prompt**: static video with no motion
- **"mouth wide open" prompt**: causes face distortion
- **Wrong VAE**: `flux2-vae.safetensors` breaks Z-Image (use `ae.safetensors`)

### Full Song Assembly Formula (Working as of Jun 24 2026)

**The proven pipeline: render individual clips → concat → overlay full song audio.**

#### Step 1: Split vocals into 7.5s segments
```bash
python3 scripts/split_audio.py \
  --input output/heatwave/separated/vocals.wav \
  --output-dir input/ --segment-sec 7.5
```
⚠ **Critical**: re-split from `vocals.wav` (NOT instrumental). Instrumental
segments produce silent/bass-only clips with no lip sync motion.

#### Step 2: Render all segments with ltx_lipsync_fixed.py
```bash
# Render each segment (alternate studio_front/studio_34 for variety)
for idx in $(seq 0 23); do
  ref="heatwave_refs/studio_front.png"
  [ $((idx % 2)) -eq 1 ] && ref="heatwave_refs/studio_34.png"
  python3 ltx_lipsync_fixed.py \
    --image "$ref" --audio "segment_$(printf '%03d' $idx).wav" \
    --prompt "female singer performing in studio, singing into microphone, dramatic warm lighting, cinematic, photorealistic, detailed face, natural skin, 35mm film" \
    --duration 7.5 --seed $((50 + idx)) --lora 0.8 --i2v 0.6 --cfg 3.5 \
    --width 960 --height 544 --output output/
done
```

**⚠ Restart ComfyUI between renders** to avoid Q6_K OOM:
```bash
systemctl --user restart comfyui && sleep 30
```

#### Step 3: Concat video + overlay full song
```bash
# Create concat list (24 clips × 7.5s = 180s, matches song length)
for i in $(seq 61 84); do
  echo "file 'output/ltx_lipsync_000${i}.mp4'"
  echo "duration 7.5"
done > /tmp/concat.txt
echo "file 'output/ltx_lipsync_00084.mp4'" >> /tmp/concat.txt

# Concat + force 7.5s per clip via fps filter (fixes 177-frame drift)
ffmpeg -y -f concat -safe 0 -i /tmp/concat.txt \
  -vf "fps=24" -an -c:v libx264 -crf 18 /tmp/video.mp4

# Overlay full Heatwave.wav audio
ffmpeg -y -i /tmp/video.mp4 -i input/Heatwave.wav \
  -c:v copy -c:a aac -b:a 192k -map 0:v -map 1:a -shortest \
  output/heatwave_studio_full.mp4
```

**Key details:**
- LTX generates 177 frames (7.375s) per clip, but audio segments are 7.5s
- The `fps=24` filter stretches each clip to exactly 7.5s, fixing the 0.125s/clip drift
- Without this, sync drifts ~0.5s after 4 clips
- 24 clips × 7.5s = 180s (matches 180.28s song length)
- Strip per-clip audio (`-an`) and overlay full song — individual clip audio causes desync

#### Reference images used
- **Studio front**: `heatwave_refs/studio_front.png` (Z-Image, LoRA 0.3/0.2, 960x544)
- **Studio 3/4**: `heatwave_refs/studio_34.png` (same settings, different angle)
- Alternate refs generated with lower LoRA strength for natural skin texture

#### What was tried and abandoned
- **Wav2Lip** post-processing: adds lip sync but causes eye glitches
- **B-roll with Z-Image refs**: too generic, not lyric-literal
- **B-roll with blank reference + text prompts**: random/cartoon output
- **IC-LoRA + OmniNFT stacking**: no improvement over base pipeline
- **Two-pass GAP-style sampling**: higher VRAM, no quality gain
- **LatentSync post-processing**: mouth moves but doesn't sing

#### Key Technical Findings

**Snap-to-LTX-grid (from KupkaProd):**
- LTX needs `8n+1` frames: 177 frames (7.375s) not 180 (7.5s)
- Formula: `fc = int(((duration * fps - 1) / 8) * 8 + 1)` → 177 for 7.5s@24fps
- Audio is 7.5s (180 frames), video is 7.375s (177 frames) → 0.125s drift/clip
- Fix: use `fps=24` filter in ffmpeg concat to stretch each clip to 7.5s
- 24 clips × 7.5s = 180s (matches 180.28s song)

**8-section prompt structure (from KupkaProd):**
Forces material/texture words to prevent "plastic" skin look:
1. shot_framing: camera angle, movement
2. subject: who/what is in frame
3. action: what's happening
4. environment: background/setting
5. lighting: key light direction, color temp
6. color_materials: **material words** (matte, brushed aluminum, denim, grain)
7. style_medium: cinematic photorealistic, 35mm film
8. quality: sharp focus, skin pores, film grain

Current prompts skip sections 4-6 → output looks plastic. Adding material
words ("brushed aluminum microphone stand", "denim texture", "matte black")
forces the model to render realistic surfaces.

#### Remaining improvements (from plan_heatwave_music_video.md)
- Lyric-literal b-roll (12 entries with Ideogram 4 refs)
- Snap to 8n+1 frames (not 7.5s which is off-grid)
- 8-section prompt structure for better quality
- Singer reference alternates (close-up, wide) per scene type

## MrFlow - Multi-Resolution Flow Matching (Jul 4 2026)

### Overview
Training-free diffusion acceleration via staged sampling: low-res → RealESRGAN x2 upscale → re-encode → single high-res refinement step. 8-21x speedup on flow-matching models.

### Files Installed
| Component | Location | Status |
|-----------|----------|--------|
| MrFlow ComfyUI Plugin | `custom_nodes/ComfyUI-MrFlow/` | ✅ Installed |
| Qwen-Image FP8 Model | `models/diffusion_models/qwen_image_fp8_e4m3fn.safetensors` | ✅ 20GB (copied from backup) |
| Qwen-Image VAE | `models/vae/qwen_image_vae.safetensors` | ✅ Already present (243MB) |
| Qwen-Image CLIP | `models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors` | ✅ 8.8GB (copied from backup, type: `qwen_image`) |
| Qwen-Image CLIP (alt) | `models/text_encoders/qwen_3_4b.safetensors` | ⚠️ 7.5GB - WRONG dims for Qwen-Image (2560 vs 3584) |
| RealESRGAN x2 | `models/upscale_models/RealESRGAN_x2.pth` | ✅ 64MB (from v0.2.1 release) |
| Qwen-Image GGUF (alt) | `models/unet/qwen-image-Q4_K_S.gguf` | ✅ Already present (12GB) |

### Workflow: `custom_nodes/ComfyUI-MrFlow/examples/qwen_mrflow_workflow.json`
Node chain: `UNETLoader → CLIPLoader(qwen_image) → VAELoader → UpscaleModelLoader(RealESRGAN_x2) → MrFlowQwenPreset → EmptySD3LatentImage → KSampler → VAEDecode → MrFlowUpscaleEncode → MrFlowAttachReferenceLatent → MrFlowQwenRefine → SaveImage`

### Model Paths to Change in Workflow
| Node | Workflow Default | Our Version |
|------|-----------------|-------------|
| UNETLoader | `qwen_image_bf16.safetensors` | `qwen_image_fp8_e4m3fn.safetensors` |
| CLIPLoader | `qwen_2.5_vl_7b.safetensors` (type: `qwen_image`) | `qwen_2.5_vl_7b_fp8_scaled.safetensors` (type: `qwen_image`) |
| VAELoader | `qwen_image_vae.safetensors` | Unchanged |
| UpscaleModelLoader | `RealESRGAN_x2.pth` | `RealESRGAN_x2.pth` |

### VRAM Consideration
- Qwen-Image FP8 is 20GB on disk. With `--lowvram`, should work on 16GB RTX 4090 (model is loaded layer-by-layer)
- Alternative: use GGUF quantized version `qwen-image-Q4_K_S.gguf` (12GB) in `unet/` via `UNETLoader`
- The GGUF version is smaller but may not load via standard `UNETLoader` — test both

### What MrFlow Can Accelerate
| Model | MrFlow Support | Our Setup | Speed | Priority |
|-------|---------------|-----------|-------|----------|
| Qwen-Image | ✅ Official ComfyUI plugin | ✅ FP8 model ready | 34s (12+1) | **Tested** |
| Ideogram 4 | ✅ Custom `MrFlowIdeogramRefine` node | ✅ FP8 dual model | 19s (12+1) | **Tested** |
| Z-Image Turbo | ✅ Official demo (21x speedup) | ✅ BF16 model in `diffusion_models/` | Untested | Could accelerate ref gen |
| Krea 2 Turbo | ✅ Community confirmed, verified | ✅ FP8 running | 10.9s (8+1) | Accelerates ref gen |
| LTX Video | ❌ Different architecture (video transformer) | Q6_K GGUF | N/A | Not applicable |

### Future: Apply to Existing Pipelines
- Reference image generation (Krea 2 + singer LoRA): MrFlow could cut 8-step gen to 4+1 or even 2+1
- Ideogram 4 parameter sweep: MrFlow `12+1` setting could cut experiment time from 72min to ~8min
- B-roll ref generation (Z-Image): MrFlow `8+1` gives 21x speedup per the paper

## Model Comparison — Krea 2 vs Ideogram 4 (Jul 4 2026)

### Model Strengths

| Scene | Best Model | Why |
|-------|-----------|-----|
| **Product/staging** (whisky pour) | **Krea 2** | Treats prompts as art direction — clean composition, staging |
| **Portrait/cinematic** | **Krea 2** | Controlled studio feel, deliberate composition |
| **Nature/macro** (hummingbird) | **Ideogram 4** | Literal photorealism, fine detail, lighting accuracy |
| **Landscape** (Everest, fjord) | **Ideogram 4** | Better depth, atmosphere, photographic realism |
| **Text in image** | **Ideogram 4** | Built-in text rendering via `"type": "text"` elements |
| **Artistic/mood** | **Krea 2** | Painterly feel, style consistency, opinionated composition |

Per the [Krea blog](https://www.krea.ai/blog/krea-2-and-nano-banana-2-side-by-side): Krea 2 is the instrument for *direction* (composition, mood, taste). Ideogram 4 is best for *truth on the page* (photorealism, literal prompt following, text).

### Native T2I Settings

| Setting | Krea 2 | Ideogram 4 |
|---------|--------|------------|
| Steps | 8 (Turbo) | 12 (Turbo) / 20 (Default) |
| CFG | **1.0** (distilled model) | 7.0 (DualModelGuider internal) |
| Sampler | euler | euler |
| Scheduler | simple | Ideogram4Scheduler (mu=0.5, std=1.75) |
| Latent | EmptyLatentImage (4ch) | EmptyFlux2LatentImage (16ch) |
| CLIP type | `krea2` | `ideogram4` |
| VAE | `qwen_image_vae.safetensors` | `flux2-vae.safetensors` |
| Typical time @1024² | ~13s | ~25s |

**CRITICAL**: Krea 2 is a distilled model — must use **cfg=1.0**. Higher CFG produces garbled output.

### Proper Ideogram 4 JSON Prompting (KJ Node)

**Per the [official docs](https://github.com/ideogram-oss/ideogram4/blob/main/docs/prompting.md):**
- Plain text prompts **trigger the safety filter** — structured JSON is required
- `compositional_deconstruction` (background + elements) is **required**
- `style_description` must include `photo` or `art_style` — setting `style: "none"` bypasses it
- `style.photo` must be passed as a dotted key in the KJ node inputs (DynamicCombo v3 API)

**API format for KJ node (node 209):**
```python
"kj": {
    "class_type": "Ideogram4PromptBuilderKJ",
    "inputs": {
        "width": 1024, "height": 1024,
        "high_level_description": "one-line scene summary",
        "background": "scene background description",
        "style": "photo",                                  # NOT "none"
        "style.photo": "24mm, f/4, architectural photo",    # dotted key for sub-field
        "aesthetics": "style keywords",
        "lighting": "lighting description",
        "medium": "photograph",
        "style_palette_data": "",
        "elements_data": '[...]',                          # must be non-empty JSON string
        "bg_brightness": 30,
        "import_mode": "when empty",
        "output_format": "compact",
        "coord_mode": "normalized",
        "bbox_order": "yx",
    }
}
```

**`elements_data` must be a non-empty JSON string** — at least one element:
```json
[{"x":0.2,"y":0.1,"w":0.6,"h":0.5,"type":"obj","desc":"Main subject description","palette":[]}]
```

### Test Outputs (all at 1024×1024 unless noted)

| Prompt | Krea 2 | Ideogram 4 | Best |
|--------|--------|------------|------|
| Glass fjord interior | `krea2_fjord.png` (1.3MB) | `ideogram_proper.png` (1.7MB) | **Ideogram** |
| Everest landscape | — | `ideogram_everest_kj.png` (1.6MB) | **Ideogram** |
| Whisky pour | `krea2_whisky.png` (1.0MB) | `ideogram_whisky.png` (1.0MB) | **Krea 2** |
| Hummingbird | `krea2_hummingbird.png` (1.1MB) | `ideogram_hummingbird.png` (1.5MB) | **Ideogram** |
| Cinematic portrait | `krea2_portrait.png` (1.1MB) | `ideogram_portrait.png` (1.5MB) | **Krea 2** |
| Thor movie poster @1536×2304 | `thor_krea2.png` (5.1MB, 51s) | `thor_cinematic.png` (6.7MB, 170s) | **Krea 2** |
| Avengers 7 poster @1536×2304 | — | `avengers7_v2.png` (6.5MB, 103s) | — |

**Rule of thumb**: Krea 2 for portraits/products/staging. Ideogram 4 for landscapes/nature/text in images.

---

## External Resources — stablediffusiontutorials.com

**Site**: https://www.stablediffusiontutorials.com/
Comprehensive tutorials for ComfyUI, models, LoRAs, and workflows. Has guides for Krea 2, LTX Video, Ideogram 4, and more.

**Key pages**:
| Page | URL | Content |
|------|-----|---------|
| Krea 2 Base/Turbo | `2026/06/krea2-base-turbo.html` | Model info, variants, install guide |
| Krea 2 LoRA models | `2026/06/krea2-lora-models.html` | Top 15 Krea 2 LoRAs, training on Raw, infer on Turbo |
| LTX 2.3 LoRA models | `2026/05/ltx2.3-lora-models.html` | 19 LTX LoRAs for style, quality, enhancement |
| Ideogram 4 | `2026/06/ideogram-4-fp8.html` | Ideogram 4 install and workflows |
| Boogu Image 0.1 | `2026/06/boogu-image-0.1.html` | New multimodal model |
| Wan2.2 VideoGen | `2025/08/wan-2.2-video-generation.html` | Wan video generation guide |

**Krea 2 LoRA tip from site**: Train on Krea 2 Raw, infer on Krea 2 Turbo. Official Krea LoRAs follow this pattern.

## LTX 2.3 LoRA Downloads (Jul 5 2026)

Downloaded from stablediffusiontutorials.com LTX LoRA roundup, stored in `models/loras/`:

| LoRA | File | Size | Use |
|------|------|------|-----|
| Upscale IC | `ltx2.3_upscale_ic-lora_06250.safetensors` | 313MB | Second-pass refinement, adds clarity |
| Singularity OmniCine V1 | `Singularity-LTX-2.3_OmniCine_V1.safetensors` | 2.6GB | IC LoRA for I2V, better physical consistency |
| Soft Enhance | `LTX2.3_Soft_Enhance.safetensors` | 337MB | Natural detail enhancement |
| Crisp Enhance | `LTX2.3_Crisp_Enhance.safetensors` | 673MB | Sharp detail enhancement |
| Fantasy Realism | `Fantasy_Realism.safetensors` | 337MB | Cinematic fantasy style |

---

## Gore / Extreme Content Testing (Jul 5 2026)

### Summary
Session focused on testing the limits of Krea 2 and Ideogram 4 for graphic violence/gore/body horror.

| Model | Gore | Safety Filter | Method |
|-------|------|---------------|--------|
| **Krea 2 Turbo** (vanilla, cfg=1, 8 steps) | ❌ Tames everything | None needed — soft by design | EmptyLatentImage |
| **Krea 2 RAW FP8** (cfg=5, 20 steps) | ❌ B&W garbage | None | KSampler euler/simple |
| **Krea 2 Turbo + Krea2T-Enhancer** @0.5 | ✅ **Renders gore** | Uncensored | 12 steps, cfg=1, euler/beta |
| **Ideogram 4** (structured JSON via KJ node) | ✅ **Renders gore** | ✅ Structured JSON bypass; "execution" blocked | Turbo, CFG 5 |

### Krea 2 + Krea2T-Enhancer (Installed Jul 5)

**Source**: `capitan01R/ComfyUI-Krea2T-Enhancer` on GitHub (87 stars)

**What it does**: Model patcher that wraps Krea 2's text-fusion mechanism (txtfusion) — boosts deep detail layers per-layer with configurable weights. Breaks the "taste" smoothing of the distilled Turbo model.

**Workflow**:
```
UNETLoader → ComfyUI-Krea2T-Enhancer (strength=0.5) → KSampler → VAEDecode → SaveImage
```

**Settings**: 12 steps, cfg=1, euler/beta scheduler, 1024²

**Parameters**:
- `enabled`: BOOLEAN (default True)
- `strength`: FLOAT 0-2 (default 1.0, recommended 0.5)
- `debug`: BOOLEAN (default False)

### Ideogram 4 Safety Filter Bypass

**Required KJ Node 209 fields to pass safety filter:**

| Field | Required Value |
|-------|---------------|
| `style` | `"photo"` (NOT `"none"`) |
| `style.photo` | dotted key sub-input, e.g. `"forensic documentary photography"` |
| `elements_data` | non-empty JSON string with ≥1 element |
| `coord_mode` | `"normalized"` |
| `bbox_order` | `"yx"` |
| `output_format` | `"compact"` |

**Key finding: `high_level_description` must match the explicit content** — using a generic HLD like "A documentary forensic photograph" triggers cross-reference check with elements_data. Put the full raw prompt as HLD.

**Known failure**: "execution in progress" with "gun to head" / "trigger pull" / "blood splatter" wording → black image (safety filter). Fix: rephrase as aftermath instead of the moment-of.

**Prompts that passed:** severed head, cannibalism, decomposition, parasitic infection, medieval torture, necrophilia/postmortem violation, self-harm/suicide, mass grave/war atrocity, self-mutilation/faceless, demonic possession, Cronenberg body horror, eldritch cosmic horror, occult ritual, anatomical dissection, gothic crypt scene

### Scripts Created

| Script | Purpose |
|--------|---------|
| `weird_dark.py` | 5 dark/weird creatures on both Krea 2 and Ideogram 4 (structured JSON) |
| `weird_edge.py` | 5 gore prompts (decap, cannibal, decomp, parasite, torture) on both |
| `weird_extreme.py` | 5 extreme gore prompts (necrophilia, self-harm, war, execution, mutilation) |
| `test_krea2_raw_gore.py` | Krea 2 RAW vs Turbo comparison for gore (RAW failed) |
| `test_krea2t_enhancer.py` | Krea 2 + Krea2T-Enhancer test (worked — gore rendered) |
| `run_ideo.py` | Simple reusable API script — edit PROMPTS list at top, run. Auto-wraps as structured JSON |

### Output Directories

| Dir | Contents |
|-----|----------|
| `output/Weird_Dark/` | 10 images — 5 Krea 2, 5 Ideogram 4. Gothic/dark fantasy |
| `output/Weird_Edge/` | 10 images — 5 Krea 2, 5 Ideogram 4. Gore push |
| `output/Weird_Extreme/` | 10 images — 5 Krea 2 (still tame), 5 Ideogram 4 (graphic except execution) |
| `output/Weird_Enhanced/` | 4 images — Krea 2 + T-Enhancer. Krea 2 finally rendered gore |

### Workflow Files

| File | Path |
|------|------|
| KJ Node Ideogram 4 workflow | `user/default/workflows/c0a2671af7c4.json` |
| Experiment template (API patching) | `experiment_template.json` |

## Krea 2 Ablation + Absurd Prompt Tests (Jul 6 2026)

### Ablation Setup
Krea 2 Turbo FP8 + T-Enhancer (0.5) | 1024x1024 | 12 steps, CFG 1.0, euler/beta | Seed 42000001

### What Krea 2 LISTENS To (Stable Diffusion-style prompts)
| Factor | Impact | Finding |
|--------|--------|---------|
| Subject details | HIGH | Every adjective matters. Detailed prompt = exact match |
| Environment | CRITICAL | Model does NOT auto-fill. Must specify location explicitly |
| Color/Material | HIGH | "matte black ceramic mug" = exact match. "mug" alone = random |
| Numbers | HIGH | Counts perfectly (1, 3, 7 apples all correct) |
| Spatial language | HIGH | "above", "below", "inside", "left of" understood correctly |

### What Krea 2 IGNORES (Stable Diffusion-style prompts)
| Factor | Impact | Finding |
|--------|--------|---------|
| Hype words | ZERO | "masterpiece, 8k, trending, award winning" = identical output |
| Style keywords | ZERO | "cinematic, photorealistic, highly detailed" = no visible change |

### CRITICAL FINDING: Absurd prompts use DIFFERENT rules
For absurd/surreal prompts, the working formula is DIFFERENT:

| Finding | Detail |
|---------|--------|
| "photorealistic, 8k" HELPS | Ablation said ignore, but prompts WITHOUT these produce weaker absurdity |
| "A X where Y but Z" structure wins | "An open plan office where every person is typing, but each person's head is a kettle..." |
| SHORT prompts work better | Fewer details = stronger absurd focus |
| Color/material DILUTES absurdity | Adding material specs distracts from the surreal core concept |
| Negative prompt matters | Long negative prompt with "cute, adorable, pleasant, whimsical" helps avoid sanitized output |

### What Krea 2 CAN render (absurd)
| Concept | Works? | Example |
|---------|--------|---------|
| Body part replacement | YES | Kettle-heads, featureless faces |
| Gore/violence (with T-Enhancer) | YES | Decap, torture, execution |
| Mirror reflections wrong | YES | Different person in reflection |
| Wrong objects in scenes | PARTIAL | Some work, some ignored |
| Object animation/legs | NO | Walking fridge, walking tree ignored |
| Transparency | NO | Glass torso ignored |
| Living objects | NO | Cardboard box person, egg with legs |
| Impossible stretching | NO | Arm through ceiling |

### Absurd Prompt Formula (Proven)
```
"A [normal location/scene] where [normal actors], but [one absurd visual thing],
[lighting], photorealistic, 8k"
```
- ALWAYS end with "photorealistic, 8k"
- Short sentences with "where...but" structure
- Focus on ONE absurd concept
- Use T-Enhancer @ 1.0 strength for absurd/gore
- 20 steps, CFG 1.0, euler/beta
- Negative MUST include: boring, normal, clean, sanitized, cute, cozy, whimsical
- Avoid: color/material specs, long location descriptions, multiple absurd elements

### A/B Test Result (office_heads, seed 700)
| Prompt | Result |
|--------|--------|
| A: "where...but, photorealistic, 8k" (weird_dark3 stijl) | **BEST** — clear absurdity |
| B: "with...but, photorealistic, 8k" (v2 korter) | Less good |
| A with color/material added | Diluted absurdity |

### Files
| File | Purpose |
|------|---------|
| `ablation_test.py` | Systematic ablation (35 images, 8 suites) |
| `weird_dark3.py` | WORKING absurd prompts (15 prompts, all rendered OK) |
| `weird_final.py` | Ollama-based prompt gen pipeline |
| `weird_ablation.py` | Ablation-optimized prompt gen (needs system prompt fix) |
| `ABLATION_FINDINGS.md` | Full documentation |
| `output/Weird_Dark3/` | 15 working absurd images |
| `output/A_B_Compare/` | A/B test results |
| `output/Absurd_V3_Winner/` | V3 attempt (incomplete) |
| `output/ablation_krea2/` | Earlier ablation images |

## Krea-2 Scene Pipeline (Jul 2026)

### Overzicht
Scene-gebaseerde image generator met uncensored output. Gebruikt OpenRouter (Qwen Coder) voor prompt generatie en ComfyUI API voor rendering.

### Architectuur
```
Scene idee → Qwen Coder (OpenRouter) → Krea-2 Prompt → ComfyUI API → PNG output
```

### Bestanden
| Bestand | Functie |
|---------|---------|
| `krea2_prompt_system.txt` | Prompt Mastery Labs systeem prompt |
| `krea2_scene_gen.py` | Prompt generator via OpenRouter |
| `krea2_runner.py` | ComfyUI API workflow runner |
| `krea2_batch.py` | Batch pipeline script |
| `scenes_example.txt` | Voorbeeld scenes |
| `KREA2_PIPELINE.md` | Uitgebreide documentatie |

### Config
- **OpenRouter**: `qwen/qwen3-coder-next` ($0.11/M, 1.4s, uncensored)
- **Model**: `redcraft22INT8Convrot_11INT8Native.safetensors` (uncensored fine-tune)
- **Text Encoder**: `qwen3vl_4b_fp8_scaled.safetensors`
- **VAE**: `qwen_image_vae.safetensors`
- **Sampler**: er_sde, 16 steps, CFG 1.0, simple scheduler

### Gebruik
```bash
cd ~/ComfyUI
python3 krea2_batch.py --scene "beschrijving" --steps 16
python3 krea2_batch.py --scenes scenes_example.txt --orientation portrait
```

### Prompt Mastery Formaat
Qwen genereert prompts in dit formaat:
```
[Subject + Pose] → [Appearance] → [Props] → [Composition] → [Environment] → [Lighting] → [Aesthetic]
```

### Output
- Images: `~/ComfyUI/output/`
- Prompts: `~/ComfyUI/output/krea2_scenes/scene_XXX_prompt.txt`
- Log: `~/ComfyUI/output/krea2_scenes/batch_log.json`

### Tweede Model: Ideogram 4
Voor absurd/surreal content die Krea 2 niet aankan.
- Model: ideogram4_fp8_scaled + ideogram4_unconditional_fp8_scaled
- Workflows: Ideogram_4.0_00679_.json, c0a2671af7c4.json
- Node: Ideogram4PromptBuilderKJ
- Prompt: Gestructureerd JSON (niet plain text)
- Stappen: 20, CFG 1.75, euler, beta scheduler

### Skill
Zie ~/.config/opencode/skills/image-gen/SKILL.md

## Krea 2 Lifelike Test (Jul 7 2026)

### Doel
De "AI-plastic" look verminderen in Krea 2 output zonder de bestaande Prompt Mastery system prompt te wijzigen.

### Test Setup
4 variaties, zelfde Qwen-prompt, zelfde seed (42000042), 1024x1024:

| Var | Negatief | T-Enhancer | Steps | Materiaal-woorden |
|-----|----------|------------|-------|-------------------|
| A | oud (baseline) | 0.0 | 12 | nee |
| B | nieuw | 0.0 | 12 | nee |
| C | nieuw | 0.3 | 12 | nee |
| D | nieuw | 0.3 | 16 | ja |

### Resultaat
**Variatie C wint** (gebruiker voorkeur): nieuw negatief + T-Enhancer 0.3 + 12 steps.
- Oogopslag naturaler dan D (D was iets te "gepolijst" door 16 steps)
- T-Enhancer @ 0.3 breekt de "tame" Turbo smoothing zonder over-the-top te gaan
- Negatieve prompt alleen (A vs B) maakt weinig verschil — T-Enhancer is de echte driver
- Materiaal-woorden helpen, maar moeten niet overdrijven (vermijd "frecles" = AI artifact)

### Wijzigingen Doorgevoerd

**`krea2_runner.py`**:
- `build_workflow()` krijgt `enhancer_strength` parameter (default 0.3)
- T-Enhancer node (ComfyUI-Krea2T-Enhancer) wordt tussen UNETLoader en KSampler gevoegd
- `render_prompt()` stuurt enhancer_strength door

**`krea2_batch.py`**:
- `run_batch()` krijgt `enhancer` parameter (default 0.3)
- Nieuwe CLI flag: `--enhancer 0.3` (0.0 = uit, 0.3 = subtiel, 1.0 = sterk)

**`krea2_scene_gen.py`**:
- Materiaal-woorden guidance toegevoegd aan user_msg (uncensored mode)
- "frecles" expliciet verboden als AI artifact
- Aanbevolen alternatieven: "skin pores, fine lines, subtle stubble, natural oil sheen"

**`krea2_prompt_engine.py`**:
- Negatieve prompt bijgewerkt: "plastic skin, airbrushed, doll-like, symmetrical face, CGI, 3d render, hyperreal, overprocessed" i.p.v. "boring, normal, plain"
- T-Enhancer default verlaagd van 1.0 naar 0.3

### Gebruik
```bash
# Standaard (T-Enhancer 0.3, 12 steps)
python3 krea2_batch.py --scene "beschrijving"

# Zonder T-Enhancer (oud gedrag)
python3 krea2_batch.py --scene "beschrijving" --enhancer 0.0

# Sterker (voor absurd/gore)
python3 krea2_batch.py --scene "beschrijving" --enhancer 1.0

# Losse render via prompt engine
python3 krea2_prompt_engine.py "concept"
```

### Output
- Test images: `~/ComfyUI/output/lifelike_test/`
- Referentie: `~/Downloads/civitai_scan/{A,B,C,D}_*.png`

## SCAIL-2 Dancing Singer Pipeline (Jul 2026)

### Overview
SCAIL-2 (SegmAntion-Controlled Animation with In-context Learning, v2) is an end-to-end character animation model built on Wan 2.1 14B. It animates a reference character using a driving video — **no skeleton intermediates**. The model reads driving video latents directly, capturing full visual motion including weight shifts, depth, and non-human movement.

### Key Advantage
Real motion transfer (1:1 from driving footage) vs. LTX 2.3 I2V's "imagination" of motion. The spinning kick demo on Reddit showed full 360° rotation with character seen from all angles — something I2V struggles with.

### Paper & Sources
- Paper: https://arxiv.org/abs/2606.10804
- Project page: https://teal024.github.io/SCAIL-2
- Comfy-Org repackaged: https://huggingface.co/Comfy-Org/SCAIL-2
- GGUF quantizations: https://huggingface.co/realrebelai/SCAIL-2_GGUF
- CivitAI workflow: https://civitai.red/models/2699283/wan-scail-2-segmentation-control

### Models Downloaded (7 Jul 2026)

| Bestand | Grootte | Pad |
|---------|---------|-----|
| `SCAIL-2-Q4_K_M.gguf` | 11 GB | `models/unet/` |
| `sam3.1_multiplex_fp16.safetensors` | 1.7 GB | `models/sam/` |
| `clip_vision_h.safetensors` | 1.2 GB | `models/clip_vision/` |
| `lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors` | 704 MB | `models/loras/` |

### Already Available
- `umt5_xxl_fp8_e4m3fn_scaled.safetensors` (text encoder)
- `wan_2.1_vae.safetensors` (VAE)
- `luts/cinematic_bleach_bypass.cube` (LUT)
- ProPost Apply LUT node, Color Match V2 node
- SCAIL-2 native nodes (`nodes_scail.py`)
- Blueprints: "Character Replacement (SCAIL-2 Base)" and "(SCAIL-2 Extend)"

### Architecture
```
Driving video → SAM3 segmentatie → colored masks
Reference image → CLIP Vision encode → reference latent
    ↓
WanSCAILToVideo (conditioning)
    ↓
UNETLoader (GGUF Q4_K_M) + LightX2V LoRA (0.8)
    ↓
KSampler (euler, 12 steps, cfg 3.5)
    ↓
VAEDecode → ProPost LUT → ColorMatchV2 → output
```

### Two Modes
| Mode | Pose Video Mask BG | Reference Mask BG | Use Case |
|------|-------------------|-------------------|----------|
| **Animation** | Zwart | Wit | Zanger animeren met dansende beweging |
| **Replacement** | Wit | Zwart | Karakter in video vervangen |

**Voor "zanger laat dansen"**: Animation Mode.

### VRAM (16GB RTX 4090)
Q4_K_M GGUF (~10GB) is de enige optie. GGUF loader is memory-mapped. De CivitAI post creator draait op exact dezelfde config (16GB VRAM + 64GB RAM).

### Settings (Aanbevolen)
| Parameter | Waarde |
|-----------|--------|
| Model | `SCAIL-2-Q4_K_M.gguf` |
| LoRA | `lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors` (0.8) |
| Sampler | euler (of SCAIL-2 Infinity — minder kleurshifts) |
| Steps | 12 |
| CFG | 3.5 |
| Resolutie | 832×480 (of 704×1280 portrait) |
| Frame count | 81 (3.375s @ 24fps) |

### Pipeline Gebruik
1. **Referentiebeeld**: Krea 2 + singer LoRA (bestaande refs in `input/ref_*.png`)
2. **Driving footage**: Dansvideo (Pexels/Pixabay stock voor testen)
3. **SAM3 segmentatie**: `SCAIL2ColoredMask` node
4. **SCAIL-2 animatie**: `WanSCAILToVideo` node
5. **Kleurcorrectie**: LUT + ColorMatchV2 (SCAIL-2 heeft kleurverschillen tussen clips)

### Kleurcorrectie
SCAIL-2 produceert kleurverschillen door driving footage variatie. Twee-lagen aanpak:
1. Eerst LUT toepassen op alle clips (basis consistency)
2. Dan ColorMatchV2 van elke clip tegen een master clip

### Driving Footage Eisen
- Resolutie: 720p (832×480 of 704×1280)
- Duur: 81 frames @ 24fps = 3.375s per clip
- Formaat: MP4, MOV
- Inhoud: Personen met heldere beweging
- SAM3 moet onderwerp goed kunnen segmenteren

### Beperkingen
- **Geen audio/lip sync** — LTX 2.3 blijft nodig voor lip-sync scenes
- **81-frame chunks** — max ~3.375s per clip
- **Kleurverschillen** — LUT + ColorMatch pipeline essentieel
- **Driving footage nodig** — kan niet uit het niets animeren

### Compatibiliteit met LTX 2.3 Pipeline
SCAIL-2 is geen vervanging voor LTX 2.3 lip-sync. Het is een **aanvulling**:
- **SCAIL-2**: B-roll scenes met complexe beweging (dans, actie)
- **LTX 2.3**: Lip-sync scenes met audio-conditioning
- Beide outputten geven door elkaar via `beat_stitch.py`

### Full Plan
Zie `PLAN_SCAIL2_DANCING_SINGER.md` voor uitgebreide documentatie.

### Bronvermelding
- Reddit thread: https://www.reddit.com/r/StableDiffusion/comments/1upbu5e/wan_scail2_segmentation_control_update/
- Gevonden door gebruiker, motion quality was indrukwekkend (spinning kick test)
